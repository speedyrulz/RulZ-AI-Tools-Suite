"""Muon (MomentUm Orthogonalized by Newton-Schulz) for LoRA, LoKr, OFT and fine-tune weights.

The update is PyTorch's `torch.optim.Muon` (torch/optim/_muon.py, BSD-3-Clause, Copyright (c) 2016- Facebook, Inc.
and the PyTorch contributors; Keller Jordan's Newton-Schulz constants): momentum, a Nesterov look-ahead, five
Newton-Schulz iterations in bf16 that replace the update matrix by an orthogonal one (every direction gets about the
same step), decoupled weight decay, and Moonshot's `match_rms_adamw` step size (0.2 * sqrt(max(rows, cols)) x lr),
so the update's RMS is about AdamW's and AdamW's learning rates carry over. Fizgig trains every picture model's
LoRA presets on it since 6 Nov 2026 with those defaults (momentum 0.95, Nesterov, 5 steps, weight decay 0.01).

Departures from torch.optim.Muon, none of which changes a 2-D matrix's update:

1. Matrices of the same shape are orthogonalized together, in batched multiplies (a LoRA is hundreds of small
   matrices, and one Newton-Schulz run each made the optimizer step the slow part; Fizgig batched it the same way and
   measured the step 12x faster). A batch holds at most `bucket_elems` numbers, so a full fine-tune's large
   matrices go a few at a time instead of all at once.
2. Other shapes are accepted instead of refused: a 3-D tensor (OFT's rotation blocks, [blocks, b, b]) is a stack of
   matrices, each orthogonalized on its own with its own step size; a 1-D or 0-D tensor (DoRA's magnitude) takes
   its momentum update scaled to an RMS of 0.2, the RMS `match_rms_adamw` gives a matrix, so it moves about as far as
   under AdamW; anything above 3-D is one matrix of its first dimension by the rest (Keller Jordan's convolution
   rule).

The state per parameter is torch's: one `momentum_buffer` the parameter's shape and dtype (fp32 for RulzTrainer's
adapters), so a state saved by torch.optim.Muon on 2-D parameters loads here.
"""
from __future__ import annotations

import math

import torch

EPS = 1e-7
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
ADJUST_LR = ("original", "match_rms_adamw")
VECTOR_RMS = 0.2  # a 1-D tensor's update RMS: the RMS match_rms_adamw gives an orthogonalized matrix
BUCKET_ELEMS = 1 << 26  # numbers per batched Newton-Schulz run (64M: 128 MB in bf16, plus its products)


def zeropower_batched(x: torch.Tensor, coefficients=NS_COEFFICIENTS, steps: int = NS_STEPS,
                      eps: float = EPS) -> torch.Tensor:
    """torch.optim.Muon's Newton-Schulz orthogonalization for a stack of same-shaped matrices [k, m, n]: each matrix
    is divided by its own Frobenius norm and iterated as torch does it (in bf16, on the wide side), in batched
    multiplies. Returns bf16 [k, m, n]."""
    if steps >= 100:
        raise ValueError("Newton-Schulz steps must be below 100")
    a, b, c = coefficients
    y = x.bfloat16()
    tall = y.size(-2) > y.size(-1)
    if tall:
        y = y.transpose(-2, -1)
    y = y / y.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = y @ y.transpose(-2, -1)
        y = torch.baddbmm(y, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), y, beta=a)
    return y.transpose(-2, -1) if tall else y


def adjusted_lr(lr: float, adjust_lr: str, rows: int, cols: int) -> float:
    """The step size of a rows x cols matrix (torch.optim.Muon's _adjust_lr)."""
    if adjust_lr == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(rows, cols))
    return lr * math.sqrt(max(1.0, rows / cols))


def matrix_view(p: torch.Tensor) -> tuple[int, int, int] | None:
    """(matrices, rows, cols) a parameter is orthogonalized as, or None for a vector (or a scalar)."""
    if p.ndim < 2:
        return None
    if p.ndim == 2:
        return 1, p.shape[0], p.shape[1]
    if p.ndim == 3:
        return p.shape[0], p.shape[1], p.shape[2]
    return 1, p.shape[0], p.numel() // p.shape[0]


class Muon(torch.optim.Optimizer):
    """torch.optim.Muon's update for any parameter list (see the module docstring), batched by shape."""

    def __init__(self, params, lr: float = 1e-4, weight_decay: float = 0.01, momentum: float = 0.95,
                 nesterov: bool = True, ns_coefficients: tuple[float, float, float] = NS_COEFFICIENTS,
                 ns_steps: int = NS_STEPS, eps: float = EPS, adjust_lr: str = "match_rms_adamw",
                 bucket_elems: int = BUCKET_ELEMS):
        if lr < 0 or weight_decay < 0 or not 0 <= momentum < 1:
            raise ValueError(f"Muon: lr and weight_decay must be 0 or more and momentum in [0, 1), got lr {lr}, "
                             f"weight_decay {weight_decay}, momentum {momentum}")
        if adjust_lr not in ADJUST_LR:
            raise ValueError(f"Muon: adjust_lr must be one of {ADJUST_LR}, got {adjust_lr!r}")
        if not 0 < ns_steps < 100:
            raise ValueError(f"Muon: ns_steps must be 1 to 99, got {ns_steps}")
        self.bucket_elems = int(bucket_elems)
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_coefficients=tuple(ns_coefficients), ns_steps=ns_steps, eps=eps,
                                      adjust_lr=adjust_lr))
        for group in self.param_groups:
            for p in group["params"]:
                if torch.is_complex(p):
                    raise ValueError("Muon does not take complex parameters")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            ps = [p for p in group["params"] if p.grad is not None]
            if not ps:
                continue
            if any(p.grad.is_sparse for p in ps):
                raise RuntimeError("Muon does not support sparse gradients")
            lr, wd, mom = float(group["lr"]), float(group["weight_decay"]), float(group["momentum"])
            grads = [p.grad for p in ps]
            bufs = []
            for p, g in zip(ps, grads):
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                bufs.append(st["momentum_buffer"])
            torch._foreach_lerp_(bufs, grads, 1.0 - mom)
            nesterov = bool(group["nesterov"])

            def update(i: int) -> torch.Tensor:  # made per matrix or chunk: a fine-tune never holds a second gradient
                return grads[i].lerp(bufs[i], mom) if nesterov else bufs[i]

            if wd:
                torch._foreach_mul_(ps, 1.0 - lr * wd)
            by_shape: dict[tuple, list[int]] = {}
            for i, p in enumerate(ps):
                view = matrix_view(p)
                if view is None:
                    u = update(i).float()
                    rms = u.pow(2).mean().sqrt().clamp(min=float(group["eps"]))
                    p.add_((u * (VECTOR_RMS / rms)).to(p.dtype), alpha=-lr)
                else:
                    by_shape.setdefault((view[1], view[2], p.device, p.dtype), []).append(i)
            for (rows, cols, _device, dtype), idx in by_shape.items():
                step = adjusted_lr(lr, group["adjust_lr"], rows, cols)
                per_batch = max(1, self.bucket_elems // (rows * cols))
                chunk: list[int] = []
                held = 0
                for i in idx + [None]:
                    k = matrix_view(ps[i])[0] if i is not None else 0
                    if chunk and (i is None or held + k > per_batch):
                        stack = torch.cat([update(j).reshape(-1, rows, cols) for j in chunk])
                        ortho = zeropower_batched(stack, group["ns_coefficients"], int(group["ns_steps"]),
                                                  float(group["eps"])).to(dtype)
                        parts = ortho.split([matrix_view(ps[j])[0] for j in chunk])
                        torch._foreach_add_([ps[j] for j in chunk],
                                            [o.reshape(ps[j].shape) for o, j in zip(parts, chunk)], alpha=-step)
                        chunk, held = [], 0
                    if i is not None:
                        chunk.append(i)
                        held += k
        return loss
