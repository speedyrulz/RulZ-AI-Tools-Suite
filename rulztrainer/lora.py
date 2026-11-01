"""Minimal LoRA for Linear layers with ComfyUI- or diffusers-compatible export, plus "extra" LoRAs (e.g. turbo /
step-distill adapters) that are applied only while sampling, and a training adapter: a frozen LoRA applied only while
training (the extras off), never saved (Fizgig's Qwen-Image 2.1 adapter keeps that model's LoRAs from collapsing).

Export key formats (Arch.lora_format):

  comfy  accepted by ComfyUI's LoRA loader for any model whose state-dict keys are "diffusion_model.<module>.weight"
    diffusion_model.<module>.lora_down.weight   [rank, in]
    diffusion_model.<module>.lora_up.weight     [out, rank]
    diffusion_model.<module>.alpha              scalar
  peft   diffusers / PEFT (pipe.load_lora_weights, transformer.load_lora_adapter); FIBO 2.0 uses it
    transformer.<module>.lora_A.weight          [rank, in]
    transformer.<module>.lora_B.weight          [out, rank]
    + "lora_adapter_metadata" (the PEFT LoraConfig, prefixed "transformer.") in the file metadata
  comfy_ab  the alpha-less layout ai-toolkit and diffusion-pipe write for MiniMax H3 (and musubi-tuner, ComfyUI and
         diffusers' MiniMaxH3LoraLoaderMixin read): no alpha, so alpha = rank; alpha / rank is folded into lora_B
    diffusion_model.<module>.lora_A.weight      [rank, in]
    diffusion_model.<module>.lora_B.weight      [out, rank]

  folded ComfyUI's and diffusers' loaders at once, for a family whose module names are the ones ComfyUI keeps under
         "diffusion_model." and diffusers converts (FLUX 3 Klein: BFL's names; diffusers refuses .alpha keys there)
    diffusion_model.<module>.lora_A.weight      [rank, in]
    diffusion_model.<module>.lora_B.weight      [out, rank], already times alpha / rank: no .alpha key
    + "rulztrainer_lora_alpha" (JSON {module: alpha}) in the metadata, so a resume continues at the trained alpha

Effective delta = (alpha / rank) * up @ down in comfy and peft; lora_B @ lora_A (scale 1) in folded and comfy_ab.

LoKr (network_type "lokr"): the delta is a Kronecker product kron(w1, w2) with w1 [a, b] small (a, b the largest
divisors of out / in that are <= lokr_factor, the LyCORIS convention) and w2 [c, d] = [out / a, in / b] full-matrix,
applied without materializing the product. Keys <stem>.lokr_w1, .lokr_w2, .alpha (1: no scaling, as ComfyUI reads
full-matrix factors), with the comfy prefix or the peft one ("transformer.", which ComfyUI maps for Krea 2);
diffusers has no LoKr loader. w1 starts kaiming-uniform and w2 at zero, so the delta is 0 at step 0 and both get
gradients from step 1. Fizgig's measured default is factor 8 (its highest likeness; ~1/64 of the weight size).

OFT (network_type "oft", LyCORIS diag-OFT as ComfyUI's weight adapter reads it): instead of adding to a layer, it
rotates the layer's output channels in blocks: oft_blocks [n, b, b] with n * b = out (b the largest divisor of out
<= oft_block_size), Q = blocks - blocks^T (skew-symmetric), R = (I + Q)(I - Q)^-1 (Cayley, orthogonal) and the
merged weight R^T W per block; the bias is not rotated. oft_constraint > 0 (COFT) caps the Frobenius norm of Q and
is saved as the .alpha key, which ComfyUI reads as that cap. Blocks start at zero (R = I). Applied to the frozen
layer's output, so it works on quantized weights as they are.

DoRA (dora: true, on LoRA or LoKr; weight-decomposed): the merged weight is W' = V * diag(m / ||V||_col) with
V = W + delta and the norm per input column, i.e. the layer sees its input scaled by g = m / ||V||_col. ComfyUI's
weight_decompose normalizes by the merged weight only on this input axis (on the output axis it divides by the
original W's norm), so the input axis is the one a file must use to load the same: <stem>.dora_scale [1, in]. m starts
at ||W||_col (g = 1 at step 0). The norm is recomputed every forward from the dequantized weight and, as the DoRA
paper suggests, kept out of the gradient. diffusers / PEFT cannot read this axis: DoRA files load in ComfyUI and here.
"""
from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

log = logging.getLogger("rulztrainer")

COMFY_PREFIX = "diffusion_model."
PEFT_PREFIX = "transformer."
LORA_FORMATS = ("comfy", "peft", "folded", "comfy_ab")
FOLDED_ALPHA_KEY = "rulztrainer_lora_alpha"  # folded files: {module: alpha} the lora_B weights were multiplied by
# peft 0.21 LoraConfig fields: diffusers rebuilds the adapter config from the file metadata with LoraConfig(**these)
PEFT_CONFIG_TEMPLATE = {
    "alora_invocation_tokens": None,
    "alpha_pattern": {},
    "arrow_config": None,
    "auto_mapping": None,
    "base_model_name_or_path": None,
    "bias": "none",
    "corda_config": None,
    "ensure_weight_tying": False,
    "eva_config": None,
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": False,
    "init_lora_weights": True,
    "kasa_config": None,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": 16,
    "lora_bias": False,
    "lora_dropout": 0.0,
    "lora_ga_config": None,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "monteclora_config": None,
    "peft_type": "LORA",
    "peft_version": "0.21.2",
    "qalora_group_size": 16,
    "r": 16,
    "rank_pattern": {},
    "revision": None,
    "target_modules": [],
    "target_parameters": None,
    "task_type": None,
    "trainable_token_indices": None,
    "use_bdlora": None,
    "use_dora": False,
    "use_qalora": False,
    "use_rslora": False,
    "velora_config": None
}
SAVE_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


KINDS = ("lora", "lokr", "oft")
PARTS = {"lora": ("lora_down", "lora_up"), "lokr": ("lokr_w1", "lokr_w2"), "oft": ("oft_blocks",)}
DORA_KINDS = ("lora", "lokr")  # DoRA decomposes an added delta; OFT rotates instead
DORA_EPS = torch.finfo(torch.float32).eps  # what ComfyUI adds to the norm (it patches in fp32)


def factorization(n: int, factor: int) -> tuple[int, int]:
    """(small, large) with small * large == n and small the largest divisor of n that is <= factor (LyCORIS)."""
    factor = max(1, int(factor))
    for cand in range(min(factor, n), 0, -1):
        if n % cand == 0:
            return cand, n // cand
    return 1, n


_FROZEN = {"depth": 0, "round": 0}  # frozen_adapters(): nesting depth and render round


@contextmanager
def frozen_adapters():
    """While samples and probes render the adapters do not change, so DoRA computes each layer's norm once per round
    instead of on every denoising step (training forwards, with gradients on, always recompute it)."""
    _FROZEN["round"] += 1
    _FROZEN["depth"] += 1
    try:
        yield
    finally:
        _FROZEN["depth"] -= 1


def oft_shape(out_features: int, block: int) -> tuple[int, int]:
    """(blocks, block size) of an OFT layer: the block size the largest divisor of out_features <= block."""
    size, n = factorization(out_features, block)
    return n, size


def oft_rotation(blocks: torch.Tensor, constraint: float = 0.0) -> torch.Tensor:
    """The orthogonal blocks R [n, b, b] from oft_blocks, as ComfyUI computes them: Q = blocks - blocks^T, its
    Frobenius norm capped at `constraint` when that is > 0, R = (I + Q)(I - Q)^-1 (in fp32)."""
    q = blocks.float() - blocks.float().transpose(1, 2)
    if constraint > 0:
        norm = q.norm() + 1e-8
        q = torch.where(norm > constraint, q * (constraint / norm), q)
    eye = torch.eye(q.shape[-1], device=q.device, dtype=q.dtype)
    return (eye + q) @ torch.linalg.inv(eye - q)


def dense_weight(base: nn.Module) -> torch.Tensor:
    """A frozen layer's weight [out, in] as a plain tensor (a quantized one dequantized), for DoRA's norms."""
    if hasattr(base, "dequantized"):
        return base.dequantized(torch.float32)
    return base.weight.detach().float()


def adapter_label(kind: str, rank: int = 16, alpha: float = 16.0, factor: int = 8, block: int = 32,
                  dora: bool = False, constraint: float = 0.0) -> str:
    """One adapter setup in words, for logs and the review: "LoRA rank 16, alpha 16 + DoRA", "OFT block 32"."""
    if kind == "oft":
        return f"OFT block {block}" + (f", constraint {constraint:g}" if constraint > 0 else "")
    text = f"LoKr factor {factor}" if kind == "lokr" else f"LoRA rank {rank}, alpha {alpha:g}"
    return text + (" + DoRA" if dora else "")


def lokr_update(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """kron(w1 [a, b], w2 [c, d]) applied to x [..., b * d] -> [..., a * c] without materializing the product:
    kron(w1, w2)[i * c + k, j * d + l] = w1[i, j] * w2[k, l] (torch.kron's layout, which ComfyUI reads)."""
    a, b = w1.shape
    c, d = w2.shape
    t = x.reshape(*x.shape[:-1], b, d) @ w2.t()  # [..., b, c]
    out = torch.einsum("...jk,ij->...ik", t, w1)  # [..., a, c]
    return out.reshape(*x.shape[:-1], a * c)


def _apply_extras(x, out, extras, extra_biases):
    for e in extras:
        if len(e) == 3:  # (down, up, scale): a LoRA
            kind, first, second, scale = "lora", *e
        else:
            kind, first, second, scale = e
        if kind == "lora":
            h = nn.functional.linear(x, first.to(x.device, x.dtype))
            out = out + nn.functional.linear(h, second.to(x.device, x.dtype)) * scale
        else:
            out = out + lokr_update(x, first.to(x.device, x.dtype), second.to(x.device, x.dtype)) * scale
    for bias, scale in extra_biases:
        out = out + bias.to(out.device, out.dtype) * scale
    return out


class LoRALinear(nn.Module):
    """Wraps a Linear (plain or QuantLinear) with a trainable adapter in fp32: a LoRA pair (kind "lora"), a LoKr
    Kronecker pair ("lokr") or OFT rotation blocks ("oft"), with DoRA's magnitude (dora_scale) on LoRA and LoKr when
    `dora` is on; plus frozen extras (sampling-only LoRA / LoKr files). The inactive kinds' tensors are empty; a
    configuration race switches kinds on the same wrapper (set_kind)."""

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float = 0.0, trainable: bool = True,
                 kind: str = "lora", factor: int = 8, block: int = 32, constraint: float = 0.0, dora: bool = False):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"network_type must be one of {KINDS}, got {kind!r}")
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.compute_dtype = getattr(base, "compute_dtype", None) or base.weight.dtype
        self.rank = rank
        self.alpha = alpha
        self.factor = int(factor)
        self.trainable = trainable
        self.lora_down = nn.Linear(self.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, self.out_features, bias=False)
        self.lora_down.float()
        self.lora_up.float()
        a, c = factorization(self.out_features, self.factor)  # out = a * c, in = b * d
        b, d = factorization(self.in_features, self.factor)
        self.lokr_w1 = nn.Parameter(torch.empty(a, b))
        self.lokr_w2 = nn.Parameter(torch.empty(c, d))
        self.block, self.constraint = int(block), float(constraint)
        self.dora = bool(dora) and kind in DORA_KINDS
        self.oft_blocks = nn.Parameter(torch.empty(*self.adapter_shapes("oft")["oft_blocks"]))
        self.dora_scale = nn.Parameter(torch.empty(self.in_features if self.dora else 0))
        self.w_col_norm: torch.Tensor | None = None  # ||W|| per input column (CPU fp32), DoRA's starting magnitude
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.kind = kind
        self.init_adapter()
        self.scale = self.alpha / self.rank if kind == "lora" else 1.0
        self.set_kind(kind)
        for p in self.adapter_params().values():
            p.requires_grad_(trainable)
        self.multiplier = 1.0
        # extras: ("lora", down [r,in], up [out,r], scale) or ("lokr", w1, w2, scale) (3-tuples: a LoRA), applied
        # only when extras_enabled; extra_biases: (bias offset [out], scale), a file's diff_b entries, likewise
        self.extras: list[tuple] = []
        self.extra_biases: list[tuple[torch.Tensor, float]] = []
        self.extras_enabled = False
        # while the extras are on: their copies on the frozen layer's device in the compute dtype (stage_extras),
        # so a render does not upload and cast them on every forward; the CPU originals above stay as they are
        self.staged_extras: list[tuple] | None = None
        self.staged_biases: list[tuple[torch.Tensor, float]] | None = None
        # the training adapter (attach_training_adapter): down [r, in] and up [out, r] (its scale folded in) in the
        # compute dtype, applied while the extras are off; buffers, so moves, block swap and model splits carry it
        self.register_buffer("train_down", None, persistent=False)
        self.register_buffer("train_up", None, persistent=False)

    # ---- the adapter API (both kinds)
    def adapter_params(self) -> dict[str, nn.Parameter]:
        """The active kind's tensors by part name: {"lora_down", "lora_up"}, {"lokr_w1", "lokr_w2"} or
        {"oft_blocks"}, plus "dora_scale" with DoRA."""
        if self.kind == "lora":
            out = {"lora_down": self.lora_down.weight, "lora_up": self.lora_up.weight}
        elif self.kind == "lokr":
            out = {"lokr_w1": self.lokr_w1, "lokr_w2": self.lokr_w2}
        else:
            return {"oft_blocks": self.oft_blocks}
        if self.dora:
            out["dora_scale"] = self.dora_scale
        return out

    def adapter_shapes(self, kind: str, rank: int | None = None, factor: int | None = None,
                       block: int | None = None, dora: bool | None = None) -> dict[str, tuple]:
        """The tensor shapes of a kind on this layer (rank / factor / block / dora default to the wrapper's)."""
        dora = (self.dora if dora is None else bool(dora)) and kind in DORA_KINDS
        if kind == "lora":
            r = int(rank if rank is not None else self.rank)
            out = {"lora_down": (r, self.in_features), "lora_up": (self.out_features, r)}
        elif kind == "lokr":
            f = int(factor if factor is not None else self.factor)
            a, c = factorization(self.out_features, f)
            b, d = factorization(self.in_features, f)
            out = {"lokr_w1": (a, b), "lokr_w2": (c, d)}
        else:
            n, size = oft_shape(self.out_features, int(block if block is not None else self.block))
            return {"oft_blocks": (n, size, size)}
        if dora:
            out["dora_scale"] = (self.in_features,)
        return out

    @torch.no_grad()
    def weight_col_norm(self) -> torch.Tensor:
        """||W|| per input column of the frozen weight (CPU fp32, computed once): DoRA's starting magnitude."""
        if self.w_col_norm is None:
            self.w_col_norm = dense_weight(self.base).norm(dim=0).float().cpu()
        return self.w_col_norm

    def init_parts(self, kind: str, rank: int | None = None, factor: int | None = None, block: int | None = None,
                   dora: bool | None = None, generator=None) -> dict[str, torch.Tensor]:
        """Fresh CPU fp32 weights of a kind on this layer: a LoRA's down and a LoKr's w1 kaiming-uniform, its other
        factor zero (delta 0); OFT blocks zero (R = I); DoRA's magnitude ||W|| per input column (scale 1)."""
        out = {}
        for part, shape in self.adapter_shapes(kind, rank, factor, block, dora).items():
            if part in ("lora_down", "lokr_w1"):
                w = torch.empty(shape)
                nn.init.kaiming_uniform_(w, a=math.sqrt(5), generator=generator)
            elif part == "dora_scale":
                w = self.weight_col_norm().clone()
            else:
                w = torch.zeros(shape)
            out[part] = w
        return out

    @torch.no_grad()
    def init_adapter(self, generator=None) -> None:
        """Fresh weights for the active kind (init_parts)."""
        params = self.adapter_params()
        for part, w in self.init_parts(self.kind, generator=generator).items():
            params[part].data = w.to(params[part].device)

    @torch.no_grad()
    def set_kind(self, kind: str, dora: bool | None = None, block: int | None = None) -> None:
        """Make `kind` (with or without DoRA, with this OFT block size) the active adapter; the inactive tensors are
        freed (empty). The active ones keep their data when their shape still fits, else they are fresh (init)."""
        if kind not in KINDS:
            raise ValueError(f"network_type must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        if block is not None:
            self.block = int(block)
        self.dora = (self.dora if dora is None else bool(dora)) and kind in DORA_KINDS
        dev = self.lora_down.weight.device
        empty = {"lora_down": (0, self.in_features), "lora_up": (self.out_features, 0), "lokr_w1": (0, 0),
                 "lokr_w2": (0, 0), "oft_blocks": (0, 0, 0), "dora_scale": (0,)}
        tensors = {"lora_down": self.lora_down.weight, "lora_up": self.lora_up.weight, "lokr_w1": self.lokr_w1,
                   "lokr_w2": self.lokr_w2, "oft_blocks": self.oft_blocks, "dora_scale": self.dora_scale}
        want = self.adapter_shapes(kind)
        fresh = self.init_parts(kind) if any(tuple(tensors[p].shape) != s for p, s in want.items()) else {}
        for part, t in tensors.items():
            if part not in want:
                t.data = torch.empty(empty[part], device=dev)
            elif tuple(t.shape) != want[part]:
                t.data = fresh[part].to(dev)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        """The added weight [out, in] of a LoRA / LoKr in fp32 (its scale applied), as ComfyUI merges it."""
        if self.kind == "lora":
            return (self.lora_up.weight.float() @ self.lora_down.weight.float()) * self.scale
        return torch.kron(self.lokr_w1.float(), self.lokr_w2.float()) * self.scale

    def dora_input_scale(self) -> torch.Tensor:
        """DoRA's per-input-column scale m / ||W + delta||_col [in]; the norm is computed without gradient (the
        DoRA paper's memory saving), the magnitude m trains. Inside frozen_adapters() without gradients the norm is
        kept for the round (keyed on the weights' storage and version as well)."""
        key = None
        if _FROZEN["depth"] and not torch.is_grad_enabled():
            key = (_FROZEN["round"], *((p.data_ptr(), p._version) for p in self.adapter_params().values()))
            cached = getattr(self, "_dora_norm", None)
            if cached is not None and cached[0] == key:
                return self.dora_scale / cached[1]
        norm = self.merged_col_norm() + DORA_EPS
        self._dora_norm = (key, norm) if key is not None else None
        return self.dora_scale / norm

    @torch.no_grad()
    def merged_col_norm(self) -> torch.Tensor:
        """||W + delta|| per input column [in] (fp32) without forming delta or W + delta: per column
        ||W||^2 + 2 <W, delta> + ||delta||^2, the last two from the delta's factors (a LoRA's up^T W [rank, in]; a
        LoKr's W read as [a, c, b, d] against w1 and w2). One dense fp32 weight per call instead of three."""
        w = dense_weight(self.base).to(self.dora_scale.device)
        w_sq = w.norm(dim=0).square()
        s = self.scale
        if self.kind == "lora":
            up, down = self.lora_up.weight.float(), self.lora_down.weight.float()
            cross = ((up.t() @ w) * down).sum(0) * s
            d_sq = (((up.t() @ up) @ down) * down).sum(0) * (s * s)
        else:
            w1, w2 = self.lokr_w1.float(), self.lokr_w2.float()
            (a, b), (c, d) = w1.shape, w2.shape
            t = torch.einsum("ikjl,ij->kjl", w.reshape(a, c, b, d), w1)  # [c, b, d]: 1/a of the weight
            cross = torch.einsum("kjl,kl->jl", t, w2).reshape(-1) * s
            d_sq = torch.outer(w1.square().sum(0), w2.square().sum(0)).reshape(-1) * (s * s)
        return (w_sq + 2.0 * cross + d_sq).clamp_min(0.0).sqrt()

    def _added(self, x):
        """The LoRA / LoKr output for input x (scaled, before the multiplier)."""
        if self.kind == "lora":
            h = self.dropout(x)
            h = nn.functional.linear(h, self.lora_down.weight.to(h.dtype))
            return nn.functional.linear(h, self.lora_up.weight.to(h.dtype)) * self.scale
        return lokr_update(x, self.lokr_w1.to(x.dtype), self.lokr_w2.to(x.dtype)) * self.scale

    def _rotated(self, y, multiplier: float):
        """OFT: the frozen output y rotated per block, R^T (y - bias) + bias, as y + (R - I)^T (y - bias); a
        multiplier blends R with the identity (ComfyUI's bypass mode does the same)."""
        rot = oft_rotation(self.oft_blocks, self.constraint)
        rot = rot - torch.eye(rot.shape[-1], device=rot.device, dtype=rot.dtype)
        if multiplier != 1.0:
            rot = rot * multiplier
        bias = self.base.bias
        yb = y - bias.to(y.dtype) if bias is not None else y
        n, size = rot.shape[0], rot.shape[1]
        delta = torch.einsum("knm,...kn->...km", rot.to(y.dtype), yb.reshape(*yb.shape[:-1], n, size))
        return y + delta.reshape(y.shape)

    @torch.no_grad()
    def stage_extras(self) -> None:
        """Copy the extras to where the frozen layer's weights are, in the compute dtype, once per render round (a
        block-swapped block's weights wait in RAM: its extras stay there too, cast, and stream in with each forward)."""
        dev, dtype = self.base.weight.device, self.compute_dtype
        pin = dev.type == "cpu" and torch.cuda.is_available()

        def place(t):
            t = t.to(dev, dtype)
            if pin:
                try:
                    t = t.pin_memory()
                except RuntimeError:  # no pinned memory left: pageable works too
                    pass
            return t

        self.staged_extras = [(e[0], place(e[1]), place(e[2]), e[3]) if len(e) == 4
                              else ("lora", place(e[0]), place(e[1]), e[2]) for e in self.extras]
        self.staged_biases = [(place(b), s) for b, s in self.extra_biases]

    def unstage_extras(self) -> None:
        self.staged_extras = self.staged_biases = None

    def _live_extras(self, x):
        """The extras a forward applies: the staged copies when they match the attached ones and sit on x's device
        (or in RAM for a swapped block), else the originals (converted per call, as before staging)."""
        staged, biases = self.staged_extras, self.staged_biases
        if staged is None or len(staged) != len(self.extras) or len(biases) != len(self.extra_biases):
            return self.extras, self.extra_biases  # attached or removed since the round started
        ref = staged[0][1] if staged else (biases[0][0] if biases else None)
        if ref is not None and (ref.dtype != x.dtype or (ref.device != x.device and ref.device.type != "cpu")):
            return self.extras, self.extra_biases  # another input dtype, or the model moved since (model.to)
        return staged, biases

    @property
    def weight(self):  # for code that only inspects .weight.dtype / .device
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        if self.multiplier == 0.0 or not self.trainable:
            out = self.base(x)
        elif self.kind == "oft":
            out = self._rotated(self.base(x), self.multiplier)
        elif self.dora:  # W' x = (W + delta)(g x): the frozen layer and the delta both see the scaled input
            h = x * self.dora_input_scale().to(x.dtype)
            out = self.base(h) + self._added(h)
            if self.multiplier != 1.0:  # ComfyUI's strength: W + s (W' - W)
                plain = self.base(x)
                out = plain + (out - plain) * self.multiplier
        else:
            out = self.base(x) + self._added(x) * self.multiplier
        if self.extras_enabled:
            out = _apply_extras(x, out, *self._live_extras(x))
        elif self.train_down is not None:
            out = out + nn.functional.linear(nn.functional.linear(x, self.train_down.to(x.dtype)),
                                             self.train_up.to(x.dtype))
        return out


def _target_matches(name: str, targets: Iterable[str]) -> bool:
    return name.rsplit(".", 1)[-1] in targets


def parse_blocks(spec: str, n_layers: int) -> list[int] | None:
    """Parse a block spec like "0-41", "10-31,40", "even", "odd"; empty/all -> None (every block)."""
    spec = (spec or "").strip().lower()
    if spec in ("", "all", "*"):
        return None
    out: set[int] = set()
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part == "even":
            out.update(range(0, n_layers, 2))
        elif part == "odd":
            out.update(range(1, n_layers, 2))
        elif part == "none":
            continue
        elif "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(max(0, lo), min(n_layers - 1, hi) + 1))
        else:
            i = int(part)
            if 0 <= i < n_layers:
                out.add(i)
    return sorted(out)


def format_blocks(blocks: Iterable[int]) -> str:
    """Compact "0-5,8,10-12" form."""
    bl = sorted(set(blocks))
    if not bl:
        return "none"
    parts, start, prev = [], bl[0], bl[0]
    for b in bl[1:] + [None]:
        if b is not None and b == prev + 1:
            prev = b
            continue
        parts.append(f"{start}-{prev}" if prev > start else str(start))
        if b is not None:
            start = prev = b
    return ",".join(parts)


def backprop_blocks(model: nn.Module, injected: dict, blocks: list[int] | None) -> list[int] | None:
    """The trained blocks as the backward pass sees them: `blocks` when every LoRA sits in the DiT blocks or after
    them (the head), so the blocks before the first trained one stay forward-only; None (every block runs a backward)
    when a LoRA sits before the blocks (Krea's text fusion, the text / patch / timestep projections, MOSS's text
    embedder, FIBO's refiners), since its gradient flows back through all of them."""
    if blocks is None:
        return None
    tail = getattr(model, "split_tail_modules", None)
    after = [model.layers, *(tail() if tail else [m for m in (getattr(model, "final_norm_video", None),
                                                             getattr(model, "final_linear_video", None)) if m])]
    inside = {id(m) for part in after for m in part.modules()}
    return blocks if all(id(mod) in inside for mod in injected.values()) else None


def _wrap(model: nn.Module, name: str, rank: int, alpha: float, dropout: float = 0.0, trainable: bool = True,
          kind: str = "lora", factor: int = 8, block: int = 32, constraint: float = 0.0, dora: bool = False):
    module = model.get_submodule(name)
    if isinstance(module, LoRALinear):
        return module
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    device = module.weight.device if hasattr(module, "weight") else "cpu"
    wrapper = LoRALinear(module, rank, alpha, dropout, trainable, kind, factor, block, constraint, dora).to(device)
    setattr(parent, child, wrapper)
    return wrapper


def _is_linear_like(m: nn.Module) -> bool:
    return isinstance(m, nn.Linear) or (hasattr(m, "in_features") and hasattr(m, "out_features")
                                        and not isinstance(m, LoRALinear))


def inject_lora(model: nn.Module, rank: int, alpha: float, targets: Iterable[str],
                include_text_embedder: bool = False, dropout: float = 0.0,
                blocks: Iterable[int] | None = None, names: Iterable[str] | None = None,
                kind: str = "lora", factor: int = 8, block: int = 32, constraint: float = 0.0,
                dora: bool = False) -> dict[str, LoRALinear]:
    """Replace matching Linear modules in-place with trainable adapter wrappers (kind "lora", "lokr" or "oft"; dora
    on LoRA / LoKr). Returns {name: LoRALinear}.

    names: the exact module names to wrap (Arch.lora_module_names); without it the MOSS-Image layout rules below
    apply. blocks: DiT block indices to train (None = all). Blocks outside the set get no LoRA; autograd then skips
    the backward pass through every block before the first trained one, which also speeds training up.
    """
    if names is not None:
        wanted = set(names)
        injected = {}
        for name, module in list(model.named_modules()):
            if name in wanted and _is_linear_like(module):
                injected[name] = _wrap(model, name, rank, alpha, dropout, trainable=True, kind=kind, factor=factor,
                                       block=block, constraint=constraint, dora=dora)
        missing = wanted - set(injected)
        if missing:
            raise ValueError(f"LoRA targets not found in the model: {sorted(missing)[:5]}")
        return injected
    targets = set(targets)
    block_set = None if blocks is None else set(int(b) for b in blocks)
    text_targets = {"qkv", "proj", "ffn_gate", "ffn_up", "ffn_down"}
    injected: dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        if not _is_linear_like(module):
            continue
        if name.startswith("layers."):
            if not _target_matches(name, targets):
                continue
            if block_set is not None and int(name.split(".")[1]) not in block_set:
                continue
        elif name.startswith("text_embedder."):
            if not include_text_embedder or not _target_matches(name, text_targets):
                continue
        else:
            continue
        injected[name] = _wrap(model, name, rank, alpha, dropout, trainable=True, kind=kind, factor=factor,
                               block=block, constraint=constraint, dora=dora)
    return injected


@torch.no_grad()
def mirror_lora(src: dict, dst: dict) -> None:
    """Copy LoRA state from one model's wrappers onto another's with the same module names (a data-parallel replica,
    the sampling model): weights (resized when the rank differs), alpha / scale / rank, on/off and dropout."""
    for name, s in src.items():
        d = dst[name]
        dora, block = getattr(s, "dora", False), getattr(s, "block", d.block)
        if (d.kind, d.dora, d.block) != (s.kind, dora, block):
            d.set_kind(s.kind, dora=dora, block=block)
        dps = d.adapter_params()
        for part, sp in s.adapter_params().items():
            dp = dps[part]
            if dp.shape != sp.shape:
                dp.data = torch.empty(sp.shape, dtype=sp.dtype, device=dp.device)
            dp.copy_(sp, non_blocking=True)
        d.alpha, d.scale, d.rank, d.factor = s.alpha, s.scale, s.rank, s.factor
        d.constraint = getattr(s, "constraint", 0.0)
        if d.trainable != s.trainable:
            d.trainable = s.trainable
        if getattr(d.dropout, "p", 0.0) != getattr(s.dropout, "p", 0.0):
            d.dropout = nn.Dropout(s.dropout.p) if getattr(s.dropout, "p", 0.0) > 0 else nn.Identity()


def lora_state(injected: dict) -> dict:
    """Every wrapper's LoRA state as plain data on the CPU (tensors, numbers, strings: torch.load weights_only reads
    it), what mirror_lora copies; lora_snapshots turns it back into something mirror_lora takes as a source."""
    return {name: {"kind": m.kind, "dora": bool(getattr(m, "dora", False)), "block": m.block,
                   "params": {k: p.detach().to("cpu", copy=True) for k, p in m.adapter_params().items()},
                   "alpha": m.alpha, "scale": m.scale, "rank": m.rank, "factor": m.factor,
                   "constraint": float(getattr(m, "constraint", 0.0)), "trainable": m.trainable,
                   "dropout": float(getattr(m.dropout, "p", 0.0))} for name, m in injected.items()}


class LoraSnapshot:
    """One wrapper's lora_state entry, readable by mirror_lora as its source."""

    def __init__(self, d: dict):
        self.kind, self.dora, self.block, self.params = d["kind"], d["dora"], d["block"], d["params"]
        self.alpha, self.scale, self.rank, self.factor = d["alpha"], d["scale"], d["rank"], d["factor"]
        self.constraint, self.trainable = d["constraint"], d["trainable"]
        self.dropout = nn.Dropout(d["dropout"]) if d["dropout"] > 0 else nn.Identity()

    def adapter_params(self) -> dict:
        return self.params


def lora_snapshots(state: dict) -> dict:
    return {name: LoraSnapshot(d) for name, d in state.items()}


def lora_parameters(injected: dict[str, LoRALinear]):
    for m in injected.values():
        yield from m.adapter_params().values()


def lora_pairs(injected: dict[str, LoRALinear]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(lora_down, lora_up) of every LoRA-kind layer, for optimizers that precondition the pair (Keel)."""
    return [(m.lora_down.weight, m.lora_up.weight) for m in injected.values() if m.kind == "lora"]


def lora_param_groups(injected: dict[str, LoRALinear], lr: float, plus_ratio: float = 0.0) -> list[dict]:
    """Optimizer param groups. With LoRA+ (ratio > 1) the LoRA up matrices get lr x ratio in their own group; LoKr,
    OFT and DoRA's magnitudes stay at lr (LoRA+ is a LoRA result)."""
    downs = [m.lora_down.weight for m in injected.values() if m.kind == "lora"]
    ups = [m.lora_up.weight for m in injected.values() if m.kind == "lora"]
    lokr = [p for m in injected.values() for part, p in m.adapter_params().items()
            if part not in ("lora_down", "lora_up")]
    if plus_ratio and plus_ratio > 1.0 and ups:
        groups = [{"params": downs, "lr": lr, "name": "lora_down"}, {"params": ups, "lr": lr * plus_ratio, "name": "lora_up"}]
        if lokr:
            groups.append({"params": lokr, "lr": lr, "name": "lora"})
        return groups
    return [{"params": downs + ups + lokr, "lr": lr, "name": "lora"}]


def lora_param_groups_by(injected: dict[str, LoRALinear], lr: float, group_of) -> list[dict]:
    """One param group per layer type (group_of(module name) -> type), for optimizers that adapt one lr per group."""
    groups: dict[str, list] = {}
    for name, m in injected.items():
        groups.setdefault(group_of(name), []).extend(m.adapter_params().values())
    return [{"params": ps, "lr": lr, "name": g} for g, ps in sorted(groups.items())]


def set_multiplier(injected: dict[str, LoRALinear], value: float) -> None:
    for m in injected.values():
        m.multiplier = value


def export_state_dict(injected: dict[str, LoRALinear], dtype=torch.bfloat16, fmt: str = "comfy") -> dict[str, torch.Tensor]:
    if fmt not in LORA_FORMATS:
        raise ValueError(f"LoRA format must be one of {LORA_FORMATS}, got {fmt!r}")
    out = {}
    for name, m in injected.items():
        if not getattr(m, "trainable", True):
            continue  # a race's switched-off layer (empty tensors)
        parts = {k: v.detach().to("cpu", dtype).contiguous() for k, v in m.adapter_params().items()}
        key = (PEFT_PREFIX if fmt == "peft" else COMFY_PREFIX) + name  # ComfyUI maps "transformer." for Krea 2
        if "dora_scale" in parts:  # DoRA on the input axis: [1, in] (see the module docstring)
            out[f"{key}.dora_scale"] = parts.pop("dora_scale").reshape(1, -1)
        if m.kind == "oft":  # LyCORIS diag-OFT keys; .alpha is the COFT norm cap (absent: none)
            out[f"{key}.oft_blocks"] = parts["oft_blocks"]
            if getattr(m, "constraint", 0.0) > 0:
                out[f"{key}.alpha"] = torch.tensor(float(m.constraint), dtype=torch.float32)
            continue
        if m.kind == "lokr":  # LyCORIS keys
            out[f"{key}.lokr_w1"], out[f"{key}.lokr_w2"] = parts["lokr_w1"], parts["lokr_w2"]
            out[f"{key}.alpha"] = torch.tensor(1.0, dtype=torch.float32)  # full factors: no scaling
            continue
        down, up = parts["lora_down"], parts["lora_up"]
        if fmt == "peft":
            out[f"{PEFT_PREFIX}{name}.lora_A.weight"] = down
            out[f"{PEFT_PREFIX}{name}.lora_B.weight"] = up
        elif fmt == "folded":
            out[f"{key}.lora_A.weight"] = down
            out[f"{key}.lora_B.weight"] = (m.lora_up.weight.detach().float().cpu() * m.scale).to(dtype).contiguous()
        elif fmt == "comfy_ab":  # read at scale 1 (alpha = rank): alpha / rank folded into lora_B
            out[f"{COMFY_PREFIX}{name}.lora_A.weight"] = down
            out[f"{COMFY_PREFIX}{name}.lora_B.weight"] = (m.lora_up.weight.detach().float() * m.scale).to(
                "cpu", dtype).contiguous()
        else:
            key = COMFY_PREFIX + name
            out[f"{key}.lora_down.weight"] = down
            out[f"{key}.lora_up.weight"] = up
            out[f"{key}.alpha"] = torch.tensor(float(m.alpha), dtype=torch.float32)
    return out


def network_kinds(injected: dict) -> set[str]:
    return {m.kind for m in injected.values() if getattr(m, "trainable", True)}


def peft_adapter_metadata(injected: dict[str, LoRALinear]) -> dict:
    """The PEFT LoraConfig for these modules, keys prefixed "transformer." as diffusers' loader expects."""
    cfg = dict(PEFT_CONFIG_TEMPLATE)
    injected = {n: m for n, m in injected.items() if m.kind == "lora" and getattr(m, "trainable", True)}
    ranks = {m.rank for m in injected.values()}
    alphas = {float(m.alpha) for m in injected.values()}
    cfg["r"] = max(ranks)
    cfg["lora_alpha"] = max(alphas)
    cfg["target_modules"] = sorted(injected)
    cfg["rank_pattern"] = {n: m.rank for n, m in injected.items() if m.rank != cfg["r"]}
    cfg["alpha_pattern"] = {n: float(m.alpha) for n, m in injected.items() if float(m.alpha) != cfg["lora_alpha"]}
    return {f"{PEFT_PREFIX}{k}": v for k, v in cfg.items()}


def save_lora(injected: dict[str, LoRALinear], path, metadata: dict | None = None, dtype=torch.bfloat16,
              fmt: str = "comfy") -> None:
    if isinstance(dtype, str):
        dtype = SAVE_DTYPES[dtype]
    meta = {"format": "pt", "rulztrainer_version": "0.1.0"}
    kinds = network_kinds(injected)
    live = [m for m in injected.values() if getattr(m, "trainable", True)]
    dora = any(getattr(m, "dora", False) for m in live)
    if "lokr" in kinds:
        meta["rulztrainer_network"] = "lokr"
        meta["ss_network_module"] = "lycoris.kohya"
        meta["ss_network_args"] = json.dumps({"algo": "lokr", "factor": max(m.factor for m in live
                                                                               if m.kind == "lokr"),
                                              **({"dora_wd": True, "wd_on_output": False} if dora else {})})
    elif "oft" in kinds:
        meta["rulztrainer_network"] = "oft"
        meta["ss_network_module"] = "lycoris.kohya"
        meta["ss_network_args"] = json.dumps({"algo": "diag-oft", "block_size": max(m.block for m in live
                                                                                     if m.kind == "oft"),
                                              "constraint": max(float(getattr(m, "constraint", 0.0)) for m in live)})
    elif dora:
        meta["rulztrainer_network"] = "lora"
        meta["ss_network_args"] = json.dumps({"dora_wd": True, "wd_on_output": False})
    if dora:
        meta["rulztrainer_dora"] = "input axis: <stem>.dora_scale [1, in], ComfyUI's weight_decompose"
    if fmt == "peft" and "lora" in kinds:
        meta["lora_adapter_metadata"] = json.dumps(peft_adapter_metadata(injected), indent=2, sort_keys=True)
    if fmt == "folded" and "lora" in kinds:
        meta[FOLDED_ALPHA_KEY] = json.dumps({n: float(m.alpha) for n, m in injected.items()
                                             if m.kind == "lora" and getattr(m, "trainable", True)}, sort_keys=True)
    if metadata:
        meta.update({k: (v if isinstance(v, str) else json.dumps(v)) for k, v in metadata.items()})
    if fmt == "comfy_ab" and "lora" in kinds:
        meta["rulztrainer_alpha"] = "folded into lora_B: read at alpha = rank"
        if "ss_network_dim" in meta:
            meta["ss_network_alpha"] = meta["ss_network_dim"]
    save_file(export_state_dict(injected, dtype, fmt), str(path), metadata=meta)


def read_peft_metadata(path) -> dict:
    """The PEFT LoraConfig stored in a diffusers-format LoRA file (prefixes stripped), or {}."""
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    raw = meta.get("lora_adapter_metadata")
    if not raw:
        return {}
    return {k.removeprefix(PEFT_PREFIX): v for k, v in json.loads(raw).items()}


def read_folded_alpha(path) -> dict | None:
    """A folded-format file's {module: alpha} (its lora_B already times alpha / rank), or None for another file."""
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    raw = meta.get(FOLDED_ALPHA_KEY)
    if raw is None:
        return None
    return {k.removeprefix(COMFY_PREFIX): float(v) for k, v in json.loads(raw).items()}


def load_lora_weights(injected: dict[str, LoRALinear], path) -> int:
    """Load a previously exported LoRA (either format) into injected modules. Returns number of modules loaded."""
    state = load_file(str(path))
    peft = any(k.endswith(".lora_A.weight") for k in state)
    peft_meta = read_peft_metadata(path) if peft else {}
    folded = read_folded_alpha(path) if peft else None  # a folded file: lora_B carries alpha / rank
    loaded = 0
    dora_fresh = []  # DoRA layers whose file has no magnitude: it starts where the loaded delta leaves the norm
    for name, m in injected.items():
        stem = next((p + name for p in (COMFY_PREFIX, PEFT_PREFIX, "") if any(
            f"{p}{name}.{part}" in state for part in ("oft_blocks", "lokr_w1", "dora_scale", "lora_down.weight",
                                                       "lora_A.weight"))), None)
        if m.dora:
            scale = state.get(f"{stem}.dora_scale") if stem else None
            if scale is None:
                dora_fresh.append(m)
            elif scale.numel() != m.in_features or (scale.dim() == 2 and scale.shape[0] != 1):
                raise ValueError(f"{name}: the file's dora_scale has {tuple(scale.shape)}, not [1, {m.in_features}] "
                                 "(an output-axis DoRA file cannot continue here)")
            else:
                with torch.no_grad():
                    m.dora_scale.copy_(scale.reshape(-1).float())
        if m.kind == "oft":
            blocks = state.get(f"{stem}.oft_blocks") if stem else None
            if blocks is None:
                continue
            if tuple(blocks.shape) != tuple(m.oft_blocks.shape):
                raise ValueError(f"OFT shape mismatch for {name}: file has {tuple(blocks.shape)}, oft_block_size "
                                 f"{m.block} gives {tuple(m.oft_blocks.shape)}")
            with torch.no_grad():
                m.oft_blocks.copy_(blocks.float())
            alpha = state.get(f"{stem}.alpha")
            m.constraint = float(alpha) if alpha is not None else 0.0
            loaded += 1
            continue
        if m.kind == "lokr":
            key = next((p + name for p in (COMFY_PREFIX, PEFT_PREFIX, "") if f"{p}{name}.lokr_w1" in state), None)
            if key is None:
                continue
            w1, w2 = state[f"{key}.lokr_w1"], state[f"{key}.lokr_w2"]
            if w1.shape != m.lokr_w1.shape or w2.shape != m.lokr_w2.shape:
                raise ValueError(f"LoKr shape mismatch for {name}: file has w1 {tuple(w1.shape)} w2 {tuple(w2.shape)}, "
                                 f"factor {m.factor} gives {tuple(m.lokr_w1.shape)} {tuple(m.lokr_w2.shape)}")
            with torch.no_grad():
                m.lokr_w1.copy_(w1.float())
                m.lokr_w2.copy_(w2.float())
            loaded += 1
            continue
        if peft:  # transformer.<module> (PEFT) or diffusion_model.<module> (folded; comfy_ab: alpha = rank)
            key = stem if folded is not None and stem else next(
                (p + name for p in (PEFT_PREFIX, COMFY_PREFIX) if f"{p}{name}.lora_A.weight" in state), PEFT_PREFIX + name)
            down, up = state.get(f"{key}.lora_A.weight"), state.get(f"{key}.lora_B.weight")
        else:
            key = COMFY_PREFIX + name
            down, up = state.get(f"{key}.lora_down.weight"), state.get(f"{key}.lora_up.weight")
        if down is None or up is None:
            continue
        if down.shape != m.lora_down.weight.shape or up.shape != m.lora_up.weight.shape:
            raise ValueError(f"Rank mismatch for {name}: file has rank {down.shape[0]}, config has {m.rank}")
        if folded is not None:  # back to the unscaled factor at the alpha it trained with (scale 1 if none recorded)
            alpha = float(folded.get(name, m.rank))
            up = up.float() / (alpha / m.rank)
        with torch.no_grad():
            m.lora_down.weight.copy_(down.float())
            m.lora_up.weight.copy_(up.float())
        if folded is not None:
            m.alpha = alpha
        elif peft and key.startswith(COMFY_PREFIX):
            alpha = state.get(f"{key}.alpha")
            if alpha is None:  # comfy_ab: the file's delta is lora_B @ lora_A at scale 1; keep the module's alpha
                with torch.no_grad():
                    m.lora_up.weight.mul_(1.0 / m.scale)
        elif peft:
            alpha = (peft_meta.get("alpha_pattern") or {}).get(name, peft_meta.get("lora_alpha"))
        else:
            alpha = state.get(f"{key}.alpha")
        if alpha is not None:
            m.alpha = float(alpha)
            m.scale = m.alpha / m.rank
        loaded += 1
    with torch.no_grad():
        for m in dora_fresh:  # m = ||W + delta|| per input column: the loaded LoRA's effect unchanged (scale 1)
            w = dense_weight(m.base).to(m.dora_scale.device)
            m.dora_scale.copy_((w + m.delta_weight()).norm(dim=0))
    if dora_fresh:
        log.info("DoRA: %d layers had no dora_scale in %s; their magnitude starts at the loaded weights' norm",
                 len(dora_fresh), Path(str(path)).name)
    return loaded


# ---------------------------------------------------------------- extra (sampling-only) LoRAs


def parse_comfy_lora(state: dict[str, torch.Tensor], default_alpha: float | None = None,
                     alpha_pattern: dict | None = None) -> dict[str, tuple]:
    """Return {module_name: (kind, first, second, scale)} from a ComfyUI/PEFT-style LoRA file or a LyCORIS LoKr
    file: ("lora", down, up, scale) or ("lokr", w1, w2, scale). Fused qkv is split. default_alpha / alpha_pattern:
    the alpha for LoRA entries without an .alpha tensor (PEFT files keep it in their metadata: lora_alpha, and per
    module where it differs). LoKr factors are rebuilt from their low-rank parts (w_a @ w_b x alpha / rank) when a
    file stores them that way; Tucker (lokr_t2) is not read."""
    out: dict[str, tuple] = {}
    stems = set()
    for k in state:
        for suf in (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight"):
            if k.endswith(suf):
                stems.add(k[: -len(suf)])
    lokr_stems = {k[: -len(suf)] for k in state for suf in (".lokr_w1", ".lokr_w1_a", ".lokr_w2", ".lokr_w2_a")
                  if k.endswith(suf)}
    for stem in lokr_stems:
        if f"{stem}.lokr_t2" in state:
            log.warning("%s: a Tucker-decomposed LoKr entry cannot be applied while sampling; skipped", stem)
            continue
        alpha = state.get(f"{stem}.alpha")
        alpha = float(alpha) if alpha is not None else None

        def factor_of(part):
            w = state.get(f"{stem}.{part}")
            if w is not None:
                return w.float()
            wa, wb = state.get(f"{stem}.{part}_a"), state.get(f"{stem}.{part}_b")
            if wa is None or wb is None:
                return None
            rank = wb.shape[0]
            return (wa.float() @ wb.float()) * ((alpha / rank) if alpha is not None else 1.0)

        w1, w2 = factor_of("lokr_w1"), factor_of("lokr_w2")
        if w1 is None or w2 is None:
            continue
        out[_module_name(stem)] = ("lokr", w1, w2, 1.0)  # full factors carry no alpha scaling (ComfyUI likewise)
    for stem in stems:
        down = state.get(f"{stem}.lora_down.weight", state.get(f"{stem}.lora_A.weight"))
        up = state.get(f"{stem}.lora_up.weight", state.get(f"{stem}.lora_B.weight"))
        if down is None or up is None:
            continue
        rank = down.shape[0]
        name = _module_name(stem)
        alpha = state.get(f"{stem}.alpha")
        alpha = float(alpha) if alpha is not None else (alpha_pattern or {}).get(name, default_alpha)
        scale = (float(alpha) / rank) if alpha is not None else 1.0
        down, up = down.float(), up.float()
        if name.endswith(".proj.qkv"):
            for n, part in zip("qkv", up.chunk(3, 0)):
                out[name[: -len("qkv")] + n] = ("lora", down, part.contiguous(), scale)
        else:
            out[name] = ("lora", down, up, scale)
    return out


def _module_name(stem: str) -> str:
    """A LoRA file's module stem without its trainer prefix (diffusion_model. / transformer. / lora_unet_)."""
    name = stem
    for p in ("diffusion_model.", "transformer.", "lora_unet_"):
        if name.startswith(p):
            name = name[len(p):]
    if name.startswith("layers_"):  # lora_unet_layers_0_proj_q style
        name = name.replace("_", ".", 2).replace("proj_", "proj.")
    return name


def parse_bias_diffs(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """{module_name: bias offset} from a LoRA file's `<module>.diff_b` entries (ComfyUI adds them to the bias)."""
    return {_module_name(k[: -len(".diff_b")]): v.float() for k, v in state.items() if k.endswith(".diff_b")}


def attach_extra_lora(model: nn.Module, path: str, strength: float = 1.0,
                      rename: Callable[[str], str] | None = None,
                      convert: Callable[[dict], dict] | None = None) -> int:
    """Attach a sampling-only LoRA (e.g. a turbo/step-distill LoRA) to the model, with its diff_b bias offsets and the
    `diff` weight offsets of normalization layers. rename: maps a file's module name that the model does not have to
    the model's own (another trainer's layout, e.g. ComfyUI's Krea 2 names); convert: rewrites the whole file first
    (ComfyUI's fused FIBO 2.0 layout). Returns the modules matched; the unmatched ones are logged."""
    state = load_file(str(path))
    if convert is not None:
        state = convert(state)
    peft_meta = read_peft_metadata(path)
    entries = parse_comfy_lora(state, default_alpha=peft_meta.get("lora_alpha"),
                               alpha_pattern=peft_meta.get("alpha_pattern"))
    biases = parse_bias_diffs(state)
    names = {n for n, _ in model.named_modules()}

    def resolve(name: str) -> str | None:
        if name not in names and rename is not None:
            name = rename(name)
        if name not in names:
            return None
        if isinstance(model.get_submodule(name), LoRALinear):
            return name
        return name if _is_linear_like(model.get_submodule(name)) else None

    matched, missed = set(), []
    for stem in [k[: -len(".diff")] for k in state if k.endswith(".diff")]:  # a norm's weight offset
        name = _module_name(stem)
        if name not in names and rename is not None:
            name = rename(name)
        module = model.get_submodule(name) if name in names else None
        weight = getattr(module, "weight", None) if module is not None and not _is_linear_like(module) else None
        delta = state[stem + ".diff"]
        if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != tuple(delta.shape):
            missed.append(name)
            continue
        diffs = model.__dict__.setdefault("_rz_param_diffs", [])
        diffs.append((weight, weight.detach().clone(), delta.to(weight.device, torch.float32) * strength))
        matched.add(name)
    for kind, table in (("lora", entries), ("bias", biases)):
        for name, value in table.items():
            target = resolve(name)
            if target is None:
                missed.append(name)
                continue
            module = model.get_submodule(target)
            if not isinstance(module, LoRALinear):
                module = _wrap(model, target, rank=1, alpha=1.0, trainable=False)
            if kind == "lora":
                ekind, first, second, scale = value
                if ekind == "lokr":
                    a, b = first.shape
                    c, d = second.shape
                    if a * c != module.out_features or b * d != module.in_features:
                        missed.append(name)
                        continue
                module.extras.append((ekind, first, second, scale * strength))
            else:
                module.extra_biases.append((value, strength))
            matched.add(target)
    if missed:
        log.warning("%s: %d of %d LoRA entries match no layer of this model (e.g. %s)", Path(path).name, len(missed),
                    len(entries) + len(biases), ", ".join(sorted(missed)[:3]))
    read = (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight", ".alpha", ".diff_b",
            ".diff", ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b")
    unread = sorted({k.rsplit(".", 2)[-2] if k.endswith(".weight") else k.rsplit(".", 1)[-1]
                     for k in state if not k.endswith(read)})
    if unread:  # LoHa (hada_w1_a...), full weight diffs (diff), DoRA (dora_scale), Tucker LoKr (lokr_t2)
        log.warning("%s: %d entries of kinds this loader cannot apply are ignored (%s); only LoRA (lora_down/up, "
                    "lora_A/B), LoKr (lokr_w1/w2) and diff_b are applied while sampling", Path(path).name,
                    sum(1 for k in state if not k.endswith(read)), ", ".join(unread[:5]))
    return len(matched)


def attach_training_adapter(model: nn.Module, path: str, strength: float = 1.0,
                            rename: Callable[[str], str] | None = None,
                            convert: Callable[[dict], dict] | None = None) -> int:
    """Attach a training adapter: a LoRA file applied while training and off while sampling (it follows the extras'
    switch: on whenever they are off), never part of the saved LoRA. Only LoRA entries (lora_down/up, lora_A/B) are
    read. Returns the modules matched; the unmatched ones are logged."""
    state = load_file(str(path))
    if convert is not None:
        state = convert(state)
    peft_meta = read_peft_metadata(path)
    entries = parse_comfy_lora(state, default_alpha=peft_meta.get("lora_alpha"),
                               alpha_pattern=peft_meta.get("alpha_pattern"))
    names = {n for n, _ in model.named_modules()}
    matched, missed, skipped = set(), [], 0
    for name, (kind, down, up, scale) in entries.items():
        if kind != "lora":
            skipped += 1
            continue
        if name not in names and rename is not None:
            name = rename(name)
        module = model.get_submodule(name) if name in names else None
        if module is None or not (isinstance(module, LoRALinear) or _is_linear_like(module)) \
                or down.shape[1] != module.in_features or up.shape[0] != module.out_features:
            missed.append(name)
            continue
        if not isinstance(module, LoRALinear):
            module = _wrap(model, name, rank=1, alpha=1.0, trainable=False)
        dev, dtype = module.lora_down.weight.device, module.compute_dtype
        down, up = down.to(dev, dtype), (up * (scale * strength)).to(dev, dtype)
        if module.train_down is not None:  # a second adapter on the layer: the ranks side by side
            down, up = torch.cat([module.train_down, down]), torch.cat([module.train_up, up], dim=1)
        module.train_down, module.train_up = down, up
        matched.add(name)
    if missed:
        log.warning("%s: %d of %d training adapter entries match no layer of this model (e.g. %s)", Path(path).name,
                    len(missed), len(entries), ", ".join(sorted(missed)[:3]))
    if skipped:
        log.warning("%s: %d LoKr entries ignored (a training adapter is read as a LoRA)", Path(path).name, skipped)
    return len(matched)


def extras_active(model: nn.Module) -> bool:
    """Whether a sampling-only LoRA is attached and switched on."""
    return any(isinstance(m, LoRALinear) and m.extras_enabled and (m.extras or m.extra_biases)
               for m in model.modules())


def set_extras_enabled(model: nn.Module, enabled: bool) -> None:
    swap = model.__dict__.get("_rz_block_swap")
    if swap is not None:  # norm offsets change the streamed weights: no stale prefetched copies
        swap.reset()
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.extras_enabled = enabled
            if enabled and (m.extras or m.extra_biases):
                m.stage_extras()  # on the layer's device once, not uploaded on every forward of the round
            else:
                m.unstage_extras()  # the GPU copies go; the originals stay in RAM
    with torch.no_grad():  # norm weight offsets: swapped in for sampling, the original weight back for training
        for weight, original, delta in model.__dict__.get("_rz_param_diffs", []):
            if weight.device != original.device:  # the model moved since the LoRA was attached
                original.data = original.data.to(weight.device)
                delta.data = delta.data.to(weight.device)
            weight.copy_((original.float() + delta).to(weight.dtype) if enabled else original)
