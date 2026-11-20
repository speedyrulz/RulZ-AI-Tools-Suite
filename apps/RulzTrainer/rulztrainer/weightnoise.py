"""Weight noise (weight_noise): small Gaussian noise added to the LoRA's own weights after every optimizer step.

Ported from the ai-toolkit-perceptual fork (MIT). Its authors report that it biases training toward flat minima,
spreads learning over more of the LoRA's rank and resists overcooking on small datasets; none of that is measured
in RulZ AI Trainer yet. It is off by default (weight_noise 0).

The noise goes on the weights, never on the gradients or the optimizer's state, in place under no_grad:
  relative (default): sigma x the tensor's own RMS, so a zero-initialised LoRA-up tensor gets none until it learns
  absolute:           sigma as it is
  bound_norm:         each tensor is scaled back to its pre-noise norm, so the noise moves the weights but cannot
                      grow them (relative noise alone lets the norm drift upward over a long run)

The noise has a torch.Generator of its own, seeded from (run seed, step): turning it on shifts no other random stream
(data order, timesteps, dropout, the global torch RNG), and a resumed run draws what the uninterrupted one would have.
It is drawn in fp32 on each tensor's own device; a bf16 / fp16 tensor is updated through an fp32 copy and rounded once.

Applies to the LoRA and LoKr tensors. DoRA's magnitude vector and OFT's rotation generators are left alone (see
config.weight_noise_reason): a fine-tune has no adapter at all."""
from __future__ import annotations

import torch

MODES = ("relative", "absolute")
NOISE_KINDS = ("lora", "lokr")  # the adapter kinds whose tensors are plain weights
SKIPPED_PARTS = ("dora_scale",)  # a magnitude vector (about the base weight's column norms), not a weight update


def noise_seed(seed: int, step: int) -> int:
    """The noise generator's seed for the optimizer step `step` (0-based count of that run's steps so far)."""
    return (int(seed) * 1_000_003 + int(step) * 7_919 + 0x5EED) % (2 ** 63 - 1)


def adapter_tensors(injected: dict) -> list[torch.Tensor]:
    """The tensors the noise goes on: every LoRA / LoKr adapter tensor of `injected` (lora.LoRALinear by name),
    nothing else. OFT layers and DoRA's magnitude vectors are skipped."""
    out = []
    for m in injected.values():
        if getattr(m, "kind", "lora") not in NOISE_KINDS:
            continue
        out.extend(p for part, p in m.adapter_params().items() if part not in SKIPPED_PARTS)
    return out


@torch.no_grad()
def apply_weight_noise(tensors, sigma: float, mode: str = "relative", bound_norm: bool = False, seed: int = 0,
                       step: int = 0, stats: bool = False) -> dict | None:
    """Add the noise to every tensor in place. With stats, returns {"weight_norm": sqrt(sum ||w||^2) measured before
    the noise, "noise_norm": sqrt(sum ||noise||^2)} over all the tensors (this reads the GPU: ask only when it will
    be shown); otherwise None. sigma 0 changes nothing (not even the generator is made)."""
    if not sigma or sigma <= 0 or not tensors:
        return None
    if mode not in MODES:
        raise ValueError(f"weight_noise_mode must be one of {', '.join(MODES)}, got {mode!r}")
    gens: dict[str, torch.Generator] = {}
    w_sq, n_sq = [], []
    for w in tensors:
        if w.numel() == 0:
            continue
        key = str(w.device)
        if key not in gens:
            gens[key] = torch.Generator(device=w.device).manual_seed(noise_seed(seed, step))
        buf = w if w.dtype == torch.float32 else w.detach().float()
        scale = float(sigma) if mode == "absolute" else float(sigma) * buf.pow(2).mean().sqrt()  # (0 for a zero tensor)
        noise = torch.randn(buf.shape, generator=gens[key], device=buf.device, dtype=torch.float32) * scale
        pre = buf.norm()
        if stats:
            w_sq.append(pre.pow(2))
            n_sq.append(noise.pow(2).sum())
        buf.add_(noise)
        if bound_norm:
            post = buf.norm()
            buf.mul_(torch.where((pre > 0) & (post > 0), pre / post.clamp_min(1e-30), torch.ones_like(pre)))
        if buf is not w:
            w.copy_(buf)
    if not stats:
        return None
    return {"weight_norm": sum(float(t) for t in w_sq) ** 0.5, "noise_norm": sum(float(t) for t in n_sq) ** 0.5}
