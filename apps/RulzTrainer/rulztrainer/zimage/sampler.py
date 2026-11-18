"""Flow-matching Euler sampler for Z-Image. Both checkpoints use ComfyUI's static shift 3 (ModelSamplingAuraFlow):
s = linspace(1, 0, steps + 1), sigma = shift * s / (1 + (shift - 1) * s). Z-Image Turbo samples 8 steps without
guidance; Base 25 steps at CFG 4 against a negative prompt (ComfyUI's template; Tongyi's own reference is shift 6 and
28-50 steps). The model predicts the velocity (noise - x0) in the trainer's convention (zimage.model negates
Z-Image's own output), so one Euler step is latents + (sigma_next - sigma) * velocity.

Guidance follows the trainer's convention for every family: uncond + cfg * (cond - uncond), cfg 1 = off. A distilled
(Turbo) model samples without guidance whatever `cfg` says."""
from __future__ import annotations

import torch

SHIFT = 3.0


def zimage_sigmas(steps: int, shift: float = SHIFT) -> torch.Tensor:
    """The static-shift schedule, ending in 0 (steps + 1 values)."""
    s = torch.linspace(1.0, 0.0, max(1, int(steps)) + 1, dtype=torch.float64)
    return (shift * s / (1.0 + (shift - 1.0) * s)).float()


def model_shift(model) -> float:
    return float(getattr(model, "zimage_shift", SHIFT))


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=25, cfg=4.0, shift=None,
           seed=0, skip_layer=-1, device="cuda", progress=None) -> torch.Tensor:
    """Returns normalized, patchified latents [1, 64, H/16, W/16] (fp32), the trainer's layout. `shift` and
    `skip_layer` are accepted for the shared sampler signature and ignored (the shift is the checkpoint's)."""
    gen = torch.Generator("cpu").manual_seed(seed)
    h, w = max(1, height // 16), max(1, width // 16)
    latents = torch.randn((1, model.latent_channels, h, w), generator=gen).to(device)
    sigmas = zimage_sigmas(steps, model_shift(model)).to(device)
    guided = cfg != 1.0 and neg_embeds is not None and not getattr(model, "is_distilled", False)
    pos_embeds, pos_mask = pos_embeds.to(device), pos_mask.to(device)
    if guided:
        neg_embeds, neg_mask = neg_embeds.to(device), neg_mask.to(device)
    n = len(sigmas) - 1
    for i in range(n):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        t = sigma * 1000.0
        x = latents.to(torch.bfloat16)
        pred = model(x, t, pos_embeds, pos_mask).float()
        if guided:
            uncond = model(x, t, neg_embeds, neg_mask).float()
            pred = uncond + cfg * (pred - uncond)
        latents = latents + (sigma_next - sigma) * pred
        if progress:
            progress(i + 1, n)
    return latents
