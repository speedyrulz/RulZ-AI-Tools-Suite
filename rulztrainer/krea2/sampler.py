"""Flow-matching Euler sampler matching diffusers' Krea2Pipeline: sigmas from 1 down to 1/steps, time-shifted
exponentially (FlowMatchEulerDiscreteScheduler with dynamic shifting: mu from 0.5 at 256 image tokens to 1.15 at 6400,
or a fixed 1.15 for the distilled Turbo checkpoint), classifier-free guidance as a second pass with the empty prompt.

Guidance follows the trainer's convention for every family, uncond + cfg * (cond - uncond) with cfg 1 = off. Krea's
own guidance_scale g computes cond + g * (cond - uncond), which is the same with cfg = 1 + g: Krea's 3.5 for Raw is
4.5 here, and Turbo (g = 0) samples at cfg 1."""
from __future__ import annotations

import math

import torch

BASE_SEQ_LEN, MAX_SEQ_LEN, BASE_SHIFT, MAX_SHIFT = 256, 6400, 0.5, 1.15
DISTILLED_MU = 1.15


def calculate_shift(image_seq_len: int) -> float:
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN)
    return image_seq_len * m + (BASE_SHIFT - m * BASE_SEQ_LEN)


def time_shift(mu: float, sigmas: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0))


def krea2_sigmas(steps: int, image_seq_len: int, distilled: bool = False) -> torch.Tensor:
    """The pipeline's schedule for an image of `image_seq_len` tokens (H/16 x W/16), with the final 0 appended."""
    sigmas = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    mu = DISTILLED_MU if distilled else calculate_shift(image_seq_len)
    return torch.cat([time_shift(mu, sigmas).float(), torch.zeros(1)])


def uses_distilled_schedule(model) -> bool:
    """Turbo (distilled) samples with a fixed shift (mu 1.15, as ComfyUI samples every Krea 2 model): a Turbo model,
    or Raw with a turbo LoRA attached and switched on for this sampling."""
    if getattr(model, "is_distilled", False):
        return True
    if getattr(model, "turbo_lora_attached", False):
        from ..lora import extras_active

        return extras_active(model)
    return False


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=28, cfg=4.5, shift=None,
           seed=0, skip_layer=-1, device="cuda", progress=None) -> torch.Tensor:
    """Returns normalized, patchified latents [1, 64, H/16, W/16] (fp32), the trainer's layout. `shift` and
    `skip_layer` are accepted for the shared sampler signature and ignored (the shift comes from the image size)."""
    gen = torch.Generator("cpu").manual_seed(seed)
    h, w = height // 16, width // 16
    latents = torch.randn((1, model.latent_channels, h, w), generator=gen).to(device)
    sigmas = krea2_sigmas(steps, h * w, uses_distilled_schedule(model)).to(device)
    pos_embeds, pos_mask = pos_embeds.to(device), pos_mask.to(device)
    if neg_embeds is not None:
        neg_embeds, neg_mask = neg_embeds.to(device), neg_mask.to(device)
    n = len(sigmas) - 1
    for i in range(n):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        t = sigma * 1000.0
        x = latents.to(torch.bfloat16)
        pred = model(x, t, pos_embeds, pos_mask).float()
        if cfg != 1.0 and neg_embeds is not None:
            uncond = model(x, t, neg_embeds, neg_mask).float()
            pred = uncond + cfg * (pred - uncond)
        latents = latents + (sigma_next - sigma) * pred
        if progress:
            progress(i + 1, n)
    return latents
