"""Flow-matching Euler sampler matching diffusers' BriaFibo2Pipeline: sigmas from 1 down to 1/steps, time-shifted by
the image size (FlowMatchEulerDiscreteScheduler with dynamic exponential shifting, base 0.5 at 256 tokens to 1.15 at
4096), classifier-free guidance as a second pass with the null caption."""
from __future__ import annotations

import math

import torch

BASE_SEQ_LEN, MAX_SEQ_LEN, BASE_SHIFT, MAX_SHIFT = 256, 4096, 0.5, 1.15


def calculate_shift(image_seq_len: int) -> float:
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN)
    return image_seq_len * m + (BASE_SHIFT - m * BASE_SEQ_LEN)


def fibo_sigmas(steps: int, image_seq_len: int) -> torch.Tensor:
    """The pipeline's schedule for an image of `image_seq_len` tokens (H/16 x W/16), with the final 0 appended."""
    sigmas = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    mu = calculate_shift(image_seq_len)
    sigmas = math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0))
    return torch.cat([sigmas.float(), torch.zeros(1)])


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=30, cfg=5.0, shift=None,
           seed=0, skip_layer=-1, device="cuda", progress=None) -> torch.Tensor:
    """Returns normalized, patchified latents [1, 128, H/16, W/16] (fp32), the trainer's layout. `shift` and
    `skip_layer` are accepted for the shared sampler signature and ignored (FIBO's shift comes from the image size)."""
    gen = torch.Generator("cpu").manual_seed(seed)
    h, w = height // 16, width // 16
    latents = torch.randn((1, model.latent_channels, h, w), generator=gen).to(device)
    sigmas = fibo_sigmas(steps, h * w).to(device)
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
