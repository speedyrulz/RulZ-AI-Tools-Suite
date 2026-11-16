"""Euler sampler for SDXL on the flow facade (model.SdxlUNet).

The schedule is diffusers' "trailing" timestep spacing (999 down, `steps` of them, as the SDXL pipelines' default
Euler / DDIM configurations space them for SDXL), mapped to the flow noise levels s(t) = b / (a + b), ending in 0.
Euler in flow space over these levels is DDIM (eta 0): each step lands exactly on (1-s') x0 + s' eps for the model's
x0 / eps prediction. The start is the VP pure noise in flow units, noise * init_noise_scale(s_max). Guidance follows
the trainer's convention, uncond + cfg * (cond - uncond), applied to the velocity (equal to guiding epsilon: the two
weights sum to 1); cfg 1 or no negative = off."""
from __future__ import annotations

import torch

from .model import NUM_TRAIN_TIMESTEPS, init_noise_scale, sigma_of_timestep


def trailing_timesteps(steps: int) -> torch.Tensor:
    """diffusers' timestep_spacing="trailing": round(arange(1000, 0, -1000/steps)) - 1, from 999 down."""
    ratio = NUM_TRAIN_TIMESTEPS / steps
    t = torch.arange(NUM_TRAIN_TIMESTEPS, 0, -ratio, dtype=torch.float64).round().long() - 1
    return t[:steps]


def sdxl_sigmas(steps: int, prediction_type: str = "epsilon", zero_terminal_snr: bool = False) -> torch.Tensor:
    """The noise levels of `steps` trailing-spaced timesteps with the final 0 appended (steps + 1 values, fp32)."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    s = sigma_of_timestep(trailing_timesteps(steps), zero_terminal_snr)
    return torch.cat([s.float(), torch.zeros(1)])


def _no_skip(skip_layer) -> bool:
    return skip_layer is None or skip_layer == "" or (isinstance(skip_layer, (int, float)) and skip_layer < 0) \
        or (isinstance(skip_layer, (list, tuple, set)) and not skip_layer)


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=30, cfg=6.0, shift=None,
           seed=0, skip_layer=-1, device="cuda", progress=None) -> torch.Tensor:
    """Returns normalized, patchified latents [1, 16, H/16, W/16] (fp32), the trainer's layout. `shift` is accepted
    for the shared sampler signature and ignored; a skip_layer request raises (SDXL has no skip-layer guidance)."""
    if not _no_skip(skip_layer):
        raise ValueError("SDXL does not support skip-layer guidance: set the skip layer to -1")
    gen = torch.Generator("cpu").manual_seed(seed)
    h, w = height // 16, width // 16
    noise = torch.randn((1, model.latent_channels, h, w), generator=gen).to(device)
    sigmas = sdxl_sigmas(steps, getattr(model, "prediction_type", "epsilon"),
                         getattr(model, "zero_terminal_snr", False)).to(device)
    latents = noise * init_noise_scale(float(sigmas[0]))
    pos_embeds, pos_mask = pos_embeds.to(device), pos_mask.to(device) if pos_mask is not None else None
    if neg_embeds is not None:
        neg_embeds = neg_embeds.to(device)
        neg_mask = neg_mask.to(device) if neg_mask is not None else None
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
