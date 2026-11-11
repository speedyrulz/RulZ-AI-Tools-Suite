"""Euler flow-matching sampler for PAN 2 one-frame images.

Schedule: `steps` model evaluations on linspace(1, 0, steps + 1) pushed through the time shift (9, the released
model's), ending at 0, as IFM's FlowEulerScheduler steps it. Guidance is plain classifier-free guidance in the
trainer's convention, uncond + cfg * (cond - uncond), which is the reference's v_neg + scale * (v_pos - v_neg);
the released settings are 50 steps at 2.5 with the structured negative prompt (pan2/loading.default_negative).

A sample is a single-frame video by default. PAN 2 saw stills in pretraining (image_sample_prob 0.2) but was finished
on video alone, so a one-frame sample shows what a LoRA learned; it is not what the model is tuned to produce. With
`frames` above 1 the sample is a clip, text-to-video: (frames - 1) / 4 + 1 latent frames (the Wan 2.2 VAE packs 4
frames into one after the first), at 24 frames per second."""
from __future__ import annotations

import torch

from .model import SHIFT, shift_sigma


def pan2_sigmas(steps: int, shift: float = SHIFT) -> torch.Tensor:
    """The sigmas for `steps` evaluations, ending in 0."""
    base = torch.linspace(1.0, 0.0, max(1, int(steps)) + 1, dtype=torch.float32)
    return torch.unique_consecutive(shift_sigma(base, float(shift)))


FPS = 24  # the released model's frame rate
TEMPORAL = 4  # frames per latent frame after the first (the Wan 2.2 VAE)


def latent_frames(frames: int) -> int:
    return (max(1, int(frames)) - 1) // TEMPORAL + 1


def snap_frames(frames: int) -> int:
    """The clip length the VAE gives back for a request: 4k + 1 frames."""
    return (latent_frames(frames) - 1) * TEMPORAL + 1


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=50, cfg=2.5, shift=None,
           seed=0, skip_layer=-1, device="cuda", progress=None, sigmas=None, frames=1) -> torch.Tensor:
    """Returns normalized latents [1, 48, H/16, W/16] (fp32), the trainer's layout; for a clip (frames above 1)
    [1, 48, T, H/16, W/16] with T = latent_frames(frames). `shift` (the trainer's
    sample_shift, made for MOSS-Image) and `skip_layer` are accepted for the shared signature and ignored: PAN 2
    samples on its own shift. Width and height are rounded down to whole 32 px (a token covers 2 x 2 latent cells)."""
    gen = torch.Generator("cpu").manual_seed(seed)
    h, w = max(2, height // 32 * 2), max(2, width // 32 * 2)
    shape = (1, model.latent_channels, h, w) if frames <= 1 else (1, model.latent_channels, latent_frames(frames), h, w)
    latents = torch.randn(shape, generator=gen).to(device)
    sigmas = pan2_sigmas(steps, getattr(model, "shift", SHIFT)) if sigmas is None else torch.as_tensor(sigmas, dtype=torch.float32)
    sigmas = sigmas.to(device)
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
