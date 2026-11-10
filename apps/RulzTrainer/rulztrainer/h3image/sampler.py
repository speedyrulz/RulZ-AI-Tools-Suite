"""Euler flow-matching sampler for MiniMax-H3-Image, matching diffusers' MiniMaxH3ImageDenoiseStep: the checkpoint's
own 6 sigmas (1, 0.968, 0.888, 0.75, 0.533, 0.232, 0), each step through the output head distilled for it, no
guidance. The update is the released scheduler's velocity form, x_next = x + (sigma_next - sigma) * v, on the
noise-ward velocity the port returns.

Other step counts read the grid piecewise-linearly in its index, as ComfyUI's MiniMaxH3ImageModelSampling does (its
stock schedulers land on the grid at 6 steps), and each step takes the step-size-weighted mean of the heads it spans
(the port's `sigma_next`, ComfyUI's _pdd_head). The model is distilled for the 6.

A CFG above 1 applies the trainer's convention uncond + cfg * (cond - uncond), for experiments only: the checkpoint is
guidance-distilled and the released pipeline has no negative prompt."""
from __future__ import annotations

import torch

from ..h3.model import H3_IMAGE_PDD_SIGMAS


def h3image_sigmas(steps: int, grid=None) -> torch.Tensor:
    """`steps` + 1 sigmas ending in 0: the heads' grid itself at its own step count, else the grid interpolated at
    evenly spaced index positions."""
    grid = torch.tensor([float(s) for s in (grid or H3_IMAGE_PDD_SIGMAS)], dtype=torch.float32)
    n, steps = grid.numel() - 1, max(1, int(steps))
    if steps == n:
        return grid
    pos = torch.linspace(0.0, float(n), steps + 1, dtype=torch.float32)
    i = pos.floor().long().clamp(max=n - 1)
    return torch.lerp(grid[i], grid[i + 1], pos - i)


def uses_distilled_schedule(model) -> bool:
    return True


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=6, cfg=1.0, shift=None,
           seed=0, skip_layer=-1, device="cuda", progress=None, sigmas=None) -> torch.Tensor:
    """Returns normalized latents [1, 24, H/16, W/16] (fp32), the trainer's layout. `shift` (the trainer's
    sample_shift) and `skip_layer` are accepted for the shared signature and ignored. Width and height are rounded
    down to whole 32 px (each token covers 2 x 2 latent cells)."""
    gen = torch.Generator("cpu").manual_seed(seed)
    h, w = max(2, height // 32 * 2), max(2, width // 32 * 2)
    latents = torch.randn((1, model.latent_channels, h, w), generator=gen).to(device)
    if sigmas is None:
        sigmas = h3image_sigmas(steps, getattr(model, "pdd_sigmas", None))
    sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
    heads = getattr(model, "pdd_steps", 1) > 1
    pos_embeds, pos_mask = pos_embeds.to(device), pos_mask.to(device)
    if neg_embeds is not None:
        neg_embeds, neg_mask = neg_embeds.to(device), neg_mask.to(device)
    n = len(sigmas) - 1
    for i in range(n):
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
        step = {"sigma_next": sigma_next} if heads else {}
        t = sigma * 1000.0
        x = latents.to(torch.bfloat16)
        pred = model(x, t, pos_embeds, pos_mask, **step)
        if cfg != 1.0 and neg_embeds is not None:
            un = model(x, t, neg_embeds, neg_mask, **step)
            pred = un + cfg * (pred - un)
        latents = latents + (sigma_next - sigma) * pred.float()
        if progress:
            progress(i + 1, n)
    return latents
