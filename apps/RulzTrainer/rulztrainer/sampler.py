"""Flow-matching Euler sampler matching the MOSS-Image reference scheduler (shifted, grid-snapped)."""
from __future__ import annotations

import torch


def moss_sigmas(steps: int, shift: float = 5.0, training_shift: float = 1.0, total: int = 1000) -> torch.Tensor:
    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, torch.zeros(1)])


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=20, cfg=3.0,
           shift=5.0, seed=0, skip_layer=22, device="cuda", progress=None) -> torch.Tensor:
    """Returns normalized latents [1,128,H/16,W/16] (fp32)."""
    gen = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn((1, model.latent_channels, height // 16, width // 16), generator=gen).to(device)
    sigmas = moss_sigmas(steps, shift).to(device)
    pos_embeds, pos_mask = pos_embeds.to(device), pos_mask.to(device)
    if neg_embeds is not None:
        neg_embeds, neg_mask = neg_embeds.to(device), neg_mask.to(device)
    skip = [skip_layer] if skip_layer is not None and skip_layer >= 0 else None
    n = len(sigmas) - 1
    for i in range(n):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        t = sigma * 1000.0
        x = latents.to(torch.bfloat16)
        pred = model(x, t, pos_embeds, pos_mask).float()
        if cfg != 1.0 and neg_embeds is not None:
            uncond = model(x, t, neg_embeds, neg_mask, skip_layers=skip).float()
            pred = uncond + cfg * (pred - uncond)
        latents = latents + (sigma_next - sigma) * pred
        if progress:
            progress(i + 1, n)
    return latents
