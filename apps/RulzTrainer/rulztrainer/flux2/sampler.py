"""Flow-matching Euler sampler for FLUX.2 Klein 9B. Base follows BFL's schedule (the diffusers Klein pipeline and
ComfyUI's Flux2Scheduler compute the same one): sigmas evenly spaced from 1 to 0, time-shifted by the empirical mu of
the token count and step count, sampled with classifier-free guidance only (Klein has no guidance embedding). The
4-step distilled Klein samples without guidance on ComfyUI's Euler simple schedule at shift 2.02, as Fizgig's previews
do, so samples look like ComfyUI's output.

Guidance follows the trainer's convention for every family: uncond + cfg * (cond - uncond), cfg 1 = off."""
from __future__ import annotations

import math

import torch

BASE_STEPS, BASE_CFG = 20, 3.5  # BFL's defaults for flux.2-klein-base-9b
DISTILLED_STEPS, DISTILLED_SHIFT = 4, 2.02  # ComfyUI's Flux2 sampling shift


def empirical_mu(tokens: int, steps: int) -> float:
    """BFL's compute_empirical_mu: mu from the image tokens and step count."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if tokens > 4300:
        return float(a2 * tokens + b2)
    m200, m10 = a2 * tokens + b2, a1 * tokens + b1
    a = (m200 - m10) / 190.0
    return float(a * steps + m200 - 200.0 * a)


def _shifted(steps: int, mu: float) -> torch.Tensor:
    t = torch.linspace(1.0, 0.0, max(1, int(steps)) + 1, dtype=torch.float64)
    e = math.exp(mu)
    return (e / (e + (1.0 / t - 1.0))).float()


def flux2_sigmas(steps: int, tokens: int) -> torch.Tensor:
    """Base's schedule, ending in 0, for an image of `tokens` latent cells."""
    return _shifted(steps, empirical_mu(tokens, max(1, int(steps))))


def distilled_sigmas(steps: int = DISTILLED_STEPS, shift: float = DISTILLED_SHIFT) -> torch.Tensor:
    """ComfyUI's simple schedule on ModelSamplingFlux(shift), ending in 0."""
    return _shifted(steps, shift)


def model_sigmas(model, steps: int, tokens: int) -> torch.Tensor:
    return distilled_sigmas(steps) if getattr(model, "is_distilled", False) else flux2_sigmas(steps, tokens)


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=BASE_STEPS, cfg=BASE_CFG,
           shift=None, seed=0, skip_layer=-1, device="cuda", progress=None) -> torch.Tensor:
    """Returns normalized, packed latents [1, 128, H/16, W/16] (fp32), the trainer's layout. `shift` and `skip_layer`
    are accepted for the shared sampler signature and ignored. The distilled model samples without guidance whatever
    `cfg` says."""
    gen = torch.Generator("cpu").manual_seed(seed)
    h, w = max(1, height // 16), max(1, width // 16)
    latents = torch.randn((1, model.latent_channels, h, w), generator=gen).to(device)
    sigmas = model_sigmas(model, steps, h * w).to(device)
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
