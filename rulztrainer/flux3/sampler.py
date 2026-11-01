"""Flow-matching Euler sampler matching diffusers' Flux3 modular pipelines (Flux3AutoBlocks / Flux3DistilledAutoBlocks):
sigmas from 1 down to 1/steps, time-shifted by the constant shift 6.93 (BFL's sampler "alpha", configured on
FlowMatchEulerDiscreteScheduler; no resolution-dependent mu), with the final 0 appended. Base samples 50 steps with
classifier-free guidance 3.5 (an empty negative prompt); the distilled flow-map Klein samples 16 steps without
guidance, each step also telling the model the sigma it jumps to (the next one, 0 after the last).

Guidance follows the trainer's convention for every family, uncond + cfg * (cond - uncond) with cfg 1 = off, which
is diffusers' ClassifierFreeGuidance (use_original_formulation False): Klein Base's 3.5 is 3.5 here."""
from __future__ import annotations

import numpy as np
import torch

SHIFT = 6.93  # config.json sampler.alpha (Base and distilled)
BASE_STEPS, BASE_CFG = 50, 3.5
DISTILLED_STEPS = 16


def flux3_sigmas(steps: int, shift: float = SHIFT) -> torch.Tensor:
    """The pipeline's schedule with the final 0 appended (the same for every image size)."""
    steps = max(1, int(steps))
    s = np.linspace(1.0, 1.0 / steps, steps)
    s = shift * s / (1 + (shift - 1) * s)
    return torch.cat([torch.tensor(s, dtype=torch.float32), torch.zeros(1)])


def uses_distilled_schedule(model) -> bool:
    """The distilled (flow-map) Klein: no guidance, the target sigma passed each step. No turbo LoRA exists for
    Klein Base."""
    return bool(getattr(model, "flow_map", False))


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=BASE_STEPS, cfg=BASE_CFG,
           shift=None, seed=0, skip_layer=-1, device="cuda", progress=None) -> torch.Tensor:
    """Returns normalized, packed latents [1, 128, H/16, W/16] (fp32), the trainer's layout. `shift` and `skip_layer`
    are accepted for the shared sampler signature and ignored (Klein's shift is fixed). A flow-map model samples
    without guidance whatever `cfg` says."""
    gen = torch.Generator("cpu").manual_seed(seed)
    h, w = max(1, height // 16), max(1, width // 16)
    latents = torch.randn((1, model.latent_channels, h, w), generator=gen).to(device)
    sigmas = flux3_sigmas(steps).to(device)
    flow_map = uses_distilled_schedule(model)
    pos_embeds, pos_mask = pos_embeds.to(device), pos_mask.to(device)
    if neg_embeds is not None:
        neg_embeds, neg_mask = neg_embeds.to(device), neg_mask.to(device)
    n = len(sigmas) - 1
    for i in range(n):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        t = sigma * 1000.0
        x = latents.to(torch.bfloat16)
        if flow_map:
            pred = model(x, t, pos_embeds, pos_mask, target_timestep=sigma_next * 1000.0).float()
        else:
            pred = model(x, t, pos_embeds, pos_mask).float()
            if cfg != 1.0 and neg_embeds is not None:
                uncond = model(x, t, neg_embeds, neg_mask).float()
                pred = uncond + cfg * (pred - uncond)
        latents = latents + (sigma_next - sigma) * pred
        if progress:
            progress(i + 1, n)
    return latents
