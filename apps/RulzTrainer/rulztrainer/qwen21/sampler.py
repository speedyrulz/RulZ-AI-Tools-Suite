"""Flow-matching Euler sampler matching diffusers' QwenImage21Pipeline with the checkpoint's scheduler_config:
sigmas from 1 down to 1/steps, time-shifted exponentially (mu from 0.5 at 256 image tokens to 0.9 at 8192), then
stretched so the last one is 0.02 (shift_terminal), with the final 0 appended. Qwen-Image-2.1-Turbo (and the base
model with Comfy-Org's Turbo LoRA switched on) samples with the Turbo checkpoint's own 8 sigmas instead, unshifted
(its model_index.json sample_sigmas; ComfyUI's Turbo workflow feeds the same list); another step count follows the
same curve, interpolated.

Guidance follows the trainer's convention for every family, uncond + cfg * (cond - uncond) with cfg 1 = off, which
is the pipeline's true_cfg_scale. Qwen-Image 2.1 "is meant to be sampled without guidance": the pipeline's and
ComfyUI's default is CFG 1 (40 and 25 steps)."""
from __future__ import annotations

import math

import numpy as np
import torch

BASE_SEQ_LEN, MAX_SEQ_LEN, BASE_SHIFT, MAX_SHIFT = 256, 8192, 0.5, 0.9
SHIFT_TERMINAL = 0.02
TURBO_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)


def calculate_shift(image_seq_len: int) -> float:
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN)
    return image_seq_len * m + (BASE_SHIFT - m * BASE_SEQ_LEN)


def time_shift(mu: float, sigmas):
    return math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0))


def qwen21_sigmas(steps: int, image_seq_len: int, distilled: bool = False) -> torch.Tensor:
    """The pipeline's schedule for an image of `image_seq_len` tokens (H/16 x W/16), with the final 0 appended."""
    steps = max(1, int(steps))
    if distilled:
        turbo = np.asarray(TURBO_SIGMAS, dtype=np.float64)
        s = turbo if steps == len(turbo) else np.interp(np.linspace(0, len(turbo) - 1, steps),
                                                         np.arange(len(turbo)), turbo)
    else:
        s = time_shift(calculate_shift(image_seq_len), np.linspace(1.0, 1.0 / steps, steps))
        one_minus = 1.0 - s
        s = 1.0 - one_minus / (one_minus[-1] / (1.0 - SHIFT_TERMINAL))
    return torch.cat([torch.tensor(s, dtype=torch.float32), torch.zeros(1)])


def uses_distilled_schedule(model) -> bool:
    """The Turbo schedule: a Turbo model, or the base model with a turbo LoRA attached and switched on for this
    sampling."""
    if getattr(model, "is_distilled", False):
        return True
    if getattr(model, "turbo_lora_attached", False):
        from ..lora import extras_active

        return extras_active(model)
    return False


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=25, cfg=1.0, shift=None,
           seed=0, skip_layer=-1, device="cuda", progress=None) -> torch.Tensor:
    """Returns normalized latents [1, 64, H/16, W/16] (fp32), the trainer's layout. `shift` and `skip_layer` are
    accepted for the shared sampler signature and ignored (the shift comes from the image size). Width and height are
    rounded down to whole 32 px, the pipeline's grid (each image slot of the model covers 2x2 latent cells)."""
    gen = torch.Generator("cpu").manual_seed(seed)
    h, w = max(2, height // 32 * 2), max(2, width // 32 * 2)
    latents = torch.randn((1, model.latent_channels, h, w), generator=gen).to(device)
    sigmas = qwen21_sigmas(steps, h * w, uses_distilled_schedule(model)).to(device)
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
