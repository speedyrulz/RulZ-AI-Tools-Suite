"""Flow-matching Euler sampler matching Ideogram 4's pipelines: a logit-normal schedule with a resolution-aware mean,
classifier-free guidance against the text-free (image-only) pass, and Ideogram's "polish" steps.

Schedule (Ideogram's scheduler.py and diffusers' _logit_normal_sigmas, the same formula): u = linspace(0, 1, N + 1),
t(u) = 1 - expit(mean + std * ndtri(u)) in the model's time (0 noise, 1 clean), clamped to the logSNR range
[-15, 18], mean = mu + 0.5 * ln(H * W / 512^2); sigma = 1 - t, descending from 0.99988, and the last step ends at 0
(diffusers' scheduler and ComfyUI's Ideogram4Scheduler both force the full denoise; Ideogram's own loop stops at
sigma 0.00055). Ideogram's presets (sampler_configs.py): V4_QUALITY_48 (mu 0, std 1.5, 45 steps at guidance 7 and 3
polish steps at 3), V4_DEFAULT_20 (mu 0, std 1.75, 18 + 2), V4_TURBO_12 (mu 0.5, std 1.75, 11 + 1). Other step counts
take V4_DEFAULT_20's (V4_TURBO_12's below 16 steps); with the community TurboTime LoRA switched on the schedule is
ai-toolkit's sampler default (mu 0, std 1.75), the one its samples were made with.

Guidance: Ideogram blends v = g * v_cond + (1 - g) * v_uncond, which is this trainer's convention
uncond + cfg * (cond - uncond) with cfg = g (7 for the presets). Its unconditional pass runs a second, separately
trained transformer (unconditional_transformer, the same size) on the image tokens alone; the trainer has the
conditional model only, so the negative pass here is the conditional model (with the LoRA) on the image tokens alone
(the empty negative prompt has no tokens), as ai-toolkit samples. That is an approximation of Ideogram's guidance:
expect samples a little off the official pipeline's. The last steps run at min(cfg, 3), the presets' polish."""
from __future__ import annotations

import math

import torch

LOGSNR_MIN, LOGSNR_MAX = -15.0, 18.0
BASE_PIXELS = 512 * 512
PRESETS = {48: (0.0, 1.5, 3), 20: (0.0, 1.75, 2), 12: (0.5, 1.75, 1)}  # steps: (mu, std, polish steps)
TURBO_SCHEDULE = (0.0, 1.75)  # ai-toolkit's Ideogram 4 sampler default
POLISH_CFG = 3.0


def resolution_mean(mu: float, height: int, width: int) -> float:
    return mu + 0.5 * math.log((height * width) / BASE_PIXELS)


def model_time(u: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Ideogram's LogitNormalSchedule: model time (0 noise, 1 clean) at quantiles u, in float64."""
    t = 1.0 - torch.special.expit(mean + std * torch.special.ndtri(u.to(torch.float64)))
    return t.clamp(1.0 / (1.0 + math.exp(0.5 * LOGSNR_MAX)), 1.0 / (1.0 + math.exp(0.5 * LOGSNR_MIN)))


def schedule_params(steps: int, distilled: bool = False) -> tuple[float, float]:
    """(mu, std) for a step count: Ideogram's preset for it, V4_DEFAULT_20's from 16 steps, V4_TURBO_12's below;
    the turbo LoRA's own (TURBO_SCHEDULE) when it is on."""
    if distilled:
        return TURBO_SCHEDULE
    if steps in PRESETS:
        return PRESETS[steps][:2]
    return PRESETS[20][:2] if steps >= 16 else PRESETS[12][:2]


def polish_steps(steps: int) -> int:
    return PRESETS[steps][2] if steps in PRESETS else max(1, round(steps / 12))


def ideogram4_sigmas(steps: int, height: int, width: int, mu: float = 0.0, std: float = 1.75) -> torch.Tensor:
    """The schedule (sigma = 1 - model time, descending, the final 0 included) for a height x width image."""
    steps = max(1, int(steps))
    u = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    sigmas = (1.0 - model_time(u, resolution_mean(mu, height, width), std)).flip(0)
    sigmas[-1] = 0.0
    return sigmas.to(torch.float32)


def guidance_schedule(steps: int, cfg: float) -> list[float]:
    """Per-step guidance: cfg, then the polish steps at min(cfg, 3) (Ideogram's presets drop 7 to 3 at the end)."""
    n = polish_steps(steps) if cfg > POLISH_CFG else 0
    return [float(cfg)] * (steps - n) + [min(float(cfg), POLISH_CFG)] * n


def uses_distilled_schedule(model) -> bool:
    """The turbo LoRA's schedule: a turbo LoRA attached and switched on for this sampling (no distilled Ideogram 4
    checkpoint exists)."""
    if getattr(model, "is_distilled", False):
        return True
    if getattr(model, "turbo_lora_attached", False):
        from ..lora import extras_active

        return extras_active(model)
    return False


def sample_sigmas(model, steps: int, height: int, width: int) -> torch.Tensor:
    return ideogram4_sigmas(steps, height, width, *schedule_params(steps, uses_distilled_schedule(model)))


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=20, cfg=7.0, shift=None,
           seed=0, skip_layer=-1, device="cuda", progress=None) -> torch.Tensor:
    """Returns normalized latents [1, 128, H/16, W/16] (fp32), the trainer's layout. `shift` and `skip_layer` are
    accepted for the shared sampler signature and ignored (the schedule comes from the step count and image size).
    The negative embeds of the empty prompt have no tokens: the text-free pass."""
    gen = torch.Generator("cpu").manual_seed(seed)
    h, w = height // 16, width // 16
    latents = torch.randn((1, model.latent_channels, h, w), generator=gen).to(device)
    sigmas = sample_sigmas(model, steps, h * 16, w * 16).to(device)
    pos_embeds, pos_mask = pos_embeds.to(device), pos_mask.to(device)
    if neg_embeds is not None:
        neg_embeds, neg_mask = neg_embeds.to(device), neg_mask.to(device)
    n = len(sigmas) - 1
    gw = guidance_schedule(n, cfg)
    for i in range(n):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        t = sigma * 1000.0
        x = latents.to(torch.bfloat16)
        pred = model(x, t, pos_embeds, pos_mask).float()
        if gw[i] != 1.0 and neg_embeds is not None:
            uncond = model(x, t, neg_embeds, neg_mask).float()
            pred = uncond + gw[i] * (pred - uncond)
        latents = latents + (sigma_next - sigma) * pred
        if progress:
            progress(i + 1, n)
    return latents
