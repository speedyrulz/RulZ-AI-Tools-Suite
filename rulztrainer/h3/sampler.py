"""Euler flow-matching sampler for MiniMax H3 one-frame images, matching diffusers' MiniMaxH3Scheduler pair.

Schedule: `steps` model evaluations on the grid linspace(1, 0, steps + 1) pushed through the video shift (12), with
float32 duplicates collapsed, ending at 0. That is the released scheduler's set_timesteps(steps + 1) (which counts grid
points: its 50 is 49 evaluations; musubi-tuner's --infer_steps N is the same grid as num_inference_steps N + 1), and
the lightx2v Turbo LoRAs' grid q_i = (N - i) / N (their model card: FL2VA Turbo 8-step v1.0 at video shift 12, audio
shift 3, so it samples on this schedule with 8 steps). The audio stream follows the same base grid on its own shift (3):
its sigma at every step is the video sigma's (h3.model.audio_sigma), as in ComfyUI's model and both released schedulers.

One-frame generation is a single-frame video: the 4 audio rows (2 latent frames x 2 channels) start as noise and are
stepped with the audio prediction alongside the video, as musubi's one-frame generation does (the audio is never
decoded). The update is the scheduler's x_next = r x + (1 - r) x0 with r = sigma_next / sigma, written as the plain
Euler step on the noise-ward velocity the port returns.

The checkpoints are guidance-distilled: sample at CFG 1 (no negative prompt). A CFG above 1 applies the trainer's
convention uncond + cfg * (cond - uncond) to both streams, for experiments only."""
from __future__ import annotations

import torch

from .model import AUDIO_CHANNELS, SHIFT_VIDEO, audio_sigma, shift_sigma


def h3_sigmas(steps: int, shift: float = SHIFT_VIDEO) -> torch.Tensor:
    """The video sigmas for `steps` evaluations, ending in 0."""
    base = torch.linspace(1.0, 0.0, max(1, int(steps)) + 1, dtype=torch.float32)
    return torch.unique_consecutive(shift_sigma(base, float(shift)))


def uses_distilled_schedule(model) -> bool:
    """H3's turbo LoRAs sample on the base schedule with fewer steps: nothing to switch (kept for the Arch API)."""
    return False


@torch.no_grad()
def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, width, height, steps=30, cfg=1.0, shift=None,
           seed=0, skip_layer=-1, device="cuda", progress=None) -> torch.Tensor:
    """Returns normalized latents [1, 24, H/16, W/16] (fp32), the trainer's layout. `shift` (the trainer's sample_shift)
    and `skip_layer` are accepted for the shared signature and ignored: H3's shifts are its own (12 video, 3 audio).
    Width and height are rounded down to whole 32 px (each token covers 2 x 2 latent cells)."""
    gen = torch.Generator("cpu").manual_seed(seed)
    h, w = max(2, height // 32 * 2), max(2, width // 32 * 2)
    latents = torch.randn((1, model.latent_channels, h, w), generator=gen).to(device)
    audio = torch.randn((1, model.audio_frames * AUDIO_CHANNELS, model.audio_channels), generator=gen).to(device)
    sigmas = h3_sigmas(steps, model.shift_video).to(device)
    pos_embeds, pos_mask = pos_embeds.to(device), pos_mask.to(device)
    if neg_embeds is not None:
        neg_embeds, neg_mask = neg_embeds.to(device), neg_mask.to(device)
    n = len(sigmas) - 1
    for i in range(n):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        t = sigma * 1000.0
        x = latents.to(torch.bfloat16)
        pred, pred_audio = model(x, t, pos_embeds, pos_mask, audio_states=audio, return_audio=True)
        if cfg != 1.0 and neg_embeds is not None:
            un, un_audio = model(x, t, neg_embeds, neg_mask, audio_states=audio, return_audio=True)
            pred, pred_audio = un + cfg * (pred - un), un_audio + cfg * (pred_audio - un_audio)
        latents = latents + (sigma_next - sigma) * pred.float()
        a, a_next = (audio_sigma(s, model.shift_video, model.shift_audio) for s in (sigma, sigma_next))
        audio = audio + (a_next - a) * pred_audio.float()
        if progress:
            progress(i + 1, n)
    return latents
