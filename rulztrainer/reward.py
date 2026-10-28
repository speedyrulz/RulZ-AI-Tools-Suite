"""Likeness reward fine-tuning: an optional second stage after the regular training (reward_steps > 0), after
Flow-GRPO (Liu et al., NeurIPS 2025), the online reinforcement learning recipe for flow-matching models.

Regular training teaches the LoRA to rebuild the training photos. This stage asks the thing that matters directly:
over prompts the photos never had, which of the LoRA's own pictures look most like the person? Each step it

  1. renders a group of images (reward_group, 4) of one prompt from different seeds, with a stochastic sampler: the
     ODE sampler turned into an SDE with the same marginals (Flow-GRPO's step, noise level reward_noise), so every
     denoising step is a Gaussian move with a known probability;
  2. scores them with the likeness probes' own scorers (face: ArcFace against the face references, body: the DINOv2
     body window against the body references), on the normalized scale, combined with face_weight / body_weight;
  3. turns the scores into advantages relative to the group (score - group mean) / group std, so only the
     differences between the four pictures count, not the prompt's difficulty;
  4. on a few of the denoising steps (reward_train_steps), raises the log-probability of the moves that led to the
     better pictures and lowers it for the worse ones (policy gradient), with a KL penalty (reward_kl) that keeps the
     moves close to the LoRA as the stage started.

The score is not differentiable here (the scorers detect, crop and embed), so the LoRA cannot chase the scorer's
gradients pixel by pixel; it can still find what the scorer likes that a person would not (reward hacking): the
KL penalty, a small learning rate and the likeness probes on their own fixed seeds (which keep the best checkpoint
and can stop the stage) are the guards. Judge the result by eye too.

Rendering a group costs reward_group x the denoising steps in forward passes, training another 2 x
reward_train_steps (the policy and its reference) per image: the most expensive stage in the trainer.
"""
from __future__ import annotations

import math

import torch

DEFAULT_PROMPTS = (
    ("face", "close-up portrait photo of {t}, soft window light"),
    ("face", "close-up photo of {t} laughing, studio lighting, plain background"),
    ("both", "photo of {t} smiling outdoors in a park, upper body, natural light"),
    ("both", "photo of {t} sitting in a cafe, warm evening light"),
    ("both", "photo of {t} in a winter coat on a snowy street, upper body"),
    ("body", "full body photo of {t} walking down a city street"),
    ("body", "full body photo of {t} standing on a beach at sunset"),
    ("body", "full body photo of {t} in a living room, evening light"),
)


def default_prompts(trigger: str, face_weight: float, body_weight: float) -> list[str]:
    """Varied prompts for the subject: close-ups, upper-body and full-body shots (only the kinds the likeness score
    counts: no full-body prompts when the body weighs nothing, no close-ups when the face does)."""
    t = trigger or "the person"
    keep = {"both"} | ({"face"} if face_weight > 0 else set()) | ({"body"} if body_weight > 0 else set())
    return [p.format(t=t) for kind, p in DEFAULT_PROMPTS if kind in keep]


PART_MIN, PART_MAX = -0.5, 1.5  # a part's reward on the probes' linear scale (0 = a stranger, 1 = the references)


def clip_part(value: float) -> tuple[float, bool]:
    """(the face or body reward kept within PART_MIN..PART_MAX, whether it fell below PART_MIN). Far below a stranger
    says nothing about likeness: the picture differs from the references in something else (a body in a coat against
    unclothed body references, an upper-body shot against full-body ones). Unclipped, such a score swamps the face
    in the group's ranking (the body's scale is several times narrower than the face's), so the stage learns to
    look like the reference photos rather than like the person."""
    return min(PART_MAX, max(PART_MIN, value)), value < PART_MIN


def is_full_body(prompt: str) -> bool:
    return "full body" in prompt.lower() or "full-body" in prompt.lower()


def sde_step_stats(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float, sigma_cap: float,
                   noise_level: float) -> tuple[torch.Tensor, float]:
    """Flow-GRPO's SDE step for rectified flow (x = (1 - sigma) x0 + sigma noise, v predicting noise - x0): the mean
    of the Gaussian move from sigma to sigma_next and its standard deviation. noise_level 0 gives the Euler step of
    the usual sampler. sigma_cap replaces sigma in 1 - sigma at the first step (sigma = 1)."""
    dt = sigma_next - sigma  # negative
    std_t = noise_level * math.sqrt(sigma / (1.0 - (sigma_cap if sigma >= 1.0 else sigma)))
    mean = x * (1 + std_t ** 2 / (2 * sigma) * dt) + v * (1 + std_t ** 2 * (1 - sigma) / (2 * sigma)) * dt
    return mean, std_t * math.sqrt(-dt)


def log_prob(x_next: torch.Tensor, mean: torch.Tensor, std: float) -> torch.Tensor:
    """[B]: the Gaussian log density of each move, averaged over the latent's elements (Flow-GRPO's convention)."""
    lp = -((x_next.detach() - mean) ** 2) / (2 * std ** 2) - math.log(std) - 0.5 * math.log(2 * math.pi)
    return lp.mean(dim=tuple(range(1, lp.ndim)))


def kl_to(mean: torch.Tensor, mean_ref: torch.Tensor, std: float) -> torch.Tensor:
    """[B]: KL divergence between two Gaussian moves with the same std (per element, averaged)."""
    return ((mean - mean_ref) ** 2).mean(dim=tuple(range(1, mean.ndim))) / (2 * std ** 2)


def advantages(rewards: list[float], eps: float = 1e-4) -> torch.Tensor | None:
    """Group-relative advantages (r - mean) / std; None when the group's rewards do not differ (nothing to learn)."""
    r = torch.tensor(rewards, dtype=torch.float32)
    std = r.std(unbiased=False)
    if float(std) < 1e-6:
        return None
    return (r - r.mean()) / (std + eps)


def velocity(model, x: torch.Tensor, sigma: float, pos_e, pos_m, neg_e, neg_m, cfg_scale: float, skip=None):
    """The guided velocity the samplers step with: v_cond, or v_uncond + cfg (v_cond - v_uncond)."""
    t = torch.full((x.shape[0],), sigma * 1000.0, device=x.device)
    v = model(x.to(torch.bfloat16), t, pos_e, pos_m).float()
    if cfg_scale != 1.0 and neg_e is not None:
        kw = {"skip_layers": skip} if skip else {}
        u = model(x.to(torch.bfloat16), t, neg_e, neg_m, **kw).float()
        v = u + cfg_scale * (v - u)
    return v


@torch.no_grad()
def sample_group(model, sigmas: torch.Tensor, pos_e, pos_m, neg_e, neg_m, *, cfg_scale: float, noise_level: float,
                 seeds: list[int], shape: tuple[int, int, int], device, skip=None, progress=None):
    """Render len(seeds) images of one prompt with the SDE sampler. Returns (latents per step [T + 1] of [G, C, h, w]
    fp32 on `device`, std per step [T]). The final latents are the images."""
    gens = [torch.Generator("cpu").manual_seed(int(s)) for s in seeds]
    x = torch.stack([torch.randn(shape, generator=g) for g in gens]).to(device)
    sig = [float(s) for s in sigmas]
    cap = sig[1] if len(sig) > 1 else 0.999
    xs, stds = [x], []
    for i in range(len(sig) - 1):
        v = velocity(model, x, sig[i], pos_e, pos_m, neg_e, neg_m, cfg_scale, skip)
        mean, std = sde_step_stats(x, v, sig[i], sig[i + 1], cap, noise_level)
        noise = torch.stack([torch.randn(shape, generator=g) for g in gens]).to(device)
        x = mean + std * noise if sig[i + 1] > 0 else mean  # the last move lands on the image itself
        xs.append(x)
        stds.append(std if sig[i + 1] > 0 else 0.0)
        if progress:
            progress(i + 1, len(sig) - 1)
    return xs, stds
