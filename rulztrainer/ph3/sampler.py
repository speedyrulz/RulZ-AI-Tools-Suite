"""P-H3-Image sampling: H3's one-frame Euler sampler (h3/sampler.py) on the checkpoint's own 8-step sigma grid, the
audio rows stepped on the same sigmas (configure() sets the audio shift equal to the video's), no guidance.

Other step counts fall back to H3's shift-12 grid (as ai-toolkit does); the model is distilled for the 8 steps."""
from __future__ import annotations

import torch

from ..h3.sampler import h3_sigmas
from ..h3.sampler import sample as h3_sample
from .loading import SAMPLE_SIGMAS


def ph3_sigmas(steps: int) -> torch.Tensor:
    if int(steps) == len(SAMPLE_SIGMAS) - 1:
        return torch.tensor(SAMPLE_SIGMAS, dtype=torch.float32)
    return h3_sigmas(steps)


def sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, *, steps=8, **kw) -> torch.Tensor:
    return h3_sample(model, pos_embeds, pos_mask, neg_embeds, neg_mask, steps=steps, sigmas=ph3_sigmas(steps), **kw)
