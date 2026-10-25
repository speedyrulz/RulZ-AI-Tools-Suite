"""Weight average (EMA) of the LoRA: saves, samples and probes use the average, training goes on with the raw weights.

At batch size 1 the raw weights lean toward the last few images they saw, so a checkpoint carries the tail of the
shuffle. A running average of the recent steps holds the likeness more steadily. Fizgig measured decay 0.98 against
no average (MiniMax H3, 4-way A/B, 50 epochs, same seed): 61.5% vs 56.5% likeness over the last 10 epochs, with less
than half the spread between epochs; 0.99 smoothed without lifting it and 0.995 lagged the run and finished below off.

The decay here is per image, so the window (about 1 / (1 - decay) images: 50 at 0.98) does not depend on batch size,
gradient accumulation or the number of GPUs: a step of N images applies the product of N per-image decays. Each one
warms up as (1 + n) / (10 + n) over the first images n, and is capped at 1 - 4 / total images, so a short run's average
never spans more than a quarter of it. With one GPU at batch 1 this is Fizgig's update exactly."""
from __future__ import annotations

import contextlib

import torch


class LoraEMA:
    """An fp32 shadow of every LoRA weight, on that weight's device."""

    def __init__(self, injected: dict, decay: float, total_images: int = 0):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.total_images = int(total_images)
        self.keys = [f"{name}.{part}" for name, m in injected.items() for part in m.adapter_params()]
        self.params = [p for m in injected.values() for p in m.adapter_params().values()]
        self.shadow: list[torch.Tensor] = []
        self.images = 0  # images averaged in so far
        self.updates = 0  # optimizer steps averaged in so far
        self._depth = 0  # applied() nesting
        self.reset()

    def image_decay(self, n: int) -> float:
        """Decay applied for the image after n images: warm-up, then the configured decay, capped for short runs."""
        d = min(self.decay, (1.0 + n) / (10.0 + n))
        if self.total_images > 0:
            d = min(d, max(0.0, 1.0 - 4.0 / self.total_images))
        return d

    def step_decay(self, images: int) -> float:
        d = 1.0
        for j in range(images):
            d *= self.image_decay(self.images + j)
        return d

    def steady_step_decay(self, images: int) -> float:
        """The decay of a step of `images` images once the warm-up is over."""
        return self.image_decay(10**9) ** max(1, int(images))

    @torch.no_grad()
    def update(self, images: int = 1) -> float:
        """Average in the current weights after an optimizer step over `images` images; returns that step's decay."""
        if self._depth:
            raise RuntimeError("EMA update while the average is applied to the weights")
        d = self.step_decay(max(1, int(images)))
        torch._foreach_lerp_(self.shadow, [p.detach() for p in self.params], 1.0 - d)
        self.images += max(1, int(images))
        self.updates += 1
        return d

    @property
    def applied_now(self) -> bool:
        return self._depth > 0

    @contextlib.contextmanager
    def applied(self):
        """The average in the LoRA weights for the duration (raw weights kept on the CPU and put back afterwards, also
        after an error). In-place copies, so the optimizer state and compiled graphs keep pointing at the same
        tensors. Nested use is a no-op."""
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        with torch.no_grad():
            backup = [p.detach().to("cpu", copy=True) for p in self.params]
            for p, s in zip(self.params, self.shadow):
                p.copy_(s)
        self._depth = 1
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(self.params, backup):
                    p.copy_(b)
            self._depth = 0

    @torch.no_grad()
    def reset(self) -> None:
        """Start the average over from the current weights."""
        self.shadow = [p.detach().float().clone() for p in self.params]
        self.images = self.updates = 0

    def state_dict(self) -> dict:
        return {"decay": self.decay, "images": self.images, "updates": self.updates,
                "shadow": {k: s.detach().cpu() for k, s in zip(self.keys, self.shadow)}}

    @torch.no_grad()
    def load_state_dict(self, sd: dict) -> None:
        saved = sd.get("shadow", {})
        if set(saved) != set(self.keys):
            raise ValueError("The saved weight average has other LoRA modules than this model")
        for k, s in zip(self.keys, self.shadow):
            s.copy_(saved[k].to(s.device, s.dtype))
        self.images, self.updates = int(sd.get("images", 0)), int(sd.get("updates", 0))

    def nbytes(self) -> int:
        return sum(s.numel() * s.element_size() for s in self.shadow)
