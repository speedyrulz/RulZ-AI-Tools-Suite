"""Likeness probes and controller for character LoRAs.

Every `probe_every` steps the trainer renders one headshot and/or one full-body probe image (fixed seeds, small
size, turbo LoRA) and scores it against the face / body references. The controller smooths those scores
(EMA) and acts on them:

  * best checkpoint + early stop   keep `<job>_best.safetensors` at the best smoothed combined score and stop
                                   after `likeness_patience` probes without improvement
  * bounded lr multiplier          hill-climb a factor on the inner optimizer's lr (works with AdamW-style
                                   optimizers through the scheduler and with Automagic 3 through its lr tensors)
  * face / body curriculum         when the face lags the body (or vice versa), sample close-up dataset images
                                   more often and shift the timestep distribution toward lower (detail) or
                                   higher (structure) noise

The controller never touches the loss; it only steers sampling, lr and when to stop.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("rulztrainer")


def combine(face, body, face_weight: float, body_weight: float):
    parts = [(v, w) for v, w in ((face, face_weight), (body, body_weight)) if v is not None and w > 0]
    if not parts:
        return None
    return sum(v * w for v, w in parts) / sum(w for _, w in parts)


@dataclass
class ProbeRecord:
    step: int
    face: float | None
    body: float | None
    combined: float | None
    face_ema: float | None
    body_ema: float | None
    combined_ema: float | None
    lr_factor: float
    face_bias: float
    best: float | None
    best_step: int | None
    since_improve: int
    images: list[str] = field(default_factory=list)


class LikenessController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.face_ema = None
        self.body_ema = None
        self.combined_ema = None
        self.prev_combined = None
        self.best = None
        self.best_step = None
        self.since_improve = 0
        self.lr_factor = 1.0
        self.face_bias = 0.0
        self.stopped = False
        self.n_probes = 0

    def _ema(self, prev, value):
        if value is None:
            return prev
        if prev is None:
            return value
        a = self.cfg.probe_ema
        return a * prev + (1 - a) * value

    def update(self, step: int, face: float | None, body: float | None, images: list[str] | None = None):
        """Feed one probe result. Returns (record, actions) where actions may contain
        save_best (bool), stop (bool), lr_ratio (float, new/old), face_bias (float)."""
        cfg = self.cfg
        actions: dict = {}
        self.n_probes += 1
        self.face_ema = self._ema(self.face_ema, face)
        self.body_ema = self._ema(self.body_ema, body)
        comb = combine(self.face_ema, self.body_ema, cfg.face_weight, cfg.body_weight)
        self.combined_ema = comb
        band = cfg.likeness_noise_band
        if comb is not None:
            if self.best is None or comb > self.best + band:
                self.best, self.best_step, self.since_improve = comb, step, 0
                actions["save_best"] = cfg.likeness_track_best
            else:
                self.since_improve += 1
                if cfg.likeness_patience > 0 and self.since_improve >= cfg.likeness_patience:
                    self.stopped = True
                    actions["stop"] = True
            if cfg.likeness_lr_control and self.prev_combined is not None:
                delta = comb - self.prev_combined
                new = self.lr_factor
                if delta > band:
                    new = min(cfg.likeness_lr_max, self.lr_factor * (1 + cfg.likeness_lr_step))
                elif delta < -band:
                    new = max(cfg.likeness_lr_min, self.lr_factor * (1 - cfg.likeness_lr_step))
                if abs(new - self.lr_factor) > 1e-12:
                    actions["lr_ratio"] = new / self.lr_factor
                    self.lr_factor = new
            self.prev_combined = comb
        if cfg.likeness_curriculum and self.face_ema is not None and self.body_ema is not None:
            gap = self.body_ema - self.face_ema  # > 0: the face lags the body
            bias = max(-1.0, min(1.0, gap / 20.0)) * cfg.likeness_curriculum_strength
            if abs(bias - self.face_bias) > 0.02:
                self.face_bias = bias
                actions["face_bias"] = bias
        rec = ProbeRecord(step, face, body, combine(face, body, cfg.face_weight, cfg.body_weight),
                          self.face_ema, self.body_ema, comb, self.lr_factor, self.face_bias, self.best,
                          self.best_step, self.since_improve, list(images or []))
        return rec, actions

    def state(self) -> dict:
        return {"best": self.best, "best_step": self.best_step, "since_improve": self.since_improve,
                "lr_factor": self.lr_factor, "face_bias": self.face_bias, "stopped": self.stopped,
                "face_ema": self.face_ema, "body_ema": self.body_ema, "combined_ema": self.combined_ema,
                "n_probes": self.n_probes}


def append_record(job_dir: Path, rec: ProbeRecord) -> None:
    with open(job_dir / "likeness.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec)) + "\n")


# ---------------------------------------------------------------- lr factor on any optimizer


def apply_lr_ratio(optimizer, scheduler, ratio: float) -> None:
    """Multiply the effective learning rate by `ratio`, for schedulers (base lrs) and Automagic 3 (lr tensors)."""
    if hasattr(optimizer, "get_avg_learning_rate"):  # Automagic 3: adaptive per-param lr tensors
        for group in optimizer.param_groups:
            for p in group["params"]:
                st = optimizer.state.get(p)
                if st is not None and "lr" in st:
                    st["lr"].mul_(ratio).clamp_(min=group["min_lr"], max=group["max_lr"])
        return
    if scheduler is not None and hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [b * ratio for b in scheduler.base_lrs]
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * ratio
        if "initial_lr" in group:
            group["initial_lr"] = group["initial_lr"] * ratio


# ---------------------------------------------------------------- curriculum


def face_area_fractions(face_scorer, items, cache_path: Path, progress=None, recache: bool = False) -> dict[str, float]:
    """Largest-face area as a fraction of each item's (cropped) source image, keyed by item.key.

    Uses the shared faces.json analysis cache (source-pixel boxes), so crop items are classified correctly."""
    from .regions import analyze_faces, face_fraction

    analysis = analyze_faces(items, face_scorer, cache_path, progress, recache)
    return {it.key: face_fraction(analysis.get(str(it.image.resolve())), it.src_box) for it in items}


def curriculum_weights(items, fractions: dict[str, float], face_bias: float, threshold: float = 0.03,
                       min_w: float = 0.2, max_w: float = 2.0) -> dict[str, float]:
    """Sampling weight per image (keyed by str(path)). face_bias > 0 favours face-heavy images."""
    out = {}
    for it in items:
        frac = fractions.get(it.key, fractions.get(str(it.image.resolve()), 0.0))
        face_heavy = frac >= threshold
        w = 1.0 + face_bias if face_heavy else 1.0 - face_bias
        out[it.key] = max(min_w, min(max_w, w))
    return out


def timestep_logit_shift(face_bias: float, scale: float = 0.5) -> float:
    """Shift for the logit-normal / sigmoid timestep mean: negative = lower sigma = more detail (face) work."""
    return -face_bias * scale


def classify_counts(fractions: dict[str, float], threshold: float = 0.03) -> tuple[int, int]:
    n_face = sum(1 for v in fractions.values() if v >= threshold)
    return n_face, len(fractions) - n_face


def probe_body_size(size: int) -> tuple[int, int]:
    """Full-body probes are portrait 2:3 at the given width, rounded to multiples of 16."""
    w = max(256, size // 16 * 16)
    h = int(math.ceil(w * 1.5 / 16) * 16)
    return w, h
