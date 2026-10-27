"""Likeness probes and controller for character LoRAs.

Every `probe_every` steps the trainer renders one headshot and/or one full-body probe image (fixed seeds, small
size, turbo LoRA) and scores it against the face / body references. The controller smooths those scores
(EMA) and acts on them:

  * best checkpoint + early stop   keep `<job>_best.safetensors` at the best smoothed combined score and stop
                                   after `likeness_patience` probes without improvement
  * resolution step-up             with `resolution_step_up`, train the lowest selected resolution first; after
                                   `step_up_patience` probes without a new per-stage best move to the next
                                   resolution, and after `step_up_stop_patience` such probes at the highest
                                   one stop training (replaces `likeness_patience` in that mode)
  * bounded lr multiplier          hill-climb a factor on the inner optimizer's lr (works with AdamW-style
                                   optimizers through the scheduler and with Automagic 3 through its lr tensors)
  * face / body curriculum         when the face lags the body (or vice versa), sample close-up dataset images
                                   more often and shift the timestep distribution toward lower (detail) or
                                   higher (structure) noise

Scores. The face score is an ArcFace cosine (a stranger ~10-25, the references against each other ~60-80); the
body score a DINOv2 cosine of the body window (bodysim.body_window; a stranger ~80, the references ~85-90) or a
proportion match (every human ~80, the references ~90-96). Raw, they live on different scales: a 0.7 / 0.3 average
is really ~93% face, and "the face lags the body" is always true. With `likeness_normalize` (the default) each score
is first mapped onto one scale, 0 = what a stranger scores (the step-0 probe: the base model without the LoRA,
rendered with at least `probe_floor_seeds` seeds, or a default) and 100 = what a typical reference scores against
the others (reference_ceiling: the median of each reference's best match, or a default), capped at 100. Each scale
is then stretched (`face_scale_span` / `body_scale_span`, 1.35: its 100 sits 1.35x as far above the stranger as the
references do) and curved (`face_scale_curve` / `body_scale_curve`, 1.5: the score is the position on that scale
raised to this power), so the references' own level reads about 64 on both and the score still has room to grow past
it. Both raw scores saturate near the references' level: an ArcFace cosine because a decent likeness already reaches
it, a DINOv2 cosine because a probe that copies the photos' outfit and setting matches them better than the
references, in their varied outfits, match each other. With the face alone stretched (until v2.1.3) the body read 100
while the face, for pictures that looked equally close, read 50-60. The combined score, the noise band, the curriculum, the lr control and the best / stop / race decisions all run on those.

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


# the raw-score scale each probe kind is normalized on: floor = a stranger's score (replaced by the step-0 probe when
# there is one), ceiling = the references' own self-similarity (replaced by the measured one when there are 2+ refs)
FLOOR_DEFAULT = {"face": 20.0, "body": 80.0}
CEILING_DEFAULT = {"face": 75.0, "body": 90.0}  # body: a window's references ~85-90, proportions ~90-96
MIN_SPAN = {"face": 10.0, "body": 8.0}  # the ceiling stays at least this far above the floor (raw points)
# ... and at least this share of the way from the floor to a perfect 100: a body ceiling (the references' typical best
# match among themselves) that sits barely above the base model's step-0 score is no measure of a likeness, and a few
# points of copied outfit then read 100 (a real run: floor 73.3, ceiling 81.3)
MIN_ROOM_SHARE = {"face": 0.0, "body": 0.5}
# matching the references better than a typical one of them does is no more likeness (a probe that copies one photo
# reads higher), so the scale stops at 100
NORM_MAX = 100.0
# both raw scores saturate (see the module docstring): each scale is stretched by its SPAN_DEFAULT and curved by its
# CURVE_DEFAULT (config: face_scale_span / _curve, body_scale_span / _curve; 1 and 1 give the plain linear scale)
FACE_SPAN_DEFAULT = BODY_SPAN_DEFAULT = 1.35
FACE_CURVE_DEFAULT = BODY_CURVE_DEFAULT = 1.5


def reference_ceiling(embs, usable=None) -> float | None:
    """The references' typical best match, the 100 of the normalized scale: each reference's best cosine against
    the others (the statistic a probe is scored with), the median over the usable ones (all of them when fewer than
    two are usable). The median, not the mean: hard references (a face turned away, one cut off by the frame) pulled
    the mean below what a clean probe of the same person scores, and on a real run a typical probe read 108%."""
    import numpy as np

    e = np.asarray(embs, dtype=np.float32)
    if e.ndim != 2 or len(e) < 2:
        return None
    sims = e @ e.T
    np.fill_diagonal(sims, -np.inf)
    best = sims.max(axis=1)
    if usable is not None:
        keep = np.asarray(usable, dtype=bool)
        if int(keep.sum()) >= 2:
            best = best[keep]
    return float(np.median(best))


def probe_seed_count(cfg, step: int) -> int:
    """Seeds a probe renders: probe_seeds, and at least probe_floor_seeds for the step-0 probe that sets the zero of
    the normalized scale. A seed's luck does not stay paired across a run (on a real run the two probe seeds' gap
    flipped sign once the LoRA learned), so more seeds at step 0 steady the zero; later probes keep probe_seeds
    (the same first seeds)."""
    n = max(1, int(getattr(cfg, "probe_seeds", 1) or 1))
    if step == 0 and getattr(cfg, "likeness_normalize", True):
        n = max(n, int(getattr(cfg, "probe_floor_seeds", 0) or 0))
    return n


def normalize(value, floor: float, ceiling: float, curve: float = 1.0):
    """A raw score as a percent of the way from `floor` (0) to `ceiling` (100), raised to `curve` (above 1: a slower
    start, so a score that saturates early reads lower), clipped to 0..NORM_MAX."""
    if value is None:
        return None
    t = max(0.0, (value - floor) / (ceiling - floor))
    return min(NORM_MAX, 100.0 * t ** curve)


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
    stage: int = 0  # resolution stage index (resolution_step_up)
    stage_best: float | None = None
    stage_since_improve: int = 0
    # normalized scores (likeness_normalize): 0 = a stranger, 100 = the references; combined* are on this scale too
    face_norm: float | None = None
    body_norm: float | None = None
    face_norm_ema: float | None = None
    body_norm_ema: float | None = None
    scale: dict = field(default_factory=dict)  # {"face": [floor, ceiling], "body": [...], "curve": {...}} raw points


class LikenessController:
    def __init__(self, cfg, face_ceiling: float | None = None, body_ceiling: float | None = None):
        """face_ceiling / body_ceiling: the references' self-similarity in raw percent (None: the default)."""
        self.cfg = cfg
        self.normalize = bool(getattr(cfg, "likeness_normalize", True))
        self.ceiling = {"face": face_ceiling if face_ceiling is not None else CEILING_DEFAULT["face"],
                        "body": body_ceiling if body_ceiling is not None else CEILING_DEFAULT["body"]}
        self.floor = {"face": None, "body": None}  # None: the default, until a step-0 probe sets it
        self.span = {"face": float(getattr(cfg, "face_scale_span", FACE_SPAN_DEFAULT)),
                     "body": float(getattr(cfg, "body_scale_span", BODY_SPAN_DEFAULT))}
        self.curve = {"face": float(getattr(cfg, "face_scale_curve", FACE_CURVE_DEFAULT)),
                      "body": float(getattr(cfg, "body_scale_curve", BODY_CURVE_DEFAULT))}
        self.face_ema = None
        self.body_ema = None
        self.combined_ema = None
        self.prev_combined = None
        self.best = None
        self.best_step = None
        self.since_improve = 0
        self.progress = None  # the smoothed score of the latest probe that counted as progress (beat it by the band)
        self.lr_factor = 1.0
        self.face_bias = 0.0
        self.stopped = False
        self.n_probes = 0
        self.last_step: int | None = None  # the step of the latest probe (the configuration's own, in a race)
        # resolution step-up: stage index into cfg.training_sides(), per-stage best and stall counter
        self.step_up = bool(getattr(cfg, "resolution_step_up", False))
        self.n_stages = len(cfg.training_sides()) if hasattr(cfg, "training_sides") else 1
        self.stage = 0
        self.stage_best = None
        self.stage_since_improve = 0
        # identity-loss guard: face EMA best and consecutive probes far below it
        self.face_best = None
        self.face_drop_count = 0
        self.id_loss_vetoed = False
        # best face / body checkpoints (likeness_track_best_face / _body): {kind: (smoothed score, step)}
        self.part_best: dict[str, tuple[float, int]] = {}

    def enter_stage(self, stage: int) -> None:
        """Start a resolution stage: the next probe becomes that stage's baseline."""
        self.stage = max(0, min(stage, self.n_stages - 1))
        self.stage_best = None
        self.stage_since_improve = 0

    @property
    def last_stage(self) -> bool:
        return self.stage >= self.n_stages - 1

    def _ema(self, prev, value):
        if value is None:
            return prev
        if prev is None:
            return value
        a = self.cfg.probe_ema
        return a * prev + (1 - a) * value

    def scale(self, kind: str) -> tuple[float, float]:
        """(floor, ceiling) in raw points for "face" / "body": the ceiling at least MIN_SPAN above the floor (the body's
        also at least halfway to 100, MIN_ROOM_SHARE), then the span stretched (face_scale_span / body_scale_span)."""
        floor = self.floor[kind] if self.floor[kind] is not None else FLOOR_DEFAULT[kind]
        least = max(MIN_SPAN[kind], MIN_ROOM_SHARE[kind] * (100.0 - floor))
        return floor, floor + (max(self.ceiling[kind], floor + least) - floor) * self.span[kind]

    def norm(self, kind: str, value):
        """A raw score on the normalized scale (the raw score itself with likeness_normalize off)."""
        if value is None or not self.normalize:
            return value
        return normalize(value, *self.scale(kind), self.curve[kind])

    def scale_record(self) -> dict:
        """The scale as the probe records keep it ({"face": [floor, ceiling], "body": [...], "curve": {...},
        "span": {...}}): the ceiling is the stretched one, the span what stretched it."""
        return {**{k: list(self.scale(k)) for k in ("face", "body")}, "curve": dict(self.curve), "span": dict(self.span)}

    def _set_floors(self, face, body) -> None:
        """The step-0 probe (the base model, no LoRA) is what a stranger scores: the floor of each scale, so step 0
        reads 0 by construction. A ceiling at or below it (a base model that already resembles the references, or
        a body self-similarity dragged down by the references' varied poses) is raised to floor + MIN_SPAN by
        scale(), never the floor lowered."""
        for kind, value in (("face", face), ("body", body)):
            if value is not None and self.floor[kind] is None:
                self.floor[kind] = float(value)

    def describe_scale(self) -> str:
        parts = []
        for kind in ("face", "body"):
            lo, hi = self.scale(kind)
            src = "step-0 probe" if self.floor[kind] is not None else "default"
            parts.append(f"{kind} {lo:.1f} -> {hi:.1f} (floor: {src}"
                         + (f", curve x{self.curve[kind]:g}" if self.curve[kind] != 1.0 else "") + ")")
        return "; ".join(parts) + (" (raw percent: 0 = a stranger; body 100 = a typical reference's best match among "
                                   "the others, the face scale reaches past it)")

    def update(self, step: int, face: float | None, body: float | None, images: list[str] | None = None):
        """Feed one probe result. Returns (record, actions) where actions may contain
        save_best (bool), save_best_face / save_best_body (bool: the smoothed face / body score beat its best by
        the noise band and its checkpoint is kept), stop (bool), lr_ratio (float, new/old), face_bias (float),
        step_up (int),
        id_loss_veto (bool: the face EMA fell more than 2 noise bands below its best for two probes in a row
        while the identity loss is on)."""
        cfg = self.cfg
        actions: dict = {}
        self.n_probes += 1
        self.last_step = step
        if self.normalize and step == 0:
            self._set_floors(face, body)
        self.face_ema = self._ema(self.face_ema, face)
        self.body_ema = self._ema(self.body_ema, body)
        face_n, body_n = self.norm("face", self.face_ema), self.norm("body", self.body_ema)
        comb = combine(face_n, body_n, cfg.face_weight, cfg.body_weight)
        self.combined_ema = comb
        band = cfg.likeness_noise_band
        if face_n is not None and face is not None:
            if self.face_best is None or face_n > self.face_best:
                self.face_best = face_n
            if face_n < self.face_best - 2 * band:
                self.face_drop_count += 1
            else:
                self.face_drop_count = 0
            if (getattr(cfg, "id_loss_weight", 0.0) > 0 and not self.id_loss_vetoed
                    and self.face_drop_count >= 2):
                self.id_loss_vetoed = True
                actions["id_loss_veto"] = True
        # a new best must beat the old one by the noise band, or with likeness_best_any_rise by any amount
        best_margin = 0.0 if getattr(cfg, "likeness_best_any_rise", False) else band
        for kind, value in (("face", face_n), ("body", body_n)):
            prev = self.part_best.get(kind)
            if value is not None and (prev is None or value > prev[0] + best_margin):
                self.part_best[kind] = (value, step)
                if getattr(cfg, f"likeness_track_best_{kind}", False):
                    actions[f"save_best_{kind}"] = True
        if comb is not None:
            if self.best is None or comb > self.best + best_margin:
                self.best, self.best_step = comb, step
                actions["save_best"] = cfg.likeness_track_best
            if self.progress is None or comb > self.progress + band:  # progress always needs the band
                self.progress, self.since_improve = comb, 0
            else:
                self.since_improve += 1
                if not self.step_up and cfg.likeness_patience > 0 and self.since_improve >= cfg.likeness_patience:
                    self.stopped = True
                    actions["stop"] = True
            if self.step_up:
                if self.stage_best is None or comb > self.stage_best + band:
                    self.stage_best, self.stage_since_improve = comb, 0
                else:
                    self.stage_since_improve += 1
                    if not self.last_stage and self.stage_since_improve >= cfg.step_up_patience:
                        actions["step_up"] = self.stage + 1
                    elif self.last_stage and self.stage_since_improve >= cfg.step_up_stop_patience:
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
        if cfg.likeness_curriculum and face_n is not None and body_n is not None:
            gap = body_n - face_n  # > 0: the face lags the body (both on the normalized scale)
            bias = max(-1.0, min(1.0, gap / 20.0)) * cfg.likeness_curriculum_strength
            if abs(bias - self.face_bias) > 0.02:
                self.face_bias = bias
                actions["face_bias"] = bias
        rec = ProbeRecord(step, face, body,
                          combine(self.norm("face", face), self.norm("body", body), cfg.face_weight, cfg.body_weight),
                          self.face_ema, self.body_ema, comb, self.lr_factor, self.face_bias, self.best,
                          self.best_step, self.since_improve, list(images or []), self.stage, self.stage_best,
                          self.stage_since_improve, self.norm("face", face), self.norm("body", body), face_n, body_n,
                          self.scale_record() if self.normalize else {})
        if "step_up" in actions:
            self.enter_stage(actions["step_up"])
        return rec, actions

    def state(self) -> dict:
        return {"best": self.best, "best_step": self.best_step, "since_improve": self.since_improve,
                "lr_factor": self.lr_factor, "face_bias": self.face_bias, "stopped": self.stopped,
                "face_ema": self.face_ema, "body_ema": self.body_ema, "combined_ema": self.combined_ema,
                "n_probes": self.n_probes, "stage": self.stage, "n_stages": self.n_stages,
                "stage_best": self.stage_best, "stage_since_improve": self.stage_since_improve,
                "id_loss_vetoed": self.id_loss_vetoed, "normalize": self.normalize,
                "part_best": {k: {"score": v, "step": s} for k, (v, s) in self.part_best.items()},
                "face_norm_ema": self.norm("face", self.face_ema), "body_norm_ema": self.norm("body", self.body_ema),
                "scale": self.scale_record() if self.normalize else {}}


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
    """Full-body probes are portrait 2:3 with the face probe's pixel count (size x size), in multiples of 16: a probe
    size equal to the sample size then renders and decodes in the samples' memory (1024 -> 832 x 1248, not 1536 tall)."""
    w = max(256, int(size * math.sqrt(2 / 3)) // 16 * 16)
    h = int(math.ceil(w * 1.5 / 16) * 16)
    return w, h
