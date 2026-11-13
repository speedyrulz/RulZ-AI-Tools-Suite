"""Region watch: how far each part of the training pictures has been learned, from the training steps' own losses.

Every training step already has its picture's squared error per 16 px cell. With the picture's parts map (the
MediaPipe parser's classes per cell, regions.build_parts) that error splits into five regions - face, hair, body skin,
clothes, background - at no extra model call. Two things are read from it:

  * "learned" per region, over every picture: the region's loss divided by a frozen baseline curve of the run's own
    start (the mean loss per noise level over the first epoch, or the first BASELINE_MIN samples of a small set),
    averaged over an epoch. 1 minus that is the share of the region's starting error removed. It reads a few points
    low: the model already learns during the baseline epoch.
  * "contrast" per picture, for the background and the clothes: that region's loss over the subject's (face, hair and
    skin together) in the same step, so the noise level and most of the picture's own difficulty cancel. It is divided
    by the baseline's ratio at that noise level and by the picture's own first START_OBS readings, then smoothed. 1 =
    as at the start; under 1 = that region is being learned faster than the subject in this picture.

The watch only observes: the loss is read before any region weighting. One training sample per picture per epoch at a
random noise level is a noisy reading, so the per-picture contrast is a smoothed trend, not a measurement, and a
region of fewer than MIN_CELLS cells is left out.

Two ways to act on the parts maps, both off by default and neither run in a real training yet:

  * region_class_weights: one fixed loss weight per region for the whole run (parse_class_weights, weight_maps).
  * region steering (region_steer): once an epoch the watch moves loss weights a step, each rule on its own switch:
      - cap    (cohort): the background's / the clothes' learned share may be at most `cap` x the subject's. Past it,
               that region is turned down in every picture, further in the pictures where it runs further ahead.
      - ahead  (per picture): a picture whose region's contrast is under `ahead` x the median picture's is turned
               down in that picture alone, whatever the cohort does.
      - boost  (cohort): a subject region (face, hair, skin) lagging the subject as a whole is raised, up to `boost`.
               Off unless asked for: a region that stays hard is often a bad picture.
      - prior  (cohort): while the background is past its cap, a prior-preservation term pulls the background's
               prediction towards the base model's, up to `prior`.
      - holdout (cohort, needs held-out pictures and regionsweep.py): a region learned `holdout` further on the
               training pictures than on the held-out ones, and no longer improving on those, is turned down in
               every picture. The one rule that may turn the face, hair or skin down.
      - subject_ahead (per picture, needs sweeps of the training pictures): a picture whose face, hair or skin has
               less than `subject_ahead` x the median picture's error left, by the sweeps, is turned down in that
               picture alone. The steps have no clean reading of one picture's subject; the sweeps do.
    `signal` chooses what the cap rule's per-picture share and the ahead rule read: the training steps' own contrast
    ("steps"), or the fixed-noise sweeps' ("sweep": cleaner, but only as fresh as the last sweep of that picture).
    A weight moves at most `step` an epoch (back up half as fast), never under `min`, and returns when its cause is
    gone. Weights are renormalised to mean 1 per sample: what is taken from a region goes to the rest of the picture.

Output in the job folder: region_watch.jsonl (one line per epoch) and region_watch.json (the whole state: a resumed
run reads it back, because a baseline taken from an already trained model would be wrong).
"""
from __future__ import annotations

import json
import logging
import math
import statistics
import threading
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("rulztrainer")

REGIONS = ("face", "hair", "skin", "clothes", "background")
SUBJECT = "subject"  # face, hair and skin together
PROTECTED = ("background", "clothes")  # the regions whose contrast with the subject is tracked
COLUMNS = REGIONS + (SUBJECT,)
# the parser's classes (regions.PART_*): background 0, hair 1, body skin 2, face skin 3, clothes 4, others 5
CHANNELS = {"face": (3,), "hair": (1,), "skin": (2,), "clothes": (4, 5), "background": (0,), SUBJECT: (1, 2, 3)}
MIN_CELLS = 2.0  # a region smaller than this in a picture is not read
BUCKETS = 12  # noise levels for the baseline curves
KERNEL = 1.5  # buckets: the ratio curves are smoothed over their neighbours (one odd picture must not dent them)
BASELINE_MIN = 64  # samples before the baseline freezes (whole epochs: a small set takes several)
START_OBS = 3  # a picture's first readings set its own starting contrast
BETA = 0.8  # the per-picture contrast's smoothing (about 5 epochs)
SUBJECT_REGIONS = ("face", "hair", "skin")
MIN_SUBJECT_LEARNED = 0.05  # steering waits until the cohort's subject has learned this much: ratios mean little before
LEARNED_BETA = 0.5  # the cohort's learned shares, smoothed over about two epochs, are what steering reads
SPREAD = (0.5, 2.0)  # how far a picture's own contrast may move its share of a cohort-wide cut
BOOST_MARGIN = 0.1  # a subject region is lagging when it is this far (as a share) behind the subject as a whole
SNAP = 0.02  # a weight this close to its target lands on it
HOLDOUT_FLAT = 2  # sweeps without a better held-out reading before the holdout rule acts on a region
MIN_PICTURES = 4  # the subject_ahead rule needs this many swept pictures with the region before it compares them
STATE_FILE, LOG_FILE = "region_watch.json", "region_watch.jsonl"


def region_losses(sq, parts: list):
    """Per sample, the mean squared error inside each of COLUMNS and its size in cells.

    sq: [B, C, h, w] per-cell squared error (no region weights). parts: per sample a parts tensor ([6 or 7, h, w]
    uint8 class fractions) or None. Returns (losses, cells), each a list per sample of [len(COLUMNS)] floats, or None
    for a sample without a parts map or with one of another size."""
    import torch

    idx = [i for i, p in enumerate(parts) if p is not None and tuple(p.shape[-2:]) == tuple(sq.shape[-2:])]
    out_l, out_c = [None] * len(parts), [None] * len(parts)
    if not idx:
        return out_l, out_c
    with torch.no_grad():
        f = torch.stack([parts[i][:6] for i in idx]).to(sq.device, torch.float32) / 255.0  # [n, 6, h, w]
        groups = torch.stack([f[:, list(CHANNELS[c])].sum(dim=1) for c in COLUMNS], dim=1)  # [n, 6, h, w]
        cell = sq.detach().float().mean(dim=1)[idx]  # [n, h, w]
        den = groups.sum(dim=(2, 3))
        num = (cell[:, None] * groups).sum(dim=(2, 3))
        losses, cells = (num / den.clamp_min(1e-6)).cpu().tolist(), den.cpu().tolist()
    for j, i in enumerate(idx):
        out_l[i], out_c[i] = losses[j], cells[j]
    return out_l, out_c


def parse_class_weights(spec: str) -> dict[str, float]:
    """region_class_weights as {region: weight}: "face=3, hair=1.5, background=0.3" (a region left out stays at 1; ""
    is {}). Raises ValueError with what to change."""
    out: dict[str, float] = {}
    for part in str(spec or "").replace(";", ",").split(","):
        if not part.strip():
            continue
        name, sep, value = part.partition("=")
        name = name.strip().lower()
        try:
            weight = float(value)
        except ValueError:
            weight = float("nan")
        if not sep or name not in REGIONS or not 0.0 <= weight <= 10.0:
            raise ValueError(f"region_class_weights: {part.strip()!r} must be <region>=<weight from 0 to 10>, the "
                             f"region one of {', '.join(REGIONS)}")
        out[name] = weight
    if out and len(out) == len(REGIONS) and not any(w > 0 for w in out.values()):
        raise ValueError("region_class_weights: at least one region needs a weight above 0")
    return out


def merge_gains(*gains: dict | None) -> dict:
    """Several {region: weight} multiplied together (a region left out of one is 1 there)."""
    out: dict[str, float] = {}
    for gain in gains:
        for region, value in (gain or {}).items():
            out[region] = out.get(region, 1.0) * float(value)
    return out


def weight_maps(parts: list, gains: list, like):
    """Per sample the loss weight per cell from a weight per region: [B, 1, h, w] with mean 1 per sample, on like's
    device and type (like: a tensor with the loss's grid). gains: per sample {region: weight} (left out = 1) or None.
    None when no sample has both a parts map of this grid and a weight other than 1."""
    import torch

    h, w = like.shape[-2:]
    out = None
    for i, gain in enumerate(gains):
        p = parts[i]
        if not gain or p is None or tuple(p.shape[-2:]) != (h, w) or all(abs(v - 1.0) < 1e-6 for v in gain.values()):
            continue
        if out is None:
            out = torch.ones(len(gains), 1, h, w, device=like.device, dtype=like.dtype)
        f = p[:6].to(like.device, torch.float32) / 255.0
        per_class = torch.ones(6, device=like.device)
        for region, value in gain.items():
            per_class[list(CHANNELS[region])] = float(value)
        wmap = (f * per_class[:, None, None]).sum(dim=0) + (1.0 - f.sum(dim=0)).clamp_min(0.0)  # (uint8 rounding)
        mean = wmap.mean()  # (a picture that is all weight 0 keeps weight 1: nothing to renormalise to)
        out[i, 0] = torch.where(mean > 1e-6, wmap / mean.clamp_min(1e-6), torch.ones_like(wmap)).to(like.dtype)
    return out


def background_mask(parts: list, like):
    """Per sample the background's share of each cell, [B, 1, h, w] float on like's device (0 for a sample without a
    parts map of this grid), or None when no sample has one."""
    import torch

    h, w = like.shape[-2:]
    out = None
    for i, p in enumerate(parts):
        if p is None or tuple(p.shape[-2:]) != (h, w):
            continue
        if out is None:
            out = torch.zeros(len(parts), 1, h, w, device=like.device, dtype=torch.float32)
        out[i, 0] = p[list(CHANNELS["background"])].to(like.device, torch.float32).sum(dim=0) / 255.0
    return out


def tracked(item) -> bool:
    """A picture the watch reads: not one of LoRA Isolation's side-by-side pairs (two people in one frame)."""
    return not getattr(item, "pair_side", "")


@dataclass
class PictureStat:
    key: str
    seen: int = 0
    start_n: dict = field(default_factory=dict)  # protected region -> readings that went into its start
    start_log: dict = field(default_factory=dict)  # protected region -> sum of log(reading) so far
    contrast_log: dict = field(default_factory=dict)  # protected region -> smoothed log contrast since its start
    cells: dict = field(default_factory=dict)  # region -> cells at the last reading
    gain: dict = field(default_factory=dict)  # protected region -> its loss weight under steering (absent = 1)

    def contrast(self, region: str) -> float | None:
        v = self.contrast_log.get(region)
        return None if v is None else math.exp(v)


def bucket(sigma: float) -> int:
    return min(BUCKETS - 1, max(0, int(float(sigma) * BUCKETS)))


class Curve:
    """A value against the noise level: per bucket the mean noise level and the mean value of what was added, read
    by straight lines between the buckets' points (and along the nearest line outside them). The loss changes fastest
    at low noise, where reading a bucket's mean for every noise level inside it would be far off."""

    def __init__(self, sums: list[float] | None = None, counts: list[float] | None = None,
                 sigmas: list[float] | None = None):
        self.sums = list(sums) if sums else [0.0] * BUCKETS
        self.counts = list(counts) if counts else [0.0] * BUCKETS
        self.sigmas = list(sigmas) if sigmas else [0.0] * BUCKETS  # the sum of the noise levels per bucket

    def add(self, sigma: float, value: float) -> None:
        b = bucket(sigma)
        self.sums[b] += value
        self.counts[b] += 1.0
        self.sigmas[b] += float(sigma)

    def points(self, smooth: float = 0.0) -> list[tuple[float, float]]:
        """(noise level, value) per bucket with something in it; smooth: each over its neighbours, by a kernel that
        many buckets wide."""
        full = [k for k in range(BUCKETS) if self.counts[k] > 0]
        if smooth <= 0:
            return [(self.sigmas[k] / self.counts[k], self.sums[k] / self.counts[k]) for k in full]
        pts = []
        for k in full:
            w = [math.exp(-((j - k) ** 2) / (2 * smooth ** 2)) for j in full]
            n = sum(wj * self.counts[j] for wj, j in zip(w, full))
            pts.append((sum(wj * self.sigmas[j] for wj, j in zip(w, full)) / n,
                        sum(wj * self.sums[j] for wj, j in zip(w, full)) / n))
        return pts

    def at(self, sigma: float, positive: bool = False, smooth: float = 0.0) -> float | None:
        """The curve at a noise level, or None with nothing added. positive: a loss curve, kept above a quarter of
        its nearest point when a line runs outside the buckets. smooth: see points."""
        pts = self.points(smooth)
        if not pts:
            return None
        if len(pts) == 1:
            return pts[0][1]
        s = float(sigma)
        hi = next((j for j, (x, _) in enumerate(pts) if x >= s), len(pts) - 1)
        hi = max(1, hi)
        (x0, y0), (x1, y1) = pts[hi - 1], pts[hi]
        y = y0 if x1 <= x0 else y0 + (y1 - y0) * (s - x0) / (x1 - x0)
        if positive:
            y = max(y, 0.25 * (y0 if abs(s - x0) < abs(s - x1) else y1))
        return y


def _move(g: float, target: float, step: float, lo: float, hi: float) -> float:
    """A weight one epoch nearer its target: down by at most `step` of itself, up half as fast (and by at least a
    little, so a weight at 0 can come back)."""
    if target < g:
        g = max(target, g * (1.0 - step))
    elif target > g:
        g = min(target, max(g * (1.0 + step / 2), g + step / 4))
    if abs(g - target) < SNAP:
        g = target
    return min(hi, max(lo, g))


class RegionWatch:
    def __init__(self, items, job_dir, resume: bool = False, steer: dict | None = None):
        """steer: None (observe only) or the steering settings {"regions": names from PROTECTED, "cap", "ahead",
        "boost", "prior", "holdout", "signal", "min", "step", "strength"} (a rule at 0 is off)."""
        self.job_dir = Path(job_dir)
        self.steer = dict(steer) if steer else None
        self.steer_off = ""  # why steering cannot act this run, when it cannot
        self.steering = False  # some weight is away from 1 right now (or the steered prior is on)
        self.learned_ema: dict[str, float] = {}
        self.class_gain: dict[str, float] = {}  # boost / holdout: a region's weight in every picture (absent = 1)
        self.prior_weight = 0.0  # the steered prior-preservation weight on the background
        self.sweep_contrast: dict[str, dict] = {}  # key -> {protected region: contrast} from the last sweeps
        self.holdout_gap: dict[str, float] = {}  # region -> learned on training pictures minus on held-out ones
        self.holdout_flat: dict[str, int] = {}  # region -> sweeps without a better held-out reading
        self.sweep_learned: dict[str, dict] = {}  # key -> {region: learned share since its first sweep}
        self.per_epoch = max(1, len(items))  # every sample counts towards the epoch, read or not
        self.stats: dict[str, PictureStat] = {str(it.key): PictureStat(str(it.key)) for it in items if tracked(it)}
        self.epoch = 0
        self.epoch_obs = 0
        self.frozen = False
        self.late = False  # the baseline was taken on an already trained model (a resume without a saved state)
        self.buffer: list[tuple] = []  # (key, sigma, losses, cells) until the baseline freezes
        self.loss_curves = {c: Curve() for c in COLUMNS}  # the frozen mean loss per noise level
        self.ratio_curves = {p: Curve() for p in PROTECTED}  # the frozen mean log(region / subject) per noise level
        self.epoch_sum = {c: 0.0 for c in COLUMNS}
        self.epoch_n = {c: 0 for c in COLUMNS}
        self.history: list[dict] = []
        self.unread = 0  # samples without a usable parts map this epoch
        self._lock = threading.Lock()
        if resume:
            self.late = not self._load()
            if self.late and self.steer:  # its baseline would be an already trained model's
                self.steer, self.steer_off = None, "a resume without a saved region watch state"
        else:  # a fresh run of the same job starts its files again
            for name in (STATE_FILE, LOG_FILE):
                try:
                    (self.job_dir / name).unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------ persistence
    def _load(self) -> bool:
        try:
            data = json.loads((self.job_dir / STATE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not data.get("frozen"):
            return False
        try:
            self.loss_curves = {c: Curve(**data["loss_curves"][c]) for c in COLUMNS}
            self.ratio_curves = {p: Curve(**data["ratio_curves"][p]) for p in PROTECTED}
            if any("sigmas" not in rec for rec in (*data["loss_curves"].values(), *data["ratio_curves"].values())):
                return False  # a state saved by 2.49.0, whose curves were read another way
            for key, rec in (data.get("pictures") or {}).items():
                if key in self.stats:
                    self.stats[key] = PictureStat(key, int(rec.get("seen", 0)), dict(rec.get("start_n") or {}),
                                                  dict(rec.get("start_log") or {}), dict(rec.get("contrast_log") or {}),
                                                  dict(rec.get("cells") or {}),
                                                  dict(rec.get("gain") or {}) if self.steer else {})
            self.epoch, self.late = int(data.get("epoch", 0)), bool(data.get("late"))
            self.history = list(data.get("history") or [])
            self.learned_ema = {str(k): float(v) for k, v in (data.get("learned_ema") or {}).items()}
            if self.steer:
                self.class_gain = {str(k): float(v) for k, v in (data.get("class_gain") or {}).items()}
                self.prior_weight = float(data.get("prior_weight") or 0.0)
        except (KeyError, TypeError, ValueError):
            return False
        self.frozen = True
        self._flag()
        return True

    def _save(self) -> None:
        data = {"epoch": self.epoch, "frozen": self.frozen, "late": self.late, "history": self.history[-400:],
                "learned_ema": self.learned_ema, "class_gain": self.class_gain, "prior_weight": self.prior_weight,
                "loss_curves": {c: {"sums": v.sums, "counts": v.counts, "sigmas": v.sigmas}
                                for c, v in self.loss_curves.items()},
                "ratio_curves": {p: {"sums": v.sums, "counts": v.counts, "sigmas": v.sigmas}
                                 for p, v in self.ratio_curves.items()},
                "pictures": {k: {"seen": st.seen, "start_n": st.start_n, "start_log": st.start_log,
                                 "contrast_log": st.contrast_log, "cells": st.cells, "gain": st.gain}
                             for k, st in self.stats.items()}}
        from .atomic import writing  # here, not at the top: the WebUI imports this module and atomic loads PyTorch

        try:  # written whole: a resumed run reads it, and a kill mid-write left a torn file that read as no state
            with writing(self.job_dir / STATE_FILE) as tmp:
                tmp.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------ observation
    def observe(self, keys: list[str], sigmas: list[float], losses: list, cells: list,
                skip: list[bool] | None = None) -> None:
        """One training batch (thread-safe): per sample its key, noise level and region_losses' two rows. skip: the
        samples not to read (a dropped caption); they still count towards the epoch."""
        with self._lock:
            for i, key in enumerate(keys):
                self.epoch_obs += 1
                if skip is not None and skip[i]:
                    continue
                st = self.stats.get(str(key))
                if st is None:
                    continue
                row, size = losses[i], cells[i]
                if row is None or size is None or not all(math.isfinite(v) for v in row):
                    self.unread += 1
                    continue
                st.seen += 1
                st.cells = {c: round(float(size[j]), 1) for j, c in enumerate(COLUMNS) if c != SUBJECT}
                if self.frozen:
                    self._ingest(st, float(sigmas[i]), row, size)
                else:
                    self.buffer.append((st.key, float(sigmas[i]), list(row), list(size)))

    def _ingest(self, st: PictureStat, sigma: float, row: list[float], size: list[float]) -> None:
        loss = {c: row[j] for j, c in enumerate(COLUMNS) if size[j] >= MIN_CELLS}
        for c, v in loss.items():
            base = self.loss_curves[c].at(sigma, positive=True)
            if base and base > 0:
                self.epoch_sum[c] += v / base
                self.epoch_n[c] += 1
        subject = loss.get(SUBJECT)
        if not subject or subject <= 0:
            return
        for p in PROTECTED:
            base = self.ratio_curves[p].at(sigma, smooth=KERNEL)
            if p not in loss or loss[p] <= 0 or base is None:
                continue
            reading = math.log(loss[p] / subject) - base
            n = int(st.start_n.get(p, 0))
            if n < START_OBS:  # its own start: the mean of its first readings (not pulled towards the cohort's:
                # that would leave a picture with an easy background reading as ahead for the whole run)
                st.start_n[p], st.start_log[p] = n + 1, float(st.start_log.get(p, 0.0)) + reading
                if n + 1 == START_OBS:
                    st.contrast_log[p] = 0.0
                continue
            v = reading - float(st.start_log[p]) / START_OBS
            st.contrast_log[p] = BETA * float(st.contrast_log.get(p, 0.0)) + (1 - BETA) * v

    def _freeze(self) -> None:
        """The baseline curves from everything read so far, then those readings again as the first ones."""
        for key, sigma, row, size in self.buffer:
            for j, c in enumerate(COLUMNS):
                if size[j] >= MIN_CELLS:
                    self.loss_curves[c].add(sigma, row[j])
            subject = row[COLUMNS.index(SUBJECT)]
            if size[COLUMNS.index(SUBJECT)] >= MIN_CELLS and subject > 0:
                for p in PROTECTED:
                    j = COLUMNS.index(p)
                    if size[j] >= MIN_CELLS and row[j] > 0:
                        self.ratio_curves[p].add(sigma, math.log(row[j] / subject))
        self.frozen = True
        buffered, self.buffer = self.buffer, []
        for key, sigma, row, size in buffered:
            self._ingest(self.stats[key], sigma, row, size)
        self.epoch_sum, self.epoch_n = {c: 0.0 for c in COLUMNS}, {c: 0 for c in COLUMNS}  # the baseline is 0 learned

    def epoch_complete(self) -> bool:
        return self.epoch_obs >= self.per_epoch

    # ------------------------------------------------------------ the epoch's numbers
    def end_epoch(self, step: int) -> dict:
        """Close the epoch: the cohort's learned share per region and the pictures' contrasts. Returns the summary."""
        with self._lock:
            self.epoch += 1
            self.epoch_obs = 0
            unread, self.unread = self.unread, 0
            if not self.frozen:
                if len(self.buffer) >= BASELINE_MIN:
                    self._freeze()
                return self.summary(step, unread=unread)
            learned = {c: (round(1.0 - self.epoch_sum[c] / self.epoch_n[c], 4) if self.epoch_n[c] else None)
                       for c in COLUMNS}
            self.epoch_sum, self.epoch_n = {c: 0.0 for c in COLUMNS}, {c: 0 for c in COLUMNS}
            for c, v in learned.items():
                if v is not None:
                    prev = self.learned_ema.get(c)
                    self.learned_ema[c] = v if prev is None else LEARNED_BETA * prev + (1 - LEARNED_BETA) * v
            row = {"epoch": self.epoch, "step": int(step), "learned": learned, "contrast": self.contrasts()}
            if self.steer:
                row["steer"] = self._steer(self.contrasts(self._signal()))
            self.history.append(row)
            try:
                with open(self.job_dir / LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except OSError:
                pass
            self._save()
            return self.summary(step, unread=unread)

    # ------------------------------------------------------------ steering
    def _flag(self) -> None:
        self.steering = bool(self.class_gain or self.prior_weight > 0 or any(st.gain for st in self.stats.values()))

    def _signal(self) -> str:
        return "sweep" if self.steer and self.steer.get("signal") == "sweep" else "steps"

    def set_sweep(self, contrasts: dict, gap: dict | None = None, flat: dict | None = None,
                  learned: dict | None = None) -> None:
        """The latest fixed-noise readings (regionsweep.RegionSweep): per training picture its contrasts and its
        learned share per region, and with held-out pictures the gap per region and how long each has gone without a
        better held-out reading."""
        with self._lock:
            self.sweep_contrast = {str(k): dict(v) for k, v in (contrasts or {}).items()}
            self.sweep_learned = {str(k): dict(v) for k, v in (learned or {}).items()}
            self.holdout_gap = {str(k): float(v) for k, v in (gap or {}).items() if v is not None}
            self.holdout_flat = {str(k): int(v) for k, v in (flat or {}).items()}

    def _contrast(self, st: PictureStat, region: str, signal: str = "steps") -> float | None:
        if signal == "sweep":
            return (self.sweep_contrast.get(st.key) or {}).get(region)
        return st.contrast(region)

    def _steer(self, contrasts: dict) -> dict:
        """This epoch's weights, each a step towards its target. Returns per steered region {"cohort": how far the
        cohort is past its cap (1 = within it), "lowered": pictures turned down, "lowest": the lowest weight}, plus
        "boost" (the raised subject regions) and "prior" (the steered prior weight)."""
        cfg, out = self.steer, {}
        step, lo, strength = float(cfg["step"]), float(cfg["min"]), float(cfg["strength"])
        subject = self.learned_ema.get(SUBJECT)
        ready = subject is not None and subject >= MIN_SUBJECT_LEARNED
        cohorts = {}
        for p in cfg["regions"]:
            cohort = 1.0
            learned = self.learned_ema.get(p)
            if float(cfg["cap"]) > 0 and ready and learned is not None:
                floor = 1.0 - float(cfg["cap"]) * subject  # the least error the region may have left
                cohort = min(1.0, max(0.0, (1.0 - learned) / floor)) if floor > 0 else 1.0
            cohorts[p] = cohort
            median = (contrasts.get(p) or {}).get("median")
            lowered, lowest = 0, 1.0
            for st in self.stats.values():
                if float(st.cells.get(p, 0.0)) < MIN_CELLS:
                    st.gain.pop(p, None)
                    continue
                want, own = cohort, self._contrast(st, p, self._signal())
                if own is not None and median:
                    rel = min(SPREAD[1], max(SPREAD[0], own / median))
                    if cohort < 1.0:
                        want = cohort * rel
                    if float(cfg["ahead"]) > 0:
                        want = min(want, rel / float(cfg["ahead"]))
                target = max(lo, min(1.0, want) ** (2.0 * strength)) if want < 1.0 else 1.0
                g = _move(float(st.gain.get(p, 1.0)), target, step, lo, 1.0)
                if g >= 1.0 - 1e-6:
                    st.gain.pop(p, None)
                else:
                    st.gain[p] = round(g, 4)
                    lowered, lowest = lowered + 1, min(lowest, g)
            out[p] = {"cohort": round(cohort, 4), "lowered": lowered, "lowest": round(lowest, 4)}
        ahead = float(cfg.get("subject_ahead") or 0.0)
        for region in SUBJECT_REGIONS:  # one picture's face / hair / skin far ahead of the other pictures', by sweeps
            left = {k: 1.0 - v[region] for k, v in self.sweep_learned.items() if region in v} if ahead > 0 else {}
            median = statistics.median(left.values()) if len(left) >= MIN_PICTURES else None
            lowered, lowest = 0, 1.0
            for st in self.stats.values():
                target = 1.0
                if median and median > 0 and st.key in left:
                    want = min(SPREAD[1], max(SPREAD[0], left[st.key] / median)) / ahead
                    target = max(lo, want ** (2.0 * strength)) if want < 1.0 else 1.0
                g = _move(float(st.gain.get(region, 1.0)), target, step, lo, 1.0)
                if g >= 1.0 - 1e-6:
                    st.gain.pop(region, None)
                else:
                    st.gain[region] = round(g, 4)
                    lowered, lowest = lowered + 1, min(lowest, g)
            if ahead > 0:
                out[region] = {"lowered": lowered, "lowest": round(lowest, 4)}
        boost, holdout = float(cfg.get("boost") or 0.0), float(cfg.get("holdout") or 0.0)
        for region in REGIONS:  # a weight for the region in every picture
            target = 1.0
            learned = self.learned_ema.get(region)
            if region in SUBJECT_REGIONS and boost > 1.0 and ready and learned is not None:
                lag = (subject - learned) / subject  # boost: a subject region lagging the subject as a whole
                target = min(boost, 1.0 + strength * max(0.0, lag - BOOST_MARGIN))
            gap = self.holdout_gap.get(region)
            if holdout > 0 and gap is not None and gap > holdout and self.holdout_flat.get(region, 0) >= HOLDOUT_FLAT:
                over = min(1.0, (gap - holdout) / max(1e-6, 1.0 - holdout))  # holdout: memorised, not learned
                target = max(lo, (1.0 - over) ** (2.0 * strength))
            g = _move(float(self.class_gain.get(region, 1.0)), target, step, lo if holdout > 0 else 1.0,
                      max(1.0, boost))
            if abs(g - 1.0) <= 1e-6:
                self.class_gain.pop(region, None)
            else:
                self.class_gain[region] = round(g, 4)
        top = float(cfg.get("prior") or 0.0)
        if top > 0 and "background" in cohorts:  # full strength once the background is half its allowed error past
            target = top * min(1.0, 2.0 * (1.0 - cohorts["background"]))
            now = self.prior_weight
            now = max(target, now - step * top) if target < now else min(target, now + step * top / 2)
            self.prior_weight = round(now, 5) if now > 1e-6 else 0.0
        else:
            self.prior_weight = 0.0
        out["boost"] = {r: g for r, g in self.class_gain.items() if g > 1.0}
        out["held"] = {r: g for r, g in self.class_gain.items() if g < 1.0}
        out["prior"] = self.prior_weight
        self._flag()
        return out

    def gains(self, keys: list) -> list[dict]:
        """Per sample of a batch its steered {region: weight} (thread-safe; {} for a picture at weight 1)."""
        with self._lock:
            return [merge_gains(self.class_gain, st.gain if st is not None else None)
                    for st in (self.stats.get(str(k)) for k in keys)]

    def contrasts(self, signal: str = "steps") -> dict:
        """Per protected region: the median contrast over the pictures that have one, and how many do. signal: the
        training steps' reading, or the sweeps'."""
        out = {}
        for p in PROTECTED:
            vals = [v for v in (self._contrast(st, p, signal) for st in self.stats.values()) if v is not None]
            out[p] = {"median": round(statistics.median(vals), 4) if vals else None, "pictures": len(vals)}
        return out

    def rows(self, limit: int = 60) -> list[dict]:
        """The pictures, the one whose background runs furthest ahead of its subject first."""
        rows = []
        for st in self.stats.values():
            if not st.seen:
                continue
            bg, cl = st.contrast("background"), st.contrast("clothes")
            swept = self.sweep_contrast.get(st.key) or {}
            rows.append({"image": Path(st.key).name, "path": st.key, "seen": st.seen,
                         "background": None if bg is None else round(bg, 3),
                         "clothes": None if cl is None else round(cl, 3),
                         "sweep_background": None if swept.get("background") is None else round(swept["background"], 3),
                         "sweep_clothes": None if swept.get("clothes") is None else round(swept["clothes"], 3),
                         "weights": " / ".join(f"{r} x{st.gain[r]:.2f}" for r in REGIONS if r in st.gain),
                         "cells": " / ".join(f"{st.cells.get(c, 0):.0f}" for c in REGIONS)})
        rows.sort(key=lambda r: (r["background"] is None, r["background"] if r["background"] is not None else 0.0))
        return rows[:limit]

    def summary(self, step: int, unread: int = 0) -> dict:
        last = self.history[-1] if self.history else {}
        return {"epoch": self.epoch, "step": int(step), "frozen": self.frozen, "late": self.late,
                "baseline": len(self.buffer), "baseline_min": BASELINE_MIN, "learned": last.get("learned") or {},
                "contrast": last.get("contrast") or {}, "pictures": len(self.stats), "unread": unread,
                "steer": (last.get("steer") or {}) if self.steer else None, "steer_off": self.steer_off,
                "rows": self.rows(), "history": [{"step": r["step"], "learned": r["learned"]}
                                                for r in self.history[-200:]]}


def describe(summary: dict) -> str:
    """One line for the log and the WebUI from a summary."""
    if not summary.get("frozen"):
        return (f"collecting the baseline ({summary.get('baseline', 0)} of {summary.get('baseline_min', BASELINE_MIN)} "
                "samples)")
    learned = summary.get("learned") or {}
    if not any(v is not None for v in learned.values()):
        return "baseline set; the first reading comes after the next epoch"
    parts = [f"{c} {100 * learned[c]:.0f}%" for c in REGIONS if learned.get(c) is not None]
    text = "learned since the baseline: " + ", ".join(parts)
    bg = (summary.get("contrast") or {}).get("background") or {}
    if bg.get("median") is not None:
        text += f" | background against subject x{bg['median']:.2f} (median of {bg['pictures']} pictures)"
    steer = summary.get("steer") or {}
    for p in (*PROTECTED, *SUBJECT_REGIONS):
        if (steer.get(p) or {}).get("lowered"):
            text += f" | steering: {p} turned down on {steer[p]['lowered']} pictures (lowest x{steer[p]['lowest']:.2f})"
    if steer.get("boost"):
        text += " | raised: " + ", ".join(f"{r} x{v:.2f}" for r, v in steer["boost"].items())
    if steer.get("held"):
        text += " | turned down for the held-out gap: " + ", ".join(f"{r} x{v:.2f}" for r, v in steer["held"].items())
    if steer.get("prior"):
        text += f" | background prior {steer['prior']:.3f}"
    if summary.get("late"):
        text += " | baseline taken after a resume, so this counts from there"
    if summary.get("unread"):
        text += f" | {summary['unread']} samples had no parts map"
    return text


def read(job_dir) -> dict | None:
    """A job's saved region watch state (region_watch.json), or None."""
    try:
        return json.loads((Path(job_dir) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
