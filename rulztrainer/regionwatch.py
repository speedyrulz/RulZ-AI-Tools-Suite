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

It only observes: the loss is read before any region weighting, and nothing about training changes. One training
sample per picture per epoch at a random noise level is a noisy reading, so the per-picture contrast is a smoothed
trend, not a measurement, and a region of fewer than MIN_CELLS cells is left out.

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
KERNEL = 1.5  # buckets: the curves are smoothed over their neighbours
BASELINE_MIN = 64  # samples before the baseline freezes (whole epochs: a small set takes several)
START_OBS = 3  # a picture's first readings set its own starting contrast
BETA = 0.8  # the per-picture contrast's smoothing (about 5 epochs)
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

    def contrast(self, region: str) -> float | None:
        v = self.contrast_log.get(region)
        return None if v is None else math.exp(v)


class Curve:
    """A mean per noise-level bucket, read smoothed over the neighbouring buckets."""

    def __init__(self, sums: list[float] | None = None, counts: list[float] | None = None):
        self.sums = list(sums) if sums else [0.0] * BUCKETS
        self.counts = list(counts) if counts else [0.0] * BUCKETS

    def add(self, b: int, value: float) -> None:
        self.sums[b] += value
        self.counts[b] += 1.0

    def at(self, b: int) -> float | None:
        num = den = 0.0
        for k in range(BUCKETS):
            if self.counts[k] <= 0:
                continue
            w = math.exp(-((k - b) ** 2) / (2 * KERNEL ** 2))
            num += w * self.sums[k]
            den += w * self.counts[k]
        return (num / den) if den > 0 else None


def bucket(sigma: float) -> int:
    return min(BUCKETS - 1, max(0, int(float(sigma) * BUCKETS)))


class RegionWatch:
    def __init__(self, items, job_dir, resume: bool = False):
        self.job_dir = Path(job_dir)
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
            for key, rec in (data.get("pictures") or {}).items():
                if key in self.stats:
                    self.stats[key] = PictureStat(key, int(rec.get("seen", 0)), dict(rec.get("start_n") or {}),
                                                  dict(rec.get("start_log") or {}), dict(rec.get("contrast_log") or {}),
                                                  dict(rec.get("cells") or {}))
            self.epoch, self.late = int(data.get("epoch", 0)), bool(data.get("late"))
            self.history = list(data.get("history") or [])
        except (KeyError, TypeError, ValueError):
            return False
        self.frozen = True
        return True

    def _save(self) -> None:
        data = {"epoch": self.epoch, "frozen": self.frozen, "late": self.late, "history": self.history[-400:],
                "loss_curves": {c: {"sums": v.sums, "counts": v.counts} for c, v in self.loss_curves.items()},
                "ratio_curves": {p: {"sums": v.sums, "counts": v.counts} for p, v in self.ratio_curves.items()},
                "pictures": {k: {"seen": st.seen, "start_n": st.start_n, "start_log": st.start_log,
                                 "contrast_log": st.contrast_log, "cells": st.cells} for k, st in self.stats.items()}}
        try:
            (self.job_dir / STATE_FILE).write_text(json.dumps(data), encoding="utf-8")
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
        b = bucket(sigma)
        loss = {c: row[j] for j, c in enumerate(COLUMNS) if size[j] >= MIN_CELLS}
        for c, v in loss.items():
            base = self.loss_curves[c].at(b)
            if base and base > 0:
                self.epoch_sum[c] += v / base
                self.epoch_n[c] += 1
        subject = loss.get(SUBJECT)
        if not subject or subject <= 0:
            return
        for p in PROTECTED:
            base = self.ratio_curves[p].at(b)
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
            b = bucket(sigma)
            for j, c in enumerate(COLUMNS):
                if size[j] >= MIN_CELLS:
                    self.loss_curves[c].add(b, row[j])
            subject = row[COLUMNS.index(SUBJECT)]
            if size[COLUMNS.index(SUBJECT)] >= MIN_CELLS and subject > 0:
                for p in PROTECTED:
                    j = COLUMNS.index(p)
                    if size[j] >= MIN_CELLS and row[j] > 0:
                        self.ratio_curves[p].add(b, math.log(row[j] / subject))
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
            row = {"epoch": self.epoch, "step": int(step), "learned": learned, "contrast": self.contrasts()}
            self.history.append(row)
            try:
                with open(self.job_dir / LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except OSError:
                pass
            self._save()
            return self.summary(step, unread=unread)

    def contrasts(self) -> dict:
        """Per protected region: the median contrast over the pictures that have one, and how many do."""
        out = {}
        for p in PROTECTED:
            vals = [v for v in (st.contrast(p) for st in self.stats.values()) if v is not None]
            out[p] = {"median": round(statistics.median(vals), 4) if vals else None, "pictures": len(vals)}
        return out

    def rows(self, limit: int = 60) -> list[dict]:
        """The pictures, the one whose background runs furthest ahead of its subject first."""
        rows = []
        for st in self.stats.values():
            if not st.seen:
                continue
            bg, cl = st.contrast("background"), st.contrast("clothes")
            rows.append({"image": Path(st.key).name, "path": st.key, "seen": st.seen,
                         "background": None if bg is None else round(bg, 3),
                         "clothes": None if cl is None else round(cl, 3),
                         "cells": " / ".join(f"{st.cells.get(c, 0):.0f}" for c in REGIONS)})
        rows.sort(key=lambda r: (r["background"] is None, r["background"] if r["background"] is not None else 0.0))
        return rows[:limit]

    def summary(self, step: int, unread: int = 0) -> dict:
        last = self.history[-1] if self.history else {}
        return {"epoch": self.epoch, "step": int(step), "frozen": self.frozen, "late": self.late,
                "baseline": len(self.buffer), "baseline_min": BASELINE_MIN, "learned": last.get("learned") or {},
                "contrast": last.get("contrast") or {}, "pictures": len(self.stats), "unread": unread,
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
