"""Scene-cut detection: find the frames where the picture changes at once (a hard cut).

The whole video is decoded once in the background, each frame shrunk to a small thumbnail, and
every frame gets a score from 0 to 100 for how different it is from the frame before it. The
scores are kept (and stored per video), so a different sensitivity only picks the cuts again
from them and never decodes anything.

A frame's score is ffmpeg's scene-detection measure (the scdet filter): the mean absolute
difference of the thumbnail from the previous one, less the typical difference of the frames
just before (so steady camera motion, where every frame differs, does not count), never more
than the difference itself. A cut also has to last: the frame a few frames later must still
differ from the frame before the cut, so a camera flash or one bad frame is not taken for one.

Slow fades and dissolves have no single frame where the picture changes at once and are
usually not found.
"""
from __future__ import annotations

import time
import traceback
from array import array
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass

import av
import numpy as np
from PySide6.QtCore import QThread, Signal

from .cliplist import STALE_DAYS, _file_stamp  # noqa: F401  (STALE_DAYS: how long scans of gone videos stay)
from .npzcache import cache_dir, cache_path, clean_dir, load_npz, save_npz

SCORE_VERSION = 1
THUMB_WIDTH = 96            # thumbnail width in pixels; the height follows the picture's shape
MOTION_FRAMES = 5           # how many earlier frame differences make up the "typical" difference
SETTLE_FRAMES = 2           # a cut must still show this many frames after it
LOOKBACK_FRAMES = 3         # ... against each of this many frames before it (a flash of up to 2 frames)
MIN_SCENE = 0.5             # seconds: of two cuts closer than this, only the stronger one is kept
SENSITIVITY = {"low": 16.0, "normal": 10.0, "high": 6.0}   # the score a cut needs
SENSITIVITY_LABELS = {"low": "Low", "normal": "Normal", "high": "High"}
DEFAULT_SENSITIVITY = "normal"


@dataclass
class SceneScores:
    """One score per frame of the video, as the scan found them."""
    times: np.ndarray    # float64 frame times in seconds (the decoder's pts), ascending
    score: np.ndarray    # float32 change from the frame before, 0-100 (motion taken out)
    settle: np.ndarray   # float32 least difference of a frame SETTLE_FRAMES later from the frames before

    def __len__(self) -> int:
        return len(self.times)

    def cuts(self, sensitivity: str = DEFAULT_SENSITIVITY) -> list[float]:
        return pick_cuts(self, SENSITIVITY.get(sensitivity, SENSITIVITY[DEFAULT_SENSITIVITY]))


def pick_cuts(s: SceneScores, threshold: float, min_scene: float = MIN_SCENE) -> list[float]:
    """The times of the frames that start a new scene: a score of at least `threshold`, a
    picture that still differs SETTLE_FRAMES later, and at least `min_scene` seconds from every
    other cut (the strongest are taken first) and from the first and last frame (a "scene" of a
    few frames there is a fade, a title or a broken last frame)."""
    if not len(s.times):
        return []
    first, last = float(s.times[0]), float(s.times[-1])
    cand = np.nonzero((s.score >= threshold) & (s.settle >= threshold * 0.5))[0]
    cand = [i for i in cand if s.times[i] - first >= min_scene and last - s.times[i] >= min_scene]
    kept: list[float] = []   # sorted
    for i in sorted(cand, key=lambda i: (-float(s.score[i]), float(s.times[i]))):
        t = float(s.times[i])
        j = bisect_left(kept, t)
        if (j > 0 and t - kept[j - 1] < min_scene) or (j < len(kept) and kept[j] - t < min_scene):
            continue
        kept.insert(j, t)
    return kept


def scenes_between(cuts: list[float], start: float, end: float) -> list[tuple[float, float]]:
    """The scenes from start to end: (first frame, end) pairs, end exclusive."""
    bounds = [start] + [c for c in cuts if start < c < end] + [end]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


# ------------------------------------------------------------------ the scan
def _thumb_size(width: int, height: int) -> tuple[int, int]:
    w = THUMB_WIDTH
    h = max(8, int(round(w * height / max(width, 1) / 2)) * 2)
    return w, h


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference of two thumbnails, 0-100."""
    return float(np.abs(a.astype(np.int16) - b).mean()) * (100.0 / 255.0)


class Cancelled(Exception):
    pass


def scan(path: str, progress=None, cancelled=None) -> SceneScores:
    """Decode the video once and score every frame. progress(fraction) is called now and then;
    cancelled() returning True stops the scan with Cancelled."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        cc = stream.codec_context
        tw, th = _thumb_size(cc.width or 16, cc.height or 9)
        tb = stream.time_base
        t0 = float(stream.start_time * tb) if stream.start_time is not None and tb else 0.0
        dur = float(stream.duration * tb) if stream.duration and tb else 0.0
        if not dur and container.duration:
            dur = container.duration / 1_000_000
        times, mafd = array("d"), array("f")
        recent: deque[np.ndarray] = deque(maxlen=SETTLE_FRAMES + LOOKBACK_FRAMES)   # the last thumbnails
        settle = array("f")
        last_report = 0.0
        for frame in _frames(container, stream):
            if frame.time is None:
                continue
            if cancelled is not None and cancelled():
                raise Cancelled
            thumb = frame.reformat(width=tw, height=th, format="rgb24", interpolation="AREA").to_ndarray()
            times.append(frame.time)
            mafd.append(_diff(thumb, recent[-1]) if recent else 0.0)
            # The frame SETTLE_FRAMES back gets its settle value now: how much this frame still
            # differs from the frames before that one, the least of up to LOOKBACK_FRAMES of them
            # (the frame after a one-frame flash differs from the flash, not from those).
            if len(recent) > SETTLE_FRAMES:
                before = list(recent)[:len(recent) - SETTLE_FRAMES][-LOOKBACK_FRAMES:]
                settle.append(min(_diff(thumb, b) for b in before))
            recent.append(thumb)
            if progress is not None and dur > 0:
                now = time.monotonic()
                if now - last_report > 0.25:
                    last_report = now
                    progress(min(max((frame.time - t0) / dur, 0.0), 1.0))
    n = len(times)
    # The last frames have no frame SETTLE_FRAMES later: their own difference stands in.
    settle_arr = np.zeros(n, dtype=np.float32)
    k = len(settle)
    settle_arr[1:1 + k] = np.frombuffer(settle, dtype=np.float32)
    m = np.frombuffer(mafd, dtype=np.float32).copy()
    settle_arr[1 + k:] = m[1 + k:]
    # (rounded the way they are stored, so a fresh scan and a reopened video pick the same cuts)
    return SceneScores(np.frombuffer(times, dtype=np.float64).copy(), _stored(motion_score(m)), _stored(settle_arr))


def _stored(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16).astype(np.float32)


def _frames(container, stream):
    """The stream's frames; a packet that can't be decoded (a damaged spot) is skipped instead of
    ending the scan."""
    for packet in container.demux(stream):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError:
            continue
        yield from frames


def motion_score(mafd: np.ndarray) -> np.ndarray:
    """scdet's score: each difference less the typical (median) of the MOTION_FRAMES before it,
    at least 0 and at most the difference itself."""
    m = np.asarray(mafd, dtype=np.float32)
    n = len(m)
    out = np.zeros(n, dtype=np.float32)
    if n < 2:
        return out
    # Frame i (from 1; frame 0 has no difference) looks back at m[i-MOTION_FRAMES .. i-1], never
    # before frame 1. With MOTION_FRAMES NaNs in front of m[1:], that is window i-1.
    padded = np.concatenate([np.full(MOTION_FRAMES, np.nan, dtype=np.float32), m[1:]])
    windows = np.lib.stride_tricks.sliding_window_view(padded, MOTION_FRAMES)[:n - 1]
    typical = np.zeros(n - 1, dtype=np.float32)
    if n > 2:
        typical[1:] = np.nanmedian(windows[1:], axis=1)   # (frame 1 has nothing before it: 0)
    cur = m[1:]
    out[1:] = np.minimum(cur, np.maximum(cur - typical, 0.0))
    return out


# ------------------------------------------------------------------ storage
def scores_dir() -> str:
    return cache_dir("scene-cuts")


def scores_path(video: str) -> str:
    return cache_path(scores_dir(), video)


def save_scores(video: str, s: SceneScores, stamp: tuple[int, float] | None = None) -> bool:
    """Keep a scan's scores for the next time the video is opened (tied to the file's size and
    date, like its clip list). `stamp` is the file's size and date when the scan began: a file
    that changed meanwhile (still being written or copied) is not stored. False when nothing
    was written."""
    return save_npz(scores_path(video), video, SCORE_VERSION,
                    {"times": s.times, "score": s.score.astype(np.float16), "settle": s.settle.astype(np.float16)},
                    stamp)


def load_scores(video: str) -> SceneScores | None:
    """The stored scores of this video, or None (never scanned, another file of that name since,
    another scan version, or unreadable)."""
    z = load_npz(scores_path(video), video, SCORE_VERSION, ("times", "score", "settle"))
    if z is None:
        return None
    try:
        times = z["times"].astype(np.float64)
        score = z["score"].astype(np.float32)
        settle = z["settle"].astype(np.float32)
    except (TypeError, ValueError):
        return None
    if not (times.ndim == score.ndim == settle.ndim == 1 and len(times) == len(score) == len(settle)):
        return None
    if not (np.isfinite(times).all() and np.isfinite(score).all() and np.isfinite(settle).all()):
        return None
    return SceneScores(times, score, settle)


def clean_scores(now: float | None = None):
    """Delete stored scores whose video no longer exists and that are older than STALE_DAYS,
    and scans that were cut off while being written."""
    clean_dir(scores_dir(), now)


class SceneScanThread(QThread):
    """Runs scan() for one video in the background."""
    progress = Signal(float, int)     # fraction done, token
    done = Signal(object, int)        # SceneScores, token
    failed = Signal(str, int)         # message, token

    def __init__(self, path: str, token: int):
        super().__init__()
        self.path = path
        self.token = token   # this scan's own number; the app drops what a scan it let go reports
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        stamp = _file_stamp(self.path)
        try:
            s = scan(self.path, lambda f: self.progress.emit(f, self.token), lambda: self._cancelled)
        except Cancelled:
            return
        except Exception as e:  # noqa: BLE001
            if not self._cancelled:
                self.failed.emit(f"{e}\n{traceback.format_exc()}", self.token)
            return
        if self._cancelled:
            return
        save_scores(self.path, s, stamp)   # (before reporting: the result is the same either way)
        self.done.emit(s, self.token)
