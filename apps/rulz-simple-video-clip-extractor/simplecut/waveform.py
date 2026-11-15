"""The sound under the timeline: a strip that draws the audio's peak envelope for a few seconds
around the playhead, so speech, claps and silence show when placing In and Out.

The first audio stream is decoded once in the background and reduced to RATE buckets a second,
each the loudest sample in it (all channels; one channel alone shows as loud) and its RMS. The
buckets sit on an absolute time grid in the container's clock, the same one the video frames'
times use: bucket k covers k/RATE to (k+1)/RATE seconds, placed by the audio frames' own times,
so sound that starts after the video, or has gaps, lines up. The envelope is stored per video
like scene-cut scores, so a reopened video shows it at once.
"""
from __future__ import annotations

import math
import time
from bisect import bisect_left, bisect_right
import traceback
from dataclasses import dataclass, field

import av
import numpy as np
from PySide6.QtCore import QEvent, QLineF, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import theme
from .cliplist import _file_stamp
from .npzcache import cache_dir, cache_path, clean_dir, load_npz, save_npz

WAVE_VERSION = 1
RATE = 200                 # buckets per second
BATCH_SAMPLES = 1 << 16    # decoded samples gathered before they are bucketed (fewer numpy calls)
JITTER = 0.5 / RATE        # seconds a frame may be off from where the one before ended
MAX_JUMP = 3600.0          # a jump in time further than this (length unknown) is a broken stamp
FLOOR_DB = -48.0           # the quietest level drawn; the height is on a dB scale above it
DEFAULT_SPAN = 8.0         # seconds shown around the playhead
MIN_SPAN, MAX_SPAN = 1.0, 120.0
ZOOM_STEP = 1.25           # per wheel notch
HEIGHT = 50
RULER = 13                 # the second ticks and labels along the bottom
_SCALE = {"uint8": 1 / 128, "int16": 1 / 32768, "int32": 1 / 2147483648,   # by sample type
          "int64": 1 / 9223372036854775808}


@dataclass
class Envelope:
    """Peak and RMS (0-1) per bucket; bucket i is absolute bucket `first` + i."""
    first: int
    peak: np.ndarray   # float32
    rms: np.ndarray    # float32
    _padded: tuple | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.peak)

    @property
    def start(self) -> float:
        return self.first / RATE

    @property
    def end(self) -> float:
        return (self.first + len(self.peak)) / RATE

    def columns(self, a: float, span: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Peak and RMS of n equal columns from a to a + span seconds (the loudest bucket in each,
        or the bucket under it when a column is narrower than one), and which columns have sound
        data at all."""
        if self._padded is None:   # (a 0 after the end: a column past it reduces to nothing)
            self._padded = (np.append(self.peak, np.float32(0)), np.append(self.rms, np.float32(0)))
        pk, rm = self._padded
        m = len(self.peak)
        pos = (a + np.arange(n + 1) * (span / n)) * RATE - self.first
        edges = np.floor(pos).astype(np.int64)
        # (from the unrounded ends: columns narrower than a bucket inside the first one have sound too)
        inside = (pos[1:] > 0) & (pos[:-1] < m)
        idx = np.clip(edges, 0, m)
        if m == 0:
            z = np.zeros(n, dtype=np.float32)
            return z, z, inside
        lo, hi = idx[0], idx[-1] + 1   # (only that stretch: the last index reduces to the array's end)
        p = np.maximum.reduceat(pk[lo:hi], idx - lo)[:n]
        r = np.maximum.reduceat(rm[lo:hi], idx - lo)[:n]
        p[~inside] = 0
        r[~inside] = 0
        return p, r, inside


def _empty() -> Envelope:
    return Envelope(0, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))


# ------------------------------------------------------------------ reading
class Cancelled(Exception):
    pass


class _Buckets:
    """Grows to whatever absolute buckets the frames land on, before or after those so far."""

    def __init__(self, capacity: int):
        self.lo: int | None = None
        self.n = 0
        self.peak = np.zeros(max(capacity, 1024), dtype=np.float32)
        self.sq = np.zeros_like(self.peak, dtype=np.float64)   # sum of squares, channels averaged
        self.cnt = np.zeros_like(self.peak, dtype=np.int64)    # samples in it

    def _fit(self, a: int, b: int):
        if self.lo is None:
            self.lo = a
        lo = min(self.lo, a)
        hi = max(self.lo + self.n, b)
        if lo == self.lo and hi - lo <= len(self.peak):
            self.n = hi - lo
            return
        cap = len(self.peak)
        while cap < hi - lo:
            cap *= 2
        shift = self.lo - lo
        for name in ("peak", "sq", "cnt"):
            old = getattr(self, name)
            new = np.zeros(cap, dtype=old.dtype)
            new[shift:shift + self.n] = old[:self.n]
            setattr(self, name, new)
        self.lo, self.n = lo, hi - lo

    def add(self, k: np.ndarray, peak: np.ndarray, sq: np.ndarray, cnt: np.ndarray):
        """Absolute buckets k (ascending, each once) get these peaks, sums of squares and counts
        on top of what earlier frames put there (overlapping frames: the louder counts)."""
        self._fit(int(k[0]), int(k[-1]) + 1)
        i = k - self.lo
        self.peak[i] = np.maximum(self.peak[i], peak)
        self.sq[i] += sq
        self.cnt[i] += cnt

    def envelope(self) -> Envelope:
        if self.lo is None:
            return _empty()
        n = self.n
        peak = np.minimum(self.peak[:n], 1.0)
        cnt = self.cnt[:n]
        rms = np.sqrt(np.divide(self.sq[:n], cnt, out=np.zeros(n), where=cnt > 0))
        # (rounded the way they are stored, so a fresh read and a reopened video draw the same)
        return Envelope(self.lo, _stored(peak), _stored(np.minimum(rms, 1.0)))


def _stored(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16).astype(np.float32)


def _frames(container, stream):
    """The stream's frames; a packet that can't be decoded (a damaged spot) is skipped instead of
    ending the read."""
    for packet in container.demux(stream):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError:
            continue
        yield from frames


def _samples(frame) -> np.ndarray:
    """The frame's samples as (channels, n) in their own type. A packed format comes as one row
    with the channels interleaved."""
    x = frame.to_ndarray()
    if x.shape[0] == 1:
        ch = x.shape[1] // frame.samples
        if ch > 1:
            x = x.reshape(-1, ch).T
    return x


def read_envelope(path: str, progress=None, cancelled=None) -> Envelope | None:
    """Decode the first audio stream and reduce it to the envelope; None when there is none.
    progress(fraction) is called now and then; cancelled() returning True stops with Cancelled."""
    with av.open(path) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        tb = stream.time_base
        t0 = float(stream.start_time * tb) if stream.start_time is not None and tb else 0.0
        dur = float(stream.duration * tb) if stream.duration and tb else 0.0
        if not dur and container.duration:
            dur = container.duration / 1_000_000
        return envelope_of(_frames(container, stream), t0, dur, progress, cancelled)


def envelope_of(frames, t0: float = 0.0, dur: float = 0.0, progress=None, cancelled=None) -> Envelope:
    """The envelope of decoded audio frames. Frames that follow on from each other (within
    JITTER: container times are often rounded to the millisecond) are taken as one run of
    samples, timed from the run's first frame; a frame further off starts a new run at its own
    time, and one without a time follows the one before (the first such at t0). `dur` (seconds
    from t0, 0 when unknown) is for the progress, and bounds the times believed: a frame stamped
    far outside the stream (an MPEG-TS clock wrap, a broken join) also follows the one before,
    instead of stretching the envelope over hours of nothing."""
    buckets = _Buckets(int(dur * RATE * 1.02) + RATE if dur else 1 << 16)
    parts: list[np.ndarray] = []   # the run's samples, (channels, n) each
    run_t = run_rate = run_kind = None
    run_n = 0
    expected = None                # where the run ends: the next frame's time if it follows on
    last_report = 0.0
    for frame in frames:
        if cancelled is not None and cancelled():
            raise Cancelled
        n, sr = frame.samples, frame.sample_rate
        if not n or not sr:
            continue
        x = _samples(frame)
        kind = (x.dtype, x.shape[0])
        t = frame.time
        if t is not None and (not (t0 - 1.0 <= t <= t0 + dur + 60.0) if dur
                              else expected is not None and abs(t - expected) > MAX_JUMP):
            t = None
        if parts and kind == run_kind and sr == run_rate and (t is None or abs(t - expected) <= JITTER):
            start = None if run_n < BATCH_SAMPLES else expected   # None: the run goes on
        else:
            start = t if t is not None else (expected if expected is not None else t0)
        if start is not None:
            if parts:
                _bucket(buckets, run_t, run_rate, parts)
            run_t, run_rate, run_kind, parts, run_n = start, sr, kind, [], 0
        parts.append(x)
        run_n += n
        expected = run_t + run_n / run_rate
        if progress is not None and dur > 0:
            now = time.monotonic()
            if now - last_report > 0.25:
                last_report = now
                progress(min(max((expected - t0) / dur, 0.0), 1.0))
    if parts:
        _bucket(buckets, run_t, run_rate, parts)
    return buckets.envelope()


def _bucket(buckets: _Buckets, t0: float, sr: int, parts: list):
    """Put a run of samples starting at t0 seconds in their buckets: sample i is at t0 + i/sr.
    The peaks come from the samples in their own type; only the squares need floats."""
    x = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
    if x.dtype.kind == "f" and not np.isfinite(x).all():   # (damaged float audio: NaN would blank columns)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    ch, n = x.shape
    k = np.arange(math.floor(t0 * RATE), math.floor((t0 + (n - 1) / sr) * RATE) + 1)
    # (a hair under: a sample exactly on a bucket's start must not slip into the next one by rounding)
    b = np.clip(np.ceil((k / RATE - t0) * sr - 1e-6).astype(np.int64), 0, n - 1)
    keep = np.diff(b, prepend=-1) > 0   # (each bucket's first sample; none for a bucket without one)
    k, b = k[keep], b[keep]
    hi = np.maximum.reduceat(x, b, axis=1).astype(np.float32)
    lo = np.minimum.reduceat(x, b, axis=1).astype(np.float32)
    xf = x.astype(np.float32)
    if x.dtype == np.uint8:
        hi -= 128
        lo -= 128
        xf -= 128
    scale = _SCALE.get(x.dtype.name, 1.0)
    peak = np.maximum(hi, -lo).max(axis=0) * scale
    np.square(xf, out=xf)
    sq = np.add.reduceat(xf, b, axis=1).sum(axis=0, dtype=np.float64) * (scale * scale / ch)
    buckets.add(k, peak, sq, np.diff(np.append(b, n)))


# ------------------------------------------------------------------ storage
def waves_dir() -> str:
    return cache_dir("waveforms")


def waves_path(video: str) -> str:
    return cache_path(waves_dir(), video)


def save_envelope(video: str, env: Envelope, stamp: tuple[int, float] | None = None) -> bool:
    """Keep a read's envelope for the next time the video is opened (tied to the file's size and
    date). A file that changed since `stamp` (when the read began) is not stored."""
    return save_npz(waves_path(video), video, WAVE_VERSION,
                    {"first": np.array(env.first, dtype=np.int64), "peak": env.peak.astype(np.float16),
                     "rms": env.rms.astype(np.float16)}, stamp)


def load_envelope(video: str) -> Envelope | None:
    """The stored envelope of this video, or None (never read, another file of that name since,
    another version, or unreadable)."""
    z = load_npz(waves_path(video), video, WAVE_VERSION, ("first", "peak", "rms"))
    if z is None:
        return None
    try:
        first = int(z["first"])
        peak = z["peak"].astype(np.float32)
        rms = z["rms"].astype(np.float32)
    except (TypeError, ValueError):
        return None
    if not (peak.ndim == rms.ndim == 1 and len(peak) == len(rms)):
        return None
    if not (np.isfinite(peak).all() and np.isfinite(rms).all()):
        return None
    return Envelope(first, np.clip(peak, 0, 1), np.clip(rms, 0, 1))


def clean_envelopes(now: float | None = None):
    """Delete stored envelopes of videos gone for a long time, and reads cut off while written."""
    clean_dir(waves_dir(), now)


class WaveformReadThread(QThread):
    """Runs read_envelope() for one video in the background."""
    progress = Signal(float, int)     # fraction done, token
    done = Signal(object, int)        # Envelope or None (no audio stream), token
    failed = Signal(str, int)         # message, token

    def __init__(self, path: str, token: int):
        super().__init__()
        self.path = path
        self.token = token   # this read's own number; the app drops what a read it let go reports
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        stamp = _file_stamp(self.path)
        try:
            env = read_envelope(self.path, lambda f: self.progress.emit(f, self.token), lambda: self._cancelled)
        except Cancelled:
            return
        except Exception as e:  # noqa: BLE001
            if not self._cancelled:
                self.failed.emit(f"{str(e) or type(e).__name__}\n{traceback.format_exc()}", self.token)
            return
        if self._cancelled:
            return
        if env is not None:
            save_envelope(self.path, env, stamp)
        self.done.emit(env, self.token)


# ------------------------------------------------------------------ the strip
# EMPTY: no video. IDLE: a video, its sound not read (the strip was hidden, or the read stopped).
EMPTY, IDLE, NO_SOUND, READING, FAILED, READY = "empty", "idle", "nosound", "reading", "failed", "ready"
TICK_STEPS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300)


def level(v: np.ndarray) -> np.ndarray:
    """Drawn height 0-1 of a level 0-1: dB above FLOOR_DB, so quiet word endings still show next
    to a loud clap."""
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(v, 1e-9))
    return np.clip(1.0 - db / FLOOR_DB, 0.0, 1.0).astype(np.float32)


def _argb(c) -> np.uint32:
    return np.uint32(c.rgba())


class WaveformStrip(QWidget):
    """Shows `span` seconds around the timeline's playhead, with the timeline's In/Out, clip box
    and scene cuts. It reads all of those from the timeline when it paints and repaints whenever
    the timeline does, so the two never disagree. Click or drag to seek, wheel to zoom."""
    seekRequested = Signal(float)
    scrubFinished = Signal()
    spanChanged = Signal(float)

    def __init__(self, timeline, span: float = DEFAULT_SPAN, parent=None):
        super().__init__(parent)
        self.timeline = timeline
        self._pad = timeline._pad
        self.span = min(max(float(span), MIN_SPAN), MAX_SPAN)
        self.state = EMPTY
        self.message = ""
        self.envelope: Envelope | None = None
        self._frozen: tuple[float, float] | None = None   # the window while the mouse drags in it
        self._press = False
        self._dragging = False
        self._image = None
        self._image_key = None
        self.setFixedHeight(HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(10)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setToolTip("The sound around the playhead. Click or drag to seek; the mouse wheel zooms.")
        self.setEnabled(False)
        timeline.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is self.timeline and ev.type() == QEvent.Paint and self.isVisible():
            self.update()
        return False

    # --- state ------------------------------------------------------------
    def set_state(self, state: str, message: str = ""):
        self._drop_drag()   # (another video, none, or hidden)
        self.state, self.message = state, message
        if state != READY:
            self.envelope = None
            self._image_key = None
        self.setEnabled(state != EMPTY)
        self.update()

    def set_envelope(self, env: Envelope):
        self.envelope = env
        self._image_key = None
        if len(env):
            self.state, self.message = READY, ""
        else:
            self.state, self.message = NO_SOUND, "No sound in this video"
        self.setEnabled(True)
        self.update()

    def set_progress(self, f: float):
        self.state = READING
        self.message = f"Reading the sound… {int(f * 100)} %"
        self.update()

    def set_span(self, span: float):
        span = min(max(float(span), MIN_SPAN), MAX_SPAN)
        if span != self.span:
            self.span = span
            self.update()
            self.spanChanged.emit(span)

    # --- geometry ---------------------------------------------------------
    def _columns(self) -> int:
        return max(int(round((self.width() - 2 * self._pad) * self.devicePixelRatioF())), 1)

    def view_span(self) -> tuple[float, float]:
        """(first second shown, seconds shown): the playhead in the middle unless that would show
        time before the start or after the end. While it slides, its start is a whole number of
        columns, so the drawing does not shimmer as playback moves it along."""
        if self._frozen is not None:
            return self._frozen
        tl = self.timeline
        length = max(tl.end - tl.start, 1e-6)
        span = min(self.span, length)
        a = tl.position - span / 2
        if a <= tl.start or a >= tl.end - span:
            return (tl.start if a <= tl.start else tl.end - span), span
        step = span / self._columns()
        return max(math.floor(a / step + 1e-6) * step, tl.start), span

    def _x(self, t: float, a: float, span: float) -> float:
        return self._pad + (t - a) / span * (self.width() - 2 * self._pad)

    def time_at(self, x: float) -> float:
        a, span = self.view_span()
        w = max(self.width() - 2 * self._pad, 1)
        t = a + (x - self._pad) / w * span
        tl = self.timeline
        return min(max(t, max(a, tl.start)), min(a + span, tl.end))

    @property
    def scrubbing(self) -> bool:
        return self._press and self._dragging

    def _drop_drag(self):
        """End a drag the mouse can't end: a hidden or disabled widget gets no mouse release, and
        the app would go on seeking the quick, approximate way and keep the frozen window."""
        dragged = self.scrubbing
        self._press, self._dragging, self._frozen = False, False, None
        if dragged:
            self.scrubFinished.emit()

    def hideEvent(self, e):
        self._drop_drag()
        super().hideEvent(e)

    def changeEvent(self, e):
        if e.type() == QEvent.EnabledChange and not self.isEnabled():
            self._drop_drag()
        super().changeEvent(e)

    # --- events -----------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton or self.state == EMPTY:
            return
        self._frozen = self.view_span()   # (else each seek recentres it under the mouse and it runs away)
        self._press, self._dragging = True, False
        self.seekRequested.emit(self.time_at(e.position().x()))

    def mouseMoveEvent(self, e):
        if not (e.buttons() & Qt.LeftButton) or not self._press:
            return
        self._dragging = True
        self.seekRequested.emit(self.time_at(e.position().x()))

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or not self._press:
            return
        dragged = self._dragging
        self._press, self._dragging, self._frozen = False, False, None
        self.update()
        if dragged:
            self.scrubFinished.emit()

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120
        if not steps or self.state == EMPTY:
            e.ignore()
            return
        self.set_span(self.span * ZOOM_STEP ** -steps)
        e.accept()

    # --- painting ---------------------------------------------------------
    def _wave_image(self, a: float, span: float, h: int) -> QImage | None:
        """The envelope from a to a + span as an image one device pixel per column, rebuilt only
        when the window, size, data or theme change."""
        env = self.envelope
        if env is None:
            return None
        n = self._columns()
        dpr = self.devicePixelRatioF()
        hd = max(int(round(h * dpr)), 3)
        key = (a, span, n, hd, id(env), theme.mode())
        if key == self._image_key:
            return self._image
        p, r, inside = env.columns(a, span, n)
        mid = (hd - 1) / 2
        half = (hd - 2) / 2
        rows = np.arange(hd, dtype=np.float32)[:, None]
        dist = np.abs(rows - mid)
        img = np.zeros((hd, n), dtype=np.uint32)
        img[(dist <= np.maximum(level(p) * half, 0.5)) & inside] = _argb(theme.Q_WAVE_PEAK)
        img[(dist <= level(r) * half) & inside & (r > 0)] = _argb(theme.Q_WAVE_RMS)
        q = QImage(img.data, n, hd, n * 4, QImage.Format_ARGB32).copy()
        q.setDevicePixelRatio(dpr)
        self._image, self._image_key = q, key
        return q

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        left, right = self._pad, w - self._pad
        wave_h = h - RULER
        p.fillRect(QRectF(left, 0, right - left, h), theme.Q_WAVE_BG)
        if self.state == EMPTY:
            p.end()
            return
        tl = self.timeline
        a, span = self.view_span()

        def x(t):
            return self._x(t, a, span)

        def clip(xa, xb):
            return max(xa, left), min(xb, right)

        # the In/Out region or the clip box, under the sound
        box = None
        if tl.selection:
            box = tl.selection
            xa, xb = clip(x(box[0]), x(box[1]))
            if xb > xa:
                p.fillRect(QRectF(xa, 0, xb - xa, wave_h), theme.Q_TL_SEL)
        elif tl.mark_in is not None or tl.mark_out is not None:
            ia = tl.mark_in if tl.mark_in is not None else tl.start
            ib = tl.mark_out if tl.mark_out is not None else tl.end
            xa, xb = clip(x(ia), x(ib))
            if xb > xa:
                p.fillRect(QRectF(xa, 0, xb - xa, wave_h), theme.Q_TL_INOUT)

        img = self._wave_image(a, span, wave_h) if self.state == READY else None
        if img is not None:
            p.drawImage(QPointF(left, 0), img)
        else:
            p.setPen(theme.Q_MUTED)
            p.drawText(QRectF(left, 0, right - left, wave_h), Qt.AlignCenter, self.message)

        def line(t, colour, width=1.0, top=0.0, bottom=wave_h):
            xx = x(t)
            if left - 1 <= xx <= right + 1:
                p.setPen(QPen(colour, width))
                p.drawLine(QLineF(xx, top, xx, bottom))

        if tl.scene_cuts:
            cuts = tl.scene_cuts
            for c in cuts[bisect_left(cuts, a):bisect_right(cuts, a + span)]:
                line(c, theme.Q_TL_SCENE, 2)
        if box is not None:
            xa, xb = x(box[0]), x(box[1])
            p.setPen(QPen(theme.Q_RED_HOVER, 2))
            p.setBrush(Qt.NoBrush)
            p.setClipRect(QRectF(left, 0, right - left, h))
            p.drawRect(QRectF(xa, 1, xb - xa, wave_h - 2))
            p.setClipping(False)
        else:
            if tl.mark_in is not None:
                line(tl.mark_in, theme.Q_TL_MARK, 2)
            if tl.mark_out is not None:
                line(tl.mark_out, theme.Q_RED_HOVER, 2)

        self._paint_ruler(p, a, span, left, right, wave_h, h, x)
        line(tl.position, theme.Q_TL_MARK, 2, 0, h)
        p.end()

    def _paint_ruler(self, p, a, span, left, right, top, bottom, x):
        """Second ticks along the bottom, labelled in the time the app shows (from the video's
        start), about one label per 70 px."""
        width = max(right - left, 1)
        step = next((s for s in TICK_STEPS if s / span * width >= 70), TICK_STEPS[-1])
        origin = self.timeline.start
        p.setPen(QPen(theme.Q_TL_BAR_BORDER, 1))
        p.drawLine(QLineF(left, top + 0.5, right, top + 0.5))
        f = QFont(p.font())
        f.setPointSizeF(max(f.pointSizeF() * 0.8, 6.0))
        p.setFont(f)
        k0 = math.ceil((a - origin) / step - 1e-6)
        k1 = math.floor((a + span - origin) / step + 1e-6)
        if k1 - k0 > 200:
            return
        small = step / 5
        p.setPen(QPen(theme.Q_TL_TICK, 1))
        for k in range(k0 - 1, k1 + 1):
            for j in range(1, 5):
                xx = x(origin + k * step + j * small)
                if left <= xx <= right:
                    p.drawLine(QLineF(xx, top, xx, top + 2))
        for k in range(k0, k1 + 1):
            t = k * step
            xx = x(origin + t)
            if not left <= xx <= right:
                continue
            p.setPen(QPen(theme.Q_TL_TICK, 1))
            p.drawLine(QLineF(xx, top, xx, top + 5))
            p.setPen(theme.Q_MUTED)
            p.drawText(QPointF(xx + 3, bottom - 2), _label(t, step))


def _label(t: float, step: float) -> str:
    """m:ss, with as many decimals as the tick step has; h:mm:ss past an hour."""
    neg = t < -1e-9
    t = abs(t)
    dec = 0
    while dec < 3 and abs(step * 10 ** dec - round(step * 10 ** dec)) > 1e-6:
        dec += 1
    t = round(t, dec)
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    sec = f"{s:0{3 + dec if dec else 2}.{dec}f}"
    text = f"{h}:{m:02d}:{sec}" if h else f"{m}:{sec}"
    return ("-" if neg else "") + text
