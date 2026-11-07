"""Frame-accurate decoder built on PyAV.

All positions are pts in seconds. Stepping forward decodes the next frame;
stepping back re-decodes from the previous keyframe (cheap enough for a GOP).

The frames just before the current one are kept (a small, memory-bounded
cache), so stepping back after stepping or seeking forward doesn't decode
anything; and the frame a backward walk ran into is kept as a lookahead, so
the next step forward doesn't seek and decode the GOP a second time.

A container seek can land *after* the requested time: MPEG-TS lands about a
GOP late, and open-GOP HEVC lands on the next CRA keyframe when the target is
one of the leading pictures just before it. Every seek checks the first frame
it decodes and backs off further when that frame is too late.

seek() also avoids the container seek when it can: a target inside the cached
frames is served from the cache, and a target ahead of the current frame with
no keyframe in between is reached by decoding forward (always cheaper than a
seek, which would have to decode from that same keyframe anyway). Scrubbing
and dragging the fixed-length box mostly move a few frames at a time. For the
moves that still need a seek, scrub() shows the keyframe before the target
when exact seeks are slow on this video (4K); the app asks for the exact frame
when the mouse is released.
"""
from __future__ import annotations

import time
from bisect import bisect_right
from collections import deque

import av
from av.video.frame import VideoFrame

BACK_CACHE_BYTES = 128 * 1024 * 1024  # decoded frames kept for stepping back
BACK_CACHE_MAX = 32
FORWARD_WALK_UNKNOWN_KF = 0.3  # seconds decoded forward instead of seeking when keyframes are unknown
SCRUB_SLOW_MS = 60.0  # exact seeks slower than this (typical for 4K) make scrub() show keyframes
EPS = 1e-4

# AVColorSpace / AVColorRange values (what VideoFrame.colorspace / .color_range return)
CS_RGB, CS_BT709, CS_UNSPECIFIED, CS_BT601 = 0, 1, 2, 5
RANGE_MPEG, RANGE_JPEG = 1, 2


def rgb_reformat_args(frame: VideoFrame) -> dict:
    """VideoFrame.reformat() arguments that turn a YUV frame into full-range RGB the way a player
    does: with the frame's own matrix, limited-range input unless tagged full. Untagged video
    gets what players assume: BT.709 from 720 lines up, BT.601 below.

    A plain reformat(format="rgb24") always uses BT.601, which shifts HD reds by up to ~14
    levels. PyAV only applies src_colorspace when the colour ranges are passed as well.
    """
    if frame.format.is_rgb:
        return {}
    cs = frame.colorspace
    if cs in (None, CS_RGB, CS_UNSPECIFIED):
        cs = CS_BT709 if frame.height >= 720 else CS_BT601
    full = frame.color_range == RANGE_JPEG or frame.format.name.startswith("yuvj")
    return {"src_colorspace": cs, "src_color_range": RANGE_JPEG if full else RANGE_MPEG,
            "dst_color_range": RANGE_JPEG}


class Decoder:
    def __init__(self, path: str, keyframes: list[float] | None = None):
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self._gen = None
        self.current: VideoFrame | None = None
        self.eof = False
        # Sorted keyframe pts in seconds (from probe). Optional: only used to decide whether
        # decoding forward beats a seek. Can be set later (the keyframe scan runs in the background).
        self.keyframes: list[float] = list(keyframes or [])
        self.restarts = 0  # container seeks done so far
        self.seek_ms = 0.0  # running average time of exact seeks that needed a container seek
        # Invariants: _back holds the frames right before `current`, oldest first;
        # _ahead holds already-decoded frames right after it, and the generator
        # continues after the last of them.
        self._back: deque[VideoFrame] = deque()
        self._ahead: deque[VideoFrame] = deque()
        cc = self.stream.codec_context
        frame_bytes = max(1, int((cc.width or 1920) * (cc.height or 1080) * 1.5))  # yuv420
        self._back_max = max(2, min(BACK_CACHE_MAX, BACK_CACHE_BYTES // frame_bytes))
        st0 = self.stream.start_time
        # Lowest seek position worth trying: a seek there always starts at the first frame.
        self._floor = min(0.0, float(st0 * self.stream.time_base)) if st0 is not None else 0.0

    def close(self):
        self._back.clear()
        self._ahead.clear()
        self.container.close()

    # ------------------------------------------------------------------
    @property
    def time(self) -> float:
        return self.current.time if self.current is not None else 0.0

    @property
    def is_keyframe(self) -> bool:
        return bool(self.current.key_frame) if self.current is not None else False

    def _restart(self, t: float):
        ts = int(t / self.stream.time_base)
        self.container.seek(ts, stream=self.stream, backward=True, any_frame=False)
        self.restarts += 1
        self._gen = self.container.decode(self.stream)
        self.eof = False
        self._back.clear()
        self._ahead.clear()

    def _restart_before(self, t: float):
        """Seek so that decoding starts at or before t.

        The first frame decoded is checked; when it is later than t (or the seek ran into the
        end of the file) the seek is repeated 0.5, 1, 2, 4… s earlier, down to the start of
        the file. The frames read for the check stay queued in _ahead.
        """
        back = 0.0
        while True:
            pos = max(t - back, self._floor)
            self._restart(pos)
            f = self._peek()
            if pos <= self._floor or (f is not None and f.time <= t + EPS):
                return
            back = back * 2 if back else 0.5

    def _peek(self) -> VideoFrame | None:
        """Decode up to the next frame that has a time and queue everything read in _ahead."""
        while True:
            f = self._next_raw()
            if f is None:
                return None
            self._ahead.append(f)
            if f.time is not None:
                return f

    def _next_raw(self) -> VideoFrame | None:
        if self._ahead:
            return self._ahead.popleft()
        if self._gen is None:
            self._restart(self._floor)
        try:
            f = next(self._gen)
        except StopIteration:
            self.eof = True
            return None
        return f

    def _remember(self, f: VideoFrame | None):
        """`f` is now behind the current frame: keep it for step_back."""
        if f is None:
            return
        self._back.append(f)
        while len(self._back) > self._back_max:
            self._back.popleft()

    # ------------------------------------------------------------------
    def _forward_is_cheap(self, frm: float, t: float) -> bool:
        """Decoding forward from frm to t costs less than a seek: no keyframe lies in between
        (a seek would start decoding at the same keyframe or earlier)."""
        if not self.keyframes:
            return t - frm <= FORWARD_WALK_UNKNOWN_KF
        i = bisect_right(self.keyframes, t + EPS)
        return i == 0 or self.keyframes[i - 1] <= frm + EPS

    def _seek_cached(self, t: float) -> VideoFrame | None:
        """seek() without a container seek, when the cache or a short forward walk reaches t.
        Returns None when a real seek is needed."""
        cur = self.current
        if cur is None or cur.time is None or self._gen is None:
            return None
        goal = t - EPS
        if cur.time >= goal:
            # At or past the target: the answer is in the cache when the cache reaches back
            # to a frame before the target (that frame proves nothing earlier qualifies).
            back = self._back
            if not back or back[0].time is None or back[0].time >= goal:
                return None
            while back[-1].time is None or back[-1].time >= goal:
                self._ahead.appendleft(self.current)
                self.current = back.pop()
            return self.current
        if not self._forward_is_cheap(cur.time, t):
            return None
        while True:
            f = self._next_raw()
            if f is None:
                return self.current  # ran past the end: keep the last frame
            if f.time is None:
                continue
            self._remember(self.current)
            self.current = f
            if f.time >= goal:
                return f

    def seek(self, t: float) -> VideoFrame | None:
        """Go to the first frame whose time >= t (frame accurate)."""
        f = self._seek_cached(t)
        if f is not None:
            return f
        t0 = time.perf_counter()
        f = self._seek_exact(t)
        ms = (time.perf_counter() - t0) * 1000
        self.seek_ms = ms if self.seek_ms == 0 else 0.5 * self.seek_ms + 0.5 * ms
        return f

    def scrub(self, t: float) -> tuple[VideoFrame | None, bool]:
        """seek() for dragging the playhead. When only a slow container seek reaches t exactly
        (exact seeks average over SCRUB_SLOW_MS on this video), show the keyframe at or before t
        instead: one decoded frame, or none when it is on screen already. Returns (frame, exact)."""
        f = self._seek_cached(t)
        if f is not None:
            return f, True
        if self.seek_ms < SCRUB_SLOW_MS:
            return self.seek(t), True
        return self.seek_keyframe(t), False

    def _seek_exact(self, t: float, eps: float = EPS) -> VideoFrame | None:
        self._restart_before(max(t, self._floor))
        while True:
            f = self._next_raw()
            if f is None:
                # ran past the end: keep the last frame we saw
                if self._back:
                    self.current = self._back.pop()
                return self.current
            if f.time is None:
                continue
            if f.time >= t - eps:
                self.current = f
                return f
            self._remember(f)

    def seek_keyframe(self, t: float) -> VideoFrame | None:
        """Show the keyframe at or before t: one seek and one decoded frame, for scrubbing
        heavy video. Does nothing when that keyframe is the current frame already."""
        if self.keyframes:
            i = bisect_right(self.keyframes, t + EPS)
            k = self.keyframes[i - 1] if i else self.keyframes[0]
            cur = self.current
            if cur is not None and cur.time is not None and abs(cur.time - k) <= EPS:
                return cur
            f = self._seek_cached(k)
            return f if f is not None else self._seek_exact(k)
        self._restart_before(max(t, self._floor))
        while True:
            f = self._next_raw()
            if f is None:
                return self.current
            if f.time is not None:
                self.current = f
                return f

    def step_forward(self) -> VideoFrame | None:
        f = self._next_raw()
        if f is None:
            return None
        self._remember(self.current)
        self.current = f
        return f

    def step_back(self) -> VideoFrame | None:
        if self.current is None:
            return None
        if self._back:  # decoded already: no seek, no decode
            self._ahead.appendleft(self.current)
            self.current = self._back.pop()
            return self.current
        target = self.current.time
        if target is None:
            return self.current
        # Seek to a keyframe strictly before the current frame.
        self._restart_before(max(target - 1e-3, self._floor))
        crossed = False
        while True:
            f = self._next_raw()
            if f is None:
                break
            if f.time is None:
                continue
            if f.time >= target - EPS:
                # the frame we came from: step_forward returns it without decoding again
                self._ahead.append(f)
                crossed = True
                break
            self._remember(f)
        if not self._back:
            # Nothing decodes before the current frame: it is the first one. Keep it (the copy
            # just decoded, so the generator continues right after it).
            if crossed:
                self.current = self._ahead.popleft()
            return self.current
        prev = self._back.pop()
        if not crossed:
            # Hit the end without meeting the current frame again; re-seek so
            # that step_forward continues correctly from prev.
            return self._seek_exact(prev.time)
        self.current = prev
        return prev

    def seek_last(self, duration_hint: float) -> VideoFrame | None:
        """Decode to the final frame of the stream."""
        self._restart_before(max(duration_hint - 0.5, self._floor))
        while True:
            f = self._next_raw()
            if f is None:
                break
            self._remember(f)
        if self._back:
            self.current = self._back.pop()
        return self.current
