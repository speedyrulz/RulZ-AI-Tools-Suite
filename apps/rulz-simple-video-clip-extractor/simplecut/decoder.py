"""Frame-accurate decoder built on PyAV.

All positions are pts in seconds. Stepping forward decodes the next frame;
stepping back re-decodes from the previous keyframe (cheap enough for a GOP).

The frames just before the current one are kept (a small, memory-bounded
cache), so stepping back after stepping or seeking forward doesn't decode
anything; and the frame a backward walk ran into is kept as a lookahead, so
the next step forward doesn't seek and decode the GOP a second time.
"""
from __future__ import annotations

from collections import deque

import av
from av.video.frame import VideoFrame

BACK_CACHE_BYTES = 128 * 1024 * 1024  # decoded frames kept for stepping back
BACK_CACHE_MAX = 32


class Decoder:
    def __init__(self, path: str):
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self._gen = None
        self.current: VideoFrame | None = None
        self.eof = False
        # Invariants: _back holds the frames right before `current`, oldest first;
        # _ahead holds already-decoded frames right after it, and the generator
        # continues after the last of them.
        self._back: deque[VideoFrame] = deque()
        self._ahead: deque[VideoFrame] = deque()
        cc = self.stream.codec_context
        frame_bytes = max(1, int((cc.width or 1920) * (cc.height or 1080) * 1.5))  # yuv420
        self._back_max = max(2, min(BACK_CACHE_MAX, BACK_CACHE_BYTES // frame_bytes))

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
        self._gen = self.container.decode(self.stream)
        self.eof = False
        self._back.clear()
        self._ahead.clear()

    def _next_raw(self) -> VideoFrame | None:
        if self._ahead:
            return self._ahead.popleft()
        if self._gen is None:
            self._restart(0)
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
    def seek(self, t: float, eps: float = 1e-4) -> VideoFrame | None:
        """Go to the first frame whose time >= t (frame accurate)."""
        self._restart(max(t, 0.0))
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
        # Seek to the keyframe strictly before the current frame.
        self._restart(max(target - 1e-3, 0.0))
        crossed = False
        while True:
            f = self._next_raw()
            if f is None:
                break
            if f.time is None:
                continue
            if f.time >= target - 1e-4:
                # the frame we came from: step_forward returns it without decoding again
                self._ahead.append(f)
                crossed = True
                break
            self._remember(f)
        if not self._back:
            # Already at the very first frame; re-seek to it.
            return self.seek(0.0)
        prev = self._back.pop()
        if not crossed:
            # Hit the end without meeting the current frame again; re-seek so
            # that step_forward continues correctly from prev.
            return self.seek(prev.time)
        self.current = prev
        return prev

    def seek_last(self, duration_hint: float) -> VideoFrame | None:
        """Decode to the final frame of the stream."""
        self._restart(max(duration_hint - 0.5, 0.0))
        while True:
            f = self._next_raw()
            if f is None:
                break
            self._remember(f)
        if self._back:
            self.current = self._back.pop()
        return self.current
