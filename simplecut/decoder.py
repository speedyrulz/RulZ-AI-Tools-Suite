"""Frame-accurate decoder built on PyAV.

All positions are pts in seconds. Stepping forward decodes the next frame;
stepping back re-decodes from the previous keyframe (cheap enough for a GOP).
"""
from __future__ import annotations

import av
from av.video.frame import VideoFrame


class Decoder:
    def __init__(self, path: str):
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self._gen = None
        self.current: VideoFrame | None = None
        self.eof = False

    def close(self):
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

    def _next_raw(self) -> VideoFrame | None:
        if self._gen is None:
            self._restart(0)
        try:
            f = next(self._gen)
        except StopIteration:
            self.eof = True
            return None
        return f

    # ------------------------------------------------------------------
    def seek(self, t: float, eps: float = 1e-4) -> VideoFrame | None:
        """Go to the first frame whose time >= t (frame accurate)."""
        self._restart(max(t, 0.0))
        last = None
        while True:
            f = self._next_raw()
            if f is None:
                # ran past the end: keep the last frame we saw
                if last is not None:
                    self.current = last
                return self.current
            if f.time is None:
                continue
            if f.time >= t - eps:
                self.current = f
                return f
            last = f

    def step_forward(self) -> VideoFrame | None:
        f = self._next_raw()
        if f is None:
            return None
        self.current = f
        return f

    def step_back(self) -> VideoFrame | None:
        if self.current is None:
            return None
        target = self.current.time
        # Seek to the keyframe strictly before the current frame.
        self._restart(max(target - 1e-3, 0.0))
        prev = None
        while True:
            f = self._next_raw()
            if f is None or f.time is None:
                if f is None:
                    break
                continue
            if f.time >= target - 1e-4:
                break
            prev = f
        if prev is None:
            # Already at the very first frame; re-seek to it.
            return self.seek(0.0)
        # The generator is now one frame past 'prev'; re-seek so that
        # step_forward continues correctly from prev.
        return self.seek(prev.time)

    def seek_last(self, duration_hint: float) -> VideoFrame | None:
        """Decode to the final frame of the stream."""
        self._restart(max(duration_hint - 0.5, 0.0))
        last = None
        while True:
            f = self._next_raw()
            if f is None:
                break
            last = f
        if last is not None:
            self.current = last
        return self.current
