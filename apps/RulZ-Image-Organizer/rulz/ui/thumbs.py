"""Asynchronous thumbnail loading with an in-memory LRU cache."""
from __future__ import annotations

import heapq
import os
import threading
from collections import OrderedDict

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap

# A queued load is dropped once this many newer requests came in without it being asked for again. Grids
# ask for every visible tile on each paint, so anything still on screen keeps getting re-stamped; what
# was scrolled past quickly falls behind and is skipped.
STALE_AFTER = 2000


class _Signals(QObject):
    done = Signal(str, QImage, bool)  # key, image, dropped


class _Job(QRunnable):
    """One job is started per queued request; each takes the *newest* request off the queue."""

    def __init__(self, loader: "ThumbLoader"):
        super().__init__()
        self.loader = loader

    def run(self):
        while True:
            item = self.loader._take()
            if item is None:
                return
            key, path, fallback, max_side, stale = item
            if stale:
                self.loader.sig.done.emit(key, QImage(), True)
                continue  # cheap: go on to the next request
            self.loader.sig.done.emit(key, _read(path, fallback, max_side), False)
            return


def _read(path: str, fallback: str | None, max_side: int) -> QImage:
    img = QImage()
    for p in (path, fallback):
        if not p or not os.path.exists(p):
            continue
        reader = QImageReader(p)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid() and min(size.width(), size.height()) > max_side:
            size.scale(QSize(max_side, max_side), Qt.KeepAspectRatioByExpanding)
            reader.setScaledSize(size)
        img = reader.read()
        if not img.isNull():
            break
    return img


class ThumbLoader(QObject):
    """Shared loader. ``get`` returns a cached pixmap or schedules a load and returns None."""

    loaded = Signal(str)

    _instance: "ThumbLoader | None" = None

    @classmethod
    def instance(cls) -> "ThumbLoader":
        if cls._instance is None:
            cls._instance = ThumbLoader()
        return cls._instance

    def __init__(self, budget_mb: int = 384):
        super().__init__()
        # Bounded by memory, not count: a panorama thumbnail can weigh as much as ten square ones.
        self.budget = budget_mb * 1024 * 1024
        self.cache_bytes = 0
        self.cache: OrderedDict[str, QPixmap] = OrderedDict()
        self.failed: set[str] = set()
        self.pending: set[str] = set()
        # Queue shared with the pool threads (guarded by _lock): newest request first.
        self._lock = threading.Lock()
        self._seq = 0
        self._stamp: dict[str, int] = {}                 # key -> latest request number
        self._args: dict[str, tuple] = {}                # key -> (path, fallback, max_side)
        self._heap: list[tuple[int, str]] = []           # (-request number, key); outdated entries skipped
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(max(2, min(6, os.cpu_count() or 4)))
        self.sig = _Signals()
        self.sig.done.connect(self._on_done)

    def get(self, path: str, fallback: str | None = None, max_side: int = 480) -> QPixmap | None:
        pm = self.cache.get(path)
        if pm is not None:
            self.cache.move_to_end(path)
            return pm
        if path in self.failed:
            return None
        new = path not in self.pending
        with self._lock:
            if not new and path not in self._stamp:
                return None  # being read on a pool thread right now
            self._seq += 1
            self._stamp[path] = self._seq
            self._args[path] = (path, fallback, max_side)
            heapq.heappush(self._heap, (-self._seq, path))
        if new:
            self.pending.add(path)
            self.pool.start(_Job(self))
        return None

    def _take(self):
        """Pool thread: the newest queued request, as (key, path, fallback, max_side, stale), or None."""
        with self._lock:
            while self._heap:
                neg, key = heapq.heappop(self._heap)
                if self._stamp.get(key) != -neg:
                    continue  # asked for again since: a newer heap entry represents it
                del self._stamp[key]
                path, fallback, max_side = self._args.pop(key)
                return key, path, fallback, max_side, self._seq + neg > STALE_AFTER
            return None

    def invalidate(self, path: str) -> None:
        pm = self.cache.pop(path, None)
        if pm is not None:
            self.cache_bytes -= _cost(pm)
        self.failed.discard(path)

    def _on_done(self, key: str, img: QImage, dropped: bool):
        self.pending.discard(key)
        if dropped:
            pass  # scrolled away before its turn; the repaint below asks again if it's back on screen
        elif img.isNull():
            self.failed.add(key)
        else:
            old = self.cache.pop(key, None)
            if old is not None:
                self.cache_bytes -= _cost(old)
            pm = QPixmap.fromImage(img)
            self.cache[key] = pm
            self.cache_bytes += _cost(pm)
            while self.cache_bytes > self.budget and len(self.cache) > 1:
                _, gone = self.cache.popitem(last=False)
                self.cache_bytes -= _cost(gone)
        self.loaded.emit(key)


def _cost(pm: QPixmap) -> int:
    return pm.width() * pm.height() * max(1, pm.depth() // 8)
