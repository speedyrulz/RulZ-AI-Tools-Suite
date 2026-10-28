"""Asynchronous thumbnail loading with an in-memory LRU cache."""
from __future__ import annotations

import os
from collections import OrderedDict

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap


class _Signals(QObject):
    done = Signal(str, QImage)


class _Job(QRunnable):
    def __init__(self, key: str, path: str, fallback: str | None, max_side: int, sig: _Signals):
        super().__init__()
        self.key, self.path, self.fallback, self.max_side, self.sig = key, path, fallback, max_side, sig

    def run(self):
        img = QImage()
        for p in (self.path, self.fallback):
            if not p or not os.path.exists(p):
                continue
            reader = QImageReader(p)
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid() and min(size.width(), size.height()) > self.max_side:
                size.scale(QSize(self.max_side, self.max_side), Qt.KeepAspectRatioByExpanding)
                reader.setScaledSize(size)
            img = reader.read()
            if not img.isNull():
                break
        self.sig.done.emit(self.key, img)


class ThumbLoader(QObject):
    """Shared loader. ``get`` returns a cached pixmap or schedules a load and returns None."""

    loaded = Signal(str)

    _instance: "ThumbLoader | None" = None

    @classmethod
    def instance(cls) -> "ThumbLoader":
        if cls._instance is None:
            cls._instance = ThumbLoader()
        return cls._instance

    def __init__(self, capacity: int = 1500):
        super().__init__()
        self.capacity = capacity
        self.cache: OrderedDict[str, QPixmap] = OrderedDict()
        self.failed: set[str] = set()
        self.pending: set[str] = set()
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(max(2, min(6, os.cpu_count() or 4)))
        self.sig = _Signals()
        self.sig.done.connect(self._on_done)

    def get(self, path: str, fallback: str | None = None, max_side: int = 480) -> QPixmap | None:
        pm = self.cache.get(path)
        if pm is not None:
            self.cache.move_to_end(path)
            return pm
        if path not in self.pending and path not in self.failed:
            self.pending.add(path)
            self.pool.start(_Job(path, path, fallback, max_side, self.sig))
        return None

    def invalidate(self, path: str) -> None:
        self.cache.pop(path, None)
        self.failed.discard(path)

    def _on_done(self, key: str, img: QImage):
        self.pending.discard(key)
        if img.isNull():
            self.failed.add(key)
        else:
            self.cache[key] = QPixmap.fromImage(img)
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.loaded.emit(key)
