"""Image data plus background jobs: thumbnail loading, full-size previews and saving."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThread, Signal

from .. import config, imageio
from ..filters import ADJUSTMENTS, process


@dataclass(eq=False)
class ImageItem:
    path: Path
    thumb: np.ndarray | None = None  # small 8-bit copy of the original
    custom: dict[str, int] = field(default_factory=dict)  # own values; missing = follow global
    preview: object = None  # QPixmap of the processed thumbnail
    preview_key: tuple | None = None
    error: str | None = None

    def value(self, key: str, globals_: dict[str, int]) -> int:
        return self.custom.get(key, globals_[key])

    def settings(self, globals_: dict[str, int]) -> dict[str, int]:
        return {k: self.value(k, globals_) for k in ADJUSTMENTS}


class _Emitter(QObject):
    done = Signal(object, object)  # (key, result or Exception)


class ThumbLoad(QRunnable):
    def __init__(self, item: ImageItem, emitter: _Emitter):
        super().__init__()
        self.item, self.emitter = item, emitter

    def run(self):
        try:
            img = imageio.shrink(imageio.load(self.item.path), config.THUMB_MAX_SIDE)
            result = imageio.to_8bit(img)
        except Exception as e:  # noqa: BLE001 - reported on the card
            result = e
        self.emitter.done.emit(self.item, result)


class FullLoad(QRunnable):
    def __init__(self, item: ImageItem, emitter: _Emitter):
        super().__init__()
        self.item, self.emitter = item, emitter

    def run(self):
        try:
            result = imageio.load(self.item.path)
        except Exception as e:  # noqa: BLE001
            result = e
        self.emitter.done.emit(self.item, result)


class Process(QRunnable):
    def __init__(self, key, img: np.ndarray, settings: dict, mode: str, emitter: _Emitter):
        super().__init__()
        self.key, self.img, self.settings, self.mode, self.emitter = key, img, settings, mode, emitter

    def run(self):
        try:
            result = process(self.img, self.settings, self.mode)
        except Exception as e:  # noqa: BLE001
            result = e
        self.emitter.done.emit(self.key, result)


def make_emitter(parent: QObject | None = None) -> _Emitter:
    return _Emitter(parent)


def output_paths(items: list[ImageItem], out_dir: Path) -> list[Path]:
    """Same file names in out_dir; never overwrite a source and never reuse a name in one batch."""
    used: set[str] = set()
    paths = []
    for it in items:
        stem, ext = it.path.stem, it.path.suffix
        candidate = out_dir / f"{stem}{ext}"
        n = 1
        while candidate.name.lower() in used or candidate.resolve() == it.path.resolve():
            suffix = "_prepared" if n == 1 else f"_prepared_{n}"
            candidate = out_dir / f"{stem}{suffix}{ext}"
            n += 1
        used.add(candidate.name.lower())
        paths.append(candidate)
    return paths


class SaveAll(QThread):
    progress = Signal(int, str)  # index, file name
    finished_all = Signal(int, list)  # saved count, [(name, error)]

    def __init__(self, jobs: list[tuple[Path, Path, dict]], mode: str):
        super().__init__()
        self.jobs = jobs  # (source, destination, settings)
        self.mode = mode
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        saved, errors = 0, []
        for i, (src, dst, settings) in enumerate(self.jobs):
            if self._cancel:
                break
            self.progress.emit(i, src.name)
            try:
                img = imageio.load(src)
                imageio.save(dst, process(img, settings, self.mode))
                saved += 1
            except Exception as e:  # noqa: BLE001
                errors.append((src.name, str(e)))
        self.progress.emit(len(self.jobs), "")
        self.finished_all.emit(saved, errors)
