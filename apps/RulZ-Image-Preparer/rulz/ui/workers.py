"""Image data plus background jobs: thumbnail loading, full-size previews and saving."""
from __future__ import annotations

import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThread, Signal

from .. import config, imageio
from ..filters import ADJUSTMENTS, BEST, is_noop, process, process_cached
from ..shadows import ShadowEdit



def _total_memory() -> int | None:
    """Physical memory in bytes, or None when it can't be told."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_avail", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_avail", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]

            st = _Status()
            st.length = ctypes.sizeof(st)
            return int(st.total) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)) else None
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


# A 24 MP picture takes about 1.7 GB while it's cleaned, a 50 MP one twice that: two at once only with 16 GB.
SAVE_WORKERS = 2 if (_total_memory() or 0) >= 15 * 2 ** 30 else 1


@dataclass(eq=False)
class ImageItem:
    path: Path
    thumb: np.ndarray | None = None  # small 8-bit copy of the original
    custom: dict[str, int] = field(default_factory=dict)  # own values; missing = follow global
    shadow: ShadowEdit | None = None  # painted shadow areas to lift (this picture only)
    preview: object = None  # QPixmap of the processed thumbnail
    preview_key: tuple | None = None
    error: str | None = None

    def value(self, key: str, globals_: dict[str, int]) -> int:
        return self.custom.get(key, globals_[key])

    def settings(self, globals_: dict[str, int]) -> dict[str, int]:
        return {k: self.value(k, globals_) for k in ADJUSTMENTS}

    def shadow_edit(self) -> ShadowEdit | None:
        """The shadow areas if they change anything, else None."""
        return self.shadow if self.shadow is not None and self.shadow.active else None


class _Emitter(QObject):
    done = Signal(object, object)  # (key, result or Exception)


class ThumbLoad(QRunnable):
    def __init__(self, item: ImageItem, emitter: _Emitter, side: int = config.THUMB_MAX_SIDE):
        super().__init__()
        self.item, self.emitter, self.side = item, emitter, side

    def run(self):
        try:
            img = imageio.shrink(imageio.load(self.item.path, self.side), self.side)
            result = imageio.to_8bit(img)
        except Exception as e:  # noqa: BLE001 - reported on the card
            result = e
        self.emitter.done.emit(self.item, result)


class FullLoad(QRunnable):
    def __init__(self, item: ImageItem, emitter: _Emitter, key=None):
        super().__init__()
        self.item, self.emitter = item, emitter
        self.key = item if key is None else key

    def run(self):
        try:
            result = imageio.load(self.item.path)
        except Exception as e:  # noqa: BLE001
            result = e
        self.emitter.done.emit(self.key, result)


class Process(QRunnable):
    """The editor's full-size preview. Emits (result, cache): pass the cache to the next Process of the same
    picture, so moving only the color noise or detail slider skips the JPEG and shadow steps."""

    def __init__(self, key, img: np.ndarray, settings: dict, mode: str, emitter: _Emitter,
                 shadow: ShadowEdit | None = None, cache=None):
        super().__init__()
        self.key, self.img, self.settings, self.mode, self.emitter = key, img, settings, mode, emitter
        self.shadow, self.cache = shadow, cache

    def run(self):
        try:
            result = process_cached(self.img, self.settings, self.mode, self.shadow, self.cache)
        except Exception as e:  # noqa: BLE001
            result = e
        self.cache = self.img = None
        self.emitter.done.emit(self.key, result)


def make_emitter(parent: QObject | None = None) -> _Emitter:
    return _Emitter(parent)


def output_paths(items: list[ImageItem], out_dir: Path) -> list[Path]:
    """Same file names in out_dir; never overwrite any picture in the batch and never reuse a name in it.

    Every source counts, not just the picture's own: with X/a.jpg and Y/a.jpg in the list, saving into X
    must not put Y's result over X's original."""
    sources = {it.path.resolve() for it in items}
    used: set[str] = set()
    paths = []
    for it in items:
        stem, ext = it.path.stem, it.path.suffix
        candidate = out_dir / f"{stem}{ext}"
        n = 1
        while candidate.name.lower() in used or candidate.resolve() in sources:
            suffix = "_prepared" if n == 1 else f"_prepared_{n}"
            candidate = out_dir / f"{stem}{suffix}{ext}"
            n += 1
        used.add(candidate.name.lower())
        paths.append(candidate)
    return paths


class SaveAll(QThread):
    progress = Signal(int, str)  # pictures done so far, file name being started ("" at the end)
    finished_all = Signal(int, list)  # saved count, [(name, error)]

    def __init__(self, jobs: list[tuple[Path, Path, dict, ShadowEdit | None]], mode: str):
        super().__init__()
        self.jobs = jobs  # (source, destination, settings, shadow areas)
        self.mode = mode
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def workers(self) -> int:
        """Pictures saved at the same time. Much of a picture's time is spent in steps that use one CPU core
        (decoding, encoding, parts of the filters), so two at once finish a batch sooner (six 24 MP photos:
        15 s instead of 22 s). Best quality runs on the GPU one picture at a time anyway."""
        return 1 if self.mode == BEST else SAVE_WORKERS

    def run(self):
        lock = threading.Lock()
        state = {"done": 0, "saved": 0}
        errors: list[tuple[int, str, str]] = []

        def one(job):
            i, (src, dst, settings, shadow) = job
            if self._cancel:
                return
            with lock:
                self.progress.emit(state["done"], src.name)
            try:
                img = imageio.load(src)  # also when it's copied: a file that can't be read is reported, not copied
                if is_noop(settings) and (shadow is None or not shadow.active) and imageio.can_copy_unchanged(src):
                    shutil.copyfile(src, dst)  # nothing to change: re-encoding would only add loss and size
                else:
                    imageio.save(dst, process(img, settings, self.mode, shadow=shadow))
                ok = True
            except Exception as e:  # noqa: BLE001
                ok = False
                with lock:
                    errors.append((i, src.name, str(e)))
            with lock:
                state["done"] += 1
                state["saved"] += ok

        with ThreadPoolExecutor(self.workers()) as ex:
            list(ex.map(one, enumerate(self.jobs)))
        self.progress.emit(len(self.jobs), "")
        self.finished_all.emit(state["saved"], [(name, err) for _, name, err in sorted(errors)])
