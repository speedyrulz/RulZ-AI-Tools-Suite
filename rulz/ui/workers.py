"""Image data plus background jobs: thumbnail loading, full-size previews and saving."""
from __future__ import annotations

import os
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
COPY_CHECK_SIDE = 64  # a picture copied as it is is decoded this small, just to check that it reads


@dataclass(eq=False)
class ImageItem:
    path: Path
    thumb: np.ndarray | None = None  # small 8-bit copy of the original
    custom: dict[str, int] = field(default_factory=dict)  # own values; missing = follow global
    shadow: ShadowEdit | None = None  # painted shadow areas to lift (this picture only)
    preview: object = None  # QPixmap of the processed thumbnail
    preview_key: tuple | None = None
    error: str | None = None
    removed: bool = False  # taken out of the list (its thumbnail isn't loaded any more)

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
        if self.item.removed:
            return
        try:
            img = imageio.shrink(imageio.load(self.item.path, self.side), self.side)
            result = imageio.to_8bit(img)
        except Exception as e:  # noqa: BLE001 - reported on the card
            result = e
        self.emitter.done.emit(self.item, result)


class FullLoad(QRunnable):
    """The editor's picture at full size. Emits (picture, QImage of it): the screen copy is made here, not on
    the GUI thread, where it held the window up to 0.3 s for a big 16-bit photo."""

    def __init__(self, item: ImageItem, emitter: _Emitter, key=None):
        super().__init__()
        self.item, self.emitter = item, emitter
        self.key = item if key is None else key

    def run(self):
        try:
            img = imageio.load(self.item.path)
            result = (img, imageio.to_qimage(img))
        except Exception as e:  # noqa: BLE001
            result = e
        self.emitter.done.emit(self.key, result)


class Process(QRunnable):
    """The editor's full-size preview. Emits (QImage of the result, cache): pass the cache to the next Process
    of the same picture, so moving only the color noise or detail slider skips the JPEG and shadow steps."""

    def __init__(self, key, img: np.ndarray, settings: dict, mode: str, emitter: _Emitter,
                 shadow: ShadowEdit | None = None, cache=None):
        super().__init__()
        self.key, self.img, self.settings, self.mode, self.emitter = key, img, settings, mode, emitter
        self.shadow, self.cache = shadow, cache

    def run(self):
        try:
            img, cache = process_cached(self.img, self.settings, self.mode, self.shadow, self.cache)
            result = (imageio.to_qimage(img), cache)
        except Exception as e:  # noqa: BLE001
            result = e
        self.cache = self.img = None
        self.emitter.done.emit(self.key, result)


def make_emitter(parent: QObject | None = None) -> _Emitter:
    return _Emitter(parent)


def path_key(path: Path, folders: dict) -> str:
    """The file's full path, case-folded and resolved (links, junctions, short names), to tell whether two
    paths are the same file. Path.resolve() costs about 2 ms a file on Windows and far more for a file that
    doesn't exist yet (3,000 pictures: over a minute), so it's done once per folder, cached in `folders`:
    each folder is resolved and listed once, and only the files its listing shows to be links (to a file
    somewhere else, maybe a picture in the list) are resolved on their own."""
    folder = path.parent
    known = folders.get(folder)
    if known is None:
        try:
            links = {os.path.normcase(e.name) for e in os.scandir(folder) if e.is_symlink()}
        except OSError:  # not there (yet): nothing in it is a link
            links = set()
        known = folders[folder] = (folder.resolve(), links)
    real, links = known
    if os.path.normcase(path.name) in links:
        return os.path.normcase(str(path.resolve()))
    return os.path.normcase(str(real / path.name))


def output_paths(items: list[ImageItem], out_dir: Path, others=()) -> list[Path]:
    """Same file names in out_dir; never overwrite any picture in the batch or in `others` (the paths of the
    rest of the list) and never reuse a name in it.

    Every source counts, not just the picture's own: with X/a.jpg and Y/a.jpg in the list, saving into X
    must not put Y's result over X's original - also when only Y/a.jpg is being saved. Links count as the
    file they point to, both in the list and in out_dir."""
    folders: dict = {}
    sources = {path_key(it.path, folders) for it in items} | {path_key(Path(p), folders) for p in others}
    used: set[str] = set()
    paths = []
    for it in items:
        stem, ext = it.path.stem, it.path.suffix
        name = f"{stem}{ext}"
        n = 1
        while name.lower() in used or path_key(out_dir / name, folders) in sources:
            name = f"{stem}{'_prepared' if n == 1 else f'_prepared_{n}'}{ext}"
            n += 1
        used.add(name.lower())
        paths.append(out_dir / name)
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
        15 s instead of 22 s). Best quality runs on the GPU one picture at a time anyway, but only pictures with
        JPEG artifact removal on use it."""
        uses_ai = self.mode == BEST and any(settings.get("jpeg", 0) > 0 for _, _, settings, _ in self.jobs)
        return 1 if uses_ai else SAVE_WORKERS

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
                data = imageio.read(src)  # read once for the decoding, the checks and the copy
                copy = is_noop(settings) and (shadow is None or not shadow.active) and imageio.can_copy_unchanged(data)
                # Decoded also when it's copied, so a file that can't be read is reported, not copied; for
                # that check a big JPEG is decoded at 1/8 size, several times faster.
                img = imageio.decode(data, src.name, COPY_CHECK_SIDE if copy else None)
                if copy:
                    data.tofile(str(dst))  # nothing to change: re-encoding would only add loss and size
                else:
                    imageio.save(dst, process(img, settings, self.mode, shadow=shadow), imageio.icc_profile(data))
                del img, data
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
