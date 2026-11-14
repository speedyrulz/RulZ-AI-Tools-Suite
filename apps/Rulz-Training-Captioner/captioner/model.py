"""Image list model, lazy thumbnail loading and caption .txt persistence."""
import functools
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import (QAbstractListModel, QModelIndex, QObject, QRunnable, QSortFilterProxyModel,
                            Qt, QThreadPool, Signal)
from PySide6.QtGui import QImage, QImageReader, QPixmap

from .config import IMAGE_EXTS

PathRole = Qt.UserRole + 1
StatusRole = Qt.UserRole + 2
HasFileRole = Qt.UserRole + 3
THUMB_GEN = 256
# Thumbnails kept in memory, by size: 192 MB holds ~770 square 256 px thumbnails (non-square ones are
# smaller), several screens' worth even at the smallest card size on a 4K monitor. A count limit
# let 1500 of them reach ~390 MB.
CACHE_BYTES = 192 * 1024 * 1024

# status values
IDLE, QUEUED, BUSY, ERROR = "idle", "queued", "busy", "error"


def _natural(s):
    # re.split with a group alternates text and digit runs, so odd positions are always digits.
    # str.isdigit() isn't a safe test: it is true for "²", which int() rejects.
    return [int(t) if i % 2 else t.lower() for i, t in enumerate(re.split(r"(\d+)", s))]


def natural_key(p):
    """Sort key for a file name: img2 before img10."""
    return _natural(os.path.basename(p))


@functools.lru_cache(maxsize=4096)
def _folder_key(folder):
    return tuple(tuple(_natural(part)) for part in Path(folder).parts)  # once per folder, not per file


def sort_key(p):
    """Sort key for a path: folder by folder, then the file name, so a folder's subfolders don't
    interleave (b\\1, a\\1, a\\2 became a\\1, a\\2, b\\1)."""
    folder, name = os.path.split(p)
    return _folder_key(folder), tuple(_natural(name))


def _has_text(caption):
    return 1 if caption.strip() else 0


def caption_path(img_path):
    return str(Path(img_path).with_suffix(".txt"))


def caption_clashes(paths):
    """Groups of images that share one caption file, e.g. photo.png and photo.jpg -> photo.txt."""
    groups = {}
    for p in paths:
        groups.setdefault(os.path.normcase(caption_path(p)), []).append(p)
    return [g for g in groups.values() if len(g) > 1]


def decode_caption(data):
    """UTF-8 (with or without BOM); captions saved by older Windows tools in the ANSI code page
    still read correctly instead of turning into replacement characters, and so do Notepad's
    "Unicode" (UTF-16 with a BOM) files, which cp1252 turned into letters with NULs between them."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def read_caption(img_path):
    """(caption, True), or ("", False) when there is no caption file. A caption file that is there but
    can't be read (locked, no permission, a folder of that name) raises OSError: treating it as
    missing let Caption All with "skip existing" caption the image and overwrite the file."""
    cp = caption_path(img_path)
    try:
        with open(cp, "rb") as f:
            return decode_caption(f.read()), True
    except (FileNotFoundError, NotADirectoryError):
        return "", False


REPLACE_TRIES = 3        # a program reading the caption file (a trainer) blocks replacing it meanwhile
REPLACE_WAIT = 0.05      # (on the UI thread, so short)


def _write_in_place(path, data):
    with open(path, "wb") as f:
        f.write(data)


def write_caption(img_path, text):
    """Write the caption file as a new file next to it, then swap it in. Writing in place emptied the
    old caption first, so a full disk, a dropped network share or a crash part way left it empty or
    cut short. Where swapping can't work, it is written in place as before: another program keeps the
    file open without allowing that, the folder allows writing but not deleting, or the longer
    temporary name is over Windows' path limit."""
    path = caption_path(img_path)
    if os.path.isfile(path) and not os.access(path, os.W_OK):  # read-only: no point in retrying
        raise PermissionError(13, "The caption file is read-only", path)
    data = text.encode("utf-8")
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                if data:  # or the swap can reach the disk before the text does (a power cut)
                    f.flush()
                    os.fsync(f.fileno())
        except (PermissionError, FileNotFoundError):  # can't create it (a full disk is raised)
            return _write_in_place(path, data)
        for attempt in range(REPLACE_TRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt + 1 < REPLACE_TRIES:
                    time.sleep(REPLACE_WAIT)
        _write_in_place(path, data)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _is_image(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def collect_images(paths, recursive=False, unreadable=None, listing=None):
    """Image files in `paths` (files, or folders and optionally their subfolders). Folders that can't
    be listed are skipped and appended to `unreadable` when given. When `listing` is a dict, it gets
    {normcase(folder): normcase'd names of everything in it} for every folder listed, which
    load_entries() uses to skip opening caption files that don't exist."""
    out = []
    folders = []
    for p in paths:
        if os.path.isdir(p):
            folders.append(p)
        elif _is_image(p) and os.path.isfile(p):
            out.append(p)
    while folders:
        d = folders.pop()
        names = set()
        try:
            with os.scandir(d) as it:  # on Windows the listing already says file or folder: no stat per file
                for e in it:
                    # every name, folders too: a folder called photo.txt is a caption that can't be read
                    names.add(os.path.normcase(e.name))
                    try:
                        if e.is_file():
                            if _is_image(e.name):
                                out.append(e.path)
                        elif recursive and e.is_dir(follow_symlinks=False):
                            folders.append(e.path)
                    except OSError:
                        pass
        except OSError:
            if unreadable is not None:
                unreadable.append(d)
            continue
        if listing is not None:
            listing[os.path.normcase(os.path.normpath(d))] = names
    return out


def unique_new(paths, known=()):
    """`paths` normalised, without duplicates or paths already in `known` (normcase'd keys). Windows
    paths are case-insensitive: C:\\A\\x.png and c:\\a\\X.PNG are the same file."""
    seen = set(known)
    new = []
    for p in map(os.path.normpath, paths):
        key = os.path.normcase(p)
        if key not in seen:
            seen.add(key)
            new.append(p)
    return new


def load_entries(paths, progress=None, listing=None):
    """List entries for `paths`, sorted, with their captions read. Touches only the disk, so it can
    run off the UI thread; progress(n) is called now and then with the number read so far. With the
    `listing` collect_images() filled in, captions its folders don't have aren't opened at all: a
    failed open costs about as much as reading a caption, and new datasets have none."""
    entries = []
    for i, p in enumerate(sorted(paths, key=sort_key), 1):
        folder, name = os.path.split(caption_path(p))
        names = listing.get(os.path.normcase(folder)) if listing else None
        e = {"path": p, "caption": "", "has_file": False, "status": IDLE}
        if names is None or os.path.normcase(name) in names:
            try:
                e["caption"], e["has_file"] = read_caption(p)
            except OSError as err:
                # there, but unreadable: shown in red, and Caption All leaves it alone
                e.update(has_file=True, status=ERROR, unreadable=str(err))
        entries.append(e)
        if progress and i % 500 == 0:
            progress(i)
    return entries


class _ThumbSignals(QObject):
    done = Signal(str, QImage)

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.running = set()  # paths a pool thread is decoding right now


class _ThumbJob(QRunnable):
    def __init__(self, path, signals):
        super().__init__()
        self.path, self.signals = path, signals

    def run(self):
        with self.signals.lock:
            self.signals.running.add(self.path)
        try:
            img = self._load()
        finally:
            with self.signals.lock:
                self.signals.running.discard(self.path)
        self.signals.done.emit(self.path, img)

    def _load(self):
        r = QImageReader(self.path)
        r.setAutoTransform(True)
        sz = r.size()
        if sz.isValid() and max(sz.width(), sz.height()) > THUMB_GEN:
            sz.scale(THUMB_GEN, THUMB_GEN, Qt.KeepAspectRatio)
            r.setScaledSize(sz)
        return r.read()


class ImageModel(QAbstractListModel):
    stats_changed = Signal()

    def __init__(self):
        super().__init__()
        self.items = []          # dicts: path, caption, has_file, status (+ unreadable: error text)
        self.index_of = {}       # path -> row
        self.captioned = 0       # items with a non-blank caption, kept up to date (counting on every
                                 # save scanned the whole list: clearing 5,000 of 20,000 took 75 s)
        self.cache = OrderedDict()  # path -> thumbnail QPixmap, least recently drawn first
        self.cache_bytes = 0
        self.pending = set()
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(4)
        self.sig = _ThumbSignals()
        self.sig.done.connect(self._thumb_ready)

    # -- Qt model API
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.items)

    def flags(self, index):
        return super().flags(index) | Qt.ItemIsEditable

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        it = self.items[index.row()]
        if role == Qt.DisplayRole:
            return os.path.basename(it["path"])
        if role in (Qt.EditRole, Qt.ToolTipRole):
            return it["caption"] if role == Qt.EditRole else it["path"]
        if role == PathRole:
            return it["path"]
        if role == StatusRole:
            return it["status"]
        if role == HasFileRole:
            return it["has_file"]
        if role == Qt.DecorationRole:
            p = it["path"]
            pm = self.cache.get(p)
            if pm is not None:
                self.cache.move_to_end(p)
                return pm
            if p not in self.pending:
                self.pending.add(p)
                self.pool.start(_ThumbJob(p, self.sig))
            return None
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole and index.isValid():
            return self.set_caption(index.row(), str(value))
        return False

    # -- thumbnails
    def drop_queued_thumbs(self):
        """Forget thumbnail jobs that haven't started (e.g. for cards scrolled past); the
        cards now on screen ask again when they repaint."""
        self.pool.clear()
        with self.sig.lock:
            self.pending = set(self.sig.running)

    @staticmethod
    def _pixmap_bytes(pm):
        return pm.width() * pm.height() * max(1, pm.depth()) // 8

    def _uncache(self, path):
        pm = self.cache.pop(path, None)
        if pm is not None:
            self.cache_bytes -= self._pixmap_bytes(pm)

    def _thumb_ready(self, path, img):
        self.pending.discard(path)
        if path not in self.index_of:
            return  # removed from the list while it was decoding
        pm = QPixmap.fromImage(img) if not img.isNull() else QPixmap()
        self._uncache(path)
        self.cache[path] = pm
        self.cache_bytes += self._pixmap_bytes(pm)
        while self.cache_bytes > CACHE_BYTES and len(self.cache) > 1:
            self._uncache(next(iter(self.cache)))
        i = self.index(self.index_of[path])
        self.dataChanged.emit(i, i, [Qt.DecorationRole])

    # -- mutation
    def known_keys(self):
        return {os.path.normcase(p) for p in self.index_of}

    def add_entries(self, entries):
        """Append entries from load_entries(), skipping images already in the list. Returns how many."""
        known = self.known_keys()
        fresh = []
        for e in entries:
            key = os.path.normcase(e["path"])
            if key not in known:  # another add may have listed it since this batch was scanned
                known.add(key)
                fresh.append(e)
        entries = fresh
        if not entries:
            return 0
        start = len(self.items)
        self.beginInsertRows(QModelIndex(), start, start + len(entries) - 1)
        for e in entries:
            self.index_of[e["path"]] = len(self.items)
            self.items.append(e)
            self.captioned += _has_text(e["caption"])
        self.endInsertRows()
        self.stats_changed.emit()
        return len(entries)

    def add_paths(self, paths):
        """Add image files now, on this thread (the window scans folders in the background instead)."""
        return self.add_entries(load_entries(unique_new(paths, self.known_keys())))

    def remove_rows(self, rows):
        # contiguous runs, last first, so each run is one removal instead of one per row
        rows = sorted(set(rows), reverse=True)
        k = 0
        while k < len(rows):
            hi = lo = rows[k]
            while k + 1 < len(rows) and rows[k + 1] == lo - 1:
                k += 1
                lo = rows[k]
            self.beginRemoveRows(QModelIndex(), lo, hi)
            for it in self.items[lo:hi + 1]:
                self._uncache(it["path"])
                self.captioned -= _has_text(it["caption"])
            del self.items[lo:hi + 1]
            self.endRemoveRows()
            k += 1
        self.index_of = {it["path"]: i for i, it in enumerate(self.items)}
        self.stats_changed.emit()

    def clear(self):
        self.beginResetModel()
        self.items, self.index_of, self.captioned = [], {}, 0
        self.cache.clear()
        self.cache_bytes = 0
        self.endResetModel()
        self.stats_changed.emit()

    def set_caption(self, row, text, force=False, stats=True):
        """Save `text` as the caption of `row`. stats=False leaves stats_changed to the caller, for
        many captions at once."""
        it = self.items[row]
        text = text.strip("\r\n") if text.strip() else ""
        if it.get("unreadable"):
            self._recheck_unreadable(row, text)
        if text == it["caption"] and it["has_file"] and not force:
            return False
        if text == it["caption"] and not it["has_file"] and not text and not force:
            return False  # nothing typed, don't create an empty file
        try:
            write_caption(it["path"], text)
        except OSError as e:
            it["status"] = ERROR
            self._changed(row)
            raise RuntimeError(f"Could not write {caption_path(it['path'])}: {e}")
        self._store(it, text)
        if it["status"] in (BUSY, QUEUED, ERROR):
            it["status"] = IDLE
        self._changed(row)
        if stats:
            self.stats_changed.emit()
        return True

    def _recheck_unreadable(self, row, text):
        """Before replacing a caption file that couldn't be read: it is shown empty, so the user would
        overwrite a caption they never saw. Raises unless it is gone, empty, or holds `text` already."""
        it = self.items[row]
        cp = caption_path(it["path"])
        try:
            disk, exists = read_caption(it["path"])
        except OSError as e:
            raise RuntimeError(f"Couldn't read {cp}, so it wasn't replaced: {e}")
        if exists and disk.strip() and disk != text:
            self.show_caption(row, disk)
            raise RuntimeError(f"{cp} couldn't be read when it was added, but it can now and it holds a "
                               "caption, which is now shown. Nothing was saved: edit that caption instead.")
        it.pop("unreadable", None)

    def show_caption(self, row, text):
        """Show `text`, read from the caption file, without writing it (the file changed on disk)."""
        it = self.items[row]
        self._store(it, text)
        it["status"] = IDLE
        self._changed(row)
        self.stats_changed.emit()

    def _store(self, it, text):
        self.captioned += _has_text(text) - _has_text(it["caption"])
        it["caption"], it["has_file"] = text, True
        it.pop("unreadable", None)

    def set_status(self, path, status):
        row = self.index_of.get(path)
        if row is not None and self.items[row]["status"] != status:
            self.items[row]["status"] = status
            self._changed(row)

    def set_statuses(self, rows, status, only=None):
        """set_status for many rows, with one change signal per run of rows instead of one per row
        (0.31 s for 20,000 at the start and again at the end of every big job). With `only`, rows
        whose status isn't in it are left as they are."""
        changed = []
        for row in rows:
            it = self.items[row]
            if it["status"] != status and (only is None or it["status"] in only):
                it["status"] = status
                changed.append(row)
        changed.sort()
        k = 0
        while k < len(changed):
            lo = hi = changed[k]
            while k + 1 < len(changed) and changed[k + 1] == hi + 1:
                k += 1
                hi = changed[k]
            self.dataChanged.emit(self.index(lo), self.index(hi), [StatusRole])
            k += 1

    def _changed(self, row):
        i = self.index(row)
        self.dataChanged.emit(i, i)

    def captioned_count(self):
        return self.captioned


class FilterProxy(QSortFilterProxyModel):
    def __init__(self):
        super().__init__()
        self.mode, self.text = "all", ""

    def set_filter(self, mode=None, text=None):
        if mode is not None:
            self.mode = mode
        if text is not None:
            self.text = text.lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        it = self.sourceModel().items[row]
        has = bool(it["caption"].strip())
        if self.mode == "missing" and has:
            return False
        if self.mode == "captioned" and not has:
            return False
        if self.text and self.text not in it["caption"].lower() and self.text not in os.path.basename(it["path"]).lower():
            return False
        return True
