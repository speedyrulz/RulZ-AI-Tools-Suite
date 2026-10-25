"""Image list model, lazy thumbnail loading and caption .txt persistence."""
import os
import re
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
CACHE_MAX = 1500

# status values
IDLE, QUEUED, BUSY, ERROR = "idle", "queued", "busy", "error"


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(p))]


def caption_path(img_path):
    return str(Path(img_path).with_suffix(".txt"))


def read_caption(img_path):
    cp = caption_path(img_path)
    try:
        with open(cp, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip("﻿"), True
    except OSError:
        return "", False


def write_caption(img_path, text):
    with open(caption_path(img_path), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def collect_images(paths, recursive=False):
    out = []
    for p in paths:
        if os.path.isdir(p):
            if recursive:
                for root, _, files in os.walk(p):
                    out += [os.path.join(root, f) for f in files if Path(f).suffix.lower() in IMAGE_EXTS]
            else:
                out += [os.path.join(p, f) for f in os.listdir(p)
                        if Path(f).suffix.lower() in IMAGE_EXTS and os.path.isfile(os.path.join(p, f))]
        elif Path(p).suffix.lower() in IMAGE_EXTS and os.path.isfile(p):
            out.append(p)
    return out


class _ThumbSignals(QObject):
    done = Signal(str, QImage)


class _ThumbJob(QRunnable):
    def __init__(self, path, signals):
        super().__init__()
        self.path, self.signals = path, signals

    def run(self):
        r = QImageReader(self.path)
        r.setAutoTransform(True)
        sz = r.size()
        if sz.isValid() and max(sz.width(), sz.height()) > THUMB_GEN:
            sz.scale(THUMB_GEN, THUMB_GEN, Qt.KeepAspectRatio)
            r.setScaledSize(sz)
        img = r.read()
        self.signals.done.emit(self.path, img)


class ImageModel(QAbstractListModel):
    stats_changed = Signal()

    def __init__(self):
        super().__init__()
        self.items = []          # dicts: path, caption, has_file, status
        self.index_of = {}       # path -> row
        self.cache = OrderedDict()
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
    def _thumb_ready(self, path, img):
        self.pending.discard(path)
        pm = QPixmap.fromImage(img) if not img.isNull() else QPixmap()
        self.cache[path] = pm
        while len(self.cache) > CACHE_MAX:
            self.cache.popitem(last=False)
        row = self.index_of.get(path)
        if row is not None:
            i = self.index(row)
            self.dataChanged.emit(i, i, [Qt.DecorationRole])

    # -- mutation
    def add_paths(self, paths):
        new = [p for p in dict.fromkeys(os.path.normpath(p) for p in paths) if p not in self.index_of]
        if not new:
            return 0
        new.sort(key=natural_key)
        entries = []
        for p in new:
            cap, has = read_caption(p)
            entries.append({"path": p, "caption": cap, "has_file": has, "status": IDLE})
        start = len(self.items)
        self.beginInsertRows(QModelIndex(), start, start + len(entries) - 1)
        for e in entries:
            self.index_of[e["path"]] = len(self.items)
            self.items.append(e)
        self.endInsertRows()
        self.stats_changed.emit()
        return len(entries)

    def remove_rows(self, rows):
        for r in sorted(set(rows), reverse=True):
            self.beginRemoveRows(QModelIndex(), r, r)
            self.cache.pop(self.items[r]["path"], None)
            del self.items[r]
            self.endRemoveRows()
        self.index_of = {it["path"]: i for i, it in enumerate(self.items)}
        self.stats_changed.emit()

    def clear(self):
        self.beginResetModel()
        self.items, self.index_of = [], {}
        self.cache.clear()
        self.endResetModel()
        self.stats_changed.emit()

    def set_caption(self, row, text, force=False):
        it = self.items[row]
        text = text.strip("\r\n") if text.strip() else ""
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
        it["caption"], it["has_file"] = text, True
        if it["status"] in (BUSY, QUEUED, ERROR):
            it["status"] = IDLE
        self._changed(row)
        self.stats_changed.emit()
        return True

    def set_status(self, path, status):
        row = self.index_of.get(path)
        if row is not None and self.items[row]["status"] != status:
            self.items[row]["status"] = status
            self._changed(row)

    def _changed(self, row):
        i = self.index(row)
        self.dataChanged.emit(i, i)

    def captioned_count(self):
        return sum(1 for it in self.items if it["caption"].strip())


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
