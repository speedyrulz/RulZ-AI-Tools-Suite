"""The clip list: selections kept for a batch export, one list per video.

ClipEntry      one clip: its range and everything that shapes its picture (rotation, flip, crop,
               scale, colour match, distortion repair), copied when it was added
ClipList       the entries of one video, stored in a small JSON file so opening the video again
               brings them back
BatchExport    exports entries one after another in the background
ClipListPanel  the list beside the video
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel, QMenu,
                               QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from . import theme
from .colormatch import write_cube_1d
from .defaults import SETTINGS_ENV
from .export import (ExportSettings, default_output_path, lossless_can_rotate, lossless_end_ok,
                     lossless_start_ok, make_export_job, unique_path)
from .probe import VideoInfo

try:
    from .repair.export import RepairExport
    from .repair.pipeline import (PRESET_KIND, RepairSettings, RepairState, describe as describe_repair,
                                  from_json as corrections_from_json)
except ImportError:  # OpenCV missing: clips that need the repair fail with a message, the rest work
    RepairExport = RepairSettings = RepairState = describe_repair = corrections_from_json = None
    PRESET_KIND = ""

LIST_KIND = "rulz-clip-list"
LIST_VERSION = 1
STALE_DAYS = 90          # a stored list whose video has been gone this long is deleted

PENDING, QUEUED, RUNNING, DONE, FAILED = "pending", "queued", "running", "done", "failed"
STATUS_TEXT = {PENDING: "", QUEUED: "queued", RUNNING: "exporting", DONE: "saved", FAILED: "failed"}
NO_REPAIR = "Distortion repair needs OpenCV, which is not installed."


def short_time(t: float) -> str:
    """m:ss.mmm for the list's narrow columns (hours are counted in the minutes)."""
    t = max(round(t, 3), 0.0)
    return f"{int(t // 60)}:{t % 60:06.3f}"


# ------------------------------------------------------------------ entries
@dataclass(eq=False)   # (identity, not value: the list, the queue and the panel hold the same objects)
class ClipEntry:
    start: float                 # absolute pts of the first frame
    end: float                   # exclusive, as the Out mark
    rotation: int = 0            # the app's clockwise rotation, on top of the source's own flag
    hflip: bool = False
    crop: tuple[int, int, int, int] | None = None    # (x, y, w, h) in the frame as shown
    out_size: tuple[int, int] | None = None          # "Scale to" width and height
    color_tables: np.ndarray | None = None           # (3, 256) uint8 from the colour match
    color_desc: str = ""
    repair: object | None = None                     # RepairSettings
    repair_data: dict | None = None                  # the same as stored; kept when OpenCV is missing
    status: str = PENDING
    out_path: str = ""           # the file written (status DONE)
    error: str = ""              # what went wrong (status FAILED)
    progress: float = 0.0        # 0..1 while RUNNING

    @property
    def needs_reencode(self) -> bool:
        return (self.crop is not None or self.out_size is not None or self.hflip
                or self.color_tables is not None or self.repair is not None or self.repair_data is not None)

    def _repair_dict(self) -> dict | None:
        if self.repair is not None:
            s = self.repair
            return {"corrections": s.corrections.to_dict(), "base_rotation": s.base_rotation,
                    "base_flip": s.base_flip}
        return self.repair_data

    def same_clip(self, other: "ClipEntry") -> bool:
        """The same frames with the same picture settings (whatever the status)."""
        if abs(self.start - other.start) > 1e-4 or abs(self.end - other.end) > 1e-4:
            return False
        if (self.rotation, self.hflip, self.crop, self.out_size) != (
                other.rotation, other.hflip, other.crop, other.out_size):
            return False
        a, b = self.color_tables, other.color_tables
        if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
            return False
        return self._repair_dict() == other._repair_dict()

    def to_dict(self) -> dict:
        return {
            "start": self.start, "end": self.end, "rotation": self.rotation, "hflip": self.hflip,
            "crop": list(self.crop) if self.crop else None,
            "out_size": list(self.out_size) if self.out_size else None,
            "color": ({"tables": np.asarray(self.color_tables, dtype=np.uint8).tolist(), "desc": self.color_desc}
                      if self.color_tables is not None else None),
            "repair": self._repair_dict(),
            # a clip that was waiting or being written when the app closed is simply not exported yet
            "status": self.status if self.status in (DONE, FAILED) else PENDING,
            "out_path": self.out_path, "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClipEntry":
        """Raises for anything that is not a usable entry: ValueError, TypeError or KeyError,
        and OverflowError for a number JSON allows but a frame can't have (Infinity). Callers
        that read stored data catch every Exception."""
        start, end = float(d["start"]), float(d["end"])
        if not (np.isfinite(start) and np.isfinite(end) and end > start):
            raise ValueError("empty clip")
        rotation = int(d.get("rotation", 0))
        if rotation not in (0, 90, 180, 270):
            raise ValueError("bad rotation")
        hflip = d.get("hflip", False)
        if not isinstance(hflip, bool):
            raise ValueError("bad flip")   # ("false" as text would count as True)
        e = cls(start, end, rotation, hflip)
        crop, out_size = d.get("crop"), d.get("out_size")
        if crop is not None:
            if not isinstance(crop, (list, tuple)) or len(crop) != 4:
                raise ValueError("bad crop")   # (a string would be read digit by digit)
            x, y, w, h = (int(v) for v in crop)
            if x < 0 or y < 0 or w < 2 or h < 2:
                raise ValueError("bad crop")
            e.crop = (x, y, w, h)
        if out_size is not None:
            if not isinstance(out_size, (list, tuple)) or len(out_size) != 2:
                raise ValueError("bad output size")
            w, h = (int(v) for v in out_size)
            if w < 2 or h < 2:
                raise ValueError("bad output size")
            e.out_size = (w, h)
        color = d.get("color")
        if color is not None:
            tables = np.asarray(color["tables"])
            if tables.shape != (3, 256) or tables.min() < 0 or tables.max() > 255:
                raise ValueError("bad colour tables")
            e.color_tables = tables.astype(np.uint8)
            e.color_desc = str(color.get("desc", ""))
        rep = d.get("repair")
        if rep is not None:
            if not isinstance(rep, dict) or not isinstance(rep.get("corrections"), dict):
                raise ValueError("bad repair settings")
            base_rotation = int(rep.get("base_rotation", 0))
            base_flip = rep.get("base_flip", False)
            if base_rotation not in (0, 90, 180, 270) or not isinstance(base_flip, bool):
                raise ValueError("bad repair orientation")
            e.repair_data = {"corrections": rep["corrections"], "base_rotation": base_rotation,
                             "base_flip": base_flip}
            if RepairSettings is not None:
                # (checked the way a preset file is: nothing unusable reaches an export)
                c = corrections_from_json(json.dumps({"kind": PRESET_KIND, "corrections": rep["corrections"]}))
                e.repair = RepairSettings(c, base_rotation, e.repair_data["base_flip"])
        status = d.get("status", PENDING)
        e.status = status if status in (DONE, FAILED) else PENDING
        e.out_path = str(d.get("out_path", "") or "")
        e.error = str(d.get("error", "") or "")
        return e

    def describe(self) -> str:
        """What is applied to the clip's picture, one item per line ('' for a plain cut)."""
        parts = []
        if self.rotation:
            parts.append(f"Rotated {self.rotation}°")
        if self.hflip:
            parts.append("Flipped horizontally")
        if self.crop:
            x, y, w, h = self.crop
            parts.append(f"Crop {w}×{h} at ({x}, {y})")
        if self.out_size:
            parts.append(f"Scaled to {self.out_size[0]}×{self.out_size[1]}")
        if self.color_tables is not None:
            parts.append("Colour matched" + (f": {self.color_desc}" if self.color_desc else ""))
        if self.repair is not None and describe_repair is not None:
            parts.append(f"Distortion repair: {describe_repair(self.repair.corrections)}")
        elif self.repair_data is not None:
            parts.append("Distortion repair")
        return "\n".join(parts)

    def tags(self) -> str:
        """The same, as a few short words for the list's column."""
        t = []
        if self.rotation or self.hflip:
            t.append("turned")
        if self.crop:
            t.append("crop")
        if self.color_tables is not None:
            t.append("colour")
        if self.repair is not None or self.repair_data is not None:
            t.append("repair")
        return ", ".join(t)


# ------------------------------------------------------------------ storage
def store_dir() -> str:
    """Where the lists are kept: beside the settings file when SIMPLECUT_SETTINGS_INI names one
    (tests, a fresh profile), else in the user's local app data."""
    ini = os.environ.get(SETTINGS_ENV)
    if ini:
        return os.path.join(os.path.dirname(os.path.abspath(ini)), "clip-lists")
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "SimpleCut", "clip-lists")


def video_key(path: str) -> str:
    """One key per video file, whatever the letter case or the way its path was typed."""
    return os.path.normcase(os.path.abspath(path))


def store_path(video: str) -> str:
    return os.path.join(store_dir(), hashlib.sha1(video_key(video).encode("utf-8")).hexdigest()[:24] + ".json")


def _file_stamp(path: str) -> tuple[int, float] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_size, st.st_mtime


class ClipList:
    """The clips kept for one video. Every change is written to the store at once (save())."""

    def __init__(self, video: str):
        self.video = os.path.abspath(video)
        # The file the entries were made on (size, modified time). Saved with them, and not
        # refreshed later: a list must not follow a different file put in the video's place.
        self.stamp = _file_stamp(self.video)
        self.entries: list[ClipEntry] = []
        self.dropped = 0       # stored entries load() could not use
        self.replaced = False  # load() found a list made for another file of this name
        self.save_error = ""   # why the last save() failed ('' when it worked)
        # load() left something of the stored file out: the next save() keeps that file as
        # "<name>.old" instead of writing over the only copy.
        self._keep_old = False

    def note(self) -> str:
        """What load() could not bring back, for the status bar ('' when everything was read)."""
        if self.replaced:
            return "the clip list stored for this video belongs to another version of the file and was not loaded"
        if self.dropped:
            n = self.dropped
            return (f"{n} stored clip{'s' if n != 1 else ''} could not be read and "
                    f"{'were' if n != 1 else 'was'} left out")
        return ""

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @classmethod
    def load(cls, video: str) -> "ClipList":
        """The stored list of this video, or an empty one. A list made for a different file of
        the same name (another size or date) is not used: its times would point at other frames."""
        lst = cls(video)
        try:
            with open(store_path(video), encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return lst
        if not isinstance(d, dict) or d.get("kind") != LIST_KIND or not isinstance(d.get("clips"), list):
            return lst
        stamp, meta = lst.stamp, d.get("video")
        try:
            same = (stamp is not None and int(meta["size"]) == stamp[0]
                    and abs(float(meta["mtime"]) - stamp[1]) < 2.0)   # (FAT keeps 2 s steps)
        except Exception:  # noqa: BLE001 - a hand-edited or damaged file: any shape, any number
            same = False
        if not same:
            lst.replaced = lst._keep_old = bool(d["clips"])
            return lst
        for c in d["clips"]:
            try:
                lst.entries.append(ClipEntry.from_dict(c))
            except Exception:  # noqa: BLE001 - one unusable entry must not cost the others
                lst.dropped += 1
        lst._keep_old = lst.dropped > 0
        return lst

    def save(self) -> bool:
        """Write the list (an empty one removes its file). False when it could not be written;
        the list then lives on for this session only."""
        path = store_path(self.video)
        tmp = f"{path}.{os.getpid()}.tmp"
        self.save_error = ""
        try:
            if self._keep_old:
                self._keep_old = False
                if os.path.exists(path):
                    os.replace(path, path + ".old")
            if not self.entries:
                if os.path.exists(path):
                    os.remove(path)
                return True
            stamp = self.stamp
            data = {"kind": LIST_KIND, "version": LIST_VERSION,
                    "video": {"path": self.video, "size": stamp[0] if stamp else -1,
                              "mtime": stamp[1] if stamp else 0.0},
                    "clips": [e.to_dict() for e in self.entries]}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, path)   # (never a half-written list)
            return True
        except (OSError, TypeError, ValueError) as e:   # (the batch saves between clips: it must go on)
            self.save_error = str(e) or type(e).__name__
            _remove(tmp)
            return False

    def matches_file(self) -> bool:
        """The video is still the file this list was made on."""
        now = _file_stamp(self.video)
        return (now is not None and self.stamp is not None and now[0] == self.stamp[0]
                and abs(now[1] - self.stamp[1]) < 2.0)

    def busy(self) -> bool:
        return any(e.status in (QUEUED, RUNNING) for e in self.entries)

    def find(self, entry: ClipEntry) -> ClipEntry | None:
        for e in self.entries:
            if e.same_clip(entry):
                return e
        return None

    def waiting(self) -> list[ClipEntry]:
        """Entries "Export list" takes: not saved yet and not already queued."""
        return [e for e in self.entries if e.status in (PENDING, FAILED)]


def clean_store(now: float | None = None):
    """Delete stored lists whose video no longer exists and that have not changed for STALE_DAYS
    (a list for a video on a drive that is merely unplugged is kept that long)."""
    d = store_dir()
    now = time.time() if now is None else now
    try:
        names = os.listdir(d)
    except OSError:
        return
    for name in names:
        p = os.path.join(d, name)
        try:
            if name.endswith(".tmp"):
                if now - os.path.getmtime(p) > 86400:
                    os.remove(p)
                continue
            if name.endswith(".old"):   # a list set aside by save(): kept as long as an orphaned one
                if now - os.path.getmtime(p) > STALE_DAYS * 86400:
                    os.remove(p)
                continue
            if not name.endswith(".json") or now - os.path.getmtime(p) < STALE_DAYS * 86400:
                continue
            with open(p, encoding="utf-8") as f:
                video = json.load(f)["video"]["path"]
            if not os.path.exists(video):
                os.remove(p)
        except (OSError, ValueError, KeyError, TypeError):
            continue


# ------------------------------------------------------------------ one entry's export
def entry_settings(info: VideoInfo, e: ClipEntry, base: ExportSettings, allow_lossless: bool) -> ExportSettings:
    """The export settings of one entry: the encode options of `base` (Quick export's) with the
    entry's own picture settings. Lossless when allowed and the clip can be cut that way.
    Raises ValueError when the entry can't be exported."""
    if e.repair_data is not None and (e.repair is None or RepairExport is None):
        raise ValueError(NO_REPAIR)
    lossless = (allow_lossless and not e.needs_reencode
                and lossless_start_ok(info, e.start) and lossless_end_ok(info, e.end)
                and (not e.rotation or lossless_can_rotate(info)))
    s = replace(base, mode="lossless" if lossless else "nvenc", rotation=e.rotation, hflip=e.hflip,
                crop=e.crop, out_size=e.out_size, lut_path=None, repair=None)
    if lossless:
        return s
    if e.color_tables is not None:
        s.lut_path = write_cube_1d(e.color_tables)
    if e.repair is not None:
        state = RepairState.build(e.repair, info.width, info.height, info.rotation)
        if not state.pipeline.changes_nothing:
            s.repair = RepairExport(state, e.rotation, e.hflip, info.rotation, (info.width, info.height))
    return s


def entry_output_path(info: VideoInfo, e: ClipEntry, s: ExportSettings, out_dir: str | None) -> str:
    return unique_path(default_output_path(info, e.start, e.end, s.mode, out_dir, cropped=e.crop is not None,
                                           repaired=s.repair is not None, rotation=e.rotation))


def _remove(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


# ------------------------------------------------------------------ batch
@dataclass(eq=False)
class _Item:
    info: VideoInfo
    clips: ClipList
    entry: ClipEntry
    base: ExportSettings
    allow_lossless: bool
    out_dir: str | None


class BatchExport(QObject):
    """Exports queued entries one at a time. It keeps what it needs of each entry's video, so
    another video can be opened (or this one closed) while it works."""
    changed = Signal()                  # an entry's status or progress changed
    clip_started = Signal(str, int, int)   # output file name, number of this clip, clips in this run
    progress = Signal(float)            # of the clip being written, 0..1
    exported = Signal(str)              # a clip was written: its path
    finished = Signal(int, int, bool)   # saved, failed, cancelled

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: deque[_Item] = deque()
        self._item: _Item | None = None
        self._job = None
        self._out = ""
        self._saved = self._failed = self._count = 0
        self._cancelling = False
        self.first_failure = ""    # of the last run: the first failed clip's message

    @property
    def running(self) -> bool:
        return self._item is not None

    def add(self, info: VideoInfo, clips: ClipList, entries: list[ClipEntry], base: ExportSettings,
            allow_lossless: bool, out_dir: str | None) -> int:
        """Queue entries (those already queued or running are skipped) and start if idle."""
        n = 0
        for e in entries:
            if e.status in (QUEUED, RUNNING):
                continue
            e.status, e.error, e.progress = QUEUED, "", 0.0
            self._queue.append(_Item(info, clips, e, replace(base), allow_lossless, out_dir))
            n += 1
        if n:
            self.changed.emit()
            if not self.running:
                self._start_run()
        return n

    def discard(self, entries) -> None:
        """Take entries that are still waiting out of the queue (the one being written stays)."""
        gone = {id(e) for e in entries}
        for it in [it for it in self._queue if id(it.entry) in gone]:
            self._queue.remove(it)
            it.entry.status = PENDING

    def cancel(self):
        """Stop the clip being written and leave everything not yet written for later."""
        if not self.running:
            return
        self._cancelling = True
        for it in self._queue:
            it.entry.status = PENDING
        self._queue.clear()
        if self._job is not None:
            self._job.cancel()

    def wait(self, msecs: int = 30000) -> bool:
        """Block until the running job has ended (after cancel(): `finished` is emitted first)."""
        job = self._job
        return job.wait(msecs) if job is not None else True

    # ---- internals
    def _start_run(self):
        self._saved = self._failed = self._count = 0
        self.first_failure = ""
        self._next()

    def _next(self):
        while self._queue:
            it = self._item = self._queue.popleft()
            e = it.entry
            self._count += 1
            try:
                if not os.path.isfile(it.info.path):
                    raise ValueError("The source video is no longer there.")
                s = entry_settings(it.info, e, it.base, it.allow_lossless)
                out = entry_output_path(it.info, e, s, it.out_dir)
                self._job = make_export_job(it.info, e.start, e.end, out, s, self)
            except Exception as err:  # noqa: BLE001 - whatever stops one clip must not stop the rest
                self._job = None
                self._fail(it, str(err) or type(err).__name__)
                continue
            self._out = out
            e.status, e.progress = RUNNING, 0.0
            # (bound methods, not a lambda holding this object and the job: that kept each alive
            # through the other, and deleting the job then destroyed its parent from inside the
            # job's own destructor)
            self._job.progress.connect(self._on_progress)
            self._job.finished.connect(self._on_finished)
            self.clip_started.emit(os.path.basename(out), self._count, self._count + len(self._queue))
            self.changed.emit()
            self._job.start()
            return
        self._finish_run(False)

    def _finish_run(self, cancelled: bool):
        self._item = None
        self._cancelling = False
        self.changed.emit()
        self.finished.emit(self._saved, self._failed, cancelled)
        # Clips queued after Cancel was pressed, while the stopped job was still ending, are a
        # run of their own (they were not part of what was cancelled).
        if cancelled and self._queue and not self.running:
            self._start_run()

    def _fail(self, it: _Item, message: str):
        it.entry.status, it.entry.error, it.entry.progress = FAILED, message, 0.0
        self._failed += 1
        if not self.first_failure:
            self.first_failure = message
        it.clips.save()

    def _on_progress(self, f: float):
        if self._item is not None:
            self._item.entry.progress = f
            self.progress.emit(f)

    def _on_finished(self, ok: bool, log: str):
        it, job = self._item, self._job
        if it is None or job is None:
            return   # (a job reports once; nothing is being written)
        self._job = None
        e = it.entry
        e.progress = 0.0
        cancelled = job.cancelled or self._cancelling
        if ok:
            # (also when Cancel came just after the clip was complete: the file is whole, and it
            # is kept rather than deleted and written again later)
            e.status, e.out_path, e.error = DONE, self._out, getattr(job, "warning", "") or ""
            self._saved += 1
            it.clips.save()
            self.exported.emit(self._out)
        elif cancelled:
            _remove(self._out)   # no half-written clip left behind
            e.status = PENDING
        else:
            _remove(self._out)
            self._fail(it, f"ffmpeg failed.\n\nCommand:\n{job.cmd_text}\n\nLog:\n{log}")
        job.deleteLater()   # (a repair job holds the full-size remap tables)
        if cancelled:
            self._finish_run(True)
        else:
            self._next()


# ------------------------------------------------------------------ panel
class ClipListPanel(QFrame):
    """The list beside the video. It never takes keyboard focus, so the single-key shortcuts
    keep working; everything is done with the mouse and the buttons."""
    goRequested = Signal(object)        # entry: show this clip
    removeRequested = Signal(object)    # list of entries
    resetRequested = Signal(object)     # list of entries: export them again
    exportRequested = Signal()
    clearRequested = Signal()

    COLUMNS = ("#", "In", "Out", "Length", "Applied", "Status")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("clipPanel")
        self.setFixedWidth(380)
        self._entries: list[ClipEntry] = []
        self._t0 = 0.0

        self.title = QLabel("Clip list")
        self.title.setObjectName("accent")
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(self.COLUMNS))
        self.tree.setHeaderLabels(list(self.COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setFocusPolicy(Qt.NoFocus)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemDoubleClicked.connect(lambda item, _c: self._go(item))
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        hdr = self.tree.header()
        hdr.setStretchLastSection(True)
        for i in range(len(self.COLUMNS) - 1):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        def btn(text, tip, signal):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(signal)
            return b

        self.b_export = btn("Export list", "Export every clip in the list that is not saved yet, one after "
                            "another, with the Quick export settings  [Ctrl+Shift+A]", self.exportRequested.emit)
        self.b_export.setObjectName("primary")
        self.b_remove = btn("Remove", "Remove the selected clips from the list (exported files are kept)",
                            lambda: self.removeRequested.emit(self.selected()))
        self.b_clear = btn("Clear", "Empty the list (exported files are kept)", self.clearRequested.emit)
        row = QHBoxLayout()
        row.addWidget(self.b_export, 1)
        row.addWidget(self.b_remove)
        row.addWidget(self.b_clear)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.title)
        lay.addWidget(self.tree, 1)
        lay.addLayout(row)
        self._update_buttons()

    def selected(self) -> list[ClipEntry]:
        return [self._entries[self.tree.indexOfTopLevelItem(i)] for i in self.tree.selectedItems()]

    def set_clips(self, entries: list[ClipEntry], t0: float = 0.0):
        """Show these entries; times are listed from t0, the video's first frame."""
        keep = {id(e) for e in self.selected()}
        self._entries, self._t0 = list(entries), t0
        self.tree.blockSignals(True)
        self.tree.clear()
        for n, e in enumerate(self._entries, 1):
            item = QTreeWidgetItem(self.tree)
            item.setText(0, str(n))
            item.setText(1, short_time(e.start - t0))
            item.setText(2, short_time(e.end - t0))
            item.setText(3, f"{e.end - e.start:.2f} s")
            item.setText(4, e.tags())
            self._show_status(item, e)
            if id(e) in keep:
                item.setSelected(True)
        self.tree.blockSignals(False)
        waiting = sum(1 for e in self._entries if e.status != DONE)
        self.title.setText(f"Clip list   {len(self._entries)} clip{'s' if len(self._entries) != 1 else ''}"
                           + (f", {waiting} to export" if waiting else ", all saved"))
        self._update_buttons()

    def update_status(self):
        """Refresh the status column only (an export moved on); the rows stay as they are."""
        for i, e in enumerate(self._entries):
            item = self.tree.topLevelItem(i)
            if item is not None:
                self._show_status(item, e)
        self._update_buttons()

    def _show_status(self, item: QTreeWidgetItem, e: ClipEntry):
        text = STATUS_TEXT.get(e.status, e.status)
        if e.status == RUNNING:
            text = f"{int(e.progress * 100)} %"
        item.setText(5, text)
        colour = {DONE: theme.Q_MUTED, FAILED: theme.Q_ACCENT_TEXT, RUNNING: theme.Q_ACCENT_TEXT}.get(e.status)
        item.setForeground(5, QBrush(colour) if colour is not None else QBrush())
        tip = e.describe() or "Plain cut"
        if e.status == DONE and e.out_path:
            tip += f"\n\nSaved: {e.out_path}" + (f"\n{e.error}" if e.error else "")
        elif e.status == FAILED and e.error:
            tip += f"\n\nFailed: {e.error[:1500]}"
        for c in range(len(self.COLUMNS)):
            item.setToolTip(c, tip)

    def _update_buttons(self):
        sel = self.selected()
        self.b_export.setEnabled(any(e.status in (PENDING, FAILED) for e in self._entries))
        self.b_remove.setEnabled(any(e.status != RUNNING for e in sel))
        self.b_clear.setEnabled(any(e.status != RUNNING for e in self._entries))

    def _go(self, item):
        i = self.tree.indexOfTopLevelItem(item)
        if 0 <= i < len(self._entries):
            self.goRequested.emit(self._entries[i])

    def _menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        if not item.isSelected():
            self.tree.clearSelection()
            item.setSelected(True)
        sel = self.selected()
        i = self.tree.indexOfTopLevelItem(item)
        # (looked up now: while the menu is open an export can finish and rebuild the rows, and
        # `item` is then a deleted object)
        clicked = self._entries[i] if 0 <= i < len(self._entries) else None
        m = QMenu(self)
        a_go = m.addAction("Go to clip")
        a_again = m.addAction("Export again")
        a_again.setEnabled(any(e.status in (DONE, FAILED) for e in sel))
        a_remove = m.addAction("Remove")
        a_remove.setEnabled(any(e.status != RUNNING for e in sel))
        chosen = m.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is a_go:
            if clicked is not None:
                self.goRequested.emit(clicked)
        elif chosen is a_again:
            self.resetRequested.emit(sel)
        elif chosen is a_remove:
            self.removeRequested.emit(sel)
