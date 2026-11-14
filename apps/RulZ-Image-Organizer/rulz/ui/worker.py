"""Background thread that runs library scans, regrouping and folder removal."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal

from .. import clustering, scanner
from ..db import Database
from ..faces import FaceEngine


class _Thread(QThread):
    progress = Signal(int, int, str)
    changed = Signal()
    rewritten = Signal(list, list)  # photo ids, face ids whose thumbnail / crop file was (re)written
    done = Signal(str)

    def __init__(self, job: str, arg):
        super().__init__()
        self.job = job
        self.arg = arg  # regroup: match threshold (or None); remove: folder id
        self.stop_requested = False

    def run(self):
        db = None
        try:
            db = Database()
            if self.job == "regroup":
                self.progress.emit(0, 0, "Regrouping faces…")
                r = clustering.regroup(db, self.arg)
                self.changed.emit()
                self.done.emit(f"Grouping updated · {r['new_people']} groups found")
                return
            if self.job == "remove":
                self.progress.emit(0, 0, "Removing the folder from your library…")
                self.rewritten.emit(*db.remove_folder(self.arg))  # their ids may come back for new photos
                self.changed.emit()
                self.done.emit("Folder removed")
                return
            # A factory: the models only load if old faces still need expressions or poses.
            msg = scanner.full_scan(db, FaceEngine, self.progress.emit, self.changed.emit,
                                    lambda: self.stop_requested, self.rewritten.emit)
            self.done.emit(msg)
        except scanner.Cancelled:
            self.changed.emit()
            self.done.emit("Scan paused")
        except Exception as e:  # keep the app alive whatever happens
            traceback.print_exc()
            self.done.emit(f"Scan failed: {e}")
        finally:
            if db is not None:
                db.close()


class ScanController(QObject):
    progress = Signal(int, int, str)
    changed = Signal()
    rewritten = Signal(list, list)
    started = Signal()
    finished = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread: _Thread | None = None
        self.queued: list[tuple[str, object]] = []
        self.last_message = ""

    @property
    def running(self) -> bool:
        return self.thread is not None

    @property
    def removing(self) -> set[int]:
        """Folder ids whose removal is queued or running."""
        ids = {arg for job, arg in self.queued if job == "remove"}
        if self.thread is not None and self.thread.job == "remove":
            ids.add(self.thread.arg)
        return ids

    def scan(self):
        self._enqueue("scan", None)

    def regroup(self, threshold: float | None = None):
        self._enqueue("regroup", threshold)

    def remove_folder(self, folder_id: int):
        """Remove a library folder on the background thread: a running scan stops first (it may be writing
        that folder's photos), then scanning carries on with the other folders. A queued regroup stays."""
        self.queued = [q for q in self.queued if q[0] != "scan"]
        if self.thread is not None and self.thread.job == "scan":
            self.thread.stop_requested = True
        self._enqueue("remove", folder_id)
        self._enqueue("scan", None)

    def stop(self):
        # Pause / close drop queued scans, but a removal the user confirmed still happens.
        self.queued = [q for q in self.queued if q[0] == "remove"]
        if self.thread:
            self.thread.stop_requested = True

    def _enqueue(self, job: str, arg):
        if self.thread is None:
            self._start(job, arg)
        elif (job, arg) not in self.queued:
            if job == "scan" and self.thread.job == "scan":
                # A scan is already running – run another afterwards to catch new changes.
                self.queued = [q for q in self.queued if q[0] != "scan"]
            self.queued.append((job, arg))

    def _start(self, job, arg):
        t = _Thread(job, arg)
        t.progress.connect(self.progress)
        t.changed.connect(self.changed)
        t.rewritten.connect(self.rewritten)
        t.done.connect(self._on_done)
        t.finished.connect(self._on_finished)
        self.thread = t
        self.started.emit()
        t.start()

    def _on_done(self, msg: str):
        self.last_message = msg

    def _on_finished(self):
        t, self.thread = self.thread, None
        if t is not None:
            t.deleteLater()
        if self.queued:
            job, arg = self.queued.pop(0)
            self._start(job, arg)
        else:
            self.finished.emit(self.last_message)

    def wait(self, ms: int = 5000):
        if self.thread:
            self.thread.stop_requested = True
            self.thread.wait(ms)
