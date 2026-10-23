"""Background thread that runs library scans and regrouping."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal

from .. import clustering, scanner
from ..db import Database
from ..faces import FaceEngine


class _Thread(QThread):
    progress = Signal(int, int, str)
    changed = Signal()
    done = Signal(str)

    def __init__(self, job: str, threshold: float | None):
        super().__init__()
        self.job = job
        self.threshold = threshold
        self.stop_requested = False

    def run(self):
        db = None
        try:
            db = Database()
            if self.job == "regroup":
                self.progress.emit(0, 0, "Regrouping faces…")
                r = clustering.regroup(db, self.threshold)
                self.changed.emit()
                self.done.emit(f"Grouping updated · {r['new_people']} groups found")
                return
            engine = FaceEngine()
            msg = scanner.full_scan(db, engine, self.progress.emit, self.changed.emit,
                                    lambda: self.stop_requested)
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
    started = Signal()
    finished = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread: _Thread | None = None
        self.queued: list[tuple[str, float | None]] = []
        self.last_message = ""

    @property
    def running(self) -> bool:
        return self.thread is not None

    def scan(self):
        self._enqueue("scan", None)

    def regroup(self, threshold: float | None = None):
        self._enqueue("regroup", threshold)

    def stop(self):
        self.queued.clear()
        if self.thread:
            self.thread.stop_requested = True

    def _enqueue(self, job: str, threshold):
        if self.thread is None:
            self._start(job, threshold)
        elif (job, threshold) not in self.queued:
            if job == "scan" and self.thread.job == "scan":
                # A scan is already running – run another afterwards to catch new changes.
                self.queued = [q for q in self.queued if q[0] != "scan"]
            self.queued.append((job, threshold))

    def _start(self, job, threshold):
        t = _Thread(job, threshold)
        t.progress.connect(self.progress)
        t.changed.connect(self.changed)
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
            job, threshold = self.queued.pop(0)
            self._start(job, threshold)
        else:
            self.finished.emit(self.last_message)

    def wait(self, ms: int = 5000):
        if self.thread:
            self.thread.stop_requested = True
            self.thread.wait(ms)
