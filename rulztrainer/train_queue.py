"""The WebUI's training queue: runs waiting for the current one to finish, started one after another.

The queue lives in output/_queue.json (the waiting configs in full, a paused flag and the last few started or failed
runs), so it survives a reload or a restarted WebUI. The WebUI advances it: when no job runs and the queue is not
paused, the first waiting run starts. A WebUI that starts with runs waiting and nothing running pauses the queue
(on_startup) instead of starting one by surprise; Resume goes on. Kill pauses it too (stopping everything is what a
kill usually means); Stop and a failed run go on to the next.

Other processes edit the queue too (scripts/vast/rulz_agent.py adds runs over ssh), so a lock file beside it guards the
file across processes: advance holds the queue while a run starts, and a run added in that moment would otherwise be
written over and lost without a word."""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_FILE = ROOT / "output" / "_queue.json"
DONE_KEPT = 10  # started / failed runs kept for the queue's "recently" list
REPLACE_TRIES = 6  # on Windows a reader (a virus scan, the search indexer) can briefly hold the queue file open


def _lock_file(f) -> None:
    """Wait for the OS lock on an open file. The OS drops it if the holder dies, so a crash can't leave it stuck."""
    if sys.platform == "win32":
        import msvcrt

        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)  # gives up after 10 s
                return
            except OSError:
                continue
    else:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_EX)


class QueueLock:
    """Re-entrant across this process's threads and exclusive across processes. A run started while it is held does
    not inherit it: Python's files aren't passed on to child processes."""

    def __init__(self, path: Path):
        self.path = path
        self.threads = threading.RLock()
        self.depth = 0
        self.file = None

    def __enter__(self):
        self.threads.acquire()
        self.depth += 1
        if self.depth == 1:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.file = open(self.path, "a+b")  # noqa: SIM115
                _lock_file(self.file)
            except OSError:  # a read-only folder and the like: the queue's own writes will say so
                self.file = None
        return self

    def __exit__(self, *exc):
        self.depth -= 1
        if self.depth == 0 and self.file is not None:
            self.file.close()  # releases the lock
            self.file = None
        self.threads.release()


class TrainQueue:
    def __init__(self, path: Path = QUEUE_FILE):
        self.path = Path(path)
        self.lock = QueueLock(self.path.with_suffix(".lock"))

    # ---- storage
    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.setdefault("waiting", [])
        data.setdefault("paused", False)
        data.setdefault("recent", [])
        return data

    def _write(self, data: dict) -> None:
        """The queue written whole (callers hold the lock, so one temporary name is enough). On Windows a reader
        holding the file briefly makes the replace fail: it is tried again, as atomic._replace does (atomic loads
        PyTorch, and rulz_fleet.py imports this module for QueueLock without it)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            for attempt in range(REPLACE_TRIES):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    if attempt == REPLACE_TRIES - 1:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    def snapshot(self) -> dict:
        """{"waiting": [entry], "paused": bool, "recent": [entry with "started" or "error"]}."""
        with self.lock:
            return self._read()

    # ---- editing
    def add(self, config: dict, label: str = "") -> dict:
        """Queue a run (its config as TrainConfig.to_dict()). Returns the entry."""
        entry = {"id": uuid.uuid4().hex[:10], "job": config.get("job_name", ""), "label": label,
                 "added": time.time(), "config": config}
        with self.lock:
            data = self._read()
            data["waiting"].append(entry)
            self._write(data)
        return entry

    def remove(self, entry_id: str) -> bool:
        with self.lock:
            data = self._read()
            keep = [e for e in data["waiting"] if e["id"] != entry_id]
            if len(keep) == len(data["waiting"]):
                return False
            data["waiting"] = keep
            self._write(data)
            return True

    def move(self, entry_id: str, delta: int) -> bool:
        """Move a waiting run up (delta < 0) or down the queue."""
        with self.lock:
            data = self._read()
            ids = [e["id"] for e in data["waiting"]]
            if entry_id not in ids:
                return False
            i = ids.index(entry_id)
            j = max(0, min(len(ids) - 1, i + delta))
            if i == j:
                return False
            data["waiting"].insert(j, data["waiting"].pop(i))
            self._write(data)
            return True

    def clear(self) -> None:
        with self.lock:
            data = self._read()
            data["waiting"] = []
            self._write(data)

    def set_paused(self, paused: bool) -> None:
        """Pause or resume (the next run then starts as soon as nothing runs)."""
        with self.lock:
            data = self._read()
            data["paused"] = bool(paused)
            self._write(data)

    def on_startup(self, running: bool) -> bool:
        """A WebUI starting with runs waiting and no job running pauses the queue (the user resumes it). True when
        it paused."""
        with self.lock:
            data = self._read()
            if running or data["paused"] or not data["waiting"]:
                return False
            data["paused"] = True
            self._write(data)
            return True

    # ---- advancing
    def advance(self, running: bool, start) -> dict | None:
        """One tick of the queue. running: whether a job runs now; start(config dict) starts one and returns
        (ok, message). Starts the first waiting run when nothing runs and the queue is not paused.
        A run that fails to start is set aside in "recent" with its message and the queue pauses (the next run would
        likely fail the same way). Returns the entry it started or failed to start, else None."""
        with self.lock:
            if running:
                return None
            data = self._read()
            if data["paused"] or not data["waiting"]:
                return None
            entry = data["waiting"].pop(0)
            ok, message = start(entry["config"])
            done = {k: v for k, v in entry.items() if k != "config"}
            if ok:
                done["started"] = time.time()
            else:
                done["error"] = message
                data["waiting"].insert(0, entry)  # it stays first, to fix and resume
                data["paused"] = True
            done["message"] = message
            data["recent"] = ([done] + data["recent"])[:DONE_KEPT]
            self._write(data)
            return done
