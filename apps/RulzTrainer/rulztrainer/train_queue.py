"""The WebUI's training queue: runs waiting for the current one to finish, started one after another.

The queue lives in output/_queue.json (the waiting configs in full, a paused flag and the last few started or failed
runs), so it survives a reload or a restarted WebUI. The WebUI advances it: when no job runs and the queue is not
paused, the first waiting run starts. A WebUI that starts with runs waiting and nothing running pauses the queue
(on_startup) instead of starting one by surprise; Resume goes on. Kill pauses it too (stopping everything is what a
kill usually means); Stop and a failed run go on to the next."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_FILE = ROOT / "output" / "_queue.json"
DONE_KEPT = 10  # started / failed runs kept for the queue's "recently" list


class TrainQueue:
    def __init__(self, path: Path = QUEUE_FILE):
        self.path = Path(path)
        self.lock = threading.RLock()

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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

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
