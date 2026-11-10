#!/usr/bin/env python3
"""Several vast.ai instances as one fleet, from this computer: watch every instance's run in one place, keep a queue
whose next run goes to the first instance that falls idle, keep every run that was ever seen, download each kind of
file by its own rule, and raise an event when a number crosses a line. For a person at a terminal, for the WebUI's
Fleet tab and for an AI assistant with a shell alike.

    python scripts/vast/rulz_fleet.py instances --refresh       # find the running instances, list them
    python scripts/vast/rulz_fleet.py status                    # every instance, its run, the queue, the watches
    python scripts/vast/rulz_fleet.py queue add --job t1 --from my_lora --set steps=300
    python scripts/vast/rulz_fleet.py queue add --job race1 --config race.yaml --split   # one part per instance
    python scripts/vast/rulz_fleet.py download probes=during loras=end --every 300
    python scripts/vast/rulz_fleet.py watch add avg_loss above 0.5 --note "diverging?"
    python scripts/vast/rulz_fleet.py daemon                    # the loop that does the work (or keep the WebUI open)
    python scripts/vast/rulz_fleet.py wait                      # blocks until a watch fires or a run ends
    python scripts/vast/rulz_fleet.py runs --last 20            # past runs; run KEY for one

Something has to keep looking for the queue, the downloads and the watches to work: `daemon`, a `wait`, or the WebUI
on this computer (its Fleet tab). Only one of them does the looking at a time (a lock file); the others read what it
wrote. Everything is kept in ~/.rulz_vast/fleet (state.json, events.jsonl, runs/); downloaded files go where
rulz_remote.py pull puts them, ~/.rulz_vast/runs/<instance>/<job>.

Like rulz_remote.py this never rents, stops or destroys an instance. Output is JSON.
Exit status: 0 fine, 1 refused, 2 wait timed out, 3 nothing reachable."""
from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import os
import queue as queue_module
import re
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE.parents[1])):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rulz_remote as remote  # noqa: E402

from rulztrainer.train_queue import QueueLock  # noqa: E402  (standard library only)


def config_bytes(config: dict) -> bytes:
    """A run's config as the instance reads it: YAML. It was sent as JSON, which the instance's YAML loader reads
    too, except that JSON writes a small float as `1e-08` and YAML takes an exponent without a decimal point for
    text, so every config with such a default was refused ("'>' not supported between 'str' and 'float'"). YAML's
    own writer puts the point in. Without PyYAML (it comes with the trainer) the JSON goes out as before, and a
    trainer from 3.4.1 reads its numbers back."""
    try:
        import yaml
    except ImportError:
        return json.dumps(config).encode("utf-8")
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode("utf-8")


DOWNLOAD_KINDS = {"logs": "Logs and settings", "probes": "Probe images", "samples": "Sample images",
                  "maps": "Training maps", "best": "Best LoRA", "loras": "Every LoRA / checkpoint"}
DOWNLOAD_MODES = {"off": "Never", "end": "When the run ends", "during": "While it runs"}
DEFAULT_SETTINGS = {"poll_every": 20, "download_every": 300, "discover_every": 120, "auto_discover": True,
                    "download": {"logs": "during", "probes": "during", "samples": "end", "maps": "end",
                                 "best": "end", "loras": "off"}}
# what a watch can be set on: a run's numbers (while it runs), its last likeness probe, and the machine
RUN_METRICS = ("step", "total", "progress", "loss", "avg_loss", "lr", "grad_norm", "eta", "elapsed", "step_time",
               "probes", "cost")
PROBE_METRICS = ("face", "body", "combined", "face_ema", "body_ema", "combined_ema", "face_norm_ema",
                 "body_norm_ema", "gold_ema", "best", "best_step", "since_improve", "lr_factor", "copy")
MACHINE_METRICS = ("disk_free_gb", "gpu_temp", "gpu_util", "gpu_mem_mb")
METRICS = RUN_METRICS + PROBE_METRICS + MACHINE_METRICS
# the events a wait ends on unless told otherwise: the ones somebody has to look at
WAIT_KINDS = ("watch", "run_done", "run_failed", "run_ended", "run_lost", "dispatch_failed", "instance_down",
              "instance_gone", "download_failed", "queue_empty")
EVENT_KINDS = WAIT_KINDS + ("run_started", "dispatched", "instance_up", "instance_added")
RECENT_KEPT = 30
FILLS_PER_TICK = 3  # runs whose settings and probes are fetched per instance and look
PULL_RETRIES = 5
DISPATCH_STALE = 900.0  # a dispatch that has not reported for this long is taken for dead
SERIES_KEPT = 400


class Refused(Exception):
    """A request that can't be carried out; its message is the answer."""


def try_lock(path: Path):
    """An open file holding an OS lock on it, or None when another process (or object) holds it. The OS drops the
    lock when the holder dies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")  # noqa: SIM115
    try:
        if sys.platform == "win32":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def metrics_of(snap: dict, dollars_per_hour=None) -> dict:
    """The numbers a watch can read from one look at an instance: the machine's always, the run's while one runs."""
    out = {"disk_free_gb": number(snap.get("disk_free_gb"))}
    gpus = snap.get("gpus") or []
    if gpus:
        out["gpu_temp"] = max(g.get("temperature") or 0 for g in gpus)
        out["gpu_util"] = min(g.get("utilization") or 0 for g in gpus)
        out["gpu_mem_mb"] = max(g.get("memory_used_mb") or 0 for g in gpus)
    job = snap.get("job") or {}
    if snap.get("job_running") and job:
        for key in ("step", "total", "loss", "avg_loss", "lr", "grad_norm", "eta", "elapsed", "step_time"):
            out[key] = number(job.get(key))
        if out.get("step") is not None and out.get("total"):
            out["progress"] = round(100.0 * out["step"] / out["total"], 2)
        out["probes"] = number(job.get("probes_count"))
        last = job.get("last_probe") or {}
        for key in PROBE_METRICS:
            out[key] = number(last.get(key))
        if dollars_per_hour and out.get("elapsed") is not None:
            out["cost"] = round(out["elapsed"] / 3600.0 * float(dollars_per_hour), 3)
    return {k: v for k, v in out.items() if v is not None}


def run_key(instance, line: dict) -> str:
    """A run's name in the history: its instance and the trainer's run id (job name and start time), so a job
    name used again is a new run."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{instance}-{line.get('run_id') or line.get('job') or 'job'}")


def race_parts(config: dict, parts: int) -> list[list[str]]:
    """A race's configuration names dealt into `parts` hands, one hand per instance."""
    rows = config.get("race_configs") or []
    if not config.get("race") or len(rows) < 2:
        raise Refused("Only a configuration race (race: true, two or more race_configs) can be split.")
    if config.get("race_follow"):
        raise Refused("A relay (race_follow) continues every configuration from each round's winner: it can't be "
                      "split over instances.")
    names = [str(row.get("name") or f"c{i + 1}") for i, row in enumerate(rows)]
    if len(set(names)) != len(names):
        raise Refused(f"The race's configuration names repeat: {', '.join(names)}")
    parts = max(1, min(int(parts), len(names)))
    if parts < 2:
        raise Refused("A split needs two or more instances (or --split N).")
    return [names[i::parts] for i in range(parts)]


def point_at_pushed(config, push) -> dict:
    """A config whose paths into a pushed folder read where the folder lands on the instance: a dataset that is
    C:\\sets\\anna here is datasets/anna there, wherever the settings name it (the dataset, its face references)."""
    roots = [(Path(p).resolve(), Path(p).name) for p in push]

    def swap(value):
        if isinstance(value, dict):
            return {k: swap(v) for k, v in value.items()}
        if isinstance(value, list):
            return [swap(v) for v in value]
        if not isinstance(value, str) or not value or len(value) > 400 or "\n" in value:
            return value
        try:
            path = Path(value).resolve()
        except (OSError, ValueError):
            return value
        for root, name in roots:
            if path == root:
                return f"datasets/{name}"
            if root in path.parents:
                return f"datasets/{name}/{path.relative_to(root).as_posix()}"
        return value

    return swap(config) if roots else config


class Lane:
    """Jobs done one after another on a daemon thread: an instance's downloads, or its dispatches. A daemon thread,
    so a wait that ends is not held up by a download still going (files copied in part come again: a pull skips a
    file only at its full size)."""

    def __init__(self, name: str):
        self.jobs: queue_module.Queue = queue_module.Queue()
        self.thread = threading.Thread(target=self._work, name=name, daemon=True)
        self.thread.start()

    def _work(self) -> None:
        while True:
            func = self.jobs.get()
            try:
                func()
            except Exception as e:  # noqa: BLE001  (a lane must outlive one failed job)
                print(f"[fleet] {threading.current_thread().name}: {e!r}", file=sys.stderr, flush=True)
            finally:
                self.jobs.task_done()


class Fleet:
    def __init__(self, home: Path | None = None, pulls: Path | None = None, key: str | None = None,
                 folder: str = remote.REMOTE_DIR, connect=None, list_instances=None, fetch=None, send=None,
                 now=time.time):
        self.home = Path(home) if home else remote.STATE / "fleet"
        self.pulls = Path(pulls) if pulls else (remote.PULLS if home is None else self.home / "files")
        self.key, self.folder, self.now = key, folder, now
        self.connect = connect or self._connect
        self.list_instances = list_instances or remote.list_instances
        self.fetch = fetch or remote.fetch
        self.send = send or remote.send
        self.lock = QueueLock(self.home / "state.lock")
        self.state_file = self.home / "state.json"
        self.events_file = self.home / "events.jsonl"
        self.snaps_file = self.home / "snapshots.json"
        self.runs_dir = self.home / "runs"
        self.ticker = None  # the open lock file while this object does the looking
        self.remotes: dict[str, object] = {}
        self.lanes: dict[tuple[str, str], Lane] = {}
        self.busy: set[tuple] = set()  # jobs handed to a lane and not done yet
        self.guard = threading.Lock()

    # ---------------------------------------------------------------- storage

    def _read(self) -> dict:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.setdefault("instances", {})
        data.setdefault("ignored", [])
        q = data.setdefault("queue", {})
        for name, default in (("paused", False), ("waiting", []), ("active", []), ("recent", [])):
            q.setdefault(name, default)
        data.setdefault("watches", [])
        settings = data.setdefault("settings", {})
        for name, default in DEFAULT_SETTINGS.items():
            settings.setdefault(name, copy.deepcopy(default))
        for kind, mode in DEFAULT_SETTINGS["download"].items():
            settings["download"].setdefault(kind, mode)
        data.setdefault("seq", 0)
        data.setdefault("readers", {})
        return data

    def read(self) -> dict:
        with self.lock:
            return self._read()

    @contextmanager
    def edit(self):
        """The state, read under the lock and written back when the block ends."""
        with self.lock:
            data = self._read()
            yield data
            self._write_json(self.state_file, data)

    @staticmethod
    def _write_json(path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        for attempt in range(20):  # Windows refuses the replace while another process reads the file
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)

    @staticmethod
    def _read_json(path: Path, default=None):
        for _ in range(3):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return default
            except (OSError, ValueError):
                time.sleep(0.05)
        return default

    def event(self, st: dict, kind: str, **fields) -> dict:
        """Record something that happened (inside an edit). Events are numbered; a reader remembers the last number
        it was given, so none is missed between two waits."""
        st["seq"] += 1
        entry = {"seq": st["seq"], "time": round(self.now(), 1), "kind": kind,
                 **{k: v for k, v in fields.items() if v is not None}}
        self.home.mkdir(parents=True, exist_ok=True)
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def events(self, after: int = 0, kinds=None, last: int = 0) -> list[dict]:
        out = []
        try:
            lines = self.events_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("seq", 0) > after and (not kinds or entry.get("kind") in kinds):
                out.append(entry)
        return out[-last:] if last > 0 else out

    def snapshots(self) -> dict:
        return self._read_json(self.snaps_file, {}) or {}

    # ---- runs (the history): one file per run

    def run_path(self, key: str) -> Path:
        return self.runs_dir / f"{key}.json"

    def load_run(self, key: str) -> dict | None:
        return self._read_json(self.run_path(key))

    def save_run(self, rec: dict) -> None:
        self._write_json(self.run_path(rec["key"]), rec)

    def update_run(self, key: str, change) -> dict | None:
        with self.lock:
            rec = self.load_run(key)
            if rec is None:
                return None
            change(rec)
            self.save_run(rec)
            return rec

    def runs(self, instance=None, state=None, job: str | None = None, group: str | None = None) -> list[dict]:
        """Every run on record, newest first."""
        out = []
        for path in self.runs_dir.glob("*.json") if self.runs_dir.is_dir() else []:
            rec = self._read_json(path)
            if not isinstance(rec, dict) or "key" not in rec:
                continue
            if instance is not None and str(rec.get("instance")) != str(instance):
                continue
            if state and rec.get("state") != state:
                continue
            if job and not fnmatch.fnmatch(rec.get("job") or "", job):
                continue
            if group and rec.get("group") != group:
                continue
            out.append(rec)
        return sorted(out, key=lambda r: r.get("ended") or r.get("modified") or r.get("first_seen") or 0, reverse=True)

    def runs_signature(self) -> list:
        """The record files' names and change times: differs whenever runs() would, without reading a run."""
        try:
            return sorted((e.name, e.stat().st_mtime_ns) for e in os.scandir(self.runs_dir) if e.name.endswith(".json"))
        except OSError:
            return []

    def find_run(self, name: str) -> dict:
        """A run by its key, or the newest one with that job name."""
        rec = self.load_run(name)
        if rec is None:
            rec = next(iter(self.runs(job=name)), None)
        if rec is None:
            raise Refused(f"No run {name!r} on record (see runs).")
        return rec

    def forget_run(self, key: str) -> None:
        """Drop a run from the history. Its downloaded files stay."""
        rec = self.find_run(key)
        if rec.get("state") == "running":
            raise Refused("That run is still going.")
        self.run_path(rec["key"]).unlink(missing_ok=True)

    # ---------------------------------------------------------------- instances

    def _connect(self, inst: dict):
        return remote.Remote(int(inst["id"]), self.key, self.folder, conn=inst.get("conn"), remember=False)

    def remote_for(self, inst: dict):
        iid = str(inst["id"])
        with self.guard:
            found = self.remotes.get(iid)
        if found is None:
            found = self.connect(inst)
            with self.guard:
                self.remotes[iid] = found
        conn = getattr(found, "conn", None)
        if conn and conn != inst.get("conn"):
            with self.edit() as st:
                if iid in st["instances"]:
                    st["instances"][iid]["conn"] = conn
            inst["conn"] = conn
        return found

    def instance(self, iid) -> dict:
        inst = self.read()["instances"].get(str(iid))
        if inst is None:
            raise Refused(f"Instance {iid} is not in the fleet (see instances).")
        return inst

    def add_instance(self, iid, label: str = "") -> dict:
        """Take an instance into the fleet by its vast.ai id."""
        iid = str(int(iid))
        try:
            listed = {str(i.get("id")): i for i in self.list_instances()}
        except remote.Unreachable:
            listed = {}
        raw = listed.get(iid) or {}
        with self.edit() as st:
            st["ignored"] = [x for x in st["ignored"] if x != iid]
            inst = st["instances"].setdefault(iid, {"id": int(iid), "enabled": True, "added": self.now()})
            inst.update({k: v for k, v in (("gpu", raw.get("gpu_name")), ("dph", raw.get("dph_total")),
                                           ("status", raw.get("actual_status")), ("label", label or None))
                         if v is not None})
            inst.pop("gone", None)
            return dict(inst)

    def remove_instance(self, iid) -> None:
        """Take an instance out of the fleet (its runs stay on record). Discovery leaves it out from then on."""
        iid = str(iid)
        with self.edit() as st:
            if st["instances"].pop(iid, None) is None:
                raise Refused(f"Instance {iid} is not in the fleet.")
            st["ignored"] = sorted(set(st["ignored"]) | {iid})
        with self.guard:
            self.remotes.pop(iid, None)

    def set_instance(self, iid, **fields) -> dict:
        with self.edit() as st:
            inst = st["instances"].get(str(iid))
            if inst is None:
                raise Refused(f"Instance {iid} is not in the fleet.")
            inst.update(fields)
            return dict(inst)

    def discover(self) -> str:
        """Ask vast.ai which instances are running: new RulzTrainer ones join the fleet, ones that stopped or were
        destroyed are marked gone (and their runs lost). Returns "" or why vast could not be asked."""
        try:
            listed = {str(i.get("id")): i for i in self.list_instances()}
        except remote.Unreachable as e:
            with self.edit() as st:
                st["discovered"] = self.now()
                st["discover_error"] = str(e)
            return str(e)
        running = [i for i in listed.values() if i.get("actual_status") == "running"]
        joining = {str(i["id"]) for i in ([i for i in running if remote.is_rulztrainer(i)] or running)}
        with self.edit() as st:
            st["discovered"] = self.now()
            st.pop("discover_error", None)
            for iid in sorted(joining - set(st["instances"]) - set(st["ignored"])):
                st["instances"][iid] = {"id": int(iid), "enabled": True, "added": self.now(), "auto": True}
                self.event(st, "instance_added", instance=int(iid), gpu=listed[iid].get("gpu_name"))
            for iid, inst in st["instances"].items():
                raw = listed.get(iid)
                status = raw.get("actual_status") if raw else "destroyed"
                if raw:
                    inst.update(gpu=raw.get("gpu_name"), dph=raw.get("dph_total"))
                inst["status"] = status
                if status == "running":
                    inst.pop("gone", None)
                elif not inst.get("gone"):
                    inst["gone"] = self.now()
                    self.event(st, "instance_gone", instance=int(iid), status=status)
                    self._lose_runs(st, iid, status)
                    with self.guard:
                        self.remotes.pop(iid, None)
        return ""

    def _lose_runs(self, st: dict, iid: str, status: str) -> None:
        for rec in self.runs(instance=iid, state="running"):
            rec.update(state="lost", ended=self.now(), final_pull=False,
                       error=rec.get("error") or f"the instance is {status}")
            self.save_run(rec)
            self.event(st, "run_lost", instance=int(iid), job=rec["job"], run=rec["key"], step=rec.get("step"))

    # ---------------------------------------------------------------- looking

    def poll(self, inst: dict) -> dict:
        """One look at an instance. Never raises: an instance out of reach gives {"ok": False, "error": ...}."""
        started = self.now()
        snap: dict
        try:
            done = self.remote_for(inst).agent("snapshot")
            if done.returncode != 0:
                raise Refused((done.stderr or b"").decode("utf-8", "replace").strip()[-300:]
                              or f"ssh ended with {done.returncode}")
            snap = dict(json.loads(done.stdout), ok=True)
        except (remote.Unreachable, Refused, SystemExit, ValueError, OSError, subprocess.SubprocessError) as e:
            snap = {"ok": False, "error": str(e)[:300] or type(e).__name__}
            with self.guard:
                self.remotes.pop(str(inst["id"]), None)  # the next look finds the route again
        snap.update(instance=int(inst["id"]), polled=started)
        return snap

    def poll_all(self, instances: list[dict]) -> dict[str, dict]:
        snaps: dict[str, dict] = {}

        def look(inst):
            snaps[str(inst["id"])] = self.poll(inst)

        threads = [threading.Thread(target=look, args=(i,), daemon=True) for i in instances]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return snaps

    def observe(self, st: dict, inst: dict, snap: dict) -> None:
        """Take one look into the record: the instance's reach, its runs' states (a run that ended is closed, its
        last files asked for), and the watches."""
        iid, now = str(inst["id"]), self.now()
        if not snap.get("ok"):
            if not inst.get("down"):
                inst["down"] = now
                inst["error"] = snap.get("error")
                self.event(st, "instance_down", instance=int(iid), error=snap.get("error"))
            return
        if inst.pop("down", None):
            inst.pop("error", None)
            self.event(st, "instance_up", instance=int(iid))
        first_look = not inst.get("last_seen")
        inst["last_seen"] = now
        if not inst.get("gpu") and snap.get("gpus"):
            inst["gpu"] = snap["gpus"][0].get("name")
        known = {r["key"]: r for r in self.runs(instance=iid)}
        seen, running_key = set(), None
        for line in snap.get("jobs") or []:
            key = run_key(iid, line)
            if key in seen:
                continue
            seen.add(key)
            rec, state = known.get(key), line.get("state") or "ended"
            before = json.dumps(rec, sort_keys=True, default=str) if rec else ""
            if rec is None:
                # a finished run nobody saw running: there before this instance was first looked at (taken into the
                # history as it is), or started and ended between two looks (a run like any other: it is closed)
                ours = first_look is False or line.get("job") in (inst.get("pending") or {})
                rec = {"key": key, "instance": int(iid), "job": line.get("job"), "run_id": line.get("run_id"),
                       "gpu": inst.get("gpu"), "dph": inst.get("dph"), "first_seen": now, "state": state, "fill": True,
                       "imported": state != "running" and not ours}
                if state == "running" or ours:
                    rec["started"] = now - float(line.get("elapsed") or 0)
                    rec.update((inst.get("pending") or {}).pop(line.get("job"), {}))
                    self.event(st, "run_started", instance=int(iid), job=rec["job"], run=key, group=rec.get("group"))
                if state != "running" and ours:
                    self._close(st, rec, state, line)
            elif rec.get("state") == "running" and state != "running":
                self._close(st, rec, state, line)
            for field in ("step", "total", "avg_loss", "elapsed", "best", "best_step", "error", "modified"):
                if line.get(field) is not None:
                    rec[field] = line[field]
            rec["state"] = state
            if state == "running":
                running_key = key
            if state == "running" and (snap.get("job") or {}).get("job") == rec["job"]:
                job, series = snap["job"], snap.get("series") or {}
                for field in ("loss", "lr", "eta", "stage", "last_probe", "probes_count", "step_time"):
                    if job.get(field) is not None:
                        rec[field] = job[field]
                rec["losses"], rec["probes"] = series.get("losses") or [], series.get("probes") or []
            if rec.get("dph") and rec.get("elapsed") is not None:
                rec["cost"] = round(float(rec["elapsed"]) / 3600.0 * float(rec["dph"]), 3)
            if json.dumps(rec, sort_keys=True, default=str) != before:
                self.save_run(rec)
        for key, rec in known.items():  # a running run whose folder is gone from the instance
            if key not in seen and rec.get("state") == "running":
                self._close(st, rec, "ended", {})
                rec["state"] = "ended"
                self.save_run(rec)
        job = snap.get("job") or {}
        running = job.get("job") if snap.get("job_running") else None
        self.check_watches(st, inst, running, metrics_of(snap, inst.get("dph")), running_key)

    def _close(self, st: dict, rec: dict, state: str, line: dict) -> None:
        """A run that was running and is not any more."""
        rec.update(ended=self.now(), final_pull=True, fill=True, pull_fails=0)
        kind = {"done": "run_done", "error": "run_failed"}.get(state, "run_ended")
        self.event(st, kind, instance=rec["instance"], job=rec["job"], run=rec["key"], group=rec.get("group"),
                   step=line.get("step", rec.get("step")), total=line.get("total", rec.get("total")),
                   best=line.get("best", rec.get("best")), avg_loss=line.get("avg_loss", rec.get("avg_loss")),
                   error=line.get("error"))

    # ---------------------------------------------------------------- watches

    def add_watch(self, metric: str, op: str, value: float, job: str = "", instance=None, repeat: bool = False,
                  note: str = "", by: str = "") -> dict:
        """Raise an event when a number goes above or below a value. It fires once per run (per instance for the
        machine's numbers) the moment the number is on that side; with repeat it fires again every time the number
        comes back to that side after leaving it."""
        if metric not in METRICS:
            raise Refused(f"No metric {metric!r}. There are: {', '.join(METRICS)}")
        if op not in ("above", "below"):
            raise Refused("A watch is 'above' or 'below' a value.")
        watch = {"id": uuid.uuid4().hex[:8], "metric": metric, "op": op, "value": float(value), "job": job or None,
                 "instance": int(instance) if instance else None, "repeat": bool(repeat), "note": note or None,
                 "by": by or None, "created": self.now(), "fired": 0, "seen": {}}
        with self.edit() as st:
            st["watches"].append(watch)
        return watch

    def remove_watch(self, watch_id: str) -> int:
        with self.edit() as st:
            keep = [w for w in st["watches"] if watch_id != "all" and w["id"] != watch_id]
            removed = len(st["watches"]) - len(keep)
            st["watches"] = keep
        if not removed and watch_id != "all":
            raise Refused(f"No watch {watch_id!r} (see watch list).")
        return removed

    def check_watches(self, st: dict, inst: dict, job: str | None, metrics: dict, run: str | None = None) -> list[dict]:
        fired = []
        iid = int(inst["id"])
        for watch in st["watches"]:
            if watch.get("instance") and watch["instance"] != iid:
                continue
            of_run = watch["metric"] not in MACHINE_METRICS
            if watch.get("job") and not (job and fnmatch.fnmatch(job, watch["job"])):
                continue
            actual = metrics.get(watch["metric"])
            if actual is None:
                continue
            scope = f"{iid}/{run or job or ''}" if of_run else str(iid)
            hit = actual > watch["value"] if watch["op"] == "above" else actual < watch["value"]
            seen = watch.setdefault("seen", {})
            was = seen.get(scope)
            if hit and not was:
                watch["fired"] = watch.get("fired", 0) + 1
                watch["last_fired"] = self.now()
                fired.append(self.event(st, "watch", watch=watch["id"], metric=watch["metric"], op=watch["op"],
                                        value=watch["value"], actual=actual, instance=iid, job=job if of_run else None,
                                        step=metrics.get("step") if of_run else None, note=watch.get("note")))
            if hit:
                seen[scope] = True
            elif watch.get("repeat"):
                seen[scope] = False
            if len(seen) > 200:  # old runs' entries
                for old in list(seen)[:-100]:
                    seen.pop(old)
        return fired

    # ---------------------------------------------------------------- the queue

    def plan(self, job: str, config: dict, sets=(), preset: str = "", instance=None, push=(), download=None,
             overwrite: bool = False, split: int = 0, label: str = "") -> list[dict]:
        """The queue entries a run makes, nothing queued yet: one, or with split one per part of a race (its
        configurations dealt over that many instances; the parts are kept together as a group)."""
        if not job or Path(job).name != job:
            raise Refused(f"Not a job name: {job!r}")
        if not isinstance(config, dict) or not config:
            raise Refused("The run has no settings: give --config FILE or --from JOB.")
        modes = dict(download or {})
        bad = [f"{k}={v}" for k, v in modes.items() if k not in DOWNLOAD_KINDS or v not in DOWNLOAD_MODES]
        if bad:
            raise Refused(f"Not a download rule: {', '.join(bad)}")
        missing = [str(p) for p in push if not Path(p).exists()]
        if missing:
            raise Refused(f"Not found here: {', '.join(missing)}")
        base = {"config": point_at_pushed(config, push), "set": list(sets), "preset": preset or "", "push": [str(Path(p)) for p in push],
                "instance": int(instance) if instance else None, "download": modes, "overwrite": bool(overwrite),
                "label": label or "", "added": self.now()}
        if split:
            hands = race_parts(config, split)
            plans = [(f"{job}_{hand[0]}" if len(hand) == 1 else f"{job}_part{i + 1}", hand)
                     for i, hand in enumerate(hands)]
        else:
            plans = [(job, [])]
        return [dict(base, id=uuid.uuid4().hex[:10], job=name, race_rows=rows, group=job if split else None)
                for name, rows in plans]

    def enqueue(self, job: str = "", config: dict | None = None, entries: list[dict] | None = None, **options):
        """Queue a run for the first instance that falls idle (or for one instance): plan()'s arguments, or the
        entries it made."""
        entries = entries if entries is not None else self.plan(job, config or {}, **options)
        with self.edit() as st:
            taken = {e["job"] for e in st["queue"]["waiting"] + st["queue"]["active"]}
            clash = [e["job"] for e in entries if e["job"] in taken]
            if clash:
                raise Refused(f"Already queued: {', '.join(clash)}")
            st["queue"]["waiting"] += entries
        return entries

    def queue_edit(self, action: str, entry_id: str = "", delta: int = 0) -> dict:
        with self.edit() as st:
            q = st["queue"]
            ids = [e["id"] for e in q["waiting"]]
            if action in ("pause", "resume"):
                q["paused"] = action == "pause"
            elif action == "clear":
                q["waiting"] = []
            elif action in ("remove", "move"):
                if entry_id not in ids:
                    raise Refused(f"No waiting run with id {entry_id!r}.")
                entry = q["waiting"].pop(ids.index(entry_id))
                if action == "move":
                    q["waiting"].insert(max(0, min(len(ids) - 1, ids.index(entry_id) + delta)), entry)
        return self.queue_view()

    def queue_view(self, st: dict | None = None) -> dict:
        q = (st or self.read())["queue"]

        def line(e):
            cfg = e.get("config") or {}
            return {k: v for k, v in {
                "id": e["id"], "job": e["job"], "instance": e.get("instance"), "group": e.get("group"),
                "race_rows": e.get("race_rows") or None, "steps": cfg.get("steps"), "preset": cfg.get("preset"),
                "set": e.get("set") or None, "push": e.get("push") or None, "label": e.get("label") or None,
                "refused_by": e.get("refused_by") or None, "sent_to": e.get("sent_to")}.items() if v is not None}

        return {"paused": q["paused"], "waiting": [line(e) for e in q["waiting"]],
                "starting": [line(e) for e in q["active"]], "recent": q["recent"][:8]}

    def idle(self, inst: dict, snap: dict | None) -> bool:
        """Free for the next queued run: reachable, its WebUI up (it starts the runs), nothing running or waiting
        there, and no dispatch of ours on its way."""
        if not inst.get("enabled", True) or inst.get("gone") or inst.get("accept") is False:
            return False
        if not snap or not snap.get("ok") or not snap.get("webui_running") or snap.get("job_running"):
            return False
        if (snap.get("queue") or {}).get("waiting"):
            return False
        sending = inst.get("dispatching")
        if sending and self.now() - sending.get("since", 0) < DISPATCH_STALE:
            return False
        return snap.get("polled", 0) > inst.get("settled", 0)

    def schedule(self, st: dict, snaps: dict[str, dict]) -> None:
        """After a look: hand each idle instance the first run it may take, and ask for the files that are due."""
        q, now = st["queue"], self.now()
        if not q["paused"]:
            for iid in sorted(st["instances"], key=int):
                inst = st["instances"][iid]
                if not self.idle(inst, snaps.get(iid)):
                    continue
                entry = next((e for e in q["waiting"] if e.get("instance") in (None, int(iid))
                              and iid not in (e.get("refused_by") or [])), None)
                if entry is None:
                    continue
                q["waiting"].remove(entry)
                entry["sent_to"] = int(iid)
                q["active"].append(entry)
                inst["dispatching"] = {"entry": entry["id"], "job": entry["job"], "since": now}
                self.submit(iid, "dispatch", ("dispatch", entry["id"]),
                            lambda inst=dict(inst), entry=copy.deepcopy(entry), snap=snaps[iid]:
                            self.dispatch(inst, entry, snap))
        settings = st["settings"]
        for iid, snap in snaps.items():
            inst = st["instances"].get(iid) or {}
            if not snap.get("ok"):
                continue
            recs = self.runs(instance=iid)
            fills = 0
            for rec in sorted(recs, key=lambda r: (r.get("state") != "running", not r.get("final_pull"))):
                key = rec["key"]
                if rec.get("fill") and fills < FILLS_PER_TICK:
                    fills += 1
                    self.submit(iid, "io", ("fill", key), lambda inst=dict(inst), key=key: self.fill(inst, key))
                modes = {**settings["download"], **(rec.get("download") or {})}
                if rec.get("final_pull"):
                    kinds = [k for k in DOWNLOAD_KINDS if modes.get(k, "off") != "off"]
                    if not kinds or rec.get("pull_fails", 0) >= PULL_RETRIES:
                        self.update_run(key, lambda r: r.update(final_pull=False))
                        continue
                    self.submit(iid, "io", ("pull", key),
                                lambda inst=dict(inst), key=key, kinds=kinds: self.pull(inst, key, kinds, final=True))
                elif rec.get("state") == "running" and now - rec.get("pulled_at", 0) >= settings["download_every"]:
                    kinds = [k for k in DOWNLOAD_KINDS if modes.get(k) == "during"]
                    if kinds:
                        self.submit(iid, "io", ("pull", key),
                                    lambda inst=dict(inst), key=key, kinds=kinds: self.pull(inst, key, kinds))

    def submit(self, iid: str, lane: str, mark: tuple, func) -> bool:
        """Hand a job to one of an instance's lanes, unless the same job is already there."""
        with self.guard:
            if mark in self.busy:
                return False
            self.busy.add(mark)
            worker = self.lanes.get((iid, lane))
            if worker is None:
                worker = self.lanes[(iid, lane)] = Lane(f"fleet-{iid}-{lane}")

        def job():
            try:
                func()
            finally:
                with self.guard:
                    self.busy.discard(mark)

        worker.jobs.put(job)
        return True

    def drain(self, lane: str | None = None) -> None:
        """Wait for the lanes' jobs (all of them, or the dispatches only)."""
        with self.guard:
            workers = [w for (_, name), w in self.lanes.items() if lane in (None, name)]
        for worker in workers:
            worker.jobs.join()

    def start_args(self, entry: dict, dry_run: bool = False) -> list[str]:
        args = ["start", "--job", entry["job"]]
        if entry.get("preset"):
            args += ["--preset", entry["preset"]]
        for item in entry.get("set") or []:
            args += ["--set", item]
        if entry.get("race_rows"):
            args += ["--race-rows", ",".join(entry["race_rows"])]
        if entry.get("overwrite"):
            args.append("--overwrite")
        return args + (["--dry-run"] if dry_run else ["--resume-queue"])

    def dispatch(self, inst: dict, entry: dict, snap: dict) -> None:
        """Send a queued run to an instance: its dataset first when it is not there, then the run into that
        instance's own queue. A refusal sends the run on to the next instance; refused by all, it is set aside."""
        iid = str(inst["id"])
        outcome, message, reply = "refused", "", {}
        try:
            target = self.remote_for(inst)
            for path in entry.get("push") or []:
                if Path(path).name not in (snap.get("datasets") or []):
                    self.send(target, [(Path(path), Path(path).name)], "datasets", hidden=False)
            done = target.agent(*self.start_args(entry), config=config_bytes(entry["config"]))
            try:
                reply = json.loads(done.stdout or b"{}")
            except ValueError:
                reply = {}
            message = (done.stderr or b"").decode("utf-8", "replace").strip()[-400:] or str(reply.get("message") or "")
            if done.returncode == remote.SSH_FAILED:
                outcome = "unreachable"
            elif done.returncode == 0 and reply.get("state") in ("running", "queued"):
                outcome = "sent"
            elif reply.get("state") == "queued":  # queued there, but that WebUI is not running to start it
                outcome, message = "sent", reply.get("note") or message
        except remote.Unreachable as e:
            outcome, message = "unreachable", str(e)
        except (SystemExit, OSError, subprocess.SubprocessError) as e:
            outcome, message = "refused", str(e)
        with self.edit() as st:
            q = st["queue"]
            q["active"] = [e for e in q["active"] if e["id"] != entry["id"]]
            live = st["instances"].get(iid)
            if live is not None:
                live.pop("dispatching", None)
                live["settled"] = self.now()
            done_line = {"id": entry["id"], "job": entry["job"], "instance": int(iid), "group": entry.get("group"),
                         "time": self.now()}
            if outcome == "sent":
                if live is not None:
                    live.setdefault("pending", {})[entry["job"]] = {
                        k: v for k, v in (("entry", entry["id"]), ("group", entry.get("group")),
                                          ("download", entry.get("download")), ("label", entry.get("label"))) if v}
                q["recent"] = ([dict(done_line, state=reply.get("state"), note=message or None)]
                               + q["recent"])[:RECENT_KEPT]
                self.event(st, "dispatched", instance=int(iid), job=entry["job"], entry=entry["id"],
                           group=entry.get("group"), state=reply.get("state"), note=message or None)
                if not q["waiting"] and not q["active"]:
                    self.event(st, "queue_empty")
                return
            entry.pop("sent_to", None)
            if outcome == "refused":
                entry["refused_by"] = sorted(set(entry.get("refused_by") or []) | {iid})
                entry["refusal"] = message
                others = [i for i, other in st["instances"].items() if other.get("enabled", True)
                          and not other.get("gone") and i not in entry["refused_by"]]
                if entry.get("instance") or not others:
                    q["recent"] = ([dict(done_line, error=message or "refused")] + q["recent"])[:RECENT_KEPT]
                    self.event(st, "dispatch_failed", instance=int(iid), job=entry["job"], entry=entry["id"],
                               group=entry.get("group"), error=message or "refused")
                    return
            q["waiting"].insert(0, entry)  # out of reach, or another instance may take it

    def recover(self) -> None:
        """Runs that were being sent when the last looker died go back to the head of the queue."""
        with self.edit() as st:
            q = st["queue"]
            for entry in reversed(q["active"]):
                entry.pop("sent_to", None)
                q["waiting"].insert(0, entry)
            q["active"] = []
            for inst in st["instances"].values():
                inst.pop("dispatching", None)

    # ---------------------------------------------------------------- an instance's lanes

    def fill(self, inst: dict, key: str) -> None:
        """Fetch a run's settings, probes and loss for its record."""
        rec = self.load_run(key)
        if rec is None:
            return
        try:
            data = self.remote_for(inst).agent_json("record", rec["job"])
        except (SystemExit, remote.Unreachable, ValueError, OSError, subprocess.SubprocessError):
            data = None

        def change(r):
            r["fill"] = False
            if not data or (data.get("run_id") or r.get("run_id")) != r.get("run_id"):
                return  # that job folder holds another run by now
            status = data.get("status") or {}
            r["config"] = data.get("config") or r.get("config") or {}
            r["probes"], r["losses"] = data.get("probes") or [], data.get("losses") or []
            for field in ("step", "total", "loss", "avg_loss", "elapsed", "error", "lr", "last_probe", "saved_count",
                          "samples_count", "probes_count", "likeness", "stage"):
                if status.get(field) is not None:
                    r[field] = status[field]
            likeness = status.get("likeness") or {}
            for field in ("best", "best_step"):
                if likeness.get(field) is not None:
                    r[field] = likeness[field]

        self.update_run(key, change)

    def pull(self, inst: dict, key: str, kinds: list[str], final: bool = False) -> dict:
        """Copy a run's files of these kinds here (the ones not here yet at their size)."""
        rec = self.load_run(key)
        if rec is None:
            return {}
        dest = self.pulls / str(rec["instance"])
        result: dict = {"folder": str(dest / rec["job"]), "fetched": 0, "mb": 0.0}
        try:
            target = self.remote_for(inst)
            listing = target.agent_json("files", rec["job"], *[a for kind in kinds for a in ("--what", kind)])
            todo = remote.pull_plan(listing["files"], dest)
            if todo:
                self.fetch(target, listing["base"], [f["path"] for f in todo], dest)
            result.update(fetched=len(todo), mb=round(sum(f["size"] for f in todo) / 1e6, 1))
            failure = None
        except (SystemExit, remote.Unreachable, ValueError, KeyError, OSError, subprocess.SubprocessError) as e:
            failure = str(e)[:300] or type(e).__name__

        def change(r):
            r["pulled_at"] = self.now()
            if failure is None:
                r["folder"] = result["folder"]
                r["downloaded"] = sorted(set(r.get("downloaded") or []) | set(kinds))
                r["downloaded_files"] = r.get("downloaded_files", 0) + result["fetched"]
                r["downloaded_mb"] = round(r.get("downloaded_mb", 0.0) + result["mb"], 1)
                r.pop("pull_error", None)
                if final:
                    r["final_pull"], r["pulled_final"] = False, self.now()
            else:
                r["pull_error"] = failure
                r["pull_fails"] = r.get("pull_fails", 0) + 1

        rec = self.update_run(key, change)
        if failure is not None:
            result["error"] = failure
            if final and rec and rec.get("pull_fails", 0) >= PULL_RETRIES:
                with self.edit() as st:
                    self.event(st, "download_failed", instance=rec["instance"], job=rec["job"], run=key, error=failure)
        return result

    def stop_run(self, name: str, kill: bool = False) -> dict:
        """The WebUI's Stop (or Kill) on the instance that runs this job; name may be an instance id instead."""
        st = self.read()
        if name in st["instances"]:
            inst = st["instances"][name]
        else:
            rec = next(iter(self.runs(state="running", job=name)), None)
            if rec is None:
                raise Refused(f"No running job {name!r} in the fleet.")
            inst = self.instance(rec["instance"])
        done = self.remote_for(inst).agent("stop", *(["--kill"] if kill else []))
        if done.returncode != 0:
            raise Refused((done.stderr or b"").decode("utf-8", "replace").strip() or "stop failed")
        return dict(json.loads(done.stdout), instance=inst["id"])

    # ---------------------------------------------------------------- the loop

    def hold(self) -> bool:
        """Become the one that looks (True when this object is, from now or from before)."""
        if self.ticker is None:
            self.ticker = try_lock(self.home / "tick.lock")
            if self.ticker is not None:
                self.recover()
        return self.ticker is not None

    def release(self) -> None:
        if self.ticker is not None:
            self.drain("dispatch")
            self.ticker.close()
            self.ticker = None

    def someone_looks(self) -> bool:
        """Whether a daemon, a wait or a WebUI is doing the looking right now."""
        if self.ticker is not None:
            return True
        probe = try_lock(self.home / "tick.lock")
        if probe is None:
            return True
        probe.close()
        return False

    def active(self, st: dict | None = None) -> bool:
        """Anything to look at: an instance in the fleet, or a run waiting for one."""
        st = st or self.read()
        return bool(st["queue"]["waiting"] or st["queue"]["active"]
                    or any(i.get("enabled", True) for i in st["instances"].values()))

    def tick(self, act: bool = True) -> dict[str, dict]:
        """One look at every instance. act=False only reads and records (runs, events, watches); with act the queue
        moves on and due files are downloaded."""
        st = self.read()
        settings = st["settings"]
        if act and settings.get("auto_discover") and self.now() - st.get("discovered", 0) >= settings["discover_every"]:
            self.discover()
            st = self.read()
        live = [i for i in st["instances"].values() if i.get("enabled", True) and not i.get("gone")]
        snaps = self.poll_all(live)
        with self.edit() as st:
            for iid, snap in snaps.items():
                if iid in st["instances"]:
                    self.observe(st, st["instances"][iid], snap)
            st["ticked"] = self.now()
            if act:
                self.schedule(st, snaps)
        kept = {k: v for k, v in self.snapshots().items() if k in st["instances"]}
        for iid, snap in snaps.items():
            if snap.get("ok") or iid not in kept:
                kept[iid] = snap
            else:  # out of reach: keep the last good look, marked stale
                kept[iid] = dict(kept[iid], ok=False, error=snap.get("error"), failed=snap.get("polled"))
        self._write_json(self.snaps_file, kept)
        return snaps

    def step(self, every: float | None = None) -> bool:
        """For a loop: look when this object is the looker, there is something to look at, and a look is due.
        True when it looked."""
        if not self.hold():
            return False
        st = self.read()
        if not self.active(st):
            return False
        if self.now() - st.get("ticked", 0) < (every or st["settings"]["poll_every"]):
            return False
        self.tick()
        return True

    def wait(self, reader: str = "cli", kinds=WAIT_KINDS, timeout: float = 0, every: float | None = None,
             sleep=time.sleep) -> list[dict]:
        """Block until an event of these kinds is there that this reader has not been given; [] at the timeout.
        Looks itself while nobody else does."""
        deadline = self.now() + timeout if timeout else None
        while True:
            self.step(every)
            cursor = self.read()["readers"].get(reader, 0)
            fresh = self.events(after=cursor)
            if fresh:
                with self.edit() as st:
                    st["readers"][reader] = fresh[-1]["seq"]
                wanted = [e for e in fresh if e["kind"] in kinds]
                if wanted:
                    return wanted
            if deadline is not None and self.now() >= deadline:
                return []
            sleep(2.0)

    # ---------------------------------------------------------------- views

    def overview(self) -> dict:
        """Every instance with its run, the queue, the watches: what `status` prints and the Fleet tab draws."""
        st, snaps, now = self.read(), self.snapshots(), self.now()
        instances = []
        for iid in sorted(st["instances"], key=int):
            inst, snap = st["instances"][iid], snaps.get(iid) or {}
            job = snap.get("job") or {}
            if inst.get("gone"):
                state = "gone"
            elif not inst.get("enabled", True):
                state = "off"
            elif inst.get("down") or snap.get("ok") is False:
                state = "unreachable"
            elif not snap:
                state = "new"
            elif snap.get("job_running"):
                state = "running"
            elif inst.get("dispatching") or (snap.get("queue") or {}).get("waiting"):
                state = "starting"
            else:
                state = "idle"
            line = {"id": int(iid), "label": inst.get("label"), "gpu": inst.get("gpu"), "dph": inst.get("dph"),
                    "state": state, "status": inst.get("status"), "error": inst.get("error"),
                    "accept": inst.get("accept", True), "webui_running": snap.get("webui_running"),
                    "disk_free_gb": snap.get("disk_free_gb"), "disk_total_gb": snap.get("disk_total_gb"),
                    "gpus": snap.get("gpus"), "datasets": snap.get("datasets"),
                    "age": round(now - snap["polled"]) if snap.get("polled") else None,
                    "queue": (snap.get("queue") or {}).get("waiting") or None}
            if job and state in ("running", "starting", "idle"):
                line["job"] = {k: job.get(k) for k in ("job", "stage", "step", "total", "loss", "avg_loss", "lr",
                                                       "eta", "elapsed", "error", "done", "last_probe", "step_time",
                                                       "probes_count", "saved_count") if job.get(k) is not None}
                line["job"]["running"] = bool(snap.get("job_running"))
                line["metrics"] = metrics_of(snap, inst.get("dph")) if snap.get("job_running") else None
                line["series"] = snap.get("series")
            instances.append({k: v for k, v in line.items() if v is not None})
        cursor = st["readers"].get("cli", 0)
        return {"looking": self.someone_looks(), "looked": round(now - st["ticked"]) if st.get("ticked") else None,
                "discover_error": st.get("discover_error"), "instances": instances, "queue": self.queue_view(st),
                "watches": [{k: v for k, v in w.items() if k != "seen" and v is not None} for w in st["watches"]],
                "settings": st["settings"], "unread_events": len(self.events(after=cursor, kinds=WAIT_KINDS))}

    def run_line(self, rec: dict) -> dict:
        """A run in a listing."""
        cfg = rec.get("config") or {}
        line = {"key": rec["key"], "job": rec.get("job"), "instance": rec.get("instance"), "gpu": rec.get("gpu"),
                "state": rec.get("state"), "group": rec.get("group"), "step": rec.get("step"),
                "total": rec.get("total"), "avg_loss": rec.get("avg_loss"), "best": rec.get("best"),
                "best_step": rec.get("best_step"), "elapsed": rec.get("elapsed"), "cost": rec.get("cost"),
                "ended": rec.get("ended") or rec.get("modified"), "preset": cfg.get("preset"),
                "optimizer": cfg.get("optimizer"), "learning_rate": cfg.get("learning_rate"),
                "network_type": cfg.get("network_type"), "lora_rank": cfg.get("lora_rank"),
                "downloaded": rec.get("downloaded"), "folder": rec.get("folder"), "error": rec.get("error")}
        return {k: v for k, v in line.items() if v is not None}

    def group(self, name: str) -> dict:
        """A split race's parts side by side, the best likeness first."""
        parts = [self.run_line(r) for r in self.runs(group=name)]
        if not parts:
            raise Refused(f"No runs of a group {name!r} on record yet.")
        parts.sort(key=lambda r: r.get("best") if r.get("best") is not None else float("-inf"), reverse=True)
        waiting = [e["job"] for e in self.read()["queue"]["waiting"] if e.get("group") == name]
        return {"group": name, "leader": parts[0]["job"] if parts[0].get("best") is not None else None,
                "finished": all(p["state"] != "running" for p in parts) and not waiting,
                "waiting": waiting, "parts": parts}

    def set_download(self, modes: dict, every: int | None = None, run: str | None = None) -> dict:
        """Each kind of file's rule (off, end, during), for every run or for one; every: seconds between the
        downloads while a run goes."""
        bad = [f"{k}={v}" for k, v in modes.items() if k not in DOWNLOAD_KINDS or v not in DOWNLOAD_MODES]
        if bad:
            raise Refused(f"Not a download rule: {', '.join(bad)}. Kinds: {', '.join(DOWNLOAD_KINDS)}; "
                          f"rules: {', '.join(DOWNLOAD_MODES)}")
        if run:
            rec = self.find_run(run)
            rec = self.update_run(rec["key"], lambda r: r.update(download={**(r.get("download") or {}), **modes}))
            return {"run": rec["key"], "download": rec["download"]}
        with self.edit() as st:
            st["settings"]["download"].update(modes)
            if every:
                st["settings"]["download_every"] = max(30, int(every))
            return {"download": st["settings"]["download"], "every": st["settings"]["download_every"]}


# ---------------------------------------------------------------- the command line


def show(data) -> None:
    print(json.dumps(data, indent=1, default=str))


def load_config(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except ValueError:
        pass
    try:
        import yaml
    except ImportError as e:
        raise Refused("Reading a YAML config needs PyYAML here: pip install pyyaml (or give a JSON file).") from e
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise Refused(f"{path} is not a config.")
    return data


def config_from(fleet: Fleet, source: str) -> dict:
    """An earlier run's settings: JOB (the newest run of that name on record) or JOB@INSTANCE (asked there)."""
    job, _, where = source.partition("@")
    if not where:
        rec = next((r for r in fleet.runs(job=job) if r.get("config")), None)
        if rec is not None:
            return rec["config"]
        rec = next(iter(fleet.runs(job=job)), None)
        if rec is None:
            raise Refused(f"No run {job!r} on record: name its instance, {job}@INSTANCE.")
        where = str(rec["instance"])
    try:
        return fleet.remote_for(fleet.instance(where)).agent_json("config", job, "--json")
    except SystemExit as e:
        raise Refused(str(e)) from e


def check_entries(fleet: Fleet, entries: list[dict]) -> list[dict]:
    """The trainer's own checks on a reachable instance before anything is queued (start --dry-run)."""
    st = fleet.read()
    plans = []
    for entry in entries:
        pool = [i for iid, i in st["instances"].items() if i.get("enabled", True) and not i.get("gone")
                and not i.get("down") and entry.get("instance") in (None, int(iid))]
        if not pool:
            plans.append({"job": entry["job"], "checked": False, "note": "no instance in reach to check it on"})
            continue
        inst = pool[0]
        done = fleet.remote_for(inst).agent(*fleet.start_args(entry, dry_run=True),
                                            config=config_bytes(entry["config"]))
        if done.returncode == 0:
            plan = json.loads(done.stdout)
            plans.append({"job": entry["job"], "checked": inst["id"], "arch": plan.get("arch"),
                          "steps": plan.get("steps"), "datasets": plan.get("datasets"), "changed": plan.get("changed")})
            continue
        message = (done.stderr or b"").decode("utf-8", "replace").strip()
        # not a reason to refuse: the dataset is pushed at the start, another instance may have the name free, or
        # this one just dropped out of reach
        fine_later = (entry.get("push") and "Dataset folder not found" in message) \
            or ("already exists" in message and len(pool) > 1) or done.returncode == remote.SSH_FAILED
        if not fine_later:
            raise Refused(f"{entry['job']} on instance {inst['id']}: {message}")
        plans.append({"job": entry["job"], "checked": inst["id"], "note": message})
    return plans


def cmd_status(fleet: Fleet, args) -> int:
    if not args.no_poll and not fleet.someone_looks() and fleet.hold():
        try:
            if fleet.read()["settings"].get("auto_discover"):
                fleet.discover()
            fleet.tick(act=False)
        finally:
            fleet.release()
    view = fleet.overview()
    if not args.series:
        for inst in view["instances"]:
            inst.pop("series", None)
            inst.pop("datasets", None)
    show(view)
    return 0


def cmd_instances(fleet: Fleet, args) -> int:
    note = fleet.discover() if args.refresh else ""
    st = fleet.read()
    show({"instances": [{k: v for k, v in inst.items() if k not in ("conn", "pending")}
                        for _, inst in sorted(st["instances"].items(), key=lambda kv: int(kv[0]))],
          "ignored": st["ignored"], **({"note": note} if note else {})})
    return 3 if note else 0


def cmd_instance_edit(fleet: Fleet, args) -> int:
    if args.command == "add":
        show(fleet.add_instance(args.id, args.label))
    elif args.command == "remove":
        fleet.remove_instance(args.id)
        show({"removed": args.id})
    else:
        fields = {"enable": {"enabled": True}, "disable": {"enabled": False}, "accept": {"accept": True},
                  "hold": {"accept": False}}[args.command]
        show({k: v for k, v in fleet.set_instance(args.id, **fields).items() if k not in ("conn", "pending")})
    return 0


def cmd_queue(fleet: Fleet, args) -> int:
    action = args.action or "show"
    if action == "add":
        if bool(args.config) == bool(args.base):
            raise Refused("Say what to start from: --config FILE or --from JOB[@INSTANCE] (one of them).")
        config = load_config(args.config) if args.config else config_from(fleet, args.base)
        split = args.split
        if split == -1:  # --split alone: one part per instance in the fleet
            split = len([i for i in fleet.read()["instances"].values() if i.get("enabled", True) and not i.get("gone")])
        download = dict(item.split("=", 1) for item in args.download if "=" in item)
        wanted = {"job": args.job, "config": config, "sets": args.set, "preset": args.preset,
                  "instance": args.instance, "push": args.push, "download": download, "overwrite": args.overwrite,
                  "split": split, "label": args.label}
        entries = fleet.plan(**wanted)
        plans = None if args.no_check else check_entries(fleet, entries)
        fleet.enqueue(entries=entries)
        show({"queued": [{"id": e["id"], "job": e["job"], "race_rows": e["race_rows"] or None, "group": e["group"]}
                         for e in entries], "checks": plans, "looking": fleet.someone_looks(),
              "queue": fleet.queue_view()})
        return 0
    if action in ("remove", "move") and not args.id:
        raise Refused(f"queue {action} needs the waiting run's id.")
    show(fleet.queue_view() if action == "show" else fleet.queue_edit(action, args.id or "", args.delta))
    return 0


def cmd_runs(fleet: Fleet, args) -> int:
    recs = fleet.runs(instance=args.instance, state=args.state, job=args.job, group=args.group)
    show([fleet.run_line(r) for r in (recs[:args.last] if args.last > 0 else recs)])
    return 0


def cmd_run(fleet: Fleet, args) -> int:
    rec = fleet.find_run(args.name)
    if args.forget:
        fleet.forget_run(rec["key"])
        show({"forgotten": rec["key"]})
        return 0
    if not args.full:
        rec = {k: v for k, v in rec.items() if k not in ("losses", "config")}
        rec["probes"] = (rec.get("probes") or [])[-args.probes:] if args.probes > 0 else rec.get("probes")
    show(rec)
    return 0


def cmd_group(fleet: Fleet, args) -> int:
    show(fleet.group(args.name))
    return 0


def cmd_download(fleet: Fleet, args) -> int:
    modes = {}
    for item in args.rules:
        kind, sep, mode = item.partition("=")
        if not sep:
            raise Refused(f"Not KIND=RULE: {item!r}")
        modes[kind] = mode
    if not modes and not args.every:
        st = fleet.read()["settings"]
        show({"download": st["download"], "every": st["download_every"], "kinds": DOWNLOAD_KINDS,
              "rules": DOWNLOAD_MODES})
        return 0
    show(fleet.set_download(modes, args.every, args.run))
    return 0


def cmd_pull(fleet: Fleet, args) -> int:
    rec = fleet.find_run(args.name)
    result = fleet.pull(fleet.instance(rec["instance"]), rec["key"], args.what or ["logs"])
    show(result)
    return 1 if result.get("error") else 0


def cmd_watch(fleet: Fleet, args) -> int:
    if args.action == "add":
        if len(args.spec) != 3:
            raise Refused("watch add METRIC above|below VALUE")
        try:
            value = float(args.spec[2])
        except ValueError as e:
            raise Refused(f"Not a number: {args.spec[2]!r}") from e
        show(fleet.add_watch(args.spec[0], args.spec[1], value, job=args.job or "", instance=args.instance,
                             repeat=args.repeat, note=args.note or "", by=args.by or ""))
    elif args.action == "remove":
        if not args.spec:
            raise Refused("watch remove ID (or all)")
        show({"removed": fleet.remove_watch(args.spec[0])})
    else:
        show([{k: v for k, v in w.items() if k != "seen" and v is not None} for w in fleet.read()["watches"]])
    return 0


def cmd_metrics(fleet: Fleet, args) -> int:
    """The numbers a watch can be set on, with each instance's values now."""
    view = fleet.overview()
    snaps, st = fleet.snapshots(), fleet.read()
    show({"metrics": {"run": RUN_METRICS, "last probe": PROBE_METRICS, "machine": MACHINE_METRICS},
          "now": {str(i["id"]): dict(metrics_of(snaps.get(str(i["id"])) or {}, st["instances"][str(i["id"])].get("dph")),
                                     job=(i.get("job") or {}).get("job")) for i in view["instances"]}})
    return 0


def cmd_events(fleet: Fleet, args) -> int:
    show(fleet.events(kinds=args.kind or None, last=args.last))
    return 0


def cmd_wait(fleet: Fleet, args) -> int:
    kinds = tuple(args.on) if args.on else WAIT_KINDS
    if "all" in kinds:
        kinds = EVENT_KINDS
    if not fleet.someone_looks():
        print("[fleet] nobody else is looking: this wait does, until it ends (run `daemon` to keep the queue and "
              "the downloads going between waits)", file=sys.stderr, flush=True)
    try:
        found = fleet.wait(reader=args.reader, kinds=kinds, timeout=args.timeout, every=args.every)
    finally:
        fleet.release()
    show(found)
    return 0 if found else 2


def cmd_daemon(fleet: Fleet, args) -> int:
    if not fleet.hold():
        print("[fleet] another daemon, wait or WebUI is already looking", file=sys.stderr)
        return 1
    print(f"[fleet] looking every {args.every or fleet.read()['settings']['poll_every']} s; Ctrl+C ends it",
          file=sys.stderr, flush=True)
    seen = fleet.read()["seq"]
    try:
        while True:
            fleet.step(args.every)
            for entry in fleet.events(after=seen):
                seen = entry["seq"]
                print(json.dumps(entry, default=str), flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    finally:
        fleet.release()


def cmd_tick(fleet: Fleet, args) -> int:
    if not fleet.hold():
        print("[fleet] another daemon, wait or WebUI is already looking", file=sys.stderr)
        return 1
    try:
        snaps = fleet.tick()
        fleet.drain()
    finally:
        fleet.release()
    show({iid: {"ok": s.get("ok"), "job_running": s.get("job_running"), "error": s.get("error")}
          for iid, s in snaps.items()})
    return 0 if all(s.get("ok") for s in snaps.values()) else 3


def cmd_stop(fleet: Fleet, args) -> int:
    show(fleet.stop_run(args.name, args.kill))
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("--key", help="ssh private key; default: ssh's own choice, or RULZ_VAST_KEY")
    ap.add_argument("--dir", default=os.environ.get("RULZ_VAST_DIR") or remote.REMOTE_DIR,
                    help="the install on the instances")
    sub = ap.add_subparsers(dest="command", required=True)

    def command(name, func, help_=""):
        p = sub.add_parser(name, help=help_ or (func.__doc__ or "").split("\n")[0])
        p.set_defaults(func=func)
        return p

    p = command("status", cmd_status, "every instance with its run, the queue, the watches")
    p.add_argument("--no-poll", action="store_true", help="only what the last look saw")
    p.add_argument("--series", action="store_true", help="with the loss and probe series and the datasets")
    p = command("instances", cmd_instances, "the fleet's instances")
    p.add_argument("--refresh", action="store_true", help="ask vast.ai first: new instances join, gone ones are marked")
    for name, text in (("add", "take an instance into the fleet"), ("remove", "take it out (its runs stay on record)"),
                       ("enable", "look at it again"), ("disable", "stop looking at it"),
                       ("hold", "keep the queue's runs off it"), ("accept", "let the queue use it again")):
        p = command(name, cmd_instance_edit, text)
        p.add_argument("id")
        if name == "add":
            p.add_argument("--label", default="")
    p = command("queue", cmd_queue, "the fleet's queue: show, add, remove ID, move ID --delta N, pause, resume, clear")
    p.add_argument("action", nargs="?", choices=("show", "add", "remove", "move", "pause", "resume", "clear"))
    p.add_argument("id", nargs="?")
    p.add_argument("--delta", type=int, default=-1, help="move: places down (negative: up)")
    p.add_argument("--job", default="", help="add: the new run's job name")
    p.add_argument("--config", help="add: a local config file (YAML or JSON)")
    p.add_argument("--from", dest="base", metavar="JOB[@INSTANCE]", help="add: an earlier run's settings")
    p.add_argument("--preset", default="")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--instance", type=int, help="add: only this instance may take it")
    p.add_argument("--push", action="append", default=[], metavar="FOLDER",
                   help="add: a local dataset folder to copy to the instance's datasets/ first when it is not there")
    p.add_argument("--download", action="append", default=[], metavar="KIND=RULE",
                   help="add: this run's own download rule (see download)")
    p.add_argument("--split", type=int, nargs="?", const=-1, default=0, metavar="PARTS",
                   help="add: deal a race's configurations over this many instances (alone: one per instance)")
    p.add_argument("--overwrite", action="store_true", help="add: replace an earlier run's folder of that name")
    p.add_argument("--label", default="")
    p.add_argument("--no-check", action="store_true", help="add: skip the trainer's checks on an instance")
    p = command("runs", cmd_runs, "every run on record, newest first")
    p.add_argument("--last", type=int, default=30)
    p.add_argument("--state", choices=("running", "done", "error", "ended", "lost"))
    p.add_argument("--job", help="job name, * and ? allowed")
    p.add_argument("--instance", type=int)
    p.add_argument("--group", help="a split race's name")
    p = command("run", cmd_run, "one run on record: its numbers, probes, files")
    p.add_argument("name", help="the run's key, or a job name (its newest run)")
    p.add_argument("--full", action="store_true", help="with its settings and loss series")
    p.add_argument("--probes", type=int, default=8, help="how many of the last probes to show; 0 = all")
    p.add_argument("--forget", action="store_true", help="drop it from the history (downloaded files stay)")
    p = command("group", cmd_group, "a split race's parts side by side, best likeness first")
    p.add_argument("name")
    p = command("download", cmd_download, "each kind of file's download rule: KIND=off|end|during")
    p.add_argument("rules", nargs="*", metavar="KIND=RULE", help=f"kinds: {', '.join(DOWNLOAD_KINDS)}")
    p.add_argument("--every", type=int, help="seconds between downloads while a run goes")
    p.add_argument("--run", help="for this run only")
    p = command("pull", cmd_pull, "download a run's files now")
    p.add_argument("name")
    p.add_argument("--what", action="append", choices=tuple(DOWNLOAD_KINDS))
    p = command("watch", cmd_watch, "watch add METRIC above|below VALUE; watch list; watch remove ID|all")
    p.add_argument("action", nargs="?", choices=("list", "add", "remove"), default="list")
    p.add_argument("spec", nargs="*")
    p.add_argument("--job", help="only runs with this job name (* and ? allowed)")
    p.add_argument("--instance", type=int)
    p.add_argument("--repeat", action="store_true", help="fire again each time the number comes back to that side")
    p.add_argument("--note", help="carried in the event: what this watch is for")
    p.add_argument("--by", help="who set it")
    command("metrics", cmd_metrics)
    p = command("events", cmd_events, "what happened: watches fired, runs started and ended, dispatches")
    p.add_argument("--last", type=int, default=20)
    p.add_argument("--kind", action="append", choices=EVENT_KINDS)
    p = command("wait", cmd_wait, "block until a watch fires or a run ends; prints the events")
    p.add_argument("--on", action="append", choices=EVENT_KINDS + ("all",), help="repeatable; default: the ones "
                   "that need a look (watch, run_done, run_failed, run_ended, dispatch_failed, ...)")
    p.add_argument("--timeout", type=int, default=0, help="give up after this many seconds (exit 2)")
    p.add_argument("--every", type=int, help="seconds between looks while this wait does the looking")
    p.add_argument("--reader", default="cli", help="whose place in the events to read from")
    p = command("daemon", cmd_daemon, "keep looking: the queue, the downloads and the watches work while this runs")
    p.add_argument("--every", type=int, help="seconds between looks; default: the setting (20)")
    command("tick", cmd_tick, "one look, the queue moved on and due files downloaded")
    p = command("stop", cmd_stop)
    p.add_argument("name", help="a running job's name, or an instance id")
    p.add_argument("--kill", action="store_true", help="kill the process: no final save")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    fleet = Fleet(key=args.key, folder=args.dir)
    try:
        return args.func(fleet, args)
    except Refused as e:
        print(f"[refused] {e}", file=sys.stderr)
        return 1
    except remote.Unreachable as e:
        print(f"[unreachable] {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
