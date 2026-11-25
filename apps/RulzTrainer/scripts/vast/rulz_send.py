#!/usr/bin/env python3
"""Sending a run to a vast.ai instance from this computer, as a job that survives a closed WebUI (docs/vast-send-contract.md,
sections 7, 9 and 10).

A send is one run on its way to an instance: rent (or start, or pick) the instance -> wait for vast.ai -> reach it by ssh
-> wait for RulZ AI Trainer there -> (update it) -> model files -> datasets -> the agent's dry run -> the Fleet's queue ->
the run -> its downloads -> the after-run action (stop, destroy or leave running). `SendManager.tick()` moves every send
one step on and never blocks for long: uploads, the dry run and an update run on a worker per instance. A restart of the
program continues where it was (every step that was going starts again; all of them can be repeated).

Money (contract section 2): only `create()` rents or starts, from a click. `tick()` can only stop (a failed setup, the
after-run action) and destroy (the after-run action, an instance this app rented). Nothing here keeps a token, an
address or an environment value; what is shown is scrubbed (rulz_files.scrub).

State: sends.json (+ .lock) in the state folder, send.lock (the one process that runs the sends). Standard library only."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE.parents[1])):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rulz_files  # noqa: E402
import rulz_fleet  # noqa: E402
import rulz_market  # noqa: E402
import rulz_plan  # noqa: E402
import rulz_remote as remote  # noqa: E402
from rulz_fleet import Fleet, Refused  # noqa: E402
from rulz_market import MarketError  # noqa: E402

from rulztrainer.train_queue import QueueLock  # noqa: E402  (standard library only)

STEPS = (("rent", "Rent the instance"), ("boot", "Instance starting"), ("reach", "Connect"),
         ("setup", "RulZ AI Trainer ready"), ("update", "Update RulZ AI Trainer"), ("files", "Model files"),
         ("datasets", "Datasets"), ("check", "Check the run"), ("queue", "Queue the run"), ("run", "Training"),
         ("download", "Download results"), ("after", "After the run"))
STEP_KEYS = tuple(k for k, _ in STEPS)
LABELS = dict(STEPS)
WORK_STEPS = ("update", "files", "datasets", "check")  # the steps a worker does (the rest are looks of tick())
FINAL = ("done", "failed", "cancelled")
LIVE = ("going", "waiting", "running", "interrupted")
AFTER_RANK = {"leave": 0, "stop": 1, "destroy": 2}  # the mildest choice of several sends wins
DEADLINES = {"boot": 25 * 60.0, "reach": 10 * 60.0, "setup": 45 * 60.0, "queue": 30 * 60.0, "run": 20 * 60.0}
DCHECK_LIMIT = 600.0  # a check before a destroy that has not answered in this long counts as failed
COUNTDOWN = 180.0  # seconds from "every condition holds" to the after-run action
LOOK_EVERY = 15.0  # seconds between this manager's own looks at an instance (the Fleet's last look is used when newer)
LIST_EVERY = 20.0  # seconds between looks at vast.ai's list
RETRIES = 5  # dropped connections in a row a worker step puts up with
IDLE_STEP_CAP = 120.0  # the most one tick adds to the queue deadline (a long gap between ticks is not idle time)
STOP_EVERY = rulz_market.STOP_EVERY
STOP_LIMIT = rulz_market.STOP_LIMIT
RENT_GRACE = 120.0  # a rent call in flight is waited for this long before no intent and no rental means failure


def job_name_problem(job) -> str:
    """Why a job name is refused ("" when it is fine): what the WebUI accepts goes (spaces, non-ASCII); a name that
    would leave its folder or cannot be a folder does not."""
    if not isinstance(job, str) or not job.strip():
        return "it is empty"
    if len(job) > 100:
        return "it is longer than 100 characters"
    if any(c in job for c in "/\\") or any(ord(c) < 32 or ord(c) == 127 for c in job):
        return "it has a path separator or a control character"
    if job in (".", "..") or job[0] in "-." or job[-1] in ". ":
        return "it starts with - or a dot, or ends with a dot or a space"
    return ""


class SendError(Exception):
    """A request that can't be carried out; its message is the answer."""


class _Wait(Exception):
    """A worker step that has to wait for something (the instance to be idle): tick() starts it again."""


class _Retry(Exception):
    """A worker step that did not get through (a dropped connection): tried again, up to RETRIES in a row."""


class _Fatal(Exception):
    """A worker step that can't go on; its message ends the send."""


def clean(text) -> str:
    return rulz_files.scrub(str(text)).strip()[:400]


# ---------------------------------------------------------------- pure decisions

# (fact, what is wrong when it does not hold): the conditions of contract section 2 rule 6 and amendment 9
CONDITIONS = (("sends_over", "another send to this instance is still going"),
              ("reachable", "the instance did not answer"),
              ("no_job", "a run is going there"),
              ("queue_empty", "its own queue is not empty"),
              ("no_tests", "a LoRA test is running there"),
              ("no_entries", "a run of the Fleet queue is waiting for it"),
              ("no_download", "a download of it is going"),
              ("finals_done", "a run's final download has not finished"),
              ("not_rebooting", "it is rebooting"),
              ("not_updating", "it is updating"))
DESTROY_CONDITIONS = (("runs_done", "a run of this app did not end as done"),
                      ("no_foreign", "a run there was not started by this app"),
                      ("tests_empty", "output/_tests is not empty"),
                      ("tree_ok", "files of a run are not all downloaded here"))


def mildest(actions) -> str:
    """The choice that wins when several sends share an instance: leave over stop over destroy."""
    return min((a for a in actions if a in AFTER_RANK), key=AFTER_RANK.get, default="leave")


def after_decision(action: str, facts: dict) -> dict:
    """What the after-run action may do now. facts: True for every condition that holds (absent counts as not
    holding). {"go", "action" (what to do: "stop" when a destroy had to fall back), "wait" (the conditions that do not
    hold), "downgrade" (why destroy became stop, or "")}."""
    if action not in ("stop", "destroy"):
        return {"go": False, "action": None, "wait": [], "downgrade": ""}
    wait = [text for key, text in CONDITIONS if not facts.get(key)]
    if wait:
        return {"go": False, "action": action, "wait": wait, "downgrade": ""}
    if action == "destroy":
        bad = [text for key, text in DESTROY_CONDITIONS if not facts.get(key)]
        if bad:
            return {"go": True, "action": "stop", "wait": [], "downgrade": "; ".join(bad)}
    return {"go": True, "action": action, "wait": [], "downgrade": ""}


def kinds_for(after: str, modes: dict) -> dict:
    """The download rule of every kind for a run that ends with this after-run action: choosing destroy raises every
    "off" to "end" (the disk is about to go)."""
    kinds = rulz_fleet.DOWNLOAD_KINDS
    if after != "destroy":
        return {}
    return {k: ("end" if modes.get(k, "off") == "off" else modes[k]) for k in kinds}


def entries_on(st: dict, iid) -> list[dict]:
    """The Fleet queue's entries that wait for, are being sent to or are uncertain on this instance. An unpinned entry
    waits for no instance in particular, so it does not count (amendment 3)."""
    q, iid = st["queue"], str(iid)
    out = [e for e in q["waiting"] if str(e.get("instance")) == iid or str(e.get("uncertain")) == iid]
    return out + [e for e in q["active"] if str(e.get("sent_to")) == iid or str(e.get("instance")) == iid]


def tree_missing(listing: dict, dest: Path) -> list[str]:
    """The files of a `files --what tree` listing that are not here at their size (path and size, not time)."""
    out = []
    for f in listing.get("files") or []:
        path = Path(dest) / f["path"]
        if not path.is_file() or path.stat().st_size != f["size"]:
            out.append(f["path"])
    return out


def choice_for(entry: dict, choices: dict) -> dict | None:
    """The choice made for a model file of a plan (as rulz_plan.apply finds it), the standard file when there is one and
    nothing was chosen, else None."""
    path = entry["path"]
    choice = choices.get(path) or next((v for k, v in choices.items() if str(k).casefold() == path.casefold()), None)
    return choice or ({"how": "standard"} if entry.get("standard") else None)


# ---------------------------------------------------------------- the manager


class SendManager:
    """`home` is the state folder (remote.STATE by default); fleet, market and files default to real ones built on it
    (files: a module with rulz_files' functions). now / sleep are replaced by tests; executor(instance, mark, func)
    runs a worker job (by default on the Fleet's own lane per instance, so uploads never block tick())."""

    def __init__(self, home=None, fleet=None, market=None, files=None, now=time.time, sleep=time.sleep,
                 executor=None):
        self.home = Path(home) if home else remote.STATE
        self.fleet = fleet or Fleet(home=self.home / "fleet" if home else None)
        self.market = market or rulz_market.Market(home=self.home if home else None)
        self.files = files or rulz_files
        self.now, self.sleep = now, sleep
        self.executor = executor or (lambda iid, mark, func: self.fleet.submit(str(iid), "send", mark, func))
        self.path = self.home / "sends.json"
        self.lock = QueueLock(self.home / "sends.json.lock")
        self.holder = None
        self.errors: list[str] = []  # what tick() caught (a bug must not stop the other sends)
        self._ticking = threading.Lock()
        self._guard = threading.Lock()
        self._cancel: dict[str, threading.Event] = {}
        self._running: set[str] = set()
        self._plans: dict[str, rulz_plan.SendPlan] = {}
        self._looks: dict[str, tuple[float, dict]] = {}
        self._rows_map: dict[int, dict] = {}
        self._rows_at = -1e18
        self._rows_ok = False
        self._noted: dict[str, float] = {}
        self._gone_seen: dict[int, float] = {}

    # ---------------------------------------------------------------- the lock

    def hold(self) -> bool:
        """Become the process that runs the sends (True when this object is, from now or from before)."""
        if self.holder is None:
            self.holder = rulz_fleet.try_lock(self.home / "send.lock")
            if self.holder is not None:
                self._recover()
        return self.holder is not None

    def release(self) -> None:
        if self.holder is not None:
            self.holder.close()
            self.holder = None

    def _recover(self) -> None:
        """A new holder starts from the saved state: a countdown begins again from its start (amendment 9), and what a
        dead worker left going (a check before a destroy) starts again."""
        def reset(data):
            for s in data["sends"]:
                s["countdown_at"] = None
                if s.get("tree") == "going":
                    s["tree"] = None
                if (s.get("dcheck") or {}).get("state") == "going":
                    s["dcheck"] = None
        self._running.clear()
        try:
            with self._edit() as data:
                reset(data)
        except SendError as e:
            self.errors.append(clean(e))

    def step(self) -> bool:
        """For a loop: does nothing unless this object holds the lock; settles open rents, then tick()."""
        if not self.hold():
            return False
        try:
            self.market.settle()
        except MarketError as e:
            self.errors.append(clean(e.message))
        try:
            self._check_orphans()
        except (SendError, MarketError) as e:
            self.errors.append(clean(getattr(e, "message", e)))
        self.tick()
        return True

    def _check_started(self, sends: dict, rows: dict, done: set) -> None:
        """An instance a send started (not rented) that the send left billing before it reached it."""
        for send in sends.values():
            iid = send["instance"]
            if (not send.get("started") or send.get("rented") or send.get("joined") or iid is None
                    or send["state"] not in ("cancelled", "failed") or send["after"] == "leave" or send.get("keep_ok")):
                continue
            row = rows.get(iid)
            if not row or not row["billing"] or any(o["instance"] == iid and o["state"] not in FINAL
                                                   for o in sends.values() if o["id"] != send["id"]):
                continue
            text = (f"Instance {iid} was started by a send that ended, and it still bills ${row['per_hour']:.2f}/hour. "
                    "Stop it on vast.ai or start a send on it.")
            self._warn(send["id"], "orphan", text)
            if f"s{iid}" in done:
                continue
            with self._edit() as data:
                data["alerted"].append(f"s{iid}")
            self._event("send_alert", instance=iid, error=clean(text))

    def _check_orphans(self) -> None:
        """A rental of this app that no send owns any more (the send is gone, or ended before it reached the instance)
        and that vast.ai still bills: one loud alert, and a warning on the send. Nothing is stopped silently."""
        sends = {s["id"]: s for s in self._all()}
        rows = self._rows()
        done = set(self._read()["alerted"])
        self._check_started(sends, rows, done)
        for rental in self.market.rentals():
            sid, iid = rental.get("send"), rental["id"]
            if not sid or rental.get("destroyed"):
                continue
            send = sends.get(sid)
            if send is not None and not (send["state"] in ("cancelled", "failed") and not send.get("joined")):
                continue
            if send is not None and (send["after"] == "leave" or send.get("keep_ok")):
                continue  # left running by choice
            row = rows.get(iid)
            if not row or not row["billing"]:
                continue
            text = (f"Instance {iid} was rented by this app but no send is using it, and it still bills "
                    f"${row['per_hour']:.2f}/hour. Stop it on vast.ai or start a send on it.")
            if send is not None:
                self._warn(sid, "orphan", text)
            if str(iid) in done:
                continue
            with self._edit() as data:
                data["alerted"].append(str(iid))
            self._event("send_alert", instance=iid, error=clean(text))

    # ---------------------------------------------------------------- storage

    def _read_file(self, path: Path):
        """(data, problem): data is None for a missing file; problem is "" or why the file cannot be used. A sharing
        violation is tried again for a moment."""
        last = ""
        for attempt in range(5):
            try:
                text = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return None, ""
            except OSError as e:
                last = type(e).__name__
                if attempt < 4:
                    time.sleep(0.05)
                continue
            try:
                data = json.loads(text)
            except ValueError:
                return None, f"{path.name} is not valid JSON, so it is left alone. Look at the file in {self.home}."
            if not isinstance(data, dict):
                return None, f"{path.name} is not what was expected, so it is left alone ({self.home})."
            return data, ""
        return None, f"{path.name} could not be read ({last}), so it is left alone ({self.home})."

    def _read(self) -> dict:
        """sends.json. Only a missing file starts empty. One that is empty, unreadable or not what it should be is
        replaced by the last good copy (sends.json.bak) with the bad one kept beside it and an alert; without a good
        copy it raises SendError and is never written over."""
        data, problem = self._read_file(self.path)
        if problem:
            good, bad = self._read_file(self.path.with_name(self.path.name + ".bak"))
            if good is None or bad:
                raise SendError(problem)
            kept = self.path.with_name(f"{self.path.name}.bad-{int(time.time())}")
            try:
                os.replace(self.path, kept)
                self._write_file(good)
            except OSError as e:
                raise SendError(f"{problem} (the last good copy could not be put back: {type(e).__name__})") from None
            text = f"sends.json was unreadable: the last good copy is back and the bad file is kept as {kept.name}."
            self.errors.append(text)
            try:
                self._event("send_alert", error=text)
            except Exception:  # noqa: BLE001
                pass
            data = good
        data = data or {}
        for name in ("sends", "stops", "pending_stops", "alerted"):
            if not isinstance(data.get(name), list):
                data[name] = []
        return data

    def _write_file(self, data: dict) -> None:
        """Whole or not at all: written and fsynced beside it, the last good file kept as .bak, then put in place."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, default=str)
            f.flush()
            os.fsync(f.fileno())
        try:
            if self.path.exists() and self.path.stat().st_size > 0:
                shutil.copy2(self.path, self.path.with_name(self.path.name + ".bak"))
        except OSError:
            pass
        for attempt in range(20):  # Windows refuses the replace while another process reads the file
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)

    @contextmanager
    def _locked(self):
        """The file lock; when it cannot be had (QueueLock goes on without it then) nothing is read or written."""
        with self.lock:
            if self.lock.file is None:
                raise SendError(f"sends.json could not be locked ({self.home}), so nothing was changed.")
            yield

    @contextmanager
    def _edit(self):
        with self._locked():
            data = self._read()
            yield data
            self._write_file(data)

    def _all(self) -> list[dict]:
        with self._locked():
            return copy.deepcopy(self._read()["sends"])

    def _raw(self, sid: str) -> dict | None:
        return next((s for s in self._all() if s["id"] == sid), None)

    def _mut(self, sid: str, change) -> dict | None:
        """Apply change(send) to the saved send and return a copy of it (None: no such send)."""
        with self._edit() as data:
            s = next((x for x in data["sends"] if x["id"] == sid), None)
            if s is None:
                return None
            change(s)
            return copy.deepcopy(s)

    def _event(self, kind: str, **fields) -> None:
        with self.fleet.edit() as st:
            self.fleet.event(st, kind, **fields)

    # ---------------------------------------------------------------- the steps' bookkeeping

    def _note(self, sid: str, text: str, frac=None, waiting: bool = False, step: str | None = None) -> None:
        def change(s):
            if s["state"] in FINAL:
                return
            row = s["steps"][step or s["step"]]
            row["text"] = clean(text)
            if frac is not None:
                row["frac"] = round(max(0.0, min(1.0, frac)), 4)
            if s["state"] in ("going", "waiting"):
                s["state"] = "waiting" if waiting else "going"
        self._mut(sid, change)

    def _progress(self, sid: str, text: str, frac) -> None:
        """A note from a transfer's callback: at most one write of sends.json every 2 seconds (and the last one)."""
        now = time.monotonic()
        if frac is not None and frac < 1.0 and now - self._noted.get(sid, -1e9) < 2.0:
            return
        self._noted[sid] = now
        self._note(sid, text, frac=frac)

    def _warn(self, sid: str, key: str, text: str | None) -> None:
        def change(s):
            if text:
                s["warnings"][key] = clean(text)
            else:
                s["warnings"].pop(key, None)
        self._mut(sid, change)

    def _advance(self, sid: str, key: str, text: str = "") -> None:
        """Step `key` is done: the next one that is not skipped begins."""
        def change(s):
            if s["state"] in FINAL or s["step"] != key:
                return
            now = self.now()
            row = s["steps"][key]
            row.update(state="done", ended=now, frac=1.0, text=clean(text) or row["text"])
            s["fails"] = 0
            later = [k for k in STEP_KEYS[STEP_KEYS.index(key) + 1:] if s["steps"][k]["state"] != "skipped"]
            if not later:
                s["state"] = "done"
                return
            s["step"] = later[0]
            s["steps"][later[0]].update(state="going", started=now, ended=None, text="")
            if s["state"] in ("going", "waiting"):
                s["state"] = "going"
        self._mut(sid, change)

    def _fail(self, sid: str, text: str, stop: bool = True, step: str | None = None) -> None:
        """The send can't go on. A setup that failed stops the instance when this app rented or started it and the
        choice is not "leave running" (never destroys it: rule 5), unless something else is running there."""
        first = []

        def change(s):
            if s["state"] in ("done", "cancelled", "failed"):
                return
            first.append(True)
            s["failed_step"] = step or s["step"]
            row = s["steps"].get(s["failed_step"])
            if row is not None and row["state"] != "done":
                row.update(state="failed", ended=self.now())
            s["state"], s["error"], s["countdown_at"] = "failed", clean(text), None
        s = self._mut(sid, change)
        if s is None or not first:
            return
        self._release_reserved(s)
        suffix = self._stop_after_failure(s) if stop else ""
        if suffix:
            self._mut(sid, lambda x: x.update(error=clean(f"{x['error']} {suffix}")))
        self._event("send_failed", send=sid, job=s["job"], instance=s["instance"], error=clean(text) + (
            f" {suffix}" if suffix else ""), step=s["failed_step"])

    def _stop_after_failure(self, s: dict) -> str:
        iid = s["instance"]
        if iid is None:
            return ""
        if s["after"] == "leave":
            return "The instance is left running, as chosen: it still bills."
        if not (s.get("rented") or s.get("started")):
            return "This app did not rent or start the instance, so it is left running."
        if any(o["instance"] == iid and o["state"] not in FINAL and o["id"] != s["id"] for o in self._all()):
            return "Another send is using the instance, so it is left running."
        if s["steps"]["reach"]["state"] == "done":
            verdict, why = self._stop_verdict(iid)
            if verdict == "leave":
                return f"{why}, so the instance is left running."
            if verdict == "unknown":
                self._add_pending_stop(s)
                return ("The instance could not be looked at, so the app will stop it as soon as it can see that "
                        "nothing is running there.")
        outcome = self._request_stop(iid, [s["id"]], after_run=False)
        if outcome == "failed":
            return "It could not be stopped: stop it on vast.ai, it still bills."
        return "The instance was stopped, not destroyed: its disk keeps what was sent."

    def _downloading(self, iid) -> bool:
        return any(e.get("instance") == int(iid) and e.get("state") == "going" for e in self.fleet.downloads()) \
            or any(self.fleet.downloading(r["key"]) for r in self.fleet.runs(instance=iid))

    def _stop_verdict(self, iid) -> tuple[str, str]:
        """("stop", ""), ("leave", why) or ("unknown", ""): a fresh look that shows no job, an empty instance queue, no
        LoRA test and no download going says stop; a look that fails decides nothing."""
        snap = self._look(iid, fresh=True)
        if not snap.get("ok"):
            return "unknown", ""
        if snap.get("job_running"):
            return "leave", "A run is going on the instance"
        if (snap.get("queue") or {}).get("waiting"):
            return "leave", "The instance's own queue is not empty"
        if snap.get("tests_running"):
            return "leave", "A LoRA test is running there"
        if self._downloading(iid):
            return "leave", "A download of the instance is going"
        return "stop", ""

    def _add_pending_stop(self, s: dict) -> None:
        now = self.now()
        with self._edit() as data:
            if not any(p["instance"] == s["instance"] for p in data["pending_stops"]):
                data["pending_stops"].append({"instance": s["instance"], "send": s["id"], "since": now,
                                              "next": now + 2 * LOOK_EVERY, "alerted": False,
                                              "created": s["created"]})

    def _pending_stops(self) -> None:
        """Stops that wait for a look at the instance to succeed (see _stop_after_failure). After 30 minutes without
        one: an alert, and the stop when this app never saw a run there since the send began."""
        now = self.now()
        with self._locked():
            pending = copy.deepcopy(self._read()["pending_stops"])
        for p in pending:
            if now < p["next"]:
                continue
            iid = p["instance"]
            row = self._rows().get(iid)
            if (row is None and self._confirmed_gone(iid)) or (row is not None and not row["billing"] and row["status"] != "running"):
                self._drop_pending(iid)
                continue
            verdict, why = self._stop_verdict(iid)
            if verdict == "stop":
                self._drop_pending(iid)
                self._request_stop(iid, [p["send"]], after_run=False)
                self._warn(p["send"], "stop", None)
                continue
            if verdict == "leave":
                self._drop_pending(iid)
                self._warn(p["send"], "stop", f"{why}, so the instance is left running.")
                continue
            quiet = not any((r.get("first_seen") or 0) >= p["created"] for r in self.fleet.runs(instance=iid))

            def later(data, iid=iid):
                for x in data["pending_stops"]:
                    if x["instance"] == iid:
                        x["next"] = now + 2 * LOOK_EVERY
                        x["alerted"] = x["alerted"] or now - x["since"] >= 1800
            if now - p["since"] >= 1800 and not p["alerted"]:
                text = f"Instance {iid} has not answered for 30 minutes and it still bills."
                self._event("send_alert", instance=iid, error=text)
                self._warn(p["send"], "stop", text + (" It is being stopped: no run was ever seen there." if quiet
                                                     else " A run was seen there, so it is not stopped."))
            if now - p["since"] >= 1800 and quiet:
                self._drop_pending(iid)
                self._request_stop(iid, [p["send"]], after_run=False)
                continue
            with self._edit() as data:
                later(data)

    def _drop_pending(self, iid) -> None:
        with self._edit() as data:
            data["pending_stops"] = [x for x in data["pending_stops"] if x["instance"] != iid]

    def _release_reserved(self, s: dict) -> None:
        iid = s.get("instance")
        if iid is None:
            return
        with self.fleet.edit() as st:
            inst = st["instances"].get(str(iid))
            if inst is not None and inst.get("reserved") == s["id"]:
                inst.pop("reserved", None)

    def _ensure_reserved(self, s: dict) -> None:
        """A reserved instance takes only this send's entry (amendment 3)."""
        iid = str(s["instance"])
        with self.fleet.edit() as st:
            inst = st["instances"].get(iid)
            if inst is not None and not inst.get("reserved"):
                inst["reserved"] = s["id"]

    # ---------------------------------------------------------------- looking at vast.ai and at an instance

    def _rows(self, fresh: bool = False) -> dict[int, dict]:
        """vast.ai's list of the account's instances, by id (kept LIST_EVERY seconds)."""
        with self._guard:
            if fresh or self.now() - self._rows_at >= LIST_EVERY:
                try:
                    self._rows_map = {r["id"]: r for r in self.market.instances()}
                    self._rows_ok = True
                    for seen in [k for k in self._gone_seen if k in self._rows_map]:
                        self._gone_seen.pop(seen)
                except MarketError:
                    self._rows_ok = False
                self._rows_at = self.now()
            return dict(self._rows_map)

    def _confirmed_gone(self, iid) -> bool:
        """The list read has not shown the instance on two reads a list interval apart (a 200 of an odd shape reads as an
        empty list, and that is no proof that the instance is gone)."""
        if iid in self._rows() or not self._rows_ok:
            self._gone_seen.pop(iid, None)
            return False
        first = self._gone_seen.setdefault(iid, self.now())
        return self.now() - first >= LIST_EVERY

    def _fleet_inst(self, iid) -> dict:
        return dict(self.fleet.read()["instances"].get(str(iid)) or {"id": int(iid)})

    def _look(self, iid, fresh: bool = False) -> dict:
        """The instance's snapshot: this manager's own look when it is `fresh` (a decision) or older than LOOK_EVERY,
        else the newest of that and the Fleet's last."""
        iid, now = str(iid), self.now()
        mine = self._looks.get(iid)
        if not fresh:
            last = (self.fleet.snapshots().get(iid) or {})
            if last.get("ok") and now - (last.get("polled") or 0) < LOOK_EVERY:
                return last
            if mine and now - mine[0] < LOOK_EVERY:
                return mine[1]
        snap = self.fleet.poll(self._fleet_inst(iid))
        self._looks[iid] = (now, snap)
        return snap

    # ---------------------------------------------------------------- the plan of a send

    def preview(self, config: dict, root, gpus: int = 1) -> rulz_plan.SendPlan:
        """What a config needs from this PC (slow: it walks the file system)."""
        return rulz_plan.plan_send(config, Path(root), gpus)

    def _plan(self, s: dict) -> rulz_plan.SendPlan:
        plan = self._plans.get(s["id"])
        if plan is None:
            plan = rulz_plan.plan_send(s["config_in"], Path(s["root"]), s["gpus"])
            if plan.blockers:
                raise _Fatal("The run's files changed since it was sent: " + "; ".join(
                    f"{b['reason']}: {b['path']}" for b in plan.blockers[:3]))
            self._plans[s["id"]] = plan
        return plan

    def _final_config(self, s: dict) -> dict:
        """The config the instance runs: the plan applied with the files' places there, datasets repointed."""
        config = rulz_plan.apply(self._plan(s), s["choices"])
        for name, new in (s.get("renames") or {}).items():
            config = rulz_fleet.repoint_dataset(config, name, new)
        return config

    # ---------------------------------------------------------------- the interface the UI uses

    def instances(self) -> list[dict]:
        """vast.ai's instances joined with the Fleet's last look (section 10)."""
        rows, snaps, st = self._rows(), self.fleet.snapshots(), self.fleet.read()
        by_instance: dict[int, list[str]] = {}
        for s in self._all():
            if s["instance"] is not None and s["state"] in LIVE:
                by_instance.setdefault(s["instance"], []).append(s["id"])
        out = []
        for iid, row in sorted(rows.items()):
            snap = snaps.get(str(iid)) or {}
            status = row["status"]
            if status == "running" and row["is_rulz"]:
                selectable, reason = True, ""
            elif status == "running":
                selectable, reason = False, "it is not set up for RulZ AI Trainer"
            elif status in ("stopped", "exited"):
                selectable, reason = True, f"stopped: starting it bills ${row['dph_total']:.3f}/hour"
            else:
                selectable, reason = False, f"it is {status or 'not available'}"
            job = (snap.get("job") or {}).get("job") if snap.get("job_running") else None
            waiting = len((snap.get("queue") or {}).get("waiting") or []) + len(
                [e for e in st["queue"]["waiting"] if str(e.get("instance")) == str(iid)])
            out.append({"id": iid, "gpu_name": row["gpu_name"], "num_gpus": row["num_gpus"],
                        "gpu_ram_mb": row["gpu_ram_mb"], "per_hour": row["per_hour"], "dph_total": row["dph_total"],
                        "storage_per_hour": row["storage_per_hour"], "status": status, "label": row["label"],
                        "ours": row.get("ours", False), "is_rulz": row["is_rulz"], "selectable": selectable,
                        "reason": reason, "free_gb": snap.get("disk_free_gb"),
                        "version": (snap.get("trainer") or {}).get("version"), "job": job, "waiting": waiting,
                        "sends": by_instance.get(iid, [])})
        return out

    def create(self, job: str, config: dict, root, *, instance=None, start: bool = False, rent: dict | None = None,
               choices: dict | None = None, after: str = "stop", idle_minutes: int = 60,
               update: bool = False, shown=None) -> dict:
        """Save a send and begin it. With `rent` the instance is rented here (after the send was saved), with
        start=True a stopped instance is started here: both only from a click (`shown`: the price per hour the dialog
        showed for that start; refused when the fresh price is more than 0.5% above it). Raises SendError / MarketError (kind
        "pending": the send exists and waits for the rent's intent)."""
        if bool(instance) == bool(rent):
            raise SendError("Say where to send it: one instance, or an offer to rent (not both).")
        if job_name_problem(job):
            raise SendError(f"Not a job name ({job_name_problem(job)}): {job!r}")
        if after not in AFTER_RANK:
            raise SendError("After the run: leave running, stop or destroy.")
        if after == "destroy" and instance and not any(r["id"] == int(instance) and not r.get("destroyed")
                                                       for r in self.market.rentals()):
            raise SendError("Destroy is offered only for an instance this app rented.")
        if not isinstance(config, dict) or not config:
            raise SendError("The run has no settings.")
        idle = int(min(1440, max(10, idle_minutes)))
        choices = {str(k): dict(v) for k, v in (choices or {}).items()}
        config = dict(copy.deepcopy(config), job_name=job)
        row, gpus, per_hour = None, 1, None
        if instance:
            row = next((r for r in self._rows(fresh=True).values() if r["id"] == int(instance)), None)
            if row is None:
                raise SendError(f"Instance {instance} is not on your vast.ai account.")
            if row["status"] == "running" and not row["is_rulz"]:
                raise SendError(f"Instance {instance} is not set up for RulZ AI Trainer.")
            if row["status"] in ("stopped", "exited") and not start:
                raise SendError(f"Instance {instance} is stopped: starting it bills ${row['dph_total']:.3f}/hour, "
                                "so it has to be started first.")
            if row["status"] not in ("running", "stopped", "exited"):
                raise SendError(f"Instance {instance} is {row['status'] or 'not available'}.")
            if start and shown is not None and row["status"] in ("stopped", "exited")                     and row["dph_total"] > float(shown) * rulz_market.PRICE_RISE_REFUSE:
                raise SendError(f"The price rose from ${float(shown):.3f} to ${row['dph_total']:.3f} per hour, so "
                                "nothing was started. Look at the new price and try again.")
            gpus, per_hour = row["num_gpus"], row["dph_total"]
        else:
            offer = rent.get("offer")
            gpus = int(getattr(offer, "num_gpus", None) or (offer or {}).get("num_gpus") or 1) if offer else 1
        plan = self.preview(config, root, gpus)
        if plan.blockers:
            raise SendError("It cannot be sent: " + "; ".join(f"{b['reason']}: {b['path'] or b['key']}"
                                                              for b in plan.blockers[:4]))
        for entry in plan.files:
            choice = choice_for(entry, choices)
            if choice is None or choice.get("how") not in ("upload", "bucket", "standard"):
                raise SendError(f"Choose how to send {Path(entry['path']).name}.")
            if choice["how"] == "standard" and not entry.get("standard"):
                raise SendError(f"{Path(entry['path']).name} has no standard file: upload it.")
            if choice["how"] in ("upload", "bucket") and not rulz_plan.remote_ok(
                    f"models/sent/{'0' * 12}/{rulz_files.clean_name(Path(entry['path']).name)}"):
                raise SendError(f"{Path(entry['path']).name} has a name the instance would not accept: rename it.")
            if choice["how"] == "bucket":
                try:
                    self.files.check_bucket_id(choice.get("bucket") or "")
                except rulz_files.FilesError as e:
                    raise SendError(str(e)) from None
            choices.setdefault(entry["path"], choice)
        queue = self.fleet.read()["queue"]
        queued = {e["job"].casefold() for e in queue["waiting"] + queue["active"]}
        sid, now = uuid.uuid4().hex[:10], self.now()
        steps = {k: {"state": "waiting", "text": "", "frac": None, "started": None, "ended": None} for k in STEP_KEYS}
        first = "rent" if rent else "boot"
        if not rent:
            steps["rent"].update(state="skipped", text="an existing instance")
        if not update:
            steps["update"].update(state="skipped")
        steps[first].update(state="going", started=now)
        send = {"id": sid, "job": job, "created": now, "instance": int(instance) if instance else None, "label": "",
                "after": after, "idle_minutes": idle, "update": bool(update), "state": "going", "step": first,
                "steps": steps, "error": None, "warnings": {}, "per_hour": per_hour, "rented": bool(rent),
                "started": bool(instance and start), "run": None, "entry": None, "queued": None, "root": str(root),
                "config_in": config, "choices": choices, "gpus": gpus, "sent": {}, "renames": {}, "fails": 0,
                "countdown_at": None, "action": None, "result": "", "joined": None, "queue_idle": 0.0,
                "queue_tick": None, "renting_since": now if rent else None, "watch": {}, "failed_step": None,
                "started_event": False, "tree": None, "after_at": None}
        with self._edit() as data:  # the taken-name check and the save are one step: two dialogs cannot both pass
            if job.casefold() in queued | {x["job"].casefold() for x in data["sends"] if x["state"] in LIVE}:
                raise SendError(f"A run called {job} is already being sent or queued.")
            data["sends"].append(send)  # saved before anything bills
        if instance and start:
            self.clear_stop(instance)  # a stop asked for earlier must not stop this start again
            try:
                self.market.start(int(instance))
            except Exception as e:  # noqa: BLE001
                if isinstance(e, MarketError) and e.definite:
                    self._drop(sid)
                    raise
                # no clear answer: the start may have been made, so the send stays and boot's deadline stops it
                self._mut(sid, lambda s: s.update(state="waiting"))
                self._warn(sid, "start", "vast.ai gave no clear answer to the start request: the app waits to see "
                                         "whether the instance runs.")
                return self.get(sid)
            self._mut(sid, lambda s: s["steps"]["boot"].update(text="vast.ai is starting it"))
        if rent:
            try:
                rental = self.market.rent(rent.get("offer"), rent.get("filters"), rent.get("template"),
                                          rent.get("disk"), f"send {job}", rent.get("bid"), sid)
            except Exception as e:  # noqa: BLE001
                if isinstance(e, MarketError) and e.definite:
                    self._drop(sid)
                    raise
                label = ((getattr(e, "data", None) or {}).get("label") or "") if isinstance(e, MarketError) else ""
                now = self.now()
                self._mut(sid, lambda s: s.update(label=label or s["label"], state="waiting",
                                                  renting_since=None if label else now))
                self._note(sid, "vast.ai gave no clear answer: the app is checking your account for the rent",
                           waiting=True)
                if isinstance(e, MarketError):
                    raise
                raise MarketError("The rent request got no clear answer, so it may have been made. The app is "
                                  "checking your vast.ai instances for it.", "pending", False) from None
            self._adopt_rental(sid, rental)
        return self.get(sid)

    def _drop(self, sid: str) -> None:
        with self._edit() as data:
            data["sends"] = [s for s in data["sends"] if s["id"] != sid]

    def _adopt_rental(self, sid: str, rental: dict) -> None:
        def change(s):
            s["instance"], s["label"], s["renting_since"] = int(rental["id"]), rental.get("label") or "", None
            s["per_hour"] = rental.get("real_price") or rental.get("price")
            if rental.get("price_over") and rental.get("note"):
                s["warnings"]["price"] = clean(rental["note"])
        self._mut(sid, change)

    def sends(self, active: bool = False) -> list[dict]:
        """Every send, newest first (active: only those under way: going, waiting, running, interrupted)."""
        items = sorted(self._all(), key=lambda s: s["created"], reverse=True)
        return [self._view(s) for s in items if not active or s["state"] in LIVE]

    def get(self, sid: str) -> dict:
        s = self._raw(sid)
        if s is None:
            raise SendError(f"No send {sid!r}.")
        return self._view(s)

    def _view(self, s: dict) -> dict:
        now = self.now()
        steps = [{"key": k, "label": LABELS[k], **{f: s["steps"][k].get(f) for f in
                                                  ("state", "text", "frac", "started", "ended")}} for k in STEP_KEYS]
        return {"id": s["id"], "job": s["job"], "created": s["created"], "instance": s["instance"],
                "label": s["label"], "after": s["after"], "idle_minutes": s["idle_minutes"], "state": s["state"],
                "step": s["step"], "steps": steps, "error": s["error"], "warnings": list(s["warnings"].values()),
                "per_hour": s["per_hour"], "rented": bool(s["rented"]), "started": bool(s.get("started")),
                "run": s["run"], "entry": s["entry"],
                "countdown": None if not s.get("countdown_at") else max(0, round(s["countdown_at"] - now)),
                "action": s.get("action"), "result": s.get("result") or ""}

    def cancel(self, sid: str, instance_action: str = "keep") -> dict:
        """End a send that has not queued its run (its files stay). instance_action "stop": also stop the instance when
        this app rented or started it. After the run is queued only the Fleet's Stop does it."""
        s = self._raw(sid)
        if s is None:
            raise SendError(f"No send {sid!r}.")
        if s["state"] in FINAL:
            raise SendError("That send has already ended.")
        if s["queued"]:
            raise SendError("The run is already queued: stop it from the Fleet (Stop).")
        if instance_action not in ("keep", "stop"):
            raise SendError("Keep the instance running, or stop it.")
        if s["step"] == "rent" and s["instance"] is None and (
                s.get("renting_since") or any(i.get("send") == sid for i in self.market.intents())):
            raise SendError("vast.ai has not answered the rent yet; the app is checking. Try again when that is "
                            "settled.")
        with self._guard:
            self._cancel.setdefault(sid, threading.Event()).set()

        late = []

        def change(x):
            if x["queued"]:  # queued while this click was under way
                late.append(True)
                return
            x["state"], x["error"], x["countdown_at"] = "cancelled", "Cancelled.", None
            x["keep_ok"] = instance_action == "keep"
            row = x["steps"].get(x["step"])
            if row is not None and row["state"] == "going":
                row.update(state="failed", ended=self.now(), text="Cancelled.")
        s = self._mut(sid, change)
        if late:
            with self._guard:
                self._cancel.pop(sid, None)
            raise SendError("The run is already queued: stop it from the Fleet (Stop).")
        self._unqueue(sid)  # an entry that went in between the decision to queue and its save
        self._release_reserved(s)
        if s["instance"] is None:
            return self.get(sid)
        stopped = False
        if instance_action == "stop" and (s.get("rented") or s.get("started")) and not any(
                o["instance"] == s["instance"] and o["state"] not in FINAL for o in self._all()):
            stopped = self._request_stop(s["instance"], [sid], after_run=False) != "failed"
        if not stopped:
            self._disarm(s)
        self._mut(sid, lambda x: x.update(result="Cancelled; the instance was stopped." if stopped else
                                          "Cancelled; the instance keeps running."))
        return self.get(sid)

    def _cancelled(self, sid: str) -> bool:
        """The cancel flag of a send: a worker looks at it between pieces."""
        with self._guard:
            flag = self._cancel.get(sid)
        return bool(flag and flag.is_set())

    def retry(self, sid: str) -> dict:
        """Start a failed send again from the step that failed. Renting is never retried, and neither is a start (it
        bills, so only the Start button does it). A run that is queued or going is never queued a second time."""
        s = self._raw(sid)
        if s is None:
            raise SendError(f"No send {sid!r}.")
        if s["state"] != "failed":
            raise SendError("Only a send that failed can be tried again.")
        step = s.get("failed_step") or s["step"]
        if step == "rent" or s["instance"] is None:
            raise SendError("Renting is not tried again: rent another instance.")
        row = self._rows(fresh=True).get(s["instance"])
        if row is None and self._rows_ok:
            raise SendError(f"Instance {s['instance']} is no longer on your vast.ai account.")
        if row is not None and row["status"] != "running" and not row["billing"]:
            raise SendError(f"Instance {s['instance']} is stopped. Start it from the Rent GPUs tab (it bills "
                            f"${row['dph_total']:.3f}/hour while it runs), then press Retry.")
        rec = self._run_of(s)
        in_queue = any(e["id"] == sid for e in self.fleet.read()["queue"]["waiting"] + self.fleet.read()["queue"]["active"])
        if step in ("queue", "run") and rec is None:
            step = "queue"
        elif rec is not None and step in ("queue", "run"):
            step = "run"
        fresh_queue = rec is None and not in_queue

        def change(x):
            now = self.now()
            x["state"], x["error"], x["step"], x["fails"], x["countdown_at"] = "going", None, step, 0, None
            for k in STEP_KEYS[STEP_KEYS.index(step):]:
                if x["steps"][k]["state"] != "skipped":
                    x["steps"][k].update(state="waiting", ended=None, text="", frac=None)
            x["steps"][step].update(state="going", started=now)
            x["queue_idle"], x["queue_tick"], x["failed_step"] = 0.0, None, None
            x["warnings"].pop("interrupted", None)
            if fresh_queue and STEP_KEYS.index(step) <= STEP_KEYS.index("queue"):
                x["queued"] = x["queueing"] = None  # nothing of it is queued or running: it queues once more
                x["entry"] = None
        self._mut(sid, change)
        with self._guard:
            self._cancel.pop(sid, None)
        return self.get(sid)

    def keep(self, sid: str) -> dict:
        """Keep running: the after-run action becomes "leave running" and the watchdog on the instance is switched off."""
        self.set_after(sid, "leave")
        return self.get(sid)

    def set_after(self, sid: str, after: str) -> dict:
        s = self._raw(sid)
        if s is None:
            raise SendError(f"No send {sid!r}.")
        if after not in AFTER_RANK:
            raise SendError("After the run: leave running, stop or destroy.")
        if after == "destroy" and not s.get("rented"):
            raise SendError("Destroy is offered only for an instance this app rented.")
        def change(x):
            x["after"], x["countdown_at"], x["action"] = after, None, None
            if after == "leave":
                x["watch"]["need_disarm"] = bool(x["watch"].get("armed"))
        self._mut(sid, change)
        if after == "leave":
            self._disarm(self._raw(sid))
        return self.get(sid)

    def dismiss(self, sid: str) -> None:
        """Remove an ended send from the list."""
        s = self._raw(sid)
        if s is None:
            return
        if s["state"] not in FINAL:
            raise SendError("That send has not ended yet.")
        self._release_reserved(s)
        self._drop(sid)
        self._plans.pop(sid, None)

    def clear_stop(self, instance_id: int) -> None:
        """An instance is being started on purpose (a send's start, or the Start button): a stop that was asked for
        earlier, and one waiting for a look, no longer apply."""
        with self._edit() as data:
            data["stops"] = [x for x in data["stops"] if x["instance"] != int(instance_id)]
            data["pending_stops"] = [x for x in data["pending_stops"] if x["instance"] != int(instance_id)]

    def _stale_stop(self, stop: dict, row: dict | None) -> bool:
        """A stop asked for before the instance was started again: for a newer send, from the Start button, or with a
        run or a queue entry active there."""
        iid = stop["instance"]
        if any(x["instance"] == iid and x["state"] in LIVE and x["created"] > stop["requested"] for x in self._all()):
            return True
        if row is not None and (row.get("started") or 0) > stop["requested"] + 5:
            return True
        if entries_on(self.fleet.read(), iid):
            return True
        return any(r.get("state") == "running" for r in self.fleet.runs(instance=iid))

    def follow_stop(self, instance_id: int) -> None:
        """Follow a stop somebody else sent (a click on Stop) until vast.ai no longer reports the instance running; a
        still-running instance after ten minutes is a loud alert (amendment 6)."""
        now = self.now()
        with self._edit() as data:
            if not any(x["instance"] == int(instance_id) for x in data["stops"]):
                data["stops"].append({"instance": int(instance_id), "requested": now, "next": now + STOP_EVERY,
                                      "alerted": False, "after_run": False, "sends": []})

    # ---------------------------------------------------------------- tick

    def tick(self) -> None:
        """Move every send one step on. Never rents or starts; never blocks for long: uploads, the update, the dry run
        and the checks before a destroy are a worker's, and every ssh call here is a short agent command with its own
        timeout."""
        if not self._ticking.acquire(blocking=False):
            return
        try:
            try:
                self._rows()
                sends = self._all()
            except SendError as e:  # sends.json cannot be read or locked: nothing is done this round
                self.errors.append(clean(e))
                return
            for s in sends:
                if s["state"] in FINAL:
                    continue
                try:
                    self._advance_send(s)
                except _Fatal as e:
                    self._fail(s["id"], str(e))
                except Exception as e:  # noqa: BLE001  (one send's bug must not stop the others)
                    self.errors.append(f"{s['id']}: {e!r}")
                    try:
                        self._warn(s["id"], "internal", f"Internal error: {type(e).__name__}")
                    except SendError:
                        pass
            for part in (self._after_runs, self._pending_stops, self._follow_stops):
                try:
                    part()
                except Exception as e:  # noqa: BLE001
                    self.errors.append(f"{part.__name__}: {e!r}")
        finally:
            self._ticking.release()

    def _advance_send(self, s: dict) -> None:
        if s["state"] == "interrupted" or s["step"] in ("reach", "setup", "update", "files", "datasets", "check",
                                                         "queue", "run"):
            if self._guard_instance(s):
                return
            s = self._raw(s["id"]) or s
        step = s["step"]
        if step in WORK_STEPS:
            if step != "update" and s["steps"]["reach"]["state"] == "done":
                self._ensure_reserved(s)
            self._watch(s)
            if s["id"] not in self._running and not self._cancelled(s["id"]):
                self._running.add(s["id"])
                if self.executor(s["instance"], ("send", s["id"]), lambda sid=s["id"]: self._work(sid)) is False:
                    self._running.discard(s["id"])  # not taken (the lane still had its mark): next tick again
            return
        if step in ("download", "after") and self._late_guard(s):
            return
        handler = getattr(self, "_do_" + step, None)
        if handler is not None:
            handler(s)

    # ---- instance guard: gone and interrupted

    def _late_guard(self, s: dict) -> bool:
        """The download and after steps: an instance that stopped (the watchdog, by hand, outbid) or is gone ends them;
        one that is running but out of reach for 30 minutes is an alert, and is stopped (never destroyed) when the run
        is over and the choice is stop or destroy. True when the send is over."""
        sid, iid, now = s["id"], s["instance"], self.now()
        row = self._rows().get(iid)
        gone = row is None and self._confirmed_gone(iid)
        if gone or (row is not None and not row["billing"] and row["status"] != "running"):
            text = "the instance is gone" if gone else "the instance stopped before the last download; the files " \
                                                       "are on its disk"
            self._mut(sid, lambda x: x.update(result=(x["result"] + " " + text.capitalize() + ".").strip()))
            if s["step"] == "download":
                self._advance(sid, "download", text)
            self._mut(sid, lambda x: x.update(countdown_at=None, action=None))
            self._advance(sid, "after", "the instance is gone" if gone else "already stopped")
            self._release_reserved(s)
            return True
        snap = self._look(iid)
        if snap.get("ok"):
            if s.get("unreach_since"):
                self._mut(sid, lambda x: x.update(unreach_since=None))
            return False
        since = s.get("unreach_since") or now
        if not s.get("unreach_since"):
            self._mut(sid, lambda x: x.update(unreach_since=since))
        if now - since >= 1800 and not s.get("unreach_alerted"):
            text = f"Instance {iid} is running but has not answered for 30 minutes."
            rec = self._run_of(s)
            stopping = s["after"] in ("stop", "destroy") and rec is not None and rec.get("state") in (
                "done", "error", "ended")
            if stopping:
                text += " The run is over, so it is being stopped (not destroyed)."
            self._mut(sid, lambda x: x.update(unreach_alerted=True))
            self._warn(sid, "unreachable", text)
            self._event("send_alert", send=sid, instance=iid, error=text)
            if stopping:
                self._request_stop(iid, [sid], after_run=False)
        return False

    def _guard_instance(self, s: dict) -> bool:
        """The instance left vast.ai, or stopped under a send that had reached it. True when the send is not to go on
        this tick."""
        iid = s["instance"]
        rows = self._rows()
        row = rows.get(iid)
        if row is None:
            if self._confirmed_gone(iid):
                self._fail(s["id"], "The instance is no longer on your vast.ai account.", stop=False)
                return True
            return False
        running = row["status"] == "running" or row["billing"]
        if s["state"] == "interrupted":
            if not running:
                return True
            snap = self._look(iid, fresh=True)
            if not snap.get("ok"):
                return True
            if snap.get("job_running"):  # a run kept going: the send picks it up again
                self._warn(s["id"], "interrupted", None)
                self._mut(s["id"], lambda x: x.update(state="going"))
                return False
            self._mut(s["id"], lambda x: x.update(after="stop" if x["after"] != "leave" else "leave"))
            self._fail(s["id"], "The instance was interrupted (a lost bid or a host reboot) and is back with no run "
                                "going. Continue it from the saved state with Retry on the instance's own page.")
            return True
        if not running and not self._stopping(iid):
            self._warn(s["id"], "interrupted", "The instance was interrupted (a lost bid or a host reboot): the app "
                                               "keeps watching and goes on when it is back.")
            self._mut(s["id"], lambda x: x.update(state="interrupted"))
            return True
        return False

    def _stopping(self, iid) -> bool:
        with self._locked():
            return any(x["instance"] == int(iid) for x in self._read()["stops"])

    # ---- the watchdog (contract section 6, amendment 1)

    def _watch(self, s: dict, touch: bool = True) -> None:
        """Arm the instance's watchdog when the choice is stop or destroy (first contact, again when a look finds it
        gone), touch its alive file while the send is before the run, switch it off for "leave running"."""
        iid, watch = s["instance"], s["watch"]
        if s["steps"]["reach"]["state"] != "done" or self.now() - (watch.get("at") or 0) < LOOK_EVERY:
            return
        self._mut(s["id"], lambda x: x["watch"].update(at=self.now()))
        snap = self._look(iid)
        block = (snap.get("watchdog") if snap.get("ok") else None) or {}
        if s["after"] == "leave":
            if watch.get("armed") or watch.get("need_disarm"):
                self._disarm(s)
            return
        if snap.get("ok") and snap.get("autostop", {}).get("armed"):
            self._warn(s["id"], "autostop", "AutoStop is armed on the instance and may stop it before the last "
                                            "download.")
        if watch.get("armed") and snap.get("ok") and block and not block.get("armed") and not block.get("fired"):
            watch = dict(watch, armed=False)  # the look says it is gone: arm again
        if not watch.get("armed") or watch.get("minutes") != s["idle_minutes"]:
            self._arm(s)
            return
        if not touch:
            return
        try:
            self.fleet.remote_for(self._fleet_inst(iid)).agent("watchdog", "alive")
        except (remote.Unreachable, SystemExit, OSError, subprocess.SubprocessError, ValueError):
            pass

    def _arm(self, s: dict) -> None:
        try:
            done = self.fleet.remote_for(self._fleet_inst(s["instance"])).agent(
                "watchdog", "arm", "--idle-minutes", str(s["idle_minutes"]))
            reply = json.loads(done.stdout or b"{}")
        except (remote.Unreachable, SystemExit, OSError, subprocess.SubprocessError, ValueError):
            return  # not there yet (the install folder may not exist): tried again at the next look
        if done.returncode != 0 or not isinstance(reply, dict):
            return
        can = reply.get("can_stop") is not False
        self._mut(s["id"], lambda x: x["watch"].update(armed=True, minutes=s["idle_minutes"], can_stop=can,
                                                       need_disarm=False))
        if not can:
            self._warn(s["id"], "can_stop", "The instance cannot stop itself: keep this PC on until the run is done.")
        else:
            self._warn(s["id"], "can_stop", None)

    def _disarm(self, s: dict | None) -> None:
        if not s or s.get("instance") is None:
            return
        try:
            self.fleet.remote_for(self._fleet_inst(s["instance"])).agent("watchdog", "disarm")
        except (remote.Unreachable, SystemExit, OSError, subprocess.SubprocessError, ValueError):
            self._mut(s["id"], lambda x: x["watch"].update(need_disarm=True))
            return
        self._mut(s["id"], lambda x: x["watch"].update(armed=False, need_disarm=False))

    # ---- rent, boot, reach, setup

    def _do_rent(self, s: dict) -> None:
        sid = s["id"]
        if s["instance"] is not None:
            self._advance(sid, "rent", "rented")
            return
        rental = next((r for r in self.market.rentals() if r.get("send") == sid), None)
        if rental is not None:
            self._adopt_rental(sid, rental)
            self._advance(sid, "rent", "rented")
            return
        intent = next((i for i in self.market.intents() if i.get("send") == sid), None)
        if intent is not None:
            self._mut(sid, lambda x: x.update(label=intent["label"]))
            text = intent.get("message") or "checking your vast.ai account for the rent"
            self._note(sid, text, waiting=True)
            return
        if s.get("renting_since") and self.now() - s["renting_since"] < RENT_GRACE:
            return
        self._fail(sid, "The rent request was not settled: no instance of it was found. Nothing was rented that this "
                        "app knows of.", stop=False, step="rent")

    def _do_boot(self, s: dict) -> None:
        sid, row = s["id"], self._rows().get(s["instance"])
        if row is not None and row["status"] == "running":
            self._mut(sid, lambda x: x.update(per_hour=row["per_hour"] or x["per_hour"]))
            self._advance(sid, "boot", "vast.ai says it is running")
            return
        text = f"vast.ai says: {row['status']}" if row else "waiting for vast.ai to list it"
        if self._late(s, "boot"):
            self._fail(sid, "The instance did not start within 25 minutes.", step="boot")
            return
        self._note(sid, text, waiting=True)

    def _late(self, s: dict, step: str) -> bool:
        started = s["steps"][step].get("started")
        return bool(started) and self.now() - started > DEADLINES[step]

    def _do_reach(self, s: dict) -> None:
        sid, iid = s["id"], s["instance"]
        error = ""
        try:
            target = self.fleet.remote_for(self._fleet_inst(iid))
            done = target.run("true", in_folder=False, timeout=30)
            if done.returncode != 0:
                error = (done.stderr or b"").decode("utf-8", "replace").strip() or "ssh did not answer"
        except (remote.Unreachable, SystemExit, OSError, subprocess.SubprocessError) as e:
            error = str(e) or "ssh did not answer"
        if not error:
            self._join(s)
            self._advance(sid, "reach", "ssh answers")
            return
        if self._late(s, "reach"):
            self._fail(sid, f"The instance could not be reached by ssh within 10 minutes ({clean(error)}).",
                       step="reach")
            return
        self._note(sid, f"waiting for ssh: {error}", waiting=True)

    def _join(self, s: dict) -> None:
        """The rental joins the Fleet once vast says it runs and ssh answers (amendment 10). A rented instance is
        reserved for its send from now; an existing one only while it uploads and queues."""
        iid, now = str(s["instance"]), self.now()
        row = self._rows().get(s["instance"]) or {}
        with self.fleet.edit() as st:
            st["ignored"] = [x for x in st["ignored"] if x != iid]
            new = iid not in st["instances"]
            inst = st["instances"].setdefault(iid, {"id": int(iid), "enabled": True, "added": now})
            inst.update({k: v for k, v in (("gpu", row.get("gpu_name")), ("dph", row.get("dph_total")),
                                           ("status", "running"), ("label", s.get("label") or None)) if v is not None})
            inst.pop("gone", None)
            if s.get("rented"):
                inst["reserved"] = s["id"]
            if new:
                self.fleet.event(st, "instance_added", instance=int(iid), gpu=row.get("gpu_name"), send=s["id"])
        self._mut(s["id"], lambda x: x.update(joined=now))

    def _do_setup(self, s: dict) -> None:
        sid = s["id"]
        snap = self._look(s["instance"])
        if snap.get("ok") and snap.get("webui_running") and (snap.get("trainer") or {}).get("version"):
            self._advance(sid, "setup", "RulZ AI Trainer " + str(snap["trainer"]["version"]))
            return
        if self._late(s, "setup"):
            self._fail(sid, "RulZ AI Trainer was not ready on the instance within 45 minutes.", step="setup")
            return
        self._watch(s)
        text = "waiting for RulZ AI Trainer to be installed and its WebUI to start"
        if not snap.get("ok") and snap.get("error"):
            text += f" ({clean(snap['error'])})"
        self._note(sid, text, waiting=True)

    # ---- the worker: update, files, datasets, check

    def _work(self, sid: str) -> None:
        """One worker per instance does the steps that take long: the update, the uploads, the dry run. Each is
        repeatable, so a worker that starts again after a restart (or a dropped connection) goes on from what is there."""
        try:
            while True:
                s = self._raw(sid)
                if s is None or s["state"] in FINAL or s["step"] not in WORK_STEPS or self._cancelled(sid):
                    return
                getattr(self, "_w_" + s["step"])(s)
        except _Wait:
            return
        except rulz_files.Cancelled:
            return
        except _Retry as e:
            self._retry(sid, str(e))
        except _Fatal as e:
            self._fail(sid, str(e))
        except rulz_files.FilesError as e:
            if e.kind == "ssh":
                self._retry(sid, str(e))
            else:
                self._fail(sid, rulz_files.scrub(str(e)))
        except (SystemExit, remote.Unreachable, OSError, subprocess.SubprocessError) as e:
            self._retry(sid, str(e) or type(e).__name__)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"{sid}: {e!r}")
            self._fail(sid, f"Unexpected error: {type(e).__name__}")
        finally:
            self._running.discard(sid)

    def _retry(self, sid: str, text: str) -> None:
        s = self._mut(sid, lambda x: x.update(fails=x["fails"] + 1))
        if s is None or s["state"] in FINAL:
            return
        if s["fails"] > RETRIES:
            self._fail(sid, f"It keeps failing: {clean(text)}")
            return
        self._note(sid, f"trying again after a failure: {clean(text)}", waiting=True)

    def _w_update(self, s: dict) -> None:
        sid, iid = s["id"], s["instance"]
        snap = self._look(iid, fresh=True)
        if not snap.get("ok"):
            raise _Retry(snap.get("error") or "the instance did not answer")
        if snap.get("job_running") or (snap.get("queue") or {}).get("waiting"):
            self._note(sid, "waiting for the instance to be idle before updating it", waiting=True)
            raise _Wait()
        if self.fleet._updating(self._fleet_inst(iid), snap) or self.fleet._rebooting(self._fleet_inst(iid)):
            # an update that was under way when this program last ran: its guard clears when a look sees it end
            self._note(sid, "waiting for an update that is going on there", waiting=True)
            raise _Wait()
        self._note(sid, "checking for a newer RulZ AI Trainer")
        try:
            plan = self.fleet.check_update(iid)
            if (plan.get("behind") or 0) > 0:
                self._note(sid, "updating RulZ AI Trainer and restarting its WebUI")
                self.fleet.update_trainer(iid)
        except Refused as e:
            raise _Fatal(f"The update failed: {e}") from None
        self._advance(sid, "update", "up to date")

    def _remote(self, s: dict):
        return self.fleet.remote_for(self._fleet_inst(s["instance"]))

    def _w_files(self, s: dict) -> None:
        sid = s["id"]
        plan = self._plan(s)
        todo = []
        for entry in plan.files:
            choice = choice_for(entry, s["choices"]) or {}
            if choice.get("how") in ("upload", "bucket"):
                todo.append((entry, choice))
        total = sum(e["bytes"] for e, _ in todo) or 1
        self._check_disk(s, plan, [e for e, _ in todo if e["path"] not in s["sent"]])
        done = sum(e["bytes"] for e, _ in todo if e["path"] in s["sent"])
        cancel = lambda: self._cancelled(sid)  # noqa: E731
        for entry, choice in todo:
            path = entry["path"]
            if path in s["sent"]:
                continue
            name, size, base = Path(path).name, entry["bytes"], done

            def progress(d, _t, base=base, name=name, how=choice["how"], size=size):
                self._progress(sid, f"{name}: " + ("uploading" if how == "upload" else "to the bucket"),
                               (base + min(d, size) * (1.0 if how == "upload" else 0.5)) / total)

            target = self.files.target_for(path, home=self.home, cancel=cancel)
            dest, remote_ = target["dest"], self._remote(s)
            if not rulz_plan.remote_ok(dest):
                raise _Fatal(f"{name} cannot be sent: its name is not one the instance accepts ({clean(dest)}).")
            if choice["how"] == "upload":
                if target["kind"] == "folder":
                    self.files.upload_tree(remote_, path, progress=progress, cancel=cancel, sleep=self.sleep,
                                           home=self.home, target=target)
                else:
                    self.files.upload_file(remote_, path, dest, sha256=target["sha256"], progress=progress,
                                           cancel=cancel, sleep=self.sleep, home=self.home)
            else:
                bucket = choice.get("bucket") or ""
                up = self.files.upload_to_bucket(bucket, path, progress=progress, cancel=cancel, home=self.home,
                                                 target=target)

                def fetching(d, _t, base=base, name=name, size=size):
                    self._progress(sid, f"{name}: the instance is downloading it", (base + size * 0.5 + d / 2) / total)

                if target["kind"] == "folder":
                    self.files.fetch_tree_from_bucket(remote_, up, dest, progress=fetching, cancel=cancel,
                                                      sleep=self.sleep)
                else:
                    self.files.fetch_from_bucket(remote_, bucket, up["files"][0]["bucket_path"], dest,
                                                 target["size"], target["sha256"], progress=fetching, cancel=cancel,
                                                 sleep=self.sleep)
            done += size

            def sent(x, p=path, d=dest, c=choice):
                x["sent"][p] = d
                x["choices"][p] = {**c, "remote": d}
            s = self._mut(sid, sent) or s
        self._advance(sid, "files", f"{len(todo)} file(s) on the instance" if todo else "nothing to upload")

    def _check_disk(self, s: dict, plan, files: list[dict]) -> None:
        """The free disk there against the bytes still to come, plus 10% (amendment 11)."""
        snap = self._look(s["instance"], fresh=True)
        if not snap.get("ok"):
            raise _Retry(snap.get("error") or "the instance did not answer")
        have = set(snap.get("datasets") or [])
        need = sum(f["bytes"] for f in files) + sum(d["bytes"] for d in plan.datasets if d["name"] not in have)
        free = snap.get("disk_free_gb")
        if free is not None and need * 1.1 > float(free) * 1e9:
            raise _Fatal(f"There is not enough disk on the instance: {need / 1e9:.1f} GB (plus 10%) have to go up and "
                         f"{float(free):.1f} GB are free. Rent a bigger disk.")

    def _w_datasets(self, s: dict) -> None:
        """Push each dataset the instance lacks, the Fleet's way (a mark with the folder's digest beside it; one that
        changed goes beside the old as <name>_<digest8>), after looking again at the instance's marks."""
        sid = s["id"]
        plan = self._plan(s)
        total = sum(d["bytes"] for d in plan.datasets) or 1
        done = 0
        for ds in plan.datasets:
            name, size = ds["name"], ds["bytes"]
            snap = self._look(s["instance"], fresh=True)  # a fresh snapshot for every decision about marks
            if not snap.get("ok"):
                raise _Retry(snap.get("error") or "the instance did not answer")
            have, marks = set(snap.get("datasets") or []), snap.get("dataset_marks") or {}
            stage = None
            try:
                local = Path(ds["path"]) if ds.get("kind") != "files" else None
                if ds.get("kind") == "files":
                    stage = Path(tempfile.mkdtemp(prefix="rulz_send_"))
                    local = stage / name
                    local.mkdir()
                    for f in ds["files"]:
                        shutil.copy2(f["path"], local / f["name"])
                if not local.exists():
                    raise _Fatal(f"The dataset folder {local.name} is no longer here.")
                digest = remote.dataset_digest(local)
                final = name
                if name in have and marks.get(name) == digest:
                    done += size
                    continue
                if name in have:
                    final = f"{name}_{digest[:8]}"
                    self._mut(sid, lambda x, a=name, b=final: x["renames"].__setitem__(a, b))
                    if final in have and marks.get(final) == digest:
                        done += size
                        continue
                counted = [0]

                def progress(n, _files, base=done, size=size, name=name):
                    counted[0] += n
                    self._progress(sid, f"{name}: sending", (base + min(counted[0], size)) / total)

                self.fleet.send(self._remote(s), [(local, final)], "datasets", hidden=False, staged=True,
                                marks={final: digest}, progress=progress)
                done += size
            finally:
                if stage is not None:
                    shutil.rmtree(stage, ignore_errors=True)
            self._note(sid, f"{name}: on the instance", frac=done / total)
        self._advance(sid, "datasets", f"{len(plan.datasets)} dataset(s) on the instance")

    def _entry(self, s: dict, config: dict) -> dict:
        modes = kinds_for(s["after"], self.fleet.read()["settings"]["download"])
        try:
            return self.fleet.plan(s["job"], config, instance=s["instance"], push=[], download=modes, overwrite=False,
                                   label=f"send {s['job']}", entry_id=s["id"], send=s["id"])[0]
        except Refused as e:
            raise _Fatal(f"The run cannot be queued: {e}") from None

    def _w_check(self, s: dict) -> None:
        sid = s["id"]
        self._note(sid, "the instance checks the run")
        entry = self._entry(s, self._final_config(s))
        try:
            plans = rulz_fleet.check_entries(self.fleet, [entry])
        except Refused as e:
            message = re.sub(r"^\S+ on instance \d+: ", "", str(e))
            raise _Fatal(f"The instance refused the run: {clean(message)}") from None
        if not plans or plans[0].get("arch") is None:  # the instance dropped out of reach: asked again
            raise _Retry((plans[0].get("note") if plans else "") or "the instance did not answer")
        self._advance(sid, "check", f"{plans[0].get('arch')}, {plans[0].get('steps')} steps")

    # ---- queue, run, download

    def _run_of(self, s: dict) -> dict | None:
        """The run record of this send: matched by its entry id, never by job name."""
        return next((r for r in self.fleet.runs(instance=s["instance"]) if r.get("entry") == s["id"]), None)

    def _do_queue(self, s: dict) -> None:
        sid, iid, now = s["id"], s["instance"], self.now()
        st = self.fleet.read()
        q = st["queue"]
        rec = self._run_of(s)
        in_queue = any(e["id"] == sid for e in q["waiting"] + q["active"])
        snap = self._look(iid)
        found = Fleet.found_there(snap, s["job"]) if snap.get("ok") else ""
        self._warn(sid, "paused", "The Fleet queue is paused: resume it for the run to start." if q["paused"] else None)
        self._ensure_reserved(s)
        self._watch(s)
        if rec is None and not in_queue and not found and not s["queued"]:
            self._queue_entry(s)
            return
        if not s["queued"]:
            self._mut(sid, lambda x: x.update(queued=now, entry=sid))
        if rec is not None or found == "running":
            self._advance(sid, "queue", "queued")
            return
        line = next((r for r in q["recent"] if r.get("id") == sid), None)
        if line and line.get("error"):
            self._fail(sid, f"The instance refused the run: {clean(line['error'])}", step="queue")
            return
        if not in_queue and not found and not line:
            self._fail(sid, "The run is no longer in the queue: it was removed before it started.", step="queue")
            return
        if not in_queue and (found or line):  # sent to the instance: its own queue or already going
            self._advance(sid, "queue", "sent to the instance")
            return
        # still waiting for its turn: the deadline counts only the time the instance is idle and free for it
        inst = st["instances"].get(str(iid)) or {}
        last = s.get("queue_tick")
        idle_for = s.get("queue_idle", 0.0) + (min(now - last, IDLE_STEP_CAP) if last and self.fleet.idle(inst, snap)
                                                else 0.0)
        self._mut(sid, lambda x: x.update(queue_tick=now, queue_idle=idle_for))
        if idle_for > DEADLINES["queue"]:
            if any(e["id"] == sid for e in q["waiting"]):
                self.fleet.queue_edit("remove", sid)
                self._fail(sid, "The run was not started within 30 minutes of the instance being free for it"
                                + (" (the Fleet queue is paused)." if q["paused"] else "."), step="queue")
            return
        self._note(sid, "waiting for the Fleet queue" + (" (paused)" if q["paused"] else ""), waiting=True)

    def _unqueue(self, sid: str) -> None:
        """Take this send's entry out of the Fleet queue when it is still waiting there (one that is already being
        sent is the Fleet's Stop)."""
        try:
            self.fleet.queue_edit("remove", sid)
        except Refused:
            pass

    def _queue_entry(self, s: dict) -> None:
        sid, marked = s["id"], []

        def mark(x):
            if x["state"] not in FINAL:  # a send that ended (cancelled) queues nothing
                x["queueing"] = self.now()
                marked.append(True)
        self._mut(sid, mark)
        if not marked:
            return
        entry = self._entry(s, self._final_config(s))
        self._mut(sid, lambda x: x.update(entry=entry["id"]))  # the entry's id is the send's: saved before enqueue
        try:
            self.fleet.enqueue(entries=[entry])
        except Refused as e:
            raise _Fatal(f"The run could not be queued: {e}") from None
        after = self._mut(sid, lambda x: x.update(queued=self.now(), queue_tick=self.now()))
        if after is None or after["state"] in FINAL:  # cancelled while the entry went in: it goes out again
            self._unqueue(sid)
            return
        self._note(sid, "in the Fleet queue", waiting=True)

    def _do_run(self, s: dict) -> None:
        sid, iid = s["id"], s["instance"]
        rec = self._run_of(s)
        if rec is None:
            snap = self._look(iid)
            found = Fleet.found_there(snap, s["job"]) if snap.get("ok") else ""
            started = s["steps"]["run"].get("started") or self.now()
            if not found and self.now() - started > DEADLINES["run"]:
                self._fail(sid, "The run did not start: the instance never reported it.", step="run")
                return
            self._note(sid, "waiting for the run to start" if found != "queued" else "waiting in the instance's queue",
                       waiting=True)
            return
        self._watch(s, touch=False)
        if not s["started_event"]:
            self._mut(sid, lambda x: x.update(started_event=True))
            self._event("send_started", send=sid, job=s["job"], instance=iid, run=rec["key"])
        self._mut(sid, lambda x: x.update(run=rec["key"]))
        self._release_reserved(s)
        state = rec.get("state")
        if state == "lost":
            snap = self._look(iid, fresh=True)  # lost is final only when no look shows the run (amendment 10)
            shown = snap.get("ok") and any(j.get("job") == rec["job"] and j.get("state") == "running"
                                           for j in snap.get("jobs") or [])
            if shown:
                state = "running"
        if state == "running":
            text = f"step {rec.get('step') or 0} of {rec.get('total') or '?'}"
            frac = (rec.get("step") or 0) / rec["total"] if rec.get("total") else None
            self._note(sid, text, frac=frac)
            self._mut(sid, lambda x: x.update(state="running") if x["state"] in ("going", "waiting") else None)
            return
        word = {"done": "finished", "error": "failed", "ended": "ended early", "lost": "was lost"}.get(state, state)
        self._mut(sid, lambda x: x.update(state="going", result=f"The run {word}."))
        self._advance(sid, "run", f"the run {word}")

    def _do_download(self, s: dict) -> None:
        sid = s["id"]
        rec = self._run_of(s)
        if rec is None:
            self._advance(sid, "download", "")
            return
        if s["after"] == "destroy":
            self._ensure_downloads(rec)
        if rec.get("final_pull") or self.fleet.downloading(rec["key"]):
            self._note(sid, "downloading the run's files", waiting=True)
            return
        if rec.get("pulled_final"):
            text = f"{rec.get('downloaded_files', 0)} file(s) here"
        elif rec.get("pull_error"):
            text = f"the download failed ({clean(rec['pull_error'])}); the files are still on the instance"
        else:
            text = "nothing to download"
        self._advance(sid, "download", text)

    def _ensure_downloads(self, rec: dict) -> None:
        """Destroy is chosen: every kind of file is downloaded at least at the end (once; a download that was off
        and is not now is asked for again)."""
        modes = {**self.fleet.read()["settings"]["download"], **(rec.get("download") or {})}
        raised = {k: "end" for k in rulz_fleet.DOWNLOAD_KINDS if modes.get(k, "off") == "off"}
        if not raised:
            return
        self.fleet.update_run(rec["key"], lambda r: r.update(download={**(r.get("download") or {}), **raised}))
        if rec.get("state") != "running" and not rec.get("final_pull"):
            self.fleet.update_run(rec["key"], lambda r: r.update(final_pull=True, pull_fails=0))

    # ---------------------------------------------------------------- after the run

    def _after_runs(self) -> None:
        """The after-run action, once per instance, when every send of it is over (see after_decision)."""
        sends = self._all()
        for iid in sorted({s["instance"] for s in sends if s["instance"] is not None and s["step"] == "after"
                           and s["state"] in ("going", "waiting")}):
            group = [s for s in sends if s["instance"] == iid and s["step"] == "after" and s["state"] in ("going", "waiting")]
            others = [s for s in sends if s["instance"] == iid and s["state"] not in FINAL and s["step"] != "after"]
            action = mildest([s["after"] for s in group])
            ids = [s["id"] for s in group]
            if action == "leave":
                for s in group:
                    self._mut(s["id"], lambda x: x.update(countdown_at=None, action=None,
                                                          result=x["result"] + " The instance is left running."))
                    self._advance(s["id"], "after", "left running")
                    self._release_reserved(s)
                self._disarm_group(group)
                continue
            now = self.now()
            due = next((s["countdown_at"] for s in group if s.get("countdown_at")), None)
            if due and now < due:
                continue  # counting down
            if not due and now - max(s.get("after_at") or 0 for s in group) < LOOK_EVERY:
                continue
            for s in group:
                self._mut(s["id"], lambda x: x.update(after_at=now))
            if not due:
                facts = self._after_facts(iid, group, others, action, need=now - 3 * LOOK_EVERY)
                decision = after_decision(action, facts)
                if facts.get("check_pending"):
                    decision = {"go": False, "action": action, "wait": ["checking the run's files before the destroy"],
                                "downgrade": ""}
                if not decision["go"]:
                    for s in group:
                        self._mut(s["id"], lambda x: x.update(countdown_at=None, action=action))
                        self._note(s["id"], "waiting: " + "; ".join(decision["wait"]), waiting=True)
                    continue
                for s in group:
                    self._mut(s["id"], lambda x, d=decision: x.update(countdown_at=now + COUNTDOWN, action=d["action"]))
                    self._note(s["id"], f"{decision['action']} in {int(COUNTDOWN // 60)} minutes"
                               + (f" (not destroy: {decision['downgrade']})" if decision["downgrade"] else ""))
                continue
            # the countdown is over: every condition is looked at again with a fresh snapshot
            now_action = mildest([s["after"] for s in self._all() if s["id"] in ids])
            facts = self._after_facts(iid, group, [s for s in self._all() if s["instance"] == iid
                                                   and s["state"] not in FINAL and s["step"] != "after"],
                                      now_action, need=due)
            if facts.get("check_pending"):
                continue  # the worker's fresh checks are not in yet: the countdown stays
            decision = after_decision(now_action, facts)
            if not decision["go"]:
                for s in group:
                    self._mut(s["id"], lambda x: x.update(countdown_at=None))
                    self._note(s["id"], "waiting: " + "; ".join(decision["wait"]), waiting=True)
                continue
            self._execute(iid, group, decision)

    def _disarm_group(self, group: list[dict]) -> None:
        for s in group:
            if s["watch"].get("armed") or s["watch"].get("need_disarm"):
                self._disarm(s)

    def _after_facts(self, iid: int, group: list[dict], others: list[dict], action: str, need: float = 0.0) -> dict:
        """The facts after_decision reads. For a destroy the slow checks (jobs --all, the tree listing and its download,
        output/_tests) come from a worker's persisted answer taken at or after `need`; without one the fact
        "check_pending" is set and the worker is started."""
        snap = self._look(iid, fresh=True)
        ok = bool(snap.get("ok"))
        st = self.fleet.read()
        inst = st["instances"].get(str(iid)) or {}
        runs = self.fleet.runs(instance=iid)
        facts = {"sends_over": not others, "reachable": ok, "no_job": ok and not snap.get("job_running"),
                 "queue_empty": ok and not (snap.get("queue") or {}).get("waiting"),
                 "no_tests": ok and not snap.get("tests_running"), "no_entries": not entries_on(st, iid),
                 "no_download": not self._downloading(iid),
                 "finals_done": all(not r.get("final_pull") and r.get("state") != "running" for r in runs),
                 "not_rebooting": not self.fleet._rebooting(inst), "not_updating": not self.fleet._updating(inst, snap)}
        if action != "destroy":
            return facts
        ids = {x["id"] for x in self._all() if x["instance"] == iid}
        ours = [r for r in runs if r.get("send") in ids or r.get("entry") in ids]
        facts["runs_done"] = bool(ours) and all(r.get("state") == "done" for r in ours)
        facts["no_foreign"] = facts["tests_empty"] = facts["tree_ok"] = False
        if not ok or not ours:
            return facts
        gids = [x["id"] for x in group]
        check = (self._raw(gids[0]) or {}).get("dcheck") or {}
        if check.get("state") == "going" and self.now() - (check.get("started") or 0) > DCHECK_LIMIT:
            check = {"state": "done", "no_foreign": False, "tests_empty": False, "tree_ok": False, "at": self.now()}
            for gid in gids:  # a check that never answered counts as failed: the destroy falls back to a stop
                self._mut(gid, lambda x: x.update(dcheck=check))
        if check.get("state") == "going":
            facts["check_pending"] = True
            return facts
        if check.get("state") != "done" or (check.get("at") or 0) < need:
            for gid in gids:
                self._mut(gid, lambda x: x.update(dcheck={"state": "going", "at": None, "started": self.now()}))
            keys = [r["key"] for r in ours]
            if self.executor(iid, ("dcheck", iid), lambda: self._dcheck(iid, gids, keys)) is False:
                for gid in gids:
                    self._mut(gid, lambda x: x.update(dcheck=None))
            check = (self._raw(gids[0]) or {}).get("dcheck") or {}
            if check.get("state") != "done" or (check.get("at") or 0) < need:
                facts["check_pending"] = True
                return facts
        facts.update(no_foreign=check["no_foreign"], tests_empty=check["tests_empty"], tree_ok=check["tree_ok"])
        return facts

    def _missing(self, target, recs: list[dict]) -> list[str]:
        missing: list[str] = []
        for rec in recs:
            listing = target.agent_json("files", rec["job"], "--what", "tree")
            missing += tree_missing(listing, self.fleet.pull_folder(rec).parent)
        return missing

    def _dcheck(self, iid: int, ids: list[str], keys: list[str]) -> None:
        """The checks before a destroy, in a worker: no job there that this app did not start, output/_tests empty,
        and every file of every run's tree here (downloaded now when it is not). An answer that is unclear counts as no."""
        result = {"state": "done", "no_foreign": False, "tests_empty": False, "tree_ok": False}
        try:
            recs = [r for r in (self.fleet.load_run(k) for k in keys) if r]
            target = self.fleet.remote_for(self._fleet_inst(iid))
            names = {j.get("job") for j in target.agent_json("jobs", "--all")}
            result["no_foreign"] = names <= {r["job"] for r in recs}
            result["tests_empty"] = target.run('[ -z "$(ls -A output/_tests 2>/dev/null)" ]', timeout=60).returncode == 0
            missing = self._missing(target, recs)
            if missing:
                for key in keys:
                    try:
                        self.fleet.pull_now(key, ["tree"])
                    except Refused:
                        pass
                missing = self._missing(target, recs)
            result["tree_ok"] = not missing
        except Exception:  # noqa: BLE001  (whatever went wrong, the answer is "not proven": a stop, not a destroy)
            pass
        result["at"] = self.now()
        for sid in ids:
            self._mut(sid, lambda x: x.update(dcheck=result))

    def _execute(self, iid: int, group: list[dict], decision: dict) -> None:
        action, ids = decision["action"], [s["id"] for s in group]
        with self._locked():  # the choices as they are now: Keep running wins, and a cancelled send takes no part
            live = [x for x in self._read()["sends"] if x["id"] in ids and x["state"] not in FINAL]
        now_action = mildest([x["after"] for x in live]) if live else "leave"
        if now_action == "leave":
            for x in live:
                self._mut(x["id"], lambda y: y.update(countdown_at=None))
            return
        if AFTER_RANK[now_action] < AFTER_RANK[action]:
            action = now_action
        group = [s for s in group if s["id"] in {x["id"] for x in live}]
        ids = [s["id"] for s in group]
        recs = [r for s in group for r in [self._run_of(s)] if r]
        lost_files = any(r.get("pull_error") and not r.get("pulled_final") for r in recs)
        reason = decision.get("downgrade") or ""
        if action == "destroy":
            try:
                self.market.destroy(iid, require_rental=True)
            except MarketError as e:
                action, reason = "stop", f"destroy was refused: {e.message}"
        if action == "destroy":
            self._event("instance_destroyed_after_run", instance=iid, sends=ids)
            result = "The instance was destroyed after the run."
        else:
            outcome = self._request_stop(iid, ids, after_run=True)
            if outcome == "failed":
                result = "It could not be stopped: stop it on vast.ai, it still bills."
            else:
                result = "The instance was stopped after the run."
                if reason:
                    result += f" It was not destroyed: {clean(reason)}."
                if lost_files:
                    result += " The final download failed: the files are still on its disk."
        for s in group:
            self._mut(s["id"], lambda x: x.update(countdown_at=None, action=action, result=(x["result"] + " " + result).strip()))
            self._advance(s["id"], "after", result)
            self._release_reserved(s)

    # ---------------------------------------------------------------- stops

    def _request_stop(self, iid: int, send_ids: list[str], after_run: bool) -> str:
        """Stop an instance and follow the stop (non-blocking: tick() looks again every 2 minutes). "requested",
        "already" (vast.ai says it is not billing) or "failed" (kept: tick() asks again every 2 minutes)."""
        iid = int(iid)
        row = self._rows(fresh=True).get(iid)
        if row is not None and not row["billing"] and row["status"] != "running":
            return "already"
        now = self.now()

        def keep(failed: bool, alerted: bool) -> None:
            with self._edit() as data:
                known = next((x for x in data["stops"] if x["instance"] == iid), None)
                if known is None:
                    data["stops"].append({"instance": iid, "requested": now, "next": now + STOP_EVERY,
                                          "alerted": alerted, "after_run": after_run, "sends": list(send_ids),
                                          "sent_at": None if failed else now, "failed": failed})
                else:
                    known["after_run"] = known["after_run"] or after_run
                    known["failed"] = failed and known.get("failed", True)
                    known["alerted"] = known["alerted"] or alerted
                    if not failed:
                        known["sent_at"] = now
        try:
            self.market.stop(iid)
        except MarketError as e:
            if e.kind == "gone" and self._rows_ok and iid not in self._rows(fresh=True):
                return "already"
            text = f"Stopping instance {iid} failed: {e.message} It still bills: stop it on vast.ai now."
            for sid in send_ids:
                self._warn(sid, "stop", text)
            self._event("send_alert", instance=iid, error=clean(text))
            keep(True, True)
            return "failed"
        keep(False, False)
        return "requested"

    def _follow_stops(self) -> None:
        """Every stop on record: a failed request is sent again every 2 minutes; an accepted one is looked at every 2
        minutes until vast.ai no longer reports the instance running, sent again every 10 minutes while it does, and
        alerted once."""
        now = self.now()
        for stop in [x for x in self._read_stops() if x["next"] <= now]:
            iid = stop["instance"]
            if self._stale_stop(stop, self._rows().get(iid)):
                with self._edit() as data:
                    data["stops"] = [x for x in data["stops"] if x["instance"] != iid]
                continue

            def update(**fields):
                with self._edit() as data:
                    for x in data["stops"]:
                        if x["instance"] == iid:
                            x.update(fields)
            if stop.get("failed"):
                try:
                    self.market.stop(iid)
                except MarketError as e:
                    if e.kind == "gone" and self._rows_ok and iid not in self._rows(fresh=True):
                        with self._edit() as data:
                            data["stops"] = [x for x in data["stops"] if x["instance"] != iid]
                    else:
                        update(next=now + STOP_EVERY)
                    continue
                update(failed=False, sent_at=now, next=now + STOP_EVERY)
                continue
            row = self._rows(fresh=True).get(iid)
            if row is None and not self._confirmed_gone(iid):
                update(next=now + STOP_EVERY)  # not listed: unclear, not proof
                continue
            if row is None or (row["status"] != "running" and not row["billing"]):
                with self._edit() as data:
                    data["stops"] = [x for x in data["stops"] if x["instance"] != iid]
                for sid in stop["sends"]:
                    self._warn(sid, "stop", None)
                if stop["after_run"]:
                    self._event("instance_stopped_after_run", instance=iid, sends=stop["sends"])
                continue
            resend = now - (stop.get("sent_at") or stop["requested"]) >= STOP_LIMIT
            alert = not stop["alerted"] and now - stop["requested"] >= STOP_LIMIT
            if resend:
                try:
                    self.market.stop(iid)
                except MarketError:
                    pass
            update(next=now + STOP_EVERY, alerted=stop["alerted"] or alert, **({"sent_at": now} if resend else {}))
            if alert:
                text = (f"Instance {iid} was told to stop but vast.ai still reports it running after "
                        f"{int(STOP_LIMIT // 60)} minutes. It is still billing: stop it on vast.ai now.")
                for sid in stop["sends"]:
                    self._warn(sid, "stop", text)
                self._event("send_alert", instance=iid, error=text)

    def _read_stops(self) -> list[dict]:
        with self._locked():
            return copy.deepcopy(self._read()["stops"])


def main(argv=None) -> int:
    """A read-only look: the sends on record."""
    manager = SendManager()
    print(json.dumps(manager.sends(), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
