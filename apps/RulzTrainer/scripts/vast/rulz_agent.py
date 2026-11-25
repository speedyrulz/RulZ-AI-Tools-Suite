#!/usr/bin/env python3
"""The instance side of rulz_remote.py: reads a RulzTrainer install's runs and queues new ones, as JSON or plain text.

rulz_remote.py pipes this file over ssh for every call (`cd /workspace/RulzTrainer && python - status`), so it works
on an instance that is already running, whatever its checkout's age. Only `update` changes the install and restarts
the WebUI. It needs the standard library, plus psutil, yaml and the repo's own modules where noted; keep it ASCII.

It talks to the trainer the way the WebUI does: a new run goes into output/_queue.json and the WebUI's queue thread
starts it, so the run shows on the Monitor and can never collide with one started from the page; Stop is the job
folder's STOP file; a running job is the pid in .webui_job.json.

Runs from the install folder (the working directory). Exit status: 0 fine, 1 refused or nothing found."""
from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

OUTPUT = Path("output")
JOB_FILE = Path(".webui_job.json")
QUEUE_FILE = OUTPUT / "_queue.json"
NOT_JOBS = {"_history", "_crops"}  # folders under output/ that are not jobs (history.SCAN_DIRS)
LOG_SUFFIXES = {".log", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".csv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SENT_CONFIG = ""  # a local config for start, base64: rulz_remote.py writes it into this line of the piped script
START_WAIT = 12.0  # seconds a start waits to see the WebUI's queue pick the run up (it looks every 3 s)
# status.json's per-step series: thousands of numbers, summed up instead of shown (job --full has them)
STATUS_SERIES = ("losses", "step_times", "grad_norms", "states", "probes", "samples", "saved")
SERIES_POINTS = 150  # the loss series sent with a snapshot, for rulz_fleet.py's small charts
SERIES_PROBES = 60  # and the probes sent with it
PROBE_COLUMNS = ("step", "face", "body", "combined", "face_ema", "body_ema", "combined_ema", "face_norm_ema",
                 "body_norm_ema", "best", "best_step", "since_improve", "lr_factor", "stage", "gold_ema", "copy")
# update: where the newest RulzTrainer comes from (provision.sh's defaults), and what it keeps on the instance
DEFAULT_REPO = "https://github.com/speedyrulz/RulZ-AI-Tools-Suite.git"
DEFAULT_BRANCH = "app/rulz-trainer"  # rebuilt from the suite's apps/RulzTrainer on every change to main
UPDATE_STATE = OUTPUT / "_update.json"  # output/ is git-ignored: an update leaves both alone
UPDATE_LOG = OUTPUT / "_update.log"
UPDATE_LOCK = OUTPUT / "_update.lock"  # held while an update (or its check) runs: two at once would mix their files
INSTALL_FILES = ("requirements.txt", "install.sh")  # a change in one of these runs install.sh again
# the WebUI's supervisor service, as provision.sh and the Dockerfile install it: copied again when it changed
SUPERVISOR_FILES = (("scripts/vast/rulztrainer.sh", "/opt/supervisor-scripts/rulztrainer.sh", 0o755),
                    ("scripts/vast/rulztrainer.conf", "/etc/supervisor/conf.d/rulztrainer.conf", 0o644))
# the requirements into the image's /venv/main (its torch kept), as provision.sh does
INSTALL_COMMAND = ["bash", "-c", "[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate; exec bash install.sh system"]
WEBUI_SERVICE = "rulztrainer"
# webui: the Instance Portal (Caddy) puts the WebUI on this port with its login in front; vast.ai maps it outside
PORTAL_FILE = Path("/etc/portal.yaml")
PORTAL_PORT = 7870  # the template's PORTAL_CONFIG: localhost:7870:17870:/:RulzTrainer
ENV_FILE = Path("/etc/environment")  # vast.ai's images write the container's variables here
FIRST_PROCESS_ENV = Path("/proc/1/environ")  # and docker starts the container's first process with them


class Refused(Exception):
    """A request that can't be carried out; its message is the answer."""


def show(data) -> None:
    print(json.dumps(data, indent=1, default=str))


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing, or torn mid-write
        return default


def read_text(path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def live_process(pid, created=None):
    """The psutil.Process with this pid while it runs, else None. A pid only counts with its create time when one is
    given: pids get reused."""
    import psutil

    try:
        proc = psutil.Process(int(pid))
        if created is not None and abs(proc.create_time() - float(created)) > 1.0:
            return None
        return proc if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE else None
    except (TypeError, ValueError, psutil.Error):
        return None


def job_process():
    """The psutil.Process of the job in .webui_job.json while it runs, else None (the WebUI's reattach_job)."""
    data = read_json(JOB_FILE) or {}
    return live_process(data.get("pid"), data.get("create_time")) if data.get("pid") is not None else None


def current_job_dir() -> Path | None:
    """The folder of the running job, or of the last one the WebUI started."""
    job_dir = (read_json(JOB_FILE) or {}).get("job_dir")
    return Path(job_dir) if job_dir and Path(job_dir).is_dir() else None


def running_dir() -> Path | None:
    return current_job_dir() if job_process() is not None else None


def job_dir(name: str | None) -> Path:
    """A job's folder by name; no name means the current job."""
    current = current_job_dir()
    if not name:
        if current is None:
            raise Refused("No job has run yet: name one (see jobs).")
        return current
    if Path(name).name != name or name in (".", ".."):
        raise Refused(f"Not a job name: {name!r}")
    if current is not None and current.name == name:
        return current  # a job with its own output folder
    d = OUTPUT / name
    if not (d / "status.json").is_file() and not (d / "config.yaml").is_file():
        raise Refused(f"No job named {name!r} (see jobs).")
    return d


def webui_running() -> bool:
    import psutil

    return any(a.endswith("webui.py") for proc in psutil.process_iter(["cmdline"])
               for a in proc.info.get("cmdline") or [])


def queue():
    from rulztrainer.train_queue import TrainQueue

    return TrainQueue(QUEUE_FILE)


def queue_view() -> dict:
    snap = queue().snapshot()
    return {"paused": snap["paused"],
            "waiting": [{"id": e["id"], "job": e["job"]} for e in snap["waiting"]],
            "recent": [{k: e.get(k) for k in ("id", "job", "started", "error", "message")}
                       for e in snap["recent"][:3]]}


def status_brief(status: dict) -> dict:
    """status.json without its per-step series: the numbers to read, the series' sizes and last entries."""
    brief = {k: v for k, v in status.items() if k not in STATUS_SERIES}
    saved = status.get("saved") or []
    brief["saved_count"], brief["last_saved"] = len(saved), saved[-1] if saved else None
    brief["samples_count"] = len(status.get("samples") or [])
    probes = status.get("probes") or []
    brief["probes_count"] = len(probes)
    if probes:
        brief["last_probe"] = {k: probes[-1].get(k) for k in PROBE_COLUMNS if probes[-1].get(k) is not None}
    return brief


def job_line(d: Path, active: Path | None) -> dict:
    """One job in a listing."""
    status = read_json(d / "status.json", {}) or {}
    likeness = status.get("likeness") or {}
    line = {"job": d.name, "running": active is not None and d.resolve() == active.resolve(),
            "step": status.get("step"), "total": status.get("total"), "done": status.get("done"),
            "error": status.get("error"), "avg_loss": status.get("avg_loss"), "elapsed": status.get("elapsed"),
            "best": likeness.get("best"), "best_step": likeness.get("best_step"),
            "modified": int(d.stat().st_mtime)}
    return {k: v for k, v in line.items() if v is not None}


# ---------------------------------------------------------------- read


def machine_status() -> dict:
    gpus = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        for row in out.strip().splitlines():
            name, used, total, util, temp = [c.strip() for c in row.split(",")]
            gpus.append({"name": name, "memory_used_mb": int(used), "memory_total_mb": int(total),
                         "utilization": int(util), "temperature": int(temp)})
    except (OSError, ValueError, subprocess.SubprocessError):
        pass  # no GPU or no driver tool: the list stays empty
    disk = shutil.disk_usage(".")
    active = running_dir()
    datasets = Path("datasets")
    names = sorted(p.name for p in datasets.iterdir() if p.is_dir() and not p.name.startswith("."))         if datasets.is_dir() else []  # not .rulz_incoming, where rulz_fleet.py unpacks a dataset before it is whole
    data = {"root": str(Path.cwd()), "webui_running": webui_running(), "gpus": gpus,
            "disk_free_gb": round(disk.free / 1e9, 1), "disk_total_gb": round(disk.total / 1e9, 1),
            "datasets": names,
            "dataset_marks": {n: mark for n in names if (mark := read_text(datasets / n / ".rulz_upload").strip())},
            "job_running": active is not None, "queue": queue_view(), "trainer": install_info(), "booted": booted(),
            "now": time.time()}  # the instance's clock, for comparing its file times with the fleet's
    update = update_state()
    if update:
        data["update"] = {k: update.get(k) for k in ("state", "started", "ended", "from", "to", "error", "note")
                          if update.get(k) is not None}
    return data


def trainer_version(text: str) -> str | None:
    found = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", text or "", re.M)
    return found.group(1) if found else None


def git(*args: str, check: bool = True, timeout: float = 300) -> str:
    """A git command in the install folder, its output stripped; a failure is a refusal (check) or ""."""
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as e:
        if check:
            raise Refused(f"git {args[0]}: {e}") from e
        return ""
    if done.returncode != 0:
        if check:
            raise Refused(f"git {args[0]} failed: {(done.stderr or done.stdout).strip()[-400:]}")
        return ""
    return done.stdout.strip()


def install_info() -> dict:
    """The RulzTrainer installed here: its version, and its commit and branch when the folder is a git checkout."""
    info = {"version": trainer_version(read_text("rulztrainer/__init__.py"))}
    if Path(".git").is_dir():
        info["commit"] = git("rev-parse", "--short=10", "HEAD", check=False, timeout=15)
        head = read_text(".git/HEAD").strip()
        if info["commit"] and head.startswith("ref: refs/heads/"):
            info["branch"] = head[len("ref: refs/heads/"):]
    return {k: v for k, v in info.items() if v}


def booted() -> float | None:
    """When this container started (its first process). It differs after a reboot, which is how rulz_fleet.py tells
    that one happened without comparing this machine's clock with its own."""
    try:
        import psutil

        return round(psutil.Process(1).create_time(), 1)
    except Exception:  # noqa: BLE001  (no psutil, or no such process: no answer)
        return None


def update_state() -> dict | None:
    """The last update's record, "interrupted" when it says running but its process is gone."""
    data = read_json(UPDATE_STATE)
    if not isinstance(data, dict):
        return None
    if data.get("state") == "running" and live_process(data.get("pid"), data.get("pid_created")) is None:
        data["state"] = "interrupted"
    return data


def cmd_status(args) -> int:
    data = machine_status()
    current = current_job_dir()
    if current is not None:
        data["job"] = dict(status_brief(read_json(current / "status.json", {}) or {}), job=current.name)
    show(data)
    return 0


def thin(values: list, limit: int = SERIES_POINTS) -> list[list]:
    """A per-step series as at most about `limit` [step, value] points, the last one always among them."""
    n = len(values)
    if n == 0:
        return []
    stride = max(1, -(-n // limit))
    picked = list(range(stride - 1, n, stride))
    if picked[-1:] != [n - 1]:
        picked.append(n - 1)
    return [[i + 1, round(float(values[i]), 5)] for i in picked if values[i] is not None]


def probe_rows(rows: list[dict], last: int = 0) -> list[dict]:
    """The probes' scores without their per-seed lists and image paths."""
    rows = rows[-last:] if last > 0 else rows
    return [{k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in PROBE_COLUMNS if r.get(k) is not None}
            for r in rows]


def job_dirs() -> list[Path]:
    """Every job folder, newest first."""
    dirs = [d for d in OUTPUT.iterdir() if d.is_dir() and d.name not in NOT_JOBS
            and (d / "status.json").is_file()] if OUTPUT.is_dir() else []
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def job_state(status: dict, running: bool) -> str:
    """running, done, error, or ended: the process is gone without finishing (killed)."""
    if running:
        return "running"
    if status.get("error"):
        return "error"
    return "done" if status.get("done") else "ended"


def all_job_dirs() -> list[Path]:
    """Every folder of output/ that can be a job (no limit, with or without a status.json), newest first."""
    dirs = [d for d in OUTPUT.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")) and d.name not in NOT_JOBS
            ] if OUTPUT.is_dir() else []
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def cmd_jobs(args) -> int:
    active = running_dir()
    if getattr(args, "all", False):
        lines = []
        for d in all_job_dirs():
            status = read_json(d / "status.json", {}) or {}
            running = active is not None and d.resolve() == active.resolve()
            lines.append(dict(job_line(d, active), run_id=status.get("run_id"), state=job_state(status, running),
                              has_status=(d / "status.json").is_file()))
        show(lines)
        return 0
    show([job_line(d, active) for d in job_dirs()])
    return 0


def cmd_snapshot(args) -> int:
    """Everything rulz_fleet.py reads in one look: the machine, every job with its state and run id, and the current
    job's numbers with its loss and probe series thinned for a small chart."""
    data = machine_status()
    active = running_dir()
    jobs = []
    for d in job_dirs()[:args.jobs]:
        status = read_json(d / "status.json", {}) or {}
        running = active is not None and d.resolve() == active.resolve()
        jobs.append(dict(job_line(d, active), run_id=status.get("run_id"), state=job_state(status, running)))
    data["jobs"] = jobs
    data["tests_running"] = tests_running_count()
    data["watchdog"] = watchdog_block(data["now"])
    data["autostop"] = autostop_block()
    current = current_job_dir()
    if current is not None:
        status = read_json(current / "status.json", {}) or {}
        times = [t for t in (status.get("step_times") or [])[-20:] if t]
        data["job"] = dict(status_brief(status), job=current.name, running=active is not None,
                           step_time=round(sum(times) / len(times), 3) if times else None)
        data["series"] = {"losses": thin(status.get("losses") or []),
                          "probes": probe_rows(status.get("probes") or [], SERIES_PROBES)}
    show(data)
    return 0


def cmd_record(args) -> int:
    """One run for rulz_fleet.py's history: its settings, its numbers, every probe's scores and the thinned loss."""
    d = job_dir(args.name)
    status = read_json(d / "status.json", {}) or {}
    config = {}
    try:
        import yaml

        config = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) or {}
    except (ImportError, OSError, ValueError):
        pass  # no config kept: the record goes without
    active = running_dir()
    show({"job": d.name, "run_id": status.get("run_id"), "config": config, "status": status_brief(status),
          "state": job_state(status, active is not None and d.resolve() == active.resolve()),
          "probes": probe_rows(status.get("probes") or []), "losses": thin(status.get("losses") or [])})
    return 0


def cmd_job(args) -> int:
    d = job_dir(args.name)
    status = read_json(d / "status.json", {}) or {}
    data = status if args.full else status_brief(status)
    show(dict(data, job=d.name, folder=str(d), files=sorted(p.name for p in d.iterdir() if p.is_file())[:args.files]))
    return 0


def cmd_log(args) -> int:
    path = job_dir(args.name) / ("console.log" if args.console else "train.log")
    if not path.is_file():
        raise Refused(f"No {path.name} in {path.parent}.")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if args.grep:
        pattern = re.compile(args.grep)
        lines = [line for line in lines if pattern.search(line)]
    print("\n".join(lines[-args.n:] if args.n > 0 else lines))
    return 0


def cmd_probes(args) -> int:
    path = job_dir(args.name) / "likeness.jsonl"
    if not path.is_file():
        raise Refused(f"No likeness.jsonl in {path.parent}: this run has no likeness probes.")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:  # the line being written
            continue
    if args.last > 0:
        rows = rows[-args.last:]
    show(rows if args.full else [{k: r[k] for k in PROBE_COLUMNS if r.get(k) is not None} for r in rows])
    return 0


def cmd_config(args) -> int:
    d = job_dir(args.name)
    if not args.diff and not args.json:
        print((d / "config.yaml").read_text(encoding="utf-8"))
        return 0
    import yaml

    if not args.diff:
        show(yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) or {})
        return 0

    ours = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) or {}
    theirs = yaml.safe_load((job_dir(args.diff) / "config.yaml").read_text(encoding="utf-8")) or {}
    show({k: [ours.get(k), theirs.get(k)] for k in sorted(set(ours) | set(theirs)) if ours.get(k) != theirs.get(k)})
    return 0


def cmd_poll(args) -> int:
    """One short look at a job, for wait: where it stands, and whether it ended well."""
    snap = queue().snapshot()
    if args.name in [e["job"] for e in snap["waiting"]]:
        show({"job": args.name, "state": "queued", "paused": snap["paused"]})
        return 0
    active = running_dir()
    d = active if active is not None and args.name in (None, active.name) else job_dir(args.name)
    status = read_json(d / "status.json", {}) or {}
    if d == active:
        state = "running"
    elif status.get("error"):
        state = "error"
    else:
        state = "done" if status.get("done") else "ended"  # ended: the process is gone without finishing (killed)
    show({"job": d.name, "state": state, **{k: status.get(k) for k in ("stage", "step", "total", "loss", "avg_loss",
                                                                       "eta", "error")}})
    return 0


def cmd_files(args) -> int:
    """A job's files to fetch: {"base": folder, "files": [{"path" under base, "size", "mtime"}]}. --what picks logs
    (default), probes, samples, maps (the training_maps folder), loras, best or all; --step keeps that step's images
    and LoRA."""
    if getattr(args, "tests", None):
        if args.name or args.what or args.step is not None:
            raise Refused("--tests cannot be combined with a job, --what or --step")
        d = test_dir(args.tests)
        files = []
        for p in sorted(d.rglob("*")):
            if p.is_file() and not p.is_symlink() and d.resolve() in p.resolve().parents:
                st = p.stat()
                files.append({"path": p.relative_to(d).as_posix(), "size": st.st_size, "mtime": int(st.st_mtime)})
        show({"base": str(d.resolve()), "files": files})
        return 0
    d = job_dir(args.name)
    tag = f"{args.step:06d}" if args.step is not None else None
    what = set(args.what or ["logs"])
    if "tree" in what:  # every file of the run but its resume states (the check before a destroy)
        if what != {"tree"}:
            raise Refused("--what tree cannot be combined with other kinds")
        show({"base": str(d.parent.resolve()), "files": tree_listing(d)})
        return 0
    picked: list[Path] = []
    if what & {"logs", "all"}:  # sub-folders too: a race's race.json, a soup's soup.json (the WebUI's job_logs)
        picked += sorted(p for pattern in ("*", "*/*") for p in d.glob(pattern)
                         if p.is_file() and p.suffix.lower() in LOG_SUFFIXES)
    for kind in ("probes", "samples"):
        if what & {kind, "all"}:
            picked += [p for p in sorted((d / kind).glob("*")) if p.suffix.lower() in IMAGE_SUFFIXES
                       and (tag is None or tag in p.name)]
    if what & {"maps", "all"}:  # training maps: the picture crops, probes.json and maps.jsonl (all captures, any --step)
        picked += sorted(p for p in (d / "training_maps").glob("*") if p.is_file())
    loras = sorted(d.glob("*.safetensors"))
    if what & {"loras", "all"}:
        picked += [p for p in loras if tag is None or tag in p.name]
    elif "best" in what:
        picked += [p for p in loras if p.stem.endswith("_best")]
    files = []
    for p in dict.fromkeys(picked):
        st = p.stat()
        files.append({"path": p.relative_to(d.parent).as_posix(), "size": st.st_size, "mtime": int(st.st_mtime)})
    show({"base": str(d.parent.resolve()), "files": files})
    return 0


# ---------------------------------------------------------------- act

TEST_FINAL = {"refused", "done", "stopped", "failed"}


def tests_root():
    root = OUTPUT / "_tests"
    if root.is_symlink() or not root.resolve().is_relative_to(OUTPUT.resolve()):
        raise Refused("Tests folder is outside output")
    return root


def test_dir(test_id, must_exist=True):
    if not isinstance(test_id, str) or not re.fullmatch(r"[a-z0-9-]+", test_id):
        raise Refused(f"Bad test_id: {test_id!r}")
    d = tests_root() / test_id
    if d.is_symlink() or (must_exist and not d.is_dir()):
        raise Refused(f"No test named {test_id!r}")
    return d


def test_status(d):
    status = read_json(d / "status.json") or {}
    if status.get("state") not in TEST_FINAL:
        pid = status.get("pid")
        if (pid is None and time.time() - d.stat().st_mtime > 120) or (
                pid is not None and live_process(pid, created=status.get("pid_created")) is None):
            status = dict(status, state="failed", error="Test process is no longer running")
    return status


def running_tests():
    root = tests_root()
    active = []
    for d in root.iterdir() if root.is_dir() else ():
        if d.is_dir() and not d.is_symlink():
            status = test_status(d)
            if status.get("state") not in TEST_FINAL:
                active.append((d, status))
    return active


def cmd_test_start(args):
    try:
        spec = json.loads(base64.b64decode(SENT_CONFIG).decode("utf-8") if SENT_CONFIG else sys.stdin.read())
    except (ValueError, UnicodeError) as e:
        raise Refused(f"Invalid spec JSON: {e}")
    if not isinstance(spec, dict):
        raise Refused("Spec must be a JSON object")
    if spec.get("test_id") is not None:
        test_dir(spec["test_id"], must_exist=False)
    lock = hold_lock(OUTPUT / "_test-start.lock")
    if lock is None:
        raise Refused("Another test start is in progress")
    with lock:
        return start_test_locked(spec)


def start_test_locked(spec):
    if job_process() is not None:
        # Training can use auxiliary GPUs dynamically, including auto placement.
        raise Refused("A training job is running; its GPU use cannot be determined safely")
    test_id = spec.get("test_id")
    d = test_dir(test_id, must_exist=False) if test_id else None
    for other, status in running_tests():
        if other == d:
            show({"test_id": test_id, "state": status.get("state")})
            return 0
        raise Refused(f"Test {other.name} is already running")
    if d is not None and d.exists():
        status = read_json(d / "status.json") or {}
        if status.get("state") in TEST_FINAL:
            raise Refused(f"Test {test_id} already ran")
        status.update(state="queued", pid=None, pid_created=None)
        write_json(d / "status.json", status)
    else:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "spec.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "rulztrainer.loratest", "new", "--spec", str(path)],
                                    capture_output=True, text=True)
        if result.returncode:
            raise Refused((result.stderr or result.stdout or "Could not create test").strip())
        test_id = result.stdout.strip()
        d = test_dir(test_id)
    try:
        with (d / "run.log").open("ab") as log:
            proc = subprocess.Popen([sys.executable, "-m", "rulztrainer.loratest", "run", str(d)],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        status = read_json(d / "status.json") or {}
        if status.get("pid") is None and status.get("state") not in TEST_FINAL:
            import psutil
            try:
                created = psutil.Process(proc.pid).create_time()
            except psutil.Error:
                created = None
            status.update(pid=proc.pid, pid_created=created)
            write_json(d / "status.json", status)
    except OSError as e:
        raise Refused(f"Could not start test: {e}")
    show({"test_id": test_id, "state": "queued"})
    return 0


def cmd_test_status(args):
    if args.test_id:
        show(test_status(test_dir(args.test_id)))
    else:
        root = tests_root()
        dirs = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()] if root.is_dir() else []
        entries = [(d, test_status(d), read_json(d / "spec.json") or {}) for d in dirs]
        entries.sort(key=lambda e: (e[1].get("started") or e[2].get("created") or e[0].name, e[0].name), reverse=True)
        show({"tests": [status for _, status, _ in entries]})
    return 0


def cmd_test_stop(args):
    d = test_dir(args.test_id)
    (d / "STOP").touch()
    show({"test_id": args.test_id, "stopping": True})
    return 0


def parse_overrides(items: list[str]) -> dict:
    """KEY=VALUE pairs, the value JSON when it parses (like rulztrainer.train --set)."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise Refused(f"Not KEY=VALUE: {item!r}")
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def base_config(args):
    """The config a new run starts from: a local file sent along, or an earlier job's."""
    from rulztrainer.config import TrainConfig

    if SENT_CONFIG:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_bytes(base64.b64decode(SENT_CONFIG))
            return TrainConfig.load_report(path)[0]
    if args.base:
        return TrainConfig.load_report(job_dir(args.base) / "config.yaml")[0]
    raise Refused("Say what to start from: --from JOB or --config FILE.")


def race_rows(data: dict, wanted: list[str]) -> dict:
    """A race's config narrowed to the named configurations, for a race split over several machines: two or more
    stay a race among themselves, one alone becomes a plain run with that row's settings (the trainer's own
    variant_configs, so its values are read the way the race reads them)."""
    from rulztrainer.config import TrainConfig

    rows = data.get("race_configs") or []
    named = {str(row.get("name") or f"c{i + 1}"): row for i, row in enumerate(rows)}
    missing = [n for n in wanted if n not in named]
    if not data.get("race") or missing or not wanted:
        raise Refused(f"No race configuration named {', '.join(missing) or '(none given)'}: this config has "
                      f"{', '.join(named) if data.get('race') and named else 'no race'}.")
    if data.get("race_follow"):
        raise Refused("A relay (race_follow) continues every configuration from each round's winner: it can't be "
                      "split over machines.")
    picked = [dict(named[n], name=n) for n in wanted]
    if len(picked) > 1:
        return dict(data, race_configs=picked)
    from rulztrainer import race

    return race.variant_configs(TrainConfig.from_dict(dict(data, race_configs=picked)))[0][1].to_dict()


def moved_aside_name(job_dir: Path) -> Path:
    """Where start --overwrite moves an earlier run's folder: <job>_replaced_<date-time> beside it (a number added if
    that is taken too), so the new run starts in an empty folder and the old run's files stay readable."""
    base = job_dir.with_name(f"{job_dir.name}_replaced_{time.strftime('%Y%m%d-%H%M%S')}")
    out, n = base, 2
    while out.exists():
        out, n = base.with_name(f"{base.name}_{n}"), n + 1
    return out


def cmd_start(args) -> int:
    """Queue a run. The checks of the WebUI's queue_run, the trainer's own validate, and refusing to overwrite an
    earlier run's folder."""
    active_tests = running_tests()
    if active_tests:
        raise Refused(f"Test {active_tests[0][0].name} is already running")
    from rulztrainer.config import TrainConfig, apply_preset

    data = base_config(args).to_dict()
    before = dict(data)
    if args.preset:
        data = apply_preset(data, args.preset)
    data.update(parse_overrides(args.set))
    if args.race_rows:
        try:
            data = race_rows(data, [n for n in args.race_rows.split(",") if n])
        except (TypeError, ValueError) as e:
            raise Refused(f"A setting is invalid: {e}") from e
    data["job_name"] = args.job
    try:
        cfg = TrainConfig.from_dict(data)
        cfg.validate()  # what the trainer checks first, a race's own checks included
    except (TypeError, ValueError) as e:
        raise Refused(f"A setting is invalid: {e}") from e
    q = queue()
    active = running_dir()
    taken = {e["job"] for e in q.snapshot()["waiting"]} | ({active.name} if active is not None else set())
    try:  # an install from before runplan.py: its WebUI starts into any folder, so the old rule (no folder) stays
        from rulztrainer import runplan
    except ImportError:
        runplan = None
    if (runplan.name_in(cfg.job_name, taken) if hasattr(runplan, "name_in") else cfg.job_name in taken):
        free = f" ({runplan.next_free_name(cfg.job_name, taken, cfg.output_dir)!r} is free)" if runplan else ""
        raise Refused(f"Job name {cfg.job_name!r} is already running or queued{free}.")
    if runplan is None:
        if cfg.job_dir().exists() and not args.overwrite:
            raise Refused(f"{cfg.job_dir()} already exists: pick a new job name, or --overwrite to replace that run.")
        aside = None
    else:
        # the WebUI refuses to start a new run in a folder that holds an earlier run's output (a folder whose run
        # saved nothing is free, and so is carrying that run on from its own state); --overwrite moves the earlier
        # run's folder aside first, so nothing of it is written over. ("already exists": rulz_fleet.py tries another
        # instance on it.)
        free = runplan.job_name_taken(cfg.job_name, cfg.output_dir, cfg.resume_state, taken)
        if free is not None and not args.overwrite:
            raise Refused(f"{cfg.job_dir()} already exists and holds an earlier run's files: pick a new job name "
                          f"({free!r} is free), or --overwrite to move that run's folder aside and start afresh.")
        aside = cfg.job_dir() if free is not None else None
    missing = [sp.path or "(none picked)" for sp in cfg.dataset_specs() if not sp.path or not Path(sp.path).is_dir()]
    if missing:
        raise Refused(f"Dataset folder not found: {', '.join(missing)}")
    reg = getattr(cfg, "reg_dir", "")  # (an install from before the regularisation dataset has no such setting)
    if reg and not Path(reg).is_dir():
        raise Refused(f"Regularisation folder not found: {reg}")
    plan = {"job": cfg.job_name, "arch": cfg.arch_name(), "steps": cfg.steps,
            "datasets": [sp.path for sp in cfg.dataset_specs()],
            "changed": {k: [before.get(k), v] for k, v in cfg.to_dict().items()
                        if before.get(k) != v and k != "job_name"}}
    if aside is not None:
        plan["moved_aside"] = str(moved_aside_name(aside))
    if args.dry_run:
        show(dict(plan, dry_run=True, job_running=active is not None, queue=queue_view()))
        return 0
    if aside is not None:
        try:
            aside.rename(plan["moved_aside"])
        except OSError as e:  # a file of it is open, no permission: nothing was moved and nothing is queued
            raise Refused(f"Could not move the earlier run's folder {aside} aside ({e}): nothing was queued. Pick a "
                          f"new job name, or close what is reading that folder.") from e
    try:
        if args.resume_queue:
            q.set_paused(False)
        # the duplicate check is made inside the queue's own lock (cmd_start's own check above is not: two starts
        # that overlap could both pass it); an install from before that option is left with the check above
        extra = {"refuse_duplicate": True} if "refuse_duplicate" in inspect.signature(q.add).parameters else {}
        try:
            entry = q.add(cfg.to_dict(), label=cfg.arch_name(), **extra)
        except ValueError as e:  # train_queue.DuplicateJob
            raise Refused(f"Job name {cfg.job_name!r} is already running or queued.") from e
    except BaseException:  # not queued after all: the earlier run goes back under its name
        if aside is not None and not aside.exists():
            try:
                Path(plan["moved_aside"]).rename(aside)
            except OSError:
                pass
        raise
    return report_start(q, entry, plan, active is not None)


def report_start(q, entry: dict, plan: dict, busy: bool) -> int:
    """Say what became of a queued run: it started, it failed to start, or why it is still waiting."""
    plan = dict(plan, id=entry["id"])
    if not webui_running():
        show(dict(plan, state="queued", note="The WebUI is not running, so nothing starts this run: "
                                             "supervisorctl restart rulztrainer"))
        return 1
    deadline = time.time() + START_WAIT
    while not busy and time.time() < deadline:
        snap = q.snapshot()
        done = next((r for r in snap["recent"] if r["id"] == entry["id"]), None)
        if done is not None:  # first: a run that fails to start also leaves the queue paused, with this run waiting
            ok = "started" in done
            show(dict(plan, state="running" if ok else "failed", message=done.get("message")))
            return 0 if ok else 1
        if snap["paused"]:
            break
        time.sleep(0.5)
    snap = q.snapshot()
    note = ("The queue is paused: queue resume starts it." if snap["paused"]
            else "It starts when the runs ahead of it end." if busy else "The WebUI has not picked it up yet.")
    show(dict(plan, state="queued", paused=snap["paused"], position=len(snap["waiting"]), note=note))
    return 0


def kill_tree(proc) -> None:
    """Kill a job and the processes it started, children first (as the WebUI's Kill): the probe process holds a second
    model on the GPU and would go on rendering after a kill of the trainer alone."""
    try:
        children = proc.children(recursive=True)  # before the kill: then they have no parent
    except Exception:  # noqa: BLE001  (gone already, access denied)
        children = []
    for child in children:
        try:
            child.kill()
        except Exception:  # noqa: BLE001  (ended on its own)
            pass
    proc.kill()


def cmd_stop(args) -> int:
    data = read_json(JOB_FILE) or {}  # one read: the process and the folder of the same job
    proc = live_process(data.get("pid"), data.get("create_time")) if data.get("pid") is not None else None
    current = Path(data["job_dir"]) if data.get("job_dir") and Path(data["job_dir"]).is_dir() else None
    if proc is None or current is None:
        raise Refused("No job running.")
    if args.job and current.name != args.job:  # the run meant has ended and the queue went on: touch nothing
        raise Refused(f"{args.job!r} is not the running job ({current.name!r} is): nothing was stopped.")
    if not args.kill:
        (current / "STOP").touch()
        show({"job": current.name, "state": "stopping",
              "note": "The trainer saves the LoRA after the current step, then ends."})
        return 0
    q = queue()
    if q.snapshot()["waiting"]:  # as the WebUI's Kill: stopping everything, so the queued runs wait for a resume
        q.set_paused(True)  # (before the kill: else the WebUI could start the next run in between)
    kill_tree(proc)
    show({"job": current.name, "state": "killed", "note": "No final save.", "queue": queue_view()})
    return 0


def cmd_queue(args) -> int:
    q = queue()
    if args.action in ("pause", "resume"):
        q.set_paused(args.action == "pause")
    elif args.action == "remove" and (not args.id or not q.remove(args.id)):
        raise Refused(f"No waiting run with id {args.id!r}.")
    show(queue_view())
    return 0


# ---------------------------------------------------------------- the install itself: update, the WebUI's address


class UpdateFailed(Exception):
    pass


def container_env(env_file: Path | None = None, first_process: Path | None = None, own=None) -> dict[str, str]:
    """The container's variables. An ssh session inherits none of what docker started the container with, so they
    come from /etc/environment (where vast.ai's images write them), then the first process's environment, and this
    process's own on top."""
    values: dict[str, str] = {}
    for line in read_text(env_file or ENV_FILE).splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name and not name.startswith("#"):
            values[name.strip()] = value.strip().strip("'\"")
    try:
        raw = (first_process or FIRST_PROCESS_ENV).read_bytes()
    except OSError:  # not Linux, or not allowed
        raw = b""
    for item in raw.split(b"\0"):
        name, sep, value = item.decode("utf-8", "replace").partition("=")
        if sep and name:
            values[name] = value
    values.update(os.environ if own is None else own)
    return values


def public_url(url: str) -> str:
    """A git remote without the user name or token some clones carry in it."""
    return re.sub(r"^([a-z][a-z0-9+.-]*://)[^/@]+@", r"\1", url or "", flags=re.I)


def update_source(branch: str = "") -> tuple[str, str, bool]:
    """Where the newest RulzTrainer comes from, and whether this folder is a checkout (its HEAD names a commit):
    the checkout's own origin and branch, else provision.sh's (RULZ_REPO, RULZ_BRANCH, or their defaults)."""
    env = container_env()
    info = install_info()
    checkout = bool(info.get("commit"))
    origin = git("remote", "get-url", "origin", check=False, timeout=15) if Path(".git").is_dir() else ""
    url = origin or env.get("RULZ_REPO") or DEFAULT_REPO
    return url, branch or info.get("branch") or env.get("RULZ_BRANCH") or DEFAULT_BRANCH, checkout


def update_plan(branch: str = "") -> dict:
    """Fetch the newest RulzTrainer and say what an update to it would change. A folder that is no checkout (a
    Docker image's copy) gets a .git of its own for the fetch; its files stay as they are until the update."""
    url, branch, checkout = update_source(branch)
    if not Path(".git").is_dir():
        git("init", "--quiet")
    if not git("remote", "get-url", "origin", check=False, timeout=15):
        git("remote", "add", "origin", url)
    git("fetch", "--quiet", "--no-tags", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
    target = f"origin/{branch}"
    current = install_info()
    latest = {"version": trainer_version(git("show", f"{target}:rulztrainer/__init__.py", check=False)),
              "commit": git("rev-parse", "--short=10", target)}
    plan = {"checkout": checkout, "repo": public_url(url), "branch": branch, "current": current,
            "latest": {k: v for k, v in latest.items() if v}, "behind": None, "ahead": None, "up_to_date": False,
            "changed": [], "changed_count": 0, "installs": True}
    if checkout:
        ahead, behind = (int(n) for n in git("rev-list", "--left-right", "--count", f"HEAD...{target}").split())
        changed = [n for n in git("diff", "--name-only", "-z", "HEAD").split("\0") if n.strip()]
        # files here that git does not track (a sync's new file, an ignored one) but the newest release brings:
        # the update writes over them
        tracked = set(git("ls-files", "-z").split("\0"))
        changed += [n for n in git("ls-tree", "-r", "--name-only", "-z", target).split("\0")
                    if n and n not in tracked and Path(n).exists()]
        plan.update(ahead=ahead, behind=behind, changed=changed[:40], changed_count=len(changed),
                    up_to_date=git("rev-parse", "HEAD") == git("rev-parse", target),
                    installs=bool(git("diff", "--name-only", "HEAD", target, "--", *INSTALL_FILES)))
    active = running_dir()
    plan.update(job_running=active is not None, job=active.name if active is not None else None)
    return plan


def copy_service_files(log) -> bool:
    """The WebUI's supervisor files where provision.sh put them, when the update changed them. True when it did."""
    changed = False
    for source, target, mode in SUPERVISOR_FILES:
        source, target = Path(source), Path(target)
        if not source.is_file() or not target.parent.is_dir():
            continue
        if target.is_file() and target.read_bytes() == source.read_bytes():
            continue
        shutil.copyfile(source, target)
        os.chmod(target, mode)
        log.write(f"== service: {source} -> {target}\n")
        changed = True
    return changed


def hold_lock(path: Path):
    """An open file holding an OS lock on path, or None when another process holds it. Files Python opens are not
    inherited by the commands an update runs, so the lock ends with this process."""
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


def supervisorctl() -> list[str] | None:
    """The command that restarts the WebUI's service (vast.ai's images run it under supervisor), or None."""
    found = shutil.which("supervisorctl") or next(
        (p for p in ("/usr/local/bin/supervisorctl", "/usr/bin/supervisorctl", "/venv/main/bin/supervisorctl")
         if os.access(p, os.X_OK)), None)  # vast.ai's images pip-install it into /usr/local/bin
    return [found] if found else None


def hold_queue():
    """Pause the WebUI's own queue while the code changes under it (a waiting run would start on new code with old
    packages); returns how to put it back as it was. Never fails an update: no queue, nothing to hold."""
    try:
        q = queue()
        was = bool(q.snapshot()["paused"])
        q.set_paused(True)
    except Exception:  # noqa: BLE001  (an old or broken install: the update goes on without)
        return lambda: None

    def restore() -> None:
        try:
            q.set_paused(was)
        except Exception:  # noqa: BLE001
            pass

    return restore


def tail(path: Path, lines: int = 12) -> str:
    return "\n".join(read_text(path).strip().splitlines()[-lines:])


def run_update(plan: dict, reinstall: bool = False) -> int:
    """Put the newest code in place, install the requirements when they changed, renew the supervisor files, and
    restart the WebUI on the new code. A dropped ssh connection does not stop it half way: it goes on, and its
    record (output/_update.json, the log next to it) says how it ended."""
    for name in ("SIGHUP", "SIGPIPE"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_IGN)
    me = live_process(os.getpid())
    record = {"state": "running", "pid": os.getpid(), "pid_created": me.create_time() if me else None,
              "started": round(time.time(), 1), "branch": plan["branch"], "from": plan["current"],
              "to": plan["latest"], "steps": []}
    write_json(UPDATE_STATE, record)
    restarted, note = False, ""
    restore_queue = hold_queue()
    with open(UPDATE_LOG, "w", encoding="utf-8") as log:
        def step(name: str, cmd: list[str], fatal: bool = True) -> bool:
            log.write(f"== {name}: {' '.join(cmd)}\n")
            log.flush()
            done = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            log.flush()
            record["steps"].append(name)
            if done.returncode != 0 and fatal:
                raise UpdateFailed(f"{name} failed (exit {done.returncode}):\n{tail(UPDATE_LOG)}")
            return done.returncode == 0

        try:
            step("code", ["git", "checkout", "--quiet", "--force", "-B", plan["branch"], f"origin/{plan['branch']}"])
            if plan.get("installs") or reinstall:
                step("requirements", list(INSTALL_COMMAND))
            service = copy_service_files(log)
            ctl = supervisorctl()
            if ctl is None:
                note = "No supervisorctl here: restart the WebUI yourself to run the new code."
            else:
                if service:
                    step("service", [*ctl, "update"], fatal=False)
                restarted = step("restart", [*ctl, "restart", WEBUI_SERVICE], fatal=False)
                if not restarted:
                    note = f"The code is new, but the WebUI did not restart:\n{tail(UPDATE_LOG, 4)}"
        except (UpdateFailed, OSError, subprocess.SubprocessError) as e:
            record.update(state="failed", ended=round(time.time(), 1), error=str(e)[-800:])
            write_json(UPDATE_STATE, record)
            show(record)
            return 1
        finally:
            restore_queue()
    record.update(state="done", ended=round(time.time(), 1), to=install_info(), restarted=restarted,
                  installed="requirements" in record["steps"], note=note or None)
    write_json(UPDATE_STATE, record)
    show(record)
    return 0


def cmd_update(args) -> int:
    """Bring this install to the newest RulzTrainer (the branch it was cloned from, app/rulz-trainer by default) and
    restart the WebUI on it. --check only says what would change. Refuses while a run goes (--force), and when files
    were changed here, by rulz_remote.py sync say (--discard replaces them with the newest release's)."""
    last = update_state()
    lock = hold_lock(UPDATE_LOCK)  # kept until this returns; the OS lets go when the process ends
    if lock is None or (last and last.get("state") == "running"):
        going = last and last.get("state") == "running"
        since = f", started {max(0, round((time.time() - (last.get('started') or time.time())) / 60))} min ago" \
            if going else ""  # minutes, not a clock time: this machine's time zone is not the reader's
        raise Refused(f"An update (or its check) is already going here{since}: see {UPDATE_LOG}.")
    plan = update_plan(args.branch)
    # an update that failed or died after the code moved on: the code compares as current, the rest is redone
    plan["unfinished"] = bool(last) and last.get("state") in ("failed", "interrupted")
    if plan["unfinished"]:
        plan["installs"] = True
    if args.check:
        show(plan)
        return 0
    if plan["job_running"] and not args.force:
        raise Refused(f"{plan['job']} is running: an update changes the code under it. Wait for it to end, stop it, "
                      "or --force.")
    if plan["changed_count"] and not args.discard:
        names = ", ".join(plan["changed"][:5]) + (", ..." if plan["changed_count"] > 5 else "")
        raise Refused(f"{plan['changed_count']} file(s) changed here since the last update ({names}), by "
                      "rulz_remote.py sync perhaps: --discard replaces them with the newest release's.")
    if plan["up_to_date"] and not plan["changed_count"] and not args.reinstall and not plan["unfinished"]:
        show(dict(plan, state="current", note=f"Already the newest RulZ AI Trainer "
                                              f"({plan['current'].get('version') or plan['current'].get('commit')})."))
        return 0
    return run_update(plan, args.reinstall)


def portal_entry(text: str, config: str = "") -> tuple[int, str]:
    """The Instance Portal's port and path for RulzTrainer: /etc/portal.yaml, else PORTAL_CONFIG
    (host:external:internal:path:name entries split by |), else the template's 7870."""
    try:
        import yaml

        apps = (yaml.safe_load(text or "") or {}).get("applications") or {}
    except Exception:  # noqa: BLE001  (no yaml, or not the file expected: the other sources)
        apps = {}
    for key, app in apps.items() if isinstance(apps, dict) else ():
        if isinstance(app, dict) and "rulztrainer" in f"{key} {app.get('name') or ''}".lower():
            try:
                return int(app.get("external_port")), str(app.get("open_path") or "/")
            except (TypeError, ValueError):
                break
    for entry in (config or "").split("|"):
        parts = entry.split(":")
        if len(parts) >= 5 and "rulztrainer" in parts[4].lower() and parts[1].isdigit():
            return int(parts[1]), parts[3] or "/"
    return PORTAL_PORT, "/"


def portal_scheme(port: int, env: dict) -> str:
    """http or https: whether the portal's port answers a TLS handshake, else what ENABLE_HTTPS says."""
    import socket
    import ssl

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname="localhost"):
                return "https"
    except ssl.SSLError:
        return "http"
    except OSError:
        pass
    return "https" if str(env.get("ENABLE_HTTPS", "")).strip().lower() in ("1", "true", "yes") else "http"


def cmd_webui(args) -> int:
    """How this instance's WebUI is reached from outside: the Instance Portal's port for it, the port vast.ai maps
    that to and the public address (when the container knows them), http or https, and the portal's token, which
    signs a browser in when a link carries it as ?token=."""
    env = container_env()
    port, path = portal_entry(read_text(PORTAL_FILE), env.get("PORTAL_CONFIG", ""))
    host_port = env.get(f"VAST_TCP_PORT_{port}", "")
    show({"port": port, "path": path, "scheme": portal_scheme(port, env), "ip": env.get("PUBLIC_IPADDR") or None,
          "host_port": int(host_port) if host_port.isdigit() else None, "token": env.get("OPEN_BUTTON_TOKEN") or None,
          "webui_running": webui_running()})
    return 0


# ---------------------------------------------------------------- files sent from the PC: models/sent

SENT_DIR = Path("models") / "sent"
SENT_RECORDS = SENT_DIR / ".ok"  # one small record per finished file or folder (path, size, sha256)
FETCH_DIR = OUTPUT / "_fetch"  # file-fetch's state files
WATCH_DIR = OUTPUT / "_watchdog"
WATCH_FILE = WATCH_DIR / "watch.py"
WATCH_STATE = WATCH_DIR / "state.json"
WATCH_ALIVE = WATCH_DIR / "alive"  # touched by the PC's manager at each look (amendment 1)
AUTOSTOP_FILE = OUTPUT / "_autostop.json"  # machine.AutoStop's record
WATCH_CONF = Path("/etc/supervisor/conf.d/rulz-watch.conf")
WATCH_PROGRAM_NAME = "rulz-watch"
WATCH_START_WAIT = 6.0  # seconds arm waits to see the program write its pid
VAST_API = "https://console.vast.ai/api/v0"
HASH_BLOCK = 8 * 1024 * 1024


def sent_parts(rel) -> tuple[str, ...]:
    """The parts of a posix path below models/sent/<12 hex digits>/, or a refusal. Names come from digests and a cleaned
    basename on the PC; nothing here may leave models/sent."""
    if not isinstance(rel, str) or not rel or any(c in rel for c in "\\\0\n\r"):
        raise Refused(f"Not a path below models/sent: {rel!r}")
    parts = PurePosixPath(rel).parts
    if ("/".join(parts) != rel or len(parts) < 4 or parts[:2] != ("models", "sent")
            or not re.fullmatch(r"[0-9a-f]{12}", parts[2]) or any(p in (".", "..", "") for p in parts)):
        raise Refused(f"Not a path below models/sent: {rel!r}")
    return parts


def sent_path(rel: str) -> Path:
    sent_parts(rel)
    full = Path(rel)
    if not full.resolve().is_relative_to(SENT_DIR.resolve()):  # a link that leads out
        raise Refused(f"Not a path below models/sent: {rel!r}")
    return full


def inner_parts(rel) -> tuple[str, ...]:
    """A path inside a sent folder (from the manifest): relative, no '..', no odd characters."""
    if not isinstance(rel, str) or not rel or any(c in rel for c in "\\\0\n\r"):
        raise Refused(f"Not a path inside the folder: {rel!r}")
    parts = PurePosixPath(rel).parts
    if "/".join(parts) != rel or rel.startswith("/") or any(p in (".", "..", "") for p in parts):
        raise Refused(f"Not a path inside the folder: {rel!r}")
    return parts


def sent_record_path(rel: str) -> Path:
    return SENT_RECORDS / (hashlib.sha1(rel.encode("utf-8")).hexdigest()[:24] + ".json")


def sent_record(rel: str) -> dict | None:
    data = read_json(sent_record_path(rel))
    return data if isinstance(data, dict) and data.get("path") == rel else None


def write_sent_record(rel: str, size: int, sha256: str = "", tree: bool = False) -> None:
    write_json(sent_record_path(rel), {"path": rel, "size": size, "sha256": sha256, "tree": tree,
                                       "at": round(time.time())})


def sent_complete(rel: str, size: int | None = None, tree: bool = False) -> bool:
    """Whole and recorded: the final name is there, with a record that agrees."""
    rec, path = sent_record(rel), Path(rel)
    if not rec or bool(rec.get("tree")) != tree:
        return False
    if tree:
        return path.is_dir()
    return path.is_file() and path.stat().st_size == rec.get("size") and (size is None or size == rec.get("size"))


def sha256_file(path, block: int = HASH_BLOCK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(block)
            if not data:
                return h.hexdigest()
            h.update(data)


def manifest() -> list[list]:
    """The files of a folder being sent, [[path inside it, size, sha256 or ""]], from the sent config."""
    try:
        data = json.loads(base64.b64decode(SENT_CONFIG).decode("utf-8")) if SENT_CONFIG else None
        rows = [[str(r[0]), int(r[1]), str(r[2]) if len(r) > 2 and r[2] else ""] for r in data["files"]]
    except (ValueError, KeyError, TypeError, IndexError, UnicodeError) as e:
        raise Refused(f"The folder's file list is missing or unreadable: {type(e).__name__}") from e
    for row in rows:
        inner_parts(row[0])
    return rows


def cmd_file_state(args) -> int:
    """How much of a file is on this machine: {"path", "done", "have", "size"}. have = bytes of <path>.part (cut back
    to nothing when it is longer than --size). --tree: a folder, its file list in the sent config; done says whether it
    was finished and recorded, missing lists the files absent or of another size."""
    rel = args.path
    path = sent_path(rel)
    if args.tree:
        missing = []
        for name, size, _ in manifest():
            f = path.joinpath(*inner_parts(name))
            if not f.is_file() or f.stat().st_size != size:
                missing.append(name)
        show({"path": rel, "done": sent_complete(rel, tree=True) and not missing, "missing": missing})
        return 0
    if sent_complete(rel, args.size):
        size = path.stat().st_size
        show({"path": rel, "done": True, "have": size, "size": size})
        return 0
    part = Path(rel + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if args.size is not None and have > args.size:  # not a part of this file: start again
        part.unlink()
        have = 0
    show({"path": rel, "done": False, "have": have, "size": None})
    return 0


def cmd_file_done(args) -> int:
    """Close a file: check <path>.part has --size bytes and --sha256, rename it to <path>, record it. --tree: a folder;
    every file of its list (sent config) is checked (size, and sha256 where given and not recorded by its own file-done),
    the stage folder is removed, the folder is recorded. A wrong checksum removes the part (answer state "mismatch",
    exit 1); a part of the wrong length is refused and kept."""
    rel = args.path
    path = sent_path(rel)
    if args.tree:
        path.mkdir(parents=True, exist_ok=True)
        rows, total = manifest(), 0
        for name, size, sha in rows:
            f = path.joinpath(*inner_parts(name))
            if not f.is_file() or f.stat().st_size != size:
                raise Refused(f"{name} is missing or of another size: send the folder again.")
            record = sent_record(f"{rel}/{name}")
            if sha and not (record and record.get("sha256") == sha and record.get("size") == size):
                if sha256_file(f) != sha.lower():
                    show({"path": rel, "state": "mismatch", "file": name})
                    return 1
            total += size
        for folder, dirs, _ in os.walk(path):  # the stage folders of staged sends, at any depth
            for name in [d for d in dirs if d == ".rulz_incoming"]:
                shutil.rmtree(Path(folder) / name, ignore_errors=True)
                dirs.remove(name)
        write_sent_record(rel, total, args.sha256 or "", tree=True)
        show({"path": rel, "state": "done", "size": total, "files": len(rows)})
        return 0
    if args.size is None or not args.sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256):
        raise Refused("file-done needs --size and a 64 digit --sha256.")
    sha = args.sha256.lower()
    part = Path(rel + ".part")
    if sent_complete(rel, args.size) and (sent_record(rel) or {}).get("sha256") == sha:
        show({"path": rel, "state": "done", "size": args.size})
        return 0
    if path.is_file() and not part.exists():  # renamed by an earlier call whose answer was lost
        if path.stat().st_size == args.size and sha256_file(path) == sha:
            write_sent_record(rel, args.size, sha)
            show({"path": rel, "state": "done", "size": args.size})
            return 0
        path.unlink()
        show({"path": rel, "state": "mismatch"})
        return 1
    if not part.is_file():
        raise Refused(f"Nothing of {rel} has arrived.")
    have = part.stat().st_size
    if have != args.size:
        raise Refused(f"{have} of {args.size} bytes of {rel} are here.")
    if sha256_file(part) != sha:
        part.unlink()
        show({"path": rel, "state": "mismatch"})
        return 1
    os.replace(part, path)
    write_sent_record(rel, args.size, sha)
    show({"path": rel, "state": "done", "size": args.size})
    return 0


# ---------------------------------------------------------------- file-fetch: a signed address downloaded here

FETCH_PROGRAM = r'''#!/usr/bin/env python3
"""Downloads one file from a signed address, detached from the agent that started it (rulz_agent.py file-fetch).
The job (address, size, sha256, destination, state file) arrives as one JSON line on stdin: the address is never on a
command line or on disk. 16 connections fetch 16 MB ranges into a preallocated .fpart; a sidecar .fmap says which
ranges are whole, so a restart goes on. Every hop of a redirect must be an allowed host. A 401 or 403 ends it with the
state "expired": the PC asks for a fresh address and starts it again."""
import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

CHUNK = 16 * 1024 * 1024
CONNECTIONS = 16
TRIES = 6
ALLOW_LOOPBACK = False  # tests only
SLEEP = time.sleep
LOOPBACK = ("127.0.0.1", "localhost", "[::1]", "::1")


class Expired(Exception):
    pass


class Failed(Exception):
    pass


def host_allowed(url):
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if ALLOW_LOOPBACK and host in LOOPBACK and parts.scheme in ("http", "https"):
        return True
    if parts.scheme != "https":
        return False
    return host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def scrub(text):
    return re.sub(r"https?://\S+", "<address>", str(text))[:300]


def open_url(url, headers=None, timeout=60):
    """GET with redirects followed by hand, each hop host-checked. Returns (response, final address)."""
    for _ in range(8):
        if not host_allowed(url):
            raise Failed("The address leads to a host that is not allowed.")
        try:
            return OPENER.open(urllib.request.Request(url, headers=headers or {}), timeout=timeout), url
        except urllib.error.HTTPError as e:
            where = e.headers.get("Location") if e.headers else None
            code = e.code
            e.close()
            if code in (301, 302, 303, 307, 308) and where:
                url = urllib.parse.urljoin(url, where)
                continue
            if code in (401, 403):
                raise Expired("The address has expired.")
            raise Failed("HTTP error %s" % code)
    raise Failed("Too many redirects.")


class Progress:
    def __init__(self, job, size):
        self.job, self.size, self.done = job, size, 0
        self.lock, self.last, self.started = threading.Lock(), 0.0, time.time()

    def add(self, n):
        with self.lock:
            self.done += n

    def write(self, state="running", error=None, force=False):
        now = time.time()
        if not force and now - self.last < 1.0:
            return
        self.last = now
        data = {"dest": self.job["dest"], "state": state, "done": min(self.done, self.size), "size": self.size,
                "pid": os.getpid(), "updated": now, "error": error,
                "speed": round(self.done / max(1.0, now - self.started))}
        try:
            data["created"] = __import__("psutil").Process().create_time()
        except Exception:
            data["created"] = None
        tmp = self.job["state"] + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.job["state"])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(8 * 1024 * 1024)
            if not data:
                return h.hexdigest()
            h.update(data)


def load_map(mapf, size, count):
    head = ("RULZ1 %d %d\n" % (size, CHUNK)).encode("ascii")
    try:
        with open(mapf, "rb") as f:
            raw = f.read()
        if raw.startswith(head) and len(raw) == len(head) + count:
            return head, bytearray(raw[len(head):])
    except OSError:
        pass
    return head, bytearray(b"0" * count)


def ranged(job, url, part, mapf, size, progress):
    count = max(1, ceil_div(size, CHUNK))
    head, marks = load_map(mapf, size, count)
    if not os.path.isfile(part) or os.path.getsize(part) != size:
        marks = bytearray(b"0" * count)
        with open(part, "wb") as f:
            f.truncate(size)
    with open(mapf, "wb") as f:
        f.write(head + bytes(marks))
    todo = [i for i in range(count) if marks[i:i + 1] != b"1"]
    progress.add(sum(min(size, (i + 1) * CHUNK) - i * CHUNK for i in range(count) if marks[i:i + 1] == b"1"))
    fd = os.open(part, os.O_RDWR | getattr(os, "O_BINARY", 0))
    mfd = os.open(mapf, os.O_RDWR | getattr(os, "O_BINARY", 0))
    lock, stop, errors = threading.Lock(), threading.Event(), []

    def write_at(offset, data):
        if hasattr(os, "pwrite"):
            os.pwrite(fd, data, offset)
        else:
            with lock:
                os.lseek(fd, offset, os.SEEK_SET)
                os.write(fd, data)

    def worker():
        while not stop.is_set():
            with lock:
                if not todo:
                    return
                i = todo.pop(0)
            a, b = i * CHUNK, min(size, (i + 1) * CHUNK) - 1
            for attempt in range(TRIES):
                got = 0
                try:
                    resp, _ = open_url(url, {"Range": "bytes=%d-%d" % (a, b)})
                    with resp:
                        if getattr(resp, "status", 200) != 206:
                            raise Failed("The server ignored the range.")
                        got_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", resp.headers.get("Content-Range") or "")
                        if not got_range or tuple(int(x) for x in got_range.groups()) != (a, b, size):
                            raise Failed("The server answered another range than asked.")
                        while not stop.is_set():
                            data = resp.read(1024 * 1024)
                            if not data:
                                break
                            write_at(a + got, data)
                            got += len(data)
                            progress.add(len(data))
                            progress.write()
                    if stop.is_set():
                        return
                    if got != b - a + 1:
                        raise Failed("Short range.")
                    with lock:
                        os.lseek(mfd, len(head) + i, os.SEEK_SET)
                        os.write(mfd, b"1")
                    break
                except Expired as e:
                    errors.append(e)
                    stop.set()
                    return
                except Exception as e:
                    progress.add(-got)
                    if attempt == TRIES - 1:
                        errors.append(e)
                        stop.set()
                        return
                    SLEEP(min(30, 2 ** attempt))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(CONNECTIONS, max(1, len(todo))))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    os.close(fd)
    os.close(mfd)
    if errors:
        raise errors[0]


def ceil_div(a, b):
    return -(-a // b)


def single(url, part, size, progress):
    resp, _ = open_url(url)
    with resp, open(part, "wb") as f:
        while True:
            data = resp.read(1024 * 1024)
            if not data:
                break
            f.write(data)
            progress.add(len(data))
            progress.write()


def run(job):
    """Download job["dest"]; returns the final state name. The state file says how it went."""
    size, dest = int(job["size"]), job["dest"]
    part, mapf = dest + ".fpart", dest + ".fmap"
    progress = Progress(job, size)
    try:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        progress.write("running", force=True)
        url = job["url"]
        if size == 0:
            open(part, "wb").close()
        else:
            resp, final = open_url(url, {"Range": "bytes=0-0"})
            with resp:
                status = getattr(resp, "status", 200)
                total = (resp.headers.get("Content-Range") or "").rpartition("/")[2]
                length = resp.headers.get("Content-Length")
            if status == 206 and total.isdigit():
                if int(total) != size:
                    raise Failed("The file there is %s bytes, not %d." % (total, size))
                ranged(job, final, part, mapf, size, progress)
            else:
                if length and length.isdigit() and int(length) != size:
                    raise Failed("The file there is %s bytes, not %d." % (length, size))
                single(final, part, size, progress)  # the server ignores ranges: one stream, from the start
        if os.path.getsize(part) != size:
            raise Failed("The download is %d bytes, not %d." % (os.path.getsize(part), size))
        progress.write("running", force=True)
        if sha256_file(part) != job["sha256"].lower():
            for p in (part, mapf):
                if os.path.exists(p):
                    os.remove(p)
            raise Failed("The checksum differs.")
        os.replace(part, dest)
        if os.path.exists(mapf):
            os.remove(mapf)
        rec = job["record"]
        os.makedirs(os.path.dirname(rec), exist_ok=True)
        with open(rec + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"path": job["rel"], "size": size, "sha256": job["sha256"].lower(), "tree": False,
                       "at": round(time.time())}, f)
        os.replace(rec + ".tmp", rec)
        progress.done = size
        progress.write("done", force=True)
        return "done"
    except Expired:
        progress.write("expired", "The address has expired.", force=True)
        return "expired"
    except Exception as e:
        progress.write("failed", scrub(e) or type(e).__name__, force=True)
        return "failed"


def main():
    for name in ("SIGHUP", "SIGPIPE"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_IGN)
    job = json.loads(sys.stdin.readline())
    return 0 if run(job) == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def embedded(program: str, name: str) -> types.ModuleType:
    """An embedded program's functions, loaded here (the agent is piped, so its programs are not files)."""
    module = types.ModuleType(name)
    exec(compile(program, f"<{name}>", "exec"), module.__dict__)  # noqa: S102  (our own constant)
    return module


def pid_running(pid, created=None) -> bool:
    try:
        return live_process(pid, created) is not None
    except ImportError:  # no psutil: the pid alone
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, TypeError, ValueError):
            return False


def fetch_paths(rel: str) -> tuple[Path, Path]:
    key = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
    return FETCH_DIR / f"{key}.json", FETCH_DIR / f"{key}.lock"


def fetch_status(rel: str, size=None) -> dict:
    """{"dest", "state": none | starting | running | done | failed | expired | interrupted, "done", "size", "error"}."""
    rec = sent_record(rel)
    if rec and not rec.get("tree") and Path(rel).is_file() and Path(rel).stat().st_size == rec.get("size"):
        return {"dest": rel, "state": "done", "done": rec["size"], "size": rec["size"]}
    state_path, _ = fetch_paths(rel)
    data = read_json(state_path)
    if not isinstance(data, dict):
        return {"dest": rel, "state": "none", "done": 0, "size": size}
    state = data.get("state")
    if state in ("starting", "running"):
        alive = data.get("pid") is not None and pid_running(data["pid"], data.get("created"))
        if not alive and (data.get("pid") is not None or time.time() - (data.get("updated") or 0) > 30):
            state = "interrupted"
    out = {"dest": rel, "state": state, "done": data.get("done") or 0, "size": data.get("size") or size}
    for key in ("error", "speed"):
        if data.get(key) is not None:
            out[key] = data[key]
    return out


def cmd_file_fetch(args) -> int:
    """Start (or find already going) a detached download of a signed address into models/sent. The address, size and
    checksum come in the sent config (stdin of this program); the child gets them through a pipe. Answers with
    fetch-state's shape; "done" at once when the file is already here."""
    rel = args.path
    sent_path(rel)
    try:
        job = json.loads(base64.b64decode(SENT_CONFIG).decode("utf-8")) if SENT_CONFIG else None
        url, size, sha = str(job["url"]), int(job["size"]), str(job["sha256"]).lower()
    except (ValueError, KeyError, TypeError, UnicodeError) as e:
        raise Refused(f"file-fetch needs {{url, size, sha256}} in the sent config ({type(e).__name__}).") from e
    if size < 0 or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise Refused("file-fetch needs a size and a 64 digit sha256.")
    fetcher = embedded(FETCH_PROGRAM, "rulz_fetch")
    if not fetcher.host_allowed(url):
        raise Refused("That address is not on an allowed host (huggingface.co, *.huggingface.co, *.hf.co).")
    if sent_complete(rel, size):
        show(fetch_status(rel, size))
        return 0
    state_path, lock_path = fetch_paths(rel)
    lock = hold_lock(lock_path)
    if lock is None:
        show(fetch_status(rel, size))
        return 0
    with lock:
        now = fetch_status(rel, size)
        if now["state"] in ("starting", "running"):
            show(now)
            return 0
        write_json(state_path, {"dest": rel, "state": "starting", "done": 0, "size": size, "pid": None,
                                "updated": time.time()})
        child = {"url": url, "size": size, "sha256": sha, "dest": rel, "rel": rel, "state": str(state_path),
                 "record": str(sent_record_path(rel))}
        flags = {"start_new_session": True} if sys.platform != "win32" else {
            "creationflags": getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        proc = subprocess.Popen([sys.executable, "-c", FETCH_PROGRAM], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, close_fds=True, **flags)
        proc.stdin.write((json.dumps(child) + "\n").encode("utf-8"))
        proc.stdin.close()
    time.sleep(0.5)
    show(fetch_status(rel, size))
    return 0


def cmd_fetch_state(args) -> int:
    sent_path(args.path)
    show(fetch_status(args.path))
    return 0


# ---------------------------------------------------------------- watchdog

WATCH_PROGRAM = r'''#!/usr/bin/env python3
"""RulZ AI Trainer's idle watchdog for a rented vast.ai instance. Written by rulz_agent.py (watchdog arm); do not edit.
It only ever stops its own instance: after N idle minutes it asks vast.ai with the container's own API key. Idle means
no run, no waiting queue entry, no LoRA test and nothing arriving (a growing file below models/sent or
datasets/.rulz_incoming, a running file-fetch, provisioning). Before any run was seen it also fires, unless the PC's
manager touched output/_watchdog/alive in the last 10 minutes. The program reads nothing of the trainer's code."""
import contextlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

POLL = 30.0
RETRY_FIRED = 600.0  # a stop vast.ai accepted but that did not happen is asked again this often
RETRY_AFTER = 300.0  # a stop that failed is asked again this often while the instance is still idle
ALIVE_FRESH = 600.0
ARRIVE_WINDOW = 180.0
REBOOT_SLACK = 5.0
TESTS_FINAL = ("refused", "done", "stopped", "failed")
VAST_API = "https://console.vast.ai/api/v0"
ENV_FILE = Path("/etc/environment")
FIRST_PROCESS_ENV = Path("/proc/1/environ")
OWNED = ("pid", "created", "idle_since", "seen_running", "fired", "fired_at", "fired_boot", "booted", "rebooted",
         "spent", "last", "next_try", "seen_at")  # what this program writes; the rest is the agent's


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("%s.%d.%d.tmp" % (path.name, os.getpid(), threading.get_ident()))
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


@contextlib.contextmanager
def locked(folder):
    """An exclusive OS lock on <folder>/lock (where fcntl exists) around a read-modify-write of state.json: the agent
    (arm, disarm) and this program both take it."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(Path(folder) / "lock", "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def booted():
    """When this container started, as the agent's booted(): its first process's start time."""
    try:
        import psutil

        return round(psutil.Process(1).create_time(), 1)
    except Exception:
        pass
    try:
        fields = Path("/proc/1/stat").read_text().rsplit(")", 1)[1].split()
        start = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime"):
                return round(int(line.split()[1]) + start, 1)
    except Exception:
        pass
    return None


def pid_alive(pid, created=None):
    try:
        import psutil
    except ImportError:
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, TypeError, ValueError):
            return False
    try:
        proc = psutil.Process(int(pid))
        if created is not None and abs(proc.create_time() - float(created)) > 1.0:
            return False
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (TypeError, ValueError, psutil.Error):
        return False


def newest_mtime(folder, fresh_after=None):
    """The newest change time (the later of mtime and ctime: tar restores old mtimes) of anything below folder,
    directories included. The whole folder is scanned, or until something newer than fresh_after is found."""
    newest = None
    for dirpath, dirs, names in os.walk(folder):
        for name in list(dirs) + list(names) + [""]:
            try:
                st = os.stat(os.path.join(dirpath, name))
            except OSError:
                continue
            m = max(st.st_mtime, st.st_ctime)
            newest = m if newest is None else max(newest, m)
            if fresh_after is not None and m >= fresh_after:
                return newest
    return newest


def tests_running(tests):
    try:
        folders = [d for d in Path(tests).iterdir() if d.is_dir() and not d.is_symlink()]
    except OSError:
        return False
    for d in folders:
        status = read_json(d / "status.json", {}) or {}
        if status.get("state") in TESTS_FINAL:
            continue
        pid = status.get("pid")
        try:
            young = time.time() - d.stat().st_mtime <= 120
        except OSError:
            continue
        if (pid is None and young) or (pid is not None and pid_alive(pid, status.get("pid_created"))):
            return True
    return False


def fetching(folder, now):
    try:
        files = list(Path(folder).glob("*.json"))
    except OSError:
        return False
    for f in files:
        data = read_json(f, {}) or {}
        if data.get("state") in ("starting", "running"):
            if data.get("pid") is not None and pid_alive(data["pid"], data.get("created")):
                return True
            if data.get("pid") is None and now - (data.get("updated") or 0) < 60:
                return True
    return False


VIEWERS = ("tail", "less", "more", "cat", "grep", "egrep", "vi", "vim", "nano", "head", "watch", "journalctl")
PROVISION_WORDS = ("provision.sh", "provisioner", "install.sh", "download_models.sh")
PROVISION_MAX_AGE = 7200.0  # a process older than this is not a provisioning that is still going


def provisioning_process(cmdline, name, age):
    """Is this process a setup step still going: younger than 2 hours, not a log viewer (tail -F provisioning.log runs
    for ever), and naming provision.sh, provisioner, install.sh, download_models.sh or `pip install` as a word of its
    arguments?"""
    args = [str(a) for a in (cmdline or [])]
    if not args or age >= PROVISION_MAX_AGE:
        return False
    if os.path.basename(str(name or args[0])) in VIEWERS or os.path.basename(args[0]) in VIEWERS:
        return False
    words = [os.path.basename(a) for a in args[1:]]
    if any(w in PROVISION_WORDS for w in words):
        return True
    pip = any(w in ("pip", "pip3") for w in [os.path.basename(args[0])] + words) or ("-m" in args and "pip" in args)
    return "install" in args[1:] and pip


def provisioning(root, now):
    models = Path(root) / "models"
    log = models / "bucket_download.log"
    try:
        if log.is_file() and not (models / "bucket" / ".download_complete").is_file() \
                and now - log.stat().st_mtime < ARRIVE_WINDOW:
            return True
    except OSError:
        pass
    try:
        import psutil

        for proc in psutil.process_iter(["cmdline", "name", "create_time"]):
            info = proc.info
            if provisioning_process(info.get("cmdline"), info.get("name"), now - (info.get("create_time") or now)):
                return True
    except Exception:
        pass
    return False


def observe(root, now=None):
    """What the machine shows now: the inputs of decide()."""
    now = time.time() if now is None else now
    root = Path(root)
    out = root / "output"
    job = read_json(root / ".webui_job.json", {}) or {}
    running = job.get("pid") is not None and pid_alive(job["pid"], job.get("create_time"))
    queue = read_json(out / "_queue.json", {}) or {}
    arriving = False
    for folder in (root / "models" / "sent", root / "datasets" / ".rulz_incoming"):
        newest = newest_mtime(folder, now - ARRIVE_WINDOW) if folder.is_dir() else None
        if newest is not None and now - newest < ARRIVE_WINDOW:
            arriving = True
    arriving = arriving or fetching(out / "_fetch", now) or provisioning(root, now)
    try:
        alive = (out / "_watchdog" / "alive").stat().st_mtime
    except OSError:
        alive = None
    return {"running": bool(running), "waiting": len(queue.get("waiting") or []), "paused": bool(queue.get("paused")),
            "tests": tests_running(out / "_tests"), "arriving": bool(arriving), "alive": alive, "booted": booted()}


def decide(now, st, obs):
    """(new state, action): action is "stop" or None. Pure: state and observations in, state out."""
    st = dict(st)
    limit = max(1.0, float(st.get("minutes") or 60)) * 60
    boot, was = obs.get("booted"), st.get("booted")
    if boot is not None:
        if was is not None and abs(boot - was) > REBOOT_SLACK:  # the container restarted: a run that was going ended
            st.update(rebooted=True, idle_since=None)
        if was is None or abs(boot - was) > REBOOT_SLACK:
            st["booted"] = boot
    if st.get("fired"):
        fired_boot = st.get("fired_boot")
        if boot is not None and fired_boot is not None and abs(boot - fired_boot) > REBOOT_SLACK:
            st["spent"] = True  # a restart after the stop: nothing more until re-armed
            return st, None
        if st.get("spent") or st.get("fired_at") is None:
            return st, None
        # vast.ai took the request but the instance did not stop: ask again every RETRY_FIRED while still idle
        waits = obs.get("waiting", 0) > 0 and not (obs.get("paused") and st.get("rebooted"))
        if obs.get("running") or obs.get("tests") or obs.get("arriving") or waits:
            st["fired_at"] = now
            return st, None
        return (st, "stop") if now - st["fired_at"] >= RETRY_FIRED else (st, None)
    if obs.get("running"):
        st.update(seen_running=True, rebooted=False, idle_since=None, next_try=None)
        return st, None
    waits = obs.get("waiting", 0) > 0 and not (obs.get("paused") and st.get("rebooted"))
    busy = bool(obs.get("tests") or obs.get("arriving") or waits)
    if not st.get("seen_running") and not busy and obs.get("alive") is not None and now - obs["alive"] < ALIVE_FRESH:
        busy = True  # a manager that looks in is still working on it
    if busy:
        st.update(idle_since=None, next_try=None)
        return st, None
    if st.get("idle_since") is None:
        st["idle_since"] = now
        return st, None
    if now - st["idle_since"] >= limit and (st.get("next_try") is None or now >= st["next_try"]):
        return st, "stop"
    return st, None


def container_env():
    """The container's variables: /etc/environment, the first process's environment, then this process's own."""
    values = {}
    try:
        for line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
            name, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and name and not name.startswith("#"):
                values[name.strip()] = value.strip().strip("'\"")
    except OSError:
        pass
    try:
        raw = FIRST_PROCESS_ENV.read_bytes()
    except OSError:
        raw = b""
    for item in raw.split(b"\0"):
        name, sep, value = item.decode("utf-8", "replace").partition("=")
        if sep and name:
            values[name] = value
    values.update(os.environ)
    return values


def stop_instance(env=None, opener=urllib.request.urlopen):
    """(ok, message): PUT {"state": "stopped"} to this instance with CONTAINER_API_KEY. The key goes in a header only
    and no message carries it or an address."""
    env = container_env() if env is None else env
    label = env.get("VAST_CONTAINERLABEL") or ""
    ident = str(env.get("CONTAINER_ID") or label.removeprefix("C."))
    key = env.get("CONTAINER_API_KEY")
    if not ident.isdigit() or not key:
        return False, "No CONTAINER_ID or CONTAINER_API_KEY here."
    request = urllib.request.Request(
        "%s/instances/%s/" % (VAST_API, ident), data=json.dumps({"state": "stopped"}).encode("utf-8"), method="PUT",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with opener(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
        try:
            answer = json.loads(body)
        except ValueError:
            answer = {}
        if isinstance(answer, dict) and answer.get("success") is False:
            return False, "vast.ai refused the stop."
        return True, "Stop sent to vast.ai."
    except urllib.error.HTTPError as e:
        return False, "vast.ai answered HTTP %s." % e.code
    except Exception as e:
        return False, "The stop request failed (%s)." % type(e).__name__


def merge_owned(path, new):
    cur = read_json(path, {}) or {}
    for k in OWNED:
        if k in new:
            cur[k] = new[k]
    write_json(path, cur)


def save_owned(path, new):
    with locked(Path(path).parent):
        merge_owned(path, new)


def step(root, now, obs, stop=stop_instance):
    """One look. Marks "fired" before the stop is sent; a failed request un-fires and tries again in RETRY_AFTER."""
    path = Path(root) / "output" / "_watchdog" / "state.json"
    with locked(path.parent):  # a disarm wins: armed is read again here, under the lock, before anything is marked
        st = read_json(path, {}) or {}
        if not st.get("armed"):
            return "disarmed"
        new, action = decide(now, st, obs)
        if action != "stop":
            merge_owned(path, new)
            return None
        new.update(fired=True, fired_at=now, fired_boot=obs.get("booted"), next_try=None)
        merge_owned(path, new)
    ok, message = stop()
    new["last"] = message
    if not ok:
        new.update(fired=False, fired_boot=None, next_try=now + RETRY_AFTER)
    save_owned(path, new)
    return "stopped" if ok else "failed"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(argv[argv.index("--root") + 1]) if "--root" in argv else Path.cwd()
    supervised = "--supervised" in argv
    os.chdir(root)
    path = root / "output" / "_watchdog" / "state.json"
    created = None
    try:
        import psutil

        created = psutil.Process().create_time()
    except Exception:
        pass
    save_owned(path, {"pid": os.getpid(), "created": created, "idle_since": None, "seen_at": time.time()})
    while True:
        st = read_json(path, {}) or {}
        if not st.get("armed"):
            if not supervised:
                return 0
        else:
            try:
                step(root, time.time(), observe(root))
            except Exception as e:  # a bad look must not end the watch
                save_owned(path, {"last": "Look failed (%s)." % type(e).__name__})
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
'''

URLOPEN = urllib.request.urlopen  # tests replace it


def own_instance(env: dict) -> tuple[str, str]:
    label = env.get("VAST_CONTAINERLABEL") or ""
    ident = str(env.get("CONTAINER_ID") or label.removeprefix("C."))
    return (ident if ident.isdigit() else ""), env.get("CONTAINER_API_KEY") or ""


def key_check(env: dict | None = None, opener=None) -> tuple[bool, str]:
    """Read this instance with its own API key: (can_stop, reason). The key only goes in a header; the reason never
    carries it or an address."""
    env = container_env() if env is None else env
    ident, key = own_instance(env)
    if not ident:
        return False, "This is not a vast.ai instance (no CONTAINER_ID)."
    if not key:
        return False, "This instance has no CONTAINER_API_KEY."
    request = urllib.request.Request(f"{VAST_API}/instances/{ident}/", method="GET",
                                     headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with (opener or URLOPEN)(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        return False, f"vast.ai answered HTTP {e.code} to the key."
    except Exception as e:  # noqa: BLE001
        return False, f"vast.ai could not be asked ({type(e).__name__})."
    if isinstance(data, dict) and data.get("success") is False:
        return False, "vast.ai refused the key."
    return True, ""


_WATCH_MODULE = None


def embedded_watch() -> types.ModuleType:
    global _WATCH_MODULE
    if _WATCH_MODULE is None:
        _WATCH_MODULE = embedded(WATCH_PROGRAM, "rulz_watch")
    return _WATCH_MODULE


def watch_process_alive(st: dict) -> bool:
    return st.get("pid") is not None and pid_running(st["pid"], st.get("created"))


def autostop_block() -> dict:
    data = read_json(AUTOSTOP_FILE)
    return {"armed": bool(isinstance(data, dict) and data.get("armed"))}


def watchdog_block(now: float | None = None) -> dict:
    """The snapshot's watchdog: {armed (set and its process alive, not fired), idle_since, minutes, fired, ...}."""
    now = time.time() if now is None else now
    st = read_json(WATCH_STATE)
    st = st if isinstance(st, dict) else {}
    alive = watch_process_alive(st)
    try:
        age = round(now - WATCH_ALIVE.stat().st_mtime, 1)
    except OSError:
        age = None
    block = {"armed": bool(st.get("armed")) and alive and not st.get("fired"), "idle_since": st.get("idle_since"),
             "minutes": st.get("minutes"), "fired": bool(st.get("fired")), "alive": alive,
             "supervised": bool(st.get("supervised")), "seen_running": bool(st.get("seen_running")),
             "rebooted": bool(st.get("rebooted")), "can_stop": st.get("can_stop"), "last": st.get("last"),
             "alive_age": age, "autostop_armed": autostop_block()["armed"]}
    if st.get("idle_since") is not None:
        block["idle_seconds"] = round(max(0.0, now - st["idle_since"]), 1)
    return block


def watch_conf_text(python: str, root: Path) -> str:
    return (f"[program:{WATCH_PROGRAM_NAME}]\n"
            f"command={python} {(root / WATCH_FILE).as_posix()} --root {root.as_posix()} --supervised\n"
            f"directory={root.as_posix()}\nautostart=true\nautorestart=true\nstartsecs=3\n"
            "stdout_logfile=/dev/null\nredirect_stderr=true\n")


def start_watch_process(root: Path) -> None:
    """The detached program, for an instance without supervisor."""
    flags = {"start_new_session": True} if sys.platform != "win32" else {
        "creationflags": getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    subprocess.Popen([sys.executable, str((root / WATCH_FILE).resolve()), "--root", str(root.resolve())],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(root),
                     close_fds=True, **flags)


def ctl_run(ctl: list[str], *words: str) -> bool:
    try:
        return subprocess.run([*ctl, *words], capture_output=True, stdin=subprocess.DEVNULL, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def update_watch_state(**fields) -> dict:
    """state.json changed under the lock the watchdog takes too (output/_watchdog/lock)."""
    with embedded_watch().locked(WATCH_DIR):
        st = read_json(WATCH_STATE)
        st = st if isinstance(st, dict) else {}
        st.update(fields)
        write_json(WATCH_STATE, st)
    return st


def watchdog_arm(minutes: int) -> dict:
    if not 10 <= minutes <= 1440:
        raise Refused("--idle-minutes must be between 10 and 1440.")
    ok, reason = key_check()
    if not ok:  # nothing to arm that could stop anything; a watchdog already there is left as it is
        return dict(watchdog_block(), can_stop=False, reason=reason)
    root = Path.cwd()
    text = WATCH_PROGRAM
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    changed = read_text(WATCH_FILE) != text
    if changed:
        tmp = WATCH_FILE.with_name(WATCH_FILE.name + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(WATCH_FILE)
    ctl = supervisorctl()
    update_watch_state(armed=True, minutes=minutes, armed_at=round(time.time(), 1), fired=False, fired_at=None,
                       fired_boot=None, spent=False, idle_since=None, seen_running=False, rebooted=False,
                       next_try=None, booted=booted(), can_stop=True, supervised=ctl is not None, last="")
    alive = watch_process_alive(read_json(WATCH_STATE) or {})
    if ctl is not None:
        conf = watch_conf_text(sys.executable, root)
        conf_changed = read_text(WATCH_CONF) != conf
        if conf_changed:
            WATCH_CONF.write_text(conf, encoding="utf-8", newline="\n")
        if conf_changed or not alive:
            ctl_run(ctl, "reread")
            ctl_run(ctl, "update")
        if alive and (changed or conf_changed):
            ctl_run(ctl, "restart", WATCH_PROGRAM_NAME)
        elif not alive:
            ctl_run(ctl, "start", WATCH_PROGRAM_NAME)
    elif not alive:
        start_watch_process(root)
    deadline = time.time() + WATCH_START_WAIT
    while time.time() < deadline and not watch_process_alive(read_json(WATCH_STATE) or {}):
        time.sleep(0.3)
    if ctl is not None and not watch_process_alive(read_json(WATCH_STATE) or {}):
        update_watch_state(supervised=False)  # supervisor did not start it: the detached one
        start_watch_process(root)
        time.sleep(1.0)
    return dict(watchdog_block(), can_stop=True)


def watchdog_disarm() -> dict:
    st = update_watch_state(armed=False, idle_since=None)
    ctl = supervisorctl()
    if ctl is not None and WATCH_CONF.exists():
        ctl_run(ctl, "stop", WATCH_PROGRAM_NAME)
        try:
            WATCH_CONF.unlink()
        except OSError:
            pass
        ctl_run(ctl, "reread")
        ctl_run(ctl, "update")
    if st.get("pid") is not None:
        try:
            proc = live_process(st["pid"], st.get("created"))
            if proc is not None:
                proc.terminate()
        except ImportError:
            pass
    return watchdog_block()


def cmd_watchdog(args) -> int:
    """arm --idle-minutes N | disarm | status | alive. Answers with the snapshot's watchdog block (arm adds can_stop and,
    when the key can't stop this instance, reason)."""
    if args.action == "arm":
        show(watchdog_arm(args.idle_minutes))
    elif args.action == "disarm":
        show(watchdog_disarm())
    elif args.action == "alive":
        WATCH_DIR.mkdir(parents=True, exist_ok=True)
        WATCH_ALIVE.touch()
        os.utime(WATCH_ALIVE, None)
        show({"alive": True, "time": time.time()})
    else:
        show(watchdog_block())
    return 0


def tests_running_count() -> int:
    try:
        return len(running_tests())
    except (Refused, OSError):
        return 0


def tree_listing(d: Path) -> list[dict]:
    """Every file below a job folder except state_* folders, remote_render/, hidden files and folders, *.part files."""
    out = []
    for folder, dirs, names in os.walk(d):
        top = Path(folder) == d
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and not x.startswith("state_")
                         and not (top and x == "remote_render"))
        for name in sorted(names):
            p = Path(folder) / name
            if name.startswith(".") or name.endswith(".part") or p.is_symlink():
                continue
            st = p.stat()
            out.append({"path": p.relative_to(d.parent).as_posix(), "size": st.st_size, "mtime": int(st.st_mtime)})
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rulz_agent", description="RulZ AI Trainer's runs on this machine")
    sub = ap.add_subparsers(dest="command", required=True)

    def command(name, func, job_name=False):
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        if job_name:
            p.add_argument("name", nargs="?", help="job name; default: the running or last job")
        return p

    command("status", cmd_status)
    p = command("jobs", cmd_jobs)
    p.add_argument("--all", action="store_true", help="every folder of output/ that can be a job, with its state")
    p = command("job", cmd_job, True)
    p.add_argument("--full", action="store_true", help="status.json with its per-step series")
    p.add_argument("--files", type=int, default=60, help="how many of the folder's file names to list")
    p = command("log", cmd_log, True)
    p.add_argument("-n", type=int, default=80, help="last n lines; 0 = all")
    p.add_argument("--console", action="store_true", help="console.log (the job's stdout and tracebacks)")
    p.add_argument("--grep", help="only the lines matching this regular expression")
    p = command("probes", cmd_probes, True)
    p.add_argument("--last", type=int, default=0, help="only the last n probes")
    p.add_argument("--full", action="store_true", help="every field, per-seed scores and image paths included")
    p = command("config", cmd_config, True)
    p.add_argument("--diff", metavar="OTHER_JOB", help="only the settings that differ, as {key: [this, other]}")
    p.add_argument("--json", action="store_true", help="the settings as JSON")
    p = command("snapshot", cmd_snapshot)
    p.add_argument("--jobs", type=int, default=100, help="how many jobs to list, newest first")
    command("record", cmd_record, True)
    command("poll", cmd_poll, True)
    p = command("files", cmd_files, True)
    p.add_argument("--tests", metavar="TEST_ID")
    p.add_argument("--what", action="append", choices=("logs", "probes", "samples", "maps", "loras", "best", "all", "tree"))
    p.add_argument("--step", type=int, help="only this step's images and LoRA")
    command("test-start", cmd_test_start)
    p = command("test-status", cmd_test_status)
    p.add_argument("test_id", nargs="?")
    p = command("test-stop", cmd_test_stop)
    p.add_argument("test_id")
    p = command("start", cmd_start)
    p.add_argument("--job", required=True, help="the new run's job name")
    p.add_argument("--from", dest="base", metavar="JOB", help="start from this job's config")
    p.add_argument("--preset", default="")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--race-rows", default="", metavar="NAME[,NAME]",
                   help="of a race config, keep only these configurations (one alone: a plain run with its settings)")
    p.add_argument("--overwrite", action="store_true",
                   help="start afresh under a name whose folder holds an earlier run (that folder is moved aside)")
    p.add_argument("--dry-run", action="store_true", help="check everything, show the plan, queue nothing")
    p.add_argument("--resume-queue", action="store_true", help="also resume the queue when it is paused")
    p = command("stop", cmd_stop)
    p.add_argument("--job", default="", help="only when this is the running job (nothing is touched otherwise)")
    p.add_argument("--kill", action="store_true", help="kill the process: no final save, a waiting queue pauses")
    p = command("queue", cmd_queue)
    p.add_argument("action", nargs="?", choices=("show", "pause", "resume", "remove"), default="show")
    p.add_argument("id", nargs="?", help="remove: the waiting run's id")
    p = command("update", cmd_update)
    p.add_argument("--check", action="store_true", help="fetch and say what would change, change nothing")
    p.add_argument("--branch", default="", help="default: the checkout's branch, else RULZ_BRANCH or app/rulz-trainer")
    p.add_argument("--discard", action="store_true", help="replace files changed here (by rulz_remote.py sync)")
    p.add_argument("--force", action="store_true", help="even while a run goes")
    p.add_argument("--reinstall", action="store_true", help="run install.sh even when the requirements are the same")
    command("webui", cmd_webui)
    p = command("file-state", cmd_file_state)
    p.add_argument("path", help="posix path below models/sent/<12 hex>/")
    p.add_argument("--size", type=int, help="the file's final size: a longer .part is cut back to nothing")
    p.add_argument("--tree", action="store_true", help="a folder; its file list comes in the sent config")
    p = command("file-done", cmd_file_done)
    p.add_argument("path")
    p.add_argument("--size", type=int)
    p.add_argument("--sha256")
    p.add_argument("--tree", action="store_true", help="a folder; its file list comes in the sent config")
    p = command("file-fetch", cmd_file_fetch)
    p.add_argument("path", help="where the file goes; {url, size, sha256} come in the sent config")
    p = command("fetch-state", cmd_fetch_state)
    p.add_argument("path")
    p = command("watchdog", cmd_watchdog)
    p.add_argument("action", choices=("arm", "disarm", "status", "alive"))
    p.add_argument("--idle-minutes", type=int, default=60, help="arm: idle minutes before the stop (10..1440)")
    return ap


def main(argv=None) -> int:
    sys.path.insert(0, os.getcwd())  # piped over ssh there is no script folder: the repo's modules are here
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except Refused as e:
        print(f"[refused] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
