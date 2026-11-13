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
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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
    tmp = path.with_name(path.name + ".tmp")
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


def cmd_jobs(args) -> int:
    active = running_dir()
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
    d = job_dir(args.name)
    tag = f"{args.step:06d}" if args.step is not None else None
    what = set(args.what or ["logs"])
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


def cmd_start(args) -> int:
    """Queue a run. The checks of the WebUI's queue_run, the trainer's own validate, and refusing to overwrite an
    earlier run's folder."""
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
    if cfg.job_name in taken:
        raise Refused(f"Job name {cfg.job_name!r} is already running or queued.")
    if cfg.job_dir().exists() and not args.overwrite:
        raise Refused(f"{cfg.job_dir()} already exists: pick a new job name, or --overwrite to replace that run.")
    missing = [sp.path or "(none picked)" for sp in cfg.dataset_specs() if not sp.path or not Path(sp.path).is_dir()]
    if missing:
        raise Refused(f"Dataset folder not found: {', '.join(missing)}")
    plan = {"job": cfg.job_name, "arch": cfg.arch_name(), "steps": cfg.steps,
            "datasets": [sp.path for sp in cfg.dataset_specs()],
            "changed": {k: [before.get(k), v] for k, v in cfg.to_dict().items()
                        if before.get(k) != v and k != "job_name"}}
    if args.dry_run:
        show(dict(plan, dry_run=True, job_running=active is not None, queue=queue_view()))
        return 0
    if args.resume_queue:
        q.set_paused(False)
    return report_start(q, q.add(cfg.to_dict(), label=cfg.arch_name()), plan, active is not None)


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


def cmd_stop(args) -> int:
    proc, current = job_process(), current_job_dir()
    if proc is None or current is None:
        raise Refused("No job running.")
    if not args.kill:
        (current / "STOP").touch()
        show({"job": current.name, "state": "stopping",
              "note": "The trainer saves the LoRA after the current step, then ends."})
        return 0
    proc.kill()
    q = queue()
    if q.snapshot()["waiting"]:  # as the WebUI's Kill: stopping everything, so the queued runs wait for a resume
        q.set_paused(True)
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
        show(dict(plan, state="current", note=f"Already the newest RulzTrainer "
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


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rulz_agent", description="RulzTrainer's runs on this machine")
    sub = ap.add_subparsers(dest="command", required=True)

    def command(name, func, job_name=False):
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        if job_name:
            p.add_argument("name", nargs="?", help="job name; default: the running or last job")
        return p

    command("status", cmd_status)
    command("jobs", cmd_jobs)
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
    p.add_argument("--what", action="append", choices=("logs", "probes", "samples", "maps", "loras", "best", "all"))
    p.add_argument("--step", type=int, help="only this step's images and LoRA")
    p = command("start", cmd_start)
    p.add_argument("--job", required=True, help="the new run's job name")
    p.add_argument("--from", dest="base", metavar="JOB", help="start from this job's config")
    p.add_argument("--preset", default="")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--race-rows", default="", metavar="NAME[,NAME]",
                   help="of a race config, keep only these configurations (one alone: a plain run with its settings)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing job folder of that name")
    p.add_argument("--dry-run", action="store_true", help="check everything, show the plan, queue nothing")
    p.add_argument("--resume-queue", action="store_true", help="also resume the queue when it is paused")
    p = command("stop", cmd_stop)
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
