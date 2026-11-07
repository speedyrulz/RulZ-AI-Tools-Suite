#!/usr/bin/env python3
"""The instance side of rulz_remote.py: reads a RulzTrainer install's runs and queues new ones, as JSON or plain text.

rulz_remote.py pipes this file over ssh for every call (`cd /workspace/RulzTrainer && python - status`), so it works
on an instance that is already running, whatever its checkout's age, and it never restarts or changes the WebUI. It
needs the standard library, plus psutil, yaml and the repo's own modules where noted; keep it ASCII.

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
PROBE_COLUMNS = ("step", "face", "body", "combined", "face_ema", "body_ema", "combined_ema", "face_norm_ema",
                 "body_norm_ema", "best", "best_step", "since_improve", "lr_factor", "stage", "gold_ema", "copy")


class Refused(Exception):
    """A request that can't be carried out; its message is the answer."""


def show(data) -> None:
    print(json.dumps(data, indent=1, default=str))


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing, or torn mid-write
        return default


def job_process():
    """The psutil.Process of the job in .webui_job.json while it runs, else None (the WebUI's reattach_job: a pid only
    counts with its create time, pids get reused)."""
    import psutil

    data = read_json(JOB_FILE) or {}
    try:
        proc = psutil.Process(int(data["pid"]))
        created = data.get("create_time")
        if created is not None and abs(proc.create_time() - float(created)) > 1.0:
            return None
        return proc if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE else None
    except (KeyError, TypeError, ValueError, psutil.Error):
        return None


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


def cmd_status(args) -> int:
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
    data = {"root": str(Path.cwd()), "webui_running": webui_running(), "gpus": gpus,
            "disk_free_gb": round(disk.free / 1e9, 1), "disk_total_gb": round(disk.total / 1e9, 1),
            "datasets": sorted(p.name for p in datasets.iterdir() if p.is_dir()) if datasets.is_dir() else [],
            "job_running": active is not None, "queue": queue_view()}
    current = current_job_dir()
    if current is not None:
        data["job"] = dict(status_brief(read_json(current / "status.json", {}) or {}), job=current.name)
    show(data)
    return 0


def cmd_jobs(args) -> int:
    active = running_dir()
    dirs = [d for d in OUTPUT.iterdir() if d.is_dir() and d.name not in NOT_JOBS
            and (d / "status.json").is_file()] if OUTPUT.is_dir() else []
    show([job_line(d, active) for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)])
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
    if not args.diff:
        print((d / "config.yaml").read_text(encoding="utf-8"))
        return 0
    import yaml

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
    """A job's files to fetch: {"base": folder, "files": [{"path" under base, "size"}]}. --what picks logs (default),
    probes, samples, loras, best or all; --step keeps that step's images and LoRA."""
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
    loras = sorted(d.glob("*.safetensors"))
    if what & {"loras", "all"}:
        picked += [p for p in loras if tag is None or tag in p.name]
    elif "best" in what:
        picked += [p for p in loras if p.stem.endswith("_best")]
    show({"base": str(d.parent.resolve()), "files": [{"path": p.relative_to(d.parent).as_posix(), "size": p.stat().st_size}
                                           for p in dict.fromkeys(picked)]})
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


def cmd_start(args) -> int:
    """Queue a run. The checks of the WebUI's queue_run, the trainer's own validate, and refusing to overwrite an
    earlier run's folder."""
    from rulztrainer.config import TrainConfig, apply_preset

    data = base_config(args).to_dict()
    before = dict(data)
    if args.preset:
        data = apply_preset(data, args.preset)
    data.update(parse_overrides(args.set))
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
    command("poll", cmd_poll, True)
    p = command("files", cmd_files, True)
    p.add_argument("--what", action="append", choices=("logs", "probes", "samples", "loras", "best", "all"))
    p.add_argument("--step", type=int, help="only this step's images and LoRA")
    p = command("start", cmd_start)
    p.add_argument("--job", required=True, help="the new run's job name")
    p.add_argument("--from", dest="base", metavar="JOB", help="start from this job's config")
    p.add_argument("--preset", default="")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--overwrite", action="store_true", help="replace an existing job folder of that name")
    p.add_argument("--dry-run", action="store_true", help="check everything, show the plan, queue nothing")
    p.add_argument("--resume-queue", action="store_true", help="also resume the queue when it is paused")
    p = command("stop", cmd_stop)
    p.add_argument("--kill", action="store_true", help="kill the process: no final save, a waiting queue pauses")
    p = command("queue", cmd_queue)
    p.add_argument("action", nargs="?", choices=("show", "pause", "resume", "remove"), default="show")
    p.add_argument("id", nargs="?", help="remove: the waiting run's id")
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
