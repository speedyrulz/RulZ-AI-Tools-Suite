"""Short training runs for speed checks: waits for idle GPUs, then reports the steady-state seconds per step.

    python scripts/bench.py --config configs/my_job.yaml --label base [--set key=value ...] [--steps 20] [--skip 5]
                            [--wait] [--profile N] [--out bench.jsonl]

The run trains --steps optimizer steps with samples, probes and saves off (unless --set turns them on), in a scratch
output folder, then reads the job's status.json and train.log and prints: the median seconds per step after the first
--skip steps, the warm-up cost (the first --skip steps beyond that median: model compile, first allocations), peak
memory per GPU (torch, and nvidia-smi for the whole card), the average loss, and the log's compile and data-parallel
summary lines. --profile N profiles step N (RULZ_PROFILE_STEP; see rulztrainer/profiling.py).

These GPUs are shared with other trainings: the script will not start while any card has more than --busy-mib in use
or another RulzTrainer training runs (--wait polls until they are free), and it stops its own run if another training
starts meanwhile."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def gpu_memory() -> dict[int, int]:
    """MiB in use per GPU, by nvidia-smi index (the whole card, every process)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines() if "," in line)}


def other_trainings(own_pid: int | None = None) -> list[str]:
    """Running RulzTrainer trainings that are not ours (CLI runs and WebUI jobs both run rulztrainer.train)."""
    import psutil

    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            args = p.info["cmdline"] or []
            is_python = (p.info["name"] or "").lower().startswith("python")
        except (psutil.Error, TypeError):
            continue
        cmd = " ".join(args)
        if is_python and "rulztrainer.train" in args and p.info["pid"] not in (own_pid, os.getpid()):
            parent = p.parent()
            if own_pid is None or parent is None or parent.pid != own_pid:
                found.append(f"pid {p.info['pid']}: {cmd[:120]}")
    return found


def busy_reason(busy_mib: int, own_pid: int | None = None) -> str | None:
    others = other_trainings(own_pid)
    if others:
        return "another training is running (" + "; ".join(others) + ")"
    if own_pid is None:
        used = gpu_memory()
        full = {i: m for i, m in used.items() if m > busy_mib}
        if full:
            return "GPU memory in use: " + ", ".join(f"GPU {i} {m} MiB" for i, m in full.items())
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", required=True, help="name of this variant in the report")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--skip", type=int, default=5, help="warm-up steps left out of the median")
    ap.add_argument("--profile", type=int, default=0, help="profile this step (0 = no profile)")
    ap.add_argument("--wait", action="store_true", help="wait for idle GPUs instead of giving up")
    ap.add_argument("--busy-mib", type=int, default=1500, help="a card above this much memory in use counts as busy")
    ap.add_argument("--outdir", default=str(Path(tempfile.gettempdir()) / "rulz_bench"))
    ap.add_argument("--out", default="", help="append the result as one JSON line to this file")
    args = ap.parse_args(argv)

    reason = busy_reason(args.busy_mib)
    while reason and args.wait:
        print(f"waiting: {reason}", flush=True)
        time.sleep(30)
        reason = busy_reason(args.busy_mib)
    if reason:
        print(f"not starting: {reason}")
        return 2

    job = f"bench_{args.label}"
    sets = [f"steps={args.steps}", "sample_every=0", "sample_prompts=[]", "sample_before_training=false",
            "probe_every=0", "probe_at_start=false", "resolution_step_up=false", "save_every=1000000",
            "save_state=false", f"output_dir={json.dumps(args.outdir)}", f"job_name={job}"] + args.set
    cmd = [sys.executable, "-m", "rulztrainer.train", "--config", args.config]
    for s in sets:
        cmd += ["--set", s]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
    if args.profile:
        env["RULZ_PROFILE_STEP"] = str(args.profile)
    job_dir = Path(args.outdir) / job
    job_dir.mkdir(parents=True, exist_ok=True)
    peak: dict[int, int] = {}
    stop = threading.Event()
    aborted: list[str] = []
    t0 = time.time()
    with open(job_dir / "bench_stdout.log", "w", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, env=env)

        def watch():
            while not stop.is_set():
                for i, m in gpu_memory().items():
                    peak[i] = max(peak.get(i, 0), m)
                r = busy_reason(args.busy_mib, own_pid=proc.pid)
                if r:
                    aborted.append(r)
                    proc.terminate()
                    return
                stop.wait(2)

        threading.Thread(target=watch, daemon=True).start()
        code = proc.wait()
        stop.set()
    if aborted:
        print(f"{args.label}: stopped, {aborted[0]}")
        return 3

    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8")) if (job_dir / "status.json").exists() \
        else {}
    times = status.get("step_times") or []
    steady = times[args.skip:] or times
    median = statistics.median(steady) if steady else None
    warmup = (sum(times[:args.skip]) - args.skip * median) if median is not None and len(times) > args.skip else None
    losses = status.get("losses") or []
    log_path = job_dir / "train.log"
    notes = []
    if log_path.exists():
        keys = ("Compile", "compile", "Data parallel", "Checkpointing", "Profile of step", "Weight average",
                "LoRA:", "Error", "Traceback")
        notes = [line.strip()[:260] for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if any(k in line for k in keys)]
    result = {"label": args.label, "exit": code, "wall_s": round(time.time() - t0), "steps": len(times),
              "median_s": round(median, 3) if median is not None else None,
              "warmup_s": round(warmup, 1) if warmup is not None else None,
              "peak_torch_gb": status.get("peak_mem"), "peak_smi_mib": peak,
              "avg_loss": round(sum(losses) / len(losses), 4) if losses else None, "sets": args.set,
              "error": status.get("error")}
    print(json.dumps(result))
    for line in notes[-12:]:
        print("   ", line)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({**result, "notes": notes[-12:]}) + "\n")
    return 0 if code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
