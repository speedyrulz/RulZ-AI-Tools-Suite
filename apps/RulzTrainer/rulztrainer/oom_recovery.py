"""Out-of-memory recovery for a whole run: when a run ends because memory ran out, the WebUI starts it again on its own.

The trainer already survives one kind of out-of-memory error: a training step that runs out with blocks keeping their
activations is trained again with fewer of them (Trainer._shrink_kept_blocks). Anything else ends the run: the last
step with nothing left to shrink, a sample or probe that runs out outside a step, loading, or the computer running out
of system memory and killing the process. This module decides what the WebUI does then (after Fizgig 9.6, its oom.py,
re-made for RulzTrainer's training states and memory settings); webui.queue_tick carries the decision out.

GPU memory, with a training state of this run to carry on from: carry on once on the same settings (fresh memory:
memory that fragments over hours is the usual cause late in a run). Out again at about the same point (within one
state interval of the last try): carry on one step lighter. Nothing saved yet: one fresh restart one step lighter, but
only before the first state was due (a later restart would throw away the steps trained). System memory (the
computer killed the run, or Python's MemoryError, Windows' "paging file is too small"): with a state, carry on once
on the same settings; then it stops, since more block swap would use more RAM, not less. At most `oom_retries` tries a
run (3; 0 turns this off). Each try is kept in the job folder (oom_recovery.json), so a restarted WebUI keeps the
count, and the logs of the failed tries are kept beside the new one (train_oom1.log, console_oom1.log, ...).

One step lighter, in this order: gradient checkpointing on; no blocks keeping their activations (checkpoint_keep_
blocks 0); one more step of block swap (an eighth of the model's blocks, at least 2). Never the batch size, the
resolution or the precision: those change what the run trains, so the run stops with advice instead. A fine-tune and
a run on several GPUs have no block swap; a fine-tune and a race have no training state to carry on from."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

LEDGER = "oom_recovery.json"
CONSOLE_TAIL_LINES = 40  # a hard kill leaves no error in status.json: the console's last lines say what happened

GPU_MARKS = ("cuda out of memory", "torch.outofmemoryerror", "outofmemoryerror", "hip out of memory",
             "cublas_status_alloc_failed", "cudnn_status_alloc_failed", "cuda error: out of memory")
RAM_MARKS = ("defaultcpuallocator: can't allocate memory", "defaultcpuallocator: not enough memory", "memoryerror",
             "paging file is too small", "os error 1455", "cannot allocate memory", "std::bad_alloc")
KILLED = (-9, 137)  # SIGKILL as Popen reports it, and as a shell or a container reports it (Linux's OOM killer)


def _kind_of(text: str) -> str | None:
    """"gpu", "ram" or None for one piece of text. The GPU's first: torch.OutOfMemoryError ends in "memoryerror"."""
    low = (text or "").lower()
    if any(m in low for m in GPU_MARKS) or ("out of memory" in low and "cpu" not in low):
        return "gpu"
    if any(m in low for m in RAM_MARKS):
        return "ram"
    return None


def classify(error: str | None, console_tail: str = "", returncode: int | None = None,
             killed_by_user: bool = False, platform: str | None = None) -> str | None:
    """What ran out when a run ended: "gpu", "ram" or None (not a memory failure, or a run you stopped or killed).

    error: status.json's error (the exception the trainer ended with). It decides when there is one: a sample or
    probe that ran out earlier is caught and logged, so "out of memory" in the console does not mean the run died of
    it. Without one (the process was killed before it could write it), the console's last lines and the exit code."""
    if killed_by_user or returncode == 0:
        return None
    if error:
        return _kind_of(error)
    tail = "\n".join((console_tail or "").splitlines()[-CONSOLE_TAIL_LINES:])
    found = _kind_of(tail)
    if found:
        return found
    if (platform or sys.platform) != "win32" and returncode in KILLED:
        return "ram"  # the OOM killer leaves no message, only SIGKILL
    return None


def complete_state(path) -> bool:
    """A training state folder the trainer finished writing (train_state.pt is written last)."""
    return bool(path) and (Path(path) / "train_state.pt").is_file()


def run_save(status: dict | None, cfg: dict) -> str | None:
    """The newest training state of this run, to carry on from: the last one its status lists, else the state it was
    itself started from (a try that failed again before saving one of its own). Never an older run's state that
    happens to sit in the same job folder. None for a race or a fine-tune, which cannot carry on from one."""
    if cfg.get("race") or cfg.get("finetune"):
        return None
    for path in reversed(list((status or {}).get("states") or [])):
        if complete_state(path):
            return str(path)
    started = cfg.get("resume_state") or ""
    return started if complete_state(started) else None


def lighter(cfg: dict, n_blocks: int) -> tuple[dict | None, str]:
    """The next lighter setup: (config, what changed) or (None, why there is none)."""
    v = dict(cfg)
    if not v.get("gradient_checkpointing", True):
        v["gradient_checkpointing"] = True
        return v, "gradient checkpointing on"
    keep = int(v.get("checkpoint_keep_blocks", -1))
    if keep != 0:
        v["checkpoint_keep_blocks"] = 0
        return v, f"blocks keeping their activations {'auto' if keep < 0 else keep} -> 0"
    if v.get("finetune"):
        return None, "a fine-tune has no block swap to raise"
    if (v.get("multi_gpu") or "off") != "off":
        return None, "block swap runs on one GPU, and this run uses several"
    if n_blocks <= 1:
        return None, "the model's block count is not known, so block swap is left as it is"
    most = n_blocks - 1  # the trainer swaps at most all blocks but one
    now = max(0, int(v.get("block_swap") or 0))
    if now >= most:
        return None, "block swap is at its most"
    new = min(most, now + max(2, math.ceil(n_blocks / 8)))
    v["block_swap"] = new
    return v, f"block swap {now} -> {new}"


def state_interval(cfg: dict) -> int:
    return int(cfg.get("save_state_every") or 0) or int(cfg.get("save_every") or 0)


def decide(kind: str | None, tries: list[dict], max_tries: int, has_save: bool, step_now: int | None,
           window: int, can_restart: bool) -> dict:
    """What to do after a run ended: {"do": "continue" | "restart" | "stop" | "none", "lighter": bool, "why": text}.
    tries: this run's earlier automatic tries ([{"kind", "step", ...}]); window: steps within which a second failure
    counts as the same point (one state interval)."""
    if kind is None:
        return {"do": "none", "why": ""}
    if max_tries <= 0:
        return {"do": "stop", "why": "out-of-memory retries are off for this run"}
    if len(tries) >= max_tries:
        return {"do": "stop", "why": f"{len(tries)} automatic tries already"}
    mine = [t for t in tries if t.get("kind") == kind]
    if kind == "ram":
        if has_save and not mine:
            return {"do": "continue", "lighter": False, "why": "carrying on from the newest state with fresh memory"}
        return {"do": "stop", "why": "system memory ran out again" if mine else "no training state to carry on from"}
    if not has_save:
        if mine:
            return {"do": "stop", "why": "it ran out of GPU memory again before saving a training state"}
        if not can_restart:
            return {"do": "stop", "why": "no training state to carry on from, and starting again would throw away "
                                         "the steps trained"}
        return {"do": "restart", "lighter": True, "why": "nothing saved yet: starting again one step lighter"}
    if not mine:
        return {"do": "continue", "lighter": False, "why": "carrying on from the newest state with fresh memory"}
    last = mine[-1].get("step")
    near = step_now is not None and last is not None and int(step_now) - int(last) <= max(1, int(window or 1))
    if not near:
        return {"do": "continue", "lighter": False, "why": "it trained well past the last try: fresh memory again"}
    return {"do": "continue", "lighter": True, "why": "out again at about the same point: one step lighter"}


def plan(cfg: dict, status: dict | None, kind: str | None, tries: list[dict], n_blocks: int) -> dict:
    """The whole decision for a run that ended: decide() with this run's numbers, and the config to start when it
    starts again ({"do", "why", "config", "change", "save"})."""
    status = status or {}
    save = run_save(status, cfg)
    step = status.get("step")
    step = int(step) if isinstance(step, (int, float)) else None
    window = state_interval(cfg)
    # a fresh start only while the first state was not due yet (or nothing at all was trained)
    can_restart = not cfg.get("finetune") and (step is None or step == 0 or (
        not cfg.get("race") and bool(cfg.get("save_state", True)) and window > 0 and step < window))
    d = decide(kind, tries, int(cfg.get("oom_retries", 3)), save is not None, step, window, can_restart)
    d.update(config=None, change="", save=save)
    if d["do"] not in ("continue", "restart"):
        return d
    new = dict(cfg)
    if d.get("lighter"):
        new, change = lighter(cfg, n_blocks)
        if new is None:
            return {**d, "do": "stop", "why": f"{d['why']}, but {change}"}
        d["change"] = change
    if d["do"] == "continue":
        new["resume_state"] = save
    d["config"] = new
    return d


# ---- the job folder's record of the tries


def read_tries(job_dir) -> list[dict]:
    try:
        data = json.loads((Path(job_dir) / LEDGER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    tries = data.get("tries") if isinstance(data, dict) else None
    return list(tries) if isinstance(tries, list) else []


def add_try(job_dir, kind: str, step, action: str) -> list[dict]:
    tries = read_tries(job_dir) + [{"kind": kind, "step": step, "action": action, "when": time.time()}]
    path = Path(job_dir) / LEDGER
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"tries": tries}, indent=1), encoding="utf-8")
    tmp.replace(path)
    return tries


def reset(job_dir) -> None:
    """A run started by you (not by recovery) starts with no tries."""
    try:
        (Path(job_dir) / LEDGER).unlink()
    except OSError:
        pass


def keep_logs(job_dir, n: int) -> list[str]:
    """Move the failed try's train.log and console.log aside (the new start empties them): train_oom<n>.log."""
    kept = []
    for name in ("train.log", "console.log"):
        p = Path(job_dir) / name
        if p.is_file():
            dest = p.with_name(f"{p.stem}_oom{n}{p.suffix}")
            try:
                p.replace(dest)
                kept.append(dest.name)
            except OSError:
                pass
    return kept


def console_tail(job_dir, max_bytes: int = 16384) -> str:
    try:
        with open(Path(job_dir) / "console.log", "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


# ---- what the log says


def advice(kind: str, save: str | None) -> str:
    """The note after a memory failure that recovery could not fix."""
    where = f" The newest training state is {save}: put it in Resume from training state folder to carry on." \
        if save else ""
    if kind == "ram":
        return ("System memory (RAM) ran out and the run was stopped. This is the computer's memory, not the graphics "
                "card's: swap fewer blocks (Block swap), close other programs, or use a machine with more RAM." + where)
    return ("The GPU ran out of memory. More block swap, gradient checkpointing on, fewer blocks keeping their "
            "activations, a smaller batch or a lower resolution all use less, and closing other programs that use the "
            "GPU helps too." + where)


def summary(tries: list[dict]) -> str:
    if not tries:
        return ""
    lines = [f"  {i}. {'GPU' if t.get('kind') == 'gpu' else 'system RAM'} ran out "
             f"{'at step ' + str(t['step']) if t.get('step') else 'before the first step'}: {t.get('action', '')}"
             for i, t in enumerate(tries, 1)]
    return "Out-of-memory recovery tried:\n" + "\n".join(lines)


__all__ = ["add_try", "advice", "classify", "complete_state", "console_tail", "decide", "keep_logs", "lighter", "plan",
           "read_tries", "reset", "run_save", "state_interval", "summary"]
