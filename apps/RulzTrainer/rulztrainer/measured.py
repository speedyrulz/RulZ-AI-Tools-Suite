"""Measured GPU memory of finished runs, for the wizard's block swap plan (webui.wizard_recommendations).

The plan is an estimate from counting: the frozen model's size, a flat allowance for a step's memory and the adapter's
counted size (runplan.py). Every run also measures what it really takes. This module keeps those measurements with the
settings that decide memory, and finds the one to plan from:

  * the record: one JSON file (RECORD_FILE, under the install's output folder, written whole: a temporary file that
    then takes its place) with one row per finished run. A run adds its row when it ends (Trainer._record_memory);
    `python -m rulztrainer.measured` adds the runs of an output folder from before this existed.
  * which runs count (measurement): only runs that finished without an error, trained MIN_STEPS steps or more, ran
    at least one likeness probe when they had probes, and, with resolution step-up, say they reached their highest
    resolution. A failed run's peak is what did not fit; a run of a few steps may not have met its largest picture,
    a sample or a probe yet. A run stopped by hand or by early stopping after that counts.
  * the lookup (find): rows of the same family, model source and file format, quantization, largest training
    resolution, batch size, gradient checkpointing, clip length and step extras (EXTRAS), on one GPU, not a
    fine-tune. A row made with another block swap count or adapter is adjusted: the weights of the blocks in
    between, and the difference of the two adapters' counted sizes (runplan.adapter_gb, the estimate's own term).
    The blocks' weights are the run's own where it kept them (block_gb: the frozen bytes of each block, read off
    the model when the run ends; for older runs the pinned GB of train.log's "Block swap: ..." line over the blocks
    it swapped), else the estimate's term, the model's size over its blocks. The estimate's term can be well off
    (Krea 2 int8: 0.48 GB a block against 0.24 GB held), which matters most when a row is asked for more blocks
    than it ran with. Of the rows whose adapter is nearest the form's, the RECENT_ROWS newest are weighed and the
    one that needs the most is used, when its adjustment is CLOSE_GB or less; otherwise the caller keeps its
    estimate.

What is measured, and why it is not simply the run's peak (status.json's peak_mem): with gradient checkpointing and
checkpoint_keep_blocks on auto (the default) the trainer fills the card's spare memory on purpose. At the start it
measures one forward and backward on the largest training bucket and lets as many blocks as fit keep their activations
(checkpointing.py), so the run's peak follows the card, not the settings: a plan made from it would never swap fewer
blocks. The figure a row is planned from (step_gb) is that start-of-run measurement itself, the step with one block
kept (Trainer._calibrate_checkpointing's peak_one, reserved memory): what a step needs before any spare memory is
spent. The run's peak is held against it: where the run went over what the calibration's own allocated peak gives
for the blocks it kept (the identity loss's decode, samples, probes, the optimizer's state), the excess is added. A
run records the calibration's figures in its status.json (checkpoint_memory); for older runs they are read from the
"Checkpointing: ..." line of train.log, which prints them rounded and without the allocated figure: step_gb is then
good to about 0.2 GB, and UNSEEN_GB is added for an excess that would hide under the reserved one. A run on auto
with neither is left out.

A run that kept a fixed number of blocks, or ran without checkpointing, did not fill the card: its peak_mem is its
figure (the allocator's peak for the whole process: caching the captions, samples and probes included, so it errs
toward swapping more).

The plan leaves free what the trainer itself leaves (margin_gb: checkpointing.memory_margin) plus SPARE_GB, and is the
fewest blocks that fit: blocks kept in spare memory are not planned for. Block swap holds INFLIGHT_BLOCKS swapped
blocks' weights on the GPU at a time: they are added when a run that swapped fewer of them is asked to swap, and
never taken off. What is still estimated: the adapter's size used to adjust a row, the per-block size for a row that
kept none, the step staying the same but for those terms, and everything for settings no finished run of yours has.
Sample and probe sizes, the sampling model, the dataset's aspect ratios and other programs on the card are not
compared.

Memory here is in GiB, the unit of a card's size (nvidia-smi MiB / 1024) and of the trainer's checkpointing log;
peak_mem, as the Monitor and the run report show it, and the estimate's sizes are in GB of 10^9 bytes (7% larger
numbers) and are turned into GiB before they meet a card's size.

Checked against faked and saved run files only: no GPU run has been planned from a measurement yet."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD_FILE = ROOT / "output" / "_vram_measured.json"
SCAN_SKIP = ("_history", "_crops", "_compare")  # folders under output/ that are not jobs
VERSION = 1
MAX_ROWS = 400  # the newest are kept

MIN_STEPS = 100  # fewer: the run may not have met its largest bucket, a sample or a probe
SPARE_GB = 0.5  # left free on top of the trainer's own margin: the calibration measures one step, a run has thousands
TIGHT_GB = 1.0  # a run whose peak came this close to its card is called out
CLOSE_GB = 4.0  # a row adjusted by more than this (blocks and adapter together) is not used
# Added to the step of a run whose calibration left no allocated figure (a train.log has the reserved one only, and
# rounded): the run's peak is allocated memory, so what it took beyond the calibration can hide under the difference
UNSEEN_GB = 0.5
RECENT_ROWS = 5  # of equally comparable runs the newest few are weighed, so one old heavy run does not hold the plan
INFLIGHT_BLOCKS = 2  # swapped blocks whose weights are on the GPU at a time (blockswap.BlockSwap max_inflight)
GIB_PER_GB = 1e9 / 2 ** 30  # peak_mem is in GB of 10^9 bytes; a card's memory and the calibration are in GiB
LOCK_WAIT, LOCK_STALE = 10.0, 30.0  # seconds a writer waits for the record's lock; the age a left-over lock is taken at

# Settings that add GPU work to a step and that the adjustment has no term for: a row counts only when they agree
EXTRAS = {"id_loss": lambda get: float(get("id_loss_weight", 0) or 0) > 0,
          "id_align": lambda get: bool(get("id_align", False)),
          "body_align": lambda get: bool(get("body_align", False)),
          "isolation": lambda get: bool(get("isolation", False)),
          "activation_offload": lambda get: bool(get("activation_offload", False)),
          "checkpoint_save_attention": lambda get: bool(get("checkpoint_save_attention", False)),
          "reward": lambda get: int(get("reward_steps", 0) or 0) > 0}
# What must agree for a row to be comparable (find); the block swap count and the adapter are adjusted for instead
SAME_KEYS = ("family", "side", "batch_size", "gradient_checkpointing", "frames", "video_side", "extras")
# The adapter as the row names it: compared as a whole when a size could not be counted for either side
ADAPTER_KEYS = ("network_type", "lora_rank", "lokr_factor", "oft_block_size", "dora", "optimizer", "weight_average",
                "race")


def margin_gb(card_gb: float, id_loss: bool = False) -> float:
    """GiB the trainer leaves unused below a card's size when it budgets a step (checkpointing.memory_margin, kept in
    step with it by a test): max(1.5, 8%), plus 1.5 while the identity loss runs."""
    return max(1.5, 0.08 * float(card_gb)) + (1.5 if id_loss else 0.0)


# ---------------------------------------------------------------- what a run's settings say about memory

def _clip(cfg, family: str, get) -> tuple[int, int]:
    """(frames per training clip, the clips' side in px) for a video family training on clips; (1, 0) for stills."""
    frames = int(get("train_frames", 1) or 1)
    if frames <= 1:
        return 1, 0
    try:
        from .archs import _ARCHS

        arch = _ARCHS[family]()
        if not arch.supports_video:
            return 1, 0
        frames = int(arch.snap_frames(frames))
    except Exception:  # noqa: BLE001  (an unknown family: the setting as it stands)
        pass
    return (frames, int(get("video_resolution", 0) or 0)) if frames > 1 else (1, 0)


def settings(cfg, overrides: dict | None = None) -> dict:
    """The settings of a config that decide a run's GPU memory, as a row keeps them. overrides: TrainConfig fields to
    read instead of cfg's (the wizard's picks)."""
    o = dict(overrides or {})

    def get(key, default=None):
        return o[key] if key in o else getattr(cfg, key, default)

    family = cfg.arch_name()
    sized = dataclasses.replace(cfg, resolutions=list(o["resolutions"])) if o.get("resolutions") else cfg
    side = max(sized.training_sides())
    frames, video_side = _clip(cfg, family, get)
    kind = str(get("network_type", "lora") or "lora")
    return {
        "family": family,
        "transformer_quant": str(get("transformer_quant", "auto") or "auto"),
        "model_source": str(get("model_source", "files") or "files"),
        "model_file_quant": str(get("model_file_quant", "") or ""),
        "side": int(side), "megapixels": round(side * side / (1024 * 1024), 2),
        "batch_size": int(get("batch_size", 1) or 1),
        "gradient_checkpointing": bool(get("gradient_checkpointing", True)),
        "checkpoint_keep_blocks": int(get("checkpoint_keep_blocks", -1)),
        "frames": frames, "video_side": video_side,
        "multi_gpu": str(get("multi_gpu", "off") or "off"),
        "finetune": bool(get("finetune", False)),
        "block_swap": max(0, int(get("block_swap", 0) or 0)),
        "network_type": kind, "lora_rank": int(get("lora_rank", 16) or 16),
        "lokr_factor": int(get("lokr_factor", 8) or 8), "oft_block_size": int(get("oft_block_size", 32) or 32),
        "dora": bool(get("dora", False)), "optimizer": str(get("optimizer", "adamw") or "adamw"),
        "weight_average": float(get("lora_ema", 0) or 0) > 0,
        "race": bool(get("race", False) and getattr(cfg, "race_configs", None)),
        "extras": {name: bool(test(get)) for name, test in EXTRAS.items()},
    }


def adapter_words(row: dict) -> str:
    """A row's adapter in a few words (the listing)."""
    if row.get("finetune"):
        return "fine-tune"
    kind = row.get("network_type", "lora")
    name = (f"LoKr f{row.get('lokr_factor')}" if kind == "lokr" else f"OFT b{row.get('oft_block_size')}"
            if kind == "oft" else f"LoRA r{row.get('lora_rank')}")
    return name + (" DoRA" if row.get("dora") and kind != "oft" else "") + (" (race)" if row.get("race") else "")


# ---------------------------------------------------------------- the start-of-run measurement (checkpointing)

def calibration(kept: int, candidates: int, peak_one: float, per_block: float, budget: float, total: float,
                alloc_one: float | None = None) -> dict:
    """One GPU's figures from Trainer._calibrate_checkpointing (all in bytes there) as status.json keeps them
    (checkpoint_memory), in GiB: the step's reserved peak with one block kept (step_gb) and its allocated peak
    (alloc_gb: what the run's own peak, an allocated figure, is held against), what each further kept block adds
    (per_block_gb), how many of the candidate blocks it kept, the peak it expected with those (peak_gb), the budget
    it kept them within and the card's size."""
    gib = 2 ** 30
    return {"kept": int(kept), "candidates": int(candidates), "step_gb": round(peak_one / gib, 3),
            "alloc_gb": None if alloc_one is None else round(alloc_one / gib, 3),
            "per_block_gb": round(per_block / gib, 3),
            "peak_gb": round((peak_one + max(0, int(kept) - 1) * per_block) / gib, 3),
            "budget_gb": round(budget / gib, 3), "total_gb": round(total / gib, 3)}


# "... blocks keep their activations, measured at 1234 tokens (cuda:0: 9 of 20 (0.31 GB per extra block, peak 14.2 of
# 16 GB, budget 14.4))" (Trainer._calibrate_checkpointing; a GPU with no candidate block is not listed)
_LOG_LINE = re.compile(r"Checkpointing: \d+ of \d+ blocks keep their activations, measured at \d+ tokens \((.*)\)\s*$")
_LOG_GPU = re.compile(r"(\S+): (\d+) of (\d+) \((-?[\d.]+) GB per extra block, peak (-?[\d.]+) of (\d+) GB, "
                      r"budget (-?[\d.]+)\)")


def log_calibration(text: str) -> dict:
    """The calibration figures of the last run in a train.log, per GPU, as calibration() gives them; {} when the log
    has no such line. The log prints them rounded (the peak to 0.1 GB, the slope to 0.01), so step_gb is good to
    about 0.2 GB, and the card's size as a whole GB (16 for a card of 15.93)."""
    found = {}
    for line in text.splitlines():
        m = _LOG_LINE.search(line)
        if not m:
            continue
        gpus = {}
        for dev, kept, cands, per_block, peak, total, budget in _LOG_GPU.findall(m.group(1)):
            kept, per_block, peak = int(kept), float(per_block), float(peak)
            gpus[dev] = {"kept": kept, "candidates": int(cands), "per_block_gb": per_block,
                         "step_gb": round(peak - max(0, kept - 1) * per_block, 3), "peak_gb": peak,
                         "budget_gb": float(budget), "total_gb": float(total), "from_log": True}
        if gpus:
            found = gpus  # a later run in the same folder (a carried-on one) writes below the earlier one's
    return found


# "Block swap: the frozen weights of blocks 0-7 (8 of 28) stay in RAM, 1.9 GB pinned, and stream to cuda:0 ..."
# (Trainer._setup_block_swap; GB of 10^9 bytes)
_LOG_SWAP = re.compile(r"Block swap: the frozen weights of blocks 0-\d+ \((\d+) of (\d+)\) stay in RAM, ([\d.]+) GB "
                       r"pinned")


def log_block_swap(text: str) -> tuple[int, float] | None:
    """(blocks swapped, GiB of their frozen weights) of the last run in a train.log; None when it swapped none. The
    log prints the size to 0.1 GB."""
    found = None
    for m in _LOG_SWAP.finditer(text):
        if int(m.group(1)) > 0 and float(m.group(3)) > 0:
            found = (int(m.group(1)), round(float(m.group(3)) * GIB_PER_GB, 3))
    return found


def _block_sizes(source) -> list[float] | None:
    """Each block's frozen weights in GiB as a run kept them (block_gb), or None when that is not a list of sizes."""
    try:
        sizes = [round(float(v), 4) for v in source] if isinstance(source, (list, tuple)) else []
    except (TypeError, ValueError):
        return None
    return sizes if sizes and all(v > 0 for v in sizes) else None


def _figures(source, device: str) -> dict | None:
    """The calibration figures for the training GPU out of a checkpoint_memory mapping; the only entry when the
    device is named differently there ("cuda" against "cuda:0")."""
    if not isinstance(source, dict) or not source:
        return None
    fig = source.get(device)
    if fig is None and len(source) == 1:
        fig = next(iter(source.values()))
    if not isinstance(fig, dict):
        return None
    try:  # every figure a number, or the run is left out (a damaged status must not stop a rebuild)
        out = {"step_gb": float(fig.get("step_gb") or 0), "kept": int(fig.get("kept") or 0),
               "candidates": int(fig.get("candidates") or 0), "per_block_gb": max(0.0, float(fig.get("per_block_gb") or 0)),
               "alloc_gb": float(fig.get("alloc_gb") or 0) or None, "total_gb": float(fig.get("total_gb") or 0) or None,
               "from_log": bool(fig.get("from_log"))}
    except (TypeError, ValueError):
        return None
    return out if out["step_gb"] > 0 and out["kept"] >= 0 and out["candidates"] >= 0 else None


def measurement(cfg, status, *, folder=None, cards: dict | None = None, when: float | None = None,
                adapter_gb: float | None = None, log_text: str | None = None,
                block_gb: list | None = None, card_fallback: float | None = None) -> tuple[dict | None, str]:
    """(row, "") for a run whose measurement counts, else (None, why not). cfg: the run's TrainConfig; status: its
    status.json; cards: GiB of each GPU it ran on when known (else what the run kept); adapter_gb: the adapter's
    counted size (None: counted here, runplan.adapter_gb); log_text: its train.log, for the calibration figures and
    the swapped blocks' size of a run that did not keep them in its status; block_gb: each block's frozen weights in
    GiB (else what the run kept); card_fallback: the card's GiB for a run that kept none (it beats a log's whole GB)."""
    if not isinstance(status, dict):
        return None, "no status.json"
    if status.get("error"):
        return None, "failed: " + " ".join(str(status["error"]).split())[:80]
    if not status.get("done") or status.get("stage") != "finished":
        return None, f"did not finish (stage: {' '.join(str(status.get('stage') or 'unknown').split())[:40]})"
    peaks = {}
    for dev, gb in (status.get("peak_mem") or {}).items() if isinstance(status.get("peak_mem"), dict) else ():
        with contextlib.suppress(TypeError, ValueError):
            if float(gb) > 0:
                peaks[str(dev)] = float(gb)
    if not peaks:
        return None, "no GPU peak in its status.json"
    try:
        step = int(status.get("step") or 0)
    except (TypeError, ValueError):
        step = 0
    if step < MIN_STEPS:
        return None, f"only {step} steps (under {MIN_STEPS})"
    if "likeness" in status and not status.get("probes"):  # a run with likeness probes that was stopped before one
        return None, "ended before any likeness probe ran"
    try:
        row = settings(cfg)
        sides = len(cfg.training_sides())
    except Exception as e:  # noqa: BLE001  (a config this version cannot size)
        return None, f"settings not read: {type(e).__name__}"
    if getattr(cfg, "resolution_step_up", False) and sides > 1:  # it must say it reached its last stage
        try:
            stage, stages = int(status["res_stage"]), max(sides, int(status.get("res_stages") or 0))
        except (KeyError, TypeError, ValueError):
            return None, "a step-up run whose status does not say which resolution it reached"
        if stage + 1 < stages:
            return None, "stopped before its highest resolution"
    device = max(peaks, key=lambda d: peaks[d])
    peak = peaks[device] * GIB_PER_GB
    fig = _figures(status.get("checkpoint_memory"), device)
    if fig is None and log_text:
        fig = _figures(log_calibration(log_text), device)
    if fig is not None:
        # What the run's peak (allocated memory) is held against: the calibration's own allocated peak with the
        # blocks the run kept, or with the 3 its last pass kept, which stays in the allocator's peak. A log has
        # only the reserved figure, which is the larger one: an excess can hide under it, hence UNSEEN_GB.
        kept, cands, per_block = fig["kept"], fig["candidates"], fig["per_block_gb"]
        base, unseen = (fig["alloc_gb"], 0.0) if fig["alloc_gb"] else (fig["step_gb"], UNSEEN_GB)
        expected = base + max(0, max(kept, min(3, cands)) - 1) * per_block
        step_gb = fig["step_gb"] + max(0.0, peak - expected) + unseen
        source = "log" if fig["from_log"] else "calibration"
        row.update(kept_blocks=kept, kept_block_gb=round(per_block, 3))
    elif row["gradient_checkpointing"] and row["checkpoint_keep_blocks"] < 0:
        return None, ("its peak follows the card (blocks kept by measurement) and it left no calibration figures"
                      + ("" if log_text else " (no train.log read)"))
    else:  # a fixed number of kept blocks, or no checkpointing: the run did not fill the card by itself
        step_gb, source = peak, "peak"
    if adapter_gb is None:
        try:
            from . import runplan

            adapter_gb = runplan.adapter_gb(cfg)
        except Exception:  # noqa: BLE001
            adapter_gb = 0.0
    totals = cards if cards is not None else status.get("gpu_total_gb")
    card = None
    if isinstance(totals, dict):
        with contextlib.suppress(TypeError, ValueError):
            card = float(totals.get(device) or 0) or None
    if card is None and fig is not None and not fig["from_log"]:
        card = fig["total_gb"]
    if card is None and card_fallback:
        card = float(card_fallback)
    if card is None and fig is not None:
        card = fig["total_gb"]
    sizes = _block_sizes(block_gb if block_gb is not None else status.get("block_gb"))
    if sizes is not None:
        row["block_gb"] = sizes
    elif log_text and row["block_swap"] > 0:
        swapped = log_block_swap(log_text)
        if swapped is not None and swapped[0] <= row["block_swap"]:  # (the trainer swaps all but one block at most)
            row.update(swapped_blocks=swapped[0], swapped_gb=swapped[1])
    row.update(job=str(getattr(cfg, "job_name", "") or ""), run_id=str(status.get("run_id") or ""),
               folder=str(folder) if folder else "", finished=float(when if when is not None else time.time()),
               adapter_gb=round(float(adapter_gb or 0.0), 3), step_gb=round(step_gb, 3), step_from=source,
               peak_gb=peaks[device], peak_mem=peaks, card_gb=None if card is None else round(card, 2), steps=step,
               planned_steps=int(getattr(cfg, "steps", 0) or 0))
    return row, ""


# ---------------------------------------------------------------- the record file

def _usable(row) -> bool:
    try:
        return (isinstance(row, dict) and bool(row.get("family")) and float(row.get("step_gb") or 0) > 0
                and float(row.get("peak_gb") or 0) > 0)
    except (TypeError, ValueError):
        return False


def load(path: Path | None = None, strict: bool = False) -> list[dict]:
    """The record's rows, oldest first; [] when there is no record or it cannot be read. strict (a writer about to
    write the record back): a file that is there but cannot be opened raises instead, after a few tries, so a
    moment's lock on it (a scanner, another program) never has the record written over as if it were empty. A file
    that opens and is not a record is one all the same: it is replaced."""
    path = Path(path or RECORD_FILE)
    for attempt in range(6):
        try:
            text = path.read_text(encoding="utf-8")
            break
        except FileNotFoundError:
            return []
        except OSError:
            if not strict:
                return []
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))
    try:
        data = json.loads(text)
    except ValueError:
        return []
    rows = data.get("runs") if isinstance(data, dict) else None
    return [r for r in rows if _usable(r)] if isinstance(rows, list) else []


def _replace(tmp: Path, path: Path) -> None:
    for attempt in range(6):  # on Windows a reader (the WebUI) can briefly lock the target
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def save(rows: list[dict], path: Path | None = None) -> Path:
    """Write the record whole: into a temporary file beside it, which then takes its place."""
    path = Path(path or RECORD_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r.get("finished") or 0)[-MAX_ROWS:]
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps({"version": VERSION, "runs": rows}, indent=1), encoding="utf-8")
        _replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return path


@contextlib.contextmanager
def _locked(path: Path):
    """One writer at a time reads, changes and writes the record (two runs on two GPUs can end together). The lock
    file holds its writer's own token, and a writer removes only a lock that still holds its token. A lock left by a
    killed writer is taken over after LOCK_STALE seconds (a write takes milliseconds); when the lock cannot be had
    within LOCK_WAIT seconds the write goes ahead without it: a row lost that way comes back with a rebuild from the
    job folders."""
    lock = path.with_name(path.name + ".lock")
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    held = False
    deadline = time.time() + LOCK_WAIT
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(token)
            held = True
            break
        except (FileExistsError, PermissionError):
            # (PermissionError: on Windows a lock another writer is just removing cannot be created over yet)
            with contextlib.suppress(OSError):
                if time.time() - lock.stat().st_mtime > LOCK_STALE:
                    lock.unlink()
                    continue
            if time.time() >= deadline:
                break
            time.sleep(0.02)
        except OSError:
            break
    try:
        yield
    finally:
        if held:
            with contextlib.suppress(OSError):
                if lock.read_text(encoding="utf-8") == token:
                    lock.unlink()


def identity(row: dict) -> tuple:
    """What makes two rows the same run: its run id, else its folder and job name."""
    return ("id", row["run_id"]) if row.get("run_id") else ("folder", row.get("folder", ""), row.get("job", ""))


def merge(rows: list[dict], new: list[dict]) -> list[dict]:
    """`rows` with `new` added, a run's earlier row replaced by its new one."""
    fresh = {identity(r) for r in new}
    return [r for r in rows if identity(r) not in fresh] + list(new)


def add(row: dict, path: Path | None = None) -> Path:
    """Add one run's row to the record (replacing that run's earlier row)."""
    path = Path(path or RECORD_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        return save(merge(load(path, strict=True), [row]), path)


# ---------------------------------------------------------------- the lookup

def comparable(row: dict, want: dict, quant: str | None) -> bool:
    """Is this row a run of the settings in `want` (settings() of the form, plus its resolved "quant"), block swap
    count and adapter aside? quant: the row's resolved quantization (None: unknown, so not comparable)."""
    if row.get("multi_gpu", "off") != "off" or row.get("finetune") or len(row.get("peak_mem") or {"": 0}) != 1:
        return False
    if quant is None or quant != want.get("quant"):
        return False
    if row.get("model_source") != want.get("model_source"):
        return False  # a file's weights against a repo's quantized as they load: not the same model in memory
    if want.get("model_source") == "files" and row.get("model_file_quant") != want.get("model_file_quant"):
        return False  # (a pruned file against a full one resolves to the same quantization)
    for key in SAME_KEYS:
        mine, theirs = want.get(key), row.get(key)
        if key == "extras":
            mine, theirs = ({k: bool((v or {}).get(k)) for k in EXTRAS} for v in (mine, theirs))
        elif key == "video_side" and int(want.get("frames") or 1) <= 1:
            continue  # stills: the clips' size plays no part
        if mine != theirs:
            return False
    return True


def swapped_weights(row: dict, per_block_gb: float, blocks: int):
    """(W, F, source): W(n) is the GiB of frozen weights the first n blocks hold, the blocks block swap takes, and
    F(n) what is on the GPU of them at a time with n swapped (INFLIGHT_BLOCKS of them, the largest). From the run's
    own block sizes ("measured"), else from what its log says its swapped blocks held, spread evenly ("log"), else
    the estimate's per_block_gb (GB of 10^9 bytes) for every block ("estimate")."""
    sizes = _block_sizes(row.get("block_gb"))
    if sizes is not None and len(sizes) >= blocks - 1:
        return ((lambda n: float(sum(sizes[:max(0, n)]))),
                (lambda n: float(sum(sorted(sizes[:max(0, n)], reverse=True)[:INFLIGHT_BLOCKS]))), "measured")
    try:
        held, count = float(row.get("swapped_gb") or 0), int(row.get("swapped_blocks") or 0)
    except (TypeError, ValueError):
        held, count = 0.0, 0
    each, source = (held / count, "log") if held > 0 and count > 0 else (float(per_block_gb) * GIB_PER_GB, "estimate")
    return (lambda n: each * max(0, n)), (lambda n: each * max(0, min(INFLIGHT_BLOCKS, n))), source


def fit(row: dict, want: dict, *, vram: float, per_block_gb: float, max_swap: int, blocks: int,
        adapter_gb: float) -> dict | None:
    """The block swap count this row's measurement gives for a card of `vram` GiB, or None when the row would need
    more than CLOSE_GB of adjusting (or its adapter cannot be compared).

    The row's step took S GiB (step_gb) with n_m blocks swapped and an adapter of a_m GB; the form's adapter counts
    a GB (both of 10^9 bytes: turned into GiB). With n blocks swapped the same step is taken to need
        S + (a - a_m) + W(n_m) - W(n) + max(0, F(n) - F(n_m))
    where W(n) is what the first n blocks' frozen weights hold and F(n) what is on the GPU of them at a time
    (swapped_weights: the run's own sizes, else the estimate's per_block_gb each), and n is the fewest blocks that
    leave the trainer's margin and SPARE_GB of the card free. The F term only ever adds: a run that swapped nothing
    held no block in flight, and one that swapped some is not credited for the ones it held."""
    n_m = min(max(0, int(row.get("block_swap") or 0)), max(0, blocks - 1))  # what the trainer swapped at most
    a_m = float(row.get("adapter_gb") or 0.0)
    if adapter_gb > 0 and a_m > 0:
        delta = (adapter_gb - a_m) * GIB_PER_GB
    elif all(row.get(k) == want.get(k) for k in ADAPTER_KEYS):
        delta = 0.0  # not counted (an unknown family), but the same adapter
    else:
        return None
    step = float(row["step_gb"])
    keep_free = margin_gb(vram, bool((want.get("extras") or {}).get("id_loss"))) + SPARE_GB
    weights, inflight, sized = swapped_weights(row, per_block_gb, blocks)

    def needs(count: int) -> float:  # what the step is expected to need with `count` blocks swapped
        return step + delta + weights(n_m) - weights(count) + max(0.0, inflight(count) - inflight(n_m))

    top = max(0, int(max_swap))
    n = next((c for c in range(top + 1) if needs(c) + keep_free <= vram + 1e-9), top)
    adjust = abs(delta) + abs(weights(n) - weights(n_m))
    if adjust > CLOSE_GB + 1e-9:
        return None
    need = needs(n)
    peak = float(row["peak_gb"]) * GIB_PER_GB  # the run's own peak, in the card's unit
    card = float(row.get("card_gb") or 0) or None
    return {"swap": n, "saved_gb": weights(n), "unswapped_gb": needs(0), "blocks_from": sized, "row": row,
            "step_gb": step, "at_blocks": n_m,
            "run_peak_gb": peak, "kept_blocks": int(row.get("kept_blocks") or 0), "from": row.get("step_from", "peak"),
            "adapter_delta_gb": delta, "adjust_gb": adjust, "need_gb": need, "free_gb": vram - need,
            "keep_free_gb": keep_free,
            "tight": need + keep_free > vram + 1e-6,  # even the most blocks leave less than that free
            "card_gb": card, "headroom_gb": (card or vram) - peak,
            # (a run whose card is not known and whose peak is over this one was made on a bigger card: no call)
            "close_call": card - peak < TIGHT_GB if card else 0 <= vram - peak < TIGHT_GB}


def find(rows: list[dict], want: dict, *, vram: float, per_block_gb: float, max_swap: int, blocks: int,
         adapter_gb: float, quant_of=None) -> dict | None:
    """The plan (fit) from the comparable rows whose adapter is nearest the form's (to 0.1 GB): of those the
    RECENT_ROWS newest are weighed and the one that needs the most memory (the most blocks) is used, the newer of
    two that need the same. None when no row is comparable and close enough. quant_of: a row's resolved
    quantization (default: its transformer_quant as set)."""
    plans = []
    for row in rows:
        try:  # a row that cannot be read (a record edited by hand) is passed over, the others still count
            if not _usable(row):
                continue
            quant = quant_of(row) if quant_of is not None else row.get("transformer_quant")
            if not comparable(row, want, quant):
                continue
            plan = fit(row, want, vram=vram, per_block_gb=per_block_gb, max_swap=max_swap, blocks=blocks,
                       adapter_gb=adapter_gb)
            if plan is None:
                continue
            plans.append((round(abs(plan["adapter_delta_gb"]), 1), float(row.get("finished") or 0), plan))
        except Exception:  # noqa: BLE001
            continue
    if not plans:
        return None
    nearest = min(p[0] for p in plans)
    recent = sorted((p for p in plans if p[0] == nearest), key=lambda p: -p[1])[:RECENT_ROWS]
    return max(recent, key=lambda p: (p[2]["unswapped_gb"], p[1]))[2]


# ---------------------------------------------------------------- past runs: python -m rulztrainer.measured

def _log_lines(path: Path) -> str:
    """The lines of a train.log this module reads (the calibration's and block swap's), in order, read through line
    by line: they are written at a run's start, however long its log grew afterwards. "" without a log."""
    kept = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "Checkpointing: " in line or "Block swap: " in line:
                    kept.append(line)
    except OSError:
        return ""
    return "".join(kept)


def scan(output, card_gb: float | None = None) -> list[dict]:
    """Every job folder under `output` (a folder with a status.json): {"folder", "job", "row" or None, "why"}. A row's
    time is its status.json's. card_gb: the card the runs were made on, for runs that did not record it."""
    from .config import TrainConfig

    output = Path(output)
    found = []
    for status_path in sorted(output.rglob("status.json")) if output.is_dir() else []:
        job_dir = status_path.parent
        if any(part in SCAN_SKIP for part in job_dir.relative_to(output).parts):
            continue
        entry = {"folder": str(job_dir), "job": job_dir.name, "row": None, "why": ""}
        found.append(entry)
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            when = status_path.stat().st_mtime
        except (OSError, ValueError):
            entry["why"] = "status.json not readable"
            continue
        try:
            cfg, _missing, _unknown = TrainConfig.load_report(job_dir / "config.yaml")
        except Exception as e:  # noqa: BLE001  (no config, or one that is not a mapping)
            entry["why"] = f"config.yaml not readable ({type(e).__name__})"
            continue
        entry["job"] = cfg.job_name or job_dir.name
        log_text = None
        if isinstance(status, dict) and not (status.get("checkpoint_memory") and status.get("block_gb")):
            log_text = _log_lines(job_dir / "train.log")
        try:
            entry["row"], entry["why"] = measurement(cfg, status, folder=job_dir, when=when, log_text=log_text,
                                                     card_fallback=card_gb)
        except Exception as e:  # noqa: BLE001  (one damaged job never stops the others)
            entry["row"], entry["why"] = None, f"not read ({type(e).__name__})"
    return found


def listing(found: list[dict]) -> str:
    """The scan as a table: the runs that count with what was measured, the others with why not."""
    head = ("run", "family", "quant", "MP", "batch", "swap", "adapter", "step GB", "from", "kept", "peak GB", "card",
            "steps", "")
    lines: list[tuple[str, ...]] = [head]
    for e in found:
        r = e["row"]
        if r is None:
            lines.append((e["job"],) + ("",) * (len(head) - 2) + ("ignored: " + e["why"],))
            continue
        usable = r.get("multi_gpu", "off") == "off" and not r.get("finetune")
        note = "counts" if usable else "kept, not used: the plan is for one GPU and an adapter"
        lines.append((r["job"], r["family"], r["transformer_quant"], f"{r['megapixels']:g}", str(r["batch_size"]),
                      str(r["block_swap"]), adapter_words(r), f"{r['step_gb']:.2f}", r["step_from"],
                      str(r["kept_blocks"]) if "kept_blocks" in r else "-", f"{r['peak_gb']:.2f}",
                      f"{r['card_gb']:g}" if r.get("card_gb") else "?", str(r["steps"]), note))
    widths = [max(len(line[i]) for line in lines) for i in range(len(head))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(line, widths)).rstrip() for line in lines)


def rebuild(output, path: Path | None = None, dry_run: bool = False, fresh: bool = False,
            card_gb: float | None = None) -> tuple[list[dict], int]:
    """Add the finished runs under `output` to the record (fresh: start from an empty record instead of keeping the
    rows it has). Returns (the scan, rows the record holds afterwards); dry_run writes nothing."""
    path = Path(path or RECORD_FILE)
    found = scan(output, card_gb)
    new = [e["row"] for e in found if e["row"] is not None]
    if dry_run:
        return found, len(merge([] if fresh else load(path), new)[-MAX_ROWS:])
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        rows = merge([] if fresh else load(path, strict=True), new)
        save(rows, path)
    return found, len(load(path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rulztrainer.measured",
        description="Rebuild the record of measured GPU memory (the wizard's block swap plan reads it) from the job "
                    "folders of past runs. Runs that failed, did not finish or trained under "
                    f"{MIN_STEPS} steps are listed and left out. 'step GB' is what a step took (GiB) and 'from' "
                    "where that comes from: calibration (the run's status.json), log (train.log, rounded) or peak "
                    "(the run's own peak, for a run that did not fill the card by itself); 'peak GB' is the run's "
                    "peak as its report shows it (10^9 bytes).")
    ap.add_argument("output", nargs="?", default=str(ROOT / "output"), help="the folder of job folders (default: "
                                                                            "the install's output folder)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be kept; write nothing")
    ap.add_argument("--fresh", action="store_true", help="start from an empty record instead of adding to it")
    ap.add_argument("--card-gb", type=float, default=None,
                    help="the memory of the card these runs were made on, in GB as nvidia-smi shows it (16 GB card: "
                         "about 16), for runs that did not record it")
    ap.add_argument("--record", default=None, help="the record file (default: output/_vram_measured.json of the "
                                                   "install)")
    args = ap.parse_args(argv)
    path = Path(args.record) if args.record else RECORD_FILE
    if not Path(args.output).is_dir():
        print(f"No such folder: {args.output}", file=sys.stderr)
        return 2
    found, total = rebuild(args.output, path, dry_run=args.dry_run, fresh=args.fresh, card_gb=args.card_gb)
    kept = sum(e["row"] is not None for e in found)
    print(listing(found) if found else f"No job folders (status.json) under {args.output}")
    print(f"\n{kept} of {len(found)} runs measured. " + (
        f"Dry run: nothing written; the record would hold {total} runs ({path})." if args.dry_run
        else f"The record now holds {total} runs ({path})."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
