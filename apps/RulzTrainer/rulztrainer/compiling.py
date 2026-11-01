"""torch.compile of the DiT blocks: dynamo limits raised to fit them, and a report of what dynamo did.

Every block compiles its own graphs, and with graph breaks (each quantized Linear's triton wrapper is opaque to
dynamo) a break guards on the block instance, so each of the 28-42 blocks keeps its own cache entries. Training,
sampling, guidance batches, the region-prior pass and every GPU add variants. torch's defaults (8 per code object,
256 across all) then run blocks eagerly without saying so; the limits are raised here, and dynamo's warnings (such as
a hit recompile limit) go to train.log. report() summarizes frames, graphs, graph breaks by reason and compile time.

The compiled kernels persist across machines (compile_cache): torch's on-disk caches already keep them between runs
on one machine, but a fresh vast.ai rental starts cold (Krea 2 nvfp4: 160 s of compiling instead of 36 s). Their
"Mega-Cache" serializes every compile artifact of the process into one file per torch / triton / GPU model, saved in
`<models_dir>/.rulz_compile` after 5 steps and at the end, and loaded before the blocks compile; a file another
torch, triton or GPU made is skipped.

RULZ_COMPILE_NESTED=1 sets nested_graph_breaks (a measurement variant: a break no longer bubbles up to the caller's
frame); RULZ_COMPILE_THREAD_LIMITS=0 leaves torch's own limits in the other threads (the behaviour before
thread_prepare, for measuring it)."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import torch

log = logging.getLogger("rulztrainer")

LIMITS = {"recompile_limit": 64, "cache_size_limit": 64, "accumulated_recompile_limit": 4096,
          "accumulated_cache_size_limit": 4096}
# compile_model: auto compiles runs with at least this many steps left. Measured on an RTX 5060 Ti at 1 MP: MOSS-Image
# int8 6.61 -> 4.90 s per step after a 25 s warm-up, Krea 2 nvfp4 8.77 -> 7.74 s after 36 s: it pays after 15-35 steps.
AUTO_MIN_STEPS = 100


def decide(setting, remaining_steps: int, devices) -> tuple[bool, str]:
    """compile_model true / false / auto -> (compile?, why, for the log; empty when simply off)."""
    if setting is True or setting == "true":
        return True, "compile_model: true"
    if setting != "auto":
        return False, ""
    try:
        import triton  # noqa: F401
    except ImportError:
        return False, "compile_model auto: triton is not installed"
    if not all(str(d).startswith("cuda") for d in devices):
        return False, "compile_model auto: not on a GPU"
    if remaining_steps < AUTO_MIN_STEPS:
        return False, f"compile_model auto: only {remaining_steps} steps to go (the warm-up pays after 15-35)"
    return True, f"compile_model auto: {remaining_steps} steps to go"


_settings: dict = {}  # the dynamo config values prepare() chose, re-applied per thread by thread_prepare()
_thread = threading.local()


def _apply(cfg) -> None:
    for name, value in _settings.items():
        if hasattr(cfg, name):
            setattr(cfg, name, value)


def thread_prepare() -> None:
    """Apply prepare()'s dynamo settings in the calling thread, once per thread. torch's dynamo config is thread-local
    (torch 2.12+): a limit raised in the main thread is still 8 in the autograd thread that recomputes checkpointed
    blocks and in the data-parallel worker threads, where dynamo would then run the blocks eagerly after 8 shapes
    without saying so. Cheap: one attribute check after the first call in a thread."""
    if not _settings or getattr(_thread, "done", False):
        return
    if os.environ.get("RULZ_COMPILE_THREAD_LIMITS") == "0":  # a measurement variant: torch's own limits in other threads
        _thread.done = True
        return
    import torch._dynamo as dynamo

    _apply(dynamo.config)
    _thread.done = True


def prepare(fall_back: bool = False, shape_swaps: bool = False) -> None:
    """Dynamo settings for compiling the blocks; call once before the first compile. fall_back (compile_model: auto):
    a frame that fails to compile runs eagerly instead of stopping the run. shape_swaps (a race): LoRA tensors change
    shape in place (param.data = a tensor of another rank), so their size guards must always be checked. The settings
    are kept for thread_prepare(), which the models call before running a block, so every thread that runs one gets
    them."""
    import torch._dynamo as dynamo

    cfg = dynamo.config
    if fall_back:
        _settings["suppress_errors"] = True
    if shape_swaps:
        # dynamo skips a module tensor's guards while the module's __dict__ is unchanged, and a .data swap leaves it
        # unchanged: the rank-32 graph would then run on rank-64 weights ("expected size 64==32")
        _settings["skip_tensor_guards_with_matching_dict_tags"] = False
    # Python float attributes (LoRA scale / multiplier) would otherwise be traced as 0-d CPU tensors, and the CPU ops
    # on them make Inductor emit C++ that needs MSVC on Windows ("InvalidCxxCompiler: cl is not found")
    _settings["specialize_float"] = True
    for name, floor in LIMITS.items():
        if hasattr(cfg, name):
            _settings[name] = max(int(getattr(cfg, name)), floor)
    if os.environ.get("RULZ_COMPILE_NESTED") == "1" and hasattr(cfg, "nested_graph_breaks"):
        _settings["nested_graph_breaks"] = True
    _apply(cfg)
    _thread.done = True
    dyn_log = logging.getLogger("torch._dynamo")
    for h in log.handlers:
        if h not in dyn_log.handlers:
            dyn_log.addHandler(h)
    dyn_log.setLevel(min(dyn_log.level or logging.WARNING, logging.WARNING))


def cache_dirs(setting: str, models_dir) -> list[Path]:
    """compile_cache -> the folders to look in, the one written first; [] when off. auto: <models_dir>/.rulz_compile,
    and a model bucket's .rulz_compile (scripts/vast/download_models.sh syncs it to models/bucket) to read from."""
    value = str(setting or "").strip()
    if value.lower() in ("", "off", "false", "none"):
        return []
    if value.lower() != "auto":
        return [Path(value)]
    root = Path(models_dir)
    return [root / ".rulz_compile", root / "bucket" / ".rulz_compile"]


def cache_name(device) -> str:
    """The cache file's name: what a cache only works with (torch, triton and the training GPU's model)."""
    import re

    import triton

    dev = torch.device(device)
    gpu = re.sub(r"[^A-Za-z0-9.]+", "-", torch.cuda.get_device_name(dev)).strip("-")
    major, minor = torch.cuda.get_device_capability(dev)
    return f"torch{torch.__version__}-triton{triton.__version__}-{gpu}-sm{major}{minor}.bin".replace("+", "_")


def load_cache(dirs: list[Path], device) -> Path | None:
    """Load the first matching cache file; the file it read, or None (none there, or one torch refused)."""
    try:
        name = cache_name(device)
    except Exception:  # noqa: BLE001  (no triton or no GPU: nothing to load)
        return None
    for d in dirs:
        f = d / name
        if not f.is_file():
            continue
        try:
            torch.compiler.load_cache_artifacts(f.read_bytes())
            return f
        except Exception as e:  # noqa: BLE001
            log.warning("compile_cache: could not load %s (%s); compiling from scratch", f, e)
    return None


def save_cache(dirs: list[Path], device) -> Path | None:
    """Write this process's compile artifacts to the first folder (replacing the file); the file, or None."""
    if not dirs:
        return None
    try:
        got = torch.compiler.save_cache_artifacts()
        if not got:
            return None
        f = dirs[0] / cache_name(device)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(got[0])
        os.replace(tmp, f)
        return f
    except Exception as e:  # noqa: BLE001  (a cache is never worth stopping a run for)
        log.warning("compile_cache: could not save (%s)", e)
        return None


def stats() -> dict:
    """What dynamo has done so far in this process."""
    try:
        from torch._dynamo.utils import compilation_time_metrics, counters
    except Exception:  # noqa: BLE001
        return {}
    breaks = dict(counters.get("graph_break", {}))
    seconds = sum(sum(v) for v in compilation_time_metrics.values()) if compilation_time_metrics else 0.0
    return {"frames": int(counters["frames"].get("total", 0)), "frames_ok": int(counters["frames"].get("ok", 0)),
            "graphs": int(counters["stats"].get("unique_graphs", 0)), "breaks": sum(breaks.values()),
            "break_reasons": sorted(breaks.items(), key=lambda kv: -kv[1]), "recompiles": int(
                sum(counters.get("recompiles", {}).values())), "seconds": seconds,
            "limits": {k: getattr(torch._dynamo.config, k, None) for k in LIMITS}}


def report(when: str) -> None:
    s = stats()
    if not s:
        return
    top = "; ".join(f"{reason[:90]} x{n}" for reason, n in s["break_reasons"][:3]) or "none"
    log.info("Compile stats (%s): %d frames compiled (%d ok), %d graphs, %d graph breaks (%s), %d recompiles, "
             "%.0f s compiling", when, s["frames"], s["frames_ok"], s["graphs"], s["breaks"], top, s["recompiles"],
             s["seconds"])
