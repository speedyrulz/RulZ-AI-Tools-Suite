"""torch.compile of the DiT blocks: dynamo limits raised to fit them, and a report of what dynamo did.

Every block compiles its own graphs, and with graph breaks (each quantized Linear's triton wrapper is opaque to
dynamo) a break guards on the block instance, so each of the 28-42 blocks keeps its own cache entries. Training,
sampling, guidance batches, the region-prior pass and every GPU add variants. torch's defaults (8 per code object,
256 across all) then run blocks eagerly without saying so; the limits are raised here, and dynamo's warnings (such as
a hit recompile limit) go to train.log. report() summarizes frames, graphs, graph breaks by reason and compile time.

RULZ_COMPILE_NESTED=1 sets nested_graph_breaks (a measurement variant: a break no longer bubbles up to the caller's
frame)."""
from __future__ import annotations

import logging
import os

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


def prepare(fall_back: bool = False) -> None:
    """Dynamo settings for compiling the blocks; call once before the first compile. fall_back (compile_model: auto):
    a frame that fails to compile runs eagerly instead of stopping the run."""
    import torch._dynamo as dynamo

    cfg = dynamo.config
    if fall_back:
        cfg.suppress_errors = True
    # Python float attributes (LoRA scale / multiplier) would otherwise be traced as 0-d CPU tensors, and the CPU ops
    # on them make Inductor emit C++ that needs MSVC on Windows ("InvalidCxxCompiler: cl is not found")
    cfg.specialize_float = True
    for name, floor in LIMITS.items():
        if hasattr(cfg, name):
            setattr(cfg, name, max(int(getattr(cfg, name)), floor))
    if os.environ.get("RULZ_COMPILE_NESTED") == "1" and hasattr(cfg, "nested_graph_breaks"):
        cfg.nested_graph_breaks = True
    dyn_log = logging.getLogger("torch._dynamo")
    for h in log.handlers:
        if h not in dyn_log.handlers:
            dyn_log.addHandler(h)
    dyn_log.setLevel(min(dyn_log.level or logging.WARNING, logging.WARNING))


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
