"""Where one training step's GPU time goes (RULZ_PROFILE_STEP=N): torch.profiler over step N, kernels grouped.

Groups: matrix multiplies (cuBLAS / CUTLASS / int8 and FP4 GEMMs), attention, RulzTrainer's own triton kernels
(quantize / dequantize / rescale), inductor-generated triton (torch.compile fusions), copies, everything else; and per
GPU the idle time (step wall time minus kernel time). The summary goes to the log and to profile_step_N.txt in the job
folder. The profiler adds overhead, so read the shares, not the absolute step time.

Also here: the run's peak RAM (PeakRam, always on) and a CUDA memory snapshot of one step
(RULZ_MEMORY_SNAPSHOT_STEP=N), both after OneTrainer's (#1609, #1682)."""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
from pathlib import Path

import torch

log = logging.getLogger("rulztrainer")

GROUPS = (  # first match wins
    ("ours", re.compile(r"^_(quant_act|rescale|rot_quant_act|nvfp4_dequant|nvfp4_quant_act|dequant\w*)_kernel")),
    ("inductor", re.compile(r"^triton_(poi|red|per|tem)_")),
    ("attention", re.compile(r"flash|fmha|attention|_attn|cudnn_generated_fort_native_sdpa", re.I)),
    ("matmul", re.compile(r"gemm|cutlass|cublas|nvjet|xmma|_int_mm|scaled_mm|splitk|s16816|i8816|s1688", re.I)),
    ("copy", re.compile(r"memcpy|memset|copy_kernel|Memcpy|Memset")),
)


def requested_step(variable: str = "RULZ_PROFILE_STEP") -> int:
    """The step to profile from RULZ_PROFILE_STEP, or another such variable (0 = none)."""
    try:
        return max(0, int(os.environ.get(variable, "0")))
    except ValueError:
        return 0


def private_ram_bytes() -> int | None:
    """This process's private memory, None when unknown. Linux: RssAnon (its own anonymous pages, pinned buffers and
    block swap's copies included; not the page cache or memory-mapped model files, which the system can drop).
    Elsewhere psutil's private bytes (Windows' commit charge), else its resident size."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", encoding="ascii") as f:
                for line in f:
                    if line.startswith("RssAnon:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "private", 0) or info.rss)
    except Exception:  # noqa: BLE001
        return None


class PeakRam:
    """The run's peak private RAM (private_ram_bytes), sampled on a daemon thread every `every` seconds: block swap,
    a fine-tune's resting layers, the text encoders and the caches live in RAM, so this is what a machine needs. The
    "64 GB advised" for H3 with block swap was an estimate; this measures it."""

    def __init__(self, every: float = 0.5):
        self.every, self.peak = float(every), 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> int:
        now = private_ram_bytes() or 0
        self.peak = max(self.peak, now)
        return now

    def start(self) -> "PeakRam":
        self.sample()
        self._thread = threading.Thread(target=self._run, name="rulz-peak-ram", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            self.sample()

    def stop(self) -> None:
        self._stop.set()
        self.sample()

    def peak_gb(self) -> float:
        return round(self.peak / 1e9, 2)


class MemorySnapshot:
    """A CUDA memory snapshot of one step (RULZ_MEMORY_SNAPSHOT_STEP=N): every allocation made during step N with its
    size and Python stack, and the memory held before it, as memory_snapshot_step_N.pickle in the job folder (_oom
    added when the step ran out of memory). Open it on https://pytorch.org/memory_viz. Recording slows that step."""

    def __init__(self, job_dir: Path, step: int):
        self.job_dir, self.step, self.recording = Path(job_dir), step, False

    def start(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
            self.recording = True
        except Exception as e:  # noqa: BLE001
            log.warning("Memory snapshot of step %d not recorded: %s", self.step, e)

    def stop(self, tag: str = "") -> Path | None:
        if not self.recording:
            return None
        path = self.job_dir / f"memory_snapshot_step_{self.step}{tag}.pickle"
        try:
            torch.cuda.memory._dump_snapshot(str(path))
            log.info("Memory snapshot of step %d: %s (open it on https://pytorch.org/memory_viz)", self.step, path.name)
            return path
        except Exception as e:  # noqa: BLE001
            log.warning("Memory snapshot of step %d not saved: %s", self.step, e)
            return None
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
            self.recording = False


def group_of(name: str) -> str:
    for group, pattern in GROUPS:
        if pattern.search(name):
            return group
    return "other"


def summarize(kernels: list[tuple[str, int, float]], wall: float | None) -> tuple[dict, list[str]]:
    """kernels: (name, device index, seconds). Returns ({device: {group: seconds, "idle": s}}, report lines)."""
    per_dev: dict[int, dict[str, float]] = {}
    by_name: dict[str, float] = {}
    for name, dev, sec in kernels:
        groups = per_dev.setdefault(dev, {})
        g = group_of(name)
        groups[g] = groups.get(g, 0.0) + sec
        by_name[name] = by_name.get(name, 0.0) + sec
    lines = []
    for dev, groups in sorted(per_dev.items()):
        busy = sum(groups.values())
        if wall is not None:
            groups["idle"] = max(0.0, wall - busy)
        total = (wall if wall is not None else busy) or 1.0
        parts = ", ".join(f"{g} {s:.2f}s ({100 * s / total:.0f}%)" for g, s in
                          sorted(groups.items(), key=lambda kv: -kv[1]))
        lines.append(f"cuda:{dev}: kernels {busy:.2f}s of {total:.2f}s: {parts}")
    lines.append("top kernels:")
    for name, sec in sorted(by_name.items(), key=lambda kv: -kv[1])[:25]:
        lines.append(f"  {sec * 1000:9.1f} ms  [{group_of(name)}] {name[:150]}")
    return per_dev, lines


class StepProfile:
    """Profile one optimizer step: start() before it, stop(wall seconds or None to discard) after it."""

    def __init__(self, job_dir: Path, step: int, devices):
        self.job_dir, self.step, self.devices = Path(job_dir), step, devices
        self.prof = None

    def start(self) -> None:
        acts = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        self.prof = torch.profiler.profile(activities=acts)
        self.prof.__enter__()

    def stop(self, wall: float | None) -> None:
        if self.prof is None:
            return
        for d in self.devices:
            if str(d).startswith("cuda"):
                torch.cuda.synchronize(torch.device(d))
        self.prof.__exit__(None, None, None)
        if wall is None:
            self.prof = None
            return
        kernels = [(e.name, e.device_index, e.device_time_total / 1e6) for e in self.prof.events()
                   if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time_total > 0]
        _, lines = summarize(kernels, wall)
        path = self.job_dir / f"profile_step_{self.step}.txt"
        path.write_text(f"step {self.step}, wall {wall:.2f}s (profiler overhead included)\n" + "\n".join(lines) + "\n",
                        encoding="utf-8")
        log.info("Profile of step %d: %s (details in %s)", self.step, " | ".join(lines[:lines.index("top kernels:")]),
                 path.name)
        self.prof = None
