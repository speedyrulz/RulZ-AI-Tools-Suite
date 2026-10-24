"""Where one training step's GPU time goes (RULZ_PROFILE_STEP=N): torch.profiler over step N, kernels grouped.

Groups: matrix multiplies (cuBLAS / CUTLASS / int8 and FP4 GEMMs), attention, RulzTrainer's own triton kernels
(quantize / dequantize / rescale), inductor-generated triton (torch.compile fusions), copies, everything else; and per
GPU the idle time (step wall time minus kernel time). The summary goes to the log and to profile_step_N.txt in the job
folder. The profiler adds overhead, so read the shares, not the absolute step time."""
from __future__ import annotations

import logging
import os
import re
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


def requested_step() -> int:
    """The step to profile from RULZ_PROFILE_STEP (0 = none)."""
    try:
        return max(0, int(os.environ.get("RULZ_PROFILE_STEP", "0")))
    except ValueError:
        return 0


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
