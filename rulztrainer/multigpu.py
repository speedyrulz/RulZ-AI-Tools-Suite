"""Two ways to use more than one GPU.

data_parallel  A full model copy on every GPU, each training its own images per step; the (small) LoRA
               gradients are averaged onto the primary copy, the optimizer steps once, and the updated LoRA
               weights are broadcast back. Throughput scales with the number of GPUs (minus the slowest one's
               lag). No torch.distributed needed, so it works on Windows. Each GPU must hold the whole model.

model_split    The DiT blocks are divided across GPUs (pipeline style); activations hop between devices.
               Halves the memory per GPU, so bf16 or larger images fit, but it is not faster: the GPUs take
               turns. The split is either a manual plan ("cuda:0=0-25;cuda:1=26-41") or auto-balanced by
               estimated cost: blocks before the first trained block only run forward (cost 1), every later
               block also runs backward and checkpoint recompute (cost 4), trained or not.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn

log = logging.getLogger("rulztrainer")


def parse_devices(spec: str, fallback: str) -> list[str]:
    devs = [d.strip() for d in (spec or "").replace(";", ",").split(",") if d.strip()]
    return devs or [fallback]


def module_device(m: nn.Module) -> torch.device | None:
    p = next(m.parameters(), None)
    if p is not None:
        return p.device
    b = next(m.buffers(), None)
    return b.device if b is not None else None


# ---------------------------------------------------------------- data parallel


class DataParallelLoRA:
    """Primary (model, injected) on devices[0] plus replicas; only LoRA weights are exchanged."""

    def __init__(self, primary_injected: dict, replicas: list[tuple[nn.Module, dict, str]]):
        self.primary = primary_injected
        self.replicas = replicas  # (model, injected, device)
        self.n = 1 + len(replicas)
        for _, inj, _ in replicas:
            if set(inj) != set(primary_injected):
                raise ValueError("Replica LoRA layout differs from the primary")
        self._pool = ThreadPoolExecutor(max_workers=self.n)

    @torch.no_grad()
    def sync_weights(self) -> None:
        """Broadcast the primary LoRA weights (and alpha) to every replica."""
        for _, inj, _ in self.replicas:
            for name, src in self.primary.items():
                dst = inj[name]
                dst.lora_down.weight.copy_(src.lora_down.weight, non_blocking=True)
                dst.lora_up.weight.copy_(src.lora_up.weight, non_blocking=True)
                dst.alpha, dst.scale = src.alpha, src.scale

    @torch.no_grad()
    def reduce_grads(self) -> None:
        """Sum replica LoRA grads into the primary (losses are already scaled by 1/n) and clear them."""
        for _, inj, _ in self.replicas:
            for name, src in self.primary.items():
                rep = inj[name]
                for ps, pr in ((src.lora_down.weight, rep.lora_down.weight), (src.lora_up.weight, rep.lora_up.weight)):
                    if pr.grad is None:
                        continue
                    g = pr.grad.to(ps.device, non_blocking=True)
                    if ps.grad is None:
                        ps.grad = g.clone()
                    else:
                        ps.grad.add_(g)
                    pr.grad = None
        for dev in {d for _, _, d in self.replicas}:
            if str(dev).startswith("cuda"):
                torch.cuda.synchronize(dev)

    def run(self, fn, jobs: list):
        """Run fn(job) for every job concurrently (one thread per device). Returns results in order."""
        return list(self._pool.map(fn, jobs))

    def train(self, mode: bool = True) -> None:
        for m, _, _ in self.replicas:
            m.train(mode)


# ---------------------------------------------------------------- model split


def _parse_range_list(spec: str, n: int) -> list[int]:
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(max(0, lo), min(n - 1, hi) + 1))
        else:
            i = int(part)
            if 0 <= i < n:
                out.add(i)
    return sorted(out)


def parse_split_plan(plan: str, n_blocks: int) -> list[str]:
    """"cuda:0=0-25;cuda:1=26-41" -> device per block. Every block must be assigned exactly once."""
    assignment: list[str | None] = [None] * n_blocks
    for entry in plan.replace("\n", ";").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Split plan entry '{entry}' must look like cuda:1=21-41")
        dev, rng = entry.split("=", 1)
        dev = dev.strip()
        for i in _parse_range_list(rng, n_blocks):
            if assignment[i] is not None:
                raise ValueError(f"Block {i} assigned twice in split plan")
            assignment[i] = dev
    missing = [i for i, d in enumerate(assignment) if d is None]
    if missing:
        raise ValueError(f"Split plan leaves blocks unassigned: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    return assignment  # type: ignore[return-value]


def block_costs(n_blocks: int, trained: list[int] | None) -> list[float]:
    """Relative per-block cost: forward only before the first trained block, forward+backward after."""
    first = 0 if not trained else min(trained)
    if trained is not None and len(trained) == 0:
        first = n_blocks
    return [1.0 if i < first else 4.0 for i in range(n_blocks)]


def balanced_assignment(n_blocks: int, devices: list[str], trained: list[int] | None = None) -> list[str]:
    """Contiguous ranges per device with roughly equal estimated cost."""
    costs = block_costs(n_blocks, trained)
    total = sum(costs)
    k = len(devices)
    assignment, acc, d = [], 0.0, 0
    for i in range(n_blocks):
        target = total * (d + 1) / k
        remaining_devices = k - d - 1
        remaining_blocks = n_blocks - i
        # move to the next device when this one has reached its share (but leave at least one block per device)
        if d < k - 1 and acc >= target - costs[i] / 2 and remaining_blocks > remaining_devices:
            d += 1
        assignment.append(devices[d])
        acc += costs[i]
    return assignment


def apply_assignment(model: nn.Module, assignment: list[str]) -> list[tuple[str, str]]:
    """Move blocks to their devices; the final norm/head follow the last block. Returns [(device, blocks)]."""
    from .lora import format_blocks

    for i, dev in enumerate(assignment):
        model.layers[i].to(dev)
    last_dev = assignment[-1]
    model.final_norm_video.to(last_dev)
    model.final_linear_video.to(last_dev)
    model.split_assignment = list(assignment)
    model.split_devices = list(dict.fromkeys(assignment))
    summary = [(dev, format_blocks([i for i, d in enumerate(assignment) if d == dev])) for dev in model.split_devices]
    log.info("Model split: %s", ", ".join(f"{d}: blocks {b}" for d, b in summary))
    return summary


def split_model(model: nn.Module, devices: list[str], trained: list[int] | None = None, plan: str = ""):
    """Split by a manual plan when given, else auto-balance by estimated cost."""
    n = len(model.layers)
    assignment = parse_split_plan(plan, n) if plan.strip() else balanced_assignment(n, devices, trained)
    return apply_assignment(model, assignment)


def gpu_memory_summary(devices: list[str]) -> str:
    parts = []
    for d in devices:
        if str(d).startswith("cuda") and torch.cuda.is_available():
            parts.append(f"{d} {torch.cuda.memory_allocated(torch.device(d)) / 1e9:.2f} GB")
    return ", ".join(parts)
