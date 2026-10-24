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

import concurrent.futures
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

    def zero_grads(self) -> None:
        """Drop the replicas' LoRA grads without reducing them (a failed step's partial gradients)."""
        for _, inj, _ in self.replicas:
            for mod in inj.values():
                mod.lora_down.weight.grad = mod.lora_up.weight.grad = None

    def run(self, fn, jobs: list, serial: bool = False):
        """Run fn(job) for every job concurrently (one thread per device), or one after another with serial=True.
        Returns results in order.

        serial is for steps that may compile (torch.compile): while one thread traces a graph, dynamo / FX patch
        nn.Module.__call__ process-wide, and the other thread's forward then fails with "Detected that you are
        using FX to symbolically trace a dynamo-optimized function"."""
        if serial:
            return [fn(job) for job in jobs]
        futures = [self._pool.submit(fn, job) for job in jobs]
        concurrent.futures.wait(futures)  # all done before an error surfaces: no replica is still mid-backward
        return [f.result() for f in futures]

    def train(self, mode: bool = True) -> None:
        for m, _, _ in self.replicas:
            m.train(mode)


def split_batches(times: list[float], total: int) -> list[int]:
    """Batches per GPU for `total` equal batches, given each GPU's seconds per batch: each next batch goes to the GPU
    that would finish it first (optimal for identical batches). Every GPU gets at least one."""
    counts = [1] * len(times)
    for _ in range(max(0, total - len(times))):
        i = min(range(len(times)), key=lambda k: (counts[k] + 1) * times[k])
        counts[i] += 1
    return counts


def step_seconds(times: list[float], counts: list[int]) -> float:
    """Wall time of a step: the GPUs run their batches side by side, the slowest one sets the pace."""
    return max(t * c for t, c in zip(times, counts))


def choose_counts(times: list[float], base: int, mode: str, min_gain: float = 0.05) -> list[int]:
    """Batches per GPU for one optimizer step (dp_balance).

    off   `base` each (lockstep, as before).
    auto  the same total (base x GPUs) split by speed: same images per step, only the waiting goes.
    grow  the slowest GPU runs `base` (or 2 x base) batches and the others fill its time: more images per step.
    A split replaces lockstep only when it is at least `min_gain` faster per image."""
    n = len(times)
    lockstep = [base] * n
    if mode == "off" or n < 2 or any(t <= 0 for t in times):
        return lockstep

    def per_image(c):
        return step_seconds(times, c) / sum(c)

    if mode == "auto":
        options = [split_batches(times, base * n)]
    elif mode == "grow":  # the faster GPUs fill the slowest one's time, rounded down or up
        import itertools
        import math

        slow = max(times)
        options = []
        for k in (1, 2):
            fill = [k * base * slow / t for t in times]
            choices = [sorted({max(1, math.floor(f + 1e-9)), max(1, math.ceil(f - 1e-9))}) for f in fill]
            options += [list(c) for c in itertools.product(*choices)]
    else:
        raise ValueError(f"dp_balance must be off, auto or grow, got {mode!r}")
    best = min(options, key=per_image)
    return best if per_image(best) <= per_image(lockstep) * (1 - min_gain) else lockstep


def job_order(counts: list[int]) -> list[int]:
    """Which GPU each drawn batch goes to, interleaved: [3, 2] -> [0, 1, 0, 1, 0]. Equal counts give the lockstep
    order (one batch per GPU in turn), so the random draws stay those of a lockstep run."""
    order, left = [], list(counts)
    while any(left):
        for i, c in enumerate(left):
            if c:
                order.append(i)
                left[i] -= 1
    return order


class SpeedMeter:
    """Seconds per batch for each GPU, as a rolling median over recent batches, normalized by image tokens (buckets
    of different size do not look like a speed change)."""

    def __init__(self, n: int, window: int = 12):
        self.samples: list[list[float]] = [[] for _ in range(n)]
        self.window = window

    def add(self, replica: int, seconds: float, tokens: int) -> None:
        s = self.samples[replica]
        s.append(seconds / max(1, tokens))
        del s[:-self.window]

    def ready(self, minimum: int = 3) -> bool:
        return all(len(s) >= minimum for s in self.samples)

    def times(self, tokens: int = 4096) -> list[float]:
        """Seconds per batch of `tokens` image tokens on each GPU."""
        import statistics

        return [statistics.median(s) * tokens if s else 0.0 for s in self.samples]

    def reset(self) -> None:
        for s in self.samples:
            s.clear()


class Balancer:
    """dp_balance for a data-parallel run: measures each GPU's seconds per batch and picks the batches per GPU per
    optimizer step. Starts in lockstep, decides once every GPU has 3 timed batches, re-decides every `every` steps
    (other jobs may share the cards) and switches only for a gain above `hysteresis`."""

    def __init__(self, n: int, base: int, mode: str, every: int = 25, hysteresis: float = 0.03):
        self.n, self.base, self.mode = n, base, mode
        self.every, self.hysteresis = every, hysteresis
        self.meter = SpeedMeter(n)
        self.counts = [base] * n
        self.decided_at: int | None = None

    def observe(self, timings: list[tuple[int, float, int]]) -> None:
        """(replica, seconds, image tokens) for every batch of a step (leave out warm-up and compiling steps)."""
        for replica, seconds, tokens in timings:
            self.meter.add(replica, seconds, tokens)

    def update(self, step: int) -> list[int] | None:
        """New batches per GPU when the split should change, else None."""
        if self.mode == "off" or not self.meter.ready():
            return None
        if self.decided_at is not None and step - self.decided_at < self.every:
            return None
        self.decided_at = step
        times = self.meter.times()
        new = choose_counts(times, self.base, self.mode)
        if new == self.counts:
            return None

        def per_image(c):
            return step_seconds(times, c) / sum(c)

        if per_image(new) > per_image(self.counts) * (1 - self.hysteresis):
            return None
        self.counts = new
        return new

    def per_image(self, counts: list[int] | None = None) -> float | None:
        if not self.meter.ready(1):
            return None
        times = self.meter.times()
        c = counts or self.counts
        return step_seconds(times, c) / sum(c)

    def reset(self) -> None:
        """Back to lockstep with fresh measurements (a new training resolution changes the speeds)."""
        self.meter.reset()
        self.counts = [self.base] * self.n
        self.decided_at = None


def stance_available() -> bool:
    """True when this torch has torch.compiler.set_stance (checked without touching the current stance)."""
    return getattr(getattr(torch, "compiler", None), "set_stance", None) is not None


def no_recompile_stance():
    """Context manager under which compiled code never compiles again (a guard miss runs that frame eagerly).

    Parallel data-parallel steps run under it, so a late recompile (a new sequence length, say) can never trace
    while another replica thread is executing. Use it only directly in a `with` statement: set_stance changes the
    process-wide stance as soon as it is called and restores the previous one when the `with` block exits.
    Returns None when this torch has no set_stance."""
    if not stance_available():
        return None
    try:
        return torch.compiler.set_stance("eager_on_recompile")
    except Exception:  # noqa: BLE001
        return None


def compile_count() -> int:
    """Number of frames dynamo has compiled so far in this process (0 without torch.compile)."""
    try:
        from torch._dynamo.utils import counters

        return int(counters["frames"]["total"])
    except Exception:  # noqa: BLE001
        return 0


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


def apply_assignment(model: nn.Module, assignment: list[str], primary: str | None = None) -> list[tuple[str, str]]:
    """Move blocks to their devices, the final norm/head after the last block, and everything else (embedders, text
    stages) to the primary device, so the model never has to fit one GPU as a whole (it may come from the CPU).
    Returns [(device, blocks)]."""
    from .lora import format_blocks

    primary = primary or getattr(model, "split_primary", None) or assignment[0]
    tail = model.split_tail_modules() if hasattr(model, "split_tail_modules") else [
        model.final_norm_video, model.final_linear_video]  # MOSS-Image
    placed = {id(model.layers), *(id(m) for m in tail)}
    for child in model.children():
        if id(child) not in placed:
            child.to(primary)
    for p in model.parameters(recurse=False):
        p.data = p.data.to(primary)
    for name, b in list(model.named_buffers(recurse=False)):
        setattr(model, name, b.to(primary))
    for i, dev in enumerate(assignment):
        model.layers[i].to(dev)
    for m in tail:
        m.to(assignment[-1])
    model.split_assignment = list(assignment)
    model.split_primary = primary
    model.split_devices = list(dict.fromkeys(assignment))
    summary = [(dev, format_blocks([i for i, d in enumerate(assignment) if d == dev])) for dev in model.split_devices]
    log.info("Model split: %s", ", ".join(f"{d}: blocks {b}" for d, b in summary))
    return summary


def split_model(model: nn.Module, devices: list[str], trained: list[int] | None = None, plan: str = ""):
    """Split by a manual plan when given, else auto-balance by estimated cost. devices[0] is the primary."""
    n = len(model.layers)
    assignment = parse_split_plan(plan, n) if plan.strip() else balanced_assignment(n, devices, trained)
    return apply_assignment(model, assignment, primary=devices[0])


def gpu_memory_summary(devices: list[str]) -> str:
    parts = []
    for d in devices:
        if str(d).startswith("cuda") and torch.cuda.is_available():
            parts.append(f"{d} {torch.cuda.memory_allocated(torch.device(d)) / 1e9:.2f} GB")
    return ", ".join(parts)


# ---------------------------------------------------------------- splitting side jobs across GPUs


def norm_device(device) -> str:
    """Canonical device string: "cuda" -> "cuda:<current>", torch.device -> "cuda:1", "cpu" stays."""
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0)
    return str(dev)


def _device_ctx(device):
    """torch.cuda.device(device) for a CUDA device (so triton / custom kernels launch on it), else a no-op."""
    import contextlib

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(dev)
    return contextlib.nullcontext()


def run_queue(workers: list, jobs: list, fn, *, serial: bool = False, on_done=None) -> list:
    """Run fn(worker, job) for every job and return the results in job order.

    workers  one entry per device (any object with a .device attribute, or a (obj, device) tuple). Each worker
             gets its own thread that pulls the next unstarted job, so a faster GPU simply takes more of them.
    serial   run in the calling thread, job i on worker i % n (used while torch.compile may still trace graphs:
             two threads tracing at once crash dynamo).
    on_done  optional callback(index, job, result), always called in the calling thread as results arrive.
    The first exception in any worker stops the others from starting new jobs and is re-raised here."""
    import queue
    import threading

    def device_of(w):
        return w[1] if isinstance(w, tuple) else getattr(w, "device", "cpu")

    results = [None] * len(jobs)
    if not jobs:
        return results
    if serial or len(workers) <= 1:
        for i, job in enumerate(jobs):
            w = workers[i % len(workers)]
            with _device_ctx(device_of(w)):
                results[i] = fn(w, job)
            if on_done:
                on_done(i, job, results[i])
        return results

    todo: "queue.SimpleQueue" = queue.SimpleQueue()
    for i, job in enumerate(jobs):
        todo.put((i, job))
    done: "queue.SimpleQueue" = queue.SimpleQueue()
    failed = threading.Event()

    def loop(w):
        with _device_ctx(device_of(w)):
            while not failed.is_set():
                try:
                    i, job = todo.get_nowait()
                except queue.Empty:
                    return
                try:
                    done.put((i, fn(w, job), None))
                except BaseException as e:  # noqa: BLE001
                    failed.set()
                    done.put((i, None, e))
                    return

    threads = [threading.Thread(target=loop, args=(w,), daemon=True, name=f"gpu-worker-{k}")
               for k, w in enumerate(workers)]
    for t in threads:
        t.start()
    error = None
    received = 0
    while received < len(jobs):
        try:
            i, res, exc = done.get(timeout=0.1)
        except queue.Empty:
            if not any(t.is_alive() for t in threads) and done.empty():
                break  # a worker failed and the rest stopped before every job started
            continue
        received += 1
        if exc is not None:
            error = error or exc
        elif error is None:
            results[i] = res
            if on_done:
                on_done(i, jobs[i], res)
    for t in threads:
        t.join()
    if error is not None:
        raise error
    return results


def clone_to(module: nn.Module, device) -> nn.Module:
    """A deep copy of a module on another device (meant for small ones such as the VAE)."""
    import copy

    return copy.deepcopy(module).to(device)
