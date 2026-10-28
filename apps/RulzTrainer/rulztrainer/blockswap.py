"""Block swap: the frozen weights of some DiT blocks live in system RAM and stream to the GPU when the block runs.

For a card the model does not fit (after musubi-tuner's block swap, H2D-only mode). The LoRA (or LoKr) adapters, the
rest of the model and the activations stay on the GPU; `block_swap` blocks' frozen weights (their quantized or bf16
Linears, norms, biases) are held in page-locked CPU memory. A forward pre-hook copies a block's weights to the GPU
right before it runs and a forward hook lets them go right after, so at most two swapped blocks' weights are on the
GPU at a time: the one running and the next one, copied in on a side CUDA stream while the current block computes.
GPU work is asynchronous, so the CPU could queue every swapped block's copy long before the GPU runs them (and a
copy's memory is reusable only once the GPU is done with it): an event is recorded as each swapped block finishes,
and before streaming in another block the CPU waits for all but the last `max_inflight` of them.

Frozen weights never change, so nothing is ever copied back (host to device only). Gradient checkpointing makes it
work in training: a checkpointed block keeps no weights for its backward; the recomputation in the backward runs the
block again, the hook streams its weights in again, and autograd releases them when that block's backward is done.
A recomputation can stop early (non-reentrant checkpointing stops as soon as it has the tensors the backward
needs), so a block's forward hook may never run then: each pre-hook also releases any other swapped block still
holding its GPU copy. The swapped blocks are the first `block_swap` blocks of the DiT (they always run checkpointed; the blocks that keep
their activations, rulztrainer.checkpointing, are the last ones). The prefetch follows the direction the blocks run in:
forward up, the backward's recomputation down.

Costs: the swapped weights in RAM (pinned), and their transfer twice per step (forward and recomputation), mostly
hidden behind the blocks' compute when the PCIe link keeps up. Not with model_split, data parallel, torch.compile or a
full fine-tune."""
from __future__ import annotations

import collections
import logging

import torch
from torch import nn

log = logging.getLogger("rulztrainer")


def _frozen_tensors(block: nn.Module):
    """(module, name, is_buffer, tensor) of every frozen tensor of a block: parameters without requires_grad, and
    buffers (a quantized Linear's int8 / fp4 data and scales). Trainable adapter weights stay where they are."""
    for mod in block.modules():
        for name, p in mod.named_parameters(recurse=False):
            if p is not None and not p.requires_grad and p.numel():
                yield mod, name, False, p
        for name, b in mod.named_buffers(recurse=False):
            if b is not None and b.numel():
                yield mod, name, True, b


def _set(mod: nn.Module, name: str, is_buffer: bool, t: torch.Tensor) -> None:
    if is_buffer:
        mod._buffers[name] = t
    else:
        getattr(mod, name).data = t


class BlockSwap:
    """Streams the frozen weights of `model.layers[i]` for i in `indices` from pinned CPU memory to `device`."""

    def __init__(self, model: nn.Module, indices, device, prefetch: bool = True, pin: bool = True,
                 max_inflight: int = 2):
        self.model = model
        self.device = torch.device(device)
        self.indices = sorted(int(i) for i in indices)
        self.cuda = self.device.type == "cuda"
        self.prefetch = prefetch and self.cuda
        self.stream = torch.cuda.Stream(self.device) if self.prefetch else None
        self.entries: dict[int, list] = {}  # block -> [(module, name, is_buffer, cpu tensor)]
        self.ready: dict[int, tuple] = {}  # block -> (gpu tensors, event) copied in ahead
        self.last: int | None = None
        self.installed: set[int] = set()  # swapped blocks whose modules hold their GPU copies right now
        self.max_inflight = max(1, int(max_inflight))
        self.inflight: collections.deque = collections.deque()  # events of swapped blocks the GPU may still be running
        self.bytes = 0
        self.hooks = []
        for i in self.indices:
            block = model.layers[i]
            rows = []
            for mod, name, is_buf, t in list(_frozen_tensors(block)):
                cpu = t.detach().to("cpu")
                if pin and self.cuda:
                    cpu = cpu.pin_memory()
                _set(mod, name, is_buf, cpu)
                rows.append((mod, name, is_buf, cpu))
                self.bytes += cpu.numel() * cpu.element_size()
            self.entries[i] = rows
            block._rz_swap_index = i
            self.hooks.append(block.register_forward_pre_hook(self._pre))
            self.hooks.append(block.register_forward_hook(self._post))
        self._place_rest()
        if self.cuda:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _place_rest(self) -> None:
        """Everything but the swapped blocks' frozen weights on the device (adapters, other blocks, embeddings)."""
        swapped = {(id(mod), name) for rows in self.entries.values() for mod, name, _, _ in rows}
        for mod in self.model.modules():
            for name, p in mod.named_parameters(recurse=False):
                if p is not None and (id(mod), name) not in swapped and p.device != self.device:
                    p.data = p.data.to(self.device)
            for name, b in mod.named_buffers(recurse=False):
                if b is not None and (id(mod), name) not in swapped and b.device != self.device:
                    mod._buffers[name] = b.to(self.device)

    @torch.no_grad()
    def reoffload(self) -> None:
        """After something moved the whole model (`model.to(device)`): the swapped blocks' weights back to RAM."""
        for i in self.entries:
            self._release(i)
        self._place_rest()
        if self.cuda:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ streaming
    def _throttle(self) -> None:
        """Let the GPU catch up to within max_inflight swapped blocks before more weights are copied in."""
        while len(self.inflight) > self.max_inflight:
            self.inflight.popleft().synchronize()

    @torch.no_grad()
    def _copy(self, i: int) -> tuple:
        rows = self.entries[i]
        if self.cuda:
            self._throttle()
        if not self.prefetch:
            return [cpu.to(self.device, non_blocking=self.cuda) for _, _, _, cpu in rows], None
        with torch.cuda.stream(self.stream):
            tensors = [cpu.to(self.device, non_blocking=True) for _, _, _, cpu in rows]
            event = torch.cuda.Event()
            event.record(self.stream)
        return tensors, event

    def _next(self, i: int) -> int | None:
        """The swapped block that runs after block i: the next one up in a forward, down in the recomputation."""
        k = self.indices.index(i)
        down = self.last is not None and i < self.last
        j = k - 1 if down else k + 1
        return self.indices[j] if 0 <= j < len(self.indices) else None

    def _release(self, i: int) -> None:
        for mod, name, is_buf, cpu in self.entries[i]:  # autograd keeps what a backward still needs
            _set(mod, name, is_buf, cpu)
        self.installed.discard(i)

    def _pre(self, block, args):
        i = block._rz_swap_index
        for j in list(self.installed):  # a recomputation that stopped early never ran its forward hook
            if j != i:
                self._release(j)
        tensors, event = self.ready.pop(i, None) or self._copy(i)
        if event is not None:
            cur = torch.cuda.current_stream(self.device)
            cur.wait_event(event)
            for t in tensors:  # allocated on the copy stream, used on this one
                t.record_stream(cur)
        for (mod, name, is_buf, _), t in zip(self.entries[i], tensors):
            _set(mod, name, is_buf, t)
        self.installed.add(i)
        nxt = self._next(i)
        self.last = i
        if self.prefetch and nxt is not None and nxt not in self.ready:
            self.ready[nxt] = self._copy(nxt)

    def _post(self, block, args, output):
        self._release(block._rz_swap_index)
        if self.cuda:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))
            self.inflight.append(ev)

    def reset(self) -> None:
        """Forget prefetched copies (between a step's passes and a render, so nothing stale is waited on)."""
        self.ready.clear()
        self.last = None
        self.inflight.clear()

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []
