"""Activation offloading (experimental, untested in a real run): checkpointed blocks' saved inputs wait in RAM.

With gradient checkpointing a block keeps only its inputs for the backward (the hidden states, about tokens x hidden
x 2 bytes each), and every checkpointed block's inputs are alive from its forward until the backward reaches it: on
MiniMax H3 at 1 MP about 0.8 GB over its 50 blocks, about 3 GB at 4 MP (estimates). With `activation_offload` they are
copied to pinned CPU memory on a side CUDA stream as the forward saves them, their GPU memory is given back once the
copy is done, and in the backward each block's inputs are copied back just before its recomputation, the next
`ahead` ones streaming in while it runs (the backward goes down the blocks). The freed memory lets
checkpoint_keep_blocks keep more blocks or block_swap swap fewer; it costs the same bytes twice over the PCIe link
per step (out in the forward, back in the backward), next to block swap's own traffic.

How: torch's non-reentrant checkpoint saves its inputs (`_NoopSaveInputs`) before it installs its own saved-tensor
hooks for the block's insides, so saved-tensor hooks wrapped around the `checkpoint()` call see exactly the block's
inputs (checkpointing.checkpointed). Only tensors that need a gradient and hold `min_bytes` or more go: the hidden
states, not the rotary tables, masks and timestep vectors every block shares. Unlike torch's save_on_cpu the copy out
does not stop the CPU, and nothing waits in the backward that could have streamed in ahead. (OneTrainer offloads
activations the same way since #1476.)"""
from __future__ import annotations

import logging
import weakref

import torch

log = logging.getLogger("rulztrainer")

MIN_BYTES = 1 << 20  # smaller saved inputs stay on the GPU (not worth a copy each way)


class _Saved:
    """One offloaded tensor: its pinned CPU copy, the event that says the copy is done, and the one packed before it
    (a weak reference: autograd's saved inputs own these, so memory goes as the backward frees them)."""

    __slots__ = ("cpu", "done", "device", "gpu", "loaded", "prev", "used", "__weakref__")

    def __init__(self, cpu, done, device, prev):
        self.cpu, self.done, self.device, self.prev = cpu, done, device, prev
        self.gpu = None  # streamed back ahead of its unpack
        self.loaded = None  # the event that says that copy is done
        self.used = False  # unpacked (its recomputation ran)


class ActivationOffload:
    """Saved-tensor hooks that keep checkpointed blocks' inputs in pinned RAM (see the module docstring). One per model
    copy; CUDA only (`enabled` is False without CUDA, and checkpointed() then skips it)."""

    def __init__(self, min_bytes: int = MIN_BYTES, ahead: int = 2):
        self.enabled = torch.cuda.is_available()
        self.min_bytes, self.ahead = int(min_bytes), max(0, int(ahead))
        self.streams: dict[torch.device, torch.cuda.Stream] = {}
        self.last = None  # weak reference to the last packed _Saved
        self.offloaded = 0  # bytes sent to RAM since the last take_bytes()

    def hooks(self):
        return torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)

    def take_bytes(self) -> int:
        """Bytes offloaded since the last call (the trainer logs one step's)."""
        n, self.offloaded = self.offloaded, 0
        return n

    def _stream(self, device: torch.device) -> torch.cuda.Stream:
        s = self.streams.get(device)
        if s is None:
            s = self.streams[device] = torch.cuda.Stream(device)
        return s

    def _pack(self, t: torch.Tensor):
        if (not t.is_cuda or not t.requires_grad or t.layout != torch.strided
                or t.numel() * t.element_size() < self.min_bytes):
            return t
        device = t.device
        src = t if t.is_contiguous() else t.contiguous()
        cur = torch.cuda.current_stream(device)
        side = self._stream(device)
        ready = torch.cuda.Event()
        ready.record(cur)
        cpu = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
        with torch.cuda.stream(side):
            side.wait_event(ready)
            cpu.copy_(src, non_blocking=True)
            done = torch.cuda.Event()
            done.record(side)
        src.record_stream(side)  # its memory is not handed out again before the copy has read it
        saved = _Saved(cpu, done, device, self.last)
        self.last = weakref.ref(saved)
        self.offloaded += cpu.numel() * cpu.element_size()
        return saved

    def _load(self, saved: _Saved) -> None:
        side = self._stream(saved.device)
        with torch.cuda.stream(side):
            side.wait_event(saved.done)
            saved.gpu = saved.cpu.to(saved.device, non_blocking=True)
            saved.loaded = torch.cuda.Event()
            saved.loaded.record(side)

    def _unpack(self, saved):
        if isinstance(saved, torch.Tensor):
            return saved
        if saved.gpu is None:
            self._load(saved)
        gpu, loaded = saved.gpu, saved.loaded
        saved.gpu = saved.loaded = None
        saved.used = True
        cur = torch.cuda.current_stream(saved.device)
        cur.wait_event(loaded)
        gpu.record_stream(cur)  # made on the side stream, used on this one
        prev, n = saved.prev, 0
        while prev is not None and n < self.ahead:  # the blocks the backward reaches next
            p = prev()
            if p is None:
                break
            if p.gpu is None and not p.used:
                self._load(p)
            prev, n = p.prev, n + 1
        return gpu
