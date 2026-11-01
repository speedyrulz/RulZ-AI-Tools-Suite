"""Full fine-tune of the model's own Linear layers, by rotation, and its extraction into a LoRA (after Fizgig's).

A LoRA confines every update to a few directions, so concepts compete for them and pull pose, framing and lighting
along with the likeness. A fine-tune moves the weights themselves. Fizgig measured, on Krea 2: a LoRA *extracted* from
a fine-tune beat one *trained* at the same rank on the same dataset, and rank 64 was indistinguishable from the full
checkpoint (about 0.5 GB against 26 GB). The full-rank phase does the work; extraction is one SVD per layer.

How it fits one card. The model stays as it was loaded (its quantized trunk: int8 or nvfp4, or bf16). The trained
Linears (the same targets as a LoRA's: lora_targets / lora_blocks / train_text_embedder) are grouped into
*components*, one Linear name across every block (every block's attention query, say); the components are packed
into *windows* of what the card holds. One window at a time is swapped up to fp32 Parameters (TunedLinear: fp32
weights, grads and optimizer state; forward in bf16), trained for finetune_rotate_epochs epochs, and swapped back:
its update is kept as a bf16 *delta* on the CPU (the delta's own precision, not the weight's, so a small update is
never rounded away) and the trunk's layer is re-quantized from original + delta. A full *cycle* of windows trains
every component once. Each trained layer's original quantized weight is also kept on the CPU (one byte per weight
for int8), so the delta is always against the exact base the model was loaded with, and the trunk never accumulates
re-quantization error. RAM: about 2 bytes per trained weight for the deltas plus the original quantized copy (MOSS
with every target: 10 + 5 GB; Krea 2: 24 + 12 GB).

Saves are LoRA extractions of the deltas (finetune_extract_ranks; alpha = rank, the trainer's own file format) at
cycle boundaries, so every saved file has each component trained equally; finetune_save_delta also writes the deltas
themselves (<job>_finetune.safetensors, bf16: the fine-tune's state, to continue from with finetune_resume or to
extract at other ranks with `python -m rulztrainer.extract_finetune`). Samples and probes render the fine-tuned
model directly (its sampling-only turbo LoRA attaches as usual).

Weight decay pulls toward the original weights, not toward zero (finetune_anchor, decoupled like AdamW's: after each
step w -= lr x finetune_anchor x (w - original)): the bounded drift of "Preserve and Personalize" (ICLR 2026), which
measured a full fine-tune held near its base this way above an unheld one in both likeness and prompt-following. The
optimizer's own weight decay, which would shrink the model's weights themselves toward zero, is off. The original
weights stay on the GPU in their own format while their window trains (1 byte per weight for int8, 2 for bf16).

Not in a fine-tune: a race, data parallel, torch.compile (the layers swap), the weight average, training states,
resume_from, the region prior (no base model to compare with). Use a much lower learning rate than a LoRA's: Fizgig's
measured rates are 1e-5 for Krea 2 (1e-4 trains but overcooks), 3e-5 for larger-weight models; the WebUI's switch sets
1e-5 with Adafactor."""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from . import quant as Q

log = logging.getLogger("rulztrainer")

BYTES_PER_PARAM = {"adafactor": 10, "adamw8bit": 12, "adamw": 18, "automagic3": 18,  # fp32 w + grad + state + bf16
                   "automagic4": 12, "keel": 12}  # 1 byte of sign history per weight; the step works in row slabs
ANCHOR_BYTES = 2  # the original weights kept for finetune_anchor (int8 / nvfp4 need less; bf16 needs this)
DELTA_SUFFIX = "_finetune.safetensors"


class TunedLinear(nn.Module):
    """A Linear being fine-tuned: fp32 weight Parameter, the forward in the activations' dtype, a frozen bias."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, compute_dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone().float().contiguous())  # its own storage, never the base's
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(compute_dtype))
        else:
            self.bias = None
        self.out_features, self.in_features = self.weight.shape
        self.compute_dtype = compute_dtype

    def forward(self, x):
        return nn.functional.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, fine-tuned (fp32)"


@dataclass
class Slot:
    """One fine-tuned Linear: where it sits, its frozen base (on the CPU) and its delta."""
    name: str
    comp: str  # component key: the name with its block index wildcarded (layers.*.proj.q)
    block: int | None
    numel: int
    shape: tuple
    fmt: str  # int8 | int8_convrot | nvfp4 | bf16 (an unquantized layer)
    base: object  # QuantWeight (CPU) or a bf16 tensor (CPU)
    compute_dtype: torch.dtype
    flags: dict = field(default_factory=dict)  # a QuantLinear's matmul settings, put back on re-quantization
    delta: torch.Tensor | None = None  # bf16 CPU [out, in]; None = untrained (zero)
    live: TunedLinear | None = None  # while its window trains
    anchor: object = None  # its original weight on the GPU (own format) while its window trains, for finetune_anchor


def component_key(name: str) -> tuple[str, int | None]:
    """("layers.*.proj.q", 7) for "layers.7.proj.q"; a name without a block index is its own component."""
    m = re.search(r"\.(\d+)\.", name)
    if not m:
        return name, None
    return name[: m.start()] + ".*." + name[m.end():], int(m.group(1))


def _set_module(model: nn.Module, name: str, new: nn.Module) -> None:
    """Put `new` where `name` is, inside a LoRALinear wrapper (a sampling LoRA's host) or in the parent."""
    from .lora import LoRALinear

    cur = model.get_submodule(name)
    if isinstance(cur, LoRALinear):
        cur.base = new
        return
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


def _inner(model: nn.Module, name: str) -> nn.Module:
    from .lora import LoRALinear

    m = model.get_submodule(name)
    return m.base if isinstance(m, LoRALinear) else m


def plan_windows(slots: list[Slot], budget: int, bytes_per_param: int) -> list[list[str]]:
    """The windows: lists of slot names. Components in model order are packed into the fewest windows whose weights
    (bytes_per_param each: fp32 weight, grad, optimizer state) fit `budget` bytes; a component larger than the
    budget is cut into block ranges. Raises ValueError when a single layer cannot fit."""
    comps: dict[str, list[Slot]] = {}
    for s in slots:
        comps.setdefault(s.comp, []).append(s)
    windows: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for comp, group in comps.items():
        group = sorted(group, key=lambda s: (s.block if s.block is not None else -1, s.name))
        size = sum(s.numel for s in group) * bytes_per_param
        if size <= budget:
            if cur and cur_bytes + size > budget:
                windows.append(cur)
                cur, cur_bytes = [], 0
            cur += [s.name for s in group]
            cur_bytes += size
            continue
        if cur:
            windows.append(cur)
            cur, cur_bytes = [], 0
        chunk: list[str] = []
        chunk_bytes = 0
        for s in group:  # cut the component into block ranges that fit
            one = s.numel * bytes_per_param
            if one > budget:
                raise ValueError(f"{s.name} alone needs {one / 2**30:.1f} GiB in a window, more than the "
                                 f"{budget / 2**30:.1f} GiB the card has for one: free VRAM, lower the training "
                                 "resolution, use adafactor, or leave this target out of the fine-tune")
            if chunk and chunk_bytes + one > budget:
                windows.append(chunk)
                chunk, chunk_bytes = [], 0
            chunk.append(s.name)
            chunk_bytes += one
        if chunk:
            windows.append(chunk)
    if cur:
        windows.append(cur)
    return windows


def describe_window(names: list[str]) -> str:
    """"layers.*.proj.q (42 blocks)" / "layers.*.proj.ffn_up_gate blocks 11-21" for a window's names."""
    comps: dict[str, list[int | None]] = {}
    for n in names:
        c, b = component_key(n)
        comps.setdefault(c, []).append(b)
    parts = []
    for c, blocks in comps.items():
        bs = [b for b in blocks if b is not None]
        if not bs:
            parts.append(c)
        elif len(bs) == 1:
            parts.append(f"{c} block {bs[0]}")
        else:
            parts.append(f"{c} blocks {min(bs)}-{max(bs)}" + (f" ({len(bs)})" if max(bs) - min(bs) + 1 != len(bs)
                                                               else ""))
    return "; ".join(parts)


class FineTuner:
    """The fine-tune of one model: its slots, windows, the active window's parameters, deltas and extraction."""

    def __init__(self, model: nn.Module, names: list[str], device, log_=None):
        self.model, self.device = model, device
        self.log = log_ or log
        self.slots: dict[str, Slot] = {}
        for name in names:
            inner = _inner(model, name)
            if isinstance(inner, Q.QuantLinear):
                qw = inner.qw
                base = Q.QuantWeight(qw.fmt, qw.qdata.to("cpu"), qw.scale.to("cpu"),
                                     None if qw.scale2 is None else qw.scale2.to("cpu"), qw.orig_shape, qw.group)
                flags = {k: getattr(inner, k) for k in ("int8_matmul", "int8_backward", "nvfp4_matmul",
                                                        "nvfp4_backward")}
                shape, fmt, cdt = tuple(qw.orig_shape), qw.fmt, inner.compute_dtype
            elif isinstance(inner, nn.Linear):
                base = inner.weight.detach().to("cpu").clone()  # its own dtype (bf16 in a real model): exact
                flags, shape, fmt, cdt = {}, tuple(inner.weight.shape), "bf16", inner.weight.dtype
            else:
                raise ValueError(f"{name}: cannot fine-tune a {type(inner).__name__}")
            comp, block = component_key(name)
            self.slots[name] = Slot(name, comp, block, shape[0] * shape[1], shape, fmt, base, cdt, flags)
        self.windows: list[list[str]] = []
        self.active: int | None = None
        self.cycles = 0  # completed cycles
        self.anchored = False  # keep the original weights on the GPU for pull_to_base

    # ------------------------------------------------------------ sizes and planning
    @property
    def n_params(self) -> int:
        return sum(s.numel for s in self.slots.values())

    def plan(self, budget: int, bytes_per_param: int) -> list[list[str]]:
        self.windows = plan_windows(list(self.slots.values()), budget, bytes_per_param)
        return self.windows

    def window_params(self, i: int) -> int:
        return sum(self.slots[n].numel for n in self.windows[i])

    # ------------------------------------------------------------ the base and the delta of a slot
    def base_weight(self, slot: Slot, device=None, dtype=torch.float32) -> torch.Tensor:
        dev = device or self.device
        if slot.fmt == "bf16":
            return slot.base.to(dev, dtype, copy=True)
        qw = slot.base
        qw = Q.QuantWeight(qw.fmt, qw.qdata.to(dev), qw.scale.to(dev), None if qw.scale2 is None else
                           qw.scale2.to(dev), qw.orig_shape, qw.group)
        return qw.dequantize(torch.float32).to(dtype)

    def delta_of(self, slot: Slot, device=None) -> torch.Tensor:
        """The slot's update as fp32 on `device` (the live window's weights against its base while it trains)."""
        dev = device or self.device
        if slot.live is not None:
            return slot.live.weight.detach().to(dev, torch.float32) - self.base_weight(slot, dev)
        if slot.delta is None:
            return torch.zeros(slot.shape, device=dev, dtype=torch.float32)
        return slot.delta.to(dev, torch.float32)

    @torch.no_grad()
    def _frozen_module(self, slot: Slot, weight: torch.Tensor, bias) -> nn.Module:
        """The trunk's layer for a slot that is not training: re-quantized from original + delta."""
        if slot.fmt == "bf16":
            lin = nn.Linear(slot.shape[1], slot.shape[0], bias=bias is not None)
            lin.weight = nn.Parameter(weight.to(self.device, slot.compute_dtype), requires_grad=False)
            if bias is not None:
                lin.bias = nn.Parameter(bias.to(self.device, slot.compute_dtype), requires_grad=False)
            return lin
        qw = Q.quantize_weight(weight.to(self.device), slot.fmt, slot.base.group)
        ql = Q.QuantLinear(qw, None if bias is None else bias.to(self.device), slot.compute_dtype)
        for k, v in slot.flags.items():
            setattr(ql, k, v)
        return ql.to(self.device)

    # ------------------------------------------------------------ windows in and out
    @torch.no_grad()
    def activate(self, i: int) -> list[nn.Parameter]:
        """Swap window i up to fp32 Parameters; returns them."""
        if self.active is not None:
            raise RuntimeError("a window is already active")
        params = []
        for name in self.windows[i]:
            slot = self.slots[name]
            inner = _inner(self.model, name)
            bias = getattr(inner, "bias", None)
            w = self.base_weight(slot)
            if slot.delta is not None:
                w += slot.delta.to(self.device, torch.float32)
            slot.live = TunedLinear(w, bias, slot.compute_dtype).to(self.device)
            if self.anchored:
                slot.anchor = self._to_device(slot)
            _set_module(self.model, name, slot.live)
            params.append(slot.live.weight)
            del inner
        self.active = i
        return params

    @torch.no_grad()
    def deactivate(self) -> None:
        """Write the active window back: its delta to the CPU, the trunk re-quantized from original + delta."""
        if self.active is None:
            return
        for name in self.windows[self.active]:
            slot = self.slots[name]
            live = slot.live
            w = live.weight.detach().float()
            base = self.base_weight(slot)
            slot.delta = (w - base).to("cpu", torch.bfloat16)
            merged = base + slot.delta.to(self.device, torch.float32)
            _set_module(self.model, name, self._frozen_module(slot, merged, live.bias))
            slot.live = slot.anchor = None
            del live, w, base, merged
        if self.active == len(self.windows) - 1:
            self.cycles += 1
        self.active = None
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def _to_device(self, slot: Slot):
        """The slot's original weight on the device in its own format (bf16, or the quantized data and scales)."""
        if slot.fmt == "bf16":
            return slot.base.to(self.device)
        qw = slot.base
        return Q.QuantWeight(qw.fmt, qw.qdata.to(self.device), qw.scale.to(self.device),
                             None if qw.scale2 is None else qw.scale2.to(self.device), qw.orig_shape, qw.group)

    @torch.no_grad()
    def pull_to_base(self, rate: float) -> None:
        """Decoupled decay toward the original weights: w -= rate * (w - original) for the active window (rate =
        lr x finetune_anchor). The original is rebuilt exactly as the window's weights were (base_weight), so an
        untrained weight stays exactly where it is."""
        if self.active is None or rate <= 0:
            return
        for name in self.windows[self.active]:
            slot = self.slots[name]
            if slot.live is None or slot.anchor is None:
                continue
            base = slot.anchor.float() if slot.fmt == "bf16" else slot.anchor.dequantize(torch.float32)
            w = slot.live.weight
            w.sub_(w.sub(base).mul_(rate))
            del base

    @property
    def at_cycle_start(self) -> bool:
        return self.active is None or self.active == 0

    # ------------------------------------------------------------ files
    def save_delta(self, path, metadata: dict | None = None) -> Path:
        """The fine-tune's state: every trained slot's bf16 delta, keyed by module name."""
        state = {}
        for name, slot in self.slots.items():
            d = self.delta_of(slot) if slot.live is not None else slot.delta
            if d is None or not bool(d.any()):  # untouched layers are left out
                continue
            state[name + ".delta"] = d.to("cpu", torch.bfloat16).contiguous()
        meta = {"rulztrainer_finetune": "deltas", "format": "pt"}
        meta.update({k: (v if isinstance(v, str) else json.dumps(v)) for k, v in (metadata or {}).items()})
        save_file(state, str(path), metadata=meta)
        return Path(path)

    def load_delta(self, path) -> int:
        """Continue from a delta file: the deltas of the slots it has, and the trunk re-quantized with them."""
        state = load_file(str(path))
        n = 0
        with torch.no_grad():
            for key, d in state.items():
                name = key[: -len(".delta")] if key.endswith(".delta") else key
                slot = self.slots.get(name)
                if slot is None or tuple(d.shape) != slot.shape:
                    continue
                slot.delta = d.to("cpu", torch.bfloat16)
                inner = _inner(self.model, name)
                merged = self.base_weight(slot) + slot.delta.to(self.device, torch.float32)
                _set_module(self.model, name, self._frozen_module(slot, merged, getattr(inner, "bias", None)))
                n += 1
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return n

    def extract(self, rank: int, min_norm: float = 1e-6, device=None):
        """A rank-r LoRA of the deltas: {name: stub} for lora.save_lora (lora_down [r, in], lora_up [out, r],
        alpha = rank, so up @ down reproduces the delta's best rank-r approximation as is). Untouched layers are
        left out."""
        out = {}
        for name, slot in self.slots.items():
            if slot.delta is None and slot.live is None:
                continue
            d = self.delta_of(slot, device)
            if float(d.norm()) < min_norm:
                continue
            down, up = lowrank_factors(d, rank)
            parts = {"lora_down": down, "lora_up": up}
            r = down.shape[0]
            out[name] = SimpleNamespace(kind="lora", alpha=float(r), rank=r, factor=8, trainable=True,
                                        adapter_params=lambda parts=parts: parts)
            del d
        return out


def lowrank_factors(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(down [r, in], up [out, r]) with up @ down the best rank-r approximation of delta (the singular values split
    evenly between the two). A randomized SVD for large matrices, an exact one for small."""
    m, n = delta.shape
    r = max(1, min(int(rank), m, n))
    d = delta.float()
    if min(m, n) <= 512 or r >= min(m, n):
        u, s, vh = torch.linalg.svd(d, full_matrices=False)
        u, s, vh = u[:, :r], s[:r], vh[:r]
    else:
        u, s, v = torch.svd_lowrank(d, q=min(r + 8, min(m, n)), niter=4)
        u, s, vh = u[:, :r], s[:r], v[:, :r].T
    sq = torch.sqrt(s)
    return (sq[:, None] * vh).contiguous().cpu(), (u * sq[None, :]).contiguous().cpu()


def plan_budget(free_bytes: int, reserve_gb: float, window_gb: float) -> int:
    """Bytes a window may hold: window_gb when given, else what is free less the activation reserve."""
    if window_gb and window_gb > 0:
        return int(window_gb * 2**30)
    return max(0, int(free_bytes - reserve_gb * 2**30))


def default_reserve_gb(max_mp: float) -> float:
    """Activation / fragmentation room left out of a window: 3 GB plus 3 GB per megapixel of the largest bucket."""
    return 3.0 + 3.0 * max(0.0, float(max_mp))


def read_delta_meta(path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        return dict(f.metadata() or {})


def extract_file(delta_path, out_path, rank: int, fmt: str, metadata: dict | None = None, device="cpu",
                 min_norm: float = 1e-6) -> int:
    """Extract a rank-r LoRA from a saved delta file without the model. Returns the layers written."""
    from .lora import save_lora

    state = load_file(str(delta_path))
    stubs = {}
    for key, d in state.items():
        name = key[: -len(".delta")] if key.endswith(".delta") else key
        d = d.to(device, torch.float32)
        if float(d.norm()) < min_norm:
            continue
        down, up = lowrank_factors(d, rank)
        parts = {"lora_down": down, "lora_up": up}
        r = down.shape[0]
        stubs[name] = SimpleNamespace(kind="lora", alpha=float(r), rank=r, factor=8, trainable=True,
                                      adapter_params=lambda parts=parts: parts)
    if not stubs:
        raise ValueError(f"{delta_path}: no layer moved; nothing to extract")
    meta = {"rulztrainer_extraction": "finetune_delta_svd", "ss_network_dim": str(rank), "ss_network_alpha": str(rank)}
    meta.update(metadata or {})
    save_lora(stubs, out_path, metadata=meta, dtype=torch.bfloat16, fmt=fmt)
    return len(stubs)


def epoch_steps(epoch_images: int, images_per_step: int, epochs: float) -> int:
    """Optimizer steps in `epochs` epochs of an image queue (at least 1)."""
    return max(1, int(math.ceil(epoch_images * max(0.0, float(epochs)) / max(1, images_per_step))))
