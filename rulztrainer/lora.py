"""Minimal LoRA for Linear layers with ComfyUI-compatible export, plus "extra" LoRAs (e.g. turbo /
step-distill adapters) that are applied only while sampling.

Export key format (accepted by ComfyUI's LoRA loader for any model whose state-dict keys are
"diffusion_model.<module>.weight"):

    diffusion_model.<module>.lora_down.weight   [rank, in]
    diffusion_model.<module>.lora_up.weight     [out, rank]
    diffusion_model.<module>.alpha              scalar

Effective delta = (alpha / rank) * up @ down.
"""
from __future__ import annotations

import json
import math
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file
from torch import nn

COMFY_PREFIX = "diffusion_model."
SAVE_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class LoRALinear(nn.Module):
    """Wraps a Linear (plain or QuantLinear). Trainable LoRA in fp32; optional frozen extras."""

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float = 0.0, trainable: bool = True):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.compute_dtype = getattr(base, "compute_dtype", None) or base.weight.dtype
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.trainable = trainable
        self.lora_down = nn.Linear(self.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, self.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.lora_down.float()
        self.lora_up.float()
        self.lora_down.weight.requires_grad_(trainable)
        self.lora_up.weight.requires_grad_(trainable)
        self.multiplier = 1.0
        # extras: list of (down [r,in], up [out,r], scale) applied only when extras_enabled
        self.extras: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        self.extras_enabled = False

    @property
    def weight(self):  # for code that only inspects .weight.dtype / .device
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        out = self.base(x)
        if self.multiplier != 0.0 and self.trainable:
            h = self.dropout(x)
            h = nn.functional.linear(h, self.lora_down.weight.to(h.dtype))
            h = nn.functional.linear(h, self.lora_up.weight.to(h.dtype))
            out = out + h * (self.scale * self.multiplier)
        if self.extras_enabled and self.extras:
            for down, up, scale in self.extras:
                h = nn.functional.linear(x, down.to(x.device, x.dtype))
                out = out + nn.functional.linear(h, up.to(x.device, x.dtype)) * scale
        return out


def _target_matches(name: str, targets: Iterable[str]) -> bool:
    return name.rsplit(".", 1)[-1] in targets


def parse_blocks(spec: str, n_layers: int) -> list[int] | None:
    """Parse a block spec like "0-41", "10-31,40", "even", "odd"; empty/all -> None (every block)."""
    spec = (spec or "").strip().lower()
    if spec in ("", "all", "*"):
        return None
    out: set[int] = set()
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part == "even":
            out.update(range(0, n_layers, 2))
        elif part == "odd":
            out.update(range(1, n_layers, 2))
        elif part == "none":
            continue
        elif "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(max(0, lo), min(n_layers - 1, hi) + 1))
        else:
            i = int(part)
            if 0 <= i < n_layers:
                out.add(i)
    return sorted(out)


def format_blocks(blocks: Iterable[int]) -> str:
    """Compact "0-5,8,10-12" form."""
    bl = sorted(set(blocks))
    if not bl:
        return "none"
    parts, start, prev = [], bl[0], bl[0]
    for b in bl[1:] + [None]:
        if b is not None and b == prev + 1:
            prev = b
            continue
        parts.append(f"{start}-{prev}" if prev > start else str(start))
        if b is not None:
            start = prev = b
    return ",".join(parts)


def _wrap(model: nn.Module, name: str, rank: int, alpha: float, dropout: float = 0.0, trainable: bool = True):
    module = model.get_submodule(name)
    if isinstance(module, LoRALinear):
        return module
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    device = module.weight.device if hasattr(module, "weight") else "cpu"
    wrapper = LoRALinear(module, rank, alpha, dropout, trainable).to(device)
    setattr(parent, child, wrapper)
    return wrapper


def _is_linear_like(m: nn.Module) -> bool:
    return isinstance(m, nn.Linear) or (hasattr(m, "in_features") and hasattr(m, "out_features")
                                        and not isinstance(m, LoRALinear))


def inject_lora(model: nn.Module, rank: int, alpha: float, targets: Iterable[str],
                include_text_embedder: bool = False, dropout: float = 0.0,
                blocks: Iterable[int] | None = None) -> dict[str, LoRALinear]:
    """Replace matching Linear modules in-place with trainable LoRA wrappers. Returns {name: LoRALinear}.

    blocks: DiT block indices to train (None = all). Blocks outside the set get no LoRA; autograd then skips the
    backward pass through every block before the first trained one, which also speeds training up.
    """
    targets = set(targets)
    block_set = None if blocks is None else set(int(b) for b in blocks)
    text_targets = {"qkv", "proj", "ffn_gate", "ffn_up", "ffn_down"}
    injected: dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        if not _is_linear_like(module):
            continue
        if name.startswith("layers."):
            if not _target_matches(name, targets):
                continue
            if block_set is not None and int(name.split(".")[1]) not in block_set:
                continue
        elif name.startswith("text_embedder."):
            if not include_text_embedder or not _target_matches(name, text_targets):
                continue
        else:
            continue
        injected[name] = _wrap(model, name, rank, alpha, dropout, trainable=True)
    return injected


def lora_parameters(injected: dict[str, LoRALinear]):
    for m in injected.values():
        yield m.lora_down.weight
        yield m.lora_up.weight


def lora_param_groups(injected: dict[str, LoRALinear], lr: float, plus_ratio: float = 0.0) -> list[dict]:
    """Optimizer param groups. With LoRA+ (ratio > 1) the up matrices get lr x ratio in their own group."""
    downs = [m.lora_down.weight for m in injected.values()]
    ups = [m.lora_up.weight for m in injected.values()]
    if plus_ratio and plus_ratio > 1.0:
        return [{"params": downs, "lr": lr, "name": "lora_down"},
                {"params": ups, "lr": lr * plus_ratio, "name": "lora_up"}]
    return [{"params": downs + ups, "lr": lr, "name": "lora"}]


def set_multiplier(injected: dict[str, LoRALinear], value: float) -> None:
    for m in injected.values():
        m.multiplier = value


def export_state_dict(injected: dict[str, LoRALinear], dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
    out = {}
    for name, m in injected.items():
        key = COMFY_PREFIX + name
        out[f"{key}.lora_down.weight"] = m.lora_down.weight.detach().to("cpu", dtype).contiguous()
        out[f"{key}.lora_up.weight"] = m.lora_up.weight.detach().to("cpu", dtype).contiguous()
        out[f"{key}.alpha"] = torch.tensor(float(m.alpha), dtype=torch.float32)
    return out


def save_lora(injected: dict[str, LoRALinear], path, metadata: dict | None = None, dtype=torch.bfloat16) -> None:
    if isinstance(dtype, str):
        dtype = SAVE_DTYPES[dtype]
    meta = {"format": "pt", "rulztrainer_version": "0.1.0"}
    if metadata:
        meta.update({k: (v if isinstance(v, str) else json.dumps(v)) for k, v in metadata.items()})
    save_file(export_state_dict(injected, dtype), str(path), metadata=meta)


def load_lora_weights(injected: dict[str, LoRALinear], path) -> int:
    """Load a previously exported LoRA into injected modules. Returns number of modules loaded."""
    state = load_file(str(path))
    loaded = 0
    for name, m in injected.items():
        key = COMFY_PREFIX + name
        down, up = state.get(f"{key}.lora_down.weight"), state.get(f"{key}.lora_up.weight")
        if down is None or up is None:
            continue
        if down.shape != m.lora_down.weight.shape or up.shape != m.lora_up.weight.shape:
            raise ValueError(f"Rank mismatch for {name}: file has rank {down.shape[0]}, config has {m.rank}")
        with torch.no_grad():
            m.lora_down.weight.copy_(down.float())
            m.lora_up.weight.copy_(up.float())
        alpha = state.get(f"{key}.alpha")
        if alpha is not None:
            m.alpha = float(alpha)
            m.scale = m.alpha / m.rank
        loaded += 1
    return loaded


# ---------------------------------------------------------------- extra (sampling-only) LoRAs


def parse_comfy_lora(state: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """Return {module_name: (down, up, scale)} from a ComfyUI/PEFT-style LoRA file. Fused qkv is split."""
    out: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    stems = set()
    for k in state:
        for suf in (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight"):
            if k.endswith(suf):
                stems.add(k[: -len(suf)])
    for stem in stems:
        down = state.get(f"{stem}.lora_down.weight", state.get(f"{stem}.lora_A.weight"))
        up = state.get(f"{stem}.lora_up.weight", state.get(f"{stem}.lora_B.weight"))
        if down is None or up is None:
            continue
        rank = down.shape[0]
        alpha = state.get(f"{stem}.alpha")
        scale = (float(alpha) / rank) if alpha is not None else 1.0
        name = stem
        for p in ("diffusion_model.", "transformer.", "lora_unet_"):
            if name.startswith(p):
                name = name[len(p):]
        if name.startswith("layers_"):  # lora_unet_layers_0_proj_q style
            name = name.replace("_", ".", 2).replace("proj_", "proj.")
        down, up = down.float(), up.float()
        if name.endswith(".proj.qkv"):
            for n, part in zip("qkv", up.chunk(3, 0)):
                out[name[: -len("qkv")] + n] = (down, part.contiguous(), scale)
        else:
            out[name] = (down, up, scale)
    return out


def attach_extra_lora(model: nn.Module, path: str, strength: float = 1.0) -> int:
    """Attach a sampling-only LoRA (e.g. a turbo/step-distill LoRA) to the model. Returns modules matched."""
    entries = parse_comfy_lora(load_file(str(path)))
    matched = 0
    names = {n for n, _ in model.named_modules()}
    for name, (down, up, scale) in entries.items():
        if name not in names:
            continue
        module = model.get_submodule(name)
        if not isinstance(module, LoRALinear):
            if not _is_linear_like(module):
                continue
            module = _wrap(model, name, rank=1, alpha=1.0, trainable=False)
        module.extras.append((down, up, scale * strength))
        matched += 1
    return matched


def set_extras_enabled(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.extras_enabled = enabled
