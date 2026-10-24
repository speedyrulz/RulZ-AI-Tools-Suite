"""Minimal LoRA for Linear layers with ComfyUI- or diffusers-compatible export, plus "extra" LoRAs (e.g. turbo /
step-distill adapters) that are applied only while sampling.

Export key formats (Arch.lora_format):

  comfy  accepted by ComfyUI's LoRA loader for any model whose state-dict keys are "diffusion_model.<module>.weight"
    diffusion_model.<module>.lora_down.weight   [rank, in]
    diffusion_model.<module>.lora_up.weight     [out, rank]
    diffusion_model.<module>.alpha              scalar
  peft   diffusers / PEFT (pipe.load_lora_weights, transformer.load_lora_adapter); FIBO 2.0 uses it
    transformer.<module>.lora_A.weight          [rank, in]
    transformer.<module>.lora_B.weight          [out, rank]
    + "lora_adapter_metadata" (the PEFT LoraConfig, prefixed "transformer.") in the file metadata

Effective delta = (alpha / rank) * up @ down in both.
"""
from __future__ import annotations

import json
import math
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

COMFY_PREFIX = "diffusion_model."
PEFT_PREFIX = "transformer."
LORA_FORMATS = ("comfy", "peft")
# peft 0.21 LoraConfig fields: diffusers rebuilds the adapter config from the file metadata with LoraConfig(**these)
PEFT_CONFIG_TEMPLATE = {
    "alora_invocation_tokens": None,
    "alpha_pattern": {},
    "arrow_config": None,
    "auto_mapping": None,
    "base_model_name_or_path": None,
    "bias": "none",
    "corda_config": None,
    "ensure_weight_tying": False,
    "eva_config": None,
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": False,
    "init_lora_weights": True,
    "kasa_config": None,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": 16,
    "lora_bias": False,
    "lora_dropout": 0.0,
    "lora_ga_config": None,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "monteclora_config": None,
    "peft_type": "LORA",
    "peft_version": "0.21.2",
    "qalora_group_size": 16,
    "r": 16,
    "rank_pattern": {},
    "revision": None,
    "target_modules": [],
    "target_parameters": None,
    "task_type": None,
    "trainable_token_indices": None,
    "use_bdlora": None,
    "use_dora": False,
    "use_qalora": False,
    "use_rslora": False,
    "velora_config": None
}
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


def backprop_blocks(model: nn.Module, injected: dict, blocks: list[int] | None) -> list[int] | None:
    """The trained blocks as the backward pass sees them: `blocks` when every LoRA sits in the DiT blocks or after
    them (the head), so the blocks before the first trained one stay forward-only; None (every block runs a backward)
    when a LoRA sits before the blocks (Krea's text fusion, the text / patch / timestep projections, MOSS's text
    embedder, FIBO's refiners), since its gradient flows back through all of them."""
    if blocks is None:
        return None
    tail = getattr(model, "split_tail_modules", None)
    after = [model.layers, *(tail() if tail else [m for m in (getattr(model, "final_norm_video", None),
                                                             getattr(model, "final_linear_video", None)) if m])]
    inside = {id(m) for part in after for m in part.modules()}
    return blocks if all(id(mod) in inside for mod in injected.values()) else None


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
                blocks: Iterable[int] | None = None, names: Iterable[str] | None = None) -> dict[str, LoRALinear]:
    """Replace matching Linear modules in-place with trainable LoRA wrappers. Returns {name: LoRALinear}.

    names: the exact module names to wrap (Arch.lora_module_names); without it the MOSS-Image layout rules below
    apply. blocks: DiT block indices to train (None = all). Blocks outside the set get no LoRA; autograd then skips
    the backward pass through every block before the first trained one, which also speeds training up.
    """
    if names is not None:
        wanted = set(names)
        injected = {}
        for name, module in list(model.named_modules()):
            if name in wanted and _is_linear_like(module):
                injected[name] = _wrap(model, name, rank, alpha, dropout, trainable=True)
        missing = wanted - set(injected)
        if missing:
            raise ValueError(f"LoRA targets not found in the model: {sorted(missing)[:5]}")
        return injected
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


@torch.no_grad()
def mirror_lora(src: dict, dst: dict) -> None:
    """Copy LoRA state from one model's wrappers onto another's with the same module names (a data-parallel replica,
    the sampling model): weights (resized when the rank differs), alpha / scale / rank, on/off and dropout."""
    for name, s in src.items():
        d = dst[name]
        for sp, dp in ((s.lora_down.weight, d.lora_down.weight), (s.lora_up.weight, d.lora_up.weight)):
            if dp.shape != sp.shape:
                dp.data = torch.empty(sp.shape, dtype=sp.dtype, device=dp.device)
            dp.copy_(sp, non_blocking=True)
        d.alpha, d.scale, d.rank = s.alpha, s.scale, s.rank
        if d.trainable != s.trainable:
            d.trainable = s.trainable
        if getattr(d.dropout, "p", 0.0) != getattr(s.dropout, "p", 0.0):
            d.dropout = nn.Dropout(s.dropout.p) if getattr(s.dropout, "p", 0.0) > 0 else nn.Identity()


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


def lora_param_groups_by(injected: dict[str, LoRALinear], lr: float, group_of) -> list[dict]:
    """One param group per layer type (group_of(module name) -> type), for optimizers that adapt one lr per group."""
    groups: dict[str, list] = {}
    for name, m in injected.items():
        groups.setdefault(group_of(name), []).extend([m.lora_down.weight, m.lora_up.weight])
    return [{"params": ps, "lr": lr, "name": g} for g, ps in sorted(groups.items())]


def set_multiplier(injected: dict[str, LoRALinear], value: float) -> None:
    for m in injected.values():
        m.multiplier = value


def export_state_dict(injected: dict[str, LoRALinear], dtype=torch.bfloat16, fmt: str = "comfy") -> dict[str, torch.Tensor]:
    if fmt not in LORA_FORMATS:
        raise ValueError(f"LoRA format must be one of {LORA_FORMATS}, got {fmt!r}")
    out = {}
    for name, m in injected.items():
        down = m.lora_down.weight.detach().to("cpu", dtype).contiguous()
        up = m.lora_up.weight.detach().to("cpu", dtype).contiguous()
        if fmt == "peft":
            out[f"{PEFT_PREFIX}{name}.lora_A.weight"] = down
            out[f"{PEFT_PREFIX}{name}.lora_B.weight"] = up
        else:
            key = COMFY_PREFIX + name
            out[f"{key}.lora_down.weight"] = down
            out[f"{key}.lora_up.weight"] = up
            out[f"{key}.alpha"] = torch.tensor(float(m.alpha), dtype=torch.float32)
    return out


def peft_adapter_metadata(injected: dict[str, LoRALinear]) -> dict:
    """The PEFT LoraConfig for these modules, keys prefixed "transformer." as diffusers' loader expects."""
    cfg = dict(PEFT_CONFIG_TEMPLATE)
    ranks = {m.rank for m in injected.values()}
    alphas = {float(m.alpha) for m in injected.values()}
    cfg["r"] = max(ranks)
    cfg["lora_alpha"] = max(alphas)
    cfg["target_modules"] = sorted(injected)
    cfg["rank_pattern"] = {n: m.rank for n, m in injected.items() if m.rank != cfg["r"]}
    cfg["alpha_pattern"] = {n: float(m.alpha) for n, m in injected.items() if float(m.alpha) != cfg["lora_alpha"]}
    return {f"{PEFT_PREFIX}{k}": v for k, v in cfg.items()}


def save_lora(injected: dict[str, LoRALinear], path, metadata: dict | None = None, dtype=torch.bfloat16,
              fmt: str = "comfy") -> None:
    if isinstance(dtype, str):
        dtype = SAVE_DTYPES[dtype]
    meta = {"format": "pt", "rulztrainer_version": "0.1.0"}
    if fmt == "peft":
        meta["lora_adapter_metadata"] = json.dumps(peft_adapter_metadata(injected), indent=2, sort_keys=True)
    if metadata:
        meta.update({k: (v if isinstance(v, str) else json.dumps(v)) for k, v in metadata.items()})
    save_file(export_state_dict(injected, dtype, fmt), str(path), metadata=meta)


def read_peft_metadata(path) -> dict:
    """The PEFT LoraConfig stored in a diffusers-format LoRA file (prefixes stripped), or {}."""
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    raw = meta.get("lora_adapter_metadata")
    if not raw:
        return {}
    return {k.removeprefix(PEFT_PREFIX): v for k, v in json.loads(raw).items()}


def load_lora_weights(injected: dict[str, LoRALinear], path) -> int:
    """Load a previously exported LoRA (either format) into injected modules. Returns number of modules loaded."""
    state = load_file(str(path))
    peft = any(k.endswith(".lora_A.weight") for k in state)
    peft_meta = read_peft_metadata(path) if peft else {}
    loaded = 0
    for name, m in injected.items():
        if peft:
            key = PEFT_PREFIX + name
            down, up = state.get(f"{key}.lora_A.weight"), state.get(f"{key}.lora_B.weight")
        else:
            key = COMFY_PREFIX + name
            down, up = state.get(f"{key}.lora_down.weight"), state.get(f"{key}.lora_up.weight")
        if down is None or up is None:
            continue
        if down.shape != m.lora_down.weight.shape or up.shape != m.lora_up.weight.shape:
            raise ValueError(f"Rank mismatch for {name}: file has rank {down.shape[0]}, config has {m.rank}")
        with torch.no_grad():
            m.lora_down.weight.copy_(down.float())
            m.lora_up.weight.copy_(up.float())
        if peft:
            alpha = (peft_meta.get("alpha_pattern") or {}).get(name, peft_meta.get("lora_alpha"))
        else:
            alpha = state.get(f"{key}.alpha")
        if alpha is not None:
            m.alpha = float(alpha)
            m.scale = m.alpha / m.rank
        loaded += 1
    return loaded


# ---------------------------------------------------------------- extra (sampling-only) LoRAs


def parse_comfy_lora(state: dict[str, torch.Tensor],
                     default_alpha: float | None = None) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """Return {module_name: (down, up, scale)} from a ComfyUI/PEFT-style LoRA file. Fused qkv is split.
    default_alpha: the alpha for entries without an .alpha tensor (PEFT files keep it in their metadata)."""
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
        alpha = float(alpha) if alpha is not None else default_alpha
        scale = (alpha / rank) if alpha is not None else 1.0
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
    peft_meta = read_peft_metadata(path)
    entries = parse_comfy_lora(load_file(str(path)), default_alpha=peft_meta.get("lora_alpha"))
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
