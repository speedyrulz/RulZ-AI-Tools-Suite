"""The attention kernel the DiT blocks run (`attention_backend`): PyTorch's own choice, or one SDPA backend first.

"torch" leaves scaled_dot_product_attention to pick (flash where it can, else memory-efficient, else math). "cudnn"
puts cuDNN's fused attention first: on the RTX 5060 Ti it ran Krea 2 nvfp4 steps 15-22% faster than PyTorch's pick at
the same loss and memory (ai-toolkit uses it for Blackwell too), and Qwen-Image 2.1 int8 steps on the RTX 4060 Ti
5-9% faster, so "auto" (the default) is cuDNN on an Ada or Blackwell training GPU and PyTorch's pick elsewhere. "flash", "efficient" and "math" put the others first.

The choice is PyTorch's process-wide priority order, set once: the chosen backend is tried first and every other one
stays enabled behind it, so a call it cannot take (a mask, a dtype, a head size) quietly runs on the next one. It used
to be a per-call sdpa_kernel(backend) context, which disables every other backend in PyTorch's global flags for the
duration of the call: with samples and probes rendering on several GPUs in threads at once, another thread's fp32 VAE
attention (Krea 2's) then found no backend enabled and failed with "No available kernel". The priority order applies to
every scaled_dot_product_attention in the process (text encoders, VAEs, scorers); each still runs on the first backend
that takes its call."""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

log = logging.getLogger("rulztrainer")

BACKENDS = {"auto": None, "torch": None, "cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}
_DEFAULT_ORDER = list(torch._C._get_sdp_priority_order()) if hasattr(torch._C, "_get_sdp_priority_order") else None
_name = "torch"


def cudnn_first(device) -> bool:
    """Compute capability 8.9 (RTX 40, Ada) or 10.x / 12.x (B200, RTX 50), where cuDNN attention measured 5-22% faster
    per step. Ampere and Hopper are not measured."""
    try:
        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return False
        return tuple(torch.cuda.get_device_capability(dev)) == (8, 9) or torch.cuda.get_device_capability(dev)[0] >= 10
    except (RuntimeError, ValueError, AssertionError):
        return False


def resolve(name: str, device="cuda") -> str:
    """The backend `name` stands for: "auto" is cuDNN on an Ada or Blackwell training GPU, else PyTorch's pick."""
    key = (name or "auto").strip().lower()
    if key not in BACKENDS:
        raise ValueError(f"attention_backend {name!r}: choose one of {', '.join(BACKENDS)}")
    if key == "auto":
        return "cudnn" if cudnn_first(device) else "torch"
    return key


def priority_order(key: str) -> list[int] | None:
    """PyTorch's backend priority order (backend ids) for a resolved name: the chosen one first, the rest in PyTorch's
    default order behind it; None where PyTorch has no priority order to set."""
    if _DEFAULT_ORDER is None:
        return None
    first = BACKENDS.get(key)
    if first is None:
        return list(_DEFAULT_ORDER)
    return [int(first), *(b for b in _DEFAULT_ORDER if b != int(first))]


def set_backend(name: str, device="cuda") -> None:
    global _name
    key = resolve(name, device)
    order = priority_order(key)
    if order is not None:
        torch._C._set_sdp_priority_order(order)
    elif BACKENDS.get(key) is not None:
        log.warning("This PyTorch has no SDPA priority order: attention_backend %s is ignored", key)
        key = "torch"
    _name = key


def backend_name() -> str:
    return _name


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None,
         is_causal: bool = False) -> torch.Tensor:
    """scaled_dot_product_attention on [B, heads, S, D] tensors (the backend order set_backend chose)."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
