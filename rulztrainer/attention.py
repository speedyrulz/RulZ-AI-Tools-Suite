"""The attention kernel the DiT blocks run (`attention_backend`): PyTorch's own choice, or one SDPA backend pinned.

"torch" leaves scaled_dot_product_attention to pick (flash where it can, else memory-efficient, else math). "cudnn"
pins cuDNN's fused attention: on the RTX 5060 Ti it ran Krea 2 nvfp4 steps 15-22% faster than PyTorch's pick at the
same loss and memory (ai-toolkit pins it for Blackwell too), so "auto" (the default) is cuDNN on a Blackwell training
GPU and PyTorch's pick elsewhere. "flash", "efficient" and "math" pin the others. A pinned backend that rejects a
call (a mask or dtype it does not take) falls back to PyTorch's pick for the rest of the process, with one warning:
the DiT's calls all have the same shape, so one failure means every call would fail."""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

log = logging.getLogger("rulztrainer")

BACKENDS = {"auto": None, "torch": None, "cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}
_pinned: SDPBackend | None = None
_name = "torch"


def is_blackwell(device) -> bool:
    """Compute capability 10.x / 12.x (B200, RTX 50), where cuDNN attention measured 15-22% faster per step."""
    try:
        dev = torch.device(device)
        return dev.type == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability(dev)[0] >= 10
    except (RuntimeError, ValueError, AssertionError):
        return False


def resolve(name: str, device="cuda") -> str:
    """The backend `name` stands for: "auto" is cuDNN on a Blackwell training GPU, else PyTorch's own pick."""
    key = (name or "auto").strip().lower()
    if key not in BACKENDS:
        raise ValueError(f"attention_backend {name!r}: choose one of {', '.join(BACKENDS)}")
    if key == "auto":
        return "cudnn" if is_blackwell(device) else "torch"
    return key


def set_backend(name: str, device="cuda") -> None:
    global _pinned, _name
    key = resolve(name, device)
    _pinned, _name = BACKENDS[key], key


def backend_name() -> str:
    return _name


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    """scaled_dot_product_attention on [B, heads, S, D] tensors, through the pinned backend when there is one."""
    global _pinned, _name
    if _pinned is None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    try:
        with sdpa_kernel(_pinned):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    except RuntimeError as e:
        log.warning("attention_backend %s rejected a call (%s); PyTorch picks the kernel from here on", _name,
                    str(e).splitlines()[0][:160])
        _pinned, _name = None, "torch"
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
