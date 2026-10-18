"""Weight-only quantization for frozen base weights (LoRA training keeps the base frozen).

Formats mirror ComfyUI / comfy_kitchen so quantized ComfyUI files load as-is and bf16 weights can be
quantized on load to the same layouts:

  int8          row-wise symmetric int8, fp32 scale [out, 1]            ("int8_tensorwise")
  int8_convrot  int8 after a grouped regular-Hadamard rotation (256)     ("int8_tensorwise" + convrot)
  nvfp4         fp4 e2m1 packed 2/byte, fp8 e4m3 block scales (16) in cuBLAS swizzled layout,
                fp32 per-tensor scale                                    ("nvfp4")

QuantLinear dequantizes on the fly in forward and again in backward, so no dequantized copy of the
weight is kept alive between the two passes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
from torch import nn

FORMATS = ("none", "int8", "int8_convrot", "nvfp4")
F4_E2M1_MAX = 6.0
F8_E4M3_MAX = 448.0
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_LUT = torch.cat([E2M1_VALUES, -E2M1_VALUES])
_HADAMARD: dict[int, torch.Tensor] = {}


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _roundup(x: int, m: int) -> int:
    return _ceil_div(x, m) * m


# ---------------------------------------------------------------- int8 (+convrot)


def hadamard(size: int) -> torch.Tensor:
    if size not in _HADAMARD:
        if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
            raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h, cur = h4, 4
        while cur < size:
            h = torch.kron(h, h4)
            cur *= 4
        _HADAMARD[size] = h / (size ** 0.5)
    return _HADAMARD[size]


def rotate_weight(w: torch.Tensor, group: int) -> torch.Tensor:
    """W_rot = W @ H_block^T (H is symmetric, so this is also the inverse)."""
    out_f, in_f = w.shape
    h = hadamard(group).to(w.device, w.dtype)
    return (w.reshape(out_f, in_f // group, group) @ h.T).reshape(out_f, in_f)


def quantize_int8(w: torch.Tensor, convrot: bool, group: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    w = w.float()
    if convrot:
        w = rotate_weight(w, group)
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-12)
    q = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, convrot: bool, group: int, dtype) -> torch.Tensor:
    w = q.float() * scale.float()
    if convrot:
        w = rotate_weight(w, group)
    return w.to(dtype)


# ---------------------------------------------------------------- nvfp4


def to_blocked(m: torch.Tensor) -> torch.Tensor:
    rows, cols = m.shape
    nr, nc = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pr, pc = nr * 128, nc * 4
    if (rows, cols) != (pr, pc):
        padded = torch.zeros((pr, pc), device=m.device, dtype=m.dtype)
        padded[:rows, :cols] = m
        m = padded
    blocks = m.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).reshape(pr, pc)


def from_blocked(b: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    nr, nc = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pr, pc = nr * 128, nc * 4
    s = b.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(1, 2)
    s = s.reshape(nr, nc, 4, 32, 4).reshape(nr, nc, 128, 4).permute(0, 2, 1, 3)
    return s.reshape(pr, pc)[:rows, :cols]


def _f32_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values (already scaled to [-6, 6]) to e2m1 codes 0..15 (sign in bit 3)."""
    sign = (x < 0).to(torch.uint8)
    a = x.abs()
    vals = E2M1_VALUES.to(x.device)
    mids = (vals[:-1] + vals[1:]) / 2  # 0.25 0.75 1.25 1.75 2.5 3.5 5
    code = torch.bucketize(a, mids, right=False).to(torch.uint8)  # ties go to the lower code...
    # ...except round-to-nearest-even prefers an even code on exact ties
    tie = torch.isin(a, mids)
    code = torch.where(tie & (code % 2 == 1), code + 1, code)
    code = code.clamp(max=7)
    return code | (sign << 3)


def quantize_nvfp4(w: torch.Tensor, per_tensor_scale: torch.Tensor | None = None):
    """Returns (packed uint8 [rows_p, cols_p/2], block scales fp8 swizzled, per-tensor fp32 scale)."""
    w = w.float()
    rows, cols = w.shape
    pr, pc = _roundup(rows, 16), _roundup(cols, 16)
    if (pr, pc) != (rows, cols):
        w = nn.functional.pad(w, (0, pc - cols, 0, pr - rows))
    if per_tensor_scale is None:
        per_tensor_scale = w.abs().amax() / (F8_E4M3_MAX * F4_E2M1_MAX)
    per_tensor_scale = torch.as_tensor(per_tensor_scale, dtype=torch.float32, device=w.device)
    x = w.reshape(pr, -1, 16)
    block_scale = x.abs().amax(dim=-1) / F4_E2M1_MAX
    scaled = (block_scale / per_tensor_scale).clamp(max=F8_E4M3_MAX)
    scaled_fp8 = scaled.to(torch.float8_e4m3fn)
    total = per_tensor_scale * scaled_fp8.float()
    safe = torch.where(total == 0, torch.ones_like(total), total)
    data = (x / safe.unsqueeze(-1)).clamp(-F4_E2M1_MAX, F4_E2M1_MAX)
    data = torch.where((total == 0).unsqueeze(-1), torch.zeros_like(data), data).reshape(pr, pc)
    codes = _f32_to_e2m1_codes(data).reshape(-1)
    packed = (codes[::2] << 4 | codes[1::2]).reshape(pr, pc // 2)
    return packed, to_blocked(scaled_fp8), per_tensor_scale


def dequantize_nvfp4(packed: torch.Tensor, block_scales: torch.Tensor, per_tensor_scale: torch.Tensor,
                     orig_shape: tuple[int, int], dtype) -> torch.Tensor:
    pr, half = packed.shape
    pc = half * 2
    lut = _E2M1_LUT.to(packed.device, torch.float32)
    codes = torch.stack([packed >> 4, packed & 0x0F], dim=-1).reshape(pr, pc)
    vals = lut[codes.int()].reshape(pr, pc // 16, 16)
    scales = from_blocked(block_scales, pr, pc // 16).float() * per_tensor_scale.float()
    w = (vals * scales.unsqueeze(-1)).reshape(pr, pc)
    return w[: orig_shape[0], : orig_shape[1]].to(dtype)


# ---------------------------------------------------------------- container + module


@dataclass
class QuantWeight:
    fmt: str  # int8 | int8_convrot | nvfp4
    qdata: torch.Tensor
    scale: torch.Tensor  # int8: [out,1] fp32 ; nvfp4: swizzled fp8 block scales
    scale2: torch.Tensor | None  # nvfp4 per-tensor fp32 scale
    orig_shape: tuple[int, int]
    group: int = 256

    def dequantize(self, dtype=torch.bfloat16) -> torch.Tensor:
        if self.fmt in ("int8", "int8_convrot"):
            return dequantize_int8(self.qdata, self.scale, self.fmt == "int8_convrot", self.group, dtype)
        return dequantize_nvfp4(self.qdata, self.scale, self.scale2, self.orig_shape, dtype)

    def split_rows(self, n: int) -> list["QuantWeight"]:
        """Split along out_features into n equal parts (used to unfuse qkv)."""
        if self.fmt == "nvfp4":
            parts = self.dequantize(torch.float32).chunk(n, 0)
            return [quantize_weight(p, "nvfp4") for p in parts]
        return [QuantWeight(self.fmt, q, s, None, (q.shape[0], self.orig_shape[1]), self.group)
                for q, s in zip(self.qdata.chunk(n, 0), self.scale.chunk(n, 0))]

    def nbytes(self) -> int:
        return self.qdata.numel() * self.qdata.element_size() + self.scale.numel() * self.scale.element_size()


def quantize_weight(w: torch.Tensor, fmt: str, group: int = 256) -> QuantWeight:
    w = w.detach()
    if fmt == "int8" or fmt == "int8_convrot":
        convrot = fmt == "int8_convrot" and w.shape[1] % group == 0
        q, s = quantize_int8(w, convrot, group)
        return QuantWeight("int8_convrot" if convrot else "int8", q, s, None, tuple(w.shape), group)
    if fmt == "nvfp4":
        packed, bs, s2 = quantize_nvfp4(w)
        return QuantWeight("nvfp4", packed, bs, s2, tuple(w.shape))
    raise ValueError(f"Unknown quant format {fmt}")


def can_quantize(w: torch.Tensor, fmt: str, min_dim: int = 1024) -> bool:
    if w.ndim != 2 or min(w.shape) < min_dim:
        return False
    if fmt == "nvfp4":
        return True  # padding handles non-multiples of 16
    return True


class _QuantLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, qw: QuantWeight):
        w = qw.dequantize(x.dtype)
        ctx.qw = qw
        return nn.functional.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad_out):
        w = ctx.qw.dequantize(grad_out.dtype)
        grad_x = grad_out @ w
        return grad_x, None, None


class QuantLinear(nn.Module):
    """Frozen Linear whose weight is stored quantized; dequantized per forward/backward."""

    def __init__(self, qw: QuantWeight, bias: torch.Tensor | None, compute_dtype=torch.bfloat16):
        super().__init__()
        self.fmt = qw.fmt
        self.group = qw.group
        self.orig_shape = tuple(qw.orig_shape)
        self.out_features, self.in_features = self.orig_shape
        self.compute_dtype = compute_dtype
        self.register_buffer("qdata", qw.qdata)
        self.register_buffer("scale", qw.scale)
        self.register_buffer("scale2", qw.scale2 if qw.scale2 is not None else torch.zeros(0))
        if bias is not None:
            self.register_buffer("bias", bias.to(compute_dtype))
        else:
            self.bias = None

    @property
    def qw(self) -> QuantWeight:
        return QuantWeight(self.fmt, self.qdata, self.scale, self.scale2 if self.scale2.numel() else None,
                           self.orig_shape, self.group)

    @property
    def weight(self):  # for code that only inspects .weight.dtype / .device
        return self.qdata.new_zeros(0, dtype=self.compute_dtype)

    def dequantized(self, dtype=None) -> torch.Tensor:
        return self.qw.dequantize(dtype or self.compute_dtype)

    def forward(self, x):
        return _QuantLinearFn.apply(x, self.bias, self.qw)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, fmt={self.fmt}, bias={self.bias is not None}"


def linear_compute_dtype(m: nn.Module) -> torch.dtype:
    d = getattr(m, "compute_dtype", None)
    return d if d is not None else m.weight.dtype


def quantize_module_linears(model: nn.Module, fmt: str, min_dim: int = 1024, log=None) -> int:
    """Replace eligible nn.Linear modules in-place with QuantLinear. Returns count."""
    if fmt in ("none", "auto", None):
        return 0
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not can_quantize(module.weight, fmt, min_dim):
            continue
        qw = quantize_weight(module.weight.data, fmt)
        ql = QuantLinear(qw, module.bias.data if module.bias is not None else None, module.weight.dtype)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, ql.to(module.weight.device))
        del module
        count += 1
    if log:
        log.info("Quantized %d Linear layers to %s", count, fmt)
    return count


# ---------------------------------------------------------------- ComfyUI file (de)serialization


def parse_comfy_quant(blob: torch.Tensor) -> dict:
    return json.loads(bytes(blob.tolist()).decode("utf-8"))


def quant_weight_from_comfy(meta: dict, weight: torch.Tensor, scale: torch.Tensor,
                            scale2: torch.Tensor | None) -> QuantWeight:
    fmt = meta.get("format")
    if fmt == "int8_tensorwise":
        convrot = bool(meta.get("convrot"))
        return QuantWeight("int8_convrot" if convrot else "int8", weight, scale.float(), None,
                           tuple(weight.shape), int(meta.get("convrot_groupsize", 256)))
    if fmt == "nvfp4":
        shape = meta.get("orig_shape")
        if shape is None:
            shape = (weight.shape[0], weight.shape[1] * 2)
        return QuantWeight("nvfp4", weight, scale, scale2.float(), tuple(shape))
    raise ValueError(f"Unsupported ComfyUI quantization format '{fmt}'. Supported: int8_tensorwise (convrot), nvfp4.")
