"""Weight-only quantization for frozen base weights (LoRA training keeps the base frozen).

Formats mirror ComfyUI / comfy_kitchen so quantized ComfyUI files load as-is and bf16 weights can be
quantized on load to the same layouts:

  int8          row-wise symmetric int8, fp32 scale [out, 1]            ("int8_tensorwise")
  int8_convrot  int8 after a grouped regular-Hadamard rotation (256)     ("int8_tensorwise" + convrot)
  nvfp4         fp4 e2m1 packed 2/byte, fp8 e4m3 block scales (16) in cuBLAS swizzled layout,
                fp32 per-tensor scale                                    ("nvfp4")

int8 / int8_convrot layers run W8A8 by default: the activations are quantized per token to int8 and
multiplied with the stored int8 weight by torch._int_mm (int32 accumulate), so the weight is never
dequantized in forward. Backward is exact by default ("bf16": grad_x = grad_out @ W in bf16) or int8
("int8": quantizes grad_out per token too). nvfp4 layers, CPU tensors and shapes the int8 kernel rejects
dequantize on the fly in forward and again in backward (no dequantized copy is kept between the passes).

The two elementwise passes around the int8 GEMM (per-token activation quantization, int32 -> compute-dtype
rescale) are single fused triton kernels when triton imports (triton-windows on Windows); for int8_convrot the
grouped Hadamard rotation of the activations is folded into the quantization kernel as well (tensor-core dot
against the cached H, rotated values stay in registers). Otherwise the eager PyTorch versions run (bf16 matmul
rotation, two/three-pass elementwise). The choice is made once at import (RULZ_W8A8_TRITON=0 forces eager).
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass

import torch
from torch import nn

log = logging.getLogger("rulztrainer")
FORMATS = ("none", "int8", "int8_convrot", "nvfp4")
INT8_BACKWARDS = ("bf16", "int8")
# process-wide defaults for QuantLinear layers built without explicit values (set_int8_defaults / TrainConfig)
DEFAULT_INT8_MATMUL = True
DEFAULT_INT8_BACKWARD = "bf16"
_W8A8_WARNED = False
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


_HADAMARD_DEV: dict[tuple, torch.Tensor] = {}


def hadamard_on(size: int, device, dtype) -> torch.Tensor:
    """hadamard(size) cached per device/dtype (a fresh .to(device) each call is a synchronous H2D copy)."""
    key = (size, torch.device(device), dtype)
    h = _HADAMARD_DEV.get(key)
    if h is None:
        h = _HADAMARD_DEV[key] = hadamard(size).to(device, dtype)
    return h


def rotate_weight(w: torch.Tensor, group: int) -> torch.Tensor:
    """W_rot = W @ H_block^T (H is symmetric, so this is also the inverse)."""
    out_f, in_f = w.shape
    h = hadamard_on(group, w.device, w.dtype)
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


# ---------------------------------------------------------------- W8A8 (int8 activations x int8 weight)


def set_int8_defaults(int8_matmul: bool, int8_backward: str = "bf16") -> None:
    """Defaults for QuantLinear layers created afterwards (the trainer sets them from TrainConfig)."""
    global DEFAULT_INT8_MATMUL, DEFAULT_INT8_BACKWARD
    if int8_backward not in INT8_BACKWARDS:
        raise ValueError(f"int8_backward must be one of {INT8_BACKWARDS}, got {int8_backward!r}")
    DEFAULT_INT8_MATMUL = bool(int8_matmul)
    DEFAULT_INT8_BACKWARD = int8_backward


def configure_int8(model: nn.Module, int8_matmul: bool, int8_backward: str = "bf16") -> int:
    """Set the W8A8 options on every QuantLinear in `model`. Returns the number of layers touched."""
    if int8_backward not in INT8_BACKWARDS:
        raise ValueError(f"int8_backward must be one of {INT8_BACKWARDS}, got {int8_backward!r}")
    n = 0
    for m in model.modules():
        if isinstance(m, QuantLinear):
            m.int8_matmul = bool(int8_matmul)
            m.int8_backward = int8_backward
            n += 1
    return n


def rotate_act(x: torch.Tensor, group: int) -> torch.Tensor:
    """Apply the grouped Hadamard to a [tokens, K] tensor: x @ H_block (same op as rotate_weight; self-inverse)."""
    return rotate_weight(x, group)


_ROUND_MAGIC = 1.5 * 2 ** 23  # adding it to an fp32 |v| < 2^22 rounds v to an integer held in the low mantissa bits
_MAGIC_CACHE: dict[torch.device, torch.Tensor] = {}


def _magic(device) -> torch.Tensor:
    t = _MAGIC_CACHE.get(device)
    if t is None:
        t = _MAGIC_CACHE[device] = torch.tensor(_ROUND_MAGIC, dtype=torch.float32, device=device)
    return t


def _quantize_act_int8_eager(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 of a [M, K] float tensor: returns (int8 [M, K], fp32 scale [M, 1]).

    Two memory passes instead of five: the row amax is a fused inf-norm, and round(x / scale) is done by
    addcmul(magic, x, 1/scale) in fp32 (round-to-nearest-even into the mantissa) followed by a cast whose
    low byte is the two's-complement value. Values are bounded by 127 by construction, so no clamp is needed.
    """
    amax = torch.linalg.vector_norm(x, ord=float("inf"), dim=1, keepdim=True, dtype=torch.float32).clamp_(min=1e-12)
    scale = amax / 127.0
    v = torch.addcmul(_magic(x.device), x, 127.0 / amax)  # fp32: magic + round(x / scale)
    q = v.view(torch.int32).to(torch.int8)
    return q, scale


def _rescale_eager(y_i32: torch.Tensor, row_scale: torch.Tensor, col_scale: torch.Tensor | None, dtype) -> torch.Tensor:
    """(y_i32 * row_scale[M,1] * col_scale[1,N]) -> dtype, with the int32 tensor read exactly once."""
    out = torch.empty(y_i32.shape, dtype=dtype, device=y_i32.device)
    torch.mul(y_i32, row_scale, out=out)  # fp32 product, stored as `dtype`
    if col_scale is not None:
        out.mul_(col_scale)
    return out


def _hadamard_sub(group: int) -> int | None:
    """sqrt(group) when hadamard(group) == kron(hadamard(sub), hadamard(sub)) with a tensor-core sized sub (>= 16)."""
    sub = math.isqrt(group)
    if sub * sub != group or sub < 16 or (sub & (sub - 1)) or math.log(sub, 4) % 1:
        return None
    return sub


# ---- fused triton kernels (one memory pass each); the eager versions above are the fallback

HAS_TRITON = False
if os.environ.get("RULZ_W8A8_TRITON", "1") != "0":
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice as _libdevice
        HAS_TRITON = True
    except Exception as _e:  # noqa: BLE001  (not installed, or a broken wheel for this torch)
        log.debug("triton unavailable (%s: %s); W8A8 uses eager elementwise kernels", type(_e).__name__, _e)
USE_TRITON = HAS_TRITON  # flipped off on the first launch failure (see _disable_triton)

if HAS_TRITON:

    @triton.jit
    def _quant_act_kernel(x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm,
                          SINGLE_TILE: tl.constexpr, BLOCK_K: tl.constexpr):
        """One program per token row: amax -> fp32 scale -> round-to-nearest-even(x * 127 / amax) -> int8."""
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * stride_xm
        q_row = q_ptr + row * stride_qm
        offs = tl.arange(0, BLOCK_K)
        if SINGLE_TILE:  # whole row in registers: read HBM once
            mask = offs < K
            x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
            amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
            inv = _libdevice.div_rn(127.0, amax)
            q = _libdevice.float2int_rn(x * inv).to(tl.int8)
            tl.store(q_row + offs, q, mask=mask)
        else:  # long rows: amax pass, then quantize pass (second read hits L2)
            amax = tl.zeros([BLOCK_K], dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                mask = (k0 + offs) < K
                x = tl.load(x_row + k0 + offs, mask=mask, other=0.0).to(tl.float32)
                amax = tl.maximum(amax, tl.abs(x))
            amax = tl.maximum(tl.max(amax, axis=0), 1e-12)
            inv = _libdevice.div_rn(127.0, amax)
            for k0 in range(0, K, BLOCK_K):
                mask = (k0 + offs) < K
                x = tl.load(x_row + k0 + offs, mask=mask, other=0.0).to(tl.float32)
                q = _libdevice.float2int_rn(x * inv).to(tl.int8)
                tl.store(q_row + k0 + offs, q, mask=mask)
        tl.store(s_ptr + row, amax * (1.0 / 127.0))  # torch computes `amax / 127.0` as this reciprocal multiply

    @triton.jit
    def _rescale_kernel(y_ptr, rs_ptr, cs_ptr, out_ptr, N, stride_ym, stride_om,
                        HAS_COL: tl.constexpr, BLOCK_N: tl.constexpr):
        """out[m, n] = y_i32[m, n] * row_scale[m] (* col_scale[n]) computed in fp32, stored as out's dtype."""
        row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        y = tl.load(y_ptr + row * stride_ym + offs, mask=mask, other=0).to(tl.float32)
        v = y * tl.load(rs_ptr + row)
        if HAS_COL:
            v = v * tl.load(cs_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row * stride_om + offs, v.to(out_ptr.dtype.element_ty), mask=mask)

    class _on_device:
        """Triton launches on the current CUDA device; switch only when the tensor lives elsewhere."""

        def __init__(self, device: torch.device):
            self.ctx = None
            if device.index is not None and device.index != torch.cuda.current_device():
                self.ctx = torch.cuda.device(device.index)

        def __enter__(self):
            if self.ctx is not None:
                self.ctx.__enter__()

        def __exit__(self, *a):
            if self.ctx is not None:
                self.ctx.__exit__(*a)

    _QUANT_MAX_TILE = 8192  # rows up to this many columns are quantized from registers in one HBM read

    def _quantize_act_int8_triton(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.stride(1) != 1:
            x = x.contiguous()
        m, k = x.shape
        q = torch.empty((m, k), dtype=torch.int8, device=x.device)
        scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
        if m == 0:
            return q, scale
        block = min(triton.next_power_of_2(max(k, 16)), _QUANT_MAX_TILE)
        with _on_device(x.device):
            _quant_act_kernel[(m,)](x, q, scale, k, x.stride(0), q.stride(0),
                                    SINGLE_TILE=k <= block, BLOCK_K=block, num_warps=8 if block >= 2048 else 4)
        return q, scale

    def _rescale_triton(y_i32: torch.Tensor, row_scale: torch.Tensor, col_scale: torch.Tensor | None,
                        dtype) -> torch.Tensor:
        if y_i32.stride(1) != 1:
            y_i32 = y_i32.contiguous()
        m, n = y_i32.shape
        out = torch.empty((m, n), dtype=dtype, device=y_i32.device)
        if m == 0 or n == 0:
            return out
        rs = row_scale.reshape(-1)
        if rs.numel() != m or rs.dtype != torch.float32:
            raise ValueError(f"row_scale must hold {m} fp32 values, got {tuple(row_scale.shape)} {row_scale.dtype}")
        rs = rs.contiguous()
        if col_scale is not None:
            cs = col_scale.reshape(-1)
            if cs.numel() != n:
                raise ValueError(f"col_scale must hold {n} values, got {tuple(col_scale.shape)}")
            cs = cs.contiguous().float()
        else:
            cs = rs  # unused placeholder (HAS_COL=False)
        block = min(triton.next_power_of_2(n), 1024)
        with _on_device(y_i32.device):
            _rescale_kernel[(m, _ceil_div(n, block))](y_i32, rs, cs, out, n, y_i32.stride(0), out.stride(0),
                                                      HAS_COL=col_scale is not None, BLOCK_N=block,
                                                      num_warps=4)
        return out

    @triton.jit
    def _rot_quant_act_kernel(x_ptr, h_ptr, q_ptr, s_ptr, M, K, stride_xm, stride_qm,
                              ROWS: tl.constexpr, SUB: tl.constexpr):
        """Grouped Hadamard rotation (group SUB*SUB) + per-token int8 quantization of ROWS token rows per program.

        H_group = kron(H_sub, H_sub), so a group of a row viewed as the SUB x SUB matrix X rotates as H_sub @ X @ H_sub:
        two batched tensor-core dots against the tiny H_sub (bf16/fp16, entries +-1/sqrt(SUB), exact) instead of one
        against the full group-sized H (which would re-read 128 KB per group). The first dot's fp32 result is split
        into a hi + lo pair of input-dtype values for the second (about 16 mantissa bits), so the rotated values
        carry fp32-like precision. Pass 1 takes the row amax of the rotated values, pass 2 recomputes the rotation
        (x comes back from L2) and rounds to int8: the rotated activations only ever live in registers.
        """
        pid = tl.program_id(0).to(tl.int64)
        rows = pid * ROWS + tl.arange(0, ROWS)
        row_mask = rows < M
        i = tl.arange(0, SUB)
        idx3 = i[None, :, None] * SUB + i[None, None, :]  # [1, SUB, SUB] offsets inside a group
        mask3 = row_mask[:, None, None]
        # H_sub replicated over the batch dim (stride 0): the same 2*SUB*SUB bytes for every row, L1-resident
        hb = tl.load(h_ptr + tl.zeros([ROWS, 1, 1], dtype=tl.int32) + idx3)
        x_rows = x_ptr + rows[:, None, None] * stride_xm + idx3
        in_dt = x_ptr.dtype.element_ty
        amax = tl.zeros([ROWS], dtype=tl.float32)
        for g in range(0, K, SUB * SUB):
            xt = tl.load(x_rows + g, mask=mask3, other=0.0)
            t = tl.dot(xt, hb)  # X @ H_sub, fp32
            t_hi = t.to(in_dt)
            t_lo = (t - t_hi.to(tl.float32)).to(in_dt)
            y = tl.dot(hb, t_lo, tl.dot(hb, t_hi))  # H_sub @ (X @ H_sub)
            amax = tl.maximum(amax, tl.max(tl.max(tl.abs(y), axis=2), axis=1))
        amax = tl.maximum(amax, 1e-12)
        inv = _libdevice.div_rn(127.0, amax)
        q_rows = q_ptr + rows[:, None, None] * stride_qm + idx3
        for g in range(0, K, SUB * SUB):
            xt = tl.load(x_rows + g, mask=mask3, other=0.0)
            t = tl.dot(xt, hb)
            t_hi = t.to(in_dt)
            t_lo = (t - t_hi.to(tl.float32)).to(in_dt)
            y = tl.dot(hb, t_lo, tl.dot(hb, t_hi))
            q = _libdevice.float2int_rn(y * inv[:, None, None]).to(tl.int8)
            tl.store(q_rows + g, q, mask=mask3)
        tl.store(s_ptr + rows, amax * (1.0 / 127.0), mask=row_mask)

    # token rows per program for the fused rotate+quantize kernel: the batch dim of the dots can be small, and
    # small tiles keep pass 2's re-read of x inside L2 and give the SMs ~1000 programs for a 4096-token call
    _ROT_QUANT_ROWS = 4

    def _rotate_quantize_act_int8_triton(x: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(f"fused rotate+quantize needs a bf16/fp16 input, got {x.dtype}")
        sub = _hadamard_sub(group)
        if sub is None or x.shape[1] % group != 0:
            raise ValueError(f"fused rotate+quantize needs K % group == 0 and group in {{256, 4096}}, "
                             f"got K={x.shape[1]} group={group}")
        if x.stride(1) != 1:
            x = x.contiguous()
        m, k = x.shape
        q = torch.empty((m, k), dtype=torch.int8, device=x.device)
        scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
        if m == 0:
            return q, scale
        h = hadamard_on(sub, x.device, x.dtype)
        rows = _ROT_QUANT_ROWS
        with _on_device(x.device):
            _rot_quant_act_kernel[(_ceil_div(m, rows),)](x, h, q, scale, m, k, x.stride(0), q.stride(0),
                                                         ROWS=rows, SUB=sub, num_warps=4 if k <= 4096 else 8,
                                                         num_stages=1)
        return q, scale


def _disable_triton(stage: str, exc: Exception) -> None:
    global USE_TRITON
    if USE_TRITON:
        USE_TRITON = False
        log.warning("triton %s kernel failed (%s: %s); W8A8 falls back to eager elementwise kernels",
                    stage, type(exc).__name__, exc)


def quantize_act_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 of a [M, K] float tensor: returns (int8 [M, K], fp32 scale [M, 1]).

    Fused triton kernel on CUDA when available (one read of x, one write of the int8), else the eager
    two-pass version. Both round to nearest even; the scale is amax / 127 in fp32.
    """
    if USE_TRITON and x.is_cuda:
        try:
            return _quantize_act_int8_triton(x)
        except Exception as e:  # noqa: BLE001
            _disable_triton("quantize_act_int8", e)
    return _quantize_act_int8_eager(x)


def _rescale(y_i32: torch.Tensor, row_scale: torch.Tensor, col_scale: torch.Tensor | None, dtype) -> torch.Tensor:
    """(y_i32 * row_scale[M,1] * col_scale[1,N]) -> dtype. Fused triton kernel on CUDA when available (one pass,
    single rounding to `dtype`), else the eager version (int32 read once, two roundings when col_scale is given)."""
    if USE_TRITON and y_i32.is_cuda:
        try:
            return _rescale_triton(y_i32, row_scale, col_scale, dtype)
        except Exception as e:  # noqa: BLE001
            _disable_triton("rescale", e)
    return _rescale_eager(y_i32, row_scale, col_scale, dtype)


def rotate_quantize_act_int8(x: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """quantize_act_int8(rotate_act(x, group)) for int8_convrot layers: returns (int8 [M, K], fp32 scale [M, 1]).

    On CUDA with triton and a bf16/fp16 input the rotation is folded into the quantization kernel (one read of
    x, one int8 write; the rotated tensor is never materialized). That kernel keeps the rotated values in fp32
    where the separate path rounds them to the input dtype first, so int8 values can differ by 1 LSB from the
    two-step result (the fused one is the closer to the exact rotation). Other inputs take the two-step path.
    """
    fusable = x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and x.shape[1] % group == 0
    if USE_TRITON and fusable and _hadamard_sub(group) is not None:
        try:
            return _rotate_quantize_act_int8_triton(x, group)
        except Exception as e:  # noqa: BLE001
            _disable_triton("rotate_quantize_act_int8", e)
    return quantize_act_int8(rotate_act(x, group))


def _int_mm_padded(a_i8: torch.Tensor, b_i8: torch.Tensor) -> torch.Tensor:
    """torch._int_mm with the M<=16 restriction worked around by zero-padding rows.

    b_i8 must be a transposed view of a row-major tensor ([K, N] seen through .t()): that is the layout the
    cuBLAS int8 kernel is fast for; a row-major [K, N] mat2 is 3-4x slower.
    """
    m = a_i8.shape[0]
    if m <= 16:
        a_i8 = nn.functional.pad(a_i8, (0, 0, 0, 17 - m))
        return torch._int_mm(a_i8, b_i8)[:m]
    return torch._int_mm(a_i8, b_i8)


def w8a8_supported(x: torch.Tensor, qw: "QuantWeight") -> bool:
    if qw.fmt not in ("int8", "int8_convrot") or not x.is_cuda or not qw.qdata.is_cuda:
        return False
    n, k = qw.qdata.shape
    return k % 8 == 0 and n % 8 == 0 and x.shape[-1] == k


def _warn_w8a8_fallback(stage: str, exc: Exception) -> None:
    global _W8A8_WARNED
    if not _W8A8_WARNED:
        _W8A8_WARNED = True
        log.warning("int8 matmul %s failed (%s: %s); falling back to dequantized matmul for this call",
                    stage, type(exc).__name__, exc)


def w8a8_forward(x2d: torch.Tensor, qw: "QuantWeight", bias: torch.Tensor | None) -> torch.Tensor:
    """x2d [M, K] (any float dtype) -> [M, N] in x2d.dtype using torch._int_mm on the stored int8 weight."""
    if qw.fmt == "int8_convrot":
        a_i8, a_scale = rotate_quantize_act_int8(x2d, qw.group)  # one fused kernel when triton is available
    else:
        a_i8, a_scale = quantize_act_int8(x2d)
    y = _int_mm_padded(a_i8, qw.qdata.t())  # int32 [M, N]
    y = _rescale(y, a_scale, qw.scale.t(), x2d.dtype)
    if bias is not None:
        y.add_(bias.to(y.dtype))
    return y


def w8a8_backward_x(grad2d: torch.Tensor, qw: "QuantWeight", mode: str) -> torch.Tensor:
    """grad wrt x for y = x W^T with W the (rotated) int8 weight; grad2d is [M, N]. Returns [M, K] in grad dtype."""
    if mode == "int8":
        g = grad2d * qw.scale.t().to(grad2d.dtype)  # fold the per-output-row weight scale into grad_out
        g_i8, g_scale = quantize_act_int8(g)
        w_nk = qw.qdata.t().contiguous().t()  # logical [N, K] in the layout the int8 kernel is fast for
        gx = _rescale(_int_mm_padded(g_i8, w_nk), g_scale, None, grad2d.dtype)  # [M, K], rotated basis
    else:  # exact: widen the stored (rotated) weight to the grad dtype; no inverse rotation of the weight
        w_rot = torch.empty(qw.qdata.shape, dtype=grad2d.dtype, device=grad2d.device)
        torch.mul(qw.qdata, qw.scale, out=w_rot)
        gx = grad2d @ w_rot
    if qw.fmt == "int8_convrot":
        gx = rotate_act(gx, qw.group)  # inverse Hadamard: the forward rotated x, so grad_x = grad_xrot @ H
    return gx


class _W8A8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, qw: QuantWeight, backward_mode: str):
        ctx.qw = qw
        ctx.backward_mode = backward_mode
        x2d = x.reshape(-1, x.shape[-1])
        try:
            y = w8a8_forward(x2d, qw, bias)
        except Exception as e:  # noqa: BLE001  (kernel rejected the shape/layout, OOM in the int32 buffer, ...)
            _warn_w8a8_fallback("forward", e)
            y = nn.functional.linear(x2d, qw.dequantize(x.dtype), bias)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, grad_out):
        qw = ctx.qw
        g2d = grad_out.reshape(-1, grad_out.shape[-1])
        try:
            gx = w8a8_backward_x(g2d, qw, ctx.backward_mode)
        except Exception as e:  # noqa: BLE001
            _warn_w8a8_fallback("backward", e)
            gx = g2d @ qw.dequantize(grad_out.dtype)
        return gx.reshape(*grad_out.shape[:-1], gx.shape[-1]), None, None, None


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
    """Frozen Linear whose weight is stored quantized.

    int8 / int8_convrot: W8A8 via torch._int_mm when `int8_matmul` is on (default from set_int8_defaults);
    `int8_backward` picks the exact bf16 backward or the int8 one. nvfp4 (and CPU / rejected shapes):
    the weight is dequantized per forward and again per backward.
    """

    def __init__(self, qw: QuantWeight, bias: torch.Tensor | None, compute_dtype=torch.bfloat16,
                 int8_matmul: bool | None = None, int8_backward: str | None = None):
        super().__init__()
        self.fmt = qw.fmt
        self.group = qw.group
        self.orig_shape = tuple(qw.orig_shape)
        self.out_features, self.in_features = self.orig_shape
        self.compute_dtype = compute_dtype
        self.int8_matmul = DEFAULT_INT8_MATMUL if int8_matmul is None else bool(int8_matmul)
        self.int8_backward = DEFAULT_INT8_BACKWARD if int8_backward is None else int8_backward
        if self.int8_backward not in INT8_BACKWARDS:
            raise ValueError(f"int8_backward must be one of {INT8_BACKWARDS}, got {self.int8_backward!r}")
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

    def uses_w8a8(self, x: torch.Tensor) -> bool:
        return self.int8_matmul and w8a8_supported(x, self.qw)

    def forward(self, x):
        qw = self.qw
        if self.int8_matmul and w8a8_supported(x, qw):
            return _W8A8LinearFn.apply(x, self.bias, qw, self.int8_backward)
        return _QuantLinearFn.apply(x, self.bias, qw)

    def extra_repr(self):
        mm = f", w8a8(backward={self.int8_backward})" if self.int8_matmul and self.fmt != "nvfp4" else ""
        return f"in={self.in_features}, out={self.out_features}, fmt={self.fmt}, bias={self.bias is not None}{mm}"


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
