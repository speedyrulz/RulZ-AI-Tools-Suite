"""Convert IFM/PAN2 (https://huggingface.co/IFM/PAN2) into single-file models for ComfyUI.

Makes up to six files under --out:

  diffusion_models/PAN2_int8_convrot.safetensors        39.8 GiB  int8 ConvRot, the fp32 norm layers kept as fp32
  diffusion_models/PAN2_int8_convrot_full.safetensors   27.4 GiB  int8 ConvRot, those layers quantized as well
  diffusion_models/PAN2_nvfp4.safetensors               15.5 GiB  NVFP4 (4-bit)
  text_encoders/PAN2_qwen3_5_int8_convrot.safetensors    8.4 GiB  Qwen3.5-9B, int8 ConvRot, embedding table bf16
  text_encoders/PAN2_qwen3_5_nvfp4.safetensors           4.6 GiB  Qwen3.5-9B, NVFP4, embedding table plain int8
  vae/PAN2_vae_fp16.safetensors                          1.3 GiB  the fp32 VAE cast to fp16

Usage:
  python convert_pan2.py --out D:/ComfyUI/models                       # download what is needed, make all six
  python convert_pan2.py --out D:/ComfyUI/models --only transformer_nvfp4 te_nvfp4 vae_fp16
  python convert_pan2.py --out D:/ComfyUI/models --src D:/PAN2          # a local copy with transformer/, text_encoder/, vae/

Needs: torch, safetensors, numpy, and huggingface_hub when --src is not given.

Formats written (what ComfyUI's quantized-weight loader reads):
  int8 ConvRot  <layer>.weight        int8 [out, in] = round((W @ H) / scale), H = block Hadamard of size 256
                <layer>.weight_scale  fp32 [out, 1], one scale per output row
                <layer>.comfy_quant   json {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
  NVFP4         <layer>.weight        uint8 [out, in/2], two 4-bit e2m1 codes per byte
                <layer>.weight_scale  fp8 e4m3 [out, in/16], one scale per block of 16, in cuBLAS block order
                <layer>.weight_scale_2  fp32 scalar
                <layer>.comfy_quant   json {"format": "nvfp4"}
Every other tensor is copied unchanged. Files are written tensor by tensor, one tensor at a time.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import struct
import time

import torch
from safetensors import safe_open

REPO = "IFM/PAN2"
GROUP = 256
ST_DTYPE = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.int8: "I8", torch.uint8: "U8"}
ITEM = {"F8_E4M3": 1, "F32": 4, "BF16": 2, "F16": 2, "I8": 1, "U8": 1, "F64": 8, "I64": 8, "I32": 4}
F4_E2M1_MAX, F8_E4M3_MAX = 6.0, 448.0
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

TRANSFORMER = {"folder": "transformer", "source": "IFM/PAN2 transformer", "prefixes": ("transformer_blocks.", "context_refiner.")}
TEXT_ENCODER = {"folder": "text_encoder", "source": "IFM/PAN2 text_encoder", "prefixes": ("layers.",)}
TARGETS = {
    "transformer_int8_convrot": dict(TRANSFORMER, out="diffusion_models/PAN2_int8_convrot.safetensors", format="int8_convrot"),
    "transformer_int8_convrot_full": dict(TRANSFORMER, out="diffusion_models/PAN2_int8_convrot_full.safetensors", format="int8_convrot", quant_f32=True),
    "transformer_nvfp4": dict(TRANSFORMER, out="diffusion_models/PAN2_nvfp4.safetensors", format="nvfp4", quant_f32=True),
    "te_int8_convrot": dict(TEXT_ENCODER, out="text_encoders/PAN2_qwen3_5_int8_convrot.safetensors", format="int8_convrot"),
    # the text encoder's matrix products stay in full precision, and its embedding table (a lookup, not a product)
    # is stored as plain row-wise int8
    "te_nvfp4": dict(TEXT_ENCODER, out="text_encoders/PAN2_qwen3_5_nvfp4.safetensors", format="nvfp4",
                     embed_int8="embed_tokens.weight", full_precision_mm=True),
    "vae_fp16": {"folder": "vae", "out": "vae/PAN2_vae_fp16.safetensors", "format": "fp16"},
}


# ---------------------------------------------------------------- int8 ConvRot

def hadamard(size):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h, cur = h4, 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def quantize_int8_convrot(w, h):
    w = w.float()
    out_f, in_f = w.shape
    rot = (w.reshape(out_f, in_f // GROUP, GROUP) @ h.T).reshape(out_f, in_f)
    scale = (rot.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-12)
    q = torch.round(rot / scale).clamp(-128, 127).to(torch.int8)
    err = ((q.float() * scale - rot).norm() / rot.norm().clamp(min=1e-30)).item()
    return q, scale, err


# ---------------------------------------------------------------- NVFP4

def _ceil_div(a, b):
    return -(-a // b)


def to_blocked(m):
    """Block scales in the order cuBLAS expects (128x4 tiles, each stored as 32x16)."""
    rows, cols = m.shape
    nr, nc = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pr, pc = nr * 128, nc * 4
    if (rows, cols) != (pr, pc):
        padded = torch.zeros((pr, pc), device=m.device, dtype=m.dtype)
        padded[:rows, :cols] = m
        m = padded
    blocks = m.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).reshape(pr, pc)


def from_blocked(b, rows, cols):
    nr, nc = _ceil_div(rows, 128), _ceil_div(cols, 4)
    pr, pc = nr * 128, nc * 4
    s = b.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(1, 2)
    s = s.reshape(nr, nc, 4, 32, 4).reshape(nr, nc, 128, 4).permute(0, 2, 1, 3)
    return s.reshape(pr, pc)[:rows, :cols]


def _e2m1_codes(x):
    """fp32 values already scaled to [-6, 6] -> 4-bit codes (sign in bit 3), round to nearest, ties to even."""
    sign = (x < 0).to(torch.uint8)
    a = x.abs()
    vals = E2M1_VALUES.to(x.device)
    mids = (vals[:-1] + vals[1:]) / 2
    code = torch.bucketize(a, mids, right=False).to(torch.uint8)
    tie = torch.isin(a, mids)
    code = torch.where(tie & (code % 2 == 1), code + 1, code).clamp(max=7)
    return code | (sign << 3)


def quantize_nvfp4(w):
    """Returns (packed uint8 [rows, cols/2], fp8 block scales in cuBLAS order, fp32 per-tensor scale)."""
    w = w.float()
    rows, cols = w.shape
    pr, pc = _ceil_div(rows, 16) * 16, _ceil_div(cols, 16) * 16
    if (pr, pc) != (rows, cols):
        w = torch.nn.functional.pad(w, (0, pc - cols, 0, pr - rows))
    per_tensor = (w.abs().amax() / (F8_E4M3_MAX * F4_E2M1_MAX)).to(torch.float32)
    x = w.reshape(pr, -1, 16)
    block = x.abs().amax(dim=-1) / F4_E2M1_MAX
    scaled_fp8 = (block / per_tensor).clamp(max=F8_E4M3_MAX).to(torch.float8_e4m3fn)
    total = per_tensor * scaled_fp8.float()
    safe = torch.where(total == 0, torch.ones_like(total), total)
    data = (x / safe.unsqueeze(-1)).clamp(-F4_E2M1_MAX, F4_E2M1_MAX)
    data = torch.where((total == 0).unsqueeze(-1), torch.zeros_like(data), data).reshape(pr, pc)
    codes = _e2m1_codes(data).reshape(-1)
    packed = (codes[::2] << 4 | codes[1::2]).reshape(pr, pc // 2)
    return packed, to_blocked(scaled_fp8), per_tensor


def dequantize_nvfp4(packed, block_scales, per_tensor, shape):
    pr, pc = packed.shape[0], packed.shape[1] * 2
    lut = torch.cat([E2M1_VALUES, -E2M1_VALUES]).to(packed.device)
    codes = torch.stack([packed >> 4, packed & 0x0F], dim=-1).reshape(pr, pc)
    vals = lut[codes.int()].reshape(pr, pc // 16, 16)
    scales = from_blocked(block_scales, pr, pc // 16).float() * per_tensor.float()
    return (vals * scales.unsqueeze(-1)).reshape(pr, pc)[: shape[0], : shape[1]]


# ---------------------------------------------------------------- converters

def eligible(name, dtype, shape, quant_f32, prefixes):
    """Large 2-D weights of the blocks. The transformer keeps some of them in fp32 (the norm1.linear layers)."""
    if not name.endswith(".weight") or len(shape) != 2 or min(shape) < 1024 or shape[1] % GROUP:
        return False
    if not name.startswith(prefixes):
        return False
    return dtype == "BF16" or (quant_f32 and dtype == "F32")


def convert_quantized(src, dst, fmt, prefixes, source, quant_f32=False, embed_int8=None, full_precision_mm=False,
                      device="cpu", **_):
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    assert shards, f"no .safetensors files in {src}"
    where, info = {}, {}
    for p in shards:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        head.pop("__metadata__", None)
        for k, v in head.items():
            assert k not in where, k
            where[k], info[k] = p, (v["dtype"], v["shape"])
    print(f"{len(shards)} source files, {len(info)} tensors")

    fp4 = fmt == "nvfp4"
    if fp4:
        cq = json.dumps({"format": "nvfp4", **({"full_precision_matrix_mult": True} if full_precision_mm else {})}).encode()
    else:
        cq = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}).encode()
    cq_embed = json.dumps({"format": "int8_tensorwise"}).encode()
    plan = []  # (name written, dtype, shape, source key, part)
    for k in sorted(info):
        dt, shape = info[k]
        base = k[: -len(".weight")]
        if k == embed_int8:
            plan.append((base + ".comfy_quant", "U8", [len(cq_embed)], k, "cqe"))
            plan.append((k, "I8", shape, k, "qe"))
            plan.append((base + ".weight_scale", "F32", [shape[0], 1], k, "scale"))
        elif eligible(k, dt, shape, quant_f32, prefixes):
            plan.append((base + ".comfy_quant", "U8", [len(cq)], k, "cq"))
            if fp4:
                assert shape[0] % 128 == 0 and shape[1] % 64 == 0, (k, shape)
                plan.append((k, "U8", [shape[0], shape[1] // 2], k, "q"))
                plan.append((base + ".weight_scale", "F8_E4M3", [shape[0], shape[1] // 16], k, "scale"))
                plan.append((base + ".weight_scale_2", "F32", [], k, "scale2"))
            else:
                plan.append((k, "I8", shape, k, "q"))
                plan.append((base + ".weight_scale", "F32", [shape[0], 1], k, "scale"))
        else:
            plan.append((k, dt, shape, k, "copy"))

    header, off = {}, 0
    for name, dt, shape, _, _ in plan:
        size = math.prod(shape) * ITEM[dt]
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + size]}
        off += size
    meta = {"format": "pt", "source": source,
            "quantization": ("nvfp4 (e2m1, fp8 block scales of 16, per-tensor scale)" if fp4 else
                             "int8_tensorwise convrot groupsize 256 (per-row scale)") + "; other tensors unchanged"}
    cfg_path = os.path.join(src, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            meta["config"] = json.dumps(json.load(f))
    hj = json.dumps({"__metadata__": meta, **header}, separators=(",", ":")).encode()
    hj += b" " * (-(8 + len(hj)) % 8)
    nq = sum(1 for p in plan if p[4] == "q")
    print(f"quantizing {nq} weights, copying {sum(1 for p in plan if p[4] == 'copy')}; "
          f"output {(8 + len(hj) + off) / 2**30:.2f} GiB")

    h = hadamard(GROUP).to(device)
    handles, stats, cache = {}, collections.defaultdict(list), {}
    t0, done = time.time(), 0
    tmp = dst + ".part"
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hj)))
        out.write(hj)
        base_off = out.tell()
        for name, dt, shape, key, part in plan:
            assert out.tell() - base_off == header[name]["data_offsets"][0], name
            if part in ("cq", "cqe"):
                out.write(cq if part == "cq" else cq_embed)
                continue
            p = where[key]
            if p not in handles:
                handles[p] = safe_open(p, framework="pt")
            if part == "copy":
                t = handles[p].get_tensor(key)
                assert ST_DTYPE[t.dtype] == dt and list(t.shape) == shape, name
            elif part == "qe":
                w = handles[p].get_tensor(key).float()
                scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-12)
                t = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
                stats[key + " (plain int8)"].append(((t.float() * scale - w).norm() / w.norm()).item())
                cache[key] = [scale]
                del w
            elif part == "q":
                w = handles[p].get_tensor(key).to(device)
                if fp4:
                    q, bs, s2 = quantize_nvfp4(w)
                    ref = w.float()
                    err = ((dequantize_nvfp4(q, bs, s2, ref.shape) - ref).norm() / ref.norm().clamp(min=1e-30)).item()
                    del ref
                    cache[key] = [bs.cpu(), s2.reshape(()).cpu()]
                else:
                    q, scale, err = quantize_int8_convrot(w, h)
                    cache[key] = [scale.cpu()]
                stats[re.sub(r"^(\w+)\.\d+\.", r"\1.N.", key)].append(err)
                t = q.cpu()
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{nq} weights, {time.time() - t0:.0f}s", flush=True)
            else:  # the scales of the weight just written
                t = cache[key].pop(0)
                if not cache[key]:
                    del cache[key]
            if t.dtype == torch.bfloat16:
                b = t.contiguous().view(torch.int16).numpy().tobytes()
            elif t.dtype == torch.float8_e4m3fn:
                b = t.contiguous().view(torch.uint8).numpy().tobytes()
            else:
                b = t.contiguous().numpy().tobytes()
            assert len(b) == math.prod(shape) * ITEM[dt], name
            out.write(b)
        assert out.tell() - base_off == off
    os.replace(tmp, dst)
    print(f"wrote {dst} ({os.path.getsize(dst) / 2**30:.2f} GiB) in {time.time() - t0:.0f}s")
    print("relative weight error per layer kind (0.01 = 1%):")
    for g, v in sorted(stats.items()):
        e = torch.tensor(v)
        print(f"  {g:55s} n={len(v):3d} mean {e.mean():.4f} max {e.max():.4f}")


def convert_vae(src, dst, **_):
    path = os.path.join(src, "diffusion_pytorch_model.safetensors")
    sd = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            assert t.dtype == torch.float32, (k, t.dtype)
            h = t.to(torch.float16)
            assert torch.isfinite(h).all(), f"{k} does not fit fp16"
            sd[k] = h
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        meta = {"format": "pt", "source": "IFM/PAN2 vae", "precision": "fp16 (cast from fp32)", "config": json.dumps(json.load(f))}
    # written by hand, not with safetensors.save_file: that one stores the metadata entries in a random order, so
    # two runs would not give the same bytes
    header, off = {}, 0
    for k in sorted(sd):
        size = sd[k].numel() * 2
        header[k] = {"dtype": "F16", "shape": list(sd[k].shape), "data_offsets": [off, off + size]}
        off += size
    hj = json.dumps({"__metadata__": meta, **header}, separators=(",", ":")).encode()
    hj += b" " * (-(8 + len(hj)) % 8)
    with open(dst + ".part", "wb") as out:
        out.write(struct.pack("<Q", len(hj)))
        out.write(hj)
        for k in sorted(sd):
            out.write(sd[k].contiguous().numpy().tobytes())
    os.replace(dst + ".part", dst)
    print(f"wrote {dst} ({os.path.getsize(dst) / 2**30:.2f} GiB), {len(sd)} tensors")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output folder, e.g. your ComfyUI models folder")
    ap.add_argument("--only", nargs="+", choices=list(TARGETS), help="make only these (default: all six)")
    ap.add_argument("--src", help="local copy of IFM/PAN2 with transformer/, text_encoder/ and vae/ (default: download)")
    ap.add_argument("--device", default="cpu", help="where the quantization math runs. cpu reproduces the reference files "
                                                    "exactly; cuda is faster and can differ in a few rounded values")
    ap.add_argument("--overwrite", action="store_true", help="redo files that already exist")
    args = ap.parse_args()

    names = args.only or list(TARGETS)
    src = args.src
    if src is None:
        from huggingface_hub import snapshot_download
        folders = sorted({TARGETS[n]["folder"] for n in names})
        print(f"downloading {', '.join(folders)} from {REPO} (transformer 63 GiB, text_encoder 15 GiB, vae 2.7 GiB)")
        src = snapshot_download(REPO, allow_patterns=[f"{f}/*" for f in folders])
    for n in names:
        t = TARGETS[n]
        dst = os.path.join(args.out, t["out"])
        if os.path.exists(dst) and not args.overwrite:
            print(f"== {n}: {dst} exists, skipped")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"== {n} -> {dst}")
        folder = os.path.join(src, t["folder"])
        if t["format"] == "fp16":
            convert_vae(folder, dst)
        else:
            convert_quantized(folder, dst, t["format"], device=args.device, **{k: v for k, v in t.items() if k != "format"})


if __name__ == "__main__":
    main()
