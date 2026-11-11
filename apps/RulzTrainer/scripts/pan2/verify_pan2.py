"""Check a file made by convert_pan2.py against the IFM/PAN2 source folder it came from.

  python verify_pan2.py <source folder> <converted file>
  e.g. python verify_pan2.py D:/PAN2/transformer D:/ComfyUI/models/diffusion_models/PAN2_nvfp4.safetensors

Reports: whether every tensor name is present, whether every copied tensor is bit-identical, and the relative
error of quantized layers after turning them back into ordinary weights (all of them with --all, else 24 at random).
Typical errors: int8 ConvRot about 0.009, NVFP4 about 0.09. For the fp16 VAE every tensor is compared with the cast.
"""
import argparse
import glob
import json
import os
import random
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # convert_pan2.py sits next to this file
from convert_pan2 import GROUP, dequantize_nvfp4, hadamard

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--all", action="store_true", help="check every quantized layer instead of a sample")
args = ap.parse_args()

where = {}
for p in sorted(glob.glob(os.path.join(args.src, "*.safetensors"))):
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            where[k] = p
handles = {}


def source(k):
    p = where[k]
    if p not in handles:
        handles[p] = safe_open(p, framework="pt")
    return handles[p].get_tensor(k)


out = safe_open(args.dst, framework="pt")
keys = set(out.keys())
side = (".comfy_quant", ".weight_scale", ".weight_scale_2")
plain = {k for k in keys if not k.endswith(side)}
quant = sorted(k[: -len(".comfy_quant")] for k in keys if k.endswith(".comfy_quant"))
print(f"tensors in file: {len(keys)} | source tensors: {len(where)} | names match: {plain == set(where)}")
print(f"quantized layers: {len(quant)}")

half = (out.metadata() or {}).get("precision", "").startswith("fp16")
bad = checked = 0
for k in sorted(plain):
    if k.endswith(".weight") and k[: -len(".weight")] in quant:
        continue
    a, b = out.get_tensor(k), source(k)
    if half:
        b = b.to(torch.float16)
    checked += 1
    if a.dtype != b.dtype or not torch.equal(a, b):
        bad += 1
        print("  MISMATCH", k)
print(f"{'fp16 cast' if half else 'copied'} tensors checked: {checked} | mismatches: {bad}")

if quant:
    h = hadamard(GROUP)
    random.seed(0)
    sample = quant if args.all or len(quant) <= 24 else random.sample(quant, 24)
    kinds, errs = {}, []
    for base in quant:
        m = bytes(out.get_tensor(base + ".comfy_quant").tolist()).decode()
        kinds[m] = kinds.get(m, 0) + 1
    print("formats:", kinds)
    for base in sample:
        meta = json.loads(bytes(out.get_tensor(base + ".comfy_quant").tolist()).decode())
        w, s = out.get_tensor(base + ".weight"), out.get_tensor(base + ".weight_scale")
        ref = source(base + ".weight").float()
        if meta["format"] == "nvfp4":
            deq = dequantize_nvfp4(w, s, out.get_tensor(base + ".weight_scale_2"), ref.shape)
        else:
            deq = w.float() * s
            if meta.get("convrot"):  # the block Hadamard matrix is its own inverse
                deq = (deq.reshape(deq.shape[0], -1, GROUP) @ h.T).reshape(deq.shape)
        errs.append(((deq - ref).norm() / ref.norm()).item())
        assert torch.isfinite(deq).all(), base
    print(f"quantized layers turned back into weights ({len(sample)} checked): "
          f"mean relative error {sum(errs) / len(errs):.4f}, max {max(errs):.4f}")
print("OK" if bad == 0 and plain == set(where) else "PROBLEMS FOUND")
