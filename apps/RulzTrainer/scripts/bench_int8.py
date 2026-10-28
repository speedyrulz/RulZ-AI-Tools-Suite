"""Micro-benchmark: bf16 Linear vs W8A8 (torch._int_mm) vs dequantize-per-call for a MOSS-sized layer.

    venv\\Scripts\\python.exe scripts\\bench_int8.py [--m 4096] [--k 3072] [--n 8192] [--iters 20]

Prints forward and forward+backward milliseconds on every CUDA device for:
  bf16        plain nn.functional.linear on a bf16 weight (reference speed)
  w8a8/bf16   QuantLinear int8_convrot, int8 matmul on, exact bf16 backward (default)
  w8a8/int8   QuantLinear int8_convrot, int8 matmul on, int8 backward
  dequant     QuantLinear int8_convrot with int8_matmul off (weight dequantized every forward and backward)
  int_mm      torch._int_mm alone (no quantization / rescale), the kernel's ceiling
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import quant as Q  # noqa: E402


def timeit(fn, iters: int, device) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / iters * 1000.0


def bench_device(device: str, m: int, k: int, n: int, iters: int, fmt: str) -> None:
    torch.manual_seed(0)
    w = (torch.randn(n, k) * 0.02).to(device, torch.bfloat16)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    qw = Q.quantize_weight(w, fmt)
    ql = Q.QuantLinear(qw, None, torch.bfloat16).to(device)
    a_i8 = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=device)
    w_t = qw.qdata.t()

    def fwd_bwd(f):
        def run():
            y = f(x)
            y.backward(g)
            x.grad = None
        return run

    def with_opts(matmul: bool, backward: str):
        def f(inp):
            ql.int8_matmul = matmul
            ql.int8_backward = backward
            return ql(inp)
        return f

    rows = [
        ("bf16", lambda: torch.nn.functional.linear(x, w), fwd_bwd(lambda t: torch.nn.functional.linear(t, w))),
        ("w8a8/bf16", lambda: with_opts(True, "bf16")(x), fwd_bwd(with_opts(True, "bf16"))),
        ("w8a8/int8", lambda: with_opts(True, "int8")(x), fwd_bwd(with_opts(True, "int8"))),
        ("dequant", lambda: with_opts(False, "bf16")(x), fwd_bwd(with_opts(False, "bf16"))),
        ("int_mm", lambda: torch._int_mm(a_i8, w_t), None),
    ]
    name = torch.cuda.get_device_name(device)
    print(f"\n{device} ({name})  M={m} K={k} N={n} fmt={fmt}  ({iters} iters)")
    print(f"  {'path':<11}{'forward ms':>12}{'fwd+bwd ms':>12}")
    for label, f_fwd, f_fb in rows:
        fwd = timeit(lambda: f_fwd(), iters, device)
        fb = timeit(f_fb, iters, device) if f_fb is not None else float("nan")
        print(f"  {label:<11}{fwd:>12.2f}{fb:>12.2f}")
    del w, x, g, qw, ql, a_i8, w_t
    torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=4096, help="tokens (rows of x)")
    ap.add_argument("--k", type=int, default=3072, help="in_features")
    ap.add_argument("--n", type=int, default=8192, help="out_features")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--fmt", default="int8_convrot", choices=["int8", "int8_convrot"])
    ap.add_argument("--devices", default="", help="comma list, default: every CUDA device")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is required")
        return 1
    devices = args.devices.split(",") if args.devices else [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print(f"torch {torch.__version__}")
    for dev in devices:
        bench_device(dev.strip(), args.m, args.k, args.n, args.iters, args.fmt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
