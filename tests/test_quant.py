import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import quant as Q  # noqa: E402
from rulztrainer import lora as lora_mod  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402


def test_blocked_roundtrip():
    for rows, cols in [(3072, 192), (100, 7), (128, 4)]:
        m = torch.randn(rows, cols).to(torch.float8_e4m3fn)
        assert torch.equal(Q.from_blocked(Q.to_blocked(m), rows, cols).float(), m.float())


def test_int8_convrot_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(512, 1024) * 0.02
    for fmt in ("int8", "int8_convrot"):
        qw = Q.quantize_weight(w, fmt)
        assert qw.fmt == fmt
        err = (qw.dequantize(torch.float32) - w).abs().mean() / w.abs().mean()
        assert err < 0.02, (fmt, err)


def test_nvfp4_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(300, 1000) * 0.02  # non-multiples of 16 exercise padding
    qw = Q.quantize_weight(w, "nvfp4")
    assert qw.qdata.shape == (304, 504) and qw.qdata.dtype == torch.uint8
    d = qw.dequantize(torch.float32)
    assert d.shape == w.shape
    err = (d - w).abs().mean() / w.abs().mean()
    assert err < 0.15, err
    # exact codes must be representable: requantizing a dequantized tensor is a fixed point
    qw2 = Q.quantize_weight(d, "nvfp4")
    assert (qw2.dequantize(torch.float32) - d).abs().max() < 1e-6


def test_quant_linear_forward_backward_matches_dense():
    torch.manual_seed(0)
    lin = torch.nn.Linear(1024, 512, bias=True).to(torch.bfloat16)
    x = torch.randn(4, 1024, dtype=torch.bfloat16, requires_grad=True)
    ref = lin(x)
    for fmt in ("int8_convrot", "nvfp4"):
        ql = Q.QuantLinear(Q.quantize_weight(lin.weight.data, fmt), lin.bias.data, torch.bfloat16)
        out = ql(x)
        rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
        assert rel < (0.03 if fmt.startswith("int8") else 0.15), (fmt, rel)
        out.float().sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        x.grad = None


def test_quantized_model_trains_lora():
    torch.manual_seed(0)
    tiny = dict(dim=1024, ffn_dim=1024, num_heads=8, num_layers=2, text_dim=32, freq_dim=16, latent_channels=128)
    model = MOSSImageTransformer(**tiny).to(torch.bfloat16)
    n = Q.quantize_module_linears(model, "int8_convrot", min_dim=1024)
    assert n > 0
    injected = lora_mod.inject_lora(model, 4, 4, ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
    assert all(isinstance(m.base, Q.QuantLinear) for m in injected.values())
    model.train()
    out = model(torch.randn(1, 128, 4, 4).to(torch.bfloat16), torch.tensor([300.0]),
                torch.randn(1, 5, 32).to(torch.bfloat16))
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in lora_mod.lora_parameters(injected))
    # extras (turbo LoRA) only act when enabled
    m = next(iter(injected.values()))
    m.extras.append((torch.randn(2, m.in_features), torch.randn(m.out_features, 2), 1.0))
    x = torch.randn(1, m.in_features).to(torch.bfloat16)
    a = m(x)
    lora_mod.set_extras_enabled(model, True)
    b = m(x)
    assert not torch.allclose(a, b)


# ---------------------------------------------------------------- W8A8 (int8 activations x int8 weight)

import pytest  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="W8A8 needs torch._int_mm on CUDA")


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_quantize_act_int8_matches_round():
    torch.manual_seed(0)
    x = torch.randn(64, 512) * 3
    x[0] = 0  # all-zero token must not divide by zero
    x[1, 7] = 1e4  # outlier token
    q, s = Q.quantize_act_int8(x)
    amax = x.abs().amax(1, keepdim=True).clamp(min=1e-12)
    ref = torch.round(x / (amax / 127)).to(torch.int8)
    assert q.dtype == torch.int8 and s.shape == (64, 1) and s.dtype == torch.float32
    assert torch.allclose(s, amax / 127)
    assert (q.int() - ref.int()).abs().max() <= 1  # only round-half-even vs half-away ties may differ
    assert q.abs().max() <= 127 and q[0].abs().max() == 0


@cuda_only
@pytest.mark.parametrize("fmt", ["int8", "int8_convrot"])
def test_w8a8_forward_matches_dequant(fmt):
    torch.manual_seed(0)
    w = (torch.randn(8192, 3072) * 0.02).to("cuda", torch.bfloat16)
    x = torch.randn(64, 3072, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(8192, device="cuda", dtype=torch.bfloat16)
    ql = Q.QuantLinear(Q.quantize_weight(w, fmt), bias, torch.bfloat16, int8_matmul=True)
    assert ql.uses_w8a8(x)
    y = ql(x)
    ql.int8_matmul = False
    ref = ql(x)
    assert y.shape == ref.shape == (64, 8192) and y.dtype == torch.bfloat16
    assert _rel(y, ref) < 0.01, _rel(y, ref)
    # 3-D input keeps its leading dims
    assert ql(x.view(2, 32, 3072)).shape == (2, 32, 8192)


@cuda_only
@pytest.mark.parametrize("fmt", ["int8", "int8_convrot"])
def test_w8a8_backward_bf16_matches_dequant(fmt):
    torch.manual_seed(0)
    w = (torch.randn(8192, 3072) * 0.02).to("cuda", torch.bfloat16)
    x = torch.randn(64, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(64, 8192, device="cuda", dtype=torch.bfloat16)
    ql = Q.QuantLinear(Q.quantize_weight(w, fmt), None, torch.bfloat16, int8_matmul=False)
    ql(x).backward(g)
    ref = x.grad.clone()
    x.grad = None
    ql.int8_matmul, ql.int8_backward = True, "bf16"
    ql(x).backward(g)
    assert _rel(x.grad, ref) < 0.01, _rel(x.grad, ref)  # exact up to bf16 rounding (inverse rotation correct)
    x.grad = None
    ql.int8_backward = "int8"
    ql(x).backward(g)
    assert _rel(x.grad, ref) < 0.03, _rel(x.grad, ref)  # quantized grad_out: ~1% error


@cuda_only
def test_w8a8_small_m_padding():
    torch.manual_seed(0)
    w = (torch.randn(2048, 1024) * 0.02).to("cuda", torch.bfloat16)
    ql = Q.QuantLinear(Q.quantize_weight(w, "int8_convrot"), None, torch.bfloat16, int8_matmul=True)
    for m in (1, 5, 16, 17):
        x = torch.randn(m, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = ql(x)
        assert y.shape == (m, 2048)
        ql.int8_matmul = False
        ref = ql(x)
        ql.int8_matmul = True
        assert _rel(y, ref) < 0.01, (m, _rel(y, ref))
        for bw in ("bf16", "int8"):
            ql.int8_backward = bw
            ql(x).float().sum().backward()
            assert x.grad.shape == (m, 1024) and torch.isfinite(x.grad).all()
            x.grad = None


def test_w8a8_defaults_and_configure():
    w = torch.randn(1024, 1024) * 0.02
    qw = Q.quantize_weight(w, "int8")
    old = (Q.DEFAULT_INT8_MATMUL, Q.DEFAULT_INT8_BACKWARD)
    try:
        Q.set_int8_defaults(False, "int8")
        ql = Q.QuantLinear(qw, None)
        assert ql.int8_matmul is False and ql.int8_backward == "int8"
        ql2 = Q.QuantLinear(qw, None, int8_matmul=True, int8_backward="bf16")
        assert ql2.int8_matmul is True and ql2.int8_backward == "bf16"
        model = torch.nn.Sequential(ql, ql2, torch.nn.Linear(4, 4))
        assert Q.configure_int8(model, True, "bf16") == 2
        assert ql.int8_matmul and ql.int8_backward == "bf16"
        with pytest.raises(ValueError):
            Q.set_int8_defaults(True, "fp8")
        # nvfp4 and CPU tensors never take the int8 path
        assert not Q.QuantLinear(Q.quantize_weight(w, "nvfp4"), None).uses_w8a8(torch.randn(4, 1024))
        assert not ql2.uses_w8a8(torch.randn(4, 1024))
    finally:
        Q.set_int8_defaults(*old)


@cuda_only
def test_quantized_model_trains_lora_w8a8():
    torch.manual_seed(0)
    tiny = dict(dim=1024, ffn_dim=1024, num_heads=8, num_layers=2, text_dim=32, freq_dim=16, latent_channels=128)
    model = MOSSImageTransformer(**tiny).to("cuda", torch.bfloat16)
    assert Q.quantize_module_linears(model, "int8_convrot", min_dim=1024) > 0
    assert Q.configure_int8(model, True, "bf16") > 0
    model.gradient_checkpointing = True
    injected = lora_mod.inject_lora(model, 4, 4, ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
    model.train()
    out = model(torch.randn(1, 128, 4, 4, device="cuda").to(torch.bfloat16), torch.tensor([300.0], device="cuda"),
                torch.randn(1, 5, 32, device="cuda").to(torch.bfloat16))
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lora_mod.lora_parameters(injected))
    assert not Q._W8A8_WARNED


# ---------------------------------------------------------------- fused triton kernels vs eager helpers

fused_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and Q.HAS_TRITON),
                                reason="fused W8A8 kernels need triton + CUDA")
_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cuda"]


@fused_cuda
@pytest.mark.parametrize("device", _devices)
def test_fused_kernels_match_eager(device):
    torch.manual_seed(0)
    shapes = [(64, 3072), (17, 8192), (5, 1000), (3, 12000)]  # single-tile, max tile, odd K, two-pass loop
    for (m, k) in shapes:
        for dt in (torch.bfloat16, torch.float32):
            x = (torch.randn(m, k, device=device) * 3).to(dt)
            x[0] = 0
            x[1, 7] = 1e4
            q_t, s_t = Q._quantize_act_int8_triton(x)
            q_e, s_e = Q._quantize_act_int8_eager(x)
            assert q_t.dtype == torch.int8 and s_t.shape == (m, 1) and s_t.dtype == torch.float32
            assert torch.equal(s_t, s_e), (m, k, dt)
            # both round to nearest even; bf16 inputs repeat values within a row, so an exact .5 tie flips every
            # copy at once and the eager fp32 product may sit on the other side of it: never more than 1 LSB
            assert (q_t.int() - q_e.int()).abs().max() <= 1, (m, k, dt)
            assert q_t[0].abs().max() == 0 and q_t.abs().max() <= 127
        y = torch.randint(-2**20, 2**20, (m, k), dtype=torch.int32, device=device)
        rs = torch.rand(m, 1, device=device) + 0.1
        cs = torch.rand(k, 1, device=device) + 0.1  # weight scale layout [N, 1]; passed transposed like w8a8_forward
        for col in (cs.t(), None):
            # fp32 output: identical operation order, bit-exact
            assert torch.equal(Q._rescale_triton(y, rs, col, torch.float32), Q._rescale_eager(y, rs, col, torch.float32))
            out_t = Q._rescale_triton(y, rs, col, torch.bfloat16)
            out_e = Q._rescale_eager(y, rs, col, torch.bfloat16)
            ref = (y.double() * rs.double() * (col.double() if col is not None else 1.0)).to(torch.bfloat16)
            assert out_t.dtype == torch.bfloat16
            # fused: one rounding to bf16; eager: two when col_scale is given -> within one bf16 ulp of each other
            assert torch.allclose(out_t.float(), ref.float(), rtol=2 ** -7, atol=0), (m, k, col is None)
            assert torch.allclose(out_t.float(), out_e.float(), rtol=2 ** -7, atol=0), (m, k, col is None)
    # the public helpers dispatch to the fused kernels when triton is on
    assert Q.USE_TRITON
    x = torch.randn(8, 1024, device=device, dtype=torch.bfloat16)
    assert torch.equal(Q.quantize_act_int8(x)[0], Q._quantize_act_int8_triton(x)[0])


@fused_cuda
@pytest.mark.parametrize("device", _devices)
def test_fused_rotate_quant_matches_rotate_then_eager(device):
    torch.manual_seed(0)
    shapes = [(64, 3072), (17, 8192), (1, 256), (33, 1024), (300, 3072)]  # several / one / many row tiles, K = 1..32 groups
    for m, k in shapes:
        for dt in (torch.bfloat16, torch.float16):
            x = (torch.randn(m, k, device=device) * 3).to(dt)
            x[0] = 0  # all-zero token must not divide by zero
            if m > 1:
                x[1, 7] = 1e4  # one outlier: the rotation spreads it over its group
            if m > 2:
                x[2, ::97] = 300
            q_t, s_t = Q._rotate_quantize_act_int8_triton(x, 256)
            xr = Q.rotate_act(x, 256)  # two-step path: the matmul rounds the rotated values to the input dtype
            q_e, s_e = Q._quantize_act_int8_eager(xr)
            assert q_t.dtype == torch.int8 and s_t.shape == (m, 1) and s_t.dtype == torch.float32
            # the fused kernel keeps the rotated values in fp32 (hi/lo split through the second dot), the two-step
            # path rounds them to bf16 (rel 2^-8): the row amax, hence the scale, differs by up to one bf16 ulp and
            # the int8 values by at most 1 LSB (|a - b| <= 0.5 LSB before rounding)
            assert torch.allclose(s_t, s_e, rtol=2 ** -7, atol=0), (m, k, dt)
            assert (q_t.int() - q_e.int()).abs().max() <= 1, (m, k, dt)
            assert q_t[0].abs().max() == 0 and q_t.abs().max() <= 127
            # against an fp64 rotation the fused result is the closer one (~0% mismatches vs ~5% for bf16 two-step)
            xr64 = Q.rotate_act(x.double(), 256)
            q64 = torch.round(xr64 / (xr64.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127))
            assert (q_t.double() - q64).abs().max() <= 1, (m, k, dt)
            fused_mism = (q_t.double() != q64).float().mean().item()
            eager_mism = (q_e.double() != q64).float().mean().item()
            assert fused_mism <= eager_mism, (m, k, dt, fused_mism, eager_mism)
    # dispatch: bf16 CUDA input takes the fused kernel; fp32 input and a K that is not a multiple of the group
    # take the two-step path (rotate_act, then the fused quantizer)
    assert Q.USE_TRITON
    x = torch.randn(8, 1024, device=device, dtype=torch.bfloat16)
    assert torch.equal(Q.rotate_quantize_act_int8(x, 256)[0], Q._rotate_quantize_act_int8_triton(x, 256)[0])
    x32 = x.float()
    q, s = Q.rotate_quantize_act_int8(x32, 256)
    q_ref, s_ref = Q.quantize_act_int8(Q.rotate_act(x32, 256))
    assert torch.equal(q, q_ref) and torch.equal(s, s_ref)
    with pytest.raises(ValueError):
        Q._rotate_quantize_act_int8_triton(torch.randn(8, 1000, device=device, dtype=torch.bfloat16), 256)
    assert Q._hadamard_sub(256) == 16 and Q._hadamard_sub(4096) == 64
    assert Q._hadamard_sub(64) is None and Q._hadamard_sub(1024) is None and Q._hadamard_sub(16) is None


@fused_cuda
def test_w8a8_under_torch_compile_keeps_fused_kernels():
    """A torch.compile'd (dynamic) W8A8 layer must not trip the triton launch wrappers: they run eagerly behind a
    graph break, USE_TRITON stays on and the result is bit-identical to the uncompiled fused forward."""
    torch.manual_seed(0)
    w = (torch.randn(2048, 1024) * 0.02).to("cuda", torch.bfloat16)
    ql = Q.QuantLinear(Q.quantize_weight(w, "int8_convrot"), None, torch.bfloat16, int8_matmul=True)
    compiled = torch.compile(ql, dynamic=True)
    assert Q.USE_TRITON
    for m in (64, 80):  # second shape: dynamic recompile / guard path
        x = torch.randn(m, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = compiled(x)
        y.float().sum().backward()
        assert Q.USE_TRITON and not Q._W8A8_WARNED
        assert torch.equal(y, ql(x))
        assert x.grad is not None and torch.isfinite(x.grad).all()
    torch._dynamo.reset()


@fused_cuda
def test_fused_kernel_failure_falls_back_to_eager(monkeypatch):
    def boom(x):
        raise RuntimeError("simulated triton failure")
    x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    q_e, s_e = Q._quantize_act_int8_eager(x)
    monkeypatch.setattr(Q, "_quantize_act_int8_triton", boom)
    monkeypatch.setattr(Q, "USE_TRITON", True)
    q, s = Q.quantize_act_int8(x)
    assert torch.equal(q, q_e) and torch.equal(s, s_e)
    assert Q.USE_TRITON is False  # switched off once, no retry
    assert torch.equal(Q.quantize_act_int8(x)[0], q_e)
    # the fused rotate+quantize follows the same switch: two-step path (rotate_act, eager quantizer)
    q_r, s_r = Q.rotate_quantize_act_int8(x, 256)
    q_r_e, s_r_e = Q._quantize_act_int8_eager(Q.rotate_act(x, 256))
    assert torch.equal(q_r, q_r_e) and torch.equal(s_r, s_r_e)


# ---------------------------------------------------------------- native NVFP4 (FP4 tensor cores)

_fp4_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count()) if Q.fp4_tensor_cores(f"cuda:{i}")]
fp4_cuda = pytest.mark.skipif(not (_fp4_devices and Q.HAS_TRITON), reason="needs a Blackwell GPU (FP4 tensor cores)")


def test_nvfp4_native_off_cpu_and_settings():
    """CPU tensors, the setting off, or a GPU without FP4 tensor cores: the dequantize path, as before."""
    torch.manual_seed(0)
    w = torch.randn(256, 512) * 0.02
    qw = Q.quantize_weight(w, "nvfp4")
    assert not Q.nvfp4_native_supported(torch.randn(4, 512), qw) and not Q.fp4_tensor_cores("cpu")
    old = Q.DEFAULT_NVFP4_MATMUL
    try:
        Q.set_nvfp4_default(False)
        ql = Q.QuantLinear(qw, None, torch.float32)
        assert ql.nvfp4_matmul is False
        Q.set_nvfp4_default(True)
        ql2 = Q.QuantLinear(qw, None, torch.float32)
        model = torch.nn.Sequential(ql, ql2, Q.QuantLinear(Q.quantize_weight(w, "int8"), None, torch.float32))
        assert Q.configure_nvfp4(model, True) == 2 and ql.nvfp4_matmul and ql2.nvfp4_matmul
    finally:
        Q.set_nvfp4_default(old)
    x = torch.randn(4, 512, requires_grad=True)
    y = ql(x)
    ref = x.detach() @ qw.dequantize(torch.float32).t()
    assert torch.allclose(y, ref, atol=1e-5)
    # the eager activation quantizer (fallback) lays scales out for cuBLAS: rows padded to 128
    q, s, g = Q.quantize_act_nvfp4(torch.randn(20, 512))
    assert q.shape == (20, 256) and s.shape == (128, 32) and g.shape == (1,)


@fused_cuda
@pytest.mark.parametrize("device", _devices)
def test_nvfp4_dequant_kernel_matches_eager(device):
    torch.manual_seed(0)
    for rows, cols in ((1000, 3000), (128, 64), (6144, 1536)):
        qw = Q.quantize_weight((torch.randn(rows, cols) * 0.02).to(device), "nvfp4")
        for dtype in (torch.bfloat16, torch.float32):
            fast = Q._dequantize_nvfp4_triton(qw.qdata, qw.scale, qw.scale2, qw.orig_shape, dtype)
            slow = Q._dequantize_nvfp4_eager(qw.qdata, qw.scale, qw.scale2, qw.orig_shape, dtype)
            assert torch.equal(fast, slow), (rows, cols, dtype)


@fused_cuda
@pytest.mark.parametrize("device", _devices)
def test_nvfp4_act_quant_kernel_matches_reference(device):
    torch.manual_seed(1)
    for m, k in ((4113, 3072), (5, 64), (300, 6144)):
        x = (torch.randn(m, k) * 3).to(device, torch.bfloat16)
        x[0, :16] = 0  # an all-zero block
        x[1, 16:32] = 0.25 * torch.tensor([1.0, -1.0] * 8)  # exact e2m1 ties after scaling
        q, s, g = Q._quantize_act_nvfp4_triton(x)
        q_r, s_r, _ = Q.quantize_nvfp4(x, g)
        assert torch.equal(q, q_r[:m])
        got = Q.from_blocked(s, Q._roundup(m, 128), Q._ceil_div(k // 16, 4) * 4)[:m, : k // 16]
        want = Q.from_blocked(s_r, Q._roundup(m, 16), k // 16)[:m]
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


@fp4_cuda
def test_nvfp4_native_forward_backward():
    """W4A4 forward = the product of the two NVFP4 operands (to bf16 rounding); backward exact against the stored
    weight; the bias and padded weight rows are handled; a QuantLinear takes the native path."""
    dev = _fp4_devices[0]
    torch.manual_seed(2)
    w = (torch.randn(1000, 3072) * 0.02).to(dev)  # 1000 rows: padded to 1008 in storage
    bias = torch.randn(1000, device=dev, dtype=torch.bfloat16)
    qw = Q.quantize_weight(w, "nvfp4")
    ql = Q.QuantLinear(qw, bias, torch.bfloat16)
    ql.nvfp4_matmul = True
    x = torch.randn(2, 77, 3072, device=dev, dtype=torch.bfloat16, requires_grad=True)
    assert ql.uses_native_fp4(x)
    y = ql(x)
    a, a_s, a_g = Q.quantize_act_nvfp4(x.detach().reshape(-1, 3072))
    x_q = Q.dequantize_nvfp4(a, a_s, a_g, (154, 3072), torch.float32)  # what the FP4 kernel sees
    ref = x_q @ qw.dequantize(torch.float32).t() + bias.float()
    assert y.shape == (2, 77, 1000)
    assert ((y.float().reshape(-1, 1000) - ref).norm() / ref.norm()) < 5e-3
    g = torch.randn_like(y)
    y.backward(g)
    want = (g.reshape(-1, 1000) @ qw.dequantize(torch.bfloat16)).reshape(x.shape)
    assert torch.equal(x.grad, want)
    ql.nvfp4_matmul = False
    assert not ql.uses_native_fp4(x)


@fp4_cuda
def test_nvfp4_stochastic_rounding_is_unbiased():
    """Stochastic rounding: block scales round up (no value clips), every value lands on one of its two e2m1
    neighbours, and the mean over seeds is the value (round to nearest is biased toward the grid points)."""
    dev = _fp4_devices[0]
    torch.manual_seed(3)
    x = (torch.randn(64, 256) * 2).to(dev, torch.bfloat16)
    x[5] *= 1e-5  # blocks whose scale is an e4m3 subnormal, or rounds to 0 to nearest
    q, s, g = Q._quantize_act_nvfp4_triton(x)
    nearest = Q.dequantize_nvfp4(q, s, g, (64, 256), torch.float32)
    step = x.float().abs().amax(dim=1, keepdim=True) * 0.4  # wider than any e2m1 gap at this scale
    exact = (x.float().reshape(64, 16, 16).abs().amax(-1) / 6 / g).clamp(max=448)
    e4m3 = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float().to(dev)  # all finite, ascending
    acc = torch.zeros(64, 256, device=dev)
    n = 400
    for seed in range(n):
        qs, ss, gs = Q._quantize_act_nvfp4_triton(x, stochastic=True, seed=seed)
        got = Q.from_blocked(ss, 128, 16)[:64].float()
        assert torch.equal(got, e4m3[torch.searchsorted(e4m3, exact)]) and torch.equal(gs, g)  # smallest >= exact
        v = Q.dequantize_nvfp4(qs, ss, gs, (64, 256), torch.float32)
        assert ((v - x.float()).abs() <= step + 1e-6).all()
        acc += v
    mean_err = ((acc / n) - x.float()).norm() / x.float().norm()
    rtn_err = (nearest - x.float()).norm() / x.float().norm()
    assert mean_err < rtn_err / 3, (mean_err.item(), rtn_err.item())


@fp4_cuda
def test_nvfp4_fp4_backward_close_to_exact():
    """nvfp4_backward="fp4": grad_x on the FP4 tensor cores stays close to the exact bf16 backward (same forward)."""
    dev = _fp4_devices[0]
    torch.manual_seed(4)
    w = (torch.randn(1000, 3072) * 0.02).to(dev)
    qw = Q.quantize_weight(w, "nvfp4")
    ql = Q.QuantLinear(qw, None, torch.bfloat16)
    Q.configure_nvfp4(torch.nn.Sequential(ql), True, "fp4")
    assert ql.nvfp4_backward == "fp4"
    x = torch.randn(2, 77, 3072, device=dev, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(2, 77, 1000, device=dev, dtype=torch.bfloat16)
    ql(x).backward(g)
    exact = (g.reshape(-1, 1000) @ qw.dequantize(torch.bfloat16)).float().reshape(x.shape)
    rel = ((x.grad.float() - exact).norm() / exact.norm()).item()
    cos = torch.nn.functional.cosine_similarity(x.grad.float().flatten(), exact.flatten(), dim=0).item()
    assert rel < 0.3 and cos > 0.95, (rel, cos)
    with pytest.raises(ValueError):
        Q.configure_nvfp4(torch.nn.Sequential(ql), True, "int4")
