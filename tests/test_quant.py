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
