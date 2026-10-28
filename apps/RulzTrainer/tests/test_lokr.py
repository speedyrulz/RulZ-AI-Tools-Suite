"""LoKr (network_type lokr): the Kronecker update against the materialized product in ComfyUI's layout, the
factorization, a zero delta at init with gradients to both factors, export keys and a load round trip, a LoKr file
as a sampling-only extra, mirroring between kinds, EMA over LoKr tensors, and a race variant that switches a
wrapper from LoRA to LoKr and back."""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer.ema import LoraEMA  # noqa: E402


def test_factorization_follows_lycoris():
    assert L.factorization(6144, 8) == (8, 768) and L.factorization(16384, 8) == (8, 2048)
    assert L.factorization(1536, 8) == (8, 192) and L.factorization(2560, 8) == (8, 320)
    assert L.factorization(64, 8) == (8, 8) and L.factorization(6912, 8) == (8, 864)
    assert L.factorization(36, 8) == (6, 6) and L.factorization(127, 8) == (1, 127)
    assert L.factorization(360, 16) == (15, 24) and L.factorization(360, 1) == (1, 360)


def test_lokr_update_matches_the_kronecker_product():
    torch.manual_seed(0)
    w1, w2 = torch.randn(3, 2), torch.randn(5, 4)  # out = 15, in = 8
    x = torch.randn(2, 7, 8)
    full = torch.kron(w1, w2)  # ComfyUI builds the delta this way (w + kron(w1, w2))
    assert torch.allclose(L.lokr_update(x, w1, w2), x @ full.t(), atol=1e-5)


def test_lokr_wrapper_forward_init_and_gradients():
    torch.manual_seed(0)
    base = nn.Linear(24, 36)
    m = L.LoRALinear(base, rank=4, alpha=4.0, kind="lokr", factor=6)
    assert m.kind == "lokr" and m.lokr_w1.shape == (6, 6) and m.lokr_w2.shape == (6, 4)
    assert m.lora_down.weight.numel() == 0 and m.lora_up.weight.numel() == 0  # the inactive kind holds nothing
    assert list(m.adapter_params()) == ["lokr_w1", "lokr_w2"] and m.scale == 1.0
    x = torch.randn(3, 24)
    assert torch.equal(m(x), base(x))  # w2 starts at zero: no delta
    with torch.no_grad():
        m.lokr_w2.normal_()
    delta = torch.kron(m.lokr_w1, m.lokr_w2)
    assert torch.allclose(m(x), base(x) + x @ delta.t(), atol=1e-5)
    m.multiplier = 0.5
    assert torch.allclose(m(x), base(x) + 0.5 * (x @ delta.t()), atol=1e-5)
    m.multiplier = 1.0
    m(x).sum().backward()
    assert m.lokr_w1.grad is not None and m.lokr_w2.grad is not None and m.lokr_w1.grad.abs().sum() > 0


def _tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 8))


def test_export_keys_round_trip_and_comfy_formula(tmp_path):
    model = _tiny_model()
    inj = L.inject_lora(model, 4, 4.0, [], names=["0", "2"], kind="lokr", factor=4)
    with torch.no_grad():
        for m in inj.values():
            m.lokr_w2.normal_()
    path = tmp_path / "k.safetensors"
    L.save_lora(inj, path, fmt="comfy", dtype=torch.float32)
    state = load_file(str(path))
    assert set(state) == {f"diffusion_model.{n}.{p}" for n in ("0", "2") for p in ("lokr_w1", "lokr_w2", "alpha")}
    assert float(state["diffusion_model.0.alpha"]) == 1.0
    delta = torch.kron(state["diffusion_model.0.lokr_w1"], state["diffusion_model.0.lokr_w2"])  # ComfyUI's merge
    x = torch.randn(2, 16)
    assert torch.allclose(inj["0"](x), model[0].base(x) + x @ delta.t(), atol=1e-5)
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        meta = f.metadata()
    assert meta["rulztrainer_network"] == "lokr" and "lokr" in meta["ss_network_args"]
    # the peft format uses the "transformer." stems ComfyUI maps for Krea 2
    L.save_lora(inj, tmp_path / "p.safetensors", fmt="peft", dtype=torch.float32)
    pstate = load_file(str(tmp_path / "p.safetensors"))
    assert "transformer.0.lokr_w1" in pstate and "lora_adapter_metadata" not in (safe_open(str(tmp_path / "p.safetensors"), "pt").metadata() or {})
    # round trip into fresh wrappers of the same factor
    model2 = _tiny_model()
    inj2 = L.inject_lora(model2, 4, 4.0, [], names=["0", "2"], kind="lokr", factor=4)
    assert L.load_lora_weights(inj2, path) == 2
    assert torch.equal(inj2["0"].lokr_w2, inj["0"].lokr_w2)
    import pytest

    inj3 = L.inject_lora(_tiny_model(), 4, 4.0, [], names=["0", "2"], kind="lokr", factor=2)
    with pytest.raises(ValueError, match="LoKr shape mismatch"):
        L.load_lora_weights(inj3, path)


def test_lokr_file_as_a_sampling_extra_matches_the_merged_weight(tmp_path):
    model = _tiny_model()
    w1, w2 = torch.randn(4, 4), torch.randn(6, 4) * 0.1  # out 24 = 4 x 6, in 16 = 4 x 4
    save_file({"diffusion_model.0.lokr_w1": w1, "diffusion_model.0.lokr_w2": w2,
               "diffusion_model.0.alpha": torch.tensor(1.0)}, str(tmp_path / "extra.safetensors"))
    merged = _tiny_model()
    with torch.no_grad():
        merged[0].weight += 0.5 * torch.kron(w1, w2)
    assert L.attach_extra_lora(model, str(tmp_path / "extra.safetensors"), 0.5) == 1
    L.set_extras_enabled(model, True)
    x = torch.randn(3, 16)
    assert torch.allclose(model(x), merged(x), atol=1e-5)
    # a decomposed w2 (w2_a @ w2_b x alpha / rank) is rebuilt the way ComfyUI does
    w2a, w2b = torch.randn(6, 2), torch.randn(2, 4)
    save_file({"diffusion_model.0.lokr_w1": w1, "diffusion_model.0.lokr_w2_a": w2a,
               "diffusion_model.0.lokr_w2_b": w2b, "diffusion_model.0.alpha": torch.tensor(1.0)},
              str(tmp_path / "dec.safetensors"))
    entries = L.parse_comfy_lora(load_file(str(tmp_path / "dec.safetensors")))
    assert entries["0"][0] == "lokr" and torch.allclose(entries["0"][2], (w2a @ w2b) * 0.5)


def test_mirror_between_kinds_and_ema_over_lokr():
    a, b = _tiny_model(), _tiny_model()
    ia = L.inject_lora(a, 4, 4.0, [], names=["0", "2"], kind="lokr", factor=4)
    ib = L.inject_lora(b, 4, 4.0, [], names=["0", "2"], kind="lora")
    with torch.no_grad():
        ia["0"].lokr_w2.normal_()
    L.mirror_lora(ia, ib)
    assert ib["0"].kind == "lokr" and torch.equal(ib["0"].lokr_w2, ia["0"].lokr_w2) and ib["0"].scale == 1.0
    assert ib["0"].lora_up.weight.numel() == 0
    x = torch.randn(2, 16)
    assert torch.allclose(a(x), b(x), atol=1e-6)
    ema = LoraEMA(ia, 0.9)
    assert ema.keys == ["0.lokr_w1", "0.lokr_w2", "2.lokr_w1", "2.lokr_w2"]
    raw0 = ia["0"].lokr_w2.detach().clone()
    with torch.no_grad():
        ia["0"].lokr_w2.add_(1.0)
    ema.update(1)  # the shadow lags behind the +1 (first update: decay (1 + 0) / (10 + 0) = 0.1)
    raw1 = ia["0"].lokr_w2.detach().clone()
    with ema.applied():
        assert torch.allclose(ia["0"].lokr_w2, raw0 * 0.1 + raw1 * 0.9, atol=1e-6)
    assert torch.equal(ia["0"].lokr_w2, raw1)  # restored
    assert ema.state_dict()["shadow"]["0.lokr_w2"].shape == (6, 4)


def test_race_variant_switches_a_wrapper_between_lora_and_lokr():
    from types import SimpleNamespace

    from rulztrainer import race

    model = _tiny_model()
    wrappers = L.inject_lora(model, 4, 4.0, [], names=["0", "2"], kind="lora")
    cfg_lora = SimpleNamespace(lora_rank=4, lora_alpha=4.0, lora_dropout=0.0, network_type="lora", lokr_factor=8)
    cfg_lokr = SimpleNamespace(lora_rank=4, lora_alpha=4.0, lora_dropout=0.0, network_type="lokr", lokr_factor=4)
    va = race.Variant(0, "lora", cfg_lora, {}, ["0", "2"])
    vb = race.Variant(1, "lokr", cfg_lokr, {}, ["0"])
    va.init_lora(wrappers, 1)
    vb.init_lora(wrappers, 1)
    assert set(va.weights["0"]) == {"lora_down", "lora_up"} and vb.weights["0"]["lokr_w1"].shape == (4, 4)
    assert vb.weights["0"]["lokr_w2"].shape == (6, 4) and vb.n_weights == 16 + 24
    race.swap_in(vb, wrappers)
    assert wrappers["0"].kind == "lokr" and wrappers["0"].trainable and wrappers["0"].scale == 1.0
    assert not wrappers["2"].trainable and all(p.numel() == 0 for p in wrappers["2"].adapter_params().values())
    with torch.no_grad():
        wrappers["0"].lokr_w2.fill_(0.25)
    race.swap_out(vb, wrappers)
    assert torch.all(vb.weights["0"]["lokr_w2"] == 0.25)
    race.swap_in(va, wrappers)
    assert wrappers["0"].kind == "lora" and wrappers["0"].lora_down.weight.shape == (4, 16) and wrappers["2"].trainable
    assert wrappers["0"].lokr_w2.numel() == 0
    view = vb.view()
    assert view["0"].kind == "lokr" and torch.all(view["0"].adapter_params()["lokr_w2"] == 0.25)
    state = L.export_state_dict(view, torch.float32, "comfy")
    assert set(state) == {"diffusion_model.0.lokr_w1", "diffusion_model.0.lokr_w2", "diffusion_model.0.alpha"}
