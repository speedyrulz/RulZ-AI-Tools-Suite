"""OFT (network_type oft) and DoRA (dora, on LoRA and LoKr): the training forward against ComfyUI's merge of the saved
keys (comfy/weight_adapter oft.py and base.py weight_decompose, reproduced below), identity at the start with
gradients to every part, the COFT cap, export keys, a load round trip and a plain LoRA resumed with DoRA on, kind
switches for a race, mirroring, optimizer groups and the config checks."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


def comfy_merged_weight(state: dict, key: str, weight: torch.Tensor) -> torch.Tensor:
    """ComfyUI's merge of one layer's keys onto its weight (strength 1, fp32), as its weight adapters compute it:
    LoRA up @ down x alpha / rank, LoKr kron(w1, w2), OFT R^T W per block with R = (I + Q)(I - Q)^-1 and
    Q = blocks - blocks^T capped at .alpha; with .dora_scale, weight_decompose: W' = (W + delta) * dora_scale /
    ||W + delta|| (on the input axis, the norm of the merged weight per column)."""
    w = weight.float().clone()
    alpha = state.get(f"{key}.alpha")
    if f"{key}.oft_blocks" in state:
        blocks = state[f"{key}.oft_blocks"].float()
        n, b, _ = blocks.shape
        q = blocks - blocks.transpose(1, 2)
        if alpha is not None and float(alpha) > 0 and q.norm() + 1e-8 > float(alpha):
            q = q * float(alpha) / (q.norm() + 1e-8)
        eye = torch.eye(b)
        r = (eye + q) @ (eye - q).inverse()
        return w + torch.einsum("knm,kn...->km...", r - eye, w.view(n, b, -1)).reshape(w.shape)
    if f"{key}.lokr_w1" in state:
        delta = torch.kron(state[f"{key}.lokr_w1"].float(), state[f"{key}.lokr_w2"].float())
    else:
        up, down = state[f"{key}.lora_up.weight"].float(), state[f"{key}.lora_down.weight"].float()
        delta = (up @ down) * (float(alpha) / down.shape[0] if alpha is not None else 1.0)
    dora = state.get(f"{key}.dora_scale")
    if dora is None:
        return w + delta
    assert dora.shape[0] != w.shape[0]  # the input axis: ComfyUI normalizes the merged weight there
    merged = w + delta
    return merged * (dora.float() / (merged.norm(dim=0, keepdim=True) + torch.finfo(torch.float32).eps))


def _trained(m: L.LoRALinear, seed: int = 0) -> L.LoRALinear:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.adapter_params().values():
            p.add_(torch.randn(p.shape, generator=g) * (0.05 if p.dim() == 3 else 0.1))
    return m


CASES = [("lora", False, 0.0), ("lora", True, 0.0), ("lokr", True, 0.0), ("oft", False, 0.0), ("oft", False, 0.05)]


@pytest.mark.parametrize("quant", [None, "int8_convrot"])
@pytest.mark.parametrize("kind,dora,constraint", CASES)
def test_the_forward_is_what_comfyui_merges_from_the_file(quant, kind, dora, constraint):
    torch.manual_seed(0)
    lin = nn.Linear(128, 192)
    base = lin if quant is None else Q.QuantLinear(Q.quantize_weight(lin.weight.data, quant, group=64), lin.bias.data,
                                                   torch.float32, int8_matmul=False)
    m = _trained(L.LoRALinear(base, 8, 4.0, kind=kind, factor=8, block=16, constraint=constraint, dora=dora))
    x = torch.randn(5, 128)
    with torch.no_grad():
        ours = m(x)
    state = {k: v.float() for k, v in L.export_state_dict({"lin": m}, torch.float32, "comfy").items()}
    ref = x @ comfy_merged_weight(state, "diffusion_model.lin", L.dense_weight(base)).t() + lin.bias
    assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-4)


def test_identity_at_the_start_and_gradients_to_every_part():
    torch.manual_seed(0)
    base = nn.Linear(48, 64)
    x = torch.randn(3, 48)
    for kind, dora in (("oft", False), ("lora", True), ("lokr", True)):
        m = L.LoRALinear(base, 4, 4.0, kind=kind, factor=4, block=16, dora=dora)
        assert torch.allclose(m(x), base(x), atol=1e-6), kind  # R = I; DoRA: g = 1 with a zero delta
        _trained(m)
        m(x).pow(2).sum().backward()
        for part, p in m.adapter_params().items():
            assert p.grad is not None and p.grad.abs().sum() > 0, (kind, part)
    m = L.LoRALinear(base, 4, 4.0, kind="oft", block=16)
    assert m.oft_blocks.shape == (4, 16, 16) and list(m.adapter_params()) == ["oft_blocks"]
    assert m.lora_down.weight.numel() == 0 and m.dora_scale.numel() == 0 and not m.dora
    m = L.LoRALinear(base, 4, 4.0, kind="lora", dora=True)
    assert list(m.adapter_params()) == ["lora_down", "lora_up", "dora_scale"]
    assert torch.allclose(m.dora_scale, base.weight.norm(dim=0))  # the magnitude starts at ||W|| per input column
    assert L.oft_shape(3072, 32) == (96, 32) and L.oft_shape(360, 32) == (12, 30)


def test_the_rotation_is_orthogonal_and_the_constraint_caps_it():
    blocks = torch.randn(3, 8, 8)
    r = L.oft_rotation(blocks)
    assert torch.allclose(r @ r.transpose(1, 2), torch.eye(8).expand(3, 8, 8), atol=1e-4)
    capped = L.oft_rotation(blocks, constraint=0.1)
    q_capped = (capped - torch.eye(8)) @ torch.linalg.inv(capped + torch.eye(8))  # the Cayley transform's inverse
    assert abs(q_capped.norm().item() - 0.1) < 1e-3
    assert torch.allclose(L.oft_rotation(blocks * 1e-4, constraint=10.0), L.oft_rotation(blocks * 1e-4))  # under it


def test_export_keys_and_metadata(tmp_path):
    torch.manual_seed(0)
    base = nn.Linear(32, 64)
    oft = L.LoRALinear(base, 4, 4.0, kind="oft", block=16, constraint=0.02)
    dora = L.LoRALinear(nn.Linear(32, 64), 4, 4.0, kind="lora", dora=True)
    state = L.export_state_dict({"a": oft, "b": dora}, torch.bfloat16, "comfy")
    assert state["diffusion_model.a.oft_blocks"].shape == (4, 16, 16) and abs(float(state["diffusion_model.a.alpha"]) - 0.02) < 1e-6
    assert state["diffusion_model.b.dora_scale"].shape == (1, 32)
    assert {k.rsplit(".", 1)[-1] for k in state if k.startswith("diffusion_model.b.")} == {"weight", "alpha", "dora_scale"}
    peft = L.export_state_dict({"b": dora}, torch.bfloat16, "peft")
    assert "transformer.b.dora_scale" in peft and "transformer.b.lora_A.weight" in peft
    assert "diffusion_model.a.alpha" not in L.export_state_dict({"a": L.LoRALinear(base, 4, 4.0, kind="oft")})
    L.save_lora({"a": oft}, tmp_path / "o.safetensors")
    from safetensors import safe_open

    with safe_open(str(tmp_path / "o.safetensors"), "pt") as f:
        meta = f.metadata()
    assert meta["rulztrainer_network"] == "oft" and '"diag-oft"' in meta["ss_network_args"]
    assert L.adapter_label("oft", block=32) == "OFT block 32" and L.adapter_label("lokr", dora=True) == "LoKr factor 8 + DoRA"


def test_load_round_trip_and_a_plain_lora_resumed_with_dora(tmp_path):
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    for kind, dora in (("oft", False), ("lora", True), ("lokr", True)):
        a = _trained(L.LoRALinear(nn.Linear(32, 64), 4, 4.0, kind=kind, factor=4, block=16, constraint=0.5,
                                  dora=dora))
        path = tmp_path / f"{kind}.safetensors"
        L.save_lora({"lin": a}, path, dtype="fp32")
        b = L.LoRALinear(a.base, 4, 4.0, kind=kind, factor=4, block=16, dora=dora)
        assert L.load_lora_weights({"lin": b}, path) == 1
        assert torch.allclose(a(x), b(x), atol=1e-5), kind
    plain = _trained(L.LoRALinear(nn.Linear(32, 64), 4, 4.0))
    L.save_lora({"lin": plain}, tmp_path / "plain.safetensors", dtype="fp32")
    with_dora = L.LoRALinear(plain.base, 4, 4.0, dora=True)
    L.load_lora_weights({"lin": with_dora}, tmp_path / "plain.safetensors")
    assert torch.allclose(with_dora(x), plain(x), atol=1e-5)  # the magnitude starts where the loaded LoRA left it
    wrong = {"diffusion_model.lin.dora_scale": torch.ones(64, 1), **{
        k: v for k, v in L.export_state_dict({"lin": plain}, torch.float32).items()}}
    from safetensors.torch import save_file

    save_file(wrong, str(tmp_path / "out_axis.safetensors"))
    with pytest.raises(ValueError, match="output-axis"):
        L.load_lora_weights({"lin": L.LoRALinear(plain.base, 4, 4.0, dora=True)}, tmp_path / "out_axis.safetensors")


def test_race_switches_mirror_and_optimizer_groups():
    from rulztrainer import race

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    wrappers = L.inject_lora(model, 4, 4.0, [], names=["0", "2"], kind="lokr", factor=4)
    cfg = lambda **kw: SimpleNamespace(**{"lora_rank": 4, "lora_alpha": 4.0, "lora_dropout": 0.0,  # noqa: E731
                                          "network_type": "lokr", "lokr_factor": 4, **kw})
    oft = race.Variant(0, "oft", cfg(network_type="oft", oft_block_size=8), {}, ["0", "2"])
    dora = race.Variant(1, "dora", cfg(dora=True), {}, ["0", "2"])
    for v in (oft, dora):
        v.init_lora(wrappers, 3)
    assert set(oft.weights["0"]) == {"oft_blocks"} and torch.count_nonzero(oft.weights["0"]["oft_blocks"]) == 0
    assert set(dora.weights["0"]) == {"lokr_w1", "lokr_w2", "dora_scale"}
    assert oft.label == "OFT block 8" and dora.label == "LoKr factor 4 + DoRA"
    race.swap_in(oft, wrappers)
    assert wrappers["0"].kind == "oft" and wrappers["0"].oft_blocks.shape == (4, 8, 8) and not wrappers["0"].dora
    race.swap_in(dora, wrappers)
    m = wrappers["0"]
    assert m.kind == "lokr" and m.dora and m.oft_blocks.numel() == 0 and m.dora_scale.shape == (16,)
    x = torch.randn(2, 16)
    assert torch.allclose(model(x), model[2].base(torch.relu(model[0].base(x))), atol=1e-6)  # both start at identity
    state = L.export_state_dict(dora.view(), torch.float32)
    assert state["diffusion_model.0.dora_scale"].shape == (1, 16)
    other = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    other.load_state_dict({k.replace(".base", ""): v for k, v in model.state_dict().items()
                           if "base" in k or not any(p in k for p in ("lora", "lokr", "oft", "dora"))}, strict=False)
    dst = L.inject_lora(other, 4, 4.0, [], names=["0", "2"], kind="lora")
    _trained(m)
    L.mirror_lora(wrappers, dst)
    assert dst["0"].kind == "lokr" and dst["0"].dora and torch.equal(dst["0"].dora_scale, m.dora_scale)
    groups = L.lora_param_groups(wrappers, 1e-4, plus_ratio=4.0)
    assert sum(p.numel() for g in groups for p in g["params"]) == sum(p.numel() for p in L.lora_parameters(wrappers))
    lora_dora = L.inject_lora(nn.Sequential(nn.Linear(16, 8)), 4, 4.0, [], names=["0"], kind="lora", dora=True)
    groups = {g["name"]: g for g in L.lora_param_groups(lora_dora, 1e-4, plus_ratio=4.0)}
    assert groups["lora_up"]["lr"] == 4e-4 and any(p is lora_dora["0"].dora_scale for p in groups["lora"]["params"])


def test_config_checks():
    TrainConfig(network_type="oft", oft_block_size=32).validate()
    TrainConfig(network_type="lokr", dora=True).validate()
    for bad, match in ((dict(network_type="loha"), "network_type"), (dict(network_type="oft", dora=True), "dora"),
                       (dict(oft_block_size=1), "oft_block_size"), (dict(dora=True, finetune=True), "dora")):
        with pytest.raises(ValueError, match=match):
            TrainConfig(**bad).validate()


def test_dora_keeps_its_norm_through_a_render_round(monkeypatch):
    torch.manual_seed(0)
    m = _trained(L.LoRALinear(nn.Linear(16, 24), 4, 4.0, dora=True))
    calls = []
    real = L.dense_weight
    monkeypatch.setattr(L, "dense_weight", lambda base: (calls.append(1), real(base))[1])
    x = torch.randn(2, 16)
    with torch.no_grad():
        m(x), m(x)
    assert len(calls) == 2  # outside a round: every forward
    with L.frozen_adapters(), torch.no_grad():
        first = m(x)
        m(x)
        assert len(calls) == 3  # once per round
        m.lora_up.weight.add_(0.1)  # weights changed in place: recomputed
        changed = m(x)
        assert len(calls) == 4 and not torch.allclose(first, changed)
    with L.frozen_adapters(), torch.no_grad():
        m(x)
    assert len(calls) == 5  # a new round
    with L.frozen_adapters():
        m(x).sum().backward()  # gradients on: training forwards always recompute
    assert len(calls) == 6
