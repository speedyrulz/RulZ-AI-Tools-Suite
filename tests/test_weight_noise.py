"""Weight noise (weight_noise): off is a no-op, relative / absolute scale, norm bound, seeded and independent of the
global RNG, only LoRA / LoKr adapter tensors are touched, config validation, race rows. CPU only, no model weights."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as lora_mod  # noqa: E402
from rulztrainer import race as R  # noqa: E402
from rulztrainer import weightnoise as wn  # noqa: E402
from rulztrainer.config import TrainConfig, weight_noise_reason  # noqa: E402


def _big(seed=0, shape=(512, 512), scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * scale


def test_off_is_an_exact_no_op():
    w = _big()
    before = w.clone()
    assert wn.apply_weight_noise([w], 0.0, "relative", False, 1, 5, stats=True) is None
    assert torch.equal(w, before)


def test_relative_leaves_zeros_alone_and_scales_with_rms():
    zero = torch.zeros(64, 64)
    wn.apply_weight_noise([zero], 0.05, "relative", False, 1, 0)
    assert torch.count_nonzero(zero) == 0  # (a zero-initialised LoRA-up gets none)
    w = _big(scale=3.0)
    rms = w.pow(2).mean().sqrt().item()
    before = w.clone()
    wn.apply_weight_noise([w], 0.0125, "relative", False, 1, 0)
    assert (w - before).std().item() == pytest.approx(0.0125 * rms, rel=0.03)


def test_absolute_std_is_sigma_even_on_zeros():
    w = torch.zeros(512, 512)
    wn.apply_weight_noise([w], 0.01, "absolute", False, 1, 0)
    assert w.std().item() == pytest.approx(0.01, rel=0.03)


def test_bound_norm_keeps_each_tensors_norm():
    a, b = _big(1), _big(2, (64, 32), scale=0.1)
    na, nb = a.norm().clone(), b.norm().clone()
    wn.apply_weight_noise([a, b], 0.05, "relative", True, 1, 0)
    assert torch.allclose(a.norm(), na, rtol=1e-5) and torch.allclose(b.norm(), nb, rtol=1e-5)
    c = _big(3)
    before = c.clone()
    wn.apply_weight_noise([c], 0.05, "relative", False, 1, 0)
    assert not torch.allclose(c.norm(), before.norm(), rtol=1e-6)  # (unbound: the norm moves)


def test_same_seed_and_step_give_the_same_noise_and_the_global_rng_stays_put():
    a, b, c = _big(), _big(), _big()
    torch.manual_seed(123)
    state = torch.get_rng_state().clone()
    wn.apply_weight_noise([a], 0.02, "relative", False, 7, 11)
    assert torch.equal(torch.get_rng_state(), state)
    wn.apply_weight_noise([b], 0.02, "relative", False, 7, 11)
    assert torch.equal(a, b)
    wn.apply_weight_noise([c], 0.02, "relative", False, 7, 12)  # another step: other noise
    assert not torch.equal(a, c)
    d = _big()
    wn.apply_weight_noise([d], 0.02, "relative", False, 8, 11)  # another seed
    assert not torch.equal(a, d)


def test_bf16_goes_through_fp32_and_stats_are_the_norms():
    w = _big(shape=(128, 128)).to(torch.bfloat16)
    pre = w.float().norm().item()
    before = w.clone()
    stats = wn.apply_weight_noise([w], 0.05, "relative", False, 1, 0, stats=True)
    assert w.dtype == torch.bfloat16 and not torch.equal(w, before)
    assert stats["weight_norm"] == pytest.approx(pre, rel=1e-5)
    assert stats["noise_norm"] == pytest.approx(0.05 * pre, rel=0.05)  # (noise norm = sigma x ||w|| for relative)


def test_bad_mode_is_refused():
    with pytest.raises(ValueError, match="weight_noise_mode"):
        wn.apply_weight_noise([_big(shape=(4, 4))], 0.01, "gaussian", False, 1, 0)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)
        self.other = torch.nn.Linear(16, 16)


def _inject(kind="lora", dora=False):
    m = _Tiny()
    injected = lora_mod.inject_lora(m, 4, 4.0, [], names=["proj"], kind=kind, dora=dora)
    for p in lora_mod.lora_parameters(injected):  # (lora_up starts at zero: give every tensor a value)
        torch.nn.init.normal_(p, std=0.1)
    return m, injected


@pytest.mark.parametrize("kind", ["lora", "lokr"])
def test_only_adapter_tensors_are_touched(kind):
    m, injected = _inject(kind)
    base = {k: v.clone() for k, v in m.state_dict().items() if "lora" not in k and "lokr" not in k}
    adapters = wn.adapter_tensors(injected)
    assert len(adapters) == 2
    before = [t.clone() for t in adapters]
    wn.apply_weight_noise(adapters, 0.05, "relative", False, 1, 0)
    assert all(not torch.equal(a, b) for a, b in zip(adapters, before))
    for k, v in m.state_dict().items():  # the base layer and the untouched module stay as they were
        if k in base:
            assert torch.equal(v, base[k]), k


def test_oft_and_dora_scale_are_left_alone():
    _, oft = _inject("oft")
    assert wn.adapter_tensors(oft) == []
    _, dora = _inject("lora", dora=True)
    assert len(wn.adapter_tensors(dora)) == 2  # lora_down / lora_up; dora_scale stays


def test_optimizer_state_is_not_touched():
    _, injected = _inject()
    params = list(lora_mod.lora_parameters(injected))
    opt = torch.optim.AdamW(params, lr=1e-3)
    sum(p.sum() for p in params).backward()
    opt.step()
    state = {id(p): {k: v.clone() for k, v in opt.state[p].items() if torch.is_tensor(v)} for p in params}
    wn.apply_weight_noise(wn.adapter_tensors(injected), 0.05, "relative", False, 1, 0)
    for p in params:
        for k, v in state[id(p)].items():
            assert torch.equal(opt.state[p][k], v)


def test_config_defaults_and_validation():
    c = TrainConfig()
    assert (c.weight_noise, c.weight_noise_mode, c.weight_noise_bound_norm) == (0.0, "relative", False)
    base = dict(datasets=[{"path": "a"}])
    for bad in (dict(weight_noise=-0.01), dict(weight_noise=0.2), dict(weight_noise_mode="gaussian")):
        with pytest.raises(ValueError, match="weight_noise"):
            TrainConfig(**base, **bad).validate()
    TrainConfig(**base, weight_noise=0.0125, weight_noise_mode="absolute", weight_noise_bound_norm=True).validate()


def test_reason_for_finetune_and_oft():
    assert weight_noise_reason(TrainConfig()) == ""
    assert weight_noise_reason(TrainConfig(network_type="lokr")) == ""
    assert "fine-tune" in weight_noise_reason(TrainConfig(finetune=True))
    assert "OFT" in weight_noise_reason(TrainConfig(network_type="oft"))


def test_race_rows_set_noise_per_configuration():
    assert {"weight_noise", "weight_noise_mode", "weight_noise_bound_norm"} <= set(R.RACE_KEYS["LR & optimizer"])
    cfg = TrainConfig(race=True, race_configs=[
        {"name": "off"}, {"name": "n1", "weight_noise": 0.0125}, {"name": "n2", "weight_noise": 0.01,
                                                                   "weight_noise_mode": "absolute",
                                                                   "weight_noise_bound_norm": True}])
    rows = {n: c for n, c, _ in R.variant_configs(cfg)}
    assert rows["off"].weight_noise == 0.0
    assert rows["n1"].weight_noise == 0.0125 and rows["n1"].weight_noise_mode == "relative"
    assert rows["n2"].weight_noise_mode == "absolute" and rows["n2"].weight_noise_bound_norm is True
    assert R.parse_overrides("weight_noise=0.0125, weight_noise_bound_norm=true") == \
        {"weight_noise": 0.0125, "weight_noise_bound_norm": True}
