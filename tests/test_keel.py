"""Keel (rulztrainer.optimizers.keel): Automagic 4 exactly with its switches off, the gauge preconditioner's
invariance to a LoRA pair's reparameterisation, the cautious mask, training and state round trips across the
Automagic kinds, and the config / race / WebUI wiring."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as lora_mod  # noqa: E402
from rulztrainer.config import AUTOMAGIC, OPTIMIZERS, TrainConfig  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402
from rulztrainer.optimizers.automagic4 import Automagic4  # noqa: E402
from rulztrainer.optimizers.keel import Keel, gauge_precondition  # noqa: E402
from rulztrainer.trainer import build_optimizer, build_scheduler, current_lr  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16, latent_channels=128)
TARGETS = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]


def _model(seed: int = 0):
    torch.manual_seed(seed)
    model = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    injected = lora_mod.inject_lora(model, 4, 4, TARGETS)
    model.train()
    return model, injected, list(lora_mod.lora_parameters(injected))


def _step(model, params, opt, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 128, 4, 4, generator=g).to(torch.bfloat16)
    c = torch.randn(1, 5, 32, generator=g).to(torch.bfloat16)
    loss = model(x, torch.tensor([300.0]), c).float().pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return loss.item()


def _cfg(**kw):
    return TrainConfig(**{**dict(optimizer="keel", learning_rate=1e-4, automagic_polarity_history=4, steps=20), **kw})


def test_keel_with_its_switches_off_is_automagic4_bitwise():
    ma, inja, pa = _model()
    mb, injb, pb = _model()
    assert all(torch.equal(a, b) for a, b in zip(pa, pb))
    oa = build_optimizer(_cfg(optimizer="automagic4"), pa)
    ob = build_optimizer(_cfg(keel_gauge=False), pb, pairs=lora_mod.lora_pairs(injb))
    assert isinstance(ob, Keel) and isinstance(oa, Automagic4) and not isinstance(oa, Keel)
    for i in range(6):
        assert _step(ma, pa, oa, i) == _step(mb, pb, ob, i)
    assert all(torch.equal(a, b) for a, b in zip(pa, pb))
    assert current_lr(oa, None) == current_lr(ob, None)
    assert ob.cautious_share() is None  # nothing preconditioned, nothing masked


def test_keel_gauge_changes_the_step_and_keeps_the_rails():
    ma, inja, pa = _model()
    mb, injb, pb = _model()
    oa = build_optimizer(_cfg(optimizer="automagic4"), pa)
    cfg = _cfg()
    ob = build_optimizer(cfg, pb, pairs=lora_mod.lora_pairs(injb))
    assert len(ob.pairs) == len(injb)
    for i in range(6):
        _step(ma, pa, oa, i)
        _step(mb, pb, ob, i)
    assert any(not torch.equal(a, b) for a, b in zip(pa, pb))
    assert all(torch.isfinite(p).all() for p in pb)
    lr = current_lr(ob, None)
    assert cfg.automagic_min_lr <= lr <= cfg.automagic_max_lr
    share = ob.cautious_share()
    assert share is not None and 0.0 < share <= 1.0


def _linear_grads(G, A, B):
    """Gradients of <G, B A> for the pair: (g_down, g_up)."""
    return B.T @ G, G @ A.T


def test_gauge_precondition_is_invariant_to_the_pairs_reparameterisation():
    torch.manual_seed(1)
    r, n_in, n_out = 4, 16, 12
    A = torch.randn(r, n_in, dtype=torch.float64)
    B = torch.randn(n_out, r, dtype=torch.float64)
    G = torch.randn(n_out, n_in, dtype=torch.float64)
    S = torch.randn(r, r, dtype=torch.float64) + 3.0 * torch.eye(r, dtype=torch.float64)  # invertible, not orthogonal
    A2, B2 = torch.linalg.solve(S, A), B @ S  # (B S)(S^-1 A) = B A
    assert torch.allclose(B2 @ A2, B @ A)

    def delta_w(a, b, eps):
        gd, gu = _linear_grads(G, a, b)
        nd, nu = gauge_precondition(gd, gu, a, b, eps)
        return nu @ a.float() + b.float() @ nd  # the first-order change of B A for a step along the gradients

    plain = lambda a, b: (_linear_grads(G, a, b)[1] @ a + b @ _linear_grads(G, a, b)[0])  # noqa: E731
    assert not torch.allclose(plain(A, B), plain(A2, B2), rtol=1e-3)  # plain gradients are not invariant
    assert torch.allclose(delta_w(A, B, 1e-9), delta_w(A2, B2, 1e-9), rtol=1e-3, atol=1e-4)
    # a zero lora_up (the LoRA's start) gives a zero lora_down gradient and no blow-up through the floor
    nd, nu = gauge_precondition(*_linear_grads(G, A, torch.zeros_like(B)), A, torch.zeros_like(B), 0.01)
    assert torch.equal(nd, torch.zeros_like(nd)) and torch.isfinite(nu).all()


def test_cautious_mask_drops_elements_that_point_against_the_gradient():
    torch.manual_seed(2)
    down = torch.nn.Parameter(torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.02]]))  # nearly dependent rows: a solve mixes
    up = torch.nn.Parameter(torch.randn(4, 2))
    other = torch.nn.Parameter(torch.randn(3))
    raw_down, raw_up = torch.randn(2, 3), torch.randn(4, 2)
    expected = gauge_precondition(raw_down, raw_up, down, up, 1e-6)
    assert any((e * g).le(0).any() for e, g in zip(expected, (raw_down, raw_up)))  # the fixture has sign flips

    def prime():
        down.grad, up.grad, other.grad = raw_down.clone(), raw_up.clone(), torch.randn(3)

    opt = Keel([down, up, other], pairs=[(down, up)], lr=1e-4, fused=False, gauge_eps=1e-6, cautious=True)
    prime()
    opt.precondition()
    for new, raw, exp in zip((down.grad, up.grad), (raw_down, raw_up), expected):
        assert (new * raw).ge(0).all()  # nothing points against the raw gradient
        agree = (exp * raw).gt(0)
        assert torch.equal(new[agree], exp[agree]) and torch.equal(new[~agree], torch.zeros_like(new[~agree]))
    share = opt.cautious_share()
    assert share is not None and 0.0 < share < 1.0
    opt.step()  # the flag clears, the state fills
    assert not opt._preconditioned and len(opt.state[down]) > 0
    # without the mask the preconditioned gradient is used as it is
    opt2 = Keel([down, up, other], pairs=[(down, up)], lr=1e-4, fused=False, gauge_eps=1e-6, cautious=False)
    prime()
    opt2.precondition()
    assert torch.equal(down.grad, expected[0]) and torch.equal(up.grad, expected[1])
    assert opt2.cautious_share() is None


def test_keel_pairs_are_checked():
    down = torch.nn.Parameter(torch.randn(2, 3))
    up = torch.nn.Parameter(torch.randn(4, 2))
    stranger = torch.nn.Parameter(torch.randn(4, 2))
    with pytest.raises(ValueError, match="not among"):
        Keel([down, up], pairs=[(down, stranger)], lr=1e-4, fused=False)
    with pytest.raises(ValueError, match="down \\[r, in\\]"):
        Keel([down, up], pairs=[(up, down)], lr=1e-4, fused=False)
    with pytest.raises(ValueError, match="gauge_eps"):
        Keel([down, up], pairs=[(down, up)], lr=1e-4, fused=False, gauge_eps=0.0)
    # no pairs (a LoKr / OFT adapter, a fine-tune): the step is Automagic 4's
    opt = Keel([down, up], lr=1e-4, fused=False)
    down.grad, up.grad = torch.randn(2, 3), torch.randn(4, 2)
    opt.step()
    assert opt.cautious_share() is None


def test_keel_trains_round_trips_and_converts_with_automagic4(tmp_path):
    model, injected, params = _model()
    cfg = _cfg()
    opt = build_optimizer(cfg, params, pairs=lora_mod.lora_pairs(injected))
    sched = build_scheduler(cfg, opt)
    before = [p.detach().clone() for p in params]
    for i in range(8):
        _step(model, params, opt, i)
        sched.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, params))
    lr = current_lr(opt, sched)
    assert abs(lr - 1e-4) > 1e-7  # the vote moved it
    sd = opt.state_dict()
    torch.save(sd, tmp_path / "keel.pt")
    saved = torch.load(tmp_path / "keel.pt", weights_only=False)
    opt2 = build_optimizer(cfg, params, pairs=lora_mod.lora_pairs(injected))
    opt2.load_state_dict(saved)
    assert abs(current_lr(opt2, sched) - lr) < 1e-6 * lr
    _step(model, params, opt2, 9)
    # a Keel state under Automagic 4 and an Automagic 4 state under Keel (the same state); loaded again from disk,
    # as a resume does (torch hands a loaded fp32 tensor to the optimizer as it is, and opt2's step moved it)
    opt3 = build_optimizer(_cfg(optimizer="automagic4"), params)
    opt3.load_state_dict(torch.load(tmp_path / "keel.pt", weights_only=False))
    assert abs(current_lr(opt3, sched) - lr) < 1e-6 * lr
    _step(model, params, opt3, 10)
    opt4 = build_optimizer(cfg, params, pairs=lora_mod.lora_pairs(injected))
    opt4.load_state_dict(opt3.state_dict())
    _step(model, params, opt4, 11)
    assert all(torch.isfinite(p).all() for p in params)


def test_lora_pairs_and_the_lokr_case():
    model, injected, _ = _model()
    pairs = lora_mod.lora_pairs(injected)
    assert len(pairs) == len(injected)
    assert all(d.shape[0] == u.shape[1] == 4 for d, u in pairs)
    torch.manual_seed(0)
    lokr_model = MOSSImageTransformer(**TINY)
    lokr = lora_mod.inject_lora(lokr_model, 4, 4.0, TARGETS, kind="lokr", factor=4)
    assert lora_mod.lora_pairs(lokr) == []
    opt = build_optimizer(_cfg(), list(lora_mod.lora_parameters(lokr)), pairs=lora_mod.lora_pairs(lokr))
    assert isinstance(opt, Keel) and opt.pairs == []


def test_keel_config_race_and_wizard_wiring(tmp_path):
    assert "keel" in OPTIMIZERS and "keel" in AUTOMAGIC
    _cfg().validate()
    with pytest.raises(ValueError, match="keel_gauge_eps"):
        _cfg(keel_gauge_eps=0).validate()
    from rulztrainer import race as R
    from rulztrainer.finetune import BYTES_PER_PARAM

    assert {"keel_gauge", "keel_gauge_eps", "keel_cautious"} <= set(R.RACE_KEYS["LR & optimizer"])
    TrainConfig(race=True, probe_every=10, face_refs=["ref.png"], steps=40, race_configs=[
        {"name": "a"}, {"name": "b", "optimizer": "keel", "keel_gauge_eps": 0.1, "keel_cautious": False}]).validate()
    assert BYTES_PER_PARAM["keel"] == BYTES_PER_PARAM["automagic4"]
    import webui

    assert {"keel_gauge", "keel_gauge_eps", "keel_cautious"} <= set(webui.RACE_LABELS)
    cfg = TrainConfig.from_dict({**_cfg().to_dict(), "dataset_dir": str(tmp_path), "trigger_word": "x"})
    rec = webui.wizard_recommendations(cfg, "character", [], None)
    assert "learning_rate" not in rec  # Keel sets its own rate, like Automagic
