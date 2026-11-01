"""Keel (rulztrainer.optimizers.keel): Automagic 4 exactly with its switches off, the gauge preconditioner's
invariance to a LoRA pair's reparameterisation, the cautious mask, the noise bands (observe, the per-band scale,
the band weight through Automagic 4's update gain, paired sigma in the trainer), the accumulation hook with
gather() and the consensus gate, training and state round trips across the Automagic kinds, and the config / race
/ WebUI wiring."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as lora_mod  # noqa: E402
from rulztrainer.config import AUTOMAGIC, OPTIMIZERS, TrainConfig, keel_band_edges, keel_band_weights  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402
from rulztrainer.optimizers import keel as keel_mod  # noqa: E402
from rulztrainer.optimizers.automagic4 import Automagic4  # noqa: E402
from rulztrainer.optimizers.keel import Keel, band_of, consensus_gate, gauge_precondition  # noqa: E402
from rulztrainer.trainer import build_optimizer, build_scheduler, current_lr, draw_sigmas, pairs_sigma  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16, latent_channels=128)
TARGETS = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]


def _model(seed: int = 0):
    torch.manual_seed(seed)
    model = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    injected = lora_mod.inject_lora(model, 4, 4, TARGETS)
    model.train()
    return model, injected, list(lora_mod.lora_parameters(injected))


def _step(model, params, opt, seed: int, sigma: float | None = None):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 128, 4, 4, generator=g).to(torch.bfloat16)
    c = torch.randn(1, 5, 32, generator=g).to(torch.bfloat16)
    loss = model(x, torch.tensor([300.0]), c).float().pow(2).mean()
    opt.zero_grad(set_to_none=True)
    if sigma is not None:  # as the trainer does: the micro-batch's noise level before its backward
        opt.observe(torch.tensor([sigma]))
    loss.backward()
    if hasattr(opt, "gather"):  # as the trainer does: the micro-batch sums back on p.grad before the clip
        opt.gather()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return loss.item()


def _cfg(**kw):
    return TrainConfig(**{**dict(optimizer="keel", learning_rate=1e-4, automagic_polarity_history=4, steps=20,
                                 keel_consensus=0.0), **kw})


def test_keel_with_its_switches_off_is_automagic4_bitwise():
    ma, inja, pa = _model()
    mb, injb, pb = _model()
    mc, injc, pc = _model()
    assert all(torch.equal(a, b) for a, b in zip(pa, pb))
    oa = build_optimizer(_cfg(optimizer="automagic4"), pa)
    ob = build_optimizer(_cfg(keel_gauge=False), pb, pairs=lora_mod.lora_pairs(injb))  # bands on, never observed
    assert len(ob._hook_handles) == len(pb)  # the accumulation hook is on every (fp32) parameter
    oc = build_optimizer(_cfg(keel_gauge=False, keel_bands=1), pc, pairs=lora_mod.lora_pairs(injc))
    assert isinstance(ob, Keel) and isinstance(oa, Automagic4) and not isinstance(oa, Keel)
    for i in range(6):
        la = _step(ma, pa, oa, i)
        assert la == _step(mb, pb, ob, i) == _step(mc, pc, oc, i, sigma=0.3)  # a single band observes nothing
    assert all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(pa, pb, pc))
    assert current_lr(oa, None) == current_lr(ob, None) == current_lr(oc, None)
    assert ob.cautious_share() is None and ob.band_stats() is None and oc.band_stats() is None


def test_keel_gauge_and_bands_change_the_step_and_keep_the_rails():
    ma, inja, pa = _model()
    mb, injb, pb = _model()
    oa = build_optimizer(_cfg(optimizer="automagic4"), pa)
    cfg = _cfg(keel_band_weights="identity")
    ob = build_optimizer(cfg, pb, pairs=lora_mod.lora_pairs(injb))
    assert len(ob.pairs) == len(injb) and ob.bands == 8 and len(ob.band_weights) == 8
    for i in range(12):
        _step(ma, pa, oa, i)
        _step(mb, pb, ob, i, sigma=(0.3 if i % 2 else 0.9))  # the trainer's bucketed path, two bands
    assert any(not torch.equal(a, b) for a, b in zip(pa, pb))
    assert all(torch.isfinite(p).all() for p in pb)
    lr = current_lr(ob, None)
    assert cfg.automagic_min_lr <= lr <= cfg.automagic_max_lr
    share = ob.cautious_share()
    assert share is not None and 0.0 < share <= 1.0
    rel, counts = ob.band_stats()
    assert len(rel) == len(counts) == 8 and counts[band_of(0.3, 8)] == 6 and counts[band_of(0.9, 8)] == 6
    assert all(r > 0 for b, r in enumerate(rel) if counts[b]) and all(r == 0 for b, r in enumerate(rel) if not counts[b])
    assert ob.update_gain is None  # cleared after every step


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


def test_keel_options_are_checked():
    down = torch.nn.Parameter(torch.randn(2, 3))
    up = torch.nn.Parameter(torch.randn(4, 2))
    stranger = torch.nn.Parameter(torch.randn(4, 2))
    with pytest.raises(ValueError, match="not among"):
        Keel([down, up], pairs=[(down, stranger)], lr=1e-4, fused=False)
    with pytest.raises(ValueError, match="down \\[r, in\\]"):
        Keel([down, up], pairs=[(up, down)], lr=1e-4, fused=False)
    with pytest.raises(ValueError, match="gauge_eps"):
        Keel([down, up], pairs=[(down, up)], lr=1e-4, fused=False, gauge_eps=0.0)
    with pytest.raises(ValueError, match="bands"):
        Keel([down, up], lr=1e-4, fused=False, bands=0)
    with pytest.raises(ValueError, match="band_weights"):
        Keel([down, up], lr=1e-4, fused=False, bands=4, band_weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="consensus"):
        Keel([down, up], lr=1e-4, fused=False, consensus=1.5)
    with pytest.raises(ValueError, match="consensus_power"):
        Keel([down, up], lr=1e-4, fused=False, consensus_power=0.0)
    with pytest.raises(ValueError, match="lookahead_pullback"):
        Keel([down, up], lr=1e-4, fused=False, lookahead_pullback=1.5)
    with pytest.raises(ValueError, match="lookahead_grace"):
        Keel([down, up], lr=1e-4, fused=False, lookahead_grace=-1)
    # no pairs (a LoKr / OFT adapter, a fine-tune): the step is Automagic 4's
    opt = Keel([down, up], lr=1e-4, fused=False)
    down.grad, up.grad = torch.randn(2, 3), torch.randn(4, 2)
    opt.step()
    assert opt.cautious_share() is None


# ---------------------------------------------------------------- the noise bands


def test_band_grid_and_weights():
    edges = keel_band_edges(8)
    assert len(edges) == 9 and edges[0] == 0.0 and edges[-1] == 1.0 and abs(edges[4] - 0.5) < 1e-9
    assert all(a < b for a, b in zip(edges, edges[1:]))
    for b in range(8):  # band_of agrees with the edges
        mid = (edges[b] + edges[b + 1]) / 2 if 0 < b < 7 else (edges[1] / 2 if b == 0 else (edges[7] + 1) / 2)
        assert band_of(mid, 8) == b
        assert band_of(edges[b + 1] - 1e-6, 8) == b
    assert band_of(0.0, 8) == 0 and band_of(1.0, 8) == 7 and band_of(0.5, 2) == 1 and band_of(0.49, 2) == 0
    assert keel_band_weights("", 8) is None and keel_band_weights("flat", 8) is None
    assert keel_band_weights("identity", 1) is None and keel_band_weights("1,2", 1) is None
    hump = keel_band_weights("identity", 8)
    assert len(hump) == 8 and max(hump) == hump[3] > hump[2] > 1.0 > hump[4] and hump[0] < 0.6 and hump[7] == 0.5
    assert keel_band_weights("1, 2,3,4", 4) == [1.0, 2.0, 3.0, 4.0]
    for bad in ("1,2,3", "a,b,c,d", "1,2,3,0"):
        with pytest.raises(ValueError, match="keel_band_weights"):
            keel_band_weights(bad, 4)
    with pytest.raises(ValueError, match="keel_bands"):
        _cfg(keel_bands=0).validate()
    with pytest.raises(ValueError, match="keel_band_weights"):
        _cfg(keel_band_weights="1,2").validate()
    _cfg(keel_band_weights="identity").validate()


def test_bands_equalise_two_noise_levels_of_different_scale():
    torch.manual_seed(3)
    w2 = torch.nn.Parameter(torch.randn(6, 5))
    w1 = torch.nn.Parameter(torch.randn(7))
    base = [torch.randn(6, 5), torch.randn(7)]
    opt = Keel([w2, w1], lr=1e-4, fused=False, bands=8, band_weights=[1, 1, 1, 2.0, 1, 1, 1, 0.5], gauge=False)
    quiet, loud, faint = 0.3, 0.9, 0.05  # bands 3, 6 and 1: gradients at 1x, 5x and 0.01x
    scales = {quiet: 1.0, loud: 5.0, faint: 0.01}
    assert band_of(quiet, 8) == 3 and band_of(loud, 8) == 6 and band_of(faint, 8) == 1

    def prime(scale):
        for p, b in zip((w2, w1), base):
            p.grad = b * scale

    for i in range(3 * keel_mod.BAND_MIN_OBS):
        sigma = (quiet, loud, faint)[i % 3]
        opt.zero_grad()
        opt.observe(torch.tensor([sigma]))
        prime(scales[sigma])
        opt.step()
    st = opt.state[w2]
    assert st["band_ms"].shape == (9,) and st["band_n"][3] == st["band_n"][6] == st["band_n"][1] == keel_mod.BAND_MIN_OBS
    assert st["band_n"][8] == 3 * keel_mod.BAND_MIN_OBS and st["band_n"][0] == 0
    assert float(st["band_ms"][6] / st["band_ms"][3]) == pytest.approx(25.0, rel=1e-3)
    assert float(st["band_ms"][3] / st["band_ms"][1]) == pytest.approx(1e4, rel=1e-3)
    rel, counts = opt.band_stats()
    assert counts[1] == counts[3] == counts[6] == keel_mod.BAND_MIN_OBS and counts[0] == 0
    assert rel[6] / rel[3] == pytest.approx(5.0, rel=1e-3) and rel[0] == 0.0
    # the bands are trusted now: a step enters at the tensor's overall RMS (from the statistics before the step),
    # the lift capped at GAIN_MAX, and the band weight becomes the update gain
    for sigma, weight in ((loud, 1.0), (quiet, 2.0), (faint, 1.0)):
        opt.zero_grad()
        opt.observe(torch.tensor([sigma]))
        prime(scales[sigma])
        before = {p: opt.state[p]["band_ms"].clone() for p in (w2, w1)}
        raw = {p: p.grad.clone() for p in (w2, w1)}
        opt._bands_step()
        for p in (w2, w1):
            b = band_of(sigma, 8)
            expected = min(float((before[p][8] / before[p][b]).sqrt()), keel_mod.GAIN_MAX)
            assert float((p.grad / raw[p]).mean()) == pytest.approx(expected, rel=1e-4)
            if sigma == faint:
                assert expected == keel_mod.GAIN_MAX
            else:
                assert float(p.grad.square().mean().sqrt()) == pytest.approx(float(before[p][8].sqrt()), rel=1e-3)
        assert opt.update_gain == (weight if weight != 1.0 else None)
    # a step whose micro-batches span two bands is left alone; its band weight is the mean
    opt.zero_grad()
    opt.observe(torch.tensor([quiet, loud]))
    prime(1.0)
    opt._bands_step()
    assert torch.equal(w2.grad, base[0]) and opt.update_gain == pytest.approx(1.5) and opt._mixed_warned
    # a band below BAND_MIN_OBS observations is not trusted: scale 1
    opt.zero_grad()
    opt.observe(torch.tensor([0.75]))
    prime(1.0)
    opt._bands_step()
    assert torch.equal(w2.grad, base[0]) and float(opt.state[w2]["band_n"][5]) == 1.0
    opt.update_gain = None


def test_update_gain_scales_the_step_and_not_the_decay():
    torch.manual_seed(4)
    p0 = torch.randn(8, 6)
    g = torch.randn(8, 6)
    runs = {}
    for gain in (None, 0.5):
        p = torch.nn.Parameter(p0.clone())
        opt = Automagic4([p], lr=1e-3, fused=False, weight_decay=0.0)
        for _ in range(2):  # the first step only measures
            p.grad = g.clone()
            opt.update_gain = gain
            opt.step()
        runs[gain] = p0 - p.detach()
    # (the weights are O(1) and the steps O(1e-3): the deltas carry fp32 rounding of about 1e-7)
    assert torch.allclose(runs[0.5], runs[None] * 0.5, rtol=0.03, atol=2e-7) and runs[None].abs().max() > 5e-4


def test_bands_with_a_low_precision_parameter_use_the_accumulated_gradient():
    torch.manual_seed(5)
    p = torch.nn.Parameter(torch.randn(16, 8).to(torch.bfloat16))
    opt = Keel([p], lr=1e-3, fused=False, bands=4)
    x = torch.randn(8, 3).to(torch.bfloat16)
    before = p.detach().clone()
    for i in range(10):
        opt.zero_grad()
        opt.observe(0.4)
        (p @ x).float().square().mean().backward()
        assert p.grad is None and getattr(p, "_accum_grad", None) is not None  # Automagic 3's accumulation hook
        opt.step()
        assert getattr(p, "_accum_grad", None) is None
    assert not torch.equal(before, p.detach()) and torch.isfinite(p).all()
    assert opt.state[p]["band_n"][band_of(0.4, 4)] == 10 and opt.state[p]["band_ms"].dtype == torch.float32


def test_draw_sigmas_pairs_the_steps_micro_batches():
    gen = torch.Generator().manual_seed(0)
    cfg = _cfg()
    assert pairs_sigma(cfg) and not pairs_sigma(_cfg(keel_pair_sigma=False)) \
        and not pairs_sigma(_cfg(optimizer="automagic4"))
    s1, carry = draw_sigmas(cfg, 1, gen, 0.0, 16, None, None)
    s2, carry2 = draw_sigmas(cfg, 3, gen, 0.0, 16, None, carry)
    s3, _ = draw_sigmas(cfg, 2, gen, 0.0, 16, None, carry2)
    assert carry is not None and carry2 is carry and s1.shape == (1,) and s2.shape == (3,) and s3.shape == (2,)
    assert torch.equal(s2, carry.expand(3)) and torch.equal(s3, carry.expand(2)) and float(s1[0]) == float(carry[0])
    unpaired = _cfg(keel_pair_sigma=False)
    u1, c1 = draw_sigmas(unpaired, 2, gen, 0.0, 16, None, None)
    u2, c2 = draw_sigmas(unpaired, 2, gen, 0.0, 16, None, c1)
    assert c1 is None and c2 is None and u1.shape == (2,) and not torch.equal(u1, u2)


def test_keel_trains_round_trips_and_converts_with_automagic4(tmp_path):
    model, injected, params = _model()
    cfg = _cfg()
    opt = build_optimizer(cfg, params, pairs=lora_mod.lora_pairs(injected))
    sched = build_scheduler(cfg, opt)
    before = [p.detach().clone() for p in params]
    for i in range(10):
        _step(model, params, opt, i, sigma=0.3)
        sched.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, params))
    lr = current_lr(opt, sched)
    assert abs(lr - 1e-4) > 1e-7  # the vote moved it
    st0 = opt.state[params[0]]
    assert st0["band_n"][band_of(0.3, 8)] == 10 and st0["band_ms"].dtype == torch.float32
    sd = opt.state_dict()
    torch.save(sd, tmp_path / "keel.pt")
    saved = torch.load(tmp_path / "keel.pt", weights_only=False)
    opt2 = build_optimizer(cfg, params, pairs=lora_mod.lora_pairs(injected))
    opt2.load_state_dict(saved)
    assert abs(current_lr(opt2, sched) - lr) < 1e-6 * lr
    st2 = opt2.state[params[0]]
    assert torch.equal(st2["band_ms"], st0["band_ms"]) and torch.equal(st2["band_n"], st0["band_n"])  # exact fp32
    _step(model, params, opt2, 11, sigma=0.3)
    assert opt2.state[params[0]]["band_n"][band_of(0.3, 8)] == 11
    # a changed band count starts the statistics fresh; a Keel state under Automagic 4 and back (the same state)
    opt4 = build_optimizer(_cfg(keel_bands=4), params, pairs=lora_mod.lora_pairs(injected))
    opt4.load_state_dict(torch.load(tmp_path / "keel.pt", weights_only=False))
    assert "band_ms" not in opt4.state[params[0]]
    _step(model, params, opt4, 12, sigma=0.3)
    assert opt4.state[params[0]]["band_ms"].shape == (5,)
    opt3 = build_optimizer(_cfg(optimizer="automagic4"), params)
    opt3.load_state_dict(torch.load(tmp_path / "keel.pt", weights_only=False))
    assert abs(current_lr(opt3, sched) - lr) < 1e-6 * lr
    _step(model, params, opt3, 13)
    opt5 = build_optimizer(cfg, params, pairs=lora_mod.lora_pairs(injected))
    opt5.load_state_dict(opt3.state_dict())
    _step(model, params, opt5, 14, sigma=0.3)
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

    keys = {"keel_gauge", "keel_gauge_eps", "keel_cautious", "keel_pair_sigma", "keel_bands", "keel_band_weights",
            "keel_consensus", "keel_consensus_power", "keel_lookahead", "keel_lookahead_pullback", "keel_lookahead_lr",
            "keel_lookahead_grace", "keel_face_gain", "keel_face_gain_max", "keel_face_gain_beta"}
    for bad in (dict(keel_lookahead_pullback=0), dict(keel_lookahead_lr=1.5), dict(keel_lookahead_grace=-1)):
        with pytest.raises(ValueError, match="keel_lookahead"):
            _cfg(**bad).validate()
    for bad in (dict(keel_face_gain=-1), dict(keel_face_gain_max=0.5), dict(keel_face_gain_beta=1)):
        with pytest.raises(ValueError, match="keel_face_gain"):
            _cfg(**bad).validate()
    assert _cfg(keel_face_gain=1.0).wants_regions() and not _cfg().wants_regions()
    with pytest.raises(ValueError, match="keel_consensus"):
        _cfg(keel_consensus=1.2).validate()
    with pytest.raises(ValueError, match="keel_consensus_power"):
        _cfg(keel_consensus_power=0).validate()
    assert keys <= set(R.RACE_KEYS["LR & optimizer"])
    TrainConfig(race=True, probe_every=10, face_refs=["ref.png"], steps=60, race_configs=[
        {"name": "a"}, {"name": "b", "optimizer": "keel", "keel_gauge_eps": 0.1, "keel_cautious": False},
        {"name": "c", "optimizer": "keel", "keel_bands": 1, "keel_pair_sigma": False, "keel_band_weights": ""},
        {"name": "d", "optimizer": "keel", "keel_consensus": 1.0, "keel_consensus_power": 1.0},
        {"name": "e", "optimizer": "keel", "keel_lookahead": True, "keel_lookahead_grace": 0},
        {"name": "f", "optimizer": "keel", "keel_face_gain": 0.5, "keel_face_gain_max": 3}]).validate()
    assert BYTES_PER_PARAM["keel"] == BYTES_PER_PARAM["automagic4"]
    import webui

    assert keys <= set(webui.RACE_LABELS)
    cfg = TrainConfig.from_dict({**_cfg().to_dict(), "dataset_dir": str(tmp_path), "trigger_word": "x"})
    rec = webui.wizard_recommendations(cfg, "character", [], None)
    assert "learning_rate" not in rec  # Keel sets its own rate, like Automagic


# ---------------------------------------------------------------- the accumulation hook and the consensus gate


def _micro_batches(p, grads, opt, sigma=None):
    """K backwards of one step through the hook: p's gradient of micro-batch k is grads[k]."""
    opt.zero_grad()
    for k, g in enumerate(grads):
        if sigma is not None:
            opt.observe(sigma[k] if isinstance(sigma, (list, tuple)) else sigma)
        (p * g).sum().backward()
        assert p.grad is None  # the hook took it


def test_hook_accumulates_the_micro_batches_and_gather_hands_the_sum_back():
    torch.manual_seed(6)
    p = torch.nn.Parameter(torch.randn(5, 4))
    q = torch.nn.Parameter(torch.randn(3))
    opt = Keel([p, q], lr=1e-4, fused=False, bands=1, consensus=0.5)
    grads = [torch.randn(5, 4) for _ in range(3)]
    assert not opt._armed  # the hook waits for zero_grad(): a backward outside a step is left alone
    (p * grads[0]).sum().backward()
    assert p.grad is not None and not hasattr(p, "_accum_grad")
    _micro_batches(p, grads, opt)
    assert opt._votes[p] == 3 and torch.equal(opt._agree[p], sum(g.sign() for g in grads).to(torch.int8))
    assert q not in opt._votes  # q got no gradient
    expected = grads[0] + grads[1] + grads[2]  # the same fp32 sum autograd would have made, in the same order
    assert torch.equal(p._accum_grad, expected)
    opt.gather()
    assert torch.equal(p.grad, expected) and not hasattr(p, "_accum_grad") and q.grad is None
    opt.gather()  # idempotent
    assert torch.equal(p.grad, expected)
    # disarmed after the step: a backward then leaves p.grad alone (another optimizer's step in a race)
    opt.step()
    assert not opt._armed and not opt._agree
    (p * grads[0]).sum().backward()
    assert p.grad is not None and not hasattr(p, "_accum_grad")
    p.grad = None
    # a zero_grad drops a failed step's partial sums (an out-of-memory retry)
    opt.zero_grad()
    (p * grads[0]).sum().backward()
    assert hasattr(p, "_accum_grad")
    opt.zero_grad()
    assert not hasattr(p, "_accum_grad") and opt._armed


def test_two_optimizers_on_the_same_tensors_keep_out_of_each_others_steps():
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(4, 4))
    a = Keel([p], lr=1e-4, fused=False, bands=1, consensus=0.5)
    b = Keel([p], lr=1e-4, fused=False, bands=1, consensus=0.5)
    _micro_batches(p, [torch.randn(4, 4), torch.randn(4, 4)], b)  # a never zeroed: its hook is not armed
    assert p in b._agree and p not in a._agree and b._votes[p] == 2
    b.step()
    assert not hasattr(p, "_accum_grad")


def test_consensus_gate_formula_and_fair_signs():
    agree = torch.tensor([4, 2, 0, -2, -4], dtype=torch.int8)
    gate = consensus_gate(agree, 4, 0.5, 2.0)
    assert torch.allclose(gate, torch.tensor([1.0, 0.625, 0.5, 0.625, 1.0]))
    assert torch.allclose(consensus_gate(agree, 4, 1.0, 2.0), torch.tensor([1.0, 0.25, 0.0, 0.25, 1.0]))
    assert torch.allclose(consensus_gate(agree, 4, 1.0, 1.0), torch.tensor([1.0, 0.5, 0.0, 0.5, 1.0]))
    assert torch.equal(consensus_gate(agree, 4, 0.0, 2.0), torch.ones(5))
    # fair random signs over K = 4: E[gate] = floor + (1 - floor) E[a^2] = 0.5 + 0.5 x 0.25
    torch.manual_seed(8)
    p = torch.nn.Parameter(torch.randn(256, 256))
    opt = Keel([p], lr=1e-4, fused=False, bands=1, consensus=0.5)
    _micro_batches(p, [torch.randn(256, 256).sign() for _ in range(4)], opt)
    gate = opt._element_gain(p)
    assert gate.shape == p.shape and abs(float(gate.mean()) - 0.625) < 0.01
    assert opt.gate_mean() == pytest.approx(float(gate.mean()))
    rows = opt._element_gain(p, 10, 20)
    assert torch.equal(rows, gate[10:20])
    opt.zero_grad()
    assert opt._element_gain(p) is None  # nothing seen this step


def test_consensus_gate_scales_the_update_where_the_micro_batches_disagree():
    torch.manual_seed(9)
    p0 = torch.randn(8, 8)
    base = torch.rand(8, 8) + 0.5
    disagree = torch.zeros(8, 8, dtype=torch.bool)
    disagree[:4] = True  # the top half: the two micro-batches point opposite ways (and do not cancel: a
    g1 = base.clone()  # tensor whose gradient rows are exactly zero is a degenerate case for the factored moment)
    g2 = torch.where(disagree, -0.5 * base, base)
    moved = {}
    for consensus in (0.0, 0.5, 1.0):
        p = torch.nn.Parameter(p0.clone())
        opt = Keel([p], lr=1e-3, fused=False, bands=1, consensus=consensus, weight_decay=0.0)
        for _ in range(2):  # the first step only measures
            _micro_batches(p, [g1, g2], opt)
            opt.step()
        moved[consensus] = (p0 - p.detach()).abs()
    agree_rows, disagree_rows = ~disagree, disagree
    # the agreeing half moves the same whatever the gate (it acts after the second moment and the balance); the
    # disagreeing half moves by the floor: half as far at 0.5, not at all under the hard AND-mask
    assert (moved[0.0] > 0).all()
    assert torch.allclose(moved[0.5][agree_rows], moved[0.0][agree_rows], rtol=0.03, atol=2e-7)
    assert torch.allclose(moved[1.0][agree_rows], moved[0.0][agree_rows], rtol=0.03, atol=2e-7)
    assert torch.allclose(moved[0.5][disagree_rows], moved[0.0][disagree_rows] * 0.5, rtol=0.03, atol=2e-7)
    assert torch.equal(moved[1.0][disagree_rows], torch.zeros_like(moved[1.0][disagree_rows]))
    # a partial disagreement moves by the gate: 3 of 4 agree -> 0.625 of the ungated step
    g3 = torch.where(disagree, -0.5 * base, base)
    moved2 = {}
    for consensus in (0.0, 0.5):
        p = torch.nn.Parameter(p0.clone())
        opt = Keel([p], lr=1e-3, fused=False, bands=1, consensus=consensus, weight_decay=0.0)
        for _ in range(2):
            _micro_batches(p, [g1, g1, g1, g3], opt)
            opt.step()
        moved2[consensus] = (p0 - p.detach()).abs()
    assert torch.allclose(moved2[0.5][disagree_rows], moved2[0.0][disagree_rows] * 0.625, rtol=0.03, atol=2e-7)
    assert torch.allclose(moved2[0.5][agree_rows], moved2[0.0][agree_rows], rtol=0.03, atol=2e-7)


def test_consensus_gate_in_the_bucketed_and_the_slab_update():
    torch.manual_seed(10)
    p0 = torch.randn(64, 16)
    base = torch.rand(64, 16) + 0.5
    g2 = base.clone()
    g2[:32] *= -0.5  # the top rows disagree (without cancelling: see the test above)
    moved = {}
    for bucket_elems in (4_000_000, 256):  # one bucket, then row slabs (bucket_elems below the tensor's size)
        p = torch.nn.Parameter(p0.clone())
        opt = Keel([p], lr=1e-3, fused=False, bands=1, consensus=1.0, weight_decay=0.0, bucket_elems=bucket_elems)
        for _ in range(2):
            _micro_batches(p, [base, g2], opt)
            opt.step()
        moved[bucket_elems] = (p0 - p.detach()).abs()
    for m in moved.values():
        assert torch.equal(m[:32], torch.zeros_like(m[:32])) and (m[32:] > 0).all()
    assert torch.allclose(moved[256], moved[4_000_000], rtol=1e-4, atol=1e-8)


def test_bands_scale_each_micro_batch_by_its_own_band_through_the_hook():
    torch.manual_seed(11)
    p = torch.nn.Parameter(torch.randn(6, 5))
    base = torch.randn(6, 5)
    opt = Keel([p], lr=1e-4, fused=False, bands=8, consensus=0.0, gauge=False)
    quiet, loud = 0.3, 0.9  # bands 3 and 6; the loud band's gradients are 5x the quiet band's
    scales = {quiet: 1.0, loud: 5.0}
    for i in range(2 * keel_mod.BAND_MIN_OBS):
        sigma = (quiet, loud)[i % 2]
        _micro_batches(p, [base * scales[sigma]], opt, sigma=sigma)
        opt.step()
    st = opt.state[p]
    assert st["band_n"][3] == st["band_n"][6] == keel_mod.BAND_MIN_OBS and st["band_n"][8] == 2 * keel_mod.BAND_MIN_OBS
    assert float(st["band_ms"][6] / st["band_ms"][3]) == pytest.approx(25.0, rel=1e-3)
    gain = opt._band_gain[p]
    assert gain.shape == (8,) and float(gain[0]) == 1.0  # an unseen band is not scaled
    expect = {b: min(float((st["band_ms"][8] / st["band_ms"][b]).sqrt()), keel_mod.GAIN_MAX) for b in (3, 6)}
    assert float(gain[3]) == pytest.approx(expect[3], rel=1e-5) and float(gain[6]) == pytest.approx(expect[6], rel=1e-5)
    assert expect[3] > 1.0 > expect[6]
    # a mixed step through the hook: each micro-batch scaled by its own band, both bands observed
    _micro_batches(p, [base, base * 5.0], opt, sigma=[quiet, loud])
    assert opt._cnt[p] == [0, 0, 0, 1, 0, 0, 1, 0]
    opt.gather()
    assert torch.allclose(p.grad, base * expect[3] + base * 5.0 * expect[6], rtol=1e-5)
    n_before = st["band_n"].clone()
    opt.step()
    assert torch.equal(st["band_n"] - n_before, torch.tensor([0, 0, 0, 1, 0, 0, 1, 0, 1], dtype=torch.float32))
    assert not opt._mixed_warned  # the per-step note is for loops without the hook
    # a micro-batch whose images span bands is left alone (no band, no statistics)
    _micro_batches(p, [base], opt, sigma=torch.tensor([quiet, loud]))
    assert opt._cur_band is None and p not in opt._pw
    opt.gather()
    assert torch.equal(p.grad, base)
    opt.step()


def test_bands_fallback_without_the_hook_is_the_per_step_path():
    torch.manual_seed(12)
    p = torch.nn.Parameter(torch.randn(6, 5))
    opt = Keel([p], lr=1e-4, fused=False, bands=8, consensus=0.5, gauge=False)
    for i in range(keel_mod.BAND_MIN_OBS + 1):  # the loop sums the micro-batches itself (the parallel race)
        opt.zero_grad()
        opt.observe(0.3)
        p.grad = torch.randn(6, 5)
        opt.step()
    assert opt.state[p]["band_n"][3] == keel_mod.BAND_MIN_OBS + 1 and "hook" in opt._noted  # the gate note, once


class _Wrap:
    """A LoRA wrapper stand-in for DataParallelLoRA: adapter_params() only."""

    def __init__(self, **ps):
        self.ps = ps

    def adapter_params(self):
        return self.ps


def test_replica_gradients_join_scaled_by_the_band_and_vote():
    from rulztrainer.multigpu import DataParallelLoRA

    torch.manual_seed(13)
    p = torch.nn.Parameter(torch.randn(6, 5))
    pr = torch.nn.Parameter(p.detach().clone())  # the replica's copy: not the optimizer's parameter
    dp = DataParallelLoRA({"w": _Wrap(down=p)}, [(None, {"w": _Wrap(down=pr)}, "cpu")])
    opt = Keel([p], lr=1e-4, fused=False, bands=8, consensus=0.5, gauge=False)
    base = torch.randn(6, 5)
    for i in range(2 * keel_mod.BAND_MIN_OBS):  # the band statistics, through the primary's hook
        _micro_batches(p, [base * (1.0, 5.0)[i % 2]], opt, sigma=(0.3, 0.9)[i % 2])
        opt.gather()
        dp.reduce_grads(take=opt.take_replica)  # no replica gradient yet
        opt.step()
    gain = float(opt._band_gain[p][3])
    assert gain > 1.0
    # a step at sigma 0.3 (paired): one micro-batch on the primary, one on the replica, both agreeing
    _micro_batches(p, [base], opt, sigma=0.3)
    opt.observe(0.3, hook=False)  # the replica's micro-batch, as work() reports it
    assert opt._cur_band == 3 and opt._step_bands == [3, 3]
    (pr * (2.0 * base)).sum().backward()
    opt.gather()
    dp.reduce_grads(take=opt.take_replica)
    assert pr.grad is None
    assert torch.allclose(p.grad, 3.0 * base * gain, rtol=1e-5)  # both shares scaled by the band
    assert opt._votes[p] == 2 and torch.equal(opt._element_gain(p), torch.ones(6, 5))
    opt.step()
    assert opt.take_replica(p, base) is base  # disarmed after the step: untouched
    gain = float(opt._band_gain[p][3])  # the band statistics moved with the step
    # the replica disagreeing everywhere: the gate falls to its floor
    _micro_batches(p, [base], opt, sigma=0.3)
    opt.observe(0.3, hook=False)
    (pr * (-0.5 * base)).sum().backward()
    opt.gather()
    dp.reduce_grads(take=opt.take_replica)
    assert torch.allclose(p.grad, 0.5 * base * gain, rtol=1e-5)
    assert torch.equal(opt._element_gain(p), torch.full((6, 5), 0.5))
    # a step whose observations span bands: the replica's share joins unscaled (and still votes)
    _micro_batches(p, [base], opt, sigma=0.3)
    opt.observe(0.9, hook=False)
    (pr * base).sum().backward()
    opt.gather()
    dp.reduce_grads(take=opt.take_replica)
    assert torch.allclose(p.grad, base * gain + base, rtol=1e-5) and opt._votes[p] == 2
    opt.step()


# ---------------------------------------------------------------- the probe-gated lookahead


class _Average:
    """A LoraEMA stand-in: the parameters and their fp32 shadow."""

    def __init__(self, params):
        self.params = list(params)
        self.shadow = [p.detach().float().clone() for p in self.params]


def test_lookahead_anchors_accepts_holds_and_pulls():
    torch.manual_seed(14)
    p = torch.nn.Parameter(torch.randn(6, 5))
    q = torch.nn.Parameter(torch.randn(3))
    off = Keel([p, q], lr=1e-4, fused=False, bands=1, consensus=0.0)
    assert off.on_probe(50.0, 1.0) is None and not off.state  # the lookahead off: nothing, no state
    opt = Keel([p, q], lr=1e-3, min_lr=1e-5, fused=False, bands=1, consensus=0.0, lookahead=True,
               lookahead_pullback=0.5, lookahead_lr=0.7, lookahead_grace=2)
    assert opt.anchor_score is None and opt.on_probe(None, 1.0) is None  # no face found: nothing
    assert opt.on_probe(50.0, 1.0) == "anchor" and opt.anchor_score == 50.0
    for t in (p, q):  # the anchor: an fp32 copy of the weights; the state initialised as a step would
        st = opt.state[t]
        assert torch.equal(st["anchor"], t.detach()) and st["anchor"].dtype == torch.float32
        assert st["anchor"].data_ptr() != t.data_ptr() and "lr" in st and float(st["anchor_score"]) == 50.0
    for _ in range(2):  # the first step only measures
        _micro_batches(p, [torch.randn(6, 5)], opt)
        opt.step()
    moved = p.detach().clone()
    assert not torch.equal(moved, opt.state[p]["anchor"])  # the step moved the weights, not the anchor
    assert opt.on_probe(51.0, 1.0) == "rising" and opt.anchor_score == 50.0  # not above the band, but up: no pull
    assert opt.on_probe(52.0, 1.0) == "accept" and opt.anchor_score == 52.0  # beats 50 + 1: the new anchor
    assert torch.equal(opt.state[p]["anchor"], moved)
    anchor = moved.clone()
    lr0 = opt.state[p]["lr"].clone()
    kept = {k: v.clone() for k, v in opt.state[p].items()
            if torch.is_tensor(v) and k not in ("anchor", "anchor_score", "lr")}
    # two misses hold, a rise since the previous probe holds too (any rise), the next miss pulls
    assert opt.on_probe(51.0, 1.0) == "hold"
    assert opt.on_probe(50.0, 1.0) == "hold"
    assert opt.on_probe(50.5, 1.0) == "rising"
    with torch.no_grad():
        p.add_(0.1)  # training wandered
    before = p.detach().clone()
    assert opt.on_probe(50.0, 1.0) == "pull"
    assert torch.allclose(p.detach(), anchor + 0.5 * (before - anchor))
    assert torch.allclose(opt.state[p]["lr"], lr0 * 0.7)
    assert opt.anchor_score == 52.0 and torch.equal(opt.state[p]["anchor"], anchor)  # the anchor stays
    for k, v in kept.items():
        assert torch.equal(opt.state[p][k], v), k  # the second moments and the sign history stay
    assert opt.on_probe(49.0, 1.0) == "hold"  # the grace count started over
    assert opt.on_probe(48.0, 1.0) == "hold"
    assert opt.on_probe(47.0, 1.0) == "pull"
    # the lr factor stays within the rails
    for st in opt.state.values():
        st["lr"].fill_(1e-5)
    opt._misses = 3
    assert opt.on_probe(46.0, 1.0) == "pull" and float(opt.state[p]["lr"]) == pytest.approx(1e-5)


def test_lookahead_follows_the_weight_average():
    torch.manual_seed(15)
    p = torch.nn.Parameter(torch.randn(4, 4))
    avg = _Average([p])
    with torch.no_grad():
        avg.shadow[0].add_(0.25)  # the average lags the raw weights
    opt = Keel([p], lr=1e-3, fused=False, bands=1, consensus=0.0, lookahead=True, lookahead_grace=0)
    assert opt.on_probe(60.0, 1.0, avg) == "anchor"
    assert torch.equal(opt.state[p]["anchor"], avg.shadow[0])  # the weights that scored: the average
    assert opt.state[p]["anchor"].data_ptr() != avg.shadow[0].data_ptr()
    anchor = avg.shadow[0].clone()
    raw0 = p.detach().clone()
    with torch.no_grad():
        avg.shadow[0].sub_(0.1)
    sh0 = avg.shadow[0].clone()
    assert opt.on_probe(59.0, 1.0, avg) == "pull"  # grace 0: the first miss pulls
    assert torch.allclose(p.detach(), anchor + 0.5 * (raw0 - anchor))
    assert torch.allclose(avg.shadow[0], anchor + 0.5 * (sh0 - anchor))


def test_an_anchor_before_the_first_step_leaves_training_bitwise():
    ma, inja, pa = _model(4)
    mb, injb, pb = _model(4)
    oa = build_optimizer(_cfg(keel_lookahead=True), pa, pairs=lora_mod.lora_pairs(inja))
    ob = build_optimizer(_cfg(), pb, pairs=lora_mod.lora_pairs(injb))
    assert oa.on_probe(40.0, 1.0) == "anchor"  # probe_at_start: the state is initialised before any gradient
    for s in range(6):
        _step(ma, pa, oa, s)
        _step(mb, pb, ob, s)
        assert all(torch.equal(a, b) for a, b in zip(pa, pb)), s
    assert oa.on_probe(45.0, 1.0) == "accept"
    assert all(torch.equal(oa.state[a]["anchor"], a.detach()) for a in pa)


def test_lookahead_state_round_trips_and_is_dropped_without_it():
    ma, inja, pa = _model(3)
    opt = build_optimizer(_cfg(keel_lookahead=True), pa, pairs=lora_mod.lora_pairs(inja))
    for s in range(2):
        _step(ma, pa, opt, s)
    assert opt.on_probe(55.0, 1.0) == "anchor"
    _step(ma, pa, opt, 2)
    sd = opt.state_dict()
    mb, injb, pb = _model(3)
    with torch.no_grad():
        for a, b in zip(pa, pb):
            b.copy_(a)
    ob = build_optimizer(_cfg(keel_lookahead=True), pb, pairs=lora_mod.lora_pairs(injb))
    ob.load_state_dict(sd)
    assert ob.anchor_score == 55.0
    for a, b in zip(pa, pb):
        assert torch.equal(ob.state[b]["anchor"], opt.state[a]["anchor"])
        assert ob.state[b]["anchor"].dtype == torch.float32
    # the restored optimizer goes on from the restored anchor
    ob._misses = 2
    before = [b.detach().clone() for b in pb]
    assert ob.on_probe(54.0, 1.0) == "pull"
    for b, x, a in zip(pb, before, pa):
        assert torch.allclose(b.detach(), torch.lerp(x, opt.state[a]["anchor"], 0.5), atol=1e-7)
    # the lookahead off: the anchor is dropped on load; Automagic 4 loads the state and steps
    mc, injc, pc = _model(3)
    oc = build_optimizer(_cfg(), pc, pairs=lora_mod.lora_pairs(injc))
    oc.load_state_dict(sd)
    assert all("anchor" not in oc.state[c] and "anchor_score" not in oc.state[c] for c in pc)
    assert oc.anchor_score is None and oc.on_probe(60.0, 1.0) is None
    md, injd, pd = _model(3)
    od = build_optimizer(_cfg(optimizer="automagic4"), pd)
    od.load_state_dict(sd)
    _step(md, pd, od, 3)


def test_trainer_hands_the_probe_decision_to_keel_and_syncs_the_replicas_after_a_pull():
    from types import SimpleNamespace

    from rulztrainer.trainer import Trainer

    torch.manual_seed(16)
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = Keel([p], lr=1e-3, fused=False, bands=1, consensus=0.0, lookahead=True, lookahead_grace=0)
    synced = []
    dp = SimpleNamespace(sync_weights=lambda: synced.append(True))
    me = SimpleNamespace(controller=SimpleNamespace(combined_ema=50.0), cfg=_cfg(likeness_noise_band=2.0), dp=dp,
                         ema=None)
    Trainer._keel_probe(me, opt, None, True)
    assert opt.anchor_score == 50.0 and not synced
    me.controller.combined_ema = 51.0  # within the band, but up since the previous probe: no pull
    Trainer._keel_probe(me, opt, None, True)
    assert not synced
    me.controller.combined_ema = 49.0  # a miss, and grace 0 pulls: the replicas follow
    Trainer._keel_probe(me, opt, None, True)
    assert synced == [True]
    me.controller.combined_ema = 48.0
    Trainer._keel_probe(me, opt, None, False)  # the parallel race: no sync
    assert synced == [True]
    me.controller = None  # no probes: nothing
    Trainer._keel_probe(me, opt, None, True)
    plain = Automagic4([p], lr=1e-3, fused=False)
    me.controller = SimpleNamespace(combined_ema=60.0)
    Trainer._keel_probe(me, plain, None, True)  # another optimizer: nothing
    assert synced == [True]


def test_an_async_probe_anchors_the_copy_that_scored_and_pulls_the_live_weights():
    from types import SimpleNamespace

    from rulztrainer.trainer import Trainer

    model, injected, params = _model(17)
    opt = build_optimizer(_cfg(keel_lookahead=True, keel_lookahead_grace=0), params,
                          pairs=lora_mod.lora_pairs(injected))
    sent = lora_mod.lora_state(injected)  # what went to the probe GPU
    with torch.no_grad():
        for p in params:  # training went on while it rendered
            p.add_(0.5)
    live = [p.detach().clone() for p in params]
    me = SimpleNamespace(controller=SimpleNamespace(combined_ema=50.0), cfg=_cfg(likeness_noise_band=1.0), dp=None,
                         ema=None, injected=injected)
    Trainer._keel_probe(me, opt, scored=sent)
    assert opt.anchor_score == 50.0
    for name, m in injected.items():
        for part, p in m.adapter_params().items():
            assert torch.equal(opt.state[p]["anchor"], sent[name]["params"][part].float())  # the copy, not the live
            assert not torch.equal(opt.state[p]["anchor"], p.detach().float())
    me.controller.combined_ema = 49.0  # a miss, grace 0: the pull moves the live weights toward the copy
    Trainer._keel_probe(me, opt, scored=sent)
    for p, x in zip(params, live):
        assert torch.allclose(p.detach().float(), torch.lerp(x.float(), opt.state[p]["anchor"], 0.5), atol=1e-2)


# ---------------------------------------------------------------- the face-token gain


def _face_pair(seed: int, n_in: int = 8, n_out: int = 6, rank: int = 4):
    torch.manual_seed(seed)
    m = lora_mod.LoRALinear(torch.nn.Linear(n_in, n_out), rank, float(rank))
    for p in m.adapter_params().values():
        torch.nn.init.normal_(p)
    return m


def test_face_token_mask_is_row_major_over_the_cells():
    assert lora_mod.face_token_mask(None, 3, 4) is None
    m = lora_mod.face_token_mask((1, 0, 3, 2), 3, 4)
    assert m.shape == (12,) and m.dtype == torch.bool and m.nonzero().flatten().tolist() == [1, 2, 5, 6]


def test_face_tap_reports_the_face_and_the_whole_gradient_energy():
    m = _face_pair(20)
    got = []
    layout = {"sample": 0, "img": (1, 3), "txt": (0, 1)}  # tokens 1 to 3 are the image, 0 and 4 text
    mask = torch.tensor([True, False, True])
    lora_mod.arm_face_tap({"a": m}, [mask, None], lambda: layout, lambda p, ef, et: got.append((p, ef, et)))
    x = torch.randn(2, 5, 8)
    w = torch.randn(2, 5, 6)
    (m(x) * w).sum().backward()
    assert len(got) == 1 and got[0][0] is m.lora_up.weight
    h = x @ m.lora_down.weight.T
    du = w * m.scale  # the gradient of the LoRA output before its scale
    g = du[0].T @ h[0]  # sample 0 alone: sample 1 has no face
    gf = (du[0, 1:4] * mask[:, None].float()).T @ h[0, 1:4]
    assert torch.allclose(got[0][2], g.pow(2).sum(), rtol=1e-4)
    assert torch.allclose(got[0][1], gf.pow(2).sum(), rtol=1e-4)
    assert torch.allclose(m.lora_up.weight.grad, m.scale * (g + du[1].T @ h[1]), rtol=1e-4)  # the real gradient
    # an image-only input (a double-stream block's image side) is all image tokens; a text-only one is skipped
    got.clear()
    (m(x[:, 1:4]) * w[:, 1:4]).sum().backward()
    assert len(got) == 1 and torch.allclose(got[0][2], (du[0, 1:4].T @ h[0, 1:4]).pow(2).sum(), rtol=1e-4)
    got.clear()
    (m(x[:, :2]) * w[:, :2]).sum().backward()
    assert not got
    with torch.no_grad():  # no gradient: no hook
        m(x)
    lora_mod.disarm_face_tap({"a": m})
    assert m.face_sink is None and m.face_masks is None
    (m(x) * w).sum().backward()
    assert not got


def test_face_tap_under_activation_checkpointing_reports_each_layer_once_with_the_same_energies():
    from rulztrainer import alignment as A

    out = {}
    for ckpt in (False, True):
        model, injected, params = _model(21)
        model.gradient_checkpointing = ckpt
        model._rz_taps = taps = A.Taps()
        names = {m.lora_up.weight: n for n, m in injected.items()}
        got = []
        mask = torch.zeros(16, dtype=torch.bool)  # 4 x 4 latent cells
        mask[5:11] = True
        lora_mod.arm_face_tap(injected, [mask], lambda: taps.layout,
                              lambda p, ef, et: got.append((names[p], float(ef), float(et))))
        g = torch.Generator().manual_seed(1)
        x = torch.randn(1, 128, 4, 4, generator=g).to(torch.bfloat16)
        c = torch.randn(1, 5, 32, generator=g).to(torch.bfloat16)
        taps.begin([[]], attention=False)
        loss = model(x, torch.tensor([300.0]), c).float().pow(2).mean()
        taps.stop()
        loss.backward()
        taps.end()
        lora_mod.disarm_face_tap(injected)
        assert sorted(n for n, _, _ in got) == sorted(injected)  # every LoRA layer once
        assert all(0 < ef <= et * (1 + 1e-4) for _, ef, et in got)
        out[ckpt] = {n: (ef, et) for n, ef, et in got}
    for n in injected:
        assert out[False][n] == pytest.approx(out[True][n], rel=1e-3), n


def _pairs(seed: int, n: int = 3):
    torch.manual_seed(seed)
    pairs = [(torch.nn.Parameter(torch.randn(4, 8)), torch.nn.Parameter(torch.randn(6, 4))) for _ in range(n)]
    return pairs, [t for pair in pairs for t in pair]


def test_face_gain_scales_each_pair_by_its_share_against_the_median():
    pairs, params = _pairs(22)
    ref, rparams = _pairs(22)
    kw = dict(lr=1e-3, fused=False, bands=1, consensus=0.0, gauge=False, cautious=False)
    with pytest.raises(ValueError, match="face_gain_max"):
        Keel(params, face_gain_max=0.5, **kw)
    with pytest.raises(ValueError, match="face_gain_beta"):
        Keel(params, face_gain_beta=1.0, **kw)
    opt = Keel(params, pairs=pairs, face_gain=1.0, face_gain_max=2.0, face_gain_beta=0.0, **kw)
    plain = Keel(rparams, pairs=ref, **kw)
    grads = [torch.randn_like(t) for t in params]
    gains = (1.6, 1.0, 0.5)  # shares 0.8, 0.5, 0.2 against the median 0.5, the last clamped at 1 / 2
    for s in range(3):
        deltas = []
        for o, ps in ((opt, params), (plain, rparams)):
            o.zero_grad(set_to_none=True)
            if o is opt and s == 2:  # the face tap's reports, two micro-batches each: summed
                for (_, up), share in zip(pairs, (0.8, 0.5, 0.2)):
                    opt.face_energy(up, torch.tensor(share), torch.tensor(1.0))
                    opt.face_energy(up, torch.tensor(share), torch.tensor(1.0))
            before = [t.detach().clone() for t in ps]
            for t, g in zip(ps, grads):
                t.grad = g.clone()
            o.step()
            deltas.append([t.detach() - b for t, b in zip(ps, before)])
        for i, ((down, up), gain) in enumerate(zip(pairs, gains)):
            for k in (2 * i, 2 * i + 1):
                if s < 2:
                    assert torch.equal(deltas[0][k], deltas[1][k])
                else:
                    assert torch.allclose(deltas[0][k], gain * deltas[1][k], rtol=1e-4, atol=1e-6), (i, k)  # a few ulps of p
            if s == 2:
                assert float(opt.state[up]["face_gain"]) == pytest.approx(gain)
                assert float(opt.state[down]["face_gain"]) == pytest.approx(gain)  # the pair's down the same
    assert opt.face_gain_stats() == pytest.approx((0.5, 1.0, 1.6, 3))
    assert not opt._face_e  # the step's reports are cleared
    opt.face_energy(pairs[0][1], torch.tensor(1.0), torch.tensor(1.0))  # outside a step: ignored
    assert not opt._face_e
    # the EMA keeps most of the previous gain; a pair reported without energy keeps its gain
    opt.face_gain_beta = 0.5
    opt.zero_grad(set_to_none=True)
    for (_, up), share in zip(pairs[:2], (0.8, 0.2)):
        opt.face_energy(up, torch.tensor(share), torch.tensor(1.0))
    opt.face_energy(pairs[2][1], torch.tensor(0.0), torch.tensor(0.0))
    for t, g in zip(params, grads):
        t.grad = g.clone()
    opt.step()
    # the median of the two reported shares is the lower middle (0.2, torch's median): gains 4 -> 2 (clamped) and 1
    assert float(opt.state[pairs[0][1]]["face_gain"]) == pytest.approx(0.5 * 1.6 + 0.5 * 2.0)
    assert float(opt.state[pairs[1][1]]["face_gain"]) == pytest.approx(0.5 * 1.0 + 0.5 * 1.0)
    assert float(opt.state[pairs[2][1]]["face_gain"]) == pytest.approx(0.5)


def test_face_gain_on_without_a_report_leaves_training_bitwise():
    ma, inja, pa = _model(5)
    mb, injb, pb = _model(5)
    oa = build_optimizer(_cfg(keel_face_gain=1.0), pa, pairs=lora_mod.lora_pairs(inja))
    ob = build_optimizer(_cfg(), pb, pairs=lora_mod.lora_pairs(injb))
    for s in range(4):
        _step(ma, pa, oa, s)
        _step(mb, pb, ob, s)
        assert all(torch.equal(a, b) for a, b in zip(pa, pb)), s
    assert oa.face_gain_stats() is None


def test_face_gain_state_round_trips_and_is_dropped_without_it():
    pairs, params = _pairs(23)
    kw = dict(lr=1e-3, fused=False, bands=1, consensus=0.0)
    opt = Keel(params, pairs=pairs, face_gain=1.0, face_gain_beta=0.0, **kw)
    grads = [torch.randn_like(t) for t in params]
    for s in range(2):
        opt.zero_grad(set_to_none=True)
        for (_, up), share in zip(pairs, (0.9, 0.3, 0.6)):
            opt.face_energy(up, torch.tensor(share), torch.tensor(1.0))
        for t, g in zip(params, grads):
            t.grad = g.clone()
        opt.step()
    sd = opt.state_dict()
    for face_gain in (1.0, 0.0):
        pairs2, params2 = _pairs(23)
        o2 = Keel(params2, pairs=pairs2, face_gain=face_gain, **kw)
        o2.load_state_dict(sd)
        for a, b in zip(params, params2):
            if face_gain > 0:
                assert torch.equal(o2.state[b]["face_gain"], opt.state[a]["face_gain"])
                assert o2.state[b]["face_gain"].dtype == torch.float32 and o2.state[b]["face_gain"].dim() == 0
            else:
                assert "face_gain" not in o2.state[b]
    _, params3 = _pairs(23)
    o3 = Automagic4(params3, lr=1e-3, fused=False)
    o3.load_state_dict(sd)
    for t, g in zip(params3, grads):
        t.grad = g.clone()
    o3.step()


def test_the_trainer_installs_the_layout_taps_for_the_face_gain():
    from types import SimpleNamespace

    from rulztrainer import alignment as A
    from rulztrainer.trainer import Trainer

    model, _, _ = _model(6)
    me = SimpleNamespace(race_rows=None, ft=None, race_parallel=False, replicas=[(model, None, None)])
    Trainer._setup_alignment(me, _cfg(keel_face_gain=1.0), [])
    taps = model._rz_taps
    assert isinstance(taps, A.Taps) and not taps.hidden_layers and not taps.attn_layers
    model2, _, _ = _model(6)
    me.replicas = [(model2, None, None)]
    Trainer._setup_alignment(me, _cfg(), [])
    assert not hasattr(model2, "_rz_taps")
    me.race_parallel = True  # the parallel race: no tap
    Trainer._setup_alignment(me, _cfg(keel_face_gain=1.0), [])
    assert not hasattr(model2, "_rz_taps")
