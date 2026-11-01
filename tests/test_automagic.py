import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as lora_mod  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402
from rulztrainer.trainer import build_optimizer, build_scheduler, current_lr  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16, latent_channels=128)


def _step(model, params, opt):
    x = torch.randn(1, 128, 4, 4).to(torch.bfloat16)
    c = torch.randn(1, 5, 32).to(torch.bfloat16)
    loss = model(x, torch.tensor([300.0]), c).float().pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return loss.item()


@pytest.mark.parametrize("name", ["automagic3", "automagic4"])
def test_automagic_trains_and_adapts_lr(tmp_path, name):
    torch.manual_seed(0)
    model = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    injected = lora_mod.inject_lora(model, 4, 4, ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
    params = list(lora_mod.lora_parameters(injected))
    cfg = TrainConfig(optimizer=name, learning_rate=1e-4, automagic_polarity_history=4, steps=20)
    opt = build_optimizer(cfg, params)
    sched = build_scheduler(cfg, opt)
    model.train()
    before = [p.detach().clone() for p in params]
    for _ in range(8):
        _step(model, params, opt)
        sched.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, params))
    lr = current_lr(opt, sched)
    assert cfg.automagic_min_lr <= lr <= cfg.automagic_max_lr
    assert abs(lr - 1e-4) > 1e-7  # the vote moved it
    assert all(torch.isfinite(p).all() for p in params)
    # state round trip (what save_state / resume_state do)
    sd = opt.state_dict()
    torch.save(sd, tmp_path / "opt.pt")
    opt2 = build_optimizer(cfg, params)
    opt2.load_state_dict(torch.load(tmp_path / "opt.pt", weights_only=False))
    assert abs(current_lr(opt2, sched) - lr) < 1e-6 * lr
    _step(model, params, opt2)


def test_layer_type_groups_each_adapt_their_own_lr():
    """automagic_groups: by_type gives Krea 2's text fusion, attention, feed-forward and the rest their own groups."""
    from rulztrainer import archs
    from rulztrainer import lora as L
    from rulztrainer.config import TrainConfig
    from rulztrainer.krea2.model import Krea2Transformer
    from rulztrainer.trainer import build_optimizer

    krea = archs.arch_by_name("krea2")
    model = Krea2Transformer(in_channels=64, num_layers=2, attention_head_dim=16, num_attention_heads=4,
                             num_key_value_heads=2, intermediate_size=96, timestep_embed_dim=32, text_hidden_dim=40,
                             num_text_layers=3, text_num_attention_heads=4, text_num_key_value_heads=4,
                             text_intermediate_size=56, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
                             axes_dims_rope=(4, 6, 6))
    targets = list(krea.default_targets) + ["embed"]
    inj = L.inject_lora(model, 4, 4.0, targets, names=krea.lora_module_names(model, targets, None))
    groups = L.lora_param_groups_by(inj, 1e-4, krea.param_group_of)
    assert [g["name"] for g in groups] == ["attn", "mlp", "other", "text"]
    assert sum(len(g["params"]) for g in groups) == 2 * len(inj)
    assert krea.param_group_of("transformer_blocks.0.attn.to_gate") == "attn"
    assert krea.param_group_of("text_fusion.refiner_blocks.0.ff.down") == "text"
    moss = archs.arch_by_name("moss_image")
    assert [moss.param_group_of(n) for n in ("layers.3.proj.q", "layers.3.proj.ffn_down", "text_embedder.0.qkv")] \
        == ["attn", "mlp", "text"]
    for name in ("automagic3", "automagic4"):
        opt = build_optimizer(TrainConfig(optimizer=name), groups)
        assert len(opt.param_groups) == 4  # kept as groups, each with the configured start lr
        single = build_optimizer(TrainConfig(optimizer=name, lora_plus_ratio=4.0), L.lora_param_groups(inj, 1e-4, 4.0))
        assert len(single.param_groups) == 1  # LoRA+ groups still collapse: Automagic owns the lr


# ---------------------------------------------------------------- Automagic 4


def _tiny_moss_lora(seed=0):
    torch.manual_seed(seed)
    model = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    injected = lora_mod.inject_lora(model, 4, 4, ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
    model.train()
    return model, list(lora_mod.lora_parameters(injected))


def _grads(params, seed):
    gen = torch.Generator().manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=gen).to(p.dtype)


def test_automagic4_measures_first_then_trains_and_round_trips(tmp_path):
    model, params = _tiny_moss_lora()
    cfg = TrainConfig(optimizer="automagic4", learning_rate=1e-4, automagic_polarity_history=4, steps=20)
    opt = build_optimizer(cfg, params)
    sched = build_scheduler(cfg, opt)
    before = [p.detach().clone() for p in params]
    _step(model, params, opt)
    assert all(torch.equal(a, b) for a, b in zip(before, params))  # the first step only measures
    assert all("sens_ref" in opt.state[p] for p in params) and opt._ref is not None
    for _ in range(7):
        _step(model, params, opt)
        sched.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, params))
    lr = current_lr(opt, sched)
    assert cfg.automagic_min_lr <= lr <= cfg.automagic_max_lr
    assert all(torch.isfinite(p).all() for p in params)
    ref = float(opt._ref)
    sd = opt.state_dict()
    torch.save(sd, tmp_path / "opt.pt")
    opt2 = build_optimizer(cfg, params)
    opt2.load_state_dict(torch.load(tmp_path / "opt.pt", weights_only=False))
    assert abs(current_lr(opt2, sched) - lr) < 1e-6 * lr
    assert abs(float(opt2._ref) - ref) < 1e-6 * ref  # the reference rides along (no measurement-only step on resume)
    snapshot = [p.detach().clone() for p in params]
    _step(model, params, opt2)
    assert any(not torch.equal(a, b) for a, b in zip(snapshot, params))


def test_automagic4_accepts_an_automagic3_state(tmp_path):
    model, params = _tiny_moss_lora()
    cfg3 = TrainConfig(optimizer="automagic3", learning_rate=1e-4, automagic_polarity_history=4)
    opt3 = build_optimizer(cfg3, params)
    for _ in range(3):
        _step(model, params, opt3)
    torch.save(opt3.state_dict(), tmp_path / "opt3.pt")
    lr3 = opt3.get_avg_learning_rate()
    cfg4 = TrainConfig(optimizer="automagic4", learning_rate=1e-4, automagic_polarity_history=4)
    opt4 = build_optimizer(cfg4, params)
    opt4.load_state_dict(torch.load(tmp_path / "opt3.pt", weights_only=False))
    assert opt4._ref is None and abs(opt4.get_avg_learning_rate() - lr3) < 1e-9
    before = [p.detach().clone() for p in params]
    _step(model, params, opt4)
    assert all(torch.equal(a, b) for a, b in zip(before, params))  # one measurement-only step
    assert all(opt4.state[p]["hist_fill"] == 3 for p in params)  # the sign history was kept
    _step(model, params, opt4)
    assert any(not torch.equal(a, b) for a, b in zip(before, params))


def _run(opt, params, steps, seed=0):
    for i in range(steps):
        _grads(params, seed + i)
        opt.step()


def test_automagic4_bucketed_and_slab_paths_match_per_tensor():
    from rulztrainer.optimizers.automagic4 import Automagic4

    shapes = [(16, 64), (16, 64), (64, 16), (64, 16), (16, 64), (32,)]
    runs = {}
    for name, kw in (("per_tensor", dict(bucketing=False, bucket_elems=0)),
                     ("bucketed", dict(bucketing=True, bucket_elems=0)),
                     ("small_buckets", dict(bucketing=True, bucket_elems=1500)),  # one tensor per bucket
                     ("slabs", dict(bucketing=True, bucket_elems=256))):  # 1024-element tensors in 4-row slabs
        torch.manual_seed(1)
        params = [torch.nn.Parameter(torch.randn(*s)) for s in shapes]
        opt = Automagic4(params, lr=1e-3, polarity_history=4, weight_decay=0.01, fused=False, **kw)
        _run(opt, params, 12)
        runs[name] = ([p.detach().clone() for p in params], opt.get_learning_rates()[0], float(opt._ref))
    ref_params, ref_lr, ref_ref = runs["per_tensor"]
    for name, (ps, lr, ref) in runs.items():
        assert all(torch.allclose(a, b, rtol=1e-5, atol=1e-7) for a, b in zip(ps, ref_params)), name
        assert abs(lr - ref_lr) < 1e-5 * ref_lr and abs(ref - ref_ref) < 1e-4 * ref_ref, name
    assert ref_lr != 1e-3  # the controller moved


def test_automagic4_holds_the_lr_on_gradient_noise():
    """The regression for ai-toolkit's v4 runaway: on pure sign noise upstream climbed 1.6% per step (x25 in 200
    steps); a zero-mean vote holds it."""
    from rulztrainer.optimizers.automagic4 import Automagic4

    for vote in ("graded", "extremes"):
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.zeros(256, 256)), torch.nn.Parameter(torch.zeros(128, 64))]
        opt = Automagic4(params, lr=1e-4, min_lr=1e-8, max_lr=1e3, polarity_history=8, fused=False, vote=vote)
        _run(opt, params, 200, seed=100)
        lr = opt.get_learning_rates()[0]
        assert abs(math.log(lr / 1e-4)) < 0.1, (vote, lr)
        assert all(torch.equal(opt.state[p]["lr"], opt.state[params[0]]["lr"]) for p in params)


def test_automagic4_raises_the_lr_on_a_trend_and_lowers_it_on_overshoot():
    from rulztrainer.optimizers.automagic4 import Automagic4

    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(128, 128))
    opt = Automagic4([p], lr=1e-4, min_lr=1e-8, max_lr=1e3, polarity_history=8, fused=False)
    sign = torch.randn(128, 128).sign()
    for i in range(40):  # a consistent direction: every window agrees
        p.grad = sign * (1.0 + 0.1 * torch.randn(128, 128).abs())
        opt.step()
    peak = opt.get_learning_rates()[0]
    assert peak > 2e-4
    assert peak <= 1e-4 * math.exp(0.05 * 40) * 1.001  # at most 5% per step
    for i in range(40):  # perfect alternation: the overshoot signature
        p.grad = sign * (1.0 + 0.1 * torch.randn(128, 128).abs()) * (-1.0 if i % 2 else 1.0)
        opt.step()
    assert opt.get_learning_rates()[0] < peak * 0.5


def test_automagic4_dead_tensors_do_not_collapse_the_reference():
    """94% of the group's elements have an all-zero gradient (LoRA downs while their ups are zero); upstream put the
    reference at 1e-15 and froze every live tensor."""
    from rulztrainer.optimizers.automagic4 import Automagic4

    torch.manual_seed(0)
    dead = torch.nn.Parameter(torch.randn(1024, 64))
    dead0 = dead.detach().clone()
    live = torch.nn.Parameter(torch.zeros(64, 64))
    opt = Automagic4([dead, live], lr=1e-3, polarity_history=4, fused=False)
    gen = torch.Generator().manual_seed(5)
    for _ in range(6):
        dead.grad = torch.zeros_like(dead)
        live.grad = torch.randn(live.shape, generator=gen)
        opt.step()
    assert live.abs().max() > 1e-3  # 5 real steps of about lr each
    assert float(opt._ref) > 1e-6  # the reference comes from the live tensor, not the dead one
    assert torch.equal(dead, dead0)  # a zero gradient moves nothing


def test_automagic4_late_parameter_starts_at_the_adapted_lr():
    from rulztrainer import likeness as L
    from rulztrainer.optimizers.automagic4 import Automagic4

    torch.manual_seed(0)
    a, b = torch.nn.Parameter(torch.zeros(64, 64)), torch.nn.Parameter(torch.zeros(64, 64))
    opt = Automagic4([a, b], lr=1e-4, polarity_history=4, fused=False)
    for i in range(3):
        a.grad = torch.randn(64, 64)
        opt.step()
    L.apply_lr_ratio(opt, None, 0.5)
    assert abs(opt.get_learning_rates()[0] - 5e-5) < 1e-9
    a.grad, b.grad = torch.randn(64, 64), torch.randn(64, 64)
    opt.step()
    assert torch.equal(opt.state[b]["lr"], opt.state[a]["lr"])  # not the configured 1e-4
    assert float(opt.state[b]["lr"]) < 6e-5
    assert "sens_ref" in opt.state[b]


def test_automagic4_survives_the_race_replacing_its_state_tensors():
    """race.move_optimizer / offload / follow_leader replace every state tensor with a copy; the next step must see
    the copies (group lr, reference) and give the same result as an untouched twin."""
    from rulztrainer.optimizers.automagic4 import Automagic4

    def make():
        torch.manual_seed(0)
        ps = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(4)]
        return ps, Automagic4(ps, lr=1e-3, polarity_history=4, fused=False)

    ps1, o1 = make()
    ps2, o2 = make()
    _run(o1, ps1, 6)
    _run(o2, ps2, 6)
    for st in o2.state.values():  # what move_optimizer does
        for k, t in list(st.items()):
            if torch.is_tensor(t):
                st[k] = t.clone()
    _run(o1, ps1, 3, seed=50)
    _run(o2, ps2, 3, seed=50)
    assert all(torch.equal(a, b) for a, b in zip(ps1, ps2))
    assert o1.get_learning_rates() == o2.get_learning_rates()
    # follow_leader: a cleared state refilled with clones of another optimizer's state
    ps3, o3 = make()
    _run(o3, ps3, 2)
    o3.state.clear()
    for p1, p3 in zip(ps1, ps3):
        o3.state[p3] = {k: t.detach().clone() if torch.is_tensor(t) else t for k, t in o1.state[p1].items()}
        p3.data.copy_(p1.data)
    assert o3.get_learning_rates() == o1.get_learning_rates()  # read before any step: no stale cache
    _run(o1, ps1, 2, seed=70)
    _run(o3, ps3, 2, seed=70)
    assert all(torch.equal(a, b) for a, b in zip(ps1, ps3))
    assert o1.get_learning_rates() == o3.get_learning_rates()


def test_automagic4_graded_vote_is_zero_mean_on_fair_signs():
    from rulztrainer.optimizers.automagic4 import Automagic4

    torch.manual_seed(3)
    H, n = 8, 200_000
    opt = Automagic4([torch.nn.Parameter(torch.zeros(8))], polarity_history=H, fused=False)
    bits = torch.randint(0, 2, (n, H), dtype=torch.uint8)
    ones = bits.sum(1, dtype=torch.uint8).unsqueeze(0)
    flips = (bits[:, 1:] ^ bits[:, :-1]).sum(1, dtype=torch.uint8).unsqueeze(0)
    w = torch.rand(1, n)
    for vote in ("graded", "extremes"):
        opt.vote = vote
        num, den = opt._vote_sums(w, ones, flips, H)
        assert abs(float(num / den)) < 0.01, vote  # the per-element vote has std ~1, the mean 1/sqrt(n) ~ 0.002


def test_automagic4_config_validation_and_groups():
    for bad in (dict(automagic_sensitivity_quantile=1.5), dict(automagic_vote="mean"), dict(automagic_max_lr_change=0),
                dict(optimizer="automagic5"), dict(automagic_min_lr=1e-2, automagic_max_lr=1e-3),
                dict(automagic_polarity_history=1)):
        with pytest.raises(ValueError):
            TrainConfig(**{"optimizer": "automagic4", **bad}).validate()
    TrainConfig(optimizer="automagic4").validate()
    from rulztrainer import archs

    krea = archs.arch_by_name("krea2")
    assert krea.param_group_of("txt_in.linear_1") == "text"
    assert krea.param_group_of("time_embed.linear_1") == "other"
    fibo = archs.arch_by_name("fibo2")
    assert fibo.param_group_of("t_embedder.mlp.0") == "other"
    assert fibo.param_group_of("layers.3.feed_forward.w1") == "mlp"


def test_automagic4_relay_into_one_group_unifies_to_the_median():
    """A by_type winner relayed into a single-group follower carries several rates into one group: they become the
    median, not whichever rate the first parameter happens to hold."""
    from rulztrainer.optimizers.automagic4 import Automagic4

    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(3)]
    opt = Automagic4(ps, lr=1e-3, polarity_history=4, fused=False)
    _run(opt, ps, 2)
    for p, lr in zip(ps, (1e-3, 4e-3, 2e-3)):
        opt.state[p]["lr"] = torch.tensor(lr)  # what follow_leader does with a by_type winner's state
    assert opt.get_learning_rates() == [pytest.approx(2e-3)]
    assert all(float(opt.state[p]["lr"]) == pytest.approx(2e-3) for p in ps)
    _run(opt, ps, 1, seed=9)
    assert len({float(opt.state[p]["lr"]) for p in ps}) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_automagic4_resume_restores_the_exact_lr_and_reference(tmp_path, dtype):
    """torch's load casts state to the param dtype; a bf16 model's lr and reference must still come back exact, and the
    first resumed step must equal an un-resumed twin's."""
    from rulztrainer.optimizers.automagic4 import Automagic4

    def make():
        torch.manual_seed(0)
        ps = [torch.nn.Parameter(torch.randn(16, 32, dtype=dtype)) for _ in range(3)]
        ps.append(torch.nn.Parameter(torch.randn(24, dtype=dtype)))
        return ps, Automagic4(ps, lr=1e-3, polarity_history=4, fused=False)

    ps1, o1 = make()
    ps2, o2 = make()
    _run(o1, ps1, 7)
    _run(o2, ps2, 7)
    torch.save(o2.state_dict(), tmp_path / "opt.pt")
    o2 = Automagic4(ps2, lr=1e-3, polarity_history=4, fused=False)
    o2.load_state_dict(torch.load(tmp_path / "opt.pt", weights_only=False))
    for p1, p2 in zip(ps1, ps2):
        assert o2.state[p2]["lr"].dtype == torch.float32 and torch.equal(o2.state[p2]["lr"], o1.state[p1]["lr"])
        assert torch.equal(o2.state[p2]["sens_ref"], o1.state[p1]["sens_ref"])
    if dtype == torch.float32:  # (bf16 params take a stochastically rounded step: no bitwise twin)
        _run(o1, ps1, 2, seed=30)
        _run(o2, ps2, 2, seed=30)
        assert all(torch.equal(a, b) for a, b in zip(ps1, ps2))


def test_automagic4_resume_applies_a_lowered_max_lr_at_once(tmp_path):
    from rulztrainer.optimizers.automagic4 import Automagic4

    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(16, 32))]
    opt = Automagic4(ps, lr=1e-3, polarity_history=4, fused=False)
    _run(opt, ps, 3)
    torch.save(opt.state_dict(), tmp_path / "opt.pt")
    low = Automagic4(ps, lr=1e-3, max_lr=2e-4, polarity_history=8, fused=False)  # new window: no vote for 8 steps
    low.load_state_dict(torch.load(tmp_path / "opt.pt", weights_only=False))
    assert low.get_learning_rates() == [pytest.approx(2e-4)]


def test_automagic4_param_group_added_later_trains():
    from rulztrainer.optimizers.automagic4 import Automagic4

    torch.manual_seed(0)
    a, b = torch.nn.Parameter(torch.randn(16, 32)), torch.nn.Parameter(torch.randn(16, 32))
    opt = Automagic4([a], lr=1e-3, polarity_history=4, fused=False)
    _run(opt, [a], 3)
    opt.add_param_group({"params": [b]})
    before = b.detach().clone()
    _run(opt, [a, b], 2, seed=5)
    assert not torch.equal(before, b) and len(opt.get_learning_rates()) == 2


def test_automagic4_next_window_inherits_the_reference():
    """A fine-tune window's new optimizer starts from the previous one's reference: its first step trains."""
    from rulztrainer.optimizers.automagic4 import Automagic4

    torch.manual_seed(0)
    a, b = torch.nn.Parameter(torch.randn(16, 32)), torch.nn.Parameter(torch.randn(16, 32))
    first = Automagic4([a], lr=1e-3, polarity_history=4, fused=False)
    _run(first, [a], 3)
    nxt = Automagic4([b], lr=1e-3, polarity_history=4, fused=False)
    nxt.inherit_reference(first)
    before = b.detach().clone()
    _run(nxt, [b], 1, seed=5)
    assert not torch.equal(before, b)
    fresh = Automagic4([a], lr=1e-3, polarity_history=4, fused=False)
    before = a.detach().clone()
    _run(fresh, [a], 1, seed=5)
    assert torch.equal(before, a)  # without it the first step only measures


def test_resume_refuses_a_state_of_another_optimizer_kind(tmp_path, monkeypatch):
    from rulztrainer import trainer as T

    _, params = _tiny_moss_lora()
    opt = build_optimizer(TrainConfig(optimizer="automagic4"), params)
    torch.save({"optimizer": opt.state_dict(), "config": {"optimizer": "adamw8bit"}, "step": 3},
               tmp_path / "train_state.pt")
    monkeypatch.setattr(T.lora_mod, "load_lora_weights", lambda inj, path: len(inj))
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = TrainConfig(optimizer="automagic4")
    with pytest.raises(ValueError, match="optimizer adamw8bit"):
        tr.load_state(tmp_path, [], opt, None, None, None)


def test_automagic4_edge_cases_empty_param_dead_first_step_and_group_after_inherit():
    from rulztrainer.optimizers.automagic4 import Automagic4

    torch.manual_seed(0)
    empty, a = torch.nn.Parameter(torch.zeros(0, 8)), torch.nn.Parameter(torch.randn(16, 32))
    opt = Automagic4([empty, a], lr=1e-3, polarity_history=4, fused=False)
    _run(opt, [empty, a], 3)  # a zero-element parameter is skipped, not a division by zero
    # every gradient zero on the first step: no reference yet, so the next step measures again
    b = torch.nn.Parameter(torch.randn(8, 8))
    opt = Automagic4([b], lr=1e-3, polarity_history=4, fused=False)
    b.grad = torch.zeros(8, 8)
    opt.step()
    assert opt._ref is None
    b.grad = torch.randn(8, 8)
    opt.step()
    assert opt._ref is not None and float(opt._ref) > 1e-6
    # add_param_group after inherit_reference keeps the inherited reference
    c, d = torch.nn.Parameter(torch.randn(16, 32)), torch.nn.Parameter(torch.randn(16, 32))
    nxt = Automagic4([c], lr=1e-3, polarity_history=4, fused=False)
    nxt.inherit_reference(opt)
    nxt.add_param_group({"params": [d]})
    before = c.detach().clone()
    _run(nxt, [c, d], 1, seed=5)
    assert not torch.equal(before, c)
