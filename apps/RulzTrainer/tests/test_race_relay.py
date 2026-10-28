"""Relay race (race_follow): nobody is dropped, each round's winner hands its LoRA, optimizer state, weight average and
likeness history to the others, who keep their own settings and seed; the last round's winner is the run's LoRA."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer import race as R  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.ema import LoraEMA  # noqa: E402
from rulztrainer.likeness import LikenessController  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=3, text_dim=32, freq_dim=16, latent_channels=128)


def relay_cfg(rows, **kw):
    base = dict(race=True, race_follow=True, probe_every=10, face_refs=["ref.png"], steps=20, race_configs=rows)
    return TrainConfig(**{**base, **kw})


def test_the_round_winner_leads_and_the_last_round_decides():
    s = R.RaceSchedule(["a", "b", "c"], warmup=2, drop_lowest=True, patience=2, total_steps=20, follow=True)
    assert not s.drop_lowest and not s.patience and not s.keeps_all  # a relay ignores the other rules
    for i, score in enumerate((10.0, 30.0, 20.0)):
        s.record(i, score, 10, best=score, best_step=10, since_improve=5)
    assert s.alive == [0, 1, 2] and s.just_dropped == [] and s.history[-1]["leader"] == "b" and s.leader == 1
    assert "b leads" in R.describe_round(s.history[-1], s)
    for i, score in enumerate((40.0, 35.0, 50.0)):
        s.record(i, score, 20, best=score, best_step=20, since_improve=0)
    assert s.done and s.alive == [2] and s.finished == {0, 1} and s.just_dropped == [0, 1]
    assert s.history[-1]["final"] and s.decide() == 2 and s.state()["winner"] == "c" and s.state()["follow"]
    assert "c wins the last round" in R.describe_round(s.history[-1], s)


def test_a_tie_keeps_the_leader_and_a_missing_score_never_leads():
    s = R.RaceSchedule(["a", "b", "c"], 0, total_steps=30, follow=True)
    for i, score in enumerate((10.0, 30.0, None)):
        s.record(i, score, 10)
    assert s.leader == 1
    for i, score in enumerate((30.0, 30.0, 30.0)):
        s.record(i, score, 20)
    assert s.leader == 1  # everyone tied: the leader stays
    for i, score in enumerate((None, None, None)):
        s.record(i, score, 30)
    assert s.done and s.alive == [1]


def test_relay_checks_and_cost():
    relay_cfg([{"name": "s1", "seed": 1}, {"name": "s2", "seed": 2, "learning_rate": 2e-4}]).validate()
    with pytest.raises(ValueError, match="shape: remove lora_rank"):
        relay_cfg([{"name": "a"}, {"name": "b", "lora_rank": 32}]).validate()
    with pytest.raises(ValueError, match="multiple of probe_every"):
        relay_cfg([{"name": "a"}, {"name": "b"}], steps=25).validate()
    with pytest.raises(ValueError, match="race_parallel"):
        relay_cfg([{"name": "a"}, {"name": "b"}], race_parallel=True).validate()
    relay_cfg([{"name": "a"}, {"name": "b", "lora_dropout": 0.1}], steps=10).validate()  # dropout keeps the shape
    assert R.estimate_steps(4, 0, 10, 20, drop_lowest=False)["total"] == 80


def test_seed_rows():
    assert R.parse_seeds("1-3, 7 9") == [1, 2, 3, 7, 9]
    assert R.seed_rows([1, 2, 2]) == [{"name": "seed1", "seed": 1}, {"name": "seed2", "seed": 2}]
    with pytest.raises(ValueError):
        R.parse_seeds("5-1")
    relay_cfg(R.seed_rows(R.parse_seeds("1-4"))).validate()


def _step(model, inj, opt, ema, seed):
    g = torch.Generator().manual_seed(seed)
    x, t, c = torch.randn(1, 128, 4, 4, generator=g), torch.tensor([300.0]), torch.randn(1, 5, 32, generator=g)
    opt.zero_grad(set_to_none=True)
    model(x, t, c).float().pow(2).mean().backward()
    opt.step()
    ema.update()


def test_a_follower_continues_exactly_where_the_winner_was():
    """b takes over a's state: b's next step (same data) is a's next step, while b keeps its own controller lr factor."""
    from rulztrainer import archs

    torch.manual_seed(0)
    model = MOSSImageTransformer(**TINY)
    names = archs.arch_by_name("moss_image").lora_module_names(model, ["q", "k", "v", "o"], None)
    wrappers = L.inject_lora(model, 4, 4.0, [], names=names)
    inj = {n: wrappers[n] for n in names}
    variants = []
    for i, seed in enumerate((1, 2)):
        cfg = TrainConfig(lora_rank=4, lora_alpha=4.0, seed=seed)
        v = R.Variant(i, f"s{seed}", cfg, {}, names)
        v.init_lora(wrappers, seed)
        R.swap_in(v, wrappers)
        v.state["optimizer"] = torch.optim.AdamW(L.lora_parameters(inj), lr=1e-2)
        v.state["ema"] = LoraEMA(inj, 0.9)
        v.state["controller"] = LikenessController(cfg)
        _step(model, inj, v.state["optimizer"], v.state["ema"], seed=10 + i)
        v.state["controller"].update(10, 50.0 + 10 * i, None)
        R.swap_out(v, wrappers)
        v.offloaded = R.offload_optimizer(v.state["optimizer"])
        R.move_average(v.state["ema"], to_cpu=True)
        variants.append(v)
    a, b = variants
    b.state["controller"].lr_factor = 1.3
    assert R.follow_leader(b, a) == []  # b won the round: a continues from it
    assert all(torch.equal(a.weights[n][p], b.weights[n][p]) for n in names for p in b.weights[n])
    assert a.state["controller"].combined_ema == b.state["controller"].combined_ema
    assert a.state["controller"].cfg is a.cfg and a.state["controller"].lr_factor == 1.0

    def next_step(v):
        R.swap_in(v, wrappers)
        R.onload_optimizer(v.state["optimizer"], v.offloaded)
        R.move_average(v.state["ema"], to_cpu=False)
        _step(model, inj, v.state["optimizer"], v.state["ema"], seed=99)
        R.swap_out(v, wrappers)
        v.offloaded = R.offload_optimizer(v.state["optimizer"])
        R.move_average(v.state["ema"], to_cpu=True)
        return {n: {p: t.clone() for p, t in parts.items()} for n, parts in v.weights.items()}, \
            [s.clone() for s in v.state["ema"].shadow]

    (wa, ea), (wb, eb) = next_step(a), next_step(b)
    assert all(torch.allclose(wa[n][p], wb[n][p], atol=1e-7) for n in names for p in wa[n])
    assert all(torch.allclose(x, y, atol=1e-7) for x, y in zip(ea, eb))
    a.state["optimizer"] = torch.optim.SGD(L.lora_parameters(inj), lr=1e-2)
    assert "starts over" in R.follow_leader(b, a)[0]  # another kind of optimizer: its state cannot be carried over


def test_trainer_hands_over_after_each_round_and_finishes_on_the_last_winner():
    from rulztrainer import trainer as T

    sched = R.RaceSchedule(["a", "b"], 0, total_steps=20, follow=True)
    variants = [SimpleNamespace(name=n, idx=i, alive=True, state={}) for i, n in enumerate("ab")]
    calls = []
    fake = SimpleNamespace(race=SimpleNamespace(schedule=sched, variants=variants, active=variants[0]))
    fake._race_record = lambda v, c: T.Trainer._race_record(fake, v, c)
    fake._race_park = lambda v: calls.append(("park", v.name))
    fake._race_follow = lambda w: calls.append(("follow", w.name))
    fake._race_activate = lambda v: (calls.append(("activate", v.name)), setattr(fake.race, "active", v))
    fake._race_drop = lambda v: (calls.append(("drop", v.name)), setattr(v, "alive", False))
    fake._race_finish = lambda: calls.append(("finish", fake.race.active.name))
    fake._race_report = lambda: None

    def probe(score, step):
        fake.controller = SimpleNamespace(combined_ema=score, last_step=step, best=score, best_step=step,
                                          since_improve=0)
        return T.Trainer._race_after_probe(fake)

    assert probe(30.0, 10) is variants[1] and calls == [("park", "a"), ("activate", "b")]
    calls.clear()
    assert probe(40.0, 10) is variants[0]  # b led round 1: a continues from it
    assert calls == [("park", "b"), ("follow", "b"), ("activate", "a")]
    calls.clear()
    probe(50.0, 20)
    calls.clear()
    assert probe(45.0, 20) is variants[0]  # the last round: a wins, b (training now) is saved and left
    assert calls == [("drop", "b"), ("activate", "a"), ("finish", "a")]


def test_relay_racers_share_one_best_checkpoint():
    from rulztrainer import trainer as T

    sched = R.RaceSchedule(["a", "b"], 0, total_steps=20, follow=True)
    fake = SimpleNamespace(race=SimpleNamespace(schedule=sched, best_saved={}), stage_tag="")
    fake._race_running = lambda: True
    fake._file_tag = lambda: "_a"
    keeps = T.Trainer._race_keeps_best
    assert keeps(fake, "", 60.0) and not keeps(fake, "", 55.0) and keeps(fake, "", 61.0)
    assert keeps(fake, "face", 40.0)  # face and body are tracked apart
    assert T.Trainer._best_tag(fake) == ""
    sched.follow = False
    assert keeps(fake, "", 1.0) and T.Trainer._best_tag(fake) == "_a"
