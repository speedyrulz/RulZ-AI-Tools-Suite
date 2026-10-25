"""Configuration race (rulztrainer.race): settings parsing and checks, the round schedule, each configuration's items,
and swapping configurations of different rank and layers in and out of one set of LoRA wrappers without either one
noticing (training A, then B, then A gives A exactly what training A alone gives)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer import race as R  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.dataset import Item  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=3, text_dim=32, freq_dim=16, latent_channels=128)


def race_cfg(rows, **kw):
    base = dict(race=True, probe_every=10, face_refs=["ref.png"], steps=40, race_configs=rows)
    return TrainConfig(**{**base, **kw})


def test_overrides_parse_and_format():
    over = R.parse_overrides('learning_rate=2e-4, lora_targets=[q, k, v], lora_blocks="10-31,40", flip_augment=true, '
                             'timestep_sampling=identity, lora_rank=32')
    assert over == {"learning_rate": 2e-4, "lora_targets": ["q", "k", "v"], "lora_blocks": "10-31,40",
                    "flip_augment": True, "timestep_sampling": "identity", "lora_rank": 32}
    assert R.parse_overrides(R.format_overrides(over)) == over
    assert R.parse_overrides("") == {}
    with pytest.raises(ValueError):
        R.parse_overrides("learning_rate 2e-4")


def test_validation():
    race_cfg([{"name": "a"}, {"name": "b", "learning_rate": 2e-4, "lora_rank": 32}]).validate()
    for rows, kw, msg in (
            ([{"name": "a"}], {}, "at least 2"),
            ([{"name": "a"}, {"name": "b", "steps": 5}], {}, "shares: steps"),
            ([{"name": "a"}, {"name": "a b"}], {}, "letters"),
            ([{"name": "a"}, {"name": "a"}], {}, "unique"),
            ([{"name": "a"}, {"name": "b", "automagic_groups": "bad"}], {}, "configuration 'b'"),
            ([{"name": "a"}, {"name": "b"}], {"probe_every": 0}, "probes"),
            ([{"name": "a"}, {"name": "b"}], {"resume_state": "x"}, "starts fresh"),
            ([{"name": "a"}, {"name": "b"}, {"name": "c"}], {"steps": 15}, "at least 20 steps")):
        with pytest.raises(ValueError, match=msg):
            race_cfg(rows, **kw).validate()
    names = [(n, c.learning_rate, c.lora_rank, c.race) for n, c, _ in R.variant_configs(
        race_cfg([{"name": "a"}, {"learning_rate": 3e-4, "lora_rank": 8}]))]
    assert names == [("a", 1e-4, 16, False), ("c2", 3e-4, 8, False)]


def test_step_estimate():
    est = R.estimate_steps(10, 2, 100, 3000)  # the example: 10 configurations, 2 warm-up probes, a probe every 100
    assert est["rounds"] == 11 and est["race_own_steps"] == 1100 and est["race_steps"] == 7400
    assert est["total"] == 9300 and est["probes"] == 74
    assert R.estimate_steps(2, 0, 50, 50)["total"] == 100  # 1 round of 2, the winner is done


def test_schedule_drops_the_lowest_after_each_round():
    s = R.RaceSchedule(["a", "b", "c"], warmup=1)
    seq = [(0, 50.0), (1, 60.0), (2, 40.0)]  # round 1: warm-up, nobody leaves
    for idx, score in seq:
        assert s.current == idx and s.record(idx, score) is None
    assert s.history[-1]["dropped"] is None and s.alive == [0, 1, 2]
    for idx, score in [(0, 55.0), (1, 61.0)]:
        assert s.record(idx, score) is None
    assert s.record(2, 70.0) == 0 and s.alive == [1, 2]  # round 2: a has the lowest
    assert s.current == 1
    s.record(1, None)  # no face found: counts as the lowest
    assert s.record(2, 10.0) == 1 and s.done and s.current == 2
    assert s.state()["winner"] == "c" and s.state()["dropped"] == {"a": 2, "b": 3}
    with pytest.raises(ValueError):
        s.record(2, 1.0)  # the race is over
    tie = R.RaceSchedule(["a", "b"], warmup=0)
    tie.record(0, 50.0)
    assert tie.record(1, 50.0) == 1  # a tie drops the later configuration
    with pytest.raises(ValueError):
        R.RaceSchedule(["a", "b"], 0).record(1, 1.0)  # not b's turn


def test_variant_items(tmp_path):
    def item(cat, flip=True):
        return Item(image=tmp_path / "x.png", caption="", width=64, height=64, latent_path=tmp_path / "l.st",
                    latent_flip_path=tmp_path / "f.st" if flip else None, text_path=tmp_path / "t.st",
                    category=cat, region_path=tmp_path / "regions" / f"{cat}_regions.safetensors")

    items = [item("image"), item("face_crop"), item("bust_crop")]
    plain = R.variant_items(items, TrainConfig())
    assert [it.category for it in plain] == ["image"] and plain[0].latent_flip_path is None
    full = R.variant_items(items, TrainConfig(crop_augment="face+bust", flip_augment=True), tmp_path / "other")
    assert [it.category for it in full] == ["image", "face_crop", "bust_crop"]
    assert full[0].latent_flip_path is not None and full[1].region_path == tmp_path / "other" / "face_crop_regions.safetensors"
    assert [it.category for it in R.variant_items(items, TrainConfig(crop_augment="face"))] == ["image", "face_crop"]
    assert R.region_dir(TrainConfig()) != R.region_dir(TrainConfig(face_loss_weight=5.0))


def _inputs(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 128, 4, 4, generator=g), torch.tensor([300.0]), torch.randn(1, 5, 32, generator=g)


def _setup(layouts):
    """A tiny MOSS model with wrappers over the union of the variants' modules, and the variants."""
    torch.manual_seed(0)
    model = MOSSImageTransformer(**TINY)
    union = sorted({n for _, _, names in layouts for n in names})
    wrappers = L.inject_lora(model, 4, 4.0, [], names=union)
    variants = []
    for i, (name, rank, names) in enumerate(layouts):
        v = R.Variant(i, name, TrainConfig(lora_rank=rank, lora_alpha=float(rank)), {}, names)
        v.init_lora(wrappers, seed=42)
        variants.append(v)
    return model, wrappers, variants


def _train_step(model, inj, opt, seed):
    opt.zero_grad(set_to_none=True)
    x, t, c = _inputs(seed)
    model(x, t, c).float().pow(2).mean().backward()
    opt.step()


def _names(model, targets, blocks):
    from rulztrainer import archs

    return archs.arch_by_name("moss_image").lora_module_names(model, targets, blocks)


def test_switching_configurations_is_invisible_to_each_of_them():
    probe = MOSSImageTransformer(**TINY)
    a_names = _names(probe, ["q", "k", "v", "o"], None)
    b_names = _names(probe, ["ffn_up_gate", "ffn_down", "q"], [1, 2])
    layouts = [("a", 4, a_names), ("b", 8, b_names)]

    def run(order):
        model, wrappers, variants = _setup(layouts)
        by_name = {v.name: v for v in variants}
        model.train()
        opts, active = {}, None
        for k, v in enumerate(by_name[n] for n in order):
            if active is not None and active is not v:
                R.swap_out(active, wrappers)
                active.offloaded = R.offload_optimizer(opts[active.name])
            R.swap_in(v, wrappers)
            inj = {n: wrappers[n] for n in v.names}
            if v.name not in opts:
                opts[v.name] = torch.optim.AdamW(L.lora_parameters(inj), lr=1e-2)
            else:
                R.onload_optimizer(opts[v.name], v.offloaded)
            assert all(not wrappers[n].trainable for n in wrappers if n not in v.names)
            assert all(wrappers[n].rank == v.rank and wrappers[n].lora_down.weight.shape[0] == v.rank for n in v.names)
            _train_step(model, inj, opts[v.name], seed=10 + k)
            active = v
        R.swap_out(active, wrappers)
        return {v.name: ({n: w["lora_down"].clone() for n, w in v.weights.items()},
                         {n: w["lora_up"].clone() for n, w in v.weights.items()})
                for v in variants}

    raced = run(["a", "b", "a"])
    # A alone (steps with the same data seeds as its turns: 10 and 12), B alone (its one turn: seed 11)
    alone_a = run_alone(layouts, "a", [10, 12])
    alone_b = run_alone(layouts, "b", [11])
    for name, ref in (("a", alone_a), ("b", alone_b)):
        for got, want in zip(raced[name], ref):
            assert set(got) == set(want)
            assert all(torch.allclose(got[n], want[n], atol=1e-6) for n in got), name


def run_alone(layouts, which, seeds):
    model, wrappers, variants = _setup(layouts)
    v = next(x for x in variants if x.name == which)
    model.train()
    R.swap_in(v, wrappers)
    inj = {n: wrappers[n] for n in v.names}
    opt = torch.optim.AdamW(L.lora_parameters(inj), lr=1e-2)
    for s in seeds:
        _train_step(model, inj, opt, seed=s)
    R.swap_out(v, wrappers)
    return ({n: w["lora_down"] for n, w in v.weights.items()}, {n: w["lora_up"] for n, w in v.weights.items()})


def test_replicas_and_views_follow_the_configuration(tmp_path):
    probe = MOSSImageTransformer(**TINY)
    layouts = [("a", 4, _names(probe, ["q"], None)), ("b", 8, _names(probe, ["ffn_down"], None))]
    model, wrappers, (va, vb) = _setup(layouts)
    replica = MOSSImageTransformer(**TINY)
    replica.load_state_dict({k: v for k, v in model.state_dict().items() if ".lora_" not in k and ".base." not in k},
                            strict=False)
    replica.load_state_dict({k.replace(".base.", "."): v for k, v in model.state_dict().items() if ".base." in k},
                            strict=False)
    rwrap = L.inject_lora(replica, 4, 4.0, [], names=list(wrappers))
    R.swap_in(vb, wrappers)
    L.mirror_lora(wrappers, rwrap)
    for n in wrappers:
        assert rwrap[n].trainable == wrappers[n].trainable and rwrap[n].rank == wrappers[n].rank
        assert torch.equal(rwrap[n].lora_down.weight, wrappers[n].lora_down.weight)
    x, t, c = _inputs(3)
    with torch.no_grad():
        for w in wrappers.values():
            if w.trainable:
                w.lora_up.weight.normal_()
        L.mirror_lora(wrappers, rwrap)
        assert torch.allclose(model(x, t, c), replica(x, t, c))
    R.swap_out(vb, wrappers)
    path = tmp_path / "b.safetensors"
    L.save_lora(vb.view(), path)  # a parked configuration exports straight from its CPU weights
    from safetensors.torch import load_file

    state = load_file(str(path))
    name = vb.names[0]
    assert torch.equal(state[f"diffusion_model.{name}.lora_up.weight"], vb.weights[name]["lora_up"].to(torch.bfloat16))
    assert state[f"diffusion_model.{name}.lora_down.weight"].shape[0] == 8


def test_race_status_carries_face_and_body_behind_the_combined_score():
    """The Monitor's race table shows face and body next to the combined score: [normalized, raw] per variant."""
    from types import SimpleNamespace

    from rulztrainer.likeness import LikenessController
    from rulztrainer.trainer import Trainer

    cfg = TrainConfig(probe_every=10, probe_ema=0.0, likeness_normalize=True)
    c1, c2 = LikenessController(cfg, face_ceiling=70.0, body_ceiling=60.0), LikenessController(cfg)
    c1.update(0, 15.0, 8.0)  # floors
    c1.update(10, 42.5, 34.0)  # halfway to both ceilings
    v1 = SimpleNamespace(name="a", state={"controller": c1, "step": 10}, overrides={"learning_rate": 1e-4})
    v2 = SimpleNamespace(name="b", state={"controller": c2}, overrides={})
    v3 = SimpleNamespace(name="c", state={}, overrides={})
    rr = SimpleNamespace(schedule=SimpleNamespace(state=lambda: {"names": ["a", "b", "c"]}), active=v2,
                         variants=[v1, v2, v3], estimate={"total": 100})
    out = Trainer._race_status(SimpleNamespace(race=rr))
    assert abs(out["scores"]["a"] - 50.0) < 1e-9 and out["scores"]["b"] is None and out["scores"]["c"] is None
    assert [round(x, 6) for x in out["face"]["a"]] == [50.0, 42.5] and [round(x, 6) for x in out["body"]["a"]] == [50.0, 34.0]
    assert out["face"]["b"] == [None, None] and "c" not in out["face"]
    assert out["steps"] == {"a": 10, "b": None, "c": 0} and out["settings"]["a"] == "learning_rate=0.0001"


def test_race_records_the_step_each_score_was_probed_at():
    s = R.RaceSchedule(["a", "b"], 0)
    assert s.record(0, 40.0, 120) is None
    assert s.steps == {0: 120}
    assert s.record(1, 50.0, 240) == 0  # round 1 ends: a (the lower score) is out
    entry = s.history[0]
    assert entry["scores"] == {"a": 40.0, "b": 50.0} and entry["steps"] == {"a": 120, "b": 240}
    assert s.steps == {}  # the next round starts clean
    s2 = R.RaceSchedule(["a", "b"], 0)  # an old caller without steps still works
    s2.record(0, 1.0)
    s2.record(1, 2.0)
    assert s2.history[0]["steps"] == {"a": None, "b": None}
