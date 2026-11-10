"""Region class weights and region steering (regionwatch.py): the fixed weights, and each steering rule on its own -
cap, ahead, boost, prior - with the limits they share. Parts tensors are made by hand: no parser needed."""
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import regionwatch as RW  # noqa: E402
from rulztrainer.config import TrainConfig, apply_preset  # noqa: E402
from rulztrainer.dataset import Item  # noqa: E402

STEER = {"regions": ("background", "clothes"), "cap": 0.6, "ahead": 0.7, "boost": 0.0, "prior": 0.0, "min": 0.3,
         "step": 0.2, "strength": 1.0}


def _items(tmp_path, n):
    return [Item(image=tmp_path / f"img{i}.png", caption=f"cap {i}", width=512, height=512,
                 latent_path=tmp_path / "l", latent_flip_path=None, text_path=tmp_path / "t") for i in range(n)]


def _parts(h, w, classes):
    p = torch.zeros(7, h, w, dtype=torch.uint8)
    for c in range(6):
        p[c][classes == c] = 255
    return p


def _sample(base, face=1.0, hair=1.0, skin=1.0, clothes=1.0, background=1.0):
    """One sample's region losses and cells (COLUMNS order) from a factor per region on a loss `base`."""
    cells = [20.0, 30.0, 40.0, 50.0, 200.0, 90.0]
    subject = (20 * face + 30 * hair + 40 * skin) / 90
    return [base * v for v in (face, hair, skin, clothes, background, subject)], cells


def _run(watch, items, epochs, factors, rng, state):
    """Feeds `epochs` epochs; factors(i, epoch) -> the _sample factors of picture i. state: {"epoch", "step"}."""
    summary = None
    for _ in range(epochs):
        for i, it in enumerate(items):
            sigma = rng.random()
            losses, cells = _sample(0.1 + 0.9 * sigma, **factors(i, state["epoch"]))
            state["step"] += 1
            watch.observe([it.key], [sigma], [losses], [cells])
        state["epoch"] += 1
        summary = watch.end_epoch(state["step"])
    return summary


def _watch(tmp_path, n=20, **steer):
    items = _items(tmp_path, n)
    watch = RW.RegionWatch(items, tmp_path, steer={**STEER, **steer})
    rng, state = random.Random(0), {"epoch": 0, "step": 0}
    _run(watch, items, 4, lambda i, e: {}, rng, state)  # the baseline: 80 samples
    assert watch.frozen
    return watch, items, rng, state


def test_class_weights_parse_and_refuse_nonsense():
    assert RW.parse_class_weights("") == {} and RW.parse_class_weights(None) == {}
    assert RW.parse_class_weights(" Face=3, hair = 1.5 ;background=0.3,") == {"face": 3.0, "hair": 1.5,
                                                                             "background": 0.3}
    assert RW.parse_class_weights("clothes=0") == {"clothes": 0.0}
    for bad in ("face", "nose=2", "face=x", "face=-1", "face=11", "face=0,hair=0,skin=0,clothes=0,background=0"):
        with pytest.raises(ValueError):
            RW.parse_class_weights(bad)
    assert RW.merge_gains({"face": 2.0}, None, {"face": 1.5, "background": 0.5}) == {"face": 3.0, "background": 0.5}


def test_weight_maps_have_mean_one_and_keep_the_ratio():
    classes = torch.zeros(4, 6, dtype=torch.long)
    classes[:, 3:] = 3  # face on the right half, background on the left
    parts, like = _parts(4, 6, classes), torch.zeros(3, 2, 4, 6)
    maps = RW.weight_maps([parts, None, parts], [{"face": 3.0, "background": 0.3}, {"face": 3.0}, {}], like)
    assert maps.shape == (3, 1, 4, 6)
    assert float(maps[0].mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(maps[0, 0, 0, 5] / maps[0, 0, 0, 0]) == pytest.approx(10.0, rel=1e-4)  # 3 against 0.3
    assert bool((maps[1] == 1).all()) and bool((maps[2] == 1).all())  # no parts map; nothing to weight
    assert RW.weight_maps([parts], [{"face": 1.0}], like[:1]) is None  # every weight 1
    assert RW.weight_maps([None], [{"face": 3.0}], like[:1]) is None
    assert RW.weight_maps([_parts(5, 6, torch.zeros(5, 6))], [{"face": 3.0}], like[:1]) is None  # another grid
    masked = RW.weight_maps([parts], [{"face": 0.0, "background": 0.0}], like[:1])
    assert bool((masked == 1).all())  # all of it at weight 0: left as it is, never a division by zero
    clothes = torch.full((4, 6), 5, dtype=torch.long)  # the parser's "others" counts as clothes
    clothes[0] = 0
    cw = RW.weight_maps([_parts(4, 6, clothes)], [{"clothes": 0.5}], like[:1])
    assert float(cw[0, 0, 0, 0] / cw[0, 0, 1, 0]) == pytest.approx(2.0, rel=1e-4)
    bg = RW.background_mask([parts, None], like[:2])
    assert bool((bg[0, 0, :, :3] == 1).all()) and bool((bg[0, 0, :, 3:] == 0).all()) and float(bg[1].sum()) == 0.0
    assert RW.background_mask([None], like[:1]) is None


def test_cap_rule_turns_the_background_down_in_steps_and_lets_it_back(tmp_path):
    watch, items, rng, state = _watch(tmp_path, ahead=0.0)
    assert not watch.steering
    # the subject has learned 30%, the background 40%: over the cap of 0.6 x 30% = 18%
    over = lambda i, e: dict(face=0.7, hair=0.7, skin=0.7, background=0.6)  # noqa: E731
    summary = _run(watch, items, 1, over, rng, state)
    gains = [st.gain.get("background", 1.0) for st in watch.stats.values()]
    assert watch.steering and all(g == pytest.approx(0.8) for g in gains)  # one step of 20%
    assert summary["steer"]["background"]["lowered"] == 20 and summary["steer"]["clothes"]["lowered"] == 0
    summary = _run(watch, items, 12, over, rng, state)
    cohort = summary["steer"]["background"]["cohort"]
    assert cohort == pytest.approx(0.6 / (1 - 0.6 * 0.3), abs=0.03)  # error left over the least allowed
    lowest = summary["steer"]["background"]["lowest"]
    assert lowest == pytest.approx(cohort ** 2, abs=0.04) and lowest >= 0.3  # the target, never under the minimum
    assert "steering: background turned down on 20 pictures" in RW.describe(summary)
    assert [g["background"] for g in watch.gains([items[0].key, "nobody"])[:1]] == [pytest.approx(lowest, abs=1e-3)]
    assert watch.gains(["nobody"]) == [{}]
    # back within the cap: the weights return, more slowly, and the flag clears
    summary = _run(watch, items, 14, lambda i, e: dict(face=0.7, hair=0.7, skin=0.7, background=0.9), rng, state)
    assert summary["steer"]["background"]["lowered"] == 0 and not watch.steering


def test_cap_rule_waits_for_the_subject_and_respects_the_minimum(tmp_path):
    watch, items, rng, state = _watch(tmp_path, ahead=0.0, min=0.5, strength=4.0)
    _run(watch, items, 6, lambda i, e: dict(face=0.98, hair=0.98, skin=0.98, background=0.6), rng, state)
    assert not watch.steering  # the subject has learned 2%: too early for a ratio
    summary = _run(watch, items, 12, lambda i, e: dict(face=0.7, hair=0.7, skin=0.7, background=0.5), rng, state)
    assert summary["steer"]["background"]["lowest"] == pytest.approx(0.5)  # a hard pull stops at the minimum


def test_ahead_rule_turns_down_only_the_picture_running_ahead(tmp_path):
    watch, items, rng, state = _watch(tmp_path, cap=0.0)

    def factors(i, epoch):
        return dict(background=max(0.3, 1.0 - 0.1 * max(0, epoch - 4))) if i == 0 else {}

    summary = _run(watch, items, 20, factors, rng, state)
    assert watch.stats[items[0].key].gain["background"] < 0.7
    assert all("background" not in watch.stats[it.key].gain for it in items[1:])
    assert summary["steer"]["background"]["lowered"] == 1 and summary["steer"]["background"]["cohort"] == 1.0
    assert summary["rows"][0]["image"] == "img0.png" and summary["rows"][0]["weights"].startswith("background x0.")
    # with the rule at 0 the same run steers nothing
    off, items2, rng2, state2 = _watch(tmp_path / "off", cap=0.0, ahead=0.0, boost=1.5)
    _run(off, items2, 20, factors, rng2, state2)
    assert not any(st.gain for st in off.stats.values())


def test_a_weight_at_zero_masks_the_region_and_can_come_back(tmp_path):
    watch, items, rng, state = _watch(tmp_path, ahead=0.0, min=0.0, strength=4.0, cap=0.2)
    over = lambda i, e: dict(face=0.6, hair=0.6, skin=0.6, background=0.3)  # noqa: E731
    _run(watch, items, 40, over, rng, state)
    assert watch.stats[items[0].key].gain["background"] < 0.02
    _run(watch, items, 60, lambda i, e: dict(face=0.6, hair=0.6, skin=0.6, background=1.0), rng, state)
    assert "background" not in watch.stats[items[0].key].gain


def test_boost_raises_a_lagging_subject_region_only_when_asked(tmp_path):
    lag = lambda i, e: dict(face=0.5, hair=0.9, skin=0.5)  # noqa: E731  (hair far behind the subject's 37%)
    watch, items, rng, state = _watch(tmp_path, boost=1.5)
    summary = _run(watch, items, 1, lag, rng, state)
    assert watch.class_gain == {"hair": pytest.approx(1.1)} and watch.steering  # up half as fast as down
    summary = _run(watch, items, 12, lag, rng, state)
    assert watch.class_gain == {"hair": pytest.approx(1.5)}  # never over the boost
    assert summary["steer"]["boost"] == {"hair": pytest.approx(1.5)} and "raised: hair x1.50" in RW.describe(summary)
    assert watch.gains([items[3].key]) == [{"hair": pytest.approx(1.5)}]  # in every picture
    _run(watch, items, 8, lambda i, e: dict(face=0.5, hair=0.5, skin=0.5), rng, state)
    assert watch.class_gain == {}
    never, items2, rng2, state2 = _watch(tmp_path / "never")  # boost 0: the default
    _run(never, items2, 8, lag, rng2, state2)
    assert never.class_gain == {} and not never.steering


def test_prior_follows_the_background_over_its_cap(tmp_path):
    watch, items, rng, state = _watch(tmp_path, ahead=0.0, prior=0.5)
    over = lambda i, e: dict(face=0.7, hair=0.7, skin=0.7, background=0.6)  # noqa: E731
    _run(watch, items, 1, over, rng, state)
    assert 0 < watch.prior_weight <= 0.05 + 1e-6  # up by step x prior / 2 an epoch
    summary = _run(watch, items, 20, over, rng, state)
    cohort = summary["steer"]["background"]["cohort"]
    assert watch.prior_weight == pytest.approx(0.5 * min(1.0, 2 * (1 - cohort)), abs=0.01)
    assert "background prior" in RW.describe(summary)
    _run(watch, items, 12, lambda i, e: dict(face=0.7, hair=0.7, skin=0.7, background=0.95), rng, state)
    assert watch.prior_weight == 0.0
    plain, items2, rng2, state2 = _watch(tmp_path / "plain", ahead=0.0)
    _run(plain, items2, 6, over, rng2, state2)
    assert plain.prior_weight == 0.0


def test_steering_survives_a_resume_and_stays_off_without_a_saved_state(tmp_path):
    watch, items, rng, state = _watch(tmp_path, ahead=0.0, boost=1.5, prior=0.5)
    _run(watch, items, 6, lambda i, e: dict(face=0.6, hair=0.9, skin=0.6, background=0.5), rng, state)
    assert watch.steering and watch.class_gain and watch.prior_weight > 0
    again = RW.RegionWatch(items, tmp_path, resume=True, steer=STEER)
    assert again.steering and again.steer_off == ""
    assert again.stats[items[0].key].gain == watch.stats[items[0].key].gain
    assert again.class_gain == watch.class_gain and again.prior_weight == watch.prior_weight
    assert again.learned_ema == pytest.approx(watch.learned_ema)
    observer = RW.RegionWatch(items, tmp_path, resume=True)  # the same state read without steering: nothing applied
    assert not observer.steering and observer.class_gain == {} and not observer.stats[items[0].key].gain
    lost = RW.RegionWatch(items, tmp_path / "elsewhere", resume=True, steer=STEER)
    assert lost.steer is None and "resume" in lost.steer_off and not lost.steering


def test_config_checks_the_region_settings():
    TrainConfig().validate_region_actions()
    cfg = TrainConfig(region_steer=True)
    cfg.validate_region_actions()
    assert cfg.wants_region_watch() and cfg.wants_part_maps() and not cfg.wants_parts()
    assert cfg.region_steer_settings() == {"regions": ("background", "clothes"), "cap": 0.6, "ahead": 0.7,
                                           "boost": 0.0, "prior": 0.0, "holdout": 0.0, "signal": "steps",
                                           "min": 0.3, "step": 0.2, "strength": 1.0}
    assert TrainConfig().region_steer_settings() is None
    assert TrainConfig(region_steer=True, region_steer_regions="clothes").region_steer_list() == ("clothes",)
    for bad in (dict(region_steer_regions="face"), dict(region_steer_regions=""), dict(region_steer_cap=3.0),
                dict(region_steer_ahead=1.0), dict(region_steer_boost=0.8), dict(region_steer_boost=4.0),
                dict(region_steer_min=1.0), dict(region_steer_step=0.0), dict(region_steer_strength=0.0),
                dict(region_steer_cap=0.0, region_steer_ahead=0.0),  # every rule off
                dict(region_steer_prior=0.5, region_steer_cap=0.0),  # the prior follows the cap rule
                dict(region_steer_prior=0.5, region_steer_regions="clothes"),
                dict(region_steer_prior=0.5, finetune=True),
                dict(optimizer="hull", hull_regions="parts")):
        with pytest.raises(ValueError):
            TrainConfig(region_steer=True, **bad).validate_region_actions()
    weights = TrainConfig(region_class_weights="face=3,background=0.3")
    weights.validate_region_actions()
    assert weights.wants_part_maps() and weights.wants_regions() and not weights.wants_region_watch()
    for bad in (dict(region_class_weights="nose=1"), dict(region_class_weights="face=3", region_weights=True),
                dict(region_class_weights="face=3", optimizer="hull", hull_regions="parts")):
        with pytest.raises(ValueError):
            TrainConfig(**bad).validate_region_actions()
    with pytest.raises(ValueError):  # the identity recipe has its own region-weighted loss
        TrainConfig.from_dict({**apply_preset({}, "identity"), "region_class_weights": "face=3"}).validate()


def test_trainer_hands_the_steering_settings_to_the_watch(tmp_path):
    from rulztrainer.trainer import Trainer

    items = _items(tmp_path, 3)
    t = Trainer.__new__(Trainer)
    t.job_dir, t.race_rows, t.region_watch_off = tmp_path, None, ""
    t.cfg = TrainConfig(region_steer=True, region_steer_boost=1.5, region_class_weights="clothes=0.5")
    watch = t._make_region_watch(items, 0)  # region_steer alone turns the watch on
    assert watch is not None and watch.steer["boost"] == 1.5 and watch.steer["regions"] == ("background", "clothes")
    t.cfg = TrainConfig(region_class_weights="clothes=0.5")
    assert t._make_region_watch(items, 0) is None  # fixed weights need no watch
    t.cfg = TrainConfig(region_steer=True)
    t.race_rows = [("a", t.cfg)]
    assert t._make_region_watch(items, 0) is None


def test_a_race_step_applies_the_fixed_class_weights():
    from types import SimpleNamespace

    from rulztrainer.dataset import RegionBatch
    from rulztrainer.race_parallel import forward_backward
    from rulztrainer.regions import RegionMap

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, x, t, embeds, mask):
            return x.float() * self.scale

    classes = torch.zeros(2, 4, dtype=torch.long)
    classes[:, 2:] = 3
    rm = RegionMap(torch.ones(1, 2, 4), torch.zeros(1, 2, 4, dtype=torch.bool), None, None, True,
                   _parts(2, 4, classes), "p")
    regions = RegionBatch(torch.ones(1, 1, 2, 4), torch.zeros(1, 1, 2, 4, dtype=torch.bool), [rm], ["t"])
    latents, noise = torch.zeros(1, 3, 2, 4), torch.zeros(1, 3, 2, 4)
    noise[..., 2:] = 1.0  # the whole error sits on the face half
    item = SimpleNamespace(key="k", loss_scale=1.0)

    def loss(spec):
        model = Model()
        ctx = SimpleNamespace(cfg=TrainConfig(region_class_weights=spec), loss_watch=None, look=None)
        job = ((model, {}, "cpu"), [item], latents, torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool),
               torch.tensor([0.5]), noise, regions, 1, 1, ctx)
        return forward_backward(job)[0]

    plain, weighted = loss(""), loss("face=3,background=1")
    assert weighted == pytest.approx(plain * 1.5, rel=1e-4)  # face cells at 3 / mean 2 = 1.5x, where all the error is


def test_sweep_signal_drives_the_ahead_rule_from_the_fixed_noise_readings(tmp_path):
    watch, items, rng, state = _watch(tmp_path, cap=0.0, signal="sweep")
    _run(watch, items, 6, lambda i, e: dict(background=0.3) if i == 0 else {}, rng, state)
    assert not watch.steering  # the steps say picture 0 runs ahead, but this run reads the sweeps: none yet
    watch.set_sweep({items[i].key: {"background": 0.4 if i == 3 else 1.0, "clothes": 1.0} for i in range(20)})
    summary = _run(watch, items, 3, lambda i, e: {}, rng, state)
    assert list(watch.stats[items[3].key].gain) == ["background"]
    assert not watch.stats[items[0].key].gain  # the steps' own reading is not what steers here
    assert summary["rows"][0]["sweep_background"] in (0.4, 1.0)  # the table shows both readings
    assert watch.contrasts("sweep")["background"] == {"median": 1.0, "pictures": 20}


def test_holdout_rule_turns_a_memorised_region_down_in_every_picture(tmp_path):
    watch, items, rng, state = _watch(tmp_path, cap=0.0, ahead=0.0, holdout=0.15, min=0.3)
    learned = lambda i, e: dict(face=0.5, hair=0.5, skin=0.5, background=0.7)  # noqa: E731
    watch.set_sweep({}, gap={"face": 0.35, "background": 0.05, "clothes": None}, flat={"face": 1, "background": 5})
    _run(watch, items, 3, learned, rng, state)
    assert watch.class_gain == {}  # the face's held-out reading is still improving (1 flat sweep): not yet
    watch.set_sweep({}, gap={"face": 0.35, "background": 0.05}, flat={"face": 2, "background": 5})
    summary = _run(watch, items, 1, learned, rng, state)
    assert watch.class_gain == {"face": pytest.approx(0.8)}  # the one rule that may turn the face down, a step a time
    summary = _run(watch, items, 12, learned, rng, state)
    target = (1 - (0.35 - 0.15) / (1 - 0.15)) ** 2
    assert watch.class_gain["face"] == pytest.approx(target, abs=0.01) and "background" not in watch.class_gain
    assert summary["steer"]["held"] == {"face": pytest.approx(target, abs=0.01)} and summary["steer"]["boost"] == {}
    assert "turned down for the held-out gap: face" in RW.describe(summary)
    assert watch.gains([items[5].key]) == [{"face": pytest.approx(target, abs=0.01)}]
    watch.set_sweep({}, gap={"face": 0.1}, flat={"face": 0})  # the gap closed
    _run(watch, items, 14, learned, rng, state)
    assert watch.class_gain == {} and not watch.steering


def test_config_checks_the_sweep_and_holdout_settings():
    ok = TrainConfig(region_steer=True, region_sweep_every=2, region_steer_signal="sweep", holdout_images=3,
                     region_steer_holdout=0.15, holdout_track_best=True, holdout_patience=4)
    ok.validate_region_actions()
    assert ok.wants_region_sweep() and ok.wants_part_maps() and ok.sweep_every_epochs() == 2
    assert ok.region_steer_settings()["signal"] == "sweep" and ok.region_steer_settings()["holdout"] == 0.15
    assert TrainConfig(holdout_images=2).sweep_every_epochs() == 1 and TrainConfig(holdout_images=2).wants_regions()
    assert not TrainConfig().wants_region_sweep()
    TrainConfig(region_steer=True, region_steer_cap=0, region_steer_ahead=0, region_steer_holdout=0.2,
                holdout_images=2).validate_region_actions()  # the holdout rule alone is a rule
    for bad in (dict(region_sweep_every=-1), dict(region_sweep_pictures=-2), dict(holdout_images=-1),
                dict(holdout_track_best=True), dict(holdout_patience=2), dict(region_sweep_pictures=-1),
                dict(holdout_images=2, resolution_step_up=True), dict(region_steer_signal="maps"),
                dict(region_steer=True, region_steer_signal="sweep"),  # no sweeps to read
                dict(region_steer=True, region_steer_signal="sweep", holdout_images=2, region_sweep_pictures=-1),
                dict(region_steer=True, region_steer_holdout=0.2),  # nothing held out
                dict(region_steer=True, region_steer_holdout=0.2, holdout_images=2, region_sweep_pictures=-1),
                dict(region_steer=True, region_steer_holdout=1.0, holdout_images=2)):
        with pytest.raises(ValueError):
            TrainConfig(**bad).validate_region_actions()
