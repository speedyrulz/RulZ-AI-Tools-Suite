"""Learning-rate schedules: warm-up (steps, or a share of the steps), the floor (lr_min_factor), REX and cosine with
restarts, Automagic ignoring the schedule, and the settings' validation."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import trainer as T  # noqa: E402
from rulztrainer.config import LR_SCHEDULERS, TrainConfig  # noqa: E402


def cfg(**kw) -> TrainConfig:
    return TrainConfig(**{"steps": 1000, "warmup_steps": 0, **kw})


def test_old_schedules_are_unchanged():
    c = cfg(lr_scheduler="cosine", warmup_steps=50)
    assert T.lr_factor(c, 0) == pytest.approx(1 / 51)
    assert T.lr_factor(c, 49) == pytest.approx(50 / 51)
    assert T.lr_factor(c, 50) == pytest.approx(1.0)
    assert T.lr_factor(c, 1000) == pytest.approx(0.0, abs=1e-12)
    assert T.lr_factor(c, 525) == pytest.approx(0.5)  # halfway through the 950 steps after the warm-up
    lin = cfg(lr_scheduler="linear", warmup_steps=0)
    assert T.lr_factor(lin, 250) == pytest.approx(0.75) and T.lr_factor(lin, 1000) == 0.0
    const = cfg(lr_scheduler="constant", warmup_steps=10)
    assert T.lr_factor(const, 500) == 1.0 and T.lr_factor(const, 4) == pytest.approx(5 / 11)


def test_floor_lifts_the_end_of_every_decaying_schedule_but_not_constant():
    for kind in ("cosine", "linear", "rex", "cosine_restarts"):
        c = cfg(lr_scheduler=kind, lr_min_factor=0.1)
        assert T.lr_factor(c, 0) == pytest.approx(1.0)
        assert T.lr_factor(c, 1000) == pytest.approx(0.1)
        assert T.lr_factor(c, 5000) == pytest.approx(0.1)  # past the end stays on the floor
    assert T.lr_factor(cfg(lr_scheduler="constant", lr_min_factor=0.5), 999) == 1.0
    assert T.lr_factor(cfg(lr_scheduler="linear", lr_min_factor=0.2), 500) == pytest.approx(0.2 + 0.8 * 0.5)


def test_rex_formula_and_endpoints():
    c = cfg(lr_scheduler="rex")
    assert T.lr_factor(c, 0) == pytest.approx(1.0)
    assert T.lr_factor(c, 1000) == pytest.approx(0.0)
    for step in (100, 250, 500, 900):
        z = step / 1000
        assert T.lr_factor(c, step) == pytest.approx((1 - z) / (0.5 + 0.5 * (1 - z)))
    assert T.lr_factor(c, 500) == pytest.approx(0.5 / 0.75)
    assert T.lr_factor(c, 500) > T.lr_factor(cfg(lr_scheduler="linear"), 500)  # stays higher than linear until late


def test_rex_runs_after_the_warmup():
    c = cfg(lr_scheduler="rex", warmup_steps=100)
    assert T.lr_factor(c, 100) == pytest.approx(1.0) and T.lr_factor(c, 1000) == pytest.approx(0.0)
    z = (550 - 100) / 900
    assert T.lr_factor(c, 550) == pytest.approx((1 - z) / (0.5 + 0.5 * (1 - z)))


def test_cosine_restarts_jump_back_to_the_peak_at_each_cycle_start():
    c = cfg(lr_scheduler="cosine_restarts", lr_cycles=4)  # cycles of 250 steps
    for start in (0, 250, 500, 750):
        assert T.lr_factor(c, start) == pytest.approx(1.0)
        assert T.lr_factor(c, start + 125) == pytest.approx(0.5)
        assert T.lr_factor(c, start + 249) < 0.001
    assert T.lr_factor(c, 1000) == pytest.approx(0.0, abs=1e-12)
    one = cfg(lr_scheduler="cosine_restarts", lr_cycles=1)  # a single cycle is the plain cosine
    plain = cfg(lr_scheduler="cosine")
    for step in (0, 100, 500, 999, 1000):
        assert T.lr_factor(one, step) == pytest.approx(T.lr_factor(plain, step))


def test_cosine_restarts_cycle_after_the_warmup_and_floor():
    c = cfg(lr_scheduler="cosine_restarts", lr_cycles=2, warmup_steps=200, lr_min_factor=0.25)  # 400-step cycles
    assert T.lr_factor(c, 200) == pytest.approx(1.0) and T.lr_factor(c, 600) == pytest.approx(1.0)
    assert T.lr_factor(c, 400) == pytest.approx(0.25 + 0.75 * 0.5)
    assert T.lr_factor(c, 1000) == pytest.approx(0.25)


def test_warmup_fraction_overrides_warmup_steps_and_small_steps_are_never_shares():
    c = cfg(steps=2000, warmup_steps=7, warmup_fraction=0.05, lr_scheduler="constant")
    assert T.warmup_length(c) == 100
    assert T.lr_factor(c, 99) == pytest.approx(100 / 101) and T.lr_factor(c, 100) == 1.0
    assert T.warmup_length(cfg(steps=999, warmup_fraction=0.1)) == round(99.9)
    one = cfg(steps=2000, warmup_steps=1, warmup_fraction=0.0)  # a 1 is one step, not 100% (OneTrainer's pitfall)
    assert T.warmup_length(one) == 1 and T.lr_factor(one, 5) == 1.0
    assert T.warmup_length(cfg(warmup_steps=50)) == 50 and T.warmup_length(cfg(warmup_steps=-3)) == 0


def test_scheduler_follows_the_factor_with_an_offset_and_automagic_ignores_it():
    c = cfg(lr_scheduler="cosine_restarts", lr_cycles=3, warmup_steps=20, lr_min_factor=0.1)
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=2.0)
    sched = T.build_scheduler(c, opt)
    for step in range(30):
        assert opt.param_groups[0]["lr"] == pytest.approx(2.0 * T.lr_factor(c, step))
        opt.step()
        sched.step()
    opt2 = torch.optim.SGD([p], lr=2.0)
    T.build_scheduler(c, opt2, offset=500)  # a fine-tune window that starts at step 500
    assert opt2.param_groups[0]["lr"] == pytest.approx(2.0 * T.lr_factor(c, 500))
    auto = cfg(lr_scheduler="rex", optimizer="automagic3", warmup_steps=20, lr_min_factor=0.5)
    opt3 = torch.optim.SGD([p], lr=2.0)
    sched3 = T.build_scheduler(auto, opt3)
    for _ in range(50):
        opt3.step()
        sched3.step()
    assert opt3.param_groups[0]["lr"] == 2.0


def test_validation_and_old_configs():
    assert {"rex", "cosine_restarts"} <= set(LR_SCHEDULERS)
    d = TrainConfig()
    assert (d.lr_min_factor, d.lr_cycles, d.warmup_fraction, d.lr_scheduler) == (0.0, 1, 0.0, "constant")
    for bad, msg in (({"warmup_fraction": 0.6}, "warmup_fraction"), ({"warmup_fraction": -0.1}, "warmup_fraction"),
                     ({"lr_min_factor": 1.5}, "lr_min_factor"), ({"lr_min_factor": -0.01}, "lr_min_factor"),
                     ({"lr_cycles": 0}, "lr_cycles"), ({"lr_scheduler": "polynomial"}, "lr_scheduler")):
        with pytest.raises(ValueError, match=msg):
            TrainConfig(**bad).validate()
    TrainConfig(lr_scheduler="rex", lr_min_factor=1.0, warmup_fraction=0.5, lr_cycles=5).validate()


def test_last_update_lies_between_the_floor_and_the_step_before():
    """The last optimizer update uses the factor of step steps - 1: just above the floor, never below it (the same
    boundary the plain cosine and linear schedules always had)."""
    for kind in ("cosine", "linear", "rex", "cosine_restarts"):
        c = cfg(steps=10, lr_scheduler=kind, lr_min_factor=0.2)
        last, before = T.lr_factor(c, 9), T.lr_factor(c, 8)
        assert 0.2 <= last <= before, kind
    assert T.lr_factor(cfg(steps=10, lr_scheduler="linear", lr_min_factor=0.2), 9) == pytest.approx(0.28)
