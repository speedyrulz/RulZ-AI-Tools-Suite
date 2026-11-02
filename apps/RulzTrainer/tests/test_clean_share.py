"""clean_share (Fizgig's MOSS Slide): the notation, the share per epoch, the solved curve offset per sampling mode, and
the trainer's offset per epoch of a configuration's image queue."""
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import clean_share as CS  # noqa: E402
from rulztrainer.archs import get_arch  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.trainer import Trainer  # noqa: E402


def test_notation_and_shares_per_epoch():
    assert CS.parse("") is None and CS.parse(None) is None
    assert CS.parse("40") == (40.0, 0.0, 40.0, 0.0, 0)
    assert CS.parse("75:5:40") == (75.0, 5.0, 40.0, 5.0, 0)
    assert CS.parse(" 60% : 7 : 30 : 15 : 2 ") == (60.0, 7.0, 30.0, 15.0, 2)
    old = CS.parse("60:7:30:15:2")
    shares = [CS.share_at(old, e) for e in range(1, 22)]
    assert shares[:7] == [60.0] * 7  # until epoch 7
    assert shares[7:14] == [56.25, 52.5, 48.75, 45.0, 41.25, 37.5, 33.75]  # sliding, epochs 8-14
    assert shares[14] == 30.0  # epoch 15, the end of the slide
    assert shares[15:] == [60.0, 30.0, 60.0, 30.0, 60.0, 30.0]  # after it the higher share, every 2nd the lower
    slide = CS.parse(CS.SLIDE)  # 60:7:30:8:3
    shares = [CS.share_at(slide, e) for e in range(1, 15)]
    assert shares == [60.0] * 7 + [30.0, 60.0, 60.0, 30.0, 60.0, 60.0, 30.0]
    lrs = [CS.lr_scale_at(slide, e) for e in range(1, 15)]
    assert lrs == [1.0] * 10 + [0.5, 1.0, 1.0, 0.5]  # half the lr on the lower-share epochs after the slide only
    rising = CS.parse("30:2:60:3:2")  # the lower share first: still the higher after the slide, every 2nd the lower
    assert [CS.share_at(rising, e) for e in range(1, 8)] == [30.0, 30.0, 60.0, 60.0, 30.0, 60.0, 30.0]
    assert CS.lr_scale_at(rising, 5) == 0.5 and CS.lr_scale_at(rising, 4) == 1.0
    assert CS.lr_scale_at(CS.parse("40"), 9) == 1.0 and CS.lr_scale_at(None, 9) == 1.0
    assert CS.parse(CS.FT_SLIDE) == (11.0, 4.0, 30.0, 8.0, 2)
    split = CS.parse("75:5:40")
    assert [CS.share_at(split, e) for e in (1, 5, 6, 30)] == [75.0, 75.0, 40.0, 40.0]
    assert CS.share_at(CS.parse("40"), 1) == 40.0 and CS.share_at(None, 3) is None
    assert CS.epoch_of(0, 10) == 1 and CS.epoch_of(9, 10) == 1 and CS.epoch_of(10, 10) == 2
    for bad in ("abc", "60:7", "0", "100", "60:7:30:5", "60:7:30:15:2:1"):
        with pytest.raises(ValueError):
            CS.parse(bad)
    with pytest.raises(ValueError, match="clean_share"):
        TrainConfig(clean_share="60:7").validate()


@pytest.mark.parametrize("mode,extra", [("logit_normal", {}), ("reference", {}), ("uniform", {}),
                                        ("logit_normal", {"timestep_shift": 3.0}),
                                        ("logit_normal", {"sigma_min": 0.1, "sigma_max": 0.9})])
def test_the_offset_reaches_the_share(mode, extra):
    cfg = TrainConfig(timestep_sampling=mode, **extra)  # MOSS-Image by default: its reference curve
    arch = get_arch(cfg)
    for share in (60.0, 30.0, 11.0):
        off, got = CS.offset_for(cfg, share, 4096, arch)
        assert got == pytest.approx(share, abs=0.5)
    # the offset moves the curve noisier for a smaller share
    assert CS.offset_for(cfg, 30.0, 4096, arch)[0] > CS.offset_for(cfg, 60.0, 4096, arch)[0]


def test_reference_moss_curve_is_about_eleven_percent():
    cfg = TrainConfig(timestep_sampling="reference")
    assert CS.measured_share(cfg, 0.0, 4096, get_arch(cfg)) == pytest.approx(11.5, abs=1.5)


def test_out_of_reach_share_warns(caplog):
    cfg = TrainConfig(sigma_min=0.6, sigma_max=1.0, clean_share="40")  # nothing can land below 0.5
    sched = CS.CleanShare(cfg, get_arch(cfg), logging.getLogger("rulztrainer"))
    with caplog.at_level(logging.WARNING, logger="rulztrainer"):
        sched.offset(0, 10, 4096)
    assert "out of reach" in caplog.text


def test_trainer_offset_follows_the_epochs(caplog):
    tr = Trainer.__new__(Trainer)
    tr.arch = get_arch(TrainConfig())
    batcher = SimpleNamespace(items=[SimpleNamespace(repeats=1)] * 4 + [SimpleNamespace(repeats=2)] * 2)  # 8 per epoch
    cfg = TrainConfig(clean_share="60:1:30:3")  # 60% in epoch 1, 45% in epoch 2, 30% from epoch 3
    with caplog.at_level(logging.INFO, logger="rulztrainer"):
        offs = [tr._clean_offset(cfg, steps, 4096, batcher, 2) for steps in (0, 3, 4, 7, 8, 40)]
    arch = tr.arch
    want = [CS.offset_for(cfg, s, 4096, arch)[0] for s in (60.0, 60.0, 45.0, 45.0, 30.0, 30.0)]
    assert offs == pytest.approx(want, abs=1e-6)
    assert caplog.text.count("Clean-end share:") == 3  # logged once per change
    assert tr._clean_offset(TrainConfig(), 5, 4096, batcher, 2) == 0.0  # unset: the sampler's own curve


def test_lr_scale_and_fine_tune_rotations(caplog):
    tr = Trainer.__new__(Trainer)
    tr.arch = get_arch(TrainConfig())
    batcher = SimpleNamespace(items=[SimpleNamespace(repeats=1)] * 8)  # 8 images per epoch, 4 steps of 2
    cfg = TrainConfig(clean_share="60:1:30:2:2")  # 60, 30, then 60 with every 2nd epoch at 30 at half the lr
    scales = [tr._clean_lr_scale(cfg, steps, batcher, 2) for steps in (0, 4, 8, 12, 16)]
    assert scales == [1.0, 1.0, 1.0, 0.5, 1.0]  # epochs 1-5
    with caplog.at_level(logging.INFO, logger="rulztrainer"):
        tr._clean_offset(cfg, 12, 4096, batcher, 2)
    assert "at half the learning rate" in caplog.text
    # a full fine-tune of 3 windows, each 4 steps: a rotation is 12 steps = 24 images, whatever the epoch holds
    ft = Trainer.__new__(Trainer)
    ft.arch, ft.ft, ft.ft_rotate_steps = tr.arch, SimpleNamespace(windows=[["a"], ["b"], ["c"]]), 4
    cfg = TrainConfig(clean_share="60:1:30:2:2")
    assert [ft._clean_lr_scale(cfg, s, batcher, 2) for s in (0, 11, 12, 35, 36, 47)] == [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
    with caplog.at_level(logging.INFO, logger="rulztrainer"):
        ft._clean_offset(cfg, 12, 4096, batcher, 2)
    assert "from rotation 2" in caplog.text
