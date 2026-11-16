"""3.24.0 defaults from the community training logs: the self-tuning optimizers' lr floor, its collapse warning,
the automagic4 warning, and MiniMax H3 still-only runs training blocks 20-49."""
import logging
from types import SimpleNamespace

import torch

from rulztrainer.config import TrainConfig
from rulztrainer.trainer import LR_FLOOR_WARN_STEPS, Trainer, automagic_floor, build_optimizer, lr_at_floor


def test_floor_default_is_2e5():
    assert TrainConfig().automagic_min_lr == 2e-5


def test_floor_kept_below_start_lr():
    assert automagic_floor(TrainConfig(learning_rate=1e-4)) == 2e-5
    assert automagic_floor(TrainConfig(learning_rate=1e-4, automagic_min_lr=3e-5)) == 3e-5


def test_floor_at_or_above_start_lr_drops_to_a_fifth():
    assert abs(automagic_floor(TrainConfig(learning_rate=1e-5)) - 2e-6) < 1e-12
    assert abs(automagic_floor(TrainConfig(learning_rate=2e-5)) - 4e-6) < 1e-12


def _params():
    return [torch.nn.Parameter(torch.randn(8, 8))]


def test_floor_reaches_every_self_tuning_optimizer():
    for name in ("automagic3", "automagic4", "keel"):
        cfg = TrainConfig(optimizer=name, learning_rate=1e-4)
        opt = build_optimizer(cfg, _params())
        assert opt.param_groups[0]["min_lr"] == 2e-5, name


def test_automagic4_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="rulztrainer"):
        build_optimizer(TrainConfig(optimizer="automagic4"), _params())
    assert any("automagic4 is unstable" in r.getMessage() for r in caplog.records)


def test_lr_at_floor():
    opt = SimpleNamespace(param_groups=[{"min_lr": 2e-5}])
    assert lr_at_floor(opt, 2e-5) and lr_at_floor(opt, 2.09e-5)
    assert not lr_at_floor(opt, 3e-5)
    assert not lr_at_floor(SimpleNamespace(param_groups=[{}]), 1e-9)  # no floor: AdamW and co.
    assert LR_FLOOR_WARN_STEPS == 100


def _shell(arch="minimax_h3"):
    tr = Trainer.__new__(Trainer)
    tr.arch, tr.race_rows = SimpleNamespace(name=arch), None
    return tr


def test_h3_stills_only_train_blocks_20_49():
    cfg = TrainConfig()
    _shell()._h3_still_blocks(cfg, [])
    assert cfg.lora_blocks == "20-49"


def test_h3_blocks_left_alone():
    for kw, clips, arch in ((dict(), [object()], "minimax_h3"),  # clips: the video run trains every block
                            (dict(lora_blocks="all"), [], "minimax_h3"),  # an explicit choice
                            (dict(lora_blocks="10-30"), [], "minimax_h3"),
                            (dict(resume_from="x.safetensors"), [], "minimax_h3"),  # the LoRA's layout is set
                            (dict(resume_state="state_000100"), [], "minimax_h3"),
                            (dict(), [], "minimax_h3_image"),  # the image model has no motion to keep
                            (dict(), [], "krea2")):
        cfg = TrainConfig(**kw)
        before = cfg.lora_blocks
        _shell(arch)._h3_still_blocks(cfg, clips)
        assert cfg.lora_blocks == before, (kw, arch)
