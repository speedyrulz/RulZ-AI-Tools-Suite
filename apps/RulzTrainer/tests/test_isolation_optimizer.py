import dataclasses

import pytest
import torch

from rulztrainer.config import AUTOMAGIC, TrainConfig, effective_optimizer
from rulztrainer.race import variant_configs
from rulztrainer.trainer import build_optimizer, build_scheduler


@pytest.mark.parametrize("name", AUTOMAGIC)
@pytest.mark.parametrize("rate,expected", [(1e-6, 2e-4), (1e-5, 1e-5), (1e-3, 1e-3), (2e-3, 2e-4)])
def test_isolation_switch(name, rate, expected):
    cfg = TrainConfig(isolation=True, optimizer=name, learning_rate=rate, warmup_steps=10)
    assert effective_optimizer(cfg)[:2] == ("adamw", expected)
    p = torch.nn.Parameter(torch.ones(1))
    opt = build_optimizer(cfg, [p])
    assert type(opt) is torch.optim.AdamW
    scheduler = build_scheduler(cfg, opt)
    assert scheduler.lr_lambdas[0](0) != scheduler.lr_lambdas[0](10)
    p.grad = torch.ones_like(p)
    opt.step()
    state = opt.state_dict()
    other = build_optimizer(cfg, [torch.nn.Parameter(torch.ones(1))])
    other.load_state_dict(state)
    assert other.state_dict()["state"]
    assert effective_optimizer(dataclasses.replace(cfg, isolation=False))[0] == name


def test_race_rows_use_own_optimizer():
    cfg = TrainConfig(isolation=True, optimizer="adamw", race=True,
                      race_configs=[{"name": "a", "optimizer": "keel3"}, {"name": "b", "optimizer": "adamw"}])
    rows = variant_configs(cfg)
    assert effective_optimizer(rows[0][1])[2]
    assert effective_optimizer(rows[1][1])[2] is None


def _saved(tmp_path, opt, scheduler, config, effective=None):
    import random

    st = {"optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(), "config": config, "step": 7,
          "torch_gen": torch.Generator().get_state(), "py_rng": random.Random(1).getstate(),
          "torch_rng": torch.get_rng_state()}
    if effective:
        st["effective_optimizer"] = effective
    torch.save(st, tmp_path / "train_state.pt")


def _resume(tmp_path, monkeypatch, cfg, params):
    import random

    from rulztrainer import trainer as T

    monkeypatch.setattr(T.lora_mod, "load_lora_weights", lambda inj, path: len(inj))
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = cfg
    opt = build_optimizer(cfg, params)
    sched = build_scheduler(cfg, opt)
    return tr.load_state(tmp_path, [], opt, sched, torch.Generator(), random.Random()), opt


def test_resume_an_automagic_state_under_isolation_starts_adamw_fresh(tmp_path, monkeypatch, caplog):
    old_cfg = TrainConfig(optimizer="automagic4")
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    old = build_optimizer(old_cfg, p)
    _saved(tmp_path, old, build_scheduler(old_cfg, old), {"optimizer": "automagic4"})  # saved before the switch
    cfg = TrainConfig(optimizer="automagic4", isolation=True, learning_rate=1e-4)
    with caplog.at_level("WARNING", logger="rulztrainer"):
        _, opt = _resume(tmp_path, monkeypatch, cfg, [torch.nn.Parameter(torch.randn(4, 4))])
    assert type(opt) is torch.optim.AdamW and "start fresh" in caplog.text


def test_resume_an_isolation_adamw_state(tmp_path, monkeypatch):
    cfg = TrainConfig(optimizer="keel", isolation=True, learning_rate=1e-4)
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    opt = build_optimizer(cfg, p)
    p[0].grad = torch.ones_like(p[0])
    opt.step()
    _saved(tmp_path, opt, build_scheduler(cfg, opt), {"optimizer": "keel"}, effective="adamw")
    _, again = _resume(tmp_path, monkeypatch, cfg, [torch.nn.Parameter(torch.randn(4, 4))])
    assert again.state_dict()["state"]  # AdamW's moments carried over
    off = dataclasses.replace(cfg, isolation=False)  # the same state without isolation: Keel cannot take AdamW's
    with pytest.raises(ValueError, match="optimizer adamw"):
        _resume(tmp_path, monkeypatch, off, [torch.nn.Parameter(torch.randn(4, 4))])


def test_no_switch_without_the_pull_back():
    cfg = TrainConfig(isolation=True, optimizer="keel", dop_weight=0.0)
    assert effective_optimizer(cfg) == ("keel", cfg.learning_rate, None)


def test_old_state_resumes_on_the_schedule_at_its_step(tmp_path, monkeypatch):
    old_cfg = TrainConfig(optimizer="automagic4")
    old = build_optimizer(old_cfg, [torch.nn.Parameter(torch.randn(4, 4))])
    _saved(tmp_path, old, build_scheduler(old_cfg, old), {"optimizer": "automagic4"})  # step 7
    cfg = TrainConfig(optimizer="automagic4", isolation=True, learning_rate=1e-4, warmup_steps=20, steps=100)
    _, opt = _resume(tmp_path, monkeypatch, cfg, [torch.nn.Parameter(torch.randn(4, 4))])
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4 * 8 / 21)  # warm-up at step 7, not step 0
