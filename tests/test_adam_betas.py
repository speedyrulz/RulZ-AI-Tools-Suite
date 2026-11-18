"""adamw / adamw8bit take their betas and epsilon from the config (adam_beta1 / adam_beta2 / adam_eps; the defaults are
the values that used to be written into the trainer), other optimizers leave them alone."""
import logging
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import trainer as T  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


def params():
    return [torch.nn.Parameter(torch.zeros(4))]


def test_defaults_keep_the_old_betas():
    c = TrainConfig(optimizer="adamw")
    assert (c.adam_beta1, c.adam_beta2, c.adam_eps) == (0.9, 0.99, 1e-8)
    g = T.build_optimizer(c, params()).param_groups[0]
    assert g["betas"] == (0.9, 0.99) and g["eps"] == 1e-8


def test_set_betas_and_eps_reach_adamw():
    c = TrainConfig(optimizer="adamw", adam_beta1=0.8, adam_beta2=0.999, adam_eps=1e-6)
    c.validate()
    g = T.build_optimizer(c, params()).param_groups[0]
    assert g["betas"] == (0.8, 0.999) and g["eps"] == 1e-6


def test_adamw8bit_gets_them_too(monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, params, **kw):
            seen.update(kw)

    fake_bnb = types.SimpleNamespace(optim=types.SimpleNamespace(AdamW8bit=Fake))
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)
    T.build_optimizer(TrainConfig(optimizer="adamw8bit", adam_beta2=0.999, adam_eps=1e-7), params())
    assert seen["betas"] == (0.9, 0.999) and seen["eps"] == 1e-7


def test_other_optimizers_note_the_ignored_settings(caplog):
    with caplog.at_level(logging.INFO, logger="rulztrainer"):
        T.build_optimizer(TrainConfig(optimizer="adafactor", adam_beta2=0.999), params())
    assert any("adam_beta1 / adam_beta2 / adam_eps are ignored with adafactor" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="rulztrainer"):
        T.build_optimizer(TrainConfig(optimizer="adafactor"), params())
    assert not any("adam_beta" in r.message for r in caplog.records)


def test_validation():
    for bad, msg in (({"adam_beta1": 1.0}, "adam_beta1"), ({"adam_beta2": -0.1}, "adam_beta2"),
                     ({"adam_beta2": 1.5}, "adam_beta2"), ({"adam_eps": 0}, "adam_eps"),
                     ({"adam_eps": -1e-8}, "adam_eps")):
        with pytest.raises(ValueError, match=msg):
            TrainConfig(**bad).validate()
    TrainConfig(adam_beta1=0.0, adam_beta2=0.9999, adam_eps=1e-12).validate()
