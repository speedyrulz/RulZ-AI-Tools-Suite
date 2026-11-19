"""The alpha-aware learning-rate readout: what a LoRA's rate acts like at alpha = rank (RulZ AI Trainer's convention),
where it does not apply, the start note, the warning thresholds and the run report line. No training math changes."""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import runreport  # noqa: E402
from rulztrainer.config import (TrainConfig, alpha_equivalent_text, alpha_lr_note, alpha_lr_warning,  # noqa: E402
                                lr_at_alpha_rank)


def cfg(**kw) -> TrainConfig:
    return TrainConfig(**kw)


def test_onetrainer_preset_acts_like_one_sixteenth():
    assert lr_at_alpha_rank(cfg(learning_rate=3e-4, lora_rank=16, lora_alpha=1.0)) == pytest.approx(1.875e-5)


def test_alpha_equal_to_rank_keeps_the_rate():
    c = cfg(learning_rate=1e-4, lora_rank=32, lora_alpha=32.0)
    assert lr_at_alpha_rank(c) == pytest.approx(1e-4)
    assert alpha_lr_note(c) is None


def test_dora_counts_as_lora():
    assert lr_at_alpha_rank(cfg(learning_rate=1e-4, lora_rank=16, lora_alpha=8.0, dora=True)) == pytest.approx(5e-5)


def test_not_for_finetune_or_other_networks():
    assert lr_at_alpha_rank(cfg(finetune=True, lora_rank=16, lora_alpha=1.0)) is None
    assert alpha_lr_note(cfg(finetune=True, lora_rank=16, lora_alpha=1.0)) is None
    for kind in ("lokr", "oft"):
        c = cfg(network_type=kind, lora_rank=16, lora_alpha=1.0)
        assert lr_at_alpha_rank(c) is None
        assert alpha_lr_note(c) is None


@pytest.mark.parametrize("name", ["automagic3", "automagic4", "keel", "keel_fb", "keel3", "keel3_fb"])
def test_self_setting_optimizers_get_no_number(name):
    c = cfg(optimizer=name, lora_rank=16, lora_alpha=1.0, learning_rate=3e-4)
    assert lr_at_alpha_rank(c) is None
    text, warn = alpha_lr_note(c, recommended=1e-4)
    assert not warn
    assert "sets its own rate" in text and "alpha still scales its updates" in text
    assert "acts like" not in text


def test_isolation_swaps_automagic_for_adamw_so_the_number_applies():
    c = cfg(optimizer="keel", isolation=True, dop_weight=1.0, lora_rank=16, lora_alpha=8.0, learning_rate=1e-4)
    assert lr_at_alpha_rank(c) == pytest.approx(5e-5)


def test_start_note_text():
    text, warn = alpha_lr_note(cfg(learning_rate=3e-4, lora_rank=16, lora_alpha=1.0))
    assert text == ("LoRA alpha 1 at rank 16 scales each update by 1/16: learning rate 3.0e-04 acts like 1.9e-05 at "
                    "alpha = rank (RulZ AI Trainer's convention).")
    assert not warn


def test_warning_thresholds():
    assert alpha_lr_warning(1e-4, 1e-4) is None
    assert alpha_lr_warning(3e-4, 1e-4) is None  # exactly 3x is allowed
    assert alpha_lr_warning(3.1e-4, 1e-4) == "That is 3.1x above the recommended 1.0e-04."
    assert alpha_lr_warning(1.9e-5, 1e-4) == "That is 5.3x below the recommended 1.0e-04."
    assert alpha_lr_warning(1e-4 / 3.1, 1e-4) is not None
    assert alpha_lr_warning(1e-4 / 3, 1e-4) is None
    assert alpha_lr_warning(1e-4, None) is None
    assert alpha_lr_warning(1e-4, 0) is None


def test_note_warns_against_the_recommendation():
    c = cfg(learning_rate=3e-4, lora_rank=16, lora_alpha=1.0)
    text, warn = alpha_lr_note(c, recommended=1e-4)
    assert warn and "5.3x below the recommended 1.0e-04" in text
    text, warn = alpha_lr_note(c, recommended=None)
    assert not warn


def test_trainer_logs_the_note_once_per_optimizer(caplog):
    from rulztrainer import trainer

    class Stub(trainer.Trainer):
        def __init__(self):
            pass

    c = cfg(learning_rate=3e-4, lora_rank=16, lora_alpha=1.0, optimizer="adamw")
    t = Stub()
    t.arch = type("A", (), {})()
    with caplog.at_level(logging.INFO, logger=trainer.log.name):
        try:
            t._make_optimizer(c, {})
        except Exception:  # noqa: BLE001 - only the note before the optimizer is built matters here
            pass
    assert sum("acts like 1.9e-05 at alpha = rank" in r.getMessage() for r in caplog.records) == 1


def test_run_report_network_line():
    base = {"network_type": "lora", "lora_rank": 16, "lora_alpha": 1.0, "learning_rate": 3e-4, "optimizer": "adamw8bit"}
    assert runreport._network(base) == "LoRA, rank 16 (alpha 1; acts like lr 1.9e-05 at alpha = rank)"
    assert runreport._network({**base, "lora_alpha": 16.0}) == "LoRA, rank 16 (alpha 16)"
    assert runreport._network({**base, "optimizer": "keel"}) == "LoRA, rank 16 (alpha 1)"
    assert alpha_equivalent_text({**base, "network_type": "lokr"}) == ""
    assert alpha_equivalent_text({}) == ""


def test_isolation_uses_adamws_rate():
    # isolation trains a Keel pick with AdamW at 2e-4 when its own lr is not AdamW-sized: the equivalent follows it
    c = cfg(optimizer="keel", isolation=True, dop_weight=2.0, learning_rate=1e-6, lora_rank=16, lora_alpha=8.0)
    from rulztrainer.config import lr_at_alpha_rank

    assert lr_at_alpha_rank(c) == pytest.approx(1e-4)
    text, _ = alpha_lr_note(c)
    assert "AdamW's learning rate 2.0e-04 (isolation) acts like 1.0e-04" in text


def test_saved_config_without_dop_weight_uses_the_default():
    saved = {"optimizer": "keel", "isolation": True, "lora_rank": 16, "lora_alpha": 8, "learning_rate": 1e-6}
    assert alpha_equivalent_text(saved) == "acts like lr 1.0e-04 at alpha = rank"
    assert alpha_equivalent_text({**saved, "dop_weight": 0}) == ""  # no DOP: Keel sets its own rate
