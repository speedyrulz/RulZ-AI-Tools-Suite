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


# ---------------------------------------------------------------------------- lr_alpha_scale (3.47.0)
from rulztrainer.config import alpha_lr_factor, alpha_scale_reason, alpha_trained_text, trained_optimizer  # noqa: E402


def test_scale_is_off_by_default_and_changes_nothing():
    c = cfg(optimizer="adamw", learning_rate=1e-4, lora_rank=16, lora_alpha=1.0)
    assert c.lr_alpha_scale is False
    assert alpha_lr_factor(c) == 1.0
    assert trained_optimizer(c) == ("adamw", pytest.approx(1e-4), pytest.approx(0.01))
    assert lr_at_alpha_rank(c) == pytest.approx(1e-4 / 16)


@pytest.mark.parametrize("name", ["adamw", "adamw8bit", "adafactor", "muon", "normuon"])
def test_scale_trains_at_rank_over_alpha_with_weight_decay_divided(name):
    c = cfg(optimizer=name, learning_rate=1e-4, weight_decay=0.01, lora_rank=16, lora_alpha=1.0, lr_alpha_scale=True)
    assert alpha_lr_factor(c) == pytest.approx(16.0)
    got = trained_optimizer(c)
    assert got[0] == name
    assert got[1] == pytest.approx(1.6e-3) and got[2] == pytest.approx(6.25e-4)
    assert lr_at_alpha_rank(c) == pytest.approx(1e-4)  # the entered rate IS the alpha = rank rate
    assert c.learning_rate == 1e-4  # the entered value is never changed


def test_scale_also_lowers_the_rate_when_alpha_is_above_rank():
    c = cfg(optimizer="adamw", learning_rate=1e-4, weight_decay=0.01, lora_rank=16, lora_alpha=32.0, lr_alpha_scale=True)
    assert alpha_lr_factor(c) == pytest.approx(0.5)
    assert trained_optimizer(c)[1] == pytest.approx(5e-5) and trained_optimizer(c)[2] == pytest.approx(0.02)


@pytest.mark.parametrize("kw", [
    dict(optimizer="automagic3"), dict(optimizer="automagic4"), dict(optimizer="keel"), dict(optimizer="keel_fb"),
    dict(optimizer="keel3"), dict(optimizer="keel3_fb"), dict(optimizer="hull", hull_rho=2e-4),
    dict(finetune=True), dict(network_type="lokr"), dict(network_type="oft"), dict(lora_alpha=16.0),
])
def test_scale_does_nothing_where_it_does_not_apply(kw):
    c = cfg(**{"optimizer": "adamw", "learning_rate": 1e-4, "lora_rank": 16, "lora_alpha": 1.0, **kw,
               "lr_alpha_scale": True})
    assert alpha_lr_factor(c) == 1.0
    name, lr, wd = trained_optimizer(c)
    assert lr == pytest.approx(1e-4) and wd == pytest.approx(0.01)
    assert alpha_scale_reason(c)  # the WebUI says why it is greyed out


def test_scale_applies_after_the_isolation_swap_not_before():
    # Keel under isolation trains AdamW at 1e-4; x16 = 1.6e-3 is above effective_optimizer's 1e-3 limit but is kept
    c = cfg(optimizer="keel", isolation=True, dop_weight=1.0, learning_rate=1e-4, lora_rank=16, lora_alpha=1.0,
            lr_alpha_scale=True)
    name, lr, wd = trained_optimizer(c)
    assert name == "adamw" and lr == pytest.approx(1.6e-3) and wd == pytest.approx(6.25e-4)
    # an out-of-range pick falls back to 2e-4 first, then scales
    c2 = cfg(optimizer="keel", isolation=True, dop_weight=1.0, learning_rate=1e-6, lora_rank=16, lora_alpha=1.0,
             lr_alpha_scale=True)
    assert trained_optimizer(c2)[1] == pytest.approx(3.2e-3)
    assert lr_at_alpha_rank(c2) == pytest.approx(2e-4)


def test_scale_reason_texts():
    assert alpha_scale_reason(cfg(lora_rank=16, lora_alpha=1.0)) == ""
    assert "Alpha equals rank" in alpha_scale_reason(cfg(lora_rank=16, lora_alpha=16.0))
    assert "sets its own rate" in alpha_scale_reason(cfg(optimizer="keel", lora_rank=16, lora_alpha=1.0))
    assert "plain LoRA" in alpha_scale_reason(cfg(network_type="oft", lora_rank=16, lora_alpha=1.0))


def test_scaled_note_text():
    c = cfg(optimizer="adamw", learning_rate=1e-4, weight_decay=0.01, lora_rank=16, lora_alpha=1.0, lr_alpha_scale=True)
    text, warn = alpha_lr_note(c)
    assert text == ("LoRA alpha 1 at rank 16 scales each update by 1/16. Scale LR to alpha = rank is on: the learning "
                    "rate 1.0e-04 x 16/1 trains at 1.6e-03, with weight decay 0.01 / 16 = 6.25e-04, so it acts like "
                    "1.0e-04 at alpha = rank (RulZ AI Trainer's convention).")
    assert not warn
    # the 3x warning compares the entered rate (the alpha = rank equivalent) with the recommendation
    text, warn = alpha_lr_note(cfg(learning_rate=4e-4, lora_rank=16, lora_alpha=1.0, lr_alpha_scale=True),
                               recommended=1e-4)
    assert warn and "4.0x above the recommended 1.0e-04" in text


def test_note_when_on_but_not_applicable():
    text, warn = alpha_lr_note(cfg(optimizer="keel", lora_rank=16, lora_alpha=1.0, lr_alpha_scale=True))
    assert "sets its own rate" in text and "Scale LR to alpha = rank does nothing here" in text and not warn
    text, _ = alpha_lr_note(cfg(lora_rank=16, lora_alpha=16.0, lr_alpha_scale=True))
    assert text == "Scale LR to alpha = rank does nothing: alpha equals rank."
    assert alpha_lr_note(cfg(network_type="oft", lora_rank=16, lora_alpha=1.0, lr_alpha_scale=True)) is None
    # off: the old text, nothing added
    assert "Scale LR" not in alpha_lr_note(cfg(lora_rank=16, lora_alpha=1.0))[0]


def test_saved_dicts():
    base = {"network_type": "lora", "lora_rank": 16, "lora_alpha": 1.0, "learning_rate": 1e-4, "optimizer": "adamw"}
    assert alpha_equivalent_text(base) == "acts like lr 6.3e-06 at alpha = rank"  # no key: off
    assert alpha_trained_text(base) == ""
    on = {**base, "lr_alpha_scale": True}
    assert alpha_equivalent_text(on) == "acts like lr 1.0e-04 at alpha = rank"
    assert alpha_trained_text(on) == "trains at 1.6e-03 (scaled x16 to alpha = rank)"
    assert alpha_trained_text({**on, "optimizer": "keel"}) == ""
    assert runreport._learning_rate(on) == "0.0001; trains at 1.6e-03 (scaled x16 to alpha = rank)"
    assert runreport._learning_rate(base) == "0.0001"


def test_a_saved_config_without_the_key_loads_off(tmp_path):
    c = cfg(lora_rank=16, lora_alpha=1.0, lr_alpha_scale=True)
    path = tmp_path / "c.yaml"
    c.save(path)
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["lr_alpha_scale"] is True
    del data["lr_alpha_scale"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert TrainConfig.load(path).lr_alpha_scale is False


def _tiny_injected():
    import torch
    from rulztrainer import lora as lora_mod

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

    return lora_mod.inject_lora(Tiny(), 4, 1.0, [], names=["proj"])


def _build(c):
    from rulztrainer import trainer

    class Stub(trainer.Trainer):
        def __init__(self):
            self.race = None

    return Stub()._make_optimizer(c, _tiny_injected())


def test_trainer_builds_the_optimizer_at_the_scaled_rate_and_decay():
    c = cfg(optimizer="adamw", learning_rate=1e-4, weight_decay=0.01, lora_rank=4, lora_alpha=1.0, lr_alpha_scale=True,
            lora_plus_ratio=4.0)
    opt, sched = _build(c)
    by_name = {g["name"]: g for g in opt.param_groups}
    assert by_name["lora_down"]["initial_lr"] == pytest.approx(4e-4)
    assert by_name["lora_down"]["weight_decay"] == pytest.approx(0.0025)
    assert by_name["lora_up"]["initial_lr"] == pytest.approx(1.6e-3)  # LoRA+ multiplies the scaled rate
    assert by_name["lora_up"]["weight_decay"] == pytest.approx(0.0025)
    assert sched.base_lrs == pytest.approx([4e-4, 1.6e-3])
    assert c.learning_rate == 1e-4 and c.weight_decay == 0.01  # the config is not mutated


def test_trainer_off_builds_the_entered_rate():
    opt, _ = _build(cfg(optimizer="adamw", learning_rate=1e-4, weight_decay=0.01, lora_rank=4, lora_alpha=1.0))
    g, = opt.param_groups
    assert g["initial_lr"] == pytest.approx(1e-4) and g["weight_decay"] == pytest.approx(0.01)


def test_trainer_isolation_keel_pick_trains_adamw_at_the_scaled_rate():
    c = cfg(optimizer="keel", isolation=True, dop_weight=1.0, learning_rate=1e-4, weight_decay=0.01, lora_rank=4,
            lora_alpha=1.0, lr_alpha_scale=True)
    opt, _ = _build(c)
    g, = opt.param_groups
    assert g["initial_lr"] == pytest.approx(4e-4) and g["weight_decay"] == pytest.approx(0.0025)


def test_trainer_automagic_ignores_the_toggle():
    c = cfg(optimizer="automagic3", learning_rate=1e-4, lora_rank=4, lora_alpha=1.0, lr_alpha_scale=True)
    opt, _ = _build(c)
    assert all(g["initial_lr"] == pytest.approx(1e-4) for g in opt.param_groups)


def test_trainer_logs_the_scaled_note(caplog):
    from rulztrainer import trainer

    with caplog.at_level(logging.INFO, logger=trainer.log.name):
        _build(cfg(optimizer="adamw", learning_rate=1e-4, lora_rank=4, lora_alpha=1.0, lr_alpha_scale=True))
    assert any("trains at 4.0e-04" in r.getMessage() for r in caplog.records)


def _resume_with(saved_cfg, now_cfg):
    """Build with saved_cfg, save its optimizer and scheduler state, load that into one built with now_cfg and
    rescale, as Trainer's resume does."""
    from rulztrainer import trainer

    opt_saved, sched_saved = _build(saved_cfg)
    opt, sched = _build(now_cfg)
    opt.load_state_dict(opt_saved.state_dict())
    sched.load_state_dict(sched_saved.state_dict())

    class Stub(trainer.Trainer):
        def __init__(self):
            self.cfg = now_cfg

    saved_lr = opt.param_groups[0]["lr"]  # (warmup: the step's rate is a share of the base rate)
    Stub()._rescale_resumed_alpha_lr(opt, sched, saved_cfg.to_dict())
    return opt, sched, saved_lr


RESUME_BASE = dict(optimizer="adamw", learning_rate=1e-4, weight_decay=0.01, lora_rank=4, lora_alpha=1.0)


def test_resume_after_turning_the_toggle_on_moves_to_the_scaled_rate():
    opt, sched, saved_lr = _resume_with(cfg(**RESUME_BASE), cfg(**RESUME_BASE, lr_alpha_scale=True))
    g, = opt.param_groups
    assert g["lr"] == pytest.approx(saved_lr * 4) and g["initial_lr"] == pytest.approx(4e-4)
    assert g["weight_decay"] == pytest.approx(0.0025)
    assert sched.base_lrs == pytest.approx([4e-4])


def test_resume_after_turning_the_toggle_off_moves_back():
    opt, sched, saved_lr = _resume_with(cfg(**RESUME_BASE, lr_alpha_scale=True), cfg(**RESUME_BASE))
    g, = opt.param_groups
    assert g["lr"] == pytest.approx(saved_lr / 4) and g["weight_decay"] == pytest.approx(0.01)
    assert sched.base_lrs == pytest.approx([1e-4])


def test_resume_with_the_same_toggle_changes_nothing():
    c = cfg(**RESUME_BASE, lr_alpha_scale=True)
    opt, sched, saved_lr = _resume_with(c, c)
    g, = opt.param_groups
    assert g["lr"] == pytest.approx(saved_lr) and g["weight_decay"] == pytest.approx(0.0025)
    assert sched.base_lrs == pytest.approx([4e-4])
