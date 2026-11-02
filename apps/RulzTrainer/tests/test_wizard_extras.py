"""The wizard's identity setup: its basics (rank 32, adamw8bit at 1e-4), the experimental extras on its last page
(each a real setting, off at its default, with what it does, what testing found and its cost), and the Train tab's
Advanced switch lists."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402
from rulztrainer.config import TrainConfig, apply_preset  # noqa: E402


def test_basics_over_the_identity_recipe():
    cfg = TrainConfig.from_dict({**apply_preset({}, "identity"), **webui.WIZARD_BASICS})
    cfg.validate()
    assert (cfg.lora_rank, cfg.lora_alpha, cfg.optimizer, cfg.learning_rate) == (32, 32.0, "adamw8bit", 1e-4)
    assert webui.wizard_recommendations(cfg, "character", [], None)["lora_rank"][0] == 32
    assert set(webui.WIZARD_FOCUS) <= set(webui.CHARACTER_FOCUS)


def test_experiments_are_real_settings_off_by_default():
    d = TrainConfig()
    for ex in webui.EXPERIMENTS:
        assert ex["group"] in webui.EXPERIMENT_GROUPS and ex["evidence"] in webui.EVIDENCE, ex["key"]
        assert all(ex[k].strip() for k in ("short", "what", "tested", "cost")), ex["key"]
        assert ex["control"] in ("switch", "select", "number", "text")
        if ex["key"] != "optimizer":  # the wizard's basics pick adamw8bit over the config's adamw
            assert getattr(d, ex["key"]) == ex["off"], ex["key"]
        if ex["control"] == "select":
            assert ex["off"] in ex["options"]
    # switching every extra back off leaves a valid identity run
    on = {ex["key"]: ex["off"] for ex in webui.EXPERIMENTS}
    TrainConfig.from_dict({**apply_preset({}, "identity"), **webui.WIZARD_BASICS, **on}).validate()


def test_train_tab_advanced_lists():
    keys = {ex["key"] for ex in webui.EXPERIMENTS}
    assert set(webui.TRAIN_EXP_KEYS) <= keys
    assert {k for k in webui.TRAIN_EXP_SECTIONS.values() if k} <= keys
