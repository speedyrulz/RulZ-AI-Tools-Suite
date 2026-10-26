"""The wizard's Character / person focus (five of them: face only, face focus, face and body, body focus, body only),
the settings each sets over the identity recipe, and the recommendations, review and prompts that follow it; plus the
recommendations for a fine-tune, LoKr, block swap, the style goal and the step count."""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402
from rulztrainer.config import TrainConfig, apply_preset  # noqa: E402

FOCUSES = ["face_only", "face_focus", "face_body", "body_focus", "body_only"]


def _identity(tmp_path, **extra) -> TrainConfig:
    data = apply_preset({"datasets": [{"path": str(tmp_path)}], "face_refs": [str(tmp_path)], "trigger_word": "zxq",
                         **extra}, "identity")
    return TrainConfig.from_dict(data)


def _focused(cfg: TrainConfig, focus: str) -> TrainConfig:
    return TrainConfig.from_dict({**cfg.to_dict(), **webui.focus_settings(focus, cfg.trigger_word)})


def test_each_focus_sets_its_weights_and_is_read_back(tmp_path):
    cfg = _identity(tmp_path)
    assert webui.character_focus(cfg) == "face_focus"  # the recipe as it is
    expect = {"face_only": (1.0, 0.0, 3.0, 1.0), "face_focus": (0.7, 0.3, 3.0, 1.5), "face_body": (0.5, 0.5, 1.5, 1.5),
              "body_focus": (0.3, 0.7, 1.5, 3.0), "body_only": (0.0, 1.0, 1.0, 3.0)}
    for focus in FOCUSES:
        c = _focused(cfg, focus)
        c.validate()
        assert (c.face_weight, c.body_weight, c.face_loss_weight, c.body_loss_weight) == expect[focus], focus
        assert webui.character_focus(c) == focus, focus
    face, body = _focused(cfg, "face_only"), _focused(cfg, "body_only")
    assert face.probe_body_prompt == "" and face.body_refs == [] and face.crop_augment == "face"
    assert not face.likeness_curriculum and face.probe_face_prompt
    assert body.probe_face_prompt == "" and body.face_refs == [] and body.crop_augment == "none"
    assert not body.likeness_curriculum and "zxq" in body.probe_body_prompt
    for focus in ("face_focus", "face_body", "body_focus"):
        c = _focused(cfg, focus)
        assert c.crop_augment == "face+bust" and c.likeness_curriculum and c.probe_face_prompt and c.probe_body_prompt


def test_recommendations_follow_the_focus(tmp_path):
    cfg = _identity(tmp_path)
    rec = lambda focus: webui.wizard_recommendations(cfg, "character", [], None, focus)  # noqa: E731
    assert rec("face_only")["probe_body_prompt"][0] == "" and rec("face_only")["body_refs"][0] == ""
    assert rec("face_only")["crop_augment"][0] == "face" and "face_refs" in rec("face_only")
    assert rec("body_only")["probe_face_prompt"][0] == "" and rec("body_only")["face_refs"][0] == ""
    assert rec("body_only")["crop_augment"][0] == "none"
    both = rec("face_body")
    assert "zxq" in both["probe_body_prompt"][0] and both["crop_augment"][0] == "face+bust"
    assert both["body_refs"][0] == str(tmp_path)  # no body references yet: the dataset folders
    picked = TrainConfig.from_dict({**cfg.to_dict(), "body_refs": [str(tmp_path / "a.png")]})
    assert "body_refs" not in webui.wizard_recommendations(picked, "character", [], None, "face_body")
    assert "crop_augment" not in webui.wizard_recommendations(cfg, "style", [], None)
    assert webui._show(None, "") == "empty"


def test_review_and_prompts_follow_the_focus(tmp_path):
    cfg = _identity(tmp_path)
    issues = lambda c: [t for _, t in webui.review_issues(c, "character", None, 1)]  # noqa: E731
    assert any("no body references" in t for t in issues(cfg))  # face focus scores the body too
    assert not any("no body references" in t for t in issues(_focused(cfg, "face_only")))
    only_body = _focused(cfg, "body_only")
    assert any("no face references" in t for t in issues(TrainConfig.from_dict(
        {**_focused(cfg, "body_focus").to_dict(), "face_refs": [], "body_refs": [str(tmp_path)]})))
    assert not any("no face references" in t for t in issues(only_body))
    assert not any("full body" in p for p in webui.suggested_prompts("character", "zxq", "face_only"))
    assert any("full body" in p for p in webui.suggested_prompts("character", "zxq", "body_only"))


def test_a_fine_tune_lokr_and_automagic_change_the_training_recommendations(tmp_path):
    gpus = [{"total": 16000}, {"total": 16000}]

    def rec(**kw):
        c = TrainConfig.from_dict({**_identity(tmp_path).to_dict(), **kw})
        return webui.wizard_recommendations(c, "concept", gpus, None)

    base = rec()
    assert base["learning_rate"][0] == 1e-4 and base["optimizer"][0] == "adamw8bit" and "lora_rank" in base
    assert base["multi_gpu"][0] == "data_parallel" and "dp_balance" in base
    ft = rec(finetune=True)
    assert ft["learning_rate"][0] == 1e-5 and ft["optimizer"][0] == "adafactor"
    assert ft["multi_gpu"][0] == "off" and "dp_balance" not in ft and "block_swap" not in ft
    assert "lora_rank" not in ft and "lora_alpha" not in ft and "network_type" not in ft
    lokr = rec(network_type="lokr")
    assert lokr["lokr_factor"][0] == 8 and "lora_rank" not in lokr
    assert "learning_rate" not in rec(optimizer="automagic3")
    assert rec(block_swap=4)["multi_gpu"][0] == "off"


def test_adapter_timesteps_and_isolation_recommendations(tmp_path):
    cfg = _identity(tmp_path)
    char = webui.wizard_recommendations(cfg, "character", [], None)
    assert char["network_type"][0] == "lokr" and char["isolation"][0] is True and "timestep_sampling" not in char
    fibo = TrainConfig.from_dict({**cfg.to_dict(), "model_path": "Bria/FIBO-2.0-dev"})
    assert webui.wizard_recommendations(fibo, "character", [], None)["network_type"][0] == "lora"  # no LoKr loader
    no_trigger = TrainConfig.from_dict({**cfg.to_dict(), "trigger_word": ""})
    assert webui.wizard_recommendations(no_trigger, "character", [], None)["isolation"][0] is False
    style = webui.wizard_recommendations(cfg, "style", [], None)
    assert style["timestep_sampling"][0] == "logsnr" and style["network_type"][0] == "lora"
    assert webui.wizard_recommendations(cfg, "concept", [], None)["timestep_sampling"][0] == "logit_normal"
    assert "isolation" not in style and "look_warmup" not in style


def test_block_swap_is_recommended_only_for_a_card_the_model_does_not_fit():
    krea = TrainConfig(model_path="krea-2-raw", model_source="hf", transformer_quant="int8_convrot")
    assert krea.arch_name() == "krea2"
    swap, saved = webui.block_swap_plan(krea, 16.0, "int8_convrot")  # 13.5 GB + 4.5 headroom on 16 GB
    assert 1 <= swap <= 26 and 0 < saved < 13.5
    assert webui.block_swap_plan(krea, 24.0, "int8_convrot") == (0, 0.0)
    moss = TrainConfig()
    assert webui.block_swap_plan(moss, 16.0, "int8") == (0, 0.0)
    big = webui.block_swap_plan(moss, 8.0, "int8")[0]
    assert webui.block_swap_plan(moss, 6.0, "int8")[0] > big  # the smaller the card, the more blocks
    rec = webui.wizard_recommendations(krea, "concept", [{"total": 16000}], None)
    assert rec["block_swap"][0] >= 0
    assert webui.wizard_recommendations(moss, "concept", [{"total": 24000}], None)["block_swap"][0] == 0


def test_steps_count_repeats_and_crops_and_checkpoints_follow_the_steps(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    for i in range(10):
        Image.new("RGB", (64, 48)).save(a / f"{i}.png")
    cfg = TrainConfig(datasets=[{"path": str(a), "repeats": 2}], steps=2000)
    data = webui.datasets_summary([str(a)])
    assert data["counts"] == {str(a): 10} and data["outliers"] == 0 and data["scanned"] is False
    steps, reason = webui.wizard_recommendations(cfg, "concept", [], data)["steps"]
    assert steps == 2000 and "x repeats = 20" in reason  # 10 pictures x 2 repeats, 100 epochs
    crops = TrainConfig(datasets=[{"path": str(a), "repeats": 2}], crop_augment="face+bust")
    assert webui.wizard_recommendations(crops, "concept", [], data)["steps"][0] == 4000  # + about 1.2 crops each
    rec = webui.wizard_recommendations(cfg, "concept", [], data)
    assert rec["keep_checkpoints"][0] == 0  # 8 checkpoints: keep them all
    assert webui.wizard_recommendations(TrainConfig(steps=9000, save_every=250), "concept", [], None)["keep_checkpoints"][0] == 8


def test_review_rows_describe_the_adapter_memory_previews_and_options():
    rows = dict(webui.extra_review_rows(TrainConfig(network_type="lokr", lokr_factor=8, block_swap=6, isolation=True,
                                                    keep_checkpoints=4, timestep_sampling="logsnr")))
    assert rows["Adapter"] == "LoKr, factor 8" and "block swap 6 blocks" in rows["Memory"]
    assert "LoRA Isolation" in rows["Options"] and "newest 4 checkpoints kept" in rows["Options"]
    assert "logsnr timesteps" in rows["Options"] and "turbo LoRA" in rows["Previews"]
    ft = dict(webui.extra_review_rows(TrainConfig(finetune=True, finetune_extract_ranks=[64, 128])))
    assert ft["Adapter"] == "full fine-tune, extracting a LoRA of rank 64, 128"
    assert dict(webui.extra_review_rows(TrainConfig()))["Adapter"].startswith("LoRA, rank 16")
