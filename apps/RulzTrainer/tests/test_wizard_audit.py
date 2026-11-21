"""The wizard audit's logic fixes: multi-GPU picks, the turbo LoRA the probes see, the steps reason's epoch count, the
review's text and Save as path, the model-card fit badge and the start checks the wizard's own picks must not trip."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402
from rulztrainer.config import MODEL_PRESETS, TrainConfig  # noqa: E402

KREA = MODEL_PRESETS["Krea-2-Raw"]
ZIMAGE = MODEL_PRESETS["Z-Image"]
KLEIN4 = MODEL_PRESETS["FLUX-2-Klein-4B-Base"]
KLEIN9 = MODEL_PRESETS["FLUX-2-Klein-9B-Base"]
MOSS = MODEL_PRESETS["MOSS-Image-Base"]


def gpu(gb: float) -> dict:
    return {"total": gb * 1024}


def rec(model, goal="character", gpus=(), data=None, **kw):
    return webui.wizard_recommendations(TrainConfig(model_path=model, **kw), goal, list(gpus), data)


# ---- W1: the one-GPU pick no longer overwrites the multi-GPU ones

def test_two_gpus_keep_the_krea_model_split():
    r = rec(KREA, gpus=[gpu(16), gpu(16)])
    assert r["multi_gpu"][0] == "model_split" and "2 GPUs" in r["multi_gpu"][1]
    assert "dp_balance" not in r


def test_two_gpus_and_a_fine_tune_say_why_data_parallel_is_off():
    r = rec(MOSS, gpus=[gpu(24), gpu(24)], finetune=True)
    assert r["multi_gpu"][0] == "off" and "fine-tune" in r["multi_gpu"][1]
    assert "dp_balance" not in r


def test_two_gpus_with_block_swap_say_block_swap_runs_on_one():
    r = rec(MOSS, gpus=[gpu(24), gpu(24)], block_swap=4)
    assert r["multi_gpu"] == ("off", "block swap runs on one GPU")


def test_two_gpus_otherwise_data_parallel_with_balance():
    r = rec(MOSS, gpus=[gpu(24), gpu(24)])
    assert r["multi_gpu"][0] == "data_parallel" and r["dp_balance"][0] == "auto"


def test_one_gpu_says_so_and_none_says_nothing():
    assert rec(MOSS, gpus=[gpu(24)])["multi_gpu"] == ("off", "one GPU found")
    assert "multi_gpu" not in rec(MOSS, gpus=[])


# ---- W3: "auto" counts as a turbo LoRA only for a family that has one

@pytest.mark.parametrize("model", [ZIMAGE, KLEIN4])
def test_probes_keep_the_preview_cfg_where_auto_attaches_no_turbo_lora(model):
    cfg = TrainConfig(model_path=model)
    assert cfg.turbo_lora == "auto" and not webui.turbo_lora_applies(cfg)
    r = webui.wizard_recommendations(cfg, "character", [], None)
    assert r["probe_cfg"][0] == r["sample_cfg"][0] != 1.0
    assert r["probe_steps"][0] == r["sample_steps"][0]
    assert "turbo" not in r["probe_steps"][1].split("without a turbo LoRA")[0]


@pytest.mark.parametrize("model", [MOSS, KREA, KLEIN9])
def test_a_family_with_a_turbo_lora_still_probes_at_cfg_1(model):
    cfg = TrainConfig(model_path=model)
    assert webui.turbo_lora_applies(cfg)
    r = webui.wizard_recommendations(cfg, "character", [], None)
    assert r["probe_cfg"][0] == 1.0 and r["probe_steps"][0] == r["sample_steps"][0]
    assert "turbo LoRA" in r["probe_steps"][1]


def test_a_file_counts_everywhere_and_none_never():
    assert webui.turbo_lora_applies(TrainConfig(model_path=ZIMAGE, turbo_lora="x.safetensors"))
    assert not webui.turbo_lora_applies(TrainConfig(model_path=MOSS, turbo_lora=""))
    r = webui.wizard_recommendations(TrainConfig(model_path=ZIMAGE, turbo_lora="x.safetensors"), "character", [], None)
    assert r["probe_cfg"][0] == 1.0  # the user's own LoRA, as before


def test_the_unfinished_probe_check_follows_the_real_turbo_lora():
    def issues(model, **kw):
        cfg = TrainConfig(model_path=model, probe_steps=8, probe_every=25, face_refs=["a"], body_refs=["b"], **kw)
        return [t for lvl, t in webui.review_issues(cfg, "character", None, 1) if "unfinished" in t]

    assert issues(ZIMAGE)  # "auto" attaches nothing on Z-Image: 8-step probes come out unfinished
    assert not issues(MOSS)  # MOSS-Image's turbo LoRA makes 8 steps right
    assert not issues(ZIMAGE, turbo_lora="x.safetensors")


def test_the_unfinished_probe_check_runs_after_the_reference_notices():
    cfg = TrainConfig(model_path=ZIMAGE, probe_steps=8, probe_every=25)  # no references at all
    texts = [t for _, t in webui.review_issues(cfg, "character", None, 1)]
    assert any("No face or body references" in t for t in texts) and any("unfinished" in t for t in texts)


# ---- W5: the steps reason names the epochs the steps really are

def test_clamped_steps_state_the_real_epochs(tmp_path):
    cfg = TrainConfig(model_path=MOSS, datasets=[{"path": str(tmp_path)}])
    counts = {str(tmp_path): 100}
    r = webui.wizard_recommendations(cfg, "style", [], {"images": 100, "counts": counts})
    steps, why = r["steps"]
    assert steps == 4000 and "capped at 4000" in why and "about 40 epochs" in why
    counts = {str(tmp_path): 5}
    r = webui.wizard_recommendations(cfg, "style", [], {"images": 5, "counts": counts})
    steps, why = r["steps"]
    assert steps == 1000 and "at least 1000" in why and "about 200 epochs" in why
    counts = {str(tmp_path): 20}
    steps, why = webui.wizard_recommendations(cfg, "style", [], {"images": 20, "counts": counts})["steps"]
    assert steps == 2000 and "about 100 epochs" in why and "capped" not in why and "at least" not in why


def test_epochs_reason_directly():
    assert webui.epochs_reason(2000, 20, 100) == "about 100 epochs"
    assert "capped at 4000 steps" in webui.epochs_reason(4000, 100, 100)
    assert "about 40 epochs" in webui.epochs_reason(4000, 100, 100)
    assert "at least 1000 steps" in webui.epochs_reason(1000, 5, 100)
    # two data-parallel GPUs train two photos per step: 4000 steps over 100 photos is 80 epochs, as the trainer counts
    assert webui.epochs_reason(4000, 100, 100, 2).startswith("about 80 epochs at 2 photos per step")
    assert "would be" not in webui.epochs_reason(4000, 100, 100)


# ---- W11, W13, W2: review and photos-page text

def test_uncaptioned_note_names_what_the_photos_train_on():
    assert "trigger word alone" in webui.uncaptioned_text(3, "zxq")
    assert "an empty prompt" in webui.uncaptioned_text(3, "") and "trigger word alone" not in webui.uncaptioned_text(3, "")


def test_likeness_row_names_the_references_not_a_count():
    cfg = TrainConfig(model_path=MOSS, probe_every=25, face_refs=["a", "b", "c"], body_refs=["d"])
    assert webui.likeness_review_text(cfg, "face_focus") == (
        "face focus: references from your photos (face and body), likeness check every 25 steps")
    assert "(face)" in webui.likeness_review_text(cfg, "face_only")
    assert "(body)" in webui.likeness_review_text(cfg, "body_only")
    assert "3" not in webui.likeness_review_text(cfg, "face_focus")


def test_model_row_names_the_preset_for_single_files():
    cfg = TrainConfig(model_path=MOSS, model_source="files")
    assert webui.model_review_text(cfg) == "MOSS-Image-Base (single files, downloaded on start)"
    own = TrainConfig(model_path=MOSS, model_source="files", transformer_file="C:/m/moss.safetensors")
    assert webui.model_review_text(own) == "moss.safetensors (your file)"
    assert webui.model_review_text(TrainConfig(model_path=MOSS, model_source="hf")) == "MOSS-Image-Base (Hugging Face)"
    assert webui.model_review_text(TrainConfig(model_path="me/custom", model_source="hf")) == "me/custom (Hugging Face)"


# ---- W9: Save as follows a renamed job, and keeps a typed path

def test_save_path_follows_the_job_until_it_is_typed():
    first = webui.default_save_path("", False, "job_a")
    assert first == "configs/job_a.yaml"
    assert webui.default_save_path(first, False, "job_b") == "configs/job_b.yaml"  # still the wizard's own: follows
    assert webui.default_save_path("configs/mine.yaml", True, "job_b") == "configs/mine.yaml"  # typed: kept
    assert webui.default_save_path("  ", True, "job_b") == "configs/job_b.yaml"
    # a typed path that happens to equal a later job's default stays typed: renaming again does not move it
    assert webui.default_save_path("configs/bob.yaml", True, "cara") == "configs/bob.yaml"


# ---- W12, F6: the Samples note and the model-card badge

def test_keep_note():
    assert webui.keep_note_text(8, 0.98) == (
        "Keeps the newest 8 checkpoints; saves, samples and probes use a weight average of the LoRA (EMA 0.98). "
        "Advanced to change.")
    assert webui.keep_note_text(0, 0).startswith("Keeps every checkpoint; no weight average")


def test_model_fit_badge_from_the_card_and_the_table():
    assert webui.model_fit_text("moss_image", 24) == "Fits your GPU"  # int8 5.5 + 4.5 headroom
    assert webui.model_fit_text("krea2", 16) == "Fits in 4-bit"  # int8 13.5 does not fit, nvfp4 8 does
    assert webui.model_fit_text("krea2", 8) == "Needs block swap"
    assert webui.model_fit_text("not_a_family", 24) is None and webui.model_fit_text("moss_image", 0) is None
    for family, sizes in webui.MODEL_GB.items():  # monotone: a bigger card never reads worse
        order = ["Needs block swap", "Fits in 4-bit", "Fits your GPU"]
        marks = [order.index(webui.model_fit_text(family, gb)) for gb in (4, 8, 12, 16, 24, 48, 96)]
        assert marks == sorted(marks), family


# ---- W19: the wizard's own Z-Image fine-tune rate is not a warning

def test_zimage_finetune_rate_is_not_warned_about(tmp_path):
    def lr_warnings(model, lr, **kw):
        cfg = TrainConfig(model_path=model, finetune=True, learning_rate=lr, datasets=[{"path": str(tmp_path)}], **kw)
        return [w for w in webui.start_warnings(cfg) if "Learning rate" in w]

    r = webui.wizard_recommendations(TrainConfig(model_path=ZIMAGE, finetune=True), "character", [], None)
    assert r["learning_rate"][0] == webui.ZIMAGE_FINETUNE_LR
    assert not lr_warnings(ZIMAGE, r["learning_rate"][0])
    assert lr_warnings(ZIMAGE, 5e-4)  # still checked
    assert lr_warnings(MOSS, 5e-5)  # another family's fine-tune at that rate is outside its usual range


# ---- W20 / W16: wording

def test_compile_reason_names_what_was_measured():
    why = rec(MOSS)["compile_model"][1]
    assert "MOSS-Image" in why and "Krea 2" in why and "other models are not measured" in why


def test_z_image_cards_say_not_yet_run():
    for name in ("Z-Image", "Z-Image-Turbo"):
        assert "Not yet run on the real model in RulZ AI Trainer" in webui.MODEL_NOTES[name][1]
        assert "Never run on real weights" not in webui.MODEL_NOTES[name][1]


# ---- F1: seven pages, the last one reviews and starts; the extras page keeps its key

def test_wizard_has_seven_pages_with_stable_keys():
    names = [n for n, _, _ in webui.Wizard.STEPS]
    assert names == ["start", "model", "dataset", "training", "samples", "extras", "review"]
    assert webui.Wizard.LAST_STEP == "review"
    assert all(hasattr(webui.Wizard, f"step_{n}") for n in names)
