"""The wizard's MOSS-Image defaults (3.54.0): reference timesteps with a shift of 1, weight decay 1e-4, mixed 512 / 768 /
1024 px buckets, no preview skip layer, and full-quality previews at 25 steps, CFG 3.5; the other families keep theirs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

webui = pytest.importorskip("webui")
from rulztrainer.config import ARCH_FULL_SAMPLE, ARCH_TURBO, TrainConfig  # noqa: E402

MOSS = "OpenMOSS-Team/MOSS-Image-Base"
GPU24, GPU8 = [{"total": 24000}], [{"total": 8000}]


def _rec(model=MOSS, goal="character", gpus=GPU24, data=None, **kw):
    return webui.wizard_recommendations(TrainConfig(model_path=model, **kw), goal, gpus, data)


def test_moss_character_defaults():
    r = _rec()
    assert r["timestep_sampling"][0] == "reference" and r["timestep_shift"][0] == 1.0
    assert r["weight_decay"][0] == 1e-4
    assert r["resolutions"][0] == [0.25, 0.5, 1.0]
    assert r["sample_skip_layer"][0] == -1
    assert "resolution_shift" not in r  # reference noise levels: no size shift on top
    # unchanged: rank 32 / alpha 32, adamw8bit at 1e-4
    assert (r["lora_rank"][0], r["lora_alpha"][0], r["optimizer"][0], r["learning_rate"][0]) == \
        (32, 32.0, "adamw8bit", 1e-4)


def test_moss_full_quality_previews_are_25_steps_cfg_3_5_and_turbo_stays_as_it_was():
    assert ARCH_FULL_SAMPLE["moss_image"] == (25, 3.5)
    r = _rec(turbo_lora="")  # no turbo LoRA: the full-quality sampler (previews and probes)
    assert (r["sample_steps"][0], r["sample_cfg"][0]) == (25, 3.5)
    assert (r["probe_steps"][0], r["probe_cfg"][0]) == (25, 3.5)
    assert webui.get_arch(TrainConfig(model_path=MOSS)).default_sample_steps == 25
    assert webui.get_arch(TrainConfig(model_path=MOSS)).default_sample_cfg == 3.5
    turbo = _rec()  # the default: the turbo LoRA (12 steps, CFG 1)
    assert (turbo["sample_steps"][0], turbo["sample_cfg"][0]) == (ARCH_TURBO["moss_image"]["steps"], 1.0) == (12, 1.0)
    assert (turbo["probe_steps"][0], turbo["probe_cfg"][0]) == (12, 1.0)


def test_moss_sizes_follow_vram_and_the_photos():
    assert _rec(gpus=[{"total": 12288}])["resolutions"][0] == [0.25, 0.5, 1.0]
    assert _rec(gpus=[])["resolutions"][0] == [0.25, 0.5, 1.0]  # no GPU found: the 16 GB+ assumption
    assert _rec(gpus=GPU8)["resolutions"][0] == [0.5]  # a small card keeps its smaller pick
    assert _rec(data={"median_mp": 0.6})["resolutions"][0] == [0.25, 0.5]  # no size above the photos'
    assert _rec(data={"median_mp": 0.3})["resolutions"][0] == [0.25]
    assert _rec(data={"median_mp": 2.0})["resolutions"][0] == [0.25, 0.5, 1.0]


def test_moss_style_and_concept_keep_their_noise_recipe():
    style = _rec(goal="style")
    assert style["timestep_sampling"][0] == "logsnr" and "timestep_shift" not in style
    assert style["weight_decay"][0] == 1e-4 and style["sample_skip_layer"][0] == -1
    assert _rec(goal="concept")["timestep_sampling"][0] == "logit_normal"


def test_no_weight_decay_recommendation_for_a_fine_tune_or_automagic():
    assert "weight_decay" not in _rec(finetune=True)
    assert "weight_decay" not in _rec(optimizer="automagic4")


@pytest.mark.parametrize("model", ["krea/Krea-2-Raw", "Qwen/Qwen-Image-2.1", "black-forest-labs/flux-3-image-klein-base"])
def test_other_families_keep_their_values(model):
    r = _rec(model=model)
    for key in ("weight_decay", "sample_skip_layer", "timestep_shift"):
        assert key not in r, (model, key)
    assert r["resolutions"][0] in ([1.0], [0.5])  # never the MOSS mix
    assert "timestep_sampling" not in r  # a character: the identity recipe's logit_normal stays


def test_krea_and_qwen_previews_and_sizes_are_what_they_were():
    krea = _rec(model="krea/Krea-2-Raw", turbo_lora="")
    assert krea["resolutions"][0] == [1.0] and (krea["sample_steps"][0], krea["sample_cfg"][0]) == (28, 5.5)
    qwen = _rec(model="Qwen/Qwen-Image-2.1", turbo_lora="")
    assert qwen["resolutions"][0] == [0.5] and (qwen["sample_steps"][0], qwen["sample_cfg"][0]) == (25, 1.0)
    assert ARCH_FULL_SAMPLE["krea2"] == (28, 5.5) and ARCH_FULL_SAMPLE["qwen_image21"] == (25, 1.0)


def test_the_wizard_applies_the_moss_values_over_the_identity_recipe():
    """Everything MODEL_WIZARD_AUTO names has a recommendation, and a value to go back to; the Noise levels extra counts
    reference as its off value while the wizard holds it (so 'no extras' does not undo it)."""
    keys = webui.MODEL_WIZARD_AUTO["moss_image"]
    r = _rec()
    assert all(k in r for k in keys) and all(k in webui.MODEL_AUTO_OFF for k in keys)
    assert webui.MODEL_AUTO_OFF["timestep_sampling"] == "logit_normal"  # the identity recipe's
    assert webui.MODEL_AUTO_OFF["weight_decay"] == TrainConfig.weight_decay == 0.01
    assert webui.MODEL_AUTO_OFF["sample_skip_layer"] == TrainConfig.sample_skip_layer == 22
    assert set(keys).isdisjoint(webui.WIZARD_AUTO)  # applied per family, not for every model
    ex = next(e for e in webui.EXPERIMENTS if e["key"] == "timestep_sampling")
    wiz = webui.Wizard.__new__(webui.Wizard)
    wiz.f, wiz.model_auto_loads = None, 0  # no config loaded since
    wiz.model_auto_applied = {}
    assert wiz.experiment_off(ex) == "logit_normal"
    wiz.model_auto_applied = {"timestep_sampling": "reference"}
    assert wiz.experiment_off(ex) == "reference"


def test_old_saved_configs_keep_their_skip_layer_and_weight_decay():
    assert TrainConfig.from_dict({"sample_skip_layer": 22, "weight_decay": 0.01}).sample_skip_layer == 22
    assert TrainConfig().sample_skip_layer == 22 and TrainConfig().weight_decay == 0.01
    assert TrainConfig.from_dict({"sample_skip_layer": -1, "weight_decay": 1e-4}).weight_decay == 1e-4


# ---- Wizard.follow_model_auto, on stand-in widgets (the real ones need a NiceGUI page)
class _W:
    def __init__(self, value):
        self.value = value


def _wizard(family_recs: dict, form_values: dict):
    """A Wizard with only what follow_model_auto needs: the form's widgets, its load count and a refresh that reads
    the family picked (state["family"]) and its recommendations."""
    from types import SimpleNamespace
    wz = webui.Wizard.__new__(webui.Wizard)
    wz.w = {k: _W(v) for k, v in form_values.items()}
    wz.f = SimpleNamespace(loads=0)
    wz.set_up, wz.family, wz.recs = True, "", {}
    wz.model_auto_family, wz.model_auto_applied, wz.model_auto_prev, wz.model_auto_loads = "", {}, {}, 0
    state = {"family": "moss_image"}

    def refresh():
        wz.family = state["family"]
        wz.recs = family_recs.get(wz.family, {})
        if wz.set_up and (wz.family != wz.model_auto_family or wz.model_auto_stale()):
            wz.follow_model_auto()

    wz.refresh = refresh
    return wz, state


MOSS_RECS = {"moss_image": {"timestep_sampling": ("reference", ""), "timestep_shift": (1.0, ""),
                            "weight_decay": (1e-4, ""), "sample_skip_layer": (-1, "")}}
FORM = {"timestep_sampling": "logit_normal", "timestep_shift": 1.0, "weight_decay": 0.005, "sample_skip_layer": 22}


def test_follow_model_auto_applies_and_gives_back_displaced_values():
    wz, state = _wizard(MOSS_RECS, FORM)
    wz.refresh()
    assert {k: w.value for k, w in wz.w.items()} == {"timestep_sampling": "reference", "timestep_shift": 1.0,
                                                    "weight_decay": 1e-4, "sample_skip_layer": -1}
    wz.w["sample_skip_layer"].value = 24  # the user's own change stays
    state["family"] = "krea2"
    wz.refresh()
    assert wz.w["timestep_sampling"].value == "logit_normal"
    assert wz.w["weight_decay"].value == 0.005  # what it displaced, not the trainer default
    assert wz.w["sample_skip_layer"].value == 24


def test_follow_model_auto_keeps_a_loaded_config():
    wz, state = _wizard(MOSS_RECS, FORM)
    wz.refresh()
    wz.f.loads += 1  # a MOSS identity config with its own values is loaded
    wz.w["timestep_sampling"].value = "logit_normal"
    ex = {"key": "timestep_sampling", "off": "logit_normal"}
    assert wz.experiment_off(ex) == "logit_normal"  # not the wizard's reference any more
    wz.refresh()
    assert wz.w["timestep_sampling"].value == "logit_normal" and wz.model_auto_applied == {}
    state["family"] = "krea2"  # leaving the family afterwards takes nothing back
    wz.refresh()
    assert wz.w["timestep_sampling"].value == "logit_normal"


def test_follow_model_auto_after_reset_falls_back_to_defaults():
    wz, state = _wizard(MOSS_RECS, {**FORM, "timestep_sampling": "reference", "weight_decay": 1e-4})
    wz.refresh()  # the form already held MOSS's values (as after Reset): nothing known to give back
    state["family"] = "krea2"
    wz.refresh()
    assert wz.w["timestep_sampling"].value == "logit_normal"
    assert wz.w["weight_decay"].value == TrainConfig.weight_decay
