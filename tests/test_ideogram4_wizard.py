"""The wizard's Ideogram 4 defaults, from community reports (not run in RulZ AI Trainer): a factor-4 LoKr, Automagic 3 from
1e-5 (AdamW8bit with isolation on), logit-normal timesteps, 512 / 768 / 1024 px buckets, batch 1, about 200 steps per
picture, a checkpoint every 250 steps, and previews without the turbo LoRA; the other families keep theirs."""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

webui = pytest.importorskip("webui")
from rulztrainer.comfy_files import planned_downloads  # noqa: E402
from rulztrainer.config import (ARCH_FULL_SAMPLE, ARCH_SAMPLE_DEFAULTS, ARCH_TURBO, ARCH_TURBO_DEFAULT_OFF,  # noqa: E402
                                TrainConfig)
from rulztrainer.trainer import automagic_floor  # noqa: E402

IDEO = "ideogram-ai/ideogram-4-fp8"
MOSS = "OpenMOSS-Team/MOSS-Image-Base"
GPU24 = [{"total": 24000}]
HANDLES = ("knochey", "plaguekind", "fitzycodesthings", "tophat889", "pashahlis", "mamad851", "brendacaro", "gordo9500",
           "dg1860", "thebollo", "fatalis", "mtmcreations", "ashtar1984", "jigen", "discord")


def _rec(model=IDEO, goal="character", gpus=GPU24, data=None, **kw):
    return webui.wizard_recommendations(TrainConfig(model_path=model, **kw), goal, gpus, data)


def test_ideogram_character_defaults():
    r = _rec(data={"images": 30}, steps=6000)
    assert r["network_type"][0] == "lokr" and r["lokr_factor"][0] == 4
    assert (r["optimizer"][0], r["learning_rate"][0]) == ("automagic3", 1e-5)
    assert r["timestep_sampling"][0] == "logit_normal" and r["resolution_shift"][0] is True
    assert r["resolutions"][0] == [0.25, 0.5, 1.0]
    assert r["batch_size"][0] == 1
    assert r["steps"][0] == 6000  # 30 images x about 200
    assert r["save_every"][0] == 250
    assert r["turbo_mode"][0] == "none"
    for key in ("network_type", "lokr_factor", "optimizer", "learning_rate", "timestep_sampling", "resolutions",
                "steps", "save_every", "turbo_mode", "batch_size"):
        assert r[key][1], key  # a why for each


def test_steps_follow_the_pictures_within_3000_to_8000():
    assert _rec(data={"images": 10})["steps"][0] == 3000
    assert _rec(data={"images": 40})["steps"][0] == 8000
    assert _rec(data={"images": 100})["steps"][0] == 8000
    assert _rec(data={"images": 25})["steps"][0] == 5000
    assert _rec()["steps"][0] == 4000  # nothing scanned yet


def test_checkpoints_every_250_and_a_dozen_kept_at_most():
    assert _rec(steps=2000)["keep_checkpoints"][0] == 0
    assert _rec(steps=6000)["keep_checkpoints"][0] == 12
    assert _rec(steps=6000)["sample_every"][0] == 250


def test_automagic_floor_with_the_recommended_start_is_a_fifth_of_it():
    cfg = TrainConfig(model_path=IDEO, optimizer="automagic3", learning_rate=1e-5)
    assert cfg.automagic_min_lr == 2e-5  # not below the start rate ...
    assert automagic_floor(cfg) == pytest.approx(2e-6)  # ... so the trainer's floor is a fifth of the start
    assert "2e-6" in _rec()["learning_rate"][1]


def test_isolation_on_picks_adamw_because_the_trainer_swaps_automagic():
    r = _rec(isolation=True)
    assert (r["optimizer"][0], r["learning_rate"][0]) == ("adamw8bit", 5e-5)
    assert (r["network_type"][0], r["lokr_factor"][0]) == ("lokr", 4)
    # what the trainer would otherwise do with Automagic at 1e-5
    from rulztrainer.config import effective_optimizer
    name, lr, _ = effective_optimizer(TrainConfig(model_path=IDEO, optimizer="automagic3", learning_rate=1e-5,
                                                  isolation=True))
    assert name == "adamw" and lr == 1e-5  # too low for AdamW: the reason the wizard picks 5e-5 instead


def test_style_goal_is_adamw_8e5_and_keeps_its_noise_recipe():
    r = _rec(goal="style")
    assert (r["optimizer"][0], r["learning_rate"][0]) == ("adamw8bit", 8e-5)
    assert r["timestep_sampling"][0] == "logsnr"
    assert "network_type" not in r or r["network_type"][0] != "lokr"
    assert r["resolutions"][0] == [1.0]


def test_resolutions_follow_vram_and_the_photos():
    assert _rec(gpus=[{"total": 12288}])["resolutions"][0] == [0.25, 0.5, 1.0]
    assert _rec(gpus=[{"total": 8000}])["resolutions"][0] == [0.5]  # a small card keeps its smaller pick
    assert _rec(data={"median_mp": 0.6})["resolutions"][0] == [0.25, 0.5]
    assert _rec(data={"median_mp": 0.3})["resolutions"][0] == [0.25]


def test_previews_are_full_quality_without_turbo_and_turbo_stays_one_pick_away():
    assert ARCH_FULL_SAMPLE["ideogram4"] == (20, 7.0)
    assert webui.get_arch(TrainConfig(model_path=IDEO)).default_sample_steps == 20
    assert webui.get_arch(TrainConfig(model_path=IDEO)).default_sample_cfg == 7.0
    assert ARCH_TURBO_DEFAULT_OFF == ("ideogram4",)
    off = _rec(turbo_lora="")  # what the wizard sets (turbo_mode none)
    assert (off["sample_steps"][0], off["sample_cfg"][0]) == (20, 7.0)
    assert (off["probe_steps"][0], off["probe_cfg"][0]) == (20, 7.0)
    on = _rec()  # turbo_lora "auto": a saved config's own pick still gets the turbo values
    assert (on["sample_steps"][0], on["sample_cfg"][0]) == (ARCH_TURBO["ideogram4"]["steps"], 1.0) == (8, 1.0)
    assert ARCH_SAMPLE_DEFAULTS["ideogram4"] == (8, 1.0)


def test_saved_configs_keep_their_turbo_lora():
    assert TrainConfig.from_dict({"model_path": IDEO, "turbo_lora": "auto"}).turbo_lora == "auto"
    assert TrainConfig.from_dict({"model_path": IDEO, "turbo_lora": "t.safetensors"}).turbo_lora == "t.safetensors"
    assert "turbo_lora" in [f for f, _, _ in planned_downloads(TrainConfig(model_path=IDEO))]  # auto still downloads
    assert "turbo_lora" not in [f for f, _, _ in planned_downloads(TrainConfig(model_path=IDEO, turbo_lora=""))]


def test_recommended_ideogram_setup_is_a_valid_config():
    r = _rec(data={"images": 30})
    cfg = TrainConfig(model_path=IDEO, optimizer=r["optimizer"][0], learning_rate=r["learning_rate"][0],
                      network_type=r["network_type"][0], lokr_factor=r["lokr_factor"][0],
                      timestep_sampling=r["timestep_sampling"][0], resolutions=r["resolutions"][0],
                      steps=r["steps"][0], save_every=r["save_every"][0], turbo_lora="", datasets=[{"path": "x"}])
    cfg.validate()


@pytest.mark.parametrize("model", [MOSS, "krea/Krea-2-Raw", "Qwen/Qwen-Image-2.1",
                                   "black-forest-labs/flux-3-image-klein-base"])
def test_other_families_are_unchanged(model):
    r = _rec(model=model)
    assert "turbo_mode" not in r and "lokr_factor" not in r
    assert r["network_type"][0] in ("lora", "oft")
    assert r["optimizer"][0] == "adamw8bit" and r["learning_rate"][0] in (1e-4, 2e-4, 1.5e-4, 5e-5)
    assert r["steps"][0] <= 4000 and r["keep_checkpoints"][0] in (0, 8)
    assert "community reports" not in " ".join(str(v[1]) for k, v in r.items() if k in ("steps", "save_every"))
    assert _rec(model=model, steps=6000)["keep_checkpoints"][0] == 8  # not Ideogram's dozen


def test_moss_and_krea_values_are_what_they_were():
    moss = _rec(model=MOSS)
    assert moss["timestep_sampling"][0] == "reference" and moss["resolutions"][0] == [0.25, 0.5, 1.0]
    assert (moss["lora_rank"][0], moss["optimizer"][0], moss["learning_rate"][0]) == (32, "adamw8bit", 1e-4)
    krea = _rec(model="krea/Krea-2-Raw", turbo_lora="")
    assert krea["resolutions"][0] == [1.0] and (krea["sample_steps"][0], krea["sample_cfg"][0]) == (28, 5.5)


def test_ideogram_values_are_given_back_on_leaving_the_family():
    keys = webui.MODEL_WIZARD_AUTO["ideogram4"]
    r = _rec()
    assert all(k in r for k in keys) and all(k in webui.MODEL_AUTO_OFF for k in keys)
    assert webui.MODEL_AUTO_OFF["network_type"] == "lora" and webui.MODEL_AUTO_OFF["optimizer"] == "adamw8bit"
    assert webui.MODEL_AUTO_OFF["learning_rate"] == 1e-4 and webui.MODEL_AUTO_OFF["turbo_mode"] == "auto"
    assert webui.MODEL_AUTO_OFF["lokr_factor"] == TrainConfig.lokr_factor == 8
    assert set(keys).isdisjoint(webui.WIZARD_AUTO)
    assert webui.MODEL_WIZARD_AUTO["moss_image"] == ("timestep_sampling", "timestep_shift", "weight_decay",
                                                     "sample_skip_layer")


class _W:
    def __init__(self, value):
        self.value = value


def _wizard(recs: dict, form_values: dict):
    wz = webui.Wizard.__new__(webui.Wizard)
    wz.w = {k: _W(v) for k, v in form_values.items()}
    wz.f = SimpleNamespace(loads=0)
    wz.set_up, wz.family, wz.recs = True, "", {}
    wz.model_auto_family, wz.model_auto_applied, wz.model_auto_prev, wz.model_auto_loads = "", {}, {}, 0
    state = {"family": "ideogram4"}

    def refresh():
        wz.family = state["family"]
        wz.recs = recs.get(wz.family, {})
        if wz.set_up and (wz.family != wz.model_auto_family or wz.model_auto_stale()):
            wz.follow_model_auto()

    wz.refresh = refresh
    return wz, state


def test_follow_model_auto_applies_the_ideogram_setup_and_gives_back_what_it_displaced():
    keys = webui.MODEL_WIZARD_AUTO["ideogram4"]
    recs = {"ideogram4": {k: v for k, v in _rec().items() if k in keys}}
    form = {"network_type": "lora", "lokr_factor": 8, "optimizer": "adamw8bit", "learning_rate": 1e-4,
            "timestep_sampling": "logit_normal", "turbo_mode": "auto"}
    wz, state = _wizard(recs, form)
    wz.refresh()
    assert {k: w.value for k, w in wz.w.items()} == {"network_type": "lokr", "lokr_factor": 4, "optimizer": "automagic3",
                                                    "learning_rate": 1e-5, "timestep_sampling": "logit_normal",
                                                    "turbo_mode": "none"}
    ex = next(e for e in webui.EXPERIMENTS if e["key"] == "network_type")
    assert wz.experiment_off(ex) == "lokr"  # answering "no extras" keeps the family's adapter
    state["family"] = "krea2"
    wz.refresh()
    assert {k: w.value for k, w in wz.w.items()} == form


def test_form_turbo_follows_the_family_but_never_a_users_pick():
    form = webui.Form.__new__(webui.Form)
    form.w = {"turbo_mode": _W("auto")}
    form.follow_family_turbo("moss_image", "ideogram4")
    assert form.w["turbo_mode"].value == "none" and form.turbo_on() is False
    form.follow_family_turbo("ideogram4", "krea2")  # the none it set goes back to auto
    assert form.w["turbo_mode"].value == "auto"
    form.w["turbo_mode"].value = "none"  # the user's own none stays
    form.follow_family_turbo("ideogram4", "krea2")
    assert form.w["turbo_mode"].value == "none"
    form.w["turbo_mode"].value = "file"
    form.follow_family_turbo("krea2", "ideogram4")  # a chosen file stays too
    assert form.w["turbo_mode"].value == "file"
    form.w["turbo_mode"].value = "auto"
    form.follow_family_turbo("ideogram4", "ideogram4")  # no change of family: nothing
    assert form.w["turbo_mode"].value == "auto"


def _public_texts() -> list[str]:
    root = Path(webui.__file__).parent
    readme = (root / "README.md").read_text(encoding="utf-8")
    section = readme.split("\n## Ideogram 4\n", 1)[1].split("\n## MiniMax H3\n", 1)[0]
    src = Path(webui.__file__).read_text(encoding="utf-8")
    help_text = src.split("\n## Ideogram 4\n", 1)[1].split("\n## MiniMax H3\n", 1)[0]
    texts = [section, help_text]
    for goal in ("character", "style", "concept"):
        for kw in ({}, {"isolation": True}, {"turbo_lora": ""}):
            texts += [str(reason) for _, reason in _rec(goal=goal, **kw).values()]
    # the form's and the wizard's Ideogram 4 hints and settings (the strings next to "Ideogram 4" in webui.py)
    texts += re.findall(r'"(Ideogram 4[^"]*)"', src)
    return texts


def test_public_text_has_no_handles_and_says_it_is_community_evidence():
    for text in _public_texts():
        low = text.lower()
        for handle in HANDLES:
            assert handle not in low, (handle, text[:80])
        assert "C:\\" not in text and "elena" not in low
    ideo = " ".join(str(v[1]) for v in _rec(data={"images": 30}).values())
    assert "community reports" in ideo and "not compared in RulZ AI Trainer" in ideo
    section = _public_texts()[0]
    assert "community reports" in section and "none of these settings has been run" in section.lower()
    assert "ostris' TurboTime" in section  # the credit stays
    for old in ("JSON trains it best", "trains best", "trained on structured JSON captions only"):
        assert old not in section and old not in _public_texts()[1]


def test_caption_texts_carry_the_disputed_reality():
    src = Path(webui.__file__).read_text(encoding="utf-8")
    readme = (Path(webui.__file__).parent / "README.md").read_text(encoding="utf-8")
    for needle in ("high_level_description", "colour palette", "multi-character", "hair or eye colour"):
        assert needle in src and needle in readme, needle
    for text in (src, readme):
        assert "[y_min, x_min, y_max, x_max]" in text or "`[y_min, x_min, y_max, x_max]`" in text
        assert "normalised 0-1000" in text and "swap x and y" in text.replace("\n  ", " ").replace("\n", " ")
