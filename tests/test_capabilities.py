"""What each model family cannot do (rulztrainer/capabilities.py): one table behind the run-start refusal and the
WebUI's greyed-out settings."""
import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import capabilities as caps  # noqa: E402
from rulztrainer.config import VIDEO_ARCHS, TrainConfig  # noqa: E402

FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}
SDXL_HF = "stabilityai/stable-diffusion-xl-base-1.0"


def sdxl(**kw) -> TrainConfig:
    kw.setdefault("model_arch", "sdxl")
    kw.setdefault("model_source", "hf")
    kw.setdefault("model_path", SDXL_HF)
    kw.setdefault("timestep_sampling", "reference")
    return TrainConfig(**kw)


def test_every_table_key_is_a_train_config_field():
    for family in ("sdxl", "moss_image", "pan2"):
        for key, rule in caps.rules_for(family).items():
            assert key in FIELDS and rule.key == key
            assert rule.reason and rule.default is not None or rule.default in (0, False, 0.0, "")
            assert rule.allows(rule.default), f"{family}.{key}: the reset value must be accepted"
            assert (rule.only or rule.forbid) and not (rule.only and rule.forbid)
            if rule.when is not None:
                assert rule.when[0] in FIELDS


def test_video_families_are_the_one_list():
    assert tuple(VIDEO_ARCHS) == caps.VIDEO_FAMILIES == ("pan2", "minimax_h3")
    assert not set(caps.FRAME_KEYS) & set(caps.rules_for("pan2"))
    assert set(caps.FRAME_KEYS) <= set(caps.rules_for("moss_image"))


# every SDXL rule: (field, a forbidden value, the config's fields to set along with it)
SDXL_BAD = [
    ("block_swap", 4, {}), ("multi_gpu", "model_split", {}), ("activation_offload", True, {}),
    ("attn_localize", True, {"trigger_word": "sks"}), ("id_align", True, {}), ("body_align", True, {}),
    ("optimizer", "keel_fb", {}), ("optimizer", "hull", {"hull_rho": 0.01}),
    ("keel_face_gain", 0.5, {"optimizer": "keel"}), ("training_maps", True, {}), ("transformer_quant", "nvfp4", {}),
    ("sample_model_quant", "nvfp4", {}), ("compile_model", True, {}), ("loss_type", "band", {}), ("loss_weighting", "sigma_sqrt", {}),
    ("finetune", True, {}), ("network_type", "lokr", {}), ("dora", True, {}), ("train_text_embedder", True, {}),
    ("turbo_lora", "x.safetensors", {}), ("sample_lora", "x.safetensors", {}),
    ("training_adapter", "x.safetensors", {}), ("checkpoint_keep_blocks", 3, {}),
    ("sample_frames", 4, {}), ("probe_frames", 4, {}), ("train_frames", 4, {}),
]


def test_the_sdxl_cases_cover_the_table():
    refused = {k for k, r in caps.rules_for("sdxl").items() if not r.ignored}
    assert {k for k, _, _ in SDXL_BAD} == refused


def test_flags_of_the_arches_and_the_table_agree():
    """The table cannot drift from what each family's Arch says it can do."""
    from rulztrainer.archs import _ARCHS, arch_by_name

    for family in _ARCHS:
        arch, rules = arch_by_name(family), caps.rules_for(family)
        assert (not arch.supports_video) == (set(caps.FRAME_KEYS) <= set(rules)), family
        assert (family in caps.VIDEO_FAMILIES) == arch.supports_video, family
        assert (not arch.supports_compile) == ("compile_model" in rules), family
        assert (not arch.supports_kept_blocks) == ("checkpoint_keep_blocks" in rules), family
        assert ("sample_shift" not in rules) == (family == "moss_image"), family
        assert ("sdxl_prediction" not in rules) == (family == "sdxl"), family
        assert ("train_video_mode" not in rules) == arch.supports_video, family


def test_ignored_settings_are_greyed_out_but_never_refused():
    for family, key, value in (("krea2", "sample_shift", 3.0), ("pan2", "sample_skip_layer", 5),
                               ("moss_image", "sdxl_prediction", "v_prediction"),
                               ("sdxl", "text_encoder_quant", "int8"), ("krea2", "train_video_mode", "spread")):
        rule = caps.rules_for(family)[key]
        assert rule.ignored and not rule.allows(value) and rule.allows(rule.default)
        assert key not in caps.violations(family, {key: value})
    TrainConfig(model_arch="krea2", sample_shift=3.0).validate()
    sdxl(text_encoder_quant="int8").validate()


@pytest.mark.parametrize("key,bad,extra", SDXL_BAD, ids=[f"{k}={b}" for k, b, _ in SDXL_BAD])
def test_sdxl_refuses_the_value_and_takes_the_default(key, bad, extra):
    with pytest.raises(ValueError, match=key) as err:
        sdxl(**{**extra, key: bad}).validate()
    assert "SDXL" in str(err.value) or key.endswith("_frames")  # names the model
    rule = caps.rules_for("sdxl")[key]
    sdxl(**{**extra, key: rule.default}).validate()  # what the WebUI resets it to is accepted
    assert key in caps.violations("sdxl", {**extra, key: bad}, ignore_when=True)
    assert key not in caps.violations("sdxl", {**extra, key: rule.default})


def test_sdxl_quant_int8_and_convrot_pass_and_nvfp4_does_not():
    for quant in ("auto", "none", "int8", "int8_convrot"):
        sdxl(transformer_quant=quant).validate()
    with pytest.raises(ValueError, match="nvfp4"):
        sdxl(transformer_quant="nvfp4").validate()
    assert caps.allowed_values("sdxl", "transformer_quant", ["auto", "none", "int8", "int8_convrot", "nvfp4"]) == [
        "auto", "none", "int8", "int8_convrot"]
    assert caps.allowed_values("krea2", "transformer_quant", ["nvfp4"]) == ["nvfp4"]  # another family: no rule
    sdxl(transformer_quant="INT8").validate()  # any case, as the loader takes it
    for bad in ("fp8", "w4a8"):  # anything unknown stops at the start, not at loading
        with pytest.raises(ValueError, match="transformer_quant"):
            sdxl(transformer_quant=bad).validate()
    sdxl(sample_model_quant="int8_convrot").validate()
    with pytest.raises(ValueError, match="sample_model_quant"):  # the sample model loads through the same loader
        sdxl(sample_model_quant="nvfp4").validate()


def test_the_sdxl_loader_takes_any_case():
    from rulztrainer.sdxl import loading

    loading._validate_quant("INT8_CONVROT")
    assert loading._norm_quant("INT8") == "int8" and loading._norm_quant(None) is None
    with pytest.raises(ValueError):
        loading._validate_quant("nvfp4")


def test_the_keel_face_gain_rule_follows_the_optimizer_in_the_form(monkeypatch):
    f = _form(monkeypatch)
    f.w["optimizer"].value = "adamw"
    f.w["keel_face_gain"] = Control(0.5)
    f.apply_family_support("sdxl", notify=True)
    assert f.w["keel_face_gain"].value == 0.5 and f.w["keel_face_gain"].enabled  # unused with adamw: left alone
    assert "keel_face_gain" not in f.notified[0]
    f.w["optimizer"].value = "keel"
    f.apply_family_support("sdxl")  # what the optimizer's change runs
    assert f.w["keel_face_gain"].value == 0.0 and not f.w["keel_face_gain"].enabled
    f.w["optimizer"].value = "adamw"
    f.apply_family_support("sdxl")
    assert f.w["keel_face_gain"].enabled


def test_a_wizard_copy_of_a_one_choice_select_is_disabled_too(monkeypatch):
    f = _form(monkeypatch)
    copy = Control("band", ["mse", "band"])
    f.family_mirrors["loss_type"] = [copy]
    f.apply_family_support("sdxl")
    assert copy.options == ["mse"] and not copy.enabled
    f.apply_family_support("moss_image")
    assert copy.enabled and copy.options == ["mse", "band"]


def test_keel_face_gain_is_refused_only_with_a_keel_optimizer():
    sdxl(optimizer="adamw", keel_face_gain=0.5).validate()  # unused, as before
    with pytest.raises(ValueError, match="keel_face_gain"):
        sdxl(optimizer="keel", keel_face_gain=0.5).validate()


def test_optimizer_names_compare_without_case_and_bool_is_not_one():
    with pytest.raises(ValueError, match="keel_fb"):
        sdxl(optimizer="Keel_FB").validate()
    assert caps.rules_for("sdxl")["block_swap"].allows(0) and not caps.rules_for("sdxl")["block_swap"].allows(False)


def test_video_frames_need_a_video_model():
    for arch in ("moss_image", "krea2", "sdxl"):
        for key in caps.FRAME_KEYS:
            cfg = TrainConfig(model_arch=arch, **{key: 3})
            with pytest.raises(ValueError, match="video model"):
                cfg.validate()
    for arch in VIDEO_ARCHS:
        TrainConfig(model_arch=arch, sample_frames=5, probe_frames=5).validate()
    TrainConfig(model_arch="moss_image").validate()


def test_a_message_names_setting_value_model_and_reason():
    msg = caps.violations("sdxl", {"training_maps": True})["training_maps"]
    assert msg.startswith("training_maps True: not available for SDXL: ")
    assert caps.reason("sdxl", "training_maps").startswith("Not available for SDXL: ")
    assert caps.reason("krea2", "training_maps") is None
    assert caps.display_name("krea2") == "Krea 2"


# ----------------------------------------------------------------------------------------------- the WebUI's side
class Control:
    """A NiceGUI control as apply_family_support uses it."""

    _ids = iter(range(1, 10_000))

    def __init__(self, value=None, options=None):
        self.id = next(self._ids)
        self.value, self.enabled, self.visible, self.is_deleted = value, True, True, False
        if options is not None:
            self.options = options

    def set_options(self, options, value=None):
        self.options, self.value = options, value

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


class Note:
    def __init__(self):
        self.text, self.visible = "", False


def _form(monkeypatch):
    import webui

    f = webui.Form()
    quants = ["auto", "none", "int8", "int8_convrot", "nvfp4"]
    f.w.update({
        "model_preset": Control("MOSS-Image-Base"), "model_path": Control(""),
        "training_maps": Control(True), "activation_offload": Control(True), "block_swap": Control(4),
        "transformer_quant": Control("nvfp4", list(quants)), "multi_gpu": Control("model_split", {
            "off": "single", "data_parallel": "dp", "model_split": "split"}),
        "compile_model": Control(True, {"auto": "auto", True: "on", False: "off"}),
        "optimizer": Control("hull", ["adamw", "adamw8bit", "hull", "keel_fb", "keel"]),
        "turbo_mode": Control("file", {"auto": "auto", "none": "none", "file": "file"}),
        "loss_type": Control("band", ["mse", "band"]), "lora_rank": Control(16)})
    f.family_mirrors["transformer_quant"] = [Control("nvfp4", list(quants))]  # the wizard's own select
    for key in ("training_maps", "turbo_mode"):
        f.family_notes[key] = [Note()]
    monkeypatch.setattr(webui.Form, "family_note", lambda self, key: None)  # needs a page; the labels above stand in
    monkeypatch.setattr(webui.ui, "notify", lambda *a, **k: f.__dict__.setdefault("notified", []).append(a[0]))
    return f


def test_switching_to_sdxl_greys_out_resets_and_explains_then_switching_back_frees_everything(monkeypatch):
    f = _form(monkeypatch)
    w = f.w
    reset = f.apply_family_support("sdxl", notify=True)
    assert set(reset) >= {"training_maps", "activation_offload", "block_swap", "transformer_quant", "multi_gpu",
                          "compile_model", "optimizer", "turbo_lora", "loss_type"}
    assert (w["training_maps"].value, w["activation_offload"].value, w["block_swap"].value) == (False, False, 0)
    assert not w["training_maps"].enabled and not w["activation_offload"].enabled and not w["block_swap"].enabled
    assert w["lora_rank"].enabled  # no rule
    assert w["transformer_quant"].value == "auto" and "nvfp4" not in w["transformer_quant"].options
    assert "int8_convrot" in w["transformer_quant"].options and w["transformer_quant"].enabled
    assert "nvfp4" not in f.family_mirrors["transformer_quant"][0].options  # the wizard's select too
    assert f.family_mirrors["transformer_quant"][0].value == "auto"
    assert set(w["multi_gpu"].options) == {"off", "data_parallel"} and w["multi_gpu"].value == "off"
    assert set(w["compile_model"].options) == {"auto", False} and w["compile_model"].value == "auto"
    assert "hull" not in w["optimizer"].options and w["optimizer"].value == "adamw8bit"
    assert w["turbo_mode"].value == "auto" and set(w["turbo_mode"].options) == {"auto", "none"}
    assert w["loss_type"].options == ["mse"] and w["loss_type"].value == "mse" and not w["loss_type"].enabled
    assert f.family_notes["training_maps"][0].text == "Not available for SDXL: " + caps.rules_for("sdxl")[
        "training_maps"].reason
    assert f.family_notes["training_maps"][0].visible
    assert len(f.notified) == 1 and "training_maps" in f.notified[0] and "not available for SDXL" in f.notified[0]
    # a second pass changes nothing and says nothing
    assert f.apply_family_support("sdxl", notify=True) == [] and len(f.notified) == 1
    # back to a family without the rules
    f.apply_family_support("moss_image")
    assert w["training_maps"].enabled and w["activation_offload"].enabled and w["block_swap"].enabled
    assert w["loss_type"].enabled and w["loss_type"].options == ["mse", "band"]
    assert "nvfp4" in w["transformer_quant"].options and "nvfp4" in f.family_mirrors["transformer_quant"][0].options
    assert set(w["multi_gpu"].options) == {"off", "data_parallel", "model_split"}
    assert set(w["turbo_mode"].options) == {"auto", "none", "file"}
    assert f.family_notes["training_maps"][0].text == "" and not f.family_notes["training_maps"][0].visible
    assert w["training_maps"].value is False  # the reset value stays


def test_an_ignored_setting_is_greyed_out_but_keeps_its_value(monkeypatch):
    f = _form(monkeypatch)
    f.w["sample_shift"] = Control(3.0)
    f.w["sdxl_prediction"] = Control("v_prediction", ["auto", "epsilon", "v_prediction"])
    f.apply_family_support("sdxl")  # SDXL ignores the shift
    assert f.w["sample_shift"].value == 3.0 and not f.w["sample_shift"].enabled
    assert f.w["sdxl_prediction"].enabled
    reset = f.apply_family_support("krea2", notify=True)  # Krea 2 ignores both: greyed out, nothing put back or said
    assert "sample_shift" not in reset and "sdxl_prediction" not in reset
    assert f.w["sdxl_prediction"].value == "v_prediction" and not f.w["sdxl_prediction"].enabled
    assert f.w["sdxl_prediction"].options == ["auto", "epsilon", "v_prediction"]
    f.apply_family_support("sdxl")  # back on SDXL the choice is still the user's
    assert f.w["sdxl_prediction"].value == "v_prediction" and f.w["sdxl_prediction"].enabled
    f.apply_family_support("moss_image")
    assert f.w["sample_shift"].value == 3.0 and f.w["sample_shift"].enabled


def test_release_lifts_the_restrictions_before_values_load(monkeypatch):
    f = _form(monkeypatch)
    f.apply_family_support("sdxl")
    f.apply_family_support(release=True)
    assert "nvfp4" in f.w["transformer_quant"].options and f.w["training_maps"].enabled
