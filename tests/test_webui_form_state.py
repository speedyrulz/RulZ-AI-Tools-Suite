"""The page keeps its settings: raw snapshots survive what to_config() would refuse or drop, what this version can no
longer take is skipped, the saved file is written whole, and a config comes in from text as it does from a file."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


class Box:
    """A form widget: .value, and .options on a choice."""

    def __init__(self, value=None, options=None):
        self.value = value
        if options is not None:
            self.options = options


class Unreadable(Box):
    """Like the race rows while one is half typed."""

    @property
    def value(self):
        raise ValueError("cannot read this")

    @value.setter
    def value(self, _):
        pass


def _form(**widgets) -> webui.Form:
    f = webui.Form()
    f.w.update({"model_preset": Box("MOSS-Image-Base"), "model_path": Box(""), **widgets})
    return f


def test_snapshot_keeps_values_to_config_would_refuse():
    f = _form(steps=Box(None), soup_files=Box("a.safetensors"), soup_run=Box(False))
    f.extras["cfg_path"] = Box("configs/mine.yaml")
    snap = f.snapshot()
    assert snap["steps"] is None
    assert snap["soup_files"] == "a.safetensors"  # to_config() drops the files while the soup is off
    assert snap["cfg_path"] == "configs/mine.yaml"


def test_snapshot_keeps_the_last_readable_value():
    f = _form(race_configs=Unreadable())
    assert f.snapshot({"race_configs": [{"name": "a"}]})["race_configs"] == [{"name": "a"}]
    assert "race_configs" not in f.snapshot()


def test_restore_round_trip_through_json():
    f = _form(steps=Box(100), resolutions=Box([1.0], [0.5, 1.0, 1.5]))
    f.w["steps"].value, f.w["resolutions"].value = 2500, [0.5, 1.5]
    saved = json.loads(json.dumps(f.snapshot()))
    f.w["steps"].value, f.w["resolutions"].value = 100, [1.0]
    f.restore(saved)
    assert f.w["steps"].value == 2500
    assert f.w["resolutions"].value == [0.5, 1.5]


def test_restore_skips_what_this_version_cannot_take():
    f = _form(device=Box("cuda:0", ["cuda:0"]), steps=Box(100))
    f.w["model_preset"].options = list(webui.MODEL_PRESETS)
    f.restore({"device": "cuda:3", "gone_option": 1, "steps": 200, "model_preset": "Removed model"})
    assert f.w["device"].value == "cuda:0"
    assert f.w["steps"].value == 200
    assert f.w["model_preset"].value == "MOSS-Image-Base"
    assert "gone_option" not in f.w


def test_restore_leaves_side_effects_off_and_follows_the_family():
    seen = []

    class Watched(Box):
        def __setattr__(self, name, value):
            if name == "value" and hasattr(self, "value"):
                seen.append((f.loading, f.preset_sync))
            super().__setattr__(name, value)

    f = _form(preset_toggle=Watched(False))
    f.restore({"preset_toggle": True, "model_preset": "Krea-2-Raw"})
    assert seen == [(True, True)]
    assert not f.loading and not f.preset_sync
    assert f.target_family == "krea2"


def test_the_wizards_format_lists_change_before_the_value():
    """The wizard's select is bound to the form's: with the old family's list, fp8_scaled came back blank."""
    calls = []

    class Choice(Box):
        def set_options(self, options, value=None):
            calls.append((self.name, list(options)))
            self.options = options
            if value is not None:
                self.value = value

    form_sel, copy = Choice("int8_convrot", ["int8_convrot"]), Choice("int8_convrot", ["int8_convrot"])
    form_sel.name, copy.name = "form", "wizard"
    f = _form(model_file_quant=form_sel)
    f.quant_copies["model_file_quant"] = copy
    f.set_quant_options("krea2")
    assert [c[0] for c in calls] == ["wizard", "form"]
    assert "fp8_scaled" in copy.options


@pytest.mark.parametrize(("value", "fits"), [("a", True), ("z", False), (["a", "b"], True), (["a", "z"], False),
                                             (None, True), ({"x": 1}, False)])
def test_fits_options_with_a_dict_of_choices(value, fits):
    assert webui.fits_options(Box("a", {"a": "A", "b": "B"}), value) is fits


def test_fits_options_without_choices():
    assert webui.fits_options(Box("a"), {"anything": 1})


def test_form_file_round_trip_leaves_no_temporary(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "FORM_FILE", tmp_path / ".webui_form.json")
    assert webui.load_form_state() is None
    assert webui.save_form_state(json.dumps({"advanced": True}))
    assert webui.load_form_state() == {"advanced": True}
    assert [p.name for p in tmp_path.iterdir()] == [".webui_form.json"]


@pytest.mark.parametrize("text", ["{torn", "[1, 2]"])
def test_a_bad_form_file_counts_as_none(tmp_path, monkeypatch, text):
    monkeypatch.setattr(webui, "FORM_FILE", tmp_path / ".webui_form.json")
    webui.FORM_FILE.write_text(text, encoding="utf-8")
    assert webui.load_form_state() is None


def test_a_failed_write_reports_false(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "FORM_FILE", tmp_path / "missing_folder" / ".webui_form.json")
    assert webui.save_form_state("{}") is False


def test_config_yaml_text_round_trip():
    cfg = TrainConfig(job_name="round_trip", steps=1234)
    back, missing, unknown = TrainConfig.parse_report(cfg.to_yaml(), "a.yaml")
    assert back.to_dict() == cfg.to_dict()
    assert missing == [] and unknown == []


def test_parse_report_reports_like_load_report(tmp_path):
    text = "job_name: old\nretired_option: 1\n"
    path = tmp_path / "old.yaml"
    path.write_text(text, encoding="utf-8")
    cfg, missing, unknown = TrainConfig.parse_report(text)
    assert cfg.job_name == "old" and unknown == ["retired_option"] and "steps" in missing
    assert TrainConfig.load_report(path)[1:] == (missing, unknown)


def test_parse_report_refuses_what_is_not_a_config():
    with pytest.raises(ValueError, match="notes.yaml is not a RulzTrainer config"):
        TrainConfig.parse_report("- a\n- b\n", "notes.yaml")


@pytest.mark.parametrize(("job", "name"), [("my_lora", "my_lora.yaml"), ('a/b:c*"d', "a_b_c__d.yaml"),
                                           ("  ..", "config.yaml"), ("", "config.yaml")])
def test_config_file_name(job, name):
    assert webui.config_file_name(job) == name


class _DatasetBox(Box):
    selected: list = []


def _filled_form(cfg: TrainConfig) -> webui.Form:
    """A form with a widget for every value from_config fills, filled from `cfg`."""
    keys = set(cfg.to_dict()) | {"model_preset", "soup_run", "noise_run", "race_paired", "sample_aspect", "sample_mp",
                                 "turbo_mode", "sample_lora_mode", "adapter_mode"}
    keys |= {f"target_{t}" for t in webui.TARGETS + list(webui.EXTRA_TARGETS)}
    f = webui.Form()
    f.w = {k: Box() for k in keys}
    f.w["datasets"] = _DatasetBox()
    f.set_quant_options = lambda *a: None
    f.from_config(cfg)
    return f


def test_loaded_settings_without_a_field_survive_to_config():
    """weight_decay, lora_dropout, the hull_* tuning, cache_dir...: no field on the Train page, so a loaded config's
    values are kept (Load then Start or Save), not replaced by the defaults; model_arch goes back to auto, the family
    following the model picked on the page. Each dataset keeps its own trigger, class and kind."""
    loaded = TrainConfig(weight_decay=0.0, lora_dropout=0.1, hull_clip=3.0, cache_dir="mycache", model_arch="krea2",
                         datasets=[{"path": "a", "trigger": "ohwx", "class": "man", "kind": "person"}])
    f = _filled_form(loaded)
    assert f.w["datasets"].value[0] == {"path": "a", "resolutions": [], "repeats": 1, "trigger": "ohwx",
                                        "class": "man", "kind": "person"}
    f.w["datasets"].value = loaded.datasets  # (the picker's own getter rebuilds these from its rows)
    out = f.to_config()
    assert (out.weight_decay, out.lora_dropout, out.hull_clip, out.cache_dir) == (0.0, 0.1, 3.0, "mycache")
    assert out.model_arch == "auto" and out.datasets[0]["trigger"] == "ohwx"
    assert webui.Form().w == {}  # (a fresh page without a loaded config: the defaults, as before)
