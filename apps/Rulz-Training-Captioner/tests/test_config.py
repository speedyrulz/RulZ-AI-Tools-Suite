"""Settings and saved prompts: bad files are set aside instead of overwritten or crashing the app."""
import json

import pytest

from captioner import config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "PROMPTS_FILE", tmp_path / "prompts.json")
    monkeypatch.setattr(config, "_protected", set())
    monkeypatch.setattr(config, "_copied", set())
    return tmp_path


def _bad_copies(folder, stem):
    return sorted(p.name for p in folder.glob(f"{stem}.bad-*.json"))


def test_first_run_saves_the_default_prompts(cfg):
    p = config.load_prompts()
    assert [x["name"] for x in p] == [x["name"] for x in config.DEFAULT_PROMPTS]
    assert json.loads((cfg / "prompts.json").read_text(encoding="utf-8")) == p


def test_corrupt_prompts_file_is_set_aside_not_overwritten(cfg):
    mine = '[{"name": "Mine", "system": "my long prompt", "user": "go"},'  # cut short: not valid JSON
    (cfg / "prompts.json").write_text(mine, encoding="utf-8")
    p = config.load_prompts()
    assert p[0]["name"] == config.DEFAULT_PROMPTS[0]["name"]
    bad = _bad_copies(cfg, "prompts")
    assert len(bad) == 1 and (cfg / bad[0]).read_text(encoding="utf-8") == mine  # the user's text survives


def test_prompts_keep_the_good_entries_and_a_copy_of_the_rest(cfg):
    original = json.dumps([
        {"name": "Mine", "system": "s"},           # no "user": taken as empty
        {"name": 3, "system": "s", "user": "u"},   # not a prompt
        "junk",
    ])
    (cfg / "prompts.json").write_text(original, encoding="utf-8")
    good = [{"name": "Mine", "system": "s", "user": ""}]
    assert config.load_prompts() == good
    bad = _bad_copies(cfg, "prompts")
    assert len(bad) == 1 and (cfg / bad[0]).read_text(encoding="utf-8") == original
    assert json.loads((cfg / "prompts.json").read_text(encoding="utf-8")) == good
    assert config.load_prompts() == good and len(_bad_copies(cfg, "prompts")) == 1  # not a copy per start


def test_prompts_file_with_a_bom_still_loads(cfg):
    (cfg / "prompts.json").write_text('[{"name": "Mine", "system": "s", "user": "u"}]', encoding="utf-8-sig")
    assert config.load_prompts()[0]["name"] == "Mine"
    assert _bad_copies(cfg, "prompts") == []


def test_settings_of_the_wrong_type_fall_back_to_the_defaults(cfg):
    (cfg / "settings.json").write_text(json.dumps({
        "thumb_size": "large", "gpus": [0, "1"], "skip_existing": 1, "temperature": 1,
        "max_tokens": 600, "model": "C:\\m.gguf", "future_option": {"x": 1},
    }), encoding="utf-8")
    s = config.load_settings()
    assert s["thumb_size"] == config.DEFAULT_SETTINGS["thumb_size"]
    assert s["gpus"] == [] and s["skip_existing"] is True
    assert s["temperature"] == 1 and s["max_tokens"] == 600 and s["model"] == "C:\\m.gguf"
    assert s["future_option"] == {"x": 1}  # unknown keys are kept for newer versions
    assert len(_bad_copies(cfg, "settings")) == 1  # the dropped values survive in a copy


def test_set_aside_never_overwrites_an_earlier_copy(cfg):
    for text in ("first broken", "second broken", "third broken"):  # all within one second
        (cfg / "settings.json").write_text(text, encoding="utf-8")
        config.load_settings()
    kept = sorted((cfg / name).read_text(encoding="utf-8") for name in _bad_copies(cfg, "settings"))
    assert kept == ["first broken", "second broken", "third broken"]


def test_wrong_types_are_copied_aside_once_per_run(cfg):
    (cfg / "settings.json").write_text('{"thumb_size": "large"}', encoding="utf-8")
    config.load_settings()
    config.load_settings()  # main.py and the window both load the settings
    assert len(_bad_copies(cfg, "settings")) == 1


def test_a_locked_settings_file_is_left_alone(cfg, monkeypatch):
    (cfg / "settings.json").write_text('{"model": "C:\\\\keep.gguf"}', encoding="utf-8")
    real_open = open

    def locked_open(path, *a, **k):  # a virus scanner or a backup holds it: it can't be read
        if str(path).endswith("settings.json"):
            raise PermissionError(13, "in use")
        return real_open(path, *a, **k)
    monkeypatch.setattr("builtins.open", locked_open)
    assert config.load_settings() == config.DEFAULT_SETTINGS
    assert _bad_copies(cfg, "settings") == []  # likely fine: not renamed as if it were broken
    monkeypatch.setattr("builtins.open", real_open)
    assert config.save_settings(dict(config.DEFAULT_SETTINGS)) is False  # nor saved over this run
    assert "keep.gguf" in (cfg / "settings.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("text", ["null", "[1, 2]", '{"thumb_size": 1'])
def test_settings_that_arent_settings_are_set_aside(cfg, text):
    (cfg / "settings.json").write_text(text, encoding="utf-8")
    assert config.load_settings() == config.DEFAULT_SETTINGS  # used to stop the app from starting
    assert len(_bad_copies(cfg, "settings")) == 1
    assert not (cfg / "settings.json").exists()


def test_gpus_are_asked_once_but_an_empty_answer_is_asked_again(monkeypatch):
    calls = []
    answers = [[], [(0, "RTX", 16384)]]
    monkeypatch.setattr(config, "_gpus", None)
    monkeypatch.setattr(config, "_query_gpus", lambda: calls.append(1) or answers.pop(0))
    assert config.detect_gpus() == []
    assert config.detect_gpus() == [(0, "RTX", 16384)]
    assert config.detect_gpus() == [(0, "RTX", 16384)]
    assert len(calls) == 2
