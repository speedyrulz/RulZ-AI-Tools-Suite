import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rulz_optimizer import settings


@pytest.fixture(autouse=True)
def cfg_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RULZ_OPT_CONFIG_DIR", str(tmp_path / "cfg"))
    settings.reset()
    yield tmp_path / "cfg"
    settings.reset()


def test_defaults():
    s = settings.get()
    assert s.advisor == "claude" and s.port == 8775 and s.trainer_dir == ""
    assert s.effective_export_dir().name == "RulZ AI Trainer Optimizer"


def test_export_dir_follows_trainer_dir(tmp_path):
    s = settings.get()
    s.trainer_dir = str(tmp_path / "trainer")
    assert s.effective_export_dir() == tmp_path / "trainer" / "configs"
    s.export_dir = str(tmp_path / "out")
    assert s.effective_export_dir() == tmp_path / "out"


def test_save_and_reload(cfg_dir):
    s = settings.get()
    s.trainer_dir = "C:/trainer"
    s.advisor = "codex"
    s.advisor_model = {"codex": "gpt-x"}
    settings.remember("model", "krea2")
    settings.reset()
    again = settings.get()
    assert again.trainer_dir == "C:/trainer" and again.advisor == "codex"
    assert again.model_for("codex") == "gpt-x" and again.model_for("claude") == ""
    assert again.last == {"model": "krea2"}
    assert not list(cfg_dir.glob("*.tmp"))  # written atomically


def test_wrong_types_and_bad_advisor_fall_back(cfg_dir):
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "settings.json").write_text(json.dumps({"port": "x", "advisor": "bogus", "run_roots": "no",
                                                       "trainer_dir": "T", "unknown": 1}), encoding="utf-8")
    s = settings.get()
    assert s.port == 8775 and s.advisor == "claude" and s.run_roots == [] and s.trainer_dir == "T"


def test_damaged_file_is_kept_aside(cfg_dir):
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "settings.json").write_text("{not json", encoding="utf-8")
    assert settings.get().advisor == "claude"
    assert list(cfg_dir.glob("settings.json.broken-*"))
    settings.save()
    assert json.loads((cfg_dir / "settings.json").read_text(encoding="utf-8"))["advisor"] == "claude"
