"""Configs saved by an older version: lenient load, new options appended to the file, Browse start folder."""
import dataclasses
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer.config import TrainConfig  # noqa: E402

OLD = """# my character job
job_name: old_job
learning_rate: 0.0002   # tuned by hand
multi_gpu: data_parallel
lora_rank: 32
removed_option: 5
"""


def test_load_report_lists_new_and_unknown_options(tmp_path):
    p = tmp_path / "old.yaml"
    p.write_text(OLD, encoding="utf-8")
    cfg, missing, unknown = TrainConfig.load_report(p)
    assert cfg.job_name == "old_job" and cfg.learning_rate == 0.0002 and cfg.lora_rank == 32
    assert unknown == ["removed_option"]
    known = [f.name for f in dataclasses.fields(TrainConfig)]
    assert missing == [k for k in known if k not in ("job_name", "learning_rate", "multi_gpu", "lora_rank")]
    assert "parallel_jobs" in missing and cfg.parallel_jobs is True  # new options take their defaults
    with pytest.raises(ValueError, match="removed_option"):
        TrainConfig.load(p)  # the strict CLI load still refuses unknown keys


def test_append_missing_keeps_the_file_and_adds_every_new_option(tmp_path):
    p = tmp_path / "old.yaml"
    p.write_text(OLD.rstrip("\n"), encoding="utf-8")  # no trailing newline
    cfg, missing, _ = TrainConfig.load_report(p)
    TrainConfig.append_missing(p, cfg, missing)
    text = p.read_text(encoding="utf-8")
    assert text.startswith(OLD.rstrip("\n") + "\n# Options added in a newer RulzTrainer version")
    assert "# tuned by hand" in text  # comments and the user's values untouched
    data = yaml.safe_load(text)
    assert data["learning_rate"] == 0.0002 and data["removed_option"] == 5
    cfg2, missing2, unknown2 = TrainConfig.load_report(p)
    assert missing2 == [] and unknown2 == ["removed_option"]
    assert cfg2 == cfg  # same configuration before and after the upgrade
    TrainConfig.append_missing(p, cfg2, missing2)  # nothing missing: the file is left alone
    assert p.read_text(encoding="utf-8") == text


def test_load_report_rejects_non_mapping(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a RulzTrainer config"):
        TrainConfig.load_report(p)


def test_dialog_start_dir_and_repo_example():
    pytest.importorskip("nicegui")
    import webui

    root = webui.ROOT
    (root / "configs").mkdir(exist_ok=True)
    assert webui.dialog_start_dir("configs/does_not_exist_yet.yaml") == str(root / "configs")
    assert webui.dialog_start_dir("configs") == str(root / "configs")
    assert webui.dialog_start_dir("") == str(root)
    assert webui.dialog_start_dir("no_such_dir/x/y.yaml") == str(root)
    assert webui.dialog_start_dir(str(root / "configs" / "example.yaml")) == str(root / "configs")
    assert webui._is_repo_example("configs/example.yaml")
    assert not webui._is_repo_example("configs/test.yaml")
