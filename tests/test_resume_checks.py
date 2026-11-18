import os

import pytest

from rulztrainer.config import TrainConfig
from rulztrainer.runplan import check_state_paths


def state(path):
    path.mkdir(parents=True)
    (path / "train_state.pt").touch()
    (path / "lora.safetensors").touch()
    return path


def test_resume_folder_and_file(tmp_path):
    folder = state(tmp_path / "state_000001")
    cfg = TrainConfig(output_dir=str(tmp_path), job_name="job", resume_state=str(folder))
    check_state_paths(cfg)
    cfg.resume_state = str(folder / "train_state.pt")
    check_state_paths(cfg)


def test_resume_parent_suggests_newest(tmp_path):
    old = state(tmp_path / "state_000001")
    new = state(tmp_path / "state_000002")
    os.utime(old, (1, 1))
    cfg = TrainConfig(output_dir=str(tmp_path), job_name="job", resume_state=str(tmp_path))
    with pytest.raises(ValueError, match="newest") as error:
        check_state_paths(cfg)
    assert str(new) in str(error.value)


def test_incomplete_and_output_state(tmp_path):
    folder = state(tmp_path / "state_000001")
    cfg = TrainConfig(output_dir=str(tmp_path), job_name=folder.name)
    with pytest.raises(ValueError, match="Output folder"):
        check_state_paths(cfg)
    (folder / "lora.safetensors").unlink()
    cfg.job_name = "job"
    cfg.resume_state = str(folder)
    with pytest.raises(ValueError, match="not a saved training state"):
        cfg.validate()


def test_a_state_not_written_yet_passes_validate_but_not_start(tmp_path):
    later = tmp_path / "jobA" / "state_001000"  # a run ahead in the queue writes it
    cfg = TrainConfig(output_dir=str(tmp_path), job_name="jobB", resume_state=str(later))
    check_state_paths(cfg, must_exist=False)
    with pytest.raises(ValueError, match="not a saved training state"):
        check_state_paths(cfg)


def test_newest_is_the_highest_step(tmp_path):
    new = state(tmp_path / "state_000200")
    old = state(tmp_path / "state_000100")
    os.utime(new, (1, 1))  # copied back: an older date than the earlier step
    cfg = TrainConfig(output_dir=str(tmp_path), job_name="job", resume_state=str(tmp_path))
    with pytest.raises(ValueError) as error:
        check_state_paths(cfg)
    assert str(new) in str(error.value) and str(old) not in str(error.value)
