import json

from rulztrainer.config import TrainConfig
from rulztrainer.savemeta import kohya_training_keys, public_config
from rulztrainer.trainer import LearningRateReader, LR_FLOOR_WARN_STEPS, lr_at_floor


def test_public_config_removes_private_fields_and_nested_paths(tmp_path):
    cfg = TrainConfig()
    for name, value in vars(cfg).items():
        if isinstance(value, str):
            setattr(cfg, name, str(tmp_path / "private"))
    cfg.datasets = [{"path": str(tmp_path), "resolutions": [0.5], "repeats": 3,
                     "kind": "person", "trigger": "private", "class": "private"}]
    cfg.sample_prompts = ["private prompt"]
    cfg.race_configs = [{"nested": [str(tmp_path / "secret")], "url": "https://example.test/token"}]
    result = public_config(cfg)
    assert tmp_path.name not in json.dumps(result)
    assert result["datasets"] == [{"resolutions": [0.5], "repeats": 3, "kind": "person"}]
    assert "sample_prompts" not in result
    assert "url" not in result["race_configs"][0]


def test_public_config_preserves_repo_ids():
    cfg = TrainConfig()
    cfg.job_name = "org/name"
    assert public_config(cfg)["job_name"] == "org/name"


def test_kohya_fields_are_strings_and_use_effective_optimizer():
    cfg = TrainConfig()
    cfg.optimizer = "automagic3"
    cfg.isolation = True
    meta = kohya_training_keys(cfg, epoch=2, num_train_images=10)
    assert meta["ss_optimizer"] == "adamw"
    assert meta["ss_epoch"] == "2" and meta["ss_num_train_images"] == "10"
    assert all(isinstance(v, str) for v in meta.values())
    assert "ss_num_reg_images" not in meta


class Adaptive:
    def __init__(self):
        self.calls = 0
        self.param_groups = [{"min_lr": 0.01}]

    def get_avg_learning_rate(self):
        self.calls += 1
        return 0.01


class Scheduler:
    def __init__(self):
        self.calls = 0

    def get_last_lr(self):
        self.calls += 1
        return [0.1]


def test_lr_reader_throttles_and_floor_counts_steps():
    opt, sched, reader = Adaptive(), Scheduler(), LearningRateReader()
    floor_steps = 0
    for step in range(100):
        lr = reader.read(opt, sched, step)
        floor_steps = floor_steps + 1 if lr_at_floor(opt, lr) else 0
    assert opt.calls == 5
    assert floor_steps == LR_FLOOR_WARN_STEPS
    reader.read(opt, sched, 100, force=True)
    assert opt.calls == 6


def test_lr_reader_reads_ordinary_optimizer_every_step():
    sched, reader = Scheduler(), LearningRateReader()
    opt = object()
    for step in range(100):
        assert reader.read(opt, sched, step) == 0.1
    assert sched.calls == 100
