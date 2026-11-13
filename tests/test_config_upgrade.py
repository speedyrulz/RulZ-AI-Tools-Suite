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


def test_lora_targets_default_per_family(tmp_path):
    """An empty lora_targets (new configs, and old ones without the key) means the family's default groups; a config
    that lists its targets keeps them."""
    from rulztrainer.config import ARCH_DEFAULT_TARGETS

    path = tmp_path / "old.yaml"
    path.write_text(OLD + "model_path: krea/Krea-2-Raw\n", encoding="utf-8")
    cfg, missing, _ = TrainConfig.load_report(path)
    assert "lora_targets" in missing and cfg.lora_targets == []
    assert cfg.resolved_lora_targets() == list(ARCH_DEFAULT_TARGETS["krea2"])
    path.write_text(OLD + "model_path: krea/Krea-2-Raw\nlora_targets: [q, k]\n", encoding="utf-8")
    assert TrainConfig.load_report(path)[0].resolved_lora_targets() == ["q", "k"]
    assert TrainConfig().resolved_lora_targets() == ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]


def test_form_follows_the_family_default_targets():
    pytest.importorskip("nicegui")
    import webui

    class Widget:
        def __init__(self, value):
            self.value = value

    form = webui.Form()
    form.w.update(model_preset=Widget("MOSS-Image-Base"), model_path=Widget(""))
    for t in webui.TARGETS + list(webui.EXTRA_TARGETS):
        form.w[f"target_{t}"] = Widget(t in webui.ARCH_DEFAULT_TARGETS["moss_image"])
    form.w["model_preset"].value = "Krea-2-Raw"
    form.follow_family_targets()
    assert form.ticked_targets() == set(webui.ARCH_DEFAULT_TARGETS["krea2"])
    form.w["target_text_fusion"].value = False  # the user's own choice is kept on the next model change
    form.w["model_preset"].value = "MOSS-Image-Base"
    form.follow_family_targets()
    assert "text_fusion" not in form.ticked_targets() and "attn_gate" in form.ticked_targets()


def test_compile_auto_rule():
    from rulztrainer import compiling

    assert TrainConfig().compile_model == "auto"
    TrainConfig(compile_model=True).validate()
    with pytest.raises(ValueError):
        TrainConfig(compile_model="sometimes").validate()
    assert compiling.decide(True, 5, ["cpu"])[0] and not compiling.decide(False, 5000, ["cuda:0"])[0]
    assert compiling.decide(False, 5000, ["cuda:0"])[1] == ""  # simply off: nothing to log
    on, why = compiling.decide("auto", 20, ["cuda:0"])
    assert not on and ("20 steps" in why or "triton" in why)  # a short test run is not worth the warm-up
    assert not compiling.decide("auto", 5000, ["cpu"])[0]
    pytest.importorskip("triton")
    assert compiling.decide("auto", 5000, ["cuda:1", "cuda:0"]) == (True, "compile_model auto: 5000 steps to go")


@pytest.mark.parametrize("bad, msg", [
    (dict(gradient_accumulation=0), "gradient_accumulation"), (dict(batch_size=0), "batch_size"),
    (dict(caption_dropout=5.0), "caption_dropout"), (dict(lr_scheduler="cosin"), "lr_scheduler"),
    (dict(timestep_sampling="logit"), "timestep_sampling"), (dict(loss_weighting="snr"), "loss_weighting")])
def test_settings_the_trainer_cannot_act_on_are_refused_at_start(bad, msg):
    """Refused before the caching and the model load: gradient_accumulation 0 trained no image and saved the LoRA
    untrained, an lr_scheduler typo ran constant, a timestep_sampling typo failed at step 1."""
    with pytest.raises(ValueError, match=msg):
        TrainConfig(**bad).validate()


def test_hand_written_switches_read_as_meant():
    """YAML 1.1 reads a bare off as false and a quoted 'false' stays a (true) string."""
    import yaml

    from rulztrainer.config import TrainConfig as TC

    cfg = TC.from_dict(yaml.safe_load("dp_balance: off\nhull_regions: off\nmulti_gpu: off\nflip_augment: 'false'\n"
                                      "recache: 'yes'\n"))
    assert (cfg.dp_balance, cfg.hull_regions, cfg.multi_gpu) == ("off", "off", "off")
    assert cfg.flip_augment is False and cfg.recache is True
    cfg.validate()


def test_multi_gpu_takes_only_its_modes():
    from rulztrainer.config import TrainConfig as TC

    with pytest.raises(ValueError, match="multi_gpu"):
        TC.from_dict({"multi_gpu": True}).validate()  # a bare on: which mode is meant?
    TC(multi_gpu="data_parallel").validate()


def test_a_misspelt_save_dtype_is_refused_before_the_run():
    """Not at the first save, which may be the last one after hours of training."""
    import pytest as _pytest

    from rulztrainer.config import TrainConfig as TC

    with _pytest.raises(ValueError, match="lora_save_dtype"):
        TC(lora_save_dtype="bfloat16").validate()
