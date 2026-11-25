import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "vast"))
sys.path.insert(0, str(ROOT))
from rulz_plan import NOT_PATHS, PATH_KEYS, apply, plan_send
from rulztrainer.config import TrainConfig


def test_path_field_classification():
    suffixes = ("_file", "_path", "_dir", "_files", "_lora", "_model", "_state")
    fields = {f.name for f in dataclasses.fields(TrainConfig) if f.name.endswith(suffixes)}
    assert fields <= PATH_KEYS | NOT_PATHS
    assert not PATH_KEYS & NOT_PATHS


def test_nested_race_refs_and_changes(tmp_path):
    dataset = tmp_path / "données with spaces"
    inner = dataset / "inner"
    inner.mkdir(parents=True)
    (inner / "photo.jpg").write_bytes(b"abc")
    (dataset / ".hidden").write_bytes(b"hidden")
    outside = tmp_path / "other" / "face.jpg"
    outside.parent.mkdir()
    outside.write_bytes(b"face")
    state = tmp_path / "state_1"
    state.mkdir()
    (state / "state.bin").write_bytes(b"state")
    config = {"job_name": "élan", "dataset_dir": str(dataset), "face_refs": [str(inner / "photo.jpg"), str(outside)],
              "resume_state": str(state), "output_dir": "D:/out", "models_dir": "D:/models", "cache_dir": "D:/cache",
              "devices": "cuda:0,cuda:3", "multi_gpu": "data_parallel",
              "race_configs": [{"name": "one", "dataset_dir": str(inner), "body_refs": [str(outside)]}]}
    plan = plan_send(config, tmp_path)
    assert not plan.blockers
    assert len(plan.datasets) == 2
    assert plan.datasets[0]["files"] == 1
    assert plan.datasets[1]["kind"] == "files"
    assert len(plan.datasets[1]["files"]) == 1
    assert plan.total_bytes == 12
    remote = "models/sent/0123456789ab/state_1"
    result = apply(plan, {str(state): {"how": "upload", "remote": remote}})
    assert result["race_configs"][0]["dataset_dir"] == result["dataset_dir"] + "/inner"
    assert result["face_refs"][0] == result["dataset_dir"] + "/inner/photo.jpg"
    assert result["face_refs"][1] == result["race_configs"][0]["body_refs"][0]
    assert result["resume_state"] == remote
    assert result["devices"] == "cuda:0,cuda:0"
    assert result["multi_gpu"] == "off" and result["cache_dir"] == ""
    assert config["resume_state"] == str(state)


def test_relative_paths_names_and_case(tmp_path):
    for parent, name in [("A", "Data"), ("B", "data")]:
        folder = tmp_path / parent / name
        folder.mkdir(parents=True)
        (folder / "x").write_text("x", encoding="utf-8")
    cfg = {"datasets": [{"path": "A/Data"}, {"path": "B/data"}, {"path": "a/data"}], "model_path": "Org/Name"}
    # Differently cased aliases exist on Windows; on other systems use the real case.
    if sys.platform != "win32":
        cfg["datasets"][2]["path"] = "A/Data"
    plan = plan_send(cfg, tmp_path)
    assert not plan.blockers
    assert len(plan.datasets) == 2
    assert len({d["name"].casefold() for d in plan.datasets}) == 2
    out = apply(plan, {})
    assert out["datasets"][0]["path"] == out["datasets"][2]["path"]
    assert out["model_path"] == "Org/Name"


def test_missing_and_invalid_choice(tmp_path):
    plan = plan_send({"dataset_dir": "missing folder"}, tmp_path)
    assert plan.blockers
    with pytest.raises(ValueError):
        apply(plan, {})
    file = tmp_path / "custom.safetensors"
    file.write_bytes(b"x")
    plan = plan_send({"transformer_file": str(file)}, tmp_path)
    assert plan.files[0]["standard"] is None
    with pytest.raises(ValueError):
        apply(plan, {})
    with pytest.raises(ValueError):
        apply(plan, {str(file): {"how": "bucket", "remote": "models/sent/../../escape"}})


def test_standard_download(tmp_path):
    from rulztrainer.config import COMFY_DIFFUSION_FILES, COMFY_DEFAULT_QUANT, COMFY_QUANT_NAMES
    model = TrainConfig().model_path
    pattern = COMFY_DIFFUSION_FILES[model]
    quant = COMFY_DEFAULT_QUANT.get("moss_image", "int8_convrot")
    quant = COMFY_QUANT_NAMES.get("moss_image", {}).get(quant, quant)
    filename = pattern.format(quant=quant)
    path = tmp_path / Path(filename).name
    path.write_bytes(b"model")
    plan = plan_send({"model_path": model, "model_arch": "moss_image", "transformer_file": str(path)}, tmp_path)
    assert plan.files[0]["standard"]["file"] == filename
    assert apply(plan, {})["transformer_file"] == ""
