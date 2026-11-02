"""The WebUI's job-folder downloads (Monitor and Previous Runs): the LoRA list, the logs and settings, the zip route,
which history run a job folder still belongs to, and the race chart's score filter."""
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from nicegui import app  # noqa: E402

import webui  # noqa: E402


def _job(tmp_path, run_id="job-1") -> Path:
    d = tmp_path / "out" / "job"
    (d / "race").mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (d / "config.yaml").write_text("job_name: job\n", encoding="utf-8")
    (d / "train.log").write_text("log\n", encoding="utf-8")
    (d / "race" / "race.json").write_text("{}", encoding="utf-8")
    (d / "samples").mkdir()
    (d / "samples" / "step000100_0.png").write_bytes(b"png")
    (d / "samples" / "lokr8_step000100_1.png").write_bytes(b"png")
    (d / "samples" / "grid.psd").write_bytes(b"not an image")
    (d / "probes").mkdir()
    (d / "probes" / "step000100_face_0.png").write_bytes(b"png")
    (d / "state_000100").mkdir()
    (d / "state_000100" / "optimizer.pt").write_bytes(b"x")
    now = time.time()
    for i, name in enumerate(("job_step000100.safetensors", "job_step000200.safetensors", "job_best.safetensors")):
        (d / name).write_bytes(b"w" * (i + 1))
        os.utime(d / name, (now + i, now + i))
    return d


def test_job_files(tmp_path):
    d = _job(tmp_path)
    assert [p.name for p, _ in webui.job_loras(d)] == ["job_best.safetensors", "job_step000200.safetensors",
                                                       "job_step000100.safetensors"]  # newest first
    assert webui.job_loras(None) == []
    logs = [p.relative_to(d).as_posix() for p in webui.job_logs(d)]
    assert logs == ["config.yaml", "race/race.json", "status.json", "train.log"]  # no images, states or LoRAs
    assert webui.size_text(5 * 2**20) == "5 MB" and webui.size_text(3 * 2**30) == "3.0 GB"
    assert [p.name for p in webui.job_images(d, "samples")] == ["lokr8_step000100_1.png", "step000100_0.png"]
    assert [p.name for p in webui.job_images(d, "probes")] == ["step000100_face_0.png"]
    assert webui.job_images(d, "race") == [] and webui.job_images(None, "probes") == []


def test_job_zip_route(tmp_path):
    d = _job(tmp_path)
    client = TestClient(app)
    r = client.get("/rz-job-zip", params={"job": str(d), "kind": "loras"})
    assert r.status_code == 200
    assert sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) == [
        "job_best.safetensors", "job_step000100.safetensors", "job_step000200.safetensors"]
    r = client.get("/rz-job-zip", params={"job": str(d), "kind": "logs"})
    assert sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) == [
        "config.yaml", "race/race.json", "status.json", "train.log"]
    for kind, names in (("samples", ["samples/lokr8_step000100_1.png", "samples/step000100_0.png"]),
                        ("probes", ["probes/step000100_face_0.png"])):
        r = client.get("/rz-job-zip", params={"job": str(d), "kind": kind})
        assert r.status_code == 200 and sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) == names
    for f in (d / "probes").iterdir():
        f.unlink()
    assert client.get("/rz-job-zip", params={"job": str(d), "kind": "probes"}).status_code == 404  # none yet
    assert client.get("/rz-job-zip", params={"job": str(d), "kind": "states"}).status_code == 404
    assert client.get("/rz-job-zip", params={"job": str(tmp_path), "kind": "logs"}).status_code == 404  # no job


def test_history_job_dir(tmp_path):
    d = _job(tmp_path, run_id="job-2")
    mine = {"id": "job-2", "job": "job", "job_dir": str(d), "started": 2}
    older = {"id": "job-1", "job": "job", "job_dir": str(d), "started": 1}
    assert webui.history_job_dir(mine, [older, mine]) == d
    assert webui.history_job_dir(older, [older, mine]) is None  # the folder holds the later run now
    (d / "status.json").write_text("{}", encoding="utf-8")  # a run from before status.json named it
    assert webui.history_job_dir(mine, [older, mine]) == d
    assert webui.history_job_dir(older, [older, mine]) is None
    assert webui.history_job_dir({**mine, "job_dir": str(tmp_path / "gone")}, [mine]) is None


def test_race_chart_score_filter():
    probes = [{"step": s, "images": [f"probes/{n}_step{s:06d}_face_0.png"], "face_norm_ema": 50.0,
               "body_norm_ema": 40.0, "combined_ema": 47.0} for s in (100, 200) for n in ("a", "b")]
    _, legend = webui.probe_chart_series(probes, ["a", "b"])
    assert legend == ["a combined", "a face", "a body", "b combined", "b face", "b body"]
    _, legend = webui.probe_chart_series(probes, ["a", "b"], ["combined", "body"])
    assert legend == ["a combined", "a body", "b combined", "b body"]
