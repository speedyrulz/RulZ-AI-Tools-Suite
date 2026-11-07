"""The WebUI's hold on the training job: once its own job has ended it picks up a newer one another WebUI on this
install started (JOB_FILE), so its queue never starts a second job on the same GPU."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

psutil = pytest.importorskip("psutil")

import webui  # noqa: E402


def test_a_newer_job_from_another_webui_is_picked_up(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "JOB_FILE", tmp_path / "job.json")
    job = tmp_path / "job"
    job.mkdir()
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    monkeypatch.setattr(webui, "_proc", ended)  # this WebUI's job, finished
    monkeypatch.setattr(webui, "_adopted", None)
    webui._record_job(ended.pid, job)
    assert webui.reattach_job() is False and webui._proc is ended  # the last job, ended: kept for the Monitor
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        webui._record_job(other.pid, job)  # the other WebUI's queue started the next run
        assert webui.reattach_job() is True and webui._running()
        assert webui._proc is None and webui._adopted.pid == other.pid
    finally:
        other.kill()
        other.wait()
