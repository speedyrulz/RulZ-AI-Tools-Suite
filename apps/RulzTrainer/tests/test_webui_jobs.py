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


def test_a_job_recorded_without_its_create_time_is_not_adopted(tmp_path, monkeypatch):
    """The job ended before it was recorded: its pid may belong to another process by now, never the job."""
    monkeypatch.setattr(webui, "JOB_FILE", tmp_path / "job.json")
    monkeypatch.setattr(webui, "_proc", None)
    monkeypatch.setattr(webui, "_adopted", None)
    job = tmp_path / "job"
    job.mkdir()
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        webui.JOB_FILE.write_text(__import__("json").dumps(
            {"pid": stranger.pid, "create_time": None, "job_dir": str(job)}), encoding="utf-8")
        assert webui.reattach_job() is False and webui._adopted is None
        assert webui._job_dir == job  # (the folder is still shown on the Monitor)
    finally:
        stranger.kill()
        stranger.wait()


def test_kill_takes_the_processes_the_job_started_with_it(tmp_path, monkeypatch):
    """A probe process (probe_async) is the job's child: a kill of the job alone would leave it on its GPU."""
    child_pid = tmp_path / "child.pid"
    code = ("import subprocess, sys, time; "
            "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            f"open({str(child_pid)!r}, 'w').write(str(c.pid)); time.sleep(60)")
    job = subprocess.Popen([sys.executable, "-c", code])
    try:
        for _ in range(200):
            if child_pid.exists() and child_pid.read_text():
                break
            __import__("time").sleep(0.05)
        child = psutil.Process(int(child_pid.read_text()))
        webui._kill_tree(job)
        job.wait(timeout=10)
        child.wait(timeout=10)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        job.kill()
        job.wait()
