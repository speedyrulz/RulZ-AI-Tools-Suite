"""Regression tests for the 2026-11-20 WebUI audit (B5-B12): a run that carries on its job folder keeps its first
segment's logs, a page load cannot rewind the job folder while a run starts, the history import runs once and off
the page, the look scan drops results for another dataset, the dataset list is written whole, download temp files
are swept, and the console tail is read from the end."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402
from rulztrainer import history  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


class FakeProc:
    pid = 4242

    def poll(self):
        return None


def _cfg(tmp_path, name="job", resume=""):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return TrainConfig(job_name=name, output_dir=str(tmp_path / "out"), dataset_dir=str(data), resume_state=resume)


@pytest.fixture
def quiet_webui(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "JOB_FILE", tmp_path / "job.json")
    monkeypatch.setattr(webui, "_proc", None)
    monkeypatch.setattr(webui, "_adopted", None)
    monkeypatch.setattr(webui, "_job_dir", None)
    return tmp_path


# ---------------------------------------------------------------- B5


def test_b5_a_run_carrying_on_its_folder_keeps_the_first_segments_logs(quiet_webui, monkeypatch):
    tmp = quiet_webui
    job = tmp / "out" / "job"
    state = job / "state_000010"
    state.mkdir(parents=True)
    (state / "train_state.pt").write_bytes(b"x")
    (state / "lora.safetensors").write_bytes(b"x")
    (job / "train.log").write_text("first segment line\n", encoding="utf-8")
    (job / "console.log").write_text("first console\n", encoding="utf-8")
    (job / "status.json").write_text('{"step": 10}', encoding="utf-8")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProc())
    msg = webui._start_training(_cfg(tmp, resume=str(state)))
    assert msg.startswith("Started job"), msg
    assert (job / "train_segment1.log").read_text(encoding="utf-8") == "first segment line\n"
    assert (job / "console_segment1.log").read_text(encoding="utf-8") == "first console\n"
    assert json.loads((job / "status_segment1.json").read_text(encoding="utf-8")) == {"step": 10}
    assert (job / "train.log").read_text(encoding="utf-8") == ""  # the Monitor's files start clean
    assert not (job / "status.json").exists()
    # a third segment does not overwrite the first rotation
    (job / "train.log").write_text("second segment\n", encoding="utf-8")
    webui.keep_earlier_logs(job)
    assert (job / "train_segment1.log").read_text(encoding="utf-8") == "first segment line\n"
    assert (job / "train_segment2.log").read_text(encoding="utf-8") == "second segment\n"


# ---------------------------------------------------------------- B6


def test_b6_a_page_load_during_a_start_does_not_rewind_the_job_folder(quiet_webui, monkeypatch):
    tmp = quiet_webui
    old = tmp / "out" / "old"
    old.mkdir(parents=True)
    webui.JOB_FILE.write_text(json.dumps({"pid": 1, "create_time": None, "job_dir": str(old)}), encoding="utf-8")
    inside, release = threading.Event(), threading.Event()

    def slow_popen(*a, **k):
        inside.set()
        release.wait(10)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", slow_popen)
    result = {}
    t = threading.Thread(target=lambda: result.update(msg=webui.start_training(_cfg(tmp, name="new"))))
    t.start()
    assert inside.wait(10)
    seen = {}
    r = threading.Thread(target=lambda: (webui.reattach_job(), seen.update(job=webui._job_dir)))
    r.start()
    time.sleep(0.3)
    assert r.is_alive()  # the page waits for the start instead of reading the old JOB_FILE
    release.set()
    t.join(10)
    r.join(10)
    assert result["msg"].startswith("Started job")
    assert webui._job_dir == tmp / "out" / "new"


def test_b6_a_failed_start_leaves_the_job_folder_alone(quiet_webui, monkeypatch):
    tmp = quiet_webui
    earlier = tmp / "out" / "earlier"
    monkeypatch.setattr(webui, "_job_dir", earlier)

    def boom(*a, **k):
        raise OSError("cannot start")

    monkeypatch.setattr(subprocess, "Popen", boom)
    with pytest.raises(OSError):
        webui._start_training(_cfg(tmp, name="never"))
    assert webui._job_dir == earlier


# ---------------------------------------------------------------- B7


def _output_tree(root: Path, jobs: int, files: int) -> Path:
    out = root / "output"
    for i in range(jobs):
        job = out / f"job{i}"
        (job / "samples").mkdir(parents=True)
        for k in range(files):
            (job / "samples" / f"s{k}.png").write_bytes(b"")
        (job / "status.json").write_text(json.dumps({"probes": []}), encoding="utf-8")
        (job / "config.yaml").write_text(f"job_name: job{i}\n", encoding="utf-8")
    return out


def test_b7_status_files_skip_picture_folders_but_find_the_same_jobs(tmp_path):
    out = _output_tree(tmp_path, 3, 5)
    (out / "_history").mkdir()
    (out / "_history" / "status.json").write_text("{}", encoding="utf-8")
    assert history.status_files(out) == sorted(p for p in out.rglob("status.json") if "_history" not in p.parts)


def test_b7_the_import_starts_once_and_off_the_calling_thread(monkeypatch):
    monkeypatch.setattr(webui, "_history_import_started", False)
    ran = []
    gate = threading.Event()

    def importer():
        gate.wait(5)
        ran.append(threading.current_thread())

    t0 = time.perf_counter()
    assert webui.start_history_import(importer) is True
    assert time.perf_counter() - t0 < 1.0 and not ran  # returned while the import is still waiting
    assert webui.start_history_import(importer) is False  # a second page build does not start another
    gate.set()
    for _ in range(100):
        if ran:
            break
        time.sleep(0.05)
    assert len(ran) == 1 and ran[0] is not threading.current_thread()


def test_b7_timing_on_a_synthetic_tree(tmp_path, capsys):
    out = _output_tree(tmp_path, 100, 300)
    t0 = time.perf_counter()
    list(out.rglob("status.json"))  # what the old scan walked (the whole tree)
    old = time.perf_counter() - t0
    t0 = time.perf_counter()
    history.status_files(out)
    new = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\nB7 scan of 100 jobs x 300 sample files: rglob {old * 1000:.0f} ms, status_files {new * 1000:.0f} ms")
    assert new < old


# ---------------------------------------------------------------- B10


def test_b10_the_dataset_list_is_not_wiped_by_a_torn_file(tmp_path):
    root = tmp_path / "datasets"
    webui.register_dataset(str(tmp_path / "a"), root)
    webui.register_dataset(str(tmp_path / "b"), root)
    assert len(webui.registered_datasets(root)) == 2
    assert not list(root.glob("*.tmp"))  # written through a temporary file that is gone
    reg = root / webui.DATASETS_REGISTRY
    reg.write_text('{"external": ["' + str(tmp_path / "a").replace("\\", "\\\\"), encoding="utf-8")  # torn
    with pytest.raises(ValueError):
        webui.register_dataset(str(tmp_path / "c"), root)
    with pytest.raises(ValueError):
        webui.unregister_dataset(str(tmp_path / "a"), root)
    assert reg.read_text(encoding="utf-8").startswith('{"external": ["')  # untouched


def test_b10_a_save_that_fails_keeps_the_old_list(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    webui.register_dataset(str(tmp_path / "a"), root)

    def boom(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", boom)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        webui.register_dataset(str(tmp_path / "b"), root)
    monkeypatch.undo()
    assert webui.registered_datasets(root) == [str(tmp_path / "a")]
    assert not list(root.glob("*.tmp"))


# ---------------------------------------------------------------- B11


def test_b11_old_download_files_are_swept_only_inside_their_folder(tmp_path):
    folder = tmp_path / "rulz_trainer_downloads"
    folder.mkdir()
    old, new = folder / "old.zip", folder / "new.zip"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"x")
    long_ago = time.time() - 3 * 86400
    os.utime(old, (long_ago, long_ago))
    os.utime(outside, (long_ago, long_ago))
    assert webui.sweep_download_tmp(86400, folder) == 1
    assert not old.exists() and new.exists() and outside.exists()


def test_b11_download_files_are_made_in_the_downloads_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "DOWNLOAD_TMP_DIR", tmp_path / "dl")
    p = Path(webui.download_tmp_file(".xlsx"))
    assert p.parent == tmp_path / "dl" and p.suffix == ".xlsx" and p.is_file()


# ---------------------------------------------------------------- B12


def test_b12_the_console_tail_comes_from_the_end_of_the_file(tmp_path):
    p = tmp_path / "console.log"
    p.write_text("\n".join(f"line {i}" for i in range(5000)) + "\n", encoding="utf-8")
    assert webui.tail_lines(p, 30) == [f"line {i}" for i in range(4970, 5000)]
    src = Path(webui.__file__).read_text(encoding="utf-8")
    assert 'cp.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]' not in src
