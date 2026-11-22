"""Second review round on the save folders: a pull keeps a run in its own folder, the in-process guard, the running
check inside apply, roots that cannot be resolved, and the path a resume returns."""
import os
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))

import rulz_remote as remote  # noqa: E402
from rulz_fleet import Fleet  # noqa: E402
from test_save_folders import add_run, default_history, fill  # noqa: E402,F401

from rulztrainer import history, storage  # noqa: E402


@pytest.fixture
def pulling_fleet(tmp_path, monkeypatch):
    """A Fleet whose instance answers one file listing and whose fetch only records where it was told to write."""
    monkeypatch.setattr(remote, "PULLS", tmp_path / "default_pulls")
    calls = []

    class Target:
        def agent_json(self, *args):
            return {"base": "/remote", "files": [{"path": "job/x.txt", "size": 3}]}

    def fetch(target, base, paths, dest, progress=None):
        calls.append(Path(dest))

    fl = Fleet(home=tmp_path / "fleet", connect=lambda inst: Target(), fetch=fetch)
    fl.calls = calls
    return fl


def test_a_pull_keeps_downloading_into_the_runs_own_folder_new_runs_go_to_the_current_root(pulling_fleet, tmp_path):
    fl = pulling_fleet
    rec, folder = add_run(fl, "job", state="running")  # under the fleet's default root
    fl.apply_pulls_folder(str(tmp_path / "fresh"), "fresh")  # the current root is another one now
    fl.pull({"id": 1}, rec["key"], ["logs"])
    assert fl.calls[-1] == folder.parent  # the existing run: its own folder
    assert fl.load_run(rec["key"])["folder"] == str(folder)
    new = {"key": "2-job", "instance": 2, "job": "job", "state": "running"}
    fl.save_run(new)
    fl.pull({"id": 2}, "2-job", ["logs"])
    assert fl.calls[-1] == tmp_path / "fresh" / "2"  # a new run: the current root
    assert fl.pull_folder(new) == tmp_path / "fresh" / "2" / "job"


def test_a_record_folder_outside_every_root_is_not_followed(pulling_fleet, tmp_path):
    fl = pulling_fleet
    other = tmp_path / "elsewhere" / "job"
    other.mkdir(parents=True)
    rec = {"key": "1-job", "instance": 1, "job": "job", "state": "running", "folder": str(other)}
    assert fl.pull_folder(rec) == fl.pulls / "1" / "job"


def test_apply_refuses_when_the_running_check_says_so_even_after_planning(default_history, tmp_path):
    (default_history).mkdir(parents=True)
    fill(default_history)
    asked = []

    def blocked():
        asked.append(1)
        return "A training run is going."

    with pytest.raises(storage.StorageError, match="training run"):
        history.apply_save_folder(str(tmp_path / "t"), "move", blocked=blocked)
    assert asked and default_history.exists() and storage.history_dir_setting() is None
    assert storage.history_move() is None  # refused before any journal was written


def test_the_running_check_is_asked_again_right_before_the_switch(default_history, tmp_path):
    fill(default_history)
    target = tmp_path / "t"
    target.mkdir()  # the copy path
    answers = iter(["", "", "A training run started."])  # fine at the start and before the copy, then a run starts
    with pytest.raises(storage.StorageError, match="started"):
        history.apply_save_folder(str(target), "move", blocked=lambda: next(answers, "A training run started."))
    assert storage.history_dir_setting() is None  # not switched
    assert storage.history_move() is not None  # the journal stays: Resume or Cancel


def test_resume_returns_the_destination_path(default_history, tmp_path):
    fill(default_history)
    target = tmp_path / "t"
    target.mkdir()
    real = storage._verify
    calls = {"n": 0}

    def flaky(o, n):
        calls["n"] += 1
        if calls["n"] == 1:
            raise storage.StorageError("bad copy")
        real(o, n)

    storage._verify, orig = flaky, storage._verify
    try:
        with pytest.raises(storage.StorageError):
            history.apply_save_folder(str(target), "move")
        out = history.resume_save_move()
    finally:
        storage._verify = orig
    assert Path(out["path"]) == target and history.history_dir() == target


def test_two_changes_of_one_folder_cannot_overlap_in_this_process(default_history, tmp_path):
    fill(default_history)
    inside = threading.Event()
    release = threading.Event()
    results = []

    def slow_blocked():
        inside.set()
        release.wait(5)
        return ""

    def first():
        try:
            history.apply_save_folder(str(tmp_path / "a"), "fresh", blocked=slow_blocked)
            results.append("ok")
        except storage.StorageError as e:
            results.append(str(e))

    t = threading.Thread(target=first)
    t.start()
    assert inside.wait(5)
    assert storage.moving_here("history")
    with pytest.raises(storage.StorageError, match="already going"):
        history.apply_save_folder(str(tmp_path / "b"), "fresh")
    with pytest.raises(storage.StorageError, match="already going"):
        history.resume_save_move()
    with pytest.raises(storage.StorageError, match="already going"):
        history.cancel_save_move()
    release.set()
    t.join(5)
    assert results == ["ok"] and not storage.moving_here("history")


def test_a_journal_with_this_pid_is_idle_unless_this_process_is_moving(tmp_path):
    journal = {"from": "a", "to": "b", "pid": os.getpid()}
    assert not storage.journal_live(journal, "history")
    assert storage.begin_move("history")
    try:
        assert storage.journal_live(journal, "history")
    finally:
        storage.end_move("history")


def test_one_root_that_cannot_be_resolved_does_not_hide_the_others(pulling_fleet, tmp_path, monkeypatch):
    import webui  # noqa: F401  (needs the same optional imports as the Fleet page)

    good = tmp_path / "good"
    (good / "1" / "job").mkdir(parents=True)
    bad = tmp_path / "bad"
    real_resolve = Path.resolve

    def resolve(self, *a, **k):
        if self == bad:
            raise OSError("share offline")
        return real_resolve(self, *a, **k)

    monkeypatch.setattr(Path, "resolve", resolve)
    rec = {"folder": str(good / "1" / "job")}
    assert webui.fleet_folder(rec, [bad, good]) == real_resolve(good / "1" / "job")
    fl = pulling_fleet
    fl.pull_roots = lambda: [bad, good]
    assert fl.root_of(good / "1" / "job") == good


def test_the_report_fallback_searches_every_root(tmp_path):
    import webui
    from test_webui_fleet import _ReportPage, _report_folder

    a, b = tmp_path / "a", tmp_path / "b"
    _report_folder(b, logs=True, probes=True, maps=True)  # b/7/job
    a.mkdir()
    page = _ReportPage({"error": "Instance 7 is gone."})
    page.run({"key": "7/job", "instance": 7, "job": "job"}, [a, b])
    assert page.made and Path(page.made[0]) == (b / "7" / "job").resolve()
