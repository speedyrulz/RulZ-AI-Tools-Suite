"""The save-folder move's journal and safety rules (rulztrainer/storage.py, rulz_fleet.py): no overwriting, phases,
cancel limits, unreadable folders, the catch-up pass, and the pull / mover lock order. Temp folders only."""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))

import rulz_remote as remote  # noqa: E402
from test_save_folders import add_run, default_history, fill, fleet  # noqa: E402,F401  (fixtures and helpers)

from rulztrainer import history, storage  # noqa: E402


def journal_io():
    box = {}
    return (lambda j: box.__setitem__("j", dict(j)), lambda: box.pop("j", None), box)


def test_a_copy_never_overwrites_a_differing_file_in_a_target_that_held_files(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    fill(old, ["a.txt"])
    new.mkdir()
    (new / "a.txt").write_text("not ours", encoding="utf-8")
    storage.write_marker(new, "history")
    set_move, clear_move, box = journal_io()
    with pytest.raises(storage.StorageError, match="differ"):
        storage.move_folder(old, new, kind="history", switch=lambda: None, set_move=set_move, clear_move=clear_move)
    assert (new / "a.txt").read_text(encoding="utf-8") == "not ours"
    assert (old / "a.txt").exists() and box["j"]["phase"] == "copying"


def test_our_own_earlier_copy_may_be_overwritten_in_a_target_that_was_empty(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    fill(old, ["a.txt"])
    new.mkdir()
    (new / "a.txt").write_text("half", encoding="utf-8")  # a partial copy of ours (target_was_empty is true)
    set_move, clear_move, _ = journal_io()
    journal = {"from": str(old), "to": str(new), "kind": "history", "phase": "copying", "target_was_empty": True}
    storage.move_folder(old, new, kind="history", switch=lambda: None, set_move=set_move, clear_move=clear_move,
                        journal=journal)  # a resume: the journal knows the target was empty when the move began
    assert (new / "a.txt").read_text(encoding="utf-8") == "a.txt" * 10 and not old.exists()


def test_resume_with_the_old_folder_missing_finishes_only_after_the_rename_or_switch(tmp_path):
    old, new = tmp_path / "gone", tmp_path / "new"
    fill(new, ["a.txt"])
    set_move, clear_move, box = journal_io()
    base = {"from": str(old), "to": str(new), "kind": "history", "phase": "copying", "target_was_empty": True}
    switched = []
    with pytest.raises(storage.StorageError, match="cannot be found; reconnect"):
        storage.move_folder(old, new, kind="history", switch=lambda: switched.append(1), set_move=set_move,
                            clear_move=clear_move, journal=base)
    assert not switched and box["j"]["pid"] == 0
    for phase in ("renamed", "switched"):
        storage.move_folder(old, new, kind="history", switch=lambda: switched.append(1), set_move=set_move,
                            clear_move=clear_move, journal={**base, "phase": phase})
    assert switched == [1, 1] and "j" not in box


def test_cancel_is_refused_once_the_switch_began_and_resume_reruns_it(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    fill(old, ["a.txt", "b/c.txt"])
    set_move, clear_move, box = journal_io()
    calls = []

    def switch():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("half done")

    with pytest.raises(storage.StorageError):
        storage.move_folder(old, new, kind="history", switch=switch, set_move=set_move, clear_move=clear_move)
    assert box["j"]["phase"] == "switched"
    with pytest.raises(storage.StorageError, match="cannot be cancelled"):
        storage.cancel_move(box["j"], clear_move)
    assert "j" in box
    storage.move_folder(old, new, kind="history", switch=switch, set_move=set_move, clear_move=clear_move,
                        journal=box["j"])
    assert len(calls) == 2 and "j" not in box and (new / "b" / "c.txt").exists()


def test_a_fleet_switch_that_stopped_half_way_is_completed_by_resume(fleet, tmp_path, monkeypatch):
    recs = [add_run(fleet, f"job{i}")[0] for i in range(3)]
    target = tmp_path / "moved"
    target.mkdir()  # exists: the copy path, so the old folder is still there to rebase from
    real = fleet.save_run
    count = {"n": 0}

    def flaky(rec):
        count["n"] += 1
        if count["n"] == 2:
            raise OSError("disk full")
        real(rec)

    monkeypatch.setattr(fleet, "save_run", flaky)
    with pytest.raises(storage.StorageError):
        fleet.apply_pulls_folder(str(target), "move")
    monkeypatch.setattr(fleet, "save_run", real)
    assert fleet.move_state()["phase"] == "switched"
    with pytest.raises(storage.StorageError, match="cannot be cancelled"):
        fleet.cancel_pulls_move()
    fleet.resume_pulls_move()
    for rec in recs:
        assert fleet.load_run(rec["key"])["folder"].startswith(str(target))
    assert fleet.pulls == target and fleet.move_state() is None


def test_cancel_removes_a_copy_only_when_size_and_time_match_and_temp_files_only_with_a_twin(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    fill(old, ["same.txt", "diff.txt", "t.txt"])
    new.mkdir()
    shutil.copy2(old / "same.txt", new / "same.txt")
    (new / "diff.txt").write_text("diff.txt" * 10, encoding="utf-8")  # same size, another time
    os.utime(new / "diff.txt", (1, 1))
    (new / "t.txt.rulzmove").write_text("x", encoding="utf-8")
    (new / "stray.rulzmove").write_text("x", encoding="utf-8")  # no twin in old
    out = storage.cancel_move({"from": str(old), "to": str(new), "phase": "copying", "target_was_empty": True},
                              lambda: None)
    assert not (new / "same.txt").exists() and not (new / "t.txt.rulzmove").exists()
    assert (new / "diff.txt").exists() and (new / "stray.rulzmove").exists()
    assert len(out["kept"]) == 2


def test_an_unreadable_folder_stops_a_move_before_the_switch(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    fill(old, ["a.txt", "d/b.txt"])
    new.mkdir()
    real_walk = os.walk

    def walk(top, topdown=True, onerror=None, followlinks=False):
        yield from real_walk(top, topdown, None, followlinks)
        if onerror and Path(top) == old:
            onerror(PermissionError("denied"))  # os.walk reports an unlistable folder to onerror

    monkeypatch.setattr(storage.os, "walk", walk)
    set_move, clear_move, _ = journal_io()
    switched = []
    with pytest.raises(storage.StorageError, match="cannot be read"):
        storage.move_folder(old, new, kind="history", switch=lambda: switched.append(1), set_move=set_move,
                            clear_move=clear_move)
    assert not switched


def test_a_write_to_the_old_folder_after_the_first_pass_is_carried_over_before_the_delete(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    fill(old, ["a.txt"])
    new.mkdir()
    set_move, clear_move, _ = journal_io()
    storage.move_folder(old, new, kind="history", set_move=set_move, clear_move=clear_move,
                        switch=lambda: fill(old, ["late.json"]))  # written between the first pass and the delete
    assert (new / "late.json").exists() and not old.exists()


def test_a_pull_and_a_mover_cannot_miss_each_other(fleet):
    """The pull checks the journal and publishes its entry under the downloads lock; the mover writes its journal and
    then looks for downloads under that lock: whichever comes second sees the other."""
    rec, _ = add_run(fleet, state="running")
    entry = {"run": rec["key"], "state": "going", "updated": fleet.now(), "pid": os.getpid()}
    assert fleet._download_begin(dict(entry)) is True  # no journal: published
    set_move, _clear = fleet._journal_callbacks()
    with pytest.raises(storage.StorageError, match="download is going"):
        set_move({"from": "a", "to": "b", "pid": os.getppid(), "phase": "copying"})  # the mover sees the entry
    assert fleet.move_state() is None  # and withdrew its journal
    fleet._download_write({**entry, "state": "done"})
    set_move, _clear = fleet._journal_callbacks()
    set_move({"from": "a", "to": "b", "pid": os.getppid(), "phase": "copying"})
    assert fleet._download_begin(dict(entry, run="other")) is False  # the pull sees the journal
    assert not any(e["run"] == "other" for e in fleet.downloads())


def test_the_cli_pull_is_refused_while_a_live_move_journal_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "STATE", tmp_path / "vast")
    assert remote.fleet_move_in_progress() is None
    (tmp_path / "vast" / "fleet").mkdir(parents=True)
    state = tmp_path / "vast" / "fleet" / "state.json"
    state.write_text(json.dumps({"settings": {"move": {"from": "a", "to": "b", "pid": os.getpid()}}}), encoding="utf-8")
    assert remote.fleet_move_in_progress()
    args = type("A", (), {"dest": None, "what": None, "step": None, "name": "x", "show": 5,
                          "remote": type("R", (), {"conn": {"id": 1}})()})()
    with pytest.raises(SystemExit, match="being moved"):
        remote.cmd_pull(args)
    state.write_text(json.dumps({"settings": {"move": {"from": "a", "to": "b", "pid": 0}}}), encoding="utf-8")
    assert remote.fleet_move_in_progress() is None


def test_a_chosen_missing_history_root_is_not_recreated_by_a_save_to_another_root(default_history, tmp_path):
    storage.set_history_dir(tmp_path / "unplugged")
    assert history.set_label("x", "y", root=tmp_path / "somewhere") is False
    assert not (tmp_path / "unplugged").exists()
