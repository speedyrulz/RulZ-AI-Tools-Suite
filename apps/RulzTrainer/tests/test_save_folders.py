"""The two save folders the user can change (rulztrainer/storage.py): the Previous Runs history and the Fleet's
downloads. No GPU, no network: temp folders only, and conftest points RULZ_TRAINER_HOME at a temp folder."""
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))

import rulz_fleet as fleet_module  # noqa: E402
import rulz_remote as remote  # noqa: E402
from rulz_fleet import Fleet, Refused  # noqa: E402

from rulztrainer import history, storage  # noqa: E402


@pytest.fixture
def default_history(tmp_path, monkeypatch):
    root = tmp_path / "install" / "output" / "_history"
    monkeypatch.setattr(history, "HISTORY_DIR", root)
    monkeypatch.setattr(history, "ROOT", tmp_path / "install")
    return root


def fill(root: Path, names=("a.json", "sub/b.json", "sub/c.bin")):
    for n in names:
        p = root / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(n * 10, encoding="utf-8")


def tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in root.rglob("*")
            if p.is_file() and p.name != storage.MARKER_NAME}


# ---------------------------------------------------------------- history_dir

def test_history_dir_follows_the_setting_and_the_monkeypatched_default(default_history, tmp_path):
    assert history.history_dir() == default_history
    other = tmp_path / "elsewhere"
    storage.set_history_dir(other)
    assert history.history_dir() == other
    storage.set_history_dir(None)
    assert history.history_dir() == default_history
    assert history.list_runs() == []  # the default root is read at call time: nothing there, no error


def test_the_tests_never_see_the_real_storage_file():
    assert storage.settings_path().is_relative_to(Path(os.environ["RULZ_TRAINER_HOME"]))


# ---------------------------------------------------------------- validation

def test_validation_refuses_equal_inside_containing_relative_and_unmarked_folders(default_history, tmp_path):
    old = tmp_path / "old"
    fill(old)
    kw = {"kind": "history", "old": old, "default": default_history, "output_dir": tmp_path / "install" / "output"}
    with pytest.raises(storage.StorageError, match="already"):
        storage.validate_root(str(old), **kw)
    with pytest.raises(storage.StorageError, match="inside"):
        storage.validate_root(str(old / "deeper"), **kw)
    with pytest.raises(storage.StorageError, match="contains"):
        storage.validate_root(str(tmp_path), **kw)
    with pytest.raises(storage.StorageError, match="full path"):
        storage.validate_root("relative/dir", **kw)
    full = tmp_path / "full"
    fill(full)
    with pytest.raises(storage.StorageError, match="not empty"):
        storage.validate_root(str(full), **kw)
    storage.write_marker(full, "downloads")  # the wrong kind of marker still refuses
    with pytest.raises(storage.StorageError, match="not empty"):
        storage.validate_root(str(full), **kw)
    storage.write_marker(tmp_path / "ok", "history")  # no dir: write_marker does nothing
    (tmp_path / "ok").mkdir()
    new, _ = storage.validate_root(str(tmp_path / "ok"), **kw)
    assert new == tmp_path / "ok"


def test_a_history_root_inside_output_is_refused_unless_it_is_the_default(default_history, tmp_path):
    old = tmp_path / "old"
    kw = {"kind": "history", "old": old, "default": default_history, "output_dir": tmp_path / "install" / "output"}
    with pytest.raises(storage.StorageError, match="training jobs"):
        storage.validate_root(str(tmp_path / "install" / "output" / "mine"), **kw)
    new, _ = storage.validate_root(str(default_history), **kw)
    assert new == default_history


def test_a_marked_folder_of_the_same_kind_is_accepted(default_history, tmp_path):
    marked = tmp_path / "marked"
    fill(marked)
    storage.write_marker(marked, "history")
    new, _ = storage.validate_root(str(marked), kind="history", old=tmp_path / "old", default=default_history)
    assert new == marked


@pytest.mark.skipif(os.name != "nt", reason="needs a case-insensitive file system")
def test_validation_sees_a_different_spelling_of_the_same_folder(tmp_path):
    old = tmp_path / "Old"
    old.mkdir()
    with pytest.raises(storage.StorageError, match="already"):
        storage.validate_root(str(tmp_path / "OLD"), kind="history", old=old, default=tmp_path / "d")


# ---------------------------------------------------------------- the history move

def test_history_move_by_rename_switches_and_marks(default_history, tmp_path):
    fill(default_history)
    target = tmp_path / "new" / "hist"
    result = history.apply_save_folder(str(target), "move")
    assert result["method"] == "rename"
    assert tree(target) == {"a.json": "a.json" * 10, "sub/b.json": "sub/b.json" * 10, "sub/c.bin": "sub/c.bin" * 10}
    assert not default_history.exists()
    assert history.history_dir() == target and storage.has_marker(target, "history")
    assert storage.history_move() is None


def test_history_move_by_copy_switches_only_after_verifying_and_deletes_the_old_copy(default_history, tmp_path,
                                                                                    monkeypatch):
    fill(default_history)
    before = tree(default_history)
    target = tmp_path / "other"
    target.mkdir()  # exists, so no rename: the copy path
    seen = {}
    real = storage._verify

    def spy(old, new):
        seen.setdefault("setting_during_verify", storage.history_dir_setting())  # the first pass
        real(old, new)

    monkeypatch.setattr(storage, "_verify", spy)
    result = history.apply_save_folder(str(target), "move")
    assert result["method"] == "copy" and result["leftovers"] == []
    assert seen["setting_during_verify"] is None  # not switched before the check
    assert tree(target) == before and not default_history.exists()
    assert history.history_dir() == target


def test_a_copy_that_fails_keeps_the_old_setting_and_resumes(default_history, tmp_path, monkeypatch):
    fill(default_history)
    before = tree(default_history)
    target = tmp_path / "other"
    target.mkdir()
    real_copy = storage.shutil.copy2
    calls = {"n": 0}

    def flaky(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk went away")
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(storage.shutil, "copy2", flaky)
    with pytest.raises(storage.StorageError, match="disk went away"):
        history.apply_save_folder(str(target), "move")
    assert storage.history_dir_setting() is None and history.history_dir() == default_history
    journal = storage.history_move()
    assert journal and journal["to"] == str(target) and not storage.history_move_active()
    assert tree(default_history) == before  # nothing was deleted
    # while the journal stands a new change is refused; writers are not blocked (nobody owns the move)
    with pytest.raises(storage.StorageError, match="not finished"):
        history.apply_save_folder(str(tmp_path / "third"), "move")
    monkeypatch.setattr(storage.shutil, "copy2", real_copy)
    history.resume_save_move()
    assert storage.history_move() is None and history.history_dir() == target
    assert tree(target) == before and not default_history.exists()


def test_cancel_removes_only_what_the_move_copied(default_history, tmp_path, monkeypatch):
    fill(default_history)
    target = tmp_path / "other"
    target.mkdir()
    real_verify = storage._verify
    monkeypatch.setattr(storage, "_verify", lambda o, n: (_ for _ in ()).throw(storage.StorageError("bad copy")))
    with pytest.raises(storage.StorageError):
        history.apply_save_folder(str(target), "move")
    monkeypatch.setattr(storage, "_verify", real_verify)
    assert (target / "a.json").exists()
    out = history.cancel_save_move()
    assert out["kept"] == [] and not target.exists()  # it was empty before: its copies and marker are gone
    assert storage.history_move() is None and history.history_dir() == default_history
    assert len(tree(default_history)) == 3


def test_cancel_leaves_a_target_that_already_held_files(default_history, tmp_path, monkeypatch):
    fill(default_history)
    target = tmp_path / "other"
    fill(target, ["mine.txt"])
    storage.write_marker(target, "history")
    monkeypatch.setattr(storage, "_verify", lambda o, n: (_ for _ in ()).throw(storage.StorageError("bad copy")))
    with pytest.raises(storage.StorageError):
        history.apply_save_folder(str(target), "move")
    out = history.cancel_save_move()
    assert (target / "mine.txt").exists() and out["kept"]


def test_start_fresh_keeps_the_old_data_and_carries_the_skip_lists(default_history, tmp_path):
    fill(default_history)
    (default_history / history.HIDDEN_VAST).write_text(json.dumps(["vast-1-a"]), encoding="utf-8")
    target = tmp_path / "fresh"
    result = history.apply_save_folder(str(target), "fresh")
    assert result["method"] == "fresh"
    assert history.history_dir() == target
    assert (default_history / "a.json").exists()  # the old data stays
    assert not (target / "a.json").exists()
    assert history.hidden_vast() == {"vast-1-a"}  # a removed vast.ai run does not come back


def test_a_missing_chosen_root_is_not_recreated_and_saves_are_skipped(default_history, tmp_path, caplog):
    gone = tmp_path / "unplugged" / "hist"
    storage.set_history_dir(gone)
    assert history.missing_root() == gone
    assert history.import_fleet_runs(fleet_dir=tmp_path) == 0
    assert history.set_label("x", "y") is False and history.delete_runs(["x"]) == 0
    assert not gone.exists() and not gone.parent.exists()


def test_history_writers_wait_while_a_live_process_moves_the_history(default_history, tmp_path):
    storage.set_history_move({"from": str(default_history), "to": str(tmp_path / "t"), "pid": os.getppid()})
    assert storage.history_move_active()
    assert history.set_label("x", "y") is False
    storage.set_history_move({"from": str(default_history), "to": str(tmp_path / "t"), "pid": 0})
    assert not storage.history_move_active()


# ---------------------------------------------------------------- the vast import stamp

def test_the_imported_vast_record_stamp_names_the_folder(default_history, tmp_path):
    runs = tmp_path / "fleet_runs"
    runs.mkdir()
    folder_a, folder_b = tmp_path / "dl_a" / "job", tmp_path / "dl_b" / "job"
    probes = [{"step": 10, "face_ema": 50.0, "body_ema": 40.0, "combined_ema": 45.0}]
    rec = {"key": "1-job", "instance": 1, "job": "job", "state": "done", "step": 10, "total": 10, "probes": probes,
           "started": 1.0, "ended": 2.0, "modified": 2.0}
    folder_a.mkdir(parents=True)
    (runs / "1-job.json").write_text(json.dumps({**rec, "folder": str(folder_a)}), encoding="utf-8")
    assert history.import_fleet_runs(fleet_dir=runs) == 1
    assert history.import_fleet_runs(fleet_dir=runs) == 0  # unchanged: not imported again
    folder_b.mkdir(parents=True)
    (runs / "1-job.json").write_text(json.dumps({**rec, "folder": str(folder_b)}), encoding="utf-8")
    assert history.import_fleet_runs(fleet_dir=runs) == 1  # the downloads moved: imported again
    stored = history.list_runs()[0]
    assert stored["job_dir"] == str(folder_b) and stored["source"]["stamp"][-1] == str(folder_b)


# ---------------------------------------------------------------- the Fleet's downloads

@pytest.fixture
def fleet(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "PULLS", tmp_path / "default_pulls")
    return Fleet(home=tmp_path / "fleet")


def add_run(fleet, name="job", instance=1, state="done", under=None):
    root = under or fleet.pulls
    folder = root / str(instance) / name
    fill(folder, ["log.txt", "probes/p.png"])
    rec = {"key": f"{instance}-{name}", "instance": instance, "job": name, "state": state, "folder": str(folder)}
    fleet.save_run(rec)
    return rec, folder


def test_pulls_resolution_order(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "PULLS", tmp_path / "default_pulls")
    assert Fleet().pulls == tmp_path / "default_pulls"  # no home given: remote.PULLS
    own = Fleet(home=tmp_path / "f")
    assert own.pulls == tmp_path / "f" / "files"  # a Fleet with its own home (what the tests rely on)
    own.apply_pulls_folder(str(tmp_path / "chosen"), "fresh")
    assert own.pulls == tmp_path / "chosen"  # the setting is read at each use
    assert Fleet(home=tmp_path / "f", pulls=tmp_path / "explicit").pulls == tmp_path / "explicit"


def test_downloads_move_rewrites_the_records_and_drops_the_old_root(fleet, tmp_path):
    rec, folder = add_run(fleet)
    old = fleet.pulls
    target = tmp_path / "moved"
    result = fleet.apply_pulls_folder(str(target), "move")
    assert result["method"] == "rename" and fleet.pulls == target
    new_folder = target / "1" / "job"
    assert fleet.load_run(rec["key"])["folder"] == str(new_folder) and new_folder.is_dir()
    assert fleet.run_folder(fleet.load_run(rec["key"])) == new_folder.resolve()
    assert not old.exists() and fleet.move_state() is None
    assert "pulls_roots" not in fleet.read()["settings"]  # a root moved away from is not kept as a past root
    assert storage.has_marker(target, "downloads")


def test_downloads_move_by_copy_and_resume_after_a_failure(fleet, tmp_path, monkeypatch):
    rec, folder = add_run(fleet)
    target = tmp_path / "moved"
    target.mkdir()
    real = storage.shutil.copy2
    calls = {"n": 0}

    def flaky(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("share dropped")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(storage.shutil, "copy2", flaky)
    with pytest.raises(storage.StorageError):
        fleet.apply_pulls_folder(str(target), "move")
    assert fleet.pulls == tmp_path / "fleet" / "files"  # still the old one
    assert fleet.load_run(rec["key"])["folder"] == str(folder)
    assert fleet.move_state() and not fleet.move_state()["active"]
    monkeypatch.setattr(storage.shutil, "copy2", real)
    fleet.resume_pulls_move()
    assert fleet.pulls == target and fleet.move_state() is None
    assert fleet.load_run(rec["key"])["folder"] == str(target / "1" / "job")
    assert (target / "1" / "job" / "probes" / "p.png").is_file() and not folder.exists()


def test_start_fresh_keeps_the_old_root_readable_and_removable(fleet, tmp_path, monkeypatch):
    rec, folder = add_run(fleet)
    old = fleet.pulls
    fleet.apply_pulls_folder(str(tmp_path / "fresh"), "fresh")
    roots = fleet.pull_roots()
    assert any(storage.same_path(r, old) for r in roots)  # (the fleet's own default root is always one)
    assert storage.has_marker(old, "downloads")
    assert fleet.run_folder(fleet.load_run(rec["key"])) == folder.resolve()  # the old run is still found
    rec2, folder2 = add_run(fleet, "job2")  # new downloads land in the new root
    assert folder2.is_relative_to(tmp_path / "fresh")
    assert fleet.run_folder(rec2) == folder2.resolve()
    monkeypatch.setattr(fleet_module, "move_to_recycle_bin", lambda p: __import__("shutil").rmtree(p) or True)
    assert fleet.delete_run(rec["key"])["folder"] == str(folder.resolve())
    # a folder outside every root is never touched
    outside = tmp_path / "outside"
    fill(outside)
    fleet.save_run({"key": "9-x", "instance": 9, "job": "x", "state": "done", "folder": str(outside)})
    assert fleet.run_folder(fleet.load_run("9-x")) is None
    # an unmarked folder listed as a past root is not trusted
    with fleet.edit() as st:
        st["settings"]["pulls_roots"] = [str(outside)]
    assert fleet.run_folder(fleet.load_run("9-x")) is None


def test_remote_pull_root_is_always_a_resolvable_root(fleet, tmp_path):
    rec, folder = add_run(fleet, under=remote.PULLS)
    assert fleet.run_folder(rec) == folder.resolve()
    assert fleet.run_folder({"folder": str(remote.PULLS)}) is None  # the root itself, never


def test_pull_is_refused_while_a_live_process_moves_the_downloads(fleet):
    rec, _ = add_run(fleet, state="running")
    with fleet.edit() as st:
        st["settings"]["move"] = {"from": "a", "to": "b", "pid": os.getppid(), "started": 1}
        st["instances"]["1"] = {"id": 1, "enabled": True}
    assert "being moved" in fleet.pull({"id": 1}, rec["key"], ["logs"])["error"]
    with pytest.raises(Refused, match="being moved"):
        fleet.pull_now(rec["key"], ["logs"])
    with fleet.edit() as st:
        st["settings"]["move"]["pid"] = 0  # nobody owns it any more: pulls go on
    assert not fleet._moving()


def test_a_download_going_blocks_a_change(fleet, tmp_path):
    fleet._download_write({"run": "1-job", "state": "going", "updated": fleet.now(), "pid": os.getpid()})
    with pytest.raises(storage.StorageError, match="download is going"):
        fleet.apply_pulls_folder(str(tmp_path / "x"), "fresh")


def test_a_missing_chosen_downloads_folder_fails_the_pull_with_a_reason(fleet, tmp_path):
    rec, _ = add_run(fleet, state="running")
    fleet.update_run(rec["key"], lambda r: r.pop("folder"))  # a new run: it would go to the chosen folder
    fleet.apply_pulls_folder(str(tmp_path / "usb"), "fresh")
    import shutil

    shutil.rmtree(tmp_path / "usb")
    assert fleet.pulls_missing() == tmp_path / "usb"
    got = fleet.pull({"id": 1}, rec["key"], ["logs"])
    assert "was not found" in got["error"] and not (tmp_path / "usb").exists()
    assert "was not found" in fleet.load_run(rec["key"])["pull_error"]


def test_reset_to_default_clears_the_setting(fleet, tmp_path):
    fleet.apply_pulls_folder(str(tmp_path / "chosen"), "fresh")
    fleet.apply_pulls_folder(str(fleet.default_pulls()), "fresh")
    assert fleet.read()["settings"].get("pulls_dir") is None and fleet.pulls == fleet.default_pulls()


def test_the_cli_and_the_check_run_read_the_same_setting(tmp_path, monkeypatch):
    vast = tmp_path / "vast"
    monkeypatch.setattr(remote, "STATE", vast)
    monkeypatch.setattr(remote, "PULLS", vast / "runs")
    monkeypatch.setenv("RULZ_VAST_HOME", str(vast))
    assert remote.pulls_root() == vast / "runs"
    (vast / "fleet").mkdir(parents=True)
    chosen = tmp_path / "chosen"
    (vast / "fleet" / "state.json").write_text(json.dumps({"settings": {"pulls_dir": str(chosen)}}), encoding="utf-8")
    assert remote.pulls_root() == chosen
    assert storage.fleet_pull_roots()[:2] == [chosen, vast / "runs"]


def test_webui_finds_a_run_in_any_of_the_roots(tmp_path):
    pytest.importorskip("psutil")
    import webui

    a, b = tmp_path / "a", tmp_path / "b"
    (b / "1" / "job").mkdir(parents=True)
    rec = {"folder": str(b / "1" / "job")}
    assert webui.fleet_folder(rec, a) is None
    assert webui.fleet_folder(rec, [a, b]) == (b / "1" / "job").resolve()
    assert webui.fleet_folder(rec, b) == (b / "1" / "job").resolve()
