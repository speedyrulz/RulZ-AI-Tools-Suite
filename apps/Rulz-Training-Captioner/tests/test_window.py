"""The main window and Settings dialog, driven without showing them (conftest sets everything up)."""
import os
import stat
import time

import pytest
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QMessageBox

from captioner import main_window, settings_dialog
from captioner.config import DEFAULT_SETTINGS
from captioner.model import BUSY, ERROR, IDLE, QUEUED


@pytest.fixture
def boxes(monkeypatch):
    """Message boxes record what they would show instead of waiting for a click."""
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda parent, title, text, *a: shown.append(title)))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda parent, title, text, *a: shown.append(title) or QMessageBox.Yes))
    return shown


@pytest.fixture
def win(boxes):
    w = main_window.MainWindow()
    yield w
    w.servers.stop()
    w.deleteLater()


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = QImage(16, 16, QImage.Format_RGB32)
    img.fill(QColor("green"))
    assert img.save(str(path))
    return str(path)


def _wait_adds(w, timeout=10):
    deadline = time.time() + timeout
    while w.adding and time.time() < deadline:
        QApplication.processEvents()
    assert not w.adding, "adding images never finished"


def test_adding_folders_runs_in_the_background(win, boxes, tmp_path):
    b1 = _image(tmp_path / "b" / "1.png")
    a10 = _image(tmp_path / "a" / "10.png")
    a2 = _image(tmp_path / "a" / "2.png")
    top_png = _image(tmp_path / "top.png")
    top_jpg = _image(tmp_path / "top.jpg")
    (tmp_path / "a" / "2.txt").write_text("a cat", encoding="utf-8")
    win._add([str(tmp_path)], recursive=True)
    # nothing is listed until the scan reports back to the UI thread: the window stayed free meanwhile
    assert win.model.items == [] and win.adding == 1
    _wait_adds(win)
    paths = [it["path"] for it in win.model.items]
    assert paths[2:] == [a2, a10, b1]  # folder by folder, natural order inside each
    assert sorted(paths[:2]) == sorted([top_png, top_jpg])
    assert win.model.items[2]["caption"] == "a cat"
    assert boxes == ["Images share a caption file"]  # top.png + top.jpg
    assert "Added 5 image(s)" in win.statusBar().currentMessage()


def test_clear_list_drops_a_scan_still_running(win, tmp_path):
    _image(tmp_path / "a.png")
    win._add([str(tmp_path)])
    win.clear_list()  # answered Yes by the fixture
    _wait_adds(win)
    assert win.model.items == []


def test_engine_signals_drive_the_status_dots(win, tmp_path):
    p = _image(tmp_path / "a.png")
    win.model.add_paths([p])
    win.engine.started.emit(p)
    assert win.model.items[0]["status"] == BUSY  # the blue "being captioned" dot
    win.engine.requeued.emit(p)
    assert win.model.items[0]["status"] == QUEUED


def test_caption_that_cant_be_saved_counts_as_failed(win, tmp_path):
    p = _image(tmp_path / "a.png")
    cap = tmp_path / "a.txt"
    cap.write_text("old", encoding="utf-8")
    win.model.add_paths([p])
    os.chmod(cap, stat.S_IREAD)  # read-only: replacing it and writing it both fail
    try:
        win.failures = 0
        win.snapshots = {p: "old"}
        win._on_captioned(p, "a cat")
    finally:
        os.chmod(cap, stat.S_IREAD | stat.S_IWRITE)
    assert win.failures == 1 and win.model.items[0]["status"] == ERROR
    assert "FAILED a.png" in win.log_view.toPlainText() and "Could not write" in win.log_view.toPlainText()
    assert cap.read_text(encoding="utf-8") == "old"


def test_kept_edit_is_not_a_failure(win, tmp_path):
    p = _image(tmp_path / "a.png")
    win.model.add_paths([p])
    win.snapshots = {p: ""}
    win.model.set_caption(0, "typed by hand")
    win.failures = 0
    win._on_captioned(p, "a cat")
    assert win.failures == 0 and win.model.items[0]["caption"] == "typed by hand"
    assert win.model.items[0]["status"] == IDLE


def _open_editor(win, row=0):
    idx = win.proxy.index(row, 0)
    win.grid.edit(idx)
    QApplication.processEvents()
    ed = win.grid.indexWidget(idx)
    assert ed is not None
    return ed


@pytest.mark.parametrize("on_disk", [b"old caption\r\n", b"old caption\n", "old\u00a0caption".encode("utf-8")])
def test_untouched_open_editor_doesnt_block_the_model_caption(win, tmp_path, on_disk):
    # the editor turns CRLF into LF and non-breaking spaces into spaces; committing it unchanged
    # rewrote the file and counted as an edit made meanwhile, so the new caption was dropped
    p = _image(tmp_path / "a.png")
    (tmp_path / "a.txt").write_bytes(on_disk)
    win.model.add_paths([p])
    win.snapshots = {p: win.model.items[0]["caption"]}
    _open_editor(win)  # clicked into the box, nothing typed
    win._on_captioned(p, "a new caption")
    assert (tmp_path / "a.txt").read_bytes() == b"a new caption"
    assert win.model.items[0]["caption"] == "a new caption"


def test_untouched_editor_closing_leaves_the_file_alone(win, tmp_path):
    p = _image(tmp_path / "a.png")
    (tmp_path / "a.txt").write_bytes(b"a cat\r\n")
    win.model.add_paths([p])
    ed = _open_editor(win)
    win.grid.commitData(ed)
    assert (tmp_path / "a.txt").read_bytes() == b"a cat\r\n"


def test_typing_in_an_open_editor_still_wins_over_the_model(win, tmp_path):
    p = _image(tmp_path / "a.png")
    win.model.add_paths([p])
    win.snapshots = {p: ""}
    ed = _open_editor(win)
    ed.insertPlainText("typed by hand")
    win._on_captioned(p, "a cat")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "typed by hand"
    assert "Kept your edit of a.png" in win.log_view.toPlainText()


def test_caption_file_changed_on_disk_meanwhile_is_kept(win, tmp_path):
    p = _image(tmp_path / "a.png")
    win.model.add_paths([p])
    win.snapshots = {p: ""}
    (tmp_path / "a.txt").write_text("written by another tool", encoding="utf-8")
    win.failures = 0
    win._on_captioned(p, "a cat")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "written by another tool"
    assert win.model.items[0]["caption"] == "written by another tool"  # the grid shows what is on disk
    assert win.model.items[0]["status"] == IDLE and win.failures == 0
    assert "changed on disk" in win.log_view.toPlainText()
    assert win.model.captioned_count() == 1


def test_deleted_caption_file_gets_the_new_caption(win, tmp_path):
    p = _image(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("old", encoding="utf-8")
    win.model.add_paths([p])
    win.snapshots = {p: "old"}
    os.remove(tmp_path / "a.txt")
    win._on_captioned(p, "a cat")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a cat"


def test_unreadable_caption_file_is_never_replaced(win, boxes, tmp_path):
    p = _image(tmp_path / "a.png")
    b = _image(tmp_path / "b.png")
    (tmp_path / "a.txt").mkdir()  # there, but can't be read as a file
    win._add([p, b])  # files, not the folder: a folder listing already tells a.txt isn't a file
    _wait_adds(win)
    a = win.model.items[0]
    assert a["status"] == ERROR and a.get("unreadable")
    assert "couldn't be read" in win.statusBar().currentMessage()
    assert "Couldn't read" in win.log_view.toPlainText()
    # Caption All with "skip existing" leaves it out
    started = []
    win.engine.start = lambda paths, *rest: started.append(paths)
    win.s.update(backend="external", external_urls="http://127.0.0.1:1/v1")
    win.skip.setChecked(True)
    win.caption("all")
    assert started == [[str(tmp_path / "b.png")]]
    # a model caption for it (Caption Selected) doesn't replace it either
    win.failures = 0
    win.snapshots = {p: ""}
    win._on_captioned(p, "a cat")
    assert win.failures == 1 and (tmp_path / "a.txt").is_dir()
    assert "wasn't replaced" in win.log_view.toPlainText()


def test_clear_captions_goes_on_past_a_file_it_cant_write(win, boxes, tmp_path):
    paths = [_image(tmp_path / f"{n}.png") for n in "abc"]
    for n in "abc":
        (tmp_path / f"{n}.txt").write_text(f"caption {n}", encoding="utf-8")
    win.model.add_paths(paths)
    os.chmod(tmp_path / "b.txt", stat.S_IREAD)
    try:
        win.grid.selectAll()
        win._clear_selected_captions()  # answered Yes by the fixture
    finally:
        os.chmod(tmp_path / "b.txt", stat.S_IREAD | stat.S_IWRITE)
    assert [(tmp_path / f"{n}.txt").read_text(encoding="utf-8") for n in "abc"] == ["", "caption b", ""]
    assert boxes[-1] == "Clear captions" and win.model.captioned_count() == 1
    assert "1 captioned" in win.stats.text()


def test_late_finished_signal_of_the_previous_job_is_ignored(win, tmp_path):
    p = _image(tmp_path / "a.png")
    win.model.add_paths([p])
    win.model.set_status(p, QUEUED)
    win.engine.active = True  # the next job already started when the old "finished" arrives
    win.a_stop.setEnabled(True)
    win._on_finished()
    assert win.model.items[0]["status"] == QUEUED and win.a_stop.isEnabled()
    win.engine.active = False


def test_servers_ready_after_they_were_stopped_doesnt_start_the_job(win, tmp_path):
    p = _image(tmp_path / "a.png")
    win.model.add_paths([p])
    win.model.set_status(p, QUEUED)
    started = []
    win.engine.start = lambda *a: started.append(a)
    win.starting = True
    win.pending_job = ([p], "", "hi", dict(win.s))
    win._servers_ready([{"url": "http://127.0.0.1:1/v1", "model": "default", "slots": 1, "name": "x"}])
    assert started == [] and win.pending_job is None and not win.starting
    assert win.model.items[0]["status"] == IDLE


def test_a_job_waiting_for_the_model_keeps_its_settings(win, monkeypatch, tmp_path):
    p = _image(tmp_path / "a.png")
    win.model.add_paths([p])
    monkeypatch.setattr(win, "_launch_servers", lambda: None)
    win.s.update(backend="managed")
    win.prefix.setText("ohwx, ")
    win.grid.selectAll()
    win.caption("selected")
    assert win.pending_job is not None
    win.prefix.setText("changed while loading, ")
    win.current_settings()  # e.g. Settings opened meanwhile
    got = []
    win.engine.start = lambda paths, eps, system, user, s: got.append(s)
    monkeypatch.setattr(win.servers, "running", lambda: True)
    win._servers_ready([{"url": "http://127.0.0.1:1/v1", "model": "default", "slots": 1, "name": "x"}])
    assert got and got[0]["prefix"] == "ohwx, "


def test_typing_survives_updates_to_the_same_card(win, tmp_path):
    # Qt reloads an open editor on every change to its row; the thumbnail arriving or the status dot
    # changing wiped what was being typed
    p = _image(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("old caption", encoding="utf-8")
    win.model.add_paths([p])
    ed = _open_editor(win)
    ed.insertPlainText(" plus words I typed")
    win.model.set_status(p, BUSY)
    win.model._thumb_ready(p, QImage(8, 8, QImage.Format_RGB32))
    win.model.set_statuses([0], QUEUED)
    assert ed.toPlainText() == "old caption plus words I typed"


def test_untouched_editor_follows_a_caption_change(win, tmp_path):
    p = _image(tmp_path / "a.png")
    win.model.add_paths([p])
    ed = _open_editor(win)
    win.model.set_caption(0, "set elsewhere")
    assert ed.toPlainText() == "set elsewhere"


def test_caption_commits_what_is_typed_in_an_open_box_first(win, tmp_path):
    # Ctrl+Return works from inside a caption box
    a, b = _image(tmp_path / "a.png"), _image(tmp_path / "b.png")
    win.model.add_paths([a, b])
    ed = _open_editor(win)
    ed.insertPlainText("typed just now")
    started = []
    win.engine.start = lambda paths, *rest: started.append(paths)
    win.s.update(backend="external", external_urls="http://127.0.0.1:1/v1")
    win.skip.setChecked(True)
    win.caption("all")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "typed just now"
    assert started == [[b]]  # a.png has a caption now, so "skip existing" leaves it out


def test_removing_images_during_a_job_drops_them_from_it(win, tmp_path):
    a, b = _image(tmp_path / "a.png"), _image(tmp_path / "b.png")
    win.model.add_paths([a, b])
    dropped = []
    win.engine.drop = lambda paths: dropped.extend(paths)
    win.engine.active = True
    win.grid.setCurrentIndex(win.proxy.index(0, 0))
    win.remove_selected()
    win.engine.active = False
    assert sorted(dropped) == [a]
    # and from a job still waiting for the model to load
    c = _image(tmp_path / "c.png")
    win.model.add_paths([c])
    win.pending_job = ([b, c], "", "hi", dict(win.s))
    win.grid.setCurrentIndex(win.proxy.index(0, 0))  # b
    win.remove_selected()
    assert win.pending_job[0] == [c] and win.pending_job[1:3] == ("", "hi")
    win.grid.setCurrentIndex(win.proxy.index(0, 0))  # c: nothing left, so no empty job runs later
    win.remove_selected()
    assert win.pending_job is None


def test_result_for_an_image_removed_during_the_job_is_ignored(win, tmp_path):
    p = _image(tmp_path / "a.png")
    win.model.add_paths([p])
    win.engine.drop([p])        # removed while its request was on its way
    win.model.remove_rows([0])
    win.model.add_paths([p])    # and added again before the reply came
    win.failures = 0
    win._on_captioned(p, "a cat")
    win._on_failed(p, "HTTP 500")
    assert not (tmp_path / "a.txt").exists() and win.failures == 0
    assert win.model.items[0]["status"] == IDLE
    win.engine._dropped = set()


def test_reply_for_a_cleared_image_added_again_is_ignored(win, tmp_path):
    p = _image(tmp_path / "a.png")
    win.model.add_paths([p])
    win.engine.active = True
    try:
        win.clear_list()  # while its request is on its way
    finally:
        win.engine.active = False
    win.model.add_paths([p])  # added again before the reply came
    win._on_captioned(p, "caption from the cleared job")
    assert not (tmp_path / "a.txt").exists()
    win.engine._dropped = set()
    win.engine._cancel.clear()


def test_prompt_save_that_writes_nothing_says_so(win, boxes, monkeypatch):
    monkeypatch.setattr(main_window.config, "save_prompts", lambda p: False)
    win.statusBar().clearMessage()
    win.save_prompt()
    assert boxes == ["Prompts not saved"] and "Saved prompt" not in win.statusBar().currentMessage()


def test_clear_list_stops_the_job(win, tmp_path):
    win.model.add_paths([_image(tmp_path / "a.png")])
    win.engine.active = True
    win.pending_job = ([], "", "hi", {})
    try:
        win.clear_list()
        assert win.engine._cancel.is_set() and win.pending_job is None
    finally:
        win.engine.active = False


def test_search_waits_for_a_pause_in_typing(win, tmp_path):
    a, b = _image(tmp_path / "cat.png"), _image(tmp_path / "dog.png")
    win.model.add_paths([a, b])
    win.search.setText("cat")
    assert win.proxy.rowCount() == 2  # not filtered on every keystroke
    deadline = time.time() + 3
    while win.proxy.rowCount() != 1 and time.time() < deadline:
        QApplication.processEvents()
    assert win.proxy.rowCount() == 1


def test_caption_all_uses_a_search_still_being_typed(win, tmp_path):
    a, b = _image(tmp_path / "cat.png"), _image(tmp_path / "dog.png")
    win.model.add_paths([a, b])
    started = []
    win.engine.start = lambda paths, *rest: started.append(paths)
    win.s.update(backend="external", external_urls="http://127.0.0.1:1/v1")
    win.search.setText("dog")
    win.caption("all")  # pressed before the search timer ran
    assert started == [[b]]


# ------------------------------------------------------------------ Settings ▸ GPUs
GPUS = [(0, "RTX A", 16384), (1, "RTX B", 16384)]


def _dialog(monkeypatch, gpus):
    monkeypatch.setattr(settings_dialog, "detect_gpus", lambda: GPUS)
    return settings_dialog.SettingsDialog(dict(DEFAULT_SETTINGS, gpus=gpus))


def test_gpu_tab_shows_no_gpu_saved_as_none_ticked(monkeypatch):
    dlg = _dialog(monkeypatch, [])
    assert [c.isChecked() for _, c in dlg.gpu_checks] == [False, False]
    assert not dlg.gpu_note.isHidden()  # explains that llama-server picks the GPUs itself
    assert dlg.result_settings()["gpus"] == []  # OK keeps it that way instead of ticking every GPU


def test_gpu_tab_shows_the_saved_gpus(monkeypatch):
    dlg = _dialog(monkeypatch, [1])
    assert [c.isChecked() for _, c in dlg.gpu_checks] == [False, True]
    assert dlg.gpu_note.isHidden()
    dlg.gpu_checks[1][1].setChecked(False)
    assert not dlg.gpu_note.isHidden()
    assert dlg.result_settings()["gpus"] == []
