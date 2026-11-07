"""The main window and Settings dialog, driven without showing them (conftest sets everything up)."""
import os
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
    (tmp_path / "a.txt").mkdir()  # a folder where the caption file should go: writing it fails
    win.model.add_paths([p])
    win.failures = 0
    win.snapshots = {p: ""}
    win._on_captioned(p, "a cat")
    assert win.failures == 1 and win.model.items[0]["status"] == ERROR
    assert "FAILED a.png" in win.log_view.toPlainText()


def test_kept_edit_is_not_a_failure(win, tmp_path):
    p = _image(tmp_path / "a.png")
    win.model.add_paths([p])
    win.snapshots = {p: ""}
    win.model.set_caption(0, "typed by hand")
    win.failures = 0
    win._on_captioned(p, "a cat")
    assert win.failures == 0 and win.model.items[0]["caption"] == "typed by hand"
    assert win.model.items[0]["status"] == IDLE


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
