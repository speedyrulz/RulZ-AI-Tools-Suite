"""Offscreen smoke tests of the main window with a real scan of generated photos."""
import sqlite3
import time

import numpy as np
import pytest
from PIL import Image


def _wait(qapp, cond, timeout=60):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise TimeoutError
        qapp.processEvents()
        time.sleep(0.01)


def _photos(folder, n):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (300, 400, 3), np.uint8)).save(folder / f"p{i:03d}.jpg")
    Image.fromarray(np.full((60, 80), 40000, np.uint16)).save(folder / "deep.png")
    return folder


@pytest.fixture
def window(qapp, data_dir, monkeypatch):
    monkeypatch.setenv("RULZ_THEME", "dark")
    from rulz.ui import theme
    from rulz.ui.main_window import MainWindow

    theme.apply(qapp)
    win = MainWindow()
    win.show()
    yield win
    if win.scanner.running:
        win.scanner.stop()
        _wait(qapp, lambda: not win.scanner.running)
    win.close()
    win.db.close()


def test_scan_view_and_people(qapp, window, tmp_path):
    folder = _photos(tmp_path / "pics", 12)
    assert window.db.add_folder(str(folder)) is None
    window.scanner.scan()
    _wait(qapp, lambda: not window.scanner.running, 120)
    _wait(qapp, lambda: not window._refresh_timer.isActive())
    s = window.db.stats()
    assert s["photos"] == 13 and s["pending"] == 0
    assert window.db.one("SELECT COUNT(*) FROM photos WHERE error IS NOT NULL")[0] == 0

    window.navigate(window.PHOTOS)
    ids = [t.key for t in window.photos.grid.flat]
    assert len(ids) == 13
    from rulz.ui.thumbs import ThumbLoader

    loader = ThumbLoader.instance()
    window.photos.grid.grab()  # paints, which requests the visible thumbnails
    _wait(qapp, lambda: not loader.pending)
    assert len(loader.cache) >= 1 and loader.cache_bytes > 0
    window.open_viewer(ids, 0)
    v = window.viewer
    _wait(qapp, lambda: not v._loading)
    assert not v.canvas.image.isNull() and v.pil is not None
    # Step quickly: only the last photo's decode may land.
    v.step(1)
    v.step(1)
    _wait(qapp, lambda: not v._loading)
    assert v.pil.size == v.canvas.image.size().toTuple()
    v.canvas.repaint()
    v.close_viewer()
    window.navigate(window.PEOPLE)
    window.navigate(window.LIBRARY)


def test_close_waits_for_running_scan(qapp, window, tmp_path):
    folder = _photos(tmp_path / "pics", 40)
    window.db.add_folder(str(folder))
    window.scanner.scan()
    _wait(qapp, lambda: window.scanner.running)
    window.close()
    assert window.isVisible()  # still finishing
    assert "Finishing" in window.windowTitle()
    _wait(qapp, lambda: not window.isVisible(), 60)
    assert not window.scanner.running


def test_slider_saves_once_settled(qapp, window):
    lib = window.library
    window.navigate(window.LIBRARY)
    writes = []
    window.db.conn.set_trace_callback(lambda sql: writes.append(sql) if "settings" in sql else None)
    for v in (35, 36, 37, 38):
        lib.slider.setValue(v)
    assert writes == []
    lib.flush_settings()
    window.db.conn.set_trace_callback(None)
    assert len(writes) == 1
    assert window.db.match_threshold() == pytest.approx(0.38)


def test_remove_folder_during_scan_runs_in_background_and_resumes(qapp, window, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt

    from rulz.ui import dialogs

    a, b = _photos(tmp_path / "a", 15), _photos(tmp_path / "b", 6)
    window.db.add_folder(str(a))
    window.db.add_folder(str(b))
    window.navigate(window.LIBRARY)
    lib = window.library
    row = next(i for i in range(lib.list.count()) if lib.list.item(i).data(Qt.UserRole + 1) == str(a))
    lib.list.setCurrentRow(row)
    monkeypatch.setattr(dialogs, "confirm", lambda *a, **k: True)
    window.scanner.scan()
    _wait(qapp, lambda: window.scanner.running)
    lib._remove()  # returns at once: no wait on the GUI thread
    assert window.scanner.running
    _wait(qapp, lambda: not window.scanner.running, 120)
    assert [r["path"] for r in window.db.folders()] == [str(b)]
    s = window.db.stats()
    assert s["photos"] == 7 and s["pending"] == 0  # the scan carried on with the other folder


def test_remove_survives_a_refresh_while_confirming_and_pause(qapp, window, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt

    from rulz.ui import dialogs

    a, b = _photos(tmp_path / "a", 15), _photos(tmp_path / "b", 3)
    window.db.add_folder(str(a))
    window.db.add_folder(str(b))
    window.navigate(window.LIBRARY)
    lib = window.library
    lib.list.setCurrentRow(next(i for i in range(lib.list.count())
                                if lib.list.item(i).data(Qt.UserRole + 1) == str(a)))

    def confirm_while_a_scan_refreshes(*args, **kwargs):
        lib.refresh()  # deletes the list item _remove started from
        return True

    monkeypatch.setattr(dialogs, "confirm", confirm_while_a_scan_refreshes)
    window.scanner.scan()
    _wait(qapp, lambda: window.scanner.running)
    lib._remove()
    lib.refresh()
    texts = [lib.list.item(i).text() for i in range(lib.list.count())]
    assert any(t.endswith("Removing…") for t in texts)  # still marked after a refresh
    window.scanner.stop()  # Pause: drops the follow-up scan, not the removal
    _wait(qapp, lambda: not window.scanner.running, 120)
    assert [r["path"] for r in window.db.folders()] == [str(b)]


def test_opening_a_person_that_is_gone_shows_the_overview(qapp, window):
    from rulz.faces import DetectedFace, FaceAttrs

    db = window.db
    db.x("INSERT INTO folders(id, path) VALUES(1, 'x')")
    db.x("INSERT INTO photos(id, folder_id, path, faces_done) VALUES(1, 1, 'x\\p.jpg', 1)")
    fid = db.insert_face(1, DetectedFace((0.1, 0.1, 0.3, 0.3), 0.9, np.ones(128, np.float32), FaceAttrs()))
    mom = db.create_person("Mom")
    db.assign_faces([fid], mom)
    window.open_person(mom)
    assert window.people.currentWidget() is window.people.detail
    window.open_person(mom + 99)  # e.g. a stale tile after a regroup
    assert window.people.currentWidget() is window.people.overview


def test_rewritten_files_leave_the_thumbnail_cache(qapp, window):
    from PySide6.QtGui import QPixmap

    from rulz import config
    from rulz.ui.thumbs import ThumbLoader

    loader = ThumbLoader.instance()
    for key in (config.thumb_file(5), config.face_crop_file(9)):
        pm = QPixmap(4, 4)
        loader.cache[key] = pm
        loader.cache_bytes += 64
    window._files_rewritten([5], [9])
    assert config.thumb_file(5) not in loader.cache and config.face_crop_file(9) not in loader.cache


def test_edit_on_a_person_that_is_gone_shows_message(qapp, window, monkeypatch):
    import sys

    from rulz import app
    from rulz.db import PersonGone
    from rulz.ui import dialogs

    shown = []
    monkeypatch.setattr(dialogs, "info", lambda parent, title, text: shown.append(title))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    app._install_busy_handler()
    version = window.data_version
    try:
        window.db.assign_faces([1], 12345)
    except PersonGone:
        sys.excepthook(*sys.exc_info())
    assert shown == ["Library changed"]
    assert window.data_version > version  # the page was refreshed


def test_locked_database_shows_message(qapp, window, monkeypatch):
    import sys

    from rulz import app
    from rulz.ui import dialogs

    shown = []
    monkeypatch.setattr(dialogs, "info", lambda parent, title, text: shown.append(title))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    app._install_busy_handler()
    window.db.conn.close()
    window.db.conn = sqlite3.connect(window.db.path, timeout=0.1, isolation_level=None)
    other = sqlite3.connect(window.db.path, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    try:
        window.db.set_setting("x", 1)
    except sqlite3.OperationalError:
        sys.excepthook(*sys.exc_info())
    finally:
        other.execute("ROLLBACK")
        other.close()
    assert shown == ["Library is busy"]
