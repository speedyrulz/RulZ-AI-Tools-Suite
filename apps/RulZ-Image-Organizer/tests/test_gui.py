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
