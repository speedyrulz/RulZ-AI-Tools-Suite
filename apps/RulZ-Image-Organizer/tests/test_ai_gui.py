"""Offscreen tests of the AI UI: the Library card, Advanced AI settings, the Look for dialog, a run from Photos."""
import json
import time

import pytest

from ai_stub import StubAI, save_photo
from rulz.ai import settings as ai_settings


def _wait(qapp, cond, timeout=60):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise TimeoutError
        qapp.processEvents()
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def no_slot_errors(monkeypatch):
    """An exception in a Qt slot only prints a traceback; make it fail the test instead."""
    import sys

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda t, e, tb: errors.append(e))
    yield
    assert not errors, errors


@pytest.fixture
def stub():
    s = StubAI()
    yield s
    s.close()


@pytest.fixture
def window(qapp, data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("RULZ_THEME", "dark")
    monkeypatch.setenv("RULZ_CAPTIONER_DIR", str(tmp_path / "no-captioner"))
    from rulz.ui import theme
    from rulz.ui.main_window import MainWindow

    theme.apply(qapp)
    win = MainWindow()
    win.show()
    yield win
    win.ai.stop()
    _wait(qapp, lambda: not win.ai.running)
    if win.scanner.running:
        win.scanner.stop()
        _wait(qapp, lambda: not win.scanner.running)
    win.close()
    from PySide6.QtCore import QTimer

    for t in win.findChildren(QTimer):  # a refresh still due would run in the next test, on a closed library
        t.stop()
    win.db.close()


def _library(qapp, window, tmp_path, colours):
    folder = tmp_path / "pics"
    folder.mkdir()
    for i, c in enumerate(colours):
        save_photo(folder / f"{i:02d}-{c}.jpg", c)
    window.db.add_folder(str(folder))
    window.scanner.scan()
    _wait(qapp, lambda: not window.scanner.running, 120)
    rows = window.db.q("SELECT id, path FROM photos ORDER BY path")
    return [r["id"] for r in rows]


def _set_own(window, url, model="vis-model"):
    card = window.library.ai_card
    card.mode.buttons[1].click()
    card.urls.setText(url)
    card._edited()
    card.model.setEditText(model)
    card.flush()
    return card


def test_ai_card_modes_save_and_warn(qapp, window, tmp_path, stub, monkeypatch):
    window.navigate(window.LIBRARY)
    card = window.library.ai_card
    assert card.mode.current() == 0 and not card.own.isVisibleTo(card) and not card.test_btn.isEnabled()
    card.mode.buttons[1].click()
    assert ai_settings.load()["mode"] == "own" and card.own.isVisibleTo(card)
    card.model.setEditText("")
    card.flush()
    assert "Choose a model" in card.status.text()
    _set_own(window, stub.url)
    s = ai_settings.load()
    assert s["urls"] == stub.url and s["model"] == "vis-model"
    assert card.status.text().startswith("Ready") and card.test_btn.isEnabled()
    assert not card.remote_warn.isVisibleTo(card)
    card.urls.setText("http://10.0.0.5:11434/v1")
    card._edited()
    card.flush()
    assert card.remote_warn.isVisibleTo(card)

    card.urls.setText(stub.url)  # the model list from the server keeps the model the user typed
    card._edited()
    card._load_models()
    _wait(qapp, lambda: card.load_btn.isEnabled() and card.model.count() == 2)
    assert card.model.currentText() == "vis-model" and "2 models" in card.status.text()

    card._test()
    _wait(qapp, lambda: not window.ai.running)
    qapp.processEvents()
    assert card.status.text().startswith("Connected"), card.status.text()
    assert not window.ai_box.isVisible()  # a test reports on the card, not in the sidebar

    cap_dir = tmp_path / "no-captioner"
    cap_dir.mkdir()
    (cap_dir / "settings.json").write_text(json.dumps({"backend": "external", "external_urls": stub.url,
                                                       "external_model": "vis-model"}), encoding="utf-8")
    card.mode.buttons[2].click()
    assert ai_settings.load()["mode"] == "captioner"
    assert stub.url in card.cap_text.text() and str(cap_dir) in card.cap_path.text()
    assert card.status.text().startswith("Ready") and not card.own.isVisibleTo(card)
    card.mode.buttons[0].click()
    assert card.status.text() == "Off." and not card.test_btn.isEnabled()


def test_advanced_dialog_is_locked_and_warns(qapp, window, monkeypatch):
    from rulz.ui import ai_panel, dialogs

    s = ai_settings.load()
    dlg = ai_panel.AdvancedDialog(window, s)
    assert not dlg.tag_system.isEnabled() and not dlg.save_btn.isEnabled()
    assert dlg.tag_system.toPlainText() == ai_settings.TAG_SYSTEM
    dlg.unlock.setChecked(True)
    assert dlg.tag_system.isEnabled() and dlg.save_btn.isEnabled()
    shown = []
    monkeypatch.setattr(dialogs, "confirm", lambda parent, title, text, **kw: shown.append(text) or True)
    dlg.accept()  # nothing changed: closes without asking
    assert shown == [] and dlg.result() == 1

    dlg = ai_panel.AdvancedDialog(window, s)
    dlg.unlock.setChecked(True)
    dlg.temperature.setValue(0.5)
    dlg.look_user.setPlainText("Is there any of this: something?")
    dlg.tag_user.setPlainText("")
    dlg.accept()
    assert len(shown) == 1
    text = shown[0]
    assert "temperature" in text and "Look for question" in text and "{query}" in text and "built-in text" in text
    ai_settings.save(dlg.result_settings(s))
    s2 = ai_settings.load()
    assert s2["temperature"] == 0.5 and s2["tag_user"] == "" and ai_settings.prompt(s2, "tag_user") == ai_settings.TAG_USER
    assert s2["look_user"] == "Is there any of this: something?"

    dlg = ai_panel.AdvancedDialog(window, s2)
    dlg.unlock.setChecked(True)
    dlg._restore()
    assert dlg.temperature.value() == pytest.approx(0.1) and dlg.look_user.toPlainText() == ai_settings.LOOK_USER


def test_look_for_dialog_scopes_and_counts(qapp, window, tmp_path):
    from rulz.ui.ai_panel import AIJobDialog

    ids = _library(qapp, window, tmp_path, ["red", "green", "blue"])
    db = window.db
    sid = db.ai_search_id("red sweater", create=True)
    db.ai_save_answers(sid, [(ids[0], db.one("SELECT fp FROM photos WHERE id=?", (ids[0],))[0], True, "m")])
    dlg = AIJobDialog(window, window, "look", selected=ids[:2], shown=ids[:2])
    texts = [rb.text() for rb, _ in dlg.scopes]
    assert texts == ["Selected photos (2)", "Photos shown (2)", "All photos (3)"]
    assert not dlg.ok.isEnabled()
    dlg.text.setText("Red Sweater, dog")
    dlg._update_info()
    assert "1 of 2 already checked (1 found), 1 to check" in dlg.info.text() and "“dog”: 2 photos" in dlg.info.text()
    job = dlg.job()
    assert job.kind == "look" and job.queries == ["red sweater", "dog"] and job.photo_ids == ids[:2]
    dlg.scopes[2][0].setChecked(True)
    dlg.force.setChecked(True)
    assert dlg.job().photo_ids is None and dlg.job().force and "6 photo checks" in dlg.info.text()
    assert dlg.saved_list.count() == 1 and "found in 1 of 1" in dlg.saved_list.item(0).text()

    tag = AIJobDialog(window, window, "tag", current=ids[0])
    assert [rb.text() for rb, _ in tag.scopes] == ["This photo", "All photos (3)"]
    assert "1 photo to tag" in tag.info.text() and tag.job().photo_ids == [ids[0]]


def test_look_for_from_photos_page_shows_matches(qapp, window, tmp_path, stub, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from rulz.ai.jobs import Job
    from rulz.ui import main_window

    ids = _library(qapp, window, tmp_path, ["red", "green", "red", "blue"])
    _set_own(window, stub.url)

    class FakeDialog:
        nothing_to_do = False

        def __init__(self, parent, ctx, kind, **kw):
            self.kw = kw

        def exec(self):
            return QDialog.Accepted

        def job(self):
            return Job("look", photo_ids=None, queries=["red sweater"])

    monkeypatch.setattr(main_window, "AIJobDialog", FakeDialog)
    window.navigate(window.PHOTOS)
    window.photos._ai("look")
    assert window.photos.search.text() == "“red sweater”" and window.ai_box.isVisible()
    _wait(qapp, lambda: not window.ai.running)
    _wait(qapp, lambda: not window.photos._search_timer.isActive())
    qapp.processEvents()
    shown = {t.key for t in window.photos.grid.flat}
    red = {i for i in ids if "red" in window.db.photo(i).path}
    assert shown == red and len(red) == 2
    assert "found in 2 photos" in window.ai_status.text() and window.ai_show_btn.isVisibleTo(window.ai_box)

    # the viewer lists the answer, and tags once the photo is tagged
    window.db.ai_save_tags([(ids[0], window.db.one("SELECT fp FROM photos WHERE id=?", (ids[0],))[0],
                             ["red sweater", "plain background"], "vis-model")])
    window.open_viewer([ids[0]], 0)
    v = window.viewer
    assert v.ai_section.isVisibleTo(v.info)
    assert "plain background" in v.ai_tags.text() and v.ai_tag_btn.text() == "Tag again"
    v._search_tag("plain background")
    assert window.photos.search.text() == "“plain background”"

    # everything checked before: "Show results" starts no job (no server, nothing sent) and shows them
    FakeDialog.nothing_to_do = True
    monkeypatch.setattr(window.ai, "start", lambda job: pytest.fail("a job started for nothing"))
    window.photos.search.setText("")
    window.photos._ai("look")
    assert window.photos.search.text() == "“red sweater”"


def test_busy_and_remote_checks(qapp, window, monkeypatch):
    from rulz.ai.jobs import Job
    from rulz.ui import dialogs

    infos, confirms, started = [], [], []
    monkeypatch.setattr(dialogs, "info", lambda parent, title, text: infos.append(title))
    monkeypatch.setattr(window.ai, "start", lambda job: started.append(job) or True)
    assert not window.run_ai(Job("tag"))  # AI is off
    assert infos == ["Set up AI first"] and window.pages.currentWidget() is window.library
    _set_own(window, "http://10.0.0.5:11434/v1")
    monkeypatch.setattr(dialogs, "confirm", lambda *a, **k: confirms.append(1) or False)
    assert not window.run_ai(Job("tag")) and not started
    monkeypatch.setattr(dialogs, "confirm", lambda *a, **k: confirms.append(1) or True)
    assert window.run_ai(Job("tag")) and window.run_ai(Job("tag"))
    assert len(confirms) == 2  # asked once, then remembered until the settings change
    window.ai_settings_changed()
    window.run_ai(Job("tag"))
    assert len(confirms) == 3
