"""Theme: tokens, settings, live switching and the page opener wiring. Offscreen Qt; no WSL, no GPU."""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import json
import threading
import time

import pytest
from PySide6.QtCore import QEvent, QCoreApplication
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

from rulz_comfy_launcher.command import launch_signature
from rulz_comfy_launcher.instance import State
from rulz_comfy_launcher.model import (THEME_DARK, THEME_LIGHT, ConfigStore, Profile, Settings)
from rulz_comfy_launcher.ui import theme

import ui_support

GUIDE_TOKENS = [
    "bg", "panel", "panel-2", "hover", "border", "border-strong", "text", "body", "muted", "faint",
    "accent", "accent-hover", "accent-soft", "accent-fg", "accent-text", "link", "danger",
    "danger-text", "danger-bg", "danger-border", "danger-hover", "warn", "warn-text", "ok",
    "banner-bg", "banner-text", "code-bg", "sunken", "subtle", "subtle-border", "track",
    "media-bg", "scrim", "shadow",
]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def real_fonts(app):
    """Widths and heights only mean something with the real UI font."""
    from PySide6.QtGui import QFontDatabase
    if "Segoe UI" not in QFontDatabase.families():
        pytest.skip("Segoe UI is not available here")


@pytest.fixture
def schemes(monkeypatch):
    """What the theme asked Qt for as the colour scheme (the title bar follows it on Windows)."""
    calls = []
    monkeypatch.setattr(theme, "_set_color_scheme", lambda app, dark: calls.append("dark" if dark else "light"))
    return calls


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    threads = set(threading.enumerate())
    win = ui_support.make_window(tmp_path, monkeypatch, gpus=ui_support.sample_gpus())
    win.show()
    app.processEvents()
    yield win
    ui_support.close_window(win)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    theme.apply_theme(app, theme.DARK)
    time.sleep(0.05)
    assert set(threading.enumerate()) <= threads, "a test left a thread running"


# ---------------------------------------------------------------- tokens
def test_tokens_complete_for_both_themes():
    assert sorted(theme.TOKENS) == sorted(GUIDE_TOKENS)
    for name in GUIDE_TOKENS:
        dark, light = theme.TOKENS[name]
        assert dark and light
        assert theme.qcolor(dark).isValid() and theme.qcolor(light).isValid()
    assert theme.palette_for(theme.DARK)["bg"] == "#050505"
    assert theme.palette_for(theme.LIGHT)["bg"] == "#ffffff"
    assert theme.palette_for(theme.DARK)["accent"] == theme.palette_for(theme.LIGHT)["accent"] == "#ed000d"
    assert theme.palette_for(theme.LIGHT)["accent-hover"] == "#c8000b"
    assert theme.palette_for(theme.DARK)["accent-soft"] == "rgba(237, 0, 13, 0.13)"
    assert theme.palette_for(theme.LIGHT)["sunken"] == "rgba(0, 0, 0, 0.03)"


def test_rgba_tokens_become_qt_alpha():
    assert theme.qss("rgba(237, 0, 13, 0.13)") == "rgba(237, 0, 13, 33)"
    assert theme.qcolor("rgba(0, 0, 0, 0.9)").alpha() == round(0.9 * 255)
    assert theme.qss("#ed000d") == "#ed000d"


def _luminance(color):
    def channel(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel(color[0]) + 0.7152 * channel(color[1]) + 0.0722 * channel(color[2])


def _over(top, base):
    c = theme.qcolor(top)
    b = theme.qcolor(base)
    a = c.alphaF()
    return tuple(c.getRgb()[i] * a + b.getRgb()[i] * (1 - a) for i in range(3))


def _contrast(fg, bg):
    hi, lo = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("name", [theme.DARK, theme.LIGHT])
def test_text_and_status_colours_are_readable(name):
    p = theme.palette_for(name)
    for ground in ("bg", "panel", "panel-2"):
        base = theme.qcolor(p[ground]).getRgb()[:3]
        for token in ("text", "body", "muted", "faint"):
            assert _contrast(theme.qcolor(p[token]).getRgb()[:3], base) >= 4.5, (name, token, ground)
    # the status pill: its text on its neutral fill over the window background
    pill = _over(p["subtle"], p["bg"])
    for tone in (theme.TONE_STOPPED, theme.TONE_BUSY, theme.TONE_OK, theme.TONE_FAILED, theme.TONE_EXTERNAL):
        text = theme.qcolor(p[theme._TONE_TEXT_TOKEN[tone]]).getRgb()[:3]
        assert _contrast(text, pill) >= 4.5, (name, tone)
    # primary button: accent-fg on the accent fill (the guide states 4.6:1)
    assert _contrast((255, 255, 255), theme.qcolor(p["accent"]).getRgb()[:3]) >= 4.5


def test_every_state_has_a_tone_and_no_colour_constant_is_kept():
    for state in State:
        assert theme.STATE_TONES[state] in theme._TONE_TOKEN
    assert not hasattr(theme, "STATE_COLORS")
    assert not hasattr(theme, "EXTERNAL_COLOR")


# ---------------------------------------------------------------- settings
def test_settings_theme_default_invalid_and_roundtrip(tmp_path):
    assert Settings().theme == THEME_DARK
    assert Settings.from_dict({}).theme == THEME_DARK
    for bad in ("blue", "", None, 3, ["light"], "Light"):
        assert Settings.from_dict({"theme": bad}).theme == THEME_DARK
    assert Settings.from_dict({"theme": "light"}).theme == THEME_LIGHT
    store = ConfigStore(tmp_path / "c.json")
    store.load()
    store.settings.theme = THEME_LIGHT
    store.save()
    other = ConfigStore(tmp_path / "c.json")
    other.load()
    assert other.settings.theme == THEME_LIGHT
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["settings"]["theme"] == "light"


def test_logo_files():
    assert theme.logo_file(theme.DARK).name == "logo_full_white.png"
    assert theme.logo_file(theme.LIGHT).name == "logo_full_black.png"
    assert theme.logo_file(theme.DARK).is_file() and theme.logo_file(theme.LIGHT).is_file()


# ---------------------------------------------------------------- live switching
def test_window_starts_in_saved_theme(app, tmp_path, monkeypatch, schemes):
    theme.apply_theme(app, theme.DARK)
    schemes.clear()
    win = ui_support.make_window(tmp_path, monkeypatch, settings={"theme": THEME_LIGHT})
    try:
        assert theme.mode() == theme.LIGHT
        assert app.palette().color(QPalette.ColorRole.Window).name() == "#ffffff"
        assert schemes == ["light"]
    finally:
        ui_support.close_window(win)
        theme.apply_theme(app, theme.DARK)


def test_switch_updates_palette_logo_widgets_and_persists(app, window, schemes):
    assert theme.mode() == theme.DARK
    assert app.palette().color(QPalette.ColorRole.Window).name() == "#050505"
    dark_logo = window.logo.pixmap().toImage()
    log = window._ensure_log(window._current.id)
    log.feed("line\n")
    assert log.styleSheet() == "" and window.pill.styleSheet() == ""   # nothing keeps a stale colour
    dot = window._rows[window._current.id].dot
    dark_dot = theme.tone_colour(dot.tone()).name()
    dark_icon = window.theme_button.icon().pixmap(16, 16).toImage()
    dark_pill = window.pill._icon.pixmap().toImage()

    window.set_theme("light")

    assert theme.mode() == theme.LIGHT
    assert window.store.settings.theme == "light"
    saved = json.loads(window.store.path.read_text(encoding="utf-8"))
    assert saved["settings"]["theme"] == "light"          # written at once, not on exit
    assert app.palette().color(QPalette.ColorRole.Window).name() == "#ffffff"
    assert app.palette().color(QPalette.ColorRole.Text).name() == "#111114"
    assert schemes == ["light"]                            # the title bar follows the theme
    assert "#ffffff" in app.styleSheet() and "#050505" not in app.styleSheet()
    assert window.logo.pixmap().toImage() != dark_logo
    assert theme.tone_colour(dot.tone()).name() != dark_dot
    assert window.theme_button.text() == "Dark mode"
    assert window.theme_button.icon().pixmap(16, 16).toImage() != dark_icon
    assert window.pill._icon.pixmap().toImage() != dark_pill
    app.processEvents()
    assert window.grab().toImage().pixelColor(700, 600) != theme.qcolor("#050505")

    window.toggle_theme()
    assert theme.mode() == theme.DARK
    assert window.store.settings.theme == "dark"
    assert json.loads(window.store.path.read_text(encoding="utf-8"))["settings"]["theme"] == "dark"
    assert schemes == ["light", "dark"]
    assert window.logo.pixmap().toImage() == dark_logo


def test_status_shapes_follow_the_theme_and_say_more_than_colour(window):
    pill = window.pill
    colours = {}
    for tone in (theme.TONE_STOPPED, theme.TONE_OK, theme.TONE_FAILED, theme.TONE_EXTERNAL, theme.TONE_BUSY):
        pill.set_status(tone, tone)
        colours[tone] = pill._icon.pixmap().toImage()
    # a different icon shape for each, so the status never depends on hue alone
    assert len({img.constBits().tobytes() for img in colours.values()}) == len(colours)
    pill.set_status("Running", theme.TONE_OK)
    before = pill._icon.pixmap().toImage()
    theme.apply_theme(QApplication.instance(), theme.LIGHT)
    assert pill._icon.pixmap().toImage() != before
    assert pill.text() == "Running"


def test_one_primary_button_in_the_view(window):
    start, open_ = window.start_button, window.open_button
    assert start.property("primary") is True and not open_.property("primary")
    assert not window.start_checked.property("primary")
    # a running configuration: Start is off, Open in browser becomes the primary button
    window.manager.state = lambda profile_id: State.RUNNING
    window._refresh_header()
    assert open_.property("primary") is True and start.property("primary") is False
    del window.manager.state
    window._refresh_header()
    assert start.property("primary") is True and open_.property("primary") is False


def test_no_focus_ring_at_start(app, window):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.focusWidget() in (None, window)


def test_buttons_take_focus_from_tab_only(app, window):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QPushButton
    app.processEvents()
    for button in window.findChildren(QPushButton):
        assert button.focusPolicy() in (Qt.FocusPolicy.TabFocus, Qt.FocusPolicy.NoFocus), button.text()


def test_control_heights_follow_the_guide(window, real_fonts):
    start = window.start_button
    assert start.sizeHint().height() == theme.CONTROL_MD
    assert window.editor.name.sizeHint().height() == theme.CONTROL_MD
    small = [b for b in window.findChildren(type(start)) if b.property("controlSize") == "sm" and b.isVisible()]
    assert small and all(b.sizeHint().height() == theme.CONTROL_SM for b in small)


# ---------------------------------------------------------------- RulZFlow wiring in the real window
class _RecordingOpener:
    def __init__(self):
        self.calls = []

    def open(self, profile_id, name, server_url, rulzflow, valid):
        self.calls.append((profile_id, name, server_url, rulzflow, valid()))

    def stop(self):
        pass


def test_rulzflow_toggle_shows_in_url_label_and_page_opener(window):
    profile = window._current
    base = "http://127.0.0.1:8188"
    assert window.url_label.text().rstrip("/") == base
    window._page_opener = _RecordingOpener()
    catalog = window.catalog_for(profile)
    signature = launch_signature(profile, catalog)

    window.editor.open_rulzflow.setChecked(True)
    assert profile.open_rulzflow is True
    assert window.url_label.text().endswith("/rulzflow/")
    assert launch_signature(profile, catalog) == signature        # opening a page needs no restart
    window._open_current()
    assert window._page_opener.calls[-1][2:4] == (base, True)
    assert window._page_opener.calls[-1][4] is True

    window.editor.open_rulzflow.setChecked(False)
    assert not window.url_label.text().endswith("/rulzflow/")
    assert launch_signature(profile, catalog) == signature
    window._open_current()
    assert window._page_opener.calls[-1][2:4] == (base, False)


def test_auto_open_uses_the_current_toggle(window):
    profile = window._current
    window.editor.auto_open.setChecked(True)
    assert profile.auto_open is True
    window._page_opener = _RecordingOpener()

    class _Inst:
        pass

    fake = _Inst()
    fake.profile = profile
    fake.url = "http://127.0.0.1:8188"
    fake.active = True
    fake.catalog = None
    fake.signature = launch_signature(profile, window.catalog_for(profile))
    window.manager.instance = lambda pid: fake
    window.editor.open_rulzflow.setChecked(True)
    window._on_ready(profile.id)
    assert window._page_opener.calls[-1][3] is True
    window.editor.open_rulzflow.setChecked(False)
    window._on_ready(profile.id)
    assert window._page_opener.calls[-1][3] is False
    del window.manager.instance


def test_rulzflow_checkboxes_fit_in_the_install_card(app, window, real_fonts):
    window.resize(window.minimumSize())
    window.show()
    app.processEvents()
    editor = window.editor
    for box in (editor.auto_open, editor.open_rulzflow):
        assert box.isVisible()
        need = box.sizeHint().width()
        assert box.width() >= need, (box.text(), box.width(), need)
        assert box.height() >= box.sizeHint().height()
    card = editor.auto_open.parentWidget()
    assert editor.open_rulzflow.geometry().bottom() <= card.height()


def test_window_shows_at_minimum_and_1280_without_horizontal_overflow(app, window, real_fonts):
    from PySide6.QtWidgets import QScrollArea
    for size in (window.minimumSize(), window.size().expandedTo(window.minimumSize())):
        window.resize(size)
        app.processEvents()
        scroll = window.editor.findChild(QScrollArea)
        assert scroll.widget().minimumSizeHint().width() <= scroll.viewport().width() + 1
    window.resize(1280, 900)
    app.processEvents()
    scroll = window.editor.findChild(QScrollArea)
    assert scroll.widget().minimumSizeHint().width() <= scroll.viewport().width()
