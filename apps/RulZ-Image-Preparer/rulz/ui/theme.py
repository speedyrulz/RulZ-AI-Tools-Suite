"""Colors, stylesheet and icon rendering (RulZ look: black + logo red #ED000D, or white + the same red)."""
from __future__ import annotations

import os
from functools import lru_cache

from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QIconEngine, QPainter, QPalette, QPixmap

# surface = panel (bars, preview panel), card = panel-2 (raised: buttons, menus, tiles), input = panel,
# select = accent-soft blended over panel, placeholder = progress/slider track.
DARK = dict(
    bg="#050505", surface="#0c0c0e", card="#141417", border="#232328", border_strong="#34343b",
    text="#f2f2f3", muted="#a3a3ab", accent="#ed000d", accent_hover="#ff2633", accent_pressed="#c8000b",
    on_accent="#ffffff", accent_text="#ff6b72",
    select="#2d0b0e", hover="#1c1c21", danger="#ff5a5f", placeholder="#191919", input="#0c0c0e",
)
LIGHT = dict(
    bg="#ffffff", surface="#f5f5f7", card="#ebebef", border="#d9d9de", border_strong="#c4c4cb",
    text="#111114", muted="#5f6068", accent="#ed000d", accent_hover="#c8000b", accent_pressed="#a8000a",
    on_accent="#ffffff", accent_text="#c8000b",
    select="#f7e5e7", hover="#e2e2e7", danger="#d4161c", placeholder="#ebebeb", input="#f5f5f7",
)

C: dict = dict(DARK)  # active palette, filled by apply()

# Segoe Fluent Icons / Segoe MDL2 Assets code points
GLYPH = dict(
    add="", folder="", back="", prev="", next="", save="",
    delete="", clear="", image="", reset="", fit="", zoom="", crop="",
)


GLYPH.update(sun=chr(0xE706), moon=chr(0xE708), check=chr(0xE73E))


@lru_cache(maxsize=1)
def icon_font_family() -> str:
    families = set(QFontDatabase.families())
    for fam in ("Segoe Fluent Icons", "Segoe MDL2 Assets"):
        if fam in families:
            return fam
    return "Segoe UI Symbol"


@lru_cache(maxsize=256)
def icon_pixmap(name: str, color: str, size: int = 20) -> QPixmap:
    dpr = 2.0
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.TextAntialiasing)
    f = QFont(icon_font_family())
    f.setPixelSize(int(size * 0.8))
    p.setFont(f)
    p.setPen(QColor(color))
    p.drawText(QRectF(0, 0, size, size), Qt.AlignCenter, GLYPH.get(name, name))
    p.end()
    return pm


class _GlyphEngine(QIconEngine):
    """A glyph icon that follows the active theme: drawn in the text colour at paint time (or in a fixed
    colour when one is given), so a live dark/light switch needs no icon rebuild."""

    def __init__(self, name: str, color: str | None, size: int):
        super().__init__()
        self.name, self.color, self.size = name, color, size

    def _pm(self, side: int, mode) -> QPixmap:
        color = self.color or (C["muted"] if mode == QIcon.Disabled else C["text"])
        return icon_pixmap(self.name, color, max(8, side))

    def paint(self, painter, rect: QRect, mode, state) -> None:
        painter.drawPixmap(rect, self._pm(rect.width(), mode))

    def pixmap(self, size: QSize, mode, state) -> QPixmap:
        pm = QPixmap(size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        self.paint(p, QRect(0, 0, size.width(), size.height()), mode, state)
        p.end()
        return pm

    def clone(self):
        return _GlyphEngine(self.name, self.color, self.size)


def icon(name: str, color: str | None = None, size: int = 20) -> QIcon:
    return QIcon(_GlyphEngine(name, color, size))


def is_dark() -> bool:
    return C.get("bg") == DARK["bg"]


@lru_cache(maxsize=1)
def ui_family() -> str:
    """Segoe UI Variable Text (Windows 11), else Segoe UI."""
    families = set(QFontDatabase.families())
    return "Segoe UI Variable Text" if "Segoe UI Variable Text" in families else "Segoe UI"


@lru_cache(maxsize=1)
def display_family() -> str:
    """Titles: Segoe UI Variable Display, else Segoe UI."""
    families = set(QFontDatabase.families())
    return "Segoe UI Variable Display" if "Segoe UI Variable Display" in families else "Segoe UI"


THEME_KEY = "theme"


def env_override() -> bool | None:
    """RULZ_THEME=light|dark forces a theme (not saved)."""
    return {"dark": True, "light": False}.get(os.environ.get("RULZ_THEME", "").strip().lower())


def saved_choice() -> bool | None:
    """True = dark, False = light, None = never chosen (first run)."""
    from .. import config

    value = str(config.settings().value(THEME_KEY, "") or "").lower()
    return {"dark": True, "light": False}.get(value)


def save_choice(dark: bool) -> None:
    from .. import config

    s = config.settings()
    s.setValue(THEME_KEY, "dark" if dark else "light")
    s.sync()


def startup_dark() -> bool:
    """Dark unless RULZ_THEME or a saved choice says light."""
    forced = env_override()
    if forced is not None:
        return forced
    saved = saved_choice()
    return True if saved is None else saved


def logo_file() -> str:
    """Full RulZ logo: white lettering on dark, the original black lettering on light."""
    from .. import config

    return str(config.resource_path("assets", "logo_full_white.png" if is_dark() else "logo_full.png"))


def apply(app, dark: bool | None = None) -> None:
    if dark is None:
        dark = startup_dark()
    C.clear()
    C.update(DARK if dark else LIGHT)
    try:  # native title bar / Fusion follow the app's choice instead of the Windows setting
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light)
    except Exception:
        pass
    app.setStyle("Fusion")
    pal = QPalette()
    for role, key in (
        (QPalette.Window, "bg"), (QPalette.WindowText, "text"), (QPalette.Base, "surface"),
        (QPalette.AlternateBase, "card"), (QPalette.Text, "text"), (QPalette.Button, "card"),
        (QPalette.ButtonText, "text"), (QPalette.Highlight, "accent"),
        (QPalette.HighlightedText, "on_accent"), (QPalette.ToolTipBase, "card"),
        (QPalette.ToolTipText, "text"), (QPalette.PlaceholderText, "muted"), (QPalette.Mid, "border"),
    ):
        pal.setColor(role, QColor(C[key]))
    app.setPalette(pal)
    f = QFont(ui_family())
    f.setPointSizeF(10.5)
    app.setFont(f)
    app.setStyleSheet(stylesheet())
    for w in app.allWidgets():  # live switch: repaint what is already on screen
        w.update()


def _check_image() -> str:
    """White check mark for checked QCheckBox indicators (stylesheets need an image file)."""
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "rulz-image-preparer-check.png")
    if not os.path.exists(path):
        icon_pixmap("check", "#ffffff", 14).save(path)
    return path.replace("\\", "/")


def stylesheet() -> str:
    return """
QWidget { color: %(text)s; }
QMainWindow, #Page { background: %(bg)s; }
#TopBar, #ToolBar { background: %(surface)s; border-bottom: 1px solid %(border)s; }
#BottomBar { background: %(surface)s; border-top: 1px solid %(border)s; }
#PageTitle { font-family: "%(display)s"; font-size: 18px; font-weight: 600; }
#H2 { font-family: "%(display)s"; font-size: 14px; font-weight: 600; }
#Muted { color: %(muted)s; }
#Value { font-size: 14px; font-weight: 700; color: %(accent_text)s; min-width: 34px; }
QSpinBox#Value { background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 2px 4px; min-height: 0px; }
QSpinBox#Value:hover { border-color: %(border)s; background: %(input)s; }
QSpinBox#Value:focus { border: 2px solid %(accent)s; padding: 1px 3px; background: %(input)s; color: %(text)s; }
#DropHint { color: %(muted)s; font-size: 15px; border: 2px dashed %(border_strong)s; border-radius: 12px; }
#DropHint[active="true"] { border-color: %(accent)s; color: %(text)s; background: %(select)s; }

QPushButton {
    background: %(card)s; border: 1px solid %(border)s; border-radius: 8px;
    padding: 7px 14px; font-weight: 500; min-height: 18px;
}
QPushButton:hover { background: %(hover)s; border-color: %(border_strong)s; }
QPushButton:pressed { background: %(select)s; }
QPushButton:focus { border: 2px solid %(accent)s; padding: 6px 13px; }
QPushButton:disabled { color: %(muted)s; }
QPushButton[primary="true"] { background: %(accent)s; color: %(on_accent)s; border: 1px solid %(accent)s; }
QPushButton[primary="true"]:hover { background: %(accent_hover)s; border-color: %(accent_hover)s; }
QPushButton[primary="true"]:pressed { background: %(accent_pressed)s; border-color: %(accent_pressed)s; }
QPushButton[primary="true"]:focus { border: 2px solid %(text)s; padding: 6px 13px; }
QPushButton[primary="true"]:disabled { background: %(placeholder)s; border-color: %(border)s; color: %(muted)s; }
QPushButton[flat="true"] { background: transparent; border: none; border-radius: 6px; color: %(muted)s; }
QPushButton[flat="true"]:hover { background: %(hover)s; color: %(text)s; }
QPushButton[flat="true"]:focus { border: 2px solid %(accent)s; padding: 5px 12px; }

#Segment { background: %(input)s; border: 1px solid %(border)s; border-radius: 10px; }
#Segment QPushButton { border: 2px solid transparent; border-radius: 8px; padding: 5px 13px; background: transparent; font-weight: 600; }
#Segment QPushButton:checked { background: %(card)s; color: %(accent_text)s; }
#Segment QPushButton:focus { border: 2px solid %(accent)s; }

QListWidget { background: %(bg)s; border: 1px solid transparent; outline: none; }
QListWidget:focus { border: 1px solid %(border_strong)s; }

QSlider::groove:horizontal { height: 4px; background: %(placeholder)s; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::handle:horizontal { background: %(accent)s; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }
QSlider::handle:horizontal:hover { background: %(accent_hover)s; }
QSlider::handle:horizontal:focus { background: %(on_accent)s; border: 3px solid %(accent)s; width: 10px; height: 10px; }

QMenu { background: %(card)s; border: 1px solid %(border)s; border-radius: 8px; padding: 4px; }
QMenu::item { padding: 7px 26px 7px 12px; border-radius: 6px; }
QMenu::item:selected { background: %(hover)s; }
QMenu::item:disabled { color: %(muted)s; }

QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar::handle:vertical { background: %(border_strong)s; border-radius: 4px; min-height: 40px; margin: 0 2px; }
QScrollBar::handle:vertical:hover { background: %(muted)s; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QProgressDialog, QDialog, QMessageBox { background: %(bg)s; }
QProgressBar {
    background: %(placeholder)s; border: none; border-radius: 9px; min-height: 18px; max-height: 18px;
    text-align: center; color: %(text)s; font-size: 11px; font-weight: 600;
}
QProgressBar::chunk { background: %(accent)s; border-radius: 9px; }
QToolTip { background: %(card)s; color: %(text)s; border: 1px solid %(border)s; border-radius: 8px; padding: 4px 8px; }

QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
    background: %(input)s; border: 1px solid %(border)s; border-radius: 8px; padding: 5px 8px; min-height: 18px;
}
/* Every spin box in the app has no arrow buttons; without this, the styled box still reserves room for
   them and the text field collapses to nothing. The inner text field also mustn't get the QLineEdit box. */
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 0px; height: 0px; border: 0; image: none; subcontrol-origin: border;
}
QAbstractSpinBox QLineEdit { background: transparent; border: none; padding: 0px 7px; min-height: 0px; }
QSpinBox#Value QLineEdit { padding: 0px; }
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus { border: 2px solid %(accent)s; padding: 4px 7px; }
QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled { color: %(muted)s; }
QComboBox QAbstractItemView { background: %(card)s; border: 1px solid %(border)s; border-radius: 8px;
                              selection-background-color: %(select)s; selection-color: %(text)s; outline: 0; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid %(border_strong)s; border-radius: 4px;
                       background: %(card)s; }
QCheckBox::indicator:hover { border-color: %(muted)s; }
QCheckBox::indicator:focus { border: 2px solid %(accent)s; width: 14px; height: 14px; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(accent)s; image: url(%(check_img)s); }
QCheckBox::indicator:checked:focus { border: 2px solid %(text)s; width: 14px; height: 14px; }
#Section { color: %(muted)s; font-weight: 600; padding-top: 10px; }
#PreviewPanel { background: %(surface)s; border: 1px solid %(border)s; border-radius: 12px; }
""" % dict(C, display=display_family(), check_img=_check_image())
