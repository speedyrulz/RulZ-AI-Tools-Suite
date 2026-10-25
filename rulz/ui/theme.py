"""Colors, stylesheet and icon rendering."""
from __future__ import annotations

import os
from functools import lru_cache

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap

# RulZ brand: logo red #ED000D. Dark (black + red) is the default look; light is white + the same red.
BRAND_RED = "#ed000d"
DARK = dict(
    bg="#0b0b0c", surface="#111113", card="#141416", border="#232328",
    text="#f2f2f3", muted="#a3a3ab", accent="#ed000d", accent_hover="#ff1f2d", on_accent="#ffffff",
    select="#3a0c10", hover="#1c1c21", danger="#ff4d58", placeholder="#1f1f24", input="#18181c",
    sidebar="#0c0c0e", side_border="#1e1e22", side_text="#ececef", side_muted="#8c8c95",
    side_hover="#1c1c21", side_select="#2b0a0d", side_select_text="#ff2b38",
)
LIGHT = dict(
    bg="#ffffff", surface="#f5f5f7", card="#ffffff", border="#d9d9de",
    text="#111114", muted="#5f6068", accent="#ed000d", accent_hover="#c8000b", on_accent="#ffffff",
    select="#fde6e7", hover="#ebebef", danger="#c8000b", placeholder="#ebebef", input="#f5f5f7",
    sidebar="#f5f5f7", side_border="#d9d9de", side_text="#111114", side_muted="#5f6068",
    side_hover="#ebebef", side_select="#fde6e7", side_select_text="#c8000b",
)

C: dict = dict(DARK)  # active palette, filled by apply()

# Segoe Fluent Icons / Segoe MDL2 Assets code points
GLYPH = dict(
    photos="", people="", library="", folder="", open_folder="",
    back="", prev="", next="", edit="", delete="", add="",
    refresh="", search="", info="", hide="", show="", merge="",
    close="", check="", more="", person="", add_person="",
    block="", stop="", face="", star="", settings="", tag="",
    box="", sun="", moon="",
    filter="",
)


@lru_cache(maxsize=1)
def icon_font_family() -> str:
    families = set(QFontDatabase.families())
    for fam in ("Segoe Fluent Icons", "Segoe MDL2 Assets"):
        if fam in families:
            return fam
    return "Segoe UI Symbol"


def icon(name: str, color: str | None = None, size: int = 20) -> QIcon:
    return QIcon(icon_pixmap(name, color or C["text"], size))


@lru_cache(maxsize=512)
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


def is_dark() -> bool:
    return C.get("bg") == DARK["bg"]


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
    forced = env_override()
    if forced is not None:
        return forced
    saved = saved_choice()
    return True if saved is None else saved


def logo_file() -> str:
    """Full RulZ logo: white lettering on dark, the original black lettering on light."""
    from .. import config

    return str(config.resource_path("assets", "logo_full_white.png" if is_dark() else "logo_full.png"))


# ---- live re-theming -------------------------------------------------------
# Widgets whose icon or inline style uses palette colors register here so a theme switch can redo them.
_ICON_PROP = "_rulzThemeIcon"
_STYLE_PROP = "_rulzThemeStyle"


def set_icon(widget, name: str, token: str = "text", size: int = 20) -> None:
    """Set a glyph icon whose color is a palette token (e.g. "text", "muted", "accent")."""
    widget.setProperty(_ICON_PROP, f"{name}|{token}|{size}")
    widget.setIcon(icon(name, C[token], size))


def set_style(widget, template: str) -> None:
    """Set an inline stylesheet using %(token)s palette placeholders."""
    widget.setProperty(_STYLE_PROP, template)
    widget.setStyleSheet(template % C)


def _icon_from_spec(spec: str) -> QIcon:
    name, token, size = spec.split("|")
    return icon(name, C[token], int(size))


def retheme_widgets(root) -> None:
    from PySide6.QtWidgets import QWidget

    for w in [root, *root.findChildren(QWidget)]:
        spec = w.property(_ICON_PROP)
        if spec:
            w.setIcon(_icon_from_spec(spec))
        tpl = w.property(_STYLE_PROP)
        if tpl:
            w.setStyleSheet(tpl % C)
        for a in QWidget.actions(w):  # some widgets shadow .actions with a layout
            spec = a.property(_ICON_PROP)
            if spec:
                a.setIcon(_icon_from_spec(spec))
        w.update()


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
    pal.setColor(QPalette.Window, QColor(C["bg"]))
    pal.setColor(QPalette.WindowText, QColor(C["text"]))
    pal.setColor(QPalette.Base, QColor(C["card"]))
    pal.setColor(QPalette.AlternateBase, QColor(C["surface"]))
    pal.setColor(QPalette.Text, QColor(C["text"]))
    pal.setColor(QPalette.Button, QColor(C["card"]))
    pal.setColor(QPalette.ButtonText, QColor(C["text"]))
    pal.setColor(QPalette.Highlight, QColor(C["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor(C["on_accent"]))
    pal.setColor(QPalette.ToolTipBase, QColor(C["card"]))
    pal.setColor(QPalette.ToolTipText, QColor(C["text"]))
    pal.setColor(QPalette.PlaceholderText, QColor(C["muted"]))
    pal.setColor(QPalette.Mid, QColor(C["border"]))
    app.setPalette(pal)
    f = QFont("Segoe UI Variable Text")
    if f.family() not in QFontDatabase.families():
        f = QFont("Segoe UI")
    f.setPointSizeF(10)
    app.setFont(f)
    app.setStyleSheet(stylesheet())


def _check_image() -> str:
    """White check mark for checked QCheckBox indicators (stylesheets need an image file)."""
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "rulz-image-organizer-check.png")
    if not os.path.exists(path):
        icon_pixmap("check", "#ffffff", 14).save(path)
    return path.replace("\\", "/")


def stylesheet() -> str:
    return """
QWidget { color: %(text)s; }
QMainWindow, #Content, #Page { background: %(bg)s; }
#Sidebar { background: %(sidebar)s; border-right: 1px solid %(side_border)s; }
#Sidebar QLabel { color: %(side_text)s; }
#BrandSub { color: %(side_muted)s; font-size: 11px; font-weight: 600; letter-spacing: 2px; padding: 0 2px; }
#SideStatus { color: %(side_muted)s; font-size: 12px; padding: 0 6px; }
#Sidebar QProgressBar { background: %(side_hover)s; }
#Sidebar QProgressBar::chunk { background: %(brand)s; }
#ThemeButton { text-align: left; padding: 7px 14px; border: none; border-radius: 10px; font-size: 13px;
               background: transparent; color: %(side_muted)s; }
#ThemeButton:hover { background: %(side_hover)s; color: %(side_text)s; }
#NavButton {
    text-align: left; padding: 9px 14px; border: none; border-radius: 10px;
    font-size: 14px; background: transparent; color: %(side_text)s;
}
#NavButton:hover { background: %(side_hover)s; }
#NavButton:checked { background: %(side_select)s; color: %(side_select_text)s; font-weight: 600; }
#PageTitle { font-size: 24px; font-weight: 700; }
#H2 { font-size: 16px; font-weight: 600; }
#Muted, QLabel[muted="true"] { color: %(muted)s; }
#Card { background: %(card)s; border: 1px solid %(border)s; border-radius: 12px; }
#SelectionBar { background: %(select)s; border-radius: 12px; }
#SelectionBar QLabel { font-weight: 600; }

QPushButton {
    background: %(card)s; border: 1px solid %(border)s; border-radius: 8px;
    padding: 7px 14px; font-weight: 500;
}
QPushButton:hover { background: %(hover)s; }
QPushButton:pressed, QPushButton:checked { background: %(select)s; }
QPushButton:disabled { color: %(muted)s; }
QPushButton::menu-indicator { image: none; width: 0; }
QPushButton[primary="true"] { background: %(accent)s; color: %(on_accent)s; border: 1px solid %(accent)s; }
QPushButton[primary="true"]:hover { background: %(accent_hover)s; }
QPushButton[flat="true"] { background: transparent; border: none; }
QPushButton[flat="true"]:hover { background: %(hover)s; }
QPushButton[danger="true"] { color: %(danger)s; }
QToolButton { border: none; border-radius: 18px; padding: 6px; background: transparent; }
QToolButton:hover { background: %(hover)s; }
QToolButton:checked { background: %(select)s; }
QToolButton::menu-indicator { image: none; }

QLineEdit {
    background: %(input)s; border: 1px solid transparent; border-radius: 16px;
    padding: 7px 14px; font-size: 13px; selection-background-color: %(accent)s;
}
QLineEdit:focus { border: 1px solid %(accent)s; background: %(card)s; }
QLineEdit[plain="true"] { border-radius: 8px; }

QListWidget { background: %(card)s; border: 1px solid %(border)s; border-radius: 10px; padding: 4px; outline: none; }
QListWidget::item { padding: 6px; border-radius: 8px; }
QListWidget::item:hover { background: %(hover)s; }
QListWidget::item:selected { background: %(select)s; color: %(text)s; }

QProgressBar { background: %(placeholder)s; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; }
QProgressBar::chunk { background: %(accent)s; border-radius: 3px; }

QSlider::groove:horizontal { height: 4px; background: %(placeholder)s; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::handle:horizontal { background: %(accent)s; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }


QMenu { background: %(card)s; border: 1px solid %(border)s; border-radius: 10px; padding: 6px; }
QMenu::item { padding: 7px 26px 7px 12px; border-radius: 6px; }
QMenu::item:selected { background: %(hover)s; }
QMenu::item:disabled { color: %(muted)s; }
QMenu::separator { height: 1px; background: %(border)s; margin: 5px 8px; }
QMenu::icon { padding-left: 10px; }

QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar::handle:vertical { background: %(border)s; border-radius: 4px; min-height: 40px; margin: 0 2px; }
QScrollBar::handle:vertical:hover { background: %(muted)s; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QScrollBar:horizontal { height: 0; }

QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid %(muted)s; border-radius: 4px;
                       background: %(card)s; }
QCheckBox::indicator:hover { border-color: %(accent)s; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(accent)s; image: url(%(check_img)s); }

QDialog { background: %(bg)s; }
QToolTip { background: %(card)s; color: %(text)s; border: 1px solid %(border)s; padding: 4px 8px; }

#Chip { background: %(input)s; border: 1px solid %(border)s; border-radius: 13px; padding: 4px 12px;
        font-weight: 500; }
#Chip:hover { background: %(hover)s; }
#Chip:checked { background: %(accent)s; color: %(on_accent)s; border-color: %(accent)s; }
#Segment { background: %(input)s; border-radius: 12px; }
#Segment QPushButton { border: none; border-radius: 9px; padding: 6px 16px; background: transparent; }
#Segment QPushButton:checked { background: %(card)s; color: %(accent)s; font-weight: 600; }

/* Photo viewer – always dark */
#Viewer { background: #0a0a0b; }
#Viewer QLabel { color: #f2f3f7; }
#ViewerBar QToolButton { color: #f2f3f7; }
#ViewerBar QToolButton:hover { background: rgba(255,255,255,0.12); }
#ViewerBar QToolButton:checked { background: rgba(238,1,13,0.45); }
#InfoPanel { background: %(card)s; border-left: 1px solid %(border)s; }
#InfoPanel QLabel { color: %(text)s; }
#FaceRow { border-radius: 10px; }
#FaceRow:hover { background: %(hover)s; }
""" % dict(C, brand=BRAND_RED, check_img=_check_image())
