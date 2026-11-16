"""RulZ branding: app name, logo assets and the black-and-red theme."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QIcon, QPalette, QPixmap

APP_NAME = "RulZ Simple Video Clip Extractor"
APP_SHORT = "RulZ Clip Extractor"

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
LOGO_DARK = os.path.join(ASSETS, "logo_dark.png")      # white lettering, for dark backgrounds
LOGO_LIGHT = os.path.join(ASSETS, "logo_light.png")    # original colours, transparent background
ICON = os.path.join(ASSETS, "icon256.png")


def _asset_url(name: str) -> str:
    return os.path.join(ASSETS, name).replace("\\", "/")

# palette -------------------------------------------------------------------
# Dark is the original black-and-red look and the default; Light is a white window with black
# text and the same red accent. set_mode() swaps every token below in place, so code that reads
# theme.X / theme.Q_X at paint time always gets the current theme.
DARK = "dark"
LIGHT = "light"

PALETTES = {
    DARK: dict(
        BG="#050505",          # window / video background
        PANEL="#0c0c0e",       # panels, inputs
        RAISED="#141417",      # buttons
        RAISED_HOVER="#1c1c21",
        BORDER="#232328",
        BORDER_STRONG="#34343b",   # hovered borders, unchecked checkbox outline, scrollbar thumb
        TEXT="#f2f2f3",
        MUTED="#a3a3ab",
        RED="#ed000d",         # logo red
        RED_HOVER="#ff2633",
        RED_DARK="#c8000b",    # pressed primary buttons
        PRESSED="#c8000b",
        SELECT="#2d0b0e",      # accent-soft over the panel: pressed secondary buttons, selected rows
        TRACK="#191919",       # progress track
        ACCENT_TEXT="#ff6b72",  # red used as text (section titles, keyframe label)
        BAR="#050505",         # menu bar / status bar
        DIS_TEXT="#55555c", DIS_BG="#0f0f12", DIS_BORDER="#1a1a1e", DIS_INPUT_BG="#0a0a0c",
        DIS_MENU="#5a5a62",
        PRIMARY_DIS_BG="#6d0309", PRIMARY_DIS_TEXT="#b79a9c", PRIMARY_DIS_BORDER="#6d0309",
        INDICATOR_BORDER="#34343b",
        # timeline (QColor accepts #AARRGGBB)
        TL_BAR="#1c1c20", TL_BAR_BORDER="#34343a", TL_TICK="#cdcdd4", TL_MARK="#ffffff",
        TL_INOUT="#be7d0a10", TL_SEL="#78e2121c",
        TL_SCENE="#f0b429",    # scene cuts
        WAVE_BG="#121215", WAVE_PEAK="#5e5e68", WAVE_RMS="#c4c4cc",   # sound strip: soft peak, brighter RMS
        ARROW_SUFFIX="",
    ),
    LIGHT: dict(
        BG="#ffffff",
        PANEL="#f5f5f7",
        RAISED="#ebebef",
        RAISED_HOVER="#e2e2e7",
        BORDER="#d9d9de",
        BORDER_STRONG="#c4c4cb",
        TEXT="#111114",
        MUTED="#5f6068",
        RED="#ed000d",
        RED_HOVER="#c8000b",
        RED_DARK="#a8000a",
        PRESSED="#a8000a",
        SELECT="#f7e5e7",
        TRACK="#ebebeb",
        ACCENT_TEXT="#c8000b",
        BAR="#ffffff",
        DIS_TEXT="#a3a3ab", DIS_BG="#f3f3f5", DIS_BORDER="#e3e3e8", DIS_INPUT_BG="#f7f7f9",
        DIS_MENU="#a3a3ab",
        PRIMARY_DIS_BG="#f6c9cb", PRIMARY_DIS_TEXT="#ffffff", PRIMARY_DIS_BORDER="#f1b4b7",
        INDICATOR_BORDER="#c4c4cb",
        TL_BAR="#ebebef", TL_BAR_BORDER="#c9c9cf", TL_TICK="#5f6068", TL_MARK="#111114",
        TL_INOUT="#55ed000d", TL_SEL="#4ced000d",
        TL_SCENE="#b7791f",
        WAVE_BG="#f3f3f6", WAVE_PEAK="#b9b9c2", WAVE_RMS="#55565e",
        ARROW_SUFFIX="_light",
    ),
}

# Colours drawn on top of the video picture itself (crop box labels) don't follow the theme.
OVERLAY_LABEL_BG = "#7d0a10"

_mode = DARK


def mode() -> str:
    return _mode


def is_light() -> bool:
    return _mode == LIGHT


def set_mode(m: str):
    """Switch the module-level colour tokens (BG, TEXT, Q_RED, ...) to theme m."""
    global _mode
    _mode = LIGHT if m == LIGHT else DARK
    g = globals()
    for k, v in PALETTES[_mode].items():
        g[k] = v
    for k in ("BG", "PANEL", "TEXT", "MUTED", "RED", "RED_HOVER", "RED_DARK", "ACCENT_TEXT",
              "TL_BAR", "TL_BAR_BORDER", "TL_TICK", "TL_MARK", "TL_INOUT", "TL_SEL", "TL_SCENE",
              "WAVE_BG", "WAVE_PEAK", "WAVE_RMS"):
        g["Q_" + k] = QColor(PALETTES[_mode][k])


set_mode(DARK)


def app_icon() -> QIcon:
    return QIcon(ICON)


def logo_pixmap(height: int, dark: bool | None = None) -> QPixmap:
    """The RulZ logo: white lettering on the dark theme, the original black lettering on light."""
    if dark is None:
        dark = not is_light()
    pm = QPixmap(LOGO_DARK if dark else LOGO_LIGHT)
    if pm.isNull():
        return pm
    return pm.scaledToHeight(height, Qt.SmoothTransformation)


def _palette() -> QPalette:
    p = QPalette()
    for role, col in (
        (QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, PANEL),
        (QPalette.AlternateBase, RAISED), (QPalette.ToolTipBase, PANEL), (QPalette.ToolTipText, TEXT),
        (QPalette.Text, TEXT), (QPalette.Button, RAISED), (QPalette.ButtonText, TEXT),
        (QPalette.BrightText, RED_HOVER), (QPalette.Highlight, RED), (QPalette.HighlightedText, "#ffffff"),
        (QPalette.Link, ACCENT_TEXT), (QPalette.PlaceholderText, MUTED),
    ):
        p.setColor(role, QColor(col))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(DIS_MENU))
    p.setColor(QPalette.Disabled, QPalette.Button, QColor(DIS_BG))
    p.setColor(QPalette.Disabled, QPalette.Base, QColor(DIS_INPUT_BG))
    return p


def _arrow(name: str) -> str:
    return _asset_url(f"{name}{ARROW_SUFFIX}.png")


def _families() -> list[str]:
    # QFontDatabase needs a QGuiApplication; the module-level STYLESHEET below is built before there is one.
    return QFontDatabase.families() if QGuiApplication.instance() is not None else []


def ui_family() -> str:
    """Segoe UI Variable Text (Windows 11), else Segoe UI."""
    return "Segoe UI Variable Text" if "Segoe UI Variable Text" in _families() else "Segoe UI"


def display_family() -> str:
    """Titles: Segoe UI Variable Display, else Segoe UI."""
    return "Segoe UI Variable Display" if "Segoe UI Variable Display" in _families() else "Segoe UI"


def mono_family() -> str:
    families = _families()
    for fam in ("Cascadia Code", "Cascadia Mono", "Consolas"):
        if fam in families:
            return fam
    return "monospace"


def stylesheet() -> str:
    return f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px 8px; }}

QMenuBar {{ background: {BAR}; border-bottom: 2px solid {RED}; padding: 2px; }}
QMenuBar::item {{ background: transparent; padding: 6px 10px; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {RAISED_HOVER}; color: {TEXT}; }}
QMenu {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 24px 6px 24px; border-radius: 6px; }}
QMenu::item:selected {{ background: {RAISED_HOVER}; color: {TEXT}; }}
QMenu::item:disabled {{ color: {DIS_MENU}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
QMenu::indicator {{ width: 12px; height: 12px; left: 6px; }}
QMenu::indicator:non-exclusive:unchecked {{ border: 1px solid {INDICATOR_BORDER}; border-radius: 3px; background: {BG}; }}
QMenu::indicator:non-exclusive:checked {{
    border: 1px solid {RED_HOVER}; border-radius: 3px; background: {RED}; image: url({_asset_url("check.png")});
}}

QStatusBar {{ background: {BAR}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ color: {TEXT}; }}

QPushButton {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 6px 14px; min-height: 18px;
}}
QPushButton:hover {{ background: {RAISED_HOVER}; border-color: {BORDER_STRONG}; }}
QPushButton:pressed {{ background: {SELECT}; border-color: {BORDER_STRONG}; }}
QPushButton:focus {{ border: 2px solid {RED}; padding: 5px 13px; }}
QPushButton:checked {{ background: {RED}; border-color: {RED_HOVER}; color: #ffffff; }}
QPushButton:checked:hover {{ background: {RED_HOVER}; }}
QPushButton:checked:focus {{ border: 2px solid {TEXT}; }}
QPushButton:disabled {{ background: {DIS_BG}; color: {DIS_TEXT}; border-color: {DIS_BORDER}; }}
QPushButton#primary {{ background: {RED}; border-color: {RED}; color: #ffffff; font-weight: 600; }}
QPushButton#primary:hover {{ background: {RED_HOVER}; border-color: {RED_HOVER}; }}
QPushButton#primary:pressed {{ background: {PRESSED}; border-color: {PRESSED}; }}
QPushButton#primary:focus {{ border: 2px solid {TEXT}; padding: 5px 13px; }}
QPushButton#primary:disabled {{
    background: {PRIMARY_DIS_BG}; color: {PRIMARY_DIS_TEXT}; border-color: {PRIMARY_DIS_BORDER};
}}
QDialogButtonBox QPushButton:default {{ border-color: {RED}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px 8px;
    selection-background-color: {RED}; selection-color: #ffffff;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 2px solid {RED}; padding: 3px 7px; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {DIS_TEXT}; background: {DIS_INPUT_BG}; border-color: {DIS_BORDER};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url({_arrow("arrow_down")}); width: 10px; height: 6px; }}
QComboBox::down-arrow:disabled {{ image: url({_arrow("arrow_down_dis")}); }}
QComboBox::down-arrow:on {{ image: url({_arrow("arrow_up")}); }}
QComboBox QAbstractItemView {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 8px; selection-background-color: {SELECT};
    selection-color: {TEXT}; outline: 0;
}}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {RAISED}; border: none; width: 16px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 7px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 7px; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{ background: {RAISED_HOVER}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({_arrow("arrow_up")}); width: 9px; height: 5px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({_arrow("arrow_down")}); width: 9px; height: 5px; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{ image: url({_arrow("arrow_up_dis")}); }}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: url({_arrow("arrow_down_dis")}); }}

QCheckBox, QRadioButton {{ spacing: 8px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px; background: {RAISED}; border: 1px solid {INDICATOR_BORDER};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {MUTED}; }}
QCheckBox::indicator:focus, QRadioButton::indicator:focus {{ border: 2px solid {RED}; width: 14px; height: 14px; }}
QCheckBox::indicator:checked {{ background: {RED}; border-color: {RED}; image: url({_asset_url("check.png")}); }}
QCheckBox::indicator:checked:focus {{ border: 2px solid {TEXT}; width: 14px; height: 14px; }}
QRadioButton::indicator:checked {{
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                stop:0 #ffffff, stop:0.35 #ffffff, stop:0.45 {RED}, stop:1 {RED});
    border-color: {RED};
}}
QCheckBox:disabled, QRadioButton:disabled {{ color: {DIS_TEXT}; }}

QSlider::groove:horizontal {{ height: 4px; background: {TRACK}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {RED}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #ffffff; border: 2px solid {RED}; width: 10px; height: 10px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:focus {{ background: {RED}; border: 3px solid {TEXT}; width: 8px; height: 8px; margin: -5px 0; }}

QProgressBar {{
    background: {TRACK}; border: none; border-radius: 8px; text-align: center; color: {TEXT};
    min-height: 16px; max-height: 16px;
}}
QProgressBar::chunk {{ background: {RED}; border-radius: 8px; }}

QGroupBox {{ border: 1px solid {BORDER}; border-radius: 8px; margin-top: 12px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {ACCENT_TEXT}; font-weight: 600; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-top: 2px solid {RED}; background: {BG}; top: -1px; }}
QTabBar::tab {{
    background: {RAISED}; border: 1px solid {BORDER}; border-bottom: none;
    padding: 6px 16px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{ background: {RED}; color: #ffffff; border-color: {RED}; }}
QTabBar::tab:hover:!selected {{ background: {RAISED_HOVER}; }}
QTabBar::tab:focus {{ border: 2px solid {TEXT}; border-bottom: none; padding: 5px 15px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER_STRONG}; border-radius: 4px; min-height: 20px; }}
QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_STRONG}; border-radius: 4px; min-width: 20px; }}
QScrollBar::handle:horizontal:hover {{ background: {MUTED}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QListView, QListWidget, QTreeView, QTableView {{ border: 1px solid transparent; border-radius: 8px; outline: 0;
    selection-background-color: {SELECT}; selection-color: {TEXT}; }}
QListView:focus, QListWidget:focus, QTreeView:focus, QTableView:focus {{ border: 1px solid {BORDER_STRONG}; }}

QLabel#note {{ color: {MUTED}; }}
QLabel#accent {{ color: {ACCENT_TEXT}; font-weight: 600; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QLabel#sectionLabel {{ color: {ACCENT_TEXT}; font-weight: 600; }}

QFrame#lookCard {{ background: {PANEL}; border: 2px solid {BORDER}; border-radius: 12px; }}
QFrame#lookCard:hover {{ border-color: {RED_HOVER}; }}
QFrame#lookCard[selected="true"] {{ border-color: {RED}; }}
QLabel#lookTitle {{ font-family: "{display_family()}"; font-size: 15px; font-weight: 600; }}
QLabel#lookHeading {{ font-family: "{display_family()}"; font-size: 20px; font-weight: 600; }}
"""


STYLESHEET = stylesheet()   # the dark sheet; kept for anything that imported the old constant


def apply_mode(app, m: str):
    """Switch the running app to theme m: palette, stylesheet, and a repaint of every window."""
    set_mode(m)
    try:  # native title bar follows the app's choice instead of the Windows setting
        app.styleHints().setColorScheme(Qt.ColorScheme.Light if _mode == LIGHT else Qt.ColorScheme.Dark)
    except Exception:  # noqa: BLE001
        pass
    f = QFont(ui_family())
    f.setPointSizeF(10.5)
    app.setFont(f)
    app.setPalette(_palette())
    app.setStyleSheet(stylesheet())
    for w in app.topLevelWidgets():
        w.update()


def apply(app, mode: str | None = None):
    """Apply the RulZ theme (the current one unless mode is given), name and icon to the QApplication."""
    app.setStyle("Fusion")
    apply_mode(app, _mode if mode is None else mode)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(app_icon())
    if os.name == "nt":
        # Give the taskbar our own identity so it shows the RulZ icon instead of Python's.
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RulZ.SimpleVideoClipExtractor")
        except Exception:  # noqa: BLE001
            pass
