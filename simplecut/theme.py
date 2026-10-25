"""RulZ branding: app name, logo assets and the black-and-red theme."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPalette, QPixmap

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
        BG="#0b0b0d",          # window / video background
        PANEL="#151518",       # panels, inputs
        RAISED="#1f1f23",      # buttons
        RAISED_HOVER="#2a2a30",
        BORDER="#34343a",
        TEXT="#ececee",
        MUTED="#8e8e96",
        RED="#e2121c",         # logo red
        RED_HOVER="#ff2b35",
        RED_DARK="#7d0a10",    # pressed buttons
        ACCENT_TEXT="#ff2b35",  # red used as text (section titles, keyframe label)
        BAR="#000000",         # menu bar / status bar
        DIS_TEXT="#55555c", DIS_BG="#141417", DIS_BORDER="#232327", DIS_INPUT_BG="#111114",
        DIS_MENU="#5a5a62",
        PRIMARY_DIS_BG="#3a1012", PRIMARY_DIS_TEXT="#8a5a5c", PRIMARY_DIS_BORDER="#4a1416",
        INDICATOR_BORDER="#55555c",
        # timeline (QColor accepts #AARRGGBB)
        TL_BAR="#1c1c20", TL_BAR_BORDER="#34343a", TL_TICK="#cdcdd4", TL_MARK="#ffffff",
        TL_INOUT="#be7d0a10", TL_SEL="#78e2121c",
        ARROW_SUFFIX="",
    ),
    LIGHT: dict(
        BG="#ffffff",
        PANEL="#f5f5f7",
        RAISED="#ebebef",
        RAISED_HOVER="#e0e0e6",
        BORDER="#d9d9de",
        TEXT="#111114",
        MUTED="#5f6068",
        RED="#ed000d",
        RED_HOVER="#c8000b",
        RED_DARK="#f7c4c7",
        ACCENT_TEXT="#c8000b",
        BAR="#ffffff",
        DIS_TEXT="#a3a3ab", DIS_BG="#f3f3f5", DIS_BORDER="#e3e3e8", DIS_INPUT_BG="#f7f7f9",
        DIS_MENU="#a3a3ab",
        PRIMARY_DIS_BG="#f6c9cb", PRIMARY_DIS_TEXT="#ffffff", PRIMARY_DIS_BORDER="#f1b4b7",
        INDICATOR_BORDER="#a3a3ab",
        TL_BAR="#ebebef", TL_BAR_BORDER="#c9c9cf", TL_TICK="#5f6068", TL_MARK="#111114",
        TL_INOUT="#55ed000d", TL_SEL="#4ced000d",
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
              "TL_BAR", "TL_BAR_BORDER", "TL_TICK", "TL_MARK", "TL_INOUT", "TL_SEL"):
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


def stylesheet() -> str:
    return f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {RED}; padding: 4px; }}

QMenuBar {{ background: {BAR}; border-bottom: 2px solid {RED}; padding: 2px; }}
QMenuBar::item {{ background: transparent; padding: 6px 10px; }}
QMenuBar::item:selected {{ background: {RED}; color: #ffffff; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 24px 6px 24px; }}
QMenu::item:selected {{ background: {RED}; color: #ffffff; }}
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
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 5px 12px; min-height: 18px;
}}
QPushButton:hover {{ background: {RAISED_HOVER}; border-color: {RED}; }}
QPushButton:pressed {{ background: {RED_DARK}; border-color: {RED}; }}
QPushButton:checked {{ background: {RED}; border-color: {RED_HOVER}; color: #ffffff; }}
QPushButton:checked:hover {{ background: {RED_HOVER}; }}
QPushButton:disabled {{ background: {DIS_BG}; color: {DIS_TEXT}; border-color: {DIS_BORDER}; }}
QPushButton#primary {{ background: {RED}; border-color: {RED_HOVER}; color: #ffffff; font-weight: bold; }}
QPushButton#primary:hover {{ background: {RED_HOVER}; }}
QPushButton#primary:disabled {{
    background: {PRIMARY_DIS_BG}; color: {PRIMARY_DIS_TEXT}; border-color: {PRIMARY_DIS_BORDER};
}}
QDialogButtonBox QPushButton:default {{ border-color: {RED}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; padding: 3px 6px;
    selection-background-color: {RED}; selection-color: #ffffff;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {RED}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {DIS_TEXT}; background: {DIS_INPUT_BG}; border-color: {DIS_BORDER};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url({_arrow("arrow_down")}); width: 10px; height: 6px; }}
QComboBox::down-arrow:disabled {{ image: url({_arrow("arrow_down_dis")}); }}
QComboBox::down-arrow:on {{ image: url({_arrow("arrow_up")}); }}
QComboBox QAbstractItemView {{
    background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {RED};
    selection-color: #ffffff; outline: 0;
}}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {RAISED}; border: none; width: 16px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 4px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 4px; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{ background: {RED_DARK}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({_arrow("arrow_up")}); width: 9px; height: 5px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({_arrow("arrow_down")}); width: 9px; height: 5px; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{ image: url({_arrow("arrow_up_dis")}); }}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: url({_arrow("arrow_down_dis")}); }}

QCheckBox, QRadioButton {{ spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px; height: 14px; background: {PANEL}; border: 1px solid {INDICATOR_BORDER};
}}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {RED}; }}
QCheckBox::indicator:checked {{ background: {RED}; border-color: {RED_HOVER}; image: url({_asset_url("check.png")}); }}
QRadioButton::indicator:checked {{
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                stop:0 {RED_HOVER}, stop:0.45 {RED}, stop:0.55 {PANEL}, stop:1 {PANEL});
    border-color: {RED};
}}
QCheckBox:disabled, QRadioButton:disabled {{ color: {DIS_TEXT}; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {RED}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #ffffff; border: 2px solid {RED}; width: 10px; height: 10px; margin: -5px 0; border-radius: 7px;
}}

QProgressBar {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; text-align: center; color: {TEXT};
}}
QProgressBar::chunk {{ background: {RED}; border-radius: 3px; }}

QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 12px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {ACCENT_TEXT}; font-weight: bold; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-top: 2px solid {RED}; background: {BG}; top: -1px; }}
QTabBar::tab {{
    background: {RAISED}; border: 1px solid {BORDER}; border-bottom: none;
    padding: 6px 16px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{ background: {RED}; color: #ffffff; border-color: {RED}; }}
QTabBar::tab:hover:!selected {{ background: {RAISED_HOVER}; }}

QScrollBar:vertical {{ background: {BG}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 20px; }}
QScrollBar::handle:vertical:hover {{ background: {RED}; }}
QScrollBar:horizontal {{ background: {BG}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 4px; min-width: 20px; }}
QScrollBar::handle:horizontal:hover {{ background: {RED}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QLabel#note {{ color: {MUTED}; }}
QLabel#accent {{ color: {ACCENT_TEXT}; font-weight: bold; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QLabel#sectionLabel {{ color: {ACCENT_TEXT}; font-weight: bold; }}

QFrame#lookCard {{ background: {PANEL}; border: 2px solid {BORDER}; border-radius: 10px; }}
QFrame#lookCard:hover {{ border-color: {RED_HOVER}; }}
QFrame#lookCard[selected="true"] {{ border-color: {RED}; }}
QLabel#lookTitle {{ font-size: 15px; font-weight: bold; }}
QLabel#lookHeading {{ font-size: 20px; font-weight: bold; }}
"""


STYLESHEET = stylesheet()   # the dark sheet; kept for anything that imported the old constant


def apply_mode(app, m: str):
    """Switch the running app to theme m: palette, stylesheet, and a repaint of every window."""
    set_mode(m)
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
