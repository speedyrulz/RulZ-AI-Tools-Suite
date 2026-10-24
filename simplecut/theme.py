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

# palette
BG = "#0b0b0d"          # window / video background
PANEL = "#151518"       # panels, inputs
RAISED = "#1f1f23"      # buttons
RAISED_HOVER = "#2a2a30"
BORDER = "#34343a"
TEXT = "#ececee"
MUTED = "#8e8e96"
RED = "#e2121c"         # logo red
RED_HOVER = "#ff2b35"
RED_DARK = "#7d0a10"

# QColor versions for custom painting
Q_BG = QColor(BG)
Q_PANEL = QColor(PANEL)
Q_TEXT = QColor(TEXT)
Q_MUTED = QColor(MUTED)
Q_RED = QColor(RED)
Q_RED_HOVER = QColor(RED_HOVER)
Q_RED_DARK = QColor(RED_DARK)


def app_icon() -> QIcon:
    return QIcon(ICON)


def logo_pixmap(height: int, dark: bool = True) -> QPixmap:
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
        (QPalette.Link, RED_HOVER), (QPalette.PlaceholderText, MUTED),
    ):
        p.setColor(role, QColor(col))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor("#5a5a62"))
    p.setColor(QPalette.Disabled, QPalette.Button, QColor("#18181b"))
    p.setColor(QPalette.Disabled, QPalette.Base, QColor("#111114"))
    return p


STYLESHEET = f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {RED}; padding: 4px; }}

QMenuBar {{ background: #000000; border-bottom: 2px solid {RED}; padding: 2px; }}
QMenuBar::item {{ background: transparent; padding: 6px 10px; }}
QMenuBar::item:selected {{ background: {RED}; color: #ffffff; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 24px 6px 18px; }}
QMenu::item:selected {{ background: {RED}; color: #ffffff; }}
QMenu::item:disabled {{ color: #5a5a62; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}

QStatusBar {{ background: #000000; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ color: {TEXT}; }}

QPushButton {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 5px 12px; min-height: 18px;
}}
QPushButton:hover {{ background: {RAISED_HOVER}; border-color: {RED}; }}
QPushButton:pressed {{ background: {RED_DARK}; border-color: {RED}; }}
QPushButton:checked {{ background: {RED}; border-color: {RED_HOVER}; color: #ffffff; }}
QPushButton:checked:hover {{ background: {RED_HOVER}; }}
QPushButton:disabled {{ background: #141417; color: #55555c; border-color: #232327; }}
QPushButton#primary {{ background: {RED}; border-color: {RED_HOVER}; color: #ffffff; font-weight: bold; }}
QPushButton#primary:hover {{ background: {RED_HOVER}; }}
QPushButton#primary:disabled {{ background: #3a1012; color: #8a5a5c; border-color: #4a1416; }}
QDialogButtonBox QPushButton:default {{ border-color: {RED}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; padding: 3px 6px;
    selection-background-color: {RED}; selection-color: #ffffff;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {RED}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: #55555c; background: #111114; border-color: #232327;
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url({_asset_url("arrow_down.png")}); width: 10px; height: 6px; }}
QComboBox::down-arrow:disabled {{ image: url({_asset_url("arrow_down_dis.png")}); }}
QComboBox::down-arrow:on {{ image: url({_asset_url("arrow_up.png")}); }}
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
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({_asset_url("arrow_up.png")}); width: 9px; height: 5px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({_asset_url("arrow_down.png")}); width: 9px; height: 5px; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{ image: url({_asset_url("arrow_up_dis.png")}); }}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: url({_asset_url("arrow_down_dis.png")}); }}

QCheckBox, QRadioButton {{ spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px; height: 14px; background: {PANEL}; border: 1px solid #55555c;
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
QCheckBox:disabled, QRadioButton:disabled {{ color: #55555c; }}

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
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {RED_HOVER}; font-weight: bold; }}

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
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

QLabel#note {{ color: {MUTED}; }}
QLabel#accent {{ color: {RED_HOVER}; font-weight: bold; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QLabel#sectionLabel {{ color: {RED_HOVER}; font-weight: bold; }}
"""


def apply(app):
    """Apply the RulZ theme, name and icon to the QApplication."""
    app.setStyle("Fusion")
    app.setPalette(_palette())
    app.setStyleSheet(STYLESHEET)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(app_icon())
    if os.name == "nt":
        # Give the taskbar our own identity so it shows the RulZ icon instead of Python's.
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RulZ.SimpleVideoClipExtractor")
        except Exception:  # noqa: BLE001
            pass
