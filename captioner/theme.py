"""Rulz themes: Fusion palette plus a stylesheet. Dark (black & red, the default) or light (white & red)."""
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPalette, QPixmap

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

DARK, LIGHT = "dark", "light"

PALETTES = {
    DARK: dict(
        RED="#e40613", RED_HOVER="#ff2a35", RED_DARK="#7a0a10", BG="#0d0d0e", PANEL="#151517", CARD="#1b1b1e",
        BASE="#101012", BORDER="#2c2c31", TEXT="#ececec", MUTED="#8a8a92", STRONG="#ffffff",
        INDICATOR="#4a4a50", SCROLL="#34343a", DISABLED="#5c5c63", TB_DISABLED="#55555b",
        PRIMARY_DISABLED_TEXT="#b9b9b9", LIGHT_ROLE="#3a3a40", MIDLIGHT="#2a2a2f", DARK_ROLE="#08080a",
        ARROW_UP="arrow_up.png", ARROW_DOWN="arrow_down.png", LOGO="logo_dark.png",
    ),
    LIGHT: dict(
        RED="#ED000D", RED_HOVER="#C8000B", RED_DARK="#f3a0a5", BG="#FFFFFF", PANEL="#F5F5F7", CARD="#EBEBEF",
        BASE="#FFFFFF", BORDER="#D9D9DE", TEXT="#111114", MUTED="#5F6068", STRONG="#111114",
        INDICATOR="#a9a9b0", SCROLL="#c4c4cb", DISABLED="#a3a3ab", TB_DISABLED="#a3a3ab",
        PRIMARY_DISABLED_TEXT="#ffffff", LIGHT_ROLE="#ffffff", MIDLIGHT="#f0f0f3", DARK_ROLE="#b4b4bb",
        ARROW_UP="arrow_up_light.png", ARROW_DOWN="arrow_down_light.png", LOGO="logo_light.png",
    ),
}

_mode = DARK


def current():
    return _mode


def colors(mode=None):
    return PALETTES[mode or _mode]


def asset(name):
    return os.path.join(ASSETS, name)


def icon():
    return QIcon(asset("icon.ico"))


def logo(height):
    """Logo scaled for the screen's DPI so it stays crisp on high-DPI displays."""
    pm = QPixmap(asset(colors()["LOGO"]))
    if pm.isNull():
        return pm
    screen = QGuiApplication.primaryScreen()
    dpr = screen.devicePixelRatio() if screen else 1.0
    pm = pm.scaledToHeight(round(height * dpr), Qt.SmoothTransformation)
    pm.setDevicePixelRatio(dpr)
    return pm


def apply(app, mode=DARK):
    """Apply a theme to the whole app; can be called again at runtime to switch live."""
    global _mode
    _mode = mode if mode in PALETTES else DARK
    c = colors()
    app.setStyle("Fusion")
    p = QPalette()
    roles = {
        QPalette.Window: c["PANEL"], QPalette.WindowText: c["TEXT"], QPalette.Base: c["BASE"],
        QPalette.AlternateBase: c["CARD"], QPalette.ToolTipBase: c["BG"], QPalette.ToolTipText: c["TEXT"],
        QPalette.Text: c["TEXT"], QPalette.Button: c["CARD"], QPalette.ButtonText: c["TEXT"],
        QPalette.BrightText: "#ffffff", QPalette.Highlight: c["RED"], QPalette.HighlightedText: "#ffffff",
        QPalette.Link: c["RED_HOVER"], QPalette.PlaceholderText: c["MUTED"], QPalette.Light: c["LIGHT_ROLE"],
        QPalette.Midlight: c["MIDLIGHT"], QPalette.Mid: c["BORDER"], QPalette.Dark: c["DARK_ROLE"],
        QPalette.Shadow: "#000000",
    }
    for role, colour in roles.items():
        p.setColor(role, QColor(colour))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(c["DISABLED"]))
    app.setPalette(p)
    app.setStyleSheet(stylesheet(c))


def _url(name):
    return asset(name).replace("\\", "/")


def stylesheet(c):
    RED, RED_HOVER, RED_DARK = c["RED"], c["RED_HOVER"], c["RED_DARK"]
    BG, PANEL, CARD, BASE, BORDER = c["BG"], c["PANEL"], c["CARD"], c["BASE"], c["BORDER"]
    TEXT, MUTED = c["TEXT"], c["MUTED"]
    return f"""
QMainWindow, QDialog {{ background: {BG}; }}
QMenuBar {{ background: {BG}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item:selected {{ background: {RED}; color: white; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; }}
QMenu::item {{ padding: 5px 22px; }}
QMenu::item:selected {{ background: {RED}; color: white; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}

QToolBar {{ background: {BG}; border: none; border-bottom: 2px solid {RED}; padding: 4px 6px; spacing: 4px; }}
QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 4px 6px; }}
QToolButton {{ padding: 5px 10px; border-radius: 4px; border: 1px solid transparent; }}
QToolButton:hover {{ background: {CARD}; border-color: {BORDER}; }}
QToolButton:pressed {{ background: {RED_DARK}; }}
QToolButton:disabled {{ color: {c["TB_DISABLED"]}; }}
QToolButton#primary {{ background: {RED}; color: white; font-weight: bold; }}
QToolButton#primary:hover {{ background: {RED_HOVER}; }}
QToolButton#primary:disabled {{ background: {RED_DARK}; color: {c["PRIMARY_DISABLED_TEXT"]}; }}

QDockWidget {{ color: {TEXT}; titlebar-close-icon: none; }}
QDockWidget::title {{ background: {BG}; padding: 6px 8px; border-left: 3px solid {RED}; font-weight: bold; }}

QLabel#section {{ color: {MUTED}; font-size: 8pt; font-weight: bold; letter-spacing: 1px; margin-top: 6px; }}

QPushButton {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 4px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {RED}; }}
QPushButton:pressed {{ background: {RED_DARK}; }}
QPushButton:default, QPushButton#primary {{ background: {RED}; border-color: {RED}; color: white; font-weight: bold; }}
QPushButton:default:hover, QPushButton#primary:hover {{ background: {RED_HOVER}; }}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BASE}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 6px;
    selection-background-color: {RED}; selection-color: white; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {RED}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {RED}; }}

QListView {{ background: {BG}; border: none; }}

QProgressBar {{ background: {BASE}; border: 1px solid {BORDER}; border-radius: 4px; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {RED}; border-radius: 3px; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {RED}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {RED_HOVER}; }}

QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; border: 1px solid {c["INDICATOR"]}; background: {BASE}; }}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked {{ background: {RED}; border-color: {RED}; image: url({_url("check.png")}); }}
QRadioButton::indicator:checked {{ background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
    stop:0 white, stop:0.35 white, stop:0.45 {RED}, stop:1 {RED}); border-color: {RED}; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {RED}; }}

QSpinBox, QDoubleSpinBox {{ padding-right: 20px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; width: 18px; background: {CARD}; border-left: 1px solid {BORDER}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 4px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 4px; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {RED_DARK}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({_url(c["ARROW_UP"])}); width: 8px; height: 5px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({_url(c["ARROW_DOWN"])}); width: 8px; height: 5px; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url({_url(c["ARROW_DOWN"])}); width: 9px; height: 6px; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 4px; top: -1px; }}
QTabBar::tab {{ background: {PANEL}; padding: 7px 16px; border: 1px solid {BORDER}; border-bottom: none;
    border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; color: {MUTED}; }}
QTabBar::tab:selected {{ background: {CARD}; color: {c["STRONG"]}; border-top: 2px solid {RED}; }}

QScrollBar:vertical {{ background: {BG}; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {c["SCROLL"]}; min-height: 30px; border-radius: 5px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: {RED}; }}
QScrollBar:horizontal {{ background: {BG}; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {c["SCROLL"]}; min-width: 30px; border-radius: 5px; margin: 2px; }}
QScrollBar::handle:horizontal:hover {{ background: {RED}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QStatusBar {{ background: {BG}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QToolTip {{ background: {BG}; color: {TEXT}; border: 1px solid {RED}; padding: 4px; }}
QSplitter::handle, QMainWindow::separator {{ background: {BORDER}; width: 2px; height: 2px; }}
QMainWindow::separator:hover {{ background: {RED}; }}
"""
