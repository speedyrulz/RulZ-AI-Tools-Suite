"""Rulz themes: Fusion palette plus a stylesheet. Dark (black & red, the default) or light (white & red)."""
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QIcon, QPalette, QPixmap

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

DARK, LIGHT = "dark", "light"

PALETTES = {
    DARK: dict(
        RED="#ed000d", RED_HOVER="#ff2633", RED_DARK="#6d0309", BG="#050505", PANEL="#0c0c0e", CARD="#141417",
        BASE="#0c0c0e", BORDER="#232328", TEXT="#f2f2f3", MUTED="#a3a3ab", STRONG="#ffffff",
        BORDER_STRONG="#34343b", HOVER="#1c1c21", PRESSED="#c8000b", SELECT="#2d0b0e", ACCENT_TEXT="#ff6b72",
        TRACK="#191919",
        INDICATOR="#34343b", SCROLL="#34343b", DISABLED="#5c5c63", TB_DISABLED="#55555b",
        PRIMARY_DISABLED_TEXT="#b9b9b9", LIGHT_ROLE="#3a3a40", MIDLIGHT="#2a2a2f", DARK_ROLE="#08080a",
        ARROW_UP="arrow_up.png", ARROW_DOWN="arrow_down.png", LOGO="logo_dark.png",
    ),
    LIGHT: dict(
        RED="#ED000D", RED_HOVER="#C8000B", RED_DARK="#f3a0a5", BG="#FFFFFF", PANEL="#F5F5F7", CARD="#EBEBEF",
        BASE="#F5F5F7", BORDER="#D9D9DE", TEXT="#111114", MUTED="#5F6068", STRONG="#111114",
        BORDER_STRONG="#C4C4CB", HOVER="#E2E2E7", PRESSED="#A8000A", SELECT="#F7E5E7", ACCENT_TEXT="#C8000B",
        TRACK="#EBEBEB",
        INDICATOR="#C4C4CB", SCROLL="#c4c4cb", DISABLED="#a3a3ab", TB_DISABLED="#a3a3ab",
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


def ui_family():
    """Segoe UI Variable Text (Windows 11), else Segoe UI."""
    return "Segoe UI Variable Text" if "Segoe UI Variable Text" in QFontDatabase.families() else "Segoe UI"


def display_family():
    """Titles: Segoe UI Variable Display, else Segoe UI."""
    return "Segoe UI Variable Display" if "Segoe UI Variable Display" in QFontDatabase.families() else "Segoe UI"


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
    try:  # native title bar follows the app's choice instead of the Windows setting
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark if _mode == DARK else Qt.ColorScheme.Light)
    except Exception:
        pass
    app.setStyle("Fusion")
    f = QFont(ui_family())
    f.setPointSizeF(10.5)
    app.setFont(f)
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
    HOVER, STRONG_B, PRESSED, SELECT = c["HOVER"], c["BORDER_STRONG"], c["PRESSED"], c["SELECT"]
    ACCENT_TEXT, DISPLAY = c["ACCENT_TEXT"], display_family()
    return f"""
QMainWindow, QDialog {{ background: {BG}; }}
QMenuBar {{ background: {BG}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {HOVER}; color: {TEXT}; }}
QMenu {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background: {HOVER}; color: {TEXT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}

QToolBar {{ background: {BG}; border: none; border-bottom: 2px solid {RED}; padding: 4px 6px; spacing: 4px; }}
QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 4px 6px; }}
QToolButton {{ padding: 5px 10px; border-radius: 6px; border: 1px solid transparent; }}
QToolButton:hover {{ background: {HOVER}; border-color: {STRONG_B}; }}
QToolButton:pressed {{ background: {SELECT}; }}
QToolButton:focus {{ border: 2px solid {RED}; padding: 4px 9px; }}
QToolButton:disabled {{ color: {c["TB_DISABLED"]}; }}
QToolButton#primary {{ background: {RED}; color: white; font-weight: 600; }}
QToolButton#primary:hover {{ background: {RED_HOVER}; border-color: {RED_HOVER}; }}
QToolButton#primary:pressed {{ background: {PRESSED}; }}
QToolButton#primary:focus {{ border: 2px solid {c["STRONG"]}; }}
QToolButton#primary:disabled {{ background: {RED_DARK}; color: {c["PRIMARY_DISABLED_TEXT"]}; }}

QDockWidget {{ color: {TEXT}; titlebar-close-icon: none; }}
QDockWidget::title {{ background: {BG}; padding: 6px 8px; border-left: 3px solid {RED}; font-weight: 600; }}

QLabel#section {{ color: {MUTED}; font-size: 8pt; font-weight: bold; letter-spacing: 1px; margin-top: 6px; }}

QPushButton {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 14px; min-height: 18px; }}
QPushButton:hover {{ background: {HOVER}; border-color: {STRONG_B}; }}
QPushButton:pressed {{ background: {SELECT}; }}
QPushButton:focus {{ border: 2px solid {RED}; padding: 5px 13px; }}
QPushButton:disabled {{ color: {c["DISABLED"]}; }}
QPushButton:default, QPushButton#primary {{ background: {RED}; border-color: {RED}; color: white; font-weight: 600; }}
QPushButton:default:hover, QPushButton#primary:hover {{ background: {RED_HOVER}; border-color: {RED_HOVER}; }}
QPushButton:default:pressed, QPushButton#primary:pressed {{ background: {PRESSED}; border-color: {PRESSED}; }}
QPushButton:default:focus, QPushButton#primary:focus {{ border: 2px solid {c["STRONG"]}; padding: 5px 13px; }}
QPushButton:default:disabled, QPushButton#primary:disabled {{ background: {RED_DARK}; border-color: {RED_DARK}; color: {c["PRIMARY_DISABLED_TEXT"]}; }}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BASE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px 8px;
    selection-background-color: {RED}; selection-color: white; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 2px solid {RED}; padding: 3px 7px; }}
QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {STRONG_B}; padding: 4px 8px; }}
QComboBox QAbstractItemView {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    selection-background-color: {SELECT}; selection-color: {TEXT}; outline: 0; }}

QListView {{ background: {BG}; border: 1px solid transparent; }}
QListView:focus {{ border: 1px solid {STRONG_B}; }}

QProgressBar {{ background: {c["TRACK"]}; border: none; border-radius: 8px; text-align: center; min-height: 16px; max-height: 16px; }}
QProgressBar::chunk {{ background: {RED}; border-radius: 8px; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {RED}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {RED_HOVER}; }}
QSlider::handle:horizontal:focus {{ background: {c["STRONG"]}; border: 3px solid {RED}; width: 8px; height: 8px; margin: -6px 0; }}

QCheckBox, QRadioButton {{ spacing: 8px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; border: 1px solid {STRONG_B}; background: {CARD}; }}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:checked {{ background: {RED}; border-color: {RED}; image: url({_url("check.png")}); }}
QRadioButton::indicator:checked {{ background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
    stop:0 white, stop:0.35 white, stop:0.45 {RED}, stop:1 {RED}); border-color: {RED}; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {MUTED}; }}
QCheckBox::indicator:focus, QRadioButton::indicator:focus {{ border: 2px solid {RED}; width: 14px; height: 14px; }}
QCheckBox::indicator:checked:focus {{ border: 2px solid {c["STRONG"]}; width: 14px; height: 14px; }}

QSpinBox, QDoubleSpinBox {{ padding-right: 20px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; width: 18px; background: {CARD}; border-left: 1px solid {BORDER}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 7px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 7px; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {HOVER}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({_url(c["ARROW_UP"])}); width: 8px; height: 5px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({_url(c["ARROW_DOWN"])}); width: 8px; height: 5px; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url({_url(c["ARROW_DOWN"])}); width: 9px; height: 6px; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; top: -1px; }}
QTabBar::tab {{ background: {PANEL}; padding: 7px 16px; border: 1px solid {BORDER}; border-bottom: none;
    border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px; color: {MUTED}; }}
QTabBar::tab:selected {{ background: {CARD}; color: {c["STRONG"]}; border-top: 2px solid {RED}; }}
QTabBar::tab:hover:!selected {{ background: {HOVER}; }}
QTabBar::tab:focus {{ border: 2px solid {RED}; border-bottom: none; padding: 6px 15px; }}

QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {c["SCROLL"]}; min-height: 30px; border-radius: 5px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {c["SCROLL"]}; min-width: 30px; border-radius: 5px; margin: 2px; }}
QScrollBar::handle:horizontal:hover {{ background: {MUTED}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QStatusBar {{ background: {BG}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QToolTip {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px 8px; }}
QSplitter::handle, QMainWindow::separator {{ background: {BORDER}; width: 2px; height: 2px; }}
QMainWindow::separator:hover {{ background: {RED}; }}
"""
