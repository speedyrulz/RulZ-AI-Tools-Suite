"""RulZ dark theme (black with the logo red #ED000D), the logo and the app icon.

The colours are the ones the RulZ AI Tools apps use, so the launcher looks like them.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QStandardPaths, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication

from ..instance import State

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# RulZ palette
BG = "#0b0b0c"            # window
SURFACE = "#111113"       # status bar
SIDEBAR = "#0c0c0e"
SIDE_BORDER = "#1e1e22"
SIDE_SELECT = "#2b0a0d"   # selected row in the sidebar
PANEL = "#141416"         # cards
INPUT = "#18181c"         # text fields
RAISED = "#1a1a1f"        # buttons
HOVER = "#232329"
BORDER = "#232328"
SELECT = "#3a0c10"        # pressed button
TEXT = "#f2f2f3"
MUTED = "#a3a3ab"
GRAY = "#6f6f78"          # disabled text, "stopped"
ACCENT = "#ed000d"        # logo red
ACCENT_HOVER = "#ff1f2d"
ON_ACCENT = "#ffffff"
LOG_BG = "#070708"

# Status colours. They say what an instance is doing, so they are not the brand red.
GREEN = "#3fcf8e"
AMBER = "#f2b84b"
RED = "#ff4d58"
BLUE = "#6aa9ff"
ON_STATUS = "#0b0b0c"     # text on a status pill

STATE_COLORS = {
    State.STOPPED: GRAY,
    State.QUEUED: AMBER,
    State.STARTING: AMBER,
    State.RUNNING: GREEN,
    State.STOPPING: AMBER,
    State.FAILED: RED,
}
EXTERNAL_COLOR = BLUE


def mono_font(size: int = 10) -> QFont:
    families = set(QFontDatabase.families())
    for family in ("Cascadia Mono", "Cascadia Code", "Consolas", "Courier New"):
        if family in families:
            font = QFont(family, size)
            break
    else:
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(size)
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def apply_theme(app: QApplication) -> None:
    try:   # the Windows title bar follows the app's dark look, not the Windows setting
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except (AttributeError, RuntimeError):
        pass
    app.setStyle("Fusion")
    family = "Segoe UI Variable Text"
    if family not in QFontDatabase.families():
        family = "Segoe UI"
    app.setFont(QFont(family, 10))

    palette = QPalette()
    roles = QPalette.ColorRole
    palette.setColor(roles.Window, QColor(BG))
    palette.setColor(roles.WindowText, QColor(TEXT))
    palette.setColor(roles.Base, QColor(PANEL))
    palette.setColor(roles.AlternateBase, QColor(SURFACE))
    palette.setColor(roles.ToolTipBase, QColor(PANEL))
    palette.setColor(roles.ToolTipText, QColor(TEXT))
    palette.setColor(roles.Text, QColor(TEXT))
    palette.setColor(roles.PlaceholderText, QColor(GRAY))
    palette.setColor(roles.Button, QColor(RAISED))
    palette.setColor(roles.ButtonText, QColor(TEXT))
    palette.setColor(roles.BrightText, QColor("#ffffff"))
    palette.setColor(roles.Link, QColor(ACCENT_HOVER))
    palette.setColor(roles.Highlight, QColor(ACCENT))
    palette.setColor(roles.HighlightedText, QColor(ON_ACCENT))
    palette.setColor(roles.Mid, QColor(BORDER))
    disabled = QPalette.ColorGroup.Disabled
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        palette.setColor(disabled, role, QColor(GRAY))
    app.setPalette(palette)
    app.setStyleSheet(_STYLE.replace("@CHECK@", _check_image()))


def _check_image() -> str:
    """Draw the tick used in checked boxes and return its path (style sheets need a file)."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.TempLocation)
    folder = Path(base) / "rulz-comfyui-launcher"
    path = folder / "check-white.png"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(ON_ACCENT), 4.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPolyline([QPointF(7, 17), QPointF(13.5, 23), QPointF(25, 10)])
        painter.end()
        pixmap.save(str(path), "PNG")
    except OSError:
        pass
    return path.as_posix()


_STYLE = f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog, QMessageBox {{ background: {BG}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 8px; }}

QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; }}
QWidget#sidebar {{ background: {SIDEBAR}; }}
QLabel#brandSub {{ color: {MUTED}; font-size: 8pt; font-weight: 600; letter-spacing: 2px; }}
QLabel#sideTitle {{ color: {MUTED}; font-size: 9pt; font-weight: 600; }}
QLabel#cardTitle {{ font-size: 11pt; font-weight: 600; }}
QLabel#pageTitle {{ font-size: 16pt; font-weight: 700; }}
QLabel#muted, QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel#hint {{ color: {MUTED}; font-size: 9pt; }}
QLabel#error {{ color: {RED}; }}
QLabel#warning {{ color: {AMBER}; }}
QLabel#notice {{ color: {AMBER}; font-size: 9pt; }}
QLabel#pill {{ border-radius: 9px; padding: 2px 10px; font-size: 9pt; font-weight: 600; }}

QCheckBox, QRadioButton {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {GRAY}; border-radius: 4px; background: {INPUT}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url("@CHECK@"); }}
QCheckBox::indicator:disabled {{ border-color: {BORDER}; background: {PANEL}; }}
QCheckBox::indicator:checked:disabled {{ background: {GRAY}; border-color: {GRAY}; }}
QRadioButton::indicator {{ width: 15px; height: 15px; border: 1px solid {GRAY}; border-radius: 8px; background: {INPUT}; }}
QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QRadioButton::indicator:checked {{ width: 7px; height: 7px; border: 5px solid {ACCENT}; background: #ffffff; }}
QRadioButton::indicator:disabled {{ border-color: {BORDER}; background: {PANEL}; }}
QRadioButton::indicator:checked:disabled {{ border-color: {GRAY}; background: {PANEL}; }}

QPushButton {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 8px;
               padding: 5px 12px; min-height: 18px; font-weight: 500; }}
QPushButton:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: {SELECT}; }}
QPushButton:disabled {{ color: {GRAY}; background: {PANEL}; }}
QPushButton[primary="true"] {{ background: {ACCENT}; border-color: {ACCENT}; color: {ON_ACCENT}; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[primary="true"]:pressed {{ background: {ACCENT}; }}
QPushButton[primary="true"]:disabled {{ background: {RAISED}; border-color: {BORDER}; color: {GRAY}; }}
QPushButton[danger="true"] {{ color: {RED}; }}
QPushButton[flat="true"] {{ background: transparent; border: none; color: {MUTED}; padding: 4px 8px; }}
QPushButton[flat="true"]:hover {{ color: {TEXT}; background: {HOVER}; }}

QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 8px; padding: 4px 6px; }}
QToolButton:hover {{ background: {HOVER}; }}
QToolButton:disabled {{ color: {GRAY}; }}
QToolButton#sectionHeader {{ text-align: left; font-weight: 600; padding: 6px 4px; border: none; }}
QToolButton#sectionHeader:hover {{ background: {RAISED}; }}

QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {INPUT}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 8px;
    selection-background-color: {ACCENT}; selection-color: {ON_ACCENT}; }}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: {GRAY}; background: {PANEL}; }}
QLineEdit[invalid="true"] {{ border-color: {RED}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER};
                               selection-background-color: {HOVER}; selection-color: {TEXT}; }}

QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 10px; margin: 2px 6px; }}
QListWidget::item:hover {{ background: {RAISED}; }}
QListWidget::item:selected {{ background: {SIDE_SELECT}; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 16px; border-bottom: 2px solid transparent;
                font-weight: 600; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}

QTableWidget {{ background: {INPUT}; border: 1px solid {BORDER}; border-radius: 6px; gridline-color: {BORDER}; }}
QHeaderView::section {{ background: {RAISED}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
                        padding: 4px 8px; }}
QTableWidget QLineEdit {{ border-radius: 0; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: {GRAY}; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 4px; min-width: 30px; margin: 2px; }}
QScrollBar::handle:horizontal:hover {{ background: {GRAY}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QProgressBar {{ background: {HOVER}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QSplitter::handle {{ background: {SIDE_BORDER}; width: 1px; }}
QStatusBar {{ background: {SURFACE}; border-top: 1px solid {BORDER}; }}
QStatusBar::item {{ border: none; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {HOVER}; }}
QMenu::item:disabled {{ color: {GRAY}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}
"""


def app_icon() -> QIcon:
    """The RulZ app icon (window, taskbar and notification area)."""
    path = ASSETS / "icon.ico"
    return QIcon(str(path)) if path.exists() else QIcon()


def logo_pixmap(width: int, scale: float = 2.0) -> QPixmap:
    """The full RulZ logo with white lettering, `width` wide. Empty if the file is missing."""
    pixmap = QPixmap(str(ASSETS / "logo_full_white.png"))
    if pixmap.isNull():
        return pixmap
    pixmap = pixmap.scaledToWidth(round(width * scale), Qt.TransformationMode.SmoothTransformation)
    pixmap.setDevicePixelRatio(scale)
    return pixmap
