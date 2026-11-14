"""Dark theme, colours and the app icon."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QStandardPaths, Qt
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QIcon, QLinearGradient, QPainter,
                           QPainterPath, QPalette, QPen, QPixmap)
from PySide6.QtWidgets import QApplication

from ..instance import State

BG = "#15171c"
PANEL = "#1c1f26"
RAISED = "#252932"
HOVER = "#2e333e"
BORDER = "#343947"
TEXT = "#e7e9ee"
MUTED = "#98a0b3"
ACCENT = "#5b9dff"
ACCENT_HOVER = "#7ab0ff"
GREEN = "#3fcf8e"
AMBER = "#f2b84b"
RED = "#f0616d"
GRAY = "#6b7385"

STATE_COLORS = {
    State.STOPPED: GRAY,
    State.QUEUED: AMBER,
    State.STARTING: AMBER,
    State.RUNNING: GREEN,
    State.STOPPING: AMBER,
    State.FAILED: RED,
}
EXTERNAL_COLOR = ACCENT


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
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    palette = QPalette()
    roles = QPalette.ColorRole
    palette.setColor(roles.Window, QColor(BG))
    palette.setColor(roles.WindowText, QColor(TEXT))
    palette.setColor(roles.Base, QColor(PANEL))
    palette.setColor(roles.AlternateBase, QColor(RAISED))
    palette.setColor(roles.ToolTipBase, QColor(RAISED))
    palette.setColor(roles.ToolTipText, QColor(TEXT))
    palette.setColor(roles.Text, QColor(TEXT))
    palette.setColor(roles.PlaceholderText, QColor(GRAY))
    palette.setColor(roles.Button, QColor(RAISED))
    palette.setColor(roles.ButtonText, QColor(TEXT))
    palette.setColor(roles.BrightText, QColor("#ffffff"))
    palette.setColor(roles.Link, QColor(ACCENT))
    palette.setColor(roles.Highlight, QColor(ACCENT))
    palette.setColor(roles.HighlightedText, QColor("#0d1117"))
    disabled = QPalette.ColorGroup.Disabled
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        palette.setColor(disabled, role, QColor(GRAY))
    app.setPalette(palette)
    app.setStyleSheet(_STYLE.replace("@CHECK@", _check_image()))


def _check_image() -> str:
    """Draw the tick used in checked boxes and return its path (style sheets need a file)."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.TempLocation)
    folder = Path(base) / "rulz-comfyui-launcher"
    path = folder / "check.png"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#0d1117"), 4.5)
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
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px; }}

QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QWidget#sidebar {{ background: {PANEL}; }}
QLabel#cardTitle {{ font-size: 11pt; font-weight: 600; }}
QLabel#pageTitle {{ font-size: 15pt; font-weight: 600; }}
QLabel#muted, QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel#hint {{ color: {MUTED}; font-size: 9pt; }}
QLabel#error {{ color: {RED}; }}
QLabel#warning {{ color: {AMBER}; }}
QLabel#notice {{ color: {AMBER}; font-size: 9pt; }}
QLabel#pill {{ border-radius: 9px; padding: 2px 10px; font-size: 9pt; font-weight: 600; }}

QCheckBox, QRadioButton {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {GRAY}; border-radius: 4px; background: {BG}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url("@CHECK@"); }}
QCheckBox::indicator:disabled {{ border-color: {BORDER}; background: {PANEL}; }}
QCheckBox::indicator:checked:disabled {{ background: {GRAY}; border-color: {GRAY}; }}
QRadioButton::indicator {{ width: 15px; height: 15px; border: 1px solid {GRAY}; border-radius: 8px; background: {BG}; }}
QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QRadioButton::indicator:checked {{ width: 7px; height: 7px; border: 5px solid {ACCENT}; background: #ffffff; }}
QRadioButton::indicator:disabled {{ border-color: {BORDER}; background: {PANEL}; }}
QRadioButton::indicator:checked:disabled {{ border-color: {GRAY}; background: {PANEL}; }}

QPushButton {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
               padding: 5px 12px; min-height: 18px; }}
QPushButton:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {GRAY}; background: {PANEL}; }}
QPushButton[primary="true"] {{ background: {ACCENT}; border-color: {ACCENT}; color: #0d1117; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {ACCENT_HOVER}; }}
QPushButton[primary="true"]:disabled {{ background: {RAISED}; border-color: {BORDER}; color: {GRAY}; }}
QPushButton[danger="true"] {{ color: {RED}; }}
QPushButton[flat="true"] {{ background: transparent; border: none; color: {MUTED}; padding: 4px 8px; }}
QPushButton[flat="true"]:hover {{ color: {TEXT}; background: {HOVER}; }}

QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 4px 6px; }}
QToolButton:hover {{ background: {HOVER}; }}
QToolButton:disabled {{ color: {GRAY}; }}
QToolButton#sectionHeader {{ text-align: left; font-weight: 600; padding: 6px 4px; border: none; }}
QToolButton#sectionHeader:hover {{ background: {RAISED}; }}

QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 8px;
    selection-background-color: {ACCENT}; selection-color: #0d1117; }}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: {GRAY}; background: {PANEL}; }}
QLineEdit[invalid="true"] {{ border-color: {RED}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {RAISED}; border: 1px solid {BORDER};
                               selection-background-color: {ACCENT}; selection-color: #0d1117; }}

QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 8px; margin: 2px 6px; }}
QListWidget::item:hover {{ background: {RAISED}; }}
QListWidget::item:selected {{ background: {HOVER}; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 16px; border-bottom: 2px solid transparent;
                font-weight: 600; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}

QTableWidget {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; gridline-color: {BORDER}; }}
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

QProgressBar {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 4px; max-height: 8px; min-height: 8px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QSplitter::handle {{ background: {BORDER}; width: 1px; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
QStatusBar::item {{ border: none; }}
QMenu {{ background: {RAISED}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT}; color: #0d1117; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
"""


def make_icon() -> QIcon:
    """The app icon: a rounded tile with a play triangle, drawn at several sizes."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        margin = size * 0.06
        rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
        gradient = QLinearGradient(0, 0, size, size)
        gradient.setColorAt(0.0, QColor("#5b9dff"))
        gradient.setColorAt(1.0, QColor("#8a5cf6"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, size * 0.22, size * 0.22)
        path = QPainterPath()
        path.moveTo(QPointF(size * 0.38, size * 0.28))
        path.lineTo(QPointF(size * 0.74, size * 0.50))
        path.lineTo(QPointF(size * 0.38, size * 0.72))
        path.closeSubpath()
        painter.setBrush(QColor("#ffffff"))
        painter.drawPath(path)
        painter.end()
        icon.addPixmap(pixmap)
    return icon
