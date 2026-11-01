"""Colors, stylesheet and icon rendering (RulZ dark: black + logo red #ED000D)."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap

C = dict(
    bg="#0b0b0c", surface="#111113", card="#141416", border="#232328",
    text="#f2f2f3", muted="#a3a3ab", accent="#ed000d", accent_hover="#ff1f2d", on_accent="#ffffff",
    select="#3a0c10", hover="#1c1c21", danger="#ff4d58", placeholder="#1f1f24", input="#18181c",
)

# Segoe Fluent Icons / Segoe MDL2 Assets code points
GLYPH = dict(
    add="", folder="", back="", prev="", next="", save="",
    delete="", clear="", image="", reset="", fit="", zoom="", crop="",
)


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


def icon(name: str, color: str | None = None, size: int = 20) -> QIcon:
    return QIcon(icon_pixmap(name, color or C["text"], size))


def logo_file() -> str:
    from .. import config

    return str(config.resource_path("assets", "logo_full_white.png"))


def apply(app) -> None:
    try:  # native title bar follows the app's dark look instead of the Windows setting
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except Exception:
        pass
    app.setStyle("Fusion")
    pal = QPalette()
    for role, key in (
        (QPalette.Window, "bg"), (QPalette.WindowText, "text"), (QPalette.Base, "card"),
        (QPalette.AlternateBase, "surface"), (QPalette.Text, "text"), (QPalette.Button, "card"),
        (QPalette.ButtonText, "text"), (QPalette.Highlight, "accent"),
        (QPalette.HighlightedText, "on_accent"), (QPalette.ToolTipBase, "card"),
        (QPalette.ToolTipText, "text"), (QPalette.PlaceholderText, "muted"), (QPalette.Mid, "border"),
    ):
        pal.setColor(role, QColor(C[key]))
    app.setPalette(pal)
    f = QFont("Segoe UI Variable Text")
    if f.family() not in QFontDatabase.families():
        f = QFont("Segoe UI")
    f.setPointSizeF(10)
    app.setFont(f)
    app.setStyleSheet(stylesheet())


def stylesheet() -> str:
    return """
QWidget { color: %(text)s; }
QMainWindow, #Page { background: %(bg)s; }
#TopBar, #ToolBar { background: %(surface)s; border-bottom: 1px solid %(border)s; }
#BottomBar { background: %(surface)s; border-top: 1px solid %(border)s; }
#PageTitle { font-size: 18px; font-weight: 700; }
#H2 { font-size: 14px; font-weight: 600; }
#Muted { color: %(muted)s; }
#Value { font-size: 14px; font-weight: 700; color: %(accent)s; min-width: 34px; }
QSpinBox#Value { background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 2px 4px; min-height: 0px; }
QSpinBox#Value:hover { border-color: %(border)s; background: %(input)s; }
QSpinBox#Value:focus { border-color: %(accent)s; background: %(input)s; color: %(text)s; }
#DropHint { color: %(muted)s; font-size: 15px; border: 2px dashed %(border)s; border-radius: 16px; }
#DropHint[active="true"] { border-color: %(accent)s; color: %(text)s; background: %(select)s; }

QPushButton {
    background: %(card)s; border: 1px solid %(border)s; border-radius: 8px;
    padding: 7px 14px; font-weight: 500;
}
QPushButton:hover { background: %(hover)s; }
QPushButton:pressed { background: %(select)s; }
QPushButton:disabled { color: %(muted)s; }
QPushButton[primary="true"] { background: %(accent)s; color: %(on_accent)s; border: 1px solid %(accent)s; }
QPushButton[primary="true"]:hover { background: %(accent_hover)s; }
QPushButton[primary="true"]:disabled { background: %(placeholder)s; border-color: %(border)s; color: %(muted)s; }
QPushButton[flat="true"] { background: transparent; border: none; }
QPushButton[flat="true"]:hover { background: %(hover)s; }

#Segment { background: %(input)s; border-radius: 10px; }
#Segment QPushButton { border: none; border-radius: 8px; padding: 6px 14px; background: transparent; font-weight: 600; }
#Segment QPushButton:checked { background: %(card)s; color: %(accent)s; }

QListWidget { background: %(bg)s; border: none; outline: none; }

QSlider::groove:horizontal { height: 4px; background: %(placeholder)s; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::handle:horizontal { background: %(accent)s; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }
QSlider::handle:horizontal:hover { background: %(accent_hover)s; }

QMenu { background: %(card)s; border: 1px solid %(border)s; border-radius: 10px; padding: 6px; }
QMenu::item { padding: 7px 26px 7px 12px; border-radius: 6px; }
QMenu::item:selected { background: %(hover)s; }

QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar::handle:vertical { background: %(border)s; border-radius: 4px; min-height: 40px; margin: 0 2px; }
QScrollBar::handle:vertical:hover { background: %(muted)s; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QProgressDialog, QDialog, QMessageBox { background: %(bg)s; }
QProgressBar {
    background: %(placeholder)s; border: none; border-radius: 6px; min-height: 18px; max-height: 18px;
    text-align: center; color: %(text)s; font-size: 11px; font-weight: 600;
}
QProgressBar::chunk { background: %(accent)s; border-radius: 6px; }
QToolTip { background: %(card)s; color: %(text)s; border: 1px solid %(border)s; padding: 4px 8px; }

QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
    background: %(input)s; border: 1px solid %(border)s; border-radius: 6px; padding: 5px 8px; min-height: 18px;
}
/* Every spin box in the app has no arrow buttons; without this, the styled box still reserves room for
   them and the text field collapses to nothing. The inner text field also mustn't get the QLineEdit box. */
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 0px; height: 0px; border: 0; image: none; subcontrol-origin: border;
}
QAbstractSpinBox QLineEdit { background: transparent; border: none; padding: 0px 7px; min-height: 0px; }
QSpinBox#Value QLineEdit { padding: 0px; }
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus { border-color: %(accent)s; }
QComboBox QAbstractItemView { background: %(card)s; border: 1px solid %(border)s; selection-background-color: %(hover)s; }
QCheckBox { spacing: 8px; }
#Section { color: %(muted)s; font-weight: 600; padding-top: 10px; }
#PreviewPanel { background: %(surface)s; border: 1px solid %(border)s; border-radius: 10px; }
""" % C
