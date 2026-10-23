"""Small shared widgets and helpers."""
from __future__ import annotations

import os
import subprocess
from datetime import datetime

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImageReader, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from .. import config
from ..db import Photo
from . import theme
from .tilegrid import Section, Tile


def button(text: str = "", icon: str | None = None, *, primary=False, flat=False, danger=False,
           tooltip: str | None = None) -> QPushButton:
    b = QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    if icon:
        color = theme.C["on_accent"] if primary else theme.C["danger"] if danger else theme.C["text"]
        b.setIcon(theme.icon(icon, color, 16))
    for prop, on in (("primary", primary), ("flat", flat), ("danger", danger)):
        if on:
            b.setProperty(prop, True)
    if tooltip:
        b.setToolTip(tooltip)
    return b


def label(text: str = "", name: str | None = None, wrap=False) -> QLabel:
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


class SelectionBar(QFrame):
    """Shown above a grid while items are selected."""

    cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SelectionBar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 12, 6)
        lay.setSpacing(6)
        self.close_btn = QPushButton()
        self.close_btn.setProperty("flat", True)
        self.close_btn.setIcon(theme.icon("close", size=16))
        self.close_btn.setToolTip("Clear selection (Esc)")
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.clicked.connect(self.cleared.emit)
        self.text = QLabel()
        lay.addWidget(self.close_btn)
        lay.addWidget(self.text)
        lay.addStretch(1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(6)
        lay.addLayout(self.actions)
        self.hide()

    def add(self, b: QPushButton) -> QPushButton:
        self.actions.addWidget(b)
        return b

    def update_count(self, n: int, noun: str = "selected") -> None:
        self.text.setText(f"{n:,} {noun}")
        self.setVisible(n > 0)


class Segmented(QFrame):
    changed = Signal(int)

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        self.setObjectName("Segment")
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons: list[QPushButton] = []
        for i, text in enumerate(labels):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.group.addButton(b, i)
            lay.addWidget(b)
            self.buttons.append(b)
        self.buttons[0].setChecked(True)
        self.group.idClicked.connect(self.changed.emit)

    def set_text(self, i: int, text: str) -> None:
        self.buttons[i].setText(text)

    def current(self) -> int:
        return self.group.checkedId()

    def set_current(self, i: int) -> None:
        self.buttons[i].setChecked(True)


def circle_pixmap(path: str | None, size: int) -> QPixmap:
    dpr = 2
    out = QPixmap(size * dpr, size * dpr)
    out.setDevicePixelRatio(dpr)
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    clip = QPainterPath()
    clip.addEllipse(QRectF(0, 0, size, size))
    p.setClipPath(clip)
    src = QPixmap()
    if path and os.path.exists(path):
        reader = QImageReader(path)
        src = QPixmap.fromImage(reader.read())
    if src.isNull():
        p.fillRect(QRectF(0, 0, size, size), QColor(theme.C["placeholder"]))
        ic = theme.icon_pixmap("person", theme.C["muted"], int(size * 0.55))
        p.drawPixmap(int(size * 0.225), int(size * 0.225), ic)
    else:
        s = min(src.width(), src.height())
        p.drawPixmap(QRectF(0, 0, size, size), src,
                     QRectF((src.width() - s) / 2, (src.height() - s) / 2, s, s))
    p.end()
    return out


def face_image(face_id: int | None) -> str:
    return str(config.face_crop_path(face_id)) if face_id else ""


def photo_tile(ph: Photo) -> Tile:
    return Tile(key=ph.id, image=str(config.thumb_path(ph.id)), fallback=ph.path, data=ph)


def month_sections(photos: list[Photo]) -> list[Section]:
    sections: list[Section] = []
    current = None
    for ph in photos:
        try:
            dt = datetime.fromtimestamp(ph.taken_at) if ph.taken_at else None
        except (OSError, OverflowError, ValueError):
            dt = None
        key = dt.strftime("%B %Y") if dt else "Unknown date"
        if key != current:
            sections.append(Section(key))
            current = key
        sections[-1].tiles.append(photo_tile(ph))
    for s in sections:
        s.subtitle = f"{len(s.tiles):,} photo{'s' if len(s.tiles) != 1 else ''}"
    return sections


def plural(n: int, word: str) -> str:
    if n == 1:
        return f"1 {word}"
    return f"{n:,} {'people' if word == 'person' else word + 's'}"


def show_in_explorer(path: str) -> None:
    if os.path.exists(path):
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    elif os.path.isdir(os.path.dirname(path)):
        os.startfile(os.path.dirname(path))


def open_default(path: str) -> None:
    try:
        os.startfile(path)
    except OSError:
        pass


def hline() -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {theme.C['border']};")
    return f


def spacer_widget() -> QWidget:
    w = QWidget()
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    return w
