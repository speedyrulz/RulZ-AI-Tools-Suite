"""Timeline bar: keyframe ticks, in/out region or draggable fixed-length
selection box, and the playhead. Click/drag to seek."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget

from . import theme

DRAG_THRESHOLD = 3


class Timeline(QWidget):
    seekRequested = Signal(float)           # seconds
    selectionMoveRequested = Signal(float)  # new selection start, seconds

    _pad = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(56)
        self.setCursor(Qt.PointingHandCursor)
        self.start = 0.0
        self.end = 1.0
        self.position = 0.0
        self.keyframes: list[float] = []
        self.mark_in: float | None = None
        self.mark_out: float | None = None
        self.selection: tuple[float, float] | None = None  # fixed-length mode
        self._press = None  # (x, mode, offset_seconds)
        self._dragging = False

    def set_range(self, start: float, end: float):
        self.start, self.end = start, max(end, start + 1e-6)
        self.update()

    def set_keyframes(self, kfs: list[float]):
        self.keyframes = kfs
        self.update()

    def set_position(self, t: float):
        self.position = t
        self.update()

    def set_marks(self, mark_in, mark_out):
        self.mark_in, self.mark_out = mark_in, mark_out
        self.update()

    def set_selection(self, sel: tuple[float, float] | None):
        self.selection = sel
        self.update()

    # --- geometry ---------------------------------------------------------
    def _x(self, t: float) -> float:
        w = self.width() - 2 * self._pad
        return self._pad + (t - self.start) / (self.end - self.start) * w

    def _t(self, x: float) -> float:
        w = self.width() - 2 * self._pad
        frac = min(max((x - self._pad) / max(w, 1), 0.0), 1.0)
        return self.start + frac * (self.end - self.start)

    def _in_selection(self, x: float) -> bool:
        if not self.selection:
            return False
        a, b = self.selection
        return self._x(a) - 2 <= x <= self._x(b) + 2

    # --- events -----------------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        x = e.position().x()
        shift = bool(e.modifiers() & Qt.ShiftModifier)
        if self.selection and not shift:
            if self._in_selection(x):
                # potential box drag; a plain click seeks inside the box
                self._press = (x, "box", self._t(x) - self.selection[0])
                self._dragging = False
                return
            self.selectionMoveRequested.emit(self._t(x))
            self._press = (x, "box", 0.0)
            self._dragging = True
            return
        self._press = (x, "seek", 0.0)
        self.seekRequested.emit(self._t(x))

    def mouseMoveEvent(self, e):
        if not (e.buttons() & Qt.LeftButton) or not self._press:
            return
        x = e.position().x()
        px, mode, offset = self._press
        if mode == "seek":
            self.seekRequested.emit(self._t(x))
            return
        if not self._dragging and abs(x - px) < DRAG_THRESHOLD:
            return
        self._dragging = True
        self.selectionMoveRequested.emit(self._t(x) - offset)

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or not self._press:
            return
        x, mode, _ = self._press
        if mode == "box" and not self._dragging:
            self.seekRequested.emit(self._t(e.position().x()))
        self._press = None
        self._dragging = False

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h = self.height()
        bar_top, bar_bot = 18, h - 14

        # base bar
        p.fillRect(QRectF(self._pad, bar_top, self.width() - 2 * self._pad, bar_bot - bar_top),
                   QColor(28, 28, 32))
        p.setPen(QPen(QColor(52, 52, 58), 1))
        p.drawRect(QRectF(self._pad, bar_top, self.width() - 2 * self._pad, bar_bot - bar_top))

        if self.selection:
            a, b = self.selection
            xa, xb = self._x(a), self._x(b)
            p.fillRect(QRectF(xa, bar_top, xb - xa, bar_bot - bar_top), QColor(226, 18, 28, 120))
            p.setPen(QPen(theme.Q_RED_HOVER, 2))
            p.drawRect(QRectF(xa, bar_top, xb - xa, bar_bot - bar_top))
            # grip
            cx = (xa + xb) / 2
            cy = (bar_top + bar_bot) / 2
            if xb - xa > 18:
                p.setPen(QPen(QColor(255, 255, 255), 1))
                for d in (-4, 0, 4):
                    p.drawLine(QPointF(cx + d, cy - 4), QPointF(cx + d, cy + 4))
        elif self.mark_in is not None or self.mark_out is not None:
            a = self.mark_in if self.mark_in is not None else self.start
            b = self.mark_out if self.mark_out is not None else self.end
            if b > a:
                p.fillRect(QRectF(self._x(a), bar_top, self._x(b) - self._x(a), bar_bot - bar_top),
                           QColor(125, 10, 16, 190))

        # keyframe ticks (thin out when there are too many for the width)
        if self.keyframes:
            p.setPen(QPen(QColor(205, 205, 212), 1))
            last_x = -10.0
            for k in self.keyframes:
                x = self._x(k)
                if x - last_x < 2:
                    continue
                last_x = x
                p.drawLine(QPointF(x, bar_top), QPointF(x, bar_top + 8))

        # in/out markers (hidden in fixed-length mode)
        if not self.selection:
            if self.mark_in is not None:
                p.setPen(QPen(QColor(255, 255, 255), 2))
                x = self._x(self.mark_in)
                p.drawLine(QPointF(x, bar_top - 6), QPointF(x, bar_bot + 4))
                p.drawText(QPointF(x + 3, bar_bot + 12), "IN")
            if self.mark_out is not None:
                p.setPen(QPen(theme.Q_RED_HOVER, 2))
                x = self._x(self.mark_out)
                p.drawLine(QPointF(x, bar_top - 6), QPointF(x, bar_bot + 4))
                p.drawText(QPointF(x - 24, bar_bot + 12), "OUT")

        # playhead
        x = self._x(self.position)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(QPointF(x, bar_top - 10), QPointF(x, bar_bot + 6))
        p.end()
