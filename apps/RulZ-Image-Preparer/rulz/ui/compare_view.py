"""Before/after viewer: side-by-side panes or a draggable split slider, with shared zoom and pan."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from .theme import C

SIDE, SLIDER = "side", "slider"
GAP = 6
MIN_ZOOM, MAX_ZOOM = 0.02, 32.0


class CompareView(QWidget):
    zoomChanged = Signal(float)  # current scale (1.0 = 100%)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)
        self._orig: QPixmap | None = None
        self._proc: QPixmap | None = None
        self._busy = False
        self.mode = SIDE
        self._zoom: float | None = None  # None = fit to window
        self._center = QPointF()  # image coordinates shown at each pane's center
        self._split = 0.5  # slider position, fraction of the width
        self._drag: str | None = None  # "pan" or "split"
        self._last = QPointF()

    # ---- content ---------------------------------------------------------
    def set_images(self, orig: QPixmap | None, proc: QPixmap | None, keep_view: bool = False):
        same_size = (
            self._orig is not None and orig is not None and self._orig.size() == orig.size()
        )
        self._orig, self._proc = orig, proc
        if not (keep_view and same_size):
            self._zoom = None
            if orig is not None:
                self._center = QPointF(orig.width() / 2, orig.height() / 2)
        self._emit_zoom()
        self.update()

    def set_processed(self, proc: QPixmap | None):
        self._proc = proc
        self.update()

    def set_busy(self, busy: bool):
        self._busy = busy
        self.update()

    def set_mode(self, mode: str):
        self.mode = mode
        self._emit_zoom()
        self.update()

    # ---- geometry --------------------------------------------------------
    def _panes(self) -> list[QRectF]:
        r = QRectF(self.rect())
        if self.mode == SIDE:
            w = (r.width() - GAP) / 2
            return [QRectF(r.left(), r.top(), w, r.height()), QRectF(r.left() + w + GAP, r.top(), w, r.height())]
        return [r]

    def _fit_scale(self) -> float:
        if self._orig is None:
            return 1.0
        pane = self._panes()[0]
        return max(MIN_ZOOM, min(pane.width() / self._orig.width(), pane.height() / self._orig.height()))

    def scale(self) -> float:
        return self._fit_scale() if self._zoom is None else self._zoom

    def _emit_zoom(self):
        self.zoomChanged.emit(self.scale())

    def _target(self, pane: QRectF) -> QRectF:
        """Where the whole image lands inside a pane."""
        s = self.scale()
        c = self._center if self._zoom is not None else QPointF(self._orig.width() / 2, self._orig.height() / 2)
        x = pane.center().x() - c.x() * s
        y = pane.center().y() - c.y() * s
        return QRectF(x, y, self._orig.width() * s, self._orig.height() * s)

    def _pane_at(self, pos: QPointF) -> QRectF:
        for p in self._panes():
            if p.contains(pos):
                return p
        return self._panes()[0]

    def _split_x(self) -> float:
        return self.width() * self._split

    # ---- zoom control ----------------------------------------------------
    def fit(self):
        self._zoom = None
        if self._orig is not None:
            self._center = QPointF(self._orig.width() / 2, self._orig.height() / 2)
        self._emit_zoom()
        self.update()

    def set_zoom(self, zoom: float, anchor: QPointF | None = None):
        if self._orig is None:
            return
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        if anchor is None:
            anchor = self._pane_at(QPointF(self.width() / 2, self.height() / 2)).center()
        pane = self._pane_at(anchor)
        old_s = self.scale()
        if self._zoom is None:
            self._center = QPointF(self._orig.width() / 2, self._orig.height() / 2)
        # keep the image point under the anchor fixed
        img_pt = self._center + (anchor - pane.center()) / old_s
        self._zoom = zoom
        self._center = img_pt - (anchor - pane.center()) / zoom
        self._clamp()
        self._emit_zoom()
        self.update()

    def _clamp(self):
        if self._orig is None:
            return
        self._center.setX(max(0.0, min(float(self._orig.width()), self._center.x())))
        self._center.setY(max(0.0, min(float(self._orig.height()), self._center.y())))

    # ---- painting --------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#060607"))
        if self._orig is None:
            p.end()
            return
        smooth = self.scale() < 1.0
        p.setRenderHint(QPainter.SmoothPixmapTransform, smooth)
        proc = self._proc or self._orig
        if self.mode == SIDE:
            left, right = self._panes()
            self._draw(p, left, self._orig)
            self._draw(p, right, proc)
            self._label(p, left, "Original", Qt.AlignLeft)
            self._label(p, right, "Processing…" if self._busy else "Processed", Qt.AlignLeft)
        else:
            pane = self._panes()[0]
            sx = self._split_x()
            self._draw(p, pane, proc)
            self._draw(p, QRectF(pane.left(), pane.top(), sx - pane.left(), pane.height()), self._orig, pane)
            self._label(p, pane, "Original", Qt.AlignLeft)
            self._label(p, pane, "Processing…" if self._busy else "Processed", Qt.AlignRight)
            self._divider(p, sx)
        p.end()

    def _draw(self, p: QPainter, clip: QRectF, pm: QPixmap, pane: QRectF | None = None):
        target = self._target(pane or clip)
        p.save()
        p.setClipRect(clip)
        visible = target.intersected(clip)
        if not visible.isEmpty():
            s = self.scale()
            src = QRectF(
                (visible.left() - target.left()) / s, (visible.top() - target.top()) / s,
                visible.width() / s, visible.height() / s,
            )
            p.drawPixmap(visible, pm, src)
        p.restore()

    def _label(self, p: QPainter, pane: QRectF, text: str, align):
        f = QFont(self.font())
        f.setPixelSize(12)
        f.setBold(True)
        p.setFont(f)
        w = p.fontMetrics().horizontalAdvance(text) + 20
        x = pane.left() + 12 if align == Qt.AlignLeft else pane.right() - 12 - w
        r = QRectF(x, pane.top() + 12, w, 24)
        path = QPainterPath()
        path.addRoundedRect(r, 12, 12)
        p.fillPath(path, QColor(0, 0, 0, 170))
        p.setPen(QColor("#ffffff"))
        p.drawText(r, Qt.AlignCenter, text)

    def _divider(self, p: QPainter, x: float):
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        c = QPointF(x, self.height() / 2)
        p.setBrush(QColor(C["accent"]))
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawEllipse(c, 15, 15)
        p.setPen(QPen(QColor("#ffffff"), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for d in (-1, 1):  # little arrows
            tip = QPointF(x + d * 9, c.y())
            p.drawPolyline([QPointF(x + d * 4, c.y() - 5), tip, QPointF(x + d * 4, c.y() + 5)])

    # ---- mouse -----------------------------------------------------------
    def _near_split(self, pos: QPointF) -> bool:
        return self.mode == SLIDER and abs(pos.x() - self._split_x()) <= 12

    def mousePressEvent(self, e):
        pos = e.position()
        if e.button() == Qt.LeftButton and self._near_split(pos):
            self._drag = "split"
        elif e.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._drag = "pan"
            self.setCursor(Qt.ClosedHandCursor)
        self._last = pos

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._drag == "split":
            self._split = max(0.0, min(1.0, pos.x() / max(1, self.width())))
            self.update()
        elif self._drag == "pan" and self._orig is not None:
            if self._zoom is None:  # start panning from the fitted view
                self._zoom = self._fit_scale()
                self._center = QPointF(self._orig.width() / 2, self._orig.height() / 2)
            self._center -= (pos - self._last) / self.scale()
            self._clamp()
            self.update()
        else:
            self.setCursor(Qt.SplitHCursor if self._near_split(pos) else Qt.OpenHandCursor)
        self._last = pos

    def mouseReleaseEvent(self, e):
        self._drag = None
        self.setCursor(Qt.SplitHCursor if self._near_split(e.position()) else Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, e):
        if self._near_split(e.position()):
            self._split = 0.5
            self.update()
        elif self._zoom is None or abs(self._zoom - 1.0) > 1e-3:
            self.set_zoom(1.0, e.position())
        else:
            self.fit()

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120.0
        if steps:
            self.set_zoom(self.scale() * (1.25 ** steps), e.position())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._emit_zoom()
