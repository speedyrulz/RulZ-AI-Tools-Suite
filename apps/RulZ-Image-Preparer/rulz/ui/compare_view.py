"""Before/after viewer: side-by-side panes or a draggable split slider, with shared zoom and pan.

In paint mode, left-dragging paints (or erases) shadow areas on the picture; right- or middle-dragging
still pans. The painted areas show as a red tint on the original.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ..shadows import ShadowEdit, Stroke, render_mask
from .theme import C

SIDE, SLIDER = "side", "slider"
PAINT, ERASE = "paint", "erase"
GAP = 6
MIN_ZOOM, MAX_ZOOM = 0.02, 32.0
OVERLAY_MAX_SIDE = 1024


class CompareView(QWidget):
    zoomChanged = Signal(float)  # current scale (1.0 = 100%)
    stroked = Signal(object)  # a finished shadows.Stroke

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
        self._drag: str | None = None  # "pan", "split" or "paint"
        self._last = QPointF()
        self.tool: str | None = None  # PAINT, ERASE or None
        self.brush = 0.05  # radius as a fraction of the picture's longer side
        self.show_areas = False
        self._edit: ShadowEdit | None = None
        self._overlay: QPixmap | None = None
        self._stroke: list[QPointF] = []  # picture coordinates (pixels of the shown pixmap)
        self._mouse: QPointF | None = None

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
        if self._edit is not None:
            self.set_shadow(self._edit)  # the overlay follows the new picture size
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

    # ---- shadow painting -------------------------------------------------
    def set_tool(self, tool: str | None):
        self.tool = tool
        self._stroke = []
        if self._drag == "paint":
            self._drag = None
        self.setCursor(Qt.BlankCursor if tool else Qt.OpenHandCursor)
        self.update()

    def set_brush(self, fraction: float):
        self.brush = fraction
        self.update()

    def set_show_areas(self, on: bool):
        self.show_areas = on
        self.update()

    def set_shadow(self, edit: ShadowEdit | None):
        """The painted areas to show; their tint is redrawn at a modest size and scaled up."""
        self._edit = edit
        self._overlay = None
        if edit is not None and edit.strokes and self._orig is not None:
            w, h = self._orig.width(), self._orig.height()
            f = min(1.0, OVERLAY_MAX_SIDE / max(w, h))
            ow, oh = max(1, round(w * f)), max(1, round(h * f))
            m = render_mask(edit, oh, ow, feather=False)
            rgba = np.zeros((oh, ow, 4), np.uint8)
            rgba[..., 0] = 237
            rgba[..., 3] = (m * 110).astype(np.uint8)
            q = QImage(rgba.data, ow, oh, rgba.strides[0], QImage.Format_RGBA8888)
            self._overlay = QPixmap.fromImage(q.copy())
        self.update()

    def _to_picture(self, pos: QPointF) -> QPointF:
        t = self._target(self._pane_at(pos))
        s = self.scale()
        return QPointF((pos.x() - t.left()) / s, (pos.y() - t.top()) / s)

    def _brush_px(self) -> float:
        """Brush radius in picture pixels."""
        return self.brush * max(self._orig.width(), self._orig.height())

    def _finish_stroke(self):
        if self._stroke and self._orig is not None:
            w, h = self._orig.width(), self._orig.height()
            pts = tuple((min(1.0, max(0.0, p.x() / w)), min(1.0, max(0.0, p.y() / h))) for p in self._stroke)
            self.stroked.emit(Stroke(pts, self.brush, self.tool == ERASE))
        self._stroke = []

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
            self._areas(p, left)
            self._draw(p, right, proc)
            self._label(p, left, "Original", Qt.AlignLeft)
            self._label(p, right, "Processing…" if self._busy else "Processed", Qt.AlignLeft)
        else:
            pane = self._panes()[0]
            sx = self._split_x()
            self._draw(p, pane, proc)
            orig_clip = QRectF(pane.left(), pane.top(), sx - pane.left(), pane.height())
            self._draw(p, orig_clip, self._orig, pane)
            self._areas(p, orig_clip, pane)
            self._label(p, pane, "Original", Qt.AlignLeft)
            self._label(p, pane, "Processing…" if self._busy else "Processed", Qt.AlignRight)
            self._divider(p, sx)
        self._brush_cursor(p)
        p.end()

    def _areas(self, p: QPainter, clip: QRectF, pane: QRectF | None = None):
        """The painted areas (and the stroke being painted) as a red tint over the original."""
        if not (self.show_areas or self.tool) or (self._overlay is None and not self._stroke):
            return
        target = self._target(pane or clip)
        p.save()
        p.setClipRect(clip)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if self._overlay is not None:
            p.drawPixmap(target, self._overlay, QRectF(self._overlay.rect()))
        if self._stroke:
            s = self.scale()
            pts = [QPointF(target.left() + q.x() * s, target.top() + q.y() * s) for q in self._stroke]
            color = QColor(0, 0, 0, 150) if self.tool == ERASE else QColor(237, 0, 13, 110)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(QPen(color, 2 * self._brush_px() * s, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            if len(pts) == 1:
                p.drawPoint(pts[0])
            else:
                p.drawPolyline(pts)
        p.restore()

    def _brush_cursor(self, p: QPainter):
        if not self.tool or self._mouse is None or self._orig is None:
            return
        r = self._brush_px() * self.scale()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(Qt.NoBrush)
        for color, width in ((QColor(0, 0, 0, 160), 3), (QColor("#ffffff"), 1.2)):
            p.setPen(QPen(color, width, Qt.DashLine if self.tool == ERASE else Qt.SolidLine))
            p.drawEllipse(self._mouse, r, r)

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
        if e.button() == Qt.LeftButton and self.tool and self._orig is not None:
            self._drag = "paint"
            self._stroke = [self._to_picture(pos)]
            self.update()
        elif e.button() == Qt.LeftButton and self._near_split(pos):
            self._drag = "split"
        elif e.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._drag = "pan"
            self.setCursor(Qt.ClosedHandCursor)
        self._last = pos

    def mouseMoveEvent(self, e):
        pos = e.position()
        self._mouse = pos
        if self._drag == "paint":
            q = self._to_picture(pos)
            # keep points a few screen pixels apart so long strokes stay light
            if (q - self._stroke[-1]).manhattanLength() * self.scale() >= 3:
                self._stroke.append(q)
            self.update()
        elif self._drag == "split":
            self._split = max(0.0, min(1.0, pos.x() / max(1, self.width())))
            self.update()
        elif self._drag == "pan" and self._orig is not None:
            if self._zoom is None:  # start panning from the fitted view
                self._zoom = self._fit_scale()
                self._center = QPointF(self._orig.width() / 2, self._orig.height() / 2)
            self._center -= (pos - self._last) / self.scale()
            self._clamp()
            self.update()
        elif self.tool:
            self.setCursor(Qt.BlankCursor)
            self.update()
        else:
            self.setCursor(Qt.SplitHCursor if self._near_split(pos) else Qt.OpenHandCursor)
        self._last = pos

    def mouseReleaseEvent(self, e):
        if self._drag == "paint":
            self._finish_stroke()
        self._drag = None
        if self.tool:
            self.setCursor(Qt.BlankCursor)
            self.update()
        else:
            self.setCursor(Qt.SplitHCursor if self._near_split(e.position()) else Qt.OpenHandCursor)

    def leaveEvent(self, e):
        self._mouse = None
        self.update()
        super().leaveEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self.tool and e.button() == Qt.LeftButton:
            self.mousePressEvent(e)  # a quick second dab, not a zoom
            return
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
