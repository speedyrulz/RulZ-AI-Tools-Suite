"""Image view: zoom/pan, grid overlay, guides, loupe and tool overlays.

From RulZ Image Distortion Repair; the background follows the Clip Extractor theme.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QLabel

import numpy as np

from ... import theme


def to_qimage(rgba: np.ndarray) -> QImage:
    if rgba.dtype != np.uint8:
        rgba = (rgba >> 8).astype(np.uint8)
    rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    img = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
    return img.copy()


class ImageView(QGraphicsView):
    zoomChanged = Signal(float, bool)      # factor (screen px per image px), is_fit
    guidesChanged = Signal()

    GUIDE_GRAB = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setMouseTracking(True)
        self.pix = QGraphicsPixmapItem()
        self.pix.setTransformationMode(Qt.SmoothTransformation)
        self.pix.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
        self.scene().addItem(self.pix)
        self.image_rect = QRectF()
        self.fit_mode = True
        self.factor = 1.0
        self.bg = QColor(theme.BG)
        self.show_grid = False
        self.grid_size = 50
        self.guides_v: list[float] = []
        self.guides_h: list[float] = []
        self.guides_visible = True
        self.guide_edit: str | None = None    # "v" | "h" | None
        self.guide_color = QColor("#ff3b44")
        self._drag_guide = None
        self._pan_from = None
        self.overlay = None
        self.loupe_pos: QPointF | None = None
        self.caption = QLabel(self)
        self.caption.setObjectName("sizeLabel")
        self.caption.hide()
        self.setMinimumSize(200, 200)

    # -- image -----------------------------------------------------------------------
    def set_image(self, qimg: QImage | None, scene_w: float = 0, scene_h: float = 0):
        if qimg is None or qimg.isNull():
            self.pix.setPixmap(QPixmap())
            self.image_rect = QRectF()
            self.viewport().update()
            return
        changed = abs(self.image_rect.width() - scene_w) > 0.5 or abs(self.image_rect.height() - scene_h) > 0.5
        pm = QPixmap.fromImage(qimg)
        self.pix.setPixmap(pm)
        self.pix.setTransform(QTransform.fromScale(scene_w / pm.width(), scene_h / pm.height()))
        self.image_rect = QRectF(0, 0, scene_w, scene_h)
        self.scene().setSceneRect(self.image_rect.adjusted(-1e5, -1e5, 1e5, 1e5))
        if self.overlay is not None:
            self.overlay.set_image_rect(self.image_rect)
        if changed and self.fit_mode:
            self.fit()
        elif changed:
            self._apply_zoom(self.factor, self.mapToScene(self.viewport().rect().center()))
        self.viewport().update()

    def set_caption(self, text: str | None):
        if not text:
            self.caption.hide()
            return
        self.caption.setText(text)
        self.caption.adjustSize()
        self._place_caption()
        self.caption.show()

    def _place_caption(self):
        self.caption.move(self.width() - self.caption.width() - 12, self.height() - self.caption.height() - 6)

    # -- zoom & pan --------------------------------------------------------------------
    def fit(self):
        self.fit_mode = True
        if self.image_rect.isEmpty():
            return
        vr = self.viewport().rect().adjusted(12, 12, -12, -12)
        f = min(vr.width() / self.image_rect.width(), vr.height() / self.image_rect.height())
        self._apply_zoom(f, self.image_rect.center())
        self.zoomChanged.emit(self.factor, True)

    def one_to_one(self, anchor: QPointF | None = None):
        self.fit_mode = False
        dpr = self.devicePixelRatioF()
        self._apply_zoom(1.0 / dpr, anchor or self.image_rect.center())
        self.zoomChanged.emit(self.factor, False)

    def zoom_by(self, k: float, anchor_view=None):
        if self.image_rect.isEmpty():
            return
        self.fit_mode = False
        anchor_view = anchor_view or self.viewport().rect().center()
        before = self.mapToScene(anchor_view)
        f = float(np.clip(self.factor * k, 0.02, 16.0))
        self.setTransform(QTransform.fromScale(f, f))
        self.factor = f
        after = self.mapToScene(anchor_view)
        self._shift(before - after)
        self.zoomChanged.emit(self.factor, False)

    def _apply_zoom(self, f: float, center: QPointF):
        self.factor = f
        self.setTransform(QTransform.fromScale(f, f))
        self.centerOn(center)

    def _shift(self, d: QPointF):
        c = self.mapToScene(self.viewport().rect().center())
        self.centerOn(c + d)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.fit_mode:
            self.fit()
        self._place_caption()

    def wheelEvent(self, e):
        k = 1.0015 ** e.angleDelta().y()
        self.zoom_by(k, e.position().toPoint())

    # -- mouse: guides and panning (overlay items get first chance through the scene) --
    def _norm(self, scene_pt: QPointF):
        r = self.image_rect
        return (scene_pt.x() - r.x()) / r.width(), (scene_pt.y() - r.y()) / r.height()

    def _guide_at(self, view_pt):
        r = self.image_rect
        for i, x in enumerate(self.guides_v):
            vx = self.mapFromScene(QPointF(r.x() + x * r.width(), 0)).x()
            if abs(vx - view_pt.x()) <= self.GUIDE_GRAB:
                return ("v", i)
        for i, y in enumerate(self.guides_h):
            vy = self.mapFromScene(QPointF(0, r.y() + y * r.height())).y()
            if abs(vy - view_pt.y()) <= self.GUIDE_GRAB:
                return ("h", i)
        return None

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and e.modifiers() & Qt.AltModifier):
            self._pan_from = e.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        if e.button() == Qt.LeftButton and self.guide_edit and not self.image_rect.isEmpty():
            hit = self._guide_at(e.position())
            sp = self.mapToScene(e.position().toPoint())
            if hit is None and self.image_rect.contains(sp):
                u, v = self._norm(sp)
                if self.guide_edit == "v":
                    self.guides_v.append(u)
                    hit = ("v", len(self.guides_v) - 1)
                else:
                    self.guides_h.append(v)
                    hit = ("h", len(self.guides_h) - 1)
            if hit:
                self._drag_guide = hit
                self.guidesChanged.emit()
                self.viewport().update()
                return
        if e.button() == Qt.LeftButton and \
                (self.overlay is None or not self.overlay.wants_press(self.mapToScene(e.position().toPoint()))):
            self._pan_from = e.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan_from is not None:
            d = e.position() - self._pan_from
            self._pan_from = e.position()
            self._shift(QPointF(-d.x() / self.factor, -d.y() / self.factor))
            return
        if self._drag_guide is not None:
            kind, i = self._drag_guide
            u, v = self._norm(self.mapToScene(e.position().toPoint()))
            if kind == "v":
                self.guides_v[i] = u
            else:
                self.guides_h[i] = v
            self.viewport().update()
            return
        if self.guide_edit:
            hit = self._guide_at(e.position())
            self.viewport().setCursor(Qt.SplitHCursor if hit and hit[0] == "v" else
                                      Qt.SplitVCursor if hit else Qt.CrossCursor)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pan_from is not None:
            self._pan_from = None
            self.viewport().unsetCursor()
            return
        if self._drag_guide is not None:
            kind, i = self._drag_guide
            vals = self.guides_v if kind == "v" else self.guides_h
            if not 0.0 <= vals[i] <= 1.0:          # dragged off the image → delete
                vals.pop(i)
            self._drag_guide = None
            self.guidesChanged.emit()
            self.viewport().update()
            return
        super().mouseReleaseEvent(e)

    # -- overlay & loupe ----------------------------------------------------------------
    def set_overlay(self, overlay):
        if self.overlay is not None:
            self.scene().removeItem(self.overlay)
            self.overlay.deleteLater()
        self.overlay = overlay
        if overlay is not None:
            self.scene().addItem(overlay)
            overlay.view = self
            overlay.set_image_rect(self.image_rect)
        self.viewport().update()

    def show_loupe(self, scene_pt: QPointF | None):
        self.loupe_pos = scene_pt
        self.viewport().update()

    # -- painting -----------------------------------------------------------------------
    def drawBackground(self, p: QPainter, rect: QRectF):
        p.fillRect(rect, self.bg)

    def drawForeground(self, p: QPainter, rect: QRectF):
        r = self.image_rect
        if r.isEmpty():
            return
        p.save()
        p.resetTransform()
        vr = self.mapFromScene(r).boundingRect()
        if self.show_grid and self.grid_size > 0:
            p.setPen(QPen(QColor(255, 255, 255, 90), 1))
            step = max(6, self.grid_size)
            x = vr.left() + step
            while x < vr.right():
                p.drawLine(x, vr.top(), x, vr.bottom())
                x += step
            y = vr.top() + step
            while y < vr.bottom():
                p.drawLine(vr.left(), y, vr.right(), y)
                y += step
        if self.guides_visible or self.guide_edit:
            p.setPen(QPen(self.guide_color, 1))
            for x in self.guides_v:
                vx = self.mapFromScene(QPointF(r.x() + x * r.width(), 0)).x()
                p.drawLine(vx, vr.top(), vx, vr.bottom())
            for y in self.guides_h:
                vy = self.mapFromScene(QPointF(0, r.y() + y * r.height())).y()
                p.drawLine(vr.left(), vy, vr.right(), vy)
        if self.loupe_pos is not None and not self.pix.pixmap().isNull():
            self._draw_loupe(p)
        p.restore()

    def _draw_loupe(self, p: QPainter):
        size, zoom = 150, 3.0
        c = self.mapFromScene(self.loupe_pos)
        center = QPointF(c.x() + 110, c.y() - 110)
        if center.x() + size / 2 > self.viewport().width():
            center.setX(c.x() - 110)
        if center.y() - size / 2 < 0:
            center.setY(c.y() + 110)
        path = QPainterPath()
        path.addEllipse(center, size / 2, size / 2)
        p.setClipPath(path)
        p.fillPath(path, self.bg)
        # source rect in pixmap coordinates
        pm = self.pix.pixmap()
        sx = pm.width() / self.image_rect.width()
        sy = pm.height() / self.image_rect.height()
        span = size / (self.factor * zoom)
        src = QRectF((self.loupe_pos.x() - span / 2) * sx, (self.loupe_pos.y() - span / 2) * sy, span * sx, span * sy)
        p.drawPixmap(QRectF(center.x() - size / 2, center.y() - size / 2, size, size), pm, src)
        p.setPen(QPen(QColor(255, 80, 80), 1))
        p.drawLine(QPointF(center.x() - 8, center.y()), QPointF(center.x() + 8, center.y()))
        p.drawLine(QPointF(center.x(), center.y() - 8), QPointF(center.x(), center.y() + 8))
        p.setClipping(False)
        p.setPen(QPen(QColor("#ed030f"), 2))
        p.drawEllipse(center, size / 2, size / 2)
