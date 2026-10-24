"""Video preview widget with an interactive crop rectangle.

The crop is kept in *frame* pixel coordinates of the displayed (rotated) frame,
so it maps 1:1 onto ffmpeg's crop filter after the same rotation is applied.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QPixmap, QImage, QBrush
from PySide6.QtWidgets import QWidget, QSizePolicy

from . import theme

MIN_SIZE = 16.0      # smallest crop edge, in frame px
HANDLE = 8           # handle half-size in widget px
DRAG_THRESHOLD = 3


def even(v: float) -> int:
    return max(int(v) & ~1, 2)


class VideoView(QWidget):
    """Shows a frame letterboxed in the widget, with an optional crop overlay."""

    cropChanged = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._pix: QPixmap | None = None
        self._frame_w = 0
        self._frame_h = 0
        self._disp = QRectF()          # where the frame is painted, widget coords
        self.crop_enabled = False
        self.crop: QRectF | None = None  # frame px
        self.aspect: float | None = None  # width / height, None = free
        self._drag = None               # (mode, press_frame_pt, orig_rect)
        self._hover_mode = None

    # ------------------------------------------------------------ frames
    def set_frame(self, img: QImage, frame_w: int, frame_h: int):
        """img is a (possibly downscaled) preview; frame_w/h is the full logical size."""
        self._pix = QPixmap.fromImage(img)
        if (frame_w, frame_h) != (self._frame_w, self._frame_h):
            self._frame_w, self._frame_h = frame_w, frame_h
            if self.crop is not None:
                self.crop = self._fit_free(self.crop)
        self._layout()
        self.update()

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_w, self._frame_h

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout()

    def _layout(self):
        if not self._pix or self._frame_w == 0:
            self._disp = QRectF()
            return
        W, H = self.width(), self.height()
        s = min(W / self._frame_w, H / self._frame_h)
        w, h = self._frame_w * s, self._frame_h * s
        self._disp = QRectF((W - w) / 2, (H - h) / 2, w, h)

    # ------------------------------------------------------- coordinates
    def _scale(self) -> float:
        return self._disp.width() / self._frame_w if self._frame_w else 1.0

    def _to_frame(self, p: QPointF) -> QPointF:
        s = self._scale()
        return QPointF((p.x() - self._disp.x()) / s, (p.y() - self._disp.y()) / s)

    def _to_widget(self, r: QRectF) -> QRectF:
        s = self._scale()
        return QRectF(self._disp.x() + r.x() * s, self._disp.y() + r.y() * s,
                      r.width() * s, r.height() * s)

    # --------------------------------------------------------- crop API
    def set_crop_enabled(self, on: bool):
        self.crop_enabled = on
        if on and self.crop is None and self._frame_w:
            self.crop = self._default_crop()
        self.setCursor(Qt.ArrowCursor)
        self.update()
        self.cropChanged.emit()

    def reset_crop(self):
        self.crop = self._default_crop() if self._frame_w else None
        self.update()
        self.cropChanged.emit()

    def set_aspect(self, aspect: float | None):
        """Lock the crop to width/height = aspect (None = free) and refit the current box."""
        self.aspect = aspect
        if self.crop is not None and aspect:
            self.crop = self._refit_aspect(self.crop, aspect)
        self.update()
        self.cropChanged.emit()

    def crop_rect_int(self) -> tuple[int, int, int, int] | None:
        """(x, y, w, h) as even integers inside the frame, or None if crop is off."""
        if not self.crop_enabled or self.crop is None or not self._frame_w:
            return None
        r = self.crop
        w, h = even(r.width()), even(r.height())
        x = int(round(r.x())) & ~1
        y = int(round(r.y())) & ~1
        x = max(0, min(x, self._frame_w - w))
        y = max(0, min(y, self._frame_h - h))
        return x, y, w, h

    def flip_crop_h(self):
        """Mirror the crop box so it stays over the same content after a horizontal flip."""
        if self.crop is not None and self._frame_w:
            r = self.crop
            self.crop = QRectF(self._frame_w - (r.x() + r.width()), r.y(), r.width(), r.height())

    def clear(self):
        self._pix = None
        self._frame_w = self._frame_h = 0
        self._disp = QRectF()
        self.crop = None
        self.crop_enabled = False
        self._drag = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def rotate_crop_cw(self, old_frame_h: int):
        """Carry the crop along when the frame is rotated 90 deg clockwise."""
        if self.crop is None:
            return
        r = self.crop
        self.crop = QRectF(old_frame_h - (r.y() + r.height()), r.x(), r.height(), r.width())
        if self.aspect:
            self.aspect = 1.0 / self.aspect

    # ---------------------------------------------------------- geometry
    def _default_crop(self) -> QRectF:
        W, H = self._frame_w, self._frame_h
        if not self.aspect:
            return QRectF(0, 0, W, H)
        w, h = W, W / self.aspect
        if h > H:
            h, w = H, H * self.aspect
        return QRectF((W - w) / 2, (H - h) / 2, w, h)

    def _refit_aspect(self, r: QRectF, aspect: float) -> QRectF:
        W, H = self._frame_w, self._frame_h
        cx, cy = r.center().x(), r.center().y()
        w = r.width()
        h = w / aspect
        if h > H:
            h, w = H, H * aspect
        if w > W:
            w, h = W, W / aspect
        x = min(max(cx - w / 2, 0), W - w)
        y = min(max(cy - h / 2, 0), H - h)
        return QRectF(x, y, w, h)

    def _fit_free(self, r: QRectF) -> QRectF:
        W, H = self._frame_w, self._frame_h
        r = r.intersected(QRectF(0, 0, W, H))
        if r.width() < MIN_SIZE or r.height() < MIN_SIZE:
            return self._default_crop()
        return r

    def _hit(self, p: QPointF) -> str | None:
        if not self.crop_enabled or not self._frame_w:
            return None
        if self.crop is None:
            return "new"
        wr = self._to_widget(self.crop)
        x, y = p.x(), p.y()
        near = lambda a, b: abs(a - b) <= HANDLE
        on_l, on_r = near(x, wr.left()), near(x, wr.right())
        on_t, on_b = near(y, wr.top()), near(y, wr.bottom())
        in_x = wr.left() - HANDLE <= x <= wr.right() + HANDLE
        in_y = wr.top() - HANDLE <= y <= wr.bottom() + HANDLE
        if in_x and in_y:
            v = "n" if on_t else "s" if on_b else ""
            hz = "w" if on_l else "e" if on_r else ""
            if v or hz:
                return v + hz
            return "move"
        return "new"

    _CURSORS = {
        "move": Qt.SizeAllCursor, "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor,
        "e": Qt.SizeHorCursor, "w": Qt.SizeHorCursor, "ne": Qt.SizeBDiagCursor,
        "sw": Qt.SizeBDiagCursor, "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
        "new": Qt.CrossCursor,
    }

    def _apply_drag(self, cur: QPointF):
        mode, p0, orig = self._drag
        W, H = self._frame_w, self._frame_h
        dx, dy = cur.x() - p0.x(), cur.y() - p0.y()
        a = self.aspect

        if mode == "move":
            x = min(max(orig.x() + dx, 0), W - orig.width())
            y = min(max(orig.y() + dy, 0), H - orig.height())
            self.crop = QRectF(x, y, orig.width(), orig.height())
            return

        if mode == "new":
            # anchor at press point; grow toward the cursor
            ax, ay = min(max(p0.x(), 0), W), min(max(p0.y(), 0), H)
            sx = 1 if dx >= 0 else -1
            sy = 1 if dy >= 0 else -1
            w, h = abs(dx), abs(dy)
            self.crop = self._grow_from_anchor(ax, ay, sx, sy, w, h, a, corner=True)
            return

        # resize modes: anchor is the opposite edge / corner
        left, top, right, bottom = orig.left(), orig.top(), orig.right(), orig.bottom()
        if mode in ("e", "w", "n", "s"):
            if mode in ("e", "w"):
                sx = 1 if mode == "e" else -1
                ax = left if mode == "e" else right
                w = max((right - left) + sx * dx, MIN_SIZE)
                if a:
                    cy = (top + bottom) / 2
                    h = w / a
                    h = min(h, 2 * cy, 2 * (H - cy))
                    w = h * a
                    w = min(w, W - ax if sx > 0 else ax)
                    h = w / a
                    self.crop = QRectF(ax if sx > 0 else ax - w, cy - h / 2, w, h)
                else:
                    w = min(w, W - ax if sx > 0 else ax)
                    self.crop = QRectF(ax if sx > 0 else ax - w, top, w, bottom - top)
            else:
                sy = 1 if mode == "s" else -1
                ay = top if mode == "s" else bottom
                h = max((bottom - top) + sy * dy, MIN_SIZE)
                if a:
                    cx = (left + right) / 2
                    w = h * a
                    w = min(w, 2 * cx, 2 * (W - cx))
                    h = w / a
                    h = min(h, H - ay if sy > 0 else ay)
                    w = h * a
                    self.crop = QRectF(cx - w / 2, ay if sy > 0 else ay - h, w, h)
                else:
                    h = min(h, H - ay if sy > 0 else ay)
                    self.crop = QRectF(left, ay if sy > 0 else ay - h, right - left, h)
            return

        # corner
        sx = 1 if "e" in mode else -1
        sy = 1 if "s" in mode else -1
        ax = left if sx > 0 else right
        ay = top if sy > 0 else bottom
        w = max((right - left) + sx * dx, MIN_SIZE)
        h = max((bottom - top) + sy * dy, MIN_SIZE)
        self.crop = self._grow_from_anchor(ax, ay, sx, sy, w, h, a, corner=True)

    def _grow_from_anchor(self, ax, ay, sx, sy, w, h, a, corner: bool) -> QRectF:
        W, H = self._frame_w, self._frame_h
        max_w = W - ax if sx > 0 else ax
        max_h = H - ay if sy > 0 else ay
        if a:
            # let the dominant drag axis lead, then derive the other
            if w >= h * a:
                h = w / a
            else:
                w = h * a
            w = min(w, max_w, max_h * a)
            h = w / a
            if w < MIN_SIZE or h < MIN_SIZE:
                w = max(MIN_SIZE, MIN_SIZE * a)
                h = w / a
        else:
            w = min(max(w, MIN_SIZE), max_w)
            h = min(max(h, MIN_SIZE), max_h)
        x = ax if sx > 0 else ax - w
        y = ay if sy > 0 else ay - h
        return QRectF(x, y, w, h)

    # ------------------------------------------------------------- events
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        mode = self._hit(e.position())
        if mode is None:
            return
        self._drag = (mode, self._to_frame(e.position()),
                      QRectF(self.crop) if self.crop is not None else QRectF())
        self._press_pos = e.position()

    def mouseMoveEvent(self, e):
        if self._drag:
            if (e.position() - self._press_pos).manhattanLength() < DRAG_THRESHOLD and self._drag[0] == "new":
                return
            self._apply_drag(self._to_frame(e.position()))
            self.update()
            self.cropChanged.emit()
            return
        mode = self._hit(e.position())
        if mode != self._hover_mode:
            self._hover_mode = mode
            self.setCursor(self._CURSORS.get(mode, Qt.ArrowCursor))

    def mouseReleaseEvent(self, e):
        if self._drag and e.button() == Qt.LeftButton:
            mode = self._drag[0]
            self._drag = None
            if mode == "new" and self.crop is not None and (
                    self.crop.width() < MIN_SIZE or self.crop.height() < MIN_SIZE):
                self.crop = self._default_crop()
            self.update()
            self.cropChanged.emit()

    # ------------------------------------------------------------- paint
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), theme.Q_BG)
        if not self._pix:
            # empty state: the RulZ logo with a hint underneath
            logo = theme.logo_pixmap(max(40, min(120, self.height() // 5)))
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            total_h = logo.height() + 40
            y = (self.height() - total_h) // 2
            if not logo.isNull():
                p.drawPixmap((self.width() - logo.width()) // 2, y, logo)
            p.setPen(theme.Q_MUTED)
            p.drawText(QRectF(0, y + logo.height() + 16, self.width(), 24), Qt.AlignCenter,
                       "Open a video (Ctrl+O) or drop a file here")
            p.end()
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawPixmap(self._disp.toRect(), self._pix)

        if self.crop_enabled and self.crop is not None:
            wr = self._to_widget(self.crop)
            # darken outside the crop
            p.setClipRect(self._disp.toRect())
            shade = QColor(0, 0, 0, 130)
            d = self._disp
            p.fillRect(QRectF(d.left(), d.top(), d.width(), wr.top() - d.top()), shade)
            p.fillRect(QRectF(d.left(), wr.bottom(), d.width(), d.bottom() - wr.bottom()), shade)
            p.fillRect(QRectF(d.left(), wr.top(), wr.left() - d.left(), wr.height()), shade)
            p.fillRect(QRectF(wr.right(), wr.top(), d.right() - wr.right(), wr.height()), shade)
            p.setClipping(False)
            # border + thirds
            p.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
            p.drawRect(wr)
            p.setPen(QPen(QColor(255, 255, 255, 70), 1))
            for i in (1, 2):
                x = wr.left() + wr.width() * i / 3
                y = wr.top() + wr.height() * i / 3
                p.drawLine(QPointF(x, wr.top()), QPointF(x, wr.bottom()))
                p.drawLine(QPointF(wr.left(), y), QPointF(wr.right(), y))
            # handles
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.setBrush(QBrush(theme.Q_RED))
            cx, cy = wr.center().x(), wr.center().y()
            for hx, hy in ((wr.left(), wr.top()), (cx, wr.top()), (wr.right(), wr.top()),
                           (wr.left(), cy), (wr.right(), cy),
                           (wr.left(), wr.bottom()), (cx, wr.bottom()), (wr.right(), wr.bottom())):
                p.drawRect(QRectF(hx - 4, hy - 4, 8, 8))
            # size label
            r = self.crop_rect_int()
            if r:
                txt = f"{r[2]}×{r[3]}  @ {r[0]},{r[1]}"
                p.setPen(QColor(255, 255, 255))
                p.setBrush(QBrush(QColor(theme.RED_DARK)))
                fm = p.fontMetrics()
                tw = fm.horizontalAdvance(txt) + 10
                ty = wr.top() - 22 if wr.top() > 24 else wr.top() + 4
                p.drawRect(QRectF(wr.left(), ty, tw, 18))
                p.drawText(QRectF(wr.left() + 5, ty, tw, 18), Qt.AlignVCenter, txt)
        p.end()
