"""Interactive on-canvas tool overlays.

Every overlay is one QGraphicsObject covering the image; it draws its handles at a constant
screen size and does its own hit-testing. Model coordinates are normalised to the image rect.
Holding Shift while dragging slows the cursor down (fine adjustment).

Copied unchanged from RulZ Image Distortion Repair (package viewpointer).
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsObject

HANDLE_R = 7.0
GRAB_R = 10.0


class Overlay(QGraphicsObject):
    changed = Signal()
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.rect = QRectF(0, 0, 1, 1)
        self.view = None
        self.drag = None
        self._last = None
        self._loupe = True
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    # geometry helpers
    def set_image_rect(self, r: QRectF):
        self.prepareGeometryChange()
        self.rect = QRectF(r) if not r.isEmpty() else QRectF(0, 0, 1, 1)
        self.update()

    def boundingRect(self):
        return self.rect.adjusted(-1e4, -1e4, 1e4, 1e4)

    def px(self) -> float:
        """Scene units per screen pixel."""
        return 1.0 / self.view.factor if self.view and self.view.factor else 1.0

    def n2s(self, u, v) -> QPointF:
        return QPointF(self.rect.x() + u * self.rect.width(), self.rect.y() + v * self.rect.height())

    def s2n(self, p: QPointF):
        return (p.x() - self.rect.x()) / self.rect.width(), (p.y() - self.rect.y()) / self.rect.height()

    # hit testing
    def handles(self):
        """Iterable of (handle_id, normalised (u, v))."""
        return []

    def hit(self, p: QPointF):
        best, bd = None, (GRAB_R * self.px()) ** 2
        for hid, (u, v) in self.handles():
            q = self.n2s(u, v)
            d = (q.x() - p.x()) ** 2 + (q.y() - p.y()) ** 2
            if d <= bd:
                best, bd = hid, d
        return best

    def wants_press(self, p: QPointF) -> bool:
        return self.hit(p) is not None

    # mouse plumbing
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            e.ignore()
            return
        p = e.scenePos()
        self.drag = self.hit(p)
        if self.drag is None:
            self.drag = self.press_empty(p, e.modifiers())
        self._last = p
        self.on_press(self.drag, p, e.modifiers())
        if self._loupe and self.drag is not None and self.view and not isinstance(self.drag, tuple):
            self.view.show_loupe(p)
        e.accept()

    def mouseMoveEvent(self, e):
        if self.drag is None:
            return
        p = e.scenePos()
        if e.modifiers() & Qt.ShiftModifier and self._last is not None:
            p = self._last + (p - self._last) * 0.2
            self._virtual = p
        self._last = p
        self.on_drag(self.drag, p, e.modifiers())
        if self._loupe and self.view and not isinstance(self.drag, tuple):
            self.view.show_loupe(p)
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, e):
        if self.drag is not None:
            self.on_release(self.drag)
            self.drag = None
            if self.view:
                self.view.show_loupe(None)
            self.update()
            self.finished.emit()

    def mouseDoubleClickEvent(self, e):
        hid = self.hit(e.scenePos())
        if self.on_double_click(hid, e.scenePos()):
            self.update()
            self.changed.emit()
            self.finished.emit()

    def hoverMoveEvent(self, e):
        hid = self.hit(e.scenePos())
        self.setCursor(self.cursor_for(hid, e.scenePos()))

    def cursor_for(self, hid, p):
        return Qt.PointingHandCursor if hid is not None else Qt.ArrowCursor

    def press_empty(self, p, mods):
        return None

    def on_press(self, hid, p, mods):
        pass

    def on_drag(self, hid, p, mods):
        pass

    def on_release(self, hid):
        pass

    def on_double_click(self, hid, p) -> bool:
        return False

    # drawing helpers
    def draw_handle(self, painter: QPainter, pt: QPointF, r: float = HANDLE_R, plus: bool = True,
                    fill: QColor = QColor(245, 245, 245)):
        s = self.px()
        painter.setPen(QPen(QColor(40, 40, 40), 1.2 * s))
        painter.setBrush(fill)
        painter.drawEllipse(pt, r * s, r * s)
        if plus:
            painter.setPen(QPen(QColor(90, 90, 90), 1.2 * s))
            painter.drawLine(QPointF(pt.x() - 3.5 * s, pt.y()), QPointF(pt.x() + 3.5 * s, pt.y()))
            painter.drawLine(QPointF(pt.x(), pt.y() - 3.5 * s), QPointF(pt.x(), pt.y() + 3.5 * s))

    def extended(self, a: QPointF, b: QPointF) -> QLineF:
        """Line through a, b clipped generously beyond the image."""
        d = b - a
        n = math.hypot(d.x(), d.y()) or 1.0
        k = (self.rect.width() + self.rect.height()) * 2 / n
        return QLineF(a - d * k, b + d * k)


# ------------------------------------------------------------------------------ lines

class LinesOverlay(Overlay):
    """Perspective guide lines: verticals / horizontals (2 lines), rect (4 points), eight (4 lines)."""

    def __init__(self, mode: str, data: list | None, color_v: QColor, color_h: QColor):
        super().__init__()
        self.mode = mode
        self.color_v, self.color_h = QColor(color_v), QColor(color_h)
        self.data = [list(map(list, d)) if isinstance(d[0], (list, tuple)) else list(d) for d in data] \
            if data else self.default(mode)

    @staticmethod
    def default(mode):
        if mode == "verticals":
            return [[[0.25, 0.25], [0.25, 0.75]], [[0.75, 0.25], [0.75, 0.75]]]
        if mode == "horizontals":
            return [[[0.25, 0.25], [0.75, 0.25]], [[0.25, 0.75], [0.75, 0.75]]]
        if mode == "rect":
            return [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]
        if mode == "eight":
            return [[[0.25, 0.4], [0.25, 0.6]], [[0.75, 0.4], [0.75, 0.6]],
                    [[0.4, 0.25], [0.6, 0.25]], [[0.4, 0.75], [0.6, 0.75]]]
        return []

    def value(self):
        return [list(map(list, d)) if self.mode != "rect" else list(d) for d in self.data]

    def handles(self):
        if self.mode == "rect":
            for i, p in enumerate(self.data):
                yield (i, None), p
        else:
            for i, line in enumerate(self.data):
                for j, p in enumerate(line):
                    yield (i, j), p

    def on_drag(self, hid, p, mods):
        i, j = hid
        u, v = self.s2n(p)
        u, v = min(max(u, -0.2), 1.2), min(max(v, -0.2), 1.2)
        if j is None:
            self.data[i] = [u, v]
        else:
            self.data[i][j] = [u, v]

    def on_double_click(self, hid, p):
        if hid is None:
            return False
        self.data = self.default(self.mode)
        return True

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        s = self.px()
        if self.mode == "rect":
            pts = [self.n2s(*p) for p in self.data]
            painter.setPen(QPen(self.color_v, 1.4 * s))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(QPolygonF(pts))
            for i in range(4):
                a, b = pts[i], pts[(i + 1) % 4]
                painter.setPen(QPen(self.color_v, 1 * s, Qt.DashLine))
                painter.drawLine(self.extended(a, b))
            for q in pts:
                self.draw_handle(painter, q)
            return
        for i, line in enumerate(self.data):
            a, b = self.n2s(*line[0]), self.n2s(*line[1])
            vertical = self.mode == "verticals" or (self.mode == "eight" and i < 2)
            col = self.color_v if vertical else self.color_h
            painter.setPen(QPen(col, 1 * s, Qt.DashLine))
            painter.drawLine(self.extended(a, b))
            painter.setPen(QPen(col, 1.6 * s))
            painter.drawLine(a, b)
            self.draw_handle(painter, a)
            self.draw_handle(painter, b)


class HorizonOverlay(Overlay):
    """A single reference line the user aligns with something horizontal (or vertical)."""

    def __init__(self, kind: str, color: QColor):
        super().__init__()
        self.kind, self.color = kind, QColor(color)
        self.reset()

    def reset(self):
        self.a, self.b = ([0.3, 0.5], [0.7, 0.5]) if self.kind == "h" else ([0.5, 0.3], [0.5, 0.7])

    def handles(self):
        yield "a", self.a
        yield "b", self.b

    def press_empty(self, p, mods):
        # drawing a fresh line: start at the press position
        u, v = self.s2n(p)
        if 0 <= u <= 1 and 0 <= v <= 1:
            self.a = [u, v]
            self.b = [u, v]
            return "b"
        return None

    def wants_press(self, p):
        return True

    def on_drag(self, hid, p, mods):
        u, v = self.s2n(p)
        if hid == "a":
            self.a = [u, v]
        else:
            self.b = [u, v]

    def angle(self, frame_aspect: float) -> float:
        """Rotation (degrees, CCW positive) that makes the drawn line level/plumb."""
        dx = (self.b[0] - self.a[0]) * frame_aspect
        dy = self.b[1] - self.a[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        a = math.degrees(math.atan2(dy, dx))       # y-down: positive = clockwise visually
        if self.kind == "h":
            a = (a + 90) % 180 - 90
        else:
            a = (a % 180) - 90
        return a

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        s = self.px()
        a, b = self.n2s(*self.a), self.n2s(*self.b)
        painter.setPen(QPen(self.color, 1 * s, Qt.DashLine))
        if a != b:
            painter.drawLine(self.extended(a, b))
        painter.setPen(QPen(self.color, 1.6 * s))
        painter.drawLine(a, b)
        self.draw_handle(painter, a)
        self.draw_handle(painter, b)


# ------------------------------------------------------------------------------ crop

class CropOverlay(Overlay):
    def __init__(self, rect_n: list, frame_px: tuple[int, int], ratio_px: float | None,
                 opacity: int, show_grid: bool, constrain=None):
        super().__init__()
        self.r = list(rect_n)
        self.frame_px = frame_px
        self.ratio_px = ratio_px
        self.opacity = opacity
        self.show_grid = show_grid
        self.constrain = constrain
        self._start = None

    def set_ratio(self, ratio_px: float | None):
        self.ratio_px = ratio_px
        if ratio_px:
            x, y, w, h = self.r
            cx, cy = x + w / 2, y + h / 2
            nr = self._nratio()
            if w / h > nr:
                w = h * nr
            else:
                h = w / nr
            self.r = self._clamp([cx - w / 2, cy - h / 2, w, h])
        self.update()

    def _nratio(self):
        fw, fh = self.frame_px
        return self.ratio_px * fh / fw

    def handles(self):
        x, y, w, h = self.r
        pts = {"tl": (x, y), "t": (x + w / 2, y), "tr": (x + w, y), "r": (x + w, y + h / 2),
               "br": (x + w, y + h), "b": (x + w / 2, y + h), "bl": (x, y + h), "l": (x, y + h / 2)}
        for k, v in pts.items():
            yield k, v

    def wants_press(self, p):
        return True

    def press_empty(self, p, mods):
        u, v = self.s2n(p)
        x, y, w, h = self.r
        return ("move",) if x <= u <= x + w and y <= v <= y + h else None

    def cursor_for(self, hid, p):
        if hid in ("tl", "br"):
            return Qt.SizeFDiagCursor
        if hid in ("tr", "bl"):
            return Qt.SizeBDiagCursor
        if hid in ("t", "b"):
            return Qt.SizeVerCursor
        if hid in ("l", "r"):
            return Qt.SizeHorCursor
        u, v = self.s2n(p)
        x, y, w, h = self.r
        return Qt.SizeAllCursor if x <= u <= x + w and y <= v <= y + h else Qt.ArrowCursor

    def on_press(self, hid, p, mods):
        self._start = (list(self.r), self.s2n(p))

    def on_drag(self, hid, p, mods):
        (x, y, w, h), (u0, v0) = self._start
        u, v = self.s2n(p)
        du, dv = u - u0, v - v0
        if hid == ("move",):
            nx = min(max(x + du, 0.0), 1.0 - w)
            ny = min(max(y + dv, 0.0), 1.0 - h)
            cand = [nx, ny, w, h]
            if self.constrain and not self.constrain(cand, check_only=True):
                return
            self.r = cand
            return
        l, t, r, b = x, y, x + w, y + h
        if "l" in hid:
            l = min(l + du, r - 0.02)
        if "r" in hid:
            r = max(r + du, l + 0.02)
        if "t" in hid:
            t = min(t + dv, b - 0.02)
        if "b" in hid:
            b = max(b + dv, t + 0.02)
        l, t, r, b = max(l, 0.0), max(t, 0.0), min(r, 1.0), min(b, 1.0)
        if self.ratio_px:
            nr = self._nratio()
            nw, nh = r - l, b - t
            if hid in ("t", "b"):
                nw = nh * nr
                cx = (x + x + w) / 2
                l, r = cx - nw / 2, cx + nw / 2
            elif hid in ("l", "r"):
                nh = nw / nr
                cy = (y + y + h) / 2
                t, b = cy - nh / 2, cy + nh / 2
            else:
                if nw / nh > nr:
                    nw = nh * nr
                else:
                    nh = nw / nr
                if "l" in hid:
                    l = r - nw
                else:
                    r = l + nw
                if "t" in hid:
                    t = b - nh
                else:
                    b = t + nh
            if l < 0 or t < 0 or r > 1 or b > 1:
                return
        cand = [l, t, r - l, b - t]
        if self.constrain:
            cand = self.constrain(cand)
        self.r = cand

    def _clamp(self, r):
        x, y, w, h = r
        w, h = min(w, 1.0), min(h, 1.0)
        return [min(max(x, 0.0), 1.0 - w), min(max(y, 0.0), 1.0 - h), w, h]

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        s = self.px()
        x, y, w, h = self.r
        cr = QRectF(self.n2s(x, y), self.n2s(x + w, y + h))
        outer = QPainterPath()
        outer.addRect(self.rect)
        inner = QPainterPath()
        inner.addRect(cr)
        painter.fillPath(outer.subtracted(inner), QColor(0, 0, 0, int(self.opacity / 100 * 255)))
        painter.setPen(QPen(QColor(255, 255, 255), 1 * s, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(cr)
        if self.show_grid:
            painter.setPen(QPen(QColor(255, 255, 255, 170), 0.8 * s, Qt.DotLine))
            for k in (1, 2):
                painter.drawLine(QPointF(cr.left() + cr.width() * k / 3, cr.top()),
                                 QPointF(cr.left() + cr.width() * k / 3, cr.bottom()))
                painter.drawLine(QPointF(cr.left(), cr.top() + cr.height() * k / 3),
                                 QPointF(cr.right(), cr.top() + cr.height() * k / 3))
        painter.setPen(QPen(QColor(30, 30, 30), 1 * s))
        painter.setBrush(QColor(240, 240, 240))
        hs = 4 * s
        for _, (u, v) in self.handles():
            q = self.n2s(u, v)
            painter.drawRect(QRectF(q.x() - hs, q.y() - hs, 2 * hs, 2 * hs))
        fw, fh = self.frame_px
        painter.setFont(QFont("Segoe UI", 1))
        f = painter.font()
        f.setPointSizeF(max(1.0, 9 * s))
        painter.setFont(f)
        painter.setPen(QColor(230, 230, 230))
        painter.drawText(QPointF(cr.left() + 6 * s, cr.bottom() - 6 * s), f"{round(w * fw)} x {round(h * fh)} px")


# ------------------------------------------------------------------------------ warp

class WarpOverlay(Overlay):
    """Reshape grid editor. Points are drawn at their displaced positions."""

    def __init__(self, rows: int, cols: int, disp: list | None, color: QColor):
        super().__init__()
        self.color = QColor(color)
        self.mode = "free"          # free | persp | anamorph
        self.sub = "move"           # move | rotate | scale
        self.sticky = False
        self.propagation = 0
        self.selected: set[tuple[int, int]] = set()
        self.band = None
        self.set_grid(rows, cols, disp)
        self._base_anamorph = None
        self.anamorph = {"tab": "hv", "h": 100, "v": 100, "d": 100, "intensity": 100}
        self.persp_kind = "eight"
        self.persp_intensity = 100

    def set_grid(self, rows, cols, disp=None):
        self.rows, self.cols = rows, cols
        gu, gv = np.meshgrid(np.linspace(0, 1, cols + 1), np.linspace(0, 1, rows + 1))
        self.base = np.dstack([gu, gv])
        d = np.asarray(disp, dtype=np.float64) if disp else None
        self.disp = d if d is not None and d.shape == self.base.shape else np.zeros_like(self.base)
        self.selected.clear()
        self.update()

    def pos(self):
        return self.base + self.disp

    def value(self):
        return self.disp.tolist() if np.any(self.disp) else []

    # selection region (grid index box) used by perspective/anamorphosis modes
    def region(self):
        if len(self.selected) >= 2:
            ii = [i for i, _ in self.selected]
            jj = [j for _, j in self.selected]
            i0, i1, j0, j1 = min(ii), max(ii), min(jj), max(jj)
            if i1 > i0 and j1 > j0:
                return i0, i1, j0, j1
        return 0, self.rows, 0, self.cols

    def handles(self):
        p = self.pos()
        if self.mode == "persp":
            i0, i1, j0, j1 = self.region()
            for (i, j) in ((i0, j0), (i0, j1), (i1, j1), (i1, j0)):
                yield (i, j), p[i, j]
            return
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                yield (i, j), p[i, j]

    def wants_press(self, p):
        return True

    def press_empty(self, p, mods):
        u, v = self.s2n(p)
        self.band = [u, v, u, v]
        if not mods & (Qt.ShiftModifier | Qt.ControlModifier):
            self.selected.clear()
        return ("band",)

    def on_press(self, hid, p, mods):
        self._start_disp = self.disp.copy()
        self._start_pt = self.s2n(p)
        if hid and hid != ("band",):
            if mods & (Qt.ShiftModifier | Qt.ControlModifier):
                self.selected ^= {hid}
            elif hid not in self.selected:
                self.selected = {hid}
        self.update()

    def _border_mask(self):
        m = np.zeros(self.base.shape[:2] + (2,), dtype=bool)
        m[:, 0, 0] = m[:, -1, 0] = True       # left/right columns: u locked
        m[0, :, 1] = m[-1, :, 1] = True       # top/bottom rows: v locked
        return m

    def _propagate(self, delta_sel: dict):
        """delta_sel: {(i, j): (du, dv)} for selected points; spread to neighbours."""
        d = self._start_disp.copy()
        p0 = self.base + self._start_disp
        if self.propagation > 0 and delta_sel:
            sigma = self.propagation / 200.0 * 0.25
            keys = list(delta_sel)
            sel_pos = np.array([p0[k] for k in keys])
            sel_d = np.array([delta_sel[k] for k in keys])
            flat = p0.reshape(-1, 2)
            dist2 = ((flat[:, None, :] - sel_pos[None]) ** 2).sum(-1)
            w = np.exp(-dist2 / (2 * sigma * sigma))
            wsum = w.sum(1, keepdims=True)
            move = (w @ sel_d) / np.maximum(wsum, 1e-9) * np.minimum(1.0, wsum)
            d += move.reshape(d.shape)
        for k, dd in delta_sel.items():
            d[k] = self._start_disp[k] + dd
        if self.sticky:
            m = self._border_mask()
            d[m] = 0.0
        self.disp = d

    def on_drag(self, hid, p, mods):
        u, v = self.s2n(p)
        if hid == ("band",):
            self.band[2], self.band[3] = u, v
            self._select_band(mods)
            self._loupe = False
            return
        self._loupe = True
        if self.mode == "persp":
            self._drag_persp_corner(hid, u, v)
            return
        su, sv = self._start_pt
        p0 = self.base + self._start_disp
        sel = self.selected or {hid}
        if self.sub == "move" or len(sel) == 1:
            delta = {k: (u - su, v - sv) for k in sel}
        else:
            c = np.mean([p0[k] for k in sel], axis=0)
            if self.sub == "rotate":
                a0 = math.atan2(sv - c[1], su - c[0])
                a1 = math.atan2(v - c[1], u - c[0])
                ca, sa = math.cos(a1 - a0), math.sin(a1 - a0)
                delta = {}
                for k in sel:
                    q = p0[k] - c
                    delta[k] = (c[0] + q[0] * ca - q[1] * sa - p0[k][0], c[1] + q[0] * sa + q[1] * ca - p0[k][1])
            else:
                r0 = math.hypot(su - c[0], sv - c[1]) or 1e-6
                k_s = math.hypot(u - c[0], v - c[1]) / r0
                delta = {k: tuple((p0[k] - c) * (k_s - 1)) for k in sel}
        self._propagate(delta)

    def _select_band(self, mods):
        u0, v0, u1, v1 = self.band
        lo_u, hi_u, lo_v, hi_v = min(u0, u1), max(u0, u1), min(v0, v1), max(v0, v1)
        p = self.pos()
        inside = {(i, j) for i in range(self.rows + 1) for j in range(self.cols + 1)
                  if lo_u <= p[i, j, 0] <= hi_u and lo_v <= p[i, j, 1] <= hi_v}
        self.selected = inside

    def on_release(self, hid):
        self.band = None
        self._loupe = True

    def on_double_click(self, hid, p):
        if hid is None or isinstance(hid[0], str):
            return False
        self.disp[hid] = 0.0
        return True

    # perspective mode -----------------------------------------------------------------
    def _corner_targets(self):
        i0, i1, j0, j1 = self.region()
        p = self.base + self._start_disp
        return i0, i1, j0, j1, [p[i0, j0].copy(), p[i0, j1].copy(), p[i1, j1].copy(), p[i1, j0].copy()]

    def _apply_region_quad(self, quad, start_disp=None):
        from ..core.geometry import homography_from_points, apply_h
        start = self._start_disp if start_disp is None else start_disp
        i0, i1, j0, j1 = self.region()
        p = self.base + start
        src = [p[i0, j0], p[i0, j1], p[i1, j1], p[i1, j0]]
        k = self.persp_intensity / 100.0
        dst = [s + k * (np.asarray(q) - s) for s, q in zip(src, quad)]
        h = homography_from_points(src, dst)
        sub = p[i0:i1 + 1, j0:j1 + 1]
        x, y = apply_h(h, sub[..., 0], sub[..., 1])
        d = start.copy()
        d[i0:i1 + 1, j0:j1 + 1, 0] = x - self.base[i0:i1 + 1, j0:j1 + 1, 0]
        d[i0:i1 + 1, j0:j1 + 1, 1] = y - self.base[i0:i1 + 1, j0:j1 + 1, 1]
        if self.sticky:
            d[self._border_mask()] = 0.0
        self.disp = d

    def _drag_persp_corner(self, hid, u, v):
        i0, i1, j0, j1, quad = self._corner_targets()
        idx = {(i0, j0): 0, (i0, j1): 1, (i1, j1): 2, (i1, j0): 3}.get(hid)
        if idx is None:
            return
        quad[idx] = np.array([u, v])
        self._apply_region_quad(quad)

    def persp_action(self, kind: str):
        """Horizon: level the region's top/bottom edges; Force parallel: plumb its sides."""
        self._start_disp = self.disp.copy()
        i0, i1, j0, j1, q = self._corner_targets()
        if kind == "horizon":
            yt = (q[0][1] + q[1][1]) / 2
            yb = (q[2][1] + q[3][1]) / 2
            q[0][1] = q[1][1] = yt
            q[2][1] = q[3][1] = yb
        elif kind == "parallel":
            xl = (q[0][0] + q[3][0]) / 2
            xr = (q[1][0] + q[2][0]) / 2
            q[0][0] = q[3][0] = xl
            q[1][0] = q[2][0] = xr
        self._apply_region_quad(q)
        self.update()
        self.changed.emit()
        self.finished.emit()

    # anamorphosis mode ----------------------------------------------------------------
    def begin_anamorph(self):
        self._base_anamorph = self.disp.copy()

    def apply_anamorph(self):
        if self._base_anamorph is None:
            self.begin_anamorph()
        d = self._base_anamorph.copy()
        i0, i1, j0, j1 = self.region()
        p = self.base + self._base_anamorph
        sub = p[i0:i1 + 1, j0:j1 + 1]
        lo, hi = sub.reshape(-1, 2).min(0), sub.reshape(-1, 2).max(0)
        c, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-6)
        x = (sub[..., 0] - c[0]) / half[0]
        y = (sub[..., 1] - c[1]) / half[1]
        k = self.anamorph["intensity"] / 100.0
        if self.anamorph["tab"] == "hv":
            kh = (self.anamorph["h"] - 100) / 100.0 * k
            kv = (self.anamorph["v"] - 100) / 100.0 * k
            nx = x + 0.35 * kh * x * (1 - np.abs(x))
            ny = y + 0.35 * kv * y * (1 - np.abs(y))
        else:
            kd = (self.anamorph["d"] - 100) / 100.0 * k
            r = np.sqrt(x * x + y * y)
            rr = np.clip(r / math.sqrt(2), 0, 1)
            f = 1 + 0.35 * kd * (1 - rr)
            nx, ny = x * f, y * f
        newx = c[0] + nx * half[0]
        newy = c[1] + ny * half[1]
        d[i0:i1 + 1, j0:j1 + 1, 0] = newx - self.base[i0:i1 + 1, j0:j1 + 1, 0]
        d[i0:i1 + 1, j0:j1 + 1, 1] = newy - self.base[i0:i1 + 1, j0:j1 + 1, 1]
        if self.sticky:
            d[self._border_mask()] = 0.0
        self.disp = d
        self.update()
        self.changed.emit()

    # painting -------------------------------------------------------------------------
    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        s = self.px()
        p = self.pos()
        painter.setPen(QPen(self.color, 1.1 * s))
        for i in range(self.rows + 1):
            painter.drawPolyline(QPolygonF([self.n2s(*p[i, j]) for j in range(self.cols + 1)]))
        for j in range(self.cols + 1):
            painter.drawPolyline(QPolygonF([self.n2s(*p[i, j]) for i in range(self.rows + 1)]))
        if self.mode in ("persp", "anamorph"):
            i0, i1, j0, j1 = self.region()
            if (i0, i1, j0, j1) != (0, self.rows, 0, self.cols) or self.mode == "persp":
                quad = [p[i0, j0], p[i0, j1], p[i1, j1], p[i1, j0]]
                painter.setPen(QPen(QColor(255, 255, 255), 2 * s))
                painter.drawPolygon(QPolygonF([self.n2s(*q) for q in quad]))
        r = 3.2 * s
        for hid, (u, v) in self.handles():
            q = self.n2s(u, v)
            sel = hid in self.selected
            painter.setPen(QPen(QColor(20, 20, 20), 1 * s))
            painter.setBrush(QColor(255, 255, 255) if sel else self.color)
            rr = r * (1.8 if self.mode == "persp" else (1.35 if sel else 1.0))
            painter.drawEllipse(q, rr, rr)
        if self.band:
            painter.setPen(QPen(QColor(255, 255, 255), 1 * s, Qt.DashLine))
            painter.setBrush(QColor(255, 255, 255, 30))
            painter.drawRect(QRectF(self.n2s(self.band[0], self.band[1]), self.n2s(self.band[2], self.band[3])).normalized())


# ------------------------------------------------------------------------------ tilt-shift

class TiltShiftOverlay(Overlay):
    def __init__(self, params, frame_aspect: float, color: QColor):
        super().__init__()
        self.p = params                  # TiltShiftParams (mutated in place)
        self.aspect = frame_aspect       # width / height of the displayed image
        self.color = QColor(color)
        self._loupe = False

    def _dirs(self):
        a = math.radians(self.p.angle)
        # along-line direction and downward normal, in height-normalised units
        return (math.cos(a), -math.sin(a)), (math.sin(a), math.cos(a))

    def _pt(self, along: float, normal: float):
        (tx, ty), (nx, ny) = self._dirs()
        cx, cy = self.p.center[0] * self.aspect, self.p.center[1]
        x, y = cx + tx * along + nx * normal, cy + ty * along + ny * normal
        return x / self.aspect, y

    def handles(self):
        rot = 0.12
        yield "center", tuple(self.p.center)
        yield "sharp_top", self._pt(0, -self.p.sharp_top)
        yield "sharp_bottom", self._pt(0, self.p.sharp_bottom)
        yield "trans_top", self._pt(0, -self.p.trans_top)
        yield "trans_bottom", self._pt(0, self.p.trans_bottom)
        yield "rot_top", self._pt(rot, -self.p.sharp_top)
        yield "rot_bottom", self._pt(rot, self.p.sharp_bottom)

    def cursor_for(self, hid, p):
        if hid == "center":
            return Qt.SizeAllCursor
        if hid and hid.startswith("rot"):
            return Qt.CrossCursor
        return Qt.SizeVerCursor if hid else Qt.ArrowCursor

    def on_drag(self, hid, p, mods):
        u, v = self.s2n(p)
        if hid == "center":
            self.p.center = [min(max(u, 0), 1), min(max(v, 0), 1)]
            return
        x, y = u * self.aspect, v
        cx, cy = self.p.center[0] * self.aspect, self.p.center[1]
        (tx, ty), (nx, ny) = self._dirs()
        dist = (x - cx) * nx + (y - cy) * ny
        if hid.startswith("rot"):
            ang = math.degrees(math.atan2(-(y - cy), x - cx))
            self.p.angle = round(((self._rot0[0] + ang - self._rot0[1] + 90) % 180) - 90, 2)
            return
        d = abs(dist)
        top = hid.endswith("top")
        if hid.startswith("sharp"):
            gap_t = self.p.trans_top - self.p.sharp_top
            gap_b = self.p.trans_bottom - self.p.sharp_bottom
            if top or self.p.sym_position:
                self.p.sharp_top = max(0.0, d)
            if not top or self.p.sym_position:
                self.p.sharp_bottom = max(0.0, d)
            # the transition zone keeps its width while the sharp band moves
            self.p.trans_top = self.p.sharp_top + gap_t
            self.p.trans_bottom = self.p.sharp_bottom + gap_b
        else:
            gap = max(0.005, d - (self.p.sharp_top if top else self.p.sharp_bottom))
            if top or self.p.sym_blur:
                self.p.trans_top = self.p.sharp_top + gap
            if not top or self.p.sym_blur:
                self.p.trans_bottom = self.p.sharp_bottom + gap

    def on_press(self, hid, p, mods):
        if hid and hid.startswith("rot"):
            u, v = self.s2n(p)
            cx, cy = self.p.center[0] * self.aspect, self.p.center[1]
            self._rot0 = (self.p.angle, math.degrees(math.atan2(-(v - cy), u * self.aspect - cx)))

    def on_double_click(self, hid, p):
        if hid is None:
            return False
        from ..core.params import TiltShiftParams
        d = TiltShiftParams()
        self.p.center, self.p.angle = d.center, d.angle
        self.p.sharp_top = self.p.sharp_bottom = d.sharp_top
        self.p.trans_top = self.p.trans_bottom = d.trans_top
        return True

    def _line(self, normal):
        a = self.n2s(*self._pt(-50, normal))
        b = self.n2s(*self._pt(50, normal))
        return QLineF(a, b)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        s = self.px()
        painter.setPen(QPen(self.color, 1.2 * s))
        painter.drawLine(self._line(-self.p.sharp_top))
        painter.drawLine(self._line(self.p.sharp_bottom))
        painter.setPen(QPen(self.color, 1 * s, Qt.DashLine))
        painter.drawLine(self._line(-self.p.trans_top))
        painter.drawLine(self._line(self.p.trans_bottom))
        hs = dict(self.handles())
        self.draw_handle(painter, self.n2s(*hs["center"]), 9, plus=False)
        for k in ("sharp_top", "sharp_bottom"):
            self.draw_handle(painter, self.n2s(*hs[k]), 6, plus=False)
        for k in ("trans_top", "trans_bottom"):
            self.draw_handle(painter, self.n2s(*hs[k]), 4.5, plus=False, fill=QColor(200, 200, 200))
        for k in ("rot_top", "rot_bottom"):
            self.draw_handle(painter, self.n2s(*hs[k]), 4.5, plus=False, fill=QColor(255, 80, 88))
