"""A fast, virtualized grid of image tiles with section headers and Google-Photos-style selection."""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractScrollArea

from . import theme
from .thumbs import ThumbLoader


@dataclass
class Tile:
    key: Any
    image: str
    fallback: str | None = None
    caption: str | None = None
    subcaption: str | None = None
    data: Any = None
    badge: str | None = None


@dataclass
class Section:
    title: str | None
    tiles: list[Tile] = field(default_factory=list)
    subtitle: str | None = None


class TileGrid(QAbstractScrollArea):
    activated = Signal(object)            # Tile
    selectionChanged = Signal()
    contextMenuRequested = Signal(object, QPoint)  # Tile, global position

    MARGIN = 24
    HEADER_H = 56
    CHECK = 24

    def __init__(self, parent=None, *, tile_size: int = 190, spacing: int = 4, circle: bool = False,
                 captions: bool = False, selectable: bool = True, zoomable: bool = True):
        super().__init__(parent)
        self.tile_target = tile_size
        self.spacing = spacing
        self.circle = circle
        self.captions = captions
        self.selectable = selectable
        self.zoomable = zoomable
        self.empty_title = ""
        self.empty_text = ""
        self.sections: list[Section] = []
        self.flat: list[Tile] = []
        self.index_of: dict[Any, int] = {}
        self.selected: set[Any] = set()
        self.anchor: int | None = None
        self.hover: int | None = None
        self._press: tuple[int, bool] | None = None
        # layout cache
        self._cols = 1
        self._tile = tile_size
        self._rows: list[tuple[int, int, int, int]] = []  # (y, section, first flat index, count)
        self._row_ys: list[int] = []
        self._headers: list[tuple[int, int]] = []          # (y, section)
        self._height = 0
        self._layout_width = -1

        self.setFrameShape(QAbstractScrollArea.NoFrame)
        self.viewport().setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.verticalScrollBar().setSingleStep(48)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ThumbLoader.instance().loaded.connect(self._on_thumb)

    # ---- data ---------------------------------------------------------------
    def set_sections(self, sections: list[Section], keep_scroll: bool = True) -> None:
        pos = self.verticalScrollBar().value() if keep_scroll else 0
        self.sections = [s for s in sections if s.tiles]
        self.flat = [t for s in self.sections for t in s.tiles]
        self.index_of = {t.key: i for i, t in enumerate(self.flat)}
        before = set(self.selected)
        self.selected &= set(self.index_of)
        self.hover = None
        self.anchor = None
        self._layout_width = -1
        self._relayout()
        self.verticalScrollBar().setValue(pos)
        self.viewport().update()
        if before != self.selected:
            self.selectionChanged.emit()

    def set_empty(self, title: str, text: str = "") -> None:
        self.empty_title, self.empty_text = title, text
        self.viewport().update()

    def selected_tiles(self) -> list[Tile]:
        return [t for t in self.flat if t.key in self.selected]

    def clear_selection(self) -> None:
        if self.selected:
            self.selected.clear()
            self.viewport().update()
            self.selectionChanged.emit()

    def select_all(self) -> None:
        if self.selectable:
            self.selected = set(self.index_of)
            self.viewport().update()
            self.selectionChanged.emit()

    # ---- layout -------------------------------------------------------------
    def _caption_h(self) -> int:
        return 48 if self.captions else 0

    def _relayout(self) -> None:
        width = self.viewport().width()
        avail = max(100, width - 2 * self.MARGIN)
        cols = max(1, (avail + self.spacing) // (self.tile_target + self.spacing))
        tile = (avail - (cols - 1) * self.spacing) // cols
        if self.circle:
            tile = min(tile, self.tile_target + 20)
        self._cols, self._tile = cols, tile
        row_h = tile + self._caption_h() + self.spacing
        y = 8
        self._rows, self._headers = [], []
        flat_i = 0
        for si, sec in enumerate(self.sections):
            if sec.title is not None:
                self._headers.append((y, si))
                y += self.HEADER_H
            for start in range(0, len(sec.tiles), cols):
                n = min(cols, len(sec.tiles) - start)
                self._rows.append((y, si, flat_i + start, n))
                y += row_h
            flat_i += len(sec.tiles)
            y += 12
        self._row_ys = [r[0] for r in self._rows]
        self._height = y + 24
        self._layout_width = width
        sb = self.verticalScrollBar()
        sb.setRange(0, max(0, self._height - self.viewport().height()))
        sb.setPageStep(self.viewport().height())

    def _tile_rect(self, row: tuple[int, int, int, int], col: int) -> QRect:
        y = row[0] - self.verticalScrollBar().value()
        if self.circle:
            # Center circles in their cell.
            cell = (self.viewport().width() - 2 * self.MARGIN - (self._cols - 1) * self.spacing) / self._cols
            x = self.MARGIN + col * (cell + self.spacing) + (cell - self._tile) / 2
            return QRect(int(x), y, self._tile, self._tile)
        return QRect(self.MARGIN + col * (self._tile + self.spacing), y, self._tile, self._tile)

    def _hit(self, pos: QPoint) -> int | None:
        y = pos.y() + self.verticalScrollBar().value()
        i = bisect.bisect_right(self._row_ys, y) - 1
        if i < 0:
            return None
        row = self._rows[i]
        for c in range(row[3]):
            r = self._tile_rect(row, c)
            if self.captions:
                r = r.adjusted(0, 0, 0, self._caption_h())
            if r.contains(pos):
                return row[2] + c
        return None

    def _check_rect(self, tile_rect: QRect) -> QRect:
        if self.circle:
            return QRect(tile_rect.right() - self.CHECK - 2, tile_rect.top() + 2, self.CHECK, self.CHECK)
        return QRect(tile_rect.left() + 8, tile_rect.top() + 8, self.CHECK, self.CHECK)

    def _rect_of(self, index: int) -> QRect | None:
        for row in self._rows:
            if row[2] <= index < row[2] + row[3]:
                return self._tile_rect(row, index - row[2])
        return None

    # ---- events -------------------------------------------------------------
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._relayout()

    def _on_thumb(self, _key):
        self.viewport().update()

    def wheelEvent(self, e):
        if self.zoomable and e.modifiers() & Qt.ControlModifier:
            step = 24 if e.angleDelta().y() > 0 else -24
            self.tile_target = max(96, min(420, self.tile_target + step))
            self._relayout()
            self.viewport().update()
            return
        super().wheelEvent(e)

    def mouseMoveEvent(self, e):
        h = self._hit(e.position().toPoint())
        if h != self.hover:
            self.hover = h
            self.viewport().update()
            self.viewport().setCursor(Qt.PointingHandCursor if h is not None else Qt.ArrowCursor)

    def leaveEvent(self, e):
        self.hover = None
        self.viewport().update()

    def mousePressEvent(self, e):
        self.setFocus()
        pos = e.position().toPoint()
        i = self._hit(pos)
        if e.button() == Qt.RightButton:
            if i is not None:
                tile = self.flat[i]
                if self.selectable and tile.key not in self.selected:
                    self.selected = {tile.key}
                    self.anchor = i
                    self.selectionChanged.emit()
                    self.viewport().update()
                self.contextMenuRequested.emit(tile, e.globalPosition().toPoint())
            return
        if e.button() != Qt.LeftButton or i is None:
            return
        handled = False
        if self.selectable:
            rect = self._rect_of(i)
            on_check = rect is not None and self._check_rect(rect).adjusted(-4, -4, 4, 4).contains(pos)
            mods = e.modifiers()
            if mods & Qt.ShiftModifier and self.anchor is not None:
                lo, hi = sorted((self.anchor, i))
                self.selected |= {self.flat[k].key for k in range(lo, hi + 1)}
                handled = True
            elif on_check or mods & Qt.ControlModifier or self.selected:
                key = self.flat[i].key
                self.selected ^= {key}
                self.anchor = i
                handled = True
            if handled:
                self.viewport().update()
                self.selectionChanged.emit()
        self._press = (i, handled)

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or self._press is None:
            return
        i, handled = self._press
        self._press = None
        if not handled and self._hit(e.position().toPoint()) == i:
            self.activated.emit(self.flat[i])

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self.selected:
            self.clear_selection()
            return
        if e.key() == Qt.Key_A and e.modifiers() & Qt.ControlModifier:
            self.select_all()
            return
        super().keyPressEvent(e)

    # ---- painting -----------------------------------------------------------
    def paintEvent(self, e):
        if self._layout_width != self.viewport().width():
            self._relayout()
        p = QPainter(self.viewport())
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        C = theme.C
        p.fillRect(self.viewport().rect(), QColor(C["bg"]))
        if not self.flat:
            self._paint_empty(p)
            return
        top = self.verticalScrollBar().value()
        bottom = top + self.viewport().height()

        title_font = QFont(self.font())
        title_font.setPixelSize(15)
        title_font.setWeight(QFont.DemiBold)
        sub_font = QFont(self.font())
        sub_font.setPixelSize(12)
        for y, si in self._headers:
            if y + self.HEADER_H < top or y > bottom:
                continue
            sec = self.sections[si]
            p.setFont(title_font)
            p.setPen(QColor(C["text"]))
            r = QRect(self.MARGIN + 2, y - top + 18, self.viewport().width() - 2 * self.MARGIN, 24)
            p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, sec.title)
            if sec.subtitle:
                w = QFontMetrics(title_font).horizontalAdvance(sec.title)
                p.setFont(sub_font)
                p.setPen(QColor(C["muted"]))
                p.drawText(r.adjusted(w + 10, 1, 0, 0), Qt.AlignLeft | Qt.AlignVCenter, sec.subtitle)

        loader = ThumbLoader.instance()
        cap_font = QFont(self.font())
        cap_font.setPixelSize(13)
        cap_font.setWeight(QFont.DemiBold)
        any_selected = bool(self.selected)
        first = max(0, bisect.bisect_right(self._row_ys, top - self._tile - self._caption_h()) - 1)
        for row in self._rows[first:]:
            if row[0] > bottom:
                break
            for c in range(row[3]):
                i = row[2] + c
                tile = self.flat[i]
                rect = self._tile_rect(row, c)
                sel = tile.key in self.selected
                hov = self.hover == i
                self._paint_tile(p, loader, tile, rect, sel, hov, any_selected)
                if self.captions:
                    cr = QRect(rect.left() - 10, rect.bottom() + 8, rect.width() + 20, 20)
                    p.setFont(cap_font)
                    muted = tile.data == "unnamed"  # e.g. "Add a name" placeholder captions
                    p.setPen(QColor(C["muted"] if muted else C["text"]))
                    fm = QFontMetrics(cap_font)
                    p.drawText(cr, Qt.AlignHCenter | Qt.AlignTop, fm.elidedText(tile.caption or "", Qt.ElideRight, cr.width()))
                    if tile.subcaption:
                        p.setFont(sub_font)
                        p.setPen(QColor(C["muted"]))
                        p.drawText(cr.translated(0, 19), Qt.AlignHCenter | Qt.AlignTop, tile.subcaption)

    def _paint_tile(self, p: QPainter, loader, tile: Tile, rect: QRect, sel: bool, hov: bool, any_sel: bool):
        C = theme.C
        inner = QRectF(rect)
        if sel:
            if not self.circle:
                p.fillRect(rect, QColor(C["select"]))
            inner = inner.adjusted(rect.width() * 0.08, rect.height() * 0.08, -rect.width() * 0.08, -rect.height() * 0.08)
        path = QPainterPath()
        if self.circle:
            path.addEllipse(inner)
        else:
            radius = 10 if sel else 4
            path.addRoundedRect(inner, radius, radius)
        p.save()
        p.setClipPath(path)
        pm = loader.get(tile.image, tile.fallback)
        if pm is None or pm.isNull():
            p.fillRect(inner, QColor(C["placeholder"]))
            if self.circle:
                p.setPen(QColor(C["muted"]))
                p.drawPixmap(inner.center().toPoint() - QPoint(16, 16), theme.icon_pixmap("person", C["muted"], 32))
        else:
            # "Cover" fit: crop the pixmap to the tile's aspect ratio.
            pw, ph = pm.width(), pm.height()
            scale = max(inner.width() / pw, inner.height() / ph)
            sw, sh = inner.width() / scale, inner.height() / scale
            src = QRectF((pw - sw) / 2, (ph - sh) / 2, sw, sh)
            p.drawPixmap(inner, pm, src)
        if hov and not sel:
            if self.circle:
                p.fillRect(inner, QColor(0, 0, 0, 30))
            else:
                grad_rect = QRectF(inner.left(), inner.top(), inner.width(), 44)
                p.fillRect(grad_rect, QColor(0, 0, 0, 60))
        p.restore()
        if tile.badge:
            self._paint_badge(p, rect, tile.badge)
        if self.circle and sel:
            p.setPen(QPen(QColor(C["accent"]), 3))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(inner.adjusted(-3, -3, 3, 3))
        if not self.selectable:
            return
        if sel or hov or any_sel:
            cr = QRectF(self._check_rect(rect))
            if sel:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(C["accent"]))
                p.drawEllipse(cr)
                p.setPen(QPen(QColor(C["on_accent"]), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                c = cr.center()
                p.drawPolyline([QPointF(c.x() - 5, c.y()), QPointF(c.x() - 1.5, c.y() + 3.5), QPointF(c.x() + 5, c.y() - 4)])
            else:
                p.setPen(QPen(QColor(255, 255, 255, 230 if hov else 150), 2))
                p.setBrush(QColor(0, 0, 0, 40))
                p.drawEllipse(cr.adjusted(1, 1, -1, -1))

    def _paint_badge(self, p: QPainter, rect: QRect, text: str):
        f = QFont(self.font())
        f.setPixelSize(11)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        fm = QFontMetrics(f)
        w = fm.horizontalAdvance(text) + 12
        r = QRectF(rect.right() - w - 6, rect.bottom() - 24, w, 18)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(r, 9, 9)
        p.setPen(QColor("white"))
        p.drawText(r, Qt.AlignCenter, text)

    def _paint_empty(self, p: QPainter):
        C = theme.C
        r = self.viewport().rect()
        f = QFont(self.font())
        f.setPixelSize(18)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor(C["text"]))
        mid = r.center().y() - 40
        p.drawText(QRect(r.left() + 40, mid - 30, r.width() - 80, 30), Qt.AlignCenter, self.empty_title)
        f.setPixelSize(13)
        f.setWeight(QFont.Normal)
        p.setFont(f)
        p.setPen(QColor(C["muted"]))
        p.drawText(QRect(r.left() + 60, mid + 6, r.width() - 120, 80), Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
                   self.empty_text)
