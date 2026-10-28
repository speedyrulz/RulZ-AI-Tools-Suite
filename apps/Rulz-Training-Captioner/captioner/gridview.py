"""Thumbnail grid with an inline editable caption box under every image."""
from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen, QTextCursor, QTextOption
from PySide6.QtWidgets import (QAbstractItemView, QListView, QPlainTextEdit, QStyle,
                               QStyledItemDelegate)

from . import theme
from .model import BUSY, ERROR, QUEUED, StatusRole

PAD = 8
NAME_H = 18
CAPTION_H = 96


class CaptionEditor(QPlainTextEdit):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTabChangesFocus(True)
        self.setFrameShape(QPlainTextEdit.Box)


class CardDelegate(QStyledItemDelegate):
    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.thumb = 180

    def card_size(self):
        return QSize(self.thumb + 2 * PAD, self.thumb + 2 * PAD + NAME_H + CAPTION_H + PAD)

    def sizeHint(self, option, index):
        return self.card_size()

    def caption_rect(self, card):
        return QRect(card.left() + PAD, card.bottom() - PAD - CAPTION_H + 1, card.width() - 2 * PAD, CAPTION_H)

    def paint(self, p, option, index):
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        pal = option.palette
        card = option.rect.adjusted(2, 2, -2, -2)
        sel = bool(option.state & QStyle.State_Selected)
        hover = bool(option.state & QStyle.State_MouseOver)
        red = pal.highlight().color()
        p.setPen(QPen(red if sel else (red.darker(160) if hover else pal.mid().color()), 2 if sel else 1))
        fill = pal.alternateBase().color()
        if sel:
            fill = QColor(red.red(), red.green(), red.blue(), 40)
            p.setBrush(pal.alternateBase())
            p.drawRoundedRect(card, 6, 6)
        p.setBrush(fill)
        p.drawRoundedRect(card, 6, 6)

        # thumbnail
        tr = QRect(card.left() + PAD, card.top() + PAD, self.thumb, self.thumb)
        pm = index.data(Qt.DecorationRole)
        if pm is not None and not pm.isNull():
            s = pm.size().scaled(tr.size(), Qt.KeepAspectRatio)
            r = QRect(0, 0, s.width(), s.height())
            r.moveCenter(tr.center())
            p.drawPixmap(r, pm)
        else:
            p.setPen(pal.mid().color())
            p.drawText(tr, Qt.AlignCenter, "…")

        # status dot
        status = index.data(StatusRole)
        has = bool(index.data(Qt.EditRole).strip())
        colour = {BUSY: "#3b82f6", QUEUED: "#a855f7", ERROR: "#ef4444"}.get(status, "#22c55e" if has else "#f59e0b")
        p.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
        p.setBrush(QColor(colour))
        p.drawEllipse(tr.left() + 4, tr.top() + 4, 12, 12)

        # filename
        nr = QRect(tr.left(), tr.bottom() + 2, self.thumb, NAME_H)
        p.setPen(pal.text().color())
        f = p.font()
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
        p.setFont(f)
        name = QFontMetrics(f).elidedText(index.data(Qt.DisplayRole), Qt.ElideMiddle, nr.width())
        p.drawText(nr, Qt.AlignVCenter | Qt.AlignLeft, name)

        # caption box
        cr = self.caption_rect(option.rect.adjusted(2, 2, -2, -2))
        p.setPen(pal.mid().color())
        p.setBrush(pal.base())
        p.drawRoundedRect(cr, 3, 3)
        text = index.data(Qt.EditRole)
        inner = cr.adjusted(4, 3, -4, -3)
        if text.strip():
            p.setPen(pal.text().color())
            opt = QTextOption()
            opt.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
            p.setClipRect(inner)
            p.drawText(inner, int(Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap), text)
        else:
            p.setPen(pal.placeholderText().color())
            p.drawText(inner, int(Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap), "Click to add caption…")
        p.restore()

    # inline editor
    def createEditor(self, parent, option, index):
        return CaptionEditor(parent)

    def setEditorData(self, editor, index):
        editor.setPlainText(index.data(Qt.EditRole))
        editor.moveCursor(QTextCursor.End)

    def setModelData(self, editor, model, index):
        try:
            model.setData(index, editor.toPlainText(), Qt.EditRole)
        except RuntimeError as e:
            self.view.error.emit(str(e))

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(self.caption_rect(option.rect.adjusted(2, 2, -2, -2)))


class ImageGrid(QListView):
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.card = CardDelegate(self)
        self.setItemDelegate(self.card)
        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setMovement(QListView.Static)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setUniformItemSizes(True)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSpacing(4)
        self.setWordWrap(True)
        self.setMouseTracking(True)
        self.empty_logo = None

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.model() is not None and self.model().rowCount() == 0:
            p = QPainter(self.viewport())
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            r = self.viewport().rect()
            if self.empty_logo is None:
                self.empty_logo = theme.logo(min(110, max(48, r.height() // 6)))
            lg = self.empty_logo
            p.setOpacity(0.18)
            p.drawPixmap(r.center().x() - lg.width() // 2, r.center().y() - lg.height() // 2 - 30, lg)
            p.setOpacity(1)
            p.setPen(self.palette().placeholderText().color())
            tr = QRect(r.left(), r.center().y() + lg.height() // 2, r.width(), 50)
            p.drawText(tr, Qt.AlignHCenter | Qt.AlignTop,
                       "Drop images or folders here\nor use Add Images… / Add Folder…")
            p.end()

    def set_thumb_size(self, n):
        self.card.thumb = n
        self.scheduleDelayedItemsLayout()
        self.viewport().update()

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        idx = self.indexAt(e.position().toPoint())
        if idx.isValid() and e.button() == Qt.LeftButton and not (e.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
            card = self.visualRect(idx).adjusted(2, 2, -2, -2)
            if self.card.caption_rect(card).contains(e.position().toPoint()):
                self.edit(idx)
