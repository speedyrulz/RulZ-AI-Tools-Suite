"""Full-window photo viewer with face boxes, face corrections and manual face tagging."""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMenu, QScrollArea, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)

from .. import config, imaging
from ..db import Face
from . import dialogs, theme
from .common import circle_pixmap, face_image, show_in_explorer, open_default


def _fmt_size(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


class ImageCanvas(QWidget):
    faceClicked = Signal(object, QPoint)   # Face, global pos
    boxDrawn = Signal(QRectF)              # normalized box
    navigate = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image = QImage()
        self.faces: list[Face] = []
        self.show_faces = True
        self.add_mode = False
        self.highlight: int | None = None
        self.hover: int | None = None
        self.message = ""
        self._drag_start: QPointF | None = None
        self._drag_now: QPointF | None = None
        self.prev_btn = self._nav_button("prev", -1)
        self.next_btn = self._nav_button("next", 1)

    def _nav_button(self, glyph: str, step: int) -> QToolButton:
        b = QToolButton(self)
        b.setIcon(theme.icon(glyph, "#ffffff", 22))
        b.setFixedSize(48, 48)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet("QToolButton { background: rgba(0,0,0,0.45); border-radius: 24px; }"
                        "QToolButton:hover { background: rgba(0,0,0,0.7); }")
        b.clicked.connect(lambda: self.navigate.emit(step))
        return b

    def set_nav(self, has_prev: bool, has_next: bool):
        self.prev_btn.setVisible(has_prev)
        self.next_btn.setVisible(has_next)

    def resizeEvent(self, e):
        self.prev_btn.move(16, self.height() // 2 - 24)
        self.next_btn.move(self.width() - 64, self.height() // 2 - 24)

    def image_rect(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        area = QRectF(self.rect()).adjusted(12, 12, -12, -12)
        iw, ih = self.image.width(), self.image.height()
        scale = min(area.width() / iw, area.height() / ih, 2.0)
        w, h = iw * scale, ih * scale
        return QRectF(area.center().x() - w / 2, area.center().y() - h / 2, w, h)

    def face_rect(self, f: Face) -> QRectF:
        r = self.image_rect()
        return QRectF(r.left() + f.x * r.width(), r.top() + f.y * r.height(), f.w * r.width(), f.h * r.height())

    def _face_at(self, pos: QPointF) -> int | None:
        if not self.show_faces:
            return None
        best = None
        for i, f in enumerate(self.faces):
            fr = self.face_rect(f).adjusted(-4, -4, 4, 4)
            if fr.contains(pos) and (best is None or fr.width() < self.face_rect(self.faces[best]).width()):
                best = i
        return best

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        p.fillRect(self.rect(), QColor("#0a0a0b"))
        if self.image.isNull():
            p.setPen(QColor("#9aa0b0"))
            p.drawText(self.rect(), Qt.AlignCenter, self.message or "")
            return
        ir = self.image_rect()
        p.drawImage(ir, self.image)
        accent = QColor(theme.BRAND_RED)
        if self.show_faces and not self.add_mode:
            f = QFont(self.font())
            f.setPixelSize(12)
            f.setWeight(QFont.DemiBold)
            p.setFont(f)
            fm = QFontMetrics(f)
            for i, face in enumerate(self.faces):
                r = self.face_rect(face)
                hot = i == self.hover or face.id == self.highlight
                color = accent if hot else QColor(255, 255, 255, 210)
                p.setPen(QPen(QColor(0, 0, 0, 90), 4))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(r, 6, 6)
                p.setPen(QPen(color, 2.5 if hot else 1.8))
                p.drawRoundedRect(r, 6, 6)
                text = face.person_name or ("Unnamed" if face.person_id else "Who's this?")
                tw = fm.horizontalAdvance(text) + 14
                tag = QRectF(r.center().x() - tw / 2, r.bottom() + 6, tw, 22)
                p.setPen(Qt.NoPen)
                p.setBrush(accent if hot else QColor(0, 0, 0, 170))
                p.drawRoundedRect(tag, 11, 11)
                p.setPen(QColor("white"))
                p.drawText(tag, Qt.AlignCenter, text)
        if self.add_mode:
            p.fillRect(self.rect(), QColor(0, 0, 0, 60))
            p.setPen(QColor("white"))
            p.drawText(QRect(0, 14, self.width(), 24), Qt.AlignCenter, "Drag a box around the face to tag it · Esc to cancel")
            if self._drag_start and self._drag_now:
                r = QRectF(self._drag_start, self._drag_now).normalized()
                p.setPen(QPen(accent, 2, Qt.DashLine))
                p.setBrush(QColor(238, 1, 13, 45))
                p.drawRoundedRect(r, 6, 6)

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self.add_mode:
            if self._drag_start is not None:
                self._drag_now = pos
                self.update()
            return
        h = self._face_at(pos)
        if h != self.hover:
            self.hover = h
            self.setCursor(Qt.PointingHandCursor if h is not None else Qt.ArrowCursor)
            self.setToolTip("Click to change who this is" if h is not None else "")
            self.update()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        if self.add_mode:
            self._drag_start = self._drag_now = e.position()
            return
        h = self._face_at(e.position())
        if h is not None:
            self.faceClicked.emit(self.faces[h], e.globalPosition().toPoint())

    def mouseReleaseEvent(self, e):
        if not self.add_mode or self._drag_start is None:
            return
        r = QRectF(self._drag_start, e.position()).normalized()
        self._drag_start = self._drag_now = None
        self.update()
        ir = self.image_rect()
        r = r.intersected(ir)
        if r.width() < 12 or r.height() < 12:
            return
        self.boxDrawn.emit(QRectF((r.left() - ir.left()) / ir.width(), (r.top() - ir.top()) / ir.height(),
                                  r.width() / ir.width(), r.height() / ir.height()))

    def leaveEvent(self, e):
        self.hover = None
        self.update()


class FaceRow(QFrame):
    clicked = Signal(object, QPoint)

    def __init__(self, face: Face):
        super().__init__()
        self.face = face
        self.setObjectName("FaceRow")
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        av = QLabel()
        av.setPixmap(circle_pixmap(face_image(face.id), 40))
        lay.addWidget(av)
        col = QVBoxLayout()
        col.setSpacing(0)
        name = QLabel(face.person_name or ("Unnamed person" if face.person_id else "Unknown"))
        name.setStyleSheet("font-weight: 600;")
        sub = QLabel("Confirmed by you" if face.assigned_by == "user" and face.person_id
                     else "Grouped automatically" if face.person_id else "Click to say who this is")
        sub.setProperty("muted", True)
        sub.setStyleSheet(f"color: {theme.C['muted']}; font-size: 12px;")
        col.addWidget(name)
        col.addWidget(sub)
        tags = face.tags()
        if tags:
            t = QLabel(" · ".join(tags))
            t.setWordWrap(True)
            t.setStyleSheet(f"color: {theme.C['accent']}; font-size: 12px; font-weight: 600;")
            t.setToolTip("Estimated automatically")
            col.addWidget(t)
        lay.addLayout(col, 1)
        more = QLabel()
        more.setPixmap(theme.icon_pixmap("more", theme.C["muted"], 16))
        lay.addWidget(more)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.face, e.globalPosition().toPoint())


class ViewerPage(QWidget):
    closed = Signal()

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName("Viewer")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.ids: list[int] = []
        self.index = 0
        self.pil = None
        self.changed = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)

        bar = QWidget()
        bar.setObjectName("ViewerBar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(10, 8, 10, 8)
        self.back_btn = self._tool("back", "Back (Esc)", self.close_viewer)
        bl.addWidget(self.back_btn)
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 14px; font-weight: 600; padding-left: 6px;")
        bl.addWidget(self.title)
        self.position = QLabel()
        self.position.setStyleSheet("color: #9aa0b0; padding-left: 8px;")
        bl.addWidget(self.position)
        bl.addStretch(1)
        self.faces_btn = self._tool("face", "Show faces (F)", self._toggle_faces, checkable=True)
        self.faces_btn.setChecked(True)
        self.add_btn = self._tool("add_person", "Tag a face that wasn't found (drag a box)", self._toggle_add,
                                  checkable=True)
        self.info_btn = self._tool("info", "Info (I)", self._toggle_info, checkable=True)
        self.info_btn.setChecked(True)
        bl.addWidget(self.faces_btn)
        bl.addWidget(self.add_btn)
        bl.addWidget(self._tool("open_folder", "Show in folder", lambda: show_in_explorer(self._photo().path)))
        bl.addWidget(self._tool("photos", "Open with default app", lambda: open_default(self._photo().path)))
        bl.addWidget(self.info_btn)
        left.addWidget(bar)

        self.canvas = ImageCanvas()
        self.canvas.faceClicked.connect(self._face_menu)
        self.canvas.boxDrawn.connect(self._add_face)
        self.canvas.navigate.connect(self.step)
        left.addWidget(self.canvas, 1)
        root.addLayout(left, 1)

        self.info = QFrame()
        self.info.setObjectName("InfoPanel")
        self.info.setFixedWidth(320)
        il = QVBoxLayout(self.info)
        il.setContentsMargins(20, 20, 16, 16)
        il.setSpacing(6)
        h = QLabel("Info")
        h.setObjectName("H2")
        il.addWidget(h)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details.setStyleSheet("line-height: 150%;")
        il.addWidget(self.details)
        il.addSpacing(10)
        ph = QLabel("People")
        ph.setObjectName("H2")
        il.addWidget(ph)
        self.people_hint = QLabel()
        self.people_hint.setWordWrap(True)
        theme.set_style(self.people_hint, "color: %(muted)s; font-size: 12px;")
        il.addWidget(self.people_hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        self.rows_host = QWidget()
        self.rows = QVBoxLayout(self.rows_host)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setSpacing(2)
        self.rows.addStretch(1)
        scroll.setWidget(self.rows_host)
        il.addWidget(scroll, 1)
        root.addWidget(self.info)

    def _tool(self, glyph, tip, slot, checkable=False) -> QToolButton:
        b = QToolButton()
        b.setIcon(theme.icon(glyph, "#f2f3f7", 18))
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setFixedSize(38, 38)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(slot)
        return b

    # ---- navigation -----------------------------------------------------------
    def open(self, photo_ids: list[int], index: int, highlight_face: int | None = None):
        self.ids = list(photo_ids)
        self.index = max(0, min(index, len(self.ids) - 1))
        self.changed = False
        self.canvas.highlight = highlight_face
        self.add_btn.setChecked(False)
        self.canvas.add_mode = False
        self._load()
        self.setFocus()

    def _photo(self):
        return self.ctx.db.photo(self.ids[self.index])

    def step(self, delta: int):
        n = self.index + delta
        if 0 <= n < len(self.ids):
            self.index = n
            self.canvas.highlight = None
            self._load()

    def _load(self):
        ph = self._photo()
        self.canvas.set_nav(self.index > 0, self.index < len(self.ids) - 1)
        if ph is None:
            self.canvas.image = QImage()
            self.canvas.message = "This photo is no longer in your library."
            self.canvas.faces = []
            self.canvas.update()
            return
        self.title.setText(os.path.basename(ph.path))
        self.position.setText(f"{self.index + 1:,} of {len(self.ids):,}" if len(self.ids) > 1 else "")
        self.setCursor(Qt.WaitCursor)
        try:
            li = imaging.load(ph.path, 2560)
            self.pil = li.image
            data = li.image.tobytes()
            w, h = li.image.size
            self.canvas.image = QImage(data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.canvas.message = ""
        except Exception as e:
            self.pil = None
            self.canvas.image = QImage()
            self.canvas.message = f"Can't open this photo\n{e}"
        finally:
            self.unsetCursor()
        self._load_faces()
        self._set_details(ph)

    def _set_details(self, ph):
        taken = datetime.fromtimestamp(ph.taken_at).strftime("%A, %B %d, %Y · %I:%M %p") if ph.taken_at else "Unknown"
        dims = f"{ph.width} × {ph.height}" if ph.width else ""
        mp = f" · {ph.width * ph.height / 1e6:.1f} MP" if ph.width else ""
        self.details.setText(
            f"<b>{taken}</b><br>"
            f"<span style='color:{theme.C['muted']}'>{dims}{mp} · {_fmt_size(ph.size)}</span><br><br>"
            f"{os.path.basename(ph.path)}<br>"
            f"<span style='color:{theme.C['muted']}'>{os.path.dirname(ph.path)}</span>")

    def retheme(self):
        """Redo the palette-colored parts of the info panel after a theme switch."""
        ph = self._photo() if self.ids else None
        if ph is not None:
            self._set_details(ph)
            self._load_faces()

    def _load_faces(self):
        ph = self._photo()
        faces = self.ctx.db.faces_in_photo(ph.id) if ph else []
        self.canvas.faces = faces
        self.canvas.hover = None
        self.canvas.update()
        while self.rows.count() > 1:
            w = self.rows.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        for f in faces:
            row = FaceRow(f)
            row.clicked.connect(self._face_menu)
            self.rows.insertWidget(self.rows.count() - 1, row)
        self.people_hint.setText(
            "Click a face to correct who it is." if faces else
            "No faces found. Use the “Tag a face” button above to add one that was missed.")

    # ---- corrections ----------------------------------------------------------
    def _face_menu(self, face: Face, pos: QPoint):
        db = self.ctx.db
        m = QMenu(self)
        title = m.addAction(face.person_name or ("Unnamed person" if face.person_id else "Unknown face"))
        title.setEnabled(False)
        m.addSeparator()
        m.addAction(theme.icon("person"), "This is someone else…" if face.person_id else "Who is this?…",
                    lambda: self._assign(face))
        if face.person_id and not face.person_name:
            m.addAction(theme.icon("edit"), "Name this person…", lambda: self._name(face))
        if face.person_id:
            if face.assigned_by != "user":
                m.addAction(theme.icon("check"), f"Confirm it's {face.person_name or 'this person'}",
                            lambda: self._apply(lambda: db.assign_faces([face.id], face.person_id)))
            m.addAction(theme.icon("block"), f"Not {face.person_name or 'this person'}",
                        lambda: self._apply(lambda: db.unassign_faces([face.id])))
            m.addAction(theme.icon("people"), f"Go to {face.person_name or 'this person'}",
                        lambda: self.ctx.open_person(face.person_id))
        m.addSeparator()
        if face.manual:
            m.addAction(theme.icon("delete"), "Remove this face tag", lambda: self._apply(
                lambda: (db.x("DELETE FROM faces WHERE id=?", (face.id,)), db.cleanup_people())))
        else:
            m.addAction(theme.icon("close"), "Not a face / ignore", lambda: self._apply(lambda: db.ignore_faces([face.id])))
        m.exec(pos)

    def _apply(self, fn):
        fn()
        self.changed = True
        self.ctx.data_changed(quiet=True)
        self._load_faces()

    def _assign(self, face: Face):
        pid = dialogs.pick_person(self, self.ctx.db, title="Who is this?",
                                  exclude={face.person_id} if face.person_id else None)
        if pid is not None:
            self._apply(lambda: self.ctx.db.assign_faces([face.id], pid))

    def _name(self, face: Face):
        if dialogs.rename_person(self, self.ctx.db, face.person_id) is not None:
            self._apply(lambda: None)

    def _add_face(self, box: QRectF):
        self.add_btn.setChecked(False)
        self.canvas.add_mode = False
        self.canvas.update()
        ph = self._photo()
        if ph is None or self.pil is None:
            return
        db = self.ctx.db
        bgr = np.ascontiguousarray(np.asarray(self.pil)[:, :, ::-1])
        found = self.ctx.face_engine().embed_region(bgr, (box.x(), box.y(), box.width(), box.height()))
        face_id = db.insert_face(ph.id, found, manual=True)
        try:
            imaging.save_face_crop(self.pil, found.box, config.face_crop_path(face_id))
        except OSError:
            pass
        pid = dialogs.pick_person(self, db, title="Who is this?")
        if pid is None:
            db.x("DELETE FROM faces WHERE id=?", (face_id,))
        else:
            db.assign_faces([face_id], pid)
            self.changed = True
            self.ctx.data_changed(quiet=True)
        self._load_faces()

    # ---- toggles / keys -------------------------------------------------------
    def _toggle_faces(self):
        self.canvas.show_faces = self.faces_btn.isChecked()
        self.canvas.update()

    def _toggle_add(self):
        self.canvas.add_mode = self.add_btn.isChecked()
        self.canvas.setCursor(Qt.CrossCursor if self.canvas.add_mode else Qt.ArrowCursor)
        self.canvas.update()

    def _toggle_info(self):
        self.info.setVisible(self.info_btn.isChecked())

    def close_viewer(self):
        self.pil = None
        self.canvas.image = QImage()
        self.closed.emit()

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Escape:
            if self.canvas.add_mode:
                self.add_btn.setChecked(False)
                self._toggle_add()
            else:
                self.close_viewer()
        elif k in (Qt.Key_Left, Qt.Key_PageUp):
            self.step(-1)
        elif k in (Qt.Key_Right, Qt.Key_PageDown, Qt.Key_Space):
            self.step(1)
        elif k == Qt.Key_I:
            self.info_btn.toggle()
            self._toggle_info()
        elif k == Qt.Key_F:
            self.faces_btn.toggle()
            self._toggle_faces()
        else:
            super().keyPressEvent(e)
