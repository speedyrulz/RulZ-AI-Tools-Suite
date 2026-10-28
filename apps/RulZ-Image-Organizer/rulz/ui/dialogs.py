"""Dialogs: text input, confirmation, person picker, and the rename/merge flows."""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QVBoxLayout, QWidget)

from ..db import Database, Person
from . import theme
from .common import button, circle_pixmap, face_image, plural


class _Base(QDialog):
    def __init__(self, parent: QWidget | None, title: str, width: int = 420):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(width)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(24, 22, 24, 20)
        self.lay.setSpacing(12)
        h = QLabel(title)
        h.setObjectName("H2")
        self.lay.addWidget(h)

    def add_buttons(self, ok_text: str, *, danger=False, cancel_text="Cancel"):
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = button(cancel_text)
        cancel.clicked.connect(self.reject)
        ok = button(ok_text, primary=not danger, danger=danger)
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(ok)
        self.lay.addSpacing(6)
        self.lay.addLayout(row)
        return ok


def ask_text(parent, title: str, prompt: str, text: str = "", ok_text: str = "Save",
             placeholder: str = "") -> str | None:
    dlg = _Base(parent, title)
    if prompt:
        lb = QLabel(prompt)
        lb.setWordWrap(True)
        lb.setProperty("muted", True)
        dlg.lay.addWidget(lb)
    edit = QLineEdit(text)
    edit.setProperty("plain", True)
    edit.setPlaceholderText(placeholder)
    edit.selectAll()
    dlg.lay.addWidget(edit)
    dlg.add_buttons(ok_text)
    edit.returnPressed.connect(dlg.accept)
    return edit.text().strip() if dlg.exec() == QDialog.Accepted else None


def confirm(parent, title: str, text: str, ok_text: str = "OK", danger: bool = False) -> bool:
    dlg = _Base(parent, title)
    lb = QLabel(text)
    lb.setWordWrap(True)
    dlg.lay.addWidget(lb)
    dlg.add_buttons(ok_text, danger=danger)
    return dlg.exec() == QDialog.Accepted


def info(parent, title: str, text: str) -> None:
    dlg = _Base(parent, title)
    lb = QLabel(text)
    lb.setWordWrap(True)
    dlg.lay.addWidget(lb)
    row = QHBoxLayout()
    row.addStretch(1)
    ok = button("OK", primary=True)
    ok.clicked.connect(dlg.accept)
    row.addWidget(ok)
    dlg.lay.addLayout(row)
    dlg.exec()


class PersonPicker(_Base):
    """Pick an existing person or type a new name. ``result_value`` is ('person', id) or ('new', name)."""

    NEW = -1

    def __init__(self, parent, db: Database, title: str = "Who is this?", exclude: set[int] | None = None,
                 allow_new: bool = True, hint: str = ""):
        super().__init__(parent, title, 440)
        self.db = db
        self.allow_new = allow_new
        self.result_value = None
        if hint:
            lb = QLabel(hint)
            lb.setWordWrap(True)
            lb.setProperty("muted", True)
            self.lay.addWidget(lb)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Type a name…" if allow_new else "Search people…")
        self.search.setProperty("plain", True)
        self.lay.addWidget(self.search)
        self.list = QListWidget()
        self.list.setIconSize(QSize(36, 36))
        self.list.setMinimumHeight(360)
        self.lay.addWidget(self.list)
        self.people = [p for p in db.people(include_hidden=True) if p.id not in (exclude or set())]
        self._icons: dict[int, QIcon] = {}
        self.search.textChanged.connect(self._fill)
        self.search.returnPressed.connect(self._enter)
        self.list.itemActivated.connect(self._choose)
        self.list.itemClicked.connect(self._choose)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = button("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        self.lay.addLayout(row)
        self._fill("")
        self.search.setFocus()

    def _icon(self, p: Person) -> QIcon:
        if p.id not in self._icons:
            self._icons[p.id] = QIcon(circle_pixmap(face_image(p.cover_face_id), 36))
        return self._icons[p.id]

    def _fill(self, text: str):
        self.list.clear()
        q = text.strip().lower()
        if self.allow_new and q and not any((p.name or "").lower() == q for p in self.people):
            it = QListWidgetItem(theme.icon("add_person", theme.C["accent"], 22), f'Add new person "{text.strip()}"')
            it.setData(Qt.UserRole, self.NEW)
            self.list.addItem(it)
        elif self.allow_new and not q:
            it = QListWidgetItem(theme.icon("add_person", theme.C["accent"], 22), "New person (type a name above)")
            it.setData(Qt.UserRole, self.NEW)
            it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
            self.list.addItem(it)
        for p in self.people:
            if q and q not in (p.name or "").lower():
                continue
            label = p.name or f"Unnamed person"
            it = QListWidgetItem(self._icon(p), f"{label}   ·   {plural(p.photo_count, 'photo')}"
                                 + ("   (hidden)" if p.hidden else ""))
            it.setData(Qt.UserRole, p.id)
            self.list.addItem(it)
        for i in range(self.list.count()):
            if self.list.item(i).flags() & Qt.ItemIsEnabled:
                self.list.setCurrentRow(i)
                break

    def _enter(self):
        it = self.list.currentItem()
        if it is not None:
            self._choose(it)

    def _choose(self, item: QListWidgetItem):
        if not item.flags() & Qt.ItemIsEnabled:
            return
        pid = item.data(Qt.UserRole)
        if pid == self.NEW:
            name = self.search.text().strip()
            if not name:
                return
            existing = self.db.person_by_name(name)
            self.result_value = ("person", existing.id) if existing else ("new", name)
        else:
            self.result_value = ("person", pid)
        self.accept()


def pick_person(parent, db: Database, **kw) -> int | None:
    """Show the picker and return a person id (creating the person if a new name was typed)."""
    dlg = PersonPicker(parent, db, **kw)
    if dlg.exec() != QDialog.Accepted or not dlg.result_value:
        return None
    kind, value = dlg.result_value
    return db.create_person(value) if kind == "new" else value


def rename_person(parent, db: Database, person_id: int) -> int | None:
    """Ask for a name. Returns the person's id afterwards (another id if merged), or None if cancelled."""
    p = db.person(person_id)
    if p is None:
        return None
    name = ask_text(parent, "Name this person" if not p.name else "Rename",
                    "Names are only stored on this PC.", p.name or "", placeholder="Name")
    if name is None:
        return None
    if not name:
        db.rename_person(person_id, None)
        return person_id
    other = db.person_by_name(name)
    if other and other.id != person_id:
        if confirm(parent, "Same person?",
                   f"You already have someone named “{other.name}”. Are these the same person?\n\n"
                   f"Their photos will be combined.", ok_text="Yes, merge"):
            db.merge_people([person_id], other.id)
            return other.id
        return None
    db.rename_person(person_id, name)
    return person_id


def merge_people(parent, db: Database, person_ids: list[int]) -> int | None:
    """Merge the given people. Returns the surviving person id."""
    people = [p for p in (db.person(i) for i in person_ids) if p]
    if len(people) < 2:
        return None
    names = sorted({p.name for p in people if p.name})
    target = max(people, key=lambda p: (p.name is not None, p.photo_count))
    if len(names) > 1:
        dlg = _Base(parent, "Merge people", 420)
        lb = QLabel("These people have different names. Which name should be kept?")
        lb.setWordWrap(True)
        dlg.lay.addWidget(lb)
        lst = QListWidget()
        lst.setIconSize(QSize(32, 32))
        for p in people:
            if p.name:
                it = QListWidgetItem(QIcon(circle_pixmap(face_image(p.cover_face_id), 32)), p.name)
                it.setData(Qt.UserRole, p.id)
                lst.addItem(it)
        lst.setCurrentRow(0)
        dlg.lay.addWidget(lst)
        dlg.add_buttons("Merge")
        lst.itemDoubleClicked.connect(lambda _: dlg.accept())
        if dlg.exec() != QDialog.Accepted:
            return None
        target = db.person(lst.currentItem().data(Qt.UserRole))
    elif not confirm(parent, "Merge people",
                     f"Combine these {len(people)} groups into one person"
                     + (f" named “{names[0]}”" if names else "") + "?", ok_text="Merge"):
        return None
    db.merge_people([p.id for p in people if p.id != target.id], target.id)
    return target.id


def merge_into(parent, db: Database, person_id: int) -> int | None:
    """Merge one person into another chosen with the picker."""
    p = db.person(person_id)
    target = pick_person(parent, db, title="Merge with…", exclude={person_id}, allow_new=False,
                         hint=f"Pick the person that “{p.display_name if p else ''}” should be combined with.")
    if target is None:
        return None
    t = db.person(target)
    if p and p.name and t and not t.name:
        # Keep the name when merging a named person into an unnamed group.
        db.merge_people([target], person_id)
        return person_id
    db.merge_people([person_id], target)
    return target


# ---- first run: choose Dark or Light -------------------------------------------------
class _ThemeCard(QAbstractButton):
    """A big selectable card with a tiny preview of one theme."""

    def __init__(self, title: str, palette: dict, parent=None):
        super().__init__(parent)
        self.title = title
        self.pal = palette
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(236, 196)
        self.setFocusPolicy(Qt.TabFocus)

    def paintEvent(self, e):
        C, P = theme.C, self.pal
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        outer = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        sel = self.isChecked()
        p.setPen(QPen(QColor(C["accent"] if sel else C["border"]), 2.5 if sel else 1))
        p.setBrush(QColor(C["select"] if sel else C["card"]) if sel or self.underMouse() else QColor(C["card"]))
        p.drawRoundedRect(outer, 14, 14)

        # mini window
        win = QRectF(16, 16, self.width() - 32, 122)
        p.setPen(QPen(QColor(P["border"]), 1))
        p.setBrush(QColor(P["bg"]))
        p.drawRoundedRect(win, 8, 8)
        side = QRectF(win.left() + 1, win.top() + 1, 52, win.height() - 2)
        path = QPainterPath()
        path.addRoundedRect(side, 7, 7)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(P["sidebar"]))
        p.drawPath(path)
        p.setBrush(QColor(P["accent"]))
        p.drawRoundedRect(QRectF(side.left() + 8, side.top() + 10, 14, 6), 2, 2)
        p.setBrush(QColor(P["side_text"]))
        p.drawRoundedRect(QRectF(side.left() + 24, side.top() + 10, 20, 6), 2, 2)
        for i in range(3):
            p.setBrush(QColor(P["side_select"] if i == 0 else P["side_hover"]))
            p.drawRoundedRect(QRectF(side.left() + 6, side.top() + 28 + i * 14, side.width() - 12, 10), 3, 3)
        x0 = side.right() + 10
        p.setBrush(QColor(P["text"]))
        p.drawRoundedRect(QRectF(x0, win.top() + 12, 54, 8), 3, 3)
        p.setBrush(QColor(P["accent"]))
        p.drawRoundedRect(QRectF(win.right() - 42, win.top() + 10, 32, 12), 6, 6)
        tw = (win.right() - 10 - x0 - 3 * 6) / 4
        for r in range(2):
            for c in range(4):
                p.setBrush(QColor(P["placeholder"]))
                p.drawRoundedRect(QRectF(x0 + c * (tw + 6), win.top() + 34 + r * (tw + 6), tw, tw), 3, 3)

        # radio + title
        cy = 166
        p.setPen(QPen(QColor(C["accent"] if sel else C["muted"]), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(20, cy - 8, 16, 16))
        if sel:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(C["accent"]))
            p.drawEllipse(QRectF(24, cy - 4, 8, 8))
        f = QFont(self.font())
        f.setPixelSize(15)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor(C["text"]))
        p.drawText(QRectF(44, cy - 12, 160, 24), Qt.AlignVCenter | Qt.AlignLeft, self.title)

    def enterEvent(self, e):
        self.update()

    def leaveEvent(self, e):
        self.update()


class ThemeChoiceDialog(QDialog):
    """First launch: pick Dark (preselected) or Light. The pick previews live and is saved on Continue;
    closing the dialog keeps Dark."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Choose your look")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 22)
        lay.setSpacing(10)
        h = QLabel("Choose your look")
        h.setObjectName("PageTitle")
        lay.addWidget(h)
        sub = QLabel("How should RulZ Image Organizer look? You can change this any time with the "
                     "Light mode / Dark mode button at the bottom of the sidebar, or in Library ▸ Appearance.")
        sub.setWordWrap(True)
        sub.setObjectName("Muted")
        sub.setMaximumWidth(500)
        lay.addWidget(sub)
        lay.addSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(16)
        self.dark_card = _ThemeCard("Dark", theme.DARK)
        self.light_card = _ThemeCard("Light", theme.LIGHT)
        self.dark_card.setChecked(True)
        for c in (self.dark_card, self.light_card):
            c.toggled.connect(self._preview)
            row.addWidget(c)
        lay.addLayout(row)
        lay.addSpacing(10)
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.ok = button("Continue", primary=True)
        self.ok.setDefault(True)
        self.ok.clicked.connect(self.accept)
        btns.addWidget(self.ok)
        lay.addLayout(btns)
        self.setFixedSize(self.sizeHint())

    def chosen_dark(self) -> bool:
        return not self.light_card.isChecked()

    def _preview(self, on: bool):
        if on:
            theme.apply(QApplication.instance(), self.chosen_dark())
            theme.retheme_widgets(self)
            for c in (self.dark_card, self.light_card):
                c.update()


def choose_theme_first_run() -> bool:
    """Shows the "Choose your look" dialog, saves and applies the result. Closing it = Dark."""
    dlg = ThemeChoiceDialog()
    dark = dlg.chosen_dark() if dlg.exec() == QDialog.Accepted else True
    theme.save_choice(dark)
    theme.apply(QApplication.instance(), dark)
    return dark
