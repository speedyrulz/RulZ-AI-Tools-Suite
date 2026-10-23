"""Dialogs: text input, confirmation, person picker, and the rename/merge flows."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout,
                               QWidget)

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
