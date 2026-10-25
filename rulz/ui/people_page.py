"""People overview and the per-person page with face corrections."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QMenu, QPushButton, QStackedWidget, QVBoxLayout,
                               QWidget)

from .. import clustering
from . import dialogs, theme
from .common import SelectionBar, Segmented, button, circle_pixmap, face_image, label, month_sections, plural
from .tilegrid import Section, Tile, TileGrid


class PeoplePage(QStackedWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.dirty = True
        self.overview = PeopleOverview(ctx, self)
        self.detail = PersonPage(ctx, self)
        self.addWidget(self.overview)
        self.addWidget(self.detail)

    def refresh(self):
        self.dirty = False
        if self.currentWidget() is self.detail:
            if not self.detail.refresh():
                self.setCurrentWidget(self.overview)
                self.overview.refresh()
        else:
            self.overview.refresh()

    def open_person(self, person_id: int, tab: int = 0):
        self.detail.person_id = person_id
        self.detail.tabs.set_current(tab)
        self.detail.stack.setCurrentIndex(tab)
        self.setCurrentWidget(self.detail)
        self.detail.refresh(reset_scroll=True)

    def back(self):
        self.setCurrentWidget(self.overview)
        self.overview.refresh()


class PeopleOverview(QWidget):
    def __init__(self, ctx, owner: PeoplePage):
        super().__init__()
        self.ctx, self.owner = ctx, owner
        self.setObjectName("Page")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(28, 22, 28, 4)
        top.setSpacing(12)
        top.addWidget(label("People", "PageTitle"))
        self.count = label("", "Muted")
        top.addWidget(self.count, 0, Qt.AlignBottom)
        top.addStretch(1)
        self.show_hidden = QCheckBox("Show hidden people")
        self.show_hidden.toggled.connect(self.refresh)
        top.addWidget(self.show_hidden)
        lay.addLayout(top)
        hint = label("RulZ groups similar faces automatically. Click a person to name them, or select "
                     "several groups of the same person and merge them.", "Muted", wrap=True)
        hint.setContentsMargins(30, 0, 30, 8)
        lay.addWidget(hint)

        self.sel_bar = SelectionBar()
        self.merge_btn = self.sel_bar.add(button("Merge", "merge", primary=True,
                                                 tooltip="These are all the same person"))
        self.merge_btn.clicked.connect(self._merge)
        self.hide_btn = self.sel_bar.add(button("Hide", "hide"))
        self.hide_btn.clicked.connect(self._toggle_hidden)
        self.sel_bar.add(button("Ungroup", "block", tooltip="Dissolve these groups – their faces become "
                                "unassigned")).clicked.connect(self._ungroup)
        wrap = QHBoxLayout()
        wrap.setContentsMargins(24, 0, 24, 6)
        wrap.addWidget(self.sel_bar)
        lay.addLayout(wrap)

        self.grid = TileGrid(tile_size=150, spacing=18, circle=True, captions=True)
        self.grid.activated.connect(lambda t: owner.open_person(t.key))
        self.grid.contextMenuRequested.connect(self._menu)
        self.grid.selectionChanged.connect(self._sel_changed)
        self.sel_bar.cleared.connect(self.grid.clear_selection)
        lay.addWidget(self.grid, 1)

    def refresh(self):
        db = self.ctx.db
        people = db.people(include_hidden=self.show_hidden.isChecked())
        named = [p for p in people if p.name]
        unnamed = [p for p in people if not p.name]

        def tile(p):
            return Tile(key=p.id, image=face_image(p.cover_face_id),
                        caption=p.name or "Add a name", subcaption=plural(p.photo_count, "photo")
                        + (" · hidden" if p.hidden else ""),
                        data=None if p.name else "unnamed")

        sections = []
        if named:
            sections.append(Section("Named", [tile(p) for p in named], plural(len(named), "person")))
        if unnamed:
            sections.append(Section("Unnamed groups", [tile(p) for p in unnamed],
                                    "Click to name, or select duplicates and merge"))
        self.grid.set_sections(sections)
        self.count.setText(plural(len(people), "person") if people else "")
        stats = db.stats()
        if stats["photos"] == 0:
            self.grid.set_empty("No people yet", "Add a folder in Library. People appear here once their "
                                "faces have been found in your photos.")
        else:
            self.grid.set_empty("No people found yet",
                                "Faces are grouped once the same person appears in at least two photos.")

    def _ids(self) -> list[int]:
        return [t.key for t in self.grid.selected_tiles()]

    def _sel_changed(self):
        ids = self._ids()
        self.sel_bar.update_count(len(ids))
        self.merge_btn.setEnabled(len(ids) >= 2)
        all_hidden = bool(ids) and all(self.ctx.db.person(i).hidden for i in ids if self.ctx.db.person(i))
        self.hide_btn.setText("Unhide" if all_hidden else "Hide")

    def _merge(self):
        if dialogs.merge_people(self, self.ctx.db, self._ids()) is not None:
            self.grid.clear_selection()
            self.ctx.data_changed()

    def _toggle_hidden(self):
        ids = self._ids()
        hide = self.hide_btn.text() == "Hide"
        for i in ids:
            self.ctx.db.set_hidden(i, hide)
        self.grid.clear_selection()
        self.ctx.data_changed()

    def _ungroup(self):
        ids = self._ids()
        if dialogs.confirm(self, "Ungroup", f"Dissolve {plural(len(ids), 'group')}? Their faces become "
                           "unassigned and won't be grouped together automatically again. Names are removed.",
                           ok_text="Ungroup", danger=True):
            for i in ids:
                self.ctx.db.ungroup_person(i)
            self.grid.clear_selection()
            self.ctx.data_changed()

    def _menu(self, tile, pos):
        db = self.ctx.db
        p = db.person(tile.key)
        if p is None:
            return
        ids = self._ids()
        m = QMenu(self)
        if len(ids) > 1:
            m.addAction(theme.icon("merge"), f"Merge {len(ids)} people", self._merge)
            m.addAction(theme.icon("hide"), "Hide / unhide", self._toggle_hidden)
            m.exec(pos)
            return
        m.addAction(theme.icon("show"), "Open", lambda: self.owner.open_person(p.id))
        m.addAction(theme.icon("edit"), "Rename…" if p.name else "Add a name…",
                    lambda: dialogs.rename_person(self, db, p.id) and self.ctx.data_changed())
        m.addAction(theme.icon("merge"), "Merge with…",
                    lambda: dialogs.merge_into(self, db, p.id) is not None and self.ctx.data_changed())
        m.addAction(theme.icon("photos"), "Show photos in timeline", lambda: self.ctx.search_photos(p.name)) \
            .setEnabled(bool(p.name))
        m.addSeparator()
        m.addAction(theme.icon("show" if p.hidden else "hide"), "Unhide" if p.hidden else "Hide",
                    lambda: (db.set_hidden(p.id, not p.hidden), self.ctx.data_changed()))
        m.addAction(theme.icon("block"), "Ungroup…", self._ungroup)
        m.exec(pos)


@dataclass
class _PersonData:
    version: int
    photo_sections: list[Section]
    face_tiles: list[Tile]
    suggestion_tiles: list[Tile]


class PersonPage(QWidget):
    TAB_PHOTOS, TAB_FACES, TAB_SUGGEST = range(3)
    CACHE_SIZE = 8  # recently opened people kept in memory

    def __init__(self, ctx, owner: PeoplePage):
        super().__init__()
        self.ctx, self.owner = ctx, owner
        self.person_id: int | None = None
        self._cache: OrderedDict[int, _PersonData] = OrderedDict()
        self._shown: tuple[int, int] | None = None  # (person, version) currently in the grids
        self.setObjectName("Page")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(16, 16, 28, 8)
        head.setSpacing(14)
        back = QPushButton()
        theme.set_icon(back, "back", "text", 18)
        back.setProperty("flat", True)
        back.setToolTip("Back to People")
        back.setCursor(Qt.PointingHandCursor)
        back.clicked.connect(owner.back)
        head.addWidget(back, 0, Qt.AlignTop)
        self.avatar = QLabel()
        self.avatar.setFixedSize(76, 76)
        head.addWidget(self.avatar)
        col = QVBoxLayout()
        col.setSpacing(2)
        name_row = QHBoxLayout()
        self.name = QPushButton()
        self.name.setProperty("flat", True)
        self.name.setCursor(Qt.PointingHandCursor)
        self.name.setStyleSheet("font-size: 24px; font-weight: 600; text-align: left; padding: 2px 6px;")
        self.name.setToolTip("Click to rename")
        self.name.clicked.connect(self._rename)
        name_row.addWidget(self.name)
        name_row.addStretch(1)
        col.addLayout(name_row)
        self.sub = label("", "Muted")
        self.sub.setContentsMargins(8, 0, 0, 0)
        col.addWidget(self.sub)
        head.addLayout(col, 1)
        self.rename_btn = button("Rename", "edit")
        self.rename_btn.clicked.connect(self._rename)
        self.merge_btn = button("Merge with…", "merge")
        self.merge_btn.clicked.connect(self._merge)
        self.hide_btn = button("Hide", "hide")
        self.hide_btn.clicked.connect(self._toggle_hidden)
        more = button("", "more", tooltip="More")
        mm = QMenu(more)
        mm.addAction(theme.icon("photos"), "Show in timeline", lambda: self.ctx.search_photos(self._p().name)
                     if self._p() and self._p().name else None)
        mm.addAction(theme.icon("block"), "Ungroup this person…", self._ungroup)
        more.setMenu(mm)
        for b in (self.rename_btn, self.merge_btn, self.hide_btn, more):
            head.addWidget(b)
        lay.addLayout(head)

        tabs_row = QHBoxLayout()
        tabs_row.setContentsMargins(28, 6, 28, 8)
        self.tabs = Segmented(["Photos", "Faces", "Review suggestions"])
        self.tabs.changed.connect(self._tab_changed)
        tabs_row.addWidget(self.tabs)
        tabs_row.addSpacing(12)
        self.tab_hint = label("", "Muted", wrap=True)
        tabs_row.addWidget(self.tab_hint, 1)
        lay.addLayout(tabs_row)

        # selection bars
        self.face_bar = SelectionBar()
        self.not_btn = self.face_bar.add(button("Not this person", "block"))
        self.not_btn.clicked.connect(self._not_person)
        self.face_bar.add(button("Move to…", "person")).clicked.connect(self._move_faces)
        self.cover_btn = self.face_bar.add(button("Use as profile photo", "star"))
        self.cover_btn.clicked.connect(self._set_cover)
        self.face_bar.add(button("Not a face", "close", tooltip="Hide this detection everywhere")) \
            .clicked.connect(self._not_a_face)
        self.sug_bar = SelectionBar()
        self.yes_btn = self.sug_bar.add(button("Yes, add", "check", primary=True))
        self.yes_btn.clicked.connect(self._accept_suggestions)
        self.no_btn = self.sug_bar.add(button("No, not them", "close"))
        self.no_btn.clicked.connect(self._reject_suggestions)
        bars = QVBoxLayout()
        bars.setContentsMargins(24, 0, 24, 6)
        bars.addWidget(self.face_bar)
        bars.addWidget(self.sug_bar)
        lay.addLayout(bars)

        self.stack = QStackedWidget()
        self.photos = TileGrid(tile_size=190, spacing=4, selectable=False)
        self.photos.activated.connect(self._open_photo)
        self.faces = TileGrid(tile_size=120, spacing=10)
        self.faces.activated.connect(self._open_face)
        self.faces.contextMenuRequested.connect(self._face_menu)
        self.faces.selectionChanged.connect(self._faces_sel)
        self.face_bar.cleared.connect(self.faces.clear_selection)
        self.suggest = TileGrid(tile_size=120, spacing=10)
        self.suggest.activated.connect(self._open_face)
        self.suggest.selectionChanged.connect(self._sug_sel)
        self.sug_bar.cleared.connect(self.suggest.clear_selection)
        for g in (self.photos, self.faces, self.suggest):
            self.stack.addWidget(g)
        lay.addWidget(self.stack, 1)

    # ---- data -----------------------------------------------------------------
    def _p(self):
        return self.ctx.db.person(self.person_id) if self.person_id else None

    def _data(self, person_id: int) -> "_PersonData":
        """Photo/face/suggestion tiles for a person, cached until the library data changes."""
        version = self.ctx.data_version
        data = self._cache.get(person_id)
        if data is None or data.version != version:
            db = self.ctx.db
            data = _PersonData(
                version,
                month_sections(db.photos([person_id])),
                [Tile(key=f.id, image=face_image(f.id), data=f) for f, _ in db.faces_of_person(person_id)],
                [Tile(key=fid, image=face_image(fid), badge=f"{round(score * 100)}%")
                 for fid, score in clustering.suggestions(db, person_id)])
            self._cache[person_id] = data
        self._cache.move_to_end(person_id)
        while len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)
        return data

    def refresh(self, reset_scroll: bool = False) -> bool:
        db = self.ctx.db
        p = self._p()
        if p is None or p.face_count == 0:
            return False
        self.name.setText(p.name or "Add a name")
        self.name.setStyleSheet("font-size: 24px; font-weight: 600; text-align: left; padding: 2px 6px;"
                                + ("" if p.name else f"color: {theme.C['accent']};"))
        self.sub.setText(f"{plural(p.photo_count, 'photo')} · {plural(p.face_count, 'face')}"
                         + (" · hidden from People" if p.hidden else ""))
        self.avatar.setPixmap(circle_pixmap(face_image(p.cover_face_id), 76))
        self.hide_btn.setText("Unhide" if p.hidden else "Hide")
        who = p.name or "this person"
        self.not_btn.setText(f"Not {p.name}" if p.name else "Not this person")
        self.yes_btn.setText(f"Yes, it's {p.name}" if p.name else "Yes, same person")

        data = self._data(p.id)
        if self._shown != (p.id, data.version) or reset_scroll:
            self._shown = (p.id, data.version)
            self.photos.set_sections(data.photo_sections, keep_scroll=not reset_scroll)
            self.faces.set_sections([Section(None, data.face_tiles)], keep_scroll=not reset_scroll)
            self.suggest.set_sections([Section(None, data.suggestion_tiles)], keep_scroll=not reset_scroll)
        s_tiles = data.suggestion_tiles
        self.suggest.set_empty("Nothing to review", f"There are no other unassigned faces that look like {who}.")
        self.tabs.set_text(self.TAB_SUGGEST, f"Review suggestions ({len(s_tiles)})" if s_tiles else "Review suggestions")
        self._update_hint()
        return True

    def _update_hint(self):
        p = self._p()
        who = (p.name if p and p.name else "this person")
        i = self.stack.currentIndex()
        self.tab_hint.setText([
            "",
            f"Every face grouped as {who}. Select any that aren't {who} and choose “Not {who}” or “Move to…”.",
            f"Unassigned faces that might be {who}. Select the right ones and confirm.",
        ][i])
        self.face_bar.setVisible(i == self.TAB_FACES and bool(self.faces.selected))
        self.sug_bar.setVisible(i == self.TAB_SUGGEST and bool(self.suggest.selected))

    def _tab_changed(self, i: int):
        self.stack.setCurrentIndex(i)
        self._update_hint()

    # ---- header actions -------------------------------------------------------
    def _rename(self):
        result = dialogs.rename_person(self, self.ctx.db, self.person_id)
        if result is not None:
            self.person_id = result  # differs if it was merged into an existing person
            self.ctx.data_changed()

    def _merge(self):
        target = dialogs.merge_into(self, self.ctx.db, self.person_id)
        if target is not None:
            self.person_id = target
            self.ctx.data_changed()

    def _toggle_hidden(self):
        p = self._p()
        if p:
            self.ctx.db.set_hidden(p.id, not p.hidden)
            self.ctx.data_changed()

    def _ungroup(self):
        if dialogs.confirm(self, "Ungroup", "Dissolve this group? Its faces become unassigned and the name is "
                           "removed.", ok_text="Ungroup", danger=True):
            self.ctx.db.ungroup_person(self.person_id)
            self.owner.back()
            self.ctx.data_changed()

    # ---- photos / faces -------------------------------------------------------
    def _open_photo(self, tile):
        ids = [t.key for t in self.photos.flat]
        self.ctx.open_viewer(ids, ids.index(tile.key))

    def _open_face(self, tile):
        f = self.ctx.db.face(tile.key)
        if f:
            self.ctx.open_viewer([f.photo_id], 0, highlight_face=f.id)

    def _faces_sel(self):
        n = len(self.faces.selected)
        self.face_bar.update_count(n, "faces selected" if n != 1 else "face selected")
        self.cover_btn.setEnabled(n == 1)
        self._update_hint()

    def _sug_sel(self):
        n = len(self.suggest.selected)
        self.sug_bar.update_count(n, "faces selected" if n != 1 else "face selected")
        self._update_hint()

    def _face_ids(self) -> list[int]:
        return [t.key for t in self.faces.selected_tiles()]

    def _not_person(self):
        self.ctx.db.unassign_faces(self._face_ids())
        self.faces.clear_selection()
        self.ctx.data_changed()

    def _move_faces(self):
        ids = self._face_ids()
        target = dialogs.pick_person(self, self.ctx.db, title="Move to…", exclude={self.person_id},
                                     hint=f"Who is in the {plural(len(ids), 'selected face')}?")
        if target is not None:
            self.ctx.db.assign_faces(ids, target)
            self.faces.clear_selection()
            self.ctx.data_changed()

    def _set_cover(self):
        ids = self._face_ids()
        if len(ids) == 1:
            self.ctx.db.set_cover(self.person_id, ids[0])
            self.faces.clear_selection()
            self.ctx.data_changed()

    def _not_a_face(self):
        self.ctx.db.ignore_faces(self._face_ids())
        self.faces.clear_selection()
        self.ctx.data_changed()

    def _face_menu(self, tile, pos):
        m = QMenu(self)
        m.addAction(theme.icon("show"), "Open photo", lambda: self._open_face(tile))
        m.addSeparator()
        m.addAction(theme.icon("block"), self.not_btn.text(), self._not_person)
        m.addAction(theme.icon("person"), "Move to…", self._move_faces)
        a = m.addAction(theme.icon("star"), "Use as profile photo", self._set_cover)
        a.setEnabled(len(self.faces.selected) == 1)
        m.addAction(theme.icon("close"), "Not a face", self._not_a_face)
        m.exec(pos)

    def _accept_suggestions(self):
        ids = [t.key for t in self.suggest.selected_tiles()]
        self.ctx.db.assign_faces(ids, self.person_id)
        self.suggest.clear_selection()
        self.ctx.data_changed()

    def _reject_suggestions(self):
        ids = [t.key for t in self.suggest.selected_tiles()]
        self.ctx.db.reject_faces(ids, self.person_id)
        self.suggest.clear_selection()
        self.ctx.data_changed()
