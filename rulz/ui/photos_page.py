"""The main timeline: every photo, grouped by month, searchable by person, expression and direction."""
from __future__ import annotations

from PySide6.QtCore import QStringListModel, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QCompleter, QHBoxLayout, QLineEdit, QMenu, QPushButton, QVBoxLayout, QWidget

from .. import attributes
from ..search import search
from . import theme
from .common import SelectionBar, button, label, month_sections, open_default, plural, show_in_explorer
from .tilegrid import TileGrid

# (chip label, search phrase from attributes.SEARCH_TERMS)
EXPRESSION_CHIPS = [("Smiling", "smiling"), ("Neutral", "neutral"), ("Surprised", "surprised"), ("Sad", "sad"),
                    ("Angry", "angry"), ("Scared", "scared")]
DIRECTION_CHIPS = [("Facing camera", "facing camera"), ("Facing left", "facing left"),
                   ("Facing right", "facing right"), ("Looking up", "looking up"), ("Looking down", "looking down"),
                   ("Side profile", "side profile"), ("Head tilted", "head tilted")]
POSE_CHIPS = [("Standing", "standing"), ("Sitting", "sitting"), ("Lying down", "lying down"),
              ("Arms raised", "arms raised")]


class PhotosPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName("Page")
        self.dirty = True
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        top = QHBoxLayout()
        top.setContentsMargins(28, 22, 28, 10)
        top.setSpacing(12)
        title = label("Photos", "PageTitle")
        self.count = label("", "Muted")
        top.addWidget(title)
        top.addWidget(self.count, 0, Qt.AlignBottom)
        top.addStretch(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search people & expressions, e.g. “Mom smiling, Dad”")
        self.search.setToolTip("Words in one group describe one face; separate groups with commas.\n"
                               "Examples:  smiling   ·   Mom facing left   ·   Mom smiling, Dad   ·   looking down")
        self.search.setClearButtonEnabled(True)
        search_icon = self.search.addAction(theme.icon("search", theme.C["muted"], 16), QLineEdit.LeadingPosition)
        search_icon.setProperty("_rulzThemeIcon", "search|muted|16")
        self.search.setMinimumWidth(380)
        self.names = QStringListModel()
        comp = QCompleter(self.names, self)
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        comp.setFilterMode(Qt.MatchContains)
        self.search.setCompleter(comp)
        top.addWidget(self.search)
        self.filter_btn = button("Filters", "filter")
        self.filter_btn.setCheckable(True)
        self.filter_btn.toggled.connect(self._show_filters)
        top.addWidget(self.filter_btn)
        lay.addLayout(top)

        self.filters = QWidget()
        fl = QVBoxLayout(self.filters)
        fl.setContentsMargins(28, 0, 28, 10)
        fl.setSpacing(6)
        self.chips: dict[str, list[QPushButton]] = {}
        for title, chips in (("Expression", EXPRESSION_CHIPS), ("Direction", DIRECTION_CHIPS), ("Pose", POSE_CHIPS)):
            row = QHBoxLayout()
            row.setSpacing(6)
            lb = label(title, "Muted")
            lb.setFixedWidth(78)
            row.addWidget(lb)
            group: list[QPushButton] = []
            for text, phrase in chips:
                b = QPushButton(text)
                b.setObjectName("Chip")
                b.setCheckable(True)
                b.setCursor(Qt.PointingHandCursor)
                b.setProperty("phrase", phrase)
                if "left" in phrase or "right" in phrase:
                    b.setToolTip(f"Head turned towards the {'left' if 'left' in phrase else 'right'} side of the photo")
                b.clicked.connect(lambda checked, b=b, g=group: self._chip_clicked(b, g))
                row.addWidget(b)
                group.append(b)
            row.addStretch(1)
            if title == "Expression":
                clear = button("Clear filters", flat=True)
                clear.clicked.connect(self._clear_chips)
                row.addWidget(clear)
            self.chips[title] = group
            fl.addLayout(row)
        hint = label("Filters apply to the people you search for (or anyone, if the search is empty). "
                     "All of these are estimated automatically and won't always be right. Pose needs the "
                     "person's legs in the photo.",
                     "Muted", wrap=True)
        hint.setStyleSheet("font-size: 12px;")
        fl.addWidget(hint)
        self.filters.hide()
        lay.addWidget(self.filters)

        self.sel_bar = SelectionBar()
        self.sel_bar.add(button("Show in folder", "open_folder")).clicked.connect(self._show_selected)
        self.sel_bar.add(button("Copy paths", "tag")).clicked.connect(self._copy_paths)
        bar_wrap = QHBoxLayout()
        bar_wrap.setContentsMargins(24, 0, 24, 6)
        bar_wrap.addWidget(self.sel_bar)
        lay.addLayout(bar_wrap)

        self.grid = TileGrid(tile_size=190, spacing=4)
        self.grid.activated.connect(self._open)
        self.grid.contextMenuRequested.connect(self._menu)
        self.grid.selectionChanged.connect(lambda: self.sel_bar.update_count(len(self.grid.selected)))
        self.sel_bar.cleared.connect(self.grid.clear_selection)
        lay.addWidget(self.grid, 1)

        self._search_timer = QTimer(self, singleShot=True, interval=250)
        self._search_timer.timeout.connect(self.refresh)
        self.search.textChanged.connect(lambda: self._search_timer.start())

    # ---- filters -------------------------------------------------------------
    def _show_filters(self, on: bool):
        self.filters.setVisible(on or bool(self._active_phrases()))

    def _chip_clicked(self, chip: QPushButton, group: list[QPushButton]):
        for other in group:  # one choice per row; clicking the active chip turns it off
            if other is not chip:
                other.setChecked(False)
        self.refresh()

    def _clear_chips(self):
        for group in self.chips.values():
            for b in group:
                b.setChecked(False)
        self.refresh()

    def _active_phrases(self) -> list[str]:
        return [b.property("phrase") for g in self.chips.values() for b in g if b.isChecked()]

    # ---- data ----------------------------------------------------------------
    def refresh(self):
        self.dirty = False
        db = self.ctx.db
        self.names.setStringList(sorted({p.name for p in db.people(include_hidden=True) if p.name}, key=str.lower))
        photos = db.photos()
        text = self.search.text().strip()
        phrases = self._active_phrases()
        self.filter_btn.setText(f"Filters ({len(phrases)})" if phrases else "Filters")
        result = search(db, text, [attributes.SEARCH_TERMS[p] for p in phrases])
        wanted = result.photo_ids
        if wanted is not None:
            photos = [p for p in photos if p.id in wanted]
        self.grid.set_sections(month_sections(photos), keep_scroll=wanted is None)
        self.count.setText(plural(len(photos), "photo") if photos else "")
        if not photos:
            if result.problem:
                self.grid.set_empty("No matching photos", result.problem)
            elif wanted is not None:
                self.grid.set_empty("No matching photos",
                                    "No faces match this search. Try fewer filters, or check that the people "
                                    "you're looking for are named on the People page.")
            elif not db.folders():
                self.grid.set_empty("Your library is empty",
                                    "Go to Library and add a folder with photos. RulZ will find them and "
                                    "organize everyone in them automatically.")
            else:
                self.grid.set_empty("Looking for photos…", "Photos will appear here as soon as they're found.")

    def _photo_ids(self) -> list[int]:
        return [t.key for t in self.grid.flat]

    def _open(self, tile):
        ids = self._photo_ids()
        self.ctx.open_viewer(ids, ids.index(tile.key))

    def _menu(self, tile, pos):
        m = QMenu(self)
        m.addAction(theme.icon("show"), "Open", lambda: self._open(tile))
        m.addAction(theme.icon("open_folder"), "Show in folder", lambda: show_in_explorer(tile.data.path))
        m.addAction(theme.icon("photos"), "Open with default app", lambda: open_default(tile.data.path))
        m.addSeparator()
        m.addAction(theme.icon("tag"), "Copy path", lambda: QGuiApplication.clipboard().setText(tile.data.path))
        m.exec(pos)

    def _selected_paths(self) -> list[str]:
        return [t.data.path for t in self.grid.selected_tiles()]

    def _show_selected(self):
        paths = self._selected_paths()
        if paths:
            show_in_explorer(paths[0])

    def _copy_paths(self):
        QGuiApplication.clipboard().setText("\n".join(self._selected_paths()))
