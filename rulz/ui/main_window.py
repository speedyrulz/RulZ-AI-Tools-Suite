"""Main window: sidebar navigation, pages, viewer and background scanning."""
from __future__ import annotations

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

from .. import config
from ..db import Database
from ..faces import FaceEngine
from . import theme
from .common import plural
from .library_page import LibraryPage
from .people_page import PeoplePage
from .photos_page import PhotosPage
from .thumbs import ThumbLoader
from .viewer import ViewerPage
from .worker import ScanController


class MainWindow(QMainWindow):
    PHOTOS, PEOPLE, LIBRARY = range(3)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        icon_file = config.resource_path("assets", "icon.ico")
        if icon_file.exists():
            self.setWindowIcon(QIcon(str(icon_file)))
        self.setMinimumSize(980, 640)
        self.db = Database()
        self.scanner = ScanController(self)
        self._engine: FaceEngine | None = None
        # Bumped whenever library data changes; pages use it to know when cached results are stale.
        self.data_version = 0

        self.root = QStackedWidget()
        self.setCentralWidget(self.root)
        shell = QWidget()
        shell_lay = QHBoxLayout(shell)
        shell_lay.setContentsMargins(0, 0, 0, 0)
        shell_lay.setSpacing(0)
        shell_lay.addWidget(self._build_sidebar())
        self.pages = QStackedWidget()
        self.pages.setObjectName("Content")
        self.photos = PhotosPage(self)
        self.people = PeoplePage(self)
        self.library = LibraryPage(self)
        for p in (self.photos, self.people, self.library):
            self.pages.addWidget(p)
        shell_lay.addWidget(self.pages, 1)
        self.root.addWidget(shell)
        self.viewer = ViewerPage(self)
        self.viewer.closed.connect(self._viewer_closed)
        self.root.addWidget(self.viewer)

        self._refresh_timer = QTimer(self, singleShot=True, interval=350)
        self._refresh_timer.timeout.connect(self._refresh_visible)
        self.scanner.changed.connect(self._schedule_refresh)
        self.scanner.progress.connect(self._progress)
        self.scanner.started.connect(self._scan_started)
        self.scanner.finished.connect(self._scan_finished)

        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self.navigate(self.PHOTOS))
        QShortcut(QKeySequence("Ctrl+2"), self, lambda: self.navigate(self.PEOPLE))
        QShortcut(QKeySequence("Ctrl+3"), self, lambda: self.navigate(self.LIBRARY))
        QShortcut(QKeySequence("F5"), self, self.scanner.scan)

        settings = QSettings("RulZ", config.APP_NAME)
        geo = settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1360, 860)

        self._refresh_visible()
        if self.db.folders():
            self.navigate(self.PHOTOS)
            QTimer.singleShot(300, self.scanner.scan)
        else:
            self.navigate(self.LIBRARY)

    # ---- sidebar -----------------------------------------------------------
    def _build_sidebar(self) -> QWidget:
        side = QFrame()
        side.setObjectName("Sidebar")
        side.setFixedWidth(232)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(12, 18, 12, 16)
        lay.setSpacing(4)
        logo = QLabel()
        logo.setContentsMargins(8, 2, 0, 0)
        png = config.resource_path("assets", "logo_full_white.png")  # sidebar is always black
        pm = QPixmap(str(png))
        if not pm.isNull():
            dpr = 2
            pm = pm.scaledToWidth(168 * dpr, Qt.SmoothTransformation)
            pm.setDevicePixelRatio(dpr)
            logo.setPixmap(pm)
        else:
            logo.setText("RulZ")
        lay.addWidget(logo)
        sub = QLabel("IMAGE ORGANIZER")
        sub.setObjectName("BrandSub")
        sub.setContentsMargins(10, 4, 0, 0)
        lay.addWidget(sub)
        lay.addSpacing(22)

        self.nav = QButtonGroup(self)
        self.nav_buttons = []
        for i, (text, glyph) in enumerate((("Photos", "photos"), ("People", "people"), ("Library", "library"))):
            btn = QPushButton(f"  {text}")
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setIconSize(btn.iconSize() * 1.1)
            btn.setProperty("glyph", glyph)
            self.nav.addButton(btn, i)
            lay.addWidget(btn)
            self.nav_buttons.append(btn)
        self.nav.idClicked.connect(self.navigate)
        lay.addStretch(1)

        self.side_status = QLabel()
        self.side_status.setWordWrap(True)
        self.side_status.setObjectName("SideStatus")
        self.side_progress = QProgressBar()
        self.side_progress.setTextVisible(False)
        self.side_progress.hide()
        lay.addWidget(self.side_status)
        lay.addWidget(self.side_progress)
        return side

    def _update_nav_icons(self):
        for i, btn in enumerate(self.nav_buttons):
            color = theme.C["side_select_text"] if btn.isChecked() else theme.C["side_text"]
            btn.setIcon(theme.icon(btn.property("glyph"), color, 20))

    # ---- navigation ----------------------------------------------------------
    def navigate(self, index: int):
        if self.root.currentWidget() is self.viewer:
            self.viewer.close_viewer()
        self.nav_buttons[index].setChecked(True)
        self._update_nav_icons()
        page = self.pages.widget(index)
        if index == self.PEOPLE and self.pages.currentWidget() is self.people:
            self.people.back()  # clicking People again returns to the overview
        self.pages.setCurrentWidget(page)
        if page.dirty or index == self.LIBRARY:
            page.refresh()

    def open_viewer(self, photo_ids: list[int], index: int, highlight_face: int | None = None):
        if not photo_ids:
            return
        self.root.setCurrentWidget(self.viewer)
        self.viewer.open(photo_ids, index, highlight_face)

    def _viewer_closed(self):
        self.root.setCurrentIndex(0)
        page = self.pages.currentWidget()
        if page.dirty:
            page.refresh()

    def open_person(self, person_id: int):
        self.navigate(self.PEOPLE)
        self.people.open_person(person_id)

    def search_photos(self, name: str | None):
        if not name:
            return
        self.navigate(self.PHOTOS)
        self.photos.search.setText(name)

    def face_engine(self) -> FaceEngine:
        if self._engine is None:
            self._engine = FaceEngine()
        return self._engine

    # ---- refresh -------------------------------------------------------------
    def data_changed(self, quiet: bool = False):
        """Called after the user edits something. ``quiet`` defers refresh (e.g. while in the viewer)."""
        self.data_version += 1
        for p in (self.photos, self.people, self.library):
            p.dirty = True
        if not quiet:
            self._refresh_visible()

    def _schedule_refresh(self):
        self.data_version += 1  # the scanner changed something
        ThumbLoader.instance().failed.clear()  # thumbnails that didn't exist yet may exist now
        self._refresh_timer.start()

    def _refresh_visible(self):
        for p in (self.photos, self.people, self.library):
            p.dirty = True
        if not self.scanner.running:
            s = self.db.stats()
            self.side_status.setText(f"{plural(s['photos'], 'photo')} · {plural(s['people'], 'person')}")
        if self.root.currentWidget() is self.viewer:
            return
        self.pages.currentWidget().refresh()

    # ---- scan status -----------------------------------------------------------
    def _scan_started(self):
        self.side_progress.show()
        self.side_progress.setRange(0, 0)
        self.side_status.setText("Scanning…")

    def _progress(self, done: int, total: int, msg: str):
        self.side_status.setText(msg)
        if total:
            self.side_progress.setRange(0, total)
            self.side_progress.setValue(done)
        else:
            self.side_progress.setRange(0, 0)

    def _scan_finished(self, msg: str):
        self.side_progress.hide()
        self._schedule_refresh()

    def closeEvent(self, e):
        QSettings("RulZ", config.APP_NAME).setValue("geometry", self.saveGeometry())
        self.scanner.wait(8000)
        super().closeEvent(e)
