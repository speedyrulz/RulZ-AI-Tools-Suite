"""Main window: sidebar navigation, pages, viewer and background scanning."""
from __future__ import annotations

import os
from urllib.parse import urlsplit

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from .. import config, search
from ..ai import settings as ai_settings
from ..ai.jobs import AIController, Job
from ..db import Database
from ..faces import FaceEngine
from . import dialogs, theme
from .ai_panel import AIJobDialog
from .common import plural
from .library_page import LibraryPage
from .people_page import PeoplePage
from .photos_page import PhotosPage
from .thumbs import ThumbLoader
from .viewer import ViewerPage
from .worker import ScanController

UI_DB_TIMEOUT = 5          # seconds a UI write waits for the scanner before saying the library is busy
REFRESH_MS = 350           # refresh delay after a change
SCAN_REFRESH_MS = 10_000   # at most this often while a scan is running
AI_REFRESH_MS = 3_000      # at most this often while AI results come in


class MainWindow(QMainWindow):
    PHOTOS, PEOPLE, LIBRARY = range(3)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        icon_file = config.resource_path("assets", "icon.ico")
        if icon_file.exists():
            self.setWindowIcon(QIcon(str(icon_file)))
        self.setMinimumSize(980, 640)
        self.db = Database(timeout=UI_DB_TIMEOUT)
        self.scanner = ScanController(self)
        self.ai = AIController(self)  # before the pages: they listen to it
        self._remote_ok: set[str] = set()  # AI servers on other computers the user agreed to send photos to
        self._ai_job: Job | None = None  # the last AI job shown in the sidebar
        self._engine: FaceEngine | None = None
        # Bumped whenever library data changes; pages use it to know when cached results are stale.
        self.data_version = 0
        self._closing = False
        self._scan_stats: dict | None = None  # library counts at the last refresh during a scan

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

        self._refresh_timer = QTimer(self, singleShot=True, interval=REFRESH_MS)
        self._refresh_timer.timeout.connect(self._scheduled_refresh)
        self.scanner.changed.connect(self._schedule_refresh)
        self.scanner.rewritten.connect(self._files_rewritten)
        self.scanner.progress.connect(self._progress)
        self.scanner.started.connect(self._scan_started)
        self.scanner.finished.connect(self._scan_finished)
        self._ai_timer = QTimer(self, singleShot=True, interval=AI_REFRESH_MS)
        self._ai_timer.timeout.connect(self._ai_refresh)
        self.ai.started.connect(self._ai_started)
        self.ai.progress.connect(self._ai_progress)
        self.ai.saved.connect(self._ai_saved)
        self.ai.finished.connect(self._ai_finished)

        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self.navigate(self.PHOTOS))
        QShortcut(QKeySequence("Ctrl+2"), self, lambda: self.navigate(self.PEOPLE))
        QShortcut(QKeySequence("Ctrl+3"), self, lambda: self.navigate(self.LIBRARY))
        QShortcut(QKeySequence("F5"), self, self.scanner.scan)
        QShortcut(QKeySequence("Ctrl+Shift+L"), self, lambda: self.set_theme(not theme.is_dark()))

        geo = config.settings().value("geometry")
        if geo is None and not os.environ.get("RULZ_DATA_DIR"):
            geo = QSettings("RulZ", config.APP_NAME).value("geometry")  # where v1.1 kept it (read only)
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
        self.logo = QLabel()
        self.logo.setContentsMargins(8, 2, 0, 0)
        self._update_logo()
        lay.addWidget(self.logo)
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

        # AI job status: separate from the scan's, both can run at once
        self.ai_box = QWidget()
        al = QVBoxLayout(self.ai_box)
        al.setContentsMargins(0, 0, 0, 8)
        al.setSpacing(4)
        self.ai_status = QLabel()
        self.ai_status.setWordWrap(True)
        self.ai_status.setObjectName("SideStatus")
        self.ai_status.setTextFormat(Qt.PlainText)  # messages quote model replies and server errors
        self.ai_progress = QProgressBar()
        self.ai_progress.setTextVisible(False)
        arow = QHBoxLayout()
        arow.setContentsMargins(0, 0, 0, 0)
        arow.setSpacing(2)
        self.ai_show_btn = self._side_button("Show results", self._ai_show_results)
        self.ai_stop_btn = self._side_button("Stop", self._ai_stop)
        self.ai_close_btn = self._side_button("Dismiss", self.ai_box.hide)
        for b in (self.ai_show_btn, self.ai_stop_btn, self.ai_close_btn):
            arow.addWidget(b)
        arow.addStretch(1)
        al.addWidget(self.ai_status)
        al.addWidget(self.ai_progress)
        al.addLayout(arow)
        self.ai_box.hide()
        lay.addWidget(self.ai_box)

        self.side_status = QLabel()
        self.side_status.setWordWrap(True)
        self.side_status.setObjectName("SideStatus")
        self.side_progress = QProgressBar()
        self.side_progress.setTextVisible(False)
        self.side_progress.hide()
        lay.addWidget(self.side_status)
        lay.addWidget(self.side_progress)
        lay.addSpacing(6)
        self.theme_btn = QPushButton()
        self.theme_btn.setObjectName("ThemeButton")
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        self.theme_btn.setToolTip("Switch between dark and light (Ctrl+Shift+L)")
        self.theme_btn.clicked.connect(lambda: self.set_theme(not theme.is_dark()))
        lay.addWidget(self.theme_btn)
        self._update_theme_button()
        return side

    @staticmethod
    def _side_button(text: str, slot) -> QPushButton:
        b = QPushButton(text)
        b.setObjectName("ThemeButton")  # the sidebar's quiet button look
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(slot)
        return b

    def _update_logo(self):
        pm = QPixmap(theme.logo_file())
        if not pm.isNull():
            dpr = 2
            pm = pm.scaledToWidth(168 * dpr, Qt.SmoothTransformation)
            pm.setDevicePixelRatio(dpr)
            self.logo.setPixmap(pm)
        else:
            self.logo.setText("RulZ")

    def _update_theme_button(self):
        dark = theme.is_dark()
        self.theme_btn.setText("  Light mode" if dark else "  Dark mode")
        self.theme_btn.setIcon(theme.icon("sun" if dark else "moon", theme.C["side_muted"], 16))

    # ---- theme -----------------------------------------------------------------
    def set_theme(self, dark: bool):
        """Switch Dark/Light right away and remember it."""
        theme.save_choice(dark)
        if dark == theme.is_dark():
            return
        theme.apply(QApplication.instance(), dark)
        theme.retheme_widgets(self)
        self._update_logo()
        self._update_theme_button()
        self._update_nav_icons()
        self.library.update_theme()
        for p in (self.photos, self.people, self.library):
            p.dirty = True
        if self.root.currentWidget() is self.viewer:
            self.viewer.retheme()
        else:
            self.pages.currentWidget().refresh()

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

    # ---- AI tagging and Look for -------------------------------------------------
    def _ai_ready(self) -> ai_settings.Connection | None:
        """The AI connection if it is set up; otherwise says what's missing and shows Library."""
        conn = ai_settings.resolve(ai_settings.load())
        if conn.problem:
            dialogs.info(self, "Set up AI first", conn.problem)
            self.navigate(self.LIBRARY)
            self.library.show_ai_card()
            return None
        return conn

    def run_ai(self, job: Job) -> bool:
        """Start an AI job after the checks: one job at a time, set up, and OK to send photos off this PC."""
        if self.ai.running:
            dialogs.info(self, "AI is busy", f"{self.ai.job.title()} is still running. Stop it in the sidebar, or "
                                             "wait until it's done.")
            return False
        conn = self._ai_ready()
        if conn is None:
            return False
        # Asked per server, so a server address that changes (in the Training Captioner too) is asked again.
        # A connection test sends only a red square, never a photo.
        hosts = sorted({urlsplit(ep["url"]).hostname or ep["url"] for ep in conn.endpoints
                        if not ai_settings.is_loopback(ep["url"])} - self._remote_ok)
        if hosts and job.kind != "test":
            if not dialogs.confirm(self, "Send photos to another computer?",
                                   f"The AI server ({', '.join(hosts)}) isn't on this PC. The photos you tag or "
                                   "search are sent to it over the network.\n\nRulZ asks once for each server "
                                   "each time it starts.", ok_text="Send photos", danger=True):
                return False
            self._remote_ok |= set(hosts)
        return self.ai.start(job)

    def ai_dialog(self, kind: str, *, current: int | None = None, selected: list[int] | None = None,
                  shown: list[int] | None = None, query: str = "", combine_with: str = "",
                  force: bool = False) -> None:
        """Ask what to look for (or confirm tagging) and for which photos, then start the job. A look for one
        thing over all photos or the photos shown, started from the Photos page, shows its matches there as
        they are found (combined with the current search when only the photos shown were checked)."""
        if self.ai.running:
            self.run_ai(Job(kind))  # says it's busy
            return
        if self._ai_ready() is None:
            return
        dlg = AIJobDialog(self, self, kind, current=current, selected=selected, shown=shown, query=query,
                          force=force)
        if dlg.exec() != QDialog.Accepted:
            return
        job = dlg.job()
        # Everything was checked before ("Show results"): no job, so no server start and nothing to send.
        if not dlg.nothing_to_do and not self.run_ai(job):
            return
        if kind == "look":
            for q in job.queries:  # now, so the search below knows them while the server is still starting
                self.db.ai_search_id(q, create=True)
            if dlg.nothing_to_do:
                self.data_changed(quiet=True)
        on_shown = shown is not None and job.photo_ids == shown
        # One photo or a selection keeps the search as it is: the sidebar's Show results shows the matches.
        if kind == "look" and len(job.queries) == 1 and self.root.currentWidget() is not self.viewer \
                and (job.photo_ids is None or on_shown):
            q = search.quote(job.queries[0])
            if combine_with and on_shown:
                q = f"{combine_with}, {q}"
            self.navigate(self.PHOTOS)
            self.photos.search.setText(q)

    def ai_settings_changed(self):
        self._remote_ok = set()  # another server: ask again before sending photos to it

    def _ai_started(self, job: Job):
        if job.kind == "test":
            return  # its result shows on the Library page
        self._ai_job = job
        self.ai_box.show()
        self.ai_status.setText(f"{job.title()}…")
        self.ai_progress.setRange(0, 0)
        self.ai_progress.show()
        self.ai_stop_btn.show()
        self.ai_show_btn.hide()
        self.ai_close_btn.hide()

    def _ai_progress(self, done: int, total: int, msg: str):
        if self.ai.job is None or self.ai.job.kind == "test":
            return
        if self.ai.stopping:
            msg = "Stopping… (waiting for the replies already asked for)"
        self.ai_status.setText(msg)
        if total:
            self.ai_progress.setRange(0, total)
            self.ai_progress.setValue(done)
        else:
            self.ai_progress.setRange(0, 0)

    def _ai_stop(self):
        self.ai.stop()
        self.ai_status.setText("Stopping… (waiting for the replies already asked for)")

    def _ai_saved(self):
        self.data_version += 1
        for p in (self.photos, self.people, self.library):
            p.dirty = True
        if not self._ai_timer.isActive():
            self._ai_timer.start()

    def _ai_refresh(self):
        """Show new AI results, at most every few seconds: only views they can change are rebuilt."""
        if self.root.currentWidget() is self.viewer:
            self.viewer.refresh_ai()
        elif self.pages.currentWidget() is self.photos and self.photos.search.text().strip():
            self.photos.refresh()
        elif self.pages.currentWidget() is self.library:
            self.library.ai_card.refresh()

    def _ai_finished(self, job: Job, msg: str, ok: bool):
        self._ai_timer.stop()
        self._ai_refresh()
        if job.kind == "test":
            return
        self.ai_box.show()
        self.ai_status.setText(msg)
        self.ai_progress.hide()
        self.ai_stop_btn.hide()
        self.ai_close_btn.show()
        self.ai_show_btn.setVisible(job.kind == "look" and bool(job.queries))

    def _ai_show_results(self):
        job = self._ai_job
        if job is None or not job.queries:
            return
        self.navigate(self.PHOTOS)
        self.photos.search.setText(search.quote(job.queries[0]))

    # ---- refresh -------------------------------------------------------------
    def data_changed(self, quiet: bool = False):
        """Called after the user edits something. ``quiet`` defers refresh (e.g. while in the viewer)."""
        self.data_version += 1
        for p in (self.photos, self.people, self.library):
            p.dirty = True
        if not quiet:
            self._refresh_visible()

    def reload_after_conflict(self):
        """An edit failed because a scan or regroup changed what it referred to: show the current state,
        in the viewer too (pages are refreshed when it closes)."""
        self.data_changed()
        if self.root.currentWidget() is self.viewer and self.viewer.ids:
            self.viewer._load_faces()

    def _schedule_refresh(self):
        self.data_version += 1  # the scanner changed something
        ThumbLoader.instance().failed.clear()  # thumbnails that didn't exist yet may exist now
        # Rebuilding the timeline is O(photos): while a scan runs, do it every few seconds at most.
        # The first refresh of a scan stays quick so newly found photos show up right away.
        interval = SCAN_REFRESH_MS if self.scanner.running and self._scan_stats is not None else REFRESH_MS
        if not self._refresh_timer.isActive() or self._refresh_timer.remainingTime() > interval:
            self._refresh_timer.start(interval)

    @staticmethod
    def _files_rewritten(photo_ids: list, face_ids: list):
        """The scanner wrote new thumbnails / face crops: cached copies under the same name are stale (an
        edited or rotated photo keeps its id, and a new face can get a deleted face's id)."""
        loader = ThumbLoader.instance()
        for pid in photo_ids:
            loader.invalidate(config.thumb_file(pid))
        for fid in face_ids:
            loader.invalidate(config.face_crop_file(fid))

    def _scheduled_refresh(self):
        if self.scanner.running:
            s = self.db.stats()
            if s == self._scan_stats:
                return  # e.g. only expressions were filled in: nothing visible changed
            self._scan_stats = s
        else:
            self._scan_stats = None
        self._refresh_visible()

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
        self._scan_stats = None
        if self._closing:
            return  # a confirmed folder removal still runs; keep saying "Finishing…"
        self.side_progress.show()
        self.side_progress.setRange(0, 0)
        self.side_status.setText("Scanning…")

    def _progress(self, done: int, total: int, msg: str):
        if self._closing:
            return
        self.side_status.setText(msg)
        if total:
            self.side_progress.setRange(0, total)
            self.side_progress.setValue(done)
        else:
            self.side_progress.setRange(0, 0)

    def _scan_finished(self, msg: str):
        if self._closing:
            self.close()  # the scan has wound down: now the window can really close
            return
        self.side_progress.hide()
        self._schedule_refresh()

    def closeEvent(self, e):
        self.library.flush_settings()
        # Stop AI work and the llama-server RulZ started (its threads end with the app; nothing to wait for).
        self.ai.shutdown()
        if self.scanner.running:
            # Destroying a running QThread crashes, and cutting a scan off mid-write helps no one: ask it to
            # stop and close once it has (usually within a second or two).
            if not self._closing:
                self._closing = True
                config.settings().setValue("geometry", self.saveGeometry())
                self.scanner.stop()
                self.setWindowTitle(f"{config.APP_NAME} – Finishing…")
                self.side_progress.setRange(0, 0)
                self.side_progress.show()
                self.side_status.setText("Finishing… RulZ closes as soon as the current step is saved.")
                self.setEnabled(False)
            e.ignore()
            return
        if not self._closing:
            config.settings().setValue("geometry", self.saveGeometry())
        super().closeEvent(e)
