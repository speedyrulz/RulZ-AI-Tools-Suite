"""Library settings: watched folders, scanning and face-grouping sensitivity."""
from __future__ import annotations

import os

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QListWidget, QListWidgetItem, QProgressBar,
                               QScrollArea, QSlider, QVBoxLayout, QWidget)

from .. import config
from . import dialogs, theme
from .common import Segmented, button, label, plural


class Card(QFrame):
    def __init__(self, title: str, subtitle: str = ""):
        super().__init__()
        self.setObjectName("Card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(22, 18, 22, 20)
        self.lay.setSpacing(10)
        self.lay.addWidget(label(title, "H2"))
        if subtitle:
            self.lay.addWidget(label(subtitle, "Muted", wrap=True))


class LibraryPage(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.dirty = True
        self.setObjectName("Page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        body.setObjectName("Page")
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(28, 22, 28, 28)
        lay.setSpacing(16)
        lay.addWidget(label("Library", "PageTitle"))
        lay.addWidget(label("RulZ watches these folders (and everything inside them) for photos. Your files are "
                            "never moved, renamed or changed.", "Muted", wrap=True))

        # folders ------------------------------------------------------------
        folders = Card("Folders")
        self.list = QListWidget()
        self.list.setIconSize(QSize(22, 22))
        self.list.setMinimumHeight(170)
        self.list.itemSelectionChanged.connect(self._sel)
        self.list.itemDoubleClicked.connect(lambda it: os.startfile(it.data(Qt.UserRole + 1)))
        folders.lay.addWidget(self.list)
        row = QHBoxLayout()
        self.add_btn = button("Add folder…", "add", primary=True)
        self.add_btn.clicked.connect(self.add_folder)
        self.remove_btn = button("Remove from library", "delete")
        self.remove_btn.clicked.connect(self._remove)
        self.open_btn = button("Open", "open_folder")
        self.open_btn.clicked.connect(lambda: os.startfile(self.list.currentItem().data(Qt.UserRole + 1)))
        row.addWidget(self.add_btn)
        row.addWidget(self.remove_btn)
        row.addWidget(self.open_btn)
        row.addStretch(1)
        folders.lay.addLayout(row)
        lay.addWidget(folders)

        # scanning -----------------------------------------------------------
        scan = Card("Scanning", "RulZ checks for new, changed and deleted photos every time it starts. "
                                "Scan now after copying in new photos.")
        self.status = label("", wrap=True)
        scan.lay.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        scan.lay.addWidget(self.progress)
        row = QHBoxLayout()
        self.scan_btn = button("Scan now", "refresh")
        self.scan_btn.clicked.connect(ctx.scanner.scan)
        self.stop_btn = button("Pause", "stop")
        self.stop_btn.clicked.connect(ctx.scanner.stop)
        row.addWidget(self.scan_btn)
        row.addWidget(self.stop_btn)
        row.addStretch(1)
        scan.lay.addLayout(row)
        self.stats = label("", "Muted")
        scan.lay.addWidget(self.stats)
        lay.addWidget(scan)

        # grouping -------------------------------------------------------------
        grp = Card("Face grouping",
                   "How similar two faces must be to count as the same person. If different people end up in "
                   "one group, move it towards Stricter. If the same person is split into many groups, move it "
                   "towards Looser. Your names and manual corrections are always kept.")
        srow = QHBoxLayout()
        srow.addWidget(label("Looser", "Muted"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(30, 55)
        self.slider.setPageStep(2)
        self.slider.valueChanged.connect(self._slider_moved)
        # Saved once the slider settles (or is released), not on every tick: each save is a database write.
        self._threshold_dirty = False
        self._save_timer = QTimer(self, singleShot=True, interval=500)
        self._save_timer.timeout.connect(self.flush_settings)
        self.slider.sliderReleased.connect(self.flush_settings)
        srow.addWidget(self.slider, 1)
        srow.addWidget(label("Stricter", "Muted"))
        grp.lay.addLayout(srow)
        self.slider_value = label("", "Muted")
        grp.lay.addWidget(self.slider_value)
        row = QHBoxLayout()
        self.regroup_btn = button("Apply and regroup", "refresh")
        self.regroup_btn.setToolTip("Rebuilds the unnamed groups. Named people and your corrections are kept.")
        self.regroup_btn.clicked.connect(self._regroup)
        reset = button("Reset to default", flat=True)
        reset.clicked.connect(lambda: self.slider.setValue(round(config.DEFAULT_MATCH_THRESHOLD * 100)))
        row.addWidget(self.regroup_btn)
        row.addWidget(reset)
        row.addStretch(1)
        grp.lay.addLayout(row)
        lay.addWidget(grp)

        # appearance -----------------------------------------------------------
        look = Card("Appearance", "Dark is the default. The sun/moon button at the bottom of the sidebar "
                                  "(or Ctrl+Shift+L) switches too.")
        self.theme_seg = Segmented(["Dark", "Light"])
        self.theme_seg.changed.connect(lambda i: ctx.set_theme(i == 0))
        look.lay.addWidget(self.theme_seg)
        lay.addWidget(look)
        self.update_theme()

        about = Card("About")
        about.lay.addWidget(label(f"{config.APP_NAME} {config.APP_VERSION}. Everything runs on this PC – no photos "
                                  "or faces ever leave your computer.", "Muted", wrap=True))
        data_row = QHBoxLayout()
        data_row.addWidget(label(f"Library data: {config.data_dir()}", "Muted"), 1)
        ob = button("Open", "open_folder", flat=True)
        ob.clicked.connect(lambda: os.startfile(config.data_dir()))
        data_row.addWidget(ob)
        about.lay.addLayout(data_row)
        lay.addWidget(about)
        lay.addStretch(1)

        ctx.scanner.progress.connect(self._progress)
        ctx.scanner.started.connect(self._scan_state)
        ctx.scanner.finished.connect(self._finished)
        self._scan_state()

    def update_theme(self):
        self.theme_seg.set_current(0 if theme.is_dark() else 1)

    # ---- refresh -------------------------------------------------------------
    def refresh(self):
        self.dirty = False
        db = self.ctx.db
        cur = self.list.currentItem().data(Qt.UserRole) if self.list.currentItem() else None
        self.list.clear()
        for f in db.folders():
            exists = os.path.isdir(f["path"])
            it = QListWidgetItem(theme.icon("folder", theme.C["accent"] if exists else theme.C["muted"], 22),
                                 f"{f['path']}\n{plural(f['n'], 'photo')}" + ("" if exists else " · folder not found"))
            it.setData(Qt.UserRole, f["id"])
            it.setData(Qt.UserRole + 1, f["path"])
            self.list.addItem(it)
            if f["id"] == cur:
                self.list.setCurrentItem(it)
        if self.list.count() == 0:
            it = QListWidgetItem("No folders yet. Click “Add folder…” to choose where your photos are.")
            it.setFlags(Qt.NoItemFlags)
            self.list.addItem(it)
        self._sel()
        self.flush_settings()  # don't let the refresh below undo a change that hasn't been saved yet
        s = db.stats()
        self.stats.setText(f"{plural(s['photos'], 'photo')} · {plural(s['faces'], 'face')} · "
                           f"{plural(s['people'], 'person')} ({s['named']} named)")
        self.slider.blockSignals(True)
        self.slider.setValue(round(db.match_threshold() * 100))
        self.slider.blockSignals(False)
        self._slider_moved(self.slider.value(), save=False)
        if not self.ctx.scanner.running:
            self.status.setText(self.ctx.scanner.last_message or "Up to date.")

    def _sel(self):
        it = self.list.currentItem()
        ok = it is not None and bool(it.flags() & Qt.ItemIsEnabled)
        self.remove_btn.setEnabled(ok)
        self.open_btn.setEnabled(ok)

    # ---- actions -------------------------------------------------------------
    def add_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Add a folder with photos", os.path.expanduser("~\\Pictures"))
        if not path:
            return
        err = self.ctx.db.add_folder(path)
        if err:
            dialogs.info(self, "Already added", err)
            return
        self.refresh()
        self.ctx.scanner.scan()

    def _remove(self):
        it = self.list.currentItem()
        if it is None:
            return
        path = it.data(Qt.UserRole + 1)
        if dialogs.confirm(self, "Remove folder", f"Stop showing photos from\n{path}?\n\nThe files on your PC "
                           "are not deleted. Face tags for these photos are removed from RulZ.",
                           ok_text="Remove", danger=True):
            self.ctx.scanner.stop()
            self.ctx.scanner.wait()
            self.ctx.db.remove_folder(it.data(Qt.UserRole))
            self.ctx.data_changed()

    def _slider_moved(self, v: int, save: bool = True):
        t = v / 100
        default = round(config.DEFAULT_MATCH_THRESHOLD * 100)
        self.slider_value.setText(f"Match threshold: {t:.2f}" + ("  (default)" if v == default else ""))
        if save:
            self._threshold_dirty = True
            self._save_timer.start()

    def flush_settings(self):
        """Save a threshold change that is still waiting for the slider to settle."""
        if self._threshold_dirty:
            self._threshold_dirty = False
            self._save_timer.stop()
            self.ctx.db.set_setting("match_threshold", f"{self.slider.value() / 100:.2f}")

    def _regroup(self):
        self.flush_settings()
        self.ctx.scanner.regroup(self.slider.value() / 100)

    # ---- scan status -----------------------------------------------------------
    def _scan_state(self):
        running = self.ctx.scanner.running
        self.scan_btn.setEnabled(not running)
        self.stop_btn.setVisible(running)
        self.regroup_btn.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.progress.setRange(0, 0)

    def _progress(self, done: int, total: int, msg: str):
        self.status.setText(msg)
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
        else:
            self.progress.setRange(0, 0)

    def _finished(self, msg: str):
        self._scan_state()
        self.status.setText(msg)
        self.refresh()
