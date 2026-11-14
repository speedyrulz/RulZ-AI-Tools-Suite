import os
import subprocess
import threading

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QAbstractItemDelegate, QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QFormLayout,
                               QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSlider, QToolBar, QVBoxLayout,
                               QWidget, QHBoxLayout)

from . import config, theme
from .backend import CaptionEngine, ServerManager, StartCancelled, external_endpoints
from .gridview import CaptionEditor, ImageGrid
from .model import (BUSY, ERROR, IDLE, QUEUED, FilterProxy, ImageModel, caption_clashes, caption_path,
                    collect_images, load_entries, read_caption, unique_new)
from .settings_dialog import SettingsDialog

MANAGED_KEYS = ("llama_server", "model", "mmproj", "gpus", "gpu_mode", "tensor_split", "ctx_size", "ngl",
                "slots_per_server", "extra_args", "base_port", "disable_thinking", "thinking_budget")
SEARCH_DELAY_MS = 200


class ServerBridge(QObject):
    log = Signal(str)
    ready = Signal(list)
    error = Signal(str)
    cancelled = Signal()


class AddBridge(QObject):
    progress = Signal(int)   # images read so far
    done = Signal(object)    # (generation, entries, unreadable folders, error or "")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rulz Training Captioner")
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        self.s = config.load_settings()
        self.prompts = config.load_prompts()
        self.model = ImageModel()
        self.proxy = FilterProxy()
        self.proxy.setSourceModel(self.model)
        self.servers = ServerManager()
        self.server_sig = None
        self.engine = CaptionEngine()
        self.bridge = ServerBridge()
        self.adder = AddBridge()
        self.adding = 0    # folder scans running in the background
        self.add_gen = 0   # bumped by Clear List, so a scan that finishes afterwards adds nothing
        self.starting = False
        self.pending_job = None
        self.failures = 0
        self.warnings = 0
        self.run_managed = False  # the current job runs on the app's own llama-servers
        self.snapshots = {}  # path -> caption when it was queued, to spot edits made meanwhile

        self.grid = ImageGrid()
        self.grid.setModel(self.proxy)
        self.grid.set_thumb_size(self.s["thumb_size"])
        self.grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._context_menu)
        self.grid.error.connect(lambda m: QMessageBox.warning(self, "Save failed", m))
        # after a fast scroll, don't decode every thumbnail scrolled past before the visible ones
        self.grid.verticalScrollBar().valueChanged.connect(lambda _: self.model.drop_queued_thumbs())
        self.setCentralWidget(self.grid)

        self._build_actions()
        self._build_toolbar()
        self._build_prompt_dock()
        self._build_log_dock()
        self._build_status()

        self.engine.started.connect(lambda p: self.model.set_status(p, BUSY))
        self.engine.requeued.connect(lambda p: self.model.set_status(p, QUEUED))
        self.engine.captioned.connect(self._on_captioned)
        self.engine.failed.connect(self._on_failed)
        self.engine.warning.connect(self._on_warning)
        self.engine.progress.connect(self._on_progress)
        self.engine.finished.connect(self._on_finished)
        self.bridge.log.connect(self._log)
        self.bridge.ready.connect(self._servers_ready)
        self.bridge.error.connect(self._servers_error)
        self.bridge.cancelled.connect(self._servers_cancelled)
        self.adder.progress.connect(lambda n: self.statusBar().showMessage(f"Adding images… {n:,} read"))
        self.adder.done.connect(self._added)
        self.model.stats_changed.connect(self._update_stats)
        self._update_stats()

    # ------------------------------------------------------------------ UI
    def _act(self, text, slot, shortcut=None):
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        return a

    def _build_actions(self):
        self.a_add = self._act("Add Images…", self.add_images, "Ctrl+O")
        self.a_folder = self._act("Add Folder…", lambda: self.add_folder(False), "Ctrl+Shift+O")
        self.a_folder_r = self._act("Add Folder (with subfolders)…", lambda: self.add_folder(True))
        self.a_remove = self._act("Remove Selected", self.remove_selected, "Del")
        self.a_clear = self._act("Clear List", self.clear_list)
        self.a_cap_all = self._act("Caption All", lambda: self.caption("all"), "Ctrl+Return")
        self.a_cap_sel = self._act("Caption Selected", lambda: self.caption("selected"), "Ctrl+Shift+Return")
        self.a_stop = self._act("Stop", self.stop_caption, "Esc")
        self.a_start_srv = self._act("Start Servers", self.start_servers_only)
        self.a_stop_srv = self._act("Stop Servers", self.stop_servers)
        self.a_settings = self._act("Settings…", self.open_settings, "Ctrl+,")
        self.a_stop.setEnabled(False)

        m = self.menuBar()
        f = m.addMenu("&File")
        for a in (self.a_add, self.a_folder, self.a_folder_r, None, self.a_remove, self.a_clear, None,
                  self._act("Exit", self.close)):
            f.addSeparator() if a is None else f.addAction(a)
        c = m.addMenu("&Captions")
        for a in (self.a_cap_all, self.a_cap_sel, self.a_stop):
            c.addAction(a)
        sv = m.addMenu("&Server")
        for a in (self.a_start_srv, self.a_stop_srv, self.a_settings):
            sv.addAction(a)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.brand = brand = QLabel()
        brand.setPixmap(theme.logo(26))
        brand.setContentsMargins(4, 0, 14, 0)
        brand.setToolTip("Rulz Training Captioner")
        tb.addWidget(brand)
        for a in (self.a_add, self.a_folder, self.a_remove):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.a_cap_all, self.a_cap_sel, self.a_stop):
            tb.addAction(a)
        tb.widgetForAction(self.a_cap_all).setObjectName("primary")
        tb.addSeparator()
        tb.addAction(self.a_settings)
        tb.addSeparator()
        tb.addWidget(QLabel(" Show: "))
        self.filter_box = QComboBox()
        self.filter_box.addItems(["All", "Missing captions", "Captioned"])
        self.filter_box.currentIndexChanged.connect(
            lambda i: self.proxy.set_filter(mode=("all", "missing", "captioned")[i]) or self._update_stats())
        tb.addWidget(self.filter_box)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search captions / names")
        self.search.setClearButtonEnabled(True)
        self.search.setMaximumWidth(220)
        # filter once typing pauses: each pass checks every image (tens of ms at 20,000, more when
        # many cards hide or show), which made typing a word stutter
        self.search_timer = QTimer(self, singleShot=True, interval=SEARCH_DELAY_MS)
        self.search_timer.timeout.connect(self._apply_search)
        self.search.textChanged.connect(lambda _: self.search_timer.start())
        tb.addWidget(self.search)
        tb.addWidget(QLabel("  Thumbnails: "))
        sl = QSlider(Qt.Horizontal)
        sl.setRange(96, 256)
        sl.setValue(self.s["thumb_size"])
        sl.setFixedWidth(130)
        sl.valueChanged.connect(self._thumb_size)
        tb.addWidget(sl)

    def _build_prompt_dock(self):
        dock = QDockWidget("Captioning Prompt", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        w = QWidget()
        v = QVBoxLayout(w)
        self.prompt_combo = QComboBox()
        self.prompt_combo.currentIndexChanged.connect(self._prompt_selected)
        v.addWidget(self._section("SAVED PROMPTS"))
        v.addWidget(self.prompt_combo)
        v.addWidget(self._section("SYSTEM PROMPT"))
        self.sys_edit = QPlainTextEdit()
        v.addWidget(self.sys_edit, 3)
        v.addWidget(self._section("USER MESSAGE (SENT WITH EACH IMAGE)"))
        self.user_edit = QPlainTextEdit()
        v.addWidget(self.user_edit, 1)
        row = QHBoxLayout()
        for text, slot in (("Save", self.save_prompt), ("Save As…", self.save_prompt_as), ("Delete", self.delete_prompt)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            if text == "Save":
                b.setObjectName("primary")
            row.addWidget(b)
        v.addLayout(row)
        v.addWidget(self._section("OUTPUT"))
        form = QFormLayout()
        self.prefix = QLineEdit(self.s["prefix"])
        self.prefix.setPlaceholderText("e.g. ohwx person, ")
        self.suffix = QLineEdit(self.s["suffix"])
        form.addRow("Caption prefix", self.prefix)
        form.addRow("Caption suffix", self.suffix)
        v.addLayout(form)
        self.skip = QCheckBox("Skip images that already have a caption")
        self.skip.setChecked(self.s["skip_existing"])
        v.addWidget(self.skip)
        dock.setWidget(w)
        dock.setMinimumWidth(320)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._reload_prompt_combo(self.s["last_prompt"])

    @staticmethod
    def _section(text):
        lbl = QLabel(text)
        lbl.setObjectName("section")
        return lbl

    def _build_log_dock(self):
        self.log_dock = QDockWidget("Log", self)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        self.log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()
        view = self.menuBar().addMenu("&View")
        view.addAction(self.log_dock.toggleViewAction())
        view.addSeparator()
        self.a_light = QAction("Light mode", self, checkable=True)
        self.a_light.setChecked(theme.current() == theme.LIGHT)
        self.a_light.toggled.connect(self.set_light_mode)
        view.addAction(self.a_light)

    def _build_status(self):
        self.progress = QProgressBar()
        self.progress.setFixedWidth(260)
        self.progress.setVisible(False)
        self.stats = QLabel()
        self.statusBar().addWidget(self.stats, 1)
        self.statusBar().addPermanentWidget(self.progress)

    # ------------------------------------------------------------------ prompts
    def _reload_prompt_combo(self, select=None):
        self.prompt_combo.blockSignals(True)
        self.prompt_combo.clear()
        self.prompt_combo.addItems([p["name"] for p in self.prompts])
        i = max(0, self.prompt_combo.findText(select or ""))
        self.prompt_combo.setCurrentIndex(i)
        self.prompt_combo.blockSignals(False)
        self._prompt_selected(i)

    def _prompt_selected(self, i):
        if 0 <= i < len(self.prompts):
            self.sys_edit.setPlainText(self.prompts[i]["system"])
            self.user_edit.setPlainText(self.prompts[i]["user"])

    def save_prompt(self):
        i = self.prompt_combo.currentIndex()
        if i < 0:
            return self.save_prompt_as()
        self.prompts[i]["system"] = self.sys_edit.toPlainText()
        self.prompts[i]["user"] = self.user_edit.toPlainText().strip()
        if self._save_prompts():
            self.statusBar().showMessage(f"Saved prompt '{self.prompts[i]['name']}'", 3000)

    def save_prompt_as(self):
        name, ok = QInputDialog.getText(self, "Save prompt", "Name:", text=self.prompt_combo.currentText())
        name = name.strip()
        if not ok or not name:
            return
        entry = {"name": name, "system": self.sys_edit.toPlainText(), "user": self.user_edit.toPlainText().strip()}
        for i, p in enumerate(self.prompts):
            if p["name"] == name:
                self.prompts[i] = entry
                break
        else:
            self.prompts.append(entry)
        self._save_prompts()
        self._reload_prompt_combo(name)

    def delete_prompt(self):
        i = self.prompt_combo.currentIndex()
        if i < 0 or QMessageBox.question(self, "Delete prompt", f"Delete '{self.prompts[i]['name']}'?") != QMessageBox.Yes:
            return
        del self.prompts[i]
        self._save_prompts()
        self._reload_prompt_combo()

    def _save_prompts(self):
        if config.save_prompts(self.prompts):
            return True
        QMessageBox.warning(self, "Prompts not saved", f"{config.PROMPTS_FILE} couldn't be read when the app "
                            "started (another program may have it open), so it isn't overwritten. Your "
                            "prompt change lasts until you close the app.")
        return False

    # ------------------------------------------------------------------ images
    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add images", self.s["last_dir"],
                                                "Images (*.jpg *.jpeg *.png *.webp *.bmp *.gif *.tif *.tiff)")
        if files:
            self.s["last_dir"] = os.path.dirname(files[0])
            self._add(files)

    def add_folder(self, recursive):
        d = QFileDialog.getExistingDirectory(self, "Add folder", self.s["last_dir"])
        if d:
            self.s["last_dir"] = d
            self._add([d], recursive)

    def _add(self, paths, recursive=False):
        """List the images in `paths`. Finding them and reading their captions happens on a background
        thread, so a big folder (or a slow network drive) doesn't freeze the window."""
        paths = [p for p in paths if p]
        if not paths:
            return
        gen, known = self.add_gen, self.model.known_keys()
        self.adding += 1
        self.statusBar().showMessage("Adding images…")

        def run():
            unreadable, listing = [], {}
            try:
                new = unique_new(collect_images(paths, recursive, unreadable, listing), known)
                result = (gen, load_entries(new, self.adder.progress.emit, listing), unreadable, "")
            except Exception as e:  # noqa: BLE001
                result = (gen, [], unreadable, str(e))
            self.adder.done.emit(result)
        threading.Thread(target=run, daemon=True).start()

    def _added(self, result):
        gen, entries, unreadable, err = result
        self.adding -= 1
        if gen != self.add_gen:
            return  # the list was cleared while this was scanning
        known = set(self.model.index_of)
        n = self.model.add_entries(entries)
        clashes = [g for g in caption_clashes(self.model.index_of) if not known.issuperset(g)] if n else []
        msg = f"Added {n:,} image(s)"
        if unreadable:
            msg += f"; {len(unreadable)} folder(s) couldn't be read (see log)"
            for d in unreadable:
                self._log(f"Couldn't read folder: {d}")
        bad = [e for e in entries if e.get("unreadable")]
        if bad:
            msg += f"; {len(bad)} caption file(s) couldn't be read (see log)"
            for e in bad:
                self._log(f"Couldn't read {caption_path(e['path'])}, so it is left alone: {e['unreadable']}")
            self.log_dock.show()
        if err:
            self._log(f"Adding images failed: {err}")
            QMessageBox.warning(self, "Add images", f"Adding images failed:\n\n{err}")
        self.statusBar().showMessage(msg, 4000)
        if clashes:
            self._warn_clashes(clashes)

    def _warn_clashes(self, groups):
        """photo.png and photo.jpg both read and write photo.txt: say so before one overwrites the other."""
        def line(g):
            return " + ".join(os.path.basename(p) for p in g) + "  →  " + os.path.basename(caption_path(g[0]))
        for g in groups:
            self._log("Shared caption file: " + " + ".join(g) + "  →  " + caption_path(g[0]))
        shown = "\n".join(line(g) for g in groups[:10])
        if len(groups) > 10:
            shown += f"\n…and {len(groups) - 10} more (see View ▸ Log)"
        QMessageBox.warning(
            self, "Images share a caption file",
            f"{len(groups)} set(s) of images have the same name with different extensions, so each set "
            f"shares one caption file:\n\n{shown}\n\nCaptioning or editing one image overwrites the caption "
            "of the other, and a trainer can't tell which image the caption belongs to. Rename or remove "
            "the duplicates.")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self._add([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()],
                  recursive=True)

    def _selected_rows(self):
        return sorted({self.proxy.mapToSource(i).row() for i in self.grid.selectedIndexes()})

    def remove_selected(self):
        rows = self._selected_rows()
        if rows:
            # don't spend GPU time on images that are off the list
            gone = {self.model.items[r]["path"] for r in rows}
            if self.engine.active:
                self.engine.drop(gone)
            if self.pending_job:  # a job still waiting for the model
                paths, *rest = self.pending_job
                paths = [p for p in paths if p not in gone]
                self.pending_job = (paths, *rest) if paths else None
            self.model.remove_rows(rows)

    def clear_list(self):
        if (self.model.items or self.adding) and QMessageBox.question(self, "Clear list", "Remove all images from the list?\n(Files on disk are not touched.)") == QMessageBox.Yes:
            self.add_gen += 1  # images still being scanned are dropped too
            if self.engine.active:
                self.engine.drop(list(self.model.index_of))  # replies on their way are unwanted too
                self.engine.cancel()  # nothing left to caption
            self.pending_job = None  # nor for a job waiting for the model (which goes on loading)
            self.model.clear()

    def _apply_search(self):
        self.search_timer.stop()
        if self.search.text().lower() != self.proxy.text:
            self.proxy.set_filter(text=self.search.text())
            self._update_stats()

    def _thumb_size(self, v):
        self.s["thumb_size"] = v
        self.grid.set_thumb_size(v)

    def _update_stats(self):
        n, c = len(self.model.items), self.model.captioned_count()
        shown = self.proxy.rowCount()
        self.stats.setText(f"{n} images · {c} captioned · {n - c} missing" + (f" · showing {shown}" if shown != n else ""))

    def _context_menu(self, pos):
        idx = self.grid.indexAt(pos)
        if not idx.isValid():
            return
        if not self.grid.selectionModel().isSelected(idx):
            self.grid.setCurrentIndex(idx)
        m = QMenu(self)
        m.addAction("Caption selected", lambda: self.caption("selected"))
        m.addAction("Clear caption(s)", self._clear_selected_captions)
        m.addAction("Show in Explorer", lambda: subprocess.Popen(
            ["explorer", "/select,", os.path.normpath(self.proxy.data(idx, Qt.ToolTipRole))]))
        m.addAction("Remove from list", self.remove_selected)
        m.exec(self.grid.viewport().mapToGlobal(pos))

    def _clear_selected_captions(self):
        rows = self._selected_rows()
        if not rows or QMessageBox.question(self, "Clear captions", f"Empty the caption of {len(rows)} image(s)?") != QMessageBox.Yes:
            return
        failed = []
        for r in rows:  # one file that can't be written no longer stops the rest (with an uncaught error)
            try:
                self.model.set_caption(r, "", stats=False)
            except RuntimeError as e:
                failed.append(str(e))
        self.model.stats_changed.emit()
        if failed:
            for m in failed:
                self._log(m)
            QMessageBox.warning(self, "Clear captions", f"{len(failed)} caption file(s) couldn't be written "
                                f"(see View ▸ Log):\n\n" + "\n".join(failed[:5]))

    # ------------------------------------------------------------------ settings / servers
    def current_settings(self):
        self.s.update(prefix=self.prefix.text(), suffix=self.suffix.text(), skip_existing=self.skip.isChecked(),
                      last_prompt=self.prompt_combo.currentText())
        return dict(self.s)

    def open_settings(self):
        dlg = SettingsDialog(self.current_settings(), self)
        if dlg.exec():
            self.s = dlg.result_settings()
            if not config.save_settings(self.s):
                QMessageBox.warning(self, "Settings not saved", f"{config.SETTINGS_FILE} couldn't be read when "
                                    "the app started (another program may have it open), so it isn't "
                                    "overwritten. The new settings last until you close the app.")

    def set_light_mode(self, on):
        """Switch the whole app between the dark and light theme right away and remember it."""
        mode = theme.LIGHT if on else theme.DARK
        if mode == theme.current():
            return
        theme.apply(QApplication.instance(), mode)
        self.brand.setPixmap(theme.logo(26))
        self.grid.empty_logo = None
        self.grid.viewport().update()
        self.s["theme"] = mode
        config.save_settings(self.current_settings())

    def _log(self, line):
        self.log_view.appendPlainText(line)

    def _managed_sig(self):
        return tuple(str(self.s[k]) for k in MANAGED_KEYS)

    def _launch_servers(self):
        """Start llama-server(s) in a background thread (loading a model takes a while)."""
        s = dict(self.s)
        self.starting = True
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.statusBar().showMessage("Loading model onto GPU(s)… (see View ▸ Log)")
        self.log_dock.show()

        ticket = self.servers.begin()  # here, so a Stop pressed from now on reaches this start

        def run():
            # start() kills what it launched if it fails or is cancelled, so no stop() here:
            # that could kill servers a newer start already brought up
            try:
                eps = self.servers.start(s, self.bridge.log.emit, ticket=ticket)
                self.bridge.ready.emit(eps)
            except StartCancelled:
                self.bridge.cancelled.emit()
            except Exception as e:  # noqa: BLE001
                self.bridge.error.emit(str(e))
        threading.Thread(target=run, daemon=True).start()

    def start_servers_only(self):
        self.current_settings()
        if self.s["backend"] != "managed":
            QMessageBox.information(self, "Servers", "The external backend is selected; there is nothing to start.")
        elif self.starting:
            return
        elif self.engine.active and self.run_managed:
            # a restart would kill the servers the job is using and fail every image left
            self.statusBar().showMessage("Captioning is using the servers: stop it first", 4000)
        elif self.servers.running() and self.server_sig == self._managed_sig():
            # same settings: reloading would only re-read the whole model from disk
            self.statusBar().showMessage(f"{len(self.servers.endpoints)} server(s) already running", 4000)
        else:
            self.server_sig = self._managed_sig()
            self._launch_servers()

    def stop_servers(self):
        if self.engine.active and self.run_managed:
            if QMessageBox.question(self, "Stop servers", "Captioning is running on these servers. "
                                    "Stop captioning and the servers?") != QMessageBox.Yes:
                return
            self.engine.cancel()  # first, so the dropped connections count as stopped, not failed
        self.servers.cancel()
        self.servers.stop()
        self.server_sig = None
        self.statusBar().showMessage("Servers stopped", 3000)

    def _servers_ready(self, eps):
        if not self.servers.running():  # stopped (or one died) between its health check and now
            return self._servers_cancelled()
        self.starting = False
        self.progress.setVisible(False)
        self.statusBar().showMessage(f"{len(eps)} server(s) ready", 4000)
        if self.pending_job:
            job, self.pending_job = self.pending_job, None
            self._run_engine(eps, *job)
        elif not self.engine.active:
            self.a_stop.setEnabled(False)  # its job was cleared away while the model loaded

    def _servers_error(self, msg):
        self.starting = False
        self.progress.setVisible(False)
        self.server_sig = None
        self._reset_queued()
        self.pending_job = None
        self.a_stop.setEnabled(False)
        QMessageBox.critical(self, "Server failed to start", msg)

    def _servers_cancelled(self):
        self.starting = False
        self.progress.setVisible(False)
        self.server_sig = None
        self._reset_queued()
        self.pending_job = None
        self.a_stop.setEnabled(False)
        self.statusBar().showMessage("Server start stopped", 4000)

    # ------------------------------------------------------------------ captioning
    def _commit_open_editors(self):
        """Save what is typed in an open caption box and close it (Ctrl+Return works from inside one),
        so the job sees it as the image's caption."""
        for ed in self.grid.viewport().findChildren(CaptionEditor):
            if not ed.isHidden():  # closed editors are hidden until they are deleted
                self.grid.commitData(ed)
                self.grid.closeEditor(ed, QAbstractItemDelegate.NoHint)

    def caption(self, which):
        if self.engine.active or self.starting:
            return
        self._commit_open_editors()
        self._apply_search()  # Caption All takes what the grid shows: a search still being typed counts
        s = self.current_settings()
        if which == "selected":
            rows = self._selected_rows()
        else:  # everything currently visible in the grid
            rows = [self.proxy.mapToSource(self.proxy.index(i, 0)).row() for i in range(self.proxy.rowCount())]
        if s["skip_existing"] and which == "all":
            # a caption file that couldn't be read may well hold a caption: leave it alone too
            rows = [r for r in rows if not self.model.items[r]["caption"].strip()
                    and not self.model.items[r].get("unreadable")]
        paths = [self.model.items[r]["path"] for r in rows]
        if not paths:
            self.statusBar().showMessage("Nothing to caption", 3000)
            return
        if s["backend"] == "external" and not external_endpoints(s):
            QMessageBox.warning(self, "Settings", "Add at least one endpoint URL in Settings.")
            return
        self.model.set_statuses(rows, QUEUED)
        self.snapshots = {self.model.items[r]["path"]: self.model.items[r]["caption"] for r in rows}
        self.failures = self.warnings = 0
        self.run_managed = s["backend"] == "managed"
        self.a_stop.setEnabled(True)
        # the settings go with the job: ones changed while the model loads apply to the next job, not
        # to this one (whose servers were started with the old ones)
        job = (paths, self.sys_edit.toPlainText(), self.user_edit.toPlainText().strip(), s)
        if s["backend"] == "external":
            self._run_engine(external_endpoints(s), *job)
        elif self.servers.running() and self.server_sig == self._managed_sig():
            self._run_engine(list(self.servers.endpoints), *job)
        else:
            self.pending_job = job
            self.server_sig = self._managed_sig()
            self._launch_servers()

    def _run_engine(self, eps, paths, system, user, s):
        self.progress.setVisible(True)
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        workers = sum(e["slots"] for e in eps)
        self._log(f"Captioning {len(paths)} image(s) with {workers} worker(s) on {len(eps)} endpoint(s)")
        self.engine.start(paths, eps, system, user, s)
        if not self.engine.active:  # no endpoints: nothing ran
            self._on_finished()
            return
        self.a_start_srv.setEnabled(not self.run_managed)

    def stop_caption(self):
        self.engine.cancel()
        self.servers.cancel()
        self.pending_job = None
        self.statusBar().showMessage("Stopping after in-flight requests finish…")

    def _reset_queued(self):
        self.model.set_statuses(range(len(self.model.items)), IDLE, only=(QUEUED, BUSY))

    def _on_captioned(self, path, text):
        row = self.model.index_of.get(path)
        if row is None or self.engine.is_dropped(path):  # removed during the job (maybe added again)
            return
        # an inline editor open on this card holds the old text: commit it now, so it can't
        # overwrite the model's caption later and any typing in it counts as an edit below
        ed = self.grid.indexWidget(self.proxy.mapFromSource(self.model.index(row)))
        if ed is not None:
            self.grid.commitData(ed)
            self.grid.closeEditor(ed, QAbstractItemDelegate.NoHint)
        before = self.snapshots.pop(path, None)
        if before is not None and self.model.items[row]["caption"] != before:
            self.model.set_status(path, IDLE)
            self._log(f"Kept your edit of {os.path.basename(path)} (model caption not saved): {text}")
            return
        # and the file itself: another program (or the other image of a pair sharing one caption file)
        # may have written it since it was read
        try:
            on_disk, exists = read_caption(path)
        except OSError as e:
            self.failures += 1
            self.model.set_status(path, ERROR)
            self._log(f"FAILED {os.path.basename(path)}: couldn't read the caption file already there, "
                      f"so it wasn't replaced ({e})")
            return
        if exists and on_disk != self.model.items[row]["caption"]:
            self.model.show_caption(row, on_disk)
            self._log(f"Kept {os.path.basename(caption_path(path))}: it changed on disk while "
                      f"{os.path.basename(path)} was being captioned (model caption not saved): {text}")
            return
        try:
            self.model.set_caption(row, text, force=True)
        except RuntimeError as e:  # captioned, but the .txt couldn't be written: that image failed
            self.failures += 1
            self._log(f"FAILED {os.path.basename(path)}: {e}")

    def _on_failed(self, path, err):
        if self.engine.is_dropped(path):
            return
        self.failures += 1
        self.model.set_status(path, ERROR)
        self._log(f"FAILED {os.path.basename(path)}: {err}")

    def _on_warning(self, path, msg):
        self.warnings += 1
        self._log(f"WARNING {os.path.basename(path)}: {msg}" if path else f"WARNING: {msg}")

    def _on_progress(self, done, total):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.statusBar().showMessage(f"Captioning {done}/{total}")

    def _on_finished(self):
        if self.engine.active:  # late signal of the previous job, after the next one started
            return
        self._reset_queued()
        self.progress.setVisible(False)
        self.a_stop.setEnabled(False)
        self.a_start_srv.setEnabled(True)
        notes = [f"{self.failures} failed"] if self.failures else []
        notes += [f"{self.warnings} warning(s)"] if self.warnings else []
        reason = self.engine.stop_reason
        msg = ("Captioning stopped: no server responding" if reason else "Captioning finished") + \
            (f" — {', '.join(notes)} (see log)" if notes else "")
        self.statusBar().showMessage(msg, 8000)
        if notes or reason:
            self.log_dock.show()
        if reason:
            self._log(reason.replace("\n\n", " "))
            QMessageBox.warning(self, "Captioning stopped", reason)

    def closeEvent(self, e):
        if self.engine.active and QMessageBox.question(self, "Quit", "Captioning is in progress. Quit anyway?") != QMessageBox.Yes:
            e.ignore()
            return
        self.current_settings()
        config.save_settings(self.s)
        self.engine.cancel()
        self.servers.stop()
        e.accept()
