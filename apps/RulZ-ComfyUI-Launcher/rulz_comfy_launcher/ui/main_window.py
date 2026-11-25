"""The main window: the list of configurations on the left, the selected one on the right."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtCore import QByteArray, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton,
                               QSizePolicy, QSpinBox, QSplitter, QStackedWidget, QSystemTrayIcon,
                               QTabWidget, QToolButton, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__, wsl
from ..command import (ENV_DUMP_SCRIPT, build_introspection_script, diff_shell_env, instance_url,
                       launch_signature, page_url, parse_env_dump, preview_command, validate,
                       validate_together)
from ..gpus import GpuMonitor
from ..instance import InstanceManager, PortWatcher, State
from ..model import GPU_AUTO, GPU_CPU, THEME_DARK, THEME_LIGHT, ConfigStore, Profile
from ..options import Catalog, builtin_catalog, parse_introspection_output
from . import theme
from .profile_editor import ProfileEditor
from .page_opener import PageOpener
from .widgets import ElidedLabel, GpuStrip, LogView, StatusDot, StatusPill, fit_button


def _short_gpu(name: str) -> str:
    for prefix in ("NVIDIA GeForce RTX ", "NVIDIA GeForce GTX ", "NVIDIA GeForce ", "NVIDIA "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def gpu_summary(profile: Profile) -> str:
    if profile.gpu_mode == GPU_CPU:
        return "CPU"
    if profile.gpu_mode == GPU_AUTO:
        return "all GPUs"
    if not profile.gpus:
        return "no GPU ticked"
    return " + ".join(_short_gpu(g.name) or "GPU" for g in profile.gpus)


class ProfileRow(QWidget):
    """One line in the configuration list."""
    check_toggled = Signal(str, bool)
    action_clicked = Signal(str)

    def __init__(self, profile_id: str, parent=None):
        super().__init__(parent)
        self.profile_id = profile_id
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 8, 7)
        layout.setSpacing(9)
        self.check = QCheckBox()
        self.check.setToolTip("Tick the configurations that 'Start checked' should start together.")
        self.check.toggled.connect(lambda on: self.check_toggled.emit(self.profile_id, on))
        self.dot = StatusDot()
        text = QVBoxLayout()
        text.setSpacing(1)
        self.name = ElidedLabel()
        self.name.setObjectName("rowName")
        self.sub = ElidedLabel()
        self.sub.setObjectName("hint")
        for label in (self.name, self.sub):
            # Long text ends in ... instead of widening the list.
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        text.addWidget(self.name)
        text.addWidget(self.sub)
        self.button = QToolButton()
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.setMinimumWidth(64)
        self.button.clicked.connect(lambda: self.action_clicked.emit(self.profile_id))
        layout.addWidget(self.check)
        layout.addWidget(self.dot)
        layout.addLayout(text, 1)
        layout.addWidget(self.button)

    def polish_children(self):
        for widget in (self, self.name, self.sub, self.check, self.button):
            widget.ensurePolished()
        self.layout().invalidate()

    def update_row(self, name: str, sub: str, tone: str, checked: bool, action: str):
        self.name.setText(name or "(no name)")
        self.sub.setText(sub)
        self.setToolTip(f"{name}\n{sub}")
        self.dot.set_tone(tone)
        if self.check.isChecked() != checked:
            self.check.blockSignals(True)
            self.check.setChecked(checked)
            self.check.blockSignals(False)
        self.button.setText(action)


class PreferencesDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(460)
        form = QFormLayout(self)
        form.setContentsMargins(20, 16, 20, 16)
        form.setVerticalSpacing(12)
        form.setHorizontalSpacing(16)
        self.theme = QComboBox()
        self.theme.addItem("Dark", THEME_DARK)
        self.theme.addItem("Light", THEME_LIGHT)
        self.theme.setCurrentIndex(max(0, self.theme.findData(settings.theme)))
        self.theme.setToolTip("The look of the launcher. The Light mode button in the sidebar switches it too.")
        self.stagger = QDoubleSpinBox()
        self.stagger.setRange(0.0, 120.0)
        self.stagger.setDecimals(1)
        self.stagger.setSuffix(" s")
        self.stagger.setValue(settings.stagger_seconds)
        self.stagger.setToolTip("When several configurations start together, wait this long between them.")
        self.stop_timeout = QSpinBox()
        self.stop_timeout.setRange(1, 300)
        self.stop_timeout.setSuffix(" s")
        self.stop_timeout.setValue(settings.stop_timeout)
        self.stop_timeout.setToolTip("How long to wait after Ctrl+C before ComfyUI is forced to stop.")
        self.log_lines = QSpinBox()
        self.log_lines.setRange(200, 200000)
        self.log_lines.setSingleStep(1000)
        self.log_lines.setValue(settings.log_max_lines)
        for spin in (self.stagger, self.stop_timeout, self.log_lines):
            spin.lineEdit().setTextMargins(9, 0, 0, 0)   # Qt ignores the style sheet's left padding here
        form.addRow("Theme", self.theme)
        form.addRow("Pause between simultaneous starts", self.stagger)
        form.addRow("Wait for a clean stop", self.stop_timeout)
        form.addRow("Log lines kept per instance", self.log_lines)
        self.close_to_tray = QCheckBox("Closing the window keeps the launcher in the tray")
        self.close_to_tray.setChecked(settings.close_to_tray)
        self.close_to_tray.setToolTip("The close button hides the window to the notification area instead "
                                      "of exiting. Use Exit in the tray menu to quit.")
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.close_to_tray.setEnabled(False)
            self.close_to_tray.setToolTip("No notification area is available on this system.")
        form.addRow(self.close_to_tray)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setProperty("primary", True)
        buttons.accepted.connect(self.accept)
        self._focus_cleared = False
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


    def showEvent(self, event):
        super().showEvent(event)
        if not self._focus_cleared:
            # no field is rung at start-up; Tab brings the ring to the first one
            self._focus_cleared = True
            QTimer.singleShot(0, lambda: self.setFocus(Qt.FocusReason.OtherFocusReason))


class MainWindow(QMainWindow):
    def __init__(self, store: ConfigStore, job=None, parent=None):
        super().__init__(parent)
        self.store = store
        theme.ensure_theme(QApplication.instance(), store.settings.theme)
        self.manager = InstanceManager(job, self)
        self.gpu_monitor = GpuMonitor(job, parent=self)
        self.watcher = PortWatcher(parent=self)
        self.builtin = builtin_catalog()
        self.catalogs: dict[str, Catalog] = {}
        self._catalog_busy: set[str] = set()
        self._catalog_tried: set[str] = set()
        self._catalog_path = self.store.path.with_name("catalogs.json")
        self._current: Profile | None = None
        self._rows: dict[str, ProfileRow] = {}
        self._items: dict[str, QListWidgetItem] = {}
        self._logs: dict[str, LogView] = {}
        self._quitting = False
        self._closing_for_real = False
        self._exit_requested = False     # Exit from the tray menu: skip close-to-tray
        self._tray_hint_shown = False
        self._save_error = ""
        self._tray: QSystemTrayIcon | None = None
        self._focus_cleared = False

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(theme.app_icon())
        self.resize(1280, 860)
        self.setMinimumSize(1100, 640)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._save_now)

        self._build_ui()
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)   # so the window itself can hold the focus at start
        QShortcut(QKeySequence("Ctrl+Shift+L"), self, activated=self.toggle_theme)
        self._page_opener = PageOpener(QDesktopServices.openUrl,
                                       lambda message: self.statusBar().showMessage(message, 10000), self)
        self._build_tray()
        self._load_catalog_cache()

        self.manager.state_changed.connect(self._on_state_changed)
        self.manager.output.connect(self._on_output)
        self.manager.ready.connect(self._on_ready)
        self.gpu_monitor.updated.connect(self._on_gpus)
        self.watcher.changed.connect(self._on_port_changed)

        self._populate_list()
        start = self.store.find(self.store.settings.last_profile) or self.store.profiles[0]
        self._select_profile(start)
        self._restore_geometry()

        self.gpu_monitor.start()
        self._sync_watched_ports()
        self.watcher.start()
        wsl.list_distros(self).finished.connect(self._on_distros)
        if self.store.load_error:
            QTimer.singleShot(200, lambda: QMessageBox.warning(self, APP_NAME, self.store.load_error))

    # ======================================================================
    # building the window
    # ======================================================================
    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter
        self.setCentralWidget(splitter)

        # ---- sidebar -------------------------------------------------------
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(330)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 18, 0, 12)
        side.setSpacing(8)

        # Brand block, the same as in the other RulZ apps: logo, then the app's name.
        self.logo = QLabel()
        self.logo.setContentsMargins(18, 0, 0, 0)
        self._update_logo()
        side.addWidget(self.logo)
        brand = QLabel("COMFYUI LAUNCHER")
        brand.setObjectName("brandSub")
        brand.setContentsMargins(20, 0, 0, 0)
        side.addWidget(brand)
        side.addSpacing(12)

        title = QLabel("Configurations")
        title.setObjectName("sideTitle")
        title.setContentsMargins(16, 0, 0, 0)
        side.addWidget(title)

        tools = QHBoxLayout()
        tools.setContentsMargins(10, 0, 10, 0)
        tools.setSpacing(4)
        for text, tip, slot, danger in (
            ("New", "Add a configuration", self._new_profile, False),
            ("Duplicate", "Copy the selected configuration", self._duplicate_profile, False),
            ("Delete", "Delete the selected configuration", self._delete_profile, True),
        ):
            button = QPushButton(text)
            button.setProperty("controlSize", "sm")
            button.setProperty("danger", danger)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            fit_button(button, padding=26)
            tools.addWidget(button)
        tools.addStretch(1)
        for icon_name, tip, delta in (("arrow-up", "Move up", -1), ("arrow-down", "Move down", 1)):
            button = QToolButton()
            button.setProperty("iconOnly", True)
            theme.set_icon(button, icon_name, "muted", 16)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, delta=delta: self._move_profile(delta))
            tools.addWidget(button)
        side.addLayout(tools)

        self.list = QListWidget()
        self.list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list.currentItemChanged.connect(self._on_list_selection)
        side.addWidget(self.list, 1)

        bottom = QVBoxLayout()
        bottom.setContentsMargins(10, 0, 10, 0)
        bottom.setSpacing(6)
        self.start_checked = QPushButton("Start checked")
        self.start_checked.setToolTip("Start every ticked configuration, one after the other.")
        self.start_checked.clicked.connect(self._start_checked)
        self.stop_all = QPushButton("Stop all")
        self.stop_all.clicked.connect(self._stop_all)
        row = QHBoxLayout()
        row.addWidget(self.start_checked, 2)
        row.addWidget(self.stop_all, 1)
        bottom.addLayout(row)
        prefs = QPushButton("Preferences")
        prefs.setProperty("flat", True)
        prefs.setProperty("controlSize", "sm")
        prefs.clicked.connect(self._open_preferences)
        self.theme_button = QPushButton()
        self.theme_button.setProperty("flat", True)
        self.theme_button.setProperty("controlSize", "sm")
        self.theme_button.setToolTip("Switch between dark and light (Ctrl+Shift+L)")
        self.theme_button.clicked.connect(self.toggle_theme)
        about = QLabel(f"v{__version__}")
        about.setObjectName("hint")
        row = QHBoxLayout()
        row.setSpacing(2)
        row.addWidget(prefs)
        row.addWidget(self.theme_button)
        row.addStretch(1)
        row.addWidget(about)
        bottom.addLayout(row)
        side.addLayout(bottom)
        splitter.addWidget(sidebar)

        # ---- main area -----------------------------------------------------
        main = QWidget()
        column = QVBoxLayout(main)
        column.setContentsMargins(16, 12, 6, 0)
        column.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(10)
        self.title = ElidedLabel()
        self.title.setObjectName("pageTitle")
        self.pill = StatusPill()
        self.notice = QLabel()
        self.notice.setObjectName("notice")
        self.notice.setVisible(False)
        heading = QVBoxLayout()
        heading.setSpacing(2)
        name_row = QHBoxLayout()
        name_row.setSpacing(10)
        name_row.addWidget(self.title)
        name_row.addWidget(self.pill)
        name_row.addStretch(0)
        heading.addLayout(name_row)
        heading.addWidget(self.notice)
        header.addLayout(heading, 1)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start_current)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_current)
        self.restart_button = QPushButton("Restart")
        self.restart_button.clicked.connect(self._restart_current)
        self.open_button = QPushButton("Open in browser")
        self.open_button.clicked.connect(self._open_current)
        for button in (self.start_button, self.stop_button, self.restart_button, self.open_button):
            fit_button(button, *(("Stop", "Force stop") if button is self.stop_button else (button.text(),)))
            header.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)
        header.setContentsMargins(0, 0, 10, 0)
        column.addLayout(header)

        self.tabs = QTabWidget()
        self.editor = ProfileEditor()
        self.editor.changed.connect(self._on_profile_edited)
        self.editor.refresh_catalog_requested.connect(lambda: self._request_catalog(self._current, True))
        self.editor.import_env_requested.connect(self._import_env)
        self.tabs.addTab(self.editor, "Settings")

        log_page = QWidget()
        log_page.setObjectName("page")
        log_layout = QVBoxLayout(log_page)
        log_layout.setContentsMargins(0, 8, 10, 10)
        log_layout.setSpacing(6)
        bar = QHBoxLayout()
        self.url_label = QLabel()
        self.url_label.setObjectName("muted")
        self.url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        wrap = QCheckBox("Wrap long lines")
        wrap.setChecked(True)
        wrap.toggled.connect(self._set_log_wrap)
        self._wrap = wrap
        copy = QPushButton("Copy all")
        copy.setProperty("controlSize", "sm")
        copy.clicked.connect(self._copy_log)
        clear = QPushButton("Clear")
        clear.setProperty("controlSize", "sm")
        clear.clicked.connect(self._clear_log)
        bar.addWidget(self.url_label, 1)
        bar.addWidget(wrap)
        bar.addWidget(copy)
        bar.addWidget(clear)
        log_layout.addLayout(bar)
        self.log_stack = QStackedWidget()
        self._empty_log = QLabel("Nothing has been started yet.")
        self._empty_log.setObjectName("muted")
        self._empty_log.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.log_stack.addWidget(self._empty_log)
        log_layout.addWidget(self.log_stack, 1)
        self.tabs.addTab(log_page, "Log")
        column.addWidget(self.tabs, 1)
        splitter.addWidget(main)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 940])

        self.gpu_strip = GpuStrip()
        self.statusBar().addWidget(self.gpu_strip, 1)
        self.statusBar().setSizeGripEnabled(False)
        self._update_theme_button()

    def _build_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        tray = QSystemTrayIcon(self.windowIcon(), self)
        menu = self._build_tray_menu()
        tray.setContextMenu(menu)
        self._tray_menu = menu
        tray.activated.connect(self._on_tray_activated)
        tray.setToolTip(APP_NAME)
        tray.show()
        self._tray = tray

    def _build_tray_menu(self) -> QMenu:
        menu = QMenu()
        show = QAction("Show window", menu)
        show.triggered.connect(self.show_and_raise)
        start = QAction("Start checked", menu)
        start.triggered.connect(self._start_checked)
        stop = QAction("Stop all", menu)
        stop.triggered.connect(self._stop_all)
        to_tray = QAction("Close to tray", menu)
        to_tray.setCheckable(True)
        to_tray.setChecked(self.store.settings.close_to_tray)
        to_tray.toggled.connect(self._set_close_to_tray)
        self._close_to_tray_action = to_tray
        leave = QAction("Exit", menu)
        leave.triggered.connect(self._exit_from_tray)
        menu.addAction(show)
        menu.addSeparator()
        menu.addAction(start)
        menu.addAction(stop)
        menu.addSeparator()
        menu.addAction(to_tray)
        menu.addAction(leave)
        return menu

    # ======================================================================
    # the configuration list
    # ======================================================================
    def _populate_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        self._rows.clear()
        self._items.clear()
        for profile in self.store.profiles:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, profile.id)
            row = ProfileRow(profile.id)
            row.check_toggled.connect(self._on_row_checked)
            row.action_clicked.connect(self._on_row_action)
            self.list.addItem(item)
            self.list.setItemWidget(item, row)
            self._rows[profile.id] = row
            self._items[profile.id] = item
            self._refresh_row(profile)
            row.polish_children()   # the theme fonts decide the height
            item.setSizeHint(row.sizeHint())
        self.list.blockSignals(False)
        self._refresh_start_checked()

    def _display_state(self, profile: Profile) -> tuple[str, str, bool]:
        """(text, tone, is an outside server on the port) for a profile. The tone is a name
        (theme.TONE_*), not a colour: the colour comes from the active theme when it is drawn."""
        state = self.manager.state(profile.id)
        if state in (State.STOPPED, State.FAILED) and self._is_external(profile):
            return "Port in use (not started here)", theme.TONE_EXTERNAL, True
        return state.value, theme.STATE_TONES[state], False

    def _refresh_row(self, profile: Profile):
        row = self._rows.get(profile.id)
        if row is None:
            return
        text, tone, external = self._display_state(profile)
        state = self.manager.state(profile.id)
        sub = f"port {profile.port}  ·  {gpu_summary(profile)}"
        if external:
            sub = "In use  ·  " + sub
        elif state != State.STOPPED:
            sub = f"{text}  ·  " + sub
        if state in (State.STOPPED, State.FAILED):
            action = "Start"
        elif state == State.STOPPING:
            action = "Force"
        else:
            action = "Stop"
        row.update_row(profile.name, sub, tone, profile.checked, action)
        row.button.setToolTip({"Start": "Start this configuration", "Stop": "Stop this instance",
                               "Force": "It is still stopping. End it now."}[action])

    def _refresh_start_checked(self):
        count = sum(1 for p in self.store.profiles if p.checked)
        self.start_checked.setText(f"Start checked ({count})" if count else "Start checked")
        self.start_checked.setEnabled(count > 0)
        self.stop_all.setEnabled(self.manager.any_active())

    def _on_list_selection(self, current: QListWidgetItem | None, _previous):
        if current is None:
            return
        profile = self.store.find(current.data(Qt.ItemDataRole.UserRole))
        if profile is not None and profile is not self._current:
            self._select_profile(profile)

    def _select_profile(self, profile: Profile):
        self._current = profile
        item = self._items.get(profile.id)
        if item is not None and self.list.currentItem() is not item:
            self.list.blockSignals(True)
            self.list.setCurrentItem(item)
            self.list.blockSignals(False)
        self.editor.set_catalog(self.catalog_for(profile))
        self.editor.set_profile(profile)
        self.editor.set_gpus(self.gpu_monitor.gpus())
        self.editor.options.set_busy(self._catalog_key(profile) in self._catalog_busy)
        self.store.settings.last_profile = profile.id
        log = self._logs.get(profile.id)
        self.log_stack.setCurrentWidget(log if log is not None else self._empty_log)
        self._refresh_header()
        self._refresh_preview()
        self._schedule_save()

    def _on_row_checked(self, profile_id: str, checked: bool):
        profile = self.store.find(profile_id)
        if profile is not None:
            profile.checked = checked
            self._refresh_start_checked()
            self._schedule_save()

    def _on_row_action(self, profile_id: str):
        profile = self.store.find(profile_id)
        if profile is None:
            return
        state = self.manager.state(profile_id)
        if state in (State.STOPPED, State.FAILED):
            self._start_profiles([profile])
        elif state == State.STOPPING:
            self.manager.force_stop(profile_id)
        else:
            self.manager.stop(profile_id)

    def _new_profile(self):
        base = self._current
        profile = Profile(name=self.store.unique_name("New configuration"), port=self.store.free_port())
        if base is not None:
            profile.distro, profile.user = base.distro, base.user
            profile.comfy_dir, profile.python = base.comfy_dir, base.python
            profile.env = [list(pair) for pair in base.env]
            profile.pre_commands = base.pre_commands
        self.store.add(profile, after=base)
        self._populate_list()
        self._select_profile(profile)
        self.tabs.setCurrentIndex(0)
        self.editor.name.setFocus()
        self.editor.name.selectAll()
        self._sync_watched_ports()

    def _duplicate_profile(self):
        base = self._current
        if base is None:
            return
        profile = base.clone(self.store.unique_name(f"{base.name} copy"))
        profile.port = self.store.free_port(base.port)
        profile.checked = False
        self.store.add(profile, after=base)
        self._populate_list()
        self._select_profile(profile)
        self._sync_watched_ports()

    def _delete_profile(self):
        profile = self._current
        if profile is None:
            return
        if self.manager.is_busy(profile.id):
            QMessageBox.information(self, APP_NAME, f"'{profile.name}' is running. Stop it before deleting it.")
            return
        if not self._confirm("Delete configuration",
                             f"Delete the configuration '{profile.name}'?\n\n"
                             "This only removes it from the launcher. Nothing inside WSL is touched.",
                             "Delete", danger=True):
            return
        index = self.store.profiles.index(profile)
        self.store.remove(profile)
        log = self._logs.pop(profile.id, None)
        if log is not None:
            self.log_stack.removeWidget(log)
            log.deleteLater()
        if not self.store.profiles:
            self.store.add(Profile())
        self._populate_list()
        self._select_profile(self.store.profiles[min(index, len(self.store.profiles) - 1)])
        self._sync_watched_ports()

    def _move_profile(self, delta: int):
        profile = self._current
        if profile is None:
            return
        profiles = self.store.profiles
        index = profiles.index(profile)
        target = index + delta
        if not 0 <= target < len(profiles):
            return
        profiles[index], profiles[target] = profiles[target], profiles[index]
        self._populate_list()
        self._select_profile(profile)

    # ======================================================================
    # the selected configuration
    # ======================================================================
    def _on_profile_edited(self):
        profile = self._current
        if profile is None:
            return
        self.editor.gpus.refresh_names()
        self._refresh_row(profile)
        self._refresh_header()
        self._refresh_preview()
        self._sync_watched_ports()
        self.editor.set_catalog(self.catalog_for(profile))
        self._schedule_save()

    def _refresh_header(self):
        profile = self._current
        if profile is None:
            return
        self.title.setText(profile.name or "(no name)")
        self.title.setToolTip(profile.name)
        text, tone, external = self._display_state(profile)
        self.pill.set_status(text, tone)
        state = self.manager.state(profile.id)
        idle = state in (State.STOPPED, State.FAILED)
        self.start_button.setEnabled(idle and not external)
        self.stop_button.setEnabled(not idle)
        self.stop_button.setText("Force stop" if state == State.STOPPING else "Stop")
        self.restart_button.setEnabled(state in (State.STARTING, State.RUNNING))
        self.open_button.setEnabled(state == State.RUNNING or external)
        # One primary button in the view: Start while the configuration can be started, Open in
        # browser once it is running (or its port is served by another program).
        open_primary = self.open_button.isEnabled() and not self.start_button.isEnabled()
        if bool(self.open_button.property("primary")) != open_primary \
                or bool(self.start_button.property("primary")) == open_primary:
            theme.set_role(self.open_button, primary=open_primary)
            theme.set_role(self.start_button, primary=not open_primary)

        inst = self.manager.instance(profile.id)
        notice = ""
        if inst is not None and inst.active and state != State.STOPPING:
            # Compared with the option list the instance was started with, so a
            # list that was re-read meanwhile does not look like an edit.
            if launch_signature(profile, inst.catalog or self.catalog_for(profile)) != inst.signature:
                notice = "Settings changed since it started. Restart to apply them."
        self.notice.setText(notice)
        self.notice.setVisible(bool(notice))
        url = inst.url if inst is not None and inst.active else instance_url(profile)
        self.url_label.setText(page_url(url, profile.open_rulzflow))

    def _refresh_preview(self):
        profile = self._current
        if profile is None:
            return
        catalog = self.catalog_for(profile)
        command = preview_command(profile, catalog, self.gpu_monitor.index_by_uuid())
        self.editor.set_preview(command, validate(profile, catalog))

    def _start_current(self):
        if self._current is not None:
            self._start_profiles([self._current])

    def _stop_current(self):
        profile = self._current
        if profile is None:
            return
        if self.manager.state(profile.id) == State.STOPPING:
            self.manager.force_stop(profile.id)
        else:
            self.manager.stop(profile.id)

    def _restart_current(self):
        profile = self._current
        if profile is None or self._quitting:
            return
        catalog = self.catalog_for(profile)
        # The same checks as for a start, made before the running instance is stopped.
        others = [p for p in self.manager.active_profiles() if p.id != profile.id]
        issues = validate(profile, catalog) + validate_together([profile], others)
        problems = [i.message for i in issues if i.level == "error"]
        inst = self.manager.instance(profile.id)
        old_port = inst.profile.port if inst is not None and inst.profile is not None else None
        if profile.port != old_port and self.watcher.is_open(profile.port) \
                and all(p.port != profile.port for p in others):
            problems.append(f"Port {profile.port} is already in use by a program that was not started "
                            "from this launcher.")
        if problems:
            QMessageBox.warning(self, "Cannot restart", "\n\n".join(problems))
            return
        notes = [i.message for i in issues if i.level == "warning"]
        if notes:
            if not self._confirm("Restart anyway?", "\n\n".join(notes) + "\n\nRestart anyway?", "Restart anyway"):
                return
        self._save_now()
        self.manager.restart(profile, catalog, self.store.settings.stop_timeout)

    def _open_current(self):
        profile = self._current
        if profile is None:
            return
        inst = self.manager.instance(profile.id)
        url = inst.url if inst is not None and inst.active else instance_url(profile)
        self._open_page(profile.id, profile, inst if inst is not None and inst.active else None, url)

    def _open_page(self, profile_id, profile, inst, url):
        saved = self.store.find(profile_id)
        current = saved or profile

        def valid():
            return (not self._quitting and not self._closing_for_real
                    and self.store.find(profile_id) is saved
                    and (inst is None or (self.manager.instance(profile_id) is inst and inst.active)))

        self._page_opener.open(profile_id, current.name, url, current.open_rulzflow, valid)

    # ======================================================================
    # starting and stopping
    # ======================================================================
    def _is_external(self, profile: Profile) -> bool:
        """True when something we did not start is listening on the profile's port."""
        if not self.watcher.is_open(profile.port):
            return False
        return all(p.port != profile.port for p in self.manager.active_profiles())

    def _start_profiles(self, profiles: list[Profile]):
        to_start = [p for p in profiles if not self.manager.is_busy(p.id)]
        if not to_start or self._quitting:
            return
        self._save_now()
        errors: list[str] = []
        warnings: list[str] = []
        for profile in to_start:
            for issue in validate(profile, self.catalog_for(profile)):
                (errors if issue.level == "error" else warnings).append(f"{profile.name}: {issue.message}")
            if self._is_external(profile):
                errors.append(f"{profile.name}: port {profile.port} is already in use by a program that was "
                              "not started from this launcher, probably another ComfyUI. Stop that one "
                              "first, or pick another port.")
        for issue in validate_together(to_start, self.manager.active_profiles()):
            (errors if issue.level == "error" else warnings).append(issue.message)
        if errors:
            QMessageBox.warning(self, "Cannot start", "\n\n".join(errors))
            return
        if warnings:
            if not self._confirm("Start anyway?", "\n\n".join(warnings) + "\n\nStart anyway?", "Start anyway"):
                return

        settings = self.store.settings
        items = []
        for profile in to_start:
            self._ensure_log(profile.id)
            items.append((profile, self.catalog_for(profile)))
            if self._catalog_key(profile) not in self._catalog_tried:
                self._request_catalog(profile, False)
        if len(items) == 1:
            profile, catalog = items[0]
            self.manager.start(profile, catalog, settings.stop_timeout)
            if profile is self._current:
                self.log_stack.setCurrentWidget(self._logs[profile.id])
                self.tabs.setCurrentIndex(1)
        else:
            self.manager.start_many(items, settings.stagger_seconds, settings.stop_timeout)
            if self._current is not None and self._current.id in self._logs:
                self.log_stack.setCurrentWidget(self._logs[self._current.id])

    def _start_checked(self):
        self._start_profiles([p for p in self.store.profiles if p.checked])

    def _stop_all(self):
        self.manager.stop_all()

    def _ensure_log(self, profile_id: str) -> LogView:
        log = self._logs.get(profile_id)
        if log is None:
            log = LogView(self.store.settings.log_max_lines)
            log.set_wrap(self._wrap.isChecked())
            self._logs[profile_id] = log
            self.log_stack.addWidget(log)
        return log

    def _on_output(self, profile_id: str, text: str):
        self._ensure_log(profile_id).feed(text)

    def _on_ready(self, profile_id: str):
        inst = self.manager.instance(profile_id)
        if inst is not None and inst.profile is not None:
            profile = self.store.find(profile_id) or inst.profile
            if profile.auto_open:
                self._open_page(profile_id, profile, inst, inst.url)

    def _on_state_changed(self, profile_id: str):
        inst = self.manager.instance(profile_id)
        if inst is not None and not inst.active and inst.profile is not None:
            # Our own server just ended: its port must not be shown as
            # "in use by another program" until the next regular check.
            self.watcher.forget(inst.profile.port)
        profile = self.store.find(profile_id)
        if profile is not None:
            self._refresh_row(profile)
            if profile is self._current:
                self._refresh_header()
        self._refresh_start_checked()
        self._sync_watched_ports()
        self._refresh_tray_tip()
        if self._quitting and not self.manager.any_active():
            self._finish_quit()

    def _sync_watched_ports(self):
        # Saved ports, plus the ports running instances were started with (the
        # saved port may have been edited since).
        ports = {p.port for p in self.store.profiles}
        ports |= {p.port for p in self.manager.active_profiles()}
        self.watcher.set_ports(ports)

    def _confirm_ready(self, profile_id: str, port: int):
        """An open port only counts as "ready" if our instance is still alive a
        moment later. If another program took the port, ComfyUI fails to bind and
        has ended by then."""
        inst = self.manager.instance(profile_id)
        if (inst is not None and inst.state == State.STARTING and inst.profile is not None
                and inst.profile.port == port and self.watcher.is_open(port)):
            inst.mark_ready()

    def _on_port_changed(self, port: int, is_open: bool):
        if is_open:
            for profile_id in self.manager.active_ids():
                inst = self.manager.instance(profile_id)
                if (inst is not None and inst.profile is not None and inst.profile.port == port
                        and inst.state == State.STARTING):
                    QTimer.singleShot(2500, self, lambda pid=profile_id: self._confirm_ready(pid, port))
        for profile in self.store.profiles:
            if profile.port == port:
                self._refresh_row(profile)
        if self._current is not None and self._current.port == port:
            self._refresh_header()

    def _on_gpus(self):
        gpus = self.gpu_monitor.gpus()
        self.gpu_strip.set_gpus(gpus)
        self.editor.set_gpus(gpus)
        if self._current is not None and self._current.gpus:
            self._refresh_preview()

    def _on_distros(self, code: int, out: str, _err: str):
        if code == 0:
            self.editor.set_distros(wsl.parse_distros(out))

    # ======================================================================
    # the option catalog
    # ======================================================================
    @staticmethod
    def _catalog_key(profile: Profile) -> str:
        return "|".join([profile.distro, profile.user, profile.comfy_dir.strip(), profile.python.strip()])

    def catalog_for(self, profile: Profile) -> Catalog:
        return self.catalogs.get(self._catalog_key(profile), self.builtin)

    def _load_catalog_cache(self):
        try:
            data = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        for key, entry in data.items():
            if isinstance(entry, dict) and isinstance(entry.get("data"), dict):
                catalog = Catalog.from_data(entry["data"], source="read from this install")
                if catalog.options:
                    self.catalogs[key] = catalog
        self._catalog_raw = data

    def _store_catalog(self, key: str, data: dict):
        raw = getattr(self, "_catalog_raw", {})
        raw[key] = {"fetched": int(time.time()), "data": data}
        self._catalog_raw = raw
        try:
            self._catalog_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._catalog_path.with_name(self._catalog_path.name + ".tmp")
            tmp.write_text(json.dumps(raw), encoding="utf-8")
            tmp.replace(self._catalog_path)
        except OSError:
            pass

    def _request_catalog(self, profile: Profile | None, by_user: bool):
        """Ask the profile's ComfyUI which options it supports (runs a tiny script in WSL)."""
        if profile is None:
            return
        key = self._catalog_key(profile)
        if key in self._catalog_busy:
            return
        self._catalog_busy.add(key)
        self._catalog_tried.add(key)
        if profile is self._current:
            self.editor.options.set_busy(True)
        source = (Path(__file__).resolve().parent.parent / "scripts" / "introspect_args.py").read_text(encoding="utf-8")
        script = build_introspection_script(profile, source)
        command = wsl.run_bash(script, profile.distro, profile.user, timeout_ms=30000, parent=self)
        command.finished.connect(lambda code, out, err: self._on_catalog(key, by_user, code, out, err))

    def _on_catalog(self, key: str, by_user: bool, code: int, out: str, err: str):
        self._catalog_busy.discard(key)
        current = self._current
        is_current = current is not None and self._catalog_key(current) == key
        if is_current:
            self.editor.options.set_busy(False)
        data = parse_introspection_output(out) if code == 0 else None
        if data is None:
            if by_user:
                detail = (err or out).strip()[-600:] or "No answer from WSL."
                QMessageBox.warning(self, "Could not read the options",
                                    "ComfyUI's option list could not be read from this install, so the "
                                    f"built-in list stays in use.\n\n{detail}")
            return
        self.catalogs[key] = Catalog.from_data(data, source="read from this install")
        self._store_catalog(key, data)
        if is_current and current is not None:
            self.editor.set_catalog(self.catalogs[key])
            self._refresh_preview()
            self._refresh_header()

    # ======================================================================
    # environment import
    # ======================================================================
    def _import_env(self):
        profile = self._current
        if profile is None:
            return
        command = wsl.run_bash(ENV_DUMP_SCRIPT, profile.distro, profile.user, timeout_ms=30000, parent=self)
        command.finished.connect(lambda code, out, err: self._on_env(profile, code, out, err))

    def _on_env(self, profile: Profile, code: int, out: str, err: str):
        if self.store.find(profile.id) is None:
            return
        base, interactive = parse_env_dump(out)
        if code != 0 or not interactive:
            QMessageBox.warning(self, "Import failed", "The environment could not be read from WSL.\n\n"
                                + (err.strip()[-400:] or "No answer from WSL."))
            return
        found = diff_shell_env(base, interactive)
        if not found:
            QMessageBox.information(self, "Nothing to import",
                                    "Your ~/.bashrc does not set any variables that a launcher start is missing.")
            return
        listing = "\n".join(f"{name} = {value if len(value) < 90 else value[:87] + '...'}" for name, value in found)
        if not self._confirm("Import variables",
                             f"Add these {len(found)} variables to '{profile.name}'?\n"
                             f"Variables with the same name are replaced.\n\n{listing}", "Add variables"):
            return
        names = {name for name, _value in found}
        profile.env = [pair for pair in profile.env if pair[0] not in names] + found
        if profile is self._current:
            self.editor.reload_env()
            self._on_profile_edited()
        else:
            self._schedule_save()

    # ======================================================================
    # log tab
    # ======================================================================
    def _current_log(self) -> LogView | None:
        return self._logs.get(self._current.id) if self._current is not None else None

    def _set_log_wrap(self, wrap: bool):
        for log in self._logs.values():
            log.set_wrap(wrap)

    def _copy_log(self):
        log = self._current_log()
        if log is not None:
            QGuiApplication.clipboard().setText(log.toPlainText())

    def _clear_log(self):
        log = self._current_log()
        if log is not None:
            log.clear_log()

    # ======================================================================
    # preferences, saving, tray, closing
    # ======================================================================
    def _confirm(self, title: str, text: str, ok_text: str, danger: bool = False) -> bool:
        """A question with its own action word on the answer button (red when it destroys something)."""
        box = QMessageBox(QMessageBox.Icon.Question, title, text, QMessageBox.StandardButton.NoButton, self)
        yes = box.addButton(ok_text, QMessageBox.ButtonRole.AcceptRole)
        theme.set_role(yes, primary=not danger, danger=danger)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(box.button(QMessageBox.StandardButton.Cancel) if danger else yes)
        box.exec()
        return box.clickedButton() is yes

    # ======================================================================
    # theme
    # ======================================================================
    def _update_logo(self):
        pixmap = theme.logo_pixmap(150)
        if pixmap.isNull():
            self.logo.setPixmap(QPixmap())
            self.logo.setText("RulZ")
            self.logo.setObjectName("pageTitle")
        else:
            self.logo.setPixmap(pixmap)

    def _update_theme_button(self):
        dark = theme.is_dark()
        self.theme_button.setText("Light mode" if dark else "Dark mode")
        theme.set_icon(self.theme_button, "sun" if dark else "moon", "muted", 16)
        fit_button(self.theme_button, "Light mode", "Dark mode", padding=44)

    def set_theme(self, mode: str):
        """Switch the whole launcher to dark or light right away, and remember the choice."""
        mode = mode if mode in (THEME_DARK, THEME_LIGHT) else THEME_DARK
        self.store.settings.theme = mode
        self._save_now()
        if mode != theme.mode():
            theme.apply_theme(QApplication.instance(), mode)
        self._update_logo()
        self._update_theme_button()
        self._refresh_header()
        for widget in QApplication.topLevelWidgets():
            theme.retheme(widget)

    def toggle_theme(self):
        self.set_theme(THEME_LIGHT if theme.is_dark() else THEME_DARK)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._focus_cleared:
            # Qt would put the focus on the first button or field and ring it at start-up.
            self._focus_cleared = True
            QTimer.singleShot(0, lambda: self.setFocus(Qt.FocusReason.OtherFocusReason))

    def _open_preferences(self):
        dialog = PreferencesDialog(self.store.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = self.store.settings
            chosen = dialog.theme.currentData()
            if chosen != settings.theme:
                self.set_theme(chosen)
            settings.stagger_seconds = dialog.stagger.value()
            settings.stop_timeout = dialog.stop_timeout.value()
            settings.log_max_lines = dialog.log_lines.value()
            for log in self._logs.values():
                log.setMaximumBlockCount(settings.log_max_lines)
            if dialog.close_to_tray.isEnabled():
                self._set_close_to_tray(dialog.close_to_tray.isChecked())
            self._schedule_save()

    def _set_close_to_tray(self, on: bool):
        """One place for the setting, so Preferences and the tray menu stay in step."""
        self.store.settings.close_to_tray = bool(on)
        action = getattr(self, "_close_to_tray_action", None)
        if action is not None and action.isChecked() != bool(on):
            action.blockSignals(True)
            action.setChecked(bool(on))
            action.blockSignals(False)
        self._schedule_save()

    def _schedule_save(self):
        self._save_timer.start()

    def _save_now(self) -> bool:
        self._save_timer.stop()
        settings = self.store.settings
        settings.geometry = bytes(self.saveGeometry().toBase64().data()).decode("ascii")
        settings.splitter = self._splitter.sizes()
        try:
            self.store.save()
        except OSError as exc:
            self._save_error = str(exc)
            self.statusBar().showMessage(f"The configuration could not be saved: {exc}", 8000)
            return False
        self._save_error = ""
        return True

    def _restore_geometry(self):
        settings = self.store.settings
        if settings.geometry:
            self.restoreGeometry(QByteArray.fromBase64(settings.geometry.encode("ascii")))
        if len(settings.splitter) == 2 and all(s > 0 for s in settings.splitter):
            self._splitter.setSizes(settings.splitter)

    def _refresh_tray_tip(self):
        if self._tray is None:
            return
        running = [p.name for p in self.store.profiles if self.manager.state(p.id) == State.RUNNING]
        self._tray.setToolTip(f"{APP_NAME}\nRunning: {', '.join(running)}" if running else APP_NAME)

    def _on_tray_activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_and_raise()

    def show_and_raise(self):
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    def _exit_from_tray(self):
        self.show_and_raise()
        self._exit_requested = True
        try:
            self.close()
        finally:
            # Still here (the user cancelled the "still running" question): the next
            # close-button click goes back to the tray again.
            self._exit_requested = False

    def _hide_to_tray(self):
        self.hide()
        if self._tray is not None and not self._tray_hint_shown:
            self._tray_hint_shown = True
            self._tray.showMessage(APP_NAME, "Still running here. Click the icon to open the window.",
                                   QSystemTrayIcon.MessageIcon.Information, 4000)

    def closeEvent(self, event):
        if self._closing_for_real:
            event.accept()
            return
        if self._quitting:
            event.ignore()
            return
        if self._tray is not None and self.store.settings.close_to_tray and not self._exit_requested:
            event.ignore()
            self._hide_to_tray()
            return
        if not self.manager.any_active():
            event.ignore()
            self._finish_quit()
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(APP_NAME)
        count = len(self.manager.active_profiles())
        box.setText(f"{count} ComfyUI instance{'s are' if count != 1 else ' is'} still running.")
        box.setInformativeText("Instances belong to the launcher: they stop when it exits.")
        stop = box.addButton("Stop all and exit", QMessageBox.ButtonRole.DestructiveRole)
        theme.set_role(stop, primary=False, danger=True)
        keep = box.addButton("Keep running in the tray", QMessageBox.ButtonRole.AcceptRole) \
            if self._tray is not None else None
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        event.ignore()
        if clicked is stop:
            self._begin_quit()
        elif keep is not None and clicked is keep:
            self._hide_to_tray()

    def _begin_quit(self):
        self._quitting = True
        self._page_opener.stop()
        self.setEnabled(False)
        self.statusBar().showMessage("Stopping all instances ...")
        self.manager.stop_all()
        limit = (self.store.settings.stop_timeout + 12) * 1000
        QTimer.singleShot(limit, self.manager.force_stop_all)
        QTimer.singleShot(limit + 5000, self._finish_quit)
        if not self.manager.any_active():
            self._finish_quit()

    def _finish_quit(self):
        if self._closing_for_real:
            return
        self._page_opener.stop()
        while not self._save_now():
            self.setEnabled(True)
            answer = QMessageBox.warning(
                self, APP_NAME,
                f"The configuration could not be saved:\n{self._save_error}\n\n"
                "Retry, or exit and lose the changes made since the last successful save?",
                QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Discard,
                QMessageBox.StandardButton.Retry)
            if answer != QMessageBox.StandardButton.Retry:
                break
        self._closing_for_real = True
        self.watcher.stop()
        self.gpu_monitor.stop()
        if self._tray is not None:
            self._tray.hide()
        self.close()
        QApplication.quit()
