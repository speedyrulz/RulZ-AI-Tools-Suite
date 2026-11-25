"""The settings page for one configuration."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QIntValidator
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from ..command import Issue
from ..gpus import GpuInfo
from ..model import Profile
from ..options import Catalog
from . import theme
from .options_panel import OptionsPanel
from .widgets import Card, EnvTable, GpuSelector, NoWheelComboBox

_LISTEN_CHOICES = [
    ("127.0.0.1", "127.0.0.1   (this PC only)"),
    ("0.0.0.0", "0.0.0.0   (all IPv4 addresses: reachable from your network)"),
    ("0.0.0.0,::", "0.0.0.0,::   (all IPv4 and IPv6 addresses)"),
]


def _label(text: str, tip: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    if tip:
        label.setToolTip(tip)
    return label


class ProfileEditor(QWidget):
    changed = Signal()
    refresh_catalog_requested = Signal()
    import_env_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._profile: Profile | None = None
        self._loading = False
        self._distros: list[str] | None = None
        self._issues: list[Issue] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        page = QWidget()
        page.setObjectName("page")
        scroll.setWidget(page)
        column = QVBoxLayout(page)
        column.setContentsMargins(4, 8, 12, 16)
        column.setSpacing(12)

        # ---- install and network -------------------------------------------
        card = Card("ComfyUI install")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(3, 3)

        self.name = QLineEdit()
        self.name.setPlaceholderText("A name for this configuration")
        self.distro = NoWheelComboBox()
        self.distro.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.distro.setMinimumContentsLength(14)
        self.comfy_dir = QLineEdit()
        self.comfy_dir.setPlaceholderText("~/ComfyUI")
        self.comfy_dir.setFont(theme.mono_font())
        self.python = QLineEdit()
        self.python.setPlaceholderText("venv/bin/python")
        self.python.setFont(theme.mono_font())
        self.port = QLineEdit()
        self.port.setValidator(QIntValidator(1, 65535, self.port))
        self.port.setMaximumWidth(110)
        self.port.setPlaceholderText("8188")
        self.listen = NoWheelComboBox()
        self.listen.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.listen.setMinimumContentsLength(14)
        self.listen.setEditable(True)
        for value, text in _LISTEN_CHOICES:
            self.listen.addItem(text, value)
        self.user = QLineEdit()
        self.user.setPlaceholderText("default user")
        self.auto_open = QCheckBox("Open the browser when the server is ready")
        self.open_rulzflow = QCheckBox("Open the RulZFlow page instead of ComfyUI's")
        self.open_rulzflow.setToolTip("Requires the RulZFlow extension on this ComfyUI. "
                                     "Otherwise the normal page opens.")

        grid.addWidget(_label("Name"), 0, 0)
        grid.addWidget(self.name, 0, 1)
        grid.addWidget(_label("Port"), 0, 2)
        grid.addWidget(self.port, 0, 3, Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(_label("WSL distro"), 1, 0)
        grid.addWidget(self.distro, 1, 1)
        grid.addWidget(_label("Listen on", "The address ComfyUI accepts connections on (--listen)."), 1, 2)
        grid.addWidget(self.listen, 1, 3)
        grid.addWidget(_label("ComfyUI folder", "The Linux path of the folder that holds main.py."), 2, 0)
        grid.addWidget(self.comfy_dir, 2, 1)
        grid.addWidget(_label("Linux user", "Leave empty to use the distro's default user."), 2, 2)
        grid.addWidget(self.user, 2, 3)
        grid.addWidget(_label("Python", "The interpreter that runs main.py: a path inside the ComfyUI folder, "
                                        "a full path, or a command name. If an 'activate' script sits next "
                                        "to it, the virtual environment is activated first."), 3, 0)
        grid.addWidget(self.python, 3, 1)
        grid.addWidget(self.auto_open, 4, 0, 1, 4)
        grid.addWidget(self.open_rulzflow, 5, 0, 1, 4)
        card.body().addLayout(grid)
        column.addWidget(card)

        # ---- GPUs ----------------------------------------------------------
        card = Card("GPUs")
        self.gpus = GpuSelector()
        card.body().addWidget(self.gpus)
        column.addWidget(card)

        # ---- options -------------------------------------------------------
        card = Card("ComfyUI options")
        self.options = OptionsPanel()
        card.body().addWidget(self.options)
        column.addWidget(card)

        # ---- environment ---------------------------------------------------
        card = Card("Environment inside WSL")
        hint = QLabel("Variables set before ComfyUI starts. $NAME is replaced by bash, so "
                      "$CUDA_HOME/bin:$PATH works. Rows are applied from top to bottom.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        card.body().addWidget(hint)
        self.env = EnvTable()
        card.body().addWidget(self.env)
        card.body().addWidget(_label("Commands to run first (bash, optional)",
                                     "Run inside the ComfyUI folder after the variables above are set, "
                                     "for example: source ~/my-env.sh"))
        self.pre_commands = QPlainTextEdit()
        self.pre_commands.setFont(theme.mono_font())
        self.pre_commands.setPlaceholderText("for example: source ~/my-env.sh")
        self.pre_commands.setFixedHeight(64)
        self.pre_commands.setTabChangesFocus(True)
        card.body().addWidget(self.pre_commands)
        column.addWidget(card)

        # ---- extra arguments and preview -----------------------------------
        card = Card("Command")
        card.body().addWidget(_label("Extra arguments (added at the end, as you would type them in a terminal)"))
        self.extra_args = QLineEdit()
        self.extra_args.setFont(theme.mono_font())
        self.extra_args.setPlaceholderText("anything the option list does not cover")
        card.body().addWidget(self.extra_args)
        row = QHBoxLayout()
        row.addWidget(_label("This is what will run inside WSL"))
        row.addStretch(1)
        copy = QPushButton("Copy")
        copy.setProperty("flat", True)
        copy.setProperty("controlSize", "sm")
        copy.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        copy.clicked.connect(self._copy_preview)
        row.addWidget(copy)
        card.body().addLayout(row)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(theme.mono_font())
        self.preview.setFixedHeight(84)
        card.body().addWidget(self.preview)
        self.issues = QLabel()
        self.issues.setWordWrap(True)
        self.issues.setTextFormat(Qt.TextFormat.RichText)
        self.issues.setVisible(False)
        card.body().addWidget(self.issues)
        column.addWidget(card)
        column.addStretch(1)

        # ---- wiring --------------------------------------------------------
        self.name.textChanged.connect(self._on_field)
        self.comfy_dir.textChanged.connect(self._on_field)
        self.python.textChanged.connect(self._on_field)
        self.user.textChanged.connect(self._on_field)
        self.extra_args.textChanged.connect(self._on_field)
        self.pre_commands.textChanged.connect(self._on_field)
        self.port.textChanged.connect(self._on_field)
        self.distro.currentIndexChanged.connect(self._on_field)
        self.listen.currentIndexChanged.connect(self._on_field)
        self.listen.editTextChanged.connect(self._on_field)
        self.auto_open.toggled.connect(self._on_field)
        self.open_rulzflow.toggled.connect(self._on_field)
        self.gpus.changed.connect(self.changed)
        self.options.changed.connect(self.changed)
        self.options.refresh_requested.connect(self.refresh_catalog_requested)
        self.env.changed.connect(self._on_field)
        self.env.import_requested.connect(self.import_env_requested)
        self.set_profile(None)

    def retheme(self):
        """Colours inside rich text come from the active palette, so draw them again."""
        self._show_issues()

    # -- data in ------------------------------------------------------------
    def profile(self) -> Profile | None:
        return self._profile

    def set_profile(self, profile: Profile | None):
        self._profile = profile
        self._loading = True
        self.setEnabled(profile is not None)
        p = profile or Profile(name="")
        self.name.setText(p.name)
        self.comfy_dir.setText(p.comfy_dir)
        self.python.setText(p.python)
        self.user.setText(p.user)
        self.port.setText(str(p.port))
        self._fill_distros(p.distro)
        index = self.listen.findData(p.listen)
        if index >= 0:
            self.listen.setCurrentIndex(index)
        else:
            self.listen.setCurrentIndex(-1)
            self.listen.setEditText(p.listen)
        if self.listen.lineEdit() is not None:
            self.listen.lineEdit().setCursorPosition(0)   # show the start of the address, not its end
        self.auto_open.setChecked(p.auto_open)
        self.open_rulzflow.setChecked(p.open_rulzflow)
        self.extra_args.setText(p.extra_args)
        self.pre_commands.setPlainText(p.pre_commands)
        self.env.set_rows(p.env)
        self.gpus.set_profile(profile)
        self.options.set_options(profile.options if profile else None)
        self._loading = False

    def set_catalog(self, catalog: Catalog):
        if self.options.catalog() is not catalog:
            self.options.set_catalog(catalog)

    def set_gpus(self, gpus: list[GpuInfo]):
        self.gpus.set_gpus(gpus)

    def set_distros(self, distros: list[str]):
        self._distros = list(distros)
        self._loading = True
        self._fill_distros(self._profile.distro if self._profile else "")
        self._loading = False

    def reload_env(self):
        if self._profile:
            self._loading = True
            self.env.set_rows(self._profile.env)
            self._loading = False

    def set_preview(self, command: str, issues: list[Issue]):
        if self.preview.toPlainText() != command:
            self.preview.setPlainText(command)
        self._issues = list(issues)
        self._show_issues()

    def _show_issues(self):
        if self._issues:
            lines = []
            for issue in self._issues:
                color = theme.C["danger-text" if issue.level == "error" else "warn-text"]
                word = "Problem" if issue.level == "error" else "Note"
                text = issue.message.replace("&", "&amp;").replace("<", "&lt;")
                lines.append(f'<span style="color:{color}"><b>{word}:</b> {text}</span>')
            self.issues.setText("<br>".join(lines))
        self.issues.setVisible(bool(self._issues))

    # -- internals ----------------------------------------------------------
    def _fill_distros(self, current: str):
        self.distro.blockSignals(True)
        self.distro.clear()
        self.distro.addItem("Default distro", "")
        known = self._distros or []
        for name in known:
            self.distro.addItem(name, name)
        if current and current not in known:
            suffix = "   (not installed?)" if self._distros is not None else ""
            self.distro.addItem(current + suffix, current)
        self.distro.setCurrentIndex(max(0, self.distro.findData(current)))
        self.distro.blockSignals(False)

    def _listen_value(self) -> str:
        text = self.listen.currentText().strip()
        for value, label in _LISTEN_CHOICES:
            if text == label:
                return value
        return text.split()[0] if text else "127.0.0.1"

    def _on_field(self, *_):
        if self._loading or self._profile is None:
            return
        p = self._profile
        p.name = self.name.text().strip()
        p.comfy_dir = self.comfy_dir.text().strip()
        p.python = self.python.text().strip()
        p.user = self.user.text().strip()
        try:
            p.port = int(self.port.text())
        except ValueError:
            p.port = 0   # empty while typing; validation reports it
        p.distro = self.distro.currentData() or ""
        p.listen = self._listen_value()
        p.auto_open = self.auto_open.isChecked()
        p.open_rulzflow = self.open_rulzflow.isChecked()
        p.extra_args = self.extra_args.text()
        p.pre_commands = self.pre_commands.toPlainText()
        p.env = self.env.rows()
        self.changed.emit()

    def _copy_preview(self):
        QGuiApplication.clipboard().setText(self.preview.toPlainText())
