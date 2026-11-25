"""The list of ComfyUI command line options, with a tick box for each one."""
from __future__ import annotations

from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout, QWidget)

from ..model import OptionState
from ..options import (CATEGORY_ORDER, MANAGED_OPTIONS, UNKNOWN_CATEGORY, Catalog, OptionSpec,
                       unknown_spec)
from . import theme
from .widgets import CollapsibleSection, NoWheelComboBox


def _placeholder(spec: OptionSpec) -> str:
    default = spec.default_text()
    if spec.kind == "optional":
        const = "" if spec.const is None else str(spec.const)
        return f"optional (uses {const} when empty)" if const else "optional"
    if spec.kind == "multi":
        if spec.choices:
            lead = "empty = all, or: " if spec.min_values == 0 else ""
            return lead + " ".join(spec.choices)
        if spec.metavar:
            return f"{spec.metavar}  (separate several with spaces)"
        return "separate several with spaces"
    if spec.kind == "value" and spec.repeat:
        return f"{spec.metavar or 'value'}  (separate several with spaces)"
    if default:
        return f"default: {default}"
    if spec.type == "path":
        return "Linux path, for example /mnt/f/output"
    return spec.metavar or ""


class OptionRow(QWidget):
    changed = Signal(str)   # option name

    def __init__(self, spec: OptionSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._loading = False
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 3, 0, 3)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(1)
        grid.setColumnStretch(1, 1)

        self.check = QCheckBox(spec.name)
        self.check.setFont(theme.mono_font())
        self.check.setMinimumWidth(270)
        self.check.toggled.connect(self._on_toggled)
        grid.addWidget(self.check, 0, 0)

        self.combo: NoWheelComboBox | None = None
        self.edit: QLineEdit | None = None
        if spec.kind != "flag":
            if spec.choices and spec.kind in ("value", "optional") and not spec.repeat:
                self.combo = NoWheelComboBox()
                self.combo.addItems(list(spec.choices))
                self.combo.setMaximumWidth(260)
                self.combo.currentTextChanged.connect(self._on_edited)
                grid.addWidget(self.combo, 0, 1, Qt.AlignmentFlag.AlignLeft)
            else:
                self.edit = QLineEdit()
                self.edit.setPlaceholderText(_placeholder(spec))
                if spec.kind in ("value", "optional") and not spec.repeat:
                    if spec.type == "int":
                        self.edit.setValidator(QIntValidator(self.edit))
                        self.edit.setMaximumWidth(260)
                    elif spec.type == "float":
                        validator = QDoubleValidator(self.edit)
                        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
                        validator.setLocale(QLocale.c())   # always "." as the decimal point
                        self.edit.setValidator(validator)
                        self.edit.setMaximumWidth(260)
                self.edit.textChanged.connect(self._on_edited)
                if self.edit.maximumWidth() < 1000:
                    self.edit.setMinimumWidth(200)
                    grid.addWidget(self.edit, 0, 1, Qt.AlignmentFlag.AlignLeft)
                else:
                    grid.addWidget(self.edit, 0, 1)

        help_text = spec.help
        self.help = QLabel(help_text)
        self.help.setObjectName("hint")
        self.help.setWordWrap(True)
        self.help.setContentsMargins(24, 0, 0, 2)
        # Added to the layout before it is shown: a widget that is made visible
        # while it has no parent opens as a window of its own for a moment.
        grid.addWidget(self.help, 1, 0, 1, 2)
        self.help.setVisible(bool(help_text))
        self._sync_editor()
        self._search_text = (spec.name + " " + spec.help).lower()

    def set_exclusive_hint(self, names: list[str]):
        if names:
            self.check.setToolTip("Only one of these can be on at a time:\n" + "\n".join([self.spec.name, *names]))

    def matches(self, needle: str) -> bool:
        return not needle or needle in self._search_text

    def is_on(self) -> bool:
        return self.check.isChecked()

    def state(self) -> OptionState:
        if self.combo is not None:
            value = self.combo.currentText()
        elif self.edit is not None:
            value = self.edit.text()
        else:
            value = ""
        return OptionState(enabled=self.check.isChecked(), value=value)

    def set_state(self, state: OptionState | None):
        self._loading = True
        state = state or OptionState()
        self.check.setChecked(state.enabled)
        if self.combo is not None:
            index = self.combo.findText(state.value)
            if index < 0:
                default = self.spec.default_text()
                index = max(0, self.combo.findText(default))
            self.combo.setCurrentIndex(index)
        elif self.edit is not None:
            self.edit.setText(state.value)
        self._loading = False
        self._sync_editor()

    def set_checked_silently(self, checked: bool):
        self._loading = True
        self.check.setChecked(checked)
        self._loading = False
        self._sync_editor()

    def _sync_editor(self):
        on = self.check.isChecked()
        for editor in (self.combo, self.edit):
            if editor is not None:
                editor.setEnabled(on)

    def _on_toggled(self, _checked):
        self._sync_editor()
        if not self._loading:
            self.changed.emit(self.spec.name)
            if self.check.isChecked() and self.edit is not None and not self.edit.text():
                self.edit.setFocus()

    def _on_edited(self, _text):
        if not self._loading:
            self.changed.emit(self.spec.name)


class OptionsPanel(QWidget):
    changed = Signal()
    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._catalog: Catalog | None = None
        self._options: dict[str, OptionState] | None = None
        self._rows: dict[str, OptionRow] = {}
        self._row_order: list[str] = []
        self._sections: dict[str, CollapsibleSection] = {}
        self._section_rows: dict[str, list[OptionRow]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search options, for example vram, attention, preview ...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        self.only_enabled = QCheckBox("Only show enabled")
        self.only_enabled.toggled.connect(self._apply_filter)
        bar.addWidget(self.search, 1)
        bar.addWidget(self.only_enabled)
        layout.addLayout(bar)

        source = QHBoxLayout()
        self.source_label = QLabel()
        self.source_label.setObjectName("hint")
        self.refresh_button = QPushButton("Re-read from ComfyUI")
        self.refresh_button.setProperty("flat", True)
        self.refresh_button.setProperty("controlSize", "sm")
        self.refresh_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.refresh_button.setToolTip("Ask this ComfyUI install which options it supports. "
                                       "Use it after updating ComfyUI.")
        self.refresh_button.clicked.connect(self.refresh_requested)
        source.addWidget(self.source_label, 1)
        source.addWidget(self.refresh_button)
        layout.addLayout(source)

        self._host = QWidget()
        self._host_layout = QVBoxLayout(self._host)
        self._host_layout.setContentsMargins(0, 0, 0, 0)
        self._host_layout.setSpacing(0)
        layout.addWidget(self._host)

        self.empty_label = QLabel("No option matches.")
        self.empty_label.setObjectName("muted")
        self.empty_label.setVisible(False)
        layout.addWidget(self.empty_label)

    # -- data in ------------------------------------------------------------
    def set_busy(self, busy: bool):
        self.refresh_button.setEnabled(not busy)
        self.refresh_button.setText("Reading ..." if busy else "Re-read from ComfyUI")

    def set_catalog(self, catalog: Catalog):
        self._catalog = catalog
        self._build()

    def set_options(self, options: dict[str, OptionState] | None):
        """Show a profile's options. The dictionary is edited in place."""
        self._options = options
        if self._rows and [s.name for s in self._specs()] == list(self._row_order):
            # Same set of rows as before: only the ticks and values change.
            for name, row in self._rows.items():
                row.set_state(options.get(name) if options else None)
                row.setEnabled(options is not None)
            for category, section in self._sections.items():
                section.set_expanded(any(r.is_on() for r in self._section_rows[category]))
            self._update_badges()
            self._apply_filter()
            return
        self._build()

    def catalog(self) -> Catalog | None:
        return self._catalog

    # -- building -----------------------------------------------------------
    def _specs(self) -> list[OptionSpec]:
        catalog = self._catalog
        specs = list(catalog.visible()) if catalog else []
        if catalog and self._options:
            for name in sorted(self._options):
                state = self._options[name]
                if (state.enabled or state.value) and catalog.get(name) is None \
                        and name not in MANAGED_OPTIONS:
                    specs.append(unknown_spec(name, bool(state.value.strip())))
        return specs

    def _build(self):
        expanded = {name for name, section in self._sections.items() if section.is_expanded()}
        first_build = not self._sections
        while self._host_layout.count():
            item = self._host_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()
        self._sections.clear()
        self._section_rows.clear()
        catalog = self._catalog
        if catalog is None:
            return
        self.source_label.setText(f"Option list: {catalog.label()}")

        by_category: dict[str, list[OptionSpec]] = {}
        specs = self._specs()
        self._row_order = [spec.name for spec in specs]
        for spec in specs:
            by_category.setdefault(spec.category, []).append(spec)
        order = [UNKNOWN_CATEGORY] + CATEGORY_ORDER
        for category in sorted(by_category, key=lambda c: order.index(c) if c in order else len(order)):
            section = CollapsibleSection(category)
            rows = []
            for spec in by_category[category]:
                row = OptionRow(spec)
                row.set_exclusive_hint(catalog.exclusive_with(spec.name))
                row.set_state(self._options.get(spec.name) if self._options else None)
                row.setEnabled(self._options is not None)
                row.changed.connect(self._on_row_changed)
                section.content_layout.addWidget(row)
                self._rows[spec.name] = row
                rows.append(row)
            self._sections[category] = section
            self._section_rows[category] = rows
            self._host_layout.addWidget(section)

        for category, section in self._sections.items():
            has_enabled = any(r.is_on() for r in self._section_rows[category])
            section.set_expanded(has_enabled or (category in expanded and not first_build))
        self._update_badges()
        self._apply_filter()

    # -- edits --------------------------------------------------------------
    def _on_row_changed(self, name: str):
        if self._options is None or self._catalog is None:
            return
        row = self._rows[name]
        state = row.state()
        if state.enabled:
            # Options in one mutually exclusive group behave like radio buttons.
            for other in self._catalog.exclusive_with(name):
                other_row = self._rows.get(other)
                if other_row is not None and other_row.is_on():
                    other_row.set_checked_silently(False)
                    self._store(other, other_row.state())
        self._store(name, state)
        self._update_badges()
        if self.only_enabled.isChecked():
            self._apply_filter()
        self.changed.emit()

    def _store(self, name: str, state: OptionState):
        if self._options is None:
            return
        if state.enabled or state.value.strip():
            self._options[name] = state
        else:
            self._options.pop(name, None)

    def _update_badges(self):
        for category, section in self._sections.items():
            count = sum(1 for r in self._section_rows[category] if r.is_on())
            section.set_badge(f"{count} on" if count else "")

    def _apply_filter(self, *_):
        needle = self.search.text().strip().lower()
        only_enabled = self.only_enabled.isChecked()
        any_visible = False
        for category, section in self._sections.items():
            visible_rows = 0
            for row in self._section_rows[category]:
                show = row.matches(needle) and (row.is_on() or not only_enabled)
                row.setVisible(show)
                visible_rows += show
            section.setVisible(visible_rows > 0)
            if visible_rows and (needle or only_enabled):
                section.set_expanded(True)
            any_visible = any_visible or visible_rows > 0
        self.empty_label.setVisible(not any_visible and bool(self._sections))
