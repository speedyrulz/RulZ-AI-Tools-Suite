"""Edit many captions at once: find and replace, or add text at the start or the end."""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QRadioButton, QVBoxLayout)

from . import edits

COUNT_DELAY_MS = 200  # the count of what would change follows the typing after this pause


class BulkEditDialog(QDialog):
    """`captions(scope)` gives [(row, caption)] for "selected" or "shown"; `apply(changes, label)` saves
    [(row, new caption)] and returns (captions changed, error messages). The dialog stays open, so
    several edits can be made one after another; each is one undo step."""

    def __init__(self, captions, apply, selected, shown, action=edits.REPLACE, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Captions")
        self.setMinimumWidth(520)
        self.captions, self.apply_changes = captions, apply

        self.action = QComboBox()
        for key, label in edits.ACTIONS:
            self.action.addItem(label, key)
        self.action.setCurrentIndex(max(0, self.action.findData(action)))
        self.find_box = QLineEdit()
        self.text_box = QLineEdit()
        self.case = QCheckBox("Match case")
        self.words = QCheckBox("Whole words")
        self.words.setToolTip("\"cat\" is found in \"a cat sits\", not in \"category\".")
        self.regex = QCheckBox("Regular expression")
        self.regex.setToolTip("Find is a Python regular expression; the replacement may use \\1, \\2… "
                              "for its groups.")
        self.r_selected = QRadioButton(f"The {selected:,} selected image(s)")
        self.r_shown = QRadioButton(f"All {shown:,} image(s) shown in the grid")
        self.r_selected.setEnabled(selected > 0)
        (self.r_selected if selected > 1 else self.r_shown).setChecked(True)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setMinimumHeight(self.note.fontMetrics().lineSpacing() * 2)

        self.form = form = QFormLayout()
        form.addRow("Action", self.action)
        form.addRow("Find", self.find_box)
        form.addRow("Replace with", self.text_box)
        opts = QHBoxLayout()
        for w in (self.case, self.words, self.regex):
            opts.addWidget(w)
        opts.addStretch(1)
        self.opts_row = opts
        form.addRow("Options", opts)
        scope = QVBoxLayout()
        scope.addWidget(self.r_selected)
        scope.addWidget(self.r_shown)
        form.addRow("Apply to", scope)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        self.apply_btn = self.buttons.button(QDialogButtonBox.Apply)
        self.apply_btn.setObjectName("primary")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self.run)
        self.buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.note)
        lay.addWidget(self.buttons)

        self.timer = QTimer(self, singleShot=True, interval=COUNT_DELAY_MS)
        self.timer.timeout.connect(self.count)
        self.action.currentIndexChanged.connect(self._action_changed)
        for w in (self.find_box, self.text_box):
            w.textChanged.connect(lambda _: self._typed())
        for w in (self.case, self.words, self.regex, self.r_selected, self.r_shown):
            w.toggled.connect(lambda _: self._typed())
        self.last = ""  # what the last Apply did, shown until something is typed
        self._action_changed()

    def _action_changed(self, *_):
        replace = self.action.currentData() == edits.REPLACE
        self.form.setRowVisible(self.find_box, replace)
        self.form.setRowVisible(self.opts_row, replace)
        self.form.labelForField(self.text_box).setText("Replace with" if replace else "Text")
        self.text_box.setPlaceholderText("leave empty to remove what is found" if replace else
                                     "e.g. ohwx person, " if self.action.currentData() == edits.START
                                     else "e.g. , film grain")
        (self.find_box if replace else self.text_box).setFocus()
        self._typed()

    def _typed(self):
        self.last = ""
        self.timer.start()

    def scope(self):
        return "selected" if self.r_selected.isChecked() else "shown"

    def _plan(self):
        """([(row, new caption)], places changed, captions looked at), or an EditError's text."""
        try:
            edit = edits.make_edit(self.action.currentData(), self.find_box.text(), self.text_box.text(),
                                   self.case.isChecked(), self.words.isChecked(), self.regex.isChecked())
        except edits.EditError as e:
            return str(e)
        captions = self.captions(self.scope())
        changes, places = edits.plan(captions, edit)
        return changes, places, len(captions)

    def count(self):
        """Say what Apply would do, and switch it off when that is nothing."""
        self.timer.stop()
        plan = self._plan()
        if isinstance(plan, str):
            empty = not (self.find_box.text() if self.action.currentData() == edits.REPLACE else self.text_box.text())
            self.note.setText(self.last or ("" if empty else plan))
            self.apply_btn.setEnabled(False)
            return
        changes, places, total = plan
        self.apply_btn.setEnabled(bool(changes))
        if self.action.currentData() == edits.REPLACE:
            now = f"{places:,} match(es) in {len(changes):,} of {total:,} caption(s)."
        else:
            now = (f"{len(changes):,} of {total:,} caption(s) get the text. Empty captions and ones that "
                   "already have it are left as they are.")
        self.note.setText(f"{self.last}\n{now}" if self.last else now)

    def run(self):
        plan = self._plan()
        if isinstance(plan, str) or not plan[0]:
            return self.count()
        label = dict(edits.ACTIONS)[self.action.currentData()].lower()
        changed, errors = self.apply_changes(plan[0], label)
        self.last = f"Changed {changed:,} caption(s). Ctrl+Z in the main window takes it back."
        if errors:
            self.last += f" {len(errors):,} couldn't be saved (see View ▸ Log)."
        self.count()
