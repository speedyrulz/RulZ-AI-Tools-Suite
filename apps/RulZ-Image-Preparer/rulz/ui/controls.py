"""Small shared widgets: an adjustment slider row and the JPEG mode switch."""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QWidget

from ..filters import ADJUSTMENTS, BALANCED, BEST
from . import theme

TIPS = {
    "color": "Removes colored speckles (red/green/blue blotches) from low-light and high-ISO photos. "
             "Only color is smoothed; detail is kept.",
    "jpeg": "Removes JPEG compression damage: 8x8 blocks, halos/ringing around edges and blocky color.",
    "detail": "Sharpens fine detail and edges back up - use it to win back crispness after JPEG "
              "artifact removal. Very small differences (leftover noise) aren't boosted.",
}


class ValueBox(QSpinBox):
    """The number next to a slider: click it and type a value, Enter to apply."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Value")
        self.setRange(0, 100)
        self.setButtonSymbols(QSpinBox.NoButtons)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setKeyboardTracking(False)  # apply on Enter / leaving the box, not every keystroke
        self.setFixedWidth(46)
        self.setToolTip("Click to type a value (0-100), then press Enter")

    def validate(self, text, pos):
        # accept anything up to 3 digits while typing; out-of-range values are capped on commit
        if text == "" or (text.isdigit() and len(text) <= 3):
            return QValidator.Acceptable if text and int(text) <= 100 else QValidator.Intermediate, text, pos
        return QValidator.Invalid, text, pos

    def _commit(self):
        text = self.lineEdit().text()
        if text.isdigit():
            self.setValue(min(100, int(text)))
        self.lineEdit().setText(str(self.value()))

    def event(self, e):
        # keep Enter/Esc for the box while typing, instead of the page's shortcuts (Esc = go back)
        if e.type() == QEvent.ShortcutOverride and e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
            e.accept()
            return True
        return super().event(e)

    def focusInEvent(self, e):
        super().focusInEvent(e)
        QTimer.singleShot(0, self.selectAll)  # so typing replaces the old number

    def focusOutEvent(self, e):
        self._commit()
        super().focusOutEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.lineEdit().setText(str(self.value()))  # throw away what was typed
            self.clearFocus()
        elif e.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._commit()
            self.clearFocus()
        else:
            super().keyPressEvent(e)


class AdjustRow(QWidget):
    """Label + slider + value, with an optional per-image 'back to global' button."""

    changed = Signal(str, int)  # key, value
    reset = Signal(str)

    def __init__(self, key: str, with_reset: bool = False, parent=None, label: str | None = None,
                 tip: str | None = None):
        super().__init__(parent)
        self.key = key
        tip = tip or TIPS[key]
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.label = QLabel(label or ADJUSTMENTS[key][0])
        self.label.setFixedWidth(170)
        self.label.setToolTip(tip)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setMinimumWidth(140)
        self.slider.setToolTip(tip)
        self.slider.valueChanged.connect(self._moved)
        self.value = ValueBox()
        self.value.valueChanged.connect(self.slider.setValue)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.value)
        self.reset_btn = None
        self.state = None
        if with_reset:
            self.state = QLabel(objectName="Muted")
            self.state.setMinimumWidth(110)
            self.reset_btn = QPushButton(theme.icon("reset"), "", clicked=lambda: self.reset.emit(self.key))
            self.reset_btn.setProperty("flat", True)
            self.reset_btn.setToolTip("Use the global value for this image again")
            lay.addWidget(self.state)
            lay.addWidget(self.reset_btn)

    def _show(self, v: int):
        self.value.blockSignals(True)
        self.value.setValue(v)
        self.value.blockSignals(False)

    def _moved(self, v: int):
        self._show(v)
        self.changed.emit(self.key, v)

    def set_value(self, v: int, custom: bool | None = None, global_value: int | None = None):
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self._show(v)
        if self.state is not None and custom is not None:
            self.state.setText(f"Custom · global {global_value}" if custom else "Global")
            self.reset_btn.setEnabled(custom)


class ModeSwitch(QFrame):
    """Balanced | Best quality (AI) segmented switch for JPEG artifact removal."""

    requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Segment")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.balanced = QPushButton("Balanced", checkable=True)
        self.best = QPushButton("Best quality (AI)", checkable=True)
        self.balanced.setToolTip("JPEG removal with a fast classic filter - instant previews")
        self.best.setToolTip("JPEG removal with an AI model (FBCNN) on your graphics card - much cleaner "
                             "results, a few seconds per photo")
        group = QButtonGroup(self)
        for b, mode in ((self.balanced, BALANCED), (self.best, BEST)):
            group.addButton(b)
            lay.addWidget(b)
            b.clicked.connect(lambda _=False, m=mode: self._clicked(m))
        self._mode = BALANCED
        self.balanced.setChecked(True)

    def _clicked(self, mode: str):
        self.set_mode(self._mode)  # stay put until the owner accepts the change
        self.requested.emit(mode)

    def set_mode(self, mode: str):
        self._mode = mode
        (self.best if mode == BEST else self.balanced).setChecked(True)
