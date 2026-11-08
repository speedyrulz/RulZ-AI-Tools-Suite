"""Reusable widgets: bubble sliders, segmented buttons, colour swatches, collapsible panels.

From RulZ Image Distortion Repair; switch and arrow colours follow the Clip Extractor theme.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QFrame, QHBoxLayout, QLabel, QSizePolicy,
                               QSlider, QStyle, QStyleOptionSlider, QToolButton, QVBoxLayout, QWidget)

from ... import theme
from . import icons


class BubbleSlider(QWidget):
    """Label + slider; a value bubble floats above the handle while hovering/dragging.

    ``scale`` converts the integer slider position to the displayed/emitted value (e.g. 0.1 for
    tenths of a degree). Double-click resets to the default. ``valueChanged`` fires continuously,
    ``editFinished`` once a drag or keyboard edit ends.
    """

    valueChanged = Signal(float)
    editFinished = Signal(float)

    def __init__(self, label: str, minimum: int, maximum: int, default: int, scale: float = 1.0,
                 decimals: int = 0, label_width: int = 120, parent=None):
        super().__init__(parent)
        self.default, self.scale, self.decimals = default, scale, decimals
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.label = QLabel(label)
        self.label.setProperty("role", "label")
        self.label.setFixedWidth(label_width)
        if not label:
            self.label.hide()
        self.slider = _Slider(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(default)
        self.slider.setMinimumHeight(26)
        self.slider.setPageStep(max(1, (maximum - minimum) // 20))
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        self.bubble = QLabel(self.window())
        self.bubble.setObjectName("bubble")
        self.bubble.hide()
        self.slider.valueChanged.connect(self._changed)
        self.slider.sliderReleased.connect(lambda: self.editFinished.emit(self.value()))
        self.slider.keyEdited.connect(lambda: self.editFinished.emit(self.value()))
        self.slider.hovered.connect(self._hover)
        self.slider.doubleClicked.connect(self.reset)

    def value(self) -> float:
        return self.slider.value() * self.scale

    def setValue(self, v: float, block: bool = True):
        if block:
            self.slider.blockSignals(True)
        self.slider.setValue(round(v / self.scale))
        if block:
            self.slider.blockSignals(False)

    def reset(self):
        self.slider.setValue(self.default)
        self.editFinished.emit(self.value())

    def _text(self) -> str:
        v = self.value()
        return f"{v:.{self.decimals}f}" if self.decimals else str(int(round(v)))

    def _changed(self, _):
        self.valueChanged.emit(self.value())
        self._hover(True)

    def _hover(self, on: bool):
        if not on and not self.slider.isSliderDown():
            self.bubble.hide()
            return
        if self.bubble.parent() is not self.window():
            self.bubble.setParent(self.window())
        self.bubble.setText(self._text())
        self.bubble.adjustSize()
        opt = QStyleOptionSlider()
        self.slider.initStyleOption(opt)
        handle = self.slider.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self.slider)
        p = self.slider.mapTo(self.window(), QPoint(handle.center().x(), 0))
        self.bubble.move(p.x() - self.bubble.width() // 2, p.y() - self.bubble.height() + 2)
        self.bubble.show()
        self.bubble.raise_()


class _Slider(QSlider):
    hovered = Signal(bool)
    doubleClicked = Signal()
    keyEdited = Signal()

    def enterEvent(self, e):
        self.hovered.emit(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.hovered.emit(False)
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if not self.underMouse():
            self.hovered.emit(False)

    def mouseDoubleClickEvent(self, e):
        self.doubleClicked.emit()

    def mousePressEvent(self, e):
        # jump straight to the clicked position instead of paging
        if e.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
            if not handle.contains(e.position().toPoint()):
                groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
                x = e.position().x() - groove.x()
                v = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), int(x), groove.width())
                self.setValue(v)
        super().mousePressEvent(e)

    def keyReleaseEvent(self, e):
        super().keyReleaseEvent(e)
        if not e.isAutoRepeat():
            self.keyEdited.emit()


class SegmentedButtons(QWidget):
    """A row of joined checkable tool buttons. ``exclusive`` → at most one checked."""

    toggled = Signal(str, bool)

    def __init__(self, items, exclusive: bool = True, allow_none: bool = True, size=QSize(40, 26),
                 icon_px: int = 18, parent=None):
        super().__init__(parent)
        self.exclusive, self.allow_none = exclusive, allow_none
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.buttons: dict[str, QToolButton] = {}
        for i, (key, icon_name, text, tip) in enumerate(items):
            b = QToolButton()
            b.setCheckable(True)
            b.setProperty("segment", True)
            b.setProperty("segFirst", i == 0)
            b.setProperty("segLast", i == len(items) - 1)
            if icon_name:
                b.setIcon(icons.icon(icon_name))
                b.setIconSize(QSize(icon_px, icon_px))
            if text:
                b.setText(text)
                b.setToolButtonStyle(Qt.ToolButtonTextOnly if not icon_name else Qt.ToolButtonTextBesideIcon)
            b.setToolTip(tip or text)
            b.setMinimumSize(size)
            b.clicked.connect(lambda checked, k=key: self._clicked(k, checked))
            lay.addWidget(b)
            self.buttons[key] = b

    def _clicked(self, key, checked):
        if self.exclusive:
            if not checked and not self.allow_none:
                self.buttons[key].setChecked(True)
                return
            for k, b in self.buttons.items():
                if k != key and b.isChecked():
                    b.setChecked(False)
        self.toggled.emit(key, self.buttons[key].isChecked())

    def checked(self) -> str | None:
        for k, b in self.buttons.items():
            if b.isChecked():
                return k
        return None

    def setChecked(self, key: str | None):
        for k, b in self.buttons.items():
            b.blockSignals(True)
            b.setChecked(k == key)
            b.blockSignals(False)


class ColorSwatch(QLabel):
    colorChanged = Signal(QColor)

    def __init__(self, color: QColor, tip: str = "", parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(18, 18)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tip or "Change color")

    def color(self) -> QColor:
        return QColor(self._color)

    def setColor(self, c: QColor):
        self._color = QColor(c)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), self._color)
        p.setPen(QPen(QColor("#777"), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def mousePressEvent(self, e):
        c = QColorDialog.getColor(self._color, self.window(), "Select color")
        if c.isValid():
            self.setColor(c)
            self.colorChanged.emit(c)


class Switch(QCheckBox):
    """Small on/off pill used in panel headers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(22, 12)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Enable / disable this correction")

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        on = self.isChecked()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.RED if on else theme.BORDER))
        p.drawRoundedRect(self.rect().adjusted(0, 1, 0, -1), 4, 4)
        p.setBrush(QColor("#ffffff" if on else theme.MUTED))
        knob = 8
        x = self.width() - knob - 2 if on else 2
        p.drawRoundedRect(x, 2, knob, self.height() - 4, 3, 3)

    def hitButton(self, pos):
        return self.rect().contains(pos)


class ArrowCheck(QCheckBox):
    """Expand/collapse triangle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self.setChecked(True)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.MUTED))
        from PySide6.QtGui import QPolygonF
        from PySide6.QtCore import QPointF
        if self.isChecked():
            pts = [QPointF(2, 4), QPointF(12, 4), QPointF(7, 10)]
        else:
            pts = [QPointF(4, 2), QPointF(10, 7), QPointF(4, 12)]
        p.drawPolygon(QPolygonF(pts))

    def hitButton(self, pos):
        return self.rect().contains(pos)


class CollapsiblePanel(QWidget):
    """Correction panel: [▾][switch]  Title  [extra][↺] / body / optional ± more-options."""

    enabledToggled = Signal(bool)
    resetClicked = Signal()
    moreToggled = Signal(bool)

    def __init__(self, title: str, has_more: bool = False, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QWidget()
        header.setObjectName("panelHeader")
        header.setFixedHeight(32)
        h = QHBoxLayout(header)
        h.setContentsMargins(9, 0, 8, 0)
        h.setSpacing(8)
        self.expand = ArrowCheck()
        self.switch = Switch()
        self.title = QLabel(title)
        self.title.setObjectName("panelTitle")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setProperty("active", False)
        self.extra = QHBoxLayout()
        self.extra.setSpacing(4)
        self.reset = QToolButton()
        self.reset.setIcon(icons.icon("reset"))
        self.reset.setIconSize(QSize(14, 14))
        self.reset.setToolTip("Reset this correction")
        h.addWidget(self.expand)
        h.addWidget(self.switch)
        h.addWidget(self.title, 1)
        h.addLayout(self.extra)
        h.addWidget(self.reset)
        outer.addWidget(header)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(19, 4, 12, 8)
        self.body_layout.setSpacing(8)
        outer.addWidget(self.body)

        self.more_body = QWidget()
        self.more_layout = QVBoxLayout(self.more_body)
        self.more_layout.setContentsMargins(19, 0, 12, 4)
        self.more_layout.setSpacing(6)
        self.more_body.setVisible(False)
        outer.addWidget(self.more_body)

        self.more_btn = None
        if has_more:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addStretch(1)
            self.more_btn = QToolButton()
            self.more_btn.setCheckable(True)
            self.more_btn.setIcon(icons.icon("plus_small", 12))
            self.more_btn.setIconSize(QSize(10, 10))
            self.more_btn.setFixedSize(13, 13)
            self.more_btn.setToolTip("Show / hide additional options")
            self.more_btn.toggled.connect(self._more)
            row.addWidget(self.more_btn)
            outer.addLayout(row)

        sep = QFrame()
        sep.setObjectName("panelSeparator")
        outer.addWidget(sep)

        self.expand.toggled.connect(self._expand)
        self.switch.toggled.connect(self._switched)
        self.reset.clicked.connect(self.resetClicked)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def _expand(self, on: bool):
        self.body.setVisible(on)
        self.more_body.setVisible(on and bool(self.more_btn and self.more_btn.isChecked()))
        if self.more_btn:
            self.more_btn.setVisible(on)

    def _more(self, on: bool):
        self.more_btn.setIcon(icons.icon("minus_small" if on else "plus_small", 12))
        self.more_body.setVisible(on and self.expand.isChecked())
        self.moreToggled.emit(on)

    def _switched(self, on: bool):
        self._style_title(on)
        self.enabledToggled.emit(on)

    def _style_title(self, on: bool):
        self.title.setProperty("active", on)
        self.title.style().unpolish(self.title)
        self.title.style().polish(self.title)

    def setActive(self, on: bool):
        self.switch.blockSignals(True)
        self.switch.setChecked(on)
        self.switch.blockSignals(False)
        self._style_title(on)
        self.switch.update()

    def add(self, w):
        if isinstance(w, QWidget):
            self.body_layout.addWidget(w)
        else:
            self.body_layout.addLayout(w)
        return w

    def add_more(self, w):
        if isinstance(w, QWidget):
            self.more_layout.addWidget(w)
        else:
            self.more_layout.addLayout(w)
        return w


def row(*widgets, stretch_after: bool = False, spacing: int = 8) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for w in widgets:
        if w is None:
            lay.addStretch(1)
        elif isinstance(w, int):
            lay.addSpacing(w)
        else:
            lay.addWidget(w)
    if stretch_after:
        lay.addStretch(1)
    return lay


def label(text: str, role: str = "label") -> QLabel:
    lb = QLabel(text)
    lb.setProperty("role", role)
    return lb


def text_button(text: str, checkable: bool = True, tip: str = "") -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setCheckable(checkable)
    b.setProperty("segment", True)
    b.setProperty("segFirst", True)
    b.setProperty("segLast", True)
    b.setMinimumSize(QSize(48, 26))
    b.setToolTip(tip or text)
    return b
