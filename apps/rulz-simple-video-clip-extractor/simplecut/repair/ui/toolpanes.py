"""Bottom tool panes (shown while a canvas tool is active) and floating warp options.

Copied unchanged from RulZ Image Distortion Repair (package viewpointer).
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QStackedWidget,
                               QVBoxLayout, QWidget)

from . import icons
from .widgets import BubbleSlider, ColorSwatch, SegmentedButtons, label


class ToolPane(QWidget):
    """[icon] | content … [Automatic preview] [Preview] [Reset] [Apply]"""

    previewClicked = Signal()
    resetClicked = Signal()
    applyClicked = Signal()
    autoPreviewToggled = Signal(bool)

    def __init__(self, icon_name: str, preview: bool = False):
        super().__init__()
        self.setObjectName("toolPane")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(50)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 14, 0)
        lay.setSpacing(12)
        ic = QLabel()
        ic.setPixmap(icons.pixmap(icon_name, 22))
        lay.addWidget(ic)
        div = QFrame()
        div.setObjectName("toolPaneDivider")
        div.setFixedWidth(1)
        lay.addWidget(div)
        self.content = QHBoxLayout()
        self.content.setSpacing(10)
        lay.addLayout(self.content)
        lay.addStretch(1)
        self.auto_preview = self.preview = None
        if preview:
            self.auto_preview = QCheckBox("Automatic preview")
            self.auto_preview.toggled.connect(self.autoPreviewToggled)
            self.preview = QPushButton("Preview")
            self.preview.setProperty("toolPane", True)
            self.preview.setCheckable(True)
            self.preview.clicked.connect(self.previewClicked)
            lay.addWidget(self.auto_preview)
            lay.addWidget(self.preview)
        self.reset = QPushButton("Reset")
        self.reset.setProperty("toolPane", True)
        self.apply = QPushButton("Apply")
        self.apply.setObjectName("applyButton")
        self.apply.setProperty("toolPane", True)
        self.reset.clicked.connect(self.resetClicked)
        self.apply.clicked.connect(self.applyClicked)
        lay.addWidget(self.reset)
        lay.addWidget(self.apply)

    def add(self, w):
        if isinstance(w, QWidget):
            self.content.addWidget(w)
        else:
            self.content.addLayout(w)
        return w


def color_row(text: str, color: QColor, tip: str = ""):
    lay = QHBoxLayout()
    lay.setSpacing(8)
    lb = label(text)
    sw = ColorSwatch(color, tip)
    lay.addWidget(lb)
    lay.addWidget(sw)
    return lay, sw


class PerspectivePane(ToolPane):
    def __init__(self, cv: QColor, ch: QColor):
        super().__init__("persp_eight", preview=True)
        self.add(label("Line color"))
        self.v_icon = QLabel()
        self.v_icon.setPixmap(icons.pixmap("persp_verticals", 18))
        self.swatch_v = ColorSwatch(cv, "Vertical lines color")
        self.h_icon = QLabel()
        self.h_icon.setPixmap(icons.pixmap("persp_horizontals", 18))
        self.swatch_h = ColorSwatch(ch, "Horizontal lines color")
        for w in (self.v_icon, self.swatch_v, self.h_icon, self.swatch_h):
            self.add(w)

    def set_mode(self, mode: str):
        two = mode == "eight"
        self.v_icon.setVisible(two)
        self.h_icon.setVisible(two)
        self.swatch_h.setVisible(two)


class HorizonPane(ToolPane):
    def __init__(self, color: QColor):
        super().__init__("horizon_h", preview=True)
        lay, self.swatch = color_row("Line color", color)
        self.add(lay)
        self.angle_label = label("")
        self.add(self.angle_label)
        self.auto_preview.hide()


class CropPane(ToolPane):
    def __init__(self, opacity: int, show_grid: bool):
        super().__init__("crop")
        self.show_grid = QCheckBox("Show grid")
        self.show_grid.setChecked(show_grid)
        self.add(self.show_grid)
        self.add(QLabel("   "))
        self.opacity = BubbleSlider("Mask opacity", 0, 100, opacity, label_width=86)
        self.opacity.setFixedWidth(240)
        self.add(self.opacity)


class WarpPane(ToolPane):
    modeChanged = Signal(str)

    def __init__(self, color: QColor):
        super().__init__("grid4")
        lay, self.swatch = color_row("Grid Color", color)
        self.add(lay)
        self.add(label("Mode"))
        self.modes = SegmentedButtons([("free", "warp_free", "", "Free transform"),
                                       ("persp", "warp_persp", "", "Perspective"),
                                       ("anamorph", "warp_anamorph", "", "Anamorphosis")],
                                      allow_none=False, size=QSize(40, 26))
        self.modes.setChecked("free")
        self.modes.toggled.connect(lambda k, on: self.modeChanged.emit(k))
        self.add(self.modes)
        self.show_settings = QCheckBox("Show settings")
        self.show_settings.setChecked(True)
        self.add(self.show_settings)


class TiltShiftPane(ToolPane):
    def __init__(self, color: QColor):
        super().__init__("tiltshift")
        lay, self.swatch = color_row("Line color", color)
        self.add(lay)


class GuidesPane(ToolPane):
    orientationChanged = Signal(str)

    def __init__(self, color: QColor):
        super().__init__("guides")
        lay, self.swatch = color_row("Line color", color)
        self.add(lay)
        self.kind = SegmentedButtons([("v", "persp_verticals", "", "Create vertical guides"),
                                      ("h", "persp_horizontals", "", "Create horizontal guides")],
                                     allow_none=False, size=QSize(46, 28))
        self.kind.setChecked("v")
        self.kind.toggled.connect(lambda k, on: self.orientationChanged.emit(k))
        self.add(self.kind)
        self.add(label("Click the image to add a guide; drag a guide off the image to delete it."))


# ----------------------------------------------------------------------------- floating

class FloatingWarpOptions(QWidget):
    """Mode-specific options floating over the bottom-left of the image."""

    subChanged = Signal(str)
    stickyToggled = Signal(bool)
    propagationChanged = Signal(int)
    perspAction = Signal(str)
    perspIntensity = Signal(int)
    anamorphChanged = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("floatingOptions")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedWidth(330)
        self.stack = QStackedWidget()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.addWidget(self.stack)
        self._title = {}

        # free transform
        free = QWidget()
        fl = QVBoxLayout(free)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)
        self.free_title = QLabel("Move")
        self.free_title.setObjectName("floatingTitle")
        self.sub = SegmentedButtons([("move", "move", "", "Move"), ("rotate", "rotate", "", "Rotate"),
                                     ("scale", "scale", "", "Scale")], allow_none=False, size=QSize(100, 26))
        self.sub.setChecked("move")
        self.sub.toggled.connect(self._sub)
        fl.addWidget(self.free_title)
        fl.addWidget(self.sub)
        self.sticky_free = QCheckBox("Sticky borders")
        self.sticky_free.toggled.connect(self._sticky)
        fl.addWidget(self.sticky_free)
        self.propagation = BubbleSlider("Propagation", 0, 200, 0, label_width=90)
        self.propagation.valueChanged.connect(lambda v: self.propagationChanged.emit(int(v)))
        fl.addWidget(self.propagation)
        self.stack.addWidget(free)

        # perspective
        persp = QWidget()
        pl = QVBoxLayout(persp)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(6)
        t = QLabel("Perspective")
        t.setObjectName("floatingTitle")
        help_ = QLabel("Select grid points to define a region (drag a box), then drag its corners.")
        help_.setWordWrap(True)
        help_.setObjectName("sizeLabel")
        pl.addWidget(t)
        pl.addWidget(help_)
        bl = QHBoxLayout()
        for key, text in (("horizon", "Horizon"), ("parallel", "Force parallel")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=key: self.perspAction.emit(k))
            bl.addWidget(b)
        pl.addLayout(bl)
        self.sticky_persp = QCheckBox("Sticky borders")
        self.sticky_persp.toggled.connect(self._sticky)
        pl.addWidget(self.sticky_persp)
        self.persp_int = BubbleSlider("Intensity", 0, 200, 100, label_width=90)
        self.persp_int.valueChanged.connect(lambda v: self.perspIntensity.emit(int(v)))
        pl.addWidget(self.persp_int)
        self.stack.addWidget(persp)

        # anamorphosis
        ana = QWidget()
        al = QVBoxLayout(ana)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(6)
        t = QLabel("Anamorphosis")
        t.setObjectName("floatingTitle")
        al.addWidget(t)
        self.ana_tab = SegmentedButtons([("hv", None, "H / V", "Horizontal / vertical"),
                                         ("diag", None, "Diagonal", "Diagonal")], allow_none=False, size=QSize(150, 24))
        self.ana_tab.setChecked("hv")
        self.ana_tab.toggled.connect(lambda k, on: self._ana_tab(k))
        al.addWidget(self.ana_tab)
        self.ana_h = BubbleSlider("Horizontal", 0, 200, 100, label_width=90)
        self.ana_v = BubbleSlider("Vertical", 0, 200, 100, label_width=90)
        self.ana_d = BubbleSlider("Diagonal", 0, 200, 100, label_width=90)
        self.ana_i = BubbleSlider("Intensity", 0, 200, 100, label_width=90)
        for w in (self.ana_h, self.ana_v, self.ana_d, self.ana_i):
            al.addWidget(w)
            w.valueChanged.connect(lambda _v: self._ana())
        self.sticky_ana = QCheckBox("Sticky borders")
        self.sticky_ana.toggled.connect(self._sticky)
        al.addWidget(self.sticky_ana)
        self.ana_d.hide()
        self.stack.addWidget(ana)

    def set_mode(self, mode: str):
        self.stack.setCurrentIndex({"free": 0, "persp": 1, "anamorph": 2}[mode])
        page = self.stack.currentWidget()
        self.stack.setFixedHeight(page.sizeHint().height())
        self.adjustSize()

    def _sub(self, key, on):
        self.free_title.setText(key.capitalize())
        self.subChanged.emit(key)

    def _sticky(self, on):
        for cb in (self.sticky_free, self.sticky_persp, self.sticky_ana):
            cb.blockSignals(True)
            cb.setChecked(on)
            cb.blockSignals(False)
        self.stickyToggled.emit(on)

    def _ana_tab(self, k):
        self.ana_h.setVisible(k == "hv")
        self.ana_v.setVisible(k == "hv")
        self.ana_d.setVisible(k == "diag")
        self.set_mode("anamorph")
        self._ana()

    def _ana(self):
        self.anamorphChanged.emit({"tab": "hv" if self.ana_tab.checked() == "hv" else "diag",
                                   "h": int(self.ana_h.value()), "v": int(self.ana_v.value()),
                                   "d": int(self.ana_d.value()), "intensity": int(self.ana_i.value())})

    def reset_anamorph(self):
        for w in (self.ana_h, self.ana_v, self.ana_d, self.ana_i):
            w.setValue(100)
