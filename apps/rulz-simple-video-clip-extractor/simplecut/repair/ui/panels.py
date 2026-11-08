"""Right-hand settings dock: one collapsible panel per correction module.

Copied unchanged from RulZ Image Distortion Repair (package viewpointer).
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QLabel, QScrollArea,
                               QSpinBox, QToolButton, QVBoxLayout, QWidget)

from ..core.tiltshift import BLUR_SHAPES
from . import icons
from .widgets import BubbleSlider, CollapsiblePanel, SegmentedButtons, label, row, text_button

ASPECTS = [("unconstrained", "Unconstrained"), ("preserve", "Preserve"), ("16/9", "16/9"), ("5/4", "5/4"),
           ("5/2", "5/2"), ("2/1", "2/1"), ("3/2", "3/2"), ("4/3", "4/3"), ("1/1", "1/1")]
AUTO_MODES = [("both", "Verticals & Horizontals"), ("verticals", "Verticals only"), ("horizontals", "Horizontals only")]


class BasePanel(CollapsiblePanel):
    """Panels write into ``ctl.c`` (the current Corrections) and notify the controller."""

    module = ""

    def __init__(self, ctl, title: str, has_more=False):
        super().__init__(title, has_more)
        self.ctl = ctl
        self.enabledToggled.connect(self._toggle_enabled)
        self.resetClicked.connect(lambda: ctl.reset_module(self.module))

    @property
    def p(self):
        return getattr(self.ctl.c, self.module)

    def _toggle_enabled(self, on: bool):
        self.p.enabled = on
        self.ctl.edited(self.module, commit=True)

    def touch(self, commit: bool):
        """A control changed: make sure the module is on, then notify."""
        if not self.p.enabled:
            self.p.enabled = True
            self.setActive(True)
        self.ctl.edited(self.module, commit=commit)

    def slider(self, text, lo, hi, default, attr, scale=1.0, decimals=0, cast=int):
        s = BubbleSlider(text, lo, hi, default, scale, decimals)

        def changed(v, commit):
            setattr(self.p, attr, cast(v))
            self.touch(commit)
        s.valueChanged.connect(lambda v: changed(v, False))
        s.editFinished.connect(lambda v: changed(v, True))
        return s

    def load(self):
        self.setActive(self.p.enabled)


# ----------------------------------------------------------------------------- distortion

class DistortionPanel(BasePanel):
    module = "distortion"

    def __init__(self, ctl):
        super().__init__(ctl, "Distortion", has_more=False)
        trash = QToolButton()
        trash.setIcon(icons.icon("trash"))
        trash.setIconSize(QSize(13, 13))
        trash.setEnabled(False)
        trash.setToolTip("Delete current lens profile")
        self.extra.addWidget(trash)
        self.auto = text_button("Auto", tip="Automatic correction with a lens profile")
        self.auto.setEnabled(False)
        self.manual = QToolButton()
        self.manual.setCheckable(True)
        self.manual.setProperty("segment", True)
        self.manual.setProperty("segFirst", True)
        self.manual.setProperty("segLast", True)
        self.manual.setIcon(icons.icon("distortion_manual"))
        self.manual.setIconSize(QSize(18, 18))
        self.manual.setMinimumSize(40, 26)
        self.manual.setToolTip("Manual distortion correction")
        self.add(row(self.auto, self.manual, None))

        self.info = QWidget()
        self.info.setObjectName("infoBox")
        il = QVBoxLayout(self.info)
        il.setContentsMargins(10, 6, 6, 6)
        msg = QLabel("No lens profile is available to correct this image automatically. "
                     "Use the manual correction instead.")
        msg.setWordWrap(True)
        ok = QToolButton()
        ok.setText("OK")
        ok.setIcon(icons.icon("ok_circle"))
        ok.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        ok.clicked.connect(self.info.hide)
        il.addWidget(msg)
        il.addLayout(row(ok, None))
        self.add(self.info)

        self.manual_box = QWidget()
        ml = QVBoxLayout(self.manual_box)
        ml.setContentsMargins(0, 0, 0, 0)
        self.intensity = self.slider("Intensity", 0, 100, 50, "intensity")
        self.kind = SegmentedButtons([("barrel", None, "Barrel", "Barrel distortion"),
                                      ("pincushion", None, "Pincushion", "Pincushion distortion"),
                                      ("fisheye", None, "Fisheye", "Fisheye distortion")],
                                     allow_none=False, size=QSize(80, 26))
        self.kind.toggled.connect(self._kind)
        ml.addWidget(self.intensity)
        ml.addLayout(row(self.kind, None))
        self.add(self.manual_box)
        self.manual.toggled.connect(self._manual)

    def _manual(self, on):
        self.manual_box.setVisible(on)
        self.p.enabled = on
        self.setActive(on)
        if on:
            self.info.hide()
        self.ctl.edited(self.module, commit=True)

    def _kind(self, key, on):
        self.p.kind = key
        self.touch(True)

    def load(self):
        super().load()
        self.manual.blockSignals(True)
        self.manual.setChecked(self.p.enabled)
        self.manual.blockSignals(False)
        self.manual_box.setVisible(self.p.enabled)
        self.intensity.setValue(self.p.intensity)
        self.kind.setChecked(self.p.kind)


# ----------------------------------------------------------------------------- volume

class VolumePanel(BasePanel):
    module = "volume"

    def __init__(self, ctl):
        super().__init__(ctl, "Volume deformation", has_more=False)
        self.tools = SegmentedButtons([("hv", "volume_hv", "", "Horizontal / Vertical"),
                                       ("diagonal", "volume_diag", "", "Diagonal (spherical)")],
                                      size=QSize(46, 26), icon_px=20)
        self.tools.toggled.connect(self._tool)
        self.add(row(self.tools, None))
        self.hv_box = QWidget()
        hl = QVBoxLayout(self.hv_box)
        hl.setContentsMargins(0, 4, 0, 0)
        self.h = self.slider("Horizontal", 0, 200, 100, "horizontal")
        self.v = self.slider("Vertical", 0, 200, 0, "vertical")
        hl.addWidget(self.h)
        hl.addWidget(self.v)
        self.d_box = QWidget()
        dl = QVBoxLayout(self.d_box)
        dl.setContentsMargins(0, 4, 0, 0)
        self.d = self.slider("Intensity", 0, 200, 150, "diagonal")
        dl.addWidget(self.d)
        self.add(self.hv_box)
        self.add(self.d_box)

    def _tool(self, key, on):
        if on:
            self.p.mode = key
        self.p.enabled = on
        self.setActive(on)
        self._show()
        self.ctl.edited(self.module, commit=True)

    def _show(self):
        on = self.p.enabled
        self.hv_box.setVisible(on and self.p.mode == "hv")
        self.d_box.setVisible(on and self.p.mode == "diagonal")

    def load(self):
        super().load()
        self.tools.setChecked(self.p.mode if self.p.enabled else None)
        self.h.setValue(self.p.horizontal)
        self.v.setValue(self.p.vertical)
        self.d.setValue(self.p.diagonal)
        self._show()


# ----------------------------------------------------------------------------- perspective

class PerspectivePanel(BasePanel):
    module = "perspective"

    def __init__(self, ctl):
        super().__init__(ctl, "Perspective", has_more=True)
        self.auto = text_button("Auto", tip="Automatic perspective correction")
        self.auto.clicked.connect(self._auto)
        self.tools = SegmentedButtons([("verticals", "persp_verticals", "", "Force vertical parallel (2 lines)"),
                                       ("horizontals", "persp_horizontals", "", "Force horizontal parallel (2 lines)"),
                                       ("rect", "persp_rect", "", "Force rectangle (4 points)"),
                                       ("eight", "persp_eight", "", "8 points")], size=QSize(46, 26))
        self.tools.toggled.connect(lambda k, on: ctl.toggle_tool("perspective", k if on else None))
        self.add(row(self.auto, 6, self.tools, None))
        self.intensity = self.slider("Intensity", 0, 100, 100, "intensity")
        self.add(self.intensity)
        self.mode_combo = QComboBox()
        for key, text in AUTO_MODES:
            self.mode_combo.addItem(text, key)
        self.mode_combo.currentIndexChanged.connect(self._auto_mode)
        lb = label("Auto correct")
        lb.setFixedWidth(120)
        self.add(row(lb, self.mode_combo))
        self.up_down = self.slider("Up / Down", -125, 125, 0, "up_down")
        self.left_right = self.slider("Left / Right", -125, 125, 0, "left_right")
        self.ratio = self.slider("H / V Ratio", -100, 100, 0, "hv_ratio")
        for w in (self.up_down, self.left_right, self.ratio):
            self.add_more(w)

    def _auto(self, on):
        if on:
            self.ctl.run_auto_perspective()
        else:
            self.p.mode = "verticals" if not self.p.lines else self.p.mode
            if self.p.mode == "auto":
                self.p.mode = "verticals"
            self.p.auto_angles = None
            self.ctl.edited(self.module, commit=True)

    def _auto_mode(self, _):
        self.p.auto_mode = self.mode_combo.currentData()
        if self.p.mode == "auto":
            self.ctl.run_auto_perspective()
        else:
            self.ctl.edited(self.module, commit=True)

    def set_tool(self, key):
        self.tools.setChecked(key)

    def load(self):
        super().load()
        self.auto.setChecked(self.p.enabled and self.p.mode == "auto")
        self.intensity.setValue(self.p.intensity)
        self.up_down.setValue(self.p.up_down)
        self.left_right.setValue(self.p.left_right)
        self.ratio.setValue(self.p.hv_ratio)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex([k for k, _ in AUTO_MODES].index(self.p.auto_mode))
        self.mode_combo.blockSignals(False)


# ----------------------------------------------------------------------------- warp

class WarpPanel(BasePanel):
    module = "warp"

    def __init__(self, ctl):
        super().__init__(ctl, "Reshape", has_more=False)
        self.presets = SegmentedButtons([("4", "grid4", "", "4 x 4 grid"), ("8", "grid8", "", "8 x 8 grid"),
                                         ("16", "grid16", "", "16 x 16 grid"), ("32", "grid32", "", "32 x 32 grid"),
                                         ("custom", "grid_custom", "", "Custom grid")], size=QSize(38, 26))
        self.presets.toggled.connect(self._preset)
        lb = label("Grid")
        self.add(row(lb, None, self.presets))
        self.active_box = QWidget()
        al = QVBoxLayout(self.active_box)
        al.setContentsMargins(0, 2, 0, 0)
        al.setSpacing(8)
        self.rows = QSpinBox()
        self.cols = QSpinBox()
        for sb in (self.rows, self.cols):
            sb.setRange(4, 32)
            sb.setFixedWidth(52)
        self.link = QToolButton()
        self.link.setCheckable(True)
        self.link.setChecked(True)
        self.link.setText("⛓")
        self.link.setToolTip("Link rows and columns")
        lbl = label("Rows & Columns")
        al.addLayout(row(lbl, None, self.rows, self.link, self.cols))
        self.subdivide = text_button("Subdivide", checkable=False, tip="Double the grid density")
        self.merge = text_button("Merge", checkable=False, tip="Halve the grid density")
        al.addLayout(row(self.subdivide, self.merge, None))
        self.add(self.active_box)
        self.active_box.hide()
        self.rows.valueChanged.connect(lambda v: self._spin("rows", v))
        self.cols.valueChanged.connect(lambda v: self._spin("cols", v))
        self.subdivide.clicked.connect(lambda: ctl.warp_resize(2.0))
        self.merge.clicked.connect(lambda: ctl.warp_resize(0.5))

    def _preset(self, key, on):
        if not on:
            self.ctl.toggle_tool("warp", None)
            return
        if key != "custom":
            n = int(key)
            self.ctl.warp_set_grid(n, n)
        self.ctl.toggle_tool("warp", key)

    def _spin(self, which, v):
        if self.link.isChecked():
            other = self.cols if which == "rows" else self.rows
            other.blockSignals(True)
            other.setValue(v)
            other.blockSignals(False)
        self.ctl.warp_set_grid(self.rows.value(), self.cols.value())

    def set_active(self, on: bool, key: str | None = None):
        self.active_box.setVisible(on)
        self.presets.setChecked(key if on else None)

    def load(self):
        super().load()
        for sb, v in ((self.rows, self.p.rows), (self.cols, self.p.cols)):
            sb.blockSignals(True)
            sb.setValue(v)
            sb.blockSignals(False)
        self.merge.setEnabled(self.p.rows >= 8 and self.p.cols >= 8 and self.p.rows % 2 == 0 and self.p.cols % 2 == 0)
        self.subdivide.setEnabled(self.p.rows <= 16 and self.p.cols <= 16)


# ----------------------------------------------------------------------------- horizon

class HorizonPanel(BasePanel):
    module = "horizon"

    def __init__(self, ctl):
        super().__init__(ctl, "Horizon", has_more=False)
        self.auto = text_button("Auto", checkable=False, tip="Automatic horizon correction")
        self.auto.clicked.connect(ctl.run_auto_horizon)
        self.tools = SegmentedButtons([("h", "horizon_h", "", "Draw a line that should be horizontal"),
                                       ("v", "horizon_v", "", "Draw a line that should be vertical")],
                                      size=QSize(46, 26))
        self.tools.toggled.connect(lambda k, on: ctl.toggle_tool("horizon", k if on else None))
        self.add(row(self.auto, 6, self.tools, None))
        self.angle = self.slider("Angle", -900, 900, 0, "angle", scale=0.1, decimals=1, cast=float)
        self.add(self.angle)

    def set_tool(self, key):
        self.tools.setChecked(key)

    def load(self):
        super().load()
        self.angle.setValue(self.p.angle)


# ----------------------------------------------------------------------------- crop

class CropPanel(BasePanel):
    module = "crop"

    def __init__(self, ctl):
        super().__init__(ctl, "Crop", has_more=False)
        self.auto = text_button("Auto", tip="Automatically crop to the largest valid area")
        self.auto.clicked.connect(self._auto)
        self.tool = QToolButton()
        self.tool.setCheckable(True)
        self.tool.setProperty("segment", True)
        self.tool.setProperty("segFirst", True)
        self.tool.setProperty("segLast", True)
        self.tool.setIcon(icons.icon("crop"))
        self.tool.setIconSize(QSize(18, 18))
        self.tool.setMinimumSize(40, 26)
        self.tool.setToolTip("Crop (C)")
        self.tool.toggled.connect(lambda on: ctl.toggle_tool("crop", "manual" if on else None))
        self.add(row(self.auto, 6, self.tool, None))
        self.aspect = QComboBox()
        for key, text in ASPECTS:
            self.aspect.addItem(text, key)
        self.aspect.insertSeparator(self.aspect.count())
        self.aspect.addItem("Add Custom…", "add_custom")
        self.aspect.currentIndexChanged.connect(self._aspect)
        lb = label("Aspect Ratio")
        lb.setFixedWidth(120)
        self.add(row(lb, self.aspect))
        self.constrain = QCheckBox("Constrain to image")
        self.constrain.toggled.connect(self._constrain)
        self.add(self.constrain)

    def _auto(self, on):
        self.p.auto = on
        if on:
            self.p.rect = None
        self.touch(True)

    def _aspect(self, _):
        key = self.aspect.currentData()
        if key == "add_custom":
            text, ok = QInputDialog.getText(self.window(), "Custom aspect ratio", "Enter ratio (W/H):", text="3/2")
            ratio = None
            if ok:
                try:
                    a, b = text.replace(":", "/").split("/")
                    ratio = float(a) / float(b)
                except (ValueError, ZeroDivisionError):
                    ratio = None
            if ratio and ratio > 0:
                idx = self.aspect.count() - 2
                self.aspect.blockSignals(True)
                self.aspect.insertItem(idx, text, f"custom:{ratio}")
                self.aspect.setCurrentIndex(idx)
                self.aspect.blockSignals(False)
                key = f"custom:{ratio}"
            else:
                self.load()
                return
        if key.startswith("custom:"):
            self.p.aspect, self.p.custom_ratio = "custom", float(key.split(":", 1)[1])
        else:
            self.p.aspect = key
        self.ctl.crop_aspect_changed()
        self.touch(True)

    def _constrain(self, on):
        self.p.constrain = on
        self.touch(True)

    def load(self):
        super().load()
        self.auto.setChecked(self.p.auto)
        self.aspect.blockSignals(True)
        if self.p.aspect == "custom":
            key = f"custom:{self.p.custom_ratio}"
            idx = self.aspect.findData(key)
            if idx < 0:
                idx = self.aspect.count() - 2
                self.aspect.insertItem(idx, f"{self.p.custom_ratio:.3g}", key)
        else:
            idx = self.aspect.findData(self.p.aspect)
        self.aspect.setCurrentIndex(max(0, idx))
        self.aspect.blockSignals(False)
        self.constrain.blockSignals(True)
        self.constrain.setChecked(self.p.constrain)
        self.constrain.blockSignals(False)


# ----------------------------------------------------------------------------- miniature

class TiltShiftPanel(BasePanel):
    module = "tiltshift"

    def __init__(self, ctl):
        super().__init__(ctl, "Miniature effect", has_more=False)
        self.tool = QToolButton()
        self.tool.setCheckable(True)
        self.tool.setProperty("segment", True)
        self.tool.setProperty("segFirst", True)
        self.tool.setProperty("segLast", True)
        self.tool.setIcon(icons.icon("tiltshift"))
        self.tool.setIconSize(QSize(18, 18))
        self.tool.setMinimumSize(40, 26)
        self.tool.setToolTip("Edit the miniature effect on the image")
        self.tool.toggled.connect(lambda on: ctl.toggle_tool("tiltshift", "edit" if on else None))
        self.add(row(self.tool, None))
        self.intensity = self.slider("Blur intensity", 0, 100, 50, "intensity")
        self.add(self.intensity)
        self.shape = QComboBox()
        for key, text in BLUR_SHAPES.items():
            self.shape.addItem(text, key)
        self.shape.currentIndexChanged.connect(self._shape)
        lb = label("Blur shape")
        lb.setFixedWidth(120)
        self.add(row(lb, self.shape))
        self.sym_pos = QCheckBox("Symmetric position")
        self.sym_blur = QCheckBox("Symmetric blur")
        self.sym_pos.toggled.connect(lambda on: self._flag("sym_position", on))
        self.sym_blur.toggled.connect(lambda on: self._flag("sym_blur", on))
        self.add(row(self.sym_pos, self.sym_blur, None))

    def _shape(self, _):
        self.p.blur_shape = self.shape.currentData()
        self.touch(True)

    def _flag(self, attr, on):
        setattr(self.p, attr, on)
        self.ctl.edited(self.module, commit=True)

    def load(self):
        super().load()
        self.intensity.setValue(self.p.intensity)
        for w, v in ((self.sym_pos, self.p.sym_position), (self.sym_blur, self.p.sym_blur)):
            w.blockSignals(True)
            w.setChecked(v)
            w.blockSignals(False)
        self.shape.blockSignals(True)
        self.shape.setCurrentIndex(max(0, self.shape.findData(self.p.blur_shape)))
        self.shape.blockSignals(False)


# ----------------------------------------------------------------------------- dock

class SettingsDock(QWidget):
    resetAll = Signal()

    def __init__(self, ctl):
        super().__init__()
        self.setObjectName("dockContents")
        self.setFixedWidth(324)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        title = QWidget()
        title.setObjectName("dockTitle")
        title.setFixedHeight(22)
        tl = QHBoxLayout(title)
        tl.setContentsMargins(6, 0, 8, 0)
        tl.addStretch(1)
        self.reset_all = QToolButton()
        self.reset_all.setObjectName("resetAllButton")
        self.reset_all.setText("Reset all")
        self.reset_all.setIcon(icons.icon("reset", 14))
        self.reset_all.setIconSize(QSize(12, 12))
        self.reset_all.setLayoutDirection(Qt.RightToLeft)
        self.reset_all.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.reset_all.clicked.connect(self.resetAll)
        tl.addWidget(self.reset_all)
        lay.addWidget(title)

        scroll = QScrollArea()
        scroll.setObjectName("settingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        contents = QWidget()
        contents.setObjectName("settingsContents")
        cl = QVBoxLayout(contents)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        self.panels = {
            "distortion": DistortionPanel(ctl),
            "volume": VolumePanel(ctl),
            "perspective": PerspectivePanel(ctl),
            "warp": WarpPanel(ctl),
            "horizon": HorizonPanel(ctl),
            "crop": CropPanel(ctl),
            "tiltshift": TiltShiftPanel(ctl),
        }
        for p in self.panels.values():
            cl.addWidget(p)
        cl.addStretch(1)
        scroll.setWidget(contents)
        lay.addWidget(scroll, 1)

    def load(self):
        for p in self.panels.values():
            p.load()
