"""The Distortion repair window: RulZ Image Distortion Repair's editor, working on one frame.

It has the same correction palette (distortion, volume deformation, perspective, reshape,
horizon, crop, miniature effect), canvas tools, undo/redo and before/after compare. The frame
shown is the video's current frame as it was on screen (rotated/flipped), and Save hands the
corrections back to the main window, which applies them to the preview, frame PNGs and exports.

The editing logic is ported from viewpointer/ui/main_window.py (MainWindow's tool, render and
undo methods); file browsing, saving images, guides and video playback are left out.
"""
from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from .. import theme
from ..defaults import settings as app_settings
from . import style
from .core.autodetect import aspect_value, auto_horizon, auto_perspective
from .core.geometry import WarpStage
from .core.params import Corrections, Orientation
from .core.render import Renderer, StillImage
from .pipeline import describe, from_json, to_json
from .ui.overlays import CropOverlay, HorizonOverlay, LinesOverlay, TiltShiftOverlay, WarpOverlay
from .ui.panels import SettingsDock
from .ui.toolpanes import CropPane, FloatingWarpOptions, HorizonPane, PerspectivePane, TiltShiftPane, WarpPane
from .ui.viewer import ImageView, to_qimage

DRAFT_SIDE = 900
LAST_KEY = "repair/last"          # JSON of the most recently saved adjustments
TITLE = "Distortion repair"


class _Signals(QObject):
    done = Signal(int, object, object)      # generation, result, error


class RenderJob(QRunnable):
    def __init__(self, gen, renderer, c, upto, max_side, signals):
        super().__init__()
        self.gen, self.renderer, self.c, self.upto, self.max_side, self.signals = gen, renderer, c, upto, max_side, signals

    def run(self):
        try:
            res = self.renderer.render(self.c, upto=self.upto, max_side=self.max_side)
            self.signals.done.emit(self.gen, res, None)
        except Exception as e:  # noqa: BLE001 - shown in the window's status line
            traceback.print_exc()
            self.signals.done.emit(self.gen, None, e)


class SnapshotCommand(QUndoCommand):
    def __init__(self, win, before: Corrections, after: Corrections, text: str):
        super().__init__(text)
        self.win, self.before, self.after = win, before, after
        self._first = True

    def redo(self):
        if self._first:            # the edit is already applied when pushed
            self._first = False
            return
        self.win.restore(self.after)

    def undo(self):
        self.win.restore(self.before)


def last_saved() -> Corrections | None:
    text = app_settings().value(LAST_KEY, "", type=str)
    if not text:
        return None
    try:
        return from_json(text)
    except ValueError:
        return None


class AdjustmentsWindow(QDialog):
    """exec() == Accepted means Save: the corrections are in `result`."""

    MODULE_NAMES = {"distortion": "Distortion", "volume": "Volume deformation", "perspective": "Perspective",
                    "warp": "Reshape", "horizon": "Horizon", "crop": "Crop", "tiltshift": "Miniature effect",
                    "": "Edit"}

    def __init__(self, rgb: np.ndarray, corrections: Corrections | None = None, note: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{TITLE} — {theme.APP_NAME}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        self.setStyleSheet(style.stylesheet())
        self.settings = app_settings()
        self.renderer = Renderer(StillImage(np.ascontiguousarray(rgb[..., :3])))
        c = corrections.copy() if corrections is not None else Corrections()
        c.orientation = Orientation()
        self.c = c
        self.initial = c.copy()
        self.committed = c.copy()
        self.result: Corrections | None = None
        self.note = note
        self.tool: tuple[str, str] | None = None
        self.tool_snapshot: Corrections | None = None
        self.previewing = False
        self._pre_preview = None
        self._crop_valid = None
        self._closing = False
        self.undo = QUndoStack(self)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.signals = _Signals()
        self.signals.done.connect(self._render_done)
        self.gen = 0
        self.busy = False
        self.pending = None
        self.last_result = None
        self.before_cache = None
        self.render_timer = QTimer(self, singleShot=True, interval=15)
        self.render_timer.timeout.connect(self._kick_render)
        self.hq_timer = QTimer(self, singleShot=True, interval=250)
        self.hq_timer.timeout.connect(lambda: self.request_render(draft=False))
        self.status_timer = QTimer(self, singleShot=True)
        self.status_timer.timeout.connect(lambda: self.status.setText(self.note))
        s = self.settings
        self.line_color_v = QColor(s.value("repair/colors/vertical", "#ed030f"))
        self.line_color_h = QColor(s.value("repair/colors/horizontal", "#ffffff"))
        self.grid_color = QColor(s.value("repair/colors/grid", "#ed030f"))
        self.horizon_color = QColor(s.value("repair/colors/horizon", "#ffffff"))
        self.tilt_color = QColor(s.value("repair/colors/tilt", "#ffffff"))

        self._build_ui()
        self.dock.load()
        self._update_actions()
        self.request_render(draft=False)
        scr = (parent.screen() if parent is not None else self.screen()).availableGeometry()
        self.resize(min(1500, int(scr.width() * 0.92)), min(950, int(scr.height() * 0.9)))

    # ------------------------------------------------------------------ construction
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        def button(text, tip, slot=None, checkable=False):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setFocusPolicy(Qt.NoFocus)
            if slot is not None:
                (b.toggled if checkable else b.clicked).connect(slot)
            return b

        top = QWidget()
        top.setObjectName("repairTopBar")
        top.setAttribute(Qt.WA_StyledBackground, True)
        tl = QHBoxLayout(top)
        tl.setContentsMargins(10, 6, 10, 6)
        tl.setSpacing(6)
        self.b_undo = button("Undo", "Undo (Ctrl+Z)", self.undo.undo)
        self.b_redo = button("Redo", "Redo (Ctrl+Y)", self.undo.redo)
        self.b_compare = button("Show original", "Show the frame without these adjustments", self._compare, True)
        self.b_fit = button("Fit", "Fit the frame in the window (Ctrl+0)", self.fit)
        self.b_one = button("1:1", "Show the frame at 100 % (Ctrl+1)", self.one_to_one)
        self.b_grid = button("Grid", "Show a grid over the frame", self._grid, True)
        for b in (self.b_undo, self.b_redo):
            tl.addWidget(b)
        tl.addSpacing(16)
        for b in (self.b_compare, self.b_fit, self.b_one, self.b_grid):
            tl.addWidget(b)
        tl.addStretch(1)
        self.b_last = button("Use last saved", "Load the adjustments you saved most recently (from any video)",
                             self.load_last)
        self.b_load = button("Load from file…", "Load adjustments saved to a file", self.load_file)
        self.b_save_file = button("Save to file…", "Save these adjustments to a file to reuse them later",
                                  self.save_file)
        for b in (self.b_last, self.b_load, self.b_save_file):
            tl.addWidget(b)
        root.addWidget(top)

        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(0)
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)
        self.view = ImageView()
        left.addWidget(self.view, 1)
        self.pane_slot = QStackedWidget()
        self.pane_slot.setFixedHeight(50)
        self.pane_slot.hide()
        left.addWidget(self.pane_slot)
        mid.addLayout(left, 1)
        self.toggle_dock_btn = QToolButton()
        self.toggle_dock_btn.setObjectName("dockToggle")
        self.toggle_dock_btn.setFixedWidth(12)
        self.toggle_dock_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.toggle_dock_btn.setText("›")
        self.toggle_dock_btn.setToolTip("Show / hide the correction palette")
        self.toggle_dock_btn.clicked.connect(lambda: self.dock.setVisible(not self.dock.isVisible()))
        mid.addWidget(self.toggle_dock_btn)
        self.dock = SettingsDock(self)
        self.dock.resetAll.connect(self.reset_all)
        mid.addWidget(self.dock)
        root.addLayout(mid, 1)
        # Video frames have no lens profile: hide the profile-only parts of the Distortion panel.
        dist = self.dock.panels["distortion"]
        dist.auto.hide()
        dist.info.hide()
        for i in range(dist.extra.count()):
            w = dist.extra.itemAt(i).widget()
            if w is not None:
                w.hide()
        self.dock.panels["crop"].tool.setToolTip("Draw the crop by hand")   # (no C shortcut here)

        bottom = QWidget()
        bottom.setObjectName("repairBottomBar")
        bottom.setAttribute(Qt.WA_StyledBackground, True)
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(10, 6, 10, 6)
        self.status = QLabel(self.note)
        self.status.setObjectName("note")
        self.status.setWordWrap(True)
        bl.addWidget(self.status, 1)
        self.b_cancel = button("Cancel", "Close without saving (Esc)", self.close_without_saving)
        self.b_save = button("Save", "Save these adjustments: the preview, frame PNGs and every export use them",
                             self.save)
        self.b_save.setObjectName("primary")
        bl.addWidget(self.b_cancel)
        bl.addWidget(self.b_save)
        root.addWidget(bottom)

        self.float_opts = FloatingWarpOptions(self.view)
        self.float_opts.hide()
        self.view.installEventFilter(self)
        self.view.viewport().installEventFilter(self)

        self.panes = {
            "perspective": PerspectivePane(self.line_color_v, self.line_color_h),
            "horizon": HorizonPane(self.horizon_color),
            "crop": CropPane(int(self.settings.value("repair/crop_opacity", 40)),
                             self.settings.value("repair/crop_grid", True, type=bool)),
            "warp": WarpPane(self.grid_color),
            "tiltshift": TiltShiftPane(self.tilt_color),
        }
        for pane in self.panes.values():
            self.pane_slot.addWidget(pane)
            pane.applyClicked.connect(self.apply_tool)
            pane.resetClicked.connect(self.reset_tool)
            pane.previewClicked.connect(self._preview_clicked)
        self.panes["perspective"].swatch_v.colorChanged.connect(lambda c: self._set_color("vertical", c))
        self.panes["perspective"].swatch_h.colorChanged.connect(lambda c: self._set_color("horizontal", c))
        self.panes["horizon"].swatch.colorChanged.connect(lambda c: self._set_color("horizon", c))
        self.panes["warp"].swatch.colorChanged.connect(lambda c: self._set_color("grid", c))
        self.panes["tiltshift"].swatch.colorChanged.connect(lambda c: self._set_color("tilt", c))
        self.panes["crop"].show_grid.toggled.connect(self._crop_grid)
        self.panes["crop"].opacity.valueChanged.connect(self._crop_opacity)
        self.panes["perspective"].autoPreviewToggled.connect(
            lambda v: self.settings.setValue("repair/persp_autopreview", v))
        self.panes["warp"].modeChanged.connect(self._warp_mode)
        self.panes["warp"].show_settings.toggled.connect(lambda on: self._place_float(on))
        fo = self.float_opts
        fo.subChanged.connect(lambda k: self._warp_attr("sub", k))
        fo.stickyToggled.connect(lambda on: self._warp_attr("sticky", on))
        fo.propagationChanged.connect(lambda v: self._warp_attr("propagation", v))
        fo.perspAction.connect(self._warp_persp_action)
        fo.perspIntensity.connect(lambda v: self._warp_attr("persp_intensity", v))
        fo.anamorphChanged.connect(self._warp_anamorph)

        # Enter applies the active canvas tool, so no button may be the dialog's default.
        for b in self.findChildren(QPushButton):
            b.setAutoDefault(False)
            b.setDefault(False)
        self.undo.canUndoChanged.connect(lambda _: self._update_actions())
        self.undo.canRedoChanged.connect(lambda _: self._update_actions())
        # (No single-letter shortcuts: they would take the keys typed into the palette's lists.)
        for seq, slot in (("Ctrl+Z", self.undo.undo), ("Ctrl+Y", self.undo.redo), ("Ctrl+Shift+Z", self.undo.redo),
                          ("Ctrl+0", self.fit), ("Ctrl+1", self.one_to_one)):
            QShortcut(QKeySequence(seq), self, activated=slot)

    # ------------------------------------------------------------------ status / actions
    def show_status(self, text: str, ms: int = 6000):
        self.status.setText(text)
        self.status_timer.start(ms)

    def _update_actions(self):
        if self._closing:   # the undo stack signals once more while it is being destroyed
            return
        self.b_undo.setEnabled(self.undo.canUndo())
        self.b_redo.setEnabled(self.undo.canRedo())
        self.b_undo.setToolTip(f"Undo {self.undo.undoText()} (Ctrl+Z)" if self.undo.canUndo() else "Undo (Ctrl+Z)")
        self.b_redo.setToolTip(f"Redo {self.undo.redoText()} (Ctrl+Y)" if self.undo.canRedo() else "Redo (Ctrl+Y)")
        self.b_last.setEnabled(bool(self.settings.value(LAST_KEY, "", type=str)))

    def is_modified(self) -> bool:
        return self.c.to_dict() != self.initial.to_dict()

    # ------------------------------------------------------------------ save / close
    def save(self):
        self.finish_tool(apply=True)
        self.result = self.c.copy()
        self.settings.setValue(LAST_KEY, to_json(self.result))
        self.accept()

    def reject(self):
        """Esc: cancels the active canvas tool if there is one, otherwise closes like Cancel."""
        if self.tool:
            self.finish_tool(apply=False)
            return
        self.close_without_saving()

    def close_without_saving(self) -> bool:
        """Cancel button / close box: unsaved changes are only thrown away after asking."""
        if self.tool:
            self.finish_tool(apply=False)
        if self.is_modified() and QMessageBox.question(
                self, TITLE, "Close without saving the changes you made here?",
                QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel) != QMessageBox.Discard:
            return False
        super().reject()
        return True

    def closeEvent(self, e):
        if self.result is not None or self.close_without_saving():
            e.accept()
        else:
            e.ignore()

    def done(self, r):
        self._closing = True
        self.render_timer.stop()
        self.hq_timer.stop()
        self.pool.waitForDone(10000)   # no render job may report to a closed window
        super().done(r)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.tool:
                self.apply_tool()
            e.accept()
            return
        super().keyPressEvent(e)   # Esc -> reject()

    # ------------------------------------------------------------------ presets
    def _load(self, c: Corrections, what: str):
        self.finish_tool(apply=False)
        c = c.copy()
        c.orientation = Orientation()
        self.c = c
        self.dock.load()
        self.edited("", commit=True)
        self.show_status(f"Loaded {what}: {describe(c)}.")

    def load_last(self):
        c = last_saved()
        if c is None:
            self.show_status("Nothing has been saved yet.")
            return
        self._load(c, "the last saved adjustments")

    def load_file(self):
        start = self.settings.value("repair/preset_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(self, "Load adjustments", start,
                                              "Distortion repair settings (*.json);;All files (*.*)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                c = from_json(f.read())
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, TITLE, f"Could not load\n{path}\n\n{e}")
            return
        self.settings.setValue("repair/preset_dir", path.rsplit("/", 1)[0])
        self._load(c, path.rsplit("/", 1)[-1])

    def save_file(self):
        self.finish_tool(apply=True)
        start = self.settings.value("repair/preset_dir", "", type=str)
        path, _ = QFileDialog.getSaveFileName(self, "Save adjustments", start + ("/" if start else "")
                                              + "distortion-repair.json", "Distortion repair settings (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(to_json(self.c))
        except OSError as e:
            QMessageBox.warning(self, TITLE, f"Could not write\n{path}\n\n{e}")
            return
        self.settings.setValue("repair/preset_dir", path.rsplit("/", 1)[0])
        self.show_status(f"Saved to {path}")

    # ------------------------------------------------------------------ editing (palette API)
    def edited(self, module: str, commit: bool):
        if commit:
            self.commit(module)
        self.request_render(draft=not commit)

    def commit(self, module: str = ""):
        if self.c.to_dict() == self.committed.to_dict():
            return
        self.undo.push(SnapshotCommand(self, self.committed.copy(), self.c.copy(),
                                       self.MODULE_NAMES.get(module, "Edit")))
        self.committed = self.c.copy()
        self._update_actions()

    def restore(self, c: Corrections):
        self.finish_tool(apply=False, restore=False)
        self.c = c.copy()
        self.committed = c.copy()
        self.dock.load()
        self.request_render(draft=False)

    def reset_module(self, module: str):
        if self.tool and self.tool[0] == module:
            self.finish_tool(apply=False)
        setattr(self.c, module, type(getattr(self.c, module))())
        self.dock.panels[module].load()
        self.edited(module, commit=True)

    def reset_all(self):
        self.finish_tool(apply=False)
        self.c = Corrections()
        self.dock.load()
        self.edited("", commit=True)

    # ------------------------------------------------------------------ rendering
    def render_upto(self):
        if self.tool is None or self.previewing:
            return None
        name = self.tool[0]
        return {"perspective": "perspective", "horizon": "horizon", "warp": "crop", "crop": "crop"}.get(name)

    def request_render(self, draft=False):
        if self._closing:
            return
        self.pending = draft
        self.render_timer.start()
        if draft:
            self.hq_timer.start()

    def _kick_render(self):
        if self.busy or self.pending is None or self._closing:
            return
        draft = self.pending
        self.pending = None
        self.busy = True
        self.gen += 1
        side = DRAFT_SIDE if draft else None
        self.pool.start(RenderJob(self.gen, self.renderer, self.c.copy(), self.render_upto(), side, self.signals))
        if self.b_compare.isChecked():
            self._ensure_before()

    def _ensure_before(self):
        if self.before_cache is None:
            base = Corrections()
            base.crop.enabled = False
            self.before_cache = self.renderer.render(base)

    def _render_done(self, gen, res, err):
        self.busy = False
        if self._closing:
            return
        if err is not None:
            self.show_status(f"This frame can't be processed with these settings: {err}")
        elif res is not None and gen == self.gen:
            self.last_result = res
            self._show_result(res)
        if self.pending is not None:
            self.render_timer.start()

    def _scene_size(self, res):
        fw, fh = res.frame_size
        return fw * res.crop[2], fh * res.crop[3]

    def _show_result(self, res):
        if self.b_compare.isChecked() and self.before_cache is not None:
            b = self.before_cache
            self.view.set_image(to_qimage(b.rgba), *self._scene_size(b))
        else:
            self.view.set_image(to_qimage(res.rgba), *self._scene_size(res))
        self._update_caption()

    def _update_caption(self):
        res = self.last_result
        if res is None:
            self.view.set_caption(None)
            return
        if self.b_compare.isChecked() and self.before_cache is not None:
            w, h = self._scene_size(self.before_cache)
            self.view.set_caption(f"Original: {round(w)} x {round(h)} px")
        else:
            w, h = self._scene_size(res)
            self.view.set_caption(f"{round(w)} x {round(h)} px")

    # ------------------------------------------------------------------ view actions
    def fit(self):
        self.view.fit()

    def one_to_one(self):
        self.view.one_to_one()

    def _compare(self, on):
        if on:
            if self.tool:
                self.finish_tool(apply=True)
            self._ensure_before()
        if self.last_result:
            self._show_result(self.last_result)

    def _grid(self, on):
        self.view.show_grid = on
        self.view.viewport().update()

    def _set_color(self, which, c: QColor):
        self.settings.setValue(f"repair/colors/{which}", c.name())
        if which == "vertical":
            self.line_color_v = c
        elif which == "horizontal":
            self.line_color_h = c
        elif which == "grid":
            self.grid_color = c
        elif which == "horizon":
            self.horizon_color = c
        elif which == "tilt":
            self.tilt_color = c
        ov = self.view.overlay
        if isinstance(ov, LinesOverlay):
            ov.color_v, ov.color_h = self.line_color_v, self.line_color_h
        elif ov is not None and hasattr(ov, "color"):
            ov.color = {"grid": self.grid_color, "horizon": self.horizon_color, "tilt": self.tilt_color}.get(which, ov.color)
        if ov is not None:
            ov.update()

    # ------------------------------------------------------------------ tools
    def toggle_tool(self, tool: str, key: str | None):
        if self.tool and self.tool[0] == tool and key is not None and self.tool[1] == key:
            return
        if self.tool:
            self.finish_tool(apply=True)
        if key is None:
            return
        if self.b_compare.isChecked():
            self.b_compare.setChecked(False)
        self.tool = (tool, key)
        self.tool_snapshot = self.c.copy()
        self.previewing = False
        res = self.last_result
        if tool == "perspective":
            p = self.c.perspective
            data = p.lines if p.mode == key and p.lines else None
            ov = LinesOverlay(key, data, self.line_color_v, self.line_color_h)
            self.panes["perspective"].set_mode(key)
            self.dock.panels["perspective"].set_tool(key)
            ov.finished.connect(self._lines_finished)
            self._set_auto_preview(self.settings.value("repair/persp_autopreview", False, type=bool))
        elif tool == "horizon":
            ov = HorizonOverlay(key, self.horizon_color)
            self.dock.panels["horizon"].set_tool(key)
            ov.changed.connect(self._horizon_angle_label)
            ov.finished.connect(self._horizon_angle_label)
        elif tool == "warp":
            w = self.c.warp
            ov = WarpOverlay(w.rows, w.cols, w.disp, self.grid_color)
            ov.changed.connect(self._warp_live)
            ov.finished.connect(self._warp_commit)
            self.dock.panels["warp"].set_active(True, key)
            self.panes["warp"].modes.setChecked("free")
            fo = self.float_opts
            ov.sticky = fo.sticky_free.isChecked()
            ov.propagation = int(fo.propagation.value())
            ov.sub = fo.sub.checked() or "move"
            ov.persp_intensity = int(fo.persp_int.value())
            fo.set_mode("free")
            self._place_float(self.panes["warp"].show_settings.isChecked())
        elif tool == "crop":
            frame = self.renderer.frame(self.c.orientation)
            rect = self.renderer.crop_rect(self.c)
            self._crop_valid = self.renderer.validity_mask(self.c, cells=300)
            ratio = aspect_value(self.c.crop.aspect, self.c.crop.custom_ratio, frame)
            if ratio and (ratio > 1) != (frame.aspect > 1) and abs(ratio - 1) > 1e-6 and self.c.crop.aspect != "custom":
                ratio = 1 / ratio
            pane = self.panes["crop"]
            ov = CropOverlay(rect, (frame.width, frame.height), ratio, int(pane.opacity.value()),
                             pane.show_grid.isChecked(), self._crop_constrain)
            ov.finished.connect(lambda: None)
            tb = self.dock.panels["crop"].tool
            tb.blockSignals(True)
            tb.setChecked(True)
            tb.blockSignals(False)
        elif tool == "tiltshift":
            if not self.c.tiltshift.enabled:
                self.c.tiltshift.enabled = True
                self.dock.panels["tiltshift"].setActive(True)
            aspect = (res.frame_size[0] * res.crop[2]) / (res.frame_size[1] * res.crop[3]) if res else 1.5
            ov = TiltShiftOverlay(self.c.tiltshift, aspect, self.tilt_color)
            ov.changed.connect(lambda: self.request_render(draft=True))
            ov.finished.connect(lambda: self.edited("tiltshift", commit=True))
            tb = self.dock.panels["tiltshift"].tool
            tb.blockSignals(True)
            tb.setChecked(True)
            tb.blockSignals(False)
        else:
            self.tool = None
            return
        self.view.set_overlay(ov)
        self.pane_slot.setCurrentWidget(self.panes[tool])
        self.pane_slot.show()
        self.request_render(draft=False)

    def finish_tool(self, apply: bool, restore: bool = True):
        if not self.tool:
            return
        tool, key = self.tool
        ov = self.view.overlay
        if apply:
            if tool == "perspective" and isinstance(ov, LinesOverlay):
                p = self.c.perspective
                p.mode, p.lines, p.enabled, p.auto_angles = key, ov.value(), True, None
            elif tool == "horizon" and isinstance(ov, HorizonOverlay):
                frame = self.renderer.frame(self.c.orientation)
                a = ov.angle(frame.aspect)
                if abs(a) > 1e-3:
                    self.c.horizon.angle = float(np.clip(round(a, 2), -90, 90))
                    self.c.horizon.enabled = True
            elif tool == "warp" and isinstance(ov, WarpOverlay):
                self.c.warp.disp = ov.value()
                self.c.warp.enabled = bool(self.c.warp.disp) or self.c.warp.enabled
            elif tool == "crop" and isinstance(ov, CropOverlay):
                self.c.crop.rect, self.c.crop.auto, self.c.crop.enabled = list(ov.r), False, True
        elif restore and self.tool_snapshot is not None:
            self.c = self.tool_snapshot.copy()
        self.tool = None
        self.tool_snapshot = None
        self.previewing = False
        self._pre_preview = None
        self.view.set_overlay(None)
        self.pane_slot.hide()
        self.float_opts.hide()
        panels = self.dock.panels
        panels["perspective"].set_tool(None)
        panels["horizon"].set_tool(None)
        panels["warp"].set_active(False)
        for name in ("crop", "tiltshift"):
            panels[name].tool.blockSignals(True)
            panels[name].tool.setChecked(False)
            panels[name].tool.blockSignals(False)
        self.dock.load()
        self.commit(tool)
        self.request_render(draft=False)
        self._update_actions()

    def apply_tool(self):
        if self.tool:
            self.finish_tool(apply=True)

    def reset_tool(self):
        if not self.tool:
            return
        tool, key = self.tool
        ov = self.view.overlay
        if tool == "perspective" and isinstance(ov, LinesOverlay):
            ov.data = LinesOverlay.default(key)
            ov.update()
        elif tool == "horizon" and isinstance(ov, HorizonOverlay):
            ov.reset()
            ov.update()
            self._horizon_angle_label()
        elif tool == "warp" and isinstance(ov, WarpOverlay):
            ov.disp[:] = 0
            ov.selected.clear()
            self.float_opts.reset_anamorph()
            ov._base_anamorph = None
            ov.update()
            self._warp_live()
            self._warp_commit()
        elif tool == "crop" and isinstance(ov, CropOverlay):
            c = self.c.copy()
            c.crop.auto, c.crop.rect = True, None
            ov.r = self.renderer.crop_rect(c)
            ov.update()
        elif tool == "tiltshift" and isinstance(ov, TiltShiftOverlay):
            ov.on_double_click("center", None)
            ov.update()
            self.edited("tiltshift", commit=True)
        if self.previewing:
            self._preview(False)

    # perspective helpers
    def _set_auto_preview(self, on):
        pane = self.panes["perspective"]
        pane.auto_preview.blockSignals(True)
        pane.auto_preview.setChecked(on)
        pane.auto_preview.blockSignals(False)
        pane.preview.setEnabled(True)

    def _lines_finished(self):
        if self.panes["perspective"].auto_preview.isChecked():
            self._preview(True)

    def _preview_clicked(self):
        self._preview(not self.previewing)

    def _preview(self, on: bool):
        if not self.tool or self.tool[0] not in ("perspective", "horizon"):
            return
        tool, key = self.tool
        ov = self.view.overlay
        pane = self.panes[tool]
        if on:
            self._pre_preview = self.c.copy()
            if tool == "perspective":
                p = self.c.perspective
                p.mode, p.lines, p.enabled, p.auto_angles = key, ov.value(), True, None
            else:
                a = ov.angle(self.renderer.frame(self.c.orientation).aspect)
                self.c.horizon.angle, self.c.horizon.enabled = float(np.clip(a, -90, 90)), True
            ov.setVisible(False)
        else:
            if self._pre_preview is not None:
                self.c = self._pre_preview
                self._pre_preview = None
            ov.setVisible(True)
        self.previewing = on
        pane.preview.setChecked(on)
        pane.preview.setText("Edit" if on else "Preview")
        self.request_render(draft=False)

    def _horizon_angle_label(self):
        ov = self.view.overlay
        if isinstance(ov, HorizonOverlay):
            a = ov.angle(self.renderer.frame(self.c.orientation).aspect)
            self.panes["horizon"].angle_label.setText(f"Rotation: {a:+.2f}°")

    def run_auto_perspective(self):
        self.finish_tool(apply=True)
        p = self.c.perspective
        probe = self.c.copy()
        probe.perspective.enabled = False
        res = self.renderer.render(probe, upto="perspective", max_side=1200)
        frame = self.renderer.frame(self.c.orientation)
        angles = auto_perspective(res.rgba, frame, self.renderer.focal, p.auto_mode)
        if angles is None:
            QMessageBox.information(self, TITLE, "The perspective of this frame could not be corrected "
                                                 "automatically. Draw lines with the tools next to Auto instead.")
            self.dock.panels["perspective"].load()
            return
        p.mode, p.auto_angles, p.enabled = "auto", angles, True
        self.dock.panels["perspective"].load()
        self.edited("perspective", commit=True)

    def run_auto_horizon(self):
        self.finish_tool(apply=True)
        probe = self.c.copy()
        probe.horizon.enabled = False
        res = self.renderer.render(probe, upto="horizon", max_side=1200)
        a = auto_horizon(res.rgba, self.renderer.frame(self.c.orientation))
        if a is None:
            QMessageBox.information(self, TITLE, "The horizon of this frame could not be found automatically. "
                                                 "Draw a line that should be level instead.")
            return
        self.c.horizon.angle, self.c.horizon.enabled = float(np.clip(a, -90, 90)), True
        self.dock.panels["horizon"].load()
        self.edited("horizon", commit=True)

    # warp helpers
    def _place_float(self, show: bool):
        if not (self.tool and self.tool[0] == "warp") or not show:
            self.float_opts.hide()
            return
        self.float_opts.adjustSize()
        self.float_opts.move(16, self.view.height() - self.float_opts.height() - 16)
        self.float_opts.show()
        self.float_opts.raise_()

    def _warp_mode(self, mode):
        ov = self.view.overlay
        if isinstance(ov, WarpOverlay):
            ov.mode = mode
            if mode == "anamorph":
                ov.begin_anamorph()
                self.float_opts.reset_anamorph()
            ov.update()
        self.float_opts.set_mode(mode)
        self._place_float(self.panes["warp"].show_settings.isChecked())

    def _warp_attr(self, attr, v):
        ov = self.view.overlay
        if isinstance(ov, WarpOverlay):
            setattr(ov, attr, v)

    def _warp_persp_action(self, kind):
        ov = self.view.overlay
        if isinstance(ov, WarpOverlay):
            ov.persp_action(kind)

    def _warp_anamorph(self, d):
        ov = self.view.overlay
        if isinstance(ov, WarpOverlay) and ov.mode == "anamorph":
            ov.anamorph = d
            ov.apply_anamorph()

    def _warp_live(self):
        ov = self.view.overlay
        if isinstance(ov, WarpOverlay):
            self.c.warp.disp = ov.value()
            self.c.warp.rows, self.c.warp.cols = ov.rows, ov.cols
            if self.c.warp.disp and not self.c.warp.enabled:
                self.c.warp.enabled = True
                self.dock.panels["warp"].setActive(True)
            self.request_render(draft=True)

    def _warp_commit(self):
        self._warp_live()
        if isinstance(self.view.overlay, WarpOverlay) and self.view.overlay.mode == "anamorph":
            self.view.overlay._base_anamorph = None
        self.commit("warp")
        self.request_render(draft=False)

    def warp_set_grid(self, rows, cols):
        w = self.c.warp
        if (rows, cols) == (w.rows, w.cols):
            return
        old = np.asarray(w.disp) if w.disp else None
        w.rows, w.cols = rows, cols
        if old is not None:
            # resample the existing displacement onto the new grid
            st = WarpStage(old)
            gu, gv = np.meshgrid(np.linspace(0, 1, cols + 1), np.linspace(0, 1, rows + 1))
            du, dv = st.grid_sample(gu, gv)
            w.disp = np.dstack([du, dv]).tolist()
        ov = self.view.overlay
        if isinstance(ov, WarpOverlay):
            ov.set_grid(rows, cols, w.disp)
        self.dock.panels["warp"].load()
        self.edited("warp", commit=True)

    def warp_resize(self, factor):
        w = self.c.warp
        r, c = int(round(w.rows * factor)), int(round(w.cols * factor))
        self.warp_set_grid(min(32, max(4, r)), min(32, max(4, c)))

    # crop helpers
    def _crop_constrain(self, rect, check_only=False):
        if not self.c.crop.constrain or self._crop_valid is None:
            return True if check_only else rect
        valid = self._crop_valid
        h, w = valid.shape

        def ok(r):
            x0, y0 = int(np.floor(r[0] * w)), int(np.floor(r[1] * h))
            x1, y1 = int(np.ceil((r[0] + r[2]) * w)), int(np.ceil((r[1] + r[3]) * h))
            sub = valid[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            return sub.size > 0 and bool(sub.all())
        if check_only:
            return ok(rect)
        r = list(rect)
        for _ in range(40):
            if ok(r):
                return r
            cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
            r = [cx - r[2] * 0.48, cy - r[3] * 0.48, r[2] * 0.96, r[3] * 0.96]
        return r

    def crop_aspect_changed(self):
        ov = self.view.overlay
        if isinstance(ov, CropOverlay):
            frame = self.renderer.frame(self.c.orientation)
            ratio = aspect_value(self.c.crop.aspect, self.c.crop.custom_ratio, frame)
            if ratio and (ratio > 1) != (frame.aspect > 1) and abs(ratio - 1) > 1e-6 and self.c.crop.aspect != "custom":
                ratio = 1 / ratio
            ov.set_ratio(ratio)

    def _crop_grid(self, on):
        self.settings.setValue("repair/crop_grid", on)
        if isinstance(self.view.overlay, CropOverlay):
            self.view.overlay.show_grid = on
            self.view.overlay.update()

    def _crop_opacity(self, v):
        self.settings.setValue("repair/crop_opacity", int(v))
        if isinstance(self.view.overlay, CropOverlay):
            self.view.overlay.opacity = int(v)
            self.view.overlay.update()

    # ------------------------------------------------------------------ Qt events
    def eventFilter(self, obj, ev):
        if self._closing:
            return False
        if obj is self.view and ev.type() == QEvent.Resize and self.float_opts.isVisible():
            self._place_float(True)
        if obj is self.view.viewport() and ev.type() == QEvent.MouseButtonPress and self.previewing \
                and self.panes["perspective"].auto_preview.isChecked():
            self._preview(False)
        return super().eventFilter(obj, ev)
