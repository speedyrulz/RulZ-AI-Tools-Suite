"""Single-image page: full-size before/after with its own adjustment sliders."""
from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout,
    QWidget,
)

from .. import config, imageio
from ..filters import ADJUSTMENTS, BALANCED, BEST
from ..shadows import ShadowEdit
from . import theme
from .compare_view import ERASE, PAINT, SIDE, SLIDER, CompareView
from .controls import AdjustRow, ModeSwitch
from .workers import FullLoad, ImageItem, Process, make_emitter

VIEW_KEY = "compare_mode"
BRUSH_KEY = "shadow_brush"
SHOW_AREAS_KEY = "shadow_show_areas"
SHADOW_TIPS = {
    "strength": "How much the painted shadows are brightened. 100 matches the light around them; lower "
                "leaves some of the shadow.",
    "blend": "How softly the painted area's edge fades into the rest of the picture.",
}


def _brush_fraction(v: int) -> float:
    """Brush slider (1-100) to a radius as a fraction of the picture's longer side: 0.3% to 20%."""
    return 0.003 + 0.197 * (v / 100.0) ** 2


def _pixmap(img) -> QPixmap:
    return QPixmap.fromImage(imageio.to_qimage(img))


class EditorPage(QWidget):
    back = Signal()
    navigate = Signal(int)  # -1 previous, +1 next
    itemChanged = Signal(object)  # ImageItem whose settings changed
    modeRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        self.item: ImageItem | None = None
        self.globals = {k: d for k, (_, d) in ADJUSTMENTS.items()}
        self.mode = BALANCED
        self._full = None
        self._shown_key = None
        self._running = False
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._emitter = make_emitter(self)
        self._emitter.done.connect(self._on_done)
        self._debounce = QTimer(self, singleShot=True, interval=60, timeout=self._request)

        # top bar
        top = QFrame(objectName="TopBar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(12, 8, 12, 8)
        tl.setSpacing(8)
        back = QPushButton(theme.icon("back"), "  All images", clicked=lambda: self.back.emit())
        back.setProperty("flat", True)
        self.prev_btn = QPushButton(theme.icon("prev"), "", clicked=lambda: self.navigate.emit(-1))
        self.next_btn = QPushButton(theme.icon("next"), "", clicked=lambda: self.navigate.emit(1))
        for b in (self.prev_btn, self.next_btn):
            b.setProperty("flat", True)
        self.prev_btn.setToolTip("Previous image (Left)")
        self.next_btn.setToolTip("Next image (Right)")
        self.title = QLabel(objectName="H2")
        self.counter = QLabel(objectName="Muted")
        tl.addWidget(back)
        tl.addSpacing(8)
        tl.addWidget(self.prev_btn)
        tl.addWidget(self.next_btn)
        tl.addWidget(self.title)
        tl.addWidget(self.counter)
        tl.addStretch(1)

        seg = QFrame(objectName="Segment")
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(3, 3, 3, 3)
        sl.setSpacing(2)
        self.side_btn = QPushButton("Side by side", checkable=True)
        self.slider_btn = QPushButton("Slider", checkable=True)
        group = QButtonGroup(self)
        for b in (self.side_btn, self.slider_btn):
            group.addButton(b)
            sl.addWidget(b)
        self.side_btn.setToolTip("Original and processed next to each other (V to switch)")
        self.slider_btn.setToolTip("Drag the divider to wipe between original and processed (V to switch)")
        self.side_btn.clicked.connect(lambda: self._set_mode(SIDE))
        self.slider_btn.clicked.connect(lambda: self._set_mode(SLIDER))
        tl.addWidget(seg)
        tl.addSpacing(12)
        self.zoom_label = QLabel(objectName="Muted")
        self.zoom_label.setMinimumWidth(44)
        self.zoom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fit = QPushButton("Fit", clicked=lambda: self.view.fit())
        one = QPushButton("100%", clicked=lambda: self.view.set_zoom(1.0))
        fit.setToolTip("Fit the whole image (double-click the image to toggle)")
        one.setToolTip("Actual pixels — best for judging noise and JPEG blocks")
        tl.addWidget(self.zoom_label)
        tl.addWidget(fit)
        tl.addWidget(one)

        self.view = CompareView()
        self.view.zoomChanged.connect(lambda s: self.zoom_label.setText(f"{s * 100:.0f}%"))

        # bottom panel: one row per adjustment
        bottom = QFrame(objectName="BottomBar")
        grid = QGridLayout(bottom)
        grid.setContentsMargins(16, 10, 16, 10)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        self.rows: dict[str, AdjustRow] = {}
        for r, key in enumerate(ADJUSTMENTS):
            row = AdjustRow(key, with_reset=True)
            row.changed.connect(self._on_slider)
            row.reset.connect(self._reset)
            self.rows[key] = row
            grid.addWidget(row, r, 0)
        self.mode_switch = ModeSwitch()
        self.mode_switch.requested.connect(self.modeRequested)
        jpeg_row = list(ADJUSTMENTS).index("jpeg")
        grid.addWidget(self.mode_switch, jpeg_row, 1)
        self.use_global = QPushButton(theme.icon("reset"), "  Use global for all", clicked=self._reset_all)
        self.use_global.setToolTip("Drop all of this image's own settings and follow the global sliders")
        grid.addWidget(self.use_global, len(ADJUSTMENTS) - 1, 1)
        self.message = QLabel(objectName="Muted")
        grid.addWidget(self.message, 0, 1)
        grid.addWidget(self._build_shadow_tools(), len(ADJUSTMENTS), 0, 1, 2)
        grid.setColumnStretch(0, 1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(top)
        lay.addWidget(self.view, 1)
        lay.addWidget(bottom)

        mode = str(config.settings().value(VIEW_KEY, SIDE))
        self._set_mode(mode if mode in (SIDE, SLIDER) else SIDE, save=False)
        self.view.stroked.connect(self._on_stroke)

        for key, fn in (
            (Qt.Key_Escape, lambda: self._set_tool(None) if self.view.tool else self.back.emit()),
            (Qt.Key_Left, lambda: self.navigate.emit(-1)),
            (Qt.Key_Right, lambda: self.navigate.emit(1)), (Qt.Key_V, self._toggle_mode),
            (Qt.Key_B, lambda: self._set_tool(None if self.view.tool == PAINT else PAINT)),
            (Qt.Key_E, lambda: self._set_tool(None if self.view.tool == ERASE else ERASE)),
            (Qt.Key_BracketLeft, lambda: self.brush.setValue(self.brush.value() - 5)),
            (Qt.Key_BracketRight, lambda: self.brush.setValue(self.brush.value() + 5)),
        ):
            QShortcut(QKeySequence(key), self, fn, context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence.Undo, self, self._undo_stroke, context=Qt.WidgetWithChildrenShortcut)

    def _build_shadow_tools(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 6, 0, 0)
        v.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(10)
        title = QLabel("Shadow areas")
        title.setFixedWidth(170)
        title.setToolTip("Paint over shadows to brighten them to match the light around them. Texture and "
                         "contrast are kept, the edge blends in, and a blue shadow tint is matched too.")
        seg = QFrame(objectName="Segment")
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(3, 3, 3, 3)
        sl.setSpacing(2)
        self.paint_btn = QPushButton("Paint", checkable=True)
        self.erase_btn = QPushButton("Erase", checkable=True)
        self.paint_btn.setToolTip("Paint over a shadow to brighten it (B). Right-drag to move the picture.")
        self.erase_btn.setToolTip("Rub out part of what you painted (E)")
        for b, tool in ((self.paint_btn, PAINT), (self.erase_btn, ERASE)):
            sl.addWidget(b)
            b.clicked.connect(lambda on, t=tool: self._set_tool(t if on else None))
        brush_label = QLabel("Brush", objectName="Muted")
        self.brush = QSlider(Qt.Horizontal, minimum=1, maximum=100)
        self.brush.setFixedWidth(150)
        self.brush.setToolTip("Brush size ([ and ] keys)")
        self.brush.setValue(int(config.settings().value(BRUSH_KEY, 40)))
        self.brush.valueChanged.connect(self._brush_changed)
        self.view.set_brush(_brush_fraction(self.brush.value()))
        self.show_areas = QCheckBox("Show areas")
        self.show_areas.setToolTip("Tint the painted areas red on the original (always shown while painting)")
        self.show_areas.setChecked(str(config.settings().value(SHOW_AREAS_KEY, "false")).lower() == "true")
        self.show_areas.toggled.connect(self._show_areas_changed)
        self.view.set_show_areas(self.show_areas.isChecked())
        self.undo_btn = QPushButton(theme.icon("reset"), "  Undo", clicked=self._undo_stroke)
        self.undo_btn.setToolTip("Take back the last stroke (Ctrl+Z)")
        self.clear_btn = QPushButton(theme.icon("clear"), "  Clear", clicked=self._clear_strokes)
        self.clear_btn.setToolTip("Remove all painted shadow areas from this picture")
        row.addWidget(title)
        row.addWidget(seg)
        row.addSpacing(8)
        row.addWidget(brush_label)
        row.addWidget(self.brush)
        row.addSpacing(8)
        row.addWidget(self.show_areas)
        row.addStretch(1)
        row.addWidget(self.undo_btn)
        row.addWidget(self.clear_btn)
        v.addLayout(row)
        sliders = QHBoxLayout()
        sliders.setSpacing(24)
        self.shadow_rows: dict[str, AdjustRow] = {}
        for key, label in (("strength", "Shadow lift"), ("blend", "Edge blend")):
            r = AdjustRow(key, label=label, tip=SHADOW_TIPS[key])
            r.changed.connect(self._on_shadow_slider)
            self.shadow_rows[key] = r
            sliders.addWidget(r, 1)
        v.addLayout(sliders)
        return box

    # ---- shadow areas ------------------------------------------------------
    def _set_tool(self, tool: str | None):
        self.paint_btn.setChecked(tool == PAINT)
        self.erase_btn.setChecked(tool == ERASE)
        self.view.set_tool(tool)
        self.view.setFocus()

    def _brush_changed(self, v: int):
        config.settings().setValue(BRUSH_KEY, v)
        self.view.set_brush(_brush_fraction(v))

    def _show_areas_changed(self, on: bool):
        config.settings().setValue(SHOW_AREAS_KEY, "true" if on else "false")
        self.view.set_show_areas(on)

    def _shadow(self) -> ShadowEdit:
        return self.item.shadow if self.item.shadow is not None else ShadowEdit()

    def _set_shadow(self, edit: ShadowEdit | None):
        self.item.shadow = edit
        self.view.set_shadow(edit)
        self._changed()

    def _on_stroke(self, stroke):
        if self.item is None:
            return
        if stroke.erase and not self._shadow().strokes:
            return  # nothing to rub out
        edit = self._shadow()
        self._set_shadow(replace(edit, strokes=edit.strokes + (stroke,)))

    def _undo_stroke(self):
        if self.item is not None and self._shadow().strokes:
            edit = self._shadow()
            self._set_shadow(replace(edit, strokes=edit.strokes[:-1]))

    def _clear_strokes(self):
        if self.item is not None and self._shadow().strokes:
            self._set_shadow(replace(self._shadow(), strokes=()))

    def _on_shadow_slider(self, key: str, v: int):
        if self.item is not None:
            self._set_shadow(replace(self._shadow(), **{key: v}))

    # ---- compare mode -----------------------------------------------------
    def _set_mode(self, mode: str, save: bool = True):
        (self.side_btn if mode == SIDE else self.slider_btn).setChecked(True)
        self.view.set_mode(mode)
        if save:
            config.settings().setValue(VIEW_KEY, mode)

    def _toggle_mode(self):
        self._set_mode(SLIDER if self.view.mode == SIDE else SIDE)

    # ---- opening ---------------------------------------------------------
    def open(self, item: ImageItem, index: int, total: int, globals_: dict, mode: str):
        self.item = item
        self.globals = dict(globals_)
        self.mode = mode
        self.mode_switch.set_mode(mode)
        self._full = None
        self._shown_key = None
        self.message.setText("")
        self.title.setText(item.path.name)
        self.title.setToolTip(str(item.path))
        self.counter.setText(f"{index + 1} / {total}")
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < total - 1)
        self._sync_controls()
        thumb = _pixmap(item.thumb) if item.thumb is not None else None
        self.view.set_shadow(None)
        self.view.set_images(thumb, item.preview if item.preview_key is not None else None)
        self.view.set_shadow(item.shadow)
        self.view.set_busy(True)
        self._pool.start(FullLoad(item, self._emitter))
        self.view.setFocus()

    def set_globals(self, globals_: dict, mode: str):
        self.globals = dict(globals_)
        self.mode = mode
        self.mode_switch.set_mode(mode)
        if self.item is not None:
            self._sync_controls()
            self._debounce.start()

    def close_image(self):
        self.item = None
        self._full = None
        self.view.set_images(None, None)

    def _sync_controls(self):
        it = self.item
        for key, row in self.rows.items():
            row.set_value(it.value(key, self.globals), key in it.custom, self.globals[key])
        self.use_global.setEnabled(bool(it.custom))
        edit = self._shadow()
        for key, row in self.shadow_rows.items():
            row.set_value(getattr(edit, key))
        self.undo_btn.setEnabled(bool(edit.strokes))
        self.clear_btn.setEnabled(bool(edit.strokes))

    # ---- settings --------------------------------------------------------
    def _on_slider(self, key: str, v: int):
        if self.item is None:
            return
        self.item.custom[key] = v
        self._changed()

    def _reset(self, key: str):
        if self.item is not None:
            self.item.custom.pop(key, None)
            self._changed()

    def _reset_all(self):
        if self.item is not None:
            self.item.custom.clear()
            self._changed()

    def _changed(self):
        self._sync_controls()
        self.itemChanged.emit(self.item)
        self._debounce.start()

    # ---- processing ------------------------------------------------------
    def _wanted(self):
        settings = self.item.settings(self.globals)
        return (tuple(sorted(settings.items())), (self.mode if settings.get("jpeg") else BALANCED),
                self.item.shadow_edit())

    def _request(self):
        if self.item is None or self._full is None:
            return
        if self._running:
            return  # _on_done asks again with the latest values
        key = self._wanted()
        if key == self._shown_key:
            self.view.set_busy(False)
            return
        self._running = True
        self.view.set_busy(True)
        if key[1] == BEST:
            self.message.setText("AI is working… a few seconds")
        self._pool.start(Process((self.item, key), self._full, dict(key[0]), key[1], self._emitter, key[2]))

    def _on_done(self, key, result):
        if isinstance(key, ImageItem):  # full-size load finished
            if key is not self.item:
                return
            if isinstance(result, Exception):
                self.view.set_busy(False)
                self.message.setText(f"Couldn't open: {result}")
                return
            self._full = result
            self.view.set_images(_pixmap(result), None)
            self.view.set_busy(True)
            self._request()
            return
        item, wanted = key
        self._running = False
        if item is self.item and self._full is not None:
            if isinstance(result, Exception):
                self.message.setText(f"Couldn't process: {result}")
                self._shown_key = wanted  # don't retry the same failing settings in a loop
            else:
                self.message.setText("")
                self._shown_key = wanted
                self.view.set_processed(_pixmap(result))
        self._request()
        if self.item is not None and self._shown_key == self._wanted():
            self.view.set_busy(False)
