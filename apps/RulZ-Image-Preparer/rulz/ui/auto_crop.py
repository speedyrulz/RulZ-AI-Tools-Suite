"""Auto crop: find what you type with SAM 3.1, preview the mask and crop, then crop and save the pictures.

The crop follows the RulZ Mask Intelligent Crop node (see ..crop). Optionally the clean-up settings are
applied first, so one press cleans and crops every selected picture.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRectF, QRunnable, QThread, QThreadPool, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListView, QListWidget, QListWidgetItem, QMessageBox, QProgressDialog, QPushButton, QScrollArea,
    QSpinBox, QVBoxLayout, QWidget,
)

from .. import ai_jpeg, ai_mask, config, imageio
from ..crop import ASPECT_LIMITS, CropSettings, crop_box
from ..filters import ADJUSTMENTS, BEST, is_noop, process_crop
from . import theme
from .ai_setup import ensure_ready
from .theme import C
from .workers import ImageItem, make_emitter, output_paths

SETTINGS_KEY = "autocrop/settings"
DIR_KEY = "autocrop/last_dir"
DISPLAY_MAX_SIDE = 1600
DISPLAY_KEPT = 12  # full previews kept for going back; older ones are loaded again when shown
OVERLAY_MAX_SIDE = 512
REFERENCES = {"crop": "The final crop", "mask": "The found area", "image": "The whole picture"}


@dataclass(frozen=True)
class FindSettings:
    prompt: str
    confidence: float = 0.5
    mask_threshold: float = 0.5
    best_only: bool = False


@dataclass
class Found:
    bits: np.ndarray  # the mask (at most ai_mask.MASK_MAX_SIDE), packed 8 pixels to a byte
    shape: tuple[int, int]  # the mask's (height, width)
    scores: list[float]
    size: tuple[int, int]  # full picture (height, width)

    @classmethod
    def of(cls, mask: np.ndarray, scores: list[float], size: tuple[int, int]) -> "Found":
        return cls(np.packbits(mask.astype(bool), axis=None), mask.shape[:2], scores, size)

    @property
    def mask(self) -> np.ndarray:
        """The bool mask."""
        h, w = self.shape
        return np.unpackbits(self.bits, count=h * w).reshape(h, w).view(bool)


def _display(img: np.ndarray) -> QImage:
    # a QImage, not a QPixmap: this runs on a worker thread
    return imageio.to_qimage(imageio.shrink(img, DISPLAY_MAX_SIDE))


# ---- background work --------------------------------------------------------------
class FindAll(QThread):
    """Loads each picture and finds the prompt in it, current one first."""

    found = Signal(object, object, object)  # item, Found or Exception, QImage or None
    finished_all = Signal()

    def __init__(self, items: list[ImageItem], find: FindSettings):
        super().__init__()
        self.items, self.find = items, find
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        for it in self.items:
            if self._cancel:
                break
            try:
                img = imageio.load(it.path)
                mask, scores = ai_mask.find(img, self.find.prompt, self.find.confidence,
                                            self.find.mask_threshold, self.find.best_only)
                self.found.emit(it, Found.of(mask, scores, img.shape[:2]), _display(img))
            except Exception as e:  # noqa: BLE001 - shown for that picture
                self.found.emit(it, e, None)
        self.finished_all.emit()


class _DisplayLoad(QRunnable):
    """Loads a picture's preview again after it was let go to save memory."""

    def __init__(self, item: ImageItem, emitter):
        super().__init__()
        self.item, self.emitter = item, emitter

    def run(self):
        try:
            image = _display(imageio.load(self.item.path, DISPLAY_MAX_SIDE))
        except Exception:  # noqa: BLE001 - the thumbnail stays up
            image = None
        self.emitter.done.emit(self.item, image)


class CropAll(QThread):
    progress = Signal(int, str)  # index, file name
    finished_all = Signal(int, list, list)  # saved count, [names with nothing found], [(name, error)]

    def __init__(self, jobs: list[tuple[ImageItem, Path, dict | None, Found | None]], find: FindSettings,
                 crop: CropSettings, mode: str):
        super().__init__()
        self.jobs = jobs  # (item, destination, clean-up settings or None, mask already found or None)
        self.find, self.crop, self.mode = find, crop, mode
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        saved, empty, errors = 0, [], []
        for i, (it, dst, cleanup, found) in enumerate(self.jobs):
            if self._cancel:
                break
            self.progress.emit(i, it.path.name)
            try:
                img = imageio.load(it.path)
                mask = found.mask if found is not None else ai_mask.find(
                    img, self.find.prompt, self.find.confidence, self.find.mask_threshold, self.find.best_only)[0]
                box = crop_box(mask, self.crop, *img.shape[:2])
                if box is None:
                    empty.append(it.path.name)
                    continue
                if cleanup is not None:  # only the crop (plus a margin) when that gives the same picture
                    out = process_crop(img, box, cleanup, self.mode, shadow=it.shadow_edit())
                else:
                    x0, y0, x1, y1 = box
                    out = np.ascontiguousarray(img[y0:y1, x0:x1])
                del img
                imageio.save(dst, out)
                saved += 1
            except Exception as e:  # noqa: BLE001
                errors.append((it.path.name, str(e)))
        self.progress.emit(len(self.jobs), "")
        self.finished_all.emit(saved, empty, errors)


# ---- preview ----------------------------------------------------------------------
class MaskPreview(QWidget):
    """The picture fitted to the panel, the found area tinted red and the crop outlined."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 300)
        self._pix: QPixmap | None = None
        self._overlay: QPixmap | None = None
        self._box: tuple[int, int, int, int] | None = None
        self._full: tuple[int, int] | None = None
        self._message = ""

    def show_picture(self, pix: QPixmap | None, found: Found | None, box, message: str = ""):
        self._pix, self._box, self._message = pix, box, message
        self._full = found.size if found is not None else None
        self._overlay = None
        if pix is not None and found is not None and found.scores:
            # at most OVERLAY_MAX_SIDE: it's a soft tint, drawn scaled up over the picture
            f = min(1.0, OVERLAY_MAX_SIDE / max(pix.width(), pix.height()))
            ow, oh = max(1, round(pix.width() * f)), max(1, round(pix.height() * f))
            m = cv2.resize(found.mask.view(np.uint8) * np.uint8(120), (ow, oh), interpolation=cv2.INTER_AREA)
            rgba = np.zeros((oh, ow, 4), np.uint8)
            rgba[..., 0] = 237
            rgba[..., 3] = m
            q = QImage(rgba.data, ow, oh, rgba.strides[0], QImage.Format_RGBA8888)
            self._overlay = QPixmap.fromImage(q.copy())
        self.update()

    def set_box(self, box):
        self._box = box
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setRenderHint(QPainter.Antialiasing)
        if self._pix is None:
            p.setPen(QColor(C["muted"]))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self._message or "Loading…")
            return
        pw, ph = self._pix.width(), self._pix.height()
        f = min((self.width() - 16) / pw, (self.height() - 16) / ph)
        target = QRectF((self.width() - pw * f) / 2, (self.height() - ph * f) / 2, pw * f, ph * f)
        p.drawPixmap(target, self._pix, QRectF(self._pix.rect()))
        if self._overlay is not None:
            p.drawPixmap(target, self._overlay, QRectF(self._overlay.rect()))
        if self._box is not None and self._full is not None:
            fh, fw = self._full
            x0, y0, x1, y1 = self._box
            sx, sy = target.width() / fw, target.height() / fh
            r = QRectF(target.left() + x0 * sx, target.top() + y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy)
            shade = QColor(0, 0, 0, 150)
            for part in (QRectF(target.left(), target.top(), target.width(), r.top() - target.top()),
                         QRectF(target.left(), r.bottom(), target.width(), target.bottom() - r.bottom()),
                         QRectF(target.left(), r.top(), r.left() - target.left(), r.height()),
                         QRectF(r.right(), r.top(), target.right() - r.right(), r.height())):
                p.fillRect(part, shade)
            p.setPen(QPen(QColor(C["text"]), 2, Qt.DashLine))
            p.drawRect(r)
        if self._message:
            p.setPen(QColor(C["text"]))
            band = QRectF(0, self.height() - 34, self.width(), 30)
            p.fillRect(band.adjusted(8, 0, -8, 0), QColor(0, 0, 0, 170))
            p.drawText(band, Qt.AlignCenter, self._message)


# ---- dialog -----------------------------------------------------------------------
def _number(lo: float, hi: float, step: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox(minimum=lo, maximum=hi, singleStep=step, decimals=decimals, suffix=suffix)
    box.setKeyboardTracking(False)
    box.setButtonSymbols(QDoubleSpinBox.NoButtons)  # type, scroll or use the arrow keys
    return box


class AutoCropDialog(QDialog):
    def __init__(self, items: list[ImageItem], globals_: dict[str, int], mode: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Auto crop - {len(items)} picture{'s' * (len(items) != 1)}")
        self.resize(1240, 820)
        self.items, self.globals, self.mode = items, globals_, mode
        self._found: dict[int, Found | Exception] = {}  # id(item) -> result for self._found_for
        self._found_for: FindSettings | None = None
        self._pix: OrderedDict[int, QPixmap] = OrderedDict()  # id(item) -> preview, last shown last
        self._loading: set[int] = set()
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._emitter = make_emitter(self)
        self._emitter.done.connect(self._on_display)
        self._finder: FindAll | None = None
        self._cropper: CropAll | None = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(16)
        lay.addWidget(self._build_settings())
        right = QVBoxLayout()
        right.setSpacing(10)
        panel = QFrame(objectName="PreviewPanel")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(4, 4, 4, 4)
        self.preview = MaskPreview()
        pl.addWidget(self.preview)
        right.addWidget(panel, 1)
        self.strip = QListWidget()
        self.strip.setViewMode(QListView.IconMode)
        self.strip.setFlow(QListView.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setMovement(QListView.Static)
        self.strip.setIconSize(QPixmap(96, 96).size())
        self.strip.setFixedHeight(128)
        self.strip.setHorizontalScrollMode(QListView.ScrollPerPixel)
        for it in items:
            li = QListWidgetItem(it.path.name)
            li.setToolTip(str(it.path))
            if it.thumb is not None:
                li.setIcon(QPixmap.fromImage(imageio.to_qimage(imageio.shrink(it.thumb, 96))))
            self.strip.addItem(li)
        self.strip.currentRowChanged.connect(lambda _: self._show_current())
        self.strip.setVisible(len(items) > 1)
        right.addWidget(self.strip)
        lay.addLayout(right, 1)

        self._load_settings()
        self.strip.setCurrentRow(0)
        self._update_buttons()

    # ---- settings panel --------------------------------------------------------
    def _build_settings(self) -> QWidget:
        scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.NoFrame)
        scroll.setFixedWidth(400)
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 8, 0)
        v.setSpacing(6)

        v.addWidget(QLabel("What to crop", objectName="Section"))
        self.prompt = QLineEdit(placeholderText="e.g. face, dog, red car")
        self.prompt.setToolTip("Describe what to find, in a few words. Every picture is cropped to it.")
        self.prompt.textChanged.connect(self._find_changed)
        self.prompt.returnPressed.connect(self._preview)
        v.addWidget(self.prompt)

        v.addWidget(QLabel("Finding it (SAM 3.1)", objectName="Section"))
        find = QFormLayout()
        find.setHorizontalSpacing(12)
        self.matches = QComboBox()
        self.matches.addItems(["All matches", "Best match only"])
        self.matches.setToolTip("All matches: the crop covers every match (e.g. every face).\n"
                                "Best match only: just the most confident one.")
        self.confidence = _number(0.05, 1.0, 0.05, 2)
        self.confidence.setToolTip("How sure SAM must be that something matches. Lower finds more, "
                                   "including wrong things; higher finds less.")
        self.mask_threshold = _number(0.05, 0.95, 0.05, 2)
        self.mask_threshold.setToolTip("Mask threshold: pixels SAM is at least this sure about count as part "
                                       "of the match. Lower makes the found area bigger, higher tighter.")
        for label, w in (("Matches", self.matches), ("Confidence", self.confidence),
                         ("Mask threshold", self.mask_threshold)):
            find.addRow(label, w)
        for w in (self.matches, self.confidence, self.mask_threshold):
            (w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged).connect(self._find_changed)
        v.addLayout(find)

        v.addWidget(QLabel("Crop (Mask Intelligent Crop)", objectName="Section"))
        crop = QFormLayout()
        crop.setHorizontalSpacing(12)
        self.padding_mode = QComboBox()
        self.padding_mode.addItem("Same on every side", "uniform")
        self.padding_mode.addItem("Each side separately", "per_edge")
        self.padding_mode.setToolTip("How much room to leave around what was found: one value for every side, "
                                     "or one for each side.")
        self.padding = _number(0, 100, 0.5, 1, " %")
        self.padding.setToolTip("Room left around what was found on every side, as a percentage of the size "
                                "picked under 'Measured against'.")
        self.edges = {e: _number(0, 100, 0.5, 1, " %") for e in ("top", "bottom", "left", "right")}
        self.reference = QComboBox()
        for key, label in REFERENCES.items():
            self.reference.addItem(label, key)
        self.reference.setToolTip("What the percentages are of. The final crop: what was found sits inside a "
                                  "border of that share of the output. The found area: a share of its size. "
                                  "The whole picture: a share of the picture's size.")
        self.aspect = QComboBox()
        for key in ASPECT_LIMITS:
            self.aspect.addItem("Off" if key == "off" else key, key)
        self.aspect.setToolTip("Never crop narrower or wider than this shape. A crop that would be too tall "
                               "is widened, too wide is made taller. It only ever grows.")
        self.divisible = QSpinBox(minimum=1, maximum=1024)
        self.divisible.setKeyboardTracking(False)
        self.divisible.setButtonSymbols(QSpinBox.NoButtons)
        self.divisible.setToolTip("Grow the crop so its width and height are multiples of this (e.g. 8, 16 or 64 "
                                  "for training). 1 = off.")
        crop.addRow("Padding", self.padding_mode)
        crop.addRow("All sides", self.padding)
        for e, w in self.edges.items():
            crop.addRow(e.capitalize(), w)
        crop.addRow("Measured against", self.reference)
        crop.addRow("Shape limit", self.aspect)
        crop.addRow("Divisible by", self.divisible)
        self._crop_form = crop
        for w in (self.padding_mode, self.reference, self.aspect):
            w.currentIndexChanged.connect(self._crop_changed)
        for w in (self.padding, self.divisible, *self.edges.values()):
            w.valueChanged.connect(self._crop_changed)
        v.addLayout(crop)

        v.addWidget(QLabel("Clean-up", objectName="Section"))
        self.cleanup = QCheckBox("Also remove noise and JPEG damage first")
        self.cleanup.setToolTip("Apply each picture's clean-up settings (color noise, JPEG artifacts, detail "
                                "recovery and painted shadow areas) before cropping, the same way Save all does.")
        v.addWidget(self.cleanup)
        self.cleanup_note = QLabel(objectName="Muted", wordWrap=True)
        v.addWidget(self.cleanup_note)
        self._describe_cleanup()
        v.addStretch(1)

        self.info = QLabel(objectName="Muted", wordWrap=True)
        v.addWidget(self.info)
        row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview", clicked=self._preview)
        self.preview_btn.setToolTip("Find what you typed in the selected pictures and show the crop (Enter)")
        self.crop_btn = QPushButton(theme.icon("save", "#ffffff"), "  Crop…", clicked=self._crop)
        self.crop_btn.setProperty("primary", True)
        self.crop_btn.setToolTip("Pick a folder, then crop every selected picture and save it there")
        close = QPushButton("Close", clicked=self.reject)
        row.addWidget(self.preview_btn)
        row.addWidget(self.crop_btn)
        row.addStretch(1)
        row.addWidget(close)
        v.addLayout(row)
        scroll.setWidget(body)
        return scroll

    def _describe_cleanup(self):
        g = ", ".join(f"{ADJUSTMENTS[k][0]} {self.globals[k]}" for k in ADJUSTMENTS)
        custom = sum(1 for it in self.items if it.custom)
        note = f"Global: {g}."
        if custom:
            note += f" {custom} picture{'s have' if custom != 1 else ' has'} its own settings."
        if self.mode == BEST:
            note += " JPEG uses Best quality (AI)."
        shadows = sum(1 for it in self.items if it.shadow_edit())
        if shadows:
            note += f" Shadow areas painted on {shadows} picture{'s' * (shadows != 1)} are lifted too."
        if all(is_noop(it.settings(self.globals)) and not it.shadow_edit() for it in self.items):
            note = "Every clean-up setting is 0 for these pictures, so this changes nothing."
        self.cleanup_note.setText(note)

    def _load_settings(self):
        try:
            d = json.loads(str(config.settings().value(SETTINGS_KEY, "{}")))
        except (TypeError, ValueError):
            d = {}
        c = CropSettings.from_dict(d.get("crop", {}))
        widgets = [self.prompt, self.matches, self.confidence, self.mask_threshold, self.padding_mode, self.padding,
                   self.reference, self.aspect, self.divisible, self.cleanup, *self.edges.values()]
        for w in widgets:
            w.blockSignals(True)
        self.prompt.setText(str(d.get("prompt", "")))
        self.matches.setCurrentIndex(1 if d.get("best_only") else 0)
        self.confidence.setValue(float(d.get("confidence", 0.5)))
        self.mask_threshold.setValue(float(d.get("mask_threshold", 0.5)))
        self.padding_mode.setCurrentIndex(max(0, self.padding_mode.findData(c.padding_mode)))
        self.padding.setValue(c.padding_percent)
        for e, w in self.edges.items():
            w.setValue(getattr(c, f"padding_{e}"))
        self.reference.setCurrentIndex(max(0, self.reference.findData(c.padding_reference)))
        self.aspect.setCurrentIndex(max(0, self.aspect.findData(c.aspect_limit)))
        self.divisible.setValue(c.divisible_by)
        self.cleanup.setChecked(bool(d.get("cleanup", False)))
        for w in widgets:
            w.blockSignals(False)
        self._show_padding_rows()

    def _save_settings(self):
        f = self.find_settings()
        config.settings().setValue(SETTINGS_KEY, json.dumps({
            "prompt": f.prompt, "best_only": f.best_only, "confidence": f.confidence,
            "mask_threshold": f.mask_threshold, "crop": self.crop_settings().to_dict(),
            "cleanup": self.cleanup.isChecked(),
        }))

    def find_settings(self) -> FindSettings:
        return FindSettings(self.prompt.text().strip(), round(self.confidence.value(), 2),
                            round(self.mask_threshold.value(), 2), self.matches.currentIndex() == 1)

    def crop_settings(self) -> CropSettings:
        return CropSettings(
            padding_mode=self.padding_mode.currentData(), padding_percent=self.padding.value(),
            padding_top=self.edges["top"].value(), padding_bottom=self.edges["bottom"].value(),
            padding_left=self.edges["left"].value(), padding_right=self.edges["right"].value(),
            padding_reference=self.reference.currentData(), aspect_limit=self.aspect.currentData(),
            divisible_by=self.divisible.value(),
        )

    def _show_padding_rows(self):
        per_edge = self.padding_mode.currentData() == "per_edge"
        self._crop_form.setRowVisible(self.padding, not per_edge)
        for w in self.edges.values():
            self._crop_form.setRowVisible(w, per_edge)

    # ---- reacting to changes -----------------------------------------------------
    def _find_changed(self, *_):
        self._update_buttons()
        self._show_current()

    def _crop_changed(self, *_):
        self._show_padding_rows()
        self._show_current()

    def _update_buttons(self):
        busy = self._finder is not None
        has_prompt = bool(self.prompt.text().strip())
        self.preview_btn.setEnabled(has_prompt and not busy)
        self.crop_btn.setEnabled(has_prompt and not busy)

    def _current(self) -> ImageItem | None:
        row = self.strip.currentRow()
        return self.items[row] if 0 <= row < len(self.items) else None

    def _show_current(self):
        it = self._current()
        if it is None:
            return
        fresh = self._found_for == self.find_settings()
        result = self._found.get(id(it)) if fresh else None
        pix = self._pix.get(id(it))
        if pix is not None:
            self._pix.move_to_end(id(it))
        else:
            if isinstance(result, Found) and id(it) not in self._loading:  # its preview was let go
                self._loading.add(id(it))
                self._pool.start(_DisplayLoad(it, self._emitter))
            if it.thumb is not None:
                pix = QPixmap.fromImage(imageio.to_qimage(it.thumb))
        if isinstance(result, Exception):
            self.preview.show_picture(pix, None, None, f"Couldn't check this picture: {result}")
        elif result is None:
            busy = self._finder is not None and fresh
            hint = "Looking…" if busy else (
                "Type what to crop, then press Preview" if not self.prompt.text().strip() else "Press Preview to see the crop")
            self.preview.show_picture(pix, None, None, hint)
        elif not result.scores:
            self.preview.show_picture(pix, result, None, f"Nothing found for “{self._found_for.prompt}”")
        elif (box := crop_box(result.mask, self.crop_settings(), *result.size)) is None:
            self.preview.show_picture(pix, None, None, f"Nothing found for “{self._found_for.prompt}”")
        else:
            x0, y0, x1, y1 = box
            n = len(result.scores)
            self.preview.show_picture(pix, result, box,
                                      f"{n} match{'es' * (n != 1)} · best {max(result.scores):.0%} · "
                                      f"crop {x1 - x0} × {y1 - y0}")
        self._update_info()

    def _update_info(self):
        if self._found_for != self.find_settings() or not self._found:
            self.info.setText("")
            return
        done = [r for r in self._found.values() if isinstance(r, Found)]
        empty = sum(1 for r in done if not r.scores)
        failed = sum(1 for r in self._found.values() if isinstance(r, Exception))
        parts = [f"Checked {len(self._found)} of {len(self.items)}"]
        if empty:
            parts.append(f"nothing found in {empty}")
        if failed:
            parts.append(f"{failed} couldn't be read")
        self.info.setText(" · ".join(parts) + ".")

    # ---- preview -------------------------------------------------------------------
    def _preview(self):
        f = self.find_settings()
        if not f.prompt or self._finder is not None:
            return
        self._save_settings()
        if self._found_for != f:
            self._found, self._found_for = {}, f
        todo = [it for it in self.items if id(it) not in self._found]
        cur = self._current()
        if cur in todo:  # the one on screen first
            todo.remove(cur)
            todo.insert(0, cur)
        if not todo:
            self._show_current()
            return
        if not self._pix:
            self.info.setText("Starting SAM 3.1… the first time takes up to half a minute.")
        self._finder = FindAll(todo, f)
        self._finder.found.connect(self._on_found)
        self._finder.finished_all.connect(self._finder_done)
        self._update_buttons()
        self._show_current()
        self._finder.start()

    def _keep_pix(self, it: ImageItem, pix: QPixmap):
        self._pix[id(it)] = pix
        self._pix.move_to_end(id(it))
        cur = self._current()
        while len(self._pix) > DISPLAY_KEPT:  # oldest first, but never the one on screen
            oldest = next(iter(self._pix))
            if cur is not None and oldest == id(cur):
                self._pix.move_to_end(oldest)
                oldest = next(iter(self._pix))
            del self._pix[oldest]

    def _on_display(self, it: ImageItem, image):
        self._loading.discard(id(it))
        if image is not None:
            self._keep_pix(it, QPixmap.fromImage(image))
            if it is self._current():
                self._show_current()

    def _on_found(self, it: ImageItem, result, image):
        if self._finder is None or self._finder.find != self._found_for:
            return
        self._found[id(it)] = result
        if image is not None:
            self._keep_pix(it, QPixmap.fromImage(image))
        if it is self._current():
            self._show_current()
        else:
            self._update_info()

    def _finder_done(self):
        self._finder = None
        self._update_buttons()
        self._show_current()

    # ---- crop -----------------------------------------------------------------------
    def _crop(self):
        f = self.find_settings()
        if not f.prompt:
            return
        cleanup = self.cleanup.isChecked()
        if cleanup and self.mode == BEST and any(it.value("jpeg", self.globals) for it in self.items) \
                and not ensure_ready(self):
            return
        self._save_settings()
        start = str(config.settings().value(DIR_KEY, "") or self.items[0].path.parent)
        out = QFileDialog.getExistingDirectory(self, "Save the cropped pictures to…", start)
        if not out:
            return
        config.settings().setValue(DIR_KEY, out)
        out_dir = Path(out)
        dests = output_paths(self.items, out_dir)
        existing = [d for d in dests if d.exists()]
        if existing and QMessageBox.question(
            self, "Replace files?",
            f"{len(existing)} file(s) with the same name already exist in that folder. Replace them?",
        ) != QMessageBox.Yes:
            return
        known = self._found if self._found_for == f else {}
        jobs = []
        for it, d in zip(self.items, dests):
            r = known.get(id(it))
            jobs.append((it, d, it.settings(self.globals) if cleanup else None, r if isinstance(r, Found) else None))

        dlg = QProgressDialog("Cropping…", "Cancel", 0, len(jobs), self)
        dlg.setWindowTitle("Auto crop")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setMinimumWidth(420)
        worker = CropAll(jobs, f, self.crop_settings(), self.mode)
        worker.progress.connect(lambda i, name: (dlg.setValue(i), dlg.setLabelText(f"Cropping {name}…") if name else None))
        dlg.canceled.connect(worker.cancel)

        def finished(saved, empty, errors):
            dlg.reset()
            ai_jpeg.release()  # Best quality's model, if the clean-up used it
            msg = f"Saved {saved} of {len(jobs)} cropped picture(s) to\n{out_dir}"
            if empty:
                msg += f"\n\nNothing found for “{f.prompt}” in {len(empty)}, so they weren't saved:\n" + \
                       "\n".join(f"• {n}" for n in empty[:10]) + ("\n…" if len(empty) > 10 else "")
            if errors:
                msg += "\n\nProblems:\n" + "\n".join(f"• {n}: {e}" for n, e in errors[:10])
            box = QMessageBox(QMessageBox.Warning if (errors or empty) else QMessageBox.Information, "Auto crop", msg,
                              parent=self)
            open_btn = box.addButton("Open folder", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Ok)
            box.exec()
            if box.clickedButton() is open_btn:
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices

                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))
            self._cropper = None

        worker.finished_all.connect(finished)
        self._cropper = worker
        worker.start()

    def done(self, r):
        for w in (self._finder, self._cropper):
            if w is not None and w.isRunning():
                w.cancel()
                w.wait()
        self._pool.clear()
        self._pool.waitForDone()
        self._pix.clear()
        ai_mask.release()  # SAM holds ~2 GB of GPU memory; it's loaded again next time
        self._save_settings()
        super().done(r)
