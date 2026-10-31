"""Match the skin tone of a video frame to a reference photo.

Model (per channel, in linear light):   out_c = k_c * in_c ** g
- k_c  : per-channel gain  -> colour balance + exposure
- g    : shared power      -> contrast / lighting spread
The parameters are fitted so the mean (geometric) skin colour and the spread of
skin luminance in the video frame match the reference. Because the model is
separable it is exactly three 256-entry lookup tables, which the preview applies
directly and ffmpeg applies through a 1D .cube LUT, so export == preview.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QImage, QImageReader, QPainter, QColor, QPen, QPixmap
from . import theme
from .defaults import settings as _settings
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QSlider, QWidget, QSizePolicy, QFileDialog, QMessageBox,
)

EPS = 1e-4
MIN_SKIN_PIXELS = 60
WORK_MAX = 900          # images are downscaled to this many px on the long side for analysis


# ------------------------------------------------------------------ colour math
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


_SRGB2LIN = srgb_to_linear(np.arange(256) / 255.0)


def qimage_to_rgb(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=bpl * h)
    return buf.reshape(h, bpl)[:, :w * 3].reshape(h, w, 3).copy()


def rgb_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Classic YCrCb skin classifier; good enough to drop hair, eyes, teeth and background."""
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 128 + 0.5 * r - 0.4187 * g - 0.0813 * b
    cb = 128 - 0.1687 * r - 0.3313 * g + 0.5 * b
    return (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135) & (y >= 40) & (y <= 250)


@dataclass
class SkinStats:
    log_mean: np.ndarray      # (3,) mean of log(linear channel)
    log_lum_std: float        # spread of log(linear luminance)
    count: int
    mean_rgb: tuple[int, int, int]
    used_skin_filter: bool


def select_pixels(rgb: np.ndarray, box: QRectF | None, use_skin: bool) -> tuple[np.ndarray | None, bool]:
    """Pixels (n, 3) inside the box, narrowed to skin-coloured ones when possible.

    Returns the flat indices into the box region as well, so the same pixels can be
    re-read from a corrected image.
    """
    h, w = rgb.shape[:2]
    if box is not None:
        x0, y0 = max(int(box.left()), 0), max(int(box.top()), 0)
        x1, y1 = min(int(box.right()) + 1, w), min(int(box.bottom()) + 1, h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None, False
        region = (slice(y0, y1), slice(x0, x1))
    else:
        region = (slice(0, h), slice(0, w))
    sub = rgb[region]
    idx = np.arange(sub.shape[0] * sub.shape[1])
    used = False
    if use_skin:
        m = skin_mask(sub).reshape(-1)
        if m.sum() >= MIN_SKIN_PIXELS:
            idx = idx[m]
            used = True
    if len(idx) < 4:
        return None, False
    return (region, idx), used


def pixels_at(rgb: np.ndarray, sel) -> np.ndarray:
    region, idx = sel
    return rgb[region].reshape(-1, 3)[idx]


def sample_stats(rgb: np.ndarray, box: QRectF | None, use_skin: bool, sel=None) -> SkinStats | None:
    if sel is None:
        sel, used = select_pixels(rgb, box, use_skin)
        if sel is None:
            return None
    else:
        used = True
    px = pixels_at(rgb, sel)
    lin = _SRGB2LIN[px]                                   # (n, 3)
    logs = np.log(lin + EPS)
    lum = 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]
    return SkinStats(logs.mean(axis=0), float(np.log(lum + EPS).std()), len(px),
                     tuple(int(v) for v in px.mean(axis=0)), used)


@dataclass
class Correction:
    gains: tuple[float, float, float]
    gamma: float

    def describe(self) -> str:
        r, g, b = self.gains
        return f"gain R {r:.2f}  G {g:.2f}  B {b:.2f}   contrast {self.gamma:.2f}"


def fit(video: SkinStats, ref: SkinStats, match_contrast: bool = True, strength: float = 1.0) -> Correction:
    g = 1.0
    if match_contrast:
        g = float(np.clip(ref.log_lum_std / max(video.log_lum_std, 1e-3), 0.4, 2.5))
    logk = ref.log_mean - g * video.log_mean
    k = np.clip(np.exp(logk), 0.1, 10.0)
    # strength blends toward the identity in parameter space
    g = 1.0 + strength * (g - 1.0)
    k = np.exp(strength * np.log(k))
    return Correction((float(k[0]), float(k[1]), float(k[2])), g)


def build_tables(corr: Correction) -> np.ndarray:
    """(3, 256) uint8: sRGB in -> sRGB out per channel."""
    tables = np.empty((3, 256), dtype=np.uint8)
    for c in range(3):
        out = corr.gains[c] * np.power(_SRGB2LIN, corr.gamma)
        tables[c] = np.clip(np.round(linear_to_srgb(out) * 255.0), 0, 255).astype(np.uint8)
    return tables


def apply_tables(rgb: np.ndarray, tables: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = tables[c][rgb[..., c]]
    return out


def write_cube_1d(tables: np.ndarray) -> str:
    """Write the tables as a 1D .cube LUT in the temp folder; returns the path."""
    lines = ["TITLE \"RulZ skin match\"", "LUT_1D_SIZE 256", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    for i in range(256):
        lines.append(f"{tables[0][i] / 255:.6f} {tables[1][i] / 255:.6f} {tables[2][i] / 255:.6f}")
    content = "\n".join(lines) + "\n"
    d = temp_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"match_{hashlib.md5(content.encode()).hexdigest()[:12]}.cube")
    if not os.path.exists(path):
        with open(path, "w", encoding="ascii") as f:
            f.write(content)
    else:
        try:
            os.utime(path)  # in use again: clean_temp() goes by age
        except OSError:
            pass
    return path


def ffmpeg_lut_filter(path: str) -> str:
    """lut1d filter string with the path escaped for ffmpeg's filtergraph parser."""
    p = path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    return f"lut1d=file='{p}':interp=linear"


def load_image(path: str) -> QImage | None:
    """Load a photo honouring EXIF rotation; fall back to ffmpeg for formats Qt lacks (HEIC)."""
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    img = reader.read()
    if not img.isNull():
        return img
    tmp = os.path.join(temp_dir(), "ref_" + hashlib.md5(path.encode("utf-8")).hexdigest()[:12] + ".png")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-frames:v", "1", tmp],
                       capture_output=True, text=True, creationflags=0x08000000)
    if r.returncode != 0:
        return None
    img = QImage(tmp)
    try:
        os.remove(tmp)  # the pixels are in memory now
    except OSError:
        pass
    return None if img.isNull() else img


def temp_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "simplecut")


def clean_temp(max_age_s: float = 24 * 3600) -> int:
    """Delete LUTs / reference images left in the temp folder by earlier sessions. Files younger
    than max_age_s are kept: another running copy of the app may still be exporting with one."""
    d = temp_dir()
    removed = 0
    now = time.time()
    try:
        names = os.listdir(d)
    except OSError:
        return 0
    for name in names:
        if not (name.startswith(("match_", "ref_")) and name.endswith((".cube", ".png"))):
            continue
        p = os.path.join(d, name)
        try:
            if now - os.path.getmtime(p) > max_age_s:
                os.remove(p)
                removed += 1
        except OSError:
            pass
    return removed


def downscale(img: QImage, max_side: int = WORK_MAX) -> QImage:
    if max(img.width(), img.height()) <= max_side:
        return img
    return img.scaled(max_side, max_side, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ----------------------------------------------------------------- widgets
class SampleView(QWidget):
    """Shows an image; drag to draw the sample box. Highlights detected skin inside it."""
    boxChanged = Signal()

    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.rgb: np.ndarray | None = None
        self._pix: QPixmap | None = None
        self._overlay: QPixmap | None = None
        self.box: QRectF | None = None      # image px
        self.show_mask = True
        self._press: QPointF | None = None
        self._disp = QRectF()

    def set_image(self, rgb: np.ndarray | None):
        self.rgb = rgb
        self._pix = QPixmap.fromImage(rgb_to_qimage(rgb)) if rgb is not None else None
        self.box = None
        self._rebuild_overlay()
        self.update()

    def set_show_mask(self, on: bool):
        self.show_mask = on
        self._rebuild_overlay()
        self.update()

    def _rebuild_overlay(self):
        self._overlay = None
        if self.rgb is None or not self.show_mask:
            return
        m = skin_mask(self.rgb)
        h, w = m.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[m] = (60, 255, 120, 90)
        img = QImage(np.ascontiguousarray(rgba).data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        self._overlay = QPixmap.fromImage(img)

    def _layout(self):
        if self._pix is None:
            self._disp = QRectF()
            return
        W, H = self.width(), self.height() - 18
        s = min(W / self._pix.width(), H / self._pix.height())
        w, h = self._pix.width() * s, self._pix.height() * s
        self._disp = QRectF((W - w) / 2, 18 + (H - h) / 2, w, h)

    def _to_img(self, p: QPointF) -> QPointF:
        s = self._disp.width() / self._pix.width()
        x = min(max((p.x() - self._disp.x()) / s, 0), self._pix.width() - 1)
        y = min(max((p.y() - self._disp.y()) / s, 0), self._pix.height() - 1)
        return QPointF(x, y)

    def _to_widget(self, r: QRectF) -> QRectF:
        s = self._disp.width() / self._pix.width()
        return QRectF(self._disp.x() + r.x() * s, self._disp.y() + r.y() * s, r.width() * s, r.height() * s)

    def mousePressEvent(self, e):
        if self._pix is not None and e.button() == Qt.LeftButton:
            self._press = self._to_img(e.position())
            self.box = QRectF(self._press, self._press)

    def mouseMoveEvent(self, e):
        if self._press is not None:
            self.box = QRectF(self._press, self._to_img(e.position())).normalized()
            self.update()

    def mouseReleaseEvent(self, e):
        if self._press is not None:
            self.box = QRectF(self._press, self._to_img(e.position())).normalized()
            self._press = None
            if self.box.width() < 4 or self.box.height() < 4:
                self.box = None
            self.update()
            self.boxChanged.emit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout()

    def paintEvent(self, _):
        self._layout()
        p = QPainter(self)
        p.fillRect(self.rect(), theme.Q_BG)
        p.setPen(theme.Q_TEXT)
        p.drawText(QRectF(0, 0, self.width(), 18), Qt.AlignCenter, self.title)
        if self._pix is None:
            p.setPen(theme.Q_MUTED)
            p.drawText(self.rect().adjusted(0, 18, 0, 0), Qt.AlignCenter, "No image")
            p.end()
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawPixmap(self._disp.toRect(), self._pix)
        if self.box is not None:
            wr = self._to_widget(self.box)
            if self._overlay is not None:
                p.setClipRect(wr)
                p.drawPixmap(self._disp.toRect(), self._overlay)
                p.setClipping(False)
            p.setPen(QPen(theme.Q_RED_HOVER, 2))
            p.drawRect(wr)
        elif self._overlay is not None:
            p.setOpacity(0.6)
            p.drawPixmap(self._disp.toRect(), self._overlay)
            p.setOpacity(1.0)
        p.end()


class ResultView(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pix: QPixmap | None = None

    def set_rgb(self, rgb: np.ndarray | None):
        self._pix = QPixmap.fromImage(rgb_to_qimage(rgb)) if rgb is not None else None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), theme.Q_BG)
        p.setPen(theme.Q_TEXT)
        p.drawText(QRectF(0, 0, self.width(), 18), Qt.AlignCenter, self.title)
        if self._pix is not None:
            W, H = self.width(), self.height() - 18
            s = min(W / self._pix.width(), H / self._pix.height())
            w, h = self._pix.width() * s, self._pix.height() * s
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawPixmap(QRectF((W - w) / 2, 18 + (H - h) / 2, w, h).toRect(), self._pix)
        p.end()


# ------------------------------------------------------------------ dialog
class ColorMatchDialog(QDialog):
    """Pick skin samples on the frame and a reference photo; returns the fitted tables."""

    def __init__(self, frame: QImage, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Match skin tone to reference photo")
        self.resize(1250, 640)
        self.frame_rgb = qimage_to_rgb(downscale(frame))
        self.ref_rgb: np.ndarray | None = None
        self.correction: Correction | None = None
        self.tables: np.ndarray | None = None

        self.v_view = SampleView("Video frame: drag a box over the person's skin")
        self.r_view = SampleView("Reference photo: drag a box over the same person's skin")
        self.out_view = ResultView("Result")
        self.v_view.set_image(self.frame_rgb)
        self.v_view.boxChanged.connect(self._recompute)
        self.r_view.boxChanged.connect(self._recompute)

        lay = QVBoxLayout(self)
        views = QHBoxLayout()
        views.addWidget(self.v_view, 1)
        views.addWidget(self.r_view, 1)
        views.addWidget(self.out_view, 1)
        lay.addLayout(views, 1)

        row = QHBoxLayout()
        self.b_photo = QPushButton("Choose reference photo…")
        self.b_photo.clicked.connect(self._choose_photo)
        self.photo_label = QLabel("No photo loaded")
        self.photo_label.setObjectName("note")
        row.addWidget(self.b_photo)
        row.addWidget(self.photo_label, 1)
        lay.addLayout(row)

        opts = QHBoxLayout()
        self.cb_skin = QCheckBox("Use only skin-coloured pixels inside the boxes (green tint)")
        self.cb_skin.setChecked(True)
        self.cb_skin.toggled.connect(self._skin_toggled)
        self.cb_contrast = QCheckBox("Match contrast (lighting spread)")
        self.cb_contrast.setChecked(True)
        self.cb_contrast.toggled.connect(self._recompute)
        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(0, 100)
        self.strength.setValue(100)
        self.strength.setFixedWidth(160)
        self.strength.valueChanged.connect(self._recompute)
        self.strength_label = QLabel("100%")
        opts.addWidget(self.cb_skin)
        opts.addSpacing(16)
        opts.addWidget(self.cb_contrast)
        opts.addSpacing(16)
        opts.addWidget(QLabel("Strength"))
        opts.addWidget(self.strength)
        opts.addWidget(self.strength_label)
        opts.addStretch(1)
        lay.addLayout(opts)

        self.stats_label = QLabel("Load a reference photo, then drag a box over the skin in each image "
                                  "(no box = whole image).")
        self.stats_label.setWordWrap(True)
        lay.addWidget(self.stats_label)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Apply")
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        self.ok_btn = btns.button(QDialogButtonBox.Ok)
        self.ok_btn.setEnabled(False)
        lay.addWidget(btns)

        last = _settings().value("ref_photo", "", type=str)
        if last and os.path.isfile(last):
            self._load_photo(last)

    # --- photo
    def _choose_photo(self):
        start = _settings().value("ref_photo", "", type=str)
        p, _ = QFileDialog.getOpenFileName(self, "Reference photo", os.path.dirname(start) if start else "",
                                           "Images (*.jpg *.jpeg *.png *.heic *.heif *.webp *.bmp *.tif *.tiff);;All files (*.*)")
        if p:
            self._load_photo(p)

    def _load_photo(self, path: str):
        img = load_image(path)
        if img is None:
            QMessageBox.warning(self, "Reference photo", f"Could not read:\n{path}")
            return
        self.ref_rgb = qimage_to_rgb(downscale(img))
        self.r_view.set_image(self.ref_rgb)
        self.photo_label.setText(path)
        _settings().setValue("ref_photo", path)
        self._recompute()

    def _skin_toggled(self, on: bool):
        self.v_view.set_show_mask(on)
        self.r_view.set_show_mask(on)
        self._recompute()

    # --- fit
    def _recompute(self, *_):
        self.strength_label.setText(f"{self.strength.value()}%")
        self.correction = self.tables = None
        if self.ref_rgb is None:
            self.out_view.set_rgb(None)
            self.ok_btn.setEnabled(False)
            return
        use_skin = self.cb_skin.isChecked()
        vsel, _ = select_pixels(self.frame_rgb, self.v_view.box, use_skin)
        vs = sample_stats(self.frame_rgb, self.v_view.box, use_skin)
        rs = sample_stats(self.ref_rgb, self.r_view.box, use_skin)
        if vs is None or rs is None:
            self.stats_label.setText("Sample box too small.")
            self.out_view.set_rgb(None)
            self.ok_btn.setEnabled(False)
            return
        self.correction = fit(vs, rs, self.cb_contrast.isChecked(), self.strength.value() / 100)
        self.tables = build_tables(self.correction)
        corrected = apply_tables(self.frame_rgb, self.tables)
        self.out_view.set_rgb(corrected)
        after = sample_stats(corrected, None, False, sel=vsel)
        def px(s):
            return f"RGB({s.mean_rgb[0]}, {s.mean_rgb[1]}, {s.mean_rgb[2]})"
        warn = ""
        if not vs.used_skin_filter or not rs.used_skin_filter:
            warn = "  ⚠ too few skin-coloured pixels in a box, used every pixel in it."
        self.stats_label.setText(
            f"Video skin {px(vs)} ({vs.count} px)   →   reference skin {px(rs)} ({rs.count} px)   |   "
            f"{self.correction.describe()}   |   same skin pixels after correction {px(after)}{warn}")
        self.ok_btn.setEnabled(True)

    def _accept(self):
        if self.tables is None:
            return
        self.accept()
