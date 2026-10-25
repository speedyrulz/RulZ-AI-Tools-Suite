"""User preferences (Settings -> Preferences...), stored with QSettings.

Built-in values live in ExportSettings' field defaults and the constants below;
anything the user changes here is layered on top of them.
"""
from __future__ import annotations

from dataclasses import dataclass

import os

from PySide6.QtCore import QRectF, QSettings, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QRadioButton, QPushButton,
    QMessageBox, QLabel, QCheckBox, QTabWidget, QWidget, QFileDialog, QFrame,
)

from .export import ExportSettings, NVENC_CODECS, FPS_PRESETS, parse_fps
from . import theme

ASPECT_PRESETS = {
    "Free": None,
    "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4, "1:1": 1.0,
    "21:9": 21 / 9, "2.39:1": 2.39, "3:2": 3 / 2, "2:3": 2 / 3,
    "Custom…": "custom",
}
PRESET_LABELS = ["p1 (fastest)", "p3", "p4", "p5 (balanced)", "p6", "p7 (best)"]
AUDIO_LABELS = ["Copy audio if MP4-compatible, else AAC", "Re-encode audio to AAC", "No audio"]
AUDIO_KEYS = ["copy", "aac", "none"]

BUILTIN_ASPECT = "Free"
BUILTIN_FIXED_SECONDS = 2.0
BUILTIN_FIXED_FRAMES = 60
BUILTIN_FIXED_UNIT = "seconds"


SETTINGS_ENV = "SIMPLECUT_SETTINGS_INI"


def settings() -> QSettings:
    """The app's preferences: QSettings("SimpleCut", "SimpleCut") (the registry on Windows).

    Set the environment variable SIMPLECUT_SETTINGS_INI to a file path to use that .ini file
    instead, e.g. to test with a fresh profile without touching your real preferences.
    """
    ini = os.environ.get(SETTINGS_ENV)
    if ini:
        return QSettings(ini, QSettings.IniFormat)
    return QSettings("SimpleCut", "SimpleCut")


# ------------------------------------------------------------------- theme
THEME_KEY = "ui/theme"   # "dark" | "light"; missing until the first-run "Choose your look" prompt


def stored_theme() -> str | None:
    """The saved theme, or None if the user hasn't chosen one yet (first run)."""
    v = settings().value(THEME_KEY, "", type=str)
    return v if v in (theme.DARK, theme.LIGHT) else None


def save_theme(mode: str):
    settings().setValue(THEME_KEY, theme.LIGHT if mode == theme.LIGHT else theme.DARK)


def parse_aspect(text: str) -> float | None:
    """'16:9', '16/9', '1.777' -> width/height. Raises ValueError on junk."""
    t = text.strip().replace("/", ":")
    if not t:
        return None
    if ":" in t:
        a, b = t.split(":", 1)
        a, b = float(a), float(b)
        if a <= 0 or b <= 0:
            raise ValueError("Aspect must be positive")
        return a / b
    v = float(t)
    if v <= 0:
        raise ValueError("Aspect must be positive")
    return v


# ---------------------------------------------------------------- load/save
# Export settings live under "defaults/..." (the Export dialog) and "quick/..."
# (Quick export). Quick export follows the Export ones unless "quick/same_as_export"
# is switched off.

def _load_encode(prefix: str, base: ExportSettings | None = None) -> ExportSettings:
    st = settings()
    s = ExportSettings()
    if base is not None:
        s.codec, s.cq, s.preset, s.audio, s.fps = base.codec, base.cq, base.preset, base.audio, base.fps
    s.codec = st.value(f"{prefix}/codec", s.codec, type=str)
    s.cq = st.value(f"{prefix}/cq", s.cq, type=int)
    s.preset = st.value(f"{prefix}/preset", s.preset, type=str)
    s.audio = st.value(f"{prefix}/audio", s.audio, type=str)
    if st.contains(f"{prefix}/fps"):
        s.fps = st.value(f"{prefix}/fps", "", type=str) or None   # "" means "Same as source"
    if s.codec not in NVENC_CODECS.values():
        s.codec = ExportSettings.codec
    if s.audio not in AUDIO_KEYS:
        s.audio = ExportSettings.audio
    return s


def _save_encode(prefix: str, s: ExportSettings):
    st = settings()
    st.setValue(f"{prefix}/codec", s.codec)
    st.setValue(f"{prefix}/cq", s.cq)
    st.setValue(f"{prefix}/preset", s.preset)
    st.setValue(f"{prefix}/audio", s.audio)
    st.setValue(f"{prefix}/fps", s.fps or "")


def load_export_defaults() -> ExportSettings:
    """Encode settings the Export dialog opens with."""
    return _load_encode("defaults")


def prefer_lossless() -> bool:
    """Export dialog: pre-select Lossless when the cut allows it."""
    return settings().value("defaults/prefer_lossless", True, type=bool)


def quick_same_as_export() -> bool:
    return settings().value("quick/same_as_export", True, type=bool)


def quick_lossless_when_possible() -> bool:
    return settings().value("quick/lossless_when_possible", False, type=bool)


def load_quick_defaults() -> ExportSettings:
    """Encode settings Quick export uses."""
    exp = load_export_defaults()
    if quick_same_as_export():
        return exp
    return _load_encode("quick", base=exp)


@dataclass
class UiDefaults:
    aspect: str = BUILTIN_ASPECT           # key of ASPECT_PRESETS
    custom_aspect: str = ""                # "W:H" text when aspect == Custom
    fixed_seconds: float = BUILTIN_FIXED_SECONDS
    fixed_frames: int = BUILTIN_FIXED_FRAMES
    fixed_unit: str = BUILTIN_FIXED_UNIT   # "seconds" | "frames"


def load_ui_defaults() -> UiDefaults:
    st = settings()
    d = UiDefaults()
    d.aspect = st.value("defaults/aspect", d.aspect, type=str)
    if d.aspect not in ASPECT_PRESETS:
        d.aspect = BUILTIN_ASPECT
    d.custom_aspect = st.value("defaults/custom_aspect", "", type=str)
    d.fixed_seconds = st.value("defaults/fixed_seconds", d.fixed_seconds, type=float)
    d.fixed_frames = st.value("defaults/fixed_frames", d.fixed_frames, type=int)
    d.fixed_unit = st.value("defaults/fixed_unit", d.fixed_unit, type=str)
    if d.fixed_unit not in ("seconds", "frames"):
        d.fixed_unit = BUILTIN_FIXED_UNIT
    return d


def save_ui_defaults(ui: UiDefaults):
    st = settings()
    st.setValue("defaults/aspect", ui.aspect)
    st.setValue("defaults/custom_aspect", ui.custom_aspect)
    st.setValue("defaults/fixed_seconds", ui.fixed_seconds)
    st.setValue("defaults/fixed_frames", ui.fixed_frames)
    st.setValue("defaults/fixed_unit", ui.fixed_unit)


def save_defaults(exp: ExportSettings, ui: UiDefaults):
    """Back-compat helper: export encode settings + crop/clip defaults."""
    _save_encode("defaults", exp)
    save_ui_defaults(ui)


def clear_defaults():
    st = settings()
    for group in ("defaults", "quick"):
        st.beginGroup(group)
        st.remove("")
        st.endGroup()


# ------------------------------------------------------------------ widgets
class EncodeForm(QWidget):
    """Codec / CQ / preset / audio / frame rate editor, shared by both export tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.codec = QComboBox()
        self.codec.addItems(list(NVENC_CODECS.keys()))
        self.cq = QSpinBox()
        self.cq.setRange(0, 51)
        self.cq.setToolTip("NVENC constant quality: lower = better quality / bigger file. 18-24 is typical.")
        self.preset = QComboBox()
        self.preset.addItems(PRESET_LABELS)
        self.audio = QComboBox()
        self.audio.addItems(AUDIO_LABELS)
        self.fps = QComboBox()
        self.fps.setEditable(True)
        self.fps.addItems(list(FPS_PRESETS.keys()))
        self.fps.setToolTip("Pick a rate or type one (e.g. 48 or 24000/1001).")
        form.addRow("Codec", self.codec)
        form.addRow("Quality (CQ)", self.cq)
        form.addRow("Preset", self.preset)
        form.addRow("Audio", self.audio)
        form.addRow("Frame rate", self.fps)

    def load(self, s: ExportSettings):
        for key, val in NVENC_CODECS.items():
            if val == s.codec:
                self.codec.setCurrentText(key)
        self.cq.setValue(s.cq)
        for i, label in enumerate(PRESET_LABELS):
            if label.split()[0] == s.preset:
                self.preset.setCurrentIndex(i)
        self.audio.setCurrentIndex(AUDIO_KEYS.index(s.audio))
        for key, val in FPS_PRESETS.items():
            if val == s.fps:
                self.fps.setCurrentText(key)
                break
        else:
            self.fps.setCurrentText(s.fps or "Same as source")

    def read(self) -> ExportSettings:
        """Raises ValueError on a bad frame rate."""
        s = ExportSettings()
        s.codec = NVENC_CODECS[self.codec.currentText()]
        s.cq = self.cq.value()
        s.preset = self.preset.currentText().split()[0]
        s.audio = AUDIO_KEYS[self.audio.currentIndex()]
        s.fps = parse_fps(self.fps.currentText())
        return s


def _note(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("note")
    lab.setWordWrap(True)
    return lab


# ------------------------------------------------------------------ dialog
class PreferencesDialog(QDialog):
    """Settings -> Preferences. On OK everything is saved; the caller re-applies it."""

    def __init__(self, parent=None, start_tab: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        # --- Export tab
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(_note("What the Export dialog (Ctrl+E) opens with."))
        self.exp_form = EncodeForm()
        v.addWidget(self.exp_form)
        self.cb_prefer_lossless = QCheckBox("Pre-select Lossless when the In and Out points allow it")
        v.addWidget(self.cb_prefer_lossless)
        v.addStretch(1)
        self.tabs.addTab(page, "Export")

        # --- Quick export tab
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(_note("Quick export (Ctrl+Shift+E) saves straight to the output folder with these "
                          "settings, no dialog. Rotation, flip, crop, scale and colour matching are always applied."))
        self.cb_same = QCheckBox("Use the same settings as Export")
        v.addWidget(self.cb_same)
        self.quick_form = EncodeForm()
        v.addWidget(self.quick_form)
        self.cb_quick_lossless = QCheckBox("Cut losslessly instead of re-encoding when the clip allows it "
                                           "(keyframe-aligned, no crop / flip / colour match)")
        v.addWidget(self.cb_quick_lossless)
        v.addStretch(1)
        self.cb_same.toggled.connect(self._same_toggled)
        self.tabs.addTab(page, "Quick export")

        # --- Crop & clip tab
        page = QWidget()
        v = QVBoxLayout(page)
        g = QGroupBox("Crop")
        form = QFormLayout(g)
        row = QHBoxLayout()
        self.aspect = QComboBox()
        self.aspect.addItems(list(ASPECT_PRESETS.keys()))
        self.custom_aspect = QLineEdit()
        self.custom_aspect.setPlaceholderText("W:H e.g. 5:4")
        self.custom_aspect.setMaximumWidth(100)
        row.addWidget(self.aspect)
        row.addWidget(self.custom_aspect)
        row.addStretch(1)
        form.addRow("Aspect ratio", row)
        self.aspect.currentTextChanged.connect(
            lambda k: self.custom_aspect.setVisible(ASPECT_PRESETS.get(k) == "custom"))
        v.addWidget(g)
        g = QGroupBox("Fixed length clip")
        form = QFormLayout(g)
        self.fixed_seconds = QDoubleSpinBox()
        self.fixed_seconds.setDecimals(2)
        self.fixed_seconds.setRange(0.01, 1e6)
        self.fixed_seconds.setSuffix(" s")
        self.fixed_frames = QSpinBox()
        self.fixed_frames.setRange(1, 10_000_000)
        self.fixed_frames.setSuffix(" frames")
        self.rb_seconds = QRadioButton("Use seconds")
        self.rb_frames = QRadioButton("Use frames")
        row = QHBoxLayout()
        row.addWidget(self.rb_seconds)
        row.addWidget(self.fixed_seconds)
        row.addSpacing(12)
        row.addWidget(self.rb_frames)
        row.addWidget(self.fixed_frames)
        row.addStretch(1)
        form.addRow("Length", row)
        form.addRow(_note("The chosen unit is the one exported exactly; the other box is derived "
                          "from the video's frame rate."))
        v.addWidget(g)
        v.addStretch(1)
        self.tabs.addTab(page, "Crop && clip")

        # --- Output tab
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(_note("Where Export, Quick export and frame PNGs are saved."))
        self.rb_next_to = QRadioButton("Next to the source video")
        self.rb_folder = QRadioButton("This folder:")
        self.folder = QLineEdit()
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.rb_folder)
        row.addWidget(self.folder, 1)
        row.addWidget(b)
        v.addWidget(self.rb_next_to)
        v.addLayout(row)
        v.addStretch(1)
        self.folder.textEdited.connect(lambda _: self.rb_folder.setChecked(True))
        self.tabs.addTab(page, "Output")

        # --- Appearance tab
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(_note("Also switchable any time with Settings \u25b8 Light mode."))
        self.rb_dark = QRadioButton("Dark (black and red)")
        self.rb_light = QRadioButton("Light (white and red)")
        v.addWidget(self.rb_dark)
        v.addWidget(self.rb_light)
        v.addStretch(1)
        self.tabs.addTab(page, "Appearance")

        # --- buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.b_reset = QPushButton("Restore built-in defaults")
        btns.addButton(self.b_reset, QDialogButtonBox.ResetRole)
        self.b_reset.clicked.connect(self._restore_builtin)
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self._load_all()
        self.tabs.setCurrentIndex(start_tab)
        self.resize(640, self.sizeHint().height())

    # --- helpers
    def _same_toggled(self, on: bool):
        self.quick_form.setEnabled(not on)
        if on:
            try:
                self.quick_form.load(self.exp_form.read())
            except ValueError:
                pass

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.folder.text())
        if d:
            self.folder.setText(d)
            self.rb_folder.setChecked(True)

    def _load_ui(self, ui: UiDefaults):
        self.aspect.setCurrentText(ui.aspect)
        self.custom_aspect.setText(ui.custom_aspect)
        self.custom_aspect.setVisible(ASPECT_PRESETS.get(ui.aspect) == "custom")
        self.fixed_seconds.setValue(ui.fixed_seconds)
        self.fixed_frames.setValue(ui.fixed_frames)
        (self.rb_frames if ui.fixed_unit == "frames" else self.rb_seconds).setChecked(True)

    def _load_all(self):
        self.exp_form.load(load_export_defaults())
        self.cb_prefer_lossless.setChecked(prefer_lossless())
        same = quick_same_as_export()
        self.quick_form.load(load_quick_defaults())
        self.cb_same.setChecked(same)
        self.quick_form.setEnabled(not same)
        self.cb_quick_lossless.setChecked(quick_lossless_when_possible())
        self._load_ui(load_ui_defaults())
        d = settings().value("output_dir", "", type=str)
        self.folder.setText(d)
        (self.rb_folder if d else self.rb_next_to).setChecked(True)
        (self.rb_light if theme.is_light() else self.rb_dark).setChecked(True)

    def _restore_builtin(self):
        """Reset the Export, Quick export and Crop & clip tabs (the output folder is left alone)."""
        self.exp_form.load(ExportSettings())
        self.cb_prefer_lossless.setChecked(True)
        self.cb_same.setChecked(True)
        self.quick_form.load(ExportSettings())
        self.cb_quick_lossless.setChecked(False)
        self._load_ui(UiDefaults())

    def _save(self):
        try:
            exp = self.exp_form.read()
        except ValueError as e:
            self.tabs.setCurrentIndex(0)
            QMessageBox.warning(self, "Preferences", f"Export: {e}")
            return
        same = self.cb_same.isChecked()
        quick = exp
        if not same:
            try:
                quick = self.quick_form.read()
            except ValueError as e:
                self.tabs.setCurrentIndex(1)
                QMessageBox.warning(self, "Preferences", f"Quick export: {e}")
                return
        ui = UiDefaults()
        ui.aspect = self.aspect.currentText()
        ui.custom_aspect = self.custom_aspect.text().strip()
        if ASPECT_PRESETS.get(ui.aspect) == "custom":
            try:
                if parse_aspect(ui.custom_aspect) is None:
                    raise ValueError("empty")
            except ValueError:
                self.tabs.setCurrentIndex(2)
                QMessageBox.warning(self, "Preferences", "Custom aspect must look like 16:9 or 1.78")
                return
        ui.fixed_seconds = self.fixed_seconds.value()
        ui.fixed_frames = self.fixed_frames.value()
        ui.fixed_unit = "frames" if self.rb_frames.isChecked() else "seconds"
        folder = self.folder.text().strip()
        if self.rb_folder.isChecked() and (not folder or not os.path.isdir(folder)):
            self.tabs.setCurrentIndex(3)
            QMessageBox.warning(self, "Preferences", "Choose an existing output folder.")
            return

        st = settings()
        _save_encode("defaults", exp)
        st.setValue("defaults/prefer_lossless", self.cb_prefer_lossless.isChecked())
        st.setValue("quick/same_as_export", same)
        _save_encode("quick", quick)
        st.setValue("quick/lossless_when_possible", self.cb_quick_lossless.isChecked())
        save_ui_defaults(ui)
        if self.rb_folder.isChecked():
            st.setValue("output_dir", folder)
        else:
            st.remove("output_dir")
        save_theme(theme.LIGHT if self.rb_light.isChecked() else theme.DARK)
        self.accept()


DefaultsDialog = PreferencesDialog   # old name


# ------------------------------------------------------------- first run
class _LookPreview(QWidget):
    """A tiny drawing of the app in one theme (fixed colours, whatever the current theme is)."""

    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.setFixedSize(220, 128)
        self._logo = theme.logo_pixmap(12, dark=(mode == theme.DARK))
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, _):
        c = {k: QColor(v) for k, v in theme.PALETTES[self.mode].items() if v.startswith("#")}
        dark = self.mode == theme.DARK
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(r, 6, 6)
        p.setClipPath(path)
        p.fillRect(r, c["BG"])
        # menu bar with the red rule and the logo
        p.fillRect(QRectF(r.left(), r.top(), r.width(), 16), c["BAR"])
        p.fillRect(QRectF(r.left(), r.top() + 16, r.width(), 2), c["RED"])
        for i, w in enumerate((18, 26)):
            p.fillRect(QRectF(r.left() + 8 + i * 26, r.top() + 6, w, 4), c["MUTED"])
        if not self._logo.isNull():
            p.drawPixmap(int(r.right() - self._logo.width() - 6), int(r.top() + 2), self._logo)
        # picture area with a crop box
        p.fillRect(QRectF(r.left() + 40, r.top() + 24, r.width() - 80, 50), QColor("#000000") if dark else c["PANEL"])
        p.setPen(QColor(255, 255, 255, 200) if dark else c["MUTED"])
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(r.left() + 70, r.top() + 32, r.width() - 140, 34))
        # timeline with a red selection
        p.setPen(c["TL_BAR_BORDER"])
        p.setBrush(c["TL_BAR"])
        p.drawRect(QRectF(r.left() + 8, r.top() + 82, r.width() - 16, 10))
        p.fillRect(QRectF(r.left() + 70, r.top() + 82, 50, 10), c["RED"])
        # buttons: two plain, one red
        p.setPen(c["BORDER"])
        p.setBrush(c["RAISED"])
        p.drawRoundedRect(QRectF(r.left() + 8, r.top() + 102, 40, 14), 3, 3)
        p.drawRoundedRect(QRectF(r.left() + 54, r.top() + 102, 40, 14), 3, 3)
        p.setPen(Qt.NoPen)
        p.setBrush(c["RED"])
        p.drawRoundedRect(QRectF(r.right() - 58, r.top() + 102, 50, 14), 3, 3)
        p.fillRect(QRectF(r.left() + 14, r.top() + 108, 28, 2), c["TEXT"])
        p.fillRect(QRectF(r.left() + 60, r.top() + 108, 28, 2), c["TEXT"])
        p.fillRect(QRectF(r.right() - 50, r.top() + 108, 34, 2), QColor("#ffffff"))
        p.setClipping(False)
        p.setPen(QPen(c["BORDER"], 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r, 6, 6)
        p.end()


class _LookCard(QFrame):
    """A clickable card: preview + name. The selected one gets a red border."""

    def __init__(self, mode: str, title: str, on_click, parent=None):
        super().__init__(parent)
        self.mode = mode
        self._on_click = on_click
        self.setObjectName("lookCard")
        self.setCursor(Qt.PointingHandCursor)
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 12)
        v.addWidget(_LookPreview(mode), 0, Qt.AlignHCenter)
        lab = QLabel(title)
        lab.setObjectName("lookTitle")
        lab.setAlignment(Qt.AlignHCenter)
        v.addWidget(lab)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._on_click(self.mode)

    def set_selected(self, on: bool):
        self.setProperty("selected", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class ChooseLookDialog(QDialog):
    """First-run prompt: Dark (preselected) or Light. Picking a card switches the theme right away
    as a preview; closing the dialog without Continue means Dark."""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.choice = theme.DARK
        self.setWindowTitle("Choose your look")
        self.setWindowIcon(theme.app_icon())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 18)
        lay.setSpacing(12)
        head = QLabel("Choose your look")
        head.setObjectName("lookHeading")
        lay.addWidget(head)
        lay.addWidget(_note("Pick how RulZ Clip Extractor looks. You can change it later with "
                            "Settings ▸ Light mode, or in Settings ▸ Preferences ▸ Appearance."))
        row = QHBoxLayout()
        row.setSpacing(16)
        self.cards = [_LookCard(theme.DARK, "Dark", self._pick), _LookCard(theme.LIGHT, "Light", self._pick)]
        for c in self.cards:
            row.addWidget(c)
        lay.addLayout(row)
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.b_continue = QPushButton("Continue")
        self.b_continue.setObjectName("primary")
        self.b_continue.setDefault(True)
        self.b_continue.clicked.connect(self.accept)
        btns.addWidget(self.b_continue)
        lay.addLayout(btns)
        self._pick(theme.DARK)

    def _pick(self, mode: str):
        self.choice = mode
        for c in self.cards:
            c.set_selected(c.mode == mode)
        if theme.mode() != mode:
            theme.apply_mode(self.app, mode)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Left, Qt.Key_Right):
            self._pick(theme.LIGHT if self.choice == theme.DARK else theme.DARK)
            return
        super().keyPressEvent(e)

    def reject(self):
        # closed without Continue: Dark
        self._pick(theme.DARK)
        super().reject()


def ensure_theme_chosen(app) -> str:
    """The stored theme; on first run (none stored) ask with ChooseLookDialog and save the answer."""
    m = stored_theme()
    if m is None:
        dlg = ChooseLookDialog(app)
        dlg.exec()
        m = dlg.choice
        save_theme(m)
    return m
