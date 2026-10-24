"""User preferences (Settings -> Preferences...), stored with QSettings.

Built-in values live in ExportSettings' field defaults and the constants below;
anything the user changes here is layered on top of them.
"""
from __future__ import annotations

from dataclasses import dataclass

import os

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QRadioButton, QPushButton,
    QMessageBox, QLabel, QCheckBox, QTabWidget, QWidget, QFileDialog,
)

from .export import ExportSettings, NVENC_CODECS, FPS_PRESETS, parse_fps

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


def settings() -> QSettings:
    return QSettings("SimpleCut", "SimpleCut")


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
        self.accept()


DefaultsDialog = PreferencesDialog   # old name
