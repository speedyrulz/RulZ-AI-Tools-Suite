"""RulZ Simple Video Clip Extractor: frame/keyframe-accurate trimming with lossless or NVENC export."""
from __future__ import annotations

import math
import os
import sys
import traceback

import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSettings, QRect, QEvent, QUrl, QElapsedTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QImage, QKeySequence, QAction, QShortcut, QTransform
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QMessageBox, QDialog, QDialogButtonBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout, QRadioButton, QLineEdit,
    QProgressBar, QStyle, QFrame, QSlider,
)

from .probe import VideoInfo, probe, have_ffmpeg
from .decoder import Decoder
from .timeline import Timeline
from .cropview import VideoView, even
from .export import (ExportSettings, ExportJob, NVENC_CODECS, FPS_PRESETS, parse_fps,
                     build_commands, default_output_path, unique_path)
from . import theme
from .colormatch import ColorMatchDialog, apply_tables, clean_temp, write_cube_1d
from .defaults import (ASPECT_PRESETS, PRESET_LABELS, AUDIO_LABELS, AUDIO_KEYS, settings,
                       parse_aspect, load_export_defaults, load_ui_defaults, DefaultsDialog,
                       PreferencesDialog, load_quick_defaults, prefer_lossless,
                       quick_lossless_when_possible, save_theme, ensure_theme_chosen)

def output_dir() -> str | None:
    """User-chosen default folder for clips and images, or None for 'next to the source'."""
    d = settings().value("output_dir", "", type=str)
    return d if d and os.path.isdir(d) else None


def remove_partial(path: str | None):
    """Delete what a cancelled export wrote, if anything."""
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


def fmt_time(t: float) -> str:
    if t < 0:
        t = 0
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


class ProbeThread(QThread):
    done = Signal(object, int)   # info, token
    failed = Signal(str, int)    # message, token

    def __init__(self, path, token: int = 0):
        super().__init__()
        self.path = path
        self.token = token  # which open_file() call this answers; a newer one makes it stale

    def run(self):
        try:
            self.done.emit(probe(self.path), self.token)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n{traceback.format_exc()}", self.token)


NO_FFMPEG = "ffmpeg / ffprobe were not found on PATH. Install them (e.g. `winget install Gyan.FFmpeg`)."


# ----------------------------------------------------------------------------
class ExportDialog(QDialog):
    exported = Signal(str)   # path of the file that was written
    def __init__(self, info: VideoInfo, start: float, end: float,
                 lossless_ok: bool, rotation: int = 0,
                 crop: tuple[int, int, int, int] | None = None,
                 out_size: tuple[int, int] | None = None, hflip: bool = False,
                 lut_path: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export clip")
        self.info, self.start, self.end = info, start, end
        self.rotation = rotation
        self.hflip = hflip
        self.crop = crop
        self.out_size = out_size
        self.lut_path = lut_path
        needs_reencode = crop is not None or out_size is not None or hflip or lut_path is not None
        self.lossless_ok = lossless_ok and not needs_reencode

        lay = QVBoxLayout(self)
        n_frames = max(1, math.ceil((end - start) * info.fps - 1e-6))
        lay.addWidget(QLabel(
            f"Clip: {fmt_time(start)}  →  {fmt_time(end)}   "
            f"({end - start:.3f} s, {n_frames} frames)"))
        if crop:
            x, y, w, h = crop
            txt = f"Crop: {w}×{h} at ({x}, {y})"
            if out_size:
                txt += f"  →  scaled to {out_size[0]}×{out_size[1]}"
            lay.addWidget(QLabel(txt))
        elif out_size:
            lay.addWidget(QLabel(f"Scaled to {out_size[0]}×{out_size[1]}"))

        self.rb_lossless = QRadioButton("Lossless (stream copy, no re-encode)")
        self.rb_nvenc = QRadioButton("Re-encode to MP4 with NVENC")
        self.rb_lossless.setEnabled(self.lossless_ok)
        if self.lossless_ok:
            (self.rb_lossless if prefer_lossless() else self.rb_nvenc).setChecked(True)
        else:
            self.rb_nvenc.setChecked(True)
            if lut_path:
                why = "colour correction requires re-encoding"
            elif hflip:
                why = "horizontal flip requires re-encoding"
            elif needs_reencode:
                why = "crop / scale require re-encoding"
            else:
                why = "In and Out must both be on keyframes"
            self.rb_lossless.setText(f"Lossless (unavailable: {why})")
        lay.addWidget(self.rb_lossless)
        if rotation and self.lossless_ok:
            note = QLabel(f"      Rotation {rotation}° is stored as a metadata flag (players honor it; pixels are untouched).")
            note.setObjectName("note")
            lay.addWidget(note)
        lay.addWidget(self.rb_nvenc)
        if lut_path:
            note = QLabel("      Skin-tone colour correction is applied.")
            note.setObjectName("note")
            lay.addWidget(note)
        if rotation or hflip:
            what = " and ".join(([f"rotation {rotation}°"] if rotation else [])
                                + (["horizontal flip"] if hflip else []))
            note = QLabel(f"      {what[0].upper() + what[1:]} baked into the re-encoded video.")
            note.setObjectName("note")
            lay.addWidget(note)

        defaults = load_export_defaults()
        form = QFormLayout()
        self.codec = QComboBox()
        self.codec.addItems(list(NVENC_CODECS.keys()))
        for key, val in NVENC_CODECS.items():
            if val == defaults.codec:
                self.codec.setCurrentText(key)
        self.cq = QSpinBox()
        self.cq.setRange(0, 51)
        self.cq.setValue(defaults.cq)
        self.cq.setToolTip("NVENC constant quality: lower = better quality / bigger file. 18-24 is typical.")
        self.preset = QComboBox()
        self.preset.addItems(PRESET_LABELS)
        for i in range(self.preset.count()):
            if self.preset.itemText(i).split()[0] == defaults.preset:
                self.preset.setCurrentIndex(i)
        self.audio = QComboBox()
        self.audio.addItems(AUDIO_LABELS)
        self.audio.setCurrentIndex(AUDIO_KEYS.index(defaults.audio))
        self.fps = QComboBox()
        self.fps.setEditable(True)
        self.fps.addItems(list(FPS_PRESETS.keys()))
        for key, val in FPS_PRESETS.items():
            if val == defaults.fps:
                self.fps.setCurrentText(key)
                break
        else:
            self.fps.setCurrentText(defaults.fps or "Same as source")
        self.fps.setToolTip(f"Source is {info.fps:.3f} fps. Pick a rate or type one (e.g. 48 or 24000/1001). "
                            "Frames are duplicated or dropped to hit the new rate.")
        form.addRow("Codec", self.codec)
        form.addRow("Quality (CQ)", self.cq)
        form.addRow("Preset", self.preset)
        form.addRow("Audio", self.audio)
        form.addRow("Frame rate", self.fps)
        self.form_widget = QWidget()
        self.form_widget.setLayout(form)
        lay.addWidget(self.form_widget)

        row = QHBoxLayout()
        self.out_path = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(QLabel("Output"))
        row.addWidget(self.out_path, 1)
        row.addWidget(browse)
        lay.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Export")
        self.buttons.accepted.connect(self._export)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

        self.rb_lossless.toggled.connect(self._mode_changed)
        self._mode_changed()
        self.job: ExportJob | None = None
        self.resize(560, self.sizeHint().height())

    def _mode(self) -> str:
        return "lossless" if self.rb_lossless.isChecked() else "nvenc"

    def _mode_changed(self):
        mode = self._mode()
        self.form_widget.setEnabled(mode == "nvenc")
        self.out_path.setText(default_output_path(self.info, self.start, self.end, mode,
                                                  output_dir(), cropped=self.crop is not None))

    def _browse(self):
        filt = "MP4 video (*.mp4)" if self._mode() == "nvenc" else "Video (*.*)"
        p, _ = QFileDialog.getSaveFileName(self, "Save clip as", self.out_path.text(), filt)
        if p:
            self.out_path.setText(p)

    def settings(self) -> ExportSettings:
        s = ExportSettings(mode=self._mode())
        s.codec = NVENC_CODECS[self.codec.currentText()]
        s.cq = self.cq.value()
        s.preset = self.preset.currentText().split()[0]
        s.audio = AUDIO_KEYS[self.audio.currentIndex()]
        s.rotation = self.rotation
        if s.mode == "nvenc":
            s.hflip = self.hflip
            s.lut_path = self.lut_path
            s.fps = parse_fps(self.fps.currentText())
            s.crop = self.crop
            s.out_size = self.out_size
        return s

    def _export(self):
        out = self.out_path.text().strip()
        if not out:
            QMessageBox.warning(self, "Export", "Choose an output file.")
            return
        if os.path.abspath(out) == os.path.abspath(self.info.path):
            QMessageBox.warning(self, "Export", "Output must differ from the source file.")
            return
        try:
            st = self.settings()
        except ValueError as e:
            QMessageBox.warning(self, "Export", str(e))
            return
        if not have_ffmpeg():
            QMessageBox.critical(self, "Export", NO_FFMPEG)
            return
        self._out = out
        self.job = ExportJob(build_commands(self.info, self.start, self.end, out, st),
                             self.end - self.start, self)
        self.job.progress.connect(lambda f: self.progress.setValue(int(f * 1000)))
        self.job.finished.connect(self._done)
        self.progress.setVisible(True)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.job.start()

    @property
    def cmd_text(self) -> str:
        return self.job.cmd_text if self.job else ""

    def _done(self, ok: bool, log: str):
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        if self.job is not None and self.job.cancelled:
            remove_partial(self._out)  # cancelled: no error box, no half-written clip left behind
            return
        if ok:
            self.progress.setValue(1000)
            self.exported.emit(self.out_path.text())
            QMessageBox.information(self, "Export", f"Saved:\n{self.out_path.text()}")
            self.accept()
        else:
            QMessageBox.critical(self, "Export failed",
                                 f"ffmpeg failed.\n\nCommand:\n{self.cmd_text}\n\nLog:\n{log}")

    def reject(self):
        """Cancel, Esc and the close button: stop a running export first."""
        if self.job is not None and not self.job.done:
            self.job.cancel()
            self.job.wait()  # delivers `finished` (-> _done, which deletes the partial file) now
        super().reject()


# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(theme.APP_NAME)
        self.setWindowIcon(theme.app_icon())
        self.resize(1200, 820)
        self.setAcceptDrops(True)

        self.info: VideoInfo | None = None
        self.dec: Decoder | None = None
        self.mark_in: float | None = None
        self.mark_out: float | None = None
        # Screen transform = hflip^flip o rotate(rotation) o rotate(source flag).
        self.rotation = 0  # clockwise degrees added by the user, on top of the source flag
        self.flip = False  # mirror horizontally, applied after rotation
        self.color_tables = None       # (3, 256) uint8 lookup tables from the skin-tone match, or None
        self.color_lut_path: str | None = None
        self.color_desc = ""
        self._probe_thread: ProbeThread | None = None  # the newest one
        self._probe_threads: list[ProbeThread] = []    # every one still running, kept referenced
        self._probe_token = 0                          # bumped per open/close; stale probe results are dropped
        clean_temp()  # colour-match LUTs / reference images left by earlier sessions

        # fixed-length selection mode
        self.fixed = False
        ui = load_ui_defaults()
        self.fixed_unit = ui.fixed_unit  # which spinbox the user last edited
        self.fixed_seconds = ui.fixed_seconds
        self.fixed_frames = ui.fixed_frames
        self.sel_start = 0.0

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)

        # Audio: QMediaPlayer plays the file's sound and acts as the master clock
        # during playback; the frame decoder follows it. Files without audio (or
        # ones the player cannot open) fall back to a wall clock.
        self.audio_out = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.mediaStatusChanged.connect(self._audio_status)
        self.player.errorOccurred.connect(lambda err, msg: self._audio_failed(msg))
        self._audio_ok = False
        self._wall = QElapsedTimer()
        self._wall_origin = 0.0
        self.audio_out.setVolume(settings().value("volume", 80, type=int) / 100)
        self.audio_out.setMuted(settings().value("muted", False, type=bool))

        # --- widgets
        self.view = VideoView()
        self.view.cropChanged.connect(self._crop_changed)
        self.timeline = Timeline()
        self.timeline.seekRequested.connect(self.seek_to)
        self.timeline.selectionMoveRequested.connect(self._move_selection)

        self.time_label = QLabel("--:--:--.--- / --:--:--.---")
        self.frame_label = QLabel("")
        self.kf_label = QLabel("")
        self.status = self.statusBar()
        self.b_mute = QPushButton()
        self.b_mute.setCheckable(True)
        self.b_mute.setChecked(self.audio_out.isMuted())
        self.b_mute.setToolTip("Mute (M)")
        self.b_mute.setFocusPolicy(Qt.NoFocus)
        self.b_mute.toggled.connect(self._set_muted)
        self._update_mute_icon()
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(int(round(self.audio_out.volume() * 100)))
        self.vol_slider.setFixedWidth(110)
        self.vol_slider.setFocusPolicy(Qt.NoFocus)
        self.vol_slider.setToolTip("Volume")
        self.vol_slider.valueChanged.connect(self._set_volume)

        st = self.style()

        def btn(text, tip, slot, icon=None, checkable=False):
            b = QPushButton(text)
            if icon is not None:
                b.setIcon(st.standardIcon(icon))
            b.setToolTip(tip)
            b.setCheckable(checkable)
            if checkable:
                b.toggled.connect(slot)
            else:
                b.clicked.connect(slot)
            b.setFocusPolicy(Qt.NoFocus)
            return b

        self.b_start = btn("", "Go to start (Home)", self.go_start, QStyle.SP_MediaSkipBackward)
        self.b_prev_kf = btn("◀K", "Previous keyframe (Shift+Left)", self.prev_keyframe)
        self.b_prev = btn("◀", "Previous frame (Left)", self.step_back)
        self.b_play = btn("", "Play / pause (Space)", self.toggle_play, QStyle.SP_MediaPlay)
        self.b_next = btn("▶", "Next frame (Right)", self.step_forward)
        self.b_next_kf = btn("K▶", "Next keyframe (Shift+Right)", self.next_keyframe)
        self.b_end = btn("", "Go to end (End)", self.go_end, QStyle.SP_MediaSkipForward)

        self.b_in = btn("Mark In  [I]", "Set clip start at current frame", self.set_in)
        self.b_out = btn("Mark Out  [O]", "Set clip end at current frame (exclusive)", self.set_out)
        self.b_clear = btn("Clear", "Clear In/Out marks", self.clear_marks)
        self.b_export = btn("Export…  [Ctrl+E]", "Export the marked clip", self.export_clip)
        self.b_quick = btn("Quick export  [Ctrl+Shift+E]",
                           "Export the selection, or the whole video if nothing is selected, straight to the "
                           "output folder with the default NVENC settings (no dialog)", self.quick_export)
        self.quick_job: ExportJob | None = None
        self.b_quick.setObjectName("primary")
        self.b_export.setObjectName("primary")
        self.b_last = btn("Open last export", "Close the current video and open the most recently exported file",
                          self.open_last_export)
        self.last_export: str | None = settings().value("last_export", "", type=str) or None
        self._quick_cancelled = False
        self.qx_label = QLabel("")
        self.qx_progress = QProgressBar()
        self.qx_progress.setRange(0, 1000)
        self.qx_progress.setFixedWidth(180)
        self.qx_cancel = QPushButton("Cancel")
        self.qx_cancel.setFocusPolicy(Qt.NoFocus)
        self.qx_cancel.clicked.connect(self._quick_cancel)
        for wdg in (self.qx_label, self.qx_progress, self.qx_cancel):
            wdg.setVisible(False)
            self.status.addPermanentWidget(wdg)
        self.b_export.setEnabled(False)
        self.b_snap = btn("Save frame PNG  [S]", "Save the current frame as a PNG to the output folder",
                          self.save_frame, QStyle.SP_DialogSaveButton)
        self.b_rotate = btn("Rotate 90° ⟳  [R]", "Rotate the view 90° clockwise per press; exports keep it",
                            self.rotate)
        self.b_flip = btn("Flip ↔  [F]", "Mirror the view horizontally; exports keep it",
                          self.flip_h)
        self.rot_label = QLabel("")
        self.rot_label.setMinimumWidth(70)

        # crop controls
        self.b_crop = btn("Crop  [C]", "Toggle crop box. Drag inside to move, drag edges/corners to resize, "
                          "drag outside to draw a new box.", self._toggle_crop, checkable=True)
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(list(ASPECT_PRESETS.keys()))
        self.aspect_combo.currentTextChanged.connect(self._aspect_changed)
        self.aspect_edit = QLineEdit()
        self.aspect_edit.setPlaceholderText("W:H e.g. 5:4")
        self.aspect_edit.setMaximumWidth(90)
        self.aspect_edit.setVisible(False)
        self.aspect_edit.editingFinished.connect(self._custom_aspect_entered)
        self.rb_native = QRadioButton("Crop size")
        self.rb_scale = QRadioButton("Scale to")
        self.rb_native.setChecked(True)
        self.rb_scale.toggled.connect(self._scale_toggled)
        self.out_w = QSpinBox()
        self.out_h = QSpinBox()
        for sb in (self.out_w, self.out_h):
            sb.setRange(2, 16384)
            sb.setSingleStep(2)
            sb.setEnabled(False)
            sb.setKeyboardTracking(False)
        self.out_w.valueChanged.connect(lambda _: self._out_size_edited("w"))
        self.out_h.valueChanged.connect(lambda _: self._out_size_edited("h"))
        self.b_crop_reset = btn("Reset box", "Reset the crop box to the full frame", self.view.reset_crop)
        self.b_match = btn("Match colour…  [B]", "Colour-balance and relight this video so a person's skin tone "
                           "matches a reference photo", self.match_color)
        self.b_match_clear = btn("Clear", "Remove the colour correction", self.clear_color)
        self.b_match_clear.setEnabled(False)
        self.color_label = QLabel("")
        self.color_label.setObjectName("accent")
        self.crop_label = QLabel("")

        # fixed-length controls
        self.b_fixed = btn("Fixed length  [L]", "Select a clip of fixed length; drag the box on the timeline "
                           "to position it. Playback and stepping stay inside it.",
                           self._toggle_fixed, checkable=True)
        self.len_sec = QDoubleSpinBox()
        self.len_sec.setDecimals(2)
        self.len_sec.setRange(0.01, 1e6)
        self.len_sec.setSingleStep(0.5)
        self.len_sec.setSuffix(" s")
        self.len_sec.setValue(self.fixed_seconds)
        self.len_sec.setKeyboardTracking(False)
        self.len_sec.valueChanged.connect(self._len_seconds_edited)
        self.len_frames = QSpinBox()
        self.len_frames.setRange(1, 10_000_000)
        self.len_frames.setSuffix(" frames")
        self.len_frames.setValue(self.fixed_frames)
        self.len_frames.setKeyboardTracking(False)
        self.len_frames.valueChanged.connect(self._len_frames_edited)
        self.sel_label = QLabel("")
        # move the fixed-length selection box (mirrors the playback buttons)
        self.b_sel_start = btn("", "Move clip to start of video (Ctrl+Home)",
                               self.sel_to_start, QStyle.SP_MediaSkipBackward)
        self.b_sel_prev_kf = btn("◀K", "Move clip start to previous keyframe (Ctrl+Shift+Left)",
                                 lambda: self.sel_move_keyframe(-1))
        self.b_sel_prev = btn("◀", "Move clip back one frame (Ctrl+Left)",
                              lambda: self.sel_move_frames(-1))
        self.b_sel_next = btn("▶", "Move clip forward one frame (Ctrl+Right)",
                              lambda: self.sel_move_frames(1))
        self.b_sel_next_kf = btn("K▶", "Move clip start to next keyframe (Ctrl+Shift+Right)",
                                 lambda: self.sel_move_keyframe(1))
        self.b_sel_end = btn("", "Move clip to end of video (Ctrl+End)",
                             self.sel_to_end, QStyle.SP_MediaSkipForward)
        self.b_sel_here = btn("Start at playhead  [I]", "Move the clip so it starts at the current frame",
                              self.sel_start_here)
        self.b_sel_next_clip = btn("Next clip  [N]", "Move the clip to start right after the current one "
                                   "(no overlap, no gap)", self.sel_next_clip)
        self.b_sel_rewind = btn("Playhead to clip start  [Home]", "Jump the playhead to the first frame of the clip",
                                self.playhead_to_sel_start)

        self.in_label = QLabel("In: —")
        self.out_label = QLabel("Out: —")
        for lab, w in ((self.time_label, 200), (self.frame_label, 90), (self.kf_label, 90),
                       (self.in_label, 200), (self.out_label, 200)):
            lab.setMinimumWidth(w)

        controls = QHBoxLayout()
        for b in (self.b_start, self.b_prev_kf, self.b_prev, self.b_play,
                  self.b_next, self.b_next_kf, self.b_end):
            controls.addWidget(b)
        controls.addSpacing(24)
        controls.addWidget(self.time_label)
        controls.addWidget(self.frame_label)
        controls.addWidget(self.kf_label)
        controls.addSpacing(12)
        controls.addWidget(self.b_mute)
        controls.addWidget(self.vol_slider)
        controls.addStretch(1)
        controls.addWidget(self.b_rotate)
        controls.addWidget(self.b_flip)
        controls.addWidget(self.rot_label)
        controls.addWidget(self.b_snap)

        marks = QHBoxLayout()
        marks.addWidget(self.b_in)
        marks.addWidget(self.in_label)
        marks.addSpacing(16)
        marks.addWidget(self.b_out)
        marks.addWidget(self.out_label)
        marks.addSpacing(16)
        marks.addWidget(self.b_clear)
        marks.addSpacing(24)
        marks.addWidget(self.b_fixed)
        marks.addWidget(self.len_sec)
        marks.addWidget(self.len_frames)
        marks.addStretch(1)
        marks.addWidget(self.b_last)
        marks.addWidget(self.b_quick)
        marks.addWidget(self.b_export)

        crop_row = QHBoxLayout()
        crop_row.addWidget(self.b_crop)
        crop_row.addSpacing(12)
        crop_row.addWidget(QLabel("Aspect"))
        crop_row.addWidget(self.aspect_combo)
        crop_row.addWidget(self.aspect_edit)
        crop_row.addSpacing(16)
        crop_row.addWidget(QLabel("Output"))
        crop_row.addWidget(self.rb_native)
        crop_row.addWidget(self.rb_scale)
        crop_row.addWidget(self.out_w)
        crop_row.addWidget(QLabel("×"))
        crop_row.addWidget(self.out_h)
        crop_row.addSpacing(12)
        crop_row.addWidget(self.b_crop_reset)
        crop_row.addSpacing(12)
        crop_row.addWidget(self.crop_label)
        crop_row.addStretch(1)
        crop_row.addWidget(self.b_match)
        crop_row.addWidget(self.color_label)
        crop_row.addWidget(self.b_match_clear)
        self.crop_panel = QFrame()
        self.crop_panel.setLayout(crop_row)

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.timeline)
        lay.addLayout(controls)
        lay.addLayout(marks)
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Move clip"))
        sel_row.addSpacing(8)
        for b in (self.b_sel_start, self.b_sel_prev_kf, self.b_sel_prev, self.b_sel_next,
                  self.b_sel_next_kf, self.b_sel_end):
            sel_row.addWidget(b)
        sel_row.addSpacing(12)
        sel_row.addWidget(self.b_sel_here)
        sel_row.addWidget(self.b_sel_next_clip)
        sel_row.addWidget(self.b_sel_rewind)
        sel_row.addSpacing(16)
        sel_row.addWidget(QLabel("Selection:"))
        sel_row.addWidget(self.sel_label)
        sel_row.addStretch(1)
        self.sel_panel = QFrame()
        self.sel_panel.setLayout(sel_row)
        self.sel_panel.setVisible(False)
        lay.addWidget(self.sel_panel)
        lay.addWidget(self.crop_panel)
        self.setCentralWidget(root)

        # --- menu + shortcuts
        m = self.menuBar().addMenu("&File")
        a_open = QAction("&Open…", self, shortcut=QKeySequence.Open, triggered=self.open_dialog)
        a_export = QAction("&Export clip…", self, shortcut="Ctrl+E", triggered=self.export_clip)
        self.a_last = QAction("Open &last export", self, shortcut="Ctrl+Shift+O", triggered=self.open_last_export)
        self.a_close = QAction("&Close video", self, shortcut="Ctrl+W", triggered=self.close_video)
        self.a_close.setEnabled(False)
        a_quit = QAction("&Quit", self, shortcut="Ctrl+Q", triggered=self.close)
        m.addAction(a_open)
        m.addAction(a_export)
        m.addAction(self.a_last)
        m.addAction(self.a_close)
        self._refresh_last_export()
        m.addSeparator()
        m.addAction(a_quit)
        sm = self.menuBar().addMenu("&Settings")
        self.a_outdir = QAction("", self, triggered=self.choose_output_dir)
        a_outdir_reset = QAction("Save next to source video", self, triggered=self.reset_output_dir)
        a_open_outdir = QAction("Open output folder", self, triggered=self.open_output_dir)
        sm.addAction(self.a_outdir)
        sm.addAction(a_outdir_reset)
        sm.addSeparator()
        sm.addAction(a_open_outdir)
        sm.addSeparator()
        self.a_light = QAction("Light mode", self, checkable=True)
        self.a_light.setChecked(theme.is_light())
        self.a_light.setToolTip("Switch between the dark and the light look")
        self.a_light.toggled.connect(lambda on: self.set_theme(theme.LIGHT if on else theme.DARK))
        sm.addAction(self.a_light)
        sm.addSeparator()
        a_prefs = QAction("&Preferences…", self, shortcut="Ctrl+,", triggered=self.edit_preferences)
        a_prefs.setMenuRole(QAction.PreferencesRole)
        sm.addAction(a_prefs)
        logo = self.brand_logo = QLabel(self.menuBar())
        logo.setPixmap(theme.logo_pixmap(22))
        logo.setContentsMargins(0, 2, 10, 2)
        logo.setToolTip(theme.APP_NAME)
        self.menuBar().setCornerWidget(logo, Qt.TopRightCorner)
        self._refresh_outdir_action()
        self._apply_ui_defaults()

        for key, slot in (
            ("Space", self.toggle_play), ("Left", self.step_back), ("Right", self.step_forward),
            ("Shift+Left", self.prev_keyframe), ("Shift+Right", self.next_keyframe),
            ("Home", self.go_start), ("End", self.go_end),
            ("I", self.set_in), ("O", self.set_out), ("S", self.save_frame), ("R", self.rotate),
            ("F", self.flip_h), ("M", lambda: self.b_mute.toggle()),
            ("Ctrl+Shift+E", self.quick_export), ("B", self.match_color),
            ("Ctrl+Left", lambda: self.sel_move_frames(-1)), ("Ctrl+Right", lambda: self.sel_move_frames(1)),
            ("Ctrl+Shift+Left", lambda: self.sel_move_keyframe(-1)),
            ("Ctrl+Shift+Right", lambda: self.sel_move_keyframe(1)),
            ("Ctrl+Home", self.sel_to_start), ("Ctrl+End", self.sel_to_end),
            ("N", self.sel_next_clip),
            ("C", lambda: self.b_crop.toggle()), ("L", lambda: self.b_fixed.toggle()),
        ):
            QShortcut(QKeySequence(key), self, activated=slot)

        self._set_enabled(False)
        self._crop_panel_enabled(False)
        self._setup_focus()
        if not have_ffmpeg():
            QMessageBox.warning(self, "ffmpeg not found",
                                "ffmpeg / ffprobe were not found on PATH. Install them (e.g. `winget install Gyan.FFmpeg`).")

    # ------------------------------------------------------------------ open
    # ------------------------------------------------------------------ focus
    def _setup_focus(self):
        """Keep keyboard focus on the video so the single-key shortcuts keep working.

        Text/number fields only take focus when clicked, and hand it back as soon
        as the user finishes editing (Enter, Escape, picking a combo entry).
        """
        self.view.setFocusPolicy(Qt.StrongFocus)
        self.timeline.setFocusPolicy(Qt.ClickFocus)
        self._inputs = (self.aspect_combo, self.aspect_edit, self.out_w, self.out_h,
                        self.len_sec, self.len_frames)
        for wdg in self._inputs:
            wdg.setFocusPolicy(Qt.ClickFocus)
            wdg.installEventFilter(self)
        for rb in (self.rb_native, self.rb_scale):
            rb.setFocusPolicy(Qt.NoFocus)
        for sb in (self.out_w, self.out_h, self.len_sec, self.len_frames, self.aspect_edit):
            sb.editingFinished.connect(self._release_focus)
        self.aspect_combo.activated.connect(lambda _: self._release_focus())
        self.view.setFocus()

    def _release_focus(self):
        self.view.setFocus()

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.KeyPress and ev.key() == Qt.Key_Escape and obj in self._inputs:
            self._release_focus()
            return True
        return super().eventFilter(obj, ev)

    def _set_enabled(self, on: bool):
        for b in (self.b_start, self.b_prev_kf, self.b_prev, self.b_play, self.b_next,
                  self.b_next_kf, self.b_end, self.b_in, self.b_out, self.b_clear, self.b_snap,
                  self.b_rotate, self.b_flip, self.b_crop, self.b_fixed, self.len_sec,
                  self.len_frames):
            b.setEnabled(on)
        self.b_quick.setEnabled(on and self.quick_job is None)
        self.b_match.setEnabled(on)
        self.a_close.setEnabled(on)

    # ----------------------------------------------------------- colour match
    def match_color(self):
        if not self.dec or self.dec.current is None:
            return
        self.stop()
        # Show the frame as it is on screen (rotated/flipped) but without any current correction,
        # so the fit is measured against the original pixels.
        saved = self.color_tables
        self.color_tables = None
        frame = self._frame_image(self.dec.current)
        self.color_tables = saved
        dlg = ColorMatchDialog(frame, self)
        if dlg.exec() == QDialog.Accepted and dlg.tables is not None:
            self._set_color(dlg.tables, dlg.correction.describe())
            self.status.showMessage("Colour correction applied; it is used by the preview, frame PNGs and exports.", 6000)

    def clear_color(self):
        self._set_color(None, "")

    def _set_color(self, tables, desc: str):
        self.color_tables = tables
        self.color_lut_path = write_cube_1d(tables) if tables is not None else None
        self.color_desc = desc
        self.color_label.setText("colour matched" if tables is not None else "")
        self.color_label.setToolTip(desc)
        self.b_match_clear.setEnabled(tables is not None)
        if self.dec is not None and self.dec.current is not None:
            self._show_current()
        self._selection_changed()

    # --------------------------------------------------------------- defaults
    def edit_preferences(self, start_tab: int = 0):
        dlg = PreferencesDialog(self, start_tab)
        if dlg.exec() == QDialog.Accepted:
            self._apply_ui_defaults()
            self._refresh_outdir_action()
            self.set_theme(theme.LIGHT if dlg.rb_light.isChecked() else theme.DARK)
            self.status.showMessage("Preferences saved.", 4000)

    edit_defaults = edit_preferences   # old name

    # ------------------------------------------------------------------ theme
    def set_theme(self, mode: str):
        """Switch Dark/Light live and remember it (Settings > Light mode, Preferences > Appearance)."""
        save_theme(mode)
        if self.a_light.isChecked() != (mode == theme.LIGHT):
            self.a_light.blockSignals(True)
            self.a_light.setChecked(mode == theme.LIGHT)
            self.a_light.blockSignals(False)
        if theme.mode() == mode:
            return
        theme.apply_mode(QApplication.instance(), mode)
        self.brand_logo.setPixmap(theme.logo_pixmap(22))
        if self.dec is not None and self.dec.current is not None:
            self._show_current()
        self.timeline.update()
        self.view.update()

    def _apply_ui_defaults(self):
        """Push the stored crop-aspect and fixed-length defaults into the controls."""
        ui = load_ui_defaults()
        if ASPECT_PRESETS.get(ui.aspect) == "custom":
            self.aspect_edit.setText(ui.custom_aspect)
        self.aspect_combo.setCurrentText(ui.aspect)
        if ASPECT_PRESETS.get(ui.aspect) == "custom":
            self._apply_aspect()
        self.fixed_unit = ui.fixed_unit
        self.fixed_seconds = ui.fixed_seconds
        self.fixed_frames = ui.fixed_frames
        self.len_sec.blockSignals(True)
        self.len_frames.blockSignals(True)
        self.len_sec.setValue(ui.fixed_seconds)
        self.len_frames.setValue(ui.fixed_frames)
        self.len_sec.blockSignals(False)
        self.len_frames.blockSignals(False)
        if self.info:
            self._sync_length_boxes()
            self._fixed_length_changed()

    # ------------------------------------------------------------ last export
    def _note_export(self, path: str):
        self.last_export = path
        settings().setValue("last_export", path)
        self._refresh_last_export()

    def _refresh_last_export(self):
        ok = bool(self.last_export) and os.path.isfile(self.last_export)
        self.b_last.setEnabled(ok)
        self.a_last.setEnabled(ok)
        tip = f"Close the current video and open:\n{self.last_export}" if ok else "No export yet"
        self.b_last.setToolTip(tip)
        self.a_last.setToolTip(tip)

    def open_last_export(self):
        if not self.last_export or not os.path.isfile(self.last_export):
            self.status.showMessage("No exported file to open.", 4000)
            self._refresh_last_export()
            return
        if self.info:
            self.close_video()
        self.open_file(self.last_export)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.open_file(urls[0].toLocalFile())

    def open_dialog(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Video files (*.mp4 *.mkv *.mov *.avi *.ts *.m2ts *.webm *.mts *.flv *.wmv);;All files (*.*)")
        if p:
            self.open_file(p)

    def open_file(self, path: str):
        self.stop()
        self.status.showMessage(f"Probing {os.path.basename(path)}…")
        self._set_enabled(False)
        self._probe_token += 1
        t = ProbeThread(path, self._probe_token)
        t.done.connect(self._probed)
        t.failed.connect(self._probe_failed)
        t.finished.connect(self._probe_thread_finished)
        self._probe_thread = t
        self._probe_threads.append(t)  # a QThread dropped while running crashes the app
        t.start()

    def _probe_thread_finished(self):
        self._probe_threads = [t for t in self._probe_threads if t.isRunning()]

    def _probe_failed(self, msg: str, token: int):
        if token != self._probe_token:
            return  # another file was opened meanwhile
        if self.info:
            self._set_enabled(True)  # the video that was open stays usable
        QMessageBox.critical(self, "Open failed", msg)

    def _probed(self, info: VideoInfo, token: int = 0):
        if token != self._probe_token:
            return  # a newer open (or close) wins over this slower probe
        if self.dec:
            self.dec.close()
            self.dec = None
        try:
            self.dec = Decoder(info.path)
        except Exception as e:  # noqa: BLE001
            self.close_video()  # the old decoder is closed: don't leave its UI pointing at it
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.info = info
        self.rotation = 0
        self.flip = False
        self._set_color(None, "")
        self._update_rot_label()
        self.view.crop = None
        if self.b_crop.isChecked():
            self.b_crop.setChecked(False)
        if self.b_fixed.isChecked():
            self.b_fixed.setChecked(False)
        self.clear_marks()
        self._audio_ok = False
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(info.path) if info.has_audio else QUrl())
        self.timeline.set_range(info.start_time, info.end_time)
        self.timeline.set_keyframes(info.keyframes)
        self.setWindowTitle(f"{os.path.basename(info.path)} — {theme.APP_NAME}")
        self.status.showMessage(
            f"{info.width}x{info.height}  {info.fps:.3f} fps  {info.codec}  "
            f"{len(info.keyframes)} keyframes  audio: {info.audio_codec or 'none'}")
        self._set_enabled(True)
        self._sync_length_boxes()
        self.dec.seek(info.start_time)
        self._show_current()
        self.view.setFocus()

    def close_video(self):
        """Unload the current video and return to the empty state."""
        self.stop()
        self._probe_token += 1  # a probe still running must not open its file afterwards
        if self.dec:
            self.dec.close()
            self.dec = None
        self.player.stop()
        self.player.setSource(QUrl())
        self._audio_ok = False
        self.info = None
        self.rotation = 0
        self.flip = False
        self._set_color(None, "")
        self.mark_in = self.mark_out = None
        self.fixed = False
        for b in (self.b_crop, self.b_fixed):
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)
        self.view.clear()
        self.timeline.set_range(0.0, 1.0)
        self.timeline.set_keyframes([])
        self.timeline.set_marks(None, None)
        self.timeline.set_selection(None)
        self.timeline.set_position(0.0)
        self.time_label.setText("--:--:--.--- / --:--:--.---")
        self.frame_label.setText("")
        self.kf_label.setText("")
        self.kf_label.setStyleSheet("")
        self.rot_label.setText("")
        self.sel_label.setText("")
        self.sel_panel.setVisible(False)
        self.crop_label.setText("")
        self.in_label.setText("In: \u2014")
        self.out_label.setText("Out: \u2014")
        self.b_export.setEnabled(False)
        self.b_export.setText("Export\u2026  [Ctrl+E]")
        self._set_enabled(False)
        self._crop_panel_enabled(False)
        self.setWindowTitle(theme.APP_NAME)
        self.status.clearMessage()

    # ---------------------------------------------------------------- settings
    def _refresh_outdir_action(self):
        d = output_dir()
        self.a_outdir.setText(f"Output folder: {d}…" if d else "Output folder: (next to source video)…")

    def choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Default folder for clips and images",
                                             output_dir() or "")
        if d:
            settings().setValue("output_dir", d)
            self._refresh_outdir_action()

    def reset_output_dir(self):
        settings().remove("output_dir")
        self._refresh_outdir_action()

    def _effective_output_dir(self) -> str:
        d = output_dir()
        if d:
            return d
        if self.info:
            return os.path.dirname(os.path.abspath(self.info.path))
        return os.path.expanduser("~")

    def open_output_dir(self):
        os.startfile(self._effective_output_dir())

    # ---------------------------------------------------------------- rotation
    @property
    def total_rotation(self) -> int:
        return ((self.info.rotation if self.info else 0) + self.rotation) % 360

    def rotate(self):
        """Turn what is on screen 90 deg clockwise.

        Screen = H^flip o R(rotation). Composing a clockwise turn on the left gives
        R90 o H = H o R270, so with the mirror on the stored rotation goes backwards
        while the picture still turns clockwise on screen.
        """
        if not self.dec:
            return
        _, old_h = self.view.frame_size
        self.rotation = (self.rotation + (270 if self.flip else 90)) % 360
        self.view.rotate_crop_cw(old_h)
        self._update_rot_label()
        self._show_current()
        self._sync_aspect_combo_from_view()
        self._crop_changed()

    def flip_h(self):
        """Mirror what is on screen left-to-right."""
        if not self.dec:
            return
        self.flip = not self.flip
        self.view.flip_crop_h()
        self._update_rot_label()
        self._show_current()
        self._crop_changed()

    def _update_rot_label(self):
        parts = []
        if self.rotation:
            parts.append(f"+{self.rotation}°")
        if self.flip:
            parts.append("↔ mirrored")
        if self.info and self.info.rotation:
            parts.append(f"(source {self.info.rotation}°)")
        self.rot_label.setText(" ".join(parts))

    def _frame_image(self, f, width: int | None = None, height: int | None = None) -> QImage:
        """Convert a decoded frame to a QImage, optionally scaled, with rotation applied."""
        kw = {"format": "rgb24"}
        if width and height:
            kw.update(width=width, height=height)
        rgb = f.reformat(**kw)
        plane = rgb.planes[0]
        # Wrap the plane's memory without copying it (no bytes(plane)); the one .copy() below
        # gives the QImage its own pixels before the frame is freed.
        buf = np.frombuffer(plane, dtype=np.uint8).reshape(rgb.height, plane.line_size)
        if self.color_tables is not None:
            arr = apply_tables(buf[:, :rgb.width * 3].reshape(rgb.height, rgb.width, 3), self.color_tables)
            img = QImage(arr.data, rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888).copy()
        else:
            img = QImage(buf.data, rgb.width, rgb.height, plane.line_size, QImage.Format_RGB888).copy()
        rot = self.total_rotation
        if rot:
            img = img.transformed(QTransform().rotate(rot))
        if self.flip:
            img = img.mirrored(True, False)
        return img

    def _rotated_frame_size(self) -> tuple[int, int]:
        w, h = self.info.width, self.info.height
        return (h, w) if self.total_rotation in (90, 270) else (w, h)

    # -------------------------------------------------------------------- crop
    def _crop_panel_enabled(self, on: bool):
        for w in (self.aspect_combo, self.aspect_edit, self.rb_native, self.rb_scale,
                  self.b_crop_reset):
            w.setEnabled(on)
        self.out_w.setEnabled(on and self.rb_scale.isChecked())
        self.out_h.setEnabled(on and self.rb_scale.isChecked())

    def _toggle_crop(self, on: bool):
        self.view.set_crop_enabled(on)
        self._crop_panel_enabled(on)
        if on and self.rb_scale.isChecked():
            self._seed_out_size()
        self._crop_changed()

    def _current_aspect(self) -> float | None:
        """Aspect the crop box is locked to: output WxH when scaling, else the preset."""
        if self.rb_scale.isChecked():
            return self.out_w.value() / self.out_h.value()
        key = self.aspect_combo.currentText()
        v = ASPECT_PRESETS.get(key)
        if v == "custom":
            try:
                return parse_aspect(self.aspect_edit.text())
            except ValueError:
                return None
        return v

    def _aspect_changed(self, key: str):
        self.aspect_edit.setVisible(ASPECT_PRESETS.get(key) == "custom")
        if ASPECT_PRESETS.get(key) == "custom" and not self.aspect_edit.text():
            self.aspect_edit.setFocus()
            return
        self._apply_aspect()

    def _custom_aspect_entered(self):
        try:
            parse_aspect(self.aspect_edit.text())
        except ValueError:
            self.status.showMessage("Aspect must look like 16:9 or 1.78", 4000)
            return
        self._apply_aspect()

    def _apply_aspect(self):
        a = self._current_aspect()
        if self.rb_scale.isChecked():
            preset = self._preset_aspect()
            if preset:
                # a new preset with scaling on: keep width, derive height
                self._block_out(True)
                self.out_h.setValue(even(self.out_w.value() / preset))
                self._block_out(False)
                a = self.out_w.value() / self.out_h.value()
        self.view.set_aspect(a)

    def _preset_aspect(self) -> float | None:
        v = ASPECT_PRESETS.get(self.aspect_combo.currentText())
        if v == "custom":
            try:
                return parse_aspect(self.aspect_edit.text())
            except ValueError:
                return None
        return v

    def _sync_aspect_combo_from_view(self):
        """After a rotation the locked aspect is inverted; reflect that in the combo."""
        a = self.view.aspect
        if a is None:
            return
        self.aspect_combo.blockSignals(True)
        for key, v in ASPECT_PRESETS.items():
            if isinstance(v, float) and abs(v - a) < 1e-3:
                self.aspect_combo.setCurrentText(key)
                break
        else:
            self.aspect_combo.setCurrentText("Custom…")
            self.aspect_edit.setVisible(True)
            self.aspect_edit.setText(f"{a:.4g}:1" if a >= 1 else f"1:{1 / a:.4g}")
        self.aspect_combo.blockSignals(False)
        if self.rb_scale.isChecked():
            self._block_out(True)
            w, h = self.out_w.value(), self.out_h.value()
            self.out_w.setValue(h)
            self.out_h.setValue(w)
            self._block_out(False)

    def _block_out(self, on: bool):
        self.out_w.blockSignals(on)
        self.out_h.blockSignals(on)

    def _seed_out_size(self):
        r = self.view.crop_rect_int()
        if r:
            self._block_out(True)
            self.out_w.setValue(r[2])
            self.out_h.setValue(r[3])
            self._block_out(False)

    def _scale_toggled(self, on: bool):
        self.out_w.setEnabled(on)
        self.out_h.setEnabled(on)
        if on:
            self._seed_out_size()
        self._apply_aspect()

    def _out_size_edited(self, which: str):
        preset = self._preset_aspect()
        if preset:
            self._block_out(True)
            if which == "w":
                self.out_h.setValue(even(self.out_w.value() / preset))
            else:
                self.out_w.setValue(even(self.out_h.value() * preset))
            self._block_out(False)
        self.view.set_aspect(self.out_w.value() / self.out_h.value())

    def _crop_changed(self):
        r = self.view.crop_rect_int()
        if not r:
            self.crop_label.setText("")
        else:
            txt = f"{r[2]}×{r[3]} at ({r[0]}, {r[1]})"
            if self.rb_scale.isChecked():
                txt += f"  →  {self.out_w.value()}×{self.out_h.value()}"
            self.crop_label.setText(txt)
        self._selection_changed()

    def _export_geometry(self):
        crop = self.view.crop_rect_int()
        out_size = None
        if crop and self.rb_scale.isChecked():
            out_size = (even(self.out_w.value()), even(self.out_h.value()))
        return crop, out_size

    # ------------------------------------------------------------ fixed length
    def _fixed_duration(self) -> float:
        if self.fixed_unit == "frames":
            return self.fixed_frames * self.info.frame_duration
        return self.fixed_seconds

    def _fixed_nframes(self) -> int:
        if self.fixed_unit == "frames":
            return self.fixed_frames
        return max(1, math.ceil(self.fixed_seconds * self.info.fps - 1e-6))

    @property
    def sel_end(self) -> float:
        return self.sel_start + self._fixed_duration()

    @property
    def sel_last_frame(self) -> float:
        return self.sel_start + (self._fixed_nframes() - 1) * self.info.frame_duration

    def _snap(self, t: float) -> float:
        fd = self.info.frame_duration
        return self.info.start_time + round((t - self.info.start_time) / fd) * fd

    def _sync_length_boxes(self):
        """Keep seconds and frames displays consistent with the authoritative value."""
        if not self.info:
            return
        self.len_sec.blockSignals(True)
        self.len_frames.blockSignals(True)
        if self.fixed_unit == "frames":
            self.len_frames.setValue(self.fixed_frames)
            self.len_sec.setValue(self.fixed_frames * self.info.frame_duration)
        else:
            self.len_sec.setValue(self.fixed_seconds)
            self.len_frames.setValue(self._fixed_nframes())
        self.len_sec.blockSignals(False)
        self.len_frames.blockSignals(False)

    def _len_seconds_edited(self, v: float):
        self.fixed_unit = "seconds"
        self.fixed_seconds = v
        self._sync_length_boxes()
        self._fixed_length_changed()

    def _len_frames_edited(self, v: int):
        self.fixed_unit = "frames"
        self.fixed_frames = v
        self._sync_length_boxes()
        self._fixed_length_changed()

    def _fixed_length_changed(self):
        if self.fixed and self.info:
            self._set_sel_start(self.sel_start, seek=False)

    def _toggle_fixed(self, on: bool):
        self.fixed = on
        if not self.info:
            return
        self.sel_panel.setVisible(on)
        if on:
            self.stop()
            self.fixed_unit = "seconds" if self.len_sec.hasFocus() else self.fixed_unit
            self._set_sel_start(self.dec.time if self.dec else self.info.start_time, seek=True)
        else:
            self.timeline.set_selection(None)
            self.sel_label.setText("")
            self._selection_changed()

    # --- moving the selection box (fixed-length mode only)
    def sel_move_frames(self, n: int):
        if self.fixed and self.info:
            self._set_sel_start(self.sel_start + n * self.info.frame_duration, seek=True)

    def sel_move_keyframe(self, direction: int):
        if not (self.fixed and self.info):
            return
        info = self.info
        k = info.next_keyframe(self.sel_start) if direction > 0 else info.prev_keyframe(self.sel_start)
        if k is None:
            k = info.end_time if direction > 0 else info.start_time
        self._set_sel_start(k, seek=True)

    def sel_start_here(self):
        """Start the clip at the current frame (same as Mark In in fixed-length mode)."""
        if not (self.fixed and self.dec):
            return
        self._set_sel_start(self.dec.time, seek=False)
        # If the clip had to be clamped at the end of the video the playhead may now
        # sit past the clip; bring it back inside.
        if self.dec.time > self.sel_last_frame + self.info.frame_duration * 0.25:
            self.seek_to(self.sel_start)

    def playhead_to_sel_start(self):
        if self.fixed and self.dec:
            self.seek_to(self.sel_start)

    def sel_next_clip(self):
        """Advance the clip to begin on the first frame the current clip does not include."""
        if not (self.fixed and self.info):
            return
        info = self.info
        nxt = self.sel_start + self._fixed_nframes() * info.frame_duration
        last_possible = info.end_time - self._fixed_duration()
        if nxt > last_possible + info.frame_duration * 0.25:
            self.status.showMessage("No room for another full clip; moved to the end of the video.", 5000)
        self._set_sel_start(nxt, seek=True)

    def sel_to_start(self):
        if self.fixed and self.info:
            self._set_sel_start(self.info.start_time, seek=True)

    def sel_to_end(self):
        if self.fixed and self.info:
            self._set_sel_start(self.info.end_time, seek=True)

    def _set_sel_start(self, t: float, seek: bool):
        info = self.info
        dur = self._fixed_duration()
        if dur > info.duration:
            # clip longer than the video: shrink to fit
            if self.fixed_unit == "frames":
                self.fixed_frames = max(1, int(info.duration * info.fps))
            else:
                self.fixed_seconds = round(info.duration, 2)
            self._sync_length_boxes()
            dur = self._fixed_duration()
        t = self._snap(t)
        t = max(info.start_time, min(t, self._snap(info.end_time - dur)))
        self.sel_start = t
        self.timeline.set_selection((self.sel_start, self.sel_end))
        self.sel_label.setText(
            f"{fmt_time(self.sel_start - info.start_time)} → {fmt_time(self.sel_end - info.start_time)}")
        self._selection_changed()
        if seek and self.dec:
            self.stop()
            self.dec.seek(self.sel_start)
            self._show_current()

    def _move_selection(self, start: float):
        if self.fixed and self.info:
            self._set_sel_start(start, seek=True)

    def _clamp_to_selection(self, t: float) -> float:
        return max(self.sel_start, min(t, self.sel_last_frame))

    def _at_selection_end(self) -> bool:
        fd = self.info.frame_duration
        return self.dec.time + fd >= self.sel_end - fd * 0.25

    # ------------------------------------------------------------- display
    def _show_current(self):
        if not self.dec or self.dec.current is None:
            return
        f = self.dec.current
        # Scale in the decoder (swscale) so we never push huge RGB buffers to Qt.
        vw, vh = max(self.view.width(), 64), max(self.view.height(), 64)
        if self.total_rotation in (90, 270):
            vw, vh = vh, vw
        scale = min(vw / f.width, vh / f.height, 1.0)
        w, h = max(int(f.width * scale) & ~1, 2), max(int(f.height * scale) & ~1, 2)
        fw, fh = self._rotated_frame_size()
        self.view.set_frame(self._frame_image(f, w, h), fw, fh)

        t = f.time or 0.0
        info = self.info
        self.timeline.set_position(t)
        self.time_label.setText(f"{fmt_time(t - info.start_time)} / {fmt_time(info.duration)}")
        self.frame_label.setText(f"  frame {round((t - info.start_time) * info.fps)}")
        on_kf = f.key_frame or info.is_keyframe_time(t)
        self.kf_label.setText("  ● keyframe" if on_kf else "")
        self.kf_label.setStyleSheet(f"color:{theme.ACCENT_TEXT};font-weight:bold" if on_kf else "")

    # ------------------------------------------------------------ playback
    def toggle_play(self):
        if not self.dec:
            return
        if self.play_timer.isActive():
            self.stop()
            return
        if self.fixed:
            eps = self.info.frame_duration * 0.25
            if self.dec.time < self.sel_start - eps or self._at_selection_end():
                self.dec.seek(self.sel_start)
                self._show_current()
        self._start_clock(self.dec.time)
        # tick at twice the frame rate so frames land close to the audio clock
        self.play_timer.start(max(int(500 / self.info.fps), 4))
        self.b_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def stop(self):
        self.play_timer.stop()
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        self.b_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    # --- audio / clock
    def _audio_status(self, st):
        if st in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia, QMediaPlayer.BufferingMedia):
            self._audio_ok = True
        elif st in (QMediaPlayer.InvalidMedia, QMediaPlayer.NoMedia):
            self._audio_ok = False

    def _audio_failed(self, msg: str):
        self._audio_ok = False
        self.status.showMessage(f"Audio unavailable: {msg}", 6000)

    def _set_muted(self, on: bool):
        self.audio_out.setMuted(on)
        settings().setValue("muted", on)
        self._update_mute_icon()

    def _update_mute_icon(self):
        icon = QStyle.SP_MediaVolumeMuted if self.b_mute.isChecked() else QStyle.SP_MediaVolume
        self.b_mute.setIcon(self.style().standardIcon(icon))

    def _set_volume(self, v: int):
        self.audio_out.setVolume(v / 100)
        settings().setValue("volume", v)

    def _start_clock(self, t: float):
        """Begin playback timing at absolute pts t."""
        self._wall_origin = t
        self._wall.start()
        if self._audio_ok:
            self.player.setPosition(int(max(t - self.info.fmt_start_time, 0.0) * 1000))
            self.player.play()

    def _clock(self) -> float:
        """Current playback time in absolute pts seconds (audio clock when available)."""
        if self._audio_ok and self.player.playbackState() == QMediaPlayer.PlayingState:
            pos = self.player.position() / 1000 + self.info.fmt_start_time
            # Until the player's seek lands, its position may still be stale: hold.
            if pos >= self._wall_origin - 0.05:
                return pos
            return self._wall_origin
        return self._wall_origin + self._wall.elapsed() / 1000

    def _play_tick(self):
        info = self.info
        fd = info.frame_duration
        now = self._clock()
        target = min(now, self.sel_last_frame) if self.fixed else now
        moved = False
        # Far behind (slow decode or a stall): jump instead of stepping through everything.
        if target - self.dec.time > 1.0:
            self.dec.seek(target)
            moved = True
        steps = 0
        while self.dec.time + fd <= target + fd * 0.25 and steps < 12:
            if self.dec.step_forward() is None:
                self.stop()
                break
            steps += 1
            moved = True
        if moved:
            self._show_current()
        if self.fixed and now >= self.sel_end - fd * 0.25:
            self.stop()

    def seek_to(self, t: float):
        if not self.dec:
            return
        self.stop()
        if self.fixed:
            t = self._clamp_to_selection(t)
        self.dec.seek(t)
        self._show_current()

    def step_forward(self):
        if not self.dec:
            return
        self.stop()
        if self.fixed:
            if self.dec.time < self.sel_start - self.info.frame_duration * 0.25:
                self.seek_to(self.sel_start)
                return
            if self._at_selection_end():
                return
        if self.dec.step_forward() is not None:
            self._show_current()

    def step_back(self):
        if not self.dec:
            return
        self.stop()
        if self.fixed:
            if self.dec.time <= self.sel_start + self.info.frame_duration * 0.25:
                return
            if self.dec.time > self.sel_last_frame + self.info.frame_duration * 0.25:
                self.seek_to(self.sel_last_frame)
                return
        self.dec.step_back()
        self._show_current()

    def next_keyframe(self):
        if not self.dec:
            return
        k = self.info.next_keyframe(self.dec.time)
        if self.fixed:
            eps = self.info.frame_duration * 0.25
            if k is None or k > self.sel_last_frame + eps:
                self.seek_to(self.sel_last_frame)
                return
        if k is not None:
            self.seek_to(k)

    def prev_keyframe(self):
        if not self.dec:
            return
        k = self.info.prev_keyframe(self.dec.time)
        if self.fixed:
            eps = self.info.frame_duration * 0.25
            if k is None or k < self.sel_start - eps:
                self.seek_to(self.sel_start)
                return
        self.seek_to(k if k is not None else self.info.start_time)

    def go_start(self):
        if self.dec:
            self.seek_to(self.sel_start if self.fixed else self.info.start_time)

    def go_end(self):
        if not self.dec:
            return
        self.stop()
        if self.fixed:
            self.seek_to(self.sel_last_frame)
            return
        self.dec.seek_last(self.info.end_time)
        self._show_current()

    # --------------------------------------------------------------- marks
    def _is_last_frame(self, t: float) -> bool:
        return t >= self.info.end_time - self.info.frame_duration * 1.5

    def set_in(self):
        if not self.dec:
            return
        if self.fixed:
            self._set_sel_start(self.dec.time, seek=False)
            return
        self.mark_in = self.dec.time
        if self.mark_out is not None and self.mark_out <= self.mark_in:
            self.mark_out = None
        self._selection_changed()

    def set_out(self):
        if not self.dec:
            return
        t = self.dec.time
        if self.fixed:
            # Out is exclusive: the selection ends just before the current frame.
            end = self.info.end_time if self._is_last_frame(t) else t
            self._set_sel_start(end - self._fixed_duration(), seek=False)
            return
        # Marking Out on the very last frame means "through the end of the video".
        self.mark_out = self.info.end_time if self._is_last_frame(t) else t
        if self.mark_in is not None and self.mark_in >= self.mark_out:
            self.mark_in = None
        self._selection_changed()

    def clear_marks(self):
        self.mark_in = self.mark_out = None
        self._selection_changed()

    def _export_range(self) -> tuple[float, float] | None:
        if self.fixed and self.info:
            return self.sel_start, self.sel_end
        if self.mark_in is not None and self.mark_out is not None and self.mark_out > self.mark_in:
            return self.mark_in, self.mark_out
        return None

    def _on_keyframe_or_edge(self, t: float) -> bool:
        info = self.info
        return (info.is_keyframe_time(t) or t <= info.start_time + 1e-4
                or t >= info.end_time - 1e-4)

    def _lossless_possible(self) -> bool:
        rng = self._export_range()
        if not rng or not self.info or self.view.crop_rect_int() or self.flip or self.color_tables is not None:
            return False
        return self._on_keyframe_or_edge(rng[0]) and self._on_keyframe_or_edge(rng[1])

    def _selection_changed(self):
        self.timeline.set_marks(self.mark_in, self.mark_out)
        info = self.info

        def lab(name, t):
            if t is None or not info:
                return f"{name}: —"
            ok = self._on_keyframe_or_edge(t)
            return f"{name}: {fmt_time(t - info.start_time)} {'● K' if ok else '(not keyframe)'}"

        self.in_label.setText(lab("In", self.mark_in))
        self.out_label.setText(lab("Out", self.mark_out))
        ready = self._export_range() is not None
        self.b_export.setEnabled(ready)
        if ready:
            self.b_export.setText("Export (lossless)…  [Ctrl+E]" if self._lossless_possible()
                                  else "Export (NVENC)…  [Ctrl+E]")
        else:
            self.b_export.setText("Export…  [Ctrl+E]")

    # ---------------------------------------------------------------- snapshot
    def save_frame(self):
        """Write the current frame straight to the output folder, honoring rotation and crop."""
        if not self.dec or self.dec.current is None:
            return
        f = self.dec.current
        img = self._frame_image(f)
        crop, out_size = self._export_geometry()
        if crop:
            img = img.copy(QRect(*crop))
            if out_size:
                img = img.scaled(out_size[0], out_size[1], Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        stem = os.path.splitext(os.path.basename(self.info.path))[0]
        t = (f.time or 0.0) - self.info.start_time
        frame_no = round(t * self.info.fps)
        name = f"{stem}_{fmt_time(t).replace(':', '-')}_f{frame_no}{'_crop' if crop else ''}.png"
        path = unique_path(os.path.join(self._effective_output_dir(), name))
        if img.save(path, "PNG"):
            self.status.showMessage(f"Saved frame: {path}", 6000)
        else:
            QMessageBox.critical(self, "Save frame", f"Could not write:\n{path}")

    # -------------------------------------------------------------- export
    def quick_export(self):
        """Export the selection (or the whole video) with the default settings, no dialog."""
        if not self.info or self.quick_job is not None:
            return
        if not have_ffmpeg():
            QMessageBox.critical(self, "Quick export", NO_FFMPEG)
            return
        self.stop()
        rng = self._export_range()
        whole = rng is None
        if whole:
            rng = (self.info.start_time, self.info.end_time)
        crop, out_size = self._export_geometry()
        s = load_quick_defaults()
        lossless = (quick_lossless_when_possible() and crop is None and out_size is None
                    and not self.flip and self.color_lut_path is None
                    and self._on_keyframe_or_edge(rng[0]) and self._on_keyframe_or_edge(rng[1]))
        mode = "lossless" if lossless else "nvenc"
        s.mode, s.rotation, s.hflip, s.crop, s.out_size = mode, self.rotation, self.flip, crop, out_size
        s.lut_path = self.color_lut_path
        out = unique_path(default_output_path(self.info, rng[0], rng[1], mode, output_dir(),
                                              cropped=crop is not None, whole=whole))
        self._quick_out = out
        self._quick_cancelled = False
        self.quick_job = ExportJob(build_commands(self.info, rng[0], rng[1], out, s), rng[1] - rng[0], self)
        self.quick_job.progress.connect(lambda f: self.qx_progress.setValue(int(f * 1000)))
        self.quick_job.finished.connect(self._quick_done)
        self.qx_label.setText(f"Exporting {'whole video' if whole else 'clip'} \u2192 {os.path.basename(out)}")
        self.qx_progress.setValue(0)
        for wdg in (self.qx_label, self.qx_progress, self.qx_cancel):
            wdg.setVisible(True)
        self.b_quick.setEnabled(False)
        self.quick_job.start()

    def _quick_cancel(self):
        if self.quick_job is not None:
            self._quick_cancelled = True
            self.quick_job.cancel()

    def _quick_done(self, ok: bool, log: str):
        for wdg in (self.qx_label, self.qx_progress, self.qx_cancel):
            wdg.setVisible(False)
        cmd_text = self.quick_job.cmd_text if self.quick_job else ""
        self.quick_job = None
        self.b_quick.setEnabled(self.info is not None)
        if self._quick_cancelled:
            remove_partial(self._quick_out)
            self.status.showMessage("Export cancelled.", 5000)
        elif ok:
            self.status.showMessage(f"Saved: {self._quick_out}", 10000)
            self._note_export(self._quick_out)
        else:
            QMessageBox.critical(self, "Quick export failed",
                                 f"ffmpeg failed.\n\nCommand:\n{cmd_text}\n\nLog:\n{log}")

    def export_clip(self):
        rng = self._export_range()
        if not self.info or rng is None:
            QMessageBox.information(self, "Export", "Set both In and Out marks first, or enable Fixed length.")
            return
        self.stop()
        crop, out_size = self._export_geometry()
        dlg = ExportDialog(self.info, rng[0], rng[1], self._lossless_possible(),
                           self.rotation, crop, out_size, self.flip, self.color_lut_path, self)
        dlg.exported.connect(self._note_export)
        dlg.exec()

    def closeEvent(self, e):
        self.stop()
        if self.quick_job is not None:  # don't leave ffmpeg running and a half-written clip behind
            self._quick_cancel()
            self.quick_job.wait()
        self._probe_token += 1
        for t in list(self._probe_threads):
            t.wait()  # a QThread destroyed while running takes the process down with it
        if self.dec:
            self.dec.close()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    theme.apply(app)
    # first run: "Choose your look" (Dark/Light); afterwards the saved choice
    theme.apply_mode(app, ensure_theme_chosen(app))
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        w.open_file(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
