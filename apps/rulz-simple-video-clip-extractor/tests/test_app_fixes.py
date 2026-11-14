"""Main-window behaviour fixed in the 2026-11 audit: fixed-length rounding, lone In/Out marks,
keyframe stepping before the scan, rotation and lossless cuts, opening and re-opening files,
the playback clock, background frame saves, the export dialog and stored settings.

Run with:  python -m pytest tests   (from the app folder, inside its .venv).
"""
import os
import subprocess
import time

import pytest

from test_core import APP, make_video  # noqa: F401  (sets up Qt, the settings file and sys.path)

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from simplecut.defaults import settings  # noqa: E402
from simplecut.export import (ExportSettings, build_command, keeps_rotation_flag,  # noqa: E402
                              lossless_can_rotate, lossless_end_ok, lossless_start_ok)
from simplecut.probe import VideoInfo, probe_streams  # noqa: E402


def _wait(cond, timeout=30):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        APP.processEvents()
        time.sleep(0.01)
    return cond()


@pytest.fixture
def window():
    from simplecut.app import MainWindow
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()
    APP.processEvents()


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """10 s at 30 fps (300 frames), a keyframe every 30."""
    return make_video(str(tmp_path_factory.mktemp("app") / "c.mp4"), n=300, fps=30, gop=30)


def _open(w, path, keyframes=True):
    before = w._video_token
    w.open_file(path)
    assert _wait(lambda: w._video_token != before and w.info is not None and w.dec is not None
                 and w.dec.current is not None)
    if keyframes:
        assert _wait(lambda: bool(w.info.keyframes))


# ------------------------------------------------------------------ fixed length
def test_fixed_length_out_excludes_the_current_frame_and_the_end_box_is_full(window, clip):
    w = window
    _open(w, clip)
    fd = w.info.frame_duration
    w.len_sec.setValue(2.01)          # 61 frames at 30 fps: not a whole number of frames
    assert w.fixed_unit == "seconds" and w._fixed_nframes() == 61
    w.seek_to(w.info.start_time + 180 * fd)
    w.b_fixed.setChecked(True)       # the box starts at frame 180, the playhead is on it
    w.set_out()
    first = round((w.sel_start - w.info.start_time) / fd)
    last = round((w.sel_last_frame - w.info.start_time) / fd)
    assert (first, last) == (119, 179)            # was 120..180: the current frame got in
    w.sel_to_end()
    first = round((w.sel_start - w.info.start_time) / fd)
    assert first == 300 - 61                      # was 240: one frame short at the end
    a, b = w._export_range()
    assert b - a == pytest.approx(2.01)           # the typed length, exactly
    assert lossless_end_ok(w.info, b)             # it ends with the video: nothing to cut off
    w.info.keyframes.append(w.info.start_time + 299 * fd)   # even with a keyframe on the last frame,
    cmd = build_command(w.info, a, b, "o.mkv", ExportSettings(mode="lossless"))
    assert float(cmd[cmd.index("-t") + 1]) == pytest.approx(2.01, abs=1e-5)  # it isn't cut before it


def test_a_short_video_shrinks_the_length_for_itself_only(window, clip, tmp_path):
    w = window
    w.len_sec.setValue(5.0)
    short = make_video(str(tmp_path / "short.mp4"), n=60, fps=30, gop=30)   # 2 s
    _open(w, short, keyframes=False)
    w.b_fixed.setChecked(True)
    assert w.fixed_seconds == pytest.approx(2.0)
    _open(w, clip, keyframes=False)
    assert w.fixed_seconds == 5.0                 # the next video gets the chosen length back


# ------------------------------------------------------------------ marks
def test_a_lone_in_or_out_mark_exports_what_the_timeline_shades(window, clip):
    w = window
    _open(w, clip, keyframes=False)
    t0, t1 = w.info.start_time, w.info.end_time
    assert w._export_range() is None
    w.seek_to(t0 + 3.0)
    w.set_in()
    assert w._export_range() == (pytest.approx(t0 + 3.0), t1) and w.b_export.isEnabled()
    w.clear_marks()
    w.seek_to(t0 + 4.0)
    w.set_out()
    assert w._export_range() == (t0, pytest.approx(t0 + 4.0))


def test_keyframe_stepping_waits_for_the_scan(window, clip):
    w = window
    _open(w, clip, keyframes=False)
    kfs = list(w.info.keyframes)
    w.info.keyframes = []                         # as while the scan is still running
    w.seek_to(w.info.start_time + 5.0)
    before = w.dec.time
    w.prev_keyframe()
    w.next_keyframe()
    assert w.dec.time == before                   # it used to jump to the start
    w._set_keyframe_buttons()
    assert not w.b_prev_kf.isEnabled()
    w.info.keyframes = kfs or [w.info.start_time]
    w._set_keyframe_buttons()
    assert w.b_prev_kf.isEnabled()


def test_a_failed_open_keeps_the_open_videos_keyframes_coming(window, clip, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)
    w = window
    _open(w, clip, keyframes=False)
    token = w._video_token
    bad = tmp_path / "not-a-video.mp4"
    bad.write_bytes(b"garbage" * 100)
    w.open_file(str(bad))
    assert _wait(lambda: not w._probe_thread.isRunning())
    APP.processEvents()
    assert w.info is not None and w.info.path == clip and w.b_play.isEnabled()
    w._keyframes_found([0.0, 1.0, 2.0], {}, token)   # the open video's scan, finishing late
    assert w.info.keyframes == [0.0, 1.0, 2.0]


# ------------------------------------------------------------------ rotation and lossless
def test_a_rotated_lossless_cut_goes_to_a_container_that_keeps_the_rotation(tmp_path, monkeypatch):
    """ffmpeg drops -display_rotation silently for MPEG-TS, AVI and FLV: a rotated lossless cut
    of those is saved as MKV (and a typed .ts output name is refused)."""
    from PySide6.QtWidgets import QMessageBox
    from simplecut.app import ExportDialog
    from simplecut.export import default_output_path
    assert keeps_rotation_flag("mov,mp4,m4a,3gp,3g2,mj2") and keeps_rotation_flag("matroska,webm")
    assert not keeps_rotation_flag("mpegts") and not keeps_rotation_flag("avi")
    assert not keeps_rotation_flag("flv")
    assert keeps_rotation_flag("mpegts", "C:/out/x.mkv") and not keeps_rotation_flag("mov", "C:/out/x.ts")
    ts = VideoInfo(path=str(tmp_path / "cam.mts"), container="mpegts", fps=25.0, duration=10.0)
    assert default_output_path(ts, 0.0, 2.0, "lossless").endswith(".mts")
    assert default_output_path(ts, 0.0, 2.0, "lossless", rotation=90).endswith(".mkv")
    mp4 = VideoInfo(path=str(tmp_path / "phone.mp4"), container="mov,mp4,m4a,3gp,3g2,mj2", fps=25.0, duration=10.0)
    assert default_output_path(mp4, 0.0, 2.0, "lossless", rotation=90).endswith(".mp4")
    dlg = ExportDialog(ts, 0.0, 2.0, lossless_ok=True, rotation=90)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    try:
        dlg.rb_lossless.setChecked(True)
        assert dlg.rb_lossless.isEnabled() and dlg.out_path.text().endswith(".mkv")
        dlg.out_path.setText(str(tmp_path / "x.ts"))
        dlg._export()
        assert warned and "rotation flag" in warned[0] and dlg.job is None
    finally:
        dlg.deleteLater()


def test_lossless_rotation_flag_survives_in_the_chosen_container(tmp_path):
    """The real thing: a rotated MPEG-TS source cut losslessly to MKV keeps the rotation."""
    from simplecut.export import ExportSettings, build_command
    from simplecut.probe import probe
    src = str(tmp_path / "cam.ts")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=25:duration=2",
                        "-c:v", "libx264", "-g", "25", src], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    info = probe(src)
    out = str(tmp_path / "cut.mkv")
    cmd = build_command(info, info.start_time, info.end_time, out, ExportSettings(mode="lossless", rotation=90))
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    assert probe_streams(out).rotation == 90


def test_a_b_frame_avi_is_rotated_by_re_encoding(window, clip):
    """AVI stores no pts, and with B-frames ffmpeg can't write its packets to MKV ("unknown
    timestamp"), so a rotated cut of such a file isn't offered losslessly; the dialog says why."""
    from simplecut.app import ExportDialog
    assert not lossless_can_rotate(VideoInfo(path="a.avi", container="avi", has_b_frames=1))
    assert lossless_can_rotate(VideoInfo(path="a.avi", container="avi", has_b_frames=0))
    assert lossless_can_rotate(VideoInfo(path="a.mp4", container="mov,mp4,m4a,3gp,3g2,mj2", has_b_frames=2))
    w = window
    _open(w, clip, keyframes=False)
    rng = (w.info.start_time, w.info.end_time)          # the whole video: both ends are edges
    w.info.container, w.info.has_b_frames = "avi", 1
    assert w._cuts_losslessly(rng)                      # unrotated: copied as it is
    w.rotation = 90
    try:
        assert not w._cuts_losslessly(rng)
        dlg = ExportDialog(w.info, *rng, lossless_ok=False, rotation=90)
        try:
            assert not dlg.rb_lossless.isEnabled() and "B-frames" in dlg.rb_lossless.text()
        finally:
            dlg.deleteLater()
    finally:
        w.rotation = 0


def test_an_end_just_after_a_keyframe_is_not_cut_in_front_of_it(tmp_path):
    """A 2.01 s box at 30 fps ending 0.01 s after a keyframe holds that keyframe's frame as its
    last; a stream copy cut in front of the keyframe would lose it, so that end isn't lossless."""
    info = VideoInfo(path=str(tmp_path / "v.mp4"), container="mov,mp4,m4a,3gp,3g2,mj2", fps=30.0,
                     duration=10.0, keyframes=[0.0, 1.0, 2.0, 3.0, 4.0])
    assert not lossless_end_ok(info, 3.01)
    assert lossless_end_ok(info, 3.0) and lossless_end_ok(info, 3.0005) and lossless_end_ok(info, 2.99)
    out = str(tmp_path / "o.mp4")
    cmd = build_command(info, 1.0, 3.01, out, ExportSettings(mode="lossless"))
    assert float(cmd[cmd.index("-t") + 1]) == pytest.approx(2.01, abs=1e-5)   # not cut at 3.0
    cmd = build_command(info, 1.0, 3.0005, out, ExportSettings(mode="lossless"))
    assert float(cmd[cmd.index("-t") + 1]) == pytest.approx(2.0, abs=1e-5)    # rounding: cut at 3.0


def test_only_an_end_counts_the_last_frame_as_the_edge():
    """With a variable frame rate the last frame can start within an average frame of the end:
    an In there is still not on a keyframe (a stream copy would start seconds earlier)."""
    info = VideoInfo(path="v.mp4", fps=29.4, duration=10.0, keyframes=[0.0, 5.0])
    assert not lossless_start_ok(info, 9.98) and lossless_end_ok(info, 9.98)
    assert lossless_start_ok(info, 0.0) and lossless_start_ok(info, 5.0) and not lossless_start_ok(info, 5.1)


# ------------------------------------------------------------------ opening files
def test_mkv_end_is_the_end_of_its_video_track(tmp_path):
    """3 s of video, 4 s of audio: Matroska has no stream duration, and the container's runs to
    the end of the audio."""
    path = str(tmp_path / "v3a4.mkv")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=30:duration=3",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:v", "libx264", "-c:a", "aac", path],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    info = probe_streams(path)
    assert info.duration == pytest.approx(3.0, abs=0.04)
    # the tag is where the track ends: a video starting at 2.0 s still lasts 3 s
    late = str(tmp_path / "late.mkv")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-c", "copy", "-output_ts_offset", "2", late],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    info = probe_streams(late)
    assert info.start_time == pytest.approx(2.0, abs=0.04) and info.duration == pytest.approx(3.0, abs=0.04)
    assert info.end_time == pytest.approx(5.0, abs=0.04)


def test_mkvmerge_duration_tags_are_lengths(monkeypatch):
    """mkvmerge's statistics DURATION tag (written with _STATISTICS_WRITING_APP) is how long the
    track lasts, not where it ends: a video starting at 0.3 s keeps its last 0.3 s."""
    import json
    import simplecut.probe as probe_mod

    def ffprobe_json(tags):
        return json.dumps({
            "format": {"format_name": "matroska,webm", "start_time": "0.000000", "duration": "5.400000"},
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 160, "height": 96,
                         "avg_frame_rate": "30/1", "start_time": "0.300000", "tags": tags}]})

    mkvmerge = {"BPS-eng": "1000", "DURATION-eng": "00:00:05.000000000",
                "_STATISTICS_WRITING_APP-eng": "mkvmerge v90.0 64-bit", "_STATISTICS_TAGS-eng": "BPS DURATION"}
    monkeypatch.setattr(probe_mod, "_run", lambda args, on_start=None: ffprobe_json(mkvmerge))
    info = probe_streams("x.mkv")
    assert info.duration == pytest.approx(5.0) and info.end_time == pytest.approx(5.3)
    ffmpeg = {"DURATION": "00:00:05.000000000", "ENCODER": "Lavc libx264"}
    monkeypatch.setattr(probe_mod, "_run", lambda args, on_start=None: ffprobe_json(ffmpeg))
    info = probe_streams("x.mkv")
    assert info.duration == pytest.approx(4.7) and info.end_time == pytest.approx(5.0)


def test_a_raw_stream_that_cannot_seek_is_refused_cleanly(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append(a[2]))
    raw = str(tmp_path / "raw.h264")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=25:duration=2",
                        "-c:v", "libx264", "-f", "h264", raw], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    w = window
    w.open_file(raw)
    assert _wait(lambda: shown or w.info is not None)
    if w.info is not None:
        pytest.skip("this PyAV build can seek in raw H.264")
    assert w.dec is None and not w.b_play.isEnabled() and "raw video stream" in shown[0]
    # with a video open, it stays open and usable (it used to be closed before the new one failed)
    good = make_video(str(tmp_path / "good.mp4"), n=50, fps=25, gop=25)
    _open(w, good, keyframes=False)
    dec = w.dec
    w.open_file(raw)
    assert _wait(lambda: len(shown) == 2)
    assert w.info.path == good and w.dec is dec and w.dec.current is not None and w.b_play.isEnabled()
    w.step_forward()   # the old decoder still works


def test_reopening_the_same_file_reloads_its_sound(window, tmp_path):
    """Setting the URL that is loaded already reports no new media status, so a second open of
    the same file left the audio marked unavailable and played it silently."""
    path = str(tmp_path / "sound.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=25:duration=2",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:v", "libx264", "-c:a", "aac", path],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    w = window
    _open(w, path, keyframes=False)
    if not _wait(lambda: w._audio_ok, timeout=10):
        pytest.skip("Qt Multimedia can't load audio here")
    _open(w, path, keyframes=False)
    assert _wait(lambda: w._audio_ok, timeout=10)


# ------------------------------------------------------------------ playback clock
class FakePlayer:
    def __init__(self):
        self.ms = 0

    def position(self):
        return self.ms

    def playbackState(self):
        from PySide6.QtMultimedia import QMediaPlayer
        return QMediaPlayer.PlayingState


def test_the_audio_clock_runs_smoothly_between_position_updates(window):
    w = window
    real = w.player
    w.info = VideoInfo(path="x.mp4", fps=30.0, duration=10.0)
    w._audio_ok = True
    w.player = FakePlayer()
    try:
        w._wall_origin = 0.0
        w._wall.start()
        w._audio_step = None
        seen = []
        t_end = time.time() + 0.2
        while time.time() < t_end:   # the player's position stays on one ~50 ms step meanwhile
            seen.append(w._clock())
            time.sleep(0.005)
        assert seen[-1] - seen[0] >= 0.1            # it ran on (it used to stay at the step)
        assert seen[-1] <= w.AUDIO_STEP_MAX_MS / 1000 + 1e-6   # but no further than the cap
        w.player.ms = 1000                            # the next step resets the run-on
        assert w._clock() == pytest.approx(1.0, abs=0.01)
    finally:
        w.player, w.info, w._audio_ok = real, None, False


# ------------------------------------------------------------------ frame PNGs
def test_frame_pngs_save_in_the_background_with_distinct_names(window, clip, tmp_path, monkeypatch):
    import threading
    from PySide6.QtGui import QImage
    gate = threading.Event()
    real_save = QImage.save

    def held_save(img, *args):   # the first PNG is still being written when S is pressed again
        gate.wait(10)
        return real_save(img, *args)

    monkeypatch.setattr(QImage, "save", held_save)
    w = window
    settings().setValue("output_dir", str(tmp_path))
    try:
        _open(w, clip, keyframes=False)
        w.save_frame()
        w.save_frame()   # same frame, before the first file exists
        assert len(w._saving) == 2 and not os.listdir(tmp_path)   # both pending, two names
        gate.set()
        assert _wait(lambda: not w._saving)
        pngs = sorted(p for p in os.listdir(tmp_path) if p.endswith(".png"))
        assert len(pngs) == 2 and pngs[0] != pngs[1]
    finally:
        settings().remove("output_dir")


# ------------------------------------------------------------------ export dialog
def test_export_dialog_reports_the_file_it_wrote(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QLabel, QMessageBox
    from simplecut.app import ExportDialog
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x")
    info = VideoInfo(path=str(src), width=160, height=96, fps=25.0, duration=10.0, start_time=1.4)
    dlg = ExportDialog(info, 3.4, 5.4, lossless_ok=True)
    try:
        (clip_label,) = [lab.text() for lab in dlg.findChildren(QLabel) if lab.text().startswith("Clip:")]
        assert "00:00:02.000" in clip_label and "00:00:04.000" in clip_label   # from the first frame
        got = []
        dlg.exported.connect(got.append)
        dlg._out = dlg.out_path.text()
        dlg._set_inputs_enabled(False)
        assert not dlg.out_path.isEnabled() and not dlg.browse.isEnabled() and not dlg.rb_nvenc.isEnabled()
        dlg.out_path.setText("something else")
        dlg._done(True, "")
        assert got == [dlg._out] and dlg.out_path.isEnabled()
    finally:
        dlg.deleteLater()


def test_fmt_time_never_shows_60_seconds():
    from simplecut.app import fmt_time
    assert fmt_time(59.9996) == "00:01:00.000"
    assert fmt_time(-1) == "00:00:00.000" and fmt_time(3661.25) == "01:01:01.250"


# ------------------------------------------------------------------ stored settings
def test_bad_stored_settings_fall_back_to_the_defaults():
    from simplecut.defaults import load_export_defaults, load_ui_defaults
    from simplecut.export import ExportSettings
    st = settings()
    keys = {"defaults/cq": 99, "defaults/preset": "p99", "defaults/fps": "fast",
            "defaults/fixed_seconds": 0.0, "defaults/fixed_frames": 0}
    try:
        for k, v in keys.items():
            st.setValue(k, v)
        s = load_export_defaults()
        assert (s.cq, s.preset, s.fps) == (ExportSettings.cq, ExportSettings.preset, ExportSettings.fps)
        ui = load_ui_defaults()
        assert ui.fixed_seconds >= 0.01 and ui.fixed_frames >= 1
    finally:
        for k in keys:
            st.remove(k)


# ------------------------------------------------------------------ crop aspect picks
def _crop_box(w):
    r = w.view.crop
    return r.x(), r.y(), r.width(), r.height()


def _type_custom(w, text):
    """Type a ratio into the Custom field and press Enter, as the user does."""
    w.aspect_edit.selectAll()
    QTest.keyClicks(w.aspect_edit, text)
    QTest.keyClick(w.aspect_edit, Qt.Key_Return)
    assert w.aspect_edit.text() == text


def test_picking_an_aspect_grows_the_crop_box_to_the_frame_edges(window, clip):
    """On the 160x96 test clip 9:16 is 54x96. Picking 16:9 next kept the 54 px width and gave a
    54x30 box; it is now the largest 16:9 box, 160x90."""
    w = window
    _open(w, clip)
    w.b_crop.setChecked(True)
    w.aspect_combo.setCurrentText("9:16")
    assert _crop_box(w) == pytest.approx((53, 0, 54, 96))
    w.aspect_combo.setCurrentText("16:9")
    assert _crop_box(w) == pytest.approx((0, 3, 160, 90))
    assert w.view.crop_rect_int() == (0, 2, 160, 90) and "160×90" in w.crop_label.text()
    w.aspect_combo.setCurrentText("1:1")
    assert _crop_box(w) == pytest.approx((32, 0, 96, 96))
    w.aspect_combo.setCurrentText("Free")              # Free keeps the box as it is
    assert _crop_box(w) == pytest.approx((32, 0, 96, 96))


def test_custom_aspect_grows_the_box_once_and_enter_again_leaves_it(window, clip):
    from PySide6.QtCore import QRectF
    w = window
    _open(w, clip)
    w.b_crop.setChecked(True)
    w.aspect_combo.setCurrentText("1:1")
    w.aspect_combo.setCurrentText("Custom…")           # nothing typed yet: the box waits
    assert _crop_box(w) == pytest.approx((32, 0, 96, 96))
    _type_custom(w, "2:1")
    assert _crop_box(w) == pytest.approx((0, 8, 160, 80))
    w.view.crop = QRectF(10, 10, 60, 30)               # resized by hand, still 2:1
    QTest.keyClick(w.aspect_edit, Qt.Key_Return)       # Enter again
    assert _crop_box(w) == pytest.approx((10, 10, 60, 30))
    _type_custom(w, "1:2")                             # a new ratio typed: grows again
    assert _crop_box(w) == pytest.approx((16, 0, 48, 96))


def test_enter_on_untouched_custom_text_never_grows_the_box(window, clip):
    """Two cases where the box's shape differs a little from the text in the Custom field, so
    Enter with nothing typed threw a hand-sized box to the largest size: an output height typed
    under "Scale to" (176x100 is not exactly 16:9), and a rotation, after which the field shows
    the rounded "1:2.333" for 9:21."""
    from PySide6.QtCore import QRectF
    w = window
    _open(w, clip)
    w.b_crop.setChecked(True)
    w.aspect_combo.setCurrentText("Custom…")
    _type_custom(w, "16:9")
    w.view.crop = QRectF(40, 20, 64, 36)
    w.rb_scale.setChecked(True)
    w.out_h.setValue(100)
    assert (w.out_w.value(), w.out_h.value()) == (176, 100)
    QTest.keyClick(w.aspect_edit, Qt.Key_Return)
    assert w.view.crop.width() == pytest.approx(64) and w.view.crop.height() < 40

    w.rb_native.setChecked(True)
    w.aspect_combo.setCurrentText("21:9")
    w.view.crop = QRectF(40, 20, 42, 18)
    w.rotate()
    assert _wait(lambda: w.view.frame_size == (96, 160))
    assert w.aspect_combo.currentText() == "Custom…" and w.aspect_edit.text() == "1:2.333"
    assert (w.view.crop.width(), w.view.crop.height()) == pytest.approx((18, 42))
    QTest.keyClick(w.aspect_edit, Qt.Key_Return)
    assert w.view.crop.width() == pytest.approx(18) and w.view.crop.height() < 43


def test_typing_an_output_size_does_not_grow_a_hand_sized_box(window, clip):
    """With a preset on, a typed output width only nudges the ratio by its even-pixel rounding
    (9:16 at width 50 is 50x88 = 0.568): that must not throw the box back to the largest size."""
    from PySide6.QtCore import QRectF
    w = window
    _open(w, clip)
    w.b_crop.setChecked(True)
    w.aspect_combo.setCurrentText("9:16")
    w.view.crop = QRectF(20, 10, 27, 48)
    w.rb_scale.setChecked(True)
    w.out_w.setValue(50)
    assert (w.out_w.value(), w.out_h.value()) == (50, 88)
    x, y, bw, bh = _crop_box(w)
    assert bw == pytest.approx(27) and bh == pytest.approx(27 * 88 / 50)
    w.aspect_combo.setCurrentText("1:1")               # a pick still grows it, scaling or not
    assert (w.out_w.value(), w.out_h.value()) == (50, 50)
    assert _crop_box(w) == pytest.approx((0, 0, 96, 96))


def test_saving_preferences_refits_the_box_without_growing_it(window, clip):
    from PySide6.QtCore import QRectF
    w = window
    _open(w, clip)
    w.b_crop.setChecked(True)
    w.aspect_combo.setCurrentText("1:1")
    w.view.crop = QRectF(40, 20, 40, 40)
    st = settings()
    try:
        st.setValue("defaults/aspect", "16:9")
        w._apply_ui_defaults()                         # what a saved Preferences dialog runs
        assert w.aspect_combo.currentText() == "16:9"
        assert _crop_box(w) == pytest.approx((40, 28.75, 40, 22.5))
        w.aspect_combo.setCurrentText("9:16")          # and picks grow the box again afterwards
        assert _crop_box(w) == pytest.approx((33, 0, 54, 96))
    finally:
        st.remove("defaults/aspect")


def test_picking_an_aspect_under_scale_to_with_free_grows_the_box(window, clip):
    from PySide6.QtCore import QRectF
    w = window
    _open(w, clip)
    w.b_crop.setChecked(True)
    w.view.crop = QRectF(10, 10, 60, 40)
    w.rb_scale.setChecked(True)                        # Free: output seeded 60x40, box unchanged
    assert (w.out_w.value(), w.out_h.value()) == (60, 40)
    assert _crop_box(w) == pytest.approx((10, 10, 60, 40))
    w.aspect_combo.setCurrentText("1:1")
    assert (w.out_w.value(), w.out_h.value()) == (60, 60)
    assert _crop_box(w) == pytest.approx((0, 0, 96, 96))
    w.b_crop.setChecked(False)                         # the box survives crop off and on
    w.b_crop.setChecked(True)
    assert _crop_box(w) == pytest.approx((0, 0, 96, 96))


def test_saving_preferences_with_a_custom_default_does_not_grow_the_box(window, clip):
    from PySide6.QtCore import QRectF
    w = window
    _open(w, clip)
    w.b_crop.setChecked(True)
    w.aspect_combo.setCurrentText("1:1")
    w.view.crop = QRectF(40, 20, 40, 40)
    st = settings()
    try:
        st.setValue("defaults/aspect", "Custom…")
        st.setValue("defaults/custom_aspect", "2:1")
        w._apply_ui_defaults()
        assert w.aspect_combo.currentText() == "Custom…" and w.aspect_edit.text() == "2:1"
        assert _crop_box(w) == pytest.approx((40, 30, 40, 20))
        QTest.keyClick(w.aspect_edit, Qt.Key_Return)   # Enter on the text Preferences wrote
        assert _crop_box(w) == pytest.approx((40, 30, 40, 20))
    finally:
        st.remove("defaults/aspect")
        st.remove("defaults/custom_aspect")


# ------------------------------------------------------------------ fixed length: the last-frame view
def _grey(view):
    """The grey level of the frame a view shows (frame i of the test clip is flat grey 3*i % 256)."""
    return view._pix.toImage().pixelColor(view._pix.width() // 2, view._pix.height() // 2).red()


def _end_settled(w):
    return _wait(lambda: w._end_key is not None and w._end_key == w._end_wanted()
                 and not w._end_timer.isActive())


def _frame_no(w, t):
    return round((t - w.info.start_time) / w.info.frame_duration)


def test_fixed_length_shows_the_clips_last_frame_in_a_second_view(window, clip):
    w = window
    _open(w, clip)
    fd = w.info.frame_duration
    assert w.end_view.isHidden() and w.end_dec is None
    w.len_sec.setValue(2.0)                              # 60 frames
    w.seek_to(w.info.start_time + 10 * fd)
    w.b_fixed.setChecked(True)
    assert not w.end_view.isHidden() and w.end_dec is not None
    assert _end_settled(w)
    assert _frame_no(w, w.sel_last_frame) == 69 and _frame_no(w, w.end_dec.time) == 69
    assert _grey(w.end_view) == pytest.approx(69 * 3, abs=8)
    assert "frame 69" in w.end_view.caption
    # the main view and its decoder are where they were: the clip's first frame
    assert _frame_no(w, w.dec.time) == 10 and _grey(w.view) == pytest.approx(30, abs=8)

    w.sel_move_frames(5)                                 # the clip box moves: both views follow
    assert _end_settled(w)
    assert _frame_no(w, w.dec.time) == 15 and _frame_no(w, w.end_dec.time) == 74
    assert _grey(w.end_view) == pytest.approx(74 * 3, abs=8) and "frame 74" in w.end_view.caption

    w.len_frames.setValue(30)                            # a shorter clip: only the end view changes
    assert _end_settled(w)
    assert _frame_no(w, w.dec.time) == 15 and _frame_no(w, w.end_dec.time) == 44
    assert _grey(w.end_view) == pytest.approx(44 * 3, abs=8)

    w.sel_to_end()                                       # the video's own last frame
    assert _end_settled(w)
    assert _frame_no(w, w.end_dec.time) == 299

    w.b_fixed.setChecked(False)
    assert w.end_view.isHidden() and w.end_dec is None and w.end_view._pix is None
    assert w.end_view.caption == "" and not w._end_timer.isActive()


def test_end_view_is_not_redrawn_while_nothing_about_it_changes(window, clip):
    """_show_current runs for every frame played: it must not seek or draw the end view again."""
    w = window
    _open(w, clip)
    w.b_fixed.setChecked(True)
    assert _end_settled(w)
    drawn = []
    real = w._render_frame
    w._render_frame = lambda view, f: (drawn.append(view), real(view, f))[1]
    for _ in range(5):
        w.step_forward()
        APP.processEvents()
    assert drawn == [w.view] * 5 and not w._end_timer.isActive()
    w.go_end()                                           # the playhead on the clip's last frame
    APP.processEvents()
    assert w.end_view not in drawn
    assert _frame_no(w, w.dec.time) == _frame_no(w, w.end_dec.time)


def test_end_view_follows_rotation_colour_and_the_crop_box(window, clip):
    import numpy as np
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    w = window
    _open(w, clip)
    w.len_sec.setValue(1.0)
    w.b_fixed.setChecked(True)                           # frames 0..29
    assert _end_settled(w)
    assert _grey(w.end_view) == pytest.approx(87, abs=8)

    w.rotate()
    assert _end_settled(w)
    assert w.view.frame_size == (96, 160) and w.end_view.frame_size == (96, 160)

    w._set_color(np.tile(np.arange(255, -1, -1, dtype=np.uint8), (3, 1)), "inverted")
    assert _end_settled(w)
    assert _grey(w.end_view) == pytest.approx(255 - 87, abs=8)
    w._set_color(None, "")
    assert _end_settled(w)
    assert _grey(w.end_view) == pytest.approx(87, abs=8)

    w.b_crop.setChecked(True)
    w.aspect_combo.setCurrentText("1:1")
    assert w.end_view.crop_enabled and w.end_view.crop == w.view.crop and w.end_view.crop is not w.view.crop
    w.view.reset_crop()
    assert w.end_view.crop == w.view.crop
    # the mouse does nothing on the end view
    before = w.end_view.crop
    p = QPointF(w.end_view.width() / 2, w.end_view.height() / 2)
    w.end_view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, p, p, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    w.end_view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, p + QPointF(30, 20), p + QPointF(30, 20),
                                          Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    w.end_view.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, p, p, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
    assert w.end_view.crop == before and w.view.crop == before
    w.b_crop.setChecked(False)
    assert not w.end_view.crop_enabled


def test_end_view_closes_with_the_video_and_on_a_new_one(window, clip, tmp_path):
    w = window
    _open(w, clip)
    w.b_fixed.setChecked(True)
    assert _end_settled(w)
    other = make_video(str(tmp_path / "other.mp4"), n=60, fps=30, gop=30)
    _open(w, other)                                      # a new video switches Fixed length off
    assert not w.b_fixed.isChecked() and w.end_view.isHidden() and w.end_dec is None
    w.b_fixed.setChecked(True)
    assert _end_settled(w)
    assert w.end_dec.container.name.endswith("other.mp4")
    w.close_video()
    assert w.end_view.isHidden() and w.end_dec is None and not w.fixed
    APP.processEvents()                                  # a queued redraw finds nothing to do
    assert w.end_view._pix is None


def test_end_view_resize_draws_its_frame_again_at_the_new_size(window, clip):
    w = window
    _open(w, clip)
    w.b_fixed.setChecked(True)
    assert _end_settled(w)
    from PySide6.QtGui import QResizeEvent
    old = w.end_view.size()
    w.end_view.resize(old.width() + 40, old.height() + 30)
    # (a window that was never shown holds resize events back: deliver it as the layout would)
    APP.sendEvent(w.end_view, QResizeEvent(w.end_view.size(), old))
    assert w._resize_timer.isActive()
    assert _wait(lambda: w._end_key is not None and w._end_key[3:5] == (w.end_view.width(), w.end_view.height()))


def _timeline_mouse(kind, x, button, buttons):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    p = QPointF(x, 10)
    return QMouseEvent(getattr(QEvent, kind), p, p, button, buttons, Qt.NoModifier)


def test_dragging_the_clip_box_on_slow_video_ends_on_the_exact_last_frame(window, clip):
    """On video with slow exact seeks (4K) a drag shows keyframes; the release must bring the
    end view to the clip's real last frame, like the main view."""
    w = window
    _open(w, clip)
    w.timeline.resize(1000, 40)
    w.len_sec.setValue(1.0)
    w.b_fixed.setChecked(True)
    assert _end_settled(w)
    w.dec.seek_ms = w.end_dec.seek_ms = 1000.0          # as measured on heavy video
    tl = w.timeline
    x0 = tl._x(w.sel_start + 0.5)                        # inside the clip box
    tl.mousePressEvent(_timeline_mouse("MouseButtonPress", x0, Qt.LeftButton, Qt.LeftButton))
    for dx in (20, 40, 77):
        tl.mouseMoveEvent(_timeline_mouse("MouseMove", x0 + dx, Qt.NoButton, Qt.LeftButton))
        APP.processEvents()
    assert w.sel_start > w.info.start_time               # the box moved
    assert w._end_key is None and w.end_dec.current.key_frame   # a keyframe stands in, marked stale
    tl.mouseReleaseEvent(_timeline_mouse("MouseButtonRelease", x0 + 77, Qt.LeftButton, Qt.NoButton))
    assert _end_settled(w)
    last = _frame_no(w, w.sel_last_frame)
    assert _frame_no(w, w.end_dec.time) == last and f"frame {last}" in w.end_view.caption
    assert _grey(w.end_view) == pytest.approx(last * 3 % 256, abs=8)


def test_end_frame_that_fails_to_decode_is_not_retried_for_every_frame_played(window, clip):
    w = window
    _open(w, clip)
    w.b_fixed.setChecked(True)
    assert _end_settled(w)
    calls = []

    def broken(t):
        calls.append(t)
        raise OSError("read error")
    w.end_dec.seek = broken
    w.sel_move_frames(3)
    assert _end_settled(w)
    assert len(calls) == 1 and "can't be decoded" in w.end_view.caption
    for _ in range(4):                                   # what playback does for each frame
        w._show_current()
        APP.processEvents()
    assert len(calls) == 1 and not w._end_timer.isActive()
    del w.end_dec.seek                                   # readable again: the next move shows it
    w.sel_move_frames(1)
    assert _end_settled(w)
    assert _frame_no(w, w.end_dec.time) == _frame_no(w, w.sel_last_frame)
    assert "frame" in w.end_view.caption


def test_end_view_follows_flip_and_a_rotation_keeps_the_box_until_the_redraw(window, clip):
    w = window
    _open(w, clip)
    w.b_fixed.setChecked(True)
    w.b_crop.setChecked(True)
    w.aspect_combo.setCurrentText("1:1")
    assert _end_settled(w)
    before = w.end_view._pix.toImage()
    w.flip_h()
    assert _end_settled(w)
    assert w._end_key[2] is True and w.end_view.crop == w.view.crop
    assert w.end_view._pix.toImage().size() == before.size()
    w.rotate()
    # not yet redrawn: the end view still has the old frame and, with it, the old box
    assert w.end_view.frame_size == (160, 96) and w.view.frame_size == (96, 160)
    assert w.end_view.crop != w.view.crop
    assert _end_settled(w)
    assert w.end_view.frame_size == (96, 160) and w.end_view.crop == w.view.crop


def test_a_failed_open_keeps_the_end_view_of_the_video_that_stays(window, clip, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    w = window
    _open(w, clip)
    w.b_fixed.setChecked(True)
    assert _end_settled(w)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)
    bad = tmp_path / "not-a-video.mp4"
    bad.write_bytes(b"nothing to decode here")
    token = w._probe_token
    w.open_file(str(bad))
    assert _wait(lambda: w._probe_token != token and w._probe_thread is not None
                 and not w._probe_thread.isRunning())
    for _ in range(20):
        APP.processEvents()
    assert w.info is not None and w.info.path == clip and w.b_fixed.isChecked()
    assert not w.end_view.isHidden() and w.end_dec is not None and w.end_view._pix is not None
