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
