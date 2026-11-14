"""Run with:  python -m pytest tests   (from the app folder, inside its .venv)."""
import os
import random
import subprocess
import sys
import tempfile
import time

import av
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")  # no window needed; "offscreen" can hang on Windows
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# never read or write the real preferences (the registry) from tests
os.environ.setdefault("SIMPLECUT_SETTINGS_INI", os.path.join(tempfile.mkdtemp(), "settings.ini"))
sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication  # noqa: E402

from simplecut import colormatch  # noqa: E402
from simplecut.decoder import Decoder  # noqa: E402
from simplecut.export import ExportJob, ExportSettings, build_commands  # noqa: E402
from simplecut.probe import VideoInfo  # noqa: E402
from simplecut.timeline import Timeline  # noqa: E402

APP = QApplication.instance() or QApplication([])
SAMPLE = os.path.join(ROOT, "samples", "test_pattern_10s.mp4")


# ------------------------------------------------------------------ test videos
def make_video(path, n=75, fps=25, gop=12, codec="mpeg4", **opts):
    with av.open(path, "w") as out:
        st = out.add_stream(codec, rate=fps)
        st.width, st.height, st.pix_fmt = 160, 96, "yuv420p"
        st.codec_context.gop_size = gop
        for k, v in opts.items():
            setattr(st.codec_context, k, v)
        for i in range(n):
            img = np.full((96, 160, 3), (i * 3) % 256, dtype=np.uint8)
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    return path


def all_times(path):
    with av.open(path) as c:
        return [f.time for f in c.decode(video=0) if f.time is not None]


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("vid")
    out = [make_video(str(d / "mpeg4.mp4"))]
    try:  # H.264 with B-frames: decode order differs from display order
        out.append(make_video(str(d / "h264.mp4"), codec="libx264"))
    except Exception:  # noqa: BLE001 - PyAV build without libx264
        pass
    if os.path.exists(SAMPLE):
        out.append(SAMPLE)
    return out


def count_restarts(dec):
    calls = []
    real = dec._restart
    dec._restart = lambda t: (calls.append(t), real(t))[1]
    return calls


# ------------------------------------------------------------------ decoder
def test_step_back_walks_every_frame_in_order(videos):
    for path in videos:
        times = all_times(path)
        dec = Decoder(path)
        dec._back_max = 3  # small cache: exercises both the cached and the GOP-walk path
        try:
            i = min(len(times) - 1, 47)
            dec.seek(times[i])
            assert dec.time == pytest.approx(times[i])
            while i > 0:
                dec.step_back()
                i -= 1
                assert dec.time == pytest.approx(times[i]), (path, i)
            for i in range(1, min(len(times), 30)):
                assert dec.step_forward() is not None
                assert dec.time == pytest.approx(times[i]), (path, i)
        finally:
            dec.close()


def test_random_steps_match_reference(videos):
    rnd = random.Random(1)
    for path in videos:
        times = all_times(path)
        dec = Decoder(path)
        dec._back_max = 4
        try:
            i = len(times) // 2
            dec.seek(times[i])
            for _ in range(200):
                op = rnd.choice("bbbff" if i > 0 else "f")
                if op == "b" and i > 0:
                    dec.step_back()
                    i -= 1
                elif op == "f" and i < len(times) - 1:
                    dec.step_forward()
                    i += 1
                assert dec.time == pytest.approx(times[i]), (path, i)
        finally:
            dec.close()


def test_step_back_then_forward_decodes_the_gop_once(videos):
    path = videos[0]
    times = all_times(path)
    dec = Decoder(path)
    try:
        dec.seek(times[30])
        dec._back.clear()  # as if nothing were cached
        calls = count_restarts(dec)
        dec.step_back()
        assert dec.time == pytest.approx(times[29]) and len(calls) == 1
        dec.step_forward()  # the frame the walk ran into is the lookahead: no second seek
        assert dec.time == pytest.approx(times[30]) and len(calls) == 1
        dec.step_back()  # now cached
        dec.step_back()
        assert dec.time == pytest.approx(times[28]) and len(calls) == 1
    finally:
        dec.close()


def test_step_back_after_stepping_forward_needs_no_seek(videos):
    path = videos[0]
    times = all_times(path)
    dec = Decoder(path)
    try:
        dec.seek(times[10])
        for _ in range(5):
            dec.step_forward()
        calls = count_restarts(dec)
        for k in range(5):
            dec.step_back()
        assert dec.time == pytest.approx(times[10]) and calls == []
    finally:
        dec.close()


def test_first_frame_and_end(videos):
    path = videos[0]
    times = all_times(path)
    dec = Decoder(path)
    try:
        dec.seek(times[0])
        dec.step_back()
        assert dec.time == pytest.approx(times[0])
        dec.seek_last(times[-1])
        assert dec.time == pytest.approx(times[-1])
        assert dec.step_forward() is None
        dec.step_back()
        assert dec.time == pytest.approx(times[-2])
        dec.step_forward()
        assert dec.time == pytest.approx(times[-1])
    finally:
        dec.close()


# ------------------------------------------------------------------ keyframe helpers
def _linear(info):
    tol = info.frame_duration / 2

    def keyframe_at(t):
        return next((k for k in info.keyframes if abs(k - t) <= tol), None)

    def next_kf(t):
        return next((k for k in info.keyframes if k > t + tol), None)

    def prev_kf(t):
        return next((k for k in reversed(info.keyframes) if k < t - tol), None)

    return keyframe_at, next_kf, prev_kf


def test_keyframe_bisect_matches_linear_scan():
    rnd = random.Random(7)
    for fps in (23.976, 25.0, 60.0):
        fd = 1 / fps
        frames = [round(i * fd, 6) for i in range(3000)]
        info = VideoInfo(path="x", fps=fps, keyframes=sorted(rnd.sample(frames, 120)))
        keyframe_at, next_kf, prev_kf = _linear(info)
        probes = frames[::7] + [rnd.uniform(-1, 130) for _ in range(500)]
        for t in probes:
            assert info.keyframe_at(t) == keyframe_at(t), t
            assert info.is_keyframe_time(t) == (keyframe_at(t) is not None), t
            assert info.next_keyframe(t) == next_kf(t), t
            assert info.prev_keyframe(t) == prev_kf(t), t
    empty = VideoInfo(path="x")
    assert empty.keyframe_at(1) is None and empty.next_keyframe(1) is None and empty.prev_keyframe(1) is None


# ------------------------------------------------------------------ timeline
def test_timeline_ticks_are_cached_and_thinned():
    tl = Timeline()
    tl.resize(600, 56)
    tl.set_range(0.0, 3 * 3600.0)
    tl.set_keyframes([i * 1.0 for i in range(10800)])
    a = tl._tick_lines(18)
    assert 0 < len(a) <= 600 // 2 + 1
    assert tl._tick_lines(18) is a  # repaint without changes: no rebuild
    tl.resize(900, 56)
    assert tl._tick_lines(18) is not a
    tl.grab()  # paints without errors


# ------------------------------------------------------------------ export
def _info():
    return VideoInfo(path="in.mp4", width=3840, height=2160, fps=30.0, duration=10.0, keyframes=[0.0, 1.0, 2.0])


def test_nvenc_tries_gpu_decoding_first_then_without():
    hw, sw = build_commands(_info(), 1.0, 3.0, "out.mp4", ExportSettings(mode="nvenc"))
    i = hw.index("-hwaccel")
    assert hw[i + 1] == "cuda" and i < hw.index("-i")
    assert "-hwaccel" not in sw
    assert [c for c in hw if c not in ("-hwaccel", "cuda")] == sw
    every, av_only = build_commands(_info(), 1.0, 2.0, "out.mp4", ExportSettings(mode="lossless"))
    assert "-hwaccel" not in every and "-hwaccel" not in av_only
    # lossless copies every track, then (if the container refuses one) just video and audio
    assert every[every.index("-map") + 1] == "0"
    assert [av_only[i + 1] for i, c in enumerate(av_only) if c == "-map"] == ["0:v", "0:a?"]


def test_lossless_cut_into_a_container_that_refuses_a_track_keeps_video_and_audio(tmp_path):
    """MKV with ASS subtitles saved as .mp4: copying every track fails ("Could not find tag for
    codec ass"), so the retry copies the video and audio and says what was left out."""
    from simplecut.export import make_export_job
    from simplecut.probe import probe
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:05,000\nhello\n")
    src = str(tmp_path / "subs.mkv")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=25",
                        "-f", "lavfi", "-i", "sine=frequency=440", "-i", str(srt), "-t", "3",
                        "-map", "0", "-map", "1", "-map", "2", "-c:v", "libx264", "-g", "25", "-c:a", "aac",
                        "-c:s", "ass", src], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    info = probe(src)
    out = str(tmp_path / "cut.mp4")
    job = make_export_job(info, info.keyframes[1], info.end_time, out, ExportSettings(mode="lossless"))
    got = _run_job(job, timeout=60)
    assert got and got[0][0] is True, got
    assert job.warning and "subtitles" in job.warning
    kinds = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", out],
                           capture_output=True, text=True).stdout.split()
    assert kinds == ["video", "audio"]
    # into MKV every track fits: one attempt, nothing to warn about
    job = make_export_job(info, info.keyframes[1], info.end_time, str(tmp_path / "cut.mkv"),
                          ExportSettings(mode="lossless"))
    got = _run_job(job, timeout=60)
    assert got and got[0][0] is True and job.warning == "" and job._attempt == 0


def test_retry_note_survives_a_long_log():
    py = sys.executable
    noisy = [py, "-c", "import sys\nfor i in range(200): print('line', i, file=sys.stderr)\nsys.exit(1)"]
    job = ExportJob([noisy, noisy], 1.0, retry_note="(the retry note)")
    got = _run_job(job)
    assert got and got[0][0] is False and got[0][1].splitlines()[-1] == "(the retry note)"


def test_clip_names_count_from_the_first_frame():
    """MPEG-TS clocks often start at 1.4 s (or hours in); the name uses the times the app shows."""
    from simplecut.export import default_output_path
    info = VideoInfo(path="C:/v/rec.ts", fps=25.0, duration=60.0, start_time=1.4)
    assert default_output_path(info, 3.4, 8.9, "nvenc").endswith("rec_clip_2_00-7_50.mp4")
    assert default_output_path(info, 1.4, 61.4, "lossless", whole=True).endswith("rec_clip_full.ts")


def _run_job(job, timeout=20):
    got = []
    job.finished.connect(lambda ok, log: got.append((ok, log)))
    job.start()
    end = time.time() + timeout
    while not got and time.time() < end:
        APP.processEvents()
        time.sleep(0.01)
    return got


def test_job_reports_ffmpeg_that_cannot_start():
    got = _run_job(ExportJob(["definitely-not-ffmpeg-xyz.exe", "-version"], 1.0))
    assert got and got[0][0] is False and "Could not start" in got[0][1]


def test_job_falls_back_to_next_command():
    py = sys.executable
    job = ExportJob([[py, "-c", "import sys; sys.exit(3)"], [py, "-c", "pass"]], 1.0)
    got = _run_job(job)
    assert got == [(True, got[0][1])] and job.cmd[-1] == "pass"


def test_cancelled_job_does_not_retry():
    py = sys.executable
    job = ExportJob([[py, "-c", "import time; time.sleep(30)"], [py, "-c", "pass"]], 1.0)
    got = []
    job.finished.connect(lambda ok, log: got.append(ok))
    job.start()
    job.proc.waitForStarted(5000)
    job.cancel()
    assert job.wait(5000)
    assert got == [False] and job.cancelled and job.cmd[-1] != "pass"


# ------------------------------------------------------------------ temp files
def test_clean_temp_removes_only_old_files(tmp_path, monkeypatch):
    monkeypatch.setattr(colormatch, "temp_dir", lambda: str(tmp_path))
    old, new, other = tmp_path / "match_aaaa.cube", tmp_path / "ref_bbbb.png", tmp_path / "keep.txt"
    for p in (old, new, other):
        p.write_text("x")
    past = time.time() - 3 * 24 * 3600
    os.utime(old, (past, past))
    os.utime(other, (past, past))
    assert colormatch.clean_temp() == 1
    assert not old.exists() and new.exists() and other.exists()


# ------------------------------------------------------------------ timeline dragging
def _mouse(widget, kind, x, held):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    types = {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove, "release": QEvent.MouseButtonRelease}
    pos = QPointF(x, 30)
    ev = QMouseEvent(types[kind], pos, widget.mapToGlobal(pos), Qt.LeftButton if kind != "move" else Qt.NoButton,
                     Qt.LeftButton if held else Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)


def test_timeline_reports_drags_but_not_clicks():
    tl = Timeline()
    tl.resize(600, 56)
    tl.set_range(0.0, 100.0)
    seen, ends = [], []
    tl.seekRequested.connect(lambda t: seen.append(tl.scrubbing))
    tl.scrubFinished.connect(lambda: ends.append(True))
    _mouse(tl, "press", 100, True)
    _mouse(tl, "release", 100, False)
    assert seen == [False] and ends == []  # a click asks for one exact frame
    _mouse(tl, "press", 100, True)
    _mouse(tl, "move", 140, True)
    _mouse(tl, "move", 180, True)
    _mouse(tl, "release", 180, False)
    assert seen == [False, False, True, True] and ends == [True] and not tl.scrubbing


# ------------------------------------------------------------------ export dialog
def test_export_dialog_suggests_a_name_that_is_not_taken(tmp_path):
    from simplecut.app import ExportDialog
    from simplecut.export import default_output_path
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x")
    info = VideoInfo(path=str(src), width=160, height=96, fps=25.0, duration=10.0)
    taken = default_output_path(info, 1.0, 2.0, "nvenc")
    open(taken, "wb").close()
    dlg = ExportDialog(info, 1.0, 2.0, lossless_ok=False)
    try:
        suggested = dlg.out_path.text()
        assert suggested != taken and not os.path.exists(suggested)
        assert os.path.dirname(suggested) == os.path.dirname(taken)
    finally:
        dlg.deleteLater()


# ------------------------------------------------------------------ opening a file
def test_probe_reports_the_video_before_its_keyframes(videos):
    from simplecut.app import ProbeThread
    got = []
    t = ProbeThread(videos[0], token=7)
    t.done.connect(lambda info, tok: got.append(("info", tok, len(info.keyframes))))
    t.keyframes.connect(lambda kfs, dts, tok: got.append(("keyframes", tok, len(kfs))))
    t.failed.connect(lambda msg, tok: got.append(("failed", msg)))
    t.start()
    end = time.time() + 30
    while len(got) < 2 and time.time() < end:
        APP.processEvents()
        time.sleep(0.01)
    t.wait()
    assert [g[:2] for g in got] == [("info", 7), ("keyframes", 7)]
    assert got[0][2] == 0 and got[1][2] > 1  # the info arrives without keyframes; they follow
