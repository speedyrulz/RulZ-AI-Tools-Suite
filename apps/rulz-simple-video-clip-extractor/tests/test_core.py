"""Run with:  python -m pytest tests   (from the app folder, inside its .venv)."""
import os
import random
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
    (lossless,) = build_commands(_info(), 1.0, 2.0, "out.mp4", ExportSettings(mode="lossless"))
    assert "-hwaccel" not in lossless


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
