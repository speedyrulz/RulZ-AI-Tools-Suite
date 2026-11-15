"""The sound strip: the envelope read from a video's audio (timing, formats, gaps), storage, the
strip's window, zoom and seeking, and the app reading it in the background."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from fractions import Fraction

import av
import numpy as np
import pytest

from test_core import APP  # noqa: F401  (sets up Qt, the settings file and sys.path)

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QWheelEvent  # noqa: E402

from simplecut import waveform  # noqa: E402
from simplecut.defaults import settings  # noqa: E402
from simplecut.probe import have_ffmpeg  # noqa: E402
from simplecut.timeline import Timeline  # noqa: E402
from simplecut.waveform import RATE, Envelope, WaveformStrip, envelope_of  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg / ffprobe not on PATH")

DUR = 5
# A tone in the left channel from 1.0 to 1.5 s and in the right one from 3.0 to 3.25 s; silence
# everywhere else.
LEFT = "if(between(t,1,1.5),0.8*sin(2*PI*440*t),0)"
RIGHT = "if(between(t,3,3.25),0.5*sin(2*PI*660*t),0)"
BURSTS = [(1.0, 1.5, 0.8), (3.0, 3.25, 0.5)]
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _wait(cond, timeout=30):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        APP.processEvents()
        time.sleep(0.01)
    return cond()


@pytest.fixture(autouse=True)
def clean_settings_and_store():
    shutil.rmtree(waveform.waves_dir(), ignore_errors=True)
    s = settings()
    for k in ("show_waveform", "waveform_span"):
        s.remove(k)
    yield
    shutil.rmtree(waveform.waves_dir(), ignore_errors=True)
    for k in ("show_waveform", "waveform_span"):
        s.remove(k)


def make_video(path: str, acodec: str | None = "aac", audio_offset: float = 0.0, video_offset: float = 0.0,
               extra: list[str] | None = None) -> str:
    args = ["ffmpeg", "-v", "error", "-y"]
    if video_offset:
        args += ["-itsoffset", str(video_offset)]
    args += ["-f", "lavfi", "-i", f"color=c=gray:s=64x36:r=25:d={DUR}"]
    if acodec:
        if audio_offset:
            args += ["-itsoffset", str(audio_offset)]
        args += ["-f", "lavfi", "-i", f"aevalsrc=exprs='{LEFT}|{RIGHT}':s=48000:d={DUR}"]
    args += ["-map", "0:v"] + (["-map", "1:a", "-c:a", acodec] if acodec else [])
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-g", "25", *(extra or []), path]
    subprocess.run(args, check=True, creationflags=NO_WINDOW)
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    if not have_ffmpeg():
        pytest.skip("ffmpeg / ffprobe not on PATH")
    d = tmp_path_factory.mktemp("waves")
    return {
        "aac": make_video(str(d / "aac.mp4"), "aac"),                    # fltp: planar float
        "s16": make_video(str(d / "s16.mkv"), "pcm_s16le"),              # s16: packed int
        "s24": make_video(str(d / "s24.mov"), "pcm_s24le"),              # s32: packed int
        "f32": make_video(str(d / "f32.mkv"), "pcm_f32le"),              # flt: packed float
        "u8": make_video(str(d / "u8.mov"), "pcm_u8"),                   # u8: packed unsigned
        "late": make_video(str(d / "late.mkv"), "pcm_s16le", audio_offset=2.0),
        "early": make_video(str(d / "early.mkv"), "pcm_s16le", video_offset=1.5),
        "mute": make_video(str(d / "mute.mp4"), None),
    }


@pytest.fixture
def video(media, tmp_path):
    """A copy of the AAC video of its own (stored envelopes are per file)."""
    p = str(tmp_path / "v.mp4")
    shutil.copyfile(media["aac"], p)
    return p


def loudest(env: Envelope, a: float, b: float) -> float:
    i0 = max(int(np.floor(a * RATE)) - env.first, 0)
    i1 = min(int(np.ceil(b * RATE)) - env.first, len(env))
    return float(env.peak[i0:i1].max()) if i1 > i0 else 0.0


def check_bursts(env: Envelope, offset: float = 0.0, tol: float = 0.03):
    """Each tone is loud where it was put and the gaps are silent, in the container's clock."""
    gaps = [(0.0, 1.0), (1.5, 3.0), (3.25, float(DUR))]
    for a, b, amp in BURSTS:
        assert loudest(env, offset + a + tol, offset + b - tol) == pytest.approx(amp, abs=0.03)
    for a, b in gaps:
        assert loudest(env, offset + a + tol, offset + b - tol) < 0.01, (a, b)


# ------------------------------------------------------------------ reading
@needs_ffmpeg
@pytest.mark.parametrize("kind", ["aac", "s16", "s24", "f32", "u8"])
def test_the_envelope_shows_the_tones_and_the_silence_between(media, kind):
    with av.open(media[kind]) as c:
        fmt = c.streams.audio[0].format.name
    assert fmt == {"aac": "fltp", "s16": "s16", "s24": "s32", "f32": "flt", "u8": "u8"}[kind]
    env = waveform.read_envelope(media[kind])
    assert env is not None and abs(env.start) < 0.05 and env.end == pytest.approx(DUR, abs=0.1)
    check_bursts(env)
    # RMS over both channels: a 0.8 sine in one of two is sqrt(0.8**2 / 2 / 2) = 0.4
    assert env.rms[int(1.25 * RATE) - env.first] == pytest.approx(0.4, abs=0.02)


@needs_ffmpeg
def test_sound_that_starts_after_the_video_is_placed_by_its_own_times(media):
    env = waveform.read_envelope(media["late"])
    assert env.start == pytest.approx(2.0, abs=0.01)
    check_bursts(env, offset=2.0)
    assert loudest(env, 0.0, 1.99) == 0.0   # (no data before it at all)


@needs_ffmpeg
def test_sound_that_starts_before_the_video_keeps_the_containers_clock(media):
    from simplecut.probe import probe_streams
    info = probe_streams(media["early"])
    assert info.start_time == pytest.approx(1.5, abs=0.05)
    env = waveform.read_envelope(media["early"])
    assert env.start == pytest.approx(0.0, abs=0.01)
    check_bursts(env)   # the tone at 1.0 s is before the video's first frame, and stays there


@needs_ffmpeg
def test_a_video_without_sound_has_no_envelope(media):
    assert waveform.read_envelope(media["mute"]) is None


@needs_ffmpeg
def test_the_read_reports_progress_and_can_be_stopped(media):
    seen = []
    waveform.read_envelope(media["aac"], progress=seen.append)
    assert all(0.0 <= f <= 1.0 for f in seen)
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 5

    with pytest.raises(waveform.Cancelled):
        waveform.read_envelope(media["aac"], cancelled=stop)


def _frame(samples: np.ndarray, fmt: str, rate: int = 8000, t: float | None = None):
    f = av.AudioFrame.from_ndarray(samples, format=fmt, layout="stereo")
    f.sample_rate = rate
    if t is not None:
        f.time_base = Fraction(1, rate)
        f.pts = int(round(t * rate))
    return f


def test_frames_without_times_follow_the_one_before_and_gaps_stay_silent():
    rate = 8000
    n = 800   # 0.1 s
    loud = np.full((2, n), 0.5, dtype=np.float32)
    quiet = np.zeros((2, n), dtype=np.float32)
    frames = [_frame(loud, "fltp", rate, 1.0), _frame(quiet, "fltp", rate), _frame(loud, "fltp", rate),
              _frame(loud, "fltp", rate, 2.0)]   # 1.0-1.1 loud, 1.1-1.2 quiet, 1.2-1.3 loud, gap, 2.0-2.1
    assert frames[1].time is None
    env = envelope_of(frames)
    assert env.start == pytest.approx(1.0)
    assert loudest(env, 1.0, 1.1) == pytest.approx(0.5, abs=1e-3)
    assert loudest(env, 1.1, 1.2) == 0.0
    assert loudest(env, 1.2, 1.3) == pytest.approx(0.5, abs=1e-3)
    assert loudest(env, 1.3, 2.0) == 0.0
    assert loudest(env, 2.0, 2.1) == pytest.approx(0.5, abs=1e-3)
    assert env.end == pytest.approx(2.1)


def test_a_frame_before_the_first_grows_the_envelope_at_the_front():
    rate = 8000
    loud = np.full((2, 400), 0.25, dtype=np.float32)
    env = envelope_of([_frame(loud, "fltp", rate, 5.0), _frame(loud * 2, "fltp", rate, 1.0)])
    assert env.start == pytest.approx(1.0) and env.end == pytest.approx(5.05)
    assert loudest(env, 1.0, 1.05) == pytest.approx(0.5) and loudest(env, 5.0, 5.05) == pytest.approx(0.25)
    assert loudest(env, 1.06, 4.99) == 0.0


def test_packed_and_planar_samples_and_every_channel_count():
    """One channel loud is loud: the envelope takes the loudest channel, not their average."""
    rate = 8000
    planar = np.zeros((2, 800), dtype=np.float32)
    planar[1] = 0.5
    packed = np.zeros((1, 1600), dtype=np.int16)
    packed[0, 1::2] = 16384          # interleaved: the right channel at half scale
    e1 = envelope_of([_frame(planar, "fltp", rate, 0.0)])
    e2 = envelope_of([_frame(packed, "s16", rate, 0.0)])
    assert e1.peak.max() == pytest.approx(0.5, abs=1e-3) and e2.peak.max() == pytest.approx(0.5, abs=1e-3)
    assert np.allclose(e1.peak, e2.peak, atol=1e-3) and np.allclose(e1.rms, e2.rms, atol=1e-3)
    u8 = np.full((1, 1600), 128, dtype=np.uint8)
    assert envelope_of([_frame(u8, "u8", rate, 0.0)]).peak.max() == 0.0   # (128 is silence)


def test_no_frames_make_an_empty_envelope():
    env = envelope_of([])
    assert len(env) == 0
    p, r, inside = env.columns(0.0, 8.0, 100)
    assert not inside.any() and not p.any()


# ------------------------------------------------------------------ columns
def test_columns_take_the_loudest_bucket_and_mark_where_there_is_no_sound():
    peak = np.zeros(400, dtype=np.float32)      # 2 s from 1.0 s
    peak[100] = 0.9                              # at 1.5 s
    env = Envelope(RATE, peak, peak / 2)
    p, r, inside = env.columns(0.0, 4.0, 40)    # 0.1 s = 20 buckets per column
    assert inside.tolist() == [False] * 10 + [True] * 20 + [False] * 10
    assert p[15] == pytest.approx(0.9) and r[15] == pytest.approx(0.45)
    assert p.sum() == pytest.approx(0.9)
    # narrower columns than buckets: each shows the bucket under it
    p, _, inside = env.columns(1.5, 0.05, 50)   # 10 buckets over 50 columns
    assert inside.all() and (p[:5] == pytest.approx(0.9)) and not p[5:].any()


def test_samples_on_a_bucket_boundary_stay_in_their_bucket():
    """Bucket k starts at k/200 s; at 48 kHz that is sample 240 k exactly, which float rounding
    used to push into the bucket after for some k."""
    rate = 48000
    x = np.zeros((2, rate), dtype=np.float32)
    x[:, ::240] = 1.0                           # the first sample of every bucket
    env = envelope_of([_frame(x, "fltp", rate, 0.0)])
    assert len(env) == RATE and (env.peak == 1.0).all()


def test_damaged_float_audio_does_not_blank_the_envelope():
    rate = 8000
    x = np.full((2, 800), 0.25, dtype=np.float32)
    x[0, 10], x[1, 400], x[0, 700] = np.nan, np.inf, -np.inf
    env = envelope_of([_frame(x, "fltp", rate, 0.0)])
    assert np.isfinite(env.peak).all() and np.isfinite(env.rms).all() and env.peak.min() >= 0.25 - 1e-3


def test_columns_narrower_than_a_bucket_show_the_first_one_too():
    peak = np.full(10, 0.5, dtype=np.float32)
    env = Envelope(RATE, peak, peak)             # from 1.0 s
    p, _, inside = env.columns(1.0, 1 / RATE, 5)   # the first bucket over 5 columns
    assert inside.all() and (p == pytest.approx(0.5))


def test_the_drawn_height_is_on_a_db_scale():
    lv = waveform.level(np.array([0.0, 10 ** (waveform.FLOOR_DB / 20), 0.1, 1.0, 2.0]))
    assert lv[0] == 0 and lv[1] == pytest.approx(0, abs=1e-6) and lv[3] == 1 and lv[4] == 1
    assert 0.5 < lv[2] < 0.65


# ------------------------------------------------------------------ storage
def test_stored_envelopes_come_back_for_the_same_file_only(tmp_path):
    v = tmp_path / "a.mp4"
    v.write_bytes(b"x" * 100)
    env = Envelope(-5, np.linspace(0, 1, 50, dtype=np.float32), np.linspace(0, 0.5, 50, dtype=np.float32))
    assert waveform.load_envelope(str(v)) is None
    assert waveform.save_envelope(str(v), env)
    back = waveform.load_envelope(str(v))
    assert back.first == -5
    assert np.allclose(back.peak, env.peak, atol=1e-3) and np.allclose(back.rms, env.rms, atol=1e-3)
    v.write_bytes(b"y" * 101)   # another file of that name
    assert waveform.load_envelope(str(v)) is None


def test_a_file_that_changed_during_the_read_is_not_stored(tmp_path):
    v = tmp_path / "a.mp4"
    v.write_bytes(b"x" * 100)
    stamp = (os.stat(v).st_size, os.stat(v).st_mtime)
    v.write_bytes(b"x" * 200)
    env = Envelope(0, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    assert not waveform.save_envelope(str(v), env, stamp)
    assert not os.path.exists(waveform.waves_path(str(v)))


@pytest.mark.parametrize("damage", ["garbage", "version", "lengths", "nan", "no-meta"])
def test_a_damaged_or_foreign_store_file_means_read_again(tmp_path, damage):
    import json
    v = tmp_path / "a.mp4"
    v.write_bytes(b"x" * 100)
    path = waveform.waves_path(str(v))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    st = os.stat(v)
    meta = {"version": waveform.WAVE_VERSION, "path": str(v), "size": st.st_size, "mtime": st.st_mtime}
    arrays = {"first": np.array(0), "peak": np.zeros(10, np.float16), "rms": np.zeros(10, np.float16)}
    if damage == "garbage":
        with open(path, "wb") as f:
            f.write(b"not a zip")
    else:
        if damage == "version":
            meta["version"] = 999
        elif damage == "lengths":
            arrays["rms"] = arrays["rms"][:5]
        elif damage == "nan":
            arrays["peak"] = arrays["peak"].copy()
            arrays["peak"][3] = np.nan
        extra = {} if damage == "no-meta" else {"meta": np.array(json.dumps(meta))}
        with open(path, "wb") as f:
            np.savez(f, **arrays, **extra)
    assert waveform.load_envelope(str(v)) is None


def test_old_envelopes_of_missing_videos_are_cleaned_up(tmp_path):
    kept, gone = tmp_path / "kept.mp4", tmp_path / "gone.mp4"
    env = Envelope(0, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    for v in (kept, gone):
        v.write_bytes(b"x" * 10)
        assert waveform.save_envelope(str(v), env)
    os.remove(gone)
    d = waveform.waves_dir()
    with open(os.path.join(d, "abc.123.tmp.npz"), "wb") as f:
        f.write(b"half")
    waveform.clean_envelopes(now=time.time() + 3600)
    assert len(os.listdir(d)) == 3
    from simplecut.cliplist import STALE_DAYS
    waveform.clean_envelopes(now=time.time() + (STALE_DAYS + 1) * 86400)
    assert os.listdir(d) == [os.path.basename(waveform.waves_path(str(kept)))]


# ------------------------------------------------------------------ the strip
@pytest.fixture
def strip():
    tl = Timeline()
    tl.resize(816, 56)
    tl.set_range(0.0, 100.0)
    tl.set_position(50.0)
    s = WaveformStrip(tl, span=8.0)
    s.resize(816, waveform.HEIGHT)   # 800 px between the paddings
    s.set_envelope(Envelope(0, np.full(100 * RATE, 0.3, np.float32), np.full(100 * RATE, 0.1, np.float32)))
    return s


def _mouse(kind, x, buttons=Qt.LeftButton):
    btn = Qt.LeftButton if kind != QEvent.MouseMove else Qt.NoButton
    return QMouseEvent(kind, QPointF(x, 20), QPointF(x, 20), btn, buttons, Qt.NoModifier)


def _wheel(notches: float):
    return QWheelEvent(QPointF(100, 20), QPointF(100, 20), QPoint(0, 0), QPoint(0, int(120 * notches)),
                       Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)


def test_the_window_has_the_playhead_in_the_middle_unless_at_an_end(strip):
    s, tl = strip, strip.timeline
    assert s.view_span() == pytest.approx((46.0, 8.0))
    tl.set_position(1.0)
    assert s.view_span() == pytest.approx((0.0, 8.0))
    tl.set_position(99.5)
    assert s.view_span() == pytest.approx((92.0, 8.0))
    tl.set_range(10.0, 13.0)   # shorter than the span: the whole video
    tl.set_position(11.0)
    assert s.view_span() == pytest.approx((10.0, 3.0))


def test_a_click_seeks_to_the_time_under_the_mouse_and_a_drag_keeps_the_window(strip):
    s, tl = strip, strip.timeline
    seeks, finished = [], []
    s.seekRequested.connect(seeks.append)
    s.scrubFinished.connect(lambda: finished.append(1))
    s.mousePressEvent(_mouse(QEvent.MouseButtonPress, 8 + 600))   # 3/4 of 46..54
    assert seeks == pytest.approx([52.0], abs=0.02) and not s.scrubbing
    tl.set_position(seeks[-1])     # the app moves the playhead; the window must stay put meanwhile
    s.mouseMoveEvent(_mouse(QEvent.MouseMove, 8 + 700))
    assert seeks[-1] == pytest.approx(53.0, abs=0.02) and s.scrubbing
    s.mouseMoveEvent(_mouse(QEvent.MouseMove, 5000))   # past the edge: the window's end
    assert seeks[-1] == pytest.approx(54.0, abs=0.02)
    s.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, 5000, Qt.NoButton))
    assert finished == [1] and not s.scrubbing
    tl.set_position(54.0)
    assert s.view_span()[0] == pytest.approx(50.0)   # released: centred on the playhead again
    # a plain click is not a scrub
    s.mousePressEvent(_mouse(QEvent.MouseButtonPress, 8))
    s.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, 8, Qt.NoButton))
    assert finished == [1] and seeks[-1] == pytest.approx(50.0, abs=0.02)


def test_a_click_never_seeks_outside_the_video(strip):
    s, tl = strip, strip.timeline
    tl.set_range(10.0, 13.0)
    s.set_span(8.0)
    seeks = []
    s.seekRequested.connect(seeks.append)
    s.mousePressEvent(_mouse(QEvent.MouseButtonPress, 0))
    s.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, 0, Qt.NoButton))
    s.mousePressEvent(_mouse(QEvent.MouseButtonPress, 2000))
    assert seeks == pytest.approx([10.0, 13.0])


def test_the_wheel_zooms_within_its_limits(strip):
    s = strip
    spans = []
    s.spanChanged.connect(spans.append)
    s.wheelEvent(_wheel(1))
    assert s.span == pytest.approx(8.0 / 1.25) and spans == [s.span]
    s.wheelEvent(_wheel(-2))
    assert s.span == pytest.approx(8.0 * 1.25)
    for _ in range(40):
        s.wheelEvent(_wheel(1))
    assert s.span == waveform.MIN_SPAN
    for _ in range(60):
        s.wheelEvent(_wheel(-1))
    assert s.span == waveform.MAX_SPAN
    n = len(spans)
    s.wheelEvent(_wheel(-1))   # at the limit: no change reported
    assert len(spans) == n


def test_painting_every_state_and_the_image_is_kept_while_the_window_stays(strip):
    s, tl = strip, strip.timeline
    tl.set_marks(48.0, 51.0)
    tl.set_scene_cuts([47.5, 49.0, 80.0])
    assert not s.grab().isNull()
    img = s._image
    assert img is not None and img.width() == s._columns()
    s.grab()
    assert s._image is img      # nothing changed: not built again
    tl.set_selection((49.0, 52.0))
    s.grab()
    assert s._image is img      # the overlays are drawn on top, the sound stays
    tl.set_position(50.5)
    s.grab()
    assert s._image is not img
    for state, msg in ((waveform.READING, "Reading the sound… 42 %"), (waveform.NO_SOUND, "No sound"),
                       (waveform.FAILED, "The sound could not be read: x"), (waveform.IDLE, ""),
                       (waveform.EMPTY, "")):
        s.set_state(state, msg)
        assert not s.grab().isNull()
    assert not s.isEnabled()
    s.set_progress(0.42)
    assert s.message == "Reading the sound… 42 %"


def test_a_frame_stamped_far_off_does_not_stretch_the_envelope():
    """An MPEG-TS clock wrap or a broken join can stamp a frame a day later; it used to make the
    envelope cover the whole gap (19 M buckets, over a GB while reading)."""
    rate = 8000
    loud = np.full((2, 800), 0.5, dtype=np.float32)
    frames = [_frame(loud, "fltp", rate, 0.0), _frame(loud, "fltp", rate, 0.1), _frame(loud, "fltp", rate, 95443.0)]
    for dur in (10.0, 0.0):    # length known, and unknown
        env = envelope_of(list(frames), 0.0, dur)
        assert env.end < 1.0 and loudest(env, 0.0, 0.3) == pytest.approx(0.5, abs=1e-3)


def test_a_drag_cut_short_by_closing_or_hiding_ends(strip):
    s, tl = strip, strip.timeline
    ended = []
    s.scrubFinished.connect(lambda: ended.append(1))
    for cut in (lambda: s.set_state(waveform.EMPTY), lambda: s.hide()):
        s.show()
        s.set_envelope(Envelope(0, np.full(100 * RATE, 0.3, np.float32), np.full(100 * RATE, 0.1, np.float32)))
        tl.set_position(50.0)
        s.mousePressEvent(_mouse(QEvent.MouseButtonPress, 400))
        s.mouseMoveEvent(_mouse(QEvent.MouseMove, 300))
        assert s.scrubbing
        cut()
        assert not s.scrubbing and s.view_span()[0] == pytest.approx(46.0)
    assert len(ended) == 2


def test_tick_labels():
    lab = waveform._label
    assert lab(0, 1) == "0:00" and lab(65, 5) == "1:05" and lab(3725, 60) == "1:02:05"
    assert lab(1.5, 0.5) == "0:01.5" and lab(1.25, 0.05) == "0:01.25" and lab(-2, 1) == "-0:02"
    assert lab(5.25, 0.25) == "0:05.25" and lab(5.5, 0.25) == "0:05.50" and lab(0.3, 0.1) == "0:00.3"


def test_paint_is_cheap_during_playback(strip):
    """Every frame played moves the window: the strip is drawn again each time."""
    s, tl = strip, strip.timeline
    tl.set_marks(48.0, 51.0)
    from PySide6.QtGui import QPixmap
    pm = QPixmap(s.size())
    n = 60
    t0 = time.perf_counter()
    for i in range(n):
        tl.set_position(50.0 + i / 30)
        s.render(pm)
    per = (time.perf_counter() - t0) / n
    assert per < 0.01, per


# ------------------------------------------------------------------ in the app
@pytest.fixture
def window():
    from simplecut.app import MainWindow
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()
    APP.processEvents()


def _open(w, path):
    before = w._video_token
    w.open_file(path)
    assert _wait(lambda: w._video_token != before and w.info is not None and w.dec is not None
                 and w.dec.current is not None)


def test_no_two_keys_of_the_window_are_the_same(window):
    from PySide6.QtGui import QAction, QShortcut
    keys = [a.shortcut().toString() for a in window.findChildren(QAction) if not a.shortcut().isEmpty()]
    keys += [s.key().toString() for s in window.findChildren(QShortcut)]
    assert "W" in keys
    assert len(keys) == len(set(keys)), sorted(k for k in keys if keys.count(k) > 1)


@needs_ffmpeg
def test_opening_a_video_reads_its_sound_and_keeps_it_for_next_time(window, video):
    w = window
    assert w.wave.state == waveform.EMPTY and not w.wave.isEnabled()
    _open(w, video)
    assert w._wave_thread is not None or w.wave.state == waveform.READY
    assert _wait(lambda: w.wave.state == waveform.READY, 60)
    assert _wait(lambda: not w._wave_threads)
    check_bursts(w.wave.envelope)
    assert os.path.exists(waveform.waves_path(video))
    assert not w.wave.isHidden() and w.a_wave.isChecked()
    w.close_video()
    assert w.wave.state == waveform.EMPTY and w.wave.envelope is None
    _open(w, video)
    assert w.wave.state == waveform.READY and w._wave_thread is None and not w._wave_threads


@needs_ffmpeg
def test_a_video_without_sound_says_so_without_reading(window, media):
    w = window
    _open(w, media["mute"])
    assert w.wave.state == waveform.NO_SOUND and "No sound" in w.wave.message
    assert w._wave_thread is None and not w._wave_threads


@needs_ffmpeg
def test_hidden_the_strip_reads_nothing_until_shown(window, video):
    w = window
    w.show_waveform(False)
    assert w.wave.isHidden() and not w.a_wave.isChecked()
    assert settings().value("show_waveform", True, type=bool) is False
    _open(w, video)
    assert w._wave_thread is None and not w._wave_threads and w.wave.state == waveform.IDLE
    w.a_wave.setChecked(True)   # (the menu item, W)
    assert not w.wave.isHidden() and w.wave_shown
    assert _wait(lambda: w.wave.state == waveform.READY, 60)


@needs_ffmpeg
def test_the_setting_is_remembered(video):
    from simplecut.app import MainWindow
    settings().setValue("show_waveform", False)
    settings().setValue("waveform_span", 20.0)
    w = MainWindow()
    try:
        assert w.wave.isHidden() and not w.a_wave.isChecked() and w.wave.span == 20.0
        _open(w, video)
        assert not w._wave_threads
        w.wave.wheelEvent(_wheel(1))
        assert settings().value("waveform_span", type=float) == pytest.approx(16.0)
    finally:
        w.close()
        w.deleteLater()
        APP.processEvents()


@needs_ffmpeg
def test_hiding_the_strip_stops_a_read(window, video):
    w = window
    _open(w, video)
    t = w._wave_thread
    if t is None:
        pytest.skip("the read finished before it could be stopped")
    w.show_waveform(False)
    assert t.cancelled and w._wave_thread is None and w.wave.state == waveform.IDLE
    assert _wait(lambda: not w._wave_threads)
    w._wave_read(Envelope(0, np.ones(5, np.float32), np.ones(5, np.float32)), t.token)   # its late result
    assert w.wave.state == waveform.IDLE


@needs_ffmpeg
def test_a_read_of_a_video_that_was_closed_or_replaced_is_dropped(window, video, media):
    w = window
    _open(w, video)
    t = w._wave_thread
    token = t.token if t is not None else -1
    w.close_video()
    assert t is None or t.cancelled
    assert _wait(lambda: not w._wave_threads)
    _open(w, media["mute"])
    late = Envelope(0, np.ones(5, np.float32), np.ones(5, np.float32))
    w._wave_read(late, token)
    w._wave_failed("late\nerror", token)
    w._wave_progress(0.5, token)
    assert w.wave.state == waveform.NO_SOUND and w.wave.envelope is None


@needs_ffmpeg
def test_a_stopped_reads_late_result_does_not_replace_the_new_one(window, video):
    w = window
    _open(w, video)
    old = w._wave_thread
    if old is None:
        pytest.skip("the read finished before it could be stopped")
    w.show_waveform(False)
    # (a read that ended just before it was stopped has stored its result, which showing the
    # strip again would rightly take: that one is gone here, so a new read starts)
    assert _wait(lambda: not old.isRunning())
    if os.path.exists(waveform.waves_path(video)):
        os.remove(waveform.waves_path(video))
    w.show_waveform(True)
    new = w._wave_thread
    assert new is not None and new.token != old.token
    w._wave_read(Envelope(0, np.ones(5, np.float32), np.ones(5, np.float32)), old.token)
    w._wave_failed("late\nerror", old.token)
    assert w._wave_thread is new and w.wave.state == waveform.READING
    assert _wait(lambda: w.wave.state == waveform.READY, 60)
    check_bursts(w.wave.envelope)


@needs_ffmpeg
def test_a_failed_read_says_why_and_never_raises(window, video):
    w = window
    _open(w, video)
    t = w._wave_thread
    if t is None:
        pytest.skip("the read finished first")
    w._wave_failed("Invalid data found when processing input\ntraceback...", t.token)
    t.cancel()
    assert w.wave.state == waveform.FAILED and "Invalid data found" in w.wave.message
    assert w._wave_thread is None
    assert not w.wave.grab().isNull()
    assert _wait(lambda: not w._wave_threads)


@needs_ffmpeg
def test_the_app_closes_cleanly_while_a_read_runs(video):
    from simplecut.app import MainWindow
    w = MainWindow()
    _open(w, video)
    threads = list(w._wave_threads)
    w.close()
    assert all(not t.isRunning() for t in threads)
    w.deleteLater()
    APP.processEvents()


@needs_ffmpeg
def test_clicking_the_strip_seeks_the_video(window, video):
    w = window
    _open(w, video)
    assert _wait(lambda: w.wave.state == waveform.READY, 60)
    w.wave.resize(816, waveform.HEIGHT)
    w.wave.set_span(4.0)
    a, span = w.wave.view_span()
    assert a == pytest.approx(w.info.start_time, abs=0.01)
    w.wave.mousePressEvent(_mouse(QEvent.MouseButtonPress, 8 + 400))     # the middle: 2 s in
    w.wave.mouseMoveEvent(_mouse(QEvent.MouseMove, 8 + 600))             # dragged to 3 s
    assert w._scrubbing()
    w.wave.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, 8 + 600, Qt.NoButton))
    assert not w._scrubbing()
    assert w.dec.time == pytest.approx(a + 3.0, abs=w.info.frame_duration)
    assert w.timeline.position == w.dec.time
