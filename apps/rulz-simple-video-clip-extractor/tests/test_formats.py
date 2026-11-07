"""Container- and codec-specific behaviour: seeks that land late (MPEG-TS, open-GOP HEVC),
lossless cuts per container, keyframes in AVI, preview colours and the colour-match LUT path.

Run with:  python -m pytest tests   (from the app folder, inside its .venv).
Tests that run ffmpeg/ffprobe are skipped when they are not on PATH.
"""
import os
import random
import subprocess
import sys

import av
import numpy as np
import pytest

from test_core import APP  # noqa: F401  (sets up Qt, the settings file and sys.path)

from simplecut import colormatch  # noqa: E402
from simplecut.decoder import Decoder, rgb_reformat_args  # noqa: E402
from simplecut.export import (ExportSettings, build_command, command_cwd, lossless_seek_nudge,  # noqa: E402
                              video_filters)
from simplecut.probe import VideoInfo, have_ffmpeg, probe  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")


def make_video(path, codec="libx264", n=75, fps=25, gop=12, size=(160, 96)):
    """Frame i is flat grey (3*i): its index can be read back from any decoded copy."""
    w, h = size
    with av.open(path, "w") as out:
        opts = {"x265-params": "log-level=error"} if codec == "libx265" else {}
        st = out.add_stream(codec, rate=fps, options=opts)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.codec_context.gop_size = gop
        for i in range(n):
            img = np.full((h, w, 3), (i * 3) % 256, dtype=np.uint8)
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    return path


def frame_indices(path):
    with av.open(path) as c:
        return [int(round(float(f.to_ndarray(format="gray").mean()) / 3)) for f in c.decode(video=0)]


def decoded(path):
    """(time, key_frame) of every frame, in display order."""
    with av.open(path) as c:
        return [(f.time, bool(f.key_frame)) for f in c.decode(video=0) if f.time is not None]


@pytest.fixture(scope="module")
def late_seek_videos(tmp_path_factory):
    """Files where a container seek lands after the target."""
    d = tmp_path_factory.mktemp("late")
    out = [make_video(str(d / "h264.ts"))]  # MPEG-TS: lands a GOP late
    if "libx265" in av.codecs_available:  # x265 default open GOP: lands on the next CRA
        out.append(make_video(str(d / "hevc.mp4"), codec="libx265", n=90, gop=15))
    return out


# ------------------------------------------------------------------ decoder: late seeks
def test_seek_lands_on_the_requested_frame(late_seek_videos):
    rnd = random.Random(3)
    for path in late_seek_videos:
        times = [t for t, _ in decoded(path)]
        order = list(range(len(times)))
        rnd.shuffle(order)  # jumps in every direction, so most need a real container seek
        dec = Decoder(path)
        try:
            for i in order:
                dec.seek(times[i])
                assert dec.time == pytest.approx(times[i]), (path, i)
        finally:
            dec.close()


def test_step_back_from_a_keyframe_goes_one_frame_back(late_seek_videos):
    for path in late_seek_videos:
        frames = decoded(path)
        times = [t for t, _ in frames]
        for i, (t, key) in enumerate(frames):
            if not key or i == 0:
                continue
            dec = Decoder(path)
            try:
                dec.seek(t)
                dec.step_back()  # the cache is empty right after a seek to a keyframe
                assert dec.time == pytest.approx(times[i - 1]), (path, i)
                dec.step_forward()
                assert dec.time == pytest.approx(t), (path, i)
            finally:
                dec.close()


def test_seek_last_finds_the_last_frame(late_seek_videos):
    for path in late_seek_videos:
        times = [t for t, _ in decoded(path)]
        dec = Decoder(path)
        try:
            dec.seek_last(times[-1] + 0.04)
            assert dec.time == pytest.approx(times[-1])
        finally:
            dec.close()


# ------------------------------------------------------------------ decoder: no needless seeks
def test_short_moves_reuse_decoded_frames(tmp_path):
    path = make_video(str(tmp_path / "gop.mp4"), gop=25)
    frames = decoded(path)
    times = [t for t, _ in frames]
    kfs = [t for t, k in frames if k]
    dec = Decoder(path, kfs)
    try:
        dec.seek(times[2])
        n = dec.restarts
        for i in (3, 5, 9, 9, 12, 7, 4, 4, 11):  # inside one GOP: forward walks and cache hits
            assert dec.seek(times[i]).time == pytest.approx(times[i])
        assert dec.restarts == n
        dec.seek(times[40])  # past the next keyframe: a real seek is cheaper
        assert dec.restarts == n + 1 and dec.time == pytest.approx(times[40])
    finally:
        dec.close()


def test_scrub_shows_keyframes_when_exact_seeks_are_slow(tmp_path):
    path = make_video(str(tmp_path / "scrub.mp4"), gop=25)
    frames = decoded(path)
    times = [t for t, _ in frames]
    kfs = [t for t, k in frames if k]
    dec = Decoder(path, kfs)
    try:
        f, exact = dec.scrub(times[37])  # fast video: exact
        assert exact and f.time == pytest.approx(times[37])
        def kf_before(t):
            return max(k for k in kfs if k <= t + 1e-6)

        assert kf_before(times[13]) < times[6] < kf_before(times[37]) == kf_before(times[40])
        dec.seek_ms = 1000.0  # pretend every exact seek is slow (4K)
        f, exact = dec.scrub(times[13])  # behind the cache, needs a seek: keyframe instead
        assert not exact and f.time == pytest.approx(kf_before(times[13]))
        n = dec.restarts
        f, exact = dec.scrub(times[11])  # forward inside that GOP: decoded on, exact
        assert exact and f.time == pytest.approx(times[11]) and dec.restarts == n
        f, exact = dec.scrub(times[6])  # back inside the frames just decoded: exact
        assert exact and f.time == pytest.approx(times[6]) and dec.restarts == n
        f, exact = dec.scrub(times[40])  # past a keyframe: that keyframe, one seek
        assert not exact and f.time == pytest.approx(kf_before(times[40])) and dec.restarts == n + 1
        assert dec.seek_keyframe(times[44]) is f and dec.restarts == n + 1  # already on screen
    finally:
        dec.close()


# ------------------------------------------------------------------ preview colours
def test_preview_uses_the_bt709_matrix_for_hd():
    y, u, v = 100, 80, 200  # a red that stays inside 0..255 with either matrix (limited range)
    planes = np.empty((1080, 1280), np.uint8)  # yuv420p layout: Y rows, then U and V (360x640 each)
    planes[:720], planes[720:900], planes[900:] = y, u, v

    def frame():  # a fresh frame per conversion: PyAV keeps the first conversion's settings per frame
        return av.VideoFrame.from_ndarray(planes, format="yuv420p")

    f = frame()
    assert f.height == 720 and f.colorspace in (0, 2)  # untagged
    rgb = f.reformat(format="rgb24", **rgb_reformat_args(f)).to_ndarray()[360, 640].astype(float)
    yy, cb, cr = 1.164 * (y - 16), u - 128, v - 128
    want = np.clip([yy + 1.793 * cr, yy - 0.213 * cb - 0.533 * cr, yy + 2.112 * cb], 0, 255)
    assert np.abs(rgb - want).max() <= 2, (rgb, want)
    old = frame().reformat(format="rgb24").to_ndarray()[360, 640].astype(float)  # plain = BT.601
    assert np.abs(old - want).max() > 10


# ------------------------------------------------------------------ export commands
def _info(**kw):
    base = dict(path="in.mkv", width=1920, height=1080, fps=25.0, duration=10.0, container="matroska,webm",
                has_b_frames=2, keyframes=[0.0, 2.0, 4.0, 6.0], keyframe_dts={0.0: -0.08, 2.0: 1.92,
                                                                                 4.0: 3.92, 6.0: 5.92})
    base.update(kw)
    return VideoInfo(**base)


def _opt(cmd, name, after_input=False):
    i = cmd.index("-i")
    rng = range(i + 1, len(cmd)) if after_input else range(0, i)
    return next((float(cmd[j + 1]) for j in rng if cmd[j] == name), None)


def test_lossless_seek_nudge_only_for_dts_seeking_formats_with_b_frames():
    assert lossless_seek_nudge(_info()) > 3 / 23
    assert lossless_seek_nudge(_info(container="flv")) > 0
    assert lossless_seek_nudge(_info(container="avi")) > 0
    assert lossless_seek_nudge(_info(has_b_frames=0)) == 0  # VP9 WebM
    assert lossless_seek_nudge(_info(container="mov,mp4,m4a,3gp,3g2,mj2")) == 0
    assert lossless_seek_nudge(_info(container="mpegts")) == 0


def test_lossless_from_the_file_start_does_not_seek():
    cmd = build_command(_info(), 0.0, 4.0, "out.mkv", ExportSettings(mode="lossless"))
    assert "-ss" not in cmd and _opt(cmd, "-t", after_input=True) == pytest.approx(3.92)
    cmd = build_command(_info(), 2.0, 4.0, "out.mkv", ExportSettings(mode="lossless"))
    nudge = lossless_seek_nudge(_info())
    assert _opt(cmd, "-ss") == pytest.approx(2.0 + nudge)
    assert _opt(cmd, "-t", after_input=True) == pytest.approx(1.92 - nudge)


def test_filters_crop_and_drop_frames_before_the_lut_and_scale():
    s = ExportSettings(mode="nvenc", rotation=90, hflip=True, crop=(10, 20, 300, 400), out_size=(150, 200),
                       lut_path="C:/x/match_0.cube", fps="24")
    names = [f.split("=")[0] for f in video_filters(_info(fps=60.0), s)]
    assert names == ["transpose", "hflip", "crop", "fps", "lut1d", "scale"]
    names = [f.split("=")[0] for f in video_filters(_info(fps=23.976), s)]  # raising the rate: fps last
    assert names == ["transpose", "hflip", "crop", "lut1d", "scale", "fps"]


@needs_ffmpeg
@pytest.mark.parametrize("ext", ["mkv", "flv", "avi", "mp4", "ts"])
def test_lossless_cut_starts_and_ends_on_the_marks(tmp_path, ext):
    path = make_video(str(tmp_path / f"src.{ext}"))
    info = probe(path)
    assert info.has_b_frames and len(info.keyframes) >= 5
    src_times = [t for t, _ in decoded(path)]
    src_idx = frame_indices(path)
    for a, b in ((2, 4), (0, 3)):
        start = info.keyframes[a] if a else info.start_time
        out = str(tmp_path / f"cut_{a}_{b}.{ext}")
        r = subprocess.run(build_command(info, start, info.keyframes[b], out, ExportSettings(mode="lossless")),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        want_first = src_idx[min(range(len(src_times)), key=lambda j: abs(src_times[j] - start))]
        want_end = src_idx[min(range(len(src_times)), key=lambda j: abs(src_times[j] - info.keyframes[b]))]
        got = frame_indices(out)
        assert (got[0], got[-1], len(got)) == (want_first, want_end - 1, want_end - want_first), (ext, a, b)


@needs_ffmpeg
def test_avi_keyframes_are_found_and_match_the_decoded_frames(tmp_path):
    path = make_video(str(tmp_path / "bf.avi"))
    info = probe(path)
    kf_frames = [t for t, k in decoded(path) if k]
    assert info.keyframes == pytest.approx(kf_frames)


@needs_ffmpeg
def test_lut_works_from_a_folder_with_an_apostrophe(tmp_path, monkeypatch):
    d = tmp_path / "O'Brien, [x]; y"
    d.mkdir()
    monkeypatch.setattr(colormatch, "temp_dir", lambda: str(d))
    lut = colormatch.write_cube_1d(np.tile(np.arange(256, dtype=np.uint8), (3, 1)))
    src = make_video(str(tmp_path / "in.mp4"), n=10)
    info = VideoInfo(path=src, width=160, height=96, fps=25.0, duration=0.4)
    s = ExportSettings(mode="nvenc", lut_path=lut, fps=None, audio="none")
    cmd = build_command(info, 0.0, 0.4, str(tmp_path / "out.mp4"), s)
    i, j = cmd.index("-c:v"), cmd.index("-pix_fmt")
    cmd = cmd[:i] + ["-c:v", "libx264", "-preset", "ultrafast"] + cmd[j:]  # no NVIDIA GPU needed
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=command_cwd(s))
    assert r.returncode == 0, r.stderr[-500:]
    assert command_cwd(s) == str(d) and command_cwd(ExportSettings(mode="lossless", lut_path=lut)) is None


@pytest.mark.skipif(sys.platform != "win32", reason="letter case only folds on Windows")
def test_same_file_ignores_letter_case(tmp_path):
    from simplecut.app import same_file
    a = tmp_path / "Clip.mp4"
    a.write_bytes(b"x")
    assert same_file(str(tmp_path / "clip.MP4"), str(a))
    assert same_file(os.path.join(str(tmp_path), "sub", "..", "CLIP.mp4"), str(a))
    assert not same_file(str(tmp_path / "clip2.mp4"), str(a))
