"""Distortion repair: the pipeline that applies saved adjustments, presets, the piped export
(ffmpeg -> Python -> ffmpeg) and the adjustments window.

Run with:  python -m pytest tests   (from the app folder, inside its .venv).
Exports use libx264 instead of NVENC and CPU decoding, so no NVIDIA GPU is needed.
"""
import json
import os
import subprocess
import sys
import time

import av
import numpy as np
import pytest

from test_core import APP  # noqa: F401  (sets up Qt, the settings file and sys.path)

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from simplecut.decoder import rgb_reformat_args  # noqa: E402
from simplecut.defaults import settings  # noqa: E402
from simplecut.export import (ExportJob, ExportSettings, build_command, build_piped_commands,  # noqa: E402
                              make_export_job)
from simplecut.probe import VideoInfo, have_ffmpeg, probe  # noqa: E402
from simplecut.repair.core.params import Corrections  # noqa: E402
from simplecut.repair.core.render import Renderer, StillImage  # noqa: E402
from simplecut.repair.export import PipedExportJob, RepairExport  # noqa: E402
from simplecut.repair.pipeline import (RepairPipeline, RepairSettings, RepairState,  # noqa: E402
                                       delta_orientation, from_json, orient, to_json)

needs_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")


def pattern(w, h):
    """Grid lines and a red patch: straight lines that every correction visibly bends."""
    img = np.full((h, w, 3), 90, np.uint8)
    img[::max(8, h // 12)] = 230
    img[:, ::max(8, w // 16)] = 230
    img[h // 8:h // 4, w // 8:w // 4] = (200, 40, 50)
    return img


def corrections():
    c = Corrections()
    c.distortion.enabled, c.distortion.intensity = True, 60
    c.horizon.enabled, c.horizon.angle = True, 4.0
    c.perspective.enabled, c.perspective.mode = True, "verticals"
    c.perspective.lines = [[[0.30, 0.10], [0.33, 0.90]], [[0.72, 0.10], [0.68, 0.90]]]
    return c


# ------------------------------------------------------------------ pipeline
def test_pipeline_matches_what_the_window_renders():
    img = pattern(640, 360)
    c = corrections()
    full = Renderer(StillImage(img)).render(c, full=True).rgba[..., :3]
    p = RepairPipeline(c, 640, 360)
    out = p.apply(img, full=True)
    assert (out.shape[1], out.shape[0]) == p.out_size
    assert p.out_size[0] % 2 == 0 and p.out_size[1] % 2 == 0 and p.out_size < (640, 360)
    h, w = out.shape[:2]
    assert np.array_equal(out, full[:h, :w])  # same remap tables, same crop
    assert not p.changes_nothing


def test_switched_on_modules_with_nothing_set_change_nothing():
    c = Corrections()
    c.perspective.enabled = True   # no lines drawn
    c.horizon.enabled = True       # angle 0
    assert RepairPipeline(c, 320, 180).changes_nothing
    c.horizon.angle = 1.0
    assert not RepairPipeline(c, 320, 180).changes_nothing


def test_later_rotation_turns_the_corrected_picture_as_a_whole():
    x = np.arange(12).reshape(3, 4)
    for base in ((0, False), (90, False), (180, True), (270, True)):
        for cur in ((0, False), (90, True), (180, False), (270, True), (90, False)):
            d = delta_orientation(*base, *cur)
            assert np.array_equal(orient(orient(x, *base), *d), orient(x, *cur))
    raw = pattern(320, 180)
    c = Corrections()
    c.crop.auto, c.crop.rect = False, [0.0, 0.0, 0.5, 1.0]   # keep the left half
    st = RepairState.build(RepairSettings(c), 320, 180, source_rotation=0)
    upright = st.screen_frame(raw, 0, 0, False, full=True)
    assert np.array_equal(upright, raw[:, :160])
    turned = st.screen_frame(raw, 0, 90, True, full=True)   # rotated and mirrored afterwards
    assert np.array_equal(turned, orient(upright, 90, True)) and st.output_size(90, True) == (180, 160)
    # made on a phone clip (source flag 90) that the user had turned once more: base = 180 in total
    st = RepairState.build(RepairSettings(c, base_rotation=90), 320, 180, source_rotation=90)
    assert np.array_equal(st.screen_frame(raw, 90, 90, False, full=True), orient(raw, 180, False)[:, :160])


def test_presets_round_trip_and_bad_files_are_refused():
    c = corrections()
    assert from_json(to_json(c)).to_dict() == c.to_dict()
    for text in ("not json", "{}", json.dumps({"kind": "something-else", "corrections": {}})):
        with pytest.raises(ValueError):
            from_json(text)
    d = json.loads(to_json(c))
    d["corrections"]["distortion"]["intensity"] = "lots"       # wrong type: default kept
    d["corrections"]["horizon"]["angle"] = float("nan")       # not finite: default kept
    d["corrections"]["orientation"] = {"rot90": 1}            # orientation is never taken from a file
    back = from_json(json.dumps(d))
    assert back.distortion.intensity == Corrections().distortion.intensity
    assert back.horizon.angle == 0.0 and back.orientation.rot90 == 0
    d["corrections"]["perspective"]["lines"] = [[1, 2]]       # malformed points
    with pytest.raises(ValueError):
        from_json(json.dumps(d))


# ------------------------------------------------------------------ export commands
def _info(**kw):
    base = dict(path="in.mp4", width=1920, height=1080, fps=30.0, duration=10.0, keyframes=[0.0, 1.0, 2.0, 3.0],
                has_audio=True, audio_codec="aac", rate="30/1")
    base.update(kw)
    return VideoInfo(**base)


def _vals(cmd, name):
    return [cmd[i + 1] for i, c in enumerate(cmd) if c == name]


def test_piped_commands_cut_like_the_nvenc_command():
    info = _info()
    s = ExportSettings(mode="nvenc", fps="24", crop=(10, 20, 600, 400), out_size=(300, 200),
                       lut_path="C:/x/match_0.cube")
    (dec_hw, enc), (dec, enc2) = build_piped_commands(info, 1.5, 3.0, "out.mp4", s, 90, True,
                                                      (1080, 1920), (1000, 1800))
    assert enc == enc2 and _vals(dec_hw, "-hwaccel") == ["cuda"] and "-hwaccel" not in dec
    ref = build_command(info, 1.5, 3.0, "out.mp4", s)
    assert _vals(dec, "-ss") == _vals(ref, "-ss") and _vals(dec, "-t") == _vals(ref, "-t")
    vf = dec[dec.index("-vf") + 1].split(",")
    assert [f.split("=")[0] for f in vf] == ["transpose", "hflip", "fps", "scale", "format"]
    assert vf[3].startswith("scale=1080:1920:") and "in_color_matrix=bt709:in_range=limited" in vf[3]
    assert dec[-4:] == ["rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    assert _vals(enc, "-s") == ["1000x1800"] and _vals(enc, "-framerate") == ["24"]
    assert _vals(enc, "-i")[0] == "pipe:0" and float(_vals(enc, "-itsoffset")[0]) == pytest.approx(0.5005)
    ef = enc[enc.index("-vf") + 1]
    assert [f.split("=")[0] for f in ef.split(",")] == ["crop", "lut1d", "scale", "format"]
    assert "scale=300:200:flags=lanczos+accurate_rnd" in ef and "out_color_matrix=bt709:out_range=limited" in ef
    assert "-colorspace" not in enc            # untagged source: untagged clip, like the plain export
    assert _vals(enc, "-c:a") == ["copy"] and "1:a:0?" in enc


def test_piped_commands_follow_the_source_colour_tags():
    s = ExportSettings(mode="nvenc", audio="none")
    tagged = _info(height=480, color_space="bt709", color_primaries="bt709", color_transfer="bt709")
    _, (dec, enc) = build_piped_commands(tagged, 0.0, 1.0, "o.mp4", s, 0, False, (1920, 480), (1920, 480))
    assert "in_color_matrix=bt709" in dec[dec.index("-vf") + 1]
    assert _vals(enc, "-colorspace") == ["bt709"] and _vals(enc, "-color_trc") == ["bt709"]
    assert "-an" in enc and "-itsoffset" not in enc and _vals(enc, "-i") == ["pipe:0"]
    _, (dec, enc) = build_piped_commands(_info(height=480, color_range="pc"), 0.0, 1.0, "o.mp4", s, 0, False,
                                         (1920, 480), (1920, 480))
    assert "in_color_matrix=bt601:in_range=full" in dec[dec.index("-vf") + 1]   # untagged SD: BT.601


def _repair_export(w, h, c=None, base=(0, False), cur=(0, False)):
    st = RepairState.build(RepairSettings(c or Corrections(), *base), w, h, 0)
    return RepairExport(st, cur[0], cur[1], 0, (w, h))


def test_make_export_job_runs_python_only_for_repaired_reencodes():
    info = _info()
    assert isinstance(make_export_job(info, 1.0, 2.0, "o.mp4", ExportSettings(mode="nvenc")), ExportJob)
    rx = _repair_export(1920, 1080, corrections())
    job = make_export_job(info, 1.0, 2.0, "o.mp4", ExportSettings(mode="nvenc", repair=rx))
    assert isinstance(job, PipedExportJob) and job.expected == 24
    assert isinstance(make_export_job(info, 1.0, 2.0, "o.mp4", ExportSettings(mode="lossless", repair=rx)),
                      ExportJob)


# ------------------------------------------------------------------ the piped job
def _wait(job, got, timeout=30):
    end = time.time() + timeout
    while not got and time.time() < end:
        APP.processEvents()
        time.sleep(0.01)


def test_piped_job_retries_on_the_cpu_when_gpu_decoding_fails_first(tmp_path):
    py = sys.executable
    w, h, n = 8, 4, 5
    rx = _repair_export(w, h)
    job = PipedExportJob(_info(width=w, height=h), 0.0, 1.0, "unused.mp4", ExportSettings(mode="nvenc", repair=rx))
    counted = tmp_path / "bytes.txt"
    enc = [py, "-c", f"import sys; open(r'{counted}', 'w').write(str(len(sys.stdin.buffer.read())))"]
    job.attempts = [([py, "-c", "import sys; sys.exit(3)"], enc),
                    ([py, "-c", f"import sys; sys.stdout.buffer.write(bytes({w * h * 3 * n}))"], enc)]
    calls = []
    real = rx.process
    rx.process = lambda a: (calls.append(1), real(a))[1]
    got = []
    job.finished.connect(lambda ok, log: got.append((ok, log)))
    job.start()
    _wait(job, got)
    assert got and got[0][0] is True, got
    assert "CPU" in got[0][1] and job.frames == n and int(counted.read_text()) == w * h * 3 * n
    assert len(calls) == 1   # identical frames (a raised frame rate) are corrected once


def test_cancelled_piped_job_stops_both_processes_and_does_not_retry():
    py = sys.executable
    rx = _repair_export(8, 4)
    job = PipedExportJob(_info(width=8, height=4), 0.0, 100.0, "unused.mp4",
                         ExportSettings(mode="nvenc", repair=rx))
    dec = [py, "-c", "import sys, time\nwhile True:\n    sys.stdout.buffer.write(bytes(96)); "
                     "sys.stdout.flush(); time.sleep(0.02)"]
    enc = [py, "-c", "import sys; sys.stdin.buffer.read()"]
    job.attempts = [(dec, enc), (dec + ["second"], enc)]
    got = []
    job.finished.connect(lambda ok, log: got.append(ok))
    job.start()
    time.sleep(0.6)
    job.cancel()
    assert job.wait(10000)
    assert got == [False] and job.cancelled and job.frames > 0 and job.cmd[0] == dec


# ------------------------------------------------------------------ real exports
def make_source(path, size="1280x720", seconds=1.2, fps=25, gop=10):
    """ffmpeg's test pattern (saturated colours, a moving element) with a tone, untagged."""
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                        "-t", str(seconds), "-c:v", "libx264", "-preset", "veryfast", "-crf", "8", "-g", str(gop),
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    return path


def decode_rgb(path):
    """Every frame as the preview converts it (the player's matrix)."""
    with av.open(path) as c:
        return [f.reformat(format="rgb24", **rgb_reformat_args(f)).to_ndarray() for f in c.decode(video=0)]


def run_cpu(job):
    """Run the job's CPU-decoding attempt with libx264 standing in for NVENC."""
    dec, enc = job.attempts[1]
    i = enc.index("-c:v")
    j = enc.index("-pix_fmt", i)   # the output's -pix_fmt (the raw input has one too)
    enc = enc[:i] + ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "6"] + enc[j:]
    job.attempts = [(dec, enc)]
    got = []
    job.finished.connect(lambda ok, log: got.append((ok, log)))
    job.start()
    _wait(job, got, timeout=120)
    assert got and got[0][0], got
    return got[0]


@needs_ffmpeg
def test_repaired_export_matches_the_preview_frame_for_frame(tmp_path):
    src = make_source(str(tmp_path / "src.mp4"))
    info = probe(src)
    frames = decode_rgb(src)
    c = corrections()
    rx = _repair_export(info.width, info.height, c)
    fd = 1 / info.fps
    start, end = info.start_time + 13 * fd, info.start_time + 23 * fd   # not on a keyframe
    out = str(tmp_path / "out.mp4")
    run_cpu(PipedExportJob(info, start, end, out, ExportSettings(mode="nvenc", fps=None, repair=rx)))
    got = decode_rgb(out)
    assert len(got) == 10 and (got[0].shape[1], got[0].shape[0]) == rx.out_size
    want = [rx.state.pipeline.apply(f, full=True).astype(np.int16) for f in frames]
    for k, g in enumerate(got):
        g = g.astype(np.int16)
        diffs = [float(np.abs(g - wnt).mean()) for wnt in want]
        assert int(np.argmin(diffs)) == 13 + k, (k, diffs)   # the right frame...
        # ...with the same colours: a wrong YUV matrix is ~14 levels off, a colour cast shows as a bias
        assert diffs[13 + k] < 3.0, diffs[13 + k]
        bias = (g - want[13 + k]).reshape(-1, 3).mean(axis=0)
        assert np.abs(bias).max() < 1.0, bias
    a = probe(out)
    assert a.has_audio and a.duration == pytest.approx(10 * fd, abs=0.05)


@needs_ffmpeg
def test_repaired_export_after_rotate_flip_crop_and_scale(tmp_path):
    src = make_source(str(tmp_path / "src.mp4"), size="320x240", seconds=0.8)
    info = probe(src)
    frames = decode_rgb(src)
    rx = _repair_export(info.width, info.height, corrections(), base=(90, False), cur=(90, True))
    sw, sh = rx.out_size
    assert sh > sw                                            # portrait: turned 90 degrees
    crop = (4, 10, sw // 2 & ~1, sh // 2 & ~1)
    s = ExportSettings(mode="nvenc", fps=None, audio="none", crop=crop, out_size=(crop[2] * 2, crop[3] * 2))
    s.repair = rx
    out = str(tmp_path / "out.mp4")
    run_cpu(PipedExportJob(info, info.start_time, info.end_time, out, s))
    got = decode_rgb(out)
    assert len(got) == len(frames) and (got[0].shape[1], got[0].shape[0]) == s.out_size
    x, y, w, h = crop
    for k in (0, len(frames) // 2):
        screen = rx.state.screen_frame(frames[k], 0, 90, True, full=True)[y:y + h, x:x + w]
        small = got[k][::2, ::2].astype(np.int16)              # undo the 2x scale roughly
        assert float(np.abs(small - screen.astype(np.int16)).mean()) < 12


# ------------------------------------------------------------------ the window
def test_window_enter_applies_a_tool_escape_cancels_it_and_save_remembers():
    from simplecut.repair.window import LAST_KEY, AdjustmentsWindow, last_saved
    win = AdjustmentsWindow(pattern(640, 360), None, "a note")
    try:
        win.show()
        win.toggle_tool("crop", "manual")
        QTest.keyClick(win, Qt.Key_Escape)
        assert win.tool is None and win.isVisible() and not win.c.perspective.enabled
        win.toggle_tool("perspective", "verticals")
        QTest.keyClick(win, Qt.Key_Return)
        assert win.tool is None and win.c.perspective.enabled and len(win.c.perspective.lines) == 2
        win.undo.undo()
        assert not win.c.perspective.enabled
        win.undo.redo()
        win.save()
        assert win.result is not None and win.result.perspective.enabled
        assert from_json(settings().value(LAST_KEY, "", type=str)).to_dict() == win.result.to_dict()
        assert last_saved().to_dict() == win.result.to_dict()
    finally:
        win.deleteLater()


def test_main_window_previews_saves_frames_and_rotates_with_the_repair(tmp_path, videos):
    from simplecut.app import MainWindow
    settings().setValue("output_dir", str(tmp_path))
    w = MainWindow()
    try:
        w.open_file(videos[0])
        end = time.time() + 30
        while (w.info is None or w.dec is None or w.dec.current is None) and time.time() < end:
            APP.processEvents()
            time.sleep(0.01)
        assert w.info is not None
        size = (w.info.width, w.info.height)
        w._set_repair(RepairSettings(corrections()))
        assert w.repair is not None and w.repair_label.text() and w.b_repair_clear.isEnabled()
        assert w.view.frame_size == w.repair.pipeline.out_size != size
        w.mark_in, w.mark_out = w.info.start_time, w.info.end_time   # both ends: lossless would be fine
        assert not w._lossless_possible()
        w.save_frame()
        (png,) = [p for p in os.listdir(tmp_path) if p.endswith(".png")]
        from PySide6.QtGui import QImage
        img = QImage(str(tmp_path / png))
        assert "_repair" in png and (img.width(), img.height()) == w.repair.pipeline.out_size
        w.rotate()
        assert w.view.frame_size == w.repair.pipeline.out_size[::-1]
        assert w._repair_export().out_size == w.view.frame_size
        w.clear_repair()
        assert w.repair is None and w.view.frame_size == size[::-1]
    finally:
        settings().remove("output_dir")
        w.close()
        w.deleteLater()


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    from test_core import make_video
    d = tmp_path_factory.mktemp("repairvid")
    return [make_video(str(d / "mpeg4.mp4"))]
