"""Scene-cut detection: the scores and how cuts are picked from them, storage, and the app's
Scenes commands (find, jump, select, add to the clip list)."""
from __future__ import annotations

import os
import shutil
import subprocess
import time

import numpy as np
import pytest

from test_core import APP  # noqa: F401  (sets up Qt, the settings file and sys.path)

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from simplecut import scenes  # noqa: E402
from simplecut.cliplist import store_dir  # noqa: E402
from simplecut.probe import have_ffmpeg  # noqa: E402
from simplecut.scenes import SceneScores, motion_score, pick_cuts, scenes_between  # noqa: E402
from simplecut.timeline import Timeline  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg / ffprobe not on PATH")

FPS = 30
# (source, frames): hard cuts at frames 45, 90, 127 and 172; a one-frame white flash at 108
# inside the colour bars; a steady pan from 127 to 171 (every frame differs, no cut inside)
SEGMENTS = [
    ("testsrc2=s=320x180:r=30", 45),
    ("mandelbrot=s=320x180:r=30", 45),
    ("smptebars=s=320x180:r=30", 18),
    ("color=white:s=320x180:r=30", 1),
    ("smptebars=s=320x180:r=30", 18),
    ("testsrc2=s=640x180:r=30", 45),     # cropped below into a pan
    ("testsrc=s=320x180:r=30", 45),
]
CUT_FRAMES = [45, 90, 127, 172]
TOTAL_FRAMES = sum(n for _, n in SEGMENTS)


def _wait(cond, timeout=30):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        APP.processEvents()
        time.sleep(0.01)
    return cond()


@pytest.fixture(autouse=True)
def empty_store():
    """No stored clip lists or scans from an earlier test (they sit beside the settings file)."""
    for d in (store_dir(), scenes.scores_dir()):
        shutil.rmtree(d, ignore_errors=True)
    yield
    for d in (store_dir(), scenes.scores_dir()):
        shutil.rmtree(d, ignore_errors=True)


def make_cuts_video(path: str, extra: list[str] | None = None) -> str:
    args = ["ffmpeg", "-v", "error", "-y"]
    for src, _ in SEGMENTS:
        args += ["-f", "lavfi", "-i", src]
    parts = []
    for i, (src, n) in enumerate(SEGMENTS):
        pan = "crop=320:180:x='min(n*6,320)':y=0," if src.startswith("testsrc2=s=640") else ""
        parts.append(f"[{i}:v]{pan}trim=end_frame={n},setpts=PTS-STARTPTS,format=yuv420p,setsar=1[v{i}]")
    graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(SEGMENTS))) + \
        f"concat=n={len(SEGMENTS)}:v=1:a=0[out]"
    args += ["-filter_complex", graph, "-map", "[out]", "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "18", "-g", "300", "-r", str(FPS), *(extra or []), path]
    subprocess.run(args, check=True, creationflags=0x08000000 if os.name == "nt" else 0)
    return path


@pytest.fixture(scope="module")
def cuts_video(tmp_path_factory):
    if not have_ffmpeg():
        pytest.skip("ffmpeg / ffprobe not on PATH")
    return make_cuts_video(str(tmp_path_factory.mktemp("scenes") / "cuts.mp4"))


@pytest.fixture
def video(cuts_video, tmp_path):
    """A copy of the cuts video of its own (stored scans are per file)."""
    p = str(tmp_path / "v.mp4")
    shutil.copyfile(cuts_video, p)
    return p


def _scores(times, score, settle=None) -> SceneScores:
    score = np.asarray(score, dtype=np.float32)
    return SceneScores(np.asarray(times, dtype=np.float64), score,
                       score.copy() if settle is None else np.asarray(settle, dtype=np.float32))


# ------------------------------------------------------------------ scores and picking
def test_steady_motion_scores_nothing_and_a_jump_scores_what_it_adds():
    m = np.array([0, 8, 8, 8, 8, 8, 8, 40, 8, 8], dtype=np.float32)
    s = motion_score(m)
    assert s[0] == 0 and s[1] == 8   # (frame 1 has nothing before it to compare with)
    assert np.all(s[2:7] == 0)       # every frame differs by the same amount: a pan, not a cut
    assert s[7] == pytest.approx(32)  # the jump, less the typical difference before it
    assert s[8] == 0
    for n in (0, 1, 2):
        assert len(motion_score(np.zeros(n, dtype=np.float32))) == n


def test_cuts_need_the_score_and_a_picture_that_stays_changed():
    t = np.arange(100) / 30
    score = np.zeros(100)
    settle = np.zeros(100)
    score[40], settle[40] = 30, 30   # a cut
    score[70], settle[70] = 30, 2    # a flash: the picture is back two frames later
    score[80], settle[80] = 8, 30    # too small a change for Normal, enough for High
    s = _scores(t, score, settle)
    assert s.cuts("normal") == pytest.approx([t[40]])
    assert s.cuts("high") == pytest.approx([t[40], t[80]])
    assert s.cuts("low") == pytest.approx([t[40]])
    assert s.cuts("nonsense") == s.cuts("normal")


def test_of_two_cuts_closer_than_the_shortest_scene_the_stronger_stays():
    t = np.arange(100) / 30
    score = np.zeros(100)
    score[3] = 50                 # within MIN_SCENE of the start: not a scene of its own
    score[30], score[36] = 20, 40  # 0.2 s apart
    score[60], score[64] = 40, 20
    assert pick_cuts(_scores(t, score), 10) == pytest.approx([t[36], t[60]])


def test_no_cut_in_the_last_half_second():
    """A broken or black last frame is not a scene of its own."""
    t = np.arange(100) / 30
    score = np.zeros(100)
    score[50], score[99], score[90] = 30, 60, 40
    assert pick_cuts(_scores(t, score), 10) == pytest.approx([t[50]])


def test_the_strongest_cuts_are_taken_first():
    """Candidates 0.4 s apart with rising scores: taking each stronger one in turn would keep
    only the last; the strongest first keeps every other one."""
    t = np.arange(150) / 30
    score = np.zeros(150)
    for k, i in enumerate((30, 42, 54, 66)):
        score[i] = 20 + k
    assert pick_cuts(_scores(t, score), 10) == pytest.approx([t[42], t[66]])


def test_scenes_between_covers_the_range_without_gaps():
    assert scenes_between([2.0, 4.0, 6.0], 0.0, 8.0) == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]
    assert scenes_between([2.0, 4.0, 6.0], 3.0, 5.0) == [(3.0, 4.0), (4.0, 5.0)]
    assert scenes_between([2.0], 2.0, 3.0) == [(2.0, 3.0)]
    assert scenes_between([], 1.0, 2.0) == [(1.0, 2.0)]


# ------------------------------------------------------------------ the scan
@needs_ffmpeg
def test_the_scan_finds_the_hard_cuts_but_not_a_flash_or_a_pan(cuts_video):
    s = scenes.scan(cuts_video)
    assert len(s) == TOTAL_FRAMES
    assert np.all(np.diff(s.times) > 0)
    expected = [s.times[0] + f / FPS for f in CUT_FRAMES]
    for sens in ("normal", "high"):
        assert s.cuts(sens) == pytest.approx(expected, abs=0.5 / FPS), sens
    flash = s.times[0] + 108 / FPS
    i = int(np.argmin(np.abs(s.times - flash)))
    assert s.score[i] > 30 and s.settle[i] < 5          # the flash itself changes a lot, briefly
    assert s.score[i + 1] > 30 and s.settle[i + 1] < 5  # and so does the frame after it


@needs_ffmpeg
def test_the_scan_reports_progress_and_can_be_stopped(cuts_video):
    seen = []
    scenes.scan(cuts_video, progress=seen.append)
    assert seen and all(0.0 <= f <= 1.0 for f in seen)
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 10

    with pytest.raises(scenes.Cancelled):
        scenes.scan(cuts_video, cancelled=stop)


# ------------------------------------------------------------------ storage
@needs_ffmpeg
def test_a_bad_last_frame_is_not_a_cut(tmp_path):
    p = str(tmp_path / "glitch.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30",
                    "-f", "lavfi", "-i", "color=white:s=320x180:r=30", "-filter_complex",
                    "[0:v]trim=end_frame=90,setpts=PTS-STARTPTS,format=yuv420p[a];"
                    "[1:v]trim=end_frame=1,setpts=PTS-STARTPTS,format=yuv420p[b];[a][b]concat=n=2:v=1:a=0[out]",
                    "-map", "[out]", "-c:v", "libx264", "-preset", "ultrafast", p],
                   check=True, creationflags=0x08000000 if os.name == "nt" else 0)
    s = scenes.scan(p)
    assert s.score[-1] > 30          # the white frame differs a lot...
    assert s.cuts("high") == []      # ...but is not a scene


@needs_ffmpeg
def test_cut_times_are_the_frames_the_decoder_lands_on(tmp_path):
    """B-frames and a start time that is not 0: the cuts are the decoder's own frame times."""
    from simplecut.decoder import Decoder
    p = make_cuts_video(str(tmp_path / "bf.mp4"), ["-bf", "3", "-output_ts_offset", "1.37"])
    s = scenes.scan(p)
    cuts = s.cuts()
    assert len(cuts) == len(CUT_FRAMES) and s.times[0] > 1.3
    dec = Decoder(p)
    try:
        for c in cuts:
            assert dec.seek(c).time == c
    finally:
        dec.close()


@needs_ffmpeg
def test_a_fresh_scan_and_a_stored_one_pick_the_same_cuts(cuts_video, tmp_path):
    s = scenes.scan(cuts_video)
    # (the scores come rounded the way they are stored: a score just under a threshold can't
    # become one at it after reopening)
    assert np.array_equal(s.score, s.score.astype(np.float16).astype(np.float32))
    v = str(tmp_path / "v.mp4")
    shutil.copyfile(cuts_video, v)
    assert scenes.save_scores(v, s)
    back = scenes.load_scores(v)
    assert np.array_equal(back.score, s.score) and np.array_equal(back.settle, s.settle)


def test_a_file_that_changed_during_the_scan_is_not_stored(tmp_path):
    v = tmp_path / "a.mp4"
    v.write_bytes(b"x" * 100)
    stamp = (os.stat(v).st_size, os.stat(v).st_mtime)
    v.write_bytes(b"x" * 200)   # still being written
    assert not scenes.save_scores(str(v), _scores([0.0], [0.0]), stamp)
    assert not os.path.exists(scenes.scores_path(str(v)))


def test_stored_scores_come_back_for_the_same_file_only(tmp_path):
    v = tmp_path / "a.mp4"
    v.write_bytes(b"x" * 100)
    s = _scores(np.arange(10) / 30, np.linspace(0, 50, 10), np.linspace(50, 0, 10))
    assert scenes.load_scores(str(v)) is None
    assert scenes.save_scores(str(v), s)
    back = scenes.load_scores(str(v))
    assert back is not None
    assert np.array_equal(back.times, s.times)
    assert np.allclose(back.score, s.score, atol=0.05) and np.allclose(back.settle, s.settle, atol=0.05)
    v.write_bytes(b"y" * 101)   # another file of that name
    assert scenes.load_scores(str(v)) is None


@pytest.mark.parametrize("damage", ["garbage", "version", "lengths", "nan", "no-meta"])
def test_a_damaged_or_foreign_store_file_means_scan_again(tmp_path, damage):
    v = tmp_path / "a.mp4"
    v.write_bytes(b"x" * 100)
    s = _scores(np.arange(10) / 30, np.ones(10))
    assert scenes.save_scores(str(v), s)
    path = scenes.scores_path(str(v))
    st = os.stat(v)
    meta = {"version": scenes.SCORE_VERSION, "path": str(v), "size": st.st_size, "mtime": st.st_mtime}
    arrays = {"times": s.times, "score": s.score, "settle": s.settle}
    if damage == "garbage":
        with open(path, "wb") as f:
            f.write(b"not a zip")
    else:
        if damage == "version":
            meta["version"] = 999
        elif damage == "lengths":
            arrays["score"] = arrays["score"][:5]
        elif damage == "nan":
            arrays["times"] = arrays["times"].copy()
            arrays["times"][3] = np.nan
        import json
        extra = {} if damage == "no-meta" else {"meta": np.array(json.dumps(meta))}
        with open(path, "wb") as f:
            np.savez(f, **arrays, **extra)
    assert scenes.load_scores(str(v)) is None


def test_old_scans_of_missing_videos_are_cleaned_up(tmp_path):
    kept, gone = tmp_path / "kept.mp4", tmp_path / "gone.mp4"
    for v in (kept, gone):
        v.write_bytes(b"x" * 10)
        assert scenes.save_scores(str(v), _scores([0.0, 0.1], [0, 0]))
    os.remove(gone)
    d = scenes.scores_dir()
    tmp = os.path.join(d, "abc.123.tmp.npz")
    with open(tmp, "wb") as f:
        f.write(b"half")
    scenes.clean_scores(now=time.time() + 3600)          # recent: everything stays
    assert len(os.listdir(d)) == 3
    scenes.clean_scores(now=time.time() + (scenes.STALE_DAYS + 1) * 86400)
    assert os.listdir(d) == [os.path.basename(scenes.scores_path(str(kept)))]


def test_the_timeline_draws_the_cuts():
    tl = Timeline()
    tl.resize(400, 56)
    tl.set_range(0.0, 10.0)
    tl.set_scene_cuts([2.0, 2.001, 5.0])   # (two in one pixel draw once)
    assert tl.scene_cuts == [2.0, 2.001, 5.0]
    assert not tl.grab().isNull()


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
    assert _wait(lambda: bool(w.info.keyframes))


def _find(w):
    w.find_scene_cuts()
    assert w.sc_progress.isVisibleTo(w) or w.scene_scores is not None
    assert _wait(lambda: w.scene_scores is not None, 60)
    assert _wait(lambda: not w._scene_threads)


def _cut_times(w):
    return [w.info.start_time + f / FPS for f in CUT_FRAMES]


@needs_ffmpeg
def test_find_marks_the_cuts_and_keeps_them_for_next_time(window, video):
    w = window
    _open(w, video)
    assert w.scene_scores is None and not w.b_next_scene.isEnabled()
    _find(w)
    assert w.scene_cuts == pytest.approx(_cut_times(w), abs=0.5 / FPS)
    assert w.timeline.scene_cuts == w.scene_cuts
    assert w.b_next_scene.isEnabled() and w.b_prev_scene.isEnabled()
    assert not w.sc_progress.isVisibleTo(w)
    assert os.path.exists(scenes.scores_path(video))
    # pressed again: no second search
    w.find_scene_cuts()
    assert w._scene_thread is None and not w._scene_threads
    # reopened: back at once, without reading the video again
    w.close_video()
    assert w.scene_scores is None and w.timeline.scene_cuts == []
    _open(w, video)
    assert w.scene_cuts == pytest.approx(_cut_times(w), abs=0.5 / FPS)
    assert not w._scene_threads


@needs_ffmpeg
def test_sensitivity_picks_again_without_reading_the_video(window, video):
    w = window
    _open(w, video)
    _find(w)
    n = len(w.scene_cuts)
    s = w.scene_scores
    s.score[25] = 7.0        # a change only High takes (and the picture stays changed)
    s.settle[25] = 7.0
    w.set_scene_sensitivity("high")
    assert len(w.scene_cuts) == n + 1 and not w._scene_threads
    assert w.sens_actions["high"].isChecked()
    w.set_scene_sensitivity("normal")
    assert len(w.scene_cuts) == n


@needs_ffmpeg
def test_jumping_between_cuts_and_selecting_a_scene(window, video):
    w = window
    _open(w, video)
    _find(w)
    cuts = w.scene_cuts
    w.next_scene()
    assert w.dec.time == pytest.approx(cuts[0], abs=0.5 / FPS)
    w.next_scene()
    assert w.dec.time == pytest.approx(cuts[1], abs=0.5 / FPS)
    w.step_forward()
    w.prev_scene()
    assert w.dec.time == pytest.approx(cuts[1], abs=0.5 / FPS)
    w.prev_scene()
    assert w.dec.time == pytest.approx(cuts[0], abs=0.5 / FPS)
    w.prev_scene()
    assert w.dec.time == pytest.approx(w.info.start_time, abs=0.5 / FPS)
    # the scene under the playhead: In on its first frame, Out where the next starts
    w.seek_to(cuts[1] + 10 / FPS)
    w.select_scene()
    assert (w.mark_in, w.mark_out) == pytest.approx((cuts[1], cuts[2]), abs=0.5 / FPS)
    w.go_end()
    w.select_scene()
    assert w.mark_in == pytest.approx(cuts[-1], abs=0.5 / FPS) and w.mark_out == w.info.end_time
    w.next_scene()   # nothing after the last cut: stays
    assert w.dec.time > cuts[-1]


@needs_ffmpeg
def test_in_fixed_length_the_clip_box_moves_to_the_cuts(window, video):
    w = window
    _open(w, video)
    _find(w)
    cuts = w.scene_cuts
    w.b_fixed.setChecked(True)
    w.next_scene()
    assert w.sel_start == pytest.approx(cuts[0], abs=0.5 / FPS)
    w.next_scene()
    assert w.sel_start == pytest.approx(cuts[1], abs=0.5 / FPS)
    w.prev_scene()
    assert w.sel_start == pytest.approx(cuts[0], abs=0.5 / FPS)


@needs_ffmpeg
def test_add_scenes_to_list_adds_every_scene_once(window, video):
    w = window
    _open(w, video)
    _find(w)
    w.add_scenes_to_list()
    bounds = [w.info.start_time] + w.scene_cuts + [w.info.end_time]
    got = [(e.start, e.end) for e in w.clips.entries]
    assert got == pytest.approx(list(zip(bounds, bounds[1:])), abs=0.5 / FPS)
    w.add_scenes_to_list()    # again: nothing twice
    assert len(w.clips) == len(bounds) - 1
    # only the scenes between In and Out, cut to them
    w.clips.entries.clear()
    cuts = w.scene_cuts
    w.mark_in, w.mark_out = cuts[0] + 5 / FPS, cuts[2] + 5 / FPS
    w.add_scenes_to_list()
    got = [(e.start, e.end) for e in w.clips.entries]
    assert got == pytest.approx([(w.mark_in, cuts[1]), (cuts[1], cuts[2]), (cuts[2], w.mark_out)], abs=1e-6)


@needs_ffmpeg
def test_in_fixed_length_each_long_enough_scene_gets_one_clip_at_its_start(window, video):
    w = window
    _open(w, video)
    _find(w)
    w.b_fixed.setChecked(True)
    w.fixed_unit, w.fixed_frames = "frames", 40   # the 37-frame bars scene is too short
    w.add_scenes_to_list()
    bounds = [w.info.start_time] + w.scene_cuts + [w.info.end_time]
    long_enough = [a for a, b in zip(bounds, bounds[1:]) if b - a >= 40 / FPS - 1e-6]
    assert len(long_enough) == len(bounds) - 2
    got = [(e.start, e.end) for e in w.clips.entries]
    assert got == pytest.approx([(a, a + 40 / FPS) for a in long_enough], abs=1e-6)
    assert "left out" in w.status.currentMessage()


@needs_ffmpeg
def test_many_scenes_are_added_only_after_asking(window, video, monkeypatch):
    w = window
    _open(w, video)
    _find(w)
    w.scene_cuts = [w.info.start_time + i * 0.2 for i in range(1, 30)]   # 30 scenes
    answers = [QMessageBox.No, QMessageBox.Yes]
    asked = []

    def question(*args, **kwargs):
        asked.append(args)
        return answers[len(asked) - 1]

    monkeypatch.setattr(QMessageBox, "question", question)
    w.add_scenes_to_list()
    assert len(asked) == 1 and len(w.clips) == 0
    w.add_scenes_to_list()
    assert len(asked) == 2 and len(w.clips) == 30


@needs_ffmpeg
def test_a_search_of_a_video_that_was_closed_or_replaced_is_dropped(window, video, cuts_video):
    w = window
    _open(w, video)
    w.find_scene_cuts()
    t = w._scene_thread
    assert t is not None
    w.close_video()
    assert t.cancelled and not w.sc_progress.isVisibleTo(w)
    assert _wait(lambda: not w._scene_threads)
    assert w.scene_scores is None
    # a result that arrives for another video than the open one is not shown
    _open(w, cuts_video)
    w._scene_found(_scores([0.0, 1.0], [0, 50]), w._video_token + 100)
    assert w.scene_scores is None and w.scene_cuts == []


@needs_ffmpeg
def test_stop_ends_the_search_and_the_app_closes_cleanly_while_one_runs(window, video):
    w = window
    _open(w, video)
    w.find_scene_cuts()
    w._scene_cancel()
    assert w._scene_thread is None and not w.sc_progress.isVisibleTo(w)
    assert _wait(lambda: not w._scene_threads)
    w.find_scene_cuts()
    assert w._scene_threads
    w.close()   # waits for the search to stop
    assert all(not t.isRunning() for t in w._scene_threads)


@needs_ffmpeg
def test_a_stopped_search_reports_nothing_and_does_not_hide_the_next_one(window, video):
    w = window
    _open(w, video)
    w.find_scene_cuts()
    old = w._scene_thread.token
    w._scene_cancel()
    w.find_scene_cuts()   # straight away, while the stopped one may still be finishing
    new = w._scene_thread
    assert new.token != old
    w._scene_found(_scores([0.0, 1.0], [0, 50]), old)    # the stopped one's late result
    w._scene_failed("late\nerror", old)
    assert w.scene_scores is None and w._scene_thread is new and w.sc_progress.isVisibleTo(w)
    assert _wait(lambda: w.scene_scores is not None, 60)
    assert w.scene_cuts == pytest.approx(_cut_times(w), abs=0.5 / FPS)
    assert not w.sc_progress.isVisibleTo(w)


@needs_ffmpeg
def test_a_failed_search_says_why_and_hides_its_progress(window, video):
    w = window
    _open(w, video)
    w.find_scene_cuts()
    t = w._scene_thread
    w._scene_failed("Invalid data found when processing input\ntraceback...", t.token)
    t.cancel()
    assert not w.sc_progress.isVisibleTo(w) and w._scene_thread is None
    assert "Invalid data found" in w.status.currentMessage()
    assert _wait(lambda: not w._scene_threads)


@needs_ffmpeg
def test_find_takes_a_result_stored_by_a_search_stopped_as_it_finished(window, video):
    w = window
    _open(w, video)
    assert scenes.save_scores(video, scenes.scan(video))   # (stored, but never reported)
    w.find_scene_cuts()
    assert w.scene_scores is not None and not w._scene_threads


@needs_ffmpeg
def test_in_fixed_length_a_cut_too_near_the_end_says_so(window, video):
    w = window
    _open(w, video)
    _find(w)
    w.b_fixed.setChecked(True)
    w.fixed_unit, w.fixed_frames = "frames", 60   # the last cut is 45 frames before the end
    for _ in range(len(w.scene_cuts)):
        w.next_scene()
    assert w.sel_start < w.scene_cuts[-1]
    assert "doesn't fit" in w.status.currentMessage()


def test_page_up_and_down_stay_in_a_number_box(window):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    w = window
    for key in (Qt.Key_PageUp, Qt.Key_PageDown):
        ev = QKeyEvent(QEvent.ShortcutOverride, key, Qt.NoModifier)
        ev.ignore()
        assert w.eventFilter(w.len_frames, ev) and ev.isAccepted()
    ev = QKeyEvent(QEvent.ShortcutOverride, Qt.Key_PageUp, Qt.NoModifier)
    ev.ignore()
    w.eventFilter(w.view, ev)
    assert not ev.isAccepted()   # (on the video the keys jump between cuts)


def test_commands_without_a_video_or_without_cuts_do_nothing(window):
    w = window
    for f in (w.find_scene_cuts, w.next_scene, w.prev_scene, w.select_scene, w.add_scenes_to_list):
        f()
    assert w.scene_scores is None and not w._scene_threads
