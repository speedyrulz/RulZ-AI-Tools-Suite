"""The clip list: entries and their storage, the export each entry gets, the background batch,
and the main window's side of it (Add to list, the panel, Export list).

Batches here cut losslessly (stream copy), so no NVIDIA GPU is needed; the re-encode path is
covered by a stand-in job.

Run with:  python -m pytest tests   (from the app folder, inside its .venv).
"""
import json
import os
import shutil
import time

import numpy as np
import pytest

from test_core import APP, make_video  # noqa: F401  (sets up Qt, the settings file and sys.path)

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from simplecut import cliplist  # noqa: E402
from simplecut.cliplist import (BatchExport, ClipEntry, ClipList, ClipListPanel, DONE, FAILED, PENDING,  # noqa: E402
                                QUEUED, RUNNING, clean_store, entry_output_path, entry_settings,
                                short_time, store_dir, store_path)
from simplecut.defaults import settings  # noqa: E402
from simplecut.export import ExportSettings  # noqa: E402
from simplecut.probe import VideoInfo, have_ffmpeg, probe_streams, scan_keyframes  # noqa: E402
from simplecut.timeline import Timeline  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg / ffprobe not on PATH")


def _wait(cond, timeout=30):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        APP.processEvents()
        time.sleep(0.01)
    return cond()


@pytest.fixture(autouse=True)
def empty_store():
    """Every test starts without stored lists (the store sits beside the tests' settings file)."""
    shutil.rmtree(store_dir(), ignore_errors=True)
    yield
    shutil.rmtree(store_dir(), ignore_errors=True)


@pytest.fixture
def clip(tmp_path):
    """10 s at 30 fps (300 frames), a keyframe every 30."""
    return make_video(str(tmp_path / "c.mp4"), n=300, fps=30, gop=30)


@pytest.fixture
def window():
    from simplecut.app import MainWindow
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()
    APP.processEvents()


def _open(w, path, keyframes=True):
    before = w._video_token
    w.open_file(path)
    assert _wait(lambda: w._video_token != before and w.info is not None and w.dec is not None
                 and w.dec.current is not None)
    if keyframes:
        assert _wait(lambda: bool(w.info.keyframes))


def _info(path) -> VideoInfo:
    info = probe_streams(path)
    info.keyframes, info.keyframe_dts = scan_keyframes(path)
    return info


def _keyframe_from(info, t):
    """The first keyframe at or after t seconds into the video."""
    return next(k for k in info.keyframes if k >= info.start_time + t - 1e-6)


def _not_a_keyframe(info):
    """A frame time that is not a keyframe (the encoder puts extra ones at picture changes)."""
    fd = info.frame_duration
    return next(info.start_time + i * fd for i in range(1, 300) if not info.is_keyframe_time(info.start_time + i * fd))


def _tables():
    t = np.tile(np.arange(256, dtype=np.uint8), (3, 1))
    t[0] = np.clip(t[0].astype(int) + 10, 0, 255)
    return t


# ------------------------------------------------------------------ entries
def test_an_entry_survives_storage_with_everything_it_applies():
    from simplecut.repair.core.params import Corrections
    from simplecut.repair.pipeline import RepairSettings
    c = Corrections()
    c.horizon.enabled, c.horizon.angle = True, 3.5
    e = ClipEntry(1.5, 4.0, rotation=90, hflip=True, crop=(10, 20, 100, 60), out_size=(64, 48),
                  color_tables=_tables(), color_desc="warmer", repair=RepairSettings(c, 90, True))
    back = ClipEntry.from_dict(json.loads(json.dumps(e.to_dict())))
    assert (back.start, back.end, back.rotation, back.hflip) == (1.5, 4.0, 90, True)
    assert back.crop == (10, 20, 100, 60) and back.out_size == (64, 48)
    assert np.array_equal(back.color_tables, _tables()) and back.color_tables.dtype == np.uint8
    assert back.color_desc == "warmer"
    assert back.repair.corrections.horizon.angle == 3.5
    assert (back.repair.base_rotation, back.repair.base_flip) == (90, True)
    assert back.same_clip(e) and back.needs_reencode
    assert "Rotated 90" in back.describe() and "Horizon" in back.describe()
    assert back.tags() == "turned, crop, colour, repair"
    plain = ClipEntry(0.0, 1.0)
    assert not plain.needs_reencode and plain.describe() == "" and plain.tags() == ""
    assert not plain.same_clip(ClipEntry(0.0, 1.0, crop=(0, 0, 8, 8)))
    assert not plain.same_clip(ClipEntry(0.0, 1.5))


def test_a_clip_caught_mid_export_is_stored_as_not_exported_yet():
    for status in (QUEUED, RUNNING):
        e = ClipEntry(0.0, 1.0, status=status, progress=0.4)
        assert ClipEntry.from_dict(e.to_dict()).status == PENDING
    e = ClipEntry(0.0, 1.0, status=DONE, out_path="x.mp4")
    back = ClipEntry.from_dict(e.to_dict())
    assert (back.status, back.out_path) == (DONE, "x.mp4")


@pytest.mark.parametrize("bad", [
    {"start": 2.0, "end": 1.0},                                   # ends before it starts
    {"start": 0.0, "end": float("nan")},
    {"start": 0.0, "end": 1.0, "rotation": 45},
    {"start": 0.0, "end": 1.0, "crop": [0, 0, 10]},               # three numbers
    {"start": 0.0, "end": 1.0, "crop": [0, 0, 0, 10]},            # no width
    {"start": 0.0, "end": 1.0, "color": {"tables": [[1, 2, 3]]}},
    {"start": 0.0, "end": 1.0, "repair": {"corrections": "no"}},
    {"end": 1.0},
    # numbers JSON allows and a frame can't have (json.loads reads Infinity and 1e999)
    {"start": 0.0, "end": float("inf")},
    {"start": 0.0, "end": 1.0, "rotation": float("inf")},
    {"start": 0.0, "end": 1.0, "crop": [0, 0, float("inf"), 10]},
    {"start": 0.0, "end": 1.0, "out_size": [64, float("inf")]},
    # text where a list or a yes/no belongs: it would be read digit by digit, or as "yes"
    {"start": 0.0, "end": 1.0, "crop": "1234"},
    {"start": 0.0, "end": 1.0, "out_size": "64"},
    {"start": 0.0, "end": 1.0, "hflip": "false"},
    {"start": 0.0, "end": 1.0, "repair": {"corrections": {}, "base_flip": "no"}},
])
def test_unusable_stored_entries_are_refused(bad):
    with pytest.raises((KeyError, TypeError, ValueError, OverflowError)):
        ClipEntry.from_dict(bad)


def test_short_time():
    assert short_time(3.0) == "0:03.000" and short_time(75.25) == "1:15.250"
    assert short_time(3723.5) == "62:03.500" and short_time(-1) == "0:00.000"
    assert short_time(59.9996) == "1:00.000"


# ------------------------------------------------------------------ storage
def test_a_list_is_stored_per_video_and_comes_back(clip):
    lst = ClipList.load(clip)
    assert len(lst) == 0 and not os.path.exists(store_path(clip))
    lst.entries += [ClipEntry(1.0, 2.0), ClipEntry(3.0, 4.5, crop=(0, 0, 80, 48), status=DONE, out_path="o.mp4")]
    assert lst.save() and os.path.isfile(store_path(clip))
    back = ClipList.load(clip.upper())                 # Windows paths ignore letter case
    assert [(e.start, e.end, e.crop, e.status) for e in back] == [
        (1.0, 2.0, None, PENDING), (3.0, 4.5, (0, 0, 80, 48), DONE)]
    assert back.matches_file() and not back.busy()
    assert [e.start for e in back.waiting()] == [1.0]
    assert back.find(ClipEntry(1.0, 2.0)) is back.entries[0] and back.find(ClipEntry(1.0, 2.5)) is None
    back.entries.clear()
    assert back.save() and not os.path.exists(store_path(clip))   # an empty list leaves no file


def test_a_list_made_on_another_file_of_the_same_name_is_not_used(clip, tmp_path):
    lst = ClipList.load(clip)
    lst.entries.append(ClipEntry(1.0, 2.0))
    lst.save()
    make_video(clip, n=60, fps=30, gop=30)             # the video is replaced by a shorter one
    assert not lst.matches_file()
    back = ClipList.load(clip)
    assert len(back) == 0 and back.replaced and "another version" in back.note()
    # ...and a save from the old list (a batch that was still running) doesn't adopt the new file
    lst.save()
    assert len(ClipList.load(clip)) == 0
    # the first clip kept for the new file sets the old list aside instead of writing over it
    back.entries.append(ClipEntry(0.5, 1.0))
    assert back.save() and not os.path.exists(store_path(clip) + ".old.old")
    with open(store_path(clip) + ".old", encoding="utf-8") as f:
        assert [c["start"] for c in json.load(f)["clips"]] == [1.0]
    again = ClipList.load(clip)
    assert [e.start for e in again] == [0.5] and not again.replaced and again.note() == ""


def test_unusable_entries_and_files_are_skipped_not_fatal(clip):
    lst = ClipList.load(clip)
    lst.entries.append(ClipEntry(1.0, 2.0))
    lst.save()
    p = store_path(clip)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    d["clips"].append({"start": 5.0, "end": 4.0})
    d["clips"].append("nonsense")
    d["clips"].append({"start": 0.0, "end": 1.0, "rotation": float("inf")})   # written as Infinity
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f)
    back = ClipList.load(clip)
    assert len(back) == 1 and back.dropped == 3 and "3 stored clips could not be read" in back.note()
    # saving what could be read keeps the file it came from, for a look at what was left out
    assert back.save() and len(ClipList.load(clip)) == 1 and ClipList.load(clip).dropped == 0
    with open(p + ".old", encoding="utf-8") as f:
        assert len(json.load(f)["clips"]) == 4
    # a size or date that is no number: the list is not this file's, and nothing is raised
    d["video"]["size"] = float("inf")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f)
    back = ClipList.load(clip)
    assert len(back) == 0 and back.replaced
    with open(p, "w", encoding="utf-8") as f:
        f.write("{ not json")
    assert len(ClipList.load(clip)) == 0
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"kind": "something else", "clips": [{"start": 0, "end": 1}]}, f)
    assert len(ClipList.load(clip)) == 0


def test_a_list_that_cannot_be_written_reports_it(clip, monkeypatch):
    lst = ClipList.load(clip)
    lst.entries.append(ClipEntry(1.0, 2.0))

    def refuse(*a, **k):
        raise PermissionError("read-only")
    monkeypatch.setattr(cliplist.os, "makedirs", refuse)
    assert not lst.save() and "read-only" in lst.save_error
    monkeypatch.undo()
    lst.entries[0].out_path = {1, 2}        # something JSON can't hold: reported, not raised
    assert not lst.save() and lst.save_error


def test_old_set_aside_lists_are_cleaned_up_too(clip):
    os.makedirs(store_dir(), exist_ok=True)
    old = store_path(clip) + ".old"
    with open(old, "w", encoding="utf-8") as f:
        f.write("{}")
    clean_store(now=time.time() + 30 * 86400)
    assert os.path.exists(old)
    clean_store(now=time.time() + (cliplist.STALE_DAYS + 1) * 86400)
    assert not os.path.exists(old)


def test_old_lists_of_missing_videos_are_cleaned_up(clip, tmp_path):
    gone = make_video(str(tmp_path / "gone.mp4"), n=30, fps=30, gop=30)
    for v in (clip, gone):
        lst = ClipList.load(v)
        lst.entries.append(ClipEntry(0.0, 0.5))
        lst.save()
    os.remove(gone)
    clean_store()                                                  # too recent: both stay
    assert os.path.exists(store_path(gone)) and os.path.exists(store_path(clip))
    clean_store(now=time.time() + (cliplist.STALE_DAYS + 1) * 86400)
    assert not os.path.exists(store_path(gone)) and os.path.exists(store_path(clip))


# ------------------------------------------------------------------ what an entry exports as
@needs_ffmpeg
def test_an_entry_is_cut_losslessly_only_when_allowed_and_possible(clip, tmp_path):
    info = _info(clip)
    ka, kb = _keyframe_from(info, 1.0), _keyframe_from(info, 3.0)
    base = ExportSettings(mode="nvenc", codec="hevc_nvenc", cq=21, fps="30", audio="none")
    aligned = ClipEntry(ka, kb)
    s = entry_settings(info, aligned, base, allow_lossless=True)
    assert s.mode == "lossless" and s.repair is None and s.lut_path is None
    assert entry_settings(info, aligned, base, allow_lossless=False).mode == "nvenc"
    off = ClipEntry(_not_a_keyframe(info), info.end_time)             # In is not on a keyframe
    assert entry_settings(info, off, base, True).mode == "nvenc"
    cropped = ClipEntry(info.start_time + 1.0, info.start_time + 3.0, crop=(0, 0, 80, 48), out_size=(40, 24),
                        hflip=True, rotation=90)
    s = entry_settings(info, cropped, base, True)
    assert (s.mode, s.crop, s.out_size, s.hflip, s.rotation) == ("nvenc", (0, 0, 80, 48), (40, 24), True, 90)
    assert (s.codec, s.cq, s.fps, s.audio) == ("hevc_nvenc", 21, "30", "none")   # Quick export's encode options
    assert base.mode == "nvenc" and base.crop is None                 # the shared settings are not changed
    assert os.path.basename(entry_output_path(info, cropped, s, str(tmp_path))) == "c_clip_1_00-3_00_crop.mp4"
    coloured = ClipEntry(ka, kb, color_tables=_tables())
    s = entry_settings(info, coloured, base, True)
    assert s.mode == "nvenc" and s.lut_path and os.path.isfile(s.lut_path)
    lossless = entry_settings(info, aligned, base, True)
    name = os.path.basename(entry_output_path(info, aligned, lossless, None))
    assert name.startswith("c_clip_") and name.endswith(".mp4") and "_crop" not in name
    assert os.path.dirname(entry_output_path(info, aligned, lossless, None)) == os.path.dirname(clip)


@needs_ffmpeg
def test_an_entry_with_a_repair_gets_the_piped_export(clip):
    from simplecut.repair.core.params import Corrections
    from simplecut.repair.pipeline import RepairSettings
    info = _info(clip)
    c = Corrections()
    c.horizon.enabled, c.horizon.angle = True, 4.0
    e = ClipEntry(_keyframe_from(info, 1.0), _keyframe_from(info, 2.0), repair=RepairSettings(c, 0, False))
    s = entry_settings(info, e, ExportSettings(mode="nvenc"), allow_lossless=True)
    assert s.mode == "nvenc" and s.repair is not None
    assert s.repair.in_size == (160, 96) and s.repair.out_size[0] <= 160
    assert "_repair" in entry_output_path(info, e, s, None)
    # stored repair settings that can't be used here (OpenCV missing) fail with a reason
    stuck = ClipEntry(0.0, 1.0, repair_data={"corrections": {}, "base_rotation": 0, "base_flip": False})
    with pytest.raises(ValueError, match="OpenCV"):
        entry_settings(info, stuck, ExportSettings(mode="nvenc"), True)


# ------------------------------------------------------------------ the batch
class FakeJob(QObject):
    """Stands in for ExportJob: ends when told to."""
    progress = Signal(float)
    finished = Signal(bool, str)
    made: list["FakeJob"] = []

    def __init__(self, info, start, end, out_path, s, parent=None):
        super().__init__(parent)
        self.out_path, self.settings = out_path, s
        self.cancelled = self.done = self.started = False
        self.warning = ""
        self.cmd_text = "ffmpeg (stand-in)"
        FakeJob.made.append(self)

    def start(self):
        self.started = True
        with open(self.out_path, "wb") as f:
            f.write(b"partial")

    def end(self, ok=True, log=""):
        self.done = True
        self.finished.emit(ok, log)

    slow_cancel = False   # True: cancel() only asks, as ffmpeg being killed; the test ends the job later

    def cancel(self):
        self.cancelled = True
        if not self.slow_cancel:
            self.end(False)

    def wait(self, msecs=0):
        return True


@pytest.fixture
def fake_jobs(monkeypatch):
    FakeJob.made = []
    monkeypatch.setattr(cliplist, "make_export_job", FakeJob)
    return FakeJob.made


def _batch_with(clip, n, tmp_path):
    info = VideoInfo(path=clip, width=160, height=96, fps=30.0, duration=10.0)
    lst = ClipList.load(clip)
    lst.entries += [ClipEntry(float(i), i + 0.5, crop=(0, 0, 80, 48)) for i in range(n)]
    lst.save()
    batch = BatchExport()
    events = {"started": [], "exported": [], "finished": []}
    batch.clip_started.connect(lambda name, i, total: events["started"].append((name, i, total)))
    batch.exported.connect(events["exported"].append)
    batch.finished.connect(lambda s, f, c: events["finished"].append((s, f, c)))
    out = tmp_path / "out"
    out.mkdir()
    return info, lst, batch, events, str(out)


def test_a_batch_runs_its_clips_one_after_another(clip, tmp_path, fake_jobs):
    info, lst, batch, ev, out = _batch_with(clip, 3, tmp_path)
    assert batch.add(info, lst, lst.entries, ExportSettings(mode="nvenc"), False, out) == 3
    assert batch.running and len(fake_jobs) == 1 and fake_jobs[0].started
    assert [e.status for e in lst] == [RUNNING, QUEUED, QUEUED] and lst.busy()
    assert batch.add(info, lst, lst.entries, ExportSettings(mode="nvenc"), False, out) == 0   # queued already
    fake_jobs[0].progress.emit(0.5)
    assert lst.entries[0].progress == 0.5
    fake_jobs[0].end(True)
    assert [e.status for e in lst] == [DONE, RUNNING, QUEUED] and len(fake_jobs) == 2
    assert lst.entries[0].out_path == fake_jobs[0].out_path and os.path.dirname(fake_jobs[0].out_path) == out
    assert [e.status for e in ClipList.load(clip)] == [DONE, PENDING, PENDING]   # stored as it goes
    fake_jobs[1].end(True)
    fake_jobs[2].end(True)
    assert not batch.running and ev["finished"] == [(3, 0, False)]
    assert ev["started"] == [(os.path.basename(j.out_path), i, 3) for i, j in enumerate(fake_jobs, 1)]
    assert ev["exported"] == [j.out_path for j in fake_jobs]
    assert len({j.out_path for j in fake_jobs}) == 3
    assert all(j.settings.crop == (0, 0, 80, 48) and j.settings.mode == "nvenc" for j in fake_jobs)


def test_a_failed_clip_is_marked_and_the_rest_still_export(clip, tmp_path, fake_jobs):
    info, lst, batch, ev, out = _batch_with(clip, 3, tmp_path)
    batch.add(info, lst, lst.entries, ExportSettings(mode="nvenc"), False, out)
    fake_jobs[0].end(False, "No NVENC capable devices found")
    first = lst.entries[0]
    assert first.status == FAILED and "No NVENC capable devices found" in first.error
    assert not os.path.exists(fake_jobs[0].out_path)       # the broken file is not left behind
    fake_jobs[1].end(True)
    fake_jobs[2].end(True)
    assert ev["finished"] == [(2, 1, False)] and "No NVENC" in batch.first_failure
    assert [e.status for e in ClipList.load(clip)] == [FAILED, DONE, DONE]
    # "Export list" takes the failed one again
    assert batch.add(info, lst, lst.waiting(), ExportSettings(mode="nvenc"), False, out) == 1
    fake_jobs[3].end(True)
    assert first.status == DONE and first.error == "" and ev["finished"][-1] == (1, 0, False)


def test_cancel_stops_the_batch_and_keeps_the_unwritten_clips_for_later(clip, tmp_path, fake_jobs):
    info, lst, batch, ev, out = _batch_with(clip, 3, tmp_path)
    batch.add(info, lst, lst.entries, ExportSettings(mode="nvenc"), False, out)
    fake_jobs[0].end(True)
    batch.cancel()
    assert not batch.running and len(fake_jobs) == 2 and fake_jobs[1].cancelled
    assert not os.path.exists(fake_jobs[1].out_path)       # the half-written clip is deleted
    assert [e.status for e in lst] == [DONE, PENDING, PENDING]
    assert ev["finished"] == [(1, 0, True)]
    batch.cancel()                                         # nothing running: nothing happens
    assert ev["finished"] == [(1, 0, True)]
    assert batch.add(info, lst, lst.waiting(), ExportSettings(mode="nvenc"), False, out) == 2   # and it restarts
    assert [e.status for e in lst] == [DONE, RUNNING, QUEUED]
    batch.cancel()


def test_clips_queued_while_a_cancelled_job_is_still_ending_are_exported(clip, tmp_path, fake_jobs, monkeypatch):
    monkeypatch.setattr(FakeJob, "slow_cancel", True)
    info, lst, batch, ev, out = _batch_with(clip, 3, tmp_path)
    batch.add(info, lst, lst.entries[:2], ExportSettings(mode="nvenc"), False, out)
    batch.cancel()                                         # ffmpeg was told to stop and has not yet
    assert batch.running and [e.status for e in lst] == [RUNNING, PENDING, PENDING]
    batch.add(info, lst, [lst.entries[2]], ExportSettings(mode="nvenc"), False, out)
    assert len(fake_jobs) == 1
    fake_jobs[0].end(False)                                # now it has
    assert ev["finished"] == [(0, 0, True)] and lst.entries[0].status == PENDING
    assert batch.running and len(fake_jobs) == 2 and lst.entries[2].status == RUNNING
    fake_jobs[1].end(True)                                 # ...and is not thrown away as "cancelled"
    assert lst.entries[2].status == DONE and os.path.exists(fake_jobs[1].out_path)
    assert ev["finished"] == [(0, 0, True), (1, 0, False)] and not batch.running


def test_a_clip_that_was_complete_when_cancel_arrived_is_kept(clip, tmp_path, fake_jobs, monkeypatch):
    monkeypatch.setattr(FakeJob, "slow_cancel", True)
    info, lst, batch, ev, out = _batch_with(clip, 2, tmp_path)
    batch.add(info, lst, lst.entries, ExportSettings(mode="nvenc"), False, out)
    batch.cancel()
    fake_jobs[0].end(True)                                 # ffmpeg had already finished the file
    assert lst.entries[0].status == DONE and os.path.exists(fake_jobs[0].out_path)
    assert lst.entries[1].status == PENDING and len(fake_jobs) == 1 and not batch.running
    assert ev["finished"] == [(1, 0, True)] and ev["exported"] == [fake_jobs[0].out_path]
    assert [e.status for e in ClipList.load(clip)] == [DONE, PENDING]


def test_an_unexpected_error_in_one_clip_does_not_stop_the_batch(clip, tmp_path, fake_jobs, monkeypatch):
    info, lst, batch, ev, out = _batch_with(clip, 2, tmp_path)
    real = cliplist.entry_settings

    def flaky(info, e, base, allow):
        if e is lst.entries[0]:
            raise RuntimeError("remap tables could not be built")
        return real(info, e, base, allow)
    monkeypatch.setattr(cliplist, "entry_settings", flaky)
    batch.add(info, lst, lst.entries, ExportSettings(mode="nvenc"), False, out)
    assert lst.entries[0].status == FAILED and "remap tables" in lst.entries[0].error
    fake_jobs[0].end(True)
    assert ev["finished"] == [(1, 1, False)] and not batch.running


def test_clips_can_be_added_and_taken_out_while_a_batch_runs(clip, tmp_path, fake_jobs):
    info, lst, batch, ev, out = _batch_with(clip, 3, tmp_path)
    batch.add(info, lst, lst.entries[:2], ExportSettings(mode="nvenc"), False, out)
    batch.discard([lst.entries[1], lst.entries[0]])        # the running one stays, the waiting one leaves
    assert [e.status for e in lst] == [RUNNING, PENDING, PENDING]
    batch.add(info, lst, [lst.entries[2]], ExportSettings(mode="nvenc"), False, out)
    fake_jobs[0].end(True)
    fake_jobs[1].end(True)
    assert [e.status for e in lst] == [DONE, PENDING, DONE] and ev["finished"] == [(2, 0, False)]
    assert ev["started"][-1][1:] == (2, 2)                 # "clip 2 of 2": counted over the whole run


def test_a_clip_that_cannot_start_fails_without_stopping_the_batch(clip, tmp_path, fake_jobs, monkeypatch):
    info, lst, batch, ev, out = _batch_with(clip, 2, tmp_path)
    lst.entries[0].repair_data = {"corrections": {}, "base_rotation": 0, "base_flip": False}   # unusable here
    batch.add(info, lst, lst.entries, ExportSettings(mode="nvenc"), False, out)
    assert lst.entries[0].status == FAILED and "OpenCV" in lst.entries[0].error
    assert lst.entries[1].status == RUNNING
    fake_jobs[0].end(True)
    assert ev["finished"] == [(1, 1, False)]
    # the source video gone: every clip fails with the reason, and the batch still ends
    gone = VideoInfo(path=str(tmp_path / "missing.mp4"), width=160, height=96, fps=30.0, duration=10.0)
    lst.entries[1].status = PENDING
    batch.add(gone, lst, [lst.entries[1]], ExportSettings(mode="nvenc"), False, out)
    assert lst.entries[1].status == FAILED and "no longer there" in lst.entries[1].error
    assert not batch.running and ev["finished"][-1] == (0, 1, False)


@needs_ffmpeg
def test_a_real_batch_cuts_keyframe_aligned_clips_losslessly(clip, tmp_path):
    info = _info(clip)
    cuts = [(_keyframe_from(info, 1.0), _keyframe_from(info, 2.0)),
            (_keyframe_from(info, 4.0), _keyframe_from(info, 6.0))]
    lst = ClipList.load(clip)
    lst.entries += [ClipEntry(a, b) for a, b in cuts]
    batch = BatchExport()
    done = []
    batch.finished.connect(lambda s, f, c: done.append((s, f, c)))
    out = tmp_path / "out"
    out.mkdir()
    batch.add(info, lst, lst.entries, ExportSettings(mode="nvenc"), True, str(out))
    assert _wait(lambda: bool(done), 60)
    assert done == [(2, 0, False)], lst.entries[0].error or lst.entries[1].error
    for e, (a, b) in zip(lst, cuts):
        assert e.status == DONE and os.path.isfile(e.out_path)
        assert probe_streams(e.out_path).duration == pytest.approx(b - a, abs=0.02)
    assert len(os.listdir(out)) == 2 and all(n.startswith("c_clip_") for n in os.listdir(out))


# ------------------------------------------------------------------ timeline and panel
def test_the_timeline_draws_the_lists_clips():
    tl = Timeline()
    tl.resize(400, 60)
    tl.set_range(0.0, 10.0)
    plain = QImage(400, 60, QImage.Format_RGB32)
    tl.render(plain)
    tl.set_clips([(1.0, 3.0, False), (5.0, 6.0, True), (9.5, 12.0, False)])   # (one runs past the end)
    drawn = QImage(400, 60, QImage.Format_RGB32)
    tl.render(drawn)
    assert drawn != plain
    y = 14                                                   # inside the strip above the bar
    assert drawn.pixel(int(tl._x(2.0)), y) != plain.pixel(int(tl._x(2.0)), y)     # waiting: filled
    assert drawn.pixel(int(tl._x(4.0)), y) == plain.pixel(int(tl._x(4.0)), y)     # between clips: untouched
    assert drawn.pixel(399, y) == plain.pixel(399, y)        # clamped to the bar
    tl.set_clips([])
    again = QImage(400, 60, QImage.Format_RGB32)
    tl.render(again)
    assert again == plain


def test_the_panel_lists_clips_and_asks_for_what_its_buttons_say():
    p = ClipListPanel()
    a = ClipEntry(11.0, 12.5, crop=(0, 0, 8, 8))
    b = ClipEntry(13.0, 14.0, status=DONE, out_path="b.mp4")
    c = ClipEntry(15.0, 16.0, status=RUNNING, progress=0.42)
    p.set_clips([a, b, c], t0=10.0)
    row = lambda i: [p.tree.topLevelItem(i).text(col) for col in range(6)]   # noqa: E731
    assert row(0) == ["1", "0:01.000", "0:02.500", "1.50 s", "crop", ""]     # times from the first frame
    assert row(1)[5] == "saved" and row(2)[5] == "42 %"
    assert "b.mp4" in p.tree.topLevelItem(1).toolTip(0)
    assert "2 to export" in p.title.text()
    assert p.b_export.isEnabled() and not p.b_remove.isEnabled() and p.b_clear.isEnabled()
    asked = {"remove": [], "go": [], "export": 0}
    p.removeRequested.connect(asked["remove"].append)
    p.goRequested.connect(asked["go"].append)
    p.exportRequested.connect(lambda: asked.__setitem__("export", asked["export"] + 1))
    p.tree.topLevelItem(1).setSelected(True)
    assert p.selected() == [b] and p.b_remove.isEnabled()
    p.b_remove.click()
    p.b_export.click()
    p.tree.itemDoubleClicked.emit(p.tree.topLevelItem(0), 0)
    assert asked == {"remove": [[b]], "go": [a], "export": 1}
    c.progress = 0.9
    p.update_status()                                        # an export moved on: rows stay, selection stays
    assert row(2)[5] == "90 %" and p.selected() == [b]
    p.set_clips([a, b, c], t0=10.0)
    assert p.selected() == [b]                               # a rebuild keeps the selection too
    p.tree.clearSelection()
    p.tree.topLevelItem(2).setSelected(True)
    assert not p.b_remove.isEnabled()                        # the clip being written can't be removed
    p.set_clips([b])
    assert "all saved" in p.title.text() and not p.b_export.isEnabled()
    p.deleteLater()


# ------------------------------------------------------------------ the main window
def test_add_to_list_keeps_the_selection_and_what_is_applied(window, clip):
    w = window
    _open(w, clip)
    t0, fd = w.info.start_time, w.info.frame_duration
    w.add_to_list()                                          # nothing selected
    assert len(w.clips) == 0 and "mark" in w.status.currentMessage()
    assert w.clip_panel.isHidden()
    w.seek_to(t0 + 1.0)
    w.set_in()
    w.seek_to(t0 + 2.0)
    w.set_out()
    w.add_to_list()
    w.rotate()
    w.flip_h()
    w._set_color(_tables(), "warmer")
    w.b_crop.setChecked(True)
    w.add_to_list()
    assert len(w.clips) == 2 and not w.clip_panel.isHidden()
    first, second = w.clips.entries
    assert (first.start, first.end) == (pytest.approx(t0 + 1.0), pytest.approx(t0 + 2.0))
    assert not first.needs_reencode                          # as it was when it was added
    assert (second.rotation, second.hflip) == (90, True) and second.crop is not None
    assert np.array_equal(second.color_tables, _tables()) and second.color_desc == "warmer"
    w.add_to_list()                                          # the same clip again: not listed twice
    assert len(w.clips) == 2 and "already" in w.status.currentMessage()
    w.color_tables[0, 0] = 99                                # a later change doesn't reach the entry
    assert second.color_tables[0, 0] == _tables()[0, 0]
    w.color_tables[0, 0] = _tables()[0, 0]
    assert w.timeline.clips == [(first.start, first.end, False)] * 2
    assert w.clip_panel.tree.topLevelItemCount() == 2
    assert w.clip_panel.tree.topLevelItem(1).text(4) == "turned, crop, colour"
    # a fixed-length selection is added the same way
    w.b_fixed.setChecked(True)
    w.add_to_list()
    assert len(w.clips) == 3
    assert w.clips.entries[2].end - w.clips.entries[2].start == pytest.approx(w._fixed_duration())


def test_a_saved_clip_added_again_is_exported_again(window, clip):
    w = window
    _open(w, clip)
    w.seek_to(w.info.start_time + 1.0)
    w.set_in()
    w.add_to_list()
    e = w.clips.entries[0]
    e.status, e.out_path = DONE, "old.mp4"
    w.add_to_list()
    assert len(w.clips) == 1 and e.status == PENDING and "again" in w.status.currentMessage()


def test_the_list_follows_the_video_and_comes_back_when_it_is_opened_again(window, clip, tmp_path):
    w = window
    other = make_video(str(tmp_path / "other.mp4"), n=90, fps=30, gop=30)
    _open(w, clip)
    w.seek_to(w.info.start_time + 3.0)
    w.set_in()
    w.add_to_list()
    kept = w.clips
    _open(w, other, keyframes=False)
    assert len(w.clips) == 0 and w.clip_panel.isHidden() and w.timeline.clips == []
    _open(w, clip, keyframes=False)
    assert w.clips is kept and not w.clip_panel.isHidden()   # the same entries a running batch would hold
    w.close_video()
    assert w.clips is None and w.clip_panel.isHidden() and w.timeline.clips == []
    w.add_to_list()                                          # no video: nothing happens
    w._clip_lists.clear()                                    # as after a restart
    _open(w, clip, keyframes=False)
    assert w.clips is not kept and [round(e.start - w.info.start_time, 2) for e in w.clips] == [3.0]
    # the video replaced by another file of the same name: the cached list is not carried over
    w.close_video()
    make_video(clip, n=60, fps=30, gop=30)
    _open(w, clip, keyframes=False)
    assert len(w.clips) == 0


def test_go_to_remove_export_again_and_clear(window, clip, monkeypatch):
    w = window
    _open(w, clip)
    t0 = w.info.start_time
    for a, b in ((1.0, 2.0), (4.0, 5.5), (7.0, 8.0)):
        w.seek_to(t0 + a)
        w.set_in()
        w.seek_to(t0 + b)
        w.set_out()
        w.add_to_list()
    e1, e2, e3 = w.clips.entries
    w.clear_marks()
    w.goto_clip(e2)
    assert (w.mark_in, w.mark_out) == (e2.start, e2.end) and w.dec.time == pytest.approx(e2.start)
    w.b_fixed.setChecked(True)
    w.goto_clip(e3)
    assert w.sel_start == pytest.approx(e3.start)            # in fixed-length mode the box goes there
    w.b_fixed.setChecked(False)
    e1.status, e1.out_path = DONE, "one.mp4"
    w._reset_clips([e1, e2])
    assert e1.status == PENDING
    w._remove_clips([e2])
    assert w.clips.entries == [e1, e3] and len(ClipList.load(clip)) == 2
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    w._clear_clips()
    assert len(w.clips) == 2
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    w._clear_clips()
    assert len(w.clips) == 0 and w.clip_panel.isHidden() and not os.path.exists(store_path(clip))


def test_export_list_runs_in_the_background_and_holds_the_other_exports(window, clip, tmp_path, fake_jobs,
                                                                         monkeypatch):
    w = window
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append(a[2]))
    out = tmp_path / "out"
    out.mkdir()
    settings().setValue("output_dir", str(out))
    try:
        _open(w, clip)
        w.export_list()
        assert not w.batch.running and "Nothing" in w.status.currentMessage()
        t0 = w.info.start_time
        for a in (1.0, 3.0):
            w.seek_to(t0 + a)
            w.set_in()
            w.seek_to(t0 + a + 1.0)
            w.set_out()
            w.add_to_list()
        w.export_list()
        assert w.batch.running and len(fake_jobs) == 1
        assert os.path.dirname(fake_jobs[0].out_path) == str(out)
        assert not w.b_quick.isEnabled() and not w.b_export.isEnabled()
        assert not w.qx_progress.isHidden() and "clip 1 of 2" in w.qx_label.text()
        w.quick_export()
        w.export_clip()
        assert w.quick_job is None and len(fake_jobs) == 1   # neither started beside the batch
        w.set_in()                                           # a selection change must not re-enable Export
        assert not w.b_export.isEnabled()
        fake_jobs[0].progress.emit(0.5)
        assert w.qx_progress.value() == 500 and w.clip_panel.tree.topLevelItem(0).text(5) == "50 %"
        # another video is opened meanwhile: the batch goes on with the first one's clips
        first = w.clips
        other = make_video(str(tmp_path / "other.mp4"), n=90, fps=30, gop=30)
        _open(w, other, keyframes=False)
        assert w.batch.running and not w.b_quick.isEnabled()
        fake_jobs[0].end(True)
        assert first.entries[0].status == DONE and len(fake_jobs) == 2
        fake_jobs[1].end(False, "boom")
        assert not w.batch.running and w.qx_progress.isHidden()
        assert w.b_quick.isEnabled() and w.last_export == fake_jobs[0].out_path
        assert len(shown) == 1 and "1 clip could not be exported" in shown[0] and "boom" in shown[0]
        assert [e.status for e in ClipList.load(clip)] == [DONE, FAILED]
    finally:
        settings().remove("output_dir")


def test_the_status_bar_cancel_and_closing_the_app_stop_a_batch(clip, tmp_path, fake_jobs):
    from simplecut.app import MainWindow
    w = MainWindow()
    try:
        _open(w, clip)
        t0 = w.info.start_time
        for a in (1.0, 3.0):
            w.seek_to(t0 + a)
            w.set_in()
            w.seek_to(t0 + a + 1.0)
            w.set_out()
            w.add_to_list()
        w.export_list()
        w.qx_cancel.click()
        assert not w.batch.running and fake_jobs[0].cancelled
        assert "cancelled" in w.status.currentMessage() and w.qx_progress.isHidden()
        assert [e.status for e in w.clips] == [PENDING, PENDING]
        w.export_list()
        assert w.batch.running
    finally:
        w.close()                                            # closing cancels and leaves the list for next time
        w.deleteLater()
        APP.processEvents()
    assert not w.batch.running and fake_jobs[-1].cancelled
    assert [e.status for e in ClipList.load(clip)] == [PENDING, PENDING]


def test_a_stored_list_that_cannot_be_read_never_stops_a_video_from_opening(window, clip, tmp_path, monkeypatch):
    w = window
    lst = ClipList.load(clip)
    lst.entries += [ClipEntry(1.0, 2.0), ClipEntry(3.0, 4.0)]
    lst.save()
    p = store_path(clip)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    d["clips"][1]["rotation"] = float("inf")                 # stored as Infinity
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f)
    _open(w, clip)
    assert [e.start for e in w.clips] == [1.0]
    assert "1 stored clip could not be read" in w.status.currentMessage()   # ...and it stays said
    assert "keyframes" in w.status.currentMessage()
    # whatever the loader raises, the video opens with an empty list of its own (not the last one's)
    other = make_video(str(tmp_path / "other.mp4"), n=90, fps=30, gop=30)

    def broken(path):
        raise RuntimeError("store unreadable")
    monkeypatch.setattr(ClipList, "load", broken)
    before = w.clips
    _open(w, other, keyframes=False)
    assert w.clips is not before and len(w.clips) == 0 and w.clips.video == os.path.abspath(other)
    assert "store unreadable" in w.status.currentMessage()
    monkeypatch.undo()
    _open(w, clip, keyframes=False)                          # the note belongs to the video it was about
    assert "could not be read" not in w.status.currentMessage()


def test_export_list_waits_for_the_keyframes_of_the_video(window, clip, tmp_path, fake_jobs):
    class Scanning:                                          # the open video's scan, still reading
        def __init__(self, token):
            self.token = token

        def isRunning(self):
            return True

        def cancel(self):
            pass

        def wait(self, *a):
            return True

    w = window
    _open(w, clip)
    w.seek_to(w.info.start_time + 1.0)
    w.set_in()
    w.add_to_list()
    kfs, w.info.keyframes = w.info.keyframes, []
    scan = Scanning(w._video_token)
    w._probe_threads.append(scan)
    try:
        w.export_list()
        assert not w.batch.running and fake_jobs == [] and "still being read" in w.status.currentMessage()
        assert w.clips.entries[0].status == PENDING
    finally:
        w._probe_threads.remove(scan)
    w.export_list()                                          # the scan ended without keyframes: re-encode
    assert w.batch.running and len(fake_jobs) == 1
    w.batch.cancel()
    w.info.keyframes = kfs


def test_a_finished_batch_leaves_quick_export_off_while_another_file_is_being_opened(window, clip, fake_jobs):
    w = window
    _open(w, clip)
    w.seek_to(w.info.start_time + 1.0)
    w.set_in()
    w.add_to_list()
    w.export_list()
    w._set_enabled(False)                                    # as open_file() does while it probes
    fake_jobs[0].end(True)
    assert not w.batch.running and not w.b_quick.isEnabled()
    w._set_enabled(True)
    assert w.b_quick.isEnabled()
