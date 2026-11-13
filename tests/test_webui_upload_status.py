"""The dataset uploader's server side (files arrive one at a time, in pieces, into the folder a page's token names)
and the run strip's activity line (how long the stage has run, how far it is, how fresh the status and log are)."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from nicegui import app  # noqa: E402

import webui  # noqa: E402


# ---------------------------------------------------------------- write_upload_piece


def test_piece_writer_whole_file_in_pieces(tmp_path):
    ds = tmp_path / "ds"
    data = os.urandom(25)
    target, done = webui.write_upload_piece(ds, "photo.JPG", 0, 25, data[:10], tag="up0001")
    assert target == ds / "photo.JPG" and not done and not target.exists()
    part = webui.upload_part_path(target, "up0001")
    assert part.name.startswith(".rzup-up0001-") and part.name.endswith(".part") and part.read_bytes() == data[:10]
    assert webui.write_upload_piece(ds, "photo.JPG", 10, 25, data[10:20], tag="up0001") == (target, False)
    assert webui.write_upload_piece(ds, "photo.JPG", 20, 25, data[20:], tag="up0001") == (target, True)
    assert target.read_bytes() == data and not part.exists()
    # the dataset's image list never sees a part file
    webui.write_upload_piece(ds, "next.png", 0, 9, b"abc", tag="up0002")
    assert [p.name for p in webui.list_dataset_images(str(ds))] == ["photo.JPG"]


def test_piece_writer_out_of_step_twice_or_late(tmp_path):
    ds = tmp_path / "ds"
    webui.write_upload_piece(ds, "a.png", 0, 30, b"x" * 10, tag="upaaaa")
    with pytest.raises(webui.UploadOffsetError) as err:
        webui.write_upload_piece(ds, "a.png", 20, 30, b"y" * 10, tag="upaaaa")  # a piece was lost on the way
    assert err.value.have == 10
    with pytest.raises(webui.UploadOffsetError) as err:  # the first piece again (or late): never written twice,
        webui.write_upload_piece(ds, "a.png", 0, 30, b"x" * 10, tag="upaaaa")  # never cuts the part short
    assert err.value.have == 10
    with pytest.raises(webui.UploadOffsetError) as err:
        webui.write_upload_piece(ds, "b.png", 10, 30, b"y" * 10, tag="upaaaa")  # nothing of b.png here yet
    assert err.value.have == 0
    # a new upload of the same name (a new id) has a part file of its own and replaces the file at its end
    target, done = webui.write_upload_piece(ds, "a.png", 0, 4, b"zzzz", tag="upbbbb")
    assert done and target.read_bytes() == b"zzzz"
    assert webui.upload_part_path(target, "upaaaa").read_bytes() == b"x" * 10  # the other one's, untouched


def test_piece_writer_last_piece_again_and_empty_and_rename_retry(tmp_path):
    ds = tmp_path / "ds"
    webui.write_upload_piece(ds, "c.webp", 0, 6, b"abc", tag="upcccc")
    assert webui.write_upload_piece(ds, "c.webp", 3, 6, b"def", tag="upcccc")[1]
    # the answer to the last piece was lost: sending it again is fine, the file stays as it is
    target, done = webui.write_upload_piece(ds, "c.webp", 3, 6, b"def", tag="upcccc")
    assert done and target.read_bytes() == b"abcdef"
    with pytest.raises(webui.UploadOffsetError) as err:  # an earlier piece, late: the upload is complete
        webui.write_upload_piece(ds, "c.webp", 0, 6, b"abc", tag="upcccc")
    assert err.value.have == 6
    # another upload's last piece is never taken for this one's, even with the file there at the same size
    with pytest.raises(webui.UploadOffsetError):
        webui.write_upload_piece(ds, "c.webp", 3, 6, b"XYZ", tag="updddd")
    assert target.read_bytes() == b"abcdef"
    target, done = webui.write_upload_piece(ds, "empty.txt", 0, 0, b"", tag="upeeee")
    assert done and target.read_bytes() == b""
    # all bytes there but the rename failed (the page sends an empty piece at the end to finish it)
    part = webui.upload_part_path(ds / "d.png", "upffff")
    part.write_bytes(b"12345")
    assert webui.write_upload_piece(ds, "d.png", 5, 5, b"", tag="upffff") == (ds / "d.png", True)
    assert (ds / "d.png").read_bytes() == b"12345" and not part.exists()


def test_piece_writer_refuses(tmp_path):
    ds = tmp_path / "ds"
    for name in ("notes.exe", ".hidden.png", "", "..", "folder/"):
        with pytest.raises(ValueError):
            webui.write_upload_piece(ds, name, 0, 3, b"abc", tag="upgggg")
    with pytest.raises(ValueError):
        webui.write_upload_piece(ds, "a.png", 0, 2, b"abc", tag="upgggg")  # longer than the file
    with pytest.raises(ValueError):
        webui.write_upload_piece(ds, "a.png", -1, 3, b"abc", tag="upgggg")
    for bad in ("", "abc", "../../x1", "a.b.c.d.e", "x" * 41):
        with pytest.raises(ValueError):
            webui.write_upload_piece(ds, "a.png", 0, 3, b"abc", tag=bad)
    target, _ = webui.write_upload_piece(ds, "../../escape.png", 0, 3, b"abc", tag="upgggg")
    assert target == ds / "escape.png" and target.is_file()  # the name only, never a folder
    assert not (tmp_path / "escape.png").exists()


def test_piece_writer_refuses_a_linked_part_file(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"keep me")
    link = webui.upload_part_path(ds / "a.png", "uphhhh")
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links need extra rights on this system")
    with pytest.raises(ValueError):
        webui.write_upload_piece(ds, "a.png", 0, 3, b"abc", tag="uphhhh")
    assert outside.read_bytes() == b"keep me"


def test_clean_upload_parts_and_slots(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    old, new, keep = ds / ".a.png.t.part", ds / ".b.png.t.part", ds / "c.png"
    for p in (old, new, keep):
        p.write_bytes(b"x")
    os.utime(old, (time.time() - 7 * 3600,) * 2)
    os.utime(keep, (time.time() - 7 * 3600,) * 2)
    assert webui.clean_upload_parts(ds) == 1
    assert not old.exists() and new.exists() and keep.exists()
    assert webui.clean_upload_parts(tmp_path / "missing") == 0
    tokens = [webui.upload_slot(ds) for _ in range(webui.UPLOAD_SLOTS_KEPT + 5)]
    assert len(set(tokens)) == len(tokens) and len(webui._upload_slots) <= webui.UPLOAD_SLOTS_KEPT
    assert tokens[0] not in webui._upload_slots and webui._upload_slots[tokens[-1]] == str(ds)


def test_upload_errors_lasting_or_passing(tmp_path):
    import errno

    assert webui.upload_error_is_lasting(OSError(errno.ENOSPC, "No space left on device"))
    assert webui.upload_error_is_lasting(OSError(errno.ENAMETOOLONG, "File name too long"))
    assert webui.upload_error_is_lasting(PermissionError(errno.EACCES, "Permission denied"))
    locked = PermissionError(errno.EACCES, "used by another process")
    locked.winerror = 32  # Windows: another program has the file open, which passes
    assert not webui.upload_error_is_lasting(locked)
    assert not webui.upload_error_is_lasting(OSError(errno.EIO, "I/O error"))
    assert webui._WINDOWS_BAD_NAME.search("a.txt:b.png") and webui._WINDOWS_BAD_NAME.search('x?.jpg')
    assert webui._WINDOWS_BAD_NAME.search("photo.png.") and not webui._WINDOWS_BAD_NAME.search("PXL_1 (2).jpg")
    if os.name == "nt":
        with pytest.raises(ValueError, match="Windows cannot store"):
            webui.write_upload_piece(tmp_path / "ds", "a.txt:b.png", 0, 3, b"abc", tag="upiiii")
    else:  # a name near the file system's limit: the part file's own name stays short, so it still uploads
        long_name = "x" * 245 + ".png"
        target, done = webui.write_upload_piece(tmp_path / "ds", long_name, 0, 3, b"abc", tag="upjjjj")
        assert done and target.read_bytes() == b"abc"


# ---------------------------------------------------------------- the /rz-upload route


def test_upload_route(tmp_path):
    ds = tmp_path / "ds"
    token = webui.upload_slot(ds)
    client = TestClient(app)
    url = f"/rz-upload/{token}"
    data = os.urandom(3000)

    def piece(name, offset, total, body, uid="lzq81abc"):
        return client.post(url, params={"name": name, "offset": offset, "total": total, "uid": uid}, content=body)

    r = piece("p.jpg", 0, 3000, data[:1000])
    assert r.status_code == 200 and r.json() == {"done": False, "have": 1000}
    r = piece("p.jpg", 2000, 3000, data[2000:])
    assert r.status_code == 409 and r.json()["have"] == 1000  # out of step: go on from 1000
    assert piece("p.jpg", 1000, 3000, data[1000:2000]).json() == {"done": False, "have": 2000}
    assert piece("p.jpg", 2000, 3000, data[2000:]).json() == {"done": True, "have": 3000}
    assert (ds / "p.jpg").read_bytes() == data
    r = piece("virus.exe", 0, 3, b"abc")
    assert r.status_code == 400 and "not an image" in r.json()["error"]
    r = piece("q.png", 0, 3, b"abc", uid="../x")
    assert r.status_code == 400 and not (ds / "q.png").exists()
    r = client.post("/rz-upload/nope", params={"name": "p.jpg", "offset": 0, "total": 3, "uid": "lzq81abc"},
                    content=b"abc")
    assert r.status_code == 404 and not (tmp_path / "p.jpg").exists()
    big = b"x" * (webui.UPLOAD_PIECE_MAX + 1)
    r = piece("big.png", 0, len(big), big)
    assert r.status_code == 413 and not (ds / "big.png").exists()


def test_upload_route_marks_lasting_errors(tmp_path, monkeypatch):
    import errno

    token = webui.upload_slot(tmp_path / "ds")
    client = TestClient(app)

    def fail(err):
        def write(*a, **k):
            raise err
        return write

    for err, fatal in ((OSError(errno.ENOSPC, "No space left on device"), True), (OSError(errno.EIO, "I/O"), False)):
        monkeypatch.setattr(webui, "write_upload_piece", fail(err))
        r = client.post(f"/rz-upload/{token}", params={"name": "a.png", "offset": 0, "total": 3, "uid": "lzq81abc"},
                        content=b"abc")
        assert r.status_code == 500 and r.json()["fatal"] is fatal


# ---------------------------------------------------------------- the run strip's activity line


def test_duration_and_last_log_line(tmp_path):
    assert webui.duration_text(0) == "0 s" and webui.duration_text(59.9) == "59 s"
    assert webui.duration_text(76) == "1 min 16 s" and webui.duration_text(3 * 3600 + 5 * 60) == "3 h 05 min"
    assert webui.duration_text(None) == "0 s"
    log = tmp_path / "train.log"
    assert webui.last_log_line(log) == ""
    log.write_text("15:35:00 INFO one\n15:36:16 WARNING Face reference x.png: no face detected\n\n  \n",
                   encoding="utf-8")
    assert webui.last_log_line(log) == "WARNING Face reference x.png: no face detected"
    log.write_text("x" * 50000 + "\n12:00:01 INFO the end", encoding="utf-8")
    assert webui.last_log_line(log) == "INFO the end"  # read from the end only


def test_run_activity(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    now = time.time()
    (job / "status.json").write_text("{}", encoding="utf-8")
    (job / "train.log").write_text("15:36:16 INFO Face references: 111 of 173 images had a face\n", encoding="utf-8")
    os.utime(job / "status.json", (now - 30, now - 30))
    os.utime(job / "train.log", (now - 5, now - 5))
    st = {"stage": "finding the faces in the face references (87 of 173)", "stage_since": now - 76,
          "stage_frac": 0.5}
    a = webui.run_activity(job, st, now, cpu=104.2)
    assert a["since"] == pytest.approx(76) and a["frac"] == 0.5 and a["quiet"] == pytest.approx(5, abs=0.01)
    assert not a["stale"] and a["line"] == "INFO Face references: 111 of 173 images had a face"
    assert a["text"] == "1 min 16 s in this stage · 50% of it done · trainer CPU 104% · last update 5 s ago"
    # an older trainer (no stage_since / stage_frac), nothing written for 11 minutes
    os.utime(job / "status.json", (now - 700, now - 700))
    os.utime(job / "train.log", (now - 660, now - 660))
    a = webui.run_activity(job, {"stage": "loading transformer"}, now)
    assert a["since"] is None and a["frac"] is None and a["stale"]
    assert a["text"].startswith("no status update or log line for 11 min 00 s")
    a = webui.run_activity(tmp_path / "gone", {"stage_frac": "bad", "stage_since": "x"}, now)
    assert a == {"since": None, "frac": None, "quiet": None, "stale": False, "text": "", "line": ""}


def test_job_cpu_percent_without_job(monkeypatch):
    monkeypatch.setattr(webui, "_proc", None)
    monkeypatch.setattr(webui, "_adopted", None)
    assert webui.job_cpu_percent() is None


def test_job_cpu_percent_of_this_process(monkeypatch):
    import psutil

    class Fake:
        pid = os.getpid()

    monkeypatch.setattr(webui, "_proc", Fake())
    webui._cpu_seen.clear()
    webui._cpu_last.update(pid=None, at=0.0, value=None)
    t = time.time()
    assert webui.job_cpu_percent(t) is None  # the first look: nothing to measure against yet
    sum(i * i for i in range(200000))
    assert webui.job_cpu_percent(t + 0.5) is None  # another page a moment later: the same reading
    value = webui.job_cpu_percent(t + webui.CPU_SAMPLE_SECONDS + 0.1)
    assert value is not None and value >= 0
    assert webui.job_cpu_percent(t + webui.CPU_SAMPLE_SECONDS + 0.2) == value
    assert psutil.pid_exists(os.getpid())


# ---------------------------------------------------------------- the trainer's stage timing


def test_status_stage_since_and_frac(tmp_path, monkeypatch):
    trainer = pytest.importorskip("rulztrainer.trainer")
    clock = [1000.0]
    monkeypatch.setattr(trainer.time, "time", lambda: clock[0])
    st = trainer.Status(tmp_path / "status.json")
    assert st.data["stage_since"] == 1000.0 and st.data["stage_frac"] is None
    clock[0] = 1010.0
    st.update(stage="reward fine-tuning 3/100: rendering 4 images", stage_frac=0.03)
    assert st.data["stage_since"] == 1010.0 and st.data["stage_frac"] == 0.03
    clock[0] = 1020.0
    st.update(stage="reward fine-tuning 4/100: rendering 4 images")  # the same stage, its counts aside
    assert st.data["stage_since"] == 1010.0 and st.data["stage_frac"] == 0.03
    st.update(loss=0.2)  # no stage: nothing changes
    assert st.data["stage_since"] == 1010.0
    clock[0] = 1030.0
    st.update(stage="training")
    assert st.data["stage_since"] == 1030.0 and st.data["stage_frac"] is None
    on_disk = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert on_disk["stage_since"] == 1030.0 and on_disk["stage"] == "training"
    assert trainer.stage_key("copying 1.5 GB (3 of 9)") == trainer.stage_key("copying 12.25 GB (4 of 9)")


def test_stage_progress_keeps_the_fraction(tmp_path):
    trainer = pytest.importorskip("rulztrainer.trainer")
    tr = trainer.Trainer.__new__(trainer.Trainer)
    tr.status = trainer.Status(tmp_path / "status.json")
    tr._stage_progress("downloading model.safetensors", 0.42)
    assert tr.status.data["stage"] == "downloading model.safetensors" and tr.status.data["stage_frac"] == 0.42
    tr._stage_progress("Loading face model buffalo_l...", None)
    assert tr.status.data["stage_frac"] is None
    tr._stage_progress("x", 7)
    assert tr.status.data["stage_frac"] == 1.0
    tr._stage_progress("x", "n/a")
    assert tr.status.data["stage_frac"] is None


def test_face_references_report_progress(monkeypatch, tmp_path):
    facesim = pytest.importorskip("rulztrainer.facesim")
    np = pytest.importorskip("numpy")

    class Face:
        def __init__(self):
            self.bbox = np.array([10, 10, 50, 50], dtype=np.float32)
            self.kps = None
            self.det_score = 0.9
            self.normed_embedding = np.ones(4, dtype=np.float32) / 2

    seen = []
    s = facesim.FaceScorer.__new__(facesim.FaceScorer)
    s.progress = lambda text, frac: seen.append((text, frac))
    s.gold = None
    paths = [tmp_path / f"r{i}.png" for i in range(3)]
    monkeypatch.setattr(facesim, "expand_refs", lambda p: list(p))
    monkeypatch.setattr(s, "faces", lambda p: [] if p.name == "r1.png" else [Face()], raising=False)
    monkeypatch.setattr(facesim, "image_size", lambda p: (100, 100))
    info = s.set_references(paths)
    assert [i["faces"] for i in info] == [1, 0, 1]
    assert seen == [(f"finding the faces in the face references ({i + 1} of 3)", i / 3) for i in range(3)]
    s.progress = lambda text, frac: 1 / 0  # a broken callback never stops the loop
    assert len(s.set_references(paths)) == 3
