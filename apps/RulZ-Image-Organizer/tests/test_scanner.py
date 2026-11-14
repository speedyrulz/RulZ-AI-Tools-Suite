import os
import sqlite3
import threading

import numpy as np
import pytest
from PIL import Image

from rulz import config, imaging, scanner
from rulz.faces import DetectedFace, FaceAttrs


def _photos(folder, n, size=(64, 48)):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", size, (i * 7 % 255, 90, 160)).save(folder / f"img{i:03d}.jpg")
    return folder


def _never():
    return False


def test_sync_lets_another_connection_write_while_fingerprinting(db, tmp_path, monkeypatch):
    folder = _photos(tmp_path / "pics", 5)
    db.add_folder(str(folder))
    other = sqlite3.connect(db.path, timeout=0, isolation_level=None)  # fails at once if the lock is held
    writes = []
    real = imaging.fingerprint

    def fingerprint(path, size):
        other.execute("INSERT INTO settings(key, value) VALUES(?, '1') "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (f"probe-{len(writes)}",))
        writes.append(path)
        return real(path, size)

    monkeypatch.setattr(imaging, "fingerprint", fingerprint)
    assert scanner.sync_files(db, lambda *a: None, _never) == 5
    assert len(writes) == 5
    other.close()


def test_sync_writes_in_batches(db, tmp_path, monkeypatch):
    folder = _photos(tmp_path / "pics", 7)
    db.add_folder(str(folder))
    monkeypatch.setattr(scanner, "BATCH", 3)
    commits = []
    real_tx = db.transaction

    def tx():
        t = real_tx()
        if not db.conn.in_transaction:
            commits.append(1)
        return t

    monkeypatch.setattr(db, "transaction", tx)
    scanner.sync_files(db, lambda *a: None, _never)
    assert db.one("SELECT COUNT(*) FROM photos")[0] == 7
    assert len(commits) >= 3  # 3 + 3 + 1, not one transaction for everything


def test_sync_stops_during_fingerprinting_and_keeps_progress(db, tmp_path, monkeypatch):
    folder = _photos(tmp_path / "pics", 10)
    db.add_folder(str(folder))
    calls = []
    real = imaging.fingerprint

    def fingerprint(path, size):
        calls.append(path)
        return real(path, size)

    monkeypatch.setattr(imaging, "fingerprint", fingerprint)
    with pytest.raises(scanner.Cancelled):
        scanner.sync_files(db, lambda *a: None, lambda: len(calls) >= 4)
    assert len(calls) == 4
    assert db.one("SELECT COUNT(*) FROM photos")[0] == 4  # flushed before stopping
    assert scanner.sync_files(db, lambda *a: None, _never) == 6  # the next scan picks up the rest


def test_sync_moved_and_changed_files(db, tmp_path):
    folder = _photos(tmp_path / "pics", 3)
    db.add_folder(str(folder))
    scanner.sync_files(db, lambda *a: None, _never)
    moved_id = db.one("SELECT id FROM photos WHERE path LIKE '%img000.jpg'")[0]
    (folder / "img000.jpg").rename(folder / "renamed.jpg")
    changed_id = db.one("SELECT id FROM photos WHERE path LIKE '%img001.jpg'")[0]
    db.x("UPDATE photos SET faces_done=1 WHERE id=?", (changed_id,))
    Image.new("RGB", (80, 80), (1, 2, 3)).save(folder / "img001.jpg")
    db.x("UPDATE photos SET mtime=mtime-100 WHERE id=?", (changed_id,))
    assert scanner.sync_files(db, lambda *a: None, _never) == 1
    assert db.one("SELECT path FROM photos WHERE id=?", (moved_id,))[0].endswith("renamed.jpg")
    assert db.one("SELECT faces_done FROM photos WHERE id=?", (changed_id,))[0] == 0


def _tagged_face(db, photo_id, person_id):
    emb = np.ones(128, np.float32) / np.sqrt(128)
    fid = db.insert_face(photo_id, DetectedFace((0.1, 0.1, 0.3, 0.3), 0.9, emb, FaceAttrs()))
    db.x("UPDATE faces SET person_id=?, assigned_by='user' WHERE id=?", (person_id, fid))
    return fid


def test_sync_date_only_change_keeps_faces(db, tmp_path):
    # A copy/backup tool, or a DST shift on a FAT32 card, moves the file date but not the picture.
    folder = _photos(tmp_path / "pics", 1)
    db.add_folder(str(folder))
    scanner.sync_files(db, lambda *a: None, _never)
    pid = db.one("SELECT id FROM photos")[0]
    db.x("UPDATE photos SET faces_done=1 WHERE id=?", (pid,))
    fid = _tagged_face(db, pid, db.create_person("Mom"))
    f = folder / "img000.jpg"
    st = f.stat()
    os.utime(f, (st.st_atime, st.st_mtime + 3600))
    assert scanner.sync_files(db, lambda *a: None, _never) == 0
    assert db.one("SELECT assigned_by FROM faces WHERE id=?", (fid,))[0] == "user"
    row = db.one("SELECT faces_done, mtime FROM photos WHERE id=?", (pid,))
    assert row[0] == 1 and abs(row[1] - (st.st_mtime + 3600)) < 1  # the new date is stored
    assert scanner.sync_files(db, lambda *a: None, _never) == 0     # and not looked at again


def test_sync_keeps_photos_in_a_folder_it_cannot_read(db, tmp_path, monkeypatch):
    folder = _photos(tmp_path / "pics", 1)
    _photos(folder / "sub", 2)
    db.add_folder(str(folder))
    scanner.sync_files(db, lambda *a: None, _never)
    assert db.one("SELECT COUNT(*) FROM photos")[0] == 3
    real = os.scandir

    def scandir(path):
        if os.path.basename(path) == "sub":
            raise PermissionError(13, "Access is denied", path)
        return real(path)

    monkeypatch.setattr(scanner.os, "scandir", scandir)
    scanner.sync_files(db, lambda *a: None, _never)
    assert db.one("SELECT COUNT(*) FROM photos")[0] == 3  # not deleted along with their tags
    monkeypatch.setattr(scanner.os, "scandir", real)
    for f in (folder / "sub").iterdir():
        f.unlink()
    scanner.sync_files(db, lambda *a: None, _never)
    assert db.one("SELECT COUNT(*) FROM photos")[0] == 1  # really gone once the folder reads again


def test_sync_renamed_duplicates_keep_their_rows(db, tmp_path):
    folder = tmp_path / "pics"
    folder.mkdir()
    Image.new("RGB", (64, 48), (10, 90, 160)).save(folder / "a.jpg")
    (folder / "b.jpg").write_bytes((folder / "a.jpg").read_bytes())
    db.add_folder(str(folder))
    scanner.sync_files(db, lambda *a: None, _never)
    ids = {os.path.basename(r["path"]): r["id"] for r in db.q("SELECT id, path FROM photos")}
    (folder / "sub").mkdir()
    (folder / "a.jpg").rename(folder / "sub" / "a.jpg")  # same name, other place
    (folder / "b.jpg").rename(folder / "B2.jpg")
    assert scanner.sync_files(db, lambda *a: None, _never) == 0
    now = {os.path.basename(r["path"]): r["id"] for r in db.q("SELECT id, path FROM photos")}
    assert now == {"a.jpg": ids["a.jpg"], "B2.jpg": ids["b.jpg"]}


def test_extra_copy_does_not_take_a_moved_duplicates_row(db, tmp_path):
    folder = tmp_path / "pics"
    folder.mkdir()
    Image.new("RGB", (64, 48), (10, 90, 160)).save(folder / "a.jpg")
    (folder / "b.jpg").write_bytes((folder / "a.jpg").read_bytes())
    db.add_folder(str(folder))
    scanner.sync_files(db, lambda *a: None, _never)
    ids = {os.path.basename(r["path"]): r["id"] for r in db.q("SELECT id, path FROM photos")}
    (folder / "sub").mkdir()
    (folder / "a.jpg").rename(folder / "sub" / "a.jpg")
    (folder / "b.jpg").rename(folder / "sub" / "b.jpg")
    (folder / "c.jpg").write_bytes((folder / "sub" / "a.jpg").read_bytes())  # a third copy, seen first
    scanner.sync_files(db, lambda *a: None, _never)
    now = {os.path.basename(r["path"]): r["id"] for r in db.q("SELECT id, path FROM photos")}
    assert now["a.jpg"] == ids["a.jpg"] and now["b.jpg"] == ids["b.jpg"]
    assert now["c.jpg"] not in ids.values()


def test_unreadable_file_keeps_its_row_and_tags(db, tmp_path, monkeypatch):
    folder = _photos(tmp_path / "pics", 2)
    db.add_folder(str(folder))
    scanner.sync_files(db, lambda *a: None, _never)
    pid = db.one("SELECT id FROM photos WHERE path LIKE '%img000.jpg'")[0]
    db.x("UPDATE photos SET faces_done=1")
    fid = _tagged_face(db, pid, db.create_person("Mom"))
    (folder / "img000.jpg").write_bytes((folder / "img000.jpg").read_bytes() + b"\0")  # size changed
    (folder / "img001.jpg").rename(folder / "moved.jpg")
    monkeypatch.setattr(imaging, "fingerprint", lambda path, size: "")  # e.g. locked by another program
    scanner.sync_files(db, lambda *a: None, _never)
    assert db.one("SELECT person_id FROM faces WHERE id=?", (fid,)) is not None  # not cleared
    assert db.one("SELECT COUNT(*) FROM photos")[0] == 2  # the moved one isn't deleted (or added) yet


def test_process_pending_reports_rewritten_files(db, tmp_path):
    folder = _photos(tmp_path / "pics", 3)
    db.add_folder(str(folder))
    scanner.sync_files(db, lambda *a: None, _never)
    calls = []
    scanner.process_pending(db, lambda *a: None, lambda: None, _never, engine_factory=_FakeEngine,
                            rewritten=lambda p, f: calls.append((p, f)))
    photos = sorted(p for c in calls for p in c[0])
    faces = sorted(f for c in calls for f in c[1])
    assert photos == sorted(r[0] for r in db.q("SELECT id FROM photos"))
    assert faces == sorted(r[0] for r in db.q("SELECT id FROM faces"))


def test_face_tagged_before_analysis_survives_the_scan(db, tmp_path):
    folder = _photos(tmp_path / "pics", 2)
    db.add_folder(str(folder))
    scanner.sync_files(db, lambda *a: None, _never)
    p1, p2 = [r[0] for r in db.q("SELECT id FROM photos ORDER BY id")]
    mom = db.create_person("Mom")
    emb = np.ones(128, np.float32) / np.sqrt(128)
    # Tagged in the viewer while the photos were still waiting: one on the face the detector will find
    # (the fake engine finds (0.25, 0.25, 0.5, 0.5)), one elsewhere.
    same = db.insert_face(p1, DetectedFace((0.3, 0.3, 0.4, 0.4), 0, emb, FaceAttrs()), manual=True)
    other = db.insert_face(p2, DetectedFace((0.0, 0.0, 0.1, 0.1), 0, emb, FaceAttrs()), manual=True)
    db.assign_faces([same, other], mom)
    scanner.process_pending(db, lambda *a: None, lambda: None, _never, engine_factory=_FakeEngine)
    assert db.one("SELECT person_id FROM faces WHERE id=?", (same,))[0] == mom
    assert db.one("SELECT person_id FROM faces WHERE id=?", (other,))[0] == mom
    assert db.one("SELECT COUNT(*) FROM faces WHERE photo_id=?", (p1,))[0] == 1  # not detected twice
    assert db.one("SELECT COUNT(*) FROM faces WHERE photo_id=?", (p2,))[0] == 2


def test_changed_file_drops_hand_drawn_faces_too(db, tmp_path):
    folder = _photos(tmp_path / "pics", 1)
    db.add_folder(str(folder))
    scanner.sync_files(db, lambda *a: None, _never)
    pid = db.one("SELECT id FROM photos")[0]
    db.x("UPDATE photos SET faces_done=1 WHERE id=?", (pid,))
    emb = np.ones(128, np.float32) / np.sqrt(128)
    db.insert_face(pid, DetectedFace((0.3, 0.3, 0.4, 0.4), 0, emb, FaceAttrs()), manual=True)
    Image.new("RGB", (80, 80), (1, 2, 3)).save(folder / "img000.jpg")  # new picture: old boxes are void
    db.x("UPDATE photos SET mtime=mtime-100 WHERE id=?", (pid,))
    scanner.sync_files(db, lambda *a: None, _never)
    assert db.one("SELECT COUNT(*) FROM faces WHERE photo_id=?", (pid,))[0] == 0


def test_data_dir_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RULZ_DATA_DIR", str(tmp_path / "one"))
    assert config.thumb_file(3).startswith(str(tmp_path / "one") + "\\thumbs\\03\\")
    monkeypatch.setenv("RULZ_DATA_DIR", str(tmp_path / "two"))
    assert config.thumb_file(3).startswith(str(tmp_path / "two") + "\\thumbs\\03\\")
    assert config.data_dir() == tmp_path / "two" and (tmp_path / "two").is_dir()


def test_backfill_loads_models_only_when_needed(db):
    def factory():
        raise AssertionError("models loaded with nothing to do")

    assert scanner.backfill_attributes(db, factory, lambda *a: None, lambda: None, _never) == 0


class _FakeEngine:
    threads: set = set()
    created = 0
    lock = threading.Lock()

    def __init__(self):
        with _FakeEngine.lock:
            _FakeEngine.created += 1

    def detect(self, bgr):
        _FakeEngine.threads.add(threading.get_ident())
        return [DetectedFace((0.25, 0.25, 0.5, 0.5), 0.9, np.ones(128, np.float32) / np.sqrt(128), FaceAttrs())]


def test_process_pending_detects_on_pool_threads(db, tmp_path):
    folder = _photos(tmp_path / "pics", 6)
    db.add_folder(str(folder))
    scanner.sync_files(db, lambda *a: None, _never)
    _FakeEngine.threads, _FakeEngine.created = set(), 0
    n = scanner.process_pending(db, lambda *a: None, lambda: None, _never, engine_factory=_FakeEngine)
    assert n == 6
    assert threading.get_ident() not in _FakeEngine.threads
    assert 1 <= _FakeEngine.created <= 4  # one per pool thread, not one per photo
    assert db.one("SELECT COUNT(*) FROM photos WHERE faces_done=0")[0] == 0
    face_ids = [r[0] for r in db.q("SELECT id FROM faces")]
    assert len(face_ids) == 6
    assert all(config.face_crop_path(f).exists() for f in face_ids)
    assert all(config.thumb_path(r[0]).exists() for r in db.q("SELECT id FROM photos"))


def test_face_crops_removed_with_faces(db, tmp_path):
    folder = _photos(tmp_path / "pics", 2)
    db.add_folder(str(folder))
    scanner.sync_files(db, lambda *a: None, _never)
    scanner.process_pending(db, lambda *a: None, lambda: None, _never, engine_factory=_FakeEngine)
    (f1, p1), (f2, p2) = [tuple(r) for r in db.q("SELECT id, photo_id FROM faces ORDER BY id")]
    assert config.face_crop_path(f1).exists() and config.face_crop_path(f2).exists()
    db.delete_faces([f1])
    assert not config.face_crop_path(f1).exists()
    assert db.one("SELECT COUNT(*) FROM faces WHERE id=?", (f1,))[0] == 0
    db.clear_photo_faces(p2)
    assert not config.face_crop_path(f2).exists()
    # Re-analysing a photo replaces its crop files rather than leaving the old ones behind.
    db.x("UPDATE photos SET faces_done=0")
    scanner.process_pending(db, lambda *a: None, lambda: None, _never, engine_factory=_FakeEngine)
    crops = list((config.data_dir() / "faces").rglob("*.jpg"))
    assert sorted(int(c.stem) for c in crops) == sorted(r[0] for r in db.q("SELECT id FROM faces"))
