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
