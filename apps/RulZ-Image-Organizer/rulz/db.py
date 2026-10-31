"""SQLite storage for folders, photos, faces and people.

Every thread opens its own ``Database``; the file is in WAL mode so the UI can
read while the background scanner writes.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import attributes, config

if TYPE_CHECKING:
    from .faces import DetectedFace, FaceAttrs

SCHEMA = """
CREATE TABLE IF NOT EXISTS folders(
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    added_at REAL
);
CREATE TABLE IF NOT EXISTS photos(
    id INTEGER PRIMARY KEY,
    folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
    path TEXT UNIQUE NOT NULL,
    size INTEGER,
    mtime REAL,
    fp TEXT,
    width INTEGER,
    height INTEGER,
    taken_at REAL,
    faces_done INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
CREATE TABLE IF NOT EXISTS people(
    id INTEGER PRIMARY KEY,
    name TEXT,
    cover_face_id INTEGER,
    hidden INTEGER NOT NULL DEFAULT 0,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS faces(
    id INTEGER PRIMARY KEY,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    x REAL, y REAL, w REAL, h REAL,          -- normalized to the oriented image
    score REAL,
    embedding BLOB,
    person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
    assigned_by TEXT,                        -- NULL (never grouped), 'auto' or 'user'
    ignored INTEGER NOT NULL DEFAULT 0,      -- user said "not a face"
    manual INTEGER NOT NULL DEFAULT 0,       -- box drawn by the user
    expression TEXT, expr_conf REAL,         -- see attributes.py
    yaw REAL, pitch REAL, roll REAL,
    attrs_done INTEGER NOT NULL DEFAULT 1,   -- 0: added before attributes existed, needs analysis
    pose TEXT, arms_up INTEGER,              -- see body.py
    body_done INTEGER NOT NULL DEFAULT 1     -- 0: added before body poses existed, needs analysis
);
CREATE TABLE IF NOT EXISTS rejections(
    face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    PRIMARY KEY(face_id, person_id)
);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS ix_photos_folder ON photos(folder_id);
CREATE INDEX IF NOT EXISTS ix_photos_taken ON photos(taken_at);
CREATE INDEX IF NOT EXISTS ix_photos_pending ON photos(faces_done);
CREATE INDEX IF NOT EXISTS ix_faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS ix_faces_person ON faces(person_id);
-- Covering index: a person's photo list and counts never touch the (large) face rows.
CREATE INDEX IF NOT EXISTS ix_faces_person_ignored_photo ON faces(person_id, ignored, photo_id);
"""

# Columns added after the first release: (name, definition). Existing faces get attrs_done=0.
MIGRATIONS = [
    ("expression", "TEXT"), ("expr_conf", "REAL"), ("yaw", "REAL"), ("pitch", "REAL"), ("roll", "REAL"),
    ("attrs_done", "INTEGER NOT NULL DEFAULT 0"),
    ("pose", "TEXT"), ("arms_up", "INTEGER"), ("body_done", "INTEGER NOT NULL DEFAULT 0"),
]


@dataclass
class Photo:
    id: int
    path: str
    taken_at: float
    width: int
    height: int
    size: int


@dataclass
class Face:
    id: int
    photo_id: int
    x: float
    y: float
    w: float
    h: float
    score: float
    person_id: int | None
    person_name: str | None
    assigned_by: str | None
    manual: int
    expression: str | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    pose: str | None = None
    arms_up: bool = False

    def tags(self) -> list[str]:
        return attributes.describe(self.expression, self.yaw, self.pitch, self.roll, self.pose, self.arms_up)


@dataclass
class Person:
    id: int
    name: str | None
    cover_face_id: int | None
    hidden: bool
    photo_count: int
    face_count: int

    @property
    def display_name(self) -> str:
        return self.name or "Unnamed"


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Database:
    def __init__(self, path: Path | str | None = None, timeout: float = 30):
        # The UI passes a short timeout: a click should fail with a message, not freeze the window.
        self.path = str(path or config.db_path())
        self.conn = sqlite3.connect(self.path, timeout=timeout, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(faces)")}
        for name, definition in MIGRATIONS:
            if name not in have:
                self.conn.execute(f"ALTER TABLE faces ADD COLUMN {name} {definition}")

    def close(self):
        self.conn.close()

    # ---- helpers -------------------------------------------------------------
    def q(self, sql: str, params=()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params=()):
        return self.conn.execute(sql, params).fetchone()

    def x(self, sql: str, params=()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def transaction(self):
        return _Tx(self.conn)

    # ---- settings ------------------------------------------------------------
    def get_setting(self, key: str, default=None):
        row = self.one("SELECT value FROM settings WHERE key=?", (key,))
        return row["value"] if row else default

    def set_setting(self, key: str, value) -> None:
        self.x("INSERT INTO settings(key,value) VALUES(?,?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))

    def match_threshold(self) -> float:
        try:
            return float(self.get_setting("match_threshold", config.DEFAULT_MATCH_THRESHOLD))
        except ValueError:
            return config.DEFAULT_MATCH_THRESHOLD

    # ---- folders -------------------------------------------------------------
    def folders(self) -> list[sqlite3.Row]:
        return self.q("SELECT f.id, f.path, (SELECT COUNT(*) FROM photos p WHERE p.folder_id=f.id) AS n "
                      "FROM folders f ORDER BY f.path COLLATE NOCASE")

    def add_folder(self, path: str) -> str | None:
        """Add a folder. Returns an error message, or None on success."""
        norm = os.path.normcase(os.path.abspath(path))
        existing = self.q("SELECT id, path FROM folders")
        for row in existing:
            other = os.path.normcase(os.path.abspath(row["path"]))
            if norm == other or norm.startswith(other.rstrip("\\/") + os.sep):
                return f"This folder is already in your library (inside {row['path']})."
        with self.transaction():
            cur = self.x("INSERT INTO folders(path, added_at) VALUES(?,?)",
                         (os.path.abspath(path), time.time()))
            new_id = cur.lastrowid
            # Absorb sub-folders that were added earlier.
            for row in existing:
                other = os.path.normcase(os.path.abspath(row["path"]))
                if other.startswith(norm.rstrip("\\/") + os.sep):
                    self.x("UPDATE photos SET folder_id=? WHERE folder_id=?", (new_id, row["id"]))
                    self.x("DELETE FROM folders WHERE id=?", (row["id"],))
        return None

    def remove_folder(self, folder_id: int) -> None:
        photo_ids = [r[0] for r in self.q("SELECT id FROM photos WHERE folder_id=?", (folder_id,))]
        self.delete_photos(photo_ids)
        self.x("DELETE FROM folders WHERE id=?", (folder_id,))
        self.cleanup_people()

    # ---- photos --------------------------------------------------------------
    def delete_photos(self, photo_ids: list[int]) -> None:
        if not photo_ids:
            return
        face_ids: list[int] = []
        with self.transaction():
            for chunk in _chunks(photo_ids, 500):
                marks = ",".join("?" * len(chunk))
                face_ids += [r[0] for r in self.q(f"SELECT id FROM faces WHERE photo_id IN ({marks})", chunk)]
                self.x(f"DELETE FROM photos WHERE id IN ({marks})", chunk)
        for pid in photo_ids:
            _unlink(config.thumb_path(pid))
        for fid in face_ids:
            _unlink(config.face_crop_path(fid))

    def photos(self, person_ids: list[int] | None = None) -> list[Photo]:
        """All photos newest first; optionally only photos containing *all* given people."""
        sql = "SELECT id, path, taken_at, width, height, size FROM photos"
        params: list = []
        if person_ids:
            # IN-subqueries walk only that person's faces (ix_faces_person_photo). A correlated EXISTS let
            # SQLite rescan all of a person's faces once per photo, which took minutes for big libraries.
            sql += " WHERE " + " AND ".join(
                "id IN (SELECT photo_id FROM faces WHERE person_id=? AND ignored=0)" for _ in person_ids)
            params += list(person_ids)
        sql += " ORDER BY taken_at DESC, id DESC"
        return [Photo(r["id"], r["path"], r["taken_at"] or 0, r["width"] or 0, r["height"] or 0, r["size"] or 0)
                for r in self.q(sql, params)]

    def photo(self, photo_id: int) -> Photo | None:
        r = self.one("SELECT id, path, taken_at, width, height, size FROM photos WHERE id=?", (photo_id,))
        return Photo(r["id"], r["path"], r["taken_at"] or 0, r["width"] or 0, r["height"] or 0, r["size"] or 0) if r else None

    def stats(self) -> dict:
        r = self.one("""SELECT
            (SELECT COUNT(*) FROM photos) AS photos,
            (SELECT COUNT(*) FROM photos WHERE faces_done=0) AS pending,
            (SELECT COUNT(*) FROM faces WHERE ignored=0) AS faces,
            (SELECT COUNT(*) FROM people) AS people,
            (SELECT COUNT(*) FROM people WHERE name IS NOT NULL) AS named""")
        return dict(r)

    # ---- faces ---------------------------------------------------------------
    _FACE_COLS = ("f.id, f.photo_id, f.x, f.y, f.w, f.h, f.score, f.person_id, p.name AS person_name, "
                  "f.assigned_by, f.manual, f.expression, f.yaw, f.pitch, f.roll, f.pose, f.arms_up")
    _FACE_SQL = f"SELECT {_FACE_COLS} FROM faces f LEFT JOIN people p ON p.id=f.person_id "

    @staticmethod
    def _face(r) -> Face:
        return Face(r["id"], r["photo_id"], r["x"], r["y"], r["w"], r["h"], r["score"] or 0,
                    r["person_id"], r["person_name"], r["assigned_by"], r["manual"],
                    r["expression"], r["yaw"], r["pitch"], r["roll"], r["pose"], bool(r["arms_up"]))

    def faces_in_photo(self, photo_id: int) -> list[Face]:
        return [self._face(r) for r in self.q(self._FACE_SQL + "WHERE f.photo_id=? AND f.ignored=0 ORDER BY f.x",
                                               (photo_id,))]

    def faces_of_person(self, person_id: int) -> list[tuple[Face, float]]:
        rows = self.q(f"SELECT {self._FACE_COLS}, ph.taken_at FROM faces f JOIN photos ph ON ph.id=f.photo_id "
                      "LEFT JOIN people p ON p.id=f.person_id WHERE f.person_id=? AND f.ignored=0 "
                      "ORDER BY ph.taken_at DESC, f.id DESC", (person_id,))
        return [(self._face(r), r["taken_at"] or 0) for r in rows]

    def face(self, face_id: int) -> Face | None:
        r = self.one(self._FACE_SQL + "WHERE f.id=?", (face_id,))
        return self._face(r) if r else None

    def insert_face(self, photo_id: int, face: "DetectedFace", manual: bool = False) -> int:
        a = face.attrs
        cur = self.x("INSERT INTO faces(photo_id, x, y, w, h, score, embedding, manual, expression, expr_conf, "
                     "yaw, pitch, roll, attrs_done, pose, arms_up, body_done) "
                     "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?,1)",
                     (photo_id, *map(float, face.box), float(face.score), to_blob(face.embedding), int(manual),
                      a.expression, a.expr_conf, a.yaw, a.pitch, a.roll, a.pose, int(a.arms_up)))
        return cur.lastrowid

    def delete_faces(self, face_ids: list[int]) -> None:
        """Delete faces and their crop files."""
        with self.transaction():
            for chunk in _chunks(face_ids, 500):
                self.x(f"DELETE FROM faces WHERE id IN ({','.join('?' * len(chunk))})", chunk)
            # Unlinked while the write lock is held, so no new face can reuse an id (and its file) meanwhile.
            for fid in face_ids:
                _unlink(config.face_crop_path(fid))

    def clear_photo_faces(self, photo_id: int) -> None:
        """Delete every face of a photo (it changed, or is being analyzed again), crop files included."""
        with self.transaction():
            self.delete_faces([r[0] for r in self.q("SELECT id FROM faces WHERE photo_id=?", (photo_id,))])

    def set_face_attrs(self, face_id: int, attrs: "FaceAttrs | None") -> None:
        if attrs is None:
            self.x("UPDATE faces SET attrs_done=1 WHERE id=?", (face_id,))
        else:
            self.x("UPDATE faces SET expression=?, expr_conf=?, yaw=?, pitch=?, roll=?, attrs_done=1 WHERE id=?",
                   (attrs.expression, attrs.expr_conf, attrs.yaw, attrs.pitch, attrs.roll, face_id))

    def set_face_pose(self, face_id: int, pose: str | None, arms_up: bool) -> None:
        self.x("UPDATE faces SET pose=?, arms_up=?, body_done=1 WHERE id=?", (pose, int(arms_up), face_id))

    def search_faces(self, person_ids: list[int] | None, conditions: list[str]) -> set[int]:
        """Photo ids having one face that is any of ``person_ids`` (None = anyone) and meets every condition.

        ``conditions`` are trusted SQL fragments from attributes.SEARCH_TERMS.
        """
        sql = "SELECT DISTINCT photo_id FROM faces WHERE ignored=0"
        params: list = []
        if person_ids is not None:
            sql += f" AND person_id IN ({','.join('?' * len(person_ids))})"
            params += person_ids
        for c in conditions:
            sql += f" AND ({c})"
        return {r[0] for r in self.q(sql, params)}

    def embeddings(self, where: str, params=()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (face_ids, person_ids (-1 for none), embeddings[N,128]) for faces matching ``where``."""
        rows = self.q(f"SELECT id, person_id, embedding FROM faces WHERE embedding IS NOT NULL AND ignored=0 AND {where}",
                      params)
        if not rows:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty((0, 128), np.float32)
        ids = np.fromiter((r[0] for r in rows), np.int64, len(rows))
        pids = np.fromiter((r[1] if r[1] is not None else -1 for r in rows), np.int64, len(rows))
        embs = np.stack([from_blob(r[2]) for r in rows])
        return ids, pids, embs

    # ---- people --------------------------------------------------------------
    def people(self, include_hidden: bool = False) -> list[Person]:
        rows = self.q("""
            SELECT p.id, p.name, p.cover_face_id, p.hidden,
                   COUNT(DISTINCT f.photo_id) AS photos, COUNT(f.id) AS faces
            FROM people p JOIN faces f ON f.person_id=p.id AND f.ignored=0
            """ + ("" if include_hidden else "WHERE p.hidden=0 ") + """
            GROUP BY p.id
            ORDER BY (p.name IS NULL), photos DESC, p.name COLLATE NOCASE""")
        return [Person(r["id"], r["name"], r["cover_face_id"], bool(r["hidden"]), r["photos"], r["faces"])
                for r in rows]

    def person(self, person_id: int) -> Person | None:
        r = self.one("""SELECT p.id, p.name, p.cover_face_id, p.hidden, c.photos, c.faces
                   FROM people p, (SELECT COUNT(DISTINCT photo_id) AS photos, COUNT(*) AS faces
                                   FROM faces WHERE person_id=? AND ignored=0) c
                   WHERE p.id=?""", (person_id, person_id))
        return Person(r["id"], r["name"], r["cover_face_id"], bool(r["hidden"]), r["photos"], r["faces"]) if r else None

    def person_by_name(self, name: str) -> Person | None:
        r = self.one("SELECT id FROM people WHERE name=? COLLATE NOCASE", (name.strip(),))
        return self.person(r[0]) if r else None

    def create_person(self, name: str | None = None) -> int:
        name = name.strip() if name else None
        return self.x("INSERT INTO people(name, created_at) VALUES(?,?)", (name or None, time.time())).lastrowid

    def rename_person(self, person_id: int, name: str | None) -> None:
        name = name.strip() if name else None
        self.x("UPDATE people SET name=? WHERE id=?", (name or None, person_id))

    def set_hidden(self, person_id: int, hidden: bool) -> None:
        self.x("UPDATE people SET hidden=? WHERE id=?", (int(hidden), person_id))

    def set_cover(self, person_id: int, face_id: int) -> None:
        self.x("UPDATE people SET cover_face_id=? WHERE id=?", (face_id, person_id))

    def merge_people(self, source_ids: list[int], target_id: int) -> None:
        with self.transaction():
            for sid in source_ids:
                if sid == target_id:
                    continue
                self.x("UPDATE faces SET person_id=? WHERE person_id=?", (target_id, sid))
                self.x("INSERT OR IGNORE INTO rejections(face_id, person_id) "
                       "SELECT face_id, ? FROM rejections WHERE person_id=?", (target_id, sid))
                # A face rejected from the source but now in the target would contradict itself.
                self.x("DELETE FROM rejections WHERE person_id=? AND face_id IN "
                       "(SELECT id FROM faces WHERE person_id=?)", (target_id, target_id))
                self.x("DELETE FROM people WHERE id=?", (sid,))

    def ungroup_person(self, person_id: int) -> None:
        """Dissolve a group: its faces become unassigned and will not be auto-regrouped together."""
        with self.transaction():
            self.x("UPDATE faces SET person_id=NULL, assigned_by='user' WHERE person_id=?", (person_id,))
            self.x("DELETE FROM people WHERE id=?", (person_id,))

    def assign_faces(self, face_ids: list[int], person_id: int) -> None:
        """User correction: these faces belong to ``person_id``."""
        with self.transaction():
            for fid in face_ids:
                row = self.one("SELECT person_id FROM faces WHERE id=?", (fid,))
                if row is None:
                    continue
                old = row[0]
                if old is not None and old != person_id:
                    self.x("INSERT OR IGNORE INTO rejections(face_id, person_id) VALUES(?,?)", (fid, old))
                self.x("DELETE FROM rejections WHERE face_id=? AND person_id=?", (fid, person_id))
                self.x("UPDATE faces SET person_id=?, assigned_by='user', ignored=0 WHERE id=?", (person_id, fid))
        self.cleanup_people()

    def unassign_faces(self, face_ids: list[int]) -> None:
        """User correction: these faces are *not* the person they are currently assigned to."""
        with self.transaction():
            for fid in face_ids:
                row = self.one("SELECT person_id FROM faces WHERE id=?", (fid,))
                if row is None:
                    continue
                if row[0] is not None:
                    self.x("INSERT OR IGNORE INTO rejections(face_id, person_id) VALUES(?,?)", (fid, row[0]))
                self.x("UPDATE faces SET person_id=NULL, assigned_by='user' WHERE id=?", (fid,))
        self.cleanup_people()

    def reject_faces(self, face_ids: list[int], person_id: int) -> None:
        """User said 'no' to a suggestion: never auto-assign these faces to ``person_id``."""
        with self.transaction():
            for fid in face_ids:
                self.x("INSERT OR IGNORE INTO rejections(face_id, person_id) VALUES(?,?)", (fid, person_id))

    def ignore_faces(self, face_ids: list[int]) -> None:
        """User said 'not a face' (or doesn't want it tracked)."""
        with self.transaction():
            for fid in face_ids:
                self.x("UPDATE faces SET ignored=1, person_id=NULL, assigned_by='user' WHERE id=?", (fid,))
        self.cleanup_people()

    def cleanup_people(self) -> None:
        """Delete unnamed people with no faces and fix stale cover faces."""
        self.x("DELETE FROM people WHERE name IS NULL AND NOT EXISTS("
               "SELECT 1 FROM faces f WHERE f.person_id=people.id AND f.ignored=0)")
        self.x("UPDATE people SET cover_face_id=NULL WHERE cover_face_id IS NOT NULL AND NOT EXISTS("
               "SELECT 1 FROM faces f WHERE f.id=people.cover_face_id AND f.person_id=people.id AND f.ignored=0)")
        self.x("""UPDATE people SET cover_face_id=(
                    SELECT f.id FROM faces f WHERE f.person_id=people.id AND f.ignored=0
                    ORDER BY (f.assigned_by='user') DESC, f.score * f.w * f.h DESC LIMIT 1)
                  WHERE cover_face_id IS NULL""")

    def rejected_pairs(self) -> set[tuple[int, int]]:
        return {(r[0], r[1]) for r in self.q("SELECT face_id, person_id FROM rejections")}


class _Tx:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.nested = False

    def __enter__(self):
        if self.conn.in_transaction:
            self.nested = True
        else:
            self.conn.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.nested:
            return False
        if exc_type is None:
            self.conn.execute("COMMIT")
        else:
            self.conn.execute("ROLLBACK")
        return False


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
