"""SQLite result cache keyed on (image_hash, rubric_hash, model_id), plus the rubric-expansion cache."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def rubric_hash(rubric: str) -> str:
    return hashlib.sha256(rubric.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TagResult:
    match: bool  # raw model answer, before thresholding
    confidence: float
    evidence: str


class ResultCache:
    def __init__(self, path: Path | str):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS results (
                image_hash  TEXT NOT NULL,
                rubric_hash TEXT NOT NULL,
                model_id    TEXT NOT NULL,
                match       INTEGER NOT NULL,
                confidence  REAL NOT NULL,
                evidence    TEXT NOT NULL,
                created_at  REAL NOT NULL,
                PRIMARY KEY (image_hash, rubric_hash, model_id)
            );
            CREATE TABLE IF NOT EXISTS expansions (
                tag_text   TEXT NOT NULL,
                model_id   TEXT NOT NULL,
                tag_id     TEXT NOT NULL,
                rubric     TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (tag_text, model_id)
            );
            -- original bytes -> the hash of what prepare_image made of them, so a cached image
            -- doesn't have to be decoded and resized again just to find its results
            CREATE TABLE IF NOT EXISTS prepared (
                raw_hash      TEXT NOT NULL,
                prep_key      TEXT NOT NULL,
                prepared_hash TEXT NOT NULL,
                created_at    REAL NOT NULL,
                PRIMARY KEY (raw_hash, prep_key)
            );
            """
        )
        self._db.commit()

    def get_many(self, image_hash: str, rubric_hashes: Iterable[str], model_id: str) -> dict[str, TagResult]:
        hashes = list(dict.fromkeys(rubric_hashes))
        if not hashes:
            return {}
        placeholders = ",".join("?" * len(hashes))
        with self._lock:
            rows = self._db.execute(
                f"SELECT rubric_hash, match, confidence, evidence FROM results "
                f"WHERE image_hash=? AND model_id=? AND rubric_hash IN ({placeholders})",
                [image_hash, model_id, *hashes],
            ).fetchall()
        return {r[0]: TagResult(bool(r[1]), float(r[2]), r[3]) for r in rows}

    def put_many(self, image_hash: str, model_id: str, items: dict[str, TagResult]) -> None:
        if not items:
            return
        now = time.time()
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?)",
                [(image_hash, rh, model_id, int(r.match), r.confidence, r.evidence, now) for rh, r in items.items()],
            )
            self._db.commit()

    def get_prepared(self, raw_hash: str, prep_key: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT prepared_hash FROM prepared WHERE raw_hash=? AND prep_key=?", (raw_hash, prep_key)
            ).fetchone()
        return row[0] if row else None

    def put_prepared(self, raw_hash: str, prep_key: str, prepared_hash: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO prepared VALUES (?,?,?,?)", (raw_hash, prep_key, prepared_hash, time.time())
            )
            self._db.commit()

    def get_expansion(self, tag_text: str, model_id: str) -> tuple[str, str] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT tag_id, rubric FROM expansions WHERE tag_text=? AND model_id=?", (tag_text, model_id)
            ).fetchone()
        return (row[0], row[1]) if row else None

    def put_expansion(self, tag_text: str, model_id: str, tag_id: str, rubric: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO expansions VALUES (?,?,?,?,?)", (tag_text, model_id, tag_id, rubric, time.time())
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
