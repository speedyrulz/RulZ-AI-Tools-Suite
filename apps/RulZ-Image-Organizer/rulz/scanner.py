"""Library indexing: find image files, make thumbnails, detect faces and group them.

UI-independent; the Qt worker in ``ui/worker.py`` drives it with callbacks.
"""
from __future__ import annotations

import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from . import clustering, config, imaging
from .db import Database
from .faces import FaceEngine

Progress = Callable[[int, int, str], None]


class Cancelled(Exception):
    pass


def _walk(root: str, should_stop: Callable[[], bool]):
    stack = [root]
    while stack:
        if should_stop():
            raise Cancelled
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    name = e.name
                    if name.startswith(".") or name.startswith("$"):
                        continue
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif os.path.splitext(name)[1].lower() in config.IMAGE_EXTS:
                            st = e.stat()
                            yield e.path, st.st_size, st.st_mtime
                    except OSError:
                        continue
        except OSError:
            continue


def sync_files(db: Database, progress: Progress, should_stop: Callable[[], bool]) -> int:
    """Bring the photos table in line with what's on disk. Returns number of new/changed photos."""
    changed = 0
    for folder in db.folders():
        fid, root = folder["id"], folder["path"]
        if not os.path.isdir(root):
            continue  # drive unplugged? keep what we have
        progress(0, 0, f"Looking for photos in {root}")
        known = {r["path"]: r for r in db.q("SELECT id, path, size, mtime, fp FROM photos WHERE folder_id=?", (fid,))}
        found: dict[str, tuple[int, float]] = {}
        for path, size, mtime in _walk(root, should_stop):
            found[path] = (size, mtime)
            if len(found) % 500 == 0:
                progress(0, 0, f"Looking for photos… {len(found):,} found")

        missing = {p: r for p, r in known.items() if p not in found}
        missing_by_fp = {r["fp"]: r for r in missing.values() if r["fp"]}
        new_rows = []
        with db.transaction():
            for path, (size, mtime) in found.items():
                row = known.get(path)
                if row is not None:
                    if row["size"] != size or abs((row["mtime"] or 0) - mtime) > 1:
                        db.x("DELETE FROM faces WHERE photo_id=?", (row["id"],))
                        db.x("UPDATE photos SET size=?, mtime=?, fp=?, faces_done=0, error=NULL WHERE id=?",
                             (size, mtime, imaging.fingerprint(path, size), row["id"]))
                        changed += 1
                    continue
                fp = imaging.fingerprint(path, size)
                moved = missing_by_fp.pop(fp, None) if fp else None
                if moved is not None:
                    # Same file, new name/location: keep its faces and people.
                    db.x("UPDATE photos SET path=?, mtime=? WHERE id=?", (path, mtime, moved["id"]))
                    missing.pop(moved["path"], None)
                    continue
                new_rows.append((fid, path, size, mtime, fp, mtime))
            db.conn.executemany("INSERT OR IGNORE INTO photos(folder_id, path, size, mtime, fp, taken_at) "
                                "VALUES(?,?,?,?,?,?)", new_rows)
        changed += len(new_rows)
        if missing:
            db.delete_photos([r["id"] for r in missing.values()])
    db.cleanup_people()
    return changed


def _prepare(row) -> tuple:
    """Runs on a pool thread: decode, thumbnail, metadata."""
    try:
        li = imaging.load(row["path"], config.DETECT_MAX_SIDE)
        imaging.save_thumbnail(li.image, config.thumb_path(row["id"]))
        return row, li, None
    except Exception as e:  # unreadable / corrupt / unsupported
        return row, None, f"{type(e).__name__}: {e}"


def process_pending(db: Database, engine: FaceEngine, progress: Progress, changed: Callable[[], None],
                    should_stop: Callable[[], bool]) -> int:
    pending = db.q("SELECT id, path, mtime FROM photos WHERE faces_done=0 ORDER BY id")
    total = len(pending)
    if not total:
        return 0
    last_notify = last_group = time.monotonic()
    done = 0
    workers = max(2, min(6, (os.cpu_count() or 4) // 2))
    pool = ThreadPoolExecutor(workers)
    rows = iter(pending)
    window: deque = deque()

    def fill():
        # Decode a few photos ahead on the pool while this thread runs face detection.
        while len(window) < workers * 3:
            row = next(rows, None)
            if row is None:
                break
            window.append(pool.submit(_prepare, row))

    try:
        fill()
        while window:
            if should_stop():
                raise Cancelled
            row, li, err = window.popleft().result()
            fill()
            if li is None:
                db.x("UPDATE photos SET faces_done=1, error=? WHERE id=?", (err, row["id"]))
            else:
                try:
                    found = engine.detect(li.bgr())
                except Exception as e:  # pragma: no cover - defensive
                    found, err = [], f"Face detection failed: {e}"
                w, h = li.full_size
                with db.transaction():
                    db.x("DELETE FROM faces WHERE photo_id=?", (row["id"],))
                    for f in found:
                        face_id = db.insert_face(row["id"], f)
                        try:
                            imaging.save_face_crop(li.image, f.box, config.face_crop_path(face_id))
                        except OSError:
                            pass
                    db.x("UPDATE photos SET width=?, height=?, taken_at=?, faces_done=1, error=? WHERE id=?",
                         (w, h, li.taken_at or row["mtime"], err, row["id"]))
            done += 1
            progress(done, total, f"Analyzing photos… {done:,} of {total:,}")
            now = time.monotonic()
            if now - last_group > 20:  # group periodically so People fills in during long scans
                clustering.cluster(db)
                last_group = now
            if now - last_notify > 1.5:
                changed()
                last_notify = now
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return done


def backfill_attributes(db: Database, engine: FaceEngine, progress: Progress, changed: Callable[[], None],
                        should_stop: Callable[[], bool]) -> int:
    """Add expression, head direction and body pose to faces found before those features existed.

    Works from the stored face boxes, so people assignments stay untouched.
    """
    want_body = engine.body is not None
    pending = "f.attrs_done=0" + (" OR f.body_done=0" if want_body else "")
    rows = db.q(f"SELECT DISTINCT p.id, p.path FROM photos p JOIN faces f ON f.photo_id=p.id "
                f"WHERE {pending} ORDER BY p.id")
    total = len(rows)
    last_notify = time.monotonic()
    for n, row in enumerate(rows, 1):
        if should_stop():
            raise Cancelled
        todo = db.q("SELECT id, x, y, w, h, ignored, attrs_done, body_done FROM faces f WHERE photo_id=? AND "
                    f"({pending})", (row["id"],))
        try:
            bgr = imaging.load(row["path"], config.DETECT_MAX_SIDE).bgr()
        except Exception:
            bgr = None
        need_pose = [f for f in todo if want_body and not f["body_done"] and not f["ignored"]]
        poses = engine.poses(bgr, [(f["x"], f["y"], f["w"], f["h"]) for f in need_pose]) if bgr is not None \
            else [None] * len(need_pose)
        pose_of = {f["id"]: body for f, body in zip(need_pose, poses)}
        with db.transaction():
            for f in todo:
                if not f["attrs_done"]:
                    attrs = None
                    if bgr is not None and not f["ignored"]:
                        match = engine.find_row(bgr, (f["x"], f["y"], f["w"], f["h"]))
                        if match is not None:
                            attrs = engine.attributes_row(bgr, match)
                    db.set_face_attrs(f["id"], attrs)
                if want_body and not f["body_done"]:
                    body = pose_of.get(f["id"])
                    db.set_face_pose(f["id"], body.pose if body else None, bool(body and body.arms_up))
        progress(n, total, f"Reading expressions and poses… {n:,} of {total:,}")
        if time.monotonic() - last_notify > 3:
            changed()
            last_notify = time.monotonic()
    return total


def full_scan(db: Database, engine: FaceEngine, progress: Progress, changed: Callable[[], None],
              should_stop: Callable[[], bool]) -> str:
    n_changed = sync_files(db, progress, should_stop)
    if n_changed:
        changed()
    n = process_pending(db, engine, progress, changed, should_stop)
    if backfill_attributes(db, engine, progress, changed, should_stop):
        changed()
    progress(0, 0, "Grouping faces…")
    clustering.cluster(db)
    changed()
    s = db.stats()
    msg = f"{s['photos']:,} photos · {s['people']:,} people"
    if n:
        msg = f"Analyzed {n:,} photos. " + msg
    return msg
