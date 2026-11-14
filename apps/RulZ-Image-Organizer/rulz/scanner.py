"""Library indexing: find image files, make thumbnails, detect faces and group them.

UI-independent; the Qt worker in ``ui/worker.py`` drives it with callbacks.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from . import clustering, config, imaging
from .db import Database, from_blob
from .faces import FaceEngine

Progress = Callable[[int, int, str], None]

# Library changes are written in short transactions of this many photos, so the UI can save its own
# edits in between instead of waiting on (or failing behind) the scanner's write lock.
BATCH = 1000
# Seconds between the grouping passes that run during a long scan (stretched when grouping is slow).
GROUP_EVERY = 20


class Cancelled(Exception):
    pass


def _walk(root: str, should_stop: Callable[[], bool], unreadable: set[str] | None = None):
    """Yield (path, size, mtime) of every image under ``root``.

    Folders and files that can't be read right now (access denied, a network hiccup) are added to
    ``unreadable``, so the caller can tell "gone" from "couldn't look".
    """
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
                        if unreadable is not None:
                            unreadable.add(e.path)
                        continue
        except OSError:
            if unreadable is not None:
                unreadable.add(d)
            continue


def _under(path: str, roots: set[str]) -> bool:
    """True if ``path`` is one of ``roots`` or inside one of them."""
    while path:
        if path in roots:
            return True
        parent = os.path.dirname(path)
        if parent == path:
            return False
        path = parent
    return False


def sync_files(db: Database, progress: Progress, should_stop: Callable[[], bool],
               rewritten: Rewritten | None = None) -> int:
    """Bring the photos table in line with what's on disk. Returns number of new/changed photos.

    ``rewritten`` hears the ids of deleted photos and faces: new rows may get those ids, and their cached
    thumbnails must not show up for them."""
    changed = 0
    for folder in db.folders():
        fid, root = folder["id"], folder["path"]
        if not os.path.isdir(root):
            continue  # drive unplugged? keep what we have
        progress(0, 0, f"Looking for photos in {root}")
        known = {r["path"]: r for r in db.q("SELECT id, path, size, mtime, fp FROM photos WHERE folder_id=?", (fid,))}
        found: dict[str, tuple[int, float]] = {}
        unreadable: set[str] = set()
        for path, size, mtime in _walk(root, should_stop, unreadable):
            found[path] = (size, mtime)
            if len(found) % 500 == 0:
                progress(0, 0, f"Looking for photos… {len(found):,} found")

        missing = {p: r for p, r in known.items() if p not in found}
        if unreadable:
            # A folder we couldn't list this time is not a deleted folder: keep its photos and their tags.
            missing = {p: r for p, r in missing.items() if not _under(p, unreadable)}
        missing_by_fp: dict[str, list] = {}
        for r in missing.values():
            if r["fp"]:
                missing_by_fp.setdefault(r["fp"], []).append(r)
        # name -> sizes of the new files with that name (a duplicate's moved copy keeps its name and size)
        new_names: dict[str, set[int]] = {}
        if missing_by_fp:
            for p, (size, _) in found.items():
                if p not in known:
                    new_names.setdefault(os.path.basename(p).lower(), set()).add(size)
        unsure_sizes: set[int] = set()  # sizes of new files that couldn't be read: a missing photo may be one
        updated: list[tuple[int, int, float, str]] = []  # (photo id, size, mtime, fp) of changed files
        touched: list[tuple[float, int]] = []            # (mtime, photo id): new date, same content
        moved: list[tuple[str, float, int]] = []
        new_rows: list[tuple] = []

        def flush():
            nonlocal changed
            if not (updated or touched or moved or new_rows):
                return
            with db.transaction():
                # A changed file is analyzed again; its faces stay until then, and the user's work on them
                # moves over to the new detection (_replace_faces).
                db.conn.executemany("UPDATE photos SET size=?, mtime=?, fp=?, faces_done=0, error=NULL WHERE id=?",
                                    [(size, mtime, fp, photo_id) for photo_id, size, mtime, fp in updated])
                db.conn.executemany("UPDATE photos SET mtime=? WHERE id=?", touched)
                db.conn.executemany("UPDATE photos SET path=?, mtime=? WHERE id=?", moved)
                db.conn.executemany("INSERT OR IGNORE INTO photos(folder_id, path, size, mtime, fp, taken_at) "
                                    "VALUES(?,?,?,?,?,?)", new_rows)
            changed += len(updated) + len(new_rows)
            updated.clear()
            touched.clear()
            moved.clear()
            new_rows.clear()

        total = len(found)
        # Fingerprinting reads every new file, which can take minutes: it happens outside any transaction.
        for n, (path, (size, mtime)) in enumerate(found.items(), 1):
            if should_stop():
                flush()  # keep what's done; the next scan carries on from here
                raise Cancelled
            if n % 250 == 0:
                progress(n, total, f"Checking photos… {n:,} of {total:,}")
            row = known.get(path)
            if row is not None:
                if row["size"] != size or abs((row["mtime"] or 0) - mtime) > 1:
                    fp = imaging.fingerprint(path, size)
                    if not fp:
                        continue  # can't read it right now: leave it (and its tags) for the next scan
                    if fp == row["fp"] and row["size"] == size and not _uncompressed(path):
                        # Only the file date moved (a copy or backup tool, a DST shift on a FAT32 card):
                        # the picture is the same, so its faces and the user's corrections stay.
                        touched.append((mtime, row["id"]))
                    else:
                        updated.append((row["id"], size, mtime, fp))
            else:
                fp = imaging.fingerprint(path, size)
                if not fp:
                    unsure_sizes.add(size)
                    continue  # added on a later scan, once it can be read
                moved_row = _take_moved(missing_by_fp, fp, path, new_names)
                if moved_row is not None:
                    # Same file, new name/location: keep its faces and people.
                    moved.append((path, mtime, moved_row["id"]))
                    missing.pop(moved_row["path"], None)
                else:
                    new_rows.append((fid, path, size, mtime, fp, mtime))
            if len(updated) + len(touched) + len(moved) + len(new_rows) >= BATCH:
                flush()
        flush()
        # A moved file keeps its size: a missing photo the size of an unreadable new file may just have moved.
        gone = [r["id"] for r in missing.values() if r["size"] not in unsure_sizes]
        if gone:
            deleted = db.delete_photos(gone)
            if rewritten is not None:
                rewritten(*deleted)
    db.cleanup_people()
    return changed


def _inter(a, b) -> float:
    """Area shared by two (x, y, w, h) boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def _iou(a, b) -> float:
    """Overlap of two (x, y, w, h) boxes: intersection over union."""
    inter = _inter(a, b)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-12)


def _box(r) -> tuple[float, float, float, float]:
    return r["x"], r["y"], r["w"], r["h"]


def _replace_faces(db: Database, row, full_size, found: list, crops: list) -> list[int]:
    """Store a fresh detection for a photo (inside the caller's transaction); returns the new face ids.

    The user's work on the old faces carries over, so a re-analysis (the file changed: an edit, or just
    metadata such as a star rating written into it) doesn't lose it:
      * a new detection on the spot of an old one (IoU >= 0.5, best overlaps first) takes over its person,
        how it was assigned, "not a face", the "not this person" answers, and being someone's profile photo;
      * boxes they drew stay while the picture is still laid out the same (same size, and old detections
        line up with new ones where there were any: not rotated or cropped), and a detection of the same
        face isn't added next to one.
    New faces are inserted before the old rows go, so none of them can reuse an old face's id here.
    """
    pid = row["id"]
    old = db.q("SELECT id, x, y, w, h, person_id, assigned_by, ignored, manual, embedding FROM faces "
               "WHERE photo_id=?", (pid,))
    prev = [o for o in old if not o["manual"]]
    boxes = [f.box for f in found]
    # Same spot AND same face: after a mirror, a crop or a turn, another person can land on the spot.
    t_same = db.match_threshold()
    pairs = sorted(((_iou(b, _box(o)), i, j) for i, b in enumerate(boxes) for j, o in enumerate(prev)),
                   key=lambda t: -t[0])
    match: dict[int, int] = {}  # found index -> prev index
    taken: set[int] = set()
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i in match or j in taken:
            continue
        emb = prev[j]["embedding"]
        if emb is not None and float(np.dot(from_blob(emb), found[i].embedding)) < t_same:
            continue
        match[i] = j
        taken.add(j)
    # Drawn boxes stay unless the picture was laid out anew: another size, or faces found again but none
    # where they were. (Nothing found at all says nothing about the layout.)
    same_layout = (row["width"] is None or (row["width"], row["height"]) == tuple(full_size)) \
        and (not prev or not found or bool(match))
    keep = [o for o in old if o["manual"] and same_layout]
    rejected: dict[int, list[int]] = {}
    covers: dict[int, list[int]] = {}
    if prev:
        marks = ",".join("?" * len(prev))
        ids = [o["id"] for o in prev]
        for f, p in db.q(f"SELECT face_id, person_id FROM rejections WHERE face_id IN ({marks})", ids):
            rejected.setdefault(f, []).append(p)
        for p, f in db.q(f"SELECT id, cover_face_id FROM people WHERE cover_face_id IN ({marks})", ids):
            covers.setdefault(f, []).append(p)

    # A detection that is an old detected face found again is never mistaken for a drawn one.
    skip = _covered_by_manual(boxes, [_box(o) for o in keep], exclude=set(match))
    new_ids: list[int] = []
    for i, (f, crop) in enumerate(zip(found, crops)):
        if i in skip:
            continue
        face_id = db.insert_face(pid, f)
        new_ids.append(face_id)
        try:
            imaging.save_crop(crop, config.face_crop_path(face_id))
        except OSError:
            pass
        if i not in match:
            continue
        best = prev[match[i]]
        db.x("UPDATE faces SET person_id=?, assigned_by=?, ignored=? WHERE id=?",
             (best["person_id"], best["assigned_by"], best["ignored"], face_id))
        db.conn.executemany("INSERT OR IGNORE INTO rejections(face_id, person_id) VALUES(?,?)",
                            [(face_id, p) for p in rejected.get(best["id"], ())])
        db.conn.executemany("UPDATE people SET cover_face_id=? WHERE id=?",
                            [(face_id, p) for p in covers.get(best["id"], ())])
    kept = {o["id"] for o in keep}
    db.delete_faces([o["id"] for o in old if o["id"] not in kept])
    return new_ids


def _covered_by_manual(found: list, manual: list, exclude: set[int] = frozenset()) -> set[int]:
    """Indexes of detections that are a hand-drawn face again. For each drawn box: of the detections at
    least half inside it (or IoU >= 0.3; drawn boxes are usually much looser than the detector's), the one
    whose center is nearest the box's center, i.e. the face it was drawn around. ``exclude``: never these."""
    skip: set[int] = set()
    for box in manual:
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        best, best_d = None, None
        for i, b in enumerate(found):
            if i in skip or i in exclude:
                continue
            if _inter(b, box) / max(b[2] * b[3], 1e-12) < 0.5 and _iou(b, box) < 0.3:
                continue
            d = (b[0] + b[2] / 2 - cx) ** 2 + (b[1] + b[3] / 2 - cy) ** 2
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is not None:
            skip.add(best)
    return skip


def _uncompressed(path: str) -> bool:
    """Formats where a local edit keeps the file size and the fingerprinted start and end of the file: for
    these a new date may mean new pixels, so the photo is analyzed again."""
    return os.path.splitext(path)[1].lower() in (".bmp", ".tif", ".tiff")


def _take_moved(missing_by_fp: dict[str, list], fp: str, path: str, new_names: dict[str, set[int]]):
    """Pick (and remove) the missing photo a new file at ``path`` was moved from, or None.

    Several missing photos can share a fingerprint (duplicate copies): prefer the one with the same file
    name, so each renamed or moved copy keeps its own row and tags. Rows whose name and size another new
    file has (``new_names``: lower-case name -> sizes) are left for that file; if every candidate is taken
    that way, this file is an extra copy and gets a row of its own.
    """
    rows = missing_by_fp.get(fp)
    if not rows:
        return None
    name = os.path.basename(path).lower()
    names = [os.path.basename(r["path"]).lower() for r in rows]
    pick = next((i for i, n in enumerate(names) if n == name), None)
    if pick is None:
        pick = next((i for i, (n, r) in enumerate(zip(names, rows)) if r["size"] not in new_names.get(n, ())),
                    None)
        if pick is None:
            return None
    row = rows.pop(pick)
    if not rows:
        del missing_by_fp[fp]
    return row


_local = threading.local()


def _prepare(row, engine_factory: Callable[[], FaceEngine]) -> tuple:
    """Runs on a pool thread: decode, thumbnail, metadata, face detection and face crops.

    Each pool thread keeps its own FaceEngine (they're not thread-safe); the scan thread only writes.
    """
    try:
        li = imaging.load(row["path"], config.DETECT_MAX_SIDE)
        imaging.save_thumbnail(li.image, config.thumb_path(row["id"]))
    except Exception as e:  # unreadable / corrupt / unsupported
        return row, None, None, [], [], f"{type(e).__name__}: {e}"
    engine = getattr(_local, "engine", None)
    if engine is None:  # model loading errors are real failures: let them end the scan
        engine = _local.engine = engine_factory()
    err = None
    try:
        found = engine.detect(li.bgr())
    except Exception as e:  # pragma: no cover - defensive
        found, err = [], f"Face detection failed: {e}"
    crops = [imaging.face_crop(li.image, f.box) for f in found]
    return row, li.full_size, li.taken_at, found, crops, err


Rewritten = Callable[[list[int], list[int]], None]


def process_pending(db: Database, progress: Progress, changed: Callable[[], None],
                    should_stop: Callable[[], bool], engine_factory: Callable[[], FaceEngine] = FaceEngine,
                    rewritten: Rewritten | None = None) -> int:
    """Analyze photos that need it. ``rewritten(photo_ids, face_ids)`` hears which thumbnails and face
    crops were (re)written, so the UI can drop stale cached copies: an edited photo keeps its id, and
    SQLite hands a deleted face's id to the next new face."""
    pending = db.q("SELECT id, path, mtime, width, height FROM photos WHERE faces_done=0 ORDER BY id")
    total = len(pending)
    if not total:
        return 0
    last_notify = last_group = time.monotonic()
    group_every = GROUP_EVERY
    new_photos: list[int] = []
    new_faces: list[int] = []

    def notify():
        if rewritten is not None and (new_photos or new_faces):
            rewritten(new_photos[:], new_faces[:])
        new_photos.clear()
        new_faces.clear()
        changed()

    done = 0
    # Detection runs on these threads too. OpenCV DNN already spreads each call over several cores and
    # every thread loads its own models (~90 MB before buffers), so a handful of threads is enough.
    workers = max(2, min(4, (os.cpu_count() or 4) // 2))
    pool = ThreadPoolExecutor(workers)
    rows = iter(pending)
    window: deque = deque()

    def fill():
        # Keep a few photos in flight so the pool never waits on this thread's database writes.
        while len(window) < workers * 2:
            row = next(rows, None)
            if row is None:
                break
            window.append(pool.submit(_prepare, row, engine_factory))

    try:
        fill()
        while window:
            if should_stop():
                raise Cancelled
            row, full_size, taken_at, found, crops, err = window.popleft().result()
            fill()
            if full_size is None or (err and not found):
                # Unreadable, or the detector failed: the faces it had (and the user's work on them) stay.
                db.x("UPDATE photos SET faces_done=1, error=? WHERE id=?", (err, row["id"]))
                if full_size is not None:
                    new_photos.append(row["id"])  # its thumbnail was written all the same
            else:
                w, h = full_size
                with db.transaction():
                    new_faces += _replace_faces(db, row, full_size, found, crops)
                    db.x("UPDATE photos SET width=?, height=?, taken_at=?, faces_done=1, error=? WHERE id=?",
                         (w, h, taken_at or row["mtime"], err, row["id"]))
                new_photos.append(row["id"])
            done += 1
            progress(done, total, f"Analyzing photos… {done:,} of {total:,}")
            now = time.monotonic()
            if now - last_group > group_every:  # group periodically so People fills in during long scans
                clustering.cluster(db)
                took = time.monotonic() - now
                # Grouping reads every face, so it slows down as the library grows: never let it take
                # more than about a fifth of the scan.
                group_every = max(GROUP_EVERY, 5 * took)
                last_group = now = time.monotonic()
            if now - last_notify > 1.5:
                notify()
                last_notify = now
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if rewritten is not None and (new_photos or new_faces):
            rewritten(new_photos[:], new_faces[:])
    return done


def backfill_attributes(db: Database, engine: FaceEngine | Callable[[], FaceEngine], progress: Progress,
                        changed: Callable[[], None], should_stop: Callable[[], bool]) -> int:
    """Add expression, head direction and body pose to faces found before those features existed.

    Works from the stored face boxes, so people assignments stay untouched. ``engine`` may be a factory:
    the models (~90 MB) are then only loaded when some face still needs this.
    """
    if not db.one("SELECT 1 FROM faces WHERE attrs_done=0 OR body_done=0 LIMIT 1"):
        return 0
    if not isinstance(engine, FaceEngine) and callable(engine):
        engine = engine()
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
        attrs_of = {}
        detections = None  # detected once per photo, then matched against every stored box
        for f in todo:
            if f["attrs_done"]:
                continue
            attrs = None
            if bgr is not None and not f["ignored"]:
                if detections is None:
                    detections = engine.detect_rows(bgr)
                match = engine.find_row(bgr, (f["x"], f["y"], f["w"], f["h"]), detections)
                if match is not None:
                    attrs = engine.attributes_row(bgr, match)
            attrs_of[f["id"]] = attrs
        with db.transaction():  # only the writes hold the lock
            for f in todo:
                if not f["attrs_done"]:
                    db.set_face_attrs(f["id"], attrs_of[f["id"]])
                if want_body and not f["body_done"]:
                    body = pose_of.get(f["id"])
                    db.set_face_pose(f["id"], body.pose if body else None, bool(body and body.arms_up))
        progress(n, total, f"Reading expressions and poses… {n:,} of {total:,}")
        if time.monotonic() - last_notify > 3:
            changed()
            last_notify = time.monotonic()
    return total


def full_scan(db: Database, engine: FaceEngine | Callable[[], FaceEngine], progress: Progress,
              changed: Callable[[], None], should_stop: Callable[[], bool],
              rewritten: Rewritten | None = None) -> str:
    n_changed = sync_files(db, progress, should_stop, rewritten)
    if n_changed:
        changed()
    n = process_pending(db, progress, changed, should_stop, rewritten=rewritten)
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
