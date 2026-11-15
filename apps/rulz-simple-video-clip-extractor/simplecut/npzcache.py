"""What a background read of a video found, kept on disk per video file (scene-cut scores, the
sound's envelope): one .npz per video, tied to the file's size and date like its clip list, so
a reopened video shows it at once and another file of that name is read again."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import numpy as np

from .cliplist import STALE_DAYS, _file_stamp, _remove, store_dir, video_key


def cache_dir(name: str) -> str:
    """A folder beside the clip lists."""
    return os.path.join(os.path.dirname(store_dir()), name)


def cache_path(folder: str, video: str) -> str:
    return os.path.join(folder, hashlib.sha1(video_key(video).encode("utf-8")).hexdigest()[:24] + ".npz")


def save_npz(path: str, video: str, version: int, arrays: dict, stamp: tuple[int, float] | None = None) -> bool:
    """Write the arrays for this video. `stamp` is the file's size and date when the read began:
    a file that changed meanwhile (still being written or copied) is not stored. False when
    nothing was written."""
    now = _file_stamp(video)
    if now is None or (stamp is not None and stamp != now):
        return False
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp.npz"   # (two reads may save at once)
    meta = json.dumps({"version": version, "path": os.path.abspath(video), "size": now[0], "mtime": now[1]})
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(tmp, meta=np.array(meta), **arrays)
        os.replace(tmp, path)
        return True
    except (OSError, ValueError):
        _remove(tmp)
        return False


def load_npz(path: str, video: str, version: int, names: tuple[str, ...]) -> dict | None:
    """The named arrays stored for this video, or None (never stored, another file of that name
    since, another version, or unreadable)."""
    stamp = _file_stamp(video)
    if stamp is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if (meta.get("version") != version or int(meta["size"]) != stamp[0]
                    or abs(float(meta["mtime"]) - stamp[1]) >= 2.0):
                return None
            return {k: np.asarray(z[k]) for k in names}
    except Exception:  # noqa: BLE001 - missing, damaged or hand-made: read again
        return None


def clean_dir(d: str, now: float | None = None):
    """Delete stored files whose video no longer exists and that are older than STALE_DAYS,
    and ones that were cut off while being written."""
    now = time.time() if now is None else now
    try:
        names = os.listdir(d)
    except OSError:
        return
    for name in names:
        p = os.path.join(d, name)
        try:
            age = now - os.path.getmtime(p)
            if ".tmp" in name:
                if age > 86400:
                    os.remove(p)
                continue
            if not name.endswith(".npz") or age < STALE_DAYS * 86400:
                continue
            with np.load(p, allow_pickle=False) as z:
                video = json.loads(str(z["meta"]))["path"]
            if not os.path.exists(video):
                os.remove(p)
        except Exception:  # noqa: BLE001 - leave what can't be read; it is cleaned up never or later
            continue
