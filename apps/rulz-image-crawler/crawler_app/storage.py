"""Writes matched images (original bytes) and matches.json into one folder per run."""

from __future__ import annotations

import io
import json
import os
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .fetcher import FetchedImage
from .urls import file_stem, site_key, slug

REPLACE_TRIES = 10  # matches.json replace attempts, 50 ms apart


def _plain_name(name: object) -> bool:
    return isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name and "\\" not in name and ":" not in name


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def make_run_dir(
    output_dir: Path, start_url: str, tag_id: str, now: datetime | None = None, label: str | None = None
) -> Path:
    """<date>_<site or label>_<tag>; `label` names a scanned folder instead of a website."""
    now = now or datetime.now()
    base = f"{now:%Y-%m-%d_%H%M}_{slug(label or site_key(start_url), 40)}_{slug(tag_id, 30)}"
    path = output_dir / base
    n = 2
    while True:
        try:
            path.mkdir(parents=True)  # not exist_ok: two runs started together get a folder each
            return path
        except FileExistsError:
            path = output_dir / f"{base}-{n}"
            n += 1


class RunStorage:
    def __init__(self, folder: Path, meta: dict[str, Any]):
        self.folder = folder
        self.meta = meta
        self.entries: list[dict[str, Any]] = []
        self._names: set[str] = set()
        # Saves come from several classify batches at once (in worker threads); one writer at a time.
        self._lock = threading.RLock()
        self._dirty = False

    @classmethod
    def load(cls, folder: Path) -> RunStorage:
        """A finished (or interrupted) run read back from its matches.json; only files that are
        listed there and still on disk are served."""
        doc = json.loads((folder / "matches.json").read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("matches.json is not an object")
        entries = doc.pop("matches", [])
        st = cls(folder, doc)
        # Plain file names only: a hand-edited "../x" or "C:\x" must never be served from outside the folder.
        st.entries = [
            e
            for e in (entries if isinstance(entries, list) else [])
            if isinstance(e, dict) and _plain_name(e.get("file")) and (folder / e["file"]).is_file()
        ]
        st._names = {e["file"] for e in st.entries}
        return st

    def save(self, img: FetchedImage, confidence: float, evidence: str) -> dict[str, Any]:
        """Content-hash naming means one file per distinct image; the caller already dedupes by hash."""
        with self._lock:
            return self._save(img, confidence, evidence)

    def _save(self, img: FetchedImage, confidence: float, evidence: str) -> dict[str, Any]:
        name = f"{file_stem(img.url)}_{img.sha256[:10]}{img.ext}"
        path = self.folder / name
        if name not in self._names:
            tmp = path.with_suffix(path.suffix + ".part")
            try:
                tmp.write_bytes(img.data)
                os.replace(tmp, path)
            except BaseException:
                _unlink_quietly(tmp)  # no half-written .part left behind (disk full, path too long)
                raise
            self._names.add(name)
        entry = {
            "file": name,
            "url": img.url if not img.url.startswith("data:") else "(inline data: image)",
            "page": img.page,
            "confidence": round(confidence, 4),
            "evidence": evidence,
            "sha256": img.sha256,
            "width": img.width,
            "height": img.height,
            "original": img.is_original,
        }
        self.entries.append(entry)
        # matches.json is rewritten by flush() (the run's ticker, twice a second, and at the end),
        # not once per match: rewriting the whole list per match is O(n^2) bytes on big runs.
        self._dirty = True
        return entry

    def _doc(self) -> dict[str, Any]:
        return {**self.meta, "matches": list(self.entries)}

    def write_manifest(self) -> None:
        """Rewrites matches.json. Marked clean only once the new file is in place: on Windows the
        replace fails while anyone has the old file open (the history list reading it), and the next
        flush has to try again rather than believe it was written."""
        with self._lock:
            tmp = self.folder / "matches.json.part"
            tmp.write_text(json.dumps(self._doc(), indent=2, ensure_ascii=False), encoding="utf-8")
            target = self.folder / "matches.json"
            for attempt in range(REPLACE_TRIES):
                try:
                    os.replace(tmp, target)
                    break
                except PermissionError:  # a reader has it open for a few milliseconds
                    if attempt == REPLACE_TRIES - 1:
                        _unlink_quietly(tmp)
                        raise
                    time.sleep(0.05)
            self._dirty = False

    @property
    def dirty(self) -> bool:
        return self._dirty

    def flush(self) -> bool:
        """Write matches.json if a match was saved since the last write. True when it wrote."""
        with self._lock:
            if not self._dirty:
                return False
            self.write_manifest()
            return True

    def has_file(self, name: str) -> bool:
        return name in self._names

    def iter_zip(self, min_confidence: float) -> Iterator[bytes]:
        """Zip of saved files at or above min_confidence, plus matches.json, made while it is sent: the
        download starts at once, and no temporary copy of the matches is written (a big run's zip used
        to be built in full on disk first). Images are already compressed, so they are stored as-is.
        A plain (blocking) generator: the web server runs it in a worker thread."""
        # Snapshot under the lock, zip outside it: saves (and the manifest flush) carry on meanwhile.
        # Saved image files are never rewritten, so reading them unlocked is safe.
        with self._lock:
            doc = self._doc()
        out = _ZipStream()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
            for e in doc["matches"]:
                conf = e.get("confidence")
                # A hand-edited entry without a number is left out: an error here would cut off a
                # download that has already started.
                if not isinstance(conf, (int, float)) or conf < min_confidence:
                    continue
                try:
                    zf.write(self.folder / e["file"], e["file"])
                except OSError:  # deleted (or locked) by hand since it was saved
                    continue
                yield out.take()
            zf.writestr("matches.json", json.dumps(doc, indent=2, ensure_ascii=False))
        yield out.take()


class _ZipStream(io.RawIOBase):
    """Where a zip being streamed is written. zipfile goes back to each entry's header to fill in its
    size and CRC once the entry's data is written, so the entry being written is held here until it
    is finished; take() hands out everything finished so far. Seeking back into bytes already handed
    out is refused (zipfile never does it)."""

    def __init__(self) -> None:
        super().__init__()
        self._buf = bytearray()
        self._base = 0  # stream offset of _buf[0]
        self._pos = 0

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_CUR:
            offset += self._pos
        elif whence == io.SEEK_END:
            offset += self._base + len(self._buf)
        if offset < self._base:
            raise OSError("cannot seek back into data that was already sent")
        self._pos = offset
        return offset

    def write(self, b) -> int:
        data = bytes(b)
        i = self._pos - self._base
        if i > len(self._buf):
            self._buf.extend(bytes(i - len(self._buf)))
        self._buf[i : i + len(data)] = data
        self._pos += len(data)
        return len(data)

    def take(self) -> bytes:
        """Everything before the current position, which zipfile has finished with."""
        n = self._pos - self._base
        chunk = bytes(self._buf[:n])
        del self._buf[:n]
        self._base = self._pos
        return chunk


def sweep_zips(zip_dir: Path, older_than_s: float = 3600.0) -> None:
    """Deletes zips left behind in zip_dir by versions before 1.4.0, which built each download there."""
    if not zip_dir.is_dir():
        return
    cutoff = time.time() - older_than_s
    for p in zip_dir.glob("*.zip"):
        try:
            if p.stat().st_mtime <= cutoff:
                p.unlink()
        except OSError:  # still being sent, or already gone
            pass
