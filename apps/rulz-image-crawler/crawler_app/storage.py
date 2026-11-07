"""Writes matched images (original bytes) and matches.json into one folder per run."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .fetcher import FetchedImage
from .urls import file_stem, site_key, slug


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
        entries = doc.pop("matches", [])
        st = cls(folder, doc)
        st.entries = [e for e in entries if isinstance(e, dict) and (folder / str(e.get("file", ""))).is_file()]
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
            tmp.write_bytes(img.data)
            os.replace(tmp, path)
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
        with self._lock:
            self._dirty = False
            tmp = self.folder / "matches.json.part"
            tmp.write_text(json.dumps(self._doc(), indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.folder / "matches.json")

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

    def build_zip(self, min_confidence: float, dest_dir: Path) -> Path:
        """Zip of saved files at or above min_confidence, plus matches.json, as a new file in dest_dir
        that the caller deletes once it is sent. Never in the run folder: a fixed name there can't be
        replaced while an earlier download still has it open (Windows), and every slider position
        would leave another full copy of the matches behind. Images are already compressed, so they
        are stored as-is."""
        # Snapshot under the lock, zip outside it: saves (and the manifest flush) carry on meanwhile.
        # Saved image files are never rewritten, so reading them unlocked is safe.
        with self._lock:
            doc = self._doc()
        dest_dir.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f"{self.folder.name[:80]}_", suffix=".zip", dir=dest_dir)
        os.close(fd)
        target = Path(name)
        try:
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as zf:
                for e in doc["matches"]:
                    path = self.folder / e["file"]
                    if e["confidence"] >= min_confidence and path.is_file():  # skip files deleted by hand
                        zf.write(path, e["file"])
                zf.writestr("matches.json", json.dumps(doc, indent=2, ensure_ascii=False))
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return target


def sweep_zips(zip_dir: Path, older_than_s: float = 3600.0) -> None:
    """Deletes zips left behind in zip_dir (a download that was cut off never runs its cleanup)."""
    if not zip_dir.is_dir():
        return
    cutoff = time.time() - older_than_s
    for p in zip_dir.glob("*.zip"):
        try:
            if p.stat().st_mtime <= cutoff:
                p.unlink()
        except OSError:  # still being sent, or already gone
            pass
