"""Writes matched images (original bytes) and matches.json into one folder per run."""

from __future__ import annotations

import json
import os
import threading
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
    while path.exists():
        path = output_dir / f"{base}-{n}"
        n += 1
    path.mkdir(parents=True)
    return path


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

    def build_zip(self, min_confidence: float = 0.0) -> Path:
        """Zip of saved files at or above min_confidence, plus matches.json. Images are already
        compressed, so they are stored as-is."""
        # Snapshot under the lock, zip outside it: saves (and the manifest flush) carry on meanwhile.
        # Saved image files are never rewritten, so reading them unlocked is safe.
        with self._lock:
            doc = self._doc()
        target = self.folder / f"matches_{round(min_confidence * 100):03d}.zip"
        tmp = self.folder / f"{target.stem}.{threading.get_ident()}.zip.part"  # two downloads at once don't collide
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
                for e in doc["matches"]:
                    if e["confidence"] >= min_confidence:
                        zf.write(self.folder / e["file"], e["file"])
                zf.writestr("matches.json", json.dumps(doc, indent=2, ensure_ascii=False))
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return target
