"""Model downloads with progress reporting (used by the face/body scorers and the WebUI).

A progress callback has the signature progress(text: str, fraction: float | None); fraction is None when the
size is unknown or the step is not a download (e.g. "loading weights").
"""
from __future__ import annotations

import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

log = logging.getLogger("rulztrainer")
ProgressFn = Callable[[str, "float | None"], None]


def _report(progress: ProgressFn | None, text: str, frac: float | None = None) -> None:
    if progress is not None:
        try:
            progress(text, frac)
        except Exception:  # noqa: BLE001
            pass
    log.info("%s%s", text, "" if frac is None else f" ({frac * 100:.0f}%)")


def download_url(url: str, dest: Path, label: str, progress: ProgressFn | None = None) -> Path:
    """Download `url` to `dest` with percentage progress (via urllib's report hook)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = {"pct": -1}

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(total, blocks * block_size)
            pct = int(done * 100 / total)
            if pct != last["pct"] and (pct % 2 == 0 or pct == 100):
                last["pct"] = pct
                _report(progress, f"Downloading {label}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", done / total)
        elif blocks % 200 == 0:
            _report(progress, f"Downloading {label}: {blocks * block_size / 1e6:.0f} MB", None)

    _report(progress, f"Downloading {label}...", 0.0)
    urllib.request.urlretrieve(url, str(tmp), reporthook=hook)
    tmp.replace(dest)
    _report(progress, f"Downloaded {label}", 1.0)
    return dest


def ensure_insightface_pack(name: str = "buffalo_l", root: str = "~/.insightface",
                            progress: ProgressFn | None = None) -> Path:
    """Fetch the InsightFace model pack (about 300 MB for buffalo_l) the same way insightface does, with progress."""
    root_dir = Path(os.path.expanduser(root))
    pack_dir = root_dir / "models" / name
    if pack_dir.is_dir() and any(pack_dir.glob("*.onnx")):
        return pack_dir
    from insightface.utils import storage

    url = storage.model_zoo_download_url(f"{name}.zip")
    zip_path = root_dir / "models" / f"{name}.zip"
    download_url(url, zip_path, f"InsightFace model pack {name}", progress)
    _report(progress, f"Extracting {name}...", None)
    pack_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(pack_dir)
    return pack_dir


def ensure_hf_file(repo_id: str, filename: str, label: str, progress: ProgressFn | None = None) -> Path:
    """One file of a Hugging Face repo, from the local cache or downloaded with progress (a cached snapshot of the
    repo does not mean this file is in it)."""
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id, filename, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached yet
        pass
    _snapshot_download(repo_id, label, progress, [filename])
    return Path(hf_hub_download(repo_id, filename, local_files_only=True))


def ensure_hf_snapshot(repo_id: str, label: str, progress: ProgressFn | None = None,
                       allow_patterns: list[str] | None = None, **kwargs) -> str:
    """Return the local snapshot path of a Hugging Face repo, downloading with progress if needed."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id, allow_patterns=allow_patterns, local_files_only=True, **kwargs)
    except Exception:  # noqa: BLE001 - not cached (or incomplete): download below
        pass
    return _snapshot_download(repo_id, label, progress, allow_patterns, **kwargs)


def _snapshot_download(repo_id: str, label: str, progress: ProgressFn | None = None,
                       allow_patterns: list[str] | None = None, **kwargs) -> str:
    """Download (the allowed files of) a repo with progress through the callback."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import tqdm as hf_tqdm

    class ReportingTqdm(hf_tqdm):  # one instance per file plus one "Fetching N files" counter
        def display(self, *a, **k):  # silent: we report through the callback instead of the console
            return None

        def update(self, n=1):
            super().update(n)
            total, done = getattr(self, "total", 0) or 0, getattr(self, "n", 0) or 0
            if getattr(self, "unit", "") == "B" and total > 50e6:  # a weights file
                _report(progress, f"Downloading {label} ({getattr(self, 'desc', '') or repo_id}): "
                                  f"{done / 1e9:.2f} / {total / 1e9:.2f} GB", done / total if total else None)

    _report(progress, f"Downloading {label} from Hugging Face ({repo_id})...", 0.0)
    path = snapshot_download(repo_id, allow_patterns=allow_patterns, tqdm_class=ReportingTqdm, **kwargs)
    _report(progress, f"Downloaded {label}", 1.0)
    return path


def ensure_mediapipe_pose(models_dir: str | Path, url: str, name: str, progress: ProgressFn | None = None) -> Path:
    p = Path(models_dir) / name
    if not p.exists():
        download_url(url, p, "MediaPipe pose model", progress)
    return p
