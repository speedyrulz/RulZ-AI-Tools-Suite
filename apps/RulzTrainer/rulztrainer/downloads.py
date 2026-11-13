"""Model downloads with progress reporting (used by the face/body scorers and the WebUI).

A progress callback has the signature progress(text: str, fraction: float | None); fraction is None when the
size is unknown or the step is not a download (e.g. "loading weights").
"""
from __future__ import annotations

import json
import logging
import os
import shutil
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


def component_ready(folder: Path) -> bool:
    """A model component folder (transformer/, text_encoder/, ...) is wholly downloaded: its config.json, and every
    shard its *.safetensors.index.json names (or a .safetensors file when it is not sharded). snapshot_download fetches
    the small files first, so an interrupted download leaves the configs and indexes without some shards; the hub
    moves each file into place only once it is complete."""
    folder = Path(folder)
    if not (folder / "config.json").is_file():
        return False
    indexes = list(folder.glob("*.safetensors.index.json"))
    if not indexes:
        return any(folder.glob("*.safetensors"))
    for index in indexes:
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
        except (OSError, ValueError):
            return False
        if not shards or any(not (folder / s).is_file() for s in shards):
            return False
    return True


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
    # into a side folder first, renamed when whole: an interrupted extraction must not leave a pack that looks
    # installed (some .onnx files) and fails later on a missing detection or recognition model
    tmp = pack_dir.with_name(pack_dir.name + ".extracting")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    inner = tmp / name  # (a pack zipped with its folder inside)
    src = inner if inner.is_dir() and not any(tmp.glob("*.onnx")) else tmp
    shutil.rmtree(pack_dir, ignore_errors=True)
    src.replace(pack_dir)
    shutil.rmtree(tmp, ignore_errors=True)
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


class _NullFile:
    def write(self, s):
        return len(s)

    def flush(self):
        pass


REPORT_EVERY_PCT = 2  # a progress line per this many percent of the bytes...
REPORT_EVERY_S = 15.0  # ...or at least this often while bytes arrive


def _reporting_tqdm(label: str, progress: ProgressFn | None):
    """A tqdm class for snapshot_download that reports byte progress through the callback and the log.

    huggingface_hub builds its bars with disable=None, which tqdm turns into "off" when stderr is no terminal (a
    training run writing train.log): a disabled bar never advances n and has no unit, so nothing was reported and the
    log sat at the opening 0%. This bar is always on and draws nothing. Since huggingface_hub 1.0 the bytes come as two
    aggregate bars: "Downloading bytes" (network, its total a growing guess: skipped) and "Reconstructing" (bytes
    written, against the summed file sizes: reported)."""
    import time

    from huggingface_hub.utils import tqdm as hf_tqdm

    class ReportingTqdm(hf_tqdm):
        def __init__(self, *a, **k):
            k["disable"], k["file"] = False, _NullFile()
            super().__init__(*a, **k)
            self._last_pct, self._last_t = -REPORT_EVERY_PCT, 0.0

        def display(self, *a, **k):  # silent: reported through the callback instead of the console
            return None

        def update(self, n=1):
            super().update(n)
            total, done = getattr(self, "total", 0) or 0, getattr(self, "n", 0) or 0
            if getattr(self, "unit", "") != "B" or str(getattr(self, "desc", "")).startswith("Downloading bytes"):
                return
            if total <= 50e6:  # configs and tokenizers: only weights are worth a line
                return
            done = min(done, total)
            pct, now = int(done * 100 / total), time.monotonic()
            if pct - self._last_pct < REPORT_EVERY_PCT and now - self._last_t < REPORT_EVERY_S and done < total:
                return
            self._last_pct, self._last_t = pct, now
            _report(progress, f"Downloading {label}: {done / 1e9:.2f} / {total / 1e9:.2f} GB", done / total)

    return ReportingTqdm


def _snapshot_download(repo_id: str, label: str, progress: ProgressFn | None = None,
                       allow_patterns: list[str] | None = None, **kwargs) -> str:
    """Download (the allowed files of) a repo with progress through the callback."""
    from huggingface_hub import snapshot_download

    _report(progress, f"Downloading {label} from Hugging Face ({repo_id})...", 0.0)
    path = snapshot_download(repo_id, allow_patterns=allow_patterns, tqdm_class=_reporting_tqdm(label, progress),
                             **kwargs)
    _report(progress, f"Downloaded {label}", 1.0)
    return path


def ensure_mediapipe_pose(models_dir: str | Path, url: str, name: str, progress: ProgressFn | None = None) -> Path:
    p = Path(models_dir) / name
    if not p.exists():
        download_url(url, p, "MediaPipe pose model", progress)
    return p
