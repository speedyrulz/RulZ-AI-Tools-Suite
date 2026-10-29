"""Downloads the model the app uses (about 9.7 GB) from Hugging Face into image_detection/LLM/.

Files that are already complete are skipped and a broken download picks up where it stopped, so it's safe to run
again. Standard library only, so it works before anything else is installed.

    python image_detection/scripts/download_model.py
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

REPO = "mradermacher/Qwen3-VL-8B-Instruct-abliterated-GGUF"
NAME = "Qwen3-VL-8B-Instruct-abliterated"
FILES = [f"{NAME}.mmproj-f16.gguf", f"{NAME}.Q8_0.gguf"]
DEST = Path(__file__).resolve().parents[1] / "LLM" / f"{NAME}.Q8_0"
CHUNK = 1024 * 1024


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "rulz-image-crawler"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def download(name: str) -> None:
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    target = DEST / name
    total = remote_size(url)
    if target.exists() and target.stat().st_size == total:
        print(f"{name}: already downloaded")
        return
    part = target.with_suffix(target.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > total:
        part.unlink()
        have = 0
    headers = {"User-Agent": "rulz-image-crawler"}
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"{name}: resuming at {have / 1e9:.2f} of {total / 1e9:.2f} GB")
    else:
        print(f"{name}: downloading {total / 1e9:.2f} GB")
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        if have and r.status != 206:  # the server ignored the range: start over
            have = 0
        with open(part, "ab" if have else "wb") as f:
            last = time.monotonic()
            while chunk := r.read(CHUNK):
                f.write(chunk)
                have += len(chunk)
                if time.monotonic() - last >= 5:
                    last = time.monotonic()
                    print(f"  {have / 1e9:.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)", flush=True)
    if have != total:
        raise SystemExit(f"{name}: the download stopped early ({have} of {total} bytes). Run this again to resume.")
    part.replace(target)
    print(f"{name}: done")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Model folder: {DEST}", flush=True)
    for name in FILES:
        download(name)


if __name__ == "__main__":
    try:
        main()
    except OSError as e:
        sys.exit(f"Download failed: {e}. Check your internet connection and run it again; it resumes.")
