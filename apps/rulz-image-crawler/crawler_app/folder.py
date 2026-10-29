"""Local folder scans: list the image files under a folder, and the native folder picker."""

from __future__ import annotations

import os
import threading
from pathlib import Path

# What the engine can decode (TIFF and SVG are not supported).
LOCAL_IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".gif", ".webp", ".bmp", ".avif", ".heic", ".heif"}


def list_image_files(root: Path, recursive: bool, exclude: Path | None = None) -> list[Path]:
    """Image files under root by extension, sorted by path. Folders that can't be read are skipped,
    and so is `exclude` (the app's own output folder, so a scan never re-reads its earlier results)."""
    exclude = exclude.resolve() if exclude else None
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        here = Path(dirpath)
        if exclude is not None:
            dirnames[:] = [d for d in dirnames if (here / d).resolve() != exclude]
        dirnames.sort()
        out.extend(here / f for f in filenames if Path(f).suffix.lower() in LOCAL_IMAGE_EXTS)
        if not recursive:
            break
    return sorted(out)


_picker_lock = threading.Lock()


def pick_folder() -> str | None:
    """Shows the Windows folder picker on this PC (the app only listens on 127.0.0.1, so the person
    using the page is at this machine). Returns None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog

    with _picker_lock:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            path = filedialog.askdirectory(title="RulZ Image Crawler - choose a folder to scan", mustexist=True, parent=root)
        finally:
            root.destroy()
    return str(Path(path)) if path else None
