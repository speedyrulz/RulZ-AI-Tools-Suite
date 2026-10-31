"""Application-wide paths and constants."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

APP_NAME = "RulZ Image Preparer"
APP_VERSION = "0.4.0"

THUMB_MAX_SIDE = 400
DEFAULT_STRENGTH = 50


def resource_path(*parts: str) -> Path:
    """Path to a bundled resource (works from source and from a PyInstaller build)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base.joinpath(*parts)


def data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    path = Path(base) / "RulZ" / "Image Preparer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def settings():
    from PySide6.QtCore import QSettings

    return QSettings("RulZ", APP_NAME)
