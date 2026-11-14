"""Application-wide paths and constants."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

APP_NAME = "RulZ Image Organizer"
APP_VERSION = "1.4.0"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".gif", ".webp",
    ".tif", ".tiff", ".heic", ".heif",
}

# Size (longest side) images are scaled to before face detection.
DETECT_MAX_SIDE = 1600
# Faces smaller than this (in pixels, at detection scale) are ignored.
MIN_FACE_PX = 32
# YuNet confidence needed to keep a detection.
DETECT_SCORE = 0.82

# Cosine-similarity thresholds for SFace embeddings (see clustering.py).
DEFAULT_MATCH_THRESHOLD = 0.40
SUGGEST_THRESHOLD = 0.28

THUMB_SHORT_SIDE = 360
FACE_CROP_SIZE = 200


def resource_path(*parts: str) -> Path:
    """Path to a bundled resource (works from source and from a PyInstaller build)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base.joinpath(*parts)


def data_dir() -> Path:
    return _data_dir(os.environ.get("RULZ_DATA_DIR") or "", os.environ.get("LOCALAPPDATA") or "")


def _data_dir_str() -> str:
    return _data_dir_text(os.environ.get("RULZ_DATA_DIR") or "", os.environ.get("LOCALAPPDATA") or "")


# Cached by the environment values: thumb_file()/face_crop_file() run once per grid tile (50k times for a big
# timeline), and resolving the folder each time (Path.home(), Path objects, a mkdir) took seconds.
@lru_cache(maxsize=8)
def _data_dir(override: str, local_appdata: str) -> Path:
    root = Path(override) if override else Path(local_appdata or Path.home()) / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


@lru_cache(maxsize=8)
def _data_dir_text(override: str, local_appdata: str) -> str:
    return str(_data_dir(override, local_appdata))


def settings():
    """App preferences (theme, window geometry) in settings.ini next to the library, so RULZ_DATA_DIR
    isolates them too."""
    from PySide6.QtCore import QSettings

    return QSettings(str(data_dir() / "settings.ini"), QSettings.IniFormat)


def db_path() -> Path:
    return data_dir() / "library.db"


def thumb_path(photo_id: int) -> Path:
    return data_dir() / "thumbs" / f"{photo_id % 256:02x}" / f"{photo_id}.jpg"


def face_crop_path(face_id: int) -> Path:
    return data_dir() / "faces" / f"{face_id % 256:02x}" / f"{face_id}.jpg"


# String versions for building thousands of grid tiles quickly (Path objects are slow in bulk).
def thumb_file(photo_id: int) -> str:
    return f"{_data_dir_str()}\\thumbs\\{photo_id % 256:02x}\\{photo_id}.jpg"


def face_crop_file(face_id: int) -> str:
    return f"{_data_dir_str()}\\faces\\{face_id % 256:02x}\\{face_id}.jpg"


DETECTOR_MODEL = resource_path("models", "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = resource_path("models", "face_recognition_sface_2021dec.onnx")
EXPRESSION_MODEL = resource_path("models", "emotion-ferplus-8.onnx")
PERSON_DETECTOR_MODEL = resource_path("models", "person_detection_mediapipe_2023mar.onnx")
POSE_MODEL = resource_path("models", "pose_estimation_mediapipe_2023mar.onnx")
