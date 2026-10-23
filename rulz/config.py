"""Application-wide paths and constants."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "RulZ Image Organizer"
APP_VERSION = "1.0.0"

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
    override = os.environ.get("RULZ_DATA_DIR")
    if override:
        root = Path(override)
    else:
        root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def db_path() -> Path:
    return data_dir() / "library.db"


def thumb_path(photo_id: int) -> Path:
    return data_dir() / "thumbs" / f"{photo_id % 256:02x}" / f"{photo_id}.jpg"


def face_crop_path(face_id: int) -> Path:
    return data_dir() / "faces" / f"{face_id % 256:02x}" / f"{face_id}.jpg"


DETECTOR_MODEL = resource_path("models", "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = resource_path("models", "face_recognition_sface_2021dec.onnx")
EXPRESSION_MODEL = resource_path("models", "emotion-ferplus-8.onnx")
