"""Qwen3-VL image tag matching: local service that judges images against tag rubrics."""

from .config import Settings
from .llm import ModelError, ModelServerDown
from .rubrics import REFERENCE_RUBRICS
from .service import DetectionService, ImageInput, Tag

__all__ = [
    "DetectionService",
    "ImageInput",
    "ModelError",
    "ModelServerDown",
    "REFERENCE_RUBRICS",
    "Settings",
    "Tag",
]
