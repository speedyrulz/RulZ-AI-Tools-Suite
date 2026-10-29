"""Runtime settings. Every knob from the spec is an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


@dataclass
class Settings:
    # llama-server connection
    llama_url: str = field(default_factory=lambda: os.environ.get("LLAMA_URL", "http://127.0.0.1:8080").rstrip("/"))
    # Several llama-servers (e.g. one per GPU): LLAMA_URLS=http://127.0.0.1:8080,http://127.0.0.1:8081.
    # Unset means just llama_url.
    llama_urls: list[str] | None = None
    # Number of slots (--parallel) per llama-server. The service runs this many model workers per server.
    llama_parallel: int = field(default_factory=lambda: _env_int("LLAMA_PARALLEL", 2))
    # Overrides the model id used for cache keys / health; default is read from llama-server.
    model_id: str | None = field(default_factory=lambda: os.environ.get("MODEL_ID") or None)
    request_timeout_s: float = field(default_factory=lambda: _env_float("LLAMA_TIMEOUT_S", 60.0))

    # Image pipeline
    max_edge: int = field(default_factory=lambda: _env_int("MAX_EDGE", 1024))
    min_short_edge: int = field(default_factory=lambda: _env_int("MIN_SHORT_EDGE", 96))
    jpeg_quality: int = field(default_factory=lambda: _env_int("JPEG_QUALITY", 90))
    fetch_timeout_s: float = field(default_factory=lambda: _env_float("FETCH_TIMEOUT_S", 10.0))
    fetch_concurrency: int = field(default_factory=lambda: _env_int("FETCH_CONCURRENCY", 8))
    max_image_bytes: int = field(default_factory=lambda: _env_int("MAX_IMAGE_BYTES", 40 * 1024 * 1024))

    # Confidence = the model's probability of its true/false answer (from token log-probabilities)
    # instead of the number it writes, which is almost always 1.0. CONFIDENCE_FROM_LOGPROBS=0 turns it off.
    confidence_from_logprobs: bool = field(default_factory=lambda: os.environ.get("CONFIDENCE_FROM_LOGPROBS", "1") != "0")

    # Prompting
    max_tags_per_call: int = field(default_factory=lambda: _env_int("MAX_TAGS_PER_CALL", 8))
    max_tokens_base: int = field(default_factory=lambda: _env_int("MAX_TOKENS_BASE", 60))
    max_tokens_per_tag: int = field(default_factory=lambda: _env_int("MAX_TOKENS_PER_TAG", 40))

    # API limits / defaults
    max_images_per_call: int = field(default_factory=lambda: _env_int("MAX_IMAGES_PER_CALL", 64))
    default_threshold: float = field(default_factory=lambda: _env_float("DEFAULT_THRESHOLD", 0.7))

    # Cache (default is inside the package folder, independent of the working directory)
    cache_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("CACHE_PATH", Path(__file__).resolve().parent / "data" / "cache.sqlite3")
        )
    )

    def __post_init__(self) -> None:
        if self.llama_urls is None:
            env = os.environ.get("LLAMA_URLS", "")
            urls = [u.strip().rstrip("/") for u in env.split(",") if u.strip()]
            self.llama_urls = urls or [self.llama_url]

    def max_tokens_for(self, n_tags: int) -> int:
        return self.max_tokens_base + self.max_tokens_per_tag * n_tags
