"""App settings. Every knob is an environment variable; engine knobs stay in image_detection.config."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


@dataclass
class AppSettings:
    # Where each run's matched images go: <output_dir>/<run folder>/
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get("CRAWLER_OUTPUT", REPO_ROOT / "output")))
    # Images being prepared or judged by the model at once. 0 = three per model slot (2 slots per
    # llama-server), which keeps every slot busy with the next image already decoded.
    classify_in_flight: int = field(default_factory=lambda: _env_int("CLASSIFY_IN_FLIGHT", 0))
    fetch_concurrency: int = field(default_factory=lambda: _env_int("CRAWLER_FETCH_CONCURRENCY", 6))
    fetch_timeout_s: float = field(default_factory=lambda: _env_float("CRAWLER_FETCH_TIMEOUT_S", 20.0))
    max_image_bytes: int = field(default_factory=lambda: _env_int("CRAWLER_MAX_IMAGE_BYTES", 40 * 1024 * 1024))
    page_timeout_s: float = field(default_factory=lambda: _env_float("PAGE_TIMEOUT_S", 30.0))
    # Scroll steps per page to trigger lazy loading.
    max_scroll_steps: int = field(default_factory=lambda: _env_int("MAX_SCROLL_STEPS", 30))
    min_short_edge: int = field(default_factory=lambda: _env_int("MIN_SHORT_EDGE", 96))
    headless: bool = field(default_factory=lambda: os.environ.get("CRAWLER_HEADLESS", "1") != "0")
