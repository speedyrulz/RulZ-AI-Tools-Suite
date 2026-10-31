"""Prompt-gated image crawler: crawl a site, keep the images that match a prompt, save them."""

# The app's version; a release is tagged rulz-image-crawler-v<this>. Keep the pyproject.toml files in step.
__version__ = "1.1.1"

from .config import AppSettings
from .runs import Run, RunManager, RunOptions

__all__ = ["AppSettings", "Run", "RunManager", "RunOptions"]
