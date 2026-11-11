"""Sample and probe clips of a video model (PAN 2): the frames as an animated WebP beside the still the galleries,
the scorers and the run report read. A WebP plays in every browser as an image, so it needs no player and no video
codec."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def clip_path(still: str | Path) -> Path:
    """The clip that belongs to a sample or probe picture: the same name, .webp."""
    return Path(still).with_suffix(".webp")


def to_uint8(frames) -> np.ndarray:
    """[F, 3, H, W] in [-1, 1] (a tensor) -> [F, H, W, 3] uint8."""
    return ((frames.permute(0, 2, 3, 1).float().cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype("uint8")


def still_index(n_frames: int) -> int:
    """The frame saved as the clip's still (and scored, for a probe): the middle one."""
    return max(0, int(n_frames) // 2)


def save_clip(path: str | Path, frames: np.ndarray, fps: float, quality: int = 80) -> Path:
    """frames: [F, H, W, 3] uint8. Returns the path written."""
    path = Path(path)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, format="WEBP", save_all=True, append_images=images[1:], duration=max(1, round(1000 / fps)),
                   loop=0, quality=quality, method=3)
    return path
