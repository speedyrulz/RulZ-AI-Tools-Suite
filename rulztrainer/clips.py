"""Clips of a video model (PAN 2, MiniMax H3).

Samples and probes: the frames as an animated WebP beside the still the galleries, the scorers and the run report
read. A WebP plays in every browser as an image, so it needs no player and no video codec.

Training clips: video files in a dataset (train_frames above 1) are read with PyAV (`av`, which bundles its own
FFmpeg). Which frames a clip takes follows ai-toolkit's dataloader (toolkit/dataloader_mixins.py):
  - "fps" (default): real time from the start of the video, at the model's frame rate (24): frame i is the source
    frame nearest i * source_fps / 24, so motion keeps its speed and the tail that does not fit is left out
    (ai-toolkit's auto_frame_count + trim_auto_frame_count_tail, the one it recommends for MiniMax H3). A video too
    short for that is spread over instead (its own fallback).
  - "spread": the whole video squeezed or stretched onto the frame count, evenly spaced (ai-toolkit's default,
    shrink_video_to_frames), which speeds motion up on a long video.
Frames are decoded in one sequential pass (no seeking, which lands off by a frame on B-frame streams)."""
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


def video_info(path: str | Path) -> tuple[int, int, int, float]:
    """(width, height, frame count, frames per second) of a video file from its header. The frame count is the
    container's (or duration x rate when it has none), so it can be off by a few; reading tolerates that."""
    import av

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        w, h = int(st.codec_context.width), int(st.codec_context.height)
        fps = float(st.average_rate or st.guessed_rate or 24.0)
        n = int(st.frames or 0)
        if n <= 0:
            dur = float(st.duration * st.time_base) if st.duration is not None else (
                float(c.duration) / 1e6 if c.duration else 0.0)
            n = int(round(dur * fps))
        rot = _rotation(st)
    if rot in (90, 270):
        w, h = h, w
    return w, h, max(1, n), fps


def _rotation(stream) -> int:
    """The display rotation a phone video carries (0, 90, 180, 270), from its side data or metadata."""
    try:
        for sd in getattr(stream, "side_data", None) or []:
            rot = getattr(sd, "rotation", None)
            if rot is not None:
                return int(round(-float(rot))) % 360
    except Exception:  # noqa: BLE001 - older PyAV: no side data
        pass
    try:
        return int(stream.metadata.get("rotate", 0)) % 360
    except (TypeError, ValueError):
        return 0


def pick_frames(total: int, src_fps: float, frames: int, fps: float, mode: str = "fps") -> list[int]:
    """The source frame indices of a clip of `frames` frames at `fps` (see the module docstring). Ascending, and
    repeated where a short video has to be stretched."""
    total, frames = max(1, int(total)), max(1, int(frames))
    last = total - 1
    if frames == 1:
        return [0]
    ratio = (src_fps / fps) if src_fps and src_fps > 0 else 1.0
    if mode == "fps" and round((frames - 1) * ratio) <= last:
        return [min(int(round(i * ratio)), last) for i in range(frames)]
    step = last / (frames - 1)
    return [min(int(round(i * step)), last) for i in range(frames)]


def _decode(path: str | Path, need: list[int]) -> tuple[dict, int]:
    """The frames at the indices `need` (ascending) as upright PIL RGB pictures, and how many frames were decoded:
    all of the video's when it ended before need[-1]."""
    import av

    got: dict[int, Image.Image] = {}
    wanted = set(need)
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        rot = _rotation(st)
        k = 0
        for frame in c.decode(st):
            if k in wanted:
                img = frame.to_image().convert("RGB")
                if rot in (90, 180, 270):  # counter-clockwise in PIL: the display turns it clockwise
                    img = img.rotate(-rot, expand=True)
                got[k] = img
            k += 1
            if k > need[-1]:
                break
    return got, k


def read_frames(path: str | Path, frames: int, fps: float, mode: str = "fps") -> list:
    """A clip of `frames` frames (PIL RGB pictures, upright) from a video file. A container that holds fewer frames
    than its header said (a WebM or MKV without a frame count, whose duration includes a longer audio track) is read
    again with the frames picked from the ones it has: else every index past its end was its last frame, a frozen
    tail in "spread" mode."""
    _, _, total, src_fps = video_info(path)
    want = pick_frames(total, src_fps, frames, fps, mode)
    need = sorted(set(want))
    got, decoded = _decode(path, need)
    if got and 1 < decoded <= need[-1]:  # the video ended before the last frame asked for
        want = pick_frames(decoded, src_fps, frames, fps, mode)
        need = sorted(set(want))
        if any(i not in got for i in need):
            got, _ = _decode(path, need)
    if not got:
        raise ValueError(f"no video frames could be read from {path}")
    last_read = max(got)
    return [got[min(i, last_read)] for i in want]
