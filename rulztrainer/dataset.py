"""Image/caption dataset with aspect-ratio buckets and on-disk latent + text-embedding cache."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file

from .archs import get_arch
from .atomic import save_safetensors
from .config import VIDEO_EXTS, TrainConfig
from .caption_text import read_caption_text
from .pictures import file_colour_tag, file_has_alpha, has_alpha, to_rgb

log = logging.getLogger("rulztrainer")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SKIP_DIRS = {".rulz_cache", "excluded_by_look"}  # folders inside a dataset that never train (cache, look filter)


def skipped(path: Path, root: Path | None = None) -> bool:
    """True for a file that never trains, below `root` when given: one inside SKIP_DIRS, and any hidden file or
    folder (a name starting with "."): a NAS's .thumbnails or .@__thumb, a tool's own cache, macOS's ._ copies.
    (OneTrainer skips hidden folders too, #1303.) Without a root only SKIP_DIRS count, as the dataset's own parent
    folders may be hidden."""
    if root is None:
        return any(p in SKIP_DIRS for p in path.parts)
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return any(p in SKIP_DIRS for p in path.parts)
    return any(p in SKIP_DIRS or p.startswith(".") for p in parts)


@dataclass
class Item:
    image: Path
    caption: str
    width: int  # bucket size (pixels)
    height: int
    latent_path: Path
    latent_flip_path: Path | None
    text_path: Path
    src_box: tuple | None = None  # crop of the source image (x0, y0, x1, y1) for augmentation items
    category: str = "image"  # image | face_crop | bust_crop
    region_path: Path | None = None  # latent-cell loss weight map (region-weighted loss)
    side: int = 0  # training resolution this item was bucketed for (square-equivalent side, area = side^2)
    repeats: int = 1  # times the item comes up per epoch (its dataset's repeats)
    dataset: str = ""  # the dataset folder it came from
    loss_scale: float = 1.0  # its loss weight (LoRA Isolation's side-by-side pairs train at isolation_pair_weight)
    dop_caption: str | None = None  # LoRA Isolation: the trigger-free caption of its DOP pass (None: no DOP)
    dop_text_path: Path | None = None  # its cached text features
    pair_side: str = ""  # LoRA Isolation's side-by-side pairs: the subject's panel ("left" / "right"; never mirrored)
    frames: int = 1  # above 1: a video clip of that many frames (category "clip", `image` is the video file)
    latent_frames: int = 1  # its latent frames (the VAE packs several frames into one)
    reg: bool = False  # a picture of the regularisation dataset (scan_reg): trains whole and plain at reg_weight
    repeat_mult: int = 1  # the part of `repeats` that comes from its "<N>_<name>" folder (folder_repeats)
    # caption_lines: the picture's other caption lines (`caption` / `text_path` are the first); each time it trains one
    # of all of them is picked at random
    alt_captions: list = field(default_factory=list)
    alt_text_paths: list = field(default_factory=list)

    def all_text_paths(self) -> set[str]:
        """Every cached text this picture can train on (caption dropout's aside)."""
        return {str(self.text_path), *(str(p) for p in self.alt_text_paths)}

    @property
    def tokens(self) -> int:
        """Latent cells of the item (16 px each, every latent frame): what its memory and time grow with."""
        return (self.width // 16) * (self.height // 16) * max(1, self.latent_frames)

    @property
    def key(self) -> str:
        """Stable identity for weights/verdicts: the image path, plus a suffix for augmentation crops."""
        if self.category == "image":
            return str(self.image)
        return f"{self.image}#{self.category}"


@dataclass
class RegionBatch:
    weight: torch.Tensor  # [B, 1, h, w]
    prior: torch.Tensor  # [B, 1, h, w] bool
    meta: list  # per-sample RegionMap (face cells, kps, has_person)
    text_paths: list = field(default_factory=list)  # per sample: the cached text it trained on (caption dropout too)


def bucket_size(w: int, h: int, cfg: TrainConfig, side: int | None = None) -> tuple[int, int]:
    """Pick a (W,H) multiple of 16 (the family's size_multiple: 32 for Qwen-Image 2.1) with area ~= side^2 preserving
    aspect ratio (side defaults to cfg.resolution).

    min/max_bucket_side are applied as a uniform rescale first (so the aspect ratio survives the limit) and only
    as a hard per-side clamp when an extreme aspect ratio cannot satisfy both limits."""
    m = get_arch(cfg).size_multiple
    side = int(side or cfg.resolution)
    if not cfg.enable_buckets:
        return side // m * m, side // m * m
    aspect = w / h
    bw = math.sqrt(side * side * aspect)
    bh = bw / aspect
    # the limits are relative to the side: the minimum never exceeds half of it and the maximum never drops below
    # twice it, so every resolution keeps its pixel count (512 -> ~512x512 of area, 2048 -> ~2048x2048) instead of
    # being inflated or shrunk until a fixed limit is met
    lo, hi = min(cfg.min_bucket_side, side // 2 // m * m), max(cfg.max_bucket_side, side * 2)
    lo, hi = -(-lo // m) * m, hi // m * m
    scale = 1.0
    if max(bw, bh) > hi:
        scale = hi / max(bw, bh)
    if min(bw, bh) * scale < lo:
        scale = lo / min(bw, bh)
    bw, bh = bw * scale, bh * scale
    bw = int(round(bw / m) * m)
    bh = int(round(bh / m) * m)
    bw = max(lo, min(hi, bw))
    bh = max(lo, min(hi, bh))
    return bw, bh


def items_for_side(items: list["Item"], side: int) -> list["Item"]:
    """The items of one training resolution (resolution_step_up's stages); video clips have their own size
    (video_resolution) and train at every stage."""
    return [it for it in items if it.side == side or it.frames > 1]


def describe_sides(cfg: TrainConfig) -> str:
    from .config import side_to_mp

    return ", ".join(f"{side_to_mp(s):g} MP" for s in cfg.training_sides())


def upright_header(path: Path) -> tuple[int, int, bool]:
    """(width, height, transparent?) as ImageOps.exif_transpose turns the picture, from the header alone: transposing
    decodes the whole image, which a scan of every training photo at every start does not need."""
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # Orientation: the rotations by 90 degrees
            w, h = h, w
        return w, h, has_alpha(im)


def upright_size(path: Path) -> tuple[int, int]:
    """(width, height) as ImageOps.exif_transpose turns the picture (upright_header without the transparency)."""
    return upright_header(path)[:2]


def latent_key(path: Path, transparent: bool | None = None, colour: str | None = None) -> str:
    """What a picture's cached latents are named after: the file (path, date, size), for a picture with transparency
    the white it is laid on, and for one that is converted to sRGB (another colour profile) or scaled down from 16-bit
    grey that conversion (colour_tag), so latents cached before that are encoded again once. Every other picture keeps
    its old name, and its latents."""
    from .regions import _file_key

    if transparent is None:
        transparent = file_has_alpha(path)
    if colour is None:
        colour = file_colour_tag(path)
    return _file_key(path) + ("|on-white" if transparent else "") + colour


def load_image_fit(path: Path, w: int, h: int, src_box=None) -> Image.Image:
    img = Image.open(path)
    img = to_rgb(ImageOps.exif_transpose(img))
    if src_box is not None:
        img = img.crop(tuple(int(round(v)) for v in src_box))
    return ImageOps.fit(img, (w, h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def read_caption(image: Path, cfg: TrainConfig, trigger: str | None = None) -> str:
    """The image's caption: the architecture picks the file (FIBO 2.0 prefers a structured <name>.json caption), cleans
    a JSON caption the way the model was trained, and puts the trigger word (its dataset's, else the run's) where it
    belongs."""
    arch = get_arch(cfg)
    cap = arch.caption_file(image, cfg)
    text = read_caption_text(cap).strip() if cap.is_file() else ""
    if cap.suffix.lower() == ".json" and text:
        text = arch.clean_prompt(text)
    return arch.add_trigger(text, cfg.trigger_word if trigger is None else trigger)


def read_captions(image: Path, cfg: TrainConfig, trigger: str | None = None) -> list[str]:
    """The picture's captions, each with the trigger word: one (read_caption) unless caption_lines is on and the
    caption is a plain-text file, then one per non-empty line (the same line twice counts once). A file with no
    line in it is the one empty caption, as without the setting."""
    first = read_caption(image, cfg, trigger)
    if not cfg.caption_lines:
        return [first]
    arch = get_arch(cfg)
    cap = arch.caption_file(image, cfg)
    if not cap.is_file() or cap.suffix.lower() == ".json":
        return [first]
    trig = cfg.trigger_word if trigger is None else trigger
    out: list[str] = []
    for line in read_caption_text(cap).splitlines():
        line = line.strip()
        if line:
            c = arch.add_trigger(line, trig)
            if c not in out:
                out.append(c)
    return out or [first]


FOLDER_REPEATS_RE = re.compile(r"^(\d+)_.+")


def folder_repeat_of(path: Path, root: Path) -> tuple[int, str]:
    """(N, the folder's path below `root`) for the nearest folder named "<N>_<name>" (N a whole number from 1) between
    `root` and the file: kohya's repeats. `root` itself never counts. (1, "") when there is none."""
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        return 1, ""
    for i in range(len(parts) - 1, -1, -1):
        m = FOLDER_REPEATS_RE.match(parts[i])
        if m and int(m.group(1)) >= 1:
            return int(m.group(1)), "/".join(parts[:i + 1])
    return 1, ""


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def scan_dataset(cfg: TrainConfig) -> list[Item]:
    """Items of every dataset of the run (cfg.dataset_specs()): one per image and training resolution of its dataset,
    carrying the dataset's repeats; latents and region maps in the dataset's own cache (cfg.dataset_cache_dir), named
    after the file's path, time and size: a picture replaced in place is encoded again."""
    specs = cfg.dataset_specs()
    for spec in specs:  # before any cache folder is made: making one would create a missing dataset folder
        if not Path(spec.path).is_dir():
            raise FileNotFoundError(f"Dataset dir not found: {spec.path}")
    (cfg.resolved_cache_dir() / "text").mkdir(parents=True, exist_ok=True)
    arch = get_arch(cfg)
    tag = arch.latent_tag  # latents of another VAE get their own files
    clip_frames = arch.snap_frames(cfg.train_frames) if cfg.train_frames > 1 and arch.supports_video else 1
    if clip_frames > 1:
        log.info("Video clips: %d frames at %g fps (%d asked for, %s), %d latent frames each, around %d px "
                 "(video_resolution)", clip_frames, arch.video_fps, cfg.train_frames,
                 "real time from the start" if cfg.train_video_mode == "fps" else "spread over the whole video",
                 arch.latent_frames(clip_frames), cfg.video_resolution or cfg.resolution)
    items = []
    summary = []
    for spec in specs:
        root = Path(spec.path)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset dir not found: {root}")
        cache = cfg.dataset_cache_dir(root)
        (cache / "latents").mkdir(parents=True, exist_ok=True)
        sides = cfg.dataset_sides(spec)
        n_images = 0
        n_skipped_videos = 0
        folders: dict[str, list[int]] = {}  # folder_repeats: folder -> [its N, pictures]
        for p in sorted(root.rglob("*")):
            mult, folder = folder_repeat_of(p, root) if cfg.folder_repeats else (1, "")
            if p.suffix.lower() in VIDEO_EXTS and not skipped(p, root):
                if clip_frames > 1:
                    clip = _clip_item(p, cfg, spec, root, cache, tag, clip_frames, arch)
                    if clip is not None:
                        clip.repeats, clip.repeat_mult = spec.repeats * mult, mult
                        if folder:
                            folders.setdefault(folder, [mult, 0])[1] += 1
                        items.append(clip)
                        n_images += 1
                else:
                    n_skipped_videos += 1
                continue
            if p.suffix.lower() not in IMAGE_EXTS or skipped(p, root):
                continue
            try:
                w, h, transparent = upright_header(p)
            except Exception as e:  # noqa: BLE001
                log.warning("Skipping unreadable image %s: %s", p, e)
                continue
            n_images += 1
            colour = file_colour_tag(p)
            captions = read_captions(p, cfg, spec.trigger or cfg.trigger_word)
            caption, alts = captions[0], captions[1:]
            if folder:
                folders.setdefault(folder, [mult, 0])[1] += 1
            seen: set[tuple[int, int]] = set()
            for side in sides:
                bw, bh = bucket_size(w, h, cfg, side)
                if (bw, bh) in seen:  # two resolutions clamped to the same bucket: keep one copy
                    continue
                seen.add((bw, bh))
                stem = _hash(latent_key(p, transparent, colour)) + f"_{bw}x{bh}"
                items.append(Item(
                    image=p, caption=caption, width=bw, height=bh,
                    latent_path=cache / "latents" / f"{stem}{tag}.safetensors",
                    latent_flip_path=((cache / "latents" / f"{stem}_flip{tag}.safetensors") if cfg.flip_augment
                                      else None),
                    text_path=text_cache_path(cfg, caption),
                    region_path=cache / "regions" / f"{stem}_regions.safetensors",
                    side=side, repeats=spec.repeats * mult, dataset=str(root), repeat_mult=mult,
                    alt_captions=alts, alt_text_paths=[text_cache_path(cfg, c) for c in alts],
                ))
        for folder, (n, count) in sorted(folders.items()):
            log.info("Dataset %s: folder %s: %d picture%s, %d repeat%s per epoch (from its name%s)", root, folder,
                     count, "" if count == 1 else "s", n * spec.repeats, "" if n * spec.repeats == 1 else "s",
                     f" x the dataset's {spec.repeats}" if spec.repeats != 1 else "")
        if n_skipped_videos:
            log.info("Dataset %s: %d video file(s) left out: %s", root, n_skipped_videos,
                     "set train_frames above 1 to train them as clips" if getattr(arch, "supports_video", False)
                     else f"{arch.label} trains on pictures (video clips need PAN 2 or MiniMax H3)")
        if not n_images:
            raise ValueError(f"No images found in {root}")
        summary.append((root, n_images, sides, spec.repeats))
    if len(specs) > 1:
        from .config import side_to_mp

        for root, n, sides, reps in summary:
            log.info("Dataset %s: %d images at %s, %d repeat%s per epoch", root, n,
                     ", ".join(f"{side_to_mp(s):g} MP" for s in sides), reps, "" if reps == 1 else "s")
    elif len(summary[0][2]) > 1:
        log.info("Training resolutions: %s (%d items = %d images x %d resolutions)", describe_sides(cfg), len(items),
                 summary[0][1], len(summary[0][2]))
    return items


REG_NAMED = 5  # file names a message about the regularisation folder lists before "and N more"


def _named(paths: list[Path]) -> str:
    return ", ".join(p.name for p in paths[:REG_NAMED]) + (f" and {len(paths) - REG_NAMED} more"
                                                           if len(paths) > REG_NAMED else "")


def reg_triggers(cfg: TrainConfig) -> list[str]:
    """Every trigger word of the run (the run's and each dataset's), longest first."""
    words = {str(cfg.trigger_word or "").strip()} | {spec.trigger for spec in cfg.dataset_specs()}
    return sorted((w for w in words if w), key=len, reverse=True)


def scan_reg(cfg: TrainConfig) -> list[Item]:
    """The regularisation dataset's items (cfg.reg_dir; [] when it is off): one per picture and training resolution of
    the run, marked `reg`, at the loss weight reg_weight, with its own caption and no trigger word, in the folder's own
    latent cache. They are kept apart from the datasets' items: nothing that reads the subject (face analysis, crops,
    region maps, held-out photos, the look filter, likeness references, the loss watch, training maps, the copy
    check) sees them; only the caches and the image queue (BucketBatcher) do.

    Every picture needs a caption that describes it: one without is refused (it would train on the empty prompt, and
    hold nothing in place under the words the subject is prompted with). A caption that uses a trigger word is warned
    about: that picture teaches the trigger someone else's look. Video files are left out."""
    root = str(cfg.reg_dir or "").strip()
    if not root:
        return []
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Regularisation folder not found: {root}")
    arch = get_arch(cfg)
    tag = arch.latent_tag
    cache = cfg.dataset_cache_dir(root)
    (cache / "latents").mkdir(parents=True, exist_ok=True)
    sides = cfg.training_sides()
    triggers = [t.lower() for t in reg_triggers(cfg)]
    items: list[Item] = []
    uncaptioned: list[Path] = []
    triggered: list[Path] = []
    n_images = n_videos = 0
    for p in sorted(root.rglob("*")):
        if skipped(p, root):
            continue
        if p.suffix.lower() in VIDEO_EXTS:
            n_videos += 1
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            w, h, transparent = upright_header(p)
        except Exception as e:  # noqa: BLE001
            log.warning("Skipping unreadable regularisation image %s: %s", p, e)
            continue
        cap_file = arch.caption_file(p, cfg)  # the file itself: a family with JSON captions wraps an empty one
        raw = read_caption_text(cap_file).strip() if cap_file.is_file() else ""
        if raw and cap_file.suffix.lower() == ".json":
            try:
                raw = raw if json.loads(raw) else ""  # {} or [] describes nothing
            except ValueError:
                pass  # not JSON after all: read as the plain caption it is
        if not raw:
            uncaptioned.append(p)
            continue
        caption = read_caption(p, cfg, "")
        colour = file_colour_tag(p)
        n_images += 1
        if any(t in caption.lower() for t in triggers):
            triggered.append(p)
        seen: set[tuple[int, int]] = set()
        for side in sides:
            bw, bh = bucket_size(w, h, cfg, side)
            if (bw, bh) in seen:
                continue
            seen.add((bw, bh))
            stem = _hash(latent_key(p, transparent, colour)) + f"_{bw}x{bh}"
            items.append(Item(
                image=p, caption=caption, width=bw, height=bh,
                latent_path=cache / "latents" / f"{stem}{tag}.safetensors",
                latent_flip_path=(cache / "latents" / f"{stem}_flip{tag}.safetensors") if cfg.flip_augment else None,
                text_path=text_cache_path(cfg, caption), region_path=None, side=side, repeats=1, dataset=str(root),
                loss_scale=float(cfg.reg_weight), reg=True))
    if uncaptioned:
        raise ValueError(f"{len(uncaptioned)} regularisation picture(s) in {root} have no caption: {_named(uncaptioned)}. "
                         "Every regularisation picture needs a caption that describes it (and does not use the "
                         "trigger word); caption them, or move them out of the folder")
    if not n_images:
        raise ValueError(f"No images found in the regularisation folder {root}")
    if triggered:
        log.warning("Regularisation: %d caption(s) use a trigger word of this run (%s): those pictures teach the "
                    "trigger what they show. Take the word out of their captions", len(triggered), _named(triggered))
    if n_videos:
        log.info("Regularisation folder %s: %d video file(s) left out (its pictures only)", root, n_videos)
    return items


def reg_quotas(sizes: dict, subject: int, per_epoch: int) -> dict:
    """How many regularisation pictures each size bucket gives an epoch. sizes: {bucket: its regularisation pictures};
    subject: the subject pictures an epoch trains (repeats counted); per_epoch: reg_per_epoch (-1 = as many as
    `subject`, 0 = all of them, N = that many; never more than there are). Each bucket gives a fixed share by its size
    (largest remainders first, so the shares add up to the count): an epoch has the same number of steps every time."""
    total = sum(sizes.values())
    want = total if per_epoch == 0 else min(total, max(0, subject if per_epoch < 0 else per_epoch))
    if want >= total:
        return dict(sizes)
    raw = {k: want * n / total for k, n in sizes.items()}
    quota = {k: int(v) for k, v in raw.items()}
    for k in sorted(sizes, key=lambda k: raw[k] - quota[k], reverse=True)[:want - sum(quota.values())]:
        quota[k] += 1
    return quota


def _clip_item(p: Path, cfg: TrainConfig, spec, root: Path, cache: Path, tag: str, frames: int, arch) -> Item | None:
    """One video file as a training clip of `frames` frames (None when it cannot be read). The caption sits beside it
    as for a picture; a family whose captions carry the clip's length (PAN 2) has it wrapped here, ahead of the text
    cache."""
    from . import clips
    from .regions import _file_key

    try:
        w, h, _, _ = clips.video_info(p)
    except Exception as e:  # noqa: BLE001
        log.warning("Skipping unreadable video %s: %s", p, e)
        return None
    side = int(cfg.video_resolution or 0) or cfg.dataset_sides(spec)[0]
    bw, bh = bucket_size(w, h, cfg, side)
    caption = arch.render_prompt(read_caption(p, cfg, spec.trigger or cfg.trigger_word), frames)
    stem = _hash(_file_key(p) + f"|clip{frames}:{cfg.train_video_mode}:{arch.video_fps:g}") + f"_{bw}x{bh}"
    return Item(
        image=p, caption=caption, width=bw, height=bh,
        latent_path=cache / "latents" / f"{stem}{tag}.safetensors",
        latent_flip_path=(cache / "latents" / f"{stem}_flip{tag}.safetensors") if cfg.flip_augment else None,
        text_path=text_cache_path(cfg, caption), region_path=None, category="clip", side=side,
        repeats=spec.repeats, dataset=str(root), frames=frames, latent_frames=arch.latent_frames(frames))


def load_clip_fit(path: Path, frames: int, w: int, h: int, cfg: TrainConfig, fps: float) -> torch.Tensor:
    """A clip's frames resized and centre-cropped to its bucket: [3, F, H, W] in [-1, 1]."""
    from . import clips

    pics = clips.read_frames(path, frames, fps, cfg.train_video_mode)
    arr = np.stack([np.asarray(ImageOps.fit(to_rgb(im), (w, h), method=Image.Resampling.LANCZOS,
                                            centering=(0.5, 0.5))) for im in pics])  # [F, H, W, 3]
    return torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 127.5 - 1.0


def crop_caption(template: str, caption: str, cfg: TrainConfig) -> str:
    raw = caption
    if cfg.trigger_word and raw.startswith(cfg.trigger_word):
        raw = raw[len(cfg.trigger_word):].lstrip(", ").strip()
    return template.format(caption=caption, raw=raw, trigger=cfg.trigger_word).strip()


def augment_items(items: list[Item], cfg: TrainConfig, analysis: dict[str, dict]) -> list[Item]:
    """Add face / bust crop items for images with a detected face (crop_augment = face | face+bust).

    Boxes are squares in source pixels around the face box, clamped into the image; items reuse the normal
    latent/text/region caches with their own stems and captions from the templates."""
    from .regions import face_fraction

    mode = (cfg.crop_augment or "none").lower()
    if mode == "none":
        return items
    out = list(items)
    added = {"face_crop": 0, "bust_crop": 0}
    for it in items:
        if it.category != "image":
            continue
        rec = analysis.get(str(it.image.resolve()))
        if not rec or not rec.get("bbox") or rec.get("det", 0.0) < 0.5:
            continue
        W, H = rec["W"], rec["H"]
        x0, y0, x1, y1 = rec["bbox"]
        fw, fh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side0 = max(fw, fh)
        frac = face_fraction(rec)
        plans = []
        if frac <= cfg.crop_skip_face_frac:
            plans.append(("face_crop", cfg.crop_face_scale, cy, cfg.crop_face_caption))
        if mode == "face+bust":
            plans.append(("bust_crop", cfg.crop_bust_scale, cy + 0.75 * side0, cfg.crop_bust_caption))
        for category, scale, center_y, template in plans:
            side = scale * side0
            side = min(side, W, H)
            min_px = cfg.crop_augment_min_px or it.side or cfg.resolution  # upscaled crops teach blurry faces: skip
            if side < min_px:
                continue
            bx0, by0 = cx - side / 2, center_y - side / 2
            bx0 = min(max(0.0, bx0), W - side)
            by0 = min(max(0.0, by0), H - side)
            box = (int(round(bx0)), int(round(by0)), int(round(bx0 + side)), int(round(by0 + side)))
            if (box[2] - box[0]) * (box[3] - box[1]) > cfg.crop_skip_cover * W * H:
                continue
            bw, bh = bucket_size(box[2] - box[0], box[3] - box[1], cfg, it.side or None)
            caption = crop_caption(template, it.caption, cfg)
            alts = []  # caption_lines: the picture's other lines, each in the crop's template
            for c in getattr(it, "alt_captions", None) or []:
                c = crop_caption(template, c, cfg)
                if c != caption and c not in alts:
                    alts.append(c)
            stem = _hash(f"{latent_key(it.image)}|crop{box[0]},{box[1]},{box[2]},{box[3]}") + f"_{bw}x{bh}"
            tag = get_arch(cfg).latent_tag
            cache = it.latent_path.parent.parent  # the source image's dataset cache
            out.append(Item(
                image=it.image, caption=caption, width=bw, height=bh,
                latent_path=cache / "latents" / f"{stem}{tag}.safetensors",
                latent_flip_path=(cache / "latents" / f"{stem}_flip{tag}.safetensors") if cfg.flip_augment else None,
                text_path=text_cache_path(cfg, caption),
                src_box=box, category=category,
                region_path=cache / "regions" / f"{stem}_regions.safetensors",
                side=it.side, repeats=it.repeats, dataset=it.dataset, repeat_mult=it.repeat_mult,
                alt_captions=alts, alt_text_paths=[text_cache_path(cfg, c) for c in alts],
            ))
            added[category] += 1
    log.info("Crop augmentation: +%d face crops, +%d bust crops (%d items total)", added["face_crop"],
             added["bust_crop"], len(out))
    return out


def empty_text_path(cfg: TrainConfig) -> Path:
    """The cached embedding of the empty prompt (FIBO 2.0's null caption)."""
    return text_cache_path(cfg, get_arch(cfg).empty_prompt)


def dropout_prompt(cfg: TrainConfig) -> str:
    """What caption dropout trains on: the trigger word alone (caption_dropout_keep_trigger, so the trigger by itself
    summons the subject, the way the LoRA is prompted), else the empty prompt."""
    arch = get_arch(cfg)
    if cfg.caption_dropout_keep_trigger and cfg.trigger_word:
        return arch.add_trigger(arch.empty_prompt, cfg.trigger_word)  # FIBO: into the null caption's short_description
    return arch.empty_prompt


def dropout_text_path(cfg: TrainConfig) -> Path:
    return text_cache_path(cfg, dropout_prompt(cfg))


# ---------------------------------------------------------------- caching


def latents_todo(items: list[Item], cfg: TrainConfig) -> list[Item]:
    return [it for it in items if cfg.recache or not it.latent_path.exists()
            or (it.latent_flip_path and not it.latent_flip_path.exists())]


def cache_latents(items: list[Item], cfg: TrainConfig, vae, progress=None, vaes: list | None = None) -> None:
    """Encode and save the latents still missing. `vaes` (one VAE per GPU) splits the images across GPUs."""
    from .multigpu import run_queue

    arch = get_arch(cfg)
    encode_images = arch.encode_images
    todo = latents_todo(items, cfg)
    workers = [(v, v.device) for v in (vaes or [vae])]
    log.info("Caching latents: %d of %d images need encoding%s", len(todo), len(items),
             f" (split across {', '.join(str(d) for _, d in workers)})" if len(workers) > 1 else "")

    def prepare(it):
        if it.frames > 1:  # a video clip: [3, F, H, W]
            return load_clip_fit(it.image, it.frames, it.width, it.height, cfg, arch.video_fps)
        img = load_image_fit(it.image, it.width, it.height, it.src_box)
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 127.5 - 1.0

    def enc(v, t):
        """[3, H, W] or a clip [3, F, H, W] -> its latents on the CPU."""
        return (arch.encode_video if t.ndim == 4 else encode_images)(v, t.unsqueeze(0))[0].cpu()

    # the next pictures are decoded and resized on CPU threads while the VAE encodes this one
    prefetch = LatentPrefetch(todo, prepare, ahead=PREFETCH_AHEAD + len(workers))

    @torch.no_grad()
    def encode(worker, i):
        v = worker[0]
        it = todo[i]
        t = prefetch.get(i)
        lat = enc(v, t)
        save_safetensors({"latent": lat.contiguous()}, it.latent_path)
        if it.latent_flip_path:
            lat_f = enc(v, t.flip(-1))
            save_safetensors({"latent": lat_f.contiguous()}, it.latent_flip_path)

    count = [0]

    def done(i, it, _):
        count[0] += 1
        if progress:
            progress(count[0], len(todo))

    try:
        run_queue(workers, list(range(len(todo))), encode, on_done=done)
    finally:
        prefetch.close()


PREFETCH_AHEAD = 2  # pictures prepared ahead of the encoders, plus one per encoder


class LatentPrefetch:
    """`fn(items[i])` run a few items ahead on a small thread pool: get(i) returns item i's result (waiting for it
    if needed) and queues the next ones up to `ahead` past i. The encoders take items roughly in order (several GPUs
    pull from one queue), so a window past the latest request covers them all."""

    def __init__(self, items: list, fn, ahead: int = 3, threads: int = 2):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        self.items, self.fn, self.ahead = items, fn, max(1, int(ahead))
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(threads)), thread_name_prefix="latent-prep")
        self.futures: dict = {}
        self.next = 0  # the first item not queued yet
        self.lock = threading.Lock()

    def get(self, i: int):
        with self.lock:
            while self.next < min(len(self.items), i + 1 + self.ahead):
                self.futures[self.next] = self.pool.submit(self.fn, self.items[self.next])
                self.next += 1
            fut = self.futures.pop(i, None)
        if fut is None:  # asked twice or out of the window: prepared here
            return self.fn(self.items[i])
        return fut.result()

    def close(self) -> None:
        with self.lock:
            for fut in self.futures.values():
                fut.cancel()
            self.futures.clear()
        self.pool.shutdown(wait=True)


def text_jobs(items: list[Item], cfg: TrainConfig, extra_prompts: list[str] | None = None) -> tuple[dict, int]:
    """(cache path -> text for every prompt still missing, total prompt count)."""
    jobs: dict[Path, str] = {}
    for it in items:
        jobs.setdefault(it.text_path, it.caption)
        for c, tp in zip(it.alt_captions, it.alt_text_paths):  # caption_lines: its other lines
            jobs.setdefault(tp, c)
        if it.dop_text_path is not None and it.dop_caption is not None:  # LoRA Isolation's trigger-free captions
            jobs.setdefault(it.dop_text_path, it.dop_caption)
    jobs.setdefault(dropout_text_path(cfg), dropout_prompt(cfg))
    for p in extra_prompts or []:
        jobs.setdefault(text_cache_path(cfg, p), p)
    return {k: v for k, v in jobs.items() if cfg.recache or not k.exists()}, len(jobs)


TEXT_BATCH = 16  # prompts per text-encoder forward when caching: one at a time leaves a big encoder launch-bound
TEXT_BATCH_CHARS = 16000  # at most this many padded caption characters per forward (~4k tokens): long captions batch less


def text_batches(jobs: list[tuple[Path, str]], size: int = TEXT_BATCH,
                 max_chars: int = TEXT_BATCH_CHARS) -> list[list[tuple[Path, str]]]:
    """(path, text) jobs grouped for batched encoding: sorted by length, so a batch (right-padded to its longest
    prompt) pads little, and cut at `size` prompts or `max_chars` padded characters, whichever comes first."""
    out: list[list[tuple[Path, str]]] = []
    cur: list[tuple[Path, str]] = []
    for job in sorted(jobs, key=lambda j: len(j[1])):
        if cur and (len(cur) >= size or (len(cur) + 1) * (len(job[1]) + 1) > max_chars):
            out.append(cur)
            cur = []
        cur.append(job)
    if cur:
        out.append(cur)
    return out


def cache_text(items: list[Item], cfg: TrainConfig, text_encoder, tokenizer,
               extra_prompts: list[str] | None = None, progress=None, encoders: list | None = None,
               max_prompts: int | None = None, encode_fn=None, encode_batch_fn=None,
               batch_size: int = TEXT_BATCH) -> None:
    """Encode and save the missing prompt embeddings. `encoders` is a list of (text_encoder, tokenizer, device),
    one per GPU, to split the prompts across GPUs (each needs its own tokenizer: fast tokenizers are not
    thread-safe). `max_prompts` encodes only the first N missing prompts (a timing probe). `encode_batch_fn` is the
    architecture's encode_prompts(text_encoder, tokenizer, texts, max_length), which encodes up to `batch_size`
    prompts per forward; `encode_fn`, a one-prompt encode_prompt, is used one prompt at a time instead. A batch that
    runs out of GPU memory is halved and retried; progress counts prompts."""
    from .multigpu import run_queue

    if encode_batch_fn is None:
        if encode_fn is not None:
            encode_batch_fn = lambda te, tok, texts, n: [encode_fn(te, tok, t, n) for t in texts]  # noqa: E731
        else:
            encode_batch_fn = get_arch(cfg).encode_prompts

    todo, total = text_jobs(items, cfg, extra_prompts)
    workers = encoders or [(text_encoder, tokenizer, "cpu")]
    if max_prompts is not None:
        log.info("Caching text embeddings: timing the first %d of %d missing prompts", min(max_prompts, len(todo)),
                 len(todo))
        todo = dict(list(todo.items())[:max_prompts])
    else:
        log.info("Caching text embeddings: %d of %d prompts need encoding%s", len(todo), total,
                 f" (split across {', '.join(str(w[2]) for w in workers)})" if len(workers) > 1 else "")

    @torch.no_grad()
    def encode(worker, batch):
        try:
            embs = encode_batch_fn(worker[0], worker[1], [text for _, text in batch], cfg.max_token_length)
        except torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(batch) // 2
            log.info("Text caching: a batch of %d prompts ran out of GPU memory; encoding it in halves", len(batch))
            encode(worker, batch[:half])
            encode(worker, batch[half:])
            return
        for (path, text), emb in zip(batch, embs):
            save_safetensors({"embeds": emb.contiguous()}, path, metadata={"prompt": text[:2000]})

    count = [0]

    def done(i, batch, _):
        count[0] += len(batch)
        if progress:
            progress(count[0], len(todo))

    batches = text_batches(list(todo.items()), size=max(1, int(batch_size)))
    run_queue([(w, w[2]) for w in workers], batches, lambda w, batch: encode(w[0], batch), on_done=done)


def text_cache_path(cfg: TrainConfig, prompt: str) -> Path:
    """Cached text features of a prompt. Each architecture has its own text encoder, so each has its own suffix.
    MOSS-Image's is "_q35": its single-file mode once read prompts with Krea 2's tokenizer when a Krea 2 folder was in
    models/, and the features cached then (no suffix) must not be reused."""
    arch = cfg.arch_name()
    tag = "_q35" if arch == "moss_image" else f"_{arch}"
    if arch not in ("minimax_h3", "minimax_h3_image"):
        # since 3.18.0 a quantized text encoder runs weight-only, as ComfyUI does (int8 / FP4 activations before, which
        # moved the features a few percent): features cached before are not mixed with new ones. H3's ran so already
        tag += "_wo"
    length = cfg.max_token_length
    if arch == "pan2":  # PAN 2 encodes 1536 tokens at the default 1024 since 3.18.0: its caches cut at 1024 are not
        from .pan2.loading import text_length  # reused (the key names the length encoded)

        length = text_length(length)
    return cfg.resolved_cache_dir() / "text" / f"{_hash(prompt)}_{length}{tag}.safetensors"


# ---------------------------------------------------------------- batching


class BucketBatcher:
    """Yields batches of Items with identical bucket size (clips batch with clips of their length), shuffled each epoch. An item with repeats N comes up N
    times per epoch (its dataset's repeats).

    Regularisation pictures (Item.reg) are queued apart: an epoch takes reg_per_epoch of them (reg_quotas: each size
    bucket its share), the next ones of a shuffled pass through the bucket, so every picture of the folder comes up
    before any comes up again, and never twice in one epoch. They batch with the subject's pictures of their size,
    and the sampling weights (the curriculum, the loss watch's exclusions) never touch them."""

    def __init__(self, items: list[Item], batch_size: int, seed: int, reg_per_epoch: int = -1):
        # `items` is the subject's alone: what reads the queue's pictures (training maps, region sweeps, the
        # curriculum, the crops' weights) never meets a regularisation picture. Those are in reg_items
        self.reg_items = [it for it in items if getattr(it, "reg", False)]
        self.items = [it for it in items if not getattr(it, "reg", False)] if self.reg_items else items
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.buckets: dict[tuple[int, int], list[Item]] = {}
        self.reg_buckets: dict[tuple[int, int], list[Item]] = {}
        for it in items:
            if getattr(it, "reg", False):
                self.reg_buckets.setdefault((it.width, it.height, it.frames), []).append(it)
                continue
            self.buckets.setdefault((it.width, it.height, it.frames), []).extend([it] * max(1, int(it.repeats or 1)))
        log.info("Buckets: %s", {f"{k[0]}x{k[1]}" + (f"x{k[2]}f" if k[2] > 1 else ""): len(v)
                                 for k, v in self.buckets.items()})
        self.reg_quota = reg_quotas({k: len(v) for k, v in self.reg_buckets.items()},
                                    sum(len(v) for v in self.buckets.values()), int(reg_per_epoch))
        self._reg_decks: dict[tuple[int, int], list[Item]] = {}  # the rest of each bucket's current pass
        if self.reg_buckets:
            log.info("Regularisation buckets: %s", {f"{k[0]}x{k[1]}": f"{self.reg_quota[k]} of {len(v)} per epoch"
                                                    for k, v in self.reg_buckets.items()})
        self.weights: dict[str, float] | None = None

    def set_weights(self, weights: dict[str, float] | None) -> None:
        """Per-image sampling weights keyed by item.key; None restores uniform shuffling."""
        self.weights = weights

    def reg_images(self) -> int:
        """Regularisation pictures an epoch trains on."""
        return sum(self.reg_quota.values())

    def carry_reg_pass(self, old: "BucketBatcher | None") -> None:
        """Go on with the pass through the regularisation pictures where the queue this one replaces had got to
        (self-training rebuilds the queue every round): a bucket's pass is taken over when it holds the same
        pictures here. Another resolution stage has other buckets and starts its own; so does a resumed run."""
        for key, deck in (getattr(old, "_reg_decks", None) or {}).items():
            mine = self.reg_buckets.get(key)
            if mine is not None and {id(it) for it in mine} == {id(it) for it in old.reg_buckets.get(key, [])}:
                self._reg_decks[key] = list(deck)

    def epoch_images(self) -> int:
        """Pictures in one epoch: the subject's (repeats counted) and the epoch's share of regularisation pictures."""
        return sum(len(v) for v in self.buckets.values()) + self.reg_images()

    def _reg_draw(self, key) -> list[Item]:
        """A bucket's regularisation pictures for this epoch: the next ones of its shuffled pass. When the pass runs
        out mid-epoch a new one starts, with the pictures already drawn this epoch at its end (a quota is never more
        than the bucket holds, so none comes up twice in one epoch)."""
        want = self.reg_quota.get(key, 0)
        out: list[Item] = []
        if want <= 0:
            return out
        deck = self._reg_decks.setdefault(key, [])
        while len(out) < want:
            if not deck:
                fresh = self.reg_buckets[key][:]
                self.rng.shuffle(fresh)
                drawn = {id(it) for it in out}
                deck.extend([it for it in fresh if id(it) not in drawn] + [it for it in fresh if id(it) in drawn])
            out.append(deck.pop(0))
        return out

    def _resample(self, group: list[Item], ws: list[float]) -> list[Item]:
        """len(group) draws in proportion to the weights, systematically: each item comes up the floor or the ceiling
        of its expected count (independent draws with replacement skip some images every epoch and repeat others,
        even where every weight is equal). Zero weights: never; all zero: an empty bucket."""
        total = sum(ws)
        if total <= 0:
            return []
        n = len(group)
        step = total / n
        pos = self.rng.random() * step
        out, cum = [], 0.0
        for it, w in zip(group, ws):
            cum += w
            while pos < cum and len(out) < n:
                out.append(it)
                pos += step
        last = next(it for it, w in zip(reversed(group), reversed(ws)) if w > 0)
        out += [last] * (n - len(out))  # floating-point shortfall at the very end
        return out

    def epoch(self):
        batches = []
        # the subject's buckets in their own order, then the sizes only regularisation pictures have (none: the
        # shuffles below draw exactly as they did before there was a regularisation dataset)
        for key in list(self.buckets) + [k for k in self.reg_buckets if k not in self.buckets]:
            group = self.buckets.get(key, [])
            if self.weights and group:
                group = self._resample(group, [max(0.0, float(self.weights.get(it.key, 1.0))) for it in group])
            else:
                group = group[:]
            if self.reg_buckets:
                group += self._reg_draw(key)
            self.rng.shuffle(group)
            for i in range(0, len(group), self.batch_size):
                batches.append(group[i:i + self.batch_size])
        self.rng.shuffle(batches)
        return batches

    def __iter__(self):
        while True:
            for b in self.epoch():
                yield b


def load_batch(batch: list[Item], cfg: TrainConfig, rng: random.Random):
    """Returns latents [B,128,h,w], embeds [B,S,4096], mask [B,S], regions (RegionBatch | None). A batch of video clips
    gives latents [B, C, T, h, w] and no regions (the face and region features read pictures).

    Only this function knows whether the flipped latent was picked, so it flips the region map (and mirrors
    its landmarks) to match."""
    from .regions import flip_region_map, load_region_map, uniform_region_map

    pad_embeds = get_arch(cfg).pad_embeds

    latents, rows, maps, paths = [], [], [], []
    empty = dropout_text_path(cfg)
    want_regions = cfg.wants_regions() and not any(it.frames > 1 for it in batch)
    for it in batch:
        path = it.latent_path
        flipped = False
        if it.latent_flip_path and rng.random() < 0.5:
            path, flipped = it.latent_flip_path, True
        latents.append(load_file(str(path))["latent"])
        # a regularisation picture keeps its caption: a dropped one trains on the trigger word alone
        # (caption_dropout_keep_trigger), which would teach the trigger that picture
        drop = cfg.caption_dropout > 0 and not it.reg and rng.random() < cfg.caption_dropout
        tp = empty if drop else it.text_path
        alts = getattr(it, "alt_text_paths", None)
        if alts and not drop:  # caption_lines: one of the picture's lines, drawn from the run's generator
            tp = rng.choice([it.text_path, *alts])
        rows.append(load_file(str(tp))["embeds"])
        paths.append(tp)
        if want_regions:
            rm = load_region_map(it.region_path) if it.region_path is not None else None
            if rm is None:
                rm = uniform_region_map(it.width, it.height)
            if flipped:
                rm = flip_region_map(rm, it.width)
            maps.append(rm)
    embeds, mask = pad_embeds(rows)
    regions = None
    if want_regions:
        regions = RegionBatch(torch.stack([m.weight for m in maps]), torch.stack([m.prior for m in maps]), maps,
                              [str(p) for p in paths])
    return torch.stack(latents), embeds, mask, regions
