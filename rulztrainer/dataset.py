"""Image/caption dataset with aspect-ratio buckets and on-disk latent + text-embedding cache."""
from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file

from .archs import get_arch
from .config import TrainConfig

log = logging.getLogger("rulztrainer")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SKIP_DIRS = {".rulz_cache", "excluded_by_look"}  # folders inside a dataset that never train (cache, look filter)


def skipped(path: Path, root: Path | None = None) -> bool:
    """True for a file inside a folder that never trains (SKIP_DIRS), below `root` when given."""
    parts = path.parts
    if root is not None:
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            pass
    return any(p in SKIP_DIRS for p in parts)


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
    """Pick a (W,H) multiple of 16 with area ~= side^2 preserving aspect ratio (side defaults to cfg.resolution).

    min/max_bucket_side are applied as a uniform rescale first (so the aspect ratio survives the limit) and only
    as a hard per-side clamp when an extreme aspect ratio cannot satisfy both limits."""
    side = int(side or cfg.resolution)
    if not cfg.enable_buckets:
        return side // 16 * 16, side // 16 * 16
    aspect = w / h
    bw = math.sqrt(side * side * aspect)
    bh = bw / aspect
    lo, hi = cfg.min_bucket_side, cfg.max_bucket_side
    scale = 1.0
    if max(bw, bh) > hi:
        scale = hi / max(bw, bh)
    if min(bw, bh) * scale < lo:
        scale = lo / min(bw, bh)
    bw, bh = bw * scale, bh * scale
    bw = int(round(bw / 16) * 16)
    bh = int(round(bh / 16) * 16)
    bw = max(lo, min(hi, bw))
    bh = max(lo, min(hi, bh))
    return bw, bh


def items_for_side(items: list["Item"], side: int) -> list["Item"]:
    return [it for it in items if it.side == side]


def describe_sides(cfg: TrainConfig) -> str:
    from .config import side_to_mp

    return ", ".join(f"{side_to_mp(s):g} MP" for s in cfg.training_sides())


def load_image_fit(path: Path, w: int, h: int, src_box=None) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if src_box is not None:
        img = img.crop(tuple(int(round(v)) for v in src_box))
    return ImageOps.fit(img, (w, h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def read_caption(image: Path, cfg: TrainConfig, trigger: str | None = None) -> str:
    """The image's caption: the architecture picks the file (FIBO 2.0 prefers a structured <name>.json caption), cleans
    a JSON caption the way the model was trained, and puts the trigger word (its dataset's, else the run's) where it
    belongs."""
    arch = get_arch(cfg)
    cap = arch.caption_file(image, cfg)
    text = cap.read_text(encoding="utf-8").strip() if cap.is_file() else ""
    if cap.suffix.lower() == ".json" and text:
        text = arch.clean_prompt(text)
    return arch.add_trigger(text, cfg.trigger_word if trigger is None else trigger)


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def scan_dataset(cfg: TrainConfig) -> list[Item]:
    """Items of every dataset of the run (cfg.dataset_specs()): one per image and training resolution of its dataset,
    carrying the dataset's repeats; latents and region maps in the dataset's own cache (cfg.dataset_cache_dir)."""
    specs = cfg.dataset_specs()
    for spec in specs:  # before any cache folder is made: making one would create a missing dataset folder
        if not Path(spec.path).is_dir():
            raise FileNotFoundError(f"Dataset dir not found: {spec.path}")
    (cfg.resolved_cache_dir() / "text").mkdir(parents=True, exist_ok=True)
    tag = get_arch(cfg).latent_tag  # latents of another VAE get their own files
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
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTS or skipped(p, root):
                continue
            try:
                with Image.open(p) as im:
                    w, h = ImageOps.exif_transpose(im).size
            except Exception as e:  # noqa: BLE001
                log.warning("Skipping unreadable image %s: %s", p, e)
                continue
            n_images += 1
            caption = read_caption(p, cfg, spec.trigger or cfg.trigger_word)
            seen: set[tuple[int, int]] = set()
            for side in sides:
                bw, bh = bucket_size(w, h, cfg, side)
                if (bw, bh) in seen:  # two resolutions clamped to the same bucket: keep one copy
                    continue
                seen.add((bw, bh))
                stem = _hash(str(p.resolve())) + f"_{bw}x{bh}"
                items.append(Item(
                    image=p, caption=caption, width=bw, height=bh,
                    latent_path=cache / "latents" / f"{stem}{tag}.safetensors",
                    latent_flip_path=((cache / "latents" / f"{stem}_flip{tag}.safetensors") if cfg.flip_augment
                                      else None),
                    text_path=text_cache_path(cfg, caption),
                    region_path=cache / "regions" / f"{stem}_regions.safetensors",
                    side=side, repeats=spec.repeats, dataset=str(root),
                ))
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
            stem = _hash(f"{it.image.resolve()}|crop{box[0]},{box[1]},{box[2]},{box[3]}") + f"_{bw}x{bh}"
            tag = get_arch(cfg).latent_tag
            cache = it.latent_path.parent.parent  # the source image's dataset cache
            out.append(Item(
                image=it.image, caption=caption, width=bw, height=bh,
                latent_path=cache / "latents" / f"{stem}{tag}.safetensors",
                latent_flip_path=(cache / "latents" / f"{stem}_flip{tag}.safetensors") if cfg.flip_augment else None,
                text_path=text_cache_path(cfg, caption),
                src_box=box, category=category,
                region_path=cache / "regions" / f"{stem}_regions.safetensors",
                side=it.side, repeats=it.repeats, dataset=it.dataset,
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

    encode_images = get_arch(cfg).encode_images
    todo = latents_todo(items, cfg)
    workers = [(v, v.device) for v in (vaes or [vae])]
    log.info("Caching latents: %d of %d images need encoding%s", len(todo), len(items),
             f" (split across {', '.join(str(d) for _, d in workers)})" if len(workers) > 1 else "")

    @torch.no_grad()
    def encode(worker, it):
        v = worker[0]
        img = load_image_fit(it.image, it.width, it.height, it.src_box)
        t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        lat = encode_images(v, t.unsqueeze(0))[0].cpu()
        save_file({"latent": lat.contiguous()}, str(it.latent_path))
        if it.latent_flip_path:
            lat_f = encode_images(v, t.flip(-1).unsqueeze(0))[0].cpu()
            save_file({"latent": lat_f.contiguous()}, str(it.latent_flip_path))

    count = [0]

    def done(i, it, _):
        count[0] += 1
        if progress:
            progress(count[0], len(todo))

    run_queue(workers, todo, encode, on_done=done)


def text_jobs(items: list[Item], cfg: TrainConfig, extra_prompts: list[str] | None = None) -> tuple[dict, int]:
    """(cache path -> text for every prompt still missing, total prompt count)."""
    jobs: dict[Path, str] = {}
    for it in items:
        jobs.setdefault(it.text_path, it.caption)
        if it.dop_text_path is not None and it.dop_caption is not None:  # LoRA Isolation's trigger-free captions
            jobs.setdefault(it.dop_text_path, it.dop_caption)
    jobs.setdefault(dropout_text_path(cfg), dropout_prompt(cfg))
    for p in extra_prompts or []:
        jobs.setdefault(text_cache_path(cfg, p), p)
    return {k: v for k, v in jobs.items() if cfg.recache or not k.exists()}, len(jobs)


def cache_text(items: list[Item], cfg: TrainConfig, text_encoder, tokenizer,
               extra_prompts: list[str] | None = None, progress=None, encoders: list | None = None,
               max_prompts: int | None = None, encode_fn=None) -> None:
    """Encode and save the missing prompt embeddings. `encoders` is a list of (text_encoder, tokenizer, device),
    one per GPU, to split the prompts across GPUs (each needs its own tokenizer: fast tokenizers are not
    thread-safe). `max_prompts` encodes only the first N missing prompts (a timing probe). `encode_fn` is the
    architecture's encode_prompt(text_encoder, tokenizer, text, max_length)."""
    from .multigpu import run_queue

    encode_prompt = encode_fn or get_arch(cfg).encode_prompt

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
    def encode(worker, job):
        path, text = job
        emb = encode_prompt(worker[0], worker[1], text, cfg.max_token_length)
        save_file({"embeds": emb.contiguous()}, str(path), metadata={"prompt": text[:2000]})

    count = [0]

    def done(i, job, _):
        count[0] += 1
        if progress:
            progress(count[0], len(todo))

    run_queue([(w, w[2]) for w in workers], list(todo.items()), lambda w, job: encode(w[0], job), on_done=done)


def text_cache_path(cfg: TrainConfig, prompt: str) -> Path:
    """Cached text features of a prompt. Each architecture has its own text encoder, so each has its own suffix.
    MOSS-Image's is "_q35": its single-file mode once read prompts with Krea 2's tokenizer when a Krea 2 folder was in
    models/, and the features cached then (no suffix) must not be reused."""
    arch = cfg.arch_name()
    tag = "_q35" if arch == "moss_image" else f"_{arch}"
    return cfg.resolved_cache_dir() / "text" / f"{_hash(prompt)}_{cfg.max_token_length}{tag}.safetensors"


# ---------------------------------------------------------------- batching


class BucketBatcher:
    """Yields batches of Items with identical bucket size, shuffled each epoch. An item with repeats N comes up N
    times per epoch (its dataset's repeats)."""

    def __init__(self, items: list[Item], batch_size: int, seed: int):
        self.items = items
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.buckets: dict[tuple[int, int], list[Item]] = {}
        for it in items:
            self.buckets.setdefault((it.width, it.height), []).extend([it] * max(1, int(it.repeats or 1)))
        log.info("Buckets: %s", {f"{k[0]}x{k[1]}": len(v) for k, v in self.buckets.items()})
        self.weights: dict[str, float] | None = None

    def set_weights(self, weights: dict[str, float] | None) -> None:
        """Per-image sampling weights keyed by item.key; None restores uniform shuffling."""
        self.weights = weights

    def epoch(self):
        batches = []
        for group in self.buckets.values():
            if self.weights:
                ws = [self.weights.get(it.key, 1.0) for it in group]
                group = self.rng.choices(group, weights=ws, k=len(group))
            else:
                group = group[:]
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
    """Returns latents [B,128,h,w], embeds [B,S,4096], mask [B,S], regions (RegionBatch | None).

    Only this function knows whether the flipped latent was picked, so it flips the region map (and mirrors
    its landmarks) to match."""
    from .regions import flip_region_map, load_region_map, uniform_region_map

    pad_embeds = get_arch(cfg).pad_embeds

    latents, rows, maps, paths = [], [], [], []
    empty = dropout_text_path(cfg)
    want_regions = cfg.wants_regions()
    for it in batch:
        path = it.latent_path
        flipped = False
        if it.latent_flip_path and rng.random() < 0.5:
            path, flipped = it.latent_flip_path, True
        latents.append(load_file(str(path))["latent"])
        tp = empty if (cfg.caption_dropout > 0 and rng.random() < cfg.caption_dropout) else it.text_path
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
