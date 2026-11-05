"""Region analysis for identity training: face boxes per source image, person masks per item, and
latent-cell loss weight maps.

Coordinates: face detection runs on the *source* image (full resolution). Training images are produced by
`dataset.load_image_fit`, i.e. an optional crop (`Item.src_box`) followed by PIL's `ImageOps.fit` centre crop +
resize to the bucket size. `fit_crop_box` / `remap_bbox` replicate that mapping exactly so a source-pixel box
lands on the right latent cells (one cell = 16 x 16 bucket pixels).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .atomic import save_safetensors

log = logging.getLogger("rulztrainer")
CELL = 16


# ---------------------------------------------------------------- geometry


def fit_crop_box(W: float, H: float, bw: int, bh: int) -> tuple[float, float, float, float]:
    """(left, top, crop_w, crop_h) in source pixels, exactly as ImageOps.fit(..., centering=(0.5, 0.5))."""
    live_ratio = W / H
    out_ratio = bw / bh
    if live_ratio == out_ratio:
        cw, ch = float(W), float(H)
    elif live_ratio >= out_ratio:
        cw, ch = out_ratio * H, float(H)
    else:
        cw, ch = float(W), W / out_ratio
    return (W - cw) * 0.5, (H - ch) * 0.5, cw, ch


def remap_bbox(bbox, W: float, H: float, bw: int, bh: int, src_box=None):
    """Map a source-pixel bbox (x0,y0,x1,y1) into bucket pixels. Returns None if it falls outside the crop."""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if src_box is not None:  # crop item: coordinates relative to the crop
        sx, sy, ex, ey = src_box
        x0, x1, y0, y1 = x0 - sx, x1 - sx, y0 - sy, y1 - sy
        W, H = ex - sx, ey - sy
    left, top, cw, ch = fit_crop_box(W, H, bw, bh)
    fx, fy = bw / cw, bh / ch
    rx0, ry0 = (x0 - left) * fx, (y0 - top) * fy
    rx1, ry1 = (x1 - left) * fx, (y1 - top) * fy
    rx0, ry0, rx1, ry1 = max(0.0, rx0), max(0.0, ry0), min(float(bw), rx1), min(float(bh), ry1)
    if rx1 - rx0 < 2 or ry1 - ry0 < 2:
        return None
    return rx0, ry0, rx1, ry1


def remap_points(pts, W: float, H: float, bw: int, bh: int, src_box=None):
    out = []
    for x, y in pts:
        x, y = float(x), float(y)
        WW, HH = W, H
        if src_box is not None:
            x, y = x - src_box[0], y - src_box[1]
            WW, HH = src_box[2] - src_box[0], src_box[3] - src_box[1]
        left, top, cw, ch = fit_crop_box(WW, HH, bw, bh)
        out.append([(x - left) * bw / cw, (y - top) * bh / ch])
    return out


def to_cells(bbox_px, bw: int, bh: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_px
    return (max(0, int(math.floor(x0 / CELL))), max(0, int(math.floor(y0 / CELL))),
            min(bw // CELL, int(math.ceil(x1 / CELL))), min(bh // CELL, int(math.ceil(y1 / CELL))))


def dilate_bbox(bbox, margin: float, bw: int, bh: int):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    return (max(0.0, x0 - margin * w), max(0.0, y0 - margin * h), min(float(bw), x1 + margin * w),
            min(float(bh), y1 + margin * h))


# ---------------------------------------------------------------- face analysis cache


def _file_key(path: Path) -> str:
    st = path.stat()
    return f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}"


def analyze_faces(items, face_scorer, cache_path: Path, progress=None, recache: bool = False,
                  embeddings: bool = False) -> dict[str, dict]:
    """Largest face per *source image* in source pixels: {str(resolved path): {bbox, kps, det, W, H}} (bbox None if
    no face), with embeddings=True also its ArcFace embedding ("emb", 512 floats; identity alignment). Cached in
    faces.json keyed by path|mtime|size; a face cached without an embedding is analysed again when one is wanted."""
    from PIL import Image, ImageOps

    cache: dict[str, dict] = {}
    if cache_path.exists() and not recache:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    sources: dict[str, Path] = {}
    for it in items:
        sources.setdefault(str(it.image.resolve()), it.image)
    todo = [(k, p) for k, p in sources.items() if _file_key(p) not in cache
            or (embeddings and cache[_file_key(p)].get("bbox") and not cache[_file_key(p)].get("emb"))]
    for i, (k, p) in enumerate(todo):
        rec: dict = {"bbox": None, "kps": None, "det": 0.0, "W": 0, "H": 0}
        try:
            with Image.open(p) as im:
                W, H = ImageOps.exif_transpose(im).size
            rec["W"], rec["H"] = W, H
            if face_scorer is not None:
                faces = face_scorer.faces(p)
                if faces:
                    f = faces[0]
                    rec["bbox"] = [float(v) for v in f.bbox]
                    kps = getattr(f, "kps", None)
                    rec["kps"] = [[float(x), float(y)] for x, y in kps] if kps is not None else None
                    rec["det"] = float(f.det_score)
                    emb = getattr(f, "normed_embedding", None)
                    if emb is not None:
                        rec["emb"] = [round(float(v), 6) for v in emb]
        except Exception as e:  # noqa: BLE001
            log.warning("face analysis failed for %s: %s", p, e)
        cache[_file_key(p)] = rec
        if progress:
            progress(i + 1, len(todo))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {k: cache[_file_key(p)] for k, p in sources.items()}


def analyze_bodies(items, embed, cache_path: Path, progress=None, recache: bool = False) -> dict[str, list | None]:
    """The DINOv2 body-window embedding of every *source image* ({str(resolved path): 768 floats, or None without a
    usable window}), body alignment's targets. embed(path) -> unit vector | None (lookfilter.body_embedder's). Cached
    in bodies.json keyed like faces.json."""
    cache: dict[str, dict] = {}
    if cache_path.exists() and not recache:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    sources: dict[str, Path] = {}
    for it in items:
        sources.setdefault(str(it.image.resolve()), it.image)
    todo = [(k, p) for k, p in sources.items() if _file_key(p) not in cache]
    for i, (k, p) in enumerate(todo):
        emb = None
        try:
            v = embed(p)
            emb = None if v is None else [round(float(x), 6) for x in v]
        except Exception as e:  # noqa: BLE001
            log.warning("body analysis failed for %s: %s", p, e)
        cache[_file_key(p)] = {"emb": emb}
        if progress:
            progress(i + 1, len(todo))
    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {k: cache[_file_key(p)].get("emb") for k, p in sources.items()}


def face_fraction(rec: dict | None, src_box=None) -> float:
    """Face area as a fraction of the (cropped) source image."""
    if not rec or not rec.get("bbox"):
        return 0.0
    x0, y0, x1, y1 = rec["bbox"]
    if src_box is not None:
        sx, sy, ex, ey = src_box
        ix0, iy0, ix1, iy1 = max(x0, sx), max(y0, sy), min(x1, ex), min(y1, ey)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return (ix1 - ix0) * (iy1 - iy0) / max(1.0, (ex - sx) * (ey - sy))
    return max(0.0, (x1 - x0) * (y1 - y0)) / max(1.0, rec["W"] * rec["H"])


# ---------------------------------------------------------------- person masks


def person_mask_from_landmarks(pts: np.ndarray, vis: np.ndarray, w: int, h: int, min_vis: float = 0.5,
                               dilate: float = 0.10) -> np.ndarray | None:
    """Dilated convex hull of the visible pose landmarks as a bool [h, w] mask (fallback when SAM 3 is off)."""
    from PIL import Image, ImageDraw

    pts_ok = [(float(x), float(y)) for (x, y), v in zip(pts, vis) if v >= min_vis]
    if len(pts_ok) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull

        arr = np.array(pts_ok)
        hull = [tuple(arr[i]) for i in ConvexHull(arr).vertices]
    except Exception:  # noqa: BLE001 - scipy missing or degenerate hull: use the bounding box
        xs, ys = [p[0] for p in pts_ok], [p[1] for p in pts_ok]
        hull = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    scale = 1.0 + dilate
    hull = [(cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in hull]
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).polygon(hull, fill=255)
    return np.asarray(img) > 0


# ---------------------------------------------------------------- weight maps

PARTS_MODEL_NAME = "selfie_multiclass_256x256.tflite"
PARTS_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/"
                   "latest/selfie_multiclass_256x256.tflite")
PART_CLASSES = 6  # the parser's: background, hair, body skin, face skin, clothes, others
PART_BACKGROUND, PART_BODY_SKIN = 0, 2
PART_RING = PART_CLASSES  # the parts tensor's last channel


@dataclass
class RegionMap:
    weight: torch.Tensor  # [1, h, w] float32, mean 1
    prior: torch.Tensor  # [1, h, w] bool: background cells (for prior preservation)
    face_cells: tuple[int, int, int, int] | None
    kps: list | None  # bucket-pixel landmarks
    has_person: bool
    parts: torch.Tensor | None = None  # [PARTS, h, w] uint8: the parser's class fractions, then the silhouette ring
    parts_print: str = ""  # what built `parts`: a mismatch rebuilds the map


def build_weight_map(bw: int, bh: int, face_bbox_px, person_mask: np.ndarray | None, face_w: float, body_w: float,
                     bg_w: float, face_margin: float = 0.25, feather: int = 1) -> RegionMap:
    h, w = bh // CELL, bw // CELL
    wmap = torch.full((h, w), float(bg_w))
    has_person = False
    if person_mask is not None and person_mask.shape == (bh, bw):
        m = torch.from_numpy(person_mask.astype(np.float32)).view(h, CELL, w, CELL).mean(dim=(1, 3)) >= 0.5
        if m.any():
            wmap[m] = float(body_w)
            has_person = True
    face_cells = None
    if face_bbox_px is not None:
        cx0, cy0, cx1, cy1 = to_cells(dilate_bbox(face_bbox_px, face_margin, bw, bh), bw, bh)
        if cx1 > cx0 and cy1 > cy0:
            wmap[cy0:cy1, cx0:cx1] = float(face_w)
            face_cells = (cx0, cy0, cx1, cy1)
    prior = wmap == float(bg_w)
    for _ in range(max(0, int(feather))):
        wmap = torch.nn.functional.avg_pool2d(wmap[None, None], 3, stride=1, padding=1, count_include_pad=False)[0, 0]
    if wmap.sum() > 0:
        wmap = wmap * (wmap.numel() / wmap.sum())
    else:
        wmap = torch.ones(h, w)
    return RegionMap(wmap[None].float().contiguous(), prior[None].contiguous(), face_cells, None, has_person)


def parts_print(cfg, bw: int, bh: int) -> str:
    """What a region map's parts tensor depends on, for its metadata: a different value rebuilds the map."""
    return f"{PARTS_MODEL_NAME}|ring={cfg.hull_ring_cells}|margin={cfg.face_margin}|{bw}x{bh}"


def build_parts(classes: np.ndarray, bw: int, bh: int, face_cells, ring_cells: int) -> torch.Tensor:
    """The parts tensor from a parser's class per pixel [bh, bw]: per 16 px cell the fraction of each of the six
    classes (uint8, 255 = all of the cell), then the silhouette ring: the person's cells (under half background)
    dilated minus eroded by `ring_cells`, outside the face box. The ring carries widths and lengths; the face box
    takes no gain, so a jaw line never reaches it."""
    h, w = bh // CELL, bw // CELL
    labels = torch.from_numpy(classes.astype(np.int64))[:h * CELL, :w * CELL]
    hot = torch.nn.functional.one_hot(labels, PART_CLASSES).float()
    fractions = hot.view(h, CELL, w, CELL, PART_CLASSES).mean(dim=(1, 3)).permute(2, 0, 1)
    person = (fractions[PART_BACKGROUND] < 0.5)[None, None].float()
    size = 2 * int(ring_cells) + 1
    grown = torch.nn.functional.max_pool2d(person, size, stride=1, padding=int(ring_cells))
    shrunk = -torch.nn.functional.max_pool2d(-person, size, stride=1, padding=int(ring_cells))
    ring = (grown[0, 0] > 0) & (shrunk[0, 0] <= 0)
    if face_cells is not None:
        cx0, cy0, cx1, cy1 = face_cells
        ring[cy0:cy1, cx0:cx1] = False
    out = torch.cat([fractions, ring[None].float()])
    return (out * 255.0).round().to(torch.uint8).contiguous()


def token_gains(parts: torch.Tensor, face_cells, body: float, background: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(gains, face) per cell [h, w] from a parts tensor: gains float32, face bool. Body skin takes `body`, background
    `background`, hair, clothing, face skin and the rest exactly 1; a ring cell takes at least `body`; the face box is
    exactly 1 whatever its cells hold."""
    f = parts[:PART_CLASSES].to(torch.float32) / 255.0
    gain = torch.ones(PART_CLASSES, dtype=torch.float32)
    gain[PART_BACKGROUND], gain[PART_BODY_SKIN] = float(background), float(body)
    base = (f * gain[:, None, None]).sum(0) / f.sum(0).clamp_min(1e-6)
    out = torch.where(parts[PART_RING] > 0, base.clamp_min(float(body)), base)
    face = torch.zeros_like(out, dtype=torch.bool)
    if face_cells is not None:
        cx0, cy0, cx1, cy1 = face_cells
        out[cy0:cy1, cx0:cx1] = 1.0
        face[cy0:cy1, cx0:cx1] = True
    return out, face


class PartsParser:
    """MediaPipe's multiclass selfie segmenter (Apache 2.0): each pixel's class, for the parts tensor. Hull's training
    masks only; a lock lets the region-map workers share it."""

    def __init__(self, models_dir: str | Path = "models", progress=None):
        import threading

        from .bodysim import available, mask_width_multiple
        from .downloads import download_url

        ok, msg = available()
        if not ok:
            raise RuntimeError(msg)
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model = Path(models_dir) / PARTS_MODEL_NAME
        if not model.exists():
            download_url(PARTS_MODEL_URL, model, "MediaPipe multiclass segmenter", progress)
        opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=str(model)),
                                            running_mode=vision.RunningMode.IMAGE, output_category_mask=True,
                                            output_confidence_masks=False)
        self._mp = mp
        self.segmenter = vision.ImageSegmenter.create_from_options(opts)
        self.pad_to = mask_width_multiple()
        self._lock = threading.Lock()

    def __call__(self, img) -> np.ndarray:
        arr = np.asarray(img.convert("RGB"))
        h, w = arr.shape[:2]
        pad = (-w) % self.pad_to  # mediapipe 1.0.x only: bodysim.mask_width_multiple
        if pad:
            arr = np.pad(arr, ((0, 0), (0, pad), (0, 0)), mode="edge")
        arr = np.ascontiguousarray(arr)
        with self._lock:
            res = self.segmenter.segment(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=arr))
            classes = np.array(res.category_mask.numpy_view())
        classes = classes.reshape(h, w + pad)[:, :w]
        return np.minimum(classes, PART_CLASSES - 1)


def save_region_map(path: Path, rm: RegionMap) -> None:
    meta = {"face_cells": ",".join(map(str, rm.face_cells)) if rm.face_cells else "",
            "kps": json.dumps(rm.kps) if rm.kps else "", "has_person": "1" if rm.has_person else "0"}
    tensors = {"weight": rm.weight, "prior": rm.prior.to(torch.uint8)}
    if rm.parts is not None:
        tensors["parts"] = rm.parts
        meta["parts_print"] = rm.parts_print
    path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors(tensors, path, metadata=meta)


def load_region_map(path: Path) -> RegionMap | None:
    if not path.exists():
        return None
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
        weight, prior = f.get_tensor("weight"), f.get_tensor("prior").bool()
        parts = f.get_tensor("parts") if "parts" in f.keys() else None
    fc = tuple(int(v) for v in meta["face_cells"].split(",")) if meta.get("face_cells") else None
    kps = json.loads(meta["kps"]) if meta.get("kps") else None
    return RegionMap(weight, prior, fc, kps, meta.get("has_person") == "1", parts, meta.get("parts_print", ""))


def _parts_stale(path: Path, want: str) -> bool:
    """A cached map without the parts tensor this run wants, or with one built another way."""
    from safetensors import safe_open

    try:
        with safe_open(str(path), "pt") as f:
            return (f.metadata() or {}).get("parts_print") != want
    except Exception:  # noqa: BLE001
        return True


def flip_region_map(rm: RegionMap, bw: int) -> RegionMap:
    w_cells = rm.weight.shape[-1]
    fc = (w_cells - rm.face_cells[2], rm.face_cells[1], w_cells - rm.face_cells[0], rm.face_cells[3]) if rm.face_cells else None
    kps = None
    if rm.kps:
        kps = [[bw - x, y] for x, y in rm.kps]
        if len(kps) == 5:  # mirrored face: the left eye / mouth corner is now the right one (keeps 5-point order)
            kps = [kps[1], kps[0], kps[2], kps[4], kps[3]]
    return RegionMap(rm.weight.flip(-1), rm.prior.flip(-1), fc, kps, rm.has_person)


def uniform_region_map(bw: int, bh: int) -> RegionMap:
    h, w = bh // CELL, bw // CELL
    return RegionMap(torch.ones(1, h, w), torch.zeros(1, h, w, dtype=torch.bool), None, None, False)


def cache_region_maps(items, cfg, analysis: dict[str, dict], person_mask_fn=None, progress=None,
                      person_mask_fns: list | None = None, parts_fn=None) -> dict[str, int]:
    """Write <latent stem>_regions.safetensors for every item (skips existing unless cfg.recache). Returns counts.

    person_mask_fns: optional list of (fn, device), one per GPU, splitting the images across GPUs (each fn is
    only ever called from its own thread). parts_fn: the parser (PartsParser), giving a map its parts tensor; a
    cached map without the parts this run wants is rebuilt."""
    from .multigpu import run_queue

    counts = {"face": 0, "person": 0, "none": 0}
    todo = [it for it in items if it.region_path is not None and (
        cfg.recache or not it.region_path.exists()
        or (parts_fn is not None and _parts_stale(it.region_path, parts_print(cfg, it.width, it.height))))]
    workers = person_mask_fns or [(person_mask_fn, "cpu")]
    if len(workers) > 1:
        log.info("Region maps: %d images split across %s", len(todo), ", ".join(str(d) for _, d in workers))
    done_n = [0]

    def done(i, it, kind):
        counts[kind] += 1
        done_n[0] += 1
        if progress:
            progress(done_n[0], len(todo))

    run_queue(workers, todo, lambda w, it: _region_map_for(it, cfg, analysis, w[0], parts_fn), on_done=done)
    return counts


def _region_map_for(it, cfg, analysis: dict[str, dict], person_mask_fn, parts_fn=None) -> str:
    """Build and save one item's region map; returns its kind (face | person | none)."""
    from .dataset import load_image_fit

    rec = analysis.get(str(it.image.resolve()))
    bbox_px = None
    kps_px = None
    if rec and rec.get("bbox"):
        bbox_px = remap_bbox(rec["bbox"], rec["W"], rec["H"], it.width, it.height, it.src_box)
        if rec.get("kps"):
            kps_px = remap_points(rec["kps"], rec["W"], rec["H"], it.width, it.height, it.src_box)
    mask = None
    if person_mask_fn is not None:
        try:
            mask = person_mask_fn(load_image_fit(it.image, it.width, it.height, it.src_box))
        except Exception as e:  # noqa: BLE001
            log.warning("person mask failed for %s: %s", it.key, e)
    rm = build_weight_map(it.width, it.height, bbox_px, mask, cfg.face_loss_weight, cfg.body_loss_weight,
                          cfg.background_loss_weight, cfg.face_margin, cfg.region_feather_cells)
    rm.kps = kps_px
    if parts_fn is not None:
        rm.parts = build_parts(parts_fn(load_image_fit(it.image, it.width, it.height, it.src_box)), it.width, it.height,
                               rm.face_cells, cfg.hull_ring_cells)
        rm.parts_print = parts_print(cfg, it.width, it.height)
    save_region_map(it.region_path, rm)
    return "face" if rm.face_cells else ("person" if rm.has_person else "none")


def render_overlay(img, rm: RegionMap, bw: int, bh: int):
    """Colour overlay of a weight map on a PIL image (for the UI preview)."""
    from PIL import Image

    w = rm.weight[0]
    norm = (w - w.min()) / max(1e-6, float(w.max() - w.min()))
    heat = (norm.numpy() * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat, "L").resize((bw, bh), Image.Resampling.NEAREST)
    red = Image.merge("RGB", (heat_img, Image.new("L", (bw, bh), 0), Image.new("L", (bw, bh), 0)))
    return Image.blend(img.convert("RGB").resize((bw, bh)), red, 0.45)
