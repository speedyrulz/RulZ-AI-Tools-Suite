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
from safetensors.torch import load_file, save_file

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


def analyze_faces(items, face_scorer, cache_path: Path, progress=None, recache: bool = False) -> dict[str, dict]:
    """Largest face per *source image* in source pixels: {str(resolved path): {bbox, kps, det, W, H}} (bbox None if
    no face). Cached in faces.json keyed by path|mtime|size."""
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
    todo = [(k, p) for k, p in sources.items() if _file_key(p) not in cache]
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
        except Exception as e:  # noqa: BLE001
            log.warning("face analysis failed for %s: %s", p, e)
        cache[_file_key(p)] = rec
        if progress:
            progress(i + 1, len(todo))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {k: cache[_file_key(p)] for k, p in sources.items()}


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


@dataclass
class RegionMap:
    weight: torch.Tensor  # [1, h, w] float32, mean 1
    prior: torch.Tensor  # [1, h, w] bool: background cells (for prior preservation)
    face_cells: tuple[int, int, int, int] | None
    kps: list | None  # bucket-pixel landmarks
    has_person: bool


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


def save_region_map(path: Path, rm: RegionMap) -> None:
    meta = {"face_cells": ",".join(map(str, rm.face_cells)) if rm.face_cells else "",
            "kps": json.dumps(rm.kps) if rm.kps else "", "has_person": "1" if rm.has_person else "0"}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"weight": rm.weight, "prior": rm.prior.to(torch.uint8)}, str(path), metadata=meta)


def load_region_map(path: Path) -> RegionMap | None:
    if not path.exists():
        return None
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
        weight, prior = f.get_tensor("weight"), f.get_tensor("prior").bool()
    fc = tuple(int(v) for v in meta["face_cells"].split(",")) if meta.get("face_cells") else None
    kps = json.loads(meta["kps"]) if meta.get("kps") else None
    return RegionMap(weight, prior, fc, kps, meta.get("has_person") == "1")


def flip_region_map(rm: RegionMap, bw: int) -> RegionMap:
    w_cells = rm.weight.shape[-1]
    fc = (w_cells - rm.face_cells[2], rm.face_cells[1], w_cells - rm.face_cells[0], rm.face_cells[3]) if rm.face_cells else None
    kps = [[bw - x, y] for x, y in rm.kps] if rm.kps else None
    return RegionMap(rm.weight.flip(-1), rm.prior.flip(-1), fc, kps, rm.has_person)


def uniform_region_map(bw: int, bh: int) -> RegionMap:
    h, w = bh // CELL, bw // CELL
    return RegionMap(torch.ones(1, h, w), torch.zeros(1, h, w, dtype=torch.bool), None, None, False)


def cache_region_maps(items, cfg, analysis: dict[str, dict], person_mask_fn=None, progress=None,
                      person_mask_fns: list | None = None) -> dict[str, int]:
    """Write <latent stem>_regions.safetensors for every item (skips existing unless cfg.recache). Returns counts.

    person_mask_fns: optional list of (fn, device), one per GPU, splitting the images across GPUs (each fn is
    only ever called from its own thread)."""
    from .multigpu import run_queue

    counts = {"face": 0, "person": 0, "none": 0}
    todo = [it for it in items if it.region_path is not None and (cfg.recache or not it.region_path.exists())]
    workers = person_mask_fns or [(person_mask_fn, "cpu")]
    if len(workers) > 1:
        log.info("Region maps: %d images split across %s", len(todo), ", ".join(str(d) for _, d in workers))
    done_n = [0]

    def done(i, it, kind):
        counts[kind] += 1
        done_n[0] += 1
        if progress:
            progress(done_n[0], len(todo))

    run_queue(workers, todo, lambda w, it: _region_map_for(it, cfg, analysis, w[0]), on_done=done)
    return counts


def _region_map_for(it, cfg, analysis: dict[str, dict], person_mask_fn) -> str:
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
