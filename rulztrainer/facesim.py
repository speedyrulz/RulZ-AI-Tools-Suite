"""Face similarity scoring for character LoRAs.

Reference faces come from one or more dataset images; each generated sample's largest detected face is
embedded with InsightFace (ArcFace "buffalo_l") and compared by cosine similarity against every reference
embedding. The reported score is the best match. A typical reference's best match among the others (the median,
faces cut off by the frame left out) is reported too: the realistic ceiling, the 100 of the normalized scale.

Cosine similarity for ArcFace: same identity is typically 0.5 to 0.8, different people 0.0 to 0.3.
"percent" is the cosine similarity clipped to [0, 1] times 100 (so 50%+ generally means "same person").

A second, held-out recognizer (set_gold: FaceNet, idloss.facenet_embedder) can score the same detected face, aligned
by the same landmarks, against its own embeddings of the references: score(gold=True) fills gold_cosine /
gold_percent, and gold_self_similarity is its reference ceiling. Nothing trains on it, so it can judge a run that
trains against ArcFace (likeness.py).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("rulztrainer")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def image_size(path) -> tuple[int, int]:
    """(width, height) as displayed: an EXIF rotation by 90 or 270 degrees swaps the stored size."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        try:
            orient = im.getexif().get(0x0112, 1)
        except Exception:  # noqa: BLE001
            orient = 1
    return (h, w) if orient in (5, 6, 7, 8) else (w, h)


def cut_off(bbox, size, margin: float = 0.05) -> bool:
    """True when a face box reaches past the photo's edge by more than `margin` of its own size: a face the frame
    cuts off (a close-up that overfills it), whose embedding is unreliable."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    w, h = size
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return x0 < -mx or y0 < -my or x1 > w + mx or y1 > h + my


def expand_refs(paths) -> list[Path]:
    """Reference entries may be image files or folders (expanded to the images inside). Missing paths are
    skipped with a warning, never raised, so a typo cannot crash a run."""
    out: list[Path] = []
    for raw in paths:
        p = Path(str(raw).strip().strip('"'))
        if p.is_dir():
            imgs = sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS and q.is_file())
            if not imgs:
                log.warning("Reference folder %s contains no images", p)
            out.extend(imgs)
        elif p.is_file():
            out.append(p)
        else:
            log.warning("Reference path not found, skipped: %s", p)
    seen, uniq = set(), []
    for p in out:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


@dataclass
class FaceScore:
    step: int
    image: str
    prompt_index: int
    faces: int  # faces detected in the sample
    cosine: float | None  # best cosine similarity to any reference face (None = no face)
    percent: float | None  # face similarity percent (None = no face)
    best_ref: str | None
    body_percent: float | None = None  # body similarity (None = no usable person / pose)
    body_ratios: int = 0  # references or proportion ratios compared
    combined_percent: float | None = None  # weighted face/body score (raw)
    # on the likeness scale (0 = a stranger, 100 = the references), when a scale was given
    face_norm: float | None = None
    body_norm: float | None = None
    combined_norm: float | None = None
    # the held-out recognizer's best cosine and percent (set_gold; None = off or no face)
    gold_cosine: float | None = None
    gold_percent: float | None = None


def combine(face, body, face_weight: float, body_weight: float):
    parts = [(v, w) for v, w in ((face, face_weight), (body, body_weight)) if v is not None and w > 0]
    if not parts:
        return None
    return sum(v * w for v, w in parts) / sum(w for _, w in parts)


def available() -> tuple[bool, str]:
    try:
        import insightface  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as e:
        return False, f"face scoring needs 'insightface' and 'onnxruntime-gpu' (pip install): {e}"
    return True, ""


def cuda_index(provider: str, device=None) -> int | None:
    """The GPU InsightFace runs on: None for the CPU provider; for "cuda" the index of `device` (e.g. the trainer's
    cuda:1) or of a "cuda:N" provider, else 0."""
    provider = str(provider or "cpu").strip().lower()
    if not provider.startswith("cuda"):
        return None
    for d in (provider, device):
        if d is not None and str(d).startswith("cuda:"):
            try:
                return int(str(d).split(":", 1)[1])
            except ValueError:
                continue
    return 0


class FaceScorer:
    def __init__(self, provider: str = "cpu", det_size: int = 640, model_name: str = "buffalo_l", progress=None,
                 device=None):
        """provider "cpu" or "cuda" ("cuda:N" for a GPU other than the first); device: the GPU a plain "cuda" runs
        on (onnxruntime's device_id; prepare's ctx_id alone only told CPU from GPU, so it was always GPU 0)."""
        ok, msg = available()
        if not ok:
            raise RuntimeError(msg)
        from insightface.app import FaceAnalysis

        from .downloads import ensure_insightface_pack

        ensure_insightface_pack(model_name, progress=progress)
        if progress:
            progress(f"Loading face model {model_name}...", None)
        index = cuda_index(provider, device)
        providers = ([("CUDAExecutionProvider", {"device_id": index}), "CPUExecutionProvider"] if index is not None
                     else ["CPUExecutionProvider"])
        self.app = FaceAnalysis(name=model_name, providers=providers,
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=index if index is not None else -1, det_size=(det_size, det_size))
        self.refs: list[tuple[str, np.ndarray]] = []
        self.ref_self_similarity: float | None = None
        self.ref_kps: list = []  # each reference face's landmarks (source pixels) and whether the frame cuts it off
        self.ref_cut: list[bool] = []
        self.gold = None  # the held-out recognizer (set_gold)
        self.gold_refs: np.ndarray | None = None
        self.gold_self_similarity: float | None = None

    # ------------------------------------------------------------ embeddings
    def _load_bgr(self, path: str | Path) -> tuple[np.ndarray, float]:
        """BGR array for the detector (thumbnailed to 2048 px) and the scale back to source pixels."""
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            scale = 1.0
            if max(im.size) > 2048:  # detector input is resized anyway; keep memory sane
                full = max(im.size)
                im.thumbnail((2048, 2048))
                scale = full / max(im.size)
            return np.asarray(im)[:, :, ::-1].copy(), scale

    FALLBACK_PADS = (2, 4)  # the image shrunk this many times on a canvas of its own size, when a face fills the frame

    def _detect(self, arr: np.ndarray):
        """Faces in a BGR array, with the boxes in its pixels. The detector rescales every input to det_size and its
        anchors stop at a face size relative to that frame, so a face that fills the frame (a tight headshot probe)
        is missed whatever the image size: when nothing is found, the image is shrunk to a half and then a quarter
        at the centre of a canvas of its own size (the same face at half / a quarter of the frame, found with
        confidence 0.9) and the boxes are mapped back. The canvas used to be the image at full size on 2x / 4x
        canvases (up to 16x the pixels); the detector shrinks it to det_size either way. The embedding is taken
        again from the source pixels at the mapped-back landmarks, as on the full-size canvas before."""
        found = self.app.get(arr)
        if found:
            return found
        import cv2

        h, w = arr.shape[:2]
        for k in self.FALLBACK_PADS:
            sw, sh = max(1, round(w / k)), max(1, round(h / k))
            canvas = np.full((h, w, 3), 114, dtype=arr.dtype)  # the detector's own neutral gray
            oy, ox = (h - sh) // 2, (w - sw) // 2
            canvas[oy:oy + sh, ox:ox + sw] = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
            found = self.app.get(canvas)
            if found:
                shift = np.array([ox, oy, ox, oy], dtype=np.float32)
                scale = np.array([w / sw, h / sh, w / sw, h / sh], dtype=np.float32)
                rec = getattr(self.app, "models", {}).get("recognition")
                for f in found:
                    f.bbox = (f.bbox - shift) * scale
                    if getattr(f, "kps", None) is not None:
                        f.kps = (f.kps - shift[:2]) * scale[:2]
                        if rec is not None:
                            rec.get(arr, f)  # the embedding of the full-resolution face
                return found
        return []

    def faces(self, path: str | Path):
        """All faces in an image, largest first, with .bbox / .kps in *source* pixels, .normed_embedding, .det_score."""
        arr, scale = self._load_bgr(path)
        found = self._detect(arr)
        if scale != 1.0:
            for f in found:
                f.bbox = f.bbox * scale
                if getattr(f, "kps", None) is not None:
                    f.kps = f.kps * scale
        return sorted(found, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

    def set_references(self, paths: list[str | Path]) -> list[dict]:
        """Embed the largest face of each reference image. Returns per-image info for the UI."""
        from .likeness import reference_ceiling

        self.refs, self.ref_kps = [], []
        info = []
        cut = []
        paths = expand_refs(paths)
        for p in paths:
            fs = self.faces(p)
            if not fs:
                info.append({"image": str(p), "faces": 0})
                log.warning("Face reference %s: no face detected", p)
                continue
            f = fs[0]
            off = cut_off(f.bbox, image_size(p))
            self.refs.append((str(p), f.normed_embedding.astype(np.float32)))
            self.ref_kps.append(getattr(f, "kps", None))
            cut.append(off)
            info.append({"image": str(p), "faces": len(fs), "bbox": [float(x) for x in f.bbox],
                         "det_score": float(f.det_score), "cut_off": off})
        # each reference's best match among the others is the statistic a probe is scored with; its median over the
        # references whose face is whole is the ceiling of the normalized scale (reference_ceiling). The mean read a
        # typical probe as 108% on a real run: profile views and cut-off faces pull it down, while a probe is
        # always a clean front-facing headshot
        self.ref_self_similarity = (reference_ceiling(np.stack([e for _, e in self.refs]), [not c for c in cut])
                                    if len(self.refs) >= 2 else None)
        self.ref_cut = cut
        if getattr(self, "gold", None) is not None:
            self._embed_gold_refs()
        log.info("Face references: %d of %d images had a face (%d cut off by the frame, left out of the ceiling); "
                 "typical best match among them %s", len(self.refs), len(paths), sum(cut),
                 f"{self.ref_self_similarity:.3f}" if self.ref_self_similarity is not None else "n/a")
        return info

    # ------------------------------------------------------------ the held-out recognizer
    def set_gold(self, embedder) -> None:
        """Score with a second recognizer too (an idloss.Embedder, e.g. FaceNet): the references are embedded with it
        now (and again whenever set_references runs)."""
        self.gold = embedder
        self._embed_gold_refs()

    def _gold_embed(self, path, kps) -> np.ndarray:
        from .idloss import embed_image, load_rgb

        return embed_image(self.gold, load_rgb(path), kps).float().cpu().numpy().astype(np.float32)

    def _embed_gold_refs(self) -> None:
        """The references' faces, at the landmarks InsightFace found, embedded by the gold recognizer; its ceiling is
        reference_ceiling over them (the frame's cut-off faces left out, as for ArcFace)."""
        from .likeness import reference_ceiling

        embs, usable = [], []
        for (path, _), kps, off in zip(self.refs, self.ref_kps, self.ref_cut or [False] * len(self.refs)):
            if kps is None:
                continue
            embs.append(self._gold_embed(path, kps))
            usable.append(not off)
        self.gold_refs = np.stack(embs) if embs else None
        self.gold_self_similarity = reference_ceiling(self.gold_refs, usable) if len(embs) >= 2 else None
        log.info("Gold face scorer (%s): %d reference face(s); typical best match among them %s",
                 getattr(self.gold, "name", "") or "held out", len(embs),
                 f"{self.gold_self_similarity:.3f}" if self.gold_self_similarity is not None else "n/a")

    # ------------------------------------------------------------ scoring
    def score(self, image: str | Path, step: int = 0, prompt_index: int = 0, gold: bool = False) -> FaceScore:
        """The largest face's best ArcFace match; gold: the held-out recognizer's too (set_gold), on the same face."""
        fs = self.faces(image)
        if not fs or not self.refs:
            return FaceScore(step, str(image), prompt_index, len(fs), None, None, None)
        emb = fs[0].normed_embedding.astype(np.float32)
        best, best_ref = -1.0, None
        for name, ref in self.refs:
            c = float(emb @ ref)
            if c > best:
                best, best_ref = c, name
        out = FaceScore(step, str(image), prompt_index, len(fs), best, max(0.0, min(1.0, best)) * 100.0, best_ref)
        kps = getattr(fs[0], "kps", None)
        if gold and getattr(self, "gold", None) is not None and self.gold_refs is not None and kps is not None:
            out.gold_cosine = float((self.gold_refs @ self._gold_embed(image, kps)).max())
            out.gold_percent = max(0.0, min(1.0, out.gold_cosine)) * 100.0
        return out

    @staticmethod
    def crop_face(path: str | Path, bbox, out: Path, pad: float = 0.25) -> Path:
        """Save a padded crop of a bbox (for the reference preview in the UI)."""
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            x0, y0, x1, y1 = bbox
            w, h = x1 - x0, y1 - y0
            box = (max(0, int(x0 - pad * w)), max(0, int(y0 - pad * h)),
                   min(im.width, int(x1 + pad * w)), min(im.height, int(y1 + pad * h)))
            im.crop(box).save(out)
        return out


def append_scores(job_dir: Path, scores: list[FaceScore]) -> None:
    with open(job_dir / "face_scores.jsonl", "a", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(asdict(s)) + "\n")


def read_scores(job_dir: Path) -> list[dict]:
    p = job_dir / "face_scores.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def score_samples(scorer, paths: list[str], step: int, job_dir: Path, body_scorer=None,
                  face_weight: float = 0.7, body_weight: float = 0.3, scale=None) -> list[FaceScore]:
    """scale: the likeness controller (or anything with .normalize and .norm(kind, value)): the samples' scores on
    the normalized likeness scale as well, so previews read like the probes."""
    t0 = time.time()
    scores = []
    for i, p in enumerate(paths):
        s = scorer.score(p, step, i) if scorer is not None else FaceScore(step, str(p), i, 0, None, None, None)
        if body_scorer is not None:
            s.body_percent, s.body_ratios = body_scorer.score(p)
        s.combined_percent = combine(s.percent, s.body_percent, face_weight, body_weight)
        if scale is not None and getattr(scale, "normalize", False):
            s.face_norm, s.body_norm = scale.norm("face", s.percent), scale.norm("body", s.body_percent)
            s.combined_norm = combine(s.face_norm, s.body_norm, face_weight, body_weight)
        scores.append(s)
    append_scores(job_dir, scores)
    for s in scores:
        face = (f"face {s.percent:.1f}% (cos {s.cosine:.3f})" if s.percent is not None
                else f"face: none ({s.faces} found)")
        body = ""
        if body_scorer is not None:
            body = (f"body {s.body_percent:.1f}% ({s.body_ratios} ratios)" if s.body_percent is not None
                    else "body: n/a (no usable body in the frame)")
        comb = f"combined {s.combined_percent:.1f}%" if s.combined_percent is not None else "combined: n/a"
        log.info("Similarity step %d sample %d: %s | %s | %s", step, s.prompt_index, face, body, comb)
    log.info("Similarity scoring took %.1fs", time.time() - t0)
    return scores
