"""Face similarity scoring for character LoRAs.

Reference faces come from one or more dataset images; each generated sample's largest detected face is
embedded with InsightFace (ArcFace "buffalo_l") and compared by cosine similarity against every reference
embedding. The reported score is the best match. The reference set's own pairwise similarity is reported
too, as the realistic ceiling for that character.

Cosine similarity for ArcFace: same identity is typically 0.5 to 0.8, different people 0.0 to 0.3.
"percent" is the cosine similarity clipped to [0, 1] times 100 (so 50%+ generally means "same person").
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
    body_percent: float | None = None  # body-proportion similarity (None = no usable pose)
    body_ratios: int = 0  # number of proportion ratios compared
    combined_percent: float | None = None  # weighted face/body score


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


class FaceScorer:
    def __init__(self, provider: str = "cpu", det_size: int = 640, model_name: str = "buffalo_l", progress=None):
        ok, msg = available()
        if not ok:
            raise RuntimeError(msg)
        from insightface.app import FaceAnalysis

        from .downloads import ensure_insightface_pack

        ensure_insightface_pack(model_name, progress=progress)
        if progress:
            progress(f"Loading face model {model_name}...", None)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=model_name, providers=providers,
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0 if provider == "cuda" else -1, det_size=(det_size, det_size))
        self.refs: list[tuple[str, np.ndarray]] = []
        self.ref_self_similarity: float | None = None

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

    def faces(self, path: str | Path):
        """All faces in an image, largest first, with .bbox / .kps in *source* pixels, .normed_embedding, .det_score."""
        arr, scale = self._load_bgr(path)
        found = self.app.get(arr)
        if scale != 1.0:
            for f in found:
                f.bbox = f.bbox * scale
                if getattr(f, "kps", None) is not None:
                    f.kps = f.kps * scale
        return sorted(found, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

    def set_references(self, paths: list[str | Path]) -> list[dict]:
        """Embed the largest face of each reference image. Returns per-image info for the UI."""
        self.refs = []
        info = []
        paths = expand_refs(paths)
        for p in paths:
            fs = self.faces(p)
            if not fs:
                info.append({"image": str(p), "faces": 0})
                log.warning("Face reference %s: no face detected", p)
                continue
            f = fs[0]
            self.refs.append((str(p), f.normed_embedding.astype(np.float32)))
            info.append({"image": str(p), "faces": len(fs), "bbox": [float(x) for x in f.bbox],
                         "det_score": float(f.det_score)})
        if len(self.refs) >= 2:
            embs = np.stack([e for _, e in self.refs])
            sims = embs @ embs.T
            n = len(embs)
            self.ref_self_similarity = float((sims.sum() - np.trace(sims)) / (n * (n - 1)))
        else:
            self.ref_self_similarity = None
        log.info("Face references: %d of %d images had a face; reference self-similarity %s", len(self.refs),
                 len(paths), f"{self.ref_self_similarity:.3f}" if self.ref_self_similarity is not None else "n/a")
        return info

    # ------------------------------------------------------------ scoring
    def score(self, image: str | Path, step: int = 0, prompt_index: int = 0) -> FaceScore:
        fs = self.faces(image)
        if not fs or not self.refs:
            return FaceScore(step, str(image), prompt_index, len(fs), None, None, None)
        emb = fs[0].normed_embedding.astype(np.float32)
        best, best_ref = -1.0, None
        for name, ref in self.refs:
            c = float(emb @ ref)
            if c > best:
                best, best_ref = c, name
        return FaceScore(step, str(image), prompt_index, len(fs), best, max(0.0, min(1.0, best)) * 100.0, best_ref)

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
                  face_weight: float = 0.7, body_weight: float = 0.3) -> list[FaceScore]:
    t0 = time.time()
    scores = []
    for i, p in enumerate(paths):
        s = scorer.score(p, step, i) if scorer is not None else FaceScore(step, str(p), i, 0, None, None, None)
        if body_scorer is not None:
            s.body_percent, s.body_ratios = body_scorer.score(p)
        s.combined_percent = combine(s.percent, s.body_percent, face_weight, body_weight)
        scores.append(s)
    append_scores(job_dir, scores)
    for s in scores:
        face = (f"face {s.percent:.1f}% (cos {s.cosine:.3f})" if s.percent is not None
                else f"face: none ({s.faces} found)")
        body = ""
        if body_scorer is not None:
            body = (f"body {s.body_percent:.1f}% ({s.body_ratios} ratios)" if s.body_percent is not None
                    else "body: no usable pose")
        comb = f"combined {s.combined_percent:.1f}%" if s.combined_percent is not None else "combined: n/a"
        log.info("Similarity step %d sample %d: %s | %s | %s", step, s.prompt_index, face, body, comb)
    log.info("Similarity scoring took %.1fs", time.time() - t0)
    return scores
