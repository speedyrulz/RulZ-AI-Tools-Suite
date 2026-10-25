"""Body-proportion similarity for character LoRAs (MediaPipe Pose landmarks).

A pose model finds 33 body landmarks. From them we compute scale-free *proportion ratios* (shoulder width
over hip width, torso length over leg length, upper over lower leg, upper arm over forearm, head width over
shoulder width, and so on). A reference profile is the per-ratio median over the body reference images; a
sample's body score is how close its ratios are to that profile.

Caveats: these are 2D ratios, so camera angle and pose change them. Use references (and sample prompts) that
show the body in a fairly neutral, upright pose for meaningful numbers. Ratios whose landmarks are not
visible are skipped; a sample with too few usable ratios gets no body score.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("rulztrainer")

POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/"
                  "float16/latest/pose_landmarker_full.task")
POSE_MODEL_NAME = "pose_landmarker_full.task"

# MediaPipe landmark indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28

RATIO_NAMES = ["shoulder/hip width", "torso/leg length", "upper/lower leg", "upper arm/forearm",
               "arm/torso length", "head/shoulder width", "torso/shoulder width", "leg/height"]
SILHOUETTE_NAMES = ["sil shoulder/height", "sil waist/height", "sil hip/height", "sil thigh/height",
                    "sil waist/hip", "sil shoulder/waist", "sil fullness"]
ALL_RATIO_NAMES = RATIO_NAMES + SILHOUETTE_NAMES


# ---------------------------------------------------------------- SAM 3 silhouette features


class SilhouetteScorer:
    """Segments the person with SAM 3 (text prompt "person") and measures silhouette widths.

    Widths are mask pixel counts per row (robust to gaps between legs), averaged over a few rows around each
    level, normalised by the mask height. Levels come from the MediaPipe skeleton when available (shoulders,
    hips, knees; waist = narrowest row between shoulders and hips), else from fixed fractions of the mask box.
    """

    def __init__(self, model_id: str = "facebook/sam3", device: str = "cuda:0", threshold: float = 0.3,
                 progress=None):
        import torch
        from transformers import Sam3Model, Sam3Processor

        from .downloads import ensure_hf_snapshot

        self.torch = torch
        if not Path(model_id).expanduser().is_dir():  # repo id: make sure it is cached, with progress
            ensure_hf_snapshot(model_id, "SAM 3", progress, allow_patterns=["*.json", "*.txt", "model.safetensors"])
        if progress:
            progress("Loading SAM 3 weights (3.4 GB)...", None)
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, dtype=torch.float32).eval()
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.threshold = threshold
        self.held = False  # True inside hold(): the model stays on the GPU between images
        log.info("SAM 3 loaded from %s (%.1fM params), scoring on %s",
                 model_id, sum(p.numel() for p in self.model.parameters()) / 1e6, self.device)

    def hold(self):
        """Context manager: keep SAM 3 on its GPU for a batch of images instead of moving it per image."""
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            self.held = True
            try:
                yield self
            finally:
                self.held = False
                if self.device != "cpu":
                    self.model.to("cpu")
                    self.torch.cuda.empty_cache()

        return _ctx()

    def clone(self, device: str) -> "SilhouetteScorer":
        """A copy on another device (own processor too: its tokenizer is not thread-safe)."""
        import copy

        other = object.__new__(SilhouetteScorer)
        other.torch, other.threshold, other.held = self.torch, self.threshold, False
        other.processor = copy.deepcopy(self.processor)
        other.model = copy.deepcopy(self.model.to("cpu")).eval()
        other.device = device
        return other

    def person_mask(self, image) -> np.ndarray | None:
        """Largest 'person' instance mask as a bool [H, W] array, or None."""
        torch = self.torch
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        inputs = self.processor(images=image, text="person", return_tensors="pt")
        try:
            self.model.to(self.device)
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(out, threshold=self.threshold,
                                                                    target_sizes=[(h, w)])[0]
        except torch.cuda.OutOfMemoryError:
            log.warning("SAM 3 ran out of GPU memory; retrying on CPU")
            self.device = "cpu"
            return self.person_mask(image)
        finally:
            if self.device != "cpu" and not self.held:
                self.model.to("cpu")
                torch.cuda.empty_cache()
        masks = res.get("masks")
        if masks is None or len(masks) == 0:
            return None
        masks = masks.bool().cpu().numpy()
        areas = masks.reshape(len(masks), -1).sum(1)
        return masks[int(areas.argmax())]

    @staticmethod
    def features(mask: np.ndarray, pts: np.ndarray | None, vis: np.ndarray | None, min_vis: float = 0.5) -> dict:
        # Only meaningful for a full-body view: require the skeleton to confirm shoulders, hips and an ankle.
        if pts is None or vis is None:
            return {}
        if not (vis[L_SHO] >= min_vis and vis[R_SHO] >= min_vis and vis[L_HIP] >= min_vis and vis[R_HIP] >= min_vis
                and max(vis[L_ANK], vis[R_ANK]) >= min_vis):
            return {}
        rows = np.where(mask.any(1))[0]
        if len(rows) < 20:
            return {}
        top, bottom = int(rows[0]), int(rows[-1])
        height = float(bottom - top)
        if height < 20:
            return {}
        counts = mask.sum(1).astype(np.float32)  # pixels per row

        def width_at(y: float, band: int = 4) -> float | None:
            y = int(round(y))
            lo, hi = max(top, y - band), min(bottom, y + band)
            if hi < lo:
                return None
            seg = counts[lo:hi + 1]
            seg = seg[seg > 0]
            return float(np.median(seg)) if len(seg) else None

        def level(idx_a, idx_b, frac):
            if pts is not None and vis is not None and vis[idx_a] >= min_vis and vis[idx_b] >= min_vis:
                return float((pts[idx_a][1] + pts[idx_b][1]) / 2)
            return top + frac * height

        sho_y = level(L_SHO, R_SHO, 0.20)
        hip_y = level(L_HIP, R_HIP, 0.50)
        knee_y = level(L_KNEE, R_KNEE, 0.72)
        if hip_y <= sho_y:
            sho_y, hip_y = top + 0.20 * height, top + 0.50 * height
        # waist: narrowest row in the middle 60% between shoulders and hips
        lo, hi = int(sho_y + 0.25 * (hip_y - sho_y)), int(hip_y - 0.15 * (hip_y - sho_y))
        waist_w = None
        if hi > lo:
            seg = counts[lo:hi + 1]
            if (seg > 0).any():
                waist_w = float(seg[seg > 0].min())
        sho_w, hip_w = width_at(sho_y), width_at(hip_y)
        thigh_w = width_at(hip_y + 0.3 * (knee_y - hip_y))
        out: dict[str, float] = {}
        if sho_w:
            out["sil shoulder/height"] = sho_w / height
        if waist_w:
            out["sil waist/height"] = waist_w / height
        if hip_w:
            out["sil hip/height"] = hip_w / height
        if thigh_w:
            out["sil thigh/height"] = thigh_w / height
        if waist_w and hip_w:
            out["sil waist/hip"] = waist_w / hip_w
        if sho_w and waist_w:
            out["sil shoulder/waist"] = sho_w / waist_w
        cols = np.where(mask.any(0))[0]
        box_w = float(cols[-1] - cols[0]) if len(cols) else 0.0
        if box_w > 0:
            out["sil fullness"] = float(mask.sum()) / (height * box_w)
        return {k: v for k, v in out.items() if math.isfinite(v)}


def available() -> tuple[bool, str]:
    try:
        import mediapipe  # noqa: F401
    except ImportError as e:
        return False, f"body scoring needs 'mediapipe' (pip install mediapipe): {e}"
    return True, ""


def ensure_model(models_dir: str | Path = "models", progress=None) -> Path:
    from .downloads import ensure_mediapipe_pose

    return ensure_mediapipe_pose(models_dir, POSE_MODEL_URL, POSE_MODEL_NAME, progress)


def _dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def proportion_ratios(pts: np.ndarray, vis: np.ndarray, min_vis: float = 0.5) -> dict[str, float]:
    """pts: [33, 2] pixel coords, vis: [33] visibility. Returns the ratios whose landmarks are visible."""

    def ok(*idx):
        return all(vis[i] >= min_vis for i in idx)

    def seg(i, j):
        return _dist(pts[i], pts[j])

    out: dict[str, float] = {}
    sho_w = seg(L_SHO, R_SHO) if ok(L_SHO, R_SHO) else None
    hip_w = seg(L_HIP, R_HIP) if ok(L_HIP, R_HIP) else None
    torso = _dist(_mid(pts[L_SHO], pts[R_SHO]), _mid(pts[L_HIP], pts[R_HIP])) if ok(L_SHO, R_SHO, L_HIP, R_HIP) else None
    legs = []
    for hip, knee, ank in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        if ok(hip, knee, ank):
            legs.append((seg(hip, knee), seg(knee, ank)))
    arms = []
    for sho, elb, wri in ((L_SHO, L_ELB, L_WRI), (R_SHO, R_ELB, R_WRI)):
        if ok(sho, elb, wri):
            arms.append((seg(sho, elb), seg(elb, wri)))
    head_w = seg(L_EAR, R_EAR) if ok(L_EAR, R_EAR) else None

    if sho_w and hip_w and hip_w > 1e-6:
        out["shoulder/hip width"] = sho_w / hip_w
    if torso and legs:
        leg_len = float(np.mean([u + l for u, l in legs]))
        if leg_len > 1e-6:
            out["torso/leg length"] = torso / leg_len
    if legs:
        out["upper/lower leg"] = float(np.mean([u / l for u, l in legs if l > 1e-6])) if any(l > 1e-6 for _, l in legs) else out.get("upper/lower leg")
    if arms:
        fore = [f for _, f in arms if f > 1e-6]
        if fore:
            out["upper arm/forearm"] = float(np.mean([u / f for u, f in arms if f > 1e-6]))
        if torso and torso > 1e-6:
            out["arm/torso length"] = float(np.mean([u + f for u, f in arms])) / torso
    if head_w and sho_w and sho_w > 1e-6:
        out["head/shoulder width"] = head_w / sho_w
    if torso and sho_w and sho_w > 1e-6:
        out["torso/shoulder width"] = torso / sho_w
    if legs and ok(NOSE) and any(ok(a) for a in (L_ANK, R_ANK)):
        ank = [pts[a] for a in (L_ANK, R_ANK) if ok(a)]
        height = float(np.mean([abs(pts[NOSE][1] - a[1]) for a in ank]))
        leg_len = float(np.mean([u + l for u, l in legs]))
        if height > 1e-6:
            out["leg/height"] = leg_len / height
    return {k: v for k, v in out.items() if v is not None and math.isfinite(v)}


def ratio_similarity(sample: dict[str, float], profile: dict[str, float], sigma: float = 0.2,
                     min_ratios: int = 3) -> tuple[float | None, int]:
    """Mean over shared ratios of exp(-(relative diff)^2 / (2 sigma^2)), as a percent. sigma=0.2 means a 20%
    deviation scores ~61%, a 40% deviation ~14%. Returns (percent | None, number of ratios compared)."""
    shared = [k for k in profile if k in sample]
    if len(shared) < min_ratios:
        return None, len(shared)
    scores = []
    for k in shared:
        a, b = sample[k], profile[k]
        rel = abs(a - b) / max(1e-6, (abs(a) + abs(b)) / 2)
        scores.append(math.exp(-(rel ** 2) / (2 * sigma ** 2)))
    return float(np.mean(scores)) * 100.0, len(shared)


class BodyScorer:
    def __init__(self, models_dir: str | Path = "models", min_visibility: float = 0.5,
                 silhouette: "SilhouetteScorer | None" = None, progress=None):
        ok, msg = available()
        if not ok:
            raise RuntimeError(msg)
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model = ensure_model(models_dir, progress)
        if progress:
            progress("Loading pose model...", None)
        opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)),
                                            running_mode=vision.RunningMode.IMAGE, num_poses=1,
                                            min_pose_detection_confidence=0.5)
        self._mp = mp
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.min_visibility = min_visibility
        self.silhouette = silhouette
        self.profile: dict[str, float] = {}
        self.ref_count = 0
        self.ref_self_similarity: float | None = None

    def landmarks(self, path: str | Path):
        """Returns (pts [33,2] pixels, vis [33]) or None if no person found."""
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > 1536:
                im.thumbnail((1536, 1536))
            arr = np.asarray(im)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))
        res = self.landmarker.detect(mp_img)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        h, w = arr.shape[:2]
        pts = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)
        vis = np.array([l.visibility if l.visibility is not None else 1.0 for l in lm], dtype=np.float32)
        return pts, vis

    def ratios(self, path: str | Path) -> dict[str, float] | None:
        """Skeleton ratios (MediaPipe) plus, when SAM 3 is enabled, silhouette ratios. None if nothing usable."""
        lm = self.landmarks(path)
        out: dict[str, float] = {}
        if lm is not None:
            out.update(proportion_ratios(lm[0], lm[1], self.min_visibility))
        if self.silhouette is not None:
            from PIL import Image, ImageOps

            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                if max(im.size) > 1536:
                    im.thumbnail((1536, 1536))
                mask = self.silhouette.person_mask(im)  # same 1536-capped size as the landmarks
                if mask is not None:
                    pts, vis = (None, None) if lm is None else lm
                    out.update(SilhouetteScorer.features(mask, pts, vis, self.min_visibility))
        return out or None

    def set_references(self, paths: list[str | Path]) -> list[dict]:
        from .facesim import expand_refs

        per_image: list[dict[str, float]] = []
        info = []
        paths = expand_refs(paths)
        for p in paths:
            r = self.ratios(p)
            if not r:
                info.append({"image": str(p), "ratios": 0})
                log.warning("Body reference %s: no pose detected", p)
                continue
            per_image.append(r)
            info.append({"image": str(p), "ratios": len(r), "values": r})
        self.profile = {}
        for k in ALL_RATIO_NAMES:
            vals = [r[k] for r in per_image if k in r]
            if vals:
                self.profile[k] = float(np.median(vals))
        self.ref_count = len(per_image)
        if len(per_image) >= 2:
            sims = [ratio_similarity(r, self.profile)[0] for r in per_image]
            sims = [s for s in sims if s is not None]
            self.ref_self_similarity = float(np.mean(sims)) / 100.0 if sims else None
        else:
            self.ref_self_similarity = None
        log.info("Body references: %d of %d images had a pose; profile ratios %s; self-similarity %s",
                 self.ref_count, len(paths), {k: round(v, 3) for k, v in self.profile.items()},
                 f"{self.ref_self_similarity:.3f}" if self.ref_self_similarity is not None else "n/a")
        return info

    @property
    def ready(self) -> bool:
        return bool(self.profile)

    def score(self, path: str | Path) -> tuple[float | None, int]:
        """(body similarity percent | None, ratios compared)."""
        if not self.profile:
            return None, 0
        r = self.ratios(path)
        if not r:
            return None, 0
        return ratio_similarity(r, self.profile)


# ---------------------------------------------------------------- appearance (DINOv2 on the person crop)

APPEARANCE_MODEL = "facebook/dinov2-base"
APPEARANCE_PATTERNS = ["config.json", "preprocessor_config.json", "model.safetensors"]


class AppearanceScorer:
    """Body likeness as a DINOv2 embedding of the person crop (MediaPipe's pose box, padded), scored like the face:
    the best cosine match against the reference crops, as a percent.

    Measured on a 15-photo dataset against 11 base-model probes: the worst same-person match (leave-one-out) 0.49,
    the best stranger 0.26 (CLS token; whole images overlapped, scene and clothing dominating). A stranger scores
    ~10-26, the person 50-70+, the face score's range. Hair, build, skin and style count, and clothing too: vary the
    outfits among the references. The proportion score (BodyScorer) could not score that dataset's references at all
    (one of eight skeleton ratios measurable)."""

    def __init__(self, models_dir: str | Path = "models", device: str = "cuda:0", model_id: str = APPEARANCE_MODEL,
                 pad: float = 0.2, progress=None):
        self.pose = BodyScorer(models_dir, progress=progress)  # the person box
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.pad = pad
        self.model = self.processor = None
        self.model_id = model_id
        self.progress = progress
        self.refs: list[tuple[str, np.ndarray]] = []
        self.ref_count = 0
        self.ref_self_similarity: float | None = None

    @classmethod
    def with_embedder(cls, embed, crop) -> "AppearanceScorer":
        """A scorer around given functions, for tests: embed(PIL image) -> unit vector, crop(path) -> image | None."""
        self = cls.__new__(cls)
        self.refs, self.ref_count, self.ref_self_similarity = [], 0, None
        self._embed, self.person_crop = embed, crop  # type: ignore[method-assign]
        return self

    @property
    def ready(self) -> bool:
        return bool(self.refs)

    def _load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        from .downloads import ensure_hf_snapshot

        path = ensure_hf_snapshot(self.model_id, "DINOv2", self.progress, allow_patterns=APPEARANCE_PATTERNS)
        if self.progress:
            self.progress("Loading DINOv2...", None)
        self.processor = AutoImageProcessor.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval().requires_grad_(False)

    def person_crop(self, path: str | Path):
        """The person's box from the visible pose landmarks, padded; None without a pose."""
        from PIL import Image, ImageOps

        lm = self.pose.landmarks(path)
        if lm is None:
            return None
        pts, vis = lm
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > 1536:  # the landmarks were found on this size
                im.thumbnail((1536, 1536))
            p = pts[vis > 0.5] if int((vis > 0.5).sum()) >= 4 else pts
            x0, y0 = p.min(0)
            x1, y1 = p.max(0)
            w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
            box = (max(0, int(x0 - self.pad * w)), max(0, int(y0 - self.pad * h)),
                   min(im.width, int(x1 + self.pad * w)), min(im.height, int(y1 + self.pad * h)))
            return im.crop(box)

    @torch.no_grad()
    def _embed(self, image) -> np.ndarray:
        self._load()
        x = self.processor(images=image, return_tensors="pt").to(self.device)
        v = self.model(**x).last_hidden_state[:, 0].float()  # the CLS token
        return torch.nn.functional.normalize(v, dim=-1)[0].cpu().numpy()

    def set_references(self, paths: list[str | Path]) -> list[dict]:
        from .facesim import expand_refs

        self.refs, info = [], []
        for p in expand_refs(paths):
            crop = self.person_crop(p)
            if crop is None:
                info.append({"image": str(p), "person": False})
                log.warning("Body reference %s: no person found", p)
                continue
            self.refs.append((str(p), self._embed(crop)))
            info.append({"image": str(p), "person": True, "crop_size": list(getattr(crop, "size", ()) or ())})
        self.ref_count = len(self.refs)
        if self.ref_count >= 2:
            embs = np.stack([e for _, e in self.refs])
            sims = embs @ embs.T
            np.fill_diagonal(sims, -1.0)
            self.ref_self_similarity = float(sims.max(axis=1).mean())  # each reference's best match: the ceiling
        else:
            self.ref_self_similarity = None
        log.info("Body references (appearance): %d of %d images had a person; self-similarity %s", self.ref_count,
                 len(info), f"{self.ref_self_similarity:.3f}" if self.ref_self_similarity is not None else "n/a")
        return info

    def score(self, path: str | Path) -> tuple[float | None, int]:
        """(appearance similarity percent | None, references compared)."""
        if not self.refs:
            return None, 0
        crop = self.person_crop(path)
        if crop is None:
            return None, 0
        emb = self._embed(crop)
        best = max(float(emb @ ref) for _, ref in self.refs)
        return max(0.0, min(1.0, best)) * 100.0, len(self.refs)
