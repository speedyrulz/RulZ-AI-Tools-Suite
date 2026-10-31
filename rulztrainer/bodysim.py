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
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        # bf16 on a GPU (1.7 GB instead of 3.4 GB); fp32 on the CPU, where bf16 convolutions are slow
        self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.model = Sam3Model.from_pretrained(model_id, dtype=self.dtype).eval()
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
        other.dtype = self.torch.float32 if device == "cpu" else self.torch.bfloat16
        return other

    def _segment(self, image, device: str, h: int, w: int) -> dict:
        torch = self.torch
        dtype = torch.float32 if device == "cpu" else getattr(self, "dtype", torch.float32)
        inputs = self.processor(images=image, text="person", return_tensors="pt")
        self.model.to(device, dtype)
        inputs = {k: (v.to(device, dtype) if torch.is_tensor(v) and v.is_floating_point() else
                      v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        for k, v in list(out.items()):  # post-processing (thresholds, resizing) in fp32
            if torch.is_tensor(v) and v.is_floating_point():
                out[k] = v.float()
        return self.processor.post_process_instance_segmentation(out, threshold=self.threshold,
                                                                 target_sizes=[(h, w)])[0]

    def person_mask(self, image) -> np.ndarray | None:
        """Largest 'person' instance mask as a bool [H, W] array, or None. An image that runs out of GPU memory is
        segmented on the CPU; the next one tries the GPU again (a transient peak, e.g. a probe beside the model,
        used to send every later image to the CPU for the rest of the run)."""
        torch = self.torch
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        res, oom = None, False
        try:
            res = self._segment(image, self.device, h, w)
        except torch.cuda.OutOfMemoryError:
            if self.device == "cpu":
                raise
            oom = True
        finally:
            if self.device != "cpu" and (oom or not self.held):
                self.model.to("cpu")
                torch.cuda.empty_cache()
        if oom:  # outside the except block: its traceback no longer holds the failed attempt's GPU tensors
            log.warning("SAM 3 ran out of GPU memory; this image is segmented on the CPU")
            res = self._segment(image, "cpu", h, w)
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


def mask_width_multiple() -> int:
    """The image width the pose model's person mask needs: mediapipe 1.0.0 and 1.0.1 abort the whole process (a native
    check, which Python cannot catch) when they read a float mask whose rows are padded, which is any width that is
    not a multiple of 4 (a 1233 px wide photo, say). 1.1.0 fixed it. On 1.0.x the image is padded on the right by up
    to 3 px (its own edge pixels) and the mask cut back, so scores can differ from 1.1+ by a hair."""
    try:
        import mediapipe

        major, minor = (int(x) for x in str(mediapipe.__version__).split(".")[:2])
    except Exception:  # noqa: BLE001
        return 1
    return 4 if (major, minor) == (1, 0) else 1


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
                 silhouette: "SilhouetteScorer | None" = None, progress=None, segment: bool = False):
        """segment: the pose model also returns its person mask (pose_and_mask), for the body window."""
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
                                            min_pose_detection_confidence=0.5, output_segmentation_masks=segment)
        self._mp = mp
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.min_visibility = min_visibility
        self.silhouette = silhouette
        self.pad_to = mask_width_multiple() if segment else 1
        if self.pad_to > 1:
            log.warning("mediapipe %s crashes reading person masks of some image widths: images are padded to a "
                        "width that is a multiple of %d (upgrade to mediapipe 1.1 or newer for exactly the scores "
                        "of other machines)", mp.__version__, self.pad_to)
        self.profile: dict[str, float] = {}
        self.ref_count = 0
        self.ref_self_similarity: float | None = None

    @staticmethod
    def _image(path: str | Path):
        """The image as RGB, EXIF-rotated, longest side at most 1536 (the size landmarks are found on)."""
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > 1536:
            im.thumbnail((1536, 1536))
        return im

    def _detect(self, im, world: bool = False):
        """(pts [33,2] pixels, vis [33], person mask [H, W] bool | None) or None if no person found; world=True adds
        the pose model's 3D landmarks [33,3] (metres around the hips: the pose's own depth estimate)."""
        arr = np.asarray(im)
        h, w = arr.shape[:2]
        pad = (-w) % getattr(self, "pad_to", 1)  # mediapipe 1.0.x only (mask_width_multiple)
        if pad:
            arr = np.pad(arr, ((0, 0), (0, pad), (0, 0)), mode="edge")
        arr = np.ascontiguousarray(arr)
        res = self.landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=arr))
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        pts = np.array([[q.x * (w + pad), q.y * h] for q in lm], dtype=np.float32)
        vis = np.array([q.visibility if q.visibility is not None else 1.0 for q in lm], dtype=np.float32)
        mask = None
        if getattr(res, "segmentation_masks", None):
            mask = np.squeeze(res.segmentation_masks[0].numpy_view())[:, :w] > 0.5
        if world:
            wl = res.pose_world_landmarks[0] if getattr(res, "pose_world_landmarks", None) else None
            xyz = None if wl is None else np.array([[q.x, q.y, q.z] for q in wl], dtype=np.float32)
            return pts, vis, mask, xyz
        return pts, vis, mask

    def landmarks(self, path: str | Path):
        """Returns (pts [33,2] pixels, vis [33]) or None if no person found."""
        found = self._detect(self._image(path))
        return None if found is None else (found[0], found[1])

    def pose_and_mask(self, path: str | Path):
        """(image, pts, vis, person mask) at the landmark size; pts, vis and mask None without a person. The mask
        needs segment=True."""
        im = self._image(path)
        found = self._detect(im)
        return (im, None, None, None) if found is None else (im, *found)

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


# ---------------------------------------------------------------- appearance (DINOv2 on the body window)

APPEARANCE_MODEL = "facebook/dinov2-base"
APPEARANCE_PATTERNS = ["config.json", "preprocessor_config.json", "model.safetensors"]
# the body window, in torso lengths (shoulder line to hip line) from the shoulder line: from the neck to mid-thigh,
# 1.6 torso lengths wide. The head belongs to the face score; legs below mid-thigh are often out of the frame (Krea 2
# draws a 2:3 "full body" probe from head to thigh)
WINDOW_TOP, WINDOW_BOTTOM, WINDOW_HALF_WIDTH = 0.15, 1.75, 0.80
WINDOW_MIN_COVER = 0.9  # a reference photo counts when this much of its window lies inside the photo
WINDOW_MIN_SCORE_COVER = 0.6  # an image with less of its window inside the frame gets no body score
WINDOW_GREY = (124, 116, 104)  # ~DINOv2's mean colour: background, out-of-frame parts, letterbox bars
_IMAGENET_MEAN, _IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def body_window(image, pts, vis, mask=None, min_vis: float = 0.5):
    """The body from the neck to mid-thigh, turned upright: the image is rotated about the shoulder midpoint until
    the shoulder->hip line points straight down, a window scaled by that torso length is cut out (WINDOW_* above),
    and every pixel outside the person mask or outside the photo turns grey. The same part of the body, the same
    way up and at the same scale in every photo and probe, so a lying, leaning or far-away body compares with a
    standing close one and the background, the head and the framing drop out.

    Returns (window image, coverage: the fraction of the window inside the photo) or None without both shoulders
    and both hips visible."""
    from PIL import Image

    if pts is None or vis is None or min(vis[L_SHO], vis[R_SHO], vis[L_HIP], vis[R_HIP]) < min_vis:
        return None
    s = (pts[L_SHO] + pts[R_SHO]) / 2.0
    v = (pts[L_HIP] + pts[R_HIP]) / 2.0 - s
    torso = float(np.hypot(v[0], v[1]))
    if torso < 16:
        return None
    angle = float(np.degrees(np.arctan2(v[0], v[1])))  # 0 with the hips straight below the shoulders
    center = (float(s[0]), float(s[1]))
    turned = image.rotate(-angle, resample=Image.BICUBIC, center=center, fillcolor=WINDOW_GREY)
    person = (Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255) if mask is not None
              else Image.new("L", image.size, 255))  # without a mask the background stays in
    person = person.rotate(-angle, resample=Image.NEAREST, center=center, fillcolor=0)  # 0 outside the photo
    inside = Image.new("L", image.size, 255).rotate(-angle, resample=Image.NEAREST, center=center, fillcolor=0)
    box = (int(round(s[0] - WINDOW_HALF_WIDTH * torso)), int(round(s[1] - WINDOW_TOP * torso)),
           int(round(s[0] + WINDOW_HALF_WIDTH * torso)), int(round(s[1] + WINDOW_BOTTOM * torso)))
    cover = float((np.asarray(inside.crop(box)) > 127).mean())
    keep = person.crop(box)  # crop() pads outside the image with 0: out of the frame is never the person
    return Image.composite(turned.crop(box), Image.new("RGB", keep.size, WINDOW_GREY), keep), cover


class AppearanceScorer:
    """Body likeness as a DINOv2 embedding of the body window (body_window: neck to mid-thigh, upright, cut out on
    grey, the head left to the face score), scored like the face: the best cosine match against the reference
    photos that show the same window, as a percent. The ceiling of the normalized scale is a typical reference's
    best match among the others (the median).

    Measured on a real run (60 photos: 20 close-ups, 40 medium to full-body shots; probes every 60 steps to step
    960): the earlier input, the padded pose box through the processor's 224 centre crop, kept half of a standing
    body and was matched against close-ups too, so the base model's stranger outscored the person's own photos
    against each other (0.73 against 0.67) and the score had no relation to training (rank correlation -0.13). The
    window against the 16 photos that show it rose with training (rank correlation +0.94; most of it survives in
    greyscale, so body shape and skin rather than lighting): 0 to about half of the way to the references by step 960,
    while the face reached them. Hair, build, skin and clothing count, so vary the outfits among the references."""

    def __init__(self, models_dir: str | Path = "models", device: str = "cuda:0", model_id: str = APPEARANCE_MODEL,
                 min_cover: float = WINDOW_MIN_COVER, progress=None):
        self.pose = BodyScorer(models_dir, progress=progress, segment=True)  # landmarks + person mask
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.min_cover = min_cover
        self.model = self.processor = None
        self.model_id = model_id
        self.progress = progress
        self.refs: list[tuple[str, np.ndarray]] = []
        self.ref_count = 0
        self.ref_self_similarity: float | None = None

    @classmethod
    def with_embedder(cls, embed, window, min_cover: float = WINDOW_MIN_COVER) -> "AppearanceScorer":
        """A scorer around given functions, for tests: embed(image) -> unit vector, window(path) -> (image,
        coverage) | None."""
        self = cls.__new__(cls)
        self.refs, self.ref_count, self.ref_self_similarity, self.min_cover = [], 0, None, min_cover
        self._embed, self.window = embed, window  # type: ignore[method-assign]
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
        self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16  # bf16 on a GPU: half the memory
        self.model = AutoModel.from_pretrained(path, dtype=self.dtype).to(self.device).eval().requires_grad_(False)

    def window(self, path: str | Path):
        """(body window image, coverage) of an image (body_window), or None without shoulders and hips."""
        im, pts, vis, mask = self.pose.pose_and_mask(path)
        return body_window(im, pts, vis, mask)

    @torch.no_grad()
    def _embed(self, image) -> np.ndarray:
        """DINOv2's CLS token of the image letterboxed to a square at 224 px: no centre crop, the whole window
        counts (the processor's own centre crop kept half of a tall crop)."""
        from PIL import Image

        self._load()
        side = max(image.size)
        square = Image.new("RGB", (side, side), WINDOW_GREY)
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        mean = np.asarray(getattr(self.processor, "image_mean", None) or _IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(getattr(self.processor, "image_std", None) or _IMAGENET_STD, dtype=np.float32)
        arr = (np.asarray(square.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0 - mean) / std
        x = torch.from_numpy(arr.transpose(2, 0, 1).copy())[None].to(self.device, self.model.dtype)
        v = self.model(pixel_values=x).last_hidden_state[:, 0].float()  # the CLS token
        return torch.nn.functional.normalize(v, dim=-1)[0].cpu().numpy()

    def set_references(self, paths: list[str | Path]) -> list[dict]:
        """Embed the body window of every reference that shows one. The photos whose window lies at least min_cover
        inside the frame are the references (a close-up has no hips, a waist-up shot no thighs); with fewer than two
        such photos every photo with shoulders and hips counts, with a warning."""
        from .facesim import expand_refs
        from .likeness import reference_ceiling

        found, info = [], []
        paths = expand_refs(paths)
        for p in paths:
            win = self.window(p)
            if win is None:
                info.append({"image": str(p), "person": False})
                continue
            image, cover = win
            found.append((str(p), self._embed(image), cover))
            info.append({"image": str(p), "person": True, "cover": round(float(cover), 3)})
        used = [f for f in found if f[2] >= self.min_cover]
        if len(used) < 2 and len(found) > len(used):
            log.warning("Body references: only %d of %d photos show the body from the neck to mid-thigh; scoring "
                        "against all %d with shoulders and hips, which is less reliable (add three-quarter or "
                        "full-length photos)", len(used), len(paths), len(found))
            used = found
        names = {p for p, _, _ in used}
        for item in info:
            if item["person"]:
                item["used"] = item["image"] in names
        self.refs = [(p, e) for p, e, _ in used]
        self.ref_count = len(self.refs)
        self.ref_self_similarity = (reference_ceiling(np.stack([e for _, e in self.refs]))
                                    if self.ref_count >= 2 else None)
        sim = self.ref_self_similarity
        log.info("Body references (appearance): %d of %d photos show shoulders and hips, %d of them the body down "
                 "to mid-thigh (the references); typical best match among them %s", len(found), len(paths),
                 self.ref_count, f"{sim:.3f}" if sim is not None else "n/a")
        return info

    def score(self, path: str | Path) -> tuple[float | None, int]:
        """(appearance similarity percent | None, references compared). None without shoulders and hips, or when
        less than WINDOW_MIN_SCORE_COVER of the window is inside the frame (a portrait, a waist-up shot)."""
        if not self.refs:
            return None, 0
        win = self.window(path)
        if win is None or win[1] < WINDOW_MIN_SCORE_COVER:
            return None, 0
        emb = self._embed(win[0])
        best = max(float(emb @ ref) for _, ref in self.refs)
        return max(0.0, min(1.0, best)) * 100.0, len(self.refs)
