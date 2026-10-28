"""Pose-independent body-shape scores (body_score "skeleton3d" and "smplx").

The "proportions" score (bodysim.BodyScorer) measures the skeleton on the flat picture, so the pose and the camera
change it more than one person differs from another: on 19 photos of one woman, her own 2D ratios spread 15-50%
(sitting, kneeling, selfies) where adults differ by 5-10%, and only 4 photos had legs to measure. These two scores
measure the body so that a bent leg is still the same leg:

  * skeleton3d: MediaPipe's 3D landmarks (its own depth estimate, metres around the hips): bone-length ratios
    (shoulder / hip width, torso / leg, thigh / shin, upper arm / forearm, arm / torso, head / shoulder width) in 3D,
    so most poses count, plus the silhouette widths of the pose model's person mask (shoulder, waist, hip, thigh,
    waist / hip) only where the person stands upright and faces the camera, the one view where a width is a width.

  * smplx: NLF (Neural Localizer Fields, Sarandi & Pons-Moll, NeurIPS 2024) fits the SMPL-X body model to the
    person: 10 shape parameters (betas) that do not depend on the pose. The betas rebuild the body in the model's
    T-pose (the SMPL-X model ships inside NLF's file), where it is measured: shoulder, leg, torso and arm length,
    bust, waist, hip and thigh girth, all over the height (one photo cannot tell absolute size), and waist / hip.
    NLF's weights are for noncommercial research use, so they are not downloaded for you: put
    nlf_l_multi_0.3.2.patch6.torchscript (github.com/isarandi/nlf/releases, v0.3.2, 550 MB) in models/nlf/.

Both compare a picture with the references through a calibrated profile (Profile): per measurement the references'
median and their own spread (a robust standard deviation, at least 3% of the median), so a measurement that varies a
lot between the photos of the one person counts for little and a steady one counts for a lot. The score is the mean
over the shared measurements of exp(-z^2 / 2), as a percent; the references' own level (the normalized 100) is each
reference scored against the others (leave one out).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from .bodysim import (L_ANK, L_EAR, L_ELB, L_HIP, L_KNEE, L_SHO, L_WRI, NOSE, R_ANK, R_EAR, R_ELB, R_HIP, R_KNEE,
                      R_SHO, R_WRI, BodyScorer, SilhouetteScorer)

log = logging.getLogger("rulztrainer")

MIN_REL = 0.03  # a measurement's spread is at least this share of its median (one person's photos never agree better)
DEFAULT_REL = 0.10  # the spread of a measurement fewer than MIN_REFS references have
MIN_REFS = 3
MIN_KEYS = 3  # a picture needs this many measurements in common with the references to be scored


class Profile:
    """The references' measurements: per key the median and a robust spread (1.4826 x the median absolute deviation,
    at least MIN_REL of the median; DEFAULT_REL with fewer than MIN_REFS references)."""

    def __init__(self, samples: list[dict[str, float]]):
        self.median: dict[str, float] = {}
        self.sd: dict[str, float] = {}
        self.count: dict[str, int] = {}
        for k in sorted({k for s in samples for k in s}):
            vals = np.array([s[k] for s in samples if k in s], dtype=np.float64)
            med = float(np.median(vals))
            sd = 1.4826 * float(np.median(np.abs(vals - med))) if len(vals) >= MIN_REFS else DEFAULT_REL * abs(med)
            self.median[k], self.sd[k], self.count[k] = med, max(sd, MIN_REL * abs(med), 1e-6), len(vals)

    def similarity(self, sample: dict[str, float], min_keys: int = MIN_KEYS) -> tuple[float | None, int]:
        """(percent | None, measurements compared): the mean of exp(-z^2 / 2) over the shared measurements."""
        shared = [k for k in self.median if k in sample]
        if len(shared) < min_keys:
            return None, len(shared)
        z = [(sample[k] - self.median[k]) / self.sd[k] for k in shared]
        return float(np.mean([math.exp(-v * v / 2) for v in z])) * 100.0, len(shared)

    @staticmethod
    def self_similarity(samples: list[dict[str, float]]) -> float | None:
        """The references' own level, 0..1: each scored against a profile of the others."""
        sims = []
        for i, s in enumerate(samples):
            others = samples[:i] + samples[i + 1:]
            if len(others) < 2:
                continue
            pct, _ = Profile(others).similarity(s)
            if pct is not None:
                sims.append(pct)
        return float(np.mean(sims)) / 100.0 if sims else None

    def describe(self) -> dict[str, str]:
        return {k: f"{self.median[k]:.3f} +/- {self.sd[k] / max(1e-9, abs(self.median[k])) * 100:.0f}%"
                for k in self.median}


class _ProfileScorer:
    """set_references / score / ready / ref_count / ref_self_similarity on top of a ratios(path) method."""

    kind = "body"

    def ratios(self, path) -> dict[str, float] | None:  # pragma: no cover - subclasses
        raise NotImplementedError

    def set_references(self, paths) -> list[dict]:
        from .facesim import expand_refs

        per_image, info = [], []
        for p in expand_refs(paths):
            r = self.ratios(p)
            info.append({"image": str(p), "ratios": len(r or {}), "values": r or {}})
            if r:
                per_image.append(r)
            else:
                log.info("Body reference %s: no measurable body (%s)", Path(p).name, self.kind)
        self.profile = Profile(per_image) if per_image else None
        self.ref_count = len(per_image)
        self.ref_self_similarity = Profile.self_similarity(per_image) if len(per_image) >= 3 else None
        if self.profile:
            log.info("Body references (%s): %d of %d measurable; profile %s; references' own level %s", self.kind,
                     len(per_image), len(info), self.profile.describe(),
                     f"{self.ref_self_similarity * 100:.1f}%" if self.ref_self_similarity is not None else "n/a")
        return info

    @property
    def ready(self) -> bool:
        return bool(getattr(self, "profile", None))

    def score(self, path) -> tuple[float | None, int]:
        """(body similarity percent | None, measurements compared)."""
        if not self.ready:
            return None, 0
        r = self.ratios(path)
        return (None, 0) if not r else self.profile.similarity(r)


# ---------------------------------------------------------------- skeleton3d (MediaPipe 3D landmarks)


def bone_ratios_3d(xyz: np.ndarray, vis: np.ndarray, min_vis: float = 0.5) -> dict[str, float]:
    """Bone-length ratios from 3D landmarks [33, 3]: the same whether a limb is bent or straight, turned or not.
    Each ratio needs its landmarks visible; a pair of limbs is averaged over the sides that are."""

    def ok(*idx):
        return all(vis[i] >= min_vis for i in idx)

    def seg(i, j):
        return float(np.linalg.norm(xyz[i] - xyz[j]))

    def side_mean(pairs):
        vals = [f() for need, f in pairs if ok(*need)]
        return float(np.mean(vals)) if vals else None

    sho_w = seg(L_SHO, R_SHO) if ok(L_SHO, R_SHO) else None
    hip_w = seg(L_HIP, R_HIP) if ok(L_HIP, R_HIP) else None
    torso = (float(np.linalg.norm((xyz[L_SHO] + xyz[R_SHO]) / 2 - (xyz[L_HIP] + xyz[R_HIP]) / 2))
             if ok(L_SHO, R_SHO, L_HIP, R_HIP) else None)
    thigh = side_mean([((L_HIP, L_KNEE), lambda: seg(L_HIP, L_KNEE)), ((R_HIP, R_KNEE), lambda: seg(R_HIP, R_KNEE))])
    shin = side_mean([((L_KNEE, L_ANK), lambda: seg(L_KNEE, L_ANK)), ((R_KNEE, R_ANK), lambda: seg(R_KNEE, R_ANK))])
    upper = side_mean([((L_SHO, L_ELB), lambda: seg(L_SHO, L_ELB)), ((R_SHO, R_ELB), lambda: seg(R_SHO, R_ELB))])
    fore = side_mean([((L_ELB, L_WRI), lambda: seg(L_ELB, L_WRI)), ((R_ELB, R_WRI), lambda: seg(R_ELB, R_WRI))])
    head_w = seg(L_EAR, R_EAR) if ok(L_EAR, R_EAR) else None
    out: dict[str, float] = {}

    def put(name, a, b):
        if a and b and b > 1e-6:
            out[name] = a / b

    put("3d shoulder/hip width", sho_w, hip_w)
    put("3d torso/leg length", torso, (thigh + shin) if thigh and shin else None)
    put("3d thigh/shin", thigh, shin)
    put("3d upper arm/forearm", upper, fore)
    put("3d arm/torso length", (upper + fore) if upper and fore else None, torso)
    put("3d shoulder width/torso", sho_w, torso)
    put("3d head/shoulder width", head_w, sho_w)
    return {k: v for k, v in out.items() if math.isfinite(v)}


def upright_frontal(pts: np.ndarray, xyz: np.ndarray | None, vis: np.ndarray, min_vis: float = 0.5,
                    size: tuple[int, int] | None = None) -> bool:
    """Standing upright, legs straight, facing the camera, head to feet in the frame: the one view where the
    silhouette's widths are the body's widths."""
    need = (NOSE, L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK)
    if xyz is None or not all(vis[i] >= min_vis for i in need):
        return False
    if size is not None:
        w, h = size
        if not all(0 <= pts[i][0] < w and 0 <= pts[i][1] < h * 0.985 for i in need):
            return False
    for a, b, c, d in ((L_SHO, L_HIP, L_KNEE, L_ANK), (R_SHO, R_HIP, R_KNEE, R_ANK)):
        if not pts[NOSE][1] < pts[a][1] < pts[b][1] < pts[c][1] < pts[d][1]:  # image y grows downward
            return False
    sho_mid, hip_mid = (pts[L_SHO] + pts[R_SHO]) / 2, (pts[L_HIP] + pts[R_HIP]) / 2
    tilt = math.degrees(math.atan2(abs(sho_mid[0] - hip_mid[0]), abs(hip_mid[1] - sho_mid[1]) + 1e-6))
    if tilt > 15:
        return False
    for hip, knee, ank in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        u, v = xyz[hip] - xyz[knee], xyz[ank] - xyz[knee]
        cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
        if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) < 155:  # a bent knee
            return False
    for a, b in ((L_SHO, R_SHO), (L_HIP, R_HIP)):  # turned away: one side much nearer the camera
        width = float(np.linalg.norm(xyz[a] - xyz[b]))
        if width < 1e-6 or abs(float(xyz[a][2] - xyz[b][2])) > 0.5 * width:
            return False
    return True


class Skeleton3DScorer(_ProfileScorer):
    """body_score "skeleton3d": 3D bone ratios everywhere, silhouette widths where upright and frontal."""

    kind = "skeleton3d"

    def __init__(self, models_dir: str | Path = "models", min_visibility: float = 0.5, progress=None):
        self.pose = BodyScorer(models_dir, min_visibility=min_visibility, progress=progress, segment=True)
        self.min_visibility = min_visibility
        self.profile = None
        self.ref_count = 0
        self.ref_self_similarity = None

    def ratios(self, path) -> dict[str, float] | None:
        im = self.pose._image(path)
        found = self.pose._detect(im, world=True)
        if found is None:
            return None
        pts, vis, mask, xyz = found
        out = bone_ratios_3d(xyz, vis, self.min_visibility) if xyz is not None else {}
        if mask is not None and upright_frontal(pts, xyz, vis, self.min_visibility, im.size):
            out.update(SilhouetteScorer.features(mask, pts, vis, self.min_visibility))
        return out or None


# ---------------------------------------------------------------- smplx (NLF's SMPL-X fit)

NLF_DIR = "nlf"
NLF_FILE = "nlf_l_multi_0.3.2.patch6.torchscript"
NLF_URL = "https://github.com/isarandi/nlf/releases/download/v0.3.2/" + NLF_FILE
NLF_BETA_REGULARIZER = 1.0  # NLF's default 10 pulls every body toward the average; 1 keeps the shape, steadily

# SMPL-X joints
PELVIS, HIP_L, HIP_R, SPINE1, KNEE_L, KNEE_R, SPINE2, ANKLE_L, ANKLE_R, SPINE3 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
NECK, COLLAR_L, COLLAR_R, SHO_L, SHO_R, ELB_L, ELB_R, WRI_L, WRI_R = 12, 13, 14, 16, 17, 18, 19, 20, 21
TORSO_PARTS = (PELVIS, SPINE1, SPINE2, SPINE3, COLLAR_L, COLLAR_R)  # skinning parts of the trunk (no arms or head)


def find_nlf(path: str = "", models_dir: str | Path = "models") -> Path | None:
    """The NLF TorchScript file: `path` (a file or a folder), else models/nlf/ (the newest nlf_*multi*.torchscript)."""
    for cand in ([Path(path).expanduser()] if path else []) + [Path(models_dir) / NLF_DIR]:
        if cand.is_file():
            return cand
        if cand.is_dir():
            files = sorted(cand.glob("nlf_*multi*.torchscript"))
            if files:
                return files[-1]
    return None


def missing_nlf_message(models_dir: str | Path = "models") -> str:
    return (f"SMPL-X body shape needs NLF's model, which is licensed for noncommercial research use and is not "
            f"downloaded automatically: download {NLF_FILE} (550 MB) from {NLF_URL} and put it in "
            f"{Path(models_dir) / NLF_DIR}")


def _convex_perimeter(pts: np.ndarray) -> float | None:
    """Perimeter of the convex hull of 2D points (Andrew's monotone chain): a girth, as a tape measure takes it."""
    pts = np.unique(np.round(pts, 6), axis=0)
    if len(pts) < 3:
        return None
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    return float(np.linalg.norm(hull - np.roll(hull, -1, axis=0), axis=1).sum())


def tpose_measurements(verts: np.ndarray, joints: np.ndarray, part: np.ndarray) -> dict[str, float]:
    """Proportions of an SMPL-X body in its T-pose (y up): lengths and girths over the height, and waist / hip.
    verts [N, 3], joints [55, 3], part [N]: each vertex's main skinning joint."""
    y = verts[:, 1]
    height = float(y.max() - y.min())
    if height <= 0:
        return {}
    band = 0.006 * height  # a slice is the vertices within this of a level: about 1 cm

    def girth(level: float, parts) -> float | None:
        sel = (np.abs(y - level) <= band) & np.isin(part, parts)
        return _convex_perimeter(verts[sel][:, [0, 2]]) if sel.sum() >= 8 else None

    def extreme(lo: float, hi: float, parts, pick) -> float | None:
        vals = [g for g in (girth(v, parts) for v in np.linspace(lo, hi, 12)) if g]
        return pick(vals) if vals else None

    jy = joints[:, 1]
    bust = extreme(jy[SPINE2], jy[SPINE3] + 0.3 * (jy[NECK] - jy[SPINE3]), TORSO_PARTS, max)
    waist = extreme(jy[SPINE1], jy[SPINE2] + 0.5 * (jy[SPINE3] - jy[SPINE2]), TORSO_PARTS, min)
    hip = extreme(jy[HIP_L] - 0.04 * height, jy[PELVIS] + 0.03 * height, (PELVIS, HIP_L, HIP_R, SPINE1), max)
    thigh = extreme(jy[KNEE_L] + 0.5 * (jy[HIP_L] - jy[KNEE_L]), jy[HIP_L] - 0.03 * height, (HIP_L,), max)

    def dist(a, b):
        return float(np.linalg.norm(joints[a] - joints[b]))

    leg = (dist(HIP_L, KNEE_L) + dist(KNEE_L, ANKLE_L) + dist(HIP_R, KNEE_R) + dist(KNEE_R, ANKLE_R)) / 2
    arm = (dist(SHO_L, ELB_L) + dist(ELB_L, WRI_L) + dist(SHO_R, ELB_R) + dist(ELB_R, WRI_R)) / 2
    out = {"smplx shoulder/height": dist(SHO_L, SHO_R) / height, "smplx leg/height": leg / height,
           "smplx torso/height": float(jy[NECK] - jy[PELVIS]) / height, "smplx arm/height": arm / height}
    for name, g in (("bust", bust), ("waist", waist), ("hip", hip), ("thigh", thigh)):
        if g:
            out[f"smplx {name}/height"] = g / height
    if waist and hip:
        out["smplx waist/hip"] = waist / hip
    return out


class SMPLXScorer(_ProfileScorer):
    """body_score "smplx": NLF's SMPL-X shape, measured in the T-pose. The model waits on the CPU between pictures
    (hold() keeps it on the GPU for a batch)."""

    kind = "smplx"

    def __init__(self, model_path: str = "", models_dir: str | Path = "models", device: str = "cuda:0",
                 beta_regularizer: float = NLF_BETA_REGULARIZER, progress=None):
        import torch
        import torchvision  # noqa: F401  (registers the detector's ops NLF's TorchScript calls)

        path = find_nlf(model_path, models_dir)
        if path is None:
            raise FileNotFoundError(missing_nlf_message(models_dir))
        if progress:
            progress("Loading NLF (SMPL-X body shape)...", None)
        self.torch = torch
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.beta_regularizer = float(beta_regularizer)
        self.held = False
        bm = self.model.body_models.smplx
        self.parts = bm.weights.detach().cpu().numpy().argmax(1)
        self.profile = None
        self.ref_count = 0
        self.ref_self_similarity = None
        self.last_betas = None
        log.info("NLF loaded from %s; SMPL-X shape scored on %s", path.name, self.device)

    def hold(self):
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

    def betas(self, path) -> np.ndarray | None:
        """The SMPL-X betas [10] of the picture's main person, or None without one whose body (shoulders, hips and
        knees) is in the frame: a close-up's shape would be the model's guess, not the person's."""
        from PIL import Image, ImageOps

        torch = self.torch
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > 1536:
            im.thumbnail((1536, 1536))
        w, h = im.size
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
        try:
            self.model.to(self.device)
            with torch.inference_mode(), torch.device(self.device):
                pred = self.model.detect_smpl_batched(x[None].to(self.device), model_name="smplx",
                                                      beta_regularizer=self.beta_regularizer)
        finally:
            if self.device != "cpu" and not self.held:
                self.model.to("cpu")
                torch.cuda.empty_cache()
        betas, j2d = pred["betas"][0], pred["joints2d"][0]
        if not len(betas):
            return None
        spans = [float((j[:, 0].max() - j[:, 0].min()) * (j[:, 1].max() - j[:, 1].min())) for j in j2d]
        i = int(np.argmax(spans))  # the largest person
        j = j2d[i].float().cpu().numpy()
        m = 0.01 * max(w, h)
        if not all(-m <= j[k][0] <= w + m and -m <= j[k][1] <= h + m
                   for k in (SHO_L, SHO_R, HIP_L, HIP_R, KNEE_L, KNEE_R)):
            return None
        return betas[i].float().cpu().numpy()

    def tpose(self, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        bm = self.model.body_models.smplx
        with torch.inference_mode():
            out = bm(shape_betas=torch.as_tensor(betas, dtype=torch.float32, device=bm.v_template.device)[None])
        return out["vertices"][0].cpu().numpy(), out["joints"][0].cpu().numpy()

    def ratios(self, path) -> dict[str, float] | None:
        b = self.betas(path)
        self.last_betas = b
        if b is None:
            return None
        verts, joints = self.tpose(b)
        return tpose_measurements(verts, joints, self.parts) or None
