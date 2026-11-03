"""The anchored body score (body_score "anchored"): how far a picture's body has moved from the base model's people
toward the references, measured on the body window with its photographic look taken out.

    window       bodysim.body_window: the neck to mid-thigh, turned upright, the person cut out on grey
    canonical    each colour channel of the person's pixels standardised (mean 0.5, sd 0.18), then the window shrunk
                 to 72 px tall and back up: no colour grade, exposure, grain or sharpness differences; build, shading,
                 skin, hair and clothing shape stay
    embeddings   SigLIP 2 (google/siglip2-base-patch16-384, its image half) and DINOv2-base's CLS token, each on the
                 canonical window letterboxed to a square
    strangers    the step-0 probe's body renders: the base model without the LoRA, the probe prompt, several seeds
    score        per model, 100 * (x - strangers' mean) . d / |d|^2 with d = references' mean - strangers' mean; the
                 two models averaged. 0 = the strangers' mean, 100 = the references' mean, linear: it can go below 0 or
                 past 100

Why. On a real comparison (one character, 10 body references, five LoRA runs to step 2000, 25 base-model renders of
the body probe prompt as strangers) every absolute body score (the DINOv2 best match, 2D proportions, SAM 3
silhouette widths, 3D bone ratios, SMPL-X shape) rated the person's own photos against each other at or below the
base model's strangers: the references vary far more in pose, outfit, light and framing than the base model's woman
differs from the person, so a ceiling built from them sat under the floor. Measured from the strangers toward the
references, the late samples (steps 1000-2000) beat the strangers in 96% of pairs; the person's own photo in the
probe scene (scored without it among the references) sat 7 stranger spreads above them. Fitted on the raw windows the
same idea separated better (98%) but learned the photos' look: the references were darker, more saturated and softer
than the base model's renders, strangers merely recoloured, softened and grained like them rose 5.5 spreads, and the
run with the softest samples ranked first. The canonical window takes that out (0.1 spreads). With 4 strangers the
ranking agreed with the 25-stranger one at 0.97, with 8 at 0.98, so the step-0 probe renders STRANGER_SEEDS body seeds.

Reading it. The references' own mean is 100, a typical reference scored against the others ~80, and a probe in one
fixed outfit and scene reaches less (the photo that probe prompt was written from scored 40): compare runs and
checkpoints with it, not with 100. A LoRA also pulls the pose toward the dataset's, and the window keeps the pose.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .bodysim import (WINDOW_GREY, WINDOW_MIN_COVER, WINDOW_MIN_SCORE_COVER, BodyScorer, _IMAGENET_MEAN, _IMAGENET_STD,
                      body_window)

log = logging.getLogger("rulztrainer")

SIGLIP_MODEL = "google/siglip2-base-patch16-384"
DINO_MODEL = "facebook/dinov2-base"
MODEL_PATTERNS = ["config.json", "preprocessor_config.json", "model.safetensors"]
BACKBONES = ("siglip", "dino")
CANON_MEAN, CANON_SD = 0.5, 0.18  # the person's pixels per channel, as a fraction of 255
CANON_ROWS = 72  # the low-pass: the window shrunk to this many rows and back up
STRANGER_SEEDS = 8  # the step-0 probe's body seeds when this score needs its strangers
MIN_STRANGERS = 2
STRANGERS_FILE = "body_strangers.npz"  # in the job's probes folder: a resumed run reads its strangers back


def person_pixels(window) -> np.ndarray:
    """The window's person mask: every pixel that is not the window's grey."""
    return np.any(np.asarray(window, np.float32) != np.asarray(WINDOW_GREY, np.float32), axis=2)


def canonical_window(window):
    """The window with its photographic look taken out (the module docstring), or None with under 100 person pixels."""
    from PIL import Image

    a = np.asarray(window, np.float32)
    keep = person_pixels(window)
    if keep.sum() < 100:
        return None
    out = np.empty_like(a)
    for c in range(3):
        v = a[..., c]
        m, s = v[keep].mean(), v[keep].std() + 1e-6
        out[..., c] = np.where(keep, ((v - m) / s * CANON_SD + CANON_MEAN) * 255.0, WINDOW_GREY[c])
    im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    h, w = a.shape[:2]
    small = im.resize((max(8, round(w * CANON_ROWS / h)), CANON_ROWS), Image.BOX)
    return small.resize((w, h), Image.BICUBIC)


def letterbox(image, side_px: int):
    """The image centred on a grey square, resized to side_px."""
    from PIL import Image

    side = max(image.size)
    square = Image.new("RGB", (side, side), WINDOW_GREY)
    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return square.resize((side_px, side_px), Image.BICUBIC)


def anchored_scores(x: np.ndarray, refs: np.ndarray, strangers: np.ndarray) -> np.ndarray:
    """100 * (x - strangers' mean) . d / |d|^2, d = refs' mean - strangers' mean; x one vector or rows of them."""
    mu = strangers.mean(0)
    d = refs.mean(0) - mu
    dd = float(d @ d)
    if dd <= 1e-12:
        return np.zeros(len(x)) if np.ndim(x) == 2 else np.float64(0.0)
    return 100.0 * ((x - mu) @ d) / dd


class AnchoredScorer:
    """The anchored body score (module docstring). set_references() embeds the references; the first base-model
    probe's body renders become the strangers through set_strangers(); score() is None until both are there."""

    def __init__(self, models_dir: str | Path = "models", device: str = "cuda:0", min_cover: float = WINDOW_MIN_COVER,
                 progress=None):
        self.pose = BodyScorer(models_dir, progress=progress, segment=True)  # landmarks + person mask
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.min_cover = min_cover
        self.progress = progress
        self.models: dict = {}
        self._clear()

    def _clear(self) -> None:
        self.refs: dict[str, np.ndarray] = {}
        self.ref_names: list[str] = []
        self.ref_count = 0
        self.strangers: dict[str, np.ndarray] = {}
        self.stranger_count = 0
        self.strangers_tried = False  # set_strangers ran (a later step-0 probe is a soup's or noise check's LoRA)
        # a typical reference scored against the others (the median, /100), known once the strangers are: the scale's
        # 100 is the references' mean, so this shows how far a photo of the person reaches on it
        self.ref_self_similarity: float | None = None

    @classmethod
    def with_embedder(cls, embed, window, min_cover: float = WINDOW_MIN_COVER) -> "AnchoredScorer":
        """A scorer around given functions, for tests: embed(window image) -> {backbone: unit vector} | None,
        window(path) -> (image, coverage) | None."""
        self = cls.__new__(cls)
        self.min_cover, self.models = min_cover, {}
        self._clear()
        self.embed, self.window = embed, window  # type: ignore[method-assign]
        return self

    @property
    def ready(self) -> bool:
        """References are set (the strangers may still be missing: see anchored)."""
        return self.ref_count > 0

    @property
    def anchored(self) -> bool:
        return self.stranger_count >= MIN_STRANGERS

    @property
    def needs_strangers(self) -> bool:
        """References are set and the strangers are still to come (from the first base-model probe)."""
        return self.ready and not self.anchored and not self.strangers_tried

    # ------------------------------------------------------------ models

    def _load(self) -> None:
        if self.models:
            return
        import transformers
        from transformers import AutoModel, SiglipVisionModel

        from .downloads import ensure_hf_snapshot

        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        sig = ensure_hf_snapshot(SIGLIP_MODEL, "SigLIP 2", self.progress, allow_patterns=MODEL_PATTERNS)
        dino = ensure_hf_snapshot(DINO_MODEL, "DINOv2", self.progress, allow_patterns=MODEL_PATTERNS)
        if self.progress:
            self.progress("Loading SigLIP 2 and DINOv2...", None)
        level = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()  # the image half of a two-tower checkpoint: the text weights unused
        try:
            self.models["siglip"] = SiglipVisionModel.from_pretrained(sig, dtype=dtype)
        finally:
            transformers.logging.set_verbosity(level)
        self.models["dino"] = AutoModel.from_pretrained(dino, dtype=dtype)
        for m in self.models.values():
            m.to(self.device).eval().requires_grad_(False)

    def _tensor(self, image, side: int, mean, std) -> torch.Tensor:
        arr = (np.asarray(letterbox(image, side), np.float32) / 255.0 - np.asarray(mean, np.float32)) / np.asarray(
            std, np.float32)
        model = self.models["dino"]
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())[None].to(self.device, model.dtype)

    @torch.no_grad()
    def embed(self, window) -> dict[str, np.ndarray] | None:
        """{backbone: unit vector} of a body window's canonical form, None for a window with no person in it."""
        canon = canonical_window(window)
        if canon is None:
            return None
        self._load()
        norm = torch.nn.functional.normalize
        v = self.models["siglip"](pixel_values=self._tensor(canon, 384, (0.5,) * 3, (0.5,) * 3)).pooler_output
        h = self.models["dino"](pixel_values=self._tensor(canon, 224, _IMAGENET_MEAN, _IMAGENET_STD)).last_hidden_state
        return {"siglip": norm(v.float(), dim=-1)[0].cpu().numpy(),
                "dino": norm(h[:, 0].float(), dim=-1)[0].cpu().numpy()}

    def window(self, path: str | Path):
        """(body window image, coverage) of an image (bodysim.body_window), or None without shoulders and hips."""
        im, pts, vis, mask = self.pose.pose_and_mask(path)
        return body_window(im, pts, vis, mask)

    def embed_path(self, path: str | Path, min_cover: float = WINDOW_MIN_SCORE_COVER):
        """(embeddings, coverage) of an image's body window; None without one, or with less than min_cover of it in
        the frame."""
        win = self.window(path)
        if win is None or win[1] < min_cover:
            return None
        emb = self.embed(win[0])
        return None if emb is None else (emb, float(win[1]))

    # ------------------------------------------------------------ references and strangers

    def set_references(self, paths: list[str | Path]) -> list[dict]:
        """Embed the body window of every reference that shows one. The photos whose window lies at least min_cover
        inside the frame are the references; with fewer than two such photos every photo with shoulders and hips
        counts, with a warning (bodysim.AppearanceScorer's rule)."""
        from .facesim import expand_refs

        found, info = [], []
        paths = expand_refs(paths)
        for p in paths:
            got = self.embed_path(p, 0.0)
            if got is None:
                info.append({"image": str(p), "person": False})
                continue
            emb, cover = got
            found.append((str(p), emb, cover))
            info.append({"image": str(p), "person": True, "cover": round(cover, 3)})
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
        self.ref_names = [p for p, _, _ in used]
        self.refs = {b: np.stack([e[b] for _, e, _ in used]) for b in BACKBONES} if used else {}
        self.ref_count = len(used)
        self._fit()
        log.info("Body references (anchored): %d of %d photos show shoulders and hips, %d of them the body down to "
                 "mid-thigh (the references); the scale's zero comes from the step-0 probe", len(found), len(paths),
                 self.ref_count)
        return info

    def set_strangers(self, paths: list[str | Path]) -> int:
        """The base model's body renders (the step-0 probe's): their mean is the scale's 0. Returns how many showed a
        scorable body window; with fewer than MIN_STRANGERS the score stays off."""
        embs = [got[0] for got in (self.embed_path(p) for p in paths) if got is not None]
        return self.set_stranger_embeddings(embs, len(paths))

    def set_stranger_embeddings(self, embs: list[dict], tried: int | None = None) -> int:
        """set_strangers() from embeddings already made (embed()); tried: the renders they came from, for the log."""
        tried = len(embs) if tried is None else tried
        self.strangers_tried = True
        if len(embs) < MIN_STRANGERS:
            log.warning("Body score (anchored): only %d of %d base-model body renders show the body from the neck to "
                        "mid-thigh, %d are needed; no body scores this run (a body probe prompt that frames the body "
                        "to mid-thigh or lower fixes it)", len(embs), tried, MIN_STRANGERS)
            return len(embs)
        self.strangers = {b: np.stack([e[b] for e in embs]) for b in BACKBONES}
        self.stranger_count = len(embs)
        self._fit()
        if self.stranger_count < 4:
            log.warning("Body score (anchored): %d base-model body renders set its zero; 4 or more steady it",
                        self.stranger_count)
        sim = self.ref_self_similarity
        log.info("Body score (anchored): zero = %d base-model body renders, 100 = the %d references' mean; a typical "
                 "reference scores %s against the others", self.stranger_count, self.ref_count,
                 "n/a" if sim is None else f"{sim * 100:.1f}")
        return self.stranger_count

    def _fit(self) -> None:
        """The references' typical score against the others (leave-one-out, the median), once both sides are set."""
        self.ref_self_similarity = None
        if not self.anchored or self.ref_count < 2:
            return
        loo = []
        for i in range(self.ref_count):
            keep = np.arange(self.ref_count) != i
            loo.append(np.mean([float(anchored_scores(self.refs[b][i], self.refs[b][keep], self.strangers[b]))
                                for b in BACKBONES]))
        self.ref_self_similarity = float(np.median(loo)) / 100.0

    def save_strangers(self, path: str | Path) -> None:
        if self.anchored:
            np.savez(path, **self.strangers)

    def load_strangers(self, path: str | Path) -> bool:
        """The strangers a run saved (save_strangers), for a resumed run that has no step-0 probe of its own."""
        try:
            with np.load(path) as z:
                embs = [{b: z[b][i] for b in BACKBONES} for i in range(len(z[BACKBONES[0]]))]
        except (OSError, KeyError, ValueError):
            return False
        return self.set_stranger_embeddings(embs) >= MIN_STRANGERS

    # ------------------------------------------------------------ scoring

    def score_embedding(self, emb: dict[str, np.ndarray]) -> float | None:
        if not (self.ready and self.anchored):
            return None
        return float(np.mean([anchored_scores(emb[b], self.refs[b], self.strangers[b]) for b in BACKBONES]))

    def score(self, path: str | Path) -> tuple[float | None, int]:
        """(anchored score | None, references compared). None before the strangers are set, without shoulders and
        hips, or when less than WINDOW_MIN_SCORE_COVER of the window is inside the frame."""
        if not (self.ready and self.anchored):
            return None, 0
        got = self.embed_path(path)
        if got is None:
            return None, 0
        return self.score_embedding(got[0]), self.ref_count
