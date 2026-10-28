"""Direct identity loss for character LoRAs (experimental, off by default: id_loss_weight = 0).

On low-noise training samples with a detected face, the predicted clean latent x0 = noisy - sigma * pred is cropped
around the face (cached region map: face_cells + kps in bucket pixels), decoded with gradients, aligned to the
face-recognition template with a differentiable affine warp, embedded, and compared with the reference identity:

    id_loss = 1 - cos(embed(align(decode(crop(x0)))), ref)

Backends:
  arcface  InsightFace buffalo_l w600k_r50.onnx ported to torch with onnx2torch: the same network the face probe
           scores with, so the probe metric becomes the training target (it can be gamed; watch the samples)
  facenet  InceptionResnetV1 / VGGFace2 (rulztrainer.facenet): independent of the probe metric

Alignment: insightface.utils.face_align.estimate_norm fits a similarity transform from the 5 landmarks to the
ArcFace template and cv2.warpAffine applies it; `align_theta` turns the same transform into an affine_grid theta so
grid_sample reproduces the warp with gradients.
"""
from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger("rulztrainer")
CELL = 16  # bucket pixels per latent cell

# insightface.utils.face_align.arcface_dst (112 x 112): left eye, right eye, nose, left / right mouth corner
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float64)


# ---------------------------------------------------------------- geometry


def similarity_transform(src, dst) -> np.ndarray:
    """Least-squares similarity (rotation, uniform scale, translation; no reflection) mapping src -> dst points,
    as a 2x3 matrix. Same estimate as skimage's SimilarityTransform (Umeyama), which estimate_norm uses."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    R = U @ np.diag(d) @ Vt
    var = (sc ** 2).sum() / n
    scale = (S * d).sum() / var if var > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]])


def estimate_norm(kps, size: int = 112) -> np.ndarray:
    """2x3 matrix taking source pixels to the `size` x `size` aligned crop (ArcFace template scaled to size)."""
    return similarity_transform(kps, ARCFACE_DST * (size / 112.0))


def align_theta(M: np.ndarray, src_w: int, src_h: int, out_w: int, out_h: int | None = None) -> torch.Tensor:
    """affine_grid theta (align_corners=False) equal to cv2.warpAffine(img, M, (out_w, out_h)) on a src_w x src_h
    image. M maps source pixels to output pixels; grid_sample needs output -> source in normalised coordinates."""
    out_h = out_h or out_w
    M3 = np.vstack([np.asarray(M, np.float64), [0.0, 0.0, 1.0]])
    Minv = np.linalg.inv(M3)
    # output normalised (u, v) -> output pixel: x = (u + 1) * W / 2 - 0.5
    D = np.array([[out_w / 2.0, 0, (out_w - 1) / 2.0], [0, out_h / 2.0, (out_h - 1) / 2.0], [0, 0, 1]])
    # source pixel -> source normalised: u = (2x + 1) / W - 1
    N = np.array([[2.0 / src_w, 0, 1.0 / src_w - 1], [0, 2.0 / src_h, 1.0 / src_h - 1], [0, 0, 1]])
    return torch.from_numpy((N @ Minv @ D)[:2]).float()


def warp(images: torch.Tensor, thetas: torch.Tensor, size: int) -> torch.Tensor:
    """Differentiable affine warp of [B,C,H,W] images to [B,C,size,size] (zeros outside, like cv2's borderValue)."""
    grid = F.affine_grid(thetas.to(images.device, images.dtype), [images.shape[0], images.shape[1], size, size],
                         align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def face_crop_cells(face_cells, margin: int, h: int, w: int) -> tuple[int, int, int, int]:
    """Face cells dilated by `margin` cells on every side and clipped to the h x w latent: (x0, y0, x1, y1)."""
    cx0, cy0, cx1, cy1 = face_cells
    return max(0, cx0 - margin), max(0, cy0 - margin), min(w, cx1 + margin), min(h, cy1 + margin)


def predicted_x0(noisy: torch.Tensor, pred: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Flow matching: noisy = (1 - s) x0 + s noise and the target is noise - x0, so x0 = noisy - s * pred."""
    return noisy - sigma.view(-1, 1, 1, 1).to(noisy) * pred


# ---------------------------------------------------------------- embedders


class Embedder(nn.Module):
    """Frozen face network: RGB [B,3,size,size] in [-1, 1] -> L2-normalised embeddings. Gradients flow to the input."""

    def __init__(self, net: nn.Module, size: int, in_scale: float = 1.0, name: str = ""):
        super().__init__()
        self.net = net.eval().requires_grad_(False)
        self.size, self.in_scale, self.name = size, in_scale, name
        self.register_buffer("_anchor", torch.zeros(0), persistent=False)  # tracks .to(device)

    @property
    def device(self) -> torch.device:
        return self._anchor.device

    def train(self, mode: bool = True):  # always eval: batch norm must use its running stats
        return super().train(False)

    def forward(self, x):
        return F.normalize(self.net(x * self.in_scale).float(), dim=-1)


def arcface_embedder(pack_dir: str | Path | None = None) -> Embedder:
    """w600k_r50 from the buffalo_l pack, converted with onnx2torch. Input (rgb - 127.5) / 127.5 = [-1, 1]."""
    try:
        import onnx
        from onnx2torch import convert
    except ImportError as e:
        raise RuntimeError("the arcface identity loss needs 'onnx2torch' (pip install onnx2torch)") from e
    pack = Path(pack_dir) if pack_dir else Path(os.path.expanduser("~/.insightface/models/buffalo_l"))
    path = pack / "w600k_r50.onnx"
    if not path.exists():
        from .downloads import ensure_insightface_pack

        ensure_insightface_pack("buffalo_l")
    # convert from the loaded proto: given a path, onnx2torch writes a temp file next to the model and re-opens it
    # while still open, which fails on Windows
    return Embedder(convert(onnx.load(str(path))), 112, 1.0, "arcface")


def facenet_embedder(models_dir: str | Path = "models", progress=None) -> Embedder:
    """InceptionResnetV1 / VGGFace2 (160 px). Input (rgb - 127.5) / 128 = [-1, 1] * 127.5 / 128."""
    from .downloads import download_url
    from .facenet import VGGFACE2_URL, InceptionResnetV1

    path = Path(models_dir) / "facenet" / VGGFACE2_URL.rsplit("/", 1)[-1]
    if not path.exists():
        download_url(VGGFACE2_URL, path, "FaceNet VGGFace2 weights", progress)
    net = InceptionResnetV1()
    net.load_pretrained(path)
    return Embedder(net, 160, 127.5 / 128.0, "facenet")


def load_rgb(path) -> torch.Tensor:
    """[1,3,H,W] in [-1, 1] at full resolution (EXIF-rotated, matching FaceScorer.faces' coordinates)."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)[None] / 127.5 - 1.0


@torch.no_grad()
def embed_image(embedder: Embedder, image: torch.Tensor, kps) -> torch.Tensor:
    """Align one [1,3,H,W] image by its landmarks and embed it (reference faces)."""
    dev = embedder.device
    theta = align_theta(estimate_norm(kps, embedder.size), image.shape[-1], image.shape[-2],
                        embedder.size)
    return embedder(warp(image.to(dev), theta[None], embedder.size))[0]


def reference_embeddings(embedder: Embedder, face_scorer, check: bool = False) -> torch.Tensor:
    """[N, D] reference embeddings from FaceScorer.refs, re-embedded by `embedder` through the differentiable
    alignment. With check (arcface backend), each one must match the ONNX scorer's embedding (cosine > 0.99),
    which validates the onnx2torch port and the alignment together."""
    out = []
    for path, onnx_emb in face_scorer.refs:
        faces = face_scorer.faces(path)
        if not faces or getattr(faces[0], "kps", None) is None:
            continue
        emb = embed_image(embedder, load_rgb(path), faces[0].kps).float().cpu()
        if check:
            cos = float(emb @ torch.from_numpy(np.asarray(onnx_emb, np.float32)))
            if cos <= 0.99:
                raise RuntimeError(f"arcface port self-check failed on {Path(path).name}: cosine {cos:.4f} vs the "
                                   "ONNX scorer (expected > 0.99)")
            log.info("Identity loss: arcface port self-check on %s: cosine %.4f vs ONNX", Path(path).name, cos)
            check = False  # one reference is enough
        out.append(emb)
    if not out:
        raise RuntimeError("no reference face with landmarks for the identity loss")
    return torch.stack(out)


# ---------------------------------------------------------------- the loss


class IdentityLoss:
    """Selects eligible samples and computes the mean 1 - cos over them.

    decode: latents [1,C,h,w] -> images [1,3,16h,16w] in [-1, 1], with gradients (the arch's decode_latents_grad).
    refs: [N, D] L2-normalised reference embeddings."""

    def __init__(self, embedder: Embedder, refs: torch.Tensor, decode, sigma_max: float = 0.35, every: int = 1,
                 margin_cells: int = 2, min_face_cells: int = 4, random_ref: bool = True, seed: int = 0):
        self.embedder, self.decode = embedder, decode
        self.refs = F.normalize(refs.float(), dim=-1)
        self.centroid = F.normalize(self.refs.mean(0), dim=0)
        self.sigma_max, self.every = float(sigma_max), max(1, int(every))
        self.margin, self.min_cells = max(0, int(margin_cells)), max(1, int(min_face_cells))
        self.random_ref = random_ref
        self.rng = random.Random(seed)  # own RNG: never perturbs the training batch / noise draws
        self.n_eligible = 0

    def eligible(self, sigma: float, meta) -> bool:
        fc = getattr(meta, "face_cells", None)
        kps = getattr(meta, "kps", None)
        if sigma > self.sigma_max or fc is None or not kps or len(kps) != 5:
            return False
        return min(fc[2] - fc[0], fc[3] - fc[1]) >= self.min_cells

    def select(self, sigmas, metas) -> list[int]:
        """Indices of the samples to use this call: every `every`-th eligible sample (counted across calls)."""
        out = []
        for i, (s, m) in enumerate(zip(sigmas, metas)):
            if self.eligible(float(s), m):
                if self.n_eligible % self.every == 0:
                    out.append(i)
                self.n_eligible += 1
        return out

    def pick_ref(self) -> torch.Tensor:
        """One random reference per step (id_loss_random_ref) or the normalised mean of all references."""
        if self.random_ref and len(self.refs) > 1:
            return self.refs[self.rng.randrange(len(self.refs))]
        return self.centroid

    def __call__(self, noisy, pred, sigma, metas, ref: torch.Tensor | None = None):
        """Returns (mean id loss as a scalar tensor with gradients, number of samples) or (None, 0)."""
        idx = self.select(sigma.detach().float().cpu().tolist(), metas)
        if not idx:
            return None, 0
        ref = self.pick_ref() if ref is None else ref
        h, w = noisy.shape[-2:]
        size = self.embedder.size
        losses = []
        for i in idx:
            meta = metas[i]
            x0, y0, x1, y1 = face_crop_cells(meta.face_cells, self.margin, h, w)
            n_crop = noisy[i:i + 1, :, y0:y1, x0:x1].float()
            p_crop = pred[i:i + 1, :, y0:y1, x0:x1].to(n_crop)  # model split: the output may sit on another GPU
            x0_lat = predicted_x0(n_crop, p_crop, sigma[i:i + 1])
            img = self.decode(x0_lat)
            kps = np.asarray(meta.kps, np.float64) - np.array([x0 * CELL, y0 * CELL], np.float64)
            theta = align_theta(estimate_norm(kps, size), img.shape[-1], img.shape[-2], size)
            face = warp(img.to(self.embedder.device).float(), theta[None], size)
            emb = self.embedder(face)[0]
            losses.append(1.0 - (emb * ref.to(emb)).sum())
        return torch.stack(losses).mean(), len(idx)


def build(cfg, face_scorer, decode, device, progress=None) -> IdentityLoss:
    """The configured backend with its references (and, for arcface, the port self-check)."""
    if cfg.id_loss_backend == "facenet":
        emb = facenet_embedder(cfg.models_dir, progress)
    else:
        emb = arcface_embedder()
    emb = emb.to(device)
    refs = reference_embeddings(emb, face_scorer, check=cfg.id_loss_backend == "arcface")
    return IdentityLoss(emb, refs, decode, cfg.id_loss_sigma_max, cfg.id_loss_every, cfg.id_loss_margin_cells,
                        cfg.id_loss_min_face_cells, cfg.id_loss_random_ref, seed=cfg.seed)


def effective_weight(w: float) -> float:
    """The applied weight: capped at 0.5 so the identity term can never dominate the diffusion loss."""
    return max(0.0, min(float(w), 0.5)) if math.isfinite(float(w)) else 0.0
