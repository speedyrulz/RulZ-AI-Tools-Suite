"""Copy check (copy_check): how close a render comes to copying one of the training photos, measured with SSCD.

SSCD (Pizzi et al., CVPR 2022; facebookresearch/sscd-copy-detection) is a copy detector: a ResNet-50 trained so that
an image and an edited copy of it (cropped, recoloured, blurred, overlaid) embed close together while different
pictures of the same scene do not. Diffusion memorization work uses it to find renders that replicate a training
image (Somepalli et al. 2023, arXiv 2305.20086, read a cosine above 0.5 as a likely partial copy). A LoRA trained on
a few photos of one person is the setting where that happens: same pose, same outfit, same background.

A render's copy score is its best cosine against the training photos (each a 512-d descriptor of the whole picture,
L2-normalized). The photos' own typical best match among themselves (the median, as reference_ceiling for likeness)
is the baseline: a single photo shoot (one room, one outfit) already matches itself highly, and a render that looks
like one more photo from that shoot is no copy. The limit a self-training picture must stay under is
max(copy_max, the baseline + COPY_MARGIN).

Uses:
  - selftrain: a picture above the limit is left out ("copies a photo"): training on the LoRA's own near-copies of a
    photo teaches the copy, not the person in new scenes;
  - probes: every probe's highest copy score (and the photo it matches) is logged and kept in likeness.jsonl; it
    decides nothing.

Model: sscd_disc_mixup (TorchScript, about 99 MB, downloaded once to <models>/sscd/; MIT), 320 x 320 input with
ImageNet normalization, as the SSCD repo's evaluation. No source calibrates SSCD on DiT LoRA renders of people, so
the limit is the literature's partial-copy threshold, read against the photos' own baseline.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SIDE = 320  # SSCD's square input
COPY_MARGIN = 0.1  # the limit sits at least this far above the photos' own typical best match
_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def copy_limit(copy_max: float, baseline: float | None) -> float:
    """The copy score above which a picture copies a photo: copy_max, raised to the photos' own typical best match
    plus COPY_MARGIN when they already match each other that closely (one shoot, one room, one outfit)."""
    return copy_max if baseline is None else max(copy_max, baseline + COPY_MARGIN)


def load_image(path) -> np.ndarray:
    """[3, SIDE, SIDE] float32, normalized: the whole picture squeezed to a square (SSCD's 320 x 320 evaluation)."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB").resize((SIDE, SIDE), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return ((arr - np.asarray(_MEAN, dtype=np.float32)) / np.asarray(_STD, dtype=np.float32)).transpose(2, 0, 1)


class CopyChecker:
    """SSCD descriptors of the training photos; score(path) gives a render's best match among them."""

    def __init__(self, models_dir: str | Path = "models", device: str = "cpu", progress=None):
        self.models_dir = Path(models_dir)
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.progress = progress
        self.model = None
        self.photos: list[str] = []
        self.descs: np.ndarray | None = None
        self.baseline: float | None = None

    @classmethod
    def with_embedder(cls, embed) -> "CopyChecker":
        """A checker around a given function, for tests: embed(path) -> unit vector."""
        self = cls.__new__(cls)
        self.photos, self.descs, self.baseline = [], None, None
        self.embed = embed  # type: ignore[method-assign]
        return self

    def _load(self) -> None:
        if self.model is not None:
            return
        from .downloads import download_url

        path = self.models_dir / "sscd" / SSCD_URL.rsplit("/", 1)[-1]
        if not path.exists():
            download_url(SSCD_URL, path, "SSCD copy detector", self.progress)
        with warnings.catch_warnings():  # torch.jit.load is deprecated in favour of torch.export; SSCD ships TorchScript
            warnings.simplefilter("ignore", FutureWarning)
            self.model = torch.jit.load(str(path), map_location=self.device).eval()

    @torch.no_grad()
    def embed(self, path) -> np.ndarray:
        self._load()
        x = torch.from_numpy(load_image(path))[None].to(self.device)
        return torch.nn.functional.normalize(self.model(x).float(), dim=-1)[0].cpu().numpy()

    def set_photos(self, paths) -> None:
        """Describe the training photos (each once) and their typical best match among themselves (the baseline)."""
        from .likeness import reference_ceiling

        self.photos = list(dict.fromkeys(str(p) for p in paths))
        self.descs = np.stack([self.embed(p) for p in self.photos]) if self.photos else None
        self.baseline = reference_ceiling(self.descs) if self.descs is not None else None

    @property
    def ready(self) -> bool:
        return self.descs is not None and len(self.photos) > 0

    def score(self, path) -> tuple[float | None, str]:
        """(the render's best cosine against the photos, that photo), or (None, "") with no photos."""
        if not self.ready:
            return None, ""
        sims = self.descs @ self.embed(path)
        i = int(np.argmax(sims))
        return float(sims[i]), self.photos[i]

    def closest(self, paths) -> tuple[float | None, str, str]:
        """(the highest copy score among the renders, the photo it matches, that render), or (None, "", "")."""
        best: tuple[float | None, str, str] = (None, "", "")
        for p in paths:
            s, photo = self.score(p)
            if s is not None and (best[0] is None or s > best[0]):
                best = (s, photo, str(p))
        return best
