"""AdaFace ViT-Base: the default held-out (gold) face recognizer on the probes (face_gold, likeness.py).

The network is CVLface's ViT "base" (github.com/mk-minchul/CVLface, MIT License, Copyright (c) Minchul Kim), rebuilt for
inference only (no dropout, drop-path or masking) after Fizgig's face_vit.py (Apache-2.0), so only the weights file is
downloaded and no code from the Hub runs: patch 8, width 512, 24 blocks, 16 heads, MLP x3 with ReLU6, a 512-d feature
head. Weights: minchul/cvlface_adaface_vit_base_webface4m (trained on WebFace4M; for non-commercial research use, as
InsightFace's ArcFace models are), pinned to a commit and checked by SHA-256.

Input: a 112 x 112 face aligned to the 5-point ArcFace template, RGB, (pixel - 127.5) / 127.5 = [-1, 1] (idloss.Embedder
with in_scale 1). The face and its mirror image are embedded and summed, as Fizgig scores it; Embedder normalises.

Why it is the default gold scorer (RulzTrainer 3.14.0): on 2,940 saved probe renders of 10 runs and two people's
photos, scored on the same InsightFace landmarks, it told the two people apart as well as ArcFace (d' 6.6 / 6.1 against
ArcFace's 6.4 / 5.7; FaceNet 4.9 / 2.2, its photo AUC on one person 0.93), ranked the probe steps more steadily than
FaceNet against the seed-to-seed noise in 8 of 10 runs, and kept 88% of a run's headshot gain on the small faces of the
body probes (FaceNet 71%). ArcFace still saw the trained renders apart from the base model's best (AUC 0.81 against
0.74), so it stays the main scorer.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

REPO = "minchul/cvlface_adaface_vit_base_webface4m"
REVISION = "c6fed2ff09ac0a29cccc72c09127ad7b4bd3fa9e"
FILE = "model.safetensors"
SHA256 = "5fafd6b7d599a3ede5fac5bd1d01ad05e9e93e89b39b7687d4a3bc93ff2aebc0"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}"
LOCAL_NAME = "adaface_vit_base_webface4m.safetensors"  # under models/adaface/
SIZE_MB = 460


class _Attn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        return self.proj((a @ v).transpose(1, 2).reshape(b, n, d))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.ReLU6(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn, self.mlp = _Attn(dim, heads), _Mlp(dim, dim * ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class AdaFaceViT(nn.Module):
    """[B, 3, 112, 112] in [-1, 1] -> [B, 512] (not normalised)."""

    def __init__(self, out_dim: int = 512, dim: int = 512, depth: int = 24, heads: int = 16, ratio: int = 3,
                 patch: int = 8, size: int = 112):
        super().__init__()
        n = (size // patch) ** 2
        self.n, self.dim = n, dim
        self.patch_embed = _PatchEmbed(dim, patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, n, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads, ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.feature = nn.Sequential(nn.Linear(dim * n, dim, bias=False), nn.BatchNorm1d(dim, eps=2e-5),
                                     nn.Linear(dim, out_dim, bias=False), nn.BatchNorm1d(out_dim, eps=2e-5))

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for b in self.blocks:
            x = b(x)
        return self.feature(self.norm(x).reshape(x.shape[0], self.n * self.dim))


class FlipSum(nn.Module):
    """The face and its mirror image, embeddings summed (the caller normalises)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x) + self.net(x.flip(3))


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: str | Path) -> AdaFaceViT:
    """The network with the weights file's tensors (CVLface's "model.net." keys), eval mode, CPU float32."""
    from safetensors.torch import load_file

    prefix = "model.net."
    state = {k[len(prefix):]: v for k, v in load_file(str(path)).items() if k.startswith(prefix)}
    net = AdaFaceViT()
    missing, unexpected = net.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"AdaFace weights don't fit the network (missing {missing[:4]}, unexpected {unexpected[:4]})")
    return net.eval()


def ensure_weights(models_dir: str | Path = "models", progress=None) -> Path:
    """models/adaface/<LOCAL_NAME>, downloaded once from the pinned commit; a file whose SHA-256 differs is refused
    (and a bad download removed)."""
    from .downloads import download_url

    path = Path(models_dir) / "adaface" / LOCAL_NAME
    if not path.exists():
        download_url(URL, path, f"AdaFace ViT face weights (about {SIZE_MB} MB)", progress)
        if sha256(path) != SHA256:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"the AdaFace weights downloaded from {REPO} do not match their pinned SHA-256")
    return path
