"""Text-prompted masks for Auto crop, with SAM 3.1 (Meta's Segment Anything 3.1, SAM License).

Type what to find ("face", "dog") and SAM 3.1 returns a mask of every match. The weights are Meta's
SAM 3.1 checkpoint in the Hugging Face Transformers format (huggingface.co/unsloth/sam3.1, pinned to one
revision, no sign-in needed). PyTorch, Transformers and the ~1.7 GB model are optional extras, set up from
the app on first use, like Best quality mode's FBCNN.
"""
from __future__ import annotations

import hashlib
import importlib.util
import threading
from pathlib import Path

import cv2
import numpy as np

from . import config, imageio

REPO = "unsloth/sam3.1"
REVISION = "cd3538905f98152ba184d2791fc5aa5d06e2ae45"
# (file, size in bytes, sha256 or "" for small files only checked by size)
FILES = [
    ("config.json", 6517, ""),
    ("processor_config.json", 613, ""),
    ("special_tokens_map.json", 588, ""),
    ("tokenizer_config.json", 799, ""),
    ("tokenizer.json", 3642073, ""),
    ("vocab.json", 862328, ""),
    ("merges.txt", 524619, ""),
    ("LICENSE", 7352, ""),
    ("model.safetensors", 1680935372, "28cd71c60e3034fc05ab67aa6b960d5340087fc72e456f2989b50294dd6b4dc9"),
]
DOWNLOAD_SIZE = sum(size for _, size, _ in FILES)
# SAM works at 1008 px, so bigger pictures are shrunk before masking; the mask is scaled back for the crop.
MASK_MAX_SIDE = 2048

READY, NO_TORCH, NO_LIBS, NO_MODEL = "ready", "no_torch", "no_libs", "no_model"
LIBS = ("transformers", "torchvision", "PIL")


def file_url(name: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"


def model_dir() -> Path:
    path = config.models_dir() / "sam3.1"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def torch_installed() -> bool:
    return _installed("torch")


def libs_installed() -> bool:
    return all(_installed(n) for n in LIBS)


def missing_files() -> list[tuple[str, int, str]]:
    d = model_dir()
    return [f for f in FILES if not ((d / f[0]).is_file() and (d / f[0]).stat().st_size == f[1])]


def status() -> str:
    if not torch_installed():
        return NO_TORCH
    if not libs_installed():
        return NO_LIBS
    if missing_files():
        return NO_MODEL
    return READY


def verify_download(path: Path, size: int, sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    if not sha256:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == sha256


# ---- model ----------------------------------------------------------------------
_model = None
_lock = threading.Lock()


def _load():
    global _model
    if _model is None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        dev = torch.device("cpu")
        if torch.cuda.is_available():  # newest architecture first, then most memory
            best = max(range(torch.cuda.device_count()), key=lambda i: (
                torch.cuda.get_device_capability(i), torch.cuda.get_device_properties(i).total_memory))
            dev = torch.device(f"cuda:{best}")
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        net = Sam3Model.from_pretrained(str(model_dir()), dtype=dtype).to(dev).eval()
        proc = Sam3Processor.from_pretrained(str(model_dir()))
        _model = (net, proc, dev, dtype)
    return _model


def device_name() -> str:
    import torch

    dev = _load()[2]
    return torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU (slow - no supported NVIDIA GPU found)"


def find(img: np.ndarray, prompt: str, confidence: float = 0.5, mask_threshold: float = 0.5,
         best_only: bool = False) -> tuple[np.ndarray, list[float]]:
    """Mask of what `prompt` describes in a BGR/BGRA/gray image (any depth).

    Returns (mask, scores): a float32 0/1 mask at MASK_MAX_SIDE or smaller, the union of every match whose
    score is at least `confidence` (only the highest-scoring one with best_only), and the scores kept.
    An empty mask means nothing was found.
    """
    import torch
    from PIL import Image

    small = imageio.to_8bit(imageio.shrink(img, MASK_MAX_SIDE))
    if small.ndim == 2 or small.shape[2] == 1:
        rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(small[:, :, :3], cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    with _lock, torch.inference_mode():
        net, proc, dev, dtype = _load()
        inputs = proc(images=Image.fromarray(rgb), text=prompt.strip(), return_tensors="pt").to(dev, dtype)
        out = net(**inputs)
        res = proc.post_process_instance_segmentation(
            out, threshold=confidence, mask_threshold=mask_threshold, target_sizes=[(h, w)])[0]
    scores = [float(s) for s in res["scores"]]
    masks = res["masks"]
    if best_only and scores:
        keep = int(np.argmax(scores))
        masks, scores = masks[keep:keep + 1], [scores[keep]]
    if not scores:
        return np.zeros((h, w), np.float32), []
    return masks.bool().any(dim=0).cpu().numpy().astype(np.float32), scores


def warm_up() -> None:
    """Load the model and run a tiny image, so the first preview doesn't pay the start-up cost."""
    try:
        find(np.zeros((64, 64, 3), np.uint8), "object")
    except Exception:  # noqa: BLE001 - the real call reports problems
        pass
