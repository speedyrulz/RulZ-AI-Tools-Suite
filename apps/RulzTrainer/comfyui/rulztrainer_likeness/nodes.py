"""ComfyUI nodes: RulzTrainer likeness scoring (references, score, export). The scoring itself is in core.py."""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from . import core
from rulztrainer import facesim

log = logging.getLogger("rulztrainer")
REPO_DIR: Path | None = None  # set by __init__: the RulzTrainer checkout (its models/ folder holds the pose model)
CATEGORY = "RulzTrainer/likeness"
NA = -1.0  # a FLOAT output with no score (no face found, no body references, ...)

PATHS_TIP = ("Image files or folders (every image directly inside), one per line. Paths as this ComfyUI sees them "
             "(a ComfyUI in WSL reads C:\\ai as /mnt/c/ai). Lines starting with # are skipped.")


# ---------------------------------------------------------------- helpers


def _lines(text: str) -> list[str]:
    return [ln.strip().strip('"') for ln in (text or "").splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _paths(text: str) -> list[Path]:
    return facesim.expand_refs(_lines(text)) if _lines(text) else []


def _temp_root() -> str | None:
    try:
        import folder_paths

        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        return folder_paths.get_temp_directory()
    except Exception:  # noqa: BLE001
        return None


def _save_batch(images, folder: str, prefix: str, label: str) -> list[tuple[str, Path]]:
    """A ComfyUI IMAGE batch ([B, H, W, C] floats 0..1) as PNG files the scorers read (the trainer scores PNGs too).
    Returns (label, path) per image."""
    from PIL import Image

    if images is None:
        return []
    arr = np.clip(np.round(images.detach().cpu().float().numpy() * 255.0), 0, 255).astype(np.uint8)
    out = []
    for i, a in enumerate(arr):
        p = Path(folder) / f"{prefix}_{i + 1:03d}.png"
        Image.fromarray(a[..., :3] if a.shape[-1] >= 3 else a[..., 0]).convert("RGB").save(p)
        out.append((f"{label} {i + 1}" if len(arr) > 1 else label, p))
    return out


def _files_signature(*texts) -> str:
    """Changes when a listed file or a folder's images change (for IS_CHANGED: paths typed in stay the same text)."""
    h = hashlib.sha256()
    for text in texts:
        for p in _paths(text or ""):
            try:
                st = p.stat()
                h.update(f"{p}|{st.st_mtime_ns}|{st.st_size}".encode())
            except OSError:
                h.update(str(p).encode())
    return h.hexdigest()


def _devices() -> list[str]:
    try:
        import torch

        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        n = 0
    # cuda:0 always listed, so a saved workflow stays valid on a machine without CUDA (the scorer then uses the CPU)
    return [f"cuda:{i}" for i in range(max(1, n))] + ["cpu"]


def _progress(total: int):
    try:
        import comfy.utils

        bar = comfy.utils.ProgressBar(max(1, total))
        return lambda i: bar.update_absolute(i, max(1, total))
    except Exception:  # noqa: BLE001
        return None


def _interrupted():
    try:
        import comfy.model_management

        return comfy.model_management.throw_exception_if_processing_interrupted
    except Exception:  # noqa: BLE001
        return None


def _f(v) -> float:
    return NA if v is None else float(v)


# ---------------------------------------------------------------- nodes


class RZLikenessReferences:
    """The identity to score against, as in the trainer: face references and body references, each from image files,
    folders or connected images. Without separate face / body references, the shared references are used for both."""

    @classmethod
    def INPUT_TYPES(cls):
        devices = _devices()
        return {
            "required": {
                "reference_paths": ("STRING", {"multiline": True, "default": "", "tooltip":
                                    "References for both the face and the body score. " + PATHS_TIP}),
                "face_paths": ("STRING", {"multiline": True, "default": "", "tooltip":
                               "Face references only (replace the shared ones for the face). " + PATHS_TIP}),
                "body_paths": ("STRING", {"multiline": True, "default": "", "tooltip":
                               "Body references only (replace the shared ones for the body). " + PATHS_TIP}),
                "body_score": (["appearance", "proportions"], {"default": "appearance", "tooltip":
                               "The trainer's body score. appearance (its default): a DINOv2 embedding of the body "
                               "from the neck to mid-thigh, cut out and turned upright. proportions: body ratios "
                               "from the pose skeleton (and SAM 3 silhouette widths with mediapipe+sam3)."}),
                "body_backend": (["mediapipe+sam3", "mediapipe"], {"default": "mediapipe+sam3", "tooltip":
                                 "Pose / mask backend for the proportions score (the appearance score always uses "
                                 "MediaPipe's own person mask, as the trainer does). SAM 3 is gated on Hugging "
                                 "Face: accept its license and log in, or it falls back to MediaPipe."}),
                "body_device": (devices, {"default": devices[0], "tooltip":
                                "Where DINOv2 (appearance) or SAM 3 (proportions) runs."}),
                "face_provider": (["cpu", "cuda"], {"default": "cpu", "tooltip":
                                  "InsightFace's onnxruntime provider (the trainer's default is cpu)."}),
                "face_det_size": ("INT", {"default": 640, "min": 160, "max": 2048, "step": 32, "tooltip":
                                  "Face detector input size (the trainer's default is 640)."}),
                "sam3_model": ("STRING", {"default": "facebook/sam3", "tooltip":
                               "SAM 3 repo id or local folder (proportions with mediapipe+sam3 only)."}),
            },
            "optional": {
                "reference_images": ("IMAGE", {"tooltip": "Reference images for both scores."}),
                "face_images": ("IMAGE", {"tooltip": "Face reference images."}),
                "body_images": ("IMAGE", {"tooltip": "Body reference images."}),
            },
        }

    RETURN_TYPES = ("RZ_LIKENESS_REFS", "STRING")
    RETURN_NAMES = ("references", "info")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    @classmethod
    def IS_CHANGED(cls, reference_paths="", face_paths="", body_paths="", **kw):
        return _files_signature(reference_paths, face_paths, body_paths)

    def build(self, reference_paths, face_paths, body_paths, body_score, body_backend, body_device, face_provider,
              face_det_size, sam3_model, reference_images=None, face_images=None, body_images=None):
        with tempfile.TemporaryDirectory(prefix="rz_refs_", dir=_temp_root()) as tmp:
            labels = {}

            def gather(text, images, prefix, label):
                items = _save_batch(images, tmp, prefix, label)
                labels.update({str(p): name for name, p in items})
                return [p for _, p in items] + _paths(text)

            shared = gather(reference_paths, reference_images, "ref", "reference image")
            face = gather(face_paths, face_images, "face", "face reference image")
            body = gather(body_paths, body_images, "body", "body reference image")
            face, body = face or shared, body or shared
            if not face and not body:
                raise ValueError("No references: connect reference images or type image files / folders")
            refs = core.build_references(face, body, face_provider=face_provider, det_size=face_det_size,
                                         body_kind=body_score, body_backend=body_backend, body_device=body_device,
                                         sam3_model=sam3_model, models_dir=str(REPO_DIR / "models"), labels=labels)
        if refs.face is None and refs.body is None:
            raise ValueError("No usable reference: no face and no body was found in any of them\n" + refs.describe())
        info = refs.describe()
        log.info("RulzTrainer likeness references:\n%s", info)
        return (refs, info)


class RZLikenessScore:
    """Scores images against the references exactly as the trainer scores its probes and samples, and averages them:
    overall, face and body on the trainer's normalized scale (0 = a stranger, 100 = a typical reference), plus each
    image's own scores. Connect images, or type image files / folders, or both."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "references": ("RZ_LIKENESS_REFS",),
                "image_paths": ("STRING", {"multiline": True, "default": "", "tooltip":
                                "Images to score (with any connected images). " + PATHS_TIP}),
                "face_weight": ("FLOAT", {"default": core.FACE_WEIGHT, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Weight of the face in the overall score (trainer: 0.7)."}),
                "body_weight": ("FLOAT", {"default": core.BODY_WEIGHT, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Weight of the body in the overall score (trainer: 0.3)."}),
                "normalize": ("BOOLEAN", {"default": True, "tooltip":
                              "The trainer's normalized scale (its default): 0 = what a stranger scores, 100 = what "
                              "a typical reference scores against the others. Off: raw similarity percentages."}),
                "baseline_paths": ("STRING", {"multiline": True, "default": "", "tooltip":
                                   "Optional: images from the base model without the LoRA, same prompts. Their mean "
                                   "score becomes the 0 of the scale, as the trainer's step-0 probe does (else a "
                                   "default stranger score). " + PATHS_TIP}),
                "trainer_run": ("STRING", {"default": "", "tooltip":
                                "Optional: a RulzTrainer job folder: use the exact scale that run used (its "
                                "likeness.jsonl), for numbers that match its probes. Weights stay the ones set here."}),
                "face_floor": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1, "tooltip":
                               "Raw face score that counts as 0 (-1: from the run, the baseline, or the default)."}),
                "body_floor": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1, "tooltip":
                               "Raw body score that counts as 0 (-1: from the run, the baseline, or the default)."}),
                "face_ceiling": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1, "tooltip":
                                 "Raw face score that counts as 100 (-1: from the run, the references, or the "
                                 "default)."}),
                "body_ceiling": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.1, "tooltip":
                                 "Raw body score that counts as 100 (-1: from the run, the references, or the "
                                 "default)."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Images to score (a batch scores every image)."}),
                "baseline_images": ("IMAGE", {"tooltip": "Optional base-model images: the 0 of the scale."}),
            },
        }

    RETURN_TYPES = ("RZ_LIKENESS_REPORT", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("report", "summary", "details", "overall", "face", "body")
    OUTPUT_TOOLTIPS = ("Everything, for the export node.", "The averages.", "Each image's scores.",
                       "Average overall score (-1: none).", "Average face score (-1: no face found).",
                       "Average body score (-1: no usable body).")
    FUNCTION = "score"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    @classmethod
    def IS_CHANGED(cls, image_paths="", baseline_paths="", trainer_run="", **kw):
        run = Path(str(trainer_run).strip().strip('"')) / "likeness.jsonl" if str(trainer_run).strip() else None
        stamp = run.stat().st_mtime_ns if run is not None and run.is_file() else 0
        return _files_signature(image_paths, baseline_paths) + f"|{stamp}"

    def score(self, references, image_paths, face_weight, body_weight, normalize, baseline_paths, trainer_run,
              face_floor, body_floor, face_ceiling, body_ceiling, images=None, baseline_images=None):
        refs = references
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="rz_score_", dir=_temp_root()) as tmp:
            items = _save_batch(images, tmp, "image", "image") + [(p.name, p) for p in _paths(image_paths)]
            base = (_save_batch(baseline_images, tmp, "baseline", "baseline image")
                    + [(p.name, p) for p in _paths(baseline_paths)])
            if not items:
                raise ValueError("Nothing to score: connect images or type image files / folders")
            progress = _progress(len(items) + len(base))
            rows = core.score_rows(refs, items, progress, _interrupted())
            base_rows = core.score_rows(refs, base, (lambda i: progress(len(items) + i)) if progress else None,
                                        _interrupted()) if base else []
            for r in rows + base_rows:  # a connected image has no file of its own (its temporary copy is gone)
                if r["path"].startswith(str(tmp)):
                    r["path"] = "(connected image)"
        job = core.job_scale(trainer_run) if str(trainer_run).strip() else None
        scale = core.build_scale(refs, normalize, base_rows, job, face_floor, body_floor, face_ceiling, body_ceiling)
        core.apply_scale(rows, scale, face_weight, body_weight)
        summary = core.summarize(rows, scale, face_weight, body_weight)
        text = core.summary_text(summary, refs)
        details = core.details_text(rows)
        log.info("RulzTrainer likeness: %s | %s (%.1fs)\n%s", text.replace("\n", " | "), scale.describe(),
                 time.time() - t0, details)
        report = {"rows": rows, "summary": summary, "scale": scale, "refs": refs, "baseline": base_rows}
        return (report, text, details, _f(summary["overall"]), _f(summary["face"]), _f(summary["body"]))


class RZLikenessExport:
    """Writes the scores to a spreadsheet in ComfyUI's output folder when switched on: an .xlsx with Summary, Images
    and References sheets (normalized and raw scores, the scale used), or CSV files."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "report": ("RZ_LIKENESS_REPORT",),
                "export_spreadsheet": ("BOOLEAN", {"default": False, "label_on": "export", "label_off": "off"}),
                "format": (["xlsx", "csv"], {"default": "xlsx"}),
                "filename_prefix": ("STRING", {"default": "likeness/likeness", "tooltip":
                                    "Inside ComfyUI's output folder; a counter is added."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("files",)
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def export(self, report, export_spreadsheet, format, filename_prefix):
        if not export_spreadsheet:
            return {"ui": {"text": ["Spreadsheet export is off"]}, "result": ("",)}
        import folder_paths

        out_dir = folder_paths.get_output_directory()
        folder, name, counter, _, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        base = Path(folder) / f"{name}_{counter:05}"
        files = core.export(report["rows"], report["summary"], report["scale"], report["refs"], base, format)
        text = "\n".join(str(f) for f in files)
        log.info("RulzTrainer likeness: spreadsheet written to %s", text.replace("\n", ", "))
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "RZLikenessReferences": RZLikenessReferences,
    "RZLikenessScore": RZLikenessScore,
    "RZLikenessExport": RZLikenessExport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RZLikenessReferences": "RulzTrainer Likeness References",
    "RZLikenessScore": "RulzTrainer Likeness Score",
    "RZLikenessExport": "RulzTrainer Likeness Export",
}
