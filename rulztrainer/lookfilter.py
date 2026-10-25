"""Look consistency filter: finds dataset images whose face drifts from the look you want (after Fizgig's).

Pick a few baseline images that nail the look (3 is a good number: one photo would bake its own angle and light into
every score). Every image's largest face is embedded with the trainer's face model (InsightFace ArcFace, as the
likeness probes) and scored by its mean cosine similarity to the baselines (the same as the similarity to their
centroid, so no one photo's framing dominates). ArcFace conventions: the same person across varied photos lands about
0.30 to 0.70; a different person rarely clears 0.25.

The outlier fence is the median minus 1.5 x the interquartile range of the scored faces, never below 0.25 (it needs at
least 4 faces). Images under it are suggested; baselines and images without a face never are. Marked images move to
<dataset>/excluded_by_look/ with their captions, keeping their sub-folders, so they can be moved back; nothing is
deleted, and training skips that folder.

The scores are saved to <dataset>/look_scores.json. With look_warmup on, training eases the outliers that stayed in
(real but unusual shots: tight angles, profiles) in: their loss weight starts at look_warmup_start and ramps to full
over look_warmup_epochs epochs, while the identity forms from the typical images. Fizgig's own fizgig_look_scores.json
is read too."""
from __future__ import annotations

import json
import logging
import shutil
import statistics
from pathlib import Path

log = logging.getLogger("rulztrainer")

SCORES_FILE = "look_scores.json"
FIZGIG_FILE = "fizgig_look_scores.json"
EXCLUDED_DIR = "excluded_by_look"
MATCH, BORDERLINE, FENCE_FLOOR = 0.45, 0.30, 0.25
MIN_FACES = 4  # the fence needs this many scored faces
CAPTION_EXTS = (".txt", ".json")


def verdict(score) -> str:
    """match (solid), borderline (same person, drifting), drift (likely off-look or someone else) or no face."""
    if score is None:
        return "no face"
    return "match" if score >= MATCH else "borderline" if score >= BORDERLINE else "drift"


def score_images(files, baselines, embed, progress=None) -> dict[str, float | None]:
    """{image: mean cosine similarity to the baselines' faces, or None without a face}. embed(path) -> unit vector or
    None. Raises ValueError naming any baseline without a face."""
    import numpy as np

    base = [embed(b) for b in baselines]
    missing = [Path(b).name for b, e in zip(baselines, base) if e is None]
    if missing:
        raise ValueError(f"No face found in baseline {', '.join(missing)}: pick images with a clear face")
    if not base:
        raise ValueError("Pick at least one baseline image")
    out: dict[str, float | None] = {}
    files = list(files)
    for i, p in enumerate(files, 1):
        emb = embed(p)
        out[str(p)] = None if emb is None else float(np.mean([float(np.dot(b, emb)) for b in base]))
        if progress:
            progress(i, len(files))
    return out


def fence(scores) -> tuple[float | None, float | None]:
    """(median, cutoff) of the scored faces: cutoff = median - 1.5 x IQR, at least FENCE_FLOOR; (None, None) under
    MIN_FACES faces."""
    vals = sorted(v for v in (scores.values() if isinstance(scores, dict) else scores) if v is not None)
    if len(vals) < MIN_FACES:
        return None, None
    q1, med, q3 = statistics.quantiles(vals, n=4, method="inclusive")
    return med, max(med - 1.5 * (q3 - q1), FENCE_FLOOR)


def suggest(scores: dict, baselines=()) -> set[str]:
    """The images under the fence; never a baseline, never one without a face."""
    _, cut = fence(scores)
    if cut is None:
        return set()
    keep = {str(b) for b in baselines}
    return {p for p, s in scores.items() if s is not None and s < cut and p not in keep}


def _rel(folder: Path, p) -> str:
    try:
        return Path(p).resolve().relative_to(folder.resolve()).as_posix()
    except ValueError:
        return Path(p).name


def save_scores(folder, baselines, scores) -> Path:
    """<folder>/look_scores.json: the baselines, the fence and every image's score, keyed by its path inside the
    dataset folder (the file travels with the dataset)."""
    folder = Path(folder)
    med, cut = fence(scores)
    payload = {"baselines": [_rel(folder, b) for b in baselines], "median": med, "cutoff": cut,
               "scores": {_rel(folder, p): s for p, s in scores.items()}}
    path = folder / SCORES_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_scores(folder) -> dict | None:
    """The saved scan of a dataset folder as {"baselines": [paths], "cutoff", "median", "scores": {path: score}} with
    absolute paths, or None. Reads Fizgig's file (scores keyed by file stem) when there is no RulzTrainer one."""
    folder = Path(folder)
    for name in (SCORES_FILE, FIZGIG_FILE):
        p = folder / name
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("Could not read %s", p)
            continue
        scores = data.get("scores") or {}
        if name == FIZGIG_FILE:  # keyed by stem: match the images in the folder itself
            from .dataset import IMAGE_EXTS

            stems = {q.stem: q for q in folder.iterdir() if q.is_file() and q.suffix.lower() in IMAGE_EXTS}
            scores = {str(stems[k]): v for k, v in scores.items() if k in stems}
            base = [str(folder / b) for b in data.get("baselines") or []]
        else:
            scores = {str(folder / k): v for k, v in scores.items()}
            base = [str(folder / b) for b in data.get("baselines") or []]
        return {"baselines": base, "cutoff": data.get("cutoff"), "median": data.get("median"), "scores": scores,
                "file": str(p)}
    return None


def outliers(folder) -> set[str]:
    """Resolved paths of the images of a dataset folder that its saved scan puts under the fence (and that are still
    there): the ones the look warm-up eases in."""
    data = load_scores(folder)
    if not data or data.get("cutoff") is None:
        return set()
    cut = float(data["cutoff"])
    return {str(Path(p).resolve()) for p, s in data["scores"].items()
            if isinstance(s, (int, float)) and s < cut and Path(p).is_file()}


def _move(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    target, n = dest, 2
    while target.exists():
        target = dest.with_name(f"{dest.stem}_{n}{dest.suffix}")
        n += 1
    shutil.move(str(src), str(target))
    return target


def move_out(folder, paths) -> tuple[list[str], list[str]]:
    """Move images and their captions into <folder>/excluded_by_look/ (same sub-folders); nothing is deleted.
    Returns (moved images, errors)."""
    folder = Path(folder)
    moved, errors = [], []
    for raw in paths:
        p = Path(raw)
        if not p.is_file():
            continue
        try:
            rel = p.resolve().relative_to(folder.resolve())
        except ValueError:
            errors.append(f"{p.name}: not inside {folder}")
            continue
        if rel.parts and rel.parts[0] == EXCLUDED_DIR:
            continue
        try:
            target = _move(p, folder / EXCLUDED_DIR / rel)
            for ext in CAPTION_EXTS:
                cap = p.with_suffix(ext)
                if cap.is_file():
                    _move(cap, target.with_suffix(ext))
            moved.append(str(p))
        except OSError as e:
            errors.append(f"{p.name}: {e}")
    return moved, errors


def excluded_images(folder) -> list[Path]:
    from .dataset import IMAGE_EXTS

    d = Path(folder) / EXCLUDED_DIR
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS) if d.is_dir() else []


def move_back(folder) -> tuple[list[str], list[str]]:
    """Move everything in <folder>/excluded_by_look/ back where it was (a file of the same name that appeared since
    gets a numbered copy beside it). Returns (moved images, errors)."""
    folder = Path(folder)
    root = folder / EXCLUDED_DIR
    moved, errors = [], []
    for p in excluded_images(folder):
        rel = p.relative_to(root)
        try:
            target = _move(p, folder / rel)
            for ext in CAPTION_EXTS:
                cap = p.with_suffix(ext)
                if cap.is_file():
                    _move(cap, target.with_suffix(ext))
            moved.append(str(target))
        except OSError as e:
            errors.append(f"{p.name}: {e}")
    for d in sorted((d for d in root.rglob("*") if d.is_dir()), reverse=True) if root.is_dir() else []:
        try:
            d.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass
    return moved, errors


def warmup_factor(epoch: float, start: float, epochs: float) -> float:
    """The loss weight of a look outlier after `epoch` epochs: start, rising linearly to 1 at `epochs`."""
    if epochs <= 0:
        return 1.0
    return start + (1.0 - start) * min(1.0, max(0.0, epoch) / epochs)


# ---------------------------------------------------------------- the face embedding

_SCORERS: dict = {}


def face_embedder(provider: str = "cpu", det_size: int = 640, progress=None):
    """embed(path) -> the largest face's ArcFace embedding (unit vector) or None: the trainer's face model, loaded
    once per process."""
    from . import facesim

    key = (provider, int(det_size))
    if key not in _SCORERS:
        _SCORERS[key] = facesim.FaceScorer(provider, int(det_size), progress=progress)
    scorer = _SCORERS[key]

    def embed(path):
        fs = scorer.faces(path)
        return fs[0].normed_embedding.astype("float32") if fs else None

    return embed
