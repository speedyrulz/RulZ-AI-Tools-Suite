"""Likeness-filtered self-training (selftrain): partway through the steps (selftrain_at, a fraction of them) the LoRA
renders the subject over varied prompts, the pictures that pass a quality filter and look more like the person than
its probes do are kept, and they train beside the photos at a lower weight for the rest of the run.

It is reward-ranked fine-tuning (RAFT, Dong et al. 2023: sample, rank by a reward, fine-tune on the best) with the
likeness probes' score as the reward, done once or twice inside the run instead of as a stage of its own. It shows
the LoRA its own best pictures of the person in scenes the photos never had; it cannot teach the identity anything
the LoRA does not already draw on a good seed.

Training on its own pictures can amplify the model's own artifacts ("model collapse": in the self-consuming LoRA
chains of arXiv 2407.17493 quality fell visibly by the third round of retraining on outputs). The guards:

  - few rounds, each replacing the last round's pictures (they never pile up or train on each other's output);
  - a cap: at most selftrain_share of the real images (repeats counted), trained at selftrain_weight;
  - a quality filter before likeness: one face, detected confidently and not cut off by the frame; a picture far
    blurrier than the round's others (below SHARP_FACTOR x their median sharpness, at the training resolution, on
    the face when it is large enough there, since a render's soft background is not blur) is rejected: a failed
    render, not a clean one (renders are compared with renders, as photos carry grain and texture renders never
    have); near-duplicates of a kept picture are dropped;
  - a bar: a picture is kept only above both selftrain_min_likeness and the probes' current smoothed score, on the
    probes' own scale, so the LoRA studies only its better-than-usual pictures.

The scorer that picks the pictures is the probes' own, so a probe score that rises after a round may partly be the
scorer being fed what it likes: judge the samples by eye too.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DET_MIN = 0.7  # the face detector's confidence a kept picture's face needs
DUP_MAX = 0.95  # thumbnail correlation above which a picture repeats one already kept
SHARP_FACTOR = 0.4  # blurry: below this x the median sharpness of the round's pictures measured the same way
SHARP_MIN_GROUP = 3  # a median needs this many pictures (fewer: no blur check for them)
SHARP_SIDE = 512  # pictures are compared for sharpness at this size (square-equivalent: the training resolution)
FACE_MIN = 64  # faces at least this large there (long edge, px) are compared alone; smaller ones go with the picture
THUMB = 32  # side of the grayscale thumbnail the duplicate check compares
REASONS = ("no face", "several faces", "unsure face", "face cut off", "blurry", "below the bar", "duplicate",
           "over the cap")


@dataclass
class Candidate:
    path: str
    prompt: str
    likeness: float | None  # combined, on the probes' scale (percent); None: nothing to score
    faces: int = 0
    det: float = 0.0
    cut_off: bool = False
    sharp: float = 0.0  # of the face when it has one, else of the picture
    face_sharp: bool = False  # sharp was measured on the face
    thumb: np.ndarray | None = field(default=None, repr=False)
    verdict: str = ""  # "kept" or the reason it was left out


def round_steps(steps: int, at: str) -> list[int]:
    """The steps the rounds run at: each fraction of `steps` in `at` ("0.5" or "0.5,0.75"), inside the run."""
    out = set()
    for part in str(at or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        f = float(part)
        if not 0.0 < f < 1.0:
            raise ValueError(f"selftrain_at takes fractions of the steps between 0 and 1, got {part!r}")
        s = int(round(steps * f))
        if 0 < s < steps:
            out.add(s)
    return sorted(out)


def plan(prompts: list[str], n: int) -> list[str]:
    """n prompts, cycling through the list (so each gets about n / len(prompts) seeds)."""
    return [prompts[i % len(prompts)] for i in range(n)] if prompts else []


def seed(base: int, rnd: int, i: int) -> int:
    """A candidate's seed: apart from the probes' and samples' fixed seeds, new each round."""
    return base * 7919 + rnd * 1009 + i + 17


def _scale(size, side: int) -> float:
    """The factor that brings a picture of `size` to the area side x side (the training resolution)."""
    return side / max(1.0, (size[0] * size[1]) ** 0.5)


def _gray(path, side: int, box=None) -> np.ndarray:
    """The picture in grayscale at the training resolution (area side x side), or the padded face box in it."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        k = _scale(im.size, side)
        im = im.resize((max(3, round(im.width * k)), max(3, round(im.height * k))), Image.Resampling.BILINEAR)
        if box is not None:  # a face, padded by a quarter, at the size training sees it
            x0, y0, x1, y1 = (float(v) * k for v in box)
            pw, ph = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
            im = im.crop((max(0, int(x0 - pw)), max(0, int(y0 - ph)), min(im.width, int(x1 + pw)),
                          min(im.height, int(y1 + ph))))
        return np.asarray(im, dtype=np.float32) / 255.0


def sharpness(path, box=None, side: int = SHARP_SIDE) -> float:
    """Variance of the Laplacian at the training resolution (area side x side; photos and renders alike, so a
    high-resolution photo is not sharper only for its pixels): low for a blurry or smeared picture. With a face box,
    of the face alone (a portrait's soft background is not blur), else of the whole picture."""
    g = _gray(path, side, box)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def thumb(path) -> np.ndarray:
    """A zero-mean, unit-length THUMB x THUMB grayscale thumbnail: near-duplicates correlate near 1."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        g = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((THUMB, THUMB), Image.Resampling.BILINEAR),
                       dtype=np.float32).ravel()
    g = g - g.mean()
    n = float(np.linalg.norm(g))
    return g / n if n > 0 else g


def big_face(box, size, side: int = SHARP_SIDE) -> bool:
    """A face large enough at the training resolution to judge its sharpness alone (FACE_MIN px or more on its long
    edge there). size: the picture's (width, height)."""
    if box is None:
        return False
    return max(float(box[2]) - float(box[0]), float(box[3]) - float(box[1])) * _scale(size, side) >= FACE_MIN


def sharp_floor(cands: list) -> tuple[float, float]:
    """(faces, whole pictures): SHARP_FACTOR x the median sharpness of the round's candidates measured on the face /
    on the whole picture; 0 (no check) for a group of fewer than SHARP_MIN_GROUP."""
    def floor(vals):
        return SHARP_FACTOR * float(np.median(vals)) if len(vals) >= SHARP_MIN_GROUP else 0.0

    return (floor([c.sharp for c in cands if c.face_sharp]), floor([c.sharp for c in cands if not c.face_sharp]))


def cap_count(items, share: float) -> int:
    """The most pictures a round may add: share of the real images, each counted as often as it comes up per epoch
    (one item per image: the lowest training resolution's, so mixed resolutions do not count an image twice)."""
    seen: dict[str, int] = {}
    for it in items:
        if it.category == "image":
            seen[str(it.image)] = max(seen.get(str(it.image), 0), max(1, int(it.repeats or 1)))
    return int(share * sum(seen.values()))


def reject_reason(c: Candidate, *, need_face: bool, sharp_min, bar: float) -> str | None:
    """sharp_min: one floor, or (faces, whole pictures) as sharp_floor gives them."""
    if sharp_min is None:
        sharp_min = 0.0
    if c.faces > 1:
        return "several faces"
    if need_face and c.faces == 0:
        return "no face"
    if c.faces == 1 and c.det < DET_MIN:
        return "unsure face"
    if c.faces == 1 and c.cut_off:
        return "face cut off"
    floor = sharp_min if not isinstance(sharp_min, tuple) else sharp_min[0 if c.face_sharp else 1]
    if c.sharp < floor:
        return "blurry"
    if c.likeness is None or c.likeness < bar:
        return "below the bar"
    return None


def select(cands: list[Candidate], *, need_face: bool, bar: float, cap: int, sharp_min=None) \
        -> tuple[list[Candidate], Counter]:
    """The pictures a round keeps, best likeness first, and how many were left out for each reason. Every candidate
    gets its verdict. sharp_min: the blur floor (default: sharp_floor of these candidates)."""
    if sharp_min is None:
        sharp_min = sharp_floor(cands)
    reasons: Counter = Counter()
    passed = []
    for c in cands:
        why = reject_reason(c, need_face=need_face, sharp_min=sharp_min, bar=bar)
        if why:
            c.verdict = why
            reasons[why] += 1
        else:
            passed.append(c)
    kept: list[Candidate] = []
    for c in sorted(passed, key=lambda c: -c.likeness):
        if c.thumb is not None and any(k.thumb is not None and float(c.thumb @ k.thumb) > DUP_MAX for k in kept):
            c.verdict = "duplicate"
        elif len(kept) >= cap:
            c.verdict = "over the cap"
        else:
            c.verdict = "kept"
            kept.append(c)
            continue
        reasons[c.verdict] += 1
    return kept, reasons


def describe(reasons: Counter) -> str:
    return ", ".join(f"{reasons[r]} {r}" for r in REASONS if reasons.get(r)) or "none"


def record(path: Path, rnd: int, step: int, bar: float, cands: list[Candidate]) -> None:
    """Append the round to selftrain.json: every candidate's scores and verdict (the kept ones rebuild the round's
    images when a run resumes)."""
    import json

    data = []
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
    data = [r for r in data if r.get("round") != rnd]
    data.append({"round": rnd, "step": step, "bar": round(bar, 2), "candidates": [
        {"image": c.path, "prompt": c.prompt, "likeness": None if c.likeness is None else round(c.likeness, 2),
         "faces": c.faces, "det": round(c.det, 3), "cut_off": c.cut_off, "sharpness": round(c.sharp, 5),
         "sharpness_of": "face" if c.face_sharp else "picture",
         "verdict": c.verdict} for c in cands]})
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def last_kept(path: Path, before: int) -> tuple[int, list[tuple[str, str]]] | None:
    """(round, [(image, prompt)]) of the latest round recorded at or before step `before`, or None."""
    import json

    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    rounds = [r for r in data if int(r.get("step", 0)) <= before]
    if not rounds:
        return None
    r = max(rounds, key=lambda r: int(r["step"]))
    return int(r["round"]), [(c["image"], c["prompt"]) for c in r.get("candidates", [])
                             if c.get("verdict") == "kept" and Path(c["image"]).exists()]
