"""Likeness scoring as the RulzTrainer trainer does it, for the ComfyUI nodes (no ComfyUI imports here).

Everything that decides a score is the trainer's own code (rulztrainer.facesim, rulztrainer.bodysim,
rulztrainer.likeness): the same face model (InsightFace buffalo_l, the largest face, the best cosine against any
reference), the same body score (the default: a DINOv2 embedding of the body window, neck to mid-thigh; or the
proportions score with MediaPipe, optionally with SAM 3 silhouettes), the same normalized scale (0 = a stranger,
100 = 1.35x as far above a stranger as a typical reference's best match among the others, curved by 1.5, capped at
100) and the same face / body weighting.

The held-out gold face (the trainer's face_gold: AdaFace ViT, or FaceNet VGGFace2, on the same detected face, against its own
embeddings of the references, on its own scale shaped like the face's) is scored too when the references have it. With
the trainer's face_gold_select it takes ArcFace's place in the overall score; the overall with ArcFace's face is kept
next to it (the trainer's arc_combined_ema).

Averages follow the trainer's probes: the raw face scores of the images with a face are averaged, likewise the body
(and gold face) scores, then each average is normalized and they are combined into the overall score. Each image also
gets its own normalized face, body, gold face and overall score."""
from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from rulztrainer import bodysim, facesim
from rulztrainer.likeness import LikenessController, combine, MIN_SPAN, normalize as normalize_score

log = logging.getLogger("rulztrainer")

FACE_WEIGHT, BODY_WEIGHT = 0.7, 0.3  # the trainer's defaults (face_weight / body_weight)
_MODELS: dict[tuple, object] = {}  # loaded scorer models, shared by every node (the references live in RefSet)
_STATE = ("refs", "profile", "ref_count", "ref_self_similarity", "ref_kps", "ref_cut",  # a scorer's per-reference
          "gold", "gold_refs", "gold_self_similarity")                                   # state
KINDS = ("face", "body", "gold")


# ---------------------------------------------------------------- models


def face_model(provider: str = "cpu", det_size: int = 640, model_name: str = "buffalo_l"):
    """The face scorer for an InsightFace pack (the trainer's face_model: buffalo_l or antelopev2), kept for reuse."""
    model_name = model_name or "buffalo_l"
    key = ("face", provider, int(det_size), model_name)
    if key not in _MODELS:
        _MODELS[key] = facesim.FaceScorer(provider, int(det_size), model_name=model_name)
    return _MODELS[key]


def gold_model(models_dir: str, provider: str = "cpu", backend: str = "adaface"):
    """The trainer's held-out gold face recognizer (AdaFace ViT or FaceNet VGGFace2, idloss.gold_embedder), on the face
    scorer's GPU as the trainer places it (the CPU with the cpu provider)."""
    import torch

    index = facesim.cuda_index(provider)
    device = f"cuda:{index}" if index is not None and torch.cuda.is_available() else "cpu"
    key = ("gold", backend, str(models_dir), device)
    if key not in _MODELS:
        from rulztrainer import idloss

        _MODELS[key] = idloss.gold_embedder(backend, models_dir).to(device)
    return _MODELS[key]


def gold_label(backend) -> str:
    return {"adaface": "AdaFace"}.get(str(backend or "").lower(), "FaceNet")


def _setup_gold(scorer, models_dir: str, provider: str, backend: str = "adaface") -> list[str]:
    """The gold recognizer on the face scorer, after its references (as the trainer's _setup_gold). Returns notes."""
    name = gold_label(backend)
    try:
        scorer.set_gold(gold_model(models_dir, provider, backend))
    except Exception as e:  # noqa: BLE001
        log.warning("Could not load the gold face scorer (%s); the faces are scored with ArcFace alone", e)
        scorer.gold = scorer.gold_refs = scorer.gold_self_similarity = None
        return [f"Gold face ({name}): could not load ({e}); no gold face score"]
    if scorer.gold_refs is None:
        scorer.gold = scorer.gold_self_similarity = None
        return [f"Gold face ({name}): no reference face with landmarks; no gold face score"]
    return []


def body_model(kind: str, models_dir: str, device: str = "cuda:0", backend: str = "mediapipe+sam3",
               sam3_model: str = "facebook/sam3"):
    """The trainer's body scorer: appearance (the default) or proportions (MediaPipe, + SAM 3 with mediapipe+sam3,
    falling back to MediaPipe alone when SAM 3 cannot load, as the trainer does)."""
    if kind == "appearance":
        key = ("appearance", device)
        if key not in _MODELS:
            _MODELS[key] = bodysim.AppearanceScorer(models_dir, device)
        return _MODELS[key]
    key = ("proportions", backend, device, sam3_model)
    if key not in _MODELS:
        sil = None
        if backend == "mediapipe+sam3":
            try:
                sil = bodysim.SilhouetteScorer(sam3_model, device)
            except Exception as e:  # noqa: BLE001
                log.warning("SAM 3 unavailable (%s); body proportions use MediaPipe only. For facebook/sam3 accept "
                            "the license on huggingface.co and log in with 'hf auth login'.", e)
        _MODELS[key] = bodysim.BodyScorer(models_dir, silhouette=sil)
    return _MODELS[key]


def _snapshot(scorer) -> dict:
    return {k: getattr(scorer, k) for k in _STATE if hasattr(scorer, k)}


def _restore(scorer, state: dict) -> None:
    for k, v in state.items():
        setattr(scorer, k, v)


# ---------------------------------------------------------------- references


@dataclass
class RefSet:
    """Face and body references with their scorers (each kind None when it has no usable reference)."""
    face: object | None = None
    face_state: dict = field(default_factory=dict)
    face_info: list = field(default_factory=list)
    body: object | None = None
    body_state: dict = field(default_factory=dict)
    body_info: list = field(default_factory=list)
    body_kind: str = "appearance"
    labels: dict = field(default_factory=dict)  # file path -> the name shown for it (images given as a batch)
    notes: list = field(default_factory=list)
    gold_name: str = ""  # the gold recognizer: adaface / facenet
    face_model: str = "buffalo_l"  # the InsightFace pack of the face score (the trainer's face_model)

    @property
    def face_ceiling(self):
        """The references' typical best match among themselves in raw percent (None: fewer than two usable)."""
        sim = self.face_state.get("ref_self_similarity")
        return None if sim is None else float(sim) * 100.0

    @property
    def body_ceiling(self):
        sim = self.body_state.get("ref_self_similarity")
        return None if sim is None else float(sim) * 100.0

    @property
    def has_gold(self) -> bool:
        """The face scorer has the held-out gold recognizer, with its references embedded."""
        return (self.face is not None and self.face_state.get("gold") is not None
                and self.face_state.get("gold_refs") is not None)

    @property
    def gold_ceiling(self):
        sim = self.face_state.get("gold_self_similarity") if self.has_gold else None
        return None if sim is None else float(sim) * 100.0

    @property
    def gold_count(self) -> int:
        return len(self.face_state.get("gold_refs")) if self.has_gold else 0

    @property
    def face_count(self) -> int:
        return len(self.face_state.get("refs") or [])

    @property
    def body_count(self) -> int:
        return int(self.body_state.get("ref_count") or 0)

    def label(self, path) -> str:
        return self.labels.get(str(path), str(path)) if path is not None else ""

    def describe(self) -> str:
        lines = []
        if self.face is not None:
            lines.append(f"Face references: {self.face_count} of {len(self.face_info)} have a face")
        else:
            lines.append("Face references: none (no face score)")
        if self.has_gold:
            lines.append(f"Gold face ({gold_label(self.gold_name)}, held out): {self.gold_count} reference face"
                         f"{'s' if self.gold_count != 1 else ''}")
        if self.body is not None:
            what = "show the body window" if self.body_kind == "appearance" else "have a pose"
            lines.append(f"Body references ({self.body_kind}): {self.body_count} of {len(self.body_info)} {what}")
        else:
            lines.append("Body references: none (no body score)")
        return "\n".join(lines + self.notes)


def build_references(face_paths, body_paths, *, face_provider="cpu", det_size=640, body_kind="appearance",
                     body_backend="mediapipe+sam3", body_device="cuda:0", sam3_model="facebook/sam3",
                     models_dir="models", labels=None, gold="adaface", face_model_name="buffalo_l") -> RefSet:
    """Embed the references as the trainer does (set_references, then the gold recognizer's set_gold). face_paths /
    body_paths: image files or folders; an empty list leaves that kind out. gold: "adaface" or "facenet" (the held-out
    gold face too) or "off". face_model_name: the face score's InsightFace pack (buffalo_l or antelopev2)."""
    refs = RefSet(body_kind=body_kind, labels=dict(labels or {}), face_model=face_model_name or "buffalo_l")
    if face_paths:
        scorer = face_model(face_provider, det_size, refs.face_model)
        scorer.gold = scorer.gold_refs = scorer.gold_self_similarity = None  # the shared scorer: no earlier gold
        refs.face_info = scorer.set_references(face_paths)
        if scorer.refs and gold in ("adaface", "facenet"):
            refs.gold_name = gold
            refs.notes += _setup_gold(scorer, models_dir, face_provider, gold)
        if scorer.refs:
            refs.face, refs.face_state = scorer, _snapshot(scorer)
        else:
            refs.notes.append("No face found in any face reference: face scoring is off")
    if body_paths:
        scorer = body_model(body_kind, models_dir, body_device, body_backend, sam3_model)
        refs.body_info = scorer.set_references(body_paths)
        if scorer.ready:
            refs.body, refs.body_state = scorer, _snapshot(scorer)
        else:
            refs.notes.append(f"No usable {'person' if body_kind == 'appearance' else 'pose'} in any body reference: "
                              "body scoring is off")
    return refs


# ---------------------------------------------------------------- the scale


@dataclass
class Scale:
    """The normalized scale: per kind (floor, ceiling) in raw percent, where each came from, and whether it is on.
    gold: None without the gold face."""
    normalize: bool
    face: tuple
    body: tuple
    sources: dict
    curve: dict = field(default_factory=lambda: {"face": 1.0, "body": 1.0, "gold": 1.0})
    gold: tuple | None = None

    def kinds(self) -> tuple[str, ...]:
        return KINDS if self.gold is not None else KINDS[:2]

    def norm(self, kind: str, value):
        if value is None or not self.normalize:
            return value
        if getattr(self, kind) is None:
            return None
        return normalize_score(value, *getattr(self, kind), self.curve.get(kind, 1.0))

    def describe(self) -> str:
        if not self.normalize:
            return "Normalization off: the scores are the raw similarity percentages"
        parts = []
        for k in self.kinds():
            lo, hi = getattr(self, k)
            parts.append(f"{'gold face' if k == 'gold' else k} {lo:.1f} -> {hi:.1f} raw "
                         f"(floor: {self.sources[k]['floor']}, "
                         f"ceiling: {self.sources[k]['ceiling']}"
                         + (f", curve x{self.curve[k]:g}" if self.curve.get(k, 1.0) != 1.0 else "") + ")")
        return "Scale: " + "; ".join(parts)


def job_scale(job_dir) -> dict:
    """The scale a trainer run used ({"face": [floor, ceiling], "body": [...], "gold": [...] with the gold face,
    "curve": {...}}) from its likeness.jsonl, with "gold_select": whether its probes' gold face was in charge."""
    p = Path(str(job_dir).strip().strip('"'))
    p = p if p.name == "likeness.jsonl" else p / "likeness.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"No likeness.jsonl in {p.parent} (a RulzTrainer run with likeness probes)")
    scale = None
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("scale"):
            scale = {**rec["scale"], "gold_select": bool(rec.get("gold_select")),
                     "gold_name": rec.get("gold_name") or "facenet",  # records before 3.14.0: FaceNet
                     # records before 3.19.0: buffalo_l, and the body score unknown
                     "face_model": rec.get("face_model") or "buffalo_l", "body_score": rec.get("body_score") or ""}
    if not scale:
        raise ValueError(f"{p} has no normalized scale (the run had likeness_normalize off)")
    return scale


def build_scale(refs: RefSet, normalize=True, baseline_rows=None, job=None, face_floor=None, body_floor=None,
                face_ceiling=None, body_ceiling=None, gold_floor=None, gold_ceiling=None) -> Scale:
    """The trainer's scale (LikenessController.scale): floor = the step-0 probe (here: the mean score of the baseline
    images, the base model without the LoRA) or the default; ceiling = the references' typical best match (2+
    references) or the default; the ceiling at least MIN_SPAN above the floor (the body's at least halfway to 100),
    then stretched by the span. A trainer run's own scale (job) and numbers typed in (>= 0, final raw numbers as the
    run logs them) take precedence, in that order: typed > run > baseline / measured > default. The gold face gets its
    own scale, shaped like the face's, when the references have it."""
    ctl = LikenessController(SimpleNamespace(likeness_normalize=normalize, resolution_step_up=False),
                             face_ceiling=refs.face_ceiling, body_ceiling=refs.body_ceiling,
                             gold_ceiling=refs.gold_ceiling)
    kinds = KINDS if refs.has_gold else KINDS[:2]
    sources = {k: {"floor": "default", "ceiling": "references" if c is not None else "default"}
               for k, c in (("face", refs.face_ceiling), ("body", refs.body_ceiling), ("gold", refs.gold_ceiling))}
    fixed: dict[str, list] = {k: [None, None] for k in KINDS}  # final raw numbers, as a run logged them
    curve = {k: ctl.curve["face" if k == "gold" else k] for k in kinds}  # the gold face shaped like ArcFace's face
    for kind in kinds:
        vals = [r[f"{kind}_raw"] for r in (baseline_rows or []) if r.get(f"{kind}_raw") is not None]
        if vals:
            ctl.floor[kind] = sum(vals) / len(vals)
            sources[kind]["floor"] = f"baseline images ({len(vals)})"
    if job:
        for kind in kinds:
            if kind == "gold" and str(job.get("gold_name") or "facenet") != (refs.gold_name or "facenet"):
                # the run's gold scale is another recognizer's raw numbers: keep the references' own
                log.info("The run's gold face scale is %s's, not %s's: the gold face keeps its own scale",
                         gold_label(job.get("gold_name")), gold_label(refs.gold_name))
                continue
            if kind == "face" and str(job.get("face_model") or "buffalo_l") != (refs.face_model or "buffalo_l"):
                # another InsightFace pack's raw numbers (face_model): the face keeps the references' own scale
                log.info("The run's face scale is %s's, not %s's: the face keeps its own scale (pick the run's face "
                         "model on the references node to use it)", job.get("face_model"), refs.face_model)
                continue
            if kind == "body" and job.get("body_score") and job["body_score"] != refs.body_kind:
                # another body score's raw numbers: the body keeps the references' own scale
                log.info("The run's body scale is body_score %s's, not %s's: the body keeps its own scale",
                         job["body_score"], refs.body_kind)
                continue
            if kind in job:
                fixed[kind] = [float(job[kind][0]), float(job[kind][1])]
                sources[kind] = {"floor": "trainer run", "ceiling": "trainer run"}
                curve[kind] = float((job.get("curve") or {}).get(kind, 1.0))  # a run from before the curve: linear
    for kind, lo, hi in (("face", face_floor, face_ceiling), ("body", body_floor, body_ceiling),
                         ("gold", gold_floor, gold_ceiling)):
        if lo is not None and lo >= 0:
            fixed[kind][0] = float(lo)
            sources[kind]["floor"] = "typed in"
        if hi is not None and hi >= 0:
            fixed[kind][1] = float(hi)
            sources[kind]["ceiling"] = "typed in"
    scales = {}
    for kind in kinds:
        if fixed[kind][0] is not None:
            ctl.floor[kind] = fixed[kind][0]  # a floor typed in or logged moves the stretched span with it
        lo, hi = ctl.scale(kind)
        if fixed[kind][1] is not None:  # a ceiling typed in or logged is the final one (the trainer prints these)
            hi = max(fixed[kind][1], lo + MIN_SPAN[kind])
        scales[kind] = (lo, hi)
    return Scale(normalize, scales["face"], scales["body"], sources, curve, scales.get("gold"))


# ---------------------------------------------------------------- scoring


def score_rows(refs: RefSet, items, progress=None, interrupted=None) -> list[dict]:
    """Raw scores of each (label, path): the face score as the trainer's FaceScorer.score (with the gold face as its
    probes score it, when the references have it), the body score as its body scorer's score. progress(i) after each
    image; interrupted() raises to stop."""
    if refs.face is not None:
        _restore(refs.face, refs.face_state)
    if refs.body is not None:
        _restore(refs.body, refs.body_state)
    rows = []
    for i, (label, path) in enumerate(items):
        if interrupted:
            interrupted()
        row = {"index": i + 1, "image": label, "path": str(path), "faces": None, "face_cosine": None,
               "face_raw": None, "best_face_ref": None, "gold_cosine": None, "gold_raw": None, "body_raw": None,
               "body_compared": None, "face_skipped": ""}
        if refs.face is not None:  # with the probes' subject check: an ambiguous or tiny face stays unscored
            s = refs.face.score(path, gold=refs.has_gold, qc=True)
            row.update(faces=s.faces, face_cosine=s.cosine, face_raw=s.percent, best_face_ref=refs.label(s.best_ref),
                       gold_cosine=s.gold_cosine, gold_raw=s.gold_percent, face_skipped=s.skipped)
        if refs.body is not None:
            row["body_raw"], row["body_compared"] = refs.body.score(path)
        rows.append(row)
        if progress:
            progress(i + 1)
    return rows


def apply_scale(rows, scale: Scale, face_weight=FACE_WEIGHT, body_weight=BODY_WEIGHT,
                gold_select=False) -> list[dict]:
    """Each image's normalized scores and overall; gold_select (the trainer's face_gold_select, with a gold scale):
    the gold face, not ArcFace's, in the overall. overall_arcface is the overall with ArcFace's face either way."""
    sel = "gold" if gold_select and scale.gold is not None else "face"
    for r in rows:
        for kind in KINDS:
            r[kind] = scale.norm(kind, r.get(f"{kind}_raw"))
        r["overall"] = combine(r[sel], r["body"], face_weight, body_weight)
        r["overall_arcface"] = combine(r["face"], r["body"], face_weight, body_weight)
        r["overall_raw"] = combine(r.get(f"{sel}_raw"), r["body_raw"], face_weight, body_weight)
    return rows


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def summarize(rows, scale: Scale, face_weight=FACE_WEIGHT, body_weight=BODY_WEIGHT, gold_select=False) -> dict:
    """The averages as the trainer averages a probe's seeds: the raw scores of the images that have one averaged,
    then normalized, then combined (with the gold face in ArcFace's place with gold_select, as apply_scale); plus the
    spread of the images' own normalized scores."""
    gold_select = bool(gold_select and scale.gold is not None)
    out = {"images": len(rows), "face_weight": face_weight, "body_weight": body_weight, "gold_select": gold_select,
           "has_gold": scale.gold is not None}
    for kind in KINDS:
        raw = [r[f"{kind}_raw"] for r in rows if r.get(f"{kind}_raw") is not None]
        own = [r[kind] for r in rows if r.get(kind) is not None]
        out[kind] = scale.norm(kind, _mean(raw))
        out[f"{kind}_raw"] = _mean(raw)
        out[f"{kind}_count"] = len(raw)
        out[f"{kind}_image_mean"] = _mean(own)
        out[f"{kind}_min"] = min(own) if own else None
        out[f"{kind}_max"] = max(own) if own else None
        out[f"{kind}_stdev"] = statistics.pstdev(own) if len(own) > 1 else None
    out["overall"] = combine(out["gold" if gold_select else "face"], out["body"], face_weight, body_weight)
    out["overall_arcface"] = combine(out["face"], out["body"], face_weight, body_weight)
    own = [r["overall"] for r in rows if r["overall"] is not None]
    out["overall_image_mean"], out["overall_count"] = _mean(own), len(own)
    out["overall_min"], out["overall_max"] = (min(own), max(own)) if own else (None, None)
    out["overall_stdev"] = statistics.pstdev(own) if len(own) > 1 else None
    return out


def pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}%"


def summary_text(summary: dict, refs: RefSet) -> str:
    n = summary["images"]
    face = f"Face {pct(summary['face'])}" + (
        f" ({summary['face_count']} of {n} with a face)" if refs.face else
        " (no face in any face reference)" if refs.face_info else " (no face references)")
    body = f"Body {pct(summary['body'])}" + (
        f" ({summary['body_count']} of {n} with a usable body)" if refs.body else
        " (no usable body in any body reference)" if refs.body_info else " (no body references)")
    lines = [f"Likeness of {n} image{'s' if n != 1 else ''}", f"Overall {pct(summary['overall'])}", face]
    if summary.get("has_gold"):
        lines.append(f"Gold face ({gold_label(refs.gold_name)}) {pct(summary['gold'])}" + (
            " (in charge of the overall score)" if summary["gold_select"] else " (not in the overall score)"))
    lines.append(body)
    if summary.get("gold_select"):
        lines.append(f"Overall with ArcFace's face {pct(summary['overall_arcface'])}")
    if n > 1 and summary["overall_count"]:
        lines.append(f"Images' overall scores range from {pct(summary['overall_min'])} to "
                     f"{pct(summary['overall_max'])}")
    return "\n".join(lines)


def details_text(rows) -> str:
    out = []
    for r in rows:
        face = (pct(r["face"]) if r["face"] is not None else f"unscored ({r['face_skipped']})"
                if r.get("face_skipped") else "no face" if r["faces"] == 0 else "n/a")
        body = pct(r["body"]) if r["body"] is not None else "n/a"
        gold = f" | gold face {pct(r['gold'])}" if r.get("gold") is not None else ""
        out.append(f"{r['index']:>3}. {Path(r['image']).name}: overall {pct(r['overall'])} | face {face}{gold} | "
                   f"body {body}")
    return "\n".join(out)


# ---------------------------------------------------------------- spreadsheet

IMAGE_COLUMNS = [  # (key, header)
    ("index", "#"), ("image", "Image"), ("overall", "Overall"), ("face", "Face"), ("body", "Body"),
    ("gold", "Gold face"), ("overall_arcface", "Overall with ArcFace's face"),
    ("faces", "Faces found"), ("face_skipped", "Face unscored because"), ("best_face_ref", "Closest face reference"),
    ("face_cosine", "Face cosine"),
    ("face_raw", "Face raw %"), ("gold_cosine", "Gold face cosine"), ("gold_raw", "Gold face raw %"),
    ("body_raw", "Body raw %"), ("body_compared", "Body references compared"),
    ("overall_raw", "Overall raw %"), ("path", "File")]
PERCENT_KEYS = {"overall", "face", "body", "gold", "overall_arcface", "face_raw", "gold_raw", "body_raw",
                "overall_raw"}
COSINE_KEYS = {"face_cosine", "gold_cosine"}


def _summary_rows(summary, scale: Scale, refs: RefSet):
    s = lambda key: [summary[f"{k}_{key}" if key else k] for k in ("overall", "face", "body", "gold")]  # noqa: E731
    yield ["", "Overall", "Face", "Body", f"Gold face ({gold_label(refs.gold_name)})"]
    yield ["Average (as the trainer averages a probe)"] + s("")
    yield ["Mean of the images' own scores"] + s("image_mean")
    yield ["Lowest image"] + s("min")
    yield ["Highest image"] + s("max")
    yield ["Standard deviation"] + s("stdev")
    yield ["Images scored"] + s("count")
    yield ["Raw average %", None, summary["face_raw"], summary["body_raw"], summary["gold_raw"]]
    yield []
    yield ["Images", summary["images"]]
    yield ["Face / body weight", f"{summary['face_weight']:g} / {summary['body_weight']:g}"]
    yield ["Face in the overall score", f"gold face ({gold_label(refs.gold_name)})" if summary["gold_select"]
           else "ArcFace"]
    yield ["Overall with ArcFace's face", summary["overall_arcface"]]
    yield ["Normalized", "yes" if scale.normalize else "no"]
    for kind in scale.kinds():
        lo, hi = getattr(scale, kind)
        yield [f"{'Gold face' if kind == 'gold' else kind.title()} scale (raw floor, ceiling)", lo, hi,
               f"floor: {scale.sources[kind]['floor']}, ceiling: {scale.sources[kind]['ceiling']}"
               + (f", curve x{scale.curve[kind]:g}" if scale.curve.get(kind, 1.0) != 1.0 else "")]
    yield ["Face references (with a face, given)", refs.face_count, len(refs.face_info)]
    yield [f"Gold face references ({gold_label(refs.gold_name)})", refs.gold_count if refs.has_gold else "off"]
    yield [f"Body references ({refs.body_kind}; usable, given)", refs.body_count, len(refs.body_info)]


def _reference_rows(refs: RefSet):
    yield ["Kind", "Reference", "Usable", "Details"]
    for info in refs.face_info:
        yield ["face", refs.label(info.get("image")), "yes" if info.get("faces") else "no face",
               "cut off by the frame" if info.get("cut_off") else ""]
    for info in refs.body_info:
        if refs.body_kind == "appearance":
            usable = ("yes" if info.get("used") else "no person" if not info.get("person")
                      else "body window not in frame")
            detail = f"window coverage {info['cover']:.2f}" if info.get("cover") is not None else ""
        else:
            usable = "yes" if info.get("ratios") else "no pose"
            detail = f"{info.get('ratios', 0)} ratios"
        yield ["body", refs.label(info.get("image")), usable, detail]


def _cell(v):
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 4)
    return v


def export(rows, summary, scale: Scale, refs: RefSet, base, fmt: str = "xlsx") -> list[Path]:
    """Write the scores: an .xlsx with Summary, Images and References sheets, or three CSV files (fmt "csv", or when
    openpyxl is missing). base: the path without an extension. Returns the files written."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "xlsx":
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            log.warning("openpyxl is not installed: writing CSV files instead of an .xlsx")
            fmt = "csv"
    if fmt == "csv":
        files = [base.with_name(base.name + "_images.csv"), base.with_name(base.name + "_summary.csv"),
                 base.with_name(base.name + "_references.csv")]
        with open(files[0], "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([h for _, h in IMAGE_COLUMNS])
            for r in rows:
                w.writerow([_cell(r.get(k)) for k, _ in IMAGE_COLUMNS])
        for path, gen in ((files[1], _summary_rows(summary, scale, refs)), (files[2], _reference_rows(refs))):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows([[_cell(v) for v in row] for row in gen])
        return files
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    for row in _summary_rows(summary, scale, refs):
        ws.append([_cell(v) for v in row])
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in range(2, 9):
        for c in range(2, 6):
            if isinstance(ws.cell(r, c).value, float):
                ws.cell(r, c).number_format = "0.0"
    ws.column_dimensions["A"].width = 46
    for col in "BCD":
        ws.column_dimensions[col].width = 14
    ws.column_dimensions["E"].width = 20
    wi = wb.create_sheet("Images")
    wi.append([h for _, h in IMAGE_COLUMNS])
    for r in rows:
        wi.append([_cell(r.get(k)) for k, _ in IMAGE_COLUMNS])
    for c in wi[1]:
        c.font = Font(bold=True)
    wi.freeze_panes = "C2"
    wi.auto_filter.ref = wi.dimensions
    for j, (key, header) in enumerate(IMAGE_COLUMNS, start=1):
        letter = get_column_letter(j)
        num = "0.0" if key in PERCENT_KEYS else "0.000" if key in COSINE_KEYS else None
        if num:
            for cell in wi[letter][1:]:
                cell.number_format = num
        longest = max([len(header)] + [len(str(_cell(r.get(key)) or "")) for r in rows])
        wi.column_dimensions[letter].width = min(60, max(8, longest + 2))
    wr = wb.create_sheet("References")
    for row in _reference_rows(refs):
        wr.append(row)
    for c in wr[1]:
        c.font = Font(bold=True)
    wr.column_dimensions["B"].width = 60
    wr.column_dimensions["C"].width = 24
    wr.column_dimensions["D"].width = 26
    path = base.with_name(base.name + ".xlsx")
    wb.save(path)
    return [path]
