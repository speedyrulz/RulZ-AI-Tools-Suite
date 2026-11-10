"""Probe prompt presets for character LoRAs, and the prompt check: does a probe render still show what its prompt
asked for?

A LoRA that overtrains on its photos' framing stops following the prompt: a full-body prompt comes back as a
head-and-shoulders portrait. The likeness scores do not see that (a closer face scores no worse), so this module
reads the framing of every probe render from what the scorers already find in it:

- the face boxes of the face scorer (facesim.FaceScorer.faces): how many faces, and how tall the largest one is as a
  share of the picture's height;
- the pose landmarks of the body scorer (bodysim.BodyScorer.landmarks): whether both shoulders and both hips are in
  the picture.

frame_of turns those into one of three frames, keeps compares it with what the prompt expects:

    headshot  a close face: the face is at least HEADSHOT_MIN_FACE of the picture's height
    full      the body down to the hips or further, with a small face (at most FULL_MAX_FACE of the height)
    upper     in between: waist-up, bust

"full" does not ask for the feet: the models draw a "full body" prompt from head to thigh as often as from head to
toe, and knees and ankles come and go with the pose, so they would report drift where the framing has not changed.

Drift is counted against the base model, not against the words: the step-0 probe of the same prompt and seed is the
reference. A render that kept its prompt at step 0 and does not now has drifted; a seed the base model itself got
wrong is left out, so a prompt the base model ignores cannot count against the LoRA. Without a step-0 probe the count
is kept / not kept.

The thresholds were set on the probe renders of real runs (docs/experimental-test-results.md, "Probe prompt check"):
one model family, one dataset. The check sees framing and the number of faces. It does not see clothing, setting,
pose or expression.

PRESETS are probe prompts for a character LoRA: a template with {trigger} and {subject}, the probe they suit (the
square face probe or the 2:3 body probe) and the frame they ask for. Pure functions, no torch: testable without models.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

EXPECTS = ("headshot", "upper", "full")
EXPECT_LABELS = {"headshot": "headshot (a close face)", "upper": "waist-up", "full": "full body (at least to the hips)"}
KINDS = ("face", "body")

# the largest face's height as a share of the picture's height
HEADSHOT_MIN_FACE = 0.34  # at or above: a headshot (measured: headshot probes 0.39-0.61, drifted body probes 0.34-0.48)
FULL_MAX_FACE = 0.24  # at or below, with the hips in the picture: full body (measured at step 0: 0.16-0.20)
MIN_VISIBILITY = 0.5  # a pose landmark counts when the pose model is this sure of it and it lies inside the picture
SECOND_FACE = 0.5  # another face at least this tall, relative to the largest, is a second person
L_SHO, R_SHO, L_HIP, R_HIP = 11, 12, 23, 24  # MediaPipe pose landmark indices

NAME_RE = re.compile(r"[A-Za-z0-9_.+-]{1,40}")
DEFAULT_SUBJECT = "a person"
DEFAULT_TRIGGER = "the character"


# ---------------------------------------------------------------- the frame of one render


def _inside(pts, vis, idx, size) -> bool:
    w, h = size
    x, y = float(pts[idx][0]), float(pts[idx][1])
    return float(vis[idx]) >= MIN_VISIBILITY and 0.0 <= x <= w and 0.0 <= y <= h


def frame_of(size, faces=None, pose=None) -> dict:
    """One render's frame from what the scorers found.

    size: (width, height) in pixels. faces: the face boxes [(x0, y0, x1, y1), ...] in pixels, [] when the face
    scorer found none, None without a face scorer. pose: (pts [33, 2] pixels, vis [33]) from the pose model, None
    when it found nobody or there is no body scorer (either way the body is unknown, never "missing": the pose model
    misses unusual poses).

    Returns {"frame": "headshot" | "upper" | "full" | None, "face": the largest face's height share | None,
    "faces": how many faces | None, "people": faces of comparable size, "hips": bool | None,
    "shoulders": bool | None}. frame None: cannot tell (no face and no pose)."""
    w, h = float(size[0]), float(size[1])
    share, n_faces, people = None, None, 0
    if faces is not None:
        heights = sorted((max(0.0, float(b[3]) - float(b[1])) for b in faces), reverse=True)
        n_faces = len(heights)
        if heights and heights[0] > 0 and h > 0:
            share = heights[0] / h
            people = sum(1 for v in heights if v >= SECOND_FACE * heights[0])
    hips = shoulders = None
    if pose is not None and pose[0] is not None and pose[1] is not None:
        pts, vis = pose
        hips = _inside(pts, vis, L_HIP, (w, h)) and _inside(pts, vis, R_HIP, (w, h))
        shoulders = _inside(pts, vis, L_SHO, (w, h)) and _inside(pts, vis, R_SHO, (w, h))
    if share is not None:
        if share >= HEADSHOT_MIN_FACE:
            frame = "headshot"
        elif share <= FULL_MAX_FACE and hips is not False:  # hips unknown: the face's size alone decides
            frame = "full"
        else:
            frame = "upper"
    elif hips:
        frame = "full"  # no face found (a back view, a face too small): the body decides
    elif shoulders:
        frame = "upper"
    else:
        frame = None
    return {"frame": frame, "face": None if share is None else round(share, 4), "faces": n_faces, "people": people,
            "hips": hips, "shoulders": shoulders}


def keeps(expect: str, frame: dict | None):
    """Does a render keep what its prompt expects? True / False, or None when there is nothing to check (no
    expectation) or the frame cannot be told. A second person of comparable size never keeps a one-person prompt."""
    if expect not in EXPECTS or not frame:
        return None
    if (frame.get("people") or 0) >= 2:
        return False
    if frame.get("frame") is None:
        return None
    return frame["frame"] == expect


_WORDS = (
    ("full", ("full-body", "full body", "fullbody", "full-length", "full length", "head to toe", "head to feet",
              "head-to-toe", "whole body")),
    ("upper", ("waist-up", "waist up", "from the waist", "upper body", "upper-body", "half-body", "half body",
               "medium shot")),
    ("headshot", ("headshot", "head shot", "close-up", "close up", "closeup", "face portrait")),
)


def guess_expect(prompt: str) -> str:
    """The frame a prompt asks for, from its words ("" when none of them is there). The earliest match wins."""
    text = " ".join(str(prompt or "").lower().split())
    best, at = "", len(text) + 1
    for expect, words in _WORDS:
        for word in words:
            i = text.find(word)
            if 0 <= i < at:
                best, at = expect, i
    return best


def resolve_expect(setting: str, prompt: str) -> str:
    """A probe's expectation from its setting: "" guesses from the prompt, "none" switches the check off."""
    setting = str(setting or "").strip().lower()
    if setting == "none" or not str(prompt or "").strip():
        return ""
    return setting if setting in EXPECTS else guess_expect(prompt)


# ---------------------------------------------------------------- presets


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    kind: str  # face: the square face probe | body: the 2:3 body probe
    template: str  # with {trigger} and, in the long ones, {subject}
    expect: str


PRESETS: tuple[Preset, ...] = (
    # the two the identity recipe has always used (short, the trigger alone): kept word for word
    Preset("portrait_plain", "Portrait, plain background (classic)", "face",
           "close-up portrait photo of {trigger}, neutral background, soft light", "headshot"),
    Preset("standing_plain", "Standing, plain background (classic)", "body",
           "full body photo of {trigger} standing straight facing the camera, arms at sides, plain background, "
           "whole body visible head to feet", "full"),
    # long natural-language prompts, the way the current models are captioned
    Preset("headshot_neutral", "Headshot, neutral, studio", "face",
           "{trigger}, a close-up headshot of {subject} looking straight at the camera with a neutral expression, "
           "plain light grey studio background, soft even light, sharp focus", "headshot"),
    Preset("headshot_outdoor", "Headshot, three-quarter view, outdoors", "face",
           "{trigger}, a close-up headshot of {subject} in three-quarter view with a slight smile, standing "
           "outdoors on a tree-lined street in soft afternoon light, the background softly out of focus",
           "headshot"),
    Preset("full_studio", "Full body, standing, studio", "body",
           "{trigger}, a full-body photo of {subject} standing upright and facing the camera with arms relaxed at "
           "the sides, wearing a plain t-shirt and jeans, plain light grey studio background, soft even light, the "
           "whole figure in the frame from head to feet", "full"),
    Preset("full_walking", "Full body, walking outdoors", "body",
           "{trigger}, a full-body photo of {subject} walking along a city sidewalk in casual clothes, a jacket "
           "and trousers, in daylight, seen from the front, the whole figure in the frame from head to feet",
           "full"),
    Preset("full_sitting", "Full body, sitting indoors", "body",
           "{trigger}, a full-body photo of {subject} sitting on a sofa in a bright living room, wearing a "
           "sweater and trousers, hands resting on the knees, daylight from a window, the whole figure in the "
           "frame", "full"),
    Preset("upper_cafe", "Waist-up, at a cafe table", "body",
           "{trigger}, a waist-up photo of {subject} sitting at a small table in a cafe, wearing a casual shirt, "
           "one hand around a cup, looking at the camera, warm indoor light", "upper"),
)
_BY_KEY = {p.key: p for p in PRESETS}
CLASSIC = {"face": "portrait_plain", "body": "standing_plain"}  # what the identity recipe fills in


def preset(key: str) -> Preset:
    try:
        return _BY_KEY[str(key)]
    except KeyError:
        raise ValueError(f"unknown probe prompt preset {key!r} (one of: {', '.join(_BY_KEY)})") from None


def presets_of(kind: str) -> list[Preset]:
    return [p for p in PRESETS if p.kind == kind]


def subject_of(cfg) -> str:
    """What the presets call the person: probe_subject, else the first dataset's subject_class (LoRA Isolation's,
    "a woman"), else "a person"."""
    values = [getattr(cfg, "probe_subject", "")]
    try:
        values += [getattr(spec, "subject_class", "") for spec in cfg.dataset_specs()]
    except Exception:  # noqa: BLE001  (a config without datasets yet, or a plain namespace in a test)
        pass
    for value in values:
        value = " ".join(str(value or "").split())
        if value:
            return value
    return DEFAULT_SUBJECT


def build(key: str, trigger: str = "", subject: str = "") -> str:
    """A preset's prompt with the trigger word and the subject filled in (without a trigger word: "the character")."""
    return preset(key).template.format(trigger=" ".join(str(trigger or "").split()) or DEFAULT_TRIGGER,
                                       subject=" ".join(str(subject or "").split()) or DEFAULT_SUBJECT)


# ---------------------------------------------------------------- what a probe checks


@dataclass(frozen=True)
class Check:
    name: str  # "face" / "body": the likeness probes; else an extra prompt's own name
    label: str
    kind: str  # face | body: the probe whose size it renders at
    prompt: str
    expect: str  # "" = rendered but not checked
    extra: bool = False  # not part of the likeness score


def normalize_checks(entries, trigger: str = "", subject: str = "") -> list[dict]:
    """probe_checks as explicit rows [{"name", "label", "prompt", "expect", "kind"}]: a preset key expands to its
    prompt; a mapping gives its own prompt (name, prompt; expect "" guesses from the prompt, kind defaults to
    body) or names a preset to change. A row's expect is one of EXPECTS or "none" (rendered, not checked), so
    normalizing a normalized list changes nothing. Raises ValueError with what to change."""
    rows, seen = [], set()
    for i, entry in enumerate(entries or []):
        if isinstance(entry, str):
            p = preset(entry.strip())
            row = {"name": p.key, "label": p.label, "prompt": build(p.key, trigger, subject), "expect": p.expect,
                   "kind": p.kind}
        elif isinstance(entry, dict):
            p = preset(str(entry["preset"]).strip()) if entry.get("preset") else None
            name = str(entry.get("name") or (p.key if p else f"check{i + 1}"))
            prompt = " ".join(str(entry.get("prompt") or (build(p.key, trigger, subject) if p else "")).split())
            if not prompt:
                raise ValueError(f"probe_checks entry {name!r} needs a prompt (or a preset)")
            expect = str(entry.get("expect") or (p.expect if p else "")).strip().lower()
            row = {"name": name, "label": str(entry.get("label") or (p.label if p else name)), "prompt": prompt,
                   "expect": expect or guess_expect(prompt) or "none",
                   "kind": str(entry.get("kind") or (p.kind if p else "body")).strip().lower()}
        else:
            raise ValueError(f"probe_checks entry {i + 1} must be a preset key or a mapping with name and prompt")
        if not NAME_RE.fullmatch(row["name"]) or row["name"] in KINDS:
            raise ValueError(f"probe_checks name {row['name']!r} must be 1-40 letters, digits or _ . + - and not "
                             f"'face' or 'body'")
        if row["name"] in seen:
            raise ValueError(f"probe_checks lists {row['name']!r} twice")
        if row["expect"] != "none" and row["expect"] not in EXPECTS:
            raise ValueError(f"probe_checks entry {row['name']!r}: expect must be one of {', '.join(EXPECTS)}, "
                             f"none or empty, got {row['expect']!r}")
        if row["kind"] not in KINDS:
            raise ValueError(f"probe_checks entry {row['name']!r}: kind must be face or body, got {row['kind']!r}")
        seen.add(row["name"])
        rows.append(row)
    return rows


def checks(cfg) -> list[Check]:
    """Everything a probe of this config renders that the prompt check looks at: the two likeness probes (their
    expectation from probe_face_expect / probe_body_expect, guessed from the prompt when empty) and the extra
    prompts of probe_checks."""
    out = []
    for kind, label in (("face", "face probe"), ("body", "body probe")):
        prompt = getattr(cfg, f"probe_{kind}_prompt", "") or ""
        if prompt.strip():
            out.append(Check(kind, label, kind, prompt, resolve_expect(getattr(cfg, f"probe_{kind}_expect", ""), prompt)))
    for row in normalize_checks(getattr(cfg, "probe_checks", None), getattr(cfg, "trigger_word", ""), subject_of(cfg)):
        out.append(Check(row["name"], row["label"], row["kind"], row["prompt"],
                         "" if row["expect"] == "none" else row["expect"], extra=True))
    return out


def extra_checks(cfg) -> list[Check]:
    return [c for c in checks(cfg) if c.extra]


def wanted(cfg) -> bool:
    """Is there anything for the prompt check to do on this config's probes?"""
    return any(c.expect for c in checks(cfg))


# ---------------------------------------------------------------- counting a probe


def verdicts(expect: str, frames) -> list:
    return [keeps(expect, f) for f in (frames or [])]


def reference(specs, frames: dict) -> dict:
    """The step-0 probe as the reference: {name: [kept True / False / None per seed]} for the checked prompts."""
    return {c.name: verdicts(c.expect, frames.get(c.name)) for c in specs if c.expect and frames.get(c.name)}


def tally(specs, frames: dict, refs: dict | None = None, base: bool = False) -> dict:
    """One probe's prompt check. specs: checks(cfg). frames: {name: [frame_of(...) per seed]}. refs: reference() of
    the step-0 probe, None or {} without one. base: this probe is the reference itself (the base model).

    Per prompt: n renders, kept, drifted (kept at step 0, not now; without a reference: not kept), base_miss (the
    base model did not keep it either: left out), unknown (cannot tell now, or no reference for that seed), and
    drift = drifted / (kept + drifted). Overall: drift_share is the drift of the prompt that drifted most ("worst"),
    kept_share 1 minus it. The prompts are not pooled: a face probe that stays a headshot must not halve the drift
    of a body probe that has turned into portraits. Both None when nothing counted or for the base probe itself."""
    prompts, kept_all, drift_all = {}, 0, 0
    worst, worst_name = None, ""
    use_ref = bool(refs) and not base
    for c in specs:
        fr = list(frames.get(c.name) or [])
        if not c.expect or not fr:
            continue
        now = verdicts(c.expect, fr)
        ref = (refs or {}).get(c.name) if use_ref else None
        kept = drifted = miss = unknown = 0
        for i, v in enumerate(now):
            if ref is not None:
                r = ref[i] if i < len(ref) else None
                if r is False:
                    miss += 1
                    continue
                if r is None:
                    unknown += 1
                    continue
            if v is True:
                kept += 1
            elif v is False:
                drifted += 1
            else:
                unknown += 1
        drift = drifted / (kept + drifted) if kept + drifted else None
        prompts[c.name] = {"label": c.label, "expect": c.expect, "n": len(fr), "kept": kept, "drifted": drifted,
                           "base_miss": miss, "unknown": unknown, "drift": None if drift is None else round(drift, 4),
                           "frames": [f.get("frame") if f else None for f in fr],
                           "face": [f.get("face") if f else None for f in fr]}
        kept_all += kept
        drift_all += drifted
        if drift is not None and (worst is None or drift > worst):
            worst, worst_name = drift, c.name
    if not prompts:
        return {}
    share = None if base else worst
    return {"prompts": prompts, "base": bool(base), "reference": use_ref, "kept": kept_all, "drifted": drift_all,
            "worst": "" if share is None else worst_name,
            "drift_share": None if share is None else round(share, 4),
            "kept_share": None if share is None else round(1.0 - share, 4)}


def describe(check: dict) -> str:
    """The log line of a probe's prompt check: "face probe 8/8 kept; body probe 3/8 kept (5 drifted)"."""
    if not check or not check.get("prompts"):
        return ""
    parts = []
    for p in check["prompts"].values():
        counted = p["kept"] + p["drifted"]
        if check.get("base"):
            text = f"{p['label']} {p['kept']}/{counted} show {p['expect']}"
            if p["drifted"]:
                text += f" ({p['drifted']} do not: left out of later counts)"
        else:
            text = f"{p['label']} {p['kept']}/{counted} kept"
            notes = []
            if p["drifted"]:
                notes.append(f"{p['drifted']} drifted" if check.get("reference") else f"{p['drifted']} not {p['expect']}")
            if p["base_miss"]:
                notes.append(f"{p['base_miss']} the base model did not keep either")
            if notes:
                text += f" ({', '.join(notes)})"
        if p["unknown"]:
            text += f", {p['unknown']} unclear"
        parts.append(text)
    tail = ""
    if check.get("base"):
        tail = " | the base model's renders: the reference later probes are compared with"
    elif not check.get("reference"):
        tail = " | no step-0 probe to compare with: counted against the prompt alone"
    elif check.get("drift_share") is not None:
        tail = f" | drift {100 * check['drift_share']:.0f}%"
        if check["drift_share"] > 0 and len(check["prompts"]) > 1 and check.get("worst") in check["prompts"]:
            tail += f" (the {check['prompts'][check['worst']]['label']}, the one that drifted most)"
    return "; ".join(parts) + tail
