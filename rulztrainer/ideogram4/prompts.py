"""Ideogram 4 captions: structured JSON, the only format the model was trained on; plain captions are wrapped.

Ideogram's prompting guide (github.com/ideogram-oss/ideogram4, docs/prompting.md): a caption is a JSON object with
an optional high_level_description (a one- or two-sentence summary), an optional style_description and a required
compositional_deconstruction holding a background string and a list of elements ({"type": "obj", "desc": ...} or
{"type": "text", "text": ..., "desc": ...}, each with an optional bbox and color_palette), keys in a fixed order,
serialized compactly (separators (",", ":"), ensure_ascii False). Plain-text prompts "will not work" (and trip the
safety filter); Ideogram's pipeline rewrites them with an LLM ("magic prompt") first. Here:

  - a <name>.json caption next to an image wins over the plain caption file (caption_file) and is used as it is,
    re-serialized compactly with its key order kept; what Ideogram's CaptionVerifier would flag is logged (verify);
  - a plain caption becomes the smallest caption the schema allows: {"high_level_description": <caption>,
    "compositional_deconstruction": {"background": "", "elements": []}} (wrap_plain);
  - the trigger word goes where a subject is described: in front of the first object element's desc, or as an
    object element of its own when there is none (add_trigger); a caption that already holds it is left alone;
  - every prompt the text encoder sees passes as_caption: JSON is compacted, a template around a JSON caption
    ("close-up portrait photo of {caption}", the crop augmentation's) goes into its high_level_description, plain
    text is wrapped. The empty prompt stays empty: no text tokens at all, the reference pipeline's unconditional
    (image-only) branch, which is the default negative.

The verifier rules are ported from Ideogram's caption_verifier.py (Apache-2.0)."""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger("rulztrainer")

TOP_KEYS = ("high_level_description", "style_description", "compositional_deconstruction")
STYLE_KEYS_PHOTO = ("aesthetics", "lighting", "photo", "medium", "color_palette")
STYLE_KEYS_ART = ("aesthetics", "lighting", "medium", "art_style", "color_palette")
ELEMENT_KEYS = {"obj": ("type", "bbox", "desc", "color_palette"), "text": ("type", "bbox", "text", "desc", "color_palette")}
OPTIONAL = ("bbox", "color_palette")
_HEX = re.compile(r"^#[0-9A-F]{6}$")
_NON_ASCII_ESCAPE = re.compile(r"\\u(?:00[89a-fA-F][0-9a-fA-F]|0[1-9a-fA-F][0-9a-fA-F]{2}|[1-9a-fA-F][0-9a-fA-F]{3})")

# what caption dropout trains on: the smallest valid caption, empty
EMPTY_CAPTION = '{"high_level_description":"","compositional_deconstruction":{"background":"","elements":[]}}'


def compact(caption: dict) -> str:
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def parse_caption(prompt: str) -> dict | None:
    """The prompt as a JSON object caption, or None (plain text, or JSON that is not an object)."""
    try:
        caption = json.loads(prompt)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return caption if isinstance(caption, dict) else None


def wrap_plain(text: str) -> dict:
    """The smallest caption the schema allows, around a plain sentence."""
    return {"high_level_description": text.strip(), "compositional_deconstruction": {"background": "", "elements": []}}


def _embedded(prompt: str) -> tuple[str, dict, str] | None:
    """(text before, caption, text after) for a JSON caption inside other text (a template around a caption)."""
    i, j = prompt.find("{"), prompt.rfind("}")
    if i < 0 or j <= i:
        return None
    caption = parse_caption(prompt[i:j + 1])
    if caption is None or "compositional_deconstruction" not in caption:
        return None
    return prompt[:i], caption, prompt[j + 1:]


def as_caption(prompt: str) -> str:
    """What the text encoder reads for a prompt (see the module docstring): always a compact JSON caption, except the
    empty prompt (the unconditional, text-free branch)."""
    if prompt is None or not prompt.strip():
        return ""
    caption = parse_caption(prompt)
    if caption is not None:
        return compact(caption)
    found = _embedded(prompt)
    if found is not None:
        before, caption, after = found
        extra = " ".join(p for p in (before.strip(" ,:;"), after.strip(" ,:;")) if p)
        if extra:
            hld = caption.get("high_level_description") or ""
            merged = {"high_level_description": f"{extra}: {hld}" if hld else extra}
            merged.update({k: v for k, v in caption.items() if k != "high_level_description"})
            caption = merged
        return compact(caption)
    return compact(wrap_plain(prompt))


def clean_prompt(prompt: str) -> str:
    """A .json caption file's text: compact, key order kept; what the verifier would flag is logged. A file that is
    not a JSON object is wrapped as a plain caption."""
    caption = parse_caption(prompt)
    if caption is None:
        log.warning("A .json caption is not a JSON object; it is used as a plain caption: %.60s", prompt)
        return compact(wrap_plain(prompt))
    issues = verify(caption) + check_ascii_escapes(prompt)
    if issues:
        log.warning("Ideogram 4 caption %.60s...: %s", compact(caption), "; ".join(issues[:3]))
    return compact(caption)


def add_trigger(caption_text: str, trigger: str) -> str:
    """A caption as a JSON caption (plain text wrapped) with the trigger word where its subject is described: in front
    of the first object element's desc, else as an object element of its own. A caption that already holds the
    trigger is left as it is."""
    caption = parse_caption(caption_text) if caption_text else None
    if caption is None:
        caption = wrap_plain(caption_text or "")
    if not trigger or trigger in compact(caption):
        return compact(caption)
    cd = caption.get("compositional_deconstruction")
    if not isinstance(cd, dict):
        cd = caption["compositional_deconstruction"] = {"background": "", "elements": []}
    elements = cd.get("elements")
    if not isinstance(elements, list):
        elements = cd["elements"] = []
    for i, el in enumerate(elements):
        if isinstance(el, dict) and el.get("type") == "obj":
            desc = el.get("desc") or ""
            el = {**el, "desc": f"{trigger}, {desc}" if desc else trigger}
            ordered = {k: el[k] for k in ELEMENT_KEYS["obj"] if k in el}  # a desc added last moves to its place
            ordered.update({k: v for k, v in el.items() if k not in ordered})
            elements[i] = ordered
            return compact(caption)
    elements.insert(0, {"type": "obj", "desc": trigger})
    return compact(caption)


# ---------------------------------------------------------------- Ideogram's CaptionVerifier rules


def _check_order(obj: dict, expected, path: str, out: list[str]) -> None:
    present = tuple(k for k in obj if k in expected)
    if present != tuple(expected):
        out.append(f"{path}: key order is {present}, expected {tuple(expected)}")
    extra = [k for k in obj if k not in expected]
    if extra:
        out.append(f"{path}: keys {extra} are not allowed in this context")


def _palette(palette, path: str, most: int, out: list[str]) -> None:
    if not isinstance(palette, list):
        out.append(f"{path}: expected a list")
        return
    if len(palette) > most:
        out.append(f"{path}: too many colors ({len(palette)}), expected at most {most}")
        return
    for i, c in enumerate(palette):
        if not isinstance(c, str) or not _HEX.match(c):
            out.append(f"{path}[{i}]: '{c}' is not a valid #RRGGBB hex color")


def verify(caption) -> list[str]:
    """The warnings Ideogram's CaptionVerifier.verify gives a parsed caption (empty: it passes)."""
    out: list[str] = []
    if not isinstance(caption, dict):
        return [f"root: expected a JSON object (dict), got {type(caption).__name__}"]
    unknown = [k for k in caption if k not in TOP_KEYS]
    if unknown:
        out.append(f"root: unknown keys {unknown} (not in schema)")
    if "high_level_description" in caption and not isinstance(caption["high_level_description"], str):
        out.append("high_level_description: expected a string")
    if "style_description" in caption:
        sd = caption["style_description"]
        if not isinstance(sd, dict):
            out.append("style_description: expected a dict")
        else:
            unknown = [k for k in sd if k not in set(STYLE_KEYS_PHOTO) | set(STYLE_KEYS_ART)]
            if unknown:
                out.append(f"style_description: unknown keys {unknown} (not in schema)")
            photo, art = "photo" in sd, "art_style" in sd
            if photo == art:
                out.append("style_description: expected exactly one of 'photo' and 'art_style'")
            else:
                order = [k for k in (STYLE_KEYS_PHOTO if photo else STYLE_KEYS_ART)
                         if k != "color_palette" or "color_palette" in sd]
                _check_order(sd, order, "style_description", out)
                if "color_palette" in sd:
                    _palette(sd["color_palette"], "style_description.color_palette", 16, out)
    if "compositional_deconstruction" not in caption:
        out.append("root: 'compositional_deconstruction' must exist")
        return out
    cd = caption["compositional_deconstruction"]
    if not isinstance(cd, dict):
        out.append("compositional_deconstruction: expected a dict")
        return out
    if not isinstance(cd.get("background"), str):
        out.append("compositional_deconstruction: 'background' must exist and be a string")
        return out
    if "elements" not in cd:
        out.append("compositional_deconstruction: 'elements' must exist")
        return out
    _check_order(cd, ("background", "elements"), "compositional_deconstruction", out)
    if not isinstance(cd["elements"], list):
        out.append("compositional_deconstruction.elements: expected a list")
        return out
    for i, el in enumerate(cd["elements"]):
        if not isinstance(el, dict):
            out.append(f"elements[{i}]: expected a dict")
            continue
        unknown = [k for k in el if k not in {"type", "bbox", "text", "desc", "color_palette"}]
        if unknown:
            out.append(f"elements[{i}]: unknown keys {unknown} (not in schema)")
        if el.get("type") not in ELEMENT_KEYS:
            out.append(f"elements[{i}]: 'type' must exist and be 'obj' or 'text'")
            continue
        _check_order(el, [k for k in ELEMENT_KEYS[el["type"]] if k not in OPTIONAL or k in el], f"elements[{i}]",
                     out)
        if "bbox" in el:
            b = el["bbox"]
            if not isinstance(b, list) or len(b) != 4:
                out.append(f"elements[{i}].bbox: expected [ymin, xmin, ymax, xmax]")
            elif not all(isinstance(v, int) for v in b):
                out.append(f"elements[{i}].bbox: all values must be int")
            else:
                if not all(0 <= v <= 1000 for v in b):
                    out.append(f"elements[{i}].bbox: values must be in [0, 1000], got {b}")
                if b[0] > b[2]:
                    out.append(f"elements[{i}].bbox: ymin ({b[0]}) > ymax ({b[2]})")
                if b[1] > b[3]:
                    out.append(f"elements[{i}].bbox: xmin ({b[1]}) > xmax ({b[3]})")
        if "color_palette" in el:
            _palette(el["color_palette"], f"elements[{i}].color_palette", 5, out)
    return out


def check_ascii_escapes(raw: str) -> list[str]:
    """The verifier's warning for a file saved with ensure_ascii=True (\\uXXXX escapes, no literal non-ASCII)."""
    if _NON_ASCII_ESCAPE.search(raw or "") and not any(ord(c) > 0x7F for c in raw):
        return ["non-ASCII characters saved as \\u escapes: save the caption with ensure_ascii=False"]
    return []
