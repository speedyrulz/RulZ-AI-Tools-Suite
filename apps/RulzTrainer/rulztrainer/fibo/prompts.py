"""FIBO 2.0 prompts: structured (VGL JSON) captions cleaned as the diffusers pipeline cleans them, plus the null caption.

A prompt is either a plain sentence (used as is) or a JSON object caption, which is cleaned the way Fibo 2.0's training
cleaned captions (empty values and the aesthetic_score / preference_score keys dropped) and serialized compactly.
Ported from diffusers' pipeline_bria_fibo2.py (Copyright 2026 Bria AI and The HuggingFace Team, Apache-2.0).
"""
from __future__ import annotations

import json
import math

# The null caption Fibo 2.0 was trained with for classifier-free guidance (every key, every value empty); the negative
# prompt by default, and what caption dropout trains on
NULL_CAPTION = ('{"short_description":"","objects":[],"background_setting":"","lighting":{},"aesthetics":{},'
                '"photographic_characteristics":{},"style_medium":"","context":"","artistic_style":""}')

# The chat turn each prompt is wrapped in for Qwen3-VL
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

# Qwen3-VL hidden-state layers behind the transformer's six text inputs: the Perceiver's bundle, then one per
# injection block; each bundle concatenates its three layers (3 x 2560 = 7680)
TEXT_ENCODER_OUT_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))

SCORE_KEYS = ("aesthetic_score", "preference_score")


def compact(caption: dict) -> str:
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def parse_caption(prompt: str) -> dict | None:
    """The prompt as a JSON object caption, or None for a plain sentence (or any other JSON)."""
    try:
        caption = json.loads(prompt)
    except (json.JSONDecodeError, TypeError):
        return None
    return caption if isinstance(caption, dict) else None


def clean_caption(caption: dict) -> dict:
    """Training captions hold no empty values (None, "", {}, [] or NaN) and no score keys, at any depth."""

    def empty(value):
        if isinstance(value, float):
            return math.isnan(value)
        return value is None or (isinstance(value, (str, dict, list)) and not value)

    def clean(value):
        if isinstance(value, dict):
            value = {k: clean(v) for k, v in value.items()}
            return {k: v for k, v in value.items() if not empty(v)}
        if isinstance(value, list):
            return [v for v in map(clean, value) if not empty(v)]
        return value

    keys = set(caption)
    out = {key: ({k: v for k, v in value.items() if k not in SCORE_KEYS} if isinstance(value, dict) else value)
           for key, value in clean(caption).items() if key not in SCORE_KEYS}
    out = {key: value for key, value in out.items() if value != {}}
    if keys == {"short_description", "text_render"}:  # keeps both keys, the list even when empty
        out = {"short_description": out.get("short_description") or "", "text_render": out.get("text_render") or []}
    return out


def clean_prompt(prompt: str) -> str:
    """A JSON caption in the form Fibo 2.0 was trained with; a plain sentence as it is."""
    if prompt.strip() == NULL_CAPTION:
        return NULL_CAPTION  # the null caption is encoded as it is (the pipeline never cleans the negative prompt)
    caption = parse_caption(prompt)
    if caption is None:
        return prompt
    return compact(clean_caption(caption) or {})


def add_trigger(caption_text: str, trigger: str) -> str:
    """Prepend a trigger word: in front of a sentence, or into a JSON caption's short_description."""
    if not trigger:
        return caption_text
    caption = parse_caption(caption_text)
    if caption is None:
        return f"{trigger}, {caption_text}" if caption_text else trigger
    desc = caption.get("short_description") or ""
    caption["short_description"] = f"{trigger}, {desc}" if desc else trigger
    return compact(caption)
