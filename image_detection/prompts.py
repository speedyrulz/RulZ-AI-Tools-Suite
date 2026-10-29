"""Prompts and JSON schemas, verbatim from the build spec."""

from __future__ import annotations

import json
from typing import Sequence

SYSTEM_PROMPT = """You are an image filter. You receive one image and a numbered list of filters. For each filter, decide whether the image satisfies it, using only what is clearly visible in the image.

Rules:
- Judge each filter independently of the others.
- A filter that describes a person is satisfied only if at least one person in the image meets ALL of its conditions. The conditions must hold for the same person, not be spread across different people.
- Judge apparent sex, age and physical features from appearance alone.
- If the relevant part of the image is not visible, too small, blurred, cropped out, or covered, answer false with low confidence. Do not infer from context, captions, clothing, setting, or what is typical.
- Confidence is your estimate that a careful human would agree with your answer, from 0.0 to 1.0.
- Evidence is at most 12 words naming what you saw that decided the answer.
- Respond with a single JSON object and nothing else: no prose, no markdown fences, no explanation."""

RETRY_SUFFIX = "\n\nReturn only the JSON object."

EXPAND_SYSTEM_PROMPT = (
    "You write precise visual checklists for an image classifier. Given a short tag, produce a rubric "
    "the classifier can apply to a single image using only what is visible. Output JSON only."
)

EXPAND_USER_TEMPLATE = """Tag: "{tag}"

Write the rubric with four labeled parts on one line each:
SUBJECT: who or what must be present.
REQUIRE: every condition that must be visibly true, all for the same subject.
EXCLUDE: common near-misses that must NOT count.
FALSE IF: situations where the answer is false because the evidence is not visible.

Guidance:
- If the tag describes people and does not mention a boy, girl, child or kid, the SUBJECT must be an adult, and EXCLUDE must list children or teenagers. If it does mention one, keep the child as the subject.
- EXCLUDE must name at least three specific near-misses the classifier would otherwise accept: neighbouring colors or sizes, the other sex, look-alike objects, things that only make the feature appear present.
- A rubric about a body part must state a minimum visible size.

Example for the tag "men with blonde hair":
{{"id":"man_blonde_hair","rubric":"SUBJECT: an adult who appears male. REQUIRE: scalp hair is visible; the hair color is blonde, from pale yellow to golden. EXCLUDE: children or teenagers; light brown, grey or white hair; dyed unnatural colors; a blonde beard with non-blonde scalp hair; a blonde person who appears female. FALSE IF: the head is out of frame or the hair is covered by a hat, hood or helmet."}}

Also produce a snake_case id of at most 4 words.
Return exactly: {{"id":"...","rubric":"SUBJECT: ... REQUIRE: ... EXCLUDE: ... FALSE IF: ..."}}"""


def build_user_message(tags: Sequence[tuple[str, str]]) -> str:
    """tags: (id, rubric) pairs, at most MAX_TAGS_PER_CALL of them."""
    lines = ["Filters:"]
    for i, (tag_id, rubric) in enumerate(tags, start=1):
        lines.append(f'{i}. id="{tag_id}" — {rubric}')
    example = {
        "results": {tag_id: {"match": False, "confidence": 0.0, "evidence": ""} for tag_id, _ in tags}
    }
    lines.append("")
    lines.append("Return exactly this shape, one entry per filter id, nothing else:")
    lines.append(json.dumps(example, separators=(",", ":")))
    return "\n".join(lines)


def build_classify_schema(tag_ids: Sequence[str]) -> dict:
    tag_schema = {
        "type": "object",
        "properties": {
            "match": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string", "maxLength": 80},
        },
        "required": ["match", "confidence", "evidence"],
    }
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "object",
                "properties": {tag_id: tag_schema for tag_id in tag_ids},
                "required": list(tag_ids),
            }
        },
        "required": ["results"],
    }


EXPAND_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "maxLength": 60},
        "rubric": {"type": "string", "maxLength": 1200},
    },
    "required": ["id", "rubric"],
}
