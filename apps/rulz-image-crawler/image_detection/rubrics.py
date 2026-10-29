"""Reference rubrics from the spec. New tags should match their shape and level of detail."""

from __future__ import annotations

import re

REFERENCE_RUBRICS: dict[str, str] = {
    "man_blonde_hair": (
        "SUBJECT: an adult who appears male. REQUIRE: scalp hair is visible; the hair color is blonde, "
        "from pale yellow to golden. EXCLUDE: light brown, grey or white hair; dyed unnatural colors; "
        "a blonde beard with non-blonde scalp hair; a blonde person who appears female. FALSE IF: the "
        "head is out of frame or the hair is covered by a hat, hood or helmet."
    ),
    # Scored 10/10 positives, 10/10 negatives on the F:\yes / F:\no set (2026-10-29).
    "woman_blonde_hair": (
        "SUBJECT: an adult who appears female. REQUIRE: scalp hair is visible; the hair color is blonde, "
        "from pale yellow to golden. EXCLUDE: children or teenagers; light brown, grey or white hair; dyed "
        "unnatural colors; a blonde person who appears male, including males with long hair. FALSE IF: the "
        "head is out of frame or the hair is covered by a hat, hood or helmet."
    ),
    "woman_long_fingers": (
        "SUBJECT: an adult who appears female. REQUIRE: at least one hand is clearly visible with fingers "
        "extended enough to judge; the fingers look noticeably longer and more slender than typical "
        "relative to the palm. EXCLUDE: hands in fists, gloved, blurred, or smaller than about 60 px tall; "
        "long nails or rings that only make fingers appear longer. FALSE IF: no hand of that person is visible."
    ),
}

# Tag ids go into the prompt and become JSON keys, so keep them to a safe alphabet.
TAG_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")


PERSON_WORDS_RE = re.compile(
    r"\b(wom[ae]n|m[ae]n|person|people|persons|lady|ladies|guys?|gentlem[ae]n|females?|males?|humans?|"
    r"someone|somebody|blondes|brunettes|redheads)\b",
    re.IGNORECASE,
)
# A tag that names a child keeps children in scope; any other person tag means adults only.
CHILD_WORDS_RE = re.compile(
    r"\b(boys?|girls?|child|children|kids?|toddlers?|bab(?:y|ies)|infants?|teens?|teenagers?|minors?)\b",
    re.IGNORECASE,
)
_PART_RE = re.compile(r"\b(SUBJECT|REQUIRE|EXCLUDE|FALSE IF)\s*:", re.IGNORECASE)
_PART_ORDER = ("SUBJECT", "REQUIRE", "EXCLUDE", "FALSE IF")


def split_rubric(rubric: str) -> dict[str, str] | None:
    """{"SUBJECT": ..., "REQUIRE": ..., ...} or None if the rubric has no SUBJECT part."""
    marks = list(_PART_RE.finditer(rubric))
    parts: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(rubric)
        parts.setdefault(m.group(1).upper(), rubric[m.end() : end].strip())
    return parts if "SUBJECT" in parts else None


def join_rubric(parts: dict[str, str]) -> str:
    out = []
    for label in _PART_ORDER:
        text = parts.get(label, "").strip()
        if text:
            out.append(f"{label}: {text if text[-1] in '.!?' else text + '.'}")
    return " ".join(out)


def enforce_adult_subject(tag_text: str, rubric: str) -> str:
    """For a person tag that does not name a child (boy, girl, child, ...), require an adult in
    SUBJECT and list children/teenagers first in EXCLUDE. Other rubrics pass through unchanged."""
    if CHILD_WORDS_RE.search(tag_text):
        return rubric
    parts = split_rubric(rubric)
    if parts is None:
        return rubric
    if not (PERSON_WORDS_RE.search(tag_text) or PERSON_WORDS_RE.search(parts["SUBJECT"])):
        return rubric
    subject = parts["SUBJECT"].rstrip(" .;")
    if not re.search(r"\badults?\b", subject, re.IGNORECASE):
        parts["SUBJECT"] = f"an adult, not a child or teenager: {subject}."
    exclude = parts.get("EXCLUDE", "").strip()
    if not re.search(r"\b(child|children|kids?|minors?|teen\w*)\b", exclude, re.IGNORECASE):
        parts["EXCLUDE"] = f"children or teenagers; {exclude}" if exclude else "children or teenagers."
    return join_rubric(parts)


def normalize_tag_id(raw: str, max_words: int = 4) -> str:
    """Coerce a model-written id into snake_case of at most max_words words."""
    words = re.findall(r"[a-z0-9]+", raw.lower())
    return "_".join(words[:max_words]) or "tag"
