"""Photo search by people, facial expression, head direction, and AI tags and look-for searches.

Syntax: comma-separated groups; words in one group describe ONE face.
    "smiling"                 -> photos with anyone smiling
    "Mom smiling"             -> photos where Mom is smiling
    "Mom smiling, Dad"        -> Mom smiling and Dad somewhere in the photo
    "facing left"             -> anyone turned towards the left of the photo
    "Dad looking down, sad"   -> Dad looking down, and someone looking sad
    "red sweater, Mom"        -> the AI found a red sweater (Look for…, or an AI tag), and Mom is in it

A group that is no one's name and no expression is looked up in the AI results: a saved Look for… search
with that exact text, and AI tags containing those words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import attributes
from .db import Database


@dataclass
class SearchResult:
    photo_ids: set[int] | None   # None = no filter (show everything)
    problem: str | None = None   # explanation when a name couldn't be found


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def split_groups(text: str) -> list[str]:
    return [g.strip() for g in re.split(r",|&|\+|;|\band\b", text, flags=re.IGNORECASE) if g.strip()]


def _groups(text: str, whole) -> list[str]:
    """split_groups(), except that a comma-separated part that is someone's whole name or a saved AI search
    ("salt and pepper") stays in one piece."""
    out: list[str] = []
    for part in re.split(r"[,;]", text):
        part = part.strip()
        if part:
            out += [part] if whole(part) else split_groups(part)
    return out


def search(db: Database, text: str, extra_conditions: list[str] | None = None) -> SearchResult:
    extra = list(extra_conditions or [])
    if not text.strip() and not extra:
        return SearchResult(None)
    people = [p for p in db.people(include_hidden=True) if p.name]
    by_name: dict[str, list[int]] = {}
    for p in people:
        by_name.setdefault(_norm(p.name), []).append(p.id)
    groups = _groups(text, lambda g: _quoted(g) is not None or _norm(g) in by_name
                     or db.ai_search_id(g) is not None or bool(db.ai_tag_matches(g)))

    def whole_words(name: str) -> list[int]:
        """People with these whole words in their name ("dad" finds "Dad Smith"; "car" doesn't find "Carl")."""
        pat = re.compile(rf"\b{re.escape(name)}\b")
        return by_name.get(name) or [p.id for p in people if pat.search(_norm(p.name))]

    def matching(name: str) -> list[int]:
        exact = by_name.get(name)
        return exact or [p.id for p in people if name in _norm(p.name)]

    queries: list[tuple[list[int] | None, list[str]]] = []
    photo_sets: list[set[int]] = []  # AI matches: whole photos, not one face
    for g in groups or [""]:
        literal = _quoted(g)
        if literal is not None:  # "…": the AI's results for exactly this text, nothing else
            ai = _ai_photos(db, literal)
            if ai is None:
                return SearchResult(set(), f"No AI tag or Look for search matches “{literal}”.")
            photo_sets.append(_with_faces(db, ai, extra))
            continue
        whole = by_name.get(_norm(g))
        if whole:  # someone is called exactly this ("Happy", "Sad Sam"): it's them, not an expression
            queries.append((whole, extra))
            continue
        name, conds, _ = attributes.parse_group(g)
        conds = conds + [c for c in extra if c not in conds]
        if not name and conds:  # only expressions, directions or poses: the face models answer those
            queries.append((None, conds))
            continue
        # The whole group as the AI saw it: a Look for search or a tag ("woman in red dress", "smiling woman").
        # Taken before parse_group's leftovers, which lose words like "in" and "with".
        ai = _ai_photos(db, g)
        if ai is not None:
            photo_sets.append(_with_faces(db, ai, extra))
            continue
        if not name:
            continue
        ids = whole_words(name)
        if ids:
            queries.append((ids, conds))
            continue
        ai = _ai_photos(db, name)
        if ai is not None:
            photo_sets.append(_with_faces(db, ai, conds))
            continue
        ids = matching(name)  # part of a name ("Mo" while typing "Mom") comes after the AI's words
        if ids:
            queries.append((ids, conds))
            continue
        # "mom dad smiling" without commas: try each word as its own person.
        words = name.split()
        per_word = [matching(w) for w in words]
        if len(words) > 1 and all(per_word):
            queries += [(ids, conds) for ids in per_word]
            continue
        problem = f"No one named “{name}”. Name people on the People page to search for them."
        if db.ai_stats()["tagged"] or db.ai_stats()["searches"]:
            problem = (f"No one named “{name}”, and no AI tag or AI search matches it. Use Look for… to have the "
                       "AI check your photos for it.")
        return SearchResult(set(), problem)

    result: set[int] | None = None
    for ids, conds in queries:
        found = db.search_faces(ids, conds)
        result = found if result is None else result & found
    for found in photo_sets:
        result = found if result is None else result & found
    return SearchResult(result)


_QUOTES = "\"“”"


def _quoted(group: str) -> str | None:
    """The text inside quotes ("red sweater"), or None when the group isn't quoted."""
    g = group.strip()
    if len(g) >= 2 and g[0] in _QUOTES and g[-1] in _QUOTES:
        return g[1:-1].strip()
    return None


def quote(text: str) -> str:
    """Search text that means exactly this AI tag or Look for search (never a person or an expression)."""
    return f"“{text}”"


def _ai_photos(db: Database, text: str) -> set[int] | None:
    """Photos the AI matched to this text, or None when no AI result knows it. A saved Look for search counts its
    YES answers, plus photos it hasn't checked whose tags contain the words; a NO answer beats a tag."""
    tags = db.ai_tag_matches(text)
    search_id = db.ai_search_id(text)
    if search_id is None:
        return tags or None
    found, checked = db.ai_answered(search_id)
    return found | (tags - checked)


def _with_faces(db: Database, photos: set[int], conds: list[str]) -> set[int]:
    """AI matches narrowed to photos where someone meets the expression/direction/pose conditions."""
    return photos & db.search_faces(None, conds) if conds and photos else photos
