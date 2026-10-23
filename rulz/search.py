"""Photo search by people, facial expression and head direction.

Syntax: comma-separated groups; words in one group describe ONE face.
    "smiling"                 -> photos with anyone smiling
    "Mom smiling"             -> photos where Mom is smiling
    "Mom smiling, Dad"        -> Mom smiling and Dad somewhere in the photo
    "facing left"             -> anyone turned towards the left of the photo
    "Dad looking down, sad"   -> Dad looking down, and someone looking sad
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


def split_groups(text: str) -> list[str]:
    return [g.strip() for g in re.split(r",|&|\+|;|\band\b", text, flags=re.IGNORECASE) if g.strip()]


def search(db: Database, text: str, extra_conditions: list[str] | None = None) -> SearchResult:
    extra = list(extra_conditions or [])
    groups = split_groups(text)
    if not groups and not extra:
        return SearchResult(None)
    people = [p for p in db.people(include_hidden=True) if p.name]

    def matching(name: str) -> list[int]:
        exact = [p.id for p in people if p.name.lower() == name]
        return exact or [p.id for p in people if name in p.name.lower()]

    queries: list[tuple[list[int] | None, list[str]]] = []
    for g in groups or [""]:
        name, conds, _ = attributes.parse_group(g)
        conds = conds + [c for c in extra if c not in conds]
        if not name:
            if conds:
                queries.append((None, conds))
            continue
        ids = matching(name)
        if ids:
            queries.append((ids, conds))
            continue
        # "mom dad smiling" without commas: try each word as its own person.
        words = name.split()
        per_word = [matching(w) for w in words]
        if len(words) > 1 and all(per_word):
            queries += [(ids, conds) for ids in per_word]
            continue
        return SearchResult(set(), f"No one named “{name}”. Name people on the People page to search for them.")

    result: set[int] | None = None
    for ids, conds in queries:
        found = db.search_faces(ids, conds)
        result = found if result is None else result & found
    return SearchResult(result)
