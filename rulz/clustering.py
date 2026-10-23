"""Automatic grouping of faces into people.

Rules that keep the user's corrections intact:
  * Faces assigned by the user (``assigned_by='user'`` with a person) are never moved.
  * A face is never auto-assigned to a person it was rejected from.
  * Faces the user removed from a group may join an *existing* person automatically, but are
    never used to start a brand-new group (otherwise an "ungrouped" cluster would reappear).
"""
from __future__ import annotations

import numpy as np

from .db import Database

CHUNK = 512
MAX_EXEMPLARS = 400  # per-person sample used for suggestions


def cluster(db: Database, threshold: float | None = None) -> dict:
    t_assign = threshold if threshold is not None else db.match_threshold()
    t_link = t_assign + 0.05  # stricter for building new groups, to avoid chaining strangers together

    cand_ids, _, cand = db.embeddings("person_id IS NULL")
    if len(cand_ids) == 0:
        db.cleanup_people()
        return {"assigned": 0, "new_people": 0}
    fresh = {r[0] for r in db.q("SELECT id FROM faces WHERE person_id IS NULL AND assigned_by IS NULL")}
    rejected = db.rejected_pairs()

    # ---- 1. match against existing people ------------------------------------
    assignments: dict[int, int] = {}
    known_ids, known_pids, known = db.embeddings("person_id IS NOT NULL")
    if len(known_ids):
        order = np.argsort(known_pids, kind="stable")
        known_pids, known = known_pids[order], known[order]
        people, starts = np.unique(known_pids, return_index=True)
        centroids = np.add.reduceat(known, starts, axis=0)
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-6)
        col_of = {int(p): j for j, p in enumerate(people)}
        rejected_cols: dict[int, list[int]] = {}
        for fid, pid in rejected:
            if pid in col_of:
                rejected_cols.setdefault(fid, []).append(col_of[pid])
        for lo in range(0, len(cand_ids), CHUNK):
            block = cand[lo:lo + CHUNK]
            cent_sim = block @ centroids.T                                   # [b, P]
            best_single = np.maximum.reduceat(block @ known.T, starts, axis=1)  # [b, P]
            # A face matches a person if it's close to their average face, or very close to one of
            # their faces (helps people whose look varies: glasses, beards, age).
            score = np.maximum(cent_sim, best_single - 0.08)
            for i in range(len(block)):
                fid = int(cand_ids[lo + i])
                s = score[i].copy()
                for j in rejected_cols.get(fid, ()):
                    s[j] = -1
                j = int(np.argmax(s))
                if s[j] < t_assign:
                    continue
                if len(s) > 1:
                    second = np.partition(s, -2)[-2]
                    if s[j] - second < 0.04:  # ambiguous between two people: leave for the user
                        continue
                assignments[fid] = int(people[j])

    # ---- 2. group remaining never-touched faces into new people --------------
    rest = [i for i, fid in enumerate(cand_ids) if int(fid) not in assignments and int(fid) in fresh]
    new_groups: list[list[int]] = []
    if len(rest) >= 2:
        sub = cand[rest]
        parent = list(range(len(rest)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for lo in range(0, len(rest), CHUNK):
            sims = sub[lo:lo + CHUNK] @ sub.T
            rows, cols = np.nonzero(sims >= t_link)
            for r, c in zip(rows, cols):
                a, b = find(lo + int(r)), find(int(c))
                if a != b:
                    parent[a] = b
        comps: dict[int, list[int]] = {}
        for i in range(len(rest)):
            comps.setdefault(find(i), []).append(i)
        for members in comps.values():
            if len(members) < 2:
                continue
            # Drop outliers that only joined through a chain of look-alikes.
            vecs = sub[members]
            cen = vecs.sum(0)
            cen /= max(float(np.linalg.norm(cen)), 1e-6)
            keep = [m for m, s in zip(members, vecs @ cen) if s >= t_assign]
            if len(keep) >= 2:
                new_groups.append([int(cand_ids[rest[m]]) for m in keep])

    # ---- 3. write -------------------------------------------------------------
    with db.transaction():
        live = {r[0] for r in db.q("SELECT id FROM people")}
        n_assigned = 0
        for fid, pid in assignments.items():
            if pid not in live:
                continue
            n_assigned += db.x("UPDATE faces SET person_id=?, assigned_by='auto' "
                               "WHERE id=? AND person_id IS NULL AND ignored=0", (pid, fid)).rowcount
        for group in new_groups:
            pid = db.create_person()
            for fid in group:
                n_assigned += db.x("UPDATE faces SET person_id=?, assigned_by='auto' "
                                   "WHERE id=? AND person_id IS NULL AND ignored=0", (pid, fid)).rowcount
    db.cleanup_people()
    return {"assigned": n_assigned, "new_people": len(new_groups)}


def regroup(db: Database, threshold: float | None = None) -> dict:
    """Undo automatic grouping for unnamed groups and run grouping again.

    Named people and every user correction are kept.
    """
    with db.transaction():
        db.x("UPDATE faces SET person_id=NULL, assigned_by=NULL WHERE assigned_by='auto' AND person_id IN "
             "(SELECT id FROM people WHERE name IS NULL)")
        db.cleanup_people()
    return cluster(db, threshold)


def suggestions(db: Database, person_id: int, limit: int = 200, threshold: float | None = None
                ) -> list[tuple[int, float]]:
    """Unassigned faces that look like ``person_id`` – for the "Is this them?" review."""
    from . import config

    t = threshold if threshold is not None else config.SUGGEST_THRESHOLD
    _, _, mine = db.embeddings("person_id=?", (person_id,))
    if len(mine) == 0:
        return []
    cand_ids, _, cand = db.embeddings("person_id IS NULL")
    if len(cand_ids) == 0:
        return []
    rejected = np.fromiter((r[0] for r in db.q("SELECT face_id FROM rejections WHERE person_id=?", (person_id,))),
                           np.int64)
    cen = mine.sum(0)
    cen /= max(float(np.linalg.norm(cen)), 1e-6)
    if len(mine) > MAX_EXEMPLARS:
        # Comparing against thousands of faces is slow and adds little: an even spread is enough.
        mine = mine[np.linspace(0, len(mine) - 1, MAX_EXEMPLARS).astype(np.int64)]
    score = np.concatenate([np.maximum(block @ cen, (block @ mine.T).max(1) - 0.08)
                            for block in (cand[lo:lo + CHUNK] for lo in range(0, len(cand), CHUNK))])
    keep = (score >= t) & ~np.isin(cand_ids, rejected)
    order = np.argsort(-score[keep], kind="stable")[:limit]
    return [(int(f), float(s)) for f, s in zip(cand_ids[keep][order], score[keep][order])]
