"""Grouping, search and storage rules that protect the user's corrections."""
import sqlite3

import numpy as np
import pytest

from rulz import clustering, search
from rulz.faces import DetectedFace, FaceAttrs


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _library(db, n):
    db.x("INSERT INTO folders(id, path) VALUES(1, 'x')")
    db.conn.executemany("INSERT INTO photos(id, folder_id, path, faces_done) VALUES(?,1,?,1)",
                        [(i + 1, f"p{i}.jpg") for i in range(n)])


def _face(db, photo_id, emb, person=None, by=None):
    fid = db.insert_face(photo_id, DetectedFace((0.1, 0.1, 0.3, 0.3), 0.9, _unit(emb), FaceAttrs()))
    if person is not None or by is not None:
        db.x("UPDATE faces SET person_id=?, assigned_by=? WHERE id=?", (person, by, fid))
    return fid


def _reference_components(vecs, t):
    """Plain union-find over every linked pair: what the vectorized version must reproduce."""
    parent = list(range(len(vecs)))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    sims = vecs @ vecs.T
    for r, c in zip(*np.nonzero(sims >= t)):
        a, b = find(int(r)), find(int(c))
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups = {}
    for i in range(len(vecs)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


@pytest.mark.parametrize("seed", range(4))
def test_components_match_union_find(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(12, 16)))
    vecs = _unit(np.concatenate([c + rng.normal(scale=0.35, size=(rng.integers(1, 30), 16)) for c in centers]
                                + [rng.normal(size=(40, 16))]))
    monkeypatch.setattr(clustering, "BLOCK_CELLS", 64 * len(vecs))  # several blocks, even for this size
    label = clustering._components(vecs, 0.6)
    got = {}
    for i, lab in enumerate(label):
        got.setdefault(int(lab), []).append(i)
    assert all(lab == members[0] for lab, members in got.items())  # label = smallest member
    assert sorted(got.values()) == _reference_components(vecs, 0.6)


def test_rejection_made_while_grouping_is_kept(db, monkeypatch):
    _library(db, 3)
    mom = db.create_person("Mom")
    e = np.ones(128)
    _face(db, 1, e, mom, "user")
    _face(db, 2, e, mom, "user")
    cand = _face(db, 3, e)
    real = db.rejected_pairs

    def read_then_user_says_no():
        pairs = real()
        db.reject_faces([cand], mom)  # the UI writes on its own connection while grouping computes
        return pairs

    monkeypatch.setattr(db, "rejected_pairs", read_then_user_says_no)
    clustering.cluster(db)
    assert db.one("SELECT person_id FROM faces WHERE id=?", (cand,))[0] is None


def test_face_touched_while_grouping_does_not_start_a_group(db, monkeypatch):
    _library(db, 2)
    e = np.ones(128)
    a, b = _face(db, 1, e), _face(db, 2, e)
    real = db.rejected_pairs

    def read_then_user_unassigns():
        pairs = real()
        db.x("UPDATE faces SET assigned_by='user' WHERE id=?", (a,))
        return pairs

    monkeypatch.setattr(db, "rejected_pairs", read_then_user_unassigns)
    r = clustering.cluster(db)
    assert db.one("SELECT person_id FROM faces WHERE id=?", (a,))[0] is None
    # b alone is not a group: no one-face person is made, and none is counted.
    assert db.one("SELECT person_id FROM faces WHERE id=?", (b,))[0] is None
    assert r["new_people"] == 0 and db.one("SELECT COUNT(*) FROM people")[0] == 0


def test_grouping_keeps_corrections(db):
    _library(db, 8)
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=128), rng.normal(size=128)
    near = lambda c: c + rng.normal(scale=0.2, size=128)  # noqa: E731
    mom = db.create_person("Mom")
    _face(db, 1, near(x), mom, "user")
    _face(db, 2, near(x), mom, "user")
    rejected = _face(db, 3, near(x))
    db.reject_faces([rejected], mom)
    joins = _face(db, 4, near(x))
    ungrouped = [_face(db, 5, near(y), None, "user"), _face(db, 6, near(y), None, "user")]
    fresh = [_face(db, 7, near(y)), _face(db, 8, near(y))]
    r = clustering.cluster(db)
    person = lambda f: db.one("SELECT person_id FROM faces WHERE id=?", (f,))[0]  # noqa: E731
    assert person(joins) == mom
    assert person(rejected) is None
    assert all(person(f) is None for f in ungrouped)  # never re-form a dissolved group
    assert person(fresh[0]) == person(fresh[1]) is not None and r["new_people"] == 1


def test_search_finds_people_named_like_an_expression(db):
    _library(db, 3)
    happy = db.create_person("Happy")
    sam = db.create_person("Sad  Sam")
    _face(db, 1, np.ones(128), happy, "user")
    _face(db, 2, np.ones(128), sam, "user")
    smiling = _face(db, 3, np.ones(128))
    db.x("UPDATE faces SET expression='happy' WHERE id=?", (smiling,))
    assert search.search(db, "happy").photo_ids == {1}
    assert search.search(db, "sad sam").photo_ids == {2}
    assert search.search(db, "smiling").photo_ids == {3}


def test_failed_commit_does_not_leave_a_transaction_open(db):
    _library(db, 1)
    with pytest.raises(sqlite3.IntegrityError):
        with db.transaction():
            db.x("PRAGMA defer_foreign_keys=ON")  # the foreign key is checked at COMMIT, which then fails
            db.x("INSERT INTO faces(photo_id, x, y, w, h) VALUES(999, 0, 0, 1, 1)")
    assert not db.conn.in_transaction
    with db.transaction():  # later writes are saved again, not swallowed by a transaction left open
        db.set_setting("after", 1)
    assert db.get_setting("after") == "1"
