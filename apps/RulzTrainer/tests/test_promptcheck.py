"""The prompt check and the probe prompt presets (rulztrainer.promptcheck): a render's frame from face boxes and pose
landmarks, what a prompt expects, the count against the step-0 probe, and the three places that read it: the likeness
controller's record and best guard, the race's drift drop, and the config's checks."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import likeness as L  # noqa: E402
from rulztrainer import promptcheck as P  # noqa: E402
from rulztrainer import race as R  # noqa: E402
from rulztrainer.config import PRESET_PROBE_PROMPTS, TrainConfig, apply_preset  # noqa: E402

SIZE = (512, 768)


def face(share, size=SIZE, x=200.0, w=100.0):
    """A face box whose height is `share` of the picture's."""
    h = share * size[1]
    return (x, 50.0, x + w, 50.0 + h)


def pose(shoulders=True, hips=True, size=SIZE, vis=0.9):
    """33 landmarks; shoulders / hips inside the picture and visible, or below it and unsure."""
    w, h = size
    pts = [[w / 2, h / 2] for _ in range(33)]
    seen = [vis] * 33
    for idx, inside in ((P.L_SHO, shoulders), (P.R_SHO, shoulders), (P.L_HIP, hips), (P.R_HIP, hips)):
        if not inside:
            pts[idx] = [w / 2, h * 1.2]
            seen[idx] = 0.1
    return pts, seen


FULL = P.frame_of(SIZE, [face(0.18)], pose())
UPPER = P.frame_of(SIZE, [face(0.27)], pose(hips=False))
HEAD = P.frame_of(SIZE, [face(0.45)], pose(hips=False))


# ---------------------------------------------------------------- the frame of one render


def test_frames_from_the_face_size_and_the_hips():
    assert FULL["frame"] == "full" and FULL["face"] == pytest.approx(0.18) and FULL["hips"] is True
    assert UPPER["frame"] == "upper" and HEAD["frame"] == "headshot"
    # a small face without the hips is not a full body; a face over the limit is not one either, hips or not
    assert P.frame_of(SIZE, [face(0.20)], pose(hips=False))["frame"] == "upper"
    assert P.frame_of(SIZE, [face(0.30)], pose())["frame"] == "upper"
    # the thresholds themselves
    assert P.frame_of(SIZE, [face(P.FULL_MAX_FACE)], pose())["frame"] == "full"
    assert P.frame_of(SIZE, [face(P.HEADSHOT_MIN_FACE)], pose())["frame"] == "headshot"
    # one hip is not enough, and a hip the pose model is unsure of does not count
    one = pose()
    one[0][P.R_HIP] = [SIZE[0] / 2, SIZE[1] * 1.3]
    assert P.frame_of(SIZE, [face(0.18)], one)["frame"] == "upper"
    assert P.frame_of(SIZE, [face(0.18)], pose(vis=0.3))["frame"] == "upper"


def test_a_missing_scorer_degrades_and_never_raises():
    # no body scorer, or the pose model found nobody: the face's size alone decides
    assert P.frame_of(SIZE, [face(0.18)], None)["frame"] == "full"
    assert P.frame_of(SIZE, [face(0.28)], None)["frame"] == "upper"
    assert P.frame_of(SIZE, [face(0.5)], None)["frame"] == "headshot"
    # no face scorer (faces None) or no face found ([]): the body decides
    assert P.frame_of(SIZE, None, pose())["frame"] == "full"
    assert P.frame_of(SIZE, [], pose(hips=False))["frame"] == "upper"
    # neither: cannot tell
    nothing = P.frame_of(SIZE, [], None)
    assert nothing["frame"] is None and nothing["faces"] == 0 and nothing["hips"] is None
    assert P.frame_of(SIZE, None, None)["frame"] is None
    assert P.frame_of(SIZE, [], pose(shoulders=False, hips=False))["frame"] is None
    assert P.frame_of(SIZE, [], (None, None))["frame"] is None  # a pose result without landmarks


def test_keeps_and_the_second_person():
    assert P.keeps("full", FULL) is True and P.keeps("full", UPPER) is False and P.keeps("full", HEAD) is False
    assert P.keeps("upper", UPPER) is True and P.keeps("headshot", HEAD) is True and P.keeps("headshot", FULL) is False
    assert P.keeps("", FULL) is None and P.keeps("none", FULL) is None and P.keeps("full", None) is None
    assert P.keeps("full", P.frame_of(SIZE, [], None)) is None  # cannot tell
    # a second face of comparable size never keeps a one-person prompt; a small one in the background does not count
    two = P.frame_of(SIZE, [face(0.18), face(0.15, x=350.0)], pose())
    assert two["people"] == 2 and two["frame"] == "full" and P.keeps("full", two) is False
    far = P.frame_of(SIZE, [face(0.18), face(0.05, x=350.0)], pose())
    assert far["people"] == 1 and far["faces"] == 2 and P.keeps("full", far) is True


@pytest.mark.parametrize("prompt, expect", [
    ("close-up portrait photo of tok, neutral background", "headshot"),
    ("tok, a close-up headshot of a woman", "headshot"),
    ("full body photo of tok standing, whole body visible head to feet", "full"),
    ("tok, a young woman sitting full-body on a bed", "full"),
    ("tok, a waist-up photo of a man at a cafe table", "upper"),
    ("A Full-Length shot of tok", "full"),
    ("tok in a garden", ""),
    ("", ""),
    ("a full-body photo of tok, then a close-up", "full"),  # the earliest match wins
])
def test_guess_expect(prompt, expect):
    assert P.guess_expect(prompt) == expect


def test_resolve_expect():
    assert P.resolve_expect("", "a full body photo") == "full"
    assert P.resolve_expect("none", "a full body photo") == ""
    assert P.resolve_expect("upper", "a full body photo") == "upper"
    assert P.resolve_expect("full", "") == ""  # no prompt, no probe, nothing to check


# ---------------------------------------------------------------- presets


def test_presets_fill_in_the_trigger_and_the_subject():
    keys = [p.key for p in P.PRESETS]
    assert len(keys) == len(set(keys)) and len(P.presets_of("face")) >= 2 and len(P.presets_of("body")) >= 4
    for p in P.PRESETS:
        assert p.kind in P.KINDS and p.expect in P.EXPECTS and "{trigger}" in p.template
        text = P.build(p.key, "tok_anna", "a tall man with a beard")
        assert "tok_anna" in text and "{" not in text
        assert P.guess_expect(text) == p.expect, p.key  # the words say what the preset expects
    long = P.build("full_walking", "tok_anna", "a tall man with a beard")
    assert long.startswith("tok_anna, a full-body photo of a tall man with a beard walking")
    # the fallbacks: no trigger word, no subject
    assert P.build("headshot_neutral").startswith("the character, a close-up headshot of a person ")
    with pytest.raises(ValueError, match="unknown probe prompt preset 'nope'"):
        P.build("nope")


def test_the_subject_falls_back_to_the_isolation_class():
    assert P.subject_of(TrainConfig(probe_subject="  a young   woman ")) == "a young woman"
    cfg = TrainConfig(datasets=[{"path": "a"}, {"path": "b", "class": "a woman"}])  # LoRA Isolation's class
    assert P.subject_of(cfg) == "a woman"
    cfg.probe_subject = "a tall woman"
    assert P.subject_of(cfg) == "a tall woman"
    assert P.subject_of(TrainConfig()) == "a person"


def test_the_identity_recipes_prompts_are_the_classic_presets():
    face_p, body_p = PRESET_PROBE_PROMPTS["identity"]
    assert face_p == "close-up portrait photo of {trigger}, neutral background, soft light"
    assert body_p == P.preset("standing_plain").template and "head to feet" in body_p
    out = apply_preset({"trigger_word": "tok"}, "identity")
    assert out["probe_face_prompt"] == P.build("portrait_plain", "tok")
    assert out["probe_body_prompt"] == P.build("standing_plain", "tok")


def test_checks_of_a_config():
    cfg = TrainConfig(trigger_word="tok", probe_subject="a woman", probe_every=100, face_refs=["a.png"],
                      probe_face_prompt="tok, a close-up headshot", probe_body_prompt="tok in a garden",
                      probe_checks=["full_walking", {"name": "beach", "prompt": "tok on a  beach, full body"},
                                    {"name": "any", "prompt": "tok somewhere", "kind": "face"},
                                    {"preset": "upper_cafe", "name": "cafe2", "expect": "none"}])
    got = P.checks(cfg)
    assert [(c.name, c.kind, c.expect, c.extra) for c in got] == [
        ("face", "face", "headshot", False), ("body", "body", "", False),
        ("full_walking", "body", "full", True), ("beach", "body", "full", True), ("any", "face", "", True),
        ("cafe2", "body", "", True)]
    assert got[2].prompt == P.build("full_walking", "tok", "a woman") and got[3].prompt == "tok on a beach, full body"
    assert P.wanted(cfg) and [c.name for c in P.extra_checks(cfg)] == ["full_walking", "beach", "any", "cafe2"]
    # the body probe's own setting: named, or switched off
    assert P.checks(TrainConfig(probe_body_prompt="tok in a garden", probe_body_expect="upper"))[0].expect == "upper"
    assert not P.wanted(TrainConfig(probe_body_prompt="tok, full body", probe_body_expect="none"))
    assert not P.wanted(TrainConfig())
    # normalizing what was normalized changes nothing (the trainer writes the rows out for LoRA Isolation)
    rows = P.normalize_checks(cfg.probe_checks, "tok", "a woman")
    assert P.normalize_checks(rows, "other", "someone") == rows and rows[3]["expect"] == "none"


@pytest.mark.parametrize("entries, match", [
    (["nope"], "unknown probe prompt preset"),
    ([{"name": "x"}], "needs a prompt"),
    ([{"name": "bad name", "prompt": "p"}], "must be 1-40 letters"),
    ([{"name": "face", "prompt": "p"}], "not 'face' or 'body'"),
    (["full_walking", "full_walking"], "twice"),
    ([{"name": "x", "prompt": "p", "expect": "legs"}], "expect must be one of"),
    ([{"name": "x", "prompt": "p", "kind": "wide"}], "kind must be face or body"),
    ([7], "must be a preset key or a mapping"),
])
def test_bad_extra_prompts_say_what_to_change(entries, match):
    with pytest.raises(ValueError, match=match):
        P.normalize_checks(entries)


# ---------------------------------------------------------------- counting a probe


SPECS = [P.Check("face", "face probe", "face", "p", "headshot"), P.Check("body", "body probe", "body", "p", "full")]


def test_tally_counts_drift_against_the_step0_probe():
    base = {"face": [HEAD] * 4, "body": [FULL, FULL, UPPER, FULL]}  # the base model missed seed 2 of the body probe
    refs = P.reference(SPECS, base)
    assert refs == {"face": [True] * 4, "body": [True, True, False, True]}
    first = P.tally(SPECS, base, None, base=True)
    assert first["base"] and first["drift_share"] is None and first["kept_share"] is None
    assert first["prompts"]["body"]["kept"] == 3 and first["prompts"]["body"]["drifted"] == 1
    assert "3/4 show full (1 do not: left out of later counts)" in P.describe(first)
    # later: seed 0 drifted, seed 2 is left out whatever it shows, seed 3 cannot be told
    now = {"face": [HEAD] * 4, "body": [HEAD, FULL, FULL, P.frame_of(SIZE, [], None)]}
    chk = P.tally(SPECS, now, refs)
    body = chk["prompts"]["body"]
    assert (body["kept"], body["drifted"], body["base_miss"], body["unknown"]) == (1, 1, 1, 1)
    assert body["frames"] == ["headshot", "full", "full", None] and body["drift"] == 0.5
    assert chk["prompts"]["face"]["drift"] == 0.0
    # the share is the prompt that drifted most, not the pool: 4 kept headshots do not halve the body's drift
    assert chk["drift_share"] == 0.5 and chk["kept_share"] == 0.5 and chk["worst"] == "body" and chk["reference"]
    line = P.describe(chk)
    assert "face probe 4/4 kept" in line and "body probe 1/2 kept (1 drifted, 1 the base model did not keep either)" in line
    assert "1 unclear" in line and "drift 50% (the body probe" in line
    # fewer seeds than the step-0 probe rendered: the first ones are compared
    assert P.tally(SPECS, {"body": [FULL, HEAD]}, refs)["prompts"]["body"]["drifted"] == 1


def test_tally_without_a_reference_and_with_nothing_to_check():
    chk = P.tally(SPECS, {"face": [HEAD, FULL], "body": [FULL, FULL]}, None)
    assert not chk["reference"] and chk["drift_share"] == 0.5 and chk["worst"] == "face"
    assert "no step-0 probe to compare with" in P.describe(chk) and "1 not headshot" in P.describe(chk)
    assert P.tally(SPECS, {}, None) == {} and P.describe({}) == ""
    assert P.tally([P.Check("body", "body probe", "body", "p", "")], {"body": [FULL]}, None) == {}
    # every render unclear: nothing counted, no share
    blank = P.tally(SPECS, {"body": [P.frame_of(SIZE, [], None)] * 2}, None)
    assert blank["drift_share"] is None and blank["prompts"]["body"]["unknown"] == 2


# ---------------------------------------------------------------- the likeness controller


def _cfg(**kw):
    base = dict(probe_every=10, probe_ema=0.0, likeness_noise_band=1.0, likeness_normalize=False,
                face_refs=["a.png"], probe_face_prompt="tok, a close-up headshot",
                probe_body_prompt="tok, a full-body photo")
    return TrainConfig(**{**base, **kw})


def _seeds(body_frames, n=4):
    return {"face": [60.0] * n, "body": [50.0] * n, "frames": {"face": [HEAD] * n, "body": list(body_frames)}}


def test_the_controller_records_the_check_and_keeps_the_reference():
    c = L.LikenessController(_cfg())
    rec, _ = c.update(0, 60.0, 50.0, seeds=_seeds([FULL] * 4))
    assert rec.prompt_check["base"] and c.last_drift is None and c.check_ref["body"] == [True] * 4
    rec, _ = c.update(10, 61.0, 50.0, seeds=_seeds([FULL, FULL, FULL, HEAD]))
    assert rec.prompt_check["drift_share"] == 0.25 and c.last_drift == 0.25 and c.last_kept == 0.75
    # a probe without frames (no scorer could read them) records nothing and clears the latest reading
    rec, _ = c.update(20, 61.0, 50.0, seeds={"face": [60.0], "body": [50.0]})
    assert rec.prompt_check == {} and c.last_drift is None
    # a resumed run compares with the same step-0 probe
    again = L.LikenessController(_cfg())
    again.load_resume_state(c.resume_state())
    rec, _ = again.update(30, 61.0, 50.0, seeds=_seeds([HEAD, HEAD, FULL, FULL]))
    assert rec.prompt_check["reference"] and rec.prompt_check["drift_share"] == 0.5
    # a run that never probed at step 0 counts against the prompt alone
    late = L.LikenessController(_cfg())
    rec, _ = late.update(10, 61.0, 50.0, seeds=_seeds([FULL, FULL, HEAD, HEAD]))
    assert not rec.prompt_check["reference"] and rec.prompt_check["drift_share"] == 0.5 and late.check_ref == {}


def test_the_best_guard_holds_a_best_back_while_the_probes_have_drifted():
    c = L.LikenessController(_cfg(likeness_best_min_kept=0.6, likeness_track_best_face=True,
                                  likeness_track_best_body=True))
    c.update(0, 40.0, 40.0, seeds=_seeds([FULL] * 4))
    rec, act = c.update(10, 50.0, 50.0, seeds=_seeds([FULL] * 4))
    assert act.get("save_best") and act.get("save_best_face") and act.get("save_best_body") and not rec.best_held
    assert c.best == 50.0 and c.best_step == 10
    # a higher score from a probe that kept only half its body renders: no new best of any kind
    rec, act = c.update(20, 70.0, 70.0, seeds=_seeds([FULL, FULL, HEAD, HEAD]))
    assert not act.get("save_best") and not act.get("save_best_face") and not act.get("save_best_body")
    assert rec.best_held and c.best == 50.0 and c.best_step == 10 and c.part_best["face"] == (50.0, 10)
    assert c.since_improve == 0  # whether the run still improves is not the guard's business
    # the framing comes back: the best moves again
    rec, act = c.update(30, 72.0, 72.0, seeds=_seeds([FULL, FULL, FULL, HEAD]))
    assert act.get("save_best") and not rec.best_held and c.best_step == 30
    # off by default: the same probes set a best
    off = L.LikenessController(_cfg())
    off.update(0, 40.0, 40.0, seeds=_seeds([FULL] * 4))
    rec, act = off.update(10, 70.0, 70.0, seeds=_seeds([HEAD] * 4))
    assert act.get("save_best") and not rec.best_held and rec.prompt_check["drift_share"] == 1.0


def test_the_best_guard_works_with_any_rise():
    c = L.LikenessController(_cfg(likeness_best_min_kept=0.9, likeness_best_any_rise=True))
    c.update(0, 40.0, 40.0, seeds=_seeds([FULL] * 4))
    rec, act = c.update(10, 40.5, 40.5, seeds=_seeds([FULL, FULL, FULL, HEAD]))
    assert rec.best_held and not act.get("save_best") and c.best == 40.0


# ---------------------------------------------------------------- the race's drift drop


def _round(s, drifts, scores=None, step=100):
    """One round: every racer still in records its probe. Returns the history entries the round added."""
    before = len(s.history)
    for i in list(s.alive):
        if s.done:
            break
        s.record(i, (scores or {}).get(i, 50.0), step, drift=drifts.get(i))
    return s.history[before:]


def test_drift_drop_needs_probes_in_a_row_and_a_clean_one_resets():
    s = R.RaceSchedule(["a", "b", "c"], 0, drop_lowest=False, drift=0.5, drift_probes=2)
    assert _round(s, {0: 0.6, 1: 0.1, 2: 0.0})[0]["left"] == {}  # one probe over the limit: not yet
    _round(s, {0: 0.2, 1: 0.1, 2: 0.0})  # a clean probe resets a's run
    _round(s, {0: 0.6, 1: 0.1, 2: 0.0})
    assert s.alive == [0, 1, 2] and s.drift_run[0] == 1
    entry = _round(s, {0: 0.5, 1: None, 2: 0.0})[0]  # at the threshold counts; None neither adds nor resets
    assert entry["left"] == {"a": "drift"} and entry["drift"] == {"a": 0.5, "b": None, "c": 0.0}
    assert s.alive == [1, 2] and s.drifted == {0} and s.dropped[0] == 4 and 0 not in s.manual
    assert s.state()["drifted"] == ["a"] and s.state()["drift"] == 0.5 and s.state()["drift_probes"] == 2
    assert "a out: its probes left their prompts' frame 2 probes in a row (drift at or over 50%)" in \
        R.describe_round(entry, s)


def test_drift_drop_never_takes_the_last_and_keeps_the_least_drifted():
    s = R.RaceSchedule(["a", "b", "c"], 0, drop_lowest=False, drift=0.5, drift_probes=1)
    entry = _round(s, {0: 0.9, 1: 0.6, 2: 0.75})[0]  # all over it: b drifted least and stays
    assert entry["left"] == {"a": "drift", "c": "drift"} and entry["drift_stay"] == "b"
    assert s.alive == [1] and s.done and "b stays although over the drift limit" in R.describe_round(entry, s)
    # a tie on the drift: the higher score stays
    t = R.RaceSchedule(["a", "b"], 0, drop_lowest=False, drift=0.5, drift_probes=1)
    _round(t, {0: 0.8, 1: 0.8}, scores={0: 40.0, 1: 45.0})
    assert t.alive == [1]
    # two left, one over: it goes and the other is the last
    u = R.RaceSchedule(["a", "b"], 0, drop_lowest=False, drift=0.5, drift_probes=1)
    _round(u, {0: 0.0, 1: 1.0})
    assert u.alive == [0] and u.done and u.drifted == {1}


def test_drift_drop_waits_for_the_warmup_and_is_off_at_zero_and_in_a_relay():
    s = R.RaceSchedule(["a", "b"], 2, drop_lowest=False, drift=0.5, drift_probes=1)
    _round(s, {0: 1.0, 1: 0.0})
    _round(s, {0: 1.0, 1: 0.0})
    assert s.alive == [0, 1]  # warm-up rounds: nobody leaves
    _round(s, {0: 1.0, 1: 0.0})
    assert s.alive == [1]
    off = R.RaceSchedule(["a", "b"], 0, drop_lowest=False)
    for _ in range(4):
        entry = _round(off, {0: 1.0, 1: 1.0})[0]
    assert off.alive == [0, 1] and "drift" not in entry and off.drift_run == {}
    relay = R.RaceSchedule(["a", "b"], 0, follow=True, total_steps=1000, drift=0.5, drift_probes=1)
    _round(relay, {0: 1.0, 1: 0.0})
    assert relay.drift == 0.0 and relay.alive == [0, 1]


def test_drift_drop_with_drop_lowest_drops_one_reason_a_round():
    s = R.RaceSchedule(["a", "b", "c"], 0, drop_lowest=True, drift=0.5, drift_probes=1)
    entry = _round(s, {0: 0.9, 1: 0.0, 2: 0.0}, scores={0: 60.0, 1: 40.0, 2: 50.0})[0]
    # a leaves for drift although it scored highest; the lowest (b) is not dropped in the same round
    assert entry["left"] == {"a": "drift"} and entry["dropped"] is None and s.alive == [1, 2]
    entry = _round(s, {1: 0.0, 2: 0.0}, scores={1: 40.0, 2: 50.0})[0]
    assert entry["dropped"] == "b" and s.alive == [2]  # the next round drops the lowest as usual
    # and without any drift the lowest goes every round
    plain = R.RaceSchedule(["a", "b", "c"], 0, drop_lowest=True, drift=0.5, drift_probes=2)
    assert _round(plain, {0: 0.0, 1: 0.0, 2: 0.0}, scores={0: 60.0, 1: 40.0, 2: 50.0})[0]["dropped"] == "b"


def test_a_drifted_configuration_cannot_win_on_its_best_score():
    s = R.RaceSchedule(["a", "b"], 0, drop_lowest=False, drift=0.5, drift_probes=1)
    s.record(0, 60.0, 100, best=60.0, best_step=100, drift=1.0)
    s.record(1, 40.0, 100, best=40.0, best_step=100, drift=0.0)
    assert s.alive == [1] and s.keeps_all
    assert s.decide({0: 60.0, 1: 41.0}) == 1  # a had the higher best and is out all the same
    # patience drops still let the best score win
    p = R.RaceSchedule(["a", "b"], 0, patience=1)
    p.record(0, 60.0, 100, best=60.0, best_step=100, since_improve=1)
    assert p.alive == [1] and p.decide({0: 60.0, 1: 41.0}) == 0


# ---------------------------------------------------------------- the config


def test_config_checks_and_yaml_round_trip(tmp_path):
    ok = TrainConfig(dataset_dir=".", probe_every=100, face_refs=["a.png"], trigger_word="tok",
                     probe_face_prompt="tok, a close-up headshot", probe_body_prompt="tok, a full-body photo",
                     probe_subject="a woman", probe_body_expect="full", likeness_best_min_kept=0.5,
                     probe_checks=["full_walking", {"name": "beach", "prompt": "tok on a beach", "expect": "upper"}],
                     probe_check_seeds=2, race_drop_drift=0.5, race_drop_drift_probes=3)
    ok.validate_prompt_check()
    path = tmp_path / "c.yaml"
    ok.save(path)
    back = TrainConfig.load(path)
    for key in ("probe_subject", "probe_face_expect", "probe_body_expect", "probe_checks", "probe_check_seeds",
                "likeness_best_min_kept", "race_drop_drift", "race_drop_drift_probes"):
        assert getattr(back, key) == getattr(ok, key), key
    # the defaults are off
    d = TrainConfig()
    assert (d.probe_checks, d.likeness_best_min_kept, d.race_drop_drift, d.probe_face_expect) == ([], 0.0, 0.0, "")
    d.validate_prompt_check()


@pytest.mark.parametrize("kw, match", [
    (dict(probe_body_expect="legs"), "probe_body_expect must be empty"),
    (dict(probe_check_seeds=0), "probe_check_seeds must be 1 to 16"),
    (dict(likeness_best_min_kept=1.5), "likeness_best_min_kept must be 0"),
    (dict(race_drop_drift=-0.1), "race_drop_drift must be 0"),
    (dict(race_drop_drift_probes=0), "race_drop_drift_probes must be 1 or more"),
    (dict(probe_checks=["nope"]), "unknown probe prompt preset 'nope'"),
    (dict(probe_checks=["full_walking"]), "probe_checks are rendered with the likeness probes"),
    (dict(likeness_best_min_kept=0.5, probe_every=100, face_refs=["a.png"], probe_body_prompt="tok in a garden"),
     "likeness_best_min_kept read the prompt check, which has nothing to check"),
    (dict(likeness_best_min_kept=0.5, probe_body_prompt="tok, a full-body photo"),
     "likeness_best_min_kept read the prompt check"),  # a frame to check, but no probes
    (dict(race=True, race_drop_drift=0.5, probe_every=100, face_refs=["a.png"], probe_body_prompt="tok, full body",
          probe_body_expect="none"), "race_drop_drift read the prompt check"),
])
def test_config_says_what_is_wrong(kw, match):
    with pytest.raises(ValueError, match=match):
        TrainConfig(**kw).validate_prompt_check()


def test_race_settings_outside_a_race_are_ignored_and_a_relay_refuses_the_drift_drop():
    # a race's winner config keeps the race's settings with race off: it must still load and validate
    TrainConfig(race_drop_drift=0.5, probe_body_prompt="tok in a garden").validate_prompt_check()
    rows = [{"name": "s1", "seed": 1}, {"name": "s2", "seed": 2}]
    cfg = TrainConfig(race=True, race_configs=rows, race_follow=True, race_drop_drift=0.5, probe_every=10, steps=40,
                      face_refs=["a.png"], probe_body_prompt="tok, a full-body photo")
    with pytest.raises(ValueError, match="a relay race .* keeps every configuration"):
        R.validate(cfg)
    cfg.race_follow = False
    R.validate(cfg)


# ---------------------------------------------------------------- the trainer's probe: jobs, frames, the record


class _Status:
    def __init__(self):
        self.data = {}

    def update(self, **kw):
        self.data.update(kw)


class _Face:
    """A face scorer whose boxes come from a table by file name; `calls` counts detections."""
    gold = None

    def __init__(self, shares):
        self.shares, self.calls = shares, 0

    def boxes(self, path):
        self.calls += 1
        share = self.shares[Path(path).name]
        if isinstance(share, Exception):
            raise share
        return [] if share is None else [face(share)]

    def score(self, path, step, index, gold=False):
        from types import SimpleNamespace

        return SimpleNamespace(percent=60.0, gold_percent=None)


class _Body:
    def __init__(self, hips):
        self.hips, self.calls = hips, 0

    def landmarks(self, path):
        import numpy as np

        self.calls += 1
        seen = self.hips[Path(path).name]
        if seen is None:
            return None
        pts, vis = pose(hips=seen)
        return np.asarray(pts, dtype=np.float32), np.asarray(vis, dtype=np.float32)

    def score(self, path):
        return 50.0, 5


def _trainer(tmp_path, face_scorer, body_scorer, **cfg):
    from rulztrainer.trainer import Trainer

    tr = Trainer.__new__(Trainer)
    base = dict(probe_every=50, probe_seeds=2, probe_floor_seeds=2, trigger_word="tok", face_refs=["a.png"],
                probe_face_prompt="tok, a close-up headshot", probe_body_prompt="tok, a full-body photo")
    tr.cfg = TrainConfig(**{**base, **cfg})
    tr.face_scorer, tr.body_scorer, tr.race = face_scorer, body_scorer, None
    tr.job_dir, tr.status = tmp_path, _Status()
    (tmp_path / "probes").mkdir(exist_ok=True)
    return tr


def _render(jobs):
    """Stand-ins for the rendered images: blank files of each job's size."""
    from PIL import Image

    paths = []
    for j in jobs:
        Image.new("RGB", SIZE).save(j.out_path)
        paths.append(str(j.out_path))
    return paths


def test_probe_jobs_add_the_extra_prompts_and_leave_tagged_probes_alone(tmp_path):
    tr = _trainer(tmp_path, _Face({}), _Body({}), probe_checks=["full_walking", "headshot_outdoor"],
                  probe_check_seeds=3)
    jobs = tr._probe_jobs(100)
    assert [j.kind for j in jobs] == ["face"] * 2 + ["body"] * 2 + ["check:full_walking"] * 3 + ["check:headshot_outdoor"] * 3
    walk = [j for j in jobs if j.kind == "check:full_walking"]
    assert [j.out_path.name for j in walk] == [f"step000100_check-full_walking_{k}.png" for k in range(3)]
    assert walk[0].prompt == P.build("full_walking", "tok") and [j.seed for j in walk] == [1234, 1235, 1236]
    body = next(j for j in jobs if j.kind == "body")
    assert (walk[0].width, walk[0].height) == (body.width, body.height)  # a body-kind prompt renders 2:3
    square = next(j for j in jobs if j.kind == "check:headshot_outdoor")
    assert square.width == square.height == tr.cfg.probe_size
    # the soup's base, the noise check: the likeness prompts alone
    assert {j.kind for j in tr._probe_jobs(0, tag="soup_base")} == {"face", "body"}
    assert {j.kind for j in tr._probe_jobs(0, n_seeds=5)} == {"face", "body"}
    # without any scorer nothing could read the renders: they are not made
    tr.face_scorer = tr.body_scorer = None
    assert tr._probe_jobs(100) == []


def test_score_probe_reads_the_frames_and_never_fails_on_them(tmp_path):
    names = {"step000100_face_0.png": 0.5, "step000100_face_1.png": 0.2, "step000100_body_0.png": 0.18,
             "step000100_body_1.png": 0.4, "step000100_check-full_walking_0.png": 0.18,
             "step000100_check-full_walking_1.png": RuntimeError("detector crashed")}
    hips = {n: True for n in names}
    hips["step000100_body_1.png"] = False
    fs, bs = _Face(names), _Body(hips)
    tr = _trainer(tmp_path, fs, bs, probe_checks=["full_walking"], probe_check_seeds=2)
    jobs = tr._probe_jobs(100)
    images = _render(jobs)
    face_score, body_score, seeds = tr._score_probe(jobs, images, 100)
    assert (face_score, body_score) == (60.0, 50.0) and seeds["face"] == [60.0, 60.0] and seeds["body"] == [50.0, 50.0]
    frames = seeds["frames"]
    assert [f["frame"] for f in frames["face"]] == ["headshot", "full"]
    assert [f["frame"] for f in frames["body"]] == ["full", "headshot"]
    assert frames["full_walking"][0]["frame"] == "full" and frames["full_walking"][1] is None  # the crash: unclear
    assert bs.calls == 3  # a headshot needs the face alone: no pose is looked for on the face probe's renders
    # the record: the face probe lost one of two, the extra prompt has one clear render
    ctl = L.LikenessController(tr.cfg)
    rec, _ = ctl.update(100, face_score, body_score, images, seeds)
    assert rec.prompt_check["prompts"]["face"]["drifted"] == 1 and rec.prompt_check["prompts"]["full_walking"]["unknown"] == 1
    # a probe that is not checked (the soup's base, the noise check) reads nothing
    fs.calls = 0
    assert "frames" not in tr._score_probe(jobs[:4], images[:4], 0, check=False)[2] and fs.calls == 0
    # no expectation, no reading: a prompt whose frame is unknown costs no detection
    quiet = _trainer(tmp_path, _Face(names), _Body(hips), probe_face_prompt="tok at home",
                     probe_body_prompt="tok in a garden")
    qjobs = quiet._probe_jobs(100)
    assert "frames" not in quiet._score_probe(qjobs, _render(qjobs), 100)[2] and quiet.face_scorer.calls == 0


def test_the_frame_reader_degrades_with_one_scorer(tmp_path):
    from PIL import Image

    img = tmp_path / "x.png"
    Image.new("RGB", SIZE).save(img)
    both = _trainer(tmp_path, _Face({"x.png": 0.18}), _Body({"x.png": True}))
    assert both._probe_frame(img, "full")["frame"] == "full"
    face_only = _trainer(tmp_path, _Face({"x.png": 0.18}), None)
    assert face_only._probe_frame(img, "full")["frame"] == "full" and face_only._probe_frame(img, "full")["hips"] is None
    body_only = _trainer(tmp_path, None, _Body({"x.png": False}))
    assert body_only._probe_frame(img, "full")["frame"] == "upper"
    assert _trainer(tmp_path, None, None)._probe_frame(img, "full") is None
    # the pose model found nobody: the face decides; neither found anything: cannot tell
    assert _trainer(tmp_path, _Face({"x.png": 0.3}), _Body({"x.png": None}))._probe_frame(img, "full")["frame"] == "upper"
    assert _trainer(tmp_path, _Face({"x.png": None}), _Body({"x.png": None}))._probe_frame(img, "full")["frame"] is None
    # an anchored body scorer keeps its pose model in .pose
    from types import SimpleNamespace

    anchored = _trainer(tmp_path, None, SimpleNamespace(pose=_Body({"x.png": True})))
    assert anchored._probe_frame(img, "full")["frame"] == "full"


def test_the_face_scorer_hands_back_the_boxes_of_the_image_it_scored():
    from types import SimpleNamespace

    import numpy as np

    from rulztrainer.facesim import FaceScorer

    fs = FaceScorer.__new__(FaceScorer)
    fs.refs, fs.calls = [], 0

    def faces(path):
        fs.calls += 1
        return [SimpleNamespace(bbox=np.array([10.0, 20.0, 110.0, 220.0], dtype=np.float32))]

    fs.faces = faces
    assert fs.score("a.png").faces == 1 and fs.calls == 1
    assert fs.boxes("a.png") == [(10.0, 20.0, 110.0, 220.0)] and fs.calls == 1  # no second detection
    assert fs.boxes("b.png") == [(10.0, 20.0, 110.0, 220.0)] and fs.calls == 2  # another image: detected
