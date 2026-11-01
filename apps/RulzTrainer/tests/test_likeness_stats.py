"""Probe noise (rulztrainer.likeness_stats): the Student t tail, paired tests on shared probe seeds, the seeds a probe
needs, a run's noise from its likeness.jsonl, the noise check's report and stage, per-seed scores in probe records, and
the race's paired drop rule (race_drop_confidence)."""
import contextlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import likeness_stats as S  # noqa: E402
from rulztrainer import race as R  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.likeness import LikenessController  # noqa: E402


def test_t_tail_matches_scipy():
    # scipy.stats.t.sf at these points (scipy is optional here, so the values are written out)
    for t, df, want in [(0.5, 2, 0.33333333), (2.1, 5, 0.04487662), (-1.3, 9, 0.88704681), (3.0, 1, 0.10241638),
                        (0.0, 4, 0.5)]:
        assert S.t_sf(t, df) == pytest.approx(want, abs=1e-7)


def test_paired_test_uses_the_seeds_both_probes_scored():
    st = S.paired([10, 12, 9, 11], [8, 11, 7, 10.5])
    assert st["n"] == 4 and st["diff"] == pytest.approx(1.375) and st["sd_diff"] == pytest.approx(0.75)
    assert st["p"] == pytest.approx(0.0175408, abs=1e-6)  # scipy.stats.ttest_rel(..., alternative="greater")
    assert st["rho"] > 0.9  # the seeds that are hard for one probe are hard for the other
    # a seed either probe left unscored drops out of the pairs; under 3 pairs there is no test
    assert S.paired([10, None, 9], [8, 11, None])["n"] == 1 and S.paired([10, None, 9], [8, 11, None])["p"] is None
    assert S.paired([5, 5, 5], [4, 4, 4])["p"] == 0.0 and S.paired([4, 4, 4], [4, 4, 4])["p"] == 1.0
    # pairing is what makes a small gap visible: the same scores unpaired spread far more than their differences
    a = [40, 70, 55, 62, 48, 66]
    b = [x - 3 for x in a]
    assert S.paired(a, b)["p"] == 0.0 and S.mean_sd(a)[1] > 10


def test_seeds_needed_and_the_noise_band():
    sd = 6 * math.sqrt(2 * 0.7)  # sigma 6, rho 0.3: the report's example
    assert [S.seeds_needed(sd, g) for g in (3, 5, 8)] == [26, 10, 4]
    assert S.seeds_needed(1.0, 8) == 3  # never below a t-test's least
    # one probe seed, no smoothing: the band is the per-seed difference's spread, to half a point
    assert S.suggested_band(4.0, 1, 0.0) == 4.0
    assert S.suggested_band(4.0, 4, 0.5) == pytest.approx(round(4.0 / 2 * math.sqrt(1 / 3) * 2) / 2)


def test_noise_report_pairs_the_files_on_their_seeds():
    files = {"mid": {"combined": [50, 60, 55, 40, 70], "face": [50, 60, 55, 40, 70], "body": []},
             "best": {"combined": [58, 66, 60, 47, 77], "face": [58, 66, 60, 47, 77], "body": []}}
    rep = S.noise_report(files, probe_seeds=2, probe_ema=0.5)
    comb = rep["kinds"]["combined"]
    assert comb["sigma"] == pytest.approx(11.1, abs=0.1) and comb["rho"] > 0.99
    assert comb["sd_diff"] == pytest.approx(1.14, abs=0.01) and comb["sd_diff_from"] == "pairs"
    assert comb["pairs"]["best - mid"]["diff"] == pytest.approx(6.6) and comb["pairs"]["best - mid"]["p"] < 0.001
    assert comb["seeds_needed"] == {"3": 3, "5": 3, "8": 3} and rep["suggested_noise_band"] == 0.5
    assert rep["kinds"]["body"]["sd_diff"] is None
    # one file: no pairs, so the difference is assumed to spread as two independent seeds do
    one = S.noise_report({"only": files["mid"]}, 1, 0.0)["kinds"]["combined"]
    assert one["sd_diff"] == pytest.approx(one["sigma"] * math.sqrt(2)) and one["rho"] is None
    assert any("likeness_noise_band 0.5" in line for line in S.describe(rep))


def test_probe_records_keep_every_seed(tmp_path):
    cfg = TrainConfig(probe_every=10, probe_ema=0.5, face_weight=0.7, body_weight=0.3, face_scale_span=1.0,
                      face_scale_curve=1.0, body_scale_span=1.0, body_scale_curve=1.0)
    c = LikenessController(cfg, face_ceiling=60.0, body_ceiling=90.0)
    c.update(0, 20.0, 80.0, [], {"face": [18.0, 22.0], "body": [80.0, 80.0]})
    rec, _ = c.update(10, 40.0, None, [], {"face": [30.0, None, 50.0], "body": [None, None, None]})
    assert rec.face_seeds == [30.0, None, 50.0] and rec.body_seeds == [None, None, None]
    assert rec.face_norm_seeds == [25.0, None, 75.0]  # floor 20, ceiling 60, linear
    assert rec.combined_seeds == [25.0, None, 75.0] and c.last_seeds == rec.combined_seeds  # the face alone
    assert c.update(20, 40.0, 85.0)[0].combined_seeds == [] and c.last_seeds == []  # an old caller: no seeds
    rows = [{"step": 0, "combined_seeds": [0.0, 0.0]},
            {"step": 10, "combined_seeds": [40.0, 50.0], "face_norm_seeds": [40.0, 50.0]},
            {"step": 20, "combined_seeds": [45.0, 53.0], "face_norm_seeds": [45.0, 53.0]},
            {"step": 30, "combined_seeds": [47.0, 52.0, 60.0], "face_norm_seeds": [47.0, 52.0, 60.0]}]
    noise = S.run_noise(rows)
    assert noise["combined"]["probes"] == 3 and noise["combined"]["sigma"] > 0  # step 0 left out
    assert noise["combined"]["sd_diff"] is not None and noise["body"]["sigma"] is None
    (tmp_path / "likeness.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert S.main([str(tmp_path)]) == 0


def _round(s, scores, seeds):
    for i, (score, sd) in enumerate(zip(scores, seeds)):
        if i in s.alive:
            s.record(i, score, step=None, seeds=sd)


def test_the_paired_rule_drops_only_a_racer_the_seeds_agree_is_behind():
    s = R.RaceSchedule(["a", "b", "c"], warmup=0, confidence=0.9, max_close=3)
    base = [40.0, 55.0, 62.0, 48.0]
    # c is 6 points behind on every seed: dropped at once, against the leader a
    _round(s, [60.0, 58.0, 52.0], [[x + 8 for x in base], [x + 6 for x in base], [x + 2 for x in base]])
    assert s.history[-1]["dropped"] == "c" and s.history[-1]["test"]["why"] == "paired"
    assert s.history[-1]["test"]["vs"] == "a" and s.history[-1]["test"]["p"] < 0.1
    assert "below a by 6.0 points over 4 paired seeds" in R.describe_round(s.history[-1], s)
    # a and b swap places seed by seed: too close to call, nobody leaves
    _round(s, [60.0, 59.5], [[50, 61, 44, 70], [52, 58, 46, 69]])
    assert s.history[-1]["dropped"] is None and s.history[-1]["close"]["lowest"] == "b" and s.close == 1
    assert "too close to call" in R.describe_round(s.history[-1], s) and s.alive == [0, 1]
    # the next rounds pool their seeds with this one's; still a coin toss, so the third close round drops b anyway
    _round(s, [60.0, 59.5], [[51, 60, 45, 69], [50, 61, 44, 70]])
    _round(s, [60.0, 59.5], [[50, 61, 44, 70], [51, 60, 45, 69]])
    assert s.close == 3 and s.alive == [0, 1]
    _round(s, [60.0, 59.5], [[50, 61, 44, 70], [50, 61, 44, 70]])
    assert s.done and s.history[-1]["test"]["why"] == "close" and s.state()["winner"] == "a"


def test_the_paired_rule_ends_the_race_once_the_steps_are_used_up_and_is_off_by_default():
    s = R.RaceSchedule(["a", "b"], warmup=0, total_steps=20, confidence=0.9, max_close=5)
    s.record(0, 50.0, step=10, seeds=[50, 51])
    s.record(1, 49.0, step=10, seeds=[49, 52])
    assert not s.done and s.history[-1]["close"]["p"] is None  # 2 seeds: no test yet
    s.record(0, 50.0, step=20, seeds=[50, 51])
    s.record(1, 49.0, step=20, seeds=[49, 52])
    assert s.done and s.history[-1]["test"]["why"] == "steps"
    off = R.RaceSchedule(["a", "b"], warmup=0)  # confidence 0: the lowest leaves on its smoothed score
    off.record(0, 50.0, seeds=[50, 51])
    assert off.record(1, 49.9, seeds=[52, 53]) == 1 and "test" not in off.history[-1]


def test_race_validates_the_confidence():
    rows = [{"name": "a"}, {"name": "b", "learning_rate": 2e-4}]
    base = dict(race=True, probe_every=10, face_refs=["ref.png"], steps=40, race_configs=rows)
    R.validate(TrainConfig(**base, race_drop_confidence=0.9))
    with pytest.raises(ValueError, match="race_drop_confidence"):
        R.validate(TrainConfig(**base, race_drop_confidence=1.0))
    with pytest.raises(ValueError, match="race_max_close_rounds"):
        R.validate(TrainConfig(**base, race_max_close_rounds=-1))


def test_noise_files_validation():
    ok = dict(face_refs=["r.png"], probe_every=10, probe_face_prompt="a face", steps=0, noise_files=["x.safetensors"])
    TrainConfig(**ok).validate()
    with pytest.raises(ValueError, match="runs alone"):
        TrainConfig(**{**ok, "steps": 100}).validate()
    with pytest.raises(ValueError, match="noise_seeds"):
        TrainConfig(**{**ok, "noise_seeds": 2}).validate()
    with pytest.raises(ValueError, match="measures the likeness probes"):
        TrainConfig(**{**ok, "face_refs": []}).validate()


def test_the_noise_stage_probes_every_file_on_the_same_seeds(tmp_path):
    from rulztrainer import archs
    from rulztrainer import trainer as T

    cfg = TrainConfig(job_name="job", noise_files=["a/mid.safetensors", "b/best.safetensors"], noise_seeds=5,
                      face_weight=1.0, body_weight=0.0, probe_seeds=2, probe_ema=0.5, face_scale_span=1.0,
                      face_scale_curve=1.0)
    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg, tr.arch, tr.job_dir, tr.devices = cfg, archs.arch_by_name("moss_image"), tmp_path, ["cpu"]
    status = {}
    tr.status = SimpleNamespace(update=lambda **k: status.update(k), data={"saved": []})
    tr.controller = LikenessController(cfg, face_ceiling=70.0)
    tr._free_for_rendering = lambda: None
    tr._sampling_slots = lambda model, step: contextlib.nullcontext([])
    tr._probe_with = lambda slots, step, tag="", n_seeds=None: (20.0, None, [], {"face": [20.0] * 4, "body": []})
    calls = []

    def probe(model, path, tag, n_seeds=None):
        calls.append((path, n_seeds))
        bump = 5.0 if "best" in path else 0.0
        face = [40.0 + bump, 55.0 + bump, None, 45.0 + bump, 60.0 + bump][:n_seeds]
        return None, None, {"face": face, "body": []}

    tr._soup_probe = probe
    tr._noise_stage(None)
    assert calls == [("a/mid.safetensors", 5), ("b/best.safetensors", 5)]
    rep = json.loads((tmp_path / "noise" / "noise.json").read_text())
    comb = rep["kinds"]["combined"]
    assert comb["pairs"]["best.safetensors - mid.safetensors"]["n"] == 4  # the unscored seed is left out
    assert comb["sd_diff"] == pytest.approx(0.0, abs=1e-9) and comb["sigma"] > 10
    assert rep["seeds"] == list(range(cfg.probe_seed, cfg.probe_seed + 5)) and rep["raw"]["mid.safetensors"]["face"][2] is None
    assert status["done"] and status["noise"]["suggested_noise_band"] == rep["suggested_noise_band"]
