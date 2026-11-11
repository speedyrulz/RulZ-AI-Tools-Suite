"""Relay race with race_pick: each round ends waiting for you to pick the racer everyone continues from; no pick in
time (or a stop) leaves the lead to the highest score."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import race as R  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


def test_a_round_waits_for_the_pick_and_the_pick_leads():
    s = R.RaceSchedule(["a", "b", "c"], 0, total_steps=20, follow=True, pick=True)
    for i, score in enumerate((10.0, 30.0, 20.0)):
        s.record(i, score, 10)
    assert s.awaiting == 1 and s.history[-1]["suggested"] == "b" and s.state()["awaiting"] == "b"
    assert "waiting for your pick (b scored highest)" in R.describe_round(s.history[-1], s)
    assert s.choose(0) == 0 and s.leader == 0 and s.awaiting is None and s.state()["leader"] == "a"
    assert "a leads (your pick; b scored highest)" in R.describe_round(s.history[-1], s)
    with pytest.raises(ValueError, match="waiting"):
        s.choose(1)
    assert s.pending == [0, 1, 2] and not s.done


def test_the_last_round_waits_too_and_the_pick_wins():
    s = R.RaceSchedule(["a", "b"], 0, total_steps=10, follow=True, pick=True)
    s.record(0, 50.0, 10)
    s.record(1, 40.0, 10)
    assert not s.done and s.just_dropped == [] and s.history[-1]["final"]  # nobody leaves before the pick
    s.choose(1)
    assert s.done and s.alive == [1] and s.just_dropped == [0] and s.finished == {0} and s.dropped == {0: 1}
    assert s.state()["winner"] == "b" and "b wins the last round (your pick; a scored highest)" in \
        R.describe_round(s.history[-1], s)


def test_no_pick_leaves_the_highest_score_and_pick_needs_a_relay():
    s = R.RaceSchedule(["a", "b"], 0, total_steps=20, follow=True, pick=True)
    s.record(0, 10.0, 10)
    s.record(1, 30.0, 10)
    assert s.choose(None, "waited") == 1 and "highest score: no pick in time" in R.describe_round(s.history[-1], s)
    assert not R.RaceSchedule(["a", "b"], 0, pick=True).pick  # not a relay: nothing to pick
    with pytest.raises(ValueError, match="race_pick_minutes"):
        TrainConfig(race=True, race_follow=True, race_pick=True, race_pick_minutes=-1, probe_every=10, steps=20,
                    face_refs=["ref.png"], race_configs=[{"name": "a"}, {"name": "b"}]).validate()


def fake_trainer(tmp_path, sched, minutes=0.0):
    from rulztrainer import trainer as T

    variants = [SimpleNamespace(name=n, idx=i, alive=True, state={}) for i, n in enumerate(sched.names)]
    calls = []
    fake = SimpleNamespace(race=SimpleNamespace(schedule=sched, variants=variants, active=variants[0],
                                                base_cfg=SimpleNamespace(race_pick_minutes=minutes)),
                           job_dir=tmp_path, stop_file=tmp_path / "STOP",
                           status=SimpleNamespace(update=lambda **kw: calls.append(("status", kw.get("stage")))))
    fake._race_record = lambda v, c: T.Trainer._race_record(fake, v, c)
    fake._race_wait_pick = lambda: T.Trainer._race_wait_pick(fake, poll=0.01)
    fake._race_park = lambda v: calls.append(("park", v.name))
    fake._race_follow = lambda w: calls.append(("follow", w.name))
    fake._race_activate = lambda v: (calls.append(("activate", v.name)), setattr(fake.race, "active", v))
    fake._race_drop = lambda v: (calls.append(("drop", v.name)), setattr(v, "alive", False))
    fake._race_finish = lambda: calls.append(("finish", fake.race.active.name))
    fake._race_report = lambda: None

    def probe(score, step):
        fake.controller = SimpleNamespace(combined_ema=score, last_step=step, best=score, best_step=step,
                                          since_improve=0)
        return T.Trainer._race_after_probe(fake)

    return probe, calls, variants


def test_trainer_waits_and_everyone_continues_from_the_pick(tmp_path):
    sched = R.RaceSchedule(["a", "b"], 0, total_steps=20, follow=True, pick=True)
    probe, calls, variants = fake_trainer(tmp_path, sched)
    probe(30.0, 10)
    (tmp_path / "RACE_PICK").write_text("nobody\na\n", encoding="utf-8")  # the latest line counts
    calls.clear()
    assert probe(40.0, 10) is variants[0]
    assert not (tmp_path / "RACE_PICK").exists()
    assert ("follow", "a") in calls and ("status", "waiting for your relay pick") in calls
    assert sched.history[-1]["picked"] == "you" and sched.history[-1]["leader"] == "a"
    probe(50.0, 20)
    (tmp_path / "RACE_PICK").write_text("b\n", encoding="utf-8")
    calls.clear()
    assert probe(45.0, 20) is variants[1]  # the last round: b (training now) is your pick and wins
    assert ("drop", "a") in calls and ("finish", "b") in calls and sched.state()["winner"] == "b"


def test_trainer_stops_waiting_on_a_stop_or_the_time_limit(tmp_path):
    sched = R.RaceSchedule(["a", "b"], 0, total_steps=20, follow=True, pick=True)
    probe, calls, _ = fake_trainer(tmp_path, sched)
    probe(30.0, 10)
    (tmp_path / "STOP").write_text("", encoding="utf-8")
    probe(40.0, 10)
    assert sched.history[-1]["picked"] == "stopped" and sched.leader == 1
    (tmp_path / "STOP").unlink()
    sched2 = R.RaceSchedule(["a", "b"], 0, total_steps=20, follow=True, pick=True)
    probe2, _, _ = fake_trainer(tmp_path, sched2, minutes=1e-5)  # under a millisecond
    probe2(50.0, 10)
    probe2(40.0, 10)
    assert sched2.history[-1]["picked"] == "waited" and sched2.leader == 0
