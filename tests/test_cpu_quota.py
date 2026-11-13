"""The container's CPU quota (machine.cpu_quota) and the PyTorch thread cap the trainer sets from it: a vast.ai
instance showing 24 CPUs with a quota of 11.5 throttled a 24-thread PyTorch to a crawl on CPU work."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import machine  # noqa: E402


def _files(tmp_path, v2=None, quota=None, period=None):
    paths = {"v2": tmp_path / "cpu.max", "v1_quota": tmp_path / "cpu.cfs_quota_us",
             "v1_period": tmp_path / "cpu.cfs_period_us"}
    for key, text in (("v2", v2), ("v1_quota", quota), ("v1_period", period)):
        if text is not None:
            paths[key].write_text(text + "\n", encoding="utf-8")
    return paths


def test_cgroup_v2(tmp_path):
    assert machine.cpu_quota(**_files(tmp_path, v2="1152000 100000")) == pytest.approx(11.52)
    assert machine.cpu_quota(**_files(tmp_path, v2="2304000 100000")) == pytest.approx(23.04)


def test_cgroup_v2_unlimited(tmp_path):
    assert machine.cpu_quota(**_files(tmp_path, v2="max 100000")) is None
    # v2 present wins over stale v1 files
    assert machine.cpu_quota(**_files(tmp_path, v2="max 100000", quota="50000", period="100000")) is None


def test_cgroup_v1(tmp_path):
    assert machine.cpu_quota(**_files(tmp_path, quota="400000", period="100000")) == pytest.approx(4.0)
    assert machine.cpu_quota(**_files(tmp_path, quota="-1", period="100000")) is None


def test_missing_and_garbage(tmp_path):
    assert machine.cpu_quota(**_files(tmp_path)) is None  # no cgroup files (Windows, a bare host)
    for i, text in enumerate(("", "garbage", "1 2 3", "abc 100000", "100000 0", "-5 100000")):
        d = tmp_path / f"v2_{i}"
        d.mkdir()
        assert machine.cpu_quota(**_files(d, v2=text)) is None, text
    for i, (quota, period) in enumerate((("x", "100000"), ("400000", None), ("400000", "0"))):
        d = tmp_path / f"v1_{i}"
        d.mkdir()
        assert machine.cpu_quota(**_files(d, quota=quota, period=period)) is None, (quota, period)


def test_thread_cap():
    assert machine.cpu_thread_cap(24, 11.52, env={}) == 11
    assert machine.cpu_thread_cap(48, 23.04, env={}) == 23
    assert machine.cpu_thread_cap(8, 11.52, env={}) is None  # already within the quota
    assert machine.cpu_thread_cap(11, 11.52, env={}) is None
    assert machine.cpu_thread_cap(24, None, env={}) is None  # no quota
    assert machine.cpu_thread_cap(4, 0.5, env={}) == 1  # never 0
    assert machine.cpu_thread_cap(24, 11.52, env={"OMP_NUM_THREADS": "16"}) is None  # the user's own setting
    assert machine.cpu_thread_cap(24, 11.52, env={"MKL_NUM_THREADS": "8"}) is None
    assert machine.cpu_thread_cap(24, 11.52, env={"OMP_NUM_THREADS": " "}) == 11  # blank counts as unset


def test_trainer_caps_torch_threads(monkeypatch, caplog):
    trainer = pytest.importorskip("rulztrainer.trainer")
    torch = trainer.torch
    seen = {"n": 24}
    monkeypatch.setattr(torch, "get_num_threads", lambda: seen["n"])
    monkeypatch.setattr(torch, "set_num_threads", lambda n: seen.update(n=n))
    caplog.set_level("INFO", logger=trainer.log.name)
    assert trainer.cap_cpu_threads(quota=11.52, env={}) == 11 and seen["n"] == 11
    assert any("CPU quota 11.5 CPUs: PyTorch CPU threads 24 -> 11" in r.getMessage() for r in caplog.records)
    assert trainer.cap_cpu_threads(quota=11.52, env={}) is None  # already capped
    seen["n"] = 24
    assert trainer.cap_cpu_threads(quota=11.52, env={"OMP_NUM_THREADS": "24"}) is None and seen["n"] == 24

    def broken(_n):
        raise RuntimeError("cannot set threads now")

    monkeypatch.setattr(torch, "set_num_threads", broken)
    assert trainer.cap_cpu_threads(quota=11.52, env={}) is None  # never stops a run from starting
