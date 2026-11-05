"""scripts/hull_agreement.py, M0(d) of docs/hull-optimizer.md: the readings' scale, the regions' parts, and that the
measurement leaves plain Hull alone."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import hull_agreement as A  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer.optimizers.hull import Hull  # noqa: E402
from rulztrainer.trainer import Trainer  # noqa: E402


def test_noise_reads_the_floor_a_constant_reads_1_and_empty_steps_do_not_count(tmp_path):
    torch.manual_seed(0)
    rec = A.Recorder(tmp_path / "x.npz", 0)
    key = (0, 1)
    for i in range(3000):
        rec.feed(key, 0, 0, torch.randn(2, 64, 32))
        rec.feed(key, 0, 1, torch.ones(2, 64, 32))
        rec.feed(key, 0, 2, torch.ones(2, 64, 32) if i % 2 else torch.zeros(2, 64, 32))
    assert all(abs(float(v[0]) - 0.23) < 0.01 for v in rec.read(key, 0, 0))
    assert all(float(v[0]) > 0.99 for v in rec.read(key, 0, 1))
    assert all(float(v[0]) > 0.99 for v in rec.read(key, 0, 2))


def test_a_region_reads_nothing_before_it_has_enough_steps(tmp_path):
    rec = A.Recorder(tmp_path / "x.npz", 0)
    rec.feed((0,), 0, 0, torch.ones(1, 4, 4))
    assert np.isnan(rec.read((0,), 0, 0)[0][0])


def test_the_parts_add_up_to_the_plain_gradient_which_stays_autograds(monkeypatch):
    monkeypatch.setattr(L.HullBackward, "backward", staticmethod(A.patched_backward))
    torch.manual_seed(0)
    x = torch.randn(2, 10, 8, requires_grad=True)
    a, b = nn.Parameter(torch.randn(4, 8)), nn.Parameter(torch.randn(8, 4))
    regions = torch.zeros(2, 10, 3)
    regions[:, :5, 0], regions[:, 5:8, 1], regions[:, 8:, 2] = 1, 1, 1
    seen = []
    L.HullBackward.apply(x, a, b, (lambda *got: seen.append(got), a, b, regions, None)).pow(2).sum().backward()
    _, _, parts_a, parts_b, _, _ = seen[0]
    a2, b2 = a.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    (x.detach() @ a2.T @ b2.T).pow(2).sum().backward()
    tolerance = 1e-5 * float(a2.grad.abs().max())
    for got, want in ((sum(parts_a), a2.grad), (sum(parts_b), b2.grad), (a.grad, a2.grad), (b.grad, b2.grad)):
        assert (got - want).abs().max() < tolerance


def _train(rec=None):
    torch.manual_seed(1)
    a, b = nn.Parameter(torch.randn(4, 16) * 0.1), nn.Parameter(torch.randn(16, 4) * 0.1)
    opt = Hull([a, b], [(a, b)], [3.0], [1.0], lr=2e-4)
    for _ in range(40):
        opt.zero_grad()
        a.grad, b.grad = torch.randn_like(a), torch.randn_like(b)
        if rec is not None:
            rec.add(a, b, [a.grad / 3] * 3, [b.grad / 3] * 3, None, None)
        opt.step()
    return a.detach().clone(), b.detach().clone()


def test_measuring_leaves_hulls_result_alone_and_undo_restores_everything(tmp_path):
    plain = _train()
    methods = (Hull._update, Hull.zero_grad, Hull.step, Trainer._hull_arm, L.HullBackward.__dict__["backward"])
    rec = A.Recorder(tmp_path / "x.npz", 1)
    undo = A.install(rec)
    try:
        measured = _train(rec)
    finally:
        undo()
    assert methods == (Hull._update, Hull.zero_grad, Hull.step, Trainer._hull_arm, L.HullBackward.__dict__["backward"])
    assert torch.equal(plain[0], measured[0]) and torch.equal(plain[1], measured[1])
    rec.save()
    with np.load(rec.out) as d:
        assert d["readings"].shape == (4, 1, 2, len(A.REGIONS), 2)
        assert np.isfinite(d["readings"][-1]).all()
        assert np.allclose(d["share"][-1], 1 / 9, atol=1e-3)  # each part is a third of the gradient


def test_the_summary_reads_a_saved_run(tmp_path, capsys):
    readings = np.full((30, 3, 2, len(A.REGIONS), 2), 0.23, dtype=np.float32)
    readings[:, :, :, A.BODY] = 0.25
    readings[:, 2] = np.nan  # a text-only layer
    path = tmp_path / "x.npz"
    np.savez(path, steps=np.arange(1, 31) * 10, readings=readings, share=np.full((30, 3, 2, 3), 0.2))
    A.summary(str(path))
    out = capsys.readouterr().out
    assert "20 snapshots" in out and "4 of 6 tensors" in out
    assert "body +0.0200" in out and "above background 100%" in out
