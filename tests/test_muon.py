"""Muon (rulztrainer.optimizers.muon): torch.optim.Muon's update on 2-D weights with match_rms_adamw, the same result
batched by shape and in small chunks, an update RMS about AdamW's, OFT's blocks as separate matrices, vectors at RMS
0.2 x lr, LoRA+ groups, weight decay, a resumed state, a trained LoRA, and the config / trainer / fine-tune / WebUI
wiring; NorMuon (optimizer normuon): rows evened out with the size kept, its state, resume, and the edge-layer
warning."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as lora_mod  # noqa: E402
from rulztrainer.config import AUTOMAGIC, OPTIMIZERS, TrainConfig  # noqa: E402
from rulztrainer.optimizers.muon import (Muon, adjusted_lr, normuon_scale, normuon_state_shape,  # noqa: E402
                                         zeropower_batched)
from rulztrainer.trainer import build_optimizer, build_scheduler  # noqa: E402

SHAPES = [(8, 32), (8, 32), (32, 8), (16, 16), (8, 32), (24, 40)]


def _params(shapes, seed: int = 0):
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s) * 0.1) for s in shapes]


def _clone(params):
    return [torch.nn.Parameter(p.detach().clone()) for p in params]


def _grads(params, step: int):
    g = torch.Generator().manual_seed(100 + step)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g)


def _run(opt, params, steps: int, first: int = 0):
    for s in range(first, first + steps):
        _grads(params, s)
        opt.step()


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon came with PyTorch 2.9")
def test_matches_torch_muon_on_matrices():
    ours, ref = _params(SHAPES), _params(SHAPES)
    _run(Muon(ours, lr=1e-2, weight_decay=0.01), ours, 5)
    _run(torch.optim.Muon(ref, lr=1e-2, weight_decay=0.01, adjust_lr_fn="match_rms_adamw"), ref, 5)
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)


def test_chunks_give_the_same_result():
    whole, chunked = _params(SHAPES), _params(SHAPES)
    _run(Muon(whole, lr=1e-2), whole, 3)
    _run(Muon(chunked, lr=1e-2, bucket_elems=1), chunked, 3)  # one matrix per batched run
    for a, b in zip(whole, chunked):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_orthogonalized_and_update_rms_near_adamws():
    x = torch.randn(3, 16, 64)
    s = torch.linalg.svdvals(zeropower_batched(x).float())
    assert s.min() > 0.4 and s.max() < 1.3  # Keller Jordan's quintic: singular values about 0.5 to 1.2, not exactly 1
    (p,) = _params([(64, 256)])
    before = p.detach().clone()
    _run(Muon([p], lr=1e-3, weight_decay=0.0), [p], 1)
    rms = (p.detach() - before).pow(2).mean().sqrt().item()
    assert 0.1e-3 < rms < 0.3e-3  # match_rms_adamw: about 0.2 x lr, AdamW's typical update RMS
    assert adjusted_lr(1e-3, "match_rms_adamw", 64, 256) == pytest.approx(1e-3 * 0.2 * 16)
    assert adjusted_lr(1e-3, "original", 256, 64) == pytest.approx(2e-3)


def test_oft_blocks_are_separate_matrices():
    torch.manual_seed(1)
    blocks = torch.nn.Parameter(torch.randn(4, 6, 6) * 0.1)
    singles = [torch.nn.Parameter(blocks.detach()[i].clone()) for i in range(4)]
    a, b = Muon([blocks], lr=1e-2), Muon(singles, lr=1e-2)
    for s in range(3):
        g = torch.Generator().manual_seed(s)
        blocks.grad = torch.randn(4, 6, 6, generator=g)
        for i, q in enumerate(singles):
            q.grad = blocks.grad[i].clone()
        a.step()
        b.step()
    torch.testing.assert_close(blocks.detach(), torch.stack([q.detach() for q in singles]), atol=1e-6, rtol=1e-5)


def test_vectors_move_at_rms_point_two_lr():
    v = torch.nn.Parameter(torch.randn(50))
    before = v.detach().clone()
    _run(Muon([v], lr=1e-3, weight_decay=0.0), [v], 1)
    assert (v.detach() - before).pow(2).mean().sqrt().item() == pytest.approx(0.2e-3, rel=1e-4)


def test_lora_plus_groups_and_weight_decay():
    down, up = _params([(8, 32), (32, 8)])
    d0, u0 = down.detach().clone(), up.detach().clone()
    opt = Muon([{"params": [down], "lr": 1e-3}, {"params": [up], "lr": 4e-3}], weight_decay=0.0)
    _run(opt, [down, up], 1)
    # same shape class (8 x 32 and 32 x 8 have the same max side): the up step is exactly 4x the down step's size
    ratio = (up.detach() - u0).norm() / (down.detach() - d0).norm()
    assert ratio.item() == pytest.approx(4.0, rel=0.05)
    (p,) = _params([(16, 16)])
    p.grad = torch.zeros_like(p)
    before = p.detach().clone()
    Muon([p], lr=1e-2, weight_decay=0.1).step()  # zero gradient: only the decoupled decay moves it
    torch.testing.assert_close(p.detach(), before * (1 - 1e-2 * 0.1))


def test_resumed_state_continues_the_same():
    straight, resumed = _params(SHAPES), _params(SHAPES)
    opt = Muon(straight, lr=1e-2)
    _run(opt, straight, 4)
    first = Muon(resumed, lr=1e-2)
    _run(first, resumed, 2)
    state = first.state_dict()
    second = Muon(resumed, lr=1e-2)
    second.load_state_dict(state)
    _run(second, resumed, 2, first=2)
    for a, b in zip(straight, resumed):
        torch.testing.assert_close(a, b)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.ModuleDict({"q": torch.nn.Linear(32, 32),
                                                                "o": torch.nn.Linear(32, 16)})])

    def forward(self, x):
        block = self.layers[0]
        return block["o"](torch.relu(block["q"](x)))


def test_trains_a_lora_through_build_optimizer():
    torch.manual_seed(0)
    lin = _Tiny()
    for p in lin.parameters():
        p.requires_grad_(False)
    injected = lora_mod.inject_lora(lin, 4, 4, ["q", "o"])
    assert len(injected) == 2
    cfg = TrainConfig(optimizer="muon", learning_rate=1e-2, lora_plus_ratio=4.0, steps=40)
    cfg.validate()
    groups = lora_mod.lora_param_groups(injected, cfg.learning_rate, cfg.lora_plus_ratio)
    opt = build_optimizer(cfg, groups)
    assert isinstance(opt, Muon) and len(opt.param_groups) == 2
    sched = build_scheduler(cfg, opt)
    x = torch.randn(64, 32)
    target = torch.randn(64, 16)
    losses = []
    for _ in range(40):
        loss = torch.nn.functional.mse_loss(lin(x), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_wiring():
    from rulztrainer import race as R
    from rulztrainer.finetune import BYTES_PER_PARAM

    assert "muon" in OPTIMIZERS and "muon" not in AUTOMAGIC  # the schedule, LoRA+ and per-image lr apply
    TrainConfig(optimizer="muon").validate()
    TrainConfig(race=True, probe_every=10, face_refs=["ref.png"], steps=60, race_configs=[
        {"name": "a"}, {"name": "b", "optimizer": "muon"}]).validate()
    assert "optimizer" in R.RACE_KEYS["LR & optimizer"]
    assert BYTES_PER_PARAM["adamw8bit"] < BYTES_PER_PARAM["muon"] < BYTES_PER_PARAM["adamw"]
    import webui

    (ex,) = [e for e in webui.EXPERIMENTS if e["key"] == "optimizer"]
    assert "muon" in ex["options"] and "muon" in ex["tested"]
    assert "normuon" in OPTIMIZERS and "normuon" not in AUTOMAGIC and "normuon" in ex["options"]
    assert BYTES_PER_PARAM["normuon"] == BYTES_PER_PARAM["muon"]


def test_normuon_evens_rows_and_keeps_each_matrix_size():
    torch.manual_seed(0)
    o = torch.randn(2, 64, 8) * torch.linspace(0.1, 3.0, 64).view(1, 64, 1)  # tall: rows along the long side
    v = torch.zeros(normuon_state_shape(2, 64, 8))
    assert v.shape == (2, 64, 1) and normuon_state_shape(1, 8, 64) == (1, 1, 64)
    out = normuon_scale(o, v, beta2=0.0)
    rms = out.square().mean(-1).sqrt()  # [matrix, row]
    assert (rms.amax(-1) / rms.amin(-1)).max().item() < 1.01  # 30x apart before, within each matrix
    torch.testing.assert_close(out.norm(dim=(-2, -1)), o.norm(dim=(-2, -1)))
    wide = torch.randn(1, 8, 64) * torch.linspace(0.1, 3.0, 64).view(1, 1, 64)
    out = normuon_scale(wide, torch.zeros(1, 1, 64), beta2=0.0)
    cols = out.square().mean(-2).sqrt()
    assert (cols.max() / cols.min()).item() < 1.01


def test_normuon_steps_state_and_resume():
    ref, nor = _params(SHAPES + [(4, 6, 6), (16,)]), _params(SHAPES + [(4, 6, 6), (16,)])
    _run(Muon(ref, lr=1e-2), ref, 3)
    opt = Muon(nor, lr=1e-2, normuon=True)
    _run(opt, nor, 3)
    assert any((a - b).abs().max() > 1e-6 for a, b in zip(ref, nor))  # it changes the update
    shapes = {tuple(p.shape): tuple(opt.state[p]["normuon_v"].shape) for p in nor if "normuon_v" in opt.state[p]}
    assert shapes[(8, 32)] == (1, 1, 32) and shapes[(32, 8)] == (1, 32, 1) and shapes[(4, 6, 6)] == (4, 6, 1)
    assert (16,) not in shapes  # vectors keep muon's rule
    for p in nor:  # step size: about muon's (the rescale keeps each matrix's size)
        assert torch.isfinite(p).all()
    straight, resumed = _params(SHAPES), _params(SHAPES)
    _run(Muon(straight, lr=1e-2, normuon=True), straight, 4)
    first = Muon(resumed, lr=1e-2, normuon=True)
    _run(first, resumed, 2)
    second = Muon(resumed, lr=1e-2, normuon=True)
    second.load_state_dict(first.state_dict())
    _run(second, resumed, 2, first=2)
    for a, b in zip(straight, resumed):
        torch.testing.assert_close(a, b)


def test_normuon_from_build_optimizer_and_the_edge_layer_warning(caplog):
    (p,) = _params([(16, 32)])
    opt = build_optimizer(TrainConfig(optimizer="normuon"), [p])
    assert isinstance(opt, Muon) and opt.param_groups[0]["normuon"]
    assert not build_optimizer(TrainConfig(optimizer="muon"), [p]).param_groups[0]["normuon"]
    with caplog.at_level("WARNING", logger="rulztrainer"):
        build_optimizer(TrainConfig(optimizer="muon", lora_targets=["q", "embed"]), [p])
    assert "embed" in caplog.text and "AdamW" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="rulztrainer"):
        build_optimizer(TrainConfig(optimizer="muon"), [p])
    assert "AdamW" not in caplog.text
