import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402
from rulztrainer.multigpu import (DataParallelLoRA, balanced_assignment, parse_devices,  # noqa: E402
                                  parse_split_plan, split_model)

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=4, text_dim=32, freq_dim=16, latent_channels=128)
TARGETS = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]


def _inputs(seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 128, 4, 4, generator=g).to(torch.bfloat16), torch.tensor([300.0]),
            torch.randn(1, 5, 32, generator=g).to(torch.bfloat16))


def test_parse_devices():
    assert parse_devices("cuda:0, cuda:1", "cpu") == ["cuda:0", "cuda:1"]
    assert parse_devices("", "cuda:0") == ["cuda:0"]


def test_data_parallel_averages_grads_and_syncs_weights():
    torch.manual_seed(0)
    primary = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    replica = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    replica.load_state_dict(primary.state_dict())
    pi = L.inject_lora(primary, 4, 4, TARGETS)
    ri = L.inject_lora(replica, 4, 4, TARGETS)
    dp = DataParallelLoRA(pi, [(replica, ri, "cpu")])
    dp.sync_weights()
    name = next(iter(pi))
    assert torch.equal(pi[name].lora_down.weight, ri[name].lora_down.weight)
    # make the LoRA active so lora_up gets a gradient too
    for m in list(pi.values()) + list(ri.values()):
        torch.nn.init.normal_(m.lora_up.weight, std=0.01)
    dp.sync_weights()
    # sequential reference: mean of the two per-sample losses on one model
    primary.train()
    xa, ta, ca = _inputs(1)
    xb, tb, cb = _inputs(2)
    ref = (primary(xa, ta, ca).float().pow(2).mean() + primary(xb, tb, cb).float().pow(2).mean()) / 2
    ref.backward()
    ref_grads = {n: m.lora_down.weight.grad.clone() for n, m in pi.items()}
    for m in pi.values():
        m.lora_down.weight.grad = None
        m.lora_up.weight.grad = None
    # data parallel: each copy takes one sample with loss / n, then grads are reduced
    def work(job):
        model, x, t, c = job
        loss = model(x, t, c).float().pow(2).mean() / 2
        loss.backward()
        return loss.item()

    dp.run(work, [(primary, xa, ta, ca), (replica, xb, tb, cb)])
    dp.reduce_grads()
    for n, m in pi.items():
        assert torch.allclose(m.lora_down.weight.grad, ref_grads[n], atol=1e-5, rtol=1e-3)
        assert ri[n].lora_down.weight.grad is None
    # weights updated on the primary propagate to the replica
    with torch.no_grad():
        pi[name].lora_down.weight.add_(1.0)
    dp.sync_weights()
    assert torch.equal(pi[name].lora_down.weight, ri[name].lora_down.weight)


def test_model_split_forward_matches_unsplit():
    torch.manual_seed(0)
    model = MOSSImageTransformer(**TINY).to(torch.bfloat16).eval()
    x, t, c = _inputs(3)
    with torch.no_grad():
        ref = model(x, t, c)
        summary = split_model(model, ["cpu", "cpu"])  # same device twice: exercises the hand-off code path
        out = model(x, t, c)
        assert model.split_assignment == ["cpu"] * 4 and len(summary) == 1
        split_model(model, ["cpu", "cpu"], plan="cpu=0-1;cpu=2-3")
        out2 = model(x, t, c)
    assert torch.equal(ref, out) and torch.equal(ref, out2)


def test_split_plan_and_balance():
    assert parse_split_plan("cuda:0=0-25;cuda:1=26-41", 42) == ["cuda:0"] * 26 + ["cuda:1"] * 16
    for bad in ("cuda:0=0-20", "cuda:0=0-41;cuda:1=41", "nonsense"):
        try:
            parse_split_plan(bad, 42)
            raise AssertionError(bad)
        except ValueError:
            pass
    # all blocks trained: even halves
    a = balanced_assignment(42, ["g0", "g1"])
    assert a.count("g0") == 21 and a.count("g1") == 21 and a == sorted(a)
    # only blocks 21-41 trained: blocks 0-20 are forward-only (cost 1), so g0 takes many more of them
    a = balanced_assignment(42, ["g0", "g1"], trained=list(range(21, 42)))
    n0 = a.count("g0")
    assert n0 > 21 and a[:n0] == ["g0"] * n0 and a[n0:] == ["g1"] * (42 - n0)
    cost0 = sum(1.0 if i < 21 else 4.0 for i in range(n0))
    cost1 = sum(1.0 if i < 21 else 4.0 for i in range(n0, 42))
    assert abs(cost0 - cost1) <= 4.0
