"""The parallel race (race_parallel): waves of one configuration per GPU, a lone one split over the GPUs or on the first,
configurations that own their parameters (masters), the engine's lockstep step on a tiny stand-in model, and the
validation. The step runs on the CPU with two stand-in 'GPUs'."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import dataset as D  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer import race as R  # noqa: E402
from rulztrainer import trainer as T  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.ema import LoraEMA  # noqa: E402
from rulztrainer.multigpu import DataParallelLoRA  # noqa: E402
from rulztrainer.race_parallel import RaceParallelMixin  # noqa: E402


def test_waves_take_one_configuration_per_gpu_and_the_odd_one_alone():
    s = R.RaceSchedule(["a", "b", "c"], warmup=0)
    assert s.wave(2) == [0, 1] and s.wave(1) == [0] and s.wave(8) == [0, 1, 2]
    s.record(0, 10.0, 3)
    s.record(1, 20.0, 3)
    assert s.wave(2) == [2]  # the odd one left of this round
    s.record(2, 30.0, 3)  # the round ends: a is out
    assert s.alive == [1, 2] and s.wave(2) == [1, 2]
    s.drop(1)
    assert s.done and s.wave(2) == []


def test_parallel_race_validation():
    ok = dict(race=True, race_parallel=True, multi_gpu="data_parallel", devices="cuda:0,cuda:1", probe_every=10,
              face_refs=["x.png"], steps=100, race_configs=[{"name": "a"}, {"name": "b"}])
    TrainConfig(**ok).validate()
    with pytest.raises(ValueError, match="two devices"):
        TrainConfig(**{**ok, "multi_gpu": "off"}).validate()
    with pytest.raises(ValueError, match="two devices"):
        TrainConfig(**{**ok, "devices": "cuda:0"}).validate()
    with pytest.raises(ValueError, match="identity loss"):
        TrainConfig(**{**ok, "race_configs": [{"name": "a"}, {"name": "b", "id_loss_weight": 0.5}]}).validate()
    with pytest.raises(ValueError, match="identity loss"):
        TrainConfig(**{**ok, "id_loss_weight": 0.5}).validate()


class Tiny(nn.Module):
    """One wrapped linear layer standing in for the DiT: x [B, 4, 1, 1] -> [B, 4, 1, 1]."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 4, bias=False)
        self.requires_grad_(False)

    def forward(self, x, t, embeds, mask):
        return self.lin(x.float().flatten(1)).reshape(x.shape).to(x.dtype)


def _lane_setup(n_replicas=2, ga=2):
    replicas = []
    for r in range(n_replicas):
        m = Tiny()
        inj = L.inject_lora(m, 4, 4.0, [], names=["lin"])
        replicas.append((m, inj, "cpu"))
    cfg = TrainConfig(gradient_accumulation=ga, batch_size=1, lora_ema=0.9, max_grad_norm=0.0, steps=100,
                      probe_every=3, learning_rate=1e-2, optimizer="adamw", lr_scheduler="constant", warmup_steps=0)
    variants = []
    for i, (name, rank) in enumerate((("a", 4), ("b", 2), ("c", 4))):
        vcfg = TrainConfig(**{**cfg.to_dict(), "lora_rank": rank, "lora_alpha": float(rank)})
        v = R.Variant(i, name, vcfg, {}, ["lin"])
        v.init_lora(replicas[0][1], 7)
        v.build_master("cpu")
        variants.append(v)
    sched = R.RaceSchedule([v.name for v in variants], 0)
    fake = SimpleNamespace(replicas=replicas, race=SimpleNamespace(variants=variants, wrappers=replicas[0][1],
                                                                    estimate={"total": 100}, active=variants[0]),
                           race_base=cfg, arch=None, status=SimpleNamespace(update=lambda **k: None),
                           _race_status=lambda: {}, _shrink_kept_blocks=lambda: False)
    fake.dp = DataParallelLoRA(replicas[0][1], replicas[1:])
    for name in ("_rp_init", "_rp_place", "_rp_split_reps", "_rp_plan", "_rp_counts", "_rp_step", "_rp_log"):
        setattr(fake, name, getattr(RaceParallelMixin, name).__get__(fake))
    fake._rp_init()
    for v in variants:
        params = [p for parts in v.weights.values() for p in parts.values()]
        v.state.update(optimizer=torch.optim.SGD(params, lr=0.05), scheduler=torch.optim.lr_scheduler.LambdaLR(
            torch.optim.SGD(params, lr=0.05), lambda s: 1.0), ema=LoraEMA(v.master, 0.9), step=0, losses=[],
            params=params, rng=None, gen=torch.Generator().manual_seed(1), ts_logit_shift=0.0, loss_watch=None)
        v.state["scheduler"] = torch.optim.lr_scheduler.LambdaLR(v.state["optimizer"], lambda s: 1.0)
        v.state["batch_iter"] = iter(lambda: [SimpleNamespace(key="k")], None)
        v.started = True
    return fake, variants, sched


@pytest.fixture
def stubbed(monkeypatch):
    """Fixed batches: batch n is the same tensors for every run, so a split and an unsplit step can be compared."""
    counter = {"n": 0}

    def load_batch(batch, cfg, rng):
        g = torch.Generator().manual_seed(100 + counter["n"])
        counter["n"] += 1
        return torch.randn(1, 4, 1, 1, generator=g), torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), None

    monkeypatch.setattr(D, "load_batch", load_batch)
    monkeypatch.setattr(T, "sample_sigmas", lambda cfg, n, gen, shift, tokens, arch: torch.full((n,), 0.5))
    return counter


def test_two_lanes_train_side_by_side_each_on_its_own_parameters(stubbed):
    fake, (a, b, c), _ = _lane_setup()
    lanes = fake._rp_plan([a, b])
    assert [(v.name, reps) for v, reps in lanes] == [("a", [0]), ("b", [1])]
    fake._rp_place(lanes)
    before = {v.name: [p.detach().clone() for parts in v.weights.values() for p in parts.values()] for v in (a, b)}
    fake._rp_step(lanes)
    for v in (a, b):
        assert v.state["step"] == 1 and len(v.state["losses"]) == 1
        after = [p.detach() for parts in v.weights.values() for p in parts.values()]
        assert any(not torch.equal(x, y) for x, y in zip(before[v.name], after))  # its optimizer stepped it
    assert fake.rp_done == 2
    # each model copy's wrappers carry their own configuration's weights (same storage on the same device)
    wa, wb = fake.replicas[0][1]["lin"].adapter_params(), fake.replicas[1][1]["lin"].adapter_params()
    assert wa["lora_down"].shape[0] == 4 and wb["lora_down"].shape[0] == 2  # ranks 4 and 2
    assert torch.equal(wa["lora_up"].data, a.weights["lin"]["lora_up"].data)
    assert torch.equal(wb["lora_up"].data, b.weights["lin"]["lora_up"].data)
    assert a.state["ema"].updates == 1 and b.state["ema"].updates == 1


def test_a_lone_configuration_split_over_the_gpus_matches_one_gpu(stubbed):
    results = {}
    for label, reps in (("split", [0, 1]), ("single", [0])):
        stubbed["n"] = 0
        fake, (a, b, c), _ = _lane_setup(ga=2)
        lanes = [(c, reps)]
        fake._rp_place(lanes)
        fake._rp_step(lanes)
        results[label] = [p.detach().clone() for parts in c.weights.values() for p in parts.values()]
    for x, y in zip(results["split"], results["single"]):
        assert torch.allclose(x, y, atol=1e-6)  # the same two batches, summed over the copies or on one


def test_the_split_needs_batches_enough_and_a_faster_split(stubbed):
    fake, (a, b, c), _ = _lane_setup(ga=1)
    assert fake._rp_split_reps() == [0]  # one batch per step cannot be split
    fake, (a, b, c), _ = _lane_setup(ga=2)
    assert fake._rp_split_reps() == [0, 1] and fake._rp_counts([0, 1]) == [1, 1]  # no measurements yet: even
    fake, _, _ = _lane_setup(ga=4)
    assert fake._rp_counts([0, 1]) == [2, 2] and fake._rp_counts([1]) == [4]
    for r, sec in ((0, 1.0), (1, 1.0)):  # equal speeds: the split is about twice as fast
        for _ in range(4):
            fake.rp_speed.add(r, sec, 4096)
    assert fake._rp_split_reps() == [0, 1]
    fake.rp_speed.reset()
    for _ in range(4):
        fake.rp_speed.add(0, 1.0, 4096)
        fake.rp_speed.add(1, 40.0, 4096)  # the second GPU is far slower: its share would only slow the step
    assert fake._rp_split_reps() == [0] and fake._rp_counts([0, 1])[0] > fake._rp_counts([0, 1])[1]


def test_master_parameters_move_with_their_optimizer_state(stubbed):
    fake, (a, b, c), _ = _lane_setup()
    fake._rp_place([(a, [0])])
    fake._rp_step([(a, [0])])
    opt = a.state["optimizer"]
    a.move_master("cpu")
    assert all(p.device.type == "cpu" for parts in a.weights.values() for p in parts.values())
    assert all(t.device.type == "cpu" for st in opt.state.values() for t in st.values() if torch.is_tensor(t))
    assert a.state["ema"].shadow[0].device.type == "cpu"
    snapshot = [p.detach().clone() for parts in a.weights.values() for p in parts.values()]
    fake._rp_place([(a, [0])])  # back on a copy: the same weights
    for x, p in zip(snapshot, [p for parts in a.weights.values() for p in parts.values()]):
        assert torch.equal(x, p.detach())
