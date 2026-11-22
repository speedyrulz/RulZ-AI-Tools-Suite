import random
from types import SimpleNamespace

import pytest
import torch

import rulztrainer.trainer as T
from rulztrainer import lora as L
from test_ema import _trainer, lora
from test_run_safety import rig, _go, _job


@pytest.mark.parametrize("where", ["optimizer", "ema", "scheduler"])
def test_crash_commit_window(rig, monkeypatch, where):
    cfg, _, _ = rig
    cfg.lora_ema = 0.9
    calls = [0]
    if where == "optimizer":
        original = torch.optim.AdamW.step

        def fail(self, *args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("injected crash")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(torch.optim.AdamW, "step", fail)
    elif where == "ema":
        from rulztrainer.ema import LoraEMA
        original = LoraEMA.update

        def fail(self, *args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("injected crash")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(LoraEMA, "update", fail)
        monkeypatch.setattr(T, "limit_step_mult", lambda cfg, mult: 0.5)
    else:
        original = T.build_scheduler

        def build(*args, **kwargs):
            sched = original(*args, **kwargs)
            cls = type(sched)
            step = cls.step

            def fail(self, *a, **k):  # on the class: an instance attribute would be pickled into the state
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("injected crash")
                return step(self, *a, **k)

            monkeypatch.setattr(cls, "step", fail)
            return sched

        monkeypatch.setattr(T, "build_scheduler", build)
    with pytest.raises(RuntimeError, match="injected crash"):
        _go(cfg)
    assert (_job(cfg) / "job_000001.safetensors").exists()
    if where == "optimizer":
        assert not list(_job(cfg).glob("state_*/train_state.pt"))
    else:
        state = torch.load(_job(cfg) / "state_000002" / "train_state.pt", weights_only=False)
        assert state["step"] == 2
        assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(cfg.learning_rate)


@pytest.mark.parametrize("mismatch", [None, "layer", "body", "old"])
def test_alignment_and_identity_state_roundtrip(tmp_path, mismatch):
    model, inj = lora()
    tr = _trainer(tmp_path, inj, model)
    tr.cfg.optimizer = "adamw"
    tr.cfg.id_align = True
    tr.id_aligner = torch.nn.Linear(2, 2)
    tr.id_align_opt = torch.optim.AdamW(tr.id_aligner.parameters())
    tr.align_layer = 3
    tr.id_aligner(torch.ones(1, 2)).sum().backward()
    tr.id_align_opt.step()
    tr.id_loss = SimpleNamespace(rng=random.Random(17), n_eligible=12)
    tr.id_loss.rng.random()
    opt = T.build_optimizer(tr.cfg, list(L.lora_parameters(inj)))
    sched = T.build_scheduler(tr.cfg, opt)
    gen, rng = torch.Generator(), random.Random()
    path = tr.save_state(4, inj, opt, sched, gen, rng)
    if mismatch == "old":
        state = torch.load(path / "train_state.pt", weights_only=False)
        for key in ("id_aligner", "id_align_opt", "align_layer", "id_loss"):
            state.pop(key)
        torch.save(state, path / "train_state.pt")
    model2, inj2 = lora(seed=1)
    other = _trainer(tmp_path, inj2, model2)
    other.cfg.optimizer = "adamw"
    other.cfg.id_align = True
    other.cfg.body_align = mismatch == "body"
    other.id_aligner = torch.nn.Linear(2, 2)
    other.id_align_opt = torch.optim.AdamW(other.id_aligner.parameters())
    other.align_layer = 4 if mismatch == "layer" else 3
    fresh = [p.detach().clone() for p in other.id_aligner.parameters()]
    opt2 = T.build_optimizer(other.cfg, list(L.lora_parameters(inj2)))
    other.load_state(path, inj2, opt2, T.build_scheduler(other.cfg, opt2), gen, rng)
    other._restore_alignment(other.cfg)
    if mismatch is None:
        assert all(torch.equal(a, b) for a, b in zip(tr.id_aligner.parameters(), other.id_aligner.parameters()))
        assert other.id_align_opt.state_dict()["state"]
        assert other.resumed_id_loss == {"rng": tr.id_loss.rng.getstate(), "n_eligible": 12}
    else:
        assert all(torch.equal(a, b) for a, b in zip(fresh, other.id_aligner.parameters()))
        assert not other.id_align_opt.state
        if mismatch == "old":
            assert other.resumed_id_loss is None
