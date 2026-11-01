"""probe_async: the probe GPU's choice, the LoRA state sent to the probe process, the process's start and errors, and
the trainer acting on a probe once it is in (the process faked for that part: no model)."""
import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer import probeworker as PW  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=3, text_dim=32, freq_dim=16, latent_channels=128)


def _cfg(**kw):
    kw.setdefault("dataset_dir", "ds")
    return TrainConfig(**kw)


def _lora(seed, **kw):
    torch.manual_seed(seed)
    model = MOSSImageTransformer(**TINY)
    inj = L.inject_lora(model, 4, 4.0, ["q", "v"], **kw)
    for m in inj.values():
        for p in m.adapter_params().values():
            torch.nn.init.normal_(p)
    return inj


def test_config():
    assert not _cfg().probe_async and _cfg().probe_device == ""
    _cfg(probe_async=True, probe_device="cuda:1").validate()
    with pytest.raises(ValueError, match="probe_device"):
        _cfg(probe_device="cpu").validate()


def test_the_probe_gpu():
    assert PW.pick_device("", "cuda:0,cuda:1", "cuda:0", 2) == ("cuda:1", "")
    assert PW.pick_device("", "cuda:0,cuda:1", "cuda:1", 2) == ("cuda:0", "")  # training on the second card
    assert PW.pick_device("cuda:1", "cuda:0", "cuda:0", 2) == ("cuda:1", "")  # probe_device wins over devices
    assert PW.pick_device("", "cuda:0,cuda:1", "cuda:0", 1)[1] == "only one GPU"
    assert "no GPU besides cuda:0" in PW.pick_device("", "cuda:0", "cuda:0", 2)[1]
    assert "not a second GPU" in PW.pick_device("cuda:0", "", "cuda:0", 2)[1]
    assert "not a second GPU" in PW.pick_device("cuda:3", "", "cuda:0", 2)[1]


@pytest.mark.parametrize("kw", [{}, {"dora": True}, {"kind": "lokr"}])
def test_the_lora_state_crosses_to_another_model(kw):
    src, dst = _lora(0, **kw), _lora(1, **kw)
    state = L.lora_state(src)
    buf = io.BytesIO()
    torch.save(state, buf)  # as the probe process gets it: plain data, read back with weights_only
    got = torch.load(io.BytesIO(buf.getvalue()), weights_only=True)
    L.mirror_lora(L.lora_snapshots(got), dst)
    for name, m in src.items():
        for part, p in m.adapter_params().items():
            assert torch.equal(p, dst[name].adapter_params()[part])
        assert (dst[name].kind, dst[name].scale, dst[name].dora) == (m.kind, m.scale, getattr(m, "dora", False))
    for p in src[next(iter(src))].adapter_params().values():  # a copy: training goes on without changing it
        p.data.add_(1.0)
    assert not torch.equal(next(iter(state[next(iter(src))]["params"].values())),
                           next(iter(src[next(iter(src))].adapter_params().values())))


def test_a_failing_probe_process_is_reported(tmp_path):
    w = PW.ProbeProcess(_cfg().to_dict(), None, "cuda:99", tmp_path / "probe_worker.log")  # loads in the background
    with pytest.raises(PW.ProbeProcessError, match="cuda:99|CUDA|Torch not compiled"):
        w.wait_ready()
    w.close()
    assert (tmp_path / "probe_worker.log").exists()


class FakeProcess:
    """ProbeProcess's interface; the render is done when `done` is set."""

    def __init__(self):
        self.device, self.pending, self.done, self.count, self.rendered, self.waited = "cpu", None, False, 0, 0.0, 0.0
        self.closed = False

    def submit(self, step, jobs, state):
        assert self.pending is None
        self.pending = PW.Pending(step, jobs, state)

    def ready(self):
        return self.pending is not None and self.done

    def take(self, wait):
        if self.pending is None or not (wait or self.done):
            return None
        p, self.pending, self.done = self.pending, None, False
        self.count += 1
        return p, [f"job{p.step}.png"]

    def close(self):
        self.closed = True


def _trainer(**cfg):
    from rulztrainer import trainer as T

    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = _cfg(probe_every=10, probe_async=True, **cfg)
    tr.controller, tr.race_rows, tr.ft, tr.ema, tr.device = object(), None, None, None, "cpu"
    status = {}
    tr.status = SimpleNamespace(update=lambda **k: status.update(k), data={})
    return T, tr, status


def test_when_it_runs(monkeypatch):
    T, tr, _ = _trainer(device="cuda:0")
    tr.device = "cuda:0"
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(T.torch.cuda, "device_count", lambda: 2)
    assert tr._probe_async_device() == "cuda:1"
    for change in ({"race_rows": [("a", None, {})]}, {"ft": object()}, {"controller": None}):
        for k, v in change.items():
            setattr(tr, k, v)
        assert tr._probe_async_device() == ""
        tr.race_rows, tr.ft, tr.controller = None, None, object()
    tr.cfg.multi_gpu = "data_parallel"  # both GPUs train already
    assert tr._probe_async_device() == ""
    tr.cfg.multi_gpu, tr.cfg.probe_async = "off", False
    assert tr._probe_async_device() == ""


def test_a_probe_is_acted_on_once_it_is_in():
    T, tr, status = _trainer()
    tr.injected = _lora(0)
    first = {n: {k: p.clone() for k, p in m.adapter_params().items()} for n, m in tr.injected.items()}
    finished = []
    tr._probe_jobs = lambda step: [f"job{step}"]
    tr._score_probe = lambda jobs, images, step: (60.0, None, {"face": [60.0], "body": []})

    def finish(step, face, body, images, seeds, took, opt, sched, injected, now=None):
        # what a best save would write: the weights that were scored, not training's newer ones
        scored = all(torch.equal(p, first[n][k]) for n, m in injected.items() for k, p in m.adapter_params().items())
        finished.append((step, face, images, took, scored, now))
        return step >= 20  # the controller says stop on the second probe

    tr._probe_finish = finish
    tr.probe_worker = w = FakeProcess()
    tr._probe_async_start(10, tr.injected)
    for m in tr.injected.values():  # training goes on
        for p in m.adapter_params().values():
            p.data.add_(0.5)
    assert tr._probe_async_take(13, None, None) is False and finished == []  # still rendering
    w.done = True
    assert tr._probe_async_take(14, None, None) is False
    step, face, images, took, scored, now = finished[0]
    assert (step, face, images, scored, now) == (10, 60.0, ["job10.png"], True, 14)
    assert took.endswith("on cpu, in at step 14")
    assert all(not torch.equal(p, first[n][k]) for n, m in tr.injected.items()  # training's weights are back
               for k, p in m.adapter_params().items())
    # run_probe (the final step): renders on the training GPU while the probe GPU finishes its probe, which is
    # acted on first
    tr._probe_async_start(20, tr.injected)
    tr._free_for_rendering = lambda: None
    tr._sampling_slots = lambda model, step: contextlib.nullcontext([])
    tr._probe_with = lambda slots, step: (70.0, None, ["here.png"], {"face": [70.0], "body": []})
    assert tr.run_probe(None, 30, None, None, tr.injected) is True
    assert [(f[0], f[1], f[5]) for f in finished[1:]] == [(20, 60.0, 30), (30, 70.0, None)] and w.pending is None
    tr._close_probe_worker()
    assert w.closed and tr.probe_worker is None


def test_a_failed_probe_process_falls_back_to_the_training_gpu():
    T, tr, _ = _trainer()
    tr.injected = _lora(0)
    tr._probe_jobs = lambda step: [f"job{step}"]
    tr.probe_worker = w = FakeProcess()
    tr._probe_async_start(10, tr.injected)

    def fail(wait):
        raise PW.ProbeProcessError("CUDA out of memory")

    w.take = fail
    assert tr._probe_async_take(12, None, None, wait=True) is False
    assert tr.probe_worker is None and w.closed
