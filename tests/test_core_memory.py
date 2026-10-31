"""Training-core memory and speed paths: sampling-only LoRA extras staged on the layer's device once per render round,
DoRA's column norm from the delta's factors (against the dense W + delta it replaces), the sampling VAE parked in RAM
between rounds, unload_model dropping weights instead of copying them to RAM, latent caching's CPU prefetch, the loss
watch's per-step log writes and the WebUI's status / log readers."""
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402

CUDA = torch.cuda.is_available()


class _Two(nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.a = nn.Linear(16, 24).to(dtype)
        self.b = nn.Linear(24, 16).to(dtype)

    def forward(self, x):
        return self.b(self.a(x))


def _extra_file(tmp_path) -> Path:
    g = torch.Generator().manual_seed(1)
    path = tmp_path / "turbo.safetensors"
    save_file({"diffusion_model.a.lora_down.weight": torch.randn(4, 16, generator=g),
               "diffusion_model.a.lora_up.weight": torch.randn(24, 4, generator=g),
               "diffusion_model.a.alpha": torch.tensor(2.0),
               "diffusion_model.a.diff_b": torch.randn(24, generator=g),
               "diffusion_model.b.lokr_w1": torch.randn(4, 4, generator=g),
               "diffusion_model.b.lokr_w2": torch.randn(4, 6, generator=g)}, str(path))
    return path


def _wrappers(m):
    return [x for x in m.modules() if isinstance(x, L.LoRALinear)]


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_extras_staged_once_and_dropped(tmp_path, device):
    torch.manual_seed(0)
    m = _Two()
    assert L.attach_extra_lora(m, str(_extra_file(tmp_path)), 0.5) == 2
    m.to(device)
    x = torch.randn(3, 16, dtype=torch.bfloat16, device=device)
    for w in _wrappers(m):  # the reference: the CPU originals converted on every forward (before staging)
        w.extras_enabled = True
    with torch.no_grad():
        want = m(x)
    L.set_extras_enabled(m, True)
    for w in _wrappers(m):
        assert len(w.staged_extras) == len(w.extras) == 1 and len(w.staged_biases) == len(w.extra_biases)
        for _, first, second, _ in w.staged_extras:
            assert first.device.type == device and first.dtype == torch.bfloat16 and second.dtype == torch.bfloat16
        assert all(e[1].device.type == "cpu" and e[1].dtype == torch.float32 for e in w.extras)  # originals kept
        assert w._live_extras(x)[0] is w.staged_extras
    with torch.no_grad():
        got = m(x)
    assert torch.equal(got, want)
    L.set_extras_enabled(m, False)
    assert all(w.staged_extras is None and w.staged_biases is None for w in _wrappers(m))
    with torch.no_grad():
        plain = m(x)
    assert not torch.equal(plain, want)


def test_extras_fall_back_when_stale(tmp_path):
    torch.manual_seed(0)
    m = _Two()
    L.attach_extra_lora(m, str(_extra_file(tmp_path)), 1.0)
    L.set_extras_enabled(m, True)
    w = m.a
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    assert w._live_extras(x)[0] is w.staged_extras
    assert w._live_extras(x.float())[0] is w.extras  # another input dtype: the exact originals
    w.extras.append(("lora", torch.randn(1, 16), torch.randn(24, 1), 1.0))  # attached after the round started
    assert w._live_extras(x)[0] is w.extras
    del w.extras[-1]
    assert w._live_extras(x)[0] is w.staged_extras


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_extras_follow_the_frozen_weights_device(tmp_path):
    """A block-swapped layer's frozen weights wait in RAM (adapters on the GPU): its extras are staged there, cast,
    and stream in per forward; a model moved after staging uses the originals."""
    torch.manual_seed(0)
    m = _Two()
    L.attach_extra_lora(m, str(_extra_file(tmp_path)), 1.0)
    m.to("cuda")
    m.a.base.weight.data = m.a.base.weight.data.cpu()  # what BlockSwap does to a swapped block
    L.set_extras_enabled(m, True)
    assert m.a.staged_extras[0][1].device.type == "cpu" and m.a.staged_extras[0][1].dtype == torch.bfloat16
    assert m.b.staged_extras[0][1].device.type == "cuda"
    x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    assert m.a._live_extras(x)[0] is m.a.staged_extras  # in RAM: uploaded per call, no cast
    m.a.base.weight.data = m.a.base.weight.data.cuda()
    m.to("cpu")  # moved after staging: the GPU copies are not used for CPU inputs
    assert m.b._live_extras(x.cpu())[0] is m.b.extras
    L.set_extras_enabled(m, False)


def _dense_norm(w: L.LoRALinear) -> torch.Tensor:
    """The previous implementation: the dense fp32 W + delta and its column norm."""
    return (L.dense_weight(w.base).to(w.dora_scale.device) + w.delta_weight()).norm(dim=0)


@pytest.mark.parametrize("kind", ["lora", "lokr"])
@pytest.mark.parametrize("quant", [None, "int8_convrot"])
@pytest.mark.parametrize("big", [False, True])
def test_dora_norm_matches_dense(kind, quant, big):
    torch.manual_seed(0)
    lin = nn.Linear(128, 96).to(torch.bfloat16)
    base = lin if quant is None else Q.QuantLinear(Q.quantize_weight(lin.weight.data.float(), quant, group=64),
                                                   lin.bias.data, torch.bfloat16, int8_matmul=False)
    w = L.LoRALinear(base, rank=8, alpha=4.0, kind=kind, factor=8, dora=True)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # a trained delta: small against W, or as large as W (big)
        for p in w.adapter_params().values():
            if p is not w.dora_scale:
                p.copy_(torch.randn(p.shape, generator=g) * (0.5 if big else 0.05))
    want, got = _dense_norm(w), w.merged_col_norm()
    assert got.shape == want.shape == (128,)
    assert torch.allclose(got, want, rtol=2e-5, atol=1e-6), (got - want).abs().max()
    x = torch.randn(5, 128, dtype=torch.bfloat16)
    s_new = w.dora_input_scale()
    assert torch.allclose(s_new, w.dora_scale / (want + L.DORA_EPS), rtol=2e-5)
    out = w(x)
    out.float().sum().backward()  # the magnitude still trains, the norm stays out of the gradient
    assert w.dora_scale.grad is not None


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_dora_norm_matches_dense_cuda():
    torch.manual_seed(0)
    w = L.LoRALinear(nn.Linear(256, 384).to(torch.bfloat16), rank=16, alpha=16.0, dora=True).cuda()
    with torch.no_grad():
        w.lora_up.weight.normal_(0, 0.05)
    assert torch.allclose(w.merged_col_norm(), _dense_norm(w), rtol=2e-5, atol=1e-6)


# ------------------------------------------------------------ the sampling VAE between rounds


class _Vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.register_buffer("scale", torch.full((4,), 0.5))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.conv(x) * self.scale.view(1, -1, 1, 1)


def _trainer_with_vae(vae, id_loss=None):
    from rulztrainer.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t.vae, t.vaes, t.id_loss = vae, {}, id_loss
    t.vae_parked = t.vae_park_rows = None
    return t


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_vae_parked_between_rounds():
    torch.manual_seed(0)
    vae = _Vae().cuda().eval().requires_grad_(False)
    x = torch.randn(1, 4, 8, 8, device="cuda")
    want = vae(x)
    t = _trainer_with_vae(vae)
    t._park_vae()
    assert t.vae_parked == "cuda:0"
    assert all(p.device.type == "cpu" and p.is_pinned() for p in vae.parameters())
    assert vae.scale.device.type == "cpu"
    rows = t.vae_park_rows
    assert t._vae_on("cuda:0") is vae and t.vae_parked is None  # the same object, back on its GPU
    assert vae.device.type == "cuda" and vae.scale.device.type == "cuda" and not t.vaes
    assert torch.equal(vae(x), want)
    t._park_vae()
    assert t.vae_park_rows is rows  # the pinned copies are made once
    assert torch.equal(t._vae_on("cuda:0")(x), want)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_vae_stays_with_identity_loss():
    vae = _Vae().cuda()
    t = _trainer_with_vae(vae, id_loss=object())
    t._park_vae()
    assert t.vae_parked is None and vae.device.type == "cuda"


def test_vae_on_cpu_not_parked():
    vae = _Vae()
    t = _trainer_with_vae(vae)
    t._park_vae()
    assert t.vae_parked is None and t._vae_on("cpu") is vae


# ------------------------------------------------------------ unload_model


def test_unload_model_drops_to_meta():
    from rulztrainer.trainer import unload_model

    m = nn.Linear(8, 8)
    unload_model(m)
    assert m.weight.device.type == "meta"


def test_unload_model_falls_back_to_cpu():
    from rulztrainer.trainer import unload_model

    moved = []

    class NoMeta(nn.Module):
        def to(self, device, *a, **k):
            if str(device) == "meta":
                raise NotImplementedError("no meta")
            moved.append(str(device))
            return self

    unload_model(NoMeta())
    assert moved == ["cpu"]


# ------------------------------------------------------------ latent caching prefetch


def test_latent_prefetch_window_and_results():
    import threading

    from rulztrainer.dataset import LatentPrefetch

    calls, lock = [], threading.Lock()

    def fn(x):
        with lock:
            calls.append(x)
        return x * 10

    items = list(range(10))
    pre = LatentPrefetch(items, fn, ahead=3)
    try:
        assert pre.get(0) == 0
        assert pre.next == 4  # item 0 plus the next three queued
        assert [pre.get(i) for i in range(1, 10)] == [i * 10 for i in range(1, 10)]
        assert pre.get(3) == 30  # asked twice: prepared again
        assert pre.next == 10 and not pre.futures
    finally:
        pre.close()
    assert sorted(calls) == sorted(items + [3])


def test_cache_latents_uses_every_image(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image

    from rulztrainer import dataset as D
    from rulztrainer.config import TrainConfig
    from safetensors.torch import load_file

    items = []
    for i in range(5):
        p = tmp_path / f"{i}.png"
        Image.fromarray(np.full((32, 32, 3), i * 40, dtype=np.uint8)).save(p)
        items.append(D.Item(image=p, caption="c", width=32, height=32, latent_path=tmp_path / f"{i}.lat",
                            latent_flip_path=None, text_path=tmp_path / "t"))

    class Arch:
        @staticmethod
        def encode_images(vae, t):
            return t.mean(dim=(2, 3), keepdim=True)

    monkeypatch.setattr(D, "get_arch", lambda cfg: Arch)
    vae = nn.Identity()
    vae.device = torch.device("cpu")
    D.cache_latents(items, TrainConfig(), vae)
    for i, it in enumerate(items):
        lat = load_file(str(it.latent_path))["latent"]
        assert torch.allclose(lat, torch.full_like(lat, i * 40 / 127.5 - 1.0), atol=1e-5)


# ------------------------------------------------------------ loss watch log


def test_loss_watch_writes_once_per_step(tmp_path, monkeypatch):
    from rulztrainer.config import TrainConfig
    from rulztrainer.dataset import Item
    from rulztrainer.losswatch import LossWatch

    items = [Item(image=tmp_path / f"{i}.png", caption=str(i), width=64, height=64, latent_path=tmp_path / "l",
                  latent_flip_path=None, text_path=tmp_path / "t") for i in range(4)]
    watch = LossWatch(TrainConfig(loss_watch=True), items, tmp_path)
    opens = []
    real_open = open

    def counting_open(path, *a, **k):
        if Path(path) == watch.log_path:
            opens.append(path)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", counting_open)
    for step in (1, 2):
        for it in items:
            watch.observe(step, it.key, 0.5, 0.3)
    assert len(opens) == 1  # step 1's four lines, when step 2 began
    watch.flush()
    lines = [json.loads(s) for s in watch.log_path.read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in lines] == [1] * 4 + [2] * 4 and len(opens) == 2
    watch.flush()  # nothing pending: no write
    assert len(opens) == 2


# ------------------------------------------------------------ WebUI readers


def test_read_status_survives_bad_files(tmp_path, monkeypatch):
    import webui

    monkeypatch.setattr(webui, "_job_dir", tmp_path)
    p = tmp_path / "status.json"
    p.write_text(json.dumps({"step": 3}), encoding="utf-8")
    assert webui.read_status() == {"step": 3}
    p.write_text('{"step": ', encoding="utf-8")  # torn
    assert webui.read_status() is None
    p.write_bytes(b'{"stage": "\xff\xfe"}')  # not UTF-8
    assert webui.read_status() is None
    p.unlink()
    p.mkdir()  # unreadable as a file (an OSError, like a file locked by the writer on Windows)
    assert webui.read_status() is None


def test_tail_lines_reads_from_the_end(tmp_path):
    import webui

    p = tmp_path / "train.log"
    rows = [f"line {i} " + "x" * (i % 37) for i in range(5000)]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert webui.tail_lines(p, 80) == rows[-80:]
    assert webui.tail_lines(p, 80, chunk=100) == rows[-80:]  # many small reads, a line cut at every boundary
    assert webui.tail_lines(p, 10000, chunk=4096) == rows
    p.write_text("a\nb", encoding="utf-8")  # no final newline
    assert webui.tail_lines(p, 1, chunk=1) == ["b"] and webui.tail_lines(p, 5) == ["a", "b"]
    assert webui.tail_lines(tmp_path / "missing.log", 5) == []
