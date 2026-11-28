"""checkpoint_save_attention (rulztrainer.checkpointing): attention outputs are kept, gradients are unchanged."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import checkpointing as C  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402

SDPA = ("scaled_dot_product",)


class Block(nn.Module):
    def __init__(self, d=16, heads=2):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Linear(d, d)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.out(a)
        return x + torch.nn.functional.gelu(self.ff(x))


class Toy(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(n)])
        self.mode = "none"  # none | plain | (save_attention flag on the model itself)

    def forward(self, x):
        for blk in self.layers:
            x = C.checkpointed(self, blk, x) if self.mode != "none" else blk(x)
        return x


def make(mode, save_attention=False, seed=0):
    torch.manual_seed(seed)
    m = Toy().double()
    m.mode = mode
    m.save_attention = save_attention
    return m


def grads(m, x):
    m.zero_grad()
    m(x.clone().requires_grad_(True)).square().sum().backward()
    return [p.grad.clone() for p in m.parameters()]


class Counter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "backward" not in str(func) and any(s in str(func) for s in SDPA):  # forward kernels only
            self.n += 1
        return func(*args, **(kwargs or {}))


def backward_sdpa_count(m, x):
    y = m(x.requires_grad_(True)).square().sum()
    c = Counter()
    with c:
        y.backward()
    return c.n


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(2, 8, 16, dtype=torch.double)


def test_gradients_equal(x):
    ref = grads(make("none"), x)
    plain = grads(make("plain"), x)
    sac = grads(make("plain", True), x)
    for a, b, c in zip(ref, plain, sac):
        assert torch.equal(a, b) and torch.equal(a, c)


def test_attention_recompute_skipped(x):
    plain = backward_sdpa_count(make("plain"), x.clone())
    sac = backward_sdpa_count(make("plain", True), x.clone())
    assert plain == 3  # one recompute per block
    assert sac == 0


def test_compiled_path_used_and_equal(x):
    ref = grads(make("none"), x)
    m = make("plain", True)
    for blk in m.layers:
        blk._rz_compiled = True
        blk.compile(backend="aot_eager", dynamic=True)  # as Trainer.compile: the block's own graph stays too
        C.compile_save_attention(blk, backend="aot_eager")
        assert callable(blk._rz_sac_call)
    with torch.no_grad():  # samples / probes call the compiled block directly
        assert torch.allclose(m.layers[0](x), make("none").layers[0](x), atol=1e-10, rtol=1e-9)
    assert backward_sdpa_count(m, x.clone()) == 0  # the compiled recompute skips attention too
    called = []
    for blk in m.layers:
        orig = blk._rz_sac_call
        blk._rz_sac_call = (lambda o: lambda *a: (called.append(1), o(*a))[1])(orig)
    got = grads(m, x)
    assert len(called) == 3
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-10, rtol=1e-9)


def test_compiled_block_without_sac_call_uses_plain_checkpoint(x, monkeypatch):
    m = make("plain", True)
    for blk in m.layers:
        blk._rz_compiled = True
    monkeypatch.setattr(C, "sac_checkpoint", lambda *a: pytest.fail("eager SAC around a compiled block"))
    ref = grads(make("none"), x)
    for a, b in zip(ref, grads(m, x)):
        assert torch.equal(a, b)
    assert backward_sdpa_count(m, x.clone()) >= 3  # plain checkpoint: attention is recomputed


def test_config_needs_gradient_checkpointing():
    TrainConfig(checkpoint_save_attention=True, gradient_checkpointing=True).validate()
    with pytest.raises(ValueError, match="checkpoint_save_attention"):
        TrainConfig(checkpoint_save_attention=True, gradient_checkpointing=False).validate()
