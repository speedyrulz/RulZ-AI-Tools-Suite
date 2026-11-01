"""The attention backend option: auto is PyTorch's own pick, a chosen backend goes first in PyTorch's priority order
with every other one still enabled (a call it cannot take runs on the next), and an unknown name is refused."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import attention as A  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    A.set_backend("torch")
    yield
    A.set_backend("torch")


def test_auto_is_cudnn_on_blackwell_only(monkeypatch):
    assert A.resolve("auto", "cpu") == "torch" and A.resolve("torch", "cpu") == "torch"
    assert A.resolve("cudnn", "cpu") == "cudnn" and A.resolve("AUTO ", "cpu") == "torch"
    monkeypatch.setattr(A, "cudnn_first", lambda device: True)
    assert A.resolve("auto", "cuda:1") == "cudnn" and A.resolve("flash", "cuda:1") == "flash"
    A.set_backend("auto", "cuda:1")
    assert A.backend_name() == "cudnn"


def test_backends_match_plain_sdpa_and_unknown_names_are_refused():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 9, 16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.equal(A.sdpa(q, k, v), ref) and A.backend_name() == "torch"
    A.set_backend("math")
    assert torch.allclose(A.sdpa(q, k, v), ref, atol=1e-6) and A.backend_name() == "math"
    mask = torch.ones(1, 1, 1, 9, dtype=torch.bool)
    mask[..., -2:] = False
    assert torch.allclose(A.sdpa(q, k, v, mask), F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-6)
    with pytest.raises(ValueError, match="attention_backend"):
        A.set_backend("sage")


def test_a_call_the_first_backend_cannot_take_runs_on_the_next():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 9, 16) for _ in range(3))
    A.set_backend("cudnn")  # cuDNN attention needs CUDA tensors: the CPU call runs on the next backend
    assert torch.equal(A.sdpa(q, k, v), F.scaled_dot_product_attention(q, k, v))
    assert A.backend_name() == "cudnn"


def test_choosing_a_backend_leaves_every_other_one_enabled():
    """The choice is PyTorch's priority order, never its global enable flags: with samples and probes rendering on
    several GPUs in threads, a per-call sdpa_kernel(cudnn) disabled the other backends for every thread, and Krea 2's
    fp32 VAE attention in another thread failed with "No available kernel"."""
    for name in ("cudnn", "flash", "efficient", "math", "torch"):
        A.set_backend(name)
        assert torch.backends.cuda.flash_sdp_enabled() and torch.backends.cuda.mem_efficient_sdp_enabled()
        assert torch.backends.cuda.math_sdp_enabled() and torch.backends.cuda.cudnn_sdp_enabled()
        order = torch._C._get_sdp_priority_order()
        assert sorted(order) == sorted(A._DEFAULT_ORDER)
        if name != "torch":
            assert order[0] == int(A.BACKENDS[name])
    assert torch._C._get_sdp_priority_order() == A._DEFAULT_ORDER  # "torch": PyTorch's own order again
