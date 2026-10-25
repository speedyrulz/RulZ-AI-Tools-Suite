"""The attention backend option: auto is plain SDPA, a pinned backend runs through sdpa_kernel, a rejected call falls
back to auto with one warning, and an unknown name is refused."""
import logging
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
    monkeypatch.setattr(A, "is_blackwell", lambda device: True)
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


def test_rejected_backend_falls_back_to_auto(caplog):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 9, 16) for _ in range(3))
    A.set_backend("cudnn")  # cuDNN attention needs CUDA tensors: the CPU call is rejected
    with caplog.at_level(logging.WARNING, logger="rulztrainer"):
        out = A.sdpa(q, k, v)
    assert torch.equal(out, F.scaled_dot_product_attention(q, k, v))
    assert A.backend_name() == "torch" and "rejected" in caplog.text
