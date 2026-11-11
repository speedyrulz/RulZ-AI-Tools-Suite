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


CUDNN_ERROR = ("Expected mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good() to be true, but got "
               "false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)")


def test_a_cudnn_attention_failure_is_told_from_other_errors():
    assert A.is_cudnn_failure(RuntimeError(CUDNN_ERROR))
    assert A.is_cudnn_failure(RuntimeError("aten::_scaled_dot_product_cudnn_attention: cuDNN Frontend error"))
    assert not A.is_cudnn_failure(RuntimeError("mat1 and mat2 shapes cannot be multiplied"))
    assert not A.is_cudnn_failure(torch.OutOfMemoryError("CUDA out of memory in mha_graph"))  # has its own retry
    assert not A.is_cudnn_failure(ValueError(CUDNN_ERROR))


def test_fall_back_switches_cudnn_attention_off_once_and_set_backend_brings_it_back():
    A.set_backend("cudnn")
    assert torch.compiler.config.cache_key_tag.endswith("|rulz-sdpa=cudnn")
    assert A.fall_back()
    assert not torch.backends.cuda.cudnn_sdp_enabled() and A.backend_name() == "torch"
    assert torch.backends.cuda.flash_sdp_enabled() and torch.backends.cuda.mem_efficient_sdp_enabled()
    assert torch.backends.cuda.math_sdp_enabled()
    assert torch._C._get_sdp_priority_order() == A._DEFAULT_ORDER
    assert torch.compiler.config.cache_key_tag.endswith("|rulz-sdpa=torch-nocudnn")
    assert not A.fall_back()  # already off: a second failure is not one a retry gets past
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 9, 16) for _ in range(3))
    assert torch.equal(A.sdpa(q, k, v), F.scaled_dot_product_attention(q, k, v))
    A.set_backend("cudnn")  # the next run in this process starts clean
    assert torch.backends.cuda.cudnn_sdp_enabled() and A.backend_name() == "cudnn"
    assert torch.compiler.config.cache_key_tag.count("rulz-sdpa=") == 1  # replaced, not stacked


def test_fall_back_drops_compiled_graphs_so_they_trace_again():
    """A compiled graph has its attention kernel written into it: after a fall-back it must compile again."""
    from torch._dynamo.utils import counters

    A.set_backend("cudnn")
    fn = torch.compile(lambda q, k, v: A.sdpa(q, k, v) + 1, backend="eager")
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 9, 16) for _ in range(3))
    ref = fn(q, k, v)
    before = int(counters["frames"]["total"])
    fn(q, k, v)
    assert int(counters["frames"]["total"]) == before  # cached
    assert A.fall_back(compiled=True)
    assert torch.equal(fn(q, k, v), ref)
    assert int(counters["frames"]["total"]) > before
