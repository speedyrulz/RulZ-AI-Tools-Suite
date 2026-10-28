"""Block swap: a checkpointed tiny DiT gives the same outputs and LoRA gradients with its first blocks' frozen weights
streamed from RAM (CPU here; on a GPU too when one is free, with the side-stream prefetch), the frozen weights stay
in RAM between passes, and the logsnr timestep mode."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402
from rulztrainer.blockswap import BlockSwap  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.trainer import sample_sigmas  # noqa: E402


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.ffn = nn.Linear(d, d)

    def forward(self, x):
        return x + self.ffn(torch.relu(self.q(self.norm(x))))


class Tiny(nn.Module):
    def __init__(self, d=64, n=5):
        super().__init__()
        torch.manual_seed(0)
        self.inp = nn.Linear(d, d)
        self.layers = nn.ModuleList(Block(d) for _ in range(n))
        self.requires_grad_(False)

    def forward(self, x):
        x = self.inp(x)
        for b in self.layers:
            x = checkpoint(b, x, use_reentrant=False) if torch.is_grad_enabled() else b(x)
        return x


def _model(device, quantize=False):
    m = Tiny()
    if quantize:
        Q.quantize_module_linears(m.layers, "int8", min_dim=64)
        for mod in m.modules():
            if isinstance(mod, Q.QuantLinear):
                mod.int8_matmul = False
    inj = L.inject_lora(m, 4, 4.0, [], names=[f"layers.{i}.q" for i in range(5)] + ["layers.4.ffn"])
    torch.manual_seed(1)
    for w in inj.values():
        nn.init.normal_(w.lora_up.weight, std=0.02)
    return m, inj


def _run(device, swap, quantize=False):
    m, inj = _model(device, quantize)
    if swap:
        bs = BlockSwap(m, range(3), device)
    else:
        m.to(device)
        bs = None
    x = torch.randn(4, 64, generator=torch.Generator().manual_seed(2)).to(device)
    out = m(x)
    out.pow(2).mean().backward()
    if bs is not None:  # the backward's recomputations stop early: none may leave its GPU copy on the module
        assert len(bs.installed) <= 1, bs.installed
    grads = {n: w.lora_up.weight.grad.detach().cpu().clone() for n, w in inj.items()}
    with torch.no_grad():
        again = m(x).cpu()
    return out.detach().cpu(), grads, again, m, bs, inj


@pytest.mark.parametrize("quantize", [False, True])
def test_same_outputs_and_grads_on_cpu(quantize):
    a = _run("cpu", False, quantize)
    b = _run("cpu", True, quantize)
    assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[2], b[2], atol=1e-6)
    for n in a[1]:
        assert torch.allclose(a[1][n], b[1][n], atol=1e-6), n
    # the swapped blocks hold their RAM copies at rest; the rest of the model and the adapters on the device
    bs, inj = b[4], b[5]
    assert bs.bytes > 0 and all(i in bs.entries for i in range(3))
    assert inj["layers.0.q"].lora_up.weight.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_swapping_lowers_the_step_peak_on_gpu():
    """The training step's peak memory with blocks swapped stays below the unswapped one (a recomputation that stops
    early once left every swapped block's GPU copy on its module, so the peak was higher than without swapping)."""
    dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    if torch.cuda.mem_get_info(torch.device(dev))[0] < 2 * 2**30:
        pytest.skip("GPU busy")
    peaks = {}
    for swap in (False, True):
        torch.cuda.empty_cache()
        m, inj = _model(dev, True)
        bs = BlockSwap(m, range(4), dev) if swap else None
        if not swap:
            m.to(dev)
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        x = torch.randn(64, 64, device=dev)
        for _ in range(2):
            m(x).pow(2).mean().backward()
        torch.cuda.synchronize(dev)
        peaks[swap] = torch.cuda.max_memory_allocated(dev)
        del m, inj, bs
    assert peaks[True] < peaks[False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_same_outputs_and_grads_on_gpu_with_prefetch():
    dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    free, total = torch.cuda.mem_get_info(torch.device(dev))
    if free < 2 * 2**30:
        pytest.skip("GPU busy")
    a = _run(dev, False, True)
    b = _run(dev, True, True)
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[2], b[2], atol=1e-5)
    for n in a[1]:
        assert torch.allclose(a[1][n], b[1][n], atol=1e-5), n
    m, bs, inj = b[3], b[4], b[5]
    block = m.layers[0]
    assert block.norm.weight.device.type == "cpu" and block.norm.weight.is_pinned()
    assert block.q.base.qdata.device.type == "cpu"  # a quantized Linear's int8 data, in RAM at rest
    assert m.layers[4].q.base.qdata.device.type == "cuda" and inj["layers.0.q"].lora_up.weight.device.type == "cuda"
    # whole-model moves (a sampling model swapped in and out) come back to the same split
    m.to("cpu")
    bs.reoffload()
    assert not bs.installed
    assert m.layers[3].ffn.weight.device.type == "cuda" and m.layers[1].ffn.weight.device.type == "cpu"


def test_block_swap_validation():
    TrainConfig(block_swap=4).validate()
    for bad, msg in ((dict(block_swap=4, multi_gpu="model_split"), "one GPU"),
                     (dict(block_swap=4, gradient_checkpointing=False), "gradient_checkpointing"),
                     (dict(block_swap=4, finetune=True, learning_rate=1e-5), "fine-tune"),
                     (dict(block_swap=-1), "number of blocks")):
        with pytest.raises(ValueError, match=msg):
            TrainConfig(**bad).validate()


def test_logsnr_timesteps():
    cfg = TrainConfig(timestep_sampling="logsnr", logsnr_mean=-6.0, logsnr_std=2.0)
    s = sample_sigmas(cfg, 20000, torch.Generator().manual_seed(0))
    # log-SNR = 2 log((1 - s) / s) ~ N(-6, 2): the median noise level is sigmoid(3) ~ 0.95
    assert float(s.median()) == pytest.approx(torch.sigmoid(torch.tensor(3.0)).item(), abs=0.01)
    logsnr = 2 * torch.log((1 - s.clamp(max=0.999)) / s)
    assert float(logsnr[s < 0.999].std()) == pytest.approx(2.0, abs=0.25)
    assert float((s < 0.5).float().mean()) < 0.08  # mostly noisy steps
    with pytest.raises(ValueError, match="logsnr_std"):
        TrainConfig(timestep_sampling="logsnr", logsnr_std=0).validate()
