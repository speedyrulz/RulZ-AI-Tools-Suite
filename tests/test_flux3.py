"""FLUX 3 Klein support without model weights: the ported transformer against diffusers' Flux3Transformer2DModel (tiny
random config in fp32, padded batches per sample, the distilled flow map's target timestep), BFL's / ComfyUI's
single-file layout (and the fused qkv of ComfyUI's int8 files) against the real key names, the diffusers layout
round trip, LoRA naming and the export both ComfyUI and diffusers read, the sampler schedule against
FlowMatchEulerDiscreteScheduler, and architecture detection, caches and the 16 px buckets."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import dataset as D  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.flux3 import loading as FL  # noqa: E402
from rulztrainer.flux3 import single_file as FSF  # noqa: E402
from rulztrainer.flux3.model import Flux3KleinTransformer  # noqa: E402
from rulztrainer.flux3.sampler import SHIFT, flux3_sigmas  # noqa: E402

diffusers = pytest.importorskip("diffusers")
pytest.importorskip("diffusers.models.transformers.transformer_flux3")

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY = dict(in_channels=16, num_layers=2, num_single_layers=3, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=24, timestep_channels=32, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4))


def pair(seed=0, flow_map=False):
    """Our port and diffusers' model with the same random weights, fp32. Ours is loaded from diffusers' state dict
    renamed to BFL's layout (the reference-image mode dropped)."""
    torch.manual_seed(seed)
    ref = diffusers.Flux3Transformer2DModel(cond_in_channels=16, flow_map=flow_map, **TINY).float().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.1)
    ours = Flux3KleinTransformer(flow_map=flow_map, **TINY).float().eval()
    state = FSF.to_model_layout(dict(ref.state_dict()))
    missing, unexpected = ours.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    return ours, ref


def ref_call(ref, x, sigma, enc, target=None):
    """diffusers' forward for one sample, its own text rows only, on the trainer's [1, C, h, w] latents."""
    from diffusers.modular_pipelines.flux3.before_denoise import Flux3TextRoPEInputsStep, _prepare_image_ids

    _, c, h, w = x.shape
    kw = {} if target is None else {"target_timestep": target}
    out = ref(hidden_states=x.flatten(2).transpose(1, 2), encoder_hidden_states=enc, timestep=sigma,
              img_ids=_prepare_image_ids(h, w, 0.5), txt_ids=Flux3TextRoPEInputsStep._prepare_text_ids(enc),
              return_dict=False, **kw)[0]
    return out.transpose(1, 2).reshape(1, c, h, w)


def test_port_matches_diffusers_per_sample_with_padding():
    ours, ref = pair()
    x = torch.randn(2, 16, 4, 6)
    sigma = torch.tensor([0.7, 0.25])
    lens = (9, 5)
    enc = torch.randn(2, max(lens), 24)
    mask = torch.zeros(2, max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        mask[i, :n] = True
    with torch.no_grad():
        got = ours(x, sigma * 1000.0, enc, mask)
        for i, n in enumerate(lens):  # each sample as the reference computes it alone, without the batch padding
            want = ref_call(ref, x[i:i + 1], sigma[i:i + 1], enc[i:i + 1, :n])
            assert torch.allclose(got[i:i + 1], want, atol=1e-5, rtol=1e-4), (got[i:i + 1] - want).abs().max()
        # an all-true mask is no mask
        assert torch.equal(ours(x[:1], 700.0, enc[:1], mask[:1] | True), ours(x[:1], 700.0, enc[:1], None))


def test_flow_map_target_timestep_matches_diffusers():
    ours, ref = pair(1, flow_map=True)
    assert ours.flow_map and ours.is_distilled and ours.ref_time_in is not None
    x, enc = torch.randn(1, 16, 2, 4), torch.randn(1, 7, 24)
    with torch.no_grad():
        got = ours(x, 600.0, enc, None, target_timestep=250.0)
        want = ref_call(ref, x, torch.tensor([0.6]), enc, target=torch.tensor([0.25]))
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)
        assert not torch.allclose(got, ours(x, 600.0, enc, None, target_timestep=0.0))
    with pytest.raises(ValueError, match="flow-map"):
        ours(x, 600.0, enc, None)  # the distilled model needs its target
    base, _ = pair(1)
    with pytest.raises(ValueError, match="flow-map"):
        base(x, 600.0, enc, None, target_timestep=250.0)


def test_bf16_gradients_checkpointing_skip_and_layout():
    ours, _ = pair(2)
    m = ours.to(torch.bfloat16)
    m.gradient_checkpointing = True
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.randn(1, 16, 3, 5).to(torch.bfloat16)  # an odd grid runs too
    enc = torch.randn(1, 4, 24).to(torch.bfloat16)
    out = m(x, torch.tensor([500.0]), enc, None)
    assert out.shape == x.shape and out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in m.single_blocks.parameters())
    assert all(p.grad is not None for b in m.mode_blocks for p in b.parameters())
    with torch.no_grad():
        skipped = m(x, 500.0, enc, None, skip_layers=[1])
    assert not torch.equal(skipped, out.detach())
    assert m.layers is m.single_blocks and len(m.layers) == 3 and m.latent_channels == 16
    assert len(m.mode_blocks) == 4 and m.split_tail_modules() == [m.final_layer]
    assert m.single_blocks[0].norm.query_norm.scale.dtype == torch.bfloat16  # cast; the norm runs in fp32 anyway


def test_model_split_hand_off_matches():
    from rulztrainer.multigpu import split_model

    ours, _ = pair(3)
    x, enc = torch.randn(1, 16, 2, 2), torch.randn(1, 4, 24)
    with torch.no_grad():
        want = ours(x, 300.0, enc, None)
        split_model(ours, ["cpu", "cpu"])  # same device twice: exercises the hand-off path
        assert torch.equal(ours(x, 300.0, enc, None), want)
