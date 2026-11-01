"""Ideogram 4 support without model weights: the ported transformer against diffusers' Ideogram4Transformer2DModel
(tiny random config; padded batches per sample, the image-only "unconditional" sequence), the release's fp8 and nf4
formats and the fused attention layout on synthetic tensors with the real key names, LoRA naming and the export
ComfyUI / diffusers read, the sampler schedule against Ideogram's own, the JSON caption wrapper against the caption
verifier's rules, and architecture detection, caches and buckets."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer.ideogram4 import single_file as ISF  # noqa: E402
from rulztrainer.ideogram4.model import Ideogram4Transformer  # noqa: E402

diffusers = pytest.importorskip("diffusers")

TINY = dict(in_channels=16, num_layers=2, attention_head_dim=64, num_attention_heads=2, intermediate_size=96,
            adaln_dim=32, llm_features_dim=40, rope_theta=5_000_000, mrope_section=(12, 10, 10), norm_eps=1e-5)


def ref_model(seed=0):
    """Random weights under which the attention, the text and the positions all move the output (norms near 1,
    sizeable gates): with this init a q/k swap in one block changes the output by about its own size, a mirrored
    image by a sixth of it."""
    torch.manual_seed(seed)
    ref = diffusers.Ideogram4Transformer2DModel(**TINY).float().eval()
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if n.endswith(("norm1.weight", "norm2.weight", "norm_q.weight", "norm_k.weight", "llm_cond_norm.weight")):
                p.normal_(1.0, 0.2)
            else:
                p.normal_(0, 0.5 if "adaln_modulation" in n else 0.05)
    return ref


def pair(seed=0):
    """Our port and diffusers' model with the same random weights, fp32 (diffusers' to_q / to_k / to_v fused)."""
    ref = ref_model(seed)
    ours = Ideogram4Transformer(**TINY).float().eval()
    state = ISF.normalize_dit_state({k: v.clone() for k, v in ref.state_dict().items()})
    missing, unexpected = ours.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    return ours, ref


def ref_call(ref, x, sigma, feats):
    """diffusers' forward on the pipeline's packed layout ([left padding][text][image], segment ids, indicators) for
    a batch: x [B, C, h, w], sigma [B], feats a list of [n_i, D] (n_i = 0: the image-only unconditional sequence).
    Returns the image tokens' output in the trainer's layout, in Ideogram's convention (clean - noise)."""
    b, c, h, w = x.shape
    lens = [f.shape[0] for f in feats]
    n = max(lens)
    pos, seg, ind = diffusers.Ideogram4Pipeline._prepare_ids(lens, h, w, n, x.device)
    hidden = torch.cat([torch.zeros(b, n, c), x.flatten(2).transpose(1, 2)], dim=1)
    enc = torch.zeros(b, n + h * w, feats[0].shape[-1])
    for i, f in enumerate(feats):
        enc[i, n - f.shape[0]:n] = f
    t = 1.0 - (sigma.float() * 1000.0) / 1000.0  # what the port computes from sigma * 1000
    out = ref(hidden_states=hidden, timestep=t, encoder_hidden_states=enc, position_ids=pos, segment_ids=seg,
              indicator=ind, return_dict=False)[0]
    return out[:, n:].transpose(1, 2).reshape(b, c, h, w)


def test_port_matches_diffusers_per_sample_with_padding():
    ours, ref = pair()
    x = torch.randn(2, 16, 3, 5)
    sigma = torch.tensor([0.7, 0.25])
    lens = (9, 4)
    enc = torch.randn(2, max(lens), 40)
    mask = torch.zeros(2, max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        mask[i, :n] = True
        enc[i, n:] = 0.0  # pad_embeds zero-fills
    with torch.no_grad():
        got = ours(x, sigma * 1000.0, enc, mask)
        # the trainer's convention is noise - clean: the port negates Ideogram's clean - noise
        for i, n in enumerate(lens):  # each sample alone, as the reference pipeline computes it
            want = -ref_call(ref, x[i:i + 1], sigma[i:i + 1], [enc[i, :n]])
            assert torch.allclose(got[i:i + 1], want, atol=1e-5, rtol=1e-4), (got[i:i + 1] - want).abs().max()
        # and the reference's own left-padded batch gives the same
        want = -ref_call(ref, x, sigma, [enc[i, :n] for i, n in enumerate(lens)])
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)
        # an all-true mask is no mask
        assert torch.equal(ours(x[:1], 700.0, enc[:1], torch.ones(1, 9, dtype=torch.bool)),
                           ours(x[:1], 700.0, enc[:1], None))
        # the comparison is sensitive: other text, or q and k swapped in one block, moves the output far more than
        # the tolerance
        assert (ours(x, sigma * 1000.0, torch.randn_like(enc) * mask[..., None], mask) - got).abs().mean() > 0.05
        w = ours.layers[1].attention.qkv.weight
        d = w.shape[0] // 3
        w.data = torch.cat([w[d:2 * d], w[:d], w[2 * d:]])
        assert (ours(x, sigma * 1000.0, enc, mask) - got).abs().mean() > 0.05


def test_image_only_sequence_is_the_unconditional_branch():
    ours, ref = pair(1)
    x = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        got = ours(x, torch.tensor([450.0]), torch.zeros(1, 0, 40), torch.zeros(1, 0, dtype=torch.bool))
        want = -ref_call(ref, x, torch.tensor([0.45]), [torch.zeros(0, 40)])
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)


def test_bf16_gradients_checkpointing_and_layers():
    ours, _ = pair(2)
    m = ours.to(torch.bfloat16)
    m.gradient_checkpointing = True
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.randn(1, 16, 3, 5).to(torch.bfloat16)
    enc = torch.randn(1, 4, 40).to(torch.bfloat16)
    out = m(x, torch.tensor([500.0]), enc, None)
    assert out.shape == x.shape and out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in m.layers.parameters())
    assert m.llm_cond_proj.weight.grad is not None and m.input_proj.weight.grad is not None
    with torch.no_grad():
        skipped = m(x, 500.0, enc, None, skip_layers=[1])
    assert not torch.equal(skipped, out.detach())
    assert len(m.layers) == 2 and m.latent_channels == 16 and m.text_dim == 40
    assert m.split_tail_modules() == [m.final_layer]


def test_model_split_hand_off_matches():
    from rulztrainer.multigpu import split_model

    ours, _ = pair(3)
    x, enc = torch.randn(1, 16, 2, 2), torch.randn(1, 4, 40)
    with torch.no_grad():
        want = ours(x, 300.0, enc, None)
        split_model(ours, ["cpu", "cpu"])  # same device twice: exercises the hand-off path
        assert torch.equal(ours(x, 300.0, enc, None), want)
