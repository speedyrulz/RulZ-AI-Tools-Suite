"""MiniMax H3 support without model weights: the ported transformer against diffusers' MiniMaxH3Transformer3DModel
(tiny random config, fp32), fed the packed one-frame layout diffusers' own pipeline builds
(MiniMaxH3PrepareLayoutStep.build_packed_sequence, MiniMaxH3SetTimestepsStep.build_row_timesteps), per sample with
padded text; the pruned AdaLN table path; the layouts (diffusers <-> original, Comfy-Org's single files, MiniMaxAI's
interleaved qkv) against diffusers' own converters; LoRA naming and the export layout the ecosystem loads; the training
adapter and the turbo LoRA in their published key layouts; the sampler schedule against MiniMaxH3Scheduler; the
VAE layouts; architecture detection, caches and the 32 px buckets."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.h3 import convert as C  # noqa: E402
from rulztrainer.h3.model import (AUDIO_TAG, TEXT_TAG, VIDEO_TAG, MiniMaxH3Transformer, audio_sigma,  # noqa: E402
                                  patchify, unpatchify)

diffusers = pytest.importorskip("diffusers")

# a tiny H3: the attention is wider than the residual stream (as the real 56 x 128 > 5376), 2 x 3 x 4 rotary
# channels of 32 per head, the real 24 video / 32 audio latent channels and the 256-wide sinusoidal time input
TINY_DIFFUSERS = dict(num_attention_heads=2, attention_head_dim=32, hidden_size=48, num_layers=2, num_refiner_layers=1,
                      ffn_dim=64, in_channels=24, audio_in_channels=32, patch_size=(1, 2, 2), text_dim=40,
                      freq_dim=256, time_embed_hidden_dim=56, time_embed_dim=24, rope_freq_dim=4, rope_theta=10000.0)
TINY = C.diffusers_config_to_port(TINY_DIFFUSERS)


def pair(seed=0):
    """Our port and diffusers' model with the same random weights, fp32."""
    torch.manual_seed(seed)
    ref = diffusers.MiniMaxH3Transformer3DModel(**TINY_DIFFUSERS).float().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.1)
    ours = MiniMaxH3Transformer(**TINY).float().eval()
    missing, unexpected = ours.load_state_dict(C.diffusers_to_port(dict(ref.state_dict())), strict=True)
    assert not missing and not unexpected
    return ours, ref


def ref_call(ref, x, sigma, enc, audio_rows):
    """diffusers' forward for one sample on the pipeline's own one-frame t2va layout: [text | 2 x 2 audio rows |
    one latent frame], the row timesteps of its plan (video t = 1 - sigma for text and video, the shifted audio time
    for the audio rows). Returns the video velocity [1, 24, h, w] and the audio rows' [1, 4, 32], data-ward."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (MiniMaxH3PrepareLayoutStep,
                                                                       MiniMaxH3SetTimestepsStep,
                                                                       patchify_video_latents)

    _, c, h, w = x.shape
    n_text = enc.shape[1]
    pos, tags, vid_idx, aud_idx, txt_idx, n_cv, n_ca = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.full((n_text,), TEXT_TAG), num_latent_frames=1, latent_height=h, latent_width=w, num_audio_latents=2,
        patch_size=(1, 2, 2), audio_channels=2, audio_tag=AUDIO_TAG, video_tag=VIDEO_TAG)
    t_v = 1.0 - float(sigma)
    t_a = 1.0 - float(audio_sigma(torch.tensor(float(sigma))))
    ts, idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(vid_idx, aud_idx, n_cv, n_ca, n_text, t_v, t_a,
                                                            max(t_v, 0.999), 1.0)
    rows = patchify_video_latents(x.unsqueeze(2), (1, 2, 2)).unsqueeze(0)
    out = ref(hidden_states=rows, audio_hidden_states=audio_rows, encoder_hidden_states=enc, timestep=ts,
              timestep_indices=idx, token_tags=tags, position_ids=pos.float(), video_indices=vid_idx,
              audio_indices=aud_idx, text_indices=txt_idx, return_dict=False)
    return unpatchify(out[0], c, h, w), out[1]


def test_layout_helpers_match_the_pipeline():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (MiniMaxH3PrepareLayoutStep,
                                                                       patchify_video_latents)

    ours = MiniMaxH3Transformer(**TINY)
    for h, w, n_text in ((4, 6, 7), (6, 4, 3), (8, 8, 1)):
        pos = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.ones(n_text, dtype=torch.long), 1, h, w, 2, (1, 2, 2), 2, AUDIO_TAG, VIDEO_TAG)[0]
        got = ours.positions(torch.tensor([n_text]), n_text, h, w)[0]
        assert torch.equal(got, pos)  # the same float64 grid, bit for bit
        x = torch.randn(2, 24, h, w)
        rows = patchify_video_latents(x.unsqueeze(2), (1, 2, 2)).reshape(2, -1, 96)
        assert torch.equal(patchify(x), rows) and torch.equal(unpatchify(rows, 24, h, w), x)


def test_port_matches_diffusers_per_sample_with_padding():
    ours, ref = pair()
    x = torch.randn(2, 24, 4, 6)
    sigma = torch.tensor([0.93, 0.31])
    lens = (9, 5)
    enc = torch.randn(2, max(lens), 40)
    mask = torch.zeros(2, max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        mask[i, :n] = True
    audio = torch.randn(2, 4, 32)
    with torch.no_grad():
        got, got_audio = ours(x, sigma * 1000.0, enc, mask, audio_states=audio, return_audio=True)
        for i, n in enumerate(lens):  # each sample as the reference computes it alone, without the padding
            want, want_audio = ref_call(ref, x[i:i + 1], sigma[i], enc[i:i + 1, :n], audio[i:i + 1])
            assert torch.allclose(got[i:i + 1], -want, atol=2e-5, rtol=1e-4)  # noise-ward = -(data-ward)
            assert torch.allclose(got_audio[i:i + 1], -want_audio, atol=2e-5, rtol=1e-4)
        # the mask matters (padding seen as text changes the shorter sample), and an all-true mask is no mask
        unmasked = ours(x, sigma * 1000.0, enc, None, audio_states=audio)
        assert (unmasked[1] - got[1]).abs().max() > 1e-4
        assert torch.equal(ours(x[:1], 700.0, enc[:1], mask[:1], audio_states=audio[:1]),
                           ours(x[:1], 700.0, enc[:1], None, audio_states=audio[:1]))


def test_audio_placeholder_is_the_noised_silence():
    ours, ref = pair(1)
    silence = torch.randn(4, 32)
    ours.audio_silence = silence
    x, enc = torch.randn(1, 24, 4, 4), torch.randn(1, 5, 40)
    sigma = torch.tensor([0.8])
    s_a = audio_sigma(sigma)
    assert abs(float(s_a) - 3 * (0.8 / (12 - 11 * 0.8)) / (1 + 2 * (0.8 / (12 - 11 * 0.8)))) < 1e-6
    seed = ours.audio_noise_seed
    with torch.no_grad():
        got = ours(x, sigma * 1000.0, enc, None)
        noise = torch.randn(1, 4, 32, generator=torch.Generator().manual_seed(seed))
        rows = (1 - s_a) * silence + s_a * noise
        want, _ = ref_call(ref, x, sigma[0], enc, rows)
        assert torch.allclose(got, -want, atol=2e-5, rtol=1e-4)
        assert ours.audio_noise_seed == seed + 1  # a fresh draw per forward
        assert not torch.equal(ours(x, sigma * 1000.0, enc, None), got)
    # the shifts agree with the released schedulers at every grid point (one base time, two shifts)
    from diffusers import MiniMaxH3Scheduler

    v, a = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    v.set_timesteps(31)
    a.set_timesteps(31)
    assert torch.allclose(audio_sigma(v.sigmas), a.sigmas, atol=1e-5)


def test_bf16_gradients_checkpointing_and_layers():
    ours, _ = pair(2)
    m = ours.to(torch.bfloat16)
    m.gradient_checkpointing = True
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.randn(1, 24, 4, 6).to(torch.bfloat16)
    enc = torch.randn(1, 4, 40).to(torch.bfloat16)
    out = m(x, torch.tensor([500.0]), enc, None)
    assert out.shape == x.shape and out.dtype == torch.float32 and torch.isfinite(out).all()
    out.pow(2).mean().backward()
    assert all(p.grad is not None for p in m.blocks.parameters())
    with torch.no_grad():
        skipped = m(x, 500.0, enc, None, skip_layers=[1])
    assert skipped.shape == out.shape
    assert m.layers is m.blocks and len(m.layers) == 2 and m.latent_channels == 24
    assert m.split_tail_modules() == [m.final_layer]
    with pytest.raises(ValueError):
        m(torch.randn(1, 24, 3, 4), 500.0, enc, None)  # an odd latent grid is not a 32 px image


def test_model_split_hand_off_matches():
    from rulztrainer.multigpu import split_model

    ours, _ = pair(3)
    x, enc, audio = torch.randn(1, 24, 2, 2), torch.randn(1, 4, 40), torch.randn(1, 4, 32)
    with torch.no_grad():
        want = ours(x, 300.0, enc, None, audio_states=audio)
        split_model(ours, ["cpu", "cpu"])  # same device twice: exercises the hand-off path
        assert torch.equal(ours(x, 300.0, enc, None, audio_states=audio), want)


def pruned_pair(seed=4):
    """A pruned port whose curve table is the full model's SiLU'd time-embedding curve (rank = t_dim, no truncation):
    the pruned path then computes exactly what the full one does, up to the table's linear interpolation."""
    full, ref = pair(seed)
    grid = 1025
    with torch.no_grad():
        t = torch.linspace(0.0, 1.0, grid)
        curve = torch.nn.functional.silu(full.time_embedding(t))  # [grid, t_dim]
    cfg = dict(TINY, adaln_curve_grid=grid)
    pruned = MiniMaxH3Transformer(**cfg).float().eval()
    state = {k: v for k, v in full.state_dict().items() if not k.startswith("time_embedder.")}
    state["adaln_t_table"] = curve
    assert not any(k.startswith("time_embedder") for k in pruned.state_dict())
    pruned.load_state_dict(state, strict=True)
    return full, pruned, ref


def test_pruned_adaln_table_path():
    full, pruned, _ = pruned_pair()
    x, enc, audio = torch.randn(1, 24, 4, 4), torch.randn(1, 6, 40), torch.randn(1, 4, 32)
    with torch.no_grad():
        for sigma in (1.0, 0.937, 0.5, 0.0):  # on and between grid points, and the ends
            got = pruned(x, sigma * 1000.0, enc, None, audio_states=audio)
            want = full(x, sigma * 1000.0, enc, None, audio_states=audio)
            assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)
        assert pruned.pruned and pruned.config["adaln_curve_grid"] == 1025


def test_layouts_round_trip_through_diffusers_own_converter():
    from diffusers.loaders.single_file_utils import convert_minimax_h3_transformer_checkpoint_to_diffusers as convert

    ours, ref = pair(5)
    want = {k: v.clone() for k, v in ref.state_dict().items()}
    port = {k: v.clone() for k, v in ours.state_dict().items()}
    back = C.port_to_diffusers(port)
    assert set(back) == set(want) and all(torch.equal(back[k], want[k]) for k in want)
    config = dict(TINY_DIFFUSERS)
    # Comfy-Org's single files: the port's names, [q; k; v] stacked, with diffusion_model. and rope.inv_freq
    comfy = {f"diffusion_model.{k}": v.clone() for k, v in port.items()}
    comfy["diffusion_model.rope.inv_freq"] = torch.zeros(4)
    got = convert({k.removeprefix("diffusion_model."): v.clone() for k, v in comfy.items()}, config=config,
                  qkv_layout="stacked")
    assert set(got) == set(want) and all(torch.equal(got[k], want[k]) for k in want)
    assert C.is_h3_state(k.removeprefix("diffusion_model.") for k in comfy)
    loaded = C.original_to_port(dict(comfy), interleaved=False, heads=2, head_dim=32)
    assert set(loaded) == set(port) and all(torch.equal(loaded[k], port[k]) for k in port)
    # MiniMaxAI's own shards: per-head interleaved qkv rows
    inter = {k: (C.interleave_qkv(v, 2, 32) if k.endswith("qkv_proj.weight") else v.clone()) for k, v in port.items()}
    got = convert({k: v.clone() for k, v in inter.items()}, config=config, qkv_layout="interleaved")
    assert all(torch.equal(got[k], want[k]) for k in want)
    loaded = C.original_to_port(inter, interleaved=True, heads=2, head_dim=32)
    assert all(torch.equal(loaded[k], port[k]) for k in port)
    cfg = C.config_from_state(port)
    assert all(cfg[k] == TINY[k] for k in cfg), (cfg, TINY)
    assert C.is_diffusers_state(want) and not C.is_diffusers_state(port)
