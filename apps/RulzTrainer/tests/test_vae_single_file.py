"""The single-file Flux2 VAE loader accepts both key layouts: diffusers (encoder.down_blocks.i.resnets.j) and the
original BFL / ComfyUI one (encoder.down.i.block.j, decoder.up.i.block.j numbered in reverse, mid.attn_1 1x1 convs)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer.moss import single_file as sf  # noqa: E402

pytest.importorskip("diffusers")
from diffusers import AutoencoderKLFlux2  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

TINY = dict(block_out_channels=(32, 64, 64, 64), decoder_block_out_channels=(32, 32, 64, 64), latent_channels=32,
            norm_num_groups=32, layers_per_block=1, sample_size=64)


def to_original(sd: dict, n: int = 4) -> dict:
    """diffusers -> original layout, the way ComfyUI's convert_vae_state_dict does it (test oracle)."""
    cmap = [("nin_shortcut", "conv_shortcut"), ("norm_out", "conv_norm_out"), ("mid.attn_1.", "mid_block.attentions.0.")]
    for i in range(n):
        for j in range(3):
            cmap.append((f"encoder.down.{i}.block.{j}.", f"encoder.down_blocks.{i}.resnets.{j}."))
            cmap.append((f"decoder.up.{n - 1 - i}.block.{j}.", f"decoder.up_blocks.{i}.resnets.{j}."))
        if i < n - 1:
            cmap.append((f"down.{i}.downsample.", f"down_blocks.{i}.downsamplers.0."))
            cmap.append((f"up.{n - 1 - i}.upsample.", f"up_blocks.{i}.upsamplers.0."))
    for i in range(2):
        cmap.append((f"mid.block_{i + 1}.", f"mid_block.resnets.{i}."))
    amap = [("norm.", "group_norm."), ("q.", "to_q."), ("k.", "to_k."), ("v.", "to_v."), ("proj_out.", "to_out.0.")]
    out = {}
    for k, v in sd.items():
        nk = k
        for sd_part, hf_part in cmap:
            nk = nk.replace(hf_part, sd_part)
        if "attentions" in k:
            for sd_part, hf_part in amap:
                nk = nk.replace(hf_part, sd_part)
        if any(f"mid.attn_1.{w}.weight" in nk for w in ("q", "k", "v", "proj_out")):
            v = v.reshape(*v.shape, 1, 1)
        out[nk] = v
    return out


def test_vae_original_layout_loads_like_diffusers_layout(tmp_path):
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(**TINY)
    with torch.no_grad():  # non-trivial batch-norm statistics so the pass-through of bn.* is checked too
        vae.bn.running_mean.normal_()
        vae.bn.running_var.uniform_(0.5, 2.0)
    sd = {k: v.detach().clone().contiguous() for k, v in vae.state_dict().items()}
    orig = to_original(sd)
    assert "decoder.up.0.block.0.conv1.weight" in orig and not any("up_blocks" in k for k in orig)
    assert orig["decoder.mid.attn_1.q.weight"].ndim == 4
    # the converter is the exact inverse of the ComfyUI direction
    back = sf.vae_original_to_diffusers(orig)
    assert back.keys() == sd.keys()
    assert all(torch.equal(back[k], sd[k]) for k in sd)
    # both files load to the same model, the decoder widths come from the checkpoint, not the config defaults
    save_file(sd, str(tmp_path / "diffusers.safetensors"))
    save_file(orig, str(tmp_path / "original.safetensors"))
    a = sf.load_vae_file(tmp_path / "diffusers.safetensors", "cpu")
    b = sf.load_vae_file(tmp_path / "original.safetensors", "cpu")
    assert a.decoder.conv_in.weight.shape[0] == 64 and a.decoder.conv_out.weight.shape[1] == 32
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(a.encode(x).latent_dist.mode(), b.encode(x).latent_dist.mode())


def test_vae_unknown_layout_is_rejected(tmp_path):
    bad = {"encoder.conv_in.weight": torch.zeros(8, 3, 3, 3), "bn.running_mean": torch.zeros(4),
           "decoder.conv_in.weight": torch.zeros(8, 32, 3, 3)}
    save_file(bad, str(tmp_path / "bad.safetensors"))
    with pytest.raises(ValueError, match="unrecognized VAE key layout"):
        sf.load_vae_file(tmp_path / "bad.safetensors", "cpu")
