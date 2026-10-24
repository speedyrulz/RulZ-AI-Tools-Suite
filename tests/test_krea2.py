"""Krea 2 support without model weights: the ported transformer against diffusers' Krea2Transformer2DModel (tiny
random config), the compacted text layout, the VAE latent layout against Krea2Pipeline's packing, LoRA naming and
PEFT export, the sampler schedule against FlowMatchEulerDiscreteScheduler, and architecture detection / caches."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import dataset as D  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.krea2 import loading as KL  # noqa: E402
from rulztrainer.krea2.model import Krea2Transformer  # noqa: E402
from rulztrainer.krea2.sampler import calculate_shift, krea2_sigmas  # noqa: E402

diffusers = pytest.importorskip("diffusers")

TINY = dict(in_channels=64, num_layers=3, attention_head_dim=16, num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=96, timestep_embed_dim=32, text_hidden_dim=40, num_text_layers=3,
            text_num_attention_heads=4, text_num_key_value_heads=4, text_intermediate_size=56,
            num_layerwise_text_blocks=1, num_refiner_text_blocks=2, axes_dims_rope=(4, 6, 6), rope_theta=1000.0)
KREA = archs.arch_by_name("krea2")


def pair(seed=0):
    """Our port and diffusers' model with the same random weights (norm scales and tables too), fp32."""
    torch.manual_seed(seed)
    ref = diffusers.Krea2Transformer2DModel(**TINY).float().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.2)
    ours = Krea2Transformer(**TINY).float().eval()
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=True)
    assert not missing and not unexpected
    return ours, ref


def ref_call(ref, x, sigma, enc, mask):
    """diffusers' forward on the trainer's [B, 64, h, w] latents (packed to tokens, position ids as the pipeline)."""
    b, c, h, w = x.shape
    ids = diffusers.Krea2Pipeline.prepare_position_ids(enc.shape[1], h, w, x.device)
    out = ref(x.flatten(2).transpose(1, 2), enc, sigma, ids, mask, return_dict=False)[0]
    return out.transpose(1, 2).reshape(b, c, h, w)


def test_port_matches_diffusers_with_padding_and_compaction():
    ours, ref = pair()
    x = torch.randn(2, 64, 3, 4)
    sigma = torch.tensor([0.7, 0.2])
    enc = torch.randn(2, 9, 3, 40)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 4:7] = False  # padded in the middle, as the pipeline lays text out ([prompt | pad | suffix])
    with torch.no_grad():
        want = ref_call(ref, x, sigma, enc, mask)
        got = ours(x, sigma * 1000.0, enc, mask)
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)
        # the cache keeps only the real tokens: dropping the padding gives the same prediction
        compact = ours(x[1:], sigma[1:] * 1000.0, enc[1:, mask[1]], None)
        assert torch.allclose(compact, want[1:], atol=1e-4, rtol=1e-4)
        # an all-true mask is no mask
        assert torch.equal(ours(x[:1], 700.0, enc[:1], mask[:1]), ours(x[:1], 700.0, enc[:1], None))


def test_bf16_gradients_checkpointing_skip_and_layers():
    ours, _ = pair(1)
    m = ours.to(torch.bfloat16)
    m.gradient_checkpointing = True
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.randn(1, 64, 2, 3).to(torch.bfloat16)
    enc = torch.randn(1, 5, 3, 40).to(torch.bfloat16)
    out = m(x, torch.tensor([500.0]), enc, None)
    assert out.shape == x.shape and out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for n, p in m.named_parameters()), [n for n, p in m.named_parameters() if p.grad is None]
    m.eval()
    with torch.no_grad():
        assert not torch.equal(m(x, 500.0, enc, None), m(x, 500.0, enc, None, skip_layers=[1]))
    assert m.layers is m.transformer_blocks and len(m.layers) == 3 and m.latent_channels == 64
    assert m.split_tail_modules() == [m.final_layer] and KREA.compile_blocks(m) == list(m.transformer_blocks)


def test_model_split_hand_off_matches():
    from rulztrainer.multigpu import split_model

    ours, _ = pair(2)
    x, enc = torch.randn(1, 64, 2, 2), torch.randn(1, 4, 3, 40)
    with torch.no_grad():
        want = ours(x, 300.0, enc, None)
        split_model(ours, ["cpu", "cpu"])  # same device twice: exercises the hand-off path
        assert torch.equal(ours(x, 300.0, enc, None), want)


class StubVAE:
    """Stands in for AutoencoderKLQwenImage: 'encodes' to a fixed raw latent, 'decodes' by returning its input."""
    device, dtype = torch.device("cpu"), torch.float32

    def __init__(self, raw):
        self.raw = raw  # [B, 16, 1, h, w]
        self.config = SimpleNamespace(z_dim=16, latents_mean=torch.randn(16).tolist(),
                                      latents_std=(torch.rand(16) + 0.5).tolist())

    def encode(self, images):
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: self.raw))

    def decode(self, z, return_dict=False):
        return (z,)


def test_latent_layout_matches_pipeline_packing():
    raw = torch.randn(1, 16, 1, 6, 8)
    vae = StubVAE(raw)
    lat = KL.encode_images(vae, torch.zeros(1, 3, 48, 64))
    assert lat.shape == (1, 64, 3, 4)
    pipe = diffusers.Krea2Pipeline.__new__(diffusers.Krea2Pipeline)
    pipe.patch_size, pipe.vae_scale_factor = 2, 8
    mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    normalized = ((raw - mean) / std)[:, :, 0]
    packed = pipe._pack_latents(normalized, 1, 16, 6, 8)  # what the pipeline feeds the transformer
    assert torch.allclose(lat.flatten(2).transpose(1, 2), packed, atol=1e-6)
    # decoding inverts it: the pipeline's unpack + de-normalization reaches the VAE as the original raw latent
    back = KL._decode(vae, lat)
    assert torch.allclose(back, raw[:, :, 0], atol=1e-5)
    unpacked = pipe._unpack_latents(packed, 48, 64) * std + mean
    assert torch.allclose(unpacked[:, :, 0], raw[:, :, 0], atol=1e-5)


def test_lora_names_match_krea_layout_and_export(tmp_path):
    ours, ref = pair(3)
    everything = KREA.lora_module_names(ours, list(KREA.target_groups), None)
    linears = [n for n, mod in ours.named_modules() if isinstance(mod, torch.nn.Linear)]
    assert sorted(everything) == sorted(linears)  # all groups together = every Linear, as Krea's own LoRAs adapt
    assert KREA.lora_module_names(ours, ["q"], [0, 2]) == ["transformer_blocks.0.attn.to_q",
                                                           "transformer_blocks.2.attn.to_q"]
    # the default adds the attention gate and the text-fusion stack (3 blocks x 8 Linears + the layer projector)
    default = KREA.lora_module_names(ours, KREA.default_targets, None)
    assert len(default) == 3 * 8 + 3 * 8 + 1
    assert "transformer_blocks.1.attn.to_gate" in default and "text_fusion.projector" in default
    assert not any(n.split(".")[0] in KREA.EMBED for n in default)
    assert TrainConfig(model_path="krea/Krea-2-Raw").resolved_lora_targets() == list(KREA.default_targets)
    assert TrainConfig(lora_targets=["q"], model_path="krea/Krea-2-Raw").resolved_lora_targets() == ["q"]
    names = KREA.lora_module_names(ours, ["q", "attn_gate", "ffn_down", "embed", "text_fusion"], None)
    inj = L.inject_lora(ours, 4, 8.0, ["q"], names=names)
    with torch.no_grad():
        for mod in inj.values():
            mod.lora_up.weight.normal_(0, 0.1)
    path = tmp_path / "krea_lora.safetensors"
    L.save_lora(inj, path, dtype=torch.float32, fmt="peft")
    from safetensors.torch import load_file

    state = load_file(str(path))
    assert "transformer.transformer_blocks.0.attn.to_gate.lora_A.weight" in state
    assert "transformer.final_layer.linear.lora_B.weight" in state and "transformer.img_in.lora_A.weight" in state
    assert "transformer.text_fusion.refiner_blocks.1.attn.to_q.lora_A.weight" in state
    assert "transformer.text_fusion.projector.lora_B.weight" in state
    # merged into diffusers' model the PEFT way (W + alpha / r * B @ A), it predicts what our LoRA model predicts
    with torch.no_grad():
        for name in names:
            a, b = state[f"transformer.{name}.lora_A.weight"], state[f"transformer.{name}.lora_B.weight"]
            ref.get_submodule(name).weight += (8.0 / 4) * b @ a
        x, enc = torch.randn(1, 64, 2, 3), torch.randn(1, 5, 3, 40)
        got = ours(x, 400.0, enc, None)
        want = ref_call(ref, x, torch.tensor([0.4]), enc, None)
    assert torch.allclose(got, want, atol=1e-3, rtol=1e-3)


def test_sampler_schedule_matches_scheduler():
    sched = diffusers.FlowMatchEulerDiscreteScheduler(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=6400,
                                                      max_shift=1.15, shift=1.0, time_shift_type="exponential",
                                                      use_dynamic_shifting=True)
    for steps, tokens, distilled in ((28, 4096, False), (8, 4096, True), (12, 1024, False)):
        sigmas = torch.linspace(1.0, 1 / steps, steps).numpy()
        mu = 1.15 if distilled else calculate_shift(tokens)
        sched.set_timesteps(sigmas=sigmas, mu=mu)
        assert torch.allclose(krea2_sigmas(steps, tokens, distilled), sched.sigmas.float(), atol=1e-5)
    assert abs(calculate_shift(256) - 0.5) < 1e-9 and abs(calculate_shift(6400) - 1.15) < 1e-9
    r = KREA.reference_sigmas(10000, torch.Generator().manual_seed(0), 4096)
    assert r.min() >= 0 and r.max() <= 1 and 0.6 < r.mean() < 0.8  # shifted toward noise at 1 MP


def test_auto_quant_counts_the_gpus_the_model_runs_on(monkeypatch):
    sizes = {0: 16, 1: 16, 2: 24, 3: 48}  # GB per CUDA index
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(KL, "_gpu_gb", lambda d: sizes[torch.device(d).index or 0])
    assert KL.resolve_quant("auto", "cuda:0") == "nvfp4"  # one 16 GB card
    assert KL.resolve_quant("auto", ["cuda:0", "cuda:1"]) == "int8"  # model split over two 16 GB cards
    assert KL.resolve_quant("auto", "cuda:2") == "int8" and KL.resolve_quant("auto", "cuda:3") == "none"
    assert KL.resolve_quant("auto", "cpu") == "nvfp4"  # a sampling model: sized for the smallest GPU
    assert KL.resolve_quant("int8", "cuda:0") == "int8"  # an explicit choice is kept


def test_pad_embeds_detection_and_cache_names(tmp_path):
    rows = [torch.randn(4, 12, 8), torch.randn(6, 12, 8)]
    embeds, mask = KL.pad_embeds(rows)
    assert embeds.shape == (2, 6, 12, 8) and mask.sum(1).tolist() == [4, 6] and torch.equal(embeds[0, :4], rows[0])
    assert TrainConfig(model_path="krea/Krea-2-Raw").arch_name() == "krea2"
    mdir = tmp_path / "model" / "transformer"
    mdir.mkdir(parents=True)
    (mdir / "config.json").write_text(json.dumps({"_class_name": "Krea2Transformer2DModel"}))
    cfg = TrainConfig(model_path=str(tmp_path / "model"))
    assert archs.detect_arch(cfg, tmp_path / "model").name == "krea2"
    data = tmp_path / "data"
    data.mkdir()
    from PIL import Image

    Image.new("RGB", (64, 48)).save(data / "a.png")
    krea = TrainConfig(model_path="krea/Krea-2-Raw", dataset_dir=str(data), resolutions=[0.25], flip_augment=True)
    moss = TrainConfig(dataset_dir=str(data), resolutions=[0.25], flip_augment=True)
    k_item, m_item = D.scan_dataset(krea)[0], D.scan_dataset(moss)[0]
    # another VAE: its own latent files; region maps (16 px cells) and the image's bucket are shared
    assert k_item.latent_path.name == m_item.latent_path.name.replace(".safetensors", "_qwenvae.safetensors")
    assert k_item.latent_flip_path.name.endswith("_flip_qwenvae.safetensors")
    assert k_item.region_path == m_item.region_path
    assert D.text_cache_path(krea, "p").name.endswith("_krea2.safetensors")
