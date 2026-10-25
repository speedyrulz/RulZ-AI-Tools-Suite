"""Krea 2 from single ComfyUI files (rulztrainer.krea2.single_file): the DiT key renames from ComfyUI's comfy/ldm/krea2
layout to ours (checked against key names and shapes read from real ComfyUI files), scaled fp8 weights, the Qwen-Image
VAE in its original (Wan) layout, the text-only Qwen3-VL build, and family detection from a file's header."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer.krea2 import single_file as ksf  # noqa: E402
from rulztrainer.krea2.model import Krea2Transformer  # noqa: E402
from rulztrainer.moss import single_file as sf  # noqa: E402

diffusers = pytest.importorskip("diffusers")
from safetensors.torch import save_file  # noqa: E402

TINY = dict(in_channels=64, num_layers=3, attention_head_dim=16, num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=96, timestep_embed_dim=32, text_hidden_dim=40, num_text_layers=3,
            text_num_attention_heads=4, text_num_key_value_heads=4, text_intermediate_size=56,
            num_layerwise_text_blocks=1, num_refiner_text_blocks=2, axes_dims_rope=(4, 6, 6), rope_theta=1000.0)


def to_comfy(state: dict) -> dict:
    """Our (diffusers) layout -> ComfyUI's, written from comfy/ldm/krea2/model.py's module names (test oracle)."""
    top = {"img_in.": "first.", "time_embed.linear_1.": "tmlp.0.", "time_embed.linear_2.": "tmlp.2.",
           "time_mod_proj.": "tproj.1.", "txt_in.linear_1.": "txtmlp.1.", "txt_in.linear_2.": "txtmlp.3.",
           "final_layer.linear.": "last.linear."}
    exact = {"txt_in.norm.weight": "txtmlp.0.scale", "final_layer.norm.weight": "last.norm.scale",
             "final_layer.scale_shift_table": "last.modulation.lin"}
    inner = ((".attn.to_q.", ".attn.wq."), (".attn.to_k.", ".attn.wk."), (".attn.to_v.", ".attn.wv."),
             (".attn.to_gate.", ".attn.gate."), (".attn.to_out.0.", ".attn.wo."),
             (".attn.norm_q.weight", ".attn.qknorm.qnorm.scale"), (".attn.norm_k.weight", ".attn.qknorm.knorm.scale"),
             (".norm1.weight", ".prenorm.scale"), (".norm2.weight", ".postnorm.scale"), (".ff.", ".mlp."))
    out = {}
    for k, v in state.items():
        if k in exact:
            out[exact[k]] = v
            continue
        hit = next((a for a in top if k.startswith(a)), None)
        if hit:
            out[top[hit] + k[len(hit):]] = v
            continue
        nk = k.replace("transformer_blocks.", "blocks.").replace("text_fusion.", "txtfusion.")
        for a, b in inner:
            nk = nk.replace(a, b)
        if nk.endswith(".scale_shift_table"):
            nk, v = nk[: -len("scale_shift_table")] + "mod.lin", v.reshape(-1)
        out[nk] = v
    return out


def test_dit_from_comfy_layout_matches_our_model(tmp_path):
    torch.manual_seed(0)
    model = Krea2Transformer(**TINY).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
    comfy = to_comfy(model.state_dict())
    assert len(comfy) == len(model.state_dict())
    # key names and shapes as in ComfyUI's own files (read from krea2 int8 / fp8 / bf16 files; dim 64 here, 6144 there)
    for key, shape in {"blocks.0.mod.lin": (6 * 64,), "last.modulation.lin": (2, 64), "txtmlp.0.scale": (40,),
                       "txtmlp.1.weight": (64, 40), "txtmlp.3.bias": (64,), "blocks.0.attn.qknorm.qnorm.scale": (16,),
                       "blocks.0.prenorm.scale": (64,), "blocks.2.postnorm.scale": (64,), "tmlp.0.weight": (64, 32),
                       "tmlp.2.bias": (64,), "tproj.1.weight": (6 * 64, 64), "txtfusion.projector.weight": (1, 3),
                       "first.weight": (64, 64), "first.bias": (64,), "blocks.1.attn.wk.weight": (32, 64),
                       "blocks.1.attn.gate.weight": (64, 64), "blocks.1.mlp.down.weight": (64, 96),
                       "txtfusion.layerwise_blocks.0.attn.wq.weight": (40, 40),
                       "txtfusion.refiner_blocks.1.mlp.up.weight": (56, 40), "last.norm.scale": (64,),
                       "last.linear.weight": (64, 64)}.items():
        assert key in comfy and tuple(comfy[key].shape) == shape, key
    path = tmp_path / "krea2_test.safetensors"
    save_file({"model.diffusion_model." + k: v.contiguous() for k, v in comfy.items()}, str(path))
    loaded = ksf.load_transformer_file(path, "cpu", quant="none", dtype=torch.float32)
    assert loaded.config["num_layers"] == 3 and loaded.config["axes_dims_rope"] == [4, 6, 6]
    assert loaded.config["text_num_attention_heads"] == 4 and loaded.config["num_key_value_heads"] == 2
    assert not loaded.is_distilled
    x, enc = torch.randn(1, 64, 2, 3), torch.randn(1, 5, 3, 40)
    with torch.no_grad():
        assert torch.allclose(loaded(x, 400.0, enc, None), model(x, 400.0, enc, None), rtol=1e-4, atol=1e-4)
    assert ksf.is_krea2_file(path) and archs.arch_from_transformer_file(str(path)) == "krea2"
    # a diffusers-layout file loads as well; "turbo" in the name marks the distilled model
    turbo = tmp_path / "krea2_turbo_test.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(turbo))
    again = ksf.load_transformer_file(turbo, "cpu", quant="none", dtype=torch.float32)
    assert again.is_distilled
    with torch.no_grad():
        assert torch.allclose(again(x, 400.0, enc, None), model(x, 400.0, enc, None), rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError, match="not a Krea 2"):
        ksf.load_transformer_file(_moss_like(tmp_path), "cpu", quant="none")


def _moss_like(tmp_path):
    path = tmp_path / "moss_like.safetensors"
    save_file({"shared_block_modulation.weight": torch.zeros(6, 4), "layers.0.proj.q.weight": torch.zeros(4, 4)},
              str(path))
    return path


def test_family_detection_from_file_headers(tmp_path):
    from rulztrainer.config import TrainConfig

    moss = _moss_like(tmp_path)
    assert archs.arch_from_transformer_file(str(moss)) == "moss_image"
    other = tmp_path / "other.safetensors"
    save_file({"x.weight": torch.zeros(2)}, str(other))
    assert archs.arch_from_transformer_file(str(other)) is None
    assert archs.arch_from_transformer_file(str(tmp_path / "missing.safetensors")) is None
    krea = tmp_path / "k2.safetensors"  # no "krea" in the name: the header decides
    save_file({"txtfusion.projector.weight": torch.zeros(1, 12)}, str(krea))
    cfg = TrainConfig(model_source="files", transformer_file=str(krea))
    assert archs.guess_arch_name(cfg) == "krea2" and archs.detect_arch(cfg, None).name == "krea2"
    assert archs.guess_arch_name(TrainConfig(model_source="files", transformer_file=str(krea), model_arch="moss_image")) \
        == "moss_image"
    with pytest.raises(ValueError, match="holds a krea2"):
        archs.detect_arch(TrainConfig(model_source="files", transformer_file=str(krea), model_arch="moss_image"), None)


def test_read_state_dequantizes_scaled_fp8(tmp_path):
    w = torch.randint(-8, 9, (16, 32)).float() * 0.5  # multiples of 4 after / 0.125: exact in e4m3
    scale = torch.tensor(0.125)
    q = (w / scale).to(torch.float8_e4m3fn)
    path = tmp_path / "fp8.safetensors"
    save_file({"lin.weight": q, "lin.weight_scale": scale, "plain.weight": w.to(torch.bfloat16)}, str(path))
    state = sf.read_state(path, torch.float32)
    assert torch.equal(state["lin.weight"], w) and "lin.weight_scale" not in state
    assert state["plain.weight"].dtype == torch.float32
    # ComfyUI's text encoders carry a comfy_quant of a float8 format next to the same scale
    blob = torch.tensor(list(json.dumps({"format": "float8_e4m3fn", "full_precision_matrix_mult": False}).encode()),
                        dtype=torch.uint8)
    path2 = tmp_path / "fp8_marked.safetensors"
    save_file({"lin.weight": q, "lin.weight_scale": scale, "lin.comfy_quant": blob}, str(path2))
    state = sf.read_state(path2, torch.bfloat16)
    assert torch.equal(state["lin.weight"].float(), w) and "lin.comfy_quant" not in state


def to_wan(state: dict) -> dict:
    """diffusers' AutoencoderKLQwenImage layout -> the original Wan / ComfyUI one (the inverse of diffusers'
    convert_wan_vae_to_diffusers, test oracle)."""
    res = (("norm1.gamma", "residual.0.gamma"), ("conv1.", "residual.2."), ("norm2.gamma", "residual.3.gamma"),
           ("conv2.", "residual.6."), ("conv_shortcut.", "shortcut."))
    out = {}
    for k, v in state.items():
        nk = k
        for part in ("encoder", "decoder"):
            nk = nk.replace(f"{part}.conv_in.", f"{part}.conv1.").replace(f"{part}.norm_out.gamma", f"{part}.head.0.gamma")
            nk = nk.replace(f"{part}.conv_out.", f"{part}.head.2.")
            nk = nk.replace(f"{part}.mid_block.resnets.0.", f"{part}.middle.0.")
            nk = nk.replace(f"{part}.mid_block.resnets.1.", f"{part}.middle.2.")
            nk = nk.replace(f"{part}.mid_block.attentions.0.", f"{part}.middle.1.")
        nk = nk.replace("quant_conv.", "conv1.") if nk.startswith("quant_conv.") else nk
        nk = nk.replace("post_quant_conv.", "conv2.") if nk.startswith("post_quant_conv.") else nk
        if nk.startswith("encoder.down_blocks."):
            nk = nk.replace("encoder.down_blocks.", "encoder.downsamples.")
        elif nk.startswith("decoder.up_blocks."):
            parts = nk.split(".")
            b = int(parts[2])
            if parts[3] == "resnets":
                nk = f"decoder.upsamples.{4 * b + int(parts[4])}." + ".".join(parts[5:])
            else:  # upsamplers.0
                nk = f"decoder.upsamples.{4 * b + 3}." + ".".join(parts[5:])
        if ".downsamples." in nk or ".upsamples." in nk or ".middle." in nk:
            for a, b2 in res:
                nk = nk.replace("." + a, "." + b2)
        out[nk] = v
    return out


def test_vae_from_original_layout_matches_diffusers(tmp_path):
    torch.manual_seed(0)
    ref = diffusers.AutoencoderKLQwenImage(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                                           temperal_downsample=[False, True, True]).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.1)
    wan = to_wan(ref.state_dict())
    assert "decoder.middle.0.residual.0.gamma" in wan and "conv1.weight" in wan and "decoder.upsamples.4.shortcut.weight" in wan
    assert "decoder.upsamples.3.resample.1.weight" in wan and not any("up_blocks" in k for k in wan)
    path = tmp_path / "qwen_image_vae.safetensors"
    save_file({k: v.contiguous() for k, v in wan.items()}, str(path))
    vae = ksf.load_vae_file(path, "cpu")
    x = torch.randn(1, 3, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(vae.encode(x).latent_dist.mode(), ref.encode(x).latent_dist.mode(), atol=1e-6)
    assert vae.config.z_dim == 4 and len(vae.config.latents_mean) == 16  # the real model's statistics travel along


def test_text_only_qwen3vl_from_comfy_layout(tmp_path):
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

    cfg = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
               head_dim=16, vocab_size=512, rms_norm_eps=1e-6, max_position_embeddings=4096, hidden_act="silu",
               attention_bias=False, model_type="qwen3_vl_text", tie_word_embeddings=True, use_cache=False,
               rope_parameters={"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_theta": 10000.0,
                                "rope_type": "default"})
    torch.manual_seed(0)
    ref = Qwen3VLTextModel(Qwen3VLTextConfig(**cfg)).eval()
    for p in ref.parameters():  # bf16 weights, the rotary tables stay fp32 (as from_pretrained(dtype=bf16) leaves them)
        p.data = p.data.to(torch.bfloat16)
    state = {"model.language_model." + k: v for k, v in ref.state_dict().items() if "inv_freq" not in k}
    state["model.visual.patch_embed.proj.weight"] = torch.zeros(4, 3, 2, 16, 16, dtype=torch.bfloat16)
    state["lm_head.weight"] = torch.zeros(512, 64, dtype=torch.bfloat16)
    path = tmp_path / "qwen3vl_text.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(path))
    model = ksf.build_text_model(sf.read_state(path, torch.bfloat16), "auto", "cpu", config=cfg)
    ids = torch.randint(0, 512, (1, 9))
    pos = torch.arange(9).view(1, 1, 9).expand(3, 1, 9)
    with torch.no_grad():
        a = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos, output_hidden_states=True)
        b = ref(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos, output_hidden_states=True)
    assert len(a.hidden_states) == 3
    assert all(torch.equal(x, y) for x, y in zip(a.hidden_states, b.hidden_states))
    assert model.get_input_embeddings().weight.shape == (512, 64)


def test_comfy_layout_sampling_lora_with_bias_offsets(tmp_path):
    """A Krea 2 LoRA in ComfyUI's module names (blocks.N.attn.wq, tproj.1, txtfusion...) with diff_b bias offsets,
    as turbo / distill LoRAs made for ComfyUI are: attached for sampling, it gives the same output as the LoRA merged
    into the weights (ComfyUI's own way of applying it)."""
    import copy

    from rulztrainer import lora as L

    torch.manual_seed(0)
    model = Krea2Transformer(**TINY).float().eval()
    merged = copy.deepcopy(model)
    ours = ["transformer_blocks.0.attn.to_q", "transformer_blocks.2.ff.down", "time_mod_proj", "img_in",
            "text_fusion.refiner_blocks.1.attn.to_gate", "final_layer.linear"]
    comfy_names = {k[: -len(".weight")]: v[: -len(".weight")]
                   for k, v in zip(model.state_dict(), to_comfy(model.state_dict())) if k[: -len(".weight")] in ours}
    assert set(comfy_names) == set(ours) and comfy_names["time_mod_proj"] == "tproj.1"
    state = {}
    with torch.no_grad():
        for name in ours:
            lin = model.get_submodule(name)
            down, up = torch.randn(4, lin.in_features) * 0.1, torch.randn(lin.out_features, 4) * 0.1
            stem = "diffusion_model." + comfy_names[name]
            state[f"{stem}.lora_down.weight"], state[f"{stem}.lora_up.weight"] = down, up
            ref = merged.get_submodule(name)
            ref.weight += 0.5 * (up @ down)
            if lin.bias is not None:
                diff = torch.randn(lin.out_features) * 0.1
                state[f"{stem}.diff_b"] = diff
                ref.bias += 0.5 * diff
    path = tmp_path / "krea2_comfy_lora.safetensors"
    save_file(state, str(path))
    assert L.attach_extra_lora(model, str(path), 0.5) == 0  # without the family's names nothing matches
    n = L.attach_extra_lora(model, str(path), 0.5, rename=archs.arch_by_name("krea2").lora_module_name)
    assert n == len(ours)
    L.set_extras_enabled(model, True)
    x, enc = torch.randn(1, 64, 2, 3), torch.randn(1, 5, 3, 40)
    with torch.no_grad():
        assert torch.allclose(model(x, 400.0, enc, None), merged(x, 400.0, enc, None), atol=1e-5)
        L.set_extras_enabled(model, False)
        assert not torch.allclose(model(x, 400.0, enc, None), merged(x, 400.0, enc, None), atol=1e-3)
    # a turbo LoRA switched on samples with the distilled schedule (fixed shift), as a Turbo model does
    from rulztrainer.krea2.sampler import uses_distilled_schedule

    model.turbo_lora_attached = True
    assert not uses_distilled_schedule(model)  # attached but off (training forwards, probes without it)
    L.set_extras_enabled(model, True)
    assert uses_distilled_schedule(model)


def test_family_sample_defaults_and_krea_guidance():
    """Krea's pipeline samples Raw at guidance_scale 4.5 (cond + g * (cond - uncond)): CFG 5.5 in this trainer's
    convention. Every family's defaults come from one table the WebUI reads too."""
    from rulztrainer.config import ARCH_SAMPLE_DEFAULTS

    for name in ("moss_image", "fibo2", "krea2"):
        a = archs.arch_by_name(name)
        assert (a.default_sample_steps, a.default_sample_cfg) == ARCH_SAMPLE_DEFAULTS[name]
    assert ARCH_SAMPLE_DEFAULTS["krea2"] == (28, 5.5)
    cond, uncond, g = torch.tensor(2.0), torch.tensor(0.5), 4.5
    assert torch.equal(cond + g * (cond - uncond), uncond + (g + 1) * (cond - uncond))


def test_unreadable_lora_kinds_and_sample_settings_are_reported(tmp_path, caplog):
    import logging
    from types import SimpleNamespace

    from rulztrainer import lora as L
    from rulztrainer.config import TrainConfig
    from rulztrainer.trainer import Trainer

    model = Krea2Transformer(**TINY).float()
    path = tmp_path / "loha.safetensors"
    save_file({"diffusion_model.blocks.0.attn.wq.hada_w1_a": torch.zeros(4, 4),
               "diffusion_model.blocks.0.attn.wq.hada_w1_b": torch.zeros(4, 16),
               "diffusion_model.blocks.0.attn.wq.alpha": torch.tensor(4.0)}, str(path))
    with caplog.at_level(logging.WARNING, logger="rulztrainer"):
        assert L.attach_extra_lora(model, str(path), 1.0) == 0
    assert "hada_w1_a" in caplog.text and "cannot apply" in caplog.text

    def warnings_for(**kw):
        caplog.clear()
        fake = SimpleNamespace(cfg=TrainConfig(**kw), arch=archs.arch_by_name("krea2"))
        with caplog.at_level(logging.WARNING, logger="rulztrainer"):
            Trainer._check_sample_settings(fake)
        return caplog.text

    prompts = dict(sample_prompts=["a photo"], probe_every=0)
    assert "overcooked" in warnings_for(turbo_lora="t.safetensors", sample_steps=8, sample_cfg=5.5, **prompts)
    assert warnings_for(turbo_lora="t.safetensors", sample_steps=8, sample_cfg=1.0, **prompts) == ""
    assert warnings_for(sample_model_path="krea/Krea-2-Turbo", sample_steps=8, sample_cfg=1.0, **prompts) == ""
    assert "unfinished" in warnings_for(sample_steps=8, sample_cfg=1.0, **prompts)
    assert warnings_for(sample_steps=28, sample_cfg=5.5, **prompts) == ""
    assert "Probes" in warnings_for(sample_prompts=[], probe_every=25, probe_steps=8, probe_cfg=1.0)


def test_preview_decode_is_tiled_and_matches_the_full_decode():
    """Samples and probes decode in 512 px tiles (the full decode peaks at +7.9 GB at 1 MP); the tiles match the full
    decode closely for this per-pixel-normalized VAE, and encoding the training latents stays untiled."""
    from rulztrainer.krea2 import loading as K

    torch.manual_seed(0)
    vae = diffusers.AutoencoderKLQwenImage(base_dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
                                           attn_scales=[], temperal_downsample=[False, True, True]).eval()
    vae.register_to_config(latents_mean=[0.0] * 16, latents_std=[1.0] * 16)
    lat = torch.randn(1, 64, 40, 40)  # 640 x 640 px: more than one tile
    with torch.no_grad():
        full = K._decode(vae, lat).clamp(-1, 1)
    tiled = K.decode_latents(vae, lat)
    assert not vae.use_tiling  # restored: encode_images never tiles
    assert tiled.shape == full.shape == (1, 3, 640, 640)
    assert (tiled - full).abs().mean() < 0.01
