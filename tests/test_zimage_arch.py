"""CPU checks for Z-Image's shared text encoder machinery."""
import torch

from rulztrainer.flux2 import single_file as fsf


def test_text_encoder_keeps_penultimate_full_model_layer():
    from transformers import Qwen3Config, Qwen3Model

    config = {**fsf.TEXT_CONFIG_4B, "vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
              "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
              "max_window_layers": 4}
    torch.manual_seed(5)
    full = Qwen3Model(Qwen3Config(**config)).eval().to(torch.bfloat16)
    state = {f"model.{key}": value.clone() for key, value in full.state_dict().items()}
    cut = fsf.build_text_model(state, config=config, keep=3)
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        expected = full(input_ids=ids, output_hidden_states=True).hidden_states[-2]
        actual = cut(input_ids=ids, output_hidden_states=True).hidden_states[-1]
    assert len(cut.layers) == 3
    assert torch.equal(actual, expected)


# ---------------------------------------------------------------- shared tiny pieces

import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402,F811

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import config as C  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer.zimage import loading as ZL  # noqa: E402
from rulztrainer.zimage import model as ZM  # noqa: E402
from rulztrainer.zimage import single_file as ZSF  # noqa: E402

TINY_Z = dict(dim=64, n_layers=2, n_refiner_layers=1, n_heads=4, n_kv_heads=4, cap_feat_dim=32,
              axes_dims=(4, 6, 6), axes_lens=(128, 64, 64))
TINY_QWEN = {**fsf.TEXT_CONFIG_4B, "vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
             "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
             "max_window_layers": 4}


def tiny_model(seed=1, **kw):
    m = ZM.ZImageTransformer(**{**TINY_Z, **kw})
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1 + (1.0 if p.ndim == 1 else 0.0))
    return m.eval()


def tiny_inputs(seed=3, h=4, w=6, cap=32):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 64, h, w, generator=g), torch.randn(1, 5, cap, generator=g),
            torch.ones(1, 5, dtype=torch.bool))


class StubTokenizer:
    """Records apply_chat_template's arguments; tokens are character codes (so offsets are simple)."""
    pad_token_id, eos_token_id = 0, 1

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text, truncation=False, max_length=None, add_special_tokens=True, return_offsets_mapping=False):
        ids = [2 + ord(c) % 90 for c in text]
        offsets = [(i, i + 1) for i in range(len(ids))]
        if truncation and max_length is not None:
            ids, offsets = ids[:max_length], offsets[:max_length]

        class Out(dict):
            input_ids = ids

        out = Out(offset_mapping=offsets)
        return out


# ---------------------------------------------------------------- text encoder


def test_klein_default_text_depth_is_unchanged():
    from transformers import Qwen3Config, Qwen3Model

    config = {**TINY_QWEN, "num_hidden_layers": 30, "max_window_layers": 30}
    full = Qwen3Model(Qwen3Config(**config)).eval().to(torch.bfloat16)
    state = {f"model.{k}": v.clone() for k, v in full.state_dict().items()}
    default = fsf.build_text_model(dict(state), config=config)
    assert len(default.layers) == fsf.TE_LAYERS == 27
    assert len(fsf.build_text_model(dict(state), config=config, keep=29).layers) == 29
    with pytest.raises(ValueError):
        fsf.build_text_model(dict(state), config=config, keep=0)


def test_chat_template_has_generation_prompt_and_thinking_on():
    tok = StubTokenizer()
    text = ZL.chat_text(tok, "a cat")
    messages, kw = tok.calls[0]
    assert messages == [{"role": "user", "content": "a cat"}]
    assert kw == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}
    assert text.endswith("<|im_start|>assistant\n") and "a cat" in text
    full, offsets, dropped, cached = ZL.prompt_tokens(tok, "a cat")
    assert full == text and dropped == 0 and cached == len(text) == len(offsets)
    assert len(ZL.tokenize(tok, "x" * 900)) == ZL.MAX_TOKENS  # cut at 512
    assert ZL.prompt_tokens(tok, "x" * 900)[3] == ZL.MAX_TOKENS


def test_encode_prompts_keep_real_tokens_of_the_penultimate_layer():
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(7)
    full = Qwen3Model(Qwen3Config(**TINY_QWEN)).eval().to(torch.bfloat16)
    state = {f"model.{k}": v.clone() for k, v in full.state_dict().items()}
    cut = fsf.build_text_model(state, config=TINY_QWEN, keep=3)
    tok = StubTokenizer()
    rows = ZL.encode_prompts(cut, tok, ["a cat", "a much longer prompt about a dog"], 512)
    assert [r.shape[0] for r in rows] == [len(ZL.tokenize(tok, p)) for p in ("a cat", "a much longer prompt about a dog")]
    for prompt, row in zip(("a cat", "a much longer prompt about a dog"), rows):
        ids = torch.tensor([ZL.tokenize(tok, prompt)])
        with torch.no_grad():
            want = full(input_ids=ids, output_hidden_states=True).hidden_states[-2][0]
        assert row.dtype == torch.bfloat16 and row.shape == want.shape
        assert torch.allclose(row.float(), want.float(), atol=2e-2), (row.float() - want.float()).abs().max()


def test_pad_embeds_pads_to_the_batch_max_with_a_mask():
    a, b = torch.ones(3, 4), torch.full((5, 4), 2.0)
    embeds, mask = ZL.pad_embeds([a, b])
    assert embeds.shape == (2, 5, 4) and mask.dtype == torch.bool
    assert mask.tolist() == [[True] * 3 + [False] * 2, [True] * 5]
    assert torch.equal(embeds[0, :3], a) and not embeds[0, 3:].any() and torch.equal(embeds[1], b)


# ---------------------------------------------------------------- VAE


def tiny_vae(channels=16, seed=2):
    from diffusers import AutoencoderKL

    torch.manual_seed(seed)
    cfg = dict(block_out_channels=[8, 16], down_block_types=["DownEncoderBlock2D"] * 2,
               up_block_types=["UpDecoderBlock2D"] * 2, latent_channels=channels, layers_per_block=1,
               norm_num_groups=4, in_channels=3, out_channels=3, sample_size=32, scaling_factor=0.3611,
               shift_factor=0.1159, use_quant_conv=False, use_post_quant_conv=False, latents_mean=None,
               latents_std=None, act_fn="silu", mid_block_add_attention=True, force_upcast=False)
    return AutoencoderKL(**cfg).eval().requires_grad_(False), cfg


def test_vae_scale_shift_and_patchify_round_trip():
    vae, _ = tiny_vae()
    img = torch.rand(2, 3, 48, 80) * 2 - 1
    z = ZL.encode_images(vae, img)
    with torch.no_grad():
        mean = vae.encode(img).latent_dist.mode()
    assert z.shape == (2, 64, 12, 20) and z.dtype == torch.float32
    assert torch.allclose(ZM.unpatchify(z), (mean - 0.1159) * 0.3611, atol=1e-6)
    assert torch.equal(ZM.patchify(ZM.unpatchify(z)), z)
    dec = ZL.decode_latents(vae, z)
    with torch.no_grad():
        want = vae.decode(mean).sample.clamp(-1, 1)
    assert dec.shape == img.shape and torch.allclose(dec, want, atol=1e-4)
    lat = z.clone().requires_grad_(True)
    ZL.decode_latents_grad(vae, lat).sum().backward()
    assert lat.grad is not None and torch.isfinite(lat.grad).all()


def _to_original_layout(state: dict, n_down: int, n_up: int) -> dict:
    """diffusers AutoencoderKL names -> BFL's ae.safetensors names (the inverse of moss' vae_original_to_diffusers)."""
    out = {}
    for k, v in state.items():
        k = re.sub(r"^encoder\.down_blocks\.(\d+)\.resnets\.", r"encoder.down.\1.block.", k)
        k = re.sub(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.", r"encoder.down.\1.downsample.", k)
        k = re.sub(r"^decoder\.up_blocks\.(\d+)\.resnets\.",
                   lambda m: f"decoder.up.{n_up - 1 - int(m.group(1))}.block.", k)
        k = re.sub(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.",
                   lambda m: f"decoder.up.{n_up - 1 - int(m.group(1))}.upsample.", k)
        k = k.replace("conv_shortcut", "nin_shortcut").replace("conv_norm_out", "norm_out")
        k = re.sub(r"mid_block\.resnets\.(\d)", lambda m: f"mid.block_{int(m.group(1)) + 1}", k)
        for new, old in (("norm", "group_norm"), ("q", "to_q"), ("k", "to_k"), ("v", "to_v"),
                         ("proj_out", "to_out.0")):
            k = k.replace(f"mid_block.attentions.0.{old}.", f"mid.attn_1.{new}.")
        if k.startswith(("encoder.mid.attn_1.", "decoder.mid.attn_1.")) or ".mid.attn_1." in k:
            if v.ndim == 2:
                v = v[:, :, None, None]
        out[k] = v
    return out


@pytest.mark.parametrize("layout", ["diffusers", "original"])
def test_vae_file_loads_in_both_layouts_and_matches(tmp_path, layout):
    from safetensors.torch import save_file

    vae, cfg = tiny_vae()
    state = {k: v.clone().contiguous() for k, v in vae.state_dict().items()}
    if layout == "original":
        state = {k: v.contiguous() for k, v in _to_original_layout(state, 2, 2).items()}
    path = tmp_path / "ae.safetensors"
    save_file(state, str(path))
    loaded = ZL.load_vae_file(path, "cpu", config=cfg)
    img = torch.rand(1, 3, 32, 32) * 2 - 1
    assert torch.allclose(ZL.encode_images(loaded, img), ZL.encode_images(vae, img), atol=1e-5)


def test_flux2_vae_is_refused(tmp_path):
    from safetensors.torch import save_file

    with pytest.raises(ValueError, match="FLUX.2"):
        ZL.check_flux1_vae_shape((512, 32, 3, 3))
    ZL.check_flux1_vae_shape((512, 16, 3, 3))
    with pytest.raises(ValueError, match="not the FLUX.1 VAE"):
        ZL.check_flux1_vae_shape((128, 16, 3, 3))
    vae32, cfg = tiny_vae(channels=32)
    path = tmp_path / "ae.safetensors"
    save_file({k: v.clone().contiguous() for k, v in vae32.state_dict().items()}, str(path))
    with pytest.raises(ValueError, match="FLUX.2"):
        ZL.load_vae_file(path, "cpu", config=cfg)


# ---------------------------------------------------------------- transformer loading, patch layout cross test


def test_state_loads_into_the_facade_from_a_comfy_file_and_from_diffusers_names(tmp_path):
    from safetensors.torch import save_file

    model = tiny_model(cap_feat_dim=2560)  # the kind check wants Qwen3-4B's 2560 text features
    x, enc, mask = tiny_inputs(cap=2560)
    with torch.no_grad():
        want = model(x, 300.0, enc, mask)
    over = dict(axes_dims=(4, 6, 6), axes_lens=(128, 64, 64))
    comfy = {"model.diffusion_model." + k: v.contiguous() for k, v in ZSF.facade_to_comfy(model.state_dict()).items()}
    path = tmp_path / "z_image_turbo_bf16.safetensors"
    save_file(comfy, str(path))
    state = ZL.to_model_layout(ZL.sf.read_state(path, torch.float32), path.name)
    ZSF.check_keys(state, model.state_dict().keys())
    for variant, distilled in (("turbo", True), ("base", False)):
        built = ZL.build_transformer(dict(state), "cpu", "none", torch.float32, variant=variant, overrides=over)
        assert built.is_distilled is distilled and built.zimage_shift == 3.0
        assert not any(p.requires_grad for p in built.parameters()) and len(built.layers) == 2
        with torch.no_grad():
            assert torch.allclose(built(x, 300.0, enc, mask), want, atol=1e-5)
    built = ZL.build_transformer({k: v.clone() for k, v in model.state_dict().items()}, "cpu", "none", torch.float32,
                                 overrides=over)  # diffusers' own names
    with torch.no_grad():
        assert torch.allclose(built(x, 300.0, enc, mask), want, atol=1e-5)
    assert ZL.config_from_state(model.state_dict())["n_layers"] == 2
    with pytest.raises(ValueError, match="do not match"):
        bad = dict(state)
        bad.pop("x_pad_token")
        ZL.build_transformer(bad, "cpu", "none", torch.float32, overrides=over)


def test_omni_and_foreign_files_are_refused_by_name():
    model = tiny_model(cap_feat_dim=2560)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["siglip_embedder.0.weight"] = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="Omni"):
        ZL.to_model_layout(state, "z_image_omni.safetensors")
    lumina = {k: v.clone() for k, v in tiny_model(cap_feat_dim=2304).state_dict().items()}
    with pytest.raises(ValueError, match="not a Z-Image"):
        ZL.to_model_layout(lumina, "lumina.safetensors")


def test_vae_latents_feed_the_facade_like_diffusers_on_the_unpatched_latent():
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "ZImageTransformer2DModel"):
        pytest.skip("diffusers has no ZImageTransformer2DModel")
    model = tiny_model(seed=4)
    d = diffusers.ZImageTransformer2DModel(
        in_channels=16, dim=64, n_layers=2, n_refiner_layers=1, n_heads=4, n_kv_heads=4, norm_eps=1e-5, qk_norm=True,
        cap_feat_dim=32, rope_theta=256.0, t_scale=1000.0, axes_dims=[4, 6, 6], axes_lens=[128, 64, 64])
    d.load_state_dict(model.state_dict(), strict=True)
    d.eval()
    vae, _ = tiny_vae()
    img = torch.rand(1, 3, 48, 80) * 2 - 1  # latent 24 x 40 -> 12 x 20 = 240 image tokens (not a multiple of 32)
    patched = ZL.encode_images(vae, img)
    z = ZM.unpatchify(patched)  # [1, 16, 24, 40], the layout diffusers takes as [C, 1, H, W]
    cap = torch.randn(7, 32, generator=torch.Generator().manual_seed(9))
    sigma = 0.37
    with torch.no_grad():
        ref = -d([z[0].unsqueeze(1)], torch.full((1,), 1.0 - sigma), [cap]).sample[0][:, 0]
        ours = model(patched, sigma * 1000.0, cap[None], torch.ones(1, 7, dtype=torch.bool))
    assert torch.allclose(ZM.unpatchify(ours)[0], ref, atol=1e-5), (ZM.unpatchify(ours)[0] - ref).abs().max()


# ---------------------------------------------------------------- variants, adapter, config


@pytest.mark.parametrize("name,variant", [
    ("Tongyi-MAI/Z-Image-Turbo", "turbo"), ("Tongyi-MAI/Z-Image", "base"), ("z_image_turbo_bf16.safetensors", "turbo"),
    ("z_image_bf16.safetensors", "base"), ("Z-Image-De-Turbo", "base"), ("z_image_de_turbo_v1_bf16.safetensors", "base"),
    ("ostris/Z-Image-DeTurbo", "base"), ("my_finetune.safetensors", "base"), ("", "base"),
    ("C:\\turbo models\\Z-Image", "base")])
def test_variant_resolution_by_name(name, variant):
    assert C.resolve_zimage_variant("auto", name) == variant
    assert C.resolve_zimage_variant("turbo", name) == "turbo" and C.resolve_zimage_variant("base", name) == "base"


def test_adapter_is_planned_for_turbo_only(tmp_path):
    from rulztrainer.comfy_files import planned_downloads

    turbo = C.TrainConfig(model_arch="zimage", model_path="Tongyi-MAI/Z-Image-Turbo", model_source="files")
    base = C.TrainConfig(model_arch="zimage", model_path="Tongyi-MAI/Z-Image", model_source="files")
    fields = lambda cfg: [f for f, _, _ in planned_downloads(cfg)]  # noqa: E731
    assert "training_adapter" in fields(turbo) and "training_adapter" not in fields(base)
    plan = dict((f, (r, n)) for f, r, n in planned_downloads(turbo))
    assert plan["training_adapter"] == ("ShootTheSound/Fizgig-Z-Image-Turbo-Training-Adapter",
                                        "fizgig_z_image_turbo_training_adapter.safetensors")
    assert plan["transformer_file"][0] == "Comfy-Org/z_image_turbo" and plan["vae_file"][1] == "split_files/vae/ae.safetensors"
    assert dict((f, r) for f, r, _ in planned_downloads(base))["transformer_file"] == "Comfy-Org/z_image"
    assert dict((f, r) for f, r, _ in planned_downloads(base))["text_encoder_file"] == "Comfy-Org/z_image_turbo"
    # the variant field wins over the name; a fine-tune and the De-Turbo name get none; None is not a crash
    assert "training_adapter" not in fields(C.TrainConfig(model_arch="zimage", model_path="Tongyi-MAI/Z-Image-Turbo",
                                                          zimage_variant="base"))
    assert "training_adapter" in fields(C.TrainConfig(model_arch="zimage", model_path="my/own", model_source="hf",
                                                       zimage_variant="turbo"))
    assert "training_adapter" not in fields(C.TrainConfig(model_arch="zimage", model_path="Tongyi-MAI/Z-Image-Turbo",
                                                          finetune=True))
    assert C.training_adapter_for("zimage", "Tongyi-MAI/Z-Image") is None
    assert C.training_adapter_for("zimage", "Tongyi-MAI/Z-Image-Turbo")[1].startswith("fizgig_z_image_turbo")
    assert C.training_adapter_for("zimage", "x", base) is None
    assert C.training_adapter_for("qwen_image21", "Qwen/Qwen-Image-2.1") is not None  # other families unchanged
    with pytest.raises(ValueError, match="nvfp4|has bf16"):
        planned_downloads(C.TrainConfig(model_arch="zimage", model_path="Tongyi-MAI/Z-Image", model_source="files",
                                        model_file_quant="nvfp4"))


def adapter_state(model, with_alpha: bool, modulation: bool, rank=3):
    """A fake training adapter over every block Linear, in Fizgig's layout (lora_A / lora_B + alpha) or ostris'
    (no alpha, adaLN_modulation.0 as well)."""
    g = torch.Generator().manual_seed(11)
    suffixes = ["attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0", "feed_forward.w1",
                "feed_forward.w2", "feed_forward.w3"] + (["adaLN_modulation.0"] if modulation else [])
    state, names = {}, []
    for n in range(len(model.layers)):
        for suffix in suffixes:
            name = f"layers.{n}.{suffix}"
            lin = model.get_submodule(name)
            state[f"diffusion_model.{name}.lora_A.weight"] = torch.randn(rank, lin.in_features, generator=g) * 0.1
            state[f"diffusion_model.{name}.lora_B.weight"] = torch.randn(lin.out_features, rank, generator=g) * 0.1
            if with_alpha:
                state[f"diffusion_model.{name}.alpha"] = torch.tensor(float(rank))
            names.append(name)
    return state, names


@pytest.mark.parametrize("fizgig", [True, False])
def test_training_adapter_attaches_on_the_facade(tmp_path, fizgig):
    from safetensors.torch import save_file

    from rulztrainer.archs import arch_by_name

    model = tiny_model(seed=5)
    x, enc, mask = tiny_inputs()
    with torch.no_grad():
        before = model(x, 400.0, enc, mask)
    state, names = adapter_state(model, with_alpha=fizgig, modulation=not fizgig)
    path = tmp_path / "adapter.safetensors"
    save_file(state, str(path))
    arch = arch_by_name("zimage")
    targets = arch.lora_module_names(model, ["q"], None)
    injected = L.inject_lora(model, 2, 2.0, ["q"], names=targets)
    assert L.attach_training_adapter(model, str(path)) == len(names)  # every module matched
    with torch.no_grad():
        training = model(x, 400.0, enc, mask)
        L.set_extras_enabled(model, True)  # a sample / probe: the adapter is off
        sampling = model(x, 400.0, enc, mask)
        L.set_extras_enabled(model, False)
    assert not torch.allclose(training, before, atol=1e-4) and torch.equal(sampling, before)
    saved = L.export_state_dict(injected, torch.float32, "comfy_ab")  # the adapter is never part of the saved LoRA
    assert set(saved) == {f"diffusion_model.{n}.lora_{ab}.weight" for n in targets for ab in "AB"}
    assert not any("feed_forward" in k or "adaLN" in k for k in saved)


def test_exported_lora_keys_match_comfyui_s_z_image_map():
    from rulztrainer.archs import arch_by_name

    model = tiny_model()
    arch = arch_by_name("zimage")
    names = arch.lora_module_names(model, list(arch.default_targets), None)
    assert len(names) == 7 * 2 and not any("refiner" in n or "embedder" in n for n in names)
    assert arch.lora_format == "comfy_ab"
    injected = L.inject_lora(model, 4, 4.0, list(arch.default_targets), names=names)
    keys = L.export_state_dict(injected, torch.float32, "comfy_ab")
    pattern = re.compile(r"^diffusion_model\.layers\.\d+\.(attention\.(to_q|to_k|to_v|to_out\.0)|feed_forward\.w[123])"
                         r"\.lora_(A|B)\.weight$")
    assert len(keys) == 14 * 2 and all(pattern.match(k) for k in keys)
    assert "diffusion_model.layers.0.attention.to_q.lora_A.weight" in keys
    # blocks and the groups that are not default
    only = arch.lora_module_names(tiny_model(), ["q", "modulation"], [1])
    assert sorted(only) == ["layers.1.adaLN_modulation.0", "layers.1.attention.to_q"]
    assert {*arch.GROUPS} <= {*arch.target_groups}


def test_lora_on_the_embedders_trains_without_breaking_the_forward():
    from rulztrainer.archs import arch_by_name

    model = tiny_model(seed=6)
    x, enc, mask = tiny_inputs()
    with torch.no_grad():
        before = model(x, 250.0, enc, mask)
    arch = arch_by_name("zimage")
    names = arch.lora_module_names(model, ["embed"], None)
    assert any(n.startswith("t_embedder.") for n in names) and any(n.startswith("all_x_embedder.2-1") for n in names)
    assert any(n.startswith("cap_embedder.") for n in names) and any(n.startswith("all_final_layer.2-1") for n in names)
    injected = L.inject_lora(model, 2, 2.0, ["embed"], names=names)
    with torch.no_grad():
        assert torch.allclose(model(x, 250.0, enc, mask), before, atol=1e-6)  # zero-init: a no-op at step 0
    model.train()
    out = model(x, 250.0, enc, mask)
    out.square().mean().backward()
    grads = [m.lora_up.weight.grad for m in injected.values() if m.lora_up.weight.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_detection_by_file_and_by_folder(tmp_path):
    from safetensors.torch import save_file

    from rulztrainer import archs

    model = tiny_model(cap_feat_dim=2560)
    comfy = {"model.diffusion_model." + k: v.contiguous() for k, v in ZSF.facade_to_comfy(model.state_dict()).items()}
    path = tmp_path / "some_name.safetensors"
    save_file(comfy, str(path))
    assert archs.arch_from_transformer_file(str(path)) == "zimage"
    assert C.TrainConfig(model_source="files", transformer_file=str(path)).family() == "zimage"
    omni = dict(comfy, **{"model.diffusion_model.siglip_embedder.0.weight": torch.zeros(2, 2)})
    opath = tmp_path / "omni.safetensors"
    save_file(omni, str(opath))
    assert archs.arch_from_transformer_file(str(opath)) == "zimage"  # detection does not throw (WebUI path probes)
    with pytest.raises(ValueError, match="Omni"):  # loading refuses it
        ZL.load_transformer_file(str(opath), "cpu", quant="none", dtype=torch.float32)
    lumina = {"model.diffusion_model." + k: v.contiguous()
              for k, v in ZSF.facade_to_comfy(tiny_model(cap_feat_dim=2304).state_dict()).items()}
    lpath = tmp_path / "lumina.safetensors"
    save_file(lumina, str(lpath))
    assert archs.arch_from_transformer_file(str(lpath)) is None
    folder = tmp_path / "repo"
    (folder / "transformer").mkdir(parents=True)
    (folder / "transformer" / "config.json").write_text(json.dumps({"_class_name": "ZImageTransformer2DModel"}))
    cfg = C.TrainConfig(model_path=str(folder), model_source="hf")
    assert archs.detect_arch(cfg, folder).name == "zimage"
    assert ZL.is_zimage_dir(folder)


def test_config_presets_names_and_validation(caplog):
    assert C.MODEL_PRESETS["Z-Image-Turbo"] == "Tongyi-MAI/Z-Image-Turbo" and C.MODEL_PRESETS["Z-Image"] == "Tongyi-MAI/Z-Image"
    assert "zimage" in C.ARCH_NAMES and C.ARCH_DEFAULT_TARGETS["zimage"] == C._CORE_TARGETS
    for name in ("Tongyi-MAI/Z-Image-Turbo", "z_image_bf16.safetensors", "zimage_turbo_training_adapter_v2", "Z-Image"):
        assert C.family_named_in(name) == "zimage"
    assert C.family_named_in("Qwen/Qwen-Image-2.1") == "qwen_image21" and C.family_named_in("PrunaAI/P-H3-Image") != "zimage"
    assert C.family_named_in("black-forest-labs/FLUX.2-klein-base-4B") == "flux2_klein4b"
    assert C.family_named_in("D:/models/zimage/flux-2-klein-base-4B.safetensors") == "flux2_klein4b"  # last part only
    assert C.family_named_in("D:/models/z-image-tests/flux2_klein_9b.safetensors") == "flux2_klein"
    assert C.TrainConfig().zimage_variant == "auto"
    with pytest.raises(ValueError, match="zimage_variant"):
        C.TrainConfig(model_arch="zimage", zimage_variant="fast").validate()
    log = logging.getLogger("rulztrainer")
    log.propagate = True
    base = C.TrainConfig(model_arch="zimage", model_path="Tongyi-MAI/Z-Image", finetune=True, timestep_shift=1.0,
                         timestep_sampling="reference")
    turbo = C.TrainConfig(model_arch="zimage", model_path="Tongyi-MAI/Z-Image-Turbo", finetune=True, timestep_shift=3.0,
                          timestep_sampling="reference")
    with caplog.at_level(logging.INFO, logger="rulztrainer"):
        base.validate_zimage()
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        turbo.validate_zimage()
    warnings = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "loses its few-step behaviour" in warnings and "Fizgig keeps" in warnings and "timestep_shift 3" in warnings
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="rulztrainer"):  # no shift 3 without reference; an adapter on Base
        C.TrainConfig(model_arch="zimage", model_path="Tongyi-MAI/Z-Image", timestep_sampling="logit_normal",
                      training_adapter="x/zimage_turbo_training_adapter_v2.safetensors").validate_zimage()
    warnings = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "without the static shift 3" in warnings and "do not belong on Base" in warnings
    from rulztrainer import capabilities

    assert capabilities.violations("zimage", C.TrainConfig(model_arch="zimage")) == {}
    assert "zimage_variant" in capabilities.rules_for("moss_image")  # ignored (greyed) on every other family
    assert "zimage_variant" not in capabilities.rules_for("zimage")
    assert C.TrainConfig(model_arch="zimage", model_path="Tongyi-MAI/Z-Image-Turbo").samples_on_distilled_model()
    assert not C.TrainConfig(model_arch="zimage", model_path="Z-Image-De-Turbo").samples_on_distilled_model()
    assert not C.TrainConfig(model_arch="zimage", model_path="Tongyi-MAI/Z-Image").samples_on_distilled_model()
    assert C.TrainConfig(model_arch="zimage", model_path="x/y", zimage_variant="turbo").samples_on_distilled_model()


def test_arch_data_sampler_and_memory_plan():
    from rulztrainer import runplan
    from rulztrainer.archs import arch_by_name
    from rulztrainer.zimage import sampler

    arch = arch_by_name("zimage")
    assert (arch.label, arch.n_blocks, arch.latent_tag, arch.size_multiple, arch.lora_format) == (
        "Z-Image", 30, "_flux1vae", 16, "comfy_ab")
    assert (arch.default_sample_steps, arch.default_sample_cfg) == (25, 4.0) and arch.default_negative
    assert arch.distilled_sample_model("Tongyi-MAI/Z-Image-Turbo") and not arch.distilled_sample_model("Z-Image-De-Turbo")
    assert C.ZIMAGE_VARIANT_DATA["turbo"]["steps"] == 8 and C.ZIMAGE_VARIANT_DATA["turbo"]["cfg"] == 1.0
    sig = sampler.zimage_sigmas(8)
    s = torch.linspace(1, 0, 9)
    assert sig.shape == (9,) and sig[0] == 1.0 and sig[-1] == 0.0
    assert torch.allclose(sig, 3 * s / (1 + 2 * s), atol=1e-6) and (sig[:-1] > sig[1:]).all()
    assert torch.equal(arch.sample_sigmas(type("M", (), {})(), 8, 1024, 1024, 5.0), sig)  # the sampler's shift, not 5
    ref = arch.reference_sigmas(4000, torch.Generator().manual_seed(1), 4096)
    assert ((ref > 0) & (ref < 1)).all() and 0.55 < float(ref.median()) < 0.8  # logit-normal median 0.5 -> 0.75 at shift 3
    assert not arch.reference_sigmas_exact
    shapes = runplan.target_shapes("zimage", tuple(C.ARCH_DEFAULT_TARGETS["zimage"]))
    assert len(shapes) == 30 * 7 and (3840, 3840) in shapes and (10240, 3840) in shapes
    assert runplan.adapter_gb(C.TrainConfig(model_arch="zimage")) > 0
    # the sampler on a tiny facade: Turbo (distilled) ignores the CFG branch, Base uses it; both give the layout
    model = tiny_model(seed=8)
    pos, neg = torch.randn(1, 5, 32), torch.randn(1, 6, 32)
    pm, nm = torch.ones(1, 5, dtype=torch.bool), torch.ones(1, 6, dtype=torch.bool)
    calls = []
    model.register_forward_hook(lambda m, a, o: calls.append(1))
    model.zimage_shift = 3.0
    model.is_distilled = False
    base = arch.sample(model, pos, pm, neg, nm, width=96, height=64, steps=2, cfg=4.0, seed=3, device="cpu")
    assert base.shape == (1, 64, 4, 6) and torch.isfinite(base).all() and len(calls) == 4
    calls.clear()
    model.is_distilled = True
    turbo = arch.sample(model, pos, pm, neg, nm, width=96, height=64, steps=2, cfg=4.0, seed=3, device="cpu")
    assert len(calls) == 2 and turbo.shape == base.shape


def test_checkrun_has_a_z_image_turbo_case():
    from rulztrainer import checkrun

    case = checkrun.case_named("zimage_turbo_lora")
    assert case.families == ("zimage",) and case.overrides["training_adapter"] == "auto"
