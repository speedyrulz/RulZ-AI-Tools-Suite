"""P-H3-Image support without model weights: the released files' headers (PrunaAI/P-H3-Image-Comfy, the diffusers
text connector, ostris' training adapter) against the port; the text connector and prompt encoding against diffusers'
PH3ImageTextConnector math; the single-frame VAE decoder against diffusers' PH3ImageViTDecoder forward (written out
here, on diffusers' own H3 blocks); the transformer configured as P-H3 runs it; the 8-step sampler; detection,
downloads, caches and the setup wizard."""
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
from rulztrainer.h3.model import MiniMaxH3Transformer, audio_sigma  # noqa: E402

pytest.importorskip("diffusers")

from rulztrainer.ph3 import loading as PL  # noqa: E402
from rulztrainer.ph3 import vae as V  # noqa: E402

HEADERS = json.loads((Path(__file__).parent / "fixtures" / "p_h3_image_headers.json").read_text())
PH3 = archs.arch_by_name("p_h3_image")
TINY = C.diffusers_config_to_port(dict(num_attention_heads=2, attention_head_dim=32, hidden_size=48, num_layers=2,
                                       num_refiner_layers=1, ffn_dim=64, in_channels=24, audio_in_channels=32,
                                       patch_size=(1, 2, 2), text_dim=40, freq_dim=256, time_embed_hidden_dim=56,
                                       time_embed_dim=24, rope_freq_dim=4, rope_theta=10000.0))


def _shape(entry):
    return tuple(entry[1])


def _meta_state(header: dict) -> dict:
    """A header's tensors on the meta device (ComfyUI's quant side tensors dropped), for config / key checks."""
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I8": torch.int8, "U8": torch.uint8}
    return {k: torch.empty(_shape(v), dtype=dtypes[v[0]], device="meta") for k, v in header.items()
            if not k.endswith((".comfy_quant", ".weight_scale"))}


def tiny_model(seed=0) -> MiniMaxH3Transformer:
    torch.manual_seed(seed)
    model = MiniMaxH3Transformer(**TINY).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)
    return PL.configure(model)


# ---------------------------------------------------------------- the released files


def test_released_dit_header_fits_the_port():
    state = _meta_state(HEADERS["dit_int8"])
    assert C.is_h3_state(state)
    cfg = C.config_from_state(C.original_to_port(dict(state), interleaved=False, heads=0, head_dim=0))
    assert (cfg["adaln_curve_grid"], cfg["time_embed_dim"]) == (4097, 16)  # Comfy-Org's H3 files: [1025, 8]
    assert (cfg["hidden_size"], cfg["num_layers"], cfg["num_attention_heads"], cfg["text_dim"]) == (5376, 50, 56, 5120)
    with torch.device("meta"):
        model = MiniMaxH3Transformer.from_config(cfg)
    want = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got = {k: _shape(v) for k, v in HEADERS["dit_int8"].items()
           if not k.endswith((".comfy_quant", ".weight_scale")) and k != "rope.inv_freq"}
    assert got == want


def test_released_training_adapter_fits_the_port_names():
    with torch.device("meta"):
        model = MiniMaxH3Transformer(**C.config_from_state(C.original_to_port(
            _meta_state(HEADERS["dit_int8"]), interleaved=False, heads=0, head_dim=0)))
    modules = dict(model.named_modules())
    adapter = HEADERS["adapter"]
    names = {k.split(".lora_")[0][len("diffusion_model."):] for k in adapter}
    assert len(names) == 208 and any(n.startswith("token_refiner.") for n in names)
    for name in names:
        m = modules[name]
        down = _shape(adapter[f"diffusion_model.{name}.lora_A.weight"])
        up = _shape(adapter[f"diffusion_model.{name}.lora_B.weight"])
        assert down == (16, m.in_features) and up == (m.out_features, 16), name
    assert all(".alpha" not in k for k in adapter)  # alpha = rank: the adapter applies at scale 1


def test_released_text_encoder_and_connector_headers():
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

    from rulztrainer.krea2 import single_file as KSF
    from rulztrainer.moss import single_file as sf

    te = HEADERS["te_int8"]
    conn = {k[len(PL.CONNECTOR_PREFIX):]: v for k, v in te.items() if k.startswith(PL.CONNECTOR_PREFIX)}
    assert conn == HEADERS["connector"] and all(v[0] == "F32" for v in conn.values())
    assert _shape(conn["proj.weight"]) == (5120, 5 * 2560)
    text = sf.strip_prefix({k: v for k, v in te.items() if not k.startswith(KSF.TE_DROP + ("model.visual.",))
                            and not k.startswith(PL.CONNECTOR_PREFIX)
                            and not k.endswith((".comfy_quant", ".weight_scale"))}, KSF.TE_PREFIXES)
    cfg = Qwen3VLTextConfig(**json.loads((Path(KSF.__file__).parent / "qwen3vl_4b_text_config.json").read_text()))
    with torch.device("meta"):
        model = Qwen3VLTextModel(cfg)
    want = {k: tuple(v.shape) for k, v in model.state_dict().items() if "inv_freq" not in k}
    assert {k: _shape(v) for k, v in text.items()} == want
    assert cfg.num_hidden_layers == max(PL.HIDDEN_STATE_LAYERS) and cfg.hidden_size * 5 == 12800


def test_released_vae_header_builds():
    from rulztrainer.h3 import single_file as HSF

    header = HEADERS["vae_bf16"]
    assert _shape(header["decoder.proj_out.weight"])[0] == PL.SINGLE_FRAME_PROJ_OUT
    assert "decoder.mask_token" not in header
    state = {k: v for k, v in _meta_state(header).items() if k not in ("latents_mean", "latents_std")}
    state = HSF.vae_original_to_diffusers(state, heads=32)
    layers = sum(1 for k in header if k.endswith(".scale1"))
    config = {**HSF.VAE_CONFIG, "decoder_num_layers": layers}
    with torch.device("meta"):
        vae = HSF.vae_class()(**config)
        vae.decoder = V.decoder_class()(in_channels=24, out_channels=3, patch_size=16, num_layers=layers,
                                        num_attention_heads=32, attention_head_dim=64, num_register_tokens=4,
                                        ffn_mult=4, rope_theta=100.0, rope_dim_ratio=0.75, norm_eps=1e-5)
    want = {k: tuple(v.shape) for k, v in vae.state_dict().items()}
    assert {k: tuple(v.shape) for k, v in state.items()} == want


# ---------------------------------------------------------------- text encoder


def _qwen3vl(layers=36, seed=0):
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

    torch.manual_seed(seed)
    cfg = Qwen3VLTextConfig(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                            rope_parameters={"rope_type": "default", "rope_theta": 5000000.0,
                                             "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    return Qwen3VLTextModel(cfg).eval()


def _connector_state(hidden=16, out=24, seed=1):
    g = torch.Generator().manual_seed(seed)
    width = 5 * hidden
    return {"input_mean": torch.randn(width, generator=g), "input_std": torch.rand(width, generator=g) + 0.5,
            "proj.weight": torch.randn(out, width, generator=g) * 0.1, "output_mean": torch.randn(out, generator=g),
            "output_std": torch.rand(out, generator=g) + 0.5, "sink_embedding": torch.randn(out, generator=g)}


def ph3_text_encoder(seed=0):
    model = _qwen3vl(36, seed)
    return PL.attach_connector(model, PL.build_connector(_connector_state()))


def reference_connector(state, hidden_states):
    """diffusers' PH3ImageTextConnector.forward."""
    x = torch.cat([hidden_states[i] for i in PL.HIDDEN_STATE_LAYERS], dim=-1).float()
    x = (x - state["input_mean"]) / state["input_std"]
    x = (x @ state["proj.weight"].T) * state["output_std"] + state["output_mean"]
    return torch.cat([state["sink_embedding"].expand(x.shape[0], 1, -1), x[:, 1:]], dim=1)


def test_connector_matches_diffusers_and_reads_the_normed_last_state():
    from test_text_batching import _Tok

    model, tok = ph3_text_encoder(), _Tok()
    state = _connector_state()
    ids = tok("a photo of zxc", add_special_tokens=False, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids=ids, position_ids=torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1),
                    output_hidden_states=True)
        want = reference_connector(state, out.hidden_states)
        got = PL.encode_prompt(model, tok, "a photo of zxc")
    assert len(out.hidden_states) == 37 and torch.equal(out.hidden_states[36], out.last_hidden_state)
    assert got.shape == (ids.shape[1], 24) and got.dtype == torch.bfloat16
    assert torch.allclose(got.float(), want[0], atol=3e-2, rtol=2e-2)
    assert torch.allclose(got[0].float(), state["sink_embedding"], atol=2e-2)  # token 0 is the sink


def test_batched_prompts_match_single_and_empty_is_a_space():
    from test_text_batching import PROMPTS, _Tok

    model, tok = ph3_text_encoder(), _Tok()
    batched = PL.encode_prompts(model, tok, PROMPTS, 1024)
    for p, b in zip(PROMPTS, batched):
        s = PL.encode_prompt(model, tok, p, 1024)
        assert b.shape == s.shape and torch.allclose(b.float(), s.float(), atol=2e-2, rtol=2e-2)
    assert batched[PROMPTS.index("")].shape[0] == 1  # " ": one token
    assert PH3._loading.prompt_tokens is __import__("rulztrainer.h3.loading", fromlist=["x"]).prompt_tokens


def test_text_encoder_file_with_connector_and_vision_tower(tmp_path):
    from safetensors.torch import save_file

    from rulztrainer.krea2 import single_file as KSF

    model = _qwen3vl(36, seed=3)
    state = {f"model.language_model.{k}": v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()
             if "inv_freq" not in k}
    state["model.visual.patch_embed.proj.weight"] = torch.zeros(4, 3, 2, 16, 16, dtype=torch.bfloat16)
    conn = _connector_state()
    state.update({f"text_connector.{k}": v.float() for k, v in conn.items()})
    path = tmp_path / "p-h3-image_text_encoder_bf16.safetensors"
    save_file(state, str(path))
    import json as _json

    cfg = _json.loads((Path(KSF.__file__).parent / "qwen3vl_4b_text_config.json").read_text())
    cfg.update(hidden_size=16, intermediate_size=32, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
               vocab_size=128, rope_parameters={"rope_type": "default", "rope_theta": 5000000.0,
                                                "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    real = KSF.build_text_model

    def tiny_build(state, quant="auto", device="cpu", config=None, work_device=None):
        return real(state, quant, device, config=cfg, work_device=work_device)

    import rulztrainer.krea2.single_file as ksf_mod

    orig = ksf_mod.build_text_model
    ksf_mod.build_text_model = tiny_build
    try:
        te = PL.load_text_encoder_file(path, "cpu", "none")
    finally:
        ksf_mod.build_text_model = orig
    assert te.text_connector.proj.weight.dtype == torch.float32  # the connector stays fp32
    assert torch.equal(te.text_connector.proj.weight, conn["proj.weight"])
    from test_text_batching import _Tok

    assert PL.encode_prompt(te, _Tok(), "a cat").shape == (5, 24)


# ---------------------------------------------------------------- VAE

TINY_VAE = dict(block_out_channels=(8, 8, 8, 8, 8, 16), layers_per_block=1, norm_num_groups=4, decoder_num_layers=1,
                decoder_num_attention_heads=2, decoder_attention_head_dim=16)


def tiny_vae(seed=0):
    from rulztrainer.h3 import single_file as HSF

    torch.manual_seed(seed)
    cfg = {**HSF.VAE_CONFIG, **TINY_VAE}
    vae = HSF.vae_class()(**cfg)
    vae.decoder = V.decoder_class()(in_channels=24, out_channels=3, patch_size=16, num_layers=1, num_attention_heads=2,
                                    attention_head_dim=16, num_register_tokens=4, ffn_mult=4, rope_theta=100.0,
                                    rope_dim_ratio=0.75, norm_eps=1e-5)
    vae = vae.float().eval()
    with torch.no_grad():
        for p in vae.parameters():
            p.normal_(0, 0.05)
    return vae


def reference_decoder(dec, z):
    """diffusers' PH3ImageViTDecoder.forward (fp32), on diffusers' own transformer blocks."""
    b, c, _, h, w = z.shape
    x = dec.proj_in(z.permute(0, 2, 3, 4, 1).reshape(b, h * w, c))
    n = x.shape[1]
    x = torch.cat([x, dec.register_tokens.expand(b, -1, -1), torch.zeros_like(x[:, :1])], dim=1)
    grids = [torch.full((1,), V.FRAME_POSITION)] + [2.0 * (torch.arange(0.5, s, dtype=torch.float32) / s) - 1.0
                                                     for s in (h, w)]
    pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2).unsqueeze(0).expand(b, -1, -1)
    pos = torch.cat([pos, pos.new_zeros(b, dec.num_register_tokens + 1, 3)], dim=1)
    rot = dec.rope(pos)
    for block in dec.transformer_blocks:
        x = block(x, rot)
    x = dec.proj_out(dec.norm_out(x))[:, :n]
    return x.view(b, h, w, 3, 16, 16).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, 1, h * 16, w * 16)


def _save_original(vae, path, dtype=torch.float32):
    from safetensors.torch import save_file

    from rulztrainer.h3 import single_file as HSF

    orig = HSF.vae_diffusers_to_original({k: v.clone() for k, v in vae.state_dict().items()}, heads=2)
    orig["latents_mean"] = torch.tensor(HSF.VAE_CONFIG["latents_mean"])
    orig["latents_std"] = torch.tensor(HSF.VAE_CONFIG["latents_std"])
    save_file({k: (v if k.startswith("latents_") else v.to(dtype)).contiguous() for k, v in orig.items()}, str(path))


def test_vae_file_decodes_one_frame_as_diffusers(tmp_path, monkeypatch):
    from rulztrainer.h3 import loading as HL
    from rulztrainer.h3 import single_file as HSF

    vae = tiny_vae(2)
    path = tmp_path / "p-h3-image_vae_bf16.safetensors"
    _save_original(vae, path)
    monkeypatch.setattr(HSF, "VAE_CONFIG", {**HSF.VAE_CONFIG, **TINY_VAE})
    loaded = PL.load_vae_file(path, "cpu", dtype=torch.float32)
    img = torch.rand(1, 3, 64, 96) * 2 - 1
    with torch.no_grad():
        lat = PH3.encode_images(loaded, img)
        assert lat.shape == (1, 24, 4, 6) and torch.isfinite(lat).all()
        out = PH3.decode_latents(loaded, lat)
        mean, std = HL._stats(loaded, lat)
        z = (lat * std + mean).unsqueeze(2)
        frame = reference_decoder(vae.decoder, vae.post_quant_conv(z))[:, :, 0]
        pm, ps = HL._pixel_norm(frame)
        want = ((frame * ps + pm) * 2 - 1).clamp(-1, 1)
    assert out.shape == img.shape and torch.allclose(out, want, atol=1e-5), float((out - want).abs().max())
    # bf16 (fp32 encoder, norms, scales and proj_out): runs, and stays near fp32
    half = PL.load_vae_file(path, "cpu", dtype=torch.bfloat16)
    assert half.decoder.proj_out.weight.dtype == torch.float32 and half.decoder.proj_in.weight.dtype == torch.bfloat16
    with torch.no_grad():
        out16 = PH3.decode_latents(half, lat)
    assert (out16 - out).abs().mean() < 2e-2
    # decoding with gradients (the identity loss) reaches the latents
    lat_g = lat.clone().requires_grad_(True)
    PH3.decode_latents_grad(loaded, lat_g).mean().backward()
    assert lat_g.grad is not None and torch.isfinite(lat_g.grad).all()


def test_h3_video_vae_is_refused(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from rulztrainer.h3 import single_file as HSF

    h3 = HSF.vae_class()(**{**HSF.VAE_CONFIG, **TINY_VAE}).float()
    orig = HSF.vae_diffusers_to_original({k: v.clone() for k, v in h3.state_dict().items()}, heads=2)
    path = tmp_path / "minimax_h3_video_vae_fp16.safetensors"
    save_file({k: v.contiguous() for k, v in orig.items()}, str(path))
    monkeypatch.setattr(HSF, "VAE_CONFIG", {**HSF.VAE_CONFIG, **TINY_VAE})
    with pytest.raises(ValueError, match="single-frame"):
        PL.load_vae_file(path, "cpu")


# ---------------------------------------------------------------- transformer, sampler, adapter


def test_configured_transformer_runs_audio_at_the_video_sigma():
    model = tiny_model()
    s = torch.tensor([0.2, 0.7, 0.95])
    assert torch.allclose(audio_sigma(s, model.shift_video, model.shift_audio), s, atol=1e-6)
    assert torch.count_nonzero(model.audio_silence) == 0 and model.audio_silence.shape == (4, 32)
    # without audio rows: the placeholder is zeros noised to the video sigma
    noise = torch.randn(1, 4, 32)
    rows = model.audio_placeholder(torch.tensor([0.4]), 1, "cpu", noise=noise)
    assert torch.allclose(rows, 0.4 * noise)


def test_sampler_uses_the_checkpoint_grid_on_both_streams():
    from rulztrainer.ph3.sampler import ph3_sigmas, sample

    assert torch.equal(ph3_sigmas(8), torch.tensor(PL.SAMPLE_SIGMAS))
    assert PH3.sample_sigmas(None, 8, 1024, 1024, 5.0).tolist() == pytest.approx(list(PL.SAMPLE_SIGMAS))
    assert ph3_sigmas(4).shape == (5,) and float(ph3_sigmas(4)[-1]) == 0.0  # other counts: H3's shift-12 grid
    model = tiny_model(4)
    enc, mask = torch.randn(1, 5, 40), torch.ones(1, 5, dtype=torch.bool)
    lat = sample(model, enc, mask, None, None, width=64, height=96, steps=8, seed=3, device="cpu")
    gen = torch.Generator("cpu").manual_seed(3)
    x, audio = torch.randn((1, 24, 6, 4), generator=gen), torch.randn((1, 4, 32), generator=gen)
    sig = PL.SAMPLE_SIGMAS
    with torch.no_grad():
        for a, b in zip(sig[:-1], sig[1:]):
            v, va = model(x.to(torch.bfloat16), a * 1000.0, enc, mask, audio_states=audio, return_audio=True)
            x, audio = x + (b - a) * v.float(), audio + (b - a) * va.float()  # the same sigma for audio
    assert torch.allclose(lat, x, atol=1e-5)
    assert PH3.default_sample_steps == 8 and PH3.default_sample_cfg == 1.0 and PH3.distilled_sample_model("")


def test_training_adapter_in_its_published_layout(tmp_path):
    from safetensors.torch import save_file

    model, plain = tiny_model(8), tiny_model(8)
    x, enc, audio = torch.randn(1, 24, 4, 4), torch.randn(1, 5, 40), torch.randn(1, 4, 32)
    a1, b1 = torch.randn(16, 48) * 0.05, torch.randn(192, 16) * 0.05  # rank 16, no alpha (ostris' file)
    adapter = tmp_path / "p_h3_image_training_adapter_v1.safetensors"
    save_file({"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": a1,
               "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": b1}, str(adapter))
    L.inject_lora(model, 2, 2.0, ["q"], names=["blocks.1.attn.qkv_proj"])
    assert L.attach_training_adapter(model, str(adapter), rename=PH3.lora_module_name) == 1
    with torch.no_grad():
        training = model(x, 500.0, enc, None, audio_states=audio)
        plain.blocks[0].attn.qkv_proj.weight += b1 @ a1
        assert torch.allclose(training, plain(x, 500.0, enc, None, audio_states=audio), atol=1e-5)


# ---------------------------------------------------------------- detection, downloads, caches, UI


def _dit_file(path, table):
    from safetensors.torch import save_file

    save_file({"video_patch_proj.weight": torch.zeros(2, 2), "audio_patch_proj.weight": torch.zeros(2, 2),
               "condition_proj.weight": torch.zeros(2, 2), "adaln_t_table": torch.zeros(*table)}, str(path))


def test_detection_downloads_and_caches(tmp_path):
    from PIL import Image

    from rulztrainer import dataset as D
    from rulztrainer.comfy_files import planned_downloads
    from rulztrainer.config import MODEL_PRESETS

    assert MODEL_PRESETS["P-H3-Image"] == "PrunaAI/P-H3-Image"
    for path in ("PrunaAI/P-H3-Image", "D:/models/p-h3-image_int8_convrot.safetensors", "PrunaAI/P-H3-Image-Comfy"):
        assert TrainConfig(model_path=path).arch_name() == "p_h3_image"
    assert TrainConfig(model_path="MiniMaxAI/MiniMax-H3").arch_name() == "minimax_h3"
    # an H3-layout file with a neutral name: P-H3's [4097, 16] AdaLN table tells it from H3's [1025, 8]
    _dit_file(tmp_path / "model_a.safetensors", (4097, 16))
    _dit_file(tmp_path / "model_b.safetensors", (1025, 8))
    assert archs.arch_from_transformer_file(str(tmp_path / "model_a.safetensors")) == "p_h3_image"
    assert archs.arch_from_transformer_file(str(tmp_path / "model_b.safetensors")) == "minimax_h3"
    # the diffusers folder: H3's transformer class, P-H3's pipeline index
    mdir = tmp_path / "PrunaAI__P-H3-Image"
    (mdir / "transformer").mkdir(parents=True)
    (mdir / "transformer" / "config.json").write_text(json.dumps({"_class_name": "MiniMaxH3Transformer3DModel"}))
    (mdir / "modular_model_index.json").write_text(json.dumps({"_class_name": "PH3ImageModularPipeline"}))
    assert archs.detect_arch(TrainConfig(model_path=str(mdir)), mdir).name == "p_h3_image"
    plan = {f: (r, n) for f, r, n in planned_downloads(TrainConfig(model_path="PrunaAI/P-H3-Image"))}
    assert plan == {
        "transformer_file": ("PrunaAI/P-H3-Image-Comfy", "diffusion_models/p-h3-image_int8_convrot.safetensors"),
        "text_encoder_file": ("PrunaAI/P-H3-Image-Comfy",
                              "text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors"),
        "vae_file": ("PrunaAI/P-H3-Image-Comfy", "vae/p-h3-image_vae_bf16.safetensors"),
        "training_adapter": ("ostris/minimax_h3_training_adapter", "p_h3_image_training_adapter_v1.safetensors")}
    bf16 = {f: n for f, _, n in planned_downloads(TrainConfig(model_path="PrunaAI/P-H3-Image", model_file_quant="bf16",
                                                              text_encoder_file_quant="bf16"))}
    assert bf16["transformer_file"].endswith("p-h3-image_bf16.safetensors")
    assert bf16["text_encoder_file"].endswith("p-h3-image_text_encoder_bf16.safetensors")
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (900, 600)).save(data / "a.png")
    cfg = TrainConfig(model_path="PrunaAI/P-H3-Image", dataset_dir=str(data), resolutions=[1.0])
    item = D.scan_dataset(cfg)[0]
    assert item.width % 32 == 0 and item.height % 32 == 0 and item.latent_path.name.endswith("_ph3vae.safetensors")
    assert D.text_cache_path(cfg, "p").name.endswith("_p_h3_image_wo.safetensors")
    assert "transformer/*" in PL.ALLOW_PATTERNS and not any("upsampler" in a for a in PL.ALLOW_PATTERNS)


def test_wizard_recommends_the_ph3_setup():
    import webui

    cfg = TrainConfig(model_path="PrunaAI/P-H3-Image")
    two = webui.wizard_recommendations(cfg, "character", [{"total": 16311}, {"total": 16380}], None)
    assert two["multi_gpu"][0] == "model_split" and two["block_swap"][0] == 0
    assert two["model_file_quant"][0] == "int8_convrot" and two["text_encoder_file_quant"][0] == "int8_convrot"
    assert two["adapter_mode"][0] == "auto" and "ostris" in two["adapter_mode"][1]
    assert (two["sample_steps"][0], two["sample_cfg"][0]) == (8, 1.0)
    assert (two["probe_steps"][0], two["probe_cfg"][0]) == (8, 1.0)
    one = webui.wizard_recommendations(cfg, "style", [{"total": 16311}], None)
    assert one["multi_gpu"][0] == "off" and one["block_swap"][0] > 0
    assert webui.N_BLOCKS["p_h3_image"] == 50 and "p_h3_image" in webui.EXTRA_TARGETS["token_refiner"][0]
    assert set(webui.file_quant_options("p_h3_image", "model")) == {"int8_convrot", "bf16"}
    assert webui.preset_arch("P-H3-Image") == "p_h3_image"
    assert webui.MODEL_NOTES["P-H3-Image"][0] == "Experimental"
