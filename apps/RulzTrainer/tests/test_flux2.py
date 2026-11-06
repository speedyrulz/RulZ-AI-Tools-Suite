"""FLUX.2 Klein 9B support without model weights: the ported transformer against diffusers' Flux2Transformer2DModel
(tiny random config in fp32), the key layouts both ways (BFL's against diffusers' own converter, the fp8 files' extra
keys), refusing FLUX.2 dev, LoRA naming and the export diffusers converts, the sampler schedule against diffusers'
scheduler and pipeline, the text features against Qwen3's hidden layers, and detection, caches, downloads and the
wizard."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import alignment as A  # noqa: E402
from rulztrainer import archs  # noqa: E402
from rulztrainer import dataset as D  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer.comfy_files import planned_downloads  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.flux2 import loading as FL  # noqa: E402
from rulztrainer.flux2 import single_file as FSF  # noqa: E402
from rulztrainer.flux2.model import Flux2KleinTransformer  # noqa: E402
from rulztrainer.flux2.sampler import distilled_sigmas, empirical_mu, flux2_sigmas, sample  # noqa: E402

diffusers = pytest.importorskip("diffusers")
pytest.importorskip("diffusers.models.transformers.transformer_flux2")

BASE = "black-forest-labs/FLUX.2-klein-base-9B"
DISTILLED = "black-forest-labs/FLUX.2-klein-9B"
TINY = dict(in_channels=16, context_in_dim=24, hidden_size=32, num_heads=2, depth=2, depth_single_blocks=3,
            axes_dim=(4, 4, 4, 4), timestep_channels=32)


def tiny_diffusers():
    return diffusers.Flux2Transformer2DModel(
        in_channels=16, num_layers=2, num_single_layers=3, attention_head_dim=16, num_attention_heads=2,
        joint_attention_dim=24, timestep_guidance_channels=32, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4),
        rope_theta=2000, guidance_embeds=False)


def random_bfl_state(seed=0):
    torch.manual_seed(seed)
    model = Flux2KleinTransformer(**TINY).float()
    return {k: torch.randn_like(v) * 0.1 for k, v in model.state_dict().items()}


def pair(seed=0):
    """Our port from BFL-named weights, and diffusers' model from the same weights through diffusers' own converter."""
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers

    state = random_bfl_state(seed)
    ours = Flux2KleinTransformer(**TINY).float().eval()
    ours.load_state_dict(state, strict=True)
    ref = tiny_diffusers().float().eval()
    converted = convert_flux2_transformer_checkpoint_to_diffusers(dict(state))
    missing, unexpected = ref.load_state_dict(converted, strict=True)
    assert not missing and not unexpected
    return ours, ref, state


def ref_call(ref, x, sigma, enc):
    _, c, h, w = x.shape
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    img_ids = torch.stack([torch.zeros(h * w).long(), rows.flatten(), cols.flatten(), torch.zeros(h * w).long()], -1)
    txt_ids = torch.stack([torch.zeros(enc.shape[1]).long()] * 3 + [torch.arange(enc.shape[1])], -1)
    out = ref(hidden_states=x.flatten(2).transpose(1, 2), encoder_hidden_states=enc, timestep=sigma, img_ids=img_ids,
              txt_ids=txt_ids, guidance=None, return_dict=False)[0]
    return out.transpose(1, 2).reshape(1, c, h, w)


@pytest.mark.parametrize("sigma", [0.1, 0.73])
def test_port_matches_diffusers(sigma):
    ours, ref, _ = pair()
    torch.manual_seed(1)
    x, enc = torch.randn(1, 16, 4, 6), torch.randn(1, 9, 24)
    with torch.no_grad():
        got = ours(x, torch.tensor(sigma * 1000), enc, torch.ones(1, 9, dtype=torch.bool))
        want = ref_call(ref, x, torch.tensor([sigma]), enc)
    assert got.abs().mean() > 1e-3
    assert torch.allclose(got, want, atol=2e-5, rtol=2e-4)


def test_batch_matches_each_sample_alone():
    ours, _, _ = pair()
    torch.manual_seed(2)
    x, enc = torch.randn(2, 16, 4, 4), torch.randn(2, 8, 24)
    sigma = torch.tensor([300.0, 800.0])
    with torch.no_grad():
        both = ours(x, sigma, enc)
        alone = torch.cat([ours(x[i:i + 1], sigma[i:i + 1], enc[i:i + 1]) for i in range(2)])
    assert torch.allclose(both, alone, atol=1e-5)


def test_bf16_gradients_checkpointing_and_block_list():
    model = Flux2KleinTransformer(**TINY).to(torch.bfloat16).train().requires_grad_(False)
    assert len(model.layers) == 5 and model.layers[0] is model.double_blocks[0] and model.layers[2] is model.single_blocks[0]
    assert not any("_chain" in k or k.startswith("layers.") for k in model.state_dict())
    assert A.model_dim(model) == 32
    arch = archs.arch_by_name("flux2_klein")
    names = arch.lora_module_names(model, arch.default_targets, None)
    L.inject_lora(model, 4, 4.0, arch.default_targets, names=names)
    x, enc = torch.randn(1, 16, 2, 2, dtype=torch.bfloat16), torch.randn(1, 6, 24, dtype=torch.bfloat16)
    plain = model(x, 500.0, enc)
    assert plain.dtype == torch.bfloat16 and plain.shape == x.shape
    model.gradient_checkpointing = True
    assert torch.allclose(model(x, 500.0, enc).float(), plain.float(), atol=1e-2)
    skipped = model(x, 500.0, enc, skip_layers=[1, 3])
    assert not torch.allclose(skipped.float(), plain.float(), atol=1e-4)
    model(x, 500.0, enc).float().pow(2).mean().backward()
    trained = [p for p in model.parameters() if p.requires_grad]
    assert trained and any(p.grad is not None and p.grad.abs().sum() > 0 for p in trained)


def test_deepcopy_keeps_the_block_list_on_its_own_blocks():
    model = Flux2KleinTransformer(**TINY)
    other = copy.deepcopy(model)
    assert other.layers[0] is other.double_blocks[0] and other.layers[0] is not model.layers[0]


def test_model_split_places_each_block_and_matches():
    from rulztrainer.multigpu import apply_assignment

    ours, _, _ = pair()
    torch.manual_seed(3)
    x, enc = torch.randn(1, 16, 4, 4), torch.randn(1, 7, 24)
    with torch.no_grad():
        whole = ours(x, 400.0, enc)
    moved, real = [], torch.nn.Module.to

    def spy(self, *args, **kw):
        moved.append(self)
        return real(self, *args, **kw)

    torch.nn.Module.to = spy
    try:
        apply_assignment(ours, ["cpu"] * 5, primary="cpu")
    finally:
        torch.nn.Module.to = real
    assert ours.double_blocks not in moved and ours.single_blocks not in moved  # placed block by block, not whole
    with torch.no_grad():
        assert torch.allclose(ours(x, 400.0, enc), whole, atol=1e-6)


def test_alignment_taps_see_text_then_image():
    model = Flux2KleinTransformer(**TINY).float().eval()
    seen = {}
    for i in (0, 3):
        block = model.layers[i]
        block._rz_tap = lambda q, k, i=i: seen.__setitem__(i, q.shape)
    with torch.no_grad():
        model(torch.randn(1, 16, 3, 4), 500.0, torch.randn(1, 5, 24))
    assert seen[0] == seen[3] == torch.Size([1, 2, 5 + 12, 16])


def test_diffusers_layout_is_the_inverse_of_diffusers_converter():
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers

    state = random_bfl_state(4)
    back = FSF.diffusers_to_bfl(convert_flux2_transformer_checkpoint_to_diffusers(dict(state)))
    assert set(back) == set(state)
    for k, v in state.items():
        assert torch.equal(back[k], v), k


def test_real_size_config_is_the_9b_model(tmp_path):
    with torch.device("meta"):
        model = Flux2KleinTransformer()
    state = model.state_dict()
    assert 9.0e9 < sum(v.numel() for v in state.values()) < 9.2e9  # the 16.9 GiB bf16 file
    assert FSF.transformer_config_from_state(state) == {k: model.config[k] for k in (
        "in_channels", "context_in_dim", "hidden_size", "num_heads", "depth", "depth_single_blocks", "axes_dim",
        "mlp_ratio", "timestep_channels")}


def test_single_files_bfl_fp8_keys_and_diffusers_folder(tmp_path, monkeypatch):
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
    from safetensors.torch import save_file

    monkeypatch.setattr(FSF, "TEXT_DIM_9B", TINY["context_in_dim"])
    state = random_bfl_state(5)
    # the fp8 repos: fp8 weights with a weight_scale, and an input_scale the bf16 forward has no use for
    fp8 = {f"diffusion_model.{k}": v.clone() for k, v in state.items()}
    fp8["diffusion_model.img_in.weight"] = state["img_in.weight"].to(torch.float8_e4m3fn)
    fp8["diffusion_model.img_in.weight_scale"] = torch.tensor(1.0)
    fp8["diffusion_model.img_in.input_scale"] = torch.tensor(2.0)
    base, distilled = tmp_path / "flux-2-klein-base-9b-fp8.safetensors", tmp_path / "flux-2-klein-9b-fp8.safetensors"
    save_file(fp8, base)
    save_file(fp8, distilled)
    m = FSF.load_transformer_file(base, "cpu", "none", torch.float32)
    assert not m.is_distilled and m.config["depth"] == 2 and m.config["depth_single_blocks"] == 3
    assert torch.allclose(m.state_dict()["txt_in.weight"].float(), state["txt_in.weight"], atol=1e-3)
    assert FSF.load_transformer_file(distilled, "cpu", "none", torch.float32).is_distilled
    assert archs.arch_from_transformer_file(str(base)) == "flux2_klein"

    folder = tmp_path / "klein" / "transformer"
    folder.mkdir(parents=True)
    converted = convert_flux2_transformer_checkpoint_to_diffusers({k: v.clone() for k, v in state.items()})
    save_file({k: v.contiguous() for k, v in converted.items()}, folder / "diffusion_pytorch_model.safetensors")
    (folder / "config.json").write_text(json.dumps({"_class_name": "Flux2Transformer2DModel"}))
    assert archs.arch_from_transformer_file(str(folder / "diffusion_pytorch_model.safetensors")) == "flux2_klein"
    m = FL.load_transformer(folder.parent, "cpu", "none", torch.float32)
    for k, v in state.items():
        assert torch.allclose(m.state_dict()[k].float(), v, atol=1e-3), k
    detected = archs.detect_arch(TrainConfig(model_path=str(folder.parent)), folder.parent)
    assert detected.name == "flux2_klein"


def test_refuses_flux2_dev_and_klein_4b():
    state = random_bfl_state(6)
    for width in (15360, 7680):  # FLUX.2 dev reads Mistral, Klein 4B reads Qwen3-4B
        wrong = {**state, "txt_in.weight": torch.zeros(32, width)}
        with pytest.raises(ValueError, match="not FLUX.2 Klein 9B"):
            FSF.to_model_layout(wrong)
    with pytest.raises(ValueError, match="not FLUX.2 Klein 9B"):
        FSF.to_model_layout({**state, "txt_in.weight": torch.zeros(32, 12288),
                             "guidance_in.in_layer.weight": torch.zeros(32, 32)})
    assert FSF.to_model_layout({**state, "txt_in.weight": torch.zeros(32, 12288)})


@pytest.mark.parametrize("quant", ["int8_convrot", "nvfp4"])
def test_quantized_fused_layers_load_and_follow_the_float_model(tmp_path, quant):
    """Wide enough for the quantizers (Q.can_quantize), the 9B text width, the fused qkv and linear1 included."""
    from safetensors.torch import save_file

    from rulztrainer import quant as Q

    cfg = dict(in_channels=128, context_in_dim=12288, hidden_size=1024, num_heads=8, depth=1, depth_single_blocks=1)
    torch.manual_seed(7)
    model = Flux2KleinTransformer(**cfg).float()
    state = {k: (torch.randn_like(v) * 0.02).clone() for k, v in model.state_dict().items()}
    path = tmp_path / "flux-2-klein-base-9b.safetensors"
    save_file(state, path)
    float_model = FSF.load_transformer_file(path, "cpu", "none", torch.float32)
    quantized = FSF.load_transformer_file(path, "cpu", quant)
    fused = [quantized.double_blocks[0].img_attn.qkv, quantized.single_blocks[0].linear1]
    assert all(isinstance(m, Q.QuantLinear) for m in fused)
    assert not isinstance(quantized.double_stream_modulation_img.lin, Q.QuantLinear)
    x, enc = torch.randn(1, 128, 2, 2), torch.randn(1, 4, 12288)
    with torch.no_grad():
        want = float_model(x, 500.0, enc)
        got = quantized(x.to(torch.bfloat16), 500.0, enc).float()
    assert (got - want).abs().mean() < 0.25 * want.abs().mean()


def test_lora_names_and_block_numbering():
    arch = archs.arch_by_name("flux2_klein")
    model = Flux2KleinTransformer(**TINY)
    names = arch.lora_module_names(model, arch.default_targets, None)
    per_double = ("img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2", "txt_attn.qkv", "txt_attn.proj",
                  "txt_mlp.0", "txt_mlp.2")
    want = [f"double_blocks.{i}.{m}" for i in range(2) for m in per_double]
    want += [f"single_blocks.{i}.{m}" for i in range(3) for m in ("linear1", "linear2")]
    assert sorted(names) == sorted(want)
    assert arch.lora_module_names(model, ["ffn_down"], [1, 2]) == ["double_blocks.1.img_mlp.2",
                                                                    "double_blocks.1.txt_mlp.2",
                                                                    "single_blocks.0.linear2"]
    extra = arch.lora_module_names(model, ["modulation", "embed", "text_in"], None)
    assert "txt_in" in extra and "single_stream_modulation.lin" in extra and "img_in" in extra
    assert arch.param_group_of("double_blocks.0.txt_attn.qkv") == "text"
    assert arch.param_group_of("double_blocks.0.img_attn.qkv") == "attn"
    assert arch.param_group_of("double_blocks.0.img_mlp.2") == "mlp"
    assert arch.lora_format == "folded" and arch.n_blocks == 32


def test_folded_lora_keys_go_through_diffusers_converter():
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_flux2_lora_to_diffusers

    model = Flux2KleinTransformer(**TINY)
    rank = 4
    state = {}
    for name, m in model.named_modules():
        if name.endswith(("qkv", "proj", "img_mlp.0", "img_mlp.2", "txt_mlp.0", "txt_mlp.2", "linear1", "linear2")):
            state[f"diffusion_model.{name}.lora_A.weight"] = torch.zeros(rank, m.in_features)
            state[f"diffusion_model.{name}.lora_B.weight"] = torch.zeros(m.out_features, rank)
    converted = _convert_non_diffusers_flux2_lora_to_diffusers(state)
    assert any("to_q.lora_A" in k for k in converted) and any("to_qkv_mlp_proj.lora_B" in k for k in converted)


def test_sampler_matches_diffusers_schedule():
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

    for tokens, steps in ((4096, 20), (2304, 50), (6000, 20)):
        assert empirical_mu(tokens, steps) == pytest.approx(compute_empirical_mu(tokens, steps), rel=1e-6)
        s = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, time_shift_type="exponential")
        s.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), mu=empirical_mu(tokens, steps))
        assert torch.allclose(flux2_sigmas(steps, tokens), s.sigmas.float(), atol=1e-4)
    d = distilled_sigmas(4)
    assert d[0] == 1.0 and d[-1] == 0.0 and len(d) == 5 and bool((d[:-1] > d[1:]).all())
    e = 2.718281828 ** 2.02
    assert d[2] == pytest.approx(e / (e + 1.0), rel=1e-4)  # t = 0.5 under exp(2.02), ComfyUI's Euler simple


def test_sampling_steps_guidance_and_distilled(monkeypatch):
    calls = []

    class Fake:
        latent_channels, is_distilled = 16, False

        def __call__(self, x, t, e, m):
            calls.append(float(t))
            return torch.zeros_like(x) + e.mean()

    e, m = torch.ones(1, 5, 24), torch.ones(1, 5, dtype=torch.bool)
    out = sample(Fake(), e, m, e * 0, m, width=64, height=48, steps=3, cfg=3.5, device="cpu")
    assert out.shape == (1, 16, 3, 4) and len(calls) == 6  # cond and uncond per step
    calls.clear()
    fake = Fake()
    fake.is_distilled = True
    sample(fake, e, m, e * 0, m, width=64, height=48, steps=4, cfg=3.5, device="cpu")
    assert len(calls) == 4  # never guided
    assert calls[0] == pytest.approx(1000.0)


def test_reference_sigmas_follow_the_token_count():
    arch = archs.arch_by_name("flux2_klein")
    gen = torch.Generator().manual_seed(0)
    small = arch.reference_sigmas(4000, gen, 256).mean()
    large = arch.reference_sigmas(4000, gen, 4096).mean()
    assert 0.5 < small < large < 0.9  # the larger the image, the noisier
    assert arch.reference_sigmas(10, gen, 1024).shape == (10,)


class StubTok:
    pad_token_id, eos_token_id = 0, 1

    def __call__(self, text, truncation=False, max_length=None, return_offsets_mapping=False):
        ids = [2 + (ord(c) % 90) for c in text]
        ids = ids[:max_length] if truncation else ids
        out = type("Enc", (dict,), {})(offset_mapping=[(i, i + 1) for i in range(len(text))])
        out.input_ids = ids
        return out


def tiny_qwen3():
    from transformers import Qwen3Config, Qwen3Model

    cfg = Qwen3Config(**{**FSF.TEXT_CONFIG, "vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
                         "num_hidden_layers": 30, "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                         "max_window_layers": 30})
    torch.manual_seed(8)
    return Qwen3Model(cfg).eval()


def test_text_features_are_layers_9_18_27_right_padded():
    te = tiny_qwen3()
    tok = StubTok()
    a, b = FL.encode_prompts(te, tok, ["a cat", "a longer prompt about a dog"], 512)
    assert a.shape == b.shape == (512, 48) and a.dtype == torch.bfloat16
    ids, mask = FL.tokenize(tok, "a cat", 512)
    assert ids.shape == mask.shape == (1, 512) and mask[0, 0] == 1 and mask[0, -1] == 0
    assert FL.chat_text("x").endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    with torch.no_grad():
        out = te(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    want = torch.cat([out.hidden_states[k][0] for k in (9, 18, 27)], dim=-1).to(torch.bfloat16)
    assert torch.allclose(a.float(), want.float(), atol=2e-2)
    cut = FL.tokenize(tok, "x" * 2000, 1024)[1]
    assert int(cut.sum()) == 512 and cut.shape[1] == 512  # max_token_length above 512 still stops at 512
    text, offsets, drop, length = FL.prompt_tokens(tok, "a cat")
    assert text == FL.chat_text("a cat") and drop == 0 and length == 512 and len(offsets) == len(text)


@pytest.mark.parametrize("rotary_takes_device", [True, False])
def test_text_model_builds_from_the_real_key_layout(monkeypatch, rotary_takes_device):
    """Comfy-Org's qwen_3_8b.safetensors: model.* with an lm_head to leave out. Some transformers
    releases dropped the device argument of the rotary embedding."""
    if not rotary_takes_device:
        from transformers.models.qwen3 import modeling_qwen3 as M

        original = M.Qwen3RotaryEmbedding.__init__
        monkeypatch.setattr(M.Qwen3RotaryEmbedding, "__init__", lambda self, config: original(self, config))
    te = tiny_qwen3()
    state = {f"model.{k}": v.clone() for k, v in te.state_dict().items()}
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    cfg_real = dict(FSF.TEXT_CONFIG)
    FSF.TEXT_CONFIG.update({"vocab_size": 100, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 30,
                            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8, "max_window_layers": 30})
    try:
        model = FSF.build_text_model(state, "none")
    finally:
        FSF.TEXT_CONFIG.clear()
        FSF.TEXT_CONFIG.update(cfg_real)
    assert model.rotary_emb.inv_freq.device.type == "cpu"
    assert model.config.rope_parameters["rope_theta"] == 1000000.0
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        assert torch.allclose(model.float()(input_ids=ids).last_hidden_state,
                              te(input_ids=ids).last_hidden_state, atol=2e-2)


def test_detection_caches_buckets_and_downloads(tmp_path):
    from rulztrainer.config import ARCH_SAMPLE_DEFAULTS, ARCH_TURBO

    for path in (BASE, DISTILLED, "C:/m/flux-2-klein-base-9b-fp8.safetensors"):
        assert TrainConfig(model_path=path).arch_name() == "flux2_klein"
    assert TrainConfig(model_path="black-forest-labs/flux-3-image-klein-base").arch_name() == "flux3_klein"
    assert TrainConfig(model_path=BASE).resolved_lora_targets() == list(archs.arch_by_name("flux2_klein").default_targets)
    data = tmp_path / "data"
    data.mkdir()
    from PIL import Image

    Image.new("RGB", (900, 600)).save(data / "a.png")
    klein = TrainConfig(model_path=BASE, dataset_dir=str(data), resolutions=[1.0])
    fibo = TrainConfig(model_path="briaai/Fibo-2.0-dev", dataset_dir=str(data), resolutions=[1.0])
    item, fitem = D.scan_dataset(klein)[0], D.scan_dataset(fibo)[0]
    assert item.width % 16 == 0 and item.height % 16 == 0
    assert item.latent_path == fitem.latent_path  # the same Flux2 VAE latents as FIBO 2.0 / MOSS-Image
    assert D.text_cache_path(klein, "p").name.endswith("_flux2_klein.safetensors")
    comfy = "Comfy-Org/Flux-2-Klein-9B"
    plan = {f: (r, n) for f, r, n in planned_downloads(TrainConfig(model_path=BASE))}
    assert plan == {
        "transformer_file": ("black-forest-labs/FLUX.2-klein-base-9b-fp8", "flux-2-klein-base-9b-fp8.safetensors"),
        "text_encoder_file": (comfy, "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"),
        "vae_file": (comfy, "split_files/vae/flux2-vae.safetensors")}  # no turbo LoRA, no training adapter
    plan = {f: (r, n) for f, r, n in planned_downloads(TrainConfig(model_path=DISTILLED, model_file_quant="bf16",
                                                                   text_encoder_file_quant="bf16"))}
    assert plan["transformer_file"] == ("black-forest-labs/FLUX.2-klein-9B", "flux-2-klein-9b.safetensors")
    assert plan["text_encoder_file"][1] == "split_files/text_encoders/qwen_3_8b.safetensors"
    with pytest.raises(ValueError, match="model_file_quant"):
        planned_downloads(TrainConfig(model_path=BASE, model_file_quant="nvfp4"))
    assert "flux2_klein" not in ARCH_TURBO and "flux2_klein" not in ARCH_SAMPLE_DEFAULTS


def test_distilled_naming_and_gated_help():
    from rulztrainer.flux3.loading import gated_error

    assert TrainConfig(model_path=BASE, sample_model_path=DISTILLED).samples_on_distilled_model()
    assert not TrainConfig(model_path=BASE).samples_on_distilled_model()
    assert FSF.is_distilled_name("C:\\m\\flux-2-klein-9b-fp8.safetensors")
    assert not FSF.is_distilled_name("flux-2-klein-base-9b.safetensors")
    text = str(gated_error("black-forest-labs/FLUX.2-klein-base-9b-fp8", RuntimeError("401")))
    assert "FLUX Non-Commercial License" in text and "FLUX.2 Klein" in text
    text = str(gated_error("black-forest-labs/flux-3-image-klein-base", RuntimeError("401")))
    assert "FLUX Kommunity License" in text


def test_auto_quant_counts_the_gpus_the_model_runs_on(monkeypatch):
    gb = {0: 16.0, 1: 16.0, 2: 24.0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(FL, "_gpu_gb", lambda d: gb[torch.device(d).index])
    assert FL.resolve_quant("auto", "cuda:0") == "nvfp4"
    assert FL.resolve_quant("auto", "cuda:2") == "int8_convrot"
    assert FL.resolve_quant("auto", ["cuda:0", "cuda:1", "cuda:2"]) == "none"
    assert FL.resolve_quant("int8_convrot", "cuda:0") == "int8_convrot"
    assert FL.resolve_text_quant("auto") == "none"  # the 24 GB card holds the 15 GB encoder
    gb[2] = 16.0
    assert FL.resolve_text_quant("auto") == "int8_convrot"


def test_wizard_review_and_block_swap(tmp_path):
    from PIL import Image

    webui = pytest.importorskip("webui")
    a = tmp_path / "a"
    a.mkdir()
    for i in range(20):
        Image.new("RGB", (64, 48)).save(a / f"{i}.png")
    data = webui.datasets_summary([str(a)])
    base = dict(model_path=BASE, datasets=[{"path": str(a)}])
    rec = webui.wizard_recommendations(TrainConfig(**base), "character", [{"total": 24000}], data)
    assert rec["lora_rank"][0] == 16 and rec["lora_alpha"][0] == 16.0 and rec["learning_rate"][0] == 1e-4
    assert (rec["sample_steps"][0], rec["sample_cfg"][0]) == (20, 3.5)
    assert (rec["probe_steps"][0], rec["probe_cfg"][0]) == (20, 3.5)
    assert "int8" in rec["transformer_quant"][1]
    distilled = TrainConfig(**base, sample_model_path=DISTILLED)
    rec = webui.wizard_recommendations(distilled, "character", [{"total": 24000}], data)
    assert (rec["sample_steps"][0], rec["sample_cfg"][0]) == (4, 1.0) and rec["probe_cfg"][0] == 1.0
    assert webui.N_BLOCKS["flux2_klein"] == archs.arch_by_name("flux2_klein").n_blocks == 32
    assert webui.block_swap_plan(TrainConfig(**base), 24.0, "auto") == (0, 0.0)  # int8 9.6 GB on 24 GB
    issues = webui.review_issues(TrainConfig(**base), "style", data, 1)
    assert any("FLUX Non-Commercial License" in t and k == "info" for k, t in issues)
    wrong = TrainConfig(model_path=DISTILLED, datasets=[{"path": str(a)}])
    assert any("distilled" in t and k == "warning" for k, t in webui.review_issues(wrong, "style", data, 1))
    assert set(webui.file_quant_options("flux2_klein", "model")) == {"fp8_scaled", "bf16"}
