"""FLUX.2 Klein 4B as its own model entry (flux2_klein4b) next to the 9B (flux2_klein), without model weights: size
detection from a state dict, config inference at the real 4B size on the meta device, the two entries refusing each
other's files, the distilled-name rule, the Qwen3-4B text encoder config and layers, arch auto-detection from files and
folders, config / download / sampling entries, and the memory thresholds."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import dataset as D  # noqa: E402
from rulztrainer.comfy_files import planned_downloads  # noqa: E402
from rulztrainer.config import ARCH_FULL_SAMPLE, ARCH_NAMES, ARCH_TURBO, MODEL_PRESETS, TrainConfig  # noqa: E402
from rulztrainer.flux2 import loading as FL  # noqa: E402
from rulztrainer.flux2 import single_file as FSF  # noqa: E402
from rulztrainer.flux2.model import Flux2KleinTransformer  # noqa: E402
from rulztrainer.runplan import KLEIN_4B_CONFIG, target_shapes  # noqa: E402

BASE4 = "black-forest-labs/FLUX.2-klein-base-4B"
DISTILLED4 = "black-forest-labs/FLUX.2-klein-4B"
BASE9 = "black-forest-labs/FLUX.2-klein-base-9B"
TINY = dict(in_channels=16, hidden_size=32, num_heads=2, depth=2, depth_single_blocks=3, axes_dim=(4, 4, 4, 4),
            timestep_channels=32)


def tiny_state(text_dim: int, seed: int = 0) -> dict:
    """A tiny BFL-layout Klein state dict whose text projection reads `text_dim` features."""
    torch.manual_seed(seed)
    model = Flux2KleinTransformer(context_in_dim=text_dim, **TINY).float()
    return {k: torch.randn_like(v) * 0.1 for k, v in model.state_dict().items()}


def save(path: Path, state: dict, prefix: str = "diffusion_model.") -> Path:
    from safetensors.torch import save_file

    save_file({f"{prefix}{k}": v.contiguous() for k, v in state.items()}, path)
    return path


def real_4b_state() -> dict:
    with torch.device("meta"):
        return Flux2KleinTransformer(**KLEIN_4B_CONFIG).state_dict()


def test_real_size_4b_config_is_inferred_from_the_state_dict():
    state = real_4b_state()
    assert 3.8e9 < sum(v.numel() for v in state.values()) < 3.95e9  # the 7.75 GB bf16 file
    config = FSF.transformer_config_from_state(FSF.to_model_layout(dict(state), "4b"))
    assert config["hidden_size"] == 3072 and config["num_heads"] == 24 and config["axes_dim"] == [32, 32, 32, 32]
    assert config["depth"] == 5 and config["depth_single_blocks"] == 20 and config["context_in_dim"] == 7680
    assert config["mlp_ratio"] == 3.0 and config["in_channels"] == 128 and config["timestep_channels"] == 256
    assert FSF.klein_size(state) == "4b"
    with torch.device("meta"):
        assert FSF.klein_size(Flux2KleinTransformer().state_dict()) == "9b"  # the 9B defaults are untouched
        rebuilt = Flux2KleinTransformer.from_config(config)
    assert [tuple(v.shape) for v in rebuilt.state_dict().values()] == [tuple(v.shape) for v in state.values()]


def test_size_detection_and_refusals_name_the_right_entry():
    four, nine = tiny_state(7680), tiny_state(12288)
    assert FSF.klein_size(four) == "4b" and FSF.klein_size(nine) == "9b"
    assert FSF.to_model_layout(four, "4b") and FSF.to_model_layout(nine, "9b") and FSF.to_model_layout(four, None)
    assert FSF.to_model_layout(nine)  # the 9B stays the default
    with pytest.raises(ValueError, match="not FLUX.2 Klein 9B but FLUX.2 Klein 4B.*flux2_klein4b"):
        FSF.to_model_layout(four)  # what the flux2_klein entry does with a 4B file
    with pytest.raises(ValueError, match="not FLUX.2 Klein 4B but FLUX.2 Klein 9B.*flux2_klein"):
        FSF.to_model_layout(nine, "4b")
    for width in (15360, 4096):  # FLUX.2 dev reads Mistral; anything else is no Klein
        with pytest.raises(ValueError, match="not FLUX.2 Klein 4B"):
            FSF.to_model_layout(tiny_state(width), "4b")
        with pytest.raises(ValueError, match="not FLUX.2 Klein 9B"):
            FSF.to_model_layout(tiny_state(width), "9b")
    dev = {**four, "guidance_in.in_layer.weight": torch.zeros(32, 32)}  # a guidance embedding: FLUX.2 dev
    with pytest.raises(ValueError, match="not FLUX.2 Klein 4B"):
        FSF.to_model_layout(dev, "4b")
    with pytest.raises(ValueError, match="FLUX.2 dev is not supported"):
        FSF.klein_size(dev)


def test_the_entries_refuse_each_others_files(tmp_path):
    four = save(tmp_path / "flux-2-klein-base-4b.safetensors", tiny_state(7680))
    nine = save(tmp_path / "flux-2-klein-base-9b.safetensors", tiny_state(12288))
    m4 = FL.load_any_transformer(str(four), "cpu", "none", size="4b")
    assert m4.text_dim == 7680 and not m4.is_distilled
    assert FL.load_any_transformer(str(nine), "cpu", "none").text_dim == 12288
    with pytest.raises(ValueError, match="flux-2-klein-base-4b.safetensors.*FLUX.2 Klein 4B.*flux2_klein4b"):
        FL.load_any_transformer(str(four), "cpu", "none")
    with pytest.raises(ValueError, match="flux-2-klein-base-9b.safetensors.*FLUX.2 Klein 9B.*flux2_klein"):
        FL.load_any_transformer(str(nine), "cpu", "none", size="4b")
    # through the Arch entries (the sampling models and the trainer go through these)
    assert archs.arch_by_name("flux2_klein4b").load_any_transformer(str(four), "cpu", "none").text_dim == 7680
    assert archs.arch_by_name("flux2_klein").load_any_transformer(str(nine), "cpu", "none").text_dim == 12288
    with pytest.raises(ValueError, match="flux2_klein4b"):
        archs.arch_by_name("flux2_klein").load_any_transformer(str(four), "cpu", "none")
    with pytest.raises(ValueError, match="FLUX.2 Klein 9B.*flux2_klein"):
        archs.arch_by_name("flux2_klein4b").load_any_transformer(str(nine), "cpu", "none")


def test_diffusers_layout_of_4b_loads_through_the_converter(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    pytest.importorskip("diffusers.models.transformers.transformer_flux2")
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers

    state = tiny_state(7680, 3)
    converted = convert_flux2_transformer_checkpoint_to_diffusers({k: v.clone() for k, v in state.items()})
    back = FSF.to_model_layout({k: v.clone() for k, v in converted.items()}, "4b")
    assert set(back) == set(state) and all(torch.equal(back[k], state[k].float()) for k in state)
    with pytest.raises(ValueError, match="FLUX.2 Klein 4B"):
        FSF.to_model_layout({k: v.clone() for k, v in converted.items()})
    assert diffusers  # (importorskip only)


def test_distilled_and_base_names():
    for name in ("flux-2-klein-4b.safetensors", "C:\\m\\flux-2-klein-4b-fp8.safetensors", "FLUX.2-klein-4B",
                 "black-forest-labs__FLUX.2-klein-4B/", DISTILLED4):
        assert FSF.is_distilled_name(name), name
    for name in ("flux-2-klein-base-4b.safetensors", "FLUX.2-klein-base-4B", BASE4, "models/black-forest-labs__"
                 "FLUX.2-klein-base-4B/"):
        assert not FSF.is_distilled_name(name), name
    arch = archs.arch_by_name("flux2_klein4b")
    assert arch.distilled_sample_model(DISTILLED4) and not arch.distilled_sample_model(BASE4)
    assert TrainConfig(model_path=BASE4, sample_model_path=DISTILLED4).samples_on_distilled_model()
    assert not TrainConfig(model_path=BASE4).samples_on_distilled_model()
    assert FSF.is_distilled_name("flux-2-klein-9b.safetensors") and not FSF.is_distilled_name("flux-2-klein-base-9b")


def test_text_encoder_config_for_4b_and_its_layers():
    c = FSF.TEXT_CONFIG_4B
    assert (c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["num_attention_heads"],
            c["num_key_value_heads"], c["head_dim"]) == (2560, 9728, 36, 32, 8, 128)
    assert c["rms_norm_eps"] == 1e-6 and c["vocab_size"] == 151936 and c["max_position_embeddings"] == 40960
    assert c["rope_parameters"]["rope_theta"] == 1000000.0 and c["tie_word_embeddings"] is True
    assert FSF.TEXT_CONFIG["hidden_size"] == 4096 and FSF.TEXT_CONFIG["tie_word_embeddings"] is False  # 9B unchanged
    assert FSF.text_config("4b") is FSF.TEXT_CONFIG_4B and FSF.text_config("9b") is FSF.TEXT_CONFIG
    assert FSF.text_config(None) is FSF.TEXT_CONFIG
    assert FL.SELECT_LAYERS == (9, 18, 27) and FSF.TE_LAYERS == 27  # diffusers' Flux2KleinPipeline, both sizes
    assert len(FL.SELECT_LAYERS) * c["hidden_size"] == FSF.TEXT_DIM_4B == 7680
    assert len(FL.SELECT_LAYERS) * FSF.TEXT_CONFIG["hidden_size"] == FSF.TEXT_DIM_9B == 12288
    from transformers import Qwen3Config

    cfg = Qwen3Config(**c)  # accepted by the installed transformers
    assert cfg.hidden_size == 2560 and cfg.num_hidden_layers == 36


def test_text_model_builds_from_a_config_and_files_of_the_other_size_are_refused(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3Model

    tiny4 = {**FSF.TEXT_CONFIG_4B, "vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
             "num_hidden_layers": 30, "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
             "max_window_layers": 30}
    torch.manual_seed(5)
    te = Qwen3Model(Qwen3Config(**tiny4)).eval()
    state = {f"model.{k}": v.clone() for k, v in te.state_dict().items()}
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()  # tied: the file may carry it
    model = FSF.build_text_model(dict(state), "none", config=tiny4)
    assert len(model.layers) == 27 and model.config.hidden_size == 16
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        got = model.float()(input_ids=ids, output_hidden_states=True).hidden_states
        want = te(input_ids=ids, output_hidden_states=True).hidden_states
    for k in FL.SELECT_LAYERS:
        assert torch.allclose(got[k], want[k], atol=2e-2)
    # a file of the other width is refused by name (the sizes are shrunk to the tiny model's)
    monkeypatch.setattr(FSF, "TEXT_CONFIG_4B", tiny4)
    monkeypatch.setattr(FSF, "TEXT_CONFIG", {**FSF.TEXT_CONFIG, "hidden_size": 24})
    path = tmp_path / "qwen_3_4b.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, path)
    wrong = {k: v for k, v in state.items()}
    wrong["model.embed_tokens.weight"] = torch.zeros(100, 24)
    wrong_path = tmp_path / "qwen_3_8b.safetensors"
    save_file({k: v.contiguous() for k, v in wrong.items()}, wrong_path)
    with pytest.raises(ValueError, match="not the text encoder of FLUX.2 Klein 4B.*FLUX.2 Klein 9B"):
        FSF.load_text_encoder_file(str(wrong_path), "cpu", "none", size="4b")
    with pytest.raises(ValueError, match="not the text encoder of FLUX.2 Klein 9B"):
        FSF.load_text_encoder_file(str(path), "cpu", "none", size="9b")


def test_arch_is_detected_from_files_and_folders(tmp_path):
    four = save(tmp_path / "flux-2-klein-base-4b.safetensors", tiny_state(7680))
    nine = save(tmp_path / "flux-2-klein-base-9b.safetensors", tiny_state(12288))
    bare = save(tmp_path / "model.safetensors", tiny_state(7680), prefix="")  # no hint in the name or the key prefix
    assert archs.arch_from_transformer_file(str(four)) == "flux2_klein4b"
    assert archs.arch_from_transformer_file(str(nine)) == "flux2_klein"
    assert archs.arch_from_transformer_file(str(bare)) == "flux2_klein4b"
    # the file's own family wins over a name that suggests the other size
    cfg = TrainConfig(model_source="files", transformer_file=str(bare), model_path=BASE9)
    assert archs.guess_arch_name(cfg) == "flux2_klein4b" and archs.get_arch(cfg).name == "flux2_klein4b"
    assert archs.guess_arch_name(TrainConfig(model_source="files", transformer_file=str(nine))) == "flux2_klein"
    with pytest.raises(ValueError, match="flux2_klein4b"):  # model_arch set to the other size
        archs.detect_arch(TrainConfig(model_source="files", transformer_file=str(four), model_arch="flux2_klein"), None)
    # diffusers folders: the transformer config's joint_attention_dim
    for dim, want in ((7680, "flux2_klein4b"), (12288, "flux2_klein")):
        folder = tmp_path / f"repo{dim}"
        (folder / "transformer").mkdir(parents=True)
        (folder / "transformer" / "config.json").write_text(json.dumps(
            {"_class_name": "Flux2Transformer2DModel", "joint_attention_dim": dim}), encoding="utf-8")
        assert archs.detect_arch(TrainConfig(model_path=str(folder), model_arch="auto"), folder).name == want


def test_family_hints_and_entries():
    for path in (BASE4, DISTILLED4, "C:/m/flux-2-klein-base-4b.safetensors", "Comfy-Org/Flux-2-Klein-4B", "flux2-klein-4B"):
        assert TrainConfig(model_path=path).arch_name() == "flux2_klein4b", path
    for path in (BASE9, "black-forest-labs/FLUX.2-klein-9B", "C:/m/flux-2-klein-base-9b-fp8.safetensors",
                 "C:/models4b/flux-2-klein-base-9b.safetensors", "C:/models/4b/flux-2-klein-base-9b.safetensors",
                 "D:\\4b\\flux-2-klein-9b.safetensors"):
        assert TrainConfig(model_path=path).arch_name() == "flux2_klein", path
    assert TrainConfig(model_path="black-forest-labs/flux-3-image-klein-base").arch_name() == "flux3_klein"
    assert "flux2_klein4b" in ARCH_NAMES and "flux2_klein4b" in archs.ARCH_NAMES
    assert MODEL_PRESETS["FLUX-2-Klein-4B-Base"] == BASE4 and MODEL_PRESETS["FLUX-2-Klein-4B"] == DISTILLED4
    assert MODEL_PRESETS["FLUX-2-Klein-9B-Base"] == BASE9  # the 9B presets are unchanged
    four, nine = archs.arch_by_name("flux2_klein4b"), archs.arch_by_name("flux2_klein")
    assert (four.label, four.modelspec, four.n_blocks) == ("FLUX.2 Klein 4B", "Flux.2-klein-4b", 25)
    assert (nine.label, nine.modelspec, nine.n_blocks) == ("FLUX.2 Klein 9B", "Flux.2-klein-9b", 32)
    assert four.lora_format == nine.lora_format == "folded" and four.latent_tag == nine.latent_tag == ""
    assert TrainConfig(model_path=BASE4).resolved_lora_targets() == list(four.default_targets)
    assert (four.default_sample_steps, four.default_sample_cfg) == ARCH_FULL_SAMPLE["flux2_klein4b"] == (50, 4.0)
    assert ARCH_FULL_SAMPLE["flux2_klein"] == (20, 3.5)
    assert "flux2_klein4b" not in ARCH_TURBO  # no distill LoRA exists for 4B
    from rulztrainer import capabilities

    assert capabilities.display_name("flux2_klein4b") == "FLUX.2 Klein 4B"
    assert set(capabilities.rules_for("flux2_klein4b")) == set(capabilities.rules_for("flux2_klein"))


def test_4b_arch_reaches_every_loading_entry_point_with_its_size(monkeypatch):
    """The base Arch calls get_tokenizer / prompt_tokens / encode_prompts on its loading module by name: the 4B's
    sized facade must have them all, with the size bound where the function takes one."""
    arch, seen = archs.arch_by_name("flux2_klein4b"), {}
    monkeypatch.setattr(FL, "get_tokenizer", lambda cfg, model_dir, size="9b": seen.setdefault("tok", size))
    monkeypatch.setattr(FL, "get_transformer", lambda cfg, md, device=None, size="9b": seen.setdefault("dit", size))
    monkeypatch.setattr(FL, "get_text_encoder", lambda cfg, md, size="9b": seen.setdefault("te", size))
    monkeypatch.setattr(FL, "load_any_transformer",
                        lambda p, d, q="auto", m="models", size="9b": seen.setdefault("any", size))
    cfg = TrainConfig(model_path=BASE4)
    arch.get_tokenizer(cfg, None), arch.get_transformer(cfg, None), arch.get_text_encoder(cfg, None)
    arch.load_any_transformer("x.safetensors", "cpu")
    assert seen == {"tok": "4b", "dit": "4b", "te": "4b", "any": "4b"}
    for name in ("prompt_tokens", "encode_prompts", "encode_prompt", "pad_embeds", "resolve_for_config", "get_vae"):
        assert getattr(arch._loading, name) is getattr(FL, name), name
    assert archs.arch_by_name("flux2_klein")._loading is FL  # the 9B keeps the plain module


def test_downloads_caches_and_latents_of_4b(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    from PIL import Image

    Image.new("RGB", (900, 600)).save(data / "a.png")
    k4 = TrainConfig(model_path=BASE4, dataset_dir=str(data), resolutions=[1.0])
    k9 = TrainConfig(model_path=BASE9, dataset_dir=str(data), resolutions=[1.0])
    assert D.scan_dataset(k4)[0].latent_path == D.scan_dataset(k9)[0].latent_path  # the shared Flux2 VAE latents
    assert D.text_cache_path(k4, "p").name.endswith("_flux2_klein4b_wo.safetensors")
    assert D.text_cache_path(k4, "p") != D.text_cache_path(k9, "p")  # 7680 and 12288 features never share a cache
    comfy = "Comfy-Org/Flux-2-Klein-4B"
    plan = {f: (r, n) for f, r, n in planned_downloads(TrainConfig(model_path=BASE4))}
    assert plan == {  # no turbo LoRA, no gating: Comfy-Org's ungated files only
        "transformer_file": (comfy, "split_files/diffusion_models/flux-2-klein-base-4b.safetensors"),
        "text_encoder_file": (comfy, "split_files/text_encoders/qwen_3_4b.safetensors"),
        "vae_file": (comfy, "split_files/vae/flux2-vae.safetensors")}
    plan = {f: (r, n) for f, r, n in planned_downloads(TrainConfig(model_path=DISTILLED4, model_file_quant="bf16"))}
    assert plan["transformer_file"] == (comfy, "split_files/diffusion_models/flux-2-klein-4b.safetensors")
    with pytest.raises(ValueError, match="model_file_quant"):
        planned_downloads(TrainConfig(model_path=BASE4, model_file_quant="nvfp4"))
    with pytest.raises(ValueError, match="text_encoder_file_quant"):
        planned_downloads(TrainConfig(model_path=BASE4, text_encoder_file_quant="fp8_scaled"))
    nine = {f: r for f, r, _ in planned_downloads(TrainConfig(model_path=BASE9))}
    assert nine["transformer_file"] == "black-forest-labs/FLUX.2-klein-base-9b-fp8"  # 9B downloads unchanged


def test_gated_help_says_4b_is_not_gated():
    from rulztrainer.flux3.loading import gated_error

    text = str(gated_error(BASE4, RuntimeError("401")))
    assert "Apache 2.0" in text and "not gated" in text and "Non-Commercial" not in text
    assert "FLUX Non-Commercial License" in str(gated_error("black-forest-labs/FLUX.2-klein-base-9b-fp8",
                                                            RuntimeError("401")))


def test_auto_quant_thresholds_are_sized_for_each_model(monkeypatch):
    gb = {0: 8.0, 1: 12.0, 2: 24.0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(FL, "_gpu_gb", lambda d: gb[torch.device(d).index])
    assert FL.resolve_quant("auto", "cuda:0", "4b") == "nvfp4"
    assert FL.resolve_quant("auto", "cuda:1", "4b") == "int8_convrot"
    assert FL.resolve_quant("auto", "cuda:2", "4b") == "none"  # 7.75 GB of bf16 weights fit a 24 GB card
    assert FL.resolve_quant("auto", "cuda:2") == "int8_convrot"  # the 9B on the same card, as before
    assert FL.resolve_quant("auto", ["cuda:0", "cuda:1"], "4b") == "none"  # a split counts both cards (8 + 12 GB)
    assert FL.resolve_text_quant("auto", "4b") == "none"  # the 24 GB card holds the 8 GB Qwen3-4B
    assert FL.resolve_text_quant("auto") == "none"
    gb[2] = 12.0
    assert FL.resolve_text_quant("auto", "4b") == "none" and FL.resolve_text_quant("auto") == "int8_convrot"
    gb[1] = gb[2] = 8.0
    assert FL.resolve_text_quant("auto", "4b") == "int8_convrot"


def test_lora_targets_blocks_and_estimates_follow_the_4b_size():
    four = archs.arch_by_name("flux2_klein4b")
    with torch.device("meta"):
        model = Flux2KleinTransformer(**KLEIN_4B_CONFIG)
    assert len(model.layers) == four.n_blocks == 25
    names = four.lora_module_names(model, ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"], None)
    assert "double_blocks.4.img_attn.qkv" in names and "single_blocks.19.linear2" in names
    assert not any(n.startswith("double_blocks.5.") or n.startswith("single_blocks.20.") for n in names)
    assert four.lora_module_names(model, ["q"], [5]) == ["single_blocks.0.linear1"]  # block 5 = the first single block
    assert four.lora_module_name("double_blocks_3_img_attn_qkv") == "double_blocks.3.img_attn.qkv"
    shapes = target_shapes("flux2_klein4b", ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"))
    assert shapes and (3072 * 3, 3072) in shapes and all(3072 in s for s in shapes)
    assert target_shapes("flux2_klein", ("q",))[0][1] == 4096  # the 9B estimate is unchanged
    # the sampler needs nothing of the size: the same schedules
    assert four.sample_sigmas.__func__ is archs.arch_by_name("flux2_klein").sample_sigmas.__func__


def test_webui_tables_know_klein_4b():
    webui = pytest.importorskip("webui")
    assert webui.N_BLOCKS["flux2_klein4b"] == archs.arch_by_name("flux2_klein4b").n_blocks == 25
    assert webui.N_BLOCKS["flux2_klein"] == 32
    assert set(webui.file_quant_options("flux2_klein4b", "model")) == {"bf16"}
    assert set(webui.file_quant_options("flux2_klein", "model")) == {"fp8_scaled", "bf16"}
    assert webui.plan_quant_of("flux2_klein4b", "hf", "int8_convrot", 16.0, "auto") == "int8"
    assert webui.plan_quant_of("flux2_klein4b", "hf", "int8_convrot", 24.0, "auto") == "bf16"
    assert webui.plan_quant_of("flux2_klein4b", "hf", "int8_convrot", 8.0, "auto") == "nvfp4"
    assert webui.plan_quant_of("flux2_klein", "hf", "int8_convrot", 24.0, "auto") == "int8"  # the 9B's thresholds
    assert webui.plan_quant_of("flux2_klein4b", "files", "bf16", 16.0, "auto") == "int8"  # a bf16 file is quantized too
    assert webui.MODEL_GB["flux2_klein4b"]["bf16"] == 7.75
    assert "flux2_klein4b" in webui.AUTO_QUANT_BY_CARD
    assert webui.preset_arch("FLUX-2-Klein-4B-Base") == "flux2_klein4b" and webui.preset_arch(
        "FLUX-2-Klein-9B-Base") == "flux2_klein"
    assert "FLUX-2-Klein-4B-Base" in webui.PRESET_NOTES if hasattr(webui, "PRESET_NOTES") else True


def test_wizard_and_review_for_4b(tmp_path):
    from PIL import Image

    webui = pytest.importorskip("webui")
    a = tmp_path / "a"
    a.mkdir()
    for i in range(20):
        Image.new("RGB", (64, 48)).save(a / f"{i}.png")
    data = webui.datasets_summary([str(a)])
    base = dict(model_path=BASE4, datasets=[{"path": str(a)}])
    rec = webui.wizard_recommendations(TrainConfig(**base), "character", [{"total": 24000}], data)
    assert (rec["sample_steps"][0], rec["sample_cfg"][0]) == (50, 4.0)
    assert rec["lora_rank"][0] == 16 and rec["timestep_sampling"][0] == "reference"
    assert "4B" in rec["transformer_quant"][1] and "Qwen3-4B" in rec["text_encoder_quant"][1]
    distilled = TrainConfig(**base, sample_model_path=DISTILLED4)
    rec = webui.wizard_recommendations(distilled, "character", [{"total": 24000}], data)
    assert (rec["sample_steps"][0], rec["sample_cfg"][0]) == (4, 1.0)
    issues = webui.review_issues(TrainConfig(**base), "style", data, 1)
    assert any("Apache 2.0" in t and "not gated" in t and k == "info" for k, t in issues)
    assert not any("Non-Commercial" in t for k, t in issues)
    wrong = TrainConfig(model_path=DISTILLED4, datasets=[{"path": str(a)}])
    assert any("distilled" in t and "FLUX-2-Klein-4B-Base" in t and k == "warning"
               for k, t in webui.review_issues(wrong, "style", data, 1))
