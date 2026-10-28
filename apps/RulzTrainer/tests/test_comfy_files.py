"""Single files by default: what a config downloads from Comfy-Org (per model and quant) and the turbo LoRA per family;
FIBO 2.0's ComfyUI layout (fused qkv / to_kv / w13, the zero-padded feed-forward) and its Turbo LoRA in Bria's layout;
norm weight offsets of a sampling LoRA swapped in only while sampling; the WebUI's sample aspect ratio and megapixels."""
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import comfy_files as CF  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402
from rulztrainer.config import ARCH_SAMPLE_DEFAULTS, ARCH_TURBO, TrainConfig  # noqa: E402
from rulztrainer.fibo import single_file as FS  # noqa: E402


def test_defaults_are_single_files_turbo_and_sam3():
    c = TrainConfig()
    assert c.model_source == "files" and c.model_file_quant == c.text_encoder_file_quant == "int8_convrot"
    assert c.turbo_lora == "auto" and c.body_backend == "mediapipe+sam3"
    assert (c.sample_steps, c.sample_cfg, c.sample_shift) == (12, 1.0, 5.0) == (*ARCH_SAMPLE_DEFAULTS["moss_image"], 5.0)
    assert ARCH_SAMPLE_DEFAULTS == {"moss_image": (12, 1.0), "krea2": (8, 1.0), "fibo2": (4, 1.0),
                                    "qwen_image21": (8, 1.0)}


def test_planned_downloads_per_model_and_quant():
    plan = dict((f, (r, p)) for f, r, p in CF.planned_downloads(TrainConfig(model_path="krea/Krea-2-Raw",
                                                                             model_file_quant="fp8_scaled",
                                                                             text_encoder_file_quant="bf16")))
    assert plan["transformer_file"] == ("Comfy-Org/Krea-2", "diffusion_models/krea2_raw_fp8_scaled.safetensors")
    assert plan["text_encoder_file"] == ("Comfy-Org/Krea-2", "text_encoders/qwen3vl_4b_bf16.safetensors")
    assert plan["vae_file"] == ("Comfy-Org/Krea-2", "vae/qwen_image_vae.safetensors")
    assert plan["turbo_lora"] == ("Comfy-Org/Krea-2", ARCH_TURBO["krea2"]["file"])
    sft = dict((f, p) for f, _, p in CF.planned_downloads(TrainConfig(model_path="OpenMOSS-Team/MOSS-Image-SFT")))
    assert sft["transformer_file"] == "diffusion_models/moss_image_sft_int8_convrot.safetensors"
    assert sft["vae_file"] == "vae/flux2-vae_small.safetensors"
    # files that are set, a turbo LoRA that is off or a file, and the Hugging Face source download nothing
    own = TrainConfig(model_path="briaai/Fibo-2.0-dev", transformer_file="a", text_encoder_file="b", vae_file="c",
                      turbo_lora="")
    assert CF.planned_downloads(own) == []
    assert [f for f, _, _ in CF.planned_downloads(TrainConfig(model_source="hf"))] == ["turbo_lora"]
    with pytest.raises(ValueError, match="fp8_scaled"):
        CF.planned_downloads(TrainConfig(model_path="briaai/Fibo-2.0-dev", model_file_quant="fp8_scaled"))
    with pytest.raises(ValueError, match="No single-file download"):
        CF.planned_downloads(TrainConfig(model_path="someone/other-moss-finetune"))


def test_resolve_downloads_sets_the_fields(monkeypatch, tmp_path):
    got = []

    def fake(repo, file, label, progress=None):
        got.append((repo, file))
        return tmp_path / file.rsplit("/", 1)[-1]

    import rulztrainer.downloads as D

    monkeypatch.setattr(D, "ensure_hf_file", fake)
    c = CF.resolve_downloads(TrainConfig(model_path="briaai/Fibo-2.0-dev", vae_file="mine.safetensors"))
    assert c.transformer_file.endswith("fibo2_dev_int8_convrot.safetensors") and c.vae_file == "mine.safetensors"
    assert c.turbo_lora.endswith("fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors")
    assert ("Comfy-Org/Fibo-2.0", "text_encoders/qwen3vl_4b_int8_convrot.safetensors") in got and len(got) == 3


def _int8(w):
    return Q.quantize_weight(w, "int8")


def test_fibo_comfy_layout_splits_fused_weights_and_cuts_the_padding():
    torch.manual_seed(0)
    dim, hidden, pad = 8, 6, 8  # Bria's width 6, ComfyUI's padded 8
    q, k, v = (torch.randn(dim, dim) for _ in range(3))
    w1, w3, w2 = torch.randn(hidden, dim), torch.randn(hidden, dim), torch.randn(dim, hidden)
    zr = torch.zeros(pad - hidden, dim)
    state = {"layers.0.attention.qkv.weight": torch.cat([q, k, v]),
             "layers.0.attention.out.weight": torch.randn(dim, dim),
             "layers.0.attention.q_norm.weight": torch.ones(4),
             "layers.6.cross_attention.to_kv.weight": torch.cat([k, v]),
             "layers.6.cross_attention.to_out.weight": torch.randn(dim, dim),
             "perceiver.layers.0.self_attention.to_out.weight": torch.randn(dim, dim),
             "layers.0.feed_forward.w13.weight": torch.cat([w1, zr, w3, zr]),
             "layers.0.feed_forward.w2.weight": torch.cat([w2, torch.zeros(dim, pad - hidden)], 1),
             "layers.0.attention.comfy_attention.config": torch.zeros(1)}
    out = FS.to_diffusers_layout(state, hidden)
    assert torch.equal(out["layers.0.attention.to_k.weight"], k) and torch.equal(out["layers.0.attention.to_v.weight"], v)
    assert torch.equal(out["layers.6.cross_attention.to_v.weight"], v)
    assert torch.equal(out["layers.0.feed_forward.w1.weight"], w1) and torch.equal(out["layers.0.feed_forward.w3.weight"], w3)
    assert torch.equal(out["layers.0.feed_forward.w2.weight"], w2)
    for name in ("layers.0.attention.to_out.0.weight", "layers.0.attention.norm_q.weight",
                 "layers.6.cross_attention.to_out.0.weight", "perceiver.layers.0.self_attention.to_out.0.weight"):
        assert name in out
    assert not any("comfy_attention" in n or "qkv" in n or "w13" in n for n in out)
    # int8 weights: rows sliced with their scales, the cut w2 quantized again
    qstate = {k2: (_int8(t) if t.dim() == 2 and "norm" not in k2 else t) for k2, t in state.items()}
    qout = FS.to_diffusers_layout(qstate, hidden)
    w1q = qout["layers.0.feed_forward.w1.weight"]
    assert isinstance(w1q, Q.QuantWeight) and w1q.orig_shape == (hidden, dim)
    assert torch.allclose(w1q.dequantize(torch.float32), w1, atol=0.05)
    w2q = qout["layers.0.feed_forward.w2.weight"]
    assert w2q.orig_shape == (dim, hidden) and torch.allclose(w2q.dequantize(torch.float32), w2, atol=0.05)


def test_fibo_turbo_lora_in_bria_layout():
    dim, hidden, half, r = 8, 6, 8, 2
    down, up_qkv = torch.randn(r, dim), torch.randn(3 * dim, r)
    up13 = torch.randn(2 * half, r)
    state = {"diffusion_model.layers.0.attention.qkv.lora_down.weight": down,
             "diffusion_model.layers.0.attention.qkv.lora_up.weight": up_qkv,
             "diffusion_model.layers.0.attention.qkv.alpha": torch.tensor(2.0),
             "diffusion_model.layers.0.attention.qkv.diff_b": torch.arange(3 * dim, dtype=torch.float32),
             "diffusion_model.layers.0.feed_forward.w13.lora_down.weight": down,
             "diffusion_model.layers.0.feed_forward.w13.lora_up.weight": up13,
             "diffusion_model.layers.0.feed_forward.w2.lora_down.weight": torch.randn(r, half),
             "diffusion_model.layers.0.attention.out.lora_up.weight": torch.randn(dim, r),
             "diffusion_model.layers.0.attention.k_norm.diff": torch.randn(4)}
    out = FS.lora_to_diffusers_layout(state, hidden)
    p = "diffusion_model.layers.0."
    assert torch.equal(out[p + "attention.to_v.lora_up.weight"], up_qkv[2 * dim:])
    assert torch.equal(out[p + "attention.to_v.lora_down.weight"], down) and float(out[p + "attention.to_k.alpha"]) == 2
    assert torch.equal(out[p + "attention.to_k.diff_b"], torch.arange(dim, 2 * dim, dtype=torch.float32))
    assert torch.equal(out[p + "feed_forward.w3.lora_up.weight"], up13[half:half + hidden])
    assert out[p + "feed_forward.w1.lora_up.weight"].shape == (hidden, r)
    assert out[p + "feed_forward.w2.lora_down.weight"].shape == (r, hidden)
    assert p + "attention.to_out.0.lora_up.weight" in out and p + "attention.norm_k.diff" in out


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def test_norm_offsets_only_while_sampling(tmp_path):
    torch.manual_seed(0)
    m = _Tiny()
    base = m.norm.weight.detach().clone()
    delta = torch.full((4,), 0.5)
    save_file({"diffusion_model.norm.diff": delta, "diffusion_model.lin.lora_down.weight": torch.randn(1, 4),
               "diffusion_model.lin.lora_up.weight": torch.randn(4, 1)}, str(tmp_path / "t.safetensors"))
    assert L.attach_extra_lora(m, str(tmp_path / "t.safetensors"), 2.0) == 2
    assert torch.equal(m.norm.weight, base)  # attached, not applied
    L.set_extras_enabled(m, True)
    assert torch.allclose(m.norm.weight, base + 1.0)  # strength 2 x 0.5
    L.set_extras_enabled(m, False)
    assert torch.equal(m.norm.weight, base)
    for _ in range(3):  # toggling never drifts
        L.set_extras_enabled(m, True)
        L.set_extras_enabled(m, False)
    assert torch.equal(m.norm.weight, base)


def test_sample_aspect_and_megapixels():
    import webui

    assert webui.aspect_size("1:1", 1.0) == (1024, 1024)
    assert webui.aspect_size("2:3", 1.0) == (832, 1248)
    w, h = webui.aspect_size("16:9", 2.0)
    assert w % 16 == 0 and h % 16 == 0 and abs(w / h - 16 / 9) < 0.02 and abs(w * h / 2 ** 21 - 1) < 0.02
    assert webui.size_aspect(832, 1248) == ("2:3", 1.0) and webui.size_aspect(1024, 1024) == ("1:1", 1.0)
    assert webui.size_aspect(1920, 1080)[0] == "16:9"
    assert set(webui.file_quant_options("krea2", "model")) == {"int8_convrot", "fp8_scaled", "bf16"}
    assert set(webui.file_quant_options("fibo2", "text")) == {"int8_convrot", "bf16"}


def test_moss_tokenizer_is_moss_own(tmp_path):
    """MOSS-Image's single-file mode took the first */tokenizer folder under models/, Krea 2's when there was one: the
    wrong token ids for its Qwen3.5 text encoder, so prompts came out as noise. Only MOSS-Image's own folders count."""
    from rulztrainer.moss import loading as ML
    from rulztrainer.moss import single_file as MS

    for name in ("krea__Krea-2-Raw", "OpenMOSS-Team__MOSS-Image-Base"):
        d = tmp_path / name / "tokenizer"
        d.mkdir(parents=True)
        (d / "tokenizer_config.json").write_text("{}")
    assert MS.resolve_tokenizer("", str(tmp_path)) == tmp_path / "OpenMOSS-Team__MOSS-Image-Base" / "tokenizer"

    class _Tok:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    ML.check_tokenizer(_Tok(248_077))
    with pytest.raises(ValueError, match="not MOSS-Image's Qwen3.5 tokenizer"):
        ML.check_tokenizer(_Tok(151_669))  # Qwen3-VL's (Krea 2, FIBO 2.0)
    from rulztrainer.dataset import text_cache_path

    assert text_cache_path(TrainConfig(), "x").name.endswith("_q35.safetensors")  # the old caches are not reused
