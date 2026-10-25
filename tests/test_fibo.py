"""FIBO 2.0 support without model weights: the tiny transformer, LoRA naming/export (PEFT format), prompts, sampler
schedule, the architecture registry and architecture-aware captions / caches."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import dataset as D  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.fibo import prompts as P  # noqa: E402
from rulztrainer.fibo.model import FiboTransformer  # noqa: E402
from rulztrainer.fibo.sampler import calculate_shift, fibo_sigmas  # noqa: E402

TINY = dict(dim=64, n_layers=4, n_refiner_layers=1, n_heads=2, cap_feat_dim=48, injection_layer_ids=(1, 3),
            perceiver_num_layers=1, min_num_gist_tokens=4, max_num_gist_tokens=8, gist_step=2, gist_min_text_len=4,
            gist_max_text_len=16, axes_dims=(8, 12, 12), axes_lens=(64, 32, 32))
FIBO = archs.arch_by_name("fibo2")


def tiny():
    torch.manual_seed(0)
    return FiboTransformer(**TINY).to(torch.bfloat16)


def test_tiny_forward_backward_padding_and_skip():
    m = tiny()
    m.gradient_checkpointing = True
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.randn(2, 128, 4, 6).to(torch.bfloat16)
    enc = torch.randn(2, 3, 9, 48).to(torch.bfloat16)  # 1 + 2 injection blocks = 3 bundles
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 5:] = False  # the padded prompt gets fewer gist tokens: the gist-mask path
    out = m(x, torch.tensor([600.0, 20.0]), enc, mask)
    assert out.shape == x.shape and out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in m.parameters())
    m.eval()
    with torch.no_grad():
        full = m(x[:1], torch.tensor([600.0]), enc[:1], None)
        skipped = m(x[:1], torch.tensor([600.0]), enc[:1], None, skip_layers=[2])
        again = m(x[:1], torch.tensor([600.0]), enc[:1], torch.ones(1, 9, dtype=torch.bool))  # all-true mask = none
    assert not torch.equal(full, skipped) and torch.equal(full, again)
    assert m.latent_channels == 128 and FIBO.compile_blocks(m)[0] is m.noise_refiner[0]


def test_token_layout_equals_diffusers_patchify():
    """The trainer's 128-channel cache layout flattens straight into the model's (channel, dy, dx) patch features."""
    x128 = torch.randn(1, 128, 3, 5)
    b, c, h, w = x128.shape
    x32 = x128.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)  # unpatchify
    bb, cc, hh, ww = x32.shape
    tokens = x32.view(bb, cc, hh // 2, 2, ww // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(bb, (hh // 2) * (ww // 2), -1)
    assert torch.equal(tokens, x128.flatten(2).transpose(1, 2))


def test_lora_names_peft_export_and_reload(tmp_path):
    m = tiny()
    names = FIBO.lora_module_names(m, ["q", "k", "v", "o", "ffn_up_gate", "ffn_down", "cross_attn", "refiner",
                                       "perceiver", "embed"], None)
    assert len(names) == 4 * 7 + 2 * 4 + 8 + 11 + 5  # block: 4 attention + 3 ffn; refiner adds its adaLN; perceiver 11
    assert "layers.1.cross_attention.to_out.0" in names and "layers.0.cross_attention.to_q" not in names
    assert FIBO.lora_module_names(m, ["q"], [0, 2]) == ["layers.0.attention.to_q", "layers.2.attention.to_q"]
    inj = L.inject_lora(m, 4, 8.0, ["q"], names=names)
    assert set(inj) == set(names)
    with torch.no_grad():
        for mod in inj.values():
            mod.lora_up.weight.normal_()
    path = tmp_path / "fibo_lora.safetensors"
    L.save_lora(inj, path, metadata={"modelspec.architecture": "bria-fibo2"}, dtype=torch.float32, fmt="peft")
    from safetensors import safe_open
    from safetensors.torch import load_file

    state = load_file(str(path))
    assert all(k.startswith("transformer.") and (k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"))
               for k in state)
    assert state["transformer.layers.0.attention.to_q.lora_A.weight"].shape == (4, 64)
    assert state["transformer.layers.0.attention.to_q.lora_B.weight"].shape == (64, 4)
    with safe_open(str(path), "pt") as f:
        meta = json.loads(f.metadata()["lora_adapter_metadata"])
    assert meta["transformer.r"] == 4 and meta["transformer.lora_alpha"] == 8.0
    assert meta["transformer.peft_type"] == "LORA" and set(meta["transformer.target_modules"]) == set(names)
    # the PEFT convention (alpha / r * B @ A) is what the wrapper applies
    mod = inj["layers.0.attention.to_q"]
    xin = torch.randn(1, 3, 64).to(torch.bfloat16)
    delta = (mod(xin) - mod.base(xin)).float()
    manual = (xin.float() @ mod.lora_down.weight.T @ mod.lora_up.weight.T) * (8.0 / 4)
    assert torch.allclose(delta, manual, atol=0.05, rtol=0.05)
    # reload into a fresh model (either format is detected)
    m2 = tiny()
    inj2 = L.inject_lora(m2, 4, 1.0, ["q"], names=names)
    assert L.load_lora_weights(inj2, path) == len(names)
    assert inj2["layers.0.attention.to_q"].alpha == 8.0
    assert torch.equal(inj2["layers.0.attention.to_q"].lora_up.weight, mod.lora_up.weight)
    # a sampling-only ("extra") LoRA in PEFT format takes its alpha from the metadata
    m3 = tiny()
    assert L.attach_extra_lora(m3, str(path), strength=0.5) == len(names)
    extra = m3.get_submodule("layers.0.attention.to_q").extras[0]
    assert extra[0] == "lora" and abs(extra[3] - (8.0 / 4) * 0.5) < 1e-6  # (kind, down, up, scale)


def test_prompts_cleaning_trigger_and_null_caption():
    raw = ('{"short_description": "A woman", "objects": [{"description": "a woman", "location": "", "texture": null}], '
           '"lighting": {}, "aesthetic_score": 7.5, "preference_score": 0.3, "style_medium": "photo", "context": ""}')
    assert P.clean_prompt(raw) == '{"short_description":"A woman","objects":[{"description":"a woman"}],"style_medium":"photo"}'
    assert P.clean_prompt("a plain sentence") == "a plain sentence"
    assert P.clean_prompt(P.NULL_CAPTION) == P.NULL_CAPTION  # the null caption is encoded as it is, like the pipeline
    assert P.clean_prompt('{"short_description": "x", "text_render": []}') == '{"short_description":"x","text_render":[]}'
    assert P.add_trigger("a plain sentence", "zxc") == "zxc, a plain sentence"
    assert P.add_trigger("", "zxc") == "zxc"
    t = json.loads(P.add_trigger('{"short_description":"A woman","style_medium":"photo"}', "zxc"))
    assert t["short_description"] == "zxc, A woman" and t["style_medium"] == "photo"
    assert json.loads(P.add_trigger('{"style_medium":"photo"}', "zxc"))["short_description"] == "zxc"
    assert FIBO.empty_prompt == P.NULL_CAPTION and FIBO.default_negative == P.NULL_CAPTION


def test_sampler_schedule_and_reference_sigmas():
    assert abs(calculate_shift(256) - 0.5) < 1e-9 and abs(calculate_shift(4096) - 1.15) < 1e-9
    s = fibo_sigmas(8, 4096)
    assert len(s) == 9 and s[0] == 1.0 and s[-1] == 0.0 and torch.all(s[:-1] > s[1:])
    assert fibo_sigmas(8, 256)[4] < fibo_sigmas(8, 4096)[4]  # a larger image shifts the schedule toward noise
    gen = torch.Generator().manual_seed(0)
    r = FIBO.reference_sigmas(20000, gen, 4096)
    assert r.min() >= 0 and r.max() <= 1
    frac = (r < 0.5).float().mean().item()
    assert 0.12 < frac < 0.21, frac  # 10% uniform + 90% sigmoid(N(1.15, 1)): about 16% below sigma 0.5
    moss = archs.arch_by_name("moss_image").reference_sigmas(20000, gen, 4096)
    assert 0.08 < (moss < 0.5).float().mean().item() < 0.18


def test_arch_detection_captions_and_cache_names(tmp_path):
    assert TrainConfig().arch_name() == "moss_image"
    assert TrainConfig(model_path="briaai/Fibo-2.0-dev").arch_name() == "fibo2"
    assert TrainConfig(model_arch="moss_image", model_path="briaai/Fibo-2.0-dev").arch_name() == "moss_image"
    mdir = tmp_path / "model" / "transformer"
    mdir.mkdir(parents=True)
    mdir.with_name("transformer").joinpath("config.json").write_text(json.dumps({"_class_name": "BriaFibo2Transformer2DModel"}))
    cfg = TrainConfig(model_path=str(tmp_path / "model"))  # no "fibo" in the path: detected from the config
    assert archs.get_arch(cfg).name == "moss_image" and archs.detect_arch(cfg, tmp_path / "model").name == "fibo2"
    # captions: FIBO prefers a structured .json (cleaned, trigger in short_description); MOSS reads the .txt as before
    img = tmp_path / "a.png"
    img.write_bytes(b"")
    img.with_suffix(".txt").write_text("a woman on a beach", encoding="utf-8")
    img.with_suffix(".json").write_text('{"short_description": "A woman", "lighting": {}, "aesthetic_score": 5}', encoding="utf-8")
    fibo_cfg = TrainConfig(model_path="briaai/Fibo-2.0-dev", trigger_word="zxc", cache_dir=str(tmp_path / "c"))
    assert D.read_caption(img, fibo_cfg) == '{"short_description":"zxc, A woman"}'
    moss_cfg = TrainConfig(trigger_word="zxc", cache_dir=str(tmp_path / "c"))
    assert D.read_caption(img, moss_cfg) == "zxc, a woman on a beach"
    # text caches of the two text encoders never collide; MOSS keeps its original file names
    assert D.text_cache_path(fibo_cfg, "p").name.endswith("_fibo2.safetensors")
    assert D.text_cache_path(moss_cfg, "p").name == f"{D._hash('p')}_{moss_cfg.max_token_length}.safetensors"
    assert D.empty_text_path(fibo_cfg) == D.text_cache_path(fibo_cfg, P.NULL_CAPTION)
    assert D.empty_text_path(moss_cfg) == D.text_cache_path(moss_cfg, "")


def test_reference_timestep_mode_in_trainer():
    from rulztrainer.trainer import sample_sigmas

    cfg = TrainConfig(model_path="briaai/Fibo-2.0-dev", timestep_sampling="reference")
    s = sample_sigmas(cfg, 64, torch.Generator().manual_seed(1), tokens=1024)
    assert s.shape == (64,) and s.min() >= 0.001 and s.max() <= 1.0
