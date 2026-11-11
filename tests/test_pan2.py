"""PAN 2 support without model weights: the one-frame port against the model's forward written out on a state dict
(the full 97-channel input, text-to-video's zero condition channels included); a padded batch against each sample
alone; the trainer's hooks; the converted files' header (tests/fixtures/pan2_headers.json: names, dtypes and shapes
only) against the port; single files in ComfyUI's quantized formats and the diffusers folder shard by shard; the text
template and its crop; the Wan 2.2 VAE as one-frame videos; the sampler; LoRA export; detection and downloads.

The port was also run against ComfyUI's PAN 2 model (comfy/ldm/pan2/model.py, the code that makes the videos) on the
same random weights: identical outputs, bit for bit, at three sizes including one that is not whole tokens."""
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402
from rulztrainer.config import TrainConfig, family_named_in  # noqa: E402
from rulztrainer.pan2 import loading as PL  # noqa: E402
from rulztrainer.pan2 import single_file as SF  # noqa: E402
from rulztrainer.pan2.model import MAX_TEXT_TOKENS, PAN2Transformer, rope_tables, shift_sigma  # noqa: E402
from rulztrainer.pan2.sampler import pan2_sigmas, sample  # noqa: E402

HEADERS = json.loads((Path(__file__).parent / "fixtures" / "pan2_headers.json").read_text("utf-8"))
ARCH = archs.arch_by_name("pan2")
TINY = dict(num_attention_heads=2, attention_head_dim=16, num_layers=3, num_refiner_layers=2, text_embed_dim=24,
            rope_axes_dim=(4, 6, 6))
WIDE = dict(num_attention_heads=2, attention_head_dim=128, num_layers=2, num_refiner_layers=1, text_embed_dim=64)


def tiny(seed=0, **kw):
    torch.manual_seed(seed)
    model = PAN2Transformer(**{**TINY, **kw}).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.08)
    return model


# ---------------------------------------------------------------- the forward, written out on the state dict


def reference(state: dict, cfg: dict, latents, timestep, text):
    """PAN2Transformer3DModel.forward for [B, 97, T, H, W] input, as diffusers and ComfyUI compute it."""
    heads, hd = cfg["num_attention_heads"], cfg["attention_head_dim"]

    def lin(name, x):
        return F.linear(x, state[name + ".weight"], state.get(name + ".bias"))

    def rms(name, x, eps):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * state[name + ".weight"]

    def split(x):
        return x.unflatten(-1, (heads, hd))

    def rope(x, cos, sin):
        a, b = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        return x * cos[None, :, None] + torch.stack([-b, a], -1).flatten(3) * sin[None, :, None]

    def attend(q, k, v):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).flatten(2)

    def ln(x):
        return F.layer_norm(x, x.shape[-1:], None, None, 1e-6)

    b, _, t_len, h, w = latents.shape
    hp, wp = h // 2, w // 2
    half = 128
    freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half) / half)
    emb = timestep[:, None] * freqs[None]
    temb = lin("time_embedder.linear_2", F.silu(lin("time_embedder.linear_1", torch.cat([emb.cos(), emb.sin()], -1))))
    cos, sin = rope_tables((t_len, hp, wp), cfg["rope_axes_dim"], 256.0, 16 / 24, "cpu")
    x = F.conv3d(latents, state["x_embedder.weight"], state["x_embedder.bias"], stride=(1, 2, 2)).flatten(2).transpose(1, 2)
    c = lin("context_embedder.1", rms("context_embedder.0", text, 1e-5))
    for i in range(cfg["num_refiner_layers"]):
        p = f"context_refiner.{i}."
        y = rms(p + "norm1", c, 1e-5)
        q, k = rms(p + "attn.norm_q", split(lin(p + "attn.to_q", y)), 1e-5), rms(p + "attn.norm_k", split(lin(p + "attn.to_k", y)), 1e-5)
        c = c + lin(p + "attn.to_out.0", attend(q, k, split(lin(p + "attn.to_v", y))))
        value, gate = lin(p + "ff.net.0.proj", rms(p + "norm2", c, 1e-5)).chunk(2, -1)
        c = c + lin(p + "ff.net.2", value * F.silu(gate))
    c = c + state["context_type_embedding"]
    n = x.shape[1]
    for i in range(cfg["num_layers"]):
        p = f"transformer_blocks.{i}."
        sh1, sc1, g1, sh2, sc2, g2 = lin(p + "norm1.linear", F.silu(temb)).unsqueeze(1).chunk(6, -1)
        csh1, csc1, cg1, csh2, csc2, cg2 = state[p + "scale_shift_table_context"].chunk(6, -1)
        nx, nc = ln(x) * (1 + sc1) + sh1, ln(c) * (1 + csc1) + csh1
        a = p + "attn."
        q = torch.cat([rope(rms(a + "norm_q", split(lin(a + "to_q", nx)), 1e-6), cos, sin),
                       rms(a + "norm_added_q", split(lin(a + "add_q_proj", nc)), 1e-6)], 1)
        k = torch.cat([rope(rms(a + "norm_k", split(lin(a + "to_k", nx)), 1e-6), cos, sin),
                       rms(a + "norm_added_k", split(lin(a + "add_k_proj", nc)), 1e-6)], 1)
        out = attend(q, k, torch.cat([split(lin(a + "to_v", nx)), split(lin(a + "add_v_proj", nc))], 1))
        x, c = x + lin(a + "to_out.0", out[:, :n]) * g1, c + lin(a + "to_add_out", out[:, n:]) * cg1
        x = x + lin(p + "ff.net.2", F.gelu(lin(p + "ff.net.0.proj", ln(x) * (1 + sc2) + sh2), approximate="tanh")) * g2
        c = c + lin(p + "ff_context.net.2", F.gelu(lin(p + "ff_context.net.0.proj", ln(c) * (1 + csc2) + csh2), approximate="tanh")) * cg2
    scale, shift = lin("norm_out.linear", F.silu(temb)).unsqueeze(1).chunk(2, -1)
    x = lin("proj_out", ln(x) * (1 + scale) + shift)
    x = x.reshape(b, t_len, hp, wp, 48, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def test_port_matches_the_reference_as_a_one_frame_text_to_video_request():
    model = tiny()
    state = model.state_dict()
    for h, w, s in ((6, 8, 9), (4, 10, 17)):
        x, text, t = torch.randn(2, 48, h, w), torch.randn(2, s, 24), torch.tensor([731.0, 55.0])
        full = torch.cat([x.unsqueeze(2), torch.zeros(2, 49, 1, h, w)], dim=1)  # no condition frames, mask 0
        with torch.no_grad():
            want = reference(state, model.config, full, t, text)[:, :, 0]
            got = model(x, t, text, torch.ones(2, s, dtype=torch.bool))
        assert got.shape == x.shape and got.dtype == torch.float32
        assert torch.allclose(got, want, atol=2e-5), float((got - want).abs().max())


def test_a_padded_batch_gives_each_sample_what_it_gets_alone():
    model = tiny(1)
    x, text, t = torch.randn(2, 48, 4, 6), torch.randn(2, 9, 24), torch.tensor([400.0, 900.0])
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, 5:] = False
    with torch.no_grad():
        both = model(x, t, text, mask)
        alone = model(x[:1], t[:1], text[:1, :5], mask[:1, :5])
        ignored = model(x, t, text, None)  # without the mask the padding is read
    assert torch.allclose(both[:1], alone, atol=1e-5)
    assert not torch.allclose(ignored[:1], alone, atol=1e-3)


def test_trainer_hooks():
    model = tiny(2)
    x, text = torch.randn(1, 48, 4, 4), torch.randn(1, 6, 24)
    assert len(model.layers) == 3 and model.split_tail_modules() == [model.norm_out, model.proj_out]
    assert ARCH.compile_blocks(model) == list(model.transformer_blocks)
    with torch.no_grad():
        want = model(x, 300.0, text, None)
        assert not torch.equal(model(x, 300.0, text, None, skip_layers=(1,)), want)
        odd = model(torch.randn(1, 48, 3, 5), 300.0, text, None)  # a probe size that is not whole 32 px
    assert odd.shape == (1, 48, 3, 5) and torch.isfinite(odd).all()
    names = ARCH.lora_module_names(model, ARCH.default_targets, None)
    per_block = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.add_q_proj", "attn.add_k_proj",
                 "attn.add_v_proj", "attn.to_add_out", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj",
                 "ff_context.net.2")
    # (the last block's text-side outputs feed nothing: they are left out, a LoRA there would never get a gradient)
    dead = {"transformer_blocks.2." + m for m in ("attn.add_q_proj", "attn.to_add_out", "ff_context.net.0.proj",
                                                  "ff_context.net.2")}
    assert sorted(names) == sorted({f"transformer_blocks.{i}.{m}" for i in range(3) for m in per_block} - dead)
    assert not any("norm1" in n or "x_embedder" in n or "proj_out" in n for n in names)
    with_text = ARCH.lora_module_names(model, ["q"], [0], include_text_embedder=True)
    assert "context_refiner.1.ff.net.0.proj" in with_text and "transformer_blocks.0.attn.add_q_proj" in with_text
    assert not any(n.startswith("transformer_blocks.1.") for n in with_text)
    # gradient checkpointing gives the same gradients as without it
    inj = L.inject_lora(model, 2, 2.0, ["q"], names=names)
    for mod in inj.values():
        nn.init.normal_(mod.lora_up.weight, std=0.05)
    model.train()
    grads = []
    for ckpt in (False, True):
        model.gradient_checkpointing = ckpt
        model.zero_grad()
        model(x, torch.tensor([600.0]), text, None).pow(2).mean().backward()
        grads.append([m.lora_down.weight.grad.clone() for m in inj.values()])
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(*grads))
    assert all(g.abs().sum() > 0 for g in grads[0])  # every target reaches the output


def test_alignment_taps_read_the_image_tokens_on_their_latent_cells():
    from rulztrainer.alignment import AttnTap, Taps

    model = tiny(3)
    taps = Taps(hidden_layers=(1,), attn_layers=(1,))
    model._rz_taps = taps
    model.transformer_blocks[1].attn._rz_tap = AttnTap(taps, 1)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 4:] = False
    taps.begin([[1, 2], [0]])
    with torch.no_grad():
        model(torch.randn(2, 48, 4, 6), torch.tensor([500.0, 500.0]), torch.randn(2, 7, 24), mask)
    assert taps.hidden[1][0].shape == (4 * 6, 32) and taps.attn[1][1].shape == (4 * 6, 7)
    probs = taps.attn[1][1]
    assert torch.allclose(probs.sum(-1), torch.ones(24), atol=1e-5) and float(probs[:, 4:].abs().max()) == 0.0
    taps.end()


# ---------------------------------------------------------------- the released layout


def test_converted_file_headers_fit_the_port():
    """Every tensor of PAN2_int8_convrot.safetensors (the diffusers checkpoint's names) has its place in the port."""
    head = HEADERS["PAN2_int8_convrot.safetensors"]
    config = json.loads(head["config"])
    assert config["_class_name"] == "PAN2Transformer3DModel"
    with torch.device("meta"):
        model = PAN2Transformer.from_config(config)
    want = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    side = (".comfy_quant", ".weight_scale", ".weight_scale_2")
    have = {k: tuple(shape) for k, (_, shape) in head["tensors"].items() if not k.endswith(side)}
    assert have == want
    assert len(model.transformer_blocks) == ARCH.n_blocks == 44 and model.hidden_size == 4096
    # what the checkpoint stores in fp32 is what the loader keeps in fp32
    fp32 = {k for k, (dtype, _) in head["tensors"].items() if dtype == "F32" and not k.endswith(side)}
    assert fp32 == {k for k in want if SF.is_fp32_key(k)}
    assert "transformer_blocks.7.scale_shift_table_context" in fp32 and "time_embedder.linear_1.bias" in fp32
    assert {n for n, m in model.named_modules() if isinstance(m, nn.Linear) and SF.is_fp32_module(n)} == {
        k.rpartition(".")[0] for k in fp32 if k.endswith(".weight")}
    quantized = {k[: -len(".comfy_quant")] for k in head["tensors"] if k.endswith(".comfy_quant")}
    assert quantized and all(SF.quant_filter(n, quant_norms=False) for n in quantized)
    assert SF.config_from_state({k: torch.empty(shape, device="meta") for k, shape in have.items()}) == {
        k: v for k, v in model.config.items() if k not in ("rope_axes_dim", "rope_theta", "rope_temporal_scale")}
    # every default LoRA target is a layer of the file, under the name ComfyUI keeps
    names = ARCH.lora_module_names(model, ARCH.default_targets, None, include_text_embedder=True)
    assert len(names) == 44 * 12 - 4 + 4 * 6 and all(n + ".weight" in have for n in names)


def comfy_file(tmp_path, model, name, quant=None, fp32=True, prefix=""):
    """`model` as a single file: fp32 islands kept fp32, the rest bf16, optionally its big layers quantized the way
    ComfyUI stores them, with the config in the metadata as the converter writes it."""
    out = {}
    for k, v in model.state_dict().items():
        mod = k.rpartition(".")[0]
        is_weight = k.endswith(".weight") and v.ndim == 2
        if quant and is_weight and SF.quant_filter(mod, quant_norms=not fp32) and min(v.shape) >= 256:
            qw = Q.quantize_weight(v.float(), quant)
            meta = ({"format": "nvfp4"} if quant == "nvfp4" else
                    {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
            out[prefix + mod + ".comfy_quant"] = torch.tensor(list(json.dumps(meta).encode()), dtype=torch.uint8)
            out[prefix + k], out[prefix + mod + ".weight_scale"] = qw.qdata, qw.scale
            if qw.scale2 is not None:
                out[prefix + mod + ".weight_scale_2"] = qw.scale2.reshape(())
        else:
            out[prefix + k] = (v.float() if fp32 and SF.is_fp32_key(k) else v.to(torch.bfloat16)).contiguous()
    path = tmp_path / name
    save_file(out, str(path), metadata={"config": json.dumps({"_class_name": "PAN2Transformer3DModel", **model.config})})
    return path


def wide(seed=0):
    torch.manual_seed(seed)
    model = PAN2Transformer(**WIDE).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.02)
    return model


def test_single_files_load_in_comfyuis_formats(tmp_path):
    model = wide()
    x, text = torch.randn(1, 48, 4, 4), torch.randn(1, 5, 64)
    with torch.no_grad():
        want = model(x, 500.0, text, None)
    plain = comfy_file(tmp_path, model, "PAN2_bf16.safetensors", prefix="diffusion_model.")
    assert archs.arch_from_transformer_file(str(plain)) == "pan2"
    loaded = SF.load_transformer_file(plain, "cpu", quant="none")
    assert loaded.time_embedder.linear_1.weight.dtype == torch.float32
    assert loaded.transformer_blocks[0].norm1.linear.weight.dtype == torch.float32
    assert loaded.transformer_blocks[1].scale_shift_table_context.dtype == torch.float32
    assert loaded.transformer_blocks[0].attn.to_q.weight.dtype == torch.bfloat16
    with torch.no_grad():
        assert torch.allclose(loaded(x, 500.0, text, None), want, atol=0.03, rtol=0.05)
        assert torch.allclose(ARCH.load_any_transformer(str(plain), "cpu", "none")(x, 500.0, text, None), want,
                              atol=0.03, rtol=0.05)
    for quant, fp32, tol in (("int8_convrot", True, 0.05), ("int8_convrot", False, 0.05), ("nvfp4", False, 0.3)):
        path = comfy_file(tmp_path, model, f"PAN2_{quant}_{fp32}.safetensors", quant=quant, fp32=fp32)
        loaded = SF.load_transformer_file(path, "cpu", quant="auto")
        to_q, norm = loaded.transformer_blocks[0].attn.to_q, loaded.transformer_blocks[1].norm1.linear
        assert isinstance(to_q, Q.QuantLinear) and to_q.fmt == quant
        assert isinstance(norm, Q.QuantLinear) != fp32  # kept fp32, or read quantized as the file has it
        assert loaded.config == model.config
        with torch.no_grad():
            got = loaded(x, 500.0, text, None)
        assert float((got - want).norm() / want.norm()) < tol, (quant, fp32)
    # a plain file quantized on load: the norm projections too, as PAN2_int8_convrot_full has them
    # (the real layers are 4096 wide and up; this model's are 256, so the size floor is lowered to reach them)
    loaded = SF.load_transformer_file(plain, "cpu", quant="int8_convrot", min_dim=256)
    assert isinstance(loaded.transformer_blocks[0].norm1.linear, Q.QuantLinear)
    assert isinstance(loaded.transformer_blocks[0].attn.add_k_proj, Q.QuantLinear)
    assert isinstance(loaded.context_refiner[0].ff.net[2], Q.QuantLinear)
    assert not isinstance(loaded.norm_out.linear, Q.QuantLinear) and not isinstance(loaded.proj_out, Q.QuantLinear)
    assert not isinstance(loaded.time_embedder.linear_2, Q.QuantLinear)
    with torch.no_grad():
        got = loaded(x, 500.0, text, None)
    assert float((got - want).norm() / want.norm()) < 0.05
    kept = SF.load_transformer_file(plain, "cpu", quant="int8_convrot", quant_norms=False, min_dim=256)
    assert not isinstance(kept.transformer_blocks[0].norm1.linear, Q.QuantLinear)
    assert isinstance(kept.transformer_blocks[0].attn.to_q, Q.QuantLinear)
    assert not isinstance(SF.load_transformer_file(plain, "cpu", quant="int8_convrot").transformer_blocks[0].attn.to_q,
                          Q.QuantLinear)  # the default floor: nothing this small is quantized
    with pytest.raises(ValueError, match="does not look like a PAN 2"):
        save_file({"x": torch.zeros(1)}, str(tmp_path / "other.safetensors"))
        SF.load_transformer_file(tmp_path / "other.safetensors", "cpu")


def test_the_diffusers_folder_loads_shard_by_shard(tmp_path):
    model = wide(1)
    tdir = tmp_path / "IFM__PAN2" / "transformer"
    tdir.mkdir(parents=True)
    (tdir / "config.json").write_text(json.dumps({"_class_name": "PAN2Transformer3DModel", **model.config}))
    items = sorted(model.state_dict().items())
    for i in range(3):  # a layer's weight and bias may sit in different shards
        save_file({k: (v if SF.is_fp32_key(k) else v.to(torch.bfloat16)).contiguous()
                   for k, v in items[i::3]}, str(tdir / f"diffusion_pytorch_model-0000{i + 1}-of-00003.safetensors"))
    x, text = torch.randn(1, 48, 4, 4), torch.randn(1, 5, 64)
    with torch.no_grad():
        want = model(x, 500.0, text, None)
    cfg = TrainConfig(model_path=str(tmp_path / "IFM__PAN2"), model_source="hf", transformer_quant="none", device="cpu")
    assert archs.detect_arch(cfg, tmp_path / "IFM__PAN2").name == "pan2"
    loaded = ARCH.get_transformer(cfg, tmp_path / "IFM__PAN2")
    assert loaded.norm_out.linear.weight.dtype == torch.float32
    with torch.no_grad():
        assert torch.allclose(loaded(x, 500.0, text, None), want, atol=0.03, rtol=0.05)
    (tdir / "diffusion_pytorch_model-00002-of-00003.safetensors").unlink()
    with pytest.raises(ValueError, match="weights not found"):
        SF.load_transformer_dir(tdir, "cpu")
    # a single file has every layer whole
    broken = {k: v.contiguous() for k, v in model.state_dict().items() if k != "proj_out.bias"}
    save_file(broken, str(tmp_path / "broken.safetensors"))
    with pytest.raises(ValueError, match="proj_out has no bias"):
        SF.load_transformer_file(tmp_path / "broken.safetensors", "cpu")


# ---------------------------------------------------------------- text


class FakeTokenizer:
    """Special tokens as one id each, every other character as its own id."""
    SPECIAL = ("<|im_start|>", "<|im_end|>", "<think>")

    def __len__(self):
        return 248_320

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_):
        ids, offsets, i = [], [], 0
        while i < len(text):
            for n, tok in enumerate(self.SPECIAL):
                if text.startswith(tok, i):
                    ids.append(200_000 + n)
                    offsets.append((i, i + len(tok)))
                    i += len(tok)
                    break
            else:
                ids.append(ord(text[i]))
                offsets.append((i, i + 1))
                i += 1
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def test_prompt_template_and_crop():
    tok = FakeTokenizer()
    text = PL.format_prompt("  a red fox \n")
    assert text == ("<|im_start|>system\nYou are a helpful assistant. Generate the video with the following "
                    "instructions.<|im_end|>\n<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n<think>\n")
    assert PL.format_prompt("") == PL.format_prompt(" ")  # the reference's empty prompt
    ids, start = PL.tokenize(tok, "a red fox")
    assert ids[start:start + 9].tolist() == [ord(c) for c in "a red fox"]
    assert ids[start - 6:start].tolist() == [200_000, *(ord(c) for c in "user\n")]  # right after the user header
    assert ids.numel() == len(tok(text)["input_ids"])  # the end of the turn and the opened assistant turn stay
    long_ids, long_start = PL.tokenize(tok, "x" * 5000)
    assert long_start == start and long_ids.numel() == start + MAX_TEXT_TOKENS
    assert PL.tokenize(tok, "x" * 5000, max_length=100)[0].numel() == start + 100
    shown, offsets, dropped, kept = PL.prompt_tokens(tok, "a red fox")
    assert shown == text and dropped == start and kept == ids.numel() - start
    assert shown[offsets[dropped][0]:offsets[dropped + 8][1]] == "a red fox"

    class Encoder(nn.Module):  # returns each token's id as its feature: the crop is visible in the output
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(1, 1)

        def forward(self, input_ids, attention_mask, **_):
            return type("Out", (), {"last_hidden_state": input_ids.float().unsqueeze(-1) * attention_mask.unsqueeze(-1)})

    rows = PL.encode_prompts(Encoder(), tok, ["a red fox", "hi"])
    assert rows[0].dtype == torch.bfloat16 and rows[0][:9, 0].tolist() == [float(ord(c)) for c in "a red fox"]
    assert rows[1].shape[0] == rows[0].shape[0] - 7 and rows[1][:2, 0].tolist() == [float(ord("h")), float(ord("i"))]
    with pytest.raises(ValueError, match="not PAN 2's Qwen3.5 tokenizer"):
        PL.check_tokenizer(range(151_000))


def test_negative_prompt_and_triggers():
    neg = json.loads(ARCH.default_negative)
    assert neg["events"][0]["time_range_sec"] == "0 to 0" and "watermark" in neg["video_style"]
    assert json.loads(PL.default_negative(5.0))["events"][0]["camera_motion"][0]["time_range_sec"] == "0 to 5"
    assert ARCH.empty_prompt == ""
    assert ARCH.add_trigger("a photo of a woman", "ohwx") == "ohwx, a photo of a woman"
    structured = json.dumps({"video_style": "film", "subjects_lifecycle": [{"subject_id": "<|Subject1|>", "display_name": "woman"}]})
    with_trigger = json.loads(ARCH.add_trigger(structured, "ohwx"))
    assert with_trigger["subjects_lifecycle"][0]["display_name"] == "ohwx woman" and with_trigger["video_style"] == "film"
    assert ARCH.add_trigger(structured.replace("woman", "ohwx woman"), "ohwx") == structured.replace("woman", "ohwx woman")
    assert ARCH.add_trigger('{"no": "subjects"}', "ohwx").startswith("ohwx, {")


# ---------------------------------------------------------------- VAE


def test_stills_go_through_the_wan22_vae_as_one_frame_videos(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "AutoencoderKLWan"):
        pytest.skip("this diffusers has no AutoencoderKLWan")
    config = json.loads((Path(PL.__file__).parent / "wan22_vae_config.json").read_text())
    assert config["z_dim"] == 48 and len(config["latents_mean"]) == len(config["latents_std"]) == 48
    import inspect

    accepted = set(inspect.signature(diffusers.AutoencoderKLWan.__init__).parameters)
    if "is_residual" not in accepted:
        pytest.skip("this diffusers' AutoencoderKLWan predates the Wan 2.2 VAE")
    small = {**config, "base_dim": 8, "decoder_base_dim": 8, "num_res_blocks": 1}
    torch.manual_seed(0)
    src = diffusers.AutoencoderKLWan(**{k: v for k, v in small.items() if k in accepted}).eval()
    path = tmp_path / "PAN2_vae_fp16.safetensors"
    save_file({k: v.to(torch.float16).contiguous() for k, v in src.state_dict().items()}, str(path),
              metadata={"config": json.dumps(small)})
    vae = SF.load_vae_file(path, "cpu")
    assert next(vae.parameters()).dtype == torch.float32
    images = torch.rand(2, 3, 64, 96) * 2 - 1
    latents = ARCH.encode_images(vae, images)
    assert latents.shape == (2, 48, 4, 6) and latents.dtype == torch.float32
    mean = torch.tensor(config["latents_mean"]).view(1, -1, 1, 1)
    std = torch.tensor(config["latents_std"]).view(1, -1, 1, 1)
    with torch.no_grad():
        raw = vae.encode(images.unsqueeze(2)).latent_dist.mode()[:, :, 0]
    assert torch.allclose(latents, (raw - mean) / std, atol=1e-5)
    decoded = ARCH.decode_latents(vae, latents)
    assert decoded.shape == images.shape and float(decoded.abs().max()) <= 1.0
    with torch.no_grad():
        want = vae.decode((latents * std + mean).unsqueeze(2), return_dict=False)[0][:, :, 0].clamp(-1, 1)
    assert torch.allclose(decoded, want, atol=1e-5)
    seed = latents.clone().requires_grad_(True)
    ARCH.decode_latents_grad(ARCH.grad_vae(vae), seed).mean().backward()
    assert seed.grad is not None and torch.isfinite(seed.grad).all()
    with pytest.raises(ValueError, match="not a Wan 2.2 VAE"):
        save_file({"encoder.conv_in.weight": torch.zeros(1), "decoder.conv_in.weight": torch.zeros(4, 16, 3, 3)},
                  str(tmp_path / "other_vae.safetensors"))
        SF.load_vae_file(tmp_path / "other_vae.safetensors", "cpu")


# ---------------------------------------------------------------- sampling, timesteps, LoRA files


def test_sampler_and_reference_timesteps():
    sig = pan2_sigmas(50)
    base = torch.linspace(1, 0, 51)
    assert sig.numel() == 51 and float(sig[0]) == 1.0 and float(sig[-1]) == 0.0
    assert torch.allclose(sig, 9 * base / (1 + 8 * base)) and torch.all(sig[:-1] > sig[1:])
    assert torch.equal(ARCH.sample_sigmas(tiny(), 50, 512, 512, 3.0), sig)  # the trainer's sample_shift is ignored
    assert (ARCH.default_sample_steps, ARCH.default_sample_cfg) == (50, 2.5)
    model = tiny(4)
    pos, neg, mask = torch.randn(1, 5, 24), torch.randn(1, 3, 24), torch.ones(1, 5, dtype=torch.bool)
    seen = []
    out = sample(model, pos, mask, neg, mask[:, :3], width=80, height=112, steps=4, cfg=2.5, seed=3, device="cpu",
                 progress=lambda i, n: seen.append((i, n)))
    assert out.shape == (1, 48, 6, 4) and out.dtype == torch.float32 and torch.isfinite(out).all()
    assert seen == [(i, 4) for i in range(1, 5)]
    plain = sample(model, pos, mask, None, None, width=80, height=112, steps=4, cfg=1.0, seed=3, device="cpu")
    assert not torch.allclose(out, plain)
    # one Euler step from pure noise at CFG 1 is noise + (sigma_1 - 1) * velocity
    noise = torch.randn((1, 48, 6, 4), generator=torch.Generator().manual_seed(3))
    one = sample(model, pos, mask, None, None, width=80, height=112, steps=1, cfg=1.0, seed=3, device="cpu")
    with torch.no_grad():
        assert torch.allclose(one, noise - model(noise.to(torch.bfloat16), 1000.0, pos, mask), atol=1e-4)
    # training noise levels: IFM's fine-tuning draw, uniform through the fixed shift 9
    gen = torch.Generator().manual_seed(0)
    drawn = ARCH.reference_sigmas(20000, gen, 1024)
    assert 0.0 < float(drawn.min()) and float(drawn.max()) < 1.0
    assert abs(float(drawn.median()) - float(shift_sigma(torch.tensor(0.5)))) < 0.01  # 0.9
    assert not ARCH.reference_sigmas_exact


def test_lora_export_is_what_comfyui_maps(tmp_path):
    model = tiny(5)
    names = ARCH.lora_module_names(model, ARCH.default_targets, [0], include_text_embedder=True)
    inj = L.inject_lora(model, 4, 8.0, ["q"], names=names)
    path = tmp_path / "pan2_lora.safetensors"
    L.save_lora(inj, path, dtype=torch.float32, fmt=ARCH.lora_format)
    saved = load_file(str(path))
    assert set(saved) == {f"diffusion_model.{n}.{part}" for n in names
                          for part in ("lora_down.weight", "lora_up.weight", "alpha")}
    assert saved["diffusion_model.transformer_blocks.0.attn.to_out.0.lora_down.weight"].shape == (4, 32)
    assert float(saved["diffusion_model.transformer_blocks.0.attn.add_q_proj.alpha"]) == 8.0
    # every key names a weight of the checkpoint: "diffusion_model." + its diffusers name, as ComfyUI keeps them
    weights = set(HEADERS["PAN2_int8_convrot.safetensors"]["tensors"])
    assert all(k.split(".lora_")[0].split(".alpha")[0][len("diffusion_model."):] + ".weight" in weights for k in saved)


# ---------------------------------------------------------------- detection and downloads


def test_detection_and_downloads(tmp_path):
    from rulztrainer.comfy_files import planned_downloads
    from rulztrainer.config import ARCH_NAMES, ARCH_SAMPLE_DEFAULTS, ARCH_TRAINING_ADAPTER, ARCH_TURBO, MODEL_PRESETS

    assert "pan2" in ARCH_NAMES and "pan2" in archs.ARCH_NAMES and MODEL_PRESETS["PAN-2"] == "IFM/PAN2"
    for path in ("IFM/PAN2", "PAN2_nvfp4.safetensors", "D:/models/pan2_int8_convrot_full.safetensors", "/data/PAN-2",
                 "pan_2.safetensors"):
        assert family_named_in(path) == "pan2", path
    for path in ("japan2.safetensors", "expand2/model", "span2", "krea/Krea-2-Raw", "MiniMaxAI/MiniMax-H3"):
        assert family_named_in(path) != "pan2", path
    cfg = TrainConfig(model_path="IFM/PAN2", model_source="hf")
    assert cfg.arch_name() == "pan2" and archs.get_arch(cfg) is ARCH
    assert ARCH.latent_tag == "_wan22vae" and ARCH.size_multiple == 32 and ARCH.lora_format == "comfy"
    assert not cfg.samples_on_distilled_model()
    # nothing to fetch: no turbo LoRA, no training adapter, and no published single files
    assert "pan2" not in ARCH_TURBO and "pan2" not in ARCH_TRAINING_ADAPTER and "pan2" not in ARCH_SAMPLE_DEFAULTS
    assert planned_downloads(cfg) == []
    with pytest.raises(ValueError, match="set transformer_file, text_encoder_file, vae_file"):
        planned_downloads(TrainConfig(model_path="IFM/PAN2"))  # files is the default source
    files = TrainConfig(model_path="IFM/PAN2", model_source="files", transformer_file="PAN2_nvfp4.safetensors")
    with pytest.raises(ValueError, match="set text_encoder_file, vae_file"):
        planned_downloads(files)
    files = TrainConfig(model_path="IFM/PAN2", model_source="files", transformer_file="a.safetensors",
                        text_encoder_file="b.safetensors", vae_file="c.safetensors")
    assert planned_downloads(files) == [] and PL.resolve_for_config(files) is None
    # a file is told by its contents, whatever it is called
    model = tiny(6)
    path = tmp_path / "video_model.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    assert archs.arch_from_transformer_file(str(path)) == "pan2"
    renamed = TrainConfig(model_path="", model_source="files", transformer_file=str(path))
    assert archs.guess_arch_name(renamed) == "pan2"
