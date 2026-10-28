"""Qwen-Image 2.1 support without model weights: the ported transformer against diffusers'
QwenImage21Transformer2DModel (tiny random config, padded batches per sample), the ComfyUI single-file layout (fused
feed-forward), LoRA naming / PEFT export and the fused-layout turbo LoRA, the training adapter (on while training, off
while sampling, never saved), the sampler schedules against FlowMatchEulerDiscreteScheduler, and architecture
detection, caches and the 32 px buckets. The VAE converter is checked against Comfy-Org's file when it is cached."""
import glob
import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import dataset as D  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.qwen21 import loading as QL  # noqa: E402
from rulztrainer.qwen21 import single_file as QSF  # noqa: E402
from rulztrainer.qwen21.model import QwenImage21Transformer  # noqa: E402
from rulztrainer.qwen21.sampler import TURBO_SIGMAS, calculate_shift, qwen21_sigmas  # noqa: E402

diffusers = pytest.importorskip("diffusers")

TINY = dict(in_channels=64, out_channels=64, num_layers=3, attention_head_dim=32, num_attention_heads=4,
            context_in_dim=48, mlp_ratio=3, axes_dims_rope=(4, 14, 14), eps=1e-6, causal_condition=True)
QWEN = archs.arch_by_name("qwen_image21")


def pair(seed=0):
    """Our port and diffusers' model with the same random weights, fp32."""
    torch.manual_seed(seed)
    ref = diffusers.QwenImage21Transformer2DModel(patch_size=1, **TINY).float().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.05)
    ours = QwenImage21Transformer(**TINY).float().eval()
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=True)
    assert not missing and not unexpected
    return ours, ref


def ref_call(ref, x, sigma, enc):
    """diffusers' forward for one sample, unpadded text, on the trainer's [1, 64, h, w] latents."""
    _, c, h, w = x.shape
    img_mask = torch.cat([torch.zeros(1, enc.shape[1], dtype=torch.bool), torch.ones(1, h * w // 4, dtype=torch.bool)],
                         dim=1)
    out = ref(hidden_states=x.flatten(2).transpose(1, 2), encoder_hidden_states=enc, timestep=sigma,
              img_shapes=[[(1, h, w)]], img_mask=img_mask, return_dict=False)[0]
    return out[:, -h * w:].transpose(1, 2).reshape(1, c, h, w)


def test_port_matches_diffusers_per_sample_with_padding():
    ours, ref = pair()
    x = torch.randn(2, 64, 4, 6)
    sigma = torch.tensor([0.7, 0.25])
    lens = (9, 5)
    enc = torch.randn(2, max(lens), 48)
    mask = torch.zeros(2, max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        mask[i, :n] = True
    with torch.no_grad():
        got = ours(x, sigma * 1000.0, enc, mask)
        for i, n in enumerate(lens):  # each sample as the reference computes it alone, without the padding
            want = ref_call(ref, x[i:i + 1], sigma[i:i + 1], enc[i:i + 1, :n])
            assert torch.allclose(got[i:i + 1], want, atol=1e-4, rtol=1e-4)
        # an all-true mask is no mask
        assert torch.equal(ours(x[:1], 700.0, enc[:1], mask[:1]), ours(x[:1], 700.0, enc[:1], None))


def test_bf16_gradients_checkpointing_odd_grid_and_layers():
    ours, _ = pair(1)
    m = ours.to(torch.bfloat16)
    m.gradient_checkpointing = True
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.randn(1, 64, 3, 5).to(torch.bfloat16)  # an odd grid runs too (the pipeline's grid is even)
    enc = torch.randn(1, 4, 48).to(torch.bfloat16)
    out = m(x, torch.tensor([500.0]), enc, None)
    assert out.shape == x.shape and out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in m.transformer_blocks.parameters())
    with torch.no_grad():
        skipped = m(x, 500.0, enc, None, skip_layers=[1])
    assert not torch.equal(skipped, out.detach())
    assert m.layers is m.transformer_blocks and len(m.layers) == 3 and m.latent_channels == 64
    assert m.split_tail_modules() == [m.norm_out, m.proj_out] and QWEN.compile_blocks(m) == list(m.transformer_blocks)


def test_model_split_hand_off_matches():
    from rulztrainer.multigpu import split_model

    ours, _ = pair(2)
    x, enc = torch.randn(1, 64, 2, 2), torch.randn(1, 4, 48)
    with torch.no_grad():
        want = ours(x, 300.0, enc, None)
        split_model(ours, ["cpu", "cpu"])  # same device twice: exercises the hand-off path
        assert torch.equal(ours(x, 300.0, enc, None), want)


def comfy_state(ref) -> dict:
    """diffusers' state dict in ComfyUI's single-file layout: gate_layer / proj fused into gate_up, diffusion_model."""
    sd = ref.state_dict()
    out = {}
    for k, v in sd.items():
        if k.endswith(".img_mlp.proj.weight"):
            continue
        if k.endswith(".img_mlp.gate_layer.weight"):
            stem = k[: -len(".gate_layer.weight")]
            out[f"diffusion_model.{stem}.gate_up.weight"] = torch.cat([v, sd[f"{stem}.proj.weight"]])
            continue
        out[f"diffusion_model.{k}"] = v
    return out


def test_dit_from_comfy_layout_matches_and_is_detected(tmp_path):
    from safetensors.torch import save_file

    ours, ref = pair(3)
    path = tmp_path / "qwen_image_2.1_turbo_bf16.safetensors"
    save_file({k: v.contiguous() for k, v in comfy_state(ref).items()}, str(path))
    loaded = QSF.load_transformer_file(path, "cpu", quant="none", dtype=torch.float32)
    assert loaded.config["num_layers"] == 3 and loaded.config["mlp_ratio"] == 3 and loaded.is_distilled
    x, enc = torch.randn(1, 64, 2, 4), torch.randn(1, 6, 48)
    with torch.no_grad():
        assert torch.allclose(loaded(x, 600.0, enc, None), ours(x, 600.0, enc, None), atol=1e-5)
    assert QSF.is_qwen21_file(path) and archs.arch_from_transformer_file(str(path)) == "qwen_image21"
    assert TrainConfig(model_source="files", transformer_file=str(path), model_arch="auto").arch_name() == \
        "qwen_image21"


def test_quantized_gate_up_splits_by_rows():
    w = torch.randn(2 * 1024, 1024)
    for fmt in ("int8", "int8_convrot", "nvfp4"):
        qw = Q.quantize_weight(w, fmt)
        state = QSF.unfuse_mlp({"transformer_blocks.0.img_mlp.gate_up.weight": qw})
        gate, proj = state["transformer_blocks.0.img_mlp.gate_layer.weight"], state[
            "transformer_blocks.0.img_mlp.proj.weight"]
        full = qw.dequantize(torch.float32)
        tol = 0.0 if fmt != "nvfp4" else 0.2
        assert torch.allclose(gate.dequantize(torch.float32), full[:1024], atol=tol + 1e-6)
        assert torch.allclose(proj.dequantize(torch.float32), full[1024:], atol=tol + 1e-6)


def test_lora_names_peft_export_and_comfy_mapping(tmp_path):
    ours, ref = pair(4)
    everything = QWEN.lora_module_names(ours, list(QWEN.target_groups), None)
    linears = [n for n, mod in ours.named_modules() if isinstance(mod, torch.nn.Linear)]
    assert sorted(everything) == sorted(linears)  # all groups together = every Linear
    default = QWEN.lora_module_names(ours, QWEN.default_targets, None)
    assert len(default) == 3 * 7  # q k v o, gate_layer proj, out per block
    assert "transformer_blocks.1.img_mlp.gate_layer" in default and "transformer_blocks.1.img_mlp.proj" in default
    assert QWEN.lora_module_names(ours, ["q"], [0, 2]) == ["transformer_blocks.0.attn.to_q",
                                                           "transformer_blocks.2.attn.to_q"]
    assert set(QWEN.lora_module_names(ours, ["q"], [0], include_text_embedder=True)) == {
        "txt_in.in_layer", "txt_in.out_layer", "transformer_blocks.0.attn.to_q"}
    assert QWEN.param_group_of("transformer_blocks.0.img_mlp.proj") == "mlp"
    assert QWEN.param_group_of("txt_in.in_layer") == "text"
    names = QWEN.lora_module_names(ours, ["q", "ffn_up_gate", "ffn_down", "embed", "text_in"], None)
    inj = L.inject_lora(ours, 4, 8.0, ["q"], names=names)
    with torch.no_grad():
        for mod in inj.values():
            mod.lora_up.weight.normal_(0, 0.1)
    path = tmp_path / "qwen_lora.safetensors"
    L.save_lora(inj, path, dtype=torch.float32, fmt=QWEN.lora_format)
    from safetensors.torch import load_file

    state = load_file(str(path))
    # ComfyUI maps bare / transformer.-prefixed gate_layer and proj onto its fused gate_up halves
    assert "transformer.transformer_blocks.0.img_mlp.gate_layer.lora_A.weight" in state
    assert "transformer.transformer_blocks.2.img_mlp.proj.lora_B.weight" in state
    assert "transformer.modulation.1.lora_A.weight" in state and "transformer.txt_in.out_layer.lora_B.weight" in state
    with torch.no_grad():  # merged into diffusers' model the PEFT way, it predicts what our LoRA model predicts
        for name in names:
            a, b = state[f"transformer.{name}.lora_A.weight"], state[f"transformer.{name}.lora_B.weight"]
            ref.get_submodule(name).weight += (8.0 / 4) * b @ a
        x, enc = torch.randn(1, 64, 2, 4), torch.randn(1, 5, 48)
        assert torch.allclose(ours(x, 400.0, enc, None), ref_call(ref, x, torch.tensor([0.4]), enc),
                              atol=1e-3, rtol=1e-3)


def test_fused_turbo_lora_lands_on_both_halves(tmp_path):
    from safetensors.torch import save_file

    ours, ref = pair(5)
    stem = "diffusion_model.transformer_blocks.1.img_mlp.gate_up"
    down, up = torch.randn(3, 128) * 0.1, torch.randn(2 * 384, 3) * 0.1
    path = tmp_path / "turbo.safetensors"
    save_file({f"{stem}.lora_down.weight": down, f"{stem}.lora_up.weight": up, f"{stem}.alpha": torch.tensor(6.0),
               "diffusion_model.transformer_blocks.0.attn.norm_q.diff": torch.full((32,), 0.5)}, str(path))
    n = L.attach_extra_lora(ours, str(path), 1.0, convert=QWEN.extra_lora_converter)
    assert n == 3  # gate_layer, proj and the norm offset
    L.set_extras_enabled(ours, True)
    with torch.no_grad():
        delta = (6.0 / 3) * up @ down
        mlp = ref.transformer_blocks[1].img_mlp
        mlp.gate_layer.weight += delta[:384]
        mlp.proj.weight += delta[384:]
        ref.transformer_blocks[0].attn.norm_q.weight += 0.5
        x, enc = torch.randn(1, 64, 2, 2), torch.randn(1, 3, 48)
        assert torch.allclose(ours(x, 800.0, enc, None), ref_call(ref, x, torch.tensor([0.8]), enc), atol=1e-4)


def test_training_adapter_is_on_while_training_only_and_never_saved(tmp_path):
    from safetensors.torch import load_file, save_file

    ours, ref = pair(6)
    name = "transformer_blocks.0.attn.to_v"
    a, b = torch.randn(4, 128) * 0.2, torch.randn(128, 4) * 0.2
    path = tmp_path / "adapter.safetensors"
    save_file({f"transformer.{name}.lora_A.weight": a, f"transformer.{name}.lora_B.weight": b,
               f"transformer.{name}.alpha": torch.tensor(2.0)}, str(path))
    x, enc = torch.randn(1, 64, 2, 2), torch.randn(1, 3, 48)
    with torch.no_grad():
        plain = ours(x, 500.0, enc, None)
    inj = L.inject_lora(ours, 2, 2.0, ["q"], names=["transformer_blocks.0.attn.to_q"])
    assert L.attach_training_adapter(ours, str(path), strength=1.5) == 1
    wrapped = ours.get_submodule(name)
    assert isinstance(wrapped, L.LoRALinear) and not wrapped.trainable
    assert not any(n.endswith("train_down") for n in ours.state_dict())  # non-persistent: in no saved state
    with torch.no_grad():
        trained = ours(x, 500.0, enc, None)  # extras off: training, the adapter is on
        L.set_extras_enabled(ours, True)  # sampling: off
        sampling = ours(x, 500.0, enc, None)
        L.set_extras_enabled(ours, False)
        ref.get_submodule(name).weight += 1.5 * (2.0 / 4) * b @ a
        want = ref_call(ref, x, torch.tensor([0.5]), enc)
    assert torch.equal(sampling, plain) and not torch.equal(trained, plain)
    assert torch.allclose(trained, want, atol=1e-4)
    out = tmp_path / "trained.safetensors"
    L.save_lora(inj, out, dtype=torch.float32, fmt="peft")
    assert not any("to_v" in k for k in load_file(str(out)))  # only the trained LoRA is saved
    ours.to(torch.float64)  # buffers follow the model (moves, block swap, model split)
    assert wrapped.train_down.dtype == torch.float64


def test_sampler_schedules_match_the_scheduler():
    base = diffusers.FlowMatchEulerDiscreteScheduler(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192,
                                                     max_shift=0.9, shift=1.0, shift_terminal=0.02,
                                                     time_shift_type="exponential", use_dynamic_shifting=True)
    for steps, tokens in ((25, 4096), (40, 2304), (8, 1024)):
        base.set_timesteps(sigmas=torch.linspace(1.0, 1 / steps, steps).numpy(), mu=calculate_shift(tokens))
        assert torch.allclose(qwen21_sigmas(steps, tokens), base.sigmas.float(), atol=1e-5)
    turbo = diffusers.FlowMatchEulerDiscreteScheduler(shift=1.0, use_dynamic_shifting=False,
                                                      time_shift_type="exponential")
    turbo.set_timesteps(sigmas=list(TURBO_SIGMAS))
    assert torch.allclose(qwen21_sigmas(8, 4096, distilled=True), turbo.sigmas.float(), atol=1e-6)
    twelve = qwen21_sigmas(12, 4096, distilled=True)  # another step count follows the same curve
    assert len(twelve) == 13 and twelve[0] == 1.0 and abs(float(twelve[-2]) - TURBO_SIGMAS[-1]) < 1e-6
    assert abs(calculate_shift(256) - 0.5) < 1e-9 and abs(calculate_shift(8192) - 0.9) < 1e-9
    r = QWEN.reference_sigmas(10000, torch.Generator().manual_seed(0), 4096)
    assert r.min() >= 0 and r.max() <= 1 and 0.55 < r.mean() < 0.75


def test_auto_quant_counts_the_gpus_the_model_runs_on(monkeypatch):
    sizes = {0: 16, 1: 8, 2: 24}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(QL, "_gpu_gb", lambda d: sizes[torch.device(d).index or 0])
    assert QL.resolve_quant("auto", "cuda:0") == "int8_convrot"
    assert QL.resolve_quant("auto", "cuda:1") == "nvfp4" and QL.resolve_quant("auto", "cuda:2") == "none"
    assert QL.resolve_quant("auto", ["cuda:0", "cuda:1"]) == "none"
    assert QL.resolve_quant("bf16", "cuda:0") == "bf16"


def test_detection_caches_buckets_and_downloads(tmp_path):
    from rulztrainer.comfy_files import planned_downloads

    embeds, mask = QL.pad_embeds([torch.randn(4, 8), torch.randn(6, 8)])
    assert embeds.shape == (2, 6, 8) and mask.sum(1).tolist() == [4, 6]
    for path in ("Qwen/Qwen-Image-2.1", "Qwen/Qwen-Image-2.1-Turbo", "C:/m/qwen_image_2.1_int8_convrot.safetensors"):
        assert TrainConfig(model_path=path).arch_name() == "qwen_image21"
    mdir = tmp_path / "model" / "transformer"
    mdir.mkdir(parents=True)
    (mdir / "config.json").write_text(json.dumps({"_class_name": "QwenImage21Transformer2DModel"}))
    assert archs.detect_arch(TrainConfig(model_path=str(tmp_path / "model")), tmp_path / "model").name == "qwen_image21"
    data = tmp_path / "data"
    data.mkdir()
    from PIL import Image

    Image.new("RGB", (900, 600)).save(data / "a.png")
    qwen = TrainConfig(model_path="Qwen/Qwen-Image-2.1", dataset_dir=str(data), resolutions=[0.5])
    item = D.scan_dataset(qwen)[0]
    assert item.width % 32 == 0 and item.height % 32 == 0 and item.latent_path.name.endswith("_qi21vae.safetensors")
    assert abs(item.width * item.height - 0.5 * 2**20) < 0.08 * 2**20
    assert D.text_cache_path(qwen, "p").name.endswith("_qwen_image21.safetensors")
    plan = {f: (r, n) for f, r, n in planned_downloads(TrainConfig(model_path="Qwen/Qwen-Image-2.1"))}
    assert plan["transformer_file"] == ("Comfy-Org/Qwen-Image-2.1",
                                        "diffusion_models/qwen_image_2.1_int8_convrot.safetensors")
    assert plan["text_encoder_file"][1] == "text_encoders/qwen3vl_8b_int8_convrot.safetensors"
    assert plan["vae_file"][1] == "vae/qwen_image_2.1_vae_bf16.safetensors"
    assert plan["turbo_lora"][1].startswith("loras/qwen_image_2.1_turbo_lora")
    assert plan["training_adapter"][0] == "ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter"
    assert "training_adapter" not in {f for f, _, _ in planned_downloads(TrainConfig(model_path="krea/Krea-2-Raw"))}
    assert "training_adapter" not in {f for f, _, _ in planned_downloads(
        TrainConfig(model_path="Qwen/Qwen-Image-2.1", training_adapter=""))}


def _cached_vae() -> str | None:
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser(
        "~/.cache/huggingface")), "hub")
    hits = glob.glob(os.path.join(root, "models--Comfy-Org--Qwen-Image-2.1", "snapshots", "*", "vae",
                                  "qwen_image_2.1_vae_bf16.safetensors"))
    return hits[0] if hits else None


@pytest.mark.skipif(_cached_vae() is None, reason="Comfy-Org's Qwen-Image 2.1 VAE is not in the Hugging Face cache")
def test_comfy_vae_converts_to_diffusers_and_round_trips():
    vae = QSF.load_vae_file(_cached_vae(), "cpu", dtype=torch.float32)
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 128), torch.linspace(-1, 1, 128), indexing="ij")
    img = torch.stack([xx, yy, (xx * yy)]).unsqueeze(0) * 0.8  # smooth gradients reconstruct well
    with torch.no_grad():
        lat = QL.encode_images(vae, img)
        assert lat.shape == (1, 64, 8, 8) and 0.3 < lat.std() < 3.0  # normalized
        back = QL.decode_latents(vae, lat)
    assert back.shape == img.shape and (back - img).pow(2).mean() < 1e-3
