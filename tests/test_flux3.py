"""FLUX 3 Klein support without model weights: the ported transformer against diffusers' Flux3Transformer2DModel (tiny
random config in fp32, padded batches per sample, the distilled flow map's target timestep), BFL's / ComfyUI's
single-file layout (and the fused qkv of ComfyUI's int8 files) against the real key names, the diffusers layout
round trip, LoRA naming and the export both ComfyUI and diffusers read, the sampler schedule against
FlowMatchEulerDiscreteScheduler, and architecture detection, caches and the 16 px buckets."""
import json
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
from rulztrainer.flux3 import loading as FL  # noqa: E402
from rulztrainer.flux3 import single_file as FSF  # noqa: E402
from rulztrainer.flux3.model import Flux3KleinTransformer  # noqa: E402
from rulztrainer.flux3.sampler import SHIFT, flux3_sigmas  # noqa: E402

diffusers = pytest.importorskip("diffusers")
pytest.importorskip("diffusers.models.transformers.transformer_flux3")

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY = dict(in_channels=16, num_layers=2, num_single_layers=3, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=24, timestep_channels=32, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4))


def pair(seed=0, flow_map=False):
    """Our port and diffusers' model with the same random weights, fp32. Ours is loaded from diffusers' state dict
    renamed to BFL's layout (the reference-image mode dropped)."""
    torch.manual_seed(seed)
    ref = diffusers.Flux3Transformer2DModel(cond_in_channels=16, flow_map=flow_map, **TINY).float().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.1)
    ours = Flux3KleinTransformer(flow_map=flow_map, **TINY).float().eval()
    state = FSF.to_model_layout(dict(ref.state_dict()))
    missing, unexpected = ours.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    return ours, ref


def ref_call(ref, x, sigma, enc, target=None):
    """diffusers' forward for one sample, its own text rows only, on the trainer's [1, C, h, w] latents."""
    from diffusers.modular_pipelines.flux3.before_denoise import Flux3TextRoPEInputsStep, _prepare_image_ids

    _, c, h, w = x.shape
    kw = {} if target is None else {"target_timestep": target}
    out = ref(hidden_states=x.flatten(2).transpose(1, 2), encoder_hidden_states=enc, timestep=sigma,
              img_ids=_prepare_image_ids(h, w, 0.5), txt_ids=Flux3TextRoPEInputsStep._prepare_text_ids(enc),
              return_dict=False, **kw)[0]
    return out.transpose(1, 2).reshape(1, c, h, w)


def test_port_matches_diffusers_per_sample_with_padding():
    ours, ref = pair()
    x = torch.randn(2, 16, 4, 6)
    sigma = torch.tensor([0.7, 0.25])
    lens = (9, 5)
    enc = torch.randn(2, max(lens), 24)
    mask = torch.zeros(2, max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        mask[i, :n] = True
    with torch.no_grad():
        got = ours(x, sigma * 1000.0, enc, mask)
        for i, n in enumerate(lens):  # each sample as the reference computes it alone, without the batch padding
            want = ref_call(ref, x[i:i + 1], sigma[i:i + 1], enc[i:i + 1, :n])
            assert torch.allclose(got[i:i + 1], want, atol=1e-5, rtol=1e-4), (got[i:i + 1] - want).abs().max()
        # an all-true mask is no mask
        assert torch.equal(ours(x[:1], 700.0, enc[:1], mask[:1] | True), ours(x[:1], 700.0, enc[:1], None))


def test_flow_map_target_timestep_matches_diffusers():
    ours, ref = pair(1, flow_map=True)
    assert ours.flow_map and ours.is_distilled and ours.ref_time_in is not None
    x, enc = torch.randn(1, 16, 2, 4), torch.randn(1, 7, 24)
    with torch.no_grad():
        got = ours(x, 600.0, enc, None, target_timestep=250.0)
        want = ref_call(ref, x, torch.tensor([0.6]), enc, target=torch.tensor([0.25]))
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)
        assert not torch.allclose(got, ours(x, 600.0, enc, None, target_timestep=0.0))
    with pytest.raises(ValueError, match="flow-map"):
        ours(x, 600.0, enc, None)  # the distilled model needs its target
    base, _ = pair(1)
    with pytest.raises(ValueError, match="flow-map"):
        base(x, 600.0, enc, None, target_timestep=250.0)


def test_bf16_gradients_checkpointing_skip_and_layout():
    ours, _ = pair(2)
    m = ours.to(torch.bfloat16)
    m.gradient_checkpointing = True
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.randn(1, 16, 3, 5).to(torch.bfloat16)  # an odd grid runs too
    enc = torch.randn(1, 4, 24).to(torch.bfloat16)
    out = m(x, torch.tensor([500.0]), enc, None)
    assert out.shape == x.shape and out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in m.single_blocks.parameters())
    assert all(p.grad is not None for b in m.mode_blocks for p in b.parameters())
    with torch.no_grad():
        skipped = m(x, 500.0, enc, None, skip_layers=[1])
    assert not torch.equal(skipped, out.detach())
    assert m.layers is m.single_blocks and len(m.layers) == 3 and m.latent_channels == 16
    assert len(m.mode_blocks) == 4 and m.split_tail_modules() == [m.final_layer]
    assert m.single_blocks[0].norm.query_norm.scale.dtype == torch.bfloat16  # cast; the norm runs in fp32 anyway


def test_model_split_hand_off_matches():
    from rulztrainer.multigpu import split_model

    ours, _ = pair(3)
    x, enc = torch.randn(1, 16, 2, 2), torch.randn(1, 4, 24)
    with torch.no_grad():
        want = ours(x, 300.0, enc, None)
        split_model(ours, ["cpu", "cpu"])  # same device twice: exercises the hand-off path
        assert torch.equal(ours(x, 300.0, enc, None), want)


def real_keys() -> dict:
    return json.loads((FIXTURES / "flux3_klein_keys.json").read_text(encoding="utf-8"))


def test_real_checkpoint_layout_builds_the_model_exactly():
    """The real Klein Base file's 357 keys and shapes (its safetensors header) give the checkpoint's config, and the
    port's state dict is exactly those keys minus the reference-image mode."""
    keys = real_keys()
    state = {k: torch.empty(v, device="meta", dtype=torch.bfloat16) for k, v in keys["base_bf16"].items()}
    assert sum(t.numel() for t in state.values()) == 5_538_458_368  # the package manifest's params
    layout = FSF.to_model_layout(dict(state))
    config = FSF.transformer_config_from_state(layout)
    assert config == dict(in_channels=128, num_layers=5, num_single_layers=28, attention_head_dim=128,
                          num_attention_heads=24, joint_attention_dim=20480, timestep_channels=256, mlp_ratio=3.0,
                          axes_dims_rope=[32, 32, 32, 32], rope_theta=10000, eps=1e-6, flow_map=False,
                          image_rope_scale=0.5)
    with torch.device("meta"):
        model = Flux3KleinTransformer.from_config(config)
    ours = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert ours == {k: tuple(v.shape) for k, v in layout.items()}
    dropped = set(state) - set(layout)
    assert dropped and all("image_cond" in k for k in dropped)
    assert sum(t.numel() for t in layout.values()) == 4_868_023_808  # 9.74 GB in bf16
    assert FSF.FLOW_MAP_MARKER in keys["distilled_extra"] and len(keys["distilled_extra"]) == 2


def test_diffusers_layout_is_the_inverse_of_diffusers_converter():
    """BFL's real keys through diffusers' own single-file converter and back through ours: the same names (and
    norm_out's swapped halves swapped back)."""
    from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers

    keys = real_keys()
    torch.manual_seed(0)
    small = {k: torch.randn(4, 2) if k.endswith("adaLN_modulation.1.weight") else torch.zeros(1)
             for k in keys["base_bf16"]}
    small["ref_time_in.in_layer.weight"] = torch.zeros(1)
    small["ref_time_in.out_layer.weight"] = torch.zeros(1)
    as_diffusers = convert_flux3_transformer_checkpoint_to_diffusers(dict(small))
    with torch.device("meta"):
        ref = diffusers.Flux3Transformer2DModel(flow_map=True)
    assert set(as_diffusers) == set(ref.state_dict())
    back = FSF.diffusers_to_bfl(as_diffusers)
    assert set(back) == set(small)
    key = "final_layer.image.adaLN_modulation.1.weight"
    assert torch.equal(back[key], small[key]) and not torch.equal(as_diffusers["norm_out.linear.weight"], small[key])
    for name in ("single_blocks.3.q_proj", "content_mode_blocks.image.1.mlp_in", "txt_in", "time_in.out_layer",
                 "early_stream_modulations.txt.lin", "final_layer.image.linear", "emb_in.image"):
        assert FSF.diffusers_module_to_bfl(FSF.bfl_module_to_diffusers(name)) == name


def test_single_files_in_bfl_and_comfy_int8_layout_load_and_are_detected(tmp_path):
    from safetensors.torch import save_file

    ours, ref = pair(4)
    bfl = FSF.diffusers_to_bfl(dict(ref.state_dict()))  # with the reference-image mode, as the files hold it
    assert any("image_cond" in k for k in bfl)
    fused = {}
    for k, v in bfl.items():  # ComfyUI's int8 files: q / k / v fused per block, attention config entries
        if k.endswith(".q_proj.weight"):
            stem = k[: -len("q_proj.weight")]
            fused[f"model.diffusion_model.{stem}qkv.weight"] = torch.cat(
                [v, bfl[stem + "k_proj.weight"], bfl[stem + "v_proj.weight"]])
            fused[f"model.diffusion_model.{stem}comfy_attention.config"] = torch.zeros(3, dtype=torch.uint8)
        elif not k.endswith((".k_proj.weight", ".v_proj.weight")):
            fused[f"model.diffusion_model.{k}"] = v
    x, enc = torch.randn(1, 16, 2, 4), torch.randn(1, 6, 24)
    for name, state in (("flux3-image-klein-base-bf16.safetensors", bfl),
                        ("flux3-image-klein-base-int8-convrot.safetensors", fused)):
        path = tmp_path / name
        save_file({k: v.contiguous() for k, v in state.items()}, str(path))
        loaded = FSF.load_transformer_file(path, "cpu", quant="none", dtype=torch.float32)
        assert not loaded.flow_map and loaded.config["num_single_layers"] == 3
        with torch.no_grad():
            assert torch.allclose(loaded(x, 600.0, enc, None), ours(x, 600.0, enc, None), atol=1e-6)
        assert FSF.is_flux3_file(path) and archs.arch_from_transformer_file(str(path)) == "flux3_klein"
        assert TrainConfig(model_source="files", transformer_file=str(path)).arch_name() == "flux3_klein"
    # ComfyUI's real int8 files fuse exactly these: every block's q / k / v (43 blocks), the rest stays apart
    real = real_keys()["int8_convrot_weights"]
    qkv = [k for k in real if k.endswith(".qkv.weight")]
    assert len(qkv) == 43 and len(real) == 4 * 43
    split = FSF.unfuse_qkv({k: torch.zeros(3, 1) for k in qkv})
    assert set(split) <= set(real_keys()["base_bf16"])


def test_quantized_qkv_splits_by_rows():
    w = torch.randn(3 * 1024, 1024)
    for fmt in ("int8", "int8_convrot", "nvfp4"):
        qw = Q.quantize_weight(w, fmt)
        state = FSF.unfuse_qkv({"single_blocks.0.qkv.weight": qw})
        full = qw.dequantize(torch.float32)
        tol = 0.0 if fmt != "nvfp4" else 0.2
        for i, n in enumerate(("q_proj", "k_proj", "v_proj")):
            part = state[f"single_blocks.0.{n}.weight"].dequantize(torch.float32)
            assert torch.allclose(part, full[i * 1024:(i + 1) * 1024], atol=tol + 1e-6)


def test_diffusers_folder_loads_and_is_detected(tmp_path):
    ours, ref = pair(5)
    ref.save_pretrained(str(tmp_path / "klein" / "transformer"))
    (tmp_path / "klein" / "config.json").write_text(json.dumps({"dit": {"image_rope_scale": 0.5}}))
    loaded = FL.load_transformer(tmp_path / "klein", "cpu", quant="none", dtype=torch.float32)
    x, enc = torch.randn(1, 16, 4, 2), torch.randn(1, 5, 24)
    with torch.no_grad():
        assert torch.allclose(loaded(x, 450.0, enc, None), ref_call(ref, x, torch.tensor([0.45]), enc), atol=1e-5)
    assert loaded.single_blocks[0].norm.query_norm.scale.dtype == torch.float32
    cfg = TrainConfig(model_source="hf", model_path=str(tmp_path / "klein"))
    assert archs.detect_arch(cfg, tmp_path / "klein").name == "flux3_klein"


def test_lora_names_folded_export_for_comfyui_and_diffusers(tmp_path):
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_flux3_lora_to_diffusers
    from safetensors.torch import load_file

    klein = archs.arch_by_name("flux3_klein")
    ours, ref = pair(6)
    everything = klein.lora_module_names(ours, list(klein.target_groups), None)
    linears = [n for n, mod in ours.named_modules() if isinstance(mod, torch.nn.Linear)]
    assert sorted(everything) == sorted(linears)  # all groups together = every Linear
    default = klein.lora_module_names(ours, klein.default_targets, None)
    assert len(default) == 6 * (3 + 2 + 2)  # q k v o, mlp in / out in the joint and both kinds of per-mode block
    assert "txt_mode_blocks.1.mlp_in" in default and "content_mode_blocks.image.0.attn_out" in default
    assert klein.lora_module_names(ours, ["q"], [0, 2]) == ["single_blocks.0.q_proj", "single_blocks.2.q_proj"]
    assert set(klein.lora_module_names(ours, ["q"], [0], include_text_embedder=True)) == {"txt_in",
                                                                                           "single_blocks.0.q_proj"}
    assert klein.param_group_of("single_blocks.0.mlp_in") == "mlp" and klein.param_group_of("txt_in") == "text"
    assert klein.param_group_of("single_blocks.0.attn_out") == "attn"
    assert klein.param_group_of("txt_mode_blocks.0.q_proj") == "text"
    names = klein.lora_module_names(ours, [*klein.default_targets, "modulation", "embed", "text_in"], None)
    inj = L.inject_lora(ours, 4, 16.0, ["q"], names=names)  # alpha / rank = 4: folded into lora_B on export
    with torch.no_grad():
        for mod in inj.values():
            mod.lora_up.weight.normal_(0, 0.05)
    path = tmp_path / "klein_lora.safetensors"
    L.save_lora(inj, path, dtype=torch.float32, fmt=klein.lora_format)
    state = load_file(str(path))
    assert not any(k.endswith(".alpha") for k in state) and all(k.startswith("diffusion_model.") for k in state)
    # ComfyUI: "diffusion_model.<key without .weight>" of the model's own state dict, which is BFL's file layout
    real = set(real_keys()["base_bf16"])
    stems = {k.rsplit(".lora_", 1)[0][len("diffusion_model."):] for k in state}
    assert all(f"{s}.weight" in real for s in stems)
    # diffusers: Flux3LoraLoaderMixin converts diffusion_model.* keys; every key must say "lora" (no .alpha)
    conv = _convert_non_diffusers_flux3_lora_to_diffusers(state)
    assert all("lora" in k and k.startswith("transformer.") for k in conv)
    ref_names = {n for n, _ in ref.named_modules()}
    merged = {}
    for k, v in conv.items():
        stem, part = k.removeprefix("transformer.").rsplit(".lora_", 1)
        merged.setdefault(stem, {})[part.split(".")[0]] = v
    # diffusers' converter renames the block modules (the top-level ones, txt_in / emb_in / ..., map in ComfyUI)
    blocks = ("single_transformer_blocks", "mode_blocks", "txt_mode_blocks")
    block_stems = [s for s in merged if s.split(".")[0] in blocks]
    assert block_stems and all(s in ref_names for s in block_stems)
    x, enc = torch.randn(1, 16, 2, 4), torch.randn(1, 5, 24)
    with torch.no_grad():
        for s in block_stems:  # PEFT at scale 1 (no alpha): delta = B @ A
            ref.get_submodule(s).weight += merged[s]["B"] @ merged[s]["A"]
        for name in names:  # the rest by our own name map
            if name.split(".")[0] in ("single_blocks", "content_mode_blocks", "txt_mode_blocks"):
                continue
            a, b = state[f"diffusion_model.{name}.lora_A.weight"], state[f"diffusion_model.{name}.lora_B.weight"]
            delta = b @ a
            if name == "final_layer.image.adaLN_modulation.1":  # diffusers stores [scale, shift]
                delta = torch.cat(delta.chunk(2)[::-1])
            ref.get_submodule(FSF.bfl_module_to_diffusers(name)).weight += delta
        assert torch.allclose(ours(x, 400.0, enc, None), ref_call(ref, x, torch.tensor([0.4]), enc), atol=1e-4)
    # back through the trainer's loader (resume): the unscaled factors and the trained alpha
    fresh, _ = pair(6)
    inj2 = L.inject_lora(fresh, 4, 4.0, ["q"], names=names)
    assert L.load_lora_weights(inj2, path) == len(names)
    assert all(m.alpha == 16.0 and m.scale == 4.0 for m in inj2.values())
    with torch.no_grad():
        assert torch.allclose(fresh(x, 400.0, enc, None), ours(x, 400.0, enc, None), atol=1e-5)


def test_diffusers_named_lora_attaches_for_sampling(tmp_path):
    from safetensors.torch import save_file

    klein = archs.arch_by_name("flux3_klein")
    ours, ref = pair(7)
    a, b = torch.randn(2, 32) * 0.1, torch.randn(32, 2) * 0.1
    save_file({"transformer.single_transformer_blocks.1.attn.to_k.lora_A.weight": a,
               "transformer.single_transformer_blocks.1.attn.to_k.lora_B.weight": b}, str(tmp_path / "d.safetensors"))
    assert L.attach_extra_lora(ours, str(tmp_path / "d.safetensors"), 1.0, rename=klein.lora_module_name) == 1
    L.set_extras_enabled(ours, True)
    with torch.no_grad():
        ref.single_transformer_blocks[1].attn.to_k.weight += b @ a
        x, enc = torch.randn(1, 16, 2, 2), torch.randn(1, 3, 24)
        assert torch.allclose(ours(x, 800.0, enc, None), ref_call(ref, x, torch.tensor([0.8]), enc), atol=1e-4)


def test_sampler_schedule_matches_the_scheduler_and_samples():
    for steps in (50, 16, 7):
        sched = diffusers.FlowMatchEulerDiscreteScheduler(shift=SHIFT)
        sched.set_timesteps(sigmas=torch.linspace(1.0, 1 / steps, steps).numpy())
        assert torch.allclose(flux3_sigmas(steps), sched.sigmas.float(), atol=1e-6)
    klein = archs.arch_by_name("flux3_klein")
    assert torch.equal(klein.sample_sigmas(None, 16, 512, 768, 5.0), flux3_sigmas(16))
    assert (klein.default_sample_steps, klein.default_sample_cfg) == (50, 3.5)
    r = klein.reference_sigmas(10000, torch.Generator().manual_seed(0), 4096)
    assert r.min() >= 0 and r.max() <= 1 and 0.75 < r.mean() < 0.9
    emb, mask = FL.pad_embeds([torch.randn(5, 24), torch.randn(3, 24)])
    assert emb.shape == (2, 5, 24) and mask.sum(1).tolist() == [5, 3]
    for flow_map in (False, True):
        model, _ = pair(8, flow_map=flow_map)
        model = model.to(torch.bfloat16)
        lat = klein.sample(model, emb[:1], mask[:1], emb[1:, :3], mask[1:, :3], width=64, height=32, steps=3,
                           cfg=3.5, seed=1, device="cpu")
        assert lat.shape == (1, 16, 2, 4) and torch.isfinite(lat).all()
    assert klein.distilled_sample_model("black-forest-labs/flux-3-image-klein")
    assert klein.distilled_sample_model("C:/m/flux3-image-klein-int8-convrot.safetensors")
    assert not klein.distilled_sample_model("black-forest-labs/flux-3-image-klein-base")
    assert not klein.distilled_sample_model("")


def test_auto_quant_counts_the_gpus_the_model_runs_on(monkeypatch):
    sizes = {0: 15.9, 1: 12, 2: 8}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(FL, "_gpu_gb", lambda d: sizes[torch.device(d).index or 0])
    assert FL.resolve_quant("auto", "cuda:0") == "none"  # a 16 GB card holds Klein in bf16
    assert FL.resolve_quant("auto", "cuda:1") == "int8_convrot" and FL.resolve_quant("auto", "cuda:2") == "nvfp4"
    assert FL.resolve_quant("auto", ["cuda:1", "cuda:2"]) == "none"
    assert FL.resolve_quant("auto", "cpu") == "nvfp4"  # a sampling model staged on the CPU: the smallest GPU
    assert FL.resolve_quant("nvfp4", "cuda:0") == "nvfp4"


def test_training_refuses_the_distilled_model_and_gated_downloads_say_why(monkeypatch):
    from huggingface_hub.errors import GatedRepoError

    from rulztrainer import comfy_files as CF
    import rulztrainer.downloads as DL

    model, _ = pair(9, flow_map=True)
    with pytest.raises(ValueError, match="flux-3-image-klein-base"):
        FL._trainable(model, "flux3-image-klein-bf16.safetensors")
    err = FL.gated_error("black-forest-labs/flux-3-image-klein-comfy", RuntimeError("401"))
    assert isinstance(err, PermissionError) and "huggingface.co/black-forest-labs/flux-3-image-klein-comfy" in str(err)
    assert "FLUX Kommunity License" in str(err)

    from types import SimpleNamespace

    class Forbidden(Exception):  # what an HTTP error from the hub carries: its response's status
        response = SimpleNamespace(status_code=403)

    def gated(repo, file, label, progress=None):
        raise Forbidden("403 Client Error")

    monkeypatch.setattr(DL, "ensure_hf_file", gated)
    with pytest.raises(PermissionError, match="accept the license"):
        CF.resolve_downloads(TrainConfig(model_path="black-forest-labs/flux-3-image-klein-base"))
    assert FL.is_gated_error(GatedRepoError.__new__(GatedRepoError)) and not FL.is_gated_error(ValueError("x"))


def test_detection_caches_buckets_and_downloads(tmp_path):
    from rulztrainer.comfy_files import planned_downloads
    from rulztrainer.config import ARCH_SAMPLE_DEFAULTS, ARCH_TURBO

    for path in ("black-forest-labs/flux-3-image-klein-base", "black-forest-labs/flux-3-image-klein",
                 "C:/m/flux3-image-klein-base-int8-convrot.safetensors"):
        assert TrainConfig(model_path=path).arch_name() == "flux3_klein"
    assert TrainConfig(model_path="Comfy-Org/flux2-klein").arch_name() != "flux3_klein"
    mdir = tmp_path / "model" / "transformer"
    mdir.mkdir(parents=True)
    (mdir / "config.json").write_text(json.dumps({"_class_name": "Flux3Transformer2DModel"}))
    assert archs.detect_arch(TrainConfig(model_path=str(tmp_path / "model")), tmp_path / "model").name == "flux3_klein"
    data = tmp_path / "data"
    data.mkdir()
    from PIL import Image

    Image.new("RGB", (900, 600)).save(data / "a.png")
    klein = TrainConfig(model_path="black-forest-labs/flux-3-image-klein-base", dataset_dir=str(data),
                        resolutions=[1.0])
    fibo = TrainConfig(model_path="briaai/Fibo-2.0-dev", dataset_dir=str(data), resolutions=[1.0])
    item, fitem = D.scan_dataset(klein)[0], D.scan_dataset(fibo)[0]
    assert item.width % 16 == 0 and item.height % 16 == 0
    assert item.latent_path == fitem.latent_path  # the same Flux2 VAE latents as FIBO 2.0 / MOSS-Image
    assert D.text_cache_path(klein, "p").name.endswith("_flux3_klein.safetensors")
    repo = "black-forest-labs/flux-3-image-klein-comfy"
    plan = {f: (r, n) for f, r, n in planned_downloads(TrainConfig(
        model_path="black-forest-labs/flux-3-image-klein-base"))}
    assert plan == {"transformer_file": (repo, "diffusion_models/flux3-image-klein-base-int8-convrot.safetensors"),
                    "text_encoder_file": (repo, "text_encoders/qwen3vl_4b_int8_convrot.safetensors"),
                    "vae_file": (repo, "vae/flux2-vae.safetensors")}  # no turbo LoRA, no training adapter
    for quant in ("nvfp4", "bf16"):
        files = {f: n for f, _, n in planned_downloads(TrainConfig(
            model_path="black-forest-labs/flux-3-image-klein", model_file_quant=quant, text_encoder_file_quant="bf16"))}
        assert files["transformer_file"] == f"diffusion_models/flux3-image-klein-{quant}.safetensors"
        assert files["text_encoder_file"] == "text_encoders/qwen3vl_4b_bf16.safetensors"
    assert "flux3_klein" not in ARCH_TURBO and "flux3_klein" not in ARCH_SAMPLE_DEFAULTS


def _qwen3vl_tokenizer() -> str | None:
    import glob
    import os

    cands = [os.environ.get("RULZ_QWEN3VL_TOKENIZER", ""), "models/tokenizer__Qwen3-VL-4B"]
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser(
        "~/.cache/huggingface")), "hub")
    cands += glob.glob(os.path.join(root, "models--Qwen--Qwen3-VL-4B-Instruct", "snapshots", "*"))
    return next((c for c in cands if c and os.path.isfile(os.path.join(c, "tokenizer_config.json"))), None)


@pytest.mark.skipif(_qwen3vl_tokenizer() is None, reason="no Qwen3-VL-4B tokenizer (models/tokenizer__Qwen3-VL-4B, "
                    "the HF cache or RULZ_QWEN3VL_TOKENIZER)")
def test_prompt_template_window_and_token_positions():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_qwen3vl_tokenizer())
    for prompt in ("a red fox in snow, shtm style", "", "x " * 300):
        want = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
        assert FL.chat_text(prompt) == want
        ids, mask, pos = FL.tokenize(tok, prompt, 1024)
        n = int(mask.sum())
        assert ids.shape[1] % 80 == 0 and 0 <= ids.shape[1] - n < 80
        assert (ids[0, n:] == tok.pad_token_id).all() and pos[0, :n].tolist() == list(range(n))
        assert (pos[0, n:] == 1).all()
        text, offsets, drop, length = FL.prompt_tokens(tok, prompt)
        assert text == want and drop == 0 and length == ids.shape[1] and len(offsets) == n
    _, mask, _ = FL.tokenize(tok, "word " * 2000, 1024)
    assert mask.shape[1] == 1024
