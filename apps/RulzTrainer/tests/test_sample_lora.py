"""MOSS-Image Base samples with Kijai's SFT LoRA (sample_lora auto), its fused proj.qkv on the model's q / k / v; another
model's sampling files (Krea 2's krea/Krea-2-Turbo preview model kept after switching to MOSS-Image) refused
instead of sampling on Krea 2 Turbo without MOSS-Image's turbo LoRA."""
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import comfy_files as CF  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer.archs import arch_by_name  # noqa: E402
from rulztrainer.config import ARCH_SAMPLE_LORA, ARCH_TURBO, TrainConfig, family_named_in  # noqa: E402

SFT = ("Kijai/MOSS-image_experimental", ARCH_SAMPLE_LORA["moss_image"]["file"])


def _plan(**kw):
    return {f: (r, p) for f, r, p in CF.planned_downloads(TrainConfig(**kw))}


def test_sft_lora_only_while_samples_render_moss_base():
    assert TrainConfig().sample_lora == "auto"
    assert _plan()["sample_lora"] == SFT  # the default: MOSS-Image Base
    assert _plan(transformer_file="models/bucket/diffusion_models/moss_image_base_int8_convrot.safetensors",
                 turbo_lora="")["sample_lora"] == SFT
    for kw in ({"model_path": "OpenMOSS-Team/MOSS-Image-SFT"},  # SFT already looks like SFT
               {"sample_model_path": "OpenMOSS-Team/MOSS-Image-SFT"},  # samples render another model
               {"model_path": "krea/Krea-2-Raw"}, {"model_path": "Qwen/Qwen-Image-2.1"},  # no sampling LoRA there
               {"sample_lora": ""}, {"sample_lora": "mine.safetensors"}, {"finetune": True}):
        assert "sample_lora" not in _plan(**kw), kw


def test_resolve_sets_or_clears_auto_sample_lora(monkeypatch, tmp_path):
    import rulztrainer.downloads as D

    monkeypatch.setattr(D, "ensure_hf_file", lambda repo, file, label, progress=None: tmp_path / Path(file).name)
    base = CF.resolve_downloads(TrainConfig(model_source="hf"))
    assert base.sample_lora == str(tmp_path / Path(SFT[1]).name)
    assert base.turbo_lora == str(tmp_path / ARCH_TURBO["moss_image"]["file"])
    assert CF.resolve_downloads(TrainConfig(model_source="hf", model_path="OpenMOSS-Team/MOSS-Image-SFT")) \
        .sample_lora == ""


def test_krea_leftovers_are_refused_on_a_moss_run():
    """The reported run: Krea 2 first (the wizard's "Preview with Krea 2 Turbo" set sample_model_path), then MOSS-Image
    Base. The Turbo name made turbo_lora auto pick none, and samples would load Krea 2 Turbo."""
    leftover = TrainConfig(model_path="OpenMOSS-Team/MOSS-Image-Base", sample_model_path="krea/Krea-2-Turbo")
    assert leftover.samples_on_distilled_model()  # why the MOSS-Image turbo LoRA was not downloaded
    assert leftover.foreign_sampling_files() == ["sample_model_path: krea/Krea-2-Turbo (a krea2 file)"]
    with pytest.raises(ValueError, match="krea/Krea-2-Turbo"):
        leftover.validate()
    krea_turbo = "/root/.cache/huggingface/hub/models--Comfy-Org--Krea-2/snapshots/x/loras/krea2_turbo_lora_rank_64_bf16" \
                 ".safetensors"
    assert TrainConfig(turbo_lora=krea_turbo).foreign_sampling_files()
    # the model's own files, auto, none, and a folder name that is not the file's own pass
    for kw in ({"sample_model_path": "OpenMOSS-Team/MOSS-Image-SFT"}, {"turbo_lora": "auto", "sample_lora": ""},
               {"turbo_lora": "/data/krea_stuff/8-steps-dmd-lora.safetensors"},
               {"model_path": "krea/Krea-2-Raw", "sample_model_path": "krea/Krea-2-Turbo"}):
        assert TrainConfig(**kw).foreign_sampling_files() == [], kw
    assert family_named_in("lodestones/Kromoss") == "moss_image" and family_named_in("my_style.safetensors") is None


def test_webui_start_refuses_foreign_sampling_files(tmp_path):
    import webui

    data = tmp_path / "data"
    data.mkdir()
    msg = webui._start_training(TrainConfig(dataset_dir=str(data), sample_model_path="krea/Krea-2-Turbo"))
    assert msg.startswith("Not started") and "krea/Krea-2-Turbo" in msg


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.proj = nn.ModuleDict({n: nn.Linear(dim, dim, bias=False) for n in ("q", "k", "v", "o")})


class _Moss(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim)])


def test_sft_lora_attaches_to_moss_layout(tmp_path):
    """Kijai's SFT LoRA is in ComfyUI's layout (one fused layers.N.proj.qkv, norm .diff offsets): the q / k / v rows
    of its up matrix land on the model's separate projections."""
    torch.manual_seed(0)
    dim, r = 4, 2
    down, up = torch.randn(r, dim), torch.randn(3 * dim, r)
    save_file({"diffusion_model.layers.0.proj.qkv.lora_down.weight": down,
               "diffusion_model.layers.0.proj.qkv.lora_up.weight": up,
               "diffusion_model.layers.0.proj.o.lora_down.weight": torch.randn(r, dim),
               "diffusion_model.layers.0.proj.o.lora_up.weight": torch.randn(dim, r),
               "diffusion_model.layers.0.q_norm.diff": torch.randn(dim)}, str(tmp_path / "sft.safetensors"))
    m = _Moss()
    assert L.attach_extra_lora(m, str(tmp_path / "sft.safetensors"), 1.0) == 5  # q, k, v, o and the norm
    _, d, u, _ = m.layers[0].proj["v"].extras[0]
    assert torch.allclose(u @ d, (up @ down)[2 * dim:])


def test_a_sampling_lora_matching_nothing_stops_the_run(tmp_path):
    from rulztrainer.trainer import Trainer

    save_file({"diffusion_model.blocks.0.attn.wq.lora_down.weight": torch.randn(2, 4),
               "diffusion_model.blocks.0.attn.wq.lora_up.weight": torch.randn(4, 2)}, str(tmp_path / "k.safetensors"))
    tr = Trainer.__new__(Trainer)
    tr.cfg, tr.arch = TrainConfig(turbo_lora=str(tmp_path / "k.safetensors")), arch_by_name("moss_image")
    with pytest.raises(ValueError, match="matches no layer"):
        tr._attach_sampling_loras(_Moss())
    tr.cfg = TrainConfig(turbo_lora="auto", sample_lora="auto")  # unresolved auto: nothing to attach
    tr._attach_sampling_loras(_Moss())
