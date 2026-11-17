"""Mixed datasets on a video model (3.32.0): still_lora_targets / still_lora_blocks. A batch of stills trains only
those LoRA modules; the rest still acts in its forward but takes no gradient from it (lora.gradless). Plus the Train
tab's "Training data" presets (config.video_data_preset)."""
import contextlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rulztrainer import lora as lora_mod
from rulztrainer.archs import arch_by_name
from rulztrainer.capabilities import violations
from rulztrainer.config import (STILL_FFN_TARGETS, TrainConfig, detect_video_data_preset, video_data_preset)
from rulztrainer.trainer import Trainer


def _pair(seed=0):
    """Two LoRA-wrapped linears in a chain (x -> a -> b), each with a non-zero up so both act in the forward."""
    torch.manual_seed(seed)
    a = lora_mod.LoRALinear(nn.Linear(6, 6), rank=2, alpha=2.0)
    b = lora_mod.LoRALinear(nn.Linear(6, 6), rank=2, alpha=2.0)
    for m in (a, b):
        with torch.no_grad():
            m.lora_up.weight.normal_()
    return a, b


def _loss(a, b, x):
    return b(a(x)).pow(2).sum()


def test_gradless_skips_only_the_masked_weights():
    a, b = _pair()
    x = torch.randn(3, 6, requires_grad=True)
    with lora_mod.gradless([a]):
        _loss(a, b, x).backward()
    assert all(p.grad is None for p in a.adapter_params().values())  # None, not zero: no optimizer moves it
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in b.adapter_params().values())
    assert all(p.requires_grad for m in (a, b) for p in m.adapter_params().values())  # restored


def test_gradless_keeps_the_forward_and_the_gradient_through_it():
    a, b = _pair()
    x = torch.randn(3, 6)
    x1 = x.clone().requires_grad_(True)
    loss1 = _loss(a, b, x1)
    loss1.backward()
    grad_b = {k: p.grad.clone() for k, p in b.adapter_params().items()}
    for m in (a, b):
        m.zero_grad(set_to_none=True)
    x2 = x.clone().requires_grad_(True)
    with lora_mod.gradless([a]):
        loss2 = _loss(a, b, x2)
        loss2.backward()
    assert torch.allclose(loss1, loss2)  # the masked LoRA still acts in the forward
    assert torch.allclose(x1.grad, x2.grad)  # and the gradient still flows through it
    for k, p in b.adapter_params().items():
        assert torch.allclose(p.grad, grad_b[k])


def test_gradless_leaves_a_clip_gradient_of_the_same_step():
    a, b = _pair()
    _loss(a, b, torch.randn(3, 6)).backward()  # a clip's batch: everything trains
    clip_a = {k: p.grad.clone() for k, p in a.adapter_params().items()}
    clip_b = {k: p.grad.clone() for k, p in b.adapter_params().items()}
    with lora_mod.gradless([a]):  # then a batch of stills (gradient accumulation)
        _loss(a, b, torch.randn(3, 6)).backward()
    for k, p in a.adapter_params().items():
        assert torch.equal(p.grad, clip_a[k])
    assert any(not torch.equal(p.grad, clip_b[k]) for k, p in b.adapter_params().items())


def test_gradless_restores_on_error():
    a, _ = _pair()
    with pytest.raises(RuntimeError):
        with lora_mod.gradless([a]):
            raise RuntimeError("out of memory")
    assert all(p.requires_grad for p in a.adapter_params().values())


def test_gradless_under_checkpointing():
    """A checkpointed block recomputes in the backward: inside the block the masked weights still get nothing."""
    from torch.utils.checkpoint import checkpoint

    a, b = _pair()
    x = torch.randn(3, 6, requires_grad=True)
    with lora_mod.gradless([a]):
        checkpoint(lambda t: b(a(t)), x, use_reentrant=False).pow(2).sum().backward()
    assert all(p.grad is None for p in a.adapter_params().values())
    assert all(p.grad is not None for p in b.adapter_params().values())


def test_adamw_leaves_a_masked_weight_alone():
    a, b = _pair()
    params = [p for m in (a, b) for p in m.adapter_params().values()]
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.1)
    _loss(a, b, torch.randn(3, 6)).backward()
    opt.step()  # a clip step: momentum on every weight
    opt.zero_grad(set_to_none=True)
    before = {k: p.detach().clone() for k, p in a.adapter_params().items()}
    with lora_mod.gradless([a]):
        _loss(a, b, torch.randn(3, 6)).backward()
    opt.step()  # a step of stills alone
    for k, p in a.adapter_params().items():
        assert torch.equal(p, before[k])


# ---- the trainer


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj, self.attn.out_proj = nn.Linear(4, 12), nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.fc1, self.mlp.fc2 = nn.Linear(4, 8), nn.Linear(8, 4)


class _H3Like(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(n))
        self.layers = self.blocks


def _shell(arch="minimax_h3"):
    tr = Trainer.__new__(Trainer)
    tr.arch, tr.race_rows = arch_by_name(arch), None
    return tr


def _layout(cfg, clips, n=4):
    """The trainer's steps for a LoRA on an H3-shaped model: _still_layout, then _still_keep before the LoRA is
    made and _still_gradless after. Returns (trainer, injected)."""
    tr = _shell()
    tr._still_layout(cfg, clips)
    model = _H3Like(n)
    blocks = lora_mod.parse_blocks(cfg.lora_blocks, n)
    names = tr.arch.lora_module_names(model, cfg.lora_targets, blocks)
    keep = tr._still_keep(model, cfg, clips)
    injected = lora_mod.inject_lora(model, 2, 2.0, cfg.lora_targets, blocks=blocks, names=names)
    tr._still_gradless(injected, keep, cfg, clips)
    return tr, injected


def _mixed_cfg(**kw):
    kw = {"still_lora_targets": list(STILL_FFN_TARGETS), **kw}
    return TrainConfig(model_path="MiniMaxAI/MiniMax-H3", train_frames=39,
                       lora_targets=list(arch_by_name("minimax_h3").default_targets), **kw)


def test_mixed_run_masks_attention_on_stills():
    tr, inj = _layout(_mixed_cfg(), clips=[object()])
    assert tr.still_gradless_names and all(".attn." in nm for nm in tr.still_gradless_names)
    assert len(tr.still_gradless_names) == 4 * 2  # qkv_proj and out_proj of each of 4 blocks
    assert set(inj) - set(tr.still_gradless_names) == {f"blocks.{i}.mlp.fc{j}" for i in range(4) for j in (1, 2)}


def test_mixed_run_block_masking():
    tr, inj = _layout(_mixed_cfg(still_lora_targets=[], still_lora_blocks="2-3"), clips=[object()])
    assert set(tr.still_gradless_names) == {nm for nm in inj if nm.split(".")[1] in ("0", "1")}


def test_still_mask_by_batch():
    tr, inj = _layout(_mixed_cfg(), clips=[object()])
    stills, clip = [SimpleNamespace(frames=1)] * 2, [SimpleNamespace(frames=39)]
    assert isinstance(tr._still_mask(inj, clip), contextlib.nullcontext)
    with tr._still_mask(inj, stills):
        assert not inj["blocks.0.attn.qkv_proj"].lora_down.weight.requires_grad
        assert inj["blocks.0.mlp.fc1"].lora_down.weight.requires_grad
    assert inj["blocks.0.attn.qkv_proj"].lora_down.weight.requires_grad


def test_no_still_settings_masks_nothing():
    cfg = TrainConfig(model_path="MiniMaxAI/MiniMax-H3", train_frames=39,
                      lora_targets=list(arch_by_name("minimax_h3").default_targets))
    tr, inj = _layout(cfg, clips=[object()])
    assert tr.still_gradless_names == []
    assert isinstance(tr._still_mask(inj, [SimpleNamespace(frames=1)]), contextlib.nullcontext)


def test_stills_only_run_makes_the_still_layers_the_lora():
    cfg = _mixed_cfg(still_lora_blocks="1-3")
    tr, inj = _layout(cfg, clips=[])
    assert cfg.lora_targets == list(STILL_FFN_TARGETS) and cfg.lora_blocks == "1-3"
    assert set(inj) == {f"blocks.{i}.mlp.fc{j}" for i in (1, 2, 3) for j in (1, 2)}
    assert tr.still_gradless_names == []  # nothing to mask: no attention layer was made


def test_h3_still_blocks_left_alone_with_still_settings():
    cfg = _mixed_cfg()
    _shell()._h3_still_blocks(cfg, [])
    assert cfg.lora_blocks == ""  # the stills' layers are chosen: no automatic 20-49


def test_nothing_left_to_train_is_refused():
    with pytest.raises(ValueError, match="nothing to train"):
        _layout(_mixed_cfg(still_lora_targets=[], still_lora_blocks="10-20"), clips=[object()])


def test_unknown_still_target_is_refused():
    with pytest.raises(ValueError, match="do not exist"):
        _shell()._still_layout(_mixed_cfg(still_lora_targets=["ffn_gate"]), [object()])


# ---- config


def test_validate_still_settings():
    _mixed_cfg().validate_still_masking()
    with pytest.raises(ValueError, match="not among the LoRA's layers"):
        TrainConfig(model_path="MiniMaxAI/MiniMax-H3", lora_targets=["q", "k", "v"],
                    still_lora_targets=["ffn_down"]).validate_still_masking()
    with pytest.raises(ValueError, match="full fine-tune"):
        _mixed_cfg(finetune=True).validate_still_masking()
    with pytest.raises(ValueError, match="race"):
        _mixed_cfg(race=True).validate_still_masking()
    with pytest.raises(ValueError, match="list"):
        _mixed_cfg(still_lora_targets="ffn_down").validate_still_masking()


def test_still_settings_survive_a_round_trip():
    cfg = _mixed_cfg(still_lora_blocks="20-49")
    back = TrainConfig.from_dict(cfg.to_dict())
    assert back.still_lora_targets == list(STILL_FFN_TARGETS) and back.still_lora_blocks == "20-49"


def test_capabilities_only_video_models_read_them():
    assert not violations("minimax_h3", {"still_lora_targets": ["ffn_down"], "still_lora_blocks": "20-49"})
    assert not violations("krea2", {"still_lora_targets": ["ffn_down"]})  # ignored, not refused


@pytest.mark.parametrize("family", ["minimax_h3", "pan2"])
def test_video_data_presets(family):
    photos = video_data_preset("photos", family)
    assert photos["train_frames"] == 1 and photos["lora_targets"] == list(STILL_FFN_TARGETS)
    assert photos["lora_blocks"] == "all"  # not "": H3 would train blocks 20-49 on photos
    mixed = video_data_preset("mixed", family)
    assert mixed["train_frames"] > 1 and mixed["still_lora_targets"] == list(STILL_FFN_TARGETS)
    assert set(mixed["still_lora_targets"]) <= set(mixed["lora_targets"])
    video = video_data_preset("video", family, train_frames=56)
    assert video["train_frames"] == 56 and video["still_lora_targets"] == []  # the user's own frame count stays
    for name in ("video", "photos", "mixed"):
        assert detect_video_data_preset(family, video_data_preset(name, family)) == name


def test_preset_frames_fit_each_vae():
    assert (video_data_preset("video", "minimax_h3")["train_frames"] - 5) % 17 == 0
    assert (video_data_preset("video", "pan2")["train_frames"] - 1) % 4 == 0


def test_detect_preset_none():
    assert detect_video_data_preset("minimax_h3", {"train_frames": 1}) == ""  # H3's own 20-49 for photos
    assert detect_video_data_preset("krea2", video_data_preset("photos", "minimax_h3")) == ""
    custom = {**video_data_preset("mixed", "minimax_h3"), "still_lora_blocks": "20-49"}
    assert detect_video_data_preset("minimax_h3", custom) == ""
    with pytest.raises(ValueError):
        video_data_preset("audio", "minimax_h3")


def test_presets_pass_validation():
    for name in ("video", "photos", "mixed"):
        vals = video_data_preset(name, "minimax_h3")
        TrainConfig(model_path="MiniMaxAI/MiniMax-H3", **vals).validate_still_masking()
