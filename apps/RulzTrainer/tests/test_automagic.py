import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as lora_mod  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402
from rulztrainer.trainer import build_optimizer, build_scheduler, current_lr  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16, latent_channels=128)


def _step(model, params, opt):
    x = torch.randn(1, 128, 4, 4).to(torch.bfloat16)
    c = torch.randn(1, 5, 32).to(torch.bfloat16)
    loss = model(x, torch.tensor([300.0]), c).float().pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return loss.item()


def test_automagic3_trains_and_adapts_lr(tmp_path):
    torch.manual_seed(0)
    model = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    injected = lora_mod.inject_lora(model, 4, 4, ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
    params = list(lora_mod.lora_parameters(injected))
    cfg = TrainConfig(optimizer="automagic3", learning_rate=1e-4, automagic_polarity_history=4, steps=20)
    opt = build_optimizer(cfg, params)
    sched = build_scheduler(cfg, opt)
    model.train()
    before = [p.detach().clone() for p in params]
    for _ in range(8):
        _step(model, params, opt)
        sched.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, params))
    lr = current_lr(opt, sched)
    assert cfg.automagic_min_lr <= lr <= cfg.automagic_max_lr
    assert all(torch.isfinite(p).all() for p in params)
    # state round trip (what save_state / resume_state do)
    sd = opt.state_dict()
    torch.save(sd, tmp_path / "opt.pt")
    opt2 = build_optimizer(cfg, params)
    opt2.load_state_dict(torch.load(tmp_path / "opt.pt", weights_only=False))
    assert abs(current_lr(opt2, sched) - lr) < 1e-6 * lr
    _step(model, params, opt2)


def test_layer_type_groups_each_adapt_their_own_lr():
    """automagic_groups: by_type gives Krea 2's text fusion, attention, feed-forward and the rest their own groups."""
    from rulztrainer import archs
    from rulztrainer import lora as L
    from rulztrainer.config import TrainConfig
    from rulztrainer.krea2.model import Krea2Transformer
    from rulztrainer.trainer import build_optimizer

    krea = archs.arch_by_name("krea2")
    model = Krea2Transformer(in_channels=64, num_layers=2, attention_head_dim=16, num_attention_heads=4,
                             num_key_value_heads=2, intermediate_size=96, timestep_embed_dim=32, text_hidden_dim=40,
                             num_text_layers=3, text_num_attention_heads=4, text_num_key_value_heads=4,
                             text_intermediate_size=56, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
                             axes_dims_rope=(4, 6, 6))
    targets = list(krea.default_targets) + ["embed"]
    inj = L.inject_lora(model, 4, 4.0, targets, names=krea.lora_module_names(model, targets, None))
    groups = L.lora_param_groups_by(inj, 1e-4, krea.param_group_of)
    assert [g["name"] for g in groups] == ["attn", "mlp", "other", "text"]
    assert sum(len(g["params"]) for g in groups) == 2 * len(inj)
    assert krea.param_group_of("transformer_blocks.0.attn.to_gate") == "attn"
    assert krea.param_group_of("text_fusion.refiner_blocks.0.ff.down") == "text"
    moss = archs.arch_by_name("moss_image")
    assert [moss.param_group_of(n) for n in ("layers.3.proj.q", "layers.3.proj.ffn_down", "text_embedder.0.qkv")] \
        == ["attn", "mlp", "text"]
    opt = build_optimizer(TrainConfig(optimizer="automagic3"), groups)
    assert len(opt.param_groups) == 4  # kept as groups, each with the configured start lr
    single = build_optimizer(TrainConfig(optimizer="automagic3", lora_plus_ratio=4.0),
                             L.lora_param_groups(inj, 1e-4, 4.0))
    assert len(single.param_groups) == 1  # LoRA+ groups still collapse: Automagic owns the lr
