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
