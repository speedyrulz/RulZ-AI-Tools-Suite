"""Weight average (rulztrainer.ema): per-image decay, swapping the average in and out, state, and the trainer's use of
it (saves and probes get the average, training states keep the raw weights and the average next to them)."""
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer.ema import LoraEMA  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=3, text_dim=32, freq_dim=16, latent_channels=128)
TARGETS = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]


def lora(seed=0):
    torch.manual_seed(seed)
    model = MOSSImageTransformer(**TINY)
    return model, L.inject_lora(model, 4, 4.0, TARGETS)


def params(inj):
    return [p for m in inj.values() for p in (m.lora_down.weight, m.lora_up.weight)]


def test_decay_warm_up_cap_and_per_image():
    _, inj = lora()
    ema = LoraEMA(inj, 0.98)
    assert ema.image_decay(0) == pytest.approx(0.1) and ema.image_decay(8) == pytest.approx(0.5)
    assert ema.image_decay(1000) == 0.98  # the warm-up (1 + n) / (10 + n) reaches 0.98 at n = 440
    assert ema.image_decay(439) < 0.98
    short = LoraEMA(inj, 0.98, total_images=100)
    assert short.image_decay(10_000) == pytest.approx(0.96)  # capped at 1 - 4 / total images
    ema.images = 500
    assert ema.step_decay(2) == pytest.approx(0.98 ** 2)
    with pytest.raises(ValueError):
        LoraEMA(inj, 1.0)


def test_one_step_of_two_images_equals_two_steps_of_one():
    _, inj = lora()
    a, b = LoraEMA(inj, 0.98), LoraEMA(inj, 0.98)
    with torch.no_grad():
        for p in params(inj):
            p.add_(1.0)  # constant weights after the shadows were taken
    a.update(images=2)
    b.update(images=1)
    b.update(images=1)
    assert a.images == b.images == 2
    for x, y in zip(a.shadow, b.shadow):
        assert torch.allclose(x, y, atol=1e-6)


def test_shadow_follows_the_closed_form():
    _, inj = lora()
    ema = LoraEMA(inj, 0.9)
    p0 = [p.detach().clone() for p in params(inj)]
    weights = []
    for k in range(5):
        with torch.no_grad():
            for p in params(inj):
                p.add_(0.1)
        weights.append([p.detach().clone() for p in params(inj)])
        ema.update()
    want = [w.clone() for w in p0]  # shadow_k = d_k * shadow_(k-1) + (1 - d_k) * w_k
    for k in range(5):
        d = min(0.9, (1 + k) / (10 + k))
        want = [d * s + (1 - d) * w for s, w in zip(want, weights[k])]
    for s, w in zip(ema.shadow, want):
        assert torch.allclose(s, w, atol=1e-6)


def test_applied_swaps_in_and_restores_bitwise():
    _, inj = lora()
    ema = LoraEMA(inj, 0.98)
    with torch.no_grad():
        for p in params(inj):
            p.normal_()
    raw = [p.detach().clone() for p in params(inj)]
    ema.update()
    with ema.applied():
        assert all(torch.equal(p, s) for p, s in zip(params(inj), ema.shadow))
        with ema.applied():  # nested: still the average, and the inner exit restores nothing yet
            pass
        assert all(torch.equal(p, s) for p, s in zip(params(inj), ema.shadow))
        with pytest.raises(RuntimeError):
            ema.update()
    assert all(torch.equal(p, r) for p, r in zip(params(inj), raw))
    with pytest.raises(KeyError):
        with ema.applied():
            raise KeyError("render failed")
    assert all(torch.equal(p, r) for p, r in zip(params(inj), raw)) and not ema.applied_now


def test_state_round_trip_mismatch_and_reset():
    _, inj = lora()
    ema = LoraEMA(inj, 0.98)
    with torch.no_grad():
        for p in params(inj):
            p.add_(1.0)
    ema.update(images=3)
    sd = ema.state_dict()
    other = LoraEMA(inj, 0.98)
    other.load_state_dict(sd)
    assert other.images == 3 and other.updates == 1
    assert all(torch.equal(x, y) for x, y in zip(other.shadow, ema.shadow))
    _, inj2 = lora()
    with pytest.raises(ValueError):
        LoraEMA({k: v for k, v in list(inj2.items())[:-1]}, 0.98).load_state_dict(sd)
    other.reset()
    assert other.images == 0 and all(torch.equal(s, p) for s, p in zip(other.shadow, params(inj)))


def _trainer(tmp_path, inj, model, decay=0.98):
    from rulztrainer import archs
    from rulztrainer.config import TrainConfig
    from rulztrainer.trainer import Status, Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainConfig(job_name="t", output_dir=str(tmp_path), lora_ema=decay, lora_save_dtype="fp32")
    trainer.job_dir = tmp_path
    trainer.arch = archs.arch_by_name("moss_image")
    trainer.status = Status(tmp_path / "status.json")
    trainer.dp = trainer.balancer = None
    trainer.res_stage = 0
    trainer.replicas = [(model, inj, "cpu")]
    trainer.ema = LoraEMA(inj, decay) if decay else None
    trainer.resumed_ema = None
    return trainer


def test_trainer_saves_the_average_and_states_keep_raw_weights(tmp_path):
    from safetensors.torch import load_file

    from rulztrainer.trainer import build_optimizer, build_scheduler

    model, inj = lora()
    trainer = _trainer(tmp_path, inj, model)
    with torch.no_grad():
        for p in params(inj):
            p.normal_()
    trainer.ema.update(images=20)
    raw = [p.detach().clone() for p in params(inj)]
    path = trainer.save(inj, 10)
    saved = load_file(str(path))
    name = next(iter(inj))
    k = trainer.ema.keys.index(f"{name}.lora_down")
    assert torch.equal(saved[f"diffusion_model.{name}.lora_down.weight"], trainer.ema.shadow[k])
    assert all(torch.equal(p, r) for p, r in zip(params(inj), raw))  # training goes on with the raw weights
    # a training state keeps the raw weights and the average beside them; loading restores both
    opt = build_optimizer(trainer.cfg.__class__(optimizer="adamw"), list(L.lora_parameters(inj)))
    sched = build_scheduler(trainer.cfg, opt)
    gen, rng = torch.Generator().manual_seed(0), random.Random(0)
    state = trainer.save_state(10, inj, opt, sched, gen, rng)
    assert torch.equal(load_file(str(state / "lora.safetensors"))[f"diffusion_model.{name}.lora_down.weight"], raw[k])
    model2, inj2 = lora(seed=1)
    trainer2 = _trainer(tmp_path, inj2, model2)
    opt2 = build_optimizer(trainer.cfg.__class__(optimizer="adamw"), list(L.lora_parameters(inj2)))
    trainer2.load_state(state, inj2, opt2, build_scheduler(trainer.cfg, opt2), gen, rng)
    assert all(torch.equal(p, r) for p, r in zip(params(inj2), raw))
    ema2 = LoraEMA(inj2, 0.98)
    ema2.load_state_dict(trainer2.resumed_ema)
    assert ema2.images == 20 and all(torch.equal(x, y) for x, y in zip(ema2.shadow, trainer.ema.shadow))


def test_data_parallel_replicas_render_with_the_average(tmp_path):
    from rulztrainer.multigpu import DataParallelLoRA

    model, inj = lora()
    replica, rinj = lora(seed=1)
    trainer = _trainer(tmp_path, inj, model)
    trainer.dp = DataParallelLoRA(inj, [(replica, rinj, "cpu")])
    with torch.no_grad():
        for p in params(inj):
            p.normal_()
    trainer.dp.sync_weights()
    trainer.ema.update()
    raw = [p.detach().clone() for p in params(inj)]
    with trainer._averaged(replicas=True):
        assert all(torch.equal(p, s) for p, s in zip(params(rinj), trainer.ema.shadow))
        with trainer._averaged(replicas=True):  # nested (a probe that also saves): no extra swap
            pass
        assert all(torch.equal(p, s) for p, s in zip(params(rinj), trainer.ema.shadow))
    assert all(torch.equal(p, r) for p, r in zip(params(rinj), raw))
    assert all(torch.equal(p, r) for p, r in zip(params(inj), raw))
