import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import likeness as L  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.dataset import BucketBatcher, Item  # noqa: E402


def _cfg(**kw):
    return TrainConfig(probe_every=10, probe_ema=0.0, likeness_noise_band=1.0, likeness_track_best=True,
                       likeness_patience=3, likeness_lr_control=True, likeness_lr_min=0.5, likeness_lr_max=2.0,
                       likeness_lr_step=0.1, likeness_curriculum=True, likeness_curriculum_strength=1.0, **kw)


def test_controller_best_stop_lr_and_curriculum():
    c = L.LikenessController(_cfg())
    # rising scores: best updates, lr factor climbs
    rec, act = c.update(0, 40.0, 80.0)
    assert act.get("save_best") and rec.best == rec.combined_ema
    rec, act = c.update(10, 50.0, 80.0)
    assert act.get("save_best") and act.get("lr_ratio", 0) > 1
    assert c.lr_factor > 1.0
    # face lags body -> positive face bias
    assert c.face_bias > 0 and act.get("face_bias", c.face_bias) > 0
    # falling scores: lr factor shrinks, patience counts, then stop
    stops = []
    for i, f in enumerate([45.0, 40.0, 35.0]):
        rec, act = c.update(20 + 10 * i, f, 80.0)
        stops.append(bool(act.get("stop")))
    assert c.lr_factor < 1.1 and stops[-1] and c.stopped
    assert c.best_step == 10
    # bounds
    for _ in range(30):
        c.update(100, 90.0, 90.0)
    assert c.lr_factor <= 2.0


def test_controller_handles_missing_modalities():
    c = L.LikenessController(_cfg())
    rec, _ = c.update(0, None, 70.0)
    assert rec.combined_ema == 70.0 and c.face_bias == 0.0
    rec, _ = c.update(10, 60.0, None)
    assert rec.face_ema == 60.0 and rec.body_ema == 70.0


def test_apply_lr_ratio_adamw_with_scheduler_and_automagic():
    from rulztrainer.trainer import build_optimizer, build_scheduler, current_lr

    p = torch.nn.Parameter(torch.zeros(4, 4))
    cfg = TrainConfig(optimizer="adamw", learning_rate=1e-4, lr_scheduler="constant", warmup_steps=0)
    opt = build_optimizer(cfg, [p])
    sch = build_scheduler(cfg, opt)
    L.apply_lr_ratio(opt, sch, 1.5)
    p.grad = torch.zeros_like(p)
    opt.step()
    sch.step()
    assert abs(current_lr(opt, sch) - 1.5e-4) < 1e-12  # survives the scheduler step
    cfg2 = TrainConfig(optimizer="automagic3", learning_rate=1e-4)
    p2 = torch.nn.Parameter(torch.zeros(4, 4))
    opt2 = build_optimizer(cfg2, [p2])
    sch2 = build_scheduler(cfg2, opt2)
    p2.grad = torch.randn_like(p2)
    opt2.step()
    before = current_lr(opt2, sch2)
    L.apply_lr_ratio(opt2, sch2, 0.5)
    assert abs(current_lr(opt2, sch2) - before * 0.5) < 1e-9


def test_batcher_weights_and_curriculum(tmp_path):
    items = [Item(image=tmp_path / f"{i}.png", caption="", width=512, height=512, latent_path=tmp_path / "l",
                  latent_flip_path=None, text_path=tmp_path / "t") for i in range(10)]
    fractions = {str(it.image.resolve()): (0.1 if i < 3 else 0.0) for i, it in enumerate(items)}
    w = L.curriculum_weights(items, fractions, face_bias=1.0)
    assert all(w[str(it.image)] == 2.0 for it in items[:3]) and all(w[str(it.image)] == 0.2 for it in items[3:])
    b = BucketBatcher(items, 1, 0)
    b.set_weights(w)
    picks = [batch[0] for _ in range(20) for batch in b.epoch()]
    face_share = sum(1 for it in picks if str(it.image) in {str(x.image) for x in items[:3]}) / len(picks)
    assert face_share > 0.6  # 3 of 10 images get most of the draws
    b.set_weights(None)
    picks = [batch[0] for batch in b.epoch()]
    assert len(set(str(it.image) for it in picks)) == 10  # uniform shuffle visits every image
    assert L.timestep_logit_shift(1.0) < 0 < L.timestep_logit_shift(-1.0)
    assert L.probe_body_size(512) == (512, 768)
