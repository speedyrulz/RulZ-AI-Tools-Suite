import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer import regions as R  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.dataset import (Item, augment_items, bucket_size, crop_caption, load_batch,  # noqa: E402
                                 load_image_fit)
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402
from rulztrainer.trainer import build_optimizer, build_scheduler, current_lr, sample_sigmas  # noqa: E402


# ---------------------------------------------------------------- phase 0: geometry


def _find_square(img: Image.Image):
    a = np.asarray(img.convert("L")) > 128
    ys, xs = np.where(a)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def test_remap_matches_imageops_fit(tmp_path):
    for (W, H, bw, bh) in [(3000, 1500, 1344, 672), (900, 1600, 592, 1056), (1024, 1024, 512, 512)]:
        img = Image.new("RGB", (W, H), (0, 0, 0))
        box = (W * 0.55, H * 0.3, W * 0.55 + 200, H * 0.3 + 160)
        img.paste((255, 255, 255), tuple(int(v) for v in box))
        fitted = ImageOps.fit(img, (bw, bh), method=Image.Resampling.NEAREST, centering=(0.5, 0.5))
        got = _find_square(fitted)
        exp = R.remap_bbox(box, W, H, bw, bh)
        assert exp is not None
        for g, e in zip(got, exp):
            assert abs(g - e) <= 2, (W, H, bw, bh, got, exp)
    # crop item: box relative to a src_box
    img = Image.new("RGB", (2000, 2000), 0)
    img.paste((255, 255, 255), (1000, 1000, 1100, 1100))
    src = (800, 800, 1400, 1400)
    fitted = ImageOps.fit(img.crop(src), (512, 512), method=Image.Resampling.NEAREST)
    got = _find_square(fitted)
    exp = R.remap_bbox((1000, 1000, 1100, 1100), 2000, 2000, 512, 512, src_box=src)
    for g, e in zip(got, exp):
        assert abs(g - e) <= 2


def test_to_cells_and_weight_map_and_flip():
    assert R.to_cells((17.0, 0.0, 40.0, 33.0), 128, 64) == (1, 0, 3, 3)
    mask = np.zeros((64, 128), dtype=bool)
    mask[:, 32:96] = True  # person in the middle half
    rm = R.build_weight_map(128, 64, (40.0, 8.0, 56.0, 24.0), mask, 3.0, 1.5, 0.3, face_margin=0.0, feather=0)
    w = rm.weight[0]
    assert w.shape == (4, 8) and abs(float(w.mean()) - 1.0) < 1e-5
    assert rm.face_cells == (2, 0, 4, 2) and rm.has_person
    raw_face, raw_bg = w[0, 2], w[0, 0]
    assert raw_face > w[2, 3] > raw_bg  # face > person > background after normalisation
    assert rm.prior[0, 0, 0] and not rm.prior[0, 0, 3]
    fl = R.flip_region_map(rm, 128)
    assert fl.face_cells == (4, 0, 6, 2) and torch.equal(fl.weight, rm.weight.flip(-1))
    uni = R.uniform_region_map(128, 64)
    assert torch.all(uni.weight == 1) and not uni.prior.any()
    # feathering keeps mean 1
    rm2 = R.build_weight_map(128, 64, (40.0, 8.0, 56.0, 24.0), None, 3.0, 1.5, 0.3, feather=2)
    assert abs(float(rm2.weight.mean()) - 1.0) < 1e-5


def test_faces_rescales_thumbnail_bboxes(tmp_path):
    from rulztrainer.facesim import FaceScorer

    big = tmp_path / "big.png"
    Image.new("RGB", (4096, 2048)).save(big)

    class Face:
        def __init__(self):
            self.bbox = np.array([100.0, 50.0, 300.0, 250.0])  # in the 2048-wide thumbnail
            self.kps = np.array([[150.0, 100.0]])
            self.det_score = 0.9

    sc = FaceScorer.__new__(FaceScorer)
    sc.app = type("App", (), {"get": lambda self, arr: [Face()]})()
    (f,) = sc.faces(big)
    assert np.allclose(f.bbox, [200, 100, 600, 500]) and np.allclose(f.kps, [[300, 200]])


def test_load_batch_returns_regions_and_flips(tmp_path):
    from safetensors.torch import save_file

    cfg = TrainConfig(region_weights=True, flip_augment=True, caption_dropout=0.0, cache_dir=str(tmp_path))
    lat = torch.zeros(128, 4, 8)
    save_file({"latent": lat}, str(tmp_path / "l.safetensors"))
    save_file({"latent": lat + 1}, str(tmp_path / "lf.safetensors"))
    save_file({"embeds": torch.zeros(3, 4096, dtype=torch.bfloat16)}, str(tmp_path / "t.safetensors"))
    rm = R.build_weight_map(128, 64, (0.0, 0.0, 32.0, 32.0), None, 3.0, 1.5, 0.3, face_margin=0.0, feather=0)
    R.save_region_map(tmp_path / "r.safetensors", rm)
    it = Item(image=tmp_path / "x.png", caption="c", width=128, height=64, latent_path=tmp_path / "l.safetensors",
              latent_flip_path=tmp_path / "lf.safetensors", text_path=tmp_path / "t.safetensors",
              region_path=tmp_path / "r.safetensors")

    class RngFlip:
        def random(self):
            return 0.0  # always pick the flip

    latents, embeds, mask, regions = load_batch([it], cfg, RngFlip())
    assert float(latents.mean()) == 1.0 and regions is not None
    assert regions.weight.shape == (1, 1, 4, 8) and regions.meta[0].face_cells == (6, 0, 8, 2)
    assert regions.weight[0, 0, 0, 7] > regions.weight[0, 0, 0, 0]
    cfg.region_weights = False
    assert load_batch([it], cfg, RngFlip())[3] is None


# ---------------------------------------------------------------- phase 1: weighted loss on the tiny model


def test_region_weighted_loss_equivalence():
    torch.manual_seed(0)
    pred = torch.randn(2, 128, 4, 8)
    target = torch.randn(2, 128, 4, 8)
    ones = torch.ones(2, 1, 4, 8)
    assert torch.allclose(((pred - target) ** 2 * ones).mean(dim=(1, 2, 3)), (pred - target).pow(2).mean(dim=(1, 2, 3)))
    rm = R.build_weight_map(128, 64, (0.0, 0.0, 32.0, 32.0), None, 3.0, 1.5, 0.3, face_margin=0.0, feather=0)
    weighted = ((pred - target) ** 2 * rm.weight[None]).mean(dim=(1, 2, 3))
    assert weighted.shape == (2,) and torch.isfinite(weighted).all()


# ---------------------------------------------------------------- phase 2: timesteps


def test_timestep_modes():
    gen = torch.Generator().manual_seed(0)
    for mode, lo, hi in (("content", 0.05, 0.25), ("identity", 0.33, 0.43), ("style", 0.75, 0.95), ("uniform", 0.45, 0.55)):
        cfg = TrainConfig(timestep_sampling=mode)
        s = sample_sigmas(cfg, 20000, gen)
        med = float(s.median())
        assert lo <= med <= hi, (mode, med)
        assert float(s.min()) >= 0.001 and float(s.max()) <= 1.0
        assert torch.allclose((s * 1000).round(), s * 1000, atol=1e-3)  # snapped to the k/1000 grid
    cfg = TrainConfig(timestep_sampling="content", sigma_max=0.9, sigma_min=0.1)
    s = sample_sigmas(cfg, 5000, gen)
    assert float(s.max()) <= 0.9 and float(s.min()) >= 0.1
    cfg = TrainConfig(timestep_sampling="content")
    base = float(sample_sigmas(cfg, 20000, gen).median())
    shifted = float(sample_sigmas(cfg, 20000, gen, logit_shift=-1.0).median())
    assert shifted < base  # negative shift -> lower sigma


# ---------------------------------------------------------------- phase 3: LoRA+


def test_lora_plus_groups_and_resume_check():
    tiny = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16, latent_channels=128)
    model = MOSSImageTransformer(**tiny).to(torch.bfloat16)
    inj = L.inject_lora(model, 4, 4, ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
    cfg = TrainConfig(optimizer="adamw", learning_rate=1e-4, lora_plus_ratio=16, warmup_steps=5)
    groups = L.lora_param_groups(inj, cfg.learning_rate, cfg.lora_plus_ratio)
    assert len(groups) == 2 and groups[1]["lr"] == 16 * groups[0]["lr"]
    opt = build_optimizer(cfg, groups)
    sch = build_scheduler(cfg, opt)
    for p in L.lora_parameters(inj):
        p.grad = torch.zeros_like(p)
    for _ in range(10):
        opt.step()
        sch.step()
    assert abs(opt.param_groups[1]["lr"] / opt.param_groups[0]["lr"] - 16) < 1e-6
    assert abs(current_lr(opt, sch) - 1e-4) < 1e-12
    assert len(L.lora_param_groups(inj, 1e-4, 0.0)) == 1
    cfg2 = TrainConfig(optimizer="automagic3", learning_rate=1e-4, lora_plus_ratio=16)
    opt2 = build_optimizer(cfg2, L.lora_param_groups(inj, 1e-4, 16))
    assert len(opt2.param_groups) == 1  # collapsed with a warning


# ---------------------------------------------------------------- phase 4: crops


def test_augment_items_geometry_and_captions(tmp_path):
    cfg = TrainConfig(crop_augment="face+bust", trigger_word="zxc", crop_augment_min_px=200, cache_dir=str(tmp_path))
    img = tmp_path / "a.png"
    Image.new("RGB", (2000, 3000)).save(img)
    base = Item(image=img, caption="zxc, a woman standing", width=896, height=1344, latent_path=tmp_path / "l",
                latent_flip_path=None, text_path=tmp_path / "t")
    analysis = {str(img.resolve()): {"bbox": [900.0, 500.0, 1100.0, 700.0], "kps": None, "det": 0.9, "W": 2000, "H": 3000}}
    items = augment_items([base], cfg, analysis)
    assert len(items) == 3
    face, bust = items[1], items[2]
    assert face.category == "face_crop" and bust.category == "bust_crop"
    assert face.src_box == (750, 350, 1250, 850)  # 2.5 x 200 square centred on the face
    assert bust.src_box[2] - bust.src_box[0] == 1000
    assert bust.src_box[0] <= face.src_box[0] and bust.src_box[2] >= face.src_box[2]  # bust contains the face box
    assert (bust.src_box[1] + bust.src_box[3]) / 2 > (face.src_box[1] + face.src_box[3]) / 2  # centred lower
    assert face.caption == "close-up portrait photo of zxc, a woman standing"
    assert crop_caption("{trigger} portrait, {raw}", base.caption, cfg) == "zxc portrait, a woman standing"
    assert face.key != bust.key != base.key and face.latent_path != base.latent_path
    assert face.width == face.height  # square bucket
    # skipped when the face is already large or the crop is too small
    big_face = {str(img.resolve()): {"bbox": [0.0, 0.0, 1500.0, 1500.0], "kps": None, "det": 0.9, "W": 2000, "H": 3000}}
    assert all(it.category != "face_crop" for it in augment_items([base], cfg, big_face))
    tiny_face = {str(img.resolve()): {"bbox": [900.0, 500.0, 940.0, 540.0], "kps": None, "det": 0.9, "W": 2000, "H": 3000}}
    assert len(augment_items([base], cfg, tiny_face)) == 2  # face crop (100 px) skipped, bust (200 px) kept
    assert len(augment_items([base], TrainConfig(crop_augment="none"), analysis)) == 1
    # load_image_fit honours src_box
    fitted = load_image_fit(img, 512, 512, (750, 350, 1250, 850))
    assert fitted.size == (512, 512)
    bw, bh = bucket_size(500, 500, cfg)
    assert bw == bh


def test_identity_preset_applies_and_fills_prompts():
    from rulztrainer.config import PRESETS, apply_preset

    base = TrainConfig(trigger_word="zxc", learning_rate=5e-4, probe_face_prompt="my own prompt").to_dict()
    out = apply_preset(base, "identity")
    cfg = TrainConfig.from_dict(out)
    assert cfg.preset == "identity" and cfg.region_weights and cfg.lora_plus_ratio == 4 and cfg.learning_rate == 1e-4
    assert cfg.probe_face_prompt == "my own prompt"  # existing prompts are kept
    assert "zxc" in cfg.probe_body_prompt  # empty ones are filled with the trigger word
    assert all(getattr(cfg, k) == v for k, v in PRESETS["identity"].items())
    try:
        apply_preset(base, "nope")
        raise AssertionError("unknown preset accepted")
    except ValueError:
        pass
