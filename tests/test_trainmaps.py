"""Training maps (rulztrainer/trainmaps.py): picking the pictures, laying maps on the grid, a capture on a toy model
(loss, effect, the trigger's attention, nothing of training disturbed), reading them back and drawing them."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from rulztrainer import trainmaps as TM
from rulztrainer.config import TrainConfig


@dataclass
class FakeItem:
    image: str
    dataset: str = ""
    category: str = "image"
    pair_side: str = ""
    width: int = 64
    height: int = 48

    @property
    def key(self):
        return self.image + ("" if self.category == "image" else f"#{self.category}")


# ---------------------------------------------------------------- picking


def test_pick_keys_spreads_evenly_and_is_stable():
    keys = [f"{i:02d}" for i in range(10)]
    assert TM.pick_keys(keys, 3) == ["00", "04", "09"]  # (round half to even: 4.5 -> 4)
    assert TM.pick_keys(list(reversed(keys)), 3) == TM.pick_keys(keys, 3)
    assert TM.pick_keys(keys, 1) == ["05"]
    assert TM.pick_keys(keys, 0) == [] and TM.pick_keys([], 3) == []
    assert TM.pick_keys(["a", "b"], 5) == ["a", "b"]


def test_pick_items_skips_crops_and_takes_one_per_dataset_and_one_pair():
    items = [FakeItem(f"a/{i}.jpg", "a") for i in range(6)] + [FakeItem(f"b/{i}.jpg", "b") for i in range(2)]
    items += [FakeItem("a/0.jpg", "a", category="face_crop"), FakeItem("pairs/p0.png", "a", pair_side="left")]
    items += [FakeItem("a/0.jpg", "a")]  # the same picture at another resolution: counted once
    got = TM.pick_items(items, 3)
    keys = [it.key for it in got]
    assert len(keys) == len(set(keys))
    assert not any("#" in k for k in keys)  # no close-up crops
    assert any(k.startswith("b/") for k in keys) and any(k.startswith("a/") for k in keys)
    assert "pairs/p0.png" in keys
    assert len(got) == 3
    assert [it.key for it in TM.pick_items(list(reversed(items)), 3)] == keys  # the same pictures every time


def test_pick_items_single_dataset_is_evenly_spread():
    items = [FakeItem(f"x/{i:02d}.jpg") for i in range(9)]
    assert [it.key for it in TM.pick_items(items, 3)] == ["x/00.jpg", "x/04.jpg", "x/08.jpg"]


# ---------------------------------------------------------------- maps from tensors


def test_to_map_latent_grid_and_tokens():
    lat = torch.zeros(1, 4, 6, 8)
    err = torch.arange(1 * 4 * 6 * 8, dtype=torch.float32).view(1, 4, 6, 8)
    assert torch.equal(TM.to_map(err, lat), err.mean(dim=1)[0])
    tok = torch.arange(48, dtype=torch.float32)  # one token per cell
    assert TM.to_map(tok, lat).shape == (6, 8)
    tok4 = torch.arange(12, dtype=torch.float32)  # 2x2 groups (Qwen-Image 2.1)
    m = TM.to_map(tok4[None], lat)
    assert m.shape == (3, 4) and m[1, 0] == 4
    with pytest.raises(RuntimeError):
        TM.to_map(torch.zeros(7), lat)


def test_grid_pools_to_max_side():
    g = TM.grid(torch.ones(100, 30))
    assert max(g["w"], g["h"]) <= TM.MAX_SIDE and len(g["v"]) == g["w"] * g["h"]
    small = TM.grid(torch.tensor([[1.0, 2.0], [3.0, 4.123456]]))
    assert small == {"w": 2, "h": 2, "v": [1.0, 2.0, 3.0, 4.123]}


# ---------------------------------------------------------------- a capture on a toy model


class FakeLora:
    def __init__(self):
        self.multiplier = 1.0


class ToyAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.q(x).unflatten(-1, (2, -1)).transpose(1, 2)  # [B, H, L, d]
        k = self.k(x).unflatten(-1, (2, -1)).transpose(1, 2)
        tap = getattr(self, "_rz_tap", None)
        if tap is not None:
            tap(q, k)
        return x


class ToyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = ToyAttn(dim)

    def forward(self, x):
        return self.attn(x)


class ToyModel(nn.Module):
    """model(latents [B, C, h, w], sigma * 1000, embeds [B, T, C], mask) with the trainer's taps convention, a
    'LoRA' adding lora_delta * multiplier, and dropout (so eval mode shows)."""

    def __init__(self, c=4, layers=2):
        super().__init__()
        self.layers = nn.ModuleList(ToyBlock(c) for _ in range(layers))
        self.lora = FakeLora()
        self.lora_delta = 0.5
        self.drop = nn.Dropout(0.5)
        self.calls = 0

    def forward(self, latents, t, embeds, mask):
        self.calls += 1
        assert latents.dtype == torch.bfloat16 and t.shape == (latents.shape[0],)
        b, c, h, w = latents.shape
        img = latents.float().flatten(2).transpose(1, 2)  # [B, h*w, C]
        txt = embeds.float()
        x = torch.cat([txt, img], dim=1)
        taps = getattr(self, "_rz_taps", None)
        if taps is not None and taps.armed:
            taps.set_layout(0, (txt.shape[1], h * w), (0, txt.shape[1]), mask.bool())
        for blk in self.layers:
            x = blk(x)
        out = self.drop(latents.float() * 0.5) + self.lora_delta * self.lora.multiplier
        return out


def _probes(c=4, h=3, w=4, trigger=None):
    lat = torch.randn(1, c, h, w, generator=torch.Generator().manual_seed(7))
    emb = torch.randn(1, 5, c, generator=torch.Generator().manual_seed(8))
    return [{"id": "0", "item": FakeItem("pic/0.jpg"), "latents": lat, "embeds": emb,
             "mask": torch.ones(1, 5, dtype=torch.bool), "trigger": trigger or {}}]


def _lines(job):
    return [json.loads(x) for x in (TM.maps_dir(job) / "maps.jsonl").read_text(encoding="utf-8").splitlines()]


def test_capture_writes_loss_and_effect_and_leaves_training_alone(tmp_path):
    model = ToyModel()
    model.train()
    injected = {"l": model.lora}
    probes = _probes()
    maps = TM.TrainingMaps(tmp_path, probes, every=10)
    maps.start(0, resume=False, pictures={"0": Image.new("RGB", (64, 48), "white")})
    torch.manual_seed(123)
    expect_next = torch.rand(3)
    torch.manual_seed(123)
    maps.capture(0, 0.0, model, injected, "cpu")
    assert torch.equal(torch.rand(3), expect_next)  # the global RNG as it was
    assert model.training and model.lora.multiplier == 1.0
    maps.capture(10, 1.0, model, injected, "cpu")
    rows = _lines(tmp_path)
    assert [r["step"] for r in rows] == [0, 10]
    first, second = rows[0]["maps"]["0"], rows[1]["maps"]["0"]
    assert set(first) == {"0.25", "0.5", "0.75"}
    assert "effect" not in first["0.5"] and "attn" not in first["0.5"]  # nothing to compare before training
    assert second["0.5"]["effect"]["v"] == pytest.approx([0.5] * 12, abs=1e-3)  # |pred - pred without the LoRA|
    # the loss map is (pred - target)^2 at the fixed noise, eval mode (no dropout), channel mean
    lat = probes[0]["latents"]
    noise = maps.noise(0, 1, lat.shape)
    noisy = (0.5 * lat + 0.5 * noise).to(torch.bfloat16)
    pred = noisy.float() * 0.5 + 0.5
    want = (pred - (noise - lat)).pow(2).mean(dim=1)[0]
    assert first["0.5"]["loss"]["v"] == pytest.approx(want.flatten().tolist(), rel=1e-3, abs=1e-3)
    assert first["0.5"]["loss"] == second["0.5"]["loss"]  # same noise, same LoRA: the same map
    meta = json.loads((TM.maps_dir(tmp_path) / "probes.json").read_text(encoding="utf-8"))
    assert meta["probes"][0]["image"] == "probe_0.jpg" and (TM.maps_dir(tmp_path) / "probe_0.jpg").exists()


def test_capture_without_adapter_has_no_effect_map(tmp_path):
    maps = TM.TrainingMaps(tmp_path, _probes(), every=5)
    maps.start(0, resume=False)
    maps.capture(5, 1.0, ToyModel(), {}, "cpu")  # a fine-tune: nothing to switch off
    assert "effect" not in _lines(tmp_path)[0]["maps"]["0"]["0.25"]


def test_trigger_attention_matches_a_softmax_over_the_text_and_restores_taps(tmp_path):
    model = ToyModel()
    sentinel = object()
    model._rz_taps = sentinel
    model.layers[0].attn._rz_tap = "training's tap"
    probes = _probes(trigger={"ohwx": [1], "gwelt": [3]})
    maps = TM.TrainingMaps(tmp_path, probes, every=5, sigmas=(0.5,))
    maps.start(0, resume=False)
    maps.capture(0, 0.0, model, {}, "cpu")
    assert model._rz_taps is sentinel and model.layers[0].attn._rz_tap == "training's tap"
    assert "_rz_tap" not in model.layers[1].attn.__dict__
    m = _lines(tmp_path)[0]["maps"]["0"]["0.5"]
    assert set(m["attn_by"]) == {"ohwx", "gwelt"}
    # by hand: each block's softmax over the 5 text keys, head mean, the trigger columns summed; block mean
    lat, emb = probes[0]["latents"], probes[0]["embeds"]
    noisy = (0.5 * lat + 0.5 * maps.noise(0, 0, lat.shape)).to(torch.bfloat16).float()
    x = torch.cat([emb, noisy.flatten(2).transpose(1, 2)], dim=1)
    shares = []
    for blk in model.layers:
        q = blk.attn.q(x).unflatten(-1, (2, -1)).transpose(1, 2)[:, :, 5:]
        k = blk.attn.k(x).unflatten(-1, (2, -1)).transpose(1, 2)[:, :, :5]
        p = (q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5).softmax(-1).mean(1)[0]  # [12, 5]
        shares.append(p[:, [1, 3]].sum(-1))
    want = torch.stack(shares).mean(0)
    assert m["attn"]["v"] == pytest.approx(want.tolist(), rel=1e-3, abs=1e-4)
    assert sum(m["attn_by"]["ohwx"]["v"]) + sum(m["attn_by"]["gwelt"]["v"]) == pytest.approx(sum(m["attn"]["v"]),
                                                                                              rel=1e-3)


def test_a_failing_capture_turns_the_maps_off_without_raising(tmp_path):
    class Broken(ToyModel):
        def forward(self, *a):
            raise RuntimeError("boom")

    model = Broken()
    model.train()
    maps = TM.TrainingMaps(tmp_path, _probes(), every=5)
    maps.start(0, resume=False)
    maps.capture(0, 0.0, model, {"l": model.lora}, "cpu")
    assert maps.failed and model.training and not maps.due(5)
    assert not (TM.maps_dir(tmp_path) / "maps.jsonl").exists()


def test_due_schedule():
    maps = TM.TrainingMaps("x", _probes(), every=4)
    maps.next_step = 0
    assert maps.due(0)
    maps.last_step, maps.next_step = 0, 4
    assert not maps.due(3) and maps.due(4)
    assert maps.due(2, last=True)  # the final step always gets one
    maps.last_step = 2
    assert not maps.due(2, last=True)  # but never twice


def test_start_clears_old_maps_unless_resumed(tmp_path):
    d = TM.maps_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "maps.jsonl").write_text('{"step": 5, "maps": {}}\n', encoding="utf-8")
    TM.TrainingMaps(tmp_path, _probes(), every=5).start(5, resume=True)
    assert (d / "maps.jsonl").exists()
    TM.TrainingMaps(tmp_path, _probes(), every=5).start(0, resume=False)
    assert not (d / "maps.jsonl").exists()


# ---------------------------------------------------------------- reading and drawing


def _write(job, rows):
    d = TM.maps_dir(job)
    d.mkdir(parents=True, exist_ok=True)
    (d / "probes.json").write_text(json.dumps({"probes": [{"id": "0", "image": "probe_0.jpg", "source": "a.jpg"}],
                                               "sigmas": [0.25, 0.5], "words": ["ohwx"]}), encoding="utf-8")
    text = "".join(json.dumps(r) + "\n" for r in rows) + '{"step": 99, "ma'  # a line still being written
    (d / "maps.jsonl").write_text(text, encoding="utf-8")


def _g(vals):
    return {"w": 2, "h": 1, "v": vals}


def test_read_keeps_the_later_of_a_repeated_step_and_skips_a_partial_line(tmp_path):
    _write(tmp_path, [{"step": 0, "epoch": 0, "maps": {"0": {"0.25": {"loss": _g([1, 1])}}}},
                      {"step": 4, "epoch": 1, "maps": {"0": {"0.25": {"loss": _g([9, 9])}}}},
                      {"step": 4, "epoch": 1, "maps": {"0": {"0.25": {"loss": _g([0.5, 2])}}}}])
    data = TM.read(tmp_path)
    assert [c["step"] for c in data["captures"]] == [0, 4]
    assert data["captures"][1]["maps"]["0"]["0.25"]["loss"]["v"] == [0.5, 2]
    assert TM.read(tmp_path / "nothing") is None


def test_series_learned_levels_and_scale(tmp_path):
    _write(tmp_path, [
        {"step": 0, "maps": {"0": {"0.25": {"loss": _g([1, 1])}, "0.5": {"loss": _g([3, 3])}}}},
        {"step": 4, "maps": {"0": {"0.25": {"loss": _g([0.5, 2]), "effect": _g([0.1, 0.2]),
                                            "attn": _g([0.3, 0.1]), "attn_by": {"ohwx": _g([0.3, 0.1])}},
                                   "0.5": {"loss": _g([1, 5])}}}}])
    data = TM.read(tmp_path)
    learned = TM.series(data, "0", "learned", "0.25")
    assert learned[0].tolist() == [[0, 0]] and learned[1].tolist() == [[0.5, -1]]
    mean = TM.series(data, "0", "loss", "all")
    assert mean[0].tolist() == [[2, 2]] and mean[1].tolist() == [[0.75, 3.5]]
    eff = TM.series(data, "0", "effect", "all")
    assert eff[0] is None and eff[1][0].tolist() == pytest.approx([0.1, 0.2])
    assert TM.series(data, "0", "attn", "0.25", "ohwx")[1][0].tolist() == pytest.approx([0.3, 0.1])
    one = {"captures": [{"step": 0, "maps": {"0": {"0.25": {"attn": _g([0.4, 0.2])}}}}]}
    assert TM.series(one, "0", "attn", "0.25", "ohwx")[0][0].tolist() == pytest.approx([0.4, 0.2])  # one word
    assert TM.scale(learned, "learned") == pytest.approx(np.percentile([0, 0, 0.5, 1], 99))
    assert TM.scale([None], "loss") == 1.0 and TM.scale([np.zeros((1, 2))], "loss") == 1.0


def test_colorize_and_render_overlay():
    rgb, alpha = TM.colorize(np.array([[1.0, -1.0, 0.0]]), "learned", 1.0)
    assert tuple(rgb[0, 0]) == TM._GOOD and tuple(rgb[0, 1]) == TM._BAD and alpha[0, 2] == 0
    rgb, alpha = TM.colorize(np.array([[0.0, 0.5, 2.0]]), "loss", 1.0)
    assert tuple(rgb[0, 0]) == (40, 80, 220) and tuple(rgb[0, 2]) == (225, 30, 30) and alpha[0, 2] == pytest.approx(0.9)
    pic = Image.new("RGB", (64, 48), (200, 10, 10))
    out = TM.render_overlay(pic, np.array([[0.0, 1.0], [1.0, 0.0]]), "loss", 1.0, width=128)
    assert out.size == (128, 96)
    plain = TM.render_overlay(pic, None, "loss", 1.0, width=128, colour=True)
    assert plain.getpixel((5, 5))[0] > 150  # the picture alone, in colour
    grey = TM.render_overlay(pic, np.zeros((2, 2)), "loss", 1.0, opacity=0.0, width=128)
    r, g, b = grey.getpixel((5, 5))
    assert r == g == b  # opacity 0: the grey picture alone


def test_config_checks():
    with pytest.raises(ValueError, match="training_maps_images"):
        _check(TrainConfig(training_maps=True, training_maps_images=0))
    with pytest.raises(ValueError, match="training_maps_every"):
        _check(TrainConfig(training_maps=True, training_maps_every=-1))
    _check(TrainConfig(training_maps=True))


def _check(cfg):
    """The training-maps lines of TrainConfig.validate()."""
    try:
        cfg.validate()
    except ValueError as e:
        if "training_maps" in str(e):
            raise


# ---------------------------------------------------------------- the trainer's side


class FakeArch:
    name = "moss_image"

    def pad_embeds(self, rows):
        e = torch.stack(rows)
        return e, torch.ones(e.shape[:2], dtype=torch.bool)

    def get_tokenizer(self, cfg, model_dir):
        return None

    def prompt_tokens(self, tokenizer, caption):
        """One token per word."""
        offsets, at = [], 0
        for word in caption.split(" "):
            offsets.append((at, at + len(word)))
            at += len(word) + 1
        return caption, offsets, 0, len(offsets)


def _stub_trainer(tmp_path, captions, **cfg_kw):
    from safetensors.torch import save_file

    from rulztrainer.dataset import BucketBatcher, Item
    from rulztrainer.trainer import Status, Trainer

    items = []
    for i, cap in enumerate(captions):
        img = tmp_path / f"p{i}.jpg"
        Image.new("RGB", (80, 48), (40 * i, 100, 200)).save(img)
        lat, txt = tmp_path / f"p{i}.lat.safetensors", tmp_path / f"p{i}.txt.safetensors"
        save_file({"latent": torch.randn(4, 3, 4)}, str(lat))
        save_file({"embeds": torch.randn(len(cap.split(" ")), 4)}, str(txt))
        items.append(Item(image=img, caption=cap, width=64, height=48, latent_path=lat, latent_flip_path=None,
                          text_path=txt))
    t = Trainer.__new__(Trainer)
    t.cfg = TrainConfig(job_name="t", output_dir=str(tmp_path), training_maps=True, **cfg_kw)
    t.job_dir = tmp_path / "job"
    t.job_dir.mkdir()
    t.arch, t.model_dir = FakeArch(), None
    t.status = Status(t.job_dir / "status.json")
    t.race_rows = t.ft = t.isolation = t.ema = t.dp = None
    t.batcher = BucketBatcher(items, 1, 0)
    model = ToyModel()
    model.train()
    t.replicas = [(model, {"l": model.lora}, "cpu")]
    return t, model


def test_trainer_sets_up_and_captures_maps(tmp_path):
    t, model = _stub_trainer(tmp_path, ["ohwx a man in a hat", "a red car on a street", "ohwx b man and dog"],
                             trigger_word="ohwx", training_maps_images=2)
    t._setup_maps(t.cfg, 0)
    assert t.maps is not None and t.maps.every == 3  # 3 images, 1 per step: one epoch = 3 steps
    assert t.maps.due(0)
    t._capture_maps(model, t.replicas[0][1], 0)
    t._capture_maps(model, t.replicas[0][1], 3)
    assert model.training and model.lora.multiplier == 1.0
    data = TM.read(t.job_dir)
    assert len(data["probes"]) == 2 and data["words"] == ["ohwx"]
    assert [c["step"] for c in data["captures"]] == [0, 3] and data["captures"][1]["epoch"] == 1.0
    pics = {p["id"]: p for p in data["probes"]}
    for pid, p in pics.items():
        assert Image.open(TM.maps_dir(t.job_dir) / p["image"]).size == (64, 48)  # the bucket crop
        last = data["captures"][1]["maps"][pid]["0.5"]
        assert ("attn" in last) == bool(p["words"]) and "effect" in last and "loss" in last
    assert any(p["words"] == ["ohwx"] for p in data["probes"])
    assert json.loads((t.job_dir / "status.json").read_text(encoding="utf-8"))["training_maps"] == 3


def test_trainer_maps_off_in_a_race_and_when_disabled(tmp_path):
    t, _ = _stub_trainer(tmp_path, ["a b c"])
    t.race_rows = [("a", t.cfg)]
    t._setup_maps(t.cfg, 0)
    assert t.maps is None
    t.race_rows = None
    t.cfg.training_maps = False
    t._setup_maps(t.cfg, 0)
    assert t.maps is None
