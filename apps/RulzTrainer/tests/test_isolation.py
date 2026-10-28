"""LoRA Isolation: subjects and fused triggers, trigger-free DOP captions, balance, side-by-side pairs, the trainer's
preparation of the items, and the DOP pass on a tiny model (CPU)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import isolation as ISO  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.trainer import Trainer  # noqa: E402


def test_subjects_and_fused_triggers():
    caps = ["gwelt, a young man smiling at a cafe", "gwelt, a man in a coat", "a photo of gwelt outdoors"]
    s = ISO.resolve_subject("ds", "gwelt", caps)
    assert (s.phrase, s.fused, s.kind) == ("a man", "gweltman", "person")
    assert ISO.fuse("gwelt, a man in a coat", s.rules) == "gweltman in a coat"
    assert ISO.fuse("a photo of gwelt outdoors", s.rules) == "a photo of gweltman outdoors"
    assert ISO.fuse("gweltman at a cafe", s.rules) == "gweltman at a cafe"  # already fused: unchanged
    # the class from "is a", from any caption when the trigger is prepended later, an animal, a style
    assert ISO.resolve_subject("ds", "rexo", ["a dog on grass"], "dog").fused == "rexodog"
    assert ISO.resolve_subject("ds", "rexo", ["a dog on grass"]).kind == "thing"
    assert ISO.resolve_subject("ds", "zw", ["a woman in red"]).phrase == "a woman"
    st = ISO.resolve_subject("ds", "papercut", ["a city"], kind="style")
    assert (st.fused, st.phrase, st.rules, st.kind) == ("papercut", "", [], "style")
    joined = ISO.resolve_subject("ds", "gweltman", ["gweltman at a cafe"])  # a trigger already joined: as written
    assert (joined.fused, joined.phrase, joined.rules) == ("gweltman", "a man", [])
    with pytest.raises(ValueError, match="never say what"):
        ISO.resolve_subject("ds", "qq", ["a landscape"])
    with pytest.raises(ValueError, match="trigger word"):
        ISO.resolve_subject("ds", "", caps)


def test_trigger_free_captions():
    pairs = [("gweltman", "a man"), ("papercut", "")]
    assert ISO.dop_caption("gweltman at a cafe", pairs) == "a man at a cafe"
    assert ISO.dop_caption("papercut, a city at night", pairs) == "a city at night"  # a style's word is dropped
    assert ISO.dop_caption("a city, papercut, at night", pairs) == "a city, at night"
    assert ISO.dop_caption("a landscape", pairs) is None and ISO.dop_caption("", pairs) is None
    pair = "an Indian man in his 30s on the left, and gweltman on the right, two people side by side"
    assert ISO.dop_caption(pair, pairs).startswith("an Indian man in his 30s on the left, and a man on the right")


def test_balance():
    assert ISO.balance_repeats({"a": 36, "b": 53, "c": 8}, {"a": 1, "b": 1, "c": 1}) == {"a": 1, "b": 1, "c": 7}
    assert ISO.balance_repeats({"a": 10, "b": 20}, {"a": 3, "b": 1}) == {"a": 3, "b": 2}  # 30 a epoch: b to 40
    assert ISO.balance_repeats({"a": 10, "b": 20}, {"a": 1, "b": 1}) == {"a": 2, "b": 1}


def _strangers(tmp_path, n=4):
    d = tmp_path / "strangers"
    d.mkdir()
    people = []
    for i in range(n):
        Image.new("RGB", (64, 80), (50 * i, 80, 120)).save(d / f"s{i}.png")
        people.append((d / f"s{i}.png", {"description": f"stranger number {i}", "sex": "man"}))
    return people


def test_pairs(tmp_path):
    photos = []
    for i in range(3):
        p = tmp_path / f"me{i}.png"
        Image.new("RGB", (100, 160), (200, 10 * i, 10)).save(p)
        photos.append(p)
    s = ISO.resolve_subject("ds", "gwelt", ["gwelt, a man"])
    people = _strangers(tmp_path)
    analysis = {str(photos[0].resolve()): {"bbox": [30, 40, 70, 80], "det": 0.9, "W": 100, "H": 160}}
    out = tmp_path / "pairs"
    pairs = ISO.build_pairs(s, photos, out, people, 5, seed=42, analysis=analysis)
    assert len(pairs) == 5 and all(p.is_file() for p, _ in pairs)
    with Image.open(pairs[0][0]) as im:
        assert im.size == (2 * ISO.PANEL, ISO.PANEL)
    for _, cap in pairs:  # the stranger first, then the subject's fused word, on opposite sides
        assert cap.startswith("stranger number") and ", and gweltman on the " in cap and cap.endswith(ISO.TAIL)
        assert ("on the left, and gweltman on the right" in cap) or ("on the right, and gweltman on the left" in cap)
    stamp = pairs[0][0].stat().st_mtime_ns
    again = ISO.build_pairs(s, photos, out, people, 5, seed=42, analysis=analysis)
    assert again == pairs and pairs[0][0].stat().st_mtime_ns == stamp  # unchanged inputs: reused
    assert ISO.face_box({"bbox": [30, 40, 70, 80], "det": 0.9, "W": 100, "H": 160}) == (6, 16, 94, 104)
    assert ISO.face_box({"bbox": [30, 40, 70, 80], "det": 0.2, "W": 100, "H": 160}) is None


def test_fusion_warning_with_a_real_tokenizer():
    from transformers import AutoTokenizer

    root = Path(__file__).resolve().parents[1] / "models"
    tok_dir = next((p.parent for p in root.glob("*/tokenizer/tokenizer_config.json")), None)
    if tok_dir is None:
        pytest.skip("no local tokenizer")
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    good = ISO.resolve_subject("ds", "gwelt", ["gwelt, a man"])
    ids = tok(good.fused, add_special_tokens=False)["input_ids"]
    noun = tok("man", add_special_tokens=False)["input_ids"]
    assert (ISO.fusion_warning(tok, good) is None) == (ids[-len(noun):] == noun)
    style = ISO.resolve_subject("ds", "papercut", ["x"], kind="style")
    assert ISO.fusion_warning(tok, style) is None


def test_config_datasets_carry_the_subject():
    cfg = TrainConfig(datasets=[{"path": "a", "trigger": "gwelt", "class": "man"}, {"path": "b", "kind": "style"}])
    a, b = cfg.dataset_specs()
    assert (a.trigger, a.subject_class, a.subject_kind) == ("gwelt", "man", "auto")
    assert (b.trigger, b.subject_kind) == ("", "style")
    TrainConfig(isolation=True, trigger_word="gwelt").validate()
    with pytest.raises(ValueError, match="finetune"):
        TrainConfig(isolation=True, finetune=True, learning_rate=1e-5).validate()
    with pytest.raises(ValueError, match="dop_every"):
        TrainConfig(isolation=True, dop_every=0).validate()


def test_trainer_prepares_subjects_pairs_and_dop(tmp_path, monkeypatch):
    from rulztrainer import dataset as D

    roots = {}
    for name, trig, n in (("anna", "zwx", 2), ("gwelt", "gwelt", 4)):
        root = tmp_path / name
        root.mkdir()
        for i in range(n):
            Image.new("RGB", (96, 128), (10 * i, 90, 140)).save(root / f"{i}.png")
            (root / f"{i}.txt").write_text(f"a {'woman' if name == 'anna' else 'man'} at a cafe {i}", encoding="utf-8")
        roots[name] = (root, trig)
    cfg = TrainConfig(datasets=[{"path": str(roots["anna"][0]), "trigger": "zwx"},
                                {"path": str(roots["gwelt"][0]), "trigger": "gwelt"}],
                      cache_dir=str(tmp_path / "cache"), isolation=True, isolation_pairs=3, resolutions=[0.25],
                      sample_prompts=["gwelt at the beach", "zwx and gwelt"], probe_face_prompt="gwelt, a close-up")
    items = D.scan_dataset(cfg)
    assert items[0].caption.startswith("zwx, a woman")  # each dataset's own trigger
    people = _strangers(tmp_path)
    monkeypatch.setattr(ISO, "strangers", lambda models_dir: people)
    tr = Trainer.__new__(Trainer)
    tr.face_analysis = {}
    tr.isolation = None
    items = tr._isolation_prepare(cfg, items)
    subj = {s.trigger: s for s in tr.isolation}
    assert subj["zwx"].fused == "zwxwoman" and subj["gwelt"].fused == "gweltman"
    imgs = [it for it in items if it.category == "image"]
    assert all(it.caption.startswith(("zwxwoman", "gweltman")) for it in imgs)
    assert {it.dop_caption.split(" at ")[0] for it in imgs} == {"a woman", "a man"}
    assert all(it.text_path == D.text_cache_path(cfg, it.caption) for it in items)
    anna = [it for it in imgs if "anna" in it.dataset]
    assert anna[0].repeats == 2 and [it for it in imgs if "gwelt" in it.dataset][0].repeats == 1  # balanced
    pairs = [it for it in items if it.category == "pair"]
    assert len(pairs) == 6 and all(it.loss_scale == 2.0 and it.latent_flip_path is None for it in pairs)
    assert all(it.dop_caption and "zwxwoman" not in it.dop_caption and "gweltman" not in it.dop_caption for it in pairs)
    assert cfg.sample_prompts == ["gweltman at the beach", "zwxwoman and gweltman"]
    assert cfg.probe_face_prompt == "gweltman, a close-up"
    jobs, _ = D.text_jobs(items, cfg)
    assert {it.dop_text_path for it in imgs} <= set(jobs)  # the trigger-free captions get cached


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(4, 4)

    def forward(self, x, t, emb, mask):
        return self.proj(x.float().permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + emb.float().mean() * 0


def test_dop_pass_pulls_the_lora_toward_nothing():
    model = Tiny()
    inj = L.inject_lora(model, 2, 2.0, [], names=["proj"])
    with torch.no_grad():
        inj["proj"].lora_up.weight.normal_()  # a LoRA that changes the output
    tr = Trainer.__new__(Trainer)
    tr.dop_values = []
    cfg = SimpleNamespace(dop_weight=2.0)
    lat = torch.randn(2, 4, 3, 3)
    dop = ([1], torch.zeros(1, 3, 8), torch.ones(1, 3, dtype=torch.bool), torch.tensor([0.5]), torch.randn(1, 4, 3, 3))
    tr._dop_pass(model, inj, "cpu", lat, 2, dop, 1, cfg)
    assert tr.dop_values and tr.dop_values[0] > 0  # the LoRA's prediction differs from the model's own
    g = inj["proj"].lora_up.weight.grad
    assert g is not None and float(g.norm()) > 0
    assert inj["proj"].multiplier == 1.0  # switched back on
    first = tr.dop_values[0]
    # a step against the gradient lowers the DOP loss
    with torch.no_grad():
        inj["proj"].lora_up.weight -= 0.1 * g
    tr.dop_values.clear()
    tr._dop_pass(model, inj, "cpu", lat, 2, dop, 1, cfg)
    assert tr.dop_values[0] < first
