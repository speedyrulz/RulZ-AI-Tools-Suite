"""Likeness-filtered self-training: the rounds' steps, the cap, the quality filter and the bar, and a round in the
trainer replacing the last round's pictures."""
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rulztrainer import selftrain as ST
from rulztrainer.config import TrainConfig


def noise(path, seed, size=(512, 512), high=256):
    arr = np.random.default_rng(seed).integers(0, high, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return str(path)


def smooth(path, size=(512, 512)):
    ramp = np.linspace(0, 255, size[0], dtype=np.float32)[None, :, None].repeat(size[1], 0).repeat(3, 2)
    Image.fromarray(ramp.astype(np.uint8)).save(path)
    return str(path)


def test_round_steps():
    assert ST.round_steps(1000, "0.5") == [500]
    assert ST.round_steps(1000, "0.75, 0.5;0.5") == [500, 750]
    assert ST.round_steps(1, "0.5") == []  # no step inside the run
    with pytest.raises(ValueError, match="fractions"):
        ST.round_steps(1000, "1.5")


def test_cap_counts_each_real_image_once_with_its_repeats():
    items = [SimpleNamespace(image=f"{i}.png", category="image", repeats=2) for i in range(10)]
    items += [SimpleNamespace(image="0.png", category="image", repeats=2)]  # a second resolution of the same photo
    items += [SimpleNamespace(image="0.png", category="face_crop", repeats=2),
              SimpleNamespace(image="g.png", category="selftrain", repeats=1)]
    assert ST.cap_count(items, 0.2) == 4  # 20 real images per epoch


def test_sharpness_and_duplicates(tmp_path):
    a, b, c = noise(tmp_path / "a.png", 1), noise(tmp_path / "b.png", 2), smooth(tmp_path / "c.png")
    assert ST.sharpness(a) > 10 * ST.sharpness(c)
    # a large face is judged alone: a sharp face on a soft background passes, a small one goes with the picture
    arr = np.full((512, 512, 3), 128, np.uint8)
    yy, xx = np.mgrid[0:200, 0:200]
    arr[150:350, 150:350] = (((yy // 8 + xx // 8) % 2) * 255)[..., None].astype(np.uint8)  # crisp edges, any scale
    Image.fromarray(arr).save(tmp_path / "portrait.png")
    face = [165, 165, 335, 335]
    assert ST.big_face(face, (512, 512)) and not ST.big_face([0, 0, 40, 50], (512, 512))
    assert not ST.big_face(face, (2048, 2048), side=512)  # a 170 px face in a 2048 px photo: 42 px at training size
    assert ST.sharpness(tmp_path / "portrait.png", face) > 2 * ST.sharpness(tmp_path / "portrait.png")
    # renders are judged against the round's own: far below its median is blurry; too few to judge: no check
    cands = [ST.Candidate(str(i), "p", 50.0, 1, 0.9, sharp=s) for i, s in enumerate((1.0, 1.1, 0.9, 0.2))]
    assert ST.sharp_floor(cands) == (0.0, pytest.approx(0.4 * 0.95))
    kept, reasons = ST.select(cands, need_face=True, bar=0, cap=10)
    assert reasons == {"blurry": 1} and cands[3].verdict == "blurry"
    assert ST.sharp_floor(cands[:2]) == (0.0, 0.0)
    assert float(ST.thumb(a) @ ST.thumb(a)) > 0.999
    assert float(ST.thumb(a) @ ST.thumb(b)) < ST.DUP_MAX


def test_select_filters_then_keeps_the_best_up_to_the_cap():
    def cand(name, likeness, faces=1, det=0.9, cut=False, sharp=1.0, vec=None):
        v = np.zeros(4, dtype=np.float32)
        v[vec if vec is not None else 0] = 1.0
        return ST.Candidate(name, "p", likeness, faces, det, cut, sharp, thumb=v if vec is not None else None)

    cands = [cand("low", 50), cand("two", 95, faces=2), cand("none", 95, faces=0), cand("unsure", 95, det=0.3),
             cand("cut", 95, cut=True), cand("blur", 95, sharp=0.01), cand("best", 90, vec=0), cand("dup", 85, vec=0),
             cand("second", 80, vec=1), cand("third", 75, vec=2)]
    kept, reasons = ST.select(cands, need_face=True, sharp_min=0.5, bar=60, cap=2)
    assert [c.path for c in kept] == ["best", "second"]
    assert {c.path: c.verdict for c in cands} == {
        "low": "below the bar", "two": "several faces", "none": "no face", "unsure": "unsure face",
        "cut": "face cut off", "blur": "blurry", "best": "kept", "dup": "duplicate", "second": "kept",
        "third": "over the cap"}
    assert ST.describe(reasons).startswith("1 no face, 1 several faces")
    # a body-only run does not need a face
    kept, _ = ST.select([cand("bodyonly", 70, faces=0)], need_face=False, sharp_min=0.5, bar=60, cap=1)
    assert len(kept) == 1


def test_config_checks():
    TrainConfig(selftrain=True, face_refs=["x"]).validate()
    # a race may self-train, per configuration (not one configuration per GPU)
    TrainConfig(face_refs=["x"], race=True, probe_every=10, steps=40,
                race_configs=[{"name": "a"}, {"name": "b", "selftrain": True}]).validate()
    for bad, match in ((dict(selftrain=True), "references"),
                       (dict(face_refs=["x"], race=True, race_parallel=True, multi_gpu="data_parallel",
                             devices="cuda:0,cuda:1", probe_every=10, steps=40,
                             race_configs=[{"name": "a"}, {"name": "b", "selftrain": "true"}]), "self-training"),
                       (dict(selftrain=True, face_refs=["x"], selftrain_share=0), "selftrain_share"),
                       (dict(selftrain=True, face_refs=["x"], selftrain_at="2"), "fractions")):
        with pytest.raises(ValueError, match=match):
            TrainConfig(**bad).validate()


def test_a_round_keeps_the_best_clean_pictures_in_place_of_the_last(tmp_path, monkeypatch):
    from rulztrainer import archs
    from rulztrainer import trainer as T
    from rulztrainer.dataset import Item, text_cache_path

    monkeypatch.setattr(T, "cache_latents", lambda items, cfg, vae: None)
    cfg = TrainConfig(dataset_dir=str(tmp_path / "ds"), steps=100, resolution=512, selftrain=True, face_refs=["x"],
                      face_weight=1.0, body_weight=0.0, selftrain_images=6, selftrain_share=0.2,
                      selftrain_prompts=["photo of zk smiling", "photo of zk in a cafe"], model_arch="krea2")
    (tmp_path / "ds").mkdir()

    def item(path, category="image", **kw):
        return Item(image=path, caption="zk", width=512, height=512, latent_path=tmp_path / "l", latent_flip_path=None,
                    text_path=tmp_path / "t", category=category, side=512, **kw)

    real = [item(T.Path(noise(tmp_path / "ds" / f"{i}.png", 100 + i, high=128))) for i in range(10)]  # less sharp
    old = item(T.Path(noise(tmp_path / "old.png", 99)), category="selftrain", loss_scale=0.5)

    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg, tr.arch, tr.job_dir = cfg, archs.arch_by_name("krea2"), tmp_path / "job"
    tr.job_dir.mkdir()
    tr.status = SimpleNamespace(update=lambda **k: None, data={})
    tr.all_items, tr.sides, tr.res_stage = real + [old], [512], 0
    tr.balancer = tr.loss_watch = tr.curriculum_w = tr.vae = tr.body_scorer = None
    tr.face_analysis, tr.trigger_pos, tr.race = {}, {}, None
    tr.controller = SimpleNamespace(combined_ema=55.0, normalize=False)  # the bar: max(60, 55)

    out = tr.job_dir / "selftrain"
    # picture: (likeness %, faces, blurry?) ; 1 repeats 0; 4 is good but over the cap of 2 (20% of 10)
    plan = {0: (80, 1, False), 1: (75, 1, False), 2: (70, 1, False), 3: (90, 1, True), 4: (65, 1, False),
            5: (95, 2, False)}

    def render(model, step, rnd, jobs):
        paths = []
        for i, job in enumerate(jobs):
            if i == 1:
                shutil.copy(paths[0], job.out_path)
                paths.append(str(job.out_path))
            elif plan[i][2]:
                paths.append(smooth(job.out_path))
            else:
                paths.append(noise(job.out_path, i))
        assert (jobs[0].width, jobs[0].height, jobs[0].steps) == (512, 512, cfg.sample_steps)
        return paths

    class Faces:
        def _i(self, path):
            return int(T.Path(path).stem.split("_")[1])

        def faces(self, path):
            if not T.Path(path).stem.startswith("round"):
                return []  # the photos (for the sharpness floor)
            n = plan[self._i(path)][1]
            return [SimpleNamespace(bbox=[100, 100, 200, 220], det_score=0.9)] * n

        def score(self, path):
            return SimpleNamespace(percent=float(plan[self._i(path)][0]))

    tr.face_scorer = Faces()
    tr._selftrain_render = render
    tr._self_train_round(None, 50, 1)

    import json

    rec = json.loads((out / "selftrain.json").read_text())
    verdicts = [c["verdict"] for c in rec[0]["candidates"]]
    assert verdicts == ["kept", "duplicate", "kept", "blurry", "over the cap", "several faces"]
    assert rec[0]["bar"] == 60
    added = [it for it in tr.all_items if it.category == "selftrain"]
    assert [it.image.name for it in added] == ["round1_00.png", "round1_02.png"]  # the last round's is gone
    assert len(tr.all_items) == 12 and all(it.loss_scale == 0.5 for it in added)
    assert added[0].caption == "photo of zk smiling" and added[0].text_path == text_cache_path(cfg, added[0].caption)
    assert old not in tr.batcher.items and added[0] in tr.batcher.items
    assert ST.last_kept(out / "selftrain.json", 60) == (1, [(str(added[0].image), added[0].caption),
                                                         (str(added[1].image), added[1].caption)])
    assert ST.last_kept(out / "selftrain.json", 40) is None
