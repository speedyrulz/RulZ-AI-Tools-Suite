"""What keeps a long run's work: whole files, caches that follow the pictures, steps with broken gradients left out, a
failed run's last save, renders that may fail, and a resume that carries on deciding as the run would have. The
end-to-end tests drive the real training loop (Trainer.run) on a tiny MOSS model on the CPU, with the caches written
ahead and the renders stubbed."""
import dataclasses
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from rulztrainer import archs
from rulztrainer import atomic
from rulztrainer import dataset as D
from rulztrainer import trainer as T
from rulztrainer.config import TrainConfig
from rulztrainer.likeness import LikenessController
from rulztrainer.moss.model import MOSSImageTransformer

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=2, text_dim=32, freq_dim=16, latent_channels=128)


# ---------------------------------------------------------------- whole files


def test_a_failed_write_keeps_the_old_file_and_leaves_no_temporary(tmp_path):
    path = tmp_path / "lora.safetensors"
    atomic.save_safetensors({"a": torch.ones(2)}, path)
    with pytest.raises(RuntimeError):
        with atomic.writing(path) as tmp:
            tmp.write_bytes(b"half a file")
            raise RuntimeError("disk full")
    assert torch.equal(load_file(str(path))["a"], torch.ones(2))
    assert [p.name for p in tmp_path.iterdir()] == ["lora.safetensors"]


def test_save_lora_replaces_the_file_in_one_move(tmp_path, monkeypatch):
    from rulztrainer import lora as L

    model = MOSSImageTransformer(**TINY).to(torch.bfloat16)
    inj = L.inject_lora(model, 2, 2.0, ["q"])
    path = tmp_path / "x_best.safetensors"
    L.save_lora(inj, path, metadata={"likeness_best": "70.00"})
    assert L.read_metadata(path)["likeness_best"] == "70.00"

    def torn(tensors, filename, metadata=None):
        Path(filename).write_bytes(b"torn")
        raise OSError("no space left")

    monkeypatch.setattr(atomic, "save_file", torn)
    with pytest.raises(OSError):
        L.save_lora(inj, path, metadata={"likeness_best": "80.00"})
    assert L.read_metadata(path)["likeness_best"] == "70.00"  # the last good best is still there


# ---------------------------------------------------------------- caches follow the pictures


def _picture(path: Path, colour: int) -> None:
    Image.new("RGB", (64, 64), (colour, colour, colour)).save(path)


def test_a_replaced_picture_gets_new_cache_files_and_an_unchanged_one_keeps_them(tmp_path):
    root = tmp_path / "set"
    root.mkdir()
    _picture(root / "a.png", 10)
    _picture(root / "b.png", 20)
    cfg = TrainConfig(dataset_dir=str(root), resolution=64, output_dir=str(tmp_path / "out"))
    before = {it.image.name: it for it in D.scan_dataset(cfg)}
    assert {it.latent_path for it in D.scan_dataset(cfg)} == {it.latent_path for it in before.values()}
    (root / "a.png").unlink()
    _picture(root / "a.png", 200)  # the same name, size and bucket: only the file's own stamp tells
    import os

    st = (root / "a.png").stat()
    os.utime(root / "a.png", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    after = {it.image.name: it for it in D.scan_dataset(cfg)}
    assert after["a.png"].latent_path != before["a.png"].latent_path
    assert after["a.png"].region_path != before["a.png"].region_path
    assert after["b.png"].latent_path == before["b.png"].latent_path
    assert after["a.png"].key == before["a.png"].key  # the loss watch and the weights still know the image


def test_a_replaced_picture_gets_new_crop_cache_files(tmp_path):
    root = tmp_path / "set"
    root.mkdir()
    img = root / "a.png"
    Image.new("RGB", (512, 512)).save(img)
    cfg = TrainConfig(dataset_dir=str(root), resolution=512, output_dir=str(tmp_path / "out"), crop_augment="face",
                      crop_augment_min_px=8)
    items = D.scan_dataset(cfg)
    rec = {"W": 512, "H": 512, "bbox": [200, 200, 240, 240], "det": 0.9}

    def crop_latent():
        out = D.augment_items(items, cfg, {str(img.resolve()): rec})
        return next(it.latent_path for it in out if it.category == "face_crop")

    first = crop_latent()
    Image.new("RGB", (512, 512), (9, 9, 9)).save(img)
    import os

    st = img.stat()
    os.utime(img, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert crop_latent() != first


# ---------------------------------------------------------------- broken gradients


def test_only_broken_gradients_count_as_bad():
    p = torch.nn.Parameter(torch.zeros(2))
    assert not T.bad_gradients([p], None)  # no gradients at all: nothing to leave out
    p.grad = torch.tensor([float("nan"), 1.0])
    assert T.bad_gradients([p], T.step_grad_norm([p], 1.0))
    p.grad = torch.ones(2)
    assert not T.bad_gradients([p], T.step_grad_norm([p], 1.0))


def test_the_loss_watch_ignores_a_broken_loss(tmp_path):
    from rulztrainer.losswatch import LossWatch

    item = D.Item(image=tmp_path / "a.png", caption="c", width=64, height=64, latent_path=tmp_path / "l",
                  latent_flip_path=None, text_path=tmp_path / "t")
    watch = LossWatch(TrainConfig(), [item], tmp_path, tmp_path / "excluded.json")
    watch.observe(1, item.key, 0.5, 0.5)
    watch.observe(2, item.key, float("nan"), 0.5)
    watch.observe(3, item.key, 0.5, 0.5)
    assert watch.expected_loss(watch.n_buckets // 2) == pytest.approx(0.5, rel=0.1)


# ---------------------------------------------------------------- the likeness controller over a resume


def _controller():
    return LikenessController(TrainConfig(likeness_patience=3, likeness_lr_control=True, face_refs=["r.png"]),
                              face_ceiling=70.0, body_ceiling=80.0)


def test_a_restored_controller_decides_as_the_uninterrupted_one():
    scores = [(0, 20.0, 30.0), (100, 45.0, 50.0), (200, 60.0, 62.0), (300, 58.0, 61.0), (400, 57.0, 60.0),
              (500, 59.0, 60.0), (600, 56.0, 59.0)]
    whole, first = _controller(), _controller()
    for s in scores[:3]:
        whole.update(*s)
        first.update(*s)
    saved = torch.load(_saved(first), weights_only=False)  # through a file, as a training state goes
    resumed = _controller()
    resumed.load_resume_state(saved)
    for s in scores[3:]:
        a, acts_a = whole.update(*s)
        b, acts_b = resumed.update(*s)
        assert dataclasses.asdict(a) == dataclasses.asdict(b)
        assert acts_a == acts_b
    assert resumed.floor == whole.floor and resumed.floor["face"] == 20.0  # the step-0 probe's scale survives


def _saved(ctrl):
    import io

    buf = io.BytesIO()
    torch.save(ctrl.resume_state(), buf)
    buf.seek(0)
    return buf


def test_a_best_on_disk_is_only_replaced_by_a_higher_score():
    ctrl = _controller()
    ctrl.update(100, 60.0, 60.0)
    high = ctrl.best + 10
    ctrl.note_saved_best(high, 300)
    ctrl.note_saved_best(high - 50, 400)
    assert (ctrl.best, ctrl.best_step) == (high, 300)
    ctrl.note_saved_best(70.0, 300, "face")
    ctrl.note_saved_best(20.0, 400, "face")
    assert ctrl.part_best["face"] == (70.0, 300)


# ---------------------------------------------------------------- the real training loop


class TinyVAE(torch.nn.Module):
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """(cfg, faults): a run on a tiny model. faults maps the n-th loss computed to "nan", "crash"; scores maps a probe
    step to its (face, body)."""
    root = tmp_path / "set"
    root.mkdir()
    for i in range(3):
        _picture(root / f"{i}.png", 40 * i)
        (root / f"{i}.txt").write_text(f"a photo number {i}", encoding="utf-8")
    cfg = TrainConfig(dataset_dir=str(root), output_dir=str(tmp_path / "out"), job_name="job", device="cpu",
                      resolution=64, min_bucket_side=32, steps=6, save_every=0, sample_every=0, lora_rank=2,
                      lora_alpha=2.0, lora_targets=["q"], optimizer="adamw", learning_rate=1e-3, warmup_steps=0,
                      compile_model=False, caption_dropout=0.0, save_state=False)
    arch = archs.arch_by_name("moss_image")
    for name, fn in (("resolve_for_config", lambda c: tmp_path),
                     ("get_transformer", lambda c, d, device=None: _tiny()),
                     ("get_vae", lambda c, d: TinyVAE()),
                     ("sample", lambda *a, **k: torch.zeros(1, 128, 4, 4)),
                     ("decode_latents", lambda v, lat: torch.zeros(1, 3, 32, 32))):
        monkeypatch.setattr(arch, name, fn)
    monkeypatch.setattr("rulztrainer.comfy_files.resolve_downloads", lambda c, progress=None: c)
    monkeypatch.setattr(T, "detect_arch", lambda c, d: arch)
    monkeypatch.setattr(T.Trainer, "_save_history", lambda self, state="running": None)
    faults, scores = {}, {}
    calls = [0]
    real_loss = T.elementwise_loss

    def loss(*a, **k):
        calls[0] += 1
        fault = faults.get(calls[0])
        if fault == "crash":
            raise RuntimeError("the GPU fell off")
        out = real_loss(*a, **k)
        return out * float("nan") if fault == "nan" else out

    monkeypatch.setattr(T, "elementwise_loss", loss)

    def run_probe(self, model, step, optimizer, scheduler, injected):
        return self._probe_finish(step, *scores[step], [], None, "0s", optimizer, scheduler, injected)

    monkeypatch.setattr(T.Trainer, "_run_probe", run_probe)
    monkeypatch.setattr(T.Trainer, "_make_controller", lambda self, items: LikenessController(
        self.cfg, face_ceiling=70.0, body_ceiling=80.0) if self.cfg.probe_every > 0 else None)
    return cfg, faults, scores


def _tiny():
    torch.manual_seed(0)
    return MOSSImageTransformer(**TINY).to(torch.bfloat16)


def _prepare(cfg):
    """Everything the run would encode, written ahead: the run needs no VAE or text encoder."""
    for it in D.scan_dataset(cfg):
        it.latent_path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"latent": torch.randn(128, it.height // 16, it.width // 16)}, str(it.latent_path))
        save_file({"embeds": torch.randn(5, 32).to(torch.bfloat16)}, str(it.text_path))
    negative = T.Trainer.negative_prompt(SimpleNamespace(cfg=cfg, arch=archs.arch_by_name("moss_image")))
    for p in (D.dropout_prompt(cfg), *cfg.sample_prompts, negative, cfg.probe_face_prompt):
        path = D.text_cache_path(cfg, p or "")
        if not path.exists():
            save_file({"embeds": torch.randn(5, 32).to(torch.bfloat16)}, str(path))


def _job(cfg) -> Path:
    return Path(cfg.output_dir) / cfg.job_name


def _status(cfg) -> dict:
    return json.loads((_job(cfg) / "status.json").read_text(encoding="utf-8"))


def _finite(path) -> bool:
    return all(torch.isfinite(v).all() for v in load_file(str(path)).values())


def _go(cfg):
    _prepare(cfg)
    tr = T.Trainer(cfg)
    tr.run()
    return tr


def test_a_step_with_broken_gradients_leaves_the_lora_healthy(rig):
    cfg, faults, _ = rig
    faults[3] = "nan"
    tr = _go(cfg)
    assert tr.skipped_steps == 1
    st = _status(cfg)
    assert st["stage"] == "finished" and st["step"] == 6 and st["skipped_steps"] == 1
    assert all(v == v for v in st["losses"])  # no NaN for the chart
    assert _finite(_job(cfg) / "job.safetensors")
    assert any(v.abs().sum() > 0 for v in load_file(str(_job(cfg) / "job.safetensors")).values())


def test_a_diverging_run_stops_and_keeps_its_healthy_lora(rig):
    cfg, faults, _ = rig
    cfg.steps = T.BAD_STEP_LIMIT + 10
    faults.update({n: "nan" for n in range(3, 200)})
    with pytest.raises(RuntimeError, match="not finite"):
        _go(cfg)
    st = _status(cfg)
    assert st["stage"] == "error" and "not finite" in st["error"]
    step = 1 + T.BAD_STEP_LIMIT  # the last bad step never finished
    assert _finite(_job(cfg) / f"job_{step:06d}.safetensors")
    assert (_job(cfg) / f"state_{step:06d}" / "train_state.pt").exists()


def test_a_crash_saves_the_lora_and_a_state_that_resumes(rig):
    cfg, faults, _ = rig
    faults[5] = "crash"
    with pytest.raises(RuntimeError, match="fell off"):
        _go(cfg)
    assert _status(cfg)["stage"] == "error"
    state = _job(cfg) / "state_000004"  # the failing step never finished
    assert _finite(_job(cfg) / "job_000004.safetensors") and (state / "train_state.pt").exists()
    faults.clear()
    resumed = dataclasses.replace(cfg, resume_state=str(state))
    _go(resumed)
    st = _status(resumed)
    assert st["stage"] == "finished" and st["step"] == 6
    assert _finite(_job(cfg) / "job.safetensors")


def test_a_crash_before_the_first_step_saves_nothing(rig):
    cfg, faults, _ = rig
    faults[1] = "crash"
    with pytest.raises(RuntimeError):
        _go(cfg)
    assert not list(_job(cfg).glob("*.safetensors")) and not list(_job(cfg).glob("state_*"))


def test_broken_weights_are_not_saved_over_good_files(rig):
    cfg, _, _ = rig
    _prepare(cfg)
    tr = T.Trainer(cfg)
    broken = torch.nn.Parameter(torch.tensor([float("nan")]))
    tr._emergency_save({}, [broken], 3)
    assert not list(_job(cfg).glob("*.safetensors"))


def test_a_failed_render_costs_that_round_and_training_goes_on(rig, monkeypatch):
    cfg, _, scores = rig
    cfg.sample_prompts, cfg.sample_every = ["a cat"], 2
    cfg.sample_before_training = False
    cfg.probe_every, cfg.probe_face_prompt, cfg.probe_at_start, cfg.likeness_patience = 2, "a face", False, 0
    scores.update({2: (50.0, None), 6: (60.0, None)})  # step 4's probe is not there: a KeyError in it

    def sample(*a, **k):
        raise torch.OutOfMemoryError("sampling ran out")

    monkeypatch.setattr(archs.arch_by_name("moss_image"), "sample", sample)
    _go(cfg)
    st = _status(cfg)
    assert st["stage"] == "finished" and st["step"] == 6
    assert "side_job_error" in st
    assert (_job(cfg) / "job.safetensors").exists()


def test_a_resumed_run_keeps_its_best_checkpoint(rig):
    cfg, faults, scores = rig
    cfg.save_state, cfg.save_state_every, cfg.keep_states = True, 2, 0
    cfg.probe_every, cfg.probe_face_prompt, cfg.probe_at_start, cfg.likeness_patience = 2, "a face", False, 0
    cfg.likeness_normalize = False
    cfg.steps = 8
    scores.update({2: (50.0, None), 4: (90.0, None)})
    faults[5] = "crash"  # killed after the step-4 probe set the best
    with pytest.raises(RuntimeError):
        _go(cfg)
    best = _job(cfg) / "job_best.safetensors"
    high = float(_meta(best)["likeness_best"])
    stamp = best.read_bytes()
    faults.clear()
    scores.clear()
    scores.update({4: (40.0, None), 6: (40.0, None), 8: (40.0, None)})  # worse than the best on disk
    resumed = dataclasses.replace(cfg, resume_state=str(_job(cfg) / "state_000002"))
    tr = _go(resumed)
    assert best.read_bytes() == stamp  # before: the first probe after a resume replaced it
    assert tr.controller.best == pytest.approx(high)
    assert tr.controller.n_probes == 1 + 3  # the state's probe and the three here


def _meta(path) -> dict:
    with safe_open(str(path), "pt") as f:
        return f.metadata()


def test_a_state_folder_without_its_last_file_is_not_a_state(tmp_path):
    from rulztrainer.config import TrainConfig as C

    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = C()
    state = tmp_path / "state_000010"
    state.mkdir()
    (state / "lora.safetensors").write_bytes(b"")
    random.seed(0)
    with pytest.raises(Exception):
        tr.load_state(state, {}, None, None, None, None)
