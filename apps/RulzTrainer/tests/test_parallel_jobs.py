"""Splitting side jobs (samples, probes, caching, region maps) across GPUs: the shared work queue and its users."""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import dataset as D  # noqa: E402
from rulztrainer import regions as R  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.multigpu import norm_device, run_queue  # noqa: E402
from rulztrainer.trainer import RenderJob, _Board  # noqa: E402


class W:
    def __init__(self, name, delay=0.0):
        self.name, self.device, self.delay = name, "cpu", delay


def test_run_queue_keeps_order_and_uses_every_worker():
    seen = {}
    lock = threading.Lock()

    def fn(w, job):
        time.sleep(w.delay)
        with lock:
            seen.setdefault(w.name, []).append(job)
        return job * 10

    workers = [W("fast", 0.01), W("slow", 0.05)]
    out = run_queue(workers, list(range(12)), fn)
    assert out == [j * 10 for j in range(12)]
    assert set(seen) == {"fast", "slow"}
    assert len(seen["fast"]) > len(seen["slow"])  # the faster worker pulls more jobs
    assert sorted(seen["fast"] + seen["slow"]) == list(range(12))


def test_run_queue_two_jobs_two_workers_run_concurrently():
    barrier = threading.Barrier(2, timeout=5)

    def fn(w, job):
        barrier.wait()  # deadlocks (BrokenBarrierError) unless both jobs run at the same time
        return w.name

    out = run_queue([W("gpu0"), W("gpu1")], ["face", "body"], fn)
    assert sorted(out) == ["gpu0", "gpu1"]


def test_run_queue_serial_is_round_robin_and_on_done_runs_in_caller_thread():
    main = threading.get_ident()
    calls = []

    def on_done(i, job, res):
        assert threading.get_ident() == main
        calls.append((i, res))

    out = run_queue([W("a"), W("b")], [0, 1, 2], lambda w, j: w.name, serial=True, on_done=on_done)
    assert out == ["a", "b", "a"]
    assert calls == [(0, "a"), (1, "b"), (2, "a")]
    calls.clear()
    run_queue([W("a"), W("b")], [0, 1, 2, 3], lambda w, j: j, on_done=on_done)
    assert sorted(calls) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_run_queue_propagates_errors_and_stops():
    started = []

    def fn(w, job):
        started.append(job)
        if job == 1:
            raise RuntimeError("boom")
        time.sleep(0.02)
        return job

    with pytest.raises(RuntimeError, match="boom"):
        run_queue([W("a"), W("b")], list(range(50)), fn)
    assert len(started) < 50  # the remaining jobs were not started after the failure
    assert run_queue([W("a")], [], fn) == []


def test_norm_device():
    assert norm_device("cpu") == "cpu"
    assert norm_device(torch.device("cpu")) == "cpu"
    if torch.cuda.is_available():
        assert norm_device("cuda") == f"cuda:{torch.cuda.current_device()}"
        assert norm_device("cuda:0") == "cuda:0"


def test_board_combines_per_gpu_progress():
    class S:
        def __init__(self):
            self.stage = None

        def update(self, stage):
            self.stage = stage

    st = S()
    b = _Board(st, "probes at step 25 on 2 GPUs")
    b.set("cuda:1", "probe body seed 1/1 | step 3/8")
    b.set("cuda:0", "probe face seed 1/1 | step 5/8")
    assert st.stage == ("probes at step 25 on 2 GPUs | cuda:0: probe face seed 1/1 | step 5/8 | "
                        "cuda:1: probe body seed 1/1 | step 3/8")
    single = _Board(st, "")
    single.set("cuda:0", "sampling 1/2 at step 50 | step 4/20")
    assert st.stage == "sampling 1/2 at step 50 | step 4/20"  # one GPU: the old single-line format


def test_render_job_shape_key():
    j = RenderJob("p", 1024, 768, 8, 1.0, 0, Path("x.png"), "probe face")
    assert j.shape_key == (1024, 768, False)
    assert RenderJob("p", 1024, 768, 8, 3.0, 0, Path("x.png"), "s").shape_key == (1024, 768, True)


def _items(tmp_path, n, flip=True):
    items = []
    for i in range(n):
        img = tmp_path / f"img{i}.png"
        Image.fromarray(np.full((64, 64, 3), i * 10, np.uint8)).save(img)
        items.append(D.Item(image=img, caption=f"cap {i}", width=64, height=64,
                            latent_path=tmp_path / f"l{i}.safetensors",
                            latent_flip_path=(tmp_path / f"lf{i}.safetensors") if flip else None,
                            text_path=tmp_path / f"t{i}.safetensors", region_path=tmp_path / f"r{i}.safetensors"))
    return items


def test_cache_latents_splits_images_across_vaes(tmp_path, monkeypatch):
    from safetensors.torch import load_file

    from rulztrainer.moss import loading

    class FakeVAE:
        def __init__(self, name):
            self.name, self.device, self.used = name, "cpu", 0

    def fake_encode(vae, images):
        vae.used += 1
        time.sleep(0.01)
        return images.mean(1, keepdim=True).repeat(1, 128, 1, 1)[:, :, ::16, ::16]

    monkeypatch.setattr(loading, "encode_images", fake_encode)
    cfg = TrainConfig(cache_dir=str(tmp_path))
    items = _items(tmp_path, 8)
    vaes = [FakeVAE("a"), FakeVAE("b")]
    progress = []
    D.cache_latents(items, cfg, vaes[0], progress=lambda i, n: progress.append((i, n)), vaes=vaes)
    assert all(it.latent_path.exists() and it.latent_flip_path.exists() for it in items)
    assert vaes[0].used > 0 and vaes[1].used > 0 and vaes[0].used + vaes[1].used == 16  # 8 images + 8 flips
    assert progress[-1] == (8, 8) and [p[0] for p in progress] == list(range(1, 9))
    flipped = load_file(str(items[3].latent_flip_path))["latent"]
    assert flipped.shape == (128, 4, 4)
    assert D.latents_todo(items, cfg) == []


def test_cache_text_splits_prompts_across_encoders(tmp_path, monkeypatch):
    from safetensors.torch import load_file

    from rulztrainer.moss import loading

    used = {}

    def fake_encode(te, tok, text, max_len):
        used.setdefault(te, []).append(text)
        time.sleep(0.01)
        return torch.full((2, 4096), float(len(text)), dtype=torch.bfloat16)

    monkeypatch.setattr(loading, "encode_prompt", fake_encode)
    cfg = TrainConfig(cache_dir=str(tmp_path))
    (tmp_path / "text").mkdir()  # scan_dataset creates it in a real run
    items = _items(tmp_path, 6)
    D.cache_text(items, cfg, "te0", "tok0", extra_prompts=["a probe prompt"],
                 encoders=[("te0", "tok0", "cpu"), ("te1", "tok1", "cpu")])
    assert set(used) == {"te0", "te1"}
    assert sum(len(v) for v in used.values()) == 6 + 2  # captions + empty prompt + probe prompt
    assert load_file(str(items[2].text_path))["embeds"][0, 0].item() == float(len("cap 2"))
    assert D.text_jobs(items, cfg, ["a probe prompt"])[0] == {}


def test_cache_region_maps_splits_images_across_mask_fns(tmp_path):
    cfg = TrainConfig(cache_dir=str(tmp_path), region_weights=True)
    items = _items(tmp_path, 6, flip=False)
    calls = {"a": 0, "b": 0}
    lock = threading.Lock()

    def mask_fn(name):
        def fn(img):
            with lock:
                calls[name] += 1
            time.sleep(0.01)
            m = np.zeros((img.height, img.width), bool)
            m[16:48, 16:48] = True
            return m
        return fn

    counts = R.cache_region_maps(items, cfg, {}, person_mask_fns=[(mask_fn("a"), "cpu"), (mask_fn("b"), "cpu")])
    assert counts == {"face": 0, "person": 6, "none": 0}
    assert calls["a"] > 0 and calls["b"] > 0 and calls["a"] + calls["b"] == 6
    assert all(it.region_path.exists() for it in items)
    # the single-function path still works and skips cached maps
    assert R.cache_region_maps(items, cfg, {}, person_mask_fn=mask_fn("a")) == {"face": 0, "person": 0, "none": 0}


def test_side_devices_follow_multi_gpu_and_switch():
    from rulztrainer.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t.devices, t.device = ["cuda:0", "cuda:1"], torch.device("cuda:0")
    t.cfg = TrainConfig(multi_gpu="data_parallel", parallel_jobs=True)
    assert t.side_devices() == ["cuda:0", "cuda:1"]
    t.cfg = TrainConfig(multi_gpu="model_split", parallel_jobs=True)
    assert t.side_devices() == ["cuda:0", "cuda:1"]
    t.cfg = TrainConfig(multi_gpu="data_parallel", parallel_jobs=False)
    assert t.side_devices() == ["cuda:0"]
    t.cfg = TrainConfig(multi_gpu="off", parallel_jobs=True)
    assert t.side_devices() == ["cuda:0"]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_text_encoder_copy_only_when_it_pays(monkeypatch):
    from rulztrainer import dataset as Dm
    from rulztrainer import trainer as T
    from rulztrainer.moss import loading

    t = T.Trainer.__new__(T.Trainer)
    t.cfg, t.model_dir = TrainConfig(), None
    from rulztrainer.archs import get_arch

    t.arch = get_arch(t.cfg)  # set by Trainer.__init__, which this test skips

    class St:
        def update(self, **kw):
            pass

    t.status = St()
    te = torch.nn.Linear(4, 4).to("cuda:0")
    per_prompt = 0.02

    def fake_cache_text(items, cfg, te_, tok, extra_prompts=None, progress=None, max_prompts=None, **kw):
        for i in range(max_prompts):
            time.sleep(0.3 if i == 0 else per_prompt)  # warm-up, then steady prompts
            progress(i + 1, max_prompts)

    loads = []
    monkeypatch.setattr(T, "cache_text", fake_cache_text)
    monkeypatch.setattr(loading, "get_text_encoder", lambda cfg, d: loads.append(cfg.text_encoder_device) or ("te2", "tok2"))
    for n_prompts, load_s, expect in ((20, 5.0, 1), (2000, 0.1, 2)):
        loads.clear()
        monkeypatch.setattr(Dm, "text_jobs", lambda items, cfg, p, n=n_prompts: ({i: "x" for i in range(n)}, n))
        out = t._extra_text_encoders(te, "tok", [], [], ["cuda:0", "cuda:1"], load_s)
        assert len(out) == expect, (n_prompts, out)
        if expect == 2:
            assert loads == ["cuda:1"] and out[1] == ("te2", "tok2", "cuda:1")
    # a copy is never made from an encoder spread over the CPU
    assert t._extra_text_encoders(torch.nn.Linear(4, 4), "tok", [], [], ["cuda:0", "cuda:1"], 0.1)[0][2] == "cpu"
