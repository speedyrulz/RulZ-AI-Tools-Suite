"""Remote render, the instance side: the shared hashes and folder helpers, the round state machine against a fake helper
(claim / done / failed / abandon races, deadlines, a stale heartbeat, the ready file), and the trainer acting on rounds
(a late probe rendered from the retained weights, the sample queue, the shadow round). No GPU, no network, no ssh."""
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import lora as L  # noqa: E402
from rulztrainer import remoterender as RR  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402

TINY = dict(dim=64, ffn_dim=128, num_heads=2, num_layers=3, text_dim=32, freq_dim=16, latent_channels=128)


# ------------------------------------------------------------ the shared pieces
def test_render_hash_ignores_line_endings(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    (pkg / "krea2").mkdir(parents=True)
    for name in RR.RENDER_SOURCES:
        (pkg / name).write_bytes(b"x = 1\nprint(x)\n")
    (pkg / "krea2" / "model.py").write_bytes(b"a\nb\n")
    monkeypatch.setattr(RR, "PACKAGE", pkg)
    lf = RR.render_hash("krea2")
    assert len(lf) == 16
    for f in list(pkg.glob("*.py")) + [pkg / "krea2" / "model.py"]:
        f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
    assert RR.render_hash("krea2") == lf  # a Windows checkout is CRLF
    (pkg / "krea2" / "model.py").write_bytes(b"a\nc\n")
    assert RR.render_hash("krea2") != lf  # the family's own folder counts
    assert RR.render_hash("krea2") == RR.render_hash("krea2") and RR.render_hash("zimage") != RR.render_hash("krea2")


def test_render_hash_of_the_real_package_covers_the_families():
    files = RR.render_sources("qwen_image21")
    names = {p.name for p in files}
    assert {"probeworker.py", "remoterender.py", "lora.py", "quant.py", "attention.py"} <= names
    assert any(p.parent.name == "qwen21" for p in files)
    assert RR.render_hash("qwen_image21") != RR.render_hash("krea2")


def test_recipe_hash_follows_what_changes_a_picture():
    cfg = TrainConfig(dataset_dir="ds").to_dict()
    files = {"transformer_file": {"name": "a.safetensors", "size": 10, "path": "/x/a.safetensors"}}
    base = RR.recipe_hash(cfg, files)
    assert RR.recipe_hash({**cfg, "dataset_dir": "other", "learning_rate": 1.0}, files) == base  # training only
    assert RR.recipe_hash({**cfg, "lora_rank": 8}, files) != base
    assert RR.recipe_hash({**cfg, "turbo_lora_strength": 0.5}, files) != base
    assert RR.recipe_hash({**cfg, "probe_size": 768}, files) != base
    moved = {"transformer_file": {"name": "a.safetensors", "size": 10, "path": "/elsewhere/a.safetensors"}}
    assert RR.recipe_hash(cfg, moved) == base  # the path is the instance's; name and size are what must agree
    assert RR.recipe_hash(cfg, {"transformer_file": {"name": "a.safetensors", "size": 11}}) != base


def test_jobs_round_trip_with_relative_paths(tmp_path):
    from rulztrainer.trainer import RenderJob

    job_dir = tmp_path / "job"
    job = RenderJob("a prompt", 512, 768, 8, 1.0, 5, job_dir / "probes" / "p.png", "probe face seed 1/2", "face",
                    frames=5, shown="shown")
    d = RR.job_to_dict(job, job_dir)
    assert d["out_path"] == "probes/p.png" and json.loads(json.dumps(d)) == d
    back = RR.job_from_dict(d, tmp_path / "pc")
    assert back == RenderJob(**{**d, "out_path": tmp_path / "pc" / "probes" / "p.png"})
    with pytest.raises(ValueError):
        RR.job_from_dict({**d, "out_path": "../escape.png"}, tmp_path)
    with pytest.raises(ValueError):
        RR.job_from_dict({**d, "out_path": "/abs.png"}, tmp_path)


def test_atomic_rename_has_one_winner(tmp_path):
    (tmp_path / "rounds" / "a").mkdir(parents=True)
    assert RR.atomic_rename(tmp_path / "rounds" / "a", tmp_path / "claimed" / "a") is True
    assert RR.atomic_rename(tmp_path / "rounds" / "a", tmp_path / "abandoned" / "a") is False  # src is gone
    (tmp_path / "rounds" / "b").mkdir()
    (tmp_path / "claimed" / "b").mkdir()
    assert RR.atomic_rename(tmp_path / "rounds" / "b", tmp_path / "claimed" / "b") is False  # dst exists


# ------------------------------------------------------------ the round state machine
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Helper:
    """What the bridge and the PC helper do to a round's folders, with the instance's clock."""

    def __init__(self, rr, clock):
        self.rr, self.clock = rr, clock

    def beat(self, token=None):
        RR.write_json(self.rr.root / "helper.json", {"run_token": token or self.rr.run_token, "host": "pc",
                                                      "gpu": "RTX 4090", "time": self.clock()})

    def ready(self, kinds=("sample", "probe"), **over):
        d = {"run_token": self.rr.run_token, "recipe_hash": self.rr.recipe, "render_hash": self.rr.render,
             "host": "pc", "gpu": "RTX 4090", "vram_gb": 24, "load_seconds": 30, "kinds": list(kinds), "time": 1}
        RR.write_json(self.rr.root / "prepare" / "ready.json", {**d, **over})

    def claim(self, r):
        return RR.atomic_rename(RR.round_path(self.rr.root, "rounds", r.name),
                                RR.round_path(self.rr.root, "claimed", r.name))

    def finish(self, r, extra=(), seconds=12.0):
        c = RR.round_path(self.rr.root, "claimed", r.name)
        if not c.is_dir():  # the trainer took the round back: the bridge has nowhere to upload to
            return False
        names = [Path(j.out_path).relative_to(self.rr.job_dir).as_posix() for j in r.jobs] + list(extra)
        for n in names:
            (c / "out" / n).parent.mkdir(parents=True, exist_ok=True)
            (c / "out" / n).write_bytes(n.encode())
        RR.write_json(c / "done.json", {"seconds": seconds, "files": names, "host": "pc", "gpu": "RTX 4090"})
        return RR.atomic_rename(c, RR.round_path(self.rr.root, "done", r.name))

    def fail(self, r):
        c = RR.round_path(self.rr.root, "claimed", r.name)
        RR.write_json(c / "error.json", {"error": "CUDA out of memory", "traceback": "tb"})
        return RR.atomic_rename(c, RR.round_path(self.rr.root, "failed", r.name))


def _jobs(job_dir, n=2, step=10, kind="probe"):
    from rulztrainer.trainer import RenderJob

    return [RenderJob("p", 64, 64, 4, 1.0, k, job_dir / f"{kind}s" / f"step{step:06d}_face_{k}.png", f"{kind} {k}",
                      "face") for k in range(n)]


def _renderer(tmp_path, clock=None, known=("probe", "sample")):
    clock = clock or Clock()
    job_dir = tmp_path / "job"
    job_dir.mkdir(exist_ok=True)
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    (cache / "e.safetensors").write_bytes(b"embeds")
    rr = RR.RemoteRenderer(job_dir, "krea2", clock=clock, sleep=lambda s: setattr(clock, "t", clock.t + s))
    rr.start_run("my_lora", {"lora_rank": 4}, {}, {"probe": "neg"}, {"a prompt": cache / "e.safetensors"})
    for k in known:
        rr.note_own(k, 4.0, 4)
    return rr, Helper(rr, clock), clock


def _state(seed=0):
    return {"x": {"params": {"w": torch.full((2, 2), float(seed))}}}


def test_start_run_clears_the_folder_and_publishes_the_run(tmp_path):
    rr, h, clock = _renderer(tmp_path)
    run = json.loads((rr.root / "run.json").read_text())
    prep = json.loads((rr.root / "prepare" / "prepare.json").read_text())
    assert run["protocol"] == 1 and run["run_token"] == rr.run_token and run["job"] == "my_lora"
    assert run["family"] == "krea2" and run["render_hash"] == rr.render and run["recipe_hash"] == rr.recipe
    assert prep["run_token"] == rr.run_token and prep["negatives"] == {"probe": "neg"}
    emb = prep["embeds"]["a prompt"]
    assert (rr.root / "prepare" / emb).read_bytes() == b"embeds" and emb.startswith("embeds/")
    h.beat()
    r = rr.submit("probe", 10, _jobs(rr.job_dir), _state(), "probe")
    assert (rr.root / "rounds" / r.name).is_dir()
    # a resume (or a crashed run) leaves nothing a helper could render into the dead run
    old = rr.run_token
    rr.start_run("my_lora", {"lora_rank": 4}, {}, {}, {})
    assert rr.run_token != old and not (rr.root / "helper.json").exists()
    assert not any(list((rr.root / s).iterdir()) for s in RR.SUBDIRS)
    assert json.loads((rr.root / "run.json").read_text())["run_token"] == rr.run_token


def test_a_round_is_published_whole_and_carries_the_exact_state(tmp_path):
    rr, h, clock = _renderer(tmp_path)
    r = rr.submit("probe", 250, _jobs(rr.job_dir), _state(7), "probe")
    assert r.name == "000001_probe_250" and not list((rr.root / "rounds").glob(".tmp-*"))
    folder = rr.root / "rounds" / r.name
    meta = json.loads((folder / "round.json").read_text())
    assert (meta["protocol"], meta["run_token"], meta["seq"], meta["kind"], meta["step"], meta["negative_kind"]) == \
        (1, rr.run_token, 1, "probe", 250, "probe")
    assert meta["jobs"][0]["out_path"] == "probes/step000010_face_0.png" and meta["deadline_s"] >= 90
    state = RR.state_from_bytes((folder / "lora.safetensors").read_bytes())
    assert torch.equal(state["x"]["params"]["w"], torch.full((2, 2), 7.0))


def test_delegation_needs_a_ready_match_a_fresh_heartbeat_and_a_known_render_time(tmp_path):
    rr, h, clock = _renderer(tmp_path, known=())
    rr.poll()
    assert not rr.can_delegate("probe")  # nobody there
    h.beat()
    h.ready()
    rr.poll()
    assert not rr.can_delegate("probe") and not rr.can_delegate("sample")  # the instance has not timed a render yet
    rr.note_own("probe", 6.0, 3)
    assert rr.can_delegate("probe") and not rr.can_delegate("sample")
    h.ready(kinds=["sample"])  # a sampling-model override: samples only
    rr.note_own("sample", 6.0, 3)
    rr.poll()
    assert rr.can_delegate("sample") and not rr.can_delegate("probe")
    h.ready()
    rr.poll()
    assert rr.can_delegate("probe") and rr.can_delegate("sample")
    clock.t += RR.FRESH_SECONDS + 1  # a sleeping PC: the heartbeat stops
    rr.poll()
    assert not rr.can_delegate("probe")
    h.beat()
    rr.poll()
    assert rr.can_delegate("probe")
    h.beat(token="another-run")  # a heartbeat for another run is not this run's helper
    rr.poll()
    assert not rr.can_delegate("probe")


@pytest.mark.parametrize("field", ["run_token", "recipe_hash", "render_hash"])
def test_a_ready_file_that_does_not_match_delegates_nothing(tmp_path, field):
    rr, h, clock = _renderer(tmp_path)
    h.beat()
    h.ready(**{field: "wrong"})
    rr.poll()
    assert rr.ready is None and not rr.can_delegate("probe")
    h.ready()
    rr.poll()
    assert rr.can_delegate("probe")


def test_deadlines(tmp_path):
    rr, h, clock = _renderer(tmp_path, known=())
    rr.note_own("probe", 20.0, 4)  # 5 s per picture
    assert rr.deadline_for("probe", 4) == 2 * 90  # the first round after ready: twice max(90, 3 x 20 = 60)
    rr.first_round = False
    assert rr.deadline_for("probe", 4) == 90
    assert rr.deadline_for("probe", 12) == 180  # 3 x 5 s x 12
    rr.first_round = True
    h.beat()
    r = rr.submit("probe", 1, _jobs(rr.job_dir), _state())
    assert r.deadline - r.submitted == 180 and rr.first_round is False
    r2 = rr.submit("probe", 2, _jobs(rr.job_dir), _state())
    assert r2.deadline - r2.submitted == 90


def test_a_finished_round_is_taken_into_the_real_out_paths(tmp_path):
    rr, h, clock = _renderer(tmp_path)
    h.beat()
    h.ready()
    jobs = _jobs(rr.job_dir)
    r = rr.submit("probe", 10, jobs, _state())
    assert rr.collect(r) == (None, "open")
    assert h.claim(r)
    assert rr.collect(r) == (None, "open")  # claimed, not done
    assert h.finish(r, extra=["probes/step000010_face_0.mp4"])  # a clip's extra file
    got, why = rr.collect(r)
    assert why == "" and got.images == [str(j.out_path) for j in jobs] and got.seconds == 12.0
    assert got.label == "pc RTX 4090"
    assert [Path(p).read_bytes() for p in got.images] == [b"probes/step000010_face_0.png",
                                                          b"probes/step000010_face_1.png"]
    assert (rr.job_dir / "probes" / "step000010_face_0.mp4").read_bytes() == b"probes/step000010_face_0.mp4"
    assert not (rr.root / "done" / r.name).exists()  # deleted once taken
    assert rr.misses == 0 and rr.stats["rounds"] == 1


def test_an_unsafe_or_incomplete_round_is_not_taken(tmp_path):
    rr, h, clock = _renderer(tmp_path)
    h.beat()
    r = rr.submit("probe", 10, _jobs(rr.job_dir), _state())
    h.claim(r)
    h.finish(r, extra=["../escape.png"])
    got, why = rr.collect(r)
    assert got is None and "could not be taken" in why and not (rr.job_dir.parent / "escape.png").exists()
    assert rr.misses == 1
    r = rr.submit("probe", 20, _jobs(rr.job_dir, step=20), _state())  # a picture missing from the upload
    h.claim(r)
    h.finish(r)
    (rr.root / "done" / r.name / "out" / "probes" / "step000020_face_1.png").unlink()
    got, why = rr.collect(r)
    assert got is None and "lacks" in why


def test_a_failed_round_is_reported_and_counted(tmp_path):
    rr, h, clock = _renderer(tmp_path)
    h.beat()
    r = rr.submit("sample", 10, _jobs(rr.job_dir, kind="sample"), _state(), "sample")
    h.claim(r)
    assert h.fail(r)
    got, why = rr.collect(r)
    assert got is None and "CUDA out of memory" in why and rr.misses == 1 and rr.off_reason == ""
    assert not (rr.root / "failed" / r.name).exists()


def test_two_misses_in_a_row_end_delegation_and_a_success_resets_the_count(tmp_path):
    rr, h, clock = _renderer(tmp_path)
    h.beat()
    h.ready()
    rr.poll()

    def miss():
        r = rr.submit("probe", 1, _jobs(rr.job_dir), _state())
        clock.t += 1000
        h.beat()
        return rr.collect(r)

    assert miss()[0] is None and rr.misses == 1 and rr.can_delegate("probe")
    r = rr.submit("probe", 2, _jobs(rr.job_dir), _state())
    h.claim(r)
    h.finish(r)
    assert rr.collect(r)[0] is not None and rr.misses == 0
    assert miss()[0] is None and miss()[0] is None
    assert rr.off_reason and not rr.can_delegate("probe") and not rr.can_delegate("sample")


def test_the_deadline_and_a_stale_heartbeat_take_a_round_back(tmp_path):
    rr, h, clock = _renderer(tmp_path)
    h.beat()
    r = rr.submit("probe", 10, _jobs(rr.job_dir), _state())
    h.claim(r)  # the helper has it and renders
    clock.t += 100
    h.beat()
    assert rr.check(r) == "open"  # before the doubled first deadline
    clock.t += 100
    h.beat()
    assert rr.check(r) == "late" and "within" in r.why  # past it, the heartbeat fresh
    r = rr.submit("probe", 20, _jobs(rr.job_dir, step=20), _state())
    h.claim(r)
    clock.t += RR.FRESH_SECONDS + 1  # the PC went to sleep mid-round
    assert rr.check(r) == "late" and "stopped answering" in r.why
    got, why = rr.collect(r)
    assert got is None and "stopped answering" in why
    assert (rr.root / "claimed" / r.name).exists() is False and not (rr.root / "rounds" / r.name).exists()


def test_a_late_done_after_abandon_is_ignored_and_a_done_that_won_is_taken(tmp_path):
    rr, h, clock = _renderer(tmp_path)
    h.beat()
    r = rr.submit("probe", 10, _jobs(rr.job_dir), _state())
    h.claim(r)
    assert rr.reclaim(r) == "abandoned"
    assert h.finish(r) is False  # the helper's claimed -> done rename finds nothing: its late result is dropped
    assert not (rr.root / "done" / r.name).exists()
    assert rr.reclaim(r) == "abandoned"
    # the helper's done rename wins the race with the trainer's take-back
    r = rr.submit("probe", 20, _jobs(rr.job_dir, step=20), _state())
    h.claim(r)
    h.finish(r)
    assert rr.reclaim(r) == "done"
    assert rr.collect(r)[0] is not None
    # a round nobody claimed is abandoned from rounds/ as well
    r = rr.submit("probe", 30, _jobs(rr.job_dir, step=30), _state())
    assert rr.reclaim(r) == "abandoned" and h.claim(r) is False


def test_a_done_round_that_is_late_is_still_taken(tmp_path):
    rr, h, clock = _renderer(tmp_path)
    h.beat()
    r = rr.submit("probe", 10, _jobs(rr.job_dir), _state())
    h.claim(r)
    h.finish(r)
    clock.t += 10_000  # long past its deadline, and the heartbeat is stale: the pictures are there
    got, why = rr.collect(r)
    assert got is not None and why == ""


def test_waiting_ends_at_the_deadline(tmp_path):
    rr, h, clock = _renderer(tmp_path)
    h.beat()
    r = rr.submit("probe", 10, _jobs(rr.job_dir), _state())
    beats = []
    real_sleep = rr.sleep

    def sleep(s):
        real_sleep(s)
        h.beat()  # the helper is alive but never finishes
        beats.append(s)

    rr.sleep = sleep
    got, why = rr.collect(r, wait=True)
    assert got is None and "within" in why and beats and clock.t >= r.deadline


# ------------------------------------------------------------ the trainer acting on rounds
def _lora(seed):
    torch.manual_seed(seed)
    model = MOSSImageTransformer(**TINY)
    inj = L.inject_lora(model, 4, 4.0, ["q", "v"])
    for m in inj.values():
        for p in m.adapter_params().values():
            torch.nn.init.normal_(p)
    return inj


def _weights(inj):
    return {n: {k: p.detach().clone() for k, p in m.adapter_params().items()} for n, m in inj.items()}


def _same(inj, ref):
    return all(torch.equal(p, ref[n][k]) for n, m in inj.items() for k, p in m.adapter_params().items())


def _trainer(tmp_path, **cfg):
    from rulztrainer import trainer as T

    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg = TrainConfig(dataset_dir="ds", output_dir=str(tmp_path / "out"), cache_dir=str(tmp_path / "cache"),
                         sample_negative="neg", model_arch="moss_image", probe_every=10,
                         sample_prompts=["a cat", "a dog"], sample_every=10, **cfg)
    tr.job_dir = tmp_path / "job"
    tr.job_dir.mkdir(exist_ok=True)
    tr.controller, tr.race_rows, tr.ft, tr.ema, tr.device = object(), None, None, None, "cpu"
    tr.arch = SimpleNamespace(name="moss_image", default_negative="neg", pad_embeds=lambda xs: (xs[0], None))
    tr.probe_worker, tr.race = None, None
    status = {}
    tr.status = SimpleNamespace(update=lambda **k: status.update(k), data={"samples": []})
    tr.injected = _lora(0)
    return T, tr, status


def _open(tr, tmp_path, clock, monkeypatch):
    monkeypatch.setattr(RR, "supported_host", lambda: True)
    cache = Path(tr.cfg.cache_dir) / "text"
    cache.mkdir(parents=True, exist_ok=True)
    from rulztrainer.dataset import text_cache_path

    for t in ("a cat", "a dog", "neg", "face prompt"):
        text_cache_path(tr.cfg, t).write_bytes(b"e:" + t.encode())
    tr.cfg.probe_face_prompt = "face prompt"
    tr._setup_remote_render()
    rr = tr.remote
    rr.clock, rr.sleep = clock, lambda s: setattr(clock, "t", clock.t + s)
    for k in ("probe", "sample"):
        rr.note_own(k, 4.0, 2)
    h = Helper(rr, clock)
    h.beat()
    h.ready()
    rr.poll()
    return rr, h


def test_run_start_publishes_the_prompts_and_does_nothing_where_it_cannot(tmp_path, monkeypatch):
    clock = Clock()
    T, tr, _ = _trainer(tmp_path)
    rr, h = _open(tr, tmp_path, clock, monkeypatch)
    prep = json.loads((rr.root / "prepare" / "prepare.json").read_text())
    assert set(prep["embeds"]) == {"a cat", "a dog", "neg", "face prompt"}
    assert prep["negatives"] == {"sample": "neg", "probe": "neg"} and prep["family"] == "moss_image"
    assert tr.remote_samples == [] and (rr.root / "run.json").exists()
    first = rr.run_token
    T2, tr2, _ = _trainer(tmp_path)  # a second start in the same job folder clears the first's
    _open(tr2, tmp_path, clock, monkeypatch)
    assert tr2.remote.run_token != first
    for change in ({"remote_render": "off"}, {"multi_gpu": "data_parallel"}):
        _, t3, _ = _trainer(tmp_path, **change)
        t3._setup_remote_render()
        assert t3.remote is None
    _, t4, _ = _trainer(tmp_path)
    t4.race_rows = [("a", None, {})]
    t4._setup_remote_render()
    assert t4.remote is None
    monkeypatch.setattr(RR, "supported_host", lambda: False)  # the user's own Windows PC: no instance, no helper
    _, t5, _ = _trainer(tmp_path)
    t5._setup_remote_render()
    assert t5.remote is None
    monkeypatch.setattr(RR, "supported_host", lambda: True)
    _, t6, _ = _trainer(tmp_path)
    t6.cfg.sample_prompts = ["not cached"]  # a prompt whose text embeds are missing: off, with a warning
    t6._setup_remote_render()
    assert t6.remote is None


def test_remote_render_validates():
    TrainConfig(dataset_dir="ds", remote_render="off").validate()
    assert TrainConfig(dataset_dir="ds").remote_render == "auto"
    with pytest.raises(ValueError, match="remote_render"):
        TrainConfig(dataset_dir="ds", remote_render="always").validate()


def _fake_rendering(T, tr, monkeypatch, seen):
    """The instance's own render, faked: it records the weights it rendered with."""
    monkeypatch.setattr(T, "load_text_embeds", lambda cfg, prompt: torch.zeros(1))
    tr._free_for_rendering = lambda: None
    tr._sampling_slots = lambda model, step: contextlib.nullcontext([])

    def render_round(slots, jobs, neg_e, neg_m, board, on_done=None):
        seen.append(_weights(tr.injected))
        return [str(j.out_path) for j in jobs]

    tr._render_round = render_round

    def boom(*a, **k):
        raise AssertionError("the weight average must not be applied over a mirrored state")

    tr._averaged = boom  # (a fallback render goes through neither run_probe nor sample_images)


def test_a_late_probe_renders_here_from_the_retained_weights_and_a_best_save_writes_them(tmp_path, monkeypatch):
    clock = Clock()
    T, tr, status = _trainer(tmp_path)
    rr, h = _open(tr, tmp_path, clock, monkeypatch)
    seen, finished = [], []
    _fake_rendering(T, tr, monkeypatch, seen)
    tr._score_probe = lambda jobs, images, step, check=True: (61.0, None, {"face": [61.0], "body": []})
    scored = _weights(tr.injected)

    def finish(step, face, body, images, seeds, took, opt, sched, injected, now=None):
        # what a best save writes: the weights that were scored, not training's newer ones
        finished.append((step, images, took, _same(injected, scored), tr.probe_rendered_on, now))
        return False

    tr._probe_finish = finish
    keel = []
    tr._keel_probe = lambda opt, ema=None, sync=True, scored=None: keel.append(scored)
    tr._probe_jobs = lambda step: _jobs(tr.job_dir, 2, step)
    tr._averaged = contextlib.nullcontext  # (submitting takes the average the way probes do)
    tr._remote_probe_submit(10, tr.injected, False)
    r = tr.remote_probe
    assert (rr.root / "rounds" / r.name).is_dir() and "rendering on pc RTX 4090" in status["stage"]
    for m in tr.injected.values():  # training goes on
        for p in m.adapter_params().values():
            p.data.add_(0.5)
    trained = _weights(tr.injected)
    _fake_rendering(T, tr, monkeypatch, seen)  # (puts the guard back)
    assert tr._remote_probe_take(None, 12, 12, None, None, wait=False) is False and finished == []  # still out
    clock.t += 10_000
    h.beat()
    assert tr._remote_probe_take(None, 14, 14, None, None, wait=False) is False
    assert len(seen) == 1 and all(torch.equal(p, scored[n][k]) for n, d in seen[0].items() for k, p in d.items())
    assert finished == [(10, [str(j.out_path) for j in r.jobs], finished[0][2], True, "instance", 14)]
    assert "after the helper's miss, in at step 14" in finished[0][2]
    assert _same(tr.injected, trained)  # training's weights are back
    assert tr.remote_probe is None and tr.probe_rendered_on == "instance"
    assert keel == [r.state]  # the Keel lookahead anchors the weights that scored


def test_a_probe_that_came_back_is_scored_and_labelled_with_the_helper(tmp_path, monkeypatch):
    clock = Clock()
    T, tr, status = _trainer(tmp_path)
    rr, h = _open(tr, tmp_path, clock, monkeypatch)
    finished = []
    tr._averaged = contextlib.nullcontext
    tr._probe_jobs = lambda step: _jobs(tr.job_dir, 2, step)
    tr._score_probe = lambda jobs, images, step, check=True: (61.0, None, {"face": [61.0], "body": []})
    tr._probe_finish = lambda step, face, body, images, seeds, took, o, s, inj, now=None: (
        finished.append((step, images, took, tr.probe_rendered_on, now)) or step >= 20)
    tr._keel_probe = lambda *a, **k: None
    tr._remote_probe_submit(10, tr.injected, False)
    r = tr.remote_probe
    h.claim(r)
    h.finish(r)
    assert tr._remote_probe_take(None, 13, 13, None, None, wait=False) is False
    assert finished[0][0] == 10 and finished[0][3] == "pc RTX 4090" and finished[0][4] == 13
    assert all(Path(p).exists() for p in finished[0][1]) and "on pc RTX 4090, rendered in 12s" in finished[0][2]
    # the next one: the controller's stop comes back through the take
    tr._remote_probe_submit(20, tr.injected, False)
    h.claim(tr.remote_probe)
    h.finish(tr.remote_probe)
    assert tr._remote_probe_take(None, 21, 21, None, None, wait=False) is True


def test_the_probe_record_says_where_it_rendered():
    from dataclasses import fields

    from rulztrainer.likeness import ProbeRecord

    assert [f.default for f in fields(ProbeRecord) if f.name == "rendered_on"] == [""]  # old records load without it


def test_a_probe_round_waits_for_the_one_still_out_and_renders_the_final_one_here(tmp_path, monkeypatch):
    clock = Clock()
    T, tr, status = _trainer(tmp_path)
    rr, h = _open(tr, tmp_path, clock, monkeypatch)
    order = []
    tr._averaged = contextlib.nullcontext
    tr._probe_jobs = lambda step: _jobs(tr.job_dir, 2, step)
    tr._score_probe = lambda jobs, images, step, check=True: (60.0, None, {"face": [60.0], "body": []})
    tr._probe_finish = lambda step, face, body, images, seeds, took, o, s, inj, now=None: (
        order.append(("acted", step)) or False)
    tr._keel_probe = lambda *a, **k: None
    tr.run_probe = lambda model, step, opt, sched, inj: order.append(("inline", step)) or False
    tr.last_probe = None
    safely = lambda job: tr._side_job("probe", job, False)  # noqa: E731
    rr.shadow_done = True
    assert tr._remote_probe_round(None, 10, None, None, tr.injected, False, False, safely) is False
    assert tr.remote_probe is not None and order == []  # delegated: nothing rendered here, training goes on
    first = tr.remote_probe
    h.claim(first)
    h.finish(first)
    # the next probe is due: the one out is acted on first; this one goes out too
    assert tr._remote_probe_round(None, 20, None, None, tr.injected, False, False, safely) is False
    assert order == [("acted", 10)] and tr.remote_probe.step == 20
    # the final probe: acted on in order, then rendered here
    second = tr.remote_probe
    h.claim(second)
    h.finish(second)
    tr._remote_probe_round(None, 30, None, None, tr.injected, True, False, safely)
    assert order == [("acted", 10), ("acted", 20), ("inline", 30)] and tr.remote_probe is None
    # a self-training round's probe (in_round) and a probe with no helper ready render here too
    rr.off_reason = "test"
    tr._remote_probe_round(None, 40, None, None, tr.injected, False, False, safely)
    assert order[-1] == ("inline", 40)


def test_a_round_that_cannot_be_handed_over_renders_here(tmp_path, monkeypatch):
    # (review: a failed submit, e.g. a full disk, must not lose the probe or the samples)
    clock = Clock()
    T, tr, status = _trainer(tmp_path)
    rr, h = _open(tr, tmp_path, clock, monkeypatch)
    order, inline = [], []
    tr._averaged = contextlib.nullcontext
    tr._probe_jobs = lambda step: _jobs(tr.job_dir, 2, step)
    tr._keel_probe = lambda *a, **k: None
    tr.run_probe = lambda model, step, opt, sched, inj: order.append(("inline", step)) or False
    tr.sample_images = lambda model, step: inline.append(step)
    tr.last_probe = None
    safely = lambda job: tr._side_job("probe", job, False)  # noqa: E731
    rr.shadow_done = True

    def full(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(rr, "submit", full)
    assert tr._remote_probe_round(None, 10, None, None, tr.injected, False, False, safely) is False
    assert tr.remote_probe is None and order == [("inline", 10)]
    tr._sample_round(None, 10)
    assert tr.remote_samples == [] and inline == [10]


def test_sample_rounds_are_bounded_at_two_and_finish_on_the_trainer_thread(tmp_path, monkeypatch):
    clock = Clock()
    T, tr, status = _trainer(tmp_path)
    rr, h = _open(tr, tmp_path, clock, monkeypatch)
    inline, finished = [], []
    tr._averaged = contextlib.nullcontext
    tr.sample_images = lambda model, step: inline.append(step)
    tr._sample_finish = lambda paths, step: finished.append((step, list(paths)))
    for step in (10, 20, 30):
        tr._sample_round(None, step)
    assert [r.step for r in tr.remote_samples] == [10, 20] and inline == [30]  # a third renders here as today
    jobs = tr.remote_samples[0].jobs
    assert [j.label for j in jobs] == ["sampling 1/2 at step 10", "sampling 2/2 at step 10"]
    h.claim(tr.remote_samples[0])
    h.finish(tr.remote_samples[0])
    tr._remote_sample_take(None, tr.remote_samples[0], False)  # the poll finds the first in
    assert [s for s, _ in finished] == [10] and len(finished[0][1]) == 2 and all(Path(p).exists() for p in finished[0][1])
    assert [r.step for r in tr.remote_samples] == [20]
    tr._sample_round(None, 40)  # a place is free again
    assert [r.step for r in tr.remote_samples] == [20, 40] and inline == [30]


def test_the_final_samples_wait_for_the_rounds_out_and_take_back_the_ones_the_helper_lost(tmp_path, monkeypatch):
    clock = Clock()
    T, tr, status = _trainer(tmp_path)
    rr, h = _open(tr, tmp_path, clock, monkeypatch)
    seen, finished = [], []
    tr._averaged = contextlib.nullcontext
    tr._sample_finish = lambda paths, step: finished.append((step, list(paths)))
    tr._sample_round(None, 10)
    tr._sample_round(None, 20)
    first, second = tr.remote_samples
    h.claim(first)
    h.finish(first)  # the first is in; the second is never claimed and the helper goes quiet
    _fake_rendering(T, tr, monkeypatch, seen)
    tr._remote_finish(None)
    assert tr.remote_samples == []
    assert [s for s, _ in finished] == [10, 20]
    assert len(seen) == 1 and not (rr.root / "rounds" / second.name).exists()  # rendered here from its own weights


def test_the_shadow_round_compares_the_helper_with_the_instance(tmp_path, monkeypatch):
    clock = Clock()
    T, tr, status = _trainer(tmp_path, likeness_noise_band=1.0)
    rr, h = _open(tr, tmp_path, clock, monkeypatch)
    tr._averaged = contextlib.nullcontext
    tr._probe_jobs = lambda step: _jobs(tr.job_dir, 2, step)
    tr._probe_finish = lambda *a, **k: False
    tr._keel_probe = lambda *a, **k: None
    ran = []

    def run_probe(model, step, opt, sched, inj):
        ran.append(step)
        tr.last_probe = (60.0, 50.0, ["x.png"], {})  # the instance's own scores
        return False

    tr.run_probe = run_probe
    tr.last_probe = None
    safely = lambda job: tr._side_job("probe", job, False)  # noqa: E731
    helper_scores = {"face": 60.4}
    tr._score_probe = lambda jobs, images, step, check=True: (helper_scores["face"], 50.2, {})
    assert tr._remote_probe_round(None, 10, None, None, tr.injected, False, False, safely) is False
    assert ran == [10] and tr.remote_probe is None and tr.remote_shadow["set"]  # rendered on both: nothing handed on
    shadow = tr.remote_shadow["round"]
    assert shadow.shadow and all("remote_render/shadow/" in Path(j.out_path).as_posix() for j in shadow.jobs)
    h.claim(shadow)
    h.finish(shadow)
    tr._remote_shadow_settle(11)
    assert tr.remote_shadow is None and rr.shadow_done and not rr.probes_off  # 0.4 points: inside the band
    assert rr.can_delegate("probe")
    tr._remote_probe_round(None, 20, None, None, tr.injected, False, False, safely)
    assert ran == [10] and tr.remote_probe is not None  # from the second probe on the helper renders alone
    # a helper that disagrees ends remote probes, samples go on
    rr.shadow_done, tr.remote_probe = False, None
    helper_scores["face"] = 63.0
    tr._remote_probe_round(None, 30, None, None, tr.injected, False, False, safely)
    shadow = tr.remote_shadow["round"]
    h.claim(shadow)
    h.finish(shadow)
    tr._remote_shadow_settle(31)
    assert rr.probes_off and not rr.can_delegate("probe") and rr.can_delegate("sample")


def test_a_shadow_round_that_never_comes_in_is_dropped_and_tried_again(tmp_path, monkeypatch):
    clock = Clock()
    T, tr, status = _trainer(tmp_path)
    rr, h = _open(tr, tmp_path, clock, monkeypatch)
    tr._averaged = contextlib.nullcontext
    tr._probe_jobs = lambda step: _jobs(tr.job_dir, 2, step)
    tr._probe_finish = lambda *a, **k: False
    tr._keel_probe = lambda *a, **k: None
    tr.run_probe = lambda model, step, opt, sched, inj: setattr(tr, "last_probe", (60.0, None, [], {})) or False
    tr.last_probe = None
    safely = lambda job: tr._side_job("probe", job, False)  # noqa: E731
    tr._remote_probe_round(None, 10, None, None, tr.injected, False, False, safely)
    assert tr.remote_shadow is not None
    tr._remote_shadow_settle(11)  # still out: nothing yet
    assert tr.remote_shadow is not None
    tr._remote_probe_round(None, 20, None, None, tr.injected, False, False, safely)  # the next probe is due
    assert rr.shadow_done is False  # (this one is a shadow round again)
    assert tr.remote_shadow is not None and tr.remote_shadow["round"].step == 20


def test_a_round_queued_behind_another_is_not_a_slow_helper(tmp_path):
    rr, h, clock = _renderer(tmp_path)
    rr.first_round = False
    h.beat()
    h.ready()
    rr.poll()
    sample = rr.submit("sample", 10, _jobs(rr.job_dir, kind="sample"), _state(), "sample")
    probe = rr.submit("probe", 10, _jobs(rr.job_dir), _state(), "probe")
    assert h.claim(sample)  # one worker: the sample renders, the probe queues
    assert rr.check(probe) == "open" and probe.behind
    clock.t += 100  # past the probe's own 90 s, inside 2 x
    h.beat()
    assert rr.check(probe) == "open"
    clock.t += 100  # past the cap with the probe never claimed
    h.beat()
    assert rr.check(probe) == "late"
    got, why = rr.collect(probe)
    assert got is None and rr.misses == 0 and not rr.off_reason and rr.can_delegate("probe")
    # claimed late behind another round: its time starts at the claim
    h.finish(sample)
    rr.collect(sample)
    a = rr.submit("sample", 20, _jobs(rr.job_dir, step=20, kind="sample"), _state(), "sample")
    b = rr.submit("probe", 20, _jobs(rr.job_dir, step=20), _state(), "probe")
    h.claim(a)
    rr.check(b)
    clock.t += 80
    h.beat()
    h.finish(a)
    rr.collect(a)
    assert h.claim(b) and rr.check(b) == "open" and b.deadline > b.submitted + b.span - 1
    clock.t += 85
    h.beat()
    assert rr.check(b) == "open"  # 85 s after its claim, under its 90 s
    clock.t += 10
    h.beat()
    assert rr.check(b) == "late"
    assert rr.collect(b)[0] is None and rr.misses == 1  # claimed and then too slow: a miss
    # unclaimed while the helper is idle: a miss
    c = rr.submit("probe", 30, _jobs(rr.job_dir, step=30), _state(), "probe")
    clock.t += 200
    h.beat()
    assert rr.collect(c)[0] is None and rr.misses == 2 and rr.off_reason


def test_modes_that_skip_remote_render_clear_an_old_runs_folder(tmp_path, monkeypatch):
    clock = Clock()
    T, tr, _ = _trainer(tmp_path)
    _open(tr, tmp_path, clock, monkeypatch)
    folder = tr.job_dir / "remote_render"
    assert (folder / "run.json").exists()
    for change in ({"remote_render": "off"}, {"multi_gpu": "data_parallel"}):
        T2, t2, _ = _trainer(tmp_path, **change)
        t2.job_dir = tr.job_dir
        (folder / "run.json").write_text("{}")
        t2._setup_remote_render()
        assert t2.remote is None and not folder.exists()
        _open(tr, tmp_path, clock, monkeypatch)
    T3, t3, _ = _trainer(tmp_path)
    t3.job_dir, t3.race_rows = tr.job_dir, [("a", None, {})]
    t3._setup_remote_render()
    assert not folder.exists()
    _open(tr, tmp_path, clock, monkeypatch)
    monkeypatch.setattr(RR, "supported_host", lambda: False)  # Windows
    T4, t4, _ = _trainer(tmp_path)
    t4.job_dir = tr.job_dir
    t4._setup_remote_render()
    assert t4.remote is None and not folder.exists()
