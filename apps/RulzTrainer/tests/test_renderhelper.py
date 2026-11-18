"""The PC helper (rulztrainer/renderhelper.py): its state machine against the real bridge script (a local subprocess over
pipes, in a tmp install folder), with a fake Renderer in place of the GPU model and a fake rulztrainer.remoterender
(the trainer-side module is another piece: only PROTOCOL, render_hash and job_from_dict are used, and the fakes have
the signatures the contract gives). No GPU, no network, no ssh."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rulztrainer import renderhelper as rh
from rulztrainer import renderhelper_ctl as ctl

APP = Path(__file__).resolve().parents[1]
TOKEN = "tok123"
EMBEDS = {"a portrait": "embeds/aaa.safetensors", "bad": "embeds/bbb.safetensors"}


class FakeRenderer:
    def __init__(self):
        self.calls = []
        self.fail_next = None
        self.during = None

    def render(self, jobs, payload, negative_kind="probe"):
        self.calls.append((jobs, payload, negative_kind))
        if self.during:
            self.during()
        if self.fail_next:
            e, self.fail_next = self.fail_next, None
            raise e
        for j in jobs:  # like Image.save: no folders made, so a missing one raises
            j.out_path.write_bytes(j.label.encode())
            if j.frames > 1:
                j.out_path.with_suffix(".webp").write_bytes(b"clip " + j.label.encode())
        return [str(j.out_path) for j in jobs], 1.5


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A tmp install folder with a running remote-render job, the helper's patched edges and a way to build helpers."""
    folder = tmp_path / "install"
    job = folder / "output" / "job"
    base = job / "remote_render"
    (base / "prepare" / "embeds").mkdir(parents=True)
    (folder / ".webui_job.json").write_text(json.dumps({"pid": os.getpid(), "job_dir": str(job)}))
    (base / "run.json").write_text(json.dumps({"protocol": 1, "run_token": TOKEN, "job": "job", "family": "zimage"}))
    w = SimpleNamespace(folder=folder, job=job, base=base, tmp=tmp_path, renderer=FakeRenderer(), loads=[], freed=[],
                        downloads=[], conns=[], embed_root=tmp_path / "helper_cache", cached={})
    w.prep = {"protocol": 1, "run_token": TOKEN, "recipe_hash": "rec", "render_hash": "h-zimage", "family": "zimage",
              "cfg": {"models_dir": "models", "transformer_file": "/workspace/x/model.safetensors", "device": "cuda:0",
                      "sample_model_path": ""},
              "files": {"transformer_file": {"path": "/workspace/x/model.safetensors", "name": "model.safetensors",
                                             "size": 10, "repo": None, "file": None}},
              "negatives": {"probe": "bad"}, "embeds": EMBEDS}
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.safetensors").write_bytes(b"0123456789")
    w.models = models
    w.write_prepare = lambda: (base / "prepare" / "prepare.json").write_text(json.dumps(w.prep))
    for rel in EMBEDS.values():
        (base / "prepare" / rel).write_bytes(b"emb " + rel.encode())

    monkeypatch.setattr(rh, "remote", lambda: SimpleNamespace(
        PROTOCOL=1, render_hash=lambda family: "h-" + family,
        job_from_dict=lambda d, b: SimpleNamespace(label=d["label"], out_path=Path(b) / d["out_path"],
                                                   frames=d.get("frames", 1))))
    monkeypatch.setattr(rh, "local_embed_path", lambda cfg, prompt: w.embed_root / Path(cfg["cache_dir"]).name /
                        (prompt.replace(" ", "_") + ".safetensors"))

    def load_renderer(cfg, device):
        w.loads.append((cfg, device))
        if getattr(w, "load_error", None):
            raise w.load_error
        return w.renderer, {"gpu": "Fake 4090", "vram_gb": 24.0}

    monkeypatch.setattr(rh, "load_renderer", load_renderer)
    monkeypatch.setattr(rh, "release_gpu", lambda: w.freed.append(1))
    monkeypatch.setattr(rh, "cached_hf_file", lambda repo, file: w.cached.get((repo, file)))

    def ensure(repo, file):
        w.downloads.append((repo, file))
        return w.cached[(repo, file)]

    monkeypatch.setattr(rh, "ensure_hf_file", ensure)
    monkeypatch.setattr(rh, "HEARTBEAT_EVERY", 0.0)

    def connect():
        proc = subprocess.Popen([sys.executable, "-c", rh.BOOTSTRAP], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                cwd=folder)
        b = rh.Bridge(proc)
        script = rh.BRIDGE_FILE.read_bytes()
        import struct

        proc.stdin.write(struct.pack(">I", len(script)) + script)
        proc.stdin.flush()
        b.call("hello", protocol=1)
        w.conns.append(b)
        return b

    w.connect = connect

    def make(**kw):
        opts = rh.Options(instance="42", device="cuda:1", models_dir=str(models), status_file=tmp_path / "status.json",
                          work_dir=tmp_path / "work", **kw)
        h = rh.Helper(opts, connect)
        w.helper = h
        return h

    w.make = make

    def add_round(name, kind, step, jobs=None, negative_kind="probe", token=TOKEN):
        d = base / "rounds" / name
        d.mkdir(parents=True)
        jobs = jobs if jobs is not None else [{"label": f"{kind}{step}", "out_path": f"{kind}s/{step}_0.png"}]
        (d / "round.json").write_text(json.dumps({"protocol": 1, "run_token": token, "seq": 1, "kind": kind, "step": step,
                                                  "negative_kind": negative_kind, "jobs": jobs, "deadline_s": 90}))
        (d / "lora.safetensors").write_bytes(b"lora " + name.encode())

    w.add_round = add_round
    yield w
    h = getattr(w, "helper", None)
    if h:
        h.stop.set()
        h._drop_hb()
        h._drop_wb()
    for b in w.conns:
        b.close()


def drive(h, until, timeout=20.0):
    """Tick (and let the worker finish) until `until()`."""
    end = time.time() + timeout
    while time.time() < end:
        h.tick()
        if h.worker is not None:
            h.worker.join(10)
        if until():
            return
        time.sleep(0.05)
    raise AssertionError(f"not reached; status {h.status.data}")


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def ready_state(w, **kw):
    w.write_prepare()
    h = w.make(**kw)
    drive(h, lambda: (w.base / "prepare/ready.json").exists())
    return h


# ---------------------------------------------------------------- preparing


def test_prepare_writes_ready_json_and_loads_with_local_paths(world):
    w = world
    h = ready_state(w)
    ready = read_json(w.base / "prepare/ready.json")
    assert ready["run_token"] == TOKEN and ready["recipe_hash"] == "rec" and ready["render_hash"] == "h-zimage"
    assert ready["kinds"] == ["sample", "probe"] and ready["gpu"] == "Fake 4090" and ready["vram_gb"] == 24.0
    assert ready["host"] == h.host and ready["load_seconds"] >= 0 and ready["time"] > 0
    cfg, device = w.loads[0]
    assert device == "cuda:1" and cfg["device"] == "cuda:1"
    assert cfg["transformer_file"] == str(w.models / "model.safetensors")  # the PC's file, not the instance's path
    cache = Path(cfg["cache_dir"])
    assert w.tmp / "work" in cache.parents  # the helper's own folder, never the real text cache
    embeds = sorted(p.name for p in (w.embed_root / cache.name).iterdir())
    assert embeds == ["a_portrait.safetensors", "bad.safetensors"]
    assert (w.embed_root / cache.name / "a_portrait.safetensors").read_bytes() == b"emb embeds/aaa.safetensors"
    assert h.status.data["state"] == "ready"


def test_versions_differ_is_not_ready_and_not_retried(world):
    w = world
    w.prep["render_hash"] = "other"
    w.write_prepare()
    h = w.make()
    drive(h, lambda: (w.base / "prepare/not_ready.json").exists())
    nr = read_json(w.base / "prepare/not_ready.json")
    assert nr["reason"] == "versions differ: update RulzTrainer on the instance or this PC" and nr["run_token"] == TOKEN
    assert not w.loads and not (w.base / "prepare/ready.json").exists()
    assert h.status.data["state"] == "not_ready" and "versions differ" in h.status.data["message"]
    assert h.next_prepare == float("inf")
    for _ in range(3):
        h.tick()
    assert not w.loads and h.worker is None or not h.worker.is_alive()


def test_protocol_mismatch_is_not_ready(world):
    w = world
    w.prep["protocol"] = 2
    w.write_prepare()
    h = w.make()
    drive(h, lambda: (w.base / "prepare/not_ready.json").exists())
    assert "versions differ" in read_json(w.base / "prepare/not_ready.json")["reason"]


def test_waits_for_prepare_json(world):
    w = world
    h = w.make()
    for _ in range(3):
        h.tick()
    assert h.status.data["state"] == "preparing" and h.worker is None
    assert (w.base / "helper.json").exists()  # but the heartbeat goes already
    assert read_json(w.base / "helper.json")["run_token"] == TOKEN


def test_missing_files_are_listed(world):
    w = world
    (w.models / "model.safetensors").write_bytes(b"short")  # same name, other size: another model
    w.prep["files"]["vae_file"] = {"path": "/w/ae.safetensors", "name": "ae.safetensors", "size": 5, "repo": None, "file": None}
    w.write_prepare()
    h = w.make()
    drive(h, lambda: (w.base / "prepare/not_ready.json").exists())
    reason = read_json(w.base / "prepare/not_ready.json")["reason"]
    assert "model.safetensors" in reason or "ae.safetensors" in reason
    assert not w.loads


def test_file_found_in_a_subfolder_by_name_and_size(world):
    w = world
    (w.models / "model.safetensors").unlink()
    (w.models / "sub" / "deeper").mkdir(parents=True)
    (w.models / "sub/deeper/model.safetensors").write_bytes(b"0123456789")
    h = ready_state(w)
    assert w.loads[0][0]["transformer_file"] == str(w.models / "sub/deeper/model.safetensors")
    assert h.prepared


def test_provenance_downloads_but_never_the_text_encoder(world):
    w = world
    (w.models / "model.safetensors").unlink()
    got = w.tmp / "hf_model.safetensors"
    got.write_bytes(b"0123456789")
    w.cached[("Comfy-Org/x", "split/model.safetensors")] = got
    w.prep["files"]["transformer_file"].update(repo="Comfy-Org/x", file="split/model.safetensors")
    w.prep["files"]["text_encoder_file"] = {"path": "/w/te.safetensors", "name": "te.safetensors", "size": 8_000_000_000,
                                           "repo": "Comfy-Org/x", "file": "te/te.safetensors"}
    ready_state(w)
    assert w.downloads == [("Comfy-Org/x", "split/model.safetensors")]
    cfg = w.loads[0][0]
    assert cfg["transformer_file"] == str(got)
    assert cfg["text_encoder_file"] == "/w/te.safetensors"  # not read here: the embeds replace it


def test_download_failure_is_not_ready(world):
    w = world
    (w.models / "model.safetensors").unlink()
    w.prep["files"]["transformer_file"].update(repo="Comfy-Org/x", file="m.safetensors")

    def boom(repo, file):
        raise OSError("offline")

    rh.ensure_hf_file = boom  # restored by monkeypatch's own undo of the fixture's patch
    w.write_prepare()
    h = w.make()
    drive(h, lambda: (w.base / "prepare/not_ready.json").exists())
    assert "offline" in read_json(w.base / "prepare/not_ready.json")["reason"]


def test_load_failure_and_oom_are_not_ready(world):
    w = world
    w.load_error = RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    w.write_prepare()
    h = w.make()
    drive(h, lambda: (w.base / "prepare/not_ready.json").exists())
    reason = read_json(w.base / "prepare/not_ready.json")["reason"]
    assert "Out of GPU memory" in reason and "cuda:1" in reason
    assert not (w.base / "prepare/ready.json").exists() and h.status.data["state"] == "not_ready"


def test_sample_override_renders_samples_only(world):
    w = world
    (w.models / "model.safetensors").unlink()  # the override replaces the transformer: no need for it
    w.write_prepare()
    h = w.make(sample_model="C:/m/turbo.safetensors", sample_quant="int8")
    drive(h, lambda: (w.base / "prepare/ready.json").exists())
    assert read_json(w.base / "prepare/ready.json")["kinds"] == ["sample"]
    cfg = w.loads[0][0]
    assert cfg["sample_model_path"] == "C:/m/turbo.safetensors" and cfg["sample_model_quant"] == "int8"
    w.add_round("000001_probe_5", "probe", 5)
    w.add_round("000002_sample_5", "sample", 5, negative_kind="sample")
    drive(h, lambda: (w.base / "done/000002_sample_5").exists())
    assert (w.base / "rounds/000001_probe_5").exists()  # a probe is left for the instance
    assert [c[2] for c in w.renderer.calls] == ["sample"]


# ---------------------------------------------------------------- rounds


def test_rounds_probe_first_with_negative_kind_and_upload(world):
    w = world
    h = ready_state(w)
    w.add_round("000001_sample_5", "sample", 5, negative_kind="sample",
                jobs=[{"label": "s1", "out_path": "samples/5_0.png"},
                      {"label": "s2", "out_path": "samples/5_1.png", "frames": 5}])
    w.add_round("000002_probe_5", "probe", 5, negative_kind="probe")
    drive(h, lambda: (w.base / "done/000001_sample_5").exists())
    assert [c[2] for c in w.renderer.calls] == ["probe", "sample"]  # probes first, though the sample is older
    assert w.renderer.calls[0][1] == b"lora 000002_probe_5"
    done = w.base / "done/000002_probe_5"
    assert (done / "out/probes/5_0.png").read_bytes() == b"probe5"
    d = read_json(done / "done.json")
    assert d["files"] == ["probes/5_0.png"] and d["seconds"] == 1.5 and d["host"] == h.host and d["gpu"] == "Fake 4090"
    d2 = read_json(w.base / "done/000001_sample_5/done.json")
    assert d2["files"] == ["samples/5_0.png", "samples/5_1.webp", "samples/5_1.png"]  # a clip's file before its still
    assert (w.base / "done/000001_sample_5/out/samples/5_1.webp").read_bytes() == b"clip s2"
    assert not list((w.base / "claimed").iterdir()) and not list((w.base / "rounds").iterdir())
    s = read_json(w.tmp / "status.json")
    assert s["rounds_done"] == 2 and s["last_round"] == "000001_sample_5" and s["last_seconds"] == 1.5
    assert not (w.tmp / "work" / TOKEN / "rounds" / "000001_sample_5").exists()  # cleaned up


def test_render_exception_goes_to_failed_with_error_json_and_the_helper_goes_on(world):
    w = world
    h = ready_state(w)
    w.renderer.fail_next = RuntimeError("boom")
    w.add_round("000001_probe_5", "probe", 5)
    drive(h, lambda: (w.base / "failed/000001_probe_5").exists())
    err = read_json(w.base / "failed/000001_probe_5/error.json")
    assert "boom" in err["error"] and "Traceback" in err["traceback"]
    assert h.status.data["rounds_done"] == 0 and h.status.data["state"] == "ready"
    w.add_round("000002_probe_10", "probe", 10)
    drive(h, lambda: (w.base / "done/000002_probe_10").exists())


def test_round_of_another_run_fails(world):
    w = world
    h = ready_state(w)
    w.add_round("000001_probe_5", "probe", 5, token="someone else")
    drive(h, lambda: (w.base / "failed/000001_probe_5").exists())
    assert not w.renderer.calls


def test_picture_path_leaving_the_work_folder_fails_the_round(world):
    w = world
    h = ready_state(w)
    w.add_round("000001_probe_5", "probe", 5, jobs=[{"label": "x", "out_path": "../../escape.png"}])
    drive(h, lambda: (w.base / "failed/000001_probe_5").exists())
    assert "work folder" in read_json(w.base / "failed/000001_probe_5/error.json")["error"]
    assert not w.renderer.calls and not (w.tmp / "work" / "escape.png").exists()


def test_missing_picture_fails_the_round(world):
    w = world
    h = ready_state(w)
    w.renderer.render = lambda jobs, payload, negative_kind="probe": ([], 1.0)  # renders nothing
    w.add_round("000001_probe_5", "probe", 5)
    drive(h, lambda: (w.base / "failed/000001_probe_5").exists())


def test_round_taken_back_before_the_claim_is_skipped(world):
    w = world
    h = ready_state(w)
    entry = {"job": "job", "run": {"run_token": TOKEN}}
    h.wb = w.connect()
    h._render(entry, "000009_probe_1")  # not there: the trainer took it
    assert not w.renderer.calls and not (w.base / "done").exists()


def test_late_done_after_abandon_is_dropped(world):
    w = world
    h = ready_state(w)
    w.add_round("000001_probe_5", "probe", 5)

    def abandon():  # the instance's deadline passes while the PC renders
        (w.base / "abandoned").mkdir(exist_ok=True)
        os.rename(w.base / "claimed/000001_probe_5", w.base / "abandoned/000001_probe_5")

    w.renderer.during = abandon
    drive(h, lambda: h.status.data["state"] == "ready" and bool(w.renderer.calls) and h.worker and not h.worker.is_alive())
    assert (w.base / "abandoned/000001_probe_5").exists() and not (w.base / "done/000001_probe_5").exists()
    assert h.status.data["rounds_done"] == 0 and "taken back" in h.status.data["message"]


def test_heartbeat_keeps_going_while_a_round_renders(world):
    w = world
    h = ready_state(w)
    gate = threading.Event()
    w.renderer.during = lambda: gate.wait(20)
    w.add_round("000001_probe_5", "probe", 5)
    h.tick()
    deadline = time.time() + 10
    while not w.renderer.calls and time.time() < deadline:
        time.sleep(0.05)
        h.tick()
    assert h.worker.is_alive() and h.status.data["state"] == "rendering"
    first = read_json(w.base / "helper.json")["time"]
    time.sleep(0.05)
    h.tick()  # the main thread beats while the worker is inside the render
    assert read_json(w.base / "helper.json")["time"] > first
    assert h.status.data["state"] == "rendering"
    gate.set()
    h.worker.join(10)
    assert (w.base / "done/000001_probe_5").exists()


def test_behind_counts_open_rounds(world):
    w = world
    h = ready_state(w)
    gate = threading.Event()
    w.renderer.during = lambda: gate.wait(20)
    w.add_round("000001_probe_5", "probe", 5)
    w.add_round("000002_probe_10", "probe", 10)
    w.add_round("000003_sample_10", "sample", 10)
    deadline = time.time() + 10
    while not w.renderer.calls and time.time() < deadline:
        h.tick()
        time.sleep(0.05)
    h.tick()
    assert h.status.data["behind"] == 2
    gate.set()
    h.worker.join(10)


# ---------------------------------------------------------------- connection, runs, stop, status


def test_connect_failures_back_off_then_recover(world):
    w = world
    attempts = []

    def flaky():
        attempts.append(time.time())
        if len(attempts) < 3:
            raise rh.Disconnected("ssh: no route")
        return w.connect()

    h = rh.Helper(rh.Options(instance="42", status_file=w.tmp / "status.json", work_dir=w.tmp / "work"), flaky)
    w.helper = h
    h.tick()
    assert h.status.data["state"] == "connecting" and "no route" in h.status.data["message"] and h.hb is None
    h.tick()  # too early: no second try yet
    assert len(attempts) == 1
    h.next_connect = 0
    h.tick()
    assert len(attempts) == 2 and h.tries == 2
    h.next_connect = 0
    h.tick()
    assert h.hb is not None and h.tries == 0


def test_dropped_connection_reconnects(world):
    w = world
    w.write_prepare()
    h = w.make()
    h.tick()
    assert h.hb is not None
    h.hb.proc.kill()
    h.hb.proc.wait()
    h.tick()
    assert h.hb is None and h.status.data["state"] == "connecting"
    h.tick()
    assert h.hb is not None and h.hb is not w.conns[0]


def test_no_running_job_is_idle_and_a_new_run_is_prepared_again(world):
    w = world
    h = ready_state(w)
    assert w.loads and not w.freed
    # the run ends: its process is gone
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (w.folder / ".webui_job.json").write_text(json.dumps({"pid": dead.pid, "job_dir": str(w.job)}))
    h.tick()
    assert h.status.data["state"] == "idle" and w.freed == [1] and h.token is None and h.renderer is None
    # a new run in the same folder
    (w.folder / ".webui_job.json").write_text(json.dumps({"pid": os.getpid(), "job_dir": str(w.job)}))
    (w.base / "run.json").write_text(json.dumps({"run_token": "tok2", "job": "job"}))
    w.prep["run_token"] = "tok2"
    w.write_prepare()
    (w.base / "prepare/ready.json").unlink()
    drive(h, lambda: (w.base / "prepare/ready.json").exists())
    assert read_json(w.base / "prepare/ready.json")["run_token"] == "tok2" and len(w.loads) == 2


def test_status_file_format(world):
    w = world
    ready_state(w)
    s = read_json(w.tmp / "status.json")
    assert set(s) == {"state", "instance", "job", "gpu", "message", "rounds_done", "last_round", "last_seconds", "behind",
                      "time", "pid"}
    assert s["state"] == "ready" and s["instance"] == "42" and s["job"] == "job" and s["pid"] == os.getpid()
    assert abs(s["time"] - time.time()) < 60


def test_shutdown_writes_stopped_and_closes_connections(world):
    w = world
    h = ready_state(w)
    h.shutdown()
    s = read_json(w.tmp / "status.json")
    assert s["state"] == "stopped" and h.hb is None and h.wb is None
    h.status.update(state="ready")  # nothing writes after the stop
    assert read_json(w.tmp / "status.json")["state"] == "stopped"


def test_run_stops_when_stop_is_set(world):
    w = world
    w.write_prepare()
    h = w.make()
    t = threading.Thread(target=h.run)
    t.start()
    time.sleep(1.5)
    h.stop.set()
    t.join(30)
    assert not t.is_alive() and read_json(w.tmp / "status.json")["state"] == "stopped"


# ---------------------------------------------------------------- command line, and the Fleet tab's side


def test_list_gpus_prints_json_and_loads_no_model(monkeypatch, capsys):
    monkeypatch.setattr(rh, "list_gpus", lambda: [{"index": 0, "name": "X", "total_gb": 24.0, "free_gb": 20.0}])
    monkeypatch.setattr(rh, "load_renderer", lambda *a: pytest.fail("loaded a model"))
    assert rh.main(["--list-gpus"]) == 0
    assert ctl.parse_gpus(capsys.readouterr().out) == [{"index": 0, "name": "X", "total_gb": 24.0, "free_gb": 20.0}]


def test_the_controllers_command_line_parses(tmp_path):
    cmd = ctl.helper_command(123, 1, tmp_path / "123.json", "C:/m/t.safetensors", "int8", "D:/models")
    assert cmd[1:3] == ["-m", "rulztrainer.renderhelper"]
    a = rh.parse_args(cmd[3:])
    assert (a.instance, a.device, a.sample_model, a.sample_model_quant, a.models_dir, a.status_file) == (
        "123", "cuda:1", "C:/m/t.safetensors", "int8", "D:/models", str(tmp_path / "123.json"))
    a = rh.parse_args(ctl.helper_command(5, 0, tmp_path / "5.json")[3:])
    assert a.sample_model == "" and a.sample_model_quant == "" and a.models_dir == ""


def test_every_state_the_helper_writes_has_a_word_in_the_fleet_tab():
    assert set(rh.STATES) <= set(ctl.STATES)


def test_bootstrap_has_no_single_quote():
    # it is one shell word after shlex.quote either way, but a plain one survives Windows ssh argument rules best
    assert "'" not in rh.BOOTSTRAP


def test_the_real_remoterender_names_fit(tmp_path):
    """The names the helper imports from the trainer's shared module, with the signatures it assumes."""
    shared = pytest.importorskip("rulztrainer.remoterender")
    assert shared.PROTOCOL == 1
    assert isinstance(shared.render_hash("zimage"), str)
    job = shared.job_from_dict({"prompt": "p", "width": 64, "height": 64, "steps": 1, "cfg_scale": 1.0, "seed": 1,
                                "out_path": "probes/5_0.png", "label": "probe", "frames": 5}, tmp_path)
    assert job.out_path == tmp_path / "probes" / "5_0.png" and job.frames == 5 and job.label == "probe"
    with pytest.raises(Exception):  # a path that leaves the work folder never gets as far as a render
        shared.job_from_dict({"prompt": "p", "width": 64, "height": 64, "steps": 1, "cfg_scale": 1.0, "seed": 1,
                              "out_path": "../escape.png", "label": "x"}, tmp_path)


def test_the_picture_folders_are_made_before_the_render(world):
    w = world
    h = ready_state(w)
    w.add_round("000001_probe_5", "probe", 5, jobs=[{"label": "a", "out_path": "probes/deep/x/5_0.png"}])
    drive(h, lambda: (w.base / "done/000001_probe_5").exists())
    assert (w.base / "done/000001_probe_5/out/probes/deep/x/5_0.png").read_bytes() == b"a"


def test_models_dir_is_this_pcs_never_the_instances(world, monkeypatch):
    w = world
    app = w.tmp / "app"
    (app / "models").mkdir(parents=True)
    (app / "models" / "model.safetensors").write_bytes(b"0123456789")
    monkeypatch.setattr(rh, "APP", app)
    w.prep["cfg"]["models_dir"] = "/workspace/RulzTrainer/models"
    w.write_prepare()
    h = rh.Helper(rh.Options(instance="42", device="cuda:1", status_file=w.tmp / "status.json", work_dir=w.tmp / "work"),
                  w.connect)
    w.helper = h
    drive(h, lambda: (w.base / "prepare/ready.json").exists())
    cfg = w.loads[0][0]
    assert cfg["models_dir"] == str(app / "models")
    assert cfg["transformer_file"] == str(app / "models" / "model.safetensors")
    assert "workspace" not in cfg["cache_dir"]


def test_a_second_model_never_loads_on_top_of_the_first(world):
    w = world
    h = ready_state(w)
    first = h.renderer
    assert first is not None
    h.prepared = False  # e.g. ready.json could not be written: the next try loads again
    h.next_prepare = 0.0
    (w.base / "prepare/ready.json").unlink()
    drive(h, lambda: (w.base / "prepare/ready.json").exists())
    assert len(w.loads) == 2 and w.freed  # the first was dropped (and the GPU released) before the second loaded


def test_an_unexpected_prepare_error_is_reported_and_not_retried_every_second(world, monkeypatch):
    w = world
    w.write_prepare()

    def boom(cfg, prompt):
        raise ZeroDivisionError("oops")

    monkeypatch.setattr(rh, "local_embed_path", boom)
    h = w.make()
    drive(h, lambda: (w.base / "prepare/not_ready.json").exists())
    assert "ZeroDivisionError" in read_json(w.base / "prepare/not_ready.json")["reason"]
    assert h.next_prepare > time.time() + 30 and not w.loads


def test_a_failed_render_releases_the_gpu(world):
    w = world
    h = ready_state(w)
    before = len(w.freed)
    w.renderer.fail_next = RuntimeError("CUDA out of memory")
    w.add_round("000001_probe_5", "probe", 5)
    drive(h, lambda: (w.base / "failed/000001_probe_5").exists())
    assert len(w.freed) > before


def test_connect_surprises_are_a_failed_connect(world, monkeypatch):
    w = world
    conn = rh.SshConnector.__new__(rh.SshConnector)
    conn.lock, conn.conn, conn.instance, conn.key = threading.Lock(), None, 1, None

    def garbled(*a, **k):
        raise ValueError("Expecting value: line 1 column 1")

    conn.remote = SimpleNamespace(saved_connection=lambda i: None, find_connection=garbled)
    with pytest.raises(rh.Disconnected, match="ValueError"):
        conn()


def test_the_workers_connection_backs_off(world):
    w = world
    calls = []

    def bad():
        calls.append(1)
        raise rh.Disconnected("no route")

    h = rh.Helper(rh.Options(instance="42", status_file=w.tmp / "status.json", work_dir=w.tmp / "work"), bad)
    w.helper = h
    for _ in range(3):
        with pytest.raises(rh.Disconnected):
            h._bridge()
    assert len(calls) == 1 and h.wtries == 1


def test_find_local_file_does_not_glob(tmp_path):
    (tmp_path / "a[1].safetensors").write_bytes(b"12345")
    (tmp_path / "a1.safetensors").write_bytes(b"123")
    assert rh.find_local_file("a[1].safetensors", 5, [tmp_path]) == tmp_path / "a[1].safetensors"
    assert rh.find_local_file("a[1].safetensors", 3, [tmp_path]) is None


_WATCH_CHILD = r'''
import sys, threading, time
from rulztrainer.renderhelper import watch_stdin
stop = threading.Event()
threading.Thread(target=watch_stdin, args=(stop, 0.05), daemon=True).start()
time.sleep(0.5)
done = threading.Event()
def load():
    import torch  # noqa: F401 - loads many DLLs: the step that hung behind a blocking stdin read on Windows
    done.set()
threading.Thread(target=load, daemon=True).start()
print("loaded" if done.wait(90) else "hung", flush=True)
print("stopped" if stop.wait(30) else "running", flush=True)
'''


def test_watch_stdin_lets_dlls_load_and_stops_on_close():
    """The WebUI starts the helper with stdin a pipe it keeps open. A blocking read of that pipe on Windows made the
    worker's torch import wait until the pipe closed (the helper sat at Preparing). The watch must let DLLs load, and
    still stop the helper when the pipe closes."""
    pytest.importorskip("torch")
    proc = subprocess.Popen([sys.executable, "-c", _WATCH_CHILD], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, cwd=str(Path(rh.__file__).resolve().parents[1]))
    try:
        assert proc.stdout.readline().strip() == "loaded"
        proc.stdin.close()  # the WebUI stopping the helper
        assert proc.stdout.readline().strip() == "stopped"
    finally:
        proc.kill()
        proc.wait(10)
