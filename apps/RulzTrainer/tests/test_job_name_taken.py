"""One name, one run (rulztrainer.runplan, webui.queue_run / _start_training): a new run is refused a job folder that
holds an earlier run's output and told the next free name; carrying that run on is not a new run. tmp_path only: no
run is started (the job's process is faked)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import runplan  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


def leftovers(job: Path) -> Path:
    """What a run that never saved anything leaves: its config and logs."""
    job.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "train.log", "console.log", "status.json", "per_image_loss.jsonl"):
        (job / name).write_text("x", encoding="utf-8")
    return job


def saved_lora(job: Path) -> Path:
    leftovers(job)
    (job / f"{job.name}_000250.safetensors").write_bytes(b"x")
    return job


def a_state(job: Path, step: int = 250) -> Path:
    leftovers(job)
    d = job / f"state_{step:06d}"
    d.mkdir()
    (d / "lora.safetensors").write_bytes(b"x")
    return d


# ---- what holds output


def test_an_empty_or_leftovers_only_folder_is_free(tmp_path):
    assert not runplan.holds_output(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    assert not runplan.holds_output(tmp_path / "empty")
    assert not runplan.holds_output(leftovers(tmp_path / "logs"))
    probes = leftovers(tmp_path / "strangers") / "probes"
    probes.mkdir()
    (probes / "body_strangers.npz").write_bytes(b"x")  # set up at the start, no picture yet
    assert not runplan.holds_output(tmp_path / "strangers")


def test_a_saved_lora_a_state_or_pictures_take_the_folder(tmp_path):
    assert runplan.holds_output(saved_lora(tmp_path / "lora"))
    a_state(tmp_path / "state")
    assert runplan.holds_output(tmp_path / "state")
    samples = leftovers(tmp_path / "samples") / "samples" / "step_000100"
    samples.mkdir(parents=True)
    (samples / "0.png").write_bytes(b"x")
    assert runplan.holds_output(tmp_path / "samples")
    probes = leftovers(tmp_path / "probed") / "probes"
    probes.mkdir()
    (probes / "step000250_face_0.png").write_bytes(b"x")
    assert runplan.holds_output(tmp_path / "probed")


def test_a_run_stopped_before_its_first_step_leaves_the_name_free(tmp_path):
    """Start, see a wrong setting, stop: the run has its step-0 sample and probe (both made before training by
    default) and the probe's record, and nothing a later run would mix with."""
    job = leftovers(tmp_path / "early")
    for folder, name in (("samples", "step000000_0.png"), ("probes", "step000000_face_0.png"),
                         ("probes", "c1_step000000_body_0.png")):
        (job / folder).mkdir(exist_ok=True)
        (job / folder / name).write_bytes(b"x")
    (job / "likeness.jsonl").write_text("{}\n", encoding="utf-8")
    assert not runplan.holds_output(job)
    (job / "samples" / "step000100_0.png").write_bytes(b"x")  # it trained on: a sample after step 0
    assert runplan.holds_output(job)


def test_a_file_inside_the_state_folder_names_that_state(tmp_path):
    """resume_state may be a file of the state (Trainer.load_state takes its folder)."""
    out = tmp_path / "output"
    own = a_state(out / "run")
    assert runplan.job_name_taken("run", out, str(own / "lora.safetensors")) is None
    assert runplan.job_name_taken("other", out, str(own / "lora.safetensors")) is None  # (a free name)
    a_state(out / "third")
    assert runplan.job_name_taken("third", out, str(own / "lora.safetensors")) == "third_2"


def test_names_are_told_apart_as_their_folders_are(tmp_path, monkeypatch):
    """On Windows `Portrait` and `portrait` are one folder, so one of them running or queued takes both."""
    monkeypatch.setattr(runplan.os.path, "normcase", str.lower)
    assert runplan.name_in("portrait", {"Portrait"}) and not runplan.name_in("portrait", {"landscape"})
    assert runplan.next_free_name("run", {"RUN_2"}, tmp_path) == "run_3"
    monkeypatch.setattr(runplan.os.path, "normcase", lambda s: s)
    assert not runplan.name_in("portrait", {"Portrait"})


def test_carrying_on_from_its_own_state_is_allowed_another_jobs_is_not(tmp_path):
    out = tmp_path / "output"
    own = a_state(out / "run")
    other = a_state(out / "other")
    assert runplan.job_name_taken("run", out, str(own)) is None
    assert runplan.job_name_taken("run", out, str(other)) == "run_2"
    assert runplan.job_name_taken("run", out, "") == "run_2"
    assert runplan.job_name_taken("fresh", out, str(other)) is None  # a new name may start from any state


def test_the_free_name_skips_busy_names_and_taken_folders(tmp_path):
    out = tmp_path / "output"
    saved_lora(out / "run")
    a_state(out / "run_2")
    leftovers(out / "run_4")  # saved nothing: free
    assert runplan.next_free_name("run", set(), out) == "run_3"
    assert runplan.next_free_name("run", {"run_3"}, out) == "run_4"
    assert runplan.next_free_name("run", {"run_3", "run_4"}, out) == "run_5"
    # a name that ends in _N counts on from N
    assert runplan.next_free_name("run_2", set(), out) == "run_3"
    assert runplan.next_free_name("run_007", set(), out) == "run_008"
    assert runplan.next_free_name("v2_lora", set(), out) == "v2_lora_2"


# ---- the WebUI


@pytest.fixture
def ui(tmp_path, monkeypatch):
    """webui with its job file and queue in tmp_path, nothing running, and Popen faked (no run starts)."""
    import webui
    from rulztrainer.train_queue import TrainQueue

    monkeypatch.setattr(webui, "JOB_FILE", tmp_path / "job.json")
    monkeypatch.setattr(webui, "QUEUE", TrainQueue(tmp_path / "queue.json"))
    monkeypatch.setattr(webui, "_proc", None)
    monkeypatch.setattr(webui, "_adopted", None)
    monkeypatch.setattr(webui, "_job_dir", None)
    started = []

    class FakeProc:
        pid = 999999

        def poll(self):
            return None

    def fake_popen(args, **kw):
        started.append(args)
        return FakeProc()

    monkeypatch.setattr(webui.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(webui, "_record_job", lambda pid, job_dir: None)
    data = tmp_path / "data"
    data.mkdir()

    def cfg(name="run", **kw):
        return TrainConfig(job_name=name, output_dir=str(tmp_path / "output"), dataset_dir=str(data), **kw)

    return webui, cfg, started


def snapshot(folder: Path) -> dict:
    return {str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns) for p in folder.rglob("*")}


def test_queue_and_start_refuse_a_taken_name_and_touch_nothing(ui):
    webui, cfg, started = ui
    job = saved_lora(cfg().job_dir())
    before = snapshot(job)
    ok, msg = webui.queue_run(cfg())
    assert not ok and "'run_2' is free" in msg and "another job name" in msg
    assert webui.QUEUE.snapshot()["waiting"] == []
    msg = webui.start_training(cfg())
    assert msg.startswith("Not started") and "'run_2' is free" in msg
    assert started == [] and snapshot(job) == before and webui._job_dir is None
    # the queue's own start (a run queued before the folder filled) is refused the same way, and says so
    ok, msg = webui._start_queued(cfg().to_dict())
    assert not ok and "run_2" in msg and snapshot(job) == before


def test_the_wizards_review_names_a_taken_job_name(ui):
    webui, cfg, _ = ui
    data = {"images": 3, "captioned": 3}
    assert not any("is free" in text for _, text in webui.review_issues(cfg(), None, data, 1))
    saved_lora(cfg().job_dir())
    errors = [text for kind, text in webui.review_issues(cfg(), None, data, 1) if kind == "error"]
    assert any("'run_2' is free" in text for text in errors)


def test_a_free_name_and_a_leftovers_folder_start(ui):
    webui, cfg, started = ui
    leftovers(cfg("run").job_dir())
    assert webui.start_training(cfg("run")).startswith("Started")
    assert len(started) == 1


def test_a_free_name_is_queued(ui):
    webui, cfg, _ = ui
    saved_lora(cfg("run").job_dir())
    ok, msg = webui.queue_run(cfg("run_2"))
    assert ok and [e["job"] for e in webui.QUEUE.snapshot()["waiting"]] == ["run_2"]
    # a waiting name is busy: the suggestion skips it
    ok, msg = webui.queue_run(cfg("run"))
    assert not ok and "'run_3' is free" in msg


def test_resume_from_its_own_state_starts(ui):
    webui, cfg, started = ui
    state = a_state(cfg().job_dir())
    assert webui.start_training(cfg(resume_state=str(state))).startswith("Started")
    assert len(started) == 1


def test_resume_from_another_jobs_state_is_refused(ui):
    webui, cfg, started = ui
    other = a_state(cfg("other").job_dir())
    saved_lora(cfg("run").job_dir())
    msg = webui.start_training(cfg("run", resume_state=str(other)))
    assert msg.startswith("Not started") and "run_2" in msg and started == []


def test_out_of_memory_recoverys_restart_carries_on(ui):
    webui, cfg, started = ui
    saved_lora(cfg().job_dir())  # a restart with nothing to resume from, in its own folder
    assert webui._start_training(cfg(), recovery_note="[oom recovery] try 1 of 3\n").startswith("Started")
    assert len(started) == 1
