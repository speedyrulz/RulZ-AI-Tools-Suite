"""Out-of-memory recovery (rulztrainer.oom_recovery, webui.recover_after_oom): what ran out, what is tried next, and the
WebUI starting the run again before the queue's next run. The failures are faked: status.json and console.log as a
run that ran out of memory leaves them."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import oom_recovery as R  # noqa: E402

CUDA_OOM = ("OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 23.99 GiB "
            "of which 1.10 GiB is free.")


def make_state(job: Path, step: int, complete: bool = True) -> str:
    d = job / f"state_{step:06d}"
    d.mkdir(parents=True)
    (d / "lora.safetensors").write_bytes(b"x")
    if complete:
        (d / "train_state.pt").write_bytes(b"x")
    return str(d)


# ---- what ran out


def test_the_error_the_run_ended_with_decides():
    assert R.classify(CUDA_OOM) == "gpu"
    assert R.classify("RuntimeError: CUDA error: out of memory") == "gpu"
    assert R.classify("MemoryError: ") == "ram"
    assert R.classify("OSError: [WinError 1455] The paging file is too small for this operation to complete") == "ram"
    # a probe that ran out earlier is in the console, but the run died of something else
    assert R.classify("ValueError: bad caption", console_tail="probe failed: " + CUDA_OOM) is None


def test_a_hard_kill_reads_the_console_and_the_exit_code():
    assert R.classify(None, console_tail="step 900\n" + CUDA_OOM, returncode=1) == "gpu"
    assert R.classify(None, console_tail="RuntimeError: DefaultCPUAllocator: not enough memory") == "ram"
    assert R.classify(None, "", returncode=-9, platform="linux") == "ram"  # the OOM killer, no message
    assert R.classify(None, "", returncode=137, platform="linux") == "ram"
    assert R.classify(None, "", returncode=-9, platform="win32") is None
    assert R.classify(None, "", returncode=1) is None


def test_a_run_you_killed_or_that_finished_is_never_a_memory_failure():
    assert R.classify(CUDA_OOM, killed_by_user=True) is None
    assert R.classify(None, "", returncode=-9, killed_by_user=True, platform="linux") is None
    assert R.classify(None, CUDA_OOM, returncode=0) is None


# ---- one step lighter


def test_lighter_goes_checkpointing_kept_blocks_then_block_swap_to_its_most():
    cfg = {"gradient_checkpointing": False, "checkpoint_keep_blocks": -1, "block_swap": 0}
    seen = []
    while True:
        new, change = R.lighter(cfg, 28)
        if new is None:
            break
        seen.append(change)
        cfg = new
    assert seen[:3] == ["gradient checkpointing on", "blocks keeping their activations auto -> 0", "block swap 0 -> 4"]
    assert seen[-1] == "block swap 24 -> 27"  # all blocks but one
    assert change == "block swap is at its most" and cfg["block_swap"] == 27


def test_lighter_without_block_swap_to_raise():
    base = {"gradient_checkpointing": True, "checkpoint_keep_blocks": 0}
    assert R.lighter({**base, "finetune": True}, 42) == (None, "a fine-tune has no block swap to raise")
    assert R.lighter({**base, "multi_gpu": "model_split"}, 42)[0] is None
    assert R.lighter(base, 0)[0] is None  # unknown block count
    new, change = R.lighter({**base, "block_swap": 30}, 42)
    assert new["block_swap"] == 36 and change == "block swap 30 -> 36"


# ---- the decision


def cfg_of(**kw):
    return {"save_state": True, "save_every": 250, "save_state_every": 0, "oom_retries": 3,
            "gradient_checkpointing": True, "checkpoint_keep_blocks": -1, "block_swap": 0, **kw}


def test_first_gpu_failure_carries_on_from_the_newest_state_on_the_same_settings(tmp_path):
    s1, s2 = make_state(tmp_path, 500), make_state(tmp_path, 812)
    d = R.plan(cfg_of(), {"step": 812, "states": [s1, s2]}, "gpu", [], 28)
    assert d["do"] == "continue" and not d["lighter"] and d["change"] == ""
    assert d["config"]["resume_state"] == s2 and d["config"]["checkpoint_keep_blocks"] == -1


def test_out_again_at_the_same_point_goes_lighter_and_far_past_it_does_not(tmp_path):
    s = make_state(tmp_path, 812)
    tries = [{"kind": "gpu", "step": 812}]
    cfg = cfg_of(resume_state=s)
    d = R.plan(cfg, {"step": 830, "states": []}, "gpu", tries, 28)  # failed again before saving: its own start state
    assert d["do"] == "continue" and d["lighter"] and d["config"]["resume_state"] == s
    assert d["config"]["checkpoint_keep_blocks"] == 0 and d["change"].endswith("-> 0")
    far = R.plan(cfg, {"step": 1400, "states": []}, "gpu", tries, 28)
    assert far["do"] == "continue" and not far["lighter"]


def test_nothing_saved_restarts_once_lighter_only_before_the_first_state_was_due():
    early = R.plan(cfg_of(gradient_checkpointing=False), {"step": 40, "states": []}, "gpu", [], 28)
    assert early["do"] == "restart" and early["config"]["gradient_checkpointing"] is True
    assert "resume_state" not in early["config"] or not early["config"]["resume_state"]
    late = R.plan(cfg_of(save_state=False), {"step": 900, "states": []}, "gpu", [], 28)
    assert late["do"] == "stop" and "throw away" in late["why"]
    again = R.plan(cfg_of(), {"step": 10, "states": []}, "gpu", [{"kind": "gpu", "step": 5}], 28)
    assert again["do"] == "stop"


def test_a_race_or_a_fine_tune_has_no_state_to_carry_on_from(tmp_path):
    s = make_state(tmp_path, 500)
    race = R.plan(cfg_of(race=True), {"step": 600, "states": [s]}, "gpu", [], 28)
    assert race["do"] == "stop"
    assert R.plan(cfg_of(race=True), {"step": 0}, "gpu", [], 28)["do"] == "restart"  # out while loading
    ft = R.plan(cfg_of(finetune=True), {"step": 0}, "gpu", [], 42)
    assert ft["do"] == "stop"


def test_system_memory_gets_one_try_on_the_same_settings(tmp_path):
    s = make_state(tmp_path, 500)
    d = R.plan(cfg_of(), {"step": 600, "states": [s]}, "ram", [], 28)
    assert d["do"] == "continue" and not d["lighter"] and d["config"]["block_swap"] == 0
    assert R.plan(cfg_of(), {"step": 650, "states": [s]}, "ram", [{"kind": "ram", "step": 600}], 28)["do"] == "stop"
    assert R.plan(cfg_of(), {"step": 5, "states": []}, "ram", [], 28)["do"] == "stop"


def test_the_tries_are_capped_and_zero_turns_it_off(tmp_path):
    s = make_state(tmp_path, 500)
    three = [{"kind": "gpu", "step": 100}, {"kind": "gpu", "step": 900}, {"kind": "gpu", "step": 1900}]
    assert R.plan(cfg_of(), {"step": 2600, "states": [s]}, "gpu", three, 28)["do"] == "stop"
    assert R.plan(cfg_of(oom_retries=0), {"step": 600, "states": [s]}, "gpu", [], 28)["do"] == "stop"
    assert R.plan(cfg_of(), {"step": 600, "states": [s]}, None, [], 28)["do"] == "none"


def test_nothing_lighter_left_stops_with_the_reason(tmp_path):
    s = make_state(tmp_path, 500)
    cfg = cfg_of(checkpoint_keep_blocks=0, block_swap=27)
    d = R.plan(cfg, {"step": 520, "states": [s]}, "gpu", [{"kind": "gpu", "step": 510}], 28)
    assert d["do"] == "stop" and "block swap is at its most" in d["why"]


def test_only_this_runs_complete_states_count(tmp_path):
    old = make_state(tmp_path, 2000)  # an earlier run of this job name: in the folder, not in this run's status
    torn = make_state(tmp_path, 300, complete=False)
    assert R.run_save({"states": [torn]}, {}) is None
    assert R.run_save({"states": []}, {"resume_state": old}) == old  # the state this try was started from
    assert R.run_save({"states": []}, {}) is None


def test_the_ledger_and_the_kept_logs(tmp_path):
    (tmp_path / "train.log").write_text("try 1", encoding="utf-8")
    (tmp_path / "console.log").write_text("console 1", encoding="utf-8")
    assert R.read_tries(tmp_path) == []
    tries = R.add_try(tmp_path, "gpu", 812, "carried on")
    assert len(tries) == 1 and R.read_tries(tmp_path)[0]["step"] == 812
    assert R.keep_logs(tmp_path, 1) == ["train_oom1.log", "console_oom1.log"]
    assert (tmp_path / "train_oom1.log").read_text(encoding="utf-8") == "try 1"
    assert "1. GPU ran out at step 812: carried on" in R.summary(tries)
    R.reset(tmp_path)
    assert R.read_tries(tmp_path) == []


# ---- the WebUI carrying it out

psutil = pytest.importorskip("psutil")


@pytest.fixture
def ui(tmp_path, monkeypatch):
    import webui
    from rulztrainer.train_queue import TrainQueue

    monkeypatch.setattr(webui, "JOB_FILE", tmp_path / "job.json")
    monkeypatch.setattr(webui, "QUEUE", TrainQueue(tmp_path / "queue.json"))
    monkeypatch.setattr(webui, "_watched", {})
    monkeypatch.setattr(webui, "_killed_pids", set())
    monkeypatch.setattr(webui, "_adopted", None)
    started = []
    sleepers = []

    def fake_start(cfg, recovery_note=None):
        started.append((cfg, recovery_note))
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        sleepers.append(p)
        webui._proc = p
        return f"Started job '{cfg.job_name}' (pid {p.pid})."

    monkeypatch.setattr(webui, "_start_training", fake_start)
    yield webui, started
    for p in sleepers:
        p.kill()
        p.wait()


def failed_job(webui, tmp_path, error=CUDA_OOM, states=1):
    from rulztrainer.config import TrainConfig

    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    cfg = TrainConfig(job_name="oomjob", output_dir=str(tmp_path / "out"), dataset_dir=str(data), model_arch="krea2")
    job = cfg.job_dir()
    job.mkdir(parents=True)
    cfg.save(job / "config.yaml")
    st = [make_state(job, 250 * (i + 1)) for i in range(states)]
    (job / "status.json").write_text(json.dumps({"stage": "error", "error": error, "done": True, "step": 812,
                                                 "states": st}), encoding="utf-8")
    (job / "train.log").write_text("the failed try\n", encoding="utf-8")
    (job / "console.log").write_text(error + "\n", encoding="utf-8")
    ended = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
    ended.wait()
    webui._proc, webui._job_dir = ended, job
    webui._watched.update(pid=ended.pid, job_dir=job)
    return job, st


def test_the_queue_starts_the_run_again_before_its_next_run(ui, tmp_path):
    webui, started = ui
    job, st = failed_job(webui, tmp_path)
    webui.QUEUE.add({"job_name": "next"})
    webui.queue_tick()
    assert len(started) == 1
    cfg, note = started[0]
    assert cfg.job_name == "oomjob" and cfg.resume_state == st[-1] and cfg.block_swap == 0
    assert note.startswith("[oom recovery] try 1 of 3") and "train_oom1.log" in note
    assert (job / "train_oom1.log").read_text(encoding="utf-8") == "the failed try\n"
    assert R.read_tries(job)[0]["kind"] == "gpu"
    assert [e["job"] for e in webui.QUEUE.snapshot()["waiting"]] == ["next"]  # still waiting: the retry runs
    webui.queue_tick()
    assert len(started) == 1  # handled once


def test_a_killed_run_goes_on_to_the_queue(ui, tmp_path):
    webui, started = ui
    job, _ = failed_job(webui, tmp_path)
    webui._killed_pids.add(webui._watched["pid"])
    webui.queue_tick()
    assert started == []  # no recovery, and the queue (empty here) starts nothing
    assert R.read_tries(job) == []


def test_no_retry_leaves_advice_in_the_log(ui, tmp_path):
    webui, started = ui
    job, _ = failed_job(webui, tmp_path, error="MemoryError: ", states=0)
    webui.queue_tick()
    assert started == []
    log = (job / "train.log").read_text(encoding="utf-8")
    assert "ran out of system memory at step 812 and is not retried" in log and "swap fewer blocks" in log
