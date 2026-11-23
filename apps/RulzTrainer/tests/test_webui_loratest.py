"""The WebUI's Test LoRAs tab, without a browser: the spec built from the form, the GPU a training job holds, the queue
hold while a test runs, the confined picture and export routes, the grid model, Start, Stop and Pick."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from nicegui import app  # noqa: E402

import webui  # noqa: E402
from rulztrainer import loratest  # noqa: E402

CONFIG = "job_name: {name}\nmodel_arch: flux\nmodel_path: m\nnetwork_type: lora\nlora_rank: 16\n"


def _job(root: Path, name: str, files=("final", "best", "step")) -> Path:
    d = root / "output" / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(CONFIG.format(name=name), encoding="utf-8")
    names = {"final": f"{name}.safetensors", "best": f"{name}_best.safetensors",
             "step": f"{name}_000500.safetensors"}
    for f in files:
        (d / names[f]).write_bytes(b"w")
    return d


def _status(tests: Path, tid: str, **fields) -> Path:
    d = tests / tid
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({"state": "rendering", "pid": os.getpid(), **fields}), encoding="utf-8")
    return d


def test_confined_file_refuses_everything_outside_the_test_folder(tmp_path):
    tests = tmp_path / "_tests"
    d = _status(tests, "t20261122-100000-abcd")
    (d / "images" / "c1").mkdir(parents=True)
    (d / "images" / "c1" / "a.png").write_bytes(b"png")
    (d / "spec.json").write_text("{}", encoding="utf-8")
    (tests / "secret.png").write_bytes(b"x")
    tid = d.name
    ok = webui.lt_confined_file(tid, "images/c1/a.png", webui.IMAGE_EXTS, root=tests, prefix="images")
    assert ok is not None and ok.name == "a.png"
    bad = [("t20261122-100000-abcd", "../secret.png"), ("t20261122-100000-abcd", "images/../../secret.png"),
           ("..", "x.png"), ("../_tests", "secret.png"), ("t20261122-100000-abcd", str(tests / "secret.png")),
           ("t20261122-100000-abcd", "images\\c1\\a.png"), ("t20261122-100000-abcd", "spec.json"),
           ("t20261122-100000-abcd", "images/c1/missing.png"), ("T-BAD", "images/c1/a.png"), ("", "images/c1/a.png"),
           ("t20261122-100000-abcd", "status.json"), ("t20261122-100000-abcd", "")]
    for test, rel in bad:
        assert webui.lt_confined_file(test, rel, webui.IMAGE_EXTS, root=tests, prefix="images") is None, (test, rel)
    # a picture outside images/ is refused by the prefix even with an image extension
    (d / "other.png").write_bytes(b"png")
    assert webui.lt_confined_file(tid, "other.png", webui.IMAGE_EXTS, root=tests, prefix="images") is None


def test_routes_serve_only_confined_files(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "lt_root", lambda: tmp_path / "_tests")
    tid = "t20261122-100000-abcd"
    d = _status(tmp_path / "_tests", tid)
    (d / "images" / "c1").mkdir(parents=True)
    (d / "images" / "c1" / "a.png").write_bytes(b"png")
    (d / "export").mkdir()
    (d / "export" / "grid.jpg").write_bytes(b"jpg")
    (d / "export" / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "_tests" / "leak.png").write_bytes(b"png")
    client = TestClient(app)
    assert client.get("/rz-test-image", params={"test": tid, "path": "images/c1/a.png"}).content == b"png"
    for path in ("../leak.png", "images/../../leak.png", "export/grid.jpg", "status.json"):
        assert client.get("/rz-test-image", params={"test": tid, "path": path}).status_code == 404
    assert client.get("/rz-test-image", params={"test": "..", "path": "leak.png"}).status_code == 404
    got = client.get("/rz-test-file", params={"test": tid, "name": "grid.jpg"})
    assert got.status_code == 200 and got.content == b"jpg"
    for name in ("notes.txt", "../status.json", "scores.xlsx", "report.pdf"):  # not exported / not allowed / absent
        assert client.get("/rz-test-file", params={"test": tid, "name": name}).status_code == 404


def test_live_test_needs_a_non_final_state_and_a_live_pid(tmp_path):
    tests = tmp_path / "_tests"
    tests.mkdir()
    assert webui.lt_live_test(tests) is None
    _status(tests, "t1", state="done")
    _status(tests, "t2", state="failed", pid=os.getpid())
    _status(tests, "t3", state="rendering", pid=2_000_000_000)  # a pid nobody has: the run died
    assert webui.lt_live_test(tests) is None
    _status(tests, "t4", state="rendering", pid=os.getpid())
    assert webui.lt_live_test(tests) == "t4"
    (tests / "t4" / "status.json").write_text(json.dumps({"state": "stopped"}), encoding="utf-8")
    assert webui.lt_live_test(tests) is None


def test_a_test_just_started_holds_until_its_status_is_too_old(tmp_path):
    tests = tmp_path / "_tests"
    d = _status(tests, "t5", state="queued", pid=None)
    mtime = (d / "status.json").stat().st_mtime
    assert webui.lt_live_test(tests, now=mtime + 5) == "t5"  # the gap before its process writes a status
    assert webui.lt_live_test(tests, now=mtime + loratest.QUEUED_START_GAP + 5) is None  # it never came up


def test_a_reused_pid_is_not_a_live_test(tmp_path):
    tests = tmp_path / "_tests"
    _status(tests, "t6", state="rendering", pid=os.getpid(), pid_created=1.0)  # another process started at that pid
    assert webui.lt_live_test(tests) is None


def test_queue_holds_while_a_test_is_live(tmp_path, monkeypatch):
    """queue_tick passes running or a live test to the queue, so nothing is started."""
    calls = []

    class FakeQueue:
        def advance(self, running, start):
            calls.append(running)

    monkeypatch.setattr(webui, "QUEUE", FakeQueue())
    monkeypatch.setattr(webui, "reattach_job", lambda: None)
    monkeypatch.setattr(webui, "_running", lambda: False)
    monkeypatch.setattr(webui, "_job_pid", lambda: None)
    monkeypatch.setattr(webui, "_watched", {})
    monkeypatch.setattr(webui, "lt_live_test", lambda *a, **k: None)
    webui.queue_tick()
    monkeypatch.setattr(webui, "lt_live_test", lambda *a, **k: "t1")
    webui.queue_tick()
    assert calls == [False, True]


def test_held_gpus_follow_the_jobs_config():
    gpus = ["cuda:0", "cuda:1"]
    assert webui.lt_held_gpus(False, {"device": "cuda:0"}, gpus) == {}
    assert list(webui.lt_held_gpus(True, {"device": "cuda:0"}, gpus)) == ["cuda:0"]
    assert list(webui.lt_held_gpus(True, {"device": "cuda:1"}, gpus)) == ["cuda:1"]
    both = webui.lt_held_gpus(True, {"multi_gpu": "data_parallel", "devices": "cuda:0,cuda:1"}, gpus)
    assert set(both) == {"cuda:0", "cuda:1"}
    probe = webui.lt_held_gpus(True, {"device": "cuda:0", "probe_async": True, "probe_device": "cuda:1"}, gpus)
    assert set(probe) == {"cuda:0", "cuda:1"} and "probe worker" in probe["cuda:1"]
    auto = webui.lt_held_gpus(True, {"device": "cuda:0", "sample_async": True, "devices": "cuda:0,cuda:1"}, gpus)
    assert set(auto) == {"cuda:0", "cuda:1"}
    # the config cannot be read: every GPU is refused, with the reason
    unknown = webui.lt_held_gpus(True, None, gpus)
    assert set(unknown) == set(gpus) and "could not be read" in unknown["cuda:0"]


def test_run_options_show_missing_files_and_cannot_pick_them(tmp_path):
    out = tmp_path / "output"
    kept = _job(tmp_path, "elena")
    _job(tmp_path, "loose", files=("final",))
    _job(tmp_path, "empty", files=())
    no_config = _job(tmp_path, "nocfg", files=("final",))
    (no_config / "config.yaml").unlink()
    runs = [{"id": "elena-1", "job": "elena", "started": 1.0}, {"id": "gone-1", "job": "gone", "started": 2.0},
            {"id": "empty-1", "job": "empty", "started": 3.0}, {"id": "nocfg-1", "job": "nocfg", "started": 4.0}]
    dirs = {"elena-1": kept, "gone-1": None, "empty-1": out / "empty", "nocfg-1": no_config}
    opts = webui.lt_run_options(runs, out, lambda r: dirs[r["id"]])
    by = {o["key"]: o for o in opts}
    assert {f["label"] for f in by["run:elena-1"]["files"]} == {"final", "best", "000500"}
    assert by["run:gone-1"]["problem"] == "No LoRA files left" and by["run:gone-1"]["files"] == []
    assert by["run:empty-1"]["problem"] == "No LoRA files left"
    assert by["run:nocfg-1"]["problem"] == "No config.yaml left"
    assert "dir:loose" in by and by["dir:loose"]["run_id"] is None  # a job folder without a history record
    assert "dir:elena" not in by and "dir:empty" not in by  # already a record / nothing to pick


def test_spec_from_the_form_validates_and_refuses_with_the_engines_text(tmp_path):
    a, b = _job(tmp_path, "a", files=("final",)), _job(tmp_path, "b", files=("final",))
    picks = [{"path": a / "a.safetensors", "config": a / "config.yaml", "label": "a final", "run_id": "a-1"},
             {"path": b / "b.safetensors", "config": b / "config.yaml", "label": "b final", "run_id": None}]
    prompts = [{"text": "a photo of x", "kind": "face"}, {"text": "x full body", "kind": "body"}]
    spec = webui.lt_build_spec("  my   test ", picks, prompts, [1, 2], [0.8, 1.0], [[0, 1]], ["cuda:0"], base=tmp_path)
    assert spec["name"] == "my test" and spec["contestants"][0]["lora"] == "output/a/a.safetensors"
    assert spec["soups"] == [{"id": "s1", "label": "soup c1+c2", "of": ["c1", "c2"], "weights": None}]
    assert spec["prompt_kinds"] == ["face", "body"] and spec["devices"] == ["cuda:0"]
    assert webui.lt_validate(spec, base=tmp_path) is None
    assert webui.lt_count(2, 1, [0.8, 1.0], prompts, [1, 2]) == 24
    # the engine's own refusals are shown as they are
    assert "strengths" in webui.lt_validate({**spec, "strengths": [3.0]}, base=tmp_path)
    assert "seeds" in webui.lt_validate({**spec, "seeds": [1, 1]}, base=tmp_path)
    (b / "config.yaml").write_text(CONFIG.format(name="b").replace("flux", "qwen"), encoding="utf-8")
    assert "base model" in webui.lt_validate(spec, base=tmp_path)


def test_number_lists_and_estimate():
    assert webui.lt_parse_numbers("1, 2  3;4", "Seeds", True) == [1, 2, 3, 4]
    assert webui.lt_parse_numbers("0.8,1", "Strengths", False) == [0.8, 1.0]
    with pytest.raises(ValueError, match="Seeds: 'x' is not a whole number"):
        webui.lt_parse_numbers("1 x", "Seeds", True)
    with pytest.raises(ValueError, match="not a whole number"):
        webui.lt_parse_numbers("1.5", "Seeds", True)
    assert webui.lt_estimate_text(84, 1).startswith("84 pictures, about 28 min to 84 min on 1 GPU")
    assert "2 GPUs" in webui.lt_estimate_text(84, 2)
    assert webui.lt_estimate_text(1, 1) == "1 picture, about 20 s to 60 s on 1 GPU"
    assert webui.lt_eta_text(125) == "2:05 left" and webui.lt_eta_text(None) == ""


def test_preset_prompts_fill_the_trigger_word():
    key = next(p.key for p in webui.promptcheck.presets_of("face") if "{trigger}" in p.template)
    text, kind = webui.lt_preset_prompt(key, "zxq", "a woman")
    assert "zxq" in text and "{trigger}" not in text
    assert kind == "face"
    assert webui.lt_preset_prompt("nope", "t", "s") is None
    mine = [{"name": "x", "kind": "body", "template": "{trigger} standing", "expect": ""}]
    assert webui.lt_preset_prompt("my:body:x", "zxq", "", mine=mine) == ("zxq standing", "body")
    assert webui.lt_preset_prompt("my:body:gone", "zxq", "", mine=mine) is None
    assert key in webui.lt_preset_options()


def test_start_writes_the_spec_launches_and_refuses_busy_gpus_and_a_second_test(tmp_path):
    a = _job(tmp_path, "a", files=("final",))
    picks = [{"path": a / "a.safetensors", "config": a / "config.yaml", "label": "a", "run_id": None}]
    spec = webui.lt_build_spec("t", picks, [{"text": "p", "kind": "both"}], [1], [1.0], [], ["cuda:0", "cuda:1"],
                               base=tmp_path)
    started = []
    tid, msg = webui.lt_start(spec, {"cuda:1": "the training job uses it"}, base=tmp_path,
                              launch=lambda d, b: started.append(d))
    assert tid is None and "cuda:1 is busy" in msg and not started
    assert not (tmp_path / "output" / "_tests").exists()  # nothing was made
    spec["devices"] = ["cuda:0"]
    tid, msg = webui.lt_start(spec, {}, base=tmp_path, launch=lambda d, b: started.append(d))
    assert tid and msg.startswith("Started")
    assert started == [tmp_path / "output" / "_tests" / tid]
    assert json.loads((started[0] / "spec.json").read_text(encoding="utf-8"))["name"] == "t"
    # the run's own status is still "queued" (fresh): a second test is refused
    again, msg = webui.lt_start(spec, {}, base=tmp_path, launch=lambda d, b: started.append(d))
    assert again is None and "still running" in msg and len(started) == 1


def test_start_that_cannot_launch_marks_the_test_failed(tmp_path):
    a = _job(tmp_path, "a", files=("final",))
    picks = [{"path": a / "a.safetensors", "config": a / "config.yaml", "label": "a", "run_id": None}]
    spec = webui.lt_build_spec("t", picks, [{"text": "p", "kind": "both"}], [1], [1.0], [], ["cuda:0"], base=tmp_path)

    def broken(d, b):
        raise OSError("no python")

    tid, msg = webui.lt_start(spec, {}, base=tmp_path, launch=broken)
    assert tid and "did not start" in msg
    st = json.loads((tmp_path / "output" / "_tests" / tid / "status.json").read_text(encoding="utf-8"))
    assert st["state"] == "failed" and "no python" in st["error"]
    assert webui.lt_live_test(tmp_path / "output" / "_tests") is None


def test_start_refuses_a_refused_spec_with_the_engines_reason(tmp_path):
    tid, msg = webui.lt_start({"contestants": [], "prompts": ["p"], "seeds": [1], "strengths": [1.0]}, {},
                              base=tmp_path, launch=lambda d, b: None)
    assert tid is None and msg.startswith("Not started:") and "contestant" in msg


def test_launch_runs_the_module_detached_with_a_log(tmp_path, monkeypatch):
    seen = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kw):
        seen.update(cmd=cmd, kw=kw)
        return FakeProc()

    monkeypatch.setattr(webui.subprocess, "Popen", fake_popen)
    pid = webui.lt_launch(tmp_path, tmp_path)
    assert pid == 4242 and seen["cmd"][1:4] == ["-m", "rulztrainer.loratest", "run"] and seen["cmd"][4] == str(tmp_path)
    assert (tmp_path / "run.log").is_file() and seen["kw"]["cwd"] == str(tmp_path)
    kw = seen["kw"]
    assert kw["stdin"] is webui.subprocess.DEVNULL and kw["stderr"] is webui.subprocess.STDOUT
    assert Path(kw["stdout"].name) == tmp_path / "run.log"  # the run's own handle on its log
    assert kw["env"]["PYTHONUNBUFFERED"] == "1"
    if sys.platform == "win32":
        assert seen["kw"]["creationflags"] & webui.subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert seen["kw"]["start_new_session"] is True


def test_stop_writes_the_stop_file(tmp_path):
    webui.lt_stop(tmp_path)
    assert (tmp_path / "STOP").is_file()


RESULTS = {"contestants": [{"id": "c1", "label": "run a final", "kind": "file"},
                           {"id": "s1", "label": "soup", "kind": "soup"}],
           "cells": [{"contestant": "c1", "strength": 1.0, "prompt": 0, "seed": 1, "image": "images/c1/s1.00_p0_seed1.png",
                      "face": 60.0, "body": None, "gold": 58.0, "combined": 59.0}],
           "columns": [{"contestant": "c1", "strength": 0.8, "combined": 50.0, "n": 1},
                       {"contestant": "c1", "strength": 1.0, "combined": 59.0, "n": 1}],
           "best_strength": {"c1": 1.0, "s1": 0.8}, "winner": {"contestant": "c1", "strength": 1.0, "combined": 59.0}}
SPEC = {"strengths": [0.8, 1.0], "prompts": ["p one", "p two"], "prompt_kinds": ["face", "body"], "seeds": [1, 2],
        "contestants": [{"id": "c1", "label": "run a final", "run_id": "a-1"}, {"id": "s1", "label": "soup"}]}


def test_grid_model_flags_the_winner_and_best_strength():
    m = webui.lt_grid_model(SPEC, RESULTS)
    assert [(c["contestant"], c["strength"], c["winner"], c["best"]) for c in m["columns"]] == [
        ("c1", 0.8, False, False), ("c1", 1.0, True, True), ("s1", 0.8, False, True), ("s1", 1.0, False, False)]
    assert len(m["rows"]) == 4 and m["rows"][0]["kind"] == "face" and m["rows"][2]["kind"] == "body"
    first = m["rows"][0]["cells"]
    assert first[1]["combined"] == 59.0 and first[0] is None and first[2] is None  # unrendered cells stay None
    assert webui.lt_score_text(first[1]) == "face 60.0 · body - · gold 58.0"
    assert webui.lt_test_url("t1", "images/a b.png") == "/rz-test-image?test=t1&path=images/a%20b.png"


def test_pick_labels_only_a_run_with_a_record(tmp_path):
    (tmp_path / "a-1.json").write_text("{}", encoding="utf-8")
    calls = []
    ok, msg = webui.lt_pick_label(SPEC, "c1", 1.0, set_label=lambda rid, label: calls.append((rid, label)) or True,
                                  root=tmp_path)
    assert ok and calls == [("a-1", "run a final @ 1.00")] and "Previous Runs" in msg
    ok, msg = webui.lt_pick_label(SPEC, "s1", 0.8, set_label=lambda *a: calls.append(a) or True, root=tmp_path)
    assert not ok and len(calls) == 1  # a soup has no run
    ok, _ = webui.lt_pick_label(SPEC, "c1", 1.0, set_label=lambda *a: False, root=tmp_path)
    assert not ok
    assert not webui.lt_pick_label(SPEC, "zz", 1.0, root=tmp_path)[0]
    # a record that does not exist (any more) is not labelled
    ok, msg = webui.lt_pick_label({**SPEC, "contestants": [{"id": "c1", "label": "x", "run_id": "gone-1"}]}, "c1", 1.0,
                                  set_label=lambda *a: calls.append(a) or True, root=tmp_path)
    assert not ok and "not in Previous Runs" in msg and len(calls) == 1


@pytest.mark.parametrize("bad", ["../a-1", "a/b", "a\\b", "..", "a-1/..", "", None, 5, "a-1\x00"])
def test_run_ids_with_a_slash_backslash_dots_or_no_record_are_refused(tmp_path, bad):
    (tmp_path / "a-1.json").write_text("{}", encoding="utf-8")
    (tmp_path.parent / "a-1.json").write_text("{}", encoding="utf-8")  # a file one level up must not be reachable
    assert not webui.lt_valid_run_id(bad, tmp_path)
    assert webui.lt_valid_run_id("a-1", tmp_path)


def test_pick_after_a_pull_uses_the_run_ids_kept_on_this_pc_not_the_pulled_spec(tmp_path):
    (tmp_path / "mine.json").write_text("{}", encoding="utf-8")
    pulled = {"contestants": [{"id": "c1", "label": "j best", "run_id": "../../evil"},
                              {"id": "c2", "label": "j final", "run_id": "mine"}]}
    calls = []
    set_label = lambda rid, label: calls.append((rid, label)) or True  # noqa: E731
    ok, _ = webui.lt_pick_label(pulled, "c1", 1.0, set_label=set_label, run_ids={"c1": "mine"}, root=tmp_path)
    assert ok and calls == [("mine", "j best @ 1.00")]
    # the spec's own id is ignored when this PC kept none for the contestant
    ok, _ = webui.lt_pick_label(pulled, "c2", 1.0, set_label=set_label, run_ids={"c1": "mine", "c2": None}, root=tmp_path)
    assert not ok and len(calls) == 1
    ok, _ = webui.lt_pick_label(pulled, "c1", 1.0, set_label=set_label, run_ids={"c1": "../../evil"}, root=tmp_path)
    assert not ok and len(calls) == 1


def test_the_tab_sits_after_previous_runs_and_is_in_the_remembered_tab_names():
    src = Path(webui.__file__).read_text(encoding="utf-8")
    assert src.index('tab_history = ui.tab("Previous Runs"') < src.index("tab_loratest = ui.tab(LT_TAB") \
        < src.index('tab_fleet = ui.tab("Fleet"')
    assert "LT_TAB, \"Fleet\"" in src


# ---------------------------------------------------------------- grid pages, instances and the Fleet downloads folder


class FakeDone:
    def __init__(self, out="", err="", code=0):
        self.stdout, self.stderr, self.returncode = out.encode(), err.encode(), code


class FakeRemote:
    """What rulz_remote.Remote offers the tab: agent(...) and agent_json(...), answered from a table."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def agent(self, *args, config=None, retry=True, timeout="auto"):
        self.calls.append((args, config, retry))
        got = self.answers[args[0]]
        return got(args, config) if callable(got) else got

    def agent_json(self, *args):
        self.calls.append((args, None, True))
        return self.answers[args[0]]


def _fleet(tmp_path, monkeypatch) -> Path:
    pulls = tmp_path / "pulls"
    pulls.mkdir()
    monkeypatch.setattr(webui, "lt_pulls_root", lambda: pulls)
    return pulls


def test_export_name_pattern_serves_grid_pages_only_by_exact_name(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "lt_root", lambda: tmp_path / "_tests")
    tid = "t20261122-100000-abcd"
    d = _status(tmp_path / "_tests", tid)
    (d / "export").mkdir()
    for n in ("grid.jpg", "grid_01.jpg", "grid_12.jpg", "grid_1.jpg", "grid_001.jpg", "grid.png", "xgrid.jpg"):
        (d / "export" / n).write_bytes(b"x")
    client = TestClient(app)
    for ok in ("grid.jpg", "grid_01.jpg", "grid_12.jpg"):
        assert client.get("/rz-test-file", params={"test": tid, "name": ok}).status_code == 200, ok
    for bad in ("grid_1.jpg", "grid_001.jpg", "grid.png", "xgrid.jpg", "grid_01.jpg/../../status.json", "../grid.jpg"):
        assert client.get("/rz-test-file", params={"test": tid, "name": bad}).status_code == 404, bad


def test_fleet_tests_root_accepts_only_digit_instances_inside_the_downloads_folder(tmp_path, monkeypatch):
    pulls = _fleet(tmp_path, monkeypatch)
    assert webui.lt_tests_root("123") == pulls / "123" / "_tests"
    for bad in ("abc", "1/2", "..", "../x", "12 3", "1234567890123", "123/../456"):
        assert webui.lt_tests_root(bad) is None, bad
    assert webui.lt_tests_root(None) == webui.lt_root()
    (tmp_path / "outside").mkdir()
    try:
        os.symlink(tmp_path / "outside", pulls / "777", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks on this system")
    assert webui.lt_tests_root("777") is None  # a link that leads out of the downloads folder


def test_fleet_routes_serve_only_tests_under_the_instance_folder(tmp_path, monkeypatch):
    pulls = _fleet(tmp_path, monkeypatch)
    monkeypatch.setattr(webui, "lt_root", lambda: tmp_path / "_tests")
    tid = "t20261122-100000-abcd"
    d = _status(pulls / "42" / "_tests", tid)
    (d / "images" / "c1").mkdir(parents=True)
    (d / "images" / "c1" / "a.png").write_bytes(b"fleetpng")
    (d / "export").mkdir()
    (d / "export" / "report.pdf").write_bytes(b"pdf")
    (pulls / "42" / "secret.png").write_bytes(b"s")
    (pulls / "secret.png").write_bytes(b"s")
    client = TestClient(app)
    got = client.get("/rz-test-image", params={"test": tid, "path": "images/c1/a.png", "inst": "42"})
    assert got.status_code == 200 and got.content == b"fleetpng"
    assert client.get("/rz-test-file", params={"test": tid, "name": "report.pdf", "inst": "42"}).content == b"pdf"
    for params in ({"test": tid, "path": "images/c1/a.png"},  # not a test of this PC
                   {"test": tid, "path": "images/c1/a.png", "inst": "43"},
                   {"test": tid, "path": "images/c1/a.png", "inst": ".."},
                   {"test": tid, "path": "images/c1/a.png", "inst": "42/_tests"},
                   {"test": tid, "path": "../../secret.png", "inst": "42"},
                   {"test": "../..", "path": "secret.png", "inst": "42"},
                   {"test": "42", "path": "_tests/" + tid + "/images/c1/a.png", "inst": "42"}):
        assert client.get("/rz-test-image", params=params).status_code == 404, params
    assert client.get("/rz-test-file", params={"test": tid, "name": "report.pdf", "inst": "x"}).status_code == 404


def test_a_vast_run_is_pickable_from_its_downloaded_folder(tmp_path):
    pulls = tmp_path / "pulls"
    folder = pulls / "42" / "job1"
    folder.mkdir(parents=True)
    (folder / "job1.safetensors").write_bytes(b"w")
    (folder / "job1_best.safetensors").write_bytes(b"w")
    (folder / "config.yaml").write_text(CONFIG.format(name="job1"), encoding="utf-8")  # no status.json
    src = {"kind": "vast", "instance": 42}
    with_dir = {"id": "vast-a", "job": "job1", "started": 1.0, "job_dir": str(folder), "source": src}
    no_dir = {"id": "vast-b", "job": "job1", "started": 2.0, "job_dir": "", "source": src}
    other = {"id": "vast-c", "job": "gone", "started": 3.0, "job_dir": "", "source": src}
    local = {"id": "plain", "job": "nope", "started": 4.0}
    for rec in (with_dir, no_dir):
        assert webui.lt_job_dir_of(rec, [rec], pulls) == folder
    assert webui.lt_job_dir_of(other, [other], pulls) is None
    assert webui.lt_job_dir_of(local, [local], pulls) is None  # not a vast run: only history_job_dir decides
    out = tmp_path / "output"
    out.mkdir()
    runs = [with_dir, other]
    by = {o["run_id"]: o for o in webui.lt_run_options(runs, out, lambda r: webui.lt_job_dir_of(r, runs, pulls))}
    assert len(by["vast-a"]["files"]) == 2 and by["vast-a"]["problem"] is None
    assert by["vast-c"]["problem"] == "No LoRA files left"


def test_instance_listing_becomes_picks_and_a_spec_with_instance_paths():
    jobs = [{"job": "old", "modified": 1}, {"job": "new", "modified": 9, "running": True}, {"nojob": 1}]
    assert [j["job"] for j in webui.lt_remote_jobs(jobs)] == ["new", "old"]
    assert webui.lt_remote_jobs(jobs)[0]["label"] == "new (running now)"
    listing = {"base": "/workspace/RulzTrainer/output", "files": [
        {"path": "new/new.safetensors", "mtime": 5}, {"path": "new/new_best.safetensors", "mtime": 4},
        {"path": "new/new_000500.safetensors", "mtime": 3}, {"path": "new/config.yaml", "mtime": 1},
        {"path": "new/train.log"}, {"path": "other/x.safetensors"}, {"path": "new/sub/y.safetensors"}]}
    info = webui.lt_remote_job_files(listing, "new")
    assert [f["label"] for f in info["files"]] == ["000500", "best", "final"]
    assert info["files"][2]["spec"] == "output/new/new.safetensors" and info["config"] == "output/new/config.yaml"
    assert info["problem"] is None
    assert webui.lt_remote_job_files({"files": [{"path": "j/config.yaml"}]}, "j")["problem"] == "No LoRA files left"
    assert webui.lt_remote_job_files({"files": [{"path": "j/j.safetensors"}]}, "j")["problem"] == "No config.yaml left"
    picks = [{"path": f["spec"], "config": info["config"], "spec_lora": f["spec"], "spec_config": info["config"],
              "run_id": None, "label": f"new {f['label']}"} for f in info["files"][1:]]
    spec = webui.lt_build_spec("remote", picks, [{"text": "p", "kind": "face"}], [1], [1.0], [], ["cuda:0", "cuda:1"])
    assert spec["contestants"][0]["lora"] == "output/new/new_best.safetensors"
    assert spec["contestants"][1]["config"] == "output/new/config.yaml"  # the instance's paths, not this PC's
    assert webui.lt_remote_precheck(spec) is None
    for bad, text in (({"contestants": []}, "Pick at least one"), ({"prompts": []}, "Add at least one prompt"),
                      ({"devices": []}, "GPU"), ({"seeds": [1, 1]}, "seeds"), ({"strengths": [3.0]}, "outside"),
                      ({"soups": [{"of": ["c1"]}]}, "soup"), ({"seeds": list(range(3000))}, "more than")):
        assert text in webui.lt_remote_precheck({**spec, **bad}), bad


def test_remote_start_shows_the_instances_refusal_verbatim_and_follows_a_started_test(tmp_path, monkeypatch):
    pulls = _fleet(tmp_path, monkeypatch)
    picks = [{"path": "output/j/j.safetensors", "config": "output/j/config.yaml", "spec_lora": "output/j/j.safetensors",
              "spec_config": "output/j/config.yaml", "run_id": None, "label": "j"}]
    spec = webui.lt_build_spec("r", picks, [{"text": "p", "kind": "both"}], [1], [1.0], [], ["cuda:0"])
    refusal = "A training job is running; its GPU use cannot be determined safely"
    refusing = FakeRemote({"test-start": FakeDone(err=refusal + "\n", code=1)})
    tid, msg = webui.lt_remote_start(refusing, 42, spec, pulls)
    assert tid is None and msg == "Not started: " + refusal
    assert refusing.calls[0][0] == ("test-start",) and refusing.calls[0][2] is False  # never retried
    assert json.loads(refusing.calls[0][1])["contestants"][0]["lora"] == "output/j/j.safetensors"
    assert not (pulls / "42").exists()
    junk = FakeRemote({"test-start": FakeDone(out="not json")})
    assert webui.lt_remote_start(junk, 42, spec, pulls)[0] is None
    good = FakeRemote({"test-start": FakeDone(out=json.dumps({"test_id": "t20261122-140000-beef", "state": "queued"}))})
    tid, msg = webui.lt_remote_start(good, 42, spec, pulls)
    assert tid == "t20261122-140000-beef" and "instance 42" in msg
    dest = pulls / "42" / "_tests" / tid
    mark = json.loads((dest / "remote.json").read_text(encoding="utf-8"))
    assert mark == {"instance": "42", "test_id": tid, "pulled": False, "run_ids": {"c1": None}}
    assert json.loads((dest / "spec.json").read_text(encoding="utf-8"))["test_id"] == tid
    # a local precheck refusal never reaches the instance
    silent = FakeRemote({})
    assert webui.lt_remote_start(silent, 42, {**spec, "prompts": []}, pulls)[0] is None and not silent.calls


def test_remote_status_mapping_poll_stop_and_pull(tmp_path, monkeypatch):
    pulls = _fleet(tmp_path, monkeypatch)
    tid = "t20261122-140000-beef"
    dest = pulls / "42" / "_tests" / tid
    dest.mkdir(parents=True)
    webui.lt_remote_mark(dest, 42, tid)
    running = {"state": "rendering", "pid": 99999999, "done": 3, "total": 8, "current": "c1 at 1.00, prompt 1, seed 1"}
    remote = FakeRemote({"test-status": FakeDone(out=json.dumps(running)), "test-stop": FakeDone(out="{}")})
    status, err = webui.lt_remote_poll(remote, dest, tid)
    assert err == "" and status["done"] == 3 and remote.calls[0][0] == ("test-status", tid)
    # the instance's pid means nothing here: a remote folder keeps the instance's state instead of reading "failed"
    assert webui.lt_state(dest, status, webui.lt_status_age(dest)) == "rendering"
    local = _status(tmp_path / "_tests", "t1", state="rendering", pid=99999999)
    assert webui.lt_state(local, webui.lt_read_json(local / "status.json")) == "failed"
    assert webui.lt_state(dest, None) == "unknown"
    # an instance that does not answer: no status, its text
    down = FakeRemote({"test-status": FakeDone(err="ssh gave no answer within 120 s", code=124)})
    assert webui.lt_remote_poll(down, dest, tid) == (None, "ssh gave no answer within 120 s")
    assert webui.lt_remote_stop(remote, tid) == (True, "{}")
    nope = FakeRemote({"test-stop": FakeDone(err="No test named x", code=1)})
    assert webui.lt_remote_stop(nope, tid) == (False, "No test named x")

    fetched = []

    class FakeRR:
        moving = None

        @classmethod
        def fleet_move_in_progress(cls):
            return cls.moving

        @staticmethod
        def pull_plan(files, d):
            return [f for f in files if not (Path(d) / f["path"]).is_file()]

        @staticmethod
        def fetch(r, base, paths, d):
            fetched.append((base, paths, d))

    listing = {"base": "/ws/output/_tests/" + tid, "files": [{"path": "results.json", "size": 1},
                                                          {"path": "status.json", "size": 1}]}
    puller = FakeRemote({"files": listing})
    assert webui.lt_remote_pull(puller, dest, tid, rr=FakeRR) == 1  # status.json is already here (the poll wrote it)
    assert fetched[0][1] == ["results.json"] and fetched[0][2] == dest
    assert puller.calls[0][0] == ("files", "--tests", tid)
    FakeRR.moving = {"pid": 1}
    with pytest.raises(OSError, match="being moved"):
        webui.lt_remote_pull(puller, dest, tid, rr=FakeRR)


def test_saved_list_shows_downloaded_tests_with_their_instance(tmp_path, monkeypatch):
    pulls = _fleet(tmp_path, monkeypatch)
    mine = {"id": "t20261122-120000-aaaa", "test_id": "t20261122-120000-aaaa", "name": "mine", "state": "done",
            "created": "x"}
    monkeypatch.setattr(loratest, "list_tests", lambda base=None: [mine])
    d = pulls / "42" / "_tests" / "t20261122-130000-bbbb"
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"name": "theirs", "created": "2026-11-22T13:00:00Z"}), encoding="utf-8")
    (d / "status.json").write_text(json.dumps({"state": "rendering", "pid": 99999999}), encoding="utf-8")
    webui.lt_remote_mark(d, 42, d.name)
    (pulls / "42" / "_tests" / "not a test").mkdir()
    (pulls / "notes").mkdir()
    rows = webui.lt_list_tests(pulls)
    assert [(r["name"], r["instance"], r["state"]) for r in rows] == [("theirs", "42", "rendering"), ("mine", None, "done")]
    assert webui.lt_tests_root("42") / d.name == d  # the open view of a downloaded test reads the instance's folder


# ---------------------------------------------------------------- a hostile or broken listing from an instance


class PullRR:
    """A rulz_remote stand-in that records what would be fetched."""

    def __init__(self):
        self.fetched = []

    def fleet_move_in_progress(self):
        return None

    @staticmethod
    def pull_plan(files, d):
        return list(files)

    def fetch(self, r, base, paths, d):
        self.fetched.append((base, paths, d))


@pytest.mark.parametrize("bad", [
    "../escape.txt", "images/../../escape.txt", "..\\escape.txt", "a\\..\\..\\escape.txt", "/etc/passwd",
    "C:\\Windows\\win.ini", "C:/Windows/win.ini", "\\\\server\\share\\x", "line\nbreak.png", "carriage\rreturn.png",
    "nul\x00byte.png", "", None, 7, ["x"]])
def test_pull_refuses_the_whole_listing_when_one_path_is_unsafe(tmp_path, monkeypatch, bad):
    pulls = _fleet(tmp_path, monkeypatch)
    tid = "t20261122-140000-beef"
    dest = pulls / "42" / "_tests" / tid
    dest.mkdir(parents=True)
    rr = PullRR()
    listing = {"base": "/ws/output/_tests/" + tid, "files": [{"path": "results.json", "size": 1},
                                                           {"path": bad, "size": 1}]}
    with pytest.raises(OSError, match="Unsafe"):
        webui.lt_remote_pull(FakeRemote({"files": listing}), dest, tid, rr=rr)
    assert rr.fetched == []  # nothing, not even the good entry, was fetched


def test_pull_refuses_a_listing_that_is_not_a_list_or_has_non_object_entries(tmp_path, monkeypatch):
    pulls = _fleet(tmp_path, monkeypatch)
    tid = "t20261122-140000-beef"
    dest = pulls / "42" / "_tests" / tid
    dest.mkdir(parents=True)
    for listing in ({"base": "/b", "files": "results.json"}, {"base": "/b", "files": ["results.json"]},
                    {"base": "/b", "files": [None]}, {"base": "/b"}):
        with pytest.raises(OSError, match="Unsafe"):
            webui.lt_remote_pull(FakeRemote({"files": listing}), dest, tid, rr=PullRR())


def test_pull_refuses_a_path_that_resolves_outside_the_folder_through_a_link(tmp_path, monkeypatch):
    pulls = _fleet(tmp_path, monkeypatch)
    tid = "t20261122-140000-beef"
    dest = pulls / "42" / "_tests" / tid
    dest.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, dest / "images", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks on this system")
    rr = PullRR()
    listing = {"base": "/b", "files": [{"path": "images/a.png", "size": 1}]}
    with pytest.raises(OSError, match="Unsafe"):
        webui.lt_remote_pull(FakeRemote({"files": listing}), dest, tid, rr=rr)
    assert rr.fetched == []


def test_pull_refuses_a_destination_that_is_a_link_or_outside_the_downloads_folder(tmp_path, monkeypatch):
    pulls = _fleet(tmp_path, monkeypatch)
    tid = "t20261122-140000-beef"
    listing = {"base": "/b", "files": [{"path": "results.json", "size": 1}]}
    outside = tmp_path / "outside"
    (outside / "_tests" / tid).mkdir(parents=True)

    # a destination that is simply not under the downloads folder
    with pytest.raises(OSError, match="outside"):
        webui.lt_remote_pull(FakeRemote({"files": listing}), outside / "_tests" / tid, tid, rr=PullRR())

    def linked(parent_of_link: Path, name: str, target: Path) -> bool:
        parent_of_link.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(target, parent_of_link / name, target_is_directory=True)
        except (OSError, NotImplementedError):
            return False
        return True

    # the instance folder is a link out of the downloads folder
    if not linked(pulls, "50", outside):
        pytest.skip("no symlinks on this system")
    with pytest.raises(OSError, match="link"):
        webui.lt_remote_pull(FakeRemote({"files": listing}), pulls / "50" / "_tests" / tid, tid, rr=PullRR())
    # the _tests folder is a link
    (pulls / "51").mkdir()
    assert linked(pulls / "51", "_tests", outside / "_tests")
    with pytest.raises(OSError, match="link"):
        webui.lt_remote_pull(FakeRemote({"files": listing}), pulls / "51" / "_tests" / tid, tid, rr=PullRR())
    # the test folder itself is a link
    (pulls / "52" / "_tests").mkdir(parents=True)
    assert linked(pulls / "52" / "_tests", tid, outside / "_tests" / tid)
    with pytest.raises(OSError, match="link"):
        webui.lt_remote_pull(FakeRemote({"files": listing}), pulls / "52" / "_tests" / tid, tid, rr=PullRR())


def test_pull_refuses_a_bad_test_id_and_accepts_a_clean_listing(tmp_path, monkeypatch):
    pulls = _fleet(tmp_path, monkeypatch)
    tid = "t20261122-140000-beef"
    dest = pulls / "42" / "_tests" / tid
    listing = {"base": "/ws/output/_tests/" + tid, "files": [{"path": "results.json", "size": 1},
                                                          {"path": "images/c1/s1.00_p0_seed1.png", "size": 2}]}
    with pytest.raises(OSError, match="Bad test id"):
        webui.lt_remote_pull(FakeRemote({"files": listing}), dest, "../x", rr=PullRR())
    rr = PullRR()
    assert webui.lt_remote_pull(FakeRemote({"files": listing}), dest, tid, rr=rr) == 2  # dest need not exist yet
    assert rr.fetched[0][1] == ["results.json", "images/c1/s1.00_p0_seed1.png"] and rr.fetched[0][2] == dest.resolve()


@pytest.mark.parametrize("which", ["dest", "tests", "instance"])
def test_pull_refuses_each_link_even_where_this_system_cannot_make_one(tmp_path, monkeypatch, which):
    pulls = _fleet(tmp_path, monkeypatch)
    tid = "t20261122-140000-beef"
    dest = pulls / "42" / "_tests" / tid
    dest.mkdir(parents=True)
    flagged = {"dest": dest, "tests": dest.parent, "instance": dest.parent.parent}[which]
    real = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == flagged or real(self))
    listing = {"base": "/b", "files": [{"path": "results.json", "size": 1}]}
    rr = PullRR()
    with pytest.raises(OSError, match="link"):
        webui.lt_remote_pull(FakeRemote({"files": listing}), dest, tid, rr=rr)
    assert rr.fetched == []


# ---------------------------------------------------------------- review round: start lock, OOM restart, GPUs, links


def test_start_holds_the_start_lock_and_asks_the_job_gpus_again_inside_it(tmp_path):
    a = _job(tmp_path, "a", files=("final",))
    picks = [{"path": a / "a.safetensors", "config": a / "config.yaml", "label": "a", "run_id": None}]
    spec = webui.lt_build_spec("t", picks, [{"text": "p", "kind": "both"}], [1], [1.0], [], ["cuda:0"], base=tmp_path)
    owned = []

    def held_fn():
        owned.append(("held", webui._start_lock._is_owned()))
        return {}

    def launch(d, b):
        owned.append(("launch", webui._start_lock._is_owned()))

    tid, _ = webui.lt_start(spec, {}, base=tmp_path, launch=launch, held_fn=held_fn)
    assert tid and owned == [("held", True), ("launch", True)]
    assert not webui._start_lock._is_owned()  # released afterwards
    # the GPUs the caller saw earlier do not count: the ones asked inside the lock decide
    tests = tmp_path / "output" / "_tests"
    for d in tests.iterdir():
        (d / "status.json").write_text(json.dumps({"state": "done"}), encoding="utf-8")
    again, msg = webui.lt_start(spec, {}, base=tmp_path, launch=launch,
                                held_fn=lambda: {"cuda:0": "the training job uses it"})
    assert again is None and "cuda:0 is busy" in msg


def test_a_training_start_is_refused_while_a_test_is_live(monkeypatch):
    monkeypatch.setattr(webui, "reattach_job", lambda: None)
    monkeypatch.setattr(webui, "_running", lambda: False)
    monkeypatch.setattr(webui, "lt_live_test", lambda *a, **k: "t9")
    msg = webui._start_training(webui.TrainConfig())
    assert msg.startswith("Not started: LoRA test t9 is running")


def _oom_setup(monkeypatch, tmp_path, live):
    """queue_tick with a watched run that ended out of memory; returns (restarts, tries added, queue calls)."""
    job = tmp_path / "job"
    job.mkdir(exist_ok=True)
    (job / "status.json").write_text(json.dumps({"error": "CUDA out of memory", "step": 50}), encoding="utf-8")
    restarts, tries, calls = [], [], []

    class FakeQueue:
        def advance(self, running, start):
            calls.append(running)

    class FakeConfig:
        job_name = "job"
        oom_retries = 3

        @classmethod
        def load(cls, path):
            return cls()

        @classmethod
        def from_dict(cls, d):
            return cls()

        def to_dict(self):
            return {}

    monkeypatch.setattr(webui, "QUEUE", FakeQueue())
    monkeypatch.setattr(webui, "reattach_job", lambda: None)
    monkeypatch.setattr(webui, "_running", lambda: False)
    monkeypatch.setattr(webui, "_job_pid", lambda: None)
    monkeypatch.setattr(webui, "_watched", {"pid": 123, "job_dir": job})
    monkeypatch.setattr(webui, "_proc", None)
    monkeypatch.setattr(webui, "lt_live_test", lambda *a, **k: live)
    monkeypatch.setattr(webui, "TrainConfig", FakeConfig)
    monkeypatch.setattr(webui, "validated_config", lambda c: c)
    monkeypatch.setattr(webui, "_oom_block_count", lambda cfg: 0)
    monkeypatch.setattr(webui, "_start_training", lambda cfg, recovery_note=None: restarts.append(recovery_note) or "Started job")
    monkeypatch.setattr(webui.oom_recovery, "classify", lambda *a, **k: "gpu")
    monkeypatch.setattr(webui.oom_recovery, "read_tries", lambda d: [])
    monkeypatch.setattr(webui.oom_recovery, "plan", lambda *a, **k: {"do": "restart", "config": {}, "change": "",
                                                                       "why": "same settings", "save": None})
    monkeypatch.setattr(webui.oom_recovery, "add_try", lambda *a, **k: tries.append(a) or [1])
    monkeypatch.setattr(webui.oom_recovery, "keep_logs", lambda *a, **k: [])
    return restarts, tries, calls


def test_oom_recovery_waits_for_a_live_test_and_leaves_the_run_stopped(monkeypatch, tmp_path, capsys):
    restarts, tries, calls = _oom_setup(monkeypatch, tmp_path, live=None)
    webui.queue_tick()  # control: with no test the run is started again
    assert len(restarts) == 1 and len(tries) == 1 and calls == [False]
    restarts, tries, calls = _oom_setup(monkeypatch, tmp_path, live="t1")
    webui.queue_tick()
    assert restarts == [] and tries == []  # no start, and no recovery try used up
    assert calls == [True]  # the queue holds too
    out = capsys.readouterr().out
    assert "not restarted because LoRA test t1 is running" in out and "left stopped" in out


def test_held_gpus_include_split_plan_text_encoder_and_body_scoring():
    base = {"device": "cuda:0"}
    held = webui.lt_job_gpus({**base, "split_plan": "cuda:0=0-25;cuda:1=26-41"})
    assert set(held) == {"cuda:0", "cuda:1"} and "split plan" in held["cuda:1"]
    assert set(webui.lt_job_gpus({**base, "text_encoder_device": "cuda:1"})) == {"cuda:0", "cuda:1"}
    assert set(webui.lt_job_gpus({**base, "body_device": "cuda:2"})) == {"cuda:0", "cuda:2"}
    assert set(webui.lt_job_gpus({**base, "text_encoder_device": "auto", "body_device": "cpu"})) == {"cuda:0"}
    assert set(webui.lt_job_gpus({"device": "cuda:1", "body_device": "cuda"})) == {"cuda:1", "cuda:0"}  # bare cuda = 0
    assert set(webui.lt_held_gpus(True, {**base, "text_encoder_device": "cuda:1"}, ["cuda:0", "cuda:1"])) == \
        {"cuda:0", "cuda:1"}


def test_validation_is_cached_by_the_built_spec_not_its_name():
    spec = {"name": "a", "prompts": ["p"], "seeds": [1], "contestants": [{"id": "c1"}]}
    key = webui.lt_spec_key(spec)
    assert webui.lt_spec_key({**spec, "name": "another name"}) == key
    assert webui.lt_spec_key({**spec, "prompts": ["q"]}) != key
    assert webui.lt_spec_key({**spec, "seeds": [2]}) != key
    # the tab validates off the event loop and not again on the timer
    src = Path(webui.__file__).read_text(encoding="utf-8")
    assert "await run.io_bound(lt_validate, spec)" in src
    assert "await run.io_bound(lambda: lt_start(" in src
    body = src[src.index("    def current() -> tuple"):src.index("    async def validate_async")]
    assert "lt_validate(" not in body and "st[\"val\"]" in body


def test_remote_follow_does_not_depend_on_the_open_tab():
    src = Path(webui.__file__).read_text(encoding="utf-8")
    body = src[src.index("    async def remote_tick"):src.index("    def refresh_saved")]
    assert "is_visible" not in body and "lt_pull_once" in body


def test_remote_start_refuses_a_download_folder_that_is_a_link_before_and_after_the_start(tmp_path, monkeypatch):
    pulls = _fleet(tmp_path, monkeypatch)
    picks = [{"path": "output/j/j.safetensors", "config": "output/j/config.yaml", "spec_lora": "output/j/j.safetensors",
              "spec_config": "output/j/config.yaml", "run_id": None, "label": "j"}]
    spec = webui.lt_build_spec("r", picks, [{"text": "p", "kind": "both"}], [1], [1.0], [], ["cuda:0"])
    started = FakeDone(out=json.dumps({"test_id": "t20261122-140000-beef", "state": "queued"}))
    real = Path.is_symlink
    # the instance folder is a link: refused before the instance is asked to start anything
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == pulls / "42" or real(self))
    remote = FakeRemote({"test-start": started})
    tid, msg = webui.lt_remote_start(remote, 42, spec, pulls)
    assert tid is None and "cannot be used" in msg and remote.calls == []
    # the test's own folder is a link: said so, and nothing is written there
    dest = pulls / "42" / "_tests" / "t20261122-140000-beef"
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == dest or real(self))
    tid, msg = webui.lt_remote_start(FakeRemote({"test-start": started}), 42, spec, pulls)
    assert tid is None and "folder here could not be used" in msg and "link" in msg
    assert not dest.exists()


def test_tests_root_refuses_an_instance_or_tests_folder_that_is_a_link(tmp_path, monkeypatch):
    pulls = _fleet(tmp_path, monkeypatch)
    (pulls / "42" / "_tests").mkdir(parents=True)
    assert webui.lt_tests_root("42") == pulls / "42" / "_tests"
    real = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == pulls / "42" or real(self))
    assert webui.lt_tests_root("42") is None
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == pulls / "42" / "_tests" or real(self))
    assert webui.lt_tests_root("42") is None


@pytest.mark.parametrize("name", ["remote.json", "./remote.json", "sub/../remote.json", "x/../../remote.json"])
def test_pull_cannot_overwrite_the_notes_this_pc_keeps(tmp_path, monkeypatch, name):
    pulls = _fleet(tmp_path, monkeypatch)
    tid = "t20261122-140000-beef"
    dest = pulls / "42" / "_tests" / tid
    dest.mkdir(parents=True)
    webui.lt_remote_mark(dest, 42, tid, run_ids={"c1": "mine"})
    rr = PullRR()
    listing = {"base": "/b", "files": [{"path": name, "size": 1}]}
    with pytest.raises(OSError, match="Unsafe"):
        webui.lt_remote_pull(FakeRemote({"files": listing}), dest, tid, rr=rr)
    assert rr.fetched == []
    assert webui.lt_read_json(dest / "remote.json")["run_ids"] == {"c1": "mine"}


def test_marking_a_test_pulled_keeps_the_run_ids(tmp_path):
    webui.lt_remote_mark(tmp_path, 42, "t1", run_ids={"c1": "vast-a", "c2": None})
    webui.lt_remote_mark(tmp_path, 42, "t1", True)
    assert webui.lt_read_json(tmp_path / "remote.json") == {"instance": "42", "test_id": "t1", "pulled": True,
                                                            "run_ids": {"c1": "vast-a", "c2": None}}


def test_two_browsers_never_pull_the_same_test_at_once(tmp_path, monkeypatch):
    import threading

    pulls = _fleet(tmp_path, monkeypatch)
    tid = "t20261122-140000-beef"
    dest = pulls / "42" / "_tests" / tid
    dest.mkdir(parents=True)
    webui.lt_remote_mark(dest, 42, tid, run_ids={"c1": "mine"})
    gate, inside = threading.Event(), threading.Event()
    fetches = []

    class SlowRR(PullRR):
        def fetch(self, r, base, paths, d):
            fetches.append(paths)
            inside.set()
            assert gate.wait(10)

    rr = SlowRR()
    listing = {"base": "/b", "files": [{"path": "results.json", "size": 1}]}
    results = []
    first = threading.Thread(target=lambda: results.append(
        webui.lt_pull_once(FakeRemote({"files": listing}), dest, 42, tid, rr=rr)))
    first.start()
    assert inside.wait(10)
    # the second browser's pull of the same test comes while the first is fetching: it does nothing
    assert webui.lt_pull_once(FakeRemote({"files": listing}), dest, 42, tid, rr=rr) is None
    gate.set()
    first.join(10)
    assert results == [1] and len(fetches) == 1
    assert webui.lt_read_json(dest / "remote.json")["pulled"] is True
    # and once it is pulled, a later look does not fetch again
    assert webui.lt_pull_once(FakeRemote({"files": listing}), dest, 42, tid, rr=rr) is None and len(fetches) == 1
    assert not webui._LT_PULLS  # the claim is released


def test_remote_picks_take_the_run_id_of_a_matching_vast_record():
    src = Path(webui.__file__).read_text(encoding="utf-8")
    assert 'st["remote_run_ids"].get(job)' in src  # looked up from the vast history records of that instance
    assert 'src.get("kind") == "vast" and str(src.get("instance")) == str(iid)' in src
