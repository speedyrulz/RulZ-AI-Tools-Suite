"""LoRA test run (rulztrainer/loratest.py, docs/lora-test-contract.md sections 1 to 4): spec validation and refusals,
soups, the render loop over contestants x strengths x prompts x seeds, STOP, status.json, scoring and results.json.
The renderer and the scorer are fakes (a runner that writes small pictures, a scorer that reads its numbers from a
table): no model is loaded and no GPU is used."""
import io
import os
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from rulztrainer import loratest as T
from rulztrainer.lora import LoRALinear, save_lora

OUT, IN, RANK = 24, 20, 4
NAMES = ("layers.0.attn.q", "layers.0.attn.k")


# ---------------------------------------------------------------- fixtures


def lora_file(path: Path, seed=0, names=NAMES, rank=RANK, alpha=2.0):
    g = torch.Generator().manual_seed(seed)
    stubs = {}
    for name in names:
        parts = {"lora_down": torch.randn(rank, IN, generator=g), "lora_up": torch.randn(OUT, rank, generator=g)}
        stubs[name] = SimpleNamespace(kind="lora", alpha=float(alpha), rank=rank, factor=4, trainable=True,
                                      adapter_params=lambda parts=parts: parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_lora(stubs, path, dtype=torch.float32)


def write_run(base: Path, name: str, **cfg) -> dict:
    """A finished run's folder: config.yaml and a final and a best LoRA; returns the spec's paths for it."""
    folder = base / "output" / name
    data = {"model_path": "some/base", "network_type": "lora", "lora_rank": RANK, "sample_width": 64,
            "sample_height": 48, "sample_steps": 5, "sample_cfg": 3.5, "sample_negative": "ugly",
            "face_refs": ["refs/face.png"], "body_refs": ["refs/body.png"], "face_model": "buffalo_l",
            "body_score": "appearance", "face_gold": "auto"}
    data.update(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    lora_file(folder / f"{name}.safetensors", seed=1)
    lora_file(folder / f"{name}_best.safetensors", seed=2)
    return {"final": f"output/{name}/{name}.safetensors", "best": f"output/{name}/{name}_best.safetensors",
            "config": f"output/{name}/config.yaml"}


def make_spec(base: Path, **over) -> dict:
    run = write_run(base, "elena")
    spec = {"version": 1, "test_id": "t20261122-153012-a3f9", "name": "Elena: best vs final",
            "created": "2026-11-22T15:30:12Z",
            "contestants": [{"id": "c1", "label": "best", "lora": run["best"], "config": run["config"], "run_id": None},
                            {"id": "c2", "label": "final", "lora": run["final"], "config": run["config"],
                             "run_id": None}],
            "soups": [], "prompts": ["a face", "a body"], "prompt_kinds": ["face", "body"], "seeds": [1, 2],
            "strengths": [0.8, 1.0, 1.2], "render": {"width": None, "height": None, "steps": None, "cfg": None,
                                                     "negative": None},
            "scoring": {"face_refs": None, "body_refs": None, "from_contestant": "c1"}, "devices": ["cuda:0"]}
    spec.update(over)
    return spec


def make_test(base: Path, spec: dict) -> Path:
    d = base / "output" / "_tests" / spec["test_id"]
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return d


class FakeRunner:
    """Writes a small picture per job; remembers what it was asked."""

    def __init__(self, on_render=None):
        self.gpus = ["Fake GPU"]
        self.used, self.rendered, self.closed, self.on_render = [], [], False, on_render

    def use(self, path, cfg):
        self.used.append(Path(path))

    def render(self, jobs, strength):
        if self.on_render:
            self.on_render(self, jobs, strength)
        for j in jobs:
            Image.new("RGB", (8, 8), (10, 20, 30)).save(j.out_path)
            self.rendered.append((self.used[-1].name, strength, j.seed, j.prompt))
        return [str(j.out_path) for j in jobs]

    def close(self):
        self.closed = True


class FakeScorer:
    """combined score from a table by (contestant id, strength); face / body follow the picture's kind."""

    def __init__(self, table, scale="run", note=None):
        self.table, self.scale, self.scale_note = table, scale, note
        self.calls = []

    def score(self, path, kind):
        m = re.search(r"images[\\/](\w+)[\\/]s(\d\.\d\d)_", str(path))
        v = self.table[(m.group(1), m.group(2))]
        self.calls.append((m.group(1), m.group(2), kind))
        return {"face": None if kind == "body" else v + 2, "body": None if kind == "face" else v - 2,
                "gold": None if kind == "body" else v + 1, "combined": v}

    def close(self):
        pass


def hooks(runner=None, scorer=None, **kw):
    made = {}

    def mk_runner(cfg, devices, test_dir):
        made["cfg"], made["devices"] = cfg, devices
        return runner or FakeRunner()

    h = T.Hooks(make_runner=mk_runner, make_job=lambda **f: SimpleNamespace(**f),
                prepare_prompts=lambda cfg, prompts, device: made.setdefault("prompts", list(prompts)),
                make_scorer=lambda plan, scale, note, fr, br: scorer, release_gpu=lambda: None, status_every=0.05, **kw)
    h.made = made
    return h


def status_of(d: Path) -> dict:
    """status.json; the writer replaces it atomically, but on Windows a read can meet the swap: try again."""
    for attempt in range(20):
        try:
            return json.loads((d / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if attempt == 19:
                raise
            time.sleep(0.02)


def results_of(d: Path) -> dict:
    return json.loads((d / "results.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- ids, paths, the spec


def test_test_id_format():
    for _ in range(20):
        assert re.fullmatch(r"t\d{8}-\d{6}-[0-9a-f]{4}", T.new_test_id())


@pytest.mark.parametrize("tid", ["T2026", "t2026 1", "../x", "", "t2026_1", "A-b"])
def test_a_bad_test_id_is_refused(tmp_path, tid):
    with pytest.raises(T.SpecError, match="test_id"):
        T.validate(make_spec(tmp_path, test_id=tid), tmp_path)


def test_a_good_spec_validates(tmp_path):
    plan = T.validate(make_spec(tmp_path), tmp_path)
    assert plan.total == 2 * 3 * 2 * 2 and plan.from_id == "c1" and plan.devices == ["cuda:0"]
    assert [c["id"] for c in plan.contestants] == ["c1", "c2"]


def test_relative_paths_must_not_climb_out(tmp_path):
    spec = make_spec(tmp_path)
    spec["contestants"][0]["lora"] = "../outside.safetensors"
    with pytest.raises(T.SpecError, match="climbs out"):
        T.validate(spec, tmp_path)
    spec["contestants"][0]["lora"] = str(tmp_path / spec["contestants"][1]["lora"])  # an absolute path is fine
    T.validate(spec, tmp_path)


@pytest.mark.parametrize("change,match", [
    (lambda s: s.update(prompts=[]), "prompts"),
    (lambda s: s.update(seeds=[]), "seeds"),
    (lambda s: s.update(strengths=[]), "strengths"),
    (lambda s: s.update(strengths=[1.0, 2.5]), "outside"),
    (lambda s: s.update(strengths=[-0.1]), "outside"),
    (lambda s: s.update(strengths=[1.0, 1.001]), "same at two decimals"),
    (lambda s: s.update(seeds=[1, 1]), "twice"),
    (lambda s: s.update(prompt_kinds=["face"]), "prompt_kinds"),
    (lambda s: s.update(prompt_kinds=["face", "feet"]), "prompt_kinds"),
    (lambda s: s.update(devices=["gpu0"]), "devices"),
    (lambda s: s.update(contestants=[]), "contestants"),
    (lambda s: s["contestants"][1].update(id="c1"), "twice"),
    (lambda s: s["scoring"].update(from_contestant="c9"), "from_contestant"),
    (lambda s: s.update(soups=[{"id": "s1", "of": ["c1"]}]), "two or more"),
    (lambda s: s.update(soups=[{"id": "s1", "of": ["c1", "s0"]}]), "not a file contestant"),
    (lambda s: s.update(soups=[{"id": "s1", "of": ["c1", "c2"], "weights": [1]}]), "weights"),
])
def test_each_spec_refusal(tmp_path, change, match):
    spec = make_spec(tmp_path)
    change(spec)
    with pytest.raises(T.SpecError, match=match):
        T.validate(spec, tmp_path)


def test_a_missing_lora_or_config_is_named(tmp_path):
    spec = make_spec(tmp_path)
    spec["contestants"][1]["lora"] = "output/elena/gone.safetensors"
    with pytest.raises(T.SpecError, match=r"gone\.safetensors"):
        T.validate(spec, tmp_path)
    spec = make_spec(tmp_path)
    spec["contestants"][0]["config"] = "output/elena/nope.yaml"
    with pytest.raises(T.SpecError, match=r"nope\.yaml"):
        T.validate(spec, tmp_path)


def test_more_than_2000_pictures_is_refused(tmp_path):
    spec = make_spec(tmp_path, seeds=list(range(1, 200)))  # 2 x 3 x 2 x 199 = 2388
    with pytest.raises(T.SpecError, match="2388"):
        T.validate(spec, tmp_path)
    T.validate(make_spec(tmp_path, seeds=list(range(1, 167))), tmp_path)  # 1992: fine


def test_contestants_of_different_base_models_are_refused(tmp_path):
    spec = make_spec(tmp_path)
    other = write_run(tmp_path, "other", model_path="another/base")
    spec["contestants"][1].update(lora=other["final"], config=other["config"])
    with pytest.raises(T.SpecError, match="different base models.*model_path"):
        T.validate(spec, tmp_path)
    other = write_run(tmp_path, "other", network_type="lokr")
    spec["contestants"][1].update(lora=other["final"], config=other["config"])
    with pytest.raises(T.SpecError, match="network_type"):
        T.validate(spec, tmp_path)


@pytest.mark.parametrize("key,value", [("sample_model_path", "x.safetensors"), ("sample_model_quant", "int8"),
                                        ("transformer_quant", "int8"), ("model_source", "hf"),
                                        ("model_arch", "krea2"), ("lora_rank", 8)])
def test_every_base_model_setting_counts(tmp_path, key, value):
    spec = make_spec(tmp_path)
    other = write_run(tmp_path, "other", **{key: value})
    spec["contestants"][1].update(lora=other["final"], config=other["config"])
    with pytest.raises(T.SpecError, match=key):
        T.validate(spec, tmp_path)


def test_a_soup_of_different_base_models_is_refused(tmp_path):
    spec = make_spec(tmp_path, soups=[{"id": "s1", "label": "s", "of": ["c1", "c2"], "weights": None}])
    other = write_run(tmp_path, "other", model_path="another/base")
    spec["contestants"][1].update(lora=other["final"], config=other["config"])
    with pytest.raises(T.SpecError, match=r"soups\[0\] \(s1\) mixes base models"):
        T.validate(spec, tmp_path)


def test_two_gpus_refuse_oft_and_dora(tmp_path):
    spec = make_spec(tmp_path, devices=["cuda:0", "cuda:1"])
    T.validate(spec, tmp_path)
    write_run(tmp_path, "elena", dora=True)
    with pytest.raises(T.SpecError, match="OFT and DoRA"):
        T.validate(spec, tmp_path)
    T.validate({**spec, "devices": ["cuda:0"]}, tmp_path)


def test_render_settings_default_to_the_first_contestants_samples(tmp_path):
    plan = T.validate(make_spec(tmp_path), tmp_path)
    assert T.render_settings(plan) == {"width": 64, "height": 48, "steps": 5, "cfg": 3.5, "negative": "ugly"}
    plan = T.validate(make_spec(tmp_path, render={"width": 32, "height": None, "steps": 2, "cfg": None,
                                                   "negative": "blurry"}), tmp_path)
    assert T.render_settings(plan) == {"width": 32, "height": 48, "steps": 2, "cfg": 3.5, "negative": "blurry"}


# ---------------------------------------------------------------- refusals through run, new, list


def test_a_refused_run_writes_status_and_renders_nothing(tmp_path):
    spec = make_spec(tmp_path, strengths=[3.0])
    d = make_test(tmp_path, spec)
    h = hooks()
    assert T.run(d, tmp_path, h) == 1
    st = status_of(d)
    assert st["state"] == "refused" and "strengths" in st["reason"] and not h.made
    assert not (d / "results.json").exists()


def test_a_missing_file_refuses_before_the_gpu(tmp_path):
    spec = make_spec(tmp_path)
    d = make_test(tmp_path, spec)
    (tmp_path / spec["contestants"][0]["lora"]).unlink()
    h = hooks()
    assert T.run(d, tmp_path, h) == 1
    assert "elena_best.safetensors" in status_of(d)["reason"] and not h.made


def test_the_folder_name_must_be_the_test_id(tmp_path):
    spec = make_spec(tmp_path)
    d = make_test(tmp_path, spec)
    moved = d.rename(d.with_name("t20261122-000000-0000"))
    assert T.run(moved, tmp_path, hooks()) == 1
    assert "folder" in status_of(moved)["reason"]


def test_an_unreadable_spec_is_refused(tmp_path):
    d = tmp_path / "output" / "_tests" / "tx"
    d.mkdir(parents=True)
    assert T.run(d, tmp_path, hooks()) == 1
    assert status_of(d)["state"] == "refused" and "spec.json" in status_of(d)["reason"]


def test_new_makes_the_folder_and_fills_the_id(tmp_path):
    spec = make_spec(tmp_path)
    del spec["test_id"], spec["created"]
    tid = T.cmd_new(json.dumps(spec), tmp_path)
    assert re.fullmatch(r"t\d{8}-\d{6}-[0-9a-f]{4}", tid)
    d = tmp_path / "output" / "_tests" / tid
    saved = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    assert saved["test_id"] == tid and saved["created"] and saved["name"] == spec["name"]
    assert status_of(d)["state"] == "queued"
    with pytest.raises(T.SpecError, match="already exists"):
        T.cmd_new(json.dumps({**spec, "test_id": tid}), tmp_path)


def test_new_refuses_a_bad_spec_and_makes_nothing(tmp_path):
    spec = make_spec(tmp_path, seeds=[])
    with pytest.raises(T.SpecError, match="seeds"):
        T.cmd_new(json.dumps(spec), tmp_path)
    assert not (tmp_path / "output" / "_tests").exists()
    with pytest.raises(T.SpecError, match="not valid JSON"):
        T.cmd_new("{nope", tmp_path)


def test_unknown_spec_keys_are_kept_and_ignored(tmp_path):
    spec = make_spec(tmp_path, future_thing={"a": 1})
    tid = T.cmd_new(json.dumps(spec), tmp_path)
    saved = json.loads((tmp_path / "output" / "_tests" / tid / "spec.json").read_text(encoding="utf-8"))
    assert saved["future_thing"] == {"a": 1}


def test_list_shows_one_entry_per_test_newest_first(tmp_path, capsys):
    a = make_spec(tmp_path, test_id="t20261122-100000-aaaa", name="old")
    b = make_spec(tmp_path, test_id="t20261122-110000-bbbb", name="new")
    da, db = make_test(tmp_path, a), make_test(tmp_path, b)
    st = T.StatusWriter(da, a["test_id"])
    st.update(state="done")
    qw = T.StatusWriter(db, b["test_id"])
    qw.data["pid"] = None
    qw.update(state="queued")
    got = T.list_tests(tmp_path)
    assert [(t["id"], t["name"], t["state"], t["created"]) for t in got] == [
        ("t20261122-110000-bbbb", "new", "queued", "2026-11-22T15:30:12Z"),
        ("t20261122-100000-aaaa", "old", "done", "2026-11-22T15:30:12Z")]
    assert T.list_tests(tmp_path / "empty") == []


def test_a_dead_pid_in_a_running_state_reads_as_failed(tmp_path, monkeypatch):
    spec = make_spec(tmp_path)
    d = make_test(tmp_path, spec)
    T.StatusWriter(d, spec["test_id"]).update(state="rendering", pid=4242)
    monkeypatch.setattr(T, "pid_alive", lambda pid, created=None: False)
    assert T.list_tests(tmp_path)[0]["state"] == "failed"
    monkeypatch.setattr(T, "pid_alive", lambda pid, created=None: True)
    assert T.list_tests(tmp_path)[0]["state"] == "rendering"


# ---------------------------------------------------------------- a full run


# scores by (contestant, strength): c1 peaks twice (1.00 and 1.20 tie: the lower strength wins), c2 ties c1's best
TABLE = {("c1", "0.80"): 50.0, ("c1", "1.00"): 60.0, ("c1", "1.20"): 60.0,
         ("c2", "0.80"): 40.0, ("c2", "1.00"): 60.0, ("c2", "1.20"): 55.0}


def test_full_run_layout_results_and_tie_rules(tmp_path):
    spec = make_spec(tmp_path)
    d = make_test(tmp_path, spec)
    runner, scorer = FakeRunner(), FakeScorer(TABLE)
    h = hooks(runner, scorer)
    assert T.run(d, tmp_path, h) == 0
    st = status_of(d)
    assert st["state"] == "done" and st["done"] == st["total"] == 24 and st["eta_s"] is None and st["error"] is None
    # the pictures: images/<contestant>/s<strength:.2f>_p<prompt>_seed<seed>.png
    for cid in ("c1", "c2"):
        for s in ("0.80", "1.00", "1.20"):
            for p in (0, 1):
                for seed in (1, 2):
                    assert (d / "images" / cid / f"s{s}_p{p}_seed{seed}.png").is_file()
    assert len(list((d / "images").rglob("*.png"))) == 24
    # the renderer was asked for each contestant's file, the first config, the prompts cached first
    assert [p.name for p in runner.used] == ["elena_best.safetensors", "elena.safetensors"] and runner.closed
    assert h.made["prompts"] == ["a face", "a body"] and h.made["devices"] == ["cuda:0"]
    assert h.made["cfg"]["sample_negative"] == "ugly"

    res = results_of(d)
    assert res["version"] == 1 and res["test_id"] == spec["test_id"] and res["complete"] is True
    assert [c["id"] for c in res["contestants"]] == ["c1", "c2"]
    assert res["contestants"][0]["kind"] == "file" and len(res["contestants"][0]["sha256"]) == 64
    assert res["contestants"][0]["lora"] == spec["contestants"][0]["lora"]
    assert len(res["cells"]) == 24
    cell = next(c for c in res["cells"] if (c["contestant"], c["strength"], c["prompt"], c["seed"]) == ("c1", 1.0, 0, 2))
    assert cell == {"contestant": "c1", "strength": 1.0, "prompt": 0, "seed": 2,
                    "image": "images/c1/s1.00_p0_seed2.png", "face": 62.0, "body": None, "gold": 61.0, "combined": 60.0}
    body = next(c for c in res["cells"] if c["prompt"] == 1)  # a body prompt: no face score
    assert body["face"] is None and body["gold"] is None and body["body"] is not None
    assert len(res["columns"]) == 6
    col = next(c for c in res["columns"] if (c["contestant"], c["strength"]) == ("c1", 1.0))
    assert set(col) == {"contestant", "strength", "face", "body", "gold", "combined", "n"}
    assert col == {"contestant": "c1", "strength": 1.0, "face": 62.0, "body": 58.0, "gold": 61.0, "combined": 60.0,
                   "n": 4}
    # ties: c1's 1.00 and 1.20 are both 60 -> the lower strength; c1 and c2 both 60 at 1.00 -> the earlier contestant
    assert res["best_strength"] == {"c1": 1.0, "c2": 1.0}
    assert res["winner"] == {"contestant": "c1", "strength": 1.0, "combined": 60.0}
    assert res["render"] == {"width": 64, "height": 48, "steps": 5, "cfg": 3.5, "seconds": res["render"]["seconds"],
                             "gpus": ["Fake GPU"]}
    # the scorer saw each picture's prompt kind
    assert {k for _, _, k in scorer.calls} == {"face", "body"} and len(scorer.calls) == 24


def test_the_jobs_carry_the_resolved_settings(tmp_path):
    spec = make_spec(tmp_path, render={"width": 32, "height": None, "steps": None, "cfg": 1.0, "negative": "blurry"},
                     strengths=[1.0], seeds=[7], prompts=["only"], prompt_kinds=["both"])
    d = make_test(tmp_path, spec)
    seen = []
    runner = FakeRunner(on_render=lambda r, jobs, s: seen.extend((j, s) for j in jobs))
    h = hooks(runner, FakeScorer({("c1", "1.00"): 1.0, ("c2", "1.00"): 2.0}))
    assert T.run(d, tmp_path, h) == 0
    job, strength = seen[0]
    assert (job.width, job.height, job.steps, job.cfg_scale, job.seed, job.prompt) == (32, 48, 5, 1.0, 7, "only")
    assert strength == 1.0 and h.made["cfg"]["sample_negative"] == "blurry"
    assert Path(job.out_path) == d / "images" / "c1" / "s1.00_p0_seed7.png"


def test_a_tie_between_contestants_goes_to_the_earlier_one_and_lower_strength_wins_first():
    cols = [{"contestant": "c2", "strength": 0.8, "combined": 70.0}, {"contestant": "c1", "strength": 1.0, "combined": 70.0},
            {"contestant": "c1", "strength": 0.6, "combined": None}]
    best, winner = T.pick_best(cols, ["c1", "c2"])
    assert best == {"c1": 1.0, "c2": 0.8} and winner["contestant"] == "c2"  # 0.8 < 1.0: strength before contestant
    cols[1]["strength"] = 0.8
    assert T.pick_best(cols, ["c1", "c2"])[1]["contestant"] == "c1"
    assert T.pick_best([{"contestant": "c1", "strength": 1.0, "combined": None}], ["c1"]) == ({"c1": None}, None)


def test_no_references_still_renders_with_null_scores(tmp_path):
    spec = make_spec(tmp_path)
    write_run(tmp_path, "elena", face_refs=[], body_refs=[])
    d = make_test(tmp_path, spec)
    h = hooks(FakeRunner(), scorer=None)
    assert T.run(d, tmp_path, h) == 0
    res = results_of(d)
    assert all(c["combined"] is None and c["face"] is None for c in res["cells"])
    assert res["winner"] is None and res["best_strength"] == {"c1": None, "c2": None}
    assert "No references" in res["scale_note"] and len(list((d / "images").rglob("*.png"))) == 24


def test_references_in_the_spec_replace_the_runs(tmp_path):
    spec = make_spec(tmp_path)
    spec["scoring"].update(face_refs=["extra/a.png"], body_refs=[])
    plan = T.validate(spec, tmp_path)
    assert T.resolve_refs(plan) == ([str(tmp_path / "extra" / "a.png")], [])
    assert T.resolve_refs(T.validate(make_spec(tmp_path), tmp_path)) == (
        [str(tmp_path / "refs" / "face.png")], [str(tmp_path / "refs" / "body.png")])


# ---------------------------------------------------------------- STOP, status, progress


def test_stop_finishes_the_picture_in_hand_and_marks_the_results_incomplete(tmp_path):
    spec = make_spec(tmp_path)
    d = make_test(tmp_path, spec)

    def stop_after_five(r, jobs, strength):
        if len(r.rendered) == 4:  # while the fifth picture is in hand
            (d / "STOP").write_text("", encoding="utf-8")

    runner = FakeRunner(on_render=stop_after_five)
    scorer = FakeScorer(TABLE)
    assert T.run(d, tmp_path, hooks(runner, scorer)) == 0
    assert status_of(d)["state"] == "stopped" and status_of(d)["done"] == 5
    res = results_of(d)
    assert res["complete"] is False and len(res["cells"]) == 5 and len(scorer.calls) == 5
    assert len(list((d / "images").rglob("*.png"))) == 5 and runner.closed
    assert res["winner"] is not None  # scored what it has


def test_stop_before_the_first_picture(tmp_path):
    spec = make_spec(tmp_path)
    d = make_test(tmp_path, spec)
    (d / "STOP").write_text("", encoding="utf-8")
    assert T.run(d, tmp_path, hooks(FakeRunner(), FakeScorer(TABLE))) == 0
    res = results_of(d)
    assert status_of(d)["state"] == "stopped" and res["complete"] is False and res["cells"] == []
    assert res["winner"] is None


def test_a_crash_ends_failed_with_the_traceback_tail(tmp_path):
    spec = make_spec(tmp_path)
    d = make_test(tmp_path, spec)

    def boom(r, jobs, strength):
        raise RuntimeError("CUDA out of memory (fake)")

    runner = FakeRunner(on_render=boom)
    assert T.run(d, tmp_path, hooks(runner, FakeScorer(TABLE))) == 1
    st = status_of(d)
    assert st["state"] == "failed" and "out of memory (fake)" in st["error"] and runner.closed
    assert not (d / "results.json").exists()


def test_status_has_progress_pid_and_eta(tmp_path):
    spec = make_spec(tmp_path)
    d = make_test(tmp_path, spec)
    seen = []

    def watch(r, jobs, strength):
        seen.append(status_of(d))
        time.sleep(0.01)

    assert T.run(d, tmp_path, hooks(FakeRunner(on_render=watch), FakeScorer(TABLE))) == 0
    first, third = seen[0], seen[2]
    assert first["state"] == "rendering" and first["total"] == 24 and first["done"] == 0 and first["eta_s"] is None
    assert third["done"] == 2 and third["current"] == "c1 at 0.80, prompt 0, seed 2"
    assert third["eta_s"] is not None and third["pid"] > 0 and third["error"] is None and third["reason"] is None
    assert set(first) - {"pid_created"} == {"state", "test_id", "pid", "started", "updated", "done", "total", "current",
                                            "eta_s", "reason", "error"}  # (pid_created is this process's start time)


def test_eta_is_the_mean_seconds_per_picture_times_the_pictures_left(tmp_path):
    w = T.StatusWriter(tmp_path, "tx")
    w.update(state="rendering", total=10)
    w.picture_done(2.0, "a")
    w.picture_done(4.0, "b")
    st = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert st["done"] == 2 and st["eta_s"] == 24 and st["current"] == "b"  # 8 left x 3 s


def test_status_is_rewritten_while_one_picture_takes_long(tmp_path, monkeypatch):
    spec = make_spec(tmp_path, strengths=[1.0], seeds=[1], prompts=["p"], prompt_kinds=["both"])
    d = make_test(tmp_path, spec)
    writes = []
    real = T.write_json
    monkeypatch.setattr(T, "write_json", lambda path, data: (writes.append(Path(path).name), real(path, data)))
    runner = FakeRunner(on_render=lambda r, jobs, s: time.sleep(0.5))  # heartbeat every 0.05 s in these tests
    assert T.run(d, tmp_path, hooks(runner, FakeScorer({("c1", "1.00"): 1.0, ("c2", "1.00"): 1.0}))) == 0
    during = writes.count("status.json")
    assert during >= 8  # one picture of 0.5 s alone brought several rewrites


# ---------------------------------------------------------------- soups


def test_a_soup_is_cooked_saved_and_rendered_as_a_contestant(tmp_path):
    from safetensors.torch import load_file

    spec = make_spec(tmp_path, strengths=[1.0], seeds=[1], prompts=["p"], prompt_kinds=["both"],
                     soups=[{"id": "s1", "label": "soup c1+c2", "of": ["c1", "c2"], "weights": None}])
    d = make_test(tmp_path, spec)
    table = {("c1", "1.00"): 50.0, ("c2", "1.00"): 55.0, ("s1", "1.00"): 70.0}
    runner = FakeRunner()
    assert T.run(d, tmp_path, hooks(runner, FakeScorer(table))) == 0
    soup_file = d / "soups" / "s1.safetensors"
    assert soup_file.is_file() and any(k.endswith("lora_down.weight") for k in load_file(str(soup_file)))
    assert [p.name for p in runner.used][-1] == "s1.safetensors" and runner.used[-1] == soup_file
    assert (d / "images" / "s1" / "s1.00_p0_seed1.png").is_file()
    res = results_of(d)
    s = res["contestants"][2]
    assert s["id"] == "s1" and s["kind"] == "soup" and s["lora"] == "soups/s1.safetensors"
    assert s["of"] == ["c1", "c2"] and s["weights"] == [0.5, 0.5] and len(s["sha256"]) == 64
    assert res["winner"]["contestant"] == "s1" and res["best_strength"]["s1"] == 1.0


def test_soup_weights_are_normalized(tmp_path):
    spec = make_spec(tmp_path, soups=[{"id": "s1", "of": ["c1", "c2"], "weights": [3, 1]}])
    plan = T.validate(spec, tmp_path)
    assert plan.soups[0]["weights"] == [0.75, 0.25] and plan.soups[0]["out"] == "soups/s1.safetensors"
    assert plan.total == 3 * 3 * 2 * 2  # the soup is a contestant of the grid


# ---------------------------------------------------------------- the scale


LIKENESS = json.dumps({"scale": {"face": [20.0, 70.0], "body": [0.0, 100.0], "curve": {"face": 1.5}},
                       "face_model": "buffalo_l", "body_score": "appearance", "gold_name": "adaface"})


def run_with_scale(tmp_path, spec, likeness=True, second_cfg=None):
    if second_cfg is not None:
        other = write_run(tmp_path, "other", **second_cfg)
        spec["contestants"][1].update(lora=other["final"], config=other["config"])
    if likeness:
        (tmp_path / "output" / "elena" / "likeness.jsonl").write_text(LIKENESS + "\n", encoding="utf-8")
    plan = T.validate(spec, tmp_path)
    return T.decide_scale(plan)


def test_the_scale_is_the_runs_when_everything_matches(tmp_path):
    assert run_with_scale(tmp_path, make_spec(tmp_path)) == ("run", None)


@pytest.mark.parametrize("key,value", [("face_model", "antelopev2"), ("body_score", "anchored"), ("face_gold", "facenet")])
def test_runs_that_differ_in_the_scale_settings_get_the_references_scale(tmp_path, key, value):
    scale, note = run_with_scale(tmp_path, make_spec(tmp_path), second_cfg={key: value})
    assert scale == "references" and key in note and "references' own scale" in note


def test_references_given_in_the_test_or_no_likeness_file_mean_the_references_scale(tmp_path):
    spec = make_spec(tmp_path)
    spec["scoring"]["face_refs"] = ["x.png"]
    scale, note = run_with_scale(tmp_path, spec)
    assert scale == "references" and "given in the test" in note
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    scale, note = run_with_scale(fresh, make_spec(fresh), likeness=False)
    assert scale == "references" and "likeness.jsonl" in note


def test_results_carry_the_scale_and_its_note(tmp_path):
    spec = make_spec(tmp_path)
    other = write_run(tmp_path, "other", face_model="antelopev2")
    spec["contestants"][1].update(lora=other["final"], config=other["config"])
    d = make_test(tmp_path, spec)
    assert T.run(d, tmp_path, hooks(FakeRunner(), FakeScorer(TABLE, scale="references", note="x"))) == 0
    res = results_of(d)
    assert res["scale"] == "references" and res["scale_note"] == "x"  # what the scorer reports wins
    # a scorer without its own opinion: the decision made from the configs
    d2 = tmp_path / "output" / "_tests" / "t20261122-153012-a3fa"
    d2.mkdir(parents=True)
    (d2 / "spec.json").write_text(json.dumps({**spec, "test_id": d2.name}), encoding="utf-8")

    class NoOpinion:
        score = FakeScorer(TABLE).score

        def close(self):
            pass

    assert T.run(d2, tmp_path, hooks(FakeRunner(), NoOpinion())) == 0
    res = results_of(d2)
    assert res["scale"] == "references" and "face_model" in res["scale_note"]


# ---------------------------------------------------------------- the real glue (no model)


def test_payload_from_file_folds_the_strength_into_the_scale(tmp_path):
    from rulztrainer.lora import LoraSnapshot

    lora_file(tmp_path / "a.safetensors", seed=3)
    cfg = {"lora_rank": RANK}
    full = torch.load(io.BytesIO(T.payload_from_file(tmp_path / "a.safetensors", cfg, 1.0)),
                      weights_only=True)
    half = torch.load(io.BytesIO(T.payload_from_file(tmp_path / "a.safetensors", cfg, 0.5)),
                      weights_only=True)
    assert set(full) == set(NAMES)
    q = full[NAMES[0]]
    assert q["kind"] == "lora" and q["rank"] == RANK and q["scale"] == pytest.approx(2.0 / RANK)
    assert half[NAMES[0]]["scale"] == pytest.approx(q["scale"] * 0.5)
    assert q["params"]["lora_down"].shape == (RANK, IN) and q["params"]["lora_up"].shape == (OUT, RANK)
    snap = LoraSnapshot(q)  # what mirror_lora reads
    assert snap.kind == "lora" and snap.params is q["params"]


def tiny_slots():
    import torch.nn as nn

    return {n: LoRALinear(nn.Linear(IN, OUT), RANK, 2.0) for n in NAMES + ("layers.1.attn.q",)}


def test_local_runner_loads_a_file_blanks_stale_layers_and_sets_the_strength(tmp_path):
    slots = tiny_slots()
    seen = []

    class FakeRenderer:
        injected = slots

        def render(self, jobs, payload):
            seen.append(({n: m.multiplier for n, m in slots.items()}, torch.load(io.BytesIO(payload),
                                                                                 weights_only=True)))
            return [str(j.out_path) for j in jobs], 0.1

    r = T.LocalRunner.__new__(T.LocalRunner)
    r.renderer, r.gpus, r.payload = FakeRenderer(), ["x"], b""
    for m in slots.values():  # an earlier contestant's weights, also in a layer the next file does not have
        m.lora_up.weight.data.fill_(7.0)
    lora_file(tmp_path / "a.safetensors", seed=4)  # covers two of the three layers
    r.use(tmp_path / "a.safetensors", {})
    r.render([SimpleNamespace(out_path="p.png")], 1.2)
    mult, state = seen[0]
    assert torch.count_nonzero(state["layers.1.attn.q"]["params"]["lora_up"]) == 0  # not in the file: nothing of c1
    assert set(mult.values()) == {1.2}  # on the Renderer's own slots, not rewritten into the file
    assert torch.count_nonzero(state[NAMES[0]]["params"]["lora_up"]) > 0
    r.render([SimpleNamespace(out_path="p.png")], 1.0)
    assert set(seen[1][0].values()) == {1.0}  # 1.0 is the plain render


def test_local_runner_refuses_a_file_that_fits_no_layer(tmp_path):
    slots = tiny_slots()
    r = T.LocalRunner.__new__(T.LocalRunner)
    r.renderer, r.gpus, r.payload = SimpleNamespace(injected=slots), ["x"], b""
    lora_file(tmp_path / "wrong.safetensors", names=("other.layer",))
    with pytest.raises(ValueError, match="none of its layers"):
        r.use(tmp_path / "wrong.safetensors", {})


def test_two_gpus_render_a_column_per_round(tmp_path):
    spec = make_spec(tmp_path, devices=["cuda:0", "cuda:1"])
    d = make_test(tmp_path, spec)
    rounds = []
    runner = FakeRunner(on_render=lambda r, jobs, s: rounds.append((len(jobs), s)))
    h = hooks(runner, FakeScorer(TABLE))
    assert T.run(d, tmp_path, h) == 0
    assert rounds == [(4, s) for s in (0.8, 1.0, 1.2)] * 2  # prompts x seeds per column
    assert h.made["devices"] == ["cuda:0", "cuda:1"] and status_of(d)["done"] == 24


def test_the_cli_new_and_list(tmp_path, monkeypatch, capsys):
    spec = make_spec(tmp_path)
    f = tmp_path / "spec_in.json"
    f.write_text(json.dumps({k: v for k, v in spec.items() if k != "test_id"}), encoding="utf-8")
    monkeypatch.setattr(T, "INSTALL_DIR", tmp_path)
    assert T.main(["new", "--spec", str(f)]) == 0
    tid = capsys.readouterr().out.strip()
    assert re.fullmatch(r"t\d{8}-\d{6}-[0-9a-f]{4}", tid)
    assert T.main(["list"]) == 0
    line = json.loads(capsys.readouterr().out.strip())
    assert line["id"] == tid and line["name"] == spec["name"] and line["state"] == "queued"
    f.write_text(json.dumps({**spec, "seeds": []}), encoding="utf-8")
    assert T.main(["new", "--spec", str(f)]) == 1
    assert "seeds" in capsys.readouterr().err


def test_local_runner_loads_any_contestant_order_after_a_soup_resized_the_slots(tmp_path):
    from rulztrainer import soup as S
    from rulztrainer.lora import lora_snapshots, mirror_lora

    slots = tiny_slots()
    deltas = []

    class MirrorRenderer:
        """Applies the payload as Renderer.render does, then reads the slots' effective change."""
        injected = slots

        def render(self, jobs, payload):
            mirror_lora(lora_snapshots(torch.load(io.BytesIO(payload), weights_only=True)), slots)
            deltas.append({n: m.lora_up.weight @ m.lora_down.weight * m.scale for n, m in slots.items()})
            return [], 0.0

    r = T.LocalRunner.__new__(T.LocalRunner)
    r.renderer, r.gpus, r.payload = MirrorRenderer(), ["x"], b""
    lora_file(tmp_path / "a.safetensors", seed=5)
    lora_file(tmp_path / "b.safetensors", seed=6)
    S.make_soup([str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")], tmp_path / "s.safetensors", rank=0)
    cfg = {"network_type": "lora", "lora_rank": RANK}
    for name in ("a", "s", "a", "b"):  # rank 4 -> the soup's rank 8 -> rank 4 again
        r.use(tmp_path / f"{name}.safetensors", cfg)
        r.render([], 1.0)
        want = {n: S.delta_of(e) for n, e in S.read_ingredient(tmp_path / f"{name}.safetensors").items()}
        got = deltas[-1]
        for n in NAMES:
            assert torch.allclose(got[n], want[n], rtol=1e-3, atol=1e-3), (name, n)
        assert torch.count_nonzero(got["layers.1.attn.q"]) == 0  # a layer in no file stays off
    assert slots[NAMES[0]].lora_down.weight.shape[0] == RANK  # resized back down after the soup


def test_clip_configs_are_refused(tmp_path):
    spec = make_spec(tmp_path)
    write_run(tmp_path, "elena", probe_frames=9)
    with pytest.raises(T.SpecError, match="probe_frames is 9"):
        T.validate(spec, tmp_path)



def kohya_file(path: Path, names, rank=RANK, alpha=2.0, seed=0):
    """A kohya (SDXL) LoRA over unet.* slots; returns each layer's effective change."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    tensors, want = {}, {}
    for n in names:  # unet.a.q -> lora_unet_a_q
        key = "lora_unet_" + n.removeprefix("unet.").replace(".", "_")
        down, up = torch.randn(rank, IN, generator=g), torch.randn(OUT, rank, generator=g)
        tensors[f"{key}.lora_down.weight"], tensors[f"{key}.lora_up.weight"] = down, up
        tensors[f"{key}.alpha"] = torch.tensor(float(alpha))
        want[n] = up @ down * (alpha / rank)
    save_file(tensors, str(path))
    return want


def test_a_kohya_file_loads_after_a_soup_resized_the_slots(tmp_path):
    import torch.nn as nn

    from rulztrainer import soup as S
    from rulztrainer.lora import lora_snapshots, mirror_lora

    names = ("unet.a.q", "unet.a.k")
    slots = {n: LoRALinear(nn.Linear(IN, OUT), RANK, 2.0) for n in names}
    deltas = []

    class MirrorRenderer:
        injected = slots

        def render(self, jobs, payload):
            mirror_lora(lora_snapshots(torch.load(io.BytesIO(payload), weights_only=True)), slots)
            deltas.append({n: m.lora_up.weight @ m.lora_down.weight * m.scale for n, m in slots.items()})
            return [], 0.0

    lora_file(tmp_path / "a.safetensors", seed=5, names=names)
    lora_file(tmp_path / "b.safetensors", seed=6, names=names)
    S.make_soup([str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")], tmp_path / "s.safetensors", rank=0)
    want_k = kohya_file(tmp_path / "k.safetensors", names, seed=7)
    r = T.LocalRunner.__new__(T.LocalRunner)
    r.renderer, r.gpus, r.payload = MirrorRenderer(), ["x"], b""
    cfg = {"network_type": "lora", "lora_rank": RANK}
    r.use(tmp_path / "s.safetensors", cfg)  # the soup: rank 8
    r.render([], 1.0)
    assert slots[names[0]].lora_down.weight.shape[0] == 2 * RANK
    r.use(tmp_path / "k.safetensors", cfg)  # then a kohya rank-4 file: slots restored to rank 4 first
    r.render([], 1.0)
    assert slots[names[0]].lora_down.weight.shape[0] == RANK
    for n in names:
        assert torch.allclose(deltas[-1][n], want_k[n], rtol=1e-3, atol=1e-3)
    r.use(tmp_path / "s.safetensors", cfg)  # and back to the soup
    r.render([], 1.0)
    want = {n: S.delta_of(e) for n, e in S.read_ingredient(tmp_path / "s.safetensors").items()}
    assert all(torch.allclose(deltas[-1][n], want[n], rtol=1e-3, atol=1e-3) for n in names)


def test_soups_of_kohya_files_and_pools_with_kohya_files_are_refused(tmp_path):
    spec = make_spec(tmp_path, soups=[{"id": "s1", "of": ["c1", "c2"], "weights": None}])
    kohya_file(tmp_path / spec["contestants"][0]["lora"], ["unet.a.q"])
    with pytest.raises(T.SpecError, match="soup s1.*kohya"):
        T.validate(spec, tmp_path)
    spec["soups"] = []
    T.validate(spec, tmp_path)  # a kohya contestant alone is fine on one device
    with pytest.raises(T.SpecError, match="kohya file.*one device"):
        T.validate({**spec, "devices": ["cuda:0", "cuda:1"]}, tmp_path)


# ---------------------------------------------------------------- review fixes


def test_two_gpus_refuse_contestants_with_different_layer_sets(tmp_path):
    spec = make_spec(tmp_path, devices=["cuda:0", "cuda:1"])
    lora_file(tmp_path / spec["contestants"][1]["lora"], seed=9, names=NAMES[:1])  # c2 lacks a layer c1 has
    d = make_test(tmp_path, spec)
    h = hooks(FakeRunner(), FakeScorer(TABLE))
    assert T.run(d, tmp_path, h) == 1
    st = status_of(d)
    assert st["state"] == "refused" and "same layers" in st["reason"] and "use one device" in st["reason"]
    assert "cfg" not in h.made  # refused before the Renderer or the prompt cache
    # one device completes the state from its own slots, so the same spec is fine there
    spec1 = {**spec, "test_id": "t20261122-153012-a3fb", "devices": ["cuda:0"]}
    d1 = make_test(tmp_path, spec1)
    assert T.run(d1, tmp_path, hooks(FakeRunner(), FakeScorer(TABLE))) == 0


def test_two_gpus_accept_a_soup_over_the_same_layers(tmp_path):
    spec = make_spec(tmp_path, devices=["cuda:0", "cuda:1"],
                     soups=[{"id": "s1", "of": ["c1", "c2"], "weights": None}])
    d = make_test(tmp_path, spec)
    assert T.run(d, tmp_path, hooks(FakeRunner(), FakeScorer({**TABLE, **{("s1", s): 50.0 for s in ("0.80", "1.00", "1.20")}}))) == 0


def test_dora_and_non_dora_contestants_are_refused_and_the_pool_checks_every_contestant(tmp_path):
    spec = make_spec(tmp_path)
    other = write_run(tmp_path, "other", dora=True)
    spec["contestants"][1].update(lora=other["final"], config=other["config"])
    with pytest.raises(T.SpecError, match="dora"):
        T.validate(spec, tmp_path)
    spec2 = make_spec(tmp_path, devices=["cuda:0", "cuda:1"])
    write_run(tmp_path, "elena")
    other = write_run(tmp_path, "other2", network_type="oft")
    spec2["contestants"][1].update(lora=other["final"], config=other["config"])
    with pytest.raises(T.SpecError, match="network_type"):  # (a mixed base is refused first; the pool check is below)
        T.validate(spec2, tmp_path)
    for both in ("oft",):  # every contestant OFT: the first is as bad as the second, and the second is named too
        a, b = write_run(tmp_path, "oa", network_type=both), write_run(tmp_path, "ob", network_type=both)
        s3 = make_spec(tmp_path, devices=["cuda:0", "cuda:1"])
        s3["contestants"][0].update(lora=a["final"], config=a["config"])
        s3["contestants"][1].update(lora=b["final"], config=b["config"])
        with pytest.raises(T.SpecError, match="OFT and DoRA"):
            T.validate(s3, tmp_path)


def valid_likeness(d: Path, **over):
    rec = json.loads(LIKENESS)
    rec.update(over)
    (d / "likeness.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")


def test_the_run_scale_needs_a_readable_scale_that_matches_the_scorer(tmp_path):
    spec = make_spec(tmp_path)
    folder = tmp_path / "output" / "elena"
    (folder / "likeness.jsonl").write_text("not json\n{}\n", encoding="utf-8")  # unreadable: never raises
    scale, note = T.decide_scale(T.validate(spec, tmp_path))
    assert scale == "references" and "no usable likeness.jsonl" in note
    valid_likeness(folder)
    assert T.decide_scale(T.validate(spec, tmp_path)) == ("run", None)
    valid_likeness(folder, face_model="antelopev2")  # recorded on another face model than the config says
    scale, note = T.decide_scale(T.validate(spec, tmp_path))
    assert scale == "references" and "recorded face_model antelopev2" in note
    valid_likeness(folder, body_score="anchored")
    assert "recorded body_score anchored" in T.decide_scale(T.validate(spec, tmp_path))[1]
    valid_likeness(folder, gold_name="facenet")  # face_gold "auto" follows whatever the run recorded: usable
    assert T.decide_scale(T.validate(spec, tmp_path)) == ("run", None)
    write_run(tmp_path, "elena", face_gold="adaface")  # explicit and not what the run recorded
    valid_likeness(folder, gold_name="facenet")
    assert "gold face" in T.decide_scale(T.validate(spec, tmp_path))[1]


def test_a_stop_that_comes_during_the_last_batch_still_ends_stopped(tmp_path):
    spec = make_spec(tmp_path, strengths=[1.0], seeds=[1], prompts=["p"], prompt_kinds=["both"])
    d = make_test(tmp_path, spec)
    count = []

    def stop_on_last(r, jobs, strength):
        count.append(1)
        if len(count) == 2:  # c2's only picture, the last of the test
            (d / "STOP").write_text("", encoding="utf-8")

    assert T.run(d, tmp_path, hooks(FakeRunner(on_render=stop_on_last), FakeScorer(TABLE))) == 0
    assert status_of(d)["state"] == "stopped" and results_of(d)["complete"] is False
    assert len(results_of(d)["cells"]) == 2


def test_queued_with_a_dead_pid_is_failed_and_without_a_pid_stays_queued(monkeypatch):
    monkeypatch.setattr(T, "pid_alive", lambda pid, created=None: False)
    assert T.effective_state({"state": "queued", "pid": 99}) == "failed"
    assert T.effective_state({"state": "queued", "pid": None}) == "queued"
    assert T.effective_state({"state": "done", "pid": 99}) == "done"
    monkeypatch.setattr(T, "pid_alive", lambda pid, created=None: True)
    assert T.effective_state({"state": "queued", "pid": 99}) == "queued"


def test_ranking_uses_full_precision_and_only_the_written_numbers_are_rounded(tmp_path):
    spec = make_spec(tmp_path, strengths=[1.0], seeds=[1], prompts=["p"], prompt_kinds=["both"])
    d = make_test(tmp_path, spec)
    table = {("c1", "1.00"): 60.003, ("c2", "1.00"): 60.004}  # equal once rounded to two decimals' first digit pair
    assert T.run(d, tmp_path, hooks(FakeRunner(), FakeScorer(table))) == 0
    res = results_of(d)
    assert res["winner"]["contestant"] == "c2"  # rounded first, c1 would win the tie
    assert res["winner"]["combined"] == 60.0 and res["cells"][0]["combined"] == 60.0 and res["columns"][0]["combined"] == 60.0


def test_malformed_spec_containers_name_the_field(tmp_path):
    for field, value, match in [("soups", {"a": 1}, "soups"), ("scoring", [1], "scoring"), ("render", "x", "render"),
                                ("contestants", {"a": 1}, "contestants"), ("prompts", "abc", "prompts"),
                                ("seeds", 5, "seeds"), ("strengths", "1.0", "strengths"), ("devices", "cuda:0", "devices"),
                                ("prompt_kinds", "face", "prompt_kinds")]:
        spec = make_spec(tmp_path)
        spec[field] = value
        with pytest.raises(T.SpecError, match=match):
            T.validate(spec, tmp_path)
    spec = make_spec(tmp_path)
    spec["scoring"]["face_refs"] = "refs/face.png"
    with pytest.raises(T.SpecError, match="face_refs"):
        T.validate(spec, tmp_path)
    spec = make_spec(tmp_path, soups=[5])
    with pytest.raises(T.SpecError, match=r"soups\[0\]"):
        T.validate(spec, tmp_path)
    spec = make_spec(tmp_path, contestants=[5])
    d = make_test(tmp_path, spec)
    assert T.run(d, tmp_path, hooks()) == 1 and status_of(d)["state"] == "refused"


def test_status_writer_keeps_keys_it_does_not_own_and_never_nulls_the_pid(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "queued", "pid": os.getpid(), "pid_created": 1700000000.5,
                                                      "future": {"a": 1}}), encoding="utf-8")
    w = T.StatusWriter(tmp_path, "tx")
    w.update(state="loading", pid=None)
    st = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert st["pid_created"] == 1700000000.5 and st["future"] == {"a": 1} and st["state"] == "loading"
    assert st["pid"] == os.getpid()  # its own pid, never null


def test_queued_without_a_pid_fails_only_after_the_start_gap(monkeypatch):
    q = {"state": "queued", "pid": None}
    assert T.effective_state(q) == "queued" and T.effective_state(q, age=5) == "queued"
    assert T.effective_state(q, age=T.QUEUED_START_GAP + 1) == "failed"
    assert T.effective_state({"state": "done", "pid": None}, age=9999) == "done"


def test_a_reused_pid_is_told_from_the_real_one_by_its_start_time():
    import os

    import psutil

    me = psutil.Process(os.getpid()).create_time()
    assert T.pid_alive(os.getpid(), me) is True
    assert T.pid_alive(os.getpid(), me + 0.5) is True  # within the 1 s tolerance
    assert T.pid_alive(os.getpid(), me + 30) is False  # another process that has this pid now
    assert T.effective_state({"state": "rendering", "pid": os.getpid(), "pid_created": me + 30}) == "failed"
    assert T.effective_state({"state": "rendering", "pid": os.getpid(), "pid_created": me}) == "rendering"


def test_a_run_records_its_own_start_time_unless_the_launcher_did_for_this_pid(tmp_path):
    import psutil

    me = psutil.Process(os.getpid()).create_time()
    w = T.StatusWriter(tmp_path, "tx")
    w.update(state="loading")
    assert status_of(tmp_path)["pid_created"] == pytest.approx(me, abs=0.01)
    # the launcher's record for this same pid stays
    (tmp_path / "status.json").write_text(json.dumps({"pid": os.getpid(), "pid_created": 123.0}), encoding="utf-8")
    T.StatusWriter(tmp_path, "tx").update(state="loading")
    assert status_of(tmp_path)["pid_created"] == 123.0
    # a record for another pid is not this process's
    (tmp_path / "status.json").write_text(json.dumps({"pid": 1, "pid_created": 123.0}), encoding="utf-8")
    T.StatusWriter(tmp_path, "tx").update(state="loading")
    assert status_of(tmp_path)["pid_created"] == pytest.approx(me, abs=0.01)


def test_new_leaves_no_pid_or_start_time_on_a_queued_test(tmp_path):
    tid = T.cmd_new(json.dumps(make_spec(tmp_path)), tmp_path)
    st = status_of(tmp_path / "output" / "_tests" / tid)
    assert st["pid"] is None and "pid_created" not in st
