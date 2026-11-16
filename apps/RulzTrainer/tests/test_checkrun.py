"""The check run (rulztrainer/checkrun.py): its catalogue's settings go through TrainConfig, the plan names what it
changes and leaves out, the runs land in the WebUI's and the fleet's queue, and the pass checks and the report read
made-up job folders. No model, no GPU, no instance: nothing here trains."""
import dataclasses
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import checkrun  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.train_queue import TrainQueue  # noqa: E402

FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


def swap_plan(cfg, vram):
    """A stand-in for the wizard's memory plan: more blocks for a bigger adapter, none on a big card."""
    if vram >= 24:
        return 0
    return {"lokr": 10}.get(cfg.network_type, 12 if cfg.lora_rank > 16 else 5)


def base(tmp_path, **kw) -> dict:
    (tmp_path / "ds").mkdir(exist_ok=True)
    (tmp_path / "reg").mkdir(exist_ok=True)
    data = {"model_arch": "krea2", "model_path": "krea/Krea-2-Raw", "dataset_dir": str(tmp_path / "ds"),
            "output_dir": str(tmp_path / "out"), "trigger_word": "zxperson", "face_refs": [str(tmp_path / "ds" / "a.png")],
            "body_refs": [str(tmp_path / "ds" / "b.png")], "steps": 2000}
    data.update(kw)
    return TrainConfig.from_dict(data).to_dict()


def planned(tmp_path, cases=None, vram: float | None = 16.0, **kw) -> dict:
    options = {"tag": "t1", "vram": vram, "reg_dir": str(tmp_path / "reg"), "planner": swap_plan}
    options.update({k: kw.pop(k) for k in list(kw) if k in ("tag", "reg_dir", "fleet", "busy", "steps")})
    picked = None if cases is None else [checkrun.case_named(n) for n in cases]
    return checkrun.plan(base(tmp_path, **kw), picked, **options)


def entry_of(manifest: dict, name: str) -> dict:
    return next(e for e in manifest["cases"] if e["name"] == name)


# ---------------------------------------------------------------- the catalogue


def test_the_catalogue_names_real_settings():
    names = [c.name for c in checkrun.CASES]
    assert len(names) == len(set(names)) and names[0] == "baseline"
    assert len(names) >= 9  # the baseline and at least eight features
    for case in checkrun.CASES:
        assert set(case.overrides) <= FIELDS, case.name
        for value in case.overrides.values():
            assert not (isinstance(value, str) and value.startswith("<")) or value in checkrun.PLANNED, case.name
        assert set(case.needs) <= set(checkrun.NEEDS), case.name
        assert not case.compare or case.compare in names, case.name
        assert case.probes in ("", "face", "body", "both"), case.name
        for pattern in case.expect.get("log", []) + [s["pattern"] for s in case.expect.get("numbers", [])] \
                + [p for p, _ in case.expect.get("unexercised", [])]:
            re.compile(pattern)
        json.dumps(case.expect)  # it travels in the manifest


# The messages the cases read, as the source writes them. A message reworded there must be reworded in the case's
# pattern too (rulztrainer/checkrun.py) and in LINES below, or a working feature would fail its check.
MESSAGES = {
    "reg_dir": ['"Regularisation dataset: %d pictures in %s at a loss weight of x%g',
                '"Regularisation: an epoch trains %d subject and %d regularisation pictures (of %d in the folder'],
    "block_swap_rank64": ['"%s: %d modules, %.2fM trainable params%s"',
                          '"Block swap: the frozen weights of blocks 0-%d (%d of %d) stay in RAM, %.1f GB pinned, and'],
    "block_swap_lokr": ['" (LoKr: w1 up to factor x factor per layer, w2 the rest)"'],
    "block_swap_compile": ['"block_swap_compile is experimental and untested: each block',
                           '"Compiled %d DiT blocks with torch.compile (first steps will be slow)%s"',
                           '"Compile stats (%s): %d frames compiled', 'compiling.report("end of training")',
                           '"compile_model requested but triton is not installed; running without compile"',
                           'log.info("torch.compile %s (%s)", "on" if do_compile else "off", why)'],
    "activation_offload": ['"activation_offload is experimental and untested: checkpointed blocks',
                           '"Activation offload: %.2f GB of block inputs sent to RAM in step %d"'],
    "normuon": ['"normuon is experimental and untested in RulZ AI Trainer"'],
    "oft_scaled_lr": ['"Scaled OFT (experimental): %s"', 'f"block {g[\'name\'][5:]} lr {g[\'lr\']:.2e}',
                      '"oft_scaled_lr is ignored with %s (it sets its own step sizes)"'],
    "oom_recovery": ['f"[oom recovery] try {len(tries)} of {cfg.oom_retries}: the last try ran out of {what} at step',
                     'f"\\n[oom recovery] The run ran out of {what} at step {step} and is not retried'],
    "face_antelopev2": ['"Could not initialise face scoring (%s); disabled"',
                        '"No faces found in any face reference image; face scoring disabled"'],
    "body_smplx_ratio": ['"NLF loaded from %s; SMPL-X shape scored on %s"',
                         '"Body references (%s): %d of %d measurable; profile %s; references',
                         '"Could not initialise body scoring (%s); disabled"',
                         '"No usable %s in any body reference image; body scoring disabled"'],
    "relay_race_pick": ['"Relay race: you pick the leader after each round on the Monitor (%s; then the highest score',
                        '"Relay race: round %d is over; pick the racer everyone continues from on the Monitor (%s scored',
                        '"Relay race: everyone continues from %s\'s LoRA (step %s)"',
                        '"Relay race over: %s won the last round (%s); its LoRA is the run\'s'],
    "h3_image_likeness": ['"clean_share on %s: the share is the steps at its last noise level (0.232'],
    "run_report": ['"Run report: %s"', '"Run report not written (%s: %s).'],
}


def test_the_log_lines_the_cases_wait_for_are_the_trainer_s_own():
    """Every message a case passes on (or is explained by) is still in the source as MESSAGES has it."""
    source = "\n".join((checkrun.ROOT / rel).read_text(encoding="utf-8") for rel in (
        "rulztrainer/trainer.py", "rulztrainer/bodyshape.py", "rulztrainer/compiling.py", "webui.py"))
    for name, fragments in MESSAGES.items():
        checkrun.case_named(name)
        for fragment in fragments:
            assert fragment in source, f"{name}: the source no longer has {fragment}"
    with_lines = {c.name for c in checkrun.CASES if c.expect.get("log") or c.expect.get("numbers")}
    assert with_lines <= set(MESSAGES) and with_lines <= set(LINES)
    # and the made-up lines the tests below use are ones the patterns take
    for case in checkrun.CASES:
        text = "\n".join(LINES.get(case.name, []))
        for pattern in case.expect.get("log", []) + [s["pattern"] for s in case.expect.get("numbers", [])]:
            assert re.search(pattern, text), f"{case.name}: {pattern}"


def test_every_case_validates_on_a_base_config_of_its_family(tmp_path):
    for family in ("krea2", "minimax_h3_image"):
        model = {"krea2": "krea/Krea-2-Raw", "minimax_h3_image": "MiniMaxAI/MiniMax-H3-Image"}[family]
        manifest = planned(tmp_path, vram=32.0, model_arch=family, model_path=model)
        for entry in manifest["cases"]:
            case = checkrun.case_named(entry["name"])
            if case.families and family not in case.families:
                assert entry["state"] == "skipped" and "only" in entry["reason"]
                continue
            assert entry["state"] == "queued", (family, entry["name"], entry["reason"])
            cfg = TrainConfig.from_dict(entry["config"])
            cfg.validate()
            assert cfg.job_name == entry["job"] == f"check_t1_{entry['name']}" and cfg.steps == entry["steps"]
            for key, value in case.overrides.items():  # the feature's settings are the ones queued
                if not (isinstance(value, str) and value.startswith("<")):
                    assert getattr(cfg, key) == value, (entry["name"], key)
    assert entry_of(manifest, "h3_image_likeness")["state"] == "queued"


def test_the_h3_recipe_is_the_wizard_button_s():
    import webui

    assert checkrun.case_named("h3_image_likeness").overrides == webui.H3_LIKENESS_RECIPE


def test_every_run_is_short_and_plain(tmp_path):
    manifest = planned(tmp_path, ["baseline"], probe_every=50, likeness_patience=3, save_every=100,
                       sample_prompts=["one", "two"], probe_face_prompt="portrait of zxperson", warmup_steps=100)
    entry = entry_of(manifest, "baseline")
    cfg = TrainConfig.from_dict(entry["config"])
    assert cfg.steps == checkrun.DEFAULT_STEPS and cfg.save_every == 0 and cfg.sample_every == 0
    assert cfg.probe_every == 0 and cfg.likeness_patience == 0 and cfg.checkpoint_keep_blocks == 0
    assert cfg.sample_prompts == ["one"] and cfg.warmup_steps == checkrun.DEFAULT_STEPS // 5
    assert entry["changed"] == {} and entry["plain"]["steps"] == [2000, checkrun.DEFAULT_STEPS]
    # a base config without a sample prompt gets one, so the sampling path runs too
    assert TrainConfig.from_dict(entry_of(planned(tmp_path, ["baseline"]), "baseline")["config"]).sample_prompts == [
        "a photo of zxperson"]


def test_probe_cases_probe_and_the_others_do_not(tmp_path):
    manifest = planned(tmp_path)
    face = TrainConfig.from_dict(entry_of(manifest, "face_antelopev2")["config"])
    assert face.face_model == "antelopev2" and face.probe_every == 15 and face.probe_face_prompt
    assert face.probe_body_prompt == "" and face.probe_floor_seeds == 1
    body = TrainConfig.from_dict(entry_of(manifest, "body_smplx_ratio")["config"])
    assert body.body_score == "smplx_ratio" and body.probe_body_prompt and body.probe_face_prompt == ""
    race = TrainConfig.from_dict(entry_of(manifest, "relay_race_pick")["config"])
    assert race.race and race.race_follow and race.race_pick and race.steps % race.probe_every == 0
    assert race.probe_face_prompt and race.probe_body_prompt and len(race.race_configs) == 2
    assert TrainConfig.from_dict(entry_of(manifest, "normuon")["config"]).probe_every == 0
    # the base config's own probe prompt is kept
    own = planned(tmp_path, ["face_antelopev2"], probe_face_prompt="headshot of zxperson")
    assert entry_of(own, "face_antelopev2")["config"]["probe_face_prompt"] == "headshot of zxperson"


def test_block_swap_follows_the_plan_for_the_adapter(tmp_path):
    manifest = planned(tmp_path, vram=16.0)
    rank64, lokr, comp = (entry_of(manifest, n) for n in ("block_swap_rank64", "block_swap_lokr", "block_swap_compile"))
    assert rank64["config"]["block_swap"] == 12 and lokr["config"]["block_swap"] == 10
    assert comp["config"]["block_swap"] == 5 and comp["config"]["compile_model"] is True
    assert rank64["expect"]["peak_max_gb"] == 16.0 and checkrun.BLOCK_SWAP_LINE in rank64["expect"]["log"]
    assert rank64["plan"]["wizard_block_swap"] == 12 and rank64["plan"]["adapter_gb"] > 4
    assert entry_of(manifest, "baseline")["plan"]["adapter_gb"] < 2
    # a card that holds it all: no blocks swapped and no block swap line asked for; the compile case still swaps 4
    big = planned(tmp_path, vram=32.0)
    assert entry_of(big, "block_swap_rank64")["config"]["block_swap"] == 0
    assert checkrun.BLOCK_SWAP_LINE not in entry_of(big, "block_swap_rank64")["expect"]["log"]
    assert entry_of(big, "block_swap_compile")["config"]["block_swap"] == 4
    # the base config's own block swap is the compile case's floor
    assert entry_of(planned(tmp_path, ["block_swap_compile"], vram=32.0, block_swap=9),
                    "block_swap_compile")["config"]["block_swap"] == 9


def test_the_wizard_s_own_plan_counts_the_adapter():
    """The real plan (webui.wizard_recommendations), on the CPU: a rank-64 LoRA swaps more blocks than the default."""
    small = checkrun.wizard_block_swap(TrainConfig(model_arch="krea2"), 16.0)
    large = checkrun.wizard_block_swap(TrainConfig(model_arch="krea2", lora_rank=64, lora_alpha=64.0), 16.0)
    assert 0 < small < large
    assert checkrun.wizard_block_swap(TrainConfig(model_arch="krea2"), 48.0) == 0


def test_cases_that_cannot_run_say_why(tmp_path):
    manifest = planned(tmp_path, vram=None, reg_dir="", face_refs=[], body_refs=[])
    why = {e["name"]: e["reason"] for e in manifest["cases"] if e["state"] == "skipped"}
    assert "--reg-dir" in why["reg_dir"] and "--vram" in why["block_swap_rank64"] and "--vram" in why["block_swap_lokr"]
    assert "face_refs" in why["face_antelopev2"] and "body_refs" in why["body_smplx_ratio"]
    assert "face_refs / body_refs" in why["relay_race_pick"] and "minimax_h3_image" in why["h3_image_likeness"]
    queued = [e["name"] for e in manifest["cases"] if e["state"] == "queued"]
    assert {"baseline", "block_swap_compile", "activation_offload", "normuon", "oom_recovery"} <= set(queued)
    # a regularisation folder that is not there (on this computer; an instance's folder is not looked up)
    gone = str(tmp_path / "nope")
    assert "not there" in entry_of(planned(tmp_path, ["reg_dir"], reg_dir=gone), "reg_dir")["reason"]
    assert entry_of(planned(tmp_path, ["reg_dir"], reg_dir=gone, fleet=True), "reg_dir")["state"] == "queued"
    # settings of the base config a case cannot go with: left out with the trainer's own message
    clash = planned(tmp_path, ["reg_dir"], reg_dir=str(tmp_path / "ds"))
    assert "reg_dir is the dataset" in entry_of(clash, "reg_dir")["reason"]


def test_the_memory_recovery_case_needs_a_card_that_holds_the_model(tmp_path):
    assert "block swap (5 blocks)" in entry_of(planned(tmp_path, ["oom_recovery"], vram=16.0), "oom_recovery")["reason"]
    assert "block swap (7 blocks)" in entry_of(planned(tmp_path, ["oom_recovery"], vram=None, block_swap=7),
                                               "oom_recovery")["reason"]
    entry = entry_of(planned(tmp_path, ["oom_recovery"], vram=32.0), "oom_recovery")
    cfg = TrainConfig.from_dict(entry["config"])
    assert entry["state"] == "queued" and not cfg.gradient_checkpointing and cfg.block_swap == 0
    assert cfg.oom_retries == 3 and cfg.save_state_every == cfg.steps  # a fresh start is allowed at any step
    from rulztrainer import oom_recovery

    d = oom_recovery.plan(entry["config"], {"step": 3, "states": []}, "gpu", [], 28)
    assert d["do"] == "restart" and d["config"]["gradient_checkpointing"] is True


def test_job_names_are_unique(tmp_path):
    out = tmp_path / "out"
    (out / "check_t1_baseline").mkdir(parents=True)
    (out / "check_t1_baseline" / "check_t1_baseline.safetensors").write_bytes(b"x")  # an earlier run's file
    (out / "check_t1_normuon").mkdir()  # a folder whose run saved nothing is free
    manifest = planned(tmp_path, ["baseline", "normuon", "reg_dir"], busy={"check_t1_reg_dir"})
    assert entry_of(manifest, "baseline")["job"] == "check_t1_baseline_2"
    assert entry_of(manifest, "normuon")["job"] == "check_t1_normuon"
    assert entry_of(manifest, "reg_dir")["job"] == "check_t1_reg_dir_2"
    with pytest.raises(ValueError):
        planned(tmp_path, ["baseline"], tag="bad tag!")
    with pytest.raises(ValueError):
        checkrun.plan({"no_such_setting": 1})


def test_a_case_brings_its_reference_along():
    assert [c.name for c in checkrun.select(["normuon"])] == ["baseline", "normuon"]
    assert [c.name for c in checkrun.select(["normuon"], ["baseline"])] == ["normuon"]
    assert [c.name for c in checkrun.select(["run_report"])] == ["run_report"]
    assert len(checkrun.select()) == len(checkrun.CASES)
    with pytest.raises(ValueError):
        checkrun.select(["nope"])


# ---------------------------------------------------------------- queueing


def test_the_runs_go_into_the_webui_queue(tmp_path):
    q = TrainQueue(tmp_path / "out" / "_queue.json")
    q.set_paused(True)
    manifest = planned(tmp_path, ["baseline", "normuon", "h3_image_likeness"])
    state = checkrun.queue_local(manifest, queue=q)
    assert [e["job"] for e in state["waiting"]] == ["check_t1_baseline", "check_t1_normuon"] and state["paused"]
    assert state["waiting"][1]["config"]["optimizer"] == "normuon" and "check run" in state["waiting"][1]["label"]
    assert entry_of(manifest, "normuon")["queue_id"] == state["waiting"][1]["id"]
    assert "queue_id" not in entry_of(manifest, "h3_image_likeness")
    for waiting in state["waiting"]:  # what the WebUI will start: a config the trainer takes
        TrainConfig.from_dict(waiting["config"]).validate()
    assert checkrun.waiting_names(q) == {"check_t1_baseline", "check_t1_normuon"}
    assert checkrun.queue_local(planned(tmp_path, ["run_report"], tag="t2"), queue=q, resume=True)["paused"] is False


def test_the_runs_go_into_the_fleet_queue(tmp_path):
    module = checkrun.fleet_module()
    fleet = module.Fleet(home=tmp_path / "fleet", list_instances=lambda: [])
    manifest = planned(tmp_path, ["baseline", "block_swap_rank64"], fleet=True)
    state = checkrun.queue_fleet(manifest, fleet=fleet, check=False)
    waiting = fleet.read()["queue"]["waiting"]
    assert [e["job"] for e in waiting] == ["check_t1_baseline", "check_t1_block_swap_rank64"]
    assert waiting[1]["config"]["lora_rank"] == 64 and waiting[1]["config"]["block_swap"] == 12
    assert waiting[0]["download"] == checkrun.FLEET_DOWNLOAD and "check run t1" in waiting[0]["label"]
    assert state["looking"] is False and len(state["queue"]["waiting"]) == 2
    # a run the fleet refuses is left out with its reason; the others still go
    again = planned(tmp_path, ["baseline", "normuon"], fleet=True)
    checkrun.queue_fleet(again, fleet=fleet, check=False)
    assert entry_of(again, "baseline")["state"] == "refused" and "Already queued" in entry_of(again, "baseline")["reason"]
    assert entry_of(again, "normuon")["state"] == "queued" and len(fleet.read()["queue"]["waiting"]) == 3


def test_the_card_size_comes_from_nvidia_smi(monkeypatch):
    class Done:
        returncode, stdout = 0, "16384\n32768\n"

    monkeypatch.setattr(checkrun.subprocess, "run", lambda *a, **k: Done())
    assert checkrun.detect_vram("cuda:0") == 16.0 and checkrun.detect_vram("cuda:1") == 32.0

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(checkrun.subprocess, "run", missing)
    assert checkrun.detect_vram() is None


# ---------------------------------------------------------------- made-up job folders


LINES = {
    "reg_dir": ["Regularisation dataset: 40 pictures in /data/reg at a loss weight of x1; they train whole and plain",
                "Regularisation: an epoch trains 20 subject and 20 regularisation pictures (of 40 in the folder"],
    "block_swap_rank64": ["Krea 2 LoRA rank 64 alpha 64: 300 modules, 228.50M trainable params",
                          "Block swap: the frozen weights of blocks 0-11 (12 of 28) stay in RAM, 5.8 GB pinned, and"],
    "block_swap_lokr": ["Krea 2 LoKr factor 8: 300 modules, 195.30M trainable params (LoKr: w1 up to factor x factor",
                        "Block swap: the frozen weights of blocks 0-9 (10 of 28) stay in RAM, 4.8 GB pinned, and"],
    "block_swap_compile": ["Block swap: the frozen weights of blocks 0-4 (5 of 28) stay in RAM, 2.4 GB pinned, and",
                           "block_swap_compile is experimental and untested: each block's forward is compiled",
                           "Compiled 28 DiT blocks with torch.compile (first steps will be slow)",
                           "Compile stats (end of training): 30 frames compiled (30 ok), 28 graphs"],
    "activation_offload": ["activation_offload is experimental and untested: checkpointed blocks' inputs wait",
                           "Activation offload: 1.42 GB of block inputs sent to RAM in step 1"],
    "normuon": ["normuon is experimental and untested in RulZ AI Trainer"],
    "oft_scaled_lr": ["Scaled OFT (experimental): block 64 lr 7.01e-05 (300 layers)"],
    "oom_recovery": ["[oom recovery] try 1 of 3: the last try ran out of GPU memory at step 0; nothing saved yet"],
    "body_smplx_ratio": ["NLF loaded from nlf_l_multi.torchscript; SMPL-X shape scored on cuda:0",
                         "Body references (smplx_ratio): 4 of 4 measurable; profile ok; references' own level 80"],
    "relay_race_pick": ["Relay race: you pick the leader after each round on the Monitor (waiting up to 1 minutes",
                        "Relay race: round 1 is over; pick the racer everyone continues from on the Monitor (a scored",
                        "Relay race: everyone continues from a's LoRA (step 10)",
                        "Relay race over: a won the last round (seed=101); its LoRA is the run's"],
    "h3_image_likeness": ["clean_share on MiniMax-H3-Image: the share is the steps at its last noise level (0.232"],
    "run_report": ["Run report: /out/job/report/job_report.png, /out/job/report/job_report.pdf"],
}


def make_job(root: Path, entry: dict, peak=11.2, lines=None, **status) -> Path:
    """A job folder as a run that passed would leave it; `status` overrides status.json's fields."""
    name, steps = entry["job"], entry["steps"]
    folder = root / name
    (folder / "samples").mkdir(parents=True, exist_ok=True)
    race = bool(entry["expect"].get("race"))
    total = steps * 2 if race else steps
    data = {"stage": "finished", "done": True, "error": None, "step": total, "total": total, "loss": 0.21,
            "avg_loss": 0.2345, "losses": [0.3, 0.25, 0.21], "step_times": [9.0, 4.0, 1.5, 1.4, 1.6, 1.5],
            "peak_mem": {"cuda:0": peak}, "saved": [f"/remote/out/{name}/{name}.safetensors"],
            "samples": [f"/remote/out/{name}/samples/step{steps:06d}_0.png"]}
    if entry["name"] == "reg_dir":
        data.update(reg_images=40, reg_per_epoch=20)
    data.update(status)
    (folder / "status.json").write_text(json.dumps(data), encoding="utf-8")
    body = LINES.get(entry["name"], []) if lines is None else lines
    (folder / "train.log").write_text("".join(f"12:00:00 INFO {line}\n" for line in body)
                                      + f"12:30:00 INFO Done. Final LoRA: {name}.safetensors\n", encoding="utf-8")
    (folder / f"{name}.safetensors").write_bytes(b"lora")
    (folder / "samples" / f"step{steps:06d}_0.png").write_bytes(b"png")
    if entry["name"] == "oom_recovery":
        (folder / "oom_recovery.json").write_text(json.dumps({"tries": [
            {"kind": "gpu", "step": 0, "action": "started again, gradient checkpointing on", "when": 1.0}]}),
            encoding="utf-8")
        (folder / "train_oom1.log").write_text("first try\n", encoding="utf-8")
    if race:
        (folder / "race.json").write_text("{}", encoding="utf-8")
        (folder / f"{name}_winner.yaml").write_text("seed: 101\n", encoding="utf-8")
    if "probe" in entry["expect"]:
        records = [{"step": s, "face": 0.31 + s / 1000, "body": 0.52, "face_model": entry["config"]["face_model"],
                    "body_score": entry["config"]["body_score"]} for s in (0, steps // 2, steps)]
        (folder / "likeness.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return folder


def failed(res: dict) -> list[str]:
    return [c["name"] for c in res["checks"] if c["ok"] is False]


def test_every_case_passes_on_the_folder_a_good_run_leaves(tmp_path):
    for family, model, vram in (("krea2", "krea/Krea-2-Raw", 16.0), ("krea2", "krea/Krea-2-Raw", 32.0),
                                ("minimax_h3_image", "MiniMaxAI/MiniMax-H3-Image", 48.0)):
        manifest = planned(tmp_path, vram=vram, model_arch=family, model_path=model, tag=f"{family[:4]}{int(vram)}")
        for entry in manifest["cases"]:
            if entry["state"] != "queued":
                continue
            lines = LINES.get(entry["name"], [])
            if entry["config"]["block_swap"] == 0:
                lines = [line for line in lines if not line.startswith("Block swap")]
            res = checkrun.evaluate(entry, make_job(tmp_path / "jobs", entry, lines=lines))
            assert res["verdict"] == "pass", (entry["name"], failed(res), res["notes"])
            assert res["peak_gb"] == 11.2 and res["step_seconds"] == 1.5 and res["loss"] == 0.2345
            assert res["step"] == res["total"]


def test_a_missing_log_line_fails_the_feature_and_says_which(tmp_path):
    manifest = planned(tmp_path)
    for name in ("reg_dir", "normuon", "activation_offload", "relay_race_pick", "run_report", "oft_scaled_lr"):
        entry = entry_of(manifest, name)
        res = checkrun.evaluate(entry, make_job(tmp_path / "jobs", entry, lines=[]))
        assert res["verdict"] == "fail" and failed(res), name
        assert all(c["feature"] for c in res["checks"] if c["ok"] is False), name
    # the baseline has no line of its own to miss
    entry = entry_of(manifest, "baseline")
    assert checkrun.evaluate(entry, make_job(tmp_path / "jobs", entry, lines=[]))["verdict"] == "pass"


def test_what_every_run_is_checked_for(tmp_path):
    entry = entry_of(planned(tmp_path, ["baseline"]), "baseline")
    jobs = tmp_path / "jobs"

    def verdict(**status):
        res = checkrun.evaluate(entry, make_job(jobs, entry, **status))
        return res["verdict"], failed(res), res

    assert verdict()[0] == "pass"
    v, bad, _ = verdict(step=12)  # stopped early
    assert v == "fail" and bad == ["it reached its last step"]
    v, bad, _ = verdict(total=40, step=40)  # another run's step count
    assert v == "fail" and "it reached its last step" in bad
    v, bad, _ = verdict(loss=float("nan"))
    assert v == "fail" and bad == ["the loss is a number"]
    assert verdict(losses=[0.3, float("inf")])[1] == ["the loss is a number"]
    assert verdict(losses=[])[1] == ["the loss is a number"]
    assert verdict(skipped_steps=2)[1] == ["no step was left out for non-finite gradients"]
    v, bad, res = verdict(error="OutOfMemoryError: CUDA out of memory", stage="error", step=3)
    assert v == "fail" and "the run finished" in bad and "CUDA out of memory" in res["checks"][0]["detail"]
    v, _, res = verdict(done=False, stage="training", step=7)
    assert v == "not finished" and "step 7 of 30" in res["notes"][0] and not res["checks"]
    v, _, res = verdict(peak_mem={})  # a run on the CPU: no peak, said in the notes
    assert v == "pass" and res["peak_gb"] is None and any("peak_mem" in n for n in res["notes"])
    # the folder as the fleet fetches it: no LoRA file and no pictures, but status.json lists them
    folder = make_job(jobs, entry)
    (folder / f"{entry['job']}.safetensors").unlink()
    (folder / "samples" / "step000030_0.png").unlink()
    res = checkrun.evaluate(entry, folder)
    assert res["verdict"] == "pass" and "listed in status.json" in res["checks"][4]["detail"]
    res = checkrun.evaluate(entry, make_job(jobs, entry, saved=[], samples=[]))
    assert res["verdict"] == "pass"  # (the files themselves are there)
    (folder / f"{entry['job']}.safetensors").unlink()
    (folder / "samples" / "step000030_0.png").unlink()
    (folder / "status.json").write_text(json.dumps({**json.loads((folder / "status.json").read_text()), "saved": [],
                                                    "samples": []}))
    assert failed(checkrun.evaluate(entry, folder)) == ["the final LoRA was saved", "the final sample was rendered"]
    # nothing there yet
    assert checkrun.evaluate(entry, None)["verdict"] == "not started"
    (jobs / "empty").mkdir()
    assert checkrun.evaluate(entry, jobs / "empty")["verdict"] == "not started"
    skipped = {"name": "reg_dir", "state": "skipped", "reason": "needs a folder", "job": ""}
    assert checkrun.evaluate(skipped)["verdict"] == "skipped" and checkrun.evaluate(skipped)["notes"] == ["needs a folder"]


def test_a_feature_that_never_got_to_act_is_not_a_pass(tmp_path):
    manifest = planned(tmp_path, vram=32.0)
    jobs = tmp_path / "jobs"
    # torch.compile without triton
    entry = entry_of(manifest, "block_swap_compile")
    res = checkrun.evaluate(entry, make_job(jobs, entry, lines=[
        LINES["block_swap_compile"][0], "compile_model requested but triton is not installed; running without compile"]))
    assert res["verdict"] == "not exercised" and "triton" in res["notes"][0]
    res = checkrun.evaluate(entry, make_job(jobs, entry, lines=[
        "torch.compile off (Keel's face-token gain hooks the LoRA layers' outputs (keel_face_gain))"]))
    assert res["verdict"] == "not exercised" and "Keel's face-token gain" in res["notes"][0]
    # the run that was to run out of memory, and did not
    entry = entry_of(manifest, "oom_recovery")
    folder = make_job(jobs, entry, lines=[])
    (folder / "oom_recovery.json").unlink()
    (folder / "train_oom1.log").unlink()
    res = checkrun.evaluate(entry, folder)
    assert res["verdict"] == "not exercised" and "never ran out of memory" in res["notes"][0]
    # it ran out and was not started again: the run itself failed
    res = checkrun.evaluate(entry, make_job(jobs, entry, error="OutOfMemoryError: CUDA out of memory", stage="error",
                                           step=0, lines=["[oom recovery] The run ran out of GPU memory at step 0 and "
                                                          "is not retried: 3 automatic tries already."]))
    assert res["verdict"] == "fail" and any("is not retried" in n for n in res["notes"])
    # an optimizer the scaled OFT rate does not act on
    entry = entry_of(manifest, "oft_scaled_lr")
    res = checkrun.evaluate(entry, make_job(jobs, entry, lines=["oft_scaled_lr is ignored with keel (it sets its own"]))
    assert res["verdict"] == "not exercised" and "keel" in res["notes"][0]


def test_probe_checks_read_the_likeness_records(tmp_path):
    manifest = planned(tmp_path)
    jobs = tmp_path / "jobs"
    entry = entry_of(manifest, "face_antelopev2")
    folder = make_job(jobs, entry)
    assert checkrun.evaluate(entry, folder)["verdict"] == "pass"
    log = folder / "likeness.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    log.write_text("".join(json.dumps({**r, "face_model": "buffalo_l"}) + "\n" for r in records))
    assert failed(checkrun.evaluate(entry, folder)) == ["the probe was measured with face_model antelopev2"]
    log.write_text("".join(json.dumps({**r, "face": None}) + "\n" for r in records))
    assert failed(checkrun.evaluate(entry, folder)) == ["a probe has a face score"]
    log.write_text(json.dumps(records[0]) + "\nnot json\n")  # the step-0 probe alone: nothing of the trained LoRA
    res = checkrun.evaluate(entry, folder)
    assert "a likeness probe of the trained LoRA is in likeness.jsonl" in failed(res)
    log.unlink()
    with open(folder / "train.log", "a", encoding="utf-8") as f:
        f.write("12:01:00 WARNING Could not initialise face scoring (antelopev2 download failed); disabled\n")
    res = checkrun.evaluate(entry, folder)
    assert res["verdict"] == "fail" and any("antelopev2 download failed" in n for n in res["notes"])


def test_the_peak_and_the_adapter_s_memory_are_held_against_the_plan(tmp_path):
    manifest = planned(tmp_path, ["baseline", "block_swap_rank64", "block_swap_lokr", "normuon"], vram=16.0)
    jobs = tmp_path / "jobs"
    base_entry, rank64, lokr = (entry_of(manifest, n) for n in ("baseline", "block_swap_rank64", "block_swap_lokr"))
    counted = rank64["plan"]["adapter_gb"] - base_entry["plan"]["adapter_gb"]  # what the plan says rank 64 adds
    make_job(jobs, base_entry, peak=14.0)
    make_job(jobs, entry_of(manifest, "normuon"), peak=13.8)
    # rank 64 swaps 12 blocks holding 5.8 GB, of which two blocks' worth are on the GPU at any time: 4.83 GB are off
    # it, so a peak of 14.0 + counted - 4.83 would be exactly as planned
    off = checkrun.off_gpu_gb("\n".join(LINES["block_swap_rank64"]))
    assert off == round(5.8 * 10 / 12, 2) and checkrun.off_gpu_gb("no such line") == 0.0
    assert checkrun.off_gpu_gb("Block swap: the frozen weights of blocks 0-1 (2 of 28) stay in RAM, 1.0 GB pinned") == 0
    make_job(jobs, rank64, peak=round(14.0 + counted - off + 0.4, 2))
    make_job(jobs, lokr, peak=17.5)  # over the 16 GB card the block swap was planned for
    rep = checkrun.report(manifest, roots=[jobs])
    by = {r["case"]: r for r in rep["results"]}
    assert by["block_swap_rank64"]["verdict"] == "pass"
    memory = next(c for c in by["block_swap_rank64"]["checks"] if "plan counts" in c["name"])
    assert memory["ok"] is True and f"the plan counts {counted:+.2f} GB" in memory["detail"]
    assert any("against baseline's 14 GB" in n for n in by["block_swap_rank64"]["notes"])
    assert by["block_swap_lokr"]["verdict"] == "fail"
    assert any("16 GB card the block swap was planned for" in name for name in failed(by["block_swap_lokr"]))
    assert any("x1.00 of baseline's" in n for n in by["normuon"]["notes"])
    assert rep["counts"] == {"pass": 3, "fail": 1} and rep["warning"] == ""
    # an adapter that holds 2 GB more than counted: the plan is too low, and that fails the case
    make_job(jobs, rank64, peak=round(14.0 + counted - off + 2.0, 2))
    by = {r["case"]: r for r in checkrun.report(manifest, roots=[jobs])["results"]}
    assert by["block_swap_rank64"]["verdict"] == "fail"
    assert failed(by["block_swap_rank64"]) == ["the adapter holds no more memory than the block swap plan counts"]
    # far under the count: passed, with a note that the baseline's peak may be a loading peak
    make_job(jobs, rank64, peak=round(14.0 + counted - off - 2.5, 2))
    by = {r["case"]: r for r in checkrun.report(manifest, roots=[jobs])["results"]}
    assert by["block_swap_rank64"]["verdict"] == "pass"
    assert any("under the plan's count" in n for n in by["block_swap_rank64"]["notes"])
    # a run that encoded prompts first had the text encoder on the GPU: its peak is not compared
    make_job(jobs, rank64, peak=30.0, lines=LINES["block_swap_rank64"] + [
        "Caching text embeddings: 3 of 40 prompts need encoding"])
    res = {r["case"]: r for r in checkrun.report(manifest, roots=[jobs])["results"]}["block_swap_rank64"]
    memory = next(c for c in res["checks"] if "plan counts" in c["name"])
    assert memory["ok"] is None and "encoded prompts first" in memory["detail"]
    assert any("encoded 3 prompt(s)" in n for n in res["notes"])
    make_job(jobs, rank64, peak=round(14.0 + counted - off, 2))
    # without a passed baseline there is nothing to compare with: said, not failed
    make_job(jobs, base_entry, error="boom", stage="error")
    rep = checkrun.report(manifest, roots=[jobs])
    by = {r["case"]: r for r in rep["results"]}
    memory = next(c for c in by["block_swap_rank64"]["checks"] if "plan counts" in c["name"])
    assert memory["ok"] is None and by["block_swap_rank64"]["verdict"] == "pass"
    assert "baseline failed" in rep["warning"]


def test_the_card_size_and_the_peak_are_read_in_the_same_unit(tmp_path):
    """nvidia-smi's 24 GB card is 24 GiB; peak_mem counts in GB of 1e9 bytes, so 25.2 GB (23.5 GiB) still fits."""
    manifest = planned(tmp_path, ["block_swap_rank64"], vram=24.0)
    entry = entry_of(manifest, "block_swap_rank64")
    lines = [LINES["block_swap_rank64"][0]]
    res = checkrun.evaluate(entry, make_job(tmp_path / "jobs", entry, peak=25.2, lines=lines))
    assert res["verdict"] == "pass" and "23.47 of the card's 24" in res["checks"][-1]["detail"]
    res = checkrun.evaluate(entry, make_job(tmp_path / "jobs", entry, peak=26.0, lines=lines))
    assert res["verdict"] == "fail"


def test_a_baseline_on_several_gpus_is_not_compared(tmp_path):
    manifest = planned(tmp_path, ["baseline", "block_swap_rank64"], vram=16.0, multi_gpu="model_split")
    assert entry_of(manifest, "baseline")["config"]["multi_gpu"] == "model_split"
    assert entry_of(manifest, "block_swap_rank64")["config"]["multi_gpu"] == "off"
    jobs = tmp_path / "jobs"
    make_job(jobs, entry_of(manifest, "baseline"), peak=7.0)
    make_job(jobs, entry_of(manifest, "block_swap_rank64"), peak=15.0)
    res = {r["case"]: r for r in checkrun.report(manifest, roots=[jobs])["results"]}["block_swap_rank64"]
    memory = next(c for c in res["checks"] if "plan counts" in c["name"])
    assert memory["ok"] is None and "several GPUs" in memory["detail"] and res["verdict"] == "pass"


def test_a_run_restarted_after_running_out_of_memory_is_not_the_run_that_was_planned(tmp_path):
    manifest = planned(tmp_path, vram=32.0)
    for entry in manifest["cases"]:  # no run but the memory recovery case may be started again by the WebUI
        if entry["state"] == "queued":
            assert entry["config"]["oom_retries"] == (3 if entry["name"] == "oom_recovery" else 0), entry["name"]
    jobs = tmp_path / "jobs"
    entry = entry_of(manifest, "block_swap_rank64")
    folder = make_job(jobs, entry, lines=[LINES["block_swap_rank64"][0]])
    (folder / "oom_recovery.json").write_text(json.dumps({"tries": [
        {"kind": "gpu", "step": 0, "action": "started again, block swap 0 -> 4"}, "not a try"]}), encoding="utf-8")
    res = checkrun.evaluate(entry, folder)
    assert res["verdict"] == "fail" and "block swap 0 -> 4" in next(c["detail"] for c in res["checks"] if not c["ok"])
    # the memory recovery case itself: a second try (the first carried on from a state on the same settings) passes
    entry = entry_of(manifest, "oom_recovery")
    folder = make_job(jobs, entry, lines=["[oom recovery] try 2 of 3: the last try ran out of GPU memory at step 4"])
    (folder / "train_oom1.log").unlink()
    (folder / "train_oom2.log").write_text("second try\n", encoding="utf-8")
    assert checkrun.evaluate(entry, folder)["verdict"] == "pass"
    # odd files do not stop the report
    (folder / "oom_recovery.json").write_text("[1, 2]", encoding="utf-8")
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    (folder / "status.json").write_text(json.dumps({**status, "step_times": 1.5, "peak_mem": [1]}), encoding="utf-8")
    res = checkrun.evaluate(entry, folder)
    assert res["verdict"] == "not exercised" and res["step_seconds"] is None and res["peak_gb"] is None


def test_the_plain_run_drops_what_would_stop_or_render_beside_it(tmp_path):
    manifest = planned(tmp_path, ["baseline", "reg_dir"], holdout_images=4, holdout_patience=2, holdout_track_best=True,
                       region_sweep_every=1, training_maps=True, isolation=True, oom_retries=3, region_steer=True)
    cfg = TrainConfig.from_dict(entry_of(manifest, "baseline")["config"])
    assert (cfg.holdout_images, cfg.holdout_patience, cfg.region_sweep_every, cfg.oom_retries) == (0, 0, 0, 0)
    assert not (cfg.training_maps or cfg.isolation or cfg.holdout_track_best or cfg.region_steer or cfg.reg_dir)
    assert entry_of(manifest, "reg_dir")["config"]["reg_dir"] == str(tmp_path / "reg")
    # a base config with its own regularisation folder: the baseline trains without it, the case with it
    own = checkrun.plan(base(tmp_path, reg_dir=str(tmp_path / "reg")),
                        [checkrun.case_named(n) for n in ("baseline", "reg_dir")], tag="t1", planner=swap_plan)
    assert entry_of(own, "baseline")["config"]["reg_dir"] == ""
    assert entry_of(own, "reg_dir")["config"]["reg_dir"] == str(tmp_path / "reg")
    assert "--reg-dir" in entry_of(planned(tmp_path, ["reg_dir"], reg_dir=""), "reg_dir")["reason"]


def test_the_names_the_webui_runs_or_holds_are_taken(tmp_path):
    q = TrainQueue(tmp_path / "out" / "_queue.json")
    q.add({"job_name": "check_t1_normuon"})
    job_file = tmp_path / ".webui_job.json"
    job_file.write_text(json.dumps({"pid": 1, "job_dir": str(tmp_path / "out" / "check_t1_baseline")}), encoding="utf-8")
    assert checkrun.waiting_names(q, job_file) == {"check_t1_normuon", "check_t1_baseline"}
    assert checkrun.waiting_names(q, tmp_path / "none.json") == {"check_t1_normuon"}
    manifest = planned(tmp_path, ["baseline", "normuon"], busy=checkrun.waiting_names(q, job_file))
    assert [e["job"] for e in manifest["cases"]] == ["check_t1_baseline_2", "check_t1_normuon_2"]
    assert len(checkrun.plan(base(tmp_path), [])["tag"]) == 13  # the default tag: date and time to the second


def test_the_report_is_written_as_markdown_and_json(tmp_path):
    manifest = planned(tmp_path, ["baseline", "normuon", "reg_dir", "h3_image_likeness", "run_report"])
    jobs = tmp_path / "jobs"
    make_job(jobs, entry_of(manifest, "baseline"))
    make_job(jobs, entry_of(manifest, "normuon"), lines=[])
    make_job(jobs, entry_of(manifest, "reg_dir"), done=False, stage="training", step=4)
    rep = checkrun.report(manifest, roots=[jobs])
    assert [r["verdict"] for r in rep["results"]] == ["pass", "fail", "not finished", "skipped", "not started"]
    paths = checkrun.save_report(rep, tmp_path / "rep")
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "| baseline | pass | 30 / 30 | 0.2345 | 11.2 | 1.5 |" in text and "| normuon | fail |" in text
    assert "failed: train.log: normuon is experimental" in text and "### h3_image_likeness: skipped" in text
    assert "- FAIL: train.log: normuon is experimental and untested in RulZ AI Trainer (no such line)" in text
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["counts"] == rep["counts"]
    words = checkrun.report_text(rep)
    assert "normuon: failed: train.log: normuon is experimental" in words and "1 pass" in words
    text.encode("ascii")  # the table prints on any console


def test_job_folders_are_found_here_and_among_the_fleet_s_fetched_runs(tmp_path, monkeypatch):
    manifest = planned(tmp_path, ["baseline"])
    entry = entry_of(manifest, "baseline")
    monkeypatch.setenv("RULZ_VAST_HOME", str(tmp_path / "vast"))
    assert checkrun.find_job(entry["job"], manifest) is None
    fetched = make_job(tmp_path / "vast" / "runs" / "4242", entry)
    assert checkrun.find_job(entry["job"], manifest) == fetched
    own = make_job(tmp_path / "out", entry)  # the check run's own output folder, written later: the newer one
    import os

    os.utime(fetched / "status.json", (1, 1))
    assert checkrun.find_job(entry["job"], manifest) == own
    assert checkrun.find_job(entry["job"], manifest, roots=[tmp_path / "elsewhere"]) == own


# ---------------------------------------------------------------- the command line


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A base config file, the manifests and the WebUI queue in a temporary folder, and no card to ask."""
    monkeypatch.setattr(checkrun, "MANIFESTS", tmp_path / "manifests")
    q = TrainQueue(tmp_path / "out" / "_queue.json")
    monkeypatch.setattr(checkrun, "webui_queue", lambda: q)
    monkeypatch.setattr(checkrun, "detect_vram", lambda device="cuda:0": None)
    monkeypatch.setattr(checkrun, "wizard_block_swap", swap_plan)
    monkeypatch.setenv("RULZ_VAST_HOME", str(tmp_path / "vast"))
    config = tmp_path / "base.yaml"
    TrainConfig.from_dict(base(tmp_path)).save(config)
    return config, q


def test_list_and_dry_run_queue_nothing(cli, tmp_path, capsys):
    config, q = cli
    assert checkrun.main(["--list"]) == 0
    listed = capsys.readouterr().out
    for case in checkrun.CASES:
        assert f"\n{case.name}  (" in listed
    assert "sets optimizer: \"normuon\"" in listed and "needs a regularisation folder" in listed
    assert checkrun.main(["list", "--family", "krea2", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == len(checkrun.CASES)
    assert checkrun.main(["queue", "--config", str(config), "--dry-run", "--vram", "16", "--reg-dir",
                          str(tmp_path / "reg"), "--tag", "dry"]) == 0
    out = capsys.readouterr().out
    assert "dry run: nothing is queued" in out and "12 of 14 cases" in out
    assert out.count(": job check_dry_") >= 8  # a plan for at least eight cases
    assert "optimizer: \"normuon\"  (was \"adamw\")" in out and "steps: 30  (was 2000)" in out
    assert "block_swap: 12  (was 0)" in out and "h3_image_likeness: SKIPPED" in out
    assert q.snapshot()["waiting"] == [] and not (tmp_path / "manifests").exists()
    # the plan as data
    assert checkrun.main(["queue", "--config", str(config), "--dry-run", "--only", "normuon", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [e["name"] for e in data["cases"]] == ["baseline", "normuon"] and data["vram_gb"] is None


def test_queue_then_report(cli, tmp_path, capsys):
    config, q = cli
    assert checkrun.main(["queue", "--config", str(config), "--only", "normuon,oft_scaled_lr", "--tag", "go",
                          "--steps", "24", "--set", "learning_rate=2e-4", "--resume"]) == 0
    out = capsys.readouterr().out
    assert "3 of 3 cases" in out and "rulztrainer.checkrun report go" in out
    waiting = q.snapshot()["waiting"]
    assert [e["job"] for e in waiting] == ["check_go_baseline", "check_go_normuon", "check_go_oft_scaled_lr"]
    assert all(e["config"]["steps"] == 24 and e["config"]["learning_rate"] == 2e-4 for e in waiting)
    manifest, path = checkrun.load_manifest("go")
    assert path == tmp_path / "manifests" / "go" / "manifest.json" and manifest["base_config"] == str(config)
    assert checkrun.load_manifest()[1] == path and checkrun.load_manifest(str(path.parent))[1] == path
    # a tag that is taken is refused: its manifest, and with it the first check run's report, would be written over
    assert checkrun.main(["queue", "--config", str(config), "--only", "normuon", "--skip", "baseline", "--tag", "go"]) == 1
    assert "is taken by an earlier check run" in capsys.readouterr().err and len(q.snapshot()["waiting"]) == 3
    assert checkrun.main(["queue", "--config", str(config), "--only", "normuon", "--tag", "go", "--dry-run"]) == 0
    assert "is taken by an earlier check run" in capsys.readouterr().err

    # nothing has run: the report says so, with its own exit status
    manifest, _ = checkrun.load_manifest("go")
    assert checkrun.main(["report", "go"]) == 2
    assert "not started" in capsys.readouterr().out
    jobs = tmp_path / "out"
    for entry in manifest["cases"]:
        make_job(jobs, entry)
    assert checkrun.main(["report", "go"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Check run go (krea2") and "3 pass" in out
    assert (path.parent / "report.md").is_file() and (path.parent / "report.json").is_file()
    make_job(jobs, entry_of(manifest, "normuon"), lines=[])
    assert checkrun.main(["report", "go", "--out", str(tmp_path / "elsewhere"), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["counts"] == {"pass": 2, "fail": 1}
    assert (tmp_path / "elsewhere" / "report.md").is_file()


def test_refusals(cli, tmp_path, capsys):
    config, q = cli
    assert checkrun.main(["report", "never"]) == 1
    assert checkrun.main(["queue", "--config", str(config), "--only", "nope"]) == 1
    assert checkrun.main(["queue", "--config", str(config), "--set", "novalue"]) == 1
    # only cases that cannot run here: nothing is queued
    assert checkrun.main(["queue", "--config", str(config), "--only", "h3_image_likeness"]) == 1
    # a dataset folder that is not on this computer: every run would fail to start and pause the queue
    assert checkrun.main(["queue", "--config", str(config), "--set", f"dataset_dir={json.dumps(str(tmp_path / 'gone'))}",
                          "--only", "baseline"]) == 1
    assert "Dataset folder not found" in capsys.readouterr().err
    assert q.snapshot()["waiting"] == [] and not (tmp_path / "manifests").exists()
