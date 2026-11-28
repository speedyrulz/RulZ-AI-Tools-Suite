"""rulztrainer/measured.py and the wizard's use of it (webui.measured_swap): the block swap plan from the measured
memory of a comparable finished run, the estimate where there is none. Faked job folders, status files and logs only:
no model, no GPU. (conftest.py points measured.RECORD_FILE at a temporary file for every test.)"""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import measured as M  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402

CARD = 16311 / 1024  # a 16 GB card as nvidia-smi reports it (MiB), in the wizard's unit: 15.93
GPU16, GPU24 = [{"total": 16311}], [{"total": 24564}]
PER_BLOCK = 13.5 / 28  # Krea 2 int8: the estimate's own figures (webui.MODEL_GB, 28 blocks), GB of 10^9 bytes
BLOCK = PER_BLOCK * 1e9 / 2 ** 30  # the same block in the card's unit: 0.449 GiB
KEEP_FREE = 2.0  # on that card: the trainer's 1.5 GB margin and the 0.5 GB spare
ESTIMATE = "the model and a step's memory do not fit 16 GB: 7 blocks' frozen weights (3.4 GB) wait in system RAM and " \
           "stream in as each block runs"


def krea(**kw) -> TrainConfig:
    return TrainConfig(model_arch="krea2", **kw)


def figures(step=11.8, kept=6, cands=20, per_block=0.45, total=15.93, **kw) -> dict:
    """What a run's start-of-run calibration leaves in status.json (checkpoint_memory) for its GPU."""
    return {"cuda:0": {"kept": kept, "candidates": cands, "step_gb": step, "alloc_gb": round(step - 0.3, 3),
                       "per_block_gb": per_block,
                       "peak_gb": round(step + max(0, kept - 1) * per_block, 3), "budget_gb": round(total - 1.5, 3),
                       "total_gb": total, **kw}}


def status(**kw) -> dict:
    """What a finished run leaves in status.json, as far as the record reads it: a step that took 11.8 GiB with one
    block kept, 6 blocks kept in spare memory, and a peak of 14.6 GB (13.6 GiB) for the whole run."""
    st = {"stage": "finished", "done": True, "error": None, "step": 2000, "total": 2000,
          "peak_mem": {"cuda:0": 14.6}, "run_id": "run-1", "checkpoint_memory": figures(), **kw}
    fig = st["checkpoint_memory"]
    if "peak_mem" not in kw and isinstance(fig, dict) and "peak_gb" in (fig.get("cuda:0") if isinstance(
            fig.get("cuda:0"), dict) else {}):
        # other figures: a peak that agrees with them (0.45 GiB under what the calibration expected)
        st["peak_mem"] = {"cuda:0": round((fig["cuda:0"]["peak_gb"] - 0.45) / M.GIB_PER_GB, 2)}
    return st


def row(cfg: TrainConfig | None = None, adapter_gb: float = 1.0, **st) -> dict:
    """A counted row for `cfg` (Krea 2, 8 blocks swapped by default) without counting its adapter (no model class)."""
    cfg = cfg or krea(job_name="krea_a", block_swap=8)
    made, why = M.measurement(cfg, status(**st), folder="output/" + cfg.job_name, adapter_gb=adapter_gb, when=1000.0)
    assert made is not None, why
    return made


def want(cfg: TrainConfig | None = None, quant: str = "int8", **over) -> dict:
    return {**M.settings(cfg or krea(), over), "quant": quant}


def find(rows, w=None, vram=CARD, adapter_gb=1.0, **kw) -> dict:
    kw = {"per_block_gb": PER_BLOCK, "max_swap": 26, "blocks": 28, "quant_of": lambda r: "int8", **kw}
    return M.find(rows, w or want(), vram=vram, adapter_gb=adapter_gb, **kw)  # type: ignore[return-value]


def job_folder(output: Path, cfg: TrainConfig, st: dict | None, log: str = "") -> Path:
    job = output / cfg.job_name
    job.mkdir(parents=True)
    (job / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    if st is not None:
        (job / "status.json").write_text(json.dumps(st), encoding="utf-8")
    if log:
        (job / "train.log").write_text(log, encoding="utf-8")
    return job


def log_line(kept=6, cands=20, per_block=0.45, peak=14.1, shown=16, budget=14.4, blocks=28) -> str:
    return (f"12:00:01 INFO Checkpointing: {kept} of {blocks} blocks keep their activations, measured at 4608 tokens "
            f"(cuda:0: {kept} of {cands} ({per_block:.2f} GB per extra block, peak {peak:.1f} of {shown} GB, budget "
            f"{budget:.1f}))\n")


# ---------------------------------------------------------------- which runs count, and what is measured

def test_a_finished_run_gives_a_row_with_what_decides_memory():
    cfg = krea(job_name="krea_a", block_swap=8, batch_size=2, resolutions=[0.5, 1.5], network_type="lokr",
               lokr_factor=4, optimizer="keel", lora_ema=0.0, id_loss_weight=0.1)
    r = row(cfg, adapter_gb=2.5)
    assert (r["family"], r["transformer_quant"], r["model_source"]) == ("krea2", "auto", "files")
    assert (r["side"], r["megapixels"]) == (1254, 1.5)  # the largest resolution it trained at
    assert (r["batch_size"], r["gradient_checkpointing"], r["block_swap"], r["frames"]) == (2, True, 8, 1)
    assert (r["network_type"], r["lokr_factor"], r["optimizer"], r["weight_average"]) == ("lokr", 4, "keel", False)
    assert r["extras"]["id_loss"] is True and r["extras"]["isolation"] is False
    assert (r["step_gb"], r["step_from"], r["kept_blocks"], r["kept_block_gb"]) == (11.8, "calibration", 6, 0.45)
    assert (r["peak_gb"], r["peak_mem"], r["card_gb"]) == (14.6, {"cuda:0": 14.6}, 15.93)
    assert (r["steps"], r["planned_steps"], r["job"], r["run_id"], r["adapter_gb"]) == (2000, 2000, "krea_a", "run-1", 2.5)
    json.dumps(r)  # plain values only: it goes into a JSON file


def test_the_step_is_the_calibration_s_and_the_run_s_excess_is_added():
    # the run's peak (13.6 GiB allocated) is under what the calibration's own allocated peak gives with 6 blocks kept
    # (11.5 + 5 x 0.45 = 13.75): the step stands as measured (11.8 reserved), however full the card got
    assert row()["step_gb"] == 11.8
    # a run that went over it (the identity loss's decode, samples, the optimizer's state): the excess counts
    over = row(peak_mem={"cuda:0": 16.0})  # 14.90 GiB, 1.15 over
    assert over["step_gb"] == pytest.approx(11.8 + 16.0 * M.GIB_PER_GB - 13.75, abs=1e-3)
    # the calibration's last pass keeps 3 blocks and stays in the allocator's peak: a run that kept 1 is held
    # against that (11.5 + 2 x 0.45 = 12.4), not against its own one-block figure
    tight = row(checkpoint_memory=figures(kept=1), peak_mem={"cuda:0": 13.2})  # 12.29 GiB
    assert tight["step_gb"] == 11.8 and tight["kept_blocks"] == 1
    assert row(checkpoint_memory=figures(kept=0), peak_mem={"cuda:0": 14.0})["step_gb"] == pytest.approx(
        11.8 + 14.0 * M.GIB_PER_GB - 12.4, abs=1e-3)
    # like with like: the peak is allocated memory, so it is held against the calibration's allocated figure. Held
    # against the reserved one (all a run from before 3.21 has, or a log) an excess under the difference would hide
    old = figures()
    del old["cuda:0"]["alloc_gb"]
    assert row(peak_mem={"cuda:0": 15.0})["step_gb"] == pytest.approx(11.8 + 15.0 * M.GIB_PER_GB - 13.75, abs=1e-3)
    # (13.97 GiB is under the reserved 14.05: nothing shows, so UNSEEN_GB is added for what may hide there)
    assert row(checkpoint_memory=old, peak_mem={"cuda:0": 15.0})["step_gb"] == 11.8 + M.UNSEEN_GB
    # the device named differently in the two places ("cuda" for the run, "cuda:0" in the calibration)
    assert row(peak_mem={"cuda": 14.6})["step_gb"] == 11.8


def test_a_run_that_did_not_fill_the_card_is_measured_by_its_peak():
    fixed = row(krea(job_name="fixed", block_swap=8, checkpoint_keep_blocks=0), checkpoint_memory=None,
                peak_mem={"cuda:0": 13.2})
    assert fixed["step_from"] == "peak" and fixed["step_gb"] == pytest.approx(13.2 * M.GIB_PER_GB, abs=1e-3)
    assert "kept_blocks" not in fixed and fixed["card_gb"] is None
    plain = row(krea(job_name="plain", gradient_checkpointing=False), checkpoint_memory=None)
    assert plain["step_from"] == "peak"


def test_a_run_on_auto_without_calibration_figures_does_not_count():
    made, why = M.measurement(krea(), status(checkpoint_memory=None), adapter_gb=1.0)
    assert made is None and "its peak follows the card" in why and "no train.log read" in why
    made, why = M.measurement(krea(), status(checkpoint_memory=None), adapter_gb=1.0, log_text="12:00 INFO nothing\n")
    assert made is None and "left no calibration figures" in why and "no train.log read" not in why
    for junk in ({}, {"cuda:0": {}}, {"cuda:0": {"step_gb": "x"}}, {"cuda:0": {"step_gb": 0}}, [1], "x",
                 {"cuda:0": {"step_gb": 12, "kept": "bad"}}, {"cuda:0": {"step_gb": 12, "per_block_gb": [1]}},
                 {"cuda:0": {"step_gb": 12, "kept": -1}}, {"cuda:0": 5}):
        made, why = M.measurement(krea(), status(checkpoint_memory=junk), adapter_gb=1.0)  # never raises
        assert made is None and "left no calibration figures" in why


def test_an_older_run_s_figures_are_read_from_its_log():
    text = ("11:59:00 INFO Loading\n" + log_line(kept=2, peak=13.0) + "12:30:00 INFO step 100\n"
            + log_line() + "13:00:00 INFO Done.\n")  # two runs wrote this log: the later one's line counts
    fig = M.log_calibration(text)["cuda:0"]
    assert (fig["kept"], fig["candidates"], fig["per_block_gb"], fig["peak_gb"]) == (6, 20, 0.45, 14.1)
    assert fig["step_gb"] == pytest.approx(14.1 - 5 * 0.45) and fig["total_gb"] == 16.0  # the whole GB it prints
    made, _ = M.measurement(krea(job_name="old", block_swap=8), status(checkpoint_memory=None), adapter_gb=1.0,
                            log_text=text)
    # the log's step, plus UNSEEN_GB: a log has no allocated figure to hold the run's peak against
    assert (made["step_from"], made["step_gb"], made["kept_blocks"], made["card_gb"]) == ("log", 12.35, 6, 16.0)
    assert M.log_calibration(log_line(shown=24, budget=22.1, peak=21.9))["cuda:0"]["total_gb"] == 24.0
    two = log_line().rstrip()[:-1] + "; cuda:1: 2 of 8 (0.40 GB per extra block, peak 13.2 of 16 GB, budget 14.4))\n"
    assert M.log_calibration(two)["cuda:1"]["step_gb"] == pytest.approx(12.8) and len(M.log_calibration(two)) == 2
    # no block kept: the printed peak is the step itself
    assert M.log_calibration(log_line(kept=0, peak=14.9))["cuda:0"]["step_gb"] == 14.9
    assert M.log_calibration("12:00 INFO Checkpointing: 3 of 28 blocks keep their activations "
                             "(checkpoint_keep_blocks)\n") == {}
    assert M.log_calibration("") == {}


def test_the_status_figures_are_the_calibration_s_in_gib():
    gib = 2 ** 30
    fig = M.calibration(6, 20, 11.8 * gib, 0.45 * gib, 14.43 * gib, 15.93 * gib, 11.5 * gib)
    assert fig == {"kept": 6, "candidates": 20, "step_gb": 11.8, "alloc_gb": 11.5, "per_block_gb": 0.45,
                   "peak_gb": 14.05, "budget_gb": 14.43, "total_gb": 15.93}
    assert M.calibration(6, 20, 11.8 * gib, 0.45 * gib, 14.43 * gib, 15.93 * gib)["alloc_gb"] is None
    assert M.calibration(0, 20, 14.9 * gib, 0.45 * gib, 14.43 * gib, 15.93 * gib)["peak_gb"] == 14.9
    json.dumps(fig)


def test_the_margin_is_the_trainer_s_own():
    ckpt = pytest.importorskip("rulztrainer.checkpointing")
    for total in (8.0, 11.9, 15.93, 18.75, 23.99, 31.84, 47.5, 79.3):
        for id_loss in (False, True):
            assert M.margin_gb(total, id_loss) == pytest.approx(
                ckpt.memory_margin(total * ckpt.GB, identity_loss=id_loss) / ckpt.GB)


def test_the_card_is_the_one_given_then_the_status_s_then_the_calibration_s():
    assert M.measurement(krea(), status(gpu_total_gb={"cuda:0": 23.99}), adapter_gb=1.0)[0]["card_gb"] == 23.99
    assert M.measurement(krea(), status(gpu_total_gb={"cuda:0": 23.99}), adapter_gb=1.0,
                         cards={"cuda:0": 31.84})[0]["card_gb"] == 31.84
    assert M.measurement(krea(), status(), adapter_gb=1.0)[0]["card_gb"] == 15.93
    bare = figures()
    del bare["cuda:0"]["total_gb"]
    assert M.measurement(krea(), status(checkpoint_memory=bare), adapter_gb=1.0)[0]["card_gb"] is None


def test_a_video_family_keeps_its_clip_length_and_a_still_family_does_not():
    clip = M.settings(TrainConfig(model_arch="pan2", train_frames=33, video_resolution=512))
    assert clip["frames"] == 33 and clip["video_side"] == 512  # PAN 2 clips are 4k + 1 frames
    assert M.settings(krea(train_frames=33))["frames"] == 1  # no clips on an image model: the setting plays no part


@pytest.mark.parametrize("st, why", [
    ({"error": "OutOfMemoryError: CUDA out of memory", "stage": "error"}, "failed"),
    ({"done": False, "stage": "training"}, "did not finish"),
    ({"stage": "error"}, "did not finish"),
    ({"step": 99}, "only 99 steps"),
    ({"step": 0, "soup": {"files": []}}, "only 0 steps"),  # a seed soup or noise check trains nothing
    ({"peak_mem": {}}, "no GPU peak"),
    ({"peak_mem": None}, "no GPU peak"),
])
def test_runs_that_failed_ended_early_or_were_short_do_not_count(st, why):
    made, reason = M.measurement(krea(), status(**st), adapter_gb=1.0)
    assert made is None and why in reason


def test_a_step_up_run_counts_only_from_its_highest_resolution():
    cfg = krea(resolutions=[0.5, 1.0], resolution_step_up=True)
    made, reason = M.measurement(cfg, status(res_stage=0, res_stages=2), adapter_gb=1.0)
    assert made is None and "highest resolution" in reason
    assert M.measurement(cfg, status(res_stage=1, res_stages=2), adapter_gb=1.0)[0] is not None
    # a status that does not say how far it got proves nothing: left out, as is one that names fewer stages than
    # the config trains
    for st in ({}, {"res_stage": None, "res_stages": None}, {"res_stage": "x"}, {"res_stages": 2},
               {"res_stage": 0, "res_stages": 1}, {"res_stage": 0}):
        made, reason = M.measurement(cfg, status(**st), adapter_gb=1.0)
        assert made is None and "resolution" in reason, st
    assert M.measurement(cfg, status(res_stage=1), adapter_gb=1.0)[0] is not None
    # step-up with one resolution has no stages to reach
    assert M.measurement(krea(resolution_step_up=True), status(), adapter_gb=1.0)[0] is not None
    # a run stopped by hand after MIN_STEPS has met its full-size steps: it counts
    assert M.measurement(krea(), status(step=M.MIN_STEPS, total=2000), adapter_gb=1.0)[0] is not None
    assert M.measurement(krea(), None)[0] is None and M.measurement(krea(), [])[0] is None


# ---------------------------------------------------------------- the record file

def test_the_record_round_trips_and_a_run_replaces_its_own_row(tmp_path):
    path = tmp_path / "sub" / "record.json"
    assert M.load(path) == []
    M.add(row(), path)
    M.add(row(krea(job_name="krea_b"), run_id="run-2"), path)
    M.add(row(peak_mem={"cuda:0": 15.0}), path)  # run-1 again (carried on): its new row takes the old one's place
    rows = M.load(path)
    assert sorted((r["run_id"], r["peak_gb"]) for r in rows) == [("run-1", 15.0), ("run-2", 14.6)]
    assert [p.name for p in path.parent.iterdir()] == ["record.json"]  # written whole: no temporary or lock file left
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == M.VERSION


def test_a_broken_record_reads_as_none_and_is_written_over(tmp_path):
    path = tmp_path / "record.json"
    for junk in ("{not json", "[]", '{"runs": 3}', '{"runs": [3, {"family": "krea2"}, {"peak_gb": 5, "step_gb": 4}, '
                 '{"family": "krea2", "peak_gb": 5}, {"family": "krea2", "peak_gb": 5, "step_gb": "x"}]}'):
        path.write_text(junk, encoding="utf-8")
        assert M.load(path) == []
    M.add(row(), path)
    assert len(M.load(path)) == 1


def test_runs_without_an_id_are_told_apart_by_folder_and_the_oldest_rows_go(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    a, b = row(run_id=""), row(krea(job_name="krea_b"), run_id="")
    M.add(a, path)
    M.add(b, path)
    M.add(dict(a, peak_gb=9.0), path)
    assert sorted(r["peak_gb"] for r in M.load(path)) == [9.0, 14.6]
    monkeypatch.setattr(M, "MAX_ROWS", 3)
    for i in range(5):
        M.add(dict(a, run_id=f"r{i}", finished=2000.0 + i), path)
    assert [r["run_id"] for r in M.load(path)] == ["r2", "r3", "r4"]


def test_a_stale_lock_does_not_stop_a_write(tmp_path):
    import os

    path = tmp_path / "record.json"
    lock = tmp_path / "record.json.lock"
    lock.write_text("", encoding="utf-8")
    os.utime(lock, (1000, 1000))  # a writer that was killed long ago
    M.add(row(), path)
    assert len(M.load(path)) == 1 and not lock.exists()


# ---------------------------------------------------------------- the adjustment arithmetic

def test_the_same_settings_on_the_same_card_give_the_fewest_blocks_that_leave_room():
    # a step took 11.8 GiB with 8 blocks swapped: 4.13 GiB of the card left. Each block fewer adds 0.449 GiB: 4 fewer
    # leave 2.33 (over the 2.0 kept free), 5 fewer would leave 1.88
    plan = find([row()])
    assert plan["at_blocks"] == 8 and plan["swap"] == 4 and plan["keep_free_gb"] == KEEP_FREE
    assert plan["need_gb"] == pytest.approx(11.8 + 4 * BLOCK)
    assert plan["free_gb"] == pytest.approx(CARD - 11.8 - 4 * BLOCK) and plan["free_gb"] >= KEEP_FREE
    assert CARD - (11.8 + 5 * BLOCK) < KEEP_FREE  # one block fewer would leave too little
    assert plan["saved_gb"] == pytest.approx(4 * BLOCK) and plan["adjust_gb"] == pytest.approx(4 * BLOCK)
    assert plan["adapter_delta_gb"] == 0.0 and not plan["tight"] and not plan["close_call"]
    assert (plan["row"]["job"], plan["step_gb"], plan["from"], plan["kept_blocks"]) == ("krea_a", 11.8, "calibration", 6)
    assert plan["run_peak_gb"] == pytest.approx(14.6 * M.GIB_PER_GB)


def test_a_smaller_card_swaps_more_and_a_bigger_adapter_too():
    base = find([row()])["swap"]
    small = find([row()], vram=12.0)  # 11.8 + 2.0 - 12 = 1.8 GiB over at 8 blocks: 4.01 blocks of 0.449, so 5 more
    assert small["swap"] == 13 and small["need_gb"] == pytest.approx(11.8 - 5 * BLOCK)
    bigger = find([row()], adapter_gb=2.2)  # the form's adapter counts 1.2 GB (1.12 GiB) more than the run's
    assert bigger["adapter_delta_gb"] == pytest.approx(1.2 * M.GIB_PER_GB)
    assert bigger["swap"] == base + 2  # 1.12 GiB is 2.49 blocks, and 4 blocks had 0.33 GiB to spare: 2 more
    assert bigger["adjust_gb"] == pytest.approx(1.2 * M.GIB_PER_GB + 2 * BLOCK)  # 2 fewer than the run's 8, plus it
    smaller = find([row(adapter_gb=1.6)], adapter_gb=1.0)  # 0.6 GB less: 1.24 blocks, one block fewer fits
    assert smaller["adapter_delta_gb"] == pytest.approx(-0.6 * M.GIB_PER_GB) and smaller["swap"] == base - 1
    # a bigger card keeps more free (8%), and the identity loss 1.5 GB more
    assert find([row()], vram=24.0)["keep_free_gb"] == pytest.approx(0.08 * 24 + 0.5)
    with_id = krea(job_name="id", block_swap=8, id_loss_weight=0.1)
    plan = find([row(with_id)], want(with_id))
    assert plan["keep_free_gb"] == KEEP_FREE + 1.5 and plan["swap"] == base + 3


def test_the_count_stays_within_the_blocks_and_says_when_even_that_is_tight():
    none = find([row(krea(job_name="k", block_swap=2), checkpoint_memory=figures(step=10.0))])
    assert none["swap"] == 0 and none["need_gb"] == pytest.approx(10.0 + 2 * BLOCK) and not none["tight"]
    tight = find([row(krea(job_name="k", block_swap=24), checkpoint_memory=figures(step=15.5, kept=0),
                      peak_mem={"cuda:0": 16.0})])
    assert tight["swap"] == 26 and tight["tight"]  # 4 more would be needed, 2 more is all there is
    # a run with more blocks than the model has swapped all but one, as the trainer does
    assert find([row(krea(job_name="k", block_swap=99))])["at_blocks"] == 27


def test_a_run_that_needs_too_much_adjusting_is_not_used():
    far = row(krea(job_name="k", block_swap=12), checkpoint_memory=figures(step=5.0))  # no swap needed: 12 blocks off
    assert find([far]) is None
    assert find([row()])["adjust_gb"] <= M.CLOSE_GB
    assert find([row()], adapter_gb=1.0 + M.CLOSE_GB + 0.5) is None  # an adapter that much larger: the estimate's job


def test_a_run_close_to_its_card_is_called_out():
    close = find([row(peak_mem={"cuda:0": 16.3})])  # 15.18 GiB on a 15.93 GiB card
    assert close["close_call"] and close["headroom_gb"] == pytest.approx(15.93 - 16.3 * M.GIB_PER_GB)
    assert close["step_gb"] > 11.8  # what it took over the calibration's expectation counts
    assert not find([row()])["close_call"]  # 13.6 GiB: the trainer's own margin, as every run on auto
    # a run from before the card was recorded is held against the card it is planned for
    bare = figures()
    del bare["cuda:0"]["total_gb"]
    unknown = find([row(checkpoint_memory=bare, peak_mem={"cuda:0": 16.3})])
    assert unknown["card_gb"] is None and unknown["close_call"]
    # a run on a bigger card that left room there is not a close call, whatever the card planned for
    roomy = row(krea(job_name="big", block_swap=2), checkpoint_memory=figures(total=23.99, kept=18),
                peak_mem={"cuda:0": 21.0})
    assert not find([roomy])["close_call"] and find([roomy])["headroom_gb"] > 4


def test_an_adapter_that_cannot_be_counted_must_be_the_same_adapter():
    r = row(adapter_gb=0.0)
    assert find([r], adapter_gb=0.0)["adapter_delta_gb"] == 0.0
    assert find([r], adapter_gb=1.0)["adapter_delta_gb"] == 0.0  # the same rank-16 LoRA, adamw, average on
    assert find([r], want(krea(lora_rank=64)), adapter_gb=0.0) is None
    assert find([r], want(krea(optimizer="adamw8bit")), adapter_gb=0.0) is None


# ---------------------------------------------------------------- which rows are comparable

@pytest.mark.parametrize("other", [
    {"resolutions": [1.5]}, {"batch_size": 2}, {"gradient_checkpointing": False}, {"id_loss_weight": 0.1},
    {"isolation": True, "trigger_word": "tok"}, {"activation_offload": True}, {"checkpoint_save_attention": True},
    {"reward_steps": 50},
    {"multi_gpu": "model_split"}, {"finetune": True}, {"model_arch": "qwen_image21"},
])
def test_a_run_with_other_settings_is_not_comparable(other):
    cfg = TrainConfig(**{"model_arch": "krea2", "job_name": "other", "block_swap": 8, **other})
    assert find([row(cfg)]) is None


def test_the_quantization_must_match_and_an_unknown_one_does_not():
    assert find([row()], quant_of=lambda r: "nvfp4") is None
    assert find([row()], quant_of=lambda r: None) is None
    explicit = row(krea(job_name="k", block_swap=8, transformer_quant="int8"))
    assert M.find([explicit], want(), vram=CARD, per_block_gb=PER_BLOCK, max_swap=26, blocks=28,
                  adapter_gb=1.0) is not None  # no resolver: the setting as the run had it
    assert M.find([explicit], want(quant="bf16"), vram=CARD, per_block_gb=PER_BLOCK, max_swap=26, blocks=28,
                  adapter_gb=1.0) is None


def test_clips_must_be_the_same_length_and_size():
    pan = lambda **kw: TrainConfig(model_arch="pan2", job_name="clip", block_swap=8, **kw)  # noqa: E731
    made = row(pan(train_frames=33, video_resolution=512))
    assert find([made], want(pan(train_frames=33, video_resolution=512))) is not None
    assert find([made], want(pan(train_frames=49, video_resolution=512))) is None
    assert find([made], want(pan(train_frames=33, video_resolution=384))) is None
    assert find([made], want(pan())) is None  # stills against clips
    still = row(pan(video_resolution=384))
    assert find([still], want(pan(video_resolution=512))) is not None  # no clips: their size plays no part


def test_the_nearest_adapter_is_used_then_the_run_that_needs_most_then_the_newer():
    same = row(krea(job_name="same"), run_id="a", checkpoint_memory=figures(step=11.0))
    other = row(krea(job_name="other"), run_id="b", adapter_gb=1.5, checkpoint_memory=figures(step=13.0))
    assert find([other, same])["row"]["job"] == "same" and find([same, other])["row"]["job"] == "same"
    # two runs of the very same settings that disagree (their samples, their probes): the one that needs more
    low = row(krea(job_name="low", block_swap=4), run_id="c", checkpoint_memory=figures(step=12.0))
    high = row(krea(job_name="high", block_swap=8), run_id="d", checkpoint_memory=figures(step=11.5))
    assert 12.0 + 4 * BLOCK < 11.5 + 8 * BLOCK  # with no block swapped, "high" needs more
    assert find([low, high])["row"]["job"] == "high" and find([high, low])["row"]["job"] == "high"
    assert find([low, high])["swap"] > find([low])["swap"]
    old, new = dict(low, job="old", run_id="e", finished=1.0), dict(low, job="new", run_id="f", finished=2.0)
    assert find([old, new])["row"]["job"] == "new" and find([new, old])["row"]["job"] == "new"
    assert find([{"family": "krea2"}, {"peak_gb": "x", "family": "krea2", "step_gb": 3}, same])["row"]["job"] == "same"


# ---------------------------------------------------------------- the wizard

def wizard(cfg, gpus=GPU16, goal="concept"):
    import webui

    return webui.wizard_recommendations(cfg, goal, gpus, None)


def record(cfg: TrainConfig, **st) -> dict:
    """A finished run of `cfg` into the record the wizard reads, its adapter counted as the wizard counts the form's."""
    made, why = M.measurement(cfg, status(**st), folder="output/" + cfg.job_name)
    assert made is not None, why
    M.add(made)
    return made


CASES = [
    (dict(model_arch="krea2"), GPU16), (dict(model_arch="krea2"), GPU24),
    (dict(model_arch="krea2", network_type="lokr", lokr_factor=8), [{"total": 18800}]),
    (dict(model_arch="krea2", lora_rank=64, optimizer="adamw"), [{"total": 12288}]),
    (dict(model_arch="moss_image"), [{"total": 8192}]), (dict(model_arch="qwen_image21"), GPU16),
    (dict(model_arch="flux2_klein", model_source="hf"), GPU16), (dict(model_arch="ideogram4"), [{"total": 11264}]),
    (dict(model_arch="minimax_h3"), GPU16), (dict(model_arch="p_h3_image"), GPU24),
    (dict(model_arch="minimax_h3_image", network_type="lokr"), GPU16), (dict(model_arch="pan2"), [{"total": 32607}]),
    (dict(model_arch="krea2"), [{"total": 16311}, {"total": 16380}]), (dict(model_arch="krea2", finetune=True), GPU16),
    (dict(model_arch="krea2"), []),
]


@pytest.mark.parametrize("kw, gpus", CASES)
def test_without_a_record_the_wizard_is_what_it_was(kw, gpus, monkeypatch):
    import webui

    assert not M.RECORD_FILE.exists()
    cfg = TrainConfig(**kw)
    now = wizard(cfg, gpus)
    M.save([])  # an empty record is no record either
    assert wizard(cfg, gpus) == now
    # the plan as it was before the lookup existed: the same code with the lookup taken out
    monkeypatch.setattr(webui, "measured_swap", lambda *a, **k: (None, False))
    before = wizard(cfg, gpus)
    assert now == before and list(now) == list(before)
    assert "measured" not in str(now.get("block_swap")) and "no comparable" not in str(now.get("block_swap"))


def test_the_estimate_s_own_words_are_unchanged():
    assert wizard(krea())["block_swap"] == (7, ESTIMATE)  # 13.5 + 4.5 - 15.93 = 2.07 GB over: 5 of 28 blocks, plus the 2 in flight
    assert wizard(krea(), GPU24)["block_swap"] == (0, "the model fits 24 GB as it is; swapping only slows the steps")


def test_the_quantization_the_plan_counts_is_what_it_always_was():
    import webui

    q = webui.plan_quant_of
    assert q("krea2", "files", "int8_convrot", 16, "auto") == "int8" and q("krea2", "files", "bf16", 16, "auto") == "bf16"
    assert q("ideogram4", "files", "nvfp4_mixed", 16, "auto") == "nvfp4"
    assert (q("krea2", "hf", "", 24, "auto"), q("krea2", "hf", "", 16, "auto")) == ("int8", "nvfp4")
    assert q("flux2_klein", "files", "int8_convrot", 16, "auto") == "nvfp4"  # its float files are quantized as loaded
    assert q("moss_image", "hf", "", 16, "auto") == "bf16" and q("krea2", "hf", "", None, "auto") is None
    assert q("krea2", "hf", "", None, "int8_convrot") == "int8" and q("krea2", "files", "x", 16, "none") == "bf16"
    assert webui.plan_quant(krea(), 16.0, "auto") == "int8"


def test_the_wizard_plans_from_a_comparable_run_and_says_so():
    record(krea(job_name="krea_a", block_swap=8))
    swap, reason = wizard(krea())["block_swap"]
    assert swap == 4  # the estimate says 5
    assert reason == ("measured on your run 'krea_a': a step took 11.8 GB at 8 blocks swapped (its 13.6 GB peak "
                      "includes the 6 blocks it kept in spare memory); adjusted for 4 fewer blocks swapped (at the "
                      "estimate's size), 4 blocks' frozen weights (1.8 GB) wait in system RAM and stream in as each "
                      "block runs, leaving a step about 2.3 GB of 16 GB free")
    # a bigger adapter on the form: the run's step plus the difference the estimate counts
    lokr_swap, lokr_reason = wizard(krea(network_type="lokr", lokr_factor=8))["block_swap"]
    assert lokr_swap > swap and "this adapter's" in lokr_reason and "GB more (counted)" in lokr_reason
    # a 24 GB card: the same run says it fits as it is
    fits, fits_reason = wizard(krea(), GPU24)["block_swap"]
    assert fits == 0 and "measured on your run 'krea_a'" in fits_reason
    assert "it fits 24 GB as it is, a step leaving about 8.6 GB of 24 GB free; swapping only slows" in fits_reason
    # the very settings the run had: nothing to adjust
    M.save([])
    record(krea(job_name="exact", block_swap=4), checkpoint_memory=figures(step=13.73, kept=2),
           peak_mem={"cuda:0": 15.2})
    assert wizard(krea())["block_swap"] == (
        4, "measured on your run 'exact': a step took 13.7 GB at 4 blocks swapped (its 14.2 GB peak includes the 2 "
           "blocks it kept in spare memory): 4 blocks' frozen weights (1.8 GB) wait in system RAM and stream in as "
           "each block runs, leaving a step about 2.2 GB of 16 GB free")


def test_a_run_measured_by_its_peak_reads_plainly():
    record(krea(job_name="fixed", block_swap=8, checkpoint_keep_blocks=0), checkpoint_memory=None,
           peak_mem={"cuda:0": 13.2}, gpu_total_gb={"cuda:0": 15.93})
    swap, reason = wizard(krea())["block_swap"]  # 12.29 GiB at 8 blocks: 3 fewer leave 2.19
    assert swap == 5 and reason.startswith("measured on your run 'fixed': 12.3 GB at 8 blocks swapped; adjusted for 3 "
                                           "fewer blocks swapped (at the estimate's size), 5 blocks' frozen weights")


def test_the_wizard_warns_when_the_measured_run_was_close_to_its_card():
    record(krea(job_name="close", block_swap=8), peak_mem={"cuda:0": 16.3})
    _swap, reason = wizard(krea())["block_swap"]
    assert reason.endswith("; that run came within 0.7 GB of its 16 GB card: watch the first steps")
    M.save([])
    record(krea(job_name="full", block_swap=8), peak_mem={"cuda:0": 17.5})  # over the card: it spilled
    assert "; that run filled its 16 GB card: watch the first steps" in wizard(krea())["block_swap"][1]
    M.save([])
    bare = figures()
    del bare["cuda:0"]["total_gb"]
    record(krea(job_name="unknown", block_swap=8), peak_mem={"cuda:0": 16.3}, checkpoint_memory=bare)
    assert "that run's peak is within 0.7 GB of this card's memory" in wizard(krea())["block_swap"][1]
    M.save([])
    record(krea(job_name="tight", block_swap=24), checkpoint_memory=figures(step=15.5, kept=0), peak_mem={"cuda:0": 16.0})
    swap, reason = wizard(krea())["block_swap"]
    assert swap == 26 and "even so the fit is tight" in reason


def test_with_a_record_but_no_comparable_run_the_wizard_says_it_estimated():
    record(krea(job_name="big", block_swap=8, batch_size=4))
    assert wizard(krea())["block_swap"] == (7, ESTIMATE + "; estimated (no comparable finished run of yours yet)")
    other = wizard(TrainConfig(model_arch="moss_image"), [{"total": 8192}])["block_swap"]
    assert other[1].endswith("estimated (no comparable finished run of yours yet)")
    assert "block_swap" not in wizard(krea(), [])  # no GPU found: no plan, nothing to say
    # the wizard's own picks decide what is comparable: it sets 1 MP, so a 1 MP run counts for a form still on 2 MP
    M.save([])
    record(krea(job_name="one_mp", block_swap=8))
    assert "measured on your run 'one_mp'" in wizard(krea(resolutions=[2.0]))["block_swap"][1]


def test_a_run_made_with_another_quantization_is_not_used_by_the_wizard():
    # Krea 2 quantized as it loads: auto is int8 from 22 GB, nvfp4 below
    record(krea(job_name="hf", model_source="hf"), checkpoint_memory=figures(step=19.0, total=23.99, kept=3),
           peak_mem={"cuda:0": 21.0})
    form = krea(model_source="hf")
    assert "measured on your run 'hf'" in wizard(form, GPU24)["block_swap"][1]
    assert "estimated (no comparable" in wizard(form, GPU16)["block_swap"][1]  # nvfp4 on 16 GB: another model
    # without its card, a run on `auto` cannot be placed: the estimate stands
    bare = figures(step=19.0, kept=3)
    del bare["cuda:0"]["total_gb"]
    M.save([])
    record(krea(job_name="hf", model_source="hf"), checkpoint_memory=bare, peak_mem={"cuda:0": 21.0})
    assert "estimated (no comparable" in wizard(form, GPU24)["block_swap"][1]


def test_the_h3_plan_uses_a_measured_run_with_its_own_block_size():
    estimate, words = wizard(TrainConfig(model_arch="minimax_h3"))["block_swap"]
    assert "an estimate" in words and "no comparable" not in words
    made = record(TrainConfig(model_arch="minimax_h3", job_name="h3_a", block_swap=40),
                  checkpoint_memory=figures(step=11.0, kept=4, cands=9))
    swap, reason = wizard(TrainConfig(model_arch="minimax_h3"))["block_swap"]
    # 11.0 + 2.0 - 15.93 = 2.93 GiB to spare at 40 blocks: 8 blocks of 0.387 GB (0.360 GiB) fewer
    assert swap == 32 and swap != estimate
    assert reason.startswith("measured on your run 'h3_a': a step took 11.0 GB at 40 blocks swapped")
    assert "32 blocks' frozen weights (11.5 GB) wait in system RAM" in reason
    M.save([dict(made, batch_size=2)])
    assert wizard(TrainConfig(model_arch="minimax_h3"))["block_swap"] == (
        estimate, words + "; estimated (no comparable finished run of yours yet)")


def test_a_record_the_lookup_cannot_use_leaves_the_estimate(monkeypatch):
    M.RECORD_FILE.write_text("{broken", encoding="utf-8")
    assert wizard(krea())["block_swap"] == (7, ESTIMATE)
    M.add(row())
    monkeypatch.setattr(M, "find", lambda *a, **k: 1 / 0)
    assert wizard(krea())["block_swap"] == (7, ESTIMATE)


# ---------------------------------------------------------------- the trainer's hooks

def _trainer(tmp_path, cfg, st):
    from rulztrainer import trainer as T

    tr = T.Trainer.__new__(T.Trainer)
    tr.cfg, tr.job_dir, tr.devices = cfg, tmp_path / cfg.job_name, ["cuda:0"]
    tr.job_dir.mkdir(parents=True, exist_ok=True)
    tr.status = T.Status(tr.job_dir / "status.json")
    tr.status.update(**st)
    return tr


def test_the_calibration_s_figures_go_into_the_status(tmp_path, monkeypatch):
    gib = 2 ** 30
    tr = _trainer(tmp_path, krea(job_name="calibrated"), {})
    seen: dict = {}
    tr._note_calibration(seen, "cuda:0", 6, 20, 11.8 * gib, 0.45 * gib, 14.43 * gib, 15.93 * gib, 11.5 * gib)
    tr._note_calibration(seen, "cuda:1", 0, 8, 14.9 * gib, 0.4 * gib, 14.4 * gib, 15.9 * gib)  # a second GPU's
    saved = json.loads((tr.job_dir / "status.json").read_text(encoding="utf-8"))["checkpoint_memory"]
    assert saved["cuda:0"] == figures()["cuda:0"] and saved["cuda:1"]["kept"] == 0 and len(saved) == 2
    monkeypatch.setattr(M, "calibration", lambda *a: 1 / 0)
    tr._note_calibration(seen, "cuda:0", 6, 20, 1.0, 1.0, 1.0, 1.0)  # never fails the calibration


def test_a_finished_run_records_its_step_and_its_card(tmp_path, monkeypatch):
    import types

    from rulztrainer import trainer as T

    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(T.torch.cuda, "get_device_properties",
                        lambda d: types.SimpleNamespace(total_memory=int(15.93 * 2 ** 30)))
    tr = _trainer(tmp_path, krea(job_name="hooked", block_swap=6), status())
    tr._record_memory()
    (kept,) = M.load()
    assert (kept["job"], kept["block_swap"], kept["step_gb"], kept["peak_gb"], kept["card_gb"]) == (
        "hooked", 6, 11.8, 14.6, 15.93)
    assert kept["adapter_gb"] > 0 and kept["folder"] == str(tr.job_dir)
    saved = json.loads((tr.job_dir / "status.json").read_text(encoding="utf-8"))
    assert saved["gpu_total_gb"] == {"cuda:0": 15.93}  # so the record can be rebuilt from the folder


def test_a_stopped_short_run_records_nothing_and_a_failing_record_never_fails_the_run(tmp_path, monkeypatch):
    from rulztrainer import trainer as T

    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: False)
    _trainer(tmp_path, krea(job_name="short"), status(step=40))._record_memory()
    assert M.load() == []
    tr = _trainer(tmp_path, krea(job_name="fine"), status())
    monkeypatch.setattr(M, "add", lambda *a, **k: 1 / 0)
    tr._record_memory()  # no exception
    assert M.load() == []


# ---------------------------------------------------------------- the rebuild command

def _past_runs(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    job_folder(out, krea(job_name="good", block_swap=8), status(run_id="good-1"))
    job_folder(out, krea(job_name="logged", block_swap=6), status(run_id="log-1", checkpoint_memory=None),
               log="11:00:00 INFO Loading\n" + log_line() + "12:00:00 INFO Done.\n")
    job_folder(out, krea(job_name="fixed keep", block_swap=4, checkpoint_keep_blocks=0),
               status(run_id="fk-1", checkpoint_memory=None, gpu_total_gb={"cuda:0": 23.99}, peak_mem={"cuda:0": 17.5}))
    job_folder(out, krea(job_name="no figures"), status(checkpoint_memory=None))
    job_folder(out, krea(job_name="oom"), status(error="OutOfMemoryError: CUDA out of memory", stage="error"))
    job_folder(out, krea(job_name="smoke"), status(step=20, total=20))
    job_folder(out, krea(job_name="killed"), status(done=False, stage="training", step=900))
    two = {**figures(), "cuda:1": figures(step=9.0)["cuda:0"]}
    job_folder(out, krea(job_name="split", multi_gpu="model_split"),
               status(run_id="split-1", peak_mem={"cuda:0": 13.0, "cuda:1": 12.5}, checkpoint_memory=two))
    job_folder(out, krea(job_name="no status"), None)
    (out / "broken").mkdir()
    (out / "broken" / "status.json").write_text("{", encoding="utf-8")
    (out / "no config").mkdir()
    (out / "no config" / "status.json").write_text(json.dumps(status()), encoding="utf-8")
    job_folder(out / "_history", krea(job_name="not a job"), status())  # the history's own folder is skipped
    return out


def test_the_dry_run_lists_every_run_and_writes_nothing(tmp_path, capsys):
    out = _past_runs(tmp_path)
    rec = tmp_path / "record.json"
    assert M.main([str(out), "--dry-run", "--record", str(rec)]) == 0
    text = capsys.readouterr().out
    assert not rec.exists() and not list(tmp_path.glob("record.json*"))
    lines = {line.split("  ")[0].strip(): " ".join(line.split()) for line in text.splitlines()}
    assert lines["good"] == "good krea2 auto 1 1 8 LoRA r16 11.80 calibration 6 14.60 15.93 2000 counts"
    assert lines["logged"] == "logged krea2 auto 1 1 6 LoRA r16 12.35 log 6 14.60 16 2000 counts"
    assert lines["fixed keep"] == "fixed keep krea2 auto 1 1 4 LoRA r16 16.30 peak - 17.50 23.99 2000 counts"
    assert lines["no figures"] == ("no figures ignored: its peak follows the card (blocks kept by measurement) and "
                                   "it left no calibration figures (no train.log read)")
    assert "ignored: failed: OutOfMemoryError" in lines["oom"]
    assert "ignored: only 20 steps (under 100)" in lines["smoke"]
    assert "ignored: did not finish (stage: training)" in lines["killed"]
    assert lines["split"].endswith("kept, not used: the plan is for one GPU and an adapter")
    assert "ignored: status.json not readable" in lines["broken"]
    assert "ignored: config.yaml not readable" in lines["no config"]
    assert "not a job" not in text and "no status" not in text
    assert "4 of 10 runs measured. Dry run: nothing written; the record would hold 4 runs" in text


def test_the_rebuild_adds_past_runs_and_keeps_the_rows_it_had(tmp_path, capsys):
    out = _past_runs(tmp_path)
    rec = tmp_path / "record.json"
    M.add(row(krea(job_name="elsewhere"), run_id="else-1"), rec)  # a run whose folder is gone
    M.add(dict(row(krea(job_name="good")), run_id="good-1", step_gb=1.0), rec)  # an older row of a scanned run
    assert M.main([str(out), "--record", str(rec), "--card-gb", "15.9"]) == 0
    assert "The record now holds 5 runs" in capsys.readouterr().out
    rows = {r["job"]: r for r in M.load(rec)}
    assert set(rows) == {"elsewhere", "good", "logged", "fixed keep", "split"}
    assert rows["good"]["step_gb"] == 11.8 and rows["good"]["card_gb"] == 15.93  # the card the run kept itself stays
    assert rows["fixed keep"]["card_gb"] == 23.99
    assert rows["logged"]["card_gb"] == 15.9  # --card-gb is for a run that kept none (its log says a whole 16)
    assert rows["good"]["adapter_gb"] > 0 and rows["good"]["folder"].endswith("good")
    assert M.main([str(out), "--record", str(rec), "--fresh"]) == 0
    assert {r["job"] for r in M.load(rec)} == {"good", "logged", "fixed keep", "split"}
    assert M.main([str(tmp_path / "nowhere"), "--record", str(rec)]) == 2


def test_rebuilt_runs_plan_the_wizard_at_once(tmp_path):
    out = _past_runs(tmp_path)
    M.rebuild(out)  # into the record the wizard reads
    swap, reason = wizard(krea())["block_swap"]
    # three runs of these settings: "fixed keep" needs the most with no block swapped (16.3 GiB + 4 blocks)
    assert "measured on your run 'fixed keep': 16.3 GB at 4 blocks swapped" in reason and swap == 10
    M.save([r for r in M.load() if r["job"] != "fixed keep"])
    swap, reason = wizard(krea())["block_swap"]
    assert swap == 4 and "measured on your run 'good': a step took 11.8 GB at 8 blocks swapped" in reason


# ---------------------------------------------------------------- found in review

def test_the_blocks_between_are_sized_from_the_run_s_own_model():
    # the estimate takes a Krea 2 int8 block for 0.48 GB; this run's model held 0.25 GiB a block
    sized = dict(row(), block_gb=[0.25] * 28)
    plan = find([sized])
    assert plan["blocks_from"] == "measured" and plan["swap"] == 0  # 11.8 + 8 x 0.25 = 13.8 fits with 2.0 kept free
    assert plan["need_gb"] == pytest.approx(13.8) and plan["adjust_gb"] == pytest.approx(2.0)
    assert find([row()])["blocks_from"] == "estimate" and find([row()])["swap"] == 4
    # asked for more blocks than it ran with, each block gives back what it really holds: more blocks than the
    # estimate's size would say
    small = find([sized], vram=12.0)  # 1.8 GiB over at 8 blocks: 8 more of 0.25 (the estimate's size says 4)
    assert small["swap"] == 16 and small["saved_gb"] == pytest.approx(4.0)
    assert find([row()], vram=12.0)["swap"] == 13
    # the first blocks are the swapped ones, and they need not be the same size (FLUX: double, then single blocks)
    mixed = dict(row(), block_gb=[0.5] * 8 + [0.2] * 20)
    assert find([mixed])["swap"] == 4 and find([mixed])["need_gb"] == pytest.approx(11.8 + 4 * 0.5)
    assert find([mixed], vram=12.0)["swap"] == 17 and find([mixed], vram=12.0)["need_gb"] == pytest.approx(10.0)
    # sizes for another model (fewer blocks than this one has) are not used
    assert find([dict(row(), block_gb=[0.25] * 12)])["blocks_from"] == "estimate"
    for junk in ([], [0.25, "x"], [0.0] * 28, "x", {"a": 1}):
        assert find([dict(row(), block_gb=junk)])["blocks_from"] == "estimate"


def test_an_older_run_s_block_size_is_read_from_its_log():
    line = ("12:00:00 INFO Block swap: the frozen weights of blocks 0-7 (8 of 28) stay in RAM, 2.1 GB pinned, and "
            "stream to cuda:0 as each block runs (prefetching the next)\n")
    assert M.log_block_swap(line) == (8, round(2.1 * M.GIB_PER_GB, 3))
    assert M.log_block_swap("12:00 INFO nothing\n") is None
    made, _ = M.measurement(krea(job_name="old", block_swap=8), status(checkpoint_memory=None), adapter_gb=1.0,
                            log_text=log_line() + line)
    assert (made["swapped_blocks"], made["swapped_gb"]) == (8, 1.956) and "block_gb" not in made
    plan = find([made])
    # 12.35 (the log's 11.85 and UNSEEN_GB) at 8 blocks of 0.245: 6 fewer leave 2.11, 7 fewer too little
    assert plan["blocks_from"] == "log" and plan["swap"] == 2
    # the run's own sizes beat the log's, and a run that swapped nothing has no such line to read
    assert "swapped_gb" not in M.measurement(krea(block_swap=8), status(block_gb=[0.25] * 28), adapter_gb=1.0,
                                             log_text=line)[0]
    assert "swapped_gb" not in M.measurement(krea(), status(), adapter_gb=1.0, log_text=line)[0]


def test_block_sizes_are_read_off_a_model_without_touching_its_weights():
    torch = pytest.importorskip("torch")
    from rulztrainer.blockswap import frozen_block_gb

    class Block(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.frozen = torch.nn.Linear(width, width, bias=False).requires_grad_(False)  # fp32: 4 bytes each
            self.lora = torch.nn.Linear(width, 4, bias=False)  # trainable: stays on the GPU, not counted
            self.register_buffer("quant", torch.zeros(width, dtype=torch.int8))  # a quantized layer's data

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(64), Block(128)])

    assert frozen_block_gb(Model()) == [round((64 * 64 * 4 + 64) / 2 ** 30, 4), round((128 * 128 * 4 + 128) / 2 ** 30, 4)]


def test_a_finished_run_keeps_its_block_sizes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from rulztrainer import trainer as T

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(2048, 2048, bias=False).requires_grad_(False) for _ in range(28)])

    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: False)
    tr = _trainer(tmp_path, krea(job_name="sized", block_swap=8), status())
    tr.replicas = [(Model(), None, "cuda:0")]
    tr._record_memory()
    (kept,) = M.load()
    assert kept["block_gb"] == [round(2048 * 2048 * 4 / 2 ** 30, 4)] * 28
    saved = json.loads((tr.job_dir / "status.json").read_text(encoding="utf-8"))
    assert saved["block_gb"] == kept["block_gb"]  # in the job folder too, for a rebuild


def test_the_reason_says_where_the_blocks_size_comes_from():
    made = record(krea(job_name="sized", block_swap=8))
    M.save([dict(made, block_gb=[0.25] * 28)])
    assert wizard(krea())["block_swap"] == (
        0, "measured on your run 'sized': a step took 11.8 GB at 8 blocks swapped (its 13.6 GB peak includes the 6 "
           "blocks it kept in spare memory); adjusted for 8 fewer blocks swapped (their weights as that run's model "
           "holds them), it fits 16 GB as it is, a step leaving about 2.1 GB of 16 GB free; swapping only slows the "
           "steps")
    M.save([dict(made, swapped_blocks=8, swapped_gb=2.0)])
    assert "8 fewer blocks swapped (at the size that run's log gives)" in wizard(krea())["block_swap"][1]


def test_one_unreadable_row_does_not_hide_the_others():
    good = row()
    bad = [dict(good, run_id="b1", block_swap="many"), dict(good, run_id="b2", extras="x"),
           dict(good, run_id="b3", finished="yesterday"), dict(good, run_id="b4", step_gb=None),
           dict(good, run_id="b5", peak_mem=7), dict(good, run_id="b6", adapter_gb="big"),
           dict(good, run_id="b7", card_gb="x"), 5, None, "row"]
    plan = find(bad + [good] + bad)
    assert plan is not None and plan["row"]["run_id"] == "run-1" and plan["swap"] == 4
    assert find(bad) is None or find(bad)["row"]["run_id"] in {"b3"}  # (a text date is the only harmless one)

    def touchy(r):
        if r.get("run_id") == "b1":
            raise RuntimeError("resolver")
        return "int8"

    assert find([dict(good, run_id="b1"), good], quant_of=touchy)["row"]["run_id"] == "run-1"


def test_a_damaged_job_does_not_stop_a_rebuild(tmp_path, monkeypatch):
    out = tmp_path / "output"
    job_folder(out, krea(job_name="a damaged"), status(checkpoint_memory={"cuda:0": {"step_gb": 12, "kept": "bad"}}))
    job_folder(out, krea(job_name="b fine", block_swap=8), status(run_id="fine-1"))
    found, total = M.rebuild(out, tmp_path / "record.json")
    assert [(e["job"], e["row"] is not None) for e in found] == [("a damaged", False), ("b fine", True)] and total == 1
    real = M.measurement

    def flaky(cfg, *a, **k):
        if cfg.job_name == "a damaged":
            raise RuntimeError("boom")
        return real(cfg, *a, **k)

    monkeypatch.setattr(M, "measurement", flaky)
    found, total = M.rebuild(out, tmp_path / "record2.json")
    assert found[0]["why"] == "not read (RuntimeError)" and found[1]["row"] is not None and total == 1


def test_the_calibration_line_is_found_however_long_the_log_grew(tmp_path):
    out = tmp_path / "output"
    steps = "".join(f"12:00:{i % 60:02d} INFO step {i}/90000 loss 0.1234 avg 0.1234 lr 1.00e-04 grad 0.12 2.1s/it "
                    "(last 5 steps 2.10s each)\n" for i in range(60000))
    assert len(steps) > 6 * 2 ** 20
    swap = ("11:59:00 INFO Block swap: the frozen weights of blocks 0-5 (6 of 28) stay in RAM, 1.6 GB pinned, and "
            "stream to cuda:0 as each block runs (prefetching the next)\n")
    job_folder(out, krea(job_name="long", block_swap=6), status(checkpoint_memory=None), log=swap + log_line() + steps)
    (entry,) = M.scan(out)
    assert entry["row"] is not None and entry["row"]["step_from"] == "log" and entry["row"]["step_gb"] == 12.35
    assert entry["row"]["swapped_blocks"] == 6


def test_a_writer_removes_only_its_own_lock(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    lock = tmp_path / "record.json.lock"
    with M._locked(path):
        assert lock.exists() and lock.read_text(encoding="utf-8")
        lock.write_text("another-writer", encoding="utf-8")  # ours was taken over as stale; the lock is theirs now
    assert lock.read_text(encoding="utf-8") == "another-writer"
    # a lock that stays held: the write waits LOCK_WAIT, then goes ahead and leaves the lock alone
    monkeypatch.setattr(M, "LOCK_WAIT", 0.2)
    M.add(row(), path)
    assert len(M.load(path)) == 1 and lock.read_text(encoding="utf-8") == "another-writer"


def test_two_writers_at_once_both_keep_their_rows(tmp_path):
    import threading

    path = tmp_path / "record.json"
    rows = [dict(row(), run_id=f"w{i}", finished=3000.0 + i) for i in range(24)]
    threads = [threading.Thread(target=M.add, args=(r, path)) for r in rows]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r["run_id"] for r in M.load(path)) == sorted(r["run_id"] for r in rows)
    assert [p.name for p in tmp_path.iterdir()] == ["record.json"]


# The wizard's block swap pick as the code before the lookup gave it (3.20.0), written down from that code: the
# plan with no record must stay exactly this.
BEFORE = [
    ("krea2", {}, 16311, (7, ESTIMATE)),
    ("krea2", {}, 12288, (15, "the model and a step's memory do not fit 12 GB: 15 blocks' frozen weights (7.2 GB) "
                              "wait in system RAM and stream in as each block runs")),
    ("moss_image", {}, 8192, (18, "the model and a step's memory do not fit 8 GB: 18 blocks' frozen weights (2.4 GB) "
                                 "wait in system RAM and stream in as each block runs")),
    ("qwen_image21", {}, 16311, (0, "the model fits 16 GB as it is; swapping only slows the steps")),
    ("pan2", {}, 24564, (17, "the model and a step's memory do not fit 24 GB: 17 blocks' frozen weights (11.4 GB) "
                             "wait in system RAM and stream in as each block runs")),
]


@pytest.mark.parametrize("fam, kw, mib, expected", BEFORE)
def test_the_plan_without_a_record_is_the_3_20_plan(fam, kw, mib, expected):
    assert wizard(TrainConfig(model_arch=fam, **kw), [{"total": mib}])["block_swap"] == expected


def test_blocks_in_flight_count_when_a_run_that_swapped_none_is_asked_to_swap():
    # block swap holds two swapped blocks' weights on the GPU at a time: a run with none swapped had none of that.
    # 14.5 GiB at no swap, blocks of 0.25: the first two swapped give nothing back, so 0.57 GiB over takes 5, not 3
    zero = dict(row(krea(job_name="zero"), checkpoint_memory=figures(step=14.5, kept=0)), block_gb=[0.25] * 28)
    plan = find([zero])
    assert plan["at_blocks"] == 0 and plan["swap"] == 5 and plan["need_gb"] == pytest.approx(14.5 - 5 * 0.25 + 0.5)
    # at the estimate's size too
    est = find([row(krea(job_name="zero"), checkpoint_memory=figures(step=14.5, kept=0))])
    assert est["swap"] == 4 and est["need_gb"] == pytest.approx(14.5 - 2 * BLOCK)  # 0.57 over: 1.27 blocks, plus 2
    # a run that swapped some already held them: fewer blocks are not credited for it (11.8 + 8 x 0.25, no less)
    assert find([dict(row(), block_gb=[0.25] * 28)])["need_gb"] == pytest.approx(13.8)
    # and of blocks of different sizes the two largest are the ones counted: 5 blocks hold 1.9, two of 0.5 in flight
    mixed = dict(row(krea(job_name="zero"), checkpoint_memory=figures(step=14.5, kept=0)), block_gb=[0.5, 0.2] * 14)
    assert find([mixed])["swap"] == 5 and find([mixed])["need_gb"] == pytest.approx(14.5 - 1.9 + 1.0)


def test_a_run_stopped_before_any_probe_does_not_count():
    probing = {"likeness": {"best": None}}
    made, why = M.measurement(krea(), status(step=150, total=2000, **probing), adapter_gb=1.0)
    assert made is None and why == "ended before any likeness probe ran"
    assert M.measurement(krea(), status(step=150, **probing, probes=[]), adapter_gb=1.0)[0] is None
    assert M.measurement(krea(), status(step=150, **probing, probes=[{"step": 0}]), adapter_gb=1.0)[0] is not None
    assert M.measurement(krea(), status(step=150, total=2000), adapter_gb=1.0)[0] is not None  # a run without probes


def test_the_kind_of_model_file_must_match():
    pruned = TrainConfig(model_arch="minimax_h3", job_name="h", block_swap=40, model_file_quant="pruned_int8_convrot")
    full = TrainConfig(model_arch="minimax_h3", job_name="h", block_swap=40, model_file_quant="int8_convrot")
    assert find([row(pruned)], want(pruned)) is not None
    assert find([row(pruned)], want(full)) is None  # both "int8" to the estimate, not the same weights in memory
    repo = krea(job_name="repo", block_swap=8, model_source="hf")
    assert find([row(repo)], want(krea(model_source="hf"))) is not None
    assert find([row(repo)], want()) is None and find([row()], want(krea(model_source="hf"))) is None


def test_a_quantization_the_plan_cannot_place_is_not_comparable():
    import webui

    q = webui.measured_quant
    assert q("krea2", "files", "int8_convrot", 16.0, "auto") == "int8" and q("krea2", "hf", "", 24.0, "auto") == "int8"
    assert q("krea2", "hf", "", None, "auto") is None  # by card size, and the card is not known
    # the estimate counts these as bf16 whatever the card; in memory they are int8 or nvfp4 by the card
    for fam in ("minimax_h3", "p_h3_image", "minimax_h3_image", "pan2", "moss_image"):
        assert q(fam, "hf", "", 32.0, "auto") is None and webui.plan_quant_of(fam, "hf", "", 32.0, "auto") == "bf16"
    assert q("minimax_h3", "files", "bf16", 32.0, "auto") is None and q("pan2", "files", "bf16", 48.0, "auto") is None
    assert q("minimax_h3", "files", "pruned_int8_convrot", 16.0, "auto") == "int8"
    assert q("krea2", "files", "bf16", 24.0, "auto") == "bf16"  # Krea 2 trains a bf16 file as it is
    assert q("minimax_h3", "hf", "", 32.0, "int8") == "int8"  # set by hand: no card needed
    # the wizard: an H3 repo on auto is planned by the estimate, with a record or without
    repo = TrainConfig(model_arch="minimax_h3", model_source="hf", job_name="h3_repo", block_swap=40)
    record(repo, checkpoint_memory=figures(step=11.0, kept=4, cands=9))
    assert "estimated (no comparable" in wizard(TrainConfig(model_arch="minimax_h3", model_source="hf"))["block_swap"][1]


def test_a_card_named_two_ways_is_one_card():
    import webui

    assert webui.same_card(16.0, 15.93) == 15.93 and webui.same_card(15.93, 15.929) == 15.929
    assert webui.same_card(23.99, 15.93) == 23.99 and webui.same_card(None, 15.93) is None
    assert webui.same_card("x", 15.93) is None
    # Qwen-Image 2.1 on a repo: bf16 from 24 GB. A 24 GB card is 23.99 to nvidia-smi and "24" in a log: the run and
    # the form are one card, and the threshold between the two figures must not part them
    qwen = TrainConfig(model_arch="qwen_image21", model_source="hf", job_name="q", resolutions=[0.5])  # (the wizard's
    # pick for Qwen-Image 2.1 is 0.5 MP)
    made, _ = M.measurement(qwen, status(checkpoint_memory=None, peak_mem={"cuda:0": 20.0}),
                            log_text=log_line(shown=24, budget=22.1, peak=19.0, kept=3))  # no card kept but the log's
    assert made["card_gb"] == 24.0
    M.save([made])
    assert "measured on your run 'q'" in wizard(TrainConfig(model_arch="qwen_image21", model_source="hf"),
                                               GPU24)["block_swap"][1]


def test_of_many_equal_runs_only_the_newest_few_are_weighed():
    heavy = dict(row(checkpoint_memory=figures(step=13.0)), job="heavy", run_id="h", finished=10.0)
    light = [dict(row(), job=f"light{i}", run_id=f"l{i}", finished=100.0 + i) for i in range(M.RECENT_ROWS)]
    assert find([heavy] + light[:-1])["row"]["job"] == "heavy"  # still among the newest five
    assert find([heavy] + light)["row"]["job"] == f"light{M.RECENT_ROWS - 1}"  # five newer runs: it no longer holds
    assert find(light + [dict(heavy, finished=500.0)])["row"]["job"] == "heavy"  # the same run made today counts


def test_a_run_over_this_card_from_an_unknown_card_is_no_close_call():
    bare = figures(step=19.0, kept=3)
    del bare["cuda:0"]["total_gb"]
    big = row(krea(job_name="big"), checkpoint_memory=bare, peak_mem={"cuda:0": 21.0})  # 19.6 GiB: a bigger card's
    plan = find([big], vram=24.0)
    assert plan is not None and plan["card_gb"] is None and not plan["close_call"]
    bare = figures()
    del bare["cuda:0"]["total_gb"]
    over = find([row(checkpoint_memory=bare, peak_mem={"cuda:0": 17.5})])  # 16.3 GiB: more than this card has
    assert over["card_gb"] is None and over["headroom_gb"] < 0 and not over["close_call"]  # so not made on it


def test_the_card_a_run_kept_beats_the_one_given_which_beats_a_log_s():
    assert M.measurement(krea(), status(), adapter_gb=1.0, card_fallback=11.0)[0]["card_gb"] == 15.93
    assert M.measurement(krea(), status(gpu_total_gb={"cuda:0": 23.99}), adapter_gb=1.0,
                         card_fallback=11.0)[0]["card_gb"] == 23.99
    logged = M.measurement(krea(), status(checkpoint_memory=None), adapter_gb=1.0, log_text=log_line(),
                           card_fallback=15.93)[0]
    assert logged["card_gb"] == 15.93
    assert M.measurement(krea(checkpoint_keep_blocks=0), status(checkpoint_memory=None), adapter_gb=1.0,
                         card_fallback=15.93)[0]["card_gb"] == 15.93


def test_the_log_lines_read_are_the_ones_the_trainer_writes_and_the_hooks_are_called():
    """measured.py reads two of the trainer's log lines by their wording, and two trainer methods feed it: a
    reworded line or a dropped call would silently leave runs out of the record."""
    import inspect

    from rulztrainer import trainer as T

    calibrate = inspect.getsource(T.Trainer._calibrate_checkpointing)
    assert '"Checkpointing: %d of %d blocks keep their activations, measured at %d tokens (%s)"' in calibrate
    assert ('f"{d}: {k} of {n} ({per_block / ckpt.GB:.2f} GB per extra block, peak "' in calibrate
            and 'f"{peak_k / ckpt.GB:.1f} of {total / ckpt.GB:.0f} GB, budget {budget / ckpt.GB:.1f})"' in calibrate)
    assert "self._note_calibration(figures, d, k, n, peak_one, per_block, budget, total, one[d][1])" in calibrate
    swap = inspect.getsource(T.Trainer._setup_block_swap)
    assert ('"Block swap: the frozen weights of blocks 0-%d (%d of %d) stay in RAM, %.1f GB pinned, and stream to "'
            in swap and "self.block_swap.bytes / 1e9" in swap)
    run = inspect.getsource(T.Trainer._run)
    assert run.index('self._save_history("finished" if step >= cfg.steps else "stopped")') < run.index(
        "self._record_memory()")
    # the lines as those formats print them are the lines the tests feed the parser
    line = "Checkpointing: %d of %d blocks keep their activations, measured at %d tokens (%s)" % (
        6, 28, 4608, f"cuda:0: {6} of {20} ({0.45:.2f} GB per extra block, peak {14.1:.1f} of {16:.0f} GB, budget "
                     f"{14.4:.1f})")
    assert "12:00:01 INFO " + line + "\n" == log_line()
    assert M.log_block_swap("Block swap: the frozen weights of blocks 0-%d (%d of %d) stay in RAM, %.1f GB pinned, "
                            "and stream to %s as each block runs (prefetching the next)" % (7, 8, 28, 2.1, "cuda:0")
                            ) == (8, round(2.1 * M.GIB_PER_GB, 3))


# MiniMax H3 on one 16 GB card, from the code before the lookup (3.20.0), plus the 2 blocks in flight (4.1.1)
H3_BEFORE = (27, "the 21 GB model does not fit 16 GB: 27 of the 50 blocks' frozen weights (10.4 GB) wait in pinned system RAM and stream in as each runs (an estimate; musubi-tuner swaps up to 48)")


def test_the_h3_estimate_s_words_are_the_3_20_words():
    assert wizard(TrainConfig(model_arch="minimax_h3"))["block_swap"] == H3_BEFORE
