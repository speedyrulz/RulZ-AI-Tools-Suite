"""The Train tab audit's pure parts: which settings the schedule greying covers, the epoch readout under Steps, the
Dataset tab's caption file choice, the Samples line and the Body score descriptions."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402


def cfg(optimizer, isolation=False, dop_weight=2.0, learning_rate=1e-4):
    return SimpleNamespace(optimizer=optimizer, learning_rate=learning_rate, isolation=isolation,
                           dop_weight=dop_weight)


def test_schedule_is_greyed_for_automagic_and_keel_only():
    for name in ("automagic3", "automagic4", "keel", "keel_fb", "keel3", "keel3_fb"):
        assert webui.schedule_reason(cfg(name)), name
    for name in ("adamw", "adamw8bit", "adafactor", "muon"):
        assert webui.schedule_reason(cfg(name)) == "", name


def test_hull_keeps_its_schedule():
    """build_scheduler holds the factor at 1 for the AUTOMAGIC set only: Hull's step follows the schedule."""
    assert webui.schedule_reason(cfg("hull")) == ""


def test_isolation_trains_keel_with_adamw_so_the_schedule_acts():
    """effective_optimizer: isolation with a DOP weight trains an Automagic or Keel pick with AdamW."""
    assert webui.schedule_reason(cfg("keel", isolation=True, dop_weight=2.0)) == ""
    assert webui.schedule_reason(cfg("keel", isolation=True, dop_weight=0.0))  # no DOP pass: still Keel


class Box:
    def __init__(self):
        self.enabled = True
        self.text = ""
        self.visible = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True


def test_sync_schedule_greys_and_explains():
    boxes, why = [Box(), Box()], Box()
    webui.sync_schedule(cfg("automagic4"), boxes, why)
    assert [b.enabled for b in boxes] == [False, False] and why.visible and "own learning rate" in why.text
    webui.sync_schedule(cfg("adamw"), boxes, why)
    assert [b.enabled for b in boxes] == [True, True] and not why.visible and why.text == ""


def test_steps_readout_counts_epochs_from_the_datasets():
    text = webui.steps_readout(1000, 2, 4, 1, [(62, 1, 0)])
    assert text == "About 129 epochs: 62 photos per epoch (repeats counted), 8 per step."
    # repeats multiply, a per-epoch cap limits a dataset, and data parallel multiplies the step
    text = webui.steps_readout(300, 1, 1, 2, [(30, 2, 0), (100, 1, 20)])
    assert text == "About 7.5 epochs: 80 photos per epoch (repeats counted), 2 per step."


def test_steps_readout_small_and_unknown_cases():
    assert webui.steps_readout(100, 1, 1, 1, [(100, 1, 0)]).startswith("About 1 epoch:")
    assert webui.steps_readout(100, 1, 1, 1, []) == ""
    assert webui.steps_readout(0, 1, 1, 1, [(10, 1, 0)]) == ""
    assert webui.steps_readout(None, 1, 1, 1, [(10, 1, 0)]) == ""
    assert webui.steps_readout(100, 1, 1, 1, [(0, 1, 0)]) == ""
    assert webui.steps_readout(100, 1, 1, 1, [(10, 1, 0)], crops=True).endswith("automatic crops are not counted.")


def test_caption_target_prefers_a_same_name_json_only_for_json_caption_models(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    assert webui.caption_target(img, True) == (tmp_path / "a.txt", False)
    (tmp_path / "a.txt").write_text("plain", encoding="utf-8")
    assert webui.caption_target(img, True) == (tmp_path / "a.txt", False)
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    assert webui.caption_target(img, True) == (tmp_path / "a.json", True)
    # MOSS, Krea, Qwen, Klein, Z-Image, SDXL train on the .txt: a .json beside it is not shown or overwritten
    assert webui.caption_target(img, False) == (tmp_path / "a.txt", False)
    assert webui.JSON_CAPTION_ARCHS == {"fibo2", "ideogram4"}


def test_json_caption_problem_warns_and_never_refuses():
    assert webui.json_caption_problem("") == ""
    assert webui.json_caption_problem('{"high_level_description": "x"}') == ""
    assert webui.json_caption_problem('{"a": ')  # a message, not an exception


def test_write_caption_saves_invalid_json_as_typed_with_a_warning(tmp_path):
    cap = tmp_path / "a.json"
    cap.write_text("{}", encoding="utf-8")
    status, warn = webui.write_caption(cap, True, '{"a": ')
    assert cap.read_text(encoding="utf-8") == '{"a": ' and warn and "not valid JSON" in status
    status, warn = webui.write_caption(cap, True, '{"a": 1}')
    assert (status, warn) == ("saved", False)
    txt = tmp_path / "b.txt"  # a plain caption is never checked as JSON, and is created when missing
    assert webui.write_caption(txt, False, "{not json") == ("saved", False) and txt.read_text(encoding="utf-8") == "{not json"
    assert webui.write_caption(txt, False, "  ") == ("saved (empty)", False)


def test_samples_info_says_when_the_grid_is_cut_and_when_columns_do_not_apply():
    assert webui.samples_info(0, None) == "No job selected"
    assert webui.samples_info(12, "output/job/samples") == "12 samples in output/job/samples"
    cut = webui.samples_info(300, "output/job/samples")
    assert cut.startswith(f"Showing the {webui.SAMPLES_SHOWN} newest of 300 samples")
    assert "Columns is not used" in webui.samples_info(12, "x", race=True)
    assert "Columns is not used" not in cut


def test_body_score_note_describes_the_picked_option():
    for key in ("appearance", "anchored", "proportions", "skeleton3d", "smplx", "smplx_ratio"):
        assert webui.body_score_note(key).startswith(webui.BODY_SCORE_NOTES[key])
    shared = webui.BODY_SCORE_SHARED
    assert shared not in webui.body_score_note("appearance") and shared not in webui.body_score_note("anchored")
    assert all(shared in webui.body_score_note(k) for k in ("proportions", "skeleton3d", "smplx", "smplx_ratio"))
    assert webui.body_score_note("nope") == ""
