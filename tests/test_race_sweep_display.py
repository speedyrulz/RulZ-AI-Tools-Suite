import json

import webui
from rulztrainer import history, runreport


def test_race_sweeps_are_kept_separate(tmp_path):
    configs = []
    for i, dropped, steps in ((0, True, (10, 100)), (1, False, (10, 30))):
        rows = [{"index": i, "config": f"racer {i}", "step": step,
                 "train": {"subject": step / 200}, "held": {"subject": step / 400}} for step in steps]
        (tmp_path / f"region_sweep.race{i}.jsonl").write_text(
            "\n".join(map(json.dumps, rows)), encoding="utf-8")
        (tmp_path / f"region_sweep.race{i}.json").write_text("{}", encoding="utf-8")
        configs.append({"index": i, "config": f"racer {i}", "dropped": dropped,
                        "step": steps[-1], "history": rows})
    status = {"region_sweep_race": {"configs": configs}}
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    assert [row["step"] for row in history.sweep_series(status, tmp_path)] == [10, 30]
    assert [row["step"] for row in history.sweep_series({}, tmp_path)] == [10, 100]
    data = webui.region_data(tmp_path)
    assert len(data["race"]) == 2
    assert len(webui.region_signature(tmp_path)) == 5
    lines, names = webui.race_sweep_gap_series(data["race"], tmp_path)
    assert names == ["racer 0 (dropped)", "racer 1"]
    assert [point[0] for point in lines[1]["data"]] == [10, 30]
    run = runreport.load(tmp_path)
    assert len(run.race_sweeps) == 2
    assert [row["step"] for row in run.sweeps] == [10, 30]


def _write_race(tmp_path, indices, status_configs=None):
    for i in indices:
        rows = [{"index": i, "config": f"racer {i}", "step": 10 * (i + 1), "train": {"subject": 0.2},
                 "held": {"subject": 0.1}}]
        (tmp_path / f"region_sweep.race{i}.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        (tmp_path / f"region_sweep.race{i}.json").write_text("{}", encoding="utf-8")
    if status_configs is not None:
        (tmp_path / "status.json").write_text(json.dumps({"region_sweep_race": {"configs": status_configs}}),
                                              encoding="utf-8")


def test_race10_sorts_after_race2(tmp_path):
    _write_race(tmp_path, (0, 1, 2, 10))
    assert [i for i, _ in webui.race_sweep_files(tmp_path)] == [0, 1, 2, 10]
    assert [name for name, *_ in webui.region_signature(tmp_path) if name.startswith("region_sweep.race")
            and name.endswith(".jsonl")] == [f"region_sweep.race{i}.jsonl" for i in (0, 1, 2, 10)]
    assert [c["index"] for c in webui.region_data(tmp_path)["race"]] == [0, 1, 2, 10]
    # configs from the status come first in the status, files add the rest; the combined list is by index
    _write_race(tmp_path, (0, 1, 2, 10), [{"index": 10, "config": "ten", "step": 110},
                                          {"index": 2, "config": "two", "step": 30}])
    assert [c["index"] for c in webui.region_data(tmp_path)["race"]] == [0, 1, 2, 10]
    run = runreport.load(tmp_path)
    assert [c["index"] for c in run.race_sweeps] == [0, 1, 2, 10]
    # the later racer (index 10) has the highest step, so it is the one the plain report shows
    assert run.sweeps[-1]["step"] == 110


def test_history_picks_highest_step_by_index_and_skips_failed(tmp_path):
    _write_race(tmp_path, (2, 10))
    assert [r["step"] for r in history.sweep_series({}, tmp_path)] == [110]
    configs = [{"index": 2, "config": "racer 2", "step": 30},
               {"index": 10, "config": "racer 10", "step": 110, "failed": True}]
    status = {"region_sweep_race": {"configs": configs}}
    assert [r["step"] for r in history.sweep_series(status, tmp_path)] == [30]
    # all failed: fall back to all of them
    for c in configs:
        c["failed"] = True
    assert [r["step"] for r in history.sweep_series(status, tmp_path)] == [110]
    _write_race(tmp_path, (2, 10), [{"index": 2, "config": "a", "failed": True},
                                    {"index": 10, "config": "b", "dropped": True}])
    assert [c["index"] for c in runreport.load(tmp_path).race_sweeps] == [2, 10]
    assert [r["step"] for r in runreport.load(tmp_path).sweeps] == [110]


def test_config_labels_show_state():
    assert webui.race_config_label({"index": 1, "config": "a"}) == "a"
    assert webui.race_config_label({"index": 1}) == "Configuration 1"
    assert webui.race_config_label({"index": 1, "config": "a", "dropped": True}) == "a (dropped)"
    assert webui.race_config_label({"index": 1, "config": "a", "finished": True}) == "a (finished)"
    assert webui.race_config_label({"index": 1, "config": "a", "failed": True}) == "a (failed)"


class _FakeSelect:
    """Stands in for ui.select: setting options/value runs the change handler, like NiceGUI."""

    def __init__(self, on_change):
        self.options, self.value, self.on_change, self.sets = {}, None, on_change, 0

    def set_options(self, options, *, value=...):
        self.options = options
        if value is not ...:
            self.value = value
            self.on_change()
        self.sets += 1


def test_select_fill_does_not_repaint_recursively():
    painting, paints = {"on": False}, []
    select = _FakeSelect(lambda: paints.append(1) if not painting["on"] else None)
    configs = [{"index": 3, "config": "c"}, {"index": 5, "config": "e", "dropped": True}]
    webui.fill_race_select(select, configs, painting)
    assert paints == [] and select.value == 3 and select.options == {3: "c", 5: "e (dropped)"}
    assert painting["on"] is False
    # a value that is gone falls back to the first, still silently; unchanged options set nothing
    select.value = 99
    webui.fill_race_select(select, configs, painting)
    assert select.value == 3 and paints == []
    sets = select.sets
    webui.fill_race_select(select, configs, painting)
    assert select.sets == sets
    assert webui.race_config_pick(configs, 5)["config"] == "e"
    assert webui.race_config_pick(configs, 77)["config"] == "c"
