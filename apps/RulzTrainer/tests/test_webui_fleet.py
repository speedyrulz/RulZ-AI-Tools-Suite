"""The WebUI's Fleet tab: how it words an instance's card, an event and a past run, and its small chart."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))  # rulz_fleet and rulz_remote, also when this file runs alone
pytest.importorskip("nicegui")
pytest.importorskip("psutil")

import webui  # noqa: E402


def test_a_cards_run_line_says_progress_loss_time_left_and_cost():
    inst = {"job": {"job": "a", "running": True, "step": 1240, "total": 2000, "avg_loss": 0.2075, "eta": 640.0},
            "metrics": {"cost": 0.281}}
    assert webui.fleet_job_line(inst) == "step 1,240 / 2,000 · loss 0.2075 · 11 min left · $0.28 so far"
    ended = {"job": {"job": "a", "running": False, "step": 2000, "total": 2000, "done": True}}
    assert webui.fleet_job_line(ended) == "step 2,000 / 2,000 · finished"
    assert webui.fleet_job_line({"job": {"running": False, "step": 5, "total": 9, "error": "x"}}).endswith("failed")
    assert webui.fleet_job_line({"job": {"running": False, "step": 5, "total": 9}}).endswith("ended early")
    assert webui.fleet_job_line({}) == ""


def test_the_last_probe_reads_as_one_line():
    probe = {"step": 1200, "face_ema": 73.61, "body_ema": 48.2, "combined_ema": 47.0, "best": 50.0, "best_step": 1000}
    assert webui.fleet_probe_text(probe) == "step 1,200: face 73.6 · body 48.2 · combined 47.0 · best 50.0 at 1,000"
    assert webui.fleet_probe_text({"face_ema": 60.0}) == "face 60.0"
    assert webui.fleet_probe_text(None) == ""


def test_the_machine_line_names_disk_gpu_and_the_looks_age():
    inst = {"disk_free_gb": 80.2, "disk_total_gb": 200.0, "age": 12,
            "gpus": [{"utilization": 97, "temperature": 60, "memory_used_mb": 9216, "memory_total_mb": 32768}]}
    assert webui.fleet_machine_line(inst) == "disk 80 GB free of 200 · GPU 97% · 60°C · 9.0 / 32 GB · seen 12 s ago"
    assert webui.fleet_machine_line({"age": 600}) == "seen 10 min ago"


def test_every_kind_of_event_reads_as_a_sentence():
    import rulz_fleet

    base = {"instance": 7, "job": "a", "error": "boom", "status": "exited", "step": 40}
    for kind in rulz_fleet.EVENT_KINDS:
        if kind != "watch":
            text = webui.fleet_event_text(dict(base, kind=kind))
            assert text and text != kind and "None" not in text, kind
    watch = {"kind": "watch", "metric": "avg_loss", "op": "above", "value": 0.5, "actual": 0.71, "instance": 7,
             "job": "a", "step": 120.0, "note": "diverging?"}
    assert webui.fleet_event_text(watch) == \
        "avg_loss went above 0.5: now 0.71 (a on instance 7, step 120) · diverging?"
    machine = {"kind": "watch", "metric": "disk_free_gb", "op": "below", "value": 15.0, "actual": 4.0, "instance": 7}
    assert webui.fleet_event_text(machine) == "disk_free_gb went below 15: now 4 (instance 7)"
    assert webui.fleet_event_text({"kind": "run_done", "job": "a", "instance": 7, "best": 55.52}) == \
        "a finished on instance 7, best likeness 55.5"


def test_past_runs_become_table_rows():
    rows = webui.fleet_run_rows([
        {"key": "7-a-1", "job": "a", "state": "done", "instance": 7, "gpu": "RTX 5090", "step": 2000, "total": 2000,
         "avg_loss": 0.2075, "best": 49.96, "elapsed": 1740.0, "cost": 0.4444, "ended": 1_000_000.0, "group": "r",
         "downloaded": ["logs", "probes"]},
        {"key": "7-b", "job": "b", "state": "running", "instance": 7}])
    assert rows[0] == dict(rows[0], job="a", state="done", where="7 · RTX 5090", steps="2,000 / 2,000", loss="0.2075",
                           best=50.0, time="29 min", cost="$0.44", group="r", files="logs, probes")
    assert rows[0]["ended"].startswith("1970-01-1")
    assert rows[1]["best"] is None and rows[1]["steps"] == "0 / 0" and rows[1]["cost"] == ""
    assert {c["field"] for c in webui.FLEET_RUN_COLUMNS} <= set(rows[0])


def test_the_download_dialog_ticks_the_kinds_a_run_has_rules_for():
    import rulz_fleet

    order = list(rulz_fleet.DOWNLOAD_KINDS)
    rules = {"logs": "during", "probes": "during", "samples": "end", "maps": "off", "best": "end", "loras": "off"}
    assert webui.fleet_download_defaults(order, rules) == ["logs", "probes", "samples", "best"]
    assert webui.fleet_download_defaults(order, rules, {"loras": "end", "logs": "off"}) == \
        ["probes", "samples", "best", "loras"]
    assert webui.fleet_download_defaults(order, {k: "off" for k in order}) == ["logs", "best"]
    assert webui.fleet_download_defaults(order, {k: "off" for k in order}, {"maps": "end"}) == ["maps"]


def test_a_card_downloads_its_running_run_or_else_its_newest():
    runs = [{"key": "7-new", "instance": 7, "state": "done"}, {"key": "7-live", "instance": 7, "state": "running"},
            {"key": "8-old", "instance": 8, "state": "error"}]
    assert webui.fleet_card_run(runs, 7)["key"] == "7-live"
    assert webui.fleet_card_run(runs[:1] + runs[2:], 7)["key"] == "7-new"
    assert webui.fleet_card_run(runs, "8")["key"] == "8-old"
    assert webui.fleet_card_run(runs, 9) is None


def test_the_small_chart_has_the_loss_and_the_likeness_on_its_own_axis():
    options = webui.fleet_spark({"losses": [[10, 0.5], [20, 0.4]],
                                 "probes": [{"step": 10, "combined_ema": 30.0}, {"step": 20, "combined": 35.0},
                                            {"step": 30}]})
    loss, likeness = options["series"]
    assert loss["data"] == [[10, 0.5], [20, 0.4]] and likeness["data"] == [[10, 30.0], [20, 35.0]]
    assert likeness["yAxisIndex"] == 1 and len(options["yAxis"]) == 2
    assert webui.fleet_spark(None)["series"][0]["data"] == []


def test_the_fleet_is_one_object_for_the_whole_webui(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "_FLEET", None)
    import rulz_remote

    monkeypatch.setattr(rulz_remote, "STATE", tmp_path)
    first = webui.fleet()
    assert webui.fleet() is first and first.home == tmp_path / "fleet"


def test_the_cards_version_line_says_what_runs_and_what_is_newer():
    trainer = {"version": "3.16.0", "commit": "1a2b3c4d5e"}
    assert webui.fleet_version_line({"trainer": trainer}) == "RulZ AI Trainer 3.16.0 (1a2b3c4d5e)"
    assert webui.fleet_version_line({"trainer": trainer, "newer": {"version": "3.17.0", "behind": 4}}) == \
        "RulZ AI Trainer 3.16.0 (1a2b3c4d5e) · 3.17.0 is out"
    assert webui.fleet_version_line({"trainer": trainer, "newer": {"version": "3.16.0", "behind": 1}}).endswith(
        "· 1 newer commit")
    assert webui.fleet_version_line({"trainer": trainer, "update": {"state": "running"}}).endswith("updating now")
    assert webui.fleet_version_line({"trainer": trainer, "update": {"state": "failed"}}).endswith("last update failed")
    assert webui.fleet_version_line({}) == ""


def test_the_update_dialog_says_versions_changes_and_whether_it_can_go():
    plan = {"checkout": True, "branch": "app/rulz-trainer", "current": {"version": "3.16.0", "commit": "aaa"},
            "latest": {"version": "3.17.0", "commit": "bbb"}, "behind": 3, "ahead": 0, "up_to_date": False,
            "changed": ["webui.py"], "changed_count": 1, "installs": True, "job_running": False}
    lines, can = webui.fleet_update_plan_text(plan)
    texts = [t for _, t in lines]
    assert can and texts[0] == "Installed: RulZ AI Trainer 3.16.0 (aaa)" and "3.17.0 (bbb)" in texts[1]
    assert "3 newer commits on the branch." in texts and any("requirements changed" in t for t in texts)
    assert ("warning", "1 file was changed on the instance (by rulz_remote.py sync?): webui.py. The update replaces "
                       "it with the newest release's.") in lines
    lines, can = webui.fleet_update_plan_text(dict(plan, job_running=True, job="a"))
    assert not can and lines[-1][0] == "warning" and lines[-1][1].startswith("a is running there")
    lines, can = webui.fleet_update_plan_text(dict(plan, up_to_date=True, changed=[], changed_count=0, behind=0))
    assert not can and "It already runs the newest RulZ AI Trainer." in [t for _, t in lines]
    lines, can = webui.fleet_update_plan_text(dict(plan, checkout=False, behind=None, changed=[], changed_count=0))
    assert can and any("not a git checkout" in t for _, t in lines)


def test_a_download_reads_as_one_line_with_a_bar_value_and_a_kind():
    going = {"run": "k", "job": "a", "instance": 7, "kinds": ["logs", "probes"], "state": "going", "files_done": 3,
             "files_total": 20, "bytes_done": 12_500_000, "bytes_total": 80_000_000, "speed": 2_500_000.0}
    text, value, kind = webui.fleet_download_line(going)
    assert kind == "going" and abs(value - 12.5 / 80) < 1e-9
    assert text == ("a · instance 7 · Logs and settings, Probe images · 12.5 / 80.0 MB · 3 / 20 files · 2.5 MB/s "
                    "· 27 s left".replace("27 s left", webui.eta_text(67.5e6 / 2.5e6) + " left"))
    listing = dict(going, files_total=0, bytes_total=0, bytes_done=0, files_done=0, speed=0.0)
    assert webui.fleet_download_line(listing) == ("a · instance 7 · Logs and settings, Probe images · listing the "
                                                  "files", 0.0, "going")
    done = {"run": "k", "job": "a", "instance": 7, "kinds": ["best"], "state": "done", "fetched": 2, "mb": 12.5}
    assert webui.fleet_download_line(done) == ("a · instance 7 · Best LoRA · 2 files, 12.5 MB", 1.0, "done")
    assert webui.fleet_download_line(dict(done, fetched=0))[0].endswith("nothing new to fetch")
    failed = {"run": "k", "job": "a", "instance": 7, "kinds": ["maps"], "state": "failed", "error": "ssh timed out",
              "bytes_done": 5, "bytes_total": 10}
    assert webui.fleet_download_line(failed) == ("a · instance 7 · Training maps · ssh timed out", 0.5, "failed")
    assert webui.fleet_download_line({"state": "going", "files_total": 4, "files_done": 9})[1] == 1.0  # clamped


def test_open_folder_makes_the_folder_and_asks_the_system_to_show_it(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(webui.sys, "platform", "win32")
    monkeypatch.setattr(webui.os, "startfile", lambda p: calls.append(("startfile", p)), raising=False)
    target = tmp_path / "pulls" / "7"
    assert webui.open_folder(target) == str(target) and target.is_dir()
    assert calls == [("startfile", str(target))]
    monkeypatch.setattr(webui.sys, "platform", "linux")
    monkeypatch.setattr(webui.subprocess, "Popen", lambda cmd, **kw: calls.append(tuple(cmd)))
    webui.open_folder(target)
    monkeypatch.setattr(webui.sys, "platform", "darwin")
    webui.open_folder(target)
    assert calls[1:] == [("xdg-open", str(target)), ("open", str(target))]


def test_region_data_reads_a_downloaded_job_folder(tmp_path):
    import json

    assert webui.region_data(tmp_path) == {"rw": None, "ba": None, "rs": None}
    assert webui.region_signature(tmp_path) == ()
    status = {"region_watch": {"epoch": 2, "step": 40, "frozen": False, "baseline": 3, "baseline_min": 8, "rows": []},
              "block_audit": {"step": 40, "line": "audit line"}}
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    sweep = {"step": 20, "train": {"face": 0.5}, "held": {"face": 0.25}, "pictures": 4, "swept": 4, "seconds": 1.0}
    (tmp_path / "region_sweep.jsonl").write_text(json.dumps({"step": 10, "first": True}) + "\n" + json.dumps(sweep) +
                                                 "\n{torn", encoding="utf-8")
    data = webui.region_data(tmp_path)
    assert data["rw"]["epoch"] == 2 and data["ba"]["line"] == "audit line"
    assert data["rs"]["step"] == 20 and data["rs"]["line"].startswith("training pictures learned face 50%")
    assert [n for n, *_ in webui.region_signature(tmp_path)] == ["status.json", "region_sweep.jsonl"]
    # a status without region work falls back to the saved region watch state
    (tmp_path / "status.json").write_text("{}", encoding="utf-8")
    (tmp_path / "region_watch.json").write_text(json.dumps(
        {"epoch": 3, "frozen": True, "history": [{"step": 60, "learned": {"face": 0.4}, "contrast": {}}]}),
        encoding="utf-8")
    data = webui.region_data(tmp_path)
    assert data["rw"]["step"] == 60 and data["ba"] is None
    assert "face 40%" in webui.region_watch_line(data["rw"])


def test_downloaded_probe_images_become_comparison_entries(tmp_path):
    probes = tmp_path / "probes"
    probes.mkdir()
    for name in ("step000200_face_0.png", "step000200_body_1.png", "lora16_step000400_face_0.png",
                 "x_audit-g1-s000600_face_0.png", "notes.txt"):
        (probes / name).write_bytes(b"x")
    # the fleet's own record keeps the agent's probe columns only: no normalised face or body, no image names
    trimmed = [{"step": 200, "face": 61.0, "body": 40.0, "combined": 52.0, "face_ema": 60.0, "body_ema": 39.0,
                "combined_ema": 51.0, "face_norm_ema": 66.0, "body_norm_ema": 44.0, "best": 52.0, "best_step": 200}]
    entries = webui.fleet_probe_entries(tmp_path, trimmed)
    assert [(e["step"], e["kind"], e["seed"], e["racer"]) for e in entries] == [
        (400, "face", 0, "lora16"), (200, "body", 1, ""), (200, "face", 0, "")]
    assert entries[2]["combined"] == 52.0 and entries[2]["face"] is None and entries[2]["body"] is None  # not raw
    # a status.json record (full=True) shows what the history shows: normalised where it has them, else raw
    full = [dict(trimmed[0], face_norm=70.0, body_norm=45.0), {"step": 400, "face": 30.0, "body": 20.0}]
    entries = webui.fleet_probe_entries(tmp_path, full, full=True)
    assert entries[2]["face"] == 70.0 and entries[2]["body"] == 45.0 and entries[0]["face"] == 30.0
    assert webui.fleet_probe_entries(tmp_path / "none", trimmed) == [] and webui.fleet_probe_entries(None) == []
    rec = {"key": "k", "job": "a", "instance": 7, "folder": str(tmp_path), "probes": trimmed}
    runs = webui.fleet_compare_runs([rec])
    assert [g[0] for g in webui.probe_image_groups(runs)] == ["a · instance 7 lora16", "a · instance 7 winner"]
    assert runs[0]["probes"]["face"] == [[200, 66.0]]  # the *_norm_ema column, as history.probe_series reads it
    import json

    (tmp_path / "status.json").write_text(json.dumps({"probes": full}), encoding="utf-8")
    runs = webui.fleet_compare_runs([rec])  # the downloaded status.json's records win
    assert runs[0]["probe_images"][2]["face"] == 70.0 and runs[0]["probe_raw"]["records"][0]["face"] == 61.0


def test_compared_runs_carry_their_downloaded_region_sweeps(tmp_path):
    import json

    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    # a: only region_sweep.jsonl (a half-written last line is skipped)
    (a / "region_sweep.jsonl").write_text(
        json.dumps({"step": 100, "train": {"face": 0.2}, "held": {"face": 0.1}}) + "\n"
        + json.dumps({"step": 200, "train": {"face": 0.4}, "held": {"face": 0.25}}) + "\n{\"step\": 3", encoding="utf-8")
    # b: the status.json's sweep history wins over the jsonl
    (b / "status.json").write_text(json.dumps({"region_sweep": {"history": [
        {"step": 50, "train": {"face": 0.5, "hair": 0.3}, "held": {}}]}}), encoding="utf-8")
    (b / "region_sweep.jsonl").write_text(json.dumps({"step": 999, "train": {"face": 0.9}}) + "\n", encoding="utf-8")
    runs = webui.fleet_compare_runs([{"key": "a", "job": "ja", "instance": 1, "folder": str(a)},
                                     {"key": "b", "job": "jb", "instance": 2, "folder": str(b)},
                                     {"key": "c", "job": "jc", "instance": 3, "folder": str(c)},
                                     {"key": "d", "job": "jd", "instance": 4}])
    assert [e["step"] for e in runs[0]["sweeps"]] == [100, 200]
    assert runs[1]["sweeps"] == [{"step": 50, "train": {"face": 0.5, "hair": 0.3}, "held": {}}]
    assert runs[2]["sweeps"] == [] and runs[3]["sweeps"] == []

    lines = webui.history_sweep_lines(runs, "face")
    assert [s["id"] for s in lines] == ["ja · instance 1 · training", "ja · instance 1 · held-out",
                                        "jb · instance 2 · training"]
    assert lines[0]["data"] == [[100, 20.0], [200, 40.0]] and lines[1]["lineStyle"]["type"] == "dashed"
    assert [s["id"] for s in webui.history_sweep_lines(runs, "face", ("held",))] == ["ja · instance 1 · held-out"]

    # a new sweep row is read on the next draw (the cache follows the file)
    with open(a / "region_sweep.jsonl", "a", encoding="utf-8") as f:
        f.write("00}\n" + json.dumps({"step": 300, "train": {"face": 0.5}}) + "\n")
    assert [e["step"] for e in webui.fleet_sweeps(a)][-1] == 300
    # outside the downloads folder nothing is read
    assert webui.fleet_compare_runs([{"key": "a", "job": "ja", "instance": 1, "folder": str(a)}],
                                    tmp_path / "pulls")[0]["sweeps"] == []


def test_the_page_only_follows_folders_inside_the_downloads_folder(tmp_path):
    root = tmp_path / "pulls"
    inside = root / "7" / "job"
    inside.mkdir(parents=True)
    assert webui.fleet_folder({"folder": str(inside)}, root) == inside.resolve()
    assert webui.fleet_folder({"folder": str(root)}, root) is None
    assert webui.fleet_folder({"folder": str(tmp_path)}, root) is None
    assert webui.fleet_folder({"folder": str(inside / ".." / ".." / "..")}, root) is None
    assert webui.fleet_folder({}, root) is None
    runs = webui.fleet_compare_runs([{"key": "k", "job": "a", "instance": 7, "folder": str(tmp_path)}], root)
    assert runs[0]["probe_images"] == []


def test_compared_runs_get_history_shaped_series_and_unique_labels():
    probes = [{"step": 0, "face": 10.0, "body": 5.0, "face_ema": 10.0, "body_ema": 5.0, "combined_ema": 8.0},
              {"step": 100, "face": 30.0, "body": 9.0, "face_ema": 20.0, "body_ema": 7.0, "combined_ema": 15.0}, {}]
    runs = webui.fleet_compare_runs([
        {"key": "a", "job": "j", "instance": 7, "probes": probes, "losses": [[1, 0.5], [2, 0.4]], "config": {"x": 1}},
        {"key": "b", "job": "j", "instance": 7}])
    assert [r["label"] for r in runs] == ["j · instance 7", "j · instance 7 (2)"]
    assert runs[0]["probes"]["combined"] == [[0, 8.0], [100, 15.0]] and runs[0]["loss"] == [[1, 0.5], [2, 0.4]]
    assert runs[1]["probes"]["combined"] == [] and runs[1]["probe_images"] == []
    series = webui.history_series(runs, "probes", ["combined"])
    assert [s["name"] for s in series] == ["j · instance 7"]


def test_a_chart_added_after_the_page_was_drawn_takes_the_pages_look():
    class Chart:
        def __init__(self):
            self.options = webui.chart_options({"type": "value"}, {"type": "value"}, [], [])
            self.updates = 0

        def update(self):
            self.updates += 1

    look = {"light": False}
    charts = webui.ChartList(look)
    dark = Chart()
    charts.append(dark)
    assert not getattr(dark, "_rz_light", False) and dark.updates == 0  # dark is how charts are built
    look["light"] = True
    light = Chart()
    charts.append(light)
    assert light._rz_light and light.updates == 1 and list(charts) == [dark, light]

    light.options["series"] = [{"name": "x", "itemStyle": {"color": "#F4F4F5"}, "lineStyle": {"color": "#60A5FA"},
                                "data": [[1, "#F4F4F5"]]}]
    before = light.updates
    charts.restyle(light)  # series set after the chart joined the list: dark colours, swapped now
    assert light.options["series"][0]["itemStyle"]["color"] == "#1C1C21"
    assert light.options["series"][0]["lineStyle"]["color"] == "#2563EB"
    assert light.options["series"][0]["data"] == [[1, "#F4F4F5"]] and light.updates == before + 1
    look["light"] = False
    dark.options["series"] = [{"itemStyle": {"color": "#F4F4F5"}}]
    charts.restyle(dark)
    assert dark.options["series"][0]["itemStyle"]["color"] == "#F4F4F5"
    charts.discard([light])
    assert list(charts) == [dark]


def test_a_chart_in_an_instance_card_makes_room_for_its_legends_second_row():
    class Chart:
        def __init__(self):
            self.options = webui.chart_options(webui._axis("step"), webui._axis("training minus held-out (points)"),
                                               [], ["face", "hair", "skin", "clothes", "background", "subject"])
            self.classes, self._rz_width = ["w-full", "h-72"], 1280

        def style(self, replace=""):
            self.height = replace

    wide, card = Chart(), Chart()
    card._rz_box = webui.FLEET_CARD_PLOT
    webui.fit_chart(wide)
    webui.fit_chart(card)
    assert wide.options["grid"]["top"] == 25 + 34  # one legend row across the page
    assert card.options["grid"]["top"] == 2 * 25 + 34 and card.height == "height: 313px"  # two rows in a card



def _report_folder(root, logs=False, probes=False, maps=False):
    folder = root / "7" / "job"
    folder.mkdir(parents=True, exist_ok=True)
    if logs:
        (folder / "config.yaml").write_text("training_maps: true\n", encoding="utf-8")
        (folder / "status.json").write_text("{}", encoding="utf-8")
    if probes:
        (folder / "probes").mkdir(exist_ok=True)
        (folder / "probes" / "step000100_face_1.png").write_bytes(b"x")
    if maps:
        (folder / "training_maps").mkdir(exist_ok=True)
        (folder / "training_maps" / "maps.jsonl").write_text("{}\n", encoding="utf-8")
    return folder


def test_a_report_downloads_only_the_kinds_the_folder_lacks(tmp_path):
    assert webui.fleet_report_kinds(None) == ["logs", "probes"]
    assert webui.fleet_report_kinds(tmp_path / "nope", {"config": {"training_maps": True}}) == \
        ["logs", "probes", "maps"]
    empty = _report_folder(tmp_path / "a")
    assert webui.fleet_report_kinds(empty) == ["logs", "probes"]
    logs_only = _report_folder(tmp_path / "b", logs=True)  # its config.yaml says the run had training maps
    assert webui.fleet_report_kinds(logs_only) == ["logs", "probes", "maps"]
    assert webui.fleet_report_missing(logs_only) == ["probes", "maps"]
    both = _report_folder(tmp_path / "c", logs=True, probes=True)
    assert webui.fleet_report_kinds(both) == ["logs", "maps"]
    plain = _report_folder(tmp_path / "d", probes=True)
    (plain / "config.yaml").write_text("training_maps: false\n", encoding="utf-8")
    (plain / "status.json").write_text("{}", encoding="utf-8")
    assert webui.fleet_report_kinds(plain) == ["logs"]  # logs always; nothing else is lacking
    assert webui.fleet_report_missing(plain) == []
    full = _report_folder(tmp_path / "e", logs=True, probes=True, maps=True)
    assert webui.fleet_report_missing(full) == []


class _ReportPage:
    """The page side of fleet_make_report: what it was told, downloaded and sent."""

    def __init__(self, pull_result, files=("report.png", "report.pdf"), fail=None, fill=None):
        self.pulls, self.notes, self.sent, self.made = [], [], [], []
        self.pull_result, self.files, self.fail, self.fill = pull_result, files, fail, fill

    async def pull(self, key, kinds):
        self.pulls.append((key, list(kinds)))
        if self.fill:
            self.fill()
        return self.pull_result

    async def make(self, folder):
        self.made.append(folder)
        if self.fail:
            raise self.fail
        return {p.split(".")[-1]: folder / "report" / p for p in self.files}

    def run(self, rec, root):
        import asyncio

        return asyncio.run(webui.fleet_make_report(rec, root, self.pull, self.make,
                                                   lambda text, kind: self.notes.append((kind, text)),
                                                   self.sent.append))


def test_a_report_downloads_then_makes_the_files_and_sends_them(tmp_path):
    root = tmp_path / "pulls"
    folder = root / "7" / "job"
    page = _ReportPage({"fetched": 3, "mb": 1.0}, fill=lambda: _report_folder(root, logs=True, probes=True))
    rec = {"key": "7/job", "instance": 7, "job": "job"}
    where = page.run(rec, root)
    # the folder did not exist yet; the downloaded config.yaml then said training maps, so they are fetched too
    assert page.pulls == [("7/job", ["logs", "probes"]), ("7/job", ["maps"])]
    assert page.made == [folder.resolve()] and where == folder.resolve() / "report"
    assert [p.name for p in page.sent] == ["report.png", "report.pdf"]
    kind, text = page.notes[-1]
    assert kind == "warning" and "missing: maps" in text  # its config.yaml says training maps, none came
    done = _ReportPage({"fetched": 0, "mb": 0})
    _report_folder(tmp_path / "x", logs=True, probes=True, maps=True)
    done.run({"key": "7/job", "instance": 7, "job": "job", "folder": str(tmp_path / "x" / "7" / "job")},
             tmp_path / "x")
    assert done.notes[-1][0] == "positive" and done.pulls[0][1] == ["logs"]


def test_a_report_of_a_gone_instance_is_made_from_what_is_there(tmp_path):
    root = tmp_path / "pulls"
    _report_folder(root, logs=True)
    page = _ReportPage({"error": "Instance 7 is gone: its files can no longer be downloaded."})
    rec = {"key": "7/job", "instance": 7, "job": "job", "config": {"training_maps": True}}
    assert page.run(rec, root) is not None
    assert len(page.made) == 1 and len(page.sent) == 2
    kind, text = page.notes[-1]
    assert kind == "warning" and "missing: probes, maps" in text and "is gone" in text


def test_a_report_without_any_folder_says_so_and_makes_nothing(tmp_path):
    page = _ReportPage({"error": "Instance 7 is gone: its files can no longer be downloaded."})
    assert page.run({"key": "7/job", "instance": 7, "job": "job"}, tmp_path / "pulls") is None
    assert page.made == [] and page.sent == []
    kind, text = page.notes[-1]
    assert kind == "warning" and "no downloaded folder" in text and "is gone" in text


def test_a_report_that_cannot_be_made_is_a_notice_not_a_crash(tmp_path):
    root = tmp_path / "pulls"
    _report_folder(root, logs=True, probes=True, maps=True)
    page = _ReportPage({"fetched": 0, "mb": 0}, fail=RuntimeError("matplotlib missing"))
    assert page.run({"key": "7/job", "instance": 7, "job": "job"}, root) is None
    assert page.sent == [] and page.notes[-1][0] == "negative" and "matplotlib missing" in page.notes[-1][1]


def test_a_failed_download_of_a_complete_folder_is_a_warning_with_the_reason(tmp_path):
    root = tmp_path / "pulls"
    _report_folder(root, logs=True, probes=True, maps=True)
    page = _ReportPage({"error": "Instance 7 is gone: its files can no longer be downloaded."})
    assert page.run({"key": "7/job", "instance": 7, "job": "job"}, root) is not None
    kind, text = page.notes[-1]
    assert kind == "warning" and "download failed" in text and "is gone" in text and "missing" not in text


def test_a_download_that_raises_still_gets_the_report_from_what_is_there(tmp_path):
    root = tmp_path / "pulls"
    _report_folder(root, logs=True, probes=True, maps=True)
    page = _ReportPage(None)

    async def boom(key, kinds):
        raise ValueError("odd failure")

    page.pull = boom
    assert page.run({"key": "7/job", "instance": 7, "job": "job"}, root) is not None
    assert len(page.made) == 1 and len(page.sent) == 2
    kind, text = page.notes[-1]
    assert kind == "warning" and "odd failure" in text


def test_the_report_guard_is_shared_by_every_page():
    assert isinstance(webui._REPORT_BUSY, set)  # module level: two browser pages see the same set
    import inspect

    assert "_REPORT_BUSY" in inspect.getsource(webui.build_fleet_tab) and "report_busy" not in         inspect.getsource(webui.build_fleet_tab)


def test_remove_plan_lists_runs_with_their_megabytes_and_skips_the_busy_ones(tmp_path):
    root = tmp_path / "pulls"
    folder = root / "7" / "done"
    folder.mkdir(parents=True)
    (folder / "f.bin").write_bytes(b"x" * 2_500_000)
    recs = [{"key": "k1", "job": "done", "instance": 7, "state": "ended", "folder": str(folder)},
            {"key": "k2", "job": "live", "instance": 7, "state": "running", "folder": str(folder)},
            {"key": "k3", "job": "pulling", "instance": 8, "state": "ended"},
            {"key": "k4", "job": "bare", "instance": 8, "state": "ended"},
            {"key": "k5", "job": "away", "instance": 8, "state": "ended", "folder": str(tmp_path)}]
    plan = webui.fleet_remove_plan(recs, root, lambda key: key == "k3")
    assert [p["skip"] for p in plan] == ["", "still going", "a download of it is going", "", ""]
    assert plan[0]["text"] == "done · instance 7: 2.5 MB downloaded" and plan[0]["mb"] == 2.5
    assert plan[1]["mb"] == 0.0 and "skipped, still going" in plan[1]["text"]
    assert plan[3]["text"].endswith("no downloaded files") and plan[4]["mb"] == 0.0  # outside the root: not measured


def test_remove_summary_names_the_megabytes_and_the_failures():
    done = [{"mb": 2.5, "folder": "a", "binned": True}, {"mb": 1.0, "folder": None, "binned": False}]
    assert webui.fleet_remove_summary(done, []) == "2 runs removed, 2.5 MB moved to the Recycle Bin."
    text = webui.fleet_remove_summary(done[:1], ["x · instance 1: still going"])
    assert text.startswith("1 run removed") and "Not removed: x · instance 1: still going" in text
    assert "no Recycle Bin" in webui.fleet_remove_summary([{"mb": 3.0, "folder": "a", "binned": False}], [])
