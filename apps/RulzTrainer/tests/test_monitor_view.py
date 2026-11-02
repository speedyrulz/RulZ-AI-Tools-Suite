"""The Monitor's image galleries (samples and probes by step, a race's configurations side by side) and its charts'
fit to the browser's width (the legend wraps; the plot starts below it and the chart grows on a phone)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402


def score(img, face, body=None):
    return {"image": img, "face_norm": face, "body_norm": body,
            "combined_norm": face if body is None else 0.7 * face + 0.3 * body}


def test_samples_of_one_run_by_step():
    samples = [f"out/samples/step{s:06d}_{i}.png" for s in (200, 100) for i in (1, 0)]
    scores = [score("x/step000200_0.png", 80, 60), score("x/step000200_1.png", 60)]
    steps = webui.sample_steps(samples, scores, [])
    assert sorted(steps) == [100, 200]
    summary, items = steps[200]
    assert [it[2] for it in items] == ["Prompt 1", "Prompt 2"]  # in prompt order
    assert items[0][3] == "face 80% · body 60% · combined 74%"
    assert items[1][3] == "face 60% · combined 60%"
    assert summary.startswith("Average of this step's samples: face 70%")
    assert steps[100] == ("", [("", samples[3], "Prompt 1", ""), ("", samples[2], "Prompt 2", "")])


def test_samples_of_a_race_side_by_side():
    names = ["lora32", "lokr8"]
    samples = [f"s/{n}_step000100_{i}.png" for i in (1, 0) for n in names]
    _, items = webui.sample_steps(samples, [], names)[100]
    assert [(g, who) for g, _, who, _ in items] == [("Sample 1", "lora32"), ("Sample 1", "lokr8"),
                                                    ("Sample 2", "lora32"), ("Sample 2", "lokr8")]


def test_probes_of_one_run_and_of_a_race():
    rec = {"step": 300, "images": ["p/step000300_face_0.png", "p/step000300_body_0.png"], "combined": 50.0,
           "face_norm": 60.0, "body_norm": 30.0}
    summary, items = webui.probe_steps([rec], [])[300]
    assert summary == "This probe: combined 50% · face 60% · body 30%"
    assert [it[2] for it in items] == ["Face probe 1", "Body probe 1"]
    names = ["a", "b"]
    recs = [{**rec, "racer": n, "images": [f"p/{n}_step000300_face_0.png", f"p/{n}_step000300_face_1.png"]}
            for n in reversed(names)]
    _, items = webui.probe_steps(recs, names)[300]
    assert [(g, who) for g, _, who, _ in items] == [("Face probe 1", "a"), ("Face probe 1", "b"),
                                                    ("Face probe 2", "a"), ("Face probe 2", "b")]
    assert webui.probe_steps([{"step": 5, "images": []}], []) == {}


class FakeChart:
    def __init__(self, legend, width, height_class="h-60"):
        self.options = webui.chart_options(webui._axis("step"), webui._axis("likeness %"), [], legend)
        self._rz_width = width
        self.classes = ["w-full", height_class]
        self.styled = None

    def style(self, replace=""):
        self.styled = replace


def test_charts_fit_the_width():
    names = [f"config{i} combined" for i in range(9)]
    wide, narrow = FakeChart(names, 1400), FakeChart(names, 375)
    webui.fit_chart(wide)
    webui.fit_chart(narrow)
    assert wide.options["legend"]["type"] == "plain"  # wraps instead of a pager
    assert narrow.options["grid"]["top"] > wide.options["grid"]["top"]  # more legend rows on a phone
    assert narrow.options["legend"]["textStyle"]["fontSize"] < wide.options["legend"]["textStyle"]["fontSize"]
    assert narrow.styled.startswith("height: ") and int(narrow.styled[8:-2]) > 240  # grows by the extra rows
    short = FakeChart(["face", "body"], 375)
    webui.fit_chart(short)
    assert short.styled == "" and short.options["grid"]["top"] == 25 + 34
    assert webui.short_scores({"percent": 40.0}) == "face 40%"
    assert webui.short_scores({"face_norm": None, "body_norm": 20.0}) == "no face · body 20%"
