"""The light look: chart colors swap to their light versions and back without loss, data is left alone, and the
theme stylesheet defines every variable it uses in both looks."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402


def test_chart_color_tables_are_disjoint_and_invertible():
    assert not set(webui.CHART_LIGHT) & set(webui.CHART_LIGHT.values())
    assert len(set(webui.CHART_LIGHT.values())) == len(webui.CHART_LIGHT)


def test_recolor_round_trip_keeps_data():
    opts = webui.chart_options(webui._axis("step"), webui._axis("loss"),
                               [webui._line("raw", "#52525B"), webui._line("body", "#F4F4F5")], ["raw", "body"])
    opts["series"][0]["data"] = [[1, 0.5], [2, 0.4]]
    light = webui._recolor(opts, webui.CHART_LIGHT)
    assert light["series"][1]["lineStyle"]["color"] == webui.CHART_LIGHT["#F4F4F5"]
    assert light["series"][0]["data"] == [[1, 0.5], [2, 0.4]]
    assert webui._recolor(light, webui.CHART_DARK) == opts


def test_css_variables_defined_for_both_looks():
    css = webui.THEME_CSS
    used = set(re.findall(r"var\((--rz-[\w-]+)\)", css))
    dark_block = css[css.index(":root {"):css.index("}", css.index(":root {"))]
    light_block = css[css.index('html[data-theme="light"] {'):]
    light_block = light_block[:light_block.index("}")]
    for name in used:
        assert f"{name}:" in dark_block, name
    light_names = set(re.findall(r"(--rz-[\w-]+):", light_block))
    assert light_names <= set(re.findall(r"(--rz-[\w-]+):", dark_block))
    # every themed color except the brand reds is redefined for the light look
    assert set(re.findall(r"(--rz-[\w-]+):", dark_block)) - light_names == {"--rz-red", "--rz-red-bright",
                                                                           "--rz-red-soft"}
