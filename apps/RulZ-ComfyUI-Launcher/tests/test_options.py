from __future__ import annotations

import json

import pytest

from rulz_comfy_launcher.options import (
    CATEGORY_ORDER, MANAGED_OPTIONS, UNKNOWN_CATEGORY, Catalog, OptionSpec, builtin_catalog,
    categorize, parse_introspection_output, unknown_spec,
)


# -- builtin catalog ---------------------------------------------------------
def test_builtin_catalog_loads(catalog):
    assert catalog.version == "0.45.0"
    assert catalog.source == "built-in list"
    assert catalog.label() == "ComfyUI 0.45.0 (built-in list)"
    assert len(catalog.options) > 50
    names = [o.name for o in catalog.options]
    assert len(names) == len(set(names))
    assert all(n.startswith("--") for n in names)


def test_builtin_catalog_kinds_and_categories(catalog):
    for spec in catalog.options:
        assert spec.kind in ("flag", "value", "optional", "multi")
        assert spec.type in ("str", "int", "float", "path")
        assert spec.category in CATEGORY_ORDER
        assert spec.known


def test_builtin_catalog_known_specs(catalog):
    assert catalog.get("--lowvram").kind == "flag"
    ps = catalog.get("--preview-method")
    assert ps.kind == "value" and ps.choices == ("none", "auto", "latent2rgb", "taesd")
    assert catalog.get("--preview-size").type == "int"
    assert catalog.get("--feature-flag").repeat is True
    assert catalog.get("--fast").kind == "multi"
    assert catalog.get("--async-offload").kind == "optional"
    assert catalog.get("--nope") is None


def test_visible_hides_managed_options(catalog):
    visible = {o.name for o in catalog.visible()}
    for name in MANAGED_OPTIONS:
        assert name not in visible
    assert {"--port", "--listen", "--cuda-device", "--cpu"} <= {o.name for o in catalog.options}
    assert "--lowvram" in visible
    assert len(visible) == len([o for o in catalog.options if o.name not in MANAGED_OPTIONS])


def test_exclusive_groups(catalog):
    assert set(catalog.exclusive_with("--lowvram")) == {"--gpu-only", "--highvram", "--novram", "--cpu"}
    assert catalog.exclusive_with("--force-fp16") == ["--force-fp32"]
    assert catalog.exclusive_with("--verbose") == []
    assert catalog.group_index("--lowvram") == catalog.group_index("--cpu")
    assert catalog.group_index("--verbose") is None


def test_every_builtin_group_member_exists(catalog):
    for group in catalog.groups:
        for name in group:
            assert catalog.get(name) is not None, name


def test_builtin_catalog_returns_fresh_object():
    assert builtin_catalog() is not builtin_catalog()


# -- Catalog.from_data -------------------------------------------------------
def test_from_data_skips_bad_and_duplicate_entries():
    cat = Catalog.from_data({
        "version": 1.5,
        "options": [
            {"name": "--a"}, {"name": "--a", "kind": "value"}, "junk", {"kind": "flag"},
            {"name": ""}, {"name": "--b", "kind": "value"},
        ],
        "groups": [["--a", "--b"], ["--solo"], "notalist", [1, 2]],
    }, source="test")
    assert [o.name for o in cat.options] == ["--a", "--b"]
    assert cat.get("--a").kind == "flag"          # first one wins
    assert cat.groups == [["--a", "--b"], ["1", "2"]]
    assert cat.version == "1.5" and cat.source == "test"
    assert cat.label() == "ComfyUI 1.5 (test)"


def test_from_data_empty():
    cat = Catalog.from_data({})
    assert cat.options == [] and cat.groups == [] and cat.label() == "built-in"
    assert cat.visible() == []


# -- OptionSpec.from_dict ----------------------------------------------------
def test_option_spec_from_dict_normalises():
    spec = OptionSpec.from_dict({"name": "--x-vae", "kind": "bogus", "type": "complex",
                                 "choices": [1, 2], "min_values": "nope", "metavar": None,
                                 "help": None, "repeat": 1})
    assert spec.kind == "flag" and spec.type == "str"
    assert spec.choices == ("1", "2")
    assert spec.min_values == 0 and spec.metavar == "" and spec.help == ""
    assert spec.repeat is True
    assert spec.category == "Precision"
    assert spec.known is True


def test_option_spec_from_dict_empty_choices_is_none():
    assert OptionSpec.from_dict({"name": "--x", "choices": []}).choices is None
    assert OptionSpec.from_dict({"name": "--x", "min_values": "2"}).min_values == 2


@pytest.mark.parametrize("default, text", [
    (None, ""), (False, ""), ([], ""), ("", ""),
    (512, "512"), (100.0, "100"), (0.5, "0.5"), (0, "0"),
    (["a", 1], "a 1"), ("sha256", "sha256"), (True, "True"),
])
def test_default_text(default, text):
    assert OptionSpec(name="--x", default=default).default_text() == text


# -- categorize --------------------------------------------------------------
@pytest.mark.parametrize("name, category", [
    ("--lowvram", "Memory and VRAM"),
    ("--cpu", "Devices"),
    ("--listen", "Network"),
    ("--bf16-unet", "Precision"),
    ("--new-thing-vae", "Precision"),
    ("--foo-text-enc", "Precision"),
    ("--weird-fp16-mode", "Precision"),
    ("--super-attention", "Attention"),
    ("--no-xformers-ever", "Attention"),
    ("--more-vram", "Memory and VRAM"),
    ("--cache-everything", "Memory and VRAM"),
    ("--swap-memory", "Memory and VRAM"),
    ("--offload-all", "Memory and VRAM"),
    ("--logs-directory", "Folders"),
    ("--some-path", "Folders"),
    ("--web-root", "Folders"),
    ("--manager-thing", "Custom nodes and manager"),
    ("--extra-nodes", "Custom nodes and manager"),
    ("--preview-format", "Previews"),
    ("--cuda-stream", "Devices"),
    ("--tls-ca", "Network"),
    ("--debug-xyz", "Logging and debugging"),
    ("--LOG-LEVEL", "Logging and debugging"),
    ("--totally-new", "Other"),
])
def test_categorize(name, category):
    assert categorize(name) == category


def test_unknown_spec():
    flag = unknown_spec("--gone", has_value=False)
    assert flag.kind == "flag" and flag.known is False and flag.category == UNKNOWN_CATEGORY
    assert flag.help
    val = unknown_spec("--gone", has_value=True)
    assert val.kind == "optional" and val.known is False


# -- parse_introspection_output ----------------------------------------------
def payload(**extra):
    data = {"version": "1.0", "options": [{"name": "--a"}], "groups": []}
    data.update(extra)
    return "RCL_JSON:" + json.dumps(data)


def test_parse_introspection_finds_payload_among_noise():
    text = "Some warning\r\nTotal VRAM 24GB\r\n" + payload() + "\r\ntrailing\n"
    data = parse_introspection_output(text)
    assert data["version"] == "1.0" and data["options"] == [{"name": "--a"}]
    cat = Catalog.from_data(data, source="ComfyUI")
    assert cat.get("--a") is not None


@pytest.mark.parametrize("text", [
    "",
    "nothing here\n",
    "RCL_JSON:{broken\n",
    "RCL_JSON:[1,2]\n",
    "RCL_JSON:" + json.dumps({"options": []}),
    "RCL_JSON:" + json.dumps({"options": "x"}),
    "RCL_JSON:" + json.dumps({"version": "1"}),
    "  " + payload(),   # must start the line
])
def test_parse_introspection_rejects(text):
    assert parse_introspection_output(text) is None


def test_parse_introspection_uses_first_marker_line():
    assert parse_introspection_output("RCL_JSON:{bad\n" + payload()) is None
    first = payload(version="first")
    assert parse_introspection_output(first + "\n" + payload(version="second"))["version"] == "first"
