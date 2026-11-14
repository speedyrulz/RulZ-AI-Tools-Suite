from __future__ import annotations

import json

import pytest

from rulz_comfy_launcher import APP_NAME
from rulz_comfy_launcher.model import (
    CONFIG_VERSION, GPU_AUTO, GPU_CPU, GPU_SELECT, ConfigStore, GpuRef, OptionState,
    Profile, Settings, default_config_dir,
)


# -- OptionState / GpuRef ----------------------------------------------------
def test_option_state_round_trip():
    s = OptionState(enabled=True, value="5")
    assert OptionState.from_dict(s.to_dict()) == s


@pytest.mark.parametrize("raw, expected", [
    (True, OptionState(True, "")),
    (False, OptionState(False, "")),
    (None, OptionState()),
    ("text", OptionState()),
    ([1, 2], OptionState()),
    ({"enabled": 1, "value": 3}, OptionState(True, "3")),
    ({"value": None}, OptionState(False, "")),
    ({}, OptionState()),
])
def test_option_state_from_dict_tolerant(raw, expected):
    assert OptionState.from_dict(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("GPU-1", GpuRef("GPU-1")),
    ("", None),
    ({"uuid": "GPU-2", "name": "RTX"}, GpuRef("GPU-2", "RTX")),
    ({"uuid": "GPU-3"}, GpuRef("GPU-3", "")),
    ({"uuid": ""}, None),
    ({"name": "x"}, None),
    (None, None),
    (5, None),
])
def test_gpu_ref_from_dict(raw, expected):
    assert GpuRef.from_dict(raw) == expected


def test_gpu_ref_null_name_is_empty():
    assert GpuRef.from_dict({"uuid": "GPU-1", "name": None}).name == ""


# -- Profile -----------------------------------------------------------------
def full_profile() -> Profile:
    return Profile(
        name="Main", distro="Ubuntu", user="comfy", comfy_dir="/opt/comfy", python="python3",
        port=8190, listen="0.0.0.0", gpu_mode=GPU_SELECT,
        gpus=[GpuRef("GPU-a", "A"), GpuRef("GPU-b", "B")],
        options={"--lowvram": OptionState(True), "--preview-size": OptionState(False, "256")},
        extra_args="--foo bar", env=[["A", "1"], ["B", "$PATH"]],
        pre_commands="echo hi", auto_open=True, checked=True,
    )


def test_profile_round_trip():
    p = full_profile()
    again = Profile.from_dict(json.loads(json.dumps(p.to_dict())))
    assert again == p


def test_profile_to_dict_drops_empty_options():
    p = Profile(options={"--a": OptionState(), "--b": OptionState(False, "x"),
                         "--c": OptionState(True)})
    assert set(p.to_dict()["options"]) == {"--b", "--c"}


def test_profile_defaults_and_new_ids():
    a, b = Profile(), Profile()
    assert a.id != b.id and len(a.id) == 12
    assert a.port == 8188 and a.gpu_mode == GPU_AUTO and a.comfy_dir == "~/ComfyUI"


@pytest.mark.parametrize("raw", [None, [], "x", 5])
def test_profile_from_non_dict_gives_default(raw):
    p = Profile.from_dict(raw)
    assert p.name == "ComfyUI" and p.port == 8188


def test_profile_from_dict_tolerant_fields():
    p = Profile.from_dict({
        "id": "", "name": None, "comfy_dir": "", "python": None,
        "port": "nope", "listen": "", "gpu_mode": "weird",
        "gpus": ["GPU-1", {"uuid": "GPU-1", "name": "dup"}, {"bad": 1}, "", {"uuid": "GPU-2"}],
        "options": ["not", "a", "dict"],
        "env": [["A", "1"], ["too", "many", "items"], "x", ["B", 2]],
        "auto_open": 1, "checked": 0,
    })
    assert p.id and len(p.id) == 12
    assert p.name == "ComfyUI" and p.comfy_dir == "~/ComfyUI" and p.python == "venv/bin/python"
    assert p.port == 8188 and p.listen == "127.0.0.1" and p.gpu_mode == GPU_AUTO
    assert [g.uuid for g in p.gpus] == ["GPU-1", "GPU-2"]   # duplicates dropped, order kept
    assert p.options == {}
    assert p.env == [["A", "1"], ["B", "2"]]
    assert p.auto_open is True and p.checked is False


def test_profile_from_dict_port_string_number():
    assert Profile.from_dict({"port": "8200"}).port == 8200
    assert Profile.from_dict({"port": None}).port == 8188


@pytest.mark.parametrize("mode", [GPU_AUTO, GPU_SELECT, GPU_CPU])
def test_profile_gpu_modes_kept(mode):
    assert Profile.from_dict({"gpu_mode": mode}).gpu_mode == mode


def test_profile_from_dict_infinite_port():
    assert Profile.from_dict({"port": float("inf")}).port == 8188


def test_profile_from_dict_gpus_string_not_split_into_chars():
    p = Profile.from_dict({"gpu_mode": "select", "gpus": "GPU-1"})
    assert [g.uuid for g in p.gpus] in ([], ["GPU-1"])


def test_clone_and_snapshot_are_deep_copies():
    p = full_profile()
    c = p.clone("Copy")
    assert c.id != p.id and c.name == "Copy"
    c.gpus.append(GpuRef("GPU-z"))
    c.options["--lowvram"].enabled = False
    c.env[0][1] = "changed"
    assert len(p.gpus) == 2 and p.options["--lowvram"].enabled and p.env[0][1] == "1"
    s = p.snapshot()
    assert s == p and s is not p
    s.options["--lowvram"].value = "x"
    assert p.options["--lowvram"].value == ""


# -- Settings ----------------------------------------------------------------
def test_settings_round_trip():
    s = Settings(stagger_seconds=1.5, stop_timeout=20, log_max_lines=1000,
                 last_profile="abc", geometry="AAAA", splitter=[100, 200])
    assert Settings.from_dict(s.to_dict()) == s


@pytest.mark.parametrize("data, field, expected", [
    ({"stagger_seconds": -5}, "stagger_seconds", 0.0),
    ({"stagger_seconds": 1000}, "stagger_seconds", 120.0),
    ({"stagger_seconds": "x"}, "stagger_seconds", 3.0),
    ({"stagger_seconds": None}, "stagger_seconds", 3.0),
    ({"stop_timeout": 0}, "stop_timeout", 1),
    ({"stop_timeout": 9999}, "stop_timeout", 300),
    ({"stop_timeout": "15"}, "stop_timeout", 15),
    ({"log_max_lines": 10}, "log_max_lines", 200),
    ({"log_max_lines": 10**9}, "log_max_lines", 200000),
    ({"splitter": [1, 2.7, "x", None]}, "splitter", [1, 2]),
    ({"splitter": "nope"}, "splitter", []),
    ({"last_profile": None}, "last_profile", ""),
])
def test_settings_clamping_and_tolerance(data, field, expected):
    assert getattr(Settings.from_dict(data), field) == expected


@pytest.mark.parametrize("raw", [None, [], "x"])
def test_settings_from_non_dict(raw):
    assert Settings.from_dict(raw) == Settings()


def test_settings_infinite_stop_timeout():
    assert Settings.from_dict({"stop_timeout": float("inf")}).stop_timeout in (10, 300)


def test_settings_nan_in_splitter():
    assert Settings.from_dict({"splitter": [float("nan"), 5]}).splitter == [5]


# -- ConfigStore -------------------------------------------------------------
def test_default_config_dir_uses_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert default_config_dir() == tmp_path / APP_NAME
    assert ConfigStore().path == tmp_path / APP_NAME / "config.json"


def test_load_missing_file_gives_one_default_profile(tmp_path):
    store = ConfigStore(tmp_path / "config.json")
    store.load()
    assert len(store.profiles) == 1 and store.load_error == ""
    assert not (tmp_path / "config.json").exists()


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "sub" / "config.json"
    store = ConfigStore(path)
    store.profiles = [full_profile(), Profile(name="Zweite é")]
    store.settings = Settings(stop_timeout=42, splitter=[1, 2])
    store.save()
    assert path.exists() and not path.with_name("config.json.tmp").exists()
    raw = path.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
    data = json.loads(raw.decode("utf-8"))
    assert data["version"] == CONFIG_VERSION
    assert "é" in raw.decode("utf-8")   # ensure_ascii=False

    other = ConfigStore(path)
    other.load()
    assert other.load_error == ""
    assert other.profiles == store.profiles
    assert other.settings == store.settings


@pytest.mark.parametrize("content", ["{not json", "[1, 2, 3]", "\"text\"", ""])
def test_load_unreadable_file_kept_as_backup(tmp_path, content):
    path = tmp_path / "config.json"
    path.write_text(content, encoding="utf-8")
    store = ConfigStore(path)
    store.load()
    assert len(store.profiles) == 1
    assert store.load_error
    assert not path.exists()
    backups = list(tmp_path.glob("config.unreadable-*.json"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == content
    assert backups[0].name in store.load_error
    # Saving afterwards starts a new file and leaves the backup alone.
    store.save()
    assert path.exists() and backups[0].read_text(encoding="utf-8") == content


def test_load_invalid_utf8_is_backed_up(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"\xff\xfe{garbage")
    store = ConfigStore(path)
    store.load()
    assert store.load_error and list(tmp_path.glob("config.unreadable-*.json"))


def test_load_duplicate_ids_get_new_ids(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"profiles": [
        {"id": "same", "name": "A"}, {"id": "same", "name": "B"}, {"id": "other", "name": "C"},
    ]}), encoding="utf-8")
    store = ConfigStore(path)
    store.load()
    ids = [p.id for p in store.profiles]
    assert [p.name for p in store.profiles] == ["A", "B", "C"]
    assert ids[0] == "same" and ids[2] == "other"
    assert len(set(ids)) == 3


def test_load_empty_profiles_gives_default(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"settings": {"stop_timeout": 7}, "profiles": []}), encoding="utf-8")
    store = ConfigStore(path)
    store.load()
    assert len(store.profiles) == 1 and store.settings.stop_timeout == 7


def test_load_resets_previous_state(tmp_path):
    store = ConfigStore(tmp_path / "config.json")
    store.profiles = [Profile(), Profile()]
    store.load_error = "old"
    store.load()
    assert len(store.profiles) == 1 and store.load_error == ""


def test_load_huge_number_does_not_crash(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"settings": {"stop_timeout": 1e999}, "profiles": [{"port": 1e999}]}',
                    encoding="utf-8")
    store = ConfigStore(path)
    store.load()
    assert len(store.profiles) == 1


def test_find_add_remove():
    store = ConfigStore("unused.json")
    a, b, c = Profile(name="a"), Profile(name="b"), Profile(name="c")
    store.add(a)
    store.add(b)
    store.add(c, after=a)
    assert [p.name for p in store.profiles] == ["a", "c", "b"]
    store.add(Profile(name="d"), after=Profile(name="not there"))
    assert store.profiles[-1].name == "d"
    assert store.find(c.id) is c and store.find("missing") is None
    store.remove(c)
    store.remove(Profile())   # not present: no error
    assert [p.name for p in store.profiles] == ["a", "b", "d"]


def test_unique_name():
    store = ConfigStore("unused.json")
    assert store.unique_name("X") == "X"
    store.profiles = [Profile(name="X"), Profile(name="X (2)"), Profile(name="X (4)")]
    assert store.unique_name("X") == "X (3)"
    assert store.unique_name("Y") == "Y"


def test_free_port():
    store = ConfigStore("unused.json")
    store.profiles = [Profile(port=8188), Profile(port=8189), Profile(port=8191)]
    assert store.free_port() == 8190
    assert store.free_port(8191) == 8192
    assert store.free_port(9000) == 9000
    store.profiles = [Profile(port=65535)]
    assert store.free_port(65535) == 65535   # gives up at the top of the range
