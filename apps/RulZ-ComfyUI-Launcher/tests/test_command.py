from __future__ import annotations

import shlex

import pytest

from rulz_comfy_launcher.command import (
    ENV_DUMP_SCRIPT, EXIT_NO_FOLDER, Issue, PID_PATTERN, READY_PATTERN, build_introspection_script,
    build_launch_script, diff_shell_env, head_args, instance_url, launch_signature,
    option_args, parse_env_dump, preview_command, quote_env_value, quote_path, split_words,
    tail_args, validate, validate_together,
)
from rulz_comfy_launcher.model import GPU_AUTO, GPU_CPU, GPU_SELECT, GpuRef, OptionState, Profile
from rulz_comfy_launcher.options import OptionSpec


def on(value=""):
    return OptionState(enabled=True, value=value)


def spec(name="--x", **kw):
    return OptionSpec(name=name, **kw)


def errors(issues):
    return [i.message for i in issues if i.level == "error"]


def warnings(issues):
    return [i.message for i in issues if i.level == "warning"]


# --------------------------------------------------------------------------
# quoting
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path, quoted", [
    ("~", '"$HOME"'),
    ("  ~  ", '"$HOME"'),
    ("~/", '"$HOME"/'),
    ("~/ComfyUI", '"$HOME"/ComfyUI'),
    ("~/my comfy", '"$HOME"/\'my comfy\''),
    ("/opt/comfy", "/opt/comfy"),
    ("/opt/my comfy", "'/opt/my comfy'"),
    ("it's", "'it'\"'\"'s'"),
    ("~user/x", "'~user/x'"),
    ("$HOME/x", "'$HOME/x'"),
    ("", "''"),
])
def test_quote_path(path, quoted):
    assert quote_path(path) == quoted


@pytest.mark.parametrize("value, quoted", [
    ("plain", '"plain"'),
    ("", '""'),
    ("$CUDA_HOME/bin:$PATH", '"$CUDA_HOME/bin:$PATH"'),
    ("${HOME}/x", '"${HOME}/x"'),
    ("$_private", '"$_private"'),
    ("a\"b", '"a\\"b"'),
    ("a`b`", '"a\\`b\\`"'),
    ("a\\b", '"a\\\\b"'),
    ("\\$HOME", '"\\$HOME"'),
    ("$(rm -rf /)", '"\\$(rm -rf /)"'),
    ("cost $5", '"cost \\$5"'),
    ("end$", '"end\\$"'),
    ("$$", '"\\$\\$"'),
    ("trailing\\", '"trailing\\\\"'),
])
def test_quote_env_value(value, quoted):
    assert quote_env_value(value) == quoted


def test_split_words():
    assert split_words("a 'b c' \"d e\" f\\ g") == ["a", "b c", "d e", "f g"]
    with pytest.raises(ValueError):
        split_words("a 'b")


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sp, value, expected", [
    (spec(kind="flag"), "ignored", ["--x"]),
    (spec(kind="value"), " 5 ", ["--x", "5"]),
    (spec(kind="value"), "-1", ["--x=-1"]),
    (spec(kind="value"), "a b", ["--x", "a b"]),
    (spec(kind="value", repeat=True), "a 'b c' -d", ["--x", "a", "--x", "b c", "--x=-d"]),
    (spec(kind="value", repeat=True), "", []),
    (spec(kind="optional"), "", ["--x"]),
    (spec(kind="optional"), "  ", ["--x"]),
    (spec(kind="optional"), "3", ["--x", "3"]),
    (spec(kind="optional"), "-3", ["--x=-3"]),
    (spec(kind="multi"), "", ["--x"]),
    (spec(kind="multi"), "a 'b c'", ["--x", "a", "b c"]),
    (spec(kind="multi"), "a 'b", ["--x", "a", "'b"]),   # unclosed quote: plain split
])
def test_option_args(sp, value, expected):
    assert option_args(sp, OptionState(True, value)) == expected


def test_head_args():
    assert head_args(Profile(port=8190)) == ["--port", "8190"]
    assert head_args(Profile(listen=" 127.0.0.1 ")) == ["--port", "8188"]
    assert head_args(Profile(listen="")) == ["--port", "8188"]
    assert head_args(Profile(listen="0.0.0.0,::")) == ["--port", "8188", "--listen", "0.0.0.0,::"]


def test_tail_args_order_and_filters(catalog):
    p = Profile(gpu_mode=GPU_CPU, options={
        "--preview-method": on("taesd"),
        "--lowvram": on(),
        "--highvram": OptionState(False, ""),          # disabled
        "--port": on("9999"),                          # managed: skipped
        "--cuda-device": on("1"),                      # managed: skipped
        "--zzz-unknown": on(),
        "--aaa-unknown": on("val"),
        "--off-unknown": OptionState(False, "x"),
    }, extra_args="--foo 'bar baz'")
    args = tail_args(p, catalog)
    assert args[0] == "--cpu"
    assert "--port" not in args and "--cuda-device" not in args and "--highvram" not in args
    # catalog options follow catalog order
    cat_names = [o.name for o in catalog.options]
    assert cat_names.index("--preview-method") < cat_names.index("--lowvram")
    assert args[1:4] == ["--preview-method", "taesd", "--lowvram"]
    # unknown options sorted by name, after catalog options, then extra args
    assert args[4:] == ["--aaa-unknown", "val", "--zzz-unknown", "--foo", "bar baz"]


def test_tail_args_no_cpu_for_other_modes(catalog):
    assert tail_args(Profile(gpu_mode=GPU_AUTO), catalog) == []
    assert tail_args(Profile(gpu_mode=GPU_SELECT, gpus=[GpuRef("G")]), catalog) == []


def test_preview_command(catalog):
    p = Profile(python="venv/bin/python", port=8190, gpu_mode=GPU_SELECT,
                gpus=[GpuRef("GPU-b"), GpuRef("GPU-a"), GpuRef("GPU-x")],
                options={"--lowvram": on()}, extra_args="--name 'a b'")
    text = preview_command(p, catalog, {"GPU-a": 0, "GPU-b": 1})
    assert text.startswith("CUDA_DEVICE_ORDER=PCI_BUS_ID ")
    words = shlex.split(text[len("CUDA_DEVICE_ORDER=PCI_BUS_ID "):])
    assert words == ["venv/bin/python", "main.py", "--port", "8190",
                     "--cuda-device", "1,0,?", "--lowvram", "--name", "a b"]


def test_preview_command_auto_and_cpu(catalog):
    assert preview_command(Profile(python=" "), catalog) == "python main.py --port 8188"
    assert preview_command(Profile(gpu_mode=GPU_CPU), catalog) == \
        "venv/bin/python main.py --port 8188 --cpu"
    # select mode without GPUs adds nothing (validate reports it)
    assert "--cuda-device" not in preview_command(Profile(gpu_mode=GPU_SELECT), catalog)


@pytest.mark.parametrize("listen, port, opts, url", [
    ("127.0.0.1", 8188, {}, "http://127.0.0.1:8188"),
    ("", 8188, {}, "http://127.0.0.1:8188"),
    ("0.0.0.0", 9000, {}, "http://127.0.0.1:9000"),
    ("::", 9000, {}, "http://127.0.0.1:9000"),
    ("*", 9000, {}, "http://127.0.0.1:9000"),
    ("0.0.0.0,::", 9000, {}, "http://127.0.0.1:9000"),
    ("192.168.1.5, 10.0.0.1", 8188, {}, "http://192.168.1.5:8188"),
    ("::1", 8188, {}, "http://[::1]:8188"),
    ("[::1]", 8188, {}, "http://[::1]:8188"),
    ("localhost", 8188, {"--tls-keyfile": on("k"), "--tls-certfile": on("c")},
     "https://localhost:8188"),
    ("localhost", 8188, {"--tls-keyfile": on("k")}, "http://localhost:8188"),
    ("localhost", 8188, {"--tls-keyfile": on("k"), "--tls-certfile": on(" ")},
     "http://localhost:8188"),
    ("localhost", 8188, {"--tls-keyfile": on("k"), "--tls-certfile": OptionState(False, "c")},
     "http://localhost:8188"),
])
def test_instance_url(listen, port, opts, url):
    assert instance_url(Profile(listen=listen, port=port, options=opts)) == url


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_validate_default_profile_is_clean(catalog):
    assert validate(Profile(), catalog) == []


def test_validate_basic_fields(catalog):
    p = Profile(name=" ", comfy_dir="", python=" ", port=70000, listen="1.2.3.4 5.6.7.8",
                gpu_mode=GPU_SELECT)
    errs = errors(validate(p, catalog))
    assert len(errs) == 6
    assert any("name" in e for e in errs)
    assert any("folder" in e for e in errs)
    assert any("Python" in e for e in errs)
    assert any("70000" in e for e in errs)
    assert any("listen" in e for e in errs)
    assert any("GPU" in e for e in errs)


@pytest.mark.parametrize("port, ok", [(0, False), (1, True), (65535, True), (65536, False), (-1, False)])
def test_validate_port_range(catalog, port, ok):
    assert (errors(validate(Profile(port=port), catalog)) == []) == ok


@pytest.mark.parametrize("name, value, problem", [
    ("--preview-size", "", "needs a value"),
    ("--preview-size", "abc", "not a whole number"),
    ("--preview-size", "1.5", "not a whole number"),
    ("--preview-size", "512", None),
    ("--reserve-vram", "1.5", None),
    ("--reserve-vram", "lots", "not a number"),
    ("--preview-method", "taesd", None),
    ("--preview-method", "fancy", "not one of"),
    ("--async-offload", "", None),             # optional: empty is fine
    ("--async-offload", "x", "not a whole number"),
    ("--async-offload", "4", None),
    ("--fast", "", None),                      # multi, min 0
    ("--fast", "autotune cublas_ops", None),
    ("--fast", "autotune bogus", "not one of"),
    ("--fast", "'autotune", "unclosed quote"),
    ("--whitelist-custom-nodes", "", "at least 1 value"),
    ("--whitelist-custom-nodes", "a b", None),
    ("--cache-ram", "4 x", "not a number"),
    ("--feature-flag", "", "at least 1 value"),    # value + repeat
    ("--feature-flag", "a=1 b", None),
    ("--lowvram", "anything", None),           # flag ignores the value
])
def test_validate_option_values(catalog, name, value, problem):
    errs = errors(validate(Profile(options={name: on(value)}), catalog))
    if problem is None:
        assert errs == []
    else:
        assert len(errs) == 1 and problem in errs[0] and name in errs[0]


def test_validate_disabled_option_not_checked(catalog):
    assert validate(Profile(options={"--preview-size": OptionState(False, "abc")}), catalog) == []


def test_validate_unknown_option_warns(catalog):
    issues = validate(Profile(options={"--from-the-future": on()}), catalog)
    assert errors(issues) == [] and len(warnings(issues)) == 1
    assert "--from-the-future" in warnings(issues)[0]


def test_validate_managed_option_ignored(catalog):
    assert validate(Profile(options={"--port": on("abc")}), catalog) == []


def test_validate_exclusive_groups(catalog):
    errs = errors(validate(Profile(options={"--lowvram": on(), "--highvram": on()}), catalog))
    assert len(errs) == 1 and "--lowvram" in errs[0] and "--highvram" in errs[0]
    errs = errors(validate(Profile(gpu_mode=GPU_CPU, options={"--novram": on()}), catalog))
    assert len(errs) == 1 and "--cpu" in errs[0] and "--novram" in errs[0]
    assert errors(validate(Profile(gpu_mode=GPU_CPU), catalog)) == []
    assert errors(validate(Profile(options={"--lowvram": on(), "--force-fp16": on()}), catalog)) == []


def test_validate_extra_args(catalog):
    errs = errors(validate(Profile(extra_args="--x 'oops"), catalog))
    assert errs == ["The extra arguments have an unclosed quote."]
    warns = warnings(validate(Profile(extra_args="--port=1 --listen x --cuda-device 0 --cpu --ok"),
                              catalog))
    assert len(warns) == 4
    assert all(any(f in w for w in warns) for f in ("--port", "--listen", "--cuda-device", "--cpu"))


def test_validate_env(catalog):
    p = Profile(env=[["", ""], ["  ", " "], ["GOOD", "x"], [" SPACED ", "y"], ["1BAD", "x"],
                     ["has-dash", "x"], ["", "orphan value"], ["ML", "a\nb"], ["CR", "a\rb"]])
    errs = errors(validate(p, catalog))
    assert len(errs) == 5
    assert any("'1BAD'" in e for e in errs) and any("'has-dash'" in e for e in errs)
    assert any("''" in e for e in errs)
    assert any("ML" in e and "single line" in e for e in errs)
    assert any("CR" in e and "single line" in e for e in errs)


def test_issue_is_frozen():
    with pytest.raises(Exception):
        Issue("error", "x").level = "warning"


# -- validate_together --------------------------------------------------------
def P(name, port, mode=GPU_AUTO, gpus=()):
    return Profile(name=name, port=port, gpu_mode=mode, gpus=[GpuRef(u, u.upper()) for u in gpus])


def test_together_port_clash_and_every_gpu():
    a, b = P("a", 8188), P("b", 8188)
    issues = validate_together([a, b], [])
    assert errors(issues) == ["'a' and 'b' both use port 8188."]
    assert len(warnings(issues)) == 1 and "every GPU" in warnings(issues)[0]


def test_together_select_overlap():
    a = P("a", 1, GPU_SELECT, ["g1", "g2"])
    b = P("b", 2, GPU_SELECT, ["g2", "g3"])
    issues = validate_together([a], [b])
    assert errors(issues) == []
    assert warnings(issues) == ["'a' and 'b' both use G2."]
    assert validate_together([a, P("c", 3, GPU_SELECT, ["g9"])], []) == []


def test_together_auto_vs_select_and_cpu():
    auto = P("auto", 1)
    sel = P("sel", 2, GPU_SELECT, ["g1"])
    cpu = P("cpu", 3, GPU_CPU)
    w = warnings(validate_together([sel], [auto]))
    assert w == ["'auto' uses every GPU, including the one(s) 'sel' uses."]
    assert validate_together([auto], [cpu]) == []
    assert validate_together([cpu, P("cpu2", 4, GPU_CPU)], []) == []
    assert validate_together([P("selnone", 5, GPU_SELECT)], [auto]) == []


def test_together_running_pairs_ignored_and_same_id_replaced():
    r1, r2 = P("r1", 8188), P("r2", 8188)
    assert validate_together([], [r1, r2]) == []
    # Restarting a running profile: the running copy (same id) is not compared.
    again = r1.snapshot()
    again.gpu_mode = GPU_CPU
    issues = validate_together([again], [r1])
    assert issues == []
    issues = validate_together([again], [r1, r2])
    assert errors(issues) == ["'r1' and 'r2' both use port 8188."]


# --------------------------------------------------------------------------
# scripts
# --------------------------------------------------------------------------
def test_launch_script_basic_shape(catalog):
    s = build_launch_script(Profile(), catalog)
    assert s.endswith("rcl_main </dev/null\n")
    assert "\r" not in s
    assert 'cd -- "$HOME"/ComfyUI ||' in s
    assert "/dev/tcp/127.0.0.1/8188" in s
    assert "RCL_PY=venv/bin/python" in s
    assert "--cuda-device" not in s.split("RCL_GPU_ARGS=()")[1].split("RCL_PY=")[0]
    assert 'exec "$RCL_PY" -u main.py "${RCL_ARGS[@]}"' in s
    assert "  RCL_ARGS=(--port 8188 \"${RCL_GPU_ARGS[@]}\" )" in s
    assert f"exit {EXIT_NO_FOLDER}" in s


def test_launch_script_env_and_pre_commands(catalog):
    p = Profile(env=[[" GOOD ", "$HOME/x"], ["bad name", "1"], ["ML", "a\nb"], ["", ""]],
                pre_commands="\r\n\r\nsource ~/a\r\necho 'b'\rlast\n\n")
    s = build_launch_script(p, catalog)
    assert '  export GOOD="$HOME/x"' in s
    assert "bad name" not in s and "export ML" not in s
    block = s.split("# --- commands from the configuration ---\n")[1].split(
        "  # --- end of commands")[0]
    assert block == "source ~/a\necho 'b'\nlast\n"
    assert "commands from the configuration" not in build_launch_script(
        Profile(pre_commands=" \n \n"), catalog)


def test_launch_script_gpu_select(catalog):
    p = Profile(gpu_mode=GPU_SELECT, gpus=[GpuRef("GPU-2"), GpuRef("GPU-1")])
    s = build_launch_script(p, catalog)
    assert "export CUDA_DEVICE_ORDER=PCI_BUS_ID" in s
    assert "for RCL_U in GPU-2 GPU-1; do" in s
    assert 'RCL_GPU_ARGS=(--cuda-device "$RCL_DEV")' in s
    auto = build_launch_script(Profile(), catalog)
    assert "CUDA_DEVICE_ORDER" not in auto and "nvidia-smi" not in auto


def test_launch_signature(catalog):
    p = Profile()
    sig = launch_signature(p, catalog)
    assert len(sig) == 40 and sig == launch_signature(p.snapshot(), catalog)
    # Settings that do not affect the server do not change it.
    for change in (dict(name="Other"), dict(auto_open=True), dict(checked=True)):
        q = p.snapshot()
        for k, v in change.items():
            setattr(q, k, v)
        assert launch_signature(q, catalog) == sig, change
    # Settings that do.
    for change in (dict(distro="Debian"), dict(user="root"), dict(port=9000),
                   dict(extra_args="--x"), dict(env=[["A", "1"]]), dict(pre_commands="true"),
                   dict(gpu_mode=GPU_CPU), dict(options={"--lowvram": on()})):
        q = p.snapshot()
        for k, v in change.items():
            setattr(q, k, v)
        assert launch_signature(q, catalog) != sig, change


def test_introspection_script():
    p = Profile(comfy_dir="~/Comfy UI", python="")
    s = build_introspection_script(p, "print('hi')\r\nprint(2)\r\n\r\n")
    lines = s.split("\n")
    assert lines[1] == f"  cd -- \"$HOME\"/'Comfy UI' || exit {EXIT_NO_FOLDER}"
    assert lines[2] == "  RCL_PY=python3"
    assert "print('hi')\nprint(2)\nRCL_PY_EOF_7f3a\n}\nrcl_main\n" in s
    assert "\r" not in s


def test_patterns():
    assert PID_PATTERN.match("[launcher] pid=123 ").group(1) == "123"
    assert not PID_PATTERN.match("x [launcher] pid=123")
    assert READY_PATTERN.search("To see the GUI go to: http://127.0.0.1:8188")


# --------------------------------------------------------------------------
# environment import
# --------------------------------------------------------------------------
def test_parse_env_dump():
    text = ("garbage\0RCL_ENV_BASE\0PATH=/usr/bin\0EQ=a=b\0FUNC=() {\n echo\n}\0noeq\0"
            "RCL_ENV_INTERACTIVE\0PATH=/x:/usr/bin\0EMPTY=\0")
    base, inter = parse_env_dump(text)
    assert base == {"PATH": "/usr/bin", "EQ": "a=b", "FUNC": "() {\n echo\n}"}
    assert inter == {"PATH": "/x:/usr/bin", "EMPTY": ""}
    assert parse_env_dump("") == ({}, {})


def test_env_dump_script_markers():
    assert "RCL_ENV_BASE" in ENV_DUMP_SCRIPT and "RCL_ENV_INTERACTIVE" in ENV_DUMP_SCRIPT


def test_diff_shell_env():
    base = {"PATH": "/usr/bin", "LD": "/lib", "SAME": "1", "CHANGED": "old", "EMPTYOLD": ""}
    inter = {
        "PATH": "/cuda/bin:/usr/bin",          # prepended
        "LD": "/lib:/cuda/lib",                # appended
        "SAME": "1",
        "CHANGED": "new $x",
        "EMPTYOLD": "v",
        "NEW": "price $5",
        "PS1": "\\u@\\h", "SHLVL": "2", "BASH_VERSINFO": "x", "_": "/bin/bash",
        "bad-name": "x", "MULTI": "a\nb",
    }
    result = dict(map(tuple, diff_shell_env(base, inter)))
    assert result == {
        "PATH": "/cuda/bin:$PATH",
        "LD": "$LD:/cuda/lib",
        "CHANGED": "new \\$x",
        "EMPTYOLD": "v",
        "NEW": "price \\$5",
    }
    assert [k for k, _ in diff_shell_env(base, inter)][:2] == ["PATH", "LD"]   # keeps order


def test_diff_shell_env_dollar_in_prefix():
    assert diff_shell_env({"P": "/b"}, {"P": "/$a:/b"}) == [["P", "/\\$a:$P"]]


def test_parse_env_dump_bashrc_noise():
    _, inter = parse_env_dump("RCL_ENV_BASE\0RCL_ENV_INTERACTIVE\0Welcome!\nFOO=1\0BAR=2\0")
    assert inter.get("FOO") == "1"


# --------------------------------------------------------------------------
# validation and script agree on env values that bash cannot parse
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["a${b", "${HOME", "${!", "${}", "${X:-$(id)}", "${X:?stop}"])
def test_only_plain_references_stay_live(value):
    # Anything but $NAME and ${NAME} is made literal, so no value can break the script.
    assert quote_env_value(value) == '"' + value.replace("$", "\\$") + '"'
