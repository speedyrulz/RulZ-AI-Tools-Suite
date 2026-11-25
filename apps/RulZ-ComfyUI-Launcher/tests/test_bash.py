"""Run generated shell text through a real bash (Git Bash on Windows) to check quoting.

Skipped when bash is missing. Never starts WSL (see conftest._find_bash).
"""
from __future__ import annotations

import socket
import stat

import pytest

from conftest import needs_bash
from rulz_comfy_launcher.command import (
    EXIT_NO_FOLDER, EXIT_NO_GPU, EXIT_NO_MAIN, EXIT_NO_PYTHON, EXIT_NO_SMI, EXIT_PORT_BUSY,
    build_introspection_script, build_launch_script, diff_shell_env, head_args, quote_env_value,
    quote_path, tail_args,
)
from rulz_comfy_launcher.model import GPU_CPU, GPU_SELECT, GpuRef, OptionState, Profile

pytestmark = needs_bash

ENV_VALUES = [
    "plain", "", "with space", "it's", 'say "hi"', "back`tick`", "a\\b", "trailing\\",
    "$(echo pwned)", "`echo pwned`", "cost $5 and $$ and $? $@ $* $1 $", "a;b&c|d>e<f",
    "\\$LIT", "\\\\server\\share", "tab\there", "{}[]()!#~*?", "\u00e9\u20ac",
    "\\$(echo x)", "$((1+2))",
]


def _out(result):
    return result.stdout.decode("utf-8", errors="replace")


@pytest.mark.parametrize("value", ENV_VALUES)
def test_quote_env_value_round_trip_literal(bash, value):
    script = f"export RCL_T={quote_env_value(value)}\nprintf %s \"$RCL_T\"\n"
    r = bash(["-s"], script)
    assert r.returncode == 0, r.stderr
    expected = value.replace("\\$", "$")   # "\$" is the documented way to write a literal dollar
    assert _out(r) == expected


@pytest.mark.parametrize("value, expected", [
    ("$RCL_A/bin:$RCL_B", "/a/bin:bee"),
    ("${RCL_A}x", "/ax"),
    ("$RCL_UNSET-x", "-x"),
    ("$_RCL_U", "under"),
    ("${RCL_A}", "/a"),
    # Only $NAME and ${NAME} are expanded; every other ${...} form is literal text.
    ("${RCL_A:-$(echo no)}", "${RCL_A:-$(echo no)}"),
    ("a${b", "a${b"),
    ("${HOME", "${HOME"),
    ("${!", "${!"),
    ("${}", "${}"),
])
def test_quote_env_value_keeps_variables_live(bash, value, expected):
    script = ("RCL_A=/a; RCL_B=bee; _RCL_U=under\n"
              f"export RCL_T={quote_env_value(value)}\nprintf %s \"$RCL_T\"\n")
    r = bash(["-s"], script)
    assert r.returncode == 0, r.stderr
    assert _out(r) == expected


@pytest.mark.parametrize("old, new", [
    ("/usr/bin", "/cuda/bin:/usr/bin"),
    ("/usr/bin", "/usr/bin:/cuda/bin"),
    ("/usr/bin", "/a$b:/usr/bin"),
    ("/usr/bin", "/usr/bin:/odd\\dir:$x"),
    ("x", "completely $new `value` \"q\""),
    (None, "fresh $HOME"),
])
def test_diff_shell_env_then_quote_reproduces_value(bash, old, new):
    base = {} if old is None else {"RCL_V": old}
    [[name, text]] = diff_shell_env(base, {"RCL_V": new})
    setup = "" if old is None else f"RCL_V={quote_env_value(old.replace('$', chr(92) + '$'))}\n"
    script = f"{setup}export {name}={quote_env_value(text)}\nprintf %s \"$RCL_V\"\n"
    r = bash(["-s"], script)
    assert r.returncode == 0, r.stderr
    assert _out(r) == new


@pytest.mark.parametrize("path", ["/opt/comfy", "~", "~/my comfy", "~/it's $here; ok", "/a b/c'd"])
def test_quote_path_round_trip(bash, path):
    r = bash(["-s"], f"HOME=/home/u\nprintf %s {quote_path(path)}\n")
    expected = path.strip()
    if expected == "~":
        expected = "/home/u"
    elif expected.startswith("~/"):
        expected = "/home/u/" + expected[2:]
    assert _out(r) == expected


# -- the launch script -------------------------------------------------------
def odd_profiles():
    yield Profile()
    yield Profile(gpu_mode=GPU_CPU, options={"--lowvram": OptionState(True)})
    yield Profile(
        comfy_dir="~/my comfy/it's $HOME; rm -rf x", python="/opt/py env/bin/python",
        port=9001, listen="0.0.0.0,::", gpu_mode=GPU_SELECT,
        gpus=[GpuRef("GPU-1111-aaaa", "A"), GpuRef("GPU-'odd' $x;", "B")],
        options={"--preview-method": OptionState(True, "taesd"),
                 "--output-directory": OptionState(True, "/mnt/d/out put/'q' $(x) `y`"),
                 "--feature-flag": OptionState(True, "a=1 'b=two words'"),
                 "--reserve-vram": OptionState(True, "-1"),
                 "--unknown-opt": OptionState(True, "v;al"),
                 "--fast": OptionState(True, "autotune")},
        extra_args="--foo 'bar baz' \"q\\\"uote\" '$HOME' ';'",
        env=[["CUDA_HOME", "/usr/local/cuda"], ["PATH", "$CUDA_HOME/bin:$PATH"],
             ["WEIRD", "it's \"x\" `y` $(z) \\$lit ;&|"]],
        pre_commands="source ~/.profile\r\nif true; then\n  echo 'hi $x'\nfi\n# comment\n",
    )
    yield Profile(python="python3", comfy_dir="/", pre_commands="cd /tmp && echo ok")


@pytest.mark.parametrize("profile", list(odd_profiles()), ids=lambda p: p.comfy_dir)
def test_launch_script_syntax(bash, catalog, profile):
    r = bash(["-n"], build_launch_script(profile, catalog))
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")


@pytest.mark.parametrize("profile", list(odd_profiles()), ids=lambda p: p.comfy_dir)
def test_launch_script_args_line_gives_exact_words(bash, catalog, profile):
    """Run just the RCL_ARGS line and compare the words bash builds with head_args/tail_args."""
    script = build_launch_script(profile, catalog)
    [line] = [ln for ln in script.split("\n") if ln.startswith("  RCL_ARGS=(")]
    test = ("RCL_GPU_ARGS=(--cuda-device 3,1)\n" + line.strip() + "\n"
            "printf '%s\\0' \"${RCL_ARGS[@]}\"\n")
    r = bash(["-s"], test)
    assert r.returncode == 0, r.stderr
    words = _out(r).split("\0")[:-1]
    assert words == head_args(profile) + ["--cuda-device", "3,1"] + tail_args(profile, catalog)


@needs_bash
def test_introspection_script_syntax(bash):
    src = "import sys\nprint('RCL_JSON:' + '{}')\nx = '''\n$HOME `y`\n'''\n"
    r = bash(["-n"], build_introspection_script(Profile(comfy_dir="~/a b"), src))
    assert r.returncode == 0, r.stderr


# -- running the whole launch script against fakes ----------------------------
FAKE_PYTHON = """#!/bin/sh
printf 'ACT=%s\\0' "$RCL_ACTIVATED"
printf 'X=%s\\0' "$RCL_X"
printf 'ORDER=%s\\0' "$CUDA_DEVICE_ORDER"
printf 'PWD=%s\\0' "$PWD"
for a in "$@"; do printf 'ARG=%s\\0' "$a"; done
"""

FAKE_SMI_OK = """#!/bin/sh
printf '0, GPU-aaa, NVIDIA Card A\\n1, GPU-bbb, NVIDIA Card B\\n'
"""

FAKE_SMI_FAIL = """#!/bin/sh
echo 'NVIDIA-SMI has failed' >&2
exit 9
"""


def _write_exec(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def comfy(tmp_path):
    """A fake ComfyUI folder with an odd name, a fake venv python and fake nvidia-smi tools."""
    folder = tmp_path / "comfy dir 'q' $HOME; & (x)"
    (folder).mkdir()
    (folder / "main.py").write_text("# fake\n", encoding="utf-8")
    _write_exec(folder / "venv" / "bin" / "python", FAKE_PYTHON)
    with open(folder / "venv" / "bin" / "activate", "w", encoding="utf-8", newline="\n") as h:
        h.write("export RCL_ACTIVATED=yes\n")
    _write_exec(tmp_path / "smi_ok" / "nvidia-smi", FAKE_SMI_OK)
    _write_exec(tmp_path / "smi_fail" / "nvidia-smi", FAKE_SMI_FAIL)
    return tmp_path, folder.name


def _run_launch(bash, tmp_path, profile, catalog):
    import subprocess
    from conftest import BASH
    script = build_launch_script(profile, catalog)
    return subprocess.run([BASH, "-s"], input=script.encode("utf-8"), capture_output=True,
                          cwd=tmp_path, timeout=30)


def _fields(result):
    out = {}
    args = []
    for item in result.stdout.decode("utf-8", "replace").split("\0"):
        item = item.rsplit("\n", 1)[-1]   # launcher messages precede the first field
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key == "ARG":
            args.append(value)
        else:
            out[key] = value
    return out, args


def test_launch_script_runs_with_fakes(bash, catalog, comfy):
    tmp_path, folder = comfy
    port = _free_port()
    p = Profile(comfy_dir=folder, port=port, gpu_mode=GPU_SELECT,
                gpus=[GpuRef("GPU-bbb"), GpuRef("GPU-aaa")],
                env=[["RCL_X", "it's \"x\" `y` $(z) \\$lit"]],
                pre_commands='export PATH="$PWD/../smi_ok:$PATH"',
                options={"--output-directory": OptionState(True, "/o u/'t'")},
                extra_args="--foo 'a b'")
    r = _run_launch(bash, tmp_path, p, catalog)
    assert r.returncode == 0, (r.stdout, r.stderr)
    text = r.stdout.decode("utf-8", "replace")
    assert "[launcher] GPU 1 (PCI bus order): NVIDIA Card B" in text
    assert "[launcher] GPU 0 (PCI bus order): NVIDIA Card A" in text
    assert "[launcher] pid=" in text
    fields, args = _fields(r)
    assert fields["ACT"] == "yes"
    assert fields["X"] == "it's \"x\" `y` $(z) $lit"
    assert fields["ORDER"] == "PCI_BUS_ID"
    assert fields["PWD"].endswith(folder)
    assert args == ["-u", "main.py", "--port", str(port), "--cuda-device", "1,0",
                    "--output-directory", "/o u/'t'", "--foo", "a b"]


def test_launch_script_cpu_mode_args(bash, catalog, comfy):
    tmp_path, folder = comfy
    p = Profile(comfy_dir=folder, port=_free_port(), gpu_mode=GPU_CPU)
    r = _run_launch(bash, tmp_path, p, catalog)
    assert r.returncode == 0, r.stderr
    fields, args = _fields(r)
    assert args[-1] == "--cpu" and "--cuda-device" not in args
    assert fields["ORDER"] == ""


@pytest.mark.parametrize("case, code", [
    ("no_folder", EXIT_NO_FOLDER),
    ("no_main", EXIT_NO_MAIN),
    ("no_python", EXIT_NO_PYTHON),
    ("no_command", EXIT_NO_PYTHON),
    ("smi_fail", EXIT_NO_SMI),
    ("no_gpu", EXIT_NO_GPU),
    ("port_busy", EXIT_PORT_BUSY),
])
def test_launch_script_exit_codes(bash, catalog, comfy, case, code):
    tmp_path, folder = comfy
    p = Profile(comfy_dir=folder, port=_free_port())
    listener = None
    if case == "no_folder":
        p.comfy_dir = "does not exist"
    elif case == "no_main":
        (tmp_path / folder / "main.py").unlink()
    elif case == "no_python":
        p.python = "venv/bin/nope"
    elif case == "no_command":
        p.python = "rcl-no-such-python"
    elif case == "smi_fail":
        p.gpu_mode, p.gpus = GPU_SELECT, [GpuRef("GPU-aaa")]
        p.pre_commands = 'export PATH="$PWD/../smi_fail:$PATH"'
    elif case == "no_gpu":
        p.gpu_mode, p.gpus = GPU_SELECT, [GpuRef("GPU-aaa"), GpuRef("GPU-zzz")]
        p.pre_commands = 'export PATH="$PWD/../smi_ok:$PATH"'
    elif case == "port_busy":
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(5)
        p.port = listener.getsockname()[1]
    try:
        r = _run_launch(bash, tmp_path, p, catalog)
    finally:
        if listener:
            listener.close()
    if case == "port_busy" and r.returncode == 0:
        pytest.skip("this bash has no working timeout or /dev/tcp")
    assert r.returncode == code, (r.stdout, r.stderr)
    assert b"[launcher] ERROR" in r.stdout
