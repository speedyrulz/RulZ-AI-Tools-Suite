"""The pid line of the launch script and the script that signals an instance."""
import subprocess
import sys

import pytest

from rulz_comfy_launcher.command import (EXIT_NOT_OURS, PID_PATTERN, build_launch_script,
                                         build_signal_script)
from rulz_comfy_launcher.model import Profile
from rulz_comfy_launcher.wsl import clean_windows_path


def test_pid_pattern_reads_pid_and_start_time():
    match = PID_PATTERN.match("[launcher] pid=161940 start=5608019")
    assert match and match.group(1) == "161940" and match.group(2) == "5608019"


def test_pid_pattern_accepts_a_line_without_start_time():
    match = PID_PATTERN.match("[launcher] pid=42")
    assert match and match.group(1) == "42" and match.group(2) is None


@pytest.mark.parametrize("line", ["[launcher] pid=", "x [launcher] pid=5 start=6", "[launcher] pid=5 start=x",
                                  "[launcher] command: main.py pid=5"])
def test_pid_pattern_rejects_other_lines(line):
    assert PID_PATTERN.match(line) is None


def test_launch_script_prints_pid_with_start_time(catalog):
    script = build_launch_script(Profile(), catalog)
    assert 'rcl_say "pid=$$ start=$(' in script
    assert "/proc/$$/stat" in script


def test_signal_script_checks_identity_before_signalling():
    script = build_signal_script(1234, "5608019", "INT")
    lines = script.splitlines()
    assert "/proc/1234/stat" in lines[0] and '= "5608019" ]' in lines[0]
    assert lines[0].endswith(f"|| exit {EXIT_NOT_OURS}")
    assert lines[1] == "kill -s INT -- -1234 2>/dev/null || kill -s INT 1234"


@pytest.mark.parametrize("pid, start, name", [(1, "", "INT"), (1, "12x", "INT"), (1, "5", "INT; rm -rf /"),
                                              (1, "5", "int"), (1, "$(id)", "KILL")])
def test_signal_script_refuses_bad_input(pid, start, name):
    with pytest.raises(ValueError):
        build_signal_script(pid, start, name)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
def test_signal_script_does_not_signal_a_process_with_another_start_time():
    victim = subprocess.Popen(["sleep", "30"])
    try:
        done = subprocess.run(["bash", "-s"], input=build_signal_script(victim.pid, "1", "KILL"), text=True)
        assert done.returncode == EXIT_NOT_OURS
        assert victim.poll() is None
    finally:
        victim.kill()


def test_start_time_extraction_survives_odd_process_names(bash, tmp_path):
    # /proc/PID/stat puts the name in parentheses and the name may hold spaces and ")".
    stat = tmp_path / "stat"
    fields = ["S"] + [str(n) for n in range(4, 22)] + ["5608019", "23", "24"]
    stat.write_text("77 (odd) name (x)) " + " ".join(fields) + "\n", encoding="utf-8", newline="\n")
    script = f"sed 's/.*) //' '{stat.as_posix()}' | cut -d' ' -f20\n"
    out = bash(["-s"], script)
    assert out.stdout.decode("utf-8").strip() == "5608019"


def test_clean_windows_path_drops_entries_wsl_cannot_translate():
    dirty = r'C:\Windows\system32;"C:\Windows;C: \;;  ;C:\Program Files\Git\cmd;D:\tools"'
    assert clean_windows_path(dirty) == r"C:\Windows\system32;C:\Windows;C:\Program Files\Git\cmd;D:\tools"
