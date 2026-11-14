"""Shared fixtures. The tests only import the pure-logic modules (no Qt, WSL or GPUs)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_bash() -> str | None:
    """A real bash (Git Bash on Windows), never the WSL launcher (System32 or WindowsApps
    bash.exe), so no test ever starts WSL."""
    candidates = []
    found = shutil.which("bash")
    if found:
        candidates.append(found)
    if os.name == "nt":
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"),
                     r"C:\Program Files"):
            if base:
                candidates.append(os.path.join(base, "Git", "bin", "bash.exe"))
                candidates.append(os.path.join(base, "Git", "usr", "bin", "bash.exe"))
    for path in candidates:
        low = path.lower()
        if os.name == "nt" and ("\\system32\\" in low or "\\windowsapps\\" in low):
            continue
        if os.path.isfile(path):
            return path
    return None


BASH = _find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash on PATH (Git Bash)")


def run_bash(args: list[str], script: str, timeout: float = 20) -> subprocess.CompletedProcess:
    """Run bash with the script on stdin; returns text output (UTF-8)."""
    return subprocess.run([BASH, *args], input=script.encode("utf-8"),
                          capture_output=True, timeout=timeout)


@pytest.fixture
def bash():
    if BASH is None:
        pytest.skip("no bash on PATH (Git Bash)")
    return run_bash


@pytest.fixture(scope="session")
def catalog():
    from rulz_comfy_launcher.options import builtin_catalog
    return builtin_catalog()
