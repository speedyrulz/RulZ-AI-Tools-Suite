"""The PyTorch wheel index for this PC's NVIDIA driver, printed for install.bat and install.sh.

A CUDA build newer than the driver supports installs fine but then sees no GPU (torch.cuda.is_available() is False),
so the newest of cu130 / cu128 / cu126 that the driver runs is picked, never newer than --default. nvidia-smi's header
says what the driver supports ("CUDA Version: 12.8"). No nvidia-smi, or no such line: --default.
(OneTrainer does the same since #1572.)

    python scripts/torch_index.py                  # Windows' default: cu130 when the driver runs it
    python scripts/torch_index.py --default cu128  # Linux's default
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

BASE = "https://download.pytorch.org/whl/"
BUILDS = (("cu130", (13, 0)), ("cu128", (12, 8)), ("cu126", (12, 6)))  # newest first, with the CUDA each needs


def driver_cuda(smi_output: str) -> tuple[int, int] | None:
    """The CUDA version nvidia-smi's header says the driver supports, or None."""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi_output)
    return (int(m.group(1)), int(m.group(2))) if m else None


def pick(driver: tuple[int, int] | None, default: str = "cu130") -> tuple[str, str]:
    """(build, warning or "") for a driver's CUDA version: the newest build it runs, at most `default`."""
    names = [b for b, _ in BUILDS]
    allowed = BUILDS[names.index(default):] if default in names else BUILDS
    if driver is None:
        return default, ""
    for build, needs in allowed:
        if driver >= needs:
            if build == default:
                return build, ""
            return build, (f"Your NVIDIA driver supports CUDA up to {driver[0]}.{driver[1]}, so the {build} build of "
                           f"PyTorch is installed instead of {default}. Updating the driver is recommended.")
    return allowed[-1][0], (f"Your NVIDIA driver supports CUDA up to {driver[0]}.{driver[1]}, older than any PyTorch "
                            f"build RulzTrainer uses: installing {allowed[-1][0]}, which will most likely not see the "
                            "GPU. Update the NVIDIA driver.")


def smi_output() -> str:
    """nvidia-smi's output ("" when it is missing or fails), WSL's copy outside PATH included."""
    wsl = "/usr/lib/wsl/lib/nvidia-smi"
    smi = shutil.which("nvidia-smi") or (wsl if os.path.exists(wsl) else None)
    if not smi:
        return ""
    try:
        return subprocess.run([smi], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--default", default="cu130", choices=[b for b, _ in BUILDS])
    args = ap.parse_args(argv)
    build, warning = pick(driver_cuda(smi_output()), args.default)
    if warning:
        print(f"[WARN] {warning}", file=sys.stderr)
    print(BASE + build)
    return 0


if __name__ == "__main__":
    sys.exit(main())
