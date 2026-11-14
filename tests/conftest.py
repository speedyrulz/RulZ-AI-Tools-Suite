"""Shared by every test: no test reads or writes this install's own record of measured peak VRAM
(output/_vram_measured.json, rulztrainer/measured.py), which the wizard's block swap plan looks up. A developer's
real runs would otherwise change what the wizard tests see."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _own_vram_record(tmp_path, monkeypatch):
    from rulztrainer import measured

    monkeypatch.setattr(measured, "RECORD_FILE", tmp_path / "_vram_measured.json")
