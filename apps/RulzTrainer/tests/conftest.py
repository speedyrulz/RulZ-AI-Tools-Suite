"""Shared by every test: no test reads or writes this install's own record of measured peak VRAM
(output/_vram_measured.json, rulztrainer/measured.py), which the wizard's block swap plan looks up. A developer's
real runs would otherwise change what the wizard tests see."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# the vast.ai state folder (scripts/vast: rentals, sends, the fleet) is read when those modules are imported, which
# the test files do at collection: this runs first, so that no test touches the user's real ~/.rulz_vast
os.environ["RULZ_VAST_HOME"] = tempfile.mkdtemp(prefix="rulz_vast_tests_")


@pytest.fixture(autouse=True)
def _fresh_sweep_state(monkeypatch):
    """The region sweep keeps a batch size cut by a GPU memory error and a picture cache for the whole process: each
    test starts with neither (a test that fakes an out-of-memory error would otherwise shrink later tests' batches)."""
    from rulztrainer import regionsweep

    monkeypatch.setattr(regionsweep, "_batch_sizes", {})
    regionsweep.clear_cache()
    yield
    regionsweep.clear_cache()


@pytest.fixture(autouse=True)
def _own_vram_record(tmp_path, monkeypatch):
    from rulztrainer import measured

    monkeypatch.setattr(measured, "RECORD_FILE", tmp_path / "_vram_measured.json")


@pytest.fixture(autouse=True)
def _own_trainer_home(tmp_path_factory, monkeypatch):
    """No test reads or writes the real user's storage.json (the chosen save folders, rulztrainer/storage.py)."""
    monkeypatch.setenv("RULZ_TRAINER_HOME", str(tmp_path_factory.mktemp("rulz_trainer_home")))


@pytest.fixture(autouse=True)
def _no_real_vast_or_hugging_face(tmp_path_factory, monkeypatch):
    """No test reaches the real vast.ai API, Hugging Face, or the user's saved keys: the API key variables are gone
    and the home folder (where the vastai CLI and huggingface_hub keep their keys, and ~/.ssh) is an empty one, and
    rulz_market's own HTTP opener refuses (a test that wants an answer hands its Api a fake opener, or sets
    VAST_API_KEY itself). It works whether or not scripts/vast is on sys.path yet."""
    home = tmp_path_factory.mktemp("rulz_home")
    for name in ("VAST_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "RULZ_VAST_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(home))
    monkeypatch.setenv("HF_HOME", str(home / "hf"))
    vast = str(Path(__file__).resolve().parents[1] / "scripts" / "vast")
    if vast not in sys.path:
        sys.path.insert(0, vast)
    refused: list[str] = []
    try:
        import rulz_market
    except ImportError:
        yield refused
        return

    def refuse(*args, **kwargs):
        refused.append(f"{args[0] if args else ''} {args[1] if len(args) > 1 else ''}".strip())
        raise AssertionError("a test reached the real vast.ai API: give the Api a fake opener")

    monkeypatch.setattr(rulz_market, "_urlopen", refuse)
    yield refused
    # the refusal is an AssertionError that code may catch (except Exception): it still fails the test, here
    assert not refused, f"a test reached the real vast.ai API: {refused}"
