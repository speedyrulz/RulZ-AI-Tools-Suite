"""RulzTrainer likeness scoring for ComfyUI: the trainer's own face / body scoring on any images.

The nodes import the scoring code from a RulzTrainer checkout, so scores match the trainer's exactly. The checkout is
found, in order: the RULZTRAINER_DIR environment variable; repo_path.txt next to this file (one line, the checkout's
folder); the checkout this folder lives in (when custom_nodes/ holds a link to it); an installed `rulztrainer`."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_repo() -> Path | None:
    cands = []
    if os.environ.get("RULZTRAINER_DIR"):
        cands.append(Path(os.environ["RULZTRAINER_DIR"]))
    txt = HERE / "repo_path.txt"
    if txt.is_file():
        line = next((ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()), "")
        if line:
            cands.append(Path(line.strip('"')))
    cands += [HERE.parent.parent, Path(__file__).parent.parent.parent]  # <checkout>/comfyui/rulztrainer_likeness
    for c in cands:
        if (c / "rulztrainer" / "facesim.py").is_file():
            return c.resolve()
    try:
        import rulztrainer

        return Path(rulztrainer.__file__).resolve().parent.parent
    except ImportError:
        return None


REPO = _find_repo()
if REPO is None:
    raise ImportError("RulzTrainer likeness nodes: the RulzTrainer checkout was not found. Link this folder from the "
                      "checkout into custom_nodes, set RULZTRAINER_DIR, or write the checkout's path into "
                      f"{HERE / 'repo_path.txt'}")
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))  # appended: the checkout never shadows ComfyUI's own modules

from . import nodes  # noqa: E402

nodes.REPO_DIR = REPO
NODE_CLASS_MAPPINGS = nodes.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = nodes.NODE_DISPLAY_NAME_MAPPINGS
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
