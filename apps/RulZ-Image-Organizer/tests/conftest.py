"""Every test gets its own library folder (RULZ_DATA_DIR), so a real library is never touched."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setenv("RULZ_DATA_DIR", str(d))
    return d


@pytest.fixture
def db(data_dir):
    from rulz.db import Database

    database = Database()
    yield database
    database.close()


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
