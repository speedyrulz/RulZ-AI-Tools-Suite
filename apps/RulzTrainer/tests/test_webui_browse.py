"""The WebUI's in-page browser of the server's files (for a remote browser, e.g. on vast.ai): file-type filter,
folder listing and the shortcut folders."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402


def test_file_suffixes_follow_the_default_filter():
    assert webui.file_suffixes() == {".safetensors"}
    assert webui.file_suffixes([("YAML", "*.yaml;*.yml")]) == {".yaml", ".yml"}
    assert webui.file_suffixes([("all files", "*.*")]) is None


def test_list_folder_sorts_filters_and_reports_errors(tmp_path):
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "A_dir").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "model.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "Other.SAFETENSORS").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("hi")
    dirs, files, err = webui.list_folder(tmp_path, {".safetensors"})
    assert err == "" and [d.name for d in dirs] == ["A_dir", "b_dir"]
    assert [(p.name, size) for p, size in files] == [("model.safetensors", 10), ("Other.SAFETENSORS", 0)]
    _, files, _ = webui.list_folder(tmp_path, None)
    assert {p.name for p, _ in files} == {"model.safetensors", "Other.SAFETENSORS", "notes.txt"}
    dirs, files, _ = webui.list_folder(tmp_path, None, kind="dir")
    assert files == [] and len(dirs) == 2
    assert ".hidden" in {d.name for d in webui.list_folder(tmp_path, None, show_hidden=True)[0]}
    dirs, files, err = webui.list_folder(tmp_path / "missing", None)
    assert dirs == files == [] and err


def test_browse_shortcuts_exist_once_and_include_the_project(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    (ws / "ComfyUI" / "models").mkdir(parents=True)
    monkeypatch.setenv("WORKSPACE", str(ws))
    shortcuts = webui.browse_shortcuts()
    names = [n for n, _ in shortcuts]
    assert names[0] == "RulzTrainer" and "workspace" in names and "ComfyUI models" in names
    paths = [p.resolve() for _, p in shortcuts]
    assert len(paths) == len(set(paths)) and all(p.is_dir() for _, p in shortcuts)
