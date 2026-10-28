"""The WebUI's dataset helpers: names and Create dataset, the list of dataset folders (the datasets folder, folders
added from elsewhere, a loaded config's), where uploads land, Copy from a folder, the cards' counts and thumbnails,
the wizard's summary over several datasets and the review line."""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webui  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


def _img(path: Path, caption: str | None = None, size=(64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (200, 80, 40)).save(path)
    if caption is not None:
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return path


def test_dataset_names_and_create(tmp_path):
    root = tmp_path / "datasets"
    assert webui.dataset_name_error("", root) and webui.dataset_name_error("  ", root)
    for bad in ("../up", "a/b", "-dash", "dot.", "con", "LPT1.txt", "x" * 65):
        assert webui.dataset_name_error(bad, root), bad
    assert webui.dataset_name_error("Anna full body_2.v1", root) is None
    made = webui.create_dataset("  anna  ", root)
    assert made == root / "anna" and made.is_dir()
    assert "already" in webui.dataset_name_error("anna", root)
    with pytest.raises(ValueError):
        webui.create_dataset("anna", root)


def test_dataset_list_registry_and_extra(tmp_path):
    root = tmp_path / "datasets"
    for name in ("b_set", "A_set", ".hidden"):
        (root / name).mkdir(parents=True)
    (root / "notes.txt").write_text("not a dataset")
    outside = tmp_path / "elsewhere" / "old_set"
    outside.mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    webui.register_dataset(str(outside), root)
    webui.register_dataset(str(outside), root)  # once only
    webui.register_dataset(str(root / "A_set"), root)  # inside the datasets folder: listed anyway, not stored
    assert webui.registered_datasets(root) == [str(outside)]
    with pytest.raises(ValueError):
        webui.register_dataset(str(root), root)  # the datasets folder itself
    with pytest.raises(ValueError):
        webui.register_dataset(str(tmp_path), root)  # a folder around it
    listed = webui.known_datasets([str(other), str(tmp_path / "missing"), str(outside)], root)
    assert listed == [str(root / "A_set"), str(root / "b_set"), str(outside), str(other)]
    webui.unregister_dataset(str(outside), root)
    assert webui.registered_datasets(root) == [] and str(outside) not in webui.known_datasets(root=root)
    (root / webui.DATASETS_REGISTRY).write_text("not json")
    assert webui.registered_datasets(root) == []  # a broken list file: no added folders, no crash


def test_upload_targets(tmp_path):
    ds = tmp_path / "ds"
    assert webui.upload_target(ds, "photo.JPG") == ds / "photo.JPG" and ds.is_dir()
    assert webui.upload_target(ds, "photo.txt") == ds / "photo.txt"
    assert webui.upload_target(ds, "photo.json") == ds / "photo.json"
    assert webui.upload_target(ds, "../../escape.png") == ds / "escape.png"  # the name only, never a folder
    assert webui.upload_target(ds, "C:\\\\Users\\\\x\\\\pic.webp") == ds / "pic.webp"
    for bad in ("model.safetensors", "script.py", ".hidden.png", ""):
        assert webui.upload_target(ds, bad) is None, bad


def test_copy_from_a_folder_keeps_existing_files(tmp_path):
    src, ds = tmp_path / "src", tmp_path / "ds"
    _img(src / "a.png", "an a")
    _img(src / "sub" / "b.jpg", "a b")
    (src / "c.json").write_text("{}")
    (src / "readme.md").write_text("skip me")
    _img(src / ".rulz_cache" / "thumbs" / "t.jpg")  # another dataset's cache is not copied
    _img(ds / "a.png", "edited caption")
    res = webui.copy_into_dataset(src, ds)
    assert res == {"copied": 3, "kept": 2, "ignored": 1}  # a.png and its edited a.txt stay; b, b.txt, c.json come
    assert (ds / "a.txt").read_text() == "edited caption" and (ds / "sub" / "b.jpg").is_file()
    assert (ds / "sub" / "b.txt").read_text() == "a b" and not (ds / ".rulz_cache").exists()
    for bad_src in (ds, ds / "sub", tmp_path):
        with pytest.raises(ValueError):
            webui.copy_into_dataset(bad_src, ds)
    with pytest.raises(ValueError):
        webui.copy_into_dataset(tmp_path / "missing", ds)


def test_cards_summary_and_review_line(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for i in range(6):
        _img(a / f"{i}.png", f"caption {i}" if i % 2 == 0 else None)
    _img(b / "x.png", "x")
    (tmp_path / "empty").mkdir()
    counts = webui.dataset_counts([str(a), str(b), str(tmp_path / "missing")])
    assert counts == {str(a): (6, 3), str(b): (1, 1)}
    cards = webui.dataset_cards([str(a), str(tmp_path / "empty")])
    assert cards[0]["name"] == "a" and cards[0]["images"] == 6 and len(cards[0]["thumbs"]) == 4
    assert all(Path(t).parent == a / ".rulz_cache" / "thumbs" for t in cards[0]["thumbs"])
    assert cards[1]["images"] == 0 and cards[1]["thumbs"] == []
    data = webui.datasets_summary([str(a), str(b), str(tmp_path / "missing")])
    assert data["images"] == 7 and data["captioned"] == 4 and data["missing"] == [str(tmp_path / "missing")]
    assert {root for root, _ in data["sample"]} == {str(a), str(b)}
    assert len(webui.summary_thumbs(data)) == len(data["sample"])
    assert webui.datasets_summary([]) is None
    cfg = TrainConfig(datasets=[{"path": str(a)}, {"path": str(b), "repeats": 3, "resolutions": [0.5, 1.0]}])
    assert webui.datasets_line(cfg, data) == "a, b (x3, 0.5 MP + 1 MP): 7 images, 4 captioned"


def test_race_images_group_by_generation_with_normalized_scores(tmp_path):
    names = ["lora16", "lokr8"]
    files = [f"{tmp_path}/samples/{n}_step000120_{i}.png" for n in names for i in (0, 1)]
    files += [f"{tmp_path}/probes/{n}_step000240_{k}_0.png" for n in ("lokr8", "lora16") for k in ("face", "body")]
    files += [f"{tmp_path}/samples/step000300_0.png", f"{tmp_path}/samples/notes.png"]  # the winner after the race
    groups = webui.group_by_generation(files, names)
    assert [(s, r) for s, r, _ in groups] == [(300, "0"), (240, "body_0"), (240, "face_0"), (120, "0"), (120, "1")]
    by = {(s, r): [w for w, _ in m] for s, r, m in groups}
    assert by[(120, "0")] == ["lora16", "lokr8"] and by[(240, "face_0")] == ["lora16", "lokr8"]  # race order
    assert by[(300, "0")] == [""]  # the winner's image has no configuration prefix
    assert webui.generation_label(120, "1") == "step 120, sample 2"
    assert webui.generation_label(240, "face_0") == "step 240, face probe 1"
    assert webui.generation_label(240, "body_1") == "step 240, body probe 2"
    rec = {"face_norm": 47.2, "body_norm": 17.4, "combined": 32.3, "face": 60.0, "body": 80.0, "scale": {"face": [1, 2]}}
    assert webui.probe_score_line(rec) == "overall 32% | face 47% | body 17%"  # never the raw scores
    assert webui.probe_score_line({"face": 60.0, "body": 80.0, "combined": 70.0, "scale": {}}) == \
        "overall 70% | face 60% | body 80%"  # normalization off: the scores themselves
    assert webui.probe_score_line(None) == ""
    (tmp_path / "race.json").write_text('{"names": ["a", "b"]}')
    assert webui.race_names(tmp_path) == ["a", "b"] and webui.race_names(tmp_path / "none") == []


def test_race_probe_chart_names_each_configuration():
    names = ["lora16", "lokr8"]

    def rec(name, step, face, body, comb):
        prefix = f"{name}_" if name else ""
        return {"step": step, "face_norm_ema": face, "body_norm_ema": body, "combined_ema": comb,
                "images": [f"out/probes/{prefix}step{step:06d}_face_0.png"]}

    probes = [rec("lora16", 100, 5.0, 16.2, 10.6), rec("lokr8", 100, 6.7, 21.9, 14.3), rec("", 300, 9.0, 25.0, 17.0)]
    assert webui.probe_racer(probes[1], names) == "lokr8" and webui.probe_racer(probes[2], names) == ""
    series, legend = webui.probe_chart_series(probes, names)
    assert legend == ["lora16 combined", "lora16 face", "lora16 body", "lokr8 combined", "lokr8 face", "lokr8 body",
                      "winner combined", "winner face", "winner body"]
    data = {s["name"]: s["data"] for s in series}
    assert data["lora16 face"] == [[100, 5.0]] and data["lokr8 face"] == [[100, 6.7]]
    assert data["lokr8 combined"] == [[100, 14.3]] and data["winner body"] == [[300, 25.0]]
    colors = {s["name"].split()[0]: s["lineStyle"]["color"] for s in series}
    assert len(set(colors.values())) == 3  # one colour per configuration
    single, legend1 = webui.probe_chart_series(probes[:1], [])  # one run: the lines and the per-probe dots
    assert legend1 == ["face", "body", "combined", "face (probe)", "body (probe)"] and single[0]["data"] == [[100, 5.0]]
