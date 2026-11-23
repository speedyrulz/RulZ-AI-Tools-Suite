"""Tests for rulztrainer/loratest_export.py: grid.jpg, scores.xlsx and report.pdf from spec.json and results.json."""
from __future__ import annotations

import json
import re
import sys
import zlib
from pathlib import Path

import pytest
from openpyxl import load_workbook
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("matplotlib")

from rulztrainer import loratest_export as LE  # noqa: E402


def _make_png(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _cell_color(c_idx: int, s_idx: int, p_idx: int, seed_idx: int) -> tuple[int, int, int]:
    """Distinct RGB colour for every (contestant, strength, prompt, seed) cell."""
    r = (30 + 60 * c_idx + 20 * s_idx) % 256
    g = (40 + 70 * p_idx + 35 * seed_idx) % 256
    b = (50 + 40 * c_idx + 55 * s_idx + 25 * p_idx + 45 * seed_idx) % 256
    return (r, g, b)


def _assert_rgb_close(actual: tuple[int, int, int], expected: tuple[int, int, int], tol: int = 8) -> None:
    for a, e in zip(actual, expected):
        assert abs(int(a) - int(e)) <= tol, f"expected {expected} (+/-{tol}), got {actual}"


def _extract_pdf_streams_text(pdf_bytes: bytes) -> str:
    """Decompress all FlateDecode streams in a PDF and return the concatenated Latin-1 string."""
    chunks: list[str] = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.DOTALL):
        raw = match.group(1)
        try:
            data = zlib.decompress(raw)
        except zlib.error:
            data = raw
        chunks.append(data.decode("latin-1", errors="ignore"))
    return "\n".join(chunks)


def _synthetic_test_dir(tmp_path: Path, complete: bool = True) -> Path:
    """Build a synthetic LoRA test folder matching docs/lora-test-contract.md with distinct cell colours."""
    test_dir = tmp_path / "t20261122-153012-a3f9"
    test_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "version": 1,
        "test_id": "t20261122-153012-a3f9",
        "name": "Elena: best vs final",
        "created": "2026-11-22T15:30:12Z",
        "contestants": [
            {
                "id": "c1",
                "label": "elena_v3 best",
                "lora": "output/elena_v3/elena_v3_best.safetensors",
                "config": "output/elena_v3/config.yaml",
                "run_id": "elena_v3-20261120-101500",
            },
            {
                "id": "c2",
                "label": "elena_v3 final",
                "lora": "output/elena_v3/elena_v3.safetensors",
                "config": "output/elena_v3/config.yaml",
                "run_id": "elena_v3-20261120-101500",
            },
        ],
        "soups": [
            {"id": "s1", "label": "soup c1+c2", "of": ["c1", "c2"], "weights": None},
        ],
        "prompts": [
            "a photo of elena, headshot, studio lighting",
            "a photo of elena, full body, walking outdoors",
        ],
        "prompt_kinds": ["face", "body"],
        "seeds": [1, 2],
        "strengths": [0.8, 1.0],
        "render": {"width": None, "height": None, "steps": None, "cfg": None, "negative": "blurry, deformed"},
        "scoring": {
            "face_refs": ["refs/face1.png"],
            "body_refs": ["refs/body1.png"],
            "from_contestant": "c1",
        },
        "devices": ["cuda:0"],
    }

    # 3 contestants (c1, c2, s1) x 2 strengths (0.8, 1.0) = 6 columns
    # 2 prompts x 2 seeds = 4 rows
    cells = []
    columns = []
    all_contestants = [
        {"id": "c1", "label": "elena_v3 best", "lora": "output/elena_v3/elena_v3_best.safetensors", "sha256": "aaa", "kind": "file"},
        {"id": "c2", "label": "elena_v3 final", "lora": "output/elena_v3/elena_v3.safetensors", "sha256": "bbb", "kind": "file"},
        {
            "id": "s1",
            "label": "soup c1+c2",
            "lora": "soups/s1.safetensors",
            "sha256": "ccc",
            "kind": "soup",
            "of": ["c1", "c2"],
            "weights": [0.5, 0.5],
        },
    ]

    for c_idx, c in enumerate(all_contestants):
        cid = c["id"]
        for s_idx, strength in enumerate(spec["strengths"]):
            col_scores = []
            for p_idx in range(len(spec["prompts"])):
                for seed_idx, seed in enumerate(spec["seeds"]):
                    rel = f"images/{cid}/s{strength:.2f}_p{p_idx}_seed{seed}.png"
                    # Leave one picture missing on disk (c2 at 1.0, prompt 1, seed 2)
                    is_missing_pic = cid == "c2" and strength == 1.0 and p_idx == 1 and seed == 2
                    # Leave one cell with null scores (s1 at 0.8, prompt 0, seed 2)
                    is_null_score = cid == "s1" and strength == 0.8 and p_idx == 0 and seed == 2
                    # When complete=False, omit the last contestant's second strength cells
                    if not complete and cid == "s1" and strength == 1.0:
                        continue
                    if not is_missing_pic:
                        _make_png(test_dir / rel, _cell_color(c_idx, s_idx, p_idx, seed_idx))
                    if is_null_score:
                        face = body = gold = combined = None
                    else:
                        base = 55.0 + 6.0 * (2 - c_idx) + 3.0 * s_idx + p_idx - 0.5 * seed
                        face = round(base + 1.5, 1)
                        body = round(base - 1.5, 1)
                        gold = round(base + 0.5, 1)
                        combined = round(base, 1)
                        col_scores.append(combined)
                    cells.append({
                        "contestant": cid,
                        "strength": strength,
                        "prompt": p_idx,
                        "seed": seed,
                        "image": rel,
                        "face": face,
                        "body": body,
                        "gold": gold,
                        "combined": combined,
                    })
            if col_scores:
                mean_c = round(sum(col_scores) / len(col_scores), 1)
                columns.append({
                    "contestant": cid,
                    "strength": strength,
                    "face": round(mean_c + 1.5, 1),
                    "body": round(mean_c - 1.5, 1),
                    "gold": round(mean_c + 0.5, 1),
                    "combined": mean_c,
                    "n": len(col_scores),
                })
            else:
                columns.append({
                    "contestant": cid,
                    "strength": strength,
                    "face": None,
                    "body": None,
                    "gold": None,
                    "combined": None,
                    "n": 0,
                })

    results = {
        "version": 1,
        "test_id": spec["test_id"],
        "complete": complete,
        "scale": "run",
        "scale_note": None,
        "contestants": all_contestants,
        "cells": cells,
        "columns": columns,
        "best_strength": {"c1": 1.0, "c2": 1.0, "s1": 0.8},
        "winner": {"contestant": "c1", "strength": 1.0, "combined": 69.8},
        "render": {"width": 1024, "height": 1024, "steps": 25, "cfg": 3.5, "seconds": 142.0, "gpus": ["RTX 5060 Ti"]},
    }

    (test_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (test_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return test_dir


def test_export_all_writes_grid_xlsx_and_pdf(tmp_path):
    test_dir = _synthetic_test_dir(tmp_path, complete=True)
    cell_px = 128
    paths = LE.export_all(test_dir, cell=cell_px)

    assert set(paths.keys()) == {"grid", "xlsx", "pdf"}
    assert paths["grid"] == test_dir / "export" / "grid.jpg"
    assert paths["xlsx"] == test_dir / "export" / "scores.xlsx"
    assert paths["pdf"] == test_dir / "export" / "report.pdf"
    assert all(p.is_file() for p in paths.values())

    gutter_w, header_h, score_h, _font_px, _small_px = LE._grid_metrics(cell_px)
    n_cols = 6  # (c1, c2, s1) x (0.8, 1.0)
    n_rows = 4  # 2 prompts x 2 seeds

    with Image.open(paths["grid"]) as im:
        expected_w = gutter_w + n_cols * cell_px
        expected_h = header_h + n_rows * (cell_px + score_h)
        assert im.size == (expected_w, expected_h)

        # Assert distinct cell colours at representative positions (prompt/seed/contestant/strength placement)
        for c_idx in range(3):
            for s_idx in range(2):
                col_idx = c_idx * 2 + s_idx
                for p_idx in range(2):
                    for seed_idx in range(2):
                        row_idx = p_idx * 2 + seed_idx
                        cx = gutter_w + col_idx * cell_px + cell_px // 2
                        cy = header_h + row_idx * (cell_px + score_h) + 12
                        # Missing cell is c2 (c_idx=1), strength 1.0 (s_idx=1), prompt 1 (p_idx=1), seed 2 (seed_idx=1)
                        if c_idx == 1 and s_idx == 1 and p_idx == 1 and seed_idx == 1:
                            _assert_rgb_close(im.getpixel((cx, cy)), LE.MISSING_BG_RGB, tol=10)
                        else:
                            _assert_rgb_close(
                                im.getpixel((cx, cy)),
                                _cell_color(c_idx, s_idx, p_idx, seed_idx),
                                tol=8,
                            )

        # Assert winner column outline on column index 1 (c1 @ 1.0)
        win_x0 = gutter_w + 1 * cell_px
        win_x1 = win_x0 + cell_px - 1
        _assert_rgb_close(im.getpixel((win_x0 + 1, header_h + cell_px // 2)), LE.WINNER_RGB, tol=16)
        _assert_rgb_close(im.getpixel((win_x1 - 1, header_h + cell_px // 2)), LE.WINNER_RGB, tol=16)
        _assert_rgb_close(im.getpixel((win_x0 + cell_px // 2, 1)), LE.WINNER_RGB, tol=16)
        _assert_rgb_close(im.getpixel((win_x0 + cell_px // 2, expected_h - 2)), LE.WINNER_RGB, tol=16)

        # Non-winner column (col 0) does not have winner outline at top edge
        non_win_top = im.getpixel((gutter_w + cell_px // 2, 1))
        assert abs(non_win_top[0] - LE.WINNER_RGB[0]) > 40

        # Assert row labels are drawn in the left gutter for every row (non-background pixels present)
        for row_idx in range(n_rows):
            y0 = header_h + row_idx * (cell_px + score_h)
            non_bg = 0
            for gx in range(4, gutter_w - 4):
                for gy in range(y0 + 4, y0 + cell_px - 4):
                    px_rgb = im.getpixel((gx, gy))
                    if any(abs(px_rgb[k] - LE.SURFACE_RGB[k]) > 35 for k in range(3)):
                        non_bg += 1
            assert non_bg >= 15, f"row {row_idx} gutter lacks label pixels"

    # 2. Check scores.xlsx sheets, null -> '-', bold winner row and rich Test sheet metadata
    wb = load_workbook(paths["xlsx"])
    assert wb.sheetnames == ["Cells", "Columns", "Test"]

    ws_cells = wb["Cells"]
    assert ws_cells.max_row == 1 + n_cols * n_rows
    cell_records = [
        (
            ws_cells.cell(r, 1).value,
            ws_cells.cell(r, 3).value,
            ws_cells.cell(r, 4).value,
            ws_cells.cell(r, 6).value,
            ws_cells.cell(r, 10).value,
        )
        for r in range(2, ws_cells.max_row + 1)
    ]
    assert ("s1", 0.8, 0, 2, "-") in cell_records

    ws_cols = wb["Columns"]
    assert ws_cols.max_row == 1 + n_cols
    bold_rows = []
    for r in range(2, ws_cols.max_row + 1):
        cid = ws_cols.cell(r, 1).value
        strength = ws_cols.cell(r, 3).value
        is_bold = bool(ws_cols.cell(r, 1).font and ws_cols.cell(r, 1).font.bold)
        if is_bold:
            bold_rows.append((cid, float(strength)))
            assert all(ws_cols.cell(r, col_j).font.bold for col_j in range(1, 10))
        else:
            assert not any(ws_cols.cell(r, col_j).font.bold for col_j in range(1, 10))
    assert bold_rows == [("c1", 1.0)]

    ws_test = wb["Test"]
    settings = {ws_test.cell(r, 1).value: ws_test.cell(r, 2).value for r in range(2, ws_test.max_row + 1)}
    assert settings["test_id"] == "t20261122-153012-a3f9"
    assert settings["name"] == "Elena: best vs final"
    assert settings["complete"] == "true"
    assert "c1 @ 1.00" in settings["winner"]
    # Contestants have lora, config and run_id
    assert "output/elena_v3/elena_v3_best.safetensors" in settings["contestants"]
    assert "output/elena_v3/config.yaml" in settings["contestants"]
    assert "elena_v3-20261120-101500" in settings["contestants"]
    assert "output/elena_v3/elena_v3.safetensors" in settings["contestants"]
    # Soups have 'of' list and weights='equal' when weights is null in spec.json
    assert "c1,c2" in settings["soups"]
    assert "equal" in settings["soups"]
    # Prompt key, render settings, scoring settings and devices
    assert "p0" in settings["prompt_key"] and "p1" in settings["prompt_key"]
    assert "headshot" in settings["p0"]
    assert "full body" in settings["p1"]
    assert settings["width"] == "1024" and settings["height"] == "1024"
    assert settings["steps"] == "25" and settings["cfg"] == "3.5"
    assert settings["negative"] == "blurry, deformed"
    assert settings["from_contestant"] == "c1"
    assert "refs/face1.png" in settings["face_refs"]
    assert "refs/body1.png" in settings["body_refs"]
    assert settings["devices"] == "cuda:0"

    # 3. Check report.pdf starts with %PDF and has 2 pages (1 cover/table + 1 grid page)
    pdf_bytes = paths["pdf"].read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    page_count = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
    assert page_count == 2


def test_multipage_grid_and_pdf_table_pagination(tmp_path):
    """Verify 30 columns (>26) split into 4 grid pages (8, 8, 8, 6) and 2 PDF table pages keeping winner at col 30."""
    test_dir = tmp_path / "t20261122-180000-beef"
    test_dir.mkdir(parents=True, exist_ok=True)

    # First test 20 columns -> 3 grid pages with cell=32, plus stale grid*.jpg removal
    export_dir = test_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    stale_single = export_dir / "grid.jpg"
    stale_old = export_dir / "grid_99.jpg"
    stale_single.write_bytes(b"stale")
    stale_old.write_bytes(b"stale")

    def write_test_with_n_cols(n_cols: int):
        contestants = [
            {
                "id": f"c{i + 1}",
                "label": f"Model_{i + 1:02d}",
                "lora": f"output/m{i + 1}/lora.safetensors",
                "config": f"output/m{i + 1}/config.yaml",
                "run_id": f"run-{i + 1}",
            }
            for i in range(n_cols)
        ]
        spec = {
            "version": 1,
            "test_id": "t20261122-180000-beef",
            "name": "Pagination test",
            "created": "2026-11-22T18:00:00Z",
            "contestants": contestants,
            "soups": [],
            "prompts": ["portrait photo", "full body shot"],
            "prompt_kinds": ["face", "body"],
            "seeds": [1, 2],
            "strengths": [1.0],
            "render": {"width": 512, "height": 512, "steps": 20, "cfg": 3.0, "negative": None},
            "scoring": {"face_refs": None, "body_refs": None, "from_contestant": "c1"},
            "devices": ["cuda:0"],
        }
        cells = []
        columns = []
        for c_idx, c in enumerate(contestants):
            cid = c["id"]
            score = round(40.0 + c_idx * 1.5, 1)
            for p_idx in range(2):
                for seed_idx, seed in enumerate(spec["seeds"]):
                    rel = f"images/{cid}/s1.00_p{p_idx}_seed{seed}.png"
                    _make_png(test_dir / rel, _cell_color(c_idx, 0, p_idx, seed_idx), size=(16, 16))
                    cells.append({
                        "contestant": cid,
                        "strength": 1.0,
                        "prompt": p_idx,
                        "seed": seed,
                        "image": rel,
                        "face": score,
                        "body": score,
                        "gold": score,
                        "combined": score,
                    })
            columns.append({
                "contestant": cid,
                "strength": 1.0,
                "face": score,
                "body": score,
                "gold": score,
                "combined": score,
                "n": 4,
            })
        win_cid = f"c{n_cols}"
        win_score = round(40.0 + (n_cols - 1) * 1.5, 1)
        results = {
            "version": 1,
            "test_id": spec["test_id"],
            "complete": True,
            "scale": "run",
            "scale_note": None,
            "contestants": contestants,
            "cells": cells,
            "columns": columns,
            "best_strength": {c["id"]: 1.0 for c in contestants},
            "winner": {"contestant": win_cid, "strength": 1.0, "combined": win_score},
            "render": {"width": 512, "height": 512, "steps": 20, "cfg": 3.0, "seconds": 50.0, "gpus": ["RTX 5060 Ti"]},
        }
        (test_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        (test_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")

    # 1. 20 columns -> 3 pages (8, 8, 4 columns) with cell=32
    write_test_with_n_cols(20)
    cell_px = 32
    gutter_w, header_h, score_h, _f, _sf = LE._grid_metrics(cell_px)
    grid_paths = LE.export_grid_pages(test_dir, cell=cell_px)
    assert [p.name for p in grid_paths] == ["grid_01.jpg", "grid_02.jpg", "grid_03.jpg"]
    assert LE.export_grid(test_dir, cell=cell_px) == export_dir / "grid_01.jpg"
    assert not stale_single.exists()
    assert not stale_old.exists()

    expected_cols_per_page = [8, 8, 4]
    for page_idx, (gpath, page_n_cols) in enumerate(zip(grid_paths, expected_cols_per_page)):
        with Image.open(gpath) as im:
            assert im.size == (gutter_w + page_n_cols * cell_px, header_h + 4 * (cell_px + score_h))
            # Check first column of this page has the expected contestant colour (c_idx = page_idx * 8)
            c_idx = page_idx * 8
            cx = gutter_w + cell_px // 2
            cy = header_h + cell_px // 2
            _assert_rgb_close(im.getpixel((cx, cy)), _cell_color(c_idx, 0, 0, 0), tol=10)

    # 2. 30 columns (>26) -> 2 PDF table pages + 4 PDF grid pages = 6 PDF pages, keeping winner c30
    write_test_with_n_cols(30)
    pdf_path = LE.export_pdf(test_dir, cell=cell_px)
    pdf_bytes = pdf_path.read_bytes()
    page_count = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
    assert page_count == 6  # 2 table pages (26 + 4 rows) + 4 grid pages (8 + 8 + 8 + 6 cols)

    # Verify winner column c30 is also outlined on the 4th grid page (column index 5 on page 4)
    grid_paths_30 = LE.export_grid_pages(test_dir, cell=cell_px)
    assert len(grid_paths_30) == 4
    with Image.open(grid_paths_30[3]) as im4:
        win_x0 = gutter_w + 5 * cell_px
        _assert_rgb_close(im4.getpixel((win_x0 + 1, header_h + cell_px // 2)), LE.WINNER_RGB, tol=16)


def test_outside_image_path_is_confined_and_drawn_as_missing(tmp_path):
    """An image path in results.json pointing outside test_dir (.. or absolute) must be drawn as missing."""
    outside_png = tmp_path / "secret_outside.png"
    _make_png(outside_png, (250, 10, 10), size=(64, 64))

    test_dir = _synthetic_test_dir(tmp_path, complete=True)
    res_path = test_dir / "results.json"
    results = json.loads(res_path.read_text(encoding="utf-8"))

    # Point cell (c1, 0.8, prompt 0, seed 1) to '../secret_outside.png'
    # and cell (c1, 0.8, prompt 0, seed 2) to the absolute path of outside_png
    for rec in results["cells"]:
        if rec["contestant"] == "c1" and rec["strength"] == 0.8 and rec["prompt"] == 0 and rec["seed"] == 1:
            rec["image"] = "../secret_outside.png"
        elif rec["contestant"] == "c1" and rec["strength"] == 0.8 and rec["prompt"] == 0 and rec["seed"] == 2:
            rec["image"] = str(outside_png.resolve())
    res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    assert LE._cell_image_path(test_dir, "c1", 0.8, 0, 1, {"image": "../secret_outside.png"}) is None
    assert LE._cell_image_path(test_dir, "c1", 0.8, 0, 2, {"image": str(outside_png.resolve())}) is None

    cell_px = 96
    gutter_w, header_h, score_h, _f, _sf = LE._grid_metrics(cell_px)
    grid_path = LE.export_grid(test_dir, cell=cell_px)
    with Image.open(grid_path) as im:
        # Both row 0 (p0, seed 1) and row 1 (p0, seed 2) in col 0 (c1 @ 0.8) must be MISSING_BG_RGB, not (250, 10, 10)
        for row_idx in (0, 1):
            cx = gutter_w + cell_px // 2
            cy = header_h + row_idx * (cell_px + score_h) + 8
            _assert_rgb_close(im.getpixel((cx, cy)), LE.MISSING_BG_RGB, tol=10)


def test_incomplete_test_and_cli_only_flags(tmp_path, capsys):
    test_dir = _synthetic_test_dir(tmp_path, complete=False)

    # Run CLI with --only grid and --cell 96
    rc = LE.main([str(test_dir), "--only", "grid", "--cell", "96"])
    assert rc == 0
    out_lines = capsys.readouterr().out.strip().splitlines()
    assert len(out_lines) == 1
    grid_path = test_dir / "export" / "grid.jpg"
    assert grid_path.is_file()
    assert not (test_dir / "export" / "scores.xlsx").exists()
    assert not (test_dir / "export" / "report.pdf").exists()
    gutter_w, header_h, score_h, _f, _sf = LE._grid_metrics(96)
    with Image.open(grid_path) as im:
        assert im.size == (gutter_w + 6 * 96, header_h + 4 * (96 + score_h))

    # Run CLI with --only xlsx
    assert LE.main([str(test_dir), "--only", "xlsx"]) == 0
    capsys.readouterr()
    xlsx_path = test_dir / "export" / "scores.xlsx"
    assert xlsx_path.is_file()
    wb = load_workbook(xlsx_path)
    ws_cols = wb["Columns"]
    # s1 @ 1.0 had no rendered cells in the incomplete test, so its scores are '-'
    s1_10 = [
        [ws_cols.cell(r, j).value for j in range(1, 9)]
        for r in range(2, ws_cols.max_row + 1)
        if ws_cols.cell(r, 1).value == "s1" and float(ws_cols.cell(r, 3).value) == 1.0
    ]
    assert len(s1_10) == 1
    assert s1_10[0][3:7] == ["-", "-", "-", "-"]
    assert s1_10[0][7] == 0
    ws_test = wb["Test"]
    settings = {ws_test.cell(r, 1).value: ws_test.cell(r, 2).value for r in range(2, ws_test.max_row + 1)}
    assert settings["complete"] == "false"

    # Run CLI with --only pdf
    assert LE.main([str(test_dir), "--only", "pdf"]) == 0
    capsys.readouterr()
    pdf_path = test_dir / "export" / "report.pdf"
    assert pdf_path.is_file()
    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page\b", pdf_bytes)) >= 2

    # Missing folder returns non-zero exit status on CLI
    assert LE.main([str(tmp_path / "does_not_exist")]) == 1
    assert "loratest_export" in capsys.readouterr().err
