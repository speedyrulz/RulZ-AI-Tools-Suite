"""Export a finished or stopped LoRA test into <test_dir>/export/: grid.jpg, scores.xlsx and report.pdf.

    python -m rulztrainer.loratest_export output/_tests/<test_id>
    python -m rulztrainer.loratest_export output/_tests/<test_id> --only grid --cell 256

Follows section 5 of docs/lora-test-contract.md. Everything comes from spec.json, results.json and the rendered
pictures already in <test_dir>, so exports need no GPU, no model and no network. Incomplete tests (complete: false)
and unscored pictures (null scores) export cleanly with '-' for missing scores and a grey 'missing' tile for any
picture not on disk or outside <test_dir>.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path

HEADER_HEIGHT = 48
SCORE_HEIGHT = 24
GUTTER_WIDTH = 112
DEFAULT_MAX_COLS = 8
DEFAULT_MAX_ROWS = 8
PDF_TABLE_ROWS_PER_PAGE = 26
PROMPT_CLIP_CHARS = 72
FONT_NAME = "Arial"

# Palette shared with runreport.py
INK_RGB = (26, 26, 25)
MUTED_RGB = (107, 106, 99)
GRID_RGB = (228, 227, 221)
SURFACE_RGB = (244, 243, 239)
CARD_RGB = (255, 255, 255)
BAND_RGB = (31, 41, 55)
BAND_INK_RGB = (255, 255, 255)
BAND_MUTED_RGB = (182, 192, 207)
MISSING_BG_RGB = (209, 213, 219)
MISSING_EDGE_RGB = (156, 163, 175)
MISSING_INK_RGB = (75, 85, 99)
WINNER_RGB = (201, 133, 0)


def _load_test(test_dir: str | Path) -> tuple[Path, dict, dict]:
    """Read spec.json and results.json from test_dir; raise FileNotFoundError when the folder or files are missing."""
    root = Path(test_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"No test folder at {root}")
    spec_path = root / "spec.json"
    res_path = root / "results.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"Missing spec.json in {root}")
    if not res_path.is_file():
        raise FileNotFoundError(f"Missing results.json in {root}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    results = json.loads(res_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(results, dict):
        raise ValueError(f"spec.json and results.json in {root} must hold JSON objects")
    return root, spec, results


def _fmt_score(value, digits: int = 1) -> str:
    """Format a score on the 0-100 scale, or '-' when null / non-finite."""
    if value is None or isinstance(value, bool):
        return "-"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{num:.{digits}f}" if math.isfinite(num) else "-"


def _skey(strength) -> float:
    return round(float(strength), 4)


def _clip_chars(text: str, limit: int = PROMPT_CLIP_CHARS) -> str:
    """Collapse whitespace and clip text to limit characters with '...'."""
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: max(1, limit - 3)].rstrip() + "..."


def _is_winner(winner: dict | None, contestant_id: str, strength: float) -> bool:
    """True when (contestant_id, strength) is the winning column recorded in results.json."""
    if not isinstance(winner, dict):
        return False
    w_id = winner.get("contestant")
    w_str = winner.get("strength")
    if w_id is None or w_str is None:
        return False
    try:
        return str(w_id) == str(contestant_id) and _skey(w_str) == _skey(strength)
    except (TypeError, ValueError):
        return False


def _contestants(spec: dict, results: dict) -> list[dict]:
    """All file and soup contestants in spec order, enriched with any metadata results.json added."""
    by_id: dict[str, dict] = {}
    order: list[str] = []
    sources = [
        (spec.get("contestants") or [], "file", True),
        (spec.get("soups") or [], "soup", True),
        (results.get("contestants") or [], None, False),
    ]
    for items, default_kind, from_spec in sources:
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            cid = str(item["id"])
            if cid not in by_id:
                order.append(cid)
                entry = dict(item)
                if default_kind and not entry.get("kind"):
                    entry["kind"] = default_kind
                if from_spec and entry.get("kind") == "soup":
                    entry["_spec_weights_present"] = True
                    entry["_spec_weights"] = item.get("weights")
                entry.setdefault("label", cid)
                by_id[cid] = entry
            else:
                for k, v in item.items():
                    if v is not None and by_id[cid].get(k) is None:
                        by_id[cid][k] = v
    return [by_id[cid] for cid in order]


def _grid_axes(spec: dict, results: dict) -> tuple[list[dict], list[float], list[tuple[int, int]], list[tuple[dict, float]]]:
    """Return (contestants, strengths, rows [(prompt_idx, seed)], cols [(contestant_dict, strength)])."""
    contestants = _contestants(spec, results)
    raw_strengths = spec.get("strengths") or []
    strengths = [float(s) for s in raw_strengths]
    if not strengths:
        seen_s: list[float] = []
        for rec in (results.get("columns") or []) + (results.get("cells") or []):
            if isinstance(rec, dict) and rec.get("strength") is not None:
                val = float(rec["strength"])
                if all(_skey(val) != _skey(x) for x in seen_s):
                    seen_s.append(val)
        strengths = seen_s

    prompts = list(spec.get("prompts") or [])
    n_prompts = len(prompts)
    if n_prompts == 0:
        p_indices = [int(c["prompt"]) for c in (results.get("cells") or []) if isinstance(c, dict) and c.get("prompt") is not None]
        n_prompts = (max(p_indices) + 1) if p_indices else 0

    seeds = [int(s) for s in (spec.get("seeds") or [])]
    if not seeds:
        seen_seeds: list[int] = []
        for rec in results.get("cells") or []:
            if isinstance(rec, dict) and rec.get("seed") is not None:
                sd = int(rec["seed"])
                if sd not in seen_seeds:
                    seen_seeds.append(sd)
        seeds = seen_seeds

    rows = [(p_idx, seed) for p_idx in range(n_prompts) for seed in seeds]
    cols = [(c, s) for c in contestants for s in strengths]
    return contestants, strengths, rows, cols


def _cell_lookup(results: dict) -> dict[tuple[str, float, int, int], dict]:
    out: dict[tuple[str, float, int, int], dict] = {}
    for rec in results.get("cells") or []:
        if not isinstance(rec, dict):
            continue
        try:
            key = (str(rec["contestant"]), _skey(rec["strength"]), int(rec["prompt"]), int(rec["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = rec
    return out


def _column_lookup(results: dict) -> dict[tuple[str, float], dict]:
    out: dict[tuple[str, float], dict] = {}
    for rec in results.get("columns") or []:
        if not isinstance(rec, dict):
            continue
        try:
            key = (str(rec["contestant"]), _skey(rec["strength"]))
        except (KeyError, TypeError, ValueError):
            continue
        out[key] = rec
    return out


def _cell_image_path(
    test_dir: Path,
    cid: str,
    strength: float,
    prompt_idx: int,
    seed: int,
    cell_rec: dict | None,
) -> Path | None:
    """Resolve a cell's image path and confine it inside test_dir; return None if it escapes."""
    try:
        root_resolved = test_dir.resolve()
    except OSError:
        return None

    raw_img = cell_rec.get("image") if isinstance(cell_rec, dict) else None
    if raw_img is not None and str(raw_img).strip():
        raw_str = str(raw_img).strip()
        rel = Path(raw_str)
        if rel.is_absolute() or rel.drive or rel.root or raw_str.startswith(("/", "\\")) or ".." in rel.parts:
            return None
        candidate = test_dir / rel
    else:
        if not cid or "/" in cid or "\\" in cid or ".." in cid:
            return None
        candidate = test_dir / "images" / cid / f"s{float(strength):.2f}_p{prompt_idx}_seed{seed}.png"

    try:
        resolved = candidate.resolve()
        resolved.relative_to(root_resolved)
    except (OSError, ValueError):
        return None
    if resolved == root_resolved:
        return None
    return resolved


def _fit_cell_image(im, cell: int):
    """Scale a picture to fit a cell x cell square while keeping its aspect ratio."""
    from PIL import Image

    im = im.convert("RGB")
    iw, ih = im.size
    if iw == cell and ih == cell:
        return im
    scale = min(cell / max(1, iw), cell / max(1, ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    if nw == cell and nh == cell:
        return resized
    tile = Image.new("RGB", (cell, cell), CARD_RGB)
    tile.paste(resized, ((cell - nw) // 2, (cell - nh) // 2))
    return tile


@functools.lru_cache(maxsize=1)
def _ttf_font_path() -> str | None:
    """Locate DejaVu Sans via matplotlib.font_manager if available."""
    try:
        from matplotlib import font_manager

        path = font_manager.findfont("DejaVu Sans")
        if path and Path(path).is_file():
            return str(path)
    except Exception:  # noqa: BLE001
        pass
    return None


@functools.lru_cache(maxsize=32)
def _load_font(size: int):
    """Load a scalable TrueType font at the requested pixel size, falling back to Pillow's default font."""
    from PIL import ImageFont

    size = max(7, int(size))
    ttf = _ttf_font_path()
    if ttf:
        try:
            return ImageFont.truetype(ttf, size=size)
        except Exception:  # noqa: BLE001
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _header_height(cell: int) -> int:
    return max(18, int(round(int(cell) * HEADER_HEIGHT / 256)))


def _score_height(cell: int) -> int:
    return max(12, int(round(int(cell) * SCORE_HEIGHT / 256)))


def _gutter_width(cell: int) -> int:
    return max(52, int(round(int(cell) * GUTTER_WIDTH / 256)))


def _grid_metrics(cell: int) -> tuple[int, int, int, int, int]:
    """Return (gutter_width, header_height, score_height, main_font_px, small_font_px) scaled from cell."""
    cell = int(cell)
    if cell <= 0:
        raise ValueError("cell size must be positive")
    gutter_w = _gutter_width(cell)
    header_h = _header_height(cell)
    score_h = _score_height(cell)
    font_px = max(8, int(round(cell * 13 / 256)))
    small_px = max(7, int(round(cell * 11 / 256)))
    return gutter_w, header_h, score_h, font_px, small_px


def _text_box(draw, text: str, font) -> tuple[int, int]:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0]), int(box[3] - box[1])
    except Exception:  # noqa: BLE001
        return len(text) * 7, 11


def _text_width(draw, text: str, font) -> int:
    return _text_box(draw, text, font)[0]


def _clip_text(draw, text: str, font, max_px: int) -> str:
    text = " ".join(str(text).split())
    if _text_width(draw, text, font) <= max_px:
        return text
    while len(text) > 1 and _text_width(draw, text + "...", font) > max_px:
        text = text[:-1].rstrip()
    return text + "..." if len(text) > 1 else text


def _row_label(prompt_idx: int, kind: str | None, seed: int) -> str:
    """Format a row label for the left gutter, e.g. 'p1 face, seed 2'."""
    k = str(kind).strip() if kind and str(kind).strip() not in ("", "-") else ""
    head = f"p{int(prompt_idx)} {k}".strip()
    return f"{head}, seed {int(seed)}"


def _prompt_key_entries(spec: dict, n_prompts: int | None = None, clip: int = PROMPT_CLIP_CHARS) -> list[tuple[int, str, str]]:
    """Return [(prompt_idx, kind, clipped_prompt_text)] for the prompt key."""
    prompts = list(spec.get("prompts") or [])
    kinds = list(spec.get("prompt_kinds") or [])
    total = len(prompts) if n_prompts is None else max(len(prompts), int(n_prompts))
    entries: list[tuple[int, str, str]] = []
    for idx in range(total):
        kind = str(kinds[idx]).strip() if idx < len(kinds) and kinds[idx] is not None else "-"
        raw_text = str(prompts[idx]) if idx < len(prompts) and prompts[idx] is not None else "-"
        entries.append((idx, kind or "-", _clip_chars(raw_text, limit=clip)))
    return entries


def _paginate_slices(items: list, step: int) -> list[list]:
    step = max(1, int(step))
    if not items:
        return [[]]
    return [items[i : i + step] for i in range(0, len(items), step)]


def _build_grid_page(
    test_dir: Path,
    spec: dict,
    results: dict,
    page_rows: list[tuple[int, int]],
    page_cols: list[tuple[dict, float]],
    cells: dict[tuple[str, float, int, int], dict],
    columns: dict[tuple[str, float], dict],
    winner: dict | None,
    cell: int = 256,
):
    """Build one PIL grid page with top header, left row-label gutter, cells, score bands and winner outline."""
    from PIL import Image, ImageDraw

    gutter_w, header_h, score_h, font_px, small_px = _grid_metrics(cell)
    font = _load_font(font_px)
    small_font = _load_font(small_px)
    kinds = list(spec.get("prompt_kinds") or [])

    n_rows = len(page_rows)
    n_cols = len(page_cols)
    width = gutter_w + max(1, n_cols * cell)
    height = header_h + max(1, n_rows * (cell + score_h))

    canvas = Image.new("RGB", (width, height), SURFACE_RGB)
    draw = ImageDraw.Draw(canvas)

    # Top header band across gutter + columns
    draw.rectangle([0, 0, width - 1, header_h - 1], fill=BAND_RGB)
    corner_txt = _clip_text(draw, "Prompt / Seed", small_font, max(16, gutter_w - 8))
    cw, ch = _text_box(draw, corner_txt, small_font)
    draw.text(
        (max(3, (gutter_w - cw) // 2), max(2, (header_h - ch) // 2)),
        corner_txt,
        fill=BAND_MUTED_RGB,
        font=small_font,
    )
    draw.line([(gutter_w - 1, 0), (gutter_w - 1, height - 1)], fill=GRID_RGB, width=1)

    for col_idx, (c, strength) in enumerate(page_cols):
        x0 = gutter_w + col_idx * cell
        cid = str(c["id"])
        label = str(c.get("label") or cid)
        col_rec = columns.get((cid, _skey(strength))) or {}
        col_score = _fmt_score(col_rec.get("combined"))
        is_win = _is_winner(winner, cid, strength)

        line1 = _clip_text(draw, ("* " if is_win else "") + label, font, max(16, cell - 8))
        line2 = _clip_text(draw, f"s={strength:.2f} ({col_score})", small_font, max(16, cell - 8))
        w1, h1 = _text_box(draw, line1, font)
        w2, h2 = _text_box(draw, line2, small_font)
        y1 = max(2, (header_h // 2 - h1) // 2)
        y2 = max(y1 + h1 + 1, header_h // 2 + (header_h // 2 - h2) // 2)
        draw.text((x0 + max(2, (cell - w1) // 2), y1), line1, fill=BAND_INK_RGB, font=font)
        draw.text((x0 + max(2, (cell - w2) // 2), y2), line2, fill=BAND_MUTED_RGB, font=small_font)
        draw.line([(x0, 0), (x0, header_h - 1)], fill=GRID_RGB, width=1)

    # Rows: left gutter label + cell tiles + score bands
    for row_idx, (prompt_idx, seed) in enumerate(page_rows):
        y0 = header_h + row_idx * (cell + score_h)
        sy0 = y0 + cell
        sy1 = sy0 + score_h - 1
        kind = str(kinds[prompt_idx]) if prompt_idx < len(kinds) and kinds[prompt_idx] is not None else ""
        full_row_label = _row_label(prompt_idx, kind, seed)

        # Left gutter cell
        draw.rectangle([0, y0, gutter_w - 1, sy1], fill=SURFACE_RGB, outline=GRID_RGB)
        max_gutter_text = max(16, gutter_w - 8)
        if _text_width(draw, full_row_label, small_font) <= max_gutter_text:
            lw, lh = _text_box(draw, full_row_label, small_font)
            draw.text(
                (max(3, (gutter_w - lw) // 2), y0 + max(2, (cell - lh) // 2)),
                full_row_label,
                fill=INK_RGB,
                font=small_font,
            )
        else:
            head_part, _, seed_part = full_row_label.partition(", ")
            l1 = _clip_text(draw, head_part, small_font, max_gutter_text)
            l2 = _clip_text(draw, seed_part or f"seed {seed}", small_font, max_gutter_text)
            w1, h1 = _text_box(draw, l1, small_font)
            w2, h2 = _text_box(draw, l2, small_font)
            total_h = h1 + h2 + 2
            top_y = y0 + max(2, (cell - total_h) // 2)
            draw.text((max(3, (gutter_w - w1) // 2), top_y), l1, fill=INK_RGB, font=small_font)
            draw.text((max(3, (gutter_w - w2) // 2), top_y + h1 + 2), l2, fill=INK_RGB, font=small_font)

        for col_idx, (c, strength) in enumerate(page_cols):
            x0 = gutter_w + col_idx * cell
            cid = str(c["id"])
            cell_rec = cells.get((cid, _skey(strength), prompt_idx, seed))
            img_path = _cell_image_path(test_dir, cid, strength, prompt_idx, seed, cell_rec)

            tile = None
            if img_path is not None and img_path.is_file():
                try:
                    with Image.open(img_path) as raw:
                        tile = _fit_cell_image(raw, cell)
                except Exception:  # noqa: BLE001
                    tile = None

            if tile is not None:
                canvas.paste(tile, (x0, y0))
            else:
                draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=MISSING_BG_RGB, outline=MISSING_EDGE_RGB)
                word = _clip_text(draw, "missing", small_font, max(12, cell - 4))
                ww, wh = _text_box(draw, word, small_font)
                draw.text(
                    (x0 + max(1, (cell - ww) // 2), y0 + max(1, (cell - wh) // 2)),
                    word,
                    fill=MISSING_INK_RGB,
                    font=small_font,
                )

            draw.rectangle([x0, sy0, x0 + cell - 1, sy1], fill=CARD_RGB, outline=GRID_RGB)
            score_str = _fmt_score(cell_rec.get("combined") if isinstance(cell_rec, dict) else None)
            sw, sh = _text_box(draw, score_str, small_font)
            draw.text(
                (x0 + max(1, (cell - sw) // 2), sy0 + max(1, (score_h - sh) // 2)),
                score_str,
                fill=INK_RGB,
                font=small_font,
            )

    # Winner column outline drawn on top so it frames the full column clearly
    for col_idx, (c, strength) in enumerate(page_cols):
        if _is_winner(winner, str(c["id"]), strength):
            x0 = gutter_w + col_idx * cell
            x1 = x0 + cell - 1
            for offset in range(4):
                draw.rectangle([x0 + offset, offset, x1 - offset, height - 1 - offset], outline=WINNER_RGB)

    return canvas


def _build_grid_pages(
    test_dir: Path,
    spec: dict,
    results: dict,
    cell: int = 256,
    max_cols: int = DEFAULT_MAX_COLS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> list:
    """Build all paginated PIL grid images (at most max_cols columns and max_rows rows per page)."""
    cell = int(cell)
    if cell <= 0:
        raise ValueError("cell size must be positive")
    max_cols = max(1, int(max_cols))
    max_rows = max(1, int(max_rows))

    _contestants_list, _strengths, rows, cols = _grid_axes(spec, results)
    cells = _cell_lookup(results)
    columns = _column_lookup(results)
    winner = results.get("winner") if isinstance(results.get("winner"), dict) else None

    row_chunks = _paginate_slices(rows, max_rows)
    col_chunks = _paginate_slices(cols, max_cols)

    pages = []
    for r_chunk in row_chunks:
        for c_chunk in col_chunks:
            pages.append(
                _build_grid_page(
                    test_dir,
                    spec,
                    results,
                    r_chunk,
                    c_chunk,
                    cells,
                    columns,
                    winner,
                    cell=cell,
                )
            )
    return pages


def _build_grid_image(
    test_dir: Path,
    spec: dict,
    results: dict,
    cell: int = 256,
    max_cols: int = DEFAULT_MAX_COLS,
    max_rows: int = DEFAULT_MAX_ROWS,
):
    """Build the first grid page PIL image."""
    return _build_grid_pages(test_dir, spec, results, cell=cell, max_cols=max_cols, max_rows=max_rows)[0]


def export_grid_pages(
    test_dir: str | Path,
    cell: int = 256,
    max_cols: int = DEFAULT_MAX_COLS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> list[Path]:
    """Write grid.jpg (if 1 page) or grid_01.jpg, grid_02.jpg, ... after removing stale grid*.jpg files."""
    root, spec, results = _load_test(test_dir)
    pages = _build_grid_pages(root, spec, results, cell=cell, max_cols=max_cols, max_rows=max_rows)
    export_dir = root / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    for stale in export_dir.glob("grid*.jpg"):
        if stale.is_file():
            stale.unlink()

    if len(pages) == 1:
        out = export_dir / "grid.jpg"
        pages[0].save(out, format="JPEG", quality=95, subsampling=0)
        return [out]

    paths: list[Path] = []
    for idx, im in enumerate(pages, start=1):
        out = export_dir / f"grid_{idx:02d}.jpg"
        im.save(out, format="JPEG", quality=95, subsampling=0)
        paths.append(out)
    return paths


def export_grid(
    test_dir: str | Path,
    cell: int = 256,
    max_cols: int = DEFAULT_MAX_COLS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> Path:
    """Write the grid page(s) into <test_dir>/export/ and return the first page's path."""
    return export_grid_pages(test_dir, cell=cell, max_cols=max_cols, max_rows=max_rows)[0]


def _format_soup_weights(c: dict) -> str:
    weights = c.get("_spec_weights") if c.get("_spec_weights_present") else c.get("weights")
    if weights is None:
        return "equal"
    if isinstance(weights, (list, tuple)):
        return ",".join(str(w) for w in weights) if weights else "equal"
    return str(weights)


def _spec_setting_rows(spec: dict, results: dict, contestants: list[dict]) -> list[tuple[str, str]]:
    """Key/value pairs describing the test settings for the Test sheet and PDF summary."""
    winner = results.get("winner") if isinstance(results.get("winner"), dict) else None
    if winner and winner.get("contestant") is not None and winner.get("strength") is not None:
        win_text = f"{winner['contestant']} @ {float(winner['strength']):.2f} ({_fmt_score(winner.get('combined'))})"
    else:
        win_text = "-"

    c_lines: list[str] = []
    c_detail_rows: list[tuple[str, str]] = []
    s_lines: list[str] = []
    s_detail_rows: list[tuple[str, str]] = []
    for c in contestants:
        cid = str(c["id"])
        label = str(c.get("label") or cid)
        if c.get("kind") == "soup":
            of_str = ",".join(str(x) for x in (c.get("of") or [])) or "-"
            w_str = _format_soup_weights(c)
            lora = str(c.get("lora") or "-")
            s_lines.append(f"{cid}: {label} (of={of_str}, weights={w_str})")
            s_detail_rows.append((f"soup.{cid}", f"label={label}; of={of_str}; weights={w_str}; lora={lora}"))
        else:
            lora = str(c.get("lora") or "-")
            cfg = str(c.get("config") or "-")
            run_id = str(c.get("run_id") or "-")
            c_lines.append(f"{cid}: {label} (lora={lora}, config={cfg}, run_id={run_id})")
            c_detail_rows.append((f"contestant.{cid}", f"label={label}; lora={lora}; config={cfg}; run_id={run_id}"))

    prompts = list(spec.get("prompts") or [])
    kinds = list(spec.get("prompt_kinds") or [])
    seeds = list(spec.get("seeds") or [])
    strengths = list(spec.get("strengths") or [])
    prompt_entries = _prompt_key_entries(spec, n_prompts=len(prompts))
    prompt_key_summary = " | ".join(f"p{idx} ({kind}): {txt}" for idx, kind, txt in prompt_entries) if prompt_entries else "-"

    render_spec = spec.get("render") if isinstance(spec.get("render"), dict) else {}
    render_res = results.get("render") if isinstance(results.get("render"), dict) else {}
    scoring = spec.get("scoring") if isinstance(spec.get("scoring"), dict) else {}

    def pick_render(key):
        val = render_res.get(key) if render_res.get(key) is not None else render_spec.get(key)
        return "-" if val is None else str(val)

    width_s = pick_render("width")
    height_s = pick_render("height")
    steps_s = pick_render("steps")
    cfg_s = pick_render("cfg")
    neg_s = "-" if render_spec.get("negative") is None else str(render_spec.get("negative"))
    render_summary = f"width={width_s}, height={height_s}, steps={steps_s}, cfg={cfg_s}, negative={neg_s}"

    from_c = str(scoring.get("from_contestant") or "-")
    face_refs = ", ".join(str(x) for x in (scoring.get("face_refs") or [])) if scoring.get("face_refs") else "-"
    body_refs = ", ".join(str(x) for x in (scoring.get("body_refs") or [])) if scoring.get("body_refs") else "-"
    scoring_summary = f"from_contestant={from_c}, face_refs={face_refs}, body_refs={body_refs}"
    devices_s = ", ".join(str(d) for d in (spec.get("devices") or [])) if spec.get("devices") else "-"

    rows: list[tuple[str, str]] = [
        ("test_id", str(spec.get("test_id") or results.get("test_id") or "-")),
        ("name", str(spec.get("name") or "-")),
        ("created", str(spec.get("created") or "-")),
        ("complete", "true" if results.get("complete", True) else "false"),
        ("scale", str(results.get("scale") or "-")),
        ("scale_note", str(results.get("scale_note") or "-")),
        ("winner", win_text),
        ("contestants", "; ".join(c_lines) if c_lines else "-"),
        *c_detail_rows,
        ("soups", "; ".join(s_lines) if s_lines else "-"),
        *s_detail_rows,
        ("prompts", " | ".join(str(p) for p in prompts) if prompts else "-"),
        ("prompt_kinds", ", ".join(str(k) for k in kinds) if kinds else "-"),
        ("prompt_key", prompt_key_summary),
        *[(f"p{idx}", f"({kind}) {txt}") for idx, kind, txt in prompt_entries],
        ("seeds", ", ".join(str(s) for s in seeds) if seeds else "-"),
        ("strengths", ", ".join(f"{float(s):.2f}" for s in strengths) if strengths else "-"),
        ("render", render_summary),
        ("width", width_s),
        ("height", height_s),
        ("steps", steps_s),
        ("cfg", cfg_s),
        ("negative", neg_s),
        ("scoring", scoring_summary),
        ("from_contestant", from_c),
        ("face_refs", face_refs),
        ("body_refs", body_refs),
        ("devices", devices_s),
    ]
    return rows


def export_xlsx(test_dir: str | Path) -> Path:
    """Write <test_dir>/export/scores.xlsx (sheets Cells, Columns with winner row bold, Test) and return its path."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    root, spec, results = _load_test(test_dir)
    contestants, _strengths, rows, cols = _grid_axes(spec, results)
    label_of = {str(c["id"]): str(c.get("label") or c["id"]) for c in contestants}
    kinds = list(spec.get("prompt_kinds") or [])
    cells = _cell_lookup(results)
    columns = _column_lookup(results)
    winner = results.get("winner") if isinstance(results.get("winner"), dict) else None

    wb = Workbook()
    ws_cells = wb.active
    ws_cells.title = "Cells"
    ws_cols = wb.create_sheet("Columns")
    ws_test = wb.create_sheet("Test")

    hdr_font = Font(name=FONT_NAME, size=10, bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", fgColor="1F2937")
    txt_font = Font(name=FONT_NAME, size=10, bold=False)
    bold_font = Font(name=FONT_NAME, size=10, bold=True)
    win_fill = PatternFill("solid", fgColor="FBF3DC")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def set_header(ws, headers: list[str], widths: list[int]):
        for j, text in enumerate(headers, 1):
            c = ws.cell(1, j, text)
            c.font = hdr_font
            c.fill = hdr_fill
            c.alignment = center
        ws.row_dimensions[1].height = 28
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = w

    def write_score_cell(ws, row_num: int, col_num: int, val, font: Font):
        if val is None or isinstance(val, bool):
            c = ws.cell(row_num, col_num, "-")
            c.font = font
            return c
        try:
            num = float(val)
        except (TypeError, ValueError):
            num = math.nan
        if not math.isfinite(num):
            c = ws.cell(row_num, col_num, "-")
        else:
            c = ws.cell(row_num, col_num, round(num, 1))
            c.number_format = "0.0"
        c.font = font
        return c

    # 1. Cells sheet: one row per cell
    set_header(
        ws_cells,
        ["Contestant", "Label", "Strength", "Prompt", "Kind", "Seed", "Face", "Body", "Gold", "Combined", "Image"],
        [14, 24, 10, 8, 10, 8, 10, 10, 10, 12, 36],
    )
    cell_row = 2
    for c, strength in cols:
        cid = str(c["id"])
        for prompt_idx, seed in rows:
            rec = cells.get((cid, _skey(strength), prompt_idx, seed)) or {}
            kind = str(kinds[prompt_idx]) if prompt_idx < len(kinds) else "-"
            img_rel = rec.get("image") or f"images/{cid}/s{float(strength):.2f}_p{prompt_idx}_seed{seed}.png"
            ws_cells.cell(cell_row, 1, cid).font = txt_font
            ws_cells.cell(cell_row, 2, label_of.get(cid, cid)).font = txt_font
            sc = ws_cells.cell(cell_row, 3, round(float(strength), 2))
            sc.font = txt_font
            sc.number_format = "0.00"
            ws_cells.cell(cell_row, 4, int(prompt_idx)).font = txt_font
            ws_cells.cell(cell_row, 5, kind).font = txt_font
            ws_cells.cell(cell_row, 6, int(seed)).font = txt_font
            write_score_cell(ws_cells, cell_row, 7, rec.get("face"), txt_font)
            write_score_cell(ws_cells, cell_row, 8, rec.get("body"), txt_font)
            write_score_cell(ws_cells, cell_row, 9, rec.get("gold"), txt_font)
            write_score_cell(ws_cells, cell_row, 10, rec.get("combined"), txt_font)
            ws_cells.cell(cell_row, 11, str(img_rel)).font = txt_font
            cell_row += 1

    # 2. Columns sheet: one row per column, winner row bold
    set_header(
        ws_cols,
        ["Contestant", "Label", "Strength", "Face", "Body", "Gold", "Combined", "Cells", "Winner"],
        [14, 24, 10, 10, 10, 10, 12, 8, 10],
    )
    col_row = 2
    for c, strength in cols:
        cid = str(c["id"])
        rec = columns.get((cid, _skey(strength))) or {}
        is_win = _is_winner(winner, cid, strength)
        row_font = bold_font if is_win else txt_font

        ws_cols.cell(col_row, 1, cid).font = row_font
        ws_cols.cell(col_row, 2, label_of.get(cid, cid)).font = row_font
        sc = ws_cols.cell(col_row, 3, round(float(strength), 2))
        sc.font = row_font
        sc.number_format = "0.00"
        write_score_cell(ws_cols, col_row, 4, rec.get("face"), row_font)
        write_score_cell(ws_cols, col_row, 5, rec.get("body"), row_font)
        write_score_cell(ws_cols, col_row, 6, rec.get("gold"), row_font)
        write_score_cell(ws_cols, col_row, 7, rec.get("combined"), row_font)
        nc = ws_cols.cell(col_row, 8, int(rec.get("n") or 0))
        nc.font = row_font
        wc = ws_cols.cell(col_row, 9, "winner" if is_win else "")
        wc.font = row_font
        if is_win:
            for j in range(1, 10):
                ws_cols.cell(col_row, j).fill = win_fill
        col_row += 1

    # 3. Test sheet: the spec's settings and prompt key
    set_header(ws_test, ["Setting", "Value"], [24, 88])
    for idx, (k, v) in enumerate(_spec_setting_rows(spec, results, contestants), start=2):
        ck = ws_test.cell(idx, 1, k)
        ck.font = bold_font
        cv = ws_test.cell(idx, 2, v)
        cv.font = txt_font

    out = root / "export" / "scores.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def export_pdf(
    test_dir: str | Path,
    cell: int = 256,
    max_cols: int = DEFAULT_MAX_COLS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> Path:
    """Write <test_dir>/export/report.pdf with prompt key, paginated column table and paginated grid pages."""
    from matplotlib.backends.backend_pdf import PdfPages

    from . import runreport as R

    root, spec, results = _load_test(test_dir)
    contestants, _strengths, rows, cols = _grid_axes(spec, results)
    label_of = {str(c["id"]): str(c.get("label") or c["id"]) for c in contestants}
    columns = _column_lookup(results)
    winner = results.get("winner") if isinstance(results.get("winner"), dict) else None
    complete = bool(results.get("complete", True))
    test_name = str(spec.get("name") or spec.get("test_id") or results.get("test_id") or root.name)
    test_id = str(spec.get("test_id") or results.get("test_id") or root.name)
    prompt_entries = _prompt_key_entries(spec)

    plt = R._mpl()
    W, H = R.A4
    out_path = root / "export" / "report.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grid_pages = _build_grid_pages(root, spec, results, cell=cell, max_cols=max_cols, max_rows=max_rows)
    table_chunks = _paginate_slices(cols, PDF_TABLE_ROWS_PER_PAGE)

    table_cols = [
        ("Contestant", 0.0),
        ("Label", 1.05),
        ("Strength", 3.1),
        ("Face", 3.95),
        ("Body", 4.65),
        ("Gold", 5.35),
        ("Combined", 6.05),
        ("Cells", 6.8),
    ]

    def make_page(title: str, page_num: int):
        page = R.Page(plt, W, H, dpi=150)
        page.box(0, 0, W, 0.62, color=R.BAND, edge="none", radius=0.0, z=0)
        page.text(0.5, 0.2, R._short(test_name, 42), size=10, color=R.BAND_INK, weight="bold")
        page.text(W - 0.5, 0.22, title, size=9, color=R.BAND_MUTED, ha="right")
        page.text(W / 2, H - 0.3, f"{page_num}", size=7.5, color=R.FAINT, ha="center")
        page.text(0.5, H - 0.3, f"RulZ AI Trainer {R._version()}", size=7, color=R.FAINT)
        return page

    def draw_column_table(page, y_top: float, chunk: list[tuple[dict, float]], card_title: str):
        row_step = 0.22
        card_h = min(H - y_top - 0.5, 0.80 + row_step * max(1, len(chunk)))
        tx, ty, tw, _th = page.card(0.5, y_top, W - 1.0, card_h, card_title)
        for label, dx in table_cols:
            page.text(tx + dx, ty, label, size=7.8, color=R.MUTED, weight="bold")
        for i, (c, strength) in enumerate(chunk):
            cid = str(c["id"])
            rec = columns.get((cid, _skey(strength))) or {}
            is_win = _is_winner(winner, cid, strength)
            ry = ty + 0.25 + i * row_step
            if is_win:
                page.box(tx - 0.06, ry - 0.03, tw + 0.12, 0.20, color="#fbf3dc", edge="none", radius=0.02)
            vals = [
                cid + (" *" if is_win else ""),
                R._short(label_of.get(cid, cid), 26),
                f"{float(strength):.2f}",
                _fmt_score(rec.get("face")),
                _fmt_score(rec.get("body")),
                _fmt_score(rec.get("gold")),
                _fmt_score(rec.get("combined")),
                str(int(rec.get("n") or 0)),
            ]
            for (_label, dx), val in zip(table_cols, vals):
                page.text(
                    tx + dx,
                    ry,
                    val,
                    size=7.9,
                    weight="bold" if is_win else "normal",
                    color=R.INK,
                )

    with PdfPages(out_path) as pdf:
        page_num = 1

        # Page 1: Cover header, winner summary, prompt key and first page of column table
        page1 = R.Page(plt, W, H, dpi=150)
        band_h = 1.55
        page1.box(0, 0, W, band_h, color=R.BAND, edge="none", radius=0.0, z=0)
        page1.text(0.5, 0.28, "LORA TEST REPORT", size=8.5, color=R.BAND_MUTED, weight="bold")
        page1.text(0.5, 0.54, R._short(test_name, 44), size=18, color=R.BAND_INK, weight="bold")
        sub_parts = [test_id, str(spec.get("created") or ""), f"scale: {results.get('scale') or '-'}"]
        if not complete:
            sub_parts.append("INCOMPLETE")
        page1.text(0.5, band_h - 0.36, "  .  ".join(x for x in sub_parts if x), size=9.2, color=R.BAND_MUTED)
        page1.text(W / 2, H - 0.3, str(page_num), size=7.5, color=R.FAINT, ha="center")
        page1.text(0.5, H - 0.3, f"RulZ AI Trainer {R._version()}", size=7, color=R.FAINT)

        # Winner card
        x, y, _w, _h = page1.card(0.5, 1.72, W - 1.0, 1.25, "Winner and test overview")
        if winner and winner.get("contestant") is not None and winner.get("strength") is not None:
            w_cid = str(winner["contestant"])
            w_str = float(winner["strength"])
            w_label = label_of.get(w_cid, w_cid)
            win_headline = f"Winner: {w_label} ({w_cid}) at strength {w_str:.2f}  -  combined {_fmt_score(winner.get('combined'))}"
            win_color = R.AQUA
        else:
            win_headline = "Winner: -"
            win_color = R.MUTED
        page1.text(x, y, win_headline, size=10.5, color=win_color, weight="bold")

        render_info = results.get("render") if isinstance(results.get("render"), dict) else {}
        size_txt = (
            f"{render_info.get('width')}x{render_info.get('height')}"
            if render_info.get("width") and render_info.get("height")
            else "-"
        )
        meta_line = (
            f"Prompts: {len(spec.get('prompts') or [])}   Seeds: {len(spec.get('seeds') or [])}   "
            f"Cells: {len(rows) * len(cols)}   Render: {size_txt}, steps {render_info.get('steps', '-')}, "
            f"cfg {render_info.get('cfg', '-')}, time {R.duration(render_info.get('seconds'))}"
        )
        page1.text(x, y + 0.30, meta_line, size=8.2, color=R.INK)

        note_line = str(results.get("scale_note") or ("Test stopped before all cells finished." if not complete else ""))
        if note_line:
            page1.text(x, y + 0.56, R._short(note_line, 105), size=7.8, color=R.ORANGE if not complete else R.MUTED)

        # Prompt key card on page 1
        shown_prompts = prompt_entries[:6] or [(0, "-", "-")]
        pkey_card_h = 0.48 + 0.18 * len(shown_prompts)
        px, py, _pw, _ph = page1.card(0.5, 3.10, W - 1.0, pkey_card_h, "Prompt key")
        for p_i, (p_idx, p_kind, p_txt) in enumerate(shown_prompts):
            page1.text(px, py + p_i * 0.18, f"p{p_idx} ({p_kind}): {p_txt}", size=7.8, color=R.INK)

        # Column table page 1
        n_table_pages = len(table_chunks)
        first_table_title = (
            "Columns (contestant x strength)"
            if n_table_pages == 1
            else f"Columns (contestant x strength, page 1 of {n_table_pages})"
        )
        draw_column_table(page1, 3.10 + pkey_card_h + 0.14, table_chunks[0], first_table_title)
        pdf.savefig(page1.fig, facecolor=R.SURFACE)
        plt.close(page1.fig)

        # Additional column table pages when len(cols) > PDF_TABLE_ROWS_PER_PAGE
        for t_idx, chunk in enumerate(table_chunks[1:], start=2):
            page_num += 1
            t_page = make_page(f"Columns ({t_idx}/{n_table_pages})", page_num)
            draw_column_table(
                t_page,
                0.90,
                chunk,
                f"Columns (contestant x strength, page {t_idx} of {n_table_pages})",
            )
            pdf.savefig(t_page.fig, facecolor=R.SURFACE)
            plt.close(t_page.fig)

        # Paginated grid pages matching export_grid_pages
        n_grid_pages = len(grid_pages)
        for g_idx, grid_im in enumerate(grid_pages, start=1):
            page_num += 1
            hdr_title = "Grid" if n_grid_pages == 1 else f"Grid ({g_idx}/{n_grid_pages})"
            card_title = (
                "Grid (rows = prompt x seed, columns = contestant x strength)"
                if n_grid_pages == 1
                else f"Grid page {g_idx} of {n_grid_pages} (rows = prompt x seed, columns = contestant x strength)"
            )
            g_page = make_page(hdr_title, page_num)
            gx, gy, gw, gh = g_page.card(0.5, 0.9, W - 1.0, H - 1.45, card_title)
            iw, ih = grid_im.size
            scale = min(gw / max(1, iw), gh / max(1, ih))
            dw, dh = iw * scale, ih * scale
            ax = g_page.fig.add_axes(g_page.rect(gx + (gw - dw) / 2, gy + (gh - dh) / 2, dw, dh))
            ax.imshow(grid_im, interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            pdf.savefig(g_page.fig, facecolor=R.SURFACE)
            plt.close(g_page.fig)

        meta = pdf.infodict()
        meta["Title"] = f"LoRA test report: {test_name}"
        meta["Creator"] = f"RulZ AI Trainer {R._version()}"

    return out_path


def export_all(
    test_dir: str | Path,
    cell: int = 256,
    max_cols: int = DEFAULT_MAX_COLS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Path]:
    """Write grid.jpg (or grid_01.jpg...), scores.xlsx and report.pdf into <test_dir>/export/ and return their paths."""
    return {
        "grid": export_grid(test_dir, cell=cell, max_cols=max_cols, max_rows=max_rows),
        "xlsx": export_xlsx(test_dir),
        "pdf": export_pdf(test_dir, cell=cell, max_cols=max_cols, max_rows=max_rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rulztrainer.loratest_export",
        description="Export a LoRA test's grid image, score workbook and PDF report into <test_dir>/export/.",
    )
    ap.add_argument("test_dir", help="the test folder (output/_tests/<test_id>)")
    ap.add_argument(
        "--only",
        choices=("grid", "xlsx", "pdf"),
        default=None,
        help="write only one export file instead of all three",
    )
    ap.add_argument("--cell", type=int, default=256, help="grid cell size in pixels (default: 256)")
    args = ap.parse_args(argv)
    try:
        if args.only == "grid":
            done = {"grid": export_grid(args.test_dir, cell=args.cell)}
        elif args.only == "xlsx":
            done = {"xlsx": export_xlsx(args.test_dir)}
        elif args.only == "pdf":
            done = {"pdf": export_pdf(args.test_dir, cell=args.cell)}
        else:
            done = export_all(args.test_dir, cell=args.cell)
    except Exception as exc:  # noqa: BLE001
        print(f"[loratest_export] {exc}", file=sys.stderr)
        return 1
    for path in done.values():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
