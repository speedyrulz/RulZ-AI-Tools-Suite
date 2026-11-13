"""Run report: one summary picture and a longer PDF of a training run, from what its job folder holds.

    python -m rulztrainer.runreport output/<job>          # both files into output/<job>/report/
    python -m rulztrainer.runreport output/<job> --pdf    # the PDF alone

`run_report: true` writes both when a training ends; the Previous Runs tab writes them for an earlier run whose job
folder is still there. Everything is read from files the run already wrote (config.yaml, status.json, likeness.jsonl,
probes/, training_maps/, region_sweep.jsonl, the dataset folders), so a report can be made long after the run and
nothing here touches a model or a GPU. A part whose files are missing is left out; the report never fails a run.

The summary picture (<job>_report.png) shows the settings, the scores over the run, the probe renders at a handful of
points with the best checkpoint's row marked and the top-scoring render of every row outlined, a race's outcome,
examples of the dataset and the training maps. The PDF (<job>_report.pdf) has the same and more: every changed
setting, every probe step's renders with their scores, the loss, the region sweeps, the dataset and its captions, the
maps at three points of the run, the saved files and the log's warnings.

Needs matplotlib (requirements.txt). The pictures in a report are the run's own: dataset photos and probe renders. It
is written next to the run and sent nowhere.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

INK, MUTED, FAINT, GRID, SURFACE, CARD = "#1a1a19", "#6b6a63", "#9b9a92", "#e4e3dd", "#f4f3ef", "#ffffff"
BAND, BAND_INK, BAND_MUTED = "#1f2937", "#ffffff", "#b6c0cf"
BLUE, ORANGE, AQUA, YELLOW, GOLD, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c98500", "#c9413f"
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
PROBE_RE = re.compile(r"^(?:(?P<racer>.+)_)?step(?P<step>\d{6})_(?P<kind>face|body|check-[^_]+)_(?P<seed>\d+)\.(?:png|jpg|jpeg|webp)$")
# the settings the summary lists, in this order: (label, config key or a function of the config)
SUMMARY_KEYS = (
    ("Network", lambda c: _network(c)), ("Optimizer", "optimizer"), ("Learning rate", "learning_rate"),
    ("Steps", "steps"), ("Batch size", "batch_size"), ("Resolution", lambda c: _resolution(c)),
    ("Trigger word", "trigger_word"), ("Seed", "seed"), ("Noise levels", "timestep_sampling"),
    ("Caption dropout", "caption_dropout"), ("Weight average (EMA)", "lora_ema"),
    ("Probes", lambda c: f"every {c.get('probe_every')} steps, {c.get('probe_seeds')} seeds" if c.get("probe_every") else "off"),
)
SWITCHES = (  # settings worth naming on the summary when they are on
    ("race", "configuration race"), ("region_watch", "region watch"), ("region_steer", "region steering"),
    ("region_class_weights", "region class weights"), ("holdout_images", "held-out photos"),
    ("training_maps", "training maps"), ("likeness_curriculum", "face / body curriculum"),
    ("likeness_lr_control", "likeness lr control"), ("likeness_track_best", "best checkpoint kept"),
    ("region_weights", "region-weighted loss"), ("loss_watch", "loss watch"), ("per_image_lr", "per-image lr"),
    ("isolation", "LoRA Isolation"), ("finetune", "full fine-tune"), ("dora", "DoRA"),
)


class ReportError(RuntimeError):
    """The report cannot be made at all (no such folder, no matplotlib)."""


@dataclass
class Run:
    job_dir: Path
    root: Path
    cfg: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)
    probes: list = field(default_factory=list)  # likeness.jsonl's records, oldest first
    images: dict = field(default_factory=dict)  # (racer, step, kind) -> {seed: Path}
    sweeps: list = field(default_factory=list)
    maps: dict = field(default_factory=dict)  # {"pictures": [...], "rows": [...], "dir": Path}
    dataset: list = field(default_factory=list)  # image paths
    lr: list = field(default_factory=list)  # [(step, lr)] from the log
    notes: list = field(default_factory=list)  # the log's warnings and errors
    started: str = ""

    @property
    def job(self) -> str:
        return str(self.cfg.get("job_name") or self.job_dir.name)


# ---------------------------------------------------------------- reading a run


def _json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _jsonl(path: Path) -> list:
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _resolve(root: Path, value) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else root / p


def dataset_images(root: Path, cfg: dict, limit: int = 4000) -> list[Path]:
    out = []
    for spec in cfg.get("datasets") or []:
        folder = _resolve(root, spec.get("path") if isinstance(spec, dict) else spec)
        try:
            out += sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES)[:limit]
        except OSError:
            pass
    return out


def probe_images(folder: Path) -> dict:
    out: dict = {}
    try:
        names = sorted(p.name for p in folder.iterdir())
    except OSError:
        return out
    for name in names:
        m = PROBE_RE.match(name)
        if m:
            out.setdefault((m["racer"] or "", int(m["step"]), m["kind"]), {})[int(m["seed"])] = folder / name
    return out


def load(job_dir, root=None) -> Run:
    """Everything a report shows, read from a job folder. root: the folder the config's relative paths start from
    (the install; by default two levels above the job folder, where output/<job> sits)."""
    job_dir = Path(job_dir)
    if not job_dir.is_dir():
        raise ReportError(f"No job folder at {job_dir}")
    root = Path(root) if root else job_dir.resolve().parent.parent
    run = Run(job_dir, root)
    try:
        import yaml

        run.cfg = yaml.safe_load((job_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001  (no config, or no PyYAML: the report goes on without the settings)
        run.cfg = {}
    run.status = _json(job_dir / "status.json", {})
    run.probes = _jsonl(job_dir / "likeness.jsonl") or list(run.status.get("probes") or [])
    run.images = probe_images(job_dir / "probes")
    run.sweeps = [r for r in _jsonl(job_dir / "region_sweep.jsonl") if not r.get("first")]
    maps_dir = job_dir / "training_maps"
    pictures = (_json(maps_dir / "probes.json", {}) or {}).get("probes") or []
    rows = _jsonl(maps_dir / "maps.jsonl")
    if pictures and rows:
        run.maps = {"pictures": pictures, "rows": rows, "dir": maps_dir}
    run.dataset = dataset_images(root, run.cfg)
    try:
        text = (job_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    seen = set()
    for line in text.splitlines():
        m = re.search(r"step (\d+)/\d+ loss [\d.eE+-]+ avg [\d.eE+-]+ lr ([\d.eE+-]+)", line)
        if m:
            run.lr.append((int(m[1]), float(m[2])))
            continue
        m = re.match(r"^\S+ (WARNING|ERROR) (.*)$", line)
        if m and m[2][:60] not in seen:
            seen.add(m[2][:60])
            run.notes.append((m[1], m[2].strip()))
    rid = str(run.status.get("run_id") or "")
    m = re.search(r"(\d{8})-(\d{6})$", rid)
    if m:
        run.started = f"{m[1][:4]}-{m[1][4:6]}-{m[1][6:]} {m[2][:2]}:{m[2][2:4]}"
    else:
        try:
            run.started = time.strftime("%Y-%m-%d %H:%M", time.localtime((job_dir / "config.yaml").stat().st_mtime))
        except OSError:
            run.started = ""
    return run


# ---------------------------------------------------------------- what the run did, in words and numbers


def _num(v, digits: int = 1) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _short(text, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:max(1, n - 1)].rstrip() + "…"


def _value(v) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_value(x) for x in v) if v else "none"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_value(x)}" for k, x in v.items())
    return "(empty)" if v == "" else str(v)


def _network(c: dict) -> str:
    kind = str(c.get("network_type") or "lora")
    if c.get("finetune"):
        return "full fine-tune"
    if kind == "oft":
        return f"OFT, block {c.get('oft_block_size')}"
    if kind == "lokr":
        return f"LoKr, factor {c.get('lokr_factor')}"
    return f"{'DoRA' if c.get('dora') else 'LoRA'}, rank {c.get('lora_rank')} (alpha {_value(c.get('lora_alpha'))})"


def _resolution(c: dict) -> str:
    res = c.get("resolutions") or []
    return ", ".join(f"{_value(r)} MP" for r in res) if res else f"{c.get('resolution')} px"


def model_name(run: Run) -> str:
    c = run.cfg
    for key in ("transformer_file",):
        if c.get(key):
            return Path(str(c[key])).stem
    return str(c.get("model_path") or c.get("model_arch") or "model")


def duration(seconds) -> str:
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "n/a"
    h, m = divmod(s // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s % 60:02d} s"


def best(run: Run) -> tuple:
    """(score, step) of the best checkpoint as the run recorded it."""
    if run.probes:
        last = run.probes[-1]
        return last.get("best"), last.get("best_step")
    like = run.status.get("likeness") or {}
    return like.get("best"), like.get("best_step")


def steps_shown(run: Run, n: int, racer: str = "") -> list[int]:
    """Up to n probe steps to show: the first, the last, the best checkpoint's, and the rest spread evenly."""
    have = sorted({s for (r, s, _k) in run.images if r == racer})
    if len(have) <= n:
        return have
    keep = {have[0], have[-1]}
    b = best(run)[1]
    if b in have:
        keep.add(b)
    spread = [have[round(i * (len(have) - 1) / (n - 1))] for i in range(n)]
    for s in spread + have:
        if len(keep) >= n:
            break
        keep.add(s)
    return sorted(keep)


def record_at(run: Run, step: int) -> dict:
    for r in run.probes:
        if r.get("step") == step:
            return r
    return {}


def seed_scores(rec: dict, kind: str) -> list:
    if kind == "face":
        return list(rec.get("gold_norm_seeds") or rec.get("face_norm_seeds") or rec.get("face_seeds") or [])
    if kind == "body":
        return list(rec.get("body_norm_seeds") or rec.get("body_seeds") or [])
    return []


def racers(run: Run) -> list[str]:
    return sorted({r for (r, _s, _k) in run.images if r})


def verdict(run: Run) -> list[tuple[str, str]]:
    """Short plain findings for the summary: (kind, text), kind good / note / warn."""
    out = []
    b, bs = best(run)
    total = run.status.get("total") or run.cfg.get("steps")
    if b is not None and bs is not None:
        out.append(("good", f"Best checkpoint at step {bs} of {total}, smoothed score {_num(b)}."))
    lost = [(r.get("step"), r.get("body_missing"), r.get("body_probes")) for r in run.probes if r.get("body_missing")]
    if lost:
        first = lost[0]
        worst = max(lost, key=lambda t: (t[1] or 0) / max(1, t[2] or 1))
        out.append(("warn", f"Body probe lost its framing: {first[1]} of {first[2]} renders without a body from step "
                            f"{first[0]}, {worst[1]} of {worst[2]} at step {worst[0]}."))
    copies = [(r.get("copy"), r.get("step")) for r in run.probes if r.get("copy") is not None]
    limit = float(run.cfg.get("copy_max") or 0.5)
    if copies:
        peak = max(copies)
        base = copies[0][0]
        if peak[0] > limit and peak[0] > (base or 0) + 0.05:
            out.append(("warn", f"Copy score rose to {peak[0]:.2f} at step {peak[1]} (start {base:.2f}, limit "
                                f"{limit:.2f}): renders may copy a training photo."))
    dup = (run.status.get("near_duplicates") or {}).get("count")
    if dup:
        out.append(("note", f"{dup} near-duplicate pair(s) of training photos."))
    if run.sweeps:
        peak = max(run.sweeps, key=lambda s: (s.get("held") or {}).get("subject") or -1)
        if (peak.get("held") or {}).get("subject") is not None:
            out.append(("note", f"Held-out photos peaked at step {peak['step']} "
                                f"({100 * peak['held']['subject']:.1f}% of the subject learned)."))
    race = run.status.get("race") or {}
    if race.get("champion") or race.get("winner"):
        out.append(("good", f"Race won by {race.get('champion') or race.get('winner')}."))
    if run.status.get("error"):
        out.append(("warn", f"The run ended with an error: {_short(run.status['error'], 140)}"))
    elif not run.status.get("done"):
        out.append(("note", "The run had not finished when this report was made."))
    return out


def changed_settings(cfg: dict) -> list[tuple[str, str]]:
    """The settings that differ from the trainer's defaults, as (key, value) text."""
    try:
        import dataclasses

        from .config import TrainConfig

        defaults = dataclasses.asdict(TrainConfig())
    except Exception:  # noqa: BLE001
        defaults = {}
    out = []
    for key, value in cfg.items():
        if key in defaults and defaults[key] == value:
            continue
        out.append((key, _value(value)))
    return out


# ---------------------------------------------------------------- drawing


def _mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ReportError("The run report needs matplotlib: pip install matplotlib") from e
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
    return plt


class Page:
    """A figure laid out in inches from its top left corner."""

    def __init__(self, plt, width: float, height: float, dpi: int = 150):
        self.plt, self.w, self.h = plt, width, height
        self.fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor=SURFACE)
        self.bg = self.fig.add_axes([0, 0, 1, 1])
        self.bg.set_xlim(0, width)
        self.bg.set_ylim(height, 0)
        self.bg.axis("off")
        self._thumbs: dict = {}

    def rect(self, x, y, w, h):
        return [x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h]

    def box(self, x, y, w, h, color=CARD, edge=GRID, lw=0.8, radius=0.08, z=1):
        from matplotlib.patches import FancyBboxPatch

        self.bg.add_patch(FancyBboxPatch((x + radius, y + radius), w - 2 * radius, h - 2 * radius,
                                         boxstyle=f"round,pad={radius}", fc=color, ec=edge, lw=lw, zorder=z))

    def text(self, x, y, s, size=9, color=INK, weight="normal", ha="left", va="top", **kw):
        return self.bg.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=5, **kw)

    def card(self, x, y, w, h, title=""):
        self.box(x, y, w, h)
        if title:
            self.text(x + 0.18, y + 0.14, title.upper(), size=7.5, color=MUTED, weight="bold")
        return x + 0.18, y + (0.5 if title else 0.18), w - 0.36, h - (0.68 if title else 0.36)

    def axes(self, x, y, w, h):
        ax = self.fig.add_axes(self.rect(x, y, w, h))
        ax.set_facecolor(CARD)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, labelsize=7.5)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        return ax

    def picture(self, x, y, w, h, path, edge=None, lw=2.2, max_px=420):
        """An image file fitted into the box (centred, its shape kept). Returns the box it drew in (x, y, w, h), None
        when the file cannot be read."""
        from PIL import Image

        try:
            key = (str(path), max_px)
            if key not in self._thumbs:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((max_px, max_px))
                    self._thumbs[key] = im.copy()
            im = self._thumbs[key]
        except Exception:  # noqa: BLE001
            return None
        iw, ih = im.size
        scale = min(w / iw, h / ih)
        dw, dh = iw * scale, ih * scale
        ax = self.fig.add_axes(self.rect(x + (w - dw) / 2, y + (h - dh) / 2, dw, dh))
        ax.imshow(im, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(bool(edge))
            if edge:
                s.set_edgecolor(edge)
                s.set_linewidth(lw)
        return (x + (w - dw) / 2, y + (h - dh) / 2, dw, dh)

    def heat(self, x, y, w, h, grid: dict, cmap="magma"):
        try:
            gw, gh, values = int(grid["w"]), int(grid["h"]), list(grid["v"])
            rows = [values[i * gw:(i + 1) * gw] for i in range(gh)]
        except Exception:  # noqa: BLE001
            return None
        scale = min(w / gw, h / gh)
        dw, dh = gw * scale, gh * scale
        ax = self.fig.add_axes(self.rect(x + (w - dw) / 2, y + (h - dh) / 2, dw, dh))
        ax.imshow(rows, cmap=cmap, interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return (x + (w - dw) / 2, y + (h - dh) / 2, dw, dh)

    def rows(self, x, y, pairs, key_w, size=8.5, pitch=0.235, value_chars=60):
        """Key / value lines; a long value runs on to a second line. Returns the y below the last one."""
        for k, v in pairs:
            lines = _wrap(v, value_chars)[:2]
            if len(_wrap(v, value_chars)) > 2:
                lines[1] = _short(lines[1] + " …", value_chars)
            self.text(x, y, k, size=size, color=MUTED)
            self.text(x + key_w, y, "\n".join(lines), size=size, linespacing=1.25)
            y += pitch * (1 + 0.78 * (len(lines) - 1))
        return y


def likeness_chart(ax, run: Run, legend=True):
    pr = [r for r in run.probes if r.get("step") is not None]
    if not pr:
        ax.text(0.5, 0.5, "no likeness probes", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    steps = [r["step"] for r in pr]
    gold = any(r.get("gold_select") for r in pr)
    gold_name = "AdaFace" if any(r.get("gold_name") == "adaface" for r in pr) else "FaceNet"
    series = [("Face" + (f" ({gold_name})" if gold else ""), "gold_norm_ema" if gold else "face_norm_ema", BLUE, "o"),
              ("Body", "body_norm_ema", ORANGE, "s"), ("Combined", "combined_ema", AQUA, "D")]
    for label, key, color, marker in series:
        ys = [r.get(key) if r.get(key) is not None else (r.get(key.replace("_norm", "")) if "_norm" in key else None)
              for r in pr]
        pts = [(s, v) for s, v in zip(steps, ys) if v is not None]
        if pts:
            ax.plot(*zip(*pts), color=color, lw=2 if label == "Combined" else 1.4, marker=marker, ms=3.5, label=label)
    b, bs = best(run)
    if b is not None and bs is not None:
        ax.axvline(bs, color=GOLD, lw=1, ls=(0, (4, 3)))
        ax.plot([bs], [b], marker="*", ms=13, color=GOLD, mec=CARD, mew=0.8, zorder=6)
        ax.annotate(f"best {_num(b)} @ {bs}", (bs, b), xytext=(6, 8), textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("training step", fontsize=7.5)
    ax.set_ylabel("smoothed score", fontsize=7.5)
    ax.set_ylim(bottom=0)
    if legend:
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3, handlelength=1.6)


def loss_chart(ax, run: Run):
    losses = [v for v in (run.status.get("losses") or []) if isinstance(v, (int, float)) and math.isfinite(v)]
    if not losses:
        ax.text(0.5, 0.5, "no loss history", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    total = int(run.status.get("step") or len(losses))
    xs = [total * (i + 1) / len(losses) for i in range(len(losses))]
    ax.plot(xs, losses, color=GRID, lw=0.6)
    win = max(5, len(losses) // 40)
    smooth, acc = [], 0.0
    for i, v in enumerate(losses):
        acc += v - (losses[i - win] if i >= win else 0.0)
        smooth.append(acc / min(i + 1, win))
    ax.plot(xs, smooth, color=BLUE, lw=1.6)
    lo, hi = min(smooth[win:] or smooth), max(smooth[win:] or smooth)
    pad = (hi - lo) * 0.6 or 0.05
    ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.set_xlabel("training step", fontsize=7.5)
    ax.set_ylabel("loss (smoothed)", fontsize=7.5)


def sweep_chart(ax, run: Run):
    steps = [s["step"] for s in run.sweeps]
    for label, key, color, marker in (("Training photos", "train", BLUE, "o"), ("Held-out photos", "held", ORANGE, "s")):
        ys = [(s.get(key) or {}).get("subject") for s in run.sweeps]
        pts = [(a, 100 * b) for a, b in zip(steps, ys) if b is not None]
        if pts:
            ax.plot(*zip(*pts), color=color, lw=1.6, marker=marker, ms=3.5, label=label)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("training step", fontsize=7.5)
    ax.set_ylabel("% of the subject's error removed", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")


def probe_grid(page: Page, x, y, run: Run, steps, kind: str, seeds: int, cell_h: float, racer: str = "",
               gap: float = 0.06, scores: bool = True):
    """Rows of probe renders, one row a step. The best checkpoint's row sits on a gold band and the top-scoring
    render of each row is outlined. Returns (width, height) used."""
    aspect = 1.0 if kind == "face" else 2 / 3
    cell_w = cell_h * aspect
    pitch = cell_h + (0.3 if scores else 0.08)
    b_step = best(run)[1]
    for r, step in enumerate(steps):
        files = run.images.get((racer, step, kind)) or {}
        rec = record_at(run, step)
        sc = seed_scores(rec, kind)
        valid = [(v, i) for i, v in enumerate(sc[:seeds]) if v is not None]
        top = max(valid)[1] if len(valid) > 1 else None
        for k in range(seeds):
            cx, cy = x + k * (cell_w + gap), y + r * pitch
            path = files.get(k)
            edge = AQUA if (top == k) else None
            if path is None or page.picture(cx, cy, cell_w, cell_h, path, edge=edge) is None:
                page.box(cx, cy, cell_w, cell_h, color=SURFACE, edge=GRID, radius=0.02)
            if scores and k < len(sc):
                page.text(cx + cell_w / 2, cy + cell_h + 0.03, "no body" if (sc[k] is None and kind == "body") else _num(sc[k], 0),
                          size=6.5, color=AQUA if top == k else MUTED, ha="center", weight="bold" if top == k else "normal")
        if step == b_step:
            page.box(x - 0.07, y + r * pitch - 0.07, seeds * (cell_w + gap) - gap + 0.14,
                     cell_h + (0.3 if scores else 0.14), color="none",
                     edge=GOLD, lw=2.2, radius=0.03, z=4)
    return seeds * (cell_w + gap) - gap, len(steps) * pitch


def step_labels(page: Page, x, y, run: Run, steps, cell_h: float, pitch: float, width: float):
    b_step = best(run)[1]
    for r, step in enumerate(steps):
        rec = record_at(run, step)
        cy = y + r * pitch
        if step == b_step:
            page.box(x - 0.08, cy - 0.07, width, cell_h + 0.3, color="#fbf3dc", edge="none", radius=0.03)
            page.text(x, cy + cell_h - 0.24, "★ BEST CHECKPOINT", size=6.5, color=GOLD, weight="bold")
        page.text(x, cy + 0.02, f"Step {step}", size=10, weight="bold")
        face = rec.get("gold_norm") if rec.get("gold_select") else rec.get("face_norm", rec.get("face"))
        lines = [f"face {_num(face, 0)}   body {_num(rec.get('body_norm', rec.get('body')), 0)}",
                 f"score {_num(rec.get('combined'), 0)}   smoothed {_num(rec.get('combined_ema'), 0)}"]
        if rec.get("copy") is not None:
            lines.append(f"copy {rec['copy']:.2f}" + (f"   no body {rec['body_missing']}/{rec.get('body_probes')}"
                                                     if rec.get("body_missing") else ""))
        for i, line in enumerate(lines):
            page.text(x, cy + 0.27 + i * 0.17, line, size=7.2, color=MUTED)


def header(page: Page, run: Run, title_size=24, band_h=1.9, tiles=True):
    page.box(0, 0, page.w, band_h, color=BAND, edge="none", radius=0.0, z=0)
    page.text(0.6, 0.42, "TRAINING RUN REPORT", size=8.5, color=BAND_MUTED, weight="bold")
    page.text(0.6, 0.66, _short(run.job, 46), size=title_size, color=BAND_INK, weight="bold")
    sub = "  ·  ".join(x for x in (_short(model_name(run), 44), _network(run.cfg), run.started) if x)
    page.text(0.6, band_h - 0.5, sub, size=10, color=BAND_MUTED)
    if not tiles:
        return
    b, bs = best(run)
    st = run.status
    stats = [("STEPS", f"{st.get('step', '?')} / {st.get('total', run.cfg.get('steps', '?'))}"),
             ("TIME", duration(st.get("elapsed"))), ("BEST SCORE", f"{_num(b)}" if b is not None else "n/a"),
             ("AT STEP", str(bs) if bs is not None else "n/a")]
    tw = 1.75
    for i, (k, v) in enumerate(stats):
        tx = page.w - 0.6 - (len(stats) - i) * (tw + 0.12) + 0.12
        page.box(tx, 0.45, tw, 1.0, color="#2b3848", edge="none", radius=0.06, z=1)
        page.text(tx + 0.14, 0.56, k, size=7, color=BAND_MUTED, weight="bold")
        page.text(tx + 0.14, 0.82, v, size=15, color=BAND_INK, weight="bold")


def settings_rows(run: Run) -> list[tuple[str, str]]:
    c = run.cfg
    rows = [("Model", model_name(run))]
    for label, key in SUMMARY_KEYS:
        try:
            value = key(c) if callable(key) else c.get(key)
        except Exception:  # noqa: BLE001
            value = None
        if value not in (None, ""):
            rows.append((label, _value(value) if not isinstance(value, str) else value))
    n = len(run.dataset)
    names = ", ".join(Path(str(d.get("path") if isinstance(d, dict) else d)).name for d in c.get("datasets") or [])
    rows.insert(1, ("Dataset", f"{names} ({n} images)" if n else names or "n/a"))
    on = [label for key, label in SWITCHES if c.get(key)]
    if on:
        rows.append(("Also on", ", ".join(on)))
    return rows


def map_views(run: Run, row: dict, pid: str) -> list[tuple[str, dict]]:
    """The map grids of one picture at one capture, at the middle noise level: [(label, grid)]."""
    levels = (row.get("maps") or {}).get(str(pid)) or {}
    if not levels:
        return []
    keys = sorted(levels, key=lambda s: abs(float(s) - 0.5))
    lv = levels[keys[0]]
    names = {"loss": "where it is wrong", "effect": "where the LoRA acts", "attn": "trigger word", "effect_off": "without trigger"}
    return [(names[k], lv[k]) for k in ("loss", "effect", "attn", "effect_off") if isinstance(lv.get(k), dict)]


# ---------------------------------------------------------------- the summary picture


def summary(run: Run, path: Path, dpi: int = 150) -> Path:
    plt = _mpl()
    W, H = 16.0, 24.0
    page = Page(plt, W, H, dpi)
    header(page, run)
    # settings | likeness
    x, y, w, h = page.card(0.6, 2.2, 7.3, 4.9, "Settings")
    page.rows(x, y, settings_rows(run), key_w=2.05, value_chars=52, pitch=0.262)
    x, y, w, h = page.card(8.1, 2.2, 7.3, 4.9, "Likeness over the run")
    likeness_chart(page.axes(x + 0.45, y, w - 0.5, h - 0.4), run)
    # probes
    face_steps = steps_shown(run, 6)
    top = 7.4
    card_h = 9.5
    x, y, w, h = page.card(0.6, top, W - 1.2, card_h, "Probe renders through the run: the same seeds at each step")
    kinds = [k for k in ("face", "body") if any(kk == k for (_r, _s, kk) in run.images)]
    race_names = racers(run)
    racer = ""
    if race_names and not any(r == "" for (r, _s, _k) in run.images):
        champ = (run.status.get("race") or {}).get("champion")
        racer = champ if champ in race_names else race_names[0]
        face_steps = steps_shown(run, 6, racer)
    if face_steps and kinds:
        cell_h, pitch = 1.14, 1.44
        step_labels(page, x + 0.08, y + 0.12, run, face_steps, cell_h, pitch, 2.35)
        gx = x + 2.5
        for kind, seeds in (("face", 5), ("body", 6)):
            if kind not in kinds:
                continue
            gw, _gh = probe_grid(page, gx, y + 0.12, run, face_steps, kind, seeds, cell_h, racer)
            page.text(gx, y - 0.12, ("Face probe" if kind == "face" else "Body probe") + (f" ({racer})" if racer else ""),
                      size=8, color=MUTED, weight="bold")
            gx += gw + 0.45
        page.text(x + w, top + 0.16, "gold frame: the best checkpoint   ·   green outline: the top-scoring render of the row",
                  size=7.2, color=MUTED, ha="right")
    else:
        page.text(x + w / 2, y + h / 2, "This run saved no probe renders", color=MUTED, ha="center", va="center", size=11)
    # dataset | maps or sweeps
    y0 = top + card_h + 0.3
    x, y, w, h = page.card(0.6, y0, 7.3, 3.5, f"Dataset examples ({len(run.dataset)} images)")
    if run.dataset:
        n = min(12, len(run.dataset))
        picks = [run.dataset[round(i * (len(run.dataset) - 1) / max(1, n - 1))] for i in range(n)]
        cols = 6
        cw, ch = (w - (cols - 1) * 0.07) / cols, (h - 0.07) / 2
        for i, pth in enumerate(picks):
            page.picture(x + (i % cols) * (cw + 0.07), y + (i // cols) * (ch + 0.07), cw, ch, pth, max_px=300)
    else:
        page.text(x + w / 2, y + h / 2, "The dataset folder is not on this computer", color=MUTED, ha="center", va="center")
    if run.maps:
        x, y, w, h = page.card(8.1, y0, 7.3, 3.5, f"Training maps at step {run.maps['rows'][-1].get('step')}")
        pics = run.maps["pictures"][:2]
        last = run.maps["rows"][-1]
        slot_w = (w - 0.2 * (len(pics) - 1)) / max(1, len(pics))
        for i, pic in enumerate(pics):
            views = map_views(run, last, pic.get("id"))[:2]
            n = 1 + len(views)
            cw = (slot_w - 0.06 * (n - 1)) / n
            sx = x + i * (slot_w + 0.2)
            drawn = page.picture(sx, y, cw, h - 0.25, run.maps["dir"] / str(pic.get("image")), max_px=320)
            if drawn:
                page.text(sx + cw / 2, drawn[1] + drawn[3] + 0.06, "training photo", size=6.8, color=MUTED, ha="center")
            for j, (label, grid) in enumerate(views):
                drawn = page.heat(sx + (j + 1) * (cw + 0.06), y, cw, h - 0.25, grid)
                if drawn:
                    page.text(sx + (j + 1) * (cw + 0.06) + cw / 2, drawn[1] + drawn[3] + 0.06, label, size=6.8,
                              color=MUTED, ha="center")
    elif run.sweeps:
        x, y, w, h = page.card(8.1, y0, 7.3, 3.5, "What was learned: training photos against held-out photos")
        sweep_chart(page.axes(x + 0.45, y, w - 0.5, h - 0.4), run)
    else:
        x, y, w, h = page.card(8.1, y0, 7.3, 3.5, "Loss")
        loss_chart(page.axes(x + 0.5, y, w - 0.55, h - 0.4), run)
    # loss (or sweeps) | result
    y1 = y0 + 3.8
    if run.maps and run.sweeps:
        x, y, w, h = page.card(0.6, y1, 7.3, 2.6, "What was learned: training photos against held-out photos")
        sweep_chart(page.axes(x + 0.45, y, w - 0.5, h - 0.38), run)
    else:
        x, y, w, h = page.card(0.6, y1, 7.3, 2.6, "Loss")
        loss_chart(page.axes(x + 0.5, y, w - 0.55, h - 0.38), run)
    x, y, w, h = page.card(8.1, y1, 7.3, 2.6, "Result")
    yy = y
    for kind, text in verdict(run)[:5]:
        mark, color = {"good": ("●", AQUA), "warn": ("▲", ORANGE), "note": ("●", FAINT)}[kind]
        page.text(x, yy, mark, size=8, color=color)
        wrapped = _wrap(text, 86)
        page.text(x + 0.2, yy, "\n".join(wrapped), size=8.2, linespacing=1.35)
        yy += 0.19 * len(wrapped) + 0.1
    saved = [Path(str(s)).name for s in run.status.get("saved") or []]
    if saved and yy < y + h - 0.3:
        page.text(x, y + h - 0.02, "Files: " + _short(", ".join(saved), 100), size=7.2, color=MUTED, va="bottom")
    page.text(0.6, H - 0.33, f"RulZ Trainer {_version()}   ·   made {time.strftime('%Y-%m-%d %H:%M')}   ·   {run.job_dir}",
              size=7, color=FAINT)
    path.parent.mkdir(parents=True, exist_ok=True)
    page.fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(page.fig)
    return path


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(" ".join(str(text).split()), width) or [""]


def _version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------- the PDF

A4 = (8.27, 11.69)


def _pdf_page(plt, run: Run, title: str, number: int):
    page = Page(plt, *A4, dpi=150)
    page.box(0, 0, A4[0], 0.62, color=BAND, edge="none", radius=0.0, z=0)
    page.text(0.5, 0.2, _short(run.job, 40), size=10, color=BAND_INK, weight="bold")
    page.text(A4[0] - 0.5, 0.22, title, size=9, color=BAND_MUTED, ha="right")
    page.text(A4[0] / 2, A4[1] - 0.3, f"{number}", size=7.5, color=FAINT, ha="center")
    page.text(0.5, A4[1] - 0.3, f"RulZ Trainer {_version()}", size=7, color=FAINT)
    return page


def pdf(run: Run, path: Path) -> Path:
    plt = _mpl()
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    n = [0]

    def new(title):
        n[0] += 1
        return _pdf_page(plt, run, title, n[0])

    def done(out, page):
        out.savefig(page.fig, facecolor=SURFACE)
        plt.close(page.fig)

    with PdfPages(path) as out:
        W = A4[0]
        # 1: the cover
        n[0] += 1
        page = Page(plt, *A4, dpi=150)
        header(page, run, title_size=20, band_h=2.05, tiles=False)
        b, bs = best(run)
        st = run.status
        tiles = [("Steps", f"{st.get('step', '?')} / {st.get('total', run.cfg.get('steps', '?'))}"),
                 ("Time", duration(st.get("elapsed"))), ("Best score", _num(b)), ("At step", str(bs) if bs is not None else "n/a"),
                 ("Final loss", _num(st.get("avg_loss"), 4)), ("Peak VRAM", ", ".join(f"{v} GB" for v in (st.get("peak_mem") or {}).values()) or "n/a")]
        tw = (W - 1.0 - 5 * 0.12) / 6
        for i, (k, v) in enumerate(tiles):
            tx = 0.5 + i * (tw + 0.12)
            page.box(tx, 2.3, tw, 0.82, radius=0.05)
            page.text(tx + 0.1, 2.38, k.upper(), size=6, color=MUTED, weight="bold")
            page.text(tx + 0.1, 2.6, _short(v, 16), size=10.5, weight="bold")
        x, y, w, h = page.card(0.5, 3.35, W - 1.0, 3.0, "Result")
        yy = y
        for kind, text in verdict(run):
            mark, color = {"good": ("●", AQUA), "warn": ("▲", ORANGE), "note": ("●", FAINT)}[kind]
            page.text(x, yy, mark, size=8, color=color)
            wrapped = _wrap(text, 104)
            page.text(x + 0.2, yy, "\n".join(wrapped), size=8.5, linespacing=1.35)
            yy += 0.19 * len(wrapped) + 0.1
        x, y, w, h = page.card(0.5, 6.55, W - 1.0, 4.4, "Likeness over the run")
        likeness_chart(page.axes(x + 0.45, y, w - 0.5, h - 0.4), run)
        done(out, page)

        # 2: settings
        rows = settings_rows(run)
        page = new("Settings")
        x, y, w, h = page.card(0.5, 0.9, W - 1.0, 0.75 + 0.235 * len(rows), "The run in short")
        page.rows(x, y, rows, key_w=1.8, value_chars=84)
        changed = changed_settings(run.cfg)
        top = 1.9 + 0.235 * len(rows) + 0.75
        per_page = int((A4[1] - top - 1.0) / 0.2)
        chunk, rest = changed[:per_page], changed[per_page:]
        x, y, w, h = page.card(0.5, top, W - 1.0, 0.75 + 0.2 * max(1, len(chunk)),
                               f"Every setting that differs from the defaults ({len(changed)})")
        page.rows(x, y, chunk or [("(none)", "")], key_w=2.6, size=7.6, pitch=0.2, value_chars=82)
        done(out, page)
        while rest:
            per_page = int((A4[1] - 2.2) / 0.2)
            chunk, rest = rest[:per_page], rest[per_page:]
            page = new("Settings, continued")
            x, y, w, h = page.card(0.5, 0.9, W - 1.0, 0.75 + 0.2 * len(chunk), "Settings that differ from the defaults")
            page.rows(x, y, chunk, key_w=2.6, size=7.6, pitch=0.2, value_chars=82)
            done(out, page)

        # 3: the scores, probe by probe
        if run.probes:
            page = new("Scores")
            x, y, w, h = page.card(0.5, 0.9, W - 1.0, 2.9, "Loss")
            loss_chart(page.axes(x + 0.5, y, w - 0.55, h - 0.4), run)
            cols = [("step", 0.0), ("face", 0.75), ("body", 1.45), ("score", 2.15), ("smoothed", 2.9), ("best so far", 3.8),
                    ("copy", 4.85), ("no body", 5.5)]
            rows_n = len(run.probes)
            x, y, w, h = page.card(0.5, 4.0, W - 1.0, 0.85 + 0.21 * rows_n, "Every probe")
            for label, dx in cols:
                page.text(x + dx, y, label, size=7.5, color=MUTED, weight="bold")
            for i, r in enumerate(run.probes[:30]):
                yy = y + 0.25 + i * 0.21
                face = r.get("gold_norm") if r.get("gold_select") else r.get("face_norm", r.get("face"))
                is_best = r.get("step") == bs
                if is_best:
                    page.box(x - 0.06, yy - 0.035, w + 0.12, 0.2, color="#fbf3dc", edge="none", radius=0.02)
                vals = [str(r.get("step")), _num(face), _num(r.get("body_norm", r.get("body"))), _num(r.get("combined")),
                        _num(r.get("combined_ema")), f"{_num(r.get('best'))} @ {r.get('best_step')}",
                        "" if r.get("copy") is None else f"{r['copy']:.2f}",
                        f"{r.get('body_missing')}/{r.get('body_probes')}" if r.get("body_probes") else ""]
                for (label, dx), v in zip(cols, vals):
                    page.text(x + dx, yy, v, size=7.8, weight="bold" if is_best else "normal")
            done(out, page)

        # 4: sweeps
        if run.sweeps:
            page = new("What was learned")
            x, y, w, h = page.card(0.5, 0.9, W - 1.0, 3.6, "The subject's error removed: training photos against held-out photos")
            sweep_chart(page.axes(x + 0.5, y, w - 0.55, h - 0.4), run)
            parts = ("face", "hair", "skin", "clothes", "background", "subject")
            x, y, w, h = page.card(0.5, 4.7, W - 1.0, 0.9 + 0.21 * len(run.sweeps), "Region sweeps: % of the starting error removed (training / held-out)")
            page.text(x, y, "step", size=7.5, color=MUTED, weight="bold")
            for j, part in enumerate(parts):
                page.text(x + 0.7 + j * 1.08, y, part, size=7.5, color=MUTED, weight="bold")
            for i, s in enumerate(run.sweeps[:28]):
                yy = y + 0.25 + i * 0.21
                page.text(x, yy, str(s.get("step")), size=7.8)
                for j, part in enumerate(parts):
                    a, b2 = (s.get("train") or {}).get(part), (s.get("held") or {}).get(part)
                    page.text(x + 0.7 + j * 1.08, yy, f"{'' if a is None else f'{100 * a:.0f}'}"
                              + ("" if b2 is None else f" / {100 * b2:.0f}"), size=7.8)
            done(out, page)

        # 5: a race
        race = run.status.get("race") or {}
        if race.get("names"):
            page = new("The race")
            names = list(race["names"])
            x, y, w, h = page.card(0.5, 0.9, W - 1.0, 0.9 + 0.24 * len(names), "Configurations")
            for label, dx in (("name", 0.0), ("best score", 2.6), ("left the race", 3.8)):
                page.text(x + dx, y, label, size=7.5, color=MUTED, weight="bold")
            champ = race.get("champion")
            left, dropped, bests = race.get("left") or {}, race.get("dropped") or {}, race.get("best") or {}
            for i, name in enumerate(names):
                yy = y + 0.27 + i * 0.24
                if name == champ:
                    page.box(x - 0.06, yy - 0.04, w + 0.12, 0.22, color="#fbf3dc", edge="none", radius=0.02)
                why = "winner" if name == champ else (f"round {dropped[name]}" + (f" ({left[name]})" if left.get(name) else "")
                                                     if name in dropped else "stayed in")
                for dx, v in ((0.0, _short(name, 34)), (2.6, _num(bests.get(name))), (3.8, why)):
                    page.text(x + dx, yy, v, size=8, weight="bold" if name == champ else "normal")
            done(out, page)

        # 6: every probe step's renders
        groups = racers(run) or [""]
        if "" in {r for (r, _s, _k) in run.images}:
            groups = [""] + [g for g in groups if g]
        for racer in groups:
            steps = sorted({s for (r, s, _k) in run.images if r == racer})
            per = 3
            for i in range(0, len(steps), per):
                page = new("Probe renders" + (f": {racer}" if racer else ""))
                yy = 0.9
                for step in steps[i:i + per]:
                    rec = record_at(run, step)
                    is_best = step == bs and (not racer or racer == (race.get("champion") or racer))
                    x, y, w, h = page.card(0.5, yy, W - 1.0, 3.3, "")
                    if is_best:
                        page.box(0.5, yy, W - 1.0, 3.3, color="none", edge=GOLD, lw=2, z=3)
                    face = rec.get("gold_norm") if rec.get("gold_select") else rec.get("face_norm", rec.get("face"))
                    page.text(x, y - 0.02, f"Step {step}" + ("   ★ best checkpoint" if is_best else ""), size=10.5, weight="bold",
                              color=GOLD if is_best else INK)
                    page.text(x + w, y, f"face {_num(face)}   body {_num(rec.get('body_norm', rec.get('body')))}   "
                              f"score {_num(rec.get('combined'))}   smoothed {_num(rec.get('combined_ema'))}"
                              + ("" if rec.get("copy") is None else f"   copy {rec['copy']:.2f}"), size=8, color=MUTED, ha="right")
                    gy = y + 0.34
                    for kind, seeds, ch in (("face", 8, 0.8), ("body", 8, 1.2)):
                        if (racer, step, kind) in run.images:
                            probe_grid(page, x + 0.02, gy, run, [step], kind, seeds, ch, racer, gap=0.05)
                            gy += ch + 0.26
                    yy += 3.5
                done(out, page)

        # 7: the dataset
        if run.dataset:
            page = new("Dataset")
            n_img = len(run.dataset)
            x, y, w, h = page.card(0.5, 0.9, W - 1.0, 6.0, f"{n_img} images: a spread through the set")
            count = min(24, n_img)
            picks = [run.dataset[round(i * (n_img - 1) / max(1, count - 1))] for i in range(count)]
            cols = 6
            cw, ch = (w - (cols - 1) * 0.06) / cols, (h - 3 * 0.06) / 4
            for i, pth in enumerate(picks):
                page.picture(x + (i % cols) * (cw + 0.06), y + (i // cols) * (ch + 0.06), cw, ch, pth, max_px=300)
            x, y, w, h = page.card(0.5, 7.1, W - 1.0, 3.6, "Captions of three of them")
            yy = y
            for pth in picks[::max(1, count // 3)][:3]:
                try:
                    cap = pth.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                wrapped = _wrap(cap, 118)[:5]
                page.text(x, yy, pth.name, size=7.5, color=MUTED, weight="bold")
                page.text(x, yy + 0.18, "\n".join(wrapped), size=7.8, linespacing=1.35)
                yy += 0.3 + 0.17 * len(wrapped)
            done(out, page)

        # 8: the maps, at three points of the run
        if run.maps:
            rows_m = run.maps["rows"]
            picks = [rows_m[0], rows_m[len(rows_m) // 2], rows_m[-1]] if len(rows_m) >= 3 else rows_m
            for pic in run.maps["pictures"][:3]:
                page = new("Training maps")
                page.text(0.5, 0.85, "One training photo, and what the maps showed at three points of the run "
                          "(middle noise level)", size=8.5, color=MUTED)
                yy = 1.15
                tile = 1.32
                for row in picks:
                    views = map_views(run, row, pic.get("id"))[:4]
                    if not views:
                        continue
                    x, y, w, h = page.card(0.5, yy, W - 1.0, 2.95, f"Step {row.get('step')}")
                    drawn = page.picture(x, y, tile, h - 0.2, run.maps["dir"] / str(pic.get("image")), max_px=360)
                    if drawn:
                        page.text(x + tile / 2, drawn[1] + drawn[3] + 0.05, "training photo", size=7, color=MUTED, ha="center")
                    for j, (label, grid) in enumerate(views):
                        tx = x + (j + 1) * (tile + 0.1)
                        drawn = page.heat(tx, y, tile, h - 0.2, grid)
                        if drawn:
                            page.text(tx + tile / 2, drawn[1] + drawn[3] + 0.05, label, size=7, color=MUTED, ha="center")
                    yy += 3.15
                done(out, page)

        # 9: files and the log's notes
        page = new("Files and notes")
        saved = [str(s) for s in st.get("saved") or []]
        x, y, w, h = page.card(0.5, 0.9, W - 1.0, 0.8 + 0.22 * max(1, len(saved)), "Saved files")
        for i, s in enumerate(saved or ["(none recorded)"]):
            page.text(x, y + i * 0.22, _short(s, 110), size=8)
        top = 1.95 + 0.22 * max(1, len(saved))
        notes = run.notes[:28]
        x, y, w, h = page.card(0.5, top, W - 1.0, min(A4[1] - top - 0.7, 0.8 + 0.34 * max(1, len(notes))),
                               f"Warnings and errors in the log ({len(run.notes)})")
        yy = y
        for level, text in notes:
            if yy > A4[1] - 1.1:
                break
            wrapped = _wrap(text, 112)[:2]
            page.text(x, yy, "▲" if level == "WARNING" else "■", size=7, color=ORANGE if level == "WARNING" else RED)
            page.text(x + 0.18, yy, "\n".join(wrapped), size=7.6, linespacing=1.3)
            yy += 0.17 * len(wrapped) + 0.08
        if not notes:
            page.text(x, y, "None.", size=8, color=MUTED)
        done(out, page)
        meta = out.infodict()
        meta["Title"] = f"Training run report: {run.job}"
        meta["Creator"] = f"RulZ Trainer {_version()}"
    return path


# ---------------------------------------------------------------- making the files


def generate(job_dir, out_dir=None, image: bool = True, pdf_file: bool = True, root=None) -> dict:
    """Write the run's report files. Returns {"image": Path, "pdf": Path} for the ones written. Raises ReportError
    when nothing can be made (no such folder, matplotlib missing)."""
    run = load(job_dir, root)
    out = Path(out_dir) if out_dir else run.job_dir / "report"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", run.job).strip("._") or "run"
    done = {}
    if image:
        done["image"] = summary(run, out / f"{name}_report.png")
    if pdf_file:
        done["pdf"] = pdf(run, out / f"{name}_report.pdf")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write a training run's report: a summary picture and a PDF.")
    ap.add_argument("job_dir", help="the run's folder (output/<job>)")
    ap.add_argument("--out", help="where to write (default: <job folder>/report)")
    ap.add_argument("--image", action="store_true", help="the summary picture alone")
    ap.add_argument("--pdf", action="store_true", help="the PDF alone")
    ap.add_argument("--root", help="the folder the config's relative paths start from (default: the install)")
    args = ap.parse_args(argv)
    both = not (args.image or args.pdf)
    try:
        done = generate(args.job_dir, args.out, image=both or args.image, pdf_file=both or args.pdf, root=args.root)
    except ReportError as e:
        print(f"[no report] {e}", file=sys.stderr)
        return 1
    for path in done.values():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
