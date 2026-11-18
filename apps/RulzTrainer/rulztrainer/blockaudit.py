"""Block audit: how much of the likeness each group of blocks carries, read from the probe by switching the group off.

Every block_audit_every probes, after the normal probe has been scored, the likeness prompts are rendered again once
per block group with that group's adapter switched off (LoRALinear.multiplier 0: the frozen layer alone), at the same
seeds as the normal probe, so each seed's score is paired with its own score at full strength. A group's credit is the
normal probe's score less the ablated one: what the group adds at this point of the run. It is read for the face, the
body and the gold face, in raw points and (when the likeness scale normalizes) on the normalized scale, and for the
frame the prompts ask for (the prompt check's drift). Modules that sit in no block (embedders, final layers) form one
extra group, "other".

Report only: nothing here touches training, the likeness controller, the best checkpoint, Keel's lookahead, races or
patience. It costs groups x seeds x (face + body) extra renders per audit.

Like the region sweep a failure is never fatal: one log line and the audits stop for the rest of the run.

Output in the job folder: block_audit.jsonl (one line per audit) and block_audit.json (the state: a resumed run reads
it back, so its probe count and its history go on).
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
from pathlib import Path

log = logging.getLogger("rulztrainer")

STATE_FILE, LOG_FILE = "block_audit.json", "block_audit.jsonl"
KINDS = ("face", "body", "gold")
OTHER = "other"
MIN_GROUPS, MAX_GROUPS = 2, 16
IMAGE_TAG = "audit-g{index}-s{step:06d}"  # an image tag the probe readers (PROBE_RE, the WebUI) do not take for a probe


def image_tag(index: int, step: int) -> str:
    """The probe image tag of group `index` (1-based) at `step`: ..._audit-g1-s000600_face_0.png. It has no `_step`
    part, so the run report's probe pictures and a race's per-configuration names never read it as a probe."""
    return IMAGE_TAG.format(index=index, step=step)


# ------------------------------------------------------------ grouping
def split_even(items: list, n: int) -> list[list]:
    """items in n contiguous parts of near-equal size (the first ones one longer); fewer parts when there are fewer
    items than n."""
    items = list(items)
    n = max(1, min(int(n), len(items)))
    if not items:
        return []
    size, extra = divmod(len(items), n)
    out, at = [], 0
    for i in range(n):
        take = size + (1 if i < extra else 0)
        out.append(items[at:at + take])
        at += take
    return out


def span_text(indices) -> str:
    """"0-9" or "0-5,8,10-12" for block indices."""
    idx = sorted(set(int(i) for i in indices))
    if not idx:
        return "none"
    parts, start, prev = [], idx[0], idx[0]
    for b in idx[1:] + [None]:
        if b is not None and b == prev + 1:
            prev = b
            continue
        parts.append(f"{start}-{prev}" if prev > start else str(start))
        if b is not None:
            start = prev = b
    return ",".join(parts)


def block_map(blocks: list, injected: dict) -> dict[str, int | None]:
    """{module name: index of the block it sits in, None for one outside every block}. blocks: the model's blocks in
    order (Arch.compile_blocks); injected: {name: LoRALinear}. Found by identity, so it does not depend on how an
    architecture names its blocks."""
    by_id = {id(m): name for name, m in injected.items()}
    out: dict[str, int | None] = {name: None for name in injected}
    for i, blk in enumerate(blocks):
        for sub in blk.modules():
            name = by_id.get(id(sub))
            if name is not None and out[name] is None:
                out[name] = i
    return out


def build_groups(block_of: dict[str, int | None], n_groups: int) -> list[dict]:
    """The audit's groups: the blocks that hold an adapter, in index order, cut into n_groups contiguous parts of
    near-equal size ({"name": "g1", "blocks": [indices], "span": "0-9", "modules": [names]}), then "other" for the
    modules in no block when there are any."""
    indices = sorted({b for b in block_of.values() if b is not None})
    groups = []
    for i, part in enumerate(split_even(indices, n_groups)):
        members = set(part)
        groups.append({"name": f"g{i + 1}", "blocks": part, "span": span_text(part),
                       "modules": sorted(n for n, b in block_of.items() if b in members)})
    loose = sorted(n for n, b in block_of.items() if b is None)
    if loose:
        groups.append({"name": OTHER, "blocks": [], "span": "no block", "modules": loose})
    return groups


def same_groups(a: list[dict], b: list[dict]) -> bool:
    """Two group lists cut the same modules the same way (a resumed run's saved groups against this run's)."""
    return [(g["name"], g["modules"]) for g in a] == [(g["name"], g["modules"]) for g in b]


# ------------------------------------------------------------ credit maths
def _mean(values: list) -> float:
    return sum(values) / len(values)


def _sd(values: list) -> float | None:
    if len(values) < 2:
        return None
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def credit(normal: dict, ablated: dict, norm=None) -> dict:
    """A group's credit per kind from the normal probe's per-seed scores and the ablated render's, both
    {"face": [...], "body": [...], "gold": [...]} in seed order (None = nothing found in that render). Only seeds
    scored in both count (paired). Per kind {"mean": normal less ablated in raw points, "sd": of the paired
    differences (None below two seeds), "n": pairs, "normal", "ablated": the means over those pairs, "norm": the
    credit on the normalized scale when norm(kind, value) is given}. A kind with no pair is left out."""
    out = {}
    for kind in KINDS:
        pairs = [(a, b) for a, b in zip(normal.get(kind) or [], ablated.get(kind) or [])
                 if a is not None and b is not None]
        if not pairs:
            continue
        diffs = [a - b for a, b in pairs]
        sd = _sd(diffs)
        entry = {"mean": round(_mean(diffs), 3), "sd": None if sd is None else round(sd, 3), "n": len(pairs),
                 "normal": round(_mean([a for a, _ in pairs]), 3), "ablated": round(_mean([b for _, b in pairs]), 3)}
        if norm is not None:
            try:
                nd = [norm(kind, a) - norm(kind, b) for a, b in pairs]
                entry["norm"] = round(_mean(nd), 3)
                nsd = _sd(nd)
                entry["norm_sd"] = None if nsd is None else round(nsd, 3)
            except Exception:  # noqa: BLE001  (a scale without that kind: the raw credit stands)
                pass
        out[kind] = entry
    return out


def frame_credit(normal_drift, ablated_drift) -> dict | None:
    """The frame the prompts ask for, with the group off: {"normal", "ablated": the drift share of each (renders that
    left their frame / renders that had it), "credit": ablated less normal}, None when either is unknown. Above 0 the
    group holds the frame: without it more renders leave the frame."""
    if normal_drift is None or ablated_drift is None:
        return None
    return {"normal": round(float(normal_drift), 4), "ablated": round(float(ablated_drift), 4),
            "credit": round(float(ablated_drift) - float(normal_drift), 4)}


def spearman(a: list, b: list) -> float | None:
    """Rank correlation of two equally long lists (ties share their mean rank); None below three pairs or when one
    side is constant."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out, i = [0.0] * len(values), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out

    ra, rb = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    ma, mb = _mean(ra), _mean(rb)
    da, db = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    if da <= 0 or db <= 0:
        return None
    return round(sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(da * db), 4)


def stability(rows: list[dict], kind: str = "face") -> list[tuple[int, float | None]]:
    """[(step, Spearman correlation of the groups' credits with the previous audit's)] for the audits after the first,
    over the groups both have (by name)."""
    out = []
    for prev, row in zip(rows, rows[1:]):
        before = {g["name"]: ((g.get("credit") or {}).get(kind) or {}).get("mean") for g in prev.get("groups") or []}
        now = {g["name"]: ((g.get("credit") or {}).get(kind) or {}).get("mean") for g in row.get("groups") or []}
        names = [n for n in now if n in before]
        out.append((int(row.get("step", 0)), spearman([before[n] for n in names], [now[n] for n in names])))
    return out


# ------------------------------------------------------------ the log line
def _signed(v: float, digits: int = 1) -> str:
    return f"{v:+.{digits}f}"


def describe(row: dict) -> str:
    """One line for the log and the WebUI from an audit's row: per group its blocks and credit per kind (raw points,
    mean +- sd of the paired seeds), and the frame."""
    parts = []
    for g in row.get("groups") or []:
        bits = [f"{g['name']} blocks {g.get('span', '')}" if g.get("blocks") else f"{g['name']} (no block)"]
        for kind in KINDS:
            c = (g.get("credit") or {}).get(kind)
            if c:
                sd = c.get("sd")
                bits.append(f"{kind} {_signed(c['mean'])}" + (f"±{sd:.1f}" if sd is not None else ""))
        frame = g.get("frame")
        if frame:
            bits.append(f"frame drift {_signed(100 * frame['credit'], 0)} points")
        parts.append(" ".join(bits))
    n = row.get("seeds")
    tail = f" ({row.get('renders', 0)} renders, {row.get('seconds', 0):.0f}s" + (f", {n} seeds" if n else "") + ")"
    return " | ".join(parts) + tail


# ------------------------------------------------------------ the audit's state
class BlockAudit:
    """every: audit after every Nth probe (the step-0 baseline is not counted). groups: set once the model's blocks
    are known (set_groups). A resumed run reads back its probe count and history when its groups are the same."""

    def __init__(self, job_dir, every: int, n_groups: int = 6, seeds: int = 0, resume: bool = False):
        self.job_dir = Path(job_dir)
        self.every = max(1, int(every))
        self.n_groups = max(MIN_GROUPS, min(MAX_GROUPS, int(n_groups)))
        self.seeds = max(0, int(seeds))
        self.groups: list[dict] = []
        self.history: list[dict] = []
        self.probes_seen = 0
        self.last_step: int | None = None
        self.failed = False
        self._saved: dict | None = None
        if resume:
            self._saved = self._read_state()
        else:
            for name in (STATE_FILE, LOG_FILE):
                with contextlib.suppress(OSError):
                    (self.job_dir / name).unlink()

    # -------------------------------------------------------- persistence
    def _read_state(self) -> dict | None:
        data = read(self.job_dir)
        return data if isinstance(data, dict) and isinstance(data.get("groups"), list) else None

    def set_groups(self, groups: list[dict], step: int | None = None) -> bool:
        """The groups of this run's model; step: the step a resumed run starts from. Returns False when a resumed
        run's saved audits were cut differently (its history and block_audit.jsonl are then dropped and the count
        starts again). Audits saved after `step` (a resume from an earlier checkpoint) are of weights this run no
        longer has: they are dropped from both, and the count starts again."""
        self.groups = groups
        saved, self._saved = self._saved, None
        if saved is None:
            return True
        if not same_groups(saved["groups"], groups):
            self._trim_log(None)
            return False
        try:
            self.history = [r for r in saved.get("history") or [] if isinstance(r, dict)]
            self.probes_seen = int(saved.get("probes_seen", 0))
            last = saved.get("last_step")
            self.last_step = None if last is None else int(last)
        except (TypeError, ValueError):
            self.history, self.probes_seen, self.last_step = [], 0, None
            self._trim_log(None)
            return False
        if step is not None and self.last_step is not None and self.last_step > step:
            self.history = [r for r in self.history if _step_of(r) <= step]
            self.probes_seen, self.last_step = 0, None
            self._trim_log(step)
        return True

    def _trim_log(self, step: int | None) -> None:
        """block_audit.jsonl cut to the audits at or before `step` (None: all of them dropped)."""
        path = self.job_dir / LOG_FILE
        keep = [] if step is None else [r for r in read_log(self.job_dir) if _step_of(r) <= step]
        with contextlib.suppress(OSError):
            if keep:
                path.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
            else:
                path.unlink()

    def _save(self) -> None:
        data = {"every": self.every, "groups": self.groups, "probes_seen": self.probes_seen,
                "last_step": self.last_step, "history": self.history[-100:]}
        from .atomic import writing  # here, not at the top: atomic loads PyTorch, which this module leaves to callers

        with contextlib.suppress(OSError), writing(self.job_dir / STATE_FILE) as tmp:
            tmp.write_text(json.dumps(data), encoding="utf-8")

    # -------------------------------------------------------- schedule
    def tick(self, step: int) -> bool:
        """A probe at `step` has been scored: is the audit due? The step-0 baseline and a step already seen are not
        counted; every `every`-th probe after them is."""
        if self.failed or step <= 0 or step == self.last_step:
            return False
        self.last_step = int(step)
        self.probes_seen += 1
        due = self.probes_seen % self.every == 0
        if not due:
            self._save()
        return due

    def record(self, row: dict) -> None:
        """Keep one audit's row: the history, the log line in block_audit.jsonl, the state."""
        self.history.append(row)
        with contextlib.suppress(OSError):
            with open(self.job_dir / LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        self._save()

    def fail(self, step: int, error: Exception) -> None:
        """An audit went wrong: the audits are off for the rest of the run, training goes on."""
        self.failed = True
        log.warning("Block audit failed at step %d, so the audits are off for the rest of this run; training goes on "
                    "(%s: %s)", step, type(error).__name__, str(error)[:200])
        log.debug("block audit error", exc_info=True)


def _step_of(row: dict) -> int:
    try:
        return int(row.get("step", 0))
    except (TypeError, ValueError):
        return 0


def is_audit_image(name: str) -> bool:
    """A file name is one of the audit's renders (image_tag), not a probe's."""
    return "audit-g" in name and "_step" not in name


def read(job_dir) -> dict | None:
    """A job's saved audit state (block_audit.json), or None."""
    try:
        return json.loads((Path(job_dir) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_log(job_dir) -> list[dict]:
    """A job's audits, oldest first (block_audit.jsonl); lines that do not read are skipped."""
    rows = []
    try:
        with open(Path(job_dir) / LOG_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and "step" in row:
                    rows.append(row)
    except OSError:
        return []
    return rows
