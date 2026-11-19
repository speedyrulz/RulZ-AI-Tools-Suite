"""Region sweep: the loss per region read at fixed noise, picture by picture, and on pictures kept out of training.

The region watch reads each picture once an epoch from its training step, at a random noise level: free, but noisy.
A sweep runs the model again on a picture at three fixed noise levels with fixed noise (as the training maps do), so
between two sweeps only the weights change, and a picture's "learned" is measured against its own first sweep. It
costs 3 forward passes per picture per sweep.

Two kinds of picture are swept:
  * training pictures (region_sweep_every, region_sweep_pictures): a clean per-picture reading, and what the "ahead"
    steering rule reads with region_steer_signal: sweep;
  * held-out pictures (holdout_images): photos kept out of training. What the model learns on them it has learned of
    the subject, not of the photos; the gap to the training pictures is memorising. The held-out subject score can
    keep a best checkpoint (holdout_track_best) and stop the run (holdout_patience).

Like the training maps it changes nothing about training: no gradients, the model in eval mode and put back, the
global RNG restored, the weight average used as samples and probes use it. A failure is never fatal: one log line and
the sweeps stop for the rest of the run.

Output in the job folder: region_sweep.jsonl (one line per sweep) and region_sweep.json (the whole state: a resumed
run reads it back, because a first sweep of an already trained model would be the wrong start).
"""
from __future__ import annotations

import contextlib
import json
import logging
import time
import zlib
from pathlib import Path

from .regionwatch import COLUMNS, MIN_CELLS, PROTECTED, REGIONS, SUBJECT, region_losses, tracked

log = logging.getLogger("rulztrainer")

SIGMAS = (0.25, 0.5, 0.75)
SEED = 4321
STATE_FILE, LOG_FILE = "region_sweep.json", "region_sweep.jsonl"
MIN_SET = 10  # photos a set needs before any may be held out
MAX_SHARE = 0.2  # of the photos, at most this share held out
STALL = 0.005  # a held-out score within this of its earlier best has not improved


def whole_photo(item) -> bool:
    """A photo as it is: not a close-up crop made from one, not a generated or paired picture."""
    return getattr(item, "category", "image") == "image" and tracked(item)


def pick_holdout(items, n: int) -> set[str]:
    """The source images to keep out of training: n of the set's photos spread evenly through them by name (the
    middles of n equal stretches, so the same set always gives the same ones)."""
    sources = sorted({str(it.image) for it in items if whole_photo(it)})
    n = max(0, min(int(n), len(sources)))
    return {sources[int((i + 0.5) * len(sources) / n)] for i in range(n)}


def check_holdout(n_photos: int, n: int) -> None:
    """Raises ValueError when n photos cannot be held out of a set of n_photos."""
    if n_photos < MIN_SET:
        raise ValueError(f"holdout_images needs a set of at least {MIN_SET} photos, this one has {n_photos}")
    if n > max(1, int(n_photos * MAX_SHARE)):
        raise ValueError(f"holdout_images {n} is more than a fifth of the set's {n_photos} photos: at most "
                         f"{max(1, int(n_photos * MAX_SHARE))}")


def split_holdout(items, n: int) -> tuple[list, list]:
    """(training items, held-out items): everything made from a held-out photo (its crops, its other resolutions)
    leaves training; the held-out items are those photos themselves."""
    held = pick_holdout(items, n)
    if not held:
        return list(items), []
    return ([it for it in items if str(it.image) not in held],
            [it for it in items if str(it.image) in held and whole_photo(it)])


def fixed_noise(key: str, level: int, shape):
    """The noise of one picture at one noise level: its own generator, the same in every sweep and every run."""
    import torch

    g = torch.Generator("cpu").manual_seed((SEED * 1000003 + zlib.crc32(str(key).encode("utf-8")) * 31 + level)
                                           % (2 ** 63 - 1))
    return torch.randn(shape, generator=g)


def forward_readings(model, device, chosen: list[dict], averaged=None) -> dict:
    """key -> per noise level [losses, cells] (COLUMNS order) at the fixed noise."""
    import torch

    from .trainmaps import eager_stance

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    out = {}
    try:
        model.eval()
        with (averaged() if averaged is not None else contextlib.nullcontext()), torch.no_grad(), eager_stance():
            for pr in chosen:
                latents, embeds, mask = pr["load"]()
                latents = latents.to(device, torch.float32)
                embeds, mask = embeds.to(device), mask.to(device)
                rows = []
                for i, s in enumerate(SIGMAS):
                    noise = fixed_noise(pr["key"], i, latents.shape).to(device)
                    noisy = ((1 - s) * latents + s * noise).to(torch.bfloat16)
                    sigma = torch.full((latents.shape[0],), s * 1000.0, device=device)
                    err = (model(noisy, sigma, embeds, mask).float() - (noise - latents)).pow(2)
                    if err.dim() != 4:
                        raise RuntimeError("the model's prediction is not a latent grid")
                    losses, cells = region_losses(err, [pr["parts"]])
                    if losses[0] is None:
                        raise RuntimeError(f"the parts map of {Path(pr['key']).name} does not fit its latent")
                    rows.append([losses[0], cells[0]])
                    del noise, noisy, err
                out[pr["key"]] = rows
                del latents, embeds, mask
    finally:
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return out


class SplitSizer:
    """sweep_split: how many of a sweep's pictures go to the second GPU, the rest being read on the training GPU. It
    learns seconds per picture on each side from the sweeps already done and keeps the second GPU's share inside
    SHARE of the time before the next sweep is due, less what the probes, samples and maps take of it. When a sweep
    was not in as the next one came due (training waited), the budget shrinks by that overrun, and grows back a little
    with every sweep that was in time."""

    SHARE = 0.8  # of the interval the second GPU may spend on a sweep (the rest is slack)
    SMOOTH = 0.5  # weight of the older seconds per picture
    LATE = 0.25  # seconds of waiting that count as an overrun

    def __init__(self):
        self.remote_spp: float | None = None  # seconds per picture on the second GPU (compute only)
        self.local_spp: float | None = None  # and on the training GPU
        self.scale = 1.0  # what the overrun feedback has left of the budget
        self.remote_total = 0.0  # compute seconds of the sweeps' remote parts so far
        self.mark: float | None = None  # the second GPU's other work (seconds) at the previous sweep

    def _note(self, old: float | None, seconds: float, pictures: int) -> float | None:
        if pictures <= 0 or seconds <= 0:
            return old
        new = seconds / pictures
        return new if old is None else self.SMOOTH * old + (1 - self.SMOOTH) * new

    def note_remote(self, seconds: float, pictures: int) -> None:
        self.remote_total += max(0.0, seconds)
        self.remote_spp = self._note(self.remote_spp, seconds, pictures)

    def note_local(self, seconds: float, pictures: int) -> None:
        self.local_spp = self._note(self.local_spp, seconds, pictures)

    def note_wait(self, waited: float, interval: float | None) -> None:
        """Training waited `waited` seconds for the last sweep when the next one came due, `interval` seconds after
        it (None: not known)."""
        if not interval or interval <= 0:
            return
        if waited > self.LATE:
            self.scale = max(0.05, self.scale * interval / (interval + waited))
        else:
            self.scale = min(1.0, self.scale * 1.1)

    def other_work(self, busy: float) -> float:
        """Seconds the second GPU spent on other jobs since the previous sweep, from its busy seconds in all (the
        sweeps' own are taken off)."""
        other = max(0.0, busy - self.remote_total)
        out = 0.0 if self.mark is None else max(0.0, other - self.mark)
        self.mark = other
        return out

    def remote_count(self, n: int, held: int, every: int, step_seconds: float | None, other: float,
                     last: bool = False) -> int:
        """How many of this sweep's n pictures the second GPU reads: the first `held` ones (the held-out pictures
        come first and always go there) and, of the rest, as many as fit in the budget. The first sweep and the last
        (training ends after it) have nothing to wait for or to hide behind, so they split by speed; so does any sweep
        before the second GPU's speed and the step time are known (an unmeasured side counts as equally fast)."""
        if n <= 0:
            return 0
        held = max(0, min(held, n))
        if last or self.remote_spp is None or not step_seconds:
            r = self.remote_spp or self.local_spp or 1.0
            lo = self.local_spp or self.remote_spp or 1.0
            # (at least one picture while the second GPU's speed is unknown, or a small sweep would never measure it)
            floor = held if self.remote_spp is not None else max(held, 1)
            return min(n, max(floor, int(round(n * lo / (lo + r)))))
        budget = self.SHARE * every * step_seconds * self.scale - other
        fit = int(budget // self.remote_spp) if budget > 0 else 0
        return min(n, max(held, fit))


class RegionSweep:
    """probes: [{"key", "held": bool, "parts": the picture's parts tensor, "load": () -> (latents [1, C, h, w],
    embeds [1, S, D], mask [1, S])}]. every: steps between sweeps. rotate: training pictures per sweep (0 = all); the
    first sweep and the held-out pictures always take all."""

    def __init__(self, job_dir, probes: list[dict], every: int, rotate: int = 0, resume: bool = False,
                 patience: int = 0):
        self.job_dir = Path(job_dir)
        self.probes = [p for p in probes if p.get("parts") is not None]
        self.every = max(1, int(every))
        self.rotate = max(0, int(rotate))
        self.patience = max(0, int(patience))
        self.base: dict[str, list] = {}  # key -> per noise level [losses, cells] at its first sweep
        self.latest: dict[str, dict] = {}  # key -> {"step", "learned": {column: share}, "contrast": {region: x}}
        self.cursor = 0
        self.next_step = 0
        self.last_step: int | None = None
        self.failed = False
        self.late = False  # the first sweep was of an already trained model (a resume without a saved state)
        self.history: list[dict] = []
        self.best: tuple[float, int] | None = None  # the held-out subject score's highest and its step
        self.mark: float | None = None  # the held-out subject score at its last rise of more than STALL
        self.since_best = 0  # sweeps since that rise (what holdout_patience counts)
        self.held_best: dict[str, float] = {}  # region -> its best held-out learned share before the last sweep
        self.held_flat: dict[str, int] = {}  # region -> sweeps in a row without a new best on the held-out pictures
        if resume:
            self.late = not self._load()
        else:
            for name in (STATE_FILE, LOG_FILE):
                with contextlib.suppress(OSError):
                    (self.job_dir / name).unlink()

    # ------------------------------------------------------------ persistence
    def _load(self) -> bool:
        try:
            data = json.loads((self.job_dir / STATE_FILE).read_text(encoding="utf-8"))
            keys = {p["key"] for p in self.probes}
            if not data.get("base") or set(data.get("held") or []) != {p["key"] for p in self.probes if p["held"]}:
                return False  # nothing swept yet, or other pictures held out: start again
            self.base = {k: v for k, v in data["base"].items() if k in keys}
            self.latest = {k: v for k, v in (data.get("latest") or {}).items() if k in keys}
            self.cursor, self.late = int(data.get("cursor", 0)), bool(data.get("late"))
            self.history = list(data.get("history") or [])
            best = data.get("best")
            self.best = (float(best[0]), int(best[1])) if best else None
            mark = data.get("mark")  # (a state saved before 3.35.2 has none: its best was the mark)
            self.mark = float(mark) if mark is not None else (self.best[0] if self.best else None)
            self.since_best = int(data.get("since_best", 0))
            self.held_best = {str(k): float(v) for k, v in (data.get("held_best") or {}).items()}
            self.held_flat = {str(k): int(v) for k, v in (data.get("held_flat") or {}).items()}
            if self.history:
                self.next_step = int(self.history[-1]["step"]) + self.every
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return False
        return True

    def _save(self) -> None:
        data = {"base": self.base, "latest": self.latest, "cursor": self.cursor, "late": self.late,
                "held": [p["key"] for p in self.probes if p["held"]], "history": self.history[-400:],
                "best": list(self.best) if self.best else None, "mark": self.mark, "since_best": self.since_best,
                "held_best": self.held_best, "held_flat": self.held_flat}
        from .atomic import writing  # here, not at the top: atomic loads PyTorch, which this module leaves to callers

        with contextlib.suppress(OSError), writing(self.job_dir / STATE_FILE) as tmp:  # whole: a resumed run reads it
            tmp.write_text(json.dumps(data), encoding="utf-8")

    # ------------------------------------------------------------ schedule
    def due(self, step: int, last: bool = False) -> bool:
        if self.failed or not self.probes or step == self.last_step:
            return False
        return step >= self.next_step or last

    def chosen(self) -> list[dict]:
        """This sweep's pictures: every held-out one, every one without a first sweep yet, and of the rest all of
        them or the next `rotate` in turn."""
        held = [p for p in self.probes if p["held"]]
        train = [p for p in self.probes if not p["held"]]
        new = [p for p in train if p["key"] not in self.base]
        old = [p for p in train if p["key"] in self.base]
        if self.rotate and len(old) > self.rotate:
            n, start = len(old), self.cursor % len(old)
            old = (old + old)[start:start + self.rotate]
            self.cursor = (start + self.rotate) % n
        return held + new + old

    # ------------------------------------------------------------ the sweep
    def run(self, step: int, epoch: float, model, device, averaged=None) -> dict | None:
        """One sweep; its summary, or None after a failure (which turns the sweeps off, never the run)."""
        if self.failed or not self.probes:
            return None
        t0 = time.time()
        try:
            chosen = self.chosen()
            readings = self._forward(model, device, chosen, averaged)
        except Exception as e:  # noqa: BLE001
            self.fail(step, e)
            return None
        finally:
            self.last_step = step
            self.next_step = step + self.every
        return self.record(step, epoch, [pr["key"] for pr in chosen], readings, time.time() - t0)

    def begin(self, step: int) -> list[dict]:
        """A sweep of `step` that is computed elsewhere (probe_async's second GPU): its pictures, with the schedule
        moved on now, so due() does not fire again while it is out. record() takes the readings when they are in."""
        chosen = self.chosen()
        self.last_step = step
        self.next_step = step + self.every
        return chosen

    def fail(self, step: int, e) -> None:
        """A sweep failed: the sweeps are off for the rest of the run, training goes on."""
        self.failed = True
        log.warning("Region sweep failed at step %d, so the sweeps are off for the rest of this run; training "
                    "goes on (%s: %s)", step, type(e).__name__, str(e)[:200])
        log.debug("region sweep error", exc_info=True)

    def record(self, step: int, epoch: float, keys: list[str], readings: dict, seconds: float) -> dict:
        """What follows a sweep's readings, wherever they were computed: each picture's reading against its own
        first sweep, the summary row, the log and the saved state. Its summary."""
        for key in keys:
            self._record(key, readings[key], step)
        row = self._summarise(step, epoch, len(keys), seconds)
        self.history.append(row)
        with contextlib.suppress(OSError):
            with open(self.job_dir / LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        self._save()
        return row

    def _forward(self, model, device, chosen: list[dict], averaged) -> dict:
        """key -> per noise level [losses, cells] (COLUMNS order) at the fixed noise."""
        return forward_readings(model, device, chosen, averaged)

    def _record(self, key: str, rows: list, step: int) -> None:
        """A picture's reading against its own first sweep: the learned share per region (the mean over the noise
        levels) and, for the background and the clothes, the contrast with its subject."""
        if key not in self.base:
            self.base[key] = rows
        base = self.base[key]
        learned, contrast = {}, {}
        j_subject = COLUMNS.index(SUBJECT)
        for j, c in enumerate(COLUMNS):
            left = [rows[i][0][j] / base[i][0][j] for i in range(len(SIGMAS))
                    if base[i][1][j] >= MIN_CELLS and base[i][0][j] > 0]
            if left:
                learned[c] = round(1.0 - sum(left) / len(left), 5)
        for p in PROTECTED:
            j = COLUMNS.index(p)
            ratios = []
            for i in range(len(SIGMAS)):
                b, r = base[i], rows[i]
                if min(b[1][j], b[1][j_subject]) < MIN_CELLS or min(b[0][j], b[0][j_subject], r[0][j_subject]) <= 0:
                    continue
                ratios.append((r[0][j] / b[0][j]) / (r[0][j_subject] / b[0][j_subject]))
            if ratios:
                contrast[p] = round(sum(ratios) / len(ratios), 5)
        self.latest[key] = {"step": int(step), "learned": learned, "contrast": contrast}

    def _mean(self, held: bool) -> dict:
        """The mean learned share per region over the held-out (or the training) pictures swept so far."""
        out = {}
        keys = [p["key"] for p in self.probes if p["held"] == held and p["key"] in self.latest]
        for c in COLUMNS:
            vals = [self.latest[k]["learned"][c] for k in keys if c in self.latest[k]["learned"]]
            out[c] = round(sum(vals) / len(vals), 5) if vals else None
        return out

    def _summarise(self, step: int, epoch: float, swept: int, seconds: float) -> dict:
        train, held = self._mean(False), self._mean(True)
        gap = {c: (round(train[c] - held[c], 5) if train[c] is not None and held[c] is not None else None)
               for c in COLUMNS}
        first = len(self.history) == 0
        new_best = False
        score = held.get(SUBJECT)
        if score is not None and not first:  # (the first sweep is the start: nothing learned, nothing to keep)
            # The best (the chart's marker, holdout_track_best's checkpoint) is the highest score: a sweep reads at
            # fixed noise, so any rise is the weights. Only the stop waits for a rise of more than STALL, counted from
            # the score at the last such rise, so a slow creep of 0.1 points a sweep neither stops the run nor leaves
            # the best behind (with one band for both, a creep under STALL kept an early sweep as the best).
            if self.best is None or score > self.best[0]:
                self.best, new_best = (score, int(step)), True
            if self.mark is None or score > self.mark + STALL:
                self.mark, self.since_best = score, 0
            else:
                self.since_best += 1
        for c in REGIONS:  # per region: has the held-out reading stopped getting better?
            v = held.get(c)
            if v is None or first:
                continue
            if c not in self.held_best or v > self.held_best[c] + STALL:
                self.held_best[c], self.held_flat[c] = v, 0
            else:
                self.held_flat[c] = self.held_flat.get(c, 0) + 1
        return {"step": int(step), "epoch": round(float(epoch), 3), "train": train, "held": held, "gap": gap,
                "first": first, "swept": swept, "pictures": len([p for p in self.probes if not p["held"]]),
                "held_pictures": len([p for p in self.probes if p["held"]]), "seconds": round(seconds, 2),
                "new_best": new_best, "best": list(self.best) if self.best else None, "since_best": self.since_best,
                "flat": dict(self.held_flat), "late": self.late,
                "stop": bool(self.patience and self.best is not None and self.since_best >= self.patience)}

    # ------------------------------------------------------------ for the region watch and its steering
    def contrasts(self) -> dict[str, dict[str, float]]:
        """key -> {protected region: its contrast with the subject since the first sweep} for the training pictures
        swept so far (1 = as at the start, under 1 = learned faster than the subject)."""
        return {p["key"]: dict(self.latest[p["key"]]["contrast"]) for p in self.probes
                if not p["held"] and p["key"] in self.latest and self.latest[p["key"]]["contrast"]}


def learned_by_picture(sweep: "RegionSweep") -> dict[str, dict[str, float]]:
    """key -> {region: learned share since its first sweep} for the training pictures swept so far."""
    return {p["key"]: {c: v for c, v in sweep.latest[p["key"]]["learned"].items() if c in REGIONS}
            for p in sweep.probes if not p["held"] and p["key"] in sweep.latest}


def describe(row: dict) -> str:
    """One line for the log and the WebUI from a sweep's summary."""
    def shares(d):
        return ", ".join(f"{c} {100 * d[c]:.0f}%" for c in REGIONS if d.get(c) is not None)

    if row.get("first"):
        return (f"first sweep of {row.get('swept', 0)} pictures ({row.get('seconds', 0):.1f}s): the start every later "
                "one is measured from")
    text = f"training pictures learned {shares(row['train'])}" if row.get("pictures") else ""
    if row.get("held_pictures"):
        text += (" | " if text else "") + f"held-out pictures learned {shares(row['held'])}"
        gap = row.get("gap") or {}
        if gap.get(SUBJECT) is not None:
            text += f" | subject gap {100 * gap[SUBJECT]:+.0f} points"
        if row.get("best"):
            text += f" | held-out subject best {100 * row['best'][0]:.0f}% at step {row['best'][1]}"
    if row.get("late"):
        text += " | first sweep taken after a resume, so this counts from there"
    return text + f" ({row.get('swept', 0)} pictures, {row.get('seconds', 0):.1f}s)"


def read(job_dir) -> dict | None:
    """A job's saved sweep state (region_sweep.json), or None."""
    try:
        return json.loads((Path(job_dir) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
