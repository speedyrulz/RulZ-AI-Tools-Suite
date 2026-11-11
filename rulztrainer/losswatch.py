"""Per-image loss watch and per-image learning rate (in the spirit of Fizgig's loss watch).

Flow-matching loss varies by an order of magnitude with the noise level, so each sample's loss is first
normalised by the running expected loss of its sigma bucket (40 buckets). The *residual* (loss / expected)
is smoothed per image with an EMA, and at every epoch boundary each image gets a verdict from its residual
relative to the cohort median and from whether it is still improving:

    easy       well below the median and still improving          x1.0
    healthy    around the median                                   x healthy_mult (default 1.0)
    suspect    well above the median: still improving, or not an   x1.0
               outlier of the cohort
    stuck      an outlier (above the fence) and not improving      x0.5, then x0.25
    exhausted  well below the median and not improving             x0.6
    excluded   stuck at the lowest throttle for `exclude_after` more epochs; weight 0, remembered on disk

The fence (loss_watch_fence, after Fizgig's of 9 November 2026) is the cohort's median + 1.5 x its interquartile
range. Late in a run almost no image still improves by loss_watch_improve_tol an epoch, so "well above the median
and not improving" alone turns into "every hard image": Fizgig replayed 39 real runs and found about a third of
each dataset throttled that way, most of them images that kept getting easier at the full rate. The ladder stops at
x0.25 for the same reason (it went on to x0.125).

The watch itself only observes. `per_image_lr` applies the multipliers: as a true per-image learning rate
when the batch is a single image, else as a per-sample loss weight.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("rulztrainer")

VERDICTS = ("new", "healthy", "easy", "suspect", "stuck", "exhausted", "excluded")
STUCK_LADDER = (0.5, 0.25)
FENCE_MIN_IMAGES = 4  # a cohort smaller than this has no quartiles worth the name: the median rule alone
EXHAUSTED_MULT = 0.6


@dataclass
class ImageStat:
    key: str
    caption_hash: str = ""
    seen: int = 0
    resid_ema: float | None = None
    epoch_sum: float = 0.0
    epoch_n: int = 0
    history: list[float] = field(default_factory=list)  # mean residual per completed epoch
    verdict: str = "new"
    stuck_epochs: int = 0
    multiplier: float = 1.0
    excluded: bool = False
    last_loss: float | None = None
    last_sigma: float | None = None

    @property
    def trend(self) -> float | None:
        """Relative change of the epoch residual vs the previous epoch (negative = improving)."""
        if len(self.history) < 2 or self.history[-2] <= 0:
            return None
        return self.history[-1] / self.history[-2] - 1.0


def _flush_at_exit(ref) -> None:
    watch = ref()
    if watch is not None:
        watch.flush()


def outlier_fence(values: list[float], median: float, k: float) -> float:
    """The residual an image must be above to count as an outlier of its cohort: median + k x the interquartile
    range (quartiles by index, as Fizgig's). -inf, so nothing is held back, with k 0 or under FENCE_MIN_IMAGES."""
    n = len(values)
    if k <= 0 or n < FENCE_MIN_IMAGES:
        return -math.inf
    ordered = sorted(values)
    iqr = ordered[int(0.75 * (n - 1))] - ordered[int(0.25 * (n - 1))]
    return median + k * iqr


def _caption_hash(caption: str) -> str:
    return hashlib.sha1(caption.encode("utf-8")).hexdigest()[:12]


class LossWatch:
    def __init__(self, cfg, items, job_dir: Path, excluded_file: Path | None = None):
        self.cfg = cfg
        self.job_dir = Path(job_dir)
        self.n_buckets = max(4, int(cfg.loss_watch_buckets))
        self.beta = float(cfg.loss_watch_beta)
        # decayed per-bucket (sum, count); the expectation is a kernel-smoothed mean over neighbouring buckets
        self.bucket_sum = [0.0] * self.n_buckets
        self.bucket_cnt = [0.0] * self.n_buckets
        self.bucket_decay = 0.98
        self.kernel_width = 2.0
        self.stats: dict[str, ImageStat] = {}
        for it in items:
            self.stats[it.key] = ImageStat(it.key, _caption_hash(it.caption))
        self.epoch = 0
        self.epoch_obs = 0
        self.excluded_file = excluded_file
        self.cohort_median: float | None = None
        self.cohort_fence: float | None = None  # None: no fence this epoch (off, or too few images)
        self.log_path = self.job_dir / "per_image_loss.jsonl"
        import threading

        self._lock = threading.Lock()
        # per_image_loss.jsonl lines of the current step, written in one go when the next step starts (one open per
        # step instead of per sample), at an epoch's end and at exit
        self._pending: list[str] = []
        self._pending_step: int | None = None
        import atexit
        import weakref

        atexit.register(_flush_at_exit, weakref.ref(self))
        self._load_exclusions()

    # ------------------------------------------------------------ persistence
    def _load_exclusions(self) -> None:
        if not self.excluded_file or not self.excluded_file.exists():
            return
        try:
            data = json.loads(self.excluded_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        n = 0
        for key, st in self.stats.items():
            rec = data.get(key)
            if rec and rec.get("caption_hash") == st.caption_hash:  # editing the caption clears the record
                st.excluded, st.verdict, st.multiplier = True, "excluded", 0.0
                n += 1
        if n:
            log.info("Loss watch: %d image(s) excluded from a previous run (edit their captions to reset)", n)

    def _save_exclusions(self) -> None:
        if not self.excluded_file:
            return
        data = {k: {"caption_hash": st.caption_hash, "epoch": self.epoch} for k, st in self.stats.items() if st.excluded}
        self.excluded_file.parent.mkdir(parents=True, exist_ok=True)
        self.excluded_file.write_text(json.dumps(data, indent=1), encoding="utf-8")

    # ------------------------------------------------------------ observation
    def active_count(self) -> int:
        return sum(1 for st in self.stats.values() if not st.excluded)

    def expected_loss(self, b: int) -> float | None:
        """Kernel-weighted mean loss around bucket b (None until something has been observed)."""
        num = den = 0.0
        for k in range(self.n_buckets):
            if self.bucket_cnt[k] <= 0:
                continue
            w = math.exp(-((k - b) ** 2) / (2 * self.kernel_width ** 2))
            num += w * self.bucket_sum[k]
            den += w * self.bucket_cnt[k]
        return (num / den) if den > 0 else None

    def observe(self, step: int, key: str, loss: float, sigma: float) -> float:
        """Record one sample (thread-safe). Returns the sample's residual (loss / expected loss at that sigma). A
        non-finite loss (the trainer skips that step) is left out: it would stay in the running averages for good."""
        if not math.isfinite(loss):
            return 1.0
        with self._lock:
            return self._observe(step, key, loss, sigma)

    def _observe(self, step: int, key: str, loss: float, sigma: float) -> float:
        b = min(self.n_buckets - 1, max(0, int(sigma * self.n_buckets)))
        exp_before = self.expected_loss(b)  # expectation excludes the current sample
        self.bucket_sum[b] = self.bucket_sum[b] * self.bucket_decay + loss
        self.bucket_cnt[b] = self.bucket_cnt[b] * self.bucket_decay + 1.0
        expected = max(1e-8, exp_before) if exp_before is not None else None
        resid = loss / expected if expected is not None else 1.0
        st = self.stats.get(key)
        if st is None:
            st = self.stats[key] = ImageStat(key)
        st.seen += 1
        if expected is not None:  # a residual without an expectation carries no information
            st.resid_ema = resid if st.resid_ema is None else self.beta * st.resid_ema + (1 - self.beta) * resid
            st.epoch_sum += resid
            st.epoch_n += 1
        st.last_loss, st.last_sigma = loss, sigma
        self.epoch_obs += 1
        if self._pending and self._pending_step != step:  # a new step: the last one's lines in one write
            self._flush()
        self._pending_step = step
        self._pending.append(json.dumps({"step": step, "image": key, "loss": loss, "sigma": sigma, "bucket": b,
                                         "expected": expected, "residual": resid}) + "\n")
        return resid

    def flush(self) -> None:
        """Write the buffered per-sample lines to per_image_loss.jsonl (thread-safe)."""
        with self._lock:
            self._flush()

    def __del__(self):
        try:
            self._flush()
        except Exception:  # noqa: BLE001  (interpreter shutdown)
            pass

    def _flush(self) -> None:
        if not getattr(self, "_pending", None):
            return
        lines, self._pending = self._pending, []
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("".join(lines))
        except OSError:
            pass

    def epoch_complete(self) -> bool:
        return self.epoch_obs >= max(1, self.active_count())

    # ------------------------------------------------------------ verdicts
    def end_epoch(self, step: int) -> dict:
        """Close the epoch: compute verdicts and multipliers. Returns a summary for logging / status."""
        cfg = self.cfg
        self.flush()
        self.epoch += 1
        self.epoch_obs = 0
        for st in self.stats.values():
            if st.epoch_n:
                # judge on the smoothed residual: one noisy observation per epoch is not enough on its own
                st.history.append(st.resid_ema if st.resid_ema is not None else st.epoch_sum / st.epoch_n)
                st.history = st.history[-12:]
            st.epoch_sum, st.epoch_n = 0.0, 0
        active = [st for st in self.stats.values() if not st.excluded and st.history]
        if not active:
            return self.summary(step)
        latest = [st.history[-1] for st in active]
        median = statistics.median(latest)
        self.cohort_median = median
        fence = outlier_fence(latest, median, float(getattr(cfg, "loss_watch_fence", 0.0)))
        self.cohort_fence = fence if math.isfinite(fence) else None
        high, low, tol = cfg.loss_watch_high, cfg.loss_watch_low, cfg.loss_watch_improve_tol
        newly_excluded = []
        for st in active:
            r = st.history[-1]
            enough = len(st.history) >= 2
            improving = enough and r < st.history[-2] * (1 - tol)
            is_high, is_low = r > median * high, r < median * low
            if is_high and r > fence and enough and not improving:
                st.stuck_epochs += 1
                st.verdict = "stuck"
                st.multiplier = STUCK_LADDER[min(st.stuck_epochs - 1, len(STUCK_LADDER) - 1)]
                if cfg.loss_watch_exclude_after > 0 and st.stuck_epochs >= len(STUCK_LADDER) + cfg.loss_watch_exclude_after:
                    st.excluded, st.verdict, st.multiplier = True, "excluded", 0.0
                    newly_excluded.append(st.key)
            elif is_high:  # hard but improving, or inside the fence: recover from a throttle one rung at a time
                # (from the lowest rung, however long it was held: stuck_epochs counts on past the ladder's end)
                st.stuck_epochs = max(0, min(st.stuck_epochs, len(STUCK_LADDER)) - 1)
                st.verdict = "suspect"
                st.multiplier = STUCK_LADDER[st.stuck_epochs - 1] if st.stuck_epochs > 0 else 1.0
            elif is_low and enough and not improving:
                st.stuck_epochs = 0
                st.verdict, st.multiplier = "exhausted", EXHAUSTED_MULT
            elif is_low:
                st.stuck_epochs = 0
                st.verdict, st.multiplier = "easy", 1.0
            else:
                st.stuck_epochs = 0
                st.verdict, st.multiplier = "healthy", float(cfg.loss_watch_healthy_mult)
        if newly_excluded:
            self._save_exclusions()
            for k in newly_excluded:
                log.info("Loss watch: excluded %s (stuck for %d epochs)", Path(k).name, self.stats[k].stuck_epochs)
        self.write_table()
        return self.summary(step)

    def multiplier(self, key: str) -> float:
        st = self.stats.get(key)
        return st.multiplier if (st is not None and self.cfg.per_image_lr) else 1.0

    def sampling_mask(self) -> dict[str, float]:
        """Weight 0 for excluded images (multiply into other sampling weights)."""
        return {k: (0.0 if st.excluded else 1.0) for k, st in self.stats.items()}

    def counts(self) -> dict[str, int]:
        out = {v: 0 for v in VERDICTS}
        for st in self.stats.values():
            out[st.verdict] = out.get(st.verdict, 0) + 1
        return {k: v for k, v in out.items() if v}

    def rows(self, limit: int = 80) -> list[dict]:
        rows = []
        for st in self.stats.values():
            rows.append({"image": Path(st.key).name, "path": st.key, "verdict": st.verdict,
                         "residual": None if not st.history else round(st.history[-1], 3),
                         "ema": None if st.resid_ema is None else round(st.resid_ema, 3),
                         "trend": None if st.trend is None else round(st.trend * 100, 1),
                         "multiplier": st.multiplier, "seen": st.seen, "stuck_epochs": st.stuck_epochs})
        order = {"excluded": 0, "stuck": 1, "suspect": 2, "exhausted": 3, "healthy": 4, "easy": 5, "new": 6}
        rows.sort(key=lambda r: (order.get(r["verdict"], 9), -(r["residual"] or 0)))
        return rows[:limit]

    def summary(self, step: int) -> dict:
        return {"epoch": self.epoch, "step": step, "counts": self.counts(), "median": self.cohort_median,
                "fence": self.cohort_fence, "rows": self.rows(), "active": self.active_count(), "total": len(self.stats)}

    def write_table(self) -> None:
        data = {"epoch": self.epoch, "median": self.cohort_median, "fence": self.cohort_fence,
                "images": {k: asdict(st) for k, st in self.stats.items()}}
        try:
            (self.job_dir / "per_image_loss.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        except OSError:
            pass


def read_verdicts(job_dir: Path | None) -> dict[str, str]:
    """{image path: verdict} from a job's per_image_loss.json (for the Dataset tab badges)."""
    if job_dir is None:
        return {}
    p = Path(job_dir) / "per_image_loss.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {k: v.get("verdict", "new") for k, v in data.get("images", {}).items()}
