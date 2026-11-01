"""Configuration race: several configurations of one run train in turn, and the likeness probes pick the winner.

Each configuration (a race_configs row: a name plus the settings it changes from the run's own) trains probe_every
steps and is probed, then the next one does. Once every configuration still in the race has been probed (a round),
the one with the lowest smoothed combined probe score is dropped; race_warmup_probes rounds pass before the first
drop. The last one left keeps training until `steps`, counted in its own steps, so the race adds the others' steps
on top of a normal run (estimate_steps).

race_drop_confidence (0 = off) makes that drop a paired test: the configurations are probed on the same seeds, so
the lowest leaves only when its per-seed scores sit below the round leader's with that confidence (a one-sided paired
t-test over the rounds since the last drop, at most POOL_ROUNDS of them, after the warm-up; likeness_stats.paired). A
round too close to call drops nobody; after race_max_close_rounds such rounds in a row, or once the lowest has trained
`steps` of its own, the lowest leaves anyway, so the race still ends.

race_patience_drop drops a configuration instead once its own probes have not improved for race_patience in a row,
while the others race on: a setting that learns slowly but ends higher is not cut for being behind early. With it, or
with race_drop_lowest off (every configuration trains all the steps unless patience ends it), the winner is the one
whose best smoothed score was the highest at any point; its best checkpoint becomes <job>_best.safetensors.

The comparison is kept fair: every configuration starts from the same LoRA initialisation for its shape, sees the
same images in the same order with the same noise (its own copies of the run's random streams), and is probed at the
same own step count with the same probe seeds. A row that sets its own `seed` opts out of that on purpose: its own
initialisation, image order and noise (seed twins, the ingredients of a seed soup, soup.py). The model, quantization, GPUs, dataset, resolutions and the probes
themselves are shared; RACE_KEYS lists what a configuration may change.

The trainer keeps one set of LoRA wrappers over the union of every configuration's layers. Switching parks the
outgoing configuration's weights, optimizer state and weight average on the CPU and puts the next one's in: its
weights (any rank), scale, dropout and enabled layers (swap_in / swap_out).

race_follow (a relay) drops nobody: after each round every configuration continues from that round's winner (the
highest smoothed score): its LoRA weights, optimizer state, weight average and likeness history, while keeping its
own settings, seed, image order and noise. With seed rows, every round tries a few random paths from the best LoRA so
far and keeps the one that probed best. Every configuration trains `steps` of its own (n x one run); the last round's
winner is the run's LoRA. The configurations must share the LoRA's shape (no rank, alpha, layer or network rows).

With race_parallel (data parallel, 2+ GPUs) the configurations train side by side instead: a "wave" of one
configuration per GPU runs the same steps at the same time, each on its GPU's own model copy, and is probed at the
end of its turn as before; an odd one left over trains alone, split over the GPUs when its step has batches enough
(gradient_accumulation 2 or more) and the split is faster than one GPU, else on the first GPU. Each configuration
then owns its parameters (build_master), so any model copy can host it: the copies' wrappers carry (attach) a
configuration's weights while it trains there."""
from __future__ import annotations

import dataclasses
import json
import math
import re
import zlib
from types import SimpleNamespace

import torch
from torch import nn

# what a race configuration may change, by group (the WebUI lists them); everything else is shared by the race
RACE_KEYS: dict[str, tuple[str, ...]] = {
    "LR & optimizer": (
        "learning_rate", "optimizer", "weight_decay", "lr_scheduler", "warmup_steps", "max_grad_norm",
        "lora_plus_ratio", "lora_ema", "automagic_min_lr", "automagic_max_lr", "automagic_polarity_history",
        "automagic_clip_threshold", "automagic_groups", "automagic_sensitivity_quantile", "automagic_vote",
        "automagic_max_lr_change"),
    "Timesteps & loss": (
        "timestep_sampling", "logit_mean", "logit_std", "sigmoid_scale", "timestep_shift", "clean_share", "logsnr_mean",
        "logsnr_std",
        "loss_weighting",
        "sigma_min", "sigma_max", "content_power", "region_weights", "face_loss_weight", "body_loss_weight",
        "background_loss_weight", "face_margin", "region_feather_cells", "region_prior_weight", "id_loss_weight",
        "id_loss_sigma_max", "id_loss_every", "id_loss_margin_cells", "id_loss_min_face_cells", "id_loss_random_ref",
        "attn_localize", "attn_localize_weight", "id_align", "id_align_weight", "body_align", "body_align_weight",
        "likeness_curriculum", "likeness_curriculum_strength", "likeness_lr_control", "likeness_lr_min",
        "likeness_lr_max", "likeness_lr_step", "loss_watch", "per_image_lr", "loss_watch_buckets",
        "loss_watch_beta", "loss_watch_high", "loss_watch_low", "loss_watch_improve_tol",
        "loss_watch_healthy_mult", "loss_watch_exclude_after"),
    "LoRA rank & layers": (
        "lora_rank", "lora_alpha", "lora_targets", "lora_blocks", "train_text_embedder", "lora_dropout",
        "network_type", "lokr_factor", "oft_block_size", "oft_constraint", "dora"),
    "Data & augmentation": ("caption_dropout", "flip_augment", "crop_augment", "crop_augment_weight", "look_warmup",
                            "look_warmup_start", "look_warmup_epochs"),
    # read at every step: the run needs isolation on (the pairs and DOP captions are prepared once for everyone)
    "LoRA Isolation": ("dop_weight", "dop_every"),
    "Self-training": ("selftrain", "selftrain_at", "selftrain_images", "selftrain_min_likeness", "selftrain_share",
                      "selftrain_weight"),
    # a row's own seed gives it its own LoRA initialisation, image order and noise: seed twins for a seed soup
    "Seed": ("seed",),
}
ALLOWED = {k for keys in RACE_KEYS.values() for k in keys}
# what a relay (race_follow) cannot change: the winner's weights must fit every configuration's LoRA as they are
SHAPE_KEYS = tuple(k for k in RACE_KEYS["LoRA rank & layers"] if k != "lora_dropout")
REGION_KEYS = ("face_loss_weight", "body_loss_weight", "background_loss_weight", "face_margin", "region_feather_cells")
NAME_RE = re.compile(r"[A-Za-z0-9_.+-]{1,40}")


# ---------------------------------------------------------------- configurations


def _value(text: str):
    """A setting's value as typed in a row: numbers (2e-4 too), true/false, [a, b] lists, else the text."""
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [_value(x) for x in _split(inner)] if inner else []
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    return s


def _split(text: str) -> list[str]:
    """Split on commas outside brackets and quotes."""
    parts, depth, cur, quote = [], 0, "", ""
    for ch in text:
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0 and not quote:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def parse_overrides(text: str) -> dict:
    """"learning_rate=2e-4, lora_targets=[q, k, v]" -> {"learning_rate": 0.0002, "lora_targets": ["q", "k", "v"]}."""
    out = {}
    for part in _split(text or ""):
        if not part.strip():
            continue
        key, sep, val = part.partition("=")
        if not sep:
            raise ValueError(f"'{part.strip()}' is not setting=value")
        out[key.strip()] = _value(val)
    return out


def format_overrides(overrides: dict) -> str:
    def fmt(v):
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return f"{v:g}"
        return f'"{v}"' if isinstance(v, str) and re.search(r"[,\[\]\s]", v) else str(v)

    return ", ".join(f"{k}={fmt(v)}" for k, v in overrides.items())


def _coerce(base, key: str, value):
    """Match the type of the setting's default (an int where a float is expected, a single name for a list)."""
    default = getattr(base, key)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(default, int) and isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(default, list) and isinstance(value, str):
        return [value]
    return value


def row_overrides(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "name"}


def variant_configs(cfg) -> list[tuple[str, object, dict]]:
    """(name, TrainConfig, overrides) for every race row: the run's settings with the row's changes, race off."""
    base = dataclasses.replace(cfg, race=False, race_configs=[])
    out = []
    for i, row in enumerate(cfg.race_configs):
        name = str(row.get("name") or f"c{i + 1}")
        over = {k: _coerce(base, k, v) for k, v in row_overrides(row).items()}
        out.append((name, dataclasses.replace(base, **over), over))
    return out


def validate(cfg) -> None:
    """Checks a race config before training; raises ValueError with what to change."""
    rows = cfg.race_configs or []
    if len(rows) < 2:
        raise ValueError("race needs at least 2 configurations (race_configs)")
    if cfg.probe_every <= 0 or not (cfg.face_refs or cfg.body_refs):
        raise ValueError("race needs likeness probes: set probe_every > 0 and face_refs and/or body_refs")
    if cfg.resolution_step_up:
        raise ValueError("race and resolution_step_up cannot be combined")
    if cfg.resume_state or cfg.resume_from:
        raise ValueError("a race starts fresh: clear resume_state / resume_from (resume a raced run from its "
                         "winner's config, <job>_winner.yaml)")
    if cfg.race_warmup_probes < 0:
        raise ValueError("race_warmup_probes must be 0 or more")
    if not 0.0 <= float(getattr(cfg, "race_drop_confidence", 0.0)) < 1.0:
        raise ValueError("race_drop_confidence must be 0 (off) or a confidence below 1, such as 0.9")
    if int(getattr(cfg, "race_max_close_rounds", 3)) < 0:
        raise ValueError("race_max_close_rounds must be 0 or more")
    if getattr(cfg, "race_patience_drop", False) and int(cfg.race_patience) < 1:
        raise ValueError("race_patience must be 1 or more (probes without improvement before a configuration leaves)")
    follow = bool(getattr(cfg, "race_follow", False))
    if follow:
        if getattr(cfg, "race_parallel", False):
            raise ValueError("a relay race (race_follow) takes turns on the GPUs: turn race_parallel off")
        shaped = sorted({k for r in rows if isinstance(r, dict) for k in r if k in SHAPE_KEYS})
        if shaped:
            raise ValueError(f"in a relay race (race_follow) every configuration continues from the round's winner, "
                             f"so they must share the LoRA's shape: remove {', '.join(shaped)} from the rows")
        if cfg.steps % cfg.probe_every:
            raise ValueError(f"in a relay race every configuration trains to the last step and is probed there: make "
                             f"steps ({cfg.steps}) a multiple of probe_every ({cfg.probe_every})")
    if getattr(cfg, "race_parallel", False):
        from .multigpu import parse_devices

        if cfg.multi_gpu != "data_parallel" or len(parse_devices(cfg.devices, cfg.device)) < 2:
            raise ValueError("race_parallel (one configuration per GPU) needs multi_gpu: data_parallel with at "
                             "least two devices")
        if any(float(r.get("id_loss_weight", cfg.id_loss_weight) or 0) > 0 for r in rows if isinstance(r, dict)) \
                or cfg.id_loss_weight > 0:
            raise ValueError("race_parallel does not support the identity loss: set id_loss_weight to 0 or turn "
                             "race_parallel off")
        if cfg.selftrain or any(_value(str(r.get("selftrain", ""))) is True for r in rows if isinstance(r, dict)):
            raise ValueError("race_parallel does not support self-training: turn selftrain off or race_parallel "
                             "off")
    names = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"race configuration {i + 1} must be a mapping of settings")
        name = str(row.get("name") or f"c{i + 1}")
        if not NAME_RE.fullmatch(name):
            raise ValueError(f"race configuration name '{name}' must be 1-40 letters, digits or _ . + -")
        names.append(name)
        bad = sorted(k for k in row_overrides(row) if k not in ALLOWED)
        if bad:
            raise ValueError(f"race configuration '{name}' changes settings a race shares: {', '.join(bad)} "
                             f"(a configuration may change: {', '.join(sorted(ALLOWED))})")
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"race configuration names must be unique: {', '.join(dup)}")
    for name, vcfg, _ in variant_configs(cfg):
        try:
            vcfg.validate()
        except ValueError as e:
            raise ValueError(f"race configuration '{name}': {e}") from e
    drop_lowest = getattr(cfg, "race_drop_lowest", True) and not follow
    if not drop_lowest and cfg.steps % cfg.probe_every:
        raise ValueError(f"without dropping the lowest every configuration trains to the last step and is probed "
                         f"there: make steps ({cfg.steps}) a multiple of probe_every ({cfg.probe_every})")
    est = estimate_steps(len(rows), cfg.race_warmup_probes, cfg.probe_every, cfg.steps, drop_lowest)
    if est["drop_lowest"] and cfg.steps < est["race_own_steps"]:
        raise ValueError(f"the race needs at least {est['race_own_steps']} steps ({len(rows)} configurations, "
                         f"{cfg.race_warmup_probes} warm-up probes, a probe every {cfg.probe_every} steps); raise "
                         f"steps or lower the others")


def estimate_steps(n: int, warmup: int, probe_every: int, steps: int, drop_lowest: bool = True) -> dict:
    """Training steps a race costs. Rounds: `warmup` with all n configurations, then n, n-1, ..., 2 (a drop after
    each). race_own_steps: each survivor's own steps when the winner is found; total: every configuration's steps
    plus the winner's remaining ones; probes: probe renders during the race. Without drop_lowest every configuration
    may train all the steps: the most it can cost (patience drops make it less)."""
    if not drop_lowest:
        probes = n * max(1, math.ceil(steps / max(1, probe_every)))
        return {"rounds": math.ceil(steps / max(1, probe_every)), "race_steps": n * steps, "race_own_steps": steps,
                "total": n * steps, "probes": probes, "ratio": float(n), "drop_lowest": False}
    rounds = warmup + n - 1
    race = probe_every * (warmup * n + n * (n + 1) // 2 - 1)
    own = rounds * probe_every
    total = race + max(0, steps - own)
    return {"rounds": rounds, "race_steps": race, "race_own_steps": own, "total": total,
            "probes": warmup * n + n * (n + 1) // 2 - 1, "ratio": total / max(1, steps), "drop_lowest": True}


# ---------------------------------------------------------------- schedule

POOL_ROUNDS = 4  # a paired drop test pools the per-seed scores of at most this many rounds (since the last drop)


class RaceSchedule:
    """Who trains next and who is dropped. Pure bookkeeping (no torch).

    drop_lowest: each round (after the warm-up) the lowest score leaves. patience > 0: a configuration leaves as soon
    as its own probes have not improved for that many in a row (the likeness controller's count), whatever the others
    do, so a slow starter is not cut for being behind. Without drop_lowest a configuration that reaches total_steps
    of its own leaves as finished. The last one in the race trains on to the end; with patience or without
    drop_lowest the winner is decided at the end of the run: the best score any configuration reached (champion)."""

    def __init__(self, names: list[str], warmup: int, drop_lowest: bool = True, patience: int = 0,
                 total_steps: int | None = None, follow: bool = False, rising: bool = False,
                 confidence: float = 0.0, max_close: int = 3):
        self.names = list(names)
        # confidence > 0: the lowest is dropped only when a paired test on the shared probe seeds puts it below the
        # round leader with this confidence; max_close undecided rounds in a row (or its steps used up) drop it anyway
        self.confidence = max(0.0, float(confidence))
        self.max_close = max(0, int(max_close))
        self.seeds: dict[int, dict[int, list]] = {}  # index -> {round: its probe's combined score per seed}
        self.close = 0  # rounds in a row too close to call
        self.last_drop_round = 0
        self.warmup = int(warmup)
        # follow (a relay): nobody leaves until every one has trained total_steps; each round names a leader that
        # the others continue from, and the last round's leader wins
        self.follow = bool(follow)
        self.drop_lowest = bool(drop_lowest) and not self.follow
        self.patience = 0 if self.follow else max(0, int(patience))
        # rising: a configuration out of patience stays while its smoothed score is still going up from its last
        # probe (coming back after a dip, not yet above its best)
        self.rising = bool(rising) and self.patience > 0
        self.last: dict[int, float | None] = {}  # each configuration's previous probe score
        self.spared: dict[int, int] = {}  # this round's: index -> probes without improvement it stayed on with
        self.leader: int | None = None  # the latest round's winner (follow)
        self.total_steps = total_steps
        self.alive = list(range(len(names)))
        self.round = 1
        self.pending = list(self.alive)  # still to train and probe this round, in order
        self.members = list(self.alive)  # in the race when this round began
        self.scores: dict[int, float | None] = {}
        self.history: list[dict] = []  # per round: {"round", "scores": {name: score}, "steps": {name: the step the
        #                                score was probed at}, "dropped": name | None, "left": {name: why}}
        self.steps: dict[int, int | None] = {}
        self.dropped: dict[int, int] = {}  # index -> round it was dropped after
        self.manual: set[int] = set()  # the ones dropped by hand (drop), not by a round's lowest score
        self.stale: set[int] = set()  # left after `patience` probes without improvement
        self.finished: set[int] = set()  # left having trained total_steps (without drop_lowest)
        self.left: dict[int, str] = {}  # this round's: index -> "patience" | "finished"
        self.just_dropped: list[int] = []  # who left with the last record() call, in order
        self.best: dict[int, float | None] = {}  # each configuration's best smoothed score so far
        self.best_step: dict[int, int | None] = {}
        self.champion: int | None = None  # the winner, once decided (decide())

    @property
    def keeps_all(self) -> bool:
        """The winner is the best score reached by anyone, decided at the end (not the last one left)."""
        return not self.follow and (self.patience > 0 or not self.drop_lowest)

    @property
    def done(self) -> bool:
        return len(self.alive) <= 1

    @property
    def current(self) -> int:
        """The configuration that trains now (the winner once the race is over)."""
        return self.pending[0] if self.pending else self.alive[0]

    def wave(self, slots: int) -> list[int]:
        """The configurations that train at the same time now: the first `slots` of those still to be probed this
        round (one per GPU). A round with an odd number left ends with a wave of one."""
        return [] if self.done else list(self.pending[:max(1, int(slots))])

    def record(self, idx: int, score: float | None, step: int | None = None, best: float | None = None,
               best_step: int | None = None, since_improve: int | None = None, seeds: list | None = None) -> int | None:
        """The probe score of `idx` after its turn this round (best / best_step / since_improve: its controller's;
        seeds: that probe's combined score per seed, for race_drop_confidence). Returns the configuration dropped as
        the round's lowest when this probe ended a round with a drop; just_dropped lists everyone who left with this
        call (patience, finished, lowest). A missing score (no face found) counts as the lowest; on a tie the later
        one goes. The last one in the race never leaves."""
        if self.done or not self.pending or idx != self.pending[0]:
            raise ValueError(f"configuration {idx} is not the one training now")
        self.pending.pop(0)
        self.scores[idx] = score
        if seeds:
            self.seeds.setdefault(idx, {})[self.round] = list(seeds)
        self.steps[idx] = step
        self.just_dropped = []
        if best is not None:
            self.best[idx], self.best_step[idx] = best, best_step
        why = None
        prev, self.last[idx] = self.last.get(idx), score
        if self.follow:
            pass  # a relay keeps everyone until the last round
        elif self.patience and since_improve is not None and since_improve >= self.patience:
            if self.rising and score is not None and prev is not None and score > prev:
                self.spared[idx] = since_improve
            else:
                why = "patience"
        elif not self.drop_lowest and self.total_steps and step is not None and step >= self.total_steps:
            why = "finished"
        if why and len(self.alive) > 1:
            self.alive.remove(idx)
            self.dropped[idx] = self.round
            (self.stale if why == "patience" else self.finished).add(idx)
            self.left[idx] = why
            self.just_dropped.append(idx)
        if self.pending and not self.done:
            return None
        entry = {"round": self.round, "scores": {self.names[i]: self.scores.get(i) for i in self.members},
                 "steps": {self.names[i]: self.steps.get(i) for i in self.members}, "dropped": None,
                 "left": {self.names[i]: w for i, w in self.left.items()}}
        if self.spared:
            entry["spared"] = {self.names[i]: n for i, n in self.spared.items()}
        dropped = None

        def key(i):
            s = self.scores.get(i)
            return (-math.inf if s is None else s, -i)

        if self.follow:  # the round's highest score leads (a tie keeps the leader, else the earlier row wins)
            self.leader = max(self.alive, key=lambda i: (key(i)[0], i == self.leader, -i))
            entry["leader"] = self.names[self.leader]
            if self.total_steps and all((self.steps.get(i) or 0) >= self.total_steps for i in self.alive):
                for i in [i for i in self.alive if i != self.leader]:
                    self.alive.remove(i)
                    self.dropped[i] = self.round
                    self.finished.add(i)
                    self.just_dropped.append(i)
                entry["final"] = True
        elif self.drop_lowest and self.round > self.warmup and not self.done:
            lowest = min(self.alive, key=key)
            if self._drop_verdict(lowest, key, entry):
                dropped = lowest
                self.alive.remove(dropped)
                self.dropped[dropped] = self.round
                entry["dropped"] = self.names[dropped]
                self.just_dropped.append(dropped)
                self.close, self.last_drop_round = 0, self.round
            else:
                self.close += 1
        self.history.append(entry)
        self.round += 1
        self.scores, self.steps, self.left, self.spared = {}, {}, {}, {}
        self.pending = [] if self.done else list(self.alive)
        self.members = list(self.alive)
        return dropped

    def paired_test(self, leader: int, lowest: int) -> dict:
        """leader's per-seed scores minus lowest's, pooled over the rounds since the last drop (after the warm-up, at
        most POOL_ROUNDS): likeness_stats.paired's n, diff, sd_diff, rho and one-sided p (leader above lowest)."""
        from .likeness_stats import paired

        first = max(self.warmup + 1, self.last_drop_round + 1, self.round - POOL_ROUNDS + 1)
        a, b = [], []
        for r in range(first, self.round + 1):
            sa, sb = self.seeds.get(leader, {}).get(r), self.seeds.get(lowest, {}).get(r)
            if sa and sb:
                n = min(len(sa), len(sb))
                a += sa[:n]
                b += sb[:n]
        return paired(a, b)

    def _drop_verdict(self, lowest: int, key, entry: dict) -> bool:
        """Does the round's lowest leave? Always without race_drop_confidence; with it, when the paired test puts it
        below the round leader with that confidence, when max_close rounds in a row were too close to call, or when
        it has trained total_steps of its own. The test and the reason go into the round's entry."""
        if self.confidence <= 0:
            return True
        leader = max((i for i in self.alive if i != lowest), key=key)
        st = self.paired_test(leader, lowest)
        test = {"lowest": self.names[lowest], "vs": self.names[leader], "n": st["n"], "diff": st["diff"],
                "p": st["p"]}
        if st["p"] is not None and st["p"] < 1.0 - self.confidence:
            test["why"] = "paired"
        elif self.total_steps and (self.steps.get(lowest) or 0) >= self.total_steps:
            test["why"] = "steps"
        elif self.close + 1 > self.max_close:
            test["why"] = "close"
        else:
            entry["close"] = test
            return False
        entry["test"] = test
        return True

    def decide(self, bests: dict[int, float | None] | None = None) -> int:
        """The winner: the last one left, or with keeps_all the highest best score any configuration reached (bests:
        the controllers' final ones, which include the last one's training after the race; a tie goes to the one
        left in the race)."""
        if bests:
            self.best.update({i: b for i, b in bests.items() if b is not None})
        if not self.keeps_all or not self.best:
            self.champion = self.alive[0]
            return self.champion
        last = self.alive[0] if self.alive else None
        self.champion = max(self.best, key=lambda i: (self.best[i], i == last, -i))
        return self.champion

    def drop(self, idx: int) -> None:
        """Take a configuration out of the race at any time, probed this round or not (a manual drop). The race goes
        on without it: a round it was the last one of to be probed ends here, with this drop standing in for the
        round's own (nobody else is dropped for that round); one left makes it the winner."""
        if self.done or idx not in self.alive:
            raise ValueError(f"configuration {idx} is not in the race")
        self.alive.remove(idx)
        if idx in self.pending:
            self.pending.remove(idx)
        self.scores.pop(idx, None)
        self.steps.pop(idx, None)
        self.dropped[idx] = self.round
        self.manual.add(idx)
        if self.done:
            self.pending = []
            return
        if not self.pending:
            entry = {"round": self.round, "scores": {self.names[i]: self.scores.get(i) for i in self.alive},
                     "steps": {self.names[i]: self.steps.get(i) for i in self.alive},
                     "dropped": self.names[idx], "manual": True}
            if self.follow:  # the round still names its leader (nobody continues from it: weights stay as they are)
                self.leader = max(self.alive, key=lambda i: (-math.inf if self.scores.get(i) is None
                                                             else self.scores[i], -i))
                entry["leader"] = self.names[self.leader]
            self.history.append(entry)
            self.round += 1
            self.scores, self.steps, self.left, self.spared = {}, {}, {}, {}
            self.pending = list(self.alive)
            self.members = list(self.alive)

    def state(self) -> dict:
        if self.champion is not None:
            winner = self.names[self.champion]
        else:
            winner = self.names[self.alive[0]] if self.done and not self.keeps_all else None
        return {"names": self.names, "warmup": self.warmup, "round": self.round, "alive": [self.names[i] for i in
                self.alive], "dropped": {self.names[i]: r for i, r in self.dropped.items()},
                "manual": [self.names[i] for i in sorted(self.manual)],
                "stale": [self.names[i] for i in sorted(self.stale)],
                "finished": [self.names[i] for i in sorted(self.finished)],
                "drop_lowest": self.drop_lowest, "patience": self.patience, "rising": self.rising,
                "follow": self.follow, "confidence": self.confidence, "max_close": self.max_close,
                "close": self.close,
                "leader": self.names[self.leader] if self.leader is not None else None,
                "best": {self.names[i]: b for i, b in self.best.items()},
                "best_step": {self.names[i]: s for i, s in self.best_step.items()},
                "last": self.names[self.alive[0]] if self.done and self.alive else None,
                "history": self.history, "winner": winner}


# ---------------------------------------------------------------- one configuration's LoRA and state


class Variant:
    """One configuration of a race: its settings, its LoRA (parked on the CPU while another one trains) and the
    per-run state the trainer swaps in when it trains."""

    def __init__(self, idx: int, name: str, cfg, overrides: dict, names: list[str]):
        self.idx, self.name, self.cfg, self.overrides = idx, name, cfg, overrides
        self.names = list(names)  # its LoRA modules (a subset of the trainer's wrappers)
        self.rank, self.alpha, self.dropout = int(cfg.lora_rank), float(cfg.lora_alpha), float(cfg.lora_dropout)
        self.kind, self.factor = str(getattr(cfg, "network_type", "lora")), int(getattr(cfg, "lokr_factor", 8))
        self.block, self.constraint = int(getattr(cfg, "oft_block_size", 32)), float(getattr(cfg, "oft_constraint", 0))
        self.dora = bool(getattr(cfg, "dora", False)) and self.kind != "oft"
        self.weights: dict[str, dict[str, torch.Tensor]] = {}  # {module: {part: CPU fp32 tensor}} while parked
        self.started = False  # optimizer, scheduler and the rest are built on its first turn
        self.state: dict = {}  # the trainer's per-run state (optimizer, scheduler, ema, step, losses, ...)
        self.offloaded: list = []  # optimizer state tensors parked on the CPU: (param, key, device)
        self.alive = True
        self.master: dict = {}  # race_parallel: {module: view over the configuration's own Parameters} (build_master)

    @property
    def scale(self) -> float:
        return self.alpha / self.rank if self.kind == "lora" else 1.0

    @property
    def label(self) -> str:
        from .lora import adapter_label

        return adapter_label(self.kind, self.rank, self.alpha, self.factor, self.block, self.dora, self.constraint)

    def _ns(self, parts: dict, **extra) -> SimpleNamespace:
        """A module-like view of the configuration's adapter on one layer (for save_lora and the optimizer)."""
        return SimpleNamespace(kind=self.kind, alpha=self.alpha, rank=self.rank, factor=self.factor,
                               block=self.block, constraint=self.constraint, dora=self.dora, trainable=True,
                               adapter_params=lambda parts=parts: parts, **extra)

    def init_lora(self, wrappers: dict, seed: int) -> None:
        """Fresh adapter weights as LoRALinear.init_parts makes them (a LoRA / LoKr: the first factor
        kaiming-uniform, the second zero; OFT: zero; DoRA's magnitude: the layer's own norm). Seeded per module name,
        so every configuration with the same shapes starts from the same weights."""
        for name in self.names:
            m = wrappers[name]
            g = torch.Generator().manual_seed((seed * 1_000_003 + zlib.crc32(name.encode())) % (2**63))
            self.weights[name] = m.init_parts(self.kind, self.rank, self.factor, self.block, self.dora, generator=g)

    def view(self, use_average: bool = False) -> dict:
        """Module-like views of the parked weights (or of its weight average) for lora.save_lora."""
        ema = self.state.get("ema") if use_average else None
        avg = dict(zip(ema.keys, ema.shadow)) if ema is not None else {}
        out = {}
        for name in self.names:
            parts = {part: avg.get(f"{name}.{part}", t) for part, t in self.weights[name].items()}
            out[name] = self._ns(parts)
        return out

    @property
    def n_weights(self) -> int:
        return sum(t.numel() for parts in self.weights.values() for t in parts.values())

    def build_master(self, device="cpu") -> None:
        """race_parallel: the configuration's own trainable parameters, in place of the shared wrappers'. `weights`
        then holds the same Parameters (saving, parking and swap_in read them); `master` has module-like views that
        the optimizer, the weight average and save_lora take like the wrappers."""
        self.master = {}
        for name in self.names:
            parts = {part: nn.Parameter(t.detach().clone().to(device)) for part, t in self.weights[name].items()}
            self.weights[name] = parts
            view = self._ns(parts, multiplier=1.0)
            if self.kind == "lora":
                view.lora_down, view.lora_up = (SimpleNamespace(weight=parts["lora_down"]),
                                                SimpleNamespace(weight=parts["lora_up"]))
            self.master[name] = view

    @torch.no_grad()
    def move_master(self, device) -> None:
        """Move the parameters, the optimizer state and the weight average to `device` (the GPU the configuration
        trains on, or the CPU while it waits). The Parameter objects stay, so the optimizer keeps pointing at them."""
        for parts in self.weights.values():
            for p in parts.values():
                p.data = p.data.to(device)
                p.grad = None
        opt = self.state.get("optimizer")
        if opt is not None:
            move_optimizer(opt, device)
        move_average(self.state.get("ema"), to_cpu=False)  # the shadow follows its parameters


def _dropout_module(p: float) -> nn.Module:
    return nn.Dropout(p) if p > 0 else nn.Identity()


@torch.no_grad()
def swap_in(v: Variant, wrappers: dict) -> None:
    """Make `v` the trained LoRA: its weights on each layer's device, its rank / alpha / dropout, and every layer it
    does not train switched off (its weights freed)."""
    for name, m in wrappers.items():
        dev = m.lora_down.weight.device  # the adapter's own device (a swapped block's frozen weight is in RAM)
        if name in v.weights:
            dora, block = getattr(v, "dora", False), getattr(v, "block", m.block)
            if (m.kind, m.dora, m.block) != (v.kind, dora, block):
                m.set_kind(v.kind, dora=dora, block=block)
            params = m.adapter_params()
            for part, t in v.weights[name].items():
                params[part].data = t.detach().to(dev)  # the same storage when already there (race_parallel)
            m.rank, m.alpha, m.factor, m.scale = v.rank, v.alpha, v.factor, v.scale
            m.constraint = getattr(v, "constraint", 0.0)
            p = getattr(m.dropout, "p", 0.0)
            if p != v.dropout:
                m.dropout = _dropout_module(v.dropout)
            m.trainable = True
        else:
            m.trainable = False
            for p in m.adapter_params().values():
                p.data = torch.empty((0,) * p.dim() if p.dim() else (0,), device=dev)


@torch.no_grad()
def swap_out(v: Variant, wrappers: dict) -> None:
    """Park `v`'s current weights on the CPU."""
    for name in v.names:
        m = wrappers[name]
        v.weights[name] = {part: p.detach().to("cpu", copy=True) for part, p in m.adapter_params().items()}


@torch.no_grad()
def attach(v: Variant, wrappers: dict) -> None:
    """Make the wrappers of one model copy carry `v`'s weights again (race_parallel, after an optimizer step or an
    event that loaded other weights): the same storage as the configuration's parameters on the copy's own device,
    a copy on another. The layers' settings (kind, rank, scale, enabled layers) are those swap_in already set."""
    for name in v.names:
        m = wrappers[name]
        dev = m.lora_down.weight.device
        params = m.adapter_params()
        for part, t in v.weights[name].items():
            params[part].data = t.detach().to(dev)


@torch.no_grad()
def collect_grads(v: Variant, replicas: list[dict]) -> None:
    """Move the gradients that backward left on the wrappers of the model copies `v` trained on into its own
    parameters (summed over the copies, losses were already scaled by the step's batch count) and clear them."""
    for name in v.names:
        for part, master in v.weights[name].items():
            total = None
            for wrappers in replicas:
                p = wrappers[name].adapter_params()[part]
                if p.grad is None:
                    continue
                g = p.grad.to(master.device)
                total = g if total is None else total.add_(g)
                p.grad = None
            master.grad = total


@torch.no_grad()
def move_optimizer(opt, device) -> None:
    """Every tensor of an optimizer's state to `device` (a parked configuration's state sits on the CPU)."""
    for st in opt.state.values():
        for k, t in list(st.items()):
            if torch.is_tensor(t) and t.device != torch.device(device):
                st[k] = t.to(device)


def offload_optimizer(opt) -> list:
    """Move an optimizer's GPU state tensors to the CPU; returns what moved (for onload_optimizer)."""
    moved = []
    for p, st in opt.state.items():
        for k, t in list(st.items()):
            if torch.is_tensor(t) and t.device.type == "cuda":
                st[k] = t.to("cpu")
                moved.append((p, k, t.device))
    return moved


def onload_optimizer(opt, moved: list) -> None:
    for p, k, dev in moved:
        opt.state[p][k] = opt.state[p][k].to(dev)


def seed_rows(seeds, prefix: str = "seed") -> list[dict]:
    """Race rows that differ only in their seed: [{"name": "seed1", "seed": 1}, ...] (duplicates dropped)."""
    out = []
    for s in seeds:
        s = int(s)
        if all(r["seed"] != s for r in out):
            out.append({"name": f"{prefix}{s}", "seed": s})
    return out


def parse_seeds(text: str) -> list[int]:
    """"1, 2, 3" or "1-4" or "7 9 11-12" -> the seeds, in order."""
    out = []
    for part in re.split(r"[,\s]+", (text or "").strip()):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if sep and lo:
            a, b = int(lo), int(hi)
            if b < a or b - a > 63:
                raise ValueError(f"'{part}' is not a seed range (low-high, at most 64 seeds)")
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


@torch.no_grad()
def follow_leader(win: Variant, v: Variant) -> list[str]:
    """A relay race: `v` continues from the round's winner `win`, both parked (weights, optimizer state and weight
    average on the CPU). v takes win's LoRA weights, optimizer state (when the optimizers are of one kind; else v's
    starts over), weight average and likeness history, and keeps its own settings, seed, image order, noise,
    learning-rate schedule and likeness lr factor and curriculum bias (they belong to its own settings and optimizer).
    The trainer's LoRA wrappers are shared, so both optimizers key their state by the same parameters. Returns notes
    on what could not be carried over."""
    import copy

    notes = []
    v.weights = {name: {part: t.detach().clone() for part, t in parts.items()} for name, parts in win.weights.items()}
    wo, vo = win.state.get("optimizer"), v.state.get("optimizer")
    if vo is not None:
        vo.state.clear()
        v.offloaded = []
        if wo is not None and type(wo) is type(vo):
            for p, st in wo.state.items():
                vo.state[p] = {k: t.detach().clone() if torch.is_tensor(t) else copy.deepcopy(t) for k, t in st.items()}
            v.offloaded = list(win.offloaded)
        else:
            notes.append(f"{v.name}'s optimizer ({type(vo).__name__}) starts over: {win.name} uses "
                         f"{type(wo).__name__ if wo is not None else 'none'}")
    we, ve = win.state.get("ema"), v.state.get("ema")
    if ve is not None:
        if we is not None:
            ve.shadow = [s.detach().clone() for s in we.shadow]
            ve.images, ve.updates = we.images, we.updates
        else:  # the average starts over from the winner's weights
            flat = {f"{name}.{part}": t for name, parts in v.weights.items() for part, t in parts.items()}
            ve.shadow = [flat[k].float().clone() for k in ve.keys]
            ve.images = ve.updates = 0
    wc, vc = win.state.get("controller"), v.state.get("controller")
    if wc is not None and vc is not None:
        c = copy.deepcopy(wc, {id(wc.cfg): vc.cfg})
        c.lr_factor, c.face_bias = vc.lr_factor, vc.face_bias
        v.state["controller"] = c
    return notes


def move_average(ema, to_cpu: bool) -> None:
    """Park a LoraEMA's shadow on the CPU, or bring it back to its weights' devices."""
    if ema is None:
        return
    ema.shadow = [s.to("cpu") if to_cpu else s.to(p.device) for s, p in zip(ema.shadow, ema.params)]


def region_key(cfg) -> tuple:
    """The settings baked into the cached region weight maps: configurations that differ here need their own maps."""
    return tuple(getattr(cfg, k) for k in REGION_KEYS)


def region_dir(cfg) -> str:
    """Folder name (inside the cache) for the region maps of these region settings."""
    return f"regions_{zlib.crc32(repr(region_key(cfg)).encode()):08x}"


def variant_items(items: list, cfg, maps_dir=None) -> list:
    """A configuration's share of the race's dataset items: its crop kinds only, no flipped latents unless it flips,
    and its own region maps (maps_dir) when its region weights differ from the run's."""
    mode = (cfg.crop_augment or "none").lower()
    kinds = {"image", "pair"} | ({"face_crop"} if "face" in mode else set()) | ({"bust_crop"} if "bust" in mode
                                                                                 else set())
    out = []
    for it in items:
        if it.category not in kinds:
            continue
        changes = {}
        if not cfg.flip_augment and it.latent_flip_path is not None:
            changes["latent_flip_path"] = None
        if maps_dir is not None and it.region_path is not None:
            changes["region_path"] = maps_dir / it.region_path.name
        out.append(dataclasses.replace(it, **changes) if changes else it)
    return out


@dataclasses.dataclass
class RaceRun:
    """A race in progress: its configurations, schedule, the trainer's LoRA wrappers and the one training now."""
    variants: list
    schedule: RaceSchedule
    wrappers: dict
    base_cfg: object
    estimate: dict
    active: Variant | None = None
    best_saved: dict = dataclasses.field(default_factory=dict)  # relay: the best scores saved so far, by kind

    @property
    def running(self) -> bool:
        return not self.schedule.done


def _gap(t: dict) -> str:
    return "n/a" if t.get("diff") is None else f"{t['diff']:.1f} points"


def _p(t: dict) -> str:
    return "n/a (under 3 seeds)" if t.get("p") is None else f"{t['p']:.2f}"


def describe_round(entry: dict, schedule: RaceSchedule) -> str:
    """A round for the log: "2: a 41.0%, b 38.5% -> b is out" (with who left on patience or finished their steps)."""
    fmt = ", ".join(f"{n} {'n/a' if s is None else f'{s:.1f}%'}" for n, s in entry["scores"].items())
    why = {"patience": f"out after {schedule.patience} probes without improvement", "finished": "finished its steps"}
    parts = [f"{n} {why[w]}" for n, w in (entry.get("left") or {}).items()]
    parts += [f"{n} stays: {k} probes without improvement but rising again" for n, k in
              (entry.get("spared") or {}).items()]
    if entry.get("leader") and entry.get("final"):
        parts.append(f"{entry['leader']} wins the last round")
    elif entry.get("leader"):
        parts.append(f"{entry['leader']} leads: everyone continues from its LoRA")
    if entry.get("dropped"):
        how = "by hand" if entry.get("manual") else "lowest"
        t = entry.get("test")
        if t:
            how += {"paired": f", below {t['vs']} by {_gap(t)} over {t['n']} paired seeds, p {_p(t)}",
                    "steps": f", its steps used up (vs {t['vs']}: {_gap(t)}, p {_p(t)})",
                    "close": f", still too close to call after {schedule.max_close} rounds (vs {t['vs']}: "
                             f"{_gap(t)}, p {_p(t)})"}[t["why"]]
        parts.append(f"{entry['dropped']} is out ({how})")
    elif entry.get("close"):
        t = entry["close"]
        parts.append(f"too close to call: {t['lowest']} is {_gap(t)} below {t['vs']} over {t['n']} paired seeds "
                     f"(p {_p(t)}, needs below {1 - schedule.confidence:.2g}); nobody is dropped")
    elif schedule.drop_lowest and entry["round"] <= schedule.warmup:
        parts.append("warm-up: nobody is dropped")
    return f"{entry['round']}: {fmt}" + (f" -> {'; '.join(parts)}" if parts else "")


def write_report(path, schedule: RaceSchedule, variants: list[Variant], extra: dict | None = None) -> None:
    rows = [{"name": v.name, "settings": format_overrides(v.overrides), "dropped_after_round":
             schedule.dropped.get(v.idx)} for v in variants]
    path.write_text(json.dumps({**schedule.state(), "configs": rows, **(extra or {})}, indent=2), encoding="utf-8")
