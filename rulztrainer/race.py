"""Configuration race: several configurations of one run train in turn, and the likeness probes pick the winner.

Each configuration (a race_configs row: a name plus the settings it changes from the run's own) trains probe_every
steps and is probed, then the next one does. Once every configuration still in the race has been probed (a round),
the one with the lowest smoothed combined probe score is dropped; race_warmup_probes rounds pass before the first
drop. The last one left keeps training until `steps`, counted in its own steps, so the race adds the others' steps
on top of a normal run (estimate_steps).

The comparison is kept fair: every configuration starts from the same LoRA initialisation for its shape, sees the
same images in the same order with the same noise (its own copies of the run's random streams), and is probed at the
same own step count with the same probe seeds. The model, quantization, GPUs, dataset, resolutions and the probes
themselves are shared; RACE_KEYS lists what a configuration may change.

The trainer keeps one set of LoRA wrappers over the union of every configuration's layers. Switching parks the
outgoing configuration's weights, optimizer state and weight average on the CPU and puts the next one's in: its
weights (any rank), scale, dropout and enabled layers (swap_in / swap_out).

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
        "automagic_clip_threshold", "automagic_groups"),
    "Timesteps & loss": (
        "timestep_sampling", "logit_mean", "logit_std", "sigmoid_scale", "timestep_shift", "loss_weighting",
        "sigma_min", "sigma_max", "content_power", "region_weights", "face_loss_weight", "body_loss_weight",
        "background_loss_weight", "face_margin", "region_feather_cells", "region_prior_weight", "id_loss_weight",
        "id_loss_sigma_max", "id_loss_every", "id_loss_margin_cells", "id_loss_min_face_cells", "id_loss_random_ref",
        "likeness_curriculum", "likeness_curriculum_strength", "likeness_lr_control", "likeness_lr_min",
        "likeness_lr_max", "likeness_lr_step", "loss_watch", "per_image_lr", "loss_watch_buckets",
        "loss_watch_beta", "loss_watch_high", "loss_watch_low", "loss_watch_improve_tol",
        "loss_watch_healthy_mult", "loss_watch_exclude_after"),
    "LoRA rank & layers": (
        "lora_rank", "lora_alpha", "lora_targets", "lora_blocks", "train_text_embedder", "lora_dropout",
        "network_type", "lokr_factor"),
    "Data & augmentation": ("caption_dropout", "flip_augment", "crop_augment", "crop_augment_weight", "look_warmup",
                            "look_warmup_start", "look_warmup_epochs"),
}
ALLOWED = {k for keys in RACE_KEYS.values() for k in keys}
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
    if getattr(cfg, "race_parallel", False):
        from .multigpu import parse_devices

        if cfg.multi_gpu != "data_parallel" or len(parse_devices(cfg.devices, cfg.device)) < 2:
            raise ValueError("race_parallel (one configuration per GPU) needs multi_gpu: data_parallel with at "
                             "least two devices")
        if any(float(r.get("id_loss_weight", cfg.id_loss_weight) or 0) > 0 for r in rows if isinstance(r, dict)) \
                or cfg.id_loss_weight > 0:
            raise ValueError("race_parallel does not support the identity loss: set id_loss_weight to 0 or turn "
                             "race_parallel off")
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
    est = estimate_steps(len(rows), cfg.race_warmup_probes, cfg.probe_every, cfg.steps)
    if cfg.steps < est["race_own_steps"]:
        raise ValueError(f"the race needs at least {est['race_own_steps']} steps ({len(rows)} configurations, "
                         f"{cfg.race_warmup_probes} warm-up probes, a probe every {cfg.probe_every} steps); raise "
                         f"steps or lower the others")


def estimate_steps(n: int, warmup: int, probe_every: int, steps: int) -> dict:
    """Training steps a race costs. Rounds: `warmup` with all n configurations, then n, n-1, ..., 2 (a drop after
    each). race_own_steps: each survivor's own steps when the winner is found; total: every configuration's steps
    plus the winner's remaining ones; probes: probe renders during the race."""
    rounds = warmup + n - 1
    race = probe_every * (warmup * n + n * (n + 1) // 2 - 1)
    own = rounds * probe_every
    total = race + max(0, steps - own)
    return {"rounds": rounds, "race_steps": race, "race_own_steps": own, "total": total,
            "probes": warmup * n + n * (n + 1) // 2 - 1, "ratio": total / max(1, steps)}


# ---------------------------------------------------------------- schedule


class RaceSchedule:
    """Who trains next and who is dropped. Pure bookkeeping (no torch)."""

    def __init__(self, names: list[str], warmup: int):
        self.names = list(names)
        self.warmup = int(warmup)
        self.alive = list(range(len(names)))
        self.round = 1
        self.pending = list(self.alive)  # still to train and probe this round, in order
        self.scores: dict[int, float | None] = {}
        self.history: list[dict] = []  # per round: {"round", "scores": {name: score}, "steps": {name: the step the
        #                                score was probed at}, "dropped": name | None}
        self.steps: dict[int, int | None] = {}
        self.dropped: dict[int, int] = {}  # index -> round it was dropped after
        self.manual: set[int] = set()  # the ones dropped by hand (drop), not by a round's lowest score

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

    def record(self, idx: int, score: float | None, step: int | None = None) -> int | None:
        """The probe score of `idx` after its turn this round; returns the configuration dropped when this probe ended
        a round with a drop. A missing score (no face found) counts as the lowest; on a tie the later one goes."""
        if self.done or not self.pending or idx != self.pending[0]:
            raise ValueError(f"configuration {idx} is not the one training now")
        self.pending.pop(0)
        self.scores[idx] = score
        self.steps[idx] = step
        if self.pending:
            return None
        entry = {"round": self.round, "scores": {self.names[i]: self.scores.get(i) for i in self.alive},
                 "steps": {self.names[i]: self.steps.get(i) for i in self.alive}, "dropped": None}
        dropped = None
        if self.round > self.warmup:
            def key(i):
                s = self.scores.get(i)
                return (-math.inf if s is None else s, -i)

            dropped = min(self.alive, key=key)
            self.alive.remove(dropped)
            self.dropped[dropped] = self.round
            entry["dropped"] = self.names[dropped]
        self.history.append(entry)
        self.round += 1
        self.scores, self.steps = {}, {}
        self.pending = [] if self.done else list(self.alive)
        return dropped

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
            self.history.append({"round": self.round, "scores": {self.names[i]: self.scores.get(i) for i in self.alive},
                                 "steps": {self.names[i]: self.steps.get(i) for i in self.alive},
                                 "dropped": self.names[idx], "manual": True})
            self.round += 1
            self.scores, self.steps = {}, {}
            self.pending = list(self.alive)

    def state(self) -> dict:
        return {"names": self.names, "warmup": self.warmup, "round": self.round, "alive": [self.names[i] for i in
                self.alive], "dropped": {self.names[i]: r for i, r in self.dropped.items()},
                "manual": [self.names[i] for i in sorted(self.manual)],
                "history": self.history, "winner": self.names[self.alive[0]] if self.done else None}


# ---------------------------------------------------------------- one configuration's LoRA and state


class Variant:
    """One configuration of a race: its settings, its LoRA (parked on the CPU while another one trains) and the
    per-run state the trainer swaps in when it trains."""

    def __init__(self, idx: int, name: str, cfg, overrides: dict, names: list[str]):
        self.idx, self.name, self.cfg, self.overrides = idx, name, cfg, overrides
        self.names = list(names)  # its LoRA modules (a subset of the trainer's wrappers)
        self.rank, self.alpha, self.dropout = int(cfg.lora_rank), float(cfg.lora_alpha), float(cfg.lora_dropout)
        self.kind, self.factor = str(getattr(cfg, "network_type", "lora")), int(getattr(cfg, "lokr_factor", 8))
        self.weights: dict[str, dict[str, torch.Tensor]] = {}  # {module: {part: CPU fp32 tensor}} while parked
        self.started = False  # optimizer, scheduler and the rest are built on its first turn
        self.state: dict = {}  # the trainer's per-run state (optimizer, scheduler, ema, step, losses, ...)
        self.offloaded: list = []  # optimizer state tensors parked on the CPU: (param, key, device)
        self.alive = True
        self.master: dict = {}  # race_parallel: {module: view over the configuration's own Parameters} (build_master)

    @property
    def scale(self) -> float:
        return self.alpha / self.rank if self.kind == "lora" else 1.0

    def init_lora(self, wrappers: dict, seed: int) -> None:
        """Fresh adapter weights as LoRALinear.init_adapter makes them (the first factor kaiming-uniform, the second
        zero). Seeded per module name, so every configuration with the same shapes starts from the same weights."""
        for name in self.names:
            m = wrappers[name]
            g = torch.Generator().manual_seed((seed * 1_000_003 + zlib.crc32(name.encode())) % (2**63))
            shapes = m.adapter_shapes(self.kind, self.rank, self.factor)
            first, second = list(shapes)
            w = torch.empty(shapes[first])
            nn.init.kaiming_uniform_(w, a=math.sqrt(5), generator=g)
            self.weights[name] = {first: w, second: torch.zeros(shapes[second])}

    def view(self, use_average: bool = False) -> dict:
        """Module-like views of the parked weights (or of its weight average) for lora.save_lora."""
        ema = self.state.get("ema") if use_average else None
        avg = dict(zip(ema.keys, ema.shadow)) if ema is not None else {}
        out = {}
        for name in self.names:
            parts = {part: avg.get(f"{name}.{part}", t) for part, t in self.weights[name].items()}
            out[name] = SimpleNamespace(kind=self.kind, alpha=self.alpha, rank=self.rank, factor=self.factor,
                                        trainable=True, adapter_params=lambda parts=parts: parts)
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
            view = SimpleNamespace(kind=self.kind, alpha=self.alpha, rank=self.rank, factor=self.factor,
                                   trainable=True, multiplier=1.0, adapter_params=lambda parts=parts: parts)
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
        dev = m.weight.device
        if name in v.weights:
            if m.kind != v.kind:
                m.set_kind(v.kind)
            params = m.adapter_params()
            for part, t in v.weights[name].items():
                params[part].data = t.detach().to(dev)  # the same storage when already there (race_parallel)
            m.rank, m.alpha, m.factor, m.scale = v.rank, v.alpha, v.factor, v.scale
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
        dev = m.weight.device
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
    kinds = {"image"} | ({"face_crop"} if "face" in mode else set()) | ({"bust_crop"} if "bust" in mode else set())
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

    @property
    def running(self) -> bool:
        return not self.schedule.done


def write_report(path, schedule: RaceSchedule, variants: list[Variant], extra: dict | None = None) -> None:
    rows = [{"name": v.name, "settings": format_overrides(v.overrides), "dropped_after_round":
             schedule.dropped.get(v.idx)} for v in variants]
    path.write_text(json.dumps({**schedule.state(), "configs": rows, **(extra or {})}, indent=2), encoding="utf-8")
