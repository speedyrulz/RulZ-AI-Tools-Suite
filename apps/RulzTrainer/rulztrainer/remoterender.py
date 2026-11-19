"""Remote render: a run on a vast.ai instance hands its samples and likeness probes to a helper on the user's PC (the
helper renders them with this package's own render code, probeworker.Renderer) through round folders, keeps training,
and takes the pictures back. Contract: docs/remote-render-contract.md.

Training never waits on the helper longer than a round's deadline and never loses a round: a round the helper does not
bring back in time (slow, gone, failing) is taken back and rendered on the instance from the same LoRA state, which
the trainer keeps. Nothing here touches the model or the GPU; the trainer thread polls (a stat or two per step).

This module holds both sides' shared pieces (hashes, job (de)serialisation, the folder layout, atomic_rename) and the
instance side: RemoteRenderer."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

log = logging.getLogger("rulztrainer")

PROTOCOL = 1
PACKAGE = Path(__file__).resolve().parent
FRESH_SECONDS = 30.0  # a heartbeat (helper.json, the instance's own clock) this recent counts as a helper that is there
MIN_DEADLINE = 90.0  # seconds a round gets at least
DEADLINE_FACTOR = 3.0  # times the instance's own measured render time
PROBE_INTERVAL_FACTOR = 1.5  # a probe round gets this times the wall seconds between the last two probe steps
MISS_LIMIT = 2  # deadline misses / errors in a row that end delegation for the run
MAX_SAMPLE_ROUNDS = 2  # sample rounds out at once; a third renders on the instance
POLL_SECONDS = 0.5  # while training waits for a round
SUBDIRS = ("rounds", "claimed", "done", "failed", "abandoned")
# the config fields that name files a render loads (the text encoder is not one: the embeds come with the run)
FILE_FIELDS = ("transformer_file", "vae_file", "turbo_lora", "sample_lora", "sample_model_path")

# package-relative files whose bytes decide what a render looks like; the family's own folder is added by
# render_sources. trainer.py holds _render_round / SampleSlot, dataset.py text_cache_path (where the embeds are read)
RENDER_SOURCES = ["probeworker.py", "remoterender.py", "trainer.py", "archs.py", "sampler.py", "clips.py",
                  "dataset.py", "lora.py", "quant.py", "attention.py", "multigpu.py"]
FAMILY_DIRS = {"moss_image": "moss", "fibo2": "fibo", "krea2": "krea2", "qwen_image21": "qwen21",
               "flux3_klein": "flux3", "flux2_klein": "flux2", "flux2_klein4b": "flux2", "ideogram4": "ideogram4",
               "minimax_h3": "h3", "p_h3_image": "ph3", "minimax_h3_image": "h3image", "pan2": "pan2",
               "sdxl": "sdxl", "zimage": "zimage"}
# the config fields that change a render's pictures (the rest of the config is training, not rendering)
RECIPE_FIELDS = ("model_arch", "model_source", "model_path", "model_file_quant", "transformer_quant",
                 "sample_model_path", "sample_model_quant", "turbo_lora", "turbo_lora_strength", "sample_lora",
                 "sample_lora_strength", "lora_rank", "lora_alpha", "lora_targets", "lora_blocks", "network_type",
                 "lokr_factor", "oft_block_size", "oft_constraint", "dora", "train_text_embedder",
                 "attention_backend", "int8_matmul", "int8_backward", "nvfp4_matmul", "nvfp4_backward",
                 "max_token_length", "sample_width", "sample_height", "sample_steps", "sample_cfg", "sample_shift",
                 "sample_skip_layer", "sample_frames", "probe_frames", "probe_size", "probe_steps", "probe_cfg")


class RemoteRenderError(RuntimeError):
    """A finished round could not be taken (a missing or unsafe file)."""


def supported_host() -> bool:
    """The instance side runs on the Linux instance only: a run on the user's own Windows PC has no helper to find
    and would copy the text embeds for nothing."""
    return sys.platform != "win32"


# ------------------------------------------------------------ shared by both sides
def family_dir(family: str) -> str:
    return FAMILY_DIRS.get(family, family)


def render_sources(family: str) -> list[Path]:
    """The files render_hash covers: RENDER_SOURCES that exist, then the family's folder (every .py in it), sorted."""
    files = [PACKAGE / name for name in RENDER_SOURCES if (PACKAGE / name).is_file()]
    folder = PACKAGE / family_dir(family)
    if folder.is_dir():
        files += sorted(folder.rglob("*.py"), key=lambda p: p.relative_to(PACKAGE).as_posix())
    return files


def render_hash(family: str) -> str:
    """sha256 over render_sources(family): each file's package-relative name and bytes with CRLF turned into LF (a
    Windows checkout is CRLF, the instance's LF), the first 16 hex digits."""
    h = hashlib.sha256()
    for p in render_sources(family):
        h.update(p.relative_to(PACKAGE).as_posix().encode() + b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def recipe_hash(cfg_dict: dict, files: dict) -> str:
    """sha256 (first 16 hex digits) over the render-relevant config fields (RECIPE_FIELDS) and each file's (name, size):
    two sides with the same hash render the same recipe. files: {config field: {"name", "size", ...}}."""
    def pair(v):
        return [v.get("name"), v.get("size")] if isinstance(v, dict) else list(v)[:2]

    blob = {"cfg": {k: cfg_dict.get(k) for k in RECIPE_FIELDS},
            "files": {k: pair(v) for k, v in sorted((files or {}).items())}}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]


def safe_relative(name: str) -> str:
    """A path from a round (a job's out name) that stays inside the folder it is joined to; ValueError otherwise."""
    text = str(name).replace("\\", "/")
    p = Path(text)
    if not text or text.startswith("/") or p.is_absolute() or ".." in p.parts or (p.parts and p.parts[0].endswith(":")):
        raise ValueError(f"unsafe path {name!r}")
    return p.as_posix()


def job_to_dict(job, job_dir) -> dict:
    """A trainer.RenderJob as JSON data; out_path relative to job_dir with forward slashes."""
    d = dataclasses.asdict(job)
    d["out_path"] = Path(os.path.relpath(job.out_path, job_dir)).as_posix()
    return d


def job_from_dict(d: dict, base_dir):
    """The RenderJob of job_to_dict, its out_path rebased under base_dir (the PC: the helper's own work folder)."""
    from .trainer import RenderJob

    d = dict(d)
    rel = safe_relative(d.pop("out_path"))
    known = {f.name for f in dataclasses.fields(RenderJob)}
    return RenderJob(out_path=Path(base_dir) / rel, **{k: v for k, v in d.items() if k in known})


def state_to_bytes(state: dict) -> bytes:
    """A lora.lora_state as the bytes of a round's lora.safetensors (torch.save data, what Renderer.render's payload
    is: the file keeps the contract's name but is not safetensors; read it with state_from_bytes)."""
    import torch

    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def state_from_bytes(data: bytes) -> dict:
    import torch

    return torch.load(io.BytesIO(data), weights_only=True)


def remote_dir(job_dir) -> Path:
    return Path(job_dir) / "remote_render"


def round_name(seq: int, kind: str, step: int) -> str:
    return f"{seq:06d}_{kind}_{step}"


def round_path(root, state: str, name: str) -> Path:
    """root/<state>/<round name>; state is one of SUBDIRS."""
    return Path(root) / state / name


def write_json(path, data) -> None:
    """Atomic write (a reader sees the old file or the whole new one)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def atomic_rename(src, dst) -> bool:
    """Move src to dst (a folder moves whole); False when src is gone or dst exists. Of two renames of one source
    only one succeeds: that is how the trainer and the helper settle who has a round."""
    src, dst = Path(src), Path(dst)
    if not src.exists() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dst)
    except OSError:
        return False
    return True


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


# ------------------------------------------------------------ the instance side
@dataclasses.dataclass
class Round:
    """One round handed to the helper: its jobs and the LoRA state the pictures are of (kept for the instance's own
    render when the round comes back late)."""
    seq: int
    kind: str  # probe | sample
    step: int
    name: str
    jobs: list
    state: dict
    negative_kind: str
    submitted: float
    deadline: float  # absolute (the instance's clock)
    shadow: bool = False
    why: str = ""  # why check() said "late"
    span: float = 0.0  # seconds the helper gets once it has the round (the deadline measured from its claim)
    cap: float = 0.0  # absolute: a round waiting behind another is never held past 2 x span from submit
    claimed_at: float | None = None
    behind: bool = False  # it waited unclaimed while the helper (one worker) rendered another round of this run
    uncounted: bool = False  # lost without ever being claimed, behind another round: not the helper's slowness


@dataclasses.dataclass
class Taken:
    """The pictures of a finished round, in their jobs' real out_paths."""
    images: list
    seconds: float
    host: str = ""
    gpu: str = ""

    @property
    def label(self) -> str:
        return f"{self.host} {self.gpu}".strip() or "helper"


class RemoteRenderer:
    """The trainer's side: publishes the run, watches the helper's heartbeat and ready file, submits rounds, and takes
    their pictures back (or reclaims a round the helper did not bring in time; the trainer then renders it itself)."""

    def __init__(self, job_dir, family: str, clock=time.time, sleep=time.sleep):
        self.job_dir = Path(job_dir)
        self.root = remote_dir(job_dir)
        self.family = family
        self.clock, self.sleep = clock, sleep
        self.run_token = self.recipe = self.render = ""
        self.helper: dict | None = None  # the last helper.json
        self.ready: dict | None = None  # ready.json once it matches this run
        self._mtimes: dict[str, tuple] = {}
        self._announced = False  # the log last said a helper is there
        self.per_job: dict[str, float] = {}  # the instance's own render seconds per job, per kind
        self.seq = 0
        self.rounds: dict[str, Round] = {}  # rounds handed over and not yet resolved
        self.misses = 0  # deadline misses / errors in a row
        self.off_reason = ""  # why delegation ended for the run
        self.probes_off = ""  # why probes (only) stay on the instance: the shadow round disagreed
        self.shadow_done = False  # the shadow round was compared
        self.first_round = True  # the next round after the helper became ready gets the doubled deadline
        self.probe_times: list[float] = []  # wall clock at the last two probe steps reached (note_probe_step)
        self.last_label = "instance"  # where the probes last rendered
        self.stats = {"rounds": 0, "helper_seconds": 0.0, "taken_back": 0, "waited": 0.0}

    # -------------------------------------------------------- the run
    def start_run(self, job: str, cfg_dict: dict, files: dict, negatives: dict, embeds: dict) -> str:
        """Clear remote_render/ (a resume or a crashed run leaves nothing a helper could render into a dead run) and
        publish the run: prepare/ first (prepare.json, the text embeds: {prompt text: cache file}), then run.json,
        which is what a helper finds a run by. Returns the run token."""
        shutil.rmtree(self.root, ignore_errors=True)
        for sub in (Path("prepare") / "embeds", *map(Path, SUBDIRS)):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.run_token = uuid.uuid4().hex
        self.render = render_hash(self.family)
        self.recipe = recipe_hash(cfg_dict, files)
        mapped = {}
        for text, src in embeds.items():
            key = hashlib.sha1(text.encode("utf-8")).hexdigest()
            _link_or_copy(Path(src), self.root / "prepare" / "embeds" / f"{key}.safetensors")
            mapped[text] = f"embeds/{key}.safetensors"
        write_json(self.root / "prepare" / "prepare.json",
                   {"protocol": PROTOCOL, "run_token": self.run_token, "recipe_hash": self.recipe,
                    "render_hash": self.render, "family": self.family, "cfg": cfg_dict, "files": files,
                    "negatives": negatives, "embeds": mapped})
        write_json(self.root / "run.json",
                   {"protocol": PROTOCOL, "run_token": self.run_token, "job": job, "family": self.family,
                    "render_hash": self.render, "recipe_hash": self.recipe, "created": self.clock()})
        return self.run_token

    # -------------------------------------------------------- the helper
    def _changed(self, path: Path) -> bool:
        """The file is new or rewritten since the last look (False too when it is not there)."""
        try:
            st = path.stat()
        except OSError:
            self._mtimes.pop(str(path), None)
            return False
        key = (st.st_mtime_ns, st.st_size)
        if self._mtimes.get(str(path)) == key:
            return False
        self._mtimes[str(path)] = key
        return True

    def helper_fresh(self) -> bool:
        h = self.helper
        return bool(h and h.get("run_token") == self.run_token
                    and abs(self.clock() - float(h.get("time") or 0.0)) <= FRESH_SECONDS)

    @property
    def label(self) -> str:
        r = self.ready or {}
        return f"{r.get('host', '')} {r.get('gpu', '')}".strip() or "helper"

    def poll(self) -> None:
        """One small look per step: helper.json and the ready / not-ready files, each read only when it changed.
        Logs when a helper becomes usable and when it goes quiet. A heartbeat that looks stale is read again even when
        the file's time and size did not change: a file system or copy with coarse file times can rewrite it within one
        tick, at the same size."""
        if self._changed(self.root / "helper.json") or (self.helper is not None and not self.helper_fresh()):
            self.helper = read_json(self.root / "helper.json")
        elif not (self.root / "helper.json").exists():
            self.helper = None
        ready_path = self.root / "prepare" / "ready.json"
        if self._changed(ready_path):
            self._read_ready(read_json(ready_path))
        elif not ready_path.exists():
            self.ready = None
        nr = self.root / "prepare" / "not_ready.json"
        if self._changed(nr):
            d = read_json(nr) or {}
            if d.get("run_token") == self.run_token:
                log.warning("Remote render: the helper cannot render this run: %s", d.get("reason", "no reason given"))
        usable = self.ready is not None and self.helper_fresh() and not self.off_reason
        if usable and not self._announced:
            r = self.ready
            log.info("Remote render: helper %s is ready (model loaded in %.0fs, %.0f GB, renders %s); %s",
                     self.label, float(r.get("load_seconds") or 0.0), float(r.get("vram_gb") or 0.0),
                     " and ".join(r.get("kinds") or []), "samples and probes go there now")
        elif self._announced and not usable and self.ready is not None and not self.off_reason:
            log.warning("Remote render: the helper %s has gone quiet (no heartbeat for %.0fs); renders stay on the "
                        "instance until it is back", self.label,
                        self.clock() - float((self.helper or {}).get("time") or 0.0))
        self._announced = usable

    def _read_ready(self, d) -> None:
        d = d if isinstance(d, dict) else {}
        want = (self.run_token, self.recipe, self.render)
        got = (d.get("run_token"), d.get("recipe_hash"), d.get("render_hash"))
        if got == want:
            if self.ready is None or self.ready.get("time") != d.get("time"):
                self.first_round = True  # (a restarted helper reloaded its model: its first round warms up again)
            self.ready = d
            return
        self.ready = None
        if d:
            what = ("another run" if got[0] != want[0] else "a different recipe (files or settings)"
                    if got[1] != want[1] else "another version of the render code")
            log.warning("Remote render: the helper's ready file is for %s; nothing is sent to it", what)

    def can_delegate(self, kind: str) -> bool:
        """A round of `kind` may go to the helper: it is ready for this run's recipe and code, its heartbeat is fresh,
        it renders that kind, delegation is not ended, and the instance has timed that kind (for the deadline)."""
        if self.off_reason or self.ready is None or not self.helper_fresh():
            return False
        if kind not in (self.ready.get("kinds") or []) or kind not in self.per_job:
            return False
        return not (kind == "probe" and self.probes_off)

    def note_own(self, kind: str, seconds: float, jobs: int) -> None:
        """The instance rendered `jobs` pictures of `kind` in `seconds`: what a round's deadline is a multiple of."""
        if jobs > 0 and seconds > 0:
            self.per_job[kind] = seconds / jobs

    def deadline_for(self, kind: str, jobs: int) -> float:
        """Seconds a round gets: max(90, 3 x the instance's own render time for that many pictures); the first round
        after the helper became ready gets twice that (its first renders compile and warm up)."""
        base = max(MIN_DEADLINE, DEADLINE_FACTOR * self.per_job.get(kind, 0.0) * max(1, jobs))
        return base * (2 if self.first_round else 1)

    def note_probe_step(self) -> None:
        """A probe step was reached now: remember the wall clock (the last two) so the interval between probes is known.
        The first one is step 0, so the first interval runs from step 0 to the first probe that may be delegated."""
        self.probe_times = (self.probe_times + [self.clock()])[-2:]

    def probe_interval(self) -> float:
        """Wall seconds the trainer took between its two most recent probe steps; 0.0 when fewer than two are known."""
        return self.probe_times[1] - self.probe_times[0] if len(self.probe_times) == 2 else 0.0

    def probe_deadline(self, jobs: int) -> tuple[float, str]:
        """(seconds, why) a probe round handed to the helper gets: the larger of the old rule (deadline_for) and
        PROBE_INTERVAL_FACTOR x the probe interval, so it has until about the next probe is due. With no interval known
        yet it is the old rule alone. (Sample rounds always use the old rule.)"""
        old = self.deadline_for("probe", jobs)
        wanted = PROBE_INTERVAL_FACTOR * self.probe_interval()
        if wanted > old:
            return wanted, f"until the next probe, about {wanted:.0f}s ({self.probe_interval():.0f}s between probes)"
        return old, f"3 x the instance's own render time, at least {MIN_DEADLINE:.0f}s"

    def switch(self, label: str) -> None:
        """Probes now render on `label`: a line in the log when that changed."""
        if label != self.last_label:
            log.info("Remote render: probes now render on %s (they rendered on %s before)", label, self.last_label)
            self.last_label = label

    # -------------------------------------------------------- rounds
    def submit(self, kind: str, step: int, jobs: list, state: dict, negative_kind: str | None = None,
               shadow: bool = False, deadline_s: float | None = None) -> Round:
        """Hand the helper a round: round.json and the LoRA state written under rounds/.tmp-<seq>, then renamed to
        rounds/<seq>_<kind>_<step> (a helper never sees half a round). deadline_s: the seconds the round gets, when the
        caller worked them out (probe_deadline); None is the old rule (deadline_for). The first-round doubling is part
        of the old rule only, and probe_deadline already includes it where it applies."""
        self.seq += 1
        name = round_name(self.seq, kind, step)
        if deadline_s is None:
            deadline_s = self.deadline_for(kind, len(jobs))
        negative_kind = negative_kind or kind
        tmp = self.root / "rounds" / f".tmp-{self.seq}"
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "lora.safetensors").write_bytes(state_to_bytes(state))
        write_json(tmp / "round.json", {"protocol": PROTOCOL, "run_token": self.run_token, "seq": self.seq,
                                        "kind": kind, "step": step, "negative_kind": negative_kind,
                                        "jobs": [job_to_dict(j, self.job_dir) for j in jobs],
                                        "deadline_s": deadline_s})
        if not atomic_rename(tmp, round_path(self.root, "rounds", name)):
            raise RemoteRenderError(f"could not publish round {name}")
        self.first_round = False
        now = self.clock()
        r = Round(self.seq, kind, step, name, jobs, state, negative_kind, now, now + deadline_s, shadow,
                  span=deadline_s, cap=now + 2 * deadline_s)
        self.rounds[name] = r
        return r

    def check(self, r: Round) -> str:
        """open (the helper has until the deadline), done, failed, or late (the deadline passed or the helper's
        heartbeat is stale: take it back)."""
        self.poll()
        if round_path(self.root, "done", r.name).is_dir():
            return "done"
        if round_path(self.root, "failed", r.name).is_dir():
            return "failed"
        now = self.clock()
        if r.claimed_at is None:
            if round_path(self.root, "claimed", r.name).is_dir():
                r.claimed_at = now
                if r.behind:  # the helper's time starts at its claim (capped: training is never held for long)
                    r.deadline = min(r.cap, now + r.span)
            elif not r.behind and any(round_path(self.root, "claimed", o.name).is_dir()
                                      for o in self.rounds.values() if o is not r):
                r.behind = True  # the helper has one worker and is busy with another round: this one queues
                r.deadline = r.cap
        if now > r.deadline:
            r.why = f"no pictures within {r.deadline - r.submitted:.0f}s"
            r.uncounted = r.behind and r.claimed_at is None
            return "late"
        if not self.helper_fresh():
            r.why = "the helper stopped answering"
            return "late"
        return "open"

    def reclaim(self, r: Round) -> str:
        """Take a round back: rounds/ or claimed/ -> abandoned/. Whichever rename succeeds wins, so when the helper's
        done/ (or failed/) rename won, that is the answer: "done", "failed"; else "abandoned"."""
        for _ in range(3):
            for state in ("rounds", "claimed"):
                if atomic_rename(round_path(self.root, state, r.name), round_path(self.root, "abandoned", r.name)):
                    return "abandoned"
            if round_path(self.root, "done", r.name).is_dir():
                return "done"
            if round_path(self.root, "failed", r.name).is_dir():
                return "failed"
        return "abandoned"

    def take(self, r: Round) -> Taken:
        """Move a finished round's pictures (done/<round>/out/*) to their jobs' real out_paths, a clip's files before
        its still, and delete done/<round>. RemoteRenderError when something is missing or unsafe (the folder stays)."""
        d = round_path(self.root, "done", r.name)
        info = read_json(d / "done.json")
        if not isinstance(info, dict):
            raise RemoteRenderError("done.json is missing or unreadable")
        out = d / "out"
        names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) if out.is_dir() else []
        stills = {Path(os.path.relpath(j.out_path, self.job_dir)).as_posix() for j in r.jobs}
        absent = [n for n in list(info.get("files") or []) + sorted(stills) if n not in names]
        if absent:
            raise RemoteRenderError(f"the helper's round lacks {', '.join(absent[:3])}")
        base = self.job_dir.resolve()
        try:
            for n in sorted(names, key=lambda n: (n in stills, n)):  # a clip's extra files first, stills last
                dst = (self.job_dir / safe_relative(n))
                if base not in dst.resolve().parents:
                    raise ValueError(f"unsafe path {n!r}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(out / n, dst)
        except ValueError as e:
            raise RemoteRenderError(str(e)) from e
        shutil.rmtree(d, ignore_errors=True)
        return Taken([str(j.out_path) for j in r.jobs], float(info.get("seconds") or 0.0),
                     str(info.get("host") or ""), str(info.get("gpu") or ""))

    def collect(self, r: Round, wait: bool = False, give_up: str = "", max_wait: float | None = None):
        """(Taken, "") when the round's pictures are in and moved; (None, "open") while it waits (wait: blocks, polling,
        until it is not open: done, failed, or late); (None, why) when the round is lost and the caller must render
        it itself (the round is then abandoned and a miss counted). give_up: a round still open is not waited for but
        taken back at once, with this text as the reason (a probe still out when the next one is due). max_wait: with
        wait, the longest the wait lasts (seconds on this renderer's clock); then the round is taken back."""
        t0 = time.time()
        c0 = self.clock()
        state = self.check(r)
        if state == "open" and give_up:
            r.why, r.uncounted, state = give_up, False, "late"
        while wait and state == "open":
            if max_wait is not None and self.clock() - c0 >= max_wait:
                r.why, r.uncounted, state = f"no pictures within {max_wait:.0f}s of waiting for them", False, "late"
                break
            self.sleep(POLL_SECONDS)
            state = self.check(r)
        if wait:
            self.stats["waited"] += time.time() - t0
        if state == "open":
            return None, "open"
        why = r.why
        if state == "failed":
            err = read_json(round_path(self.root, "failed", r.name) / "error.json") or {}
            why = f"the helper failed it: {str(err.get('error') or 'no reason given').strip().splitlines()[-1]}"
        elif state == "late":
            state = self.reclaim(r)
            if state == "failed":
                why = "the helper failed it"
        if state == "done":
            try:
                got = self.take(r)
            except (RemoteRenderError, OSError, ValueError) as e:
                state, why = "error", f"its pictures could not be taken: {e}"
            else:
                self.misses = 0
                self.rounds.pop(r.name, None)
                self.stats["rounds"] += 1
                self.stats["helper_seconds"] += got.seconds
                return got, ""
        self.rounds.pop(r.name, None)
        if r.uncounted:  # (it queued behind a round the helper was rendering: slow, not failing)
            self.stats["taken_back"] += 1
            log.info("Remote render: round %s was never started (the helper was busy with another round); "
                     "rendering it here", r.name)
        else:
            self._miss(r, why or "no reason given")
        shutil.rmtree(round_path(self.root, "abandoned", r.name), ignore_errors=True)
        shutil.rmtree(round_path(self.root, "failed", r.name), ignore_errors=True)
        shutil.rmtree(round_path(self.root, "done", r.name), ignore_errors=True)
        return None, why or "no reason given"

    def _miss(self, r: Round, why: str) -> None:
        self.misses += 1
        self.stats["taken_back"] += 1
        if self.misses >= MISS_LIMIT and not self.off_reason:
            self.off_reason = f"{self.misses} rounds in a row did not come back ({why})"
            log.warning("Remote render: %s; everything renders on the instance for the rest of the run", self.off_reason)

    def stop_probes(self, reason: str) -> None:
        self.probes_off = reason

    def summary(self) -> str:
        s = self.stats
        return (f"{s['rounds']} round(s) rendered on the helper in {s['helper_seconds']:.0f}s, {s['taken_back']} taken "
                f"back and rendered here; training waited {s['waited']:.0f}s for the helper")
