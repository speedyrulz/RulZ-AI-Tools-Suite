"""Check run: one command that queues a set of short real-model training runs, each switching on one feature that has
never run on a real model, and a second one that reads the finished job folders and writes a pass / fail table.

    python -m rulztrainer.checkrun --list                              # the cases, and what each one needs
    python -m rulztrainer.checkrun queue --config base.yaml --dry-run  # the plan: jobs and changed settings, nothing queued
    python -m rulztrainer.checkrun queue --config base.yaml            # into the WebUI's queue (output/_queue.json)
    python -m rulztrainer.checkrun queue --config base.yaml --fleet --vram 24   # into the vast.ai fleet's queue
    python -m rulztrainer.checkrun report                              # the table of the newest check run

The base config is a config that already trains on the machine the runs go to (its model, dataset and output
folder). Every case is that config made short and plain (plain_settings: a few dozen steps, one sample at the end, no
probes, no race, no blocks keeping their activations, so two runs' peak VRAM can be compared, and no restart after
running out of memory, so a run passes on the settings it was planned with) plus the settings that switch its one
feature on. `baseline` is the short plain run itself: the others' peak VRAM and step time are shown
beside it, and when it fails the base config is what to fix first.

A case passes on what its run leaves in the job folder, nothing else: status.json (the run reached its last step, the
loss is a number, the peak VRAM), train.log (the line the feature logs), the LoRA and the sample, likeness.jsonl where
the case probes, and the files the feature writes. The queue step writes a manifest (output/_checkrun/<tag>/
manifest.json: every case's job name, settings and what counts as a pass), so the report needs only that file and the
folders, also on another computer and with a later version of this catalogue.

This module starts nothing itself: the WebUI's queue (or the fleet's) starts the runs, one after another. The check
run was built and tested with made-up job folders only; it has not been run on a real model yet."""
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "output" / "_checkrun"
DEFAULT_STEPS = 30
TAG_RE = re.compile(r"[A-Za-z0-9_-]{1,24}")
# what a fleet run's files are fetched as (scripts/vast/rulz_fleet.py): the logs, settings and records the checks
# read while it runs, the pictures at its end, no LoRA files (status.json lists the ones the run saved)
FLEET_DOWNLOAD = {"logs": "during", "probes": "end", "samples": "end", "maps": "off", "best": "off", "loras": "off"}
BLOCK_SWAP_LINE = r"Block swap: the frozen weights of blocks 0-\d+ \((\d+) of \d+\) stay in RAM, ([\d.]+) GB pinned"
# log lines quoted in the notes of a case that did not pass: why a scorer, a render or the run gave up
EXPLAIN = (r"Could not initialise \w+ scoring \(.*", r"No usable .* in any body reference image.*",
           r"No faces found in any face reference image.*", r"\[oom recovery\] The run ran out of .*",
           r"Run report not written \(.*", r"torch\.compile failed \(.*")
PICTURES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm")


# ---------------------------------------------------------------- the catalogue


@dataclass(frozen=True)
class Case:
    """One short run. overrides: the TrainConfig settings that switch the feature on (a value of "<...>" is filled
    in when the run is planned: PLANNED below). expect: what counts as a pass besides the checks every run gets
    (evaluate), as plain data so it travels in the manifest."""
    name: str
    feature: str  # what has never run on a real model, in a line
    overrides: dict = field(default_factory=dict)
    families: tuple = ()  # the model families it runs on; () = any
    steps: int = 0  # 0 = the check run's (DEFAULT_STEPS, --steps)
    needs: tuple = ()  # settings the planned config must have: reg_dir, face_refs, body_refs, refs (either), vram
    whole_model: bool = False  # only where the card holds the model without block swap (the case turns it off)
    probes: str = ""  # likeness probes: "" = off, face, body, or both (whichever the config has references for)
    expect: dict = field(default_factory=dict)
    compare: str = ""  # a case whose peak VRAM and step time are shown beside this one's
    note: str = ""  # what to know before reading its verdict


# values filled in when a run is planned
PLANNED = {
    "<reg_dir>": "the regularisation folder (--reg-dir, else the base config's reg_dir)",
    "<plan>": "the block swap the wizard's memory plan gives this adapter on a card of --vram GB",
    "<plan, at least 4>": "the wizard's block swap for a card of --vram GB, the base config's if that is more, and "
                          "at least 4 blocks",
    "<steps>": "the run's own step count",
}

CASES: tuple[Case, ...] = (
    Case("baseline", "The base config, short and plain: the reference the other runs' peak VRAM and step time are "
                     "shown beside, and the proof that the base config itself trains"),
    Case("reg_dir", "Regularisation dataset (reg_dir, 3.20.0)",
         overrides={"reg_dir": "<reg_dir>", "reg_weight": 1.0, "reg_per_epoch": -1, "isolation": False},
         needs=("reg_dir",),
         expect={"log": [r"Regularisation dataset: \d+ pictures in ",
                         r"Regularisation: an epoch trains \d+ subject and \d+ regularisation pictures"],
                 "status": [["reg_images", ">", 0], ["reg_per_epoch", ">", 0]]},
         compare="baseline"),
    Case("block_swap_rank64", "The adapter-aware block swap plan (runplan.adapter_extra_gb, 3.20.0) on a rank-64 LoRA",
         overrides={"network_type": "lora", "lora_rank": 64, "lora_alpha": 64.0, "dora": False, "multi_gpu": "off",
                    "block_swap": "<plan>"},
         needs=("vram",),
         expect={"log": [r"\d+ modules, [\d.]+M trainable params"], "peak_within_vram": True,
                 "adapter_memory": {"against": "baseline", "tolerance_gb": 1.0}},
         compare="baseline",
         note="The plan counts what the adapter, its gradients, optimizer state and weight average hold beyond the "
              "baseline's; the run measures it as the difference in peak VRAM, with the swapped blocks' weights "
              "that are off the GPU added back on both sides. Both peaks are the whole process's, loading included."),
    Case("block_swap_lokr", "The adapter-aware block swap plan on a LoKr (factor 8)",
         overrides={"network_type": "lokr", "lokr_factor": 8, "dora": False, "multi_gpu": "off",
                    "block_swap": "<plan>"},
         needs=("vram",),
         expect={"log": [r"\d+ modules, [\d.]+M trainable params \(LoKr"], "peak_within_vram": True,
                 "adapter_memory": {"against": "baseline", "tolerance_gb": 1.0}},
         compare="baseline",
         note="As block_swap_rank64. A LoKr also builds each layer's weight change in the forward pass, which the "
              "plan does not count: a measured excess here says how much that is."),
    Case("block_swap_compile", "torch.compile with block swap (block_swap_compile)",
         overrides={"block_swap": "<plan, at least 4>", "block_swap_compile": True, "compile_model": True,
                    "multi_gpu": "off"},
         steps=40,
         expect={"log": [BLOCK_SWAP_LINE, r"block_swap_compile is experimental and untested",
                         r"Compiled \d+ DiT blocks with torch\.compile", r"Compile stats \(end of training\)"],
                 "unexercised": [[r"triton is not installed; running without compile",
                                  "torch.compile needs triton, which that machine does not have"],
                                 [r"torch\.compile off \((.*)\)", "torch.compile was turned off: {0}"]]},
         compare="baseline",
         note="The first steps compile and are slow; read the step time beside the baseline's with that in mind, and "
              "the compile report in train.log for recompiles."),
    Case("activation_offload", "Activation offloading (activation_offload)",
         overrides={"activation_offload": True, "gradient_checkpointing": True},
         expect={"log": [r"activation_offload is experimental and untested"],
                 "numbers": [{"name": "GB of block inputs sent to RAM in a step",
                              "pattern": r"Activation offload: ([\d.]+) GB of block inputs sent to RAM in step",
                              "op": ">", "value": 0}]},
         compare="baseline",
         note="It should lower the peak VRAM against the baseline at some cost in step time: both are in the notes."),
    Case("normuon", "The NorMuon optimizer (optimizer: normuon)",
         overrides={"optimizer": "normuon"},
         expect={"log": [r"normuon is experimental and untested in RulZ AI Trainer"]},
         compare="baseline"),
    Case("oft_scaled_lr", "Scaled OFT learning rate (oft_scaled_lr) on OFT blocks of 64",
         overrides={"network_type": "oft", "oft_block_size": 64, "oft_scaled_lr": True, "dora": False,
                    "optimizer": "adamw"},
         expect={"log": [r"Scaled OFT \(experimental\): block \d+ lr "],
                 "unexercised": [[r"oft_scaled_lr is ignored with (\S+)",
                                  "oft_scaled_lr is ignored with the optimizer {0}"]]},
         note="The optimizer is set to adamw for this run: the self-tuning optimizers and hull ignore the setting."),
    Case("oom_recovery", "Out-of-memory recovery of a whole run (oom_retries), which the WebUI carries out",
         overrides={"gradient_checkpointing": False, "activation_offload": False, "block_swap": 0, "oom_retries": 3,
                    "save_state": True, "save_state_every": "<steps>"},
         expect={"recovery": {"min_tries": 1,
                              "else": "the run never ran out of memory on that card, so nothing was recovered: "
                                      "force it with a larger size (--set resolutions=[2.0]) or batch (--set "
                                      "batch_size=4), or run this case on a smaller card"},
                 "files": ["train_oom*.log"], "log": [r"\[oom recovery\] try \d+ of \d+"]},
         whole_model=True,
         note="The run starts without gradient checkpointing and without block swap so that it runs out of GPU "
              "memory; the WebUI should then start it again one step lighter (gradient checkpointing on). Whether "
              "the first try runs out depends on the card. Only the WebUI's queue recovers a run, so it must stay "
              "open."),
    Case("face_antelopev2", "Likeness probes scored with the antelopev2 face pack (face_model, 3.19.0)",
         overrides={"face_model": "antelopev2"}, needs=("face_refs",), probes="face",
         expect={"probe": {"last": {"face_model": "antelopev2"}, "scored": ["face"]}}),
    Case("body_smplx_ratio", "The smplx_ratio body score (body_score, 3.19.0); needs NLF's model in models/nlf",
         overrides={"body_score": "smplx_ratio"}, needs=("body_refs",), probes="body",
         expect={"log": [r"NLF loaded from ", r"Body references \(smplx_ratio\): \d+ of \d+ measurable"],
                 "probe": {"last": {"body_score": "smplx_ratio"}, "scored": ["body"]}}),
    Case("relay_race_pick", "A relay race that waits for a pick each round (race_follow with race_pick, 3.12.0)",
         overrides={"race": True, "race_follow": True, "race_pick": True, "race_pick_minutes": 1.0,
                    "race_parallel": False, "race_warmup_probes": 0,
                    "race_configs": [{"name": "a", "seed": 101}, {"name": "b", "seed": 202}]},
         steps=20, needs=("refs",), probes="both",
         expect={"race": True,
                 "log": [r"Relay race: you pick the leader after each round on the Monitor",
                         r"Relay race: round 1 is over; pick the racer everyone continues from",
                         r"Relay race: everyone continues from \S+'s LoRA", r"Relay race over: \S+ won the last round"],
                 "files": ["race.json", "{job}_winner.yaml"]},
         note="Two seed rows, two rounds. Each round waits a minute for a pick on the Monitor and then lets the "
              "highest score lead, so the run also finishes unattended; pick one by hand to try the pick itself."),
    Case("h3_image_likeness", "Fizgig's MiniMax-H3-Image likeness recipe (the Train tab's H3 Image likeness button, "
                              "3.15.0)",
         overrides={"clean_share": "16.67,75,33.33,75,50,75", "optimizer": "automagic4", "learning_rate": 2.5e-4,
                    "automagic_max_lr": 2.5e-4, "resolutions": [0.25], "timestep_sampling": "reference"},
         families=("minimax_h3_image",),
         expect={"log": [r"clean_share on .*: the share is the steps at its last noise level"]},
         note="A short run sees the first epoch's share only; it shows that the recipe trains, not how well."),
    Case("zimage_turbo_lora", "A rank-16 LoRA on Z-Image Turbo under Fizgig's training adapter, the first run of the "
                              "Z-Image family on real weights (training_adapter auto, 8-step sample at CFG 1)",
         overrides={"network_type": "lora", "lora_rank": 16, "lora_alpha": 16.0, "dora": False, "finetune": False,
                    "training_adapter": "auto"},
         families=("zimage",),
         note="Needs a Z-Image Turbo base config (Tongyi-MAI/Z-Image-Turbo or Comfy-Org's z_image_turbo files). The "
              "run passes on the checks every run gets (it trains, saves the LoRA and renders the 8-step sample); "
              "read train.log for the training adapter's 'matched N modules' line and the sample for whether Turbo "
              "still renders at 8 steps with the adapter off."),
    Case("run_report", "The run report written when a training ends (run_report, 3.5.0); needs matplotlib",
         overrides={"run_report": True},
         expect={"log": [r"Run report: .*_report\.(png|pdf)"]}),
)


def case_named(name: str) -> Case:
    for case in CASES:
        if case.name == name:
            return case
    raise ValueError(f"No case named {name!r}: {', '.join(c.name for c in CASES)}")


# ---------------------------------------------------------------- planning


def plain_settings(base: dict, steps: int) -> dict:
    """What every run of the check has over the base config: short, one save and one sample at the end, and nothing
    that probes, races, resumes, stretches or stops the run, or renders or trains beside it (training maps, sweeps,
    a regularisation folder, LoRA Isolation), so each case adds one feature to the same plain run.
    checkpoint_keep_blocks 0: its default fills the free GPU memory with kept activations, which would put every
    run's peak VRAM at the card's size. oom_retries 0: a run that runs out of memory would otherwise be started
    again with more block swap by the WebUI and pass on settings the case never planned."""
    prompts = [p for p in (base.get("sample_prompts") or []) if str(p).strip()][:1]
    if not prompts:
        prompts = [f"a photo of {str(base.get('trigger_word') or '').strip() or 'a person'}"]
    return {
        "steps": int(steps), "warmup_steps": min(int(base.get("warmup_steps") or 0), max(0, int(steps) // 5)),
        "save_every": 0, "save_state_every": 0, "sample_every": 0, "sample_before_training": False,
        "sample_prompts": prompts, "checkpoint_keep_blocks": 0, "compile_model": False,
        "probe_every": 0, "probe_checks": [], "likeness_patience": 0, "likeness_best_min_kept": 0.0,
        "race_drop_drift": 0.0, "resolution_step_up": False, "race": False, "race_configs": [],
        "reward_steps": 0, "selftrain": False, "soup_files": [], "noise_files": [], "resume_from": "",
        "resume_state": "", "finetune": False, "run_report": False, "recache": False, "oom_retries": 0,
        "training_maps": False, "region_sweep_every": 0, "holdout_images": 0, "holdout_track_best": False,
        "holdout_patience": 0, "region_steer": False, "reg_dir": "", "isolation": False,
    }


def probe_settings(data: dict, steps: int, which: str) -> dict:
    """Likeness probes for a case that needs them: one at the start, one half-way and the final LoRA's, a seed each.
    which: face, body, or both (the ones the config has references for). A probe prompt the base config lacks is the
    identity preset's."""
    from .config import PRESET_PROBE_PROMPTS

    face_p, body_p = PRESET_PROBE_PROMPTS["identity"]
    subject = str(data.get("trigger_word") or "").strip() or "the character"
    face = which == "face" or (which == "both" and bool(data.get("face_refs")))
    body = which == "body" or (which == "both" and bool(data.get("body_refs")))
    return {
        "probe_every": max(1, int(steps) // 2), "probe_at_start": True, "probe_seeds": 1, "probe_floor_seeds": 1,
        "probe_async": False, "sample_async": False, "maps_async": False,
        "sweep_async": False,
        "probe_face_prompt": (data.get("probe_face_prompt") or face_p.format(trigger=subject)) if face else "",
        "probe_body_prompt": (data.get("probe_body_prompt") or body_p.format(trigger=subject)) if body else "",
    }


def wizard_block_swap(cfg, vram_gb: float) -> int:
    """The block swap the WebUI wizard's memory plan gives this config on one card of vram_gb (it counts the
    adapter: runplan.adapter_extra_gb). webui.py holds the plan and its tables of model sizes, so it is loaded here
    (it starts nothing when imported)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import webui

    picks = webui.wizard_recommendations(cfg, None, [{"total": float(vram_gb) * 1024}], None)
    return max(0, int(picks.get("block_swap", (0,))[0] or 0))


def adapter_gb(cfg) -> float | None:
    """GB the config's adapter holds with its gradients, optimizer state and weight average (runplan.adapter_gb), or
    None when it cannot be counted."""
    try:
        from . import runplan

        got = float(runplan.adapter_gb(cfg))
        return round(got, 3) if got > 0 else None
    except Exception:  # noqa: BLE001  (no torch here, an unknown family): no estimate
        return None


def detect_vram(device: str = "cuda:0") -> float | None:
    """The card's memory in GB as nvidia-smi reports it (nothing is loaded on the GPU), or None."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20)
        rows = [float(line) for line in done.stdout.split("\n") if line.strip()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if done.returncode != 0 or not rows:
        return None
    index = int(m.group(1)) if (m := re.fullmatch(r"cuda:(\d+)", str(device).strip())) else 0
    return round(rows[index if index < len(rows) else 0] / 1024, 1)


def parse_sets(items) -> dict:
    """KEY=VALUE pairs, the value JSON when it parses (as rulztrainer.train --set)."""
    out = {}
    for item in items or []:
        key, sep, value = str(item).partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set takes KEY=VALUE, got {item!r}")
        try:
            out[key.strip()] = json.loads(value)
        except ValueError:
            out[key.strip()] = value
    return out


def local_path(path: str) -> Path:
    """A config's path as the WebUI reads it on this computer: relative ones from the trainer's folder."""
    p = Path(str(path))
    return p if p.is_absolute() else ROOT / p


def select(only=(), skip=()) -> list[Case]:
    """The cases to run, in the catalogue's order. A case whose result is read beside another's brings that one
    along (the baseline), unless it was skipped by name."""
    only, skip = [n for n in only if n], [n for n in skip if n]
    for name in (*only, *skip):
        case_named(name)  # raises on a name that is no case
    picked = [c for c in CASES if (not only or c.name in only) and c.name not in skip]
    names = {c.name for c in picked}
    extra = {c.compare for c in picked if c.compare and c.compare not in names and c.compare not in skip}
    return [c for c in CASES if c.name in names | extra]


def plan(base: dict, cases=None, tag: str = "", steps: int = DEFAULT_STEPS, vram: float | None = None,
         reg_dir: str = "", fleet: bool = False, busy=(), planner=None) -> dict:
    """The check run as a manifest, nothing queued: for every case its job name, the full config to queue, what it
    changes against the base config and what counts as a pass; or why the case is left out (state "skipped").

    base: the base config as a dict of TrainConfig fields. vram: the card's GB the block swap cases plan for.
    fleet: the runs go to vast.ai instances, so nothing is looked up on this computer's disk. busy: job names
    running or waiting. planner: the wizard's memory plan as a function (cfg, vram) -> blocks to swap
    (wizard_block_swap)."""
    from . import __version__, runplan
    from .config import TrainConfig

    tag = tag or time.strftime("%y%m%d_%H%M%S")
    if not TAG_RE.fullmatch(tag):
        raise ValueError(f"--tag is 1 to 24 letters, digits, _ or -, got {tag!r}")
    base = TrainConfig.from_dict(dict(base)).to_dict()  # every field, and unknown keys refused
    base_cfg = TrainConfig.from_dict(base)
    family = base_cfg.arch_name()
    planner = planner or wizard_block_swap
    output_dir = str(base.get("output_dir") or "output")
    taken = set(busy)
    entries = []
    for case in (select() if cases is None else list(cases)):
        n_steps = int(case.steps or steps)
        if case.probes or case.expect.get("race"):
            n_steps = max(2, n_steps - n_steps % 2)  # a probe half-way, and a relay's steps divide by its rounds
        entry = {"name": case.name, "feature": case.feature, "note": case.note, "compare": case.compare,
                 "steps": n_steps, "state": "queued", "reason": "", "job": "", "plan": {}}
        entries.append(entry)

        def leave_out(why: str, entry=entry) -> None:
            entry.update(state="skipped", reason=why)

        if case.families and family not in case.families:
            leave_out(f"runs on {', '.join(case.families)} only (this base config trains {family})")
            continue
        data = dict(base)
        plain = plain_settings(base, n_steps)
        data.update(plain)
        own = dict(case.overrides)
        if case.probes:
            own.update(probe_settings(data, n_steps, case.probes))
        if own.get("reg_dir") == "<reg_dir>":
            own["reg_dir"] = str(reg_dir or base.get("reg_dir") or "").strip()
        if own.get("save_state_every") == "<steps>":
            own["save_state_every"] = n_steps
        data.update({k: v for k, v in own.items() if k != "block_swap"})
        missing = [need for need in case.needs if need != "vram" and not (
            (data.get("face_refs") or data.get("body_refs")) if need == "refs" else data.get(need))]
        if missing:
            leave_out("needs " + " and ".join(NEEDS[m] for m in missing))
            continue
        if "reg_dir" in case.needs and not fleet and not local_path(data["reg_dir"]).is_dir():
            leave_out(f"the regularisation folder is not there: {data['reg_dir']}")
            continue
        if case.whole_model:  # (a run without gradient checkpointing cannot swap blocks: config.validate)
            needed = int(base.get("block_swap") or 0)
            if not needed and vram is not None:
                try:
                    needed = int(planner(TrainConfig.from_dict(dict(data, block_swap=0)), float(vram)))
                except Exception:  # noqa: BLE001  (no plan: the base config's own block swap decides)
                    needed = 0
            if needed:
                leave_out(f"the model needs block swap ({needed} blocks) to fit this card, and this case runs "
                          f"without it: its first try would fail while loading and again after the one lighter "
                          f"restart. Run it on a card that holds the model whole")
                continue
        swap = own.get("block_swap")
        if isinstance(swap, str):  # the wizard's memory plan for this adapter on a card of `vram` GB
            floor = 4 if "at least" in swap else 0
            if vram is None and not floor:
                leave_out("needs " + NEEDS["vram"])
                continue
            planned = None
            if vram is not None:
                try:
                    planned = int(planner(TrainConfig.from_dict(dict(data, block_swap=0)), float(vram)))
                except Exception as e:  # noqa: BLE001  (webui.py could not be loaded, say)
                    if not floor:
                        leave_out(f"the wizard's memory plan could not be made ({type(e).__name__}: {e})")
                        continue
            own["block_swap"] = max(floor, planned or 0, int(base.get("block_swap") or 0) if floor else 0)
            entry["plan"].update(vram_gb=vram, wizard_block_swap=planned)
        if "block_swap" in own:
            data["block_swap"] = own["block_swap"]
        job = f"check_{tag}_{case.name}"
        if runplan.name_in(job, taken) or (not fleet and runplan.holds_output(local_path(output_dir) / job)):
            job = runplan.next_free_name(job, taken, output_dir if fleet else local_path(output_dir))
        data["job_name"] = job
        try:
            cfg = TrainConfig.from_dict(data)
            cfg.validate()
        except (TypeError, ValueError) as e:
            leave_out(f"its settings do not go with this base config: {e}")
            continue
        taken.add(job)
        expect = json.loads(json.dumps(case.expect))  # a copy the manifest owns
        expect["steps"] = n_steps
        if cfg.block_swap > 0 and case.name.startswith("block_swap") and BLOCK_SWAP_LINE not in expect.get("log", []):
            expect.setdefault("log", []).append(BLOCK_SWAP_LINE)
        if expect.pop("peak_within_vram", False) and vram is not None:
            expect["peak_max_gb"] = float(vram)
        if case.probes:
            expect.setdefault("probe", {})
        if expect.get("files"):
            expect["files"] = [pattern.replace("{job}", job) for pattern in expect["files"]]
        entry["plan"]["adapter_gb"] = adapter_gb(cfg)
        entry.update(job=job, config=cfg.to_dict(), expect=expect,
                     changed={k: [base.get(k), v] for k, v in cfg.to_dict().items()
                              if k in own and base.get(k) != v},
                     plain={k: [base.get(k), v] for k, v in plain.items()
                            if k not in own and base.get(k) != v})
    return {"tag": tag, "created": time.time(), "version": __version__, "mode": "fleet" if fleet else "local",
            "family": family, "output_dir": output_dir, "steps": int(steps), "vram_gb": vram, "cases": entries}


NEEDS = {
    "reg_dir": "a regularisation folder (--reg-dir, or reg_dir in the base config)",
    "face_refs": "face reference pictures (face_refs in the base config)",
    "body_refs": "body reference pictures (body_refs in the base config)",
    "refs": "face or body reference pictures (face_refs / body_refs in the base config)",
    "vram": "the card's memory to plan the block swap for (--vram GB)",
}


# ---------------------------------------------------------------- queueing


def webui_queue():
    """The WebUI's own training queue (output/_queue.json)."""
    from .train_queue import TrainQueue

    return TrainQueue()


def queue_local(manifest: dict, queue=None, resume: bool = False) -> dict:
    """Add the planned runs to the WebUI's own training queue. The WebUI starts them one after another; it must be
    running for anything to start."""
    q = queue or webui_queue()
    for entry in manifest["cases"]:
        if entry["state"] == "queued":
            entry["queue_id"] = q.add(entry["config"], label=f"check run: {entry['name']}")["id"]
    if resume:
        q.set_paused(False)
    return q.snapshot()


def waiting_names(queue=None, job_file: Path | None = None) -> set[str]:
    """The job names the WebUI's queue holds and the one it runs now (its .webui_job.json), as far as they can be
    read: a run planned under one of them would be refused at its start, which pauses the queue."""
    names = set()
    try:
        names = {str(e.get("job") or "") for e in (queue or webui_queue()).snapshot()["waiting"]}
    except Exception:  # noqa: BLE001  (an unreadable queue file: no names to skip)
        pass
    running = _json(job_file or ROOT / ".webui_job.json", {})
    if isinstance(running, dict) and running.get("job_dir"):
        names.add(Path(str(running["job_dir"])).name)
    return names - {""}


def fleet_module():
    """scripts/vast/rulz_fleet.py, which is a script beside its helpers, not a package."""
    folder = str(ROOT / "scripts" / "vast")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import rulz_fleet

    return rulz_fleet


def queue_fleet(manifest: dict, fleet=None, instance=None, push=(), check: bool = True) -> dict:
    """Add the planned runs to the fleet's queue (scripts/vast/rulz_fleet.py): each goes to the first instance that
    falls idle, or to `instance`. push: folders on this computer sent to the instance first (the dataset, the
    regularisation folder). check: the trainer's own checks on a reachable instance before a run is queued; a run it
    refuses is left out and says why (state "refused")."""
    module = fleet_module()
    fleet = fleet or module.Fleet()
    for entry in manifest["cases"]:
        if entry["state"] != "queued":
            continue
        try:
            made = fleet.plan(entry["job"], entry["config"], instance=instance, push=list(push),
                              download=dict(FLEET_DOWNLOAD), label=f"check run {manifest['tag']}: {entry['name']}")
            if check:
                entry["checked"] = module.check_entries(fleet, made)
            fleet.enqueue(entries=made)
            entry["queue_id"] = made[0]["id"]
        except module.Refused as e:
            entry.update(state="refused", reason=str(e))
    return {"looking": fleet.someone_looks(), "queue": fleet.queue_view()}


def manifest_dir(tag: str, root: Path | None = None) -> Path:
    return (root or MANIFESTS) / tag


def save_manifest(manifest: dict, root: Path | None = None) -> Path:
    path = manifest_dir(manifest["tag"], root) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return path


def load_manifest(tag: str = "", root: Path | None = None) -> tuple[dict, Path]:
    """A check run's manifest and its file: by tag, by path (the file or its folder), or the newest one."""
    root = root or MANIFESTS
    if tag and (Path(tag).is_file() or (Path(tag) / "manifest.json").is_file()):
        path = Path(tag) if Path(tag).is_file() else Path(tag) / "manifest.json"
    elif tag:
        path = root / tag / "manifest.json"
    else:
        found = sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise FileNotFoundError(f"No check run in {root} yet: queue one first (queue --config FILE)")
        path = found[-1]
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except OSError as e:
        raise FileNotFoundError(f"No check run manifest at {path}") from e


# ---------------------------------------------------------------- reading a finished run


def _json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_job(folder) -> dict:
    """What a run left in its job folder, as far as the checks read it."""
    folder = Path(folder)
    records = []
    for line in _text(folder / "likeness.jsonl").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    tries = _json(folder / "oom_recovery.json", {})
    tries = tries.get("tries") if isinstance(tries, dict) else None
    files = sorted(p.relative_to(folder).as_posix() for pattern in ("*", "*/*") for p in folder.glob(pattern)
                   if p.is_file())
    status = _json(folder / "status.json", {})
    return {"folder": folder, "status": status if isinstance(status, dict) else {},
            "log": _text(folder / "train.log"), "records": records, "files": files,
            "tries": [t for t in tries if isinstance(t, dict)] if isinstance(tries, list) else []}


def find_job(job: str, manifest: dict, roots=()) -> Path | None:
    """The job's folder: under one of `roots` (--jobs-dir), the check run's own output folder on this computer, or
    where the fleet puts a run's fetched files (~/.rulz_vast/runs/<instance>/<job>). The newest when several hold it."""
    spots = [Path(r) / job for r in roots]
    spots.append(local_path(manifest.get("output_dir") or "output") / job)
    pulls = Path(os.environ.get("RULZ_VAST_HOME") or Path.home() / ".rulz_vast") / "runs"
    spots += sorted(pulls.glob(f"*/{job}")) if pulls.is_dir() else []
    found = [p for p in spots if p.is_dir()]
    if not found:
        return None

    def age(p: Path) -> float:
        try:
            return (p / "status.json").stat().st_mtime
        except OSError:
            return 0.0

    return max(found, key=age)


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _holds(value, op: str, want) -> bool:
    if op == "==":
        return value == want
    if op == "!=":
        return value != want
    a, b = _number(value), _number(want)
    if a is None or b is None:
        return False
    return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}[op]


def peak_gb(status: dict) -> float | None:
    """The run's peak VRAM in GB: the highest of status.json's peak_mem (PyTorch's own count per training GPU)."""
    values = [v for v in ((status.get("peak_mem") or {}).values() if isinstance(status.get("peak_mem"), dict) else [])
              if _number(v) is not None]
    return round(max(values), 2) if values else None


def step_seconds(status: dict) -> float | None:
    """Seconds a step took: the median of status.json's step_times, the first two (warm-up, compile) left out when
    there are enough."""
    times = status.get("step_times")
    times = [t for t in (times if isinstance(times, list) else []) if _number(t) is not None]
    times = times[2:] if len(times) > 4 else times
    return round(statistics.median(times), 3) if times else None


def pinned_gb(log_text: str) -> float:
    """GB of swapped blocks' weights the run kept in RAM (its block swap line), 0 without block swap."""
    found = re.findall(BLOCK_SWAP_LINE, log_text)
    return float(found[-1][1]) if found else 0.0


def off_gpu_gb(log_text: str) -> float:
    """GB of the swapped blocks' weights that are off the GPU at any moment: all of them but the two block swap
    keeps there (the block running and the next one, blockswap.py), which the run's peak VRAM already holds."""
    found = re.findall(BLOCK_SWAP_LINE, log_text)
    if not found:
        return 0.0
    blocks, gb = int(found[-1][0]), float(found[-1][1])
    return round(gb * max(0, blocks - 2) / blocks, 2) if blocks > 0 else 0.0


def text_cached(log_text: str) -> int:
    """Prompts this run encoded before training (its text caching line). The text encoder was on the GPU for it, so
    the run's peak VRAM may be the text encoder's and not the training's."""
    return max([int(n) for n in re.findall(r"Caching text embeddings: (\d+) of \d+ prompts need encoding", log_text)]
               or [0])


def evaluate(entry: dict, folder=None) -> dict:
    """One case's result from its job folder: {"verdict", "checks": [{"name", "ok", "detail", "feature"}], ...}.

    Verdicts: pass; fail; not exercised (the run finished, but the feature never got to act: its checks say why);
    not finished (no end in status.json: still running, waiting, or killed); not started (no job folder or no
    status.json); skipped / refused (never queued: the manifest's reason)."""
    out = {"case": entry["name"], "feature": entry.get("feature", ""), "job": entry.get("job", ""),
           "folder": str(folder) if folder else "", "verdict": "", "checks": [], "notes": [], "step": None,
           "total": None, "loss": None, "peak_gb": None, "step_seconds": None, "pinned_gb": 0.0, "off_gpu_gb": 0.0,
           "text_cached": 0, "ran": False, "adapter_gb": (entry.get("plan") or {}).get("adapter_gb")}
    if entry.get("state") != "queued":
        out.update(verdict=entry.get("state") or "skipped", notes=[entry.get("reason") or ""])
        return out
    job = read_job(folder) if folder else None
    if job is None or not job["status"]:
        out.update(verdict="not started", notes=[
            "no job folder found: the run has not started, or its files are somewhere else (--jobs-dir)"
            if job is None else "the job folder holds no status.json that can be read yet"])
        return out
    status, log_text, expect, name = job["status"], job["log"], entry.get("expect") or {}, entry["job"]
    out.update(step=status.get("step"), total=status.get("total"), loss=_number(status.get("avg_loss")), peak_gb=peak_gb(status),
               step_seconds=step_seconds(status), pinned_gb=pinned_gb(log_text), off_gpu_gb=off_gpu_gb(log_text),
               text_cached=text_cached(log_text))
    checks = out["checks"]

    def check(label: str, ok, detail: str = "", feature: bool = True) -> None:
        checks.append({"name": label, "ok": None if ok is None else bool(ok), "detail": detail, "feature": feature})

    error = status.get("error")
    if not status.get("done") and not error:
        out.update(verdict="not finished", notes=[
            f"stage {status.get('stage')!r} at step {status.get('step')} of {status.get('total')}: still running, "
            f"or it ended without saying so (see console.log)"])
        return out
    # ---- every run
    check("the run finished", bool(status.get("done")) and not error and status.get("stage") == "finished",
          f"error: {str(error)[:300]}" if error else f"stage {status.get('stage')!r}", feature=False)
    step, total = _number(status.get("step")), _number(status.get("total"))
    reached = step is not None and total is not None and total > 0 and step >= total
    if not expect.get("race"):  # (a race counts every configuration's steps: its own total)
        reached = reached and total == expect.get("steps", total)
    check("it reached its last step", reached,
          f"step {status.get('step')} of {status.get('total')}" + (
              "" if expect.get("race") else f" ({expect.get('steps')} planned)"), feature=False)
    losses = [v for v in (status.get("losses") or []) if not isinstance(v, (list, tuple))]
    bad = [v for v in [status.get("loss"), status.get("avg_loss"), *losses] if _number(v) is None]
    check("the loss is a number", not bad and bool(losses),
          f"{len(bad)} value(s) that are not finite numbers" if bad else
          f"last {status.get('loss')}, average {status.get('avg_loss')}" if losses else "no loss was recorded",
          feature=False)
    skipped = _number(status.get("skipped_steps")) or 0
    check("no step was left out for non-finite gradients", skipped == 0, f"{int(skipped)} step(s) skipped",
          feature=False)
    final = f"{name}.safetensors"
    on_disk = (job["folder"] / final).is_file() and (job["folder"] / final).stat().st_size > 0
    listed = any(Path(str(p).replace("\\", "/")).name == final for p in status.get("saved") or [])
    check("the final LoRA was saved", on_disk or listed,
          final if on_disk else f"{final} is listed in status.json (the file is not in this folder)" if listed
          else f"no {final}", feature=False)
    last = f"step{int(expect.get('steps') or 0):06d}"
    shots = [f for f in job["files"] if f.startswith("samples/") and f.lower().endswith(PICTURES) and last in f]
    listed = [s for s in status.get("samples") or [] if last in str(s)]
    check("the final sample was rendered", bool(shots or listed),
          shots[0] if shots else "listed in status.json (the picture is not in this folder)" if listed
          else f"no sample of {last} in samples/", feature=False)
    if "recovery" not in expect and job["tries"]:  # (every other case runs with oom_retries 0: a guard)
        check("the run was not started again with other settings after running out of memory", False,
              f"{len(job['tries'])} automatic tries: " + "; ".join(str(t.get("action")) for t in job["tries"]),
              feature=False)
    # ---- the feature
    for pattern in expect.get("log") or []:
        m = re.search(pattern, log_text)
        check(f"train.log: {pattern}", m is not None, m.group(0)[:160] if m else "no such line")
    for spec in expect.get("numbers") or []:
        found = [float(x) for x in re.findall(spec["pattern"], log_text)]
        best = max(found) if found else None
        check(f"{spec['name']} {spec['op']} {spec['value']}", best is not None and _holds(best, spec["op"], spec["value"]),
              f"{best:g}" if best is not None else "no such line in train.log")
    for key, op, want in expect.get("status") or []:
        check(f"status.json: {key} {op} {want}", _holds(status.get(key), op, want), f"{key} is {status.get(key)!r}")
    for pattern in expect.get("files") or []:
        pattern = pattern.replace("{job}", name)
        hit = [f for f in job["files"] if fnmatch.fnmatch(f, pattern)]
        check(f"file {pattern}", bool(hit), hit[0] if hit else "not in the job folder")
    if "probe" in expect:
        _check_probes(expect["probe"], job["records"], check)
    if "recovery" in expect:
        n = len(job["tries"])
        check("the run was started again after running out of memory", n >= int(expect["recovery"]["min_tries"]),
              f"{n} automatic tries: " + "; ".join(str(t.get("action")) for t in job["tries"]) if n
              else "oom_recovery.json holds no try")
    peak = out["peak_gb"]
    if "peak_max_gb" in expect:
        # the card size is the driver's (nvidia-smi: MiB / 1024), peak_mem is bytes / 1e9: both read in GiB here
        limit = float(expect["peak_max_gb"])
        gib = None if peak is None else peak * 1e9 / 2 ** 30
        check(f"the peak VRAM stays within the {limit:g} GB card the block swap was planned for",
              None if gib is None else gib <= limit,
              "no peak_mem in status.json" if gib is None else f"{peak:g} GB, which is {gib:.2f} of the card's {limit:g}")
    # ---- the verdict
    unexercised = []
    for pattern, why in expect.get("unexercised") or []:
        m = re.search(pattern, log_text)
        if m:
            unexercised.append(why.format(*m.groups()))
    if "recovery" in expect and not job["tries"]:
        unexercised.append(expect["recovery"]["else"])
    run_ok = out["ran"] = all(c["ok"] for c in checks if not c["feature"])
    feature_ok = all(c["ok"] is not False for c in checks if c["feature"])
    if not run_ok:
        out["verdict"] = "fail"
    elif unexercised:
        out["verdict"] = "not exercised"
        out["notes"] += unexercised
    else:
        out["verdict"] = "pass" if feature_ok else "fail"
    if out["verdict"] != "pass":
        for pattern in EXPLAIN:
            m = re.search(pattern, log_text)
            if m:
                out["notes"].append("train.log: " + m.group(0).strip()[:300])
    if peak is None:
        out["notes"].append("status.json has no peak_mem (the run did not train on a CUDA GPU?)")
    if out["text_cached"]:
        out["notes"].append(f"this run encoded {out['text_cached']} prompt(s) first, with the text encoder on the "
                            f"GPU: its peak VRAM may be that and not the training's")
    return out


def _check_probes(spec: dict, records: list[dict], check) -> None:
    """The likeness probes of a case: a record of the trained LoRA (a step after 0), the fields its last record must
    carry (which face pack or body score measured it), and the scores that must be numbers in some record."""
    trained = [r for r in records if (_number(r.get("step")) or 0) > 0]
    check("a likeness probe of the trained LoRA is in likeness.jsonl", bool(trained),
          f"{len(records)} record(s), {len(trained)} after step 0")
    for key, want in (spec.get("last") or {}).items():
        got = trained[-1].get(key) if trained else None
        check(f"the probe was measured with {key} {want}", got == want, f"{key} is {got!r}")
    for key in spec.get("scored") or []:
        scores = [r.get(key) for r in records if _number(r.get(key)) is not None]
        check(f"a probe has a {key} score", bool(scores),
              f"{len(scores)} of {len(records)} record(s), last {scores[-1]:.4g}" if scores
              else f"no record has a {key} score")


def compare(results: list[dict], manifest: dict) -> None:
    """What needs two runs: each case's peak VRAM and step time beside its reference case's (the baseline), and for
    the block swap plan cases the adapter's measured memory against what the plan counted (a check)."""
    by_name = {r["case"]: r for r in results}
    configs = {e["name"]: e.get("config") or {} for e in manifest["cases"]}
    for entry in manifest["cases"]:
        res = by_name[entry["name"]]
        if not res["ran"]:  # a run that did not get to its end has no peak or step time to hold against another's
            continue
        ref = by_name.get(entry.get("compare") or "")
        if ref is not None and ref["verdict"] == "pass":
            if res["peak_gb"] is not None and ref["peak_gb"] is not None:
                res["notes"].append(f"peak VRAM {res['peak_gb']:g} GB, {res['peak_gb'] - ref['peak_gb']:+.2f} GB "
                                    f"against {ref['case']}'s {ref['peak_gb']:g} GB")
            if res["step_seconds"] and ref["step_seconds"]:
                res["notes"].append(f"{res['step_seconds']:g} s a step, "
                                    f"x{res['step_seconds'] / ref['step_seconds']:.2f} of {ref['case']}'s "
                                    f"{ref['step_seconds']:g} s")
        spec = (entry.get("expect") or {}).get("adapter_memory")
        if not spec:
            continue
        label = "the adapter holds no more memory than the block swap plan counts"
        ref = by_name.get(spec["against"])
        why = ""
        if ref is None or ref["verdict"] != "pass":
            why = f"it needs a passed {spec['against']} run"
        elif None in (res["peak_gb"], ref["peak_gb"], res["adapter_gb"], ref["adapter_gb"]):
            why = "it needs both runs' peak VRAM and counted adapter size"
        elif {str(configs[res["case"]].get("multi_gpu") or "off"),
              str(configs[ref["case"]].get("multi_gpu") or "off")} != {"off"}:
            why = f"{ref['case']} trains on several GPUs (multi_gpu), so its peak is one card's share"
        elif res["text_cached"] or ref["text_cached"]:
            who = " and ".join(r["case"] for r in (res, ref) if r["text_cached"])
            why = (f"{who} encoded prompts first, so that peak VRAM may be the text encoder's: queue the check run "
                   f"again now that the prompts are cached")
        if why or ref is None:
            res["checks"].append({"name": label, "ok": None, "feature": True, "detail": f"not compared: {why}"})
            continue
        # the peak holds the weights on the GPU: adding back the swapped ones that are off it gives the model, the
        # adapter and the step's memory, on both sides
        measured = (res["peak_gb"] + res["off_gpu_gb"]) - (ref["peak_gb"] + ref["off_gpu_gb"])
        counted = res["adapter_gb"] - ref["adapter_gb"]
        allowed = float(spec.get("tolerance_gb", 1.0))
        ok = measured - counted <= allowed
        res["checks"].append({"name": label, "ok": ok, "feature": True, "detail": (
            f"measured {measured:+.2f} GB against {ref['case']} (peak {res['peak_gb']:g} GB + {res['off_gpu_gb']:g} "
            f"GB of swapped weights off the GPU, against {ref['peak_gb']:g} + {ref['off_gpu_gb']:g}); the plan "
            f"counts {counted:+.2f} GB ({res['adapter_gb']:g} against {ref['adapter_gb']:g}); allowed over: "
            f"{allowed:g} GB")})
        if counted - measured > 2 * allowed:
            res["notes"].append(f"the adapter measures {counted - measured:.2f} GB under the plan's count: either "
                                f"the count is generous, or {ref['case']}'s peak was set while loading and not "
                                f"while training (both peaks are the whole process's)")
        if not ok and res["verdict"] == "pass":
            res["verdict"] = "fail"


def report(manifest: dict, roots=()) -> dict:
    """The check run's results: every case evaluated from its job folder, then compared."""
    results = [evaluate(entry, find_job(entry["job"], manifest, roots) if entry.get("job") else None)
               for entry in manifest["cases"]]
    compare(results, manifest)
    counts: dict[str, int] = {}
    for res in results:
        counts[res["verdict"]] = counts.get(res["verdict"], 0) + 1
    base = next((r for r in results if r["case"] == "baseline"), None)
    warning = ""
    if base is not None and base["verdict"] == "fail":
        warning = ("The baseline failed: the base config itself does not train as queued, so the other cases' "
                   "failures may be the same problem. Fix the baseline first.")
    return {"tag": manifest["tag"], "version": manifest.get("version"), "mode": manifest.get("mode"),
            "family": manifest.get("family"), "vram_gb": manifest.get("vram_gb"), "made": time.time(),
            "counts": counts, "warning": warning, "results": results}


# ---------------------------------------------------------------- words


def _cell(value) -> str:
    return "" if value is None else str(value).replace("|", "/").replace("\n", " ")


def _failed(res: dict) -> list[str]:
    return [f"{c['name']} ({c['detail']})" if c["detail"] else c["name"] for c in res["checks"] if c["ok"] is False]


def markdown(rep: dict) -> str:
    """The report as Markdown: the table, then every case's checks."""
    lines = [f"# RulZ AI Trainer check run {rep['tag']}", "",
             f"RulZ AI Trainer {rep.get('version')}, {rep.get('family')}, queued to "
             f"{'the vast.ai fleet' if rep.get('mode') == 'fleet' else 'the WebUI queue'}"
             + (f", block swap planned for a {rep['vram_gb']:g} GB card" if rep.get("vram_gb") else "") + ". "
             + ", ".join(f"{n} {v}" for v, n in sorted(rep["counts"].items())) + ".", ""]
    if rep.get("warning"):
        lines += [f"**{rep['warning']}**", ""]
    lines += ["| Case | Verdict | Step | Loss | Peak VRAM (GB) | s / step | Notes |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for res in rep["results"]:
        step = f"{res['step']} / {res['total']}" if res["step"] is not None else ""
        loss = f"{res['loss']:.4f}" if res["loss"] is not None else ""
        notes = "; ".join([*(["failed: " + "; ".join(_failed(res))] if _failed(res) else []), *res["notes"]])
        lines.append(f"| {res['case']} | {res['verdict']} | {step} | {loss} | {_cell(res['peak_gb'])} | "
                     f"{_cell(res['step_seconds'])} | {_cell(notes)} |")
    lines += ["", "Peak VRAM is PyTorch's own count of what the training process allocated (status.json's peak_mem), "
                  "which is below what nvidia-smi shows for the card.", "", "## The checks", ""]
    for res in rep["results"]:
        lines += [f"### {res['case']}: {res['verdict']}", "", res["feature"], ""]
        if res["job"]:
            lines += [f"Job `{res['job']}`" + (f", read from `{res['folder']}`" if res["folder"] else ""), ""]
        for c in res["checks"]:
            mark = {True: "pass", False: "FAIL", None: "n/a"}[c["ok"]]
            lines.append(f"- {mark}: {c['name']}" + (f" ({c['detail']})" if c["detail"] else ""))
        lines += [f"- note: {n}" for n in res["notes"] if n]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_report(rep: dict, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"markdown": folder / "report.md", "json": folder / "report.json"}
    paths["markdown"].write_text(markdown(rep), encoding="utf-8", newline="\n")
    paths["json"].write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return paths


def _show(value) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= 110 else text[:107] + "..."


def passes_when(expect: dict) -> list[str]:
    """What counts as a pass, in words (besides what every run is checked for)."""
    out = [f"train.log has a line matching: {p}" for p in expect.get("log") or []]
    out += [f"{s['name']} {s['op']} {s['value']} (from train.log)" for s in expect.get("numbers") or []]
    out += [f"status.json: {k} {op} {v}" for k, op, v in expect.get("status") or []]
    out += [f"the job folder holds {p}" for p in expect.get("files") or []]
    probe = expect.get("probe")
    if probe is not None:
        out.append("likeness.jsonl holds a probe of the trained LoRA" + "".join(
            f", measured with {k} {v}" for k, v in (probe.get("last") or {}).items()) + "".join(
            f", with a {k} score" for k in probe.get("scored") or []))
    if "recovery" in expect:
        out.append("oom_recovery.json holds at least one automatic try (else: not exercised)")
    if "peak_max_gb" in expect:
        out.append(f"the peak VRAM is at most {expect['peak_max_gb']:g} GB")
    if expect.get("peak_within_vram"):
        out.append("the peak VRAM is at most the card size the block swap was planned for (--vram)")
    if "adapter_memory" in expect:
        spec = expect["adapter_memory"]
        out.append(f"the adapter's measured memory against {spec['against']} is at most {spec['tolerance_gb']:g} GB "
                   f"over what the plan counts")
    return out


EVERY_RUN = ("status.json says the run finished without an error, at its last step", "the loss is a finite number and "
             "no step was left out for non-finite gradients", "the final LoRA <job>.safetensors was saved",
             "the final sample was rendered", "the peak VRAM (peak_mem) and the step time go into the table")


def list_text(family: str = "") -> str:
    lines = ["RulZ AI Trainer check run: the cases (python -m rulztrainer.checkrun queue --config FILE --dry-run shows "
             "them planned on a config)", ""]
    for case in CASES:
        if family and case.families and family not in case.families:
            continue
        lines.append(f"{case.name}  ({case.steps or DEFAULT_STEPS} steps"
                     + (f"; {', '.join(case.families)} only" if case.families else "") + ")")
        lines.append(f"    {case.feature}")
        for key, value in case.overrides.items():
            lines.append(f"    sets {key}: " + (PLANNED[value] if isinstance(value, str) and value in PLANNED
                                                 else _show(value)))
        if case.probes:
            lines.append(f"    sets likeness probes ({case.probes}): at the start, half-way and of the final LoRA")
        for need in case.needs:
            lines.append(f"    needs {NEEDS[need]}")
        if case.whole_model:
            lines.append("    needs a card that holds the model without block swap")
        lines += [f"    passes when {line}" for line in passes_when(case.expect)]
        if case.note:
            lines.append(f"    note: {case.note}")
        lines.append("")
    lines += ["Every run is also checked for:"] + [f"    {line}" for line in EVERY_RUN]
    return "\n".join(lines) + "\n"


def plan_text(manifest: dict, dry_run: bool = True) -> str:
    cases = manifest["cases"]
    queued = [e for e in cases if e["state"] == "queued"]
    where = "the vast.ai fleet's queue" if manifest["mode"] == "fleet" else "the WebUI's queue"
    lines = [f"Check run {manifest['tag']}: {len(queued)} of {len(cases)} cases for {where}"
             + (" (dry run: nothing is queued)" if dry_run else ""),
             f"RulZ AI Trainer {manifest['version']}, {manifest['family']}, output folder {manifest['output_dir']}"
             + (f", block swap planned for a {manifest['vram_gb']:g} GB card" if manifest.get("vram_gb") else
                ", no card size given (--vram): the block swap plan cases are left out"), ""]
    plain = next((e["plain"] for e in queued if e["name"] == "baseline"), queued[0]["plain"] if queued else {})
    if plain:
        lines.append("Every run, against the base config:")
        lines += [f"    {k}: {_show(new)}  (was {_show(old)})" for k, (old, new) in plain.items()]
        lines.append("")
    for i, e in enumerate(cases, 1):
        if e["state"] != "queued":
            lines += [f"{i:2d}. {e['name']}: {e['state'].upper()}, {e['reason']}", ""]
            continue
        lines.append(f"{i:2d}. {e['name']}: job {e['job']}, {e['steps']} steps")
        lines.append(f"    {e['feature']}")
        for k, (old, new) in e["changed"].items():
            lines.append(f"    {k}: {_show(new)}  (was {_show(old)})")
        if not e["changed"]:
            lines.append("    (nothing beyond what every run changes)")
        extra = {k: v for k, v in e["plain"].items() if list(plain.get(k) or []) != list(v)}
        lines += [f"    {k}: {_show(new)}  (was {_show(old)})" for k, (old, new) in extra.items()]
        if e["plan"].get("vram_gb") is not None:
            swapped = e["plan"].get("wizard_block_swap")
            lines.append(f"    the wizard's plan for {e['plan']['vram_gb']:g} GB: {swapped} blocks swapped" + (
                " (the model and this adapter fit whole: give --vram a smaller card's size to try the block swap)"
                if swapped == 0 else ""))
        if e["plan"].get("adapter_gb") is not None:
            lines.append(f"    the adapter is counted at {e['plan']['adapter_gb']:g} GB")
        lines += [f"    passes when {line}" for line in passes_when(e["expect"])]
        if e.get("note"):
            lines.append(f"    note: {e['note']}")
        lines.append("")
    return "\n".join(lines)


def report_text(rep: dict) -> str:
    rows = [("case", "verdict", "step", "loss", "peak GB", "s/step")]
    for res in rep["results"]:
        rows.append((res["case"], res["verdict"],
                     f"{res['step']}/{res['total']}" if res["step"] is not None else "",
                     f"{res['loss']:.4f}" if res["loss"] is not None else "", _cell(res["peak_gb"]),
                     _cell(res["step_seconds"])))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = [f"Check run {rep['tag']} ({rep.get('family')}, RulZ AI Trainer {rep.get('version')})", ""]
    lines += ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows]
    lines.append("")
    for res in rep["results"]:
        for line in [*("failed: " + f for f in _failed(res)), *res["notes"]]:
            if line:
                lines.append(f"{res['case']}: {line}")
    lines += ["", ", ".join(f"{n} {v}" for v, n in sorted(rep["counts"].items()))]
    if rep.get("warning"):
        lines += ["", rep["warning"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the command line


def cmd_list(args) -> int:
    if args.json:
        print(json.dumps([{"name": c.name, "feature": c.feature, "families": list(c.families),
                           "steps": c.steps or DEFAULT_STEPS, "overrides": c.overrides, "needs": list(c.needs),
                           "probes": c.probes, "expect": c.expect, "note": c.note} for c in CASES], indent=1))
    else:
        print(list_text(args.family), end="")
    return 0


def cmd_queue(args) -> int:
    from .config import TrainConfig

    cfg, _missing, unknown = TrainConfig.load_report(args.config)
    if unknown:
        print(f"Ignored settings this version does not have: {', '.join(unknown)}", file=sys.stderr)
    base = cfg.to_dict()
    base.update(parse_sets(args.set))
    cases = select([n for part in args.only for n in part.split(",")], [n for part in args.skip for n in part.split(",")])
    vram = args.vram
    if vram is None and not args.fleet:
        vram = detect_vram(str(base.get("device") or "cuda:0"))
    manifest = plan(base, cases, tag=args.tag, steps=args.steps, vram=vram, reg_dir=args.reg_dir, fleet=args.fleet,
                    busy=set() if args.fleet else waiting_names())
    manifest["base_config"] = str(args.config)
    queued = [e for e in manifest["cases"] if e["state"] == "queued"]
    problem = ""
    if not args.fleet:
        gone = [sp.path or "(none picked)" for sp in TrainConfig.from_dict(base).dataset_specs()
                if not sp.path or not local_path(sp.path).is_dir()]
        if gone:
            problem = f"Dataset folder not found on this computer: {', '.join(gone)}"
    used = (manifest_dir(manifest["tag"]) / "manifest.json").is_file()
    if args.dry_run:
        print(json.dumps(manifest, indent=1, default=str) if args.json else plan_text(manifest), end="\n")
        if problem:
            print(f"{problem}. Nothing would start: fix the base config first.", file=sys.stderr)
        if used:
            print(f"The tag {manifest['tag']!r} is taken by an earlier check run: queue with another --tag.",
                  file=sys.stderr)
        return 0 if queued and not problem else 1
    if used:  # its manifest would be written over, and that check run's report with it
        raise ValueError(f"the tag {manifest['tag']!r} is taken by an earlier check run "
                         f"({manifest_dir(manifest['tag'])}): give this one another --tag")

    if problem or not queued:
        print(plan_text(manifest))
        print(problem or "No case can run on this base config: nothing was queued.", file=sys.stderr)
        return 1
    if args.fleet:
        state = queue_fleet(manifest, instance=args.instance, push=args.push, check=not args.no_check)
        note = ("Something is looking after the fleet's queue, so the runs go out as instances fall idle."
                if state["looking"] else "Nothing is looking after the fleet's queue: start `python "
                "scripts/vast/rulz_fleet.py daemon`, or open the WebUI's Fleet tab on this computer.")
    else:
        state = queue_local(manifest, resume=args.resume)
        note = ("The WebUI starts them one after another: keep it open." + (
            " The queue is paused: press Resume on the Monitor (or queue with --resume)." if state["paused"] else ""))
    path = save_manifest(manifest)
    if args.json:
        print(json.dumps({"manifest": str(path), **manifest}, indent=1, default=str))
    else:
        print(plan_text(manifest, dry_run=False))
        print(f"Manifest: {path}\n{note}\nWhen they have ended: python -m rulztrainer.checkrun report {manifest['tag']}")
    return 0 if any(e["state"] == "queued" for e in manifest["cases"]) else 1


def cmd_report(args) -> int:
    try:
        manifest, path = load_manifest(args.tag)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    rep = report(manifest, roots=args.jobs_dir)
    paths = save_report(rep, Path(args.out) if args.out else path.parent)
    if args.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print(report_text(rep), end="")
        print(f"\nSaved {paths['markdown']} and {paths['json']}")
    verdicts = set(rep["counts"])
    return 1 if "fail" in verdicts else 2 if verdicts & {"not finished", "not started"} else 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rulztrainer.checkrun", description=(
        "Queue short real-model runs, one per feature that has never run for real, and report pass / fail from "
        "their job folders. It starts nothing itself: the WebUI's queue (or the vast.ai fleet's) starts the runs."))
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list", help="the cases and what each one sets, needs and is checked for (also: --list)")
    p.add_argument("--family", default="", help="only the cases that run on this model family")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("queue", help="plan the runs on a base config and add them to a queue (--dry-run: plan only)")
    p.add_argument("--config", required=True, help="the base config (YAML): its model, dataset and output folder")
    p.add_argument("--dry-run", action="store_true", help="show the plan and queue nothing")
    p.add_argument("--only", action="append", default=[], metavar="CASES", help="only these cases (comma-separated)")
    p.add_argument("--skip", action="append", default=[], metavar="CASES", help="leave these cases out")
    p.add_argument("--tag", default="", help="the check run's name, part of every job name (default: date and time)")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"steps per run (default {DEFAULT_STEPS})")
    p.add_argument("--vram", type=float, default=None, metavar="GB",
                   help="the card's memory the block swap is planned for (default: this computer's card, from "
                        "nvidia-smi; with --fleet the block swap plan cases are left out without it). A smaller "
                        "number plans for a smaller card")
    p.add_argument("--reg-dir", default="", help="the regularisation folder of the reg_dir case")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="change the base config first (JSON values, as rulztrainer.train --set)")
    p.add_argument("--resume", action="store_true", help="also resume the WebUI's queue when it is paused")
    p.add_argument("--fleet", action="store_true", help="queue to the vast.ai fleet instead of this computer's WebUI")
    p.add_argument("--instance", default=None, help="--fleet: only this instance takes the runs")
    p.add_argument("--push", action="append", default=[], metavar="FOLDER",
                   help="--fleet: a folder on this computer to send to the instance first (the dataset)")
    p.add_argument("--no-check", action="store_true", help="--fleet: skip the instance's own check of each run")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_queue)
    p = sub.add_parser("report", help="read the finished job folders: the table, saved as Markdown and JSON")
    p.add_argument("tag", nargs="?", default="", help="the check run's tag or its manifest file (default: the newest)")
    p.add_argument("--jobs-dir", action="append", default=[], metavar="FOLDER",
                   help="a folder that holds the job folders (default: the output folder, then the fleet's fetched "
                        "runs in ~/.rulz_vast/runs)")
    p.add_argument("--out", default="", help="where report.md and report.json go (default: beside the manifest)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--list":
        argv[0] = "list"
    args = parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as e:
        print(f"Refused: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
