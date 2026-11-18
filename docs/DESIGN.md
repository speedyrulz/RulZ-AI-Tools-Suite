# RulZ AI Trainer Optimizer: design

Status: 0.1.0 design, 2026-11-16. This file is the contract between the parts of the app. Change it only together
with the code that depends on it.

## What the app does

1. **Recommend**: the user picks a model, points at a dataset folder (or several), sets goals, and gets starting
   settings for a character LoRA in RulZ AI Trainer, each with a reason, a confidence and its sources.
2. **Export**: writes a RulZ AI Trainer config YAML (flat `TrainConfig` keys) and either
   - saves it for the trainer's WebUI "Config file" Load,
   - queues it in a local RulZ AI Trainer (`output/_queue.json`, through the trainer's own `TrainQueue`), or
   - queues it on vast.ai through the trainer's own `scripts/vast/rulz_fleet.py queue add` (which pushes the
     dataset and repoints its paths).
   The app never writes the trainer's files by hand: every write goes through the trainer's own Python code, run
   with the trainer's own interpreter, so the formats stay owned by the trainer.
3. **Analyse runs**: reads RulZ AI Trainer run folders (local `output/<job>/`, pulled vast runs in
   `~/.rulz_vast/runs/` and `~/.rulz_vast/fleet/runs/`, or any folder), shows the best checkpoint by face, body and
   combined likeness with its probe images, detects flaws from the numbers, lets the user tick flaws they see in the
   images, and proposes a corrected next config (exportable like above).
4. **Ask an AI**: sends the knowledge for the chosen model, the dataset summary, the recommendation and/or the run
   analysis to Claude Code, Codex, Grok Build or Antigravity (the user's choice), read-only, and shows the answer.

No torch, no GPU. Dependencies: `nicegui>=4.0,<5`, `pyyaml`, `pillow`. Python 3.10+.

## Layout

```
apps/RulZ-AI-Trainer-Optimizer/
  README.md  LICENSE  LOGO-NOTICE.md  pyproject.toml  requirements.txt
  start-windows.bat  start-linux.sh
  assets/icon.png  assets/icon.ico
  docs/DESIGN.md
  knowledge/                     # the knowledge base, data only (see "Knowledge base")
    general.yaml
    models/<arch>.yaml           # one per RulZ AI Trainer arch
  rulz_optimizer/
    __init__.py                  # __version__ = "0.1.0", APP_NAME = "RulZ AI Trainer Optimizer"
    __main__.py  main.py         # NiceGUI app, port search from 8775, /api/rulz-ping
    settings.py                  # app settings (see "Settings")
    knowledge.py                 # load + validate the knowledge base
    dataset_scan.py              # scan dataset folders
    recommend.py                 # recommendation engine
    trainer_keys.py              # snapshot of RulZ AI Trainer TrainConfig keys (generated)
    export.py                    # build + write the trainer YAML
    runs.py                      # read run folders
    diagnose.py                  # run -> flaws -> fixes -> next config
    bridge.py                    # talk to a RulZ AI Trainer install (validate, queue local, queue vast)
    advisor.py                   # run Claude Code / Codex / Grok Build / Antigravity read-only
    ui/ theme.py  recommend_page.py  runs_page.py  advisor_page.py  settings_page.py  common.py
  tools/snapshot_trainer_keys.py # regenerates trainer_keys.py from apps/RulzTrainer/rulztrainer/config.py
  tests/                         # pytest, no network, no GPU, no real CLIs
```

## Arch ids

Exactly RulZ AI Trainer's `ARCH_NAMES`: `moss_image fibo2 krea2 qwen_image21 flux3_klein flux2_klein flux2_klein4b zimage ideogram4
minimax_h3 p_h3_image minimax_h3_image pan2 sdxl`. `minimax_h3` covers both FL2VA and Ref2VA (`model_path`
`MiniMaxAI/MiniMax-H3` or `MiniMaxAI/MiniMax-H3#ref2va`); a model file may define `variants` for that. `zimage` covers
Base and Turbo the same way (variants `base` and `turbo`, which differ in `model_path`, the training adapter and the
sample steps and CFG).

## Knowledge base

YAML, UTF-8, LF. One file per arch in `knowledge/models/`, plus `knowledge/general.yaml`. The public files carry no
Discord handles, no personal details and no dataset specifics: a source names the place, the month, how many people
agreed and the kind of evidence.

### Confidence

`CONF` (two or more independent people/sources agree, or numbers / an A-B were shown), `REP` (one source), `DISP`
(sources disagree; `why` gives both sides), `MEAS` (measured in our own RulZ AI Trainer runs), `DOC` (official model
card / trainer documentation), `GUESS` (no evidence; a reasoned default, shown with a warning).

### Setting value form

Every recommended setting is a `Setting`:

```yaml
lora_rank:
  value: 16                      # a TrainConfig value
  why: "Low rank wins for one person on H3: ..."
  confidence: CONF
  sources: [h3t_2026_10]         # ids from this file's `sources` (or general.yaml's)
```

### Model file (`knowledge/models/<arch>.yaml`)

```yaml
id: krea2                         # must equal the file name and a RulZ AI Trainer arch
name: Krea 2
kind: image                       # image | video
updated: 2026-11-16
summary: >-                       # 2-4 sentences: what matters most when training a character on this model
  ...
model_preset: Krea-2-Raw          # RulZ AI Trainer MODEL_PRESETS key to train on (optional)
model_path: krea/Krea-2-Raw       # written to model_path when the user gives no base config (optional)
variants:                         # optional, e.g. minimax_h3 fl2va / ref2va; each may override `base` keys
  - id: fl2va
    label: FL2VA
    model_path: MiniMaxAI/MiniMax-H3
    base: {}
facts:                            # things to know, shown in the UI and sent to the AI
  - text: "Train on Raw; Turbo is for sampling only."
    confidence: CONF
    sources: [krea_2026_06]
base:                             # the character starting recipe: TrainConfig key -> Setting
  network_type: {value: lora, why: ..., confidence: ..., sources: [...]}
  lora_rank: {...}
  learning_rate: {...}
steps:                            # how many steps; total = clamp(round_to(epochs * images * repeats / batch))
  epochs: 100
  min: 1000
  max: 4000
  round_to: 50
  why: ...
  confidence: ...
  sources: [...]
adjust:                           # rules applied in file order after `base`; later rules win
  - id: tiny_dataset
    when: {images_max: 15}        # all conditions must hold (see "Conditions")
    set:                          # key -> Setting, or key -> {op: mul|add|min|max, value: x, why..., confidence...}
      lora_rank: {value: 8, why: ..., confidence: REP, sources: [...]}
    note: "optional text shown with the rule"
evaluate:                         # how to judge checkpoints (shown, and sent to the AI)
  strength: "1.0 (0.8-1.2)"
  sampler: "..."
  steps: "..."
  cfg: "..."
  notes: ["Save every 100-250 steps; checkpoints are not monotonic.", ...]
flaws:                            # model-specific fixes; symptom ids come from general.yaml `symptoms`
  - symptom: plastic_skin
    causes: ["overtrained", ...]
    fixes:
      - set: {learning_rate: {op: mul, value: 0.5}}
        why: ...
        confidence: REP
        sources: [...]
      - action: "Pick an earlier checkpoint (about 60-70% of the best step)."   # advice without a setting
        confidence: CONF
        sources: [...]
open_questions: ["..."]           # what nobody has settled; shown and sent to the AI
sources:
  krea_2026_06:
    kind: community               # community | web | official | measured
    where: "banodoco Discord #krea-2-image"
    date: "2026-06"               # YYYY-MM or YYYY-MM-DD
    people: 4                     # distinct people who agreed (community)
    url: null                     # web/official only
    note: "optional"
```

### `knowledge/general.yaml`

```yaml
updated: 2026-11-16
symptoms:                         # the shared vocabulary for flaws
  - id: plastic_skin
    label: "Plastic, waxy or peach-fuzz skin"
    kind: visual                  # visual (the user ticks it) | metric (diagnose.py detects it) | both
    description: "..."
general_facts: [ {text, confidence, sources} ]
flaws:                            # cross-model fixes, same form as a model file's `flaws`; model fixes come first
  - symptom: identity_bleed
    causes: [...]
    fixes: [...]
dataset_rules:                    # advice on the dataset itself: informational, shown on the Knowledge tab; it changes no setting
  - id: few_closeups
    text: "..."
sources: {...}
```

The symptom ids the code relies on (metric ones are detected by `diagnose.py`): `no_likeness`, `slow_likeness`
(still rising at the last probe), `peaked_early` (best well before the end and then falling), `likeness_unstable`
(large seed spread / jumpy), `lost_framing` (body probes without a body, prompt-check drift), `memorized` (copy check
or `*_over` above 1), `lr_collapsed` (automagic LR below 1e-5 for most of the run), `nan_or_diverged`, `burned`
(visual: overcooked, oversaturated, crunchy), and visual ones: `plastic_skin`, `washed_colors`, `soft_blurry`,
`identity_bleed`, `everyone_same_face`, `wrong_at_distance`, `wrong_at_angles`, `same_pose_outfit`,
`trigger_as_text`, `needs_high_strength`, `too_strong`, `deformed_bodies`, `lost_motion`, `bad_audio`,
`ignores_prompt`. The knowledge base may add more; the code must not crash on unknown ids.

### Conditions (`when`)

All optional; all given ones must hold. `images_min`, `images_max` (image count after repeats are NOT applied),
`videos_min`, `videos_max`, `goal` (one id or a list), `focus` (one id or a list), `vram_min_gb`, `vram_max_gb`,
`target` (`local` | `vast`), `has_captions` (bool: at least 80% of images have a caption file), `variant` (id or
list), `median_mp_max`, `median_mp_min`.

## Goals (user inputs)

- `goal`: `likeness` (most faithful identity), `balanced` (identity that still follows prompts, clothes, scenes;
  the default), `quick` (a fast first test).
- `focus`: `face` (face only) or `face_body` (face and body).
- `target`: `local` or `vast`; `vram_gb` (number; from nvidia-smi when local, typed for vast).
- `trigger`: trigger word (default: from the folder name, lower-case, no spaces).
- `base_config`: optional path to an existing RulZ AI Trainer `config.yaml` (from a previous run or the WebUI).
  Its model / hardware keys (model paths, quant, block swap, attention, devices) are kept; the recommendation
  overlays training keys only.

## Module APIs (Python)

```python
# knowledge.py
class KnowledgeError(Exception): ...
@dataclass class Setting: key: str; value: Any; why: str; confidence: str; sources: list[str]; op: str | None = None
@dataclass class Fix: set: dict[str, Setting]; action: str | None; why: str; confidence: str; sources: list[str]
@dataclass class Flaw: symptom: str; causes: list[str]; fixes: list[Fix]
@dataclass class Rule: id: str; when: dict; set: dict[str, Setting]; note: str | None
@dataclass class ModelKnowledge: id, name, kind, updated, summary, model_preset, model_path, variants: list[dict],
    facts: list[dict], base: dict[str, Setting], steps: dict, adjust: list[Rule], evaluate: dict,
    flaws: list[Flaw], open_questions: list[str], sources: dict[str, dict]
@dataclass class Symptom: id: str; label: str; kind: str; description: str
@dataclass class Knowledge: general: dict; symptoms: dict[str, Symptom]; general_flaws: list[Flaw];
    models: dict[str, ModelKnowledge]; general_sources: dict[str, dict]
def default_dir() -> Path                      # <app>/knowledge
def load(root: Path | None = None) -> Knowledge          # raises KnowledgeError with file + key on bad data
def validate(k: Knowledge, trainer_keys: set[str]) -> list[str]   # problems: unknown TrainConfig keys,
    # unknown source ids, bad confidence, unknown arch ids, `when` keys not in Conditions
def source_text(k: Knowledge, model_id: str, source_id: str) -> str  # "banodoco Discord #x, 2026-10, 4 people"

# dataset_scan.py
@dataclass class ImageInfo: path: Path; width: int; height: int; mp: float; caption: str | None
@dataclass class DatasetScan: folders: list[Path]; images: list[ImageInfo]; videos: list[Path];
    captioned: int; median_mp: float; min_side: int; aspect_buckets: dict[str, int];
    duplicates: list[tuple[Path, Path]]     # exact or near (8x8 average hash, Hamming <= 4)
    small: list[Path]                        # shortest side < 512
    trigger_found: str | None                # the most common first caption token if it is in >= 60% captions
    warnings: list[str]
    @property image_count: int; has_captions: bool (>= 80% captioned)
def scan(folders: list[Path], trigger: str | None = None, caption_ext: str = ".txt") -> DatasetScan
    # images: .png .jpg .jpeg .webp .bmp; videos: .mp4 .mov .mkv .webm; recursive (os.walk) over every sub-folder,
    # skipping hidden folders and files (a leading dot) and excluded_by_look; PIL only

# recommend.py
@dataclass class Goals: goal="balanced"; focus="face_body"; target="local"; vram_gb: float | None = None;
    trigger: str | None = None; variant: str | None = None; base_config: Path | None = None
@dataclass class Choice: key: str; value: Any; why: str; confidence: str; sources: list[str]; origin: str
    # origin: "base" | "rule:<id>" | "steps" | "default" | "dataset" | "base_config"
@dataclass class Recommendation: model_id: str; goals: Goals; choices: dict[str, Choice]; steps: int;
    warnings: list[str]; notes: list[str]; evaluate: dict; facts: list[dict]; open_questions: list[str]
    def config(self) -> dict          # key -> value, only keys in trainer_keys.KEYS
def recommend(k: Knowledge, model_id: str, scan: DatasetScan | None, goals: Goals) -> Recommendation
    # base -> variant base -> adjust rules (conditions vs scan + goals) -> steps -> save/sample/probe cadence
    # (save_every = sample_every = probe_every ~ steps/10 rounded to 50, min 50 for <= 1500 steps else 100)
    # -> dataset keys (datasets: [{path, repeats, trigger}] or dataset_dir, trigger_word, caption_ext)
    # -> model_arch, model_path (unless base_config gives model keys)

# trainer_keys.py  (generated; do not edit by hand)
TRAINER_VERSION = "3.28.0"
KEYS: dict[str, str]        # key -> python type name ("int", "float", "str", "bool", "list", "dict")
DEFAULTS: dict[str, Any]
OPTIMIZERS: tuple[str, ...]; TIMESTEP_SAMPLING: tuple[str, ...]; LR_SCHEDULERS: tuple[str, ...]
ARCH_NAMES: tuple[str, ...]; MODEL_PRESETS: dict[str, str]
MODEL_KEYS: tuple[str, ...] # keys copied from a base config (model paths, quant, attention, block swap, devices)

# export.py
def build_config(rec: Recommendation, job_name: str, base_config: dict | None = None,
                 extra: dict | None = None) -> dict        # only KEYS; floats written so YAML reads them as numbers
def to_yaml(cfg: dict, header: str) -> str                 # header = comment lines: app version, trainer version,
                                                            # model, date, "made by RulZ AI Trainer Optimizer"
def write(cfg: dict, path: Path, header: str) -> Path      # atomic (tmp + os.replace)
def load_trainer_config(path: Path) -> dict                # lenient read of a trainer config.yaml

# runs.py
@dataclass class ProbePoint: step: int; face: float|None; body: float|None; combined: float|None;
    face_seeds: list[float]; body_seeds: list[float]; best: float|None; best_step: int|None;
    body_missing: int|None; drift_share: float|None; kept_share: float|None; copy: float|None;
    face_over: float|None; body_over: float|None
@dataclass class Checkpoint: path: Path; step: int | None; kind: str   # "periodic" | "final" | "best" | "best_face"
                                                                        # | "best_body" | "racer"
@dataclass class Run: path: Path; job: str; arch: str | None; config: dict; status: dict; probes: list[ProbePoint];
    checkpoints: list[Checkpoint]; probe_images: dict[int, list[Path]]; samples: list[Path];
    lr_series: list[tuple[int, float]]; loss_series: list[tuple[int, float]]; errors: list[str];
    finished: bool; race: dict | None
def find_runs(roots: list[Path]) -> list[Path]   # folders holding config.yaml plus status.json or likeness.jsonl
def default_roots(trainer_dir: Path | None) -> list[Path]   # <trainer>/output, ~/.rulz_vast/runs, ~/.rulz_vast/fleet/runs
def load_run(path: Path) -> Run                  # tolerant: missing files give empty fields, never raises on bad lines
def best_points(run: Run) -> dict[str, ProbePoint | None]   # "face", "body", "combined" (normalised scores preferred)

# diagnose.py
@dataclass class Finding: symptom: str; detail: str; evidence: dict; severity: str  # "info" | "warn" | "bad"
@dataclass class Proposal: symptom: str; fix: Fix; source_model: str | None   # None = general.yaml
def detect(run: Run) -> list[Finding]            # metric symptoms only (see list above); thresholds as module
                                                 # constants with a comment each
def propose(k: Knowledge, model_id: str, symptoms: list[str]) -> list[Proposal]
    # model flaws first, then general; a fix that sets a key an earlier fix already sets gets alternative_of = that
    # fix's index. The UI shows it indented under that fix, unticked, and ticking one unticks the other
def next_config(run: Run, proposals: list[Proposal], chosen: list[int]) -> dict
    # run.config with the chosen proposals' settings applied (op mul/add/min/max respected); job_name gets _v2, or the
    # next number when it already ends in _vN (example -> example_v2 -> example_v3). The Export field starts with it

# bridge.py
@dataclass class TrainerInstall: root: Path; python: Path; version: str | None
def find_trainer(hint: Path | None) -> TrainerInstall | None   # settings path, then the launcher's install folder
    # (%LOCALAPPDATA%/RulZ AI Tools/apps/rulz-trainer is NOT assumed: check settings, then sibling
    # ../RulzTrainer of this app, then ask); python = <root>/venv/Scripts/python.exe or venv/bin/python
def validate(inst: TrainerInstall, cfg_path: Path) -> tuple[bool, str]   # TrainConfig.load(path).validate() in the
    # trainer's python; returns (ok, message)
def queue_local(inst: TrainerInstall, cfg_path: Path, label: str) -> tuple[bool, str]  # TrainQueue(output).add(...)
def queue_vast(inst: TrainerInstall, cfg_path: Path, job: str, datasets: list[Path]) -> tuple[bool, str]
    # rulz_fleet.py queue add ... in the trainer's python; exit codes 0 ok, 1 refused, 2 timeout, 3 unreachable
def open_webui(inst: TrainerInstall) -> None     # starts run_webui.bat / run_webui.sh if not already running
# All subprocesses: list args (no shell), timeout, capture output, never block the UI thread (run in a thread /
# nicegui run.io_bound).

# advisor.py
PROVIDERS = ("claude", "codex", "grok", "agy")
@dataclass class AdvisorResult: provider: str; ok: bool; text: str; command: list[str]; seconds: float
def available() -> dict[str, Path | None]        # shutil.which per CLI (claude, codex, grok, agy)
def build_context(k, model_id, *, scan=None, rec=None, run=None, findings=None, proposals=None, question="") -> str
    # Markdown: the model's knowledge (summary, facts, base, flaws, open questions with confidence), dataset summary,
    # recommendation table, run analysis; no file paths outside the run/dataset, no personal details
def ask(provider: str, context: str, question: str, model: str | None = None, timeout: int = 600) -> AdvisorResult
    # writes context to a fresh temp folder as advisor_context.md and runs the CLI there, read-only:
    #   claude: claude -p "<instruction>" --permission-mode plan --output-format text [--model m]
    #   codex:  codex exec --sandbox read-only --skip-git-repo-check [-m m] "<instruction>"
    #   grok:   grok -p "<instruction>" --permission-mode plan --output-format plain [-m m]
    #   agy:    agy -p "<instruction>" --mode plan --output-format text [--model m]
    # <instruction> = "Read advisor_context.md in the working directory. Answer the question at its end. Do not
    #   edit or create files." Windows: resolve .cmd shims with shutil.which; no shell=True.
```

## Settings

`settings.json` in `%APPDATA%/RulZ-AI-Trainer-Optimizer/` (Windows) or `~/.config/rulz-ai-trainer-optimizer/`:
`trainer_dir`, `advisor` (`claude|codex|grok|agy|none`), `advisor_model` per provider, `export_dir` (default
`<trainer_dir>/configs` when known, else `~/Documents/RulZ AI Trainer Optimizer`), `run_roots` (extra folders),
`last` (last model, folders, goals). Written atomically.

## UI (NiceGUI, the suite's one look)

Dark default and light theme from the suite tokens (`--rz-*`, red `#ED000D`, Segoe UI Variable / Cascadia Code,
32/36/44 px buttons, 6/8/12 px radii, red primary button, 2 px focus ring); copy `rulz_hf_manager/ui/theme.py`.
Tabs: **Recommend** (model picker with summary, dataset folders + scan results and warnings, goals, the settings
table with value / why / confidence chip / sources, evaluation advice, Export panel: job name, base config, Save
YAML, Validate, Queue locally, Queue on vast.ai, Open RulZ AI Trainer), **Runs** (run list from roots, run detail:
likeness chart, best face/body/combined with probe images, findings, visual-flaw checkboxes, proposals with
checkboxes, Export next config; a Samples section with a step picker, the best checkpoint's step marked), **Ask AI**
(provider + model, context preview, question, answer), **Knowledge** (browse per model, the dataset rules, and the
problems `knowledge.validate()` found, read-only), **Settings**.

The knowledge files keep the codes `CONF`, `MEAS`, `DOC`, `REP`, `DISP`, `GUESS`. The UI never shows the codes: a chip
shows the word (Confirmed, Measured, Documented, Reported, Disputed, Guess) with an icon and a tooltip, and the
colours are suite tokens told apart by lightness. Rows that come from the trainer's setup wizard show "RulZ AI
Trainer setup wizard (webui.py)" as their source. A key the knowledge sets that RulZ AI Trainer does not have is
shown in the table, left out of the exported config, and named in the recommendation's warnings. Samples are read
from `samples/` and grouped by the step in their file name (`step000250_0.png`). A log line with an out-of-memory
error (or a status that says so) gives the `out_of_memory` finding and the general fixes for it.
