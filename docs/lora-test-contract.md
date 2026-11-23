# LoRA tests: the contract

How a LoRA test is described, run, saved and reached from another computer. `rulztrainer/loratest.py` (the run),
`rulztrainer/loratest_export.py` (the files a test exports), the WebUI's Test LoRAs tab and `scripts/vast/rulz_agent.py`
(the instance side) all follow it. Scope and reasons: `lora-tester-scope.md`.

## 1. Words

- **test**: one comparison, kept in `output/_tests/<test_id>/`. `test_id` = `t` + UTC `YYYYMMDD-HHMMSS` + `-` + 4
  lowercase hex characters (e.g. `t20261122-153012-a3f9`). Only `[a-z0-9-]`; anything else is refused.
- **contestant**: one LoRA file from one run (or a soup made in this test). Each has an `id` (`c1`, `c2`, ...; a soup
  `s1`, `s2`, ...), a `label` shown in the grid, and the run's config.
- **cell**: one picture: (prompt index, seed, contestant id, strength).
- **strength**: the LoRA's multiplier while rendering (`lora.set_multiplier` on the Renderer's own `injected` dict only,
  never the sampling LoRAs). 1.0 is exactly the probes' render. Files are never rewritten.

## 2. The spec: `output/_tests/<test_id>/spec.json`

Written once by whoever starts the test (the tab, or `rulz_agent.py test-start`), never changed afterwards. UTF-8,
keys as below; unknown keys are kept and ignored.

```json
{
  "version": 1,
  "test_id": "t20261122-153012-a3f9",
  "name": "Elena: best vs final",
  "created": "2026-11-22T15:30:12Z",
  "contestants": [
    {"id": "c1", "label": "elena_v3 best", "lora": "output/elena_v3/elena_v3_best.safetensors",
     "config": "output/elena_v3/config.yaml", "run_id": "elena_v3-20261120-101500"},
    {"id": "c2", "label": "elena_v3 final", "lora": "output/elena_v3/elena_v3.safetensors",
     "config": "output/elena_v3/config.yaml", "run_id": "elena_v3-20261120-101500"}
  ],
  "soups": [
    {"id": "s1", "label": "soup c1+c2", "of": ["c1", "c2"], "weights": null}
  ],
  "prompts": ["a photo of elena, headshot, ...", "a photo of elena, full body, ..."],
  "prompt_kinds": ["face", "body"],
  "seeds": [1, 2],
  "strengths": [0.8, 1.0, 1.2],
  "render": {"width": null, "height": null, "steps": null, "cfg": null, "negative": null},
  "scoring": {"face_refs": null, "body_refs": null, "from_contestant": "c1"},
  "devices": ["cuda:0"]
}
```

- `lora` and `config` paths: relative to the install folder (the folder holding `webui.py`), or absolute. A relative
  path must not climb out of it (`..`).
- `run_id`: the Previous Runs record it came from, or `null`.
- `soups[].weights`: `null` = equal weights; else one number per `of` entry. A soup's `of` names only file contestants
  (no soup of soups) of the same base model.
- `prompt_kinds[i]`: `"face"`, `"body"` or `"both"`: which likeness score the picture gets (as `score_folders.py`'s
  kinds). Same length as `prompts`.
- `render`: `null` fields take the first contestant's sample settings (its config's sample size, steps, guidance,
  negative). One set for the whole test, never per contestant.
- `scoring`: `face_refs` / `body_refs` lists override; `null` takes them from `from_contestant`'s config
  (`face_refs`, `body_refs`). No references at all: the test still renders and the scores are `null`.
- `devices`: the GPUs to render on (`cuda:N`); two or more use `renderpool.RenderPool`.

**Refusals before any GPU work** (the run writes `status.json` with `state: "refused"` and a `reason`, exit status 1):
a missing LoRA or config file (named); contestants whose configs differ in base model (`model_type` / `arch`,
`model_source` / model path, `sample_model_path`, quant, `network_type`); a soup of different base models; empty
`prompts`, `seeds` or `strengths`; a strength outside 0..2; more than 2000 cells; a bad `test_id`.

## 3. Running: `python -m rulztrainer.loratest run <test_dir>`

- Runs from the install folder. One process per test; it owns the GPUs in `devices` until it ends.
- Order: refusals → cook soups (`soup.make_soup`, saved as `<test_dir>/soups/<id>.safetensors`) → cache the prompts'
  text embeds (`prepare_prompts`: the Renderer has no text encoder and reads each prompt and the negative from the text
  cache; missing ones are encoded with the first contestant's config, then the encoder is unloaded) → load the
  Renderer (`renderhelper.load_renderer` with the first contestant's config) → for each contestant: put its file in
  the Renderer's slots (plain LoRA / LoKr files and soups as a state built from the file, which `mirror_lora` copies in
  and resizes where the rank differs; OFT / DoRA through `lora.load_lora_weights`; layers a file lacks are zeroed),
  then for each strength: `set_multiplier`, render every prompt x seed → score every picture → write `results.json` →
  `state: "done"`. With two or more devices the strength is folded into each layer's scale in the state sent to the
  pool workers (plain LoRA / LoKr only), and a STOP waits for the round (one contestant at one strength) in hand.
- More refusals (before any GPU work), beyond section 2's: contestants whose configs differ in LoRA layout
  (`lora_targets`, `lora_blocks`, `lora_rank`, `lokr_factor`, `oft_block_size`); a duplicate seed, or two strengths equal
  at two decimals (their pictures would overwrite each other); a soup with a kohya (`lora_unet_*`), OFT or DoRA
  ingredient; two or more devices with a kohya file (kohya files load on one device only, through
  `lora.load_lora_weights` after the slots are restored to the config's rank); a config with `probe_frames` above 1 (clip models: a
  test renders single pictures only); two or more devices with an OFT or DoRA config.
- The `"run"` scale (section 4) also needs the `likeness.jsonl` of `from_contestant`'s run next to that contestant's
  config, readable (`core.job_scale`), with its recorded face model, body score and gold recognizer matching the
  config (`face_gold: auto` follows the recorded one); otherwise the scale is `"references"` with a note, never a failure.
- Two or more devices also need every contestant and soup to cover the same layers (the workers keep their previous
  weights in a layer a state lacks), else the test is refused (`state: "refused"`, found after the soups are cooked,
  before any GPU work). `dora` is one of the base-model keys compared. A STOP that arrives during the last batch still
  ends the test as stopped. Scores are aggregated and ranked at full precision; only the written numbers are rounded
  (two decimals).
- Pictures: `<test_dir>/images/<contestant_id>/s<strength>_p<prompt_index>_seed<seed>.png`, strength written with two
  decimals (`s1.00`).
- Stop: a file named `STOP` in `<test_dir>`; the run finishes the picture in hand, scores what it has, writes
  `results.json` with `"complete": false` and ends `state: "stopped"`.
- Other commands: `python -m rulztrainer.loratest new --spec <file>` (validates, creates the folder, prints the
  `test_id`; `spec.json`'s `test_id` is filled when missing) and `python -m rulztrainer.loratest list` (one JSON line per
  test folder: id, name, state, created).

### `status.json` (rewritten atomically, `atomic.py`, at least every 5 s while running)

```json
{"state": "rendering", "test_id": "...", "pid": 1234, "started": "...", "updated": "...",
 "done": 37, "total": 96, "current": "c2 at 1.20, prompt 1, seed 2", "eta_s": 410,
 "reason": null, "error": null}
```

`state`: `queued` (folder made, not started) → `refused` | `loading` → `rendering` → `scoring` → `done` | `stopped` |
`failed` (`error`: the last lines of the traceback). A `pid` that no longer runs while the state is not final means it
died: readers show it as failed.

## 4. Results: `<test_dir>/results.json`

```json
{
  "version": 1, "test_id": "...", "complete": true, "scale": "run" ,
  "scale_note": null,
  "contestants": [{"id": "c1", "label": "...", "lora": "...", "sha256": "...", "kind": "file"},
                  {"id": "s1", "label": "...", "lora": "soups/s1.safetensors", "sha256": "...", "kind": "soup",
                   "of": ["c1", "c2"], "weights": [0.5, 0.5]}],
  "cells": [{"contestant": "c1", "strength": 1.0, "prompt": 0, "seed": 1,
             "image": "images/c1/s1.00_p0_seed1.png",
             "face": 61.2, "body": 55.0, "gold": 58.4, "combined": 59.3}],
  "columns": [{"contestant": "c1", "strength": 1.0, "face": 60.1, "body": 54.2, "gold": 57.9, "combined": 58.3,
               "n": 4}],
  "best_strength": {"c1": 1.0, "s1": 0.8},
  "winner": {"contestant": "c1", "strength": 1.0, "combined": 58.3},
  "render": {"width": 1024, "height": 1024, "steps": 25, "cfg": 3.5, "seconds": 812.4, "gpus": ["RTX 5060 Ti"]}
}
```

- Scores are the trainer's normalized likeness numbers (0-100, `null` when not scored). `combined` = the trainer's
  face / body weighting (`likeness.combine`).
- `scale`: `"run"` when every contestant's run shares `face_model`, `body_score` and gold face and the references come
  from that run's config (the run's own scale); else `"references"` (the references' own scale) and `scale_note` says
  why. Scores on the two scales are not comparable.
- `columns`: means over a column's scored cells; `n` = how many. `best_strength`: per contestant, the strength with the
  highest column `combined`; `winner`: the highest column `combined` overall. Ties: the lower strength, then the earlier
  contestant.

## 5. Exports (`rulztrainer/loratest_export.py`, into `<test_dir>/export/`)

Read only `spec.json`, `results.json` and the pictures; need no GPU and no model.

- `grid.jpg`: rows = prompt x seed, columns = contestant x strength, a header row with labels and strengths, each cell's
  combined score under it, the winner's column outlined, a left gutter naming each row (`p<index> <kind>, seed <seed>`).
  Pictures scaled to a cell size (default 256 px). A grid larger than 8 x 8 cells is split into pages of at most 8
  columns and 8 rows, each with its own header and gutter, written as `grid_01.jpg`, `grid_02.jpg`, ... instead
  (earlier `grid*.jpg` files are removed first). A cell's `image` must resolve inside the test folder, else it is drawn
  as missing.
- `scores.xlsx`: sheet `Cells` (one row per cell), sheet `Columns` (one row per column, winner bold), sheet `Test`
  (the spec's settings).
- `report.pdf`: first page the winner and the column table, then the grid, in `runreport.py`'s style.

## 6. On a vast.ai instance (`scripts/vast/rulz_agent.py`)

The PC never starts processes on the instance directly; it pipes `rulz_agent.py` over ssh as for every other call.

| Command | Does | Prints (JSON) |
| --- | --- | --- |
| `test-start` (spec JSON on stdin) | refuses when a training job is running on a GPU in `devices` or another test is running; else writes the folder (via `loratest new`) and starts `loratest run` detached (`nohup`, own process group), logs to `<test_dir>/run.log` | `{"test_id": ..., "state": "queued"}` |
| `test-status [<test_id>]` | `status.json` of one test, or of all (newest first) | the status, or `{"tests": [...]}` |
| `test-stop <test_id>` | writes `STOP` | `{"test_id": ..., "stopping": true}` |
| `files --tests <test_id>` | lists `output/_tests/<test_id>/` for `rulz_remote.py pull`, as the existing `files` listing does | as `files` |

Paths in a spec sent to an instance are relative to the instance's install; the PC builds them from the instance's own
job folders (`rulz_agent.py jobs` / `files`), never from the PC's download copies. `rulz_remote.py` gets matching
`test-start --spec <file>`, `test-status`, `test-stop` and `pull --tests <test_id>` (downloads into the Fleet download
folder under `<instance id>/_tests/<test_id>/`).
