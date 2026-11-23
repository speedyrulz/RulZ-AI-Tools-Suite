# LoRA Tester: scope (2026-11-22, built in 4.4.0)

A **Test LoRAs** tab in the RulZ AI Trainer WebUI. It renders finished LoRAs from your own runs side by side on fixed
prompts and seeds, across strengths, scores every picture on the trainer's likeness scale, can average chosen LoRAs
into a seed soup and test that too, and exports the result.

Decisions taken (owner, 2026-11-22): a tab in RulZ AI Trainer, not a separate app; RulZ AI Trainer runs only; renders on
this PC's GPUs or on a vast.ai instance; first version has the grid with likeness scores, a strength sweep, seed soup
and export.

## Goal

Answer "which of my finished LoRAs is best, and at what strength?" with the same renderer and the same scores as the
run's own probes, so a test's numbers can be read next to the Monitor's and Previous Runs'.

## Out of scope (v1)

- LoRAs from other trainers (kohya, ai-toolkit, downloads). `score_folders.py` already scores their pictures.
- Rendering through ComfyUI.
- Comparing different base models in one grid (see decision D1).
- Editing or deleting LoRA files. The tester only reads them; soups are written as new files.

## What already exists (reused, not rebuilt)

| Need | Existing code |
| --- | --- |
| Render a LoRA exactly as the probes do, any model family | `probeworker.Renderer` (built from a run's config, `render(jobs, payload)`) |
| Split a render over several local GPUs | `renderpool.RenderPool` (one process per GPU, `real_worker_class` seam for fakes) |
| Load a saved LoRA file into the slots | `lora.load_lora_weights(injected, path)`, `lora.read_metadata` |
| Likeness scores, normalization, gold face | `likeness.py`, `facesim.py`, `bodysim.py`, `adaface.py` |
| Probe prompt presets, own presets | `configs/probe_presets.json`, the Wizard's preset picker |
| Seed soup | `soup.make_soup` |
| Spreadsheet of scores per run | `score_folders.build_workbook` |
| Report PDF layout | `runreport.py` |
| List of runs, labels, vast imports | `history.list_runs`, `history.set_label`, `history.import_fleet_runs` |
| Reach a vast instance, download its files | `rulz_fleet.py`, Fleet downloads (3.39) |

## How a test works

1. **Contestants.** Pick runs from Previous Runs; for each, pick checkpoints (final, `_best`, step saves) found in its
   job folder or its Fleet download folder. A contestant is (run, LoRA file, strength). Each carries its run's
   `config.yaml`.
2. **Prompts and seeds.** Probe presets, own presets or typed prompts; a fixed seed list; size, steps and CFG default to
   the first run's sample settings and can be overridden for the whole test (never per contestant, so the grid is fair).
3. **Strengths.** A list such as 0.6 / 0.8 / 1.0 / 1.2, applied to every contestant.
4. **Soup (optional).** Pick 2+ contestants of the same base model, equal weights (or greedy, using this test's scores
   once rendered); the soup is saved under the test folder and joins the grid as one more contestant.
5. **Render.** One Renderer per GPU for the test's base model; each contestant's LoRA is loaded into the slots, scaled,
   and its pictures rendered. Local: the render pool over the GPUs picked. vast.ai: the instance runs the same test (see
   "vast.ai" below).
6. **Score.** Every picture against the face / body references (the first run's by default, or picked). Contestants from
   runs with a different `face_model`, `body_score` or gold face are scored on the references' own scale, as the ComfyUI
   likeness nodes do, and the page says so.
7. **Results.** Grid: rows prompt x seed, columns contestant x strength; each cell shows face / body / gold / combined.
   Column means, best strength per contestant, overall winner highlighted. "Pick" puts a label on that run in Previous
   Runs (`set_label`); nothing is moved.
8. **Saved.** `output/_tests/<test-id>/`: `test.json` (settings, contestants, file hashes, scores), the pictures, any
   soup. A saved test reopens without rendering again.
9. **Export.** Grid as one labelled image, scores as .xlsx, a PDF in the run report's style.

## vast.ai

A vast run's LoRAs are usually still on the instance that trained them, so the test goes to the instance rather than the
files coming here:

- The PC's Fleet card gets a "Test LoRAs" action; it sends the test spec (no files) to the instance's WebUI, which runs
  it with its own GPU through the same code path as a local test.
- The PC follows progress and, when it ends, downloads `output/_tests/<test-id>/` with the existing Fleet download code.
- If the instance no longer has a file (destroyed and recreated), the test refuses that contestant by name. Uploading
  LoRAs back to an instance is left for later.

This adds a PC <-> instance interface (start, status, cancel, result), so its contract goes to the principal engineer
before it is built (see the build plan).

## Decisions settled by default (change any before building)

- **D1. One base model per test.** All contestants must share the sampling model, arch and quant, so one Renderer
  serves the whole test. Mixing models would mean loading one model after another and grids that are not comparable.
- **D2. No test while a training job uses the same GPU.** The tab refuses the GPU a running job (or its probe worker)
  holds and offers the others.
- **D3. Strength is a scale on the LoRA's change at render time**, never a rewritten file. It must be exactly neutral at
  1.0 (bitwise the same pictures as today's probes).
- **D4. Pick = a label in Previous Runs.** No copying or renaming files.

## Risks and limits

- **VRAM.** The Renderer loads the whole sampling model on one GPU, without block swap. MiniMax-H3, Ideogram 4 and
  other large models may not fit on a 16 GB card; the tab checks the run's measured peak (`measured.py`) and warns
  before starting, and suggests the vast route.
- **`lora.py` was not changed.** Strength uses the existing `lora.set_multiplier` (ComfyUI-style, neutral at 1.0) on
  the Renderer's own slots; with two or more GPUs it is folded into each layer's scale, which is the same for LoRA and
  LoKr.
- **LoKr / DoRA.** Scaling a LoKr or DoRA change is not the same arithmetic as a plain LoRA; each kind needs its own
  test.
- **Job folders get reused.** A run's LoRA files may be gone even though its history record is there; those runs show
  as "no files" instead of failing.
- **Speed.** contestants x strengths x prompts x seeds renders add up fast (5 x 4 x 4 x 2 = 160 pictures); the page
  shows the count and an estimate from the run's measured render time before starting.
- **Never run for real yet.** Like the other recent features, v1 will be unit-tested with a fake Renderer; a real render
  goes to a vast.ai instance first. Local GPU runs during development need the owner's OK each time.

## Build (team, 2026-11-22)

Level 2, one independent review per part by another provider than its author. No `lora.py` change was needed, so the
level-3 part and the principal-engineer question were dropped: the instance interface follows the existing
`rulz_agent.py` pattern (`docs/lora-test-contract.md`).

| Part | Files | Built by | Reviewed by |
| --- | --- | --- | --- |
| Test engine | `rulztrainer/loratest.py`, `tests/test_loratest.py` | Sonnet | Codex |
| Exports | `rulztrainer/loratest_export.py`, `tests/test_loratest_export.py` | Antigravity | Codex |
| Instance commands | `scripts/vast/rulz_agent.py`, `scripts/vast/rulz_remote.py`, `tests/test_loratest_remote.py` | Codex | Grok |
| Test LoRAs tab, Run on instance | `webui.py`, `tests/test_webui_loratest.py` | Sonnet | Grok |

The saved test is `spec.json` + `status.json` + `results.json` (not one `test.json`), as the contract describes.
