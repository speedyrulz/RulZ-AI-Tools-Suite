# RulzTrainer likeness scoring for ComfyUI

Score any images against a person's reference photos exactly as the RulzTrainer trainer scores its likeness probes
and samples: the same face model, the same body score, the same normalized scale and the same face / body weights.
The nodes import the trainer's own scoring code from a RulzTrainer checkout, so the numbers are the trainer's.

## Install

Link this folder into ComfyUI's `custom_nodes` from the RulzTrainer checkout (it finds the checkout through the
link), then restart ComfyUI:

```bash
ln -s /mnt/c/ai/RulzTrainer/comfyui/rulztrainer_likeness ~/ComfyUI/custom_nodes/rulztrainer_likeness
```

On Windows: `mklink /J C:\path\to\ComfyUI\custom_nodes\rulztrainer_likeness C:\ai\RulzTrainer\comfyui\rulztrainer_likeness`.
A copied folder instead needs the checkout's path in `repo_path.txt` (one line) next to `__init__.py`, or in the
`RULZTRAINER_DIR` environment variable.

ComfyUI's Python needs `insightface`, `onnxruntime` (or `onnxruntime-gpu`), `mediapipe`, `transformers` and, for
.xlsx files, `openpyxl` (`requirements.txt`). mediapipe 1.1 or newer is recommended: 1.0.0 and 1.0.1 crash on some
image widths when reading the person mask, which RulzTrainer works around by padding those images by up to 3 pixels,
so their body scores can differ from other machines by a hair.

The first run downloads InsightFace `buffalo_l` (into `~/.insightface`), DINOv2 base (into the Hugging Face cache)
and the MediaPipe pose model (into the checkout's `models/`), as the trainer does.

Load `workflows/RulzTrainer Likeness Scoring.json` for a ready-made workflow.

## Nodes

**RulzTrainer Likeness References**: who the person is.
- `reference_paths` / `reference_images`: references for both scores (image files or folders, one per line, and/or
  connected images).
- `face_paths` / `face_images` and `body_paths` / `body_images`: separate face or body references. Given, they replace
  the shared ones for that score, like `face_refs` and `body_refs` in the trainer.
- `body_score`: `appearance` (the trainer's default: a DINOv2 embedding of the body from the neck to mid-thigh, cut out
  and turned upright) or `proportions` (body ratios from the pose skeleton, plus SAM 3 silhouette widths with
  `body_backend: mediapipe+sam3`; SAM 3 is gated on Hugging Face).
- `face_provider`, `face_det_size`, `body_device`: as the trainer's settings of the same names.

**RulzTrainer Likeness Score**: scores the images and averages them.
- `images` and/or `image_paths`: one image, a batch, files or folders.
- `face_weight` / `body_weight`: 0.7 / 0.3 as in the trainer.
- `normalize` (on): 0 = what a stranger scores, 100 = what a typical reference scores against the others, capped at
  100. The 100 is measured from the references (2 or more); the 0 is a default stranger score unless:
  - `baseline_paths` / `baseline_images`: images from the base model without the LoRA. Their mean score is the 0,
    as the trainer's step-0 probe sets it.
  - `trainer_run`: a RulzTrainer job folder. Its `likeness.jsonl` gives the exact scale that run used.
  - `face_floor`, `body_floor`, `face_ceiling`, `body_ceiling`: raw values typed in (-1 = automatic).

  Typed values win over the run's, the run's over the baseline and the references, and those over the defaults.
- Outputs:
  - `summary`: overall, face and body averages.
  - `details`: one line per image.
  - `overall`, `face`, `body` as numbers, -1 when there is no score.
  - `report`: for the export node.

The averages follow the trainer's probes: the raw face scores of the images with a face are averaged, likewise the
body scores; each average is then normalized and the two are combined. An image without a face, or without a usable
body, simply has no score of that kind (a headshot has no body score: the body window needs shoulders and hips).

**RulzTrainer Likeness Export**: with `export_spreadsheet` on, writes to ComfyUI's output folder:
- an `.xlsx` with three sheets:
  - **Summary**: the averages, the mean, lowest, highest and spread of the images' own scores, and the scale used.
  - **Images**: every image's normalized and raw scores, faces found, the closest face reference, and the file.
  - **References**: which references were usable.
- or the same as three CSV files.

Paths are as the ComfyUI process sees them: a ComfyUI running in WSL reads `C:\ai` as `/mnt/c/ai`.
