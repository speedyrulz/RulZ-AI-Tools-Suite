# RulzTrainer likeness scoring for ComfyUI

Score any images against a person's reference photos exactly as the RulzTrainer trainer scores its likeness probes
and samples: the same face model, the same held-out gold face (AdaFace, or FaceNet), the same body score, the same normalized
scale and the same face / body weights.
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

The first run downloads InsightFace `buffalo_l` (into `~/.insightface`), DINOv2 base (into the Hugging Face cache),
the MediaPipe pose model and the gold face's weights (AdaFace ViT, about 460 MB, or FaceNet VGGFace2; into the checkout's `models/`), as the trainer does.

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
- `face_model` (optional, `buffalo_l`): the trainer's face score model, `buffalo_l` (ArcFace R50) or `antelopev2`
  (ArcFace R100, about 360 MB, downloaded once). The two score on different scales: with a run's scale loaded, pick
  the run's own model. A run of the other model (or of another `body_score`) keeps the references' own face (or
  body) scale; runs from before 3.19.0 count as `buffalo_l`.
- `gold_face` (`adaface`): the trainer's held-out gold face (`face_gold`). AdaFace ViT-Base (the trainer's default
  since 3.14.0) or FaceNet (VGGFace2, the one before), which nothing trains on, scores the same detected face against its own embeddings of the references. A run that trains against ArcFace
  (the reward stage, the identity loss on its arcface backend, identity alignment) can raise the ArcFace face score
  without raising the likeness; the gold face shows when that happens. `off` skips it. With a run's scale
  loaded, pick the run's own gold scorer: a run's FaceNet scale is not used for AdaFace scores (or the reverse).

**RulzTrainer Likeness Score**: scores the images and averages them.
- `images` and/or `image_paths`: one image, a batch, files or folders.
- `face_weight` / `body_weight`: 0.7 / 0.3 as in the trainer.
- `normalize` (on): the trainer's scale. 0 = what a stranger scores. The references' own level (a typical
  reference's best match among the others, measured from 2 or more references) is stretched 1.35x above the stranger
  and the scale curved by 1.5, so the references' level reads about 64 and a score still has room to grow past it;
  capped at 100. The body's 100 also sits at least halfway from the stranger to a perfect match. The gold face has
  its own scale, shaped like the face's. The 0 is a default stranger score unless:
  - `baseline_paths` / `baseline_images`: images from the base model without the LoRA. Their mean score is the 0,
    as the trainer's step-0 probe sets it.
  - `trainer_run`: a RulzTrainer job folder. Its `likeness.jsonl` gives the exact scale that run used.
  - `face_floor`, `body_floor`, `face_ceiling`, `body_ceiling`, `gold_floor`, `gold_ceiling`: raw values typed in
    (-1 = automatic). A typed ceiling is the final one, as the trainer logs it (already stretched).

  Typed values win over the run's, the run's over the baseline and the references, and those over the defaults.
- `overall_face`: which face goes into the overall score. `gold` puts the gold face in ArcFace's place, as the
  trainer's `face_gold_select` does whenever its gold scorer is on; `arcface` keeps ArcFace's, as a run without the
  gold scorer; `auto` follows the `trainer_run`, else ArcFace. The overall with ArcFace's face is always in the
  details and the spreadsheet.
- Outputs:
  - `summary`: overall, face, gold face and body averages.
  - `details`: one line per image. A face is left unscored, as in the trainer's probes, when a second face is at
    least 60% the size of the largest or the largest is under 48 pixels; the line and the spreadsheet say why.
  - `overall`, `face`, `body`, `gold_face` as numbers, -1 when there is no score.
  - `report`: for the export node.

The averages follow the trainer's probes: the raw face scores of the images with a face are averaged, likewise the
gold face and body scores; each average is then normalized and they are combined. An image without a face, or without
a usable body, simply has no score of that kind (a headshot has no body score: the body window needs shoulders and
hips).

**RulzTrainer Likeness Export**: with `export_spreadsheet` on, writes to ComfyUI's output folder:
- an `.xlsx` with three sheets:
  - **Summary**: the averages, the mean, lowest, highest and spread of the images' own scores, and the scale used.
  - **Images**: every image's normalized and raw scores (ArcFace and gold face), faces found, the closest face
    reference, and the file.
  - **References**: which references were usable.
- or the same as three CSV files.

Paths are as the ComfyUI process sees them: a ComfyUI running in WSL reads `C:\ai` as `/mnt/c/ai`.
