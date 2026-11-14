# RulZ Image Preparer

A Windows app for preparing image datasets before training models.

Part of the [RulZ AI Tools Suite](https://github.com/speedyrulz/RulZ-AI-Tools-Suite): the easiest way to install
it is the suite's launcher (under **AI Training**). Guide: [the wiki](https://github.com/speedyrulz/RulZ-AI-Tools-Suite/wiki/Image-Preparer).

## Adjustments

Each adjustment has a **global** slider (main screen) that applies to every image, and the same
slider per image (click a photo) to override it for just that one. Overridden images get a red
**Custom** badge; the ↺ button next to a slider sends that image back to the global value, and
**Use global for all** does it for all three. **Set all to 0** moves the global sliders to 0 (main screen) or
gives the open image its own 0 on all three (single-image view). **Reset all to global** on the main screen
clears every image's own values.

| Adjustment | What it does |
|---|---|
| **Color noise removal** | Removes colored speckles (red/green/blue blotches) from low-light and high-ISO photos. Only color is smoothed; brightness and detail are untouched. |
| **JPEG artifact removal** | Removes compression damage: 8×8 blocks, halos/ringing around edges, blocky color. |
| **Detail recovery** | Sharpens fine detail back up after artifact removal. Tiny differences (leftover noise) aren't boosted. |

JPEG artifact removal has two modes (switch next to its slider):

- **Balanced** — a fast classic filter (edge-preserving smoothing + color cleanup). Instant previews.
- **Best quality (AI)** — [FBCNN](https://github.com/jiaxi-jiang/FBCNN) (Jiang et al., ICCV 2021,
  Apache-2.0), a neural network made for JPEG artifact removal, run on your NVIDIA GPU. Much
  cleaner results; about 1 s per 2 MP and 6–8 s per 24 MP photo. The first time you pick it, the
  app offers a one-time setup: PyTorch with CUDA (~3 GB) and the model file (288 MB, checksum
  verified), stored in `%LOCALAPPDATA%\RulZ\Image Preparer\models`.

## Shadow areas

Open a photo and paint over a shadow to brighten it to the light level around it:

- **Paint** (`B`) and **Erase** (`E`) — left-drag on the picture; right-drag still moves it.
  **Brush** sets the size (`[` / `]`), **Undo** (`Ctrl+Z`) takes back the last stroke, **Clear**
  removes them all. **Show areas** tints what you painted red on the original.
- **Shadow lift** — how far the shadow is brightened (100 matches its surroundings).
- **Edge blend** — how softly the painted area fades into the rest of the picture.

It doesn't just raise brightness: the app separates the lighting from the surface detail, measures
how deep the shadow is, and undoes it the way extra light would, so texture and contrast are kept,
dark and bright things in the shadow keep their difference, lit parts you brush over are left alone,
the shadow's soft edge is brightened only as much as it was darkened (so brushing past it onto open lit
ground leaves no bright ring), and a blue shadow tint is matched to the surroundings. Painted areas are per photo and are applied by
**Save all** and by Auto crop's clean-up option. They aren't kept after the app closes.

## Auto crop

Crops pictures to something you describe, such as `face` or `dog`, found by AI. Select pictures
(Ctrl+click or Shift+click; with none selected, all are used) and click **Auto crop…**:

1. Type what to crop.
2. **Preview** finds it with [SAM 3.1](https://ai.meta.com/sam3/) (Meta's Segment Anything 3.1) and
   shows the found area in red and the crop as a dashed box. Use the strip below to check each picture.
   Crop settings update the box straight away; changes to the words or the "Finding it" settings need
   another Preview (a Preview still running stops after the picture it's on, so you can start the new one).
3. **Crop…** asks for a folder, then crops every selected picture and saves it there with the same name as
   **Save all** would use (`_prepared` is added rather than writing over any picture in the list, selected or
   not), asking first
   if files of those names already exist. Pictures where nothing was found are skipped and listed.

| Setting | What it does |
|---|---|
| **Matches** | **All matches** crops to cover every match (every face); **Best match only** to the most confident one. |
| **Confidence** | How sure SAM must be that something matches (default 0.5). Lower finds more, including wrong things. |
| **Mask threshold** | How sure SAM must be about a pixel for it to count as part of the match (default 0.5). Lower = bigger area. |
| **Padding** | Room left around what was found: **Same on every side** (one %) or **Each side separately** (top, bottom, left, right). Default 10%. |
| **Measured against** | What the % is of: **The final crop** (default; what was found sits inside a border of that share), **The found area**, or **The whole picture**. |
| **Shape limit** | Never crop narrower or wider than this shape (default 2:3 / 3:2). A too-tall crop is widened, too wide made taller; it only grows. |
| **Divisible by** | Grow the crop so width and height are multiples of this (e.g. 64 for training). 1 = off. |
| **Also remove noise and JPEG damage first** | Applies each picture's clean-up settings before cropping, so one press cleans and crops. |

The crop follows the [RulZ Mask Intelligent Crop](https://github.com/speedyrulz/ComfyUI-RulZnodes#rulz-mask-intelligent-crop)
ComfyUI node exactly. The first time you use Auto crop, the app offers a one-time setup: PyTorch with CUDA
(~3 GB, shared with Best quality), Transformers and TorchVision, and the SAM 3.1 model (1.7 GB, from
[huggingface.co/unsloth/sam3.1](https://huggingface.co/unsloth/sam3.1), no sign-in needed, checksum
verified), stored in `%LOCALAPPDATA%\RulZ\Image Preparer\models\sam3.1`. SAM loads each time you open
Auto crop (closing it frees the video memory), so the first Preview takes up to half a minute; after that it's well under a second per picture on an
NVIDIA GPU (about 2 GB of video memory).

## Using it

- Add photos with **Add images** or drag photos or whole folders onto the window.
- Click a photo to compare original vs. processed:
  - **Side by side** or **Slider** (drag the divider). Press `V` to switch.
  - Scroll to zoom, drag to pan, double-click for 100% / fit. Both sides stay in sync.
  - `←` / `→` move between photos (right after you drag a slider, they nudge that slider instead; click the
    picture to go back to moving between photos), `Esc` goes back.
- Thumbnails only show color noise removal and painted shadow areas — JPEG blocks and fine detail are too small to see at
  that size. Open a photo to judge those.
- **Save all…** writes every image with its own settings to a folder you pick. File names and
  formats are kept; a file is never written over any picture in the list (`_prepared` is added instead).
  Pictures with nothing to change are copied as they are, so they lose nothing to re-compression.
  JPEGs are saved at quality 97 with full color resolution (4:4:4), so the color edges the clean-up
  restores aren't blurred again. A photo's color profile (Display P3, Adobe RGB…) is kept in JPEG, PNG
  and WebP files, so it looks the same in other apps; other details (camera, date, location) are only
  kept in pictures copied as they are.

Supported: JPG, PNG, WebP, BMP, TIFF (8- and 16-bit, transparency kept for PNG/WebP/TIFF; a gray TIFF
with transparency loses it when it's cleaned up, and a cleaned-up TIFF loses its color profile).

## Running

Double-click `Launch RulZ Image Preparer.bat`. The first run creates a `.venv` and installs the
requirements (Python 3.10+ needed).

Or manually:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r requirements-ai.txt
.venv\Scripts\python run.py
```

(`requirements-ai.txt` is optional — only for Best quality mode and Auto crop. Auto crop also needs
`torchvision` from the same CUDA index, then `requirements-crop.txt` from PyPI, in two separate installs:
with both indexes in one install, pip takes PyPI's CPU-only PyTorch.)

Tests: `.venv\Scripts\python -m unittest discover tests`

## License

MIT for the code — see `LICENSE`. The RulZ name and logo are not covered; see `LOGO-NOTICE.md`.
FBCNN's network design and weights are by their authors under Apache-2.0. SAM 3.1 is by Meta under the
SAM License, which comes with the model download.
