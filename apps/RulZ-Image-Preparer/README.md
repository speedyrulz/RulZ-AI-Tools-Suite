# RulZ Image Preparer

A Windows app for preparing image datasets before training models.

## Adjustments

Each adjustment has a **global** slider (main screen) that applies to every image, and the same
slider per image (click a photo) to override it for just that one. Overridden images get a red
**Custom** badge; the ↺ button next to a slider sends that image back to the global value.

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

## Using it

- Add photos with **Add images** or drag photos or whole folders onto the window.
- Click a photo to compare original vs. processed:
  - **Side by side** or **Slider** (drag the divider). Press `V` to switch.
  - Scroll to zoom, drag to pan, double-click for 100% / fit. Both sides stay in sync.
  - `←` / `→` move between photos, `Esc` goes back.
- Thumbnails only show color noise removal — JPEG blocks and fine detail are too small to see at
  that size. Open a photo to judge those.
- **Save all…** writes every image with its own settings to a folder you pick. File names and
  formats are kept; a file is never written over its own original (`_prepared` is added instead).

Supported: JPG, PNG, WebP, BMP, TIFF (8- and 16-bit, transparency kept for PNG/WebP/TIFF).

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

(`requirements-ai.txt` is optional — only for Best quality mode.)

## License

MIT for the code — see `LICENSE`. The RulZ name and logo are not covered; see `LOGO-NOTICE.md`.
FBCNN's network design and weights are by their authors under Apache-2.0.
