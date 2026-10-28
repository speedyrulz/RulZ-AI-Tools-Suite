# RulZ Image Organizer

A Windows desktop photo organizer that works like Google Photos' People view: add folders to your
Library, and RulZ finds every photo, detects the faces in them, and groups them by person automatically.
You name people and fix any mistakes; your corrections are always kept.

Everything runs on your PC. No photos or face data ever leave your computer.

**[Download](https://github.com/speedyrulz/RulZ-Image-Organizer/releases/latest)** · **[Wiki: guides, FAQ and troubleshooting](https://github.com/speedyrulz/RulZ-Image-Organizer/wiki)**

| Dark (default) | Light |
| --- | --- |
| ![Photos timeline in dark mode](docs/screenshots/01-photos-dark.png) | ![Photos timeline in light mode](docs/screenshots/02-photos-light.png) |

![Photo viewer with the info panel](docs/screenshots/03-photo-viewer.png)

![Search filters for expression, head direction and pose](docs/screenshots/04-search-filters.png)

<sub>Screenshots show generated demo pictures, not real photos.</sub>

## Features

- **Library**: add any number of folders (subfolders included). Scans run in the background on startup,
  or when you click **Scan now** (F5). New, changed, deleted and *moved/renamed* photos are picked up;
  moved files keep their face tags.
- **Photos**: a timeline of all photos grouped by month (from EXIF date taken). Ctrl + mouse wheel zooms
  the grid. Search by person, facial expression, head direction and body pose. Words in one
  comma-separated group describe one person:

  | Search | Finds |
  | --- | --- |
  | `Mom` | photos of Mom |
  | `Mom, Dad` | photos with both |
  | `smiling` | anyone smiling |
  | `Mom smiling, Dad` | Mom smiling, with Dad somewhere in the photo |
  | `Dad facing left` | Dad's head turned towards the left side of the photo |
  | `looking down`, `side profile`, `head tilted` | anyone posed that way |
  | `Mom sitting`, `standing`, `lying down`, `arms raised` | body pose |

  Expressions: smiling, neutral, surprised, sad, angry, disgusted, scared, smirking (and "not smiling").
  Directions: facing camera, facing left/right, looking up/down, side profile, head tilted.
  Poses: standing, sitting, lying down, arms raised.
  The **Filters** button shows the same options as clickable chips; they apply to the people you searched
  for, or to anyone if the search box is empty. The photo viewer lists each face's expression, direction
  and pose. These are estimated automatically and won't always be right (a smirk may read as a smile, and
  up/down is less reliable than left/right). Pose is only decided when the person's lower legs are in the
  photo, so head-and-shoulders and waist-up shots get no pose, and only people whose face was found are
  covered.
- **People**: automatic face groups. Click a group to name it. Select several groups (hover → tick the
  circle) and **Merge** them. Naming a group with an existing name offers to merge them. Hide people you
  don't care about. **Ungroup** dissolves a wrong group.
- **Corrections** (on a person's page):
  - **Faces** tab: select faces → **Not {name}**, **Move to…**, **Use as profile photo**, **Not a face**.
  - **Review suggestions** tab: unassigned faces that look like this person → **Yes** / **No**.
- **Photo viewer**: face boxes with names. Click any face to say who it is, confirm it, remove it from a
  person, or mark it "not a face". **Tag a face** (person+ icon) lets you drag a box around a face that
  wasn't detected. ← / → to browse, `F` toggles face boxes, `I` toggles info, Esc closes.
- **Face grouping sensitivity** (Library page): move towards *Stricter* if different people get mixed,
  towards *Looser* if one person is split into many groups, then **Apply and regroup**. Named people and
  manual corrections are never undone.
- **Dark or light**: dark is the default. The first launch asks which look you want (*Choose your look*);
  switch any time with the **Light mode / Dark mode** button at the bottom of the sidebar, in
  **Library ▸ Appearance**, or with Ctrl+Shift+L. The change applies immediately and is remembered.

  ![Choose your look](docs/screenshots/05-choose-your-look.png)
- Supports JPG, PNG, WebP, BMP, GIF, TIFF and HEIC/HEIF (iPhone) photos.

### How corrections stick
- Faces you assign by hand are never moved automatically.
- Once you say a face is *not* someone, it will never be auto-assigned to them again.
- Faces from a dissolved group can still join existing people, but won't re-form the same group.

## Running

**Packaged app:** run `dist\RulZ Image Organizer\RulZ Image Organizer.exe` (the whole folder must stay
together; you can copy it anywhere and pin the .exe to Start/taskbar).

**From source:**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

**Build the .exe:** `powershell -ExecutionPolicy Bypass -File .\build.ps1`

Library data (database, thumbnails, face crops) lives in `%LOCALAPPDATA%\RulZ Image Organizer`.
Your preferences (theme, window size) are kept there too, in `settings.ini`.
Set `RULZ_DATA_DIR` to use a different location, and `RULZ_THEME=light|dark` to force a theme for that run.

## How it works

| Part | File |
| --- | --- |
| Face detection: OpenCV **YuNet** (`models/face_detection_yunet_2023mar.onnx`) | `rulz/faces.py` |
| Face recognition: OpenCV **SFace** 128-d embeddings (`models/face_recognition_sface_2021dec.onnx`) | `rulz/faces.py` |
| Expressions: **FER+** emotion model (`models/emotion-ferplus-8.onnx`) on the aligned face crop | `rulz/faces.py` |
| Head direction: yaw/pitch/roll from the 5 face landmarks; search vocabulary | `rulz/attributes.py`, `rulz/search.py` |
| Body pose: MediaPipe **BlazePose** person detector + 33 body landmarks, run around each face; joint-angle rules | `rulz/body.py` |
| Grouping: match to existing people (centroid + nearest face), then link remaining faces into new groups | `rulz/clustering.py` |
| Scanning, thumbnails, moved-file detection | `rulz/scanner.py`, `rulz/imaging.py` |
| SQLite storage (WAL) | `rulz/db.py` |
| Qt (PySide6) UI | `rulz/ui/` |

The face models come from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo) and the expression
model from the [ONNX Model Zoo](https://github.com/onnx/models); see [models/README.md](models/README.md)
for their licenses. Libraries created before
expressions or poses were added are analyzed once in the background on the next scan; people assignments
are kept.

## License

Open source under the [MIT License](LICENSE). The face, expression and pose models in `models/` keep their own licenses; see [models/README.md](models/README.md).

The RulZ name and logo are excluded from the MIT License; all rights reserved. If you fork or redistribute this
project, remove or replace the RulZ branding. See [LOGO-NOTICE.md](LOGO-NOTICE.md).
