# Catalog screenshots

How the images in `catalog/<id>/screenshots/` were made, and how to make them again. `01-*` is each app's
card thumbnail.

Everything shown is demo content made for the purpose: procedurally drawn pictures, header-only model files,
fresh settings folders and blank app profiles. No real photos, faces, chats, datasets, training samples or app
settings appear, and none of the app repos or their `%APPDATA%`/`%LOCALAPPDATA%`/registry settings are
written to. Window captures take one window only (`QWidget.grab()`, `PrintWindow`, or the DevTools protocol),
so nothing else on the desktop can end up in a picture.

## Workspace

All demo data, profiles and raw captures go in a temp folder mapped to a drive letter, so the paths the apps
show read `R:\...` instead of `C:\Users\<name>\AppData\Local\Temp\...`:

```bat
mkdir "%TEMP%\rulz-shots"
subst R: "%TEMP%\rulz-shots"
rem ... captures ...
subst R: /d
```

Python commands use an app's own venv (no installs). `-B` keeps `__pycache__` out of the repos.
`IO` below means `"C:\coding\RulZ Image Organizer\.venv\Scripts\python.exe"`; that venv has Pillow and numpy.

| Script | What it does |
| --- | --- |
| `make_demo_images.py` | Draws landscapes, seascapes, dunes, aurora, bokeh and geometric abstracts from random numbers. Optional EXIF dates (`--exif`, `--month-counts`) and same-name `.txt` captions (`--captions`). |
| `make_demo_gguf.py` | A fake models folder: header-only GGUF files named after well-known open models, made sparse so they show realistic sizes without using disk space. They can't be loaded. |
| `wincap.py` | ctypes helpers: find a process's windows, `PrintWindow(PW_RENDERFULLCONTENT)` a window, post key presses. |
| `cdp_capture.mjs` | Node 22+ DevTools-protocol driver: forces a 1600x960 viewport, runs a list of click/check steps and saves `Page.captureScreenshot` PNGs (`*.plan.json`). |
| `capture_*.py` | Per-app harnesses (below). |
| `build_tournament_copy.ps1` | Builds a redirected copy of Image Tournament outside its repo (below). |
| `finalize.py` | Re-saves the reviewed PNGs into `catalog/<id>/screenshots/` (optimized; JPEG only if one were over 1.5 MB). |

After all captures: `IO -B finalize.py R:\out "C:\coding\RulZ AI Tools Suite\catalog"`.
Look at every image before you commit it.

## image-organizer (PySide6)

```bat
IO -B make_demo_images.py "R:\Photos\Camera Roll" --count 40 --seed 7 --exif --start 2026-03 --month-counts 4,6,6,6,6,12
IO -B make_demo_images.py "R:\Photos\Wallpapers" --count 12 --seed 21 --exif --names titles --start 2025-12 --month-counts 12
IO -B capture_image_organizer.py --data R:\io-data --folder "R:\Photos\Camera Roll" --folder "R:\Photos\Wallpapers" --out R:\out\image-organizer --viewer-file IMG_4272.jpg
```

`RULZ_DATA_DIR` points the library at `R:\io-data`; `RULZ_THEME=dark`. The harness adds the two folders to
the new library, lets the normal startup scan finish (the face models find no faces, so People stays empty
and isn't shown), and grabs the Photos timeline, the viewer, the search Filters, and the Library page.
QSettings (window geometry) is redirected to an .ini in `--data`, so the registry isn't touched.

## image-tournament (WinForms)

The app keeps `settings.ini` and the resumable `tournament.txt` in `%APPDATA%\RulZ Image Tournament`, rewrites
both on exit, and deletes the saved tournament if you decline its resume prompt. .NET ignores a changed
`APPDATA` variable, so the capture runs a copy built outside the repo with one changed line (`Settings.Folder`
also honours `RULZ_TOURNAMENT_DATA`):

```bat
powershell -ExecutionPolicy Bypass -File build_tournament_copy.ps1 -Out R:\tt-build
IO -B make_demo_images.py R:\Tournament --count 8 --seed 33 --names titles
IO -B capture_image_tournament.py --exe "R:\tt-build\RulZ Image Tournament.exe" --data R:\tt-data --images R:\Tournament --pair "Sea Glow 08.jpg" --out R:\out\image-tournament
```

`--pair` swaps one image for itself plus a warmer colour grade, so the first match is a "which edit?"
comparison. The script waits for the first match, captures it (PrintWindow, client area), posts `V` for the
split view and wheel-zooms both images together. It never presses Keep or an arrow key, so no file is moved;
`--data` must be a fresh folder (the script refuses one that has a saved tournament).

## llm-suite (Electron)

```bat
IO -B make_demo_gguf.py R:\LLM
mkdir R:\llm-profile3
echo { "modelsDir": "R:\\LLM" } > R:\llm-profile3\settings.json
start "" "C:\coding\RulZ LLM Suite\dist\win-unpacked\RulZ LLM Suite.exe" --user-data-dir=R:\llm-profile3 --remote-debugging-port=9335
node cdp_capture.mjs llm-suite.plan.json
```

`--user-data-dir` gives the app a blank profile: no chats (the plan checks "No chats yet"), no MCP servers,
and the Settings page shows `R:\llm-profile3` as its data folder. The models folder is the demo one; the
user's own `C:\LLM` list is not shown. Engine detection (`llama-server --version`) and GPU names are real; no
model is loaded. Close the app with the window's close button or CDP `Browser.close`.

## training-captioner (PySide6)

```bat
IO -B make_demo_images.py R:\Dataset\landscapes --count 18 --seed 11 --captions --names dataset --size 1024
del R:\Dataset\landscapes\landscape_007.txt R:\Dataset\landscapes\landscape_013.txt R:\Dataset\landscapes\landscape_016.txt
"C:\coding\Rulz Training Captioner\.venv\Scripts\python.exe" -B capture_training_captioner.py --appdata R:\tc-config --folder R:\Dataset\landscapes --out R:\out\training-captioner
```

`APPDATA` is set to `--appdata` before the app's config module loads (use a fresh folder each run), so
`settings.json`/`prompts.json` are created there. Images are added with `MainWindow._add()` (what
drag-and-drop calls). Three captions are deleted so the grid shows the "missing" state. The Settings dialog is
captured with PrintWindow and composited over the main window at its real position. No llama-server is
started.

## clip-extractor (PySide6)

```bat
"C:\simplest-video-edit\.venv\Scripts\python.exe" -B capture_clip_extractor.py --config R:\ce-config --out R:\out\clip-extractor
```

`01-crop-and-fixed-length.png` is the repo's `docs\screenshot.png`, copied. The others use the repo's own
`samples\test_pattern_10s.mp4` and `samples\skin_cast.mp4` + `skin_reference.png` (a drawn cartoon face). The
app's QSettings (registry) are swapped for an .ini in `--config`, so default preferences are shown and the
registry is left alone. Modal dialogs are captured from a timer inside their own event loop and then
cancelled; nothing is exported.

## rulz-trainer (NiceGUI)

The real folder has outputs, configs and a job file that the UI would reattach to (training samples are
private), so the WebUI runs from a clean clone:

```bat
git clone --no-hardlinks C:\ai\RulzTrainer R:\RulzTrainer
cd /d R:\RulzTrainer
start "" "C:\ai\RulzTrainer\venv\Scripts\python.exe" -B webui.py --port 7899
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --disable-gpu --hide-scrollbars --no-first-run --disable-sync --disable-extensions --remote-debugging-port=9341 --user-data-dir=R:\chrome-profile --window-size=1600,1000 http://127.0.0.1:7899/
node cdp_capture.mjs rulz-trainer.plan.json
```

Use Chrome, not Edge, for the headless browser: a new Edge profile signs itself in to the Windows account and
starts syncing. No training run is started and no model is downloaded. Stop the server and the browser when
done (`taskkill /PID <webui pid> /T /F`, CDP `Browser.close`).

## rulz-image-crawler (FastAPI web page)

The app and a model server run from a suite checkout (`apps\rulz-image-crawler`) with a venv made outside it
(`pip install -e image_detection -e crawler_app`, `playwright install chromium`). Output and the image cache go to
the workspace, and the model server reads an existing copy of the model, so nothing is written to an install:

```bat
IO -B make_demo_images.py R:\Pictures\Wallpapers --count 30 --seed 17 --names titles --size 1600
powershell -ExecutionPolicy Bypass -File image_detection\scripts\start_llama_server.ps1 -ModelDir <model folder> -Gpu 1 -Port 8099
set CRAWLER_OUTPUT=R:\Image Crawler\output
set CACHE_PATH=R:\ic-cache.sqlite3
set LLAMA_URL=http://127.0.0.1:8099
VENV\python.exe -B -m crawler_app --port 8011
VENV\python.exe -B capture_image_crawler.py --folder R:\Pictures\Wallpapers --real "%TEMP%\rulz-shots" --out R:\out\rulz-image-crawler
```

It scans the demo folder for "landscapes with mountains" with the real model: the checklist the model wrote, the
matches grid, and the same run in light mode. Stop the app and the model server afterwards.
