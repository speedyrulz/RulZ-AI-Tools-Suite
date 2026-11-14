# Screenshots

How the launcher's screenshots (`docs/screenshots/`), each app's catalog gallery (`catalog/<id>/screenshots/`) and
the README pictures in `apps/<folder>/docs/screenshots/` were made, and how to make them again. `01-*` is each app's
card thumbnail; the launcher lists every file in that folder.

Everything shown is demo content made for the purpose: procedurally drawn pictures, header-only model files,
scripted model replies, made-up accounts and instances, fresh settings folders and blank app profiles. No real
photos, faces, chats, datasets, training samples, accounts or app settings appear, and none of the app repos or
their `%APPDATA%`/`%LOCALAPPDATA%`/registry settings are written to. Window captures take one window only
(`QWidget.grab()`, `PrintWindow`, the DevTools protocol or Playwright), so nothing else on the desktop can end up
in a picture.

## Workspace

All demo data, profiles and raw captures go in a temp folder mapped to a drive letter, so the paths the apps
show read `R:\...` instead of `C:\Users\<name>\AppData\Local\Temp\...`:

```bat
mkdir "%TEMP%\rulz-shots"
subst R: "%TEMP%\rulz-shots"
rem ... captures ...
subst R: /d
```

Python commands use an app's own venv (no installs). `-B` keeps `__pycache__` out of the repos. `IO` below means
`"%LOCALAPPDATA%\Programs\RulZ AI Tools\RulZ Image Preparer\.venv\Scripts\python.exe"` (the launcher's Image
Preparer install), which has PySide6, Pillow, numpy and OpenCV. If the screen is scaled (125%, say), the Qt
harnesses still render at 100% so the images keep their sizes.

| Script | What it does |
| --- | --- |
| `make_demo_images.py` | Draws landscapes, seascapes, dunes, aurora, bokeh and geometric abstracts from random numbers. Optional EXIF dates (`--exif`, `--month-counts`) and same-name `.txt` captions (`--captions`). |
| `make_demo_gguf.py` | A fake models folder: header-only GGUF files named after well-known open models, made sparse so they show realistic sizes without using disk space. They can't be loaded. |
| `make_demo_llm_chats.py` | Three made-up chats for a RulZ LLM Suite profile. |
| `wincap.py` | ctypes helpers: find a process's windows, `PrintWindow(PW_RENDERFULLCONTENT)` a window, post key presses. |
| `cdp_capture.mjs` | Node 22+ DevTools-protocol driver: forces a viewport size, runs a list of click/check steps and saves `Page.captureScreenshot` PNGs (`*.plan.json`; `beforeShot` runs before every capture, e.g. to remove toasts). |
| `capture_*.py`, `capture_coding_agent.ts`, `rulz_trainer_fake_fleet.py` | Per-app harnesses (below). |
| `build_tournament_copy.ps1` | Builds a copy of Image Tournament outside its repo (below). |
| `finalize.py` | Re-saves reviewed PNGs from `<staging>/<id>/NN-*.png` into `catalog/<id>/screenshots/` (optimized; JPEG only if one were over 1.5 MB). |

Copy the catalog images you are refreshing into a staging folder of their own (`R:\final\<id>\NN-name.png`) and run
`IO -B finalize.py R:\final "<suite>\catalog"`; pointing it at a shared `R:\out` would also take the other apps'
raw captures. README pictures keep their names in `apps/<folder>/docs/screenshots/` and are re-saved the same way
(Pillow, RGB, `optimize=True`). Then run `python tools/sync_catalog.py` so the README copies in
`catalog/<id>/repo/` follow. Look at every image before you commit it.

## launcher (Electron)

```bat
npm ci
npx electron-vite build
mkdir R:\launcher-profile
echo {"installRoot": "R:\\Programs\\RulZ AI Tools"} > R:\launcher-profile\settings.json
start "" node_modules\electron\dist\electron.exe . --user-data-dir=R:\launcher-profile --remote-debugging-port=9361 --force-device-scale-factor=1
node tools\screenshots\cdp_capture.mjs tools\screenshots\launcher.plan.json
```

Run from the suite's root, after the apps' catalog images are in place (the cards show them). `--user-data-dir`
gives the launcher a blank profile, and the install folder setting makes the install dialog show an `R:\` path
instead of one with your user name. The plan picks Dark, then captures the home page, the Rulz Training Captioner
page, its install dialog (closed with Cancel; nothing is installed) and the home page in light mode, at 1400x900.
The status counters are real: RulZ LLM Suite and RulZ Coding Agent installs on the PC are found and counted.

## image-organizer (PySide6)

```bat
IO -B make_demo_images.py "R:\Photos\Camera Roll" --count 40 --seed 7 --exif --start 2026-03 --month-counts 4,6,6,6,6,12
IO -B make_demo_images.py "R:\Photos\Wallpapers" --count 12 --seed 21 --exif --names titles --start 2025-12 --month-counts 12
IO -B capture_image_organizer.py --data R:\io-data --folder "R:\Photos\Camera Roll" --folder "R:\Photos\Wallpapers" --out R:\out\image-organizer --readme R:\out-readme\image-organizer --viewer-file IMG_4272.jpg
```

`RULZ_DATA_DIR` points the library (and its `settings.ini`) at `--data`; `RULZ_THEME=dark`. The harness adds the
two folders to the new library, lets the normal startup scan finish (the face models find no faces, so People
stays empty and isn't shown), and grabs the Photos timeline, the viewer, the search Filters, and the Library page;
`--readme` adds the README set (dark and light timeline, viewer, filters, Choose your look). It runs without
`pillow_heif`.

## image-tournament (WinForms)

The app keeps `settings.ini` and the resumable `tournament.txt` in `%APPDATA%\RulZ Image Tournament`, rewrites
both on exit, and deletes the saved tournament if you decline its resume prompt. Since 1.2.0 it reads them from
`<RULZ_TOURNAMENT_DATA>\RulZ Image Tournament` instead when that variable is set, which the harness does. The
build script compiles an unchanged copy outside the repo, so no `bin\` folder appears in it:

```bat
powershell -ExecutionPolicy Bypass -File build_tournament_copy.ps1 -Out R:\tt-build
IO -B make_demo_images.py R:\Tournament --count 8 --seed 33 --names titles
xcopy /e /i R:\Tournament R:\Tournament-pair
IO -B capture_image_tournament.py --exe "R:\tt-build\RulZ Image Tournament.exe" --data R:\tt-data --images R:\Tournament-pair --pair "Sea Glow 08.jpg" --out R:\out\image-tournament
IO -B capture_image_tournament.py --exe "R:\tt-build\RulZ Image Tournament.exe" --data R:\tt-data-readme --images R:\Tournament --out R:\out-readme\image-tournament --choose-look 05-choose-your-look.png --light-shot 02-light.png --shots 01-dark.png,03-split-view.png,04-synced-zoom.png --zoom-at 0.2 0.45
```

`--pair` swaps one image for itself plus a warmer colour grade, so the first match is a "which edit?" comparison;
it changes the folder, so give it a copy. The script waits for the first match, captures it (PrintWindow, client
area), posts `V` for the split view and wheel-zooms both images together. It never presses Keep or an arrow key, so
no file is moved; `--data` must be a fresh folder (the script refuses one that has a saved tournament). The app runs
DPI-unaware for that process only, so the images keep their size on a scaled screen.

## llm-suite (Electron)

```bat
cd apps\rulz-llm-suite
npm ci
npx electron-vite build
IO -B ..\..\tools\screenshots\make_demo_gguf.py R:\LLM
mkdir R:\llm-profile
echo {"modelsDir": "R:\\LLM"} > R:\llm-profile\settings.json
IO -B ..\..\tools\screenshots\make_demo_llm_chats.py R:\llm-profile
start "" node_modules\electron\dist\electron.exe . --user-data-dir=R:\llm-profile --remote-debugging-port=9335
node ..\..\tools\screenshots\cdp_capture.mjs ..\..\tools\screenshots\llm-suite.plan.json
```

`--user-data-dir` gives the app a blank profile with the three demo chats and no MCP servers; the models folder is
the demo one. Before the Reverse Engineering shot, set `re.kitDir` in the profile's `settings.json` to an empty
folder (the default is a path on the developer's PC, and its practice tasks would show) and restart the app.
Engine detection (`llama-server --version`) and GPU names are real; no model is loaded. The plan saves the README
set (`chat-dark`, `chat-light`, `choose-your-look`, `load-model`, `my-models`) and the gallery pages (Reverse
Engineering, MCP Tools, Developer, Settings); the gallery's `01-chat` is `chat-dark`. Close the app with the
window's close button or CDP `Browser.close`.

## coding-agent (Electron)

```bat
cd apps\rulz-coding-agent
npm ci
npx electron-vite build
cd ..\..
node tools\screenshots\capture_coding_agent.ts apps\rulz-coding-agent R:\out\coding-agent
```

The script starts the app's own stand-in for RulZ LLM Suite (`tests/helpers/mockRulz.ts`) with scripted replies
and drives the built app with Playwright: the first-run wizard, a to-do list app built with one approval
(`npm install`, which installs nothing), **Run my app**, a second task that edits `style.css` (the diff), the light
look, and Settings with *Let my phone connect* on (the QR code holds a throw-away pairing code for `127.0.0.1`).
Settings, projects and an empty MCP config go in `R:\out\coding-agent\work`, so the real ones are never read. A
third argument names another `electron.exe` if `npm ci` skipped Electron's download.

## training-captioner (PySide6)

```bat
IO -B make_demo_images.py R:\Dataset\landscapes --count 18 --seed 11 --captions --names dataset --size 1024
del R:\Dataset\landscapes\landscape_007.txt R:\Dataset\landscapes\landscape_013.txt R:\Dataset\landscapes\landscape_016.txt
"%LOCALAPPDATA%\Programs\RulZ AI Tools\Rulz Training Captioner\.venv\Scripts\python.exe" -B capture_training_captioner.py --appdata R:\tc-config --folder R:\Dataset\landscapes --out R:\out\training-captioner --readme R:\out-readme\training-captioner --app-dir "R:\Rulz Training Captioner"
```

`RULZ_CAPTIONER_DIR` points the app's settings at `--appdata` (use a fresh folder each run), so
`settings.json`/`prompts.json` are created there; `--app-dir` makes the Settings dialog show an `R:\` llama-server
path (an empty placeholder file). Images are added with `MainWindow._add()` (what drag-and-drop calls). Three
captions are deleted so the grid shows the "missing" state. The Settings dialog is captured with PrintWindow and
composited over the main window at its real position. No llama-server is started.

## image-preparer (PySide6)

```bat
IO -B make_demo_images.py R:\Dataset\preparer --count 12 --seed 21 --names titles --size 1600
IO -B capture_image_preparer.py --images R:\Dataset\preparer --config R:\ip-config --out R:\out\image-preparer
```

The app runs from `apps\RulZ-Image-Preparer`. The demo pictures are copied into `--config\photos` with
blotchy color noise added and saved as low-quality JPEGs, so there is something to clean up. The app's QSettings
(registry) are swapped for an .ini in `--config` and `LOCALAPPDATA` points there too; Best quality (AI) mode and
Auto crop aren't used, so no model is loaded. `02` and `03` open the `Hillside` picture (`--closeup`). `--shadow`
also captures a painted shadow area; that one isn't in the gallery, because the lifted shadow left a visible edge.

## clip-extractor (PySide6)

```bat
"C:\simplest-video-edit\.venv\Scripts\python.exe" -B capture_clip_extractor.py --videos R:\Videos --config R:\ce-config --out R:\out\clip-extractor --readme R:\out-readme\clip-extractor --cv2-path "%LOCALAPPDATA%\Programs\RulZ AI Tools\RulZ Image Preparer\.venv\Lib\site-packages"
```

Any venv with the app's requirements works. The harness copies the repo's own `samples\test_pattern_10s.mp4` and
`samples\skin_cast.mp4` + `skin_reference.png` (a drawn cartoon face) to `--videos`, and makes a short video of a
drawn building facade for the Distortion repair window (which needs OpenCV: `--cv2-path` adds a site-packages folder
that has it). The app's own `SIMPLECUT_SETTINGS_INI` points its preferences at `--config`, so defaults are shown
and the registry is left alone. Modal dialogs are captured from a timer inside their own event loop and then
cancelled; nothing is exported.

## rulz-trainer (NiceGUI)

The WebUI runs from a clean copy of `apps\RulzTrainer` (no outputs, configs or job file to reattach to), with a
made-up fleet of two vast.ai instances and a demo dataset:

```bat
git archive HEAD apps/RulzTrainer | tar -x -C R:\trainer
IO -B tools\screenshots\make_demo_images.py R:\trainer\apps\RulzTrainer\datasets\landscapes --count 16 --seed 11 --captions --names dataset --size 1024
set RULZ_VAST_HOME=R:\vast-home
start "" "%LOCALAPPDATA%\Programs\RulZ AI Tools\RulzTrainer\venv\Scripts\python.exe" -B tools\screenshots\rulz_trainer_fake_fleet.py R:\trainer\apps\RulzTrainer 7899
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --disable-gpu --hide-scrollbars --no-first-run --disable-sync --disable-extensions --remote-debugging-port=9341 --user-data-dir=R:\chrome-trainer --window-size=1600,1000 http://127.0.0.1:7899/
node tools\screenshots\cdp_capture.mjs tools\screenshots\rulz-trainer.plan.json
node tools\screenshots\cdp_capture.mjs tools\screenshots\rulz-trainer-fleet.plan.json
```

`rulz_trainer_fake_fleet.py` swaps `rulz_fleet.Fleet` for one over `tests/test_vast_fleet.py`'s fake machines (no
ssh, no vast.ai) with demo run names, then runs `webui.py`; `RULZ_VAST_HOME` must be a fresh folder, so the real
`~/.rulz_vast` fleet is never read. Use Chrome, not Edge, for the headless browser: a new Edge profile signs itself
in to the Windows account and starts syncing. No training run is started and no model is downloaded. The Monitor's
GPU and disk tiles are real. Stop the server and the browser when done (`taskkill /PID <webui pid> /T /F`, CDP
`Browser.close`). The gallery's names differ from the README's: `01-train-settings` is `01-train-dark`,
`02-wizard-model` is `03-wizard-model`, `03-monitor` is `04-monitor`, `05-dataset` is `07-dataset`.

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
matches grid, and the same run in light mode. Stop the app and the model server afterwards. (Last made 2026-10-29;
the 2026-11-13 refresh skipped it because it needs the model on a GPU.)

## hf-manager (NiceGUI)

`capture_hf_manager.py` runs the app from the suite checkout on a made-up account: who the token belongs to, the
namespace listing, repo summaries, branches, history and file listings are replaced with "demo-user" (organization
"demo-lab") and its repos, so nothing reaches huggingface.co. The vast.ai page gets a made-up attached instance
(`203.0.113.24`, an address range reserved for documentation) with demo folders and transfers; no ssh is run and
nothing reaches vast.ai. Settings go to the workspace with a fake token, and `HF_TOKEN` and the hf CLI login are
hidden, so no real token or account is read. Any venv with the app's requirements works (`nicegui`,
`huggingface_hub`), for example the launcher's install:

```bat
start "" VENV\python.exe -B capture_hf_manager.py --app "<suite>\apps\RulZ-Huggingface-Manager" --data R:\hf-data --port 8799 --choose-look
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --disable-gpu --hide-scrollbars --no-first-run --disable-sync --disable-extensions --remote-debugging-port=9351 --user-data-dir=R:\chrome-profile --window-size=1600,1000 http://127.0.0.1:8799/
node cdp_capture.mjs hf-manager.plan.json
```

`--choose-look` starts it as a first run, so the plan can capture the "Choose your look" dialog before it picks
Dark; it ends by switching to the light look. `07-vast-transfers.png` was taken afterwards, scrolled down the vast.ai
page. Copy `01-my-hub.png` to the app's `docs/screenshots/` too (its README shows it). Stop the server and the
browser when done.

## image-grid (WPF)

Not refreshed on 2026-11-13 (1.2.0 changed little on screen); its images are from 1.1.x. It has no harness yet:
opening Export or switching the theme needs UI automation.
