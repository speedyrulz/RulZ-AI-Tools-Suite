# RulZ Image Crawler

Crawl a website and keep only the images that match a description such as "women with red hair". A local
Qwen3-VL model judges every image; matches are saved to disk with a `matches.json` describing each one.

## Start it

Double-click **`Start Image Crawler.bat`**. It:

1. loads the model once on **each** NVIDIA card with room for it (ports 8080, 8081, …), reusing any that are
   already running; images are spread over all of them, so two GPUs check about twice as many per second,
2. waits for the model to load (about 10–20 s); a card that fails (e.g. busy with another program) is skipped,
3. starts the app (port 8000) and opens <http://127.0.0.1:8000> in your browser.

Close that window (or press Ctrl+C) to stop everything it started. A model server that was already running
before you launched is left running.

From a terminal, with options:

```powershell
powershell -ExecutionPolicy Bypass -File start_app.ps1 -Gpu 1 -NoBrowser
```

| Option | Default | |
|---|---|---|
| `-Gpu` | `all` | which cards get a copy of the model: `all`, or indexes as in `nvidia-smi` like `0` or `0,1` |
| `-Port` / `-LlamaPort` | `8000` / `8080` | app and model ports |
| `-TimeoutSec` | `180` | how long to wait for the model to load |
| `-NoBrowser` | off | don't open the browser |

## Using it

1. Choose **A website** (enter its address) or **A folder on this PC** (type the path or **Browse…**), describe
   what to keep, then **Generate checklist**.
2. Read the checklist the model wrote and tighten it if needed (e.g. add "drawings" or "soccer players" to
   EXCLUDE). Set the confidence to save at (0.8 is suggested for small body details like hands or eyes), a
   **minimum size in megapixels** (0 = any) and the crawl limits.
3. **Crawl & Download** (or **Scan & Copy matches**). Matches appear as they're found, each with its size;
   Pause, Resume and Cancel at any time.
4. Use the slider to hide less certain matches and **Download shown as zip**, or grab single images.
5. **Past searches** lists every earlier run (also from before a restart); **View** reopens its matches.

**Full-size originals** (on by default): for each picture the app first tries the full-size original — the
image file a thumbnail links to, or the address with the site's resize markers removed (Wikimedia,
WordPress `-300x200`, Shopify, Google/Blogger, `?w=…` CDN parameters) — and falls back to the page's own copy.
At most two downloads per site run at once; if a site answers "too many requests" the app uses the page's
copies for that site until it says to try again, and notes this under Problems.

**Confidence** is the model's own probability for its yes/no answer, so clear-cut images score ~1.00 and
borderline ones lower; the slider is most useful for subtle prompts.

Each run is saved to `output\<date>_<site or folder>_<tag>\` with the image files and `matches.json`. Folder
scans **copy** matches there; the scanned folder is never changed.
Expect roughly 1–2 s per new image; images checked before come from the cache instantly.

## Where things are

| Path | What |
|---|---|
| `crawler_app/` | the app: crawler, image fetcher, run pipeline, API and the web page (`static/index.html`) |
| `image_detection/` | the detection engine and the model files (`LLM/`); see its README for the API and tuning knobs |
| `output/` | saved matches, one folder per run |
| `logs/` | `llama-server-gpu<N>.log(.err)` and `crawler-app.log` from the launcher |
| `build-plan.md` | the build plan this was built from |

## Tests

```powershell
python -m pytest crawler_app        # 23 tests, real Chromium + a fake model; no GPU needed
python -m pytest image_detection    # 37 engine tests, no GPU needed
python image_detection\scripts\eval_labeled.py   # accuracy on your labeled images (see image_detection\README.md)
```

## Requirements

Python 3.11+, an NVIDIA GPU with ~16 GB, LM Studio's CUDA 12 llama.cpp backend (or set `LLAMA_SERVER_EXE`), and
once: `pip install -e image_detection -e crawler_app` and `python -m playwright install chromium`.
