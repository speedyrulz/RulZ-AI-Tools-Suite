# RulZ Image Crawler

Crawl a website and keep only the images that match a description such as "women with red hair". A local
Qwen3-VL model judges every image; matches are saved to disk with a `matches.json` describing each one.

Part of the [RulZ AI Tools Suite](https://github.com/speedyrulz/RulZ-AI-Tools-Suite): the easiest way to install
it is the suite's launcher, which sets everything below up for you. Guide:
[the wiki](https://github.com/speedyrulz/RulZ-AI-Tools-Suite/wiki/Image-Crawler).

## Install

**With the RulZ AI Tools Suite launcher:** open *RulZ Image Crawler* and click **Install**. It installs Python if
needed, makes a Python environment in the app folder (`.venv`), installs the app, the Chromium browser it crawls
with and a llama.cpp CUDA build (`llama.cpp\`), and downloads the model (about 9.7 GB). Then **Launch**.

**By hand:**

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .\image_detection -e .\crawler_app
.venv\Scripts\python.exe -m playwright install chromium
.venv\Scripts\python.exe image_detection\scripts\download_model.py   # about 9.7 GB; run again to resume
```

and put a [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) CUDA build's files in a `llama.cpp` folder
here (or have LM Studio installed, or set `LLAMA_SERVER_EXE`).

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
| `image_detection/` | the detection engine and the model files (`LLM/`, downloaded by `scripts/download_model.py`); see its README for the API and tuning knobs |
| `output/` | saved matches, one folder per run |
| `logs/` | `llama-server-gpu<N>.log(.err)` and `crawler-app.log` from the launcher |
| `llama.cpp/` | the llama-server build the model runs in (optional: LM Studio's is used otherwise) |
| `.venv/` | the app's Python environment |

## Tests

```powershell
python -m pytest crawler_app        # 23 tests, real Chromium + a fake model; no GPU needed
python -m pytest image_detection    # 37 engine tests, no GPU needed
python image_detection\scripts\eval_labeled.py   # accuracy on your labeled images (see image_detection\README.md)
```

## Requirements

Windows, Python 3.11+, an NVIDIA GPU with ~12 GB or more, and a llama.cpp CUDA build: the `llama.cpp` folder here, LM
Studio's CUDA 12 backend, or `LLAMA_SERVER_EXE`. The start script uses `.venv` when it's there, otherwise
`python` on PATH. See [Install](#install).

## License

MIT, see [LICENSE](LICENSE). The RulZ name and logo are excluded; see [LOGO-NOTICE.md](LOGO-NOTICE.md).
