# Rulz Training Captioner

Windows desktop app for captioning image datasets with a local vision LLM.

- Add images, folders (or drag & drop). Thumbnail grid with an editable caption box under every image.
- Captions live in `image.txt` next to `image.png`. Editing a box (adding or removing text) writes the file, creating it if needed.
- Save/edit/delete your own system prompts (stored in `%APPDATA%\RulzCaptioner\prompts.json`).
- Multi-GPU: **Parallel** (one model copy per GPU, images shared out) or **Split** (one model spread across GPUs via `--tensor-split`).
- Works with any OpenAI-compatible server too (LM Studio, Ollama, vLLM…); list several URLs to caption in parallel.

## Run

Double-click `run.bat` (creates a venv and installs PySide6 + requests on first run).

## Set up the model (managed mode)

1. **llama.cpp**: download `llama-bXXXX-bin-win-cuda-13.x-x64.zip` + `cudart-llama-bin-win-cuda-13.x-x64.zip` from
   <https://github.com/ggml-org/llama.cpp/releases> (CUDA 13 builds support RTX 50-series) and unzip both into a
   `llama.cpp\` folder next to `main.py`, where it's auto-detected. Unzip newer builds over it to update.
   Installing the app with the [RulZ AI Tools Suite](https://github.com/speedyrulz/RulZ-AI-Tools-Suite) launcher
   does this for you.
2. Download a vision-capable **GGUF** model and its **mmproj** file (e.g. Qwen2.5-VL / Qwen3-VL, Gemma 3, InternVL, MiniCPM-V, Mistral Small 3.x vision).
3. Settings ▸ Backend: choose the exe, model and mmproj. Settings ▸ GPUs: tick GPUs and the mode.
4. Press **Caption All** (or select images ▸ **Caption Selected**). The servers start on first use and stay loaded; **Server ▸ Stop Servers** frees VRAM.

### Which GPU mode?
- *Parallel*: fastest when the model + context fits on each card. Mixed cards are fine; each gets its own server.
- *Split*: needed when the model doesn't fit on one card. Use *Tensor split* for uneven ratios (e.g. `16,16`).
- *Concurrent requests per server* adds batching on a server at the cost of VRAM (context is multiplied).

## Tips
- Keep **Disable thinking** on (Settings ▸ Generation) for reasoning models like Qwen3.5/3.6 and Gemma 4, otherwise
  they spend the whole token budget thinking and return nothing.
- Prefix/suffix fields add a trigger word or fixed text to every generated caption.
- "Skip images that already have a caption" applies to *Caption All*; *Caption Selected* always overwrites.
- The log (View ▸ Log) shows llama-server output, useful if a model fails to load.
