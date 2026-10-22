# RulZ LLM Suite

A Windows desktop app for running local LLMs, modelled on LM Studio's feature set. Models run in your own
llama.cpp build (`llama-server.exe`), one process per loaded model, bound to `127.0.0.1`.

**v0.1 covers chat and model loading.** Discover/downloads, the local API server, RAG and a CLI are planned.

## Features

- **My Models** — scans a folder (default `C:\LLM`) for `.gguf` files, reads GGUF metadata (architecture,
  parameters, quantization, trained context, layers), pairs vision projectors (`mmproj`) and MTP draft files with
  their model, joins split shards, and watches the folder for changes.
- **Model loading** (`Ctrl+L`) — context length, GPU offload (auto-fit / all / N layers), single-GPU or split
  across GPUs, flash attention, KV cache quantization, threads, batch sizes, mmap/mlock, chat template override and
  extra arguments; per-model saved defaults; memory estimate with a fits/partial/likely-OOM verdict; load progress;
  live server log; several models loaded at once.
- **Chat** — streaming replies, collapsible reasoning ("thinking") blocks, Markdown with tables, LaTeX and
  highlighted code, image and text-file attachments (drag-drop / paste), edit, regenerate, continue, branch,
  delete; tok/s, time-to-first-token and token stats; context usage meter with "truncate middle" or "stop" overflow
  policy; folders, pinning, search; per-chat system prompt and sampling settings; presets; JSON-schema structured output.
- **Chat handoff** — at 85% context (or any time via the header button) the model writes a handoff summary and
  a new chat with the same settings continues from it.
- **MCP tools** — configure servers in `mcp.json` (same format as LM Studio / Cursor; stdio `command` or remote
  `url`) on the MCP Tools page. Enabled tools are offered to the model; each call asks for approval unless the
  tool is marked "always allow", and calls and results show inline in the chat.
- GPU/RAM meters (nvidia-smi), dark/light theme, keyboard shortcuts (see Settings).

## Engine

Build `llama-server` with CUDA in a separate folder (sm_89 = RTX 40xx, sm_120 = RTX 50xx):

```
cmake -S C:\coding\llama.cpp -B C:\coding\llama.cpp\build-cuda -G "Visual Studio 17 2022" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89;120" -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
cmake --build C:\coding\llama.cpp\build-cuda --config Release --target llama-server -j
```

The app auto-detects `C:\coding\llama.cpp\build-cuda\bin\Release`; any other folder containing `llama-server.exe`
can be set in Settings. Child processes get `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so GPU numbers match nvidia-smi.

## Development

```
npm install
node node_modules/electron/install.js   # first time only: Electron downloads its binary lazily
npm run dev          # run with hot reload
npm test             # unit tests (GGUF reader, model grouping, args, estimates, SSE, context fitting)
npm run typecheck
npm run build:win    # installer → dist/RulZ-LLM-Suite-Setup-<version>.exe
```

App data (settings, chats, presets, metadata cache) lives in `%APPDATA%\RulZ LLM Suite`.
