<p align="center">
  <img src="src/renderer/src/assets/logo.png" alt="RulZ" width="420">
</p>

# RulZ AI Tools Suite

One launcher for the RulZ apps. Browse every tool with screenshots and its full README, install it with one
click, keep it updated, and launch it, all from a single black-and-red window (or white-and-red, in light mode).

| Dark (default) | Light |
| --- | --- |
| ![The launcher's home page in dark mode](docs/screenshots/home-dark.png) | ![The launcher's home page in light mode](docs/screenshots/home-light.png) |

![An app's page: screenshot gallery, details, requirements and its README](docs/screenshots/app-page.png)

![Installing an app: folder, shortcuts, optional components and prerequisites](docs/screenshots/install-dialog.png)

| App | What it does | How it installs |
| --- | --- | --- |
| [RulzTrainer](https://github.com/speedyrulz/RulzTrainer) | LoRA trainer for MOSS-Image, FIBO 2.0 and Krea 2, with a guided WebUI | From source, then its `install.bat` (venv + CUDA PyTorch) |
| [RulZ Image Organizer](https://github.com/speedyrulz/RulZ-Image-Organizer) | Offline face grouping and search for your photo library | Portable release zip |
| [RulZ Image Tournament](https://github.com/speedyrulz/RulZ-Image-Tournament) | Finds your favorite image, two at a time | Portable release zip (a single .exe) |
| [RulZ LLM Suite](https://github.com/speedyrulz/rulz-llm-suite) | Local LLMs with llama.cpp: models, chat, MCP tools, API server | Its setup program, run silently |
| [Rulz Training Captioner](https://github.com/speedyrulz/Rulz-Training-Captioner) | Captions image datasets with a local vision LLM | From source (venv), plus an optional llama.cpp CUDA build |
| [RulZ Simple Video Clip Extractor](https://github.com/speedyrulz/rulz-simple-video-clip-extractor) | Frame-accurate clips with lossless cuts and NVENC export | Portable release zip |

## Using the launcher

- **Home** shows a card per app with its screenshot, description and status (installed, update available,
  running). Filter by category or search.
- **Click an app** for its page: a screenshot gallery (click for full size, arrow keys to browse), the full README,
  version details, requirements and shortcuts.
- **Install** asks where to put the app (default `%LOCALAPPDATA%\Programs\RulZ AI Tools\<app>`), whether to add
  **Desktop** and **Start Menu** shortcuts (the Start Menu ones go in a *RulZ AI Tools Suite* folder), and any
  optional parts. Progress, download speed and the setup log show on the page.
- **Launch** starts the installed app. **Update** appears when GitHub has a newer release (or newer commits, for
  apps installed from source; those update with `git pull` when Git is installed).
- The **⋯** menu has *Open install folder*, *Repair* (runs the setup again), *Locate an existing install* (for a
  copy you already have, e.g. a dev checkout), *View on GitHub*, the install log, and *Uninstall*.
- **Uninstall** removes the shortcuts and moves the app's folder to the Recycle Bin (apps with a setup program run
  their own uninstaller first). Installs the launcher found or that you located are only forgotten; their files
  are never touched.
- **Prerequisites** (Python, FFmpeg, Git, an NVIDIA GPU, .NET Framework) are checked on each app's page. When you
  install an app, missing ones it needs are installed first with winget (Python with *Add to PATH*). Anything winget
  can't install (a GPU driver, or any of them on a PC without winget) gets a step-by-step guide with a *Check again*
  button, and the install waits until it's found.

- **Dark or light:** the first launch asks which look you want (dark is the default). Switch any time with the
  sun/moon button in the top bar or under Settings ▸ Appearance.

Settings and the list of installed apps are kept in `%APPDATA%\RulZ AI Tools Suite`.

## Install

Download `RulZ-AI-Tools-Suite-Setup-<version>.exe` from the Releases page and run it.

To run it from source instead, double-click `install.bat` (needs Node.js 22+), then `run.bat`.

## The apps' source

Each app lives in its own repository, included here as a git submodule in `apps/`, pinned to its current release.
The launcher doesn't need them (it downloads apps from GitHub); they're for browsing and working on the source:

```
git clone --recurse-submodules https://github.com/speedyrulz/RulZ-AI-Tools-Suite.git
git submodule update --init          # in an existing clone
git submodule update --remote        # move every app to its latest commit
```

## Development

```
npm install
node node_modules/electron/install.js   # first time only: Electron downloads its binary lazily
npm run dev          # run with hot reload
npm test             # unit tests (catalog, README handling, versions, command lines, archives)
npm run typecheck
npm run build:win    # installer → dist/RulZ-AI-Tools-Suite-Setup-<version>.exe
```

Pushing a `v*` tag builds the installer on GitHub Actions and attaches it to a release.

### Layout

| Part | Where |
| --- | --- |
| The app list: names, descriptions, how each installs and launches | `src/shared/catalog.ts` |
| Screenshots, icons and a README copy for each app | `catalog/<app id>/` |
| Install, update, launch, uninstall, shortcuts | `src/main/apps.ts`, `src/main/shortcuts.ts` |
| GitHub releases and downloads | `src/main/github.ts`, `src/main/download.ts`, `src/main/archive.ts` |
| Requirement checks and winget installs | `src/main/requirements.ts` |
| UI (React + Tailwind) | `src/renderer/src/` |
| Logo and icon generation from `assets/logo-source.webp` | `tools/make_logo.py` |
| How the catalog screenshots were made (demo content only) | `tools/screenshots/` |
| The apps' repositories (submodules) | `apps/` |

### Adding an app

1. Add an entry to `src/shared/catalog.ts`. `install.kind` is `zip` (a portable zip on the latest release),
   `installer` (an NSIS setup .exe on the latest release) or `source` (the repository plus setup commands), and
   `launch.target` is the program to start, relative to the install folder.
2. Add its repository as a submodule (`git submodule add <url> apps/<repo name>`), add it to
   `tools/sync_catalog.py`, and run `npm run sync-catalog` to copy its README, README images and icon into
   `catalog/<id>/`. Run the same command after updating the submodules to refresh every app's README copy.
3. Put 2–4 screenshots in `catalog/<id>/screenshots/` (`01-…` is the card thumbnail).

The launcher shows the README from GitHub when it can, and the copy in `catalog/` when offline.

## License

The source code is released under the MIT License. The RulZ name and logo are not covered by it; all rights reserved.
