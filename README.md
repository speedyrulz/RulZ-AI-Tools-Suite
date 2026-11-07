# RulZ Huggingface Manager

A friendly local web app for managing your Hugging Face **models, datasets, spaces and storage buckets**,
built with [NiceGUI](https://nicegui.io) and the official `huggingface_hub` library.

Moving, renaming, uploading and downloading files is much quicker than on the website or with the CLI:
select files in a table, click a button, done.

## Quick start

### Windows

Double-click **`start-windows.bat`**.

The first run creates a virtual environment (`.venv`) and installs the dependencies. After that the app opens
in your browser at <http://localhost:8765>. You need Python 3.10 or newer
(<https://www.python.org/downloads/>, tick "Add python.exe to PATH" during install).

### WSL / Ubuntu

```bash
cd "/mnt/c/coding/RulZ Huggingface Manager"   # or wherever you put the folder
./start-linux.sh
```

The first run creates `.venv-linux` (kept separate from the Windows `.venv`, so the same folder works from
both Windows and WSL). Under WSL the app opens your **Windows** browser automatically.

If Ubuntu says the `venv` module is missing:

```bash
sudo apt update && sudo apt install -y python3 python3-venv
```

### Options

```text
--port 8765       use another port (also configurable in Settings)
--host 0.0.0.0    listen on the network (not recommended: anyone who can reach it can use your token)
--no-browser      don't open a browser window
```

Example: `start-windows.bat --port 9000` or `./start-linux.sh --no-browser`.

Starting the app a second time just opens the already-running instance.

## First steps

1. Create a token at <https://huggingface.co/settings/tokens>. Pick **Write** access (or a fine-grained token
   with write permission on your repos), otherwise you can only browse and download.
2. In the app, open **Settings**, paste the token and click **Save & test**.
   If you already ran `hf auth login`, the app uses that login automatically (or click **Import** to save it).
   Note: recent `hf` versions log in through the browser with a *temporary* OAuth token that expires after a
   few weeks; Settings shows the expiry date. For a setup that keeps working, save a regular access token.
3. Pick a **download folder** (default: `Downloads/HuggingFace`; under WSL this is your Windows Downloads folder).

The token is stored only on your computer:

| Platform | Settings file |
| --- | --- |
| Windows | `%APPDATA%\RulZ-HF-Manager\settings.json` |
| WSL / Linux | `~/.config/rulz-hf-manager/settings.json` (permissions `600`) |

Windows and WSL keep separate settings, so add the token once in each.

## Features

**My Hub (dashboard)**
- Every model, dataset, space and bucket in your account or organizations in one sortable, filterable table,
  with size, visibility and last update. Type any other username to browse their public repos.
- Storage totals per type.
- Create repos/buckets, rename or transfer them to an org, make them public/private, duplicate, delete
  (delete asks you to type the name).
- Favorites and recently opened repos for one-click access.
- The search box at the top opens anything: `user/name`, a huggingface.co link, or an `hf://` path.

**File browser (repos and buckets)**
- Folder navigation with breadcrumbs, multi-select, filter, sizes, LFS/Xet badges.
- **Upload** files and whole folders straight from disk (best for big model files), or drag & drop in the browser.
  Files that would be overwritten are flagged before you confirm.
- **Download** a selection (or everything in the current folder) to your download folder, keeping the folder
  structure; or straight through the browser (several files are zipped).
- **Rename** (type `sub/new-name` to move at the same time), **Move** and **Copy** with a destination-folder picker.
  Folders are handled recursively.
  - In repos, all changes go into **one commit**. You can set the commit message or open a **pull request** instead.
  - In buckets, moves and copies run **server-side** (nothing is downloaded or re-uploaded).
- **Copy to...** another repo or bucket (server-side whenever the Hub supports it).
- **Delete** files and folders (one commit in repos).
- **Preview and edit** text files (README, JSON, YAML, Python...) with syntax highlighting, and preview images.
  Saving creates a commit (or PR).
- **New folder**: create (nested) folders, then upload into them.
- Repos: switch **branches**, create branches, see the **commit history**.
- Buckets: **Sync** a local folder up to the bucket, or a bucket folder down to your computer, rsync-style:
  preview exactly what will be uploaded/downloaded/deleted, then run it.
- Repos: **Push a whole folder** in one commit, optionally mirroring (deleting remote files that are gone locally).
- Copy the `hf://` path of any folder for use with the `hf` CLI or Python.

**Activity panel**: uploads, downloads and moves run in the background with progress; the panel lists every
job with its result, errors in plain language, and an **Open folder** button for downloads.

## Notes

- The app listens on `127.0.0.1` only by default, so only your own computer can reach it.
- Files you pick with "from this computer" are read directly from disk, so multi-GB uploads don't pass through
  the browser. Under WSL, Windows drives are under `/mnt/c`, `/mnt/d`...; you can also paste a Windows path
  like `C:\Users\me\model` and it is converted automatically.
- Downloads into the download folder include a small `.cache/huggingface` folder used by `huggingface_hub` to
  skip unchanged files next time.
- Requires `nicegui>=4.0` and `huggingface_hub>=2.4` (bucket support).

## Project layout

```text
rulz_hf_manager/
  main.py            command line + server start
  config.py          settings file (token, download folder, preferences)
  hub.py             huggingface_hub wrapper: one file-system interface for repos and buckets
  jobs.py            background jobs with progress
  platform_utils.py  Windows / WSL / Linux helpers
  ui/                NiceGUI pages: dashboard, browser, settings, pickers, dialogs
start-windows.bat    Windows launcher
start-linux.sh       Ubuntu / WSL launcher
```
