#!/usr/bin/env bash
# RulZ Huggingface Manager - launcher for Ubuntu / WSL (and other Linux distros).
# Usage: ./start-linux.sh [--port 8765] [--no-browser]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Separate venv from Windows (.venv) so the same folder can be used from both Windows and WSL.
VENV=".venv-linux"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "python3 not found. Install it with:  sudo apt update && sudo apt install -y python3 python3-venv"
    exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating virtual environment in $VENV ..."
    if ! "$PYTHON" -m venv "$VENV"; then
        echo
        echo "Could not create a virtual environment. On Ubuntu/WSL run:"
        echo "    sudo apt update && sudo apt install -y python3-venv"
        rm -rf "$VENV"
        exit 1
    fi
fi

if ! "$VENV/bin/python" -c "import nicegui, huggingface_hub" >/dev/null 2>&1; then
    echo "Installing dependencies ..."
    "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
    "$VENV/bin/python" -m pip install -r requirements.txt
fi

exec "$VENV/bin/python" -m rulz_hf_manager "$@"
