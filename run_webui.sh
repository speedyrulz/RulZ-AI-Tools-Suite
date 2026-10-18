#!/usr/bin/env bash
# Start the RulzTrainer WebUI.  ./run_webui.sh [port] [host]   (default 7870 on 127.0.0.1; use 0.0.0.0 for remote access)
cd "$(dirname "$0")"
PORT="${1:-7870}"
HOST="${2:-127.0.0.1}"
if [ -x venv/bin/python ]; then PY=venv/bin/python; else PY="$(command -v python3 || command -v python)"; fi
echo "Starting RulzTrainer WebUI on http://$HOST:$PORT"
exec "$PY" webui.py --port "$PORT" --host "$HOST"
