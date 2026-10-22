#!/usr/bin/env bash
# Start the RulzTrainer WebUI.  ./run_webui.sh [port] [host] [tailscale]
# (default 7870 on 127.0.0.1; "tailscale" also serves it to your Tailscale devices over HTTPS, see README)
cd "$(dirname "$0")"
PORT="${1:-7870}"
HOST="${2:-127.0.0.1}"
EXTRA=()
if [ "${3:-}" = "tailscale" ]; then EXTRA=(--tailscale); fi
if [ -x venv/bin/python ]; then PY=venv/bin/python; else PY="$(command -v python3 || command -v python)"; fi
echo "Starting RulzTrainer WebUI on http://$HOST:$PORT"
exec "$PY" webui.py --port "$PORT" --host "$HOST" "${EXTRA[@]}"
