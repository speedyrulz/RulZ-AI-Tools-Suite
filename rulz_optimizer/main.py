"""Entry point: parses command line options and starts the NiceGUI server."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import urllib.request

from . import APP_NAME, __version__, settings

PING_PATH = "/api/rulz-ping"
DEFAULT_PORT = 8775


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if sys.platform != "win32":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _already_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{PING_PATH}", timeout=1.5) as r:
            return json.loads(r.read().decode()).get("app") == APP_NAME
    except Exception:  # noqa: BLE001
        return False


def main(argv: list[str] | None = None) -> None:
    cfg = settings.get()
    parser = argparse.ArgumentParser(prog="rulz-ai-trainer-optimizer", description=APP_NAME)
    parser.add_argument("--host", default=None, help=f"Interface to listen on (default: {cfg.host})")
    parser.add_argument("--port", type=int, default=None, help=f"Port (default: {cfg.port or DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window on start")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    host = args.host or cfg.host or "127.0.0.1"
    port = args.port or cfg.port or DEFAULT_PORT
    url = f"http://localhost:{port}"

    if not _port_free(host, port):
        # an earlier start may have fallen back to a later port: look for it on the whole fallback range
        running = next((p for p in range(port, port + 30) if not _port_free(host, p) and _already_running(p)), None)
        if running is not None:
            url = f"http://localhost:{running}"
            print(f"{APP_NAME} is already running at {url} - opening it.")
            if not args.no_browser:
                import webbrowser
                webbrowser.open(url)
            return
        for candidate in range(port + 1, port + 30):
            if _port_free(host, candidate):
                print(f"Port {port} is busy, using {candidate} instead.")
                port, url = candidate, f"http://localhost:{candidate}"
                break

    from nicegui import app, ui

    from .ui import advisor_page, knowledge_page, recommend_page, runs_page, settings_page, theme  # noqa: F401
    from .ui import common

    theme.setup()
    common.register_routes()

    @app.get(PING_PATH)
    def ping() -> dict:
        return {"app": APP_NAME, "version": __version__}

    @ui.page("/")
    def index() -> None:
        common.build_main_page()

    print(f"\n  {APP_NAME} {__version__}")
    print(f"  Open {url} in your browser  (Ctrl+C to stop)\n")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("  WARNING: listening on the network. Anyone who can reach this port can read your run folders "
              "and start the AI helpers.\n")

    ui.run(
        host=host,
        port=port,
        title=APP_NAME,
        favicon=theme.FAVICON,
        dark=theme.saved_dark(),
        reload=False,
        show=not args.no_browser,
        show_welcome_message=False,
        reconnect_timeout=10.0,
    )


if __name__ == "__main__":
    main()
