"""Entry point: parses command line options and starts the NiceGUI server."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import urllib.request

from . import APP_NAME, __version__, config
from .platform_utils import is_wsl, open_url, platform_label

PING_PATH = '/api/rulz-ping'


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if sys.platform != 'win32':
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _already_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{PING_PATH}', timeout=1.5) as r:
            return json.loads(r.read().decode()).get('app') == APP_NAME
    except Exception:
        return False


def main(argv: list[str] | None = None) -> None:
    settings = config.get()
    parser = argparse.ArgumentParser(prog='rulz-hf-manager', description=APP_NAME)
    parser.add_argument('--host', default=None, help=f'Interface to listen on (default: {settings.host})')
    parser.add_argument('--port', type=int, default=None, help=f'Port (default: {settings.port})')
    parser.add_argument('--no-browser', action='store_true', help='Do not open a browser window on start')
    parser.add_argument('--version', action='version', version=f'{APP_NAME} {__version__}')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    for noisy in ('httpx', 'httpx2', 'httpcore', 'httpcore2'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    host = args.host or settings.host or '127.0.0.1'
    port = args.port or settings.port or 8765
    url = f'http://localhost:{port}'

    if not _port_free(host, port):
        if _already_running(port):
            print(f'{APP_NAME} is already running at {url} - opening it.')
            if not args.no_browser:
                open_url(url)
            return
        for candidate in range(port + 1, port + 30):
            if _port_free(host, candidate):
                print(f'Port {port} is busy, using {candidate} instead.')
                port, url = candidate, f'http://localhost:{candidate}'
                break

    from nicegui import app, ui

    from .ui import browser, dashboard, settings_page, theme  # noqa: F401  (registers the pages)

    theme.setup()

    @app.get(PING_PATH)
    def ping() -> dict:
        return {'app': APP_NAME, 'version': __version__}

    print(f'\n  {APP_NAME} {__version__} on {platform_label()}')
    print(f'  Open {url} in your browser  (Ctrl+C to stop)\n')
    if host not in {'127.0.0.1', 'localhost'}:
        print('  WARNING: listening on the network. Anyone who can reach this port can use your HF token.\n')

    # NiceGUI's own browser opening does not work from inside WSL; open the Windows browser ourselves.
    show = not args.no_browser and not is_wsl()
    if is_wsl() and not args.no_browser:
        app.on_startup(lambda: open_url(url))

    ui.run(
        host=host,
        port=port,
        title=APP_NAME,
        favicon=theme.FAVICON,
        dark=theme.saved_dark(),  # pages start in the saved look; theme.save_look() keeps this up to date
        reload=False,
        show=show,
        show_welcome_message=False,
        reconnect_timeout=10.0,
    )


if __name__ == '__main__':
    main()
