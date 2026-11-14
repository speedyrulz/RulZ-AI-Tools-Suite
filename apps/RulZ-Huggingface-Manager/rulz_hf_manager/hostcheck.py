"""Refuse requests whose Host header is not this computer, while the app listens on 127.0.0.1 only.

Listening on 127.0.0.1 keeps other computers out, but not other web pages in your own browser: a page on any site
can point its own name at 127.0.0.1 ("DNS rebinding") and then load this app as if it were that site, read the
Settings page (your Hugging Face token) and press buttons (uploads from your disk, commands on your vast.ai instance).
The browser still sends that site's name in the Host header, so refusing every name that is not a loopback name
closes the hole. Covers plain HTTP and the websocket NiceGUI talks over.

Extra names (a hosts-file alias, a reverse proxy in front of the app) can be allowed with the environment variable
``RULZ_HF_ALLOWED_HOSTS`` (comma separated); ``*`` turns the check off.
"""

from __future__ import annotations

import ipaddress
import os

LOOPBACK_NAMES = {'localhost', 'localhost.localdomain', 'ip6-localhost', 'ip6-loopback'}


def is_loopback_bind(host: str) -> bool:
    """Whether the server listens on this computer only (the check applies only then)."""
    host = (host or '').strip().strip('[]').lower()
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def host_name(header: str) -> str:
    """The name part of a Host header: ``localhost:8765`` -> ``localhost``, ``[::1]:8765`` -> ``::1``."""
    header = (header or '').strip().lower()
    if header.startswith('['):
        end = header.find(']')
        return header[1:end] if end > 0 else ''
    return header.rsplit(':', 1)[0] if header.count(':') == 1 else header


def allowed(header: str, extra: frozenset[str] = frozenset()) -> bool:
    name = host_name(header).rstrip('.')
    if not name:
        return False
    if name in LOOPBACK_NAMES or name.endswith('.localhost') or name in extra:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def extra_hosts_from_env() -> frozenset[str] | None:
    """Names from RULZ_HF_ALLOWED_HOSTS; None when it says ``*`` (no check at all)."""
    raw = os.environ.get('RULZ_HF_ALLOWED_HOSTS', '')
    names = {n.strip().lower().rstrip('.') for n in raw.split(',') if n.strip()}
    if '*' in names:
        return None
    return frozenset(names)


class LocalHostOnly:
    """Pure ASGI middleware (so it sees websockets too, which Starlette's BaseHTTPMiddleware skips)."""

    def __init__(self, app, extra: frozenset[str] = frozenset()) -> None:
        self.app = app
        self.extra = extra

    async def __call__(self, scope, receive, send) -> None:
        if scope.get('type') not in {'http', 'websocket'}:
            await self.app(scope, receive, send)
            return
        host = ''
        for key, value in scope.get('headers') or []:
            if key == b'host':
                host = value.decode('latin-1')
                break
        if allowed(host, self.extra):
            await self.app(scope, receive, send)
            return
        if scope['type'] == 'websocket':
            await receive()  # the websocket.connect message; closing before accepting refuses the handshake
            await send({'type': 'websocket.close', 'code': 1008})
            return
        body = b'This address is not allowed. Open the app at http://localhost:<port> on this computer.\n'
        await send({'type': 'http.response.start', 'status': 403,
                    'headers': [(b'content-type', b'text/plain; charset=utf-8'),
                                (b'content-length', str(len(body)).encode())]})
        await send({'type': 'http.response.body', 'body': body})
