"""Exercise browser fallback with a local HTTP server and offscreen Qt."""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PySide6.QtWidgets import QApplication

from rulz_comfy_launcher.ui.page_opener import PageOpener


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_for(app, predicate, timeout=4):
    end = time.monotonic() + timeout
    while not predicate() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)
    assert predicate()


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.server.received.set()
            self.server.release.wait(2)
            self.send_response(self.server.code)
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.code = 200
    httpd.received = threading.Event()
    httpd.release = threading.Event()
    httpd.release.set()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.release.set()
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.mark.parametrize("code,fallback", [(200, False), (302, False), (404, True), (503, True)])
def test_http_results(app, server, code, fallback):
    server.code = code
    urls, messages = [], []
    opener = PageOpener(lambda url: urls.append(url.toString()), messages.append, app)
    base = f"http://127.0.0.1:{server.server_port}"
    opener.open("id", "Main", base, True, lambda: True)
    wait_for(app, lambda: bool(urls))
    assert urls == [base if fallback else base + "/rulzflow/"]
    assert bool(messages) is fallback


def test_refused(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    urls = []
    opener = PageOpener(lambda url: urls.append(url.toString()), lambda _: None, app)
    base = f"http://127.0.0.1:{port}"
    opener.open("id", "Main", base, True, lambda: True)
    wait_for(app, lambda: bool(urls))
    assert urls == [base + "/rulzflow/"]


@pytest.mark.parametrize("action", ["double", "quit", "removed"])
def test_pending_check(app, server, action):
    server.release.clear()
    urls = []
    alive = [True]
    opener = PageOpener(lambda url: urls.append(url.toString()), lambda _: None, app)
    base = f"http://127.0.0.1:{server.server_port}"
    args = ("id", "Main", base, True, lambda: alive[0])
    opener.open(*args)
    wait_for(app, server.received.is_set)
    if action == "double":
        opener.open(*args)
    elif action == "quit":
        opener.stop()
    else:
        alive[0] = False
    server.release.set()
    wait_for(app, lambda: not opener._pending)
    assert urls == ([base + "/rulzflow/"] if action == "double" else [])
