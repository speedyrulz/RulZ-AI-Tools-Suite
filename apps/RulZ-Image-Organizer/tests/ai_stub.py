"""A stand-in OpenAI-compatible vision server for the AI tests.

It decodes the image of every request and answers by its colour: YES to "Does this photo show: red …?" for
a red picture, the tags "red, square, plain background" for tagging, "Red" to the connection test.
"""
from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

COLOURS = {"red": (220, 20, 30), "green": (20, 200, 40), "blue": (30, 40, 220)}


def colour_of(b64: str) -> str:
    px = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((8, 8)).tobytes()
    mean = tuple(sum(px[i::3]) / (len(px) / 3) for i in range(3))
    return min(COLOURS, key=lambda c: sum((a - b) ** 2 for a, b in zip(COLOURS[c], mean)))


def save_photo(path, colour: str, size=(320, 240)):
    Image.new("RGB", size, COLOURS[colour]).save(path, quality=92)
    return str(path)


class StubAI:
    def __init__(self):
        self.requests: list[dict] = []
        self.statuses: list[int] = []       # HTTP status for the next requests (500, 503…), then 200
        self.reject_template_kwargs = False  # answer 400 to bodies with chat_template_kwargs
        self.reply = None                    # optional function(body, colour, text) -> reply text
        self.delay = 0.0
        self.models = ["vis-model", "text-only"]
        self.lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, obj):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.endswith("/v1/models"):
                    self._send(200, {"data": [{"id": m} for m in stub.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with stub.lock:
                    stub.requests.append(body)
                    status = stub.statuses.pop(0) if stub.statuses else 200
                if stub.delay:
                    time.sleep(stub.delay)
                if status != 200:
                    self._send(status, {"error": {"message": f"stub status {status}"}})
                    return
                if stub.reject_template_kwargs and "chat_template_kwargs" in body:
                    self._send(400, {"error": {"message": "Unrecognized request argument: chat_template_kwargs"}})
                    return
                content = body["messages"][-1]["content"]
                text = content[0]["text"]
                colour = colour_of(content[1]["image_url"]["url"].split(",", 1)[1])
                if stub.reply is not None:
                    reply = stub.reply(body, colour, text)
                elif "Does this photo show" in text:
                    m = re.search(r"show: (.*?)\?", text)
                    reply = "YES" if m and colour in m.group(1) else "NO"
                elif "main colour" in text:
                    reply = colour.capitalize()
                else:
                    reply = f"{colour}, square, plain background"
                self._send(200, {"choices": [{"message": {"role": "assistant", "content": reply},
                                              "finish_reason": "stop"}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
