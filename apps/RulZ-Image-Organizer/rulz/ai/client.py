"""Talking to an OpenAI-compatible vision model (Ollama, LM Studio, vLLM, llama-server) and reading its replies.

Standard library only: the app doesn't ship `requests`, and its build drops Qt's network module.
"""
from __future__ import annotations

import base64
import http.client
import io
import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Callable

from .. import imaging
from .settings import is_loopback

MAX_ATTEMPTS = 4             # one try plus three retries
RETRY_STATUSES = {408, 429}  # plus every 5xx
THINKING_EXTRA_TOKENS = 2048  # room for the thinking when it isn't switched off


class AIError(RuntimeError):
    """A request failed; the message is meant for the user."""


class EndpointDown(AIError):
    """The server refused or dropped the connection on every attempt: it isn't running (or isn't there)."""


class Stopped(AIError):
    """The job was stopped while a request waited to be retried."""


class HTTPStatusError(AIError):
    def __init__(self, status: int, text: str, retry_after: str | None = None):
        super().__init__(f"HTTP {status}: {text}")
        self.status = status
        self.text = text
        self.retry_after = retry_after


# ---- HTTP ---------------------------------------------------------------------------------------------
CONNECT_TIMEOUT = 10.0  # seconds to reach the server; the reply gets the (much longer) reply timeout


class ConnectTimeout(ConnectionError):
    """The server didn't accept the connection in time: asleep, firewalled, or gone. Counted as down, so a
    job stops instead of every photo waiting the whole reply timeout."""


def _connect_limited(conn, connect):
    timeout = conn.timeout
    conn.timeout = min(timeout, CONNECT_TIMEOUT) if isinstance(timeout, (int, float)) else CONNECT_TIMEOUT
    try:
        connect()
    except (TimeoutError, socket.timeout):
        raise ConnectTimeout(f"no connection within {CONNECT_TIMEOUT:.0f} s") from None
    finally:
        conn.timeout = timeout
    if conn.sock is not None:
        conn.sock.settimeout(timeout)


class _HTTPConnection(http.client.HTTPConnection):
    def connect(self):
        _connect_limited(self, super().connect)


class _HTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        _connect_limited(self, super().connect)


class _HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_HTTPConnection, req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_HTTPSConnection, req, context=self._context)


_direct = urllib.request.build_opener(urllib.request.ProxyHandler({}), _HTTPHandler, _HTTPSHandler)
_default = urllib.request.build_opener(_HTTPHandler, _HTTPSHandler)


def _opener(url: str):
    # Windows' system proxy (or HTTP_PROXY) would otherwise catch requests to 127.0.0.1: the Training
    # Captioner's health checks never passed behind one.
    return _direct if is_loopback(url) else _default


def _request(url: str, api_key: str, timeout: float, body: dict | None = None) -> dict:
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with _opener(url).open(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", "replace")
        except OSError:
            text = ""
        raise HTTPStatusError(e.code, _server_message(text) or e.reason or "", e.headers.get("Retry-After")) from None
    try:
        return json.loads(raw)
    except ValueError:
        raise AIError(f"{url} didn't answer like an AI server (the reply isn't JSON). Check the address.") from None


def _server_message(text: str) -> str:
    """The error message out of an OpenAI-style error body, or the start of the body."""
    try:
        err = json.loads(text).get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:300]
        if isinstance(err, str):
            return err[:300]
    except (ValueError, AttributeError):
        pass
    return text.strip()[:300]


def _is_timeout(e: BaseException) -> bool:
    if isinstance(e, urllib.error.URLError):
        e = e.reason if isinstance(e.reason, BaseException) else e
    return isinstance(e, (TimeoutError, socket.timeout))


def list_models(url: str, api_key: str = "", timeout: float = 10) -> list[str]:
    """Model names the server offers (GET /v1/models)."""
    try:
        data = _request(url.rstrip("/") + "/models", api_key, timeout)
    except (urllib.error.URLError, OSError) as e:
        raise AIError(_unreachable(url, e)) from None
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []
    return sorted({str(m["id"]) for m in models if isinstance(m, dict) and m.get("id")}, key=str.lower)


def _unreachable(url: str, e: BaseException) -> str:
    if _is_timeout(e):
        return f"{url} didn't answer in time."
    hint = " (for Ollama: is the Ollama app running?)" if ":11434" in url else ""
    return f"Nothing answered at {url}{hint}. Check that the server is running and the address is right."


# ---- images -------------------------------------------------------------------------------------------
def encode_photo(path: str, max_side: int) -> str:
    """The photo, EXIF-rotated and scaled to max_side, as base64 JPEG (HEIC and 16-bit files included)."""
    li = imaging.load(path, max_side)
    buf = io.BytesIO()
    li.image.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---- chat ---------------------------------------------------------------------------------------------
def chat(ep: dict, api_key: str, system: str, user: str, image_b64: str, *, max_tokens: int, temperature: float,
         disable_thinking: bool, timeout: float) -> tuple[str, str | None]:
    """One request with one image. Returns (reply text, finish_reason)."""
    msgs = []
    if system.strip():
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": [
        {"type": "text", "text": user},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
    ]})
    body = {"model": ep["model"], "messages": msgs, "temperature": float(temperature), "stream": False,
            "max_tokens": int(max_tokens) + (0 if disable_thinking else THINKING_EXTRA_TOKENS)}
    if disable_thinking:
        # llama-server and vLLM read this; Ollama and LM Studio ignore fields they don't know
        body["chat_template_kwargs"] = {"enable_thinking": False}
    url = ep["url"] + "/chat/completions"
    try:
        data = _request(url, api_key, timeout, body)
    except HTTPStatusError as e:
        # A strict OpenAI-style server may refuse the field it doesn't know: ask once more without it.
        if e.status == 400 and "chat_template_kwargs" in body and "chat_template_kwargs" in e.text:
            body.pop("chat_template_kwargs")
            data = _request(url, api_key, timeout, body)
        else:
            raise
    try:
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        raise AIError("The server's reply had no answer in it.") from None
    finish = choice.get("finish_reason")
    if not str(content).strip() and finish == "length":
        raise AIError(OUT_OF_TOKENS)
    return str(content), finish


def is_retryable(e: BaseException) -> bool:
    """Connection problems, 408/429 and 5xx are worth another go. A reply that took the whole timeout is
    not: three more tries could hold one photo for a long time."""
    if isinstance(e, HTTPStatusError):
        return e.status in RETRY_STATUSES or 500 <= e.status <= 599
    if _is_timeout(e):
        return False
    # HTTPException: a reply cut off mid-way (IncompleteRead) or garbled by a restarting server
    return isinstance(e, (urllib.error.URLError, ConnectionError, OSError, http.client.HTTPException)) \
        and not isinstance(e, AIError)


def retry_delay(attempt: int, retry_after: str | None = None) -> float:
    try:
        if retry_after is not None:
            return max(0.0, min(30.0, float(retry_after)))
    except (TypeError, ValueError):
        pass
    return float(min(30, 2 ** (attempt - 1)))


def chat_with_retry(ep: dict, api_key: str, system: str, user: str, image_b64: str, *, wait: Callable[[float], bool],
                    **kw) -> tuple[str, str | None]:
    """chat(), retried with backoff. `wait(seconds)` pauses and returns True to give up (a stop)."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return chat(ep, api_key, system, user, image_b64, **kw)
        except Exception as e:  # noqa: BLE001
            if not is_retryable(e):
                if _is_timeout(e):
                    raise AIError(f"{ep['name']} didn't answer within {kw.get('timeout')} s.") from None
                raise
            if attempt == MAX_ATTEMPTS:
                if wait(0):  # stopped meanwhile: the photo wasn't failed by the server
                    raise Stopped("stopped") from None
                if isinstance(e, HTTPStatusError):
                    raise AIError(f"{ep['name']}: {e} (gave up after {MAX_ATTEMPTS} attempts)") from None
                raise EndpointDown(_unreachable(ep["url"], e)) from None
            if wait(retry_delay(attempt, getattr(e, "retry_after", None))):
                raise Stopped("stopped") from None
    raise AssertionError("unreachable")


# ---- reading replies ----------------------------------------------------------------------------------
OUT_OF_TOKENS = ("The model ran out of tokens before answering (it was still thinking). Tick 'Turn off thinking' "
                 "or raise the token limits in Advanced AI settings.")

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_OPEN = re.compile(r"<think>", re.I)
_THINK_CLOSE = re.compile(r"</think>", re.I)
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.I)
_LABEL = re.compile(r"^\s*(?:tags?|keywords?)\s*:\s*", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•·]+|\d+[.)])\s*")
_WORD = re.compile(r"[a-z]+")
YES_WORDS = {"yes", "true"}
NO_WORDS = {"no", "false", "nope"}
ANSWER_LABELS = {"answer", "final", "reply", "visible", "present", "found", "result", "assistant"}
HEDGE = re.compile(r"\b(not sure|not certain|not clear|unsure|maybe|perhaps|unclear|cannot|can'?t|can not|"
                   r"possibly|might|hard to (say|tell))\b")


def strip_thinking(text: str) -> str:
    text = _THINK.sub("", text)
    text = _THINK_CLOSE.split(text)[-1]  # the chat template opened <think> in the prompt
    if _THINK_OPEN.search(text):  # thinking that never finished
        raise AIError(OUT_OF_TOKENS)
    return text.strip()


def clean_tag(t: str) -> str:
    t = _BULLET.sub("", t)
    t = t.replace("|", " ").strip().strip("\"'`*_.:;!?()[]{}").strip()
    return " ".join(t.lower().split())


CUT_OFF_ANSWER = ("The reply was cut off at the token limit before a clear YES or NO (a thinking model?). Tick "
                  "'Turn off thinking' or raise the Look for token limit in Advanced AI settings.")


def parse_answer_reply(text: str, finish: str | None) -> bool:
    """parse_answer() for a reply from the server. A reply cut off at the token limit isn't trusted: it is
    likely unfinished thinking, where a stray "yes" says nothing."""
    if finish == "length":
        raise AIError(CUT_OFF_ANSWER)
    return parse_answer(text)


def parse_tags(text: str, max_tags: int, cut_off: bool = False) -> list[str]:
    """The model's reply as a clean tag list: lowercase, no duplicates, no sentences, at most max_tags.
    `cut_off`: the reply hit the token limit, so its last tag may be half a word and is dropped."""
    if cut_off:
        parts = re.split(r"[,;\n]", text.rstrip())
        text = ",".join(parts[:-1]) if len(parts) > 1 else parts[0]
    text = _FENCE.sub("", strip_thinking(text))
    text = _LABEL.sub("", text)
    tags: list[str] = []
    seen = set()
    for part in re.split(r"[,;\n]+", text):
        t = clean_tag(part)
        if not t or len(t) > 60 or len(t.split()) > 6 or t in seen:
            continue  # empty, or a sentence rather than a tag
        seen.add(t)
        tags.append(t)
        if len(tags) >= max_tags:
            break
    if not tags:
        raise AIError(f"The model's reply had no tags in it: “{text[:80]}”")
    return tags


def parse_answer(text: str) -> bool:
    """True for a YES reply, False for NO. Raises AIError when the reply says neither or hedges: a wrong
    answer would be kept, an error is asked again next time."""
    reply = strip_thinking(text)
    low = reply.lower().replace("’", "'")
    words = _WORD.findall(low)
    while words and words[0] in ANSWER_LABELS:  # "Answer: yes", "Visible: no"
        words = words[1:]
    hedged = HEDGE.search(low) is not None
    if words and words[0] in YES_WORDS:
        return True
    if words and words[0] in NO_WORDS:
        return False
    if words and words[0] == "not" and not hedged:  # "Not visible", "Not in this photo"
        return False
    if hedged:
        raise AIError(f"The model wasn't sure: “{reply[:80]}”")
    yes, no = "yes" in words, "no" in words
    if yes != no:  # "I think yes"
        return yes
    raise AIError(f"Couldn't read a YES or NO in the model's reply: “{reply[:80]}”")


# ---- connection test ----------------------------------------------------------------------------------
def vision_test(ep: dict, api_key: str, *, timeout: float, disable_thinking: bool) -> tuple[bool, str]:
    """Send a plain red square and ask its colour. (sees images, reply)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (96, 96), (220, 20, 30)).save(buf, "JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    start = time.monotonic()
    reply, _ = chat(ep, api_key, "", "What is the main colour of this image? Reply with one word.", b64,
                    max_tokens=32, temperature=0.0, disable_thinking=disable_thinking, timeout=timeout)
    reply = strip_thinking(reply)
    took = time.monotonic() - start
    return "red" in reply.lower(), f"{reply[:60]!r} in {took:.1f} s"
