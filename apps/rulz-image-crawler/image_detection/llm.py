"""Client for llama-server's OpenAI-compatible endpoint, with defensive output parsing."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Sequence

import httpx

from .cache import TagResult
from .config import Settings
from .preprocess import PreparedImage
from .prompts import (
    EXPAND_SCHEMA,
    EXPAND_SYSTEM_PROMPT,
    EXPAND_USER_TEMPLATE,
    RETRY_SUFFIX,
    SYSTEM_PROMPT,
    build_classify_schema,
    build_user_message,
)

log = logging.getLogger("image_detection.llm")

MISSING = TagResult(match=False, confidence=0.0, evidence="missing")


class ModelServerDown(Exception):
    """llama-server is unreachable or not ready. The service maps this to HTTP 503."""


class ModelError(Exception):
    """A single model call failed (timeout, bad output twice, HTTP error). Fails only that image."""


class OutputParseError(Exception):
    pass


@dataclass
class Stats:
    calls: int = 0
    retries: int = 0
    failures: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)

    def note_retry(self, reason: str) -> None:
        self.retries += 1
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1


_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)


def extract_json_object(text: str) -> Any:
    """Strip markdown fences, then parse the substring from the first '{' to the last '}'."""
    text = _FENCE_RE.sub("", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise OutputParseError("no JSON object in output")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise OutputParseError(f"invalid JSON: {e}") from e


def _coerce_tag(value: Any) -> TagResult | None:
    if not isinstance(value, dict):
        return None
    match = value.get("match")
    conf = value.get("confidence")
    if isinstance(match, str):
        match = {"true": True, "false": False}.get(match.strip().lower())
    if not isinstance(match, bool):
        return None
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        return None
    conf = min(1.0, max(0.0, conf))
    evidence = value.get("evidence", "")
    evidence = evidence if isinstance(evidence, str) else str(evidence)
    return TagResult(match=match, confidence=conf, evidence=evidence[:120])


def parse_classify_output(text: str, tag_ids: Sequence[str]) -> tuple[dict[str, TagResult], set[str]]:
    """Returns ({tag_id: result}, missing_ids). Missing/malformed tag entries become MISSING;
    the other tags stand. Raises OutputParseError if there is no usable JSON at all."""
    obj = extract_json_object(text)
    if not isinstance(obj, dict):
        raise OutputParseError("top level is not an object")
    results = obj.get("results", obj)  # tolerate the model dropping the wrapper
    if not isinstance(results, dict):
        raise OutputParseError("'results' is not an object")
    out: dict[str, TagResult] = {}
    missing: set[str] = set()
    for tag_id in tag_ids:
        parsed = _coerce_tag(results.get(tag_id))
        if parsed is None:
            out[tag_id] = MISSING
            missing.add(tag_id)
        else:
            out[tag_id] = parsed
    return out, missing


_MATCH_KEY_RE = re.compile(r'"match"\s*:\s*$')


def match_probabilities(tokens: Sequence[dict] | None, tag_ids: Sequence[str]) -> dict[str, float]:
    """P(match is true) per tag, read from the log-probabilities of the token generated right after
    each "match": key and normalised over true/false. Tags whose value token isn't found are left out."""
    out: dict[str, float] = {}
    text = ""
    for tok in tokens or ():
        piece = tok.get("token") or ""
        if _MATCH_KEY_RE.search(text[-40:]):
            alts = tok.get("top_logprobs") or [{"token": piece, "logprob": tok.get("logprob", 0.0)}]
            p_true = p_false = 0.0
            for alt in alts:
                word = (alt.get("token") or "").strip().lower()
                if word in ("true", "false"):
                    p = math.exp(float(alt.get("logprob", -math.inf)))
                    if word == "true":
                        p_true += p
                    else:
                        p_false += p
            owner = max(tag_ids, key=lambda t: text.rfind(f'"{t}"'), default=None)
            if owner is not None and p_true + p_false > 0 and text.rfind(f'"{owner}"') >= 0 and owner not in out:
                out[owner] = p_true / (p_true + p_false)
        text += piece
    return out


def apply_probabilities(
    results: dict[str, TagResult], missing: set[str], probs: dict[str, float]
) -> dict[str, TagResult]:
    """Replace the written confidence with the probability of the answer the model gave."""
    out = dict(results)
    for tag_id, p in probs.items():
        if tag_id in out and tag_id not in missing:
            r = out[tag_id]
            out[tag_id] = TagResult(r.match, round(p if r.match else 1.0 - p, 4), r.evidence)
    return out


class LlamaClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None, url: str | None = None):
        self.s = settings
        self.url = (url or settings.llama_url).rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=settings.request_timeout_s)
        self._owns_client = client is None
        self.stats = Stats()
        self._model_id: str | None = settings.model_id

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ---- low level -------------------------------------------------------------------

    async def _chat(self, body: dict) -> tuple[str, str, list[dict] | None]:
        """POST /v1/chat/completions. Returns (content, finish_reason, token logprobs if requested)."""
        try:
            resp = await self._client.post(
                f"{self.url}/v1/chat/completions", json=body, timeout=self.s.request_timeout_s
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError) as e:
            raise ModelServerDown(f"{type(e).__name__}: {e}") from e
        except httpx.TimeoutException as e:
            raise ModelError(f"model request timed out after {self.s.request_timeout_s:.0f}s") from e
        except httpx.HTTPError as e:
            raise ModelError(f"model request failed: {type(e).__name__}: {e}") from e
        if resp.status_code == 503:  # llama-server answers 503 while the model is loading
            raise ModelServerDown("model server is loading")
        if resp.status_code >= 400:
            raise ModelError(f"model server HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
            choice = data["choices"][0]
            logprobs = (choice.get("logprobs") or {}).get("content")
            return choice["message"].get("content") or "", choice.get("finish_reason") or "", logprobs
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise ModelError(f"unexpected model server response: {e}") from e

    # ---- classify --------------------------------------------------------------------

    async def classify(self, image: PreparedImage, tags: Sequence[tuple[str, str]]) -> tuple[dict[str, TagResult], set[str]]:
        """One model call for one image and <= MAX_TAGS_PER_CALL (id, rubric) pairs.
        Retries once on unparseable or truncated output."""
        tag_ids = [t for t, _ in tags]
        user_text = build_user_message(tags)
        schema = build_classify_schema(tag_ids)
        max_tokens = self.s.max_tokens_for(len(tags))

        want_probs = self.s.confidence_from_logprobs

        def body(text: str, mt: int) -> dict:
            extra = {"logprobs": True, "top_logprobs": 5} if want_probs else {}
            return {
                "model": "qwen3-vl-8b",
                "temperature": 0,
                "max_tokens": mt,
                **extra,
                "response_format": {"type": "json_schema", "json_schema": {"name": "filters", "schema": schema}},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image.data_url}},
                            {"type": "text", "text": text},
                        ],
                    },
                ],
            }

        self.stats.calls += 1
        content, finish, logprobs = await self._chat(body(user_text, max_tokens))
        reason = self._check(content, finish, tag_ids)
        if reason is None:
            return self._scored(content, logprobs, tag_ids)

        self.stats.note_retry(reason)
        log.warning("classify retry (%s) image=%s tags=%s output=%r", reason, image.sha256[:12], tag_ids, content[:200])
        retry_tokens = max_tokens * 2 if reason == "truncated" else max_tokens
        self.stats.calls += 1
        content, finish, logprobs = await self._chat(body(user_text + RETRY_SUFFIX, retry_tokens))
        reason2 = self._check(content, finish, tag_ids)
        if reason2 is None:
            return self._scored(content, logprobs, tag_ids)
        self.stats.failures += 1
        raise ModelError(f"model output unusable after retry ({reason2})")

    @staticmethod
    def _scored(content: str, logprobs: list[dict] | None, tag_ids: Sequence[str]) -> tuple[dict[str, TagResult], set[str]]:
        results, missing = parse_classify_output(content, tag_ids)
        return apply_probabilities(results, missing, match_probabilities(logprobs, tag_ids)), missing

    @staticmethod
    def _check(content: str, finish: str, tag_ids: Sequence[str]) -> str | None:
        if finish == "length":
            return "truncated"
        try:
            parse_classify_output(content, tag_ids)
        except OutputParseError:
            return "parse"
        return None

    # ---- rubric expansion ------------------------------------------------------------

    async def expand_tag(self, text: str) -> tuple[str, str]:
        body = {
            "model": "qwen3-vl-8b",
            "temperature": 0,
            "max_tokens": 500,
            "response_format": {"type": "json_schema", "json_schema": {"name": "rubric", "schema": EXPAND_SCHEMA}},
            "messages": [
                {"role": "system", "content": EXPAND_SYSTEM_PROMPT},
                {"role": "user", "content": EXPAND_USER_TEMPLATE.format(tag=text.replace('"', "'"))},
            ],
        }
        last_err: Exception | None = None
        for _ in range(2):
            content, finish, _ = await self._chat(body)
            try:
                obj = extract_json_object(content)
                tag_id, rubric = obj["id"], obj["rubric"]
                if isinstance(tag_id, str) and isinstance(rubric, str) and rubric.strip() and finish != "length":
                    return tag_id, " ".join(rubric.split())
                last_err = OutputParseError("empty or truncated rubric")
            except (OutputParseError, KeyError, TypeError) as e:
                last_err = e
            body["max_tokens"] *= 2
            body["messages"][1]["content"] += RETRY_SUFFIX
        raise ModelError(f"rubric expansion failed: {last_err}")

    # ---- server info -----------------------------------------------------------------

    async def is_up(self) -> bool:
        try:
            r = await self._client.get(f"{self.url}/health", timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def slots_free(self) -> int | None:
        try:
            r = await self._client.get(f"{self.url}/slots", timeout=3)
            if r.status_code != 200:
                return None
            slots = r.json()
            return sum(1 for s in slots if not s.get("is_processing"))
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            return None

    async def model_id(self) -> str:
        """GGUF filename of the loaded model (cache key component). Raises ModelServerDown if unknown."""
        if self._model_id:
            return self._model_id
        try:
            r = await self._client.get(f"{self.url}/props", timeout=5)
            if r.status_code == 200:
                path = r.json().get("model_path")
                if path:
                    self._model_id = PurePath(path.replace("\\", "/")).name
                    return self._model_id
            r = await self._client.get(f"{self.url}/v1/models", timeout=5)
            if r.status_code == 200:
                mid = r.json()["data"][0]["id"]
                self._model_id = PurePath(mid.replace("\\", "/")).name
                return self._model_id
        except httpx.HTTPError as e:
            raise ModelServerDown(f"{type(e).__name__}: {e}") from e
        except (ValueError, KeyError, IndexError, TypeError):
            pass
        raise ModelServerDown("could not determine model id from llama-server")
