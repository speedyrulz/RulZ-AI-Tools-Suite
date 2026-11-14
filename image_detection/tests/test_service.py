"""Unit tests against a fake llama-server (no GPU needed)."""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from image_detection.config import Settings
from image_detection.llm import LlamaClient, extract_json_object, match_probabilities, parse_classify_output
from image_detection.preprocess import prepare_image
from image_detection.prompts import build_user_message
from image_detection.rubrics import REFERENCE_RUBRICS, enforce_adult_subject
from image_detection.server import create_app
from image_detection.service import DetectionService

RUBRIC_A = "SUBJECT: a red disc. REQUIRE: red. EXCLUDE: orange. FALSE IF: not visible."
RUBRIC_B = "SUBJECT: a blue square. REQUIRE: blue. EXCLUDE: purple. FALSE IF: not visible."


def png_b64(size=(400, 300), color="red", mode="RGB") -> str:
    im = Image.new(mode, size, color)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeLlama:
    """Records chat requests and answers from a scripted queue (or a default responder)."""

    def __init__(self):
        self.requests: list[dict] = []
        self.queue: list = []  # items: str content, (content, finish_reason), or Exception
        self.down = False
        self.stalled = False  # takes chat requests but never answers (httpx.ReadTimeout)
        self.model_path = "C:/models/Test-Q8_0.gguf"

    def default(self, body: dict) -> str:
        tag_ids = body["response_format"]["json_schema"]["schema"]["properties"]["results"]["required"]
        return json.dumps(
            {"results": {t: {"match": True, "confidence": 0.9, "evidence": f"saw {t}"} for t in tag_ids}}
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/props":
            return httpx.Response(200, json={"model_path": self.model_path})
        if path == "/slots":
            return httpx.Response(200, json=[{"is_processing": False}, {"is_processing": True}])
        if path == "/v1/chat/completions":
            body = json.loads(request.content)
            self.requests.append(body)
            if self.stalled:
                raise httpx.ReadTimeout("timed out", request=request)
            item = self.queue.pop(0) if self.queue else self.default(body)
            if isinstance(item, Exception):
                raise item
            logprobs = None
            if isinstance(item, dict):  # {"content": ..., "logprobs": [token dicts]}
                content, finish, logprobs = item["content"], "stop", item["logprobs"]
            else:
                content, finish = item if isinstance(item, tuple) else (item, "stop")
            choice = {"message": {"content": content}, "finish_reason": finish}
            if logprobs is not None:
                choice["logprobs"] = {"content": logprobs}
            return httpx.Response(200, json={"choices": [choice]})
        return httpx.Response(404)


@pytest.fixture
def fake():
    return FakeLlama()


@pytest.fixture
def client(fake, tmp_path):
    s = Settings(cache_path=tmp_path / "cache.sqlite3", llama_url="http://llama")
    llama = LlamaClient(s, httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))
    svc = DetectionService(s, llama=llama)
    with TestClient(create_app(s, svc)) as c:
        yield c


def classify(client, images, tags, **kw):
    return client.post("/classify", json={"images": images, "tags": tags, **kw})


TAGS = [{"id": "red_disc", "rubric": RUBRIC_A}, {"id": "blue_square", "rubric": RUBRIC_B}]


# ---- parsing ---------------------------------------------------------------------------


def test_extract_strips_fences_and_prose():
    assert extract_json_object('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_missing_tag_becomes_false_others_stand():
    out, missing = parse_classify_output('{"results":{"a":{"match":true,"confidence":0.8,"evidence":"x"}}}', ["a", "b"])
    assert out["a"].match and out["a"].confidence == 0.8
    assert missing == {"b"} and out["b"].evidence == "missing" and out["b"].confidence == 0


def test_user_message_lists_exact_keys():
    msg = build_user_message([("man_blonde_hair", "R1"), ("woman_long_fingers", "R2")])
    assert '1. id="man_blonde_hair" — R1' in msg
    assert '{"results":{"man_blonde_hair":{"match":false,"confidence":0.0,"evidence":""},' in msg


# ---- preprocessing ---------------------------------------------------------------------


def test_resize_never_upscales_and_caps_long_edge():
    raw = base64.b64decode(png_b64((3000, 1500)))
    p = prepare_image(raw, max_edge=1024, min_short_edge=96)
    assert (p.width, p.height) == (1024, 512) and p.mime == "image/jpeg"
    small = prepare_image(base64.b64decode(png_b64((500, 400))), 1024, 96)
    assert (small.width, small.height) == (500, 400)


def test_transparency_kept_as_png():
    raw = base64.b64decode(png_b64((300, 300), (255, 0, 0, 0), "RGBA"))
    assert prepare_image(raw, 1024, 96).mime == "image/png"


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def test_16bit_grayscale_is_scaled_not_clipped_to_white():
    from PIL import ImageStat

    w, h = 512, 256
    im = Image.new("I;16", (w, h))
    im.putdata([x * 65535 // (w - 1) for x in range(w)] * h)  # black to white, left to right
    raw = _png(im)
    assert Image.open(io.BytesIO(raw)).mode == "I;16"
    p = prepare_image(raw, 1024, 96)
    out = Image.open(io.BytesIO(p.data)).convert("L")
    assert 120 < ImageStat.Stat(out).mean[0] < 136  # was ~254: every value above 255 became white
    lo, hi = out.getextrema()
    assert lo < 10 and hi > 245

    # big enough for the fast reduce step, which Pillow refuses on I;16 (this used to raise ValueError)
    big = prepare_image(_png(Image.new("I;16", (7000, 200), 30000)), 1024, 96)
    assert big.width == 1024
    assert ImageStat.Stat(Image.open(io.BytesIO(big.data)).convert("L")).mean[0] == pytest.approx(30000 / 256, abs=2)


def test_high_bit_depth_modes_to_8bit_others_untouched():
    from image_detection.preprocess import _to_8bit

    for mode in ("I", "F"):
        im = Image.new(mode, (256, 1))
        im.putdata([x * 16 for x in range(256)])  # 0..4080: scaled by 255 / 4080
        out = _to_8bit(im)
        assert out.mode == "L" and out.getextrema() == (0, 255)
        assert out.getpixel((128, 0)) == round(128 * 16 * 255 / 4080)
    small = Image.new("I", (4, 1))
    small.putdata([0, 10, 200, 255])  # already within 0..255: values kept
    assert list(_to_8bit(small).tobytes()) == [0, 10, 200, 255]
    for mode in ("1", "L", "LA", "P", "RGB", "RGBA", "CMYK"):
        im = Image.new(mode, (4, 4))
        assert _to_8bit(im) is im


# ---- service behaviour -----------------------------------------------------------------


def test_classify_happy_path_and_request_shape(client, fake):
    r = classify(client, [{"id": "img1", "data_b64": png_b64()}], TAGS)
    assert r.status_code == 200
    res = r.json()["results"]["img1"]
    assert res["status"] == "ok" and res["cached"] is False
    assert res["tags"]["red_disc"] == {"match": True, "confidence": 0.9, "evidence": "saw red_disc"}
    body = fake.requests[0]
    assert body["temperature"] == 0 and body["max_tokens"] == 60 + 40 * 2
    user = body["messages"][1]["content"]
    assert user[0]["type"] == "image_url" and user[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert user[1]["type"] == "text"


def test_threshold_applied_but_raw_confidence_returned(client):
    r = classify(client, [{"id": "i", "data_b64": png_b64()}], TAGS, threshold=0.95)
    t = r.json()["results"]["i"]["tags"]["red_disc"]
    assert t["match"] is False and t["confidence"] == 0.9


def test_cache_hit_and_incremental_tag(client, fake):
    img = [{"id": "i", "data_b64": png_b64()}]
    classify(client, img, TAGS[:1])
    assert len(fake.requests) == 1
    r = classify(client, img, TAGS[:1])
    assert r.json()["results"]["i"]["cached"] is True and len(fake.requests) == 1
    # Adding one tag runs the model with only that rubric.
    r = classify(client, img, TAGS)
    assert len(fake.requests) == 2
    ids = fake.requests[1]["response_format"]["json_schema"]["schema"]["properties"]["results"]["required"]
    assert ids == ["blue_square"]
    assert r.json()["results"]["i"]["cached"] is False


def test_editing_rubric_invalidates_only_that_tag(client, fake):
    img = [{"id": "i", "data_b64": png_b64()}]
    classify(client, img, TAGS)
    edited = [TAGS[0], {"id": "blue_square", "rubric": RUBRIC_B + " Edited."}]
    classify(client, img, edited)
    ids = fake.requests[1]["response_format"]["json_schema"]["schema"]["properties"]["results"]["required"]
    assert ids == ["blue_square"]


def test_tiny_image_skipped_never_sent(client, fake):
    r = classify(client, [{"id": "t", "data_b64": png_b64((50, 50))}], TAGS)
    assert r.json()["results"]["t"]["status"] == "skipped" and fake.requests == []


def test_bad_image_does_not_fail_batch(client):
    r = classify(client, [{"id": "bad", "data_b64": "bm90IGFuIGltYWdl"}, {"id": "ok", "data_b64": png_b64()}], TAGS)
    res = r.json()["results"]
    assert res["bad"]["status"] == "error" and "message" in res["bad"]
    assert res["ok"]["status"] == "ok"


def test_parse_failure_retries_once_then_succeeds(client, fake):
    fake.queue = ["I think the image shows a red disc."]
    r = classify(client, [{"id": "i", "data_b64": png_b64()}], TAGS)
    assert r.json()["results"]["i"]["status"] == "ok"
    assert len(fake.requests) == 2
    assert fake.requests[1]["messages"][1]["content"][1]["text"].endswith("Return only the JSON object.")
    assert fake.requests[1]["max_tokens"] == fake.requests[0]["max_tokens"]


def test_truncation_retry_doubles_max_tokens(client, fake):
    fake.queue = [('{"results":{"red_disc":{"match":tr', "length")]
    classify(client, [{"id": "i", "data_b64": png_b64()}], TAGS)
    assert fake.requests[1]["max_tokens"] == fake.requests[0]["max_tokens"] * 2


def test_two_failures_error_and_nothing_cached(client, fake):
    fake.queue = ["nope", "still nope"]
    img = [{"id": "i", "data_b64": png_b64()}]
    r = classify(client, img, TAGS)
    assert r.json()["results"]["i"]["status"] == "error"
    r = classify(client, img, TAGS)  # default responder now; must hit the model again
    assert r.json()["results"]["i"]["cached"] is False and len(fake.requests) == 3


def test_missing_key_not_cached(client, fake):
    fake.queue = ['{"results":{"red_disc":{"match":true,"confidence":0.9,"evidence":"x"}}}']
    img = [{"id": "i", "data_b64": png_b64()}]
    r = classify(client, img, TAGS)
    assert r.json()["results"]["i"]["tags"]["blue_square"]["evidence"] == "missing"
    classify(client, img, TAGS)
    ids = fake.requests[1]["response_format"]["json_schema"]["schema"]["properties"]["results"]["required"]
    assert ids == ["blue_square"]


def test_more_than_eight_tags_split_across_calls(client, fake):
    tags = [{"id": f"tag_{i}", "rubric": f"SUBJECT: thing {i}."} for i in range(11)]
    r = classify(client, [{"id": "i", "data_b64": png_b64()}], tags)
    assert len(r.json()["results"]["i"]["tags"]) == 11
    sizes = sorted(len(b["response_format"]["json_schema"]["schema"]["properties"]["results"]["required"]) for b in fake.requests)
    assert sizes == [3, 8]


def test_model_server_down_returns_503(client, fake):
    fake.down = True
    r = classify(client, [{"id": "i", "data_b64": png_b64()}], TAGS)
    assert r.status_code == 503 and r.json() == {"error": "model server down"}


def test_server_dying_mid_request_returns_503(client, fake):
    fake.queue = [httpx.RemoteProtocolError("server disconnected")]
    r = classify(client, [{"id": "i", "data_b64": png_b64()}], TAGS)
    assert r.status_code == 503


def test_validation_limits(client):
    imgs = [{"id": str(i), "data_b64": "x"} for i in range(65)]
    assert classify(client, imgs, TAGS).status_code == 422
    assert classify(client, [{"id": "a", "data_b64": "x"}], [{"id": "Bad Id", "rubric": "r"}]).status_code == 422
    assert classify(client, [{"id": "a"}], TAGS).status_code == 422


def test_health(client, fake):
    assert client.get("/health").json()["model"] == "Test-Q8_0"
    h = client.get("/health").json()
    assert h["server"] == "up" and h["slots_free"] == 1
    fake.down = True
    assert client.get("/health").json()["server"] == "down"


def test_expand_tag_cached_and_normalized(client, fake):
    fake.queue = ['{"id":"Men With Blonde Hair Today Extra","rubric":"SUBJECT: a man.  REQUIRE: blonde."}']
    r = client.post("/expand_tag", json={"text": "men with blonde hair"})
    assert r.json() == {
        "id": "men_with_blonde_hair",
        "rubric": "SUBJECT: an adult, not a child or teenager: a man. REQUIRE: blonde. EXCLUDE: children or teenagers.",
    }
    client.post("/expand_tag", json={"text": "  Men with   blonde hair "})
    assert len(fake.requests) == 1


def test_expand_prompt_carries_age_rule(client, fake):
    fake.queue = ['{"id":"x","rubric":"SUBJECT: a cat."}']
    client.post("/expand_tag", json={"text": "cats"})
    user = fake.requests[0]["messages"][1]["content"]
    assert "does not mention a boy, girl, child or kid" in user and "man_blonde_hair" in user


# ---- adult rule --------------------------------------------------------------------------

GENERATED = (
    "SUBJECT: a woman REQUIRE: visible blonde hair on the same person "
    "EXCLUDE: non-human, non-woman, or obscured hair FALSE IF: hair color not visible or not blonde"
)


def test_person_tag_gets_adult_subject_and_child_exclusion():
    out = enforce_adult_subject("women with blonde hair", GENERATED)
    assert out.startswith("SUBJECT: an adult, not a child or teenager: a woman. REQUIRE: visible blonde hair")
    assert "EXCLUDE: children or teenagers; non-human, non-woman, or obscured hair." in out
    assert out.endswith("FALSE IF: hair color not visible or not blonde.")


@pytest.mark.parametrize("text", ["girls with blonde hair", "blonde boy", "a child smiling", "Kids playing", "babies"])
def test_child_words_leave_rubric_alone(text):
    assert enforce_adult_subject(text, GENERATED) == GENERATED


def test_non_person_tag_untouched():
    r = "SUBJECT: a red car. REQUIRE: the body paint is red. EXCLUDE: orange cars. FALSE IF: no car is visible."
    assert enforce_adult_subject("red cars", r) == r


def test_already_adult_rubric_untouched():
    r = REFERENCE_RUBRICS["woman_blonde_hair"]
    assert enforce_adult_subject("women with blonde hair", r) == r


def test_person_detected_from_subject_when_tag_has_no_person_word():
    r = "SUBJECT: a person with freckles. REQUIRE: freckles on the face."
    out = enforce_adult_subject("freckles", r)
    assert "an adult" in out and "EXCLUDE: children or teenagers." in out


def test_men_does_not_match_inside_other_words():
    r = "SUBJECT: a monument. REQUIRE: stone."
    assert enforce_adult_subject("monuments and statements", r) == r


# ---- confidence from token probabilities ----------------------------------------------


def _tok(token: str, alts: dict[str, float] | None = None) -> dict:
    import math

    alts = alts or {token: 1.0}
    return {
        "token": token,
        "logprob": math.log(alts.get(token, 1.0)),
        "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in alts.items()],
    }


def _scored_answer() -> dict:
    """red_disc: says true, P(true)=0.8. blue_square: says false, P(false)=0.6. Both *write* 1.0."""
    tokens = [
        _tok('{"results":{"red_disc":{"match":'),
        _tok("true", {"true": 0.8, "false": 0.2}),
        _tok(',"confidence":1.0,"evidence":"red"},"blue_square":{"match":'),
        _tok("false", {"false": 0.6, " true": 0.3, "true": 0.1}),
        _tok(',"confidence":1.0,"evidence":"no"}}}'),
    ]
    return {"content": "".join(t["token"] for t in tokens), "logprobs": tokens}


def test_match_probabilities_reads_each_tags_value_token():
    ans = _scored_answer()
    probs = match_probabilities(ans["logprobs"], ["red_disc", "blue_square"])
    assert probs["red_disc"] == pytest.approx(0.8)
    assert probs["blue_square"] == pytest.approx(0.4)  # " true" and "true" both count as true
    assert match_probabilities(None, ["red_disc"]) == {}


def test_confidence_comes_from_probabilities_not_written_number(client, fake):
    fake.queue = [_scored_answer()]
    r = classify(client, [{"id": "i", "data_b64": png_b64()}], TAGS, threshold=0.7).json()["results"]["i"]
    assert fake.requests[0]["logprobs"] is True
    assert r["tags"]["red_disc"] == {"match": True, "confidence": 0.8, "evidence": "red"}
    assert r["tags"]["blue_square"]["match"] is False and r["tags"]["blue_square"]["confidence"] == 0.6
    # Cached with the probability, and a stricter threshold now actually separates results.
    again = classify(client, [{"id": "i", "data_b64": png_b64()}], TAGS, threshold=0.9).json()["results"]["i"]
    assert again["cached"] and again["tags"]["red_disc"]["match"] is False


def test_without_logprobs_the_written_confidence_is_kept(client):
    r = classify(client, [{"id": "i", "data_b64": png_b64()}], TAGS).json()["results"]["i"]
    assert r["tags"]["red_disc"]["confidence"] == 0.9


def _value(tok: dict) -> list[dict]:
    """Token stream for one tag "a" whose match value token is `tok`."""
    return [_tok('{"results":{"a":{"match":'), tok, _tok(',"confidence":1.0,"evidence":"x"}}}')]


def test_both_answers_listed_normalised_over_the_two():
    # Unchanged behaviour: variants summed, other tokens ignored, 0.6 / (0.6 + 0.2).
    probs = match_probabilities(_value(_tok("true", {"true": 0.5, " true": 0.1, "false": 0.2, "yes": 0.1})), ["a"])
    assert probs["a"] == pytest.approx(0.75)


def test_only_one_answer_listed_is_not_certain():
    # "false" isn't listed: it gets its largest possible probability, here what the list leaves over
    # (1 - 0.99 = 0.01, below the least likely listed token, 0.03). It used to come out as exactly 1.0.
    probs = match_probabilities(_value(_tok("true", {"true": 0.9, "maybe": 0.06, "yes": 0.03})), ["a"])
    assert probs["a"] == pytest.approx(0.9 / 0.91)
    # here the least likely listed token (0.1) is the tighter bound (1 - 0.6 = 0.4 is left over)
    probs = match_probabilities(_value(_tok("true", {"true": 0.4, " true": 0.1, "yes": 0.1})), ["a"])
    assert probs["a"] == pytest.approx(0.5 / 0.6)
    # and the same the other way round: only "false" listed
    probs = match_probabilities(_value(_tok("false", {"false": 0.7, "no": 0.2})), ["a"])
    assert probs["a"] == pytest.approx(0.1 / 0.8)
    # a list that really holds everything still says certain
    assert match_probabilities(_value(_tok("true", {"true": 1.0})), ["a"])["a"] == pytest.approx(1.0)


def test_without_top_logprobs_confidence_is_the_chosen_tokens_probability(client, fake):
    import math

    def chosen(token: str, p: float) -> dict:
        return {"token": token, "logprob": math.log(p)}  # no top_logprobs

    assert match_probabilities(_value(chosen("true", 0.7)), ["a"])["a"] == pytest.approx(0.7)
    assert match_probabilities(_value(chosen("false", 0.8)), ["a"])["a"] == pytest.approx(0.2)

    tokens = [
        {"token": '{"results":{"red_disc":{"match":', "logprob": 0.0},
        chosen("true", 0.7),
        {"token": ',"confidence":1.0,"evidence":"red"},"blue_square":{"match":', "logprob": 0.0},
        chosen("false", 0.8),
        {"token": ',"confidence":1.0,"evidence":"no"}}}', "logprob": 0.0},
    ]
    fake.queue = [{"content": "".join(t["token"] for t in tokens), "logprobs": tokens}]
    r = classify(client, [{"id": "i", "data_b64": png_b64()}], TAGS, threshold=0).json()["results"]["i"]
    assert r["tags"]["red_disc"]["confidence"] == 0.7 and r["tags"]["blue_square"]["confidence"] == 0.8


# ---- several model servers (one per GPU) ------------------------------------------------


def _two_server_service(tmp_path, parallel=1):
    import asyncio  # noqa: F401

    fakes = [FakeLlama(), FakeLlama()]
    s = Settings(cache_path=tmp_path / "cache.sqlite3", llama_parallel=parallel)
    clients = [
        LlamaClient(s, httpx.AsyncClient(transport=httpx.MockTransport(f.handler)), url=f"http://llama{i}")
        for i, f in enumerate(fakes)
    ]
    return fakes, DetectionService(s, llamas=clients)


def test_work_is_spread_over_both_servers(tmp_path):
    import asyncio

    from image_detection.service import ImageInput, Tag

    fakes, svc = _two_server_service(tmp_path)
    imgs = [ImageInput(str(i), data_b64=png_b64(color=(i * 20, 0, 0))) for i in range(6)]
    out = asyncio.run(svc.classify(imgs, [Tag("red_disc", RUBRIC_A)]))
    assert all(r["status"] == "ok" for r in out.values())
    assert len(fakes[0].requests) > 0 and len(fakes[1].requests) > 0
    assert len(fakes[0].requests) + len(fakes[1].requests) == 6


def test_one_server_down_work_moves_to_the_other(tmp_path):
    import asyncio

    from image_detection import ModelServerDown
    from image_detection.service import ImageInput, Tag

    fakes, svc = _two_server_service(tmp_path, parallel=2)
    fakes[0].down = True
    imgs = [ImageInput(str(i), data_b64=png_b64(color=(0, i * 20, 0))) for i in range(4)]
    out = asyncio.run(svc.classify(imgs, [Tag("red_disc", RUBRIC_A)]))
    assert all(r["status"] == "ok" for r in out.values())
    assert len(fakes[1].requests) == 4

    h = asyncio.run(svc.health())
    assert h["server"] == "up" and [x["up"] for x in h["servers"]] == [False, True]

    fakes[1].down = True
    with pytest.raises(ModelServerDown):
        asyncio.run(svc.classify([ImageInput("x", data_b64=png_b64(color="blue"))], [Tag("red_disc", RUBRIC_A)]))


def test_stalled_server_work_moves_to_the_other(tmp_path):
    import asyncio

    from image_detection.service import ImageInput, Tag

    tag = [Tag("red_disc", RUBRIC_A)]
    fakes, svc = _two_server_service(tmp_path)
    fakes[0].stalled = True  # takes the request, never answers
    out = asyncio.run(svc.classify([ImageInput("a", data_b64=png_b64())], tag))
    assert out["a"]["status"] == "ok"
    assert len(fakes[0].requests) == 1 and len(fakes[1].requests) == 1  # tried there first, then moved

    # kept out of rotation for a while: the next images go straight to the other server
    imgs = [ImageInput(str(i), data_b64=png_b64(color=(0, 0, 40 + i * 20))) for i in range(3)]
    out = asyncio.run(svc.classify(imgs, tag))
    assert all(r["status"] == "ok" for r in out.values())
    assert len(fakes[0].requests) == 1 and len(fakes[1].requests) == 4

    # every server stalled: that image fails on its own, never ModelServerDown (which pauses a crawler run)
    fakes[1].stalled = True
    out = asyncio.run(svc.classify([ImageInput("b", data_b64=png_b64(color="blue"))], tag))
    assert out["b"]["status"] == "error" and "timed out" in out["b"]["message"]
    asyncio.run(svc.aclose())


def test_stalled_single_server_fails_only_the_image(client, fake):
    fake.stalled = True
    r = classify(client, [{"id": "i", "data_b64": png_b64()}], TAGS)
    assert r.status_code == 200
    res = r.json()["results"]["i"]
    assert res["status"] == "error" and "timed out" in res["message"]
    assert len(fake.requests) == 1  # one server: tried once, as before


# ---- raw-bytes cache, in-process bytes, faster resize -------------------------------------


def _jpeg(size=(4000, 3000), color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def test_large_jpeg_draft_still_caps_long_edge():
    p = prepare_image(_jpeg((4000, 3000)), max_edge=1024, min_short_edge=96)
    assert (p.width, p.height) == (1024, 768) and p.mime == "image/jpeg"


def test_too_small_detected_without_decoding():
    from image_detection.preprocess import ImageTooSmall

    with pytest.raises(ImageTooSmall):
        prepare_image(_jpeg((80, 2000)), 1024, 96)


def test_cached_raw_bytes_skip_preprocessing(tmp_path, fake, monkeypatch):
    import asyncio

    from image_detection import service as service_mod
    from image_detection.service import ImageInput, Tag

    s = Settings(cache_path=tmp_path / "cache.sqlite3", llama_url="http://llama")
    svc = DetectionService(s, llama=LlamaClient(s, httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))))
    calls = []
    real = service_mod.prepare_image
    monkeypatch.setattr(service_mod, "prepare_image", lambda *a, **k: calls.append(1) or real(*a, **k))
    raw = _jpeg((2000, 1500))
    tag = Tag("red_disc", RUBRIC_A)

    first = asyncio.run(svc.classify([ImageInput("a", data=raw)], [tag]))["a"]
    assert first["status"] == "ok" and not first["cached"] and len(calls) == 1 and len(fake.requests) == 1

    # same bytes (as base64 this time): answered from the cache without decoding
    again = asyncio.run(svc.classify([ImageInput("b", data_b64=base64.b64encode(raw).decode())], [tag]))["b"]
    assert again["status"] == "ok" and again["cached"] and again["tags"] == first["tags"]
    assert len(calls) == 1 and len(fake.requests) == 1

    # a new tag for the same image needs the prepared image again, and only one model call
    again = asyncio.run(svc.classify([ImageInput("c", data=raw)], [tag, Tag("blue_square", RUBRIC_B)]))["c"]
    assert again["status"] == "ok" and not again["cached"]
    assert len(calls) == 2 and len(fake.requests) == 2

    # different MAX_EDGE: the raw-hash mapping isn't reused
    s.max_edge = 512
    asyncio.run(svc.classify([ImageInput("d", data=raw)], [tag]))
    assert len(calls) == 3
    asyncio.run(svc.aclose())


def test_fetch_url_stops_at_max_bytes():
    import asyncio

    from image_detection.preprocess import ImageError, fetch_url

    big = b"x" * 5000

    def handler(request):
        return httpx.Response(200, content=big)  # no reliable Content-Length check: body is counted

    async def go(limit):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetch_url(c, "http://img/a.jpg", 5, limit)

    assert asyncio.run(go(10_000)) == big
    with pytest.raises(ImageError, match="too large"):
        asyncio.run(go(1000))


# ---- one bad image never fails the others ------------------------------------------------


def test_image_too_large_to_open_fails_only_itself(client, monkeypatch):
    # Pillow refuses images over 2 x MAX_IMAGE_PIXELS with DecompressionBombError, which is not an
    # OSError; it used to escape and fail the whole request.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50_000)  # 400x300 is now "too large"
    images = [{"id": "huge", "data_b64": png_b64((400, 300))}, {"id": "ok", "data_b64": png_b64((200, 150))}]
    res = classify(client, images, TAGS).json()["results"]
    assert res["huge"]["status"] == "error" and "too large" in res["huge"]["message"]
    assert res["ok"]["status"] == "ok"


def test_unexpected_error_fails_only_that_image(tmp_path, fake, monkeypatch):
    import asyncio

    from image_detection import service as service_mod
    from image_detection.service import ImageInput, Tag

    s = Settings(cache_path=tmp_path / "cache.sqlite3", llama_url="http://llama")
    svc = DetectionService(s, llama=LlamaClient(s, httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))))
    bad, good = _jpeg((400, 300), "blue"), _jpeg((400, 300), "red")
    real = service_mod.prepare_image

    def prepare(raw, *a, **k):
        if raw == bad:
            raise RuntimeError("a bug in a decoder")
        return real(raw, *a, **k)

    monkeypatch.setattr(service_mod, "prepare_image", prepare)
    out = asyncio.run(svc.classify([ImageInput("bad", data=bad), ImageInput("ok", data=good)], [Tag("red_disc", RUBRIC_A)]))
    assert out["bad"]["status"] == "error" and "RuntimeError" in out["bad"]["message"]
    assert out["ok"]["status"] == "ok"
    asyncio.run(svc.aclose())


def test_caller_supplied_hash_is_not_computed_again(tmp_path, fake, monkeypatch):
    import asyncio
    import hashlib

    from image_detection import service as service_mod
    from image_detection.service import ImageInput, Tag

    s = Settings(cache_path=tmp_path / "cache.sqlite3", llama_url="http://llama")
    svc = DetectionService(s, llama=LlamaClient(s, httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))))
    hashed = []
    monkeypatch.setattr(service_mod, "_sha256", lambda d: hashed.append(1) or hashlib.sha256(d).hexdigest())
    raw = _jpeg((400, 300))
    tag = Tag("red_disc", RUBRIC_A)
    first = asyncio.run(svc.classify([ImageInput("a", data=raw, sha256=hashlib.sha256(raw).hexdigest())], [tag]))["a"]
    assert first["status"] == "ok" and hashed == []
    # the same bytes without a hash are hashed, and find the cached answer under the same key
    again = asyncio.run(svc.classify([ImageInput("b", data=raw)], [tag]))["b"]
    assert again["cached"] and hashed == [1]
    asyncio.run(svc.aclose())


def test_raising_min_short_edge_skips_a_cached_image(tmp_path, fake):
    import asyncio

    from image_detection.service import ImageInput, Tag

    s = Settings(cache_path=tmp_path / "cache.sqlite3", llama_url="http://llama", min_short_edge=96)
    svc = DetectionService(s, llama=LlamaClient(s, httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))))
    raw, tag = _jpeg((400, 150)), [Tag("red_disc", RUBRIC_A)]
    assert asyncio.run(svc.classify([ImageInput("a", data=raw)], tag))["a"]["status"] == "ok"
    assert asyncio.run(svc.classify([ImageInput("a", data=raw)], tag))["a"]["cached"]
    s.min_short_edge = 200  # the fully cached image must still be checked against the new minimum
    assert asyncio.run(svc.classify([ImageInput("a", data=raw)], tag))["a"]["status"] == "skipped"
    assert len(fake.requests) == 1
    asyncio.run(svc.aclose())


# ---- model id after a server restart -------------------------------------------------------


def test_model_id_read_again_after_server_restart(tmp_path, fake):
    import asyncio

    from image_detection.service import ImageInput, Tag

    s = Settings(cache_path=tmp_path / "cache.sqlite3", llama_url="http://llama", model_id=None)
    svc = DetectionService(s, llama=LlamaClient(s, httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))))
    img, tag = [ImageInput("a", data=_jpeg((400, 300)))], [Tag("red_disc", RUBRIC_A)]

    async def go():
        assert (await svc.health())["model"] == "Test-Q8_0"
        assert not (await svc.classify(img, tag))["a"]["cached"]
        fake.model_path = "C:/models/Other-Q4_K_M.gguf"
        # never seen down: the id stays cached (no /props call on every health poll)
        assert (await svc.health())["model"] == "Test-Q8_0"
        assert (await svc.classify(img, tag))["a"]["cached"]
        # restarted with the other model; a health poll sees it go down and come back
        fake.down = True
        assert (await svc.health())["server"] == "down"
        fake.down = False
        assert (await svc.health())["model"] == "Other-Q4_K_M"
        again = (await svc.classify(img, tag))["a"]
        assert again["status"] == "ok" and not again["cached"]  # the old model's answer isn't reused
        await svc.aclose()

    asyncio.run(go())
    assert len(fake.requests) == 2


def test_model_id_override_kept_after_server_restart(tmp_path, fake):
    import asyncio

    s = Settings(cache_path=tmp_path / "cache.sqlite3", llama_url="http://llama", model_id="Pinned-Q8_0")
    llama = LlamaClient(s, httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))

    async def go():
        fake.down = True
        assert not await llama.is_up()
        fake.down = False
        assert await llama.is_up()
        return await llama.model_id()

    assert asyncio.run(go()) == "Pinned-Q8_0"


# ---- model download script -------------------------------------------------------------


def test_download_finishes_a_complete_part_file_without_a_request(tmp_path, monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "download_model.py"
    spec = importlib.util.spec_from_file_location("download_model_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "DEST", tmp_path)
    monkeypatch.setattr(mod, "remote_size", lambda url: 10)

    def no_request(*a, **k):  # a Range of bytes=10- would get HTTP 416 from the server
        raise AssertionError("no download request expected")

    monkeypatch.setattr(mod.urllib.request, "urlopen", no_request)
    (tmp_path / "model.gguf.part").write_bytes(b"x" * 10)  # complete; the last run stopped before the rename
    mod.download("model.gguf")
    assert (tmp_path / "model.gguf").read_bytes() == b"x" * 10
    assert not (tmp_path / "model.gguf.part").exists()
    assert "model.gguf: done" in capsys.readouterr().out
