"""AI tagging and Look for: settings, the Training Captioner's settings, replies, the cache and the jobs."""
import json
import time

import pytest

from ai_stub import StubAI, save_photo
from rulz import search
from rulz.ai import client, settings as ai_settings


@pytest.fixture
def stub():
    s = StubAI()
    yield s
    s.close()


@pytest.fixture
def captioner_dir(tmp_path, monkeypatch):
    d = tmp_path / "captioner"
    d.mkdir()
    monkeypatch.setenv("RULZ_CAPTIONER_DIR", str(d))
    return d


def _photos(db, tmp_path, colours):
    """Library rows for solid-colour photos; returns their ids in order."""
    db.x("INSERT OR IGNORE INTO folders(id, path) VALUES(1, ?)", (str(tmp_path),))
    ids = []
    for i, c in enumerate(colours):
        path = save_photo(tmp_path / f"{i:02d}-{c}.jpg", c)
        ids.append(db.x("INSERT INTO photos(folder_id, path, fp, taken_at, faces_done) VALUES(1,?,?,?,1)",
                        (path, f"fp{i}", 1000 - i)).lastrowid)
    return ids


def _own(stub, **kw):
    s = ai_settings.load()
    s.update({"mode": "own", "urls": stub.url, "model": "vis-model", **kw})
    ai_settings.save(s)
    return s


# ---- settings ---------------------------------------------------------------------------------------------
def test_settings_defaults_roundtrip_and_blank_default_prompts(data_dir):
    s = ai_settings.load()
    assert s["mode"] == "off" and s["urls"].endswith(":11434/v1")
    s.update(mode="own", model="m", tag_system=ai_settings.TAG_SYSTEM + "  ", look_user="Is {query} here?")
    ai_settings.save(s)
    saved = json.loads(ai_settings.settings_file().read_text(encoding="utf-8"))
    assert saved["tag_system"] == ""  # the built-in prompt: later versions can improve it
    assert saved["look_user"] == "Is {query} here?"
    s2 = ai_settings.load()
    assert s2["mode"] == "own" and ai_settings.prompt(s2, "tag_system") == ai_settings.TAG_SYSTEM
    assert ai_settings.look_user_prompt(s2, "red sweater") == "Is red sweater here?"
    s2["look_user"] = "Answer YES or NO."
    assert ai_settings.look_user_prompt(s2, "a dog") == "Answer YES or NO.\na dog"


def test_settings_bad_values_and_broken_file(data_dir):
    path = ai_settings.settings_file()
    path.write_text(json.dumps({"mode": "nonsense", "timeout": 999999, "temperature": "hot", "concurrency": True,
                                "unknown": 1}), encoding="utf-8")
    s = ai_settings.load()
    assert s["mode"] == "off" and s["timeout"] == 3600 and s["temperature"] == 0.1 and s["concurrency"] == 1
    path.write_text("{not json", encoding="utf-8")
    assert ai_settings.load() == ai_settings.DEFAULTS
    assert not path.exists() and list(path.parent.glob("ai_settings.bad-*.json"))  # kept for repair


def test_endpoints_and_loopback():
    eps = ai_settings.endpoints_from_urls("127.0.0.1:11434, http://box:1234/v1/chat/completions/\n\nhttp://box:1234",
                                          "m", 2)
    assert [e["url"] for e in eps] == ["http://127.0.0.1:11434/v1", "http://box:1234/v1"]
    assert all(e["slots"] == 2 and e["model"] == "m" for e in eps)
    assert ai_settings.is_loopback("http://localhost:1/v1") and ai_settings.is_loopback("http://[::1]:5/v1")
    assert not ai_settings.is_loopback("http://192.168.1.5:11434/v1")


def test_resolve_modes(data_dir, captioner_dir):
    s = dict(ai_settings.DEFAULTS)
    assert "off" in ai_settings.resolve(s).problem
    s.update(mode="own", model="")
    assert "Choose a model" in ai_settings.resolve(s).problem
    s.update(model="llava", urls="http://10.0.0.2:11434")
    conn = ai_settings.resolve(s)
    assert conn.problem is None and conn.remote and conn.endpoints[0]["url"] == "http://10.0.0.2:11434/v1"

    s["mode"] = "captioner"
    assert "weren't found" in ai_settings.resolve(s).problem
    (captioner_dir / "settings.json").write_text("{oops", encoding="utf-8")
    assert "isn't valid JSON" in ai_settings.resolve(s).problem
    (captioner_dir / "settings.json").write_text(json.dumps({
        "backend": "external", "external_urls": "http://127.0.0.1:11434/v1", "external_model": "qwen2.5vl:7b",
        "api_key": "k", "slots_per_server": 3, "ctx_size": "bad"}), encoding="utf-8")
    conn = ai_settings.resolve(s)
    assert conn.problem is None and conn.source == "captioner" and conn.kind == "external"
    assert conn.endpoints[0]["slots"] == 3 and conn.endpoints[0]["model"] == "qwen2.5vl:7b" and conn.api_key == "k"
    assert conn.captioner["ctx_size"] == 4096  # a wrong type falls back to the Captioner's default
    (captioner_dir / "settings.json").write_text(json.dumps({"backend": "managed", "model": ""}), encoding="utf-8")
    assert "no model set" in ai_settings.resolve(s).problem
    (captioner_dir / "settings.json").write_text(json.dumps({"backend": "managed", "model": "C:\\m\\q.gguf"}),
                                                 encoding="utf-8")
    conn = ai_settings.resolve(s)
    assert conn.problem is None and conn.kind == "managed" and conn.model == "q.gguf"


# ---- replies ----------------------------------------------------------------------------------------------
def test_parse_tags():
    assert client.parse_tags("Tags: Red Sweater, dog,\n1. Beach.\n- sunset; dog; \"Two People\"", 40) == \
        ["red sweater", "dog", "beach", "sunset", "two people"]
    assert client.parse_tags("<think>let me see</think>cat, a | b", 40) == ["cat", "a b"]
    assert client.parse_tags("```\nx, y, z\n```", 2) == ["x", "y"]
    long = "a woman in a red sweater is standing next to a dog"
    assert client.parse_tags(f"{long}, dog", 40) == ["dog"]  # a sentence isn't a tag
    with pytest.raises(client.AIError):
        client.parse_tags(" , ; ", 40)
    with pytest.raises(client.AIError, match="ran out of tokens"):
        client.parse_tags("<think>still going", 40)


@pytest.mark.parametrize("text,answer", [
    ("YES", True), ("no.", False), ("**Yes**, a red sweater.", True), ("No, there isn't.", False),
    ("Not visible", False), ("<think>maybe yes</think>\nNO", False), ("I think yes", True), ("True", True),
    ("Visible: no", False), ("Answer: YES", True), ("No — I can't see a sweater.", False),
])
def test_parse_answer(text, answer):
    assert client.parse_answer(text) is answer


def test_parse_answer_unclear():
    for text in ("maybe", "It could be either yes or no.", "", "Not sure.", "Found: none", "I cannot say yes",
                 "Hard to tell, maybe yes"):
        with pytest.raises(client.AIError):
            client.parse_answer(text)


def test_encode_photo_scales(tmp_path):
    import base64
    import io

    from PIL import Image

    path = save_photo(tmp_path / "big.jpg", "red", (3000, 1500))
    im = Image.open(io.BytesIO(base64.b64decode(client.encode_photo(path, 512))))
    assert im.size == (512, 256) and im.format == "JPEG"


# ---- requests ---------------------------------------------------------------------------------------------
def test_chat_body_retry_and_template_kwargs_fallback(stub, tmp_path):
    ep = {"url": stub.url, "model": "vis-model", "name": "stub", "slots": 1}
    b64 = client.encode_photo(save_photo(tmp_path / "r.jpg", "red"), 256)
    kw = dict(max_tokens=20, temperature=0.1, disable_thinking=True, timeout=10)
    stub.statuses = [503, 500]
    waits = []
    reply, _ = client.chat_with_retry(ep, "key", "sys", "Does this photo show: red sweater?", b64,
                                      wait=lambda t: waits.append(t) or False, **kw)
    assert reply == "YES" and waits == [1.0, 2.0] and len(stub.requests) == 3
    body = stub.requests[-1]
    assert body["model"] == "vis-model" and body["max_tokens"] == 20 and body["messages"][0]["content"] == "sys"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    stub.reject_template_kwargs = True
    reply, _ = client.chat(ep, "", "", "Does this photo show: blue?", b64, **kw)
    assert reply == "NO" and "chat_template_kwargs" not in stub.requests[-1]
    stub.statuses = [404]
    with pytest.raises(client.HTTPStatusError, match="stub status 404"):
        client.chat_with_retry(ep, "", "", "x", b64, wait=lambda t: False, **kw)
    kw["disable_thinking"] = False
    stub.reject_template_kwargs = False
    client.chat(ep, "", "", "x", b64, **kw)
    assert stub.requests[-1]["max_tokens"] == 20 + client.THINKING_EXTRA_TOKENS


def test_unreachable_server_is_down(tmp_path):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # nothing listens here once closed
    ep = {"url": f"http://127.0.0.1:{port}/v1", "model": "m", "name": "gone", "slots": 1}
    with pytest.raises(client.EndpointDown, match="Nothing answered"):
        client.chat_with_retry(ep, "", "", "x", "AAAA", wait=lambda t: False, max_tokens=5, temperature=0,
                               disable_thinking=True, timeout=5)
    with pytest.raises(client.Stopped):
        client.chat_with_retry(ep, "", "", "x", "AAAA", wait=lambda t: True, max_tokens=5, temperature=0,
                               disable_thinking=True, timeout=5)


def test_models_and_vision_test(stub):
    assert client.list_models(stub.url) == ["text-only", "vis-model"]
    ok, detail = client.vision_test({"url": stub.url, "model": "vis-model", "name": "s"}, "", timeout=10,
                                    disable_thinking=True)
    assert ok and "Red" in detail
    stub.reply = lambda body, colour, text: "I can't see images."
    ok, _ = client.vision_test({"url": stub.url, "model": "vis-model", "name": "s"}, "", timeout=10,
                               disable_thinking=True)
    assert not ok


# ---- the cache in the library --------------------------------------------------------------------------------
def test_ai_cache_follows_the_file_and_deletes(db, tmp_path):
    a, b, c = _photos(db, tmp_path, ["red", "green", "blue"])
    assert [r[0] for r in db.ai_todo_tags(None)] == [a, b, c]
    db.ai_save_tags([(a, "fp0", ["red sweater", "dog"], "m"), (b, "fp1", ["cat"], "m"), (999, "x", ["z"], "m")])
    assert [r[0] for r in db.ai_todo_tags(None)] == [c]
    assert [r[0] for r in db.ai_todo_tags([c, a, b], force=True)] == [c, a, b]  # in the order asked for
    sid = db.ai_search_id("  Red   SWEATER ", create=True)
    assert db.ai_search_id("red sweater") == sid and db.ai_search_id('"red sweater"') == sid
    db.ai_save_answers(sid, [(a, "fp0", True, "m"), (b, "fp1", False, "m")])
    assert db.ai_found(sid) == {a}
    assert [r[0] for r in db.ai_todo_search(sid, [a, b, c])] == [c]
    assert db.ai_searches()[0] | {} == {"id": sid, "text": "red sweater", "created_at": pytest.approx(time.time(), abs=60),
                                       "checked": 2, "found": 1}
    info = db.ai_photo(a)
    assert info["tags"] == ["red sweater", "dog"] and not info["stale"] and info["answers"] == [("red sweater", True)]

    db.x("UPDATE photos SET fp='edited' WHERE id=?", (a,))  # the file changed: its results no longer count
    assert db.ai_found(sid) == set() and db.ai_tag_matches("dog") == set()
    assert {r[0] for r in db.ai_todo_tags(None)} == {a, c}
    assert db.ai_photo(a)["stale"] and db.ai_photo(a)["answers"] == []
    assert db.ai_searches()[0]["checked"] == 1

    db.delete_photos([b])
    assert db.one("SELECT COUNT(*) FROM ai_tags WHERE photo_id=?", (b,))[0] == 0
    assert db.one("SELECT COUNT(*) FROM ai_answers WHERE photo_id=?", (b,))[0] == 0
    db.ai_forget_search(sid)
    assert db.one("SELECT COUNT(*) FROM ai_answers")[0] == 0
    db.ai_save_answers(sid, [(c, "fp2", True, "m")])  # a search forgotten mid-job: skipped, no error
    db.ai_save_tags([(c, "old", ["x"], "m")])  # out of date at once: not counted as tagged, but clearable
    assert db.ai_stats()["tagged"] == 0 and db.ai_stats()["rows"] == 2  # with photo a's out-of-date tags
    db.ai_clear()
    assert db.ai_stats() == {"tagged": 0, "searches": 0, "rows": 0}


def test_tag_words_and_plurals(db, tmp_path):
    a, b, c = _photos(db, tmp_path, ["red", "green", "blue"])
    db.ai_save_tags([(a, "fp0", ["red sweater", "dogs"], "m"), (b, "fp1", ["cathedral", "100% cotton"], "m"),
                     (c, "fp2", ["cat_toy"], "m")])
    assert db.ai_tag_matches("sweater") == {a} and db.ai_tag_matches("red sweater") == {a}
    assert db.ai_tag_matches("dog") == {a} and db.ai_tag_matches("Dogs") == {a}
    assert db.ai_tag_matches("cat") == set()  # not "cathedral"
    assert db.ai_tag_matches("100%") == {b} and db.ai_tag_matches("%") == set()
    assert db.ai_tag_matches("cat_toy") == {c} and db.ai_tag_matches("cat toy") == set()
    assert db.ai_tag_matches("sweater dogs") == set()  # words of two different tags


def test_search_uses_ai_results(db, tmp_path):
    from rulz.faces import DetectedFace, FaceAttrs
    import numpy as np

    a, b, c = _photos(db, tmp_path, ["red", "green", "blue"])
    fid = db.insert_face(a, DetectedFace((0.1, 0.1, 0.3, 0.3), 0.9, np.ones(128, np.float32), FaceAttrs()))
    mom = db.create_person("Mom")
    db.assign_faces([fid], mom)
    assert "Name people" in search.search(db, "red sweater").problem  # no AI results yet: the old message
    db.ai_save_tags([(a, "fp0", ["red sweater"], "m"), (b, "fp1", ["beach", "salt"], "m")])
    sid = db.ai_search_id("salt and pepper", create=True)
    db.ai_save_answers(sid, [(b, "fp1", True, "m"), (c, "fp2", True, "m")])
    smile = db.ai_search_id("smiling", create=True)
    db.ai_save_answers(smile, [(c, "fp2", True, "m")])
    assert search.search(db, "sweater").photo_ids == {a}
    assert search.search(db, "Mom, red sweater").photo_ids == {a}
    assert search.search(db, "Mom, beach").photo_ids == set()
    assert search.search(db, "salt and pepper").photo_ids == {b, c}  # a saved search isn't split at "and"
    assert search.search(db, "salt and pepper, beach").photo_ids == {b}
    assert search.search(db, "smiling").photo_ids == set()  # expressions come first (no one is smiling here)
    assert search.search(db, "“smiling”").photo_ids == {c}  # quoted: the AI's answers for exactly this
    assert search.search(db, '"giraffe"').problem.startswith("No AI tag")
    assert search.search(db, "mom").photo_ids == {a}  # people still come first
    r = search.search(db, "giraffe")
    assert r.photo_ids == set() and "Look for" in r.problem


def test_search_ai_words_beat_parts_of_names_and_keep_their_words(db, tmp_path):
    from rulz.faces import DetectedFace, FaceAttrs
    import numpy as np

    a, b, c = _photos(db, tmp_path, ["red", "green", "blue"])
    for photo, name in ((a, "Carl"), (b, "Herman")):
        fid = db.insert_face(photo, DetectedFace((0.1, 0.1, 0.3, 0.3), 0.9, np.ones(128, np.float32), FaceAttrs()))
        db.assign_faces([fid], db.create_person(name))
    db.ai_save_tags([(c, "fp2", ["car", "man", "woman in red dress", "smiling woman", "person"], "m")])
    assert search.search(db, "car").photo_ids == {c}       # not Carl
    assert search.search(db, "man").photo_ids == {c}       # not Herman
    assert search.search(db, "car, man").photo_ids == {c}
    assert search.search(db, "carl").photo_ids == {a}      # a whole name still finds the person
    assert search.search(db, "Her").photo_ids == {b}       # part of a name, while typing it
    assert search.search(db, "woman in red dress").photo_ids == {c}  # "in" isn't dropped
    assert search.search(db, "smiling woman").photo_ids == {c}       # the tag, not "a smiling face"
    assert search.search(db, "person").photo_ids == {c}
    # a NO answer beats a tag that has the words
    sid = db.ai_search_id("red dress", create=True)
    db.ai_save_answers(sid, [(c, "fp2", False, "m")])
    assert search.search(db, "red dress").photo_ids == set()
    db.ai_forget_search(sid)
    assert search.search(db, "red dress").photo_ids == {c}


# ---- jobs -------------------------------------------------------------------------------------------------
def _run(qapp, ctl, job, timeout=60):
    results = []
    ctl.finished.connect(lambda j, msg, ok: results.append((msg, ok)))
    assert ctl.start(job)
    end = time.monotonic() + timeout
    while ctl.running:
        qapp.processEvents()
        if time.monotonic() > end:
            raise TimeoutError
        time.sleep(0.01)
    qapp.processEvents()
    return results[-1]


def test_tag_and_look_jobs_save_and_reuse_results(qapp, db, tmp_path, stub):
    from rulz.ai.jobs import AIController, Job

    ids = _photos(db, tmp_path, ["red", "green", "blue", "red"])
    _own(stub, concurrency=2)
    ctl = AIController()
    msg, ok = _run(qapp, ctl, Job("tag"))
    assert ok and "AI tagged 4 photos" in msg
    assert db.ai_photo(ids[0])["tags"] == ["red", "square", "plain background"]
    assert search.search(db, "plain background").photo_ids == set(ids)
    n = len(stub.requests)
    msg, ok = _run(qapp, ctl, Job("tag"))
    assert ok and "already" in msg and len(stub.requests) == n  # cached: nothing asked again

    msg, ok = _run(qapp, ctl, Job("look", photo_ids=ids[:3], queries=["red sweater", "blue"]))
    assert ok and "“red sweater” found in 1 photo" in msg and "“blue” found in 1 photo" in msg, msg
    assert len(stub.requests) == n + 6
    msg, ok = _run(qapp, ctl, Job("look", queries=["red sweater"]))  # only the one photo not checked yet
    assert ok and "found in 2 photos" in msg and len(stub.requests) == n + 7
    assert search.search(db, "red sweater").photo_ids == {ids[0], ids[3]}
    msg, ok = _run(qapp, ctl, Job("look", photo_ids=[ids[1]], queries=["red sweater"], force=True))
    assert ok and len(stub.requests) == n + 8
    assert all(r["temperature"] == pytest.approx(0.1) for r in stub.requests)


def test_job_counts_failures_and_unreadable_photos(qapp, db, tmp_path, stub):
    from rulz.ai.jobs import AIController, Job

    ids = _photos(db, tmp_path, ["red", "green"])
    db.x("INSERT INTO photos(folder_id, path, fp, faces_done) VALUES(1, ?, 'fpx', 1)", (str(tmp_path / "gone.jpg"),))
    _own(stub)
    stub.reply = lambda body, colour, text: "Hmm, hard to say." if colour == "green" else "YES"
    msg, ok = _run(qapp, AIController(), Job("look", queries=["red"]))
    assert "2 photos failed" in msg, msg
    sid = db.ai_search_id("red")
    assert db.ai_found(sid) == {ids[0]} and len(db.ai_todo_search(sid, None)) == 2  # failures aren't cached


def test_job_stops_when_no_server_answers(qapp, db, tmp_path, data_dir):
    import socket

    from rulz.ai.jobs import AIController, Job

    _photos(db, tmp_path, ["red"] * 6)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    s = ai_settings.load()
    s.update(mode="own", urls=f"http://127.0.0.1:{port}/v1", model="m")
    ai_settings.save(s)
    from rulz.ai import jobs

    orig = client.retry_delay
    client.retry_delay = lambda attempt, retry_after=None: 0.0
    try:
        msg, ok = _run(qapp, AIController(), Job("tag"))
    finally:
        client.retry_delay = orig
    assert not ok and "No AI server is answering" in msg
    assert jobs.log_file().exists()


def test_stop_keeps_what_was_found(qapp, db, tmp_path, stub):
    from rulz.ai.jobs import AIController, Job

    _photos(db, tmp_path, ["red"] * 30)
    _own(stub)
    stub.delay = 0.05
    ctl = AIController()
    results = []
    ctl.finished.connect(lambda j, msg, ok: results.append((msg, ok)))
    ctl.saved.connect(ctl.stop)  # stop as soon as the first results are written
    ctl.start(Job("tag"))
    end = time.monotonic() + 30
    while ctl.running and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    msg, ok = results[-1]
    tagged = db.ai_stats()["tagged"]
    assert not ok and "(stopped)" in msg and 0 < tagged < 30


def test_test_job_and_setup_problems(qapp, data_dir, stub):
    from rulz.ai.jobs import AIController, Job

    ctl = AIController()
    msg, ok = _run(qapp, ctl, Job("test"))
    assert not ok and "off" in msg
    _own(stub)
    msg, ok = _run(qapp, ctl, Job("test"))
    assert ok and msg.startswith("Connected"), msg
    s = ai_settings.load()
    s["model"] = "missing-model"
    ai_settings.save(s)
    stub.statuses = [404]
    msg, ok = _run(qapp, ctl, Job("test"))
    assert not ok and "Models on this server: text-only, vis-model" in msg


def test_connect_timeout_counts_as_down(monkeypatch):
    monkeypatch.setattr(client, "CONNECT_TIMEOUT", 0.3)
    ep = {"url": "http://10.255.255.1:9/v1", "model": "m", "name": "silent", "slots": 1}  # drops packets
    start = time.monotonic()
    with pytest.raises(client.EndpointDown):
        client.chat_with_retry(ep, "", "", "x", "AAAA", wait=lambda t: False, max_tokens=5, temperature=0,
                               disable_thinking=True, timeout=300)
    assert time.monotonic() - start < 20  # four short connect tries, not four 300 s reply waits


def test_a_dead_server_hands_its_photos_to_another(qapp, db, tmp_path, stub):
    import socket

    from rulz.ai.jobs import AIController, Job

    _photos(db, tmp_path, ["red", "green", "blue", "red", "green"])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    _own(stub, urls=f"http://127.0.0.1:{dead}/v1, {stub.url}")
    monkey = client.retry_delay
    client.retry_delay = lambda attempt, retry_after=None: 0.0
    try:
        msg, ok = _run(qapp, AIController(), Job("tag"))
    finally:
        client.retry_delay = monkey
    assert ok and msg == "AI tagged 5 photos.", msg


def test_nothing_to_do_starts_no_server(qapp, db, tmp_path, captioner_dir):
    from rulz.ai.jobs import AIController, Job

    a, = _photos(db, tmp_path, ["red"])
    (captioner_dir / "settings.json").write_text(json.dumps({"backend": "managed", "model": r"C:\m\q.gguf"}),
                                                 encoding="utf-8")
    s = ai_settings.load()
    s["mode"] = "captioner"
    ai_settings.save(s)
    sid = db.ai_search_id("red", create=True)
    db.ai_save_answers(sid, [(a, "fp0", True, "m")])
    ctl = AIController()
    ctl.servers = lambda: pytest.fail("llama-server started for nothing")
    msg, ok = _run(qapp, ctl, Job("look", queries=["red"]))
    assert ok and "all checked earlier" in msg


def test_closing_mid_request_ends_the_process(tmp_path, stub):
    """A request still waiting on the server must not keep the process alive after the app ends."""
    import subprocess
    import sys
    from pathlib import Path

    stub.delay = 30
    app_dir = Path(__file__).resolve().parent.parent
    script = tmp_path / "run.py"
    script.write_text(f"""
import os, sys, time
sys.path.insert(0, {str(app_dir)!r}); sys.path.insert(0, {str(app_dir / 'tests')!r})
os.environ["QT_QPA_PLATFORM"] = "offscreen"; os.environ["RULZ_DATA_DIR"] = {str(tmp_path / 'data')!r}
from PySide6.QtWidgets import QApplication
app = QApplication([])
from ai_stub import save_photo
from rulz.db import Database
from rulz.ai import settings as ai_settings
from rulz.ai.jobs import AIController, Job
db = Database()
db.x("INSERT INTO folders(id, path) VALUES(1, 'x')")
db.x("INSERT INTO photos(folder_id, path, fp) VALUES(1, ?, 'fp')", (save_photo({str(tmp_path / 'p.jpg')!r}, 'red'),))
s = ai_settings.load(); s.update(mode="own", urls={stub.url!r}, model="m"); ai_settings.save(s)
ctl = AIController(); ctl.start(Job("tag"))
time.sleep(1.5)
ctl.shutdown()
print("main done", flush=True)
""", encoding="utf-8")
    start = time.monotonic()
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert "main done" in out.stdout, out.stderr
    assert time.monotonic() - start < 15, "the process waited for the request"
