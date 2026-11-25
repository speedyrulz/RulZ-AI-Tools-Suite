"""scripts/vast/rulz_files.py and the agent's file commands (file-state, file-done, file-fetch, fetch-state, files --what
tree, jobs --all): the resumable ssh upload, folders, the Hugging Face bucket route, the downloader on the instance.
No ssh and no network beyond a loopback HTTP server: the agent runs in this process on a fake install folder, and a fake
Remote plays ssh (its put command is the real shell text, read back here)."""
import base64
import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))

import rulz_agent as agent  # noqa: E402
import rulz_files as files  # noqa: E402
import rulz_remote as rr  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- a fake instance


class FakeRemote:
    """Plays Remote: agent() runs rulz_agent here in the install folder, run() understands rulz_files' put command."""

    def __init__(self, root: Path):
        self.root = root
        self.agent_calls, self.puts = [], []
        self.put_faults = {}  # put number (1 based) -> bytes of it that arrive before the connection drops
        self.agent_faults = []  # agent call numbers (1 based) that fail as a dropped ssh
        self.corrupt_puts = 0  # this many first puts arrive with a flipped byte
        self.configs = []

    def agent(self, *args, config=None, retry=True, timeout="auto"):
        self.agent_calls.append(args)
        self.configs.append(config)
        if len(self.agent_calls) in self.agent_faults:
            return subprocess_result(255, b"", b"ssh: connection dropped")
        agent.SENT_CONFIG = base64.b64encode(config).decode("ascii") if config else ""
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    code = agent.main(list(args))
                except SystemExit as e:
                    code = e.code if isinstance(e.code, int) else 2
        finally:
            agent.SENT_CONFIG = ""
        return subprocess_result(code, out.getvalue().encode(), err.getvalue().encode())

    def run(self, shell, data=None, in_folder=True, retry=True, timeout=None):
        assert retry is False, "a piece must not be retried by Remote.run: the offset would no longer match"
        m = re.match(r"\{ P=(.+?); D=", shell)
        part = shlex.split(m.group(1))[0]
        offset = int(re.search(r'\)" = (\d+) \]', shell).group(1))
        n = len(self.puts) + 1
        self.puts.append((part, offset, len(data)))
        path = self.root / part
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        if path.stat().st_size != offset:
            return subprocess_result(1, b"", b"")
        if self.corrupt_puts and n <= self.corrupt_puts:
            data = bytes([data[0] ^ 1]) + data[1:]
        keep = self.put_faults.get(n)
        with open(path, "ab") as f:
            f.write(data if keep is None else data[:keep])
        return subprocess_result(0 if keep is None else 255, b"", b"")


def subprocess_result(code, out, err):
    return SimpleNamespace(returncode=code, stdout=out, stderr=err)


@pytest.fixture
def inst(tmp_path, monkeypatch):
    root = tmp_path / "inst"
    (root / "output").mkdir(parents=True)
    (tmp_path / "local dir").mkdir()  # a space in the local path
    monkeypatch.chdir(root)
    monkeypatch.setattr(rr, "STATE", tmp_path / "state")
    return root


@pytest.fixture
def local(tmp_path):
    return tmp_path / "local dir"


def data_file(local, name="my model (v2).safetensors", size=5500, seed=1):
    payload = bytes((i * 7 + seed) % 251 for i in range(size))
    (local / name).write_bytes(payload)
    return local / name, payload


class Sleeps(list):
    def __call__(self, seconds):
        self.append(seconds)


# ---------------------------------------------------------------- names and digests


def test_names_are_cleaned_and_dests_must_stay_below_models_sent():
    assert files.clean_name("my model (v2).safetensors") == "my_model_v2_.safetensors"
    assert files.clean_name("..hidden") == "hidden" and files.clean_name("???") == "_" and files.clean_name("") == "file"
    good = "models/sent/0123456789ab/name.bin"
    assert files.check_dest(good) == good
    for bad in ("models/sent/0123456789ab", "models/other/0123456789ab/x", "models/sent/xyz/x",
                "models/sent/0123456789ab/../x", "/models/sent/0123456789ab/x", "models/sent/0123456789ab/a\\b",
                "models/sent/0123456789ab//x"):
        with pytest.raises(files.FilesError):
            files.check_dest(bad)


def test_digest_is_cached_by_path_size_and_mtime(local, inst):
    path, payload = data_file(local)
    assert files.digest(path) == sha(payload)
    cache = files.cache_path()
    entry = json.loads(cache.read_text(encoding="utf-8"))[str(path.resolve())]
    assert entry["sha256"] == sha(payload)
    entry["sha256"] = "ab" * 32  # proof that an unchanged file is not read again
    cache.write_text(json.dumps({str(path.resolve()): entry}), encoding="utf-8")
    assert files.digest(path) == "ab" * 32
    path.write_bytes(payload[::-1] + b"x")  # changed: read again
    assert files.digest(path) == sha(payload[::-1] + b"x")


def test_tree_digest_changes_with_content_and_ignores_hidden_files(local, inst):
    folder = local / "model dir"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.json").write_text("{}")
    (folder / "sub" / "b.bin").write_bytes(b"1234")
    (folder / ".cache").mkdir()
    (folder / ".cache" / "x").write_bytes(b"zzz")
    (folder / ".hidden").write_text("h")
    first, rows = files.tree_digest(folder)
    assert [r["rel"] for r in rows] == ["a.json", "sub/b.bin"]
    (folder / ".hidden").write_text("changed")
    assert files.tree_digest(folder)[0] == first
    (folder / "sub" / "b.bin").write_bytes(b"1235")
    assert files.tree_digest(folder)[0] != first


# ---------------------------------------------------------------- straight upload


def test_a_file_goes_up_in_pieces_and_is_verified_and_skipped_next_time(inst, local):
    path, payload = data_file(local)
    remote, seen = FakeRemote(inst), []
    out = files.upload_file(remote, path, progress=lambda d, t: seen.append((d, t)), piece=1000)
    dest = f"models/sent/{sha(payload)[:12]}/my_model_v2_.safetensors"
    assert out == {"kind": "file", "dest": dest, "size": 5500, "sha256": sha(payload), "skipped": False}
    assert (inst / dest).read_bytes() == payload and not (inst / (dest + ".part")).exists()
    assert len(remote.puts) == 6 and [p[1] for p in remote.puts] == [0, 1000, 2000, 3000, 4000, 5000]
    assert seen[-1] == (5500, 5500) and [d for d, _ in seen] == sorted(d for d, _ in seen)
    again = files.upload_file(remote, path, piece=1000)
    assert again["skipped"] is True and len(remote.puts) == 6  # nothing sent: the record and the size agree


def test_a_dropped_connection_asks_how_much_arrived_and_goes_on(inst, local):
    path, payload = data_file(local)
    remote, sleeps = FakeRemote(inst), Sleeps()
    remote.put_faults = {3: 400}  # the third piece drops after 400 of its 1000 bytes
    out = files.upload_file(remote, path, piece=1000, sleep=sleeps)
    assert (inst / out["dest"]).read_bytes() == payload
    assert [p[1] for p in remote.puts] == [0, 1000, 2000, 2400, 3400, 4400, 5400]  # went on from byte 2400, no resend
    assert sleeps == [2]
    asked = [c for c in remote.agent_calls if c[0] == "file-state"]
    assert len(asked) == 2  # at the start and after the drop


def test_a_piece_refused_for_its_offset_is_asked_about_again(inst, local):
    path, payload = data_file(local, size=3000)
    remote, sleeps = FakeRemote(inst), Sleeps()
    (inst / "models" / "sent").mkdir(parents=True)
    real_run = remote.run
    state = {"first": True}

    def run(shell, data=None, **kw):
        if state.pop("first", False):  # a late write from an earlier dropped connection lands first
            part = f"models/sent/{sha(payload)[:12]}/my_model_v2_.safetensors.part"
            (inst / part).parent.mkdir(parents=True, exist_ok=True)
            (inst / part).write_bytes(payload[:500])
        return real_run(shell, data, **kw)

    remote.run = run
    out = files.upload_file(remote, path, piece=1000, sleep=sleeps)
    assert (inst / out["dest"]).read_bytes() == payload and sleeps  # exit 1 counted as a failure, state asked again


def test_ten_drops_in_a_row_give_up_with_growing_waits(inst, local):
    path, _ = data_file(local, size=2000)
    remote, sleeps = FakeRemote(inst), Sleeps()
    remote.put_faults = {n: 0 for n in range(1, 40)}
    with pytest.raises(files.FilesError) as e:
        files.upload_file(remote, path, piece=1000, sleep=sleeps)
    assert e.value.kind == "ssh" and sleeps == [2, 4, 8, 16, 32, 60, 60, 60, 60, 60]


def test_a_dropped_agent_call_is_asked_again(inst, local):
    path, payload = data_file(local, size=1500)
    remote, sleeps = FakeRemote(inst), Sleeps()
    remote.agent_faults = [1]  # the first file-state is lost
    out = files.upload_file(remote, path, piece=1000, sleep=sleeps)
    assert (inst / out["dest"]).read_bytes() == payload and sleeps == [2]


def test_a_copy_that_arrives_changed_is_sent_again_once_then_refused(inst, local):
    path, payload = data_file(local, size=1500)
    remote = FakeRemote(inst)
    remote.corrupt_puts = 1  # the first piece of the first try is damaged
    out = files.upload_file(remote, path, piece=1000)
    assert (inst / out["dest"]).read_bytes() == payload  # file-done removed the bad part; the second try was whole
    remote2 = FakeRemote(inst)
    other, _ = data_file(local, name="other.bin", size=1500, seed=9)
    remote2.corrupt_puts = 99
    with pytest.raises(files.FilesError) as e:
        files.upload_file(remote2, other, piece=1000)
    assert e.value.kind == "checksum"


def test_cancel_stops_between_pieces_and_keeps_the_part(inst, local):
    path, payload = data_file(local)
    remote, count = FakeRemote(inst), {"n": 0}

    def cancel():
        count["n"] += 1
        return len(remote.puts) >= 2

    with pytest.raises(files.Cancelled):
        files.upload_file(remote, path, piece=1000, cancel=cancel)
    dest = f"models/sent/{sha(payload)[:12]}/my_model_v2_.safetensors"
    assert (inst / (dest + ".part")).stat().st_size == 2000
    out = files.upload_file(FakeRemote(inst), path, piece=1000)  # a later send continues from 2000
    assert (inst / out["dest"]).read_bytes() == payload


def test_a_local_file_that_changes_while_it_is_sent_is_refused(inst, local):
    path, _ = data_file(local, size=3000)
    remote = FakeRemote(inst)
    real = remote.run

    def run(shell, data=None, **kw):
        out = real(shell, data, **kw)
        path.write_bytes(b"z" * 3000)
        return out

    remote.run = run
    with pytest.raises(files.FilesError) as e:
        files.upload_file(remote, path, piece=1000)
    assert e.value.kind == "local"


def test_put_command_is_one_quoted_group_and_names_only_posix_paths():
    cmd = files.put_command("models/sent/0123456789ab/my dir/it's.bin", 64)
    assert cmd.startswith("{ P=") and cmd.endswith("; }")
    assert shlex.quote("models/sent/0123456789ab/my dir/it's.bin.part") in cmd and "\\" not in cmd.replace("\\\"", "")
    with pytest.raises(files.FilesError):
        files.put_command("models/sent/../x/y/z", 0)


def bash():
    git = shutil.which("git")
    candidates = [Path(git).parents[n] / "usr" / "bin" / "bash.exe" for n in (1, 2)] if git else []
    candidates += [Path(p) for p in ("/bin/bash", "/usr/bin/bash")]
    return next((str(c) for c in candidates if c.exists()), None)


@pytest.mark.skipif(bash() is None, reason="no bash")
def test_the_put_command_appends_only_at_the_exact_offset(tmp_path):
    work = tmp_path / "w x"
    work.mkdir()
    dest = "models/sent/0123456789ab/my dir/f.bin"

    def put(offset, data):
        return subprocess.run([bash(), "-c", f"cd {shlex.quote(work.as_posix())} && {files.put_command(dest, offset)}"],
                              input=data, capture_output=True)

    assert put(0, b"hello").returncode == 0
    assert put(0, b"again").returncode != 0 and put(7, b"x").returncode != 0
    assert put(5, b" world").returncode == 0
    assert (work / (dest + ".part")).read_bytes() == b"hello world"


# ---------------------------------------------------------------- folders


def make_folder(local):
    folder = local / "flux dir"
    (folder / "text encoder").mkdir(parents=True)
    (folder / "index.json").write_text('{"a": 1}')
    (folder / "text encoder" / "config.json").write_text("{}")
    (folder / "text encoder" / "weights one.bin").write_bytes(bytes(range(256)) * 20)  # 5120: "large" below
    (folder / ".cache").mkdir()
    (folder / ".cache" / "junk").write_text("junk")
    return folder


def fake_sender(inst, log, faults=None):
    def sender(remote, items, target, hidden=True, staged=False, marks=None, progress=None):
        log.append((target, [n for _, n in items], staged))
        if faults:
            faults.pop()
            raise SystemExit("tar on the instance ended with 255: the copy may be incomplete")
        for path, name in items:
            dst = inst / target / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
            progress(Path(path).stat().st_size, 0)
        (inst / target / ".rulz_incoming").mkdir(exist_ok=True)  # what a staged send leaves
    return sender


def test_a_folder_goes_up_large_files_one_by_one_and_the_rest_staged(inst, local):
    folder = make_folder(local)
    remote, log, seen = FakeRemote(inst), [], []
    out = files.upload_tree(remote, folder, piece=2000, large=4000, sender=fake_sender(inst, log),
                            progress=lambda d, t: seen.append((d, t)))
    dest = out["dest"]
    assert re.fullmatch(r"models/sent/[0-9a-f]{12}/flux_dir", dest) and out["files"] == 3 and out["skipped"] is False
    assert (inst / dest / "index.json").read_text() == '{"a": 1}'
    assert (inst / dest / "text encoder" / "weights one.bin").read_bytes() == bytes(range(256)) * 20
    assert not (inst / dest / ".cache").exists() and not list((inst / dest).rglob(".rulz_incoming"))
    assert [(t.rsplit("/flux_dir", 1)[1], n, staged) for t, n, staged in log] == [
        ("", ["index.json"], True), ("/text encoder", ["config.json"], True)]
    assert {p[0] for p in remote.puts} == {dest + "/text encoder/weights one.bin.part"}
    assert seen[-1] == (out["size"], out["size"])
    log.clear()
    again = files.upload_tree(remote, folder, piece=2000, large=4000, sender=fake_sender(inst, log))
    assert again["skipped"] is True and not log


def test_a_failed_small_file_copy_is_asked_about_and_repeated(inst, local):
    folder = make_folder(local)
    remote, log, sleeps = FakeRemote(inst), [], Sleeps()
    out = files.upload_tree(remote, folder, piece=2000, large=4000, sender=fake_sender(inst, log, faults=[1]),
                            sleep=sleeps)
    assert (inst / out["dest"] / "index.json").exists() and sleeps == [2]
    assert [t for t, _, _ in log].count(f"{out['dest']}") == 2  # index.json twice: the first drop sent nothing


def test_a_folder_that_arrives_changed_is_refused(inst, local):
    folder = make_folder(local)
    remote = FakeRemote(inst)

    def sender(remote, items, target, hidden=True, staged=False, marks=None, progress=None):
        for path, name in items:
            dst = inst / target / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(Path(path).read_bytes()[:-1] + b"?")  # the same size, other content
    with pytest.raises(files.FilesError) as e:
        files.upload_tree(remote, folder, piece=2000, large=4000, sender=sender)
    assert e.value.kind == "checksum"


def test_send_tells_its_progress_by_bytes(tmp_path):
    folder = tmp_path / "set"
    folder.mkdir()
    (folder / "a.bin").write_bytes(b"x" * 30000)
    code = "import sys; sys.stdin.buffer.read()"
    fake = SimpleNamespace(command=lambda shell, in_folder=True: [sys.executable, "-c", code])
    seen = []
    rr.send(fake, [(folder, "set")], "datasets", progress=lambda b, n: seen.append((b, n)))
    assert sum(b for b, _ in seen) >= 30000 and sum(n for _, n in seen) == 1
    rr.send(fake, [(folder, "set")], "datasets")  # still works without it


# ---------------------------------------------------------------- the agent's own commands


def run_agent(remote, *args, config=None):
    done = remote.agent(*args, config=config)
    return done.returncode, (json.loads(done.stdout) if done.stdout.strip() else None), done.stderr.decode()


def test_paths_outside_models_sent_are_refused_by_the_agent(inst):
    remote = FakeRemote(inst)
    for bad in ("models/sent/../../etc/passwd", "output/x", "models/sent/0123456789ab", "/etc/passwd"):
        code, _, err = run_agent(remote, "file-state", bad)
        assert code == 1 and "Not a path below models/sent" in err
    if hasattr(os, "symlink"):
        (inst / "models" / "sent").mkdir(parents=True)
        try:
            os.symlink(inst / "output", inst / "models" / "sent" / "0123456789ab", target_is_directory=True)
        except OSError:
            return  # no symlink right on this machine
        code, _, err = run_agent(remote, "file-state", "models/sent/0123456789ab/x")
        assert code == 1


def test_file_state_cuts_back_a_part_that_is_too_long_and_file_done_checks_the_length(inst):
    remote, dest = FakeRemote(inst), "models/sent/0123456789ab/x.bin"
    part = inst / (dest + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"abcdef")
    assert run_agent(remote, "file-state", dest, "--size", "10")[1]["have"] == 6
    assert run_agent(remote, "file-state", dest, "--size", "4")[1]["have"] == 0 and not part.exists()
    part.write_bytes(b"abcdef")
    code, _, err = run_agent(remote, "file-done", dest, "--size", "10", "--sha256", sha(b"abcdefghij"))
    assert code == 1 and "6 of 10" in err and part.exists()
    code, answer, _ = run_agent(remote, "file-done", dest, "--size", "6", "--sha256", sha(b"zzz"))
    assert code == 1 and answer["state"] == "mismatch" and not part.exists()


def test_file_done_is_repeatable_after_a_lost_answer(inst):
    remote, dest = FakeRemote(inst), "models/sent/0123456789ab/x.bin"
    part = inst / (dest + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"abcdef")
    args = ("file-done", dest, "--size", "6", "--sha256", sha(b"abcdef"))
    assert run_agent(remote, *args)[1]["state"] == "done"
    assert run_agent(remote, *args)[1]["state"] == "done"  # the part is gone; the record answers
    (inst / "models" / "sent" / ".ok").rename(inst / "models" / "sent" / ".ok2")  # a file without its record
    assert run_agent(remote, *args)[1]["state"] == "done"  # checked again and recorded
    assert run_agent(remote, "file-state", dest, "--size", "6")[1]["done"] is True
    assert run_agent(remote, "file-state", dest, "--size", "7")[1]["done"] is False  # another size is another file


def test_tree_listing_leaves_out_states_remote_render_hidden_and_parts(inst):
    job = inst / "output" / "run1"
    for rel in ("config.yaml", "lora.safetensors", "samples/a.png", "state_000100/x.pt", "remote_render/r.json",
                ".hidden", "samples/.thumb", "probes/p.png.part", "sub/state_000050/y", "sub/ok.txt"):
        f = job / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    (inst / "output" / "run1" / "status.json").write_text("{}")
    remote = FakeRemote(inst)
    code, data, _ = run_agent(remote, "files", "run1", "--what", "tree")
    assert code == 0 and data["base"] == str((inst / "output").resolve())
    assert sorted(f["path"] for f in data["files"]) == [
        "run1/config.yaml", "run1/lora.safetensors", "run1/samples/a.png", "run1/status.json", "run1/sub/ok.txt"]
    assert all(set(f) == {"path", "size", "mtime"} for f in data["files"])
    assert run_agent(remote, "files", "run1", "--what", "tree", "--what", "logs")[0] == 1


def test_jobs_all_lists_every_job_folder_without_a_limit(inst):
    for i in range(130):
        d = inst / "output" / f"job{i:03d}"
        d.mkdir()
        if i % 2:
            (d / "status.json").write_text(json.dumps({"step": 1, "total": 1, "done": True, "run_id": f"r{i}"}))
    for internal in ("_tests", "_watchdog", "_history", ".hidden"):
        (inst / "output" / internal).mkdir()
    remote = FakeRemote(inst)
    code, rows, _ = run_agent(remote, "jobs", "--all")
    assert code == 0 and len(rows) == 130 and {r["job"] for r in rows} == {f"job{i:03d}" for i in range(130)}
    odd = next(r for r in rows if r["job"] == "job001")
    assert odd["state"] == "done" and odd["run_id"] == "r1" and odd["has_status"] is True
    assert next(r for r in rows if r["job"] == "job002")["has_status"] is False
    assert len(run_agent(remote, "jobs")[1]) == 65  # plain jobs: only folders with a status.json


# ---------------------------------------------------------------- Hugging Face buckets


class FakeApi:
    def __init__(self):
        self.existing = {}  # bucket path -> size
        self.added, self.created, self.buckets_list = [], [], []
        self.fail = None

    def bucket_info(self, bucket_id):
        if bucket_id != "me/b":
            raise type("BucketNotFoundError", (Exception,), {})("nope")
        return SimpleNamespace(id=bucket_id, private=True)

    def list_buckets(self, namespace=None):
        return [SimpleNamespace(id="me/b", private=True, size=5), SimpleNamespace(id="me/c", private=False, size=None)]

    def create_bucket(self, bucket_id, private=None, exist_ok=False):
        self.created.append((bucket_id, private, exist_ok))

    def list_bucket_tree(self, bucket_id, prefix=None, recursive=None):
        return [SimpleNamespace(path=p, size=s) for p, s in self.existing.items() if p.startswith(prefix or "")]

    def batch_bucket_files(self, bucket_id, add=None, copy=None, delete=None):
        if self.fail:
            raise self.fail
        for src, dst in add:
            self.added.append((bucket_id, src, dst))
            self.existing[dst] = Path(src).stat().st_size


def test_bucket_calls_and_the_ids_they_accept():
    api = FakeApi()
    assert files.bucket_state("me/b", api=api) == {"id": "me/b", "exists": True, "private": True, "error": None}
    assert files.bucket_state("me/zzz", api=api)["exists"] is False
    assert [b["id"] for b in files.buckets(api=api)] == ["me/b", "me/c"]
    assert files.create_bucket("me/new", api=api) == {"id": "me/new", "private": True}
    assert api.created == [("me/new", True, True)]
    for bad in ("nobucket", "a/b/c", "../x", "me/ b"):
        with pytest.raises(files.FilesError):
            files.bucket_state(bad, api=api)


def test_a_file_goes_to_the_bucket_under_its_digest_and_is_skipped_when_there(local, inst):
    path, payload = data_file(local)
    api, seen = FakeApi(), []
    out = files.upload_to_bucket("me/b", path, api=api, progress=lambda d, t: seen.append((d, t)))
    root = f"rulz-trainer/{sha(payload)[:12]}/my_model_v2_.safetensors"
    assert out["root"] == root and api.added == [("me/b", str(path), root)] and out["skipped"] is False
    assert out["files"] == [{"rel": "", "bucket_path": root, "size": 5500, "sha256": sha(payload)}]
    assert seen[-1] == (5500, 5500)
    again = files.upload_to_bucket("me/b", path, api=api)
    assert again["skipped"] is True and len(api.added) == 1
    api.existing[root] = 1  # the same name at another size is not the same file
    assert files.upload_to_bucket("me/b", path, api=api)["skipped"] is False


def test_a_folder_goes_to_the_bucket_file_by_file(local, inst):
    folder = make_folder(local)
    api = FakeApi()
    out = files.upload_to_bucket("me/b", folder, api=api)
    assert sorted(f["rel"] for f in out["files"]) == ["index.json", "text encoder/config.json",
                                                      "text encoder/weights one.bin"]
    assert all(dst.startswith(out["root"] + "/") for _, _, dst in api.added) and len(api.added) == 3
    api.fail = RuntimeError("secret hf_abcdefghijklmnop failed")
    api.existing.clear()
    with pytest.raises(files.FilesError) as e:
        files.upload_to_bucket("me/b", folder, api=api)
    assert "hf_abcdef" not in str(e.value) and e.value.kind == "hf"


def redirect(location, code=302):
    def opener(request, timeout=30):
        redirect.requests.append((request.full_url, dict(request.header_items())))
        raise urllib.error.HTTPError(request.full_url, code, "Found", {"Location": location}, None)
    return opener


redirect.requests = []


def test_the_signed_address_is_asked_without_following_the_redirect():
    redirect.requests.clear()
    addr = files.signed_address("me/b", "rulz-trainer/ab/my model.bin", "hf_secret_token_value",
                                opener=redirect("https://cas-bridge.xethub.hf.co/x/y?sig=1"))
    assert addr == "https://cas-bridge.xethub.hf.co/x/y?sig=1"
    url, headers = redirect.requests[0]
    assert url == "https://huggingface.co/buckets/me/b/resolve/rulz-trainer/ab/my%20model.bin"
    assert headers["Authorization"] == "Bearer hf_secret_token_value"
    for host in ("https://evil.example.com/x", "http://cdn.hf.co/x", "https://huggingface.co.evil.com/x",
                 "https://nothf.co/x"):
        with pytest.raises(files.FilesError, match="not allowed"):
            files.signed_address("me/b", "p", "tok", opener=redirect(host))
    for good in ("https://huggingface.co/x", "https://a.huggingface.co/x", "https://b.c.hf.co/x"):
        assert files.host_allowed(good)
    with pytest.raises(files.FilesError, match="refused the token"):
        files.signed_address("me/b", "p", "tok", opener=redirect("", 401))

    def plain(request, timeout=30):
        return SimpleNamespace(close=lambda: None)

    with pytest.raises(files.FilesError, match="not an address"):
        files.signed_address("me/b", "p", "tok", opener=plain)


class ScriptedRemote:
    """file-fetch and fetch-state answers from a script; remembers every config sent."""

    def __init__(self, states):
        self.states, self.calls, self.configs = list(states), [], []

    def agent(self, *args, config=None, retry=True, timeout="auto"):
        self.calls.append(args)
        self.configs.append(config)
        state = self.states.pop(0)
        if state == "drop":
            return subprocess_result(255, b"", b"")
        return subprocess_result(0, json.dumps(state).encode(), b"")


def test_an_expired_address_gets_a_fresh_one_and_the_token_never_leaves_this_pc():
    redirect.requests.clear()
    addresses = iter(["https://a.hf.co/1?sig=A", "https://a.hf.co/2?sig=B"])

    def opener(request, timeout=30):
        raise urllib.error.HTTPError(request.full_url, 302, "Found", {"Location": next(addresses)}, None)

    remote = ScriptedRemote([{"state": "running", "done": 10}, {"state": "running", "done": 40},
                             {"state": "expired", "done": 40}, {"state": "running", "done": 50},
                             {"state": "done", "done": 100}])
    seen, sleeps = [], Sleeps()
    out = files.fetch_from_bucket(remote, "me/b", "rulz-trainer/ab/x.bin", "models/sent/0123456789ab/x.bin", 100,
                                  "a" * 64, token="hf_secret_token_value", opener=opener, sleep=sleeps,
                                  progress=lambda d, t: seen.append(d))
    assert out == {"dest": "models/sent/0123456789ab/x.bin", "size": 100, "sha256": "a" * 64}
    starts = [(a, c) for a, c in zip(remote.calls, remote.configs) if a[0] == "file-fetch"]
    assert len(starts) == 2
    assert json.loads(starts[0][1])["url"].endswith("sig=A") and json.loads(starts[1][1])["url"].endswith("sig=B")
    assert not any("hf_secret" in " ".join(a) for a in remote.calls)  # nothing on a command line
    assert not any(c and b"hf_secret" in c for c in remote.configs)  # nor in the program text
    assert seen[-1] == 100 and seen == sorted(seen)


def test_a_failed_download_and_a_dropped_ssh_while_polling():
    def opener(request, timeout=30):
        raise urllib.error.HTTPError(request.full_url, 302, "Found", {"Location": "https://a.hf.co/1?sig=A"}, None)

    remote = ScriptedRemote([{"state": "running", "done": 1}, "drop", {"state": "running", "done": 2},
                             {"state": "failed", "error": "The checksum differs. https://a.hf.co/1?sig=A"}])
    with pytest.raises(files.FilesError) as e:
        files.fetch_from_bucket(remote, "me/b", "p", "models/sent/0123456789ab/x.bin", 100, "b" * 64, token="t",
                                opener=opener, sleep=Sleeps())
    assert e.value.kind == "remote" and "checksum" in str(e.value) and "sig=A" not in str(e.value)


def test_an_address_that_keeps_expiring_ends_it():
    def opener(request, timeout=30):
        raise urllib.error.HTTPError(request.full_url, 302, "Found", {"Location": "https://a.hf.co/1?sig=A"}, None)

    remote = ScriptedRemote([{"state": "expired"}] * 20)
    with pytest.raises(files.FilesError) as e:
        files.fetch_from_bucket(remote, "me/b", "p", "models/sent/0123456789ab/x.bin", 100, "b" * 64, token="t",
                                opener=opener, sleep=Sleeps(), max_refresh=2)
    assert e.value.kind == "expired" and len(remote.calls) == 3


def test_hf_login_never_keeps_or_shows_the_token():
    got = {}

    def login(token, add_to_git_credential):
        got["token"], got["git"] = token, add_to_git_credential

    api = SimpleNamespace(whoami=lambda token=None: {"name": "me"})
    assert files.hf_login("  hf_abc  ", login=login, api=api) == {"ok": True, "user": "me"}
    assert got == {"token": "hf_abc", "git": False}

    def bad(token, add_to_git_credential):
        raise ValueError(f"invalid token {token}")

    with pytest.raises(files.FilesError) as e:
        files.hf_login("hf_secretsecret", login=bad, api=api)
    assert "secretsecret" not in str(e.value)
    with pytest.raises(files.FilesError):
        files.hf_login("   ", login=login, api=api)


def test_hf_token_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", " hf_env ")
    assert files.hf_token() == "hf_env"


# ---------------------------------------------------------------- the downloader on the instance (a loopback server)


class Server:
    """A file server: ranges (or not), a failure to inject, and a log of what was asked."""

    def __init__(self, body: bytes, ranges=True, status=None, redirect_to=None, lie=False):
        self.body, self.ranges, self.status, self.redirect_to, self.lie = body, ranges, status, redirect_to, lie
        self.log, self.live, self.peak, self.lock = [], 0, 0, threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                with outer.lock:
                    outer.live += 1
                    outer.peak = max(outer.peak, outer.live)
                    outer.log.append(self.headers.get("Range"))
                try:
                    if outer.redirect_to:
                        self.send_response(302)
                        self.send_header("Location", outer.redirect_to)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    if outer.status:
                        self.send_response(outer.status)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    rng = self.headers.get("Range")
                    if rng and outer.ranges:
                        a, b = re.fullmatch(r"bytes=(\d+)-(\d+)", rng).groups()
                        a, b = int(a), min(int(b), len(outer.body) - 1)
                        time.sleep(0.05)  # lets the connections overlap
                        self.send_response(206)
                        shown = a + 1 if outer.lie and a > 0 else a  # a range that is not the one asked for
                        self.send_header("Content-Range", f"bytes {shown}-{b}/{len(outer.body)}")
                        self.send_header("Content-Length", str(b - a + 1))
                        self.end_headers()
                        self.wfile.write(outer.body[a:b + 1])
                    else:
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(outer.body)))
                        self.end_headers()
                        self.wfile.write(outer.body)
                finally:
                    with outer.lock:
                        outer.live -= 1

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/f.bin"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fetcher():
    module = agent.embedded(agent.FETCH_PROGRAM, "rulz_fetch_test")
    module.ALLOW_LOOPBACK = True
    module.CHUNK = 64 * 1024
    module.SLEEP = lambda s: None
    return module


def job_for(inst, url, body, name="x.bin", sha256=None):
    dest = f"models/sent/0123456789ab/{name}"
    (inst / "output" / "_fetch").mkdir(parents=True, exist_ok=True)
    return {"url": url, "size": len(body), "sha256": sha256 or sha(body), "dest": dest, "rel": dest,
            "state": str(inst / "output" / "_fetch" / "s.json"), "record": str(agent.sent_record_path(dest))}


def state_of(inst):
    return json.loads((inst / "output" / "_fetch" / "s.json").read_text(encoding="utf-8"))


BODY = bytes((i * 31 + 7) % 256 for i in range(16 * 64 * 1024 + 123))  # 17 chunks of 64 KB (the last one short)


def test_the_downloader_fetches_ranges_over_many_connections_and_verifies(inst, fetcher):
    server = Server(BODY)
    try:
        job = job_for(inst, server.url, BODY)
        assert fetcher.run(job) == "done"
    finally:
        server.close()
    dest = inst / job["dest"]
    assert dest.read_bytes() == BODY and not Path(job["dest"] + ".fpart").exists() and not Path(job["dest"] + ".fmap").exists()
    assert server.peak > 1 and len([r for r in server.log if r and r != "bytes=0-0"]) == 17  # ranges, in parallel
    assert state_of(inst)["state"] == "done" and state_of(inst)["done"] == len(BODY)
    assert agent.sent_complete(job["rel"], len(BODY))  # the record file-state and the PC look for


def test_a_server_that_ignores_ranges_is_read_as_one_stream(inst, fetcher):
    server = Server(BODY, ranges=False)
    try:
        assert fetcher.run(job_for(inst, server.url, BODY)) == "done"
    finally:
        server.close()
    assert (inst / "models/sent/0123456789ab/x.bin").read_bytes() == BODY


def test_a_restart_fetches_only_the_ranges_not_marked_done(inst, fetcher):
    server = Server(BODY)
    job = job_for(inst, server.url, BODY)
    part, mapf = Path(job["dest"] + ".fpart"), Path(job["dest"] + ".fmap")
    part.parent.mkdir(parents=True, exist_ok=True)
    marks = bytearray(b"0" * 17)
    with open(part, "wb") as f:
        f.truncate(len(BODY))
        for i in (0, 1, 2, 3, 16):  # five ranges were whole
            f.seek(i * 65536)
            f.write(BODY[i * 65536:(i + 1) * 65536])
            marks[i] = ord("1")
    mapf.write_bytes(f"RULZ1 {len(BODY)} 65536\n".encode() + bytes(marks))
    try:
        assert fetcher.run(job) == "done"
    finally:
        server.close()
    asked = sorted(int(r.split("=")[1].split("-")[0]) // 65536 for r in server.log if r and r != "bytes=0-0")
    assert asked == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert (inst / job["dest"]).read_bytes() == BODY


def test_a_401_or_403_ends_it_as_expired_and_keeps_what_it_has(inst, fetcher):
    for status in (401, 403):
        server = Server(BODY, status=status)
        try:
            assert fetcher.run(job_for(inst, server.url, BODY)) == "expired"
        finally:
            server.close()
        assert state_of(inst)["state"] == "expired"
    assert not (inst / "models/sent/0123456789ab/x.bin").exists()


def test_a_wrong_checksum_or_size_fails_and_clears_the_part(inst, fetcher):
    server = Server(BODY)
    try:
        job = job_for(inst, server.url, BODY, sha256="0" * 64)
        assert fetcher.run(job) == "failed"
        assert "checksum" in state_of(inst)["error"] and not Path(job["dest"] + ".fpart").exists()
        job = job_for(inst, server.url, BODY[:-5])  # the server's file is longer than the one expected
        assert fetcher.run(job) == "failed" and "bytes" in state_of(inst)["error"]
    finally:
        server.close()


def test_every_redirect_hop_must_be_an_allowed_host(inst, fetcher):
    server = Server(BODY, redirect_to="https://evil.example.com/steal")
    try:
        assert fetcher.run(job_for(inst, server.url, BODY)) == "failed"
    finally:
        server.close()
    err = state_of(inst)["error"]
    assert "not allowed" in err and "evil.example.com" not in err
    fetcher.ALLOW_LOOPBACK = False
    for url in ("https://huggingface.co/x", "https://a.huggingface.co/x", "https://x.xethub.hf.co/x"):
        assert fetcher.host_allowed(url)
    for url in ("http://huggingface.co/x", "https://hf.co/x", "https://huggingface.co.evil.com/x",
                "https://evilhuggingface.co/x", "http://127.0.0.1/x", "ftp://a.hf.co/x", "https://a.hf.co.evil/x"):
        assert not fetcher.host_allowed(url)


def test_file_fetch_refuses_bad_addresses_and_never_writes_the_address_down(inst, monkeypatch):
    remote = FakeRemote(inst)
    dest = "models/sent/0123456789ab/x.bin"
    good = json.dumps({"url": "https://a.hf.co/x?sig=SECRETSIG", "size": 3, "sha256": "a" * 64}).encode()
    spawned = []

    class Proc:
        def __init__(self, cmd, **kw):
            spawned.append((cmd, kw))
            self.stdin = io.BytesIO()
            self.stdin.close = lambda: spawned.append(("closed", bytes(self.stdin.getvalue())))

    monkeypatch.setattr(agent.subprocess, "Popen", Proc)
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    bad = json.dumps({"url": "https://evil.example.com/x", "size": 3, "sha256": "a" * 64}).encode()
    assert run_agent(remote, "file-fetch", dest, config=bad)[0] == 1
    assert run_agent(remote, "file-fetch", dest)[0] == 1  # no config
    code, answer, _ = run_agent(remote, "file-fetch", dest, config=good)
    assert code == 0 and answer["state"] == "starting"
    cmd, kw = spawned[0]
    assert "SECRETSIG" not in " ".join(cmd) and cmd[1] == "-c"  # the address is not on the command line
    assert b"SECRETSIG" in spawned[1][1]  # it went through the pipe
    for f in inst.rglob("*"):
        if f.is_file():
            assert b"SECRETSIG" not in f.read_bytes()  # and is on no file
    # a second call finds the first still going and starts nothing
    monkeypatch.setattr(agent, "pid_running", lambda pid, created=None: True)
    state_file = agent.fetch_paths(dest)[0]
    state = json.loads(state_file.read_text())
    state.update(state="running", pid=4242)
    state_file.write_text(json.dumps(state))
    assert run_agent(remote, "file-fetch", dest, config=good)[1]["state"] == "running" and len([s for s in spawned if isinstance(s[0], list)]) == 1


def test_file_fetch_end_to_end_with_a_real_detached_child(inst, monkeypatch):
    server = Server(BODY)
    program = agent.FETCH_PROGRAM.replace("ALLOW_LOOPBACK = False", "ALLOW_LOOPBACK = True").replace(
        "CHUNK = 16 * 1024 * 1024", "CHUNK = 65536")
    monkeypatch.setattr(agent, "FETCH_PROGRAM", program)
    remote, dest = FakeRemote(inst), "models/sent/0123456789ab/big.bin"
    try:
        config = json.dumps({"url": server.url, "size": len(BODY), "sha256": sha(BODY)}).encode()
        assert run_agent(remote, "file-fetch", dest, config=config)[0] == 0
        deadline = time.time() + 40
        while time.time() < deadline:
            state = run_agent(remote, "fetch-state", dest)[1]
            if state["state"] not in ("starting", "running"):
                break
            time.sleep(0.3)
    finally:
        server.close()
    assert state["state"] == "done", state
    assert (inst / dest).read_bytes() == BODY
    assert run_agent(remote, "file-state", dest, "--size", str(len(BODY)))[1]["done"] is True
    assert run_agent(remote, "file-fetch", dest, config=config)[1]["state"] == "done"  # already there: nothing started


def test_fetch_state_says_interrupted_when_the_child_is_gone(inst):
    dest = "models/sent/0123456789ab/x.bin"
    state_file = agent.fetch_paths(dest)[0]
    agent.write_json(state_file, {"dest": dest, "state": "running", "done": 5, "size": 10, "pid": 999999999,
                                  "updated": time.time()})
    assert run_agent(FakeRemote(inst), "fetch-state", dest)[1]["state"] == "interrupted"
    assert run_agent(FakeRemote(inst), "fetch-state", "models/sent/0123456789ab/none.bin")[1]["state"] == "none"


def test_the_agent_stays_ascii_and_the_new_commands_are_known_to_remote():
    assert all(ord(c) < 128 for c in (REPO / "scripts" / "vast" / "rulz_agent.py").read_text(encoding="utf-8"))
    assert all(ord(c) < 128 for c in agent.FETCH_PROGRAM + agent.WATCH_PROGRAM)
    new = {"file-state", "file-done", "file-fetch", "fetch-state", "watchdog"}
    assert new <= rr.AGENT_COMMANDS and {"file-state", "file-fetch", "fetch-state", "watchdog"} <= rr.SHORT_COMMANDS
    assert "file-done" not in rr.SHORT_COMMANDS  # it hashes the file: no 120 s limit


def test_a_range_that_is_not_the_one_asked_for_is_a_failed_piece_and_keeps_what_is_done(inst, fetcher):
    server = Server(BODY, lie=True)
    try:
        job = job_for(inst, server.url, BODY)
        assert fetcher.run(job) == "failed"
    finally:
        server.close()
    assert "another range" in state_of(inst)["error"]
    assert Path(job["dest"] + ".fmap").exists() and not (inst / job["dest"]).exists()
    marks = Path(job["dest"] + ".fmap").read_bytes().split(b"\n", 1)[1]
    assert marks[:1] == b"1"  # the first range (start 0, told truly) was kept


def test_an_empty_file_gets_its_empty_part_before_file_done(inst, local):
    path = local / "empty.bin"
    path.write_bytes(b"")
    remote = FakeRemote(inst)
    out = files.upload_file(remote, path)
    dest = f"models/sent/{sha(b'')[:12]}/empty.bin"
    assert out["dest"] == dest and out["size"] == 0 and out["skipped"] is False
    assert (inst / dest).is_file() and (inst / dest).stat().st_size == 0
    assert files.upload_file(remote, path)["skipped"] is True
