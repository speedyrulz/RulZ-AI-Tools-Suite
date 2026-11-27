"""Regression tests for the open follow-up items in hub.py and remote_agent.py. No network."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import errors as hf_errors  # noqa: E402

from rulz_hf_manager import hub, remote_agent  # noqa: E402


class FakeApi:
    endpoint = 'https://huggingface.co'
    token = None
    library_name = library_version = None

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return types.SimpleNamespace(sha='abc123')
        return record


def make_runner(tmp_path, items=None) -> remote_agent.Runner:
    folder = tmp_path / 'job'
    folder.mkdir(exist_ok=True)
    (folder / 'job.json').write_text(json.dumps({
        'id': 'j1', 'dest': str(tmp_path / 'dest'), 'mode': 'keep', 'items': items or [],
    }))
    return remote_agent.Runner(str(folder))


def http_error(status: int) -> hf_errors.HfHubHTTPError:
    response = types.SimpleNamespace(status_code=status, headers={}, request=None)
    return hf_errors.HfHubHTTPError('hub said no', response=response)


class StatusError(Exception):
    def __init__(self, status) -> None:
        super().__init__('nope')
        self.response = types.SimpleNamespace(status_code=status)


# ---------------------------------------------------------------- A1 parent commit

def test_write_bytes_commits_against_the_parent_it_was_given(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(hub, 'api', lambda: api)
    sha = 'a' * 40
    hub.RepoFS('model', 'a/m', 'dev').write_bytes('README.md', b'hi', 'edit', False, sha)
    name, args, kw = api.calls[0]
    assert name == 'create_commit'
    assert kw['parent_commit'] == sha
    assert kw['revision'] == 'dev' and kw['commit_message'] == 'edit' and kw['create_pr'] is False
    assert args[1][0].path_in_repo == 'README.md' and args[1][0].path_or_fileobj == b'hi'


def test_write_bytes_without_a_parent_does_not_send_one(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(hub, 'api', lambda: api)
    hub.RepoFS('model', 'a/m').write_bytes('notes.txt', b'x', 'edit')
    assert api.calls[0][0] == 'create_commit' and 'parent_commit' not in api.calls[0][2]


def test_bucket_write_bytes_accepts_and_ignores_a_parent_commit(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(hub, 'api', lambda: api)
    hub.BucketFS('bucket', 'a/b').write_bytes('f.txt', b'hi', 'edit', parent_commit='abc')
    assert api.calls[0][0] == 'batch_bucket_files'
    assert all(c[0] != 'create_commit' for c in api.calls)


def test_revision_sha_is_the_commit_the_revision_points_at(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(hub, 'api', lambda: api)
    assert hub.RepoFS('dataset', 'a/d', 'dev').revision_sha() == 'abc123'
    assert api.calls[0][0] == 'repo_info'
    assert api.calls[0][2]['revision'] == 'dev' and 'sha' in api.calls[0][2]['expand']
    assert hub.BucketFS('bucket', 'a/b').revision_sha() is None
    assert len(api.calls) == 1  # a bucket does not ask the Hub


def test_stale_parent_tells_the_user_to_reopen():
    text = hub.friendly_error(http_error(412))
    assert 'changed on the Hub' in text and 'opened' in text and 'Reopen' in text
    other = hub.friendly_error(http_error(401))
    assert '401' in other and 'changed on the Hub' not in other


# ---------------------------------------------------------------- A2 bucket paths

def _file(path: str) -> hub.Entry:
    return hub.Entry(path, False, size=4, xet_hash='hash', raw=object())


def test_bucket_download_passes_paths_so_a_deleted_file_can_raise(monkeypatch, tmp_path):
    api = FakeApi()
    monkeypatch.setattr(hub, 'api', lambda: api)
    entry = _file('dir/a.bin')
    hub.BucketFS('bucket', 'ns/b').download([entry], tmp_path)
    remote, _dest = api.calls[0][2]['files'][0]
    assert remote == 'dir/a.bin' and isinstance(remote, str)
    assert api.calls[0][2]['raise_on_missing_files'] is True


def test_bucket_copy_to_a_repo_passes_paths(monkeypatch):
    api = FakeApi()

    def download_bucket_files(*args, **kwargs):
        api.calls.append(('download_bucket_files', args, kwargs))
        for _remote, dest in kwargs['files']:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b'x')

    api.download_bucket_files = download_bucket_files
    monkeypatch.setattr(hub, 'api', lambda: api)
    entry = _file('dir/a.bin')
    hub.BucketFS('bucket', 'ns/b').copy_to([entry], 'model', 'a/m', 'dst')
    download = next(c for c in api.calls if c[0] == 'download_bucket_files')
    remote, _dest = download[2]['files'][0]
    assert remote == 'dir/a.bin' and isinstance(remote, str)
    assert download[2]['raise_on_missing_files'] is True


# ---------------------------------------------------------------- A3 pip timeout, A4 marker

def test_install_hub_stops_a_stuck_pip(monkeypatch, tmp_path):
    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get('timeout') or 0, output=b'still resolving\n')

    monkeypatch.setattr(remote_agent.subprocess, 'run', run)
    runner = make_runner(tmp_path)
    with pytest.raises(RuntimeError, match='Could not install huggingface_hub on the instance') as raised:
        runner.install_hub(str(tmp_path / 'pylib'), str(tmp_path / '.rulz-ok'))
    assert 'timed out' in str(raised.value) and 'still resolving' in str(raised.value)
    assert not (tmp_path / '.rulz-ok').exists()


def test_install_hub_gives_pip_fifteen_minutes_and_records_the_requirement(monkeypatch, tmp_path):
    seen = []
    installs = []

    def run(cmd, **kwargs):
        seen.append(kwargs.get('timeout'))
        if 'ensurepip' in cmd:
            return types.SimpleNamespace(returncode=0, stdout=b'')
        installs.append(cmd)
        if len(installs) == 1:
            return types.SimpleNamespace(returncode=1, stdout=b'no pip\n')
        return types.SimpleNamespace(returncode=0, stdout=b'')

    monkeypatch.setattr(remote_agent.subprocess, 'run', run)
    marker = tmp_path / '.rulz-ok'
    make_runner(tmp_path).install_hub(str(tmp_path / 'pylib'), str(marker))
    assert seen == [15 * 60, 15 * 60, 15 * 60]
    assert marker.read_text(encoding='utf-8') == remote_agent.HUB_REQUIREMENT


def _call_hub(tmp_path, monkeypatch, marker_text):
    monkeypatch.setattr(remote_agent, 'STATE', str(tmp_path))
    pylib = tmp_path / ('pylib-py%d%d' % sys.version_info[:2])
    pylib.mkdir()
    if marker_text is not None:
        (pylib / '.rulz-ok').write_text(marker_text, encoding='utf-8')
    calls = []

    def install(self, pylib_path, marker):
        calls.append(marker)
        Path(marker).write_text(remote_agent.HUB_REQUIREMENT, encoding='utf-8')

    monkeypatch.setattr(remote_agent.Runner, 'install_hub', install)
    saved = list(sys.path)
    try:
        make_runner(tmp_path).hub()
    finally:
        sys.path[:] = saved
    return calls


def test_hub_reinstalls_when_the_marker_has_a_different_requirement(tmp_path, monkeypatch):
    assert _call_hub(tmp_path, monkeypatch, 'huggingface_hub[hf_xet]>=1,<2')


def test_hub_reinstalls_when_an_old_marker_is_empty(tmp_path, monkeypatch):
    assert _call_hub(tmp_path, monkeypatch, '')


def test_hub_keeps_an_install_that_matches_the_requirement(tmp_path, monkeypatch):
    assert _call_hub(tmp_path, monkeypatch, remote_agent.HUB_REQUIREMENT) == []


# ---------------------------------------------------------------- A5 auth is fatal

def _failures(tmp_path, monkeypatch, status, name=None):
    monkeypatch.setattr(remote_agent.time, 'sleep', lambda *_a, **_k: None)
    runner = make_runner(tmp_path)
    tries = []

    def fetch(_hf, _api, _item):
        tries.append(1)
        if name:
            raise type(name, (Exception,), {})('missing')
        raise StatusError(status)

    runner.fetch = fetch
    runner.one(None, None, {'repo': 'a/m', 'path': 'f.bin', 'size': 1})
    return tries, runner.failed


def test_unauthorized_and_forbidden_are_not_retried(tmp_path, monkeypatch):
    tries, failed = _failures(tmp_path, monkeypatch, 401)
    assert len(tries) == 1 and failed
    tries, failed = _failures(tmp_path, monkeypatch, 403)
    assert len(tries) == 1 and failed


def test_a_plain_server_error_is_still_retried(tmp_path, monkeypatch):
    tries, failed = _failures(tmp_path, monkeypatch, 500)
    assert len(tries) == 4 and failed


def test_a_named_fatal_error_is_still_not_retried(tmp_path, monkeypatch):
    tries, failed = _failures(tmp_path, monkeypatch, None, name='RepositoryNotFoundError')
    assert len(tries) == 1 and failed
