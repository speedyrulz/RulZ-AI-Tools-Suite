"""Tests for the file-system layer, the host check, the transfer runner and the ssh stream. No network, no token,
no instance: the Hub and ssh are replaced by small stand-ins."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulz_hf_manager import config, hostcheck, hub, platform_utils, remote_agent, vast  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, '_settings', config.Settings())
    monkeypatch.setattr(config, 'settings_path', lambda: tmp_path / 'settings.json')


# ---------------------------------------------------------------- hub: moves, paths, links

def entry(path: str) -> hub.Entry:
    return hub.Entry(path, False, size=1)


def test_move_keeps_a_source_that_is_also_a_destination():
    # Flattening x/x into x: x/x/a goes to x/a, and x/x/x/a lands on x/x/a, which must not be deleted afterwards.
    plan = [(entry('x/x/a'), 'x/a'), (entry('x/x/x/a'), 'x/x/a'), (entry('x/x/b'), 'x/b')]
    assert hub.sources_to_delete(plan) == ['x/x/x/a', 'x/x/b']
    assert hub.sources_to_delete([(entry('a'), 'b')]) == ['a']


def test_repo_move_commit_has_no_delete_for_a_destination(monkeypatch):
    fs = hub.RepoFS('model', 'u/r')
    seen = {}
    monkeypatch.setattr(hub.RepoFS, '_commit', lambda self, ops, message, create_pr: seen.setdefault('ops', ops))
    fs.move([(entry('x/x/a'), 'x/a'), (entry('x/x/x/a'), 'x/x/a')], 'm')
    deletes = [op.path_in_repo for op in seen['ops'] if isinstance(op, hub.CommitOperationDelete)]
    copies = [op.path_in_repo for op in seen['ops'] if isinstance(op, hub.CommitOperationCopy)]
    assert copies == ['x/a', 'x/x/a'] and deletes == ['x/x/x/a']


def test_local_target_stays_inside(tmp_path):
    assert hub.local_target(tmp_path, 'a/b.txt') == tmp_path / 'a' / 'b.txt'
    for bad in ('../x', 'a/../../x', '', 'a//b', './a'):
        with pytest.raises(ValueError):
            hub.local_target(tmp_path, bad)
    if os.name == 'nt':
        for bad in ('C:x', 'a\\..\\..\\x'):
            with pytest.raises(ValueError):
                hub.local_target(tmp_path, bad)


@pytest.mark.parametrize('text, kind, rev', [
    ('https://huggingface.co/a/b/tree/dev/sub', 'model', 'dev'),
    ('https://huggingface.co/a/b/blob/main/x.json', 'model', None),
    ('https://huggingface.co/datasets/a/b/resolve/refs%2Fpr%2F3/x.json', 'dataset', 'refs/pr/3'),
    ('hf://datasets/a/b@v2/images', 'dataset', 'v2'),
    ('hf://a/b@refs%2Fconvert%2Fparquet', 'model', 'refs/convert/parquet'),
    ('a/b', 'model', None),
    ('https://huggingface.co/buckets/a/b/tree/x', 'bucket', None),
])
def test_parse_revision(text, kind, rev):
    assert hub.parse_revision(text, kind) == rev


def test_hf_uri_round_trips_through_parse_revision():
    uri = hub.hf_uri('model', 'a/b', 'sub', 'refs/pr/1')
    assert hub.parse_revision(uri, 'model') == 'refs/pr/1'
    assert hub.parse_location(uri)[1:] == ('a/b', 'sub')


def test_upload_walk_skips_hf_cache_but_not_other_cache(tmp_path):
    root = tmp_path / 'model'
    (root / '.cache' / 'huggingface' / 'download').mkdir(parents=True)
    (root / '.cache' / 'huggingface' / 'download' / 'x.lock').write_text('x')
    (root / '.cache' / 'other').mkdir(parents=True)
    (root / '.cache' / 'other' / 'keep.txt').write_text('x')
    (root / '.git').mkdir()
    (root / '.git' / 'HEAD').write_text('x')
    (root / 'w.safetensors').write_text('x')
    pairs = hub.local_items_to_uploads([root], 'dest')
    assert sorted(d for _, d in pairs) == ['dest/model/.cache/other/keep.txt', 'dest/model/w.safetensors']


def test_zip_method_stores_weights_and_big_files(tmp_path):
    small = tmp_path / 'config.json'
    small.write_text('{}')
    weights = tmp_path / 'model.safetensors'
    weights.write_bytes(b'0' * 10)
    assert hub.zip_method(small) == zipfile.ZIP_DEFLATED
    assert hub.zip_method(weights) == zipfile.ZIP_STORED


def test_sweep_stale_temp_only_takes_old_app_folders(tmp_path):
    old, new, other = tmp_path / 'rulz-hf-dl-old', tmp_path / 'rulz-hf-up-new', tmp_path / 'someone-else'
    for p in (old, new, other):
        p.mkdir()
        (p / 'f').write_text('x')
    long_ago = time.time() - 4 * 86400
    for p in (old, old / 'f', other, other / 'f'):
        os.utime(p, (long_ago, long_ago))
    assert hub.sweep_stale_temp(72, tmp_path) == 1
    assert not old.exists() and new.exists() and other.exists()


# ---------------------------------------------------------------- host check (DNS rebinding)

@pytest.mark.parametrize('header, ok', [
    ('localhost:8765', True), ('127.0.0.1:8765', True), ('[::1]:8765', True), ('LOCALHOST', True),
    ('app.localhost:8765', True), ('127.1.2.3', True), ('localhost.:8765', True),
    ('evil.example.com:8765', False), ('', False), ('192.168.1.5:8765', False), ('localhost.evil.com', False),
])
def test_host_header_check(header, ok):
    assert hostcheck.allowed(header) is ok


def test_host_check_extra_names_and_bind_detection(monkeypatch):
    assert hostcheck.allowed('myalias:8765', frozenset({'myalias'}))
    assert hostcheck.is_loopback_bind('127.0.0.1') and hostcheck.is_loopback_bind('localhost')
    assert hostcheck.is_loopback_bind('::1') and not hostcheck.is_loopback_bind('0.0.0.0')
    monkeypatch.setenv('RULZ_HF_ALLOWED_HOSTS', 'a.example, B.example')
    assert hostcheck.extra_hosts_from_env() == frozenset({'a.example', 'b.example'})
    monkeypatch.setenv('RULZ_HF_ALLOWED_HOSTS', '*')
    assert hostcheck.extra_hosts_from_env() is None


def _call(app, scope_type: str, host: str) -> list[dict]:
    sent: list[dict] = []
    reached: list[bool] = []

    async def inner(scope, receive, send):
        reached.append(True)

    async def receive():
        return {'type': 'websocket.connect'} if scope_type == 'websocket' else {'type': 'http.request'}

    async def send(message):
        sent.append(message)

    scope = {'type': scope_type, 'headers': [(b'host', host.encode())]}
    asyncio.run(hostcheck.LocalHostOnly(inner)(scope, receive, send))
    return [{'reached': True}] if reached else sent


def test_host_check_middleware_blocks_http_and_websocket():
    assert _call(None, 'http', 'localhost:1') == [{'reached': True}]
    blocked = _call(None, 'http', 'evil.example:1')
    assert blocked[0]['status'] == 403
    assert _call(None, 'websocket', 'evil.example:1') == [{'type': 'websocket.close', 'code': 1008}]
    assert _call(None, 'websocket', '127.0.0.1:1') == [{'reached': True}]


# ---------------------------------------------------------------- terminal output

def test_terminal_decoder_joins_split_characters_and_escapes():
    d = vast.TerminalDecoder()
    raw = 'ä 100%\x1b[32m ok\x1b[0m'.encode('utf-8')
    out = ''.join(d.feed(raw[i:i + 1]) for i in range(len(raw))) + d.flush()
    assert out == 'ä 100% ok'
    d = vast.TerminalDecoder()
    assert d.feed(b'abc\x1b[3') == 'abc'
    assert d.feed(b'2mgreen') == 'green'


# ---------------------------------------------------------------- following a transfer

def test_stream_raises_when_the_helper_answers_nothing(monkeypatch):
    monkeypatch.setattr(vast, 'ssh_command', lambda conn, batch=True: [
        sys.executable, '-c', 'import sys; sys.stdin.buffer.read(); sys.stderr.write("no python here"); sys.exit(3)'])
    rem = vast.Remote(vast.Conn(host='h'))
    with pytest.raises(vast.VastError, match='no python here'):
        list(rem.stream('watch', id='x'))


def test_stream_yields_lines(monkeypatch):
    monkeypatch.setattr(vast, 'ssh_command', lambda conn, batch=True: [
        sys.executable, '-c', 'import sys; sys.stdin.buffer.read(); print("noise"); print(\'{"state": "done"}\')'])
    assert list(vast.Remote(vast.Conn(host='h')).stream('watch', id='x')) == [{'state': 'done'}]


def test_follow_backs_off_when_a_watch_brings_nothing(monkeypatch):
    calls = {'n': 0}
    sleeps: list[float] = []

    def stream(self, cmd, **args):
        calls['n'] += 1
        if calls['n'] < 3:
            return iter(())  # connected, but no status: used to reconnect at once, forever
        return iter([{'state': 'done', 'files_done': 1}])

    monkeypatch.setattr(vast.Remote, 'stream', stream)
    monkeypatch.setattr(vast.time, 'sleep', sleeps.append)
    st = vast.follow('job', conn=vast.Conn(host='h'))
    assert st['state'] == 'done' and calls['n'] == 3 and sleeps == [5, 10]


def test_follow_gives_up_on_a_helper_error(monkeypatch):
    monkeypatch.setattr(vast.Remote, 'stream', lambda self, cmd, **a: iter([{'error': 'bad job id'}]))
    with pytest.raises(vast.VastError, match='bad job id'):
        vast.follow('job', conn=vast.Conn(host='h'))


# ---------------------------------------------------------------- the runner on the instance

def make_runner(tmp_path, items, mode='keep') -> remote_agent.Runner:
    folder = tmp_path / 'job'
    folder.mkdir(exist_ok=True)
    (folder / 'job.json').write_text(json.dumps({'id': 'j1', 'dest': str(tmp_path / 'dest'), 'mode': mode,
                                                 'items': items}))
    return remote_agent.Runner(str(folder))


class FakeHub:
    """hf_hub_download into local_dir, writing the repo's name into the file so a mix-up shows."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def hf_hub_download(self, repo, path, repo_type=None, revision=None, local_dir=None, cache_dir=None):
        self.calls.append((repo, path, local_dir))
        target = Path(local_dir, *path.split('/'))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(repo)
        return str(target)


def test_runner_stages_each_repo_apart(tmp_path):
    items = [{'kind': 'model', 'repo': 'alice/m', 'rev': None, 'path': 'config.json', 'size': 7,
              'dest': 'alice/config.json'},
             {'kind': 'model', 'repo': 'bob/m', 'rev': 'refs/pr/1', 'path': 'config.json', 'size': 5,
              'dest': 'bob/config.json'}]
    r = make_runner(tmp_path, items)
    hf = FakeHub()
    for it in items:
        r.fetch(hf, None, it)
    staging_dirs = {c[2] for c in hf.calls}
    assert len(staging_dirs) == 2 and all(d.startswith(r.staging) for d in staging_dirs)
    assert (tmp_path / 'dest' / 'alice' / 'config.json').read_text() == 'alice/m'
    assert (tmp_path / 'dest' / 'bob' / 'config.json').read_text() == 'bob/m'


def test_runner_skips_only_the_same_size(tmp_path):
    item = {'kind': 'model', 'repo': 'a/m', 'rev': None, 'path': 'empty.txt', 'size': 0, 'dest': 'empty.txt'}
    r = make_runner(tmp_path, [item])
    final = tmp_path / 'dest' / 'empty.txt'
    final.parent.mkdir(parents=True)
    final.write_text('an old, much bigger file')
    hf = FakeHub()
    r.fetch(hf, None, item)
    assert hf.calls and r.skipped == 0  # size 0 in the repo, not 0 here: fetched again
    final.write_text('')
    r.fetch(hf, None, item)
    assert len(hf.calls) == 1 and r.skipped == 1  # same size now: skipped
    with pytest.raises(ValueError):
        r.fetch(hf, None, {**item, 'path': '../escape.txt', 'size': 3})


def test_write_json_uses_a_temp_file_of_its_own(tmp_path, monkeypatch):
    seen = []
    real_replace = os.replace
    monkeypatch.setattr(remote_agent.os, 'replace', lambda a, b: (seen.append(a), real_replace(a, b)))
    remote_agent.write_json(str(tmp_path / 'status.json'), {'a': 1})
    assert json.loads((tmp_path / 'status.json').read_text()) == {'a': 1}
    assert str(os.getpid()) in seen[0] and seen[0] != str(tmp_path / 'status.json') + '.tmp'
    assert [p.name for p in tmp_path.iterdir()] == ['status.json']


def test_retry_keeps_the_status_a_fast_runner_wrote(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_agent, 'JOBS', str(tmp_path))
    folder = tmp_path / 'j1'
    folder.mkdir()
    (folder / 'job.json').write_text(json.dumps({'id': 'j1', 'dest': str(tmp_path), 'items': []}))
    (folder / 'status.json').write_text(json.dumps({'id': 'j1', 'state': 'error', 'finished': 1}))

    def spawn(folder_, token, source):  # the runner starts and writes its own status before spawn returns
        remote_agent.write_json(os.path.join(folder_, 'status.json'), {'id': 'j1', 'state': 'running',
                                                                        'pid': os.getpid()})
        return 999999

    monkeypatch.setattr(remote_agent, 'spawn', spawn)
    st = remote_agent.cmd_retry({'id': 'j1', 'token': '', 'source': ''})
    assert st['state'] == 'running' and st['pid'] == os.getpid()


# ---------------------------------------------------------------- platform

@pytest.mark.skipif(not platform_utils.IS_WINDOWS, reason='Windows drive letters')
def test_windows_drive_letters_has_the_system_drive():
    letters = platform_utils.windows_drive_letters()
    assert os.environ.get('SystemDrive', 'C:')[0].upper() in letters


def test_quick_locations_has_home():
    assert any(label == 'Home' for label, _icon, _p in platform_utils.quick_locations())


# ---------------------------------------------------------------- hub calls, with a stand-in HfApi

class FakeApi:
    endpoint = 'https://huggingface.co'
    token = None
    library_name = library_version = None

    def __init__(self, **overrides) -> None:
        self.calls: list[tuple] = []
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return types.SimpleNamespace(sha='abc123', pr_revision=None, private=False, likes=0, last_modified=None,
                                         used_storage=1, downloads=2, gated=False)
        return record


def test_space_summary_does_not_ask_for_gated(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(hub, 'api', lambda: api)
    hub.repo_summary('space', 'a/s')
    hub.repo_summary('model', 'a/m')
    (_, _, space_kw), (_, _, model_kw) = api.calls
    assert 'gated' not in space_kw['expand'] and 'downloads' not in space_kw['expand']
    assert {'gated', 'downloads'} <= set(model_kw['expand'])


def test_copy_folder_to_bucket_merges_contents(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(hub, 'api', lambda: api)
    fs = hub.RepoFS('model', 'a/m')
    fs.copy_to([hub.Entry('loras', True), hub.Entry('x.bin', False, size=1)], 'bucket', 'a/b', 'backup')
    pairs = [(c[1][0], c[1][1]) for c in api.calls if c[0] == 'copy_files']
    assert pairs == [('hf://a/m/loras/', 'hf://buckets/a/b/backup/loras/'),
                     ('hf://a/m/x.bin', 'hf://buckets/a/b/backup/')]


def test_download_stops_when_the_hub_cannot_be_reached(monkeypatch, tmp_path):
    def repo_info(*a, **k):
        raise ConnectionError('offline')
    api = FakeApi(repo_info=repo_info)
    monkeypatch.setattr(hub, 'api', lambda: api)
    (tmp_path / 'old.txt').write_text('from an earlier download')
    with pytest.raises(ConnectionError):
        hub.RepoFS('model', 'a/m').download([hub.Entry('x.json', False, size=3)], tmp_path)
    assert not api.calls  # snapshot_download (which would have returned the old folder) never ran


def test_download_one_file_skips_the_repo_listing(monkeypatch, tmp_path):
    api = FakeApi()
    monkeypatch.setattr(hub, 'api', lambda: api)
    hub.RepoFS('dataset', 'a/d').download([hub.Entry('README.md', False, size=3)], tmp_path)
    assert [c[0] for c in api.calls] == ['repo_info', 'hf_hub_download']
    assert api.calls[1][2]['revision'] == 'abc123'
    hub.RepoFS('dataset', 'a/d').download([hub.Entry('data', True)], tmp_path)
    assert api.calls[-1][0] == 'snapshot_download' and api.calls[-1][2]['allow_patterns'] == ['data/*']


def test_mirror_push_deletes_last_in_its_own_commit(monkeypatch, tmp_path):
    api = FakeApi()
    monkeypatch.setattr(hub, 'api', lambda: api)
    local = tmp_path / 'local'
    (local / 'sub').mkdir(parents=True)
    (local / 'keep.bin').write_text('x')
    (local / 'sub' / 'new.bin').write_text('x')
    remote = ['dst/keep.bin', 'dst/gone.bin', 'dst/sub/old.bin', 'dst/.gitattributes']
    monkeypatch.setattr(hub.RepoFS, 'list_files', lambda self, folder='': [hub.Entry(p, False) for p in remote])
    commits = []
    monkeypatch.setattr(hub.RepoFS, '_commit', lambda self, ops, message, create_pr: commits.append(ops))
    hub.RepoFS('model', 'a/m').upload_folder(local, 'dst', 'msg', mirror=True)
    assert api.calls[0][0] == 'upload_folder' and 'delete_patterns' not in api.calls[0][2]
    # dst/.gitattributes is not the repo's root one, so it goes like any other file gone locally
    assert [op.path_in_repo for op in commits[0]] == ['dst/gone.bin', 'dst/sub/old.bin', 'dst/.gitattributes']


def test_root_gitattributes_survives_a_mirror(monkeypatch, tmp_path):
    monkeypatch.setattr(hub, 'api', lambda: FakeApi())
    (tmp_path / 'a.txt').write_text('x')
    monkeypatch.setattr(hub.RepoFS, 'list_files', lambda self, folder='': [hub.Entry('.gitattributes', False),
                                                                           hub.Entry('b.txt', False)])
    commits = []
    monkeypatch.setattr(hub.RepoFS, '_commit', lambda self, ops, message, create_pr: commits.append(ops))
    hub.RepoFS('model', 'a/m').upload_folder(tmp_path, '', 'msg', mirror=True)
    assert [op.path_in_repo for op in commits[0]] == ['b.txt']


# ---------------------------------------------------------------- editor round trip, settings file, dialogs

def test_editor_keeps_encoding_and_line_endings():
    from rulz_hf_manager.ui import browser
    data = 'a;b\r\ncafé\r\n'.encode('latin-1')
    text, enc, nl = browser.decode_text(data)
    assert (text, enc, nl) == ('a;b\ncafé\n', 'latin-1', '\r\n')
    assert browser.encode_text(text, enc, nl) == data
    assert browser.encode_text(text + 'x\n', enc, nl) == data + b'x\r\n'
    with pytest.raises(UnicodeEncodeError):
        browser.encode_text('€', 'latin-1', '\n')
    assert browser.decode_text('ü\n'.encode()) == ('ü\n', 'utf-8', '\n')


def test_broken_settings_file_is_kept_aside(monkeypatch, tmp_path):
    path = tmp_path / 'settings.json'
    monkeypatch.setattr(config, 'settings_path', lambda: path)
    for bad in ('{not json', '["a list"]'):
        path.write_text(bad)
        assert config._load() == config.Settings()
        assert not path.exists() and any(p.name.startswith('settings.json.broken-') for p in tmp_path.iterdir())
        for p in tmp_path.iterdir():
            p.unlink()
    path.write_text(json.dumps({'token': 'hf_x', 'port': 'oops', 'unknown': 1}))
    s = config._load()
    assert s.token == 'hf_x' and s.port == 8765


def test_settings_save_is_private_on_posix(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'settings_path', lambda: tmp_path / 'settings.json')
    config.get().token = 'hf_secret'
    config.save()
    assert json.loads((tmp_path / 'settings.json').read_text())['token'] == 'hf_secret'
    if not platform_utils.IS_WINDOWS:
        assert (tmp_path / 'settings.json').stat().st_mode & 0o077 == 0


def test_dialog_follow_ups_await_async_callbacks():
    from rulz_hf_manager.ui import repo_actions
    ran = []

    async def reload(*_):
        ran.append('async')

    asyncio.run(repo_actions._done(reload, 'x'))
    asyncio.run(repo_actions._done(lambda *_: ran.append('sync')))
    asyncio.run(repo_actions._done(None))
    assert ran == ['async', 'sync']


def test_mirror_plans_deletions_before_the_upload(monkeypatch, tmp_path):
    order = []
    api = FakeApi(upload_folder=lambda **kw: order.append('upload') or types.SimpleNamespace(pr_revision=None))
    monkeypatch.setattr(hub, 'api', lambda: api)
    (tmp_path / 'a.txt').write_text('x')

    def list_files(self, folder=''):
        order.append(('list', self.revision))
        return [hub.Entry('b.txt', False)]

    monkeypatch.setattr(hub.RepoFS, 'list_files', list_files)
    monkeypatch.setattr(hub.RepoFS, '_commit', lambda self, ops, message, create_pr: order.append(('delete', self.revision)))
    hub.RepoFS('model', 'a/m', 'dev').upload_folder(tmp_path, '', 'msg', mirror=True)
    # a file someone adds while the upload runs is not in the plan, so it is not deleted
    assert order == [('list', 'dev'), 'upload', ('delete', 'dev')]


def test_mirror_into_a_new_folder_deletes_nothing(monkeypatch, tmp_path):
    api = FakeApi()
    monkeypatch.setattr(hub, 'api', lambda: api)
    (tmp_path / 'a.txt').write_text('x')

    def missing(self, folder=''):
        raise hub.hf_errors.EntryNotFoundError('no such folder')

    monkeypatch.setattr(hub.RepoFS, 'list_files', missing)
    commits = []
    monkeypatch.setattr(hub.RepoFS, '_commit', lambda self, ops, message, create_pr: commits.append(ops))
    hub.RepoFS('model', 'a/m').upload_folder(tmp_path, 'new', 'msg', mirror=True)
    assert api.calls[0][0] == 'upload_folder' and not commits


def test_sweep_keeps_a_folder_still_written_into(tmp_path):
    busy = tmp_path / 'rulz-hf-dl-busy'
    (busy / 'files' / 'deep').mkdir(parents=True)
    (busy / 'files' / 'deep' / 'part').write_text('fresh')
    long_ago = time.time() - 10 * 86400
    for p in (busy, busy / 'files', busy / 'files' / 'deep'):
        os.utime(p, (long_ago, long_ago))
    assert hub.sweep_stale_temp(72, tmp_path) == 0 and busy.exists()


def test_staging_tells_similar_revisions_apart(tmp_path):
    r = make_runner(tmp_path, [])
    base = {'kind': 'model', 'repo': 'a/m', 'path': 'x', 'dest': 'x'}
    names = {r.staging_for({**base, 'rev': rev}) for rev in ('a/b', 'a_b', None)}
    assert len(names) == 3


def test_start_leaves_the_status_to_the_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_agent, 'JOBS', str(tmp_path))

    def spawn(folder_, token, source):
        remote_agent.write_json(os.path.join(folder_, 'status.json'), {'id': 'j9', 'state': 'done',
                                                                        'pid': os.getpid()})
        return 999999

    monkeypatch.setattr(remote_agent, 'spawn', spawn)
    st = remote_agent.cmd_start({'job': {'id': 'j9', 'dest': str(tmp_path / 'd'), 'items': []}, 'token': '',
                                 'source': ''})
    assert st['state'] == 'done'


def test_mirror_with_pr_never_deletes_on_the_branch(monkeypatch, tmp_path):
    # Nothing changed in the upload, so huggingface_hub opened no pull request: the deletions must open it.
    api = FakeApi(upload_folder=lambda **kw: types.SimpleNamespace(pr_revision=None))
    monkeypatch.setattr(hub, 'api', lambda: api)
    (tmp_path / 'a.txt').write_text('x')
    monkeypatch.setattr(hub.RepoFS, 'list_files', lambda self, folder='': [hub.Entry('gone.txt', False)])
    seen = []
    monkeypatch.setattr(hub.RepoFS, '_commit', lambda self, ops, message, create_pr: seen.append(create_pr))
    hub.RepoFS('model', 'a/m').upload_folder(tmp_path, '', 'msg', mirror=True, create_pr=True)
    assert seen == [True]


@pytest.mark.parametrize('text, expected', [
    ('hf://datasets/u/d@refs/convert/parquet/default/train/0.parquet',
     ('dataset', 'u/d', 'default/train/0.parquet', 'refs/convert/parquet')),
    ('hf://spaces/u/s@refs/pr/3/app.py', ('space', 'u/s', 'app.py', 'refs/pr/3')),
    ('https://huggingface.co/blob/model', ('model', 'blob/model', '', None)),  # an owner called "blob"
    ('https://huggingface.co/tree/x/tree/dev/sub', ('model', 'tree/x', 'sub', 'dev')),
    ('https://huggingface.co/buckets/u/b/a/c.bin', ('bucket', 'u/b', 'a/c.bin', None)),
    ('u/m', (None, 'u/m', '', None)),
])
def test_link_parsing(text, expected):
    kind, repo_id, path = hub.parse_location(text)
    assert (kind, repo_id, path, hub.parse_revision(text, kind)) == expected


def test_terminal_decoder_does_not_hold_back_short_escapes():
    d = vast.TerminalDecoder()
    assert d.feed(b'progress\x1b7 Do you want to continue? [Y/n] ') == 'progress Do you want to continue? [Y/n] '
    assert vast.TerminalDecoder().feed(b'a\x1b[>4;1mb') == 'ab'
    assert vast.TerminalDecoder().feed(b'title\x1b]0;x') == 'title'  # an OSC title still being written: wait


def test_settings_with_a_bom_load_and_unreadable_files_are_kept(monkeypatch, tmp_path):
    path = tmp_path / 'settings.json'
    monkeypatch.setattr(config, 'settings_path', lambda: path)
    path.write_bytes(b'\xef\xbb\xbf' + json.dumps({'token': 'hf_bom'}).encode())
    assert config._load().token == 'hf_bom' and path.exists()
    monkeypatch.setattr(config.Path, 'read_text', lambda self, encoding=None: (_ for _ in ()).throw(OSError('locked')))
    monkeypatch.setattr(config.time, 'sleep', lambda s: None)
    monkeypatch.setattr(config, '_settings', config._load())
    assert config._unread and path.exists()  # not renamed, not overwritten yet
    monkeypatch.undo()
    monkeypatch.setattr(config, 'settings_path', lambda: path)
    monkeypatch.setattr(config, '_settings', config.Settings())
    config._unread = True
    config.save()
    kept = [p for p in tmp_path.iterdir() if p.name.startswith('settings.json.unread-')]
    assert kept and json.loads(kept[0].read_text(encoding='utf-8-sig'))['token'] == 'hf_bom'


def test_runner_does_not_start_over_a_cancel(tmp_path):
    folder = tmp_path / 'job'
    folder.mkdir()
    (folder / 'job.json').write_text(json.dumps({'id': 'j1', 'dest': str(tmp_path / 'd'), 'items': []}))
    (folder / 'status.json').write_text(json.dumps({'id': 'j1', 'state': 'cancelled'}))
    with pytest.raises(SystemExit):
        remote_agent.Runner(str(folder))
    assert json.loads((folder / 'status.json').read_text())['state'] == 'cancelled'


def test_cancel_finds_a_runner_that_has_not_written_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_agent, 'JOBS', str(tmp_path))
    folder = tmp_path / 'j2'
    folder.mkdir()
    (folder / 'status.json').write_text(json.dumps({'id': 'j2', 'state': 'queued', 'spawned': time.time()}))
    (folder / 'runner.pid').write_text('424242')
    assert remote_agent.runner_pid(str(folder), {}) == 424242
    assert remote_agent.runner_pid(str(folder), {'pid': 7}) == 7


def test_masked_hint_never_shows_the_key():
    from rulz_hf_manager.ui.common import masked_hint
    hint = masked_hint('hf_abcdefghijklmnopqrstuvwxyz1234')
    assert 'hf_' in hint and '1234' in hint and 'abcdefghijkl' not in hint
    assert masked_hint('', 'hf_...') == 'hf_...'
    assert 'short' not in masked_hint('short')


# ---------------------------------------------------------------- file dates

def test_repo_list_is_quick_and_dates_come_separately(monkeypatch):
    from datetime import datetime, timezone
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    day = datetime(2025, 3, 1, tzinfo=timezone.utc)
    pulled = []

    def list_repo_tree(repo_id, path_in_repo=None, *, expand=False, **kw):
        def items():
            for i in range(10):
                pulled.append(i)
                commit = {'id': 'c', 'title': 't', 'date': '2025-03-01T00:00:00.000Z'} if expand else None
                if i == 0:
                    yield RepoFolder(path='sub', oid='o', **({'lastCommit': commit} if commit else {}))
                else:
                    yield RepoFile(path=f'f{i}.txt', size=i, oid='o', **({'lastCommit': commit} if commit else {}))
        api.calls.append(('list_repo_tree', expand))
        return items()

    api = FakeApi(list_repo_tree=list_repo_tree)
    monkeypatch.setattr(hub, 'api', lambda: api)
    fs = hub.RepoFS('model', 'a/m')
    entries = fs.list_dir('')
    assert api.calls[-1] == ('list_repo_tree', False) and all(e.modified is None for e in entries)
    pulled.clear()
    dates = fs.last_commit_dates('', limit=4)
    assert api.calls[-1] == ('list_repo_tree', True)
    assert dates == {'sub': day, 'f1.txt': day, 'f2.txt': day, 'f3.txt': day}
    assert len(pulled) <= 5  # stopped reading pages at the limit
    assert fs.list_folders('') == ['sub'] and api.calls[-1] == ('list_repo_tree', False)


def test_bucket_entries_keep_the_upload_date():
    from datetime import datetime, timezone
    own, up = datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2025, 1, 1, tzinfo=timezone.utc)
    fs = hub.BucketFS('bucket', 'a/b')
    item = types.SimpleNamespace(type='file', path='x.bin', size=1, mtime=own, uploaded_at=up, xet_hash='h')
    e = fs._entry(item)
    assert (e.modified, e.uploaded) == (own, up)
    item.mtime = None
    assert fs._entry(item).modified == up  # no date of its own: the upload


def test_repo_summary_has_the_creation_date(monkeypatch):
    from datetime import datetime, timezone
    made = datetime(2022, 3, 2, tzinfo=timezone.utc)

    def model_info(repo_id, expand=None):
        assert 'createdAt' in expand
        return types.SimpleNamespace(private=False, used_storage=1, likes=0, downloads=0, last_modified=None,
                                     created_at=made, sha='s', gated=False)
    monkeypatch.setattr(hub, 'api', lambda: FakeApi(model_info=model_info))
    assert hub.repo_summary('model', 'a/m')['created'] == made


def test_local_created_time(tmp_path):
    from rulz_hf_manager.ui.pickers import _scan, created_time
    f = tmp_path / 'a.txt'
    f.write_text('x')
    st = f.stat()
    rows = _scan(tmp_path, False, True)
    if os.name == 'nt':
        assert created_time(st) and rows[0]['created']
    elif not hasattr(st, 'st_birthtime'):
        assert created_time(st) is None and rows[0]['created'] == ''  # Linux keeps no creation date
    assert rows[0]['modified']


def test_instance_listing_has_modification_times(tmp_path):
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'm.safetensors').write_bytes(b'12')
    r = remote_agent.cmd_ls({'path': str(tmp_path)})
    assert r['dirs'][0]['mtime'] > 0 and r['files'][0]['mtime'] > 0 and r['files'][0]['size'] == 2


def test_repo_dates_stop_when_nobody_waits(monkeypatch):
    from huggingface_hub.hf_api import RepoFile
    pulled = []

    def list_repo_tree(*a, **k):
        for i in range(100):
            pulled.append(i)
            yield RepoFile(path=f'f{i}', size=1, oid='o', lastCommit={'id': 'c', 'title': 't',
                                                                     'date': '2025-03-01T00:00:00.000Z'})
    monkeypatch.setattr(hub, 'api', lambda: FakeApi(list_repo_tree=list_repo_tree))
    dates = hub.RepoFS('model', 'a/m').last_commit_dates('', should_stop=lambda: len(pulled) >= 3)
    assert len(dates) == 2 and len(pulled) == 3
