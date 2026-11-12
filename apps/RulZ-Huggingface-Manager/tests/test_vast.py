"""Tests for the vast.ai transfer planning and the instance-side helper. No network, no instance, no token."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulz_hf_manager import config, remote_agent, vast  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, '_settings', config.Settings())
    monkeypatch.setattr(config, 'settings_path', lambda: tmp_path / 'settings.json')


# ---------------------------------------------------------------- ssh command parsing

@pytest.mark.parametrize('text, host, port, user', [
    ('ssh -p 22094 root@ssh5.vast.ai -L 8080:localhost:8080', 'ssh5.vast.ai', 22094, 'root'),
    ('ssh -p 44568 root@159.48.242.43 -L 8080:localhost:8080', '159.48.242.43', 44568, 'root'),
    ('root@1.2.3.4:41022', '1.2.3.4', 41022, 'root'),
    ('1.2.3.4 41022', '1.2.3.4', 41022, 'root'),
    ('ssh -i ~/.ssh/id_rsa -p 2222 ubuntu@box.example.com', 'box.example.com', 2222, 'ubuntu'),
    ('ssh root@host.example.org', 'host.example.org', 22, 'root'),
])
def test_parse_ssh_command(text, host, port, user):
    conn = vast.parse_ssh_command(text)
    assert (conn.host, conn.port, conn.user) == (host, port, user)


def test_parse_ssh_command_rejects_nonsense():
    with pytest.raises(ValueError):
        vast.parse_ssh_command('hello')
    with pytest.raises(ValueError):
        vast.parse_ssh_command('')


def test_routes_direct_first():
    raw = {'public_ipaddr': ' 1.2.3.4 ', 'ports': {'22/tcp': [{'HostPort': '40000'}]},
           'ssh_host': 'ssh5.vast.ai', 'ssh_port': 22094}
    assert vast.routes(raw) == [{'route': 'direct', 'host': '1.2.3.4', 'port': 40000},
                                {'route': 'proxy', 'host': 'ssh5.vast.ai', 'port': 22094}]
    assert vast.routes({}) == []


# ---------------------------------------------------------------- planning

FILES = [('model_index.json', 10, 'model_index.json'),
         ('transformer/config.json', 20, 'transformer/config.json'),
         ('transformer/diffusion_pytorch_model.safetensors', 1000, 'transformer/diffusion_pytorch_model.safetensors'),
         ('vae/config.json', 30, 'vae/config.json'),
         ('vae/diffusion_pytorch_model.safetensors', 500, 'vae/diffusion_pytorch_model.safetensors')]


def plan(layout, sources=None, listed=None, **kw):
    sources = sources or [vast.Source('model', 'someone/pipe')]
    return vast.build_plan(sources, layout, listed=listed or {0: FILES}, **kw)


def test_keep_layout_keeps_folders():
    p = plan('keep')
    assert [i.dest for i in p.items] == [f[0] for f in FILES]
    assert p.total == 1560
    assert not p.collisions()


def test_repo_layout_adds_repo_folder():
    assert plan('repo').items[1].dest == 'pipe/transformer/config.json'


def test_flat_layout_detects_name_clashes():
    p = plan('flat')
    assert p.collisions() == ['config.json', 'diffusion_pytorch_model.safetensors']
    with pytest.raises(vast.VastError, match='same name'):
        vast.start_transfer(p, '/workspace', 'flat', 't')


def test_filters():
    p = plan('keep', include=['*.safetensors'])
    assert [i.path for i in p.items] == [FILES[2][0], FILES[4][0]]
    assert p.skipped_by_filter == 3
    p = plan('keep', exclude=['vae/*'])
    assert all(not i.path.startswith('vae/') for i in p.items)
    assert vast.split_patterns('*.bin, *.pt ;onnx/*') == ['*.bin', '*.pt', 'onnx/*']


def test_cache_layout_refuses_buckets():
    with pytest.raises(ValueError, match='bucket'):
        plan('cache', sources=[vast.Source('bucket', 'me/b')])


def test_relative_to_selection():
    src = vast.Source('model', 'a/b', None, ['transformer', 'loras/x/style.safetensors'], ['transformer'])
    assert vast._relative('transformer/config.json', src) == 'transformer/config.json'
    assert vast._relative('loras/x/style.safetensors', src) == 'style.safetensors'
    nested = vast.Source('model', 'a/b', None, ['split/transformer'], ['split/transformer'])
    assert vast._relative('split/transformer/a/b.json', nested) == 'transformer/a/b.json'


def test_source_labels():
    assert vast.Source('model', 'a/b').label == 'a/b (whole model)'
    assert vast.Source('model', 'a/b', None, ['x'], ['x']).label == 'a/b/x/'
    assert vast.Source('dataset', 'a/b', None, ['x', 'y']).label == 'a/b (2 items)'


def test_source_from_links_without_network():
    s = vast.source_from_text('https://huggingface.co/a/b/blob/main/sub/model.safetensors')
    assert (s.kind, s.repo_id, s.revision, s.paths, s.dirs) == ('model', 'a/b', None, ['sub/model.safetensors'], [])
    s = vast.source_from_text('https://huggingface.co/datasets/a/b/tree/v2/images')
    assert (s.kind, s.revision, s.paths, s.dirs) == ('dataset', 'v2', ['images'], ['images'])
    s = vast.source_from_text('hf://datasets/a/b')
    assert s.whole and s.kind == 'dataset'


def test_nothing_to_send():
    with pytest.raises(vast.VastError, match='Nothing'):
        vast.start_transfer(vast.Plan([]), '/workspace', 'keep', 't')


def test_attached_round_trip():
    assert vast.attached() is None
    conn = vast.Conn(host='1.2.3.4', port=40000, route='direct', instance_id=7, label='x')
    vast.save_attached(conn)
    assert vast.attached() == conn
    vast.save_attached(None)
    assert vast.attached() is None


# ---------------------------------------------------------------- the instance-side helper, run here

def run_agent(payload: dict) -> dict:
    done = subprocess.run([sys.executable, '-'], input=vast.agent_program(payload), capture_output=True, timeout=60)
    assert done.returncode == 0, done.stderr.decode()
    return json.loads(done.stdout.decode().strip().splitlines()[-1])


def test_agent_program_carries_payload():
    program = vast.agent_program({'cmd': 'df', 'path': '/x'}).decode()
    encoded = program.split("PAYLOAD = '", 1)[1].split("'", 1)[0]
    assert json.loads(base64.b64decode(encoded)) == {'cmd': 'df', 'path': '/x'}
    assert vast.runner_source()  # the runner's own copy goes along on start


def test_agent_ls_and_df(tmp_path):
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'file.bin').write_bytes(b'12345')
    (tmp_path / '.hidden').write_text('x')
    r = run_agent({'cmd': 'ls', 'path': str(tmp_path)})
    assert [d['name'] for d in r['dirs']] == ['sub']
    assert r['files'] == [{'name': 'file.bin', 'size': 5}]
    assert r['free'] > 0
    r = run_agent({'cmd': 'ls', 'path': str(tmp_path), 'hidden': True})
    assert {f['name'] for f in r['files']} == {'file.bin', '.hidden'}
    r = run_agent({'cmd': 'ls', 'path': str(tmp_path / 'missing')})
    assert r['exists'] is False
    r = run_agent({'cmd': 'df', 'path': str(tmp_path / 'not' / 'yet')})
    assert r['exists'] is False and r['free'] > 0
    assert 'error' in run_agent({'cmd': 'nope'})


def test_agent_refuses_bad_job_ids():
    for bad in ('', '../x', '.hidden', 'a/b'):
        with pytest.raises(ValueError):
            remote_agent.job_dir(bad)


def test_runner_target_stays_inside_destination(tmp_path):
    folder = tmp_path / 'job'
    folder.mkdir()
    (folder / 'job.json').write_text(json.dumps({'id': 'j1', 'dest': str(tmp_path / 'dest'), 'items': []}))
    r = remote_agent.Runner(str(folder))
    assert r.target({'dest': 'a/b.json'}) == str(tmp_path / 'dest' / 'a' / 'b.json')
    for bad in ('../escape', 'a/../../b', '', 'a//b'):
        with pytest.raises(ValueError):
            r.target({'dest': bad})
