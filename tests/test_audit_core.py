"""Tests for the core audit fixes: bucket downloads, settings backups, plan clashes, unlimited jobs."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulz_hf_manager import config, hub, jobs, vast  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(config, '_settings', config.Settings())
    monkeypatch.setattr(config, '_unread', False)
    monkeypatch.setattr(config, 'settings_path', lambda: tmp_path / 'settings.json')


# ---------------------------------------------------------------- B1

def test_bucket_download_raises_on_missing_files(monkeypatch, tmp_path):
    seen = {}
    fs = hub.BucketFS('bucket', 'a/b')
    fake = types.SimpleNamespace(download_bucket_files=lambda repo, files, **kw: seen.update(kw))
    monkeypatch.setattr(hub, 'api', lambda: fake)
    fs.download([hub.Entry('x.bin', False, size=1)], tmp_path / 'out')
    assert seen.get('raise_on_missing_files') is True


# ---------------------------------------------------------------- B2

def test_damaged_file_that_cannot_move_is_backed_up_before_save(monkeypatch, tmp_path):
    path = tmp_path / 'settings.json'
    path.write_bytes(b'{"token": "hf_secret", broken')
    real_replace = Path.replace

    def locked(self, target):
        if self == path:
            raise OSError('held open')
        return real_replace(self, target)

    monkeypatch.setattr(Path, 'replace', locked)
    s = config._load()
    assert config._unread and path.exists()
    monkeypatch.setattr(config, '_settings', s)
    config.save()
    backups = list(tmp_path.glob('settings.json.unread-*'))
    assert len(backups) == 1 and backups[0].read_bytes() == b'{"token": "hf_secret", broken'
    assert '"token"' in path.read_text() and not config._unread


def test_save_fails_without_overwriting_when_no_backup_is_possible(monkeypatch, tmp_path):
    path = tmp_path / 'settings.json'
    original = b'{"token": "hf_secret"}'
    path.write_bytes(original)
    monkeypatch.setattr(config, '_unread', True)
    monkeypatch.setattr(config, '_settings', config.Settings())
    monkeypatch.setattr(Path, 'replace', lambda self, target: (_ for _ in ()).throw(OSError('locked')))
    monkeypatch.setattr(config.shutil, 'copy2', lambda a, b: (_ for _ in ()).throw(OSError('no space')))
    with pytest.raises(OSError):
        config.save()
    assert path.read_bytes() == original and config._unread


def test_normal_save_and_load_still_work(tmp_path):
    config.get().token = 'hf_x'
    config.save()
    assert not list(tmp_path.glob('*.unread-*'))
    assert config._load().token == 'hf_x'


# ---------------------------------------------------------------- B3

def _item(kind='model', repo='a/m', rev=None, path='f.bin', dest='f.bin'):
    return vast.PlanItem(kind, repo, rev, path, 1, dest)


def test_collisions_see_revision_and_kind():
    assert vast.Plan([_item(rev='dev'), _item(rev=None)]).collisions() == ['f.bin']
    assert vast.Plan([_item(kind='dataset'), _item()]).collisions() == ['f.bin']
    assert vast.Plan([_item(rev='main'), _item(rev=None)]).collisions() == []  # None is main


def test_build_plan_drops_exact_duplicates():
    whole = vast.Source('model', 'a/m')
    folder = vast.Source('model', 'a/m', paths=['sub'], dirs=['sub'])
    listed = {0: [('sub/x.bin', 3, 'sub/x.bin'), ('y.bin', 1, 'y.bin')], 1: [('sub/x.bin', 3, 'sub/x.bin')]}
    plan = vast.build_plan([whole, folder], 'keep', listed=listed)
    assert sorted(i.path for i in plan.items) == ['sub/x.bin', 'y.bin']
    assert plan.collisions() == []


# ---------------------------------------------------------------- B5

def test_unlimited_job_does_not_take_a_slot(monkeypatch):
    async def io_bound(fn, *a):
        return await asyncio.to_thread(fn, *a)

    tasks = []
    monkeypatch.setattr(jobs.run, 'io_bound', io_bound)
    monkeypatch.setattr(jobs.background_tasks, 'create', lambda coro, name=None: tasks.append(asyncio.ensure_future(coro)))

    async def scenario():
        import threading
        m = jobs.JobManager(max_concurrent=1)
        gate, ran = threading.Event(), threading.Event()
        long = m.submit('follow', lambda j: gate.wait(10), limited=False)
        short = m.submit('copy', lambda j: ran.set())
        for _ in range(100):
            await asyncio.sleep(0.02)
            if short.status == 'done':
                break
        assert long.status == 'running' and short.status == 'done' and ran.is_set()
        gate.set()
        await asyncio.gather(*tasks)
        assert long.status == 'done'

    asyncio.run(scenario())


def test_unlimited_jobs_stay_out_of_the_shared_thread_pool(monkeypatch):
    # Review finding: followed transfers last hours; in NiceGUI's shared io_bound pool a few of them would leave no
    # worker for the rest of the app. Here the "shared pool" has one worker and five follows are running.
    import concurrent.futures
    import threading
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def io_bound(fn, *a):
        return await asyncio.get_running_loop().run_in_executor(pool, fn, *a)

    tasks = []
    monkeypatch.setattr(jobs.run, 'io_bound', io_bound)
    monkeypatch.setattr(jobs.background_tasks, 'create', lambda coro, name=None: tasks.append(asyncio.ensure_future(coro)))

    async def scenario():
        m = jobs.JobManager(max_concurrent=3)
        gate = threading.Event()
        follows = [m.submit(f'follow {n}', lambda j: gate.wait(10), limited=False) for n in range(5)]

        def boom(j):
            raise RuntimeError('lost the instance')
        failing = m.submit('follow that fails', boom, limited=False)
        short = m.submit('copy', lambda j: 'copied')
        for _ in range(100):
            await asyncio.sleep(0.02)
            if short.status == 'done' and failing.status == 'error':
                break
        assert short.status == 'done' and short.result == 'copied'
        assert failing.status == 'error' and 'lost the instance' in failing.error
        assert all(f.status == 'running' for f in follows)
        gate.set()
        await asyncio.gather(*tasks)
        assert all(f.status == 'done' for f in follows)

    try:
        asyncio.run(scenario())
    finally:
        pool.shutdown(wait=False)


def test_failed_browser_download_leaves_no_temp_folder(monkeypatch, tmp_path):
    # Review finding: download_to_temp's folder was only cleaned when the download came back.
    monkeypatch.setattr(hub.tempfile, 'tempdir', str(tmp_path))
    fs = hub.BucketFS('bucket', 'a/b')

    def gone(*a, **k):
        raise RuntimeError('1 file(s) not found')
    monkeypatch.setattr(fs, 'download', gone)
    with pytest.raises(RuntimeError):
        fs.download_to_temp([hub.Entry('x.bin', False, size=1)])
    assert not list(tmp_path.glob('rulz-hf-dl-*'))
