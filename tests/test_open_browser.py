"""Browser page helpers behind the open-item fixes: double clicks, upload tally, JSON off the loop, parent sha."""

import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulz_hf_manager import hub
from rulz_hf_manager.ui import browser


def test_once_guard_ignores_second_start_until_release():
    g = browser.OnceGuard()
    assert g.acquire('a', now=0)
    assert not g.acquire('a', now=1)  # still loading
    assert g.acquire('b', now=1)  # another file is fine
    g.release('a', hold=0.5, now=2)
    assert not g.acquire('a', now=2.2)  # the second click of a double click, just after the first finished
    assert g.acquire('a', now=3)


def test_once_guard_hold_expires_without_release():
    g = browser.OnceGuard()
    assert g.acquire('dl', hold=1.5, now=0)
    assert not g.acquire('dl', hold=1.5, now=1.0)
    assert g.acquire('dl', hold=1.5, now=1.6)


def test_upload_tally_does_not_count_failed_or_aborted_requests():
    t = browser.UploadTally()
    t.uploading()
    t.uploading()
    t.uploading()
    assert t.pending == 3
    t.settled()  # one uploaded
    t.settled()  # one failed
    assert t.pending == 1
    t.finished()  # the third was aborted (a removed file raises no 'failed'): nothing is open any more
    assert t.pending == 0


def test_upload_tally_counts_files_still_being_saved():
    t = browser.UploadTally()
    t.uploading()
    t.settled()
    t.saving += 1
    assert t.pending == 1
    t.finished()  # 'finish' must not forget a save that is still running
    assert t.pending == 1
    t.saving -= 1
    assert t.pending == 0


def test_upload_tally_never_goes_negative():
    t = browser.UploadTally()
    t.settled()
    assert t.pending == 0


def test_pretty_json_indents_and_keeps_bad_json():
    assert browser._pretty_json('{"a":[1,2],"b":"é"}') == json.dumps({'a': [1, 2], 'b': 'é'}, indent=2,
                                                                      ensure_ascii=False)
    assert browser._pretty_json('{"a": 1') == '{"a": 1'


class _FakeFS:
    def __init__(self, kind='model', revision='dev'):
        self.kind, self.repo_id, self.revision = kind, 'u/r', revision
        self.calls = []
        self.sha = lambda: 'abc123'

    def revision_sha(self):
        self.calls.append(('sha',))
        return self.sha()

    def read_bytes(self, path):
        self.calls.append(('read', path))
        return b'data'


def test_read_for_preview_returns_the_repo_sha(monkeypatch):
    fs = _FakeFS()
    assert browser._read_for_preview(fs, 'a.txt') == (b'data', 'abc123')
    assert fs.calls == [('sha',), ('read', 'a.txt')]  # the sha first, then the content


def test_read_for_preview_without_a_sha_still_reads(monkeypatch):
    def boom():
        raise RuntimeError('offline')
    fs = _FakeFS()
    fs.sha = boom
    assert browser._read_for_preview(fs, 'a.txt') == (b'data', None)


def test_read_for_preview_skips_the_sha_for_buckets(monkeypatch):
    def boom():
        raise AssertionError('buckets have no commits')
    fs = _FakeFS(kind='bucket', revision=None)
    fs.sha = boom
    assert browser._read_for_preview(fs, 'a.txt') == (b'data', None)


def test_upload_tally_drops_a_file_removed_while_uploading():
    # Removing a file mid-upload aborts its request; QUploader sends 'removed' (not 'failed') and,
    # while another upload still runs, no 'finish' either.
    t = browser.UploadTally()
    t.uploading()
    t.uploading()
    t.removed(1)  # what the js_handler emits: how many removed files were still uploading
    assert t.pending == 1
    t.removed([0])  # NiceGUI may hand the emitted value over as a one-item list
    t.removed(0)
    assert t.pending == 1
    t.settled()
    assert t.pending == 0
