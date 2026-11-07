"""Friendly layer over ``huggingface_hub`` for repos (model / dataset / space) and storage buckets.

Everything in here is blocking I/O and is meant to be called from a worker thread
(``nicegui.run.io_bound``) or from a background job (see ``jobs.py``).
"""

from __future__ import annotations

import glob
import os
import posixpath
import re
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlparse

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi, get_token
from huggingface_hub import errors as hf_errors
from huggingface_hub.hf_api import RepoFolder
from huggingface_hub.utils import tqdm as hf_tqdm

from . import __version__, config

if TYPE_CHECKING:
    from .jobs import Job

KINDS = ('model', 'dataset', 'space', 'bucket')
KIND_LABEL = {'model': 'Model', 'dataset': 'Dataset', 'space': 'Space', 'bucket': 'Bucket'}
KIND_ICON = {'model': 'view_in_ar', 'dataset': 'dataset', 'space': 'rocket_launch', 'bucket': 'inventory_2'}
SPACE_SDKS = ['gradio', 'docker', 'static']
_URI_PREFIX = {'model': '', 'dataset': 'datasets/', 'space': 'spaces/', 'bucket': 'buckets/'}
_URL_PREFIX_TO_KIND = {'datasets': 'dataset', 'spaces': 'space', 'buckets': 'bucket', 'models': 'model'}

# Files we never upload when walking a local folder.
_SKIP_DIRS = {'.git', '__pycache__', '.ipynb_checkpoints'}
_SKIP_SUFFIX = ('.cache/huggingface',)


# --------------------------------------------------------------------------------------------------
# API client & identity
# --------------------------------------------------------------------------------------------------

_api_lock = threading.Lock()
_api_cache: tuple[str | None, HfApi] | None = None


def configured_token() -> str | None:
    return config.get().token.strip() or None


def api() -> HfApi:
    """Return an HfApi bound to the token from settings (falls back to HF_TOKEN / `hf auth login`)."""
    global _api_cache
    token = configured_token()
    with _api_lock:
        if _api_cache is None or _api_cache[0] != token:
            _api_cache = (token, HfApi(token=token, library_name='rulz-hf-manager', library_version=__version__))
        return _api_cache[1]


def token_source() -> str:
    """Where the active token comes from: 'app', 'env', 'cli' or 'none'."""
    if configured_token():
        return 'app'
    if os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'):
        return 'env'
    try:
        return 'cli' if get_token() else 'none'
    except Exception:
        return 'none'


def external_token() -> str | None:
    """Token from HF_TOKEN or the HF CLI login, if any (used for the 'import' button)."""
    try:
        return get_token()
    except Exception:
        return None


@dataclass
class Identity:
    name: str
    fullname: str = ''
    avatar: str = ''
    orgs: list[str] = field(default_factory=list)
    token_name: str = ''
    token_role: str = ''  # 'read' | 'write' | 'fineGrained' ('' for OAuth logins)
    auth_type: str = ''  # 'access_token' | 'oauth' | ...
    expires: datetime | None = None  # OAuth logins (e.g. from "hf auth login" in a browser) expire

    @property
    def is_oauth(self) -> bool:
        return self.auth_type == 'oauth'

    @property
    def namespaces(self) -> list[str]:
        return [self.name, *self.orgs]

    @property
    def can_write(self) -> bool:
        return self.token_role != 'read'


_who_cache: dict[str | None, tuple[float, Identity | None]] = {}


def whoami(force: bool = False) -> Identity | None:
    """Return who the token belongs to, or None when no (valid) token is available."""
    token = configured_token()
    cached = _who_cache.get(token)
    if cached and not force and time.time() - cached[0] < 300:
        return cached[1]
    if token_source() == 'none':
        _who_cache[token] = (time.time(), None)
        return None
    info = api().whoami()
    auth = info.get('auth') or {}
    access = auth.get('accessToken') or {}
    expires = None
    if auth.get('expiresAt'):
        try:
            expires = datetime.fromisoformat(str(auth['expiresAt']).replace('Z', '+00:00'))
        except ValueError:
            pass
    ident = Identity(
        name=info.get('name', ''),
        fullname=info.get('fullname', '') or '',
        avatar=info.get('avatarUrl', '') or '',
        orgs=[o.get('name') for o in info.get('orgs', []) if o.get('name')],
        token_name=access.get('displayName', '') or '',
        token_role=access.get('role', '') or '',
        auth_type=auth.get('type', '') or '',
        expires=expires,
    )
    if ident.avatar.startswith('/'):
        ident.avatar = 'https://huggingface.co' + ident.avatar
    _who_cache[token] = (time.time(), ident)
    return ident


def forget_identity() -> None:
    _who_cache.clear()


def friendly_error(e: BaseException) -> str:
    """Turn huggingface_hub exceptions into short, human-readable messages."""
    if isinstance(e, hf_errors.GatedRepoError):
        return 'This repository is gated. Accept its terms on huggingface.co first.'
    if isinstance(e, hf_errors.RepositoryNotFoundError):
        return 'Repository not found, or your token cannot access it.'
    if isinstance(e, hf_errors.BucketNotFoundError):
        return 'Bucket not found, or your token cannot access it.'
    if isinstance(e, hf_errors.RevisionNotFoundError):
        return 'Branch / revision not found.'
    if isinstance(e, (hf_errors.EntryNotFoundError, hf_errors.RemoteEntryNotFoundError)):
        return 'File not found on the Hub.'
    if isinstance(e, hf_errors.LocalTokenNotFoundError):
        return 'No Hugging Face token configured. Add one in Settings.'
    if isinstance(e, hf_errors.BucketBatchError):
        return f'Some bucket operations failed: {e}'
    if isinstance(e, hf_errors.HfHubHTTPError):
        status = getattr(getattr(e, 'response', None), 'status_code', None)
        server_msg = (getattr(e, 'server_message', None) or '').strip()
        if status == 401:
            return 'Your token was rejected (401). Check it in Settings.'
        if status == 403:
            return 'Permission denied (403). Your token may be read-only or lack access to this namespace.'
        if status == 429:
            return 'Rate limited by the Hub (429). Wait a minute and try again.'
        first = (server_msg or str(e)).strip().splitlines()[0] if (server_msg or str(e)).strip() else ''
        return f'Hub error {status or ""}: {first}'.strip()
    if isinstance(e, (ConnectionError, TimeoutError)):
        return f'Network problem: {e}'
    text = str(e).strip() or type(e).__name__
    return text.splitlines()[0] if len(text) < 400 else text[:400] + '...'


# --------------------------------------------------------------------------------------------------
# URLs, URIs, parsing user input
# --------------------------------------------------------------------------------------------------

def endpoint() -> str:
    return api().endpoint.rstrip('/')


def web_url(kind: str, repo_id: str, path: str = '', revision: str | None = None, is_dir: bool = False) -> str:
    base = f'{endpoint()}/{_URI_PREFIX[kind]}{repo_id}'
    if kind == 'bucket' or not path:
        if kind != 'bucket' and revision and revision != 'main':
            return f'{base}/tree/{quote(revision, safe="")}'
        return base
    rev = quote(revision or 'main', safe='')
    return f'{base}/{"tree" if is_dir else "blob"}/{rev}/{quote(path)}'


def hf_uri(kind: str, repo_id: str, path: str = '', revision: str | None = None) -> str:
    rev = f'@{quote(revision, safe="")}' if revision and kind != 'bucket' else ''
    uri = f'hf://{_URI_PREFIX[kind]}{repo_id}{rev}'
    return f'{uri}/{path}' if path else uri


_ID_RE = re.compile(r'^[\w.\-]+/[\w.\-]+$')


def parse_location(text: str) -> tuple[str | None, str, str]:
    """Parse a URL / hf:// URI / id typed by the user into (kind or None, repo_id, path)."""
    text = text.strip().strip('"').strip("'")
    if not text:
        raise ValueError('Nothing to open.')
    path = ''
    if text.startswith('hf://'):
        parts = text[5:].strip('/').split('/')
    elif '://' in text:
        parsed = urlparse(text)
        parts = [p for p in parsed.path.split('/') if p]
    else:
        parts = [p for p in text.split('/') if p]
    kind: str | None = None
    if parts and parts[0] in _URL_PREFIX_TO_KIND:
        kind = _URL_PREFIX_TO_KIND[parts.pop(0)]
    if len(parts) < 2:
        raise ValueError('Expected something like "username/name" or a huggingface.co link.')
    repo_id = f'{parts[0]}/{parts[1].split("@")[0]}'
    rest = parts[2:]
    if rest and rest[0] in {'tree', 'blob', 'resolve'} and len(rest) >= 2:
        path = '/'.join(rest[2:])
    elif kind == 'bucket' and rest:
        path = '/'.join(rest)
    if kind is None and '://' in text:
        kind = 'model'  # huggingface.co/<user>/<name> is a model
    if not _ID_RE.match(repo_id):
        raise ValueError(f'"{repo_id}" does not look like a valid repo id.')
    return kind, repo_id, path


def detect_kind(repo_id: str) -> str:
    """Find out whether an id is a model, dataset, space or bucket (first match wins)."""
    a = api()
    if a.repo_exists(repo_id, repo_type='model'):
        return 'model'
    if a.repo_exists(repo_id, repo_type='dataset'):
        return 'dataset'
    if a.repo_exists(repo_id, repo_type='space'):
        return 'space'
    try:
        a.bucket_info(repo_id)
        return 'bucket'
    except hf_errors.HfHubHTTPError:
        pass
    raise ValueError(f'Could not find a model, dataset, space or bucket named "{repo_id}".')


# --------------------------------------------------------------------------------------------------
# Namespace-level listing & management
# --------------------------------------------------------------------------------------------------

@dataclass
class RepoRow:
    kind: str
    id: str
    visibility: str = 'public'  # public | private | protected
    updated: datetime | None = None
    size: int | None = None
    downloads: int | None = None
    likes: int | None = None
    files: int | None = None


def list_namespace(namespace: str, kinds: tuple[str, ...] = KINDS) -> list[RepoRow]:
    """List repos and buckets in a user/org namespace."""
    a = api()
    ident = None
    try:
        ident = whoami()
    except Exception:
        pass
    # Fast path: the "settings/repositories" endpoint returns everything incl. storage for your own namespaces.
    if ident and (namespace == ident.name or namespace in ident.orgs):
        try:
            rows = [
                RepoRow(kind=r.type, id=r.id, visibility=r.visibility, updated=r.updated_at, size=r.storage)
                for r in a.list_user_repos(namespace=None if namespace == ident.name else namespace)
                if r.type in kinds
            ]
            return rows
        except hf_errors.HfHubHTTPError:
            pass  # e.g. org member without admin rights -> fall back to the public listing APIs

    # Note: the list endpoints do not accept the "usedStorage" expand field (only single-repo info does).
    def models() -> list[RepoRow]:
        return [RepoRow('model', m.id, 'private' if m.private else 'public', m.last_modified, None,
                        m.downloads, m.likes)
                for m in a.list_models(author=namespace, limit=2000,
                                       expand=['private', 'lastModified', 'downloads', 'likes'])]

    def datasets() -> list[RepoRow]:
        return [RepoRow('dataset', d.id, 'private' if d.private else 'public', d.last_modified, None,
                        d.downloads, d.likes)
                for d in a.list_datasets(author=namespace, limit=2000,
                                         expand=['private', 'lastModified', 'downloads', 'likes'])]

    def spaces() -> list[RepoRow]:
        return [RepoRow('space', s.id, 'private' if s.private else 'public', s.last_modified, None, None, s.likes)
                for s in a.list_spaces(author=namespace, limit=2000, expand=['private', 'lastModified', 'likes'])]

    def buckets() -> list[RepoRow]:
        return [RepoRow('bucket', b.id, 'private' if b.private else 'public', b.created_at, b.size,
                        files=b.total_files)
                for b in a.list_buckets(namespace=namespace)]

    rows: list[RepoRow] = []
    errors: list[Exception] = []
    listers = {'model': models, 'dataset': datasets, 'space': spaces, 'bucket': buckets}
    for kind in kinds:
        try:
            rows.extend(listers[kind]())
        except Exception as e:  # noqa: BLE001  (one failing type must not hide the others)
            errors.append(e)
    if errors and not rows:
        raise errors[0]
    return rows


def repo_summary(kind: str, repo_id: str) -> dict[str, Any]:
    a = api()
    if kind == 'bucket':
        b = a.bucket_info(repo_id)
        return {'private': b.private, 'size': b.size, 'files': b.total_files, 'created': b.created_at}
    getter = {'model': a.model_info, 'dataset': a.dataset_info, 'space': a.space_info}[kind]
    expand = ['private', 'lastModified', 'usedStorage', 'likes', 'sha', 'gated']
    if kind != 'space':
        expand.append('downloads')
    info = getter(repo_id, expand=expand)
    return {
        'private': bool(info.private), 'size': getattr(info, 'used_storage', None), 'likes': info.likes,
        'downloads': getattr(info, 'downloads', None), 'updated': info.last_modified, 'sha': info.sha,
        'gated': getattr(info, 'gated', None),
    }


def create_repo(kind: str, repo_id: str, private: bool, space_sdk: str | None = None) -> str:
    a = api()
    if kind == 'bucket':
        return a.create_bucket(repo_id, private=private).url
    url = a.create_repo(repo_id, repo_type=kind, private=private,
                        space_sdk=space_sdk if kind == 'space' else None)
    return str(url)


def delete_repo(kind: str, repo_id: str) -> None:
    if kind == 'bucket':
        api().delete_bucket(repo_id)
    else:
        api().delete_repo(repo_id, repo_type=kind)


def rename_repo(kind: str, from_id: str, to_id: str) -> None:
    if kind == 'bucket':
        api().move_bucket(from_id, to_id)
    else:
        api().move_repo(from_id, to_id, repo_type=kind)


def set_private(kind: str, repo_id: str, private: bool) -> None:
    if kind == 'bucket':
        api().update_bucket_settings(repo_id, private=private)
    else:
        api().update_repo_settings(repo_id, private=private, repo_type=kind)


def duplicate_repo(kind: str, from_id: str, to_id: str, private: bool) -> str:
    return str(api().duplicate_repo(from_id, to_id, repo_type=kind, private=private))


# --------------------------------------------------------------------------------------------------
# File-system abstraction
# --------------------------------------------------------------------------------------------------

@dataclass
class Entry:
    path: str
    is_dir: bool
    size: int | None = None
    modified: datetime | None = None
    lfs: bool = False
    xet_hash: str | None = None
    raw: Any = None

    @property
    def name(self) -> str:
        return self.path.rsplit('/', 1)[-1]

    @property
    def parent(self) -> str:
        return posixpath.dirname(self.path)


def join(folder: str, name: str) -> str:
    folder = folder.strip('/')
    name = name.strip('/')
    return f'{folder}/{name}' if folder else name


def validate_name(name: str) -> str:
    name = name.strip().strip('/')
    if not name:
        raise ValueError('Name cannot be empty.')
    if any(part in {'', '.', '..'} for part in name.split('/')):
        raise ValueError('Name contains an invalid path segment.')
    if '\\' in name:
        raise ValueError('Use "/" to separate folders, not "\\".')
    return name


def local_items_to_uploads(local_paths: list[str | Path], dest_folder: str) -> list[tuple[Path, str]]:
    """Expand local files/folders into (local file, remote path) pairs under dest_folder."""
    pairs: list[tuple[Path, str]] = []
    for raw in local_paths:
        p = Path(raw)
        if p.is_file():
            pairs.append((p, join(dest_folder, p.name)))
        elif p.is_dir():
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
                rel_root = Path(root).relative_to(p.parent).as_posix()
                if rel_root.endswith(_SKIP_SUFFIX) or '/.cache/huggingface' in rel_root:
                    continue
                for f in files:
                    pairs.append((Path(root) / f, join(dest_folder, f'{rel_root}/{f}')))
    return pairs


def job_tqdm(job: Job):
    """A tqdm class that reports progress into a Job instead of the console."""

    class JobTqdm(hf_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._job_desc = str(kwargs.get('desc') or '')
            self._job_unit = kwargs.get('unit', 'it')
            kwargs['disable'] = True
            super().__init__(*args, **kwargs)
            self._job_n = kwargs.get('initial') or 0

        def update(self, n: float | None = 1) -> None:  # type: ignore[override]
            self._job_n += n or 0
            self._job_report()

        def __iter__(self):  # type: ignore[override]
            for item in self.iterable:
                yield item
                self._job_n += 1
                self._job_report()

        def _job_report(self) -> None:
            total = getattr(self, 'total', None)
            if self._job_unit == 'B':
                if self._job_desc.startswith('Downloading'):
                    job.set(detail=f'{human_size(self._job_n)} transferred')
            elif total:
                job.set(progress=min(1.0, self._job_n / total), message=f'{int(self._job_n)}/{int(total)} files')

    return JobTqdm


def human_size(n: float | None) -> str:
    if n is None:
        return ''
    n = float(n)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1024 or unit == 'TB':
            return f'{int(n)} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} PB'


class RemoteFS:
    """Common interface for browsing and editing files in a repo or a bucket."""

    kind: str

    def __init__(self, kind: str, repo_id: str, revision: str | None = None) -> None:
        self.kind = kind
        self.repo_id = repo_id
        self.revision = revision if kind != 'bucket' else None

    @property
    def api(self) -> HfApi:
        return api()

    @property
    def is_bucket(self) -> bool:
        return self.kind == 'bucket'

    def uri(self, path: str = '') -> str:
        return hf_uri(self.kind, self.repo_id, path, self.revision)

    # ---- listing -----------------------------------------------------------------------------
    def list_dir(self, folder: str) -> list[Entry]:
        raise NotImplementedError

    def list_files(self, folder: str = '') -> list[Entry]:
        raise NotImplementedError

    def list_folders(self, folder: str) -> list[str]:
        return sorted(e.path for e in self.list_dir(folder) if e.is_dir)

    def existing(self, paths: list[str]) -> set[str]:
        raise NotImplementedError

    def expand(self, entries: list[Entry]) -> list[Entry]:
        files: list[Entry] = []
        for e in entries:
            files.extend(self.list_files(e.path) if e.is_dir else [e])
        return files

    def plan_moves(self, pairs: list[tuple[Entry, str]]) -> list[tuple[Entry, str]]:
        """Expand (entry, new path) pairs into per-file (file, new file path) pairs."""
        plan: list[tuple[Entry, str]] = []
        for entry, new_path in pairs:
            new_path = new_path.strip('/')
            if entry.is_dir:
                if new_path == entry.path or new_path.startswith(entry.path + '/'):
                    if new_path != entry.path:
                        raise ValueError(f'Cannot move folder "{entry.path}" into itself.')
                    continue
                for f in self.list_files(entry.path):
                    plan.append((f, new_path + f.path[len(entry.path):]))
            elif new_path != entry.path:
                plan.append((entry, new_path))
        dests = [d for _, d in plan]
        if len(dests) != len(set(dests)):
            raise ValueError('Two files would end up at the same destination path.')
        return plan

    # ---- mutations ---------------------------------------------------------------------------
    def upload(self, items: list[tuple[str | Path | bytes, str]], message: str, create_pr: bool = False) -> Any:
        raise NotImplementedError

    def delete(self, entries: list[Entry], message: str, create_pr: bool = False) -> Any:
        raise NotImplementedError

    def move(self, plan: list[tuple[Entry, str]], message: str, create_pr: bool = False) -> Any:
        raise NotImplementedError

    def copy(self, plan: list[tuple[Entry, str]], message: str, create_pr: bool = False) -> Any:
        raise NotImplementedError

    def download(self, entries: list[Entry], local_dir: Path, job: Job | None = None) -> Path:
        raise NotImplementedError

    def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    def write_bytes(self, path: str, data: bytes, message: str, create_pr: bool = False) -> Any:
        return self.upload([(data, path)], message, create_pr)

    def copy_to(self, entries: list[Entry], dest_kind: str, dest_id: str, dest_folder: str,
                dest_revision: str | None = None, message: str = '', job: Job | None = None) -> None:
        """Copy entries (files or folders) into another repo or bucket."""
        dest_folder = dest_folder.strip('/')
        if dest_kind == 'bucket' and self.kind == 'bucket':
            files = self.expand(entries)
            pairs = [(f, join(dest_folder, _rel_to_selection(f, entries))) for f in files]
            self.api.batch_bucket_files(dest_id, copy=[('bucket', self.repo_id, f.xet_hash, d) for f, d in pairs])
        elif dest_kind == 'bucket':
            for i, e in enumerate(entries):
                if job:
                    job.set(progress=i / len(entries), message=e.path)
                # copy_files copies a folder's *contents*, so name the destination folder explicitly
                # (keeps "sel/x.bin" as "<dest>/sel/x.bin", like the other copy paths).
                target = join(dest_folder, e.name) if e.is_dir else dest_folder
                self.api.copy_files(self.uri(e.path), hf_uri('bucket', dest_id, target) + '/')
        elif self.kind == 'bucket':
            # The Hub cannot copy bucket -> repo server-side: download to a temp folder then commit.
            with tempfile.TemporaryDirectory(prefix='rulz-hf-', ignore_cleanup_errors=True) as tmp:
                if job:
                    job.set(message='Downloading from bucket...')
                files = self.expand(entries)
                self.api.download_bucket_files(self.repo_id, files=[(f.raw or f.path, Path(tmp) / f.path) for f in files])
                if job:
                    job.set(message='Committing to repository...')
                ops = [CommitOperationAdd(join(dest_folder, _rel_to_selection(f, entries)), str(Path(tmp) / f.path))
                       for f in files]
                self.api.create_commit(dest_id, ops, repo_type=dest_kind, revision=dest_revision,
                                       commit_message=message or f'Copy {len(files)} file(s) from bucket {self.repo_id}')
        else:
            files = self.expand(entries)
            ops = [CommitOperationCopy(src_path_in_repo=f.path,
                                       path_in_repo=join(dest_folder, _rel_to_selection(f, entries)),
                                       src_revision=self.revision, src_repo_id=self.repo_id, src_repo_type=self.kind)
                   for f in files]
            self.api.create_commit(dest_id, ops, repo_type=dest_kind, revision=dest_revision,
                                   commit_message=message or f'Copy {len(files)} file(s) from {self.repo_id}')

    def download_to_temp(self, entries: list[Entry], job: Job | None = None) -> tuple[Path, str]:
        """Download entries for the browser: a single file as-is, otherwise a .zip. Returns (path, filename)."""
        tmp = Path(tempfile.mkdtemp(prefix='rulz-hf-dl-'))
        self.download(entries, tmp / 'files', job)
        root = tmp / 'files'
        if len(entries) == 1 and not entries[0].is_dir:
            return root / entries[0].path, entries[0].name
        base = entries[0].parent if entries else ''
        name = (entries[0].name if len(entries) == 1 else self.repo_id.split('/')[-1]) + '.zip'
        zip_path = tmp / name
        if job:
            job.set(message='Creating zip...')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for f in root.rglob('*'):
                rel = f.relative_to(root).as_posix()
                if f.is_file() and not rel.startswith('.cache/'):
                    arc = rel[len(base) + 1:] if base and rel.startswith(base + '/') else rel
                    zf.write(f, arc)
        return zip_path, name


def _rel_to_selection(f: Entry, selection: list[Entry]) -> str:
    """Path of a file relative to the parent folder of the selected entry it belongs to."""
    for e in selection:
        if f.path == e.path or (e.is_dir and f.path.startswith(e.path + '/')):
            return f.path[len(e.parent) + 1:] if e.parent else f.path
    return f.name


class RepoFS(RemoteFS):
    """Model / dataset / space repository (git based: every change is a commit)."""

    def _entry(self, item: Any) -> Entry:
        last = getattr(item, 'last_commit', None)
        modified = getattr(last, 'date', None) if last else None
        if isinstance(item, RepoFolder):
            return Entry(item.path, True, modified=modified, raw=item)
        return Entry(item.path, False, size=item.size, modified=modified, lfs=item.lfs is not None,
                     xet_hash=item.xet_hash, raw=item)

    def list_dir(self, folder: str) -> list[Entry]:
        items = self.api.list_repo_tree(self.repo_id, path_in_repo=folder or None, recursive=False,
                                        expand=config.get().load_last_modified,
                                        repo_type=self.kind, revision=self.revision)
        return [self._entry(i) for i in items]

    def list_files(self, folder: str = '') -> list[Entry]:
        items = self.api.list_repo_tree(self.repo_id, path_in_repo=folder or None, recursive=True,
                                        repo_type=self.kind, revision=self.revision)
        return [self._entry(i) for i in items if not isinstance(i, RepoFolder)]

    def existing(self, paths: list[str]) -> set[str]:
        if not paths:
            return set()
        found: set[str] = set()
        for i in range(0, len(paths), 500):
            infos = self.api.get_paths_info(self.repo_id, paths[i:i + 500], repo_type=self.kind, revision=self.revision)
            found.update(x.path for x in infos)
        return found

    def _commit(self, ops: list, message: str, create_pr: bool) -> Any:
        return self.api.create_commit(self.repo_id, ops, commit_message=message, repo_type=self.kind,
                                      revision=self.revision, create_pr=create_pr)

    def upload(self, items, message, create_pr=False):
        ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=src if isinstance(src, bytes) else str(src))
               for src, dest in items]
        return self._commit(ops, message, create_pr)

    def upload_folder(self, local_dir: Path, dest_folder: str, message: str, mirror: bool,
                      create_pr: bool = False) -> Any:
        return self.api.upload_folder(repo_id=self.repo_id, folder_path=local_dir, path_in_repo=dest_folder or None,
                                      commit_message=message, repo_type=self.kind, revision=self.revision,
                                      create_pr=create_pr, delete_patterns=['*'] if mirror else None,
                                      ignore_patterns=['.git/*', '.cache/huggingface/*'])

    def delete(self, entries, message, create_pr=False):
        ops = [CommitOperationDelete(path_in_repo=e.path + '/' if e.is_dir else e.path, is_folder=e.is_dir)
               for e in entries]
        return self._commit(ops, message, create_pr)

    def move(self, plan, message, create_pr=False):
        ops: list = [CommitOperationCopy(src_path_in_repo=f.path, path_in_repo=dest, src_revision=self.revision)
                     for f, dest in plan]
        ops += [CommitOperationDelete(path_in_repo=f.path) for f, _ in plan]
        return self._commit(ops, message, create_pr)

    def copy(self, plan, message, create_pr=False):
        ops = [CommitOperationCopy(src_path_in_repo=f.path, path_in_repo=dest, src_revision=self.revision)
               for f, dest in plan]
        return self._commit(ops, message, create_pr)

    def download(self, entries, local_dir, job=None):
        patterns = [glob.escape(e.path) + ('/*' if e.is_dir else '') for e in entries]
        local_dir.mkdir(parents=True, exist_ok=True)
        self.api.snapshot_download(self.repo_id, repo_type=self.kind, revision=self.revision,
                                   local_dir=local_dir, allow_patterns=patterns,
                                   tqdm_class=job_tqdm(job) if job else None)
        return local_dir

    def read_bytes(self, path):
        with tempfile.TemporaryDirectory(prefix='rulz-hf-', ignore_cleanup_errors=True) as tmp:
            local = self.api.hf_hub_download(self.repo_id, path, repo_type=self.kind, revision=self.revision,
                                             local_dir=tmp)
            return Path(local).read_bytes()

    # ---- git specifics -----------------------------------------------------------------------
    def branches(self) -> list[str]:
        refs = self.api.list_repo_refs(self.repo_id, repo_type=self.kind)
        names = [b.name for b in refs.branches]
        return sorted(names, key=lambda n: (n != 'main', n))

    def create_branch(self, name: str, from_revision: str | None) -> None:
        self.api.create_branch(self.repo_id, branch=name, revision=from_revision, repo_type=self.kind)

    def history(self, limit: int = 200) -> list[dict]:
        commits = self.api.list_repo_commits(self.repo_id, repo_type=self.kind, revision=self.revision)
        return [{'id': c.commit_id, 'short': c.commit_id[:8], 'title': c.title, 'authors': ', '.join(c.authors),
                 'date': c.created_at} for c in commits[:limit]]


class BucketFS(RemoteFS):
    """Storage bucket (S3-like, no git history: changes apply immediately)."""

    def _entry(self, item: Any) -> Entry:
        if item.type == 'file':
            return Entry(item.path, False, size=item.size, modified=item.mtime or item.uploaded_at,
                         xet_hash=item.xet_hash, raw=item)
        return Entry(item.path.rstrip('/'), True, modified=getattr(item, 'uploaded_at', None), raw=item)

    def list_dir(self, folder: str) -> list[Entry]:
        folder = folder.strip('/')
        items = self.api.list_bucket_tree(self.repo_id, prefix=folder or None, recursive=False)
        prefix = folder + '/' if folder else ''
        out: list[Entry] = []
        for item in items:
            e = self._entry(item)
            rest = e.path[len(prefix):] if e.path.startswith(prefix) else None
            if rest and '/' not in rest:  # direct children only
                out.append(e)
        return out

    def list_files(self, folder: str = '') -> list[Entry]:
        folder = folder.strip('/')
        items = self.api.list_bucket_tree(self.repo_id, prefix=folder or None, recursive=True)
        prefix = folder + '/' if folder else ''
        return [e for e in (self._entry(i) for i in items) if not e.is_dir and e.path.startswith(prefix)]

    def existing(self, paths: list[str]) -> set[str]:
        if not paths:
            return set()
        return {x.path for x in self.api.get_bucket_paths_info(self.repo_id, paths)}

    def upload(self, items, message='', create_pr=False):
        add = [(src if isinstance(src, bytes) else str(src), dest) for src, dest in items]
        return self.api.batch_bucket_files(self.repo_id, add=add)

    def delete(self, entries, message='', create_pr=False):
        files = self.expand(entries)
        if files:
            self.api.batch_bucket_files(self.repo_id, delete=[f.path for f in files])
        return len(files)

    def _server_copy(self, plan: list[tuple[Entry, str]]) -> None:
        missing = [f.path for f, _ in plan if not f.xet_hash]
        if missing:
            raise ValueError(f'Missing content hash for {len(missing)} file(s); refresh and try again.')
        self.api.batch_bucket_files(self.repo_id, copy=[('bucket', self.repo_id, f.xet_hash, d) for f, d in plan])

    def move(self, plan, message='', create_pr=False):
        if not plan:
            return None
        self._server_copy(plan)  # copy first; only delete the originals once every copy succeeded
        self.api.batch_bucket_files(self.repo_id, delete=[f.path for f, _ in plan])
        return None

    def copy(self, plan, message='', create_pr=False):
        if plan:
            self._server_copy(plan)

    def download(self, entries, local_dir, job=None):
        files = self.expand(entries)
        local_dir.mkdir(parents=True, exist_ok=True)
        if job:
            job.set(message=f'{len(files)} file(s), {human_size(sum(f.size or 0 for f in files))}')
        self.api.download_bucket_files(self.repo_id, files=[(f.raw or f.path, local_dir / f.path) for f in files])
        return local_dir

    def read_bytes(self, path):
        with tempfile.TemporaryDirectory(prefix='rulz-hf-', ignore_cleanup_errors=True) as tmp:
            dest = Path(tmp) / 'file'
            self.api.download_bucket_files(self.repo_id, files=[(path, dest)], raise_on_missing_files=True)
            return dest.read_bytes()

    def sync(self, local_dir: str, remote_folder: str, direction: str, delete: bool,
             include: list[str] | None, exclude: list[str] | None, plan_file: str | None = None,
             apply_file: str | None = None) -> Any:
        """direction: 'up' (local -> bucket) or 'down' (bucket -> local)."""
        if apply_file:
            return self.api.sync_bucket(apply=apply_file, quiet=True)
        remote = self.uri(remote_folder.strip('/'))
        src, dst = (local_dir, remote) if direction == 'up' else (remote, local_dir)
        return self.api.sync_bucket(src, dst, delete=delete, include=include or None, exclude=exclude or None,
                                    plan=plan_file, quiet=True)


def open_fs(kind: str, repo_id: str, revision: str | None = None) -> RemoteFS:
    return BucketFS(kind, repo_id) if kind == 'bucket' else RepoFS(kind, repo_id, revision)


def cleanup_temp(path: Path) -> None:
    """Remove a temp folder created by download_to_temp (best effort)."""
    try:
        root = path if path.name.startswith('rulz-hf-dl-') else next(
            (p for p in path.parents if p.name.startswith('rulz-hf-dl-')), None)
        if root:
            shutil.rmtree(root, ignore_errors=True)
    except Exception:
        pass
