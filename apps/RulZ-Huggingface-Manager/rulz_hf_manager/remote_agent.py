"""The instance's side of "Send to vast.ai". Standard library only, Python 3.8+.

RulZ Huggingface Manager pipes this file into ``python -`` over ssh on every call, with the call's arguments written
into ``PAYLOAD`` below (base64 JSON), so nothing has to be installed on the instance beforehand and the instance always
runs the same version as the app. Arguments never go on the command line: the Hugging Face token travels in this
program's text on stdin, and the background download gets it through its environment only.

Calls (``cmd`` in the payload): info, ls, mkdir, df, start, jobs, status, watch, cancel, retry, forget.

A transfer runs as a detached background process (``--run <job dir>``) that survives the ssh connection: closing the
app, or losing the network, does not stop it. Its state lives in ``~/.rulz-hf/jobs/<id>/status.json``.

Downloads use the instance's own internet connection, straight from Hugging Face: nothing passes through the user's
computer. ``huggingface_hub`` is installed once into ``~/.rulz-hf/pylib-<python version>`` (it never touches the
instance's own Python environments).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

PAYLOAD = ''

STATE = os.path.join(os.path.expanduser('~'), '.rulz-hf')
JOBS = os.path.join(STATE, 'jobs')
HUB_REQUIREMENT = 'huggingface_hub[hf_xet]>=2.4,<3'
FINAL = ('done', 'error', 'cancelled', 'stopped')
COMFY_DIRS = ('checkpoints', 'diffusion_models', 'unet', 'loras', 'text_encoders', 'clip', 'vae', 'clip_vision',
              'controlnet', 'upscale_models', 'embeddings', 'model_patches', 'style_models', 'audio_encoders')


def out(obj) -> None:
    sys.stdout.write(json.dumps(obj) + '\n')
    sys.stdout.flush()


def expand(path: str) -> str:
    path = os.path.expanduser((path or '').strip() or '~')
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(os.path.expanduser('~'), path))


def nearest_existing(path: str) -> str:
    while path and not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path or '/'


def disk(path: str) -> dict:
    try:
        u = shutil.disk_usage(nearest_existing(expand(path)))
        return {'free': u.free, 'total': u.total}
    except OSError:
        return {'free': None, 'total': None}


def read_json(path: str, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path: str, data) -> None:
    # A temp name of its own per process and thread: the runner, start/retry and cancel can all write status.json
    # at once, and a shared temp file would let one writer move the other's half-written file into place.
    tmp = '%s.%d.%d.tmp' % (path, os.getpid(), threading.get_ident())
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def pid_alive(pid) -> bool:
    if not pid:
        return False
    if os.name == 'nt':
        # The agent runs on Linux instances; this is only the test suite on Windows, where os.kill(pid, 0) does not
        # test anything: signal 0 is CTRL_C_EVENT, which presses Ctrl+C in the whole console.
        return True
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    try:  # a finished child that nobody reaped is a zombie: it still "exists" but is not running
        with open('/proc/%d/stat' % int(pid)) as f:
            return f.read().split(')')[-1].split()[0] != 'Z'
    except OSError:
        return True


# ------------------------------------------------------------------------------------------------ info & files

def hf_home_candidates() -> list:
    found = []
    for path in (os.environ.get('HF_HOME'), '/workspace/.hf_home', '/workspace/.cache/huggingface',
                 os.path.expanduser('~/.cache/huggingface')):
        if path and path not in found and (os.path.isdir(path) or not found):
            found.append(path)
    return found


def comfy_models_dirs() -> list:
    found = []
    for root in ('/workspace/ComfyUI', '/opt/ComfyUI', '/ComfyUI', os.path.expanduser('~/ComfyUI'),
                 '/workspace/comfyui', '/app/ComfyUI'):
        models = os.path.join(root, 'models')
        if os.path.isdir(models) and models not in found:
            found.append(os.path.realpath(models) if os.path.islink(models) else models)
    return found


def suggestions() -> list:
    """Folders worth one click in the app's destination box, the most likely first."""
    out_ = []

    def add(label: str, path: str, kind: str = 'folder') -> None:
        if path and all(s['path'] != path for s in out_):
            out_.append({'label': label, 'path': path, 'kind': kind})

    for models in comfy_models_dirs():
        for name in COMFY_DIRS:
            if os.path.isdir(os.path.join(models, name)):
                add('ComfyUI ' + name, os.path.join(models, name), 'comfy')
        add('ComfyUI models', models, 'comfy')
    for root in ('/workspace/RulzTrainer', '/workspace/rulztrainer'):
        if os.path.isdir(root):
            add('RulZ Trainer models', os.path.join(root, 'models'), 'trainer')
    if os.path.isdir('/workspace'):
        add('/workspace', '/workspace')
        add('/workspace/models', '/workspace/models')
    add('Home', os.path.expanduser('~'))
    return out_


def gpu_names() -> list:
    try:
        done = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
        return [line.strip() for line in done.stdout.decode(errors='replace').splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def cmd_info(p: dict) -> dict:
    jobs = list_jobs()
    return {
        'hostname': os.uname().nodename if hasattr(os, 'uname') else '',
        'home': os.path.expanduser('~'),
        'python': sys.executable, 'python_version': '%d.%d.%d' % sys.version_info[:3],
        'gpus': gpu_names(),
        'workspace': disk('/workspace') if os.path.isdir('/workspace') else None,
        'home_disk': disk('~'),
        'hf_home': hf_home_candidates(),
        'suggestions': suggestions(),
        'running': sum(1 for j in jobs if j.get('state') not in FINAL),
    }


def cmd_ls(p: dict) -> dict:
    path = expand(p.get('path') or ('/workspace' if os.path.isdir('/workspace') else '~'))
    if not os.path.isdir(path):
        return {'error': 'No folder %s on the instance.' % path, 'path': path, 'parent': os.path.dirname(path),
                'exists': False, **disk(path)}
    dirs, files = [], []
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_dir():
                        dirs.append({'name': e.name})
                    elif e.is_file():
                        files.append({'name': e.name, 'size': e.stat().st_size})
                except OSError:
                    continue
    except OSError as e:
        return {'error': str(e), 'path': path, 'parent': os.path.dirname(path), 'exists': True}
    hidden = bool(p.get('hidden'))
    dirs = sorted((d for d in dirs if hidden or not d['name'].startswith('.')), key=lambda d: d['name'].lower())
    files = sorted((f for f in files if hidden or not f['name'].startswith('.')), key=lambda f: f['name'].lower())
    return {'path': path, 'parent': os.path.dirname(path) if path != '/' else None, 'exists': True,
            'dirs': dirs[:3000], 'files': files[:3000], 'more': len(dirs) > 3000 or len(files) > 3000, **disk(path)}


def cmd_mkdir(p: dict) -> dict:
    path = expand(p['path'])
    os.makedirs(path, exist_ok=True)
    return {'path': path}


def cmd_df(p: dict) -> dict:
    path = expand(p.get('path') or '~')
    return {'path': path, 'exists': os.path.isdir(path), **disk(path)}


# ------------------------------------------------------------------------------------------------ jobs

def job_dir(job_id: str) -> str:
    if not job_id or '/' in job_id or job_id.startswith('.'):
        raise ValueError('bad job id')
    return os.path.join(JOBS, job_id)


def job_status(job_id: str) -> dict:
    folder = job_dir(job_id)
    st = read_json(os.path.join(folder, 'status.json'), None)
    job = read_json(os.path.join(folder, 'job.json'), {}) or {}
    if st is None:
        st = {'state': 'stopped' if not os.path.isdir(folder) else 'queued'}
    st.setdefault('id', job_id)
    st.setdefault('title', job.get('title', job_id))
    st.setdefault('dest', job.get('dest'))
    st.setdefault('mode', job.get('mode'))
    st.setdefault('created', job.get('created'))
    if st.get('state') not in FINAL and not pid_alive(runner_pid(folder, st)):
        started = st.get('spawned') or 0
        if time.time() - started > 20:  # give a fresh process time to write its first status
            st['state'] = 'stopped'
            st.setdefault('error', 'The transfer process ended unexpectedly (instance restarted?). Retry resumes it.')
    return st


def list_jobs() -> list:
    try:
        names = sorted(os.listdir(JOBS), reverse=True)
    except OSError:
        return []
    return [job_status(n) for n in names if os.path.isdir(os.path.join(JOBS, n))]


def spawn(folder: str, token: str, source: str) -> int:
    with open(os.path.join(folder, 'runner.py'), 'w', encoding='utf-8') as f:
        f.write(source)
    env = dict(os.environ)
    if token:
        env['HF_TOKEN'] = token
    env.update({'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
                'HF_XET_HIGH_PERFORMANCE': '1', 'PYTHONUNBUFFERED': '1'})
    log = open(os.path.join(folder, 'runner.log'), 'ab')
    proc = subprocess.Popen([sys.executable, os.path.join(folder, 'runner.py'), '--run', folder],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                            cwd=folder, start_new_session=True, close_fds=True)
    # In a file of its own, which the runner never writes: status.json belongs to the runner from here on, but a
    # Cancel that comes before the runner's first status still has to find the process.
    write_json(os.path.join(folder, 'runner.pid'), proc.pid)
    return proc.pid


def runner_pid(folder: str, st: dict):
    return st.get('pid') or read_json(os.path.join(folder, 'runner.pid'), None)


def cmd_start(p: dict) -> dict:
    job = p['job']
    job_id = job.get('id') or time.strftime('%Y%m%d-%H%M%S') + '-%04x' % (os.getpid() & 0xffff)
    folder = job_dir(job_id)
    os.makedirs(folder, exist_ok=True)
    job['id'] = job_id
    job['dest'] = expand(job['dest'])
    job['created'] = time.time()
    write_json(os.path.join(folder, 'job.json'), job)
    total = sum(int(i.get('size') or 0) for i in job.get('items', []))
    write_json(os.path.join(folder, 'status.json'),
               {'id': job_id, 'state': 'queued', 'title': job.get('title'), 'dest': job['dest'], 'mode': job.get('mode'),
                'created': job['created'], 'spawned': time.time(), 'files_total': len(job.get('items', [])),
                'files_done': 0, 'bytes_total': total, 'bytes_done': 0})
    # From here on only the runner writes status.json (its first write, with its pid, comes as it starts): a write
    # from this side after the spawn could put an older state back over the runner's newer one.
    # Until then job_status counts a fresh "queued" job without a pid as starting (20 s grace).
    spawn(folder, p.get('token') or '', base64.b64decode(p['source']).decode('utf-8'))
    return job_status(job_id)


def cmd_retry(p: dict) -> dict:
    folder = job_dir(p['id'])
    st = job_status(p['id'])
    if st.get('state') not in FINAL:
        return st
    st.update({'state': 'queued', 'error': '', 'failed': [], 'spawned': time.time(), 'finished': None, 'pid': None})
    write_json(os.path.join(folder, 'status.json'), st)
    spawn(folder, p.get('token') or '', base64.b64decode(p['source']).decode('utf-8'))  # the runner writes from now on
    return job_status(p['id'])


def cmd_cancel(p: dict) -> dict:
    folder = job_dir(p['id'])
    st = job_status(p['id'])
    if st.get('state') in FINAL:
        return st
    pid = runner_pid(folder, st)
    if pid and pid_alive(pid):
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except OSError:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except OSError:
                pass
        for _ in range(30):
            if not pid_alive(pid):
                break
            time.sleep(0.2)
        # The whole group, whether or not the runner itself went quietly: a child it started (pip installing
        # huggingface_hub, say) must not outlive a cancelled transfer.
        try:
            os.killpg(int(pid), signal.SIGKILL)
        except OSError:
            pass
    st = read_json(os.path.join(folder, 'status.json'), {}) or st
    if st.get('state') not in FINAL:
        st.update({'state': 'cancelled', 'finished': time.time()})
        write_json(os.path.join(folder, 'status.json'), st)
    return job_status(p['id'])


def cmd_forget(p: dict) -> dict:
    folder = job_dir(p['id'])
    st = job_status(p['id'])
    if st.get('state') not in FINAL:
        return {'error': 'The transfer is still running: cancel it first.'}
    job = read_json(os.path.join(folder, 'job.json'), {}) or {}
    staging = job.get('staging')
    if staging and os.path.basename(staging).startswith('.rulz-hf-partial-'):
        shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(folder, ignore_errors=True)
    return {'forgotten': p['id']}


def cmd_watch(p: dict) -> None:
    """Print the job's status as a JSON line every ``interval`` seconds until it ends or ``timeout`` passes."""
    end = time.time() + float(p.get('timeout') or 600)
    interval = max(0.5, float(p.get('interval') or 2))
    while True:
        st = job_status(p['id'])
        out(st)
        if st.get('state') in FINAL or time.time() > end:
            return
        time.sleep(interval)


# ------------------------------------------------------------------------------------------------ the runner

class Runner:
    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.job = read_json(os.path.join(folder, 'job.json'), {})
        self.status_path = os.path.join(folder, 'status.json')
        old = read_json(self.status_path, {}) or {}
        if old.get('state') == 'cancelled':
            raise SystemExit(0)  # cancelled before this process got going: leave it cancelled, start nothing
        self.items = self.job.get('items', [])
        self.mode = self.job.get('mode') or 'keep'
        self.dest = self.job['dest']
        self.staging = os.path.join(self.dest, '.rulz-hf-partial-' + self.job['id'])
        self.lock = threading.Lock()
        self.done_bytes = 0
        self.files_done = 0
        self.skipped = 0
        self.failed: list = []
        self.current: list = []
        self.stop = threading.Event()
        self.st = {'id': self.job['id'], 'title': self.job.get('title'), 'dest': self.dest, 'mode': self.mode,
                   'created': self.job.get('created'), 'pid': os.getpid(), 'spawned': old.get('spawned'),
                   'started': time.time(), 'state': 'starting', 'files_total': len(self.items), 'files_done': 0,
                   'bytes_total': sum(int(i.get('size') or 0) for i in self.items), 'bytes_done': 0,
                   'message': '', 'error': '', 'failed': [], 'speed': 0}
        self.job['staging'] = self.staging
        write_json(os.path.join(folder, 'job.json'), self.job)
        write_json(self.status_path, self.st)  # "starting", with this process's pid: the status is ours now

    # ---- status
    def cache_dir(self) -> str:
        return self.dest if os.path.basename(self.dest.rstrip('/')) == 'hub' else os.path.join(self.dest, 'hub')

    def partial_dirs(self) -> list:
        if self.mode != 'cache':
            return [self.staging]
        repos = {(i['kind'], i['repo']) for i in self.items}
        return [os.path.join(self.cache_dir(), '%ss--%s' % (k, r.replace('/', '--')), 'blobs') for k, r in repos]

    def partial_bytes(self) -> int:
        total = 0
        for root, _dirs, files in (w for d in self.partial_dirs() if os.path.isdir(d) for w in os.walk(d)):
            for name in files:
                if name.endswith('.incomplete') or (self.mode != 'cache' and '.cache' not in root):
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
        return total

    def write(self, **kw) -> None:
        with self.lock:
            self.st.update(kw)
            self.st['bytes_done'] = self.done_bytes
            self.st['files_done'] = self.files_done
            self.st['skipped'] = self.skipped
            self.st['failed'] = self.failed[:50]
            self.st['current'] = self.current[:4]
            write_json(self.status_path, self.st)

    def reporter(self) -> None:
        last = (time.time(), 0)
        while not self.stop.wait(2):
            try:
                partial = self.partial_bytes()
                now_bytes = self.done_bytes + partial
                t = time.time()
                speed = max(0.0, (now_bytes - last[1]) / max(0.1, t - last[0]))
                last = (t, now_bytes)
                with self.lock:
                    self.st['partial'] = partial
                    self.st['speed'] = speed
                self.write()
            except Exception:  # noqa: BLE001  (a status hiccup must never stop the downloads)
                pass

    # ---- huggingface_hub
    def hub(self):
        pylib = os.path.join(STATE, 'pylib-py%d%d' % sys.version_info[:2])
        marker = os.path.join(pylib, '.rulz-ok')
        if not os.path.exists(marker):
            os.makedirs(STATE, exist_ok=True)
            # Two transfers started close together on a fresh instance must not run pip into the same folder at
            # once: the second waits here, then finds the marker the first one left.
            with open(pylib + '.lock', 'w') as lock:
                try:
                    import fcntl  # noqa: PLC0415  (POSIX only; the instance is Linux)
                    fcntl.flock(lock, fcntl.LOCK_EX)
                except (ImportError, OSError):
                    pass
                if not os.path.exists(marker):
                    self.install_hub(pylib, marker)
        sys.path.insert(0, pylib)
        import huggingface_hub  # noqa: PLC0415
        return huggingface_hub

    def install_hub(self, pylib: str, marker: str) -> None:
        self.write(state='installing', message='Installing huggingface_hub on the instance (once)...')
        cmd = [sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
               '--no-warn-script-location', '--upgrade', '--target', pylib, HUB_REQUIREMENT]
        done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if done.returncode != 0:
            ensure = subprocess.run([sys.executable, '-m', 'ensurepip', '--upgrade'],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) \
                if ensure.returncode == 0 else done
        if done.returncode != 0:
            tail = done.stdout.decode(errors='replace').strip().splitlines()[-6:]
            raise RuntimeError('Could not install huggingface_hub on the instance:\n' + '\n'.join(tail))
        open(marker, 'w').close()

    # ---- one file
    @staticmethod
    def parts(path: str, what: str) -> list:
        rel = (path or '').strip('/')
        if not rel or any(part in ('', '.', '..') for part in rel.split('/')):
            raise ValueError('bad %s path %r' % (what, path))
        return rel.split('/')

    def target(self, item: dict) -> str:
        return os.path.join(self.dest, *self.parts(item['dest'], 'destination'))

    def staging_for(self, item: dict) -> str:
        """Each repo (and revision) gets a staging folder of its own: two repos that both have a config.json are
        downloaded side by side by different workers and must never share, or swap, one temporary file."""
        name = '%s--%s' % (item['kind'], item['repo'].replace('/', '--'))
        if item.get('rev'):
            # Readable, plus a hash of the exact revision: "a/b" and "a_b" must not share a folder.
            readable = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in item['rev'])[:40]
            name += '@%s-%s' % (readable, hashlib.sha1(item['rev'].encode('utf-8')).hexdigest()[:8])
        return os.path.join(self.staging, name)

    def fetch(self, hf, api, item: dict) -> None:
        kind, repo, path = item['kind'], item['repo'], item['path']
        size = item.get('size')
        if self.mode == 'cache':
            if kind == 'bucket':
                raise ValueError('buckets have no Hugging Face cache layout: pick a folder instead')
            hf.hf_hub_download(repo, path, repo_type=kind, revision=item.get('rev') or None,
                               cache_dir=self.cache_dir())
            return
        final = self.target(item)
        # Only the same size counts as "already there": an empty file in the repo must not keep an old big file.
        if size is not None and os.path.isfile(final) and os.path.getsize(final) == int(size):
            with self.lock:
                self.skipped += 1
            return
        staging = self.staging_for(item)
        staged = os.path.join(staging, *self.parts(path, 'repo file'))
        if kind == 'bucket':
            os.makedirs(os.path.dirname(staged), exist_ok=True)
            api.download_bucket_files(repo, files=[(path, staged)], raise_on_missing_files=True)
        else:
            staged = hf.hf_hub_download(repo, path, repo_type=kind, revision=item.get('rev') or None,
                                        local_dir=staging)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        try:
            os.replace(staged, final)
        except OSError:
            shutil.move(staged, final)

    def one(self, hf, api, item: dict) -> None:
        label = item['repo'] + '/' + item['path']
        with self.lock:
            self.current.append(label)
        try:
            for attempt in range(4):
                if self.stop.is_set():
                    return
                try:
                    self.fetch(hf, api, item)
                    break
                except Exception as e:  # noqa: BLE001
                    fatal = type(e).__name__ in ('RepositoryNotFoundError', 'GatedRepoError', 'EntryNotFoundError',
                                                 'RevisionNotFoundError', 'RemoteEntryNotFoundError', 'ValueError')
                    if fatal or attempt == 3:
                        with self.lock:
                            self.failed.append({'file': label, 'error': '%s: %s' % (type(e).__name__, str(e)[:400])})
                        traceback.print_exc()
                        return
                    time.sleep(3 * (attempt + 1))
            with self.lock:
                self.files_done += 1
                self.done_bytes += int(item.get('size') or 0)
        finally:
            with self.lock:
                if label in self.current:
                    self.current.remove(label)

    def run(self) -> None:
        def terminated(*_):
            self.stop.set()
            self.write(state='cancelled', message='Cancelled', finished=time.time())
            os._exit(0)
        signal.signal(signal.SIGTERM, terminated)
        try:
            hf = self.hub()
            api = hf.HfApi()
            os.makedirs(self.dest, exist_ok=True)
            self.write(state='running', message='Downloading')
            threading.Thread(target=self.reporter, daemon=True).start()
            # Big files first: they set the pace, and small ones fill the gaps.
            items = sorted(self.items, key=lambda i: -int(i.get('size') or 0))
            workers = max(1, min(int(self.job.get('workers') or 4), 8))
            lock = threading.Lock()
            queue = list(items)

            def worker() -> None:
                while not self.stop.is_set():
                    with lock:
                        if not queue:
                            return
                        item = queue.pop(0)
                    self.one(hf, api, item)

            threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.stop.set()
            if self.mode != 'cache' and not self.failed:
                shutil.rmtree(self.staging, ignore_errors=True)
            if self.failed:
                self.write(state='error', finished=time.time(), partial=0,
                           error='%d of %d file(s) failed. Retry resumes them.' % (len(self.failed), len(self.items)))
            else:
                self.write(state='done', finished=time.time(), partial=0, message='Done')
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.stop.set()
            self.write(state='error', error='%s: %s' % (type(e).__name__, e), finished=time.time())


COMMANDS = {'info': cmd_info, 'ls': cmd_ls, 'mkdir': cmd_mkdir, 'df': cmd_df, 'start': cmd_start,
            'jobs': lambda p: {'jobs': list_jobs()}, 'status': lambda p: job_status(p['id']), 'cancel': cmd_cancel,
            'retry': cmd_retry, 'forget': cmd_forget}


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == '--run':
        Runner(sys.argv[2]).run()
        return
    payload = json.loads(base64.b64decode(PAYLOAD).decode('utf-8')) if PAYLOAD else {'cmd': 'info'}
    cmd = payload.get('cmd')
    try:
        if cmd == 'watch':
            cmd_watch(payload)
            return
        if cmd not in COMMANDS:
            out({'error': 'unknown command %r' % cmd})
            return
        out(COMMANDS[cmd](payload))
    except Exception as e:  # noqa: BLE001
        out({'error': '%s: %s' % (type(e).__name__, e)})


if __name__ == '__main__':
    main()
