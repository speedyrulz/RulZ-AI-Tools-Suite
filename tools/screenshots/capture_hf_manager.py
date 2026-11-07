"""Runs RulZ Huggingface Manager on a made-up account, for the catalog screenshots.

Nothing reaches huggingface.co. Who the token belongs to, the namespace listing, repo summaries, branches, commit
history and the file listings are replaced with a fictional account ("demo-user", organization "demo-lab") and
its repos. Settings go to a fresh folder (--data, with a fake token), and HF_TOKEN and the hf CLI login are hidden,
so no real token or account is ever read.

    <venv>\\python.exe -B capture_hf_manager.py --app <suite>\\apps\\RulZ-Huggingface-Manager --data R:\\hf-data --port 8799

--choose-look starts as a first run (the "Choose your look" dialog shows); --light starts in the light look.
The venv needs the app's requirements (nicegui, huggingface_hub). Stop it with Ctrl+C or taskkill.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

NOW = datetime.now(timezone.utc)
GB = 1024 ** 3
MB = 1024 ** 2
KB = 1024


def ago(**kw) -> datetime:
    return NOW - timedelta(**kw)


# kind, id, visibility, updated, size, downloads, likes, files
REPOS = [
    ('model', 'demo-user/watercolor-sdxl-lora', 'public', ago(hours=3), int(1.71 * GB), 1840, 42, None),
    ('model', 'demo-user/tiny-chat-3b-gguf', 'public', ago(days=2), int(9.84 * GB), 12503, 118, None),
    ('model', 'demo-user/ink-sketch-lora-v2', 'private', ago(days=5), int(436 * MB), None, None, None),
    ('model', 'demo-lab/caption-vlm-7b', 'private', ago(days=9), int(15.2 * GB), None, None, None),
    ('model', 'demo-user/pixel-art-lora', 'public', ago(days=31), int(218 * MB), 655, 17, None),
    ('dataset', 'demo-user/landscape-captions-10k', 'public', ago(days=1), int(2.37 * GB), 930, 21, None),
    ('dataset', 'demo-user/watercolor-training-set', 'private', ago(days=4), int(812 * MB), None, None, None),
    ('dataset', 'demo-lab/chat-eval-prompts', 'public', ago(days=18), int(36 * MB), 411, 8, None),
    ('space', 'demo-user/caption-playground', 'public', ago(days=6), int(4 * MB), None, 64, None),
    ('space', 'demo-user/lora-gallery', 'private', ago(days=22), int(2 * MB), None, 3, None),
    ('bucket', 'demo-user/training-runs', 'private', ago(hours=20), int(48.6 * GB), None, None, 1284),
    ('bucket', 'demo-lab/shared-checkpoints', 'private', ago(days=12), int(121.4 * GB), None, None, 312),
]

# folder -> [(name, is_dir, size, modified, storage)]   storage: '' | 'lfs' | 'xet'
MODEL_FILES = {
    '': [
        ('checkpoints', True, None, None, ''),
        ('samples', True, None, None, ''),
        ('.gitattributes', False, 1_570, ago(days=40), ''),
        ('README.md', False, 4_812, ago(hours=3), ''),
        ('training_config.yaml', False, 2_104, ago(days=2), ''),
        ('watercolor_sdxl_lora.safetensors', False, int(228.5 * MB), ago(hours=3), 'xet'),
        ('watercolor_sdxl_lora_r16.safetensors', False, int(114.3 * MB), ago(days=2), 'xet'),
        ('preview.png', False, int(1.9 * MB), ago(hours=3), 'xet'),
        ('trigger_words.txt', False, 64, ago(days=2), ''),
    ],
    'checkpoints': [
        ('step_01000.safetensors', False, int(228.5 * MB), ago(days=3), 'xet'),
        ('step_02000.safetensors', False, int(228.5 * MB), ago(days=3), 'xet'),
        ('step_03000.safetensors', False, int(228.5 * MB), ago(days=2), 'xet'),
        ('step_04000.safetensors', False, int(228.5 * MB), ago(days=2), 'xet'),
    ],
    'samples': [(f'sample_{i:02d}.png', False, int((1.2 + i / 10) * MB), ago(days=2), 'lfs') for i in range(1, 9)],
}

README = """---
license: openrail++
base_model: stabilityai/stable-diffusion-xl-base-1.0
tags: [lora, watercolor]
---

# Watercolor SDXL LoRA

Soft watercolor washes with visible paper texture. Trigger word: `wtrclr`.

| Setting | Value |
| --- | --- |
| Rank | 32 |
| Steps | 4000 |
| Images | 120 |
"""


def install_demo(hub) -> None:
    """Replace everything in hub that talks to the Hub with the demo account."""
    ident = hub.Identity(name='demo-user', fullname='Demo User', orgs=['demo-lab'], token_name='rulz-manager',
                         token_role='write', auth_type='access_token')
    hub.whoami = lambda force=False: ident
    hub.token_source = lambda: 'app'
    hub.external_token = lambda: None

    def list_namespace(namespace, kinds=hub.KINDS):
        return [hub.RepoRow(k, i, v, u, s, d, li, f) for k, i, v, u, s, d, li, f in REPOS
                if i.split('/')[0] == namespace and k in kinds]
    hub.list_namespace = list_namespace

    def repo_summary(kind, repo_id):
        row = next((r for r in REPOS if r[0] == kind and r[1] == repo_id), None)
        if not row:
            raise hub.hf_errors.RepositoryNotFoundError('not found', response=None)
        k, i, v, u, s, d, li, f = row
        out = {'private': v == 'private', 'size': s, 'updated': u}
        if kind == 'bucket':
            return {**out, 'files': f, 'created': u}
        return {**out, 'likes': li, 'downloads': d, 'sha': '3f9c2a7e', 'gated': None}
    hub.repo_summary = repo_summary

    class DemoFS(hub.RepoFS):
        def list_dir(self, folder):
            entries = []
            for name, is_dir, size, modified, storage in MODEL_FILES.get(folder.strip('/'), []):
                entries.append(hub.Entry(hub.join(folder, name), is_dir, size, modified, lfs=storage == 'lfs',
                                         xet_hash='demo' if storage == 'xet' else None, raw=object()))
            return entries

        def list_files(self, folder=''):
            return [e for e in self.list_dir(folder) if not e.is_dir]

        def existing(self, paths):
            return set()

        def branches(self):
            return ['main', 'experiment-r16']

        def history(self, limit=200):
            titles = ['Upload watercolor_sdxl_lora.safetensors', 'Add samples', 'Update README.md',
                      'Upload checkpoints', 'Add training config', 'Initial commit']
            return [{'id': f'{n:x}' * 40, 'short': (f'{n:x}' * 8)[:8], 'title': t, 'authors': 'demo-user',
                     'date': ago(hours=3 + 30 * n)} for n, t in enumerate(titles, start=1)][:limit]

        def read_bytes(self, path):
            return README.encode() if path.lower().endswith('.md') else b'{}'

    hub.open_fs = lambda kind, repo_id, revision=None: DemoFS(kind, repo_id, revision)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--app', required=True, help='The RulZ-Huggingface-Manager folder')
    p.add_argument('--data', required=True, help='Fresh folder for settings (and NiceGUI storage)')
    p.add_argument('--port', type=int, default=8799)
    p.add_argument('--light', action='store_true')
    p.add_argument('--choose-look', action='store_true')
    a = p.parse_args()

    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    for var in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_TOKEN_PATH'):
        os.environ.pop(var, None)
    os.environ['HF_HOME'] = str(data / 'hf-home')  # no hf CLI login here
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['RULZ_HF_CONFIG_DIR'] = str(data / 'config')
    (data / 'config').mkdir(exist_ok=True)
    (data / 'config' / 'settings.json').write_text(json.dumps({
        'token': 'hf_demoTokenForScreenshotsOnly0000000',
        'download_dir': r'R:\Downloads\HuggingFace',
        'dark_mode': 'light' if a.light else 'dark',
        'look_chosen': not a.choose_look,
        'favorites': [{'kind': 'model', 'id': 'demo-user/watercolor-sdxl-lora'},
                      {'kind': 'bucket', 'id': 'demo-user/training-runs'}],
        'recent': [{'kind': 'dataset', 'id': 'demo-user/landscape-captions-10k'},
                   {'kind': 'model', 'id': 'demo-user/tiny-chat-3b-gguf'}],
    }, indent=2), encoding='utf-8')
    os.chdir(data)  # NiceGUI keeps its .nicegui storage folder in the working directory

    sys.path.insert(0, str(Path(a.app).resolve()))
    from rulz_hf_manager import hub, main as app_main

    install_demo(hub)
    app_main.main(['--port', str(a.port), '--no-browser'])


if __name__ == '__main__':
    main()
