"""Exercise every real recipe with the trainer's own strict config validation."""
import sys
from pathlib import Path
import pytest
from PIL import Image
from rulz_optimizer.dataset_scan import DatasetScan, ImageInfo
from rulz_optimizer.export import build_config
from rulz_optimizer.knowledge import load, validate
from rulz_optimizer.recommend import Goals, recommend
from rulz_optimizer.trainer_keys import KEYS
from rulz_optimizer.diagnose import next_config, propose
from rulz_optimizer.runs import load_run

APP = Path(__file__).resolve().parents[1]
ROOT = APP / 'knowledge'


@pytest.mark.parametrize(('count', 'expected'), [(10, 3000), (30, 6000), (60, 8000)])
def test_ideogram_character_recipe(tmp_path, count, expected):
    image = ImageInfo(tmp_path / 'image.png', 1024, 1024, 1.0, 'subject')
    scan = DatasetScan([tmp_path], [image] * count, [], count, 1.0, 1024, {}, [], [], 'subject', [])
    rec = recommend(load(ROOT), 'ideogram4', scan, Goals('balanced', 'face_body', vram_gb=24))
    cfg = build_config(rec, 'test')
    assert {key: cfg[key] for key in ('network_type', 'lokr_factor', 'optimizer', 'learning_rate',
                                       'timestep_sampling', 'resolutions', 'batch_size', 'turbo_lora',
                                       'sample_steps', 'steps', 'save_every', 'keep_checkpoints',
                                       'resolution_shift')} == {
        'network_type': 'lokr', 'lokr_factor': 4, 'optimizer': 'automagic3', 'learning_rate': 1e-5,
        'timestep_sampling': 'logit_normal', 'resolutions': [0.25, 0.5, 1.0], 'batch_size': 1,
        'turbo_lora': '', 'sample_steps': 20, 'steps': expected,
        'save_every': 250, 'keep_checkpoints': 12, 'resolution_shift': True,
    }


def test_real_slow_likeness_step_rounding():
    knowledge = load(ROOT)
    proposals = propose(knowledge, 'minimax_h3_image', ['slow_likeness'])
    index = next(i for i, proposal in enumerate(proposals)
                 if 'steps' in proposal.fix.set and proposal.fix.set['steps'].op == 'mul')
    run = load_run(APP / 'tests' / 'fixtures' / 'run')
    run.config['steps'] = 1050
    result = next_config(run, proposals, [index])
    assert result['steps'] == 1350
    assert type(result['steps']) is int


def test_real_knowledge(tmp_path):
    files = list((ROOT / 'models').glob('*.yaml'))
    if not files:
        pytest.skip('Real knowledge models are not present')
    sys.path.insert(0, str(APP.parent / 'RulzTrainer'))
    try:
        from rulztrainer.config import TrainConfig
    except ImportError as exc:
        pytest.skip(f'Trainer config unavailable: {exc}')
    try:
        import torch  # noqa: F401  TrainConfig.validate() imports it
        full = True
    except ImportError:
        full = False  # CI: the type checks still run; validate() needs the trainer's own venv
    k = load(ROOT)
    errors = validate(k, set(KEYS))
    assert not errors, '\n'.join(errors)
    Image.new('RGB', (32, 32), 'red').save(tmp_path / 'image.png')
    failures = []
    for model in k.models:
        for goal in ('likeness', 'balanced', 'quick'):
            for focus in ('face', 'face_body'):
                for count in (10, 30, 120):
                    for vram in (12, 48):
                        image = ImageInfo(tmp_path / 'image.png', 1024, 1024, 1.0, 'subject')
                        scan = DatasetScan([tmp_path], [image] * count, [], count, 1.0, 1024, {}, [], [], 'subject', [])
                        try:
                            cfg = build_config(recommend(k, model, scan, Goals(goal, focus, vram_gb=vram)), 'test')
                            for key, value in cfg.items():
                                kind = KEYS[key]
                                if kind in ('list', 'int', 'float', 'bool'):
                                    assert type(value).__name__ == kind, (key, value, kind)
                            tc = TrainConfig.from_dict(cfg)
                            if full:
                                tc.validate()
                        except Exception as exc:
                            failures.append(f'{model}/{goal}/{focus}/{count}/{vram}: {exc}')
    assert not failures, '\n'.join(failures)
