"""Samples, path scrubbing in the AI context, out-of-memory detection, unknown knowledge keys."""
import shutil
from pathlib import Path

from PIL import Image

from rulz_optimizer import advisor
from rulz_optimizer.dataset_scan import DatasetScan
from rulz_optimizer.diagnose import detect, propose
from rulz_optimizer.knowledge import Setting
from rulz_optimizer.recommend import Goals, recommend
from rulz_optimizer.runs import load_run, sample_steps

FIXTURE = Path(__file__).parent / 'fixtures' / 'run'


def test_samples_grouped_by_step(tmp_path):
    run_dir = tmp_path / 'run'
    shutil.copytree(FIXTURE, run_dir)
    (run_dir / 'samples').mkdir()
    for name in ('step000250_0.png', 'step000250_1.png', 'job_step000500_0.png', 'odd.png'):
        Image.new('RGB', (8, 8)).save(run_dir / 'samples' / name)
    groups = sample_steps(load_run(run_dir))
    assert list(groups) == [-1, 250, 500]
    assert [p.name for p in groups[250]] == ['step000250_0.png', 'step000250_1.png']


def test_context_has_no_paths_outside_dataset_and_run():
    home = str(Path.home())
    scan = DatasetScan([Path(home) / 'photos'], [], [], 0, 0.0, 0, {}, [], [], None,
                       [f'{home}\\photos\\broken one.png: cannot identify image file',
                        f'{home}/photos/b.png: cannot identify image file', f'{home}\\gone: [WinError 3]'])
    text = advisor.build_context(None, 'krea2', scan=scan, question='q')
    assert 'broken one.png: cannot identify' in text and 'b.png: cannot identify' in text
    assert home not in text and 'photos\\broken' not in text and 'photos/b.png' not in text
    assert advisor.scrub_paths(f'{home}\\x\\y.png: bad') == 'y.png: bad'


def test_out_of_memory_detected_and_proposed(tmp_path):
    run_dir = tmp_path / 'run'
    shutil.copytree(FIXTURE, run_dir)
    with (run_dir / 'train.log').open('a', encoding='utf-8') as log:
        log.write('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n')
    run = load_run(run_dir)
    assert 'out_of_memory' in {f.symptom for f in detect(run)}
    clean = load_run(FIXTURE)
    assert 'out_of_memory' not in {f.symptom for f in detect(clean)}
    clean.status = {'error': 'CUDA out of memory'}
    assert 'out_of_memory' in {f.symptom for f in detect(clean)}


def test_real_general_knowledge_has_out_of_memory_fixes():
    from rulz_optimizer import knowledge
    k = knowledge.load()
    model_id = next(iter(k.models))
    assert propose(k, model_id, ['out_of_memory'])


def test_unknown_knowledge_key_is_warned_not_dropped(knowledge):
    knowledge.models['krea2'].base['no_such_setting'] = Setting('no_such_setting', 1, 'why', 'DOC', [])
    rec = recommend(knowledge, 'krea2', None, Goals())
    assert 'no_such_setting' not in rec.config()
    assert any('no_such_setting' in w for w in rec.warnings)
