from dataclasses import replace
from pathlib import Path
import math
import pytest

from rulz_optimizer.diagnose import detect, next_config, propose
from rulz_optimizer.knowledge import Fix, Flaw, Setting, source_text, validate
from rulz_optimizer.recommend import Goals, _matches, apply_operation, coerce
from rulz_optimizer.runs import _point, load_run, recorded_best_point


def test_types_and_operations():
    assert coerce('learning_rate', '1e-4') == 0.0001
    assert apply_operation(0, 0.05, 'add', 'caption_dropout') == 0.05
    assert isinstance(apply_operation(None, 0.5, 'mul', 'lora_rank'), int)
    with pytest.raises(ValueError):
        coerce('face_refs', 'one\ntwo')
    with pytest.raises(ValueError):
        coerce('steps', True)


def test_video_only_does_not_match_images(dataset):
    from rulz_optimizer.dataset_scan import scan
    data = scan([dataset])
    data.images = []
    data.videos = [Path('clip.mp4')]
    assert not _matches({'images_max': 100}, data, Goals())


def test_race_and_divergence():
    run = load_run(Path(__file__).parent / 'fixtures' / 'run')
    run.race = {'winner': 'a', 'last': 'a', 'best_step': {'a': 10}}
    a = _point({'step': 10, 'combined_norm': 40, 'images': ['a_step000010_face_1.png']})
    b = _point({'step': 10, 'combined_norm': 90, 'images': ['b_step000010_face_1.png']})
    run.probes = [a, b]
    assert recorded_best_point(run) is a
    run.loss_series = [(1, math.nan)]
    assert any(f.symptom == 'nan_or_diverged' and 'a:' in f.detail for f in detect(run))
    run.loss_series = []
    run.status['skipped_steps'] = 1
    assert any(f.symptom == 'nan_or_diverged' for f in detect(run))


def test_multi_key_fix_and_validation(knowledge):
    setting = lambda key, value: Setting(key, value, '', 'DOC', [])
    fix = Fix({'lora_rank': setting('lora_rank', 8), 'lora_alpha': setting('lora_alpha', 8)}, None, '', 'DOC', [])
    other = Fix({'lora_alpha': setting('lora_alpha', 16)}, None, '', 'DOC', [])
    knowledge.models['krea2'].flaws = [Flaw('plastic_skin', [], [fix, other])]
    proposals = propose(knowledge, 'krea2', ['plastic_skin'])
    assert len(proposals[0].fix.set) == 2 and proposals[1].alternative_of == 0
    run = load_run(Path(__file__).parent / 'fixtures' / 'run')
    cfg = next_config(run, proposals, [0])
    assert cfg['lora_rank'] == cfg['lora_alpha'] == 8
    knowledge.models['krea2'].steps['min'] = 999999
    assert any('min exceeds max' in error for error in validate(knowledge, set(__import__('rulz_optimizer.trainer_keys', fromlist=['KEYS']).KEYS)))
    assert source_text(knowledge, 'unknown', 'unknown') == 'unknown'
