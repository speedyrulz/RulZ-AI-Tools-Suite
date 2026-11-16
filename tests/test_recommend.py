from rulz_optimizer.dataset_scan import scan
from rulz_optimizer.recommend import Goals, recommend
from rulz_optimizer.recommend import _matches, apply_operation
import pytest


def test_recommend(knowledge, dataset):
    rec = recommend(knowledge, 'krea2', scan([dataset]), Goals())
    assert rec.config()['lora_rank'] == 8
    assert rec.config()['steps'] == 1000
    assert rec.config()['datasets'][0]['trigger'] == dataset.name  # the folder name, not the captions' first word


def test_variant(knowledge):
    rec = recommend(knowledge, 'minimax_h3', None, Goals(variant='ref2va'))
    assert rec.config()['model_path'].endswith('#ref2va')
    assert rec.config()['lora_rank'] == 8


@pytest.mark.parametrize('condition', [
    {'images_min': 1}, {'images_max': 1}, {'videos_min': 0}, {'videos_max': 0},
    {'goal': ['balanced']}, {'focus': 'face_body'}, {'vram_min_gb': 12}, {'vram_max_gb': 12},
    {'target': 'local'}, {'has_captions': True}, {'variant': ['test']},
    {'median_mp_min': 0.001}, {'median_mp_max': 0.002},
])
def test_conditions(dataset, condition):
    assert _matches(condition, scan([dataset]), Goals(vram_gb=12, variant='test'))


@pytest.mark.parametrize('op,value,expected', [('mul', 2, 20), ('add', 2, 12), ('min', 2, 2), ('max', 12, 12)])
def test_operations(op, value, expected):
    assert apply_operation(10, value, op) == expected


def test_steps_and_base(knowledge, dataset, tmp_path):
    data = scan([dataset])
    spec = knowledge.models['krea2'].steps
    spec.update(epochs=1025, min=1000, max=4000, round_to=50)
    base = tmp_path / 'base.yaml'
    base.write_text('model_path: custom\nblock_swap: 3\nlora_rank: 99\n', encoding='utf-8')
    rec = recommend(knowledge, 'krea2', data, Goals(base_config=base))
    assert rec.steps == 1050 and rec.config()['model_path'] == 'custom'
    assert rec.config()['block_swap'] == 3 and rec.config()['lora_rank'] == 8
    spec['epochs'] = 9000
    assert recommend(knowledge, 'krea2', data, Goals()).steps == 4000
    spec['epochs'] = 2
    assert recommend(knowledge, 'krea2', data, Goals()).steps == 1000
    assert rec.config()['probe_every'] and rec.config()['face_refs'] and rec.config()['body_refs']
