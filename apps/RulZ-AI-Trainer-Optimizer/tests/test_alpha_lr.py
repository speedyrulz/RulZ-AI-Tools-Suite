"""Alpha-aware learning rates: knowledge conversion to alpha = rank and the run-analysis finding."""
import shutil
from dataclasses import replace
from pathlib import Path
import pytest
from rulz_optimizer.diagnose import alpha_findings, lr_at_alpha_rank
from rulz_optimizer.knowledge import KnowledgeError, load, validate
from rulz_optimizer.recommend import Goals, recommend
from rulz_optimizer.runs import load_run
from rulz_optimizer.trainer_keys import KEYS

APP = Path(__file__).resolve().parents[1]
FIXTURES = APP / 'tests' / 'fixtures'
OLD = '  learning_rate: {value: 0.0001, why: Test rate, confidence: REP, sources: [test]}'


def knowledge_with(tmp_path, line):
    root = tmp_path / 'knowledge'
    shutil.copytree(FIXTURES / 'knowledge', root)
    path = root / 'models' / 'krea2.yaml'
    text = path.read_bytes().decode('utf-8')
    assert text.count(OLD) == 1
    path.write_bytes(text.replace(OLD, line).encode('utf-8'))
    return root


def test_conversion_to_alpha_rank(tmp_path):
    root = knowledge_with(tmp_path, '  learning_rate: {value: 3.0e-4, source_alpha: 1, source_rank: 16, why: OT preset, '
                                    'confidence: REP, sources: [test]}')
    k = load(root)
    setting = k.models['krea2'].base['learning_rate']
    assert setting.value == pytest.approx(1.875e-5)
    assert (setting.source_value, setting.source_alpha, setting.source_rank) == (3.0e-4, 1.0, 16.0)
    assert 'alpha 1 and rank 16' in setting.why and 'OT preset' in setting.why
    assert validate(k, set(KEYS)) == []
    rec = recommend(k, 'krea2', None, Goals())
    assert rec.choices['learning_rate'].value == pytest.approx(1.875e-5)


def test_unannotated_entries_unchanged(knowledge, tmp_path):
    setting = knowledge.models['krea2'].base['learning_rate']
    assert setting.value == 0.0001 and setting.source_alpha is None and setting.why == 'Test rate'
    same = load(knowledge_with(tmp_path, '  learning_rate: {value: 1.0e-4, source_alpha: 16, source_rank: 16, why: Test rate, '
                                         'confidence: REP, sources: [test]}'))
    assert same.models['krea2'].base['learning_rate'].value == pytest.approx(1.0e-4)
    assert same.models['krea2'].base['learning_rate'].why == 'Test rate'  # alpha = rank: nothing to say


@pytest.mark.parametrize('line', [
    '{value: 3.0e-4, source_alpha: 1}',
    '{value: 3.0e-4, source_alpha: 1, source_rank: 0}',
    '{value: 0.5, op: mul, source_alpha: 1, source_rank: 16}',
])
def test_bad_annotations_refused(tmp_path, line):
    root = knowledge_with(tmp_path, f'  learning_rate: {line}'.replace('}', ', why: x, confidence: REP, sources: [test]}'))
    with pytest.raises(KnowledgeError):
        load(root)


def run_with(**config):
    run = load_run(FIXTURES / 'run')
    base = {'model_arch': 'krea2', 'network_type': 'lora', 'optimizer': 'adamw8bit', 'learning_rate': 1e-4,
            'lora_rank': 16, 'lora_alpha': 16.0}
    return replace(run, config={**base, **config})


def test_lr_at_alpha_rank_rules():
    assert lr_at_alpha_rank(run_with(learning_rate=3e-4, lora_alpha=1.0).config) == pytest.approx(1.875e-5)
    assert lr_at_alpha_rank(run_with(network_type='oft', lora_alpha=1.0).config) is None
    assert lr_at_alpha_rank(run_with(network_type='lokr', lora_alpha=1.0).config) is None
    assert lr_at_alpha_rank(run_with(finetune=True, lora_alpha=1.0).config) is None
    for name in ('automagic3', 'automagic4', 'keel', 'keel3_fb', 'hull'):
        assert lr_at_alpha_rank(run_with(optimizer=name, lora_alpha=1.0).config) is None
    # Isolation (DOP) trains an Automagic pick with AdamW, so the rate applies again (trainer: effective_optimizer).
    assert lr_at_alpha_rank(run_with(optimizer='automagic4', isolation=True, dop_weight=2.0, learning_rate=1e-4,
                                     lora_alpha=8.0).config) == pytest.approx(5e-5)


def test_finding_shown_and_thresholds(knowledge):
    recommended = recommend(knowledge, 'krea2', None, Goals()).choices['learning_rate'].value
    assert alpha_findings(run_with(), knowledge) == []  # alpha = rank: nothing to show
    # Within 3x: shown as information with the alpha = rank rate, not flagged.
    f, = alpha_findings(run_with(learning_rate=recommended * 2.9 * 16, lora_alpha=1.0), knowledge)
    assert f.severity == 'info' and f.symptom == 'alpha_lr'
    assert f.evidence['lr_at_alpha_rank'] == pytest.approx(recommended * 2.9)
    assert 'at alpha = rank' in f.detail
    f, = alpha_findings(run_with(learning_rate=recommended * 3.1 * 16, lora_alpha=1.0), knowledge)
    assert f.severity == 'warn' and f.symptom == 'alpha_lr_off' and 'above the recommended' in f.detail
    f, = alpha_findings(run_with(learning_rate=recommended / 2.9 * 16, lora_alpha=1.0), knowledge)
    assert f.severity == 'info'
    f, = alpha_findings(run_with(learning_rate=recommended / 3.1 * 16, lora_alpha=1.0), knowledge)
    assert f.severity == 'warn' and 'below the recommended' in f.detail
    # The OneTrainer preset (3e-4, alpha 1, rank 16) is about 1.9e-5, over 3x below a 1e-4 recommendation.
    f, = alpha_findings(run_with(learning_rate=3e-4, lora_alpha=1.0), knowledge)
    assert f.symptom == 'alpha_lr_off' and '1.9e-05' in f.detail


def test_finding_skipped_where_not_applicable(knowledge):
    for config in ({'optimizer': 'automagic4'}, {'optimizer': 'keel3'}, {'network_type': 'oft'}, {'finetune': True}):
        assert alpha_findings(run_with(lora_alpha=1.0, **config), knowledge) == []


def test_finding_without_knowledge_is_information_only():
    f, = alpha_findings(run_with(learning_rate=3e-4, lora_alpha=1.0), None)
    assert f.severity == 'info' and f.evidence['lr_at_alpha_rank'] == pytest.approx(1.875e-5)


def test_self_setting_optimizers_match_the_trainer():
    # the copy in diagnose.py must follow the trainer's list when the trainer sits next to this app
    import sys
    from pathlib import Path
    from rulz_optimizer.diagnose import SELF_SETTING_RATE

    trainer = Path(__file__).resolve().parents[2] / 'RulzTrainer'
    if not (trainer / 'rulztrainer' / 'config.py').exists():
        pytest.skip('RulZ AI Trainer is not next to this app')
    sys.path.insert(0, str(trainer))
    try:
        from rulztrainer.config import SELF_SETTING_RATE as TRAINER
    except ImportError as exc:
        pytest.skip(f'Trainer config unavailable: {exc}')
    assert set(SELF_SETTING_RATE) == set(TRAINER)


def test_finding_under_isolation_names_adamws_rate():
    f, = alpha_findings(run_with(optimizer='keel', isolation=True, dop_weight=2.0, learning_rate=1e-6,
                                 lora_rank=16, lora_alpha=8.0), None)
    assert f.evidence['lr_at_alpha_rank'] == pytest.approx(1e-4)
    assert "AdamW's learning rate under isolation 2.0e-04 acts like 1.0e-04" in f.detail
