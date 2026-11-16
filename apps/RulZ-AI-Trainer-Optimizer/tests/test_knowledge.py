from rulz_optimizer.knowledge import validate, source_text
from rulz_optimizer.trainer_keys import KEYS
from rulz_optimizer.knowledge import Rule, Setting


def test_validate(knowledge):
    assert validate(knowledge, set(KEYS)) == []
    assert source_text(knowledge, 'krea2', 'test') == 'Test, 2026-10, 1 people'


def test_validation_errors(knowledge):
    model = knowledge.models['krea2']
    model.base['not_a_key'] = Setting('not_a_key', 1, 'why', 'BAD', ['missing'])
    model.adjust.append(Rule('bad', {'unknown_condition': 1}, {}, None))
    errors = ' '.join(validate(knowledge, set(KEYS)))
    assert all(s in errors for s in ('not_a_key', 'BAD', 'missing', 'unknown_condition'))
