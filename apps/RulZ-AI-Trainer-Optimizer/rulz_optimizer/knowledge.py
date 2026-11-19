"""Load and validate the data-only recommendation knowledge base."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from .trainer_keys import ARCH_NAMES


class KnowledgeError(Exception):
    pass


@dataclass
class Setting:
    key: str
    value: Any
    why: str
    confidence: str
    sources: list[str]
    op: str | None = None
    source_value: Any = None  # the rate as the source quoted it, when it was converted to alpha = rank
    source_alpha: float | None = None
    source_rank: float | None = None


@dataclass
class Fix:
    set: dict[str, Setting]
    action: str | None
    why: str
    confidence: str
    sources: list[str]


@dataclass
class Flaw:
    symptom: str
    causes: list[str]
    fixes: list[Fix]


@dataclass
class Rule:
    id: str
    when: dict
    set: dict[str, Setting]
    note: str | None


@dataclass
class ModelKnowledge:
    id: str
    name: str
    kind: str
    updated: Any
    summary: str
    model_preset: str | None
    model_path: str | None
    variants: list[dict]
    facts: list[dict]
    base: dict[str, Setting]
    steps: dict
    adjust: list[Rule]
    evaluate: dict
    flaws: list[Flaw]
    open_questions: list[str]
    sources: dict[str, dict]


@dataclass
class Symptom:
    id: str
    label: str
    kind: str
    description: str


@dataclass
class Knowledge:
    general: dict
    symptoms: dict[str, Symptom]
    general_flaws: list[Flaw]
    models: dict[str, ModelKnowledge]
    general_sources: dict[str, dict]


def default_dir() -> Path:
    return Path(__file__).resolve().parents[1] / 'knowledge'


def to_alpha_rank(rate, alpha, rank):
    """A LoRA update is scaled by alpha / rank, so a learning rate quoted at another alpha and rank acts like
    rate x alpha / rank at alpha = rank (RulZ AI Trainer's convention, scale 1)."""
    return rate * alpha / rank


def _positive(key, name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ValueError(f'{key}: {name} must be a positive number')
    return float(value)


def _setting(key, value, parent):
    why = value.get('why', parent.get('why', ''))
    amount, op = value['value'], value.get('op')
    confidence = value.get('confidence', parent.get('confidence', ''))
    sources = value.get('sources', parent.get('sources', []))
    has_alpha, has_rank = 'source_alpha' in value, 'source_rank' in value
    if not (has_alpha or has_rank):
        return Setting(key, amount, why, confidence, sources, op)
    # Optional: the alpha and rank the quoted learning rate was trained at. Converted here, so the recommender sees
    # an alpha = rank rate and unannotated entries behave as before.
    if not (has_alpha and has_rank):
        raise ValueError(f'{key}: source_alpha and source_rank must be given together')
    if key != 'learning_rate':
        raise ValueError(f'{key}: source_alpha and source_rank apply to learning_rate only')
    if op == 'mul':
        raise ValueError(f'{key}: a multiplier is not a rate; source_alpha and source_rank cannot go with op mul')
    alpha = _positive(key, 'source_alpha', value['source_alpha'])
    rank = _positive(key, 'source_rank', value['source_rank'])
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise ValueError(f'{key}: a converted learning rate must be a number')
    converted = to_alpha_rank(float(amount), alpha, rank)
    if abs(alpha - rank) > 1e-9:
        why = (why + ' ' if why else '') + (f'Quoted as {float(amount):.4g} at alpha {alpha:g} and rank {rank:g}; '
                                            f'{converted:.4g} at alpha = rank.')
    return Setting(key, converted, why, confidence, sources, op, amount, alpha, rank)


def _settings(data, parent=None):
    parent = parent or {}
    for key, value in data.items():
        if not isinstance(value, dict) or 'value' not in value:
            raise ValueError(f'{key}: setting must be a mapping with a value')
    return {key: _setting(key, value, parent) for key, value in data.items()}


def _flaws(data):
    return [Flaw(f['symptom'], f.get('causes', []), [
        Fix(_settings(x.get('set', {}), x), x.get('action'), x.get('why', ''),
            x.get('confidence', ''), x.get('sources', [])) for x in f.get('fixes', [])]) for f in data]


def _read(path):
    try:
        value = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('root must be a mapping')
        return value
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise KnowledgeError(f'{path}: {exc}') from exc


def load(root: Path | None = None) -> Knowledge:
    root = Path(root) if root is not None else default_dir()
    path = root / 'general.yaml'
    general = _read(path)
    try:
        symptoms = {s['id']: Symptom(s['id'], s['label'], s['kind'], s['description'])
                    for s in general['symptoms']}
        general_flaws = _flaws(general.get('flaws', []))
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise KnowledgeError(f'{path}: {exc}') from exc
    models = {}
    for path in sorted((root / 'models').glob('*.yaml')):
        data = _read(path)
        try:
            if data['id'] != path.stem:
                raise ValueError('id must match filename')
            variants = []
            for variant in data.get('variants', []):
                variant = dict(variant)
                variant['base'] = _settings(variant.get('base', {}))
                variants.append(variant)
            model = ModelKnowledge(data['id'], data['name'], data['kind'], data['updated'], data['summary'],
                data.get('model_preset'), data.get('model_path'), variants, data.get('facts', []),
                _settings(data['base']), data['steps'],
                [Rule(r['id'], r.get('when', {}), _settings(r['set']), r.get('note')) for r in data.get('adjust', [])],
                data.get('evaluate', {}), _flaws(data.get('flaws', [])), data.get('open_questions', []), data.get('sources', {}))
            models[model.id] = model
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise KnowledgeError(f'{path}: {exc}') from exc
    return Knowledge(general, symptoms, general_flaws, models, general.get('sources', {}))


def validate(k: Knowledge, trainer_keys: set[str]) -> list[str]:
    problems = []
    confidences = {'CONF', 'REP', 'DISP', 'MEAS', 'DOC', 'GUESS'}
    conditions = {'images_min', 'images_max', 'videos_min', 'videos_max', 'goal', 'focus', 'vram_min_gb',
                  'vram_max_gb', 'target', 'has_captions', 'variant', 'median_mp_max', 'median_mp_min'}
    def check(item, sources, where):
        confidence = item.confidence if hasattr(item, 'confidence') else item.get('confidence')
        refs = item.sources if hasattr(item, 'sources') else item.get('sources', [])
        if confidence not in confidences:
            problems.append(f'{where}: bad confidence {confidence}')
        for ref in refs:
            if ref not in sources:
                problems.append(f'{where}: unknown source {ref}')
    def settings(values, sources, where):
        for key, value in values.items():
            if key not in trainer_keys:
                problems.append(f'{where}: unknown TrainConfig key {key}')
            check(value, sources, f'{where}.{key}')
            if value.op not in (None, 'mul', 'add', 'min', 'max'):
                problems.append(f'{where}.{key}: unknown operation {value.op}')
    def flaws(items, sources, where):
        for flaw in items:
            if flaw.symptom not in k.symptoms:
                problems.append(f'{where}: unknown symptom {flaw.symptom}')
            for fix in flaw.fixes:
                check(fix, sources, f'{where}.{flaw.symptom}')
                settings(fix.set, sources, f'{where}.{flaw.symptom}')
    flaws(k.general_flaws, k.general_sources, 'general.flaws')
    for fact in k.general.get('general_facts', []):
        check(fact, k.general_sources, 'general.general_facts')
    for model in k.models.values():
        sources = {**k.general_sources, **model.sources}
        if model.id not in ARCH_NAMES:
            problems.append(f'{model.id}: unknown arch id')
        settings(model.base, sources, f'{model.id}.base')
        for variant in model.variants:
            settings(variant.get('base', {}), sources, f'{model.id}.variants')
        for rule in model.adjust:
            for key in rule.when.keys() - conditions:
                problems.append(f'{model.id}.adjust.{rule.id}: unknown condition {key}')
            settings(rule.set, sources, f'{model.id}.adjust.{rule.id}')
        check(model.steps, sources, f'{model.id}.steps')
        for key in ('epochs', 'min', 'max', 'round_to'):
            value = model.steps.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f'{model.id}.steps: {key} must be a number')
        lo, hi = model.steps.get('min'), model.steps.get('max')
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
            problems.append(f'{model.id}.steps: min exceeds max')
        if isinstance(model.steps.get('round_to'), (int, float)) and model.steps['round_to'] <= 0:
            problems.append(f'{model.id}.steps: round_to must be positive')
        for fact in model.facts:
            check(fact, sources, f'{model.id}.facts')
        flaws(model.flaws, sources, f'{model.id}.flaws')
    return problems


def source_text(k: Knowledge, model_id: str, source_id: str) -> str:
    model = k.models.get(model_id)
    source = (model.sources.get(source_id) if model else None) or k.general_sources.get(source_id)
    if not source:
        return source_id
    parts = [str(source[x]) for x in ('where', 'date') if source.get(x)]
    if source.get('people') is not None:
        parts.append(f"{source['people']} people")
    return ', '.join(parts)
