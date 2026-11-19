"""Turn a model recipe and dataset summary into trainer settings."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from .dataset_scan import DatasetScan
from .knowledge import Knowledge, Setting
from .trainer_keys import KEYS, DEFAULTS, MODEL_KEYS, ARCH_FULL_SAMPLE, ARCH_TURBO


@dataclass
class Goals:
    goal: str = 'balanced'
    focus: str = 'face_body'
    target: str = 'local'
    vram_gb: float | None = None
    trigger: str | None = None
    variant: str | None = None
    base_config: Path | None = None


@dataclass
class Choice:
    key: str
    value: Any
    why: str
    confidence: str
    sources: list[str]
    origin: str


@dataclass
class Recommendation:
    model_id: str
    goals: Goals
    choices: dict[str, Choice]
    steps: int
    warnings: list[str]
    notes: list[str]
    evaluate: dict
    facts: list[dict]
    open_questions: list[str]

    def config(self) -> dict:
        return {key: coerce(key, c.value) for key, c in self.choices.items() if key in KEYS}


def coerce(key, value):
    kind = KEYS[key]
    if kind in ('int', 'float') and isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f'{key}: expected {kind}') from None
    if kind == 'int' and isinstance(value, (int, float)) and not isinstance(value, bool):
        if int(value) != value:
            raise ValueError(f'{key}: expected an integer')
        return int(value)
    if kind == 'float' and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if kind in ('list', 'bool', 'str', 'dict') and type(value).__name__ != kind:
        raise ValueError(f'{key}: expected {kind}')
    if kind in ('int', 'float'):
        raise ValueError(f'{key}: expected {kind}')
    return value


def apply_operation(old, value, op, key=None):
    if key and op:
        old = coerce(key, DEFAULTS[key] if old is None else old)
        # Multipliers may be fractional even for integer settings.
        value = float(value) if isinstance(value, str) else value
    if op is None:
        return coerce(key, value) if key else value
    if old is None:
        return value
    result = {'mul': lambda: old * value, 'add': lambda: old + value,
              'min': lambda: min(old, value), 'max': lambda: max(old, value)}[op]()
    if key and KEYS[key] == 'int':
        result = round(result)
    return coerce(key, result) if key else result


def _matches(conditions, scan, goals):
    values = {'images': scan.image_count if scan else None, 'videos': len(scan.videos) if scan else None,
              'median_mp': scan.median_mp if scan and scan.images else None, 'vram': goals.vram_gb}
    for key, wanted in conditions.items():
        if key in ('goal', 'focus', 'target', 'variant'):
            actual = getattr(goals, key)
            if actual not in (wanted if isinstance(wanted, list) else [wanted]):
                return False
        elif key == 'has_captions':
            if scan is None or scan.has_captions != wanted:
                return False
        else:
            name = key.replace('_min_gb', '').replace('_max_gb', '').replace('_min', '').replace('_max', '')
            actual = values.get(name)
            # Image-count rules describe still-image datasets, not video-only datasets.
            if name == 'images' and actual == 0:
                return False
            if actual is None or ('_min' in key and actual < wanted) or ('_max' in key and actual > wanted):
                return False
    return True


def recommend(k: Knowledge, model_id: str, scan: DatasetScan | None, goals: Goals) -> Recommendation:
    model = k.models[model_id]
    choices, notes, warnings = {}, [], list(scan.warnings) if scan else []
    def put(key, value, why, confidence='GUESS', sources=None, origin='default'):
        choices[key] = Choice(key, value, why, confidence, sources or [], origin)
    def apply(settings, origin):
        for key, setting in settings.items():
            if key not in KEYS:  # shown and warned about at the end; config() leaves it out
                put(key, setting.value, setting.why, setting.confidence, setting.sources, origin)
                continue
            old = choices[key].value if key in choices else None
            put(key, apply_operation(old, setting.value, setting.op, key), setting.why,
                setting.confidence, setting.sources, origin)
    def wizard(key, value, why):
        put(key, value, why, 'DOC', origin='trainer_wizard')
    paths = []
    for path in (scan.folders if scan else []):
        path = Path(path).resolve()
        if path not in paths:
            paths.append(path)
    paths = [p for p in paths if not any(p != other and other in p.parents for other in paths)]
    folders = [str(p) for p in paths]
    trigger = goals.trigger or (paths[0].name.lower().replace(' ', '') if paths else 'character')
    if not trigger:
        trigger = 'subject'
        warnings.append('Dataset folder has no name; using subject as the trigger.')
    if scan and scan.trigger_found and scan.trigger_found.lower() != trigger.lower():
        warnings.append(f'Caption trigger appears to be {scan.trigger_found}, not {trigger}.')
    wizard('gradient_checkpointing', True, 'Saves most activation memory for a modest slowdown.')
    wizard('checkpoint_keep_blocks', -1, 'Measured at the start: as many blocks as fit skip recompute.')
    wizard('compile_model', 'auto', 'Auto skips compilation for short test runs.')
    # The trainer uses 1 MP from 12 GB and 0.5 MP below that.
    target = 1.0 if goals.vram_gb is None or goals.vram_gb >= 12 else 0.5
    reason = '1 MP is the model training size.' if target == 1.0 else '0.5 MP keeps memory low.'
    # Avoid upscaling when the median image is below 90% of the target.
    if scan and scan.median_mp and scan.median_mp < target * 0.9:
        target = max(0.25, int(scan.median_mp * 4) / 4)
        reason = 'Training above the median image size only upscales it.'
    if model_id == 'qwen_image21' and target >= 0.5:
        target, reason = 0.5, 'Qwen Image 2.1 trains quickly at 0.5 MP and retains sharpness.'
    if model_id in ('minimax_h3', 'minimax_h3_image', 'p_h3_image'):
        target, reason = 1.0, 'H3 uses about 1 MP and 1024 image tokens a step.'
    wizard('resolutions', [target], reason)
    wizard('enable_buckets', True, 'Keeps every image aspect ratio instead of cropping it square.')
    wizard('flip_augment', False, 'Faces are not symmetric; mirrored copies blur details.')
    wizard('crop_augment', 'face' if goals.focus == 'face' else 'face+bust', 'Close-up crops teach face detail and, for face and body, the bust.')
    wizard('region_weights', True, 'The best measured single setting for face likeness.')
    wizard('look_warmup', False, 'No outliers are available from saved look scans.')
    wizard('isolation', bool(trigger), 'Keeps the subject face off other people; needs a trigger word.')
    wizard('batch_size', 1, 'Small batches with more steps beat large batches for LoRAs.')
    wizard('optimizer', 'adamw8bit', 'AdamW with 8-bit state uses less memory.')
    wizard('learning_rate', 0.0002 if model_id == 'qwen_image21' else 0.0001, 'The usual LoRA learning rate; lower it if samples look burnt.')
    wizard('network_type', 'oft' if model_id == 'krea2' else 'lora', 'The wizard character recipe for this model.')
    rank = 8 if model_id == 'qwen_image21' else 16 if model_id in ('flux2_klein', 'flux2_klein4b', 'zimage', 'minimax_h3', 'minimax_h3_image', 'p_h3_image') else 32
    wizard('lora_rank', rank, 'The wizard character rank leaves room for face detail.')
    wizard('lora_alpha', float(rank), 'Alpha equal to rank keeps the adapter scale at 1.')
    wizard('lora_ema', 0.98, 'Weight averaging improves likeness and reduces checkpoint swings.')
    if model_id in ('minimax_h3', 'minimax_h3_image', 'p_h3_image', 'sdxl', 'flux2_klein', 'flux2_klein4b', 'zimage'):
        wizard('timestep_sampling', 'reference', 'Uses the model reference timestep schedule.')
    if model_id in ('krea2', 'qwen_image21', 'flux3_klein', 'moss_image', 'ideogram4', 'fibo2'):
        wizard('resolution_shift', True, 'Shifts the noise levels by image size, as the sampler does; 1024 px learns slowly without it.')
    if model_id in ('minimax_h3', 'minimax_h3_image', 'p_h3_image', 'qwen_image21'):
        wizard('training_adapter', 'auto', 'Use the model training adapter to prevent drift.')
    if model_id == 'flux3_klein':
        wizard('lr_scheduler', 'constant', 'BFL Klein recipe uses a constant schedule.')
        wizard('warmup_steps', 200, 'BFL Klein recipe uses 200 warm-up steps.')
    wizard('face_refs', folders, 'Every image with a face becomes a reference.')
    wizard('body_refs', folders if goals.focus == 'face_body' else [], 'Body references score the full body; face-only probes skip them.')
    sample_steps, sample_cfg = ARCH_FULL_SAMPLE.get(model_id, (20, 3.0))
    if model_id in ARCH_TURBO:
        turbo = ARCH_TURBO[model_id]
        wizard('turbo_lora', 'auto', 'Use the model turbo LoRA for fast previews.')
        sample_steps, sample_cfg = turbo['steps'], turbo['cfg']
    for key, value in (('sample_steps', sample_steps), ('sample_cfg', sample_cfg), ('probe_steps', sample_steps), ('probe_cfg', sample_cfg)):
        wizard(key, value, 'Use the model sampler settings for previews and probes.')
    wizard('sample_prompts', [f'portrait photo of {trigger}', f'full body photo of {trigger}'], 'Use the trigger in sample prompts.')
    wizard('probe_face_prompt', f'portrait photo of {trigger}', 'A plain headshot scores the face reliably.')
    wizard('probe_body_prompt', f'full body photo of {trigger}' if goals.focus == 'face_body' else '', 'An upright full-body shot scores the build.')
    apply(model.base, 'base')
    variant = next((v for v in model.variants if v.get('id') == goals.variant), None)
    if goals.variant and variant is None:
        warnings.append(f'Unknown variant: {goals.variant}')
    if variant:
        apply(variant.get('base', {}), 'base')
    deferred = []
    for rule in model.adjust:
        if _matches(rule.when, scan, goals):
            cadence_keys = {'save_every', 'sample_every', 'probe_every'}
            apply({k: v for k, v in rule.set.items() if k not in cadence_keys}, 'rule:' + rule.id)
            deferred.append((rule, {k: v for k, v in rule.set.items() if k in cadence_keys}))
            if rule.note:
                notes.append(rule.note)
    spec = model.steps
    crop = choices.get('crop_augment').value if 'crop_augment' in choices else 'none'
    factor = {'face': 1.6, 'face+bust': 2.2}.get(crop, 1)
    count = (scan.image_count * factor + len(scan.videos)) if scan else 0
    repeats = choices.get('repeats').value if 'repeats' in choices else 1
    batch = choices.get('batch_size').value if 'batch_size' in choices else 1
    quantum = spec['round_to']
    raw = spec['epochs'] * count * repeats / max(1, batch)
    steps = max(spec['min'], min(spec['max'], int(raw / quantum + 0.5) * quantum))
    put('steps', steps, spec['why'], spec['confidence'], spec['sources'], 'steps')
    # Ten checkpoints provide a useful first comparison; intervals snap to 50 steps.
    cadence = max(50 if steps <= 1500 else 100, int(steps / 10 / 50 + 0.5) * 50)
    for key in ('save_every', 'sample_every', 'probe_every'):
        if key not in choices:
            wizard(key, cadence if key != 'probe_every' or folders else 0,
                   'Save and evaluate about ten times during training when references are available.')
    for rule, settings in deferred:
        apply(settings, 'rule:' + rule.id)
        if settings:
            reference = next(iter(settings))
            for key in ('sample_every', 'probe_every'):
                if key not in settings and key != reference and (key != 'probe_every' or folders):
                    put(key, choices[reference].value, choices[reference].why, origin='rule:' + rule.id)
    if 'keep_checkpoints' not in choices:
        wizard('keep_checkpoints', 0 if steps // max(1, choices['save_every'].value) <= 10 else 8,
               'Keep all checkpoints when there are at most ten; otherwise keep the newest eight.')
    if scan and scan.folders:
        put('datasets', [{'path': str(p), 'repeats': repeats, 'trigger': trigger} for p in paths],
            'Use the selected dataset folders.', origin='dataset')
        put('trigger_word', trigger, 'Use the selected trigger.', origin='dataset')
        put('caption_ext', '.txt', 'Use text captions.', origin='dataset')
    put('model_arch', model_id, 'Selected model architecture.', origin='default')
    path = (variant or {}).get('model_path') or model.model_path
    if path:
        put('model_path', path, 'Selected model path.', origin='default')
    if goals.base_config:
        try:
            base = yaml.safe_load(goals.base_config.read_text(encoding='utf-8')) or {}
            if not isinstance(base, dict):
                raise ValueError('expected a mapping')
            if base.get('model_arch', 'auto') not in ('auto', model_id):
                warnings.append('Base config architecture differs from selected model; model settings were not copied.')
            else:
                for key in MODEL_KEYS:
                    if key in base and key != 'model_arch':
                        put(key, base[key], 'Preserved from base config.', origin='base_config')
        except (OSError, yaml.YAMLError, ValueError) as exc:
            warnings.append(f'Could not read base config: {exc}')
    if str(choices['network_type'].value).lower() == 'oft':
        choices.pop('lora_rank', None)
        choices.pop('lora_alpha', None)
        if model_id == 'qwen_image21':
            wizard('learning_rate', 5e-5, 'The wizard OFT learning rate.')
        if model_id == 'krea2':
            wizard('isolation', False, 'Krea OFT does not use isolation.')
    if str(choices['optimizer'].value).lower().startswith(('automagic', 'keel3')):
        choices.pop('learning_rate', None)
    if 'resolution_shift' in choices and choices['resolution_shift'].value and 'timestep_sampling' in choices \
            and str(choices['timestep_sampling'].value) == 'reference':
        wizard('resolution_shift', False, 'The reference noise levels already carry the model shift; this setting has no effect with them.')
    unknown = sorted(key for key in choices if key not in KEYS)
    if unknown:
        warnings.append('Not exported, RulZ AI Trainer has no such setting: ' + ', '.join(unknown) + '.')
    if any(c.confidence == 'GUESS' for c in choices.values()):
        warnings.append('Some settings are reasoned defaults without direct evidence.')
    return Recommendation(model_id, goals, choices, steps, warnings, notes, model.evaluate, model.facts, model.open_questions)
