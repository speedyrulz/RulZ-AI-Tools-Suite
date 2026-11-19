"""Detect measurable symptoms and apply explicitly selected fixes."""
from dataclasses import dataclass
from statistics import median
from dataclasses import replace
import json
import math
import re
from .knowledge import Fix, Knowledge
from .recommend import Goals, apply_operation, recommend
from .trainer_keys import DEFAULTS
from .runs import OOM_LINE, Run, recorded_best_point

# Normalized scores below 20 indicate little likeness.
NO_LIKENESS = 20.0
# A peak before 70% of the run followed by an 8-point drop is early.
EARLY_FRACTION = 0.7
PEAK_DROP = 8.0
# A peak within the last 15% of probes can indicate ongoing improvement.
RECENT_FRACTION = 0.15
# Normal seed swings can reach 20 points; over 35 is concerning.
UNSTABLE_SPREAD = 35.0
# Median jumps above 15 normalized points indicate instability.
UNSTABLE_JUMP = 15.0
# Fewer than 60% of checks keeping framing indicates a problem.
KEPT_SHARE = 0.6
# Prompt drift affecting at least 30% of checks is substantial.
DRIFT_SHARE = 0.3
# Overfit ratios above 1 signal memorization.
MEMORIZED_LIMIT = 1.0
# Copy scores of 0.6–0.7 marked memorized rows in our runs.
COPY_LIMIT = 0.6
# An automagic rate below 1e-5 for a majority of recorded steps is collapsed.
COLLAPSED_LR = 1e-5
# Losses above 100 are treated as divergent.
DIVERGED_LOSS = 100.0
# A learning rate at alpha = rank more than 3x above or below the recommendation draws a warning.
ALPHA_LR_WARN_FACTOR = 3.0
# Optimizers whose learning_rate field is not the rate the weights get. Mirrors config.py in RulZ AI Trainer
# (AUTOMAGIC + hull). A copy, because the trainer is not importable from this app.
SELF_SETTING_RATE = ('automagic3', 'automagic4', 'keel', 'keel_fb', 'keel3', 'keel3_fb', 'hull')


@dataclass
class Finding:
    symptom: str
    detail: str
    evidence: dict
    severity: str


@dataclass
class Proposal:
    symptom: str
    fix: Fix
    source_model: str | None
    alternative_of: int | None = None
    step_round_to: int | float | None = None


def detect(run: Run) -> list[Finding]:
    racers = {p.racer for p in run.probes if p.racer}
    if racers:
        winner = (run.race or {}).get('winner')
        selected = [winner] if winner in racers else sorted(racers)
        findings = []
        for racer in selected:
            subset = [replace(p, racer='') for p in run.probes if p.racer == racer or (winner == racer and (run.race or {}).get('last') == winner and not p.racer)]
            for finding in detect(replace(run, probes=subset, race=None)):
                finding.detail = f'{racer}: {finding.detail}'
                findings.append(finding)
        return findings
    findings = []
    def add(symptom, detail, evidence, severity='warn'):
        findings.append(Finding(symptom, detail, evidence, severity))
    if not run.probes:
        add('no_likeness_probes', 'no likeness probes; turn on probe_every with face_refs/body_refs', {}, 'info')
    points = [p for p in run.probes if p.scale == 'norm' and p.combined is not None]
    if any(p.scale == 'raw' for p in run.probes):
        add('raw_score_scale', 'Raw likeness scores: normalized score thresholds were skipped.', {}, 'info')
    best = recorded_best_point(run)
    if points and best and best.scale == 'norm' and best.combined is not None:
        last = points[-1]
        if best.combined < NO_LIKENESS:
            add('no_likeness', 'Likeness remained low.', {'best': best.combined})
        if len(points) >= 3 and best in points and points.index(best) >= (len(points) - 1) * (1 - RECENT_FRACTION) and last.combined > points[-3].combined:
            add('slow_likeness', 'Likeness was still rising at the last probe.', {'last': last.combined})
        if best.step < run.probes[-1].step * EARLY_FRACTION and last.combined <= best.combined - PEAK_DROP:
            add('peaked_early', 'Likeness peaked well before the final probe.', {'best_step': best.step, 'last_step': last.step})
    jumps = [abs(a.combined - b.combined) for a, b in zip(points, points[1:])]
    if any((len(s) > 1 and max(s) - min(s) > UNSTABLE_SPREAD) for p in points for s in (p.face_seeds, p.body_seeds)) or (jumps and median(jumps) > UNSTABLE_JUMP):
        add('likeness_unstable', 'Probe scores vary sharply.', {})
    if sum((p.body_missing or 0) > 0 for p in run.probes) >= 2 or (best and ((best.kept_share is not None and best.kept_share < KEPT_SHARE) or (best.drift_share is not None and best.drift_share >= DRIFT_SHARE))):
        add('lost_framing', 'Body probes were missing or prompt checks drifted.', {})
    if any((p.copy is not None and p.copy >= COPY_LIMIT) or any(v is not None and v > MEMORIZED_LIMIT for v in (p.face_over, p.body_over)) for p in run.probes):
        add('memorized', 'A copy or overfit score exceeded its limit.', {})
    optimizer = str(run.config.get('optimizer', '')).lower()
    if optimizer.startswith(('automagic', 'keel')) and run.lr_series and sum(v < COLLAPSED_LR for _, v in run.lr_series) > len(run.lr_series) / 2:
        add('lr_collapsed', 'Automagic learning rate was below 1e-5 for most logged steps.', {})
    if any(not math.isfinite(v) or v > DIVERGED_LOSS for _, v in run.loss_series) or any(any(s in line.lower() for s in ('nan', 'diverg', 'non-finite', 'not finite')) for line in run.errors) or (run.status.get('skipped_steps') or 0) > 0:
        add('nan_or_diverged', 'The log indicates a non-finite or divergent loss.', {}, 'bad')
    if any(OOM_LINE.search(line) for line in run.errors) or OOM_LINE.search(json.dumps(run.status, default=str)):
        add('out_of_memory', 'The log or the run status reports an out-of-memory error.', {}, 'bad')
    return findings


def _float(value):
    try:
        return None if isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None


def lr_at_alpha_rank(config: dict) -> float | None:
    """The learning rate a run config acts like at alpha = rank (RulZ AI Trainer's convention, scale 1):
    learning_rate x alpha / rank. Mirrors lr_at_alpha_rank in the trainer's config.py. None where it does not apply:
    a fine-tune, LoKr / OFT, a missing rank, and an optimizer that sets its own rate."""
    def get(key):
        return config[key] if config.get(key) is not None else DEFAULTS.get(key)
    rank, alpha, lr = _float(get('lora_rank')), _float(get('lora_alpha')), _float(get('learning_rate'))
    if str(get('network_type')).lower() != 'lora' or get('finetune') or not rank or rank <= 0 or alpha is None or lr is None:
        return None
    name, lr = _trained_rate(config)
    if name in SELF_SETTING_RATE or lr is None:
        return None
    return lr * alpha / rank


def _trained_rate(config: dict) -> tuple[str, float | None]:
    """The optimizer and learning rate a run trains with, as the trainer's effective_optimizer: isolation (DOP) trains
    an Automagic or Keel pick with AdamW, at the learning rate when it is AdamW-sized (above 1e-6, at most 1e-3),
    else 2e-4."""
    def get(key):
        return config[key] if config.get(key) is not None else DEFAULTS.get(key)
    name, lr = str(get('optimizer')).lower(), _float(get('learning_rate'))
    if get('isolation') and (_float(get('dop_weight')) or 0) > 0 and name in SELF_SETTING_RATE[:-1]:
        name, lr = 'adamw', (lr if lr is not None and 1e-6 < lr <= 1e-3 else 2e-4)
    return name, lr


def alpha_findings(run: Run, k: Knowledge | None) -> list[Finding]:
    """When a LoRA run's alpha differs from its rank, show the learning rate at alpha = rank, and warn when it is more
    than 3x above or below this model's recommendation. Empty when alpha equals rank or the rate does not apply."""
    config = run.config
    equivalent = lr_at_alpha_rank(config)
    rank = _float(config.get('lora_rank', DEFAULTS['lora_rank']))
    alpha = _float(config.get('lora_alpha', DEFAULTS['lora_alpha']))
    if equivalent is None or not rank or alpha is None or abs(alpha - rank) <= 1e-9:
        return []
    name, rate = _trained_rate(config)  # (isolation trains an Automagic / Keel pick with AdamW at its own rate)
    swapped = name != str(config.get('optimizer', DEFAULTS.get('optimizer'))).lower()
    text = (f'alpha {alpha:g} at rank {rank:g} scales each update by {alpha / rank:.3g}: '
            + ("AdamW's learning rate under isolation " if swapped else 'learning rate ')
            + f'{rate:.1e} acts like {equivalent:.1e} at alpha = rank.')
    evidence = {'lr_at_alpha_rank': equivalent, 'alpha': alpha, 'rank': rank}
    model = run.arch or config.get('model_arch')
    recommended = None
    if k is not None and model in k.models:
        try:
            choice = recommend(k, model, None, Goals()).choices.get('learning_rate')
            recommended = _float(choice.value) if choice else None
        except Exception:  # noqa: BLE001  a knowledge problem must not hide the run's findings
            recommended = None
    if recommended and recommended > 0 and equivalent > 0:
        evidence['recommended'] = recommended
        ratio = equivalent / recommended
        if ratio > ALPHA_LR_WARN_FACTOR:
            return [Finding('alpha_lr_off', f'{text} That is {ratio:.1f}x above the recommended {recommended:.1e}.', evidence, 'warn')]
        if ratio < 1 / ALPHA_LR_WARN_FACTOR:
            return [Finding('alpha_lr_off', f'{text} That is {1 / ratio:.1f}x below the recommended {recommended:.1e}.', evidence, 'warn')]
        text += f' Recommended {recommended:.1e}: within 3x.'
    return [Finding('alpha_lr', text, evidence, 'info')]


def propose(k: Knowledge, model_id: str, symptoms: list[str]) -> list[Proposal]:
    result = []
    first = {}
    for flaws, source in ((k.models[model_id].flaws, model_id), (k.general_flaws, None)):
        for flaw in flaws:
            if flaw.symptom in symptoms:
                for fix in flaw.fixes:
                    overlap = next((first[key] for key in fix.set if key in first), None)
                    result.append(Proposal(flaw.symptom, fix, source, overlap,
                                           k.models[model_id].steps.get('round_to')))
                    for key in fix.set:
                        first.setdefault(key, len(result) - 1)
    return result


def next_config(run: Run, proposals: list[Proposal], chosen: list[int]) -> dict:
    config = dict(run.config)
    applied = set()
    for index in chosen:
        if not 0 <= index < len(proposals):
            continue
        for key, setting in proposals[index].fix.set.items():
            if key not in applied:
                old = config.get(key)
                config[key] = apply_operation(old, setting.value, setting.op, key)
                quantum = proposals[index].step_round_to
                if key == 'steps' and setting.op and isinstance(quantum, (int, float)) and quantum > 0:
                    config[key] = int(math.floor(config[key] / quantum + 0.5) * quantum)
                applied.add(key)
    name = str(config.get('job_name') or run.job)
    match = re.search(r'_v(\d+)$', name)
    config['job_name'] = name[:match.start()] + f'_v{int(match.group(1)) + 1}' if match else name + '_v2'
    return config
