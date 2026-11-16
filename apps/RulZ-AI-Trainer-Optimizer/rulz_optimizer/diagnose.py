"""Detect measurable symptoms and apply explicitly selected fixes."""
from dataclasses import dataclass
from statistics import median
from dataclasses import replace
import json
import math
import re
from .knowledge import Fix, Knowledge
from .recommend import apply_operation
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
