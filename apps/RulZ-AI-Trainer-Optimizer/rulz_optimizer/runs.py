"""Read incomplete or damaged trainer run folders without interrupting the UI."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import yaml


@dataclass
class ProbePoint:
    step: int
    face: float | None
    body: float | None
    combined: float | None
    face_seeds: list[float]
    body_seeds: list[float]
    best: float | None
    best_step: int | None
    body_missing: int | None
    drift_share: float | None
    kept_share: float | None
    copy: float | None
    face_over: float | None
    body_over: float | None
    scale: str = 'raw'
    racer: str = ''


@dataclass
class Checkpoint:
    path: Path
    step: int | None
    kind: str


@dataclass
class Run:
    path: Path
    job: str
    arch: str | None
    config: dict
    status: dict
    probes: list[ProbePoint]
    checkpoints: list[Checkpoint]
    probe_images: dict[int, list[Path]]
    samples: list[Path]
    lr_series: list[tuple[int, float]]
    loss_series: list[tuple[int, float]]
    errors: list[str]
    finished: bool
    race: dict | None


# A CUDA or system out-of-memory line in train.log (a traceback line carries no WARNING or ERROR word).
OOM_LINE = re.compile(r'out of memory|outofmemoryerror', re.I)
SAMPLE_NAME = re.compile(r'(?:.+_)?step(\d{6})_(\d+)\.[A-Za-z]+$')


def sample_steps(run) -> dict[int, list[Path]]:
    """The sample images of a run grouped by the step they were rendered at; names without a step go to -1."""
    groups: dict[int, list[Path]] = {}
    for image in run.samples:
        match = SAMPLE_NAME.match(image.name)
        groups.setdefault(int(match.group(1)) if match else -1, []).append(image)
    return dict(sorted(groups.items()))


def _read(path, parser, default):
    try:
        value = parser(path.read_text(encoding='utf-8', errors='replace'))
        return value if isinstance(value, type(default)) else default
    except (OSError, ValueError, yaml.YAMLError):
        return default


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _point(row):
    step = _number(row.get('step'))
    if step is None:
        return None
    def score(name):
        for key in (name + '_norm_ema', name + '_norm', name + '_ema', name):
            value = _number(row.get(key))
            if value is not None:
                return value
        return None
    def seeds(name):
        values = row.get(name + '_norm_seeds') or row.get(name + '_seeds') or []
        return [n for v in values if (n := _number(v)) is not None] if isinstance(values, list) else []
    check = row.get('prompt_check') if isinstance(row.get('prompt_check'), dict) else {}
    best_step = _number(row.get('best_step'))
    missing = _number(row.get('body_missing'))
    scale = 'norm' if any(_number(row.get(k)) is not None for k in ('combined_norm_ema', 'combined_norm')) else 'raw' if score('combined') is not None else 'missing'
    racer = str(row.get('racer') or '')
    if not racer:
        for image in row.get('images') or []:
            match = re.match(r'(.+)_step\d{6}_(?:face|body|check-)', Path(image).name)
            if match:
                racer = match.group(1)
                break
    return ProbePoint(int(step), score('face'), score('body'), score('combined'), seeds('face'), seeds('body'),
                      _number(row.get('best')), int(best_step) if best_step is not None else None,
                      int(missing) if missing is not None else None, _number(check.get('drift_share')),
                      _number(check.get('kept_share')), _number(row.get('copy')),
                      _number(row.get('face_over')), _number(row.get('body_over')), scale, racer)


def default_roots(trainer_dir: Path | None) -> list[Path]:
    roots = [Path(trainer_dir) / 'output'] if trainer_dir else []
    return roots + [Path.home() / '.rulz_vast' / 'runs', Path.home() / '.rulz_vast' / 'fleet' / 'runs']


def find_runs(roots: list[Path]) -> list[Path]:
    found = []
    for root in map(Path, roots):
        try:
            candidates = [root] + [p for p in root.iterdir() if p.is_dir()]
            for path in candidates:
                if (path / 'config.yaml').is_file() and ((path / 'status.json').is_file() or (path / 'likeness.jsonl').is_file()) and path not in found:
                    found.append(path)
        except OSError:
            continue
    return found


def load_run(path: Path) -> Run:
    path = Path(path)
    cfg = _read(path / 'config.yaml', yaml.safe_load, {})
    status = _read(path / 'status.json', json.loads, {})
    rows = []
    try:
        for line in (path / 'likeness.jsonl').read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except ValueError:
                continue
    except OSError:
        pass
    if not rows and isinstance(status.get('probes'), list):
        rows = [r for r in status['probes'] if isinstance(r, dict)]
    probes = [p for r in rows if (p := _point(r)) is not None]
    job = str(cfg.get('job_name') or path.name)
    checkpoints, images, samples, lr, loss, errors = [], {}, [], [], [], []
    try:
        for p in path.glob('*.safetensors'):
            stem = p.stem
            match = re.fullmatch(re.escape(job) + r'_(\d+)', stem)
            if match:
                checkpoints.append(Checkpoint(p, int(match.group(1)), 'periodic'))
            elif stem == job:
                checkpoints.append(Checkpoint(p, None, 'final'))
            elif stem in (job + '_best', job + '_best_face', job + '_best_body'):
                checkpoints.append(Checkpoint(p, None, stem[len(job) + 1:]))
            elif stem in (job + '_holdout_best', job + '_soup'):
                checkpoints.append(Checkpoint(p, None, stem[len(job) + 1:]))
            elif stem.startswith(job + '_') and stem.endswith(('_best', '_final', '_dropped')):
                checkpoints.append(Checkpoint(p, None, 'racer'))
        pattern = re.compile(r'^(?:.+_)?step(\d{6})_(?:face|body|check-[^_]+)_\d+\.(?:png|jpg|jpeg|webp)$', re.I)
        for p in (path / 'probes').glob('*'):
            match = pattern.match(p.name)
            if match:
                images.setdefault(int(match.group(1)), []).append(p)
        samples = sorted(p for p in (path / 'samples').glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'))
    except OSError:
        pass
    pattern = re.compile(r'step (\d+)/\d+ loss (\S+) avg \S+ lr (\S+)')
    seen_errors = set()
    try:
        with (path / 'train.log').open(encoding='utf-8', errors='replace') as log:
            for line in log:
                line = line.rstrip('\r\n')
                match = pattern.search(line)
                if match:
                    step = int(match.group(1))
                    try:
                        loss.append((step, float(match.group(2))))
                    except ValueError:
                        pass
                    rval = _number(match.group(3))
                    if rval is not None:
                        lr.append((step, rval))
                elif (re.search(r'\b(?:WARNING|ERROR)\b', line) or OOM_LINE.search(line)) and line not in seen_errors:
                    errors.append(line)
                    seen_errors.add(line)
    except OSError:
        pass
    race = status.get('race') if isinstance(status.get('race'), dict) else _read(path / 'race.json', json.loads, {})
    return Run(path, job, cfg.get('model_arch'), cfg, status, probes, checkpoints, images, samples, lr, loss,
               errors, bool(status.get('done')), race or None)


def best_points(run: Run) -> dict[str, ProbePoint | None]:
    return {key: max((p for p in run.probes if getattr(p, key) is not None),
                     key=lambda p: getattr(p, key), default=None) for key in ('face', 'body', 'combined')}


def recorded_best_point(run: Run) -> ProbePoint | None:
    """Resolve the trainer's recorded best checkpoint, falling back to observed scores."""
    race = run.race or {}
    winner = race.get('winner')
    if not winner and any(p.racer for p in run.probes):
        winner = run.probes[-1].racer
    if not any(p.racer for p in run.probes):
        winner = None
    owners = {winner, ''} if winner and race.get('last') == winner else {winner} if winner else {''}
    candidates = [p for p in run.probes if p.racer in owners]
    race_steps = race.get('best_step') if isinstance(race.get('best_step'), dict) else {}
    likeness = run.status.get('likeness') if isinstance(run.status.get('likeness'), dict) else {}
    step = race_steps.get(winner) if winner else None
    if step is None and candidates:
        step = candidates[-1].best_step
    if step is None:
        step = likeness.get('best_step')
    if step is not None:
        for point in reversed(candidates):
            if point.step == step:
                return point
    norm = [p for p in candidates if p.scale == 'norm' and p.combined is not None]
    return max(norm, key=lambda p: p.combined, default=None)
