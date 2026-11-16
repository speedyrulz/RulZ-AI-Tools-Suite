from pathlib import Path
from rulz_optimizer.diagnose import detect, next_config, propose
from rulz_optimizer.runs import load_run
from rulz_optimizer.runs import ProbePoint
from rulz_optimizer.knowledge import Fix, Flaw, Setting
from dataclasses import replace


def test_diagnosis(knowledge):
    run = load_run(Path(__file__).parent / 'fixtures' / 'run')
    assert {'peaked_early', 'lost_framing', 'lr_collapsed'} <= {f.symptom for f in detect(run)}
    proposals = propose(knowledge, 'krea2', ['plastic_skin'])
    cfg = next_config(run, proposals, [0])
    assert cfg['learning_rate'] == 0.00005 and cfg['job_name'] == 'example_v2'


def test_metrics(knowledge):
    run = load_run(Path(__file__).parent / 'fixtures' / 'run')
    template = run.probes[0]
    run.probes = [replace(template, step=10, combined=10, face_seeds=[0, 40], copy=0.6),
                  replace(template, step=20, combined=30, body_missing=1),
                  replace(template, step=100, combined=10, body_missing=1)]
    symptoms = {f.symptom for f in detect(run)}
    assert {'peaked_early', 'likeness_unstable', 'lost_framing', 'memorized'} <= symptoms
    run.probes = [replace(template, step=i, combined=float(i), body_missing=0, face_seeds=[]) for i in (10, 15, 19)]
    assert {'no_likeness', 'slow_likeness'} <= {f.symptom for f in detect(run)}
    run.probes = [replace(p, scale='raw') for p in run.probes]
    assert 'raw_score_scale' in {f.symptom for f in detect(run)}
    assert 'no_likeness' not in {f.symptom for f in detect(run)}
    run.probes = []
    assert any(f.severity == 'info' and 'no likeness probes' in f.detail for f in detect(run))
    run.errors.append('ERROR loss nan')
    assert 'nan_or_diverged' in {f.symptom for f in detect(run)}


def test_proposal_alternatives_and_version(knowledge):
    run = load_run(Path(__file__).parent / 'fixtures' / 'run')
    setting = Setting('learning_rate', 0.1, 'why', 'DOC', [], 'mul')
    rank = Setting('lora_rank', 0.5, 'why', 'DOC', [], 'mul')
    knowledge.general_flaws.append(Flaw('plastic_skin', [], [Fix({'learning_rate': setting}, None, '', 'DOC', [])]))
    proposals = propose(knowledge, 'krea2', ['plastic_skin'])
    assert proposals[1].alternative_of == 0
    run.config.update(job_name='example_v2', lora_rank=15)
    proposals.append(type(proposals[0])('plastic_skin', Fix({'lora_rank': rank}, None, '', 'DOC', []), None))
    result = next_config(run, proposals, [0, 1, 2])
    assert result['job_name'] == 'example_v3' and isinstance(result['lora_rank'], int)
    assert result['learning_rate'] == 0.00005
