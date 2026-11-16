from pathlib import Path
from rulz_optimizer.runs import load_run, best_points


def test_run():
    run = load_run(Path(__file__).parent / 'fixtures' / 'run')
    assert len(run.probes) == 2 and run.finished
    assert best_points(run)['face'].step == 100
    assert run.lr_series == [(100, 0.000001)]


def test_missing(tmp_path):
    assert not load_run(tmp_path / 'missing').probes


def test_corrupt_files(tmp_path):
    (tmp_path / 'likeness.jsonl').write_text('garbage\n{"step": 2, "combined_norm": 30}\n{}\n', encoding='utf-8')
    (tmp_path / 'train.log').write_text('step 1/3 loss garbage avg 1 lr wat\nstep 2/3 loss 0.5 avg 1 lr 1e-4\n', encoding='utf-8')
    run = load_run(tmp_path)
    assert run.status == {} and len(run.probes) == 1 and run.probes[0].scale == 'norm'
    assert run.lr_series == [(2, 0.0001)]
