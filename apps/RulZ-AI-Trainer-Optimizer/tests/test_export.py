import yaml
from rulz_optimizer.export import build_config, to_yaml
from rulz_optimizer.recommend import Goals, recommend


def test_export(knowledge):
    rec = recommend(knowledge, 'krea2', None, Goals())
    cfg = build_config(rec, 'job', {'model_path': 'custom'}, {'unknown': 1})
    assert cfg['model_path'] == 'custom' and 'unknown' not in cfg
    cfg['automagic_min_lr'] = 1e-5
    loaded = yaml.safe_load(to_yaml(cfg, 'made by optimizer'))
    assert loaded['learning_rate'] == 0.0001 and isinstance(loaded['learning_rate'], float)
    assert loaded['automagic_min_lr'] == 1e-5 and isinstance(loaded['automagic_min_lr'], float)
