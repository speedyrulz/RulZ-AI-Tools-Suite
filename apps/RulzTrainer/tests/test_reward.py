"""Likeness reward fine-tuning (reward.py): the stochastic sampler without noise is each family's own sampler, its
steps' log-probabilities and KL, group advantages, a toy policy that learns what the reward prefers, the default
prompts and the config checks."""
import math
import random
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import reward as RW  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.krea2.model import Krea2Transformer  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402


@pytest.mark.parametrize("family", ["moss_image", "krea2"])
def test_without_noise_the_sampler_is_the_familys_own(family):
    torch.manual_seed(0)
    if family == "moss_image":
        model = MOSSImageTransformer(dim=64, ffn_dim=128, num_heads=2, num_layers=4, text_dim=32, freq_dim=16,
                                     latent_channels=128)
        emb = torch.randn(1, 5, 32)
    else:
        model = Krea2Transformer(in_channels=64, num_layers=2, attention_head_dim=16, num_attention_heads=4,
                                 num_key_value_heads=2, intermediate_size=96, timestep_embed_dim=32,
                                 text_hidden_dim=40, num_text_layers=3, text_num_attention_heads=4,
                                 text_num_key_value_heads=4, text_intermediate_size=56, num_layerwise_text_blocks=1,
                                 num_refiner_text_blocks=1, axes_dims_rope=(4, 6, 6))
        emb = torch.randn(1, 5, 3, 40)
    model = model.to(torch.bfloat16).eval()
    arch = archs.arch_by_name(family)
    mask = torch.ones(1, 5, dtype=torch.bool)
    ref = arch.sample(model, emb, mask, None, None, width=64, height=48, steps=6, cfg=1.0, shift=3.0, seed=11,
                      device="cpu")
    sigmas = arch.sample_sigmas(model, 6, 48, 64, 3.0)
    xs, stds = RW.sample_group(model, sigmas, emb, mask, None, None, cfg_scale=1.0, noise_level=0.0, seeds=[11],
                               shape=(model.latent_channels, 3, 4), device="cpu")
    assert len(xs) == len(sigmas) and all(sd == 0 for sd in stds)
    assert torch.allclose(xs[-1], ref, atol=1e-4)


def test_sde_step_log_prob_and_kl():
    x, v = torch.randn(2, 4, 3, 3), torch.randn(2, 4, 3, 3)
    euler, std0 = RW.sde_step_stats(x, v, 0.6, 0.4, 0.9, 0.0)
    assert std0 == 0 and torch.allclose(euler, x + (0.4 - 0.6) * v)  # no noise: the Euler step
    mean, std = RW.sde_step_stats(x, v, 0.6, 0.4, 0.9, 0.7)
    expected_std = 0.7 * math.sqrt(0.6 / 0.4) * math.sqrt(0.2)
    assert abs(std - expected_std) < 1e-9
    _, std_first = RW.sde_step_stats(x, v, 1.0, 0.8, 0.9, 0.7)  # sigma 1: the cap stands in for 1 - sigma
    assert abs(std_first - 0.7 * math.sqrt(1.0 / 0.1) * math.sqrt(0.2)) < 1e-9
    lp_at_mean = RW.log_prob(mean, mean, std)
    lp_away = RW.log_prob(mean + std, mean, std)
    assert torch.allclose(lp_at_mean - lp_away, torch.full((2,), 0.5), atol=1e-5)  # one std away: -1/2 nats
    assert torch.allclose(RW.kl_to(mean, mean, std), torch.zeros(2))
    assert torch.allclose(RW.kl_to(mean + std, mean, std), torch.full((2,), 0.5), atol=1e-5)


def test_advantages_are_relative_to_the_group():
    a = RW.advantages([0.2, 0.4, 0.6, 0.8])
    assert abs(float(a.mean())) < 1e-6 and float(a[3]) > 0 > float(a[0])
    assert RW.advantages([0.5, 0.5, 0.5]) is None  # all alike: nothing to learn


class _Toy(nn.Module):
    """A 'model' whose velocity is a learned constant per channel: the policy's only knob."""

    latent_channels = 2

    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(2))

    def forward(self, x, t, e, m, **kw):
        return (x.float() * 0 + self.b.view(1, -1, 1, 1)).to(x.dtype)


def test_the_policy_gradient_moves_toward_what_the_reward_prefers():
    torch.manual_seed(0)
    model = _Toy()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    sigmas = torch.linspace(1, 0, 6)
    sig = [float(s) for s in sigmas]
    rng = random.Random(0)

    def reward(x):  # the final image's channel 0 should be high, channel 1 low
        return float(x[0].mean() - x[1].mean())

    for it in range(30):
        seeds = [it * 10 + k for k in range(4)]
        xs, stds = RW.sample_group(model, sigmas, None, None, None, None, cfg_scale=1.0, noise_level=0.7, seeds=seeds,
                                   shape=(2, 2, 2), device="cpu")
        rewards = [reward(xs[-1][k]) for k in range(4)]
        adv = RW.advantages(rewards)
        if adv is None:
            continue
        opt.zero_grad()
        for j in rng.sample([j for j, sd in enumerate(stds) if sd > 0], 3):
            v = RW.velocity(model, xs[j], sig[j], None, None, None, None, 1.0)
            mean, std = RW.sde_step_stats(xs[j], v, sig[j], sig[j + 1], sig[1], 0.7)
            lp = RW.log_prob(xs[j + 1], mean, std)
            (-(adv * torch.exp(lp - lp.detach()))).mean().backward()
        opt.step()
    # x0 = x_T - v (integrated over sigma 1 -> 0): a higher reward wants v[0] down and v[1] up
    b = model.b.detach()
    assert float(b[0]) < -0.3 and float(b[1]) > 0.3


def test_default_prompts_follow_the_score_weights():
    both = RW.default_prompts("zkrea", 0.7, 0.3)
    assert len(both) == 8 and all("zkrea" in p for p in both)
    face = RW.default_prompts("zkrea", 1.0, 0.0)
    assert not any(RW.is_full_body(p) for p in face) and any("close-up" in p for p in face)
    body = RW.default_prompts("", 0.0, 1.0)
    assert not any("close-up" in p for p in body) and all("the person" in p for p in body)


def test_config_checks():
    TrainConfig(reward_steps=50, face_refs=["x"]).validate()
    TrainConfig(reward_steps=50, steps=0, resume_from="a.safetensors", body_refs=["x"]).validate()
    for bad, match in ((dict(reward_steps=50), "references"),
                       (dict(reward_steps=50, face_refs=["x"], race=True,
                             race_configs=[{"name": "a"}, {"name": "b"}]), "race"),
                       (dict(reward_steps=50, face_refs=["x"], reward_group=1), "reward_group"),
                       (dict(reward_steps=50, face_refs=["x"], steps=0), "resume_from")):
        with pytest.raises(ValueError, match=match):
            TrainConfig(**bad).validate()


def test_the_reward_stage_closes_with_a_likeness_probe():
    from rulztrainer.trainer import Trainer

    class Status:
        def update(self, **kw):
            pass

    calls = []
    t = Trainer.__new__(Trainer)
    t.status = Status()
    t.run_probe = lambda model, step, opt, sched, injected: calls.append(step) or False
    t.controller = object()
    Trainer._reward_final_probe(t, None, 1450, None, None, {})
    assert calls == [1450]
    t.controller = None  # probes off: nothing to probe with
    Trainer._reward_final_probe(t, None, 1500, None, None, {})
    assert calls == [1450]


def test_a_part_far_below_a_stranger_is_capped():
    assert RW.clip_part(0.4) == (0.4, False)
    assert RW.clip_part(-2.3) == (RW.PART_MIN, True)
    assert RW.clip_part(3.0) == (RW.PART_MAX, False)


def test_clothed_bodies_against_unclothed_references_leave_the_ranking_to_the_face():
    """my_lora18's log: bodies 20+ raw points below the stranger level on clothed prompts made rewards of -2 and
    worse, so the body, not the face, picked each group's winner."""
    import logging

    from rulztrainer.config import TrainConfig
    from rulztrainer.trainer import Trainer

    class Face:
        def __init__(self, pcts):
            self.pcts = iter(pcts)

        def score(self, path, *_):
            return type("S", (), {"percent": next(self.pcts)})()

    class Body:
        def __init__(self, pcts):
            self.pcts = iter(pcts)

        def score(self, path):
            return next(self.pcts), 8

    class Scale:
        normalize = True

        def scale(self, kind):
            return (10.0, 80.0) if kind == "face" else (78.0, 89.0)  # the body's scale is ~6x narrower

    t = Trainer.__new__(Trainer)
    t.cfg = TrainConfig(face_weight=0.5, body_weight=0.5, reward_group=4, reward_prompts=["photo of x in a coat"])
    faces = [70.0, 60.0, 50.0, 40.0]  # the first picture looks most like the person
    bodies = [52.0, 56.0, 50.0, 54.0]  # all far below a stranger (78): coats against unclothed references
    t.face_scorer, t.body_scorer = Face(faces), Body(bodies)
    t._reward_body_seen = t._reward_body_low = 0
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logging.getLogger("rulztrainer").addHandler(handler)
    try:
        rewards = [t._reward_of(f"p{k}.png", "photo of x in a coat", Scale()) for k in range(4)]
    finally:
        logging.getLogger("rulztrainer").removeHandler(handler)
    assert rewards == sorted(rewards, reverse=True)  # the face decides
    assert min(rewards) > -1.0
    assert any("far below a stranger" in r.getMessage() for r in records)


# ------------------------------------------------------------ reward_probe_every: own cadence and change readout
def test_reward_probe_every_config_checks():
    TrainConfig(reward_probe_every=0).validate()
    TrainConfig(reward_steps=50, face_refs=["x"], reward_probe_every=1).validate()
    TrainConfig(reward_steps=50, steps=0, resume_from="a.safetensors", body_refs=["x"], probe_every=0,
                reward_probe_every=5).validate()
    with pytest.raises(ValueError, match="reward_probe_every"):
        TrainConfig(reward_probe_every=-1).validate()
    with pytest.raises(ValueError, match="reward_probe_every"):  # probes need references, like probe_every
        TrainConfig(reward_probe_every=2).validate()


def _picture(path, value, size=(8, 8)):
    from PIL import Image

    Image.new("RGB", size, (value, value, value)).save(path)
    return str(path)


def test_likeness_change_is_now_minus_base_per_part():
    base = {"combined": 40.0, "face": 50.0, "body": None}
    now = {"combined": 42.5, "face": 49.0, "body": 30.0}
    out = RW.likeness_change(base, now)
    assert out["combined"] == pytest.approx(2.5) and out["face"] == pytest.approx(-1.0) and out["body"] is None
    assert RW.likeness_change({"combined": None}, {"combined": 3.0})["combined"] is None


def test_image_change_pairs_by_prompt_and_seed_and_skips_other_sizes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    base = [_picture(a / "step000000_face_0.png", 100), _picture(a / "step000000_body_0.png", 0),
            _picture(a / "step000000_face_1.png", 10, size=(8, 8))]
    # face_0 differs by 51 of 255 (20 %), body_0 by 0, face_1 has another size (skipped), face_2 has no partner
    now = [_picture(b / "step000010_face_0.png", 151), _picture(b / "step000010_body_0.png", 0),
           _picture(b / "step000010_face_1.png", 200, size=(4, 4)), _picture(b / "step000010_face_2.png", 9)]
    assert RW.picture_key(base[0]) == RW.picture_key(now[0]) == "step_face_0.png"
    assert RW.image_change_pct(base, now) == pytest.approx(10.0)  # (20 + 0) / 2
    assert RW.image_change_pct(base, [now[2], now[3]]) is None  # nothing pairs
    assert RW.image_change_pct(base, [str(tmp_path / "gone" / "step000010_face_0.png")]) is None


def test_the_change_line_reads_like_the_log_example():
    line = RW.describe_change(120, {"combined": 2.3, "face": 3.1, "body": -0.4}, 4.2)
    assert line == ("Reward probe step 120: likeness +2.3 points since the stage began (face +3.1, body -0.4), "
                    "pictures changed 4.2%")
    assert "n/a" in RW.describe_change(1, {"combined": None}, None)


class _Rec:
    def __init__(self, combined, face, body, images):
        self.combined, self.face_norm, self.body_norm, self.images = combined, face, body, images


def _stage_trainer(tmp_path, monkeypatch, stops, **cfg_kw):
    """A Trainer with just enough around it to run _reward_stage on a toy model: the renders are faked (every picture
    scores alike, so no update runs) and run_probe records its steps and returns its next scripted stop."""
    from types import SimpleNamespace

    from rulztrainer import trainer as TR

    cfg = TrainConfig(reward_steps=6, face_refs=["x"], reward_prompts=["a photo of t"], reward_group=2,
                      probe_steps=2, save_every=0, **cfg_kw)
    t = TR.Trainer.__new__(TR.Trainer)
    t.cfg, t.ema, t.dp, t.devices, t.device = cfg, None, None, ["cpu"], "cpu"
    t.face_scorer, t.body_scorer, t.probe_worker = object(), None, None
    t.stop_file, t.job_dir = tmp_path / "STOP", tmp_path
    t.last_probe_rec = t.last_probe_step = None
    data: dict = {}
    t.status = SimpleNamespace(data=data, update=lambda **kw: data.update(kw))
    t.controller = SimpleNamespace(normalize=False, n_probes=1, state=lambda: {})
    t.arch = SimpleNamespace(name="x",
                             pad_embeds=lambda rows: (torch.zeros(len(rows), 2, 2), torch.ones(len(rows), 2)),
                             sample_sigmas=lambda *a: torch.tensor([1.0, 0.5, 0.0]),
                             init_noise_scale=lambda s: 1.0,
                             decode_latents=lambda vae, x: torch.zeros(1, 3, 8, 8))
    t._likeness_scale = lambda: None
    t._home_vae, t._park_vae = lambda: None, lambda: None
    t.negative_prompt = lambda: "neg"
    t._reward_of = lambda path, prompt, scale: 0.5
    t.save = lambda injected, step, final=False: tmp_path / f"{step}.safetensors"
    calls = []
    scripted = list(stops)

    def run_probe(model, step, opt, sched, injected):
        calls.append(step)
        shot = _picture(tmp_path / f"step{step:06d}_face_0.png", min(255, 5 * len(calls)))
        t.last_probe_rec, t.last_probe_step = _Rec(40.0 + 2 * len(calls), 10.0, None, [shot]), step
        return scripted.pop(0) if scripted else False

    t.run_probe = run_probe
    toy = nn.Module()
    toy.layers = nn.ModuleList()
    toy.latent_channels = 2
    monkeypatch.setattr(TR, "load_text_embeds", lambda cfg, prompt: torch.zeros(1, 2))
    monkeypatch.setattr(TR.lora_mod, "lora_parameters", lambda injected: [nn.Parameter(torch.zeros(1))])
    monkeypatch.setattr(RW, "sample_group", lambda *a, **k: ([torch.zeros(2, 2, 1, 1)] * 3, [0.0, 0.0]))
    return t, toy, calls


def test_reward_probes_follow_their_own_cadence_and_only_watch(tmp_path, monkeypatch):
    t, toy, calls = _stage_trainer(tmp_path, monkeypatch, [False, True, True, True], reward_probe_every=2,
                                   probe_every=50)
    t._reward_stage(toy, {}, 100)
    # a baseline probe at the stage start (nothing taken at step 100), then after reward steps 2, 4 and 6 (the stop
    # return of the probes at 102 and 104 is ignored: the stage runs on to step 106)
    assert calls == [100, 102, 104, 106]
    rows = t.status.data["reward_change"]
    assert [r[0] for r in rows] == [102, 104, 106]
    assert rows[0][1] == pytest.approx(2.0) and rows[2][1] == pytest.approx(6.0)  # combined 42 vs 40, 46 vs 40
    assert rows[0][2] is not None  # the pictures differ from the baseline's


def test_reward_probes_reuse_the_probe_taken_at_the_stage_start(tmp_path, monkeypatch):
    t, toy, calls = _stage_trainer(tmp_path, monkeypatch, [], reward_probe_every=3, probe_every=0)
    shot = _picture(tmp_path / "step000100_face_0.png", 0)
    t.last_probe_rec, t.last_probe_step = _Rec(30.0, 10.0, None, [shot]), 100
    t._reward_stage(toy, {}, 100)
    assert calls == [103, 106]  # no baseline probe: the last one scored these weights
    assert [r[0] for r in t.status.data["reward_change"]] == [103, 106]


def test_without_reward_probe_every_a_probe_still_ends_the_stage(tmp_path, monkeypatch):
    t, toy, calls = _stage_trainer(tmp_path, monkeypatch, [False, True], reward_probe_every=0, probe_every=2)
    t._reward_stage(toy, {}, 100)
    assert calls == [102, 104]  # the second probe's stop ended the stage at step 4; no baseline, no readout
    assert "reward_change" not in t.status.data


def test_a_reward_probe_every_run_gets_a_controller_without_probe_every():
    from rulztrainer.trainer import Trainer

    t = Trainer.__new__(Trainer)
    t.cfg = TrainConfig(reward_steps=10, face_refs=["x"], probe_every=0, reward_probe_every=1)
    assert t._reward_probes()
    t.cfg = TrainConfig(reward_steps=10, face_refs=["x"], probe_every=0)
    assert not t._reward_probes() and t._make_controller([]) is None
    t.cfg = TrainConfig(reward_steps=0, face_refs=["x"], probe_every=0, reward_probe_every=1)
    assert not t._reward_probes()
