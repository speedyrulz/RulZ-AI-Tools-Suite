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
