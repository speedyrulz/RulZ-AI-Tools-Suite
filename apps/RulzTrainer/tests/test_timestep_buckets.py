"""timestep_buckets: stratified sigma draws (a deck of N buckets over the whole run) that keep the sampling curve's
shape, leave runs without it unchanged, and resume where a saved state left off."""
import math

import pytest
import torch

from rulztrainer import race as R
from rulztrainer.config import TrainConfig
from rulztrainer.trainer import TimestepStrata, draw_sigmas, make_strata, sample_sigmas


def buckets_of(u: torch.Tensor, n: int) -> list[int]:
    return [min(n - 1, int(x * n)) for x in u.tolist()]


def test_every_n_draws_take_one_value_from_each_bucket():
    strata, gen = TimestepStrata(8), torch.Generator().manual_seed(0)
    u = torch.cat([strata.uniform(k, gen) for k in (3, 5, 1, 7, 2, 6, 8)])  # 32 draws in uneven batches
    for i in range(0, 32, 8):
        assert sorted(buckets_of(u[i:i + 8], 8)) == list(range(8))
    assert ((u > 0) & (u < 1)).all()


def test_free_draws_are_unchanged_without_buckets():
    cfg = TrainConfig(timestep_sampling="logit_normal", logit_mean=0.3, logit_std=1.2)
    got = sample_sigmas(cfg, 64, torch.Generator().manual_seed(5))
    g = torch.Generator().manual_seed(5)
    want = torch.sigmoid(torch.randn(64, generator=g) * 1.2 + 0.3)
    assert torch.equal(got, (want * 1000).round().clamp(1, 1000) / 1000)


@pytest.mark.parametrize("mode", ["logit_normal", "sigmoid", "logsnr", "identity", "uniform", "content", "style"])
def test_buckets_keep_the_curve(mode):
    cfg = TrainConfig(timestep_sampling=mode, timestep_buckets=10, timestep_shift=2.0)
    free = sample_sigmas(cfg, 40000, torch.Generator().manual_seed(1), logit_shift=0.4)
    bucketed = sample_sigmas(cfg, 40000, torch.Generator().manual_seed(2), logit_shift=0.4, strata=make_strata(cfg))
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    assert torch.allclose(torch.quantile(bucketed, qs), torch.quantile(free, qs), atol=0.02)


def test_ten_draws_in_a_row_hold_one_from_each_tenth_of_the_curve():
    cfg = TrainConfig(timestep_sampling="logit_normal", timestep_buckets=10)
    deciles = torch.quantile(sample_sigmas(cfg, 40000, torch.Generator().manual_seed(1)), torch.linspace(0.1, 0.9, 9))
    strata, gen = make_strata(cfg), torch.Generator().manual_seed(4)
    for _ in range(5):
        tenths = torch.bucketize(sample_sigmas(cfg, 10, gen, strata=strata), deciles)
        assert len(set(tenths.tolist())) >= 9  # (the 1000-step grid can put one on a decile's edge)


def test_ten_bucketed_draws_cover_more_of_the_curve_than_free_ones():
    cfg = TrainConfig(timestep_sampling="uniform", timestep_buckets=10)
    strata, spread_b, spread_f = make_strata(cfg), [], []
    for seed in range(200):
        gen = torch.Generator().manual_seed(seed)
        spread_b.append(len(set(buckets_of(sample_sigmas(cfg, 10, gen, strata=strata) - 1e-9, 10))))
        spread_f.append(len(set(buckets_of(sample_sigmas(cfg, 10, gen) - 1e-9, 10))))
    assert min(spread_b) >= 9 and sum(spread_f) / len(spread_f) < 7.5  # free: about 6.5 of 10 tenths


def test_paired_sigmas_take_one_card_per_step():
    cfg = TrainConfig(optimizer="keel", keel_pair_sigma=True, timestep_buckets=4, timestep_sampling="uniform")
    strata, gen = make_strata(cfg), torch.Generator().manual_seed(0)
    firsts = []
    for _ in range(4):
        sigma, step_sigma = draw_sigmas(cfg, 3, gen, 0.0, 0, None, None, strata)
        again, _ = draw_sigmas(cfg, 3, gen, 0.0, 0, None, step_sigma, strata)  # the step's next micro-batch
        assert torch.equal(sigma, again) and (sigma == sigma[0]).all()
        firsts.append(float(sigma[0]))
    assert sorted(buckets_of(torch.tensor(firsts) - 1e-9, 4)) == [0, 1, 2, 3]


def test_a_saved_deck_resumes_where_it_left_off():
    a, gen = TimestepStrata(6), torch.Generator().manual_seed(3)
    a.uniform(4, gen)
    state, gen_state = a.state_dict(), gen.get_state()
    want = a.uniform(5, gen)
    b, gen2 = TimestepStrata(6), torch.Generator()
    b.load_state_dict(state)
    gen2.set_state(gen_state)
    assert torch.equal(b.uniform(5, gen2), want)
    c = TimestepStrata(8)
    c.load_state_dict(state)  # another bucket count: a fresh deck
    assert c.deck == []
    c.load_state_dict(None)


def test_settings():
    assert make_strata(TrainConfig()) is None
    assert make_strata(TrainConfig(timestep_buckets=10, timestep_sampling="reference")) is None
    with pytest.raises(ValueError, match="timestep_buckets"):
        TrainConfig(timestep_buckets=1).validate()
    with pytest.raises(ValueError, match="timestep_buckets"):
        TrainConfig(timestep_buckets=-2).validate()
    TrainConfig(timestep_buckets=2).validate()
    assert "timestep_buckets" in R.RACE_KEYS["Timesteps & loss"]
    assert not math.isnan(float(sample_sigmas(TrainConfig(timestep_buckets=2), 2, torch.Generator(),
                                              strata=TimestepStrata(2)).sum()))
