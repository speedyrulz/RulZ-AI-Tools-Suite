"""resolution_shift: the noise levels of every timestep_sampling mode but reference shifted by the family's own image-size
shift (exp(mu), mu from Arch.resolution_mu), composed with timestep_shift, off by default."""
import logging
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import capabilities as caps  # noqa: E402
from rulztrainer import clean_share as CS  # noqa: E402
from rulztrainer import trainer as T  # noqa: E402
from rulztrainer.archs import arch_by_name, get_arch  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


def draw(n=4000, tokens=4096, seed=3, strata=None, logit_shift=0.0, **kw):
    cfg = TrainConfig(model_arch=kw.pop("arch", "moss_image"), **kw)
    gen = torch.Generator("cpu").manual_seed(seed)
    return T.sample_sigmas(cfg, n, gen, logit_shift, tokens=tokens, arch=get_arch(cfg), strata=strata)


def shifted(s, shift):
    return shift * s / (1 + (shift - 1) * s)


def test_default_is_off_and_identical():
    assert TrainConfig().resolution_shift is False
    for mode in ("logit_normal", "uniform", "sigmoid", "identity", "content", "style", "logsnr", "shift"):
        a = draw(timestep_sampling=mode)
        b = draw(timestep_sampling=mode, resolution_shift=False)
        assert torch.equal(a, b)
        assert not torch.equal(a, draw(timestep_sampling=mode, resolution_shift=True)), mode


def test_default_formula_known_values():
    arch = arch_by_name("moss_image")
    assert arch.resolution_mu(256) == pytest.approx(0.5)
    assert arch.resolution_mu(4096) == pytest.approx(1.15)
    assert arch.resolution_mu(1024) == pytest.approx(0.5 + 0.65 * 768 / 3840)
    shift = math.exp(1.15)
    assert shift == pytest.approx(3.158, abs=1e-3)
    assert shifted(torch.tensor(0.5), shift).item() == pytest.approx(0.7593, abs=1e-3)
    # the median of the default logit-normal (sigma 0.5) lands at the shifted value, on the 1000-step grid
    s = draw(n=20001, tokens=4096, timestep_sampling="logit_normal", resolution_shift=True)
    assert s.median().item() == pytest.approx(0.759, abs=0.01)
    plain = draw(n=20001, tokens=4096, timestep_sampling="logit_normal")
    assert plain.median().item() == pytest.approx(0.5, abs=0.01)
    # a smaller picture is shifted less
    small = draw(n=20001, tokens=1024, timestep_sampling="logit_normal", resolution_shift=True)
    assert 0.5 < small.median().item() < s.median().item()


def test_is_exactly_the_shifted_draw():
    plain = draw(timestep_sampling="uniform")
    s = draw(timestep_sampling="uniform", resolution_shift=True)
    expected = (shifted(plain.double(), math.exp(1.15)) * 1000).round().clamp(1, 1000) / 1000
    assert (s.double() - expected).abs().max().item() <= 0.004  # (the plain draw was already snapped to the grid)


@pytest.mark.parametrize("family,module", [("krea2", "krea2"), ("qwen_image21", "qwen21"), ("fibo2", "fibo")])
def test_sampler_families_use_their_own_calculate_shift(family, module):
    sampler = __import__(f"rulztrainer.{module}.sampler", fromlist=["calculate_shift"])
    arch = arch_by_name(family)
    for tokens in (256, 1024, 4096, 6400, 8192, 16384):
        assert arch.resolution_mu(tokens) == sampler.calculate_shift(tokens)  # unclamped, extrapolating, as the sampler
    assert sampler.calculate_shift(4096) != arch_by_name("moss_image").resolution_mu(4096) or family == "fibo2"
    plain = draw(timestep_sampling="uniform", arch=family)
    s = draw(timestep_sampling="uniform", arch=family, resolution_shift=True)
    expected = (shifted(plain.double(), math.exp(sampler.calculate_shift(4096))) * 1000).round().clamp(1, 1000) / 1000
    assert (s.double() - expected).abs().max().item() <= 0.004


def test_klein_and_ideogram_mu():
    for family in ("flux2_klein", "flux2_klein4b"):
        arch = arch_by_name(family)
        assert arch.resolution_mu(256) == pytest.approx(0.5) and arch.resolution_mu(4096) == pytest.approx(1.15)
    ideo = arch_by_name("ideogram4")
    assert ideo.resolution_mu(1024) == pytest.approx(0.5 + 0.5 * math.log(512 * 512 / (512 * 512)))  # 512 px: 0.5
    assert ideo.resolution_mu(4096) == pytest.approx(0.5 + 0.5 * math.log(4))  # 1024 px


def test_reference_mode_is_not_shifted():
    for family in ("krea2", "moss_image", "zimage"):
        a = draw(timestep_sampling="reference", arch=family)
        b = draw(timestep_sampling="reference", arch=family, resolution_shift=True)
        assert torch.equal(a, b), family


def test_composes_with_timestep_shift_and_range():
    both = draw(timestep_sampling="uniform", timestep_shift=2.0, resolution_shift=True)
    one = draw(timestep_sampling="uniform", timestep_shift=2.0 * math.exp(1.15))
    assert (both - one).abs().max().item() <= 0.004  # shifts multiply (up to the grid snap between them)
    s = draw(timestep_sampling="logit_normal", resolution_shift=True, sigma_min=0.2, sigma_max=0.9)
    assert s.min().item() >= 0.2 and s.max().item() <= 0.9
    # the curriculum's logit shift still moves the curve before the resolution shift
    up = draw(timestep_sampling="logit_normal", resolution_shift=True, logit_shift=1.0)
    assert up.mean() > draw(timestep_sampling="logit_normal", resolution_shift=True).mean()


def test_buckets_still_cover_the_curve():
    cfg = TrainConfig(model_arch="moss_image", timestep_buckets=10, resolution_shift=True)
    strata = T.make_strata(cfg)
    gen = torch.Generator("cpu").manual_seed(1)
    arch = get_arch(cfg)
    cuts = [shifted(torch.tensor(float(q)), math.exp(1.15)).item() for q in torch.sigmoid(torch.special.ndtri(
        torch.tensor([i / 10 for i in range(1, 10)], dtype=torch.float64))).tolist()]
    for _ in range(5):
        s = T.sample_sigmas(cfg, 10, gen, 0.0, tokens=4096, arch=arch, strata=strata)
        slots = sorted(sum(1 for c in cuts if v > c) for v in s.tolist())
        assert slots == list(range(10))  # one draw in each tenth of the shifted curve


def test_unknown_tokens_apply_no_shift_and_warn_once(caplog):
    T._RESOLUTION_SHIFT_WARNED = False
    plain = draw(tokens=0, timestep_sampling="uniform")
    with caplog.at_level(logging.WARNING, logger="rulztrainer"):
        s = draw(tokens=0, timestep_sampling="uniform", resolution_shift=True)
        draw(tokens=0, timestep_sampling="uniform", resolution_shift=True)
    assert torch.equal(plain, s)
    assert sum("resolution_shift" in r.message for r in caplog.records) == 1


def test_clean_share_counts_the_resolution_shift():
    cfg = TrainConfig(model_arch="moss_image", resolution_shift=True)
    off = TrainConfig(model_arch="moss_image")
    assert CS.measured_share(cfg, 0.0, tokens=4096) < CS.measured_share(off, 0.0, tokens=4096) - 10  # fewer clean steps
    arch = get_arch(cfg)
    off_offset, _ = CS.offset_for(off, 40.0, 4096, arch)
    on_offset, got = CS.offset_for(cfg, 40.0, 4096, arch)
    assert got == pytest.approx(40.0, abs=1.0)  # the share is still met after the shift
    assert on_offset < off_offset - 0.5  # by moving the curve toward clean to cancel what the shift added
    # and per picture size: a smaller picture is shifted less, so it needs a smaller offset
    small, _ = CS.offset_for(cfg, 40.0, 1024, arch)
    assert on_offset < small < off_offset + 0.01


def test_start_note_and_reference_warning():
    arch = get_arch(TrainConfig(model_arch="krea2"))
    assert T.resolution_shift_note(TrainConfig(model_arch="krea2"), arch) == []
    (level, text), = T.resolution_shift_note(TrainConfig(model_arch="krea2", resolution_shift=True,
                                                         resolutions=[0.25, 1.0]), arch)
    assert level == "info" and "Resolution shift: x1.7" in text and "at 512 px" in text and "at 1024 px" in text
    (level, text), = T.resolution_shift_note(TrainConfig(model_arch="krea2", resolution_shift=True,
                                                         timestep_sampling="reference"), arch)
    assert level == "warning" and "no effect" in text


def test_capabilities_grey_it_out_where_nothing_is_sampled():
    for family in ("sdxl", "minimax_h3_image"):
        rule = caps.rules_for(family)["resolution_shift"]
        assert rule.ignored and not rule.allows(True) and rule.allows(False)
        assert "resolution_shift" not in caps.violations(family, {"resolution_shift": True})
    assert "resolution_shift" not in caps.rules_for("krea2")


def test_greyed_out_families_are_not_shifted_in_any_mode():
    # greyed out as having no effect, so it must not move their draws even outside reference mode
    for family in ("sdxl", "minimax_h3_image"):
        for mode in ("sigmoid", "uniform"):
            cfg = TrainConfig(model_arch=family, timestep_sampling=mode)
            on = TrainConfig(model_arch=family, timestep_sampling=mode, resolution_shift=True)
            a = T.sample_sigmas(cfg, 500, torch.Generator("cpu").manual_seed(1), tokens=4096, arch=arch_by_name(family))
            b = T.sample_sigmas(on, 500, torch.Generator("cpu").manual_seed(1), tokens=4096, arch=arch_by_name(family))
            assert torch.equal(a, b), (family, mode)


def test_config_round_trip_and_keys():
    from rulztrainer import history, race

    cfg = TrainConfig(resolution_shift=True)
    assert cfg.resolution_shift is True
    assert "resolution_shift" in race.ALLOWED
    assert "resolution_shift" in history.CONFIG_KEYS
