import pytest
import torch

from rulztrainer.fibo import model as fibo
from rulztrainer.zimage import model as zimage


def old_tables(ids, dims, lens, theta):
    tables = []
    for dim, length in zip(dims, lens):
        frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        angles = torch.outer(torch.arange(length, dtype=torch.float64), frequencies).float()
        tables.append(torch.polar(torch.ones_like(angles), angles).to(torch.complex64))
    return torch.cat([tables[i][ids[..., i]] for i in range(len(dims))], dim=-1)


def old_apply(x, freqs):
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * freqs.unsqueeze(2)).flatten(3).type_as(x)


@pytest.mark.parametrize("module", [zimage, fibo])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_rope_matches_complex(module, dtype):
    torch.manual_seed(17)
    dims, lens, theta = (4, 6, 8), (11, 13, 17), 256.0
    ids = torch.stack([torch.randint(length, (2, 5), dtype=torch.long) for length in lens], dim=-1)
    embedder = module.RopeEmbedder(theta, dims, lens)
    if module is fibo:
        # Fibo constructs one shared sequence, then adds a batch dimension.
        ids = ids[0]
        actual_freqs = embedder(ids).unsqueeze(0)
        expected_freqs = old_tables(ids, dims, lens, theta).unsqueeze(0)
    else:
        actual_freqs = embedder(ids)
        expected_freqs = old_tables(ids, dims, lens, theta)
    x = torch.randn(2, 5, 3, sum(dims)).to(dtype)
    actual = module._apply_rope(x, actual_freqs)
    expected = old_apply(x, expected_freqs)
    assert actual.dtype == dtype
    assert not actual_freqs.is_complex()
    torch.testing.assert_close(actual, expected, atol=1e-6 if dtype == torch.float32 else 0.016,
                               rtol=1e-6 if dtype == torch.float32 else 0.016)


def test_zimage_zeroed_padding_matches_complex():
    ids = torch.zeros(1, 5, 3, dtype=torch.long)
    dims, lens = (4, 6, 8), (3, 3, 3)
    actual = zimage.RopeEmbedder(256.0, dims, lens)(ids)
    expected = old_tables(ids, dims, lens, 256.0)
    actual[:, 3:] = 0
    expected[:, 3:] = 0
    x = torch.randn(1, 5, 2, sum(dims))
    torch.testing.assert_close(zimage._apply_rope(x, actual), old_apply(x, expected), atol=1e-6, rtol=1e-6)
