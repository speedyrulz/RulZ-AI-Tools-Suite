import math
from pathlib import Path

import pytest
import torch

from rulztrainer.archs import arch_by_name
from rulztrainer.dataset import Item


FAMILIES = ("flux2_klein", "krea2", "qwen_image21", "fibo2", "ideogram4", "zimage", "moss_image")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("height,width", [(1024, 1024), (768, 1344)])
def test_image_tokens_are_16_pixel_cells(family, height, width):
    assert arch_by_name(family).name == family
    path = Path("image.png")
    item = Item(image=path, caption="", width=width, height=height, latent_path=path,
                latent_flip_path=None, text_path=path)
    assert item.tokens == (height // 16) * (width // 16)


@pytest.mark.parametrize("family", FAMILIES[:5])
def test_reference_sigmas_increase_with_token_count(family):
    arch = arch_by_name(family)
    small = arch.reference_sigmas(10001, torch.Generator().manual_seed(42), 1024)
    large = arch.reference_sigmas(10001, torch.Generator().manual_seed(42), 4096)
    assert large.median() > small.median()


def test_flux2_shift_at_4096_tokens():
    seed, count = 42, 100
    unshifted = torch.sigmoid(torch.randn(count, generator=torch.Generator().manual_seed(seed)))
    shifted = arch_by_name("flux2_klein").reference_sigmas(count, torch.Generator().manual_seed(seed), 4096)
    # Odds are multiplied by the resolution-dependent shift.
    shift = (shifted / (1 - shifted)) / (unshifted / (1 - unshifted))
    torch.testing.assert_close(shift, torch.full_like(shift, math.exp(1.15)), atol=0, rtol=1e-5)  # float32 odds
