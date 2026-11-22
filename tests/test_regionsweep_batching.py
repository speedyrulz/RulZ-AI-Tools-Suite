"""Fixed-noise sweep batching agrees with the original per-picture algorithm."""
import contextlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rulztrainer import regionsweep as RS  # noqa: E402
from rulztrainer.probeworker import _sweep_picture  # noqa: E402
from rulztrainer.trainmaps import eager_stance  # noqa: E402


class Toy(torch.nn.Module):
    def __init__(self, limit=None):
        super().__init__()
        self.limit = limit
        self.batches = []

    def forward(self, noisy, sigma, embeds, mask):
        self.batches.append((noisy.shape, embeds.shape, mask.shape))
        if self.limit and noisy.shape[0] > self.limit:
            raise torch.cuda.OutOfMemoryError("fake out of memory")
        text = (embeds * mask.unsqueeze(-1)).sum(dim=(1, 2)).view(-1, 1, 1, 1)
        return noisy.float() * 0.3 + sigma.view(-1, 1, 1, 1) * 0.001 + text * 0.01


def pictures():
    result = []
    for i, (h, w, length) in enumerate([(4, 6, 2), (3, 5, 3), (4, 6, 2), (4, 6, 3), (3, 5, 3)]):
        g = torch.Generator().manual_seed(100 + i)
        latents = torch.randn(1, 2, h, w, generator=g)
        embeds = torch.randn(1, length, 3, generator=g)
        mask = torch.ones(1, length, dtype=torch.bool)
        if i % 2:
            mask[0, -1] = False
        parts = torch.zeros(7, h, w, dtype=torch.uint8)
        parts[i % 6, :, :w // 2] = 255
        parts[(i + 3) % 6, :, w // 2:] = 255
        def load(latents=latents, embeds=embeds, mask=mask):
            return latents, embeds, mask
        result.append({"key": f"picture{i}.png", "parts": parts, "load": load})
    return result


def _reference(model, device, chosen, averaged=None):
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    out = {}
    try:
        model.eval()
        with (averaged() if averaged is not None else contextlib.nullcontext()), torch.no_grad(), eager_stance():
            for pr in chosen:
                latents, embeds, mask = pr["load"]()
                latents = latents.to(device, torch.float32)
                embeds, mask = embeds.to(device), mask.to(device)
                rows = []
                for i, s in enumerate(RS.SIGMAS):
                    noise = RS.fixed_noise(pr["key"], i, latents.shape).to(device)
                    noisy = ((1 - s) * latents + s * noise).to(torch.bfloat16)
                    sigma = torch.full((latents.shape[0],), s * 1000.0, device=device)
                    err = (model(noisy, sigma, embeds, mask).float() - (noise - latents)).pow(2)
                    if err.dim() != 4:
                        raise RuntimeError("the model's prediction is not a latent grid")
                    losses, cells = RS.region_losses(err, [pr["parts"]])
                    if losses[0] is None:
                        raise RuntimeError(f"the parts map of {Path(pr['key']).name} does not fit its latent")
                    rows.append([losses[0], cells[0]])
                out[pr["key"]] = rows
    finally:
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return out


def assert_readings(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        for actual_row, expected_row in zip(actual[key], expected[key], strict=True):
            for actual_values, expected_values in zip(actual_row, expected_row, strict=True):
                assert actual_values == pytest.approx(expected_values, rel=1e-5, abs=1e-7)


@pytest.fixture(autouse=True)
def clean_sizes(monkeypatch):
    monkeypatch.setattr(RS, "_batch_sizes", {})
    RS.clear_cache()
    yield
    RS.clear_cache()


def test_batch_cache_and_shapes(monkeypatch):
    chosen = pictures()
    expected = _reference(Toy(), "cpu", chosen)
    model = Toy()
    actual = RS.forward_readings(model, "cpu", chosen)
    assert_readings(actual, expected)
    assert sorted(shape[0][0] for shape in model.batches) == [3, 6, 6]
    assert all(a[0] == b[0] == c[0] for a, b, c in model.batches)
    for pr in chosen:
        pr["load"] = lambda: pytest.fail("cached picture was reloaded")
    monkeypatch.setattr(RS, "fixed_noise", lambda *args: pytest.fail("cached noise was regenerated"))
    assert_readings(RS.forward_readings(Toy(), "cpu", chosen), expected)


def test_budget_and_single_picture(monkeypatch):
    chosen = pictures()
    expected = _reference(Toy(), "cpu", chosen)
    monkeypatch.setattr(RS, "CACHE_BYTES", 1)
    assert_readings(RS.forward_readings(Toy(), "cpu", chosen), expected)
    assert not RS._cache and RS._cache_bytes == 0
    runner = type("Runner", (), {"model": Toy(), "device": "cpu"})()
    assert_readings(_sweep_picture(runner, chosen[0]), {chosen[0]["key"]: expected[chosen[0]["key"]]})


def test_oom_backoff_is_remembered():
    chosen = pictures()
    expected = _reference(Toy(), "cpu", chosen)
    model = Toy(limit=2)
    assert_readings(RS.forward_readings(model, "cpu", chosen), expected)
    assert RS._batch_sizes["cpu"] == 1
    again = Toy(limit=2)
    assert_readings(RS.forward_readings(again, "cpu", chosen), expected)
    assert all(shape[0][0] == 1 for shape in again.batches)


def test_shared_file_cache_and_changed_stamp(tmp_path):
    latent = tmp_path / "latent"
    text = tmp_path / "text"
    latent.write_bytes(b"a")
    text.write_bytes(b"b")
    first = pictures()[0]
    original_load = first["load"]
    calls = []

    def load():
        calls.append(1)
        return original_load()

    first.update(latent_path=latent, text_path=text, load=load)
    second = dict(first)
    expected = _reference(Toy(), "cpu", [first])
    calls.clear()
    assert_readings(RS.forward_readings(Toy(), "cpu", [first]), expected)
    size = RS._cache_bytes
    assert size > 0
    assert_readings(RS.forward_readings(Toy(), "cpu", [second]), expected)
    assert len(calls) == 1 and RS._cache_bytes == size
    latent.write_bytes(b"changed size")
    assert_readings(RS.forward_readings(Toy(), "cpu", [second]), expected)
    assert len(calls) == 2 and RS._cache_bytes == size and len(RS._cache) == 1
    RS.clear_cache()
    assert RS._cache_bytes == 0 and not RS._cache


def test_uncached_pictures_are_loaded_one_at_a_time(monkeypatch):
    chosen = pictures()
    expected = _reference(Toy(), "cpu", chosen)
    monkeypatch.setattr(RS, "CACHE_BYTES", 1)
    loads = []
    for pr in chosen:
        original = pr["load"]
        def load(original=original):
            loads.append(1)
            return original()
        pr["load"] = load

    class CheckLoads(Toy):
        def forward(self, *args):
            assert len(loads) == len(self.batches) + 1
            return super().forward(*args)

    model = CheckLoads()
    assert_readings(RS.forward_readings(model, "cpu", chosen), expected)
    assert len(loads) == len(model.batches) == len(chosen)


def test_tail_oom_halves_tried_batch_size():
    class Once(Toy):
        def forward(self, *args):
            if not self.batches:
                self.batches.append((args[0].shape, args[2].shape, args[3].shape))
                raise torch.cuda.OutOfMemoryError("fake")
            return super().forward(*args)

    chosen = pictures()[:1]
    expected = _reference(Toy(), "cpu", chosen)
    assert_readings(RS.forward_readings(Once(), "cpu", chosen), expected)
    assert RS._batch_sizes["cpu"] == 6


def test_oom_at_one_is_raised():
    class AlwaysOOM(Toy):
        def forward(self, *args):
            raise torch.cuda.OutOfMemoryError("fake")

    RS._batch_sizes["cpu"] = 1
    with pytest.raises(torch.cuda.OutOfMemoryError):
        RS.forward_readings(AlwaysOOM(), "cpu", pictures()[:1])
