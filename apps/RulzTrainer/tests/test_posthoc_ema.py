"""Post-hoc EMA: moving-average weights over one run's checkpoints in step order, applied to their weight changes."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rulztrainer import posthoc_ema as P  # noqa: E402
from rulztrainer import soup as S  # noqa: E402
from rulztrainer.lora import save_lora  # noqa: E402

OUT, IN, RANK = 24, 20, 4
NAMES = ("layers.0.attn.q", "layers.0.attn.k")


def checkpoint(path, step: int | None, seed: int, meta: dict | None = None) -> dict:
    """A rank-4 LoRA checkpoint (ss_steps = step when given); returns its weight changes."""
    g = torch.Generator().manual_seed(seed)
    stubs, out = {}, {}
    for name in NAMES:
        parts = {"lora_down": torch.randn(RANK, IN, generator=g), "lora_up": torch.randn(OUT, RANK, generator=g)}
        stubs[name] = SimpleNamespace(kind="lora", alpha=2.0, rank=RANK, factor=4, trainable=True,
                                      adapter_params=lambda parts=parts: parts)
        out[name] = parts["lora_up"] @ parts["lora_down"] * (2.0 / RANK)
    metadata = dict(meta or {})
    if step is not None:
        metadata["ss_steps"] = str(step)
    save_lora(stubs, path, metadata=metadata, dtype=torch.float32)
    return out


def test_constant_decay_weights():
    assert P.ema_weights([100, 200, 300], beta=0.5) == pytest.approx([0.25, 0.25, 0.5])
    w = P.ema_weights(list(range(1, 11)), beta=0.9)
    assert w[-1] == pytest.approx(0.1) and w[0] == pytest.approx(0.9 ** 9)
    assert sum(w) == pytest.approx(1.0)


def test_a_sliding_decay_follows_the_recurrence():
    steps, b1, b2 = [1, 2, 3, 4, 5], 0.5, 0.9
    avg = {0: 1.0}
    for j in range(4):  # the merge of musubi-tuner's lora_post_hoc_ema.py, on one-hot "files"
        b = b1 + (b2 - b1) * j / 3
        avg = {k: v * b for k, v in avg.items()}
        avg[j + 1] = 1 - b
    assert P.ema_weights(steps, b1, b2) == pytest.approx([avg[i] for i in range(5)])


def test_power_function_weights():
    assert P.sigma_rel_to_gamma(0.1) == pytest.approx(6.94, abs=0.01)  # EDM2's table
    assert P.sigma_rel_to_gamma(0.05) == pytest.approx(16.97, abs=0.01)
    w = P.ema_weights([100, 200, 300, 400], sigma_rel=0.1)
    assert w == sorted(w) and sum(w) == pytest.approx(1.0)
    # a checkpoint that stands for twice the steps weighs twice as much at the same step
    even = P.ema_weights([100, 200, 300], sigma_rel=0.2)
    uneven = P.ema_weights([100, 150, 300], sigma_rel=0.2)
    assert uneven[2] / uneven[1] > even[2] / even[1]
    with pytest.raises(ValueError, match="sigma_rel"):
        P.sigma_rel_to_gamma(0.3)
    with pytest.raises(ValueError, match="beta"):
        P.ema_weights([1, 2], beta=1.0)


def test_the_average_of_the_changes_in_step_order(tmp_path):
    deltas = {}
    for step, seed in ((300, 3), (100, 1), (200, 2)):  # given out of order
        deltas[step] = checkpoint(tmp_path / f"run_{step:06d}.safetensors", step, seed,
                                  {"ss_base_model": "base.safetensors"})
    files = [tmp_path / f"run_{s:06d}.safetensors" for s in (300, 100, 200)]
    rep = P.make_ema(files, tmp_path / "ema.safetensors", beta=0.5, rank=0)
    assert rep["steps"] == [100, 200, 300] and rep["weights"] == pytest.approx([0.25, 0.25, 0.5])
    got = {n: S.delta_of(e) for n, e in S.read_ingredient(rep["out"]).items()}
    for name in NAMES:
        want = 0.25 * deltas[100][name] + 0.25 * deltas[200][name] + 0.5 * deltas[300][name]
        assert torch.allclose(got[name], want, rtol=0.02, atol=0.02)  # (saved in bf16)
    with safe_open(rep["out"], "pt") as f:
        meta = f.metadata()
    assert meta["ss_base_model"] == "base.safetensors" and "beta 0.5" in meta["rulztrainer_posthoc_ema"]
    assert rep["effective_step"] == pytest.approx(225.0)
    assert "3 checkpoints (steps 100-300)" in P.describe(rep)


def test_default_rank_is_the_inputs(tmp_path):
    files = [tmp_path / f"run_{s:06d}.safetensors" for s in (100, 200)]
    for i, p in enumerate(files):
        checkpoint(p, (i + 1) * 100, i)
    rep = P.make_ema(files, tmp_path / "ema.safetensors")
    assert rep["rank"] == RANK
    assert {e[1].shape[0] for e in S.read_ingredient(rep["out"]).values()} == {RANK}


def test_order_without_metadata(tmp_path):
    a, b = tmp_path / "run_000200.safetensors", tmp_path / "run_000100.safetensors"
    checkpoint(a, None, 1)
    checkpoint(b, None, 2)
    assert P.order_checkpoints([a, b]) == [(b, 100), (a, 200)]  # by the step in the name
    c, d = tmp_path / "late.safetensors", tmp_path / "early.safetensors"
    checkpoint(c, None, 3)
    checkpoint(d, None, 4)
    os.utime(d, (1_000_000, 1_000_000))
    assert P.order_checkpoints([c, d]) == [(d, 1), (c, 2)]  # by modification time
    checkpoint(tmp_path / "copy.safetensors", 100, 5)
    with pytest.raises(ValueError, match="two checkpoints of step 100"):
        P.order_checkpoints([b, tmp_path / "copy.safetensors"])
    with pytest.raises(ValueError, match="2 or more"):
        P.make_ema([a], tmp_path / "x.safetensors")


def test_cli(tmp_path, capsys):
    import posthoc_ema as cli

    files = [tmp_path / f"run_{s:06d}.safetensors" for s in (100, 200, 300)]
    for i, p in enumerate(files):
        checkpoint(p, (i + 1) * 100, i)
    assert cli.main([*map(str, files), "--out", str(tmp_path / "o.safetensors"), "--sigma_rel", "0.1"]) == 0
    text = capsys.readouterr().out
    assert "sigma_rel 0.1" in text and "run_000300.safetensors" in text and (tmp_path / "o.safetensors").is_file()
