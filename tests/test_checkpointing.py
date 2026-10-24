"""Selective gradient checkpointing (rulztrainer.checkpointing): kept blocks change speed and memory, never results."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import archs  # noqa: E402
from rulztrainer import checkpointing as C  # noqa: E402
from rulztrainer import lora as L  # noqa: E402
from rulztrainer.fibo.model import FiboTransformer  # noqa: E402
from rulztrainer.krea2.model import Krea2Transformer  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402


def moss():
    m = MOSSImageTransformer(dim=256, ffn_dim=256, num_heads=4, num_layers=4, text_dim=32, freq_dim=16,
                             latent_channels=128)
    x, t = torch.randn(1, 128, 4, 4), torch.tensor([300.0])
    return m, "moss_image", (x, t, torch.randn(1, 5, 32), torch.ones(1, 5, dtype=torch.bool))


def fibo():
    m = FiboTransformer(dim=64, n_layers=4, n_refiner_layers=1, n_heads=2, cap_feat_dim=48, injection_layer_ids=(1, 3),
                        perceiver_num_layers=1, min_num_gist_tokens=4, max_num_gist_tokens=8, gist_step=2,
                        gist_min_text_len=4, gist_max_text_len=16, axes_dims=(8, 12, 12), axes_lens=(64, 32, 32))
    return m, "fibo2", (torch.randn(1, 128, 4, 6), torch.tensor([600.0]), torch.randn(1, 3, 9, 48), None)


def krea():
    m = Krea2Transformer(in_channels=64, num_layers=4, attention_head_dim=16, num_attention_heads=4,
                         num_key_value_heads=2, intermediate_size=96, timestep_embed_dim=32, text_hidden_dim=40,
                         num_text_layers=3, text_num_attention_heads=4, text_num_key_value_heads=4,
                         text_intermediate_size=56, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
                         axes_dims_rope=(4, 6, 6))
    return m, "krea2", (torch.randn(1, 64, 3, 4), torch.tensor([400.0]), torch.randn(1, 5, 3, 40), None)


@pytest.mark.parametrize("build", [moss, fibo, krea])
def test_kept_blocks_give_identical_loss_and_grads(build):
    torch.manual_seed(0)
    model, arch_name, inputs = build()
    arch = archs.arch_by_name(arch_name)
    names = arch.lora_module_names(model, arch.default_targets, [1, 2, 3])
    inj = L.inject_lora(model, 4, 4.0, list(arch.default_targets), names=names)
    with torch.no_grad():
        for mod in inj.values():
            mod.lora_up.weight.normal_(0, 0.05)
    model.train()
    params = list(L.lora_parameters(inj))

    def run(checkpointing: bool, keep: list[int]):
        model.gradient_checkpointing = checkpointing
        C.set_kept_blocks(model, keep)
        for p in params:
            p.grad = None
        loss = model(*inputs).float().pow(2).mean()
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in params]

    loss_ref, grads_ref = run(False, [])
    for checkpointing, keep in ((True, []), (True, [2, 3]), (True, [1, 2, 3])):
        loss, grads = run(checkpointing, keep)
        assert torch.equal(loss, loss_ref)
        assert all(torch.equal(a, b) for a, b in zip(grads, grads_ref)), (arch_name, keep)
    assert C.kept_blocks(model) == [1, 2, 3]


def test_block_selection_and_budget():
    assert C.trained_block_indices(6, None) == [0, 1, 2, 3, 4, 5]
    assert C.trained_block_indices(6, [3, 5]) == [3, 4, 5]  # everything after the first trained block backprops
    assert C.trained_block_indices(6, []) == []
    assert C.kept_indices([3, 4, 5], 2) == [4, 5] and C.kept_indices([3, 4, 5], 0) == []
    assert C.kept_indices([3, 4, 5], 9) == [3, 4, 5]
    gb = C.GB
    # the first kept block costs nothing extra (peak_one is measured with it kept); each further one costs per_block
    assert C.choose_kept_blocks(10, 8 * gb, 1 * gb, 11.5 * gb) == 4
    assert C.choose_kept_blocks(10, 8 * gb, 1 * gb, 8 * gb) == 1
    assert C.choose_kept_blocks(10, 8 * gb, 1 * gb, 7 * gb) == 0  # even one kept block is above the budget
    assert C.choose_kept_blocks(4, 8 * gb, 1 * gb, 30 * gb) == 4  # clamped to the candidates
    assert C.choose_kept_blocks(4, 8 * gb, 0, 9 * gb) == 4  # further blocks measured free
    assert C.choose_kept_blocks(0, 1, 1, 10) == 0
    assert C.memory_margin(16 * gb) == pytest.approx(1.5 * gb)
    assert C.memory_margin(80 * gb) == pytest.approx(6.4 * gb)
    assert C.memory_margin(16 * gb, identity_loss=True) == pytest.approx(3.0 * gb)


def test_out_of_memory_fallback_halves_the_kept_blocks():
    from rulztrainer.trainer import Trainer

    torch.manual_seed(0)
    model, _, _ = moss()
    trainer = Trainer.__new__(Trainer)
    trainer.replicas = [(model, {}, "cpu")]
    trainer.kept = {id(model): [0, 1, 2, 3]}
    C.set_kept_blocks(model, [0, 1, 2, 3])
    assert trainer._shrink_kept_blocks() and C.kept_blocks(model) == [2, 3]
    assert trainer._shrink_kept_blocks() and C.kept_blocks(model) == [3]
    assert trainer._shrink_kept_blocks() and C.kept_blocks(model) == []
    assert not trainer._shrink_kept_blocks()  # nothing left to give back: the error is raised as before


def test_allocator_retries_give_back_one_kept_block_per_gpu():
    from rulztrainer.trainer import Trainer

    torch.manual_seed(0)
    model, _, _ = moss()
    model.split_assignment = ["cuda:1", "cuda:1", "cuda:0", "cuda:0"]
    trainer = Trainer.__new__(Trainer)
    trainer.replicas = [(model, {}, "cuda:1")]
    trainer.devices = ["cuda:1", "cuda:0"]
    trainer.kept = {id(model): [0, 1, 2, 3]}
    C.set_kept_blocks(model, [0, 1, 2, 3])
    counts = {"cuda:1": 5, "cuda:0": 2}
    trainer._alloc_retries = lambda: dict(counts)
    trainer._ease_memory_pressure({"cuda:1": 5, "cuda:0": 2})  # no new retries: nothing changes
    assert C.kept_blocks(model) == [0, 1, 2, 3]
    counts["cuda:0"] = 3  # only the second GPU retried: its earliest kept block goes
    trainer._ease_memory_pressure({"cuda:1": 5, "cuda:0": 2})
    assert C.kept_blocks(model) == [0, 1, 3]


def test_calibration_uses_the_longest_caption(tmp_path):
    """The blocks see text and image tokens, so the activation memory is measured with the longest cached caption
    (for each family's own embeds layout)."""
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from rulztrainer.trainer import _longest_text_path

    for name, shape in (("krea2", lambda s: (s, 12, 8)), ("moss_image", lambda s: (s, 8)),
                        ("fibo2", lambda s: (6, s, 8))):
        items = []
        for i, s in enumerate((5, 40, 12)):
            p = tmp_path / f"{name}_{i}.safetensors"
            save_file({"embeds": torch.zeros(shape(s))}, str(p))
            items += [SimpleNamespace(text_path=p)] * 2
        assert _longest_text_path(items, archs.arch_by_name(name).pad_embeds).name == f"{name}_1.safetensors"
