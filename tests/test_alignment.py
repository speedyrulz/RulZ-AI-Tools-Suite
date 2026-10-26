"""Attention localization and identity alignment (alignment.py): the trigger's tokens in a prompt, the localization
and alignment losses, gradients through the taps that match with and without gradient checkpointing on the three tiny
models (a block's recompute in the backward must save the same tensors), face embeddings in the face analysis and
the config checks."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import alignment as A  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.fibo.model import FiboTransformer  # noqa: E402
from rulztrainer.krea2.model import Krea2Transformer  # noqa: E402
from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402

H, W = 4, 6


def _regions(batch: int, positions: list[list[int]]):
    prior = torch.ones(batch, 1, H, W, dtype=torch.bool)
    prior[:, :, 1:4, 1:4] = False  # the subject: a 3 x 3 block of cells
    meta = [SimpleNamespace(prior=prior[b], face_cells=(1, 1, 3, 3), has_person=True) for b in range(batch)]
    return SimpleNamespace(prior=prior, meta=meta, text_paths=[str(b) for b in range(batch)]), positions


def _install(model, attn_layers, hidden_layer):
    taps = A.Taps([hidden_layer], attn_layers)
    model._rz_taps = taps
    for i in attn_layers:
        A.attention_module(model.layers[i])._rz_tap = A.AttnTap(taps, i)
    return taps


def _moss():
    m = MOSSImageTransformer(dim=64, ffn_dim=128, num_heads=2, num_layers=6, text_dim=32, freq_dim=16,
                             latent_channels=128)
    enc = torch.randn(2, 7, 32)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False  # MOSS runs each sample on its own valid tokens: the two text lengths differ
    return m, torch.randn(2, 128, H, W), enc, mask, [2, 3], 2


def _krea():
    m = Krea2Transformer(in_channels=64, num_layers=2, attention_head_dim=16, num_attention_heads=4,
                         num_key_value_heads=2, intermediate_size=96, timestep_embed_dim=32, text_hidden_dim=40,
                         num_text_layers=3, text_num_attention_heads=4, text_num_key_value_heads=4,
                         text_intermediate_size=56, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
                         axes_dims_rope=(4, 6, 6))
    enc = torch.randn(2, 7, 3, 40)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    return m, torch.randn(2, 64, H, W), enc, mask, [1], 0


def _fibo():
    m = FiboTransformer(dim=64, n_layers=4, n_refiner_layers=1, n_heads=2, cap_feat_dim=48, injection_layer_ids=(1, 3),
                        perceiver_num_layers=1, min_num_gist_tokens=4, max_num_gist_tokens=8, gist_step=2,
                        gist_min_text_len=4, gist_max_text_len=16, axes_dims=(8, 12, 12), axes_lens=(64, 32, 32))
    enc = torch.randn(2, 3, 9, 48)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 5:] = False
    return m, torch.randn(2, 128, H, W), enc, mask, [1, 3], 2


def _grads(make, checkpointing: bool):
    torch.manual_seed(0)
    model, x, enc, mask, attn_layers, hidden_layer = make()
    model.gradient_checkpointing = checkpointing
    model.train()
    taps = _install(model, attn_layers, hidden_layer)
    aligner = A.IdAligner(A.model_dim(model))
    regions, positions = _regions(2, [[0, 1], [2]])
    targets = [torch.nn.functional.normalize(torch.randn(A.FACE_EMB_DIM), dim=0) for _ in range(2)]
    taps.begin(positions)
    out = model(x, torch.tensor([600.0, 200.0]), enc, mask)
    taps.stop()
    loc, n_loc = A.localization_loss(taps, regions)
    ali, n_ali = aligner.loss(taps, hidden_layer, regions, targets)
    assert n_loc == 2 and n_ali == 2 and 0 < loc.item() < 1 and 0 < ali.item() < 2
    (out.float().pow(2).mean() + 0.5 * loc + 0.5 * ali).backward()
    taps.end()
    return [p.grad.clone() for p in list(model.parameters()) + list(aligner.parameters()) if p.grad is not None], \
        float(loc), float(ali)


@pytest.mark.parametrize("make", [_moss, _krea, _fibo], ids=["moss", "krea2", "fibo2"])
def test_the_losses_backpropagate_the_same_with_checkpointing(make):
    plain, loc, ali = _grads(make, False)
    ckpt, loc2, ali2 = _grads(make, True)
    assert abs(loc - loc2) < 1e-6 and abs(ali - ali2) < 1e-6
    assert len(plain) == len(ckpt)
    for a, b in zip(plain, ckpt):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)


def test_taps_record_only_while_armed():
    model, x, enc, mask, attn_layers, hidden_layer = _krea()
    taps = _install(model, attn_layers, hidden_layer)
    with torch.no_grad():
        model(x, torch.tensor([600.0, 200.0]), enc, mask)  # not armed: a sampling forward
    assert taps.attn == {} and taps.hidden == {}
    taps.begin([[0], [1]], attention=False)  # identity alignment alone: no attention is read
    with torch.no_grad():
        model(x, torch.tensor([600.0, 200.0]), enc, mask)
    assert taps.attn == {} and set(taps.hidden) == {hidden_layer} and taps.hidden[hidden_layer][1].shape == (H * W, 64)
    taps.end()
    assert not taps.armed and taps.hidden == {}


def test_localization_loss_is_the_share_off_the_subject():
    regions, _ = _regions(1, [[0]])
    taps = A.Taps([], [5])
    taps.begin([[0]])
    on = torch.zeros(H * W, 3)
    on[:, 1:] = 1.0
    subject = (~regions.prior[0]).flatten()
    on[subject, 0] = 5.0  # the trigger (token 0) is attended to only on the subject
    taps.attn = {5: {0: on / on.sum(-1, keepdim=True)}}
    loss, n = A.localization_loss(taps, regions)
    assert n == 1 and float(loss) < 1e-6
    off = torch.ones(H * W, 3)
    off[subject, 0] = 0.0
    taps.attn = {5: {0: off / off.sum(-1, keepdim=True)}}
    assert abs(float(A.localization_loss(taps, regions)[0]) - 1.0) < 1e-6
    taps.positions = [[]]  # no trigger in this caption: nothing to localize
    assert A.localization_loss(taps, regions) == (None, 0)
    regions.meta[0].face_cells, regions.meta[0].has_person = None, False  # no subject found in the picture
    taps.positions = [[0]]
    assert A.localization_loss(taps, regions) == (None, 0)


def test_trigger_tokens_from_offsets_and_layers():
    text = "<s>system zkrea, a photo of zkrea"
    offsets = [(0, 3), (3, 10), (10, 12), (12, 15), (15, 16), (16, 18), (18, 24), (24, 27), (27, 30), (30, 33)]
    assert A.token_positions(text, offsets, ["zkrea"], drop=0) == [2, 3, 8, 9]
    assert A.token_positions(text, offsets, ["ZKREA", ""], drop=3) == [0, 5, 6]  # case-insensitive; dropped prefix
    assert A.token_positions(text, offsets, ["nobody"]) == []
    assert A.auto_attn_layers("krea2", 28) == [9, 14] and A.auto_attn_layers("fibo2", 44, (6, 14)) == [6, 14]
    assert A.auto_hidden_layer(42) == 14 and A.parse_layers("", 10) is None and A.parse_layers("3, 1", 10) == [1, 3]
    with pytest.raises(ValueError):
        A.parse_layers("12", 10)


def test_face_analysis_keeps_embeddings(tmp_path):
    from PIL import Image

    from rulztrainer import regions as R

    img = tmp_path / "a.png"
    Image.new("RGB", (32, 32)).save(img)
    calls = []

    class Scorer:
        def faces(self, p):
            calls.append(p)
            return [SimpleNamespace(bbox=[1, 2, 10, 12], kps=[[3, 4]] * 5, det_score=0.9,
                                    normed_embedding=[0.5] * A.FACE_EMB_DIM)]

    item = SimpleNamespace(image=img)
    cache = tmp_path / "faces.json"
    no_emb = R.analyze_faces([item], Scorer(), cache)
    rec = next(iter(no_emb.values()))
    assert "emb" in rec and len(calls) == 1  # stored whenever the scorer gives one
    rec.pop("emb")
    import json

    cache.write_text(json.dumps({R._file_key(img): rec}), encoding="utf-8")
    R.analyze_faces([item], Scorer(), cache)
    assert len(calls) == 1  # no embedding wanted: the cached face is enough
    again = R.analyze_faces([item], Scorer(), cache, embeddings=True)
    assert len(calls) == 2 and len(next(iter(again.values()))["emb"]) == A.FACE_EMB_DIM


def test_config_checks():
    TrainConfig(attn_localize=True, trigger_word="zkrea", id_align=True).validate()
    assert TrainConfig(attn_localize=True).wants_regions() and TrainConfig(attn_localize=True).wants_person_masks()
    assert TrainConfig(id_align=True).wants_regions() and not TrainConfig(id_align=True).wants_person_masks()
    with pytest.raises(ValueError, match="trigger"):
        TrainConfig(attn_localize=True).validate()
    with pytest.raises(ValueError, match="weight"):
        TrainConfig(id_align_weight=-1).validate()


def test_isolation_pairs_use_the_subjects_panel():
    from rulztrainer import isolation as iso

    caption = "a man with a beard on the left, and gweltman on the right, two people side by side, two portraits"
    assert iso.pair_side(caption, "gweltman") == "right" and iso.pair_side(caption, "zkrea") == ""
    assert iso.pair_side("zkrea on the left, a woman on the right", "zkrea") == "left"
    left = A.panel_mask("left", 2, 5).view(2, 5)
    assert left[:, :2].eq(1).all() and left[:, 2:].eq(0).all()  # the odd middle column belongs to neither
    assert A.panel_mask("right", 2, 4).view(2, 4)[:, 2:].eq(1).all() and A.panel_mask("", 2, 4) is None
    # a pair has no region map (uniform: no subject), so only the panel makes it count
    regions = SimpleNamespace(prior=torch.zeros(1, 1, H, W, dtype=torch.bool),
                              meta=[SimpleNamespace(prior=torch.zeros(1, H, W, dtype=torch.bool), face_cells=None,
                                                    has_person=False)])
    taps = A.Taps([], [3])
    taps.begin([[0]])
    on_right = torch.full((H * W, 2), 0.5)
    right = A.panel_mask("right", H, W).bool()
    on_right[right, 0], on_right[right, 1] = 1.0, 0.0  # the trigger (token 0) is attended to on the right only
    on_right[~right, 0], on_right[~right, 1] = 0.0, 1.0
    taps.attn = {3: {0: on_right}}
    assert A.localization_loss(taps, regions) == (None, 0)
    loss, n = A.localization_loss(taps, regions, [A.panel_mask("right", H, W)])
    assert n == 1 and float(loss) < 1e-6
    assert abs(float(A.localization_loss(taps, regions, [A.panel_mask("left", H, W)])[0]) - 1.0) < 1e-6
