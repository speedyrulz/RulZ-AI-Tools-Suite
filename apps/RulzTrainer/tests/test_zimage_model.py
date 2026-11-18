"""Z-Image transformer facade: parity with diffusers' ZImageTransformer2DModel on a tiny config (CPU), the patch
layout, the ComfyUI single-file conversion, detection, checkpointing and skip_layers."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer.zimage import model as M  # noqa: E402
from rulztrainer.zimage import single_file as SF  # noqa: E402

TINY = dict(dim=64, n_layers=2, n_refiner_layers=1, n_heads=4, n_kv_heads=4, cap_feat_dim=32,
            axes_dims=(4, 6, 6), axes_lens=(128, 64, 64))
LAT_H, LAT_W = 26, 38  # 13 x 19 = 247 image tokens: not a multiple of 32


def _randomize(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1 + (1.0 if p.ndim == 1 else 0.0))
    return model


def _facade(dtype=torch.float32, seed=1, **kw):
    m = M.ZImageTransformer(**{**TINY, **kw})
    _randomize(m, seed)
    return m.to(dtype).eval()


def _diffusers(facade, dtype=torch.float32):
    diffusers = pytest.importorskip("diffusers")
    cfg = {k: facade.config[k] for k in ("in_channels", "dim", "n_layers", "n_refiner_layers", "n_heads", "n_kv_heads",
                                         "norm_eps", "qk_norm", "cap_feat_dim", "rope_theta", "t_scale")}
    d = diffusers.ZImageTransformer2DModel(**cfg, axes_dims=list(facade.config["axes_dims"]),
                                           axes_lens=list(facade.config["axes_lens"]))
    d.load_state_dict(facade.state_dict(), strict=True)
    return d.to(dtype).eval()


def _batch(dtype=torch.float32, seed=3):
    g = torch.Generator().manual_seed(seed)
    lens = [7, 40]
    latent = torch.randn(2, 16, LAT_H, LAT_W, generator=g).to(dtype)
    caps = [torch.randn(n, 32, generator=g).to(dtype) for n in lens]
    embeds = torch.zeros(2, 40, 32, dtype=dtype)
    mask = torch.zeros(2, 40, dtype=torch.bool)
    for i, c in enumerate(caps):
        embeds[i, :len(c)] = c
        mask[i, :len(c)] = True
    return latent, caps, embeds, mask


def _reference(d, latent, caps, sigma):
    """What the diffusers pipeline does: images as [C, 1, H, W], t = 1 - sigma, captions as a list, output negated."""
    t = torch.full((latent.shape[0],), 1.0 - sigma, dtype=torch.float32)
    with torch.no_grad():
        out = d([x.unsqueeze(1) for x in latent], t, list(caps)).sample
    return M.patchify(-torch.stack([o.squeeze(1) for o in out]))


@pytest.mark.parametrize("sigma", [0.02, 0.3, 0.75, 0.98])
def test_parity_with_diffusers_fp32(sigma):
    torch.manual_seed(0)
    f = _facade()
    d = _diffusers(f)
    latent, caps, embeds, mask = _batch()
    with torch.no_grad():
        got = f(M.patchify(latent), torch.tensor(sigma * 1000.0), embeds, mask)
    ref = _reference(d, latent, caps, sigma)
    assert got.shape == ref.shape == (2, 64, LAT_H // 2, LAT_W // 2)
    for i in range(2):  # row by row
        torch.testing.assert_close(got[i], ref[i], atol=1e-4, rtol=1e-4)
    assert ref.abs().mean() > 1e-3  # the comparison is not between two near-zero outputs


def test_parity_depends_on_caption_length_rope():
    """The image RoPE starts at the padded caption length: the same latent with a different caption length differs."""
    f = _facade()
    latent, _, embeds, mask = _batch()
    emb = embeds[:1]
    m7, m40 = mask[:1], torch.ones(1, 40, dtype=torch.bool)
    with torch.no_grad():
        a = f(M.patchify(latent[:1]), torch.tensor(500.0), emb, m7)
        b = f(M.patchify(latent[:1]), torch.tensor(500.0), emb, m40)
    assert not torch.allclose(a, b, atol=1e-4)


def test_pad_tokens_matter_and_batch_rows_are_independent():
    f = _facade()
    latent, caps, embeds, mask = _batch()
    with torch.no_grad():
        both = f(M.patchify(latent), torch.tensor([300.0, 700.0]), embeds, mask)
        one = f(M.patchify(latent[1:]), torch.tensor(700.0), embeds[1:], mask[1:])
        first = f(M.patchify(latent[:1]), torch.tensor(300.0), embeds[:1, :7], mask[:1, :7])
    torch.testing.assert_close(both[1:], one, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(both[:1], first, atol=1e-4, rtol=1e-4)


def test_parity_with_diffusers_bf16_near_zero_sigma():
    torch.manual_seed(0)
    f = _facade(torch.bfloat16)
    d = _diffusers(f, torch.bfloat16)
    latent, caps, embeds, mask = _batch(torch.bfloat16)
    sigma = 0.02
    with torch.no_grad():
        got = f(M.patchify(latent), torch.tensor(sigma * 1000.0), embeds, mask).float()
    ref = _reference(d, latent, caps, sigma).float()
    scale = ref.abs().mean()
    assert (got - ref).abs().mean() < 0.05 * scale
    # a timestep rounded to bf16 first (1 - 0.02 -> 0.98 is not representable) moves the output visibly further
    with torch.no_grad():
        t = (1.0 - torch.tensor(sigma)).to(torch.bfloat16) * 1000.0
        wrong = f(M.patchify(latent), 1000.0 - t.float(), embeds, mask).float()
    assert (got - ref).abs().mean() <= (wrong - ref).abs().mean() + 1e-6


def test_patchify_round_trip_and_channel_order():
    x = torch.randn(2, 16, 8, 12)
    p = M.patchify(x)
    assert p.shape == (2, 64, 4, 6)
    torch.testing.assert_close(M.unpatchify(p), x)
    for c, ph, pw in [(0, 0, 0), (3, 1, 0), (15, 1, 1), (7, 0, 1)]:
        torch.testing.assert_close(p[:, c * 4 + ph * 2 + pw], x[:, c, ph::2, pw::2])
    with pytest.raises(ValueError):
        M.patchify(torch.randn(1, 16, 7, 8))


def _to_comfy_state(diff_state, prefix="model.diffusion_model."):
    """Build a ComfyUI-format dict from a diffusers one independently of single_file (the reference conversion)."""
    out = {}
    for k, v in diff_state.items():
        if ".attention.to_q." in k:
            base = k.replace(".to_q.", ".qkv.")
            kk, vv = k.replace(".to_q.", ".to_k."), k.replace(".to_q.", ".to_v.")
            out[prefix + base] = torch.cat([v, diff_state[kk], diff_state[vv]], 0)
            continue
        if ".attention.to_k." in k or ".attention.to_v." in k:
            continue
        k = (k.replace(".to_out.0.", ".out.").replace(".norm_q.", ".q_norm.").replace(".norm_k.", ".k_norm.")
             .replace("all_x_embedder.2-1.", "x_embedder.").replace("all_final_layer.2-1.", "final_layer."))
        out[prefix + k] = v
    return out


@pytest.mark.parametrize("prefix", ["model.diffusion_model.", "diffusion_model.", ""])
def test_comfy_round_trip_and_strict_load(prefix):
    f = _facade()
    state = {k: v.clone() for k, v in f.state_dict().items()}
    comfy = _to_comfy_state(state, prefix)
    assert any(k.endswith("attention.qkv.weight") for k in comfy)
    assert not any(".to_q." in k for k in comfy)
    conv = SF.comfy_to_facade(comfy)
    assert set(conv) == set(state)
    for k in state:
        torch.testing.assert_close(conv[k], state[k])
    fresh = M.ZImageTransformer(**TINY).eval()
    fresh.load_state_dict(conv, strict=True)
    latent, _, embeds, mask = _batch()
    with torch.no_grad():
        a = f(M.patchify(latent), torch.tensor(400.0), embeds, mask)
        b = fresh(M.patchify(latent), torch.tensor(400.0), embeds, mask)
    torch.testing.assert_close(a, b)
    back = SF.facade_to_comfy(conv)
    assert set(back) == {k[len(prefix):] for k in comfy}
    for k, v in comfy.items():
        torch.testing.assert_close(back[k[len(prefix):]], v)
    # diffusers names pass through unchanged
    same = SF.comfy_to_facade(state)
    assert set(same) == set(state)


def test_comfy_biases_and_missing_part():
    st = {"layers.0.attention.qkv.bias": torch.arange(12.0), "layers.0.attention.qkv.weight": torch.randn(12, 4)}
    conv = SF.comfy_to_facade(st)
    assert set(conv) == {f"layers.0.attention.to_{w}.{k}" for w in "qkv" for k in ("weight", "bias")}
    torch.testing.assert_close(conv["layers.0.attention.to_v.bias"], torch.arange(8.0, 12.0))
    torch.testing.assert_close(SF.facade_to_comfy(conv)["layers.0.attention.qkv.bias"], torch.arange(12.0))
    conv.pop("layers.0.attention.to_k.weight")
    with pytest.raises(KeyError):
        SF.facade_to_comfy(conv)


def test_strict_error_lists_missing_and_unexpected(tmp_path):
    from safetensors.torch import save_file
    f = _facade()
    comfy = _to_comfy_state(f.state_dict())
    path = tmp_path / "z.safetensors"
    save_file({k: v.contiguous() for k, v in comfy.items()}, str(path))
    state = SF.load_state(path, expected_keys=f.state_dict().keys())
    assert set(state) == set(f.state_dict())
    del comfy["model.diffusion_model.layers.1.attention.qkv.weight"]
    comfy["model.diffusion_model.bogus.weight"] = torch.zeros(1)
    save_file({k: v.contiguous() for k, v in comfy.items()}, str(path))
    with pytest.raises(ValueError) as e:
        SF.load_state(path, expected_keys=f.state_dict().keys())
    msg = str(e.value)
    assert "layers.1.attention.to_q.weight" in msg and "bogus.weight" in msg
    SF.load_state(path, expected_keys=f.state_dict().keys(), strict=False)
    assert SF.check_keys({"a": 1}, ["a", "b"], strict=False) == (["b"], [])


def _shapes(cap_dim=2560, extra=()):
    s = {"cap_pad_token": (1, 3840), "context_refiner.0.attention.to_q.weight": (3840, 3840),
         "cap_embedder.0.weight": (cap_dim,), "cap_embedder.1.weight": (3840, cap_dim), "layers.0.attention.qkv.weight": (1, 1)}
    s.update({k: (1,) for k in extra})
    return s


def test_zimage_kind():
    assert SF.zimage_kind(_shapes()) == "zimage"
    assert SF.zimage_kind({"model.diffusion_model." + k: v for k, v in _shapes().items()}) == "zimage"
    assert SF.zimage_kind(_shapes(2304)) is None  # Lumina2 reads Gemma
    lumina = {k: v for k, v in _shapes(2304).items() if k != "cap_pad_token"}
    assert SF.zimage_kind(lumina) is None
    assert SF.zimage_kind(_shapes(extra=["siglip_embedder.1.weight"])) == "zimage_omni"
    assert SF.zimage_kind({"double_blocks.0.img_attn.qkv.weight": (3, 3)}) is None
    assert SF.zimage_kind({k: v for k, v in _shapes().items() if not k.startswith("context")}) is None
    f = _facade(cap_feat_dim=2560, dim=64)  # real cap dim on a small model: names and shapes decide
    shapes = {k: tuple(v.shape) for k, v in f.state_dict().items()}
    assert SF.zimage_kind(shapes) == "zimage"
    assert SF.zimage_kind({k: tuple(v.shape) for k, v in _facade().state_dict().items()}) is None  # 32 wide caption


def test_gradient_checkpointing_same_output_and_grads():
    latent, _, embeds, mask = _batch()
    lat = M.patchify(latent)
    results = []
    for ckpt in (False, True):
        f = _facade()
        f.train()
        f.gradient_checkpointing = ckpt
        x = lat.clone().requires_grad_(True)
        out = f(x, torch.tensor(600.0), embeds, mask)
        out.pow(2).mean().backward()
        grads = {n: p.grad.clone() for n, p in f.named_parameters() if p.grad is not None}
        results.append((out.detach(), x.grad.clone(), grads))
    (o0, g0, p0), (o1, g1, p1) = results
    torch.testing.assert_close(o0, o1)
    torch.testing.assert_close(g0, g1, atol=1e-6, rtol=1e-5)
    assert set(p0) == set(p1) and len(p0) > 10
    for n in p0:
        torch.testing.assert_close(p0[n], p1[n], atol=1e-6, rtol=1e-5)
    assert any(n.startswith("layers.1.") for n in p0)


def test_skip_layers_skips_a_block():
    f = _facade()
    latent, _, embeds, mask = _batch()
    lat = M.patchify(latent)
    seen = []
    hooks = [blk.register_forward_hook(lambda m, i, o, n=n: seen.append(n)) for n, blk in enumerate(f.layers)]
    with torch.no_grad():
        full = f(lat, torch.tensor(500.0), embeds, mask)
        assert seen == [0, 1]
        seen.clear()
        part = f(lat, torch.tensor(500.0), embeds, mask, skip_layers=[1])
    assert seen == [0]
    assert not torch.allclose(full, part)
    for h in hooks:
        h.remove()


def test_vector_timestep_and_no_mask():
    f = _facade()
    latent, _, embeds, _ = _batch()
    lat = M.patchify(latent)
    with torch.no_grad():
        a = f(lat, torch.tensor([250.0, 250.0]), embeds, None)
        b = f(lat, torch.tensor(250.0), embeds, torch.ones(2, 40, dtype=torch.bool))
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError):
        f(lat[:, :32], torch.tensor(250.0), embeds, None)


def test_zero_arg_construction_on_meta():
    with torch.device("meta"):
        m = M.ZImageTransformer()
    assert len(m.layers) == 30 and m.latent_channels == 64 and m.text_dim == 2560
    assert len(m.noise_refiner) == len(m.context_refiner) == 2
    assert m.gradient_checkpointing is False
    assert M.ZIMAGE_CONFIG["dim"] == 3840 and M.ZIMAGE_CONFIG["n_heads"] == 30
    keys = set(m.state_dict())
    for k in ("layers.0.attention.to_q.weight", "layers.29.attention.to_out.0.weight", "layers.0.attention.norm_q.weight",
              "layers.0.feed_forward.w1.weight", "layers.0.adaLN_modulation.0.bias", "noise_refiner.1.adaLN_modulation.0.weight",
              "context_refiner.0.attention_norm1.weight", "cap_embedder.1.weight", "t_embedder.mlp.2.bias",
              "all_x_embedder.2-1.weight", "all_final_layer.2-1.adaLN_modulation.1.weight", "x_pad_token",
              "cap_pad_token"):
        assert k in keys, k
    assert not any(k.startswith("context_refiner.") and "adaLN" in k for k in keys)
    # the real parameter count of the released model (6.15B)
    assert abs(sum(p.numel() for p in m.parameters()) / 1e9 - 6.15) < 0.05


def test_full_precision_and_resident_lists():
    assert M.matches("t_embedder.mlp.0", M.FULL_PRECISION) and M.matches("cap_pad_token", M.FULL_PRECISION)
    assert M.matches("all_x_embedder.2-1", M.FULL_PRECISION)
    assert M.matches("layers.3.attention_norm1", M.FULL_PRECISION)
    assert not M.matches("layers.3.attention.to_q", M.FULL_PRECISION)
    assert not M.matches("layers.3.feed_forward.w1", M.FULL_PRECISION)
    for name in ("x_pad_token", "cap_pad_token", "noise_refiner.0.attention.to_q", "context_refiner.1", "t_embedder"):
        assert M.matches(name, M.RESIDENT)
    assert not M.matches("layers.0.attention.to_q", M.RESIDENT)
