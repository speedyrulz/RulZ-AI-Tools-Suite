"""Memory savings in the model adapters that must not change results: the trigger-attention map computed in chunks of
heads, quantization on load that consumes its state dict (optionally on a GPU), "auto" quant over a model split, the
bf16 VAE copies, meta-device VAE loads, Krea 2's grouped-query SDPA and the text-only HF text encoders."""
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import alignment as A  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402
from rulztrainer.moss import single_file as sf  # noqa: E402

diffusers = pytest.importorskip("diffusers")


def _reference_attention(q, k, mask=None):
    """text_attention as it was: all heads at once."""
    logits = torch.matmul(q, k.transpose(-1, -2)).float() * (q.shape[-1] ** -0.5)
    if mask is not None:
        logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
    return logits.softmax(-1).mean(1)


@pytest.mark.parametrize("chunk", [1, 7 * 5 * 3, 1 << 23])  # one head at a time, a few, all at once
def test_chunked_text_attention_matches_all_heads_at_once(chunk):
    torch.manual_seed(0)
    q = torch.randn(2, 6, 7, 16, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 6, 5, 16, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    w = torch.randn(2, 7, 5, dtype=torch.float64)
    ref = _reference_attention(q, k, mask)
    gq, gk = torch.autograd.grad((ref * w).sum(), (q, k))
    got = A.text_attention(q, k, mask, chunk_elements=chunk)
    hq, hk = torch.autograd.grad((got * w).sum(), (q, k))
    assert torch.allclose(got, ref, atol=1e-12) and torch.allclose(hq, gq, atol=1e-12)
    assert torch.allclose(hk, gk, atol=1e-12)
    with torch.no_grad():
        assert torch.allclose(A.text_attention(q, k, mask, chunk_elements=chunk), ref, atol=1e-12)


def test_chunked_text_attention_inside_a_checkpointed_block():
    """A tapped block runs under non-reentrant checkpointing: the chunks' own checkpoints nest inside it."""
    torch.manual_seed(0)
    x = torch.randn(1, 9, 32, dtype=torch.float64, requires_grad=True)
    proj = torch.nn.Linear(32, 64, dtype=torch.float64)

    def block(x, chunk):
        q, k = proj(x).unflatten(-1, (8, 8)).transpose(1, 2).chunk(2, dim=1)  # 4 heads each
        return A.text_attention(q[:, :, :6], k[:, :, 6:], chunk_elements=chunk).pow(2).sum()

    ref = torch.autograd.grad(block(x, 1 << 23), x)[0]
    got = torch.autograd.grad(checkpoint(block, x, 1, use_reentrant=False), x)[0]
    assert torch.allclose(got, ref, atol=1e-12)


def test_an_unrecording_tap_skips_forwards_without_gradients():
    taps = A.Taps(attn_layers=[0])
    tap = A.AttnTap(taps, 0)
    taps.begin([[0]])
    taps.set_layout(0, (0, 3), (3, -1))
    calls = []
    real = A.text_attention
    try:
        A.text_attention = lambda *a, **kw: calls.append(1) or real(*a, **kw)
        q, k = torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
        tap(q, k)  # the main forward records
        taps.stop()
        with torch.no_grad():
            tap(q, k)  # the prior pass: nothing to record or recompute
        tap(q.requires_grad_(), k)  # a checkpointed block's recompute still computes
    finally:
        A.text_attention = real
    assert len(calls) == 2 and set(taps.attn[0]) == {0}


def test_fibo_auto_quant_counts_the_split_gpus(monkeypatch):
    from rulztrainer.fibo import loading as FL

    sizes = {0: 16, 1: 16, 2: 24}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(FL, "_gpu_gb", lambda d: sizes[torch.device(d).index or 0])
    assert FL.resolve_quant("auto", "cuda:0") == "int8"  # one 16 GB card
    assert FL.resolve_quant("auto", ["cuda:0", "cuda:1"]) == "none"  # bf16 over two 16 GB cards
    assert FL.resolve_quant("auto", "cuda:2") == "none" and FL.resolve_quant("auto", "cpu") == "none"
    assert FL.resolve_quant("nvfp4", ["cuda:0", "cuda:1"]) == "nvfp4"


def test_quant_target_by_where_the_model_goes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert sf.quant_target("cuda:1") == {"work_device": torch.device("cuda:1"), "keep_on_work_device": True}
    assert sf.quant_target("cpu", ["cuda:0", "cuda:1"]) == {"work_device": torch.device("cuda:0")}  # a split
    assert sf.quant_target("cpu") == {}  # block swap / a CPU model: as before
    assert sf.quant_target("auto") == {"work_device": torch.device("cuda:0")}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert sf.quant_target("cuda:0") == {}


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(256, 512)
        self.b = torch.nn.Linear(512, 256, bias=False)
        self.small = torch.nn.Linear(256, 8)


def _state():
    torch.manual_seed(0)
    return {k: v.to(torch.bfloat16) for k, v in _Net().state_dict().items()}


@pytest.mark.parametrize("fmt", ["int8", "int8_convrot", "nvfp4"])
def test_install_state_consumes_the_dict(fmt):
    state = _state()
    with torch.device("meta"):
        model = _Net()
    sf.install_state(model, state, fmt, min_dim=256)
    assert not any(k.startswith(("a.", "b.")) for k in state)  # the Linear weights left the caller's dict
    assert isinstance(model.a, Q.QuantLinear) and isinstance(model.b, Q.QuantLinear)
    assert isinstance(model.small, torch.nn.Linear) and model.small.weight.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("fmt", ["int8", "int8_convrot", "nvfp4"])
@pytest.mark.parametrize("keep", [False, True])
def test_quantizing_on_the_gpu_matches_the_cpu(fmt, keep):
    def build(**kw):
        with torch.device("meta"):
            model = _Net()
        sf.install_state(model, _state(), fmt, min_dim=256, **kw)
        return model

    cpu, gpu = build(), build(work_device="cuda:0", keep_on_work_device=keep)
    assert gpu.a.qdata.device.type == ("cuda" if keep else "cpu")
    for name in ("a", "b"):
        x = getattr(cpu, name).qw.dequantize(torch.float32)
        y = sf._qw_to(getattr(gpu, name).qw, "cpu").dequantize(torch.float32)
        # the same codes but where CUDA's division (a multiply by the reciprocal) lands a value on the other side
        # of a rounding tie: one code step on a few elements in 10^4
        off = (x - y).abs() > x.abs().max() * 1e-6
        assert off.float().mean() < 1e-3, f"{name}: {off.float().mean():.2e} of the weights differ"


def _flux2_vae():
    torch.manual_seed(0)
    return diffusers.AutoencoderKLFlux2(block_out_channels=(32, 64, 64, 64), decoder_block_out_channels=(32, 32, 64, 64),
                                        latent_channels=32, norm_num_groups=32, layers_per_block=1,
                                        sample_size=64).eval()


def test_bf16_copy_casts_the_weights_and_keeps_the_latent_statistics():
    from rulztrainer.moss import loading as M

    vae = _flux2_vae()
    vae.bn.running_var.fill_(1.2345678)
    copy = M.bf16_copy(vae)
    assert copy is not vae and copy.dtype == torch.bfloat16 and not any(p.requires_grad for p in copy.parameters())
    assert copy.bn.running_var.dtype == torch.float32 and float(copy.bn.running_var[0]) == pytest.approx(1.2345678)
    for (n, p), (_, q) in zip(vae.named_parameters(), copy.named_parameters()):
        assert torch.equal(p.to(torch.bfloat16), q), n
    assert vae.dtype == torch.float32  # the shared fp32 VAE is untouched
    lat = torch.randn(1, 128, 2, 2)
    with torch.no_grad():
        full = M._decode(vae, lat).clamp(-1, 1)
    half = M._decode(copy, lat).float().clamp(-1, 1)
    assert (full - half).abs().mean() < 0.02


def test_flux2_vae_file_loads_on_meta_without_num_batches_tracked(tmp_path):
    from safetensors.torch import save_file

    vae = _flux2_vae()
    state = {k: v.contiguous() for k, v in vae.state_dict().items() if "num_batches_tracked" not in k}
    path = tmp_path / "flux2_vae.safetensors"
    save_file(state, str(path))
    loaded = sf.load_vae_file(path, "cpu")
    assert all(t.device.type == "cpu" for t in (*loaded.parameters(), *loaded.buffers()))
    assert int(loaded.bn.num_batches_tracked) == 0
    assert torch.equal(loaded.decoder.conv_out.weight, vae.decoder.conv_out.weight)


def test_assign_on_meta_builds_no_random_weights_and_names_missing_ones():
    vae = _flux2_vae()
    state = dict(vae.state_dict())
    built = sf.assign_on_meta(_flux2_vae, dict(state), "tiny")
    assert all(torch.equal(a, b) for a, b in zip(built.state_dict().values(), vae.state_dict().values()))
    state.pop("decoder.conv_out.weight")
    with pytest.raises(ValueError, match="VAE state mismatch"):
        sf.assign_on_meta(_flux2_vae, state, "tiny")


def test_krea2_grouped_query_sdpa_matches_repeated_heads():
    from rulztrainer.krea2.model import Attention

    torch.manual_seed(0)
    att = Attention(64, heads=8, kv_heads=2).double().eval()
    x = torch.randn(2, 11, 64, dtype=torch.float64)
    gqa = att(x)  # no mask: grouped heads straight into SDPA
    rep = att(x, mask=torch.ones(2, 1, 1, 11, dtype=torch.bool))  # a mask: the repeated heads
    assert torch.allclose(gqa, rep, atol=1e-12)


def _tiny_qwen3vl(cls_name: str, tmp_path: Path):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3VLConfig(
        text_config=dict(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, vocab_size=512, rms_norm_eps=1e-6,
                         max_position_embeddings=4096, tie_word_embeddings=True,
                         rope_parameters={"mrope_interleaved": True, "mrope_section": [4, 2, 2],
                                          "rope_theta": 10000.0, "rope_type": "default"}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=64,
                           deepstack_visual_indexes=[0]),
        tie_word_embeddings=True)
    torch.manual_seed(0)
    model = getattr(transformers, cls_name)(cfg).eval()
    for p in model.parameters():  # bf16 weights, the rotary tables stay fp32 (as from_pretrained(dtype=bf16) leaves them)
        p.data = p.data.to(torch.bfloat16)
    model.save_pretrained(str(tmp_path / "text_encoder"))
    return model


@pytest.mark.parametrize("cls_name", ["Qwen3VLModel", "Qwen3VLForConditionalGeneration"])  # Krea 2's, FIBO's
def test_hf_text_encoder_loads_text_only_with_the_same_hidden_states(cls_name, tmp_path):
    from rulztrainer.krea2 import single_file as ksf

    full = _tiny_qwen3vl(cls_name, tmp_path)
    text = ksf.load_hf_text_model(tmp_path / "text_encoder", "none", "cpu")
    assert not any(n.startswith(("visual", "lm_head")) for n, _ in text.named_parameters())
    ids = torch.randint(0, 500, (1, 9))
    pos = torch.arange(9).view(1, 1, 9).expand(3, 1, 9)
    with torch.no_grad():
        a = text(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos, output_hidden_states=True)
        b = full(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos, output_hidden_states=True)
        c = text(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)  # FIBO: no positions
        d = full(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    assert len(a.hidden_states) == len(b.hidden_states) == 3
    assert all(torch.equal(x, y) for x, y in zip(a.hidden_states, b.hidden_states))
    assert all(torch.equal(x, y) for x, y in zip(c.hidden_states, d.hidden_states))
