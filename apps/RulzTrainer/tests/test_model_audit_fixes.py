"""Fixes from the per-model audit, across the families: prompt_tokens cuts a caption where its cache was cut (so a
trigger's token positions stay inside the cached rows), every text encoder runs its quantized layers weight-only, a
transformer loaded into RAM for block swap is quantized on the training GPU, and MOSS-Image's reference sigmas stay
truncated and rescaled when the first draw is topped up. CPU only, tiny random models, no downloads."""
import sys
import threading
import types
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rulztrainer import archs  # noqa: E402
from rulztrainer import quant as Q  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402
from rulztrainer.moss import single_file as msf  # noqa: E402
from test_text_batching import _qwen3vl, _qwen35  # noqa: E402

LONG = "zxc woman, " + "a long caption about the light on the water " * 40


class _Tok:
    """One id per character, offsets per character, a chat template and pad / eos ids: enough for every family's
    encode_prompts and prompt_tokens."""

    pad_token_id = 0
    eos_token_id = 1

    def __len__(self):
        return 248_320

    def __call__(self, text=None, add_special_tokens=False, truncation=False, max_length=None, return_tensors=None,
                 return_offsets_mapping=False, padding=None, images=None):
        texts = text if isinstance(text, list) else [text]
        rows = [[3 + (ord(c) % 90) for c in t] for t in texts]
        offs = [[(i, i + 1) for i in range(len(t))] for t in texts]
        if truncation and max_length is not None:
            rows, offs = [r[:max_length] for r in rows], [o[:max_length] for o in offs]
        out = type("T", (dict,), {})()
        ids = torch.tensor(rows) if return_tensors == "pt" else (rows if isinstance(text, list) else rows[0])
        out.input_ids = out["input_ids"] = ids
        if return_offsets_mapping:
            out["offset_mapping"] = offs if isinstance(text, list) else offs[0]
        return out

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        return "<u>" + messages[0]["content"] + "</u><a>"


def _qwen3(layers: int):
    from transformers import Qwen3Config, Qwen3Model

    from rulztrainer.flux2 import single_file as fsf

    torch.manual_seed(0)
    return Qwen3Model(Qwen3Config(**{**fsf.TEXT_CONFIG, "vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
                                     "num_hidden_layers": layers, "num_attention_heads": 2, "num_key_value_heads": 2,
                                     "head_dim": 8})).eval()


def _families():
    """(name, loading module, tiny encoder, axis of the token rows in a cached feature)."""
    from rulztrainer.fibo import loading as fibo
    from rulztrainer.flux2 import loading as flux2
    from rulztrainer.flux3 import loading as flux3
    from rulztrainer.h3 import loading as h3
    from rulztrainer.ideogram4 import loading as ideogram4
    from rulztrainer.krea2 import loading as krea2
    from rulztrainer.moss import loading as moss
    from rulztrainer.pan2 import loading as pan2
    from rulztrainer.qwen21 import loading as qwen21

    return [("moss", moss, _qwen35, 0), ("pan2", pan2, _qwen35, 0), ("krea2", krea2, lambda: _qwen3vl(36), 0),
            ("qwen21", qwen21, lambda: _qwen3vl(2), 0), ("h3", h3, lambda: _qwen3vl(2), 0),
            ("ideogram4", ideogram4, lambda: _qwen3vl(36), 0), ("flux3", flux3, lambda: _qwen3vl(33), 0),
            ("fibo", fibo, lambda: _qwen3vl(36), 1), ("flux2", flux2, lambda: _qwen3(28), 0)]


# ---------------------------------------------------------------- prompt_tokens against the cache


@pytest.mark.parametrize("name", [f[0] for f in _families()])
def test_prompt_tokens_cut_where_the_cache_cuts(name):
    """A caption longer than max_token_length is cut by encode_prompts; prompt_tokens has to cut it at the same token
    and give the cached length, or the trainer's guard drops every trigger position of that caption (and a position
    past the cut would point at a padding row or past the cached features)."""
    from rulztrainer import alignment as A

    _, mod, make, axis = next(f for f in _families() if f[0] == name)
    model, tok = make(), _Tok()
    for max_length in (64, 300, 1024):
        with torch.no_grad():
            cached = mod.encode_prompts(model, tok, [LONG, "zxc woman"], max_length)
        for prompt, row in zip([LONG, "zxc woman"], cached):
            text, offsets, drop, length = mod.prompt_tokens(tok, prompt, max_length)
            assert length == row.shape[axis], (name, max_length, length, row.shape)
            positions = A.token_positions(text, offsets, ["zxc"], drop)
            assert all(p < length for p in positions), (name, max_length)
            # (at 64 tokens PAN 2's and Ideogram 4's JSON wrapping can end before the caption starts)
            assert positions or max_length < 300, (name, max_length)
            # the offsets end where the encoded prompt ends: no token past the cut is offered as a position
            assert len(offsets) - drop <= length


def test_arch_prompt_tokens_take_the_runs_max_length():
    """Arch.prompt_tokens passes the run's max_token_length on; without one it is the config's default (1024), which
    MOSS-Image's prompt_tokens used to ignore (it cut at 4096 while the cache cut at 1024)."""
    tok = _Tok()
    moss = archs.arch_by_name("moss_image")
    assert moss.prompt_tokens(tok, LONG * 3)[3] == TrainConfig.max_token_length == 1024
    assert moss.prompt_tokens(tok, LONG * 3, 200)[3] == 200
    pan2 = archs.arch_by_name("pan2")
    _, _, drop, kept = pan2.prompt_tokens(tok, LONG * 3)
    assert kept == 1536  # PAN 2 reads its full 1536 at the default max_token_length
    assert pan2.prompt_tokens(tok, LONG * 3, 700)[3] == 700


# ---------------------------------------------------------------- quantized text encoders run weight-only


def _quantize_two(state: dict, model: nn.Module) -> None:
    """Replace the weights of the first two Linear layers inside the decoder with an int8 and an nvfp4 QuantWeight,
    as a quantized single file holds them ("auto" keeps their formats)."""
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.startswith("layers.0.")][:2]
    for name, fmt in zip(names, ("int8", "nvfp4")):
        state[f"{name}.weight"] = Q.quantize_weight(state[f"{name}.weight"].float(), fmt)


def _off(layer: Q.QuantLinear) -> bool:
    """No int8 activations (W8A8) and, for an nvfp4 layer, no FP4 activations (an int8 layer never reads its nvfp4
    flag)."""
    return not layer.int8_matmul and not (layer.fmt == "nvfp4" and layer.nvfp4_matmul)


def _assert_weight_only(model: nn.Module) -> None:
    layers = [m for m in model.modules() if isinstance(m, Q.QuantLinear)]
    assert {m.fmt for m in layers} == {"int8", "nvfp4"}
    assert all(_off(m) for m in layers)


def _qwen35_config(tmp_path: Path) -> Path:
    tiny = _qwen35()
    (tmp_path / "qwen35_9b_text_config.json").write_text(tiny.config.to_json_string(), encoding="utf-8")
    return tmp_path


@pytest.fixture
def w8a8_defaults(monkeypatch):
    """The trainer makes W8A8 and native FP4 the default of every QuantLinear built after it reads the config,
    before the captions are cached."""
    monkeypatch.setattr(Q, "DEFAULT_INT8_MATMUL", True)
    monkeypatch.setattr(Q, "DEFAULT_NVFP4_MATMUL", True)


def test_moss_and_pan2_text_encoders_run_weight_only(monkeypatch, tmp_path, w8a8_defaults):
    from rulztrainer.pan2 import single_file as psf

    here = _qwen35_config(tmp_path)
    monkeypatch.setattr(msf, "HERE", here)
    monkeypatch.setattr(psf, "MOSS_HERE", here)
    for build in (msf.build_text_model, psf.build_text_model):
        tiny = _qwen35()
        state = {k: v.to(torch.bfloat16) for k, v in tiny.state_dict().items()}
        _quantize_two(state, tiny)
        _assert_weight_only(build(state, "auto"))


def test_qwen3vl_text_encoders_run_weight_only(w8a8_defaults):
    """krea2.single_file.build_text_model builds every Qwen3-VL text encoder: Krea 2, FIBO 2.0, FLUX 3, P-H3-Image,
    Qwen-Image 2.1 and Ideogram 4."""
    from rulztrainer.krea2 import single_file as ksf

    for layers in (None, 1):
        tiny = _qwen3vl(2)
        state = {k: v.to(torch.bfloat16) for k, v in tiny.state_dict().items()}
        _quantize_two(state, tiny)
        _assert_weight_only(ksf.build_text_model(state, "auto", config=tiny.config.to_dict(), layers=layers))


def test_flux2_text_encoder_runs_weight_only(monkeypatch, w8a8_defaults):
    from rulztrainer.flux2 import single_file as fsf

    tiny = _qwen3(3)
    monkeypatch.setattr(fsf, "TEXT_CONFIG", tiny.config.to_dict())
    state = {f"model.{k}": v.to(torch.bfloat16) for k, v in tiny.state_dict().items()}
    names = [n for n, m in tiny.named_modules() if isinstance(m, nn.Linear) and n.startswith("layers.0.")][:2]
    for name, fmt in zip(names, ("int8", "nvfp4")):
        state[f"model.{name}.weight"] = Q.quantize_weight(state[f"model.{name}.weight"].float(), fmt)
    _assert_weight_only(fsf.build_text_model(state, "auto"))


def test_weight_only_turns_off_both_kinds_of_low_precision_activations(w8a8_defaults):
    layers = nn.Sequential(Q.QuantLinear(Q.quantize_weight(torch.randn(32, 64), "int8"), None),
                           Q.QuantLinear(Q.quantize_weight(torch.randn(32, 64), "nvfp4"), None))
    assert layers[0].int8_matmul and layers[1].nvfp4_matmul
    assert msf.weight_only(layers) is layers
    assert all(_off(m) for m in layers)


# ---------------------------------------------------------------- block swap: quantized on the training GPU


def test_quant_target_inside_the_staging_block(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert msf.quant_target("cpu") == {}
    with msf.quantize_staged_on("cuda:1"):
        assert msf.quant_target("cpu") == {"work_device": torch.device("cuda:1")}  # back to the CPU afterwards
        assert msf.quant_target("cpu", ["cuda:0"]) == {"work_device": torch.device("cuda:0")}  # a split: its GPUs
        assert msf.quant_target("cuda:0") == {"work_device": torch.device("cuda:0"), "keep_on_work_device": True}
        assert msf.quant_target("auto") == {"work_device": torch.device("cuda:0")}
        seen = []
        other = threading.Thread(target=lambda: seen.append(msf.quant_target("cpu")))
        other.start()
        other.join()
        assert seen == [{}]  # another thread's model is not affected
        with msf.quantize_staged_on(None):
            assert msf.quant_target("cpu") == {}
        assert msf.quant_target("cpu") == {"work_device": torch.device("cuda:1")}
    assert msf.quant_target("cpu") == {}


def test_staging_gpu_only_for_a_model_in_ram_of_a_gpu_run():
    gpu = TrainConfig(device="cuda:1")
    assert msf.staging_gpu(gpu, "cpu") == "cuda:1"
    assert msf.staging_gpu(gpu, None) is None and msf.staging_gpu(gpu, "cuda:0") is None
    assert msf.staging_gpu(TrainConfig(device="cpu"), "cpu") is None


def _loader_patches():
    """(arch name, module, attribute) of the loader each family's get_transformer calls for a single file."""
    import rulztrainer.fibo.single_file as fibo
    import rulztrainer.flux2.single_file as flux2
    import rulztrainer.flux3.single_file as flux3
    import rulztrainer.h3.single_file as h3
    import rulztrainer.h3image.loading as h3image
    import rulztrainer.ideogram4.single_file as ideogram4
    import rulztrainer.krea2.single_file as krea2
    import rulztrainer.ph3.loading as ph3
    import rulztrainer.qwen21.single_file as qwen21

    return [("moss_image", msf, "load_transformer_file"), ("fibo2", fibo, "load_transformer_file"),
            ("krea2", krea2, "load_transformer_file"), ("qwen_image21", qwen21, "load_transformer_file"),
            ("flux2_klein", flux2, "load_transformer_file"), ("flux3_klein", flux3, "load_transformer_file"),
            ("ideogram4", ideogram4, "load_transformer_file"), ("minimax_h3", h3, "load_transformer_file"),
            ("minimax_h3_image", h3image, "load_transformer_file"), ("p_h3_image", ph3, "load_transformer_file")]


@pytest.mark.parametrize("arch_name", [p[0] for p in _loader_patches()])
def test_block_swap_quantizes_on_the_training_gpu(monkeypatch, tmp_path, arch_name):
    """Block swap loads the transformer with device "cpu": it is quantized on the training GPU and comes back to the
    CPU (where the swapped blocks stay), as PAN 2 already did. The loader still gets device "cpu" and no GPUs of its
    own, so quant "auto" sizes the format exactly as before; a model loaded straight to the GPU is unchanged."""
    _, module, attr = next(p for p in _loader_patches() if p[0] == arch_name)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seen = []

    def loader(path, device, quant="auto", *args, **kwargs):
        quant_devices = kwargs.get("quant_devices", args[-1] if args else None)
        seen.append((str(device), quant_devices, msf.quant_target(device)))
        return types.SimpleNamespace(is_distilled=False, flow_map=False)

    monkeypatch.setattr(module, attr, loader)
    if arch_name == "minimax_h3":
        import rulztrainer.h3.loading as hl

        monkeypatch.setattr(hl, "attach_silence", lambda model, cfg, model_dir: model)
    weights = tmp_path / "transformer.safetensors"
    weights.write_bytes(b"")
    cfg = TrainConfig(model_arch=arch_name, model_source="files", transformer_file=str(weights), device="cuda:1",
                      transformer_quant="nvfp4")
    arch = archs.arch_by_name(arch_name)
    arch.get_transformer(cfg, None, device="cpu")
    assert seen == [("cpu", None, {"work_device": torch.device("cuda:1")})]
    assert msf.quant_target("cpu") == {}  # only while it loads
    seen.clear()
    arch.get_transformer(cfg, None, device="cuda:1")
    assert seen == [("cuda:1", None, {"work_device": torch.device("cuda:1"), "keep_on_work_device": True})]


# ---------------------------------------------------------------- MOSS-Image reference sigmas


def test_moss_reference_sigmas_top_up_is_truncated_and_rescaled(monkeypatch):
    """S = 1 - sigmoid(Z) is truncated to S <= 0.95 and rescaled by 1 / 0.95; a draw that is topped up has to be too
    (it once appended raw values, up to 1.0, unscaled)."""
    import math

    def z_for(s):  # the randn value that gives sigma s: z = r * 0.95 - 1.09, s = sigmoid(-z)
        return (-math.log(s / (1 - s)) + 1.09) / 0.95

    calls = []

    def fake_randn(n, generator=None):
        calls.append(n)
        if len(calls) == 1:  # every sigma above the truncation: the first draw keeps none
            return torch.full((n,), z_for(0.99))
        return torch.tensor([z_for(0.5), z_for(0.99)] * (n // 2))

    monkeypatch.setattr(archs.torch, "randn", fake_randn)
    monkeypatch.setattr(archs.torch, "rand", lambda n, generator=None: torch.ones(n))  # no uniform 1% draws
    s = archs.arch_by_name("moss_image").reference_sigmas(8, torch.Generator(), 4096)
    assert len(calls) == 2 and s.shape == (8,)
    assert torch.allclose(s, torch.full((8,), 0.5 / 0.95))
