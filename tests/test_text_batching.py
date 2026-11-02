"""Batched text-embedding caching: every family's encode_prompts gives each prompt the features of its own
one-prompt forward, and cache_text batches, counts and recovers from running out of memory."""
import pytest
import torch

from rulztrainer import dataset as D
from rulztrainer.config import TrainConfig

PROMPTS = ["a cat", "a photo of zxc woman standing on a beach at sunset", "", "red bicycle in the rain, close up"]


class _Tok:
    """A stand-in tokenizer: one id per character, a chat template, pad / eos ids."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text=None, add_special_tokens=False, truncation=False, max_length=None, return_tensors=None,
                 return_offsets_mapping=False, padding=None, images=None):
        texts = text if isinstance(text, list) else [text]
        rows = [[3 + (ord(c) % 90) for c in t][: max_length or None] for t in texts]
        out = type("T", (dict,), {})()
        ids = torch.tensor(rows) if return_tensors == "pt" else (rows if isinstance(text, list) else rows[0])
        out.input_ids = out["input_ids"] = ids
        return out

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        return "<u>" + messages[0]["content"] + "</u><a>"


def _qwen3vl(layers: int, seed: int = 0):
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

    torch.manual_seed(seed)
    cfg = Qwen3VLTextConfig(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                            rope_parameters={"rope_type": "default", "rope_theta": 5000000.0,
                                             "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    return Qwen3VLTextModel(cfg).eval()


def _qwen35(seed: int = 0):
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
                            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                            layer_types=["linear_attention"] * 3 + ["full_attention"])
    return Qwen3_5TextModel(cfg).eval()


def _families():
    from rulztrainer.fibo import loading as fibo
    from rulztrainer.flux3 import loading as flux3
    from rulztrainer.h3 import loading as h3
    from rulztrainer.ideogram4 import loading as ideogram4
    from rulztrainer.krea2 import loading as krea2
    from rulztrainer.moss import loading as moss
    from rulztrainer.qwen21 import loading as qwen21

    return [
        ("moss", moss, _qwen35, 512),
        ("krea2", krea2, lambda: _qwen3vl(36), 512),
        ("qwen21", qwen21, lambda: _qwen3vl(2), 1024),
        ("h3", h3, lambda: _qwen3vl(2), 1024),
        ("ideogram4", ideogram4, lambda: _qwen3vl(36), 1024),
        ("flux3", flux3, lambda: _qwen3vl(33), 1024),
        ("fibo", fibo, lambda: _qwen3vl(36), 4096),
    ]


@pytest.mark.parametrize("name", [f[0] for f in _families()])
def test_batched_features_match_one_prompt_forwards(name):
    _, mod, make, max_len = next(f for f in _families() if f[0] == name)
    model = make()
    tok = _Tok()
    with torch.no_grad():
        batched = mod.encode_prompts(model, tok, PROMPTS, max_len)
        single = [mod.encode_prompt(model, tok, p, max_len) for p in PROMPTS]
    assert len(batched) == len(PROMPTS)
    for p, b, s in zip(PROMPTS, batched, single):
        assert b.shape == s.shape, (name, p, b.shape, s.shape)
        assert b.dtype == torch.bfloat16
        # bf16 output; right padding must not reach a row's real tokens (a leak shows as O(1) differences)
        assert torch.allclose(b.float(), s.float(), atol=2e-2, rtol=2e-2), (name, p, float((b - s).abs().max()))
    # rows are their own tensors, not views of the batch (a cached row must not keep the whole batch alive)
    assert all(b.untyped_storage().nbytes() == b.numel() * b.element_size() for b in batched if b.numel())


def test_text_batches_sort_and_cut():
    jobs = [(f"p{i}", "x" * n) for i, n in enumerate((50, 5, 500, 20, 10, 3000))]
    batches = D.text_batches(jobs, size=3, max_chars=1200)
    flat = [len(t) for b in batches for _, t in b]
    assert flat == sorted(flat)  # shortest first: a batch pads to its longest prompt
    assert all(len(b) <= 3 for b in batches)
    assert all(len(b) == 1 or len(b) * (max(len(t) for _, t in b) + 1) <= 1200 for b in batches)
    assert sum(len(b) for b in batches) == len(jobs)
    assert [len(b) for b in batches] == [3, 2, 1]  # 3x51; 2x501 (a third 3001 would pass 1200); the long one alone


def _items(tmp_path, n):
    items = []
    for i in range(n):
        items.append(D.Item(image=tmp_path / f"{i}.png", caption="cap " + "w" * i, width=64, height=64,
                            latent_path=tmp_path / f"{i}.lat", latent_flip_path=None, text_path=tmp_path / "text" / f"{i}.safetensors"))
    return items


def test_cache_text_batches_prompts_and_counts_them(tmp_path):
    from safetensors.torch import load_file

    cfg = TrainConfig(cache_dir=str(tmp_path))
    (tmp_path / "text").mkdir()
    items = _items(tmp_path, 10)
    calls, progress = [], []

    def fake_batch(te, tok, texts, max_len):
        calls.append(list(texts))
        return [torch.full((2, 8), float(len(t)), dtype=torch.bfloat16) for t in texts]

    D.cache_text(items, cfg, "te", "tok", extra_prompts=["probe"], progress=lambda i, n: progress.append((i, n)),
                 encode_batch_fn=fake_batch, batch_size=4)
    assert [len(c) for c in calls] == [4, 4, 4]  # 10 captions + the empty prompt + the probe
    assert progress[-1] == (12, 12) and [p[0] for p in progress] == [4, 8, 12]
    assert load_file(str(items[3].text_path))["embeds"][0, 0].item() == float(len("cap www"))
    assert D.text_jobs(items, cfg, ["probe"])[0] == {}


def test_cache_text_halves_a_batch_that_runs_out_of_memory(tmp_path):
    cfg = TrainConfig(cache_dir=str(tmp_path))
    (tmp_path / "text").mkdir()
    items = _items(tmp_path, 7)  # + the empty prompt = 8
    sizes = []

    def fake_batch(te, tok, texts, max_len):
        sizes.append(len(texts))
        if len(texts) > 2:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return [torch.zeros(1, 4, dtype=torch.bfloat16) for _ in texts]

    D.cache_text(items, cfg, "te", "tok", encode_batch_fn=fake_batch, batch_size=8)
    assert sizes == [8, 4, 2, 2, 4, 2, 2]
    assert D.text_jobs(items, cfg)[0] == {}


def test_cache_text_one_prompt_fn_still_works(tmp_path):
    cfg = TrainConfig(cache_dir=str(tmp_path))
    (tmp_path / "text").mkdir()
    items = _items(tmp_path, 3)
    seen = []
    D.cache_text(items, cfg, "te", "tok",
                 encode_fn=lambda te, tok, t, n: seen.append(t) or torch.zeros(1, 4, dtype=torch.bfloat16))
    assert len(seen) == 4 and D.text_jobs(items, cfg)[0] == {}
