"""Dynamo's config is thread-local: the limits prepare() raises must reach the autograd thread that recomputes
checkpointed blocks and the data-parallel workers. thread_prepare() applies them once per thread."""
import sys
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import compiling  # noqa: E402


def test_prepare_settings_reach_other_threads_through_thread_prepare():
    import torch._dynamo.config as cfg

    compiling.prepare()
    assert cfg.recompile_limit >= compiling.LIMITS["recompile_limit"]
    seen = {}

    def worker():
        seen["before"] = cfg.recompile_limit
        compiling.thread_prepare()
        seen["after"] = (cfg.recompile_limit, cfg.accumulated_recompile_limit, cfg.specialize_float)
        compiling._thread.done = False  # a second call would be a no-op; force it to show it is idempotent
        compiling.thread_prepare()
        seen["again"] = cfg.recompile_limit

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["after"] == (compiling.LIMITS["recompile_limit"], compiling.LIMITS["accumulated_recompile_limit"], True)
    assert seen["again"] == compiling.LIMITS["recompile_limit"]
    # (the thread saw torch's own defaults first, which is the point; the exact default is torch's business)
    assert seen["before"] <= seen["after"][0]


def test_models_apply_the_limits_in_the_thread_that_runs_their_blocks():
    """The models call thread_prepare() before running a block, so a worker thread (a data-parallel replica; on CUDA
    also autograd's recompute thread) sees the raised limits by the time the first block runs. On the CPU autograd
    recomputes on the calling thread, so the forward is run in a worker thread here."""
    import torch._dynamo.config as cfg

    from rulztrainer.krea2.model import Krea2Transformer

    compiling.prepare()
    model = Krea2Transformer(in_channels=64, num_layers=1, attention_head_dim=16, num_attention_heads=4,
                             num_key_value_heads=2, intermediate_size=96, timestep_embed_dim=32, text_hidden_dim=40,
                             num_text_layers=3, text_num_attention_heads=4, text_num_key_value_heads=4,
                             text_intermediate_size=56, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
                             axes_dims_rope=(4, 6, 6)).float()
    block = model.transformer_blocks[0]
    orig_forward, seen = block.forward, {}

    def spy(*a, **k):
        seen["in_block"] = cfg.recompile_limit
        return orig_forward(*a, **k)

    block.forward = spy

    def worker():
        seen["before"] = cfg.recompile_limit  # torch's own default in a fresh thread
        with torch.no_grad():
            model(torch.randn(1, 64, 2, 3), 400.0, torch.randn(1, 5, 3, 40), None)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["in_block"] == compiling.LIMITS["recompile_limit"]
    assert seen["before"] <= seen["in_block"]
