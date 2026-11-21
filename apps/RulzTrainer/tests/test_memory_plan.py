"""The block swap plan counts the trainable adapter (rulztrainer.runplan, webui.block_swap_plan): a LoKr or a high rank
holds more than the default rank-16 LoRA the flat step allowance was set for. No GPU, no model files: the shapes come
from the family's model class built on the meta device."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer import runplan  # noqa: E402
from rulztrainer.config import TrainConfig  # noqa: E402


def krea(**kw) -> TrainConfig:
    return TrainConfig(model_arch="krea2", **kw)


def test_the_default_run_adds_nothing():
    for fam in ("moss_image", "krea2", "qwen_image21", "minimax_h3"):
        cfg = TrainConfig(model_arch=fam)
        assert runplan.adapter_gb(cfg) > 0  # counted, and exactly what the allowance covers
        assert runplan.adapter_extra_gb(cfg) == 0.0


def test_a_bigger_adapter_adds_more_than_a_small_one():
    low = runplan.adapter_extra_gb(krea(lora_rank=8))
    high = runplan.adapter_extra_gb(krea(lora_rank=64))
    lokr = runplan.adapter_extra_gb(krea(network_type="lokr", lokr_factor=8))
    assert low == 0.0 and high > 2.0 and lokr > 2.0
    # Fizgig's count for LoKr factor 8 on Krea 2: about 200M parameters
    n = runplan.adapter_params(runplan.target_shapes("krea2", tuple(krea().resolved_lora_targets())), "lokr", factor=8)
    assert 150e6 < n < 250e6
    # a lighter optimizer or no weight average holds less
    assert runplan.adapter_extra_gb(krea(lora_rank=64, optimizer="adamw8bit")) < high
    assert runplan.adapter_extra_gb(krea(lora_rank=64, lora_ema=0.0)) < high
    # a fine-tune has no adapter
    assert runplan.adapter_extra_gb(krea(finetune=True)) == 0.0


def test_a_known_shape_set_matches_a_hand_count():
    shapes = [(3072, 3072), (8192, 3072)]
    assert runplan.adapter_params(shapes, "lora", rank=16) == 16 * (3072 + 3072) + 16 * (3072 + 8192)
    # LoKr factor 8: 3072 = 8 x 384, 8192 = 8 x 1024 -> w1 8x8, w2 384x384 and 1024x384
    assert runplan.adapter_params(shapes, "lokr", factor=8) == (64 + 384 * 384) + (64 + 1024 * 384)
    # OFT block 32: out x 32 per layer; DoRA adds a magnitude per input
    assert runplan.adapter_params(shapes, "oft", block=32) == 3072 * 32 + 8192 * 32
    assert runplan.adapter_params(shapes, "lora", rank=4, dora=True) == 4 * (6144 + 11264) + 3072 * 2
    # bytes per parameter: fp32 weight + gradient + AdamW's two fp32 moments + the average's fp32 shadow
    assert runplan.bytes_per_param("adamw", ema=True) == 20
    assert runplan.bytes_per_param("adamw8bit", ema=False) == 10
    assert runplan.bytes_per_param("automagic4", ema=False, polarity_history=16) == 10
    assert runplan.bytes_per_param("keel", ema=False, lookahead=True) == 13


def test_the_count_agrees_with_the_adapter_the_trainer_builds():
    torch = pytest.importorskip("torch")
    from rulztrainer.lora import LoRALinear

    for kind, kw in (("lora", {}), ("lokr", {"factor": 4}), ("oft", {"block": 8})):
        for dora in (False, True):
            if dora and kind == "oft":
                continue
            base = torch.nn.Linear(96, 160, bias=False)
            m = LoRALinear(base, 8, 8.0, kind=kind, factor=kw.get("factor", 8), block=kw.get("block", 32), dora=dora)
            built = sum(p.numel() for p in m.adapter_params().values())
            assert runplan.adapter_params([(160, 96)], kind, rank=8, dora=dora, **kw) == built, (kind, dora)


def test_the_shapes_follow_the_targets_and_blocks():
    every = runplan.target_shapes("krea2", tuple(krea().resolved_lora_targets()))
    half = runplan.target_shapes("krea2", tuple(krea().resolved_lora_targets()), "0-13")
    attn = runplan.target_shapes("krea2", ("q", "k", "v", "o"))
    assert len(every) > len(half) > 0 and 0 < len(attn) < len(every)
    # fewer layers than the default: less than the allowance covers, nothing extra
    assert runplan.adapter_extra_gb(krea(lora_blocks="0-13")) == 0.0


def test_an_unknown_model_adds_nothing_and_the_plan_is_unchanged():
    import webui

    runplan.target_shapes.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(runplan.FAMILY_MODEL, "krea2", ("krea2", "NoSuchTransformer"))
            lokr = krea(network_type="lokr")
            assert runplan.target_shapes("krea2", tuple(lokr.resolved_lora_targets())) == ()
            assert runplan.adapter_extra_gb(lokr) == 0.0
            # today's plan: int8 13.5 GB + the 4.5 GB allowance on a 16 GB card, plus the 2 blocks in flight
            n = math.ceil(28 * (13.5 + webui.STEP_HEADROOM_GB - 16.0) / 13.5) + 2
            assert webui.block_swap_plan(lokr, 16.0, "int8") == (n, 13.5 * n / 28)
            assert webui.block_swap_plan(lokr, 16.0, "int8") == webui.block_swap_plan(krea(), 16.0, "int8")
            runplan.target_shapes.cache_clear()  # the bogus class's empty answer must not outlive the patch
    finally:
        runplan.target_shapes.cache_clear()
    assert runplan.adapter_extra_gb(TrainConfig(model_arch="krea2", network_type="lokr")) > 0  # counted again


def test_a_big_adapter_swaps_more_blocks_where_the_default_just_fits():
    import webui

    vram = 13.5 + webui.STEP_HEADROOM_GB + 0.3  # Krea 2 int8 plus the allowance, with a little to spare
    assert webui.block_swap_plan(krea(), vram, "int8") == (0, 0.0)
    swap, saved = webui.block_swap_plan(krea(network_type="lokr", lokr_factor=8), vram, "int8")
    extra = runplan.adapter_extra_gb(krea(network_type="lokr", lokr_factor=8))
    assert swap > 0 and saved >= extra - 0.3
    more, _ = webui.block_swap_plan(krea(lora_rank=64), vram, "int8")
    assert more > 0
    # an explicit figure wins over the count (the wizard passes its own picks')
    assert webui.block_swap_plan(krea(), vram, "int8", adapter_gb=0.0) == (0, 0.0)


def test_a_race_counts_its_largest_configuration():
    """One configuration is on the GPU at a time, and a row may change the adapter: the plan is for the largest."""
    rows = [{"name": "small", "lora_rank": 16}, {"name": "big", "network_type": "lokr", "lokr_factor": 8}]
    race = krea(race=True, race_configs=rows)
    lokr = runplan.adapter_extra_gb(krea(network_type="lokr", lokr_factor=8))
    assert runplan.adapter_extra_gb(race) == pytest.approx(lokr) and lokr > 2.0
    assert runplan.adapter_extra_gb(krea(race_configs=rows)) == 0.0  # the rows are there, the race is off
    # a row's own value beats one of the wizard's picks; the others get the pick
    rows = [{"name": "avg", "lora_ema": 0.98, "lora_rank": 64}, {"name": "plain", "lora_rank": 64}]
    race = krea(race=True, race_configs=rows)
    assert runplan.adapter_gb(race, {"lora_ema": 0.0}) == pytest.approx(runplan.adapter_gb(krea(lora_rank=64)))
    # rows the race itself would refuse do not break the plan
    assert runplan.adapter_extra_gb(krea(race=True, race_configs=[{"name": "x", "no_such_setting": 1}])) >= 0.0


def test_the_wizard_says_what_it_counted():
    import webui

    cfg = krea(network_type="lokr", lokr_factor=8)
    rec = webui.wizard_recommendations(cfg, "concept", [{"total": 18800}], None)
    swap, reason = rec["block_swap"]
    assert swap > 0 and "LoKr (factor 8)" in reason and "GB more" in reason
    plain = webui.wizard_recommendations(krea(), "concept", [{"total": 24000}], None)["block_swap"]
    assert plain[0] == 0 and "GB more" not in plain[1]  # the wizard's rank-16 adamw8bit LoRA: nothing extra


def test_the_two_blocks_in_flight_free_nothing():
    """Block swap streams the running block and the next one into the GPU (measured.INFLIGHT_BLOCKS), so n swapped
    blocks free n - 2 blocks' weights at a step's peak. A card a little short used to get 1 block, which freed
    nothing (Fizgig b926d67c found its own planner doing the same)."""
    import webui
    from rulztrainer.measured import INFLIGHT_BLOCKS

    assert INFLIGHT_BLOCKS == 2
    per = 13.5 / 28  # Krea 2 int8, 28 blocks
    short = 13.5 + webui.STEP_HEADROOM_GB - 0.1  # 0.1 GB short: one block's worth
    swap, saved = webui.block_swap_plan(krea(), short, "int8", adapter_gb=0.0)
    assert swap == 3 and saved == pytest.approx(3 * per)  # the GB is what the blocks hold in RAM
    for vram in (16.0, 14.0, 12.0, 10.0):
        n, _, tight = webui.block_swap_fit(krea(), vram, "int8", 0.0)
        over = 13.5 + webui.STEP_HEADROOM_GB - vram
        assert (n - INFLIGHT_BLOCKS) * per >= over - 1e-9 and not tight  # what is freed covers the shortfall
        assert (n - INFLIGHT_BLOCKS - 1) * per < over  # and no more blocks than that
    # all but 2 swapped still leaves too much: tight
    n, _, tight = webui.block_swap_fit(krea(), 4.0, "int8", 0.0)
    assert n == 26 and tight


def test_h3_plans_room_for_the_two_blocks_in_flight():
    """MiniMax H3's own count (50 blocks of 0.387 GB): the resident blocks are what the room holds minus the 2
    streaming through it."""
    import webui

    vram = 24000 / 1024  # the wizard reads the card's MiB
    one = webui.wizard_recommendations(TrainConfig(model_arch="minimax_h3"), "concept", [{"total": 24000}], None)
    resident = vram - 4.5 - 1.6
    assert one["block_swap"][0] == math.ceil(50 - resident / 0.387) + 2
