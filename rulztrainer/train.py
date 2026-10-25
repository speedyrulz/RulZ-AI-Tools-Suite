"""CLI: python -m rulztrainer.train --config configs/example.yaml [--set key=value ...]"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _allocator_config() -> None:
    """Expandable segments for the CUDA caching allocator, before torch first touches CUDA: memory a training step
    freed then serves large blocks again. Without it the freed activations stay reserved in hundreds of segments split
    by small live tensors (measured: 3.9 GB cached in 784 segments, the largest free block 0.19 GB), so a large
    sample or probe after training ran out of memory where the same one before training fit. Linux only (the Windows
    CUDA driver has no support); a setting of the user's own is kept."""
    if sys.platform.startswith("linux") and not (os.environ.get("PYTORCH_ALLOC_CONF")
                                                 or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


_allocator_config()

from .config import TrainConfig  # noqa: E402
from .trainer import Trainer  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="RulzTrainer: LoRA trainer for MOSS-Image")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Override a config value (JSON parsed when possible)")
    ap.add_argument("--preset", default="", help="Apply a named preset (e.g. identity) before --set overrides")
    args = ap.parse_args(argv)
    cfg = TrainConfig.load(args.config)
    data = cfg.to_dict()
    if args.preset:
        from .config import apply_preset

        data = apply_preset(data, args.preset)
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            data[key] = json.loads(value)
        except json.JSONDecodeError:
            data[key] = value
    cfg = TrainConfig.from_dict(data)
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
