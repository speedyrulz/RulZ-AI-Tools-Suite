"""CLI: python -m rulztrainer.train --config configs/example.yaml [--set key=value ...]"""
from __future__ import annotations

import argparse
import json

from .config import TrainConfig
from .trainer import Trainer


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
