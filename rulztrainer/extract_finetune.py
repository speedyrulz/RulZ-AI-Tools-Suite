"""Extract a LoRA from a fine-tune's delta file (finetune_save_delta: <job>_finetune.safetensors).

    python -m rulztrainer.extract_finetune output/myjob/myjob_finetune.safetensors --rank 32 64 128

One file per rank next to the delta file (<stem>_r64.safetensors), in the file's own LoRA format (the model
family's: ComfyUI names for MOSS-Image, PEFT names for Krea 2 and FIBO 2.0), alpha = rank."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .finetune import extract_file, read_delta_meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("delta", help="the fine-tune's delta file (<job>_finetune.safetensors)")
    ap.add_argument("--rank", type=int, nargs="+", default=[64], help="LoRA rank(s) to extract (default 64)")
    ap.add_argument("--out", default="", help="output file (one rank) or folder; default: next to the delta file")
    ap.add_argument("--format", default="", choices=["", "comfy", "peft"], help="LoRA file layout (default: the "
                    "model family's, from the delta file)")
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    args = ap.parse_args(argv)
    delta = Path(args.delta)
    meta = read_delta_meta(delta)
    fmt = args.format or meta.get("lora_format") or "comfy"
    keep = {k: v for k, v in meta.items() if k.startswith(("ss_", "modelspec.", "rulztrainer_"))
            and k not in ("rulztrainer_finetune",)}
    stem = delta.name[: -len("_finetune.safetensors")] if delta.name.endswith("_finetune.safetensors") else delta.stem
    for rank in args.rank:
        if args.out and len(args.rank) == 1 and args.out.endswith(".safetensors"):
            out = Path(args.out)
        else:
            folder = Path(args.out) if args.out else delta.parent
            folder.mkdir(parents=True, exist_ok=True)
            out = folder / f"{stem}_r{rank}.safetensors"
        n = extract_file(delta, out, rank, fmt, {**keep, "ss_network_dim": str(rank), "ss_network_alpha": str(rank),
                                                "rulztrainer_config": meta.get("rulztrainer_config", "{}")},
                         device=args.device)
        print(f"rank {rank}: {n} layers -> {out}")
    return 0


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    sys.exit(main())
