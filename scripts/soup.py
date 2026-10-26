"""Seed soup: average the weight changes of 2 or more LoRA / LoKr files of one subject into one LoRA.

    python scripts/soup.py a.safetensors b.safetensors c.safetensors --out soup.safetensors --rank 32

--rank 0 keeps the average exactly (LoRA files only; its rank is the sum of theirs). --strength scales the soup (an
average of LoRAs that disagree is weaker than each). The report: how much the files agree (cosine between their
changes), the soup's size against theirs, and the share of the average kept at that rank. To pick the ingredients and
strength by the likeness probes, use a run with soup_files (the Tools tab's Seed soup, "Score with the probes").
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer.soup import describe, make_soup  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="LoRA or LoKr .safetensors of one subject and model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32, help="rank of the soup (0 = exact, LoRA files only)")
    ap.add_argument("--weights", default="", help="comma-separated weight per file (default: equal)")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda:N (the SVD of large layers is faster on a GPU)")
    ap.add_argument("--format", choices=["comfy", "peft"], default=None, help="default: peft if any input is")
    args = ap.parse_args(argv)
    weights = [float(x) for x in args.weights.split(",")] if args.weights.strip() else None
    report = make_soup(args.files, args.out, args.rank, weights, args.strength, args.device, args.format)
    print(describe(report))
    print(f"saved {report['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
