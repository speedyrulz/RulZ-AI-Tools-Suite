"""Post-hoc EMA: average one run's step checkpoints into one LoRA with moving-average weights, after the run.

    python scripts/posthoc_ema.py output/cat/cat_*.safetensors --out cat_ema.safetensors --sigma_rel 0.15
    python scripts/posthoc_ema.py output/cat/cat_*.safetensors --out cat_ema.safetensors --beta 0.5

The files are put in step order (their ss_steps, else the _NNNNNN in the name). --sigma_rel uses EDM2's power-function
average (0.05 to 0.2; smaller leans on the last steps), whatever the number of checkpoints. --beta is the decay per
checkpoint (musubi-tuner's default 0.95); the oldest file starts as the whole average, so with few checkpoints a high
decay leans on it. --beta2 slides the decay from --beta to --beta2 over the files. The average is saved at the inputs'
rank; --rank 0 keeps it exactly. The report: each file's weight, about which step the average sits at, and the share
of the average kept at that rank.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulztrainer.posthoc_ema import describe, make_ema  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="LoRA or LoKr step checkpoints of one run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.95, help="decay per checkpoint (default 0.95)")
    ap.add_argument("--beta2", type=float, default=None, help="slide the decay from --beta to this over the files")
    ap.add_argument("--sigma_rel", type=float, default=None, help="power-function EMA width (replaces --beta)")
    ap.add_argument("--rank", type=int, default=None, help="rank of the result (default: the inputs'; 0 = exact)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:N (the SVD of large layers is faster on a GPU)")
    ap.add_argument("--format", choices=["comfy", "peft"], default=None, help="default: the newest input's")
    args = ap.parse_args(argv)
    report = make_ema(args.files, args.out, args.beta, args.beta2, args.sigma_rel, args.rank, args.device,
                      args.format)
    print(describe(report))
    for path, w in zip(report["files"], report["weights"]):
        print(f"  {w:6.1%}  {Path(path).name}")
    print(f"saved {report['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
