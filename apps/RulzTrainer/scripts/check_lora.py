"""Verify an exported LoRA maps onto MOSS-Image keys the way ComfyUI's generic loader expects.

    python scripts/check_lora.py output/my_lora/my_lora.safetensors
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors import safe_open  # noqa: E402

from rulztrainer.moss.model import MOSSImageTransformer  # noqa: E402


def main(path):
    model_keys = set(MOSSImageTransformer().state_dict().keys())
    bad, modules = [], set()
    with safe_open(path, "pt") as f:
        meta = f.metadata() or {}
        keys = list(f.keys())
        for k in keys:
            if not k.startswith("diffusion_model."):
                bad.append(k)
                continue
            stem = k[len("diffusion_model."):]
            for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            modules.add(stem)
            if stem + ".weight" not in model_keys:
                bad.append(k)
        rank = f.get_tensor(next(k for k in keys if k.endswith("lora_down.weight"))).shape[0]
    print(f"{path}: {len(keys)} tensors, {len(modules)} LoRA modules, rank {rank}")
    print("metadata:", {k: v[:80] for k, v in meta.items()})
    if bad:
        print("UNMAPPED KEYS:", bad[:20])
        return 1
    print("OK: every key maps to a diffusion_model.<module>.weight in MOSS-Image")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
