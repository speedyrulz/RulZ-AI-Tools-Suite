"""Runs inside the RulZ AI Trainer's own Python (standard library plus the trainer's own modules).

    python _trainer_helper.py validate <trainer_root> <config.yaml>
    python _trainer_helper.py queue <trainer_root> <config.yaml> <label>

Prints one line for the caller and exits 0 (fine) or 1 (refused). The trainer's code does every read and write, so the
formats stay owned by the trainer.
"""
import sys
from pathlib import Path


def main(argv):
    command, root, cfg_path = argv[1], Path(argv[2]), argv[3]
    sys.path.insert(0, str(root))
    try:
        from rulztrainer.config import TrainConfig
    except Exception as e:  # noqa: BLE001
        print(f"Could not import the trainer from {root}: {type(e).__name__}: {e}")
        return 1
    try:
        cfg = TrainConfig.load(cfg_path)
        cfg.validate()
    except Exception as e:  # noqa: BLE001
        print(f"{type(e).__name__}: {e}")
        return 1
    if command == "validate":
        print(f"Valid: {cfg.job_name or '(no job name)'} on {getattr(cfg, 'model_arch', '?')}")
        return 0
    if command == "queue":
        from rulztrainer.train_queue import TrainQueue

        queue = TrainQueue(root / "output" / "_queue.json")
        entry = queue.add(cfg.to_dict(), argv[4] if len(argv) > 4 else "")
        print(f"Queued {entry['job']} (id {entry['id']}) in {root / 'output' / '_queue.json'}")
        return 0
    print(f"Unknown command {command}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
