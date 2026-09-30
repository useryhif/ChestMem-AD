from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def create_chestmem_split(
    source: Path,
    output: Path,
    unlabeled_per_class: int,
    seed: int,
) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    normal = list(data["test"]["0"])
    abnormal = list(data["test"]["1"])
    rng = random.Random(seed)
    rng.shuffle(normal)
    rng.shuffle(abnormal)
    if unlabeled_per_class >= min(len(normal), len(abnormal)):
        raise ValueError("unlabeled_per_class must leave samples in both test classes")
    prepared = {
        "train": {
            "0": list(data["train"]["0"]),
            "unlabeled": {
                "0": normal[:unlabeled_per_class],
                "1": abnormal[:unlabeled_per_class],
            },
        },
        "test": {
            "0": normal[unlabeled_per_class:],
            "1": abnormal[unlabeled_per_class:],
        },
    }
    output.write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    print(
        f"wrote {output}: known_normal={len(prepared['train']['0'])}, "
        f"unlabeled={unlabeled_per_class * 2}, "
        f"test={len(prepared['test']['0']) + len(prepared['test']['1'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a leakage-free ChestMem-AD split")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unlabeled-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_chestmem_split(args.source, args.output, args.unlabeled_per_class, args.seed)


if __name__ == "__main__":
    main()
