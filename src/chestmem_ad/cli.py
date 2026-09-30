from __future__ import annotations

import argparse
import json

from .config import load_config
from .engine import evaluate, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-distribution chest X-ray anomaly detection")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train one side of the ChestMem-AD ensemble")
    train_parser.add_argument("--module", choices=("a", "b"), required=True)
    subparsers.add_parser("evaluate", help="Evaluate trained module A and B ensembles")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "train":
        paths = train(config, args.module)
        print("Saved checkpoints:")
        for path in paths:
            print(path)
    else:
        print(json.dumps(evaluate(config), indent=2))


if __name__ == "__main__":
    main()
