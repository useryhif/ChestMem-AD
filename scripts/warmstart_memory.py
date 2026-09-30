"""Seed memory-augmented checkpoints from plain checkpoints.

All matching weights transfer; the memory bank stays freshly initialised
(one independent draw per ensemble member).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ddad.config import load_config
from ddad.engine import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    source = Path(args.source)
    destination = Path(args.destination)
    checkpoints = sorted(source.rglob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {source}")
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        torch.manual_seed(payload.get("seed", config.train.seed))
        model = build_model(config, torch.device("cpu"))
        target_state = model.state_dict()
        filtered = {
            key: value
            for key, value in payload["model"].items()
            if key in target_state and target_state[key].shape == value.shape
        }
        dropped = [key for key in payload["model"] if key not in filtered]
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys for {path}: {unexpected}")
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "seed": payload.get("seed", 42)}, target)
        print(f"{path} -> {target} (fresh keys: {len(missing)}, dropped: {dropped})")


if __name__ == "__main__":
    main()
