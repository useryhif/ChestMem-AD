"""Warm-start a no-skip model from checkpoints of the skip model.

Encoder, latent, upsampling and output weights transfer directly. Fusion
convolution weights keep only the upsampled-feature half of the input
channels, matching the no-skip forward path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("fuse") and value.dim() == 4:
            channels = value.shape[1] // 2
            converted[key] = value[:, :channels].clone()
        else:
            converted[key] = value.clone()
    return converted


def convert_directory(source: Path, destination: Path) -> None:
    checkpoints = sorted(source.glob("*/*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {source}")
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = convert_state(payload["model"])
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": state, "seed": payload.get("seed", 42)}, target)
        print(f"{path} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    convert_directory(Path(args.source), Path(args.destination))


if __name__ == "__main__":
    main()
