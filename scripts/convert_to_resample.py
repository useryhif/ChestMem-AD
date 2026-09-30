"""Warm-start a resample-decoder model from no-skip checkpoints.

Encoder, latent and fusion weights transfer directly. Each transposed
convolution becomes an upsample + 3x3 convolution initialised from the
centre crop of the old 4x4 kernel, and the following batch-norm shifts to
the new position in the module list.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("up") and key.endswith(".0.weight"):
            converted[key[: -len(".0.weight")] + ".1.weight"] = (
                value[:, :, 1:4, 1:4].permute(1, 0, 2, 3).contiguous().clone()
            )
        elif key.startswith("up") and ".1." in key:
            converted[key.replace(".1.", ".2.")] = value.clone()
        elif key == "output.weight":
            converted["output.1.weight"] = value[:, :, 1:4, 1:4].permute(1, 0, 2, 3).contiguous().clone()
        else:
            converted[key] = value.clone()
    return converted


def convert_directory(source: Path, destination: Path) -> None:
    checkpoints = sorted(source.rglob("*.pt"))
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
