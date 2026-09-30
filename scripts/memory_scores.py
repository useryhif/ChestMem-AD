"""Test a memory-bank nearest-neighbour score using an existing normal-trained encoder."""

from __future__ import annotations

import argparse

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ddad.config import load_config
from ddad.data import ChestXrayDataset, make_loader
from ddad.engine import _load_ensemble, resolve_device


def features(model, images: torch.Tensor, level: int) -> torch.Tensor:
    if hasattr(model, "down1"):
        x = model.down1(images)
        if level == 1:
            return x
        x = model.down2(x)
        if level == 2:
            return x
        x = model.down3(x)
        if level == 3:
            return x
        return model.down4(x)
    x = images
    for index in range(level * 3):
        x = model.encoder[index](x)
    return x


def nearest_distances(patches: torch.Tensor, memory: torch.Tensor, block: int = 4096) -> torch.Tensor:
    best = torch.full((patches.shape[0],), float("inf"), device=patches.device)
    for start in range(0, memory.shape[0], block):
        chunk = memory[start : start + block]
        distances = torch.cdist(patches, chunk).min(dim=1).values
        best = torch.minimum(best, distances)
    return best.square()


def run(config_path: str, level: int, memory_size: int, test_batch: int) -> None:
    config = load_config(config_path)
    device = resolve_device(config.experiment.device)
    module_b = _load_ensemble(config, "b", device)
    phi = module_b[0]
    train_set = ChestXrayDataset(config.data, "train", "b", seed=config.train.seed)
    memory_images = torch.stack([train_set[index][0] for index in range(memory_size)]).to(device)
    with torch.inference_mode():
        memory = features(phi, memory_images, level)
        memory = memory.permute(0, 2, 3, 1).reshape(-1, memory.shape[1])

    loader = make_loader(config.data, "test", "b", test_batch, config.train.seed)
    labels: list[int] = []
    map_means: list[float] = []
    top_means: list[float] = []
    with torch.inference_mode():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            maps = features(phi, images, level)
            batch = maps.shape[0]
            flat = maps.permute(0, 2, 3, 1).reshape(-1, maps.shape[1])
            distances = nearest_distances(flat, memory).view(batch, -1)
            map_means.extend(distances.mean(dim=1).cpu().tolist())
            keep = max(1, int(distances.shape[1] * 0.05))
            top_means.extend(distances.topk(keep, dim=1).values.mean(dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())

    print(f"config={config_path} level={level} memory={memory_size}")
    for name, values in (("map_mean", map_means), ("top5_mean", top_means)):
        auroc = roc_auc_score(labels, values)
        average_precision = average_precision_score(labels, values)
        print(f"  {name:9s} auroc={auroc:.4f}  ap={average_precision:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--memory-size", type=int, default=64)
    parser.add_argument("--test-batch", type=int, default=25)
    args = parser.parse_args()
    run(args.config, args.level, args.memory_size, args.test_batch)


if __name__ == "__main__":
    main()
