"""Print per-class score statistics for one or more experiment configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from ddad.config import load_config
from ddad.data import make_loader
from ddad.engine import _load_ensemble, resolve_device


def class_stats(config_path: str, batches_per_class: int = 3, batch_size: int = 100) -> None:
    config = load_config(config_path)
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)
    loader = make_loader(config.data, "test", "b", batch_size, config.train.seed)
    total_batches = (len(loader.dataset) + batch_size - 1) // batch_size
    abnormal_start = total_batches // 2
    stats: dict[str, dict[str, list[torch.Tensor]]] = {}

    with torch.inference_mode():
        for index, (images, _, _) in enumerate(loader):
            if index < batches_per_class:
                split = "normal"
            elif abnormal_start <= index < abnormal_start + batches_per_class:
                split = "abnormal"
            else:
                continue
            images = images.to(device)
            rec_a = torch.stack([model(images) for model in module_a])
            rec_b = torch.stack([model(images) for model in module_b])
            mean_a = rec_a.mean(dim=0)
            mean_b = rec_b.mean(dim=0)
            values = {
                "reconstruction": (images - mean_b).square().mean(dim=(1, 2, 3)),
                "inter_discrepancy": (mean_a - mean_b).abs().mean(dim=(1, 2, 3)),
                "intra_discrepancy": rec_b.std(dim=0, correction=0).mean(dim=(1, 2, 3)),
            }
            kernel = torch.ones(1, 1, 5, 5, device=images.device) / 25.0
            blurred = F.conv2d(images, kernel, padding=2)
            values["input_high_frequency"] = (images - blurred).square().mean(dim=(1, 2, 3))
            bucket = stats.setdefault(split, {name: [] for name in values})
            for name, value in values.items():
                bucket[name].append(value.cpu())

    print(f"config: {config_path}")
    for name in ("reconstruction", "inter_discrepancy", "intra_discrepancy", "input_high_frequency"):
        for split in ("normal", "abnormal"):
            values = torch.cat(stats[split][name])
            print(
                f"  {name:18s} {split:8s} n={values.numel():4d}"
                f"  mean={float(values.mean()):.6f}  median={float(values.median()):.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--batches-per-class", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    for config_path in args.configs:
        class_stats(config_path, args.batches_per_class, args.batch_size)


if __name__ == "__main__":
    main()
