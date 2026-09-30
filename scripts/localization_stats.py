"""Quantify how often normal/abnormal samples stay clean or flag hotspots.

Threshold for a "hot" pixel is the 99th percentile of normal-sample pixel
values of the A/B discrepancy map. An image counts as flagged when the
fraction of hot pixels exceeds ``--area``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ddad.config import load_config
from ddad.data import make_loader
from ddad.engine import _load_ensemble, resolve_device
from ddad.scoring import anomaly_maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--area", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)
    loader = make_loader(config.data, "test", "b", args.batch_size, config.train.seed)

    labels: list[int] = []
    maps: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            rec_a = torch.stack([model(images) for model in module_a])
            rec_b = torch.stack([model(images) for model in module_b])
            batch_maps, _ = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
            maps.append(batch_maps["inter_discrepancy"].cpu().numpy())
            labels.extend(batch_labels.tolist())

    stacked = np.concatenate(maps)
    labels_array = np.asarray(labels)
    threshold = float(np.quantile(stacked[labels_array == 0], 0.99))
    areas = (stacked > threshold).mean(axis=(1, 2))
    maxes = stacked.max(axis=(1, 2))
    means = stacked.mean(axis=(1, 2))

    print(f"config: {args.config}  pool={config.scoring.pool}  threshold={threshold:.5f}")
    for value, name in ((0, "normal"), (1, "abnormal")):
        mask = labels_array == value
        flagged = float((areas[mask] > args.area).mean())
        print(
            f"  {name:8s} n={int(mask.sum())}"
            f"  mean_score={float(np.median(means[mask])):.5f}"
            f"  median_max={float(np.median(maxes[mask])):.5f}"
            f"  flagged_fraction={flagged:.3f}"
        )


if __name__ == "__main__":
    main()
