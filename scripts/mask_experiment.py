"""Test border-ring masking of the A/B discrepancy map."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ddad.config import load_config
from ddad.data import make_loader
from ddad.engine import _load_ensemble, resolve_device
from ddad.scoring import anomaly_maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.05, 0.08, 0.1, 0.12, 0.15])
    parser.add_argument("--area", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)
    loader = make_loader(config.data, "test", "b", args.batch_size, config.train.seed)

    labels: list[int] = []
    maps_all: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            rec_a = torch.stack([model(images) for model in module_a])
            rec_b = torch.stack([model(images) for model in module_b])
            batch_maps, _ = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
            maps_all.append(batch_maps["inter_discrepancy"].cpu().numpy())
            labels.extend(batch_labels.tolist())

    stacked = np.concatenate(maps_all)
    labels_array = np.asarray(labels)
    height, width = stacked.shape[-2:]
    print(f"config: {args.config}  pool={config.scoring.pool}  samples={len(labels)}")
    for margin in args.margins:
        top = int(round(height * margin))
        left = int(round(width * margin))
        masked = stacked.copy()
        if top or left:
            masked[:, :top, :] = 0.0
            masked[:, height - top :, :] = 0.0
            masked[:, :, :left] = 0.0
            masked[:, :, width - left :] = 0.0
        means = masked.mean(axis=(1, 2))
        threshold = float(np.quantile(masked[labels_array == 0], 0.99))
        areas = (masked > threshold).mean(axis=(1, 2))
        auroc = roc_auc_score(labels_array, means)
        normal_flagged = float((areas[labels_array == 0] > args.area).mean())
        abnormal_flagged = float((areas[labels_array == 1] > args.area).mean())
        print(
            f"  margin={margin:.2f}  inter_auroc={auroc:.4f}"
            f"  normal_flagged={normal_flagged:.3f}  abnormal_flagged={abnormal_flagged:.3f}"
        )


if __name__ == "__main__":
    main()
