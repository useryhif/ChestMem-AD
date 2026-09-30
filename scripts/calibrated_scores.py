"""Calibrate the A/B discrepancy map against per-pixel normal statistics."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from chestmem_ad.config import load_config
from chestmem_ad.data import make_loader
from chestmem_ad.engine import _load_ensemble, resolve_device
from chestmem_ad.scoring import anomaly_maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--calibration-count", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--area", type=float, default=0.001)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)

    train_loader = make_loader(config.data, "train", "b", args.batch_size, config.train.seed)
    calibration_maps: list[np.ndarray] = []
    seen = 0
    with torch.inference_mode():
        for images, _, _ in train_loader:
            if seen >= args.calibration_count:
                break
            images = images.to(device)
            rec_a = torch.stack([model(images) for model in module_a])
            rec_b = torch.stack([model(images) for model in module_b])
            maps, _ = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
            calibration_maps.append(maps["inter_discrepancy"].cpu().numpy())
            seen += images.shape[0]
    calibration = np.concatenate(calibration_maps)[: args.calibration_count]
    mean_field = calibration.mean(axis=0)
    std_field = calibration.std(axis=0) + 1e-6

    test_loader = make_loader(config.data, "test", "b", args.batch_size, config.train.seed)
    labels: list[int] = []
    test_maps: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_labels, _ in test_loader:
            images = images.to(device)
            rec_a = torch.stack([model(images) for model in module_a])
            rec_b = torch.stack([model(images) for model in module_b])
            maps, _ = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
            test_maps.append(maps["inter_discrepancy"].cpu().numpy())
            labels.extend(batch_labels.tolist())
    stacked = np.concatenate(test_maps)
    labels_array = np.asarray(labels)
    residual = stacked - mean_field
    calibrated = residual / std_field

    threshold = float(np.quantile((calibration - mean_field) / std_field, 0.99))
    print(f"config: {args.config}  calibration images={args.calibration_count}  threshold={threshold:.3f}")
    for name, values in (
        ("raw_mean", stacked.mean(axis=(1, 2))),
        ("residual_mean", residual.mean(axis=(1, 2))),
        ("calibrated_mean", calibrated.mean(axis=(1, 2))),
    ):
        auroc = roc_auc_score(labels_array, values)
        print(f"  {name:16s} auroc={auroc:.4f}")
    areas = (calibrated > threshold).mean(axis=(1, 2))
    normal_flagged = float((areas[labels_array == 0] > args.area).mean())
    abnormal_flagged = float((areas[labels_array == 1] > args.area).mean())
    print(f"  calibrated flagged: normal={normal_flagged:.3f}  abnormal={abnormal_flagged:.3f}")


if __name__ == "__main__":
    main()
