"""Compare inter-discrepancy levels for different ensemble subset sizes."""

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
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)
    loader = make_loader(config.data, "test", "b", args.batch_size, config.train.seed)
    max_members = min(len(module_a), len(module_b))

    labels: list[int] = []
    scores: dict[int, list[float]] = {size: [] for size in range(1, max_members + 1)}
    with torch.inference_mode():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            for size in range(1, max_members + 1):
                rec_a = torch.stack([model(images) for model in module_a[:size]])
                rec_b = torch.stack([model(images) for model in module_b[:size]])
                maps, _ = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
                scores[size].extend(maps["inter_discrepancy"].mean(dim=(1, 2)).cpu().tolist())
            labels.extend(batch_labels.tolist())

    labels_array = np.asarray(labels)
    print(f"config: {args.config}  members available={max_members}")
    for size in range(1, max_members + 1):
        values = np.asarray(scores[size])
        normal = float(values[labels_array == 0].mean())
        abnormal = float(values[labels_array == 1].mean())
        auroc = roc_auc_score(labels_array, values)
        print(
            f"  K={size}  normal_mean={normal:.5f}  abnormal_mean={abnormal:.5f}"
            f"  ratio={abnormal / normal:.2f}  inter_auroc={auroc:.4f}"
        )


if __name__ == "__main__":
    main()
