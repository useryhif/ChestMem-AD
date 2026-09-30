"""Evaluate anomaly scores computed at average-pooled resolutions."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import average_precision_score, roc_auc_score

from chestmem_ad.config import load_config
from chestmem_ad.data import make_loader
from chestmem_ad.engine import _load_ensemble, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)
    loader = make_loader(config.data, "test", "b", 100, config.train.seed)

    labels: list[int] = []
    scores = {
        (name, factor): []
        for factor in args.factors
        for name in ("reconstruction", "inter_discrepancy", "intra_discrepancy")
    }
    with torch.inference_mode():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            rec_a = torch.stack([model(images) for model in module_a])
            rec_b = torch.stack([model(images) for model in module_b])
            mean_a = rec_a.mean(dim=0)
            mean_b = rec_b.mean(dim=0)
            for factor in args.factors:
                if factor > 1:
                    x = F.avg_pool2d(images, factor)
                    a_mean = F.avg_pool2d(mean_a, factor)
                    b_mean = F.avg_pool2d(mean_b, factor)
                    b_pooled = F.avg_pool2d(rec_b.flatten(0, 1), factor).view(
                        rec_b.shape[0],
                        rec_b.shape[1],
                        rec_b.shape[2],
                        b_mean.shape[-2],
                        b_mean.shape[-1],
                    )
                else:
                    x = images
                    a_mean = mean_a
                    b_mean = mean_b
                    b_pooled = rec_b
                values = {
                    "reconstruction": (x - b_mean).square().mean(dim=(1, 2, 3)),
                    "inter_discrepancy": (a_mean - b_mean).abs().mean(dim=(1, 2, 3)),
                    "intra_discrepancy": b_pooled.std(dim=0, correction=0).mean(dim=(1, 2, 3)),
                }
                for name, value in values.items():
                    scores[(name, factor)].extend(value.cpu().tolist())
            labels.extend(batch_labels.tolist())

    print(f"config: {args.config}")
    for factor in args.factors:
        for name in ("reconstruction", "inter_discrepancy", "intra_discrepancy"):
            values = scores[(name, factor)]
            auroc = roc_auc_score(labels, values)
            average_precision = average_precision_score(labels, values)
            print(f"  factor={factor}  {name:18s} auroc={auroc:.4f}  ap={average_precision:.4f}")


if __name__ == "__main__":
    main()
