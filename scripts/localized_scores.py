"""Test localized anomaly scores: central-crop means and top-k hotspot means."""

from __future__ import annotations

import argparse

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ddad.config import load_config
from ddad.data import make_loader
from ddad.engine import _load_ensemble, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)
    loader = make_loader(config.data, "test", "b", args.batch_size, config.train.seed)

    names = [
        "recon_full",
        "recon_crop",
        "recon_top",
        "inter_full",
        "inter_crop",
        "inter_top",
        "intra_full",
        "intra_crop",
        "intra_top",
    ]
    labels: list[int] = []
    scores: dict[str, list[float]] = {name: [] for name in names}
    with torch.inference_mode():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            rec_a = torch.stack([model(images) for model in module_a])
            rec_b = torch.stack([model(images) for model in module_b])
            mean_a = rec_a.mean(dim=0)
            mean_b = rec_b.mean(dim=0)
            height, width = images.shape[-2:]
            top = int(height * args.margin)
            left = int(width * args.margin)
            crop = (slice(top, height - top), slice(left, width - left))

            recon_map = (images - mean_b).square().mean(dim=1)
            inter_map = (mean_a - mean_b).abs().mean(dim=1)
            intra_map = rec_b.std(dim=0, correction=0).mean(dim=1)
            for prefix, maps in (
                ("recon", recon_map),
                ("inter", inter_map),
                ("intra", intra_map),
            ):
                flat = maps.flatten(1)
                keep = max(1, int(flat.shape[1] * args.top_fraction))
                scores[f"{prefix}_full"].extend(flat.mean(dim=1).cpu().tolist())
                scores[f"{prefix}_crop"].extend(maps[(slice(None), *crop)].flatten(1).mean(dim=1).cpu().tolist())
                scores[f"{prefix}_top"].extend(flat.topk(keep, dim=1).values.mean(dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())

    print(f"config: {args.config}  margin={args.margin}  top_fraction={args.top_fraction}")
    for name in names:
        auroc = roc_auc_score(labels, scores[name])
        average_precision = average_precision_score(labels, scores[name])
        print(f"  {name:12s} auroc={auroc:.4f}  ap={average_precision:.4f}")


if __name__ == "__main__":
    main()
