"""Compare pixel-space and feature-space anomaly scores on existing checkpoints."""

from __future__ import annotations

import argparse

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ddad.config import load_config
from ddad.data import make_loader
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)
    phi = module_b[0]
    loader = make_loader(config.data, "test", "b", args.batch_size, config.train.seed)

    names = [
        "recon_pixel",
        "inter_pixel",
        "intra_pixel",
        "recon_feat3",
        "inter_feat3",
        "intra_feat3",
        "recon_feat4",
        "inter_feat4",
        "intra_feat4",
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
            scores["recon_pixel"].extend((images - mean_b).square().mean(dim=(1, 2, 3)).cpu().tolist())
            scores["inter_pixel"].extend((mean_a - mean_b).abs().mean(dim=(1, 2, 3)).cpu().tolist())
            scores["intra_pixel"].extend(rec_b.std(dim=0, correction=0).mean(dim=(1, 2, 3)).cpu().tolist())

            for level in (3, 4):
                fx = features(phi, images, level)
                fa = features(phi, mean_a, level)
                fb = features(phi, mean_b, level)
                fb_members = torch.stack([features(phi, rec_b[index], level) for index in range(rec_b.shape[0])])
                scores[f"recon_feat{level}"].extend((fx - fb).square().mean(dim=(1, 2, 3)).cpu().tolist())
                scores[f"inter_feat{level}"].extend((fa - fb).abs().mean(dim=(1, 2, 3)).cpu().tolist())
                scores[f"intra_feat{level}"].extend(
                    fb_members.std(dim=0, correction=0).mean(dim=(1, 2, 3)).cpu().tolist()
                )
            labels.extend(batch_labels.tolist())

    print(f"config: {args.config}  samples: {len(labels)}")
    for name in names:
        auroc = roc_auc_score(labels, scores[name])
        average_precision = average_precision_score(labels, scores[name])
        print(f"  {name:12s} auroc={auroc:.4f}  ap={average_precision:.4f}")


if __name__ == "__main__":
    main()
