"""Render the most strongly flagged normal samples for hotspot inspection."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from ddad.config import load_config
from ddad.data import ChestXrayDataset, make_loader
from ddad.engine import _load_ensemble, resolve_device
from ddad.scoring import anomaly_maps
from ddad.visualize import save_difference_panel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)
    loader = make_loader(config.data, "test", "b", 100, config.train.seed)
    labels, maps_all, means = [], [], []
    scale_samples = {"reconstruction": [], "inter": [], "intra": []}
    with torch.inference_mode():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            rec_a = torch.stack([m(images) for m in module_a])
            rec_b = torch.stack([m(images) for m in module_b])
            maps, _ = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
            inter = maps["inter_discrepancy"]
            labels.extend(batch_labels.tolist())
            means.extend(inter.mean(dim=(1, 2)).cpu().tolist())
            scale_samples["reconstruction"].append(maps["reconstruction"][:2].cpu())
            scale_samples["inter"].append(inter[:2].cpu())
            scale_samples["intra"].append(maps["intra_discrepancy"][:2].cpu())
            maps_all.append(inter.cpu().numpy())
    labels = np.asarray(labels)
    stacked = np.concatenate(maps_all)
    threshold = float(np.quantile(stacked[labels == 0], 0.99))
    hot_areas = (stacked > threshold).mean(axis=(1, 2))
    normal = np.where(labels == 0)[0]
    ranked = normal[np.argsort(hot_areas[normal])[::-1][: args.count]]
    scales = {k: max(float(torch.quantile(torch.cat(v).flatten(), 0.99)), 1e-8) for k, v in scale_samples.items()}
    dataset = ChestXrayDataset(config.data, "test", "b", seed=config.train.seed)
    out = config.experiment.output_dir / "flagged-normals"
    out.mkdir(parents=True, exist_ok=True)
    for index in ranked:
        image, _, name = dataset[int(index)]
        images = image.unsqueeze(0).to(device)
        with torch.inference_mode():
            rec_a = torch.stack([m(images) for m in module_a])
            rec_b = torch.stack([m(images) for m in module_b])
            maps, _ = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
            mean_a = rec_a.mean(dim=0); mean_b = rec_b.mean(dim=0)
        save_difference_panel(
            images[0, 0].cpu().numpy(), mean_b[0, 0].cpu().numpy(),
            maps["reconstruction"][0].cpu().numpy(), maps["inter_discrepancy"][0].cpu().numpy(),
            out / f"flagged-{index:05d}.png",
            vmax_error=scales["reconstruction"], vmax_inter=scales["inter"],
            reconstruction_a=mean_a[0, 0].cpu().numpy(),
            intra_spread=maps["intra_discrepancy"][0].cpu().numpy(), vmax_intra=scales["intra"],
        )
        print(f"index={index} hot_area={hot_areas[index]:.4f} mean={means[index]:.5f} name={name}")
    print(f"threshold={threshold:.5f}")
    print("rendered to", out)

if __name__ == "__main__":
    main()
