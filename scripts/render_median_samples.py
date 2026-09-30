"""Render the median normal and median abnormal test samples of an experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from chestmem_ad.config import load_config
from chestmem_ad.data import ChestXrayDataset, make_loader
from chestmem_ad.engine import _load_ensemble, resolve_device
from chestmem_ad.scoring import anomaly_maps
from chestmem_ad.visualize import save_difference_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)
    loader = make_loader(config.data, "test", "b", 100, config.train.seed)

    scores: list[float] = []
    labels: list[int] = []
    scale_samples = {"reconstruction": [], "inter": [], "intra": []}
    with torch.inference_mode():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            rec_a = torch.stack([model(images) for model in module_a])
            rec_b = torch.stack([model(images) for model in module_b])
            maps, _ = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
            error_maps = maps["reconstruction"]
            inter_maps = maps["inter_discrepancy"]
            intra_maps = maps["intra_discrepancy"]
            scores.extend(inter_maps.mean(dim=(1, 2)).cpu().tolist())
            labels.extend(batch_labels.tolist())
            scale_samples["reconstruction"].append(error_maps[:2].cpu())
            scale_samples["inter"].append(inter_maps[:2].cpu())
            scale_samples["intra"].append(intra_maps[:2].cpu())

    scores_array = np.asarray(scores)
    labels_array = np.asarray(labels)
    normal_sorted = np.where(labels_array == 0)[0][np.argsort(scores_array[labels_array == 0])]
    abnormal_sorted = np.where(labels_array == 1)[0][np.argsort(scores_array[labels_array == 1])]
    normal_pick = int(normal_sorted[len(normal_sorted) // 2])
    abnormal_pick = int(abnormal_sorted[len(abnormal_sorted) // 2])

    scales = {
        name: max(float(torch.quantile(torch.cat(chunks).flatten(), 0.99)), 1e-8)
        for name, chunks in scale_samples.items()
    }
    print(
        f"scales: error={scales['reconstruction']:.5f} inter={scales['inter']:.5f} intra={scales['intra']:.5f}"
    )
    print(f"normal median index={normal_pick} score={scores_array[normal_pick]:.5f}")
    print(f"abnormal median index={abnormal_pick} score={scores_array[abnormal_pick]:.5f}")

    dataset = ChestXrayDataset(config.data, "test", "b", seed=config.train.seed)
    output_dir = Path(args.output_dir) if args.output_dir else config.experiment.output_dir / "median-samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, tag in ((normal_pick, "normal"), (abnormal_pick, "abnormal")):
        image, label, name = dataset[index]
        images = image.unsqueeze(0).to(device)
        with torch.inference_mode():
            rec_a = torch.stack([model(images) for model in module_a])
            rec_b = torch.stack([model(images) for model in module_b])
            mean_a = rec_a.mean(dim=0)
            mean_b = rec_b.mean(dim=0)
            maps, _ = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
            error_map = maps["reconstruction"][0].cpu().numpy()
            inter_map = maps["inter_discrepancy"][0].cpu().numpy()
            intra_map = maps["intra_discrepancy"][0].cpu().numpy()
        hot_fraction = float((inter_map > 0.5 * inter_map.max()).mean())
        print(
            f"{tag}: label={label} name={name} mean_inter={inter_map.mean():.5f}"
            f" max_inter={inter_map.max():.5f} hot_fraction={hot_fraction:.3f}"
        )
        save_difference_panel(
            images[0, 0].cpu().numpy(),
            mean_b[0, 0].cpu().numpy(),
            error_map,
            inter_map,
            output_dir / f"{tag}-{index:05d}.png",
            vmax_error=scales["reconstruction"],
            vmax_inter=scales["inter"],
            reconstruction_a=mean_a[0, 0].cpu().numpy(),
            intra_spread=intra_map,
            vmax_intra=scales["intra"],
        )
    print(f"rendered to {output_dir}")


if __name__ == "__main__":
    main()
