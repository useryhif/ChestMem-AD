"""Render a fixed random sample of normal and abnormal test cases as grids."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ddad.config import load_config
from ddad.data import ChestXrayDataset
from ddad.engine import _load_ensemble, resolve_device
from ddad.scoring import anomaly_maps
from ddad.visualize import build_difference_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)
    dataset = ChestXrayDataset(config.data, "test", "b", seed=config.train.seed)
    labels = np.asarray([label for _, label in dataset.samples])
    rng = np.random.default_rng(args.seed)
    normal_indices = rng.choice(np.where(labels == 0)[0], size=args.count, replace=False)
    abnormal_indices = rng.choice(np.where(labels == 1)[0], size=args.count, replace=False)

    output_dir = Path(config.experiment.output_dir) / "grids"
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = (("normal", normal_indices), ("abnormal", abnormal_indices))
    renderings: dict[str, list[tuple]] = {}
    scale_samples = {"reconstruction": [], "inter_discrepancy": [], "intra_discrepancy": []}
    for name, indices in selections:
        render_input = []
        for index in indices:
            image, _, _ = dataset[int(index)]
            images = image.unsqueeze(0).to(device)
            with torch.inference_mode():
                rec_a = torch.stack([model(images) for model in module_a])
                rec_b = torch.stack([model(images) for model in module_b])
                maps, _ = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
                mean_a = rec_a.mean(dim=0)
                mean_b = rec_b.mean(dim=0)
            render_input.append((images, maps, mean_a, mean_b))
            for key in scale_samples:
                scale_samples[key].append(maps[key][:1].cpu())
        renderings[name] = render_input
    scales = {
        key: max(float(torch.quantile(torch.cat(chunks).flatten(), 0.99)), 1e-8)
        for key, chunks in scale_samples.items()
    }
    for name, indices in selections:
        strips = []
        render_input = renderings[name]
        for images, maps, mean_a, mean_b in render_input:
            strips.append(
                build_difference_panel(
                    images[0, 0].cpu().numpy(),
                    mean_b[0, 0].cpu().numpy(),
                    maps["reconstruction"][0].cpu().numpy(),
                    maps["inter_discrepancy"][0].cpu().numpy(),
                    vmax_error=scales["reconstruction"],
                    vmax_inter=scales["inter_discrepancy"],
                    reconstruction_a=mean_a[0, 0].cpu().numpy(),
                    intra_spread=maps["intra_discrepancy"][0].cpu().numpy(),
                    vmax_intra=scales["intra_discrepancy"],
                )
            )
        width = max(strip.width for strip in strips)
        height = sum(strip.height for strip in strips)
        grid = Image.new("RGB", (width, height), "white")
        offset = 0
        for strip in strips:
            grid.paste(strip, (0, offset))
            offset += strip.height
        path = output_dir / f"{name}-grid.png"
        grid.save(path)
        print(f"{name}: indices={list(map(int, indices))} -> {path}")


if __name__ == "__main__":
    main()
