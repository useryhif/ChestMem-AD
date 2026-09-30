"""Render input, raw discrepancy maps, and ASR-refined maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from chestmem_ad.config import load_config
from chestmem_ad.engine import _load_ensemble, resolve_device
from chestmem_ad.scoring import anomaly_maps
from heldout_eval import HeldoutDataset
from train_asr import ASRNet, refined_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    output = config.experiment.output_dir / "asr"
    payload = json.loads((config.experiment.output_dir / "heldout" / "manifest.json").read_text(encoding="utf-8"))
    samples = [tuple(item) for item in payload["samples"][: args.count]]
    device = resolve_device(config.experiment.device)
    models_a = _load_ensemble(config, "a", device)
    models_b = _load_ensemble(config, "b", device)
    net = ASRNet().to(device)
    checkpoint = torch.load(output / "asr.pt", map_location=device, weights_only=True)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    loader = DataLoader(HeldoutDataset(config, samples), batch_size=args.count, shuffle=False)
    images, _, names = next(iter(loader))
    images = images.to(device)
    with torch.inference_mode():
        rec_a = torch.stack([model(images) for model in models_a])
        rec_b = torch.stack([model(images) for model in models_b])
        maps, _ = anomaly_maps(images, rec_a, rec_b, pool=1)
        inputs = torch.stack((maps["inter_discrepancy"], maps["intra_discrepancy"]), dim=1)
        refined = torch.sigmoid(refined_logits(net, inputs))[:, 0]
    target = output / "visualizations"
    target.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(images[index, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[0].set_title("Input")
        axes[1].imshow(maps["inter_discrepancy"][index].cpu(), cmap="magma")
        axes[1].set_title("Raw inter")
        axes[2].imshow(maps["intra_discrepancy"][index].cpu(), cmap="magma")
        axes[2].set_title("Raw intra")
        axes[3].imshow(refined[index].cpu(), cmap="magma", vmin=0, vmax=1)
        axes[3].set_title("ASR refined")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout(pad=0.3)
        fig.savefig(target / f"{index:02d}-{name}.png", dpi=160)
        plt.close(fig)
    print(f"saved {len(names)} panels to {target}")


if __name__ == "__main__":
    main()
