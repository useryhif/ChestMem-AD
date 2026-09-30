"""Train and evaluate a compact anomaly-score refinement network.

The reconstruction ensembles stay frozen.  The refinement network sees only
the inter/intra discrepancy maps, and is trained with synthetic patch blends
made from normal chest X-rays.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from ddad.config import load_config
from ddad.data import ChestXrayDataset
from ddad.engine import _load_ensemble, resolve_device
from ddad.scoring import anomaly_maps


class ASRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.net(maps)


def refined_logits(net: ASRNet, inputs: torch.Tensor) -> torch.Tensor:
    """Predict a residual over the raw inter map so spatial evidence is retained."""
    base_probability = (1.0 - torch.exp(-inputs[:, :1] / 0.10)).clamp(1e-4, 1.0 - 1e-4)
    return torch.logit(base_probability) + net(inputs)


def make_synthetic(images: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend a random rectangular patch from another normal image."""
    batch, _, height, width = images.shape
    permutation = torch.randperm(batch, generator=generator, device=images.device)
    donor = images[permutation]
    mask = torch.zeros(batch, 1, height, width, device=images.device)
    for index in range(batch):
        patch_size = int(torch.randint(max(4, height // 10), max(5, height * 4 // 10 + 1), (1,), generator=generator, device=images.device).item())
        center_y = int(torch.randint(height // 10, max(height // 10 + 1, height * 9 // 10), (1,), generator=generator, device=images.device).item())
        center_x = int(torch.randint(width // 10, max(width // 10 + 1, width * 9 // 10), (1,), generator=generator, device=images.device).item())
        y0 = max(0, center_y - patch_size // 2)
        x0 = max(0, center_x - patch_size // 2)
        y1 = min(height, y0 + patch_size)
        x1 = min(width, x0 + patch_size)
        mask[index, :, y0:y1, x0:x1] = 1.0
    alpha = torch.rand(batch, 1, 1, 1, generator=generator, device=images.device) * 0.7 + 0.3
    synthetic = images * (1.0 - mask) + ((1.0 - alpha) * images + alpha * donor) * mask
    return synthetic, mask


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, positive_weight: float = 4.0) -> torch.Tensor:
    weights = torch.where(target > 0.5, torch.full_like(target, positive_weight), torch.ones_like(target))
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    return (weights * (1.0 - pt).pow(gamma) * bce).mean()


def discrepancy_inputs(images, models_a, models_b):
    with torch.no_grad():
        rec_a = torch.stack([model(images) for model in models_a])
        rec_b = torch.stack([model(images) for model in models_b])
        maps, _ = anomaly_maps(images, rec_a, rec_b, pool=1)
        return torch.stack((maps["inter_discrepancy"], maps["intra_discrepancy"]), dim=1)


def train_refiner(config, output: Path, epochs: int, batch_size: int, max_batches: int, seed: int):
    device = resolve_device(config.experiment.device)
    models_a = _load_ensemble(config, "a", device)
    models_b = _load_ensemble(config, "b", device)
    normal = ChestXrayDataset(config.data, split="train", module="b", seed=seed)
    train_size = int(len(normal) * 0.85)
    train_set, _ = random_split(normal, [train_size, len(normal) - train_size], generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=config.data.workers)
    net = ASRNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = torch.Generator(device=device).manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        for batch_index, (images, _, _) in enumerate(loader):
            if batch_index >= max_batches:
                break
            images = images.to(device)
            synthetic, target = make_synthetic(images, rng)
            inputs = discrepancy_inputs(synthetic, models_a, models_b)
            logits = refined_logits(net, inputs)
            loss = focal_loss(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"asr epoch={epoch}/{epochs} loss={mean_loss:.6f}")
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "epoch": epochs, "seed": seed}, output / "asr.pt")
    (output / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return net, models_a, models_b, device


def evaluate_refiner(config, output: Path, net, models_a, models_b, device, batch_size: int, heldout_name: str):
    if heldout_name == "test":
        dataset = ChestXrayDataset(config.data, split="test", module="b", seed=config.train.seed)
    else:
        manifest_path = output.parent / heldout_name / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Run scripts/heldout_eval.py before evaluating ASR")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        from heldout_eval import HeldoutDataset

        samples = [tuple(item) for item in payload["samples"]]
        dataset = HeldoutDataset(config, samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=config.data.workers)
    labels, names, baseline_inter, baseline_intra, refined = [], [], [], [], []
    net.eval()
    with torch.inference_mode():
        for images, batch_labels, batch_names in loader:
            images = images.to(device)
            inputs = discrepancy_inputs(images, models_a, models_b)
            refined.extend(torch.sigmoid(refined_logits(net, inputs)).flatten(1).mean(1).cpu().tolist())
            baseline_inter.extend(inputs[:, 0].flatten(1).mean(1).cpu().tolist())
            baseline_intra.extend(inputs[:, 1].flatten(1).mean(1).cpu().tolist())
            labels.extend(batch_labels.tolist())
            names.extend(batch_names)
    metrics = {
        "baseline_inter_pool1": {
            "auroc": float(roc_auc_score(labels, baseline_inter)),
            "average_precision": float(average_precision_score(labels, baseline_inter)),
        },
        "baseline_intra_pool1": {
            "auroc": float(roc_auc_score(labels, baseline_intra)),
            "average_precision": float(average_precision_score(labels, baseline_intra)),
        },
        "asr_refined": {
            "auroc": float(roc_auc_score(labels, refined)),
            "average_precision": float(average_precision_score(labels, refined)),
        },
    }
    metrics_name = "metrics.json" if heldout_name == "heldout" else f"metrics-{heldout_name}.json"
    (output / metrics_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    scores_name = "scores.csv" if heldout_name == "heldout" else f"scores-{heldout_name}.csv"
    with (output / scores_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "label", "baseline_inter_pool1", "baseline_intra_pool1", "asr_refined"])
        for index, name in enumerate(names):
            writer.writerow([name, labels[index], baseline_inter[index], baseline_intra[index], refined[index]])
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--heldout-name", default="heldout")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = config.experiment.output_dir / "asr"
    if args.evaluate_only:
        device = resolve_device(config.experiment.device)
        models_a = _load_ensemble(config, "a", device)
        models_b = _load_ensemble(config, "b", device)
        net = ASRNet().to(device)
        checkpoint = torch.load(output / "asr.pt", map_location=device, weights_only=True)
        net.load_state_dict(checkpoint["model"])
    else:
        net, models_a, models_b, device = train_refiner(config, output, args.epochs, args.batch_size, args.max_batches, args.seed)
    evaluate_refiner(config, output, net, models_a, models_b, device, args.batch_size, args.heldout_name)


if __name__ == "__main__":
    main()
