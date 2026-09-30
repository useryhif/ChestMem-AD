"""Evaluate an untouched subset of the unlabeled pool.

The training manifest reserves the tail of each unlabeled class for this check:
the existing training run uses the first 1,600 normal and 2,400 abnormal pool
images, so this script samples from the remaining images only.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from ddad.config import load_config
from ddad.engine import _load_ensemble, resolve_device
from ddad.scoring import anomaly_maps


class HeldoutDataset(Dataset):
    def __init__(self, config, samples: list[tuple[str, int]]):
        self.config = config
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, label = self.samples[index]
        path = self.config.data.root / self.config.data.image_dir / name
        with Image.open(path) as image:
            image = image.convert("L").resize(
                (self.config.data.image_size, self.config.data.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).unsqueeze(0), label, Path(name).stem


def choose_samples(config, count: int, seed: int, excluded: set[str] | None = None) -> list[tuple[str, int]]:
    manifest = json.loads((config.data.root / config.data.manifest).read_text(encoding="utf-8"))
    used = set(manifest["train"]["0"]) | set(manifest["test"]["0"]) | set(manifest["test"]["1"])
    normal = list(manifest["train"]["unlabeled"]["0"])[1600:]
    abnormal = list(manifest["train"]["unlabeled"]["1"])[2400:]
    if used & (set(normal) | set(abnormal)):
        raise ValueError("reserved heldout pool overlaps the training or test manifest")
    excluded = excluded or set()
    normal = [name for name in normal if name not in excluded]
    abnormal = [name for name in abnormal if name not in excluded]
    rng = random.Random(seed)
    if count > len(normal) or count > len(abnormal):
        raise ValueError(f"count={count} exceeds the reserved pool")
    normal = rng.sample(normal, count)
    abnormal = rng.sample(abnormal, count)
    samples = [(name, 0) for name in normal] + [(name, 1) for name in abnormal]
    rng.shuffle(samples)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-name", default="heldout")
    parser.add_argument("--exclude-manifest", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    excluded: set[str] = set()
    if args.exclude_manifest:
        previous = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
        excluded = {item[0] for item in previous["samples"]}
    samples = choose_samples(config, args.count, args.seed, excluded)
    loader = DataLoader(
        HeldoutDataset(config, samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.data.workers,
        pin_memory=device.type == "cuda",
    )
    models_a = _load_ensemble(config, "a", device)
    models_b = _load_ensemble(config, "b", device)
    labels: list[int] = []
    names: list[str] = []
    scores = {"reconstruction": [], "inter_discrepancy": [], "intra_discrepancy": []}
    with torch.inference_mode():
        for images, batch_labels, batch_names in loader:
            images = images.to(device)
            rec_a = torch.stack([model(images) for model in models_a])
            rec_b = torch.stack([model(images) for model in models_b])
            _, batch_scores = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
            labels.extend(batch_labels.tolist())
            names.extend(batch_names)
            for key, value in batch_scores.items():
                scores[key].extend(value.cpu().tolist())
    metrics = {
        key: {
            "auroc": float(roc_auc_score(labels, value)),
            "average_precision": float(average_precision_score(labels, value)),
        }
        for key, value in scores.items()
    }
    output = config.experiment.output_dir / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps({"seed": args.seed, "count_per_class": args.count, "samples": samples}, indent=2),
        encoding="utf-8",
    )
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "label", *scores])
        for index, (name, label) in enumerate(zip(names, labels)):
            writer.writerow([name, label, *(scores[key][index] for key in scores)])
    print(json.dumps(metrics, indent=2))
    print(f"saved heldout artifacts to {output}")


if __name__ == "__main__":
    main()
