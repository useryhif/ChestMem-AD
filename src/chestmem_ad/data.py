from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig


class ChestXrayDataset(Dataset):
    """Lazy-loading dataset driven by the experiment JSON manifest."""

    def __init__(self, config: DataConfig, split: str, module: str = "b", seed: int = 42):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if module not in {"a", "b"}:
            raise ValueError("module must be 'a' or 'b'")

        self.config = config
        manifest_path = config.root / config.manifest
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if split == "test":
            normal = list(manifest["test"]["0"])
            abnormal = list(manifest["test"]["1"])
            self.samples = [(name, 0) for name in normal] + [(name, 1) for name in abnormal]
        else:
            known_normal = list(manifest["train"]["0"])
            self.samples = [(name, 0) for name in known_normal]
            if module == "a" and config.extra_unlabeled > 0:
                unlabeled_normal = list(manifest["train"]["unlabeled"]["0"])
                unlabeled_abnormal = list(manifest["train"]["unlabeled"]["1"])
                if config.shuffle_unlabeled_pool:
                    rng = random.Random(seed)
                    rng.shuffle(unlabeled_normal)
                    rng.shuffle(unlabeled_abnormal)
                abnormal_count = round(config.extra_unlabeled * config.anomaly_ratio)
                normal_count = config.extra_unlabeled - abnormal_count
                if normal_count > len(unlabeled_normal) or abnormal_count > len(unlabeled_abnormal):
                    raise ValueError("Requested unlabeled samples exceed the manifest contents")
                self.samples += [(name, 0) for name in unlabeled_normal[:normal_count]]
                self.samples += [(name, 1) for name in unlabeled_abnormal[:abnormal_count]]
        self.cached_images = (
            [self._load_image(filename) for filename, _ in self.samples]
            if config.cache_images
            else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        filename, label = self.samples[index]
        tensor = (
            self.cached_images[index]
            if self.cached_images is not None
            else self._load_image(filename)
        )
        return tensor, label, Path(filename).stem

    def _load_image(self, filename: str) -> torch.Tensor:
        path = self.config.root / self.config.image_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Image listed in manifest does not exist: {path}")
        with Image.open(path) as image:
            image = image.convert("L").resize(
                (self.config.image_size, self.config.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor


def make_loader(
    config: DataConfig,
    split: str,
    module: str,
    batch_size: int,
    seed: int,
) -> DataLoader:
    dataset = ChestXrayDataset(config, split=split, module=module, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        drop_last=split == "train" and len(dataset) >= batch_size,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
