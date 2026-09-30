from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    root: Path
    manifest: str = "data.json"
    image_dir: str = "train_png_512"
    image_size: int = 64
    extra_unlabeled: int = 4000
    anomaly_ratio: float = 0.6
    workers: int = 0
    cache_images: bool = False
    shuffle_unlabeled_pool: bool = True


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "compact"
    latent_size: int = 16
    width_multiplier: float = 1.0
    memory_size: int = 0
    shrink_threshold: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    epochs: int = 250
    ensemble_size: int = 3
    seed: int = 42
    resume: bool = False
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    checkpoint_interval: int = 10
    entropy_loss_weight: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = Path("outputs/rsna-ae")
    device: str = "auto"


@dataclass(frozen=True)
class ScoringConfig:
    pool: int = 1


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = raw.get("data", {})
    model = raw.get("model", {})
    train = raw.get("train", {})
    experiment = raw.get("experiment", {})
    scoring = raw.get("scoring", {})

    root = Path(_require(data, "root"))
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()

    output_dir = Path(experiment.get("output_dir", "outputs/rsna-ae"))
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()

    anomaly_ratio = float(data.get("anomaly_ratio", 0.6))
    if not 0.0 <= anomaly_ratio <= 1.0:
        raise ValueError("data.anomaly_ratio must be between 0 and 1")
    scoring_pool = int(scoring.get("pool", 1))
    if scoring_pool < 1:
        raise ValueError("scoring.pool must be at least 1")

    return Config(
        data=DataConfig(
            root=root,
            manifest=data.get("manifest", "data.json"),
            image_dir=data.get("image_dir", "train_png_512"),
            image_size=int(data.get("image_size", 64)),
            extra_unlabeled=int(data.get("extra_unlabeled", 4000)),
            anomaly_ratio=anomaly_ratio,
            workers=int(data.get("workers", 0)),
            cache_images=bool(data.get("cache_images", False)),
            shuffle_unlabeled_pool=bool(data.get("shuffle_unlabeled_pool", True)),
        ),
        model=ModelConfig(
            architecture=str(model.get("architecture", "compact")),
            latent_size=int(model.get("latent_size", 16)),
            width_multiplier=float(model.get("width_multiplier", 1.0)),
            memory_size=int(model.get("memory_size", 0)),
            shrink_threshold=float(model.get("shrink_threshold", 0.0)),
        ),
        train=TrainConfig(
            batch_size=int(train.get("batch_size", 64)),
            learning_rate=float(train.get("learning_rate", 5e-4)),
            weight_decay=float(train.get("weight_decay", 0.0)),
            epochs=int(train.get("epochs", 250)),
            ensemble_size=int(train.get("ensemble_size", 3)),
            seed=int(train.get("seed", 42)),
            resume=bool(train.get("resume", False)),
            optimizer=str(train.get("optimizer", "adamw")),
            beta1=float(train.get("beta1", 0.9)),
            beta2=float(train.get("beta2", 0.999)),
            checkpoint_interval=int(train.get("checkpoint_interval", 10)),
            entropy_loss_weight=float(train.get("entropy_loss_weight", 0.0)),
        ),
        experiment=ExperimentConfig(
            output_dir=output_dir,
            device=str(experiment.get("device", "auto")),
        ),
        scoring=ScoringConfig(pool=scoring_pool),
    )
