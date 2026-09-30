import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from chestmem_ad.config import load_config
from chestmem_ad.data import ChestXrayDataset
from chestmem_ad.engine import evaluate, train


def _workspace(name: str) -> Path:
    root = Path(__file__).parent / "runtime" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _make_dataset(root: Path) -> None:
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    groups = {
        "known": [f"known-{i}.png" for i in range(4)],
        "unlabeled_normal": [f"un-{i}.png" for i in range(2)],
        "unlabeled_abnormal": [f"ua-{i}.png" for i in range(2)],
        "test_normal": [f"tn-{i}.png" for i in range(2)],
        "test_abnormal": [f"ta-{i}.png" for i in range(2)],
    }
    rng = np.random.default_rng(3)
    for group, names in groups.items():
        for name in names:
            array = rng.normal(80, 8, (16, 16)).clip(0, 255).astype(np.uint8)
            if "abnormal" in group:
                array[5:11, 5:11] = 240
            Image.fromarray(array).save(image_dir / name)
    manifest = {
        "train": {
            "0": groups["known"],
            "unlabeled": {"0": groups["unlabeled_normal"], "1": groups["unlabeled_abnormal"]},
        },
        "test": {"0": groups["test_normal"], "1": groups["test_abnormal"]},
    }
    (root / "data.json").write_text(json.dumps(manifest), encoding="utf-8")


def _make_config(tmp_path: Path) -> Path:
    data_root = tmp_path / "dataset"
    _make_dataset(data_root)
    config = tmp_path / "smoke.yaml"
    config.write_text(
        f"""
data:
  root: {data_root.as_posix()}
  image_dir: images
  image_size: 16
  extra_unlabeled: 4
  anomaly_ratio: 0.5
model:
  latent_size: 4
  width_multiplier: 0.25
train:
  batch_size: 4
  epochs: 1
  ensemble_size: 2
  seed: 7
experiment:
  output_dir: {(tmp_path / 'output').as_posix()}
  device: cpu
""",
        encoding="utf-8",
    )
    return config


def test_module_a_uses_unlabeled_mixture() -> None:
    tmp_path = _workspace("dataset")
    config = load_config(_make_config(tmp_path))
    module_a = ChestXrayDataset(config.data, "train", "a", seed=7)
    module_b = ChestXrayDataset(config.data, "train", "b", seed=7)
    assert len(module_a) == 8
    assert len(module_b) == 4
    assert sum(label for _, label in module_a.samples) == 2


def test_cpu_train_and_evaluate_smoke() -> None:
    tmp_path = _workspace("pipeline")
    config = load_config(_make_config(tmp_path))
    assert len(train(config, "a")) == 2
    assert len(train(config, "b")) == 2
    metrics = evaluate(config)
    assert set(metrics) == {"reconstruction", "inter_discrepancy", "intra_discrepancy"}
    assert (config.experiment.output_dir / "metrics.json").is_file()
    assert list((config.experiment.output_dir / "visualizations").glob("*.png"))
