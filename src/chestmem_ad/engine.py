from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from .config import Config
from .data import make_loader
from .model import Autoencoder, PaperAutoencoder, SkipAutoencoder
from .scoring import anomaly_maps
from .visualize import save_difference_panel


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: Config, device: torch.device) -> nn.Module:
    architecture = config.model.architecture
    model_class = {
        "compact": Autoencoder,
        "paper": PaperAutoencoder,
        "skip": SkipAutoencoder,
        "noskip": SkipAutoencoder,
        "resample": SkipAutoencoder,
    }.get(architecture)
    if model_class is None:
        raise ValueError(f"Unknown model architecture: {config.model.architecture}")
    extra: dict[str, object] = {}
    if architecture == "noskip":
        extra["use_skips"] = False
    elif architecture == "resample":
        extra["use_skips"] = False
        extra["decoder_mode"] = "resample"
    if architecture in {"skip", "noskip", "resample"} and config.model.memory_size > 0:
        extra["memory_size"] = config.model.memory_size
        extra["shrink_threshold"] = config.model.shrink_threshold
    return model_class(
        image_size=config.data.image_size,
        latent_size=config.model.latent_size,
        width_multiplier=config.model.width_multiplier,
        **extra,
    ).to(device)


def train(config: Config, module: str) -> list[Path]:
    if module not in {"a", "b"}:
        raise ValueError("module must be 'a' or 'b'")
    device = resolve_device(config.experiment.device)
    checkpoint_dir = config.experiment.output_dir / "checkpoints" / module
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for member in range(config.train.ensemble_size):
        seed = config.train.seed + member
        seed_everything(seed)
        loader = make_loader(
            config.data, "train", module, config.train.batch_size, seed
        )
        model = build_model(config, device)
        path = checkpoint_dir / f"member-{member:02d}.pt"
        payload = None
        if config.train.resume and path.is_file():
            payload = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(payload["model"])
            print(f"resuming module={module} member={member + 1} from {path}")
        optimizer_name = config.train.optimizer.lower()
        optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(optimizer_name)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer: {config.train.optimizer}")
        optimizer = optimizer_class(
            model.parameters(),
            lr=config.train.learning_rate,
            betas=(config.train.beta1, config.train.beta2),
            weight_decay=config.train.weight_decay,
        )
        start_epoch = 1
        if payload is not None and "optimizer" in payload and "epoch" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            start_epoch = int(payload["epoch"]) + 1
        loss_fn = nn.MSELoss()
        for epoch in range(start_epoch, config.train.epochs + 1):
            model.train()
            losses: list[float] = []
            for images, _, _ in loader:
                images = images.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(images), images)
                if config.train.entropy_loss_weight > 0 and getattr(model, "memory", None) is not None:
                    loss = loss + config.train.entropy_loss_weight * model.memory.entropy()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            print(f"module={module} member={member + 1} epoch={epoch}/{config.train.epochs} loss={mean_loss:.6f}")
            if epoch % config.train.checkpoint_interval == 0 or epoch == config.train.epochs:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "seed": seed,
                    },
                    path,
                )

        if not path.is_file():
            torch.save({"model": model.state_dict(), "seed": seed}, path)
        saved.append(path)
    return saved


def _load_ensemble(config: Config, module: str, device: torch.device) -> list[nn.Module]:
    paths = sorted((config.experiment.output_dir / "checkpoints" / module).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found for module {module!r}")
    models = []
    for path in paths:
        model = build_model(config, device)
        payload = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
        model.eval()
        models.append(model)
    return models


def evaluate(config: Config) -> dict[str, dict[str, float]]:
    device = resolve_device(config.experiment.device)
    module_a = _load_ensemble(config, "a", device)
    module_b = _load_ensemble(config, "b", device)
    loader = make_loader(config.data, "test", "b", config.train.batch_size, config.train.seed)
    labels: list[int] = []
    scores = {"reconstruction": [], "inter_discrepancy": [], "intra_discrepancy": []}
    visual_dir = config.experiment.output_dir / "visualizations"
    pending_panels: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]
    ] = []
    scale_samples: dict[str, list[torch.Tensor]] = {
        "reconstruction": [],
        "inter_discrepancy": [],
        "intra_discrepancy": [],
    }

    with torch.inference_mode():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            rec_a = torch.stack([model(images) for model in module_a])
            rec_b = torch.stack([model(images) for model in module_b])
            maps, values = anomaly_maps(images, rec_a, rec_b, pool=config.scoring.pool)
            mean_a = rec_a.mean(dim=0)
            mean_b = rec_b.mean(dim=0)
            error_maps = maps["reconstruction"]
            inter_maps = maps["inter_discrepancy"]
            intra_maps = maps["intra_discrepancy"]
            scale_samples["reconstruction"].append(error_maps[:2].cpu())
            scale_samples["inter_discrepancy"].append(inter_maps[:2].cpu())
            scale_samples["intra_discrepancy"].append(intra_maps[:2].cpu())
            for index in range(min(images.shape[0], 8)):
                pending_panels.append(
                    (
                        images[index, 0].cpu().numpy(),
                        mean_a[index, 0].cpu().numpy(),
                        mean_b[index, 0].cpu().numpy(),
                        error_maps[index].cpu().numpy(),
                        inter_maps[index].cpu().numpy(),
                        intra_maps[index].cpu().numpy(),
                        visual_dir / f"{len(labels) + index:05d}.png",
                    )
                )
            labels.extend(batch_labels.tolist())
            for name, value in values.items():
                scores[name].extend(value.cpu().tolist())

    scales = {
        name: max(float(torch.quantile(torch.cat(chunks).flatten(), 0.99)), 1e-8)
        for name, chunks in scale_samples.items()
    }
    print(
        f"heatmap scales: error vmax={scales['reconstruction']:.6f}"
        f" inter vmax={scales['inter_discrepancy']:.6f}"
        f" intra vmax={scales['intra_discrepancy']:.6f}"
    )
    for image, recon_a, recon_b, error_map, inter_map, intra_map, path in pending_panels:
        save_difference_panel(
            image,
            recon_b,
            error_map,
            inter_map,
            path,
            vmax_error=scales["reconstruction"],
            vmax_inter=scales["inter_discrepancy"],
            reconstruction_a=recon_a,
            intra_spread=intra_map,
            vmax_intra=scales["intra_discrepancy"],
        )

    if len(set(labels)) < 2:
        raise ValueError("Evaluation requires at least one normal and one abnormal sample")
    metrics = {
        name: {
            "auroc": float(roc_auc_score(labels, values)),
            "average_precision": float(average_precision_score(labels, values)),
        }
        for name, values in scores.items()
    }
    config.experiment.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = config.experiment.output_dir / "metrics.json"
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
