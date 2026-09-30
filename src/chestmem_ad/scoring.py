from __future__ import annotations

import torch
from torch.nn import functional as F


def anomaly_maps(
    images: torch.Tensor,
    rec_a: torch.Tensor,
    rec_b: torch.Tensor,
    pool: int = 1,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute anomaly maps (upsampled to input size) and per-image scores.

    With ``pool > 1`` the maps are averaged over ``pool x pool`` blocks before
    scoring, which suppresses pixel-level noise while keeping structural
    differences. Display maps are bilinearly upsampled back to input size.
    """
    size = images.shape[-2:]
    mean_a = rec_a.mean(dim=0)
    mean_b = rec_b.mean(dim=0)
    if pool > 1:
        images_low = F.avg_pool2d(images, pool)
        mean_a_low = F.avg_pool2d(mean_a, pool)
        mean_b_low = F.avg_pool2d(mean_b, pool)
        rec_b_low = F.avg_pool2d(rec_b.flatten(0, 1), pool).view(
            rec_b.shape[0], rec_b.shape[1], rec_b.shape[2], *images_low.shape[-2:]
        )
    else:
        images_low, mean_a_low, mean_b_low, rec_b_low = images, mean_a, mean_b, rec_b

    maps_low = {
        "reconstruction": (images_low - mean_b_low).square().mean(dim=1),
        "inter_discrepancy": (mean_a_low - mean_b_low).abs().mean(dim=1),
        "intra_discrepancy": rec_b_low.std(dim=0, correction=0).mean(dim=1),
    }
    scores = {name: value.flatten(1).mean(dim=1) for name, value in maps_low.items()}
    if pool > 1:
        maps = {
            name: F.interpolate(
                value.unsqueeze(1), size=size, mode="bilinear", align_corners=False
            ).squeeze(1)
            for name, value in maps_low.items()
        }
    else:
        maps = maps_low
    return maps, scores
