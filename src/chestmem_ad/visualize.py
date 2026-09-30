from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def _gray(array: np.ndarray) -> Image.Image:
    array = np.asarray(array, dtype=np.float32)
    array = (array - array.min()) / (array.max() - array.min() + 1e-8)
    return Image.fromarray(np.uint8(array * 255), mode="L")


def _xray(array: np.ndarray) -> Image.Image:
    """Render tensors normalized to [-1, 1] without per-image contrast stretching."""
    array = np.asarray(array, dtype=np.float32)
    array = np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def _heatmap(array: np.ndarray, vmax: float) -> Image.Image:
    """Render a fixed-scale heatmap so colors are comparable across samples."""
    array = np.asarray(array, dtype=np.float32)
    array = np.clip(array / vmax, 0.0, 1.0)
    gray = Image.fromarray(np.uint8(array * 255), mode="L")
    return ImageOps.colorize(gray, black=(8, 24, 80), mid=(245, 220, 60), white=(190, 15, 15))


def build_difference_panel(
    image: np.ndarray,
    reconstruction: np.ndarray,
    reconstruction_error: np.ndarray,
    inter_discrepancy: np.ndarray,
    vmax_error: float = 0.25,
    vmax_inter: float = 0.25,
    reconstruction_a: np.ndarray | None = None,
    intra_spread: np.ndarray | None = None,
    vmax_intra: float = 0.25,
) -> Image.Image:
    """Build the input/reconstruction/heatmap panel strip."""
    smooth = Image.Resampling.LANCZOS
    soft = Image.Resampling.BILINEAR
    tiles: list[tuple[str, Image.Image, Image.Resampling]] = [
        ("Input", _xray(image).convert("RGB"), smooth)
    ]
    if reconstruction_a is not None:
        tiles.append(("Recon A", _xray(reconstruction_a).convert("RGB"), smooth))
    tiles.append(("Recon B", _xray(reconstruction).convert("RGB"), smooth))
    tiles.append(
        ("Squared error", _heatmap(reconstruction_error, vmax=max(vmax_error, 1e-8)), soft)
    )
    tiles.append(
        ("A/B discrepancy", _heatmap(inter_discrepancy, vmax=max(vmax_inter, 1e-8)), soft)
    )
    if intra_spread is not None:
        tiles.append(("B spread", _heatmap(intra_spread, vmax=max(vmax_intra, 1e-8)), soft))
    scale = 4
    panels = [
        (title, tile.resize((tile.width * scale, tile.height * scale), resample))
        for title, tile, resample in tiles
    ]
    width, height = panels[0][1].size
    title_height = 14
    panel = Image.new("RGB", (width * len(panels), height + title_height), "white")
    draw = ImageDraw.Draw(panel)
    for index, (title, tile) in enumerate(panels):
        x = index * width
        draw.text((x + 2, 1), title, fill="black")
        panel.paste(tile, (x, title_height))
    return panel


def save_difference_panel(
    image: np.ndarray,
    reconstruction: np.ndarray,
    reconstruction_error: np.ndarray,
    inter_discrepancy: np.ndarray,
    output_path: str | Path,
    vmax_error: float = 0.25,
    vmax_inter: float = 0.25,
    reconstruction_a: np.ndarray | None = None,
    intra_spread: np.ndarray | None = None,
    vmax_intra: float = 0.25,
) -> None:
    """Save input, module reconstructions, error, and discrepancy maps."""
    panel = build_difference_panel(
        image,
        reconstruction,
        reconstruction_error,
        inter_discrepancy,
        vmax_error=vmax_error,
        vmax_inter=vmax_inter,
        reconstruction_a=reconstruction_a,
        intra_spread=intra_spread,
        vmax_intra=vmax_intra,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
