"""Inspect periodic texture in the input and reconstruction of one test image."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from chestmem_ad.config import load_config
from chestmem_ad.data import make_loader
from chestmem_ad.engine import _load_ensemble, resolve_device


def spectrum_peaks(patch: np.ndarray) -> list[tuple[int, int, float]]:
    window = np.hanning(patch.shape[0])[:, None] * np.hanning(patch.shape[1])[None, :]
    spectrum = np.abs(np.fft.fft2((patch - patch.mean()) * window))
    height, width = spectrum.shape
    centre_y, centre_x = height // 2, width // 2
    spectrum[centre_y - 3 : centre_y + 4, centre_x - 3 : centre_x + 4] = 0
    flat = np.argsort(spectrum.ravel())[::-1][:5]
    peaks = []
    for flat_index in flat:
        y, x = divmod(flat_index, width)
        peaks.append((x - centre_x, y - centre_y, float(spectrum[y, x])))
    return peaks


def upsample(array: np.ndarray, scale: int = 8) -> Image.Image:
    image = Image.fromarray(np.uint8(np.clip(array * 255, 0, 255)), mode="L")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--index", type=int, default=1024)
    parser.add_argument("--output", default="outputs/grid-analysis-input-recon.png")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.experiment.device)
    module_b = _load_ensemble(config, "b", device)
    loader = make_loader(config.data, "test", "b", 128, config.train.seed)

    target_batch = args.index // 128
    offset = args.index % 128
    batch = None
    for batch_index, (candidate, _, _) in enumerate(loader):
        if batch_index == target_batch:
            batch = candidate
            break
    if batch is None:
        raise ValueError(f"Index {args.index} not found in test loader")
    image = batch[offset : offset + 1].to(device)
    with torch.inference_mode():
        reconstruction = torch.stack([model(image) for model in module_b]).mean(dim=0)
    x = image[0, 0].cpu().numpy()
    r = reconstruction[0, 0].cpu().numpy()

    for name, array in (("input", x), ("recon", r)):
        peaks = spectrum_peaks(array)
        print(f"{name} top spectrum peaks (dx, dy, magnitude): {[(dx, dy, round(m)) for dx, dy, m in peaks]}")

    crop = (slice(60, 100), slice(35, 95))
    def diagonal_means(array: np.ndarray) -> tuple[float, float]:
        return (
            float(np.abs(array[:-1, :-1] - array[1:, 1:]).mean()),
            float(np.abs(array[:-1, 1:] - array[1:, :-1]).mean()),
        )

    for name, array in (("input", x), ("recon", r)):
        main_diag, anti_diag = diagonal_means(array[crop])
        print(f"{name} crop diagonal means: main={main_diag:.4f} anti={anti_diag:.4f}")
    for name, array in (("input crop", x[crop]), ("recon crop", r[crop])):
        peaks = spectrum_peaks(array)
        print(f"{name} top peaks (dx, dy, magnitude): {[(dx, dy, round(m)) for dx, dy, m in peaks]}")
    height = crop[0].stop - crop[0].start
    width = crop[1].stop - crop[1].start
    strip = Image.new("L", (width * 8, height * 8 * 2 + 10), 255)
    strip.paste(upsample(x[crop]), (0, 0))
    strip.paste(upsample(r[crop]), (0, height * 8 + 10))
    output = args.output
    strip.save(output)
    print("saved", output)


if __name__ == "__main__":
    main()
