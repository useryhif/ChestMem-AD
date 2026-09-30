from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def hard_shrink_relu(inputs: torch.Tensor, lambd: float = 0.0, epsilon: float = 1e-12) -> torch.Tensor:
    """ReLU-based hard shrinkage for non-negative attention weights."""
    return (F.relu(inputs - lambd) * inputs) / (torch.abs(inputs - lambd) + epsilon)


class MemoryBank(nn.Module):
    """MemAE-style memory: latents attend over learned prototype vectors."""

    def __init__(self, memory_size: int, feature_dim: int, shrink_threshold: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(memory_size, feature_dim))
        stdv = 1.0 / math.sqrt(feature_dim)
        nn.init.uniform_(self.weight, -stdv, stdv)
        self.shrink_threshold = shrink_threshold
        self.last_attention: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention = F.softmax(F.linear(features, self.weight), dim=1)
        if self.shrink_threshold > 0:
            attention = hard_shrink_relu(attention, lambd=self.shrink_threshold)
            attention = F.normalize(attention, p=1, dim=1)
        self.last_attention = attention
        return F.linear(attention, self.weight.transpose(0, 1))

    def entropy(self) -> torch.Tensor:
        if self.last_attention is None:
            raise RuntimeError("MemoryBank.entropy requires a forward pass first")
        attention = self.last_attention
        return -(attention * torch.log(attention + 1e-12)).sum(dim=1).mean()


class Autoencoder(nn.Module):
    def __init__(self, image_size: int = 64, latent_size: int = 16, width_multiplier: float = 1.0):
        super().__init__()
        if image_size < 16 or image_size % 16:
            raise ValueError("image_size must be at least 16 and divisible by 16")
        channels = [max(4, int(value * width_multiplier)) for value in (16, 32, 64, 64)]
        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        feature_size = image_size // 16
        flattened = channels[-1] * feature_size * feature_size
        hidden = max(128, min(2048, flattened))
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(flattened, hidden), nn.ReLU(), nn.Linear(hidden, latent_size))
        self.from_latent = nn.Sequential(nn.Linear(latent_size, hidden), nn.ReLU(), nn.Linear(hidden, flattened))

        decoder: list[nn.Module] = []
        reversed_channels = list(reversed(channels))
        for index, current in enumerate(reversed_channels):
            final = index == len(reversed_channels) - 1
            target = 1 if final else reversed_channels[index + 1]
            decoder.append(nn.ConvTranspose2d(current, target, 4, 2, 1, bias=False))
            if not final:
                decoder.extend([nn.BatchNorm2d(target), nn.ReLU(inplace=True)])
        self.decoder = nn.Sequential(*decoder)
        self.feature_size = feature_size
        self.last_channels = channels[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(images)
        latent = self.to_latent(encoded)
        decoded = self.from_latent(latent).view(
            images.shape[0], self.last_channels, self.feature_size, self.feature_size
        )
        return self.decoder(decoded)


class PaperAutoencoder(nn.Module):
    """Paper-compatible autoencoder used for the reproduction baseline."""

    def __init__(self, image_size: int = 64, latent_size: int = 16, width_multiplier: float = 1.0):
        super().__init__()
        if image_size < 16 or image_size % 16:
            raise ValueError("image_size must be at least 16 and divisible by 16")
        channels = [int(value * width_multiplier) for value in (16, 32, 64, 64)]
        encoder: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            encoder.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder)
        self.feature_size = image_size // 16
        self.last_channels = channels[-1]
        flattened = self.last_channels * self.feature_size * self.feature_size
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, latent_size),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, flattened),
        )
        decoder: list[nn.Module] = []
        reversed_channels = list(reversed(channels))
        for index, current in enumerate(reversed_channels):
            final = index == len(reversed_channels) - 1
            target = 1 if final else reversed_channels[index + 1]
            decoder.append(nn.ConvTranspose2d(current, target, 4, 2, 1, bias=False))
            if not final:
                decoder.extend([nn.BatchNorm2d(target), nn.ReLU(inplace=True)])
        self.decoder = nn.Sequential(*decoder)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(images)
        latent = self.to_latent(encoded)
        decoded = self.from_latent(latent).view(
            images.shape[0], self.last_channels, self.feature_size, self.feature_size
        )
        return self.decoder(decoded)


class SkipAutoencoder(nn.Module):
    """Higher-resolution autoencoder with a latent bottleneck and U-Net-style skips."""

    def __init__(
        self,
        image_size: int = 128,
        latent_size: int = 128,
        width_multiplier: float = 1.0,
        use_skips: bool = True,
        decoder_mode: str = "deconv",
        memory_size: int = 0,
        shrink_threshold: float = 0.0,
    ):
        super().__init__()
        if image_size < 32 or image_size % 16:
            raise ValueError("image_size must be at least 32 and divisible by 16")
        if decoder_mode not in {"deconv", "resample"}:
            raise ValueError("decoder_mode must be 'deconv' or 'resample'")
        channels = [max(8, int(value * width_multiplier)) for value in (32, 64, 128, 256)]
        self.use_skips = use_skips
        self.decoder_mode = decoder_mode

        def down(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.down1 = down(1, channels[0])
        self.down2 = down(channels[0], channels[1])
        self.down3 = down(channels[1], channels[2])
        self.down4 = down(channels[2], channels[3])

        self.feature_size = image_size // 16
        self.last_channels = channels[3]
        flattened = self.last_channels * self.feature_size * self.feature_size
        hidden = 1024
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, latent_size),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_size, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, flattened),
        )
        self.memory = (
            MemoryBank(memory_size, latent_size, shrink_threshold) if memory_size > 0 else None
        )

        def up(in_channels: int, out_channels: int) -> nn.Sequential:
            if decoder_mode == "resample":
                return nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def fuse(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        skip_channels = 2 if use_skips else 1
        self.up3 = up(channels[3], channels[2])
        self.fuse3 = fuse(channels[2] * skip_channels, channels[2])
        self.up2 = up(channels[2], channels[1])
        self.fuse2 = fuse(channels[1] * skip_channels, channels[1])
        self.up1 = up(channels[1], channels[0])
        self.fuse1 = fuse(channels[0] * skip_channels, channels[0])
        if decoder_mode == "resample":
            self.output = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels[0], 1, 3, 1, 1, bias=False),
            )
        else:
            self.output = nn.ConvTranspose2d(channels[0], 1, 4, 2, 1, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        enc1 = self.down1(images)
        enc2 = self.down2(enc1)
        enc3 = self.down3(enc2)
        enc4 = self.down4(enc3)
        latent = self.to_latent(enc4)
        if self.memory is not None:
            latent = self.memory(latent)
        decoded = self.from_latent(latent).view(
            images.shape[0], self.last_channels, self.feature_size, self.feature_size
        )
        decoded = self.up3(decoded)
        if self.use_skips:
            decoded = torch.cat((decoded, enc3), dim=1)
        decoded = self.fuse3(decoded)
        decoded = self.up2(decoded)
        if self.use_skips:
            decoded = torch.cat((decoded, enc2), dim=1)
        decoded = self.fuse2(decoded)
        decoded = self.up1(decoded)
        if self.use_skips:
            decoded = torch.cat((decoded, enc1), dim=1)
        decoded = self.fuse1(decoded)
        return self.output(decoded)
