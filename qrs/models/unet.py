"""Small U-Net encoder/decoder for LandCover.ai segmentation, with a
per-spatial-location bottleneck slot (classical / quantum / parameter-matched
control), matching HQ-UNet's "non-pooling QCNN at the point of highest
compression" design.

Channel widths deliberately modest (16/32/48/64/80), same "small backbone"
lesson as Phase 2 on EuroSAT: if the encoder is too large, the bottleneck
slot's parameter count (12-108, same budgets as the classification sweep)
is invisible against it.

Input: 128x128x4 (HSI+Edge, same preprocessing as the classification arms,
downsized from the native 512x512 tiles). Encoder downsamples 128 -> 4 (five
2x poolings), so the bottleneck is a tiny 4x4 = 16 spatial locations -- the
quantum slot runs once per location (16 x batch_size circuit evaluations per
forward pass), not once per pixel of the original image; still "non-pooling"
in that spatial structure through the bottleneck is preserved for the
decoder, but bounded to something a CPU simulator can actually run.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ENCODER_CHANNELS = [16, 32, 48, 64, 80]  # 128 -> 64 -> 32 -> 16 -> 8 -> 4
BOTTLENECK_SPATIAL = 4  # 128 / 2**len(ENCODER_CHANNELS)


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        channels = [in_channels, *ENCODER_CHANNELS]
        self.blocks = nn.ModuleList(
            [_double_conv(channels[i], channels[i + 1]) for i in range(len(ENCODER_CHANNELS))]
        )
        self.out_channels = ENCODER_CHANNELS[-1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        divisor = 2 ** len(ENCODER_CHANNELS)
        h, w = x.shape[-2], x.shape[-1]
        if h % divisor != 0 or w % divisor != 0:
            raise ValueError(
                f"UNetEncoder needs H and W divisible by {divisor} (5 poolings), "
                f"got {h}x{w}. Use a tile_size that's a multiple of {divisor}."
            )

        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        return x, skips  # x: (B, 80, 4, 4) bottleneck; skips: pre-pool feature maps


class UNetDecoder(nn.Module):
    def __init__(self, bottleneck_channels: int, n_classes: int) -> None:
        super().__init__()
        # Mirror the encoder in reverse: stage i concatenates the *previous*
        # stage's output (in_channels[i]) with its skip (skip_channels[i]),
        # then convs down to skip_channels[i] -- which becomes in_channels[i+1]
        # for the next stage. in_channels[0] is the bottleneck itself.
        skip_channels = list(reversed(ENCODER_CHANNELS))
        in_channels = [bottleneck_channels, *skip_channels[:-1]]

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.merge_convs = nn.ModuleList(
            [
                _double_conv(in_channels[i] + skip_channels[i], skip_channels[i])
                for i in range(len(ENCODER_CHANNELS))
            ]
        )
        self.final_conv = nn.Conv2d(skip_channels[-1], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for merge_conv, skip in zip(self.merge_convs, reversed(skips)):
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = merge_conv(x)
        return self.final_conv(x)  # (B, n_classes, H, W) logits
