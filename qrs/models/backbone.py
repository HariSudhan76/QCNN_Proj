"""Shared CNN trunk, identical across every arm. Operates on the 4-channel
(H, S, I, Edge) tensor produced by qrs.data.preprocessing.

Three variants:
  - "large": the original 4->32->64->128->feature_width, 4 conv blocks
    (~242k params at feature_width=128). Default, so Phase 1 configs and
    results stay reproducible.
  - "small": 4->8->16->24, 3 conv blocks (~5k params). Phase 2 needs the
    quantum/control slot to be a measurable fraction of the whole model,
    which it is not against a 242k-param backbone.
  - "none": zero learnable parameters -- fixed (non-trained) average pooling
    straight from the raw 4-channel input to a 24-d vector, matching
    "small"'s output width for direct comparability. Answers a different
    question than "small" does: with no trained feature extractor anywhere
    upstream of the slot, can the quantum/control block itself separate on a
    raw, non-adapted summary of the image? Rules out "the backbone is doing
    the real work and hiding the slot's contribution" as an explanation for
    a null result.

`self.feature_width` is the actual output width in all three cases --
callers that need to size a projection off the backbone should read that
attribute rather than assume 128.
"""

from __future__ import annotations

import torch
import torch.nn as nn

BACKBONE_VARIANTS = ("large", "small", "none")
_NONE_VARIANT_POOL_GRID = (2, 3)  # 4 channels x 2x3 = 24-d, matching "small"


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        feature_width: int = 128,
        variant: str = "large",
    ) -> None:
        super().__init__()
        if variant not in BACKBONE_VARIANTS:
            raise ValueError(f"variant must be one of {BACKBONE_VARIANTS}, got {variant!r}")
        self.variant = variant
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if variant == "large":
            self.block1 = _conv_block(in_channels, 32)
            self.block2 = _conv_block(32, 64)
            self.block3 = _conv_block(64, 128)
            self.block4 = _conv_block(128, feature_width)
            self.feature_width = feature_width
        elif variant == "small":
            self.block1 = _conv_block(in_channels, 8)
            self.block2 = _conv_block(8, 16)
            self.block3 = _conv_block(16, 24)
            self.feature_width = 24
        else:  # "none": no blocks, no learnable parameters at all
            self.fixed_pool = nn.AdaptiveAvgPool2d(_NONE_VARIANT_POOL_GRID)
            self.feature_width = in_channels * _NONE_VARIANT_POOL_GRID[0] * _NONE_VARIANT_POOL_GRID[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "large":
            x = self.pool(self.block1(x))  # 64 -> 32
            x = self.pool(self.block2(x))  # 32 -> 16
            x = self.pool(self.block3(x))  # 16 -> 8
            x = self.block4(x)  # not pooled, matches the original large backbone
            x = self.gap(x)  # (B, feature_width, 1, 1)
            return x.flatten(1)
        elif self.variant == "small":
            x = self.pool(self.block1(x))  # 64 -> 32
            x = self.pool(self.block2(x))  # 32 -> 16
            x = self.pool(self.block3(x))  # 16 -> 8
            x = self.gap(x)  # (B, feature_width, 1, 1)
            return x.flatten(1)
        else:  # "none"
            x = self.fixed_pool(x)  # (B, in_channels, 2, 3), no learnable params
            return x.flatten(1)  # (B, feature_width)
