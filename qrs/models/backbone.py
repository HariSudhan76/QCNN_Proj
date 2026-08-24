"""Shared CNN trunk, identical across every arm. Operates on the 4-channel
(H, S, I, Edge) tensor produced by qrs.data.preprocessing."""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Backbone(nn.Module):
    def __init__(self, in_channels: int = 4, feature_width: int = 128) -> None:
        super().__init__()
        self.block1 = _conv_block(in_channels, 32)
        self.block2 = _conv_block(32, 64)
        self.block3 = _conv_block(64, 128)
        self.block4 = _conv_block(128, feature_width)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block1(x))  # 64 -> 32
        x = self.pool(self.block2(x))  # 32 -> 16
        x = self.pool(self.block3(x))  # 16 -> 8
        x = self.block4(x)
        x = self.gap(x)  # (B, feature_width, 1, 1)
        return x.flatten(1)  # (B, feature_width)
