"""HSI channel attention gate: squeeze-and-excitation over the 4 input
channels, applied before the backbone. Toggleable by config (`attention`).

`.last_weights` exposes the most recent forward's per-sample channel gate
values -- their per-class distribution is a reportable explainability result.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttentionGate(nn.Module):
    def __init__(self, n_channels: int = 4, reduction: int = 2) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(n_channels, reduction)
        self.fc2 = nn.Linear(reduction, n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.gap(x).flatten(1)  # (B, n_channels)
        w = self.sigmoid(self.fc2(self.relu(self.fc1(s))))  # (B, n_channels)
        self.last_weights = w.detach()
        return x * w.view(*w.shape, 1, 1)
