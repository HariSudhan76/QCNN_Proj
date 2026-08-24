"""Classifier heads. Identical across arms; only the block feeding into them differs."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
