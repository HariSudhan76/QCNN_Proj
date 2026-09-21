"""Frozen pretrained ResNet-18 backbone for the high-accuracy RGB experiment.

Unlike qrs.models.backbone.Backbone, this operates on 3-channel RGB (not the
4-channel HSI+Edge tensor) because pretrained ImageNet weights are defined
for RGB. Weights are frozen (requires_grad=False) and BatchNorm is pinned to
eval mode, so this backbone contributes zero trainable parameters and is
identical, static, across every arm -- like backbone_variant="none", except
it supplies pretrained features instead of raw pooled channel averages.

Separate module (not a Backbone variant) because it needs torchvision, a
different input channel count, and pretrained-weight downloading -- none of
which the HSI+Edge backbones require.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenResNet18Backbone(nn.Module):
    feature_width = 512

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18  # noqa: PLC0415

        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        net.fc = nn.Identity()
        for p in net.parameters():
            p.requires_grad = False
        self.net = net.eval()

    def train(self, mode: bool = True) -> "FrozenResNet18Backbone":
        # Always eval: BatchNorm running stats must never update, and there
        # are no trainable parameters for train mode to affect.
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
