"""Assembles a U-Net segmentation model from config. Reuses config.arm's
existing classical/quantum/control vocabulary (config.task disambiguates
classification vs segmentation) rather than inventing a parallel arm-name
scheme, and reuses QuantumLayer / build_parameter_matched_control /
quantum_param_count unchanged from the classification build path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qrs.config import Config
from qrs.data.landcover import CLASSES
from qrs.models.classical_control import build_parameter_matched_control
from qrs.models.quantum_layer import QuantumLayer, quantum_param_count
from qrs.models.unet import UNetDecoder, UNetEncoder


class SpatialSlot(nn.Module):
    """Applies compress -> slot -> expand independently at every spatial
    location of a (B, C, H, W) feature map -- the "non-pooling" bottleneck:
    spatial structure passes through untouched, only the per-location
    channel vector is transformed, the same role a 1x1 conv would play, but
    through the quantum/control slot instead."""

    def __init__(self, channels: int, slot: nn.Module, n_slot_features: int) -> None:
        super().__init__()
        self.compress = nn.Linear(channels, n_slot_features)
        self.slot = slot
        self.expand = nn.Linear(n_slot_features, channels)
        self.n_quantum_params = getattr(slot, "n_quantum_params", 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        flat = self.expand(self.slot(self.compress(flat)))
        return flat.reshape(b, h, w, c).permute(0, 3, 1, 2)


class UNetArmModel(nn.Module):
    def __init__(self, encoder: nn.Module, bottleneck: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck  # nn.Identity() for the classical (unconstrained) arm
        self.decoder = decoder
        self.n_quantum_params = getattr(bottleneck, "n_quantum_params", 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_feat, skips = self.encoder(x)
        bottleneck_feat = self.bottleneck(bottleneck_feat)
        return self.decoder(bottleneck_feat, skips)


def build_unet_model(config: Config) -> nn.Module:
    encoder = UNetEncoder(in_channels=4)
    channels = encoder.out_channels
    n_classes = len(CLASSES)
    decoder = UNetDecoder(bottleneck_channels=channels, n_classes=n_classes)

    if config.arm == "classical":
        return UNetArmModel(encoder, nn.Identity(), decoder)

    if config.arm == "quantum":
        quantum = QuantumLayer(
            n_qubits=config.n_qubits,
            n_layers=config.n_layers,
            entangle=config.entangle,
            data_reuploading=config.data_reuploading,
        )
        bottleneck = SpatialSlot(channels, quantum, config.n_qubits)
        return UNetArmModel(encoder, bottleneck, decoder)

    if config.arm == "control":
        target_params = quantum_param_count(config.n_qubits, config.n_layers)
        control = build_parameter_matched_control(config.n_qubits, config.n_qubits, target_params)
        bottleneck = SpatialSlot(channels, control, config.n_qubits)
        # No quantum parameters in the control arm by construction.
        bottleneck.n_quantum_params = 0
        return UNetArmModel(encoder, bottleneck, decoder)

    raise NotImplementedError(
        f"Arm {config.arm!r} not supported for segmentation (classical/quantum/control only)."
    )
