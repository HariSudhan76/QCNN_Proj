"""Assembles an arm from config. The backbone, feature width, and head are
identical across arms (per CLAUDE.md rule 1) -- only the block between them
(none / quantum / parameter-matched dense / quantum+attention) differs."""

from __future__ import annotations

import torch
import torch.nn as nn

from qrs.config import Config
from qrs.data.eurosat import CLASSES
from qrs.models.attention import ChannelAttentionGate
from qrs.models.backbone import Backbone
from qrs.models.classical_control import build_parameter_matched_control
from qrs.models.heads import ClassifierHead
from qrs.models.quantum_layer import QuantumLayer, quantum_param_count


class SquashToAngleRange(nn.Module):
    """sigmoid * pi -- the exact input squashing QuantumLayer applies before
    encoding. Parameter-free; lets the `control_scaled` arm see the same [0, pi]
    input range as the quantum circuit."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * torch.pi


class ArmModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        middle: nn.Module,
        head: nn.Module,
        attention: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.attention = attention if attention is not None else nn.Identity()
        self.backbone = backbone
        self.middle = middle
        self.head = head
        # Non-quantum arms leave this at 0; quantum/control arms overwrite it
        # below with the middle block's actual quantum parameter count.
        self.n_quantum_params = getattr(middle, "n_quantum_params", 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        return self.head(self.middle(self.backbone(x)))


def build_model(config: Config) -> nn.Module:
    # `quantum_attn` is architecturally the quantum arm with the attention
    # gate forced on; `attention` itself is a generic per-config toggle
    # usable as an ablation switch on any arm.
    effective_arm = "quantum" if config.arm == "quantum_attn" else config.arm
    use_attention = config.attention or config.arm == "quantum_attn"
    attention = ChannelAttentionGate(n_channels=4) if use_attention else None

    backbone = Backbone(
        in_channels=4,
        feature_width=config.feature_width,
        variant=config.backbone_variant,
        pool_grid=config.none_pool_grid,
    )
    # Read the backbone's real output width rather than assuming config.feature_width
    # -- the "small" variant is a fixed 8/16/24 architecture whose output is 24,
    # independent of config.feature_width.
    feat_width = backbone.feature_width
    n_classes = len(CLASSES)

    if effective_arm == "classical":
        head = ClassifierHead(feat_width, n_classes)
        return ArmModel(backbone, nn.Identity(), head, attention)

    if effective_arm == "quantum":
        compression = nn.Linear(feat_width, config.n_qubits)
        quantum = QuantumLayer(
            n_qubits=config.n_qubits,
            n_layers=config.n_layers,
            entangle=config.entangle,
            data_reuploading=config.data_reuploading,
        )
        middle = nn.Sequential(compression, quantum)
        middle.n_quantum_params = quantum.n_quantum_params
        head = ClassifierHead(config.n_qubits, n_classes)
        return ArmModel(backbone, middle, head, attention)

    if effective_arm in ("control", "control_scaled"):
        compression = nn.Linear(feat_width, config.n_qubits)
        target_params = quantum_param_count(config.n_qubits, config.n_layers)
        control = build_parameter_matched_control(config.n_qubits, config.n_qubits, target_params)
        if effective_arm == "control_scaled":
            # Same as control, but the slot sees the quantum layer's [0, pi]
            # input range. Separates "quantum structure helps" from "bounded
            # input scaling helps". Zero extra parameters.
            middle = nn.Sequential(compression, SquashToAngleRange(), control)
        else:
            middle = nn.Sequential(compression, control)
        # No quantum parameters in the control arm by construction.
        middle.n_quantum_params = 0
        head = ClassifierHead(config.n_qubits, n_classes)
        return ArmModel(backbone, middle, head, attention)

    raise NotImplementedError(f"Arm {config.arm!r} not implemented yet (fused: September).")
