"""Parameterised quantum circuit (PQC) wrapped as a torch.nn.Module.

Simulator: `lightning.qubit` (never `default.qubit` -- far slower). Differentiation:
`adjoint`, which is exact and fast on a simulator but is not available on real
hardware; real hardware would instead need `parameter-shift`.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn


def quantum_param_count(n_qubits: int, n_layers: int) -> int:
    """Trainable parameter count of the variational block: a Rot (3 angles)
    per qubit per layer, independent of the entangle/data_reuploading toggles
    (both change gate structure, not parameter count)."""
    return n_layers * n_qubits * 3


class QuantumLayer(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        entangle: bool = True,
        data_reuploading: bool = False,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangle = entangle
        self.data_reuploading = data_reuploading

        device = qml.device("lightning.qubit", wires=n_qubits)
        qnode = qml.QNode(self._circuit, device, interface="torch", diff_method="adjoint")

        if entangle:
            # StronglyEntanglingLayers: a full Rot (RZ-RY-RZ) per qubit per layer
            # plus ring-entangling CNOTs. Deliberately NOT an RZ-only trainable
            # layer -- a lone RZ commutes with a Z-basis measurement on its own
            # qubit, making that parameterisation nearly dead for this readout.
            weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        else:
            # Ablation: single-qubit rotations only, no entangling gates. Same
            # per-qubit Rot parameterisation as StronglyEntanglingLayers so the
            # only thing removed is entanglement.
            weight_shape = (n_layers, n_qubits, 3)

        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": weight_shape})

        # Second qnode sharing the same device, taking angles already scaled
        # to [0, pi] directly (skips the sigmoid) -- used by
        # analysis.explain.quantum_input_sensitivity, which must perturb the
        # actual encoded angle rather than the pre-scaling raw feature.
        self._raw_qnode = qml.QNode(
            self._raw_circuit, device, interface="torch", diff_method="adjoint"
        )

    def _encode(self, inputs: torch.Tensor) -> None:
        # RY angle encoding, one feature per qubit. Inputs must land in [0, pi];
        # sigmoid guarantees that regardless of the compression layer's raw range.
        scaled = torch.sigmoid(inputs) * torch.pi
        qml.templates.AngleEmbedding(scaled, wires=range(self.n_qubits), rotation="Y")

    def _variational_layer(self, weights: torch.Tensor) -> None:
        """Apply one layer's worth of `weights`, shape (n_qubits, 3)."""
        for wire in range(self.n_qubits):
            qml.Rot(*weights[wire], wires=wire)
        if self.entangle:
            for wire in range(self.n_qubits):
                qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor):
        if self.data_reuploading:
            # Highest-value expressivity upgrade for a small qubit budget:
            # repeat the encoding block between variational layers instead of
            # encoding once up front.
            for layer in range(self.n_layers):
                self._encode(inputs)
                self._variational_layer(weights[layer])
        else:
            self._encode(inputs)
            if self.entangle:
                qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            else:
                for layer in range(self.n_layers):
                    self._variational_layer(weights[layer])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlayer(x)

    def _raw_circuit(self, theta: torch.Tensor, weights: torch.Tensor):
        """Same as _circuit but `theta` is used as-is (already in [0, pi]),
        no sigmoid scaling. Non-reuploading variational block only -- this
        entry point exists for explainability probes, not training."""
        qml.templates.AngleEmbedding(theta, wires=range(self.n_qubits), rotation="Y")
        if self.entangle:
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        else:
            for layer in range(self.n_layers):
                self._variational_layer(weights[layer])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward_from_angles(self, theta: torch.Tensor) -> torch.Tensor:
        """theta already scaled to [0, pi]. See `_raw_circuit`."""
        if self.data_reuploading:
            raise NotImplementedError(
                "forward_from_angles only supports the non-reuploading variational block."
            )
        # Unlike TorchLayer, a bare QNode returns one tensor per measurement
        # rather than a single stacked tensor -- stack them here.
        return torch.stack(self._raw_qnode(theta, self.qlayer.weights), dim=-1)

    @property
    def n_quantum_params(self) -> int:
        return sum(p.numel() for p in self.qlayer.parameters())
