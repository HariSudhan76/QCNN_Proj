"""Minimal explainability: Grad-CAM on the classical backbone, and quantum
input sensitivity via a parameter-shift-style finite difference.

Deliberately does NOT implement SHAP or LIME on the quantum circuit:
finite-shot randomness on real quantum hardware makes both attribution
methods unreliable for PQCs (Gil-Fuster et al., 2024).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from qrs.models.quantum_layer import QuantumLayer


class GradCAM:
    """Standard Grad-CAM against a target conv layer (e.g. model.backbone.block4)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """x: a single-sample batch, shape (1, C, H, W). Returns an (H, W)
        heatmap normalised to [0, 1]."""
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))[0, 0]

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def quantum_input_sensitivity(
    quantum_layer: QuantumLayer,
    raw_inputs: torch.Tensor,
    epsilon: float = float(np.pi / 100),
) -> torch.Tensor:
    """S_i = |f(theta_i + eps) - f(theta_i - eps)| / (2*eps) for each encoded
    angle theta_i, reusing the finite-difference machinery hardware gradients
    would need anyway (parameter-shift is exact only for gate parameters, but
    the same finite-difference probe is the standard input-sensitivity proxy).

    Args:
        raw_inputs: pre-encoding features (the compression layer's output,
            *before* the internal sigmoid*pi scaling), shape (batch, n_qubits).

    Returns:
        (batch, n_qubits) sensitivity of sum(circuit output) to each encoded angle.
    """
    theta = torch.sigmoid(raw_inputs) * torch.pi
    n_qubits = theta.shape[1]
    sensitivities = torch.zeros_like(theta)

    with torch.no_grad():
        for i in range(n_qubits):
            theta_plus = theta.clone()
            theta_minus = theta.clone()
            theta_plus[:, i] = (theta_plus[:, i] + epsilon).clamp(0.0, float(np.pi))
            theta_minus[:, i] = (theta_minus[:, i] - epsilon).clamp(0.0, float(np.pi))

            f_plus = quantum_layer.forward_from_angles(theta_plus).sum(dim=1)
            f_minus = quantum_layer.forward_from_angles(theta_minus).sum(dim=1)
            sensitivities[:, i] = (f_plus - f_minus).abs() / (2 * epsilon)

    return sensitivities
