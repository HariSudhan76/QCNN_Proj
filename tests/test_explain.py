import numpy as np
import torch

from qrs.analysis.explain import GradCAM, quantum_input_sensitivity
from qrs.config import Config
from qrs.models.build import build_model
from qrs.models.quantum_layer import QuantumLayer


def test_gradcam_heatmap_shape_and_range():
    config = Config(arm="classical", feature_width=16)
    model = build_model(config)
    x = torch.rand(1, 4, 64, 64)

    cam = GradCAM(model, model.backbone.block4)
    heatmap = cam(x)

    assert heatmap.shape == (8, 8)  # spatial size after 3 maxpools on a 64x64 input
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    assert np.isfinite(heatmap).all()


def test_quantum_input_sensitivity_shape_and_nonzero():
    n_qubits, n_layers = 4, 2
    layer = QuantumLayer(n_qubits, n_layers, entangle=True)
    raw_inputs = torch.rand(3, n_qubits)

    sens = quantum_input_sensitivity(layer, raw_inputs)

    assert sens.shape == (3, n_qubits)
    assert torch.isfinite(sens).all()
    assert torch.any(sens > 0)


def test_sensitivity_reuploading_raises():
    layer = QuantumLayer(4, 2, entangle=True, data_reuploading=True)
    raw_inputs = torch.rand(2, 4)
    try:
        quantum_input_sensitivity(layer, raw_inputs)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised
