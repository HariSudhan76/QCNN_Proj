import pytest
import torch

from qrs.config import Config
from qrs.models.build import build_model


def test_classical_arm_forward_shape():
    config = Config(arm="classical", feature_width=16)
    model = build_model(config)
    x = torch.rand(4, 4, 64, 64)
    out = model(x)
    assert out.shape == (4, 10)  # 10 EuroSAT classes


def test_quantum_arm_forward_shape_and_param_counts():
    config = Config(arm="quantum", feature_width=16, n_qubits=4, n_layers=2)
    model = build_model(config)
    x = torch.rand(2, 4, 64, 64)
    out = model(x)
    assert out.shape == (2, 10)
    assert model.n_quantum_params == 2 * 4 * 3  # n_layers * n_qubits * 3
    assert model.n_quantum_params < sum(p.numel() for p in model.parameters())


def test_control_arm_forward_shape_and_matches_quantum_param_count():
    config = Config(arm="control", feature_width=16, n_qubits=4, n_layers=2)
    quantum_config = Config(arm="quantum", feature_width=16, n_qubits=4, n_layers=2)

    model = build_model(config)
    quantum_model = build_model(quantum_config)

    x = torch.rand(2, 4, 64, 64)
    out = model(x)
    assert out.shape == (2, 10)
    assert model.n_quantum_params == 0

    control_middle_params = sum(p.numel() for p in model.middle.parameters() if p.requires_grad)
    quantum_middle_params = sum(
        p.numel() for p in quantum_model.middle.parameters() if p.requires_grad
    )
    # control's middle (compression + control MLP) should be within 5% of
    # quantum's middle (compression + quantum layer) param count.
    assert abs(control_middle_params - quantum_middle_params) / quantum_middle_params <= 0.05


def test_unimplemented_arm_raises():
    config = Config(arm="fused")
    with pytest.raises(NotImplementedError):
        build_model(config)


def test_attention_gate_applied_and_exposes_last_weights():
    config = Config(arm="classical", feature_width=16, attention=True)
    model = build_model(config)
    x = torch.rand(3, 4, 64, 64)
    out = model(x)
    assert out.shape == (3, 10)
    assert model.attention.last_weights.shape == (3, 4)
    assert torch.all((model.attention.last_weights >= 0) & (model.attention.last_weights <= 1))


def test_quantum_attn_arm_forces_attention_on():
    config = Config(arm="quantum_attn", feature_width=16, n_qubits=4, n_layers=2)
    model = build_model(config)
    x = torch.rand(2, 4, 64, 64)
    out = model(x)
    assert out.shape == (2, 10)
    assert model.n_quantum_params == 2 * 4 * 3
    assert model.attention.last_weights is not None
