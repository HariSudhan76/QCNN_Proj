import pytest
import torch

from qrs.models.quantum_layer import QuantumLayer


@pytest.mark.parametrize("entangle", [True, False])
@pytest.mark.parametrize("data_reuploading", [False, True])
def test_forward_backward_shapes_and_grads(entangle, data_reuploading):
    n_qubits, n_layers, batch = 4, 2, 3
    layer = QuantumLayer(n_qubits, n_layers, entangle=entangle, data_reuploading=data_reuploading)

    x = torch.rand(batch, n_qubits, requires_grad=True)
    out = layer(x)

    assert out.shape == (batch, n_qubits)
    assert torch.isfinite(out).all()

    out.sum().backward()

    assert x.grad is not None
    assert torch.any(x.grad != 0)
    for p in layer.qlayer.parameters():
        assert p.grad is not None
        assert torch.any(p.grad != 0)


def test_n_quantum_params_matches_weight_count():
    n_qubits, n_layers = 6, 3
    layer = QuantumLayer(n_qubits, n_layers, entangle=True)
    assert layer.n_quantum_params == n_layers * n_qubits * 3


def test_entangle_false_has_no_cnots_but_same_param_count():
    n_qubits, n_layers = 4, 2
    entangled = QuantumLayer(n_qubits, n_layers, entangle=True)
    unentangled = QuantumLayer(n_qubits, n_layers, entangle=False)
    assert entangled.n_quantum_params == unentangled.n_quantum_params
