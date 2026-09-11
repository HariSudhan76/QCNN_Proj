import pytest
import torch

from qrs.config import Config
from qrs.models.build import build_model
from qrs.models.quantum_layer import QuantumLayer


@pytest.mark.parametrize(
    "n_qubits, n_layers",
    [(2, 2), (4, 2), (6, 3), (6, 6)],  # sweep budgets: 12 / 24 / 54 / 108 params
)
def test_quantum_arm_param_count_across_sweep_budgets(n_qubits, n_layers):
    config = Config(
        arm="quantum",
        backbone_variant="small",
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    model = build_model(config)
    assert model.n_quantum_params == n_qubits * n_layers * 3


def test_quantum_layers_at_different_sizes_do_not_share_circuit_state():
    # Each QuantumLayer must build its own device/qnode/weights from its
    # constructor args -- no module- or class-level caching that would carry a
    # stale circuit from a previous (n_qubits, n_layers) into the next.
    small = QuantumLayer(n_qubits=2, n_layers=2)
    big = QuantumLayer(n_qubits=6, n_layers=3)

    assert small.qlayer.qnode.device is not big.qlayer.qnode.device
    assert small.n_quantum_params == 12
    assert big.n_quantum_params == 54
    assert small(torch.rand(2, 2)).shape == (2, 2)
    assert big(torch.rand(2, 6)).shape == (2, 6)


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
