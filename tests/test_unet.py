import pytest
import torch

from qrs.config import Config
from qrs.data.landcover import CLASSES
from qrs.models.unet import BOTTLENECK_SPATIAL, ENCODER_CHANNELS, UNetDecoder, UNetEncoder
from qrs.models.unet_build import SpatialSlot, build_unet_model


def test_encoder_output_shape_and_bottleneck_spatial_size():
    encoder = UNetEncoder(in_channels=4)
    x = torch.rand(2, 4, 128, 128)
    bottleneck, skips = encoder(x)

    assert bottleneck.shape == (2, ENCODER_CHANNELS[-1], BOTTLENECK_SPATIAL, BOTTLENECK_SPATIAL)
    assert len(skips) == len(ENCODER_CHANNELS)
    # skip spatial sizes halve each stage: 128, 64, 32, 16, 8
    expected_sizes = [128 // (2**i) for i in range(len(ENCODER_CHANNELS))]
    for skip, ch, size in zip(skips, ENCODER_CHANNELS, expected_sizes):
        assert skip.shape == (2, ch, size, size)


def test_decoder_reconstructs_input_spatial_size():
    encoder = UNetEncoder(in_channels=4)
    decoder = UNetDecoder(bottleneck_channels=encoder.out_channels, n_classes=len(CLASSES))
    x = torch.rand(2, 4, 128, 128)
    bottleneck, skips = encoder(x)
    logits = decoder(bottleneck, skips)
    assert logits.shape == (2, len(CLASSES), 128, 128)


def test_spatial_slot_preserves_shape_and_transforms_per_location():
    n_qubits = 4
    slot = torch.nn.Linear(n_qubits, n_qubits)  # stand-in slot for shape testing
    spatial_slot = SpatialSlot(channels=8, slot=slot, n_slot_features=n_qubits)

    x = torch.rand(2, 8, 4, 4)
    out = spatial_slot(x)
    assert out.shape == x.shape


@pytest.mark.parametrize("arm", ["classical", "quantum", "control"])
def test_unet_arms_build_and_forward(arm):
    config = Config(arm=arm, task="segmentation", n_qubits=4, n_layers=2, tile_size=32)
    model = build_unet_model(config)
    x = torch.rand(2, 4, 32, 32)
    out = model(x)
    assert out.shape == (2, len(CLASSES), 32, 32)


def test_quantum_and_control_arms_are_parameter_matched():
    quantum_config = Config(arm="quantum", task="segmentation", n_qubits=4, n_layers=2)
    control_config = Config(arm="control", task="segmentation", n_qubits=4, n_layers=2)

    quantum_model = build_unet_model(quantum_config)
    control_model = build_unet_model(control_config)

    quantum_total = sum(p.numel() for p in quantum_model.parameters() if p.requires_grad)
    control_total = sum(p.numel() for p in control_model.parameters() if p.requires_grad)

    assert quantum_total == control_total
    assert quantum_model.n_quantum_params == 4 * 2 * 3
    assert control_model.n_quantum_params == 0


def test_classical_arm_has_no_quantum_params_and_fewer_total_params():
    classical_config = Config(arm="classical", task="segmentation")
    quantum_config = Config(arm="quantum", task="segmentation", n_qubits=6, n_layers=3)

    classical_model = build_unet_model(classical_config)
    quantum_model = build_unet_model(quantum_config)

    classical_total = sum(p.numel() for p in classical_model.parameters() if p.requires_grad)
    quantum_total = sum(p.numel() for p in quantum_model.parameters() if p.requires_grad)

    assert classical_model.n_quantum_params == 0
    # classical has no compress/expand around the bottleneck at all, so it's
    # smaller than quantum's compress+slot+expand, not just missing the slot.
    assert classical_total < quantum_total


def test_invalid_segmentation_arm_raises():
    config = Config(arm="quantum_attn", task="segmentation")
    with pytest.raises(NotImplementedError):
        build_unet_model(config)


def test_quantum_arm_gradients_flow_through_bottleneck_and_encoder():
    # Debug-ladder check (per CLAUDE.md): gradients w.r.t. quantum parameters
    # must be non-zero, and the encoder itself must still receive gradient
    # through the bottleneck, not just the slot in isolation.
    config = Config(arm="quantum", task="segmentation", n_qubits=4, n_layers=2, tile_size=32)
    model = build_unet_model(config)
    x = torch.rand(2, 4, 32, 32)
    mask = torch.randint(0, len(CLASSES), (2, 32, 32))

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, mask)
    loss.backward()

    quantum_weights = next(model.bottleneck.slot.qlayer.parameters())
    assert quantum_weights.grad is not None
    assert torch.any(quantum_weights.grad != 0)

    first_encoder_param = next(model.encoder.parameters())
    assert first_encoder_param.grad is not None
    assert torch.any(first_encoder_param.grad != 0)
