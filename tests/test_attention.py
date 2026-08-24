import torch

from qrs.models.attention import ChannelAttentionGate


def test_output_shape_matches_input():
    gate = ChannelAttentionGate(n_channels=4, reduction=2)
    x = torch.rand(5, 4, 16, 16)
    out = gate(x)
    assert out.shape == x.shape


def test_gate_scales_channels_multiplicatively():
    gate = ChannelAttentionGate(n_channels=4, reduction=2)
    x = torch.rand(2, 4, 8, 8)
    out = gate(x)
    w = gate.last_weights
    assert w.shape == (2, 4)
    assert torch.allclose(out, x * w.view(2, 4, 1, 1), atol=1e-6)


def test_weights_in_unit_interval():
    gate = ChannelAttentionGate(n_channels=4, reduction=2)
    gate(torch.rand(8, 4, 10, 10))
    assert torch.all((gate.last_weights >= 0) & (gate.last_weights <= 1))
