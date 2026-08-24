import pytest
import torch

from qrs.models.classical_control import build_parameter_matched_control


@pytest.mark.parametrize("target_params", [24, 72, 144, 288])
def test_matches_target_within_tolerance(target_params):
    model = build_parameter_matched_control(8, 8, target_params, tolerance=0.05)
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert abs(actual - target_params) / target_params <= 0.05


def test_forward_shape():
    model = build_parameter_matched_control(8, 8, 72)
    out = model(torch.rand(5, 8))
    assert out.shape == (5, 8)


def test_impossible_target_raises_assertion_loudly():
    # A single Linear(1000, 1000) already has ~1e6 params -- impossible to
    # get within 5% of a target of 1.
    with pytest.raises(AssertionError):
        build_parameter_matched_control(1000, 1000, target_params=1, tolerance=0.05)
