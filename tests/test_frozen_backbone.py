import pytest
import torch

pytest.importorskip("torchvision")

from qrs.models.frozen_backbone import FrozenResNet18Backbone  # noqa: E402


def test_frozen_backbone_has_zero_trainable_params():
    backbone = FrozenResNet18Backbone()
    assert sum(p.numel() for p in backbone.parameters() if p.requires_grad) == 0
    assert sum(p.numel() for p in backbone.parameters()) > 10_000_000  # still ~11.7M frozen weights


def test_frozen_backbone_output_shape_matches_feature_width():
    backbone = FrozenResNet18Backbone()
    out = backbone(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, backbone.feature_width) == (2, 512)


def test_frozen_backbone_stays_in_eval_mode_even_if_train_is_called():
    backbone = FrozenResNet18Backbone()
    backbone.train()  # must not flip BatchNorm to train mode
    assert not backbone.net.training
    assert not backbone.training


def test_frozen_backbone_deterministic_given_input():
    # No dropout, frozen BatchNorm stats -- same input, same output regardless
    # of how many times it's called.
    backbone = FrozenResNet18Backbone()
    x = torch.rand(2, 3, 64, 64)
    assert torch.equal(backbone(x), backbone(x))
