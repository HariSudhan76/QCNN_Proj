import torch

from qrs.models.backbone import Backbone


def _param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_small_backbone_param_count_in_target_range():
    backbone = Backbone(in_channels=4, variant="small")
    n = _param_count(backbone)
    assert 4500 <= n <= 5500, f"small backbone has {n} params, expected 4500-5500"


def test_small_backbone_output_width_is_24():
    backbone = Backbone(in_channels=4, variant="small")
    assert backbone.feature_width == 24
    out = backbone(torch.rand(2, 4, 64, 64))
    assert out.shape == (2, 24)


def test_large_backbone_unchanged():
    # feature_width still drives the large variant's block4 width, and the
    # output width equals it -- Phase 1 behaviour.
    backbone = Backbone(in_channels=4, feature_width=128, variant="large")
    assert backbone.feature_width == 128
    out = backbone(torch.rand(2, 4, 64, 64))
    assert out.shape == (2, 128)
    # Large backbone param count is the Phase 1 figure (~242k), far above small.
    assert _param_count(backbone) > 200_000


def test_none_backbone_has_zero_learnable_parameters():
    backbone = Backbone(in_channels=4, variant="none")
    assert _param_count(backbone) == 0


def test_none_backbone_output_width_matches_small():
    # Same 24-d output width as "small", for direct comparability between
    # "quantum/control fed a trained 24-d summary" and "...a fixed one".
    backbone = Backbone(in_channels=4, variant="none")
    assert backbone.feature_width == 24
    out = backbone(torch.rand(2, 4, 64, 64))
    assert out.shape == (2, 24)


def test_none_backbone_output_is_deterministic_given_input():
    # No learnable parameters and no randomness (e.g. dropout) -- same input
    # must give bit-identical output regardless of training/eval mode or
    # multiple calls, unlike "small"/"large" whose BatchNorm running stats
    # can shift between calls in train mode.
    backbone = Backbone(in_channels=4, variant="none")
    x = torch.rand(2, 4, 64, 64)
    assert torch.equal(backbone(x), backbone(x))


def test_invalid_variant_rejected():
    import pytest

    with pytest.raises(ValueError):
        Backbone(variant="medium")


def test_none_backbone_pool_grid_sets_width():
    backbone = Backbone(in_channels=4, variant="none", pool_grid=(4, 4))
    assert backbone.feature_width == 64
    assert backbone(torch.rand(2, 4, 64, 64)).shape == (2, 64)
    assert _param_count(backbone) == 0


def test_none_backbone_default_grid_unchanged():
    assert Backbone(in_channels=4, variant="none").feature_width == 24
