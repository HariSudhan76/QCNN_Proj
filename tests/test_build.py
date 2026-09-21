import pytest
import torch

from qrs.config import Config
from qrs.models.build import build_model


@pytest.mark.parametrize("arm", ["classical", "quantum", "control", "quantum_attn"])
@pytest.mark.parametrize("backbone_variant", ["large", "small", "none"])
def test_all_arms_build_with_all_backbone_variants(arm, backbone_variant):
    # 4 arms x 3 backbone variants = 12 combinations; each must build and do
    # a forward pass with no shape error, whether the backbone emits 128-d
    # (large), 24-d (small, trained), or 24-d (none, fixed/untrained).
    config = Config(
        arm=arm,
        backbone_variant=backbone_variant,
        n_qubits=4,
        n_layers=2,
    )
    model = build_model(config)
    out = model(torch.rand(2, 4, 64, 64))
    assert out.shape == (2, 10)


def test_no_backbone_variant_isolates_slot_as_dominant_param_fraction():
    # The whole point of "none": with zero backbone params, the
    # quantum/control slot should be a large fraction of the total model,
    # not diluted by a trained feature extractor.
    config = Config(arm="quantum", backbone_variant="none", n_qubits=6, n_layers=3)
    model = build_model(config)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert model.n_quantum_params / total > 0.15  # ~20% in practice (54/274)


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


def test_grid4x4_configs_build_with_64d_input():
    from qrs.config import load_config
    from qrs.models.build import build_model

    for arm in ("quantum", "control", "classical"):
        cfg = load_config(f"configs/phase2/grid4x4_{arm}.yaml")
        assert cfg.none_pool_grid == (4, 4)
        model = build_model(cfg)
        assert model.backbone.feature_width == 64
    q = build_model(load_config("configs/phase2/grid4x4_quantum.yaml"))
    assert q.n_quantum_params == 54


def test_control_scaled_matches_control_params_and_squashes_input():
    import torch

    from qrs.config import load_config
    from qrs.models.build import SquashToAngleRange, build_model

    ctl = build_model(load_config("configs/nobackbone_control.yaml"))
    scaled = build_model(load_config("configs/nobackbone_control_scaled.yaml"))
    n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)  # noqa: E731
    assert n(ctl) == n(scaled) == 274
    assert scaled.n_quantum_params == 0
    assert any(isinstance(m, SquashToAngleRange) for m in scaled.middle)

    out = SquashToAngleRange()(torch.tensor([-100.0, 0.0, 100.0]))
    assert out.min() >= 0 and out.max() <= torch.pi
    assert torch.isclose(out[1], torch.tensor(torch.pi / 2))

    grid = build_model(load_config("configs/phase2/grid4x4_control_scaled.yaml"))
    assert n(grid) == 514


def test_frozen_resnet18_backbone_builds_across_arms():
    import pytest as _pytest

    _pytest.importorskip("torchvision")
    from qrs.config import Config
    from qrs.models.build import build_model

    for arm in ("classical", "quantum", "control"):
        config = Config(arm=arm, backbone_variant="frozen_resnet18", n_qubits=6, n_layers=3)
        model = build_model(config)
        assert model.backbone.feature_width == 512
        # Frozen backbone contributes zero trainable parameters.
        assert sum(p.numel() for p in model.backbone.parameters() if p.requires_grad) == 0


def test_frozen_resnet18_rejects_attention():
    import pytest as _pytest

    _pytest.importorskip("torchvision")
    from qrs.config import Config
    from qrs.models.build import build_model

    config = Config(arm="quantum", backbone_variant="frozen_resnet18", attention=True)
    with _pytest.raises(ValueError, match="attention"):
        build_model(config)
