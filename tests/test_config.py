import pytest

from qrs.config import Config, load_config
from qrs.models.build import build_model


def test_defaults_valid():
    config = Config()
    assert config.arm == "classical"
    assert config.split == (0.70, 0.15, 0.15)
    assert config.backbone_variant == "large"  # backward-compatible default


def test_sweep_fields_load_from_yaml_and_reach_build_model(tmp_path):
    yaml_path = tmp_path / "sweep_point.yaml"
    yaml_path.write_text(
        "arm: quantum\n"
        "backbone_variant: small\n"
        "n_qubits: 4\n"
        "n_layers: 2\n"
    )
    config = load_config(yaml_path)
    assert config.backbone_variant == "small"
    assert config.n_qubits == 4
    assert config.n_layers == 2

    model = build_model(config)
    assert model.backbone.feature_width == 24  # small backbone
    assert model.n_quantum_params == 4 * 2 * 3


def test_extends_merge_preserves_sweep_fields(tmp_path):
    (tmp_path / "base.yaml").write_text("backbone_variant: small\nn_layers: 6\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\narm: quantum\nn_qubits: 6\n")
    config = load_config(tmp_path / "child.yaml")
    assert config.backbone_variant == "small"
    assert config.n_qubits == 6
    assert config.n_layers == 6


def test_invalid_backbone_variant_rejected():
    with pytest.raises(ValueError):
        Config(backbone_variant="medium")


def test_invalid_n_layers_rejected():
    with pytest.raises(ValueError):
        Config(n_layers=0)


def test_arm_classical_yaml_extends_base():
    config = load_config("configs/arm_classical.yaml")
    assert config.arm == "classical"
    assert config.epochs == 30
    assert config.n_qubits == 8
    assert config.feature_width == 128


def test_invalid_arm_rejected():
    with pytest.raises(ValueError):
        Config(arm="not_a_real_arm")


def test_invalid_split_rejected():
    with pytest.raises(ValueError):
        Config(split=(0.5, 0.5, 0.5))
