from qrs.config import Config, load_config


def test_defaults_valid():
    config = Config()
    assert config.arm == "classical"
    assert config.split == (0.70, 0.15, 0.15)


def test_arm_classical_yaml_extends_base():
    config = load_config("configs/arm_classical.yaml")
    assert config.arm == "classical"
    assert config.epochs == 30
    assert config.n_qubits == 8
    assert config.feature_width == 128


def test_invalid_arm_rejected():
    import pytest

    with pytest.raises(ValueError):
        Config(arm="not_a_real_arm")


def test_invalid_split_rejected():
    import pytest

    with pytest.raises(ValueError):
        Config(split=(0.5, 0.5, 0.5))
