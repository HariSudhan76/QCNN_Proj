"""Tests for profile_models.py's CLI/config-threading and the analytic
gate-ops / GFLOPs formulas. No real profiling runs (no latency/memory
timing) -- those are exercised manually, not as part of the test suite."""

import profile_models as pm
from qrs.config import Config


def _parse(argv):
    return pm.build_arg_parser().parse_args(argv)


def test_no_flags_no_config_yields_config_defaults():
    # Every override flag must default to None in argparse (not a Config
    # default value) so build_config can distinguish "flag not passed" from
    # "user explicitly chose the value that happens to match the default" --
    # otherwise an unset flag would always clobber --config's value.
    args = _parse([])
    assert args.arms == pm.ALL_ARMS
    for flag in ("backbone_variant", "n_qubits", "n_layers", "batch_size", "image_size"):
        assert getattr(args, flag) is None

    config = pm.build_config("classical", args)
    assert config.backbone_variant == Config.backbone_variant
    assert config.n_qubits == Config.n_qubits
    assert config.n_layers == Config.n_layers
    assert config.batch_size == Config.batch_size
    assert config.image_size == Config.image_size


def test_flags_thread_into_built_config():
    args = _parse(
        [
            "--arms",
            "quantum",
            "control",
            "--backbone-variant",
            "small",
            "--n-qubits",
            "6",
            "--n-layers",
            "3",
            "--batch-size",
            "16",
        ]
    )
    assert args.arms == ["quantum", "control"]

    config = pm.build_config("quantum", args)
    assert config.arm == "quantum"
    assert config.backbone_variant == "small"
    assert config.n_qubits == 6
    assert config.n_layers == 3
    assert config.batch_size == 16


def test_unset_flags_do_not_clobber_yaml_base():
    # Regression test: an earlier version defaulted --backbone-variant etc.
    # to Config's default value rather than None, so build_config always
    # "overrode" the YAML base with that default even when the user never
    # passed the flag -- silently defeating --config for those fields.
    yaml_path_args = _parse(["--config", "unused.yaml"])  # no override flags passed
    assert yaml_path_args.backbone_variant is None
    assert yaml_path_args.n_qubits is None
    assert yaml_path_args.n_layers is None
    assert yaml_path_args.batch_size is None


def test_config_flag_overrides_yaml_base(tmp_path):
    yaml_path = tmp_path / "base.yaml"
    yaml_path.write_text("backbone_variant: small\nn_qubits: 2\nn_layers: 2\nbatch_size: 4\n")

    args = _parse(["--config", str(yaml_path), "--n-qubits", "6", "--n-layers", "6"])
    config = pm.build_config("quantum", args)

    # CLI flags win over the YAML base for the fields the CLI controls.
    assert config.n_qubits == 6
    assert config.n_layers == 6
    # Fields the CLI doesn't touch pass through from the YAML base.
    assert config.backbone_variant == "small"
    assert config.batch_size == 4


def test_invalid_arm_rejected_by_argparse():
    import pytest

    with pytest.raises(SystemExit):
        _parse(["--arms", "not_a_real_arm"])


def test_gate_ops_and_sim_gflops_zero_for_non_quantum_arms():
    for arm in ("classical", "control"):
        config = Config(arm=arm, n_qubits=6, n_layers=3)
        assert pm.quantum_gate_ops_per_forward(config) == 0
        assert pm.quantum_sim_gflops(config, batch_size=32) == 0.0


def test_gate_ops_formula_no_reuploading_no_entangle():
    config = Config(arm="quantum", n_qubits=6, n_layers=3, entangle=False, data_reuploading=False)
    # encode once (6) + Rot per qubit per layer (18) + no CNOTs
    assert pm.quantum_gate_ops_per_forward(config) == 6 + 18


def test_gate_ops_formula_entangle_and_reuploading():
    config = Config(arm="quantum", n_qubits=4, n_layers=2, entangle=True, data_reuploading=True)
    # re-encode each layer (4*2=8) + Rot per qubit per layer (8) + CNOT ring per layer (8)
    assert pm.quantum_gate_ops_per_forward(config) == 8 + 8 + 8


def test_sim_gflops_scales_with_state_dimension_and_batch():
    config = Config(arm="quantum", n_qubits=4, n_layers=2, entangle=True, data_reuploading=False)
    gates = pm.quantum_gate_ops_per_forward(config)  # 4 + 8 + 8 = 20
    expected = gates * (2**4) * 8 * 32 / 1e9
    assert pm.quantum_sim_gflops(config, batch_size=32) == expected


def test_result_fields_are_stable():
    # Locks the output schema so a future edit can't silently drop/rename a
    # column without a test failure.
    assert pm.RESULT_FIELDS == [
        "arm",
        "backbone_variant",
        "n_qubits",
        "n_layers",
        "params",
        "quantum_params",
        "storage_mb",
        "infer_mem_mb",
        "latency_ms_per_image",
        "fps",
        "classical_gflops",
        "quantum_gate_ops",
        "quantum_sim_gflops",
    ]
