"""Efficiency profiler for the built arms.

No training: builds a model from a config (the same `Config` / `build_model`
path `qrs.run` uses) and measures it on four axes kept separate on purpose --
parameters, storage, inference memory, and latency -- plus a FLOPs / quantum
gate-ops breakdown.

    python profile_models.py --arms quantum control \
        --backbone-variant small --n-qubits 6 --n-layers 3

`--config X.yaml` supplies a base config; the CLI flags override it, so any
arm / backbone / qubit / layer combination can be profiled, not just the
defaults.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import io
import time

import psutil
import torch
from torch.utils.flop_counter import FlopCounterMode

from qrs.config import Config, load_config
from qrs.models.build import build_model

ALL_ARMS = ["classical", "quantum", "control", "quantum_attn"]

RESULT_FIELDS = [
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


_OVERRIDE_FLAGS = ("backbone_variant", "n_qubits", "n_layers", "batch_size", "image_size")


def build_config(arm: str, args: argparse.Namespace) -> Config:
    """Same config-building path as qrs.run: start from a YAML base if given,
    else Config() defaults, then apply only the CLI flags the user actually
    passed. Every override flag defaults to None so an unset flag doesn't
    silently clobber a value the YAML base set -- only an explicit --flag
    value should ever win over --config."""
    base = load_config(args.config) if args.config else Config()
    overrides = {
        field: getattr(args, field) for field in _OVERRIDE_FLAGS if getattr(args, field) is not None
    }
    return dataclasses.replace(base, arm=arm, **overrides)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum = int(getattr(model, "n_quantum_params", 0))
    return total, quantum


def storage_mb(model: torch.nn.Module) -> float:
    """On-disk checkpoint size: the serialized state_dict, in MB."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024**2)


def inference_memory_mb(model: torch.nn.Module, x: torch.Tensor) -> float:
    """Process RSS growth across a no-grad forward pass, in MB. Approximate on
    CPU -- there is no allocator-level peak tracker like CUDA's -- but good for
    a relative comparison between arms measured the same way."""
    proc = psutil.Process()
    gc.collect()
    before = proc.memory_info().rss
    with torch.no_grad():
        for _ in range(3):
            model(x)
    after = proc.memory_info().rss
    return max(0.0, (after - before) / (1024**2))


def latency_ms_fps(model: torch.nn.Module, x: torch.Tensor, iters: int) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(3):  # warm-up
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        elapsed = time.perf_counter() - t0
    per_image_ms = (elapsed / iters) / x.shape[0] * 1000.0
    fps = 1000.0 / per_image_ms if per_image_ms > 0 else float("inf")
    return per_image_ms, fps


def classical_gflops(model: torch.nn.Module, x: torch.Tensor) -> float:
    """FLOPs of the torch-dispatch-visible ops (backbone, projections, head,
    attention, classical-control MLP), for the given batch. The quantum
    layer's cost is NOT here -- lightning.qubit runs in C++, invisible to the
    FLOP counter -- it is reported separately as quantum_sim_gflops."""
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    return counter.get_total_flops() / 1e9


def quantum_gate_ops_per_forward(config: Config) -> int:
    """Quantum gate applications per circuit evaluation (one image). 0 for
    non-quantum arms. Encoding: RY per qubit (per layer if data re-uploading).
    Variational: a Rot per qubit per layer, plus a CNOT ring per layer when
    entangling."""
    if config.arm not in ("quantum", "quantum_attn"):
        return 0
    q, layers = config.n_qubits, config.n_layers
    encode = q * layers if config.data_reuploading else q
    rot = q * layers
    cnot = q * layers if config.entangle else 0
    return encode + rot + cnot


def quantum_sim_gflops(config: Config, batch_size: int) -> float:
    """Rough state-vector simulation cost: every gate updates the full
    2**n_qubits complex amplitude vector; ~8 flops per complex multiply-add.
    Scales linearly with batch size (one circuit evaluation per image)."""
    if config.arm not in ("quantum", "quantum_attn"):
        return 0.0
    gates = quantum_gate_ops_per_forward(config)
    state_dim = 2**config.n_qubits
    flops = gates * state_dim * 8 * batch_size
    return flops / 1e9


def profile_arm(config: Config, iters: int) -> dict:
    model = build_model(config)
    model.eval()
    x = torch.rand(config.batch_size, 4, config.image_size, config.image_size)

    total_params, quantum_params = count_params(model)
    latency_ms, fps = latency_ms_fps(model, x, iters)

    return {
        "arm": config.arm,
        "backbone_variant": config.backbone_variant,
        "n_qubits": config.n_qubits,
        "n_layers": config.n_layers,
        "params": total_params,
        "quantum_params": quantum_params,
        "storage_mb": storage_mb(model),
        "infer_mem_mb": inference_memory_mb(model, x),
        "latency_ms_per_image": latency_ms,
        "fps": fps,
        "classical_gflops": classical_gflops(model, x),
        "quantum_gate_ops": quantum_gate_ops_per_forward(config) * config.batch_size,
        "quantum_sim_gflops": quantum_sim_gflops(config, config.batch_size),
    }


def _format_table(rows: list[dict]) -> str:
    widths = {f: len(f) for f in RESULT_FIELDS}
    cells: list[dict[str, str]] = []
    for row in rows:
        formatted = {}
        for f in RESULT_FIELDS:
            v = row[f]
            if isinstance(v, float):
                s = f"{v:.4f}"
            else:
                s = str(v)
            formatted[f] = s
            widths[f] = max(widths[f], len(s))
        cells.append(formatted)

    header = "  ".join(f.ljust(widths[f]) for f in RESULT_FIELDS)
    lines = [header, "  ".join("-" * widths[f] for f in RESULT_FIELDS)]
    for formatted in cells:
        lines.append("  ".join(formatted[f].ljust(widths[f]) for f in RESULT_FIELDS))
    return "\n".join(lines)


def _write_csv(rows: list[dict], path: str) -> None:
    import csv

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    # Every override flag below defaults to None (not a Config default value)
    # so build_config can tell "user didn't pass this" apart from "user
    # explicitly chose the same value the default happens to have" -- only
    # the former should fall through to --config / Config() untouched.
    parser.add_argument("--backbone-variant", default=None, choices=["large", "small"])
    parser.add_argument("--n-qubits", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--iters", type=int, default=50, help="forward passes timed for latency")
    parser.add_argument("--config", default=None, help="base YAML config; CLI flags override it")
    parser.add_argument("--csv", default=None, help="also write rows to this CSV path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rows = [profile_arm(build_config(arm, args), args.iters) for arm in args.arms]
    print(_format_table(rows))
    if args.csv:
        _write_csv(rows, args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
