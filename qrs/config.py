"""Config dataclasses and YAML loader. No hyperparameters are hardcoded elsewhere."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass
class EarlyStoppingConfig:
    monitor: str = "val_loss"
    patience: int = 5


@dataclasses.dataclass
class Config:
    arm: str = "classical"

    dataset: str = "eurosat"
    data_dir: str = "data/eurosat"
    cache_dir: str = "data/cache"
    image_size: int = 64
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)

    epochs: int = 30
    batch_size: int = 32
    optimizer: str = "adam"
    lr: float = 1.0e-3

    early_stopping: EarlyStoppingConfig = dataclasses.field(default_factory=EarlyStoppingConfig)

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    feature_width: int = 128

    n_qubits: int = 8
    n_layers: int = 3
    entangle: bool = True
    data_reuploading: bool = False

    attention: bool = False

    results_csv: str = "results/results.csv"
    checkpoint_dir: str = "results/checkpoints"

    def __post_init__(self) -> None:
        if isinstance(self.split, list):
            self.split = tuple(self.split)
        if isinstance(self.seeds, list):
            self.seeds = tuple(self.seeds)
        if isinstance(self.early_stopping, dict):
            self.early_stopping = EarlyStoppingConfig(**self.early_stopping)

        if abs(sum(self.split) - 1.0) > 1e-6:
            raise ValueError(f"split must sum to 1.0, got {self.split}")
        if self.n_qubits < 2:
            raise ValueError(f"n_qubits must be >= 2, got {self.n_qubits}")
        valid_arms = {"classical", "quantum", "control", "quantum_attn", "fused"}
        if self.arm not in valid_arms:
            raise ValueError(f"arm must be one of {valid_arms}, got {self.arm!r}")


def load_config(path: str | Path) -> Config:
    """Load a Config from YAML. Supports a single-level `extends: other.yaml`
    (resolved relative to `path`'s directory) so per-arm configs only need to
    state what differs from the shared base — every other setting stays
    identical across arms, per the project's controlled-comparison rule."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if extends:
        base = load_config(path.parent / extends)
        merged = dataclasses.asdict(base)
        merged.update(raw)
        raw = merged

    return Config(**raw)
