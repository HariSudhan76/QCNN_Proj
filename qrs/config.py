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
    task: str = "classification"  # "classification" | "segmentation"

    dataset: str = "eurosat"
    data_dir: str = "data/eurosat"
    cache_dir: str = "data/cache"
    image_size: int = 64
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)

    # Segmentation (LandCover.ai) only. data_dir is the (typically read-only,
    # Kaggle-attached) raw dataset root; tiles_dir is where the writable
    # tile_orthophotos() cache goes.
    tiles_dir: str = "data/landcover_tiles"
    tile_size: int = 128

    epochs: int = 30
    batch_size: int = 32
    optimizer: str = "adam"
    lr: float = 1.0e-3

    early_stopping: EarlyStoppingConfig = dataclasses.field(default_factory=EarlyStoppingConfig)

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    backbone_variant: str = "large"
    feature_width: int = 128
    # Only used by backbone_variant "none": AdaptiveAvgPool2d grid, giving
    # 4 channels x grid_h x grid_w features. Default keeps the original 24-d.
    none_pool_grid: tuple[int, int] = (2, 3)

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
        if isinstance(self.none_pool_grid, list):
            self.none_pool_grid = tuple(self.none_pool_grid)
        if len(self.none_pool_grid) != 2 or min(self.none_pool_grid) < 1:
            raise ValueError(f"none_pool_grid must be two positive ints, got {self.none_pool_grid}")
        if isinstance(self.seeds, list):
            self.seeds = tuple(self.seeds)
        if isinstance(self.early_stopping, dict):
            self.early_stopping = EarlyStoppingConfig(**self.early_stopping)

        if abs(sum(self.split) - 1.0) > 1e-6:
            raise ValueError(f"split must sum to 1.0, got {self.split}")
        if self.n_qubits < 2:
            raise ValueError(f"n_qubits must be >= 2, got {self.n_qubits}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        valid_arms = {"classical", "quantum", "control", "control_scaled", "quantum_attn", "fused"}
        if self.arm not in valid_arms:
            raise ValueError(f"arm must be one of {valid_arms}, got {self.arm!r}")
        valid_variants = {"large", "small", "none", "frozen_resnet18"}
        if self.backbone_variant not in valid_variants:
            raise ValueError(
                f"backbone_variant must be one of {valid_variants}, got {self.backbone_variant!r}"
            )
        valid_tasks = {"classification", "segmentation"}
        if self.task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got {self.task!r}")


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
