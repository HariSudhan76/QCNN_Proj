from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class EarlyStopping:
    """Tracks the monitored value's best epoch and its model state (checkpoint).

    `step(value)` returns True once `patience` epochs have passed without
    improvement -- the caller should break out of the training loop.
    """

    def __init__(self, patience: int = 5, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.best_state: dict[str, Any] | None = None
        self.counter = 0
        self.should_stop = False

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best

    def step(self, value: float) -> bool:
        if self.is_improvement(value):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def checkpoint_path(checkpoint_dir: str | Path, arm: str, seed: int) -> Path:
    return Path(checkpoint_dir) / f"{arm}_seed{seed}.pt"


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str | Path, map_location: str = "cpu") -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model
