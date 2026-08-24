from __future__ import annotations

from typing import Any


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
