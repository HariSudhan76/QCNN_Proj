"""Single source of truth for reproducibility. All randomness must flow through set_all_seeds."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed Python `random`, NumPy, and PyTorch (CPU + CUDA).

    The PennyLane device seed is set separately at circuit-construction time
    (qml.device(..., seed=seed)) since it is bound to a specific device instance,
    not a global RNG.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
