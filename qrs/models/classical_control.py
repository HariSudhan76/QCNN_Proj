"""Parameter-matched classical replacement for the quantum layer -- the
mandatory control arm (CLAUDE.md rule 3). Builds a classical MLP whose
trainable parameter count is within `tolerance` of a target count and
asserts that match loudly at construction time; it must never silently
proceed with a mismatched control.
"""

from __future__ import annotations

import torch.nn as nn


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_parameter_matched_control(
    in_features: int,
    out_features: int,
    target_params: int,
    tolerance: float = 0.05,
) -> nn.Module:
    """Classical MLP, in_features -> (optional hidden) -> out_features, with a
    trainable parameter count within `tolerance` of `target_params`.

    Searches single-Linear and single-hidden-layer MLP widths, with each
    layer's bias toggled on/off independently, for the closest match --
    bias toggling gives much finer-grained achievable parameter counts than
    hidden width alone, which matters for small in/out dimensions where a
    single hidden unit already changes the count by in+out+1. Raises
    AssertionError if nothing within tolerance is found -- a hard failure,
    not a warning.
    """
    candidates: list[nn.Module] = [
        nn.Linear(in_features, out_features, bias=True),
        nn.Linear(in_features, out_features, bias=False),
    ]

    max_h = min(max(target_params, 50), 500)
    for h in range(1, max_h + 1):
        for bias1 in (True, False):
            for bias2 in (True, False):
                candidates.append(
                    nn.Sequential(
                        nn.Linear(in_features, h, bias=bias1),
                        nn.ReLU(),
                        nn.Linear(h, out_features, bias=bias2),
                    )
                )

    best = min(candidates, key=lambda m: abs(_count_params(m) - target_params))
    best_count = _count_params(best)
    rel_error = abs(best_count - target_params) / target_params

    print(
        f"[classical_control] target_params={target_params} "
        f"actual_params={best_count} rel_error={rel_error:.3%}"
    )

    assert rel_error <= tolerance, (
        f"Parameter-matched control failed: target={target_params}, "
        f"closest achievable={best_count} ({rel_error:.1%} off), tolerance={tolerance:.0%}. "
        "This is a hard requirement (CLAUDE.md rule 3) -- do not silently proceed."
    )

    return best
