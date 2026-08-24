"""CSV logging: one row per (arm, seed, dataset, epoch-final), single file."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pandas as pd

RESULT_COLUMNS = [
    "arm",
    "dataset",
    "seed",
    "n_trainable_params",
    "n_quantum_params",
    "f1_weighted",
    "accuracy",
    "precision",
    "recall",
    "train_wallclock_s",
    "inference_wallclock_s",
    "epochs_run",
    "git_sha",
]

# Metrics aggregated as mean +/- std across seeds; params/epochs are
# constant-ish per config so mean is reported without a std column.
AGGREGATE_METRIC_COLUMNS = [
    "f1_weighted",
    "accuracy",
    "precision",
    "recall",
    "train_wallclock_s",
    "inference_wallclock_s",
]
AGGREGATE_MEAN_ONLY_COLUMNS = ["n_trainable_params", "n_quantum_params", "epochs_run"]


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "no-git"


def append_result(row: dict, csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col) for col in RESULT_COLUMNS})


def aggregate_results(csv_path: str | Path) -> pd.DataFrame:
    """Mean +/- std across seeds, grouped by (arm, dataset). One row per arm."""
    df = pd.read_csv(csv_path)

    agg = {col: ["mean", "std"] for col in AGGREGATE_METRIC_COLUMNS}
    agg.update({col: ["mean"] for col in AGGREGATE_MEAN_ONLY_COLUMNS})
    agg["seed"] = ["count"]

    grouped = df.groupby(["arm", "dataset"]).agg(agg)
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped = grouped.rename(columns={"seed_count": "n_seeds"}).reset_index()
    return grouped
