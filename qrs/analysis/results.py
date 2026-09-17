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

# Segmentation (LandCover.ai) uses different metrics (mIoU, pixel accuracy)
# entirely, not a superset/subset of the classification metrics -- kept as a
# SEPARATE results file (config.results_csv defaults differently per task)
# rather than widening RESULT_COLUMNS, since there's already a live
# classification results.csv in the wild with the schema above: appending
# new columns to RESULT_COLUMNS wouldn't rewrite that file's existing header,
# so new rows would silently carry more values than the header has names for.
SEGMENTATION_RESULT_COLUMNS = [
    "arm",
    "dataset",
    "seed",
    "n_trainable_params",
    "n_quantum_params",
    "miou",
    "pixel_accuracy",
    "train_wallclock_s",
    "inference_wallclock_s",
    "epochs_run",
    "git_sha",
]

# Metrics aggregated as mean +/- std across seeds; params/epochs are
# constant-ish per config so mean is reported without a std column. Checked
# against whichever columns are actually present in a given results CSV
# (classification vs segmentation), so aggregate_results works on either.
_POSSIBLE_AGGREGATE_METRIC_COLUMNS = [
    "f1_weighted",
    "accuracy",
    "precision",
    "recall",
    "miou",
    "pixel_accuracy",
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


def append_result(row: dict, csv_path: str | Path, columns: list[str] = RESULT_COLUMNS) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col) for col in columns})


def aggregate_results(csv_path: str | Path) -> pd.DataFrame:
    """Mean +/- std across seeds, grouped by (arm, dataset). One row per arm.
    Works on either a classification or segmentation results CSV -- the
    metric columns aggregated are whichever of the possible ones are
    actually present, rather than a schema fixed in advance."""
    df = pd.read_csv(csv_path)

    metric_columns = [c for c in _POSSIBLE_AGGREGATE_METRIC_COLUMNS if c in df.columns]
    agg = {col: ["mean", "std"] for col in metric_columns}
    agg.update({col: ["mean"] for col in AGGREGATE_MEAN_ONLY_COLUMNS if col in df.columns})
    agg["seed"] = ["count"]

    grouped = df.groupby(["arm", "dataset"]).agg(agg)
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped = grouped.rename(columns={"seed_count": "n_seeds"}).reset_index()
    return grouped
