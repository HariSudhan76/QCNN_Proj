import csv

import pytest

from qrs.analysis.results import RESULT_COLUMNS, aggregate_results


@pytest.fixture
def multi_seed_csv(tmp_path):
    csv_path = tmp_path / "results.csv"
    rows = [
        {
            "arm": "classical",
            "dataset": "eurosat",
            "seed": seed,
            "n_trainable_params": 243114,
            "n_quantum_params": 0,
            "f1_weighted": f1,
            "accuracy": f1,
            "precision": f1,
            "recall": f1,
            "train_wallclock_s": 100.0,
            "inference_wallclock_s": 5.0,
            "epochs_run": 20,
            "git_sha": "abc123",
        }
        for seed, f1 in enumerate([0.80, 0.82, 0.84])
    ] + [
        {
            "arm": "quantum",
            "dataset": "eurosat",
            "seed": seed,
            "n_trainable_params": 243040,
            "n_quantum_params": 72,
            "f1_weighted": f1,
            "accuracy": f1,
            "precision": f1,
            "recall": f1,
            "train_wallclock_s": 400.0,
            "inference_wallclock_s": 20.0,
            "epochs_run": 20,
            "git_sha": "abc123",
        }
        for seed, f1 in enumerate([0.75, 0.77, 0.79])
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def test_aggregate_has_one_row_per_arm(multi_seed_csv):
    agg = aggregate_results(multi_seed_csv)
    assert set(agg["arm"]) == {"classical", "quantum"}
    assert len(agg) == 2


def test_aggregate_mean_and_std_correct(multi_seed_csv):
    agg = aggregate_results(multi_seed_csv).set_index("arm")
    assert agg.loc["classical", "f1_weighted_mean"] == pytest.approx(0.82, abs=1e-9)
    assert agg.loc["classical", "n_seeds"] == 3
    assert agg.loc["quantum", "n_quantum_params_mean"] == 72
    assert agg.loc["classical", "f1_weighted_std"] > 0
