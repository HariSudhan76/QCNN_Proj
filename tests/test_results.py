import csv

from qrs.analysis.results import RESULT_COLUMNS, append_result


def test_append_result_writes_header_once_and_rows(tmp_path):
    csv_path = tmp_path / "results.csv"
    row = {
        "arm": "classical",
        "dataset": "eurosat",
        "seed": 0,
        "n_trainable_params": 123,
        "n_quantum_params": 0,
        "f1_weighted": 0.9,
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "train_wallclock_s": 1.0,
        "inference_wallclock_s": 0.1,
        "epochs_run": 5,
        "git_sha": "abc123",
    }

    append_result(row, csv_path)
    append_result({**row, "seed": 1}, csv_path)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["seed"] == "0"
    assert rows[1]["seed"] == "1"
    assert list(rows[0].keys()) == RESULT_COLUMNS


def test_append_result_ignores_extra_keys(tmp_path):
    csv_path = tmp_path / "results.csv"
    append_result({"arm": "classical", "confusion_matrix": "should be dropped"}, csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert "confusion_matrix" not in rows[0]
