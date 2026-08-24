"""CLI entrypoint: python -m qrs.run --config configs/X.yaml

Trains every seed in config.seeds for the arm named in the config, appending
one row per seed to config.results_csv.
"""

from __future__ import annotations

import argparse

from qrs.analysis.results import append_result, get_git_sha
from qrs.config import Config, load_config
from qrs.data.loaders import build_dataloaders
from qrs.models.build import build_model
from qrs.seeds import set_all_seeds
from qrs.train.loop import train_model


def run(config: Config) -> None:
    git_sha = get_git_sha()

    for seed in config.seeds:
        set_all_seeds(seed)

        train_loader, val_loader, test_loader = build_dataloaders(config, seed)
        model = build_model(config)
        metrics = train_model(model, config, train_loader, val_loader, test_loader)
        metrics.pop("confusion_matrix", None)

        row = {"arm": config.arm, "dataset": config.dataset, "seed": seed, "git_sha": git_sha, **metrics}
        append_result(row, config.results_csv)

        print(
            f"[seed={seed}] f1_weighted={metrics['f1_weighted']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} epochs_run={metrics['epochs_run']} "
            f"train_wallclock_s={metrics['train_wallclock_s']:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
