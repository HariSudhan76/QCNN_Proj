"""CLI entrypoint: python -m qrs.run --config configs/X.yaml

Trains every seed in config.seeds for the arm named in the config, appending
one row per seed to config.results_csv. Dispatches on config.task:
"classification" (EuroSAT, F1/precision/recall) or "segmentation"
(LandCover.ai, mIoU/pixel accuracy) -- separate data pipelines, models, loops
and results-CSV schemas, since the two tasks share nothing but the
arm-name vocabulary (classical/quantum/control) and the project's overall
methodology (identical conditions, parameter-matched control, seeded runs).
"""

from __future__ import annotations

import argparse

from qrs.analysis.results import (
    RESULT_COLUMNS,
    SEGMENTATION_RESULT_COLUMNS,
    append_result,
    get_git_sha,
)
from qrs.config import Config, load_config
from qrs.data.landcover import prepare_landcover
from qrs.data.landcover_loaders import build_landcover_dataloaders
from qrs.data.loaders import build_dataloaders
from qrs.models.build import build_model
from qrs.models.unet_build import build_unet_model
from qrs.seeds import set_all_seeds
from qrs.train.callbacks import checkpoint_path, save_checkpoint
from qrs.train.loop import train_model
from qrs.train.segmentation_loop import train_segmentation_model


def _run_classification_seed(config: Config, seed: int) -> tuple:
    train_loader, val_loader, test_loader = build_dataloaders(config, seed)
    model = build_model(config)
    metrics = train_model(model, config, train_loader, val_loader, test_loader)
    return model, metrics


def _run_segmentation_seed(config: Config, seed: int) -> tuple:
    tiles_dir, split = prepare_landcover(config.data_dir, config.tiles_dir, config.tile_size)
    train_loader, val_loader, test_loader = build_landcover_dataloaders(
        tiles_dir, split.train, split.val, split.test, config.cache_dir, config.batch_size
    )
    model = build_unet_model(config)
    metrics = train_segmentation_model(model, config, train_loader, val_loader, test_loader)
    return model, metrics


def run(config: Config) -> None:
    git_sha = get_git_sha()
    result_columns = RESULT_COLUMNS if config.task == "classification" else SEGMENTATION_RESULT_COLUMNS

    for seed in config.seeds:
        print(f"[seed={seed}] starting (arm={config.arm}, task={config.task})", flush=True)
        set_all_seeds(seed)

        if config.task == "classification":
            model, metrics = _run_classification_seed(config, seed)
        elif config.task == "segmentation":
            model, metrics = _run_segmentation_seed(config, seed)
        else:
            raise NotImplementedError(f"task {config.task!r} not supported")

        metrics.pop("confusion_matrix", None)
        row = {"arm": config.arm, "dataset": config.dataset, "seed": seed, "git_sha": git_sha, **metrics}
        append_result(row, config.results_csv, columns=result_columns)

        ckpt_path = checkpoint_path(config.checkpoint_dir, config.arm, seed)
        save_checkpoint(model, ckpt_path)

        if config.task == "classification":
            summary = (
                f"f1_weighted={metrics['f1_weighted']:.4f} accuracy={metrics['accuracy']:.4f}"
            )
        else:
            summary = f"miou={metrics['miou']:.4f} pixel_accuracy={metrics['pixel_accuracy']:.4f}"

        print(
            f"[seed={seed}] done {summary} epochs_run={metrics['epochs_run']} "
            f"train_wallclock_s={metrics['train_wallclock_s']:.1f} "
            f"checkpoint={ckpt_path}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
