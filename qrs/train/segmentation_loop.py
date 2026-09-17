"""Training loop for segmentation arms (task="segmentation"). Mirrors
qrs.train.loop's structure and CLI-visible behaviour (early stopping,
per-epoch progress logging, checkpoint restore) but accumulates a confusion
matrix per epoch instead of concatenating full pixel tensors -- segmentation
datasets are pixel-heavy (a single 128x128 tile is 16,384 "samples"), so
holding every pixel's prediction in memory for a whole epoch doesn't scale
the way it did for per-image classification labels."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from qrs.config import Config
from qrs.data.landcover import CLASSES
from qrs.train.callbacks import EarlyStopping
from qrs.train.segmentation_metrics import metrics_from_confusion_matrix

N_CLASSES = len(CLASSES)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: str,
    train: bool,
) -> tuple[float, np.ndarray]:
    model.train(train)
    total_loss = 0.0
    n_samples = 0
    cm_total = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    labels_range = list(range(N_CLASSES))

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()
            logits = model(x)  # (B, N_CLASSES, H, W)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

            preds = logits.argmax(dim=1).detach().cpu().numpy().ravel()
            labels = y.detach().cpu().numpy().ravel()
            cm_total += confusion_matrix(labels, preds, labels=labels_range)

    avg_loss = total_loss / n_samples
    return avg_loss, cm_total


def train_segmentation_model(
    model: nn.Module,
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    if config.optimizer != "adam":
        raise NotImplementedError(f"optimizer {config.optimizer!r} not supported")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config.early_stopping.patience, mode="min")

    epochs_run = 0
    train_start = time.time()
    for epoch in range(config.epochs):
        epoch_start = time.time()
        train_loss, _ = _run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, _ = _run_epoch(model, val_loader, None, criterion, device, train=False)
        epochs_run = epoch + 1

        if verbose:
            print(
                f"    epoch {epochs_run}/{config.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"epoch_time={time.time() - epoch_start:.1f}s "
                f"total_elapsed={time.time() - train_start:.1f}s",
                flush=True,
            )

        if early_stopping.is_improvement(val_loss):
            early_stopping.best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
        if early_stopping.step(val_loss):
            if verbose:
                print(
                    f"    early stopping at epoch {epochs_run} "
                    f"(no val_loss improvement for {config.early_stopping.patience} epochs)",
                    flush=True,
                )
            break
    train_wallclock_s = time.time() - train_start

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    inference_start = time.time()
    _, test_cm = _run_epoch(model, test_loader, None, criterion, device, train=False)
    inference_wallclock_s = time.time() - inference_start

    metrics = metrics_from_confusion_matrix(test_cm)
    metrics["confusion_matrix"] = test_cm
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_quantum_params = getattr(model, "n_quantum_params", 0)

    return {
        **metrics,
        "n_trainable_params": n_trainable_params,
        "n_quantum_params": n_quantum_params,
        "train_wallclock_s": train_wallclock_s,
        "inference_wallclock_s": inference_wallclock_s,
        "epochs_run": epochs_run,
    }
