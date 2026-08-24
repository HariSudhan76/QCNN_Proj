"""The single training loop used by every arm. No per-arm branching lives
here -- arms differ only in the model built by qrs.models.build.build_model."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from qrs.config import Config
from qrs.train.callbacks import EarlyStopping
from qrs.train.metrics import compute_metrics


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: str,
    train: bool,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.train(train)
    total_loss = 0.0
    n_samples = 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            all_preds.append(logits.argmax(dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

    avg_loss = total_loss / n_samples
    return avg_loss, torch.cat(all_preds), torch.cat(all_labels)


def train_model(
    model: nn.Module,
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
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
        _run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, _, _ = _run_epoch(model, val_loader, None, criterion, device, train=False)
        epochs_run = epoch + 1

        if early_stopping.is_improvement(val_loss):
            early_stopping.best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
        if early_stopping.step(val_loss):
            break
    train_wallclock_s = time.time() - train_start

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    inference_start = time.time()
    _, test_preds, test_labels = _run_epoch(
        model, test_loader, None, criterion, device, train=False
    )
    inference_wallclock_s = time.time() - inference_start

    metrics = compute_metrics(test_labels.numpy(), test_preds.numpy())
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
