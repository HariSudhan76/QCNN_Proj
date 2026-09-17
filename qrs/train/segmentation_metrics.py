"""Segmentation metrics: mIoU and pixel accuracy, computed from an aggregated
confusion matrix over ALL evaluated pixels (not a per-batch average of IoU --
that biases rare classes; standard segmentation-benchmark practice is to sum
confusion matrices across batches, then compute IoU once from the total)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def metrics_from_confusion_matrix(cm: np.ndarray) -> dict:
    """cm: (num_classes, num_classes) confusion matrix, rows=true, cols=pred.
    A class with zero union (never true, never predicted) contributes NaN to
    its own IoU and is excluded from the mIoU mean, not counted as 0."""
    cm = np.asarray(cm, dtype=np.float64)
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        iou_per_class = np.where(union > 0, intersection / union, np.nan)

    return {
        "miou": float(np.nanmean(iou_per_class)),
        "pixel_accuracy": float(intersection.sum() / cm.sum()),
        "iou_per_class": iou_per_class.tolist(),
    }


def compute_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """y_true, y_pred: flat per-pixel class-index arrays."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics = metrics_from_confusion_matrix(cm)
    metrics["confusion_matrix"] = cm
    return metrics
