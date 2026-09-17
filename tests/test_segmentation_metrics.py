import numpy as np

from qrs.train.segmentation_metrics import compute_segmentation_metrics, metrics_from_confusion_matrix


def test_perfect_prediction_gives_miou_one():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    metrics = compute_segmentation_metrics(y_true, y_pred, num_classes=3)
    assert metrics["miou"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0


def test_completely_wrong_prediction_gives_miou_zero():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    metrics = compute_segmentation_metrics(y_true, y_pred, num_classes=2)
    assert metrics["miou"] == 0.0
    assert metrics["pixel_accuracy"] == 0.0


def test_known_hand_computed_iou():
    # y_true=[0,0,0,0,1,1], y_pred=[0,0,0,1,1,0] -> cm=[[3,1],[1,1]]
    # class 0: intersection=3, union=row0(4)+col0(4)-3=5 -> IoU=3/5
    # class 1: intersection=1, union=row1(2)+col1(2)-1=3 -> IoU=1/3
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0])
    metrics = compute_segmentation_metrics(y_true, y_pred, num_classes=2)
    assert metrics["iou_per_class"][0] == 3 / 5
    assert abs(metrics["iou_per_class"][1] - 1 / 3) < 1e-9
    assert abs(metrics["miou"] - (3 / 5 + 1 / 3) / 2) < 1e-9


def test_absent_class_excluded_from_miou_not_counted_as_zero():
    # class 2 never appears in true or pred -- union is 0, must be NaN'd out
    # of the mean, not silently scored as 0 (which would unfairly punish a
    # model just because a class wasn't present in this particular batch).
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    metrics = compute_segmentation_metrics(y_true, y_pred, num_classes=3)
    assert metrics["miou"] == 1.0  # only classes 0,1 count, both perfect
    assert np.isnan(metrics["iou_per_class"][2])


def test_metrics_from_confusion_matrix_matches_compute_segmentation_metrics():
    y_true = np.array([0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 2, 2, 1])
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    from_cm = metrics_from_confusion_matrix(cm)
    from_arrays = compute_segmentation_metrics(y_true, y_pred, num_classes=3)
    assert from_cm["miou"] == from_arrays["miou"]
    assert from_cm["pixel_accuracy"] == from_arrays["pixel_accuracy"]


def test_confusion_matrices_are_additive_across_batches():
    # The segmentation training loop accumulates cm_total += cm_batch across
    # an epoch's batches; verify that's equivalent to computing one cm over
    # all pixels at once, not just "seems reasonable".
    y_true_a, y_pred_a = np.array([0, 0, 1]), np.array([0, 1, 1])
    y_true_b, y_pred_b = np.array([1, 2, 2]), np.array([1, 2, 0])

    from sklearn.metrics import confusion_matrix

    cm_a = confusion_matrix(y_true_a, y_pred_a, labels=[0, 1, 2])
    cm_b = confusion_matrix(y_true_b, y_pred_b, labels=[0, 1, 2])
    summed = metrics_from_confusion_matrix(cm_a + cm_b)

    all_true = np.concatenate([y_true_a, y_true_b])
    all_pred = np.concatenate([y_pred_a, y_pred_b])
    combined = compute_segmentation_metrics(all_true, all_pred, num_classes=3)

    assert summed["miou"] == combined["miou"]
    assert summed["pixel_accuracy"] == combined["pixel_accuracy"]
