import torch
from torch.utils.data import DataLoader, TensorDataset

from qrs.config import Config
from qrs.data.landcover import CLASSES
from qrs.models.unet_build import build_unet_model
from qrs.train.segmentation_loop import train_segmentation_model

N_CLASSES = len(CLASSES)


def _synthetic_loaders(n_samples=8, size=32, batch_size=4):
    x = torch.rand(n_samples, 4, size, size)
    y = torch.randint(0, N_CLASSES, (n_samples, size, size))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size)
    return loader, loader, loader  # same tiny set for train/val/test is fine for a smoke test


def test_classical_arm_trains_and_returns_expected_keys():
    config = Config(
        arm="classical",
        task="segmentation",
        epochs=2,
        early_stopping={"monitor": "val_loss", "patience": 2},
    )
    model = build_unet_model(config)
    train_loader, val_loader, test_loader = _synthetic_loaders()

    metrics = train_segmentation_model(
        model, config, train_loader, val_loader, test_loader, verbose=False
    )

    for key in (
        "miou",
        "pixel_accuracy",
        "iou_per_class",
        "n_trainable_params",
        "n_quantum_params",
        "train_wallclock_s",
        "inference_wallclock_s",
        "epochs_run",
    ):
        assert key in metrics
    assert metrics["epochs_run"] == 2
    assert metrics["n_quantum_params"] == 0
    assert 0.0 <= metrics["miou"] <= 1.0 or metrics["miou"] != metrics["miou"]  # allow NaN


def test_quantum_arm_trains_end_to_end_small():
    config = Config(
        arm="quantum",
        task="segmentation",
        n_qubits=2,
        n_layers=1,
        epochs=1,
        batch_size=2,
    )
    model = build_unet_model(config)
    train_loader, val_loader, test_loader = _synthetic_loaders(n_samples=4, size=32, batch_size=2)

    metrics = train_segmentation_model(
        model, config, train_loader, val_loader, test_loader, verbose=False
    )
    assert metrics["n_quantum_params"] == 2 * 1 * 3
    assert metrics["epochs_run"] == 1


def test_loss_decreases_over_epochs_on_learnable_synthetic_task():
    # A real (if tiny) check that training actually learns something, not
    # just that the loop runs without crashing: build a task where the mask
    # is a deterministic function of the input (top-left quadrant vs rest),
    # easy enough for a few epochs to visibly reduce loss.
    torch.manual_seed(0)
    size = 32
    n_samples = 16
    x = torch.rand(n_samples, 4, size, size)
    y = torch.zeros(n_samples, size, size, dtype=torch.long)
    y[:, : size // 2, : size // 2] = 1  # top-left quadrant is class 1, rest class 0
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    config = Config(
        arm="classical",
        task="segmentation",
        epochs=8,
        lr=1e-2,
        early_stopping={"monitor": "val_loss", "patience": 8},
    )
    model = build_unet_model(config)

    from qrs.train.segmentation_loop import _run_epoch

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss()

    first_loss, _ = _run_epoch(model, loader, optimizer, criterion, "cpu", train=True)
    for _ in range(6):
        _run_epoch(model, loader, optimizer, criterion, "cpu", train=True)
    last_loss, _ = _run_epoch(model, loader, optimizer, criterion, "cpu", train=True)

    assert last_loss < first_loss
