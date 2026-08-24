import torch

from qrs.config import Config
from qrs.models.build import build_model
from qrs.train.callbacks import EarlyStopping, checkpoint_path, load_checkpoint, save_checkpoint


def test_stops_after_patience_epochs_without_improvement():
    es = EarlyStopping(patience=2, mode="min")
    values = [1.0, 0.9, 0.95, 0.96, 0.97]  # improves, then plateaus
    stopped_at = None
    for i, v in enumerate(values):
        if es.step(v):
            stopped_at = i
            break
    assert stopped_at == 3  # two epochs (idx 2, 3) without improvement after best at idx 1


def test_tracks_best_value():
    es = EarlyStopping(patience=5, mode="min")
    for v in [1.0, 0.5, 0.8]:
        es.step(v)
    assert es.best == 0.5


def test_checkpoint_path_naming(tmp_path):
    path = checkpoint_path(tmp_path, "classical", 3)
    assert path == tmp_path / "classical_seed3.pt"


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    config = Config(arm="classical", feature_width=16)
    model = build_model(config)
    x = torch.rand(2, 4, 64, 64)
    with torch.no_grad():
        original_out = model(x)

    path = checkpoint_path(tmp_path, config.arm, seed=0)
    save_checkpoint(model, path)
    assert path.exists()

    fresh_model = build_model(config)
    load_checkpoint(fresh_model, path)
    with torch.no_grad():
        loaded_out = fresh_model(x)

    assert torch.allclose(original_out, loaded_out)
