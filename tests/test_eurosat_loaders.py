import numpy as np
import torch
from PIL import Image

from qrs.data.loaders import EuroSATDataset


def _make_tile(dir_, cls, name):
    (dir_ / cls).mkdir(exist_ok=True)
    Image.fromarray((np.random.default_rng(0).random((8, 8, 3)) * 255).astype("uint8")).save(
        dir_ / cls / name
    )


def test_euro_sat_dataset_rgb_mode_returns_3channel(tmp_path):
    _make_tile(tmp_path, "AnnualCrop", "t1.jpg")
    ds = EuroSATDataset(tmp_path, tmp_path / "cache", ["AnnualCrop/t1.jpg"], input_mode="rgb")
    tensor, label = ds[0]
    assert tensor.shape == (3, 8, 8)
    assert label == 0
    assert isinstance(tensor, torch.Tensor)


def test_euro_sat_dataset_hsi_edge_mode_returns_4channel(tmp_path):
    _make_tile(tmp_path, "AnnualCrop", "t1.jpg")
    ds = EuroSATDataset(tmp_path, tmp_path / "cache", ["AnnualCrop/t1.jpg"], input_mode="hsi_edge")
    tensor, label = ds[0]
    assert tensor.shape == (4, 8, 8)


def test_euro_sat_dataset_defaults_to_hsi_edge(tmp_path):
    _make_tile(tmp_path, "AnnualCrop", "t1.jpg")
    ds = EuroSATDataset(tmp_path, tmp_path / "cache", ["AnnualCrop/t1.jpg"])
    tensor, _ = ds[0]
    assert tensor.shape == (4, 8, 8)
