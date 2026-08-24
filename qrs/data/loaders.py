"""DataLoader construction. Identical code path for every arm -- only the
model built from the config differs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from qrs.config import Config
from qrs.data.eurosat import CLASSES, download_eurosat, load_or_create_split
from qrs.data.preprocessing import preprocess_tile_cached

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


class EuroSATDataset(Dataset):
    def __init__(self, extracted_dir: str | Path, cache_dir: str | Path, tile_paths: list[str]) -> None:
        self.extracted_dir = Path(extracted_dir)
        self.preprocess_cache_dir = Path(cache_dir) / "preprocessed"
        self.tile_paths = tile_paths

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rel_path = self.tile_paths[idx]
        label = CLASS_TO_IDX[rel_path.split("/")[0]]

        img = Image.open(self.extracted_dir / rel_path).convert("RGB")
        rgb = np.asarray(img, dtype=np.float64) / 255.0

        tensor = preprocess_tile_cached(rgb, self.preprocess_cache_dir)
        return torch.from_numpy(tensor), label


def build_dataloaders(config: Config, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    extracted_dir = download_eurosat(config.data_dir)
    split = load_or_create_split(extracted_dir, config.cache_dir, config.split, seed)

    train_ds = EuroSATDataset(extracted_dir, config.cache_dir, split.train)
    val_ds = EuroSATDataset(extracted_dir, config.cache_dir, split.val)
    test_ds = EuroSATDataset(extracted_dir, config.cache_dir, split.test)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
