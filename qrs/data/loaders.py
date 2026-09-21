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
from qrs.data.preprocessing import (
    preprocess_tile_cached,
    preprocess_tile_rgb,
    preprocess_tile_rich2_cached,
)

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


class EuroSATDataset(Dataset):
    def __init__(
        self,
        extracted_dir: str | Path,
        cache_dir: str | Path,
        tile_paths: list[str],
        input_mode: str = "hsi_edge",
    ) -> None:
        self.extracted_dir = Path(extracted_dir)
        self.preprocess_cache_dir = Path(cache_dir) / "preprocessed"
        self.rich2_cache_dir = Path(cache_dir) / "rich2"
        self.tile_paths = tile_paths
        self.input_mode = input_mode

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rel_path = self.tile_paths[idx]
        label = CLASS_TO_IDX[rel_path.split("/")[0]]

        img = Image.open(self.extracted_dir / rel_path).convert("RGB")
        rgb = np.asarray(img, dtype=np.float64) / 255.0

        if self.input_mode == "rgb":
            tensor = preprocess_tile_rgb(rgb)  # cheap normalisation, no disk cache needed
        elif self.input_mode == "rich2":
            tensor = preprocess_tile_rich2_cached(rgb, self.rich2_cache_dir)
        else:
            tensor = preprocess_tile_cached(rgb, self.preprocess_cache_dir)
        return torch.from_numpy(tensor), label


def build_dataloaders(config: Config, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    extracted_dir = download_eurosat(config.data_dir)
    split = load_or_create_split(extracted_dir, config.cache_dir, config.split, seed)

    # The frozen pretrained backbone needs ImageNet-normalised RGB; the
    # "rich_features" backbone needs the precomputed rich2 vector; every
    # other backbone_variant uses the project's HSI+Edge tensor.
    if config.backbone_variant == "frozen_resnet18":
        input_mode = "rgb"
    elif config.backbone_variant == "rich_features":
        input_mode = "rich2"
    else:
        input_mode = "hsi_edge"

    train_ds = EuroSATDataset(extracted_dir, config.cache_dir, split.train, input_mode)
    val_ds = EuroSATDataset(extracted_dir, config.cache_dir, split.val, input_mode)
    test_ds = EuroSATDataset(extracted_dir, config.cache_dir, split.test, input_mode)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
