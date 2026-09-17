"""LandCover.ai Dataset: loads (image, mask) tile pairs produced by
qrs.data.landcover.tile_orthophotos, applying the same HSI+Edge preprocessing
used for EuroSAT (preprocess_tile_cached is dataset-agnostic -- it just takes
an RGB array) so the quantum bottleneck's input encoding stays consistent
project-wide, across both the classification and segmentation experiments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from qrs.data.landcover import tile_paths
from qrs.data.preprocessing import preprocess_tile_cached


class LandCoverDataset(Dataset):
    def __init__(
        self,
        tiles_dir: str | Path,
        tile_ids: list[str],
        preprocess_cache_dir: str | Path,
    ) -> None:
        self.tiles_dir = Path(tiles_dir)
        self.tile_ids = tile_ids
        self.preprocess_cache_dir = Path(preprocess_cache_dir)

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile_id = self.tile_ids[idx]
        img_path, mask_path = tile_paths(self.tiles_dir, tile_id)

        img = Image.open(img_path).convert("RGB")
        rgb = np.asarray(img, dtype=np.float64) / 255.0
        image_tensor = torch.from_numpy(
            preprocess_tile_cached(rgb, self.preprocess_cache_dir)
        )  # (4, H, W) float32

        mask = Image.open(mask_path)  # single-channel, pixel value == class index
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64))  # (H, W)

        return image_tensor, mask_tensor


def build_landcover_dataloaders(
    tiles_dir: str | Path,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    preprocess_cache_dir: str | Path,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = LandCoverDataset(tiles_dir, train_ids, preprocess_cache_dir)
    val_ds = LandCoverDataset(tiles_dir, val_ids, preprocess_cache_dir)
    test_ds = LandCoverDataset(tiles_dir, test_ids, preprocess_cache_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
