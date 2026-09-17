import numpy as np
import torch
from PIL import Image

from qrs.data.landcover import tile_paths
from qrs.data.landcover_loaders import LandCoverDataset, build_landcover_dataloaders


def _write_tile(tiles_dir, tile_id, size=16, class_value=2):
    tiles_dir.mkdir(parents=True, exist_ok=True)
    img_path, mask_path = tile_paths(tiles_dir, tile_id)
    rgb = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(img_path)
    mask = np.full((size, size), class_value, dtype=np.uint8)
    Image.fromarray(mask).save(mask_path)


def test_dataset_returns_4channel_image_and_class_index_mask(tmp_path):
    tiles_dir = tmp_path / "tiles"
    _write_tile(tiles_dir, "t_0", size=16, class_value=3)

    ds = LandCoverDataset(tiles_dir, ["t_0"], tmp_path / "cache")
    assert len(ds) == 1

    image, mask = ds[0]
    assert image.shape == (4, 16, 16)
    assert image.dtype == torch.float32
    assert mask.shape == (16, 16)
    assert mask.dtype == torch.int64
    assert torch.all(mask == 3)


def test_dataset_preprocessing_is_cached_across_instances(tmp_path):
    tiles_dir = tmp_path / "tiles"
    _write_tile(tiles_dir, "t_0", size=16)
    cache_dir = tmp_path / "cache"

    ds1 = LandCoverDataset(tiles_dir, ["t_0"], cache_dir)
    image1, _ = ds1[0]

    ds2 = LandCoverDataset(tiles_dir, ["t_0"], cache_dir)
    image2, _ = ds2[0]

    assert torch.equal(image1, image2)
    assert any(cache_dir.iterdir())  # something got cached to disk


def test_build_landcover_dataloaders_splits_correctly(tmp_path):
    tiles_dir = tmp_path / "tiles"
    for tid in ["a_0", "a_1", "b_0", "c_0"]:
        _write_tile(tiles_dir, tid, size=8)

    train_loader, val_loader, test_loader = build_landcover_dataloaders(
        tiles_dir,
        train_ids=["a_0", "a_1"],
        val_ids=["b_0"],
        test_ids=["c_0"],
        preprocess_cache_dir=tmp_path / "cache",
        batch_size=2,
    )

    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 1

    images, masks = next(iter(train_loader))
    assert images.shape == (2, 4, 8, 8)
    assert masks.shape == (2, 8, 8)
