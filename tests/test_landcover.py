import numpy as np
import pytest
from PIL import Image

from qrs.data.landcover import (
    CLASSES,
    load_official_split,
    prepare_landcover,
    tile_orthophotos,
    tile_paths,
)


def _make_orthophoto(path, width, height, mode, fill):
    if mode == "RGB":
        arr = np.full((height, width, 3), fill, dtype=np.uint8)
    else:
        arr = np.full((height, width), fill, dtype=np.uint8)
    Image.fromarray(arr).save(path)  # mode inferred from array shape/dtype


def test_tiling_grid_and_k_gaps_match_hand_computation(tmp_path):
    # 10x9 (WxH) orthophoto, target_size=4: grid is x in {0,4,8}, y in {0,4,8}
    # (9 iterations, k=0..8), but only tiles fully within bounds get written.
    # Valid (y,x) pairs in nested-loop order: (0,0)->k0, (0,4)->k1,
    # (0,8)->k2 INVALID (x+4=12>10), (4,0)->k3, (4,4)->k4, (4,8)->k5 INVALID,
    # (8,0)->k6 INVALID (y+4=12>9), (8,4)->k7 INVALID, (8,8)->k8 INVALID.
    # Expected written k's: {0, 1, 3, 4} -- exactly the gap pattern the real
    # train.txt shows (some k's present, some missing, non-contiguous).
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    _make_orthophoto(images_dir / "tile-a.tif", width=10, height=9, mode="RGB", fill=100)
    _make_orthophoto(masks_dir / "tile-a.tif", width=10, height=9, mode="L", fill=2)

    output_dir = tile_orthophotos(images_dir, masks_dir, tmp_path / "output", target_size=4)

    written_img_ks = sorted(
        int(p.stem.rsplit("_", 1)[1]) for p in output_dir.glob("tile-a_*.jpg")
    )
    written_mask_ks = sorted(
        int(p.stem.removesuffix("_m").rsplit("_", 1)[1])
        for p in output_dir.glob("tile-a_*_m.png")
    )
    assert written_img_ks == [0, 1, 3, 4]
    assert written_mask_ks == [0, 1, 3, 4]


def test_tile_pixel_values_preserved(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    _make_orthophoto(images_dir / "t.tif", width=8, height=8, mode="RGB", fill=200)
    _make_orthophoto(masks_dir / "t.tif", width=8, height=8, mode="L", fill=3)  # class "water"

    output_dir = tile_orthophotos(images_dir, masks_dir, tmp_path / "output", target_size=8)

    img_tile = np.asarray(Image.open(output_dir / "t_0.jpg"))
    mask_tile = np.asarray(Image.open(output_dir / "t_0_m.png"))

    assert img_tile.shape == (8, 8, 3)
    # JPEG is lossy (matches the official pipeline) -- allow a small tolerance.
    assert abs(int(img_tile.mean()) - 200) <= 5
    assert mask_tile.shape == (8, 8)
    assert (mask_tile == 3).all()  # PNG is lossless -- exact class index preserved


def test_tiling_is_idempotent(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    _make_orthophoto(images_dir / "t.tif", width=8, height=8, mode="RGB", fill=1)
    _make_orthophoto(masks_dir / "t.tif", width=8, height=8, mode="L", fill=1)

    output_dir = tmp_path / "output"
    tile_orthophotos(images_dir, masks_dir, output_dir, target_size=8)
    (output_dir / "sentinel.txt").write_text("should survive a second call")

    tile_orthophotos(images_dir, masks_dir, output_dir, target_size=8)
    assert (output_dir / "sentinel.txt").exists()


def test_mismatched_image_mask_pair_raises(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    _make_orthophoto(images_dir / "t.tif", width=8, height=8, mode="RGB", fill=1)
    _make_orthophoto(masks_dir / "t.tif", width=4, height=4, mode="L", fill=1)  # wrong size

    with pytest.raises(ValueError):
        tile_orthophotos(images_dir, masks_dir, tmp_path / "output", target_size=8)


def test_load_official_split_parses_tile_ids(tmp_path):
    (tmp_path / "train.txt").write_text("M-33-20-D-c-4-2_0\nM-33-20-D-c-4-2_1\n")
    (tmp_path / "val.txt").write_text("N-33-104-A-c-1_5\n")
    (tmp_path / "test.txt").write_text("N-33-60-D-c-4_2\n\n")  # trailing blank line

    split = load_official_split(tmp_path)
    assert split.train == ["M-33-20-D-c-4-2_0", "M-33-20-D-c-4-2_1"]
    assert split.val == ["N-33-104-A-c-1_5"]
    assert split.test == ["N-33-60-D-c-4_2"]


def test_tile_paths_naming():
    img_path, mask_path = tile_paths("tiles", "M-33-20-D-c-4-2_0")
    assert str(img_path).replace("\\", "/") == "tiles/M-33-20-D-c-4-2_0.jpg"
    assert str(mask_path).replace("\\", "/") == "tiles/M-33-20-D-c-4-2_0_m.png"


def test_classes_has_five_entries_background_first():
    assert CLASSES[0] == "background"
    assert len(CLASSES) == 5


def test_prepare_landcover_wires_tiling_and_split_together(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "images").mkdir(parents=True)
    (raw_dir / "masks").mkdir(parents=True)
    _make_orthophoto(raw_dir / "images" / "t.tif", width=8, height=8, mode="RGB", fill=1)
    _make_orthophoto(raw_dir / "masks" / "t.tif", width=8, height=8, mode="L", fill=1)
    (raw_dir / "train.txt").write_text("t_0\n")
    (raw_dir / "val.txt").write_text("")
    (raw_dir / "test.txt").write_text("")

    tiles_dir, split = prepare_landcover(raw_dir, tmp_path / "tiles", target_size=8)
    assert (tiles_dir / "t_0.jpg").exists()
    assert (tiles_dir / "t_0_m.png").exists()
    assert split.train == ["t_0"]
