"""Real end-to-end smoke test for qrs.run.run() with task="segmentation":
synthetic raw orthophotos -> tiling -> dataset -> U-Net training -> a real
row in a segmentation results CSV. Not mocked -- this is the same pipeline
qrs.run.main() drives from the CLI, just with tiny synthetic data standing
in for the real ~1.55GB LandCover.ai dataset."""

import csv

import numpy as np
from PIL import Image

from qrs.analysis.results import SEGMENTATION_RESULT_COLUMNS
from qrs.config import Config
from qrs.run import run


def _write_synthetic_raw_dataset(raw_dir, size=64, n_orthophotos=1):
    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    tile_ids = []
    for i in range(n_orthophotos):
        name = f"ortho-{i}"
        rgb = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        mask = np.random.randint(0, 5, (size, size), dtype=np.uint8)
        Image.fromarray(rgb).save(images_dir / f"{name}.tif")
        Image.fromarray(mask).save(masks_dir / f"{name}.tif")
        # With tile_size=32 and a 64x64 orthophoto: grid is 2x2, all 4 tiles
        # fully in-bounds, k = 0..3, all written (see qrs.data.landcover's
        # tiling behaviour verified in test_landcover.py).
        tile_ids.extend(f"{name}_{k}" for k in range(4))

    (raw_dir / "train.txt").write_text("\n".join(tile_ids[:2]))
    (raw_dir / "val.txt").write_text(tile_ids[2])
    (raw_dir / "test.txt").write_text(tile_ids[3])


def test_run_segmentation_end_to_end(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_synthetic_raw_dataset(raw_dir, size=64)

    results_csv = tmp_path / "results_segmentation.csv"
    config = Config(
        arm="quantum",
        task="segmentation",
        data_dir=str(raw_dir),
        tiles_dir=str(tmp_path / "tiles"),
        tile_size=32,
        cache_dir=str(tmp_path / "cache"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_csv=str(results_csv),
        n_qubits=2,
        n_layers=1,
        epochs=1,
        batch_size=2,
        seeds=(0,),
    )

    run(config)

    assert results_csv.exists()
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert list(row.keys()) == SEGMENTATION_RESULT_COLUMNS
    assert row["arm"] == "quantum"
    assert row["n_quantum_params"] == str(2 * 1 * 3)
    assert float(row["miou"]) == float(row["miou"])  # not NaN-as-string-crash, just parses
    assert (tmp_path / "checkpoints" / "quantum_seed0.pt").exists()
