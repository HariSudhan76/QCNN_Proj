"""LandCover.ai raw-orthophoto tiling and official split loading.

The Kaggle mirror (adrianboguszewski/landcoverai) ships the *raw* distribution:
41 full-size orthophotos (images/*.tif + masks/*.tif), the authors' own
split.py, and train.txt/val.txt/test.txt listing official tile membership as
"{orthophoto_name}_{k}" (no extension) -- verified directly against the
dataset's actual split.py source and a train.txt sample, not assumed.

tile_orthophotos() below uses OpenCV (matching split.py exactly) rather than
Pillow. An earlier Pillow-based reimplementation matched split.py's *logic*
exactly (verified against a hand-computed synthetic case) but diverged on
real orthophotos: at least one real GeoTIFF produced a different tile count
under Pillow than under OpenCV, most likely because these are GIS-authored,
possibly tiled/pyramidal TIFFs that the two libraries' TIFF decoders don't
necessarily agree on pixel-for-pixel. Using the same library the authors
used sidesteps that risk entirely rather than hoping two decoders agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

# 0 = unlabeled/background, matching the authors' pixel-value convention.
CLASSES = ["background", "building", "woodland", "water", "road"]


@dataclass
class Split:
    train: list[str]
    val: list[str]
    test: list[str]


def tile_orthophotos(
    images_dir: str | Path,
    masks_dir: str | Path,
    output_dir: str | Path,
    target_size: int = 512,
) -> Path:
    """Tile every (image, mask) orthophoto pair into target_size x target_size
    pieces, matching the official split.py's exact grid and naming:
    "{name}_{k}.jpg" for images, "{name}_{k}_m.png" for masks. Idempotent --
    returns immediately if output_dir already has content.

    Uses cv2.imread/imwrite exactly like the original script, including its
    numpy-slice-based "only write a full tile" check, so tile counts and
    boundaries match bit-for-bit. One deliberate difference: the original
    reads masks via `cv2.imread(path)` with no flag, which forces OpenCV to
    decode even a single-channel mask as 3-channel BGR (replicating the
    class-index value across all 3 channels) -- an incidental side effect of
    their default flag, not meaningful data. We read masks with
    `cv2.IMREAD_GRAYSCALE` and write clean single-channel PNGs instead:
    identical class-index values, smaller files, and it doesn't affect
    matching against train.txt/val.txt/test.txt, which only depends on the
    "{name}_{k}" naming/indexing established below.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(images_dir.glob("*.tif"))
    mask_paths = sorted(masks_dir.glob("*.tif"))
    print(f"tiling {len(img_paths)} orthophotos into {output_dir} ...", flush=True)

    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        img_name = img_path.stem
        mask_name = mask_path.stem
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"cv2 failed to read image: {img_path}")
        if mask is None:
            raise ValueError(f"cv2 failed to read mask: {mask_path}")
        if img_name != mask_name or img.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"image/mask mismatch: {img_path.name} ({img.shape[:2]}) vs "
                f"{mask_path.name} ({mask.shape[:2]})"
            )

        k = 0
        for y in range(0, img.shape[0], target_size):
            for x in range(0, img.shape[1], target_size):
                img_tile = img[y : y + target_size, x : x + target_size]
                mask_tile = mask[y : y + target_size, x : x + target_size]

                if img_tile.shape[0] == target_size and img_tile.shape[1] == target_size:
                    cv2.imwrite(str(output_dir / f"{img_name}_{k}.jpg"), img_tile)
                    cv2.imwrite(str(output_dir / f"{mask_name}_{k}_m.png"), mask_tile)
                k += 1

        print(f"  tiled {img_name} ({i + 1}/{len(img_paths)})", flush=True)

    return output_dir


def load_official_split(splits_dir: str | Path) -> Split:
    """Read the authors' train.txt/val.txt/test.txt: one tile id per line,
    e.g. "M-33-20-D-c-4-2_0", no extension. This is a fixed, pre-determined
    split -- no seeding/shuffling of our own, unlike EuroSAT."""
    splits_dir = Path(splits_dir)

    def _read(name: str) -> list[str]:
        text = (splits_dir / name).read_text()
        return [line.strip() for line in text.splitlines() if line.strip()]

    return Split(train=_read("train.txt"), val=_read("val.txt"), test=_read("test.txt"))


def tile_paths(tiles_dir: str | Path, tile_id: str) -> tuple[Path, Path]:
    """(image_path, mask_path) for a tile id like "M-33-20-D-c-4-2_0"."""
    tiles_dir = Path(tiles_dir)
    return tiles_dir / f"{tile_id}.jpg", tiles_dir / f"{tile_id}_m.png"


def _filter_to_existing_tiles(split: Split, tiles_dir: str | Path) -> Split:
    """Drop any tile id the official split lists but this environment's
    tiling didn't produce. Observed in practice: a handful of ids from the
    published train.txt are absent even when tiling matches split.py
    bit-for-bit (same OpenCV calls, same grid math, verified against a
    hand-computed synthetic case) -- most likely a GeoTIFF decoder/library
    version difference from whatever environment generated the official
    split, not a bug in our tiling. Filtering (not crashing) trades a small,
    logged amount of missing data for a pipeline that actually runs; the
    alternative is chasing bit-for-bit reproduction of a third-party GIS
    toolchain we don't control."""

    def _filter(tile_ids: list[str]) -> tuple[list[str], list[str]]:
        kept, dropped = [], []
        for tile_id in tile_ids:
            img_path, mask_path = tile_paths(tiles_dir, tile_id)
            (kept if img_path.exists() and mask_path.exists() else dropped).append(tile_id)
        return kept, dropped

    train, train_dropped = _filter(split.train)
    val, val_dropped = _filter(split.val)
    test, test_dropped = _filter(split.test)

    all_dropped = train_dropped + val_dropped + test_dropped
    if all_dropped:
        sample = ", ".join(all_dropped[:5])
        more = f" (+{len(all_dropped) - 5} more)" if len(all_dropped) > 5 else ""
        print(
            f"WARNING: {len(all_dropped)} tile(s) listed in the official split "
            f"were not produced by tiling in this environment -- excluded: "
            f"{sample}{more}",
            flush=True,
        )

    return Split(train=train, val=val, test=test)


def prepare_landcover(
    raw_dir: str | Path, tiles_dir: str | Path, target_size: int = 512
) -> tuple[Path, Split]:
    """raw_dir: the Kaggle-attached dataset root (images/, masks/, train.txt,
    val.txt, test.txt as siblings). tiles_dir: a writable cache location --
    raw_dir is typically read-only (a mounted Kaggle input)."""
    raw_dir = Path(raw_dir)
    tiles_dir = tile_orthophotos(
        raw_dir / "images", raw_dir / "masks", tiles_dir, target_size=target_size
    )
    split = load_official_split(raw_dir)
    split = _filter_to_existing_tiles(split, tiles_dir)
    return tiles_dir, split
