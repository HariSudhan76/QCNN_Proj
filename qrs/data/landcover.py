"""LandCover.ai raw-orthophoto tiling and official split loading.

The Kaggle mirror (adrianboguszewski/landcoverai) ships the *raw* distribution:
41 full-size orthophotos (images/*.tif + masks/*.tif), the authors' own
split.py, and train.txt/val.txt/test.txt listing official tile membership as
"{orthophoto_name}_{k}" (no extension) -- verified directly against the
dataset's actual split.py source and a train.txt sample, not assumed.

tile_orthophotos() below is a faithful reimplementation of that split.py:
same TARGET_SIZE grid, same non-overlapping stride, and critically the same
"k increments every grid position, but only full-size tiles get written"
behaviour -- this produces the exact gaps (..., _1, _10, _100, _102, ...)
seen in the official train.txt, which is why tile numbering must match
bit-for-bit rather than just "tile similarly". Reimplemented with Pillow
(already a dependency) instead of the original's OpenCV to avoid adding
opencv-python -- see the docstring notes below on the one behavioural
difference this requires accounting for (mask channel handling).

Known risk, not verified here (no local access to the ~1.55GB dataset):
PIL's built-in libtiff binding usually opens standard GeoTIFFs fine, but some
compression/tiling schemes need `tifffile` or `rasterio` instead. If
Image.open() fails on the real orthophotos, that's the first thing to try.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

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

    The original script reads masks via `cv2.imread(path)` with no flag,
    which forces OpenCV to decode even a single-channel mask as 3-channel
    BGR (replicating the class-index value across all 3 channels) -- an
    incidental side effect of their default flag, not meaningful data. We
    read masks as single-channel directly via Pillow and write clean
    single-channel PNGs: identical class-index values, smaller files, and it
    doesn't affect matching against train.txt/val.txt/test.txt, which only
    depends on the "{name}_{k}" naming/indexing established below.
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
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if img_name != mask_name or img.size != mask.size:
            raise ValueError(
                f"image/mask mismatch: {img_path.name} ({img.size}) vs "
                f"{mask_path.name} ({mask.size})"
            )

        width, height = img.size  # PIL size is (W, H)
        k = 0
        for y in range(0, height, target_size):
            for x in range(0, width, target_size):
                # PIL's crop() always returns a box of the requested size,
                # padding if it runs past the edge -- unlike the original's
                # numpy slicing, which just shrinks. Check bounds explicitly
                # instead of the cropped result's size, so "only write a
                # full tile" means the same thing here as it did there.
                if x + target_size <= width and y + target_size <= height:
                    box = (x, y, x + target_size, y + target_size)
                    img.crop(box).save(output_dir / f"{img_name}_{k}.jpg")
                    mask.crop(box).save(output_dir / f"{mask_name}_{k}_m.png")
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
    return tiles_dir, split
