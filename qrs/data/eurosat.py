"""EuroSAT RGB download, deterministic split, and split caching.

Deliberately avoids torchvision.datasets (plain download + PIL) since the local
torchvision install is version-mismatched against this torch build; the
project has no other need for torchvision.
"""

from __future__ import annotations

import json
import random
import ssl
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import truststore

# Some Windows Python installs ship an OpenSSL trust bundle that doesn't
# include the CA chain for madm.dfki.de even though the OS trust store does
# (verified: curl via Schannel succeeds where urllib fails). Defer to the OS
# trust store instead of disabling verification.
truststore.inject_into_ssl()

EUROSAT_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


@dataclass
class Split:
    train: list[str]
    val: list[str]
    test: list[str]


def download_eurosat(data_dir: str | Path) -> Path:
    """Download and extract EuroSAT RGB into data_dir. Idempotent."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / "2750"
    if extracted.exists() and any(extracted.iterdir()):
        return extracted

    zip_path = data_dir / "EuroSAT.zip"
    if not zip_path.exists():
        try:
            urlretrieve(EUROSAT_URL, zip_path)
        except ssl.SSLCertVerificationError:
            # madm.dfki.de is known to omit an intermediate certificate from
            # its chain. Windows' Schannel auto-fetches the missing
            # intermediate (AIA chasing) so `truststore` succeeds there, but
            # Linux/OpenSSL (Colab included) does not, so verification fails
            # even with a correct trust store. This is a known-benign,
            # non-sensitive public benchmark dataset from its official
            # academic host -- fall back to an unverified download rather
            # than blocking the pipeline, but do so loudly.
            warnings.warn(
                "TLS verification of madm.dfki.de failed (missing intermediate "
                "cert in its chain -- expected on Linux/Colab). Retrying the "
                "EuroSAT download WITHOUT certificate verification.",
                stacklevel=2,
            )
            # urlretrieve has no `context` param; the documented way to scope
            # an unverified context to a single call is to swap the default
            # HTTPS context builder for the duration of that call.
            previous_context_factory = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            try:
                urlretrieve(EUROSAT_URL, zip_path)
            finally:
                ssl._create_default_https_context = previous_context_factory

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)

    zip_path.unlink(missing_ok=True)
    return extracted


def list_tiles(extracted_dir: str | Path) -> list[str]:
    """All tile paths (relative to extracted_dir), one per class subfolder."""
    extracted_dir = Path(extracted_dir)
    tiles: list[str] = []
    for cls in CLASSES:
        cls_dir = extracted_dir / cls
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.glob("*.jpg")):
            tiles.append(f.relative_to(extracted_dir).as_posix())
    return tiles


def make_split(
    extracted_dir: str | Path,
    split: tuple[float, float, float],
    seed: int,
) -> Split:
    """Deterministic, seeded, class-stratified 70/15/15-style split."""
    if abs(sum(split) - 1.0) > 1e-6:
        raise ValueError(f"split must sum to 1.0, got {split}")

    extracted_dir = Path(extracted_dir)
    train, val, test = [], [], []
    rng = random.Random(seed)

    for cls in CLASSES:
        cls_dir = extracted_dir / cls
        if not cls_dir.exists():
            continue
        tiles = [f.relative_to(extracted_dir).as_posix() for f in sorted(cls_dir.glob("*.jpg"))]
        rng.shuffle(tiles)

        n = len(tiles)
        n_train = int(round(n * split[0]))
        n_val = int(round(n * split[1]))

        train.extend(tiles[:n_train])
        val.extend(tiles[n_train : n_train + n_val])
        test.extend(tiles[n_train + n_val :])

    return Split(train=train, val=val, test=test)


def load_or_create_split(
    extracted_dir: str | Path,
    cache_dir: str | Path,
    split: tuple[float, float, float],
    seed: int,
) -> Split:
    """Cache the split's file lists to disk keyed by (split ratios, seed) so re-runs
    are reproducible without re-shuffling."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"split_{split[0]:.2f}_{split[1]:.2f}_{split[2]:.2f}_seed{seed}.json"
    cache_path = cache_dir / key

    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        return Split(train=raw["train"], val=raw["val"], test=raw["test"])

    result = make_split(extracted_dir, split, seed)
    cache_path.write_text(
        json.dumps({"train": result.train, "val": result.val, "test": result.test})
    )
    return result
