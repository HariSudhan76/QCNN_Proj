"""RGB -> HSI + edge-channel preprocessing, with disk caching.

Formulas implemented explicitly per the project spec (no colorsys/skimage shortcut):
    I = (R+G+B)/3
    S = 1 - 3*min(R,G,B)/(R+G+B)                          (guarded)
    theta = arccos( 0.5*((R-G)+(R-B)) / sqrt((R-G)^2 + (R-B)(G-B)) )   (guarded)
    H = theta            if B <= G
        2*pi - theta     otherwise
    H normalised to [0, 1]
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_EPS = 1e-8


def rgb_to_hsi(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to HSI.

    Args:
        rgb: array of shape (..., H, W, 3), float, values in [0, 1].

    Returns:
        array of shape (..., H, W, 3) with channels (H, S, I), H and S in
        [0, 1], I in [0, 1].
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    intensity = (r + g + b) / 3.0

    channel_sum = r + g + b
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = 1.0 - 3.0 * min_rgb / np.where(channel_sum < _EPS, _EPS, channel_sum)
    saturation = np.clip(saturation, 0.0, 1.0)

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    denominator = np.where(denominator < _EPS, _EPS, denominator)
    cos_arg = np.clip(numerator / denominator, -1.0, 1.0)
    theta = np.arccos(cos_arg)

    hue = np.where(b <= g, theta, 2.0 * np.pi - theta)
    hue = hue / (2.0 * np.pi)

    return np.stack([hue, saturation, intensity], axis=-1)


def compute_edge_channel(hsi: np.ndarray) -> np.ndarray:
    """4-neighbour Euclidean distance in (H, S, I) space, vectorised (no Python pixel loops).

    Args:
        hsi: array of shape (H, W, 3).

    Returns:
        array of shape (H, W) with the averaged 4-neighbour edge response.
        Border pixels average over however many in-bounds neighbours they have.
    """
    hsi = np.asarray(hsi, dtype=np.float64)
    h, w, _ = hsi.shape

    total = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)

    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy, dx in shifts:
        shifted = np.full_like(hsi, np.nan)
        src_y0, src_y1 = max(0, -dy), h - max(0, dy)
        src_x0, src_x1 = max(0, -dx), w - max(0, dx)
        dst_y0, dst_y1 = max(0, dy), h - max(0, -dy)
        dst_x0, dst_x1 = max(0, dx), w - max(0, -dx)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = hsi[src_y0:src_y1, src_x0:src_x1]

        valid = ~np.isnan(shifted).any(axis=-1)
        diff = hsi - np.nan_to_num(shifted)
        dist = np.sqrt(np.sum(diff**2, axis=-1))

        total += np.where(valid, dist, 0.0)
        count += valid.astype(np.float64)

    count = np.where(count < _EPS, 1.0, count)
    return total / count


def preprocess_tile(rgb: np.ndarray) -> np.ndarray:
    """RGB tile -> 4-channel (H, S, I, Edge) tensor, channel-first.

    Args:
        rgb: array of shape (H, W, 3), float in [0, 1].

    Returns:
        array of shape (4, H, W), dtype float32.
    """
    hsi = rgb_to_hsi(rgb)
    edge = compute_edge_channel(hsi)
    stacked = np.concatenate([hsi, edge[..., None]], axis=-1)  # (H, W, 4)
    return np.transpose(stacked, (2, 0, 1)).astype(np.float32)  # (4, H, W)


def _cache_key(rgb: np.ndarray) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    return f"{digest}.npy"


def preprocess_tile_cached(rgb: np.ndarray, cache_dir: str | Path) -> np.ndarray:
    """Same as preprocess_tile but memoised to disk under cache_dir."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_key(rgb)

    if cache_path.exists():
        return np.load(cache_path)

    result = preprocess_tile(rgb)
    np.save(cache_path, result)
    return result
