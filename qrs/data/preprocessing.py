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

import cv2
import numpy as np

_EPS = 1e-8

# Standard ImageNet normalisation, required by torchvision's pretrained
# ResNet-18 weights (qrs.models.frozen_backbone). Only used when a config's
# backbone_variant is "frozen_resnet18" -- every other arm uses the HSI+Edge
# tensor below instead.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_tile_rgb(rgb: np.ndarray) -> np.ndarray:
    """RGB tile -> ImageNet-normalised (3, H, W) tensor.

    Args:
        rgb: array of shape (H, W, 3), float in [0, 1].

    Returns:
        array of shape (3, H, W), dtype float32.
    """
    normed = (rgb.astype(np.float32) - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(normed, (2, 0, 1)).astype(np.float32)


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


def pooled_mean_std(tensor: np.ndarray, grid: tuple[int, int] = (4, 4)) -> np.ndarray:
    """Per-cell mean and standard deviation of a (C, H, W) tensor over a fixed
    grid. Zero trainable parameters, fully deterministic.

    Equivalent to `nn.AdaptiveAvgPool2d(grid)` for the mean component (both
    just average non-overlapping cells); std is the population std of each
    cell via a block reshape, so grid must evenly divide (H, W) -- unlike
    AdaptiveAvgPool2d, which also handles uneven divisions, this does not.

    Args:
        tensor: (C, H, W) float array.
        grid: (grid_h, grid_w).

    Returns:
        1-D float32 array of length C * grid_h * grid_w * 2 -- the flattened
        mean block (C * grid_h * grid_w values), followed by the flattened
        std block (same length).
    """
    c, h, w = tensor.shape
    gh, gw = grid
    if h % gh != 0 or w % gw != 0:
        raise ValueError(f"grid {grid} must evenly divide input size {(h, w)}")
    cell_h, cell_w = h // gh, w // gw
    cells = tensor.reshape(c, gh, cell_h, gw, cell_w)
    mean = cells.mean(axis=(2, 4))
    std = cells.std(axis=(2, 4))
    return np.concatenate([mean.flatten(), std.flatten()]).astype(np.float32)


# 4 orientations x 2 frequencies = 8 fixed Gabor kernels, applied to each of
# the 4 HSI+Edge channels. Reasonable defaults for a 64x64 tile: a kernel
# support smaller than the tile, and wavelengths (1/frequency) short enough
# to fit several cycles across it.
_GABOR_ORIENTATIONS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)  # 0, 45, 90, 135 deg
_GABOR_FREQUENCIES = (0.1, 0.25)  # cycles/pixel -> wavelengths of 10px and 4px
_GABOR_KSIZE = 15
_GABOR_SIGMA = 4.0
_GABOR_GAMMA = 0.5


def _gabor_kernel_pair(theta: float, freq: float) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature pair (cosine, sine phase) for one orientation/frequency,
    combined via magnitude to get a phase-invariant response."""
    lambd = 1.0 / freq
    real = cv2.getGaborKernel(
        (_GABOR_KSIZE, _GABOR_KSIZE), _GABOR_SIGMA, theta, lambd, _GABOR_GAMMA, psi=0, ktype=cv2.CV_32F
    )
    imag = cv2.getGaborKernel(
        (_GABOR_KSIZE, _GABOR_KSIZE),
        _GABOR_SIGMA,
        theta,
        lambd,
        _GABOR_GAMMA,
        psi=np.pi / 2,
        ktype=cv2.CV_32F,
    )
    return real, imag


def gabor_features(tensor: np.ndarray, pool_grid: tuple[int, int] = (2, 2)) -> np.ndarray:
    """Fixed Gabor filter bank magnitude response, mean-pooled per channel
    per kernel. Zero trainable parameters, fully deterministic -- kernels are
    hand-set (not learned), via cv2.getGaborKernel/cv2.filter2D (cv2 is
    already a project dependency; scikit-image is not).

    Args:
        tensor: (C, H, W) float array, e.g. the 4-channel HSI+Edge tensor.
        pool_grid: (grid_h, grid_w) for the per-response mean pool.

    Returns:
        1-D float32 array of length C * 8 * grid_h * grid_w (8 = 4
        orientations x 2 frequencies), ordered orientation, frequency,
        channel, pooled cell.
    """
    c, h, w = tensor.shape
    gh, gw = pool_grid
    if h % gh != 0 or w % gw != 0:
        raise ValueError(f"pool_grid {pool_grid} must evenly divide input size {(h, w)}")
    cell_h, cell_w = h // gh, w // gw

    features = []
    for theta in _GABOR_ORIENTATIONS:
        for freq in _GABOR_FREQUENCIES:
            real_k, imag_k = _gabor_kernel_pair(theta, freq)
            for ch in range(c):
                channel = tensor[ch].astype(np.float32)
                real_resp = cv2.filter2D(channel, cv2.CV_32F, real_k)
                imag_resp = cv2.filter2D(channel, cv2.CV_32F, imag_k)
                magnitude = np.sqrt(real_resp**2 + imag_resp**2)
                pooled = magnitude.reshape(gh, cell_h, gw, cell_w).mean(axis=(1, 3))
                features.append(pooled.flatten())
    return np.concatenate(features).astype(np.float32)


def preprocess_tile_rich2(rgb: np.ndarray) -> np.ndarray:
    """RGB tile -> the "rich2" feature vector: fixed mean+std pooling of the
    HSI+Edge tensor, concatenated with the fixed Gabor filter bank response.
    Zero trainable parameters, fully deterministic; see pooled_mean_std and
    gabor_features. Layout: [mean+std block][Gabor block].

    Args:
        rgb: array of shape (H, W, 3), float in [0, 1].

    Returns:
        1-D float32 array, length pooled_mean_std's + gabor_features' (see
        those functions' docstrings for the exact counts at default grids).
    """
    tensor = preprocess_tile(rgb)  # (4, H, W)
    mean_std = pooled_mean_std(tensor, grid=(4, 4))
    gabor = gabor_features(tensor, pool_grid=(2, 2))
    return np.concatenate([mean_std, gabor]).astype(np.float32)


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


def preprocess_tile_rich2_cached(rgb: np.ndarray, cache_dir: str | Path) -> np.ndarray:
    """Same as preprocess_tile_rich2 but memoised to disk under cache_dir.
    Separate cache directory from preprocess_tile_cached -- same rgb input,
    different (and much smaller) output array."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_key(rgb)

    if cache_path.exists():
        return np.load(cache_path)

    result = preprocess_tile_rich2(rgb)
    np.save(cache_path, result)
    return result
