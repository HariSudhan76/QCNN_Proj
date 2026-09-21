import numpy as np
import pytest

from qrs.data.preprocessing import (
    compute_edge_channel,
    preprocess_tile,
    rgb_to_hsi,
)


def _hsi_of(r, g, b):
    return rgb_to_hsi(np.array([[[r, g, b]]], dtype=np.float64))[0, 0]


@pytest.mark.parametrize(
    "rgb, expected_hsi",
    [
        ((1.0, 0.0, 0.0), (0.0, 1.0, 1 / 3)),  # pure red
        ((0.0, 1.0, 0.0), (1 / 3, 1.0, 1 / 3)),  # pure green
        ((0.0, 0.0, 1.0), (2 / 3, 1.0, 1 / 3)),  # pure blue
        ((1.0, 1.0, 1.0), (0.25, 0.0, 1.0)),  # white (achromatic, H convention = 0.25)
        ((0.5, 0.5, 0.5), (0.25, 0.0, 0.5)),  # mid-gray (achromatic)
    ],
)
def test_known_rgb_to_hsi_values(rgb, expected_hsi):
    h, s, i = _hsi_of(*rgb)
    exp_h, exp_s, exp_i = expected_hsi
    assert h == pytest.approx(exp_h, abs=1e-6)
    assert s == pytest.approx(exp_s, abs=1e-6)
    assert i == pytest.approx(exp_i, abs=1e-6)


def test_black_is_guarded_not_nan():
    h, s, i = _hsi_of(0.0, 0.0, 0.0)
    assert np.isfinite([h, s, i]).all()
    assert i == pytest.approx(0.0, abs=1e-6)


def test_hsi_batch_no_nan_or_inf_on_random_input():
    rng = np.random.default_rng(0)
    rgb = rng.random((16, 8, 8, 3))
    hsi = rgb_to_hsi(rgb)
    assert np.isfinite(hsi).all()
    assert hsi[..., 0].min() >= 0.0 and hsi[..., 0].max() <= 1.0  # H in [0,1]
    assert hsi[..., 1].min() >= 0.0 and hsi[..., 1].max() <= 1.0  # S in [0,1]


def test_edge_channel_zero_on_uniform_image():
    hsi = np.full((10, 10, 3), 0.5, dtype=np.float64)
    edge = compute_edge_channel(hsi)
    assert edge.shape == (10, 10)
    assert np.allclose(edge, 0.0)


def test_edge_channel_nonzero_at_step_boundary():
    hsi = np.zeros((8, 8, 3), dtype=np.float64)
    hsi[:, 4:, :] = 1.0  # vertical step edge between columns 3 and 4

    edge = compute_edge_channel(hsi)

    # columns adjacent to the boundary should have nonzero edge response
    assert np.all(edge[:, 3] > 0)
    assert np.all(edge[:, 4] > 0)
    # columns far from the boundary should be zero
    assert np.allclose(edge[:, 0], 0.0)
    assert np.allclose(edge[:, 7], 0.0)


def test_edge_channel_no_nan_or_inf():
    rng = np.random.default_rng(1)
    hsi = rng.random((12, 12, 3))
    edge = compute_edge_channel(hsi)
    assert np.isfinite(edge).all()


def test_preprocess_tile_shape_and_range():
    rng = np.random.default_rng(2)
    rgb = rng.random((64, 64, 3))
    out = preprocess_tile(rgb)
    assert out.shape == (4, 64, 64)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_preprocess_tile_rgb_shape_and_normalisation():
    from qrs.data.preprocessing import preprocess_tile_rgb

    rgb = np.full((8, 8, 3), 0.5, dtype=np.float64)
    out = preprocess_tile_rgb(rgb)
    assert out.shape == (3, 8, 8)
    assert out.dtype == np.float32
    # (0.5 - mean) / std, per ImageNet channel stats -- not just passed through.
    assert not np.allclose(out[0], 0.5)
    assert np.isfinite(out).all()


def test_pooled_mean_std_dimension_and_values():
    from qrs.data.preprocessing import pooled_mean_std

    tensor = np.zeros((4, 64, 64), dtype=np.float32)
    tensor[:, :, :32] = 1.0  # left half 1, right half 0 -> known per-cell mean/std
    out = pooled_mean_std(tensor, grid=(4, 4))
    assert out.shape == (128,)
    # First two grid columns (fully in the "1" half) have mean 1, std 0.
    mean = out[:64].reshape(4, 4, 4)
    std = out[64:].reshape(4, 4, 4)
    assert np.allclose(mean[:, :, :2], 1.0)
    assert np.allclose(mean[:, :, 2:], 0.0)
    assert np.allclose(std, 0.0)  # every cell is uniform (all-1 or all-0)


def test_pooled_mean_std_rejects_non_dividing_grid():
    from qrs.data.preprocessing import pooled_mean_std

    with pytest.raises(ValueError, match="evenly divide"):
        pooled_mean_std(np.zeros((4, 64, 64)), grid=(3, 3))


def test_gabor_features_dimension_and_finite():
    from qrs.data.preprocessing import gabor_features

    rng = np.random.default_rng(3)
    tensor = rng.random((4, 64, 64)).astype(np.float32)
    out = gabor_features(tensor, pool_grid=(2, 2))
    assert out.shape == (128,)  # 4 orientations x 2 freqs x 4 channels x 4 cells
    assert np.isfinite(out).all()


def test_gabor_features_uniform_on_constant_image():
    from qrs.data.preprocessing import gabor_features

    tensor = np.full((4, 64, 64), 0.5, dtype=np.float32)
    out = gabor_features(tensor, pool_grid=(2, 2))
    # A finite-window Gabor kernel has some nonzero DC response even to a
    # constant image, but that response must be spatially uniform (same in
    # every one of the 4 pooled cells for a given orientation/freq/channel,
    # since there's no edge/texture anywhere for the filter to react to).
    per_filter_channel = out.reshape(4, 2, 4, 4)  # (orient, freq, channel, cell)
    for cell_group in per_filter_channel.reshape(-1, 4):
        assert np.allclose(cell_group, cell_group[0], atol=1e-5)


def test_preprocess_tile_rich2_shape_and_matches_components():
    from qrs.data.preprocessing import gabor_features, pooled_mean_std, preprocess_tile_rich2

    rng = np.random.default_rng(4)
    rgb = rng.random((64, 64, 3))
    out = preprocess_tile_rich2(rgb)
    assert out.shape == (256,)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()

    tensor = preprocess_tile(rgb)
    expected = np.concatenate(
        [pooled_mean_std(tensor, grid=(4, 4)), gabor_features(tensor, pool_grid=(2, 2))]
    )
    assert np.array_equal(out, expected)


def test_preprocess_tile_rich2_cached_matches_uncached(tmp_path):
    from qrs.data.preprocessing import preprocess_tile_rich2, preprocess_tile_rich2_cached

    rng = np.random.default_rng(5)
    rgb = rng.random((64, 64, 3))
    cached_first = preprocess_tile_rich2_cached(rgb, tmp_path)
    cached_second = preprocess_tile_rich2_cached(rgb, tmp_path)  # hits the cache
    assert np.array_equal(cached_first, preprocess_tile_rich2(rgb))
    assert np.array_equal(cached_first, cached_second)
