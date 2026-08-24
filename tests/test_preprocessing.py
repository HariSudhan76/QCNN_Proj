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
