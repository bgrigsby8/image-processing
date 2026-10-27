"""Tests for sharpness.py - slanted-edge MTF50 on checkerboard edges.

Synthetic edges are rendered at 8x, blurred with a Gaussian of known sigma,
then area-averaged down to pixels, so the true MTF is known exactly:
exp(-2 pi^2 sigma^2 f^2) for the blur times sinc(f) for the pixel aperture.
"""

import math

import cv2
import numpy as np
import pytest

from models import sharpness as sh

UP = 8


def _expected_mtf50(sigma_px: float) -> float:
    f = np.linspace(1e-4, 1.0, 200_001)
    mtf = np.exp(-2 * np.pi ** 2 * sigma_px ** 2 * f ** 2) * np.abs(np.sinc(f))
    return float(f[np.argmax(mtf < 0.5)])


def _render(fine: np.ndarray, sigma_px: float, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Blur at 8x, area-average to pixels, scale to linear [0.05, 0.8]."""
    if sigma_px > 0:
        fine = cv2.GaussianBlur(fine, (0, 0), sigma_px * UP)
    h, w = fine.shape[0] // UP, fine.shape[1] // UP
    img = fine[: h * UP, : w * UP].reshape(h, UP, w, UP).mean(axis=(1, 3))
    img = 0.05 + 0.75 * img
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    return img.astype(np.float32)


def _edge_image(angle_deg: float, sigma_px: float, size: int = 120, noise: float = 0.0):
    """A dark/light edge through the image centre, ``angle_deg`` off vertical."""
    n = size * UP
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    cx = cy = n / 2
    t = math.radians(angle_deg)
    # Signed distance from a line through the centre, direction (sin t, cos t).
    d = (xs + 0.5 - cx) * math.cos(t) - (ys + 0.5 - cy) * math.sin(t)
    fine = (d > 0).astype(np.float32)
    img = _render(fine, sigma_px, noise)
    c = size / 2
    half = 40
    p0 = (c - half * math.sin(t), c - half * math.cos(t))
    p1 = (c + half * math.sin(t), c + half * math.cos(t))
    return img, p0, p1


@pytest.mark.parametrize("sigma", [0.7, 1.0, 1.5, 2.5])
def test_edge_mtf50_matches_known_gaussian(sigma):
    img, p0, p1 = _edge_image(5.0, sigma)
    result = sh.edge_mtf(img, p0, p1, across_px=20)
    assert result.error is None, result.error
    assert result.mtf50 == pytest.approx(_expected_mtf50(sigma), rel=0.05)
    assert result.angle_deg == pytest.approx(5.0, abs=0.3)


def test_edge_mtf_polarity_does_not_matter():
    img, p0, p1 = _edge_image(6.0, 1.2)
    a = sh.edge_mtf(img, p0, p1, across_px=20)
    b = sh.edge_mtf(img, p1, p0, across_px=20)        # flipped direction
    c = sh.edge_mtf(0.85 - img, p0, p1, across_px=20)  # inverted contrast
    assert a.mtf50 == pytest.approx(b.mtf50, rel=0.01)
    assert a.mtf50 == pytest.approx(c.mtf50, rel=0.02)


def test_edge_mtf_tolerates_noise():
    img, p0, p1 = _edge_image(5.0, 1.0, noise=0.01)
    result = sh.edge_mtf(img, p0, p1, across_px=20)
    assert result.mtf50 == pytest.approx(_expected_mtf50(1.0), rel=0.1)


def test_edge_mtf_rejects_axis_aligned_edge():
    img, p0, p1 = _edge_image(0.0, 1.0)
    result = sh.edge_mtf(img, p0, p1, across_px=20)
    assert result.mtf50 is None
    assert "off axis" in result.error


def test_edge_mtf_rejects_clipped_and_flat_regions():
    img, p0, p1 = _edge_image(5.0, 1.0)
    clipped = sh.edge_mtf(np.clip(img * 2, 0, 1), p0, p1, across_px=20)
    assert clipped.mtf50 is None and "clipped" in clipped.error
    flat = sh.edge_mtf(np.full_like(img, 0.3), p0, p1, across_px=20)
    assert flat.mtf50 is None and "contrast" in flat.error


def test_edge_mtf_rejects_region_outside_frame():
    img, _, _ = _edge_image(5.0, 1.0)
    result = sh.edge_mtf(img, (5, -30), (8, 40), across_px=20)
    assert "leaves the frame" in result.error


# ---------------------------------------------------------------------------
# Whole board
# ---------------------------------------------------------------------------

PATTERN = (7, 5)          # inner corners; 8x6 squares
SQUARE_PX = 40


def _board_image(sigma_px: float, angle_deg: float = 5.0, noise: float = 0.0):
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    w, h = cols * SQUARE_PX + 160, rows * SQUARE_PX + 160
    n_w, n_h = w * UP, h * UP
    ys, xs = np.mgrid[0:n_h, 0:n_w].astype(np.float32)
    t = math.radians(angle_deg)
    cx, cy = n_w / 2, n_h / 2
    # Board coordinates of each fine pixel (rotate back by the board angle).
    bx = ((xs + 0.5 - cx) * math.cos(t) + (ys + 0.5 - cy) * math.sin(t)) / (SQUARE_PX * UP) + cols / 2
    by = (-(xs + 0.5 - cx) * math.sin(t) + (ys + 0.5 - cy) * math.cos(t)) / (SQUARE_PX * UP) + rows / 2
    inside = (bx >= 0) & (bx < cols) & (by >= 0) & (by < rows)
    squares = ((np.floor(bx) + np.floor(by)) % 2 == 0)
    fine = np.where(inside, np.where(squares, 0.0, 1.0), 1.0).astype(np.float32)
    return _render(fine, sigma_px, noise)


def test_find_and_measure_board():
    img = _board_image(1.0)
    corners = sh.find_board_corners(img, PATTERN)
    assert corners is not None
    edges, spacing = sh.measure_board(img, corners, PATTERN)
    assert spacing == pytest.approx(SQUARE_PX, rel=0.02)
    cols, rows = PATTERN
    assert len(edges) == rows * (cols - 1) + cols * (rows - 1)
    result = sh.FrameSharpness(path="synthetic", image_size=(img.shape[1], img.shape[0]), edges=edges)
    sh.summarize_edges(result)
    assert result.error is None
    assert result.edges_used >= 0.9 * result.edges_total
    assert result.mtf50 == pytest.approx(_expected_mtf50(1.0), rel=0.05)
    assert result.mtf50_vertical is not None and result.mtf50_horizontal is not None
    assert result.mtf50_lw_ph == pytest.approx(2 * result.mtf50 * img.shape[0])


def test_board_ranks_focus():
    corners = sh.find_board_corners(_board_image(0.8), PATTERN)
    medians = []
    for sigma in (0.8, 1.5, 3.0):
        edges, _ = sh.measure_board(_board_image(sigma, noise=0.005), corners, PATTERN)
        medians.append(np.median([e.mtf50 for e in edges if e.mtf50 is not None]))
    assert medians[0] > medians[1] > medians[2]


def test_summarize_reports_too_few_edges():
    result = sh.FrameSharpness(path="x", edges=[sh.EdgeMTF("vertical", (0, 0), 5.0, error="nope")])
    sh.summarize_edges(result)
    assert result.mtf50 is None
    assert "0 of 1" in result.error and "nope" in result.error


# ---------------------------------------------------------------------------
# Focus curve
# ---------------------------------------------------------------------------

def test_fit_focus_peak_finds_vertex_between_samples():
    pos = np.arange(56000, 64001, 1000)
    val = 0.3 - 1e-9 * (pos - 59700.0) ** 2
    peak = sh.fit_focus_peak(pos, val)
    assert peak.fit == "parabola"
    assert peak.best_position == pytest.approx(59700, abs=1)
    assert not peak.at_edge_of_sweep
    lo, hi = peak.usable_range
    assert lo < 59700 < hi


def test_fit_focus_peak_skips_missing_and_flags_edge():
    peak = sh.fit_focus_peak([1, 2, 3, 4, 5], [None, 0.1, 0.2, 0.3, 0.4])
    assert peak.at_edge_of_sweep
    assert peak.best_position == 5
    assert sh.fit_focus_peak([1, 2], [0.1, 0.2]) is None


def test_fit_focus_peak_usable_range_interpolates():
    peak = sh.fit_focus_peak([0, 10, 20, 30, 40], [0.0, 0.5, 1.0, 0.5, 0.0], usable_fraction=0.75)
    # Peak from the parabola is slightly above 1.0; the 0.75 crossing sits
    # just inside the midpoints between samples.
    lo, hi = peak.usable_range
    assert 10 < lo < 16 and 24 < hi < 30
    assert peak.best_position == pytest.approx(20)


# ---------------------------------------------------------------------------
# Tenengrad
# ---------------------------------------------------------------------------

def test_tenengrad_ranks_blur_and_ignores_exposure():
    sharp = _board_image(0.8)
    soft = _board_image(2.5)
    assert sh.tenengrad(sharp) > sh.tenengrad(soft)
    assert sh.tenengrad(sharp * 0.5) == pytest.approx(sh.tenengrad(sharp), rel=1e-4)
