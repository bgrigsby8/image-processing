"""Slanted-edge MTF50 on a ColorChecker Classic's patch edges (sharpness.py's
ColorChecker mode), in the focus sweep and in calibrate_color."""

import asyncio
import math

import cv2
import numpy as np
import pytest

from models import sharpness as sh
from models.calibration import REFERENCE_SRGB
from models.focus_sweep import coerce_pattern, options_from_command, parse_pattern
from models.image_io import linear_to_srgb, srgb_to_linear

from test_color_correction import _FakeSource, _component
from test_sharpness import _expected_mtf50

UP = 4
PITCH = 150          # px between patch centres, about the rig's at 2.6 m / 35 mm
GAP = 20             # black border between patches, px (~13 % of the pitch)
ROWS, COLS = 4, 6


def _chart(sigma_px: float, angle_deg: float = 5.0, *, background: float = 0.6):
    """A ColorChecker on its black card, tilted, blurred by a Gaussian of
    ``sigma_px``: (linear RGB, patch centres in row-major order)."""
    margin = 150
    w, h = COLS * PITCH + 2 * margin, ROWS * PITCH + 2 * margin
    n_w, n_h = w * UP, h * UP
    ys, xs = np.mgrid[0:n_h, 0:n_w].astype(np.float32)
    t = math.radians(angle_deg)
    cx, cy = n_w / 2, n_h / 2
    # Chart coordinates in pitches, origin at the chart's top-left patch corner.
    bx = ((xs + 0.5 - cx) * math.cos(t) + (ys + 0.5 - cy) * math.sin(t)) / (PITCH * UP) + COLS / 2
    by = (-(xs + 0.5 - cx) * math.sin(t) + (ys + 0.5 - cy) * math.cos(t)) / (PITCH * UP) + ROWS / 2
    card = (bx > -0.3) & (bx < COLS + 0.3) & (by > -0.3) & (by < ROWS + 0.3)
    fx, fy = bx - np.floor(bx), by - np.floor(by)
    g = GAP / 2 / PITCH
    in_patch = (bx >= 0) & (bx < COLS) & (by >= 0) & (by < ROWS) & \
        (fx > g) & (fx < 1 - g) & (fy > g) & (fy < 1 - g)
    idx = (np.clip(np.floor(by), 0, ROWS - 1) * COLS + np.clip(np.floor(bx), 0, COLS - 1)).astype(int)
    ref = srgb_to_linear(REFERENCE_SRGB).astype(np.float32)
    black = 0.03
    out = np.empty((h, w, 3), np.float32)
    for ch in range(3):
        fine = np.where(card, black, background).astype(np.float32)
        fine = np.where(in_patch, ref[idx, ch], fine)
        if sigma_px > 0:
            fine = cv2.GaussianBlur(fine, (0, 0), sigma_px * UP)
        out[..., ch] = fine.reshape(h, UP, w, UP).mean(axis=(1, 3))
    centers = []
    for r in range(ROWS):
        for c in range(COLS):
            px, py = (c + 0.5 - COLS / 2) * PITCH, (r + 0.5 - ROWS / 2) * PITCH
            centers.append((w / 2 + px * math.cos(t) - py * math.sin(t),
                            h / 2 + px * math.sin(t) + py * math.cos(t)))
    return out, np.asarray(centers, np.float32)


@pytest.fixture(scope="module")
def sharp_chart():
    return _chart(1.0)


def test_measures_known_blur(sharp_chart):
    linear, centers = sharp_chart
    edges, pitch = sh.measure_colorchecker(np.ascontiguousarray(linear[..., 1]), centers)
    assert pitch == pytest.approx(PITCH, rel=0.01)
    assert len(edges) == 96
    result = sh.FrameSharpness(path="synthetic", image_size=(linear.shape[1], linear.shape[0]),
                               edges=edges)
    sh.summarize_edges(result)
    assert result.edges_used >= 60
    assert result.mtf50 == pytest.approx(_expected_mtf50(1.0), rel=0.08)
    # The dark patches don't stand out from the black border.
    assert any(e.error == "too little edge contrast" for e in edges)


def test_ranks_focus():
    values = []
    for sigma in (0.8, 1.5, 2.5):
        linear, centers = _chart(sigma)
        result, _ = sh.measure_linear(linear, sh.COLORCHECKER, corners=centers)
        values.append(result.mtf50)
    assert values[0] > values[1] > values[2]


def test_detects_the_chart_or_uses_the_hint(sharp_chart):
    linear, centers = sharp_chart
    found = sh.find_colorchecker_centers(linear)
    if found is None:
        pytest.skip("cv2.mcc does not detect this synthetic chart")
    assert np.abs(found - centers).max() < 0.1 * PITCH
    # A frame where detection fails falls back to the hint.
    blank = np.full_like(linear, 0.5)
    result, used = sh.measure_linear(blank, sh.COLORCHECKER, corners_hint=centers)
    assert result.corners_from_hint and used is not None


def test_not_found_reports_the_chart():
    result, used = sh.measure_linear(np.full((300, 400, 3), 0.5, np.float32), sh.COLORCHECKER)
    assert used is None and result.error == sh.CHART_NOT_FOUND


def test_sweep_accepts_colorchecker():
    assert parse_pattern("ColorChecker") == sh.COLORCHECKER
    assert coerce_pattern("colorchecker") == sh.COLORCHECKER
    assert options_from_command({"pattern": "colorchecker"}).pattern == sh.COLORCHECKER
    assert coerce_pattern("11x8") == (11, 8)


def test_calibrate_color_reports_chart_sharpness(sharp_chart, monkeypatch):
    import models.color_correction as cc_mod

    linear, centers = sharp_chart
    monkeypatch.setattr(cc_mod, "load_linear_rgb", lambda path, **kw: linear)
    monkeypatch.setattr(cc_mod, "render_raw_for_detection",
                        lambda path: (linear_to_srgb(linear) * 255).astype(np.uint8))
    cc = _component(_FakeSource(saved_path=None))
    out = asyncio.run(cc.do_command({"calibrate_color": {
        "path": "/photos/chart.ARW", "patch_centers": centers.tolist(),
        "compute_wb": False, "radius": 10,
    }}))["calibrate_color"]
    assert out["sharpness"]["mtf50"] == pytest.approx(_expected_mtf50(1.0), rel=0.1)
    assert out["sharpness"]["edges_used"] >= 60

    skipped = asyncio.run(cc.do_command({"calibrate_color": {
        "path": "/photos/chart.ARW", "patch_centers": centers.tolist(),
        "compute_wb": False, "radius": 10, "measure_sharpness": False,
    }}))["calibrate_color"]
    assert "sharpness" not in skipped
