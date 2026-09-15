"""Tests for distortion.py and calibrate_distortion.py - the lens distortion /
lateral-CA library and the calibration CLI's maths.

Everything runs on small synthetic frames (1200x800) with a known camera model
(heavy barrel like the real 16 mm), no RAW files, no camera. Board and line
images are rendered *directly in the distorted frame*: for each output pixel we
undistort its coordinate, look up the analytic pattern there, and so know the
exact distorted corner / line positions to compare against.
"""

import json
import os

import cv2
import numpy as np
import pytest

from models import calibrate_distortion as cd
from models.distortion import (
    CalibrationError,
    CalibrationMismatch,
    DistortionCalibration,
    LateralCA,
    Undistorter,
    build_maps,
    distort_points,
    estimate_map_memory_bytes,
    fit_lateral_ca,
    lateral_ca_shift,
    undistort_points,
)

W, H = 1200, 800
K = np.array([[900.0, 0.0, 604.0], [0.0, 900.0, 397.0], [0.0, 0.0, 1.0]])
# Heavy barrel (k1 ~ -0.10) with a little tangential, like the real lens.
DIST = np.array([-0.10, 0.02, 0.0005, -0.0003, 0.0])
PATTERN = (10, 7)
SQUARE = 55.0
OBJ = cd.board_object_points(PATTERN, SQUARE)


def _calib(lateral_ca=None, meta=None, dist=DIST, k=K):
    return DistortionCalibration(
        image_size=(W, H), camera_matrix=k, dist_coeffs=dist,
        lateral_ca=lateral_ca, meta=meta or {},
    )


def _poses(n, seed=0, tilt=0.5, spread=(330, 220)):
    """Random board poses whose corners all land inside the frame - spread so
    the coverage grid fills up (the solver needs edge/corner corners)."""
    rng = np.random.default_rng(seed)
    centre = np.array([OBJ[:, 0].mean(), OBJ[:, 1].mean(), 0.0])
    poses = []
    while len(poses) < n:
        rvec = rng.uniform(-tilt, tilt, 3)
        tvec = np.array([rng.uniform(-spread[0], spread[0]), rng.uniform(-spread[1], spread[1]), rng.uniform(750, 1200)])
        rot, _ = cv2.Rodrigues(rvec)
        tvec = tvec - rot @ centre  # keep the board *centre* near the sampled point
        pts, _ = cv2.projectPoints(OBJ, rvec, tvec, K, DIST)
        pts = pts.reshape(-1, 2)
        if pts[:, 0].min() > 15 and pts[:, 0].max() < W - 15 and pts[:, 1].min() > 15 and pts[:, 1].max() < H - 15:
            poses.append((rvec, tvec, pts))
    return poses


def _covering_poses(n_random, seed=42, tilt=0.3):
    """A board centred over each of the 4x3 coverage cells (as far out as the
    frame allows, so edge and corner cells see corners), plus random extras -
    the frame set a careful operator would shoot."""
    rng = np.random.default_rng(seed)
    centre = np.array([OBJ[:, 0].mean(), OBJ[:, 1].mean(), 0.0])
    poses = []
    for gx in np.linspace(-330, 330, 4):
        for gy in np.linspace(-215, 215, 3):
            for _ in range(200):
                rvec = rng.uniform(-tilt, tilt, 3)
                tvec = np.array([gx, gy, rng.uniform(950, 1150)])
                rot, _ = cv2.Rodrigues(rvec)
                tvec = tvec - rot @ centre
                pts, _ = cv2.projectPoints(OBJ, rvec, tvec, K, DIST)
                pts = pts.reshape(-1, 2)
                if pts[:, 0].min() > 15 and pts[:, 0].max() < W - 15 and pts[:, 1].min() > 15 and pts[:, 1].max() < H - 15:
                    poses.append((rvec, tvec, pts))
                    break
    return poses + _poses(n_random, seed=seed + 1, tilt=tilt, spread=(420, 280))


def _undistorted_grid(calib, scale=1):
    """For every pixel of a (scale x) supersampled distorted frame, the ideal
    (undistorted) pixel coordinate it images."""
    w, h = W * scale, H * scale
    xs = (np.arange(w) + 0.5) / scale - 0.5
    ys = (np.arange(h) + 0.5) / scale - 0.5
    gx, gy = np.meshgrid(xs, ys)
    q = np.stack([gx.ravel(), gy.ravel()], axis=1)
    u = undistort_points(calib, q)
    return u[:, 0].reshape(h, w), u[:, 1].reshape(h, w)


def _downsample(img, scale):
    if scale == 1:
        return img.astype(np.float32)
    return cv2.resize(img.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)


def _render_board(calib, rvec, tvec, scale=2, blur=0.6):
    """Anti-aliased checkerboard (float32 [0,1], white surround) as the lens
    images it, rendered straight into the distorted frame."""
    ux, uy = _undistorted_grid(calib, scale)
    rot, _ = cv2.Rodrigues(rvec)
    hmat = K @ np.column_stack([rot[:, 0], rot[:, 1], tvec.reshape(3)])
    hinv = np.linalg.inv(hmat)
    ones = np.ones_like(ux)
    bx = hinv[0, 0] * ux + hinv[0, 1] * uy + hinv[0, 2] * ones
    by = hinv[1, 0] * ux + hinv[1, 1] * uy + hinv[1, 2] * ones
    bw = hinv[2, 0] * ux + hinv[2, 1] * uy + hinv[2, 2] * ones
    bx, by = bx / bw, by / bw
    cols, rows = PATTERN
    ix = np.floor(bx / SQUARE)
    iy = np.floor(by / SQUARE)
    inside = (ix >= -1) & (ix <= cols - 1) & (iy >= -1) & (iy <= rows - 1)
    dark = ((ix + iy) % 2 == 0) & inside
    img = np.where(dark, 0.08, 0.92).astype(np.float32)
    img = _downsample(img, scale)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def _match_to_truth(detected, truth):
    """Detected corner order may be flipped relative to the truth; pair each
    detected corner with its nearest truth corner and require a bijection."""
    d = np.linalg.norm(detected[:, None, :] - truth[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    assert len(set(nearest.tolist())) == len(truth), "detected corners did not map 1:1 onto the truth"
    return d[np.arange(len(detected)), nearest]


# ---------------------------------------------------------------------------
# Synthetic calibration round trip
# ---------------------------------------------------------------------------

def test_calibrate_from_points_recovers_known_camera():
    poses = _poses(25)
    res = cd.calibrate_from_points(OBJ, [p for _, _, p in poses], (W, H), "standard")
    assert res.rms < 1e-3
    assert np.allclose(res.camera_matrix, K, atol=0.05)
    assert np.allclose(res.dist_coeffs, DIST, atol=1e-4)
    assert res.model == "opencv_standard" and res.dist_coeffs.shape == (5,)
    assert len(res.per_image_rms) == 25 and max(res.per_image_rms) < 1e-3


def test_calibrate_from_points_rational_model_has_eight_coefficients():
    poses = _poses(25, seed=3)
    res = cd.calibrate_from_points(OBJ, [p for _, _, p in poses], (W, H), "rational")
    assert res.model == "opencv_rational" and res.dist_coeffs.shape == (8,)
    assert res.rms < 1e-2
    with pytest.raises(ValueError, match="standard' or 'rational"):
        cd.calibrate_from_points(OBJ, [p for _, _, p in poses], (W, H), "fisheye")


def test_holdout_reprojection_is_near_zero_for_a_consistent_frame():
    poses = _poses(6, seed=9)
    calib = _calib()
    rms, rvec, _ = cd.reprojection_rms(OBJ, poses[0][2], calib.camera_matrix, calib.dist_coeffs)
    assert rms < 1e-3
    # And the recovered pose matches the one we projected with.
    assert np.allclose(rvec.reshape(3), poses[0][0], atol=1e-3)
    assert cd.board_tilt_deg(np.zeros(3)) == pytest.approx(0.0)
    assert cd.board_tilt_deg(np.array([0.3, 0.0, 0.0])) == pytest.approx(np.degrees(0.3), abs=1e-6)


def test_rejection_drops_a_corrupted_frame_and_refits():
    poses = _poses(12, seed=5)
    frames = [cd.FrameResult(f"/x/{i:02d}.ARW", f"{i:02d}.ARW", (W, H), corners_g=p) for i, (_, _, p) in enumerate(poses)]
    frames[4].corners_g = frames[4].corners_g + np.random.default_rng(1).normal(0, 4.0, frames[4].corners_g.shape)
    logs = []
    res, used, rejected = cd._fit_with_rejection(OBJ, frames, (W, H), "standard", 1.0, logs.append)
    assert list(rejected) == ["04.ARW"]
    assert len(used) == 11 and res.rms < 1e-3
    assert any("rejecting 1 frame" in line for line in logs)


# ---------------------------------------------------------------------------
# Point helpers and coverage / straight-line maths
# ---------------------------------------------------------------------------

def test_distort_and_undistort_points_round_trip():
    calib = _calib()
    pts = np.array([[20.0, 30.0], [604.0, 397.0], [1150.0, 780.0], [300.0, 700.0]])
    back = undistort_points(calib, distort_points(calib, pts))
    assert np.abs(back - pts).max() < 1e-3
    # Barrel: points move toward the centre when imaged.
    far = np.array([[1150.0, 780.0]])
    assert np.linalg.norm(distort_points(calib, far) - calib.principal_point) < np.linalg.norm(far - calib.principal_point)
    assert undistort_points(calib, np.zeros((0, 2))).shape == (0, 2)


def test_line_bow_sees_distortion_and_undistort_removes_it():
    calib = _calib()
    _, _, pts = _poses(1, seed=11, tilt=0.3)[0]
    before = cd.line_bow(pts, PATTERN)
    after = cd.line_bow(undistort_points(calib, pts), PATTERN)
    assert before > 0.2          # the lens bends the rows
    assert after < 1e-3          # a pinhole keeps them straight
    report = cd.straight_line_report(pts, PATTERN, calib, "frame.ARW")
    assert report["frame"] == "frame.ARW"
    assert report["max_bow_px_after"] < 1e-3
    assert report["center_crop"]["crop"] == "center 50%"
    # A region too small to hold 3 corners of any line yields None, not a crash.
    assert cd.line_bow(pts, PATTERN, (0, 0, 1, 1)) is None


def test_coverage_grid_counts_frames_not_corners():
    pts_a = np.array([[10.0, 10.0], [20.0, 20.0]])            # both in cell (0, 0)
    pts_b = np.array([[10.0, 10.0], [1190.0, 790.0]])         # cells (0, 0) and (2, 3)
    grid = cd.coverage_grid([pts_a, pts_b], (W, H))
    assert len(grid) == 3 and len(grid[0]) == 4
    assert grid[0][0] == 2 and grid[2][3] == 1
    assert sum(sum(r) for r in grid) == 3


def test_invalid_output_fraction_is_zero_for_barrel_and_positive_for_pincushion():
    assert cd.invalid_output_fraction(_calib()) == 0.0
    assert cd.invalid_output_fraction(_calib(dist=np.array([0.10, 0.0, 0.0, 0.0, 0.0]))) > 0.0


def test_pick_holdout_is_deterministic_and_validated():
    names = [f"{i:02d}.ARW" for i in range(12)]
    opts = cd.RunOptions(inputs=[], pattern=PATTERN, square_mm=SQUARE, out="x.json")
    assert cd._pick_holdout(names, opts) == cd._pick_holdout(names, opts)
    assert len(cd._pick_holdout(names, opts)) == 3          # default for >= 10 frames
    assert cd._pick_holdout(names[:8], opts) == []           # too few frames: no holdout
    opts.holdout_files = ["03.ARW", "07.ARW"]
    assert cd._pick_holdout(names, opts) == ["03.ARW", "07.ARW"]
    opts.holdout_files = ["99.ARW"]
    with pytest.raises(ValueError, match="not among the detected frames"):
        cd._pick_holdout(names, opts)
    opts.holdout_files = []
    opts.holdout = 9
    with pytest.raises(ValueError, match="at least 5"):
        cd._pick_holdout(names, opts)


def test_parse_pattern():
    assert cd.parse_pattern("10x7") == (10, 7)
    assert cd.parse_pattern("9X6") == (9, 6)
    with pytest.raises(ValueError, match="10x7"):
        cd.parse_pattern("ten by seven")
    with pytest.raises(ValueError, match="at least 3x3"):
        cd.parse_pattern("2x7")


def test_board_object_points_are_row_major_in_mm():
    obj = cd.board_object_points((4, 3), 10.0)
    assert obj.shape == (12, 3)
    assert np.array_equal(obj[1], [10.0, 0.0, 0.0])
    assert np.array_equal(obj[4], [0.0, 10.0, 0.0])


# ---------------------------------------------------------------------------
# Detection + sub-pixel refinement
# ---------------------------------------------------------------------------

def test_detection_and_subpixel_land_within_0_3px_of_the_distorted_corners():
    calib = _calib()
    rvec, tvec, truth = _poses(1, seed=21, tilt=0.35)[0]
    img = _render_board(calib, rvec, tvec)
    gray8 = cd.to_gray8(img)
    coarse = cd.detect_checkerboard(gray8, PATTERN, detect_scale=0.5)
    assert coarse is not None and coarse.shape == (70, 2)
    refined = cd.refine_corners(img, coarse)
    errors = _match_to_truth(refined, truth)
    assert errors.max() < 0.3, f"worst corner error {errors.max():.3f} px"


def test_detection_reports_none_when_no_board():
    blank = np.full((H, W), 200, np.uint8)
    assert cd.detect_checkerboard(blank, PATTERN, detect_scale=0.5) is None
    with pytest.raises(ValueError, match="detect_scale"):
        cd.detect_checkerboard(blank, PATTERN, detect_scale=0.0)


def test_refine_corners_returns_a_new_array_and_leaves_the_start_untouched():
    # Regression: cornerSubPix refines in place and ascontiguousarray returned
    # the caller's own float32 buffer, so the G, R and B corner sets ended up
    # as one shared array and the CA fit was an exact zero on real frames.
    calib = _calib()
    rvec, tvec, _ = _poses(1, seed=22, tilt=0.3)[0]
    img = _render_board(calib, rvec, tvec)
    coarse = cd.detect_checkerboard(cd.to_gray8(img), PATTERN, detect_scale=0.5)
    assert coarse.dtype == np.float32 and coarse.flags["C_CONTIGUOUS"]
    start = coarse + np.float32(1.5)
    snapshot = start.copy()
    refined = cd.refine_corners(img, start)
    assert refined is not start
    assert not np.shares_memory(refined, start)
    np.testing.assert_array_equal(start, snapshot)
    assert np.abs(refined - start).max() > 0.5   # it did refine


def test_subpix_window_scales_with_the_imaged_square_size():
    cols, rows = PATTERN
    grid = np.stack(np.meshgrid(np.arange(cols) * 108.0, np.arange(rows) * 108.0), -1).reshape(-1, 2)
    assert cd.corner_spacing_px(grid, PATTERN) == pytest.approx(108.0)
    assert cd.subpix_half_window(108.0) == 22
    assert cd.subpix_half_window(10.0) == cd.SUBPIX_MIN_HALF_WINDOW
    assert cd.subpix_half_window(5000.0) == cd.SUBPIX_MAX_HALF_WINDOW


def test_soft_board_refinement_does_not_degrade_the_coarse_corners():
    # A soft edge (wide lens, f/8, 61 MP) wider than an 11x11 window made the
    # fixed-window refinement wander several px; the spacing-scaled window
    # must stay at least as good as the SB detector's coarse result.
    calib = _calib()
    rvec, tvec, truth = _poses(1, seed=23, tilt=0.2)[0]
    img = _render_board(calib, rvec, tvec, blur=3.0)
    coarse = cd.detect_checkerboard(cd.to_gray8(img), PATTERN, detect_scale=0.5)
    half = cd.subpix_half_window(cd.corner_spacing_px(coarse, PATTERN))
    assert half > cd.SUBPIX_MIN_HALF_WINDOW
    coarse_err = _match_to_truth(coarse, truth).max()
    refined_err = _match_to_truth(cd.refine_corners(img, coarse, half), truth).max()
    assert refined_err <= max(coarse_err, 0.3) + 0.05, (coarse_err, refined_err)


def test_stretch_for_detection_gamma_encodes_between_percentiles():
    lin = np.linspace(0.0, 0.25, 10000, dtype=np.float32).reshape(100, 100)
    out = cd.stretch_for_detection(lin)
    assert out.dtype == np.float32 and out.min() == 0.0 and out.max() == 1.0
    # Mid-linear lands well above 0.5 after gamma (the point: linear looks dark).
    assert out[50, 0] > 0.6
    assert cd.to_gray8(out).dtype == np.uint8


# ---------------------------------------------------------------------------
# Undistorter
# ---------------------------------------------------------------------------

LINE_XS = [150.0, 350.0, 550.0, 750.0, 950.0, 1100.0]
LINE_YS = [80.0, 250.0, 420.0, 600.0, 740.0]


def _line_pattern(ux, uy, sigma=1.2):
    """Dark Gaussian-profile lines on white at LINE_XS / LINE_YS, evaluated at
    ideal coordinates (ux, uy)."""
    img = np.ones_like(ux, dtype=np.float32)
    for x in LINE_XS:
        img *= 1.0 - np.exp(-((ux - x) ** 2) / (2 * sigma ** 2))
    for y in LINE_YS:
        img *= 1.0 - np.exp(-((uy - y) ** 2) / (2 * sigma ** 2))
    return img.astype(np.float32)


def _distorted_lines(calib):
    ux, uy = _undistorted_grid(calib)
    return _line_pattern(ux, uy)


def _vertical_line_centroids(plane, nominal_x, rows, half=6):
    """Darkness-weighted x centroid around ``nominal_x`` on each of ``rows``."""
    xs = []
    for y in rows:
        x0 = int(round(nominal_x)) - half
        window = plane[int(y), x0:x0 + 2 * half + 1]
        weights = 1.0 - window
        xs.append(float((weights * np.arange(x0, x0 + 2 * half + 1)).sum() / weights.sum()))
    return np.array(xs)


def test_undistorter_straightens_lines_and_keeps_centre_scale():
    calib = _calib()
    distorted = _distorted_lines(calib)
    rows = np.arange(30, H - 30, 20)
    img = np.stack([distorted] * 3, axis=-1)
    out = Undistorter(calib).apply(img)
    assert out.shape == img.shape and out.dtype == np.float32
    for x in LINE_XS:
        # Rows away from the horizontal lines, so the centroid sees one line.
        clear_rows = [y for y in rows if all(abs(y - ly) > 8 for ly in LINE_YS)]
        centroids = _vertical_line_centroids(out[..., 1], x, clear_rows)
        fit = np.polyval(np.polyfit(clear_rows, centroids, 1), clear_rows)
        assert np.abs(centroids - fit).max() < 0.5, f"line at x={x} still bows"
        # newCameraMatrix = K: the recovered line sits where the ideal one is.
        assert np.abs(centroids - x).max() < 0.5, f"line at x={x} moved"


def test_undistorter_distorted_input_really_was_bent():
    """Guard for the test above: without correction the same lines deviate by
    well over the tolerance, so the straightness assertion is meaningful."""
    calib = _calib()
    distorted = _distorted_lines(calib)
    rows = np.array([y for y in np.arange(30, H - 30, 20) if all(abs(y - ly) > 8 for ly in LINE_YS)])
    # The far-right line, tracked from its distorted position.
    truth = distort_points(calib, np.column_stack([np.full(rows.size, LINE_XS[-1]), rows]))
    centroids = np.array([
        _vertical_line_centroids(distorted, tx, [y])[0] for (tx, _), y in zip(truth, rows)
    ])
    fit = np.polyval(np.polyfit(rows, centroids, 1), rows)
    assert np.abs(centroids - fit).max() > 2.0


def test_undistorter_handles_uint16_and_single_channel_and_rejects_others():
    calib = _calib()
    rng = np.random.default_rng(0)
    img16 = rng.integers(0, 65535, (H, W, 3), dtype=np.uint16)
    out16 = Undistorter(calib, interpolation="linear").apply(img16)
    assert out16.dtype == np.uint16 and out16.shape == img16.shape
    plane = rng.random((H, W)).astype(np.float32)
    assert Undistorter(calib).apply(plane).shape == (H, W)
    with pytest.raises(TypeError, match="uint16 or float32"):
        Undistorter(calib).apply(img16.astype(np.uint8))
    with pytest.raises(ValueError, match="HxW or HxWx3"):
        Undistorter(calib).apply(np.zeros((H, W, 4), np.float32))
    with pytest.raises(ValueError, match="unknown interpolation"):
        Undistorter(calib, interpolation="nearest")


def test_undistorter_float_output_stays_in_unit_range():
    calib = _calib()
    step = np.zeros((H, W, 3), np.float32)
    step[:, W // 2:] = 1.0                      # a hard edge for cubic to ring on
    out = Undistorter(calib, interpolation="lanczos4").apply(step)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_undistorter_raises_on_size_mismatch():
    calib = _calib()
    with pytest.raises(CalibrationMismatch, match="does not match the calibration"):
        Undistorter(calib).apply(np.zeros((H // 2, W // 2, 3), np.float32))


def test_map_caching_builds_once_and_no_cache_rebuilds_and_frees():
    calib = _calib()
    img = np.zeros((H, W, 3), np.float32)
    cached = Undistorter(calib)
    assert not cached.maps_cached
    cached.apply(img)
    cached.apply(img)
    assert cached.build_count == 1 and cached.maps_cached
    cached.release()
    assert not cached.maps_cached

    uncached = Undistorter(calib, cache_maps=False)
    uncached.apply(img)
    uncached.apply(img)
    assert uncached.build_count == 2 and not uncached.maps_cached


def test_map_memory_estimate():
    calib = _calib()
    assert Undistorter(calib).map_memory_bytes == W * H * 8          # G only, float32 pair
    ca = LateralCA(r_norm=calib.r_norm, coeffs_r=(0.0, 0.0, 0.0), coeffs_b=(0.0, 0.0, 0.0))
    assert Undistorter(_calib(lateral_ca=ca)).map_memory_bytes == W * H * 8 * 3
    assert Undistorter(_calib(lateral_ca=ca), fixed_point=True).map_memory_bytes == W * H * 6 * 3
    # correct_ca is moot without a CA block in the file.
    assert not Undistorter(calib, correct_ca=True).corrects_lateral_ca
    assert estimate_map_memory_bytes((9564, 6376), correct_ca=True, fixed_point=False) > 1.4e9


def test_fixed_point_maps_match_float_maps_closely():
    calib = _calib()
    rng = np.random.default_rng(2)
    img = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 2.0)
    a = Undistorter(calib).apply(img)
    b = Undistorter(calib, fixed_point=True).apply(img)
    assert np.abs(a - b).max() < 0.02


# ---------------------------------------------------------------------------
# Lateral CA
# ---------------------------------------------------------------------------

CA_R = (0.0020, -0.0010, 0.0005)
CA_B = (-0.0015, 0.0008, -0.0003)


def test_fit_lateral_ca_recovers_coefficients_through_noise():
    calib = _calib()
    g = np.vstack([p for _, _, p in _poses(20, seed=4)])
    rng = np.random.default_rng(0)
    shift = lateral_ca_shift(CA_R, g, calib.principal_point, calib.r_norm)
    r = g + shift + rng.normal(0, 0.05, g.shape)
    coeffs, rms = fit_lateral_ca(g, r, calib.principal_point, calib.r_norm)
    assert 0.03 < rms < 0.1          # the injected noise, and nothing else
    # a1/a2 trade off against each other under noise (rho^2 vs rho^4 are
    # nearly collinear), so judge the fit by the shift it predicts: within
    # 0.1 px of the truth everywhere in the frame, including the corners.
    gx, gy = np.meshgrid(np.linspace(0, W - 1, 13), np.linspace(0, H - 1, 9))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    predicted = lateral_ca_shift(coeffs, grid, calib.principal_point, calib.r_norm)
    truth = lateral_ca_shift(CA_R, grid, calib.principal_point, calib.r_norm)
    assert np.linalg.norm(predicted - truth, axis=1).max() < 0.2
    assert abs(coeffs[0] - CA_R[0]) < 2e-4
    # Exact data fits exactly.
    coeffs0, rms0 = fit_lateral_ca(g, g + shift, calib.principal_point, calib.r_norm)
    assert np.allclose(coeffs0, CA_R, atol=1e-9) and rms0 < 1e-9
    with pytest.raises(ValueError, match="differ in shape"):
        fit_lateral_ca(g, g[:-1], calib.principal_point, calib.r_norm)


def test_lateral_ca_max_shift_is_at_an_image_corner():
    calib = _calib()
    ca = LateralCA(r_norm=calib.r_norm, coeffs_r=CA_R, coeffs_b=CA_B)
    corner = np.array([[W - 1.0, H - 1.0]])
    expected = np.linalg.norm(lateral_ca_shift(CA_R, corner, calib.principal_point, calib.r_norm))
    assert ca.max_shift_px("R", (W, H), calib.principal_point) >= expected - 1e-9
    with pytest.raises(ValueError, match="R and B"):
        ca.coeffs("G")


def _radially_scaled_copy(plane, a0, centre):
    """R(q) = G(p) with q = c0 + (p - c0) * (1 + a0): the channel imaged with a
    pure lateral scale (a0-only CA), built by inverse mapping."""
    cx, cy = centre
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx = (cx + (gx - cx) / (1.0 + a0)).astype(np.float32)
    my = (cy + (gy - cy) / (1.0 + a0)).astype(np.float32)
    return cv2.remap(plane, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def test_composed_ca_maps_realign_the_channels():
    """G is an undistorted line grid, R and B are radially scaled copies (the
    lateral CA of a real lens, a0-only so the inverse is exact). After
    Undistorter.apply with CA the channel edges coincide; without it they are
    the couple of px apart they started."""
    a_r, a_b = 0.003, -0.002
    calib_geom = _calib(dist=np.zeros(5))           # isolate CA from distortion
    centre = calib_geom.principal_point
    ux, uy = _undistorted_grid(calib_geom)
    g = _line_pattern(ux, uy)
    r = _radially_scaled_copy(g, a_r, centre)
    b = _radially_scaled_copy(g, a_b, centre)
    img = np.stack([r, g, b], axis=-1)

    ca = LateralCA(r_norm=calib_geom.r_norm, coeffs_r=(a_r, 0.0, 0.0), coeffs_b=(a_b, 0.0, 0.0))
    calib = _calib(lateral_ca=ca, dist=np.zeros(5))
    rows = [y for y in np.arange(40, H - 40, 20) if all(abs(y - ly) > 8 for ly in LINE_YS)]
    x = LINE_XS[-1]                                  # far from centre: largest CA shift

    misaligned_r = _vertical_line_centroids(img[..., 0], x + a_r * (x - centre[0]), rows)
    aligned_g = _vertical_line_centroids(img[..., 1], x, rows)
    assert np.abs(misaligned_r - aligned_g).mean() > 1.0   # ~1.5 px of CA to remove

    out = Undistorter(calib, correct_ca=True).apply(img)
    assert Undistorter(calib).corrects_lateral_ca
    for ch in (0, 2):
        centroids = _vertical_line_centroids(out[..., ch], x, rows)
        reference = _vertical_line_centroids(out[..., 1], x, rows)
        assert np.abs(centroids - reference).max() < 0.15, f"channel {ch} still misaligned"

    # Turning CA off leaves R where it was.
    off = Undistorter(calib, correct_ca=False).apply(img)
    assert not Undistorter(calib, correct_ca=False).corrects_lateral_ca
    assert np.abs(_vertical_line_centroids(off[..., 0], x + a_r * (x - centre[0]), rows) - aligned_g).mean() > 1.0


def test_build_maps_shares_g_maps_without_ca():
    calib = _calib()
    maps = build_maps(calib, correct_ca=True)
    assert maps["R"] is maps["G"] and maps["B"] is maps["G"]
    ca = LateralCA(r_norm=calib.r_norm, coeffs_r=CA_R, coeffs_b=CA_B)
    maps = build_maps(_calib(lateral_ca=ca), correct_ca=True)
    assert maps["R"] is not maps["G"]
    # The R map is G's map shifted by d_R evaluated at the G source position.
    gx, gy = maps["G"]
    src = np.stack([gx, gy], axis=-1)
    expected = src + lateral_ca_shift(CA_R, src, calib.principal_point, calib.r_norm)
    assert np.allclose(maps["R"][0], expected[..., 0], atol=1e-3)
    assert np.allclose(maps["R"][1], expected[..., 1], atol=1e-3)


# ---------------------------------------------------------------------------
# File format, match check
# ---------------------------------------------------------------------------

def _full_calib():
    ca = LateralCA(
        r_norm=float(np.hypot(W, H) / 2), coeffs_r=CA_R, coeffs_b=CA_B,
        stats={"R": {"rms_residual_px": 0.11, "max_shift_px_at_corner": 2.8},
               "B": {"rms_residual_px": 0.13, "max_shift_px_at_corner": 3.4}},
    )
    meta = {
        "created_at": "2026-09-10T15:42:00Z",
        "tool": "image-processing calibrate_distortion 0.1.0",
        "camera": {"body": "ILCE-7RM5", "serial": None, "lens": "FE PZ 16-35mm F4 G",
                   "zoom_position": 0, "focus_position": 1234, "units": "sdk_raw",
                   "aperture": "f/8", "shutter": "1/200", "iso": 100},
        "develop": {"user_flip": 0, "half_size": False},
        "quality": {"rms_reprojection_px": 0.38, "holdout_rms_px": 0.42, "n_frames_used": 31},
        "sources": ["DSC00123.ARW"],
        "sha256_of_sources": "abc",
    }
    return _calib(lateral_ca=ca, meta=meta, dist=np.array([-0.1012345678901234, 0.0212345678901234, 5e-4, -3e-4, 1e-5]))


def test_json_round_trip_preserves_arrays_exactly(tmp_path):
    calib = _full_calib()
    path = str(tmp_path / "cal" / "a7rv.json")
    calib.save(path)
    loaded = DistortionCalibration.load(path)
    assert loaded.image_size == calib.image_size
    assert np.array_equal(loaded.camera_matrix, calib.camera_matrix)
    assert np.array_equal(loaded.dist_coeffs, calib.dist_coeffs)
    assert loaded.lateral_ca == calib.lateral_ca
    assert loaded.meta == calib.meta
    assert loaded.model == "opencv_standard"
    with open(path) as f:
        doc = json.load(f)
    assert doc["schema_version"] == 1
    assert list(doc)[:4] == ["schema_version", "created_at", "tool", "camera"]
    assert doc["lateral_ca"]["R"]["rms_residual_px"] == 0.11
    assert doc["model"] == "opencv_standard"


def test_load_rejects_unknown_schema_and_malformed_files(tmp_path):
    good = _full_calib().to_dict()

    def write(doc, name="c.json"):
        p = tmp_path / name
        p.write_text(json.dumps(doc))
        return str(p)

    with pytest.raises(CalibrationError, match="schema_version"):
        DistortionCalibration.load(write({**good, "schema_version": 2}))
    with pytest.raises(CalibrationError, match="missing"):
        DistortionCalibration.load(write({k: v for k, v in good.items() if k != "camera_matrix"}))
    with pytest.raises(CalibrationError, match="3x3"):
        DistortionCalibration.load(write({**good, "camera_matrix": [[1, 0], [0, 1]]}))
    with pytest.raises(CalibrationError, match="5 .* or 8"):
        DistortionCalibration.load(write({**{k: v for k, v in good.items() if k != "model"}, "dist_coeffs": [1, 2, 3]}))
    with pytest.raises(CalibrationError, match="needs 8"):
        DistortionCalibration.load(write({**good, "model": "opencv_rational"}))
    with pytest.raises(CalibrationError, match="unknown model"):
        DistortionCalibration.load(write({**good, "model": "fisheye"}))
    with pytest.raises(CalibrationError, match="lateral_ca"):
        DistortionCalibration.load(write({**good, "lateral_ca": {"r_norm": 1.0, "R": {"coeffs": [1]}, "B": {"coeffs": [1, 2, 3]}}}))
    with pytest.raises(CalibrationError, match="not found"):
        DistortionCalibration.load(str(tmp_path / "missing.json"))
    p = tmp_path / "junk.json"
    p.write_text("{not json")
    with pytest.raises(CalibrationError, match="cannot read"):
        DistortionCalibration.load(str(p))
    # A file without a CA block loads with lateral_ca None.
    loaded = DistortionCalibration.load(write({**good, "lateral_ca": None}))
    assert loaded.lateral_ca is None


def test_check_match_size_first_then_metadata_warnings():
    calib = _full_calib()
    assert calib.check_match(image_size=(W, H)) == []
    msgs = calib.check_match(
        image_size=(W // 2, H // 2), lens="FE 24-70mm F2.8 GM", zoom_position=5, focus_position=1300,
    )
    assert len(msgs) == 4
    assert "does not match the calibration" in msgs[0]
    assert "lens" in msgs[1] and "zoom_position 5" in msgs[2] and "focus_position 1300" in msgs[3]
    # Same lens written differently is not a mismatch; None on either side skips.
    assert calib.check_match(image_size=(W, H), lens="  fe pz  16-35mm f4 g ") == []
    assert calib.check_match(image_size=None, zoom_position=0, focus_position=1234) == []
    assert _calib().check_match(image_size=(W, H), lens="anything", zoom_position=9) == []


def test_calibration_validates_geometry_on_construction():
    with pytest.raises(CalibrationError, match="3x3"):
        DistortionCalibration((W, H), np.eye(2), DIST, None, {})
    with pytest.raises(CalibrationError, match="5 .* or 8"):
        DistortionCalibration((W, H), K, np.zeros(6), None, {})
    with pytest.raises(CalibrationError, match="positive"):
        DistortionCalibration((0, H), K, DIST, None, {})
    with pytest.raises(CalibrationError, match="non-finite"):
        DistortionCalibration((W, H), K, np.array([np.nan, 0, 0, 0, 0]), None, {})


# ---------------------------------------------------------------------------
# EXIF, hashing, acceptance
# ---------------------------------------------------------------------------

def test_read_raw_exif_best_effort(tmp_path):
    import tifffile

    p = str(tmp_path / "shot.tif")
    # Make/Model are baseline TIFF tags; that is enough to prove the reader
    # walks the IFD. (An EXIF sub-IFD is what the real ARW carries.)
    tifffile.imwrite(p, np.zeros((4, 4), np.uint8), extratags=[(272, "s", 0, "ILCE-7RM5", True)])
    exif = cd.read_raw_exif(p)
    assert exif["body"] == "ILCE-7RM5"
    assert exif["lens"] is None and exif["aperture"] is None
    junk = tmp_path / "x.CR3"
    junk.write_bytes(b"\x00" * 64)
    assert all(v is None for v in cd.read_raw_exif(str(junk)).values())
    assert cd._rational((110, 10)) == 11.0 and cd._rational("x") is None


def test_sha256_of_sources_is_order_independent(tmp_path):
    a = tmp_path / "a.ARW"
    b = tmp_path / "b.ARW"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")
    assert cd.sha256_of_sources([str(a), str(b)]) == cd.sha256_of_sources([str(b), str(a)])
    b.write_bytes(b"ccc")
    assert cd.sha256_of_sources([str(a), str(b)]) != cd.sha256_of_sources([str(a), str(a)])


def test_evaluate_acceptance_flags_each_criterion():
    quality = {
        "rms_reprojection_px": 0.38, "holdout_rms_px": 0.42,
        "per_image_rms_px": {"a": 0.35, "b": 0.6},
        "coverage_grid_4x3": [[5, 7, 6, 4], [8, 12, 11, 7], [4, 6, 6, 5]],
        "straight_line_test": {"frame": "x", "max_bow_px_before": 9.6, "max_bow_px_after": 0.4,
                               "center_crop": {"max_bow_px_before": 3.1, "max_bow_px_after": 0.3}},
    }
    ca = {"R": {"rms_residual_px": 0.11, "max_shift_px_at_corner": 2.8},
          "B": {"rms_residual_px": 0.13, "max_shift_px_at_corner": 3.4}}
    checks = cd.evaluate_acceptance(quality, ca)
    assert all(c.ok for c in checks)
    assert [c.name for c in checks][:3] == ["reprojection RMS", "per-image max RMS", "holdout RMS"]

    bad = dict(quality, rms_reprojection_px=0.7, holdout_rms_px=1.2)
    bad["coverage_grid_4x3"] = [[1, 7, 6, 4], [8, 12, 11, 7], [4, 6, 6, 5]]
    bad["straight_line_test"] = dict(quality["straight_line_test"], center_crop={"max_bow_px_before": 3.1, "max_bow_px_after": 1.4})
    by_name = {c.name: c for c in cd.evaluate_acceptance(bad, {"R": {"rms_residual_px": 0.5}, "B": {"rms_residual_px": 0.1}})}
    assert not by_name["reprojection RMS"].ok and by_name["reprojection RMS"].label == "WARN"
    assert not by_name["holdout RMS"].ok
    assert not by_name["frame coverage"].ok and "(0, 0)" in by_name["frame coverage"].detail
    assert not by_name["straight-line (center 50%)"].ok
    assert not by_name["lateral CA fit R"].ok and by_name["lateral CA fit B"].ok

    none = cd.evaluate_acceptance({"rms_reprojection_px": 0.3, "per_image_rms_px": {"a": 0.3}}, None)
    by_name = {c.name: c for c in none}
    assert by_name["holdout RMS"].ok is None and by_name["holdout RMS"].label == "N/A "
    assert by_name["lateral CA fit"].ok is None


# ---------------------------------------------------------------------------
# End to end: the CLI run on synthetic "ARWs" (load_linear_rgb is replaced by
# the board renderer, so every stage after the demosaic runs for real).
# ---------------------------------------------------------------------------

def test_cli_run_end_to_end_on_synthetic_boards(tmp_path, monkeypatch):
    calib_true = _calib()
    poses = _covering_poses(8)
    assert len(poses) == 20
    src_dir = tmp_path / "calib"
    src_dir.mkdir()
    rendered = {}
    for i, (rvec, tvec, _) in enumerate(poses):
        name = f"DSC{i:05d}.ARW"
        (src_dir / name).write_bytes(b"synthetic")
        rendered[str(src_dir / name)] = _render_board(calib_true, rvec, tvec)
    # One frame with no board at all: must be skipped, not fatal.
    (src_dir / "DSC99999.ARW").write_bytes(b"blank")
    rendered[str(src_dir / "DSC99999.ARW")] = np.full((H, W), 0.9, np.float32)

    def fake_load(path, **kwargs):
        assert kwargs["user_flip"] == 0 and kwargs["half_size"] is False
        return np.stack([rendered[path]] * 3, axis=-1)

    monkeypatch.setattr(cd, "load_linear_rgb", fake_load)

    out = str(tmp_path / "out" / "synthetic.json")
    logs = []
    result = cd.run(cd.RunOptions(
        inputs=[str(src_dir / "*.ARW")], pattern=PATTERN, square_mm=SQUARE, out=out,
        zoom_position=0, focus_position=1234, holdout=3, detect_scale=0.5, jobs=2,
        debug_dir=str(tmp_path / "debug"), straight_edge_frame="DSC00003.ARW",
    ), log=logs.append)

    assert result["overall"] == "PASS", "\n".join(logs)
    loaded = DistortionCalibration.load(out)
    assert loaded.image_size == (W, H)
    assert np.allclose(loaded.camera_matrix, K, atol=1.5)
    # k1/k2 trade off against each other; what matters is that the recovered
    # model undistorts like the true lens - to well under a pixel everywhere.
    # (Inside the region the boards covered; the extreme frame corners are
    # extrapolation for a 20-frame synthetic set.)
    gx, gy = np.meshgrid(np.linspace(60, W - 60, 13), np.linspace(60, H - 60, 9))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    assert np.linalg.norm(undistort_points(loaded, grid) - undistort_points(calib_true, grid), axis=1).max() < 0.5
    q = loaded.quality_meta()
    assert q["n_frames_used"] + q["n_frames_rejected"] + q["n_holdout"] == 20
    assert q["detection_failures"] == ["DSC99999.ARW"]
    assert q["rms_reprojection_px"] < 0.5 and q["holdout_rms_px"] < 0.6
    assert q["straight_line_test"]["frame"] == "DSC00003.ARW"
    assert q["straight_line_test"]["center_crop"]["max_bow_px_after"] < 1.0
    # The synthetic channels are identical, so the CA fit must find ~no shift.
    assert loaded.lateral_ca is not None
    assert abs(loaded.lateral_ca.max_shift_px("R", (W, H), loaded.principal_point)) < 0.3
    cam = loaded.camera_meta()
    assert cam["zoom_position"] == 0 and cam["focus_position"] == 1234
    assert loaded.meta["develop"]["user_flip"] == 0
    assert "sha256_of_sources" in loaded.meta
    assert any("skip DSC99999.ARW" in line for line in logs)
    debug = sorted(os.listdir(tmp_path / "debug"))
    assert "coverage.png" in debug and "residual_quiver.png" in debug
    assert "DSC00003_before.jpg" in debug and "DSC00003_after.jpg" in debug
    assert "DSC00000_corners.jpg" in debug

    # And the file it wrote drives an Undistorter on a frame of that size.
    Undistorter(loaded).apply(np.zeros((H, W, 3), np.float32))


def test_cli_run_refuses_too_few_frames(tmp_path):
    with pytest.raises(ValueError, match="at least 5"):
        cd.run(cd.RunOptions(inputs=[str(tmp_path / "*.ARW")], pattern=PATTERN, square_mm=SQUARE,
                             out=str(tmp_path / "x.json")), log=lambda *_: None)


def test_main_maps_errors_to_exit_codes(tmp_path, capsys):
    code = cd.main(["--input", str(tmp_path / "*.ARW"), "--pattern", "10x7",
                    "--square-mm", "55", "--out", str(tmp_path / "x.json")])
    assert code == 2
    assert "error:" in capsys.readouterr().err
