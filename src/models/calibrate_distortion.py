"""
calibrate_distortion.py
-----------------------
Turn a folder of checkerboard ARWs into a distortion + lateral-CA calibration
JSON for ``distortion.py`` (consumed by the color-correction ``develop`` path).

    venv/bin/python scripts/calibrate_distortion.py \\
        --input 'calib/2026-09-10-16mm-f8/*.ARW' \\
        --pattern 10x7 --square-mm 55.0 \\
        --out calibration/a7rv-selp1635g-16mm-f8.json \\
        --zoom-position 0 --focus-position 1234 \\
        [--model standard|rational|auto] [--no-ca] [--holdout N]
        [--holdout-files a.ARW b.ARW] [--straight-edge-frame DSC00160.ARW]
        [--detect-scale 0.25] [--reject-above-px 1.0] [--debug-dir ./debug] [--jobs N]

What it does, per frame: demosaic with the production develop's sensor-frame
parameters (``image_io.SENSOR_FRAME_KWARGS`` - one source of truth with the
pipeline that will apply the result), find the board on a downscaled
gamma-encoded copy, refine the corners on the full-res G channel, then refine
again on R and B starting from the G corners for the CA fit. Across frames:
``cv2.calibrateCamera``, reject outlier frames once, hold out validation
frames, fit the CA polynomials, run a straight-line test, and write the JSON
with a quality block the module logs at load. The summary ends in PASS/WARN
per acceptance criterion:

    RMS reprojection <= 0.5 px, per-image max <= 1.0 px after rejection;
    holdout RMS within 20 % of training; no 4x3 coverage cell under 3 frames;
    straight-line bow in the central 50 % crop <= 1.0 px after; CA fit
    residual <= 0.3 px RMS per channel.

The library half (pure functions over point sets) is importable so the
tests can exercise it without ARWs or a camera.
"""

import argparse
import glob
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from models.distortion import (
    DistortionCalibration,
    LateralCA,
    Undistorter,
    distort_points,
    fit_lateral_ca,
    undistort_points,
)
from models.image_io import (
    DEFAULT_DEMOSAIC,
    SENSOR_FRAME_KWARGS,
    load_linear_rgb,
    sensor_frame_record,
)

TOOL_VERSION = "0.1.0"

# Acceptance criteria (spec section 9). The CLI reports each as PASS/WARN.
MAX_RMS_PX = 0.5
MAX_PER_IMAGE_RMS_PX = 1.0
MAX_HOLDOUT_RATIO = 1.2
# ...but a ratio alone flags noise when both numbers are tiny (0.10 vs 0.13 px
# is not overfit), so the holdout may also sit within this many px of training.
HOLDOUT_ABS_FLOOR_PX = 0.1
MIN_COVERAGE_PER_CELL = 3
MAX_CENTER_BOW_AFTER_PX = 1.0
MAX_CA_RESIDUAL_PX = 0.3

COVERAGE_COLS, COVERAGE_ROWS = 4, 3

# Sub-pixel refinement: cornerSubPix's winSize is a half-width, so 5 gives an
# 11x11 window - a few px of CA shift converges to the channel's own corner
# from the G start without wandering to a neighbour.
SUBPIX_HALF_WINDOW = 5
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-4)


# ---------------------------------------------------------------------------
# Per-frame development + detection
# ---------------------------------------------------------------------------

def stretch_for_detection(channel: np.ndarray) -> np.ndarray:
    """
    Linear float channel -> gamma-encoded (1/2.2) float32 in [0, 1], stretched
    between the 1st and 99.5th percentiles. Linear 16-bit looks nearly black
    to the corner detector and the strobe leaves a lot of headroom, so
    normalise before detecting; percentile rather than min/max so a specular
    or a dead pixel can't own the range.
    """
    sample = channel[::8, ::8] if channel.size > 1_000_000 else channel
    lo, hi = np.percentile(sample, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((channel.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return np.power(norm, 1.0 / 2.2, dtype=np.float32)


def to_gray8(stretched: np.ndarray) -> np.ndarray:
    return np.rint(stretched * 255.0).clip(0, 255).astype(np.uint8)


def parse_pattern(text: str) -> Tuple[int, int]:
    """``"10x7"`` -> (10, 7) inner corners per row, per column."""
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"--pattern must look like 10x7 (inner corners), got {text!r}") from exc
    if cols < 3 or rows < 3:
        raise ValueError(f"--pattern needs at least 3x3 inner corners, got {text!r}")
    if cols == rows or (cols % 2) == (rows % 2):
        # Even x odd keeps the board's 180-degree orientation unambiguous.
        print(
            f"warning: pattern {cols}x{rows} is orientation-ambiguous (use even x odd); "
            f"calibration still works, corner ordering may flip between frames",
            file=sys.stderr,
        )
    return cols, rows


def board_object_points(pattern: Tuple[int, int], square_mm: float) -> np.ndarray:
    """Planar board corners in mm, row-major to match OpenCV's corner order."""
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_mm)
    return grid


def detect_checkerboard(
    gray8: np.ndarray, pattern: Tuple[int, int], detect_scale: float = 0.25
) -> Optional[np.ndarray]:
    """
    Coarse corner detection on a downscaled copy (61 MP is slow and no more
    reliable for the *find* step). ``findChessboardCornersSB`` first - it is
    the more accurate, orientation-consistent detector - with the classic
    detector + ``cornerSubPix`` as the fallback. Returns ``(N, 2)`` full-res
    coordinates (coarse; refine with ``refine_corners``), or None when the
    whole board is not found.
    """
    if detect_scale <= 0 or detect_scale > 1:
        raise ValueError(f"detect_scale must be in (0, 1], got {detect_scale}")
    if detect_scale < 1.0:
        small = cv2.resize(gray8, None, fx=detect_scale, fy=detect_scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray8
    ok, corners = cv2.findChessboardCornersSB(
        small, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if not ok:
        ok, corners = cv2.findChessboardCorners(
            small, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ok:
            corners = cv2.cornerSubPix(
                small, np.ascontiguousarray(corners, dtype=np.float32),
                (SUBPIX_HALF_WINDOW, SUBPIX_HALF_WINDOW), (-1, -1), _SUBPIX_CRITERIA,
            )
    if not ok or corners is None or len(corners) != pattern[0] * pattern[1]:
        return None
    pts = corners.reshape(-1, 2).astype(np.float64)
    # INTER_AREA maps full-res pixel centres to small-image pixel centres as
    # small = (full + 0.5) * s - 0.5; invert that rather than a bare divide.
    return ((pts + 0.5) / detect_scale - 0.5).astype(np.float32)


def refine_corners(
    channel: np.ndarray,
    corners: np.ndarray,
    half_window: int = SUBPIX_HALF_WINDOW,
) -> np.ndarray:
    """``cornerSubPix`` on a single float32 (or uint8) channel, starting from
    ``corners`` ``(N, 2)``. Returns the refined ``(N, 2)`` float32 array."""
    if channel.dtype not in (np.float32, np.uint8):
        channel = channel.astype(np.float32)
    pts = np.ascontiguousarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    refined = cv2.cornerSubPix(
        channel, pts, (half_window, half_window), (-1, -1), _SUBPIX_CRITERIA
    )
    return refined.reshape(-1, 2)


@dataclass
class FrameResult:
    path: str
    name: str
    image_size: Tuple[int, int]                   # (w, h)
    corners_g: Optional[np.ndarray] = None        # (N, 2) float32
    corners_r: Optional[np.ndarray] = None
    corners_b: Optional[np.ndarray] = None
    error: Optional[str] = None
    seconds: float = 0.0
    small_gray: Optional[np.ndarray] = None       # debug overlay base

    @property
    def detected(self) -> bool:
        return self.corners_g is not None


def process_frame(
    path: str,
    pattern: Tuple[int, int],
    *,
    detect_scale: float = 0.25,
    want_ca: bool = True,
    demosaic: str = DEFAULT_DEMOSAIC,
    keep_debug: bool = False,
) -> FrameResult:
    """Develop one RAW in the sensor frame and measure its corners. Never
    raises for a frame problem: a missing board is reported in ``error`` and
    the frame is skipped, not fatal for the run."""
    start = time.perf_counter()
    name = os.path.basename(path)
    try:
        linear = load_linear_rgb(
            path, white_balance="camera", demosaic=demosaic, **SENSOR_FRAME_KWARGS
        )
    except Exception as exc:  # noqa: BLE001 - reported per frame
        return FrameResult(path, name, (0, 0), error=f"develop failed: {exc}")
    h, w = linear.shape[:2]
    result = FrameResult(path, name, (w, h))
    try:
        g = stretch_for_detection(linear[..., 1])
        gray8 = to_gray8(g)
        coarse = detect_checkerboard(gray8, pattern, detect_scale)
        if keep_debug:
            result.small_gray = cv2.resize(gray8, None, fx=detect_scale, fy=detect_scale,
                                           interpolation=cv2.INTER_AREA)
        if coarse is None:
            result.error = "checkerboard not found (whole board must be visible)"
            return result
        result.corners_g = refine_corners(g, coarse)
        if want_ca:
            result.corners_r = refine_corners(stretch_for_detection(linear[..., 0]), result.corners_g)
            result.corners_b = refine_corners(stretch_for_detection(linear[..., 2]), result.corners_g)
    finally:
        result.seconds = time.perf_counter() - start
    return result


# ---------------------------------------------------------------------------
# Calibration maths over point sets (hardware-free)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms: float
    per_image_rms: List[float]
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    model: str

    def to_calibration(
        self, image_size: Tuple[int, int], lateral_ca: Optional[LateralCA], meta: Dict[str, Any]
    ) -> DistortionCalibration:
        return DistortionCalibration(
            image_size=image_size, camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs, lateral_ca=lateral_ca, meta=meta,
        )


def calibrate_from_points(
    object_points: np.ndarray,
    image_points: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    model: str = "standard",
) -> CalibrationResult:
    """
    ``cv2.calibrateCamera`` over ``image_points`` (one ``(N, 2)`` array per
    frame, all sharing ``object_points``). ``model`` is ``"standard"`` (k1, k2,
    p1, p2, k3) or ``"rational"`` (adds k4-k6). Returns intrinsics, RMS, and
    the per-image RMS the rejection step keys on.
    """
    if model not in ("standard", "rational"):
        raise ValueError(f"model must be 'standard' or 'rational', got {model!r}")
    if len(image_points) < 3:
        raise ValueError(f"need at least 3 frames to calibrate, got {len(image_points)}")
    obj = [np.ascontiguousarray(object_points, dtype=np.float32)] * len(image_points)
    img = [np.ascontiguousarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in image_points]
    flags = cv2.CALIB_RATIONAL_MODEL if model == "rational" else 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)
    rms, k, dist, rvecs, tvecs, _, _, per_view = cv2.calibrateCameraExtended(
        obj, img, tuple(int(v) for v in image_size), None, None,
        flags=flags, criteria=criteria,
    )
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    n = 8 if model == "rational" else 5
    dist = dist[:n]
    return CalibrationResult(
        camera_matrix=np.asarray(k, dtype=np.float64), dist_coeffs=dist,
        rms=float(rms), per_image_rms=[float(v) for v in np.asarray(per_view).reshape(-1)],
        rvecs=list(rvecs), tvecs=list(tvecs), model=f"opencv_{model}",
    )


def reprojection_rms(
    object_points: np.ndarray, image_pts: np.ndarray, k: np.ndarray, dist: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Solve the pose of one frame under fixed intrinsics and report its
    reprojection RMS (px) plus the pose - the holdout check."""
    obj = np.ascontiguousarray(object_points, dtype=np.float64)
    img = np.ascontiguousarray(image_pts, dtype=np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed on a holdout frame")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    err = proj.reshape(-1, 2) - img.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))), rvec, tvec


def board_tilt_deg(rvec: np.ndarray) -> float:
    """Angle between the board normal and the optical axis - 0 is square-on."""
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return float(np.degrees(np.arccos(np.clip(abs(rot[2, 2]), -1.0, 1.0))))


def coverage_grid(
    image_points: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    cols: int = COVERAGE_COLS,
    rows: int = COVERAGE_ROWS,
) -> List[List[int]]:
    """How many frames put at least one corner in each cell of a cols x rows
    grid over the frame (returned rows-first, like the JSON). Thin edge and
    corner cells are the usual reason for a bad calibration: that is where the
    distortion signal lives."""
    w, h = image_size
    counts = np.zeros((rows, cols), dtype=int)
    for pts in image_points:
        p = np.asarray(pts).reshape(-1, 2)
        cx = np.clip((p[:, 0] / w * cols).astype(int), 0, cols - 1)
        cy = np.clip((p[:, 1] / h * rows).astype(int), 0, rows - 1)
        hit = np.zeros((rows, cols), dtype=bool)
        hit[cy, cx] = True
        counts += hit
    return counts.tolist()


def _line_residual_max(pts: np.ndarray) -> float:
    """Max perpendicular distance of ``pts`` from their total-least-squares
    line."""
    p = np.asarray(pts, dtype=np.float64)
    centre = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - centre, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    return float(np.abs((p - centre) @ normal).max())


def line_bow(
    corners: np.ndarray,
    pattern: Tuple[int, int],
    region: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[float]:
    """
    Max deviation (px) of any board row or column of corners from a straight
    line. Straight lines stay straight under a pinhole whatever the board's
    tilt, so any bow is lens distortion. ``region`` = (x0, y0, x1, y1) keeps
    only corners inside it (the central-crop variant); lines with fewer than
    3 corners left are skipped. None when no line qualifies.
    """
    cols, rows = pattern
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    lines = [grid[r, :, :] for r in range(rows)] + [grid[:, c, :] for c in range(cols)]
    worst: Optional[float] = None
    for line in lines:
        pts = line
        if region is not None:
            x0, y0, x1, y1 = region
            keep = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
            pts = pts[keep]
        if len(pts) < 3:
            continue
        dev = _line_residual_max(pts)
        worst = dev if worst is None else max(worst, dev)
    return worst


def straight_line_report(
    corners: np.ndarray, pattern: Tuple[int, int], calib: DistortionCalibration, frame: str
) -> Dict[str, Any]:
    """Before/after bow for one board, full frame and the central 50 % crop
    (half of each axis, centred - the product crop region)."""
    w, h = calib.image_size
    centre = (w * 0.25, h * 0.25, w * 0.75, h * 0.75)
    after = undistort_points(calib, corners)
    return {
        "frame": frame,
        "max_bow_px_before": line_bow(corners, pattern),
        "max_bow_px_after": line_bow(after, pattern),
        "center_crop": {
            "crop": "center 50%",
            "max_bow_px_before": line_bow(corners, pattern, centre),
            "max_bow_px_after": line_bow(after, pattern, centre),
        },
    }


def invalid_output_fraction(calib: DistortionCalibration) -> float:
    """Fraction of output pixels whose source falls outside the frame (black
    after remap). ~0 for barrel with newCameraMatrix = K; a mustache lens can
    leave slivers at the extreme edges."""
    w, h = calib.image_size
    step = max(1, min(w, h) // 512)
    xs = np.arange(0, w, step, dtype=np.float64)
    ys = np.arange(0, h, step, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    src = distort_points(calib, pts)
    outside = (src[:, 0] < 0) | (src[:, 0] > w - 1) | (src[:, 1] < 0) | (src[:, 1] > h - 1)
    return float(outside.mean())


# ---------------------------------------------------------------------------
# EXIF (best effort; ARW is TIFF-based so tifffile can read the IFDs)
# ---------------------------------------------------------------------------

def _rational(value: Any) -> Optional[float]:
    if isinstance(value, (tuple, list)) and len(value) == 2 and value[1]:
        return float(value[0]) / float(value[1])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def read_raw_exif(path: str) -> Dict[str, Any]:
    """
    Body / serial / lens / aperture / shutter / ISO from a TIFF-based RAW
    (ARW, NEF, DNG). Best effort: anything unreadable is None, and a file
    tifffile can't open (CR3 is ISO-BMFF, not TIFF) yields all-None rather
    than an error - the metadata is for the match check, not the maths.
    """
    out: Dict[str, Any] = {
        "body": None, "serial": None, "lens": None,
        "aperture": None, "shutter": None, "iso": None,
    }
    try:
        import tifffile  # type: ignore
    except Exception:  # pragma: no cover - depends on the host
        return out
    try:
        with tifffile.TiffFile(path) as tf:
            tags = tf.pages[0].tags
            model = tags.get("Model")
            if model is not None:
                out["body"] = str(model.value).strip()
            exif_tag = tags.get("ExifTag")
            exif = exif_tag.value if exif_tag is not None and isinstance(exif_tag.value, dict) else {}
    except Exception:  # noqa: BLE001 - not a TIFF container, or truncated
        return out
    lens = exif.get("LensModel")
    if lens:
        out["lens"] = str(lens).strip()
    serial = exif.get("BodySerialNumber")
    if serial:
        out["serial"] = str(serial).strip()
    f_number = _rational(exif.get("FNumber"))
    if f_number:
        out["aperture"] = f"f/{f_number:g}"
    exposure = _rational(exif.get("ExposureTime"))
    if exposure:
        out["shutter"] = f"1/{round(1.0 / exposure):d}" if exposure < 1 else f"{exposure:g}s"
    iso = exif.get("ISOSpeedRatings", exif.get("PhotographicSensitivity"))
    if isinstance(iso, (tuple, list)) and iso:
        iso = iso[0]
    if isinstance(iso, (int, float)):
        out["iso"] = int(iso)
    return out


def sha256_of_sources(paths: Sequence[str]) -> str:
    """One digest over the per-file SHA-256s (in name order), so the JSON can
    say which exact set of ARWs produced it without listing 35 hashes."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=os.path.basename):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        digest.update(h.hexdigest().encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    ok: Optional[bool]          # None = not evaluated (no data)
    detail: str

    @property
    def label(self) -> str:
        return "PASS" if self.ok else ("N/A " if self.ok is None else "WARN")


def evaluate_acceptance(quality: Dict[str, Any], lateral_ca: Optional[Dict[str, Any]]) -> List[Check]:
    checks: List[Check] = []
    rms = quality.get("rms_reprojection_px")
    per_image = quality.get("per_image_rms_px") or {}
    worst = max(per_image.values()) if per_image else None
    checks.append(Check(
        "reprojection RMS", rms is not None and rms <= MAX_RMS_PX,
        f"{rms:.3f} px (limit {MAX_RMS_PX})" if rms is not None else "no calibration",
    ))
    checks.append(Check(
        "per-image max RMS", worst is not None and worst <= MAX_PER_IMAGE_RMS_PX,
        f"{worst:.3f} px (limit {MAX_PER_IMAGE_RMS_PX})" if worst is not None else "no frames",
    ))
    holdout = quality.get("holdout_rms_px")
    if holdout is None or not rms:
        checks.append(Check("holdout RMS", None, "no holdout frames (use --holdout N)"))
    else:
        checks.append(Check(
            "holdout RMS", holdout <= max(MAX_HOLDOUT_RATIO * rms, rms + HOLDOUT_ABS_FLOOR_PX),
            f"{holdout:.3f} px vs training {rms:.3f} px "
            f"(limit {MAX_HOLDOUT_RATIO:.1f}x or +{HOLDOUT_ABS_FLOOR_PX} px; much worse = "
            f"overfit, usually the rational model on too few frames)",
        ))
    grid = quality.get(f"coverage_grid_{COVERAGE_COLS}x{COVERAGE_ROWS}")
    if grid:
        thin = [(r, c) for r, row in enumerate(grid) for c, n in enumerate(row) if n < MIN_COVERAGE_PER_CELL]
        checks.append(Check(
            "frame coverage", not thin,
            f"min {min(min(r) for r in grid)} frames/cell (need {MIN_COVERAGE_PER_CELL})"
            + (f"; thin cells (row, col): {thin} - edge/corner cells are where the "
               f"distortion signal lives, shoot more boards there" if thin else ""),
        ))
    sl = quality.get("straight_line_test") or {}
    centre = sl.get("center_crop") or {}
    after = centre.get("max_bow_px_after")
    before = centre.get("max_bow_px_before")
    if after is None:
        checks.append(Check("straight-line (center 50%)", None, "no board available for the test"))
    else:
        checks.append(Check(
            "straight-line (center 50%)", after <= MAX_CENTER_BOW_AFTER_PX,
            f"bow {before:.2f} px -> {after:.2f} px in {sl.get('frame')} "
            f"(limit {MAX_CENTER_BOW_AFTER_PX}); full frame "
            f"{sl.get('max_bow_px_before', float('nan')):.2f} -> {sl.get('max_bow_px_after', float('nan')):.2f} px",
        ))
    if lateral_ca:
        for ch in ("R", "B"):
            res = lateral_ca.get(ch, {}).get("rms_residual_px")
            checks.append(Check(
                f"lateral CA fit {ch}", res is not None and res <= MAX_CA_RESIDUAL_PX,
                f"residual {res:.3f} px RMS, corner shift "
                f"{lateral_ca[ch].get('max_shift_px_at_corner', float('nan')):.2f} px "
                f"(limit {MAX_CA_RESIDUAL_PX})" if res is not None else "not fitted",
            ))
    else:
        checks.append(Check("lateral CA fit", None, "disabled (--no-ca)"))
    return checks


# ---------------------------------------------------------------------------
# Debug output
# ---------------------------------------------------------------------------

def _write_corner_overlay(frame: FrameResult, pattern: Tuple[int, int], scale: float, out_dir: str) -> None:
    if frame.small_gray is None or frame.corners_g is None:
        return
    bgr = cv2.cvtColor(frame.small_gray, cv2.COLOR_GRAY2BGR)
    pts = ((frame.corners_g + 0.5) * scale - 0.5).astype(np.float32).reshape(-1, 1, 2)
    cv2.drawChessboardCorners(bgr, pattern, pts, True)
    cv2.imwrite(os.path.join(out_dir, os.path.splitext(frame.name)[0] + "_corners.jpg"), bgr)


def _write_quiver(
    frames: Sequence[FrameResult], object_points: np.ndarray, result: CalibrationResult,
    image_size: Tuple[int, int], out_dir: str, scale: float = 0.1, magnify: float = 50.0,
) -> None:
    w, h = image_size
    canvas = np.full((int(h * scale) + 1, int(w * scale) + 1, 3), 255, np.uint8)
    for frame, rvec, tvec in zip(frames, result.rvecs, result.tvecs):
        proj, _ = cv2.projectPoints(object_points, rvec, tvec, result.camera_matrix, result.dist_coeffs)
        obs = frame.corners_g.reshape(-1, 2)
        err = proj.reshape(-1, 2) - obs
        for p, e in zip(obs, err):
            start = (int(p[0] * scale), int(p[1] * scale))
            end = (int(p[0] * scale + e[0] * magnify), int(p[1] * scale + e[1] * magnify))
            cv2.arrowedLine(canvas, start, end, (0, 0, 200), 1, tipLength=0.3)
    cv2.putText(canvas, f"reprojection residuals x{magnify:g}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(out_dir, "residual_quiver.png"), canvas)


def _write_coverage(grid: List[List[int]], out_dir: str) -> None:
    rows, cols = len(grid), len(grid[0])
    cell = 120
    canvas = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
    peak = max(max(r) for r in grid) or 1
    for r in range(rows):
        for c in range(cols):
            n = grid[r][c]
            shade = int(255 - 180 * n / peak)
            colour = (shade, 255, shade) if n >= MIN_COVERAGE_PER_CELL else (200, 200, 255)
            cv2.rectangle(canvas, (c * cell, r * cell), ((c + 1) * cell - 1, (r + 1) * cell - 1), colour, -1)
            cv2.rectangle(canvas, (c * cell, r * cell), ((c + 1) * cell - 1, (r + 1) * cell - 1), (0, 0, 0), 1)
            cv2.putText(canvas, str(n), (c * cell + 45, r * cell + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "coverage.png"), canvas)


def _write_before_after(
    path: str, calib: DistortionCalibration, demosaic: str, out_dir: str, scale: float = 0.25
) -> None:
    """Undistort the straight-line frame's G channel at full res (the real
    maps, the real interpolation) and save downscaled before/after JPEGs."""
    linear = load_linear_rgb(path, white_balance="camera", demosaic=demosaic, **SENSOR_FRAME_KWARGS)
    g = stretch_for_detection(linear[..., 1])
    stem = os.path.splitext(os.path.basename(path))[0]
    before = cv2.resize(to_gray8(g), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_before.jpg"), before)
    und = Undistorter(calib, correct_ca=False, cache_maps=False).apply(np.ascontiguousarray(g))
    after = cv2.resize(to_gray8(und), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_after.jpg"), after)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

@dataclass
class RunOptions:
    inputs: List[str]
    pattern: Tuple[int, int]
    square_mm: float
    out: str
    zoom_position: Optional[int] = None
    focus_position: Optional[int] = None
    model: str = "standard"                   # standard | rational | auto
    fit_ca: bool = True
    holdout: Optional[int] = None
    holdout_files: List[str] = field(default_factory=list)
    straight_edge_frame: Optional[str] = None
    detect_scale: float = 0.25
    reject_above_px: float = 1.0
    debug_dir: Optional[str] = None
    jobs: int = 2
    demosaic: str = DEFAULT_DEMOSAIC
    hash_sources: bool = True


def _expand_inputs(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        if not hits and os.path.exists(os.path.expanduser(pat)):
            hits = [os.path.expanduser(pat)]
        paths.extend(hits)
    seen = set()
    unique = [p for p in paths if not (p in seen or seen.add(p))]
    return unique


def _pick_holdout(names: Sequence[str], opts: RunOptions) -> List[str]:
    if opts.holdout_files:
        wanted = {os.path.basename(f) for f in opts.holdout_files}
        missing = wanted - set(names)
        if missing:
            raise ValueError(f"--holdout-files not among the detected frames: {sorted(missing)}")
        return [n for n in names if n in wanted]
    n = opts.holdout
    if n is None:
        n = 3 if len(names) >= 10 else 0
    if n <= 0:
        return []
    if len(names) - n < 5:
        raise ValueError(
            f"--holdout {n} leaves {len(names) - n} training frames; need at least 5"
        )
    # Evenly spaced through the (sorted) set: deterministic, and spread over
    # whatever order the boards were shot in.
    idx = np.linspace(0, len(names) - 1, n + 2)[1:-1].round().astype(int)
    return [names[i] for i in sorted(set(idx.tolist()))]


def _fit_with_rejection(
    object_points: np.ndarray, frames: List[FrameResult], image_size: Tuple[int, int],
    model: str, reject_above_px: float, log,
) -> Tuple[CalibrationResult, List[FrameResult], Dict[str, float]]:
    """Calibrate, drop frames whose RMS is above the threshold or > 3x the
    median, and calibrate once more on what is left."""
    result = calibrate_from_points(object_points, [f.corners_g for f in frames], image_size, model)
    median = float(np.median(result.per_image_rms))
    limit = min(reject_above_px, 3.0 * median) if median > 0 else reject_above_px
    rejected = {
        f.name: rms for f, rms in zip(frames, result.per_image_rms) if rms > limit
    }
    if rejected and len(frames) - len(rejected) >= 5:
        log(f"  rejecting {len(rejected)} frame(s) above {limit:.2f} px "
            f"(threshold {reject_above_px}, 3x median {3 * median:.2f}): "
            + ", ".join(f"{n} ({v:.2f})" for n, v in sorted(rejected.items())))
        frames = [f for f in frames if f.name not in rejected]
        result = calibrate_from_points(object_points, [f.corners_g for f in frames], image_size, model)
    elif rejected:
        log(f"  {len(rejected)} frame(s) exceed {limit:.2f} px but too few frames remain to reject them")
        rejected = {}
    return result, frames, rejected


def run(opts: RunOptions, log=print) -> Dict[str, Any]:
    t0 = time.perf_counter()
    paths = _expand_inputs(opts.inputs)
    if len(paths) < 5:
        raise ValueError(f"need at least 5 RAW frames, --input matched {len(paths)}")
    if opts.debug_dir:
        os.makedirs(opts.debug_dir, exist_ok=True)
    object_points = board_object_points(opts.pattern, opts.square_mm)

    # 1-3. Develop + detect (+ CA corners), in parallel.
    log(f"developing and detecting {len(paths)} frames "
        f"(pattern {opts.pattern[0]}x{opts.pattern[1]}, detect scale {opts.detect_scale}, "
        f"{opts.jobs} job(s))...")
    with ThreadPoolExecutor(max_workers=max(1, opts.jobs)) as pool:
        frames = list(pool.map(
            lambda p: process_frame(
                p, opts.pattern, detect_scale=opts.detect_scale, want_ca=opts.fit_ca,
                demosaic=opts.demosaic, keep_debug=bool(opts.debug_dir),
            ),
            paths,
        ))
    failures = [f for f in frames if not f.detected]
    frames = [f for f in frames if f.detected]
    for f in failures:
        log(f"  skip {f.name}: {f.error}")
    log(f"  {len(frames)} frames with a full board, {len(failures)} skipped, "
        f"{sum(f.seconds for f in frames + failures):.0f}s of frame work")
    if len(frames) < 5:
        raise ValueError(f"only {len(frames)} frames with a detected board; need at least 5")
    sizes = {f.image_size for f in frames}
    if len(sizes) != 1:
        raise ValueError(f"frames differ in developed size: {sorted(sizes)}; calibrate one camera/setting at a time")
    image_size = frames[0].image_size
    log(f"  sensor frame {image_size[0]}x{image_size[1]} ({dict(SENSOR_FRAME_KWARGS)})")
    if opts.debug_dir:
        for f in frames:
            _write_corner_overlay(f, opts.pattern, opts.detect_scale, opts.debug_dir)

    # Holdout split.
    names = [f.name for f in frames]
    holdout_names = _pick_holdout(names, opts)
    holdout = [f for f in frames if f.name in holdout_names]
    training = [f for f in frames if f.name not in holdout_names]

    # 4-5. Calibrate (with one rejection pass), holdout, model selection.
    def _calibrate(model: str):
        res, used, rejected = _fit_with_rejection(
            object_points, training, image_size, model, opts.reject_above_px, log,
        )
        hold = [
            reprojection_rms(object_points, f.corners_g, res.camera_matrix, res.dist_coeffs)
            for f in holdout
        ]
        hold_rms = float(np.sqrt(np.mean([h[0] ** 2 for h in hold]))) if hold else None
        return res, used, rejected, hold, hold_rms

    candidates = ["standard", "rational"] if opts.model == "auto" else [opts.model]
    fits = {}
    for model in candidates:
        log(f"calibrating ({model} model)...")
        fits[model] = _calibrate(model)
        res, used, _, _, hold_rms = fits[model]
        log(f"  RMS {res.rms:.3f} px over {len(used)} frames"
            + (f", holdout {hold_rms:.3f} px over {len(holdout)}" if hold_rms is not None else ""))
    chosen = candidates[0]
    selection_note = None
    if opts.model == "auto":
        std_hold = fits["standard"][4]
        rat_hold = fits["rational"][4]
        if std_hold is None or rat_hold is None:
            selection_note = "auto: no holdout frames, kept the standard model (rational always fits training better)"
        elif rat_hold < 0.9 * std_hold:
            chosen = "rational"
            selection_note = f"auto: rational holdout {rat_hold:.3f} px beat standard {std_hold:.3f} px by >10 %"
        else:
            selection_note = f"auto: standard kept (holdout {std_hold:.3f} vs rational {rat_hold:.3f} px, within noise)"
        log(f"  {selection_note}")
    result, used, rejected, hold, hold_rms = fits[chosen]

    grid = coverage_grid([f.corners_g for f in used], image_size)
    log("  coverage grid (frames with corners per cell, rows top->bottom):")
    for row in grid:
        log("    " + " ".join(f"{n:3d}" for n in row))
    thin = sum(1 for row in grid for n in row if n < MIN_COVERAGE_PER_CELL)
    if thin:
        log(f"  {thin} cell(s) under {MIN_COVERAGE_PER_CELL} frames - thin edge/corner coverage is the "
            f"most common cause of a bad calibration; shoot boards there")

    # 6. Lateral CA.
    lateral_ca: Optional[LateralCA] = None
    ca_block: Optional[Dict[str, Any]] = None
    cx, cy = float(result.camera_matrix[0, 2]), float(result.camera_matrix[1, 2])
    r_norm = float(np.hypot(*image_size) / 2.0)
    if opts.fit_ca:
        g_all = np.vstack([f.corners_g for f in used])
        stats: Dict[str, Dict[str, float]] = {}
        coeffs: Dict[str, Tuple[float, float, float]] = {}
        for ch, attr in (("R", "corners_r"), ("B", "corners_b")):
            c_all = np.vstack([getattr(f, attr) for f in used])
            coeffs[ch], rms = fit_lateral_ca(g_all, c_all, (cx, cy), r_norm)
            stats[ch] = {"rms_residual_px": rms}
        lateral_ca = LateralCA(r_norm=r_norm, coeffs_r=coeffs["R"], coeffs_b=coeffs["B"], stats=stats)
        for ch in ("R", "B"):
            stats[ch]["max_shift_px_at_corner"] = lateral_ca.max_shift_px(ch, image_size, (cx, cy))
            log(f"  lateral CA {ch}: a=({', '.join(f'{c:+.3e}' for c in lateral_ca.coeffs(ch))}) "
                f"residual {stats[ch]['rms_residual_px']:.3f} px RMS, "
                f"corner shift {stats[ch]['max_shift_px_at_corner']:.2f} px")
        ca_block = lateral_ca.to_dict()

    # 7. Straight-line test.
    quality: Dict[str, Any] = {
        "n_frames_detected": len(frames),
        "n_frames_failed_detection": len(failures),
        "n_frames_used": len(used),
        "n_frames_rejected": len(rejected),
        "n_holdout": len(holdout),
        "rms_reprojection_px": result.rms,
        "holdout_rms_px": hold_rms,
        "per_image_rms_px": {f.name: rms for f, rms in zip(used, result.per_image_rms)},
        "holdout_per_image_rms_px": {f.name: h[0] for f, h in zip(holdout, hold)},
        "rejected_frames_rms_px": rejected,
        "detection_failures": [f.name for f in failures],
        f"coverage_grid_{COVERAGE_COLS}x{COVERAGE_ROWS}": grid,
    }
    if selection_note:
        quality["model_selection"] = selection_note
    meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": f"image-processing calibrate_distortion {TOOL_VERSION}",
        "camera": {},
        "develop": dict(sensor_frame_record(), demosaic=opts.demosaic),
        "quality": quality,
        "sources": [f.name for f in used],
        "holdout_sources": [f.name for f in holdout],
    }
    calib = result.to_calibration(image_size, lateral_ca, meta)

    sl_frame: Optional[FrameResult] = None
    if opts.straight_edge_frame:
        wanted = os.path.basename(opts.straight_edge_frame)
        sl_frame = next((f for f in frames if f.name == wanted), None)
        if sl_frame is None:
            log(f"  --straight-edge-frame {wanted} has no detected board; using the flattest holdout instead")
    if sl_frame is None and holdout:
        tilts = [board_tilt_deg(h[1]) for h in hold]
        sl_frame = holdout[int(np.argmin(tilts))]
    if sl_frame is None:
        # No holdout: use the flattest training board.
        tilts = [board_tilt_deg(r) for r in result.rvecs]
        sl_frame = used[int(np.argmin(tilts))]
    quality["straight_line_test"] = straight_line_report(sl_frame.corners_g, opts.pattern, calib, sl_frame.name)
    sl = quality["straight_line_test"]
    log(f"  straight-line test on {sl['frame']}: full frame bow "
        f"{sl['max_bow_px_before']:.2f} -> {sl['max_bow_px_after']:.2f} px; "
        f"center 50% crop {sl['center_crop']['max_bow_px_before']:.2f} -> "
        f"{sl['center_crop']['max_bow_px_after']:.2f} px")
    quality["invalid_output_fraction"] = invalid_output_fraction(calib)
    if quality["invalid_output_fraction"] > 0:
        log(f"  note: {quality['invalid_output_fraction'] * 100:.2f}% of output pixels sample "
            f"outside the source frame (black after remap) - pincushion at the edge")

    # Camera metadata for the match check.
    exif = read_raw_exif(used[0].path)
    meta["camera"] = {
        "body": exif["body"], "serial": exif["serial"], "lens": exif["lens"],
        "zoom_position": opts.zoom_position, "focus_position": opts.focus_position,
        "units": "sdk_raw",
        "aperture": exif["aperture"], "shutter": exif["shutter"], "iso": exif["iso"],
    }
    if opts.zoom_position is None:
        log("  note: no --zoom-position given; the module's match check cannot enforce the zoom setpoint")
    if opts.hash_sources:
        log("  hashing source files...")
        meta["sha256_of_sources"] = sha256_of_sources([f.path for f in used])

    # 8. Write + debug.
    calib.save(opts.out)
    log(f"wrote {opts.out}")
    if opts.debug_dir:
        _write_quiver(used, object_points, result, image_size, opts.debug_dir)
        _write_coverage(grid, opts.debug_dir)
        try:
            _write_before_after(sl_frame.path, calib, opts.demosaic, opts.debug_dir)
        except Exception as exc:  # noqa: BLE001 - debug output must not fail the run
            log(f"  before/after image failed: {exc}")
        log(f"debug output in {opts.debug_dir}")

    checks = evaluate_acceptance(quality, ca_block)
    log("")
    log(f"summary ({time.perf_counter() - t0:.0f}s): {calib.model}, {image_size[0]}x{image_size[1]}, "
        f"{len(used)} frames used, {len(rejected)} rejected, {len(holdout)} holdout, "
        f"{len(failures)} undetected")
    for check in checks:
        log(f"  [{check.label}] {check.name}: {check.detail}")
    overall = "PASS" if all(c.ok for c in checks if c.ok is not None) else "WARN"
    log(f"overall: {overall}")
    return {"calibration": calib, "checks": checks, "overall": overall, "path": opts.out}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calibrate_distortion",
        description="Checkerboard distortion + lateral CA calibration for the RAW develop pipeline.",
    )
    p.add_argument("--input", required=True, nargs="+",
                   help="RAW file globs (quote them), e.g. 'calib/2026-09-10-16mm-f8/*.ARW'")
    p.add_argument("--pattern", required=True, help="inner corners as COLSxROWS, e.g. 10x7")
    p.add_argument("--square-mm", required=True, type=float, help="measured printed square size in mm")
    p.add_argument("--out", required=True, help="calibration JSON to write")
    p.add_argument("--zoom-position", type=int, default=None, help="sony-remote zoom_position (raw SDK units)")
    p.add_argument("--focus-position", type=int, default=None, help="sony-remote focus_position (raw SDK units)")
    p.add_argument("--model", choices=("standard", "rational", "auto"), default="standard",
                   help="standard = 5 coefficients (default); rational = 8; auto = pick by holdout RMS")
    p.add_argument("--no-ca", action="store_true", help="skip the lateral CA fit")
    p.add_argument("--holdout", type=int, default=None,
                   help="number of frames to hold out for validation (default 3 when >= 10 frames)")
    p.add_argument("--holdout-files", nargs="*", default=[], help="specific frames to hold out")
    p.add_argument("--straight-edge-frame", default=None,
                   help="frame for the straight-line report (default: flattest holdout board)")
    p.add_argument("--detect-scale", type=float, default=0.25, help="downscale for the coarse detection")
    p.add_argument("--reject-above-px", type=float, default=1.0, help="per-image RMS rejection threshold")
    p.add_argument("--debug-dir", default=None, help="write overlays, quiver, coverage, before/after here")
    p.add_argument("--jobs", type=int, default=2, help="parallel frame develops (each ~1 GB at 61 MP)")
    p.add_argument("--demosaic", default=DEFAULT_DEMOSAIC, help="rawpy demosaic (value-only; no geometry effect)")
    p.add_argument("--no-hash", action="store_true", help="skip hashing the source ARWs (faster)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    opts = RunOptions(
        inputs=args.input, pattern=parse_pattern(args.pattern), square_mm=args.square_mm,
        out=args.out, zoom_position=args.zoom_position, focus_position=args.focus_position,
        model=args.model, fit_ca=not args.no_ca, holdout=args.holdout,
        holdout_files=args.holdout_files, straight_edge_frame=args.straight_edge_frame,
        detect_scale=args.detect_scale, reject_above_px=args.reject_above_px,
        debug_dir=args.debug_dir, jobs=args.jobs, demosaic=args.demosaic,
        hash_sources=not args.no_hash,
    )
    try:
        outcome = run(opts)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if outcome["overall"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
