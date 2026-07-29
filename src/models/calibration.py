"""
calibration.py
--------------
ColorChecker Classic calibration math: the reference patch values, automatic
chart detection (cv2.mcc), patch sampling, CCM fitting, and the
:class:`ColorCorrector` that applies a fitted matrix to images.

Everything here is pure numpy/OpenCV with no Viam dependency, so it can be
unit-tested or driven from a script (see debug_calibration.py and
compare_renditions.py at the repo root). The Viam component in
color_correction.py wires these pieces to the `calibrate_color` DoCommand and
the streaming path.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from models.image_io import linear_to_srgb, srgb_to_linear

# OpenCV's ColorChecker detector (cv2.mcc) lives in opencv-contrib; import lazily
# so the module still loads (and the streaming/develop paths work) on a host
# without it - calibration raises a clean, actionable error at point of use.
try:
    import cv2  # type: ignore

    _CV2_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the host
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# ColorChecker Classic reference values (24 patches)
# ---------------------------------------------------------------------------
# Rather than hard-code an sRGB table (the previous one matched the pre-2014
# colorants and was off by up to ~18/255 on the saturated patches - worst on
# blue), we keep the *authoritative* CIE xyY data and derive sRGB from it, so
# the reference is traceable to source and unambiguous.
#
# These are the X-Rite published values for the "After November 2014"
# formulation - the colorants in every ColorChecker Classic made since, so the
# right ones for a current chart. Values via the colour-science dataset
# (ColorChecker24 - After November 2014), CIE 1931 2-degree observer, ICC D50.
# Order: dark skin -> black, matching the patch layout (text at top).
_COLORCHECKER_XYY_D50 = np.array([
    [0.4325, 0.3788, 0.1034],  # 1  Dark Skin
    [0.4191, 0.3748, 0.3525],  # 2  Light Skin
    [0.2761, 0.3004, 0.1847],  # 3  Blue Sky
    [0.3700, 0.4501, 0.1335],  # 4  Foliage
    [0.3020, 0.2877, 0.2324],  # 5  Blue Flower
    [0.2856, 0.3910, 0.4174],  # 6  Bluish Green
    [0.5291, 0.4075, 0.3117],  # 7  Orange
    [0.2339, 0.2155, 0.1140],  # 8  Purplish Blue
    [0.5008, 0.3293, 0.1979],  # 9  Moderate Red
    [0.3326, 0.2556, 0.0644],  # 10 Purple
    [0.3989, 0.4998, 0.4435],  # 11 Yellow Green
    [0.4962, 0.4428, 0.4358],  # 12 Orange Yellow
    [0.2040, 0.1696, 0.0579],  # 13 Blue
    [0.3270, 0.5033, 0.2307],  # 14 Green
    [0.5709, 0.3298, 0.1268],  # 15 Red
    [0.4694, 0.4732, 0.6081],  # 16 Yellow
    [0.4177, 0.2704, 0.2007],  # 17 Magenta
    [0.2151, 0.3037, 0.1903],  # 18 Cyan
    [0.3488, 0.3628, 0.9129],  # 19 White 9.5
    [0.3451, 0.3596, 0.5885],  # 20 Neutral 8
    [0.3446, 0.3590, 0.3595],  # 21 Neutral 6.5
    [0.3438, 0.3589, 0.1912],  # 22 Neutral 5
    [0.3423, 0.3576, 0.0893],  # 23 Neutral 3.5
    [0.3439, 0.3565, 0.0320],  # 24 Black 2
], dtype=np.float64)

# Bradford chromatic adaptation D50 -> D65, and CIE XYZ (D65) -> linear sRGB
# (Lindbloom / sRGB spec). The chart data is D50; sRGB is a D65 space.
_BRADFORD_D50_TO_D65 = np.array([
    [0.9555766, -0.0230393, 0.0631636],
    [-0.0282895, 1.0099416, 0.0210077],
    [0.0122982, -0.0204830, 1.3299098],
])
_XYZ_D65_TO_LINEAR_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def _xyy_d50_to_linear_srgb(xyY: np.ndarray) -> np.ndarray:
    """CIE xyY (D50) -> *linear* sRGB, unclipped. Out-of-gamut patches keep
    their negative/over-range components here so gamut membership stays
    measurable; clipping happens where an sRGB rendering is actually needed."""
    x, y, big_y = xyY[:, 0], xyY[:, 1], xyY[:, 2]
    xyz = np.stack([big_y * x / y, big_y, big_y * (1.0 - x - y) / y], axis=1)
    xyz_d65 = xyz @ _BRADFORD_D50_TO_D65.T
    return xyz_d65 @ _XYZ_D65_TO_LINEAR_SRGB.T


_REFERENCE_LINEAR_UNCLIPPED = _xyy_d50_to_linear_srgb(_COLORCHECKER_XYY_D50)

# Patches whose true color lies outside sRGB (cyan, on the post-2014 chart).
# Their clipped sRGB targets are fabrications - the nearest representable
# color, not the patch's real color - so the CCM fit down-weights them rather
# than letting an impossible target pull the matrix (see CCM_FIT_WEIGHTS).
_OUT_OF_GAMUT = np.any(
    (_REFERENCE_LINEAR_UNCLIPPED < -1e-4)
    | (_REFERENCE_LINEAR_UNCLIPPED > 1.0 + 1e-4),
    axis=1,
)

# Gamma-encoded sRGB [0, 1], dark skin -> black; the CCM fit and the
# neutral-brightness readout both reference this. Out-of-gamut patches are
# clipped after the linear transform, as any sRGB rendering of the chart must.
REFERENCE_SRGB = linear_to_srgb(
    np.clip(_REFERENCE_LINEAR_UNCLIPPED, 0.0, 1.0)
).astype(np.float32)

# Per-patch weights for the CCM fit. Unweighted least squares in linear light
# over-weights bright patches: a fixed linear-RGB error on White 9.5 (Y~0.91)
# is numerically ~10x one on Black 2 (Y~0.03) yet perceptually far *smaller*,
# so an unweighted solver polishes highlights at the expense of shadows and
# dark saturated colors - where portrait/product work actually lives.
# Weighting each patch ~1/Y roughly equalises *relative* linear error across
# the tonal range (a first-order stand-in for minimising delta-E; the epsilon
# keeps the darkest patches from dominating outright). Out-of-gamut patches
# are further down-weighted because their targets are clipped approximations.
# Normalised to mean 1 so the weights read as relative emphasis.
CCM_FIT_WEIGHTS = 1.0 / (_COLORCHECKER_XYY_D50[:, 2] + 0.05)
CCM_FIT_WEIGHTS[_OUT_OF_GAMUT] *= 0.25
CCM_FIT_WEIGHTS = (CCM_FIT_WEIGHTS / CCM_FIT_WEIGHTS.mean()).astype(np.float32)


# Canonical sRGB transfer functions live in image_io so the decode/export path
# and the color math agree exactly; aliased here to keep call sites readable.
_srgb_to_linear = srgb_to_linear
_linear_to_srgb = linear_to_srgb


# ---------------------------------------------------------------------------
# CIE Lab / delta-E (the perceptual error metric for calibration reports)
# ---------------------------------------------------------------------------

# Linear sRGB -> CIE XYZ (D65), the inverse of _XYZ_D65_TO_LINEAR_SRGB
# (Lindbloom / sRGB spec).
_LINEAR_SRGB_TO_XYZ_D65 = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
# White point taken as this matrix's own image of RGB (1,1,1), so pure white
# lands on exactly L*=100, a*=b*=0 regardless of rounding in the matrix.
_XYZ_D65_WHITE = _LINEAR_SRGB_TO_XYZ_D65.sum(axis=1)


def _linear_srgb_to_lab(rgb_linear: np.ndarray) -> np.ndarray:
    """(N, 3) linear sRGB in [0, 1] -> (N, 3) CIE L*a*b* (D65 white)."""
    xyz = np.asarray(rgb_linear, dtype=np.float64) @ _LINEAR_SRGB_TO_XYZ_D65.T
    t = np.maximum(xyz / _XYZ_D65_WHITE, 0.0)
    delta = 6.0 / 29.0
    f = np.where(t > delta ** 3, np.cbrt(t), t / (3.0 * delta ** 2) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[:, 0] = 116.0 * f[:, 1] - 16.0
    lab[:, 1] = 500.0 * (f[:, 0] - f[:, 1])
    lab[:, 2] = 200.0 * (f[:, 1] - f[:, 2])
    return lab


def delta_e76(a_linear: np.ndarray, b_linear: np.ndarray) -> np.ndarray:
    """Per-row CIE delta-E*ab (1976) between two (N, 3) **linear** sRGB arrays -
    Euclidean distance in Lab, the standard perceptual yardstick (rules of
    thumb: <1 imperceptible, ~2 barely visible side by side, >5 obvious).
    Unlike a linear-RGB distance it doesn't over-weight bright patches, and the
    numbers are comparable to other tools' calibration reports."""
    return np.linalg.norm(
        _linear_srgb_to_lab(a_linear) - _linear_srgb_to_lab(b_linear), axis=1
    )


# Row-4 neutral ramp: name -> index into REFERENCE_SRGB / the 24 sampled patches.
_NEUTRAL_PATCHES = {
    "white_9_5": 18,
    "neutral_8": 19,
    "neutral_6_5": 20,
    "neutral_5": 21,
    "neutral_3_5": 22,
    "black_2": 23,
}


def _neutral_brightness_report(measured_linear: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    As-shot brightness of each neutral patch as an sRGB-encoded 0-255 value -
    the same readout a grey-card picker (e.g. Capture One's) shows. ``measured``
    is the white-balanced value straight off the sensor with no exposure
    compensation, so it responds directly to light power: adjust the flash
    until ``measured`` matches ``reference`` (at which point ``exposure_stops``
    lands near 0) instead of changing camera exposure or digital gain.
    """
    report: Dict[str, Dict[str, float]] = {}
    for name, idx in _NEUTRAL_PATCHES.items():
        measured = _linear_to_srgb(np.clip(measured_linear[idx], 0.0, 1.0))
        report[name] = {
            "measured": round(float(np.mean(measured)) * 255.0, 1),
            "reference": round(float(np.mean(REFERENCE_SRGB[idx])) * 255.0, 1),
        }
    return report


def _fit_ccm(
    measured: np.ndarray,
    reference: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Least-squares fit of a 3x3 Color Correction Matrix.

    Solves ``reference ~= measured @ CCM.T`` (each reference row = CCM @ measured_row).

    Parameters
    ----------
    measured  : (N, 3) float32, linear-light measured RGB, normalised [0, 1]
    reference : (N, 3) float32, linear-light reference RGB, normalised [0, 1]
    weights   : optional (N,) per-patch weights (e.g. ``CCM_FIT_WEIGHTS``):
                patch i's squared error counts ``weights[i]`` times in the
                loss. ``None`` fits unweighted.

    Returns
    -------
    ccm : (3, 3) float32
    """
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=np.float64)).reshape(-1, 1)
        measured = measured * w
        reference = reference * w
    solution, _, _, _ = np.linalg.lstsq(measured, reference, rcond=None)
    return solution.T  # shape (3, 3)


# ---------------------------------------------------------------------------
# Automatic ColorChecker detection (cv2.mcc)
# ---------------------------------------------------------------------------

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 chart corners as [top-left, top-right, bottom-right, bottom-left]
    *of the image frame* (x+y / x-y heuristic).

    This fixes the winding only - it says nothing about which corner sits next
    to the dark-skin patch. The calibration render is deliberately unrotated
    (``user_flip=0``), so a portrait shot puts the chart on its side;
    ``_oriented_chart_grid`` below tries all four 90-degree assignments and
    keeps the one whose colors match the reference layout.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[np.argmin(s)],  # top-left      (smallest x+y)
        pts[np.argmax(d)],  # top-right     (largest  x-y)
        pts[np.argmax(s)],  # bottom-right  (largest  x+y)
        pts[np.argmin(d)],  # bottom-left   (smallest x-y)
    ], dtype=np.float32)


def _orientation_score(measured_srgb: np.ndarray) -> float:
    """
    How well 24 sampled patch colors match REFERENCE_SRGB's layout: per-channel
    Pearson correlation, summed over R/G/B (max 3.0). Standardizing each channel
    makes the score invariant to exposure and to per-channel gain - so an
    uncorrected white-balance cast can't disguise the right orientation.
    Wrong rotations land near 0.
    """
    score = 0.0
    for c in range(3):
        m, r = measured_srgb[:, c], REFERENCE_SRGB[:, c]
        ms, rs = float(m.std()), float(r.std())
        if ms < 1e-6 or rs < 1e-6:
            continue
        score += float((((m - m.mean()) / ms) * ((r - r.mean()) / rs)).mean())
    return score


# Below this, no candidate rotation matched the reference layout - the detector
# most likely latched onto something that isn't a ColorChecker Classic.
_MIN_ORIENTATION_SCORE = 0.75


def _oriented_chart_grid(
    img_rgb: np.ndarray, box: np.ndarray, *, rows: int = 4, cols: int = 6
) -> Optional[Dict[str, Any]]:
    """
    Map a detected chart quad to the 24 patch centres in REFERENCE_SRGB order,
    robust to the chart sitting at any 90-degree rotation in the frame.

    Tries the four cyclic corner assignments, samples the patch colors each
    would imply, and keeps the orientation that correlates with the reference
    layout. Returns ``None`` if none does (false-positive detection).

    Returns a dict with ``centers`` (24, 2), ``neutral_boxes_norm`` (axis-
    aligned inner boxes over Neutral 8 / 6.5 for raw WB sampling),
    ``suggested_radius`` (patch-size-relative sampling half-width), and
    ``orientation_score``.
    """
    corners = _order_corners(box)
    h, w = img_rgb.shape[:2]
    img_f = img_rgb.astype(np.float32) / 255.0

    def make_grid(c0, c1, c2, c3):
        # c0->c1 spans the `cols` axis, c0->c3 the `rows` axis.
        def grid_point(u: float, v: float) -> np.ndarray:
            top = c0 + (c1 - c0) * u
            bot = c3 + (c2 - c3) * u
            return top + (bot - top) * v
        centers = np.zeros((rows * cols, 2), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                centers[r * cols + c] = grid_point((c + 0.5) / cols, (r + 0.5) / rows)
        return centers, grid_point

    def sample(centers: np.ndarray, radius: int) -> np.ndarray:
        out = np.zeros((len(centers), 3), dtype=np.float32)
        for i, (x, y) in enumerate(centers):
            xi, yi = int(round(float(x))), int(round(float(y)))
            x0, y0 = max(0, xi - radius), max(0, yi - radius)
            patch = img_f[y0:yi + radius, x0:xi + radius].reshape(-1, 3)
            if patch.size:
                out[i] = np.median(patch, axis=0)
        return out

    best_score, best = -np.inf, None
    cycle = list(corners)
    for _ in range(4):
        c0, c1, c2, c3 = cycle
        centers, grid_point = make_grid(c0, c1, c2, c3)
        patch_px = min(
            float(np.linalg.norm(c1 - c0)) / cols,
            float(np.linalg.norm(c3 - c0)) / rows,
        )
        radius = max(2, int(0.15 * patch_px))
        score = _orientation_score(sample(centers, radius))
        if score > best_score:
            best_score, best = score, (centers, grid_point, radius)
        cycle = cycle[1:] + cycle[:1]

    if best is None or best_score < _MIN_ORIENTATION_SCORE:
        return None
    centers, grid_point, radius = best

    # Neutral 8 (#19) and Neutral 6.5 (#20): mid-grey patches for raw white
    # balance - not the white patch (clips) or black (noisy). Inner ~40% of
    # each, as an axis-aligned box built from the patch's own step vectors so
    # any chart rotation works.
    neutral_boxes_norm: List[Tuple[float, float, float, float]] = []
    for idx in ((rows - 1) * cols + 1, (rows - 1) * cols + 2):
        r, c = divmod(idx, cols)
        u, v = (c + 0.5) / cols, (r + 0.5) / rows
        center = grid_point(u, v)
        half_u = (grid_point(u + 0.5 / cols, v) - grid_point(u - 0.5 / cols, v)) * 0.2
        half_v = (grid_point(u, v + 0.5 / rows) - grid_point(u, v - 0.5 / rows)) * 0.2
        pts = np.array([
            center + half_u + half_v, center + half_u - half_v,
            center - half_u + half_v, center - half_u - half_v,
        ])
        neutral_boxes_norm.append((
            float(pts[:, 0].min()) / w, float(pts[:, 1].min()) / h,
            float(pts[:, 0].max()) / w, float(pts[:, 1].max()) / h,
        ))

    return {
        "centers": centers,
        "neutral_boxes_norm": neutral_boxes_norm,
        "suggested_radius": radius,
        "orientation_score": best_score,
    }


def detect_colorchecker(
    img_rgb: np.ndarray, *, rows: int = 4, cols: int = 6
) -> Optional[Dict[str, Any]]:
    """
    Auto-detect a ColorChecker Classic anywhere in ``img_rgb`` (uint8 RGB).

    Returns ``None`` if no chart is found, else a dict with:
      ``centers``            (24, 2) float patch centres in pixel coords, in
                             REFERENCE_SRGB order (dark skin -> black).
      ``neutral_boxes_norm`` (x0, y0, x1, y1) boxes (fractions of W/H) over the
                             Neutral 8 and Neutral 6.5 patches, for raw white
                             balance sampling.
      ``suggested_radius``   patch-size-relative sampling half-width (px).
      ``orientation_score``  reference-layout correlation of the chosen
                             rotation (max 3.0).

    Patch centres come from bilinearly interpolating the detected chart box
    over a rows x cols grid, after resolving which of the four 90-degree
    rotations the chart sits at (see ``_oriented_chart_grid``) - so a portrait
    shot, whose calibration render is deliberately unrotated, still maps
    correctly. The centres are geometry, valid on any co-registered render
    (the linear CCM render, the raw CFA).
    """
    if cv2 is None:
        raise RuntimeError(
            f"ColorChecker auto-detection needs OpenCV, which isn't available "
            f"({_CV2_IMPORT_ERROR}); install `opencv-contrib-python-headless`"
        )
    if not hasattr(cv2, "mcc"):
        raise RuntimeError(
            "this OpenCV build has no `mcc` module; the ColorChecker detector "
            "ships in opencv-contrib - install `opencv-contrib-python-headless` "
            "(replacing `opencv-python-headless`), or pass explicit `patch_centers`"
        )

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    detector = cv2.mcc.CCheckerDetector_create()
    # OpenCV 5 dropped the chartType argument from process(); its second
    # parameter is now `nc` (max charts to find). Passing 4.x's MCC24 (== 0)
    # there means "find zero charts", which silently detects nothing - so the
    # call has to branch on the major version.
    if int(cv2.__version__.split(".")[0]) >= 5:
        processed = detector.process(bgr, 1)
    else:
        processed = detector.process(bgr, cv2.mcc.MCC24)
    if not processed:
        return None
    checkers = detector.getListColorChecker()
    if not checkers:
        return None

    box = np.asarray(checkers[0].getBox(), dtype=np.float32).reshape(-1, 2)
    if box.shape[0] != 4:
        return None
    return _oriented_chart_grid(img_rgb, box, rows=rows, cols=cols)


class PatchSampler:
    """Locate and sample the 24 ColorChecker patches from an RGB image."""

    @staticmethod
    def auto_sample(img_rgb: np.ndarray, grid: Tuple[int, int] = (4, 6)) -> np.ndarray:
        """
        Divide the image into a (rows x cols) grid and sample the centre of each
        cell. Works well only when the ColorChecker fills the frame and is
        reasonably upright; for anything else pass explicit ``patch_centers``.

        Returns (N, 3) float32 linear-light RGB.
        """
        rows, cols = grid
        h, w = img_rgb.shape[:2]
        # Shrink sampling box slightly to avoid dark borders
        margin_y = int(h * 0.05)
        margin_x = int(w * 0.05)
        cell_h = (h - 2 * margin_y) // rows
        cell_w = (w - 2 * margin_x) // cols

        samples = []
        for r in range(rows):
            for c in range(cols):
                cy = margin_y + r * cell_h + cell_h // 2
                cx = margin_x + c * cell_w + cell_w // 2
                # Sample a small region and median to reduce noise
                patch = img_rgb[cy - 10:cy + 10, cx - 10:cx + 10].reshape(-1, 3)
                samples.append(np.median(patch, axis=0))

        measured_srgb = np.array(samples, dtype=np.float32) / 255.0
        return _srgb_to_linear(measured_srgb)

    @staticmethod
    def sample_at_centers(
        img_rgb: np.ndarray,
        centers: Sequence[Tuple[int, int]],
        radius: int = 10,
    ) -> np.ndarray:
        """
        Sample patches at explicit pixel (x, y) centres - the reliable path when
        the chart does not fill the frame.

        Parameters
        ----------
        img_rgb : (H, W, 3) uint8
        centers : 24 (x, y) pixel coords, in the same order as REFERENCE_SRGB
        radius  : half-side of the square sampling region

        Returns (24, 3) float32 linear-light RGB.
        """
        h, w = img_rgb.shape[:2]
        samples = []
        for x, y in centers:
            x, y = int(x), int(y)
            # Clamp the region to the frame: a centre closer to the edge than
            # `radius` would otherwise produce negative slice indices, which
            # numpy wraps around to the far side of the image.
            x0, y0 = max(0, x - radius), max(0, y - radius)
            x1, y1 = min(w, x + radius), min(h, y + radius)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(
                    f"patch centre ({x}, {y}) lies outside the {w}x{h} image"
                )
            patch = img_rgb[y0:y1, x0:x1].reshape(-1, 3)
            samples.append(np.median(patch, axis=0))
        measured_srgb = np.array(samples, dtype=np.float32) / 255.0
        return _srgb_to_linear(measured_srgb)

    @staticmethod
    def sample_linear_at_centers(
        img_linear: np.ndarray,
        centers: Sequence[Tuple[float, float]],
        radius: int = 10,
    ) -> np.ndarray:
        """
        Sample patches from an already-**linear-light** float RGB image at the
        given (x, y) centres - the precise path when calibrating from a 16-bit
        linear RAW render (no sRGB round-trip). Returns (N, 3) float32 linear RGB.
        """
        h, w = img_linear.shape[:2]
        samples = []
        for x, y in centers:
            x, y = int(round(float(x))), int(round(float(y)))
            x0, y0 = max(0, x - radius), max(0, y - radius)
            x1, y1 = min(w, x + radius), min(h, y + radius)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(
                    f"patch centre ({x}, {y}) lies outside the {w}x{h} image"
                )
            patch = img_linear[y0:y1, x0:x1].reshape(-1, 3)
            samples.append(np.median(patch, axis=0))
        return np.asarray(samples, dtype=np.float32)


class ColorCorrector:
    """
    Holds a fitted 3x3 Color Correction Matrix and applies it to RGB images.

    All image math lives here, decoupled from Viam, so the same logic can be
    unit-tested or driven from a script. Operates on numpy uint8 RGB arrays;
    callers convert to/from PIL or base64 at the edges.
    """

    def __init__(self, ccm: np.ndarray):
        ccm = np.asarray(ccm, dtype=np.float32)
        if ccm.shape != (3, 3):
            raise ValueError(f"CCM must be a 3x3 matrix, got shape {ccm.shape}")
        self.ccm = ccm

    @classmethod
    def identity(cls) -> "ColorCorrector":
        """A no-op corrector (passes colors through unchanged)."""
        return cls(np.eye(3, dtype=np.float32))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def calibrate_from_rgb(
        cls,
        img_rgb: np.ndarray,
        patch_centers: Optional[Sequence[Tuple[int, int]]] = None,
        radius: int = 10,
    ) -> "ColorCorrector":
        """
        Fit a CCM from an RGB image of the ColorChecker Classic.

        If ``patch_centers`` is given (24 (x, y) coords in REFERENCE_SRGB order)
        the patches are sampled there; otherwise auto-sampling assumes the chart
        fills the frame.
        """
        if patch_centers is not None:
            measured_linear = PatchSampler.sample_at_centers(img_rgb, patch_centers, radius)
        else:
            measured_linear = PatchSampler.auto_sample(img_rgb)

        reference_linear = _srgb_to_linear(REFERENCE_SRGB)
        ccm = _fit_ccm(measured_linear, reference_linear, weights=CCM_FIT_WEIGHTS)
        return cls(ccm)

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.ccm, np.eye(3, dtype=np.float32)))

    def apply_to_linear(self, img_linear: np.ndarray) -> np.ndarray:
        """
        Apply the CCM to an (H, W, 3) **linear-light** float RGB array, returning
        a linear float array. This is the high-precision path: callers working
        from 16-bit RAW stay in linear float end to end and only encode the
        output transfer curve at export. Identity is a no-op passthrough.
        """
        if self.is_identity:
            return img_linear
        h, w = img_linear.shape[:2]
        corrected = (img_linear.reshape(-1, 3) @ self.ccm.T).reshape(h, w, 3)
        return corrected.astype(np.float32)

    def apply_to_rgb(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Apply the CCM to an (H, W, 3) uint8 sRGB array, returning uint8 sRGB.

        Convenience wrapper for the 8-bit streaming path (proxied JPEG/PNG
        frames): sRGB -> linear -> CCM -> sRGB. A no-op (identity) matrix returns
        the input untouched, avoiding gamma round-trip rounding.
        """
        if self.is_identity:
            return img_rgb
        img_linear = _srgb_to_linear(img_rgb.astype(np.float32) / 255.0)
        corrected_srgb = _linear_to_srgb(self.apply_to_linear(img_linear))
        return np.rint(corrected_srgb * 255.0).clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def delta_e_report(
        self,
        img_rgb: np.ndarray,
        patch_centers: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> dict:
        """
        Per-patch color error before vs. after correction, as a quick measure of
        calibration quality. Reported as CIE delta-E*ab 1976 (see ``delta_e76``);
        mean below ~3 is a solid calibration, individual patches above ~5 are
        visibly off.
        """
        if patch_centers is not None:
            measured = PatchSampler.sample_at_centers(img_rgb, patch_centers)
        else:
            measured = PatchSampler.auto_sample(img_rgb)
        ref_linear = _srgb_to_linear(REFERENCE_SRGB)
        corrected_linear = (measured @ self.ccm.T).clip(0, 1)

        before = delta_e76(measured.clip(0, 1), ref_linear)
        after = delta_e76(corrected_linear, ref_linear)
        return {
            "before": {"mean": float(before.mean()), "max": float(before.max())},
            "after": {"mean": float(after.mean()), "max": float(after.max())},
        }
