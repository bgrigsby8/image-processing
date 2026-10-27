"""
Focus quality from a photographed checkerboard: ISO 12233 slanted-edge
(e-SFR) MTF50 on every square edge of the board.

Why the checkerboard. It is already on the rig, and every edge between two
inner corners is a straight, high-contrast step. Turn the board ~5 degrees on
its mount and each of those edges is a *slanted* edge - the one thing e-SFR
needs. At the outfits position (~2.6 m, 16 mm) a 30 mm square is ~50 px, so
the 11x8 board gives ~150 edges; the median of their MTF50s is stable enough
to rank frames in a focus sweep.

Per edge (``edge_mtf``):

1. Take the pixels around the segment between two neighbouring corners - the
   middle half of it, so the perpendicular edges at the corners stay out -
   with no resampling (interpolation would itself blur the edge).
2. Re-fit the edge line from the data: the 50 % crossing of each 1 px strip
   along the edge, least-squares line through them. The corners only seed it.
3. Project every pixel onto its signed distance from that line and average in
   1/4 px bins: the oversampled edge spread function (ESF). The slant is what
   fills the bins - an axis-aligned edge (or a low-order rational slope like
   45 degrees) leaves most of them empty, and such edges are rejected.
4. Differentiate to the line spread function (LSF), Hamming-window it about
   its centroid, FFT, normalise to DC, and divide out the derivative's own
   response. MTF50 is where that curve first falls through 0.5.

MTF50 is in cycles/pixel (0.5 = Nyquist); ``FrameSharpness`` also reports it
as line widths per picture height (LW/PH = 2 x cy/px x height), the unit
Imatest and most lens reviews use. Measure on *linear* RAW data with no tone
curve or sharpening - sharpening raises MTF without the lens doing anything.

Absolute values depend on the demosaic and the print, so compare frames from
one sweep (same board, same develop) rather than across setups. That relative
comparison is what a focus sweep needs.

``tenengrad`` is the no-reference fallback for frames with no board in them
(a product shot): only meaningful between frames of the *same* scene.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ESF oversampling: bins per pixel (ISO 12233 uses 4).
OVERSAMPLE = 4
# Fraction of the corner-to-corner segment kept, centred - stays clear of the
# perpendicular edges meeting at each corner.
ALONG_FRACTION = 0.5
# Half-width of the region across the edge, as a fraction of the square size.
ACROSS_FRACTION = 0.4
# An edge closer than this to the pixel axes cannot fill the ESF bins.
MIN_SLANT_DEG = 2.0
# Reject an edge whose ESF has more empty bins than this (rational slopes).
MAX_EMPTY_BIN_FRACTION = 0.1
# Reject an edge when this much of its region is at or above clip level.
CLIP_LEVEL = 0.98
MAX_CLIPPED_FRACTION = 0.01
# Light/dark ratio below which the "edge" is noise (too far out of focus to
# measure, or not an edge at all).
MIN_EDGE_CONTRAST = 1.5
# Fewer usable edges than this and the frame's median is not trusted.
MIN_EDGES = 5
# Detection scales tried in order (a small board at 2.6 m can be too small
# for the 1/4-scale pass that suits a frame-filling board).
DETECT_SCALES = (0.25, 0.5, 1.0)


@dataclass
class EdgeMTF:
    """One square edge: where it is, and its MTF50 (None when rejected)."""
    kind: str                       # "vertical" or "horizontal" (edge direction)
    center: Tuple[float, float]     # (x, y) image pixels
    angle_deg: float                # edge slant off the nearest pixel axis
    mtf50: Optional[float] = None   # cycles/pixel
    error: Optional[str] = None


@dataclass
class FrameSharpness:
    path: str
    image_size: Tuple[int, int] = (0, 0)   # (w, h)
    mtf50: Optional[float] = None          # median over usable edges, cy/px
    mtf50_lw_ph: Optional[float] = None    # same, line widths / picture height
    mtf50_iqr: Optional[float] = None      # spread of the edge MTF50s, cy/px
    mtf50_vertical: Optional[float] = None
    mtf50_horizontal: Optional[float] = None
    edges_used: int = 0
    edges_total: int = 0
    corners_from_hint: bool = False
    error: Optional[str] = None
    seconds: float = 0.0
    edges: List[EdgeMTF] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """JSON-ready, without the per-edge list."""
        keys = ("path", "image_size", "mtf50", "mtf50_lw_ph", "mtf50_iqr",
                "mtf50_vertical", "mtf50_horizontal", "edges_used", "edges_total",
                "corners_from_hint", "error", "seconds")
        out = {k: getattr(self, k) for k in keys}
        out["image_size"] = list(self.image_size)
        return out


# ---------------------------------------------------------------------------
# Slanted-edge MTF
# ---------------------------------------------------------------------------

def _slant_off_axis_deg(direction: np.ndarray) -> float:
    angle = math.degrees(math.atan2(abs(direction[1]), abs(direction[0])))
    return min(angle, 90.0 - angle)


def _strip_crossings(
    s: np.ndarray, d: np.ndarray, v: np.ndarray, lo: float, hi: float
) -> Tuple[np.ndarray, np.ndarray]:
    """The 50 % crossing (in d) of each 1 px strip along the edge."""
    mid = 0.5 * (lo + hi)
    rising = hi > lo
    ss, ds = [], []
    for k in np.unique(np.floor(s)):
        sel = np.floor(s) == k
        if sel.sum() < 6:
            continue
        order = np.argsort(d[sel])
        dk, vk = d[sel][order], v[sel][order]
        # Smooth the strip's noise before looking for the crossing.
        if len(vk) >= 5:
            vk = np.convolve(np.pad(vk, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        above = vk >= mid if rising else vk <= mid
        idx = np.flatnonzero(above)
        if idx.size == 0 or idx[0] == 0:
            continue
        i = idx[0]
        v0, v1 = vk[i - 1], vk[i]
        t = 0.5 if v1 == v0 else (mid - v0) / (v1 - v0)
        ds.append(dk[i - 1] + t * (dk[i] - dk[i - 1]))
        ss.append(k + 0.5)
    return np.asarray(ss), np.asarray(ds)


def mtf_from_esf(esf: np.ndarray, bin_px: float = 1.0 / OVERSAMPLE) -> Tuple[np.ndarray, np.ndarray]:
    """Oversampled ESF -> (frequencies in cy/px, MTF normalised to 1 at DC)."""
    lsf = np.gradient(esf)
    n = len(lsf)
    idx = np.arange(n)
    weight = np.abs(lsf)
    centre = float((idx * weight).sum() / weight.sum()) if weight.sum() > 0 else n / 2
    # Hamming window centred on the LSF, wide enough to cover the whole array
    # from that centre (ISO 12233:2017 sec. D.4).
    half = max(centre, n - 1 - centre)
    window = 0.54 + 0.46 * np.cos(np.pi * np.clip((idx - centre) / half, -1, 1))
    spectrum = np.abs(np.fft.rfft(lsf * window))
    if spectrum[0] <= 0:
        raise ValueError("flat LSF")
    mtf = spectrum / spectrum[0]
    freqs = np.fft.rfftfreq(n, d=bin_px)
    # np.gradient is a central difference over 2 bins: response sinc(2 f bin).
    correction = np.sinc(2.0 * freqs * bin_px)
    usable = freqs <= 1.0
    mtf = np.where(usable & (correction > 0.1), mtf / np.maximum(correction, 0.1), mtf)
    return freqs, mtf


def mtf50_from_curve(freqs: np.ndarray, mtf: np.ndarray) -> Optional[float]:
    """First frequency where the MTF falls through 0.5 (linear interpolation)."""
    below = np.flatnonzero(mtf < 0.5)
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:
        return None
    f0, f1, m0, m1 = freqs[i - 1], freqs[i], mtf[i - 1], mtf[i]
    return float(f0 + (m0 - 0.5) * (f1 - f0) / (m0 - m1))


def edge_mtf(
    channel: np.ndarray,
    p0: Sequence[float],
    p1: Sequence[float],
    across_px: float,
    *,
    kind: str = "",
) -> EdgeMTF:
    """
    MTF50 of the straight edge running from ``p0`` to ``p1`` (image (x, y)
    pixels) in a single linear float ``channel``. ``across_px`` is how far
    either side of the edge to sample. Never raises for a bad edge: the
    reason is in ``error``.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    seg = p1 - p0
    length = float(np.hypot(*seg))
    centre = 0.5 * (p0 + p1)
    if length < 8:
        return EdgeMTF(kind, tuple(centre), 0.0, error="edge too short")
    u = seg / length                     # along the edge
    n = np.array([-u[1], u[0]])          # across it
    result = EdgeMTF(kind, (float(centre[0]), float(centre[1])), _slant_off_axis_deg(u))
    if result.angle_deg < MIN_SLANT_DEG:
        result.error = f"edge only {result.angle_deg:.1f} deg off axis (need >= {MIN_SLANT_DEG})"
        return result

    half_along = 0.5 * ALONG_FRACTION * length
    # Axis-aligned bounding box of the oriented region, then the exact mask.
    corners = np.array([centre + a * u + b * n
                        for a in (-half_along, half_along) for b in (-across_px, across_px)])
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + 1
    h, w = channel.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        result.error = "edge region leaves the frame"
        return result
    ys, xs = np.mgrid[y0:y1, x0:x1]
    rel = np.stack([xs - centre[0], ys - centre[1]], axis=-1).astype(np.float64)
    s = rel @ u
    d = rel @ n
    mask = (np.abs(s) <= half_along) & (np.abs(d) <= across_px)
    s, d = s[mask] + half_along, d[mask]
    v = channel[y0:y1, x0:x1].astype(np.float64)[mask]

    if np.mean(v >= CLIP_LEVEL) > MAX_CLIPPED_FRACTION:
        result.error = "clipped highlights in the edge region"
        return result
    lo = float(np.median(v[d < -0.5 * across_px]))
    hi = float(np.median(v[d > 0.5 * across_px]))
    dark, light = min(lo, hi), max(lo, hi)
    if light <= 0 or (dark > 0 and light / dark < MIN_EDGE_CONTRAST):
        result.error = "too little edge contrast"
        return result

    # Refine the edge line from the data (the corners only seed it).
    ss, ds = _strip_crossings(s, d, v, lo, hi)
    if len(ss) < 5:
        result.error = "edge not found along enough of its length"
        return result
    slope, offset = np.polyfit(ss, ds, 1)
    dist = d - (offset + slope * s)
    if lo > hi:                          # make every ESF rise dark -> light
        dist = -dist
    result.angle_deg = _slant_off_axis_deg(
        np.array([u[0] + slope * n[0], u[1] + slope * n[1]])
    )
    if result.angle_deg < MIN_SLANT_DEG:
        result.error = f"edge only {result.angle_deg:.1f} deg off axis (need >= {MIN_SLANT_DEG})"
        return result

    # Oversampled ESF over the span every strip reaches on both sides (the
    # refined line is offset and tilted from the corner line the mask used).
    shift = max(abs(offset), abs(offset + slope * 2 * half_along))
    reach = across_px - shift - 1.0
    if reach < 4:
        result.error = "edge region too narrow"
        return result
    nbins = int(2 * reach * OVERSAMPLE)
    edges_ = np.linspace(-reach, reach, nbins + 1)
    which = np.digitize(dist, edges_) - 1
    keep = (which >= 0) & (which < nbins)
    counts = np.bincount(which[keep], minlength=nbins)
    sums = np.bincount(which[keep], weights=v[keep], minlength=nbins)
    empty = counts == 0
    if empty.mean() > MAX_EMPTY_BIN_FRACTION:
        result.error = f"{empty.mean():.0%} of ESF bins empty (slant too close to a rational angle)"
        return result
    esf = np.zeros(nbins)
    esf[~empty] = sums[~empty] / counts[~empty]
    if empty.any():
        centres = 0.5 * (edges_[:-1] + edges_[1:])
        esf[empty] = np.interp(centres[empty], centres[~empty], esf[~empty])

    try:
        freqs, mtf = mtf_from_esf(esf)
    except ValueError as exc:
        result.error = str(exc)
        return result
    result.mtf50 = mtf50_from_curve(freqs, mtf)
    if result.mtf50 is None:
        result.error = "MTF never falls below 0.5"
    return result


# ---------------------------------------------------------------------------
# Checkerboard frames
# ---------------------------------------------------------------------------

def board_edges(
    corners: np.ndarray, pattern: Tuple[int, int]
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Every square edge between two neighbouring inner corners, as
    ``(kind, p0, p1)``; ``kind`` is the edge's direction in the image."""
    cols, rows = pattern
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    pairs = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                pairs.append((grid[r, c], grid[r, c + 1]))
            if r + 1 < rows:
                pairs.append((grid[r, c], grid[r + 1, c]))
    out = []
    for p0, p1 in pairs:
        dx, dy = np.abs(p1 - p0)
        out.append(("vertical" if dy > dx else "horizontal", p0, p1))
    return out


def measure_board(
    channel: np.ndarray,
    corners: np.ndarray,
    pattern: Tuple[int, int],
) -> Tuple[List[EdgeMTF], float]:
    """MTF50 of every board edge in ``channel``; also returns the square size
    in pixels (median corner spacing)."""
    cols, rows = pattern
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    spacing = float(np.median(np.concatenate([
        np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
        np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
    ])))
    across = ACROSS_FRACTION * spacing
    edges = [edge_mtf(channel, p0, p1, across, kind=kind)
             for kind, p0, p1 in board_edges(corners, pattern)]
    return edges, spacing


def summarize_edges(result: FrameSharpness) -> None:
    """Fill the frame's medians from its per-edge results."""
    values = [e.mtf50 for e in result.edges if e.mtf50 is not None]
    result.edges_total = len(result.edges)
    result.edges_used = len(values)
    if len(values) < MIN_EDGES:
        result.error = result.error or (
            f"only {len(values)} of {len(result.edges)} edges measurable "
            f"(need {MIN_EDGES}); first problem: "
            + next((e.error for e in result.edges if e.error), "none")
        )
        return
    arr = np.asarray(values)
    result.mtf50 = float(np.median(arr))
    q1, q3 = np.percentile(arr, (25, 75))
    result.mtf50_iqr = float(q3 - q1)
    if result.image_size[1]:
        result.mtf50_lw_ph = 2.0 * result.mtf50 * result.image_size[1]
    for kind in ("vertical", "horizontal"):
        sub = [e.mtf50 for e in result.edges if e.kind == kind and e.mtf50 is not None]
        if sub:
            setattr(result, f"mtf50_{kind}", float(np.median(sub)))


def find_board_corners(
    channel: np.ndarray, pattern: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Detect and sub-pixel-refine the board in a linear channel, trying
    progressively larger detection scales. None when the whole board is not
    found."""
    # Imported here: calibrate_distortion pulls in the RAW stack, and the
    # edge maths above should stay importable without it.
    from models.calibrate_distortion import (
        corner_spacing_px, detect_checkerboard, refine_corners,
        stretch_for_detection, subpix_half_window, to_gray8,
    )
    stretched = stretch_for_detection(channel)
    gray8 = to_gray8(stretched)
    for scale in DETECT_SCALES:
        coarse = detect_checkerboard(gray8, pattern, scale)
        if coarse is not None:
            half = subpix_half_window(corner_spacing_px(coarse, pattern))
            return refine_corners(stretched, coarse, half)
    return None


def analyze_raw(
    path: str,
    pattern: Tuple[int, int],
    *,
    corners_hint: Optional[np.ndarray] = None,
    demosaic: Optional[str] = None,
) -> Tuple[FrameSharpness, Optional[np.ndarray]]:
    """
    Develop one capture to linear (sensor frame, camera WB, no tone or
    sharpening) and measure the board's MTF50 on the green channel.

    ``corners_hint``: corners from another frame of the same sweep. Used when
    this frame's board cannot be detected - far out of focus the detector
    gives up long before the edges stop being measurable, and neither the
    board nor the camera moves during a sweep. Returns the result and the
    corners used (for the next frame's hint). Never raises for a frame
    problem; it is reported in ``error``.
    """
    from models.image_io import DEFAULT_DEMOSAIC, SENSOR_FRAME_KWARGS, load_linear_rgb

    start = time.perf_counter()
    result = FrameSharpness(path=path)
    corners = None
    try:
        linear = load_linear_rgb(
            path, white_balance="camera", demosaic=demosaic or DEFAULT_DEMOSAIC,
            **SENSOR_FRAME_KWARGS,
        )
        green = np.ascontiguousarray(linear[..., 1])
        del linear
        result.image_size = (green.shape[1], green.shape[0])
        corners = find_board_corners(green, pattern)
        if corners is None and corners_hint is not None:
            corners = np.asarray(corners_hint, dtype=np.float32)
            result.corners_from_hint = True
        if corners is None:
            result.error = "checkerboard not found (whole board must be visible)"
            return result, None
        result.edges, _ = measure_board(green, corners, pattern)
        summarize_edges(result)
    except Exception as exc:  # noqa: BLE001 - reported per frame
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.seconds = round(time.perf_counter() - start, 2)
    return result, corners


# ---------------------------------------------------------------------------
# Focus-curve maths (hardware-free)
# ---------------------------------------------------------------------------

@dataclass
class FocusPeak:
    best_position: float
    peak_mtf50: float
    fit: str                                    # "parabola" or "sample"
    usable_range: Optional[Tuple[float, float]]  # positions >= usable_fraction x peak
    usable_fraction: float
    at_edge_of_sweep: bool


def fit_focus_peak(
    positions: Sequence[float],
    values: Sequence[Optional[float]],
    *,
    usable_fraction: float = 0.85,
) -> Optional[FocusPeak]:
    """
    Best focus from an MTF50-vs-position sweep: a parabola through the best
    sample and its neighbours (up to two each side), falling back to the best
    sample when the fit is not a downward parabola peaking inside them.
    ``usable_range`` is where the linearly interpolated curve stays at or
    above ``usable_fraction`` of the peak - how much focus error is
    invisible. ``at_edge_of_sweep`` means the best sample is the first or
    last one: the real peak may lie outside the sweep.
    """
    pts = sorted((float(p), float(v)) for p, v in zip(positions, values) if v is not None)
    if len(pts) < 3:
        return None
    pos = np.array([p for p, _ in pts])
    val = np.array([v for _, v in pts])
    i = int(np.argmax(val))
    lo, hi = max(0, i - 2), min(len(pts), i + 3)
    best, peak, how = float(pos[i]), float(val[i]), "sample"
    if hi - lo >= 3:
        a, b, c = np.polyfit(pos[lo:hi], val[lo:hi], 2)
        if a < 0:
            vertex = -b / (2 * a)
            if pos[lo] <= vertex <= pos[hi - 1]:
                best = float(vertex)
                peak = float(a * vertex ** 2 + b * vertex + c)
                how = "parabola"
    threshold = usable_fraction * peak
    usable = None
    above = val >= threshold
    if above.any():
        j0 = j1 = i
        while j0 > 0 and above[j0 - 1]:
            j0 -= 1
        while j1 < len(val) - 1 and above[j1 + 1]:
            j1 += 1

        def _cross(a_i: int, b_i: int) -> float:
            pa, pb, va, vb = pos[a_i], pos[b_i], val[a_i], val[b_i]
            return float(pa + (threshold - va) * (pb - pa) / (vb - va)) if vb != va else float(pa)

        left = _cross(j0 - 1, j0) if j0 > 0 else float(pos[j0])
        right = _cross(j1, j1 + 1) if j1 < len(val) - 1 else float(pos[j1])
        usable = (left, right)
    return FocusPeak(
        best_position=best, peak_mtf50=peak, fit=how, usable_range=usable,
        usable_fraction=usable_fraction, at_edge_of_sweep=i in (0, len(pts) - 1),
    )


# ---------------------------------------------------------------------------
# No-reference sharpness
# ---------------------------------------------------------------------------

def tenengrad(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Mean squared Sobel gradient magnitude, divided by the squared mean level
    so an exposure change does not read as a focus change. Only comparable
    between frames of the same scene: content sets the scale.
    """
    g = np.asarray(gray, dtype=np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        energy, g = energy[m], g[m]
    level = float(np.mean(g))
    if level <= 0:
        return 0.0
    return float(np.mean(energy) / (level * level))
