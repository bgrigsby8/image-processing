"""
color_correction.py
--------------------
A Viam camera component that color-corrects images from a source camera using
a 3x3 Color Correction Matrix (CCM) fitted from a Calibrite / X-Rite
ColorChecker Classic.

Two ways to get corrected images out of this component:

1. Streaming path - ``get_images`` proxies the source camera, applies the CCM
   to every JPEG/PNG frame, and returns the corrected images (names preserved).
   This is what the control tab, data manager, and vision services use.

2. DoCommand path - the studio RAW workflow, for cameras (the PTP model, or
   the Canon CCAPI module) whose full-resolution stills are exposed through
   DoCommand rather than the streaming ``Images`` method:

       {"capture": {"white_balance": "camera",
                    "output_formats": ["tiff16", "jpeg"]}}
           -> trigger a still on the source camera; if it's a RAW (CR3/NEF/...)
              downloaded to disk, demosaic it to 16-bit *linear*, apply white
              balance + the CCM, and write rendered exports next to it - leaving
              the RAW untouched as the master. Returns the export paths, a JSON
              sidecar path recording the development, and a small base64 JPEG
              preview (the full image stays on disk).

   The source still arrives either inline as ``image_base64`` (small JPEGs from
   CCAPI) or as a downloaded file path in ``saved_to`` - the PTP RAW handoff.
   Wire the PTP component as this model's ``camera`` dependency and give PTP a
   ``download_dir`` so its captures land on disk where this model can read them.

       {"capture": {"defer": true}}
       {"capture_result": {"id": "<capture_id>", "wait_sec": 60}}
           -> pipelined capture for rigs that move between shots: ``capture``
              with ``defer`` returns {"capture_id", "status", "camera_path"}
              as soon as the shutter has fired (exposure done - the rig is
              free to move), while the USB download, half-size demosaic, CCM,
              and preview encode continue in the background.
              ``capture_result`` then returns {"source_path", "image_base64",
              ...} (or {"status": "pending"} if not done within ``wait_sec``).
              Deferred captures never write exports or a sidecar - run
              ``develop`` on the returned ``source_path`` when the files are
              actually needed. Requires the ptp model (its ``trigger``
              command) as the source camera.

   This is a non-destructive, Capture One-style pipeline: 16-bit linear math,
   no auto-brightness, the original RAW preserved, adjustments recorded in a
   sidecar. See image_io.py for the decode/export details and color-space notes.

   Relevant config attributes: ``output_dir`` (default: next to the source),
   ``output_formats`` (default ["tiff16", "jpeg", "png16", "png8"]),
   ``jpeg_quality`` (95), ``white_balance`` ("camera"), ``exposure_stops`` (0.0 -
   paste the value `calibrate_color` reports to render at the calibrated
   brightness), ``tone`` ("none" - delivery look: "none" is colour-accurate /
   colorimetric, "c1" matches Capture One's brightness, "medium"/"bright" are
   lighter hand-tuned lifts; all applied to luminance only so hue is preserved),
   ``sharpen`` ("none" - capture sharpening: "light"/"medium"/"strong",
   since RAW is soft before sharpening), ``demosaic`` ("DHT" - RAW demosaic
   algorithm, sharper than libraw's stock AHD), ``write_sidecar`` (true),
   ``delete_after_upload`` (false), and ``nines_api_key`` /
   ``nines_organization_slug`` / ``nines_base_url`` (Nines partner-API
   delivery: enables the ``sku`` option on ``upload`` and the ``nines_upload``
   command below).

       {"develop": {"path": "/photos/IMG_0042.CR3"}}
       {"develop": {"paths": ["/photos/a.CR3", "/photos/b.CR3"]}}
           -> develop existing RAW/image file(s) already on disk through the
              same pipeline, with no camera trigger. Takes the same
              white_balance / exposure_stops / output_formats / output_dir
              options as ``capture``. A single ``path`` returns that file's
              result; ``paths`` returns {"developed": [...], "count": N}.

       {"delete": {"paths": ["/photos/a.CR3", "/photos/a.jpg"]}}
           -> remove files from this machine's disk: skipped captures the
              operator discarded, or sets already safe in the cloud. Guarded -
              only files inside the configured ``output_dir`` can be deleted.
              For kept shots, prefer ``delete_after_upload`` (config attribute
              or per-``upload`` option), which removes each file only after
              its upload succeeds.

Calibration:

       {"calibrate_color": {}}
           -> grab a live-view frame, auto-detect the ColorChecker (cv2.mcc),
              fit a CCM, and return it. No white balance (no RAW from live view).
       {"calibrate_color": {"use_capture": true}}
           -> trigger a full-resolution RAW still, auto-detect the chart,
              measure white balance from the raw CFA under the neutral patches
              ([r,g,b,g2] for rawpy's user_wb), and fit the CCM under that same
              white balance. Returns both.
       {"calibrate_color": {"path": "/photos/chart.CR3"}}
           -> same, from a RAW already on disk (no camera trigger).

The chart is detected automatically anywhere in frame (cv2.mcc, from
opencv-contrib) - no clicking patch centres. Pass ``patch_centers`` (24 [x,y])
to override detection, or ``compute_wb: false`` to skip white balance.
`calibrate_color` returns the fitted 3x3 ``ccm`` and the 4-value
``white_balance``; copy them into the component's ``ccm`` / ``white_balance``
config attributes to make the calibration persist across restarts.
"""

import asyncio
import base64
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from io import BytesIO
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from PIL import Image
from typing_extensions import Self

from viam.app.data_client import DataClient
from viam.components.camera import Camera
from viam.media.utils.pil import pil_to_viam_image, viam_to_pil_image
from viam.rpc.dial import Credentials, DialOptions, _dial_app
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.datasync import (
    DataType,
    FileData,
    FileUploadRequest,
    FileUploadResponse,
    UploadMetadata,
)
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from models.image_io import (
    DEFAULT_DEMOSAIC,
    DEMOSAIC_ALGORITHMS,
    EXPORT_FORMATS,
    SHARPEN_OPTIONS,
    TONE_OPTIONS,
    compute_raw_wb_multipliers,
    export_renditions,
    is_raw,
    linear_to_jpeg_base64,
    linear_to_srgb,
    load_linear_rgb,
    render_raw_for_detection,
    srgb_to_linear,
)

# OpenCV's ColorChecker detector (cv2.mcc) lives in opencv-contrib; import lazily
# so the module still loads (and the streaming/develop paths work) on a host
# without it - calibration raises a clean, actionable error at point of use.
try:
    import cv2  # type: ignore

    _CV2_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the host
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc

# Default delivery set when `output_formats` isn't configured. Override in
# config to trim it (e.g. just ["tiff16", "jpeg"] for a master + proof).
DEFAULT_OUTPUT_FORMATS = ["tiff16", "jpeg", "png16", "png8"]

# app.viam.com rejects gRPC messages over 32 MiB, and the SDK's `file_upload`
# ships the whole file as a single message - too small for a CR3 (~53 MB) or a
# 16-bit TIFF (~250 MB). The FileUpload RPC is client-streaming, so we send
# the file ourselves in chunks safely under that cap.
UPLOAD_CHUNK_BYTES = 1024 * 1024

# Nines partner-API delivery (the REST API documented in the nines-webapp
# repo's partner-api-guide.md): products ("reference items") are upserted by
# `external_id` - our SKU - and imagery is appended to them.
NINES_DEFAULT_BASE_URL = "https://review-app.ninesstyle.com"

# Content types the Nines image-append endpoint accepts, by file extension.
# The RAW master, TIFFs, and JSON sidecar of a capture set are Viam-archival
# only - Nines rejects them.
NINES_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class NinesAPIError(RuntimeError):
    """A failed Nines partner-API call; ``status`` is the HTTP status code, or
    ``None`` when the API was unreachable."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def _read_base64(path: str) -> str:
    """Whole-file base64 for the Nines inline image form (run in a thread)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

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


def _xyy_d50_to_srgb(xyY: np.ndarray) -> np.ndarray:
    """CIE xyY (D50) -> gamma-encoded sRGB in [0, 1]. Out-of-gamut patches (the
    chart's blue/cyan fall outside sRGB) are clipped after the linear transform,
    as any sRGB rendering of the chart must."""
    x, y, big_y = xyY[:, 0], xyY[:, 1], xyY[:, 2]
    xyz = np.stack([big_y * x / y, big_y, big_y * (1.0 - x - y) / y], axis=1)
    xyz_d65 = xyz @ _BRADFORD_D50_TO_D65.T
    linear = np.clip(xyz_d65 @ _XYZ_D65_TO_LINEAR_SRGB.T, 0.0, 1.0)
    return linear_to_srgb(linear).astype(np.float32)


# Gamma-encoded sRGB [0, 1], dark skin -> black; the CCM fit and the
# neutral-brightness readout both reference this.
REFERENCE_SRGB = _xyy_d50_to_srgb(_COLORCHECKER_XYY_D50)


# Canonical sRGB transfer functions live in image_io so the decode/export path
# and the color math agree exactly; aliased here to keep call sites readable.
_srgb_to_linear = srgb_to_linear
_linear_to_srgb = linear_to_srgb


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


def _fit_ccm(measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Least-squares fit of a 3x3 Color Correction Matrix.

    Solves ``reference ~= measured @ CCM.T`` (each reference row = CCM @ measured_row).

    Parameters
    ----------
    measured  : (N, 3) float32, linear-light measured RGB, normalised [0, 1]
    reference : (N, 3) float32, linear-light reference RGB, normalised [0, 1]

    Returns
    -------
    ccm : (3, 3) float32
    """
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
    if not detector.process(bgr, cv2.mcc.MCC24):
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
        samples = []
        for x, y in centers:
            x, y = int(x), int(y)
            patch = img_rgb[y - radius:y + radius, x - radius:x + radius].reshape(-1, 3)
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
            patch = img_linear[y0:y + radius, x0:x + radius].reshape(-1, 3)
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
        ccm = _fit_ccm(measured_linear, reference_linear)
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
        calibration quality. Distances are Euclidean in linear RGB (a rough ΔE
        proxy, not CIE ΔE), scaled by 100.
        """
        if patch_centers is not None:
            measured = PatchSampler.sample_at_centers(img_rgb, patch_centers)
        else:
            measured = PatchSampler.auto_sample(img_rgb)
        ref_linear = _srgb_to_linear(REFERENCE_SRGB)
        corrected_linear = (measured @ self.ccm.T).clip(0, 1)

        def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.sqrt(np.sum((a - b) ** 2, axis=1)) * 100

        before = dist(measured, ref_linear)
        after = dist(corrected_linear, ref_linear)
        return {
            "before": {"mean": float(before.mean()), "max": float(before.max())},
            "after": {"mean": float(after.mean()), "max": float(after.max())},
        }


# ---------------------------------------------------------------------------
# Image <-> base64 / PIL helpers (used on the DoCommand boundary)
# ---------------------------------------------------------------------------

def _base64_to_rgb(image_base64: str) -> np.ndarray:
    """Decode a base64-encoded image (JPEG/PNG) into an (H, W, 3) uint8 RGB array."""
    raw = base64.b64decode(image_base64)
    pil = Image.open(BytesIO(raw)).convert("RGB")
    return np.array(pil)


class ColorCorrection(Camera, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("brad-grigsby", "image-processing"), "color-correction"
    )

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """Create a new instance of this Camera component.

        ``EasyResource.new`` only constructs the instance - it does *not* call
        ``reconfigure``, and viam-server only calls ``reconfigure`` on later
        config changes, not on the initial add. So we must configure here, or
        ``self.camera``/``self.corrector`` won't exist when the first request
        arrives.
        """
        instance = cls(config.name)
        instance.reconfigure(config, dependencies)
        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """Validate config and declare the source camera as a required dependency."""
        attrs = struct_to_dict(config.attributes)

        camera = attrs.get("camera")
        if not camera:
            raise ValueError("Missing required attribute `camera` in config")

        ccm = attrs.get("ccm")
        if ccm is not None and np.array(ccm, dtype=np.float32).shape != (3, 3):
            raise ValueError("`ccm` must be a 3x3 matrix")

        formats = attrs.get("output_formats")
        if formats is not None:
            unknown = [f for f in formats if f not in EXPORT_FORMATS]
            if unknown:
                raise ValueError(
                    f"unknown `output_formats` {unknown}; valid: "
                    f"{sorted(EXPORT_FORMATS)}"
                )

        output_dir = attrs.get("output_dir")
        if output_dir is not None and not isinstance(output_dir, str):
            raise ValueError("`output_dir` must be a string path")

        exposure_stops = attrs.get("exposure_stops")
        if exposure_stops is not None and not isinstance(exposure_stops, (int, float)):
            raise ValueError("`exposure_stops` must be a number (stops of exposure)")

        tone = attrs.get("tone")
        if tone is not None and tone not in TONE_OPTIONS:
            raise ValueError(f"`tone` must be one of {list(TONE_OPTIONS)}")

        sharpen = attrs.get("sharpen")
        if sharpen is not None and sharpen not in SHARPEN_OPTIONS:
            raise ValueError(f"`sharpen` must be one of {list(SHARPEN_OPTIONS)}")

        demosaic = attrs.get("demosaic")
        if demosaic is not None and demosaic not in DEMOSAIC_ALGORITHMS:
            raise ValueError(f"`demosaic` must be one of {list(DEMOSAIC_ALGORITHMS)}")

        for key in ("nines_api_key", "nines_organization_slug", "nines_base_url"):
            value = attrs.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"`{key}` must be a string")

        return [str(camera)], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """Wire up the source camera and load the CCM from the `ccm` attribute."""
        attrs = struct_to_dict(config.attributes)

        camera = attrs.get("camera")
        source = dependencies.get(Camera.get_resource_name(str(camera)))
        if source is None:
            raise ValueError(f"Could not resolve source camera dependency `{camera}`")
        self.camera: Camera = source

        # The CCM lives inline in the `ccm` config attribute. Run the
        # `calibrate_color` DoCommand to compute one, then copy the returned
        # matrix into this attribute to make the correction persist.
        ccm = attrs.get("ccm")
        if ccm is not None:
            self.corrector = ColorCorrector(np.array(ccm, dtype=np.float32))
        else:
            self.corrector = ColorCorrector.identity()
            self.logger.info("No `ccm` configured; passing images through uncorrected")

        # Studio export settings (used by the `capture` DoCommand RAW pipeline).
        # output_dir defaults to wherever the source file was downloaded.
        self._output_dir: Optional[str] = attrs.get("output_dir") or None
        self._output_formats: List[str] = list(
            attrs.get("output_formats") or DEFAULT_OUTPUT_FORMATS
        )
        self._jpeg_quality: int = int(attrs.get("jpeg_quality", 95))
        self._white_balance = attrs.get("white_balance", "camera")
        # Default exposure compensation (stops) applied at the raw stage, the
        # digital counterpart to flash power. `calibrate_color` reports the
        # offset the chart implied vs. the reference; paste it here (alongside
        # `ccm` / `white_balance`) to render every capture at the calibrated
        # brightness when the flash can't reach the reference optically. Per-call
        # `exposure_stops` on capture/develop still overrides this.
        self._exposure_stops: float = float(attrs.get("exposure_stops", 0.0))
        # Optional delivery "look" layered on the colour-accurate render: "none"
        # (default) is pure colorimetric output; "c1" matches Capture One's
        # brightness, "medium"/"bright" are lighter hand-tuned lifts. The curve
        # is applied to luminance only, so hue/saturation are untouched - only
        # lightness/contrast changes. Applied to every export and the preview.
        self._tone: str = attrs.get("tone") or "none"
        # Capture sharpening ("none"/"light"/"medium"/"strong"): RAW is soft
        # before sharpening, so an unsharpened export looks blurry next to a
        # Capture One / Lightroom render. Default "none" (opt-in). And the
        # demosaic algorithm used to decode RAW (DHT default - sharper than
        # libraw's stock AHD). Both feed every export and the preview.
        self._sharpen: str = attrs.get("sharpen") or "none"
        self._demosaic: str = attrs.get("demosaic") or DEFAULT_DEMOSAIC
        self._write_sidecar: bool = bool(attrs.get("write_sidecar", True))

        # Local-disk hygiene, mirroring ptp's `delete_after_download`: once a
        # file is confirmed in the cloud, the local copy is redundant. Files
        # that fail to upload are kept for retry.
        self._delete_after_upload: bool = bool(attrs.get("delete_after_upload", False))

        # Optional Nines partner-API delivery (the REST API in the nines-webapp
        # repo's partner-api-guide.md). With an API key and organization slug
        # configured, `upload` calls that carry a `sku` also upsert the Nines
        # product for that SKU and append the shot's delivery image to it, and
        # the `nines_upload` command sends arbitrary on-disk images. Left
        # unconfigured, `upload` behaves exactly as before.
        self._nines_api_key: Optional[str] = (
            attrs.get("nines_api_key") or os.environ.get("NINES_API_KEY") or None
        )
        self._nines_org_slug: Optional[str] = (
            attrs.get("nines_organization_slug") or None
        )
        self._nines_base_url: str = str(
            attrs.get("nines_base_url") or NINES_DEFAULT_BASE_URL
        ).rstrip("/")
        # Upserted reference-item ids keyed by (org slug, SKU), so a multi-shot
        # submit upserts each product once. The org is part of the key because
        # one machine can serve multiple orgs (the webapp may pass a per-request
        # `shots_organization_slug`) and the same external_id/SKU can exist in
        # more than one org - a SKU-only key would deliver one org's shot to
        # another org's product. Reset on reconfigure: the ids are also scoped
        # to the base URL, which may just have changed.
        self._nines_item_ids: Dict[Tuple[str, str], str] = {}

        # The `upload` DoCommand authenticates to the cloud with the API key
        # Viam injects into every module process (VIAM_API_KEY / VIAM_API_KEY_ID),
        # so no credentials are configured here. part_id falls back to the
        # machine's env var. The data client is created lazily and reused.
        self._part_id: Optional[str] = (
            attrs.get("part_id") or os.environ.get("VIAM_MACHINE_PART_ID") or None
        )
        self._data_client: Optional[DataClient] = None

        # Bound the cloud round-trips so a stuck auth dial or a stalled file
        # transfer surfaces as a clear error / per-file failure (the file is
        # kept for retry) instead of hanging the submit forever. Timeouts are
        # per-file, not per-batch, so a large shoot never trips a single global
        # deadline. The dial only happens on the first upload, which is the
        # usual place a submit silently wedges when the machine can't reach the
        # cloud.
        self._upload_dial_timeout_s: float = float(
            attrs.get("upload_dial_timeout_s", 30.0)
        )
        self._upload_file_timeout_s: float = float(
            attrs.get("upload_file_timeout_s", 180.0)
        )

        # In-flight deferred captures (`capture` with `defer: true`), keyed by
        # the capture_id handed back to the caller. Preserved across
        # reconfigure so a mid-sequence config change doesn't orphan results.
        self._pending_captures: Dict[str, "asyncio.Task"] = getattr(
            self, "_pending_captures", {}
        )
        self._capture_seq: int = getattr(self, "_capture_seq", 0)

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _correct_viam_image(self, image: ViamImage) -> ViamImage:
        """Apply the CCM to a single ViamImage, preserving its mime type."""
        pil_image = viam_to_pil_image(image).convert("RGB")
        corrected = self.corrector.apply_to_rgb(np.array(pil_image))
        mime = image.mime_type if image.mime_type == CameraMimeType.PNG else CameraMimeType.JPEG
        return pil_to_viam_image(Image.fromarray(corrected), mime)

    async def get_images(
        self,
        *,
        filter_source_names: Optional[Sequence[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Sequence[NamedImage], ResponseMetadata]:
        images, metadata = await self.camera.get_images(
            filter_source_names=filter_source_names,
            extra=extra,
            timeout=timeout,
            **kwargs,
        )

        corrected: List[NamedImage] = []
        for image in images:
            if image.mime_type in (CameraMimeType.JPEG, CameraMimeType.PNG):
                viam_img = self._correct_viam_image(image)
                corrected.append(NamedImage(image.name, viam_img.data, viam_img.mime_type))
            else:
                # Pass non-image payloads (e.g. depth) through untouched.
                self.logger.debug(
                    f"Passing through image '{image.name}' with uncorrectable "
                    f"mime type {image.mime_type}"
                )
                corrected.append(image)

        return corrected, metadata

    async def get_point_cloud(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[bytes, str]:
        # Color correction doesn't apply to point clouds; proxy the source.
        return await self.camera.get_point_cloud(extra=extra, timeout=timeout, **kwargs)

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Camera.Properties:
        # Report the source camera's properties; we don't change resolution,
        # mime types, or intrinsics.
        return await self.camera.get_properties(timeout=timeout, **kwargs)

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        resp: Dict[str, ValueTypes] = {}

        if "calibrate_color" in command:
            resp["calibrate_color"] = await self._calibrate_color(
                command.get("calibrate_color") or {}, timeout
            )

        if "capture" in command:
            resp["capture"] = await self._capture_corrected(
                command.get("capture") or {}, timeout
            )

        if "capture_result" in command:
            resp["capture_result"] = await self._capture_result(
                command.get("capture_result") or {}
            )

        if "develop" in command:
            resp["develop"] = await self._develop(command.get("develop") or {})

        if "upload" in command:
            resp["upload"] = await self._upload(command.get("upload") or {})

        if "nines_upload" in command:
            resp["nines_upload"] = await self._nines_upload(
                command.get("nines_upload") or {}
            )

        if "delete" in command:
            resp["delete"] = self._delete_local(command.get("delete") or {})

        if not resp:
            raise ValueError(
                "no recognized command; supported: calibrate_color, capture, "
                "capture_result, develop, upload, nines_upload, delete"
            )
        return resp

    # ------------------------------------------------------------------
    # DoCommand handlers
    # ------------------------------------------------------------------

    def _linear_from_capture_response(
        self,
        capture: Any,
        white_balance: Any,
        exposure_stops: float,
        half_size: bool = False,
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Turn a source camera's ``capture`` DoCommand response into a
        **linear-light** float RGB array (sRGB primaries) plus the source path.

        Two shapes are supported, in priority order:

        * ``image_base64`` - an inline JPEG/PNG, small enough to ship over gRPC
          (the Canon CCAPI flow). Decoded as 8-bit sRGB and linearized; ``None``
          path since there's no file on disk.
        * ``saved_to`` (or ``path``) - a file the source wrote to disk. This is
          the PTP model's handoff for full-resolution stills, including RAW
          (CR3/NEF/ARW/...). It's demosaiced to 16-bit linear by
          ``image_io.load_linear_rgb`` (applying white balance / exposure at the
          raw stage), with no precision lost before color correction.
        """
        if not isinstance(capture, Mapping):
            raise ValueError("source camera `capture` returned an unexpected response")

        image_b64 = capture.get("image_base64")
        if image_b64:
            rgb8 = _base64_to_rgb(image_b64).astype(np.float32) / 255.0
            return srgb_to_linear(rgb8).astype(np.float32), None

        path = capture.get("saved_to") or capture.get("path")
        if path:
            linear = load_linear_rgb(
                str(path), white_balance=white_balance,
                exposure_stops=exposure_stops, half_size=half_size,
                demosaic=self._demosaic,
            )
            return linear, str(path)

        raise ValueError(
            "source camera `capture` returned neither an `image_base64` field "
            "nor a `saved_to` path; if the source is the PTP camera, configure "
            "its `download_dir` so captures are written to disk"
        )

    async def _acquire_calibration_source(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Get the source to calibrate from, as ``(raw_path, rgb8)``.

        ``raw_path`` is set when a RAW file is on disk - the path that unlocks
        raw-CFA white balance. Otherwise ``rgb8`` is an 8-bit sRGB frame for
        CCM-only calibration. Exactly one is non-None.

        Resolution order: explicit ``path`` -> ``use_capture`` (trigger a still
        on the source, prefer its ``saved_to`` RAW) -> the streaming frame.
        """
        def _file_to_rgb8(p: str) -> np.ndarray:
            linear = load_linear_rgb(
                str(p), white_balance="camera", demosaic=self._demosaic
            )
            return (linear_to_srgb(linear) * 255.0).clip(0, 255).astype(np.uint8)

        path = opts.get("path")
        if path:
            if is_raw(str(path)):
                return str(path), None
            return None, _file_to_rgb8(str(path))

        if opts.get("use_capture"):
            capture_opts = opts.get("capture_options", {"af": True})
            source_resp = await self.camera.do_command(
                {"capture": capture_opts}, timeout=timeout
            )
            capture = source_resp.get("capture", source_resp)
            if isinstance(capture, Mapping):
                p = capture.get("saved_to") or capture.get("path")
                if p and is_raw(str(p)):
                    return str(p), None
                if p:
                    return None, _file_to_rgb8(str(p))
                b64 = capture.get("image_base64")
                if b64:
                    return None, _base64_to_rgb(b64)
            raise ValueError(
                "source `capture` returned nothing usable for calibration "
                "(no RAW `saved_to`, file `path`, or inline `image_base64`)"
            )

        images, _ = await self.camera.get_images(timeout=timeout)
        for image in images:
            if image.mime_type in (CameraMimeType.JPEG, CameraMimeType.PNG):
                return None, np.array(viam_to_pil_image(image).convert("RGB"))
        raise ValueError("source camera returned no JPEG/PNG image to use")

    async def _calibrate_color(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Mapping[str, ValueTypes]:
        """
        Auto-calibrate from a ColorChecker frame: detect the chart (cv2.mcc),
        measure white balance from the raw CFA, and fit a CCM under that same
        white balance. Both are applied to this component immediately and
        returned so they can be copied into the ``ccm`` / ``white_balance``
        config attributes to persist across restarts.

        Options (all optional):
          ``use_capture``    trigger a full-res still on the source (needed for
                             white balance - the RAW must be on disk).
          ``path``           calibrate from a RAW/image file already on disk.
          ``capture_options``forwarded to the source ``capture`` (e.g. {"af": true}).
          ``compute_wb``     derive white balance from the chart (default true).
          ``patch_centers``  24 [x, y] coords to override auto-detection.
          ``radius``         patch sampling half-width in px (default: ~15% of
                             the detected patch size, or 10 with manual centers).

        Returns ``ccm`` (pure-colour, ~unity-gain), ``white_balance``
        ([r,g,b,g2] or null), ``exposure_stops`` (the brightness offset the chart
        implied vs. the reference - pass it back as ``exposure_stops`` on
        capture/develop to render at the ColorChecker's nominal brightness),
        ``neutral_brightness`` (as-shot 0-255 sRGB picker readout per neutral
        patch with its reference target - adjust the light power until measured
        matches reference to hit nominal brightness without touching camera
        exposure), and a ``delta_e`` report whose ``after`` figure is
        exposure-normalised colour accuracy.
        """
        raw_path, rgb8 = await self._acquire_calibration_source(opts, timeout)
        compute_wb = bool(opts.get("compute_wb", True))
        radius = int(opts["radius"]) if "radius" in opts else None
        manual_centers = opts.get("patch_centers")

        # Image used to *locate* the patches. For a RAW we render it unrotated
        # (user_flip=0) so the centres map straight onto the CFA and the linear
        # CCM render below.
        detect_img = render_raw_for_detection(raw_path) if raw_path else rgb8

        if manual_centers is not None:
            centers = np.array(
                [(int(x), int(y)) for x, y in manual_centers], dtype=np.float32
            )
            neutral_boxes = None
            radius = radius if radius is not None else 10
        else:
            detection = detect_colorchecker(detect_img)
            if detection is None:
                raise ValueError(
                    "could not auto-detect the ColorChecker; ensure the whole "
                    "chart is visible and unobstructed, or pass `patch_centers`"
                )
            if radius is None:
                radius = int(detection["suggested_radius"])
            centers = detection["centers"]
            neutral_boxes = detection["neutral_boxes_norm"]

        # White balance from the raw Bayer/CFA under the neutral patches.
        wb: Optional[List[float]] = None
        wb_note: Optional[str] = None
        if compute_wb:
            if raw_path and neutral_boxes:
                wb = compute_raw_wb_multipliers(raw_path, neutral_boxes)
            elif raw_path:
                wb_note = (
                    "skipped: white balance needs auto-detected neutral patches; "
                    "omit `patch_centers` to enable it"
                )
            else:
                wb_note = (
                    "skipped: raw-CFA white balance needs a RAW capture - set "
                    "`use_capture: true` with a RAW source, or pass a RAW `path`"
                )

        # Fit the CCM on patches developed with the SAME white balance the
        # captures will use, so the matrix and the WB stay consistent.
        reference_linear = _srgb_to_linear(REFERENCE_SRGB)
        if raw_path:
            linear = load_linear_rgb(
                raw_path,
                white_balance=(wb if wb is not None else "camera"),
                user_flip=0,  # match detect_img so `centers` line up
                demosaic=self._demosaic,
            )
            measured_linear = PatchSampler.sample_linear_at_centers(linear, centers, radius)
        else:
            measured_linear = PatchSampler.sample_at_centers(
                detect_img, [(int(x), int(y)) for x, y in centers], radius
            )

        # Decouple exposure (a single scalar) from colour (the matrix). The chart
        # is typically exposed below the reference's nominal brightness; fitting
        # the CCM directly makes it absorb that gain (diagonal >> 1), which then
        # brightens *every* developed frame and clips highlights early. Instead we
        # scale the measured patches so the neutral ramp matches the reference
        # luminance, fit the CCM on that (keeping it ~unity-gain, pure colour),
        # and report the implied exposure offset for the caller to dial in via
        # `exposure_stops`. Neutral 8 / 6.5 / 5 / 3.5 - skip the clip-prone white
        # and noisy black ends of the ramp.
        neutral_fit = [19, 20, 21, 22]
        meas_neutral = measured_linear[neutral_fit].reshape(-1)
        ref_neutral = reference_linear[neutral_fit].reshape(-1)
        energy = float(np.dot(meas_neutral, meas_neutral))
        exposure_scale = float(np.dot(meas_neutral, ref_neutral) / energy) if energy > 0 else 1.0
        measured_fit = measured_linear * exposure_scale

        ccm = _fit_ccm(measured_fit, reference_linear)
        corrector = ColorCorrector(ccm)

        def _delta_e_stats(values: np.ndarray) -> Dict[str, float]:
            d = np.sqrt(np.sum((np.clip(values, 0, 1) - reference_linear) ** 2, axis=1)) * 100
            return {"mean": float(d.mean()), "max": float(d.max())}

        # "after" is exposure-normalised, so it reflects pure colour accuracy
        # independent of how bright you choose to render (via exposure_stops).
        report = {
            "before": _delta_e_stats(measured_linear),
            "after": _delta_e_stats(measured_fit @ ccm.T),
        }
        exposure_stops = float(np.log2(exposure_scale)) if exposure_scale > 0 else 0.0
        neutral_brightness = _neutral_brightness_report(measured_linear)

        self.corrector = corrector
        if wb is not None:
            # Subsequent capture/develop default to this WB unless overridden.
            self._white_balance = wb

        self.logger.info(
            f"Calibrated CCM (delta-E mean {report['before']['mean']:.1f} -> "
            f"{report['after']['mean']:.1f}, exposure {exposure_stops:+.2f} stops, "
            f"neutral 6.5 as-shot {neutral_brightness['neutral_6_5']['measured']:.0f}"
            f"/{neutral_brightness['neutral_6_5']['reference']:.0f})"
            + (f"; white balance [{', '.join(f'{v:.3f}' for v in wb)}]" if wb else "")
            + "; copy `ccm`"
            + (" and `white_balance`" if wb else "")
            + " into the component config to persist"
        )
        result: Dict[str, ValueTypes] = {
            "ccm": corrector.ccm.tolist(),
            "white_balance": wb,
            "exposure_stops": exposure_stops,
            "neutral_brightness": neutral_brightness,
            "delta_e": report,
        }
        if wb_note:
            result["white_balance_note"] = wb_note
        return result

    async def _capture_corrected(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Mapping[str, ValueTypes]:
        """
        Studio capture: trigger a full-resolution still on the source camera,
        develop it through the 16-bit linear pipeline (white balance + CCM),
        and write rendered exports - leaving any RAW original untouched.

        ``opts`` (all optional):
          ``capture_options``  forwarded to the source's ``capture`` (e.g. {"af": true})
          ``white_balance``    "camera" (default) | "auto" | "daylight" | [r,g,b,g2]
          ``exposure_stops``   exposure compensation applied at the raw stage
          ``tone``             delivery look: "none" (colour-accurate) | "c1"
                               (matches Capture One) | "medium" | "bright"
                               (lighter lifts); luminance-only, hue preserved
          ``sharpen``          capture sharpening: "none" | "light" | "medium"
                               | "strong"
          ``demosaic``         RAW demosaic algorithm (DHT/AHD/AAHD/DCB/VNG/PPG)
          ``output_formats``   subset of tiff16/tiff8/jpeg/png16/png8; pass []
                               to skip exports (preview-only capture - develop
                               the RAW later with the ``develop`` command)
          ``output_dir``       where to write exports (default: next to the source file)
          ``defer``            true -> return as soon as the shutter has fired
                               (the rig is free to move); download/decode/preview
                               continue in the background and are fetched with
                               ``capture_result``. Requires a source camera with
                               a ``trigger`` DoCommand (the ptp model). Deferred
                               captures never write exports or a sidecar - run
                               ``develop`` on the RAW for those.

        Returns the written export paths, the sidecar path, and a small base64
        JPEG preview (not the full-res image - that stays on disk). With
        ``defer`` it instead returns {"capture_id", "status": "pending",
        "camera_path"} immediately after the shutter fires.
        """
        if opts.get("defer"):
            return await self._capture_deferred(opts, timeout)

        capture_opts = opts.get("capture_options", {"af": True})
        white_balance = opts.get("white_balance", self._white_balance)
        exposure_stops = float(opts.get("exposure_stops", self._exposure_stops))
        formats = list(opts.get("output_formats", self._output_formats))
        out_dir_override = opts.get("output_dir") or self._output_dir
        tone = opts.get("tone", self._tone)
        sharpen = opts.get("sharpen", self._sharpen)
        # When nothing is being exported, the decode only feeds the preview -
        # a half-resolution demosaic is ~4x faster and indistinguishable there.
        preview_only = not formats

        start = time.perf_counter()
        source_resp = await self.camera.do_command({"capture": capture_opts}, timeout=timeout)
        capture = source_resp.get("capture", source_resp)
        self.logger.debug(
            f"[timing] source camera capture (incl. download): "
            f"{time.perf_counter() - start:.2f}s"
        )

        t_decode = time.perf_counter()
        # The decode and develop/export steps are seconds of pure CPU; run them
        # in a worker thread so the event loop keeps serving other requests.
        linear, source_path = await asyncio.to_thread(
            self._linear_from_capture_response,
            capture, white_balance, exposure_stops, preview_only,
        )
        self.logger.debug(
            f"[timing] decode to linear RGB (incl. white balance): "
            f"{time.perf_counter() - t_decode:.2f}s"
        )
        result = await asyncio.to_thread(
            self._develop_one,
            linear, source_path, white_balance, exposure_stops, formats,
            out_dir_override, True, tone, sharpen,
        )
        self.logger.debug(
            f"[timing] capture pipeline total: {time.perf_counter() - start:.2f}s"
        )
        return result

    async def _capture_deferred(
        self, opts: Mapping[str, Any], timeout: Optional[float]
    ) -> Mapping[str, ValueTypes]:
        """
        Fire the shutter and return as soon as the exposure is done, so the
        caller can move the rig while the slow parts (USB download, demosaic,
        preview encode) run in a background task. The source's lock serializes
        camera access, so a background download naturally queues ahead of the
        next pose's trigger.
        """
        white_balance = opts.get("white_balance", self._white_balance)
        exposure_stops = float(opts.get("exposure_stops", self._exposure_stops))
        tone = opts.get("tone", self._tone)
        sharpen = opts.get("sharpen", self._sharpen)

        start = time.perf_counter()
        try:
            resp = await self.camera.do_command(
                {"trigger": opts.get("capture_options", {})}, timeout=timeout
            )
        except Exception as exc:
            raise RuntimeError(
                f"deferred capture needs a source camera with a `trigger` "
                f"DoCommand (the ptp model); triggering failed: {exc}"
            ) from exc
        trig = resp.get("trigger") or {}
        camera_path = trig.get("path")
        if not camera_path:
            raise ValueError("source camera `trigger` returned no `path`")
        self.logger.debug(
            f"[timing] deferred capture trigger (shutter + settle): "
            f"{time.perf_counter() - start:.2f}s"
        )

        self._capture_seq += 1
        stem = os.path.splitext(os.path.basename(str(camera_path)))[0]
        capture_id = f"{self._capture_seq}-{stem}"
        self._pending_captures[capture_id] = asyncio.create_task(
            self._finish_deferred_capture(
                capture_id, str(camera_path), white_balance, exposure_stops,
                tone, sharpen,
            )
        )
        # Drop completed-and-collected stragglers if a caller never fetched
        # them, so an unattended sequence can't grow the table without bound.
        if len(self._pending_captures) > 64:
            for key in [
                k for k, t in self._pending_captures.items() if t.done()
            ][:-64]:
                self._pending_captures.pop(key, None)

        return {
            "capture_id": capture_id,
            "status": "pending",
            "camera_path": str(camera_path),
        }

    async def _finish_deferred_capture(
        self,
        capture_id: str,
        camera_path: str,
        white_balance: Any,
        exposure_stops: float,
        tone: Optional[str] = None,
        sharpen: Optional[str] = None,
    ) -> Dict[str, ValueTypes]:
        """Background half of a deferred capture: download the still from the
        camera, decode at half size, apply the CCM, and build the preview. No
        exports or sidecar - the RAW on disk is the handoff to ``develop``."""
        start = time.perf_counter()
        resp = await self.camera.do_command({"download": {"path": camera_path}})
        meta = resp.get("download") or {}
        saved = meta.get("saved_to")
        if not saved:
            raise ValueError(
                f"source camera did not save {camera_path!r} to disk; configure "
                f"its `download_dir` so deferred captures can be developed later"
            )
        linear = await asyncio.to_thread(
            load_linear_rgb, str(saved),
            white_balance=white_balance, exposure_stops=exposure_stops,
            half_size=True, demosaic=self._demosaic,
        )
        corrected = await asyncio.to_thread(self.corrector.apply_to_linear, linear)
        preview = await asyncio.to_thread(
            linear_to_jpeg_base64, corrected, tone=tone, sharpen=sharpen
        )
        self.logger.debug(
            f"[timing] deferred capture {capture_id} background "
            f"(download + decode + preview): {time.perf_counter() - start:.2f}s"
        )
        return {
            "capture_id": capture_id,
            "status": "done",
            "source_path": str(saved),
            "image_base64": preview,
            "mime_type": CameraMimeType.JPEG.value,
            "ccm_applied": not self.corrector.is_identity,
            "color_space": "sRGB",
        }

    async def _capture_result(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Fetch the result of a deferred capture.

        ``opts``:
          ``id``        the capture_id returned by ``capture`` with ``defer`` (required)
          ``wait_sec``  how long to wait for the background work (default 60;
                        0 polls). Returns {"status": "pending"} on timeout -
                        call again to keep waiting.
        """
        capture_id = opts.get("id") or opts.get("capture_id")
        if not capture_id:
            raise ValueError(
                "`capture_result` needs the `id` returned by `capture` with `defer`"
            )
        capture_id = str(capture_id)
        task = self._pending_captures.get(capture_id)
        if task is None:
            raise ValueError(
                f"unknown capture id {capture_id!r}; it may have already been "
                f"collected, or the module restarted since the capture"
            )
        wait_sec = float(opts.get("wait_sec", 60.0))
        try:
            # shield() so a timeout here doesn't cancel the background work.
            result = await asyncio.wait_for(asyncio.shield(task), timeout=wait_sec)
        except asyncio.TimeoutError:
            return {"capture_id": capture_id, "status": "pending"}
        except Exception as exc:
            self._pending_captures.pop(capture_id, None)
            raise RuntimeError(f"deferred capture {capture_id} failed: {exc}") from exc
        self._pending_captures.pop(capture_id, None)
        return result

    async def _develop(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Develop existing image file(s) already on disk - no camera trigger.
        Point this at a RAW (CR3/NEF/ARW/...) or any JPEG/PNG/TIFF and it runs
        the same 16-bit linear pipeline (white balance + CCM) and writes the
        rendered exports + sidecar, leaving the original untouched.

        ``opts``:
          ``path``           a single file path (returns that file's result), OR
          ``paths``          a list of file paths (returns {"developed": [...]})
          ``white_balance``  "camera" (default) | "auto" | "daylight" | [r,g,b,g2]
          ``exposure_stops`` exposure compensation applied at the raw stage
          ``tone``           delivery look: "none" | "c1" | "medium" | "bright"
          ``sharpen``        capture sharpening: "none"|"light"|"medium"|"strong"
          ``demosaic``       RAW demosaic algorithm (DHT/AHD/AAHD/DCB/VNG/PPG)
          ``output_formats`` subset of tiff16/tiff8/jpeg/png16/png8
          ``output_dir``     where to write exports (default: next to each file)
        """
        raw_paths = opts.get("paths")
        single = raw_paths is None
        if single:
            one = opts.get("path")
            if not one:
                raise ValueError(
                    "`develop` needs a `path` (string) or `paths` (list of strings)"
                )
            raw_paths = [one]
        paths = [str(p) for p in raw_paths]

        white_balance = opts.get("white_balance", self._white_balance)
        exposure_stops = float(opts.get("exposure_stops", self._exposure_stops))
        formats = list(opts.get("output_formats", self._output_formats))
        out_dir_override = opts.get("output_dir") or self._output_dir
        tone = opts.get("tone", self._tone)
        sharpen = opts.get("sharpen", self._sharpen)

        results: List[Mapping[str, ValueTypes]] = []
        for path in paths:
            t_file = time.perf_counter()
            # Decode + export are seconds of pure CPU per file; keep them off
            # the event loop so other requests stay responsive mid-batch.
            linear = await asyncio.to_thread(
                load_linear_rgb,
                path, white_balance=white_balance, exposure_stops=exposure_stops,
                demosaic=self._demosaic,
            )
            results.append(
                await asyncio.to_thread(
                    self._develop_one,
                    linear, path, white_balance, exposure_stops, formats,
                    out_dir_override,
                    # Skip the per-file base64 preview in batch mode to keep the
                    # response small; a single develop still returns its preview.
                    include_preview=single,
                    tone=tone,
                    sharpen=sharpen,
                )
            )
            self.logger.debug(
                f"[timing] develop {os.path.basename(path)} total: "
                f"{time.perf_counter() - t_file:.2f}s"
            )

        if single:
            return results[0]
        return {"developed": results, "count": len(results)}

    def _develop_one(
        self,
        linear: np.ndarray,
        source_path: Optional[str],
        white_balance: Any,
        exposure_stops: float,
        formats: Sequence[str],
        out_dir_override: Optional[str],
        include_preview: bool = True,
        tone: Optional[str] = None,
        sharpen: Optional[str] = None,
    ) -> Dict[str, ValueTypes]:
        """
        Shared core for ``capture`` and ``develop``: apply the CCM in linear
        light, write the rendered exports (non-destructively) and a sidecar, and
        return the result. ``linear`` is linear-light float RGB; ``source_path``
        is the originating file (or None for an inline base64 capture).
        """
        t_ccm = time.perf_counter()
        corrected = self.corrector.apply_to_linear(linear)
        self.logger.debug(
            f"[timing] apply color correction (CCM): {time.perf_counter() - t_ccm:.2f}s"
        )

        # Exports land alongside the source file unless an output_dir is set.
        out_dir = out_dir_override or (
            os.path.dirname(source_path) if source_path else None
        )
        stem = (
            os.path.splitext(os.path.basename(source_path))[0]
            if source_path else "capture"
        )
        # A RAW source (.cr3/.nef/...) never collides with our .tif/.jpg/.png
        # exports, so its name is preserved. But if the source is itself a
        # JPEG/PNG/TIFF, a same-name export would overwrite the original - so
        # suffix the exports to keep the pipeline non-destructive.
        if source_path and not is_raw(source_path):
            stem = stem + "_corrected"
        exports: Dict[str, str] = {}
        if out_dir:
            t_export = time.perf_counter()
            exports = export_renditions(
                corrected, out_dir, stem, formats,
                quality=self._jpeg_quality, tone=tone, sharpen=sharpen,
            )
            self.logger.debug(
                f"[timing] export {len(exports)} format(s): "
                f"{time.perf_counter() - t_export:.2f}s"
            )
            self.logger.info(f"exported {list(exports)} for {stem} to {out_dir}")
        else:
            self.logger.warning(
                "no `output_dir` configured and source has no path; "
                "returning a preview only (nothing written to disk)"
            )

        sidecar = None
        if self._write_sidecar and source_path:
            sidecar = self._write_sidecar_file(
                source_path, white_balance, exposure_stops, formats, exports,
                tone, sharpen,
            )

        result: Dict[str, ValueTypes] = {
            "source_path": source_path,
            "exports": exports,
            "sidecar": sidecar,
            "ccm_applied": not self.corrector.is_identity,
            "color_space": "sRGB",
        }
        if include_preview:
            result["image_base64"] = linear_to_jpeg_base64(
                corrected, tone=tone, sharpen=sharpen
            )
            result["mime_type"] = CameraMimeType.JPEG.value
        return result

    def _write_sidecar_file(
        self,
        source_path: str,
        white_balance: Any,
        exposure_stops: float,
        formats: Sequence[str],
        exports: Mapping[str, str],
        tone: Optional[str] = None,
        sharpen: Optional[str] = None,
    ) -> str:
        """
        Write a ``<stem>.json`` sidecar next to the (untouched) source file
        recording exactly how it was developed - the non-destructive record that
        lets a capture be reproduced or re-exported later.
        """
        sidecar_path = os.path.splitext(source_path)[0] + ".json"
        record = {
            "source": os.path.basename(source_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "white_balance": white_balance,
            "exposure_stops": exposure_stops,
            "tone": tone or "none",
            "sharpen": sharpen or "none",
            "demosaic": self._demosaic,
            "ccm": self.corrector.ccm.tolist(),
            "ccm_applied": not self.corrector.is_identity,
            "color_space": "sRGB",
            "output_formats": list(formats),
            "exports": {k: os.path.basename(v) for k, v in exports.items()},
        }
        with open(sidecar_path, "w") as f:
            json.dump(record, f, indent=2)
        return sidecar_path

    async def _get_data_client(self) -> DataClient:
        """
        Lazily build (and cache) a cloud ``DataClient`` from the API key Viam
        injects into the module process (``VIAM_API_KEY`` / ``VIAM_API_KEY_ID``).

        We dial the app channel directly rather than via
        ``ViamClient.create_from_env_vars``: in viam-sdk 0.77.0 that path
        authenticates the channel inside ``_dial_app`` and then authenticates a
        *second* time, which the server rejects with "already authenticated;
        cannot re-authenticate". ``_dial_app`` alone performs the single, correct
        auth, and the resulting channel carries the bearer token we hand to the
        ``DataClient``.
        """
        if self._data_client is not None:
            return self._data_client
        api_key = os.environ.get("VIAM_API_KEY")
        api_key_id = os.environ.get("VIAM_API_KEY_ID")
        if not api_key or not api_key_id:
            raise ValueError(
                "`upload` could not authenticate: VIAM_API_KEY / VIAM_API_KEY_ID "
                "were not present in the module environment. This requires "
                "running on a cloud-connected machine."
            )
        dial_options = DialOptions(
            credentials=Credentials(type="api-key", payload=api_key),
            auth_entity=api_key_id,
        )
        self.logger.info(
            "`upload` authenticating to app.viam.com "
            f"(dial timeout {self._upload_dial_timeout_s:.0f}s)"
        )
        t_dial = time.perf_counter()
        try:
            channel = await asyncio.wait_for(
                _dial_app("app.viam.com", dial_options),
                timeout=self._upload_dial_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"`upload` timed out after {self._upload_dial_timeout_s:.0f}s "
                "dialing app.viam.com to authenticate — the machine may be "
                "offline or unable to reach the cloud."
            ) from exc
        self.logger.info(
            f"`upload` authenticated in {time.perf_counter() - t_dial:.2f}s"
        )
        metadata = getattr(channel, "_metadata", {})
        self._data_client = DataClient(channel, metadata)
        return self._data_client

    async def _upload(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Upload files already on disk to Viam, tagged for later retrieval.

        The full-resolution captures (CR3 + the rendered TIFF/JPEG exports + the
        JSON sidecar) live on this machine's filesystem; this ships them straight
        to the cloud so they never have to travel back through the browser. The
        webapp passes every path that shares a capture's filename stem, so a
        single selected shot uploads as a complete set under one SKU tag.

        ``opts``:
          ``paths``               list of file paths on disk to upload (required)
          ``tags``                tags to attach to every uploaded file (e.g. SKU)
          ``name``                operator-chosen file name stem (no dir, no
                                  extension) that replaces the camera capture
                                  stem on every file in this set; each file keeps
                                  its full post-stem suffix (``_16.png``,
                                  ``.cr3``, ``.json``, ...). When absent, the
                                  camera's on-disk basename is used.
          ``sku``                 product code for Nines delivery: when set (and
                                  ``nines_api_key`` plus an org slug are
                                  available), the Nines product with this
                                  ``external_id`` is upserted and the set's
                                  delivery image (the full-res JPEG, by
                                  preference) is appended to it. Reported under
                                  ``nines`` in the response. A Nines failure
                                  never marks the Viam uploads as failed, and
                                  keeps the delivery image on disk for retry
                                  even with ``delete_after_upload``.
          ``shots_organization_slug``
                                  deliver to this Nines org instead of the
                                  configured ``nines_organization_slug`` (so one
                                  machine can serve multiple orgs); falls back to
                                  the config slug when absent
          ``part_id``             override the configured / env machine part id
          ``component_name``      camera name to associate the data with (optional)
          ``delete_after_upload`` override the config attribute: remove each
                                  local file once its upload succeeds (failed
                                  uploads keep their files for retry)
        """
        raw_paths = opts.get("paths") or []
        if not raw_paths:
            raise ValueError("`upload` needs a non-empty `paths` list")
        paths = [str(p) for p in raw_paths]
        tags = [str(t) for t in (opts.get("tags") or [])]
        delete_after = bool(opts.get("delete_after_upload", self._delete_after_upload))
        sku = str(opts.get("sku") or "").strip() or None
        org_slug = str(opts.get("shots_organization_slug") or "").strip() or None

        name = opts.get("name")
        name = str(name) if name else None      # falsy/empty -> keep current behavior
        # All paths in one upload call share the capture stem (e.g. "IMG_0042").
        # commonprefix over splitext-stems lands cleanly on the bare stem even
        # when _16 variants and a .json sidecar are mixed in - a basename-level
        # commonprefix would land mid-token on extension-only sets.
        capture_stem = (
            os.path.commonprefix([os.path.splitext(os.path.basename(p))[0] for p in paths])
            if name else None
        )

        part_id = opts.get("part_id") or self._part_id
        if not part_id:
            raise ValueError(
                "no part id available for upload; set `part_id` in config or pass "
                "it in the command (VIAM_MACHINE_PART_ID was not set)"
            )
        component_name = opts.get("component_name") or self.name

        data_client = await self._get_data_client()

        uploaded: List[str] = []
        failed: List[Dict[str, str]] = []
        for i, path in enumerate(paths):
            try:
                size = os.path.getsize(path)
                self.logger.info(
                    f"uploading {os.path.basename(path)} ({size / 1e6:.1f} MB) "
                    f"[{i + 1}/{len(paths)}] (timeout {self._upload_file_timeout_s:.0f}s)"
                )
                t_upload = time.perf_counter()
                file_name = (
                    self._renamed_basename(path, name, capture_stem)
                    if name else None
                )
                await asyncio.wait_for(
                    self._file_upload_chunked(
                        data_client,
                        path,
                        part_id=str(part_id),
                        component_name=str(component_name),
                        tags=tags or None,
                        file_name=file_name,
                    ),
                    timeout=self._upload_file_timeout_s,
                )
                self.logger.info(
                    f"[timing] uploaded {os.path.basename(path)} "
                    f"({size / 1e6:.1f} MB) in {time.perf_counter() - t_upload:.2f}s"
                )
                uploaded.append(path)
            except asyncio.TimeoutError:
                # File kept for retry — a per-file deadline means one stalled
                # transfer can't wedge the whole submit.
                msg = (
                    f"upload timed out after {self._upload_file_timeout_s:.0f}s"
                )
                self.logger.error(f"failed to upload {path}: {msg}")
                failed.append({"path": path, "error": msg})
            except Exception as exc:  # noqa: BLE001 - report per-file, keep going
                self.logger.error(f"failed to upload {path}: {exc}")
                failed.append({"path": path, "error": str(exc)})

        # Nines delivery runs before the delete pass so `delete_after_upload`
        # can't remove the delivery image out from under it. It's independent
        # of the Viam results: an archival failure doesn't block delivery, and
        # a delivery failure is reported under `nines`, never in `failed`.
        nines: Optional[Dict[str, ValueTypes]] = None
        nines_keep: Optional[str] = None
        if sku:
            nines, nines_keep = await self._nines_deliver_for_upload(
                sku, paths, name, capture_stem, org_slug=org_slug
            )

        deleted: List[str] = []
        if delete_after:
            for path in uploaded:
                if path == nines_keep:
                    self.logger.info(
                        f"keeping {os.path.basename(path)} on disk for a Nines "
                        "retry despite delete_after_upload"
                    )
                    continue
                try:
                    os.remove(path)
                    deleted.append(path)
                except OSError as exc:
                    # The upload itself succeeded - don't let a cleanup
                    # hiccup mark the file as failed.
                    self.logger.warning(f"uploaded but could not delete {path}: {exc}")

        self.logger.info(
            f"uploaded {len(uploaded)}/{len(paths)} file(s)"
            + (f" with tags {tags}" if tags else "")
            + (f", deleted {len(deleted)} local cop(ies)" if delete_after else "")
        )
        result: Dict[str, ValueTypes] = {
            "uploaded": uploaded,
            "count": len(uploaded),
            "failed": failed,
            "deleted": deleted,
        }
        if nines is not None:
            result["nines"] = nines
        return result

    def _delete_local(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Delete files from this machine's disk - the cleanup half of the studio
        flow. Captures the operator skipped (never developed or uploaded) have
        no other exit path and would otherwise accumulate in the download dir
        forever; the webapp sends their paths here at submit time.

        Guarded: only files inside the configured ``output_dir`` may be
        deleted, so a caller can't reach arbitrary paths on the host. Requires
        ``output_dir`` to be set (without it there is no boundary to enforce).

        ``opts``:
          ``paths``  list of file paths to delete (required)

        Already-missing files are reported in ``missing`` rather than treated
        as errors, so retried cleanups stay idempotent.
        """
        raw_paths = opts.get("paths") or []
        if not raw_paths:
            raise ValueError("`delete` needs a non-empty `paths` list")
        if not self._output_dir:
            raise ValueError(
                "`delete` requires `output_dir` to be configured: it only "
                "removes files inside that directory"
            )
        root = os.path.realpath(self._output_dir)

        deleted: List[str] = []
        missing: List[str] = []
        failed: List[Dict[str, str]] = []
        for raw in raw_paths:
            path = str(raw)
            # realpath also resolves symlinks, so a link inside output_dir
            # pointing elsewhere can't smuggle a delete outside the boundary.
            real = os.path.realpath(path)
            if os.path.commonpath([real, root]) != root:
                self.logger.warning(f"refusing to delete outside output_dir: {path}")
                failed.append(
                    {"path": path, "error": f"outside output_dir {self._output_dir}"}
                )
                continue
            try:
                os.remove(real)
                deleted.append(path)
            except FileNotFoundError:
                missing.append(path)
            except OSError as exc:
                self.logger.error(f"failed to delete {path}: {exc}")
                failed.append({"path": path, "error": str(exc)})

        self.logger.info(
            f"deleted {len(deleted)}/{len(raw_paths)} local file(s)"
            + (f", {len(missing)} already gone" if missing else "")
        )
        return {
            "deleted": deleted,
            "count": len(deleted),
            "missing": missing,
            "failed": failed,
        }

    @staticmethod
    async def _file_upload_chunked(
        data_client: DataClient,
        path: str,
        *,
        part_id: str,
        component_name: str,
        tags: Optional[List[str]],
        file_name: Optional[str] = None,
    ) -> str:
        """
        Stream a file to Viam over the client-streaming FileUpload RPC in
        UPLOAD_CHUNK_BYTES pieces. ``DataClient.file_upload`` sends the whole
        file as one gRPC message, which app.viam.com rejects past 32 MiB -
        silently dropping every CR3 and TIFF from a submit. Chunking also
        keeps a 250 MB TIFF from being held in memory all at once.
        """
        metadata = UploadMetadata(
            part_id=part_id,
            component_type="rdk:component:camera",
            component_name=component_name,
            type=DataType.DATA_TYPE_FILE,
            file_name=file_name or os.path.basename(path),
            file_extension=os.path.splitext(path)[1],  # e.g. ".cr3", ".tif"
            tags=tags,
        )
        async with data_client._data_sync_client.FileUpload.open(  # noqa: SLF001
            metadata=data_client._metadata  # noqa: SLF001
        ) as stream:
            await stream.send_message(FileUploadRequest(metadata=metadata))
            with open(path, "rb") as f:
                chunk = f.read(UPLOAD_CHUNK_BYTES)
                while True:
                    next_chunk = f.read(UPLOAD_CHUNK_BYTES)
                    # An empty file still needs one (empty) FileData message.
                    await stream.send_message(
                        FileUploadRequest(file_contents=FileData(data=chunk)),
                        end=not next_chunk,
                    )
                    if not next_chunk:
                        break
                    chunk = next_chunk
            response: Optional[FileUploadResponse] = await stream.recv_message()
            if not response:
                # Raises the appropriate gRPC error for the failed upload.
                await stream.recv_trailing_metadata()
                raise TypeError("FileUpload response cannot be empty")
            return response.binary_data_id

    # ------------------------------------------------------------------
    # Nines partner-API delivery
    # ------------------------------------------------------------------

    @staticmethod
    def _renamed_basename(
        path: str, name: Optional[str], capture_stem: Optional[str]
    ) -> str:
        """
        Apply the operator-chosen upload ``name`` to one file of a capture set:
        the shared ``capture_stem`` prefix is swapped for ``name``, preserving
        the file's post-stem suffix (``_16.tif``, ``.json``, ...). With no
        ``name`` the on-disk basename is returned unchanged. Used for both the
        Viam cloud file name and the Nines image filename, so the two sides
        agree on what a shot is called.
        """
        base = os.path.basename(path)
        if not name or capture_stem is None:
            return base
        return name + base[len(capture_stem):]

    def _nines_ready(self, org_slug: Optional[str]) -> bool:
        """Whether Nines delivery can proceed for the given effective org slug.
        Needs an API key plus an org slug from *somewhere* - the per-request
        ``org_slug`` when the webapp supplies one, else the configured slug. A
        machine configured with only a key can still serve any org the webapp
        names; that's what lets one machine deliver to multiple orgs."""
        return bool(self._nines_api_key and (org_slug or self._nines_org_slug))

    @staticmethod
    def _nines_pick_image(paths: Sequence[str]) -> Optional[str]:
        """
        Choose the one file of a capture set to deliver to Nines. The partner
        API wants exactly one full-resolution original per view and accepts
        only jpeg/png/webp/gif - so out of a set that also carries the RAW
        master, TIFFs, and the sidecar, prefer the full-res JPEG, then the
        8-bit PNG, then the 16-bit PNG, then webp/gif. Returns ``None`` when
        nothing in the set is eligible.
        """
        def rank(path: str) -> Optional[Tuple[int, str]]:
            ext = os.path.splitext(path)[1].lower()
            if ext not in NINES_CONTENT_TYPES:
                return None
            if ext in (".jpg", ".jpeg"):
                order = 0
            elif ext == ".png":
                order = 2 if path.lower().endswith("_16.png") else 1
            elif ext == ".webp":
                order = 3
            else:  # .gif
                order = 4
            return order, path

        ranked = sorted(r for r in map(rank, paths) if r is not None)
        return ranked[0][1] if ranked else None

    def _nines_request(
        self, method: str, path: str, body: Mapping[str, Any], timeout_s: float
    ) -> Dict[str, Any]:
        """
        One JSON request to the Nines partner API. Synchronous (urllib) - call
        it via ``asyncio.to_thread``. Raises :class:`NinesAPIError` carrying
        the HTTP status and the API's ``error`` description on a non-2xx
        response, or without a status when the API was unreachable.
        """
        request = urllib.request.Request(
            f"{self._nines_base_url}{path}",
            data=json.dumps(body).encode(),
            method=method,
            headers={
                "Authorization": f"Bearer {self._nines_api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode()).get("error", "")
            except Exception:  # noqa: BLE001 - error bodies aren't guaranteed JSON
                pass
            raise NinesAPIError(
                f"Nines API {method} {path} failed with {exc.code}"
                + (f": {detail}" if detail else ""),
                status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise NinesAPIError(
                f"Nines API {method} {path} unreachable: {exc.reason}"
            ) from exc

    async def _nines_upsert_item(
        self, sku: str, product_name: Optional[str], org_slug: Optional[str] = None
    ) -> str:
        """
        Upsert the Nines reference item whose ``external_id`` is ``sku`` in the
        effective org (``org_slug`` when given, else the configured slug) and
        cache its id under ``(org, sku)``. Deliberately sends no ``images``
        field - on an existing product that would *replace* all of its imagery;
        appending happens through the non-destructive images endpoint only.
        """
        org = org_slug or self._nines_org_slug
        response = await asyncio.to_thread(
            self._nines_request,
            "POST",
            "/api/v1/reference_items",
            {
                "shots_organization_slug": org,
                "name": product_name or sku,
                "external_id": sku,
            },
            self._upload_dial_timeout_s,
        )
        item_id = str(response.get("id") or "")
        if not item_id:
            raise NinesAPIError("Nines upsert returned no reference item id")
        self._nines_item_ids[(org, sku)] = item_id
        self.logger.info(
            f"Nines reference item {item_id} "
            f"({'created' if response.get('created') else 'updated'}) "
            f"for SKU {sku!r} in org {org!r}"
        )
        return item_id

    async def _nines_deliver(
        self,
        sku: str,
        images: Sequence[Tuple[str, str, List[str]]],
        product_name: Optional[str] = None,
        org_slug: Optional[str] = None,
    ) -> Dict[str, ValueTypes]:
        """
        Deliver on-disk image files to the Nines product identified by ``sku``
        in the effective org (``org_slug`` when given, else the configured
        slug): upsert the reference item (once per ``(org, SKU)`` since the last
        reconfigure), then append every image non-destructively as inline
        base64. ``images`` is ``[(path, upload_filename, tags)]``; every file
        must carry a jpeg/png/webp/gif extension. Raises on any API failure -
        callers decide whether that fails their operation.
        """
        org = org_slug or self._nines_org_slug
        cached = (org, sku) in self._nines_item_ids
        item_id = self._nines_item_ids.get((org, sku)) or await self._nines_upsert_item(
            sku, product_name, org_slug=org
        )

        payload: List[Dict[str, Any]] = []
        for path, filename, tags in images:
            image: Dict[str, Any] = {
                "data": await asyncio.to_thread(_read_base64, path),
                "filename": filename,
                "content_type": NINES_CONTENT_TYPES[os.path.splitext(path)[1].lower()],
            }
            if tags:
                image["tags"] = tags
            payload.append(image)

        async def append(rid: str) -> Dict[str, Any]:
            return await asyncio.to_thread(
                self._nines_request,
                "POST",
                f"/api/v1/reference_items/{rid}/images",
                {"shots_organization_slug": org, "images": payload},
                self._upload_file_timeout_s,
            )

        try:
            response = await append(item_id)
        except NinesAPIError as exc:
            # A cached id can go stale (product deleted on the Nines side);
            # re-upsert once and retry rather than wedging every later shot of
            # the session on the dead id.
            if not (cached and exc.status == 404):
                raise
            self.logger.warning(
                f"cached Nines item {item_id} for SKU {sku!r} in org {org!r} is "
                "gone (404); re-upserting and retrying"
            )
            self._nines_item_ids.pop((org, sku), None)
            item_id = await self._nines_upsert_item(sku, product_name, org_slug=org)
            response = await append(item_id)

        return {
            "reference_item_id": item_id,
            "external_id": sku,
            "added_count": response.get("added_count"),
            "images_count": response.get("images_count"),
        }

    async def _nines_deliver_for_upload(
        self,
        sku: str,
        paths: Sequence[str],
        name: Optional[str],
        capture_stem: Optional[str],
        org_slug: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, ValueTypes]], Optional[str]]:
        """
        The ``upload``-integrated Nines delivery: pick the one delivery image
        out of the capture set and append it to the SKU's product in the
        effective org (``org_slug`` when the webapp names one, else the
        configured slug), tagged with its final filename stem. Returns
        ``(nines_result, keep_path)`` where ``keep_path`` names a file the
        delete pass must leave on disk for a retry (the delivery image, when
        delivery failed). Never raises: a Nines problem is reported in the
        result, not allowed to fail the Viam half of the submit.
        """
        org = org_slug or self._nines_org_slug
        if not self._nines_ready(org):
            self.logger.info(
                f"`upload` got sku {sku!r} but Nines delivery is not configured"
            )
            return {
                "skipped": "Nines delivery not configured: set `nines_api_key` "
                           "and an org slug (config `nines_organization_slug` "
                           "or a per-request `shots_organization_slug`)"
            }, None

        delivery = self._nines_pick_image(paths)
        if delivery is None:
            return {
                "error": "no Nines-compatible image (jpeg/png/webp/gif) in "
                         "this upload set"
            }, None

        filename = self._renamed_basename(delivery, name, capture_stem)
        try:
            result = await self._nines_deliver(
                sku, [(delivery, filename, [os.path.splitext(filename)[0]])],
                org_slug=org,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fails the upload
            self.logger.error(f"Nines delivery failed for SKU {sku!r}: {exc}")
            return {"error": str(exc)}, delivery
        self.logger.info(
            f"delivered {filename} to Nines item "
            f"{result.get('reference_item_id')} (SKU {sku!r}, "
            f"{result.get('images_count')} image(s) total)"
        )
        return result, None

    async def _nines_upload(self, opts: Mapping[str, Any]) -> Mapping[str, ValueTypes]:
        """
        Deliver image files already on disk to the Nines partner API - the
        manual / retry counterpart to the ``sku`` option on ``upload``. Sends
        exactly the files listed (no best-of-set picking, no Viam upload, no
        local deletion), appended to the SKU's product non-destructively.

        ``opts``:
          ``sku``                       product code upserted as the Nines
                                        ``external_id`` (required)
          ``paths``                     image files to append; each must be
                                        jpeg/png/webp/gif (required)
          ``shots_organization_slug``   deliver to this org instead of the
                                        configured one, so a webapp retry lands
                                        in the same org as the original submit;
                                        falls back to the config slug
          ``tags``                      Nines tags applied to every appended
                                        image (e.g. ["front"])
          ``product_name``              product display name used if the SKU
                                        doesn't exist on the Nines side yet
                                        (default: the sku)

        Requires ``nines_api_key`` plus an org slug - from config
        (``nines_organization_slug``) or the per-request
        ``shots_organization_slug``. Returns ``{"reference_item_id",
        "external_id", "added_count", "images_count"}``.
        """
        sku = str(opts.get("sku") or "").strip()
        if not sku:
            raise ValueError("`nines_upload` needs a `sku`")
        raw_paths = opts.get("paths") or []
        if not raw_paths:
            raise ValueError("`nines_upload` needs a non-empty `paths` list")
        org_slug = str(opts.get("shots_organization_slug") or "").strip() or None
        if not self._nines_ready(org_slug):
            raise ValueError(
                "Nines delivery is not configured: set the `nines_api_key` "
                "config attribute (the key may also come from the NINES_API_KEY "
                "env var) and an org slug - `nines_organization_slug` in config "
                "or a per-request `shots_organization_slug`"
            )
        paths = [str(p) for p in raw_paths]
        ineligible = [
            p for p in paths
            if os.path.splitext(p)[1].lower() not in NINES_CONTENT_TYPES
        ]
        if ineligible:
            raise ValueError(
                "not Nines-compatible (the API accepts jpeg/png/webp/gif): "
                f"{ineligible}"
            )
        tags = [str(t) for t in (opts.get("tags") or [])]
        product_name = opts.get("product_name")
        return await self._nines_deliver(
            sku,
            [(p, os.path.basename(p), tags) for p in paths],
            product_name=str(product_name) if product_name else None,
            org_slug=org_slug,
        )

    async def get_geometries(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Sequence[Geometry]:
        return await self.camera.get_geometries(extra=extra, timeout=timeout)
