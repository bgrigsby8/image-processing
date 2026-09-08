"""
distortion.py
-------------
Lens distortion + lateral chromatic-aberration correction for the RAW develop
pipeline: load/save a calibration file, build the remap tables, and apply
them to a linear RGB frame.

Why this exists: we shoot wide (16 mm) and crop the product out of the frame
centre. Sony corrects the lens in-camera for JPEGs and embeds the profile in
the ARW, but rawpy/LibRaw does not apply it, so the RAW pipeline sees the
uncorrected lens - a half-frame crop still carries ~0.5-1 % barrel, several
pixels of bow along a straight edge. A one-time OpenCV checkerboard
calibration (``calibrate_distortion.py``) produces the JSON file this module
consumes.

Coordinate frame - the one rule everything here depends on:

    Calibrate and apply in exactly the same pixel frame: the output of
    ``rawpy.postprocess`` at full size with ``user_flip=0`` (sensor
    orientation, no EXIF rotation). ``image_io.load_linear_rgb`` decodes in
    that frame, runs the ``sensor_transform`` hook (this module's
    ``Undistorter.apply``), and only then applies the EXIF rotation. The
    camera hangs off an arm whose orientation sensor could flip the EXIF
    orientation between shots; the distortion map is in sensor coordinates
    and must not be rotated underneath us.

Output framing: undistort with ``newCameraMatrix = K`` (the calibrated
intrinsics) at the same output size. For barrel distortion that yields a
fully valid output with the centre scale unchanged, so a normalized centre
crop covers the same physical field of view before and after. Deliberately
*not* ``getOptimalNewCameraMatrix``: it rescales the frame and would silently
move every crop rect the webapp has stored.

Lateral CA: a constrained radial model fitted from corner displacements
between the G corners (geometric reference) and the R / B corners:

    d_c(p) = (p - c0) * (a0 + a1*rho^2 + a2*rho^4),   rho = |p - c0| / r_norm

with c0 the principal point from K and r_norm half the image diagonal. Each
colour channel gets its own remap: ``map_c = map_G + d_c(map_G)`` - d_c is
evaluated at the *source* (distorted) position, because that is where the
corners were measured.

No Viam imports here, so the CLI and the tests can use it without a module
context.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

SCHEMA_VERSION = 1

#: Interpolation choices for the remap. Resampling happens on 16-bit linear
#: data (the right place for it). Cubic is the default: sharper than linear,
#: far cheaper than Lanczos and without its ringing on hard product edges.
INTERPOLATIONS: Dict[str, int] = {
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
    "linear": cv2.INTER_LINEAR,
}

#: OpenCV distortion models by coefficient count. Both are Brown-Conrady
#: pinhole models; "rational" adds k4-k6 (CALIB_RATIONAL_MODEL). Not the
#: fisheye model - this is a rectilinear lens.
MODEL_COEFFS: Dict[str, int] = {"opencv_standard": 5, "opencv_rational": 8}

_CHANNELS = ("R", "G", "B")


class CalibrationError(ValueError):
    """The calibration file is missing, unparseable, or fails validation."""


class CalibrationMismatch(RuntimeError):
    """The frame being developed does not match the calibration (size, or a
    metadata mismatch under ``strict_calibration_match``)."""


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Lateral chromatic aberration model
# ---------------------------------------------------------------------------

def lateral_ca_shift(
    coeffs: Sequence[float],
    pts: np.ndarray,
    principal_point: Sequence[float],
    r_norm: float,
) -> np.ndarray:
    """
    Evaluate ``d_c(p) = (p - c0) * (a0 + a1*rho^2 + a2*rho^4)`` at ``pts``
    (``(..., 2)`` or a pair of same-shape x/y arrays stacked on the last
    axis). Returns the per-point shift with the same shape. Works on the
    full-frame map grids as well as on corner lists.
    """
    a0, a1, a2 = (float(c) for c in coeffs)
    cx, cy = (float(v) for v in principal_point)
    pts = np.asarray(pts, dtype=np.float64)
    dx = pts[..., 0] - cx
    dy = pts[..., 1] - cy
    rho2 = (dx * dx + dy * dy) / (float(r_norm) ** 2)
    gain = a0 + a1 * rho2 + a2 * rho2 * rho2
    return np.stack([dx * gain, dy * gain], axis=-1)


def fit_lateral_ca(
    corners_g: np.ndarray,
    corners_c: np.ndarray,
    principal_point: Sequence[float],
    r_norm: float,
) -> Tuple[Tuple[float, float, float], float]:
    """
    Fit ``(a0, a1, a2)`` for one colour channel by linear least squares over
    matched corners: ``corners_c - corners_g = d_c(corners_g)``. Both the x
    and the y equation of every corner enter the fit. Returns the coefficients
    and the RMS residual in px (root mean of the squared 2-D residual norm).

    Fitted directly from displacements rather than from three independent
    ``calibrateCamera`` runs: on a planar target focal length and distance
    are nearly degenerate, so per-channel intrinsics would be noisy, whereas
    the *difference* between channels is well conditioned.
    """
    g = np.asarray(corners_g, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(corners_c, dtype=np.float64).reshape(-1, 2)
    if g.shape != c.shape:
        raise ValueError(
            f"corner sets differ in shape: G {g.shape} vs channel {c.shape}"
        )
    if g.shape[0] < 3:
        raise ValueError("fit_lateral_ca needs at least 3 matched corners")
    cx, cy = (float(v) for v in principal_point)
    dx = g[:, 0] - cx
    dy = g[:, 1] - cy
    rho2 = (dx * dx + dy * dy) / (float(r_norm) ** 2)
    basis = np.stack([np.ones_like(rho2), rho2, rho2 * rho2], axis=1)
    design = np.vstack([basis * dx[:, None], basis * dy[:, None]])
    target = np.concatenate([c[:, 0] - g[:, 0], c[:, 1] - g[:, 1]])
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = (design @ coeffs - target).reshape(2, -1)
    rms = float(np.sqrt(np.mean(residual[0] ** 2 + residual[1] ** 2)))
    return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2])), rms


@dataclass(frozen=True)
class LateralCA:
    r_norm: float
    coeffs_r: Tuple[float, float, float]
    coeffs_b: Tuple[float, float, float]
    #: Fit diagnostics carried through the file (rms residual, corner shift);
    #: opaque to the maths.
    stats: Optional[Dict[str, Dict[str, float]]] = None

    def coeffs(self, channel: str) -> Tuple[float, float, float]:
        if channel == "R":
            return self.coeffs_r
        if channel == "B":
            return self.coeffs_b
        raise ValueError(f"lateral CA is defined for R and B, not {channel!r}")

    def shift(
        self, channel: str, pts: np.ndarray, principal_point: Sequence[float]
    ) -> np.ndarray:
        return lateral_ca_shift(self.coeffs(channel), pts, principal_point, self.r_norm)

    def max_shift_px(
        self, channel: str, image_size: Sequence[int], principal_point: Sequence[float]
    ) -> float:
        """The implied shift at the worst image corner - the "how much CA
        did this lens have" number for the report."""
        w, h = image_size
        corners = np.array(
            [[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]]
        )
        return float(np.linalg.norm(self.shift(channel, corners, principal_point), axis=1).max())

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"r_norm": float(self.r_norm)}
        for name, coeffs in (("R", self.coeffs_r), ("B", self.coeffs_b)):
            entry: Dict[str, Any] = {"coeffs": [float(c) for c in coeffs]}
            if self.stats and name in self.stats:
                entry.update({k: float(v) for k, v in self.stats[name].items()})
            out[name] = entry
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LateralCA":
        try:
            r_norm = float(d["r_norm"])
            channels = {}
            stats: Dict[str, Dict[str, float]] = {}
            for name in ("R", "B"):
                entry = d[name]
                coeffs = [float(c) for c in entry["coeffs"]]
                if len(coeffs) != 3:
                    raise CalibrationError(
                        f"lateral_ca.{name}.coeffs must have 3 values, got {len(coeffs)}"
                    )
                channels[name] = (coeffs[0], coeffs[1], coeffs[2])
                extra = {
                    k: float(v) for k, v in entry.items()
                    if k != "coeffs" and isinstance(v, (int, float))
                }
                if extra:
                    stats[name] = extra
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"malformed lateral_ca block: {exc}") from exc
        if r_norm <= 0:
            raise CalibrationError("lateral_ca.r_norm must be positive")
        return cls(r_norm=r_norm, coeffs_r=channels["R"], coeffs_b=channels["B"],
                   stats=stats or None)


# ---------------------------------------------------------------------------
# Calibration file
# ---------------------------------------------------------------------------

def _normalize_lens(name: Any) -> str:
    return " ".join(str(name).split()).lower()


@dataclass(frozen=True)
class DistortionCalibration:
    image_size: Tuple[int, int]          # (w, h) of the rawpy user_flip=0 frame
    camera_matrix: np.ndarray            # 3x3 float64
    dist_coeffs: np.ndarray              # 5 (standard) or 8 (rational) float64
    lateral_ca: Optional[LateralCA]
    #: Everything else in the file - camera, develop, quality, sources,
    #: created_at... Carried verbatim; opaque to the maths.
    meta: Dict[str, Any]

    def __post_init__(self):
        w, h = (int(v) for v in self.image_size)
        if w <= 0 or h <= 0:
            raise CalibrationError(f"image_size must be positive, got {self.image_size}")
        k = np.asarray(self.camera_matrix, dtype=np.float64)
        if k.shape != (3, 3):
            raise CalibrationError(f"camera_matrix must be 3x3, got {k.shape}")
        d = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1)
        if d.size not in MODEL_COEFFS.values():
            raise CalibrationError(
                f"dist_coeffs must have 5 (standard) or 8 (rational) values, got {d.size}"
            )
        if not (np.all(np.isfinite(k)) and np.all(np.isfinite(d))):
            raise CalibrationError("camera_matrix / dist_coeffs contain non-finite values")
        object.__setattr__(self, "image_size", (w, h))
        object.__setattr__(self, "camera_matrix", k)
        object.__setattr__(self, "dist_coeffs", d)

    # -- derived -----------------------------------------------------------

    @property
    def model(self) -> str:
        for name, n in MODEL_COEFFS.items():
            if n == self.dist_coeffs.size:
                return name
        raise AssertionError("unreachable: validated in __post_init__")

    @property
    def principal_point(self) -> Tuple[float, float]:
        return float(self.camera_matrix[0, 2]), float(self.camera_matrix[1, 2])

    @property
    def r_norm(self) -> float:
        """Half the image diagonal - the CA model's radius normaliser."""
        w, h = self.image_size
        return float(np.hypot(w, h) / 2.0)

    def camera_meta(self) -> Dict[str, Any]:
        cam = self.meta.get("camera")
        return dict(cam) if isinstance(cam, Mapping) else {}

    def quality_meta(self) -> Dict[str, Any]:
        q = self.meta.get("quality")
        return dict(q) if isinstance(q, Mapping) else {}

    # -- (de)serialisation --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Build the JSON document. Key order follows the documented schema so
        a saved file reads top-down: provenance, camera, geometry, quality."""
        out: Dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        for key in ("created_at", "tool", "camera", "develop"):
            if key in self.meta:
                out[key] = self.meta[key]
        out["image_size"] = [int(self.image_size[0]), int(self.image_size[1])]
        out["model"] = self.model
        out["camera_matrix"] = self.camera_matrix.tolist()
        out["dist_coeffs"] = self.dist_coeffs.tolist()
        out["lateral_ca"] = self.lateral_ca.to_dict() if self.lateral_ca else None
        for key, value in self.meta.items():
            if key not in out:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DistortionCalibration":
        if not isinstance(d, Mapping):
            raise CalibrationError("calibration must be a JSON object")
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise CalibrationError(
                f"unsupported calibration schema_version {version!r} "
                f"(this build reads {SCHEMA_VERSION})"
            )
        missing = [k for k in ("image_size", "camera_matrix", "dist_coeffs") if k not in d]
        if missing:
            raise CalibrationError(f"calibration is missing {missing}")
        try:
            size = tuple(int(v) for v in d["image_size"])
            if len(size) != 2:
                raise CalibrationError("image_size must be [width, height]")
            k = np.asarray(d["camera_matrix"], dtype=np.float64)
            dist = np.asarray(d["dist_coeffs"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"malformed geometry: {exc}") from exc
        model = d.get("model")
        if model is not None and model not in MODEL_COEFFS:
            raise CalibrationError(
                f"unknown model {model!r}; valid: {sorted(MODEL_COEFFS)}"
            )
        if model is not None and MODEL_COEFFS[model] != dist.size:
            raise CalibrationError(
                f"model {model!r} needs {MODEL_COEFFS[model]} dist_coeffs, file has {dist.size}"
            )
        ca_raw = d.get("lateral_ca")
        lateral_ca = LateralCA.from_dict(ca_raw) if ca_raw else None
        meta = {
            k: v for k, v in d.items()
            if k not in ("schema_version", "image_size", "model", "camera_matrix",
                         "dist_coeffs", "lateral_ca")
        }
        return cls(image_size=size, camera_matrix=k, dist_coeffs=dist,
                   lateral_ca=lateral_ca, meta=meta)

    @classmethod
    def load(cls, path: str) -> "DistortionCalibration":
        if not os.path.isfile(path):
            raise CalibrationError(f"calibration file not found: {path}")
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            raise CalibrationError(f"cannot read calibration {path}: {exc}") from exc
        return cls.from_dict(data)

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    # -- matching -----------------------------------------------------------

    def check_match(
        self,
        *,
        image_size: Optional[Sequence[int]],
        lens: Optional[str] = None,
        zoom_position: Optional[int] = None,
        focus_position: Optional[int] = None,
    ) -> List[str]:
        """
        Human-readable mismatch messages; an empty list is a clean match. A
        size mismatch is always first (it is the one that makes the maps
        inapplicable - callers treat it as a hard error, the rest as
        warnings). Metadata that is ``None`` on either side is not compared:
        an old file developed without its capture record simply isn't
        checked.
        """
        msgs: List[str] = []
        if image_size is not None:
            w, h = (int(v) for v in image_size)
            if (w, h) != self.image_size:
                msgs.append(
                    f"image size {w}x{h} does not match the calibration's "
                    f"{self.image_size[0]}x{self.image_size[1]}; the remap tables "
                    f"cannot apply (different camera, crop, half-size decode, or "
                    f"a LibRaw version with a different border crop?)"
                )
        cam = self.camera_meta()
        cal_lens = cam.get("lens")
        if lens is not None and cal_lens is not None and _normalize_lens(lens) != _normalize_lens(cal_lens):
            msgs.append(f"lens {lens!r} differs from the calibrated {cal_lens!r}")
        cal_zoom = cam.get("zoom_position")
        if zoom_position is not None and cal_zoom is not None and int(zoom_position) != int(cal_zoom):
            msgs.append(
                f"zoom_position {int(zoom_position)} differs from the calibrated "
                f"{int(cal_zoom)}; the calibration is only valid at the recorded "
                f"zoom setpoint"
            )
        cal_focus = cam.get("focus_position")
        if focus_position is not None and cal_focus is not None and int(focus_position) != int(cal_focus):
            msgs.append(
                f"focus_position {int(focus_position)} differs from the calibrated "
                f"{int(cal_focus)} (distortion changes mildly with focus distance)"
            )
        return msgs


# ---------------------------------------------------------------------------
# Point helpers (tests, the CLI's straight-line check)
# ---------------------------------------------------------------------------

def undistort_points(calib: DistortionCalibration, pts: np.ndarray) -> np.ndarray:
    """Distorted pixel coordinates -> ideal pixel coordinates in the same K
    frame (``P = K``, matching ``Undistorter``'s output framing)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    if p.shape[0] == 0:
        return np.zeros((0, 2))
    out = cv2.undistortPoints(
        p, calib.camera_matrix, calib.dist_coeffs, P=calib.camera_matrix
    )
    return out.reshape(-1, 2)


def distort_points(calib: DistortionCalibration, pts: np.ndarray) -> np.ndarray:
    """Ideal pixel coordinates -> where the lens actually images them. The
    forward model (``projectPoints`` of the normalised ray), used to build
    synthetic test frames and to verify ``undistort_points``."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] == 0:
        return np.zeros((0, 2))
    k = calib.camera_matrix
    x = (p[:, 0] - k[0, 2]) / k[0, 0]
    y = (p[:, 1] - k[1, 2]) / k[1, 1]
    rays = np.stack([x, y, np.ones_like(x)], axis=1).reshape(-1, 1, 3)
    zero = np.zeros(3)
    out, _ = cv2.projectPoints(rays, zero, zero, k, calib.dist_coeffs)
    return out.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Remap tables
# ---------------------------------------------------------------------------

def estimate_map_memory_bytes(
    image_size: Sequence[int], *, correct_ca: bool, fixed_point: bool
) -> int:
    """Footprint of the cached maps: one (x, y) pair per channel that gets its
    own map - just G (applied to all three) without CA, all three with it.
    float32 pairs are 8 bytes/px; ``convertMaps`` fixed point is 6."""
    w, h = (int(v) for v in image_size)
    per_pixel = 6 if fixed_point else 8
    return w * h * per_pixel * (3 if correct_ca else 1)


def build_maps(
    calib: DistortionCalibration,
    *,
    correct_ca: bool = True,
    fixed_point: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Remap tables per channel, ``{"R": (m1, m2), "G": ..., "B": ...}``. Without
    CA (or without a CA block in the file) all three point at the same G
    arrays. ``fixed_point`` runs ``cv2.convertMaps`` to ``CV_16SC2/CV_16UC1``
    (25 % less memory, faster remap, 1/32 px sample-position quantisation).
    """
    w, h = calib.image_size
    k = calib.camera_matrix
    # newCameraMatrix = K: same centre scale, same principal point, so the
    # centre crop covers the same physical field before and after.
    mx, my = cv2.initUndistortRectifyMap(
        k, calib.dist_coeffs, None, k, (w, h), cv2.CV_32FC1
    )

    def _finish(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if fixed_point:
            return cv2.convertMaps(x, y, cv2.CV_16SC2)
        return x, y

    g_maps = _finish(mx, my)
    maps = {"G": g_maps}
    if correct_ca and calib.lateral_ca is not None:
        cx, cy = calib.principal_point
        r2 = float(calib.lateral_ca.r_norm) ** 2
        # rho^2 at the *source* position (mx, my): the corners were detected in
        # the distorted frame, so d_c is defined there.
        dx = mx - np.float32(cx)
        dy = my - np.float32(cy)
        rho2 = (dx * dx + dy * dy) / np.float32(r2)
        for name in ("R", "B"):
            a0, a1, a2 = calib.lateral_ca.coeffs(name)
            gain = np.float32(a0) + np.float32(a1) * rho2 + np.float32(a2) * rho2 * rho2
            maps[name] = _finish(mx + dx * gain, my + dy * gain)
        del dx, dy, rho2
    else:
        maps["R"] = g_maps
        maps["B"] = g_maps
    return maps


class Undistorter:
    """
    Applies a ``DistortionCalibration`` to frames in the calibration frame.

    Maps are built lazily on the first ``apply`` and cached for the lifetime
    of the object (``cache_maps=True``): at 61 MP that is ~1.4 GB of float32
    maps with CA (~0.5 GB without), and a few seconds to build - comparable to
    the demosaic itself. ``cache_maps=False`` rebuilds on every call and frees
    afterwards, for a memory-constrained rig PC.
    """

    def __init__(
        self,
        calib: DistortionCalibration,
        *,
        interpolation: str = "cubic",
        correct_ca: bool = True,
        cache_maps: bool = True,
        fixed_point: bool = False,
    ):
        if interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"unknown interpolation {interpolation!r}; valid: {sorted(INTERPOLATIONS)}"
            )
        self.calib = calib
        self.interpolation = interpolation
        self.correct_ca = bool(correct_ca) and calib.lateral_ca is not None
        self.cache_maps = bool(cache_maps)
        self.fixed_point = bool(fixed_point)
        self._maps: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
        self.build_count = 0

    @property
    def corrects_lateral_ca(self) -> bool:
        """True when R and B get their own maps (CA requested *and* the file
        carries a CA block)."""
        return self.correct_ca

    @property
    def maps_cached(self) -> bool:
        return self._maps is not None

    @property
    def map_memory_bytes(self) -> int:
        """Footprint of the maps this instance holds (or will hold once
        built) - what the module logs at load."""
        return estimate_map_memory_bytes(
            self.calib.image_size, correct_ca=self.correct_ca, fixed_point=self.fixed_point
        )

    def _build_maps(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        self.build_count += 1
        return build_maps(
            self.calib, correct_ca=self.correct_ca, fixed_point=self.fixed_point
        )

    def _get_maps(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if self._maps is not None:
            return self._maps
        maps = self._build_maps()
        if self.cache_maps:
            self._maps = maps
        return maps

    def release(self) -> None:
        """Drop the cached maps (a reconfigure that changes options)."""
        self._maps = None

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Undistort (and CA-correct) ``img`` - ``HxWx3`` RGB or ``HxW`` in the
        calibration frame, ``uint16`` or ``float32`` linear. Returns the same
        shape and dtype. Raises ``CalibrationMismatch`` when the frame size
        differs from the calibration's (the maps cannot apply).
        """
        if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] != 3):
            raise ValueError(f"expected an HxW or HxWx3 image, got shape {img.shape}")
        if img.dtype not in (np.uint16, np.float32):
            raise TypeError(
                f"expected uint16 or float32 linear data, got {img.dtype}"
            )
        h, w = img.shape[:2]
        if (w, h) != self.calib.image_size:
            raise CalibrationMismatch(
                self.calib.check_match(image_size=(w, h))[0]
            )
        maps = self._get_maps()
        interp = INTERPOLATIONS[self.interpolation]
        try:
            if img.ndim == 2:
                out = self._remap(img, maps["G"], interp)
            else:
                channels = [
                    self._remap(np.ascontiguousarray(img[..., i]), maps[name], interp)
                    for i, name in enumerate(_CHANNELS)
                ]
                out = np.stack(channels, axis=-1)
        finally:
            if not self.cache_maps:
                del maps
        return out

    @staticmethod
    def _remap(
        plane: np.ndarray, maps: Tuple[np.ndarray, np.ndarray], interp: int
    ) -> np.ndarray:
        out = cv2.remap(
            plane, maps[0], maps[1], interp,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        if out.dtype == np.float32:
            # Cubic / Lanczos overshoot: uint16 saturates inside remap, so give
            # linear float the same [0, 1] clamp (negative light isn't a thing).
            np.clip(out, 0.0, 1.0, out=out)
        return out
