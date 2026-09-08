# Distortion calibration files

This directory holds the lens distortion + lateral chromatic aberration
calibrations that `color-correction` applies in `develop` / `capture` when its
`distortion_calibration` attribute points at one. They are produced on the rig
by `scripts/calibrate_distortion.py` from a folder of checkerboard ARWs (see the
[module README](../README.md#lens-distortion-calibration) for the shooting and
CLI procedure). Nothing in here is a real calibration until Brad shoots one;
`schema.json` (JSON Schema draft-07) and the example below define the format.

One file per (body, lens, zoom setpoint, focus setpoint). Name them so the rig
config reads well, e.g. `a7rv-selp1635g-16mm-f8.json`.

## Example

```json
{
  "schema_version": 1,
  "created_at": "2026-09-10T15:42:00Z",
  "tool": "image-processing calibrate_distortion 0.1.0",
  "camera": {
    "body": "ILCE-7RM5",
    "serial": null,
    "lens": "FE PZ 16-35mm F4 G",
    "zoom_position": 0,
    "focus_position": 1234,
    "units": "sdk_raw",
    "aperture": "f/8",
    "shutter": "1/200",
    "iso": 100
  },
  "develop": {
    "user_flip": 0,
    "half_size": false,
    "note": "geometry-affecting rawpy.postprocess params used",
    "rawpy_version": "0.27.0",
    "libraw_version": "0.22.1",
    "demosaic": "DHT"
  },
  "image_size": [9564, 6376],
  "model": "opencv_standard",
  "camera_matrix": [[7012.3, 0.0, 4790.1], [0.0, 7010.8, 3186.4], [0.0, 0.0, 1.0]],
  "dist_coeffs": [-0.0512, 0.0219, 0.00011, -0.00008, -0.0031],
  "lateral_ca": {
    "r_norm": 5747.3,
    "R": { "coeffs": [0.00031, -0.00012, 0.00004], "rms_residual_px": 0.11, "max_shift_px_at_corner": 2.8 },
    "B": { "coeffs": [-0.00027, 0.00009, -0.00002], "rms_residual_px": 0.13, "max_shift_px_at_corner": 3.4 }
  },
  "quality": {
    "n_frames_detected": 33,
    "n_frames_failed_detection": 1,
    "n_frames_used": 31,
    "n_frames_rejected": 2,
    "n_holdout": 4,
    "rms_reprojection_px": 0.38,
    "holdout_rms_px": 0.42,
    "per_image_rms_px": { "DSC00123.ARW": 0.35 },
    "holdout_per_image_rms_px": { "DSC00160.ARW": 0.41 },
    "rejected_frames_rms_px": { "DSC00131.ARW": 1.4 },
    "detection_failures": ["DSC00140.ARW"],
    "coverage_grid_4x3": [[5, 7, 6, 4], [8, 12, 11, 7], [4, 6, 6, 5]],
    "straight_line_test": {
      "frame": "DSC00160.ARW",
      "max_bow_px_before": 9.6,
      "max_bow_px_after": 0.4,
      "center_crop": { "crop": "center 50%", "max_bow_px_before": 3.1, "max_bow_px_after": 0.3 }
    },
    "invalid_output_fraction": 0.0
  },
  "sources": ["DSC00123.ARW"],
  "holdout_sources": ["DSC00160.ARW"],
  "sha256_of_sources": "…"
}
```

The numbers above are illustrative, not measured. Note `image_size`: rawpy
outputs **9564×6376** for the A7R V (LibRaw's full sensor area, not the
9504×6336 the camera's own JPEG crop uses). The CLI records whatever rawpy
produced; the module refuses to apply a calibration whose `image_size` differs
from the frame it is developing.

## Fields

| Field | Used by | Meaning |
|---|---|---|
| `schema_version` | loader | Must be `1`. Anything else is rejected. |
| `created_at`, `tool` | audit | UTC timestamp and CLI version. `created_at` is echoed in every `develop` result. |
| `camera.body`, `serial`, `lens`, `aperture`, `shutter`, `iso` | match check / audit | Read from the ARW EXIF (Sony keeps the serial in the makernote, so it is usually `null`). `lens` is compared, case- and whitespace-insensitively, against what the source camera's `get_status` reports. |
| `camera.zoom_position`, `focus_position`, `units` | match check | Raw Sony SDK integers passed on the CLI (`--zoom-position`, `--focus-position`) — they are not in EXIF. Compared against the capture response. A mismatch warns, or errors under `strict_calibration_match`. |
| `develop` | audit | The geometry-affecting rawpy parameters the frames were developed with (always `user_flip: 0`, `half_size: false`) plus the rawpy/LibRaw versions, because LibRaw's border crop can move between versions. |
| `image_size` | **hard check** | `[width, height]` of the rawpy sensor-frame output. A developed frame of any other size is an error: the remap tables cannot apply. |
| `model` | loader | `opencv_standard` (5 coefficients) or `opencv_rational` (8, `CALIB_RATIONAL_MODEL`). Must agree with the length of `dist_coeffs`. |
| `camera_matrix` | maths | 3×3 pinhole intrinsics `K`. Also the output camera matrix — the frame is not rescaled, so normalized crop rects keep meaning. |
| `dist_coeffs` | maths | Brown–Conrady `[k1, k2, p1, p2, k3]` (+ `[k4, k5, k6]` for rational), as OpenCV defines them. |
| `lateral_ca` | maths | `null` when fitted with `--no-ca`. Otherwise `r_norm` (half the image diagonal, px) and per-channel `coeffs` `[a0, a1, a2]` for `d_c(p) = (p − c0)·(a0 + a1·ρ² + a2·ρ⁴)`, `ρ = |p − c0| / r_norm`, `c0` the principal point from `K`. G is the reference; R and B are remapped by `map_G + d_c(map_G)`. `rms_residual_px` and `max_shift_px_at_corner` are fit diagnostics. |
| `quality` | audit / log | Everything the CLI measured. The module logs `rms_reprojection_px`, `holdout_rms_px`, `n_frames_used` at load and puts `rms_reprojection_px` in each result. See the README for what each number should look like. |
| `sources`, `holdout_sources`, `sha256_of_sources` | provenance | The frames used, the frames held out, and one SHA-256 over the per-file SHA-256s (name order) — keep the source folder; this is how you prove which shots made the file. |

Unknown top-level keys are carried through the loader untouched (`meta`), so
extra provenance can be added without a schema bump.
