# Task: Checkerboard distortion + lateral CA calibration for the Sony RAW pipeline

**Repo:** `~/projects/image-processing`
**Model touched:** `brad-grigsby:image-processing:color-correction` (the `develop` / `capture` RAW pipeline)
**Camera:** Sony A7R V (61 MP) + FE PZ 16-35mm F4 G, zoom parked at the 16 mm setpoint, f/8
**Author:** Brad Grigsby — 2026-09-04

---

## 1. Why

We shoot wide at 16 mm and crop the product out of the center of the 61 MP frame. A center crop reduces barrel distortion a lot (it falls off roughly with r³) but does not remove it; the 16-35 at 16 mm has ~4–5 % native barrel at the frame edge, so a half-frame crop is still carrying on the order of 0.5–1 % — several pixels of bow along a straight edge in a 15 MP crop. Sony corrects this in-camera for JPEGs and embeds the profile in the ARW, but **rawpy/LibRaw does not apply it**, so our RAW pipeline sees the uncorrected lens.

Fix: a one-time OpenCV checkerboard calibration at the 16 mm setpoint and production focus, then undistort (and correct lateral chromatic aberration) every developed frame *before* it is cropped downstream. Calibration is zoom-dependent, which is why `sony-remote` logs the zoom position on every capture.

## 2. What exists today (read this code first)

- `color-correction` wraps a source camera. `DoCommand capture` triggers the source and develops the resulting RAW; `DoCommand develop` processes an existing RAW on disk. Pipeline: RAW → rawpy demosaic to 16-bit linear → white balance → 3×3 CCM → export (16-bit TIFF / JPEG / …). The original RAW is never modified.
- Streaming `get_images` applies the CCM to live-view frames from the source camera. **Leave the streaming path alone** — live-view frames come from the camera at a different size and geometry (see open question §11.1).
- Crops are applied downstream in `nines-webapp` as normalized [0,1] rects against the developed full frame. That is why undistortion belongs in `develop`: the webapp then crops an already-corrected frame with no changes on its side.
- House style: see `~/projects/comxim` — Python Viam module, extensive hardware-free tests against a simulated device, quirks documented in code. Match it.

Before writing anything: locate the demosaic call and its `rawpy.postprocess(...)` parameters, the config validation code, the `develop`/`capture` response shape, and the test layout. Existing tests must keep passing with no calibration configured.

## 3. Deliverables

1. `image_processing/distortion.py` — library: load/save calibration, build remap tables, apply undistort + CA to a 16-bit linear RGB frame.
2. `scripts/calibrate_distortion.py` (or `python -m image_processing.calibrate_distortion`) — CLI that turns a folder of checkerboard ARWs into a calibration JSON, with validation and debug output.
3. Integration into the `color-correction` develop path, behind new config attributes (off when not configured).
4. Hardware-free tests.
5. README section: how to shoot the target, run the CLI, read the report, configure the module, and when to recalibrate.
6. Example calibration file schema (`calibration/README.md` or a JSON schema) — not a real calibration; Brad produces that on the rig.

## 4. Design decisions (locked — don't relitigate, do flag problems)

**Distortion model.** OpenCV pinhole + Brown–Conrady via `cv2.calibrateCamera`. Default: standard 5 coefficients (k1, k2, p1, p2, k3). `--model rational` enables `CALIB_RATIONAL_MODEL` (8 coefficients) as an option; pick whichever gives lower *holdout* reprojection error, default to standard if they're within noise. Not the fisheye model — this is a rectilinear lens.

**Coordinate frame.** Calibrate and apply in exactly the same pixel frame: the output of `rawpy.postprocess` with the same geometry-affecting parameters the production develop uses, full size (`half_size=False`), no cropping, and **`user_flip=0`** in both places. The camera hangs off an arm and its orientation sensor could flip the EXIF orientation between shots; the distortion map is in sensor coordinates and must not be rotated underneath us. If the production develop currently relies on rawpy's default orientation handling, change it to `user_flip=0` and handle any rotation explicitly *after* undistortion (and note it in the PR).

**Output framing.** Undistort with `newCameraMatrix = K` (the calibrated intrinsics), same output size as input. For barrel distortion this yields a fully-valid output (no black borders) with the center scale unchanged, so a center crop covers the same physical field of view before and after. Do not use `getOptimalNewCameraMatrix` — it rescales the frame and would silently move every crop rect.

**Interpolation.** Operate on 16-bit linear data (correct place to resample). Default `cv2.INTER_CUBIC`; `lanczos4` and `linear` selectable. Remap per channel (single-channel `uint16`), then merge.

**Lateral CA model.** Constrained radial model fitted directly from corner displacements, not three independent `calibrateCamera` runs (planar targets make focal length and distance nearly degenerate, so per-channel intrinsics would be noisy). For channel c ∈ {R, B}, with principal point **c₀** = (cx, cy) from K and ρ = |p − c₀| / R_norm (R_norm = half the image diagonal):

    d_c(p) = (p − c₀) · (a₀ + a₁ ρ² + a₂ ρ⁴)

Fit a₀, a₁, a₂ per channel by linear least squares over all corners in all frames (x and y equations both). Corner positions for R and B come from `cv2.cornerSubPix` run on the R and B channel images, **initialized at the G corners** (CA shift is a few px at most; an 11×11 window converges to the channel's own corner). G is the geometric reference.

**Map composition.** One remap per channel:

    (mx, my) = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)   # G map
    map_G = (mx, my)
    map_c = map_G + d_c(map_G)          # for c in {R, B}; d_c evaluated at the *source* position

(Corners were detected in the distorted source frame, so d_c is defined there: the R content matching G source position s lives at s + d_R(s).) Optionally `cv2.convertMaps` to fixed-point (`CV_16SC2`/`CV_16UC1`) for speed and memory.

**Caching / memory.** Build maps lazily on first `apply` and cache for the lifetime of the calibration. Three float32 map pairs at 9504×6336 are ~1.4 GB; fixed-point is ~1.1 GB; distortion-only (no CA) is one third of that. Building the maps takes a few seconds at 61 MP, comparable to the demosaic itself. Expose `cache_undistort_maps` (default true) so a memory-constrained rig PC can trade time for RAM. Log the footprint once at load.

**Failure policy.** Image size ≠ calibration `image_size` → hard error (the maps cannot apply). Lens / zoom / focus metadata mismatch → warn and apply by default, error if `strict_calibration_match: true`. Calibration file configured but missing/unparseable → reconfigure error, never a silent skip. No calibration configured → pipeline byte-identical to today.

## 5. Calibration file (JSON)

```json
{
  "schema_version": 1,
  "created_at": "2026-09-10T15:42:00Z",
  "tool": "image-processing calibrate_distortion 0.1.0",
  "camera": {
    "body": "ILCE-7RM5", "serial": "…",
    "lens": "FE PZ 16-35mm F4 G",
    "zoom_position": 0, "focus_position": 1234, "units": "sdk_raw",
    "aperture": "f/8", "shutter": "1/200", "iso": 100
  },
  "develop": { "user_flip": 0, "half_size": false, "note": "geometry-affecting rawpy params used" },
  "image_size": [9504, 6336],
  "model": "opencv_standard",
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [k1, k2, p1, p2, k3],
  "lateral_ca": {
    "r_norm": 5708.9,
    "R": { "coeffs": [a0, a1, a2], "rms_residual_px": 0.11, "max_shift_px_at_corner": 2.8 },
    "B": { "coeffs": [a0, a1, a2], "rms_residual_px": 0.13, "max_shift_px_at_corner": 3.4 }
  },
  "quality": {
    "n_frames_used": 31, "n_frames_rejected": 2, "n_holdout": 4,
    "rms_reprojection_px": 0.38, "holdout_rms_px": 0.42,
    "per_image_rms_px": { "DSC00123.ARW": 0.35, "…": 0.0 },
    "coverage_grid_4x3": [[5, 7, 6, 4], [8, 12, 11, 7], [4, 6, 6, 5]],
    "straight_line_test": { "frame": "DSC00160.ARW", "max_bow_px_before": 9.6, "max_bow_px_after": 0.4, "crop": "center 50%" }
  },
  "sources": ["DSC00123.ARW", "…"],
  "sha256_of_sources": "…"
}
```

`camera.*` is metadata for the match check: read what you can from the ARW EXIF (body, serial, lens, aperture, shutter, ISO); `zoom_position` / `focus_position` are raw SDK integers passed on the CLI (`--zoom-position`, `--focus-position`) because they are not in EXIF.

## 6. CLI spec

```
calibrate_distortion \
  --input  'calib/2026-09-10-16mm-f8/*.ARW' \
  --pattern 10x7 --square-mm 55.0 \
  --out    calibration/a7rv-selp1635g-16mm-f8.json \
  --zoom-position 0 --focus-position 1234 \
  [--model standard|rational]  [--no-ca]  [--holdout N]  [--holdout-files a.ARW b.ARW]
  [--straight-edge-frame DSC00160.ARW]  [--detect-scale 0.25]  [--reject-above-px 1.0]
  [--debug-dir ./debug]  [--jobs N]
```

Steps:

1. **Develop each ARW** with rawpy using the shared geometry parameters (import them from the same place the production develop does — one source of truth). Produce: G channel as float32 for sub-pixel work; R and B channels likewise if CA enabled; and an 8-bit gamma-encoded (≈1/2.2), percentile-normalized grayscale for detection (linear 16-bit looks nearly black to the detector).
2. **Detect corners** on a downscaled copy (`--detect-scale`, default 0.25 → ~2376×1584) with `cv2.findChessboardCornersSB(..., CALIB_CB_EXHAUSTIVE | CALIB_CB_ACCURACY)`; fall back to `findChessboardCorners` + `cornerSubPix` if SB fails. Scale corner coordinates back to full res and refine with `cornerSubPix` on the full-res G channel (window ~11×11, tight termination criteria). Frames where the full board is not found are reported and skipped, not fatal.
3. **CA corners:** `cornerSubPix` on R and B at full res, initialized from the refined G corners.
4. **Calibrate:** `cv2.calibrateCamera` on the non-holdout frames. Report RMS and per-image RMS. Frames above `--reject-above-px` (default 1.0) or > 3× the median are rejected and calibration re-run once. Print the coverage grid (how many frames put corners in each 4×3 cell of the frame) — thin edge/corner cells are the most common reason for a bad calibration, say so in the output.
5. **Holdout:** solve PnP for each holdout frame with the fitted intrinsics, report reprojection RMS. Holdout RMS much worse than training RMS = overfit (usually the rational model with too few frames).
6. **CA fit:** least squares per channel as in §4; report RMS residual and the implied shift at the frame corner.
7. **Straight-line test** (if `--straight-edge-frame` given, else use the flattest holdout board): undistort the frame's detected corners, fit a line to each row/column of corners, report max deviation before vs after, both for the full frame and for the central 50 % crop.
8. **Write JSON** and, if `--debug-dir`: corner overlays per frame, a residual quiver plot, before/after undistorted board images (downscaled), and the coverage heatmap. Print a short human-readable summary ending in PASS/WARN per the acceptance criteria in §9.

## 7. Library API (`image_processing/distortion.py`)

```python
@dataclass(frozen=True)
class LateralCA:
    r_norm: float
    coeffs_r: tuple[float, float, float]
    coeffs_b: tuple[float, float, float]

@dataclass(frozen=True)
class DistortionCalibration:
    image_size: tuple[int, int]          # (w, h)
    camera_matrix: np.ndarray            # 3x3 float64
    dist_coeffs: np.ndarray              # 5 or 8 float64
    lateral_ca: LateralCA | None
    meta: dict                           # camera, develop, quality, sources — opaque

    @classmethod
    def load(cls, path) -> "DistortionCalibration": ...
    def save(self, path) -> None: ...
    def check_match(self, *, image_size, lens=None, zoom_position=None, focus_position=None) -> list[str]:
        """Returns human-readable mismatch messages; empty list = clean. Size mismatch is always first."""

class Undistorter:
    def __init__(self, calib, *, interpolation="cubic", correct_ca=True, cache_maps=True): ...
    def apply(self, img: np.ndarray) -> np.ndarray:
        """img: HxWx3 uint16 linear RGB in the calibration frame. Returns same shape/dtype.
        Raises CalibrationMismatch on size mismatch."""
    @property
    def map_memory_bytes(self) -> int: ...

def fit_lateral_ca(corners_g, corners_c, principal_point, r_norm) -> tuple[coeffs, rms_px]: ...
def undistort_points(calib, pts) -> np.ndarray: ...   # for tests and the straight-line check
```

Pure functions where possible; no Viam imports in this file so the CLI and tests can use it without a module context.

## 8. Pipeline integration (`color-correction`)

New config attributes (all optional; today's behavior when `distortion_calibration` is absent):

```json
{
  "distortion_calibration": "/opt/nines/calibration/a7rv-selp1635g-16mm-f8.json",
  "undistort": true,
  "correct_lateral_ca": true,
  "undistort_interpolation": "cubic",
  "strict_calibration_match": false,
  "cache_undistort_maps": true
}
```

- On (re)configure: load and validate the file, construct the `Undistorter`, log one summary line (lens, zoom/focus, rms, holdout rms, n_frames, created_at, map memory). Missing or invalid file → configuration error.
- In `develop`: RAW → demosaic (16-bit linear, `user_flip=0`) → WB → **`undistorter.apply`** → CCM → export. Undistortion is geometric resampling; WB and CCM are per-pixel linear, so the exact placement among them doesn't change the math — put it right after demosaic/WB so it reads like the doc.
- Metadata check: run `check_match` with the developed image size plus whatever the source camera reported for this capture (the `sony-remote` `capture` response and per-capture log include `zoom_position` / `focus_position`; `get_status` has `lens`). If those aren't available for a given call (e.g. `develop` of an old file), log that the match was not checked rather than failing.
- Response: add to the `capture` / `develop` result `"undistorted": true|false`, `"lateral_ca_corrected": true|false`, `"calibration": {"path": …, "created_at": …, "rms_reprojection_px": …, "sha256": …}`, and `"calibration_warnings": [...]`. This is the audit trail when Nines questions an image.
- Do not undistort the streaming `get_images` preview (see §11.1).
- Original RAW stays untouched; never write into the capture directory.

## 9. Acceptance criteria

CLI on real rig data (Brad runs this; the CLI must compute and print every one of these):

- RMS reprojection error ≤ 0.5 px at full resolution; per-image max ≤ 1.0 px after rejection.
- Holdout RMS within ~20 % of training RMS.
- Coverage grid: no 4×3 cell with fewer than 3 frames contributing corners (edge and corner cells matter most).
- Straight-line test in the central 50 % crop: max bow after undistortion ≤ 1.0 px (report the before number too — that's the "was this worth it" figure).
- Lateral CA fit residual ≤ 0.3 px RMS per channel.

Code:

- All existing tests pass with no calibration configured; developed output is byte-identical to today in that case.
- New tests in §10 pass hardware-free.
- `develop` of a 61 MP frame with undistort + CA adds no more than a few seconds with maps cached.

## 10. Tests (hardware-free, house style)

Use small synthetic images (e.g. 1200×800) so the suite stays fast; mark any full-resolution test `slow`.

- **Synthetic calibration round-trip:** generate checkerboard object points, choose a known K and dist (e.g. k1 ≈ −0.10, k2 ≈ 0.02 — heavy barrel like the real lens), project with `cv2.projectPoints` under 25 random poses that cover the frame, feed the point sets to the calibration routine (bypassing detection), and assert recovered K and dist within tolerance and RMS ≈ 0.
- **Detection:** render an ideal checkerboard image, warp it with the known distortion, run the detection + sub-pixel path, and assert corners within 0.3 px of the analytically distorted positions.
- **Undistorter straightness:** draw a grid of straight lines, distort with the known model (build the distorted image by sampling the ideal image at `undistortPoints` of each output pixel), run `Undistorter.apply`, fit lines to the recovered edges, assert max deviation < 0.5 px. Also assert output shape/dtype unchanged and center pixel scale unchanged (a known point near center lands within 0.5 px of its ideal location).
- **Lateral CA:** synthesize R/B corners as G corners plus a known radial polynomial shift (plus small noise); assert `fit_lateral_ca` recovers the coefficients and that the composed map removes the shift (apply to a synthetic 3-channel image where R/B are radially scaled copies of G; assert per-channel edge alignment after correction).
- **Failure policy:** size mismatch raises; lens/zoom mismatch → warnings list; `strict_calibration_match` turns warnings into errors at the module layer; missing file → reconfigure error; no config → `develop` output identical to the pre-change path (golden comparison on a small synthetic RAW-like input or by mocking the demosaic step).
- **File format:** JSON save/load round-trip preserves arrays exactly; unknown `schema_version` rejected.
- **Map caching:** second `apply` doesn't rebuild maps (spy/counter); `cache_maps=False` rebuilds and frees.

## 11. Open questions to verify on hardware (not blockers for the code)

1. **Are Sony live-view frames lens-corrected in-camera?** If yes, the operator's crop preview (streaming path) and the developed frame (now also corrected) agree geometrically; if no, they differ by the distortion and crop rects drawn on the preview land slightly off on the developed frame. The webapp can already re-crop against the retained developed full frame, which is the safe path either way. Check with a straight edge in live view vs. a developed frame.
2. **Focus dependence.** Distortion changes mildly with focus distance (breathing). One calibration at the hyperfocal f/8 focus setpoint is the plan. If the shoe station (~0.5 m) and outfit station (~1.5 m) end up with different per-station focus setpoints, shoot validation boards at both and check the straight-line residual; if it exceeds §9, this task's follow-up is per-station calibration files keyed by `focus_position`.
3. **Zoom repeatability** across power cycles is already on the `sony-remote` SMOKE checklist; the calibration is only valid at the recorded `zoom_position`, which is what the match check enforces.
4. **rawpy output size** for the A7R V and whether LibRaw's border crop is stable across versions — pin the rawpy/LibRaw version in requirements and record it in the calibration `develop` block.

## 12. Capture procedure (for Brad — the CLI's assumptions match this)

**Target.** Checkerboard with **10×7 inner corners** (11×8 squares), squares ~50–60 mm → roughly 600×450 mm. Matte print (no gloss — strobes), mounted dead flat on foam board, Dibond, or glass. Measure the printed square size with a ruler after printing (printers scale) and pass the true value to `--square-mm`. Even×odd corner count keeps orientation unambiguous.

**Camera.** Identical to production: zoom asserted at the 16 mm setpoint via `set_zoom_position`, production focus position, f/8, strobes at production power, production shutter/ISO. Note the raw `zoom_position` and `focus_position` values for the CLI.

**Frames.** 25–40 calibration frames, whole board visible in every one:

- Move the board around so that, across the set, corners land in **every region of the frame, especially the corners and edges** — that is where the distortion signal lives; center-only frames teach the solver nothing.
- Tilt the board ±20–35° about both axes in most frames; a few square-on.
- Vary distance across the working range, ~0.4–1.6 m, so the board appears at different scales.
- Strobes make motion blur a non-issue; watch for specular glare on the target.

Plus **3–5 validation frames** (board square-on at different frame positions, reserved via `--holdout-files`) and **one straight-edge frame** at each working distance (a level or long ruler near the frame edge, for the before/after bow report).

Shoot ARW like production, into a dated folder (`calib/2026-09-10-16mm-f8/`). Keep the folder — it's the provenance for the calibration file.

**Recalibrate when:** the lens or body changes, the zoom setpoint changes, the production focus setpoint changes materially, the lens is knocked or serviced, or the straight-line spot check drifts above §9.

## 13. Out of scope (follow-ups)

- Vignetting / flat-field correction (same harness could produce it; not now).
- ChArUco targets (tolerate partial boards; consider if frame-edge coverage proves hard with a full-visibility checkerboard).
- Per-station calibration files keyed by focus position (only if §11.2 says so).
- Webapp-side changes (none needed: it crops the developed frame).