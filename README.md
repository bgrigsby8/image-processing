# Module image-processing

Image-processing camera components for Viam.

## Models

- [`brad-grigsby:image-processing:color-correction`](brad-grigsby_image-processing_color-correction.md) — wraps a source camera and applies a 3×3 Color Correction Matrix fitted from a ColorChecker Classic, so colors stay consistent under your lighting. Corrects the streaming `get_images` path, and offers a non-destructive studio RAW developer via `DoCommand`: `capture` triggers and develops a still from the source camera (e.g. the `ptp` model), and `develop` processes existing CR3/NEF/… files already on disk. RAW is demosaiced to 16-bit linear, white-balanced and color-corrected, then exported (16-bit TIFF, JPEG, etc.) with the original left untouched.
- [`brad-grigsby:image-processing:ptp`](brad-grigsby_image-processing_ptp.md) — talks directly to a USB-connected still camera (Canon/Nikon/Sony/etc.) over PTP via libgphoto2. Capture stills, list the card, and download images over a USB-C cable through `DoCommand`; `get_images` streams a live-view preview.

## System requirements

`setup.sh` builds the Python venv on the target device. The RAW pipeline used by
`color-correction` depends on native libraries shipped inside their wheels
(`rawpy` bundles LibRaw, `opencv-python-headless` bundles its own libs,
`tifffile` is pure Python), so a normal install on `linux/amd64`, `linux/arm64`,
or `darwin/arm64` needs nothing extra.

On a minimal/headless target the wheels can still fail to load. `setup.sh`
detects this and, on Debian/Ubuntu, auto-installs the system packages:

- **`libraw-dev`** — if pip has to build `rawpy` from source (no matching wheel)
- **`libglib2.0-0`** — if OpenCV-headless can't load `libgthread-2.0.so`

On non-apt systems, install the LibRaw and glib equivalents for your OS
(e.g. `brew install libraw` on macOS) and re-run.

## Running the tests

The unit tests live in `tests/` and import the module straight from `src/`
(`tests/conftest.py` puts `src` on `sys.path`, mirroring `run.sh`). They need
Python ≥3.10.

Build the environment, add the dev dependencies (`pytest`, kept in
`requirements-dev.txt` rather than `requirements.txt`), and run from the repo
root:

```sh
./setup.sh                              # creates ./venv and installs requirements.txt
venv/bin/pip install -r requirements-dev.txt
venv/bin/python -m pytest
```

To run a single file or filter by name:

```sh
venv/bin/python -m pytest tests/test_color_correction.py
venv/bin/python -m pytest tests/test_color_correction.py -k upload
```

## Lens distortion calibration

We shoot wide (16 mm on the A7R V) and crop the product out of the frame
centre. Sony corrects the lens in-camera for JPEGs, but rawpy/LibRaw does not
apply that profile, so the RAW pipeline sees the uncorrected lens: a half-frame
crop still carries ~0.5–1 % barrel — several pixels of bow along a straight
edge. `color-correction` can undistort (and correct lateral chromatic
aberration on) every developed frame from a one-time OpenCV checkerboard
calibration, before the frame is cropped downstream. Calibration is
zoom-dependent and mildly focus-dependent, so it is done at the production zoom
and focus setpoints.

The format is documented in [`calibration/README.md`](calibration/README.md).
The library is `src/models/distortion.py`; the CLI is
`scripts/calibrate_distortion.py`.

### 1. Shoot the target

**Target.** Checkerboard with **10×7 inner corners** (11×8 squares), squares
~50–60 mm (about 600×450 mm overall). Matte print — strobes and gloss don't
mix — mounted dead flat (foam board, Dibond, glass). Measure a printed square
with a ruler after printing (printers scale) and pass the true value to
`--square-mm`. Even×odd corners keep the board's orientation unambiguous.

**Camera.** Identical to production: zoom at the 16 mm setpoint via
`set_zoom_position`, the production focus position, f/8, strobes at production
power, production shutter/ISO. Note the raw `zoom_position` and
`focus_position` values — the CLI records them and the module checks captures
against them.

**Frames.** 25–40 calibration frames with the *whole* board visible in every
one:

- Move the board so that, across the set, corners land in **every region of
  the frame — especially the corners and edges**. That is where the distortion
  signal lives; centre-only frames teach the solver nothing. The CLI prints a
  4×3 coverage grid; every cell should see at least 3 frames.
- Tilt the board ±20–35° about both axes in most frames; a few square-on.
- Vary distance over the working range (~0.4–1.6 m) so the board appears at
  different scales.
- Watch for specular glare on the target; strobes make motion blur a non-issue.

Plus **3–5 validation frames** (board square-on at different positions;
reserve them with `--holdout-files`) and a square-on board at each working
distance for the straight-line report.

Shoot ARW into a dated folder (`calib/2026-09-10-16mm-f8/`) and keep it — the
calibration JSON records a hash of exactly these files.

### 2. Run the CLI

```sh
venv/bin/python scripts/calibrate_distortion.py \
  --input 'calib/2026-09-10-16mm-f8/*.ARW' \
  --pattern 10x7 --square-mm 55.0 \
  --out calibration/a7rv-selp1635g-16mm-f8.json \
  --zoom-position 0 --focus-position 1234 \
  --holdout-files DSC00160.ARW DSC00161.ARW DSC00162.ARW \
  --debug-dir calib/2026-09-10-16mm-f8/debug
```

Useful options: `--model auto` fits both the 5-coefficient standard and the
8-coefficient rational model and keeps rational only if it beats standard on
the holdout frames by more than 10 % (the default `standard` is right for a
rectilinear lens; rational overfits small sets); `--no-ca` skips the lateral
CA fit; `--holdout N` holds out N evenly spaced frames instead of named ones;
`--reject-above-px` (default 1.0) drops frames whose reprojection RMS is above
it (or above 3× the median) and refits once; `--jobs N` develops frames in
parallel (each ~1 GB at 61 MP); `--detect-scale` (default 0.25) is the
downscale for the coarse board search — sub-pixel refinement always runs at
full resolution.

Each ARW is developed with exactly the production develop's geometry
parameters (`image_io.SENSOR_FRAME_KWARGS`: `user_flip=0`, full size), so the
calibration is in the same pixel frame the module applies it in. Expect a few
seconds per frame for the demosaic plus up to ~20 s for the exhaustive board
search on frames where the board is hard to find.

### 3. Read the report

The run ends in a summary with one PASS/WARN line per acceptance criterion:

| Check | Target | If it WARNs |
|---|---|---|
| reprojection RMS | ≤ 0.5 px at full resolution | Board not flat, glare, motion, or a wrong `--pattern`/`--square-mm`. Look at the `_corners.jpg` overlays in `--debug-dir`. |
| per-image max RMS | ≤ 1.0 px after rejection | One bad frame the rejection couldn't drop (too few frames left). Reshoot or remove it. |
| holdout RMS | within 20 % of training (or +0.1 px) | Overfit — usually `--model rational` with too few frames. Use `standard`, or shoot more boards. |
| frame coverage | every 4×3 cell ≥ 3 frames | Shoot more boards in the named cells (edges/corners). `coverage.png` shows the grid. |
| straight-line (center 50%) | bow ≤ 1.0 px after | The *before* number is the "was this worth it" figure. If *after* is high, the fit is wrong somewhere — check coverage. |
| lateral CA fit R / B | residual ≤ 0.3 px RMS | Noisy channels (underexposed board) or a real non-radial CA. Try more frames; `--no-ca` disables it. |

`--debug-dir` also writes `residual_quiver.png` (reprojection residuals ×50 —
a systematic pattern means the model is wrong, random dots mean noise) and
`<frame>_before.jpg` / `_after.jpg` for the straight-line frame.

The JSON's `quality` block carries every number; the module logs the headline
ones at load and echoes `rms_reprojection_px` in every develop result.

### 4. Configure the module

Copy the JSON somewhere stable on the rig PC and point `color-correction` at
it:

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

A configured file that is missing or unparseable is a configuration error,
never a silent skip. With the attribute absent the pipeline is byte-identical
to a build without this feature. `cache_undistort_maps` holds the remap
tables for the life of the process (~1.4 GB at 61 MP with CA, ~0.5 GB
without); set it false on a memory-constrained PC to trade a couple of
seconds per develop for the RAM. Every `capture`/`develop` result reports
`undistorted`, `lateral_ca_corrected`, the `calibration` used (path,
created_at, rms, sha256) and any `calibration_warnings` — the audit trail
when an image is questioned. See the
[color-correction docs](brad-grigsby_image-processing_color-correction.md#attributes)
for the attribute table and the match-check behaviour.

Previews (deferred-capture thumbnails, the `preview` command, streaming
`get_images`) are **not** undistorted — they decode at half size, which is not
the calibration frame. A crop rect drawn on a preview lands within a fraction
of a preview pixel of the same spot on the corrected frame (the centre scale is
unchanged by design), and the webapp crops the retained developed frame anyway.

### 5. Recalibrate when

- the lens or body changes, or the lens is knocked or serviced;
- the zoom setpoint changes (the module warns on every capture whose
  `zoom_position` differs from the file's);
- the production focus setpoint changes materially (distortion breathes with
  focus — if the shoe and outfit stations end up with different focus
  setpoints, shoot a validation board at both and compare the straight-line
  residual; per-station calibration files are the follow-up if it exceeds
  1 px);
- rawpy/LibRaw is upgraded (the output frame size can change — the module
  refuses to apply a calibration whose `image_size` differs, which is the
  signal);
- a straight-edge spot check on a developed frame drifts above 1 px of bow in
  the centre crop.
