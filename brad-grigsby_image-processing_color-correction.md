# Model brad-grigsby:image-processing:color-correction

A camera component that wraps another (source) camera and applies a 3×3 Color
Correction Matrix (CCM) to its images. The CCM is fitted from a photo of a
Calibrite / X-Rite **ColorChecker Classic** (24 patches), so colors come out
consistent under your lighting.

It works two ways:

- **Streaming** — `get_images` proxies the source camera and color-corrects
  every JPEG/PNG frame (names preserved). Used by the control tab, data
  manager, and vision services.
- **DoCommand** — a studio-grade RAW developer. `capture` triggers a still on
  the source camera (e.g. the `ptp` model), and `develop` processes RAW/image
  files already on disk. Both run the same pipeline:

  - RAW (CR3/NEF/ARW/RAF/DNG/…) is demosaiced to **16-bit linear** with
    [rawpy](https://pypi.org/project/rawpy/) (auto-brightness off, white balance
    applied at the raw stage), so no tonal precision is lost before correction.
  - White balance and the CCM are applied in **linear light**.
  - The result is written out as rendered exports (16-bit TIFF, JPEG, etc.),
    tagged with an sRGB ICC profile.
  - It's **non-destructive**: the original RAW is never modified. The
    adjustments are recorded in a `<name>.json` sidecar next to it, the way
    Capture One / Lightroom keep edits separate from the negative.

> **Color space:** output is sRGB, because the CCM is fitted against sRGB
> ColorChecker references. True wide-gamut output (ProPhoto, etc.) would need the
> calibration reworked against wide-gamut references.

## Requirements

The RAW pipeline needs native libraries (`rawpy` bundling LibRaw, `tifffile`,
`opencv-python-headless`). These install from `requirements.txt` as prebuilt
wheels on `linux/amd64`, `linux/arm64`, and `darwin/arm64`. See the
[module README](README.md#system-requirements) for the minimal/headless cases
where a system package (`libraw-dev`, `libglib2.0-0`) is needed — `setup.sh`
installs those automatically on Debian/Ubuntu.

## Configuration

```json
{
  "camera": "my-ptp-cam",
  "ccm": [
    [1.16, -0.08, -0.04],
    [-0.10, 1.28, -0.10],
    [0.00, 0.02, 0.72]
  ],
  "output_dir": "/photos/exports",
  "output_formats": ["tiff16", "jpeg"]
}
```

### Attributes

| Name             | Type         | Inclusion | Description                                                                             |
|------------------|--------------|-----------|-----------------------------------------------------------------------------------------|
| `camera`         | string       | Required  | Source camera to wrap; declared as a dependency. Required even for `develop`.           |
| `ccm`            | 3×3 array    | Optional  | Color correction matrix from `calibrate_color`. Omit to pass images through unchanged.  |
| `output_dir`     | string       | Optional  | Where `capture`/`develop` write exports. Default: next to the source file.              |
| `output_formats` | string[]     | Optional  | Any of `tiff16`/`tiff8`/`jpeg`/`png16`/`png8`. Default: all four.                       |
| `jpeg_quality`   | int          | Optional  | JPEG export quality. Default `95`.                                                      |
| `white_balance`  | string/array | Optional  | RAW white balance: `camera` (default), `auto`, `daylight`, or `[r,g,b,g2]` multipliers. |
| `exposure_stops` | number       | Optional  | Default exposure compensation (stops) applied at the raw stage. Paste the value `calibrate_color` reports to render captures at the calibrated reference brightness. Default `0`. A per-call `exposure_stops` overrides it. |
| `tone`           | string       | Optional  | Delivery "look" applied on export: `none` (default — colour-accurate / colorimetric output), `medium`, or `bright` (a Capture One-style midtone lift). Only lightness/contrast changes — the CCM keeps hue accurate. Applied to every export and the preview; recorded in the sidecar. A per-call `tone` overrides it. |
| `sharpen`        | string       | Optional  | Capture sharpening (luminance unsharp mask): `none` (default), `light`, `medium`, `strong`. RAW is soft before sharpening, so an unsharpened export looks blurry next to a Capture One / Lightroom render. Applied to every export and the preview; recorded in the sidecar. A per-call `sharpen` overrides it. |
| `demosaic`       | string       | Optional  | RAW demosaic algorithm: `DHT` (default — sharper than libraw's stock AHD), or `AHD`/`AAHD`/`DCB`/`VNG`/`PPG`. (AMAZE/LMMSE need GPL demosaic packs not bundled in the libraw wheels.) A per-call `demosaic` overrides it. |
| `write_sidecar`  | boolean      | Optional  | Write a `<name>.json` sidecar recording the development. Default `true`.                |
| `part_id`        | string       | Optional  | Machine part to attach `upload`s to. Defaults to `VIAM_MACHINE_PART_ID` from the env.   |
| `delete_after_upload` | boolean | Optional  | Remove each local file once its `upload` succeeds (failed uploads keep their files for retry). Default `false`. |
| `nines_api_key`  | string       | Optional  | Nines partner-API key (`nines_live_…`). Falls back to the `NINES_API_KEY` env var. Nines delivery (the `sku` option on `upload`, and `nines_upload`) is enabled when this is set and an org slug is available — `nines_organization_slug` here, or a per-request `shots_organization_slug` (which is how one machine serves multiple orgs). |
| `nines_organization_slug` | string | Optional | Sent as `shots_organization_slug` on every Nines call — the brand you upload for (list valid slugs with the API's `GET /api/v1/organizations`). If a call is refused with 403, the log names the orgs the key can actually reach (fetched once), so a wrong slug is an actionable error rather than a per-shot mystery. |
| `nines_base_url` | string       | Optional  | Nines API base URL. Default `https://review-app.ninesstyle.com`. Must be `https://` (the API key rides every request as a bearer token); `http://` is accepted only for `localhost`. |
| `nines_retry_first_delay_s` | number | Optional | How long after a failed Nines delivery the first re-attempt is scheduled. Default `3`. Spread ±20% so a fleet coming back from one outage doesn't re-hit the API in lockstep. When the API answers with a `Retry-After` header (e.g. on a 429), that wait is honored as a floor on the backoff. |
| `nines_retry_max_delay_s` | number | Optional | Ceiling on the doubling backoff between re-attempts. Default `300`. |
| `nines_retry_max_attempts` | int  | Optional  | Total delivery attempts including the inline one, after which the delivery is abandoned and its file left on disk. Default `6` (delays of 3, 6, 12, 24, 48s). `1` disables retrying. |
| `nines_retry_journal` | string | Optional | Where pending retries are written so they survive a module restart. Default `<output_dir>/.nines_retry_queue.json`; with no `output_dir` there is nowhere obvious to put it and persistence is off (the queue still works, it just forgets on restart). |
| `upload_dial_timeout_s` | number | Optional  | Deadline for authenticating to app.viam.com on the first `upload`. Default `30`. Also bounds the small Nines lookup/create calls. |
| `upload_file_timeout_s` | number | Optional  | Per-file deadline for an `upload` transfer (a stalled file fails on its own rather than wedging the submit). Default `180`. Also bounds the Nines image append. |

If no `ccm` is given, the component passes images through unchanged (identity
matrix); the RAW develop still runs (demosaic + export) but applies no color
correction. To get a matrix, run `calibrate_color` and copy the returned `ccm`
into this attribute.

### Export file naming

Exports are written to `output_dir` (or next to the source) as
`<stem><suffix>`: `tiff16` → `_16.tif`, `tiff8` → `.tif`, `jpeg` → `.jpg`,
`png16` → `_16.png`, `png8` → `.png`. When the **source itself** is a JPEG/PNG/
TIFF (not a RAW), exports get a `_corrected` suffix so the original is never
overwritten.

## DoCommand

### Calibrate

Place the ColorChecker Classic in frame, then fit a CCM. With `use_capture:
true` it calibrates from a full-resolution still via the source camera's
`capture` command; otherwise it uses the source's live/streaming frame. The
fitted matrix is applied immediately and returned — copy it into the `ccm`
config attribute to persist it across restarts.

```json
{
  "calibrate_color": {
    "use_capture": true
  }
}
```

Options: `use_capture` (bool), `capture_options` (object forwarded to the source
capture, default `{"af": true}`), `white_balance` (used when developing the RAW
for calibration), `patch_centers` (24 `[x, y]` pixel coords in ColorChecker
order — use this when the chart does not fill the frame), `radius` (int, patch
sampling radius).

Returns the fitted `ccm`, the measured `white_balance`, and `exposure_stops` —
copy all three into the matching config attributes to persist the calibration.
`exposure_stops` is the brightness offset the chart implied vs. the reference;
setting it renders captures at the calibrated brightness without re-shooting (use
it when the flash can't reach the reference optically). Also returns a `delta_e`
quality report (mean/max color error before vs. after) and a `neutral_brightness`
readout (measured vs. reference per grey patch).

### Capture a corrected still

Triggers a full-resolution still on the source camera, develops it through the
pipeline, and writes the exports + sidecar. Returns a small base64 JPEG
**preview** (the full-resolution image stays on disk).

```json
{
  "capture": {
    "capture_options": { "af": true },
    "white_balance": "camera",
    "exposure_stops": 0,
    "output_formats": ["tiff16", "jpeg"]
  }
}
```

Options (all optional): `capture_options` (forwarded to the source's `capture`),
`white_balance`, `exposure_stops` (exposure compensation applied at the raw
stage), `tone` (delivery look: `none`/`medium`/`bright`), `sharpen` (capture
sharpening: `none`/`light`/`medium`/`strong`), `demosaic` (RAW demosaic
algorithm), `output_formats`, `output_dir`. Each overrides the config default
for this call.

> **Sharpness:** RAW captures are soft before sharpening — every developer
> (Capture One, Lightroom) applies a default capture sharpen, so an unsharpened
> export looks blurry next to theirs. Set `sharpen: medium` to match. The
> `demosaic` default (`DHT`) is also sharper than libraw's stock AHD.

> **Accurate vs. the Capture One look:** with `tone: none` (default) the pipeline
> is colorimetric — a mid-grey card lands on its true sRGB value (~160), which
> looks darker than Capture One because C1 applies a default tone curve that lifts
> midtones (mid-grey ~200). Set `tone: bright` to reproduce that lift (or `medium`
> for roughly half). The CCM is untouched, so hue stays accurate — only
> lightness/contrast changes.

Returns:

```json
{
  "source_path": "/photos/IMG_0042.CR3",
  "exports": { "tiff16": "/photos/IMG_0042_16.tif", "jpeg": "/photos/IMG_0042.jpg" },
  "sidecar": "/photos/IMG_0042.json",
  "ccm_applied": true,
  "color_space": "sRGB",
  "image_base64": "<downsized JPEG preview>",
  "mime_type": "image/jpeg"
}
```

### Develop existing files (no camera)

Point at a RAW or image file already on disk — no camera trigger needed. Takes
the same options as `capture`.

```json
{ "develop": { "path": "/photos/IMG_0042.CR3" } }
```

Batch several files at once with `paths`:

```json
{ "develop": { "paths": ["/photos/a.CR3", "/photos/b.CR3"], "output_dir": "/exports" } }
```

A single `path` returns the same shape as `capture` (including a preview). A
`paths` list returns `{"developed": [ ...per-file results... ], "count": N}`
with previews omitted to keep the response small.

### Upload to Viam

Uploads files already on disk straight to the Viam cloud, tagged for retrieval —
so full-resolution RAW/TIFF/JPEG never have to travel back through a browser to
be saved. Pass every path you want stored together (e.g. all the files sharing a
capture's stem) and a tag like the SKU.

```json
{
  "upload": {
    "paths": ["/photos/IMG_0042.CR3", "/photos/IMG_0042_16.tif", "/photos/IMG_0042.jpg"],
    "tags": ["sku:ABC123", "ABC123"],
    "sku": "ABC123"
  }
}
```

Options: `paths` (required), `tags`, `name` (operator-chosen stem replacing the
capture stem on every file), `sku` (deliver to Nines — see below), `part_id`
(override the configured/env part id), `component_name` (camera to associate
the data with; defaults to this component's name), `delete_after_upload`
(override the config attribute), `upload_images_to_viam` (default `true`; when
`false` only the `.json` sidecars in `paths` are archived to Viam, and the
image files — reported back under `skipped_viam` — are left to the Nines
delivery, still deleting under `delete_after_upload` as if they had uploaded).
Authentication uses the `VIAM_API_KEY` /
`VIAM_API_KEY_ID` that Viam injects into the module process — no credentials
need configuring, but the machine must be cloud-connected. Returns
`{"uploaded": [...paths], "count": N, "failed": [{"path", "error"}], "deleted":
[...paths]}` — a failed file is reported but does not abort the others, and is
never deleted locally.

When `sku` is set and the `nines_*` attributes are configured, the set's
delivery image — the full-res JPEG by preference (then 8-bit PNG, 16-bit PNG,
webp/gif; the RAW/TIFF/sidecar are Viam-archival only) — is also appended to
the Nines product whose `external_id` is the SKU, creating the product on
first use. The result lands under a `nines` key in the response:
`{"reference_item_id", "external_id", "added_count", "images_count"}` on
success, `{"error": …}` on failure, or `{"skipped": …}` when Nines isn't
configured. A Nines failure never marks the Viam uploads failed, and the
delivery image is kept on disk for retry even with `delete_after_upload`.

A delivery that failed on something transient — the API unreachable, timed
out, rate-limited, or a 5xx — is retried in the background, and the response
carries a `retry` block next to the `error`:
`{"job_id", "attempt", "next_attempt_in_s", "queued"}`. The first re-attempt
is about three seconds out, and each further failure doubles the gap. If other
deliveries are already waiting, a failed one goes *behind* them rather than
ahead — one product Nines refuses can't stall the rest of a shoot — so
`next_attempt_in_s` is a floor rather than a promise; `queued` above 1 says so.
When a re-attempt succeeds, the delivery image that was held back is finally
removed if that upload asked for `delete_after_upload`; when the attempts run
out, the file stays on disk to be sent by hand with `nines_upload`. Track both
with `nines_status`.

A failure that *can't* improve — a bad key, the wrong organization, an image
the API rejected — is reported as `error` alone, with no `retry` block.
Retrying it would only delay the moment the operator finds out.

Pending retries are written to `nines_retry_journal`, so a module restart or a
power cycle mid-shoot doesn't drop deliveries whose images are still sitting on
disk waiting for them. They come back on the next `reconfigure` and resume on
the first command after it — rescheduled from now, so a machine that was off
for an hour retries immediately rather than sitting out a backoff it already
served. A restored delivery always checks the product before appending: there
is no way to know whether the attempt that was in flight when the process
stopped reached Nines. The journal carries the product's image count from
*before* that attempt along with the job, so the check across a restart is the
same arithmetic as one in memory — without it a restored retry would read the
count after its own committed append and deliver the shot twice. The journal is removed once nothing is outstanding, and
a damaged one is reported and ignored rather than blocking configuration.

Deliveries whose answer was lost (a timeout or a 5xx, where the API may have
committed the append before the connection dropped) are checked against the
product's existing images before being re-sent, so a retry doesn't leave a
duplicate. The check is arithmetic — how many images the product had before
the attempt versus after — which costs one extra `GET
/api/v1/reference_items/:id` the first time a pre-loaded product is delivered
to in a session, and nothing after that (the append response keeps the count
current, and a product created by delivery starts from zero). Where the count
is unavailable the check falls back to matching image tags, which can only
prove a batch *absent*, never present; an image whose fate can't be determined
is re-sent anyway and the log says so — losing a shot is worse than a
duplicate a human can delete.

### Deliver to Nines (manual / retry)

Appends image files already on disk to a Nines product — the manual
counterpart to the `sku` option on `upload`. Sends exactly the files listed
(each must be jpeg/png/webp/gif), non-destructively, with no Viam upload and
no local deletion. Requires the `nines_api_key` attribute plus an org slug —
`nines_organization_slug` in config, or `shots_organization_slug` in the
command. The whole batch is held in memory base64-encoded while it uploads,
so send a very large set in a few calls rather than one.

```json
{
  "nines_upload": {
    "sku": "ABC123",
    "paths": ["/photos/front.jpg", "/photos/back.jpg"],
    "tags": ["on-model"],
    "product_name": "Northwood Chore Coat"
  }
}
```

Options: `sku` (required — the product's `external_id`, upserted on first
use), `paths` (required), `tags` (applied to every appended image),
`product_name` (display name if the product doesn't exist yet; defaults to
the sku). Returns `{"reference_item_id", "external_id", "added_count",
"images_count"}`.

### Nines retry status

Reports the Nines deliveries still waiting to be re-attempted, and the ones
recently given up on. Takes no options — a queued retry is otherwise invisible
after the `upload` response that announced it.

```json
{"nines_status": {}}
```

Returns `{"pending": [...], "pending_count": N, "abandoned": [...]}`. Each
pending entry is `{"job_id", "sku", "org", "attempt", "next_attempt_in_s",
"files", "error"}` — `files` being the local copies held back for the retry.
Each abandoned entry (the most recent 32) is `{"job_id", "sku", "org",
"attempts", "error", "files"}`; those files are still on disk and can be
re-sent with `nines_upload`.

### Delete local files

Removes files from the machine's disk — skipped captures that were never
uploaded, or sets already confirmed in the cloud. Nothing else in the pipeline
deletes local files, so without this (or `delete_after_upload`) the download
directory grows by every frame ever shot.

```json
{ "delete": { "paths": ["/photos/IMG_0041.CR3"] } }
```

Guarded: requires `output_dir` to be configured, and only files inside it can
be deleted (symlinks are resolved before the check). Returns `{"deleted":
[...], "count": N, "missing": [...], "failed": [{"path", "error"}]}` —
already-missing files land in `missing`, so retrying a cleanup is harmless.

## Typical workflows

**Live capture (with the `ptp` model):**

1. Configure the `ptp` component with a `download_dir`, and configure
   `color-correction` with `camera` pointing at it. Both must run on the same
   machine (shared filesystem).
2. Frame the ColorChecker and run `calibrate_color` (`use_capture: true`) once;
   check the returned `delta_e.after.mean` is low (a few units).
3. Copy the returned `ccm` into the `ccm` config attribute and save.
4. Run `capture` to shoot, develop, and export in one step.

**Develop a folder of existing RAWs:**

1. Configure `ccm` (and `output_dir` if you don't want exports next to the
   originals).
2. Call `develop` with `paths` listing the CR3 files. Each gets its exports +
   sidecar; the RAWs are left untouched.
