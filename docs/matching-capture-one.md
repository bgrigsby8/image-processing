# Matching Capture One's "punch"

Notes from diagnosing why this repo's exports looked duller than Capture One (C1).

## The actual problem

C1 images aren't more *colorful* than ours — they're **brighter**, especially in
the mid-tones (about one stop). That brightness is what reads as "punch." A
darker render just looks duller, even when the hue/saturation are correct.

Two related facts, proven by measuring 24 ColorChecker patches in CIELAB
(`compare_renditions.py`, comparing `capture_one.tif` vs a render of
`color_correction.CR3`):

- **Saturation is a red herring.** Adding saturation/vibrance moved every
  variant *further* from C1. Don't add a saturation stage.
- **The old `tone: bright` curve caused the orange→yellow hue shift**, because it
  brightened each R/G/B channel independently. Brightening *lightness only*
  fixes that.

Matching C1's brightness curve (lightness-only) took the mean color error from
~24 dE down to ~7.6 dE — a close match. The remaining ~6.6 dE is **color**, not
lightness: it's C1's own camera profile + its wider Adobe RGB color space vs our
sRGB. That's a fixed floor, not the dull look.

## Option 1 — IMPLEMENTED

1. Made `apply_tone_curve` (in `src/models/image_io.py`) brighten **luminance
   only** (scale RGB by `f(luma)/luma`) instead of per-channel — stops hue
   shifting for any curve.
2. Added a `c1` tone preset fitted from C1's neutral ramp. Current anchors:
   - input  (our sRGB): `[0, 53, 92, 129, 168, 206, 245, 255]`
   - output (C1 sRGB):  `[0, 43, 103, 159, 199, 223, 239, 255]`

   This is an **S-curve**: it deepens the blacks (53->43), lifts the mid-tones,
   and rolls off the highlights (245->239) - which is C1's actual look.

To use it: set `"tone": "c1"` in the color-correction component config.

### Lesson learned (why this matters for Option 2)

The first `c1` fit used an *underexposed* reference shot, so it mistook C1's
midtone lift for one big uniform brightening - it even *lifted* the blacks
(`37->43`). On a properly exposed shot that washed everything out (blacks went
from L17 to L33). Re-fitting from a correctly exposed reference gave the S-curve
above, where the exposure gap turned out to be only ~+0.37 stops.

Takeaway: **a fixed curve fitted to one shot still bakes in that shot's
exposure.** The S-curve shape is robust (anchored at both ends, won't blow out),
but if your shots vary much in brightness the midtone match will drift. That is
exactly what Option 2 fixes.

---

## Option 2 — Do it properly (exposure-consistent)

**VALIDATED from RAW (after_changes.CR3 vs reference_c1.tif):** with this
session's correct exposure (`exposure_stops ~ 0.9`, vs the stale 0.096) **plus**
the `c1` curve, the whole neutral ramp matched C1 within ~1.5 L and overall
error fell to 7.5 dE (the color/gamut floor). Exposure was the missing piece.

**The key finding:** captures within a session are consistent (before/after
shots rendered identically), but `exposure_stops` is **session-specific** -
0.096 had been calibrated on a brighter session, leaving this one ~0.8 stop dark
(darker than even the repo's own colorimetric target). A fixed exposure can't
track a session it wasn't calibrated for; that's what made `c1` look too bright
on one shot and too dark on another.

**The operational fix (no code change):** recalibrate exposure per session, like
white balance. Run `{"calibrate_color": {"path": "<a chart shot>"}}`, copy the
reported `exposure_stops` into config, keep `"tone": "c1"`. The CCM, WB, and the
curve stay put; only `exposure_stops` tracks the lighting.

**Optional code enhancement:** auto-expose every shot to a fixed mid-gray target
at the raw stage so `exposure_stops` doesn't need manual recalibration. Harder
for normal shots (no chart in frame) - would need a metering heuristic (e.g.
percentile of the linear histogram), so it's a real feature, not a one-liner.

**What to do:**
1. Get exposure consistent first — make mid-gray (e.g. ColorChecker Neutral 6.5)
   land at a fixed target brightness on every shot, via `exposure_stops`
   (or an auto-exposure-to-target step at the raw stage).
2. *Then* apply one fixed C1-shaped contrast curve (lightness-only) on top.

This separates "how bright is the shot" (exposure) from "the C1 look" (a fixed
curve), so the look stays consistent regardless of small exposure variation.

**Nice-to-have:** a `calibrate_tone` DoCommand that fits the tone curve from any
C1 reference TIF automatically — same idea as `calibrate_color` does for the
CCM. The fitting logic already exists in `compare_renditions.py`
(`fit_c1_tone_lut`): detect the chart in both images, map our neutral ramp onto
C1's, build a monotone curve.

---

## Option 3 — Match C1's color exactly (wider gamut / profile, NOT implemented)

> Detailed scope with measured data and approach trade-offs:
> [option-3-scope.md](option-3-scope.md).

**Why:** the last ~6.6 dE is C1 using its own camera profile and delivering
**Adobe RGB** (a wider color space) while we deliver **sRGB**. Very saturated
real-world colors (e.g. a vivid shirt) can fall outside sRGB, where C1 keeps
saturation we physically can't in an sRGB file.

**What to do (any/all):**
- Export in Adobe RGB (or ProPhoto) instead of sRGB: change the encode + the
  embedded ICC profile in `image_io.py`'s export path (currently hardcoded sRGB:
  `linear_to_srgb` + `_srgb_icc_bytes`). Our color math is in linear sRGB
  primaries, so this also means converting linear sRGB → the target space before
  encoding.
- Adopt a C1-style 3D color profile (a hue/saturation lookup) instead of just a
  3x3 CCM, if you want C1's exact color rendering rather than colorimetric
  accuracy.

**Caveat:** only matters for very saturated subjects. For everyday shots,
Options 1/2 already match C1. Most clients/web also expect sRGB, so a wider gamut
is a delivery decision, not a quality upgrade by itself.

---

## Tools

- `compare_renditions.py` (repo root) — the measurement script. Re-run it after
  any change to see the dE-vs-C1 table. Edit `CONFIG`, `C1_TIF`, `REPO_CR3` at
  the top to point at new files.
