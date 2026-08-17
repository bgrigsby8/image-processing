# Option 3 scope — matching Capture One's saturated-color rendering

After Options 1+2 (luminance-only `c1` tone curve + per-session exposure), the
neutral ramp matches C1 within ~2 L and overall error is ~7.2 dE. This documents
what the remaining ~7 dE actually is and what it would take to close it.

## What the residual actually is (measured, updated_exposure.CR3 vs reference_c1.tif)

- **0/24 chart patches fall outside the sRGB gamut.** For the ColorChecker, the
  gap is NOT sRGB clipping — it's a color *rendering* difference.
- **Hue already matches** (within ±4.5°, mostly ±2°). Leave hue alone.
- **The gap is saturation, and it is hue-dependent and bidirectional:**
  - repo *more* saturated than C1: yellow +15, yellow-green +10, bluish-green
    +10, orange +5, magenta +5
  - repo *less* saturated than C1: blue −17, purple −9, foliage −9, red −8,
    dark skin −7

**Consequences for the approach:**
- A global saturation/vibrance stage CANNOT work (opposite signs per hue).
- Re-fitting the 3×3 CCM CANNOT work (already the least-squares optimum; a linear
  map can't bend saturation per-hue without wrecking hue/lightness).
- Closing it requires a **per-hue (nonlinear) color transform** — i.e. a real
  camera-profile-style mapping. This is also philosophically a shift from
  "colorimetrically accurate" to "match C1's look."

There are two *independent* levers. Pick either, both, or neither.

---

## Lever A — wider-gamut output (delivery container)

**What it fixes:** real-world ultra-saturated subjects (a vivid shirt, saturated
fabrics) that genuinely fall outside sRGB. The *chart* doesn't need this (0 OOG),
but real subjects can. C1 delivers Adobe RGB, which holds those colors.

**What it does NOT fix:** the chart's saturation gap above — that's in-gamut, so
a wider container alone won't change it.

**Work:**
- Add an `output_color_space` option (`srgb` default | `adobe_rgb` | maybe
  `display_p3` / `prophoto`).
- In the export encode (`image_io.py`, currently hardcoded sRGB:
  `linear_to_srgb` + `_srgb_icc_bytes`): convert linear sRGB-primary RGB → the
  target's linear primaries (a 3×3), apply that space's transfer function (Adobe
  RGB gamma ≈ 2.199), and embed the matching ICC profile bytes.
- Restrict wide gamut to **16-bit** outputs (8-bit Adobe RGB / ProPhoto band
  badly); warn or refuse for jpeg/png8/tiff8.

**Effort:** small–moderate (one focused change to the encode path + ICC bytes +
config plumbing + tests).

**Risk / caveat:** wider gamut only looks better in a **color-managed** viewer.
In a non-managed app (lots of web/social), Adobe RGB content displays *less*
saturated. So this is a delivery decision tied to where the images go, not a
free quality win.

---

## Lever B — match C1's color rendering (per-hue transform)

This is what closes the chart residual. Options, best-to-worst fit for our case:

### B1. Root-polynomial CCM, fit to C1 (RECOMMENDED if we pursue Lever B)

Replace the 3×3 linear fit with a **root-polynomial regression** (Finlayson
2015): expand each pixel `[r,g,b]` into exposure-invariant terms
`[r, g, b, √(rg), √(gb), √(rb), ...]` (6 terms at degree 2, more at degree 3) and
fit a 3×N matrix by least squares — still a closed-form solve, just more columns.
Unlike a 3×3, this *can* bend saturation per-hue (push blue, pull yellow).

To get C1's look specifically, fit the target = **C1's rendered patch values**
(paired data we already extract in `compare_renditions.py`) instead of the
colorimetric Calibrite reference.

**Work:**
- `color_correction.py`: add `_fit_root_polynomial`, a polynomial-expansion
  apply in `ColorCorrector` (hold N coeffs, not a 3×3), config schema for the
  coefficients (new field; keep 3×3 `ccm` working for back-compat), and
  `validate_config`.
- A calibration path that fits against a C1 reference TIF: detect the chart in
  both the repo render and the C1 export, pair the 24 patches, fit. The pairing
  + Lab machinery already exists in `compare_renditions.py`.

**Effort:** moderate (~half-day focused + careful validation).

**Risk:** **overfitting on 24 patches.** A degree-2 root-polynomial has 6 terms ×
3 = 18 coeffs fit on 24 points — tight. Degree 3 will overfit. Mitigations:
- keep it degree 2,
- hold out patches and check out-of-sample error,
- **always sanity-check on a real subject** (the pink shirt) — a profile that
  scores great on the chart can still do something ugly to a color the chart
  doesn't sample. This is the main reason to be cautious here.
- ideally shoot a denser target (see B2 note) for a trustworthy fit.

### B2. 3D LUT fit to C1 (most accurate, most work)

Bake a 3D LUT (e.g. 33³) mapping repo→C1. Needs **dense** paired data — 24
patches is far too sparse. Would require shooting a ColorChecker Digital SG (140
patches) or several charts, scattered-data interpolation (RBF / thin-plate
spline — note: `scipy` is NOT currently installed) to fill the cube, then
trilinear apply at runtime and LUT storage (a file, not inline config).
**Effort:** high. Only worth it if B1 proves too coarse.

### B3. Per-hue saturation stage (hand-tuned, vendor-independent)

A small HSL-style stage with a saturation multiplier per hue band (push
blue/red/purple, pull yellow/green — the signs we measured). Less principled
than B1, but simple, fully controllable, exposure-independent, and doesn't tie
you to C1. Good if the goal is "pleasing/punchy" rather than "bit-exact C1."
**Effort:** low–moderate. Tuning to *match C1 exactly* is fiddly; tuning to
taste is quick.

### B4. Import C1's actual profile — likely NOT viable

C1's camera profiles are proprietary and generally not exportable as a standard
ICC/DCP we could apply. DCP (Adobe) profiles carry hue-twist tables that need a
DNG/DCP rendering pipeline we don't have. Mentioned for completeness; don't
pursue unless C1 can export a usable profile.

---

## Recommendation & phasing

1. **First decide if this is worth doing at all.** 7.2 dE with hue matched and
   neutrals within 2 L is already a strong match; everyday colors (skin, the
   pink shirt, neutrals) are in-gamut and land well. Option 3 chases saturated
   blues/yellows/purples and carries real overfit risk. Only pursue if those
   specific colors matter for the work.
2. If yes, and the goal is **match C1**: do **B1** (root-polynomial fit to C1),
   degree 2, with held-out + real-subject validation. Consider shooting a denser
   chart first so the fit is trustworthy.
3. If the goal is just **"more punch, my own look"** (not bit-exact C1): do
   **B3** (per-hue saturation) — simpler, no C1 dependency.
4. Do **Lever A** (Adobe RGB export) only if you actually deliver to
   color-managed wide-gamut destinations and have real subjects exceeding sRGB.
   It's orthogonal to B and can be added anytime.

## Decisions needed before implementing

- **Target:** bit-exact C1 (→ B1/B2) or your own pleasing look (→ B3)?
- **Chart:** stick with the 24-patch Classic (limits us to a low-order fit) or
  shoot a denser target (ColorChecker SG) for a robust profile?
- **Delivery:** sRGB only, or do you need wide-gamut files (→ Lever A)?
- **Accept the philosophy shift?** B1/B2 fit to C1 means giving up colorimetric
  accuracy in favor of matching a vendor's look. (The CCM today is traceable to
  the Calibrite reference; a C1-fit profile is not.)
