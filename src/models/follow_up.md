# f/8 follow-up — programmatic test plan (handoff for Claude Code)

Context: Nines AI capture rig. Sony A7R V + FE PZ 16-35mm F4 G at 16mm, Profoto Pro-D3 strobe,
Viam modules in `~/projects/image-processing` (color-correction / develop pipeline) and the
in-progress `sony-remote` camera control. Aperture just changed f/11 → f/8 (same strobe power,
same scene). Observed: `exposure_stops` 1.2 → 2.4, images ~1 stop brighter, and the stored
focus setpoint no longer lands right.

Work through these in order. Tasks 1–2 are code reading + offline RAW analysis (no hardware).
Tasks 3–5 need the camera.

---

## 1. Pin down what `exposure_stops` means (image-processing repo)

- Find where `exposure_stops` is computed in `~/projects/image-processing` and answer:
  - What is it measured against (gray patch? raw histogram percentile? fixed target level)?
  - Is it *informational only*, or is it applied as a gain/normalization during `develop`?
- If it IS applied during develop, explain why the output images still look brighter at f/8
  (candidates: gain applied before/after clipping, preview path skips it, cap on correction,
  or it's applied in a nonlinear space).
- Deliverable: one paragraph on semantics + whether the +1.2 stop reading is "raw sensor got
  1.2 stops more light" (expected) vs. something pipeline-side.

## 2. RAW clipping report (offline, rawpy)

Brightness itself is fixable by cutting strobe power; **clipped highlights are not**. Write a
small script that, for a given ARW file:

- Loads with rawpy, reads `raw_image_visible` + per-channel black/white levels.
- Reports per-channel: % of pixels at/above (white_level − small margin), and the 99.9th
  percentile level as a fraction of full scale.
- Run it on matched f/11 and f/8 frames of the same scene and report the delta in stops
  (log2 of the percentile ratio) — this independently verifies the +1.2 stop number and shows
  whether f/8 @ current strobe power is actually clipping (check speculars on glossy product
  surfaces and the white ColorChecker patch especially).

Acceptance: a table per frame (R/G/B clipped %, headroom in stops). If headroom < ~0.3 stop
or clipping > ~0.01% on the subject, the strobe must come down before any color work.

## 3. Focus diagnosis (hardware; use whatever control path currently works — sony-remote build or Sony RemoteCli)

The stored focus setpoint "being wrong" has three distinct candidate causes. Test in this order:

**3a. Did zoom move?** Focus raw values on a zoom lens are zoom-dependent. Read the zoom
position and compare with the value recorded when the focus setpoint was stored. If zoom
drifted (power cycle, PZ creep), that alone explains it. Log zoom alongside every focus reading
from now on.

**3b. Does the focus scale survive power cycles / reconnects / aperture changes?**
Focus-by-wire lenses have no absolute encoder guarantee across power-ups. Harness:

- set_focus(X) → read back, ×20 → report spread (repeatability baseline)
- set aperture f/11 → read focus → set aperture f/8 → read focus (does the aperture command
  itself disturb the reported/actual position?)
- power-cycle or USB-reconnect the camera ×5; after each: read focus, then set_focus(X),
  read back → does the same raw X land in the same place? (Verify with 3c's sharpness metric,
  not just readback — readback can be self-consistent while the physical mapping shifted.)

**3c. Focus-sweep sharpness harness** (also the fix, not just the diagnosis):

- Put a detail target (or product) at the working distance.
- Capture a bracket of ~15–20 focus positions around the stored setpoint (use the JPEG or a
  center crop of the raw; compute Laplacian variance or Tenengrad on a fixed center window).
- Plot metric vs. focus position → the peak is the correct setpoint at f/8.
- Repeat at the near (~0.5 m) and far (~2 m) stations. Compare peaks with the old stored values.

Interpretation guide:
- Peak moved by roughly the same raw offset at all distances → scale shifted (3b problem):
  re-zero on connect (e.g., AF once on a fixture, or drive to a mechanical end stop and re-derive).
- Old setpoint still peaks at the near station but the 2 m station is soft → not a lens problem
  at all: it's the **hyperfocal shift**. A setpoint chosen as "hyperfocal at f/11" no longer
  reaches infinity at f/8 — opening one stop pulls the far DOF limit in (order of 2 m → ~1.4 m
  with the CoC we've been using). Fix: restore focus slightly *farther* (the f/8 hyperfocal,
  ~0.55 m equivalent) or store per-station setpoints from the sweep peaks.

## 4. Strobe power step-down + verification

- Drop the Pro-D3 one stop (its power scale is in stops) and re-run the task-2 clipping report;
  target: `exposure_stops` back to ~the f/11 baseline (~1.2) and comfortable raw headroom.
- Note the new power setting — it becomes part of the machine's capture recipe.

## 5. After power is settled: ColorChecker + CCM refit

Do this LAST (power changes shift flash duration and color temperature, and the Sony needs its
own CCM regardless — the Canon CCM does not transfer):

- Shoot ColorChecker Classic at the final strobe power, f/8, at the working distance.
- Refit the 3×3 CCM via the existing color-correction calibration flow; record ΔE2000 stats.
- Until this is done, don't judge color in any f/8 output — WB + CCM are both stale.

---

## Results so far (2026-08-13, offline tasks)

**Task 1 — done.** `exposure_stops` is computed by `calibrate_color`
(`src/models/color_correction.py:705-738`): a single scalar gain is least-squares
fitted so the neutral-ramp patches (Neutral 8/6.5/5/3.5; white and black are
skipped) match the reference chart luminance in linear light, and
`exposure_stops = log2(gain)` — i.e. "the chart came out X stops darker than
reference". It is informational at calibration time, but once pasted into the
component config it becomes real digital gain: develop applies it at the raw
stage via libraw `exp_shift = 2**stops`, clamped to [-2, +3] stops
(`src/models/image_io.py:228-239`). Calibration itself always measures as-shot
(it loads without exposure_stops). So the f/8 brightness is expected: the sensor
gets ~1 stop more light and develop still adds the configured f/11-era +1.2 on
top. **However, the 1.2 → 2.4 reading is anomalous**: under these semantics more
light must push exposure_stops *down* (toward ~0.2), not up. A 2.4 reading means
the f/8 calibration frame measured ~1.2 stops *darker* than the f/11 baseline —
consistent with the strobe not firing (ambient-only frame), a shutter/sync
change, or the number coming from a different frame than assumed. The task-2
report on the actual frames adjudicates this independently.

**Task 2 — script ready, run on two f/8 chart frames.** `raw_clipping_report.py`
(repo root): `python raw_clipping_report.py f11.ARW f8.ARW` prints per-CFA-channel
clipped %, 99.9th-percentile level, headroom in stops, and the pair delta in
stops. Uses `camera_white_level_per_channel` when present (libraw's generic
white_level overstates saturation on some bodies and would hide clipping).

Run on DSC00110.ARW (f/11) vs DSC00108/DSC00109.ARW (f/8) — all 1/160s ISO 100:

- **f/8 vs f/11 delta: +1.09 stops** (all four channels within 0.01) —
  the aperture change itself behaves exactly as expected. (109 reads +0.85:
  one frame of shot-to-shot light variance, see below.)
- **Zero clipping in any frame**; but these numbers don't certify f/8, see verdict.
- `debug_calibration.py DSC00108.ARW` measures **exposure_stops +2.41**
  (chart white 115 vs reference 245 sRGB); the f/11 frame is 1.09 stops darker
  → would measure **≈ +3.5**, vs the +1.2 the original f/11 calibration
  reported. Today's whole session is ~2.3 stops darker than that baseline.
- **Root cause: the strobe never fired.** EXIF Flash tag = 0x10
  ("off, did not fire") on all three frames — the camera itself suppressed
  flash. These frames are modeling-lamp/ambient only, which also explains the
  perfect +1.09 aperture tracking (continuous light) and 109's ±0.2-stop
  wobble (lamp/ambient flicker at 1/160 s). Likely culprits: camera flash
  mode set to Off, silent/electronic-shutter mode (Sony disables flash
  there — check what the sony-remote PC-remote path selects), or the
  Air trigger off/unpaired.
- Side note: cv2.mcc finds the chart only in the brightest frame (108);
  109/110 are too dark for detection.

**Verdict:** the exposure_stops 1.2 → 2.4 observation was never an aperture or
pipeline effect — the recalibration frames were shot without strobe. Task 4
(power step-down) is on hold: fix flash triggering first, then shoot a
strobe-lit f/11+f/8 pair and re-run the report. Expected with strobe at f/8:
exposure_stops ≈ +0.2; clipping acceptance must be re-judged on that frame —
today's headroom numbers describe ambient light and do not transfer.

**Follow-up (same day, later): the retrieval path is handing back stale
frames.** A frame shot at ~17:47 with the strobe confirmed firing (operator
next to it) arrived as `DSC00111.ARW` — but its EXIF says f/11, 16:36:44,
Flash 0x10, and its pixels match DSC00110 within 0.1 stop: it's the second
shot of the earlier f/11 ambient pair, not the new capture. Capture-time
EXIF vs download mtime for all four files:

| file | EXIF time | f-stop | downloaded |
| --- | --- | --- | --- |
| DSC00108 | 17:13:33 | f/8 | 17:14 |
| DSC00109 | 17:13:40 | f/8 | 17:15 |
| DSC00110 | 16:36:37 | f/11 | 17:40 |
| DSC00111 | 16:36:44 | f/11 | 17:47 |

File numbers run *backwards* in capture time (110/111 predate 108/109) —
same smell as the Canon dual-slot fallback bug (file numbering diverges
across slots/folders, retrieval picks lexically-next file). Every conclusion
above about "the strobe never fired" describes the stale frames, not
necessarily what the camera actually shot; the real strobe-lit frames are
presumably still on the card. **Fix/verify file retrieval before any further
exposure work — and note the focus-setpoint symptom is equally suspect if
sweep evaluations were run on stale downloads.**

**Follow-up 2: correct frame analyzed (DSC00014.ARW, f/11, 17:39:59, strobe
visibly fired) — the flash is firing but its light is not in the exposure.**
(DSC00015.ARW, the f/8 mate, analyzed after re-download: f/8, Flash 0x10,
exposure_stops **+2.24**, +0.86 stops vs its f/11 mate, zero clipping — same
strobe-less pattern; the pair reproduces the original "1.2 → 2.4" observation
shifted by the missing flash.)

- DSC00014 is only **+0.2 stops** brighter than the ambient-only f/11 frame
  (DSC00110), and its chart measures exposure_stops **+2.96** vs the original
  f/11 baseline of +1.2. If the strobe were contributing at baseline power the
  frame would be ~1.8 stops brighter. EXIF still says Flash 0x10
  ("off, did not fire") — the camera did not command a hot-shoe flash.
- Row-profile analysis (raw row means, frame A vs frame B in stops): every
  consecutive pair — including the two "ambient" f/8 frames 7 s apart — shows
  a smooth, bottom-of-frame-heavy brightness gradient that changes shot to
  shot (up to ~+0.5 stop within a pair, +1.0 vs the hour-earlier frame), with
  no periodic banding (residual ~0.05 stops rms). A smooth shot-varying
  vertical gradient = a light source varying *during* a slow rolling readout
  → consistent with **electronic shutter** (A7R V full-res e-shutter readout
  is slow, and Sony does not support flash with it) plus a mistimed flash
  burst landing mid-readout.
- Working hypothesis: the capture path (sony-remote / PC-remote or the
  camera's Shutter Type menu) is using the silent/electronic shutter. The
  camera therefore suppresses flash sync; the burst the operator sees is
  mistimed relative to the row exposure windows and contributes only a weak
  smeared gradient.
- **Decisive 30-second test at the rig:** set Shutter Type to Mechanical (or
  EFCS), shoot one f/11 frame, re-run `debug_calibration.py`. Expect EXIF
  Flash to read "fired" and exposure_stops to drop from ~+3.0 to ~+1.2.
  Open question if that fails: how is the Pro-D3 triggered — hot-shoe Air
  remote, or a separate software/BLE command from the capture flow?

**Follow-up 3: strobe-off control test.** Operator turned the Pro-D3 fully
off and reshot; the live image went much darker — but the downloaded file
(`DSC00018.ARW`) is EXIF 15:23:46 (hours before the test), f/11, and
pixel-identical (±0.03 st) to DSC00014. Stale-retrieval strike three. Two
conclusions:

- **File numbering collides across folders/slots**: DSC00014/15 (17:39) vs
  DSC00018 (15:23) cannot share a counter — the camera has multiple
  `DSC000xx` sequences, so any name-based "newest file" retrieval is
  fundamentally broken. Needs the same pre-trigger snapshot-diff fix as the
  Canon module.
- **The modeling lamp is the scene's dominant light.** Strobe head off →
  much darker live image, while all frame data shows the flash *burst* never
  lands in the exposure. So captures have been exposing under the Pro-D3's
  modeling LED (which also explains the smooth shot-to-shot gradient wobble
  against a rolling e-shutter readout, and the ~1.8-stop deficit vs the
  strobe-lit baseline).

**Follow-up 4 (2026-08-14): mechanical-shutter test — e-shutter theory
retracted.** Shutter Type was mechanical all along (operator). Fresh frame
DSC00002.ARW (f/8, 10:21:33, yet another colliding file counter):
exposure_stops **+2.01**, Flash still 0x10, and row analysis shows **no flash
band anywhere** (sharpest smoothed row step 0.002 stops; a burst clipped by
curtain travel would leave >0.1). At 1/160 s — below the 1/250 sync speed —
any hot-shoe-synced burst MUST land fully in the frame. It doesn't, at all.

Conclusion: **the camera's sync signal is not what fires the strobe.** The
visible flash must come from a trigger path outside the shutter sync (a
software/BLE fire command in the capture flow, a mistimed/delayed remote, an
optical-slave arrangement), or the shoe is disabled (camera Flash Mode: Off —
EXIF 0x10 means "compulsory flash suppression" on every frame all day).
Open rig questions that now decide everything: how is the Pro-D3 physically
triggered; what does the camera's Flash Mode menu show; and which path
downloads the ARWs (sony-remote vs manual) — for the stale-file bug.

**Rig facts (operator, 2026-08-14):** Air remote in hot shoe; Flash Mode =
Fill-flash; files reach the repo via camera→USB save-to-PC directory→manual
rsync. With mechanical shutter at 1/160 (below 1/250 sync) + Fill-flash, a
shoe-synced burst cannot miss the frame — yet Flash=0x10 (compulsory
suppression) on every frame. Something overrides the menus: prime suspects
are **Silent Mode ON** (forces e-shutter + disables flash while Shutter Type
still displays Mechanical) or **Interval shooting** (disables flash). The
stale-file issue is the manual flow, not sony-remote: save-to-PC restarts
numbering per session (counters DSC00108/DSC00014/DSC00002 seen in one day),
so name-based picking grabs old frames — select by mtime and verify EXIF
time.

**Next decisive test: 1/10 s frame.** f/8, ISO 100, strobe on, watch the
strobe, take the newest file by mtime. At 1/10 s even a ±50 ms mistimed burst
lands in-frame. Outcomes: flash in frame + 0x10 → mistimed external trigger;
no flash + none seen → camera suppression (find Silent Mode/Interval); no
flash but visibly fired → trigger path outside the shoe.

---

## RESET — 2026-08-14 controlled 2×2 (supersedes Follow-ups 2–4)

Operator disclosed: the modeling lamp was **always off**, and the +1.2 f/11
baseline has **unknown provenance**. A clean 2×2 was shot (f/11 and f/8,
strobe on/off, 1/160 s ISO 100, verified by EXIF, picked by mtime,
`./images/DSC00002-5.ARW`). Definitive findings:

| measurement | result |
| --- | --- |
| flash contribution over ambient | +5.1 st (f/11), +5.3 st (f/8) |
| ambient floor | negligible (p99.9 ≤ 0.8% full scale) |
| aperture step, flash on | +0.84 st (expect +1.0; wobble) |
| exposure_stops, f/11 + flash | **+2.92** |
| exposure_stops, f/8 + flash | **+2.18** |
| clipping | zero anywhere; f/8 flash-on headroom 1.63 st (G) |
| ΔE after per-frame CCM fit | mean 3.1 / 3.0 (good, < 5) |

**Retractions:** the flash lands in every armed frame and always did —
the "flash absent" (Follow-up 2), e-shutter (2–3, retracted earlier), and
"camera suppresses flash" (4) theories are all withdrawn. Root error: trusting
the unverified +1.2 baseline and the EXIF Flash tag (0x10 appears on
verified flash-lit frames too — it is meaningless with this trigger). The
**stale-file problem is still real** (independently verified; use mtime +
EXIF check, never filenames — save-to-PC restarts numbering per session).

**Current true state:** rig works; at current power the chart is simply
~2.2 stops under reference at f/8. Unexplained residual: ±0.15–0.25 st
shot-to-shot flash variance (check Air remote MAN vs TTL; if TTL, switch to
manual before calibrating).

## Task 3 (focus) — derived 2026-08-14, power-cycle CONFIRMED

Tooling: `derive_focus.py` (client sweep driver) + `focus_score.py` (raw
green-channel Laplacian/Tenengrad scorer), run on the machine against the
sony-remote module's focus DoCommands. Findings along the way:

- **3a (zoom):** not tested directly; setpoint recorded at 16 mm — log zoom
  with every stored focus value.
- **3b (scale stability): the emulated focus scale was NOT repeatable at the
  default 0.03 s nudge interval** — same count landed at different physical
  focus (two-spike sweep curves; 20× score differences at identical counts).
  Mechanism: the body drops/coalesces near_far nudges streamed at 30 ms.
  At `emulated_nudge_interval_s: 0.15` the scale is repeatable: two identical
  fine sweeps matched point-for-point (1–3%), and 5 repeated excursions to
  one position spread 1.1% (scorer noise floor). Rare slips still occur
  (1 outlier in 26 excursions) — auto-retry-on-soft-frame is a sensible
  sony-remote backlog item. Also fixed in sony-remote: focus-motion job
  timeouts now scale with travel×interval (session.py) instead of the fixed
  40 s the slower interval blew through.
- **3c (sweep): setpoint = 88 emulated nudges** (backpack target at working
  distance, f/8, dome peak reproduced at 88 in both fine sweeps, refined
  ~87.5). Scale definition this is valid for: `emulated_step_size: 3`,
  `emulated_nudge_interval_s: 0.15`, `emulated_travel_nudges: 300` (trimmed
  from 600 for homing speed; safe — physical travel ends well below).
  Changing any of the three invalidates the setpoint.
- **Power-cycle confirm passed**: repeat test at slope position 82 after a
  full camera power cycle matched pre-cycle scores within ~0.3% (means 1.259
  vs 1.256; the slope is ~4%/nudge, so the scale re-zeroed to well under a
  nudge). Deploy: `focus_on_connect: 88`.
- **Post-derivation addenda (evening 08-14):** (1) intervals faster than
  0.15 s all drop nudges on this body — 0.03, 0.05, and 0.10 each failed the
  sweep/repeat tests; 0.15 s is the floor, treat connect cost (~58 s) as the
  price of a trustworthy scale. (2) The lens barrel's **AF/MF switch in AF
  silently disables all focus nudges** while the module keeps reporting
  success (dead-flat sweeps at the old defocus floor are the symptom); it got
  bumped during power cycles and cost an hour. sony-remote backlog: surface
  focus mode in get_status and warn/fail focus commands when not MF; also
  auto-retry-on-soft-frame for the rare (~1/26) excursion slip.
- **Native absolute focus is a verified hardware dead end** on this body+lens
  (FocusPositionSetting never appears in GetDeviceProperties with the PZ
  16-35; confirmed against Sony RemoteCli — see crsdk_ext.cpp comments), so
  per-station focus is either hand-focus-per-batch (zero machinery; focus
  holds across power cycles) or emulated setpoints derived by sweep.
  **Hand-focusing breaks the emulated counter silently** (module still
  reports homed + position 0) — after touching the ring, home_focus/reconnect
  before trusting emulated playback. DOF facts settled by eye: at 16 mm f/8
  the 61 MP sharpness bar needs per-distance focus (0.5 m and 1.5 m cannot
  share one setpoint; 0.75 m harmonic-mean focus was not sharp enough at
  either end).

**Recommendation (Task 4, inverted):** *raise* strobe power **+1 stop** →
expected exposure_stops ≈ +1.2 at f/8 with ~0.6 st specular headroom
(meets the ≥0.3 st acceptance). Raising the full 2.2 stops would clip
(headroom is only 1.63 st). Then Task 5: shoot the chart at final power
(2–3 frames to check flash consistency), run `calibrate_color`, paste
ccm/white_balance/exposure_stops into the config. Task 3 (focus) unchanged.

## Task 4 + 5 (offline portion) — DONE 2026-08-14

Strobe raised +1 stop (measured +0.71). Three chart frames at final power
(`high_strobe_images/DSC00006-8.ARW`, f/8, 1/160 s, ISO 100, 2 s apart):

- **Repeatability: 0.01–0.03 st** across frames — earlier ±0.2 st wobble not
  reproduced; strobe behaves like a manual Profoto should.
- **Zero clipping; headroom 0.9 st (G)** — passes the ≥0.3 st acceptance
  with margin for glossy speculars.
- **exposure_stops: +1.57 / +1.56 / +1.54**; ΔE(76) after fit: mean 3.3,
  max 10.8 (bar: mean < ~5). (Plan said ΔE2000; the pipeline computes ΔE76.)

Config values from the calibrate_color fit (frame DSC00007):

```json
"ccm": [[1.125, -0.203, 0.026], [0.008, 0.959, -0.012], [0.031, -0.043, 0.949]],
"white_balance": [2.764, 0.999, 1.451, 1.001],
"exposure_stops": 1.56
```

(WB r=2.76 is a hair above the 2.0–2.5 daylight sanity band — lens/sensor
specific, chart-derived, fine.) Remaining: record the Pro-D3 dial value in
the capture recipe; optionally re-run the module's `calibrate_color`
DoCommand on-machine for the canonical path (values should match).
**Task 3 (focus diagnosis) is now the only open item** — needs the camera,
and captures/downloads can finally be trusted (pick by mtime!).

---

## Reporting

For each task: what was run, numbers observed, and a one-line verdict. The focus harness output
(spread, power-cycle behavior, sweep plots) decides whether sony-remote needs a re-zero-on-connect
step, so keep those raw numbers.
