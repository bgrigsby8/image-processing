"""
RAW highlight-clipping report (offline; no develop pipeline involved).

Usage:
    python raw_clipping_report.py frame1.ARW [frame2.ARW ...]

Reads the undemosaiced sensor data (``raw_image_visible``) and, per CFA
channel (R / G / B / G2), reports:
  * % of pixels at/above white_level minus a small margin (clipped)
  * the 99.9th-percentile level as a fraction of full scale (black-subtracted)
  * headroom to clipping, in stops, from that percentile

With exactly two files the second is compared against the first: the
per-channel delta in stops is log2(p99.9_B / p99.9_A), so an f/11 baseline
followed by an f/8 frame of the same scene should read ~+1.0 if the only
change was the aperture. This measures the sensor directly - white balance,
exposure_stops, and the CCM never touch these numbers.

Options:
    --margin-frac F   clip threshold is white_level - F * (white - black)
                      per channel (default 0.02)
    --percentile P    percentile used for the headroom/delta readout
                      (default 99.9)
"""

import argparse
import os
import sys

import numpy as np
import rawpy

CHANNEL_NAMES = {0: "R", 1: "G", 2: "B", 3: "G2"}


def channel_stats(path: str, margin_frac: float, percentile: float) -> dict:
    """
    Per-CFA-channel clipping stats for one RAW file.

    Returns {channel_name: {"black": .., "white": .., "clipped_pct": ..,
    "pctl_frac": .., "headroom_stops": ..}} where ``pctl_frac`` is the
    black-subtracted percentile level as a fraction of (white - black).
    """
    with rawpy.imread(path) as raw:
        mosaic = raw.raw_image_visible.astype(np.float64)
        colors = raw.raw_colors_visible
        blacks = list(raw.black_level_per_channel)
        # Prefer the camera-reported saturation point when present; libraw's
        # generic white_level overstates it on some bodies (incl. some Sonys),
        # which would hide real clipping.
        cam_white = getattr(raw, "camera_white_level_per_channel", None)
        whites = list(cam_white) if cam_white else [raw.white_level] * 4

    stats = {}
    for ch in np.unique(colors):
        name = CHANNEL_NAMES.get(int(ch), str(ch))
        black, white = float(blacks[ch]), float(whites[ch])
        span = white - black
        values = mosaic[colors == ch]

        threshold = white - margin_frac * span
        clipped_pct = 100.0 * float(np.count_nonzero(values >= threshold)) / values.size
        pctl_frac = (float(np.percentile(values, percentile)) - black) / span
        pctl_frac = max(pctl_frac, 1e-6)
        stats[name] = {
            "black": black,
            "white": white,
            "clipped_pct": clipped_pct,
            "pctl_frac": min(pctl_frac, 1.0),
            "headroom_stops": -np.log2(min(pctl_frac, 1.0)) if pctl_frac < 1.0 else 0.0,
        }
    return stats


def print_report(path: str, stats: dict, percentile: float) -> None:
    print(f"\n{os.path.basename(path)}")
    print(f"  {'ch':<3} {'black':>6} {'white':>6} {'clipped %':>10} "
          f"{f'p{percentile}':>8} {'headroom':>9}")
    for name, s in stats.items():
        print(f"  {name:<3} {s['black']:>6.0f} {s['white']:>6.0f} "
              f"{s['clipped_pct']:>9.4f}% {s['pctl_frac']:>7.1%} "
              f"{s['headroom_stops']:>7.2f} st")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-channel RAW clipping / headroom report"
    )
    parser.add_argument("paths", nargs="+", help="RAW file(s); with exactly two, "
                        "the second is compared against the first")
    parser.add_argument("--margin-frac", type=float, default=0.02)
    parser.add_argument("--percentile", type=float, default=99.9)
    args = parser.parse_args()

    all_stats = []
    for path in args.paths:
        if not os.path.exists(path):
            sys.exit(f"file not found: {path}")
        stats = channel_stats(path, args.margin_frac, args.percentile)
        print_report(path, stats, args.percentile)
        all_stats.append((path, stats))

    if len(all_stats) == 2:
        (path_a, a), (path_b, b) = all_stats
        print(f"\ndelta ({os.path.basename(path_b)} vs {os.path.basename(path_a)}), "
              f"log2 of p{args.percentile} ratio:")
        for name in a:
            delta = np.log2(b[name]["pctl_frac"] / a[name]["pctl_frac"])
            note = "  (at/near clip - understated)" if b[name]["pctl_frac"] >= 1.0 else ""
            print(f"  {name:<3} {delta:+.2f} stops{note}")


if __name__ == "__main__":
    main()
