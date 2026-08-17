"""
Focus sharpness score for focus hill-climbing / sweep evaluation.

Usage:
    python focus_score.py frame1.ARW [frame2.ARW ...]

Prints one line per file: Laplacian-variance and Tenengrad scores computed on
a fixed center window of the green channel, straight off the raw mosaic (no
demosaic, no develop settings involved). Higher = sharper. Scores are only
comparable between frames of the SAME scene at the SAME framing/exposure -
use them to rank a focus bracket, not as an absolute quality number.

Options:
    --window-frac F   center window size as a fraction of the short edge
                      (default 0.25)
"""

import argparse
import os

import cv2
import numpy as np
import rawpy


def sharpness(path: str, window_frac: float, cx: float = 0.5, cy: float = 0.5) -> dict:
    with rawpy.imread(path) as raw:
        mosaic = raw.raw_image_visible.astype(np.float32)
        colors = raw.raw_colors_visible
        black = float(np.mean(raw.black_level_per_channel))

    # Green channel at half resolution: every CFA quad contributes one G pixel.
    green = np.where(colors == 1, mosaic, np.nan)
    g = np.nanmean(
        np.stack([green[0::2, 0::2], green[0::2, 1::2],
                  green[1::2, 0::2], green[1::2, 1::2]]), axis=0)
    g = np.nan_to_num(g - black, nan=0.0)

    h, w = g.shape
    half = int(min(h, w) * window_frac / 2)
    row = min(max(int(h * cy), half), h - half)
    col = min(max(int(w * cx), half), w - half)
    win = g[row - half:row + half, col - half:col + half]
    # Normalize so flash-power / exposure differences don't masquerade as
    # sharpness changes.
    win = win / (win.mean() + 1e-6)

    lap = cv2.Laplacian(win, cv2.CV_32F)
    gx = cv2.Sobel(win, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(win, cv2.CV_32F, 0, 1)
    return {
        "laplacian_var": float(lap.var()),
        "tenengrad": float(np.mean(gx * gx + gy * gy)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank focus-bracket frames by sharpness")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--cx", type=float, default=0.5,
                        help="window centre x as a fraction of width (default 0.5)")
    parser.add_argument("--cy", type=float, default=0.5,
                        help="window centre y as a fraction of height (default 0.5)")
    args = parser.parse_args()

    results = [(p, sharpness(p, args.window_frac, args.cx, args.cy)) for p in args.paths]
    best = max(r["laplacian_var"] for _, r in results)
    print(f"{'file':<24} {'laplacian_var':>14} {'tenengrad':>12}")
    for p, r in results:
        marker = "  <-- sharpest" if r["laplacian_var"] == best else ""
        print(f"{os.path.basename(p):<24} {r['laplacian_var']:>14.5f} "
              f"{r['tenengrad']:>12.5f}{marker}")


if __name__ == "__main__":
    main()
