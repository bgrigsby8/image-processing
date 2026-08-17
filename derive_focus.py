"""
derive_focus.py - automated focus-sweep against the live rig.

Connects to the machine, drives the `sony` camera's absolute focus through a
bracket of positions, captures a frame at each, scores sharpness on the RAW
(same green-channel Laplacian/Tenengrad as focus_score.py), and reports the
peak - the focus setpoint to store for this station.

Usage (typical, from the repo root):

    venv/bin/python derive_focus.py --af --span 20 --steps 15
    venv/bin/python derive_focus.py --center 74 --span 10 --steps 11 --apply

Center of the sweep comes from --center, or --af (autofocus once, then read
back), or the lens's current position (default). Positions are in the module's
focus units (`sdk_raw` or `emulated_nudges` - whatever get_focus_position
speaks; the units are printed at start).

The RAW lands in the machine's capture_dir, so the script needs a way to read
it for scoring:

    --fetch local           saved_to is readable as-is (script runs on the
                            machine, or capture_dir is a mounted/synced path)
    --fetch scp --remote user@host
                            pull each frame with scp (same access your rsync
                            flow uses)

Frames are scored as they arrive, so the sweep prints a live table. Results
are also written to a CSV next to this script. With --apply, focus is driven
to the winning position at the end (otherwise the lens is left at the last
sweep position). Store the winner per station - and record the zoom position
with it; on the PZ 16-35 the focus scale is only valid at the zoom it was
derived at.
"""

import argparse
import asyncio
import csv
import os
import subprocess
import tempfile
import time

from viam.robot.client import RobotClient
from viam.components.camera import Camera

from focus_score import sharpness


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key='9mutj1rhbn3oypsmohhzoci3bqxbuw9f',
        api_key_id='eef423e7-6863-4920-9c07-a514b833bae5'
    )
    return await RobotClient.at_address('nines-photographer.g02wbvukyo.viam.cloud', opts)


def fetch(saved_to: str, args, workdir: str) -> str:
    """Make the captured file readable locally; returns the local path."""
    if args.fetch == "local":
        if not os.path.exists(saved_to):
            raise FileNotFoundError(
                f"{saved_to} is not readable here - if this script isn't "
                f"running on the machine, re-run with: --fetch scp --remote user@host"
            )
        return saved_to
    if not args.remote:
        raise ValueError("--fetch scp needs --remote user@host")
    local = os.path.join(workdir, os.path.basename(saved_to))
    subprocess.run(
        ["scp", "-q", f"{args.remote}:{saved_to}", local],
        check=True, timeout=120,
    )
    return local


async def sweep_positions(args, sony) -> list:
    """Resolve the list of target positions, worst-first backlash approach."""
    if args.positions:
        positions = sorted(int(p) for p in args.positions.split(","))
    else:
        if args.af:
            af = await sony.do_command({"command": "autofocus_once"})
            print(f"autofocus_once: {af}")
        if args.center is not None:
            center = int(args.center)
        else:
            got = await sony.do_command({"command": "get_focus_position"})
            center = int(got["position"])
            print(f"sweep center = current position {center} ({got.get('units')})")
        if args.steps < 3:
            raise ValueError("--steps must be >= 3")
        lo, hi = center - args.span, center + args.span
        positions = sorted({round(lo + (hi - lo) * i / (args.steps - 1))
                            for i in range(args.steps)})
    return positions


# Tenengrad ranks the sweep: it responds monotonically even far from focus,
# where Laplacian variance is flat noise. Both peak at the same position.
METRIC = "tenengrad"


def refine_peak(rows: list) -> float:
    """Sub-step peak estimate: parabola through the best point and neighbours.

    Falls back to the best sampled position when the peak sits on the sweep
    edge (which also means the sweep should be re-centred and re-run).
    """
    best = max(range(len(rows)), key=lambda i: rows[i][METRIC])
    if best in (0, len(rows) - 1):
        print("NOTE: peak is at the edge of the sweep - re-run centred there")
        return float(rows[best]["position"])
    x0, x1, x2 = (rows[best + d]["position"] for d in (-1, 0, 1))
    y0, y1, y2 = (rows[best + d][METRIC] for d in (-1, 0, 1))
    denom = (y0 - 2 * y1 + y2)
    if denom >= 0:  # not concave; sampled best is the honest answer
        return float(x1)
    return float(x1 + 0.5 * (y0 - y2) / denom * ((x2 - x0) / 2))


async def main():
    parser = argparse.ArgumentParser(description="Focus sweep against the live rig")
    parser.add_argument("--center", type=int, help="sweep midpoint (focus units)")
    parser.add_argument("--af", action="store_true",
                        help="run autofocus_once first and centre the sweep there")
    parser.add_argument("--span", type=int, default=20,
                        help="sweep +/- this far around the centre (default 20)")
    parser.add_argument("--steps", type=int, default=15,
                        help="number of positions in the sweep (default 15)")
    parser.add_argument("--positions",
                        help="comma-separated explicit positions (overrides center/span/steps)")
    parser.add_argument("--fetch", choices=("local", "scp"), default="local",
                        help="how to read captured RAWs (default local)")
    parser.add_argument("--remote", help="user@host for --fetch scp")
    parser.add_argument("--no-home", action="store_true",
                        help="skip home_focus at sweep start")
    parser.add_argument("--per-frame-home", action="store_true",
                        help="re-home before every position and approach it in a "
                             "single batch - matches how focus_on_connect will "
                             "approach the stored setpoint. Use this when the "
                             "emulated scale looks path-dependent (same count "
                             "scoring differently across runs). Slower: one full "
                             "homing travel per frame.")
    parser.add_argument("--settle-s", type=float, default=1.0,
                        help="pause after each capture, for strobe recycle (default 1.0)")
    parser.add_argument("--window-frac", type=float, default=0.25,
                        help="sharpness window as fraction of frame (default 0.25)")
    parser.add_argument("--cx", type=float, default=0.5,
                        help="window centre x, fraction of width (default 0.5)")
    parser.add_argument("--cy", type=float, default=0.5,
                        help="window centre y, fraction of height (default 0.5)")
    parser.add_argument("--apply", action="store_true",
                        help="drive focus to the winning position when done")
    parser.add_argument("--repeat", type=int, metavar="N",
                        help="repeatability test instead of a sweep: drive to "
                             "--center N times (homing per frame with "
                             "--per-frame-home), score each frame, report the "
                             "spread. Pick a --center on a steep part of the "
                             "sweep curve so positional slop shows in the score.")
    parser.add_argument("--out", default="focus_sweep.csv", help="results CSV path")
    args = parser.parse_args()

    async with await connect() as machine:
        sony = Camera.from_robot(machine, "sony")

        status = await sony.do_command({"command": "get_status"})
        print(f"camera: {status.get('model')}  lens: {status.get('lens')}  "
              f"connected: {status.get('connected')}")
        settings = await sony.do_command({"command": "get_settings"})
        print(f"settings: {settings}")

        if not args.no_home:
            homed = await sony.do_command({"command": "home_focus"})
            print(f"home_focus: {homed}")

        if args.repeat:
            if args.center is None:
                raise ValueError("--repeat needs --center (a slope position)")
            positions = [int(args.center)] * args.repeat
            print(f"repeatability test: {args.repeat}x at position {args.center}")
        else:
            positions = await sweep_positions(args, sony)
            print(f"sweep positions: {positions}")

        if not args.per_frame_home:
            # Approach every target from the near side so focus-by-wire backlash
            # can't smear the scale across the sweep.
            pre = max(0, positions[0] - max(2, (positions[-1] - positions[0]) // 4))
            await sony.do_command({"command": "set_focus_position", "position": pre})

        rows = []
        workdir = tempfile.mkdtemp(prefix="focus_sweep_")
        print(f"\n{'target':>7} {'achieved':>9} {'laplacian_var':>14} {'tenengrad':>11}  file")
        last_achieved = None
        for target in positions:
            if args.per_frame_home:
                await sony.do_command({"command": "home_focus"})
            setr = await sony.do_command(
                {"command": "set_focus_position", "position": target})
            achieved = int(setr.get("position", target))
            if achieved == last_achieved and not args.repeat:
                print(f"{target:>7} clamped to {achieved} (already sampled); skipping")
                continue
            last_achieved = achieved
            if not setr.get("ok", True):
                print(f"{target:>7} set_focus_position reported ok=false "
                      f"(achieved {achieved}); scoring anyway")

            for attempt in (1, 2):
                try:
                    cap = (await sony.do_command({"capture": {}}))["capture"]
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise
                    print(f"{target:>7} capture failed ({exc}); retrying once")
                    await asyncio.sleep(2.0)

            local = await asyncio.to_thread(
                fetch, cap.get("saved_to") or cap.get("path"), args, workdir)
            score = await asyncio.to_thread(
                sharpness, local, args.window_frac, args.cx, args.cy)
            rows.append({"position": achieved, "target": target,
                         "file": cap.get("name"), **score})
            print(f"{target:>7} {achieved:>9} {score['laplacian_var']:>14.5f} "
                  f"{score['tenengrad']:>11.5f}  {cap.get('name')}")
            if args.fetch == "scp" and local.startswith(workdir):
                os.remove(local)
            time.sleep(args.settle_s)

        if args.repeat:
            scores = [r[METRIC] for r in rows]
            mean = sum(scores) / len(scores)
            spread_pct = 100.0 * (max(scores) - min(scores)) / mean
            print(f"\n{METRIC} over {len(scores)} frames at position {args.center}: "
                  f"mean {mean:.5f}, spread {spread_pct:.1f}% of mean")
            print("scorer noise floor is ~1.5-2% (same-focus frames); a spread "
                  "near that = repeatable at this interval, well above it = "
                  "nudges are being dropped")
            return

        peak = refine_peak(rows)
        best = max(rows, key=lambda r: r[METRIC])
        print(f"\nsharpest sampled position: {best['position']} "
              f"({METRIC} {best[METRIC]:.5f}, file {best['file']})")
        print(f"refined peak estimate:     {peak:.1f}")
        print("store the *sampled* integer position for this station; record the "
              "zoom position alongside it")

        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["target", "position", "file",
                                "laplacian_var", "tenengrad"])
            writer.writeheader()
            writer.writerows(
                {k: r[k] for k in writer.fieldnames} for r in rows)
        print(f"results written to {args.out}")

        if args.apply:
            final = await sony.do_command(
                {"command": "set_focus_position", "position": int(round(peak))})
            print(f"applied: {final}")


if __name__ == '__main__':
    asyncio.run(main())
