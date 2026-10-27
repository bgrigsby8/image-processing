#!/usr/bin/env python3
"""
Find the sharpest focus position for a fixed shooting distance: step the
still camera's focus across a range, shoot the checkerboard at each step and
rank the frames by slanted-edge MTF50 (``sharpness.py``).

    venv/bin/python scripts/focus_sweep.py \\
        --address <machine>.viam.cloud --pattern 11x8 \\
        --start 45000 --stop 65535 --coarse-step 2000 \\
        --compare 59142 --out-dir focus/2026-10-27-outfits

Set-up: the board flat at the product plane (where the outfit stands), whole
board in frame near where the product's centre would be, turned ~5 degrees on
its mount so every square edge is slanted. Production zoom, aperture, strobes,
shutter and ISO. Nothing else may drive the camera during the run - stop the
webapp's sweeps first.

Phases:

1. **coarse** - ``--start`` to ``--stop`` every ``--coarse-step``.
2. **fine** - ``--fine-step`` steps over two coarse steps either side of the
   coarse peak.
3. **repeat** - ``--repeat`` returns to the best position from each side.
   This is where the movie-mode -> stills-mode switch shows up: if the
   read-back agrees but MTF50 scatters, focus is not surviving the switch.
4. **compare** - one frame at each ``--compare`` position (the current
   production value, say), so the report says how much it is losing.

Every move within a phase is made in the same direction (toward far): the
first target of a phase is approached from ``--approach-margin`` below it.
Focus is put back where it started at the end (``--leave-at-best`` to stay).

Analysis needs the RAWs: run this on the rig host (the camera writes to its
own disk), or pass ``--copy-to``. When the files are not reachable the coarse
captures are still taken and recorded; copy them next to the manifest and run
``--analyze <out-dir>``, which re-measures every recorded frame offline.

Focus positions are the camera module's units: with ``focus_method: movie``,
follow-focus 0 (near stop) .. 65535 (far stop).
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from models.sharpness import FocusPeak, analyze_raw, fit_focus_peak

TOOL_VERSION = "0.1.0"
FOLLOW_FOCUS_MAX = 0xFFFF
MANIFEST = "focus_sweep.json"
_NOT_FOUND = "checkerboard not found"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_pattern(text: str) -> Tuple[int, int]:
    """``"11x8"`` -> (11, 8) inner corners per row, per column."""
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--pattern must look like 11x8, got {text!r}") from exc
    if cols < 3 or rows < 3:
        raise argparse.ArgumentTypeError(f"--pattern needs at least 3x3 inner corners, got {text!r}")
    return cols, rows


# ---------------------------------------------------------------------------
# Planning (hardware-free)
# ---------------------------------------------------------------------------

def _clamp(position: float) -> int:
    return int(min(max(round(position), 0), FOLLOW_FOCUS_MAX))


def coarse_positions(start: int, stop: int, step: int) -> List[int]:
    """``start`` .. ``stop`` every ``step``, always including ``stop``."""
    if step <= 0:
        raise ValueError("--coarse-step must be positive")
    lo, hi = _clamp(min(start, stop)), _clamp(max(start, stop))
    out = list(range(lo, hi + 1, step))
    if out[-1] != hi:
        out.append(hi)
    return out


def fine_positions(centre: float, half_span: int, step: int) -> List[int]:
    """``centre`` +/- ``half_span`` every ``step``, centre included."""
    if step <= 0:
        raise ValueError("--fine-step must be positive")
    n = max(1, half_span // step)
    return sorted({_clamp(centre + k * step) for k in range(-n, n + 1)})


def peak_of(records: Sequence[Dict[str, Any]]) -> Optional[FocusPeak]:
    """Focus peak over records with a measured MTF50, by read-back position
    (what the lens reported) falling back to the commanded target."""
    pos, val = [], []
    for r in records:
        mtf = (r.get("sharpness") or {}).get("mtf50")
        where = r.get("readback")
        if where is None:
            where = r.get("target")
        if mtf is not None and where is not None:
            pos.append(float(where))
            val.append(float(mtf))
    return fit_focus_peak(pos, val)


def format_table(records: Sequence[Dict[str, Any]], height: Optional[int] = None) -> str:
    """One line per frame with an MTF50 bar, scaled to the sweep's best."""
    vals = [(r.get("sharpness") or {}).get("mtf50") for r in records]
    best = max((v for v in vals if v is not None), default=None)
    lines = [f"{'phase':<8}{'target':>7}{'read':>7}  {'MTF50':>6}  {'LW/PH':>6}  edges"]
    for r, v in zip(records, vals):
        s = r.get("sharpness") or {}
        read = r.get("readback")
        bar = "#" * int(round(40 * v / best)) if (v is not None and best) else ""
        lw = s.get("mtf50_lw_ph")
        lines.append(
            f"{r.get('phase', ''):<8}{r.get('target', ''):>7}{'' if read is None else read:>7}  "
            f"{'' if v is None else f'{v:.3f}':>6}  {'' if lw is None else f'{lw:.0f}':>6}  "
            f"{s.get('edges_used', 0):>3}/{s.get('edges_total', 0):<3} {bar}"
            + (f"  ({s['error']})" if s.get("error") and v is None else "")
        )
    return "\n".join(lines)


def format_peak(peak: Optional[FocusPeak], label: str) -> str:
    if peak is None:
        return f"{label}: not enough measured frames for a peak"
    text = (f"{label}: best focus {peak.best_position:.0f} "
            f"(MTF50 {peak.peak_mtf50:.3f} cy/px, {peak.fit})")
    if peak.usable_range:
        lo, hi = peak.usable_range
        text += (f"; >= {peak.usable_fraction:.0%} of peak from {lo:.0f} to {hi:.0f} "
                 f"(width {hi - lo:.0f})")
    if peak.at_edge_of_sweep:
        text += "\n  WARNING: best frame is at the end of the sweep - widen --start/--stop"
    return text


def repeat_summary(records: Sequence[Dict[str, Any]], best_mtf: Optional[float]) -> Dict[str, Any]:
    """Scatter of the repeat-phase frames, per approach side."""
    out: Dict[str, Any] = {}
    for side in ("below", "above"):
        rs = [r for r in records if r.get("phase") == "repeat" and r.get("approach") == side]
        mtf = [r["sharpness"]["mtf50"] for r in rs if (r.get("sharpness") or {}).get("mtf50") is not None]
        read = [r["readback"] for r in rs if r.get("readback") is not None]
        if not rs:
            continue
        out[side] = {
            "frames": len(rs),
            "mtf50_mean": float(np.mean(mtf)) if mtf else None,
            "mtf50_min": float(np.min(mtf)) if mtf else None,
            "mtf50_std": float(np.std(mtf)) if len(mtf) > 1 else None,
            "readback_mean": float(np.mean(read)) if read else None,
            "readback_std": float(np.std(read)) if len(read) > 1 else None,
        }
        if best_mtf and mtf:
            out[side]["worst_vs_peak"] = float(np.min(mtf) / best_mtf)
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

class Sweeper:
    def __init__(
        self,
        args: argparse.Namespace,
        manifest: Optional[Dict[str, Any]] = None,
        *,
        cam: Any = None,
        log: Callable[[str], None] = print,
    ):
        """``cam`` is anything with the camera's ``do_command`` - passed in
        when running inside the module, connected by ``run`` from the CLI."""
        self.args = args
        self.log = log
        self.corners_hint: Optional[np.ndarray] = None
        self.can_analyze = True
        self.phase = "starting"
        self.report_text: Optional[str] = None
        self.robot = None
        self.cam = cam
        if manifest is not None:
            # Re-analysing an earlier run: keep its records and settings.
            self.manifest = manifest
            self.records: List[Dict[str, Any]] = manifest.setdefault("records", [])
            return
        self.records = []
        self.manifest = {
            "tool": f"image-processing focus_sweep {TOOL_VERSION}",
            "created_at": _now(),
            "address": args.address,
            "camera": args.camera,
            "pattern": list(args.pattern),
            "settings": {k: getattr(args, k) for k in (
                "start", "stop", "coarse_step", "fine_step", "repeat",
                "approach_margin", "tolerance", "compare")},
            "records": self.records,
        }

    # -- connection -------------------------------------------------------

    async def connect(self) -> None:
        from viam.components.camera import Camera
        from viam.robot.client import RobotClient

        key = self.args.api_key or os.environ.get("VIAM_API_KEY")
        key_id = self.args.api_key_id or os.environ.get("VIAM_API_KEY_ID")
        if not key or not key_id:
            raise SystemExit("need --api-key/--api-key-id or VIAM_API_KEY/VIAM_API_KEY_ID")
        opts = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
        self.robot = await RobotClient.at_address(self.args.address, opts)
        self.cam = Camera.from_robot(self.robot, self.args.camera)

    async def close(self) -> None:
        if self.robot is not None:
            await self.robot.close()

    # -- camera helpers -----------------------------------------------------

    async def get_focus(self) -> Dict[str, Any]:
        return dict(await self.cam.do_command({"get_focus_position": {}}, timeout=self.args.focus_timeout_s))

    async def set_focus(self, position: int) -> Dict[str, Any]:
        cmd: Dict[str, Any] = {"position": int(position)}
        if self.args.tolerance is not None:
            cmd["tolerance"] = int(self.args.tolerance)
        return dict(await self.cam.do_command({"set_focus_position": cmd}, timeout=self.args.focus_timeout_s))

    async def capture(self) -> Dict[str, Any]:
        resp = await self.cam.do_command({"capture": {}}, timeout=self.args.capture_timeout_s)
        cap = resp.get("capture", resp) if isinstance(resp, dict) else {}
        keep = ("path", "saved_to", "name", "size", "focus_position", "capture_count",
                "duration_s", "settings")
        return {k: cap.get(k) for k in keep if k in cap}

    # -- one frame ------------------------------------------------------------

    async def shoot(self, phase: str, target: int, *, approach: Optional[str] = None,
                    pre_position: Optional[int] = None) -> Dict[str, Any]:
        a = self.args
        rec: Dict[str, Any] = {"phase": phase, "target": int(target), "time": _now()}
        if approach:
            rec["approach"] = approach
        if pre_position is not None:
            rec["pre_position"] = int(pre_position)
            await self.set_focus(pre_position)
        t0 = time.perf_counter()
        focus = await self.set_focus(target)
        rec["set_focus"] = focus
        rec["readback"] = focus.get("position")
        if focus.get("ok") is False:
            self.log(f"    focus missed {target}: read back {focus.get('position')} "
                  f"({focus.get('stop_reason', 'no reason given')}); shooting anyway")
        rec["capture"] = await self.capture()
        path = rec["capture"].get("saved_to") or rec["capture"].get("path")
        if a.copy_to and path and os.path.exists(path):
            dst = os.path.join(a.copy_to, os.path.basename(path))
            shutil.copy2(path, dst)
            rec["copied_to"] = dst
        rec["shoot_s"] = round(time.perf_counter() - t0, 1)
        self.records.append(rec)
        await self.analyze(rec)
        s = rec.get("sharpness") or {}
        mtf = s.get("mtf50")
        self.log(f"  [{phase:<7}] target {target:>5}  read {rec['readback']!s:>5}  "
              f"-> {os.path.basename(path) if path else '?'}  "
              + (f"MTF50 {mtf:.3f} cy/px ({s.get('edges_used')}/{s.get('edges_total')} edges)"
                 if mtf is not None else f"MTF50 -  {s.get('error', 'not analysed')}")
              + f"  [{rec['shoot_s']}s + {s.get('seconds', 0)}s]")
        self.write_manifest()
        return rec

    def local_path(self, rec: Dict[str, Any]) -> Optional[str]:
        """Where this frame's RAW can be read from here, if anywhere."""
        cap = rec.get("capture") or {}
        candidates = [rec.get("copied_to"), cap.get("saved_to"), cap.get("path")]
        name = cap.get("name") or os.path.basename(cap.get("path") or "")
        if name:
            candidates.append(os.path.join(self.args.out_dir, name))
        return next((p for p in candidates if p and os.path.exists(p)), None)

    async def analyze(self, rec: Dict[str, Any], *, force: bool = False) -> None:
        path = self.local_path(rec) if (self.can_analyze or force) else None
        if path is None:
            if self.can_analyze:
                self.log("    the RAW is not readable from here; frames are recorded but not "
                      "measured (run on the rig host, or use --copy-to / --analyze)")
            self.can_analyze = False
            rec["sharpness"] = {"error": "RAW not reachable from this host"}
            return
        result, corners = await asyncio.to_thread(
            analyze_raw, path, self.args.pattern, corners_hint=self.corners_hint
        )
        if corners is not None and not result.corners_from_hint:
            self.corners_hint = corners
        rec["sharpness"] = result.summary()

    async def retry_undetected(self) -> None:
        """Frames shot before any board was found get measured again at the
        board's position from a later, sharper frame."""
        if self.corners_hint is None:
            return
        for rec in self.records:
            err = (rec.get("sharpness") or {}).get("error") or ""
            if err.startswith(_NOT_FOUND):
                await self.analyze(rec, force=True)

    # -- phases ---------------------------------------------------------------

    async def sweep(self, phase: str, positions: Sequence[int]) -> None:
        self.phase = phase
        self.log(f"\n-- {phase}: {len(positions)} frames, {positions[0]} .. {positions[-1]}")
        pre = _clamp(positions[0] - self.args.approach_margin)
        for i, target in enumerate(positions):
            await self.shoot(phase, target, approach="below", pre_position=pre if i == 0 else None)
        await self.retry_undetected()

    async def run(self) -> int:
        await self.connect()
        try:
            return await self.execute()
        finally:
            await self.close()

    async def execute(self) -> int:
        a = self.args
        status = dict(await self.cam.do_command({"get_status": {}}, timeout=10))
        status = status.get("get_status", status)
        self.manifest["camera_status"] = {k: status.get(k) for k in ("model", "lens", "connected", "capture_dir")}
        start_focus = await self.get_focus()
        self.manifest["start_focus"] = start_focus
        self.log(f"camera: {status.get('model')} connected={status.get('connected')}; "
              f"focus now {start_focus.get('position')} ({start_focus.get('units')})")
        if start_focus.get("units") not in (None, "follow_focus"):
            self.log(f"WARNING: focus units are {start_focus.get('units')!r}, not follow_focus; "
                  f"positions below are in those units")

        coarse = coarse_positions(a.start, a.stop, a.coarse_step)
        self.log(f"plan: coarse {len(coarse)} frames {coarse[0]}..{coarse[-1]} step {a.coarse_step}; "
              f"fine +/-{2 * a.coarse_step} step {a.fine_step}; repeat {a.repeat}x each side; "
              f"compare {a.compare or 'none'}")
        if a.dry_run:
            self.log("[DRY RUN] nothing moved, nothing shot")
            return 0
        if a.copy_to:
            os.makedirs(a.copy_to, exist_ok=True)

        try:
            await self.sweep("coarse", coarse)
            peak = peak_of([r for r in self.records if r["phase"] == "coarse"])
            self.log("\n" + format_peak(peak, "coarse"))
            if peak is None:
                self.log("no measurable frames - analyse offline (--analyze) and re-run "
                      "with a narrower --start/--stop")
                return 1 if self.can_analyze else 0

            if a.fine_step and a.fine_step < a.coarse_step:
                await self.sweep("fine", fine_positions(peak.best_position, 2 * a.coarse_step, a.fine_step))
            peak = peak_of([r for r in self.records if r["phase"] in ("coarse", "fine")])
            best = _clamp(peak.best_position)

            if a.repeat:
                self.phase = "repeat"
                self.log(f"\n-- repeat: {a.repeat}x to {best} from each side")
                for i in range(a.repeat):
                    await self.shoot("repeat", best, approach="below",
                                     pre_position=_clamp(best - a.approach_margin))
                    await self.shoot("repeat", best, approach="above",
                                     pre_position=_clamp(best + a.approach_margin))

            self.phase = "compare"
            for pos in a.compare or []:
                self.log(f"\n-- compare: {pos}")
                await self.shoot("compare", pos, approach="below",
                                 pre_position=_clamp(pos - a.approach_margin))
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.log("\ninterrupted")
            self.manifest["interrupted"] = True
        finally:
            self.phase = "restoring"
            back_to = None if a.leave_at_best else start_focus.get("position")
            if back_to is not None:
                try:
                    self.log(f"\nrestoring focus to {back_to}")
                    await self.set_focus(int(back_to))
                except Exception as exc:  # noqa: BLE001
                    self.log(f"could not restore focus: {exc}")
            self.finish()
        return 0

    # -- report -----------------------------------------------------------------

    def finish(self) -> None:
        self.report_text = report(self.records, self.manifest)
        self.log("\n" + self.report_text)
        self.write_manifest()
        self.log(f"\nmanifest: {os.path.join(self.args.out_dir, MANIFEST)}")

    def write_manifest(self) -> None:
        os.makedirs(self.args.out_dir, exist_ok=True)
        path = os.path.join(self.args.out_dir, MANIFEST)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2, default=_jsonable)
        os.replace(tmp, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def report(records: Sequence[Dict[str, Any]], manifest: Dict[str, Any]) -> str:
    """The end-of-run summary; also stores the numbers in ``manifest``."""
    lines = [format_table(records)]
    sweep = [r for r in records if r.get("phase") in ("coarse", "fine")]
    peak = peak_of(sweep)
    lines.append("")
    lines.append(format_peak(peak, "sweep"))
    manifest["peak"] = None if peak is None else {
        "best_position": round(peak.best_position),
        "peak_mtf50": peak.peak_mtf50,
        "fit": peak.fit,
        "usable_range": None if peak.usable_range is None else [round(v) for v in peak.usable_range],
        "usable_fraction": peak.usable_fraction,
        "at_edge_of_sweep": peak.at_edge_of_sweep,
    }
    if peak is not None:
        reps = repeat_summary(records, peak.peak_mtf50)
        manifest["repeat"] = reps
        for side, s in reps.items():
            if s.get("mtf50_mean") is None:
                continue
            lines.append(
                f"repeat from {side}: MTF50 {s['mtf50_mean']:.3f} mean, worst "
                f"{s.get('worst_vs_peak', 0):.0%} of peak; read-back "
                f"{s['readback_mean']:.0f}" + (f" +/- {s['readback_std']:.0f}" if s.get("readback_std") is not None else "")
            )
        for r in records:
            if r.get("phase") != "compare":
                continue
            mtf = (r.get("sharpness") or {}).get("mtf50")
            if mtf is not None:
                lines.append(f"compare {r['target']}: MTF50 {mtf:.3f} = "
                             f"{mtf / peak.peak_mtf50:.0%} of peak")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Offline re-analysis
# ---------------------------------------------------------------------------

def analyze_offline(args: argparse.Namespace) -> int:
    """Re-measure every frame in an existing manifest (RAWs next to it, or at
    their recorded paths) and rewrite the report."""
    path = os.path.join(args.analyze, MANIFEST)
    with open(path) as f:
        manifest = json.load(f)
    args.out_dir = args.analyze
    args.pattern = tuple(manifest.get("pattern") or args.pattern or ())
    if len(args.pattern) != 2:
        print("the manifest has no pattern; pass --pattern", file=sys.stderr)
        return 2
    sweeper = Sweeper(args, manifest)

    async def _go() -> None:
        # The retry pass covers frames measured before any board was found.
        for rec in sweeper.records:
            await sweeper.analyze(rec, force=True)
            s = rec.get("sharpness") or {}
            print(f"  {rec.get('phase'):<7} {rec.get('target'):>5}: "
                  + (f"MTF50 {s['mtf50']:.3f}" if s.get("mtf50") is not None else s.get("error", "?")))
        await sweeper.retry_undetected()

    asyncio.run(_go())
    sweeper.manifest["analyzed_at"] = _now()
    sweeper.finish()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    con = p.add_argument_group("machine")
    con.add_argument("--address", default=None, help="machine address (not needed with --analyze)")
    con.add_argument("--api-key", default=None, help="or VIAM_API_KEY")
    con.add_argument("--api-key-id", default=None, help="or VIAM_API_KEY_ID")
    con.add_argument("--camera", default="sony",
                     help="still camera with set_focus_position / capture DoCommands (default: sony)")

    plan = p.add_argument_group("plan")
    plan.add_argument("--pattern", type=parse_pattern, default=None, help="inner corners COLSxROWS, e.g. 11x8")
    plan.add_argument("--start", type=int, default=45000, help="first coarse position (default: 45000)")
    plan.add_argument("--stop", type=int, default=FOLLOW_FOCUS_MAX, help="last coarse position (default: 65535)")
    plan.add_argument("--coarse-step", type=int, default=2000, help="default: 2000")
    plan.add_argument("--fine-step", type=int, default=400,
                      help="fine pass step over +/-2 coarse steps around the coarse peak; "
                           "0 skips the fine pass (default: 400)")
    plan.add_argument("--repeat", type=int, default=3,
                      help="returns to the best position from each side (default: 3)")
    plan.add_argument("--compare", type=int, nargs="*", default=[],
                      help="also shoot these positions (e.g. the production value)")
    plan.add_argument("--approach-margin", type=int, default=3000,
                      help="how far below (or above) a target the approach starts (default: 3000)")
    plan.add_argument("--tolerance", type=int, default=None,
                      help="set_focus_position tolerance; default: the camera's movie_focus_tolerance")

    run = p.add_argument_group("run")
    run.add_argument("--dry-run", action="store_true", help="connect, read focus, print the plan; move nothing")
    run.add_argument("--leave-at-best", action="store_true", help="do not restore the starting focus")
    run.add_argument("--focus-timeout-s", type=float, default=120.0)
    run.add_argument("--capture-timeout-s", type=float, default=60.0)
    run.add_argument("--analyze", metavar="DIR", default=None,
                     help="re-measure the frames recorded in DIR/focus_sweep.json; no machine needed")

    out = p.add_argument_group("output")
    out.add_argument("--out-dir", default=None, help="where focus_sweep.json goes (default: focus/<date>)")
    out.add_argument("--copy-to", default=None,
                     help="copy each RAW here right after capture (on the rig host); protects "
                          "the set from the camera module's retention")
    return p


def coerce_pattern(value: Any) -> Tuple[int, int]:
    """A board pattern given as ``"11x8"`` or ``[11, 8]`` (inner corners)."""
    if value is None:
        raise ValueError("need `pattern` (inner corners, e.g. \"11x8\" or [11, 8])")
    if isinstance(value, str):
        try:
            return parse_pattern(value)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(str(exc)) from exc
    pattern = tuple(int(v) for v in value)
    if len(pattern) != 2 or min(pattern) < 3:
        raise ValueError(f"`pattern` must be two inner-corner counts >= 3, got {value!r}")
    return pattern


# DoCommand options a sweep accepts (the CLI's plan/run settings, minus the
# ones that only make sense from outside the machine).
COMMAND_OPTIONS = (
    "pattern", "start", "stop", "coarse_step", "fine_step", "repeat", "compare",
    "approach_margin", "tolerance", "leave_at_best", "focus_timeout_s",
    "capture_timeout_s", "out_dir", "copy_to",
)


def options_from_command(opts: Mapping[str, Any]) -> argparse.Namespace:
    """A ``focus_sweep`` DoCommand's options as the CLI's settings: CLI
    defaults, overridden by what the command sets. ``pattern`` is required,
    as ``"11x8"`` or ``[11, 8]``. Unknown keys are refused so a typo can't
    silently run the default plan."""
    unknown = sorted(set(opts) - set(COMMAND_OPTIONS))
    if unknown:
        raise ValueError(f"unknown focus_sweep option(s) {unknown}; valid: {list(COMMAND_OPTIONS)}")
    args = build_parser().parse_args([])
    for key in COMMAND_OPTIONS:
        if opts.get(key) is not None:
            setattr(args, key, opts[key])
    args.pattern = coerce_pattern(args.pattern)
    for key in ("start", "stop", "coarse_step", "fine_step", "repeat", "approach_margin"):
        setattr(args, key, int(getattr(args, key)))
    if args.tolerance is not None:
        args.tolerance = int(args.tolerance)
    args.compare = [int(v) for v in (args.compare or [])]
    args.leave_at_best = bool(args.leave_at_best)
    if args.coarse_step <= 0:
        raise ValueError("`coarse_step` must be positive")
    args.address = None
    args.dry_run = False
    args.analyze = None
    return args


def frame_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One compact row per frame for a status response."""
    rows = []
    for r in records:
        s = r.get("sharpness") or {}
        cap = r.get("capture") or {}
        rows.append({
            "phase": r.get("phase"),
            "target": r.get("target"),
            "readback": r.get("readback"),
            "approach": r.get("approach"),
            "mtf50": s.get("mtf50"),
            "mtf50_lw_ph": s.get("mtf50_lw_ph"),
            "edges_used": s.get("edges_used"),
            "edges_total": s.get("edges_total"),
            "error": s.get("error"),
            "path": cap.get("saved_to") or cap.get("path"),
        })
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analyze:
        return analyze_offline(args)
    if not args.address or not args.pattern:
        print("--address and --pattern are required (except with --analyze)", file=sys.stderr)
        return 2
    if args.out_dir is None:
        args.out_dir = os.path.join("focus", datetime.now().strftime("%Y-%m-%d"))
    try:
        return asyncio.run(Sweeper(args).run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
