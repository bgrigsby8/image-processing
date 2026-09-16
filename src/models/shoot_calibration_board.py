#!/usr/bin/env python3
"""
Drive the arm-mounted camera through a set of views of a *stationary*
checkerboard and fire the still camera at each one - the capture half of the
lens-distortion calibration (``calibrate_distortion.py`` is the fitting half).

    venv/bin/python scripts/shoot_calibration_board.py \\
        --address <machine>.viam.cloud --pattern 10x7 \\
        --count 40 --holdouts 5 --out-dir calib/2026-09-16-16mm-f8 --dry-run

Only one thing is recorded by hand: the *nominal* pose, with the board centred
and fully in frame. Everything else is generated from it.

Why rotations, not translations. At 16 mm the frame is ~2.2 m wide at 1 m, so
putting a 600 mm board into a frame corner by translating the camera needs
~0.8 m of lateral travel the arm does not have. Panning / tilting the camera a
few tens of degrees about its wrist does the same thing in a few degrees of
joint motion, and it *is* the board tilt the calibration wants. (Pure lateral
translation - all board planes parallel - is the one degenerate case for
Zhang's method, so nothing is lost by skipping it.) Distance is varied with a
translation along the camera's viewing axis via ``move_to_position``.

Each shot: move (joint space, small wrist deltas from the nominal), wait for
the arm to settle, grab a live-view frame and check the *whole* board is in
it, then fire ``capture``. A pose where the board leaves the frame is pulled
back toward the nominal and retried, so the pan/tilt ranges do not need to be
tuned to the lens. ``--dry-run`` does everything except fire the shutter and
prints the 4x3 coverage grid the calibration CLI will grade, so the plan can be
rehearsed at low arm speed first.

Poses are commanded with ``move_to_joint_positions``, which bypasses the motion
service's obstacle checking: the deltas are wrist-only and small, the nominal
pose is one you drove to by hand, and the dry run is the rehearsal. Joint
limits mirroring the rig's motion-service ``input_range_override`` are checked
before any move.
"""

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

TOOL_VERSION = "0.1.0"

# Frame coverage grid graded by calibrate_distortion (COLS x ROWS).
COVERAGE_GRID = (4, 3)

# Default joint limits (degrees), 1-indexed joints 1..6. These mirror the
# Nines rig's motion-service `input_range_override` for the xArm850 so a
# planned pose can never ask for more than the planner itself would allow.
DEFAULT_JOINT_LIMITS_DEG: List[Tuple[float, float]] = [
    (0.0, 179.9),
    (-179.9, 45.0),
    (-154.1, 0.0),
    (-74.5, 45.0),
    (-97.4, 179.9),
    (-10.0, 154.7),
]

# When the live view shows the board leaving the frame, pull the pan/tilt/roll
# deltas toward the nominal by this factor and try again, up to this many times.
SHRINK_FACTOR = 0.7
MAX_SHRINK_ATTEMPTS = 3

# get_images attempts per frame before the check is treated as failed.
LIVEVIEW_RETRIES = 4

# Roll levels cycled across the main set (degrees). Roll adds nothing to the
# coverage but decorrelates the board's edges from the pixel grid.
ROLL_CYCLE = (0.0, 1.0, -1.0)   # multiplied by --roll-max


@dataclass
class PlannedShot:
    """One view of the board, as wrist deltas from the nominal pose (degrees)
    plus a translation along the viewing axis (mm)."""
    index: int
    distance_mm: float
    pan_deg: float
    tilt_deg: float
    roll_deg: float
    holdout: bool = False


@dataclass
class ShotRecord:
    """What actually happened for one planned shot - the manifest row."""
    index: int
    holdout: bool
    distance_mm: float
    planned: Dict[str, float]
    commanded: Dict[str, float] = field(default_factory=dict)
    joints_commanded_deg: List[float] = field(default_factory=list)
    joints_actual_deg: List[float] = field(default_factory=list)
    attempts: int = 0
    board_found: Optional[bool] = None
    board_bbox_norm: Optional[List[float]] = None   # [x0, y0, x1, y1] in [0,1]
    board_span_norm: Optional[float] = None          # bbox diagonal / frame diagonal
    coverage_cells: List[List[int]] = field(default_factory=list)
    capture: Optional[Dict[str, Any]] = None
    copied_to: Optional[str] = None
    skipped: Optional[str] = None


# --------------------------------------------------------------------------
# Planning (pure functions - tested without hardware)
# --------------------------------------------------------------------------

def parse_pattern(text: str) -> Tuple[int, int]:
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"pattern must look like 10x7, got {text!r}")
    if cols < 2 or rows < 2:
        raise argparse.ArgumentTypeError("pattern needs at least 2x2 inner corners")
    return cols, rows


def _levels(maximum: float, n: int) -> List[float]:
    """``n`` evenly spaced values in [-maximum, maximum]."""
    if n <= 1:
        return [0.0]
    return [float(v) for v in np.linspace(-maximum, maximum, n)]


def _snake(rows: Sequence[Sequence[Any]]) -> List[Any]:
    """Row-major with every other row reversed - minimal wrist travel."""
    out: List[Any] = []
    for i, row in enumerate(rows):
        out.extend(reversed(row) if i % 2 else row)
    return out


def _holdout_offsets(n: int, pan_max: float, tilt_max: float) -> List[Tuple[float, float]]:
    """Mild, low-tilt views for the holdout set: centre first, then the four
    half-range cardinal positions, then quarter-range diagonals."""
    base = [
        (0.0, 0.0),
        (pan_max / 2, 0.0), (-pan_max / 2, 0.0),
        (0.0, tilt_max / 2), (0.0, -tilt_max / 2),
        (pan_max / 4, tilt_max / 4), (-pan_max / 4, -tilt_max / 4),
        (pan_max / 4, -tilt_max / 4), (-pan_max / 4, tilt_max / 4),
    ]
    out: List[Tuple[float, float]] = []
    k = 0
    while len(out) < n:
        p, t = base[k % len(base)]
        # Past the base list, jitter a little so repeats aren't identical views.
        rep = k // len(base)
        out.append((p + rep * pan_max / 8, t - rep * tilt_max / 8))
        k += 1
    return out


def plan_shots(
    *,
    count: int,
    holdouts: int,
    pan_max_deg: float,
    tilt_max_deg: float,
    roll_max_deg: float,
    distances_mm: Sequence[float],
    pan_levels: int = 5,
    tilt_levels: int = 4,
    seed: int = 0,
) -> List[PlannedShot]:
    """
    ``count`` calibration views plus ``holdouts`` validation views, spread over
    ``distances_mm`` (each a translation along the viewing axis from the
    nominal; include 0 for the nominal distance).

    Per distance the main set is a ``pan_levels x tilt_levels`` grid in snake
    order with roll cycled through ``ROLL_CYCLE``. If the grid across all
    distances has fewer cells than ``count``, seeded random pan/tilt views top
    it up; if more, the grid is evenly subsampled. Shots are ordered by
    distance so the arm translates once per distance; each distance's holdouts
    come right after its main views.
    """
    if count < 0 or holdouts < 0:
        raise ValueError("count and holdouts must be non-negative")
    distances = list(distances_mm) or [0.0]
    rng = np.random.default_rng(seed)

    pans = _levels(pan_max_deg, pan_levels)
    tilts = _levels(tilt_max_deg, tilt_levels)
    grid = _snake([[(p, t) for p in pans] for t in tilts])
    per_dist_main = [list(grid) for _ in distances]

    total_grid = len(grid) * len(distances)
    if total_grid > count:
        # Evenly subsample the *flattened* grid so every distance keeps a
        # spread of pan/tilt, then regroup by distance.
        flat = [(di, pt) for di in range(len(distances)) for pt in grid]
        keep = np.linspace(0, len(flat) - 1, count).round().astype(int)
        per_dist_main = [[] for _ in distances]
        for k in sorted(set(int(i) for i in keep)):
            di, pt = flat[k]
            per_dist_main[di].append(pt)
    elif total_grid < count:
        extra = count - total_grid
        for k in range(extra):
            di = k % len(distances)
            per_dist_main[di].append((
                float(rng.uniform(-pan_max_deg, pan_max_deg)),
                float(rng.uniform(-tilt_max_deg, tilt_max_deg)),
            ))

    hold = _holdout_offsets(holdouts, pan_max_deg, tilt_max_deg)
    per_dist_hold: List[List[Tuple[float, float]]] = [[] for _ in distances]
    for k, pt in enumerate(hold):
        per_dist_hold[k % len(distances)].append(pt)

    shots: List[PlannedShot] = []
    roll_i = 0
    for di, d in enumerate(distances):
        for (p, t) in per_dist_main[di]:
            roll = ROLL_CYCLE[roll_i % len(ROLL_CYCLE)] * roll_max_deg
            roll_i += 1
            shots.append(PlannedShot(len(shots), float(d), round(p, 3), round(t, 3), round(roll, 3)))
        for (p, t) in per_dist_hold[di]:
            shots.append(PlannedShot(len(shots), float(d), round(p, 3), round(t, 3), 0.0, holdout=True))
    return shots


def joints_for(
    base_joints_deg: Sequence[float],
    pan_deg: float,
    tilt_deg: float,
    roll_deg: float,
    *,
    pan_joint: int,
    tilt_joint: int,
    roll_joint: int,
) -> List[float]:
    """Apply wrist deltas (degrees) to a base joint vector. Joints are 1-indexed."""
    joints = [float(v) for v in base_joints_deg]
    for j in (pan_joint, tilt_joint, roll_joint):
        if not 1 <= j <= len(joints):
            raise ValueError(f"joint {j} out of range for a {len(joints)}-joint arm")
    if len({pan_joint, tilt_joint, roll_joint}) != 3:
        raise ValueError("pan, tilt and roll joints must be distinct")
    joints[pan_joint - 1] += pan_deg
    joints[tilt_joint - 1] += tilt_deg
    joints[roll_joint - 1] += roll_deg
    return joints


def check_joint_limits(
    joints_deg: Sequence[float], limits_deg: Sequence[Tuple[float, float]]
) -> List[str]:
    """Human-readable violations; empty list = within limits."""
    problems = []
    for i, v in enumerate(joints_deg):
        if i >= len(limits_deg):
            break
        lo, hi = limits_deg[i]
        if v < lo or v > hi:
            problems.append(f"joint {i + 1} = {v:.1f} deg outside [{lo:.1f}, {hi:.1f}]")
    return problems


def parse_joint_limits(text: str) -> List[Tuple[float, float]]:
    """``lo:hi,lo:hi,...`` in degrees, one pair per joint."""
    out = []
    for pair in text.split(","):
        lo, hi = (float(v) for v in pair.split(":"))
        if lo > hi:
            raise argparse.ArgumentTypeError(f"joint limit {pair!r} has lo > hi")
        out.append((lo, hi))
    return out


AXES = ("tool-z", "+x", "-x", "+y", "-y", "+z", "-z")


def translate_pose(pose_xyz: Sequence[float], ov: Sequence[float], mm: float, axis: str) -> List[float]:
    """
    Translate a position by ``mm`` along ``axis``. ``tool-z`` is the tool's
    own z axis, which in Viam's orientation-vector convention *is* the unit
    vector ``(o_x, o_y, o_z)`` - no rotation maths needed. The others are
    world axes. Positive ``mm`` moves along the axis; for ``tool-z`` that is
    toward whatever the camera is looking at, i.e. *closer* to the board.
    """
    x, y, z = (float(v) for v in pose_xyz)
    if axis == "tool-z":
        ox, oy, oz = (float(v) for v in ov)
        n = math.sqrt(ox * ox + oy * oy + oz * oz)
        if n < 1e-9:
            raise ValueError("orientation vector has zero length")
        d = (ox / n, oy / n, oz / n)
    elif axis in AXES:
        sign = 1.0 if axis[0] == "+" else -1.0
        d = {"x": (sign, 0.0, 0.0), "y": (0.0, sign, 0.0), "z": (0.0, 0.0, sign)}[axis[1]]
    else:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
    return [x + mm * d[0], y + mm * d[1], z + mm * d[2]]


# --------------------------------------------------------------------------
# Live-view board check + coverage (pure functions)
# --------------------------------------------------------------------------

def find_board(gray8: np.ndarray, pattern: Tuple[int, int]) -> Optional[np.ndarray]:
    """Whole-board corner detection on a (small) live-view frame. Returns
    ``(N, 2)`` pixel coordinates or None. Fast flags - this is a yes/no check
    at preview size, not the calibration's sub-pixel detection."""
    # Live view sits at the strobe exposure with no strobe firing, so the
    # frame is very dark (median ~13/255 on the rig). Stretch it first.
    lo, hi = np.percentile(gray8, [1.0, 99.5])
    if hi - lo > 1:
        gray8 = np.clip((gray8.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    ok, corners = cv2.findChessboardCornersSB(gray8, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        ok, corners = cv2.findChessboardCorners(
            gray8, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
    if not ok or corners is None or len(corners) != pattern[0] * pattern[1]:
        return None
    return corners.reshape(-1, 2).astype(np.float32)


def decode_liveview(data: bytes) -> Optional[np.ndarray]:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


def board_bbox_norm(corners: np.ndarray, size_wh: Tuple[int, int]) -> List[float]:
    w, h = size_wh
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    return [float(x0 / w), float(y0 / h), float(x1 / w), float(y1 / h)]


def board_span_norm(corners: np.ndarray, size_wh: Tuple[int, int]) -> float:
    """Board bbox diagonal as a fraction of the frame diagonal - a scale
    proxy, so distance steps can be seen to do something in the dry run."""
    w, h = size_wh
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    return float(math.hypot(x1 - x0, y1 - y0) / math.hypot(w, h))


def coverage_cells(
    corners: np.ndarray, size_wh: Tuple[int, int], grid: Tuple[int, int] = COVERAGE_GRID
) -> List[List[int]]:
    """Which ``grid`` (COLS x ROWS) cells hold at least one corner - the same
    definition calibrate_distortion grades ("frames putting corners in each
    cell"). Returns ``[[col, row], ...]``."""
    w, h = size_wh
    cols, rows = grid
    cx = np.clip((corners[:, 0] / w * cols).astype(int), 0, cols - 1)
    cy = np.clip((corners[:, 1] / h * rows).astype(int), 0, rows - 1)
    cells = sorted({(int(a), int(b)) for a, b in zip(cx, cy)})
    return [[a, b] for a, b in cells]


def coverage_grid(records: Sequence[ShotRecord], grid: Tuple[int, int] = COVERAGE_GRID) -> np.ndarray:
    """ROWS x COLS counts of frames (board found, not skipped) touching each cell."""
    cols, rows = grid
    out = np.zeros((rows, cols), dtype=int)
    for r in records:
        if r.board_found and not r.skipped:
            for c, rr in r.coverage_cells:
                out[rr, c] += 1
    return out


def format_coverage(counts: np.ndarray, minimum: int = 3) -> str:
    lines = ["frame coverage (frames with corners in each 4x3 cell; want >= %d):" % minimum]
    for row in counts:
        lines.append("   " + "  ".join(f"{int(v):3d}{'' if v >= minimum else '!'}" for v in row))
    thin = int((counts < minimum).sum())
    lines.append("   all cells covered" if thin == 0 else f"   {thin} thin cell(s) marked '!'")
    return "\n".join(lines)


def range_hint(records: Sequence[ShotRecord], pan_max: float, tilt_max: float) -> Optional[str]:
    """If many shots had to be pulled toward the nominal, suggest ranges that
    would have fit first time - the lens's field of view is what decides
    them, and the defaults assume the 16 mm end."""
    shrunk = [r for r in records if r.attempts > 1 and not r.skipped and r.board_found]
    if len(shrunk) < max(3, len(records) // 4):
        return None
    # The largest pan/tilt that fit among the shots that had to shrink: the
    # pan-only and tilt-only views fit at full range, it is the combined
    # corner views that don't, so judge by those.
    ok_pan = max((abs(r.commanded.get("pan_deg", 0.0)) for r in shrunk), default=0.0)
    ok_tilt = max((abs(r.commanded.get("tilt_deg", 0.0)) for r in shrunk), default=0.0)
    sug_pan, sug_tilt = math.floor(ok_pan), math.floor(ok_tilt)
    if sug_pan >= pan_max and sug_tilt >= tilt_max:
        return None
    return (f"{len(shrunk)} of {len(records)} shots had to be pulled toward the nominal; "
            f"next time try --pan-max {min(sug_pan, math.floor(pan_max))} "
            f"--tilt-max {min(sug_tilt, math.floor(tilt_max))} "
            f"(currently {pan_max:g} / {tilt_max:g}) so the corner views land where planned.")


def calibration_command_hint(
    out_dir: str, pattern: Tuple[int, int], records: Sequence[ShotRecord],
    zoom_position: Optional[int], focus_position: Optional[int], square_mm: Optional[float],
) -> str:
    holdout_names = [
        os.path.basename((r.capture or {}).get("path") or (r.capture or {}).get("name") or "")
        for r in records if r.holdout and r.capture and not r.skipped
    ]
    holdout_names = [n for n in holdout_names if n]
    parts = [
        "venv/bin/python scripts/calibrate_distortion.py",
        f"  --input '{os.path.join(out_dir, '*.ARW')}'",
        f"  --pattern {pattern[0]}x{pattern[1]} --square-mm {square_mm if square_mm else '<measured>'}",
        f"  --out calibration/<body>-<lens>-<zoom>-<aperture>.json",
    ]
    if zoom_position is not None or focus_position is not None:
        parts.append(
            f"  --zoom-position {zoom_position if zoom_position is not None else '<n>'}"
            f" --focus-position {focus_position if focus_position is not None else '<n>'}"
        )
    if holdout_names:
        parts.append("  --holdout-files " + " ".join(holdout_names))
    parts.append(f"  --debug-dir {os.path.join(out_dir, 'debug')}")
    return " \\\n".join(parts)


# --------------------------------------------------------------------------
# Hardware runner
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Shooter:
    def __init__(self, args: argparse.Namespace, pattern: Tuple[int, int]):
        self.args = args
        self.pattern = pattern
        self.records: List[ShotRecord] = []
        self.manifest: Dict[str, Any] = {
            "tool": f"image-processing shoot_calibration_board {TOOL_VERSION}",
            "created_at": _now(),
            "address": args.address,
            "arm": args.arm, "camera": args.camera,
            "pattern": list(pattern),
            "dry_run": bool(args.dry_run),
            "shots": [],
        }
        self.robot = None
        self.arm = None
        self.cam = None
        self.nominal_joints: List[float] = []
        self.pose0: Optional[Dict[str, float]] = None
        self.liveview_size: Optional[Tuple[int, int]] = None

    # -- connection -------------------------------------------------------

    async def connect(self) -> None:
        from viam.components.arm import Arm
        from viam.components.camera import Camera
        from viam.robot.client import RobotClient

        key = self.args.api_key or os.environ.get("VIAM_API_KEY")
        key_id = self.args.api_key_id or os.environ.get("VIAM_API_KEY_ID")
        if not key or not key_id:
            raise SystemExit("need --api-key/--api-key-id or VIAM_API_KEY/VIAM_API_KEY_ID")
        opts = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
        self.robot = await RobotClient.at_address(self.args.address, opts)
        self.arm = Arm.from_robot(self.robot, self.args.arm)
        self.cam = Camera.from_robot(self.robot, self.args.camera)

    async def close(self) -> None:
        if self.robot is not None:
            await self.robot.close()

    # -- arm helpers --------------------------------------------------------

    async def read_joints_deg(self) -> List[float]:
        jp = await self.arm.get_joint_positions()
        return [float(v) for v in jp.values]

    async def move_joints(self, joints_deg: Sequence[float]) -> None:
        from viam.components.arm import JointPositions

        problems = check_joint_limits(joints_deg, self.args.joint_limits)
        if problems:
            raise RuntimeError("refusing move: " + "; ".join(problems))
        await self.arm.move_to_joint_positions(JointPositions(values=list(joints_deg)))
        await self.wait_settled()

    async def wait_settled(self) -> None:
        deadline = time.monotonic() + self.args.move_timeout_s
        while await self.arm.is_moving():
            if time.monotonic() > deadline:
                raise RuntimeError(f"arm still moving after {self.args.move_timeout_s}s")
            await asyncio.sleep(0.1)
        await asyncio.sleep(self.args.settle_s)

    async def move_distance(self, mm: float) -> List[float]:
        """Translate from the nominal pose along the viewing axis and return
        the joint vector there (the base for that distance's wrist deltas)."""
        from viam.proto.common import Pose

        if abs(mm) < 1e-6:
            await self.move_joints(self.nominal_joints)
            return list(self.nominal_joints)
        p0 = self.pose0
        x, y, z = translate_pose(
            (p0["x"], p0["y"], p0["z"]), (p0["o_x"], p0["o_y"], p0["o_z"]), mm, self.args.approach_axis
        )
        pose = Pose(x=x, y=y, z=z, o_x=p0["o_x"], o_y=p0["o_y"], o_z=p0["o_z"], theta=p0["theta"])
        await self.arm.move_to_position(pose)
        await self.wait_settled()
        joints = await self.read_joints_deg()
        problems = check_joint_limits(joints, self.args.joint_limits)
        if problems:
            raise RuntimeError(f"distance {mm:+.0f} mm landed outside joint limits: " + "; ".join(problems))
        return joints

    # -- camera helpers -----------------------------------------------------

    async def liveview_gray(self) -> Optional[np.ndarray]:
        """One live-view frame as grayscale. The rig's stream server
        intermittently fails a get_images with a JPEG-dimension IndexError
        while another client is streaming; that is transient, so retry a few
        times before giving up on the frame (a None here must not be mistaken
        for "board out of frame")."""
        last_exc: Optional[Exception] = None
        for attempt in range(LIVEVIEW_RETRIES):
            try:
                images, _ = await self.cam.get_images(timeout=self.args.liveview_timeout_s)
            except Exception as exc:  # noqa: BLE001 - report and carry on
                last_exc = exc
                await asyncio.sleep(0.5)
                continue
            for im in images:
                gray = decode_liveview(bytes(im.data))
                if gray is not None:
                    return gray
            await asyncio.sleep(0.5)
        print(f"    live view failed {LIVEVIEW_RETRIES}x: {str(last_exc)[:160] if last_exc else 'undecodable frame'}")
        return None

    async def capture(self) -> Dict[str, Any]:
        resp = await self.cam.do_command({"capture": {}}, timeout=self.args.capture_timeout_s)
        cap = resp.get("capture", resp) if isinstance(resp, dict) else {}
        keep = ("path", "saved_to", "name", "size", "focus_position", "zoom_position",
                "capture_count", "duration_s", "settings", "paths")
        return {k: cap.get(k) for k in keep if k in cap}

    async def camera_status(self) -> Dict[str, Any]:
        try:
            resp = await self.cam.do_command({"get_status": {}}, timeout=10)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        return resp.get("get_status", resp) if isinstance(resp, dict) else {}

    # -- the run ------------------------------------------------------------

    async def run(self) -> int:
        a = self.args
        await self.connect()
        try:
            return await self._run()
        finally:
            await self.close()

    async def _run(self) -> int:
        a = self.args
        status = await self.camera_status()
        self.manifest["camera_status"] = {
            k: status.get(k) for k in ("model", "serial", "lens", "connected", "capture_dir")
        }
        print(f"camera: {status.get('model')} / {status.get('lens')} connected={status.get('connected')}")

        if a.focus_position is not None and not a.dry_run:
            resp = await self.cam.do_command(
                {"set_focus_position": {"position": a.focus_position}}, timeout=120
            )
            print(f"focus: {resp}")
            self.manifest["set_focus_position"] = resp

        # Nominal pose: the one thing recorded by hand.
        if a.nominal_deg:
            self.nominal_joints = list(a.nominal_deg)
            print(f"moving to --nominal-deg {['%.1f' % v for v in self.nominal_joints]}")
            await self.move_joints(self.nominal_joints)
        else:
            self.nominal_joints = await self.read_joints_deg()
            print(f"nominal = current joints {['%.1f' % v for v in self.nominal_joints]}")
        p = await self.arm.get_end_position()
        self.pose0 = {k: float(getattr(p, k)) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}
        self.manifest["nominal_joints_deg"] = self.nominal_joints
        self.manifest["nominal_pose"] = self.pose0

        # Sanity: the board must be visible from the nominal pose.
        gray = await self.liveview_gray()
        if gray is None:
            print("WARNING: no live view; board checks disabled (shooting blind)")
            a.no_liveview_check = True
        else:
            self.liveview_size = (gray.shape[1], gray.shape[0])
            self.manifest["liveview_size"] = list(self.liveview_size)
            c = find_board(gray, self.pattern)
            if c is None:
                print("ERROR: the whole board is not visible in live view from the nominal pose. "
                      "Centre it with margin first, or check --pattern.")
                if not a.force:
                    return 2
            else:
                print(f"nominal: board found, span {board_span_norm(c, self.liveview_size):.2f} of frame")

        shots = plan_shots(
            count=a.count, holdouts=a.holdouts,
            pan_max_deg=a.pan_max, tilt_max_deg=a.tilt_max, roll_max_deg=a.roll_max,
            distances_mm=a.distances_mm, pan_levels=a.pan_levels, tilt_levels=a.tilt_levels,
            seed=a.seed,
        )
        self.manifest["plan"] = [asdict(s) for s in shots]
        print(f"planned {len(shots)} shots ({a.count} calibration + {a.holdouts} holdout) "
              f"over distances {a.distances_mm} mm"
              + ("  [DRY RUN - no shutter]" if a.dry_run else ""))

        if a.copy_to and not a.dry_run:
            os.makedirs(a.copy_to, exist_ok=True)

        try:
            await self._shoot_all(shots)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\ninterrupted - stopping the arm")
            try:
                await self.arm.stop()
            except Exception:  # noqa: BLE001
                pass
            self.manifest["interrupted"] = True
        finally:
            try:
                print("returning to nominal")
                await self.move_joints(self.nominal_joints)
            except Exception as exc:  # noqa: BLE001
                print(f"could not return to nominal: {exc}")
            self._write_manifest()

        counts = coverage_grid(self.records)
        print()
        print(format_coverage(counts))
        good = [r for r in self.records if r.capture and not r.skipped]
        print(f"{len(good)} frames captured, {sum(1 for r in self.records if r.skipped)} skipped")
        hint = range_hint(self.records, a.pan_max, a.tilt_max)
        if hint:
            print(hint)
        if not a.dry_run and good:
            focus = next((r.capture.get("focus_position") for r in good if r.capture.get("focus_position") is not None), None)
            print("\nnext:\n" + calibration_command_hint(
                a.copy_to or a.out_dir, self.pattern, self.records, a.zoom_position, focus, a.square_mm))
        return 0

    async def _shoot_all(self, shots: Sequence[PlannedShot]) -> None:
        a = self.args
        base_for_distance: Dict[float, List[float]] = {}
        current_distance: Optional[float] = None
        bad_distances: set = set()
        for shot in shots:
            if shot.distance_mm != current_distance:
                print(f"\n-- distance {shot.distance_mm:+.0f} mm")
                current_distance = shot.distance_mm
                base_for_distance[shot.distance_mm] = await self.move_distance(shot.distance_mm)
                if not await self._board_visible_here():
                    print(f"   board not fully visible from the base pose at {shot.distance_mm:+.0f} mm; "
                          f"skipping this distance (check --approach-axis / --distances-mm)")
                    bad_distances.add(shot.distance_mm)
            if shot.distance_mm in bad_distances:
                rec = ShotRecord(
                    index=shot.index, holdout=shot.holdout, distance_mm=shot.distance_mm,
                    planned={"pan_deg": shot.pan_deg, "tilt_deg": shot.tilt_deg, "roll_deg": shot.roll_deg},
                    skipped="board not visible from this distance's base pose",
                )
                self.records.append(rec)
                self.manifest["shots"].append(asdict(rec))
                continue
            base = base_for_distance[shot.distance_mm]
            rec = await self._shoot_one(shot, base)
            self.records.append(rec)
            self.manifest["shots"].append(asdict(rec))
            self._write_manifest()   # partial manifests survive a crash

    async def _board_visible_here(self) -> bool:
        """Whole board in live view at the current pose (True when checks are off)."""
        if self.args.no_liveview_check:
            return True
        gray = await self.liveview_gray()
        if gray is None:
            return True   # no live view at all was already reported; don't block
        c = find_board(gray, self.pattern)
        if c is not None:
            print(f"   base: board found, span {board_span_norm(c, (gray.shape[1], gray.shape[0])):.2f} of frame")
        return c is not None

    async def _shoot_one(self, shot: PlannedShot, base: Sequence[float]) -> ShotRecord:
        a = self.args
        rec = ShotRecord(
            index=shot.index, holdout=shot.holdout, distance_mm=shot.distance_mm,
            planned={"pan_deg": shot.pan_deg, "tilt_deg": shot.tilt_deg, "roll_deg": shot.roll_deg},
        )
        scale = 1.0
        tag = "H" if shot.holdout else " "
        for attempt in range(1, MAX_SHRINK_ATTEMPTS + 2):
            pan, tilt, roll = (shot.pan_deg * scale, shot.tilt_deg * scale, shot.roll_deg * scale)
            joints = joints_for(base, pan, tilt, roll,
                                pan_joint=a.pan_joint, tilt_joint=a.tilt_joint, roll_joint=a.roll_joint)
            rec.attempts = attempt
            rec.commanded = {"pan_deg": pan, "tilt_deg": tilt, "roll_deg": roll}
            rec.joints_commanded_deg = joints
            problems = check_joint_limits(joints, a.joint_limits)
            if problems:
                print(f"[{shot.index:02d}]{tag} pan {pan:+6.1f} tilt {tilt:+6.1f} roll {roll:+6.1f}: "
                      f"outside joint limits ({problems[0]}); shrinking")
                scale *= SHRINK_FACTOR
                continue
            await self.move_joints(joints)
            rec.joints_actual_deg = await self.read_joints_deg()

            if a.no_liveview_check:
                rec.board_found = None
                break
            gray = await self.liveview_gray()
            if gray is None:
                # No frame at all: the pose may be fine, we just can't tell.
                # Fire anyway (a bad frame is cheap to drop at fit time) but
                # record that it went unchecked.
                print(f"[{shot.index:02d}]{tag} pan {pan:+6.1f} tilt {tilt:+6.1f} roll {roll:+6.1f}: "
                      f"no live view; shooting unchecked")
                rec.board_found = None
                break
            corners = find_board(gray, self.pattern)
            if corners is None:
                print(f"[{shot.index:02d}]{tag} pan {pan:+6.1f} tilt {tilt:+6.1f} roll {roll:+6.1f}: "
                      f"board not fully visible; pulling toward nominal")
                rec.board_found = False
                scale *= SHRINK_FACTOR
                continue
            size = (gray.shape[1], gray.shape[0])
            rec.board_found = True
            rec.board_bbox_norm = board_bbox_norm(corners, size)
            rec.board_span_norm = board_span_norm(corners, size)
            rec.coverage_cells = coverage_cells(corners, size)
            break
        else:
            rec.skipped = "board never fully visible after shrinking"
            print(f"[{shot.index:02d}]{tag} skipped: {rec.skipped}")
            return rec

        if rec.board_found is False:
            rec.skipped = "board not visible"
            print(f"[{shot.index:02d}]{tag} skipped: {rec.skipped}")
            return rec

        where = ""
        if rec.board_bbox_norm:
            x0, y0, x1, y1 = rec.board_bbox_norm
            where = f" board x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}] span {rec.board_span_norm:.2f}"
        if a.dry_run:
            print(f"[{shot.index:02d}]{tag} pan {pan:+6.1f} tilt {tilt:+6.1f} roll {roll:+6.1f}{where}  (dry)")
            return rec

        t0 = time.perf_counter()
        rec.capture = await self.capture()
        path = rec.capture.get("path") or rec.capture.get("saved_to")
        if a.copy_to and path and os.path.exists(path):
            dst = os.path.join(a.copy_to, os.path.basename(path))
            shutil.copy2(path, dst)
            rec.copied_to = dst
        print(f"[{shot.index:02d}]{tag} pan {pan:+6.1f} tilt {tilt:+6.1f} roll {roll:+6.1f}{where}"
              f"  -> {os.path.basename(path) if path else rec.capture}  "
              f"focus={rec.capture.get('focus_position')}  ({time.perf_counter() - t0:.1f}s)")
        return rec

    def _write_manifest(self) -> None:
        os.makedirs(self.args.out_dir, exist_ok=True)
        path = os.path.join(self.args.out_dir, "shoot_manifest.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2)
        os.replace(tmp, path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    con = p.add_argument_group("machine")
    con.add_argument("--address", required=True, help="machine address, e.g. nines-photographer-main.xxxx.viam.cloud")
    con.add_argument("--api-key", default=None, help="or VIAM_API_KEY")
    con.add_argument("--api-key-id", default=None, help="or VIAM_API_KEY_ID")
    con.add_argument("--arm", default="arm", help="arm component name (default: arm)")
    con.add_argument("--camera", default="sony", help="still camera component with a `capture` DoCommand (default: sony)")

    plan = p.add_argument_group("plan")
    plan.add_argument("--pattern", type=parse_pattern, required=True, help="inner corners COLSxROWS, e.g. 10x7")
    plan.add_argument("--count", type=int, default=40, help="calibration frames (default: 40)")
    plan.add_argument("--holdouts", type=int, default=5, help="validation frames, low tilt (default: 5)")
    plan.add_argument("--pan-max", type=float, default=25.0, help="max pan delta, degrees (default: 25)")
    plan.add_argument("--tilt-max", type=float, default=20.0, help="max tilt delta, degrees (default: 20)")
    plan.add_argument("--roll-max", type=float, default=15.0, help="roll delta cycled 0/+/- (default: 15)")
    plan.add_argument("--pan-levels", type=int, default=5)
    plan.add_argument("--tilt-levels", type=int, default=4)
    plan.add_argument("--distances-mm", type=float, nargs="+", default=[0.0, -150.0],
                      help="translations along --approach-axis from the nominal; 0 = nominal; "
                           "positive tool-z = toward the board (default: 0 -150)")
    plan.add_argument("--approach-axis", choices=AXES, default="tool-z",
                      help="axis for --distances-mm; tool-z is the direction the tool points, "
                           "so positive is closer to the board (default)")
    plan.add_argument("--pan-joint", type=int, default=4, help="1-indexed joint used for pan (default: 4)")
    plan.add_argument("--tilt-joint", type=int, default=5, help="1-indexed joint used for tilt (default: 5)")
    plan.add_argument("--roll-joint", type=int, default=6, help="1-indexed joint used for roll (default: 6)")
    plan.add_argument("--nominal-deg", type=float, nargs="+", default=None,
                      help="nominal joints in degrees; default: wherever the arm is when the script starts")
    plan.add_argument("--joint-limits", dest="joint_limits", type=parse_joint_limits,
                      default=DEFAULT_JOINT_LIMITS_DEG,
                      help="lo:hi,... degrees per joint (default mirrors the rig's motion-service overrides)")
    plan.add_argument("--seed", type=int, default=0)

    run = p.add_argument_group("run")
    run.add_argument("--dry-run", action="store_true", help="move and check live view, never fire the shutter")
    run.add_argument("--no-liveview-check", action="store_true", help="skip the board-in-frame check")
    run.add_argument("--force", action="store_true", help="continue even if the board is not seen from the nominal pose")
    run.add_argument("--settle-s", type=float, default=1.5, help="pause after the arm stops (default: 1.5)")
    run.add_argument("--move-timeout-s", type=float, default=30.0)
    run.add_argument("--liveview-timeout-s", type=float, default=10.0)
    run.add_argument("--capture-timeout-s", type=float, default=60.0)
    run.add_argument("--focus-position", type=int, default=None,
                     help="drive the camera to this focus position first (production setpoint)")
    run.add_argument("--zoom-position", type=int, default=None, help="recorded for the CLI hint only")
    run.add_argument("--square-mm", type=float, default=None, help="recorded for the CLI hint only")

    out = p.add_argument_group("output")
    out.add_argument("--out-dir", default=None,
                     help="where shoot_manifest.json goes (default: calib/<date>)")
    out.add_argument("--copy-to", default=None,
                     help="copy each ARW here right after capture (when running on the rig host); "
                          "protects the set from the camera module's retention")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out_dir is None:
        args.out_dir = os.path.join("calib", datetime.now().strftime("%Y-%m-%d"))
    if len({args.pan_joint, args.tilt_joint, args.roll_joint}) != 3:
        print("pan, tilt and roll joints must be distinct", file=sys.stderr)
        return 2
    shooter = Shooter(args, args.pattern)
    try:
        return asyncio.run(shooter.run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
