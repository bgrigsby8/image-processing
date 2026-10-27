"""Tests for shoot_calibration_board.py - the arm-driven checkerboard capture.

Only the pure parts are tested: pose planning, joint maths, the live-view
board check and the coverage grid. No arm, no camera, no Viam connection.
"""

import cv2
import numpy as np
import pytest

from models import shoot_calibration_board as sb


PATTERN = (10, 7)


def test_plan_counts_and_holdouts():
    shots = sb.plan_shots(count=40, holdouts=5, pan_max_deg=25, tilt_max_deg=20,
                          roll_max_deg=15, distances_mm=[0, 200])
    assert len(shots) == 45
    assert sum(1 for s in shots if s.holdout) == 5
    assert sum(1 for s in shots if not s.holdout) == 40
    assert [s.index for s in shots] == list(range(45))
    # Every view within the requested ranges.
    for s in shots:
        assert abs(s.pan_deg) <= 25 + 1e-9
        assert abs(s.tilt_deg) <= 20 + 1e-9
        assert abs(s.roll_deg) <= 15 + 1e-9
    # Holdouts are low-tilt and unrolled.
    for s in shots:
        if s.holdout:
            assert s.roll_deg == 0.0
            assert abs(s.pan_deg) <= 12.5 and abs(s.tilt_deg) <= 10


def test_plan_grouped_by_distance_with_one_translation_each():
    shots = sb.plan_shots(count=40, holdouts=5, pan_max_deg=25, tilt_max_deg=20,
                          roll_max_deg=15, distances_mm=[0, 200, -150])
    seen, order = set(), []
    for s in shots:
        if s.distance_mm not in seen:
            seen.add(s.distance_mm)
            order.append(s.distance_mm)
    # Distances appear as contiguous blocks, in the order given.
    assert order == [0.0, 200.0, -150.0]
    blocks = [s.distance_mm for s in shots]
    assert blocks == sorted(blocks, key=order.index)
    # Holdouts spread over distances, each block ends with its holdouts.
    for d in order:
        block = [s for s in shots if s.distance_mm == d]
        flags = [s.holdout for s in block]
        assert flags == sorted(flags)   # False... then True...


def test_plan_tops_up_or_subsamples_grid():
    # Grid 5x4x1 = 20 < 40: random top-up, deterministic under a seed.
    a = sb.plan_shots(count=40, holdouts=0, pan_max_deg=25, tilt_max_deg=20,
                      roll_max_deg=0, distances_mm=[0], seed=1)
    b = sb.plan_shots(count=40, holdouts=0, pan_max_deg=25, tilt_max_deg=20,
                      roll_max_deg=0, distances_mm=[0], seed=1)
    assert len(a) == 40 and [(s.pan_deg, s.tilt_deg) for s in a] == [(s.pan_deg, s.tilt_deg) for s in b]
    # Grid 5x4x3 = 60 > 12: subsample, still every distance represented.
    c = sb.plan_shots(count=12, holdouts=0, pan_max_deg=25, tilt_max_deg=20,
                      roll_max_deg=0, distances_mm=[0, 100, 200])
    assert len(c) == 12
    assert {s.distance_mm for s in c} == {0.0, 100.0, 200.0}


def test_plan_covers_extremes_and_cycles_roll():
    shots = sb.plan_shots(count=40, holdouts=0, pan_max_deg=25, tilt_max_deg=20,
                          roll_max_deg=15, distances_mm=[0, 200])
    pans = {s.pan_deg for s in shots}
    tilts = {s.tilt_deg for s in shots}
    assert {-25.0, 25.0} <= pans and {-20.0, 20.0} <= tilts
    rolls = [s.roll_deg for s in shots]
    assert set(rolls) == {0.0, 15.0, -15.0}


def test_plan_rejects_negative():
    with pytest.raises(ValueError):
        sb.plan_shots(count=-1, holdouts=0, pan_max_deg=1, tilt_max_deg=1, roll_max_deg=0, distances_mm=[0])


def test_joints_for_applies_deltas_to_named_joints():
    base = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    j = sb.joints_for(base, 5, -3, 2, pan_joint=4, tilt_joint=5, roll_joint=6)
    assert j == [10.0, 20.0, 30.0, 45.0, 47.0, 62.0]
    assert base == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]   # untouched
    with pytest.raises(ValueError):
        sb.joints_for(base, 0, 0, 0, pan_joint=4, tilt_joint=4, roll_joint=6)
    with pytest.raises(ValueError):
        sb.joints_for(base, 0, 0, 0, pan_joint=7, tilt_joint=5, roll_joint=6)


def test_joint_limits():
    lim = sb.DEFAULT_JOINT_LIMITS_DEG
    ok = [90.0, -30.0, -40.0, 4.0, 77.0, 89.0]      # ~ the rig's outfits pose
    assert sb.check_joint_limits(ok, lim) == []
    bad = list(ok)
    bad[3] = 60.0
    msgs = sb.check_joint_limits(bad, lim)
    assert len(msgs) == 1 and msgs[0].startswith("joint 4")
    parsed = sb.parse_joint_limits("0:180,-180:45")
    assert parsed == [(0.0, 180.0), (-180.0, 45.0)]


def test_translate_pose_along_tool_z_and_world_axes():
    xyz, ov = (100.0, 200.0, 300.0), (0.0, 0.6, 0.8)
    assert sb.translate_pose(xyz, ov, 100, "tool-z") == pytest.approx([100.0, 260.0, 380.0])
    assert sb.translate_pose(xyz, ov, -50, "+x") == pytest.approx([50.0, 200.0, 300.0])
    assert sb.translate_pose(xyz, ov, 50, "-z") == pytest.approx([100.0, 200.0, 250.0])
    # Orientation vector is normalised before use.
    assert sb.translate_pose(xyz, (0, 0, 2), 10, "tool-z") == pytest.approx([100.0, 200.0, 310.0])
    with pytest.raises(ValueError):
        sb.translate_pose(xyz, ov, 1, "tool-x")


def _render_board(size_wh, origin, square_px, pattern=PATTERN, angle_deg=0.0):
    """A checkerboard with ``pattern`` inner corners drawn on a grey field,
    optionally rotated about its centre. Returns (gray8, inner-corner bbox)."""
    w, h = size_wh
    cols, rows = pattern
    img = np.full((h, w), 128, np.uint8)
    board = np.zeros(((rows + 1) * square_px, (cols + 1) * square_px), np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[r * square_px:(r + 1) * square_px, c * square_px:(c + 1) * square_px] = 255
    bh, bw = board.shape
    ox, oy = origin
    if angle_deg:
        M = cv2.getRotationMatrix2D((bw / 2, bh / 2), angle_deg, 1.0)
        board = cv2.warpAffine(board, M, (bw, bh), borderValue=128)
    # Clip so a board placed partly off-frame is simply cut off.
    vis = board[: max(0, min(bh, h - oy)), : max(0, min(bw, w - ox))]
    img[oy:oy + vis.shape[0], ox:ox + vis.shape[1]] = vis
    bbox = (ox + square_px, oy + square_px, ox + cols * square_px, oy + rows * square_px)
    return img, bbox


def test_find_board_and_coverage_cells():
    size = (1024, 680)
    img, (x0, y0, x1, y1) = _render_board(size, origin=(40, 30), square_px=40)
    corners = sb.find_board(img, PATTERN)
    assert corners is not None and corners.shape == (70, 2)
    bbox = sb.board_bbox_norm(corners, size)
    assert bbox == pytest.approx([x0 / 1024, y0 / 680, x1 / 1024, y1 / 680], abs=0.01)
    span = sb.board_span_norm(corners, size)
    assert 0.3 < span < 0.6
    cells = sb.coverage_cells(corners, size)
    # Board occupies the top-left ~44% x 41% of the frame: columns 0-1, rows 0-1.
    assert cells == [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_find_board_rejects_partial_board():
    size = (1024, 680)
    # Origin pushes the right edge of the board off the frame.
    img, _ = _render_board(size, origin=(700, 30), square_px=40)
    assert sb.find_board(img, PATTERN) is None


def test_find_board_on_dark_frame():
    # The rig's live view at strobe exposure: whites ~36/255, blacks ~1/255.
    img, _ = _render_board((1024, 680), origin=(40, 30), square_px=40)
    dark = (img.astype(np.float32) / 255.0 * 35 + 1).astype(np.uint8)
    assert sb.find_board(dark, PATTERN) is not None


def test_decode_liveview_roundtrip():
    img, _ = _render_board((640, 480), origin=(20, 20), square_px=30)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    gray = sb.decode_liveview(buf.tobytes())
    assert gray is not None and gray.shape == (480, 640)
    assert sb.find_board(gray, PATTERN) is not None
    assert sb.decode_liveview(b"not an image") is None


def test_coverage_grid_and_format():
    recs = []
    for i, cells in enumerate([[[0, 0], [1, 0]], [[0, 0]], [[3, 2]]]):
        r = sb.ShotRecord(index=i, holdout=False, distance_mm=0, planned={})
        r.board_found = True
        r.coverage_cells = cells
        recs.append(r)
    skipped = sb.ShotRecord(index=9, holdout=False, distance_mm=0, planned={})
    skipped.board_found = True
    skipped.coverage_cells = [[2, 1]]
    skipped.skipped = "board not visible"
    recs.append(skipped)
    counts = sb.coverage_grid(recs)
    assert counts.shape == (3, 4)
    assert counts[0, 0] == 2 and counts[0, 1] == 1 and counts[2, 3] == 1
    assert counts[1, 2] == 0            # skipped frames don't count
    text = sb.format_coverage(counts)
    assert "thin cell" in text and "!" in text


def test_calibration_command_hint_lists_holdouts():
    recs = []
    for i, name in enumerate(["DSC00001.ARW", "DSC00002.ARW", "DSC00003.ARW"]):
        r = sb.ShotRecord(index=i, holdout=(i == 1), distance_mm=0, planned={})
        r.capture = {"path": f"/home/viam/images/{name}"}
        recs.append(r)
    hint = sb.calibration_command_hint("calib/x", PATTERN, recs, 0, 16, 55.0)
    assert "--holdout-files DSC00002.ARW" in hint
    assert "--pattern 10x7 --square-mm 55.0" in hint
    assert "--zoom-position 0 --focus-position 16" in hint


def test_parser_defaults_are_40_plus_5():
    args = sb.build_parser().parse_args(["--address", "x", "--pattern", "10x7"])
    assert args.count == 40 and args.holdouts == 5
    assert args.pattern == (10, 7)
    assert args.distances_mm == [0.0, -150.0]
    assert args.joint_limits == sb.DEFAULT_JOINT_LIMITS_DEG


def test_range_hint_only_when_many_shots_shrank():
    def rec(i, attempts, pan, tilt):
        r = sb.ShotRecord(index=i, holdout=False, distance_mm=0, planned={})
        r.attempts, r.board_found = attempts, True
        r.commanded = {"pan_deg": pan, "tilt_deg": tilt, "roll_deg": 0.0}
        return r
    fine = [rec(i, 1, 25, 20) for i in range(8)]
    assert sb.range_hint(fine, 25, 20) is None
    shrunk = [rec(i, 2, 17.5, 14.0) for i in range(6)] + [rec(9, 1, 25.0, 0.0)]
    hint = sb.range_hint(shrunk, 25, 20)
    assert hint and "--pan-max 17 --tilt-max 14" in hint
    # Suggesting the current values is no suggestion at all.
    same = [rec(i, 2, 17.0, 14.0) for i in range(6)] + [rec(9, 1, 17.0, 0.0)]
    assert sb.range_hint(same, 17, 14) is None
