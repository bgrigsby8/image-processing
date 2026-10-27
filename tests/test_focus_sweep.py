"""Tests for focus_sweep.py - planning, peak and report maths, and the sweep
run against a fake camera. No Viam connection: the fake stands in for the
`sony` camera's DoCommands, and analysis is stubbed with an MTF50 curve."""

import argparse
import asyncio
import json
import os

import pytest

from models import focus_sweep as fs
from models.sharpness import FrameSharpness

BEST = 61300


def test_coarse_positions_include_stop_and_clamp():
    assert fs.coarse_positions(45000, 50000, 2000) == [45000, 47000, 49000, 50000]
    assert fs.coarse_positions(64000, 70000, 1000)[-1] == fs.FOLLOW_FOCUS_MAX
    assert fs.coarse_positions(50000, 45000, 2500) == [45000, 47500, 50000]
    with pytest.raises(ValueError):
        fs.coarse_positions(0, 10, 0)


def test_fine_positions_centred_and_clamped():
    assert fs.fine_positions(60000, 1000, 400) == [59200, 59600, 60000, 60400, 60800]
    assert max(fs.fine_positions(65400, 1000, 400)) == fs.FOLLOW_FOCUS_MAX


def _rec(phase, target, mtf, readback=None, approach=None):
    r = {"phase": phase, "target": target, "readback": target if readback is None else readback,
         "sharpness": {"mtf50": mtf, "edges_used": 150, "edges_total": 157}}
    if approach:
        r["approach"] = approach
    return r


def _curve(pos):
    return max(0.05, 0.3 - 4e-9 * (pos - BEST) ** 2)


def test_peak_uses_readback_and_skips_unmeasured():
    recs = [_rec("coarse", p, _curve(p + 200), readback=p + 200) for p in range(55000, 65001, 2000)]
    recs.append({"phase": "coarse", "target": 1, "readback": 1, "sharpness": {"error": "x"}})
    peak = fs.peak_of(recs)
    assert peak.best_position == pytest.approx(BEST, abs=300)


def test_report_includes_repeat_and_compare():
    recs = [_rec("coarse", p, _curve(p)) for p in range(55000, 65001, 2000)]
    recs += [_rec("repeat", BEST, 0.29, readback=BEST + 100, approach="below"),
             _rec("repeat", BEST, 0.20, readback=BEST - 100, approach="above"),
             _rec("compare", 59142, _curve(59142))]
    manifest = {}
    text = fs.report(recs, manifest)
    assert "best focus" in text
    assert "repeat from above" in text and "repeat from below" in text
    assert "compare 59142" in text
    assert manifest["peak"]["best_position"] == pytest.approx(BEST, abs=300)
    assert manifest["repeat"]["above"]["worst_vs_peak"] < manifest["repeat"]["below"]["worst_vs_peak"]
    json.dumps(manifest)


def test_format_peak_warns_at_sweep_edge():
    peak = fs.fit_focus_peak([1, 2, 3], [0.1, 0.2, 0.3])
    assert "WARNING" in fs.format_peak(peak, "x")
    assert "not enough" in fs.format_peak(None, "x")


# ---------------------------------------------------------------------------
# A whole run against a fake camera
# ---------------------------------------------------------------------------

class FakeCamera:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.position = 59142
        self.moves = []
        self.shots = 0

    async def do_command(self, cmd, timeout=None):
        if "get_status" in cmd:
            return {"model": "ILCE-7RM5", "connected": True}
        if "get_focus_position" in cmd:
            return {"position": self.position, "units": "follow_focus"}
        if "set_focus_position" in cmd:
            target = cmd["set_focus_position"]["position"]
            self.moves.append(target)
            self.position = target
            return {"position": target, "target": target, "ok": True, "units": "follow_focus"}
        if "capture" in cmd:
            self.shots += 1
            path = os.path.join(self.raw_dir, f"DSC{self.shots:05d}.ARW")
            with open(path, "w") as f:
                f.write(str(self.position))   # the stub analysis reads it back
            return {"capture": {"path": path, "saved_to": path, "name": os.path.basename(path)}}
        raise AssertionError(cmd)


def _stub_analyze(path, pattern, corners_hint=None, demosaic=None):
    with open(path) as f:
        pos = int(f.read())
    r = FrameSharpness(path=path, image_size=(9504, 6336), mtf50=_curve(pos),
                       edges_used=150, edges_total=157)
    return r, "corners"


def _args(tmp_path, **over):
    a = argparse.Namespace(
        address="x", api_key=None, api_key_id=None, camera="sony", pattern=(11, 8),
        start=53000, stop=65535, coarse_step=2000, fine_step=400, repeat=2,
        compare=[59142], approach_margin=3000, tolerance=None, dry_run=False,
        leave_at_best=False, focus_timeout_s=10, capture_timeout_s=10, analyze=None,
        out_dir=str(tmp_path / "out"), copy_to=None,
    )
    vars(a).update(over)
    return a


def test_full_run_finds_peak_and_restores_focus(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "analyze_raw", _stub_analyze)
    cam = FakeCamera(str(tmp_path))
    sweeper = fs.Sweeper(_args(tmp_path))
    sweeper.cam = cam
    assert asyncio.run(sweeper.execute()) == 0

    phases = [r["phase"] for r in sweeper.records]
    assert phases.count("coarse") == 8 and "fine" in phases
    assert phases.count("repeat") == 4 and phases.count("compare") == 1
    # Each sweep phase opens by approaching its first target from below.
    first_fine = next(r for r in sweeper.records if r["phase"] == "fine")
    assert first_fine["pre_position"] == first_fine["target"] - 3000
    with open(os.path.join(tmp_path, "out", fs.MANIFEST)) as f:
        manifest = json.load(f)
    assert manifest["peak"]["best_position"] == pytest.approx(BEST, abs=150)
    assert cam.moves[-1] == 59142          # focus put back where it was


def test_unreachable_raws_are_recorded_not_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "analyze_raw", _stub_analyze)
    cam = FakeCamera(str(tmp_path))

    async def capture_elsewhere(cmd, timeout=None, _orig=cam.do_command):
        resp = await _orig(cmd, timeout)
        if "capture" in cmd:
            os.remove(resp["capture"]["path"])
        return resp

    cam.do_command = capture_elsewhere
    sweeper = fs.Sweeper(_args(tmp_path))
    sweeper.cam = cam
    assert asyncio.run(sweeper.execute()) == 0
    assert all(r["phase"] == "coarse" for r in sweeper.records)
    assert all("not reachable" in r["sharpness"]["error"] for r in sweeper.records)


def test_analyze_offline_remeasures_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "analyze_raw", _stub_analyze)
    sweeper = fs.Sweeper(_args(tmp_path, fine_step=0, repeat=0, compare=[]))
    sweeper.cam = FakeCamera(str(tmp_path))
    asyncio.run(sweeper.execute())
    out = str(tmp_path / "out")
    with open(os.path.join(out, fs.MANIFEST)) as f:
        m = json.load(f)
    for r in m["records"]:
        r["sharpness"] = {}
    m.pop("peak")
    with open(os.path.join(out, fs.MANIFEST), "w") as f:
        json.dump(m, f)
    assert fs.main(["--analyze", out]) == 0
    with open(os.path.join(out, fs.MANIFEST)) as f:
        again = json.load(f)
    assert again["peak"]["best_position"] == pytest.approx(BEST, abs=400)
    assert "analyzed_at" in again
