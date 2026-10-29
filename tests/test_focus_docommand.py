"""color-correction's focus DoCommands: `measure_sharpness`, and the
background `focus_sweep` with its status / cancel commands and the guard
that keeps other callers off the camera while a sweep runs."""

import asyncio

import cv2
import numpy as np
import pytest

from models import focus_sweep as fs
from models.image_io import linear_to_srgb

from test_color_correction import _component
from test_focus_sweep import BEST, FakeCamera, _failing, _stub_analyze
from test_sharpness import PATTERN, _board_image, _expected_mtf50


@pytest.fixture
def cc(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "analyze_raw", _stub_analyze)
    cam = FakeCamera(str(tmp_path))
    component = _component(cam, output_dir=str(tmp_path / "out"))
    return component, cam


SWEEP = {"pattern": "11x8", "start": 53000, "stop": 65535, "repeat": 1, "compare": [59142],
         "settle_s": 0}


def test_focus_sweep_runs_in_background_and_reports(cc, tmp_path):
    component, cam = cc

    async def run():
        started = (await component.do_command({"focus_sweep": SWEEP}))["focus_sweep"]
        assert started["status"] == "running"
        assert started["plan"]["coarse"][0] == 53000
        assert started["out_dir"].startswith(str(tmp_path / "out" / "focus_sweeps"))
        return (await component.do_command(
            {"focus_sweep_status": {"id": started["id"], "wait_sec": 30}}
        ))["focus_sweep_status"]

    status = asyncio.run(run())
    assert status["status"] == "done"
    assert status["peak"]["best_position"] == pytest.approx(BEST, abs=150)
    assert status["repeat"]["below"]["frames"] == 1
    assert "best focus" in status["report"]
    assert status["frames"] == len(status["rows"])
    assert {r["phase"] for r in status["rows"]} == {"coarse", "fine", "repeat", "compare"}
    assert cam.moves[-1] == 59142   # focus restored


def test_camera_guarded_while_sweep_runs_then_cancel_restores(cc):
    component, cam = cc

    async def run():
        await component.do_command({"focus_sweep": SWEEP})
        # The sweep task has not run yet: it is "running" from here on.
        with pytest.raises(RuntimeError, match="driving the camera"):
            await component.do_command({"capture": {}})
        with pytest.raises(RuntimeError, match="driving the camera"):
            await component.do_command({"set_focus_position": {"position": 1}})
        with pytest.raises(RuntimeError, match="already running"):
            await component.do_command({"focus_sweep": SWEEP})
        # Read-only commands still reach the source.
        assert (await component.do_command({"get_status": {}}))["connected"] is True
        await asyncio.sleep(0)          # let it start shooting
        return (await component.do_command({"focus_sweep_cancel": {}}))["focus_sweep_cancel"]

    status = asyncio.run(run())
    assert status["status"] == "cancelled"
    assert cam.moves[-1] == 59142
    # The guard lifts once the sweep is over.
    assert asyncio.run(component.do_command({"get_focus_position": {}}))["position"] == 59142


def test_focus_sweep_that_cannot_capture_reports_failed(cc):
    component, cam = cc
    _failing(cam, lambda: True)

    async def run():
        await component.do_command({"focus_sweep": SWEEP})
        return (await component.do_command(
            {"focus_sweep_status": {"wait_sec": 30}}
        ))["focus_sweep_status"]

    status = asyncio.run(run())
    assert status["status"] == "failed"
    assert "3 frames in a row failed" in status["error"]
    assert all("no file appeared" in r["error"] for r in status["rows"])
    assert cam.moves[-1] == 59142


def test_focus_sweep_rejects_bad_options(cc):
    component, _ = cc
    with pytest.raises(ValueError, match="unknown focus_sweep option"):
        asyncio.run(component.do_command({"focus_sweep": {"pattern": "11x8", "strat": 1}}))
    with pytest.raises(ValueError, match="pattern"):
        asyncio.run(component.do_command({"focus_sweep": {"start": 1}}))
    with pytest.raises(ValueError, match="pattern"):
        asyncio.run(component.do_command({"focus_sweep": {"pattern": [11]}}))


def test_focus_sweep_status_before_any_sweep(cc):
    component, _ = cc
    assert asyncio.run(component.do_command({"focus_sweep_status": {}})) == {
        "focus_sweep_status": {"status": "none"}
    }
    assert asyncio.run(component.do_command({"focus_sweep_cancel": {}})) == {
        "focus_sweep_cancel": {"status": "not_running"}
    }


def test_measure_sharpness_on_a_file(cc, tmp_path):
    component, _ = cc
    srgb = linear_to_srgb(np.clip(_board_image(1.0), 0, 1))
    path = str(tmp_path / "board.png")
    cv2.imwrite(path, np.rint(srgb * 255).astype(np.uint8))
    out = asyncio.run(component.do_command(
        {"measure_sharpness": {"path": path, "pattern": list(PATTERN), "edges": True}}
    ))["measure_sharpness"]
    assert out["error"] is None
    assert out["mtf50"] == pytest.approx(_expected_mtf50(1.0), rel=0.1)
    assert len(out["edges"]) == out["edges_total"]


def test_measure_sharpness_needs_an_existing_path(cc, tmp_path):
    component, _ = cc
    with pytest.raises(ValueError, match="path"):
        asyncio.run(component.do_command({"measure_sharpness": {"pattern": "11x8"}}))
    with pytest.raises(FileNotFoundError):
        asyncio.run(component.do_command(
            {"measure_sharpness": {"path": str(tmp_path / "nope.ARW"), "pattern": "11x8"}}
        ))
