"""Tests for the color-correction math: CCM fitting, application, sampling.

Only the Viam-decoupled pieces are exercised (ColorCorrector, PatchSampler,
_fit_ccm, _order_corners, detect_colorchecker plumbing) — no viam-server, no
camera hardware.
"""

import numpy as np
import pytest

from models.color_correction import (
    _oriented_chart_grid,
    REFERENCE_SRGB,
    ColorCorrector,
    PatchSampler,
    _fit_ccm,
    _neutral_brightness_report,
    _order_corners,
)
from models.image_io import linear_to_srgb, srgb_to_linear


# ---------------------------------------------------------------------------
# _fit_ccm
# ---------------------------------------------------------------------------

def test_fit_ccm_recovers_known_matrix():
    """If reference = measured @ A.T exactly, the fit must recover A."""
    rng = np.random.default_rng(42)
    measured = rng.uniform(0.05, 0.9, size=(24, 3)).astype(np.float32)
    a = np.array(
        [[1.2, -0.1, 0.05],
         [-0.08, 1.1, -0.02],
         [0.03, -0.15, 1.3]],
        dtype=np.float32,
    )
    reference = measured @ a.T
    ccm = _fit_ccm(measured, reference)
    assert np.allclose(ccm, a, atol=1e-4)


def test_fit_ccm_identity_when_measured_equals_reference():
    reference = srgb_to_linear(REFERENCE_SRGB)
    ccm = _fit_ccm(reference, reference)
    assert np.allclose(ccm, np.eye(3), atol=1e-4)


# ---------------------------------------------------------------------------
# ColorCorrector
# ---------------------------------------------------------------------------

def test_corrector_rejects_bad_shape():
    with pytest.raises(ValueError, match="3x3"):
        ColorCorrector(np.eye(4))


def test_identity_is_noop_passthrough():
    corrector = ColorCorrector.identity()
    assert corrector.is_identity
    img = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    # Identity returns the input object untouched (no gamma round-trip).
    assert corrector.apply_to_rgb(img) is img
    linear = img.astype(np.float32) / 255.0
    assert corrector.apply_to_linear(linear) is linear


def test_apply_to_linear_matches_manual_matmul():
    ccm = np.array(
        [[0.9, 0.1, 0.0],
         [0.0, 1.0, 0.0],
         [0.05, -0.05, 1.0]],
        dtype=np.float32,
    )
    corrector = ColorCorrector(ccm)
    linear = np.random.default_rng(1).uniform(0, 1, (4, 5, 3)).astype(np.float32)
    out = corrector.apply_to_linear(linear)
    expected = (linear.reshape(-1, 3) @ ccm.T).reshape(4, 5, 3)
    assert out.shape == linear.shape
    assert np.allclose(out, expected, atol=1e-6)


def test_apply_to_rgb_round_trips_through_linear():
    """A diagonal gain CCM must scale colors in *linear* light, not sRGB."""
    ccm = np.diag([0.5, 1.0, 1.0]).astype(np.float32)
    corrector = ColorCorrector(ccm)
    img = np.full((4, 4, 3), 188, dtype=np.uint8)
    out = corrector.apply_to_rgb(img)
    expected_r = linear_to_srgb(srgb_to_linear(np.float32(188 / 255.0)) * 0.5) * 255.0
    assert np.allclose(out[..., 0], expected_r, atol=1)
    assert np.allclose(out[..., 1:], 188, atol=1)


def test_apply_to_rgb_clips_to_uint8_range():
    corrector = ColorCorrector(np.diag([3.0, 3.0, 3.0]).astype(np.float32))
    img = np.full((2, 2, 3), 250, dtype=np.uint8)
    out = corrector.apply_to_rgb(img)
    assert out.dtype == np.uint8
    assert out.max() == 255


# ---------------------------------------------------------------------------
# Corner ordering (chart detection geometry)
# ---------------------------------------------------------------------------

def test_order_corners_handles_any_winding():
    tl, tr, br, bl = (10, 20), (200, 25), (205, 150), (8, 145)
    for perm in ([br, tl, bl, tr], [tr, br, tl, bl], [bl, tr, br, tl]):
        ordered = _order_corners(np.array(perm, dtype=np.float32))
        assert np.allclose(ordered, np.array([tl, tr, br, bl], dtype=np.float32))


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

def _synthetic_chart(patch_px: int = 60) -> np.ndarray:
    """Render REFERENCE_SRGB as a borderless 4x6 grid filling the frame."""
    rows, cols = 4, 6
    chart8 = (REFERENCE_SRGB * 255.0).round().astype(np.uint8)
    img = np.zeros((rows * patch_px, cols * patch_px, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            img[r * patch_px:(r + 1) * patch_px, c * patch_px:(c + 1) * patch_px] = (
                chart8[r * cols + c]
            )
    return img


def test_sample_at_centers_reads_exact_patches():
    img = _synthetic_chart()
    centers = [(c * 60 + 30, r * 60 + 30) for r in range(4) for c in range(6)]
    measured = PatchSampler.sample_at_centers(img, centers)
    assert measured.shape == (24, 3)
    assert np.allclose(measured, srgb_to_linear(REFERENCE_SRGB), atol=0.005)


def test_sample_linear_at_centers_clamps_at_edges():
    linear = np.random.default_rng(2).uniform(0, 1, (32, 32, 3)).astype(np.float32)
    # Centers at the very corners must not produce empty slices or wrap around.
    samples = PatchSampler.sample_linear_at_centers(linear, [(0, 0), (31, 31)], radius=10)
    assert samples.shape == (2, 3)
    assert np.all(np.isfinite(samples))


def test_calibrate_from_rgb_on_perfect_chart_is_near_identity():
    """A frame-filling chart at exactly the reference colors needs ~no correction."""
    img = _synthetic_chart()
    centers = [(c * 60 + 30, r * 60 + 30) for r in range(4) for c in range(6)]
    corrector = ColorCorrector.calibrate_from_rgb(img, patch_centers=centers)
    assert np.allclose(corrector.ccm, np.eye(3), atol=0.02)


def test_calibrate_from_rgb_corrects_a_cast():
    """Calibrating on a green-tinted chart yields a CCM that undoes the tint."""
    tint = np.diag([0.8, 1.1, 0.9]).astype(np.float32)
    img = _synthetic_chart()
    tinted_linear = srgb_to_linear(img.astype(np.float32) / 255.0) @ tint.T
    tinted = (linear_to_srgb(tinted_linear) * 255.0).round().astype(np.uint8)

    centers = [(c * 60 + 30, r * 60 + 30) for r in range(4) for c in range(6)]
    corrector = ColorCorrector.calibrate_from_rgb(tinted, patch_centers=centers)
    corrected = corrector.apply_to_rgb(tinted)

    ref8 = (REFERENCE_SRGB * 255.0).round()
    sampled = np.array([corrected[y, x] for x, y in centers], dtype=np.float32)
    assert np.abs(sampled - ref8).mean() < 3.0


def test_neutral_brightness_report_at_nominal_matches_reference():
    """Patches measured at exactly the reference colors read back the reference
    0-255 values. Neutral 6.5 lands at ~161 (the real "After November 2014"
    chart's grey isn't exactly 160 - it's derived from the published xyY)."""
    report = _neutral_brightness_report(srgb_to_linear(REFERENCE_SRGB))
    assert set(report) == {
        "white_9_5", "neutral_8", "neutral_6_5", "neutral_5", "neutral_3_5", "black_2"
    }
    for patch in report.values():
        assert patch["measured"] == pytest.approx(patch["reference"], abs=0.5)
    assert report["neutral_6_5"]["reference"] == pytest.approx(161.4, abs=1.0)


def test_reference_srgb_derived_from_after_nov2014_xyy():
    """REFERENCE_SRGB is computed from the published xyY (After Nov 2014) data,
    not the stale pre-2014 sRGB table."""
    srgb = REFERENCE_SRGB * 255.0
    assert REFERENCE_SRGB.shape == (24, 3)
    assert REFERENCE_SRGB.min() >= 0.0 and REFERENCE_SRGB.max() <= 1.0
    # Neutral row (19-23) stays near-neutral (white carries a slight real warm
    # tint), at ~91%-reflectance white (not 255).
    for idx in range(18, 24):
        r, g, b = srgb[idx]
        assert max(r, g, b) - min(r, g, b) < 6.0
    assert srgb[18].mean() == pytest.approx(245, abs=2)   # white ~245, not 255
    # The post-2014 fix: blue patch (13) red channel ~38, NOT the stale 56.
    assert srgb[12, 0] == pytest.approx(38, abs=3)


def test_neutral_brightness_report_tracks_light_power():
    """A one-stop-under chart reads darker than reference on every patch -
    the readout the user watches while dialing flash power."""
    under = srgb_to_linear(REFERENCE_SRGB) * 0.5
    report = _neutral_brightness_report(under)
    for patch in report.values():
        assert patch["measured"] < patch["reference"]
    # Half the linear light on Neutral 6.5 (0.353 linear) lands around sRGB 117.
    assert 110 < report["neutral_6_5"]["measured"] < 125


# ---------------------------------------------------------------------------
# Orientation-robust chart grid (_oriented_chart_grid)
# ---------------------------------------------------------------------------

def _grid_corners(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float32)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_oriented_grid_resolves_any_90_degree_rotation(k):
    """A chart rotated k*90deg in frame must still map patches in reference order."""
    img = np.ascontiguousarray(np.rot90(_synthetic_chart(), k))
    detection = _oriented_chart_grid(img, _grid_corners(img))
    assert detection is not None
    assert detection["orientation_score"] > 2.0

    centers = [(int(x), int(y)) for x, y in detection["centers"]]
    measured = PatchSampler.sample_at_centers(img, centers)
    assert np.allclose(measured, srgb_to_linear(REFERENCE_SRGB), atol=0.005)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_oriented_grid_neutral_boxes_land_on_grays(k):
    """The WB boxes must cover Neutral 8 / 6.5 regardless of chart rotation."""
    img = np.ascontiguousarray(np.rot90(_synthetic_chart(), k))
    h, w = img.shape[:2]
    detection = _oriented_chart_grid(img, _grid_corners(img))
    assert detection is not None
    for box, expected in zip(detection["neutral_boxes_norm"], ([200] * 3, [160] * 3)):
        x0, y0, x1, y1 = box
        region = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        assert region.size > 0
        assert np.allclose(np.median(region.reshape(-1, 3), axis=0), expected, atol=2)


def test_oriented_grid_survives_white_balance_cast():
    """A strong channel cast (the wrong-WB case) must not confuse orientation."""
    img = _synthetic_chart()
    cast_linear = srgb_to_linear(img.astype(np.float32) / 255.0) * [0.4, 1.0, 2.2]
    cast = (linear_to_srgb(np.clip(cast_linear, 0, 1)) * 255).round().astype(np.uint8)
    cast = np.ascontiguousarray(np.rot90(cast, 1))

    detection = _oriented_chart_grid(cast, _grid_corners(cast))
    assert detection is not None

    upright = _synthetic_chart()
    upright_detection = _oriented_chart_grid(upright, _grid_corners(upright))
    # Same patch (white) must land at the rotated position of the upright one.
    ux, uy = upright_detection["centers"][18]
    rx, ry = detection["centers"][18]
    # rot90(k=1) maps (x, y) -> (y, W-1-x) where W is the original width
    assert abs(rx - uy) < 1.5 and abs(ry - (upright.shape[1] - 1 - ux)) < 1.5


def test_oriented_grid_rejects_non_chart():
    """A quad over random noise has no orientation that matches the reference."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, size=(240, 360, 3), dtype=np.uint8)
    assert _oriented_chart_grid(noise, _grid_corners(noise)) is None


# ---------------------------------------------------------------------------
# Deferred capture (`capture` with `defer` + `capture_result`): the pipelined
# flow for rigs that move between shots. Exercised against a fake source
# camera - no viam-server, no hardware.
# ---------------------------------------------------------------------------

import asyncio

from PIL import Image

from models.color_correction import ColorCorrection


class _FakeSource:
    """Fake PTP-style source camera: `trigger` hands back an on-camera path,
    `download` "saves" a file that already exists at `saved_path`."""

    def __init__(self, saved_path, supports_trigger=True, saves_to_disk=True):
        self.saved_path = saved_path
        self.supports_trigger = supports_trigger
        self.saves_to_disk = saves_to_disk
        self.commands = []

    async def do_command(self, command, *, timeout=None, **kwargs):
        self.commands.append(command)
        if "trigger" in command:
            if not self.supports_trigger:
                raise ValueError("no recognized command")
            return {"trigger": {"path": "/store/DCIM/IMG_0042.PNG",
                                "name": "IMG_0042.PNG"}}
        if "download" in command:
            return {"download": {
                "path": command["download"]["path"],
                "name": "IMG_0042.PNG",
                "saved_to": self.saved_path if self.saves_to_disk else None,
            }}
        if "capture" in command:
            return {"capture": {"saved_to": self.saved_path}}
        raise ValueError("no recognized command")


def _component(source, output_dir=None):
    cc = ColorCorrection("test-cc")
    cc.camera = source
    cc.corrector = ColorCorrector.identity()
    cc._white_balance = "camera"
    cc._exposure_stops = 0.0
    cc._tone = "none"
    cc._sharpen = "none"
    cc._demosaic = "DHT"
    cc._output_formats = ["tiff16", "jpeg"]
    cc._output_dir = output_dir
    cc._jpeg_quality = 95
    cc._write_sidecar = False
    cc._part_id = None
    cc._data_client = None
    cc._delete_after_upload = False
    cc._upload_dial_timeout_s = 30.0
    cc._upload_file_timeout_s = 180.0
    cc._nines_api_key = None
    cc._nines_org_slug = None
    cc._nines_base_url = "https://nines.test"
    cc._nines_item_ids = {}
    cc._pending_captures = {}
    cc._capture_seq = 0
    return cc


def _write_still(tmp_path):
    p = str(tmp_path / "IMG_0042.PNG")
    Image.fromarray(np.full((8, 8, 3), 120, np.uint8)).save(p, format="PNG")
    return p


def test_deferred_capture_round_trip(tmp_path):
    source = _FakeSource(_write_still(tmp_path))
    cc = _component(source, output_dir=str(tmp_path / "out"))

    async def run():
        ticket = (await cc.do_command({"capture": {"defer": True}}))["capture"]
        assert ticket["status"] == "pending"
        assert ticket["camera_path"] == "/store/DCIM/IMG_0042.PNG"
        result = (await cc.do_command(
            {"capture_result": {"id": ticket["capture_id"], "wait_sec": 30}}
        ))["capture_result"]
        return ticket, result

    ticket, result = asyncio.run(run())
    assert result["status"] == "done"
    assert result["source_path"] == source.saved_path
    assert result["image_base64"]  # preview present
    # Deferred captures hand off the RAW only - no exports, no sidecar.
    assert "exports" not in result
    # The ticket is collected exactly once.
    with pytest.raises(ValueError, match="unknown capture id"):
        asyncio.run(cc._capture_result({"id": ticket["capture_id"]}))


def test_deferred_capture_requires_trigger_support():
    source = _FakeSource(saved_path=None, supports_trigger=False)
    cc = _component(source)
    with pytest.raises(RuntimeError, match="`trigger`"):
        asyncio.run(cc.do_command({"capture": {"defer": True}}))


def test_deferred_capture_surfaces_background_failure(tmp_path):
    """A source without a download_dir fails in the background task; the
    error must surface on collect, not vanish."""
    source = _FakeSource(saved_path=None, saves_to_disk=False)
    cc = _component(source, output_dir=str(tmp_path / "out"))

    async def run():
        ticket = (await cc.do_command({"capture": {"defer": True}}))["capture"]
        with pytest.raises(RuntimeError, match="download_dir"):
            await cc.do_command(
                {"capture_result": {"id": ticket["capture_id"], "wait_sec": 30}}
            )

    asyncio.run(run())


def test_preview_only_capture_skips_exports(tmp_path):
    """`output_formats: []` is the preview-only fast path: no files written,
    preview still returned, RAW path handed back for a later `develop`."""
    source = _FakeSource(_write_still(tmp_path))
    cc = _component(source, output_dir=str(tmp_path / "out"))

    resp = asyncio.run(cc.do_command({"capture": {"output_formats": []}}))
    out = resp["capture"]
    assert out["exports"] == {}
    assert out["image_base64"]
    assert out["source_path"] == source.saved_path


def test_configured_exposure_stops_flows_into_develop(tmp_path, monkeypatch):
    """The `exposure_stops` config default reaches the raw decode when a call
    doesn't override it (the digital counterpart to flash power), and a per-call
    value still wins."""
    import models.color_correction as cc_mod

    cc = _component(_FakeSource(saved_path=None), output_dir=str(tmp_path / "out"))
    cc._exposure_stops = 0.87

    seen = {}
    real = cc_mod.load_linear_rgb

    def spy(path, **kwargs):
        seen["exposure_stops"] = kwargs.get("exposure_stops")
        return real(path, **kwargs)

    monkeypatch.setattr(cc_mod, "load_linear_rgb", spy)
    p = _write_still(tmp_path)

    asyncio.run(cc.do_command({"develop": {"path": p}}))
    assert seen["exposure_stops"] == 0.87  # config default applied

    asyncio.run(cc.do_command({"develop": {"path": p, "exposure_stops": 0.0}}))
    assert seen["exposure_stops"] == 0.0  # per-call override wins


def test_configured_tone_flows_into_export(tmp_path, monkeypatch):
    """The `tone` config default reaches the export (and sidecar), and a per-call
    `tone` overrides it."""
    import json

    import models.color_correction as cc_mod

    cc = _component(_FakeSource(saved_path=None), output_dir=str(tmp_path / "out"))
    cc._tone = "bright"
    cc._write_sidecar = True

    seen = {}
    real = cc_mod.export_renditions

    def spy(*args, **kwargs):
        seen["tone"] = kwargs.get("tone")
        return real(*args, **kwargs)

    monkeypatch.setattr(cc_mod, "export_renditions", spy)
    p = _write_still(tmp_path)

    out = asyncio.run(cc.do_command({"develop": {"path": p}}))["develop"]
    assert seen["tone"] == "bright"  # config default applied
    # ...and recorded in the sidecar for reproducibility.
    with open(out["sidecar"]) as f:
        assert json.load(f)["tone"] == "bright"

    asyncio.run(cc.do_command({"develop": {"path": p, "tone": "none"}}))
    assert seen["tone"] == "none"  # per-call override wins


# ---------------------------------------------------------------------------
# Chunked file upload: app.viam.com caps gRPC messages at 32 MiB, so `upload`
# must stream files in pieces rather than one message per file (which is what
# the SDK's `file_upload` does, silently dropping every CR3/TIFF). Exercised
# against a fake FileUpload stream - no cloud.
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from viam.proto.app.datasync import FileUploadResponse

from models.color_correction import UPLOAD_CHUNK_BYTES


class _FakeUploadStream:
    """Records every (request, end) pair sent over the FileUpload stream."""

    def __init__(self, sent):
        self.sent = sent

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send_message(self, msg, end=False):
        self.sent.append((msg, end))

    async def recv_message(self):
        return FileUploadResponse(binary_data_id="fake-binary-id")


class _FakeDataClient:
    def __init__(self):
        self.sent = []
        self._metadata = {"authorization": "Bearer fake"}
        self._data_sync_client = SimpleNamespace(
            FileUpload=SimpleNamespace(
                open=lambda metadata=None: _FakeUploadStream(self.sent)
            )
        )


def test_chunked_upload_splits_large_file(tmp_path):
    """A file larger than one chunk goes out as metadata + several FileData
    messages, each under the cap, reassembling to the original bytes."""
    payload = bytes(range(256)) * ((2 * UPLOAD_CHUNK_BYTES + 1234) // 256 + 1)
    path = tmp_path / "big.tif"
    path.write_bytes(payload)

    client = _FakeDataClient()
    binary_id = asyncio.run(
        ColorCorrection._file_upload_chunked(
            client, str(path),
            part_id="part-1", component_name="cc", tags=["sku:123"],
        )
    )

    assert binary_id == "fake-binary-id"
    meta_msg, meta_end = client.sent[0]
    assert meta_msg.metadata.file_name == "big.tif"
    assert meta_msg.metadata.file_extension == ".tif"
    assert list(meta_msg.metadata.tags) == ["sku:123"]
    assert meta_end is False

    chunks = [msg.file_contents.data for msg, _ in client.sent[1:]]
    assert len(chunks) > 1
    assert all(len(c) <= UPLOAD_CHUNK_BYTES for c in chunks)
    assert b"".join(chunks) == payload
    # Only the final message closes the stream.
    ends = [end for _, end in client.sent[1:]]
    assert ends == [False] * (len(ends) - 1) + [True]


def test_chunked_upload_honors_file_name_override(tmp_path):
    """A passed `file_name` becomes the cloud file_name; the extension is still
    derived from the on-disk path, independent of the renamed stem."""
    path = tmp_path / "IMG_0042.cr3"
    path.write_bytes(b"raw")

    client = _FakeDataClient()
    asyncio.run(
        ColorCorrection._file_upload_chunked(
            client, str(path),
            part_id="p", component_name="cc", tags=None,
            file_name="sku-front.cr3",
        )
    )

    meta = client.sent[0][0].metadata
    assert meta.file_name == "sku-front.cr3"
    assert meta.file_extension == ".cr3"


def test_chunked_upload_empty_file_sends_one_chunk(tmp_path):
    """An empty file still sends one (empty) FileData message, matching the
    SDK's behavior, so the stream is closed properly."""
    path = tmp_path / "empty.json"
    path.write_bytes(b"")

    client = _FakeDataClient()
    asyncio.run(
        ColorCorrection._file_upload_chunked(
            client, str(path), part_id="p", component_name="cc", tags=None,
        )
    )

    assert len(client.sent) == 2  # metadata + one empty chunk
    chunk_msg, chunk_end = client.sent[1]
    assert chunk_msg.file_contents.data == b""
    assert chunk_end is True


# ---------------------------------------------------------------------------
# Local-disk cleanup: nothing else in the pipeline ever deletes a local file,
# so the download dir grows by every frame shot. `delete` removes skipped
# captures (guarded to output_dir); `delete_after_upload` removes each file
# once it is confirmed in the cloud.
# ---------------------------------------------------------------------------


def _uploader_component(tmp_path, monkeypatch, fail_paths=()):
    """Component with upload wired to a fake cloud; uploads of `fail_paths` fail.

    Each call records the (path, file_name) forwarded to the chunked uploader on
    ``cc._uploaded_names`` so tests can assert how names were derived.
    """
    cc = _component(_FakeSource(None), output_dir=str(tmp_path))
    cc._part_id = "part-1"
    cc._uploaded_names = []

    async def fake_get_data_client():
        return _FakeDataClient()

    async def fake_upload_chunked(client, path, **kwargs):
        cc._uploaded_names.append((path, kwargs.get("file_name")))
        if path in fail_paths:
            raise RuntimeError("simulated upload failure")
        return "fake-binary-id"

    monkeypatch.setattr(cc, "_get_data_client", fake_get_data_client)
    monkeypatch.setattr(ColorCorrection, "_file_upload_chunked",
                        staticmethod(fake_upload_chunked))
    return cc


def test_upload_keeps_files_by_default(tmp_path, monkeypatch):
    path = tmp_path / "a.CR3"
    path.write_bytes(b"raw")
    cc = _uploader_component(tmp_path, monkeypatch)

    out = asyncio.run(cc._upload({"paths": [str(path)]}))
    assert out["uploaded"] == [str(path)]
    assert out["deleted"] == []
    assert path.exists()


def test_upload_name_replaces_stem_without_collision(tmp_path, monkeypatch):
    """An operator `name` replaces the capture stem on every file in the set,
    keeping the full post-stem suffix - so the default config's `_16.png` and
    `.png` siblings stay distinct rather than both collapsing to one name."""
    names = ["IMG_0042.cr3", "IMG_0042_16.png", "IMG_0042.png", "IMG_0042.json"]
    paths = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"x")
        paths.append(str(p))
    cc = _uploader_component(tmp_path, monkeypatch)

    asyncio.run(cc._upload({"paths": paths, "name": "sku-front"}))

    forwarded = {fn for _, fn in cc._uploaded_names}
    assert forwarded == {
        "sku-front.cr3",
        "sku-front_16.png",
        "sku-front.png",
        "sku-front.json",
    }


def test_upload_without_name_forwards_none(tmp_path, monkeypatch):
    """No `name` opt leaves the file_name to the chunked uploader's basename
    fallback (forwarded as None), preserving today's behavior."""
    path = tmp_path / "IMG_0042.cr3"
    path.write_bytes(b"raw")
    cc = _uploader_component(tmp_path, monkeypatch)

    asyncio.run(cc._upload({"paths": [str(path)]}))

    assert cc._uploaded_names == [(str(path), None)]


def test_delete_after_upload_removes_only_successful(tmp_path, monkeypatch):
    """Uploaded files are deleted; a failed upload keeps its file for retry."""
    ok = tmp_path / "a.CR3"
    bad = tmp_path / "b.CR3"
    ok.write_bytes(b"raw")
    bad.write_bytes(b"raw")
    cc = _uploader_component(tmp_path, monkeypatch, fail_paths=(str(bad),))
    cc._delete_after_upload = True

    out = asyncio.run(cc._upload({"paths": [str(ok), str(bad)]}))
    assert out["uploaded"] == [str(ok)]
    assert out["deleted"] == [str(ok)]
    assert [f["path"] for f in out["failed"]] == [str(bad)]
    assert not ok.exists()
    assert bad.exists()


def test_delete_after_upload_command_overrides_config(tmp_path, monkeypatch):
    path = tmp_path / "a.CR3"
    path.write_bytes(b"raw")
    cc = _uploader_component(tmp_path, monkeypatch)  # config default: keep

    out = asyncio.run(
        cc._upload({"paths": [str(path)], "delete_after_upload": True})
    )
    assert out["deleted"] == [str(path)]
    assert not path.exists()


def test_delete_removes_files_inside_output_dir(tmp_path):
    cc = _component(_FakeSource(None), output_dir=str(tmp_path))
    keep = tmp_path / "keep.CR3"
    drop = tmp_path / "drop.CR3"
    keep.write_bytes(b"raw")
    drop.write_bytes(b"raw")

    out = asyncio.run(cc.do_command({"delete": {"paths": [str(drop)]}}))["delete"]
    assert out["deleted"] == [str(drop)]
    assert out["count"] == 1
    assert not drop.exists()
    assert keep.exists()


def test_delete_missing_file_is_idempotent(tmp_path):
    cc = _component(_FakeSource(None), output_dir=str(tmp_path))
    gone = str(tmp_path / "already-gone.CR3")

    out = asyncio.run(cc.do_command({"delete": {"paths": [gone]}}))["delete"]
    assert out["deleted"] == []
    assert out["missing"] == [gone]
    assert out["failed"] == []


def test_delete_refuses_paths_outside_output_dir(tmp_path):
    """Absolute paths, `..` traversal, and symlinks out of output_dir are all
    refused - the command must not be able to reach the rest of the host."""
    images = tmp_path / "images"
    images.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"important")
    link = images / "sneaky.CR3"
    link.symlink_to(outside)
    cc = _component(_FakeSource(None), output_dir=str(images))

    out = asyncio.run(cc.do_command({"delete": {"paths": [
        str(outside),
        str(images / ".." / "secret.txt"),
        str(link),
    ]}}))["delete"]

    assert out["deleted"] == []
    assert len(out["failed"]) == 3
    assert all("outside output_dir" in f["error"] for f in out["failed"])
    assert outside.exists()


def test_delete_requires_output_dir(tmp_path):
    cc = _component(_FakeSource(None), output_dir=None)
    with pytest.raises(ValueError, match="output_dir"):
        asyncio.run(cc.do_command({"delete": {"paths": ["/anything"]}}))


def test_delete_requires_paths(tmp_path):
    cc = _component(_FakeSource(None), output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="paths"):
        asyncio.run(cc.do_command({"delete": {}}))


# ---------------------------------------------------------------------------
# Nines partner-API delivery: `upload` with a `sku` upserts the Nines product
# (external_id = SKU) and appends the set's delivery image; `nines_upload`
# sends exactly the listed files. Exercised against a recording fake of
# `_nines_request` - no HTTP.
# ---------------------------------------------------------------------------

import base64
import os

from models.color_correction import NinesAPIError


class _FakeNinesAPI:
    """Records every `_nines_request` call; scripted upsert/append responses.

    ``dead_item_ids`` 404 their appends (a product deleted server-side);
    ``append_error`` fails every append.
    """

    def __init__(self, item_id="ritem_1", append_error=None, dead_item_ids=()):
        self.item_id = item_id
        self.append_error = append_error
        self.dead_item_ids = dead_item_ids
        self.calls = []

    def __call__(self, method, path, body, timeout_s):
        self.calls.append((method, path, body, timeout_s))
        if path == "/api/v1/reference_items":
            return {"id": self.item_id, "external_id": body["external_id"],
                    "created": True, "updated": False, "images_count": 0}
        if path.endswith("/images"):
            rid = path.split("/")[-2]
            if rid in self.dead_item_ids:
                raise NinesAPIError(
                    f"Nines API POST {path} failed with 404: not found",
                    status=404,
                )
            if self.append_error is not None:
                raise self.append_error
            return {"id": rid, "added_count": len(body["images"]),
                    "images_count": len(body["images"]) + 2}
        raise AssertionError(f"unexpected Nines path {path}")


def _nines_component(tmp_path, monkeypatch, **fake_kwargs):
    """Uploader component with Nines configured and `_nines_request` faked."""
    cc = _uploader_component(tmp_path, monkeypatch)
    cc._nines_api_key = "nines_live_test"
    cc._nines_org_slug = "viam-org"
    fake = _FakeNinesAPI(**fake_kwargs)
    monkeypatch.setattr(cc, "_nines_request", fake)
    return cc, fake


def _shot_set(tmp_path, stem="IMG_0042"):
    """A capture set on disk: RAW master, 16-bit TIFF, JPEG, sidecar."""
    paths = []
    for n in (f"{stem}.cr3", f"{stem}_16.tif", f"{stem}.jpg", f"{stem}.json"):
        p = tmp_path / n
        p.write_bytes(f"bytes-of-{n}".encode())
        paths.append(str(p))
    return paths


def test_nines_pick_image_prefers_full_res_jpeg():
    pick = ColorCorrection._nines_pick_image
    files = ["/d/a.cr3", "/d/a_16.tif", "/d/a_16.png", "/d/a.png", "/d/a.jpg",
             "/d/a.json"]
    assert pick(files) == "/d/a.jpg"
    # Without a JPEG: 8-bit PNG beats the 16-bit variant.
    assert pick([p for p in files if not p.endswith(".jpg")]) == "/d/a.png"
    assert pick(["/d/a.cr3", "/d/a_16.png"]) == "/d/a_16.png"
    # RAW/TIFF/sidecar alone: nothing Nines accepts.
    assert pick(["/d/a.cr3", "/d/a_16.tif", "/d/a.json"]) is None
    assert pick([]) is None


def test_renamed_basename_swaps_stem_and_keeps_suffix():
    rename = ColorCorrection._renamed_basename
    assert rename("/d/IMG_0042_16.tif", "front", "IMG_0042") == "front_16.tif"
    assert rename("/d/IMG_0042.json", "front", "IMG_0042") == "front.json"
    # No operator name: the on-disk basename passes through.
    assert rename("/d/IMG_0042.jpg", None, None) == "IMG_0042.jpg"
    assert rename("/d/IMG_0042.jpg", "", None) == "IMG_0042.jpg"


def test_upload_with_sku_upserts_and_appends_jpeg(tmp_path, monkeypatch):
    paths = _shot_set(tmp_path)
    cc, fake = _nines_component(tmp_path, monkeypatch)

    out = asyncio.run(
        cc._upload({"paths": paths, "name": "front", "sku": "NWC-1042"})
    )

    # Upsert first: keyed by the SKU, small-call timeout, and never an
    # `images` field (that would replace an existing product's imagery).
    method, upsert_path, body, timeout = fake.calls[0]
    assert (method, upsert_path) == ("POST", "/api/v1/reference_items")
    assert body["external_id"] == "NWC-1042"
    assert body["name"] == "NWC-1042"
    assert body["shots_organization_slug"] == "viam-org"
    assert "images" not in body
    assert timeout == cc._upload_dial_timeout_s

    # Then one non-destructive append of just the delivery JPEG, renamed to
    # the operator's stem and tagged with it.
    method, append_path, body, timeout = fake.calls[1]
    assert append_path == "/api/v1/reference_items/ritem_1/images"
    assert body["shots_organization_slug"] == "viam-org"
    assert timeout == cc._upload_file_timeout_s
    (img,) = body["images"]
    assert img["filename"] == "front.jpg"
    assert img["content_type"] == "image/jpeg"
    assert base64.b64decode(img["data"]) == b"bytes-of-IMG_0042.jpg"
    assert img["tags"] == ["front"]

    assert out["nines"]["reference_item_id"] == "ritem_1"
    assert out["nines"]["external_id"] == "NWC-1042"
    assert out["nines"]["added_count"] == 1
    assert out["failed"] == []


def test_upload_same_sku_upserts_once(tmp_path, monkeypatch):
    """The reference-item id is cached per SKU, so a multi-shot submit hits
    the upsert endpoint once and the append endpoint per shot."""
    cc, fake = _nines_component(tmp_path, monkeypatch)

    asyncio.run(cc._upload({"paths": _shot_set(tmp_path, "IMG_0001"),
                            "sku": "NWC-1042"}))
    asyncio.run(cc._upload({"paths": _shot_set(tmp_path, "IMG_0002"),
                            "sku": "NWC-1042"}))

    upserts = [c for c in fake.calls if c[1] == "/api/v1/reference_items"]
    appends = [c for c in fake.calls if c[1].endswith("/images")]
    assert len(upserts) == 1
    assert len(appends) == 2


def test_upload_with_sku_unconfigured_reports_skipped(tmp_path, monkeypatch):
    """A machine without Nines config must not fail submits that carry a sku -
    the response says delivery was skipped and no HTTP happens."""
    paths = _shot_set(tmp_path)
    cc = _uploader_component(tmp_path, monkeypatch)  # no Nines config
    fake = _FakeNinesAPI()
    monkeypatch.setattr(cc, "_nines_request", fake)

    out = asyncio.run(cc._upload({"paths": paths, "sku": "NWC-1042"}))

    assert "not configured" in out["nines"]["skipped"]
    assert fake.calls == []
    assert out["failed"] == []


def test_upload_without_sku_never_touches_nines(tmp_path, monkeypatch):
    cc, fake = _nines_component(tmp_path, monkeypatch)

    out = asyncio.run(cc._upload({"paths": _shot_set(tmp_path)}))

    assert "nines" not in out
    assert fake.calls == []


def test_nines_failure_keeps_delivery_image_for_retry(tmp_path, monkeypatch):
    """`delete_after_upload` removes the archived set, but a failed Nines
    delivery keeps the JPEG on disk for retry - and is reported under `nines`,
    never as a per-file Viam failure."""
    paths = _shot_set(tmp_path)
    cc, fake = _nines_component(
        tmp_path, monkeypatch,
        append_error=NinesAPIError("Nines API append failed with 422: bad image",
                                   status=422),
    )
    cc._delete_after_upload = True

    out = asyncio.run(
        cc._upload({"paths": paths, "name": "front", "sku": "NWC-1042"})
    )

    assert "422" in out["nines"]["error"]
    assert out["failed"] == []
    jpg = next(p for p in paths if p.endswith(".jpg"))
    assert os.path.exists(jpg)  # kept for the retry
    assert jpg not in out["deleted"]
    for p in paths:
        if p != jpg:
            assert not os.path.exists(p)  # the rest archived and cleaned up


def test_stale_cached_item_id_reupserts_once(tmp_path, monkeypatch):
    """A cached reference-item id that 404s (product deleted on the Nines
    side) is dropped, re-upserted, and the append retried once."""
    paths = _shot_set(tmp_path)
    cc, fake = _nines_component(tmp_path, monkeypatch, item_id="ritem_new",
                                dead_item_ids=("ritem_dead",))
    # Cache is keyed by (org, SKU); the effective org is the configured slug.
    cc._nines_item_ids[("viam-org", "NWC-1042")] = "ritem_dead"

    out = asyncio.run(cc._upload({"paths": paths, "sku": "NWC-1042"}))

    assert [(m, p) for m, p, _, _ in fake.calls] == [
        ("POST", "/api/v1/reference_items/ritem_dead/images"),
        ("POST", "/api/v1/reference_items"),
        ("POST", "/api/v1/reference_items/ritem_new/images"),
    ]
    assert out["nines"]["reference_item_id"] == "ritem_new"
    assert cc._nines_item_ids[("viam-org", "NWC-1042")] == "ritem_new"


def test_nines_upload_command_validates(tmp_path, monkeypatch):
    cc, fake = _nines_component(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="sku"):
        asyncio.run(cc.do_command({"nines_upload": {"paths": ["/a.jpg"]}}))
    with pytest.raises(ValueError, match="paths"):
        asyncio.run(cc.do_command({"nines_upload": {"sku": "X"}}))
    with pytest.raises(ValueError, match="jpeg/png/webp/gif"):
        asyncio.run(cc.do_command(
            {"nines_upload": {"sku": "X", "paths": ["/a_16.tif"]}}
        ))
    assert fake.calls == []


def test_nines_upload_command_requires_config(tmp_path, monkeypatch):
    cc = _uploader_component(tmp_path, monkeypatch)  # no Nines config
    with pytest.raises(ValueError, match="not configured"):
        asyncio.run(cc.do_command(
            {"nines_upload": {"sku": "X", "paths": ["/a.jpg"]}}
        ))


def test_nines_upload_command_sends_listed_files(tmp_path, monkeypatch):
    """The manual command sends exactly the files given (no best-of-set
    picking), with shared tags and the optional product display name."""
    cc, fake = _nines_component(tmp_path, monkeypatch)
    front = tmp_path / "front.jpg"
    front.write_bytes(b"jjj")
    back = tmp_path / "back.png"
    back.write_bytes(b"ppp")

    out = asyncio.run(cc.do_command({"nines_upload": {
        "sku": "NWC-1042",
        "paths": [str(front), str(back)],
        "tags": ["on-model"],
        "product_name": "Northwood Chore Coat",
    }}))["nines_upload"]

    _, _, upsert_body, _ = fake.calls[0]
    assert upsert_body["name"] == "Northwood Chore Coat"
    _, _, append_body, _ = fake.calls[1]
    images = append_body["images"]
    assert [i["filename"] for i in images] == ["front.jpg", "back.png"]
    assert [i["content_type"] for i in images] == ["image/jpeg", "image/png"]
    assert all(i["tags"] == ["on-model"] for i in images)
    assert base64.b64decode(images[0]["data"]) == b"jjj"
    assert out["added_count"] == 2
    assert out["external_id"] == "NWC-1042"


def test_upload_request_slug_overrides_config_slug(tmp_path, monkeypatch):
    """A per-request `shots_organization_slug` targets that org instead of the
    configured one - both the upsert and the append carry the request slug."""
    cc, fake = _nines_component(tmp_path, monkeypatch)  # config slug "viam-org"

    out = asyncio.run(cc._upload({
        "paths": _shot_set(tmp_path),
        "sku": "NWC-1042",
        "shots_organization_slug": "buyer-org",
    }))

    _, upsert_path, upsert_body, _ = fake.calls[0]
    assert upsert_path == "/api/v1/reference_items"
    assert upsert_body["shots_organization_slug"] == "buyer-org"
    _, _, append_body, _ = fake.calls[1]
    assert append_body["shots_organization_slug"] == "buyer-org"
    assert out["nines"]["external_id"] == "NWC-1042"


def test_upload_caches_reference_item_per_org(tmp_path, monkeypatch):
    """The reference-item cache is keyed by (org, SKU): the same SKU delivered
    to two different orgs upserts once per org, so a cached id from one org can
    never deliver another org's shot to the wrong product."""
    cc, fake = _nines_component(tmp_path, monkeypatch)  # config slug "viam-org"

    asyncio.run(cc._upload({"paths": _shot_set(tmp_path, "IMG_0001"),
                            "sku": "NWC-1042",
                            "shots_organization_slug": "org-a"}))
    asyncio.run(cc._upload({"paths": _shot_set(tmp_path, "IMG_0002"),
                            "sku": "NWC-1042",
                            "shots_organization_slug": "org-b"}))

    upserts = [c for c in fake.calls if c[1] == "/api/v1/reference_items"]
    appends = [c for c in fake.calls if c[1].endswith("/images")]
    assert len(upserts) == 2  # one per org, not shared across orgs
    assert len(appends) == 2
    assert {b["shots_organization_slug"] for _, _, b, _ in upserts} == {"org-a",
                                                                        "org-b"}
    assert ("org-a", "NWC-1042") in cc._nines_item_ids
    assert ("org-b", "NWC-1042") in cc._nines_item_ids


def test_upload_with_request_slug_needs_only_api_key(tmp_path, monkeypatch):
    """The multi-org unlock: a machine configured with only an API key (no
    `nines_organization_slug`) still delivers when the webapp supplies the slug
    per request - it must not report `skipped`."""
    cc = _uploader_component(tmp_path, monkeypatch)  # no Nines config...
    cc._nines_api_key = "nines_live_test"            # ...except the API key
    assert cc._nines_org_slug is None
    fake = _FakeNinesAPI()
    monkeypatch.setattr(cc, "_nines_request", fake)

    out = asyncio.run(cc._upload({
        "paths": _shot_set(tmp_path),
        "sku": "NWC-1042",
        "shots_organization_slug": "buyer-org",
    }))

    assert "skipped" not in out["nines"]
    assert out["nines"]["reference_item_id"] == "ritem_1"
    _, _, upsert_body, _ = fake.calls[0]
    assert upsert_body["shots_organization_slug"] == "buyer-org"


def test_nines_upload_command_honors_request_slug(tmp_path, monkeypatch):
    """A manual `nines_upload` retry can name the org so it lands in the same
    org as the original submit, overriding the configured slug."""
    cc, fake = _nines_component(tmp_path, monkeypatch)  # config slug "viam-org"
    front = tmp_path / "front.jpg"
    front.write_bytes(b"jjj")

    asyncio.run(cc.do_command({"nines_upload": {
        "sku": "NWC-1042",
        "paths": [str(front)],
        "shots_organization_slug": "retry-org",
    }}))

    _, _, upsert_body, _ = fake.calls[0]
    assert upsert_body["shots_organization_slug"] == "retry-org"
    _, _, append_body, _ = fake.calls[1]
    assert append_body["shots_organization_slug"] == "retry-org"


# ---------------------------------------------------------------------------
# develop: crop + output_stem
# ---------------------------------------------------------------------------

def _write_still_sized(tmp_path, w, h, name="IMG_0042.PNG"):
    p = str(tmp_path / name)
    Image.fromarray(np.full((h, w, 3), 120, np.uint8)).save(p, format="PNG")
    return p


def test_parse_crop_accepts_mapping_and_sequence():
    from models.color_correction import ColorCorrection as CC

    assert CC._parse_crop({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}) == (0.1, 0.2, 0.3, 0.4)
    assert CC._parse_crop([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)


def test_parse_crop_normalizes_no_op_and_none():
    from models.color_correction import ColorCorrection as CC

    assert CC._parse_crop(None) is None
    # A full-frame rect is not a crop — skip the crop path entirely.
    assert CC._parse_crop({"x": 0, "y": 0, "w": 1, "h": 1}) is None


def test_parse_crop_rejects_malformed_rects():
    from models.color_correction import ColorCorrection as CC

    with pytest.raises(ValueError, match="missing"):
        CC._parse_crop({"x": 0.1, "y": 0.2, "w": 0.3})
    with pytest.raises(ValueError, match="exactly 4"):
        CC._parse_crop([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="must be positive"):
        CC._parse_crop({"x": 0.1, "y": 0.2, "w": 0.0, "h": 0.4})
    with pytest.raises(ValueError, match="within the frame"):
        CC._parse_crop({"x": 0.8, "y": 0.0, "w": 0.5, "h": 1.0})
    with pytest.raises(ValueError, match="x/y/w/h or a 4-element list"):
        CC._parse_crop("half")


def test_develop_crop_shrinks_the_exports(tmp_path):
    """A crop must change what lands on disk, not just the response — and it must
    leave the source file untouched."""
    out_dir = str(tmp_path / "out")
    cc = _component(_FakeSource(saved_path=None), output_dir=out_dir)
    cc._output_formats = ["jpeg"]
    p = _write_still_sized(tmp_path, 200, 100)
    before = open(p, "rb").read()

    out = asyncio.run(cc.do_command({
        "develop": {"path": p, "crop": {"x": 0.25, "y": 0.0, "w": 0.5, "h": 0.5}},
    }))
    exported = out["develop"]["exports"]["jpeg"]
    with Image.open(exported) as img:
        assert img.size == (100, 50)
    assert out["develop"]["crop"] == [0.25, 0.0, 0.5, 0.5]
    # Non-destructive: the master is byte-identical.
    assert open(p, "rb").read() == before


def test_develop_without_crop_exports_the_full_frame(tmp_path):
    cc = _component(_FakeSource(saved_path=None), output_dir=str(tmp_path / "out"))
    cc._output_formats = ["jpeg"]
    p = _write_still_sized(tmp_path, 200, 100)

    out = asyncio.run(cc.do_command({"develop": {"path": p}}))
    with Image.open(out["develop"]["exports"]["jpeg"]) as img:
        assert img.size == (200, 100)
    assert "crop" not in out["develop"]


def test_develop_output_stem_keeps_a_cropped_variant_off_the_master(tmp_path):
    """The whole point of `output_stem`: develop one RAW twice (uncropped master
    + cropped variant) and end up with two distinct sets of files on disk."""
    out_dir = str(tmp_path / "out")
    cc = _component(_FakeSource(saved_path=None), output_dir=out_dir)
    cc._output_formats = ["jpeg"]
    cc._write_sidecar = True
    p = _write_still_sized(tmp_path, 200, 100)

    master = asyncio.run(cc.do_command({"develop": {"path": p}}))["develop"]
    variant = asyncio.run(cc.do_command({
        "develop": {
            "path": p,
            "crop": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0},
            "output_stem": "IMG_0042_crop-2",
        },
    }))["develop"]

    assert master["exports"]["jpeg"] != variant["exports"]["jpeg"]
    assert master["sidecar"] != variant["sidecar"]
    assert os.path.basename(variant["exports"]["jpeg"]).startswith("IMG_0042_crop-2")
    # Both survived — the variant didn't clobber the master.
    with Image.open(master["exports"]["jpeg"]) as img:
        assert img.size == (200, 100)
    with Image.open(variant["exports"]["jpeg"]) as img:
        assert img.size == (100, 100)


def test_develop_sidecar_records_the_crop(tmp_path):
    import json

    cc = _component(_FakeSource(saved_path=None), output_dir=str(tmp_path / "out"))
    cc._output_formats = ["jpeg"]
    cc._write_sidecar = True
    p = _write_still_sized(tmp_path, 200, 100)

    out = asyncio.run(cc.do_command({
        "develop": {"path": p, "crop": [0.1, 0.2, 0.3, 0.4], "output_stem": "v2"},
    }))
    with open(out["develop"]["sidecar"]) as f:
        record = json.load(f)
    assert record["crop"] == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}

    # No crop -> recorded as null, so a sidecar always states the region.
    out2 = asyncio.run(cc.do_command({"develop": {"path": p}}))
    with open(out2["develop"]["sidecar"]) as f:
        assert json.load(f)["crop"] is None


def test_develop_output_stem_rejected_for_multiple_paths(tmp_path):
    cc = _component(_FakeSource(saved_path=None), output_dir=str(tmp_path / "out"))
    a = _write_still_sized(tmp_path, 20, 20, "a.PNG")
    b = _write_still_sized(tmp_path, 20, 20, "b.PNG")

    with pytest.raises(Exception, match="output_stem"):
        asyncio.run(cc.do_command({
            "develop": {"paths": [a, b], "output_stem": "shared"},
        }))


from io import BytesIO

# ---------------------------------------------------------------------------
# preview: cropped previews at full requested resolution
# ---------------------------------------------------------------------------

def test_preview_crop_fills_max_dim_instead_of_shrinking(tmp_path):
    """The bug this command exists for: a tight crop must not come back tiny.

    Cropping a 1024px preview to 10% leaves ~102px. Cropping a fresh decode and
    encoding to max_dim gives the full 1024 back."""
    cc = _component(_FakeSource(saved_path=None), output_dir=str(tmp_path / "out"))
    p = _write_still_sized(tmp_path, 4000, 4000)

    out = asyncio.run(cc.do_command({
        "preview": {"path": p, "crop": {"x": 0.45, "y": 0.45, "w": 0.1, "h": 0.1},
                    "max_dim": 1024},
    }))["preview"]

    raw = base64.b64decode(out["image_base64"])
    with Image.open(BytesIO(raw)) as img:
        # 10% of 4000 = 400px of source, and max_dim only ever downsizes, so we
        # get the source region at its own size — far more than 1024 * 0.1.
        assert max(img.size) == 400
    assert out["crop"] == [0.45, 0.45, 0.1, 0.1]


def test_preview_downsizes_a_large_crop_to_max_dim(tmp_path):
    cc = _component(_FakeSource(saved_path=None), output_dir=str(tmp_path / "out"))
    p = _write_still_sized(tmp_path, 4000, 3000)

    out = asyncio.run(cc.do_command({
        "preview": {"path": p, "crop": {"x": 0, "y": 0, "w": 1.0, "h": 0.9},
                    "max_dim": 800},
    }))["preview"]
    with Image.open(BytesIO(base64.b64decode(out["image_base64"]))) as img:
        assert max(img.size) == 800


def test_preview_writes_nothing_to_disk(tmp_path):
    """Safe to call on every crop adjustment: no exports, no sidecar, source intact."""
    out_dir = tmp_path / "out"
    cc = _component(_FakeSource(saved_path=None), output_dir=str(out_dir))
    cc._write_sidecar = True
    p = _write_still_sized(tmp_path, 400, 300)
    before = sorted(os.listdir(tmp_path))
    original = open(p, "rb").read()

    resp = asyncio.run(cc.do_command({"preview": {"path": p}}))["preview"]

    assert "exports" not in resp and "sidecar" not in resp
    assert sorted(os.listdir(tmp_path)) == before
    assert not out_dir.exists()
    assert open(p, "rb").read() == original


def test_preview_full_frame_needs_no_crop(tmp_path):
    cc = _component(_FakeSource(saved_path=None), output_dir=str(tmp_path / "out"))
    p = _write_still_sized(tmp_path, 400, 300)
    out = asyncio.run(cc.do_command({"preview": {"path": p, "max_dim": 200}}))["preview"]
    assert "crop" not in out
    assert out["mime_type"] == "image/jpeg"
    with Image.open(BytesIO(base64.b64decode(out["image_base64"]))) as img:
        assert img.size == (200, 150)


def test_preview_requires_a_path_and_positive_max_dim(tmp_path):
    cc = _component(_FakeSource(saved_path=None), output_dir=str(tmp_path / "out"))
    p = _write_still_sized(tmp_path, 100, 100)
    with pytest.raises(Exception, match="needs a `path`"):
        asyncio.run(cc.do_command({"preview": {}}))
    with pytest.raises(Exception, match="positive `max_dim`"):
        asyncio.run(cc.do_command({"preview": {"path": p, "max_dim": 0}}))


def test_half_size_suffices_picks_the_cheaper_decode_only_when_it_can(tmp_path):
    """Half size is 4x faster, so prefer it — but never at the cost of the
    resolution the caller asked for."""
    cc = _component(_FakeSource(saved_path=None))
    p = _write_still_sized(tmp_path, 8000, 6000)

    # Full frame: nothing to lose, always half.
    assert cc._half_size_suffices(p, None, 4096) is True
    # Half of a 50% crop is 2000px — clears a 1024 request.
    assert cc._half_size_suffices(p, (0.0, 0.0, 0.5, 0.5), 1024) is True
    # Half of a 10% crop is 400px — does not, so decode full.
    assert cc._half_size_suffices(p, (0.0, 0.0, 0.1, 0.1), 1024) is False
    # A missing file can't be probed: prefer quality over speed.
    assert cc._half_size_suffices(str(tmp_path / "nope.CR3"), (0, 0, 0.5, 0.5), 512) is False


def test_preview_honors_the_crop_region(tmp_path):
    """The returned pixels must be the requested region, not just the right size."""
    # Left half black, right half white.
    arr = np.zeros((200, 400, 3), np.uint8)
    arr[:, 200:] = 255
    p = str(tmp_path / "halves.PNG")
    Image.fromarray(arr).save(p, format="PNG")
    cc = _component(_FakeSource(saved_path=None), output_dir=str(tmp_path / "out"))

    left = asyncio.run(cc.do_command({
        "preview": {"path": p, "crop": {"x": 0, "y": 0, "w": 0.4, "h": 1.0}, "max_dim": 64},
    }))["preview"]
    right = asyncio.run(cc.do_command({
        "preview": {"path": p, "crop": {"x": 0.6, "y": 0, "w": 0.4, "h": 1.0}, "max_dim": 64},
    }))["preview"]

    def mean(res):
        with Image.open(BytesIO(base64.b64decode(res["image_base64"]))) as img:
            return np.asarray(img.convert("L"), dtype=float).mean()

    assert mean(left) < 40    # dark half
    assert mean(right) > 215  # light half
