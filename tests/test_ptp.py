"""Tests for the PTP model's pure helpers and the USB auto-recovery logic
(no camera hardware needed)."""

import asyncio
import time

import pytest

from models import ptp as ptp_module
from models.ptp import (
    PTP,
    PTPSession,
    _device_gone,
    _is_image,
    _mime_for,
    _retry_once_on_device_gone,
)

gp = pytest.importorskip("gphoto2")


def test_is_image_accepts_stills_and_raws():
    assert _is_image("IMG_0042.JPG")
    assert _is_image("IMG_0042.cr3")
    assert _is_image("DSC_0001.NEF")
    assert not _is_image("MVI_0042.MP4")
    assert not _is_image("IMG_0042.JPG.tmp")


def test_mime_for_known_and_opaque_types():
    assert _mime_for("IMG_0042.JPG") == "image/jpeg"
    assert _mime_for("shot.png") == "image/png"
    # RAW is opaque bytes-on-the-wire, not a previewable image.
    assert _mime_for("IMG_0042.CR3") == "application/octet-stream"


# ---------------------------------------------------------------------------
# USB device-gone detection / auto-recovery
# ---------------------------------------------------------------------------

def test_device_gone_codes():
    assert _device_gone(gp.GPhoto2Error(gp.GP_ERROR_IO_USB_FIND))   # -52
    assert _device_gone(gp.GPhoto2Error(gp.GP_ERROR_IO_USB_CLAIM))  # -53
    assert _device_gone(gp.GPhoto2Error(gp.GP_ERROR_IO))            # -7
    # A generic capture failure (e.g. autofocus) is NOT a vanished device.
    assert not _device_gone(gp.GPhoto2Error(-1))
    assert not _device_gone(RuntimeError("not a gphoto error"))


class _FlakySession:
    """Stand-in for PTPSession: fails with a device-gone error until
    ``reconnect`` is called, mimicking a camera that slept and woke up."""

    def __init__(self, error_code=None, recoverable=True):
        self.error_code = error_code
        self.recoverable = recoverable
        self.calls = 0
        self.reconnects = 0

    def reconnect(self):
        self.reconnects += 1
        if self.recoverable:
            self.error_code = None

    @_retry_once_on_device_gone
    def op(self):
        self.calls += 1
        if self.error_code is not None:
            raise gp.GPhoto2Error(self.error_code)
        return "ok"


def test_retry_recovers_after_reconnect():
    session = _FlakySession(error_code=gp.GP_ERROR_IO_USB_FIND)
    assert session.op() == "ok"
    assert session.reconnects == 1
    assert session.calls == 2


def test_retry_gives_up_after_one_attempt():
    session = _FlakySession(error_code=gp.GP_ERROR_IO_USB_FIND, recoverable=False)
    with pytest.raises(gp.GPhoto2Error):
        session.op()
    assert session.reconnects == 1
    assert session.calls == 2  # original + exactly one retry, no loop


def test_retry_does_not_touch_other_errors():
    session = _FlakySession(error_code=-1)  # generic failure, not device-gone
    with pytest.raises(gp.GPhoto2Error):
        session.op()
    assert session.reconnects == 0
    assert session.calls == 1


def test_no_retry_when_healthy():
    session = _FlakySession()
    assert session.op() == "ok"
    assert session.calls == 1
    assert session.reconnects == 0


# ---------------------------------------------------------------------------
# capture(): retries the trigger once, and never re-fires a started capture
# ---------------------------------------------------------------------------

class _FakeCam:
    """Fake libgphoto2 Camera covering the calls capture() makes."""

    def __init__(self, trigger_errors):
        self.trigger_errors = list(trigger_errors)
        self.triggers = 0

    def trigger_capture(self):
        self.triggers += 1
        if self.trigger_errors:
            raise gp.GPhoto2Error(self.trigger_errors.pop(0))

    def wait_for_event(self, _timeout_ms):
        class _Added:
            folder = "/store/DCIM/100CANON"
            name = "IMG_0001.CR3"

        return gp.GP_EVENT_FILE_ADDED, _Added()


def _session_with_cam(cam):
    session = PTPSession.__new__(PTPSession)  # skip __init__'s gp probe
    session._camera = cam
    session.model_name = "Fake"
    session.port_path = "usb:000,000"
    session.reconnected = False

    def reconnect():
        session.reconnected = True

    session.reconnect = reconnect
    return session


def test_capture_retries_trigger_after_device_gone():
    cam = _FakeCam(trigger_errors=[gp.GP_ERROR_IO_USB_FIND])
    session = _session_with_cam(cam)
    path = session.capture(settle=0.1)
    assert path == "/store/DCIM/100CANON/IMG_0001.CR3"
    assert session.reconnected
    assert cam.triggers == 2


def test_capture_device_gone_twice_raises_usb_message():
    cam = _FakeCam(trigger_errors=[gp.GP_ERROR_IO_USB_FIND, gp.GP_ERROR_IO_USB_FIND])
    session = _session_with_cam(cam)
    with pytest.raises(RuntimeError, match="not reachable on USB"):
        session.capture(settle=0.1)
    assert cam.triggers == 2  # exactly one retry


def test_capture_generic_error_keeps_autofocus_hint_and_no_retry():
    cam = _FakeCam(trigger_errors=[-1])
    session = _session_with_cam(cam)
    with pytest.raises(RuntimeError, match="Check autofocus"):
        session.capture(settle=0.1)
    assert not session.reconnected
    assert cam.triggers == 1


# ---------------------------------------------------------------------------
# capture() card-scan fallback: identifies the NEW file by pre-trigger diff
# (dual-slot bodies broke newest-by-lexical-sort: every path on the second
# store sorts after every path on the first)
# ---------------------------------------------------------------------------

class _FakeCamNoEvent(_FakeCam):
    """A body that never reports FILE_ADDED (Canon after its first capture)."""

    def wait_for_event(self, _timeout_ms):
        return gp.GP_EVENT_TIMEOUT, None


class _StableInfo:
    """file_get_info result whose size never changes."""

    def __init__(self, size=100):
        class _F:
            pass

        self.file = _F()
        self.file.size = size


class _FakeCamCaptureComplete(_FakeCam):
    """A body that reports CAPTURE_COMPLETE but never FILE_ADDED."""

    def wait_for_event(self, _timeout_ms):
        return gp.GP_EVENT_CAPTURE_COMPLETE, None


def _session_with_listings(listings, sizes=None, cam_cls=_FakeCamNoEvent):
    """Session on a no-event body whose card contents step through `listings`
    (the last entry repeats). `sizes` optionally scripts file_get_info sizes."""
    cam = cam_cls(trigger_errors=[])
    if sizes is None:
        cam.file_get_info = lambda folder, name: _StableInfo()
    else:
        seq = list(sizes)
        cam.file_get_info = lambda folder, name: _StableInfo(
            seq.pop(0) if len(seq) > 1 else seq[0]
        )
    session = _session_with_cam(cam)
    session.refresh = lambda: None  # no cached filesystem on the fake body
    steps = [list(files) for files in listings]
    session.list_image_files = lambda: steps.pop(0) if len(steps) > 1 else steps[0]
    return session


def test_capture_fallback_picks_new_file_not_lexical_last():
    # The stale frame on store 2 sorts after the new frame on store 1; the old
    # newest-by-sort fallback returned it, the diff must not.
    old = ["/store_00010001/DCIM/9Q0A0001.CR3", "/store_00020001/DCIM/9Q0A0001.CR3"]
    new = "/store_00010001/DCIM/9Q0A0002.CR3"
    session = _session_with_listings([old, old + [new]])
    assert session.capture(settle=0.05) == new


def test_capture_fallback_prefers_raw_on_first_store():
    # One exposure can land as RAW+JPEG mirrored to both slots; prefer the RAW
    # on the first (fast) card.
    new = [
        "/store_00010001/DCIM/9Q0A0002.JPG",
        "/store_00010001/DCIM/9Q0A0002.CR3",
        "/store_00020001/DCIM/9Q0A0002.CR3",
    ]
    session = _session_with_listings([[], new])
    assert session.capture(settle=0.05) == "/store_00010001/DCIM/9Q0A0002.CR3"


def test_capture_scans_during_the_settle_window_instead_of_waiting_it_out():
    # Canon bodies stop emitting FILE_ADDED after the first capture of a
    # session, so on a no-event body the card scan has to *race* the event
    # wait. When the scan ran only after the event wait expired, every frame
    # after the first paid the whole settle window before looking at the card.
    old = ["/store_00010001/DCIM/9Q0A0001.CR3"]
    new = "/store_00010001/DCIM/9Q0A0002.CR3"
    session = _session_with_listings([old, old + [new]])
    start = time.monotonic()
    assert session.capture(settle=30.0) == new
    assert time.monotonic() - start < 5.0  # not the 30s ceiling


def test_capture_scans_as_soon_as_the_body_reports_capture_complete():
    # CAPTURE_COMPLETE means the write is done: it's both the earliest a scan
    # can succeed (an earlier one gets -110 busy) and the earliest it can find
    # anything, so it must pre-empt the head-start timer.
    old = ["/store_00010001/DCIM/9Q0A0001.CR3"]
    new = "/store_00010001/DCIM/9Q0A0002.CR3"
    session = _session_with_listings(
        [old, old + [new]], cam_cls=_FakeCamCaptureComplete
    )
    start = time.monotonic()
    assert session.capture(settle=30.0) == new
    assert time.monotonic() - start < ptp_module._SCAN_HEAD_START_SEC


def test_capture_stops_scanning_speculatively_once_the_body_reports_the_event():
    # A scan before CAPTURE_COMPLETE never finds anything on such a body (the
    # file isn't visible until the write lands) and costs real USB bus time
    # contending with that write - measured at 0.29-0.95s per scan on an R5.
    # So after the first capture teaches us the body reports it, later captures
    # must wait for the event rather than poll the card.
    first = ["/store_00010001/DCIM/9Q0A0001.CR3"]
    second = first + ["/store_00010001/DCIM/9Q0A0002.CR3"]
    third = second + ["/store_00010001/DCIM/9Q0A0003.CR3"]
    # snapshot, scan (capture 1), then snapshot, scan (capture 2)
    session = _session_with_listings(
        [first, second, second, third], cam_cls=_FakeCamCaptureComplete
    )
    scans = []
    listing = session.list_image_files
    session.list_image_files = lambda: (scans.append(1), listing())[1]

    assert session.capture(settle=30.0) == second[-1]
    assert session._reports_capture_complete
    before = len(scans)

    # Second capture: the event still drives it, and no timer scan precedes it.
    assert session.capture(settle=30.0) == third[-1]
    assert len(scans) - before == 2  # the pre-trigger snapshot and one real scan


def test_capture_survives_a_busy_rescan_instead_of_propagating_it():
    # A scan that lands while the body is still flushing raises -110; that's
    # "not ready", not a failed capture.
    old = ["/store_00010001/DCIM/9Q0A0001.CR3"]
    new = "/store_00010001/DCIM/9Q0A0002.CR3"
    session = _session_with_listings([old])
    calls = []

    def listing():
        calls.append(1)
        if len(calls) == 1:
            return old  # the pre-trigger snapshot
        if len(calls) == 2:
            raise gp.GPhoto2Error(gp.GP_ERROR_CAMERA_BUSY)
        return old + [new]

    session.list_image_files = listing
    assert session.capture(settle=30.0) == new


def test_capture_fallback_no_new_file_raises():
    files = ["/store_00010001/DCIM/9Q0A0001.CR3"]
    session = _session_with_listings([files])
    with pytest.raises(RuntimeError, match="no new image"):
        session.capture(settle=0.05)


# ---------------------------------------------------------------------------
# refresh(): a busy body is not a broken connection
# ---------------------------------------------------------------------------

class _BusyCam:
    """A camera whose first `busy_inits` re-inits report -110 (still writing)."""

    def __init__(self, busy_inits):
        self.busy_inits = busy_inits
        self.inits = 0

    def exit(self):
        pass

    def init(self):
        self.inits += 1
        if self.inits <= self.busy_inits:
            raise gp.GPhoto2Error(gp.GP_ERROR_CAMERA_BUSY)


def _session_tracking_reopen(cam, monkeypatch):
    monkeypatch.setattr(ptp_module, "_BUSY_RETRY_DELAY_SEC", 0.0)
    session = _session_with_cam(cam)
    reopened = []
    session.close = lambda: reopened.append("close")
    session.open = lambda: reopened.append("open")
    return session, reopened


def test_refresh_waits_out_a_busy_camera(monkeypatch):
    cam = _BusyCam(busy_inits=2)
    session, reopened = _session_tracking_reopen(cam, monkeypatch)
    session.refresh()
    assert cam.inits == 3  # two busy, then through
    assert not reopened


def test_refresh_raises_when_busy_rather_than_killing_the_session(monkeypatch):
    # -110 while the body flushes a frame is transient. The old code reopened
    # on any error, and reopening mid-write can't claim the interface (-53) -
    # which killed a live session and made every later call fail.
    cam = _BusyCam(busy_inits=99)
    session, reopened = _session_tracking_reopen(cam, monkeypatch)
    with pytest.raises(gp.GPhoto2Error):
        session.refresh()
    assert cam.inits == ptp_module._BUSY_RETRIES
    assert not reopened


def test_refresh_still_reopens_on_a_real_handshake_failure(monkeypatch):
    cam = _BusyCam(busy_inits=0)
    cam.init = lambda: (_ for _ in ()).throw(gp.GPhoto2Error(gp.GP_ERROR_IO_USB_FIND))
    session, reopened = _session_tracking_reopen(cam, monkeypatch)
    session.refresh()
    assert reopened == ["close", "open"]


def test_wait_size_stable_waits_for_settled_size():
    session = _session_with_listings([[]], sizes=[10, 50, 90, 90])
    session._wait_size_stable("/store/DCIM/IMG.CR3", timeout=5.0, interval=0.01)
    # the scripted sequence was consumed down to its stable tail
    assert session._cam.file_get_info("x", "y").file.size == 90


def test_wait_size_stable_drops_fs_cache_before_every_read():
    # libgphoto2 caches file info with the directory listing; without a
    # refresh per read a mid-write file reports the same stale size twice and
    # passes as "stable" (then downloads truncated - seen on the R5's SD slot).
    session = _session_with_listings([[]], sizes=[10, 50, 90, 90])
    refreshes = []
    session.refresh = lambda: refreshes.append(1)
    session._wait_size_stable("/store/DCIM/IMG.CR3", timeout=5.0, interval=0.01)
    assert len(refreshes) == 4  # one cache drop per size read


def test_wait_size_stable_skips_when_info_unavailable():
    session = _session_with_listings([[]])
    session._cam.file_get_info = None  # not callable -> TypeError inside
    # must swallow the error and return rather than raise
    session._wait_size_stable("/store/DCIM/IMG.CR3", timeout=0.2, interval=0.01)


# ---------------------------------------------------------------------------
# `trigger` DoCommand: fires the shutter, returns the on-camera path, and
# never downloads (the deferred-pipeline handoff)
# ---------------------------------------------------------------------------

class _TriggerOnlySession:
    """Fake PTPSession that fails loudly if anything tries to download."""

    def __init__(self):
        self.captures = 0

    def capture(self, settle):
        self.captures += 1
        return "/store/DCIM/100CANON/IMG_0042.CR3"

    def read_file(self, path):
        raise AssertionError("trigger must not download the file")


def _ptp_component(session):
    ptp = PTP("test-ptp")
    ptp._session = session
    ptp._lock = asyncio.Lock()
    ptp._capture_settle = 0.0
    ptp._download_dir = None
    ptp._delete_after_download = False
    ptp._downloaded = set()
    return ptp


def test_trigger_returns_camera_path_without_download():
    session = _TriggerOnlySession()
    ptp = _ptp_component(session)

    resp = asyncio.run(ptp.do_command({"trigger": {}}))

    out = resp["trigger"]
    assert out["path"] == "/store/DCIM/100CANON/IMG_0042.CR3"
    assert out["name"] == "IMG_0042.CR3"
    assert isinstance(out["mime_type"], str)
    assert "saved_to" not in out  # nothing was downloaded or written
    assert session.captures == 1


# ---------------------------------------------------------------------------
# cleanup(): clears the local download_dir without touching the camera card
# ---------------------------------------------------------------------------

def test_cleanup_removes_files_in_download_dir(tmp_path):
    (tmp_path / "IMG_0001.CR3").write_bytes(b"a")
    (tmp_path / "IMG_0002.JPG").write_bytes(b"b")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.JPG").write_bytes(b"c")  # subdirectory is left alone

    ptp = _ptp_component(_TriggerOnlySession())
    ptp._download_dir = str(tmp_path)

    resp = asyncio.run(ptp.do_command({"cleanup": {}}))

    out = resp["cleanup"]
    assert out["count"] == 2
    assert set(out["removed"]) == {"IMG_0001.CR3", "IMG_0002.JPG"}
    assert out["dry_run"] is False
    assert not (tmp_path / "IMG_0001.CR3").exists()
    assert not (tmp_path / "IMG_0002.JPG").exists()
    assert (sub / "nested.JPG").exists()  # subdir contents untouched


def test_cleanup_dry_run_reports_without_deleting(tmp_path):
    (tmp_path / "IMG_0001.CR3").write_bytes(b"a")

    ptp = _ptp_component(_TriggerOnlySession())
    ptp._download_dir = str(tmp_path)

    resp = asyncio.run(ptp.do_command({"cleanup": {"dry_run": True}}))

    out = resp["cleanup"]
    assert out["count"] == 1
    assert out["removed"] == ["IMG_0001.CR3"]
    assert out["dry_run"] is True
    assert (tmp_path / "IMG_0001.CR3").exists()  # nothing actually deleted


def test_cleanup_requires_download_dir():
    ptp = _ptp_component(_TriggerOnlySession())
    ptp._download_dir = None

    with pytest.raises(ValueError, match="download_dir"):
        asyncio.run(ptp.do_command({"cleanup": {}}))


# ---------------------------------------------------------------------------
# list_widgets(): walks the camera's config tree for focus discovery, with an
# opt-in live-view toggle that exposes the step-drive widgets
# ---------------------------------------------------------------------------

class _FakeWidget:
    """Stand-in for a libgphoto2 CameraWidget covering the accessors the walk
    uses. ``get_value`` raises when ``raises=True`` to exercise skip-on-error."""

    def __init__(
        self, name, wtype, label=None, value=None, readonly=False,
        choices=None, rng=None, children=None, raises=False,
    ):
        self._name = name
        self._type = wtype
        self._label = name if label is None else label
        self._value = value
        self._readonly = readonly
        self._choices = choices
        self._range = rng
        self._children = children or []
        self._raises = raises
        self.set_values = []  # history of set_value() calls, for assertions

    def get_type(self):
        return self._type

    def get_name(self):
        return self._name

    def get_label(self):
        return self._label

    def get_readonly(self):
        return self._readonly

    def get_value(self):
        if self._raises:
            raise gp.GPhoto2Error(-1)
        return self._value

    def set_value(self, v):
        self._value = v
        self.set_values.append(v)

    def count_choices(self):
        return len(self._choices or [])

    def get_choice(self, i):
        return self._choices[i]

    def get_range(self):
        return self._range

    def get_children(self):
        return list(self._children)

    def get_child_by_name(self, name):
        # libgphoto2 searches descendants, not just direct children.
        found = self._find(name)
        if found is None:
            raise gp.GPhoto2Error(gp.GP_ERROR)
        return found

    def _find(self, name):
        if self._name == name:
            return self
        for child in self._children:
            hit = child._find(name)
            if hit is not None:
                return hit
        return None


class _FakeCamForConfig:
    """Fake Camera that hands out a (possibly evolving) config tree.

    ``get_config()`` returns successive trees by call index (clamped to the
    last), so a test can model the tree changing once live view turns on.
    ``set_config`` is recorded, and can be made to fail on chosen call numbers
    to exercise the best-effort restore.
    """

    def __init__(self, trees, fail_set_config=()):
        self._trees = trees
        self._fail_set_config = set(fail_set_config)
        self.get_config_calls = 0
        self.set_config_calls = 0

    def get_config(self):
        idx = min(self.get_config_calls, len(self._trees) - 1)
        self.get_config_calls += 1
        return self._trees[idx]

    def set_config(self, config):
        self.set_config_calls += 1
        if self.set_config_calls in self._fail_set_config:
            raise gp.GPhoto2Error(gp.GP_ERROR_IO_USB_FIND)


def _radio(name, choices, value):
    return _FakeWidget(name, gp.GP_WIDGET_RADIO, choices=choices, value=value)


def _root(children):
    return _FakeWidget("main", gp.GP_WIDGET_WINDOW, children=children)


def test_list_widgets_reports_all_widgets_with_focus_flag():
    tree = _root([
        _FakeWidget("capture", gp.GP_WIDGET_SECTION, children=[
            _radio("manualfocusdrive", ["None", "Near 1", "Far 1"], "None"),
            _radio("focusmode", ["One Shot", "Manual"], "Manual"),
            _FakeWidget("eosviewfinder", gp.GP_WIDGET_TOGGLE, value=0),
            _radio("eosremoterelease", ["None", "Press Half", "Press Full"], "None"),
            _radio("iso", ["100", "200"], "100"),
            _FakeWidget("lightmeter", gp.GP_WIDGET_RANGE, value=0.0, rng=(-3.0, 3.0, 0.5)),
        ]),
    ])
    session = _session_with_cam(_FakeCamForConfig([tree]))

    widgets = session.list_widgets()
    by_name = {w["name"]: w for w in widgets}

    # Every leaf is reported - sections/windows are not, but iso is.
    assert set(by_name) == {
        "manualfocusdrive", "focusmode", "eosviewfinder",
        "eosremoterelease", "iso", "lightmeter",
    }
    # focus_relevant flags: focus widgets + name-based extras True, iso False.
    assert by_name["manualfocusdrive"]["focus_relevant"] is True
    assert by_name["focusmode"]["focus_relevant"] is True
    assert by_name["eosviewfinder"]["focus_relevant"] is True
    assert by_name["eosremoterelease"]["focus_relevant"] is True
    assert by_name["iso"]["focus_relevant"] is False
    # Choices and range are surfaced for calibration.
    assert by_name["manualfocusdrive"]["choices"] == ["None", "Near 1", "Far 1"]
    assert by_name["manualfocusdrive"]["type"] == "radio"
    assert by_name["lightmeter"]["range"] == {"min": -3.0, "max": 3.0, "step": 0.5}


def test_list_widgets_skips_unreadable_widget():
    tree = _root([
        _radio("focusmode", ["One Shot", "Manual"], "Manual"),
        _FakeWidget("brokenfocus", gp.GP_WIDGET_TEXT, raises=True),
    ])
    session = _session_with_cam(_FakeCamForConfig([tree]))

    widgets = session.list_widgets()
    names = {w["name"] for w in widgets}

    assert names == {"focusmode"}  # the raising widget is skipped, not fatal


def test_list_widgets_live_view_toggles_and_restores(monkeypatch):
    monkeypatch.setattr("models.ptp._LIVE_VIEW_SETTLE_SEC", 0.0)
    # The viewfinder widget is shared across trees so its set_value history
    # captures both the toggle-on and the restore.
    vf = _FakeWidget("eosviewfinder", gp.GP_WIDGET_TOGGLE, value=0)
    before = _root([vf, _radio("focusmode", ["Manual"], "Manual")])
    after = _root([
        vf,
        _radio("focusmode", ["Manual"], "Manual"),
        _radio("manualfocusdrive", ["None", "Near 1"], "None"),
    ])
    cam = _FakeCamForConfig([before, after, after])
    session = _session_with_cam(cam)

    widgets = session.list_widgets(live_view=True)
    names = {w["name"] for w in widgets}

    # Re-fetch happened: manualfocusdrive only exists in the post-toggle tree.
    assert "manualfocusdrive" in names
    # Toggled on, then restored to the original value.
    assert vf.set_values == [1, 0]
    assert cam.set_config_calls == 2
    assert cam.get_config_calls == 3  # initial, re-fetch, restore


def test_list_widgets_restore_is_best_effort(monkeypatch):
    monkeypatch.setattr("models.ptp._LIVE_VIEW_SETTLE_SEC", 0.0)
    vf = _FakeWidget("eosviewfinder", gp.GP_WIDGET_TOGGLE, value=0)
    before = _root([vf])
    after = _root([vf, _radio("manualfocusdrive", ["None"], "None")])
    # Fail the restore set_config (the 2nd one) with a device-gone error.
    cam = _FakeCamForConfig([before, after, after], fail_set_config={2})
    session = _session_with_cam(cam)

    widgets = session.list_widgets(live_view=True)
    names = {w["name"] for w in widgets}

    # The discovery result still comes back; the restore failure is swallowed
    # and does NOT propagate (which would otherwise trip the device-gone retry).
    assert "manualfocusdrive" in names
    assert session.reconnected is False


def test_do_command_list_widgets():
    class _ListWidgetsSession:
        def __init__(self, widgets):
            self._widgets = widgets
            self.calls = []

        def list_widgets(self, live_view=False):
            self.calls.append(live_view)
            return self._widgets

    session = _ListWidgetsSession([
        {"name": "manualfocusdrive", "focus_relevant": True, "value": (1, 2)},
        {"name": "iso", "focus_relevant": False, "value": "100"},
    ])
    ptp = _ptp_component(session)

    resp = asyncio.run(ptp.do_command({"list_widgets": {"live_view": True}}))
    out = resp["list_widgets"]

    assert out["count"] == 2
    assert out["focus_count"] == 1
    assert session.calls == [True]  # live_view forwarded
    # Non-primitive widget values are coerced to gRPC-safe types (tuple->list).
    by_name = {w["name"]: w for w in out["widgets"]}
    assert by_name["manualfocusdrive"]["value"] == [1, 2]
