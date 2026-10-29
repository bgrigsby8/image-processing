"""raw_focus: reading the lens's focus byte from a Sony ARW, the per-zoom
table that says what it should be, and color-correction's per-shot check."""

import asyncio
import os
import struct

import pytest

from models import color_correction as cc_mod
from models.raw_focus import (
    DEFAULT_FOCUS_TABLE,
    FocusTable,
    RawFocus,
    _DECIPHER,
    check_focus,
    read_raw_focus,
)

from test_color_correction import _FakeLogger, _FakeSource, _component, _write_still

#: A real rig ARW, when this checkout has one (not in git: 60 MB each).
REAL_ARW = os.path.join(os.path.dirname(__file__), "..", "images", "DSC00256.ARW")


def _encipher(data: bytes) -> bytes:
    return bytes(pow(b, 3, 249) if b < 249 else b for b in data)


def _ifd(e, entries, base):
    """Serialize one IFD placed at file offset ``base``; ``entries`` is
    [(tag, type, count, payload bytes)]. Returns (ifd bytes, data bytes)
    with the data area following the IFD directly."""
    head = struct.pack(e + "H", len(entries))
    data = b""
    data_at = base + 2 + 12 * len(entries) + 4
    body = b""
    for tag, typ, count, payload in entries:
        if len(payload) <= 4:
            value = payload.ljust(4, b"\0")
        else:
            value = struct.pack(e + "I", data_at + len(data))
            data += payload
        body += struct.pack(e + "HHI", tag, typ, count) + value
    return head + body + b"\0\0\0\0", data


def _fake_arw(path, *, focus=176, focal=35.0, model="ILCE-7RM5", order="<",
              sony_header=False, block_len=400):
    """A minimal TIFF shaped like a Sony ARW: IFD0 -> Exif IFD -> MakerNote
    -> enciphered 0x9402 block holding ``focus`` at byte 0x2d."""
    e = order
    block = bytearray(block_len)
    if block_len > 0x2D:
        block[0x2D] = focus
    model_b = model.encode() + b"\0"
    lens_b = b"FE PZ 16-35mm F4 G\0"
    # Layout: header | IFD0 + data | Exif IFD + data | MakerNote IFD + data
    ifd0_at = 8
    ifd0_len = 2 + 12 * 2 + 4 + len(model_b)
    exif_at = ifd0_at + ifd0_len
    exif_entries_n = 3
    exif_len = 2 + 12 * exif_entries_n + 4 + 8 + len(lens_b)
    note_at = exif_at + exif_len
    note_ifd_at = note_at + (12 if sony_header else 0)
    note_ifd, note_data = _ifd(
        e, [(0x9402, 7, len(block), _encipher(bytes(block)))], note_ifd_at
    )
    note = (b"SONY DSC \0\0\0" if sony_header else b"") + note_ifd + note_data
    exif_ifd, exif_data = _ifd(e, [
        (0x920A, 5, 1, struct.pack(e + "II", int(focal * 10), 10)),
        (0x927C, 7, len(note), b"\0" * 4),  # inline placeholder, fixed below
        (0xA434, 2, len(lens_b), lens_b),
    ], exif_at)
    # Point the MakerNote entry at note_at (its bytes follow the Exif data).
    entry = 2 + 12 * 1
    exif_ifd = (exif_ifd[:entry + 8] + struct.pack(e + "I", note_at)
                + exif_ifd[entry + 12:])
    ifd0, ifd0_data = _ifd(e, [
        (0x0110, 2, len(model_b), model_b),
        (0x8769, 4, 1, struct.pack(e + "I", exif_at)),
    ], ifd0_at)
    header = (b"II" if e == "<" else b"MM") + struct.pack(e + "HI", 42, ifd0_at)
    blob = header + ifd0 + ifd0_data + exif_ifd + exif_data + note
    assert len(header + ifd0 + ifd0_data) == exif_at
    assert len(header + ifd0 + ifd0_data + exif_ifd + exif_data) == note_at
    with open(path, "wb") as fh:
        fh.write(blob)
    return str(path)


def test_decipher_inverts_sonys_cipher():
    assert bytes(_DECIPHER[c] for c in _encipher(bytes(range(256)))) == bytes(range(256))


@pytest.mark.parametrize("order", ["<", ">"])
@pytest.mark.parametrize("sony_header", [False, True])
def test_reads_focus_byte_and_focal_length(tmp_path, order, sony_header):
    path = _fake_arw(tmp_path / "a.ARW", focus=176, focal=35.0, order=order,
                     sony_header=sony_header)
    raw = read_raw_focus(path)
    assert raw == RawFocus(176, 35.0, "ILCE-7RM5", "FE PZ 16-35mm F4 G")


def test_not_a_sony_raw_is_none(tmp_path):
    png = _write_still(tmp_path)
    assert read_raw_focus(png) is None
    assert read_raw_focus(_fake_arw(tmp_path / "c.ARW", model="DSC-RX100M7")) is None
    assert read_raw_focus(_fake_arw(tmp_path / "s.ARW", block_len=16)) is None


@pytest.mark.skipif(not os.path.exists(REAL_ARW), reason="no rig ARW in this checkout")
def test_real_arw_matches_exiftool():
    # exiftool -FocusPosition2 -FocalLength on this file: 153, 16.0 mm
    raw = read_raw_focus(REAL_ARW)
    assert (raw.focus_position2, raw.focal_length_mm) == (153, 16.0)


def test_table_interpolates_per_focal_length():
    table = FocusTable(DEFAULT_FOCUS_TABLE)
    assert table.expected(58085, 35.0) == 176
    assert table.expected(59395, 35.0) == pytest.approx(180.5)  # midway 58085..60705
    assert table.expected(-10, 35.0) == 114 and table.expected(70000, 35.0) == 255
    assert table.expected(39920, 16.0) == 161
    assert table.expected(39920, 35.2) == 144  # within FOCAL_MATCH_MM
    assert table.expected(39920, 24.0) is None  # no entry for this zoom
    assert table.expected(39920, None) is None


def test_table_from_config():
    assert FocusTable.from_config(None).table == FocusTable(DEFAULT_FOCUS_TABLE).table
    t = FocusTable.from_config({"50": [[60000, 200], [0, 100]]})
    assert t.expected(30000, 50) == 150
    for bad in ({}, [1, 2], {"50": [[0, 1]]}, {"x": [[0, 1], [1, 2]]}, {"50": [[0], [1]]}):
        with pytest.raises(ValueError):
            FocusTable.from_config(bad)


def test_check_focus_statuses():
    table = FocusTable(DEFAULT_FOCUS_TABLE)
    at = RawFocus(176, 35.0, "ILCE-7RM5", None)
    ok = check_focus(at, 58085, table, 3)
    assert ok["status"] == "ok" and ok["delta"] == 0
    # The rig's 2026-10-29 miss: asked for 50000, the RAW says 176 (58085).
    missed = check_focus(at, 50000, table, 3)
    assert missed["status"] == "missed" and missed["delta"] > 15
    assert check_focus(None, 58085, table, 3)["status"] == "unchecked"
    assert check_focus(at, None, table, 3)["status"] == "unchecked"
    other_zoom = check_focus(RawFocus(176, 24.0, None, None), 58085, table, 3)
    assert other_zoom["status"] == "unchecked" and "24.0 mm" in other_zoom["reason"]


class _FocusSource(_FakeSource):
    """A source that also answers the focus commands color-correction forwards."""

    async def do_command(self, command, *, timeout=None):
        if "set_focus_position" in command or command.get("command") == "set_focus_position":
            self.commands.append(command)
            return {"ok": True, "position": 50131, "target": 50000}
        if "focus_near_far" in command:
            self.commands.append(command)
            return {"ok": True}
        return await super().do_command(command, timeout=timeout)


def test_forwarded_focus_commands_set_and_clear_the_request():
    cc = _component(_FocusSource(None))
    asyncio.run(cc.do_command({"set_focus_position": {"position": 59142}}))
    assert cc._requested_focus == 59142
    asyncio.run(cc.do_command({"command": "set_focus_position", "position": 40000}))
    assert cc._requested_focus == 40000
    asyncio.run(cc.do_command({"focus_near_far": {"step": 3}}))
    assert cc._requested_focus is None


def test_deferred_capture_flags_a_focus_miss(tmp_path, monkeypatch):
    """The rig's real miss: set_focus_position(50000) said ok, the lens sat at
    58085. The RAW's byte (176) catches it and the capture result says so."""
    source = _FocusSource(_write_still(tmp_path))
    cc = _component(source, output_dir=str(tmp_path / "out"))
    cc.logger = _FakeLogger()
    monkeypatch.setattr(cc_mod, "read_raw_focus",
                        lambda path: RawFocus(176, 35.0, "ILCE-7RM5", None))

    async def shoot():
        await cc.do_command({"set_focus_position": {"position": 50000}})
        ticket = (await cc.do_command({"capture": {"defer": True}}))["capture"]
        # A later focus move must not change what this shot is checked against.
        await cc.do_command({"set_focus_position": {"position": 58085}})
        return (await cc.do_command(
            {"capture_result": {"id": ticket["capture_id"], "wait_sec": 30}}
        ))["capture_result"]

    check = asyncio.run(shoot())["focus_check"]
    assert check["status"] == "missed"
    assert check["requested_position"] == 50000 and check["focus_position2"] == 176
    assert any("out of focus" in m for m in cc.logger.messages("warning"))


def test_capture_reports_focus_check(tmp_path, monkeypatch):
    source = _FocusSource(_write_still(tmp_path))
    cc = _component(source, output_dir=str(tmp_path / "out"))
    cc._output_formats = []
    monkeypatch.setattr(cc_mod, "read_raw_focus",
                        lambda path: RawFocus(176, 35.0, "ILCE-7RM5", None))

    async def shoot():
        await cc.do_command({"set_focus_position": {"position": 58085}})
        return (await cc.do_command({"capture": {}}))["capture"]

    assert asyncio.run(shoot())["focus_check"]["status"] == "ok"


def test_validate_config_checks_focus_check_attributes():
    from test_color_correction import _component_config
    validate = cc_mod.ColorCorrection.validate_config
    validate(_component_config(raw_focus_table={"35": [[0, 114], [65535, 255]]},
                               raw_focus_tolerance=2))
    with pytest.raises(ValueError, match="raw_focus_table"):
        validate(_component_config(raw_focus_table={"35": [[0, 114]]}))
    with pytest.raises(ValueError, match="raw_focus_tolerance"):
        validate(_component_config(raw_focus_tolerance=-1))


def test_sweep_table_and_rows_show_the_raw_byte():
    from models.focus_sweep import format_table, frame_rows
    recs = [{"phase": "coarse", "target": 58000, "readback": 58085, "focus_position2": 176,
             "sharpness": {"mtf50": 0.2, "edges_used": 9, "edges_total": 9}}]
    header, line = format_table(recs).splitlines()
    assert "raw" in header and " 176 " in line
    assert frame_rows(recs)[0]["focus_position2"] == 176
