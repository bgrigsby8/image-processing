"""
models/raw_focus.py
-------------------
Read where the lens was really focused straight from a Sony ARW, and check it
against the focus position the rig asked for.

Every Sony RAW carries the lens's focus position at the moment of exposure in
a private MakerNote field that ExifTool calls ``FocusPosition2``: one byte,
0-255, 255 = infinity, larger = farther. It is written by the camera, so it
is true even when the camera's follow-focus read-back over USB is stale (see
sony-remote's "the read-back is late"). That makes it a free per-photo focus
check: a shot whose byte is far from what its focus setting should give was
taken somewhere else.

Where it lives: MakerNote tag 0x9402, byte 0x2d. The 0x94xx blocks are
scrambled with a byte substitution - the camera stores ``b**3 % 249`` for each
byte ``b`` below 249 (bytes 249-255 are stored as-is) - so reading it needs
the inverse table, built below. Layout and cipher are ExifTool's (Sony.pm,
Tag9402 and Decipher); this reads only that one byte, with the standard
library, so the rig needs neither ExifTool nor a MakerNote library.

The byte is **not** in sony-remote's follow-focus units (0-65535), and its
scale depends on the zoom. ``FocusTable`` maps one to the other per focal
length from measured pairs; the default is the rig's FE PZ 16-35mm F4 G,
measured 2026-10-29 (follow-focus position read twice before each shot ->
the byte in that shot's RAW). Near the far end one byte step is ~300 units;
near the close end it is ~1500.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Mapping, Optional, Sequence, Tuple

#: Decipher table for Sony's 0x94xx blocks: stored byte -> real byte. Cubing
#: is a bijection mod 249 (= 3 * 83, and gcd(3, 82) = 1), so this is exact.
_DECIPHER = bytes(
    {pow(b, 3, 249): b for b in range(249)}[c] if c < 249 else c for c in range(256)
)

_TAG_EXIF_IFD = 0x8769
_TAG_MODEL = 0x0110
_TAG_MAKERNOTE = 0x927C
_TAG_FOCAL_LENGTH = 0x920A
_TAG_LENS_MODEL = 0xA434
_TAG_SONY_9402 = 0x9402
_FOCUS_POSITION2_OFFSET = 0x2D

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}

#: Measured on the rig (FE PZ 16-35mm F4 G, ILCE-7RM5): focal length (mm) ->
#: [(follow-focus position, FocusPosition2), ...].
DEFAULT_FOCUS_TABLE: Dict[float, List[Tuple[int, int]]] = {
    16.0: [(0, 132), (19548, 144), (39625, 160), (39920, 161), (49808, 173),
           (60151, 200), (65535, 255)],
    35.0: [(0, 114), (19948, 127), (39920, 144), (50310, 158), (58085, 176),
           (60705, 185), (65535, 255)],
}


#: Byte steps a shot may sit from its expected value and still pass. The
#: module's own tolerance (400 units) is ~1.5 steps near the far end, and the
#: table interpolates linearly over a curve, so allow a little more.
DEFAULT_RAW_FOCUS_TOLERANCE = 3.0


@dataclass(frozen=True)
class RawFocus:
    """What one RAW says about its own focus."""

    focus_position2: int
    focal_length_mm: Optional[float]
    model: Optional[str]
    lens: Optional[str]


class _Tiff:
    """Just enough TIFF walking to reach one MakerNote byte."""

    def __init__(self, fh: BinaryIO):
        self.fh = fh
        order = fh.read(2)
        if order == b"II":
            self.e = "<"
        elif order == b"MM":
            self.e = ">"
        else:
            raise ValueError("not a TIFF-based RAW")
        (magic,) = struct.unpack(self.e + "H", fh.read(2))
        if magic != 42:
            raise ValueError("not a TIFF-based RAW")
        (self.ifd0,) = struct.unpack(self.e + "I", fh.read(4))

    def read(self, offset: int, n: int) -> bytes:
        self.fh.seek(offset)
        data = self.fh.read(n)
        if len(data) != n:
            raise ValueError("truncated file")
        return data

    def entries(self, offset: int) -> Dict[int, Tuple[int, int, int]]:
        """IFD at ``offset`` -> {tag: (type, count, offset of the value)}."""
        (n,) = struct.unpack(self.e + "H", self.read(offset, 2))
        if n > 1000:
            raise ValueError("implausible IFD")
        out = {}
        raw = self.read(offset + 2, 12 * n)
        for i in range(n):
            tag, typ, count = struct.unpack_from(self.e + "HHI", raw, 12 * i)
            size = _TYPE_SIZES.get(typ, 1) * count
            if size <= 4:
                where = offset + 2 + 12 * i + 8
            else:
                (where,) = struct.unpack_from(self.e + "I", raw, 12 * i + 8)
            out[tag] = (typ, count, where)
        return out

    def ascii(self, entry: Optional[Tuple[int, int, int]]) -> Optional[str]:
        if entry is None:
            return None
        _, count, where = entry
        return self.read(where, count).split(b"\0", 1)[0].decode("ascii", "replace").strip() or None

    def rational(self, entry: Optional[Tuple[int, int, int]]) -> Optional[float]:
        if entry is None or entry[0] != 5:
            return None
        num, den = struct.unpack(self.e + "II", self.read(entry[2], 8))
        return num / den if den else None


def read_raw_focus(path: str) -> Optional[RawFocus]:
    """The lens's focus byte (and focal length) from a Sony ARW, or None
    when the file isn't a Sony RAW or doesn't carry the field. Raises only
    for a file that can't be opened."""
    with open(path, "rb") as fh:
        try:
            t = _Tiff(fh)
            ifd0 = t.entries(t.ifd0)
            model = t.ascii(ifd0.get(_TAG_MODEL))
            if model is None or model.startswith(("DSC-", "Stellar")):
                return None  # compacts keep this byte elsewhere, or not at all
            exif_entry = ifd0.get(_TAG_EXIF_IFD)
            if exif_entry is None:
                return None
            (exif_off,) = struct.unpack(t.e + "I", t.read(exif_entry[2], 4))
            exif = t.entries(exif_off)
            note = exif.get(_TAG_MAKERNOTE)
            if note is None:
                return None
            note_off = note[2]
            # Some bodies prefix the MakerNote IFD with "SONY DSC \0\0\0".
            if t.read(note_off, 4) == b"SONY":
                note_off += 12
            block = t.entries(note_off).get(_TAG_SONY_9402)
            if block is None or block[1] <= _FOCUS_POSITION2_OFFSET:
                return None
            stored = t.read(block[2] + _FOCUS_POSITION2_OFFSET, 1)[0]
            return RawFocus(
                focus_position2=_DECIPHER[stored],
                focal_length_mm=t.rational(exif.get(_TAG_FOCAL_LENGTH)),
                model=model,
                lens=t.ascii(exif.get(_TAG_LENS_MODEL)),
            )
        except (ValueError, struct.error):
            return None


class FocusTable:
    """Follow-focus position -> expected ``FocusPosition2``, per focal length,
    by linear interpolation between measured pairs."""

    #: How close (mm) a shot's focal length must be to a table entry to use it.
    FOCAL_MATCH_MM = 0.5

    def __init__(self, table: Mapping[float, Sequence[Tuple[int, int]]]):
        self.table: Dict[float, List[Tuple[int, int]]] = {}
        for focal, pairs in table.items():
            pts = sorted((int(p), int(v)) for p, v in pairs)
            if len(pts) < 2:
                raise ValueError(f"focus table for {focal} mm needs at least two points")
            self.table[float(focal)] = pts

    @classmethod
    def from_config(cls, value: Any) -> "FocusTable":
        """``{"35": [[position, focus_position2], ...], ...}`` -> table; None
        means the built-in rig table."""
        if value is None:
            return cls(DEFAULT_FOCUS_TABLE)
        if not isinstance(value, Mapping) or not value:
            raise ValueError(
                "`raw_focus_table` must map a focal length in mm to a list of "
                "[follow-focus position, FocusPosition2] pairs"
            )
        try:
            return cls({
                float(k): [(int(p[0]), int(p[1])) for p in v] for k, v in value.items()
            })
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"`raw_focus_table` is malformed: {exc}") from exc

    def pairs_for(self, focal_length_mm: Optional[float]) -> Optional[List[Tuple[int, int]]]:
        if focal_length_mm is None:
            return None
        focal = min(self.table, key=lambda f: abs(f - focal_length_mm))
        if abs(focal - focal_length_mm) > self.FOCAL_MATCH_MM:
            return None
        return self.table[focal]

    def expected(self, position: float, focal_length_mm: Optional[float]) -> Optional[float]:
        pts = self.pairs_for(focal_length_mm)
        if pts is None:
            return None
        if position <= pts[0][0]:
            return float(pts[0][1])
        for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
            if position <= p1:
                return v0 + (v1 - v0) * (position - p0) / (p1 - p0) if p1 != p0 else float(v1)
        return float(pts[-1][1])


def check_focus(
    raw: Optional[RawFocus],
    requested: Optional[int],
    table: FocusTable,
    tolerance: float,
) -> Dict[str, Any]:
    """
    Compare a shot's RAW focus byte with what its requested focus position
    should give. ``status`` is ``"ok"``, ``"missed"`` (more than ``tolerance``
    byte steps off) or ``"unchecked"`` (with a ``reason``).
    """
    out: Dict[str, Any] = {"requested_position": requested}
    if raw is None:
        return {**out, "status": "unchecked", "reason": "no Sony focus data in this file"}
    out["focus_position2"] = raw.focus_position2
    out["focal_length_mm"] = raw.focal_length_mm
    if requested is None:
        return {**out, "status": "unchecked",
                "reason": "no set_focus_position since the lens last moved another way"}
    expected = table.expected(requested, raw.focal_length_mm)
    if expected is None:
        return {**out, "status": "unchecked",
                "reason": f"no raw_focus_table entry for {raw.focal_length_mm} mm"}
    delta = raw.focus_position2 - expected
    out.update(
        expected_focus_position2=round(expected, 1),
        delta=round(delta, 1),
        status="ok" if abs(delta) <= tolerance else "missed",
    )
    return out
