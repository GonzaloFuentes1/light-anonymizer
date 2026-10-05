"""Helpers shared by the engines: file type by content, safe output names, errors, geometry, timing."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif"}
IMAGE_SAVE_OPTIONS = {
    "JPEG": {"quality": 92},
    "WEBP": {"quality": 92},
    "PNG": {},
    "TIFF": {"compression": "tiff_deflate"},
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
# PDF pages are rendered at this resolution for OCR, faces and QR codes (``pdf.raster_zones``).
OCR_DPI = 200
# PDF pages with less text than this are treated as scanned: OCR and faces over the whole page.
SCANNED_MAX_CHARS = 50

log = logging.getLogger(__name__)


class Cancelled(Exception):
    """The user cancelled the analysis."""


class FileError(Exception):
    """The file cannot be processed; ``code`` is a key of ``model.ERROR_MESSAGES``."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sniff(path: str | Path) -> str:
    """Real type of a file by its content: pdf | image | empty | heic | format."""
    with open(path, "rb") as fh:
        head = fh.read(12)
    if not head:
        return "empty"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith((b"\xff\xd8\xff", b"\x89PNG", b"II*\x00", b"MM\x00*")) or (
        head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    ):
        return "image"
    if is_heic(head):
        return "heic"
    return "format"


# Major brands of HEIC/HEIF photos (iPhones and many Android phones). They are not supported (D5):
# the only reader without GPL code is LGPL, and phones and Windows convert them to JPG.
HEIC_BRANDS = frozenset({b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs", b"mif1", b"msf1"})
HEIC_SUFFIXES = frozenset({".heic", ".heif", ".hif"})


def is_heic(head: bytes) -> bool:
    """The first 12 bytes of a file are those of a HEIC/HEIF image (an ISO-BMFF "ftyp" box)."""
    return head[4:8] == b"ftyp" and head[8:12] in HEIC_BRANDS


def unique_path(folder: Path, name: str) -> Path:
    """``folder/name``, or ``name (2)``, ``name (3)``... when it already exists: never overwrite."""
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate = folder / name
    n = 2
    while candidate.exists():
        candidate = folder / f"{stem} ({n}){suffix}"
        n += 1
    return candidate


def publish(staged: Path, folder: Path, name: str) -> Path:
    """Copies ``staged`` into ``folder`` under ``name`` (or ``name (2)``...) and returns its path.

    The copy is written under a hidden temporary name and only then given its final name, without
    ever replacing an existing file: if the app is closed in the middle, no half-written file is
    left with a name that looks finished (at most a hidden ``.partial`` file).
    """
    partial = folder / f".anonimizador_{uuid.uuid4().hex[:12]}.partial"
    try:
        shutil.copyfile(staged, partial)
        while True:
            output = unique_path(folder, name)
            try:
                _rename_no_replace(partial, output)
            except FileExistsError:  # created by someone else in the meantime: next free name
                continue
            return output
    finally:
        partial.unlink(missing_ok=True)


def _rename_no_replace(source: Path, target: Path) -> None:
    if os.name == "nt":
        os.rename(source, target)  # raises FileExistsError when the target exists
        return
    try:
        os.link(source, target)  # raises FileExistsError when the target exists
    except FileExistsError:
        raise
    except OSError:  # file systems without hard links
        if target.exists():
            raise FileExistsError(target) from None
        os.rename(source, target)
        return
    source.unlink()


def bbox_of(polygon) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def rect_polygon(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class Zone(NamedTuple):
    """A detection over a raster: ``polygon`` is an Nx2 array in pixels of that raster."""

    type: str
    polygon: Any
    text: str
    detector: str
    score: float
    doubt: str | None = None
    optional: bool = False  # D12: it only covers URLs that are not personal


# ---------------------------------------------------------------------------
# Time of each analysis stage
# ---------------------------------------------------------------------------

_current = threading.local()


class StageClock:
    """Working time of each stage of one analysis, in seconds.

    Stages: "text" (text layer, patterns, names and context rules), "render" (a page rendered or
    an image decoded for OCR, faces or QR), "ocr", "faces" and "qr". While a clock is running in a
    thread (``running``), ``stage`` adds the time of a block to its stage. Time spent waiting for
    a lock shared with another analysis (``waiting_for``: OCR, faces, PyMuPDF) is left out of the
    stage that waited. Two analyses still share the processor, so a stage can take longer while
    another file is analyzed at the same time; that is not subtracted.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.waited = 0.0

    @contextmanager
    def running(self) -> Iterator[StageClock]:
        previous = getattr(_current, "clock", None)
        _current.clock = self
        try:
            yield self
        finally:
            _current.clock = previous

    def rounded(self) -> dict[str, float]:
        return {name: round(value, 3) for name, value in self.seconds.items()}


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Counts the block in stage ``name`` of the clock running in this thread (if any). Not nested."""
    clock = getattr(_current, "clock", None)
    if clock is None:
        yield
        return
    started, waited = time.perf_counter(), clock.waited
    try:
        yield
    finally:
        spent = time.perf_counter() - started - (clock.waited - waited)
        clock.seconds[name] = clock.seconds.get(name, 0.0) + max(0.0, spent)


@contextmanager
def waiting_for(lock) -> Iterator[None]:
    """Holds ``lock``; the time spent waiting for it is not counted in the current stage."""
    started = time.perf_counter()
    with lock:
        clock = getattr(_current, "clock", None)
        if clock is not None:
            clock.waited += time.perf_counter() - started
        yield


def disable_power_throttling() -> bool:
    """Asks Windows not to run this process in efficiency mode (EcoQoS). True when it applied.

    Windows 11 throttles processes without a visible foreground window (the server in browser
    mode, the desktop app while minimized or behind another window, a batch run from a console)
    and moves their threads to the efficiency cores: OCR became five to six times slower (a
    scanned page went from about 9 s to about 50 s). The setting only affects this process and
    ends with it. Elsewhere it does nothing.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class _PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    process_power_throttling, current_version, execution_speed = 4, 1, 0x1
    # ControlMask says which policy is set; StateMask 0 turns throttling off for it.
    state = _PowerThrottlingState(current_version, execution_speed, 0)
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        ok = bool(
            kernel32.SetProcessInformation(
                kernel32.GetCurrentProcess(), process_power_throttling, ctypes.byref(state), ctypes.sizeof(state)
            )
        )
    except (AttributeError, OSError):  # older Windows without this API
        ok = False
    if not ok:
        log.info("could not turn off power throttling for this process")
    return ok
