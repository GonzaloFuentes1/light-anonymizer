"""Helpers shared by the engines: file type by content, safe output names, errors, geometry."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import uuid
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
    """Real type of a file by its content: pdf | image | empty | format."""
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
    return "format"


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
