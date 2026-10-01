"""Images: frames in view space (EXIF orientation applied), redaction by solid fill, clean rewrite.

The output is a new image built from raw pixels: no EXIF, XMP, ICC profile, comments or text
chunks are carried over. Multi-page TIFF keeps its pages.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

from anonymizer.engine.common import IMAGE_FORMATS, IMAGE_SAVE_OPTIONS, FileError

# Filled polygons are enlarged by 4 % around their center (OCR boxes are tight on the letters).
FILL_GROWTH = 1.04


def _open(path: str) -> Image.Image:
    try:
        img = Image.open(path)
        img.load()
        return img
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        # The magic bytes said "image" (see ``common.sniff``): if it does not open, it is damaged.
        raise FileError("corrupt", repr(exc)) from exc


def output_format(img: Image.Image) -> str:
    """Format of the output: the same as the input (MPO, a phone's JPEG with a preview, is JPEG)."""
    fmt = "JPEG" if img.format == "MPO" else img.format
    return fmt if fmt in IMAGE_FORMATS else "PNG"


def _frames(img: Image.Image) -> Iterator[Image.Image]:
    if img.format == "MPO":  # the extra frames of an MPO are previews of the same picture
        yield img
        return
    yield from ImageSequence.Iterator(img)


def frame_count(path: str) -> int:
    with _open(path) as img:
        return 1 if img.format == "MPO" else getattr(img, "n_frames", 1)


def frames(path: str) -> Iterator[Image.Image]:
    """RGB frames in view space (EXIF orientation applied). Raises ``FileError`` if damaged."""
    with _open(path) as img:
        try:
            for frame in _frames(img):
                yield ImageOps.exif_transpose(frame.copy()).convert("RGB")
        except (OSError, SyntaxError, ValueError) as exc:
            raise FileError("corrupt", repr(exc)) from exc


def redact(source: str, polygons_by_page: dict[int, list[list[list[float]]]], folder: Path) -> tuple[Path, str]:
    """Writes the redacted image into ``folder`` and returns its path and format."""
    with _open(source) as img:
        fmt = output_format(img)
    outputs: list[Image.Image] = []
    for n, frame in enumerate(frames(source)):
        arr = np.array(frame)
        for polygon in polygons_by_page.get(n, []):
            pol = np.asarray(polygon, np.float64)
            center = pol.mean(axis=0)
            grown = (pol - center) * FILL_GROWTH + center
            cv2.fillPoly(arr, [np.round(grown).astype(np.int32)], (0, 0, 0))
        # A new image from raw pixels: no metadata is carried over.
        outputs.append(Image.fromarray(arr))
    staged = folder / ("output" + IMAGE_FORMATS[fmt])
    options = IMAGE_SAVE_OPTIONS[fmt]
    if len(outputs) > 1 and fmt in ("TIFF", "WEBP", "PNG"):
        outputs[0].save(staged, fmt, save_all=True, append_images=outputs[1:], **options)
    else:
        outputs[0].save(staged, fmt, **options)
    return staged, fmt
