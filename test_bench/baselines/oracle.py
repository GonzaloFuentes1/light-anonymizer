"""``oracle`` baseline: redacts exactly the ground truth and removes the metadata.

It must give 100 % recall and zero leaks. Otherwise the ground truth (or its coordinates) is wrong,
or the evaluator has a bug.

- PDF: every page is rendered in its visible orientation at 200 dpi, the polygons of every element
  with ``type != "text"`` in visible layers (``text``, ``raster``, ``vector``; the ``hidden`` layer
  vanishes by itself when rasterizing) are filled with black, and a new PDF of image pages is built,
  with the same visible size and no metadata.
- Images: the EXIF orientation is applied, the polygons are filled (in every frame of a multipage
  TIFF) and the result is saved in the same format with no metadata at all.
- Files with ``expected == "error:<code>"``: ``error=<code>`` is reported with no output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageDraw, ImageOps, ImageSequence

from test_bench.schema import Element, FileEntry, FileResult, Manifest, Redaction

DPI = 200
# Margin by which every fill is grown, so that the anti-aliased edge when the page is rendered
# again (the evaluator renders it at 144 dpi) or the JPEG compression leave no pixels off-color.
PDF_MARGIN_PT = 1.0
IMAGE_MARGIN_PX = 2.0
JPEG_MARGIN_PX = 4.0
VISIBLE_LAYERS = ("text", "raster", "vector")


class OracleError(Exception):
    """File the oracle cannot process; ``code`` goes to the report's ``error`` field."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def targets(file_entry: FileEntry) -> list[Element]:
    """Elements to fill: every piece of personal data with a polygon in a visible layer."""
    return [e for e in file_entry.elements if e.type != "text" and e.polygon is not None and e.layer in VISIBLE_LAYERS]


def fill(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: Any, margin: float) -> None:
    """Fills the polygon and grows it ``margin`` pixels outwards (rounded stroke along the edge)."""
    draw.polygon(points, fill=color)
    if margin <= 0:
        return
    draw.line([*points, points[0]], fill=color, width=max(1, round(2 * margin)), joint="curve")
    for x, y in points:
        draw.ellipse([x - margin, y - margin, x + margin, y + margin], fill=color)


def _black(mode: str) -> Any:
    return {"L": 0, "LA": (0, 255), "RGB": (0, 0, 0), "RGBA": (0, 0, 0, 255)}[mode]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _pdf(file_entry: FileEntry, source: Path, dest: Path) -> int:
    doc = pymupdf.open(source)
    try:
        if doc.needs_pass:
            raise OracleError("password")
        new = pymupdf.open()
        by_page: dict[int, list[Element]] = {}
        for e in targets(file_entry):
            by_page.setdefault(e.page, []).append(e)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=DPI, alpha=False, colorspace=pymupdf.csRGB)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            visible = page.rect  # visible size (already rotated), origin at (0, 0)
            sx, sy = pix.width / visible.width, pix.height / visible.height
            # From unrotated space to visible pixels: the same transform the evaluator uses.
            matrix = page.rotation_matrix * pymupdf.Matrix(sx, sy)
            draw = ImageDraw.Draw(img)
            for e in by_page.get(i, []):
                assert e.polygon is not None
                points = [tuple(pymupdf.Point(x, y) * matrix) for x, y in e.polygon]
                fill(draw, points, (0, 0, 0), PDF_MARGIN_PT * sx)
            clean = pymupdf.Pixmap(pymupdf.csRGB, img.width, img.height, img.tobytes(), False)
            sheet = new.new_page(width=visible.width, height=visible.height)
            sheet.insert_image(sheet.rect, pixmap=clean)
        n_pages = doc.page_count
    finally:
        doc.close()
    new.set_metadata({})
    new.del_xml_metadata()
    dest.parent.mkdir(parents=True, exist_ok=True)
    new.save(dest, garbage=4, deflate=True)
    new.close()
    return n_pages


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

_DIRECT_MODES = ("L", "LA", "RGB", "RGBA")


def _normalize_mode(img: Image.Image) -> Image.Image:
    if img.mode in _DIRECT_MODES:
        return img
    if img.mode == "P":
        return img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode in ("1", "I;16", "I;16B", "I;16L", "I", "F"):
        return img.convert("L")
    if img.mode == "PA":
        return img.convert("RGBA")
    return img.convert("RGB")


def _without_metadata(img: Image.Image) -> Image.Image:
    """Pixels-only copy: no ``info`` (EXIF, ICC, XMP, PNG texts, TIFF tags)."""
    return Image.frombytes(img.mode, img.size, img.tobytes())


def _image(file_entry: FileEntry, source: Path, dest: Path) -> int:
    with Image.open(source) as original:
        fmt = (original.format or file_entry.format).upper()
        frames = [ImageOps.exif_transpose(c.copy()) for c in ImageSequence.Iterator(original)]
    if fmt == "MPO":
        fmt, frames = "JPEG", frames[:1]
    margin = JPEG_MARGIN_PX if fmt == "JPEG" else IMAGE_MARGIN_PX
    cleaned: list[Image.Image] = []
    for i, frame in enumerate(frames):
        img = _normalize_mode(frame)
        if fmt == "JPEG" and img.mode in ("RGBA", "LA"):
            img = img.convert(img.mode[:-1])
        img = _without_metadata(img)
        draw = ImageDraw.Draw(img)
        for e in targets(file_entry):
            if e.page == i:
                assert e.polygon is not None
                fill(draw, [(x, y) for x, y in e.polygon], _black(img.mode), margin)
        cleaned.append(img)

    dest.parent.mkdir(parents=True, exist_ok=True)
    first, rest = cleaned[0], cleaned[1:]
    if fmt == "JPEG":
        first.save(dest, "JPEG", quality=95, subsampling=0)
    elif fmt == "PNG":
        first.save(dest, "PNG")
    elif fmt == "WEBP":
        options: dict[str, Any] = {"lossless": True}
        if rest:
            options.update(save_all=True, append_images=rest)
        first.save(dest, "WEBP", **options)
    elif fmt == "TIFF":
        first.save(dest, "TIFF", compression="tiff_deflate", save_all=True, append_images=rest)
    else:
        first.save(dest, fmt)
    return len(cleaned)


# ---------------------------------------------------------------------------


def process(file_entry: FileEntry, manifest: Manifest, folder: Path, details: dict[str, Any]) -> FileResult:
    if file_entry.expected.startswith("error:"):
        return FileResult(input=file_entry.path, output=None, error=file_entry.expected.removeprefix("error:"))
    source = Path(manifest.root) / file_entry.path
    dest = folder / file_entry.path
    try:
        if file_entry.format == "pdf":
            n_pages = _pdf(file_entry, source, dest)
        else:
            n_pages = _image(file_entry, source, dest)
    except OracleError as err:
        dest.unlink(missing_ok=True)
        return FileResult(input=file_entry.path, output=None, error=err.code)
    except Exception as err:  # noqa: BLE001 - the report records the exception type
        dest.unlink(missing_ok=True)
        return FileResult(input=file_entry.path, output=None, error=f"exception:{type(err).__name__}")
    redactions = [
        Redaction(page=e.page, polygon=e.polygon, type=e.type, detector="oracle", text=e.value)
        for e in targets(file_entry)
        if e.polygon is not None
    ]
    return FileResult(input=file_entry.path, output=file_entry.path, redactions=redactions, pages_processed=n_pages)
