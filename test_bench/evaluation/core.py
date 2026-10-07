"""File-by-file evaluation: applies checks C, T, B, P, I, O and V.

Every manifest element gets a status:

- ``redacted``: every check that applies to it according to its layer passed.
- ``leak``: at least one failed (they are listed in ``failures``).
- ``not_processed``: the system delivered no output for the file (rejection or missing result).
  It counts as not detected in recall, but not as a leak (there is no file to publish).
- ``neutral``: text without personal data (``type == "text"``) outside the "all text" mode;
  it is only used for the over-redaction and text preservation metrics.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from test_bench.evaluation import coverage as cov
from test_bench.evaluation.image import ImageOutput, format_from_signature, tiff_structure
from test_bench.evaluation.pdf import SCALE, PdfOutput, PlacedImage
from test_bench.evaluation.report import spanish_label
from test_bench.evaluation.search import Haystack, Needle, metadata_needles, needles, texts_from_bytes
from test_bench.schema import (
    Element,
    FileEntry,
    FileResult,
    Manifest,
    MetadataEntry,
    Redaction,
    RedactionReport,
)
from test_bench.visualize import to_visual

# Checks that apply to each layer (table of section 3).
PDF_CHECKS = {
    "text": ("C", "T", "B", "P"),
    "hidden": ("T", "B"),
    "vector": ("C", "P", "V"),
    "raster": ("C", "P", "I", "O"),
}
IMAGE_CHECKS = {
    "text": ("C", "P"),
    "hidden": ("B",),
    "vector": ("C", "P"),
    "raster": ("C", "P"),
}
# Non-textual structures: their mere presence is a leak even if the metadata entry has a canary.
ALWAYS_STRUCTURAL = ("exif.gps", "exif.thumbnail", "pdf.previous_revision")
PAGE_TOLERANCE_PT = 1.0


@dataclass
class ElementEval:
    id: str
    file: str
    category: str
    format: str
    page: int
    type: str
    level: str
    layer: str
    value: str | None
    tags: dict[str, Any]
    status: str = "not_processed"  # redacted | leak | not_processed | neutral
    target: bool = True  # counts in recall and leaks (neutral text only in "all text" mode)
    checks: dict[str, bool | None] = field(default_factory=dict)  # True = passes
    failures: list[str] = field(default_factory=list)
    detected: bool = False
    coverage: float | None = None
    core_coverage: float | None = None
    height_px: float | None = None  # height of the data in output pixels (PDF: at 144 dpi)
    extractable: bool | None = None  # neutral text of the text layer that is still extractable
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetadataEval:
    id: str
    file: str
    category: str
    location: str
    value: str | None
    status: str = "not_processed"  # removed | leak | not_processed
    reasons: list[str] = field(default_factory=list)


@dataclass
class FileEval:
    id: str
    path: str
    format: str
    category: str
    expected: str
    # processed | rejected | no_result | no_output | unreadable_output (processable)
    # correct_rejection | wrong_code_rejection | not_rejected | no_result (expected error)
    status: str
    error: str | None = None
    time_s: float | None = None
    n_pages: int = 0
    output_format: str | None = None
    all_text_mode: bool = False
    elements: list[ElementEval] = field(default_factory=list)
    metadata: list[MetadataEval] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    geometry_mismatch: list[str] = field(default_factory=list)
    n_redactions: int = 0
    redactions_without_data: int = 0
    tags: dict[str, Any] = field(default_factory=dict)
    evaluator_failure: str | None = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _polygon_height(polygon: Sequence[Sequence[float]], face: bool = False) -> float:
    """Height of a quadrilateral, invariant to rotation.

    Text: the shorter pair of opposite sides (a line is wider than it is tall).
    Face: the longer one (the box of a face is taller than it is wide).
    """
    p = np.asarray(polygon, dtype=np.float64)
    if len(p) == 4:
        sides = np.hypot(*(np.roll(p, -1, axis=0) - p).T)
        pairs = ((sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2)
        return float(max(pairs) if face else min(pairs))
    return float(p[:, 1].max() - p[:, 1].min())


def _element_eval(a: FileEntry, e: Element) -> ElementEval:
    return ElementEval(
        id=e.id,
        file=a.id,
        category=a.category,
        format=a.format,
        page=e.page,
        type=e.type,
        level=e.level,
        layer=e.layer,
        value=e.value,
        tags=dict(e.tags),
        target=e.type != "text",
        height_px=None
        if e.polygon is None
        else round(_polygon_height(e.polygon, e.type == "face") * (SCALE if a.format == "pdf" else 1.0), 2),
    )


def _metadata_eval(a: FileEntry, m: MetadataEntry) -> MetadataEval:
    return MetadataEval(id=m.id, file=a.id, category=a.category, location=m.location, value=m.value)


def _expected_code(expected: str) -> str:
    return expected.split(":", 1)[1] if ":" in expected else expected


def _code(error: str | None) -> str:
    return (error or "").split(":", 1)[0].strip().lower()


def _found(haystack: Haystack, needles_: list[Needle], origins: Sequence[str] | None = None) -> list[str]:
    # sorted by Spanish label: the same order as when the origin and part codes were in Spanish
    return sorted({f"{origin}:{a.part}" for a, origin in haystack.search(needles_, origins)}, key=spanish_label)


# ---------------------------------------------------------------------------
# Metadata structures
# ---------------------------------------------------------------------------


def structure_present(location: str, st: dict[str, Any]) -> bool | None:
    """Is the structure named by ``location`` still present? ``None`` if the location is not recognized."""
    d = location.lower()
    # the container as prefix ("png.exif.gps", "webp.xmp") does not change the structure looked for
    for container in ("png.", "jpeg.", "jpg.", "webp."):
        if d.startswith(tuple(container + x for x in ("exif", "xmp", "iptc", "com"))):
            d = d[len(container) :]
            location = location[len(container) :]
            break
    if d.startswith("exif.gps"):
        return bool(st.get("exif.gps"))
    if d.startswith("exif.thumbnail"):
        return bool(st.get("exif.thumbnail"))
    if d.startswith("exif"):
        return bool(st.get("exif"))
    if d.startswith("xmp"):
        return bool(st.get("xmp"))
    if d.startswith("iptc"):
        return bool(st.get("iptc"))
    if d.startswith(("com", "jpeg.com")):
        return bool(st.get("com"))
    if d.startswith("png.text"):
        keys = [c.lower() for c in st.get("png.text", [])]
        parts = location.split(".", 2)
        return (parts[2].lower() in keys) if len(parts) == 3 else bool(keys)
    if d.startswith("tiff"):
        key = tiff_structure(location)
        return bool(st.get(key)) if key else bool(st.get("tiff.tags"))
    if d.startswith("pdf.info"):
        parts = d.split(".", 2)
        return bool(st.get(f"pdf.info.{parts[2]}")) if len(parts) == 3 else bool(st.get("pdf.info"))
    for key in ("pdf.xmp", "pdf.annotation", "pdf.attachment", "pdf.ocg", "pdf.form", "pdf.bookmark"):
        if d.startswith(key):
            return bool(st.get(key))
    if d.startswith("pdf.javascript"):
        return bool(st.get("pdf.javascript"))
    if d.startswith("pdf.previous_revision"):
        return bool(st.get("pdf.previous_revision"))
    return None


# ---------------------------------------------------------------------------
# Evaluation of one file
# ---------------------------------------------------------------------------


def evaluate_file(a: FileEntry, res: FileResult | None, output_dir: Path, input_dir: Path | None = None) -> FileEval:
    """Evaluates a manifest file against its result (``None`` if the report does not include it).

    ``input_dir`` is the dataset root (where the originals are). Check O needs it to know which
    input images contain data, and P uses it to recognize pixels that stayed equal to the
    original even if they look uniform (a dark face or a low-contrast text). Without it those
    parts are not evaluated and a warning is left.
    """
    ev = FileEval(
        id=a.id,
        path=a.path,
        format=a.format,
        category=a.category,
        expected=a.expected,
        status="no_result",
        n_pages=len(a.pages),
        tags=dict(a.tags),
    )
    if res is not None:
        ev.error, ev.time_s, ev.all_text_mode = res.error, res.time_s, bool(res.all_text_mode)
        ev.n_redactions = sum(1 for c in res.redactions if c.status != "dismissed")

    # -- files that must be rejected -----------------------------------------
    if a.expected != "process":
        if res is None:
            ev.status = "no_result"
        elif not res.error:
            ev.status = "not_rejected"
        elif _code(res.error) == _expected_code(a.expected).lower():
            ev.status = "correct_rejection"
        else:
            ev.status = "wrong_code_rejection"
        return ev

    ev.elements = [_element_eval(a, e) for e in a.elements]
    ev.metadata = [_metadata_eval(a, m) for m in a.sensitive_metadata]
    for ee in ev.elements:
        if not ee.target:
            ee.status = "neutral"
        if ev.all_text_mode and ee.type == "text":
            ee.target = True
            ee.status = "not_processed"
    if res is None:
        return ev
    if res.error:
        ev.status = "rejected"
        return ev
    path = output_dir / res.output if res.output else None
    if path is None or not path.is_file():
        ev.status = "no_output"
        return ev
    try:
        _evaluate_output(a, res, path, ev, input_dir)
    except Exception:  # noqa: BLE001 - an evaluator failure must not hide a leak
        ev.evaluator_failure = traceback.format_exc(limit=6)
        ev.status = "unreadable_output"
        for ee in ev.elements:
            if ee.target:
                ee.status, ee.failures = "leak", ["evaluator_error"]
        for em in ev.metadata:
            em.status, em.reasons = "leak", ["evaluator_error"]
    return ev


def _evaluate_output(a: FileEntry, res: FileResult, path: Path, ev: FileEval, input_dir: Path | None) -> None:
    with path.open("rb") as f:
        head = f.read(1024)
    output_format = format_from_signature(head + b"\x00" * 16)
    ev.output_format = output_format
    is_pdf = a.format == "pdf"
    output: PdfOutput | ImageOutput | None = None
    unreadable = None
    try:
        if output_format == "pdf":
            output = PdfOutput(path)
        else:
            output = ImageOutput(path)
    except Exception as ex:  # noqa: BLE001
        unreadable = f"{type(ex).__name__}: {ex}"
    ev.status = "processed"
    zones: dict[int, list[list[list[float]]]] = {}
    for c in res.redactions:
        if c.status != "dismissed":
            zones.setdefault(c.page, []).append(c.polygon)
    samples = cov.PDF_SAMPLES if is_pdf else cov.IMAGE_SAMPLES

    # coverage (independent of the output file)
    for e, ee in zip(a.elements, ev.elements, strict=True):
        if e.polygon is None or ee.layer == "hidden":
            continue
        z = zones.get(e.page, [])
        ee.coverage = round(cov.coverage(e.polygon, z, samples), 4)
        if e.type == "face" and e.core:
            ee.core_coverage = round(cov.coverage(e.core, z, samples), 4)
    ev.redactions_without_data = _redactions_without_data(a, res.redactions, ev)

    if output is None:
        ev.status = "unreadable_output"
        ev.warnings.append(f"no se pudo abrir la salida: {unreadable}")
        for ee in ev.elements:
            if ee.target:
                ee.status, ee.failures = "leak", ["unreadable_output"]
        _evaluate_metadata_raw_bytes(a, path.read_bytes(), ev)
        return
    original = OriginalInput(a, ev, input_dir)
    try:
        pages_ok = _geometry(a, output, is_pdf, output_format, ev)
        for e, ee in zip(a.elements, ev.elements, strict=True):
            if ee.type == "text" and not ee.target:
                _evaluate_neutral(e, ee, output, is_pdf)
                continue
            _evaluate_element(a, e, ee, output, is_pdf, pages_ok, original)
        _evaluate_metadata(a, ev, output)
        ev.warnings.extend(output.read_warnings)
    finally:
        output.close()
        original.close()


class OriginalInput:
    """The input file, opened only if some check asks for it (O and the P control)."""

    def __init__(self, a: FileEntry, ev: FileEval, input_dir: Path | None) -> None:
        self.path = None if input_dir is None else Path(input_dir) / a.path
        self.is_pdf = a.format == "pdf"
        self.ev = ev
        self._reader: PdfOutput | ImageOutput | None = None
        self._attempted = False

    def reader(self) -> PdfOutput | ImageOutput | None:
        if not self._attempted:
            self._attempted = True
            if self.path is None or not self.path.is_file():
                self.ev.warnings.append(
                    "no se encontró el archivo de entrada: no se evaluaron la comprobación O ni el control "
                    "de píxeles sin cambios de P"
                )
            else:
                try:
                    self._reader = PdfOutput(self.path) if self.is_pdf else ImageOutput(self.path)
                except Exception as ex:  # noqa: BLE001
                    self.ev.warnings.append(f"no se pudo abrir la entrada ({ex}): O y el control de P no se evaluaron")
        return self._reader

    def pdf(self) -> PdfOutput | None:
        reader = self.reader()
        return reader if isinstance(reader, PdfOutput) else None

    def visible_page(self, index: int) -> np.ndarray | None:
        """Input page ``index`` as it is seen (PDF at 144 dpi; image with EXIF applied)."""
        reader = self.reader()
        try:
            if isinstance(reader, PdfOutput):
                return reader.render(index) if index < len(reader.doc) else None
            if isinstance(reader, ImageOutput):
                return reader.frames[index] if index < len(reader.frames) else None
        except Exception:  # noqa: BLE001 - without a readable input there is no control
            return None
        return None

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()


def _geometry(
    a: FileEntry, output: PdfOutput | ImageOutput, is_pdf: bool, output_format: str, ev: FileEval
) -> list[bool]:
    """For every manifest page, whether the output has the same geometry."""
    n = len(a.pages)
    if is_pdf != isinstance(output, PdfOutput):
        ev.geometry_mismatch.append(f"la entrada es {a.format} y la salida es {output_format}")
        return [False] * n
    geo = output.geometry()
    if len(geo) != n:
        ev.geometry_mismatch.append(f"la salida tiene {len(geo)} páginas y se esperaban {n}")
        return [False] * n
    ok = []
    for pg, g in zip(a.pages, geo, strict=True):
        if is_pdf:
            # The page is compared as it is seen: an output that draws the page rotated on a
            # sheet without /Rotate (width and height swapped) has the same visible geometry.
            w, h, rot = g
            output_visible = (w, h) if rot % 180 == 0 else (h, w)
            input_visible = (pg.width, pg.height) if pg.rotation % 180 == 0 else (pg.height, pg.width)
            equal = (
                abs(output_visible[0] - input_visible[0]) <= PAGE_TOLERANCE_PT
                and abs(output_visible[1] - input_visible[1]) <= PAGE_TOLERANCE_PT
            )
            if not equal:
                ev.geometry_mismatch.append(
                    f"página {pg.index}: {w:.1f}x{h:.1f} rot {rot} (se esperaba {pg.width:.1f}x{pg.height:.1f} rot {pg.rotation})"
                )
        else:
            w, h = g
            equal = abs(w - pg.width) < 0.5 and abs(h - pg.height) < 0.5
            if not equal:
                ev.geometry_mismatch.append(f"imagen {pg.index}: {w}x{h} (se esperaba {pg.width:g}x{pg.height:g})")
        ok.append(equal)
    return ok


def _to_output(a: FileEntry, index: int, polygon: list[list[float]], pdf: PdfOutput) -> list[list[float]]:
    """Manifest polygon (unrotated input page) -> unrotated space of the output page.

    It is the identity if the output page has the same rotation and size; otherwise (same
    visible page saved with another /Rotate), it goes through the visible page.
    """
    pg = a.pages[index]
    w, h, rot = pdf.geometry()[index]
    if (
        rot % 360 == pg.rotation % 360
        and abs(w - pg.width) <= PAGE_TOLERANCE_PT
        and abs(h - pg.height) <= PAGE_TOLERANCE_PT
    ):
        return polygon
    m = pdf.doc[index].derotation_matrix
    out = []
    for x, y in to_visual(a, index, polygon):
        q = pymupdf.Point(x / SCALE, y / SCALE) * m
        out.append([q.x, q.y])
    return out


def _pixel_zone(e: Element) -> tuple[list[list[float]] | None, float | None]:
    """Polygon and threshold for P/I: for faces, the core (or the whole face with the coverage threshold)."""
    if e.type == "face":
        if e.core:
            return e.core, None
        return e.polygon, cov.FACE_NO_CORE_THRESHOLD
    return e.polygon, None


def _detected(e: Element, ee: ElementEval) -> bool:
    if ee.coverage is None:
        return False
    if e.type == "face":
        if e.core:
            return (ee.core_coverage or 0) >= cov.FACE_CORE_THRESHOLD and ee.coverage >= cov.FACE_BOX_THRESHOLD
        return ee.coverage >= cov.FACE_NO_CORE_THRESHOLD
    return ee.coverage >= cov.COVERAGE_THRESHOLD


def _evaluate_element(
    a: FileEntry,
    e: Element,
    ee: ElementEval,
    output: PdfOutput | ImageOutput,
    is_pdf: bool,
    pages_ok: list[bool],
    original: OriginalInput | None = None,
) -> None:
    layer = e.layer if e.layer in PDF_CHECKS else "raster"
    codes = (PDF_CHECKS if is_pdf else IMAGE_CHECKS)[layer]
    geometry_ok = 0 <= e.page < len(pages_ok) and pages_ok[e.page]
    pdf = output if isinstance(output, PdfOutput) else None
    img = output if isinstance(output, ImageOutput) else None
    nd = needles(e.type, e.value)
    checks: dict[str, bool | None] = {}
    for code in codes:
        if code == "C":
            checks["C"] = _detected(e, ee)
        elif code == "T":
            if not nd or pdf is None:
                checks["T"] = None if pdf is not None else False
                continue
            hits = _found(pdf.text_haystack, nd)
            checks["T"] = not hits
            if hits:
                ee.detail["T"] = hits
        elif code == "B":
            if not nd:
                checks["B"] = None
                continue
            hits = _found(output.bytes_haystack, nd)
            checks["B"] = not hits
            if hits:
                ee.detail["B"] = hits
        elif not geometry_ok:
            checks[code] = False
        elif code == "P":
            polygon, threshold = _pixel_zone(e)
            if polygon is None:
                checks["P"] = None
                continue
            if pdf is not None:
                canvas = pdf.render(e.page)
                pts = [list(p) for p in to_visual(a, e.page, polygon)]
            else:
                assert img is not None
                canvas = img.frames[e.page]
                pts = polygon
            ok, f = cov.is_uniform(canvas, pts, threshold)
            checks["P"] = True if ok is None else ok
            if checks["P"] and original is not None:
                # "uniform" zone that is the same as in the original: low-contrast data left unredacted
                source = original.visible_page(e.page)
                r = None if source is None else cov.correlation(canvas, source, pts)
                if r is not None:
                    ee.detail["original_correlation"] = round(r, 4)
                    if r >= cov.UNCHANGED_THRESHOLD:
                        checks["P"] = False
                        ee.detail["P"] = "píxeles iguales a los del original"
            ee.detail["uniformity"] = None if f is None else round(f, 4)
        elif code == "I":
            checks["I"] = _check_images(a, e, ee, pdf) if pdf is not None else None
        elif code == "O":
            source_pdf = original.pdf() if original is not None and pdf is not None else None
            checks["O"] = _check_originals(e, ee, pdf, source_pdf) if pdf is not None and source_pdf else None
        elif code == "V":
            checks["V"] = _check_strokes(a, e, ee, pdf) if pdf is not None else None
    ee.checks = checks
    ee.failures = [c for c, v in checks.items() if v is False]
    if not geometry_ok:
        ee.failures.insert(0, "geometry_mismatch")
    if e.layer == "hidden":
        ee.detected = all(checks.get(c) is not False for c in codes)
    else:
        ee.detected = bool(checks.get("C"))
    ee.status = "leak" if ee.failures else "redacted"


def _images_under(
    polygon: list[list[float]], images: list[PlacedImage]
) -> list[tuple[PlacedImage, np.ndarray | None, float]]:
    """Placed images that cover at least 10 % of the polygon: (image, pixels under the polygon
    or ``None`` if the image could not be decoded, area of the polygon in image px)."""
    bbox = cov.box(polygon)
    page_area = max(cov.area(polygon), 1e-9)
    out: list[tuple[PlacedImage, np.ndarray | None, float]] = []
    for im in images:
        if not cov.boxes_touch(bbox, im.bbox):
            continue
        if im.pixels is None:
            out.append((im, None, 0.0))
            continue
        in_image = im.to_image(polygon)
        scale = abs(im.transform.a * im.transform.d - im.transform.b * im.transform.c) / max(1, im.width * im.height)
        pix, image_area = cov.polygon_pixels(im.pixels, in_image)
        # fraction of the polygon that falls on this image (area on the page)
        if len(pix) == 0 or len(pix) * scale / page_area < 0.10:
            continue
        out.append((im, pix, image_area))
    return out


def _check_images(a: FileEntry, e: Element, ee: ElementEval, pdf: PdfOutput) -> bool:
    """I: the pixels of every image under the polygon must be uniform (covering them is not enough)."""
    polygon, threshold = _pixel_zone(e)
    if polygon is None:
        return True
    polygon = _to_output(a, e.page, polygon, pdf)
    result = True
    checked = []
    for im, pix, image_area in _images_under(polygon, pdf.images(e.page)):
        if pix is None:
            checked.append({"xref": im.xref, "unreadable": True})
            result = False
            continue
        f = cov.uniformity(pix)
        u = (
            threshold
            if threshold is not None
            else (cov.SMALL_UNIFORM_THRESHOLD if image_area < cov.SMALL_AREA else cov.UNIFORM_THRESHOLD)
        )
        checked.append({"xref": im.xref, "uniformity": round(f, 4)})
        if f < u:
            result = False
    if checked:
        ee.detail["images"] = checked
    return result


def _check_originals(e: Element, ee: ElementEval, pdf: PdfOutput, source: PdfOutput) -> bool:
    """O: no input image that contains the data may remain intact in the output, whether it is
    drawn or not (orphan objects, previous revisions)."""
    if e.polygon is None or not 0 <= e.page < len(source.doc):
        return True
    intact = []
    for im, _, _ in _images_under(e.polygon, source.images(e.page)):
        if im.pixels is None or im.xref <= 0:
            continue  # inline image: it lives in the content stream (checked by I and P)
        where = pdf.contains_image(im.pixels, source.raw_stream(im.xref))
        if where:
            intact.append({"input_xref": im.xref, "in_output": where})
    if intact:
        ee.detail["original_images"] = intact
    return not intact


def _check_strokes(a: FileEntry, e: Element, ee: ElementEval, pdf: PdfOutput) -> bool:
    """V: no Bézier curves (glyph outlines) may remain inside the polygon."""
    if e.polygon is None:
        return True
    points = pdf.curves(e.page)
    if len(points) == 0:
        return True
    polygon = _to_output(a, e.page, e.polygon, pdf)
    x0, y0, x1, y1 = cov.box(polygon)
    near = points[(points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)]
    n = int(cov.points_in_polygon(near, polygon).sum())
    if n:
        ee.detail["curve_points"] = n
    return n == 0


def _evaluate_neutral(e: Element, ee: ElementEval, output: PdfOutput | ImageOutput, is_pdf: bool) -> None:
    """Neutral text: it only matters whether it is still extractable (preservation) and whether it got covered (over-redaction)."""
    if is_pdf and isinstance(output, PdfOutput) and e.layer == "text" and e.value:
        nd = [x for x in needles("text", e.value) if x.part == "value"]
        ee.extractable = bool(output.text_haystack.search(nd))


def _redactions_without_data(a: FileEntry, redactions: list[Redaction], ev: FileEval) -> int:
    """Active redacted zones that touch no personal data (over-redaction)."""
    targets: dict[int, list[list[list[float]]]] = {}
    for e, ee in zip(a.elements, ev.elements, strict=True):
        if ee.target and e.polygon is not None:
            targets.setdefault(e.page, []).append(e.polygon)
    n = 0
    for c in redactions:
        if c.status == "dismissed":
            continue
        if not any(cov.intersect(c.polygon, p) for p in targets.get(c.page, [])):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _evaluate_metadata(a: FileEntry, ev: FileEval, output: PdfOutput | ImageOutput) -> None:
    texts, st, warn = output.metadata[:3]
    ev.warnings.extend(warn)
    if not ev.metadata:
        return
    meta_haystack = Haystack()
    for location, text in texts:
        meta_haystack.add(".".join(location.split(".")[:2]), text)
    haystacks = [meta_haystack, output.bytes_haystack]
    if isinstance(output, PdfOutput):
        haystacks.append(output.text_haystack)
    for em, m in zip(ev.metadata, a.sensitive_metadata, strict=True):
        reasons = []
        nd = metadata_needles(m.value, m.tags.get("type"))
        for haystack in haystacks:
            reasons.extend(f"canario en {h}" for h in _found(haystack, nd))
        present = structure_present(m.location, st)
        structural = m.value is None or any(m.location.lower().startswith(s) for s in ALWAYS_STRUCTURAL)
        if structural:
            if present is None:
                reasons.append(f"lugar no reconocido por el evaluador: {m.location}")
            elif present:
                reasons.append(f"estructura presente: {m.location}")
        em.reasons = reasons
        em.status = "leak" if reasons else "removed"


def _evaluate_metadata_raw_bytes(a: FileEntry, data: bytes, ev: FileEval) -> None:
    """Unreadable output: metadata can only be searched in the bytes (without structure, a leak is assumed)."""
    haystack = Haystack()
    haystack.add("bytes", texts_from_bytes(data), binary=True)
    for em, m in zip(ev.metadata, a.sensitive_metadata, strict=True):
        hits = _found(haystack, metadata_needles(m.value, m.tags.get("type")))
        em.reasons = [f"canario en {h}" for h in hits] or ["unreadable_output"]
        em.status = "leak"


# ---------------------------------------------------------------------------
# The whole dataset
# ---------------------------------------------------------------------------


def _job(args: tuple[FileEntry, FileResult | None, Path, Path]) -> FileEval:
    a, res, output_dir, input_dir = args
    return evaluate_file(a, res, output_dir, input_dir)


def _path_key(path: str) -> str:
    """Comparable relative path: "/" separator and no leading "./"."""
    r = path.replace("\\", "/")
    while r.startswith("./"):
        r = r[2:]
    return r


def match_results(manifest: Manifest, report: RedactionReport) -> tuple[dict[str, FileResult], list[str]]:
    """Result of every manifest file (by input path) and results without a file."""
    by_path: dict[str, FileResult] = {}
    for r in report.results:
        by_path[_path_key(r.input)] = r
    paths = {_path_key(a.path) for a in manifest.files}
    orphans = sorted(set(by_path) - paths)
    return by_path, orphans


def evaluate_files(
    manifest: Manifest, report: RedactionReport, output_dir: Path, processes: int | None = None
) -> tuple[list[FileEval], list[str]]:
    """Evaluates every file (in parallel if ``processes`` > 1)."""
    by_path, orphans = match_results(manifest, report)
    jobs = [(a, by_path.get(_path_key(a.path)), Path(output_dir), Path(manifest.root)) for a in manifest.files]
    if processes is None:
        processes = min(8, max(1, (os.cpu_count() or 2) // 2)) if len(jobs) >= 8 else 1
    if processes <= 1:
        results = [_job(t) for t in jobs]
    else:
        try:
            with ProcessPoolExecutor(max_workers=processes) as ex:
                results = list(ex.map(_job, jobs, chunksize=1))
        except (BrokenProcessPool, OSError):
            # the system did not allow creating or keeping the processes: evaluate in this same process
            results = [_job(t) for t in jobs]
    return results, orphans


def evaluate(
    manifest: Manifest, report: RedactionReport, output_dir: Path, processes: int | None = None
) -> dict[str, Any]:
    """Evaluates a redaction report and returns the complete result (see ``aggregate.aggregate``)."""
    from test_bench.evaluation.aggregate import aggregate

    files, orphans = evaluate_files(manifest, report, output_dir, processes)
    return aggregate(manifest, report, files, orphans)
