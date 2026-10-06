"""PDF: text layer with the geometry of each character, image regions, redaction and cleanup.

Every PyMuPDF call goes through ``PDF_LOCK`` (it is not thread-safe); OCR and face detection
run outside the lock, over numpy copies of the rendered page. Coordinates here are PyMuPDF's
*unrotated* page space (the space of ``add_redact_annot``); the engine converts them to view
space with ``page.rotation_matrix`` for the findings.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymupdf

from anonymizer.engine import context, faces, raster, signatures, strokes
from anonymizer.engine.common import OCR_DPI, SCANNED_MAX_CHARS, FileError, Zone, stage, waiting_for
from anonymizer.engine.locks import PDF_LOCK
from anonymizer.engine.model import DetectionOptions
from anonymizer.engine.ocr import OcrLine
from anonymizer.engine.patterns import normalize_1to1
from anonymizer.engine.text import (
    DETECTOR_PRIORITY,
    DOUBT_CONTEXT_NAME,
    Span,
    dedup_spans,
    detect_spans,
    in_list,
    is_personal_url,
    needle,
    rut_doubt,
    strong_needle,
)

_CATALOG_KEYS = ("Names", "OpenAction", "AA", "AcroForm", "OCProperties", "Outlines", "Metadata", "PageLabels",
                 "StructTreeRoot", "MarkInfo", "PieceInfo")  # fmt: skip
_PAGE_KEYS = ("AA", "PieceInfo", "Thumb", "Metadata")


@dataclass
class PageZone:
    """Something to redact on a page, in unrotated page space."""

    page: int
    rect: pymupdf.Rect
    type: str
    text: str
    detector: str
    score: float | None
    doubt: str | None
    source: str  # "text" (text layer) or "raster" (pixels: OCR, faces, QR; and vector signatures)
    optional: bool = False  # D12: it only covers a URL that is not personal


@dataclass
class TextPage:
    """The text layer of a page and the personal data found in it."""

    index: int
    text: str
    boxes: list[pymupdf.Rect | None]
    lines: list[tuple[int, int, context.Line]]
    spans: list[Span] = field(default_factory=list)
    rects: list[tuple[str, float, float, float, float]] = field(default_factory=list)  # context zones without text
    list_ranges: list[tuple[int, int]] = field(default_factory=list)  # spans that came from the name list
    # Where data other than URLs was found (also inside a URL, before ``dedup_spans``): a URL that
    # touches one of these ranges is never optional (D12).
    data_ranges: list[tuple[int, int]] = field(default_factory=list)

    @property
    def scanned(self) -> bool:
        return len(self.text.strip()) < SCANNED_MAX_CHARS


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def open_pdf(path: str) -> pymupdf.Document:
    """Opens a PDF (call with ``PDF_LOCK`` held). Raises ``FileError`` (corrupt, password)."""
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - any failure to open means a damaged file
        raise FileError("corrupt", repr(exc)) from exc
    if doc.needs_pass:
        doc.close()
        raise FileError("password")
    if doc.page_count == 0:  # it starts like a PDF (an empty file is caught before) but has no pages
        doc.close()
        raise FileError("corrupt")
    return doc


def reveal_layers(doc: pymupdf.Document) -> None:
    """Turns every optional layer on: hidden layers are read, shown and removed like the rest."""
    doc.xref_set_key(doc.pdf_catalog(), "OCProperties", "null")


# ---------------------------------------------------------------------------
# Text layer
# ---------------------------------------------------------------------------


def chars(page: pymupdf.Page) -> tuple[str, list[pymupdf.Rect | None], list[bool]]:
    """Text of the page, the box of each character (``None`` at line ends) and, per line, if it is horizontal.

    Text outside the page box is included: it is invisible but extractable.
    """
    tp = page.get_textpage(clip=pymupdf.INFINITE_RECT(), flags=pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_MEDIABOX_CLIP)
    data = page.get_text("rawdict", textpage=tp)
    text: list[str] = []
    boxes: list[pymupdf.Rect | None] = []
    horizontal: list[bool] = []
    for block in data["blocks"]:
        for line in block.get("lines", []):
            horizontal.append(abs(line.get("dir", (1.0, 0.0))[1]) < 0.02)
            for span in line["spans"]:
                for ch in span["chars"]:
                    text.append(ch["c"])
                    box = pymupdf.Rect(ch["bbox"])
                    # Spaces get an empty box: they are not redacted and they allow splitting the zone.
                    boxes.append(pymupdf.Rect() if ch["c"].isspace() or box.is_empty else box)
            text.append("\n")
            boxes.append(None)
    return "".join(text), boxes, horizontal


def text_lines(
    text: str, boxes: list[pymupdf.Rect | None], horizontal: list[bool] | None = None
) -> list[tuple[int, int, context.Line]]:
    """Lines of the text layer: (start, end, Line with its box in points).

    ``horizontal``: one flag per line of ``chars`` (all horizontal if it is not given).
    """
    output = []
    start = 0
    n_line = 0
    for i, c in enumerate([*boxes, None]):
        if c is None:
            rects = [r for r in boxes[start:i] if r is not None and not r.is_empty]
            if rects and text[start:i].strip():
                u = rects[0]
                for r in rects[1:]:
                    u = u | r
                flat = horizontal[n_line] if horizontal is not None and n_line < len(horizontal) else True
                output.append((start, i, context.Line(text[start:i], u.x0, u.y0, u.x1, u.y1, flat)))
            start = i + 1
            n_line += 1
    return output


def span_rects(boxes: list[pymupdf.Rect | None], a: int, b: int) -> list[pymupdf.Rect]:
    """Redaction rectangles of a span: one per line, split where there is a large gap.

    Spaces (empty boxes) are not covered. To avoid erasing very close neighbouring lines, see
    ``clip_against_neighbors``.
    """
    rects: list[pymupdf.Rect] = []
    current: pymupdf.Rect | None = None
    for c in boxes[a:b]:
        if c is None:
            if current is not None:
                rects.append(current)
            current = None
            continue
        if c.is_empty:
            continue
        if current is not None and c.x0 - current.x1 > 2 * max(c.height, 1):
            rects.append(current)
            current = None
        current = pymupdf.Rect(c) if current is None else current | c
    if current is not None:
        rects.append(current)
    return [pymupdf.Rect(r.x0 - 1, r.y0 - 0.5, r.x1 + 1, r.y1 + 0.5) for r in rects if not r.is_empty]


def clip_against_neighbors(r: pymupdf.Rect, a: int, b: int, lines: list[tuple[int, int, context.Line]]) -> pymupdf.Rect:
    """Clips the zone so it does not touch neighbouring lines that are not part of the span.

    MuPDF erases every character whose box touches the redaction zone, and in texts with tight
    line spacing a character's box invades the line above: without this clipping, redacting a
    name also erased the position written below it ("FUNCIONARIA").

    Only horizontal text is clipped: the boxes of vertical or tilted lines (margin signatures,
    stamps) are not their shape and overlap each other, and clipping them left the data visible.
    """
    own = [ln for start, end, ln in lines if start < b and end > a]
    if r.height > r.width or not all(ln.horizontal for ln in own):
        return r
    r = pymupdf.Rect(r)
    original_height = r.height
    for start, end, ln in lines:
        if (start < b and end > a) or not ln.horizontal:  # a line of the span itself, or not horizontal
            continue
        if ln.x1 <= r.x0 or ln.x0 >= r.x1 or ln.y1 <= r.y0 or ln.y0 >= r.y1:
            continue
        if (ln.y0 + ln.y1) / 2 > (r.y0 + r.y1) / 2:
            r.y1 = min(r.y1, ln.y0 - 0.1)
        else:
            r.y0 = max(r.y0, ln.y1 + 0.1)
    return r if r.height >= original_height * 0.35 else pymupdf.Rect(r.x0, r.y0, r.x1, r.y0 + original_height * 0.35)


def read_text_page(page: pymupdf.Page, name_list: tuple[str, ...], options: DetectionOptions | None = None) -> TextPage:
    """Text layer of a page with its spans of personal data (call with ``PDF_LOCK`` held).

    ``options``: the detection groups that run (default: all of them). Every URL is a span: the
    ones that are not personal become optional zones (``text_zones``).
    """
    options = options or DetectionOptions()
    text, boxes, horizontal = chars(page)
    tp = TextPage(page.number, text, boxes, text_lines(text, boxes, horizontal))
    spans = detect_spans(text, name_list, personal_urls=options.urls_personal, given_names=options.names_context)
    ctx, tp.rects = context.context_rules(
        [ln for _, _, ln in tp.lines], page.rect.width, page.rect.height, names=options.names_context
    )
    spans += [(type_, tp.lines[i][0] + a, tp.lines[i][0] + b, "context") for type_, i, a, b in ctx]
    tp.list_ranges = [(a, b) for _, a, b, det in spans if det == "name_list"]
    tp.data_ranges = [(a, b) for type_, a, b, _ in spans if type_ != "url"]
    tp.spans = dedup_spans(spans)
    return tp


def propagate(pages: list[TextPage]) -> int:
    """Finds the text of every span again: on its page, and in the whole document when it is specific.

    A name found by a rule in one place (a signature, a table cell) is redacted wherever it
    appears. Context finds stay on their page: they are less certain. Returns the spans added.
    """
    normalized = [normalize_1to1(tp.text) for tp in pages]
    sources: dict[str, tuple[str, str, bool, set[int]]] = {}  # text -> type, detector, whole document, pages
    for tp in pages:
        for type_, a, b, detector in tp.spans:
            value = tp.text[a:b].strip()
            if not value:
                continue
            whole = detector != "context" and strong_needle(value)
            previous = sources.get(value)
            on_pages = previous[3] if previous else set()
            on_pages.add(tp.index)
            if previous is None or DETECTOR_PRIORITY.get(detector, 9) < DETECTOR_PRIORITY.get(previous[1], 9):
                sources[value] = (type_, detector, whole, on_pages)
    added = 0
    for value, (type_, detector, whole, on_pages) in sources.items():
        pattern = needle(value)
        if pattern is None:
            continue
        targets = range(len(pages)) if whole else sorted(on_pages)
        for n in targets:
            tp = pages[n]
            for m in pattern.finditer(normalized[n]):
                a, b = m.start(), m.end()
                if any(s[1] < b and a < s[2] for s in tp.spans):
                    continue
                tp.spans.append((type_, a, b, detector))
                added += 1
    return added


def data_in_urls(pages: list[TextPage]) -> None:
    """Marks the URLs that contain a datum found anywhere in the document (D12: never optional).

    A name found by a rule in a signature can appear again inside a URL (".../ana-luisa-soto");
    propagation does not add it there, because it overlaps the URL's span. Call after ``propagate``.
    """
    values = {tp.text[a:b].strip() for tp in pages for type_, a, b, _ in tp.spans if type_ != "url"}
    needles = [p for p in (needle(v) for v in values if v) if p is not None]
    for tp in pages:
        urls = [(a, b) for type_, a, b, _ in tp.spans if type_ == "url"]
        if not urls:
            continue
        normalized = normalize_1to1(tp.text).replace("_", " ")  # ".../ana_soto": "_" is a word character
        for pattern in needles:
            for m in pattern.finditer(normalized):
                if any(a < m.end() and m.start() < b for a, b in urls):
                    tp.data_ranges.append((m.start(), m.end()))


def text_zones(tp: TextPage, name_list: tuple[str, ...]) -> list[PageZone]:
    """Redaction zones of the spans of a page (text layer).

    A URL that is not personal and touches no other datum is an optional zone (D12); so are its
    copies found elsewhere by ``propagate``, since they are the same text.
    """
    zones: list[PageZone] = []
    for type_, a, b, detector in tp.spans:
        value = tp.text[a:b]
        doubt = None
        if type_ == "rut":
            doubt = rut_doubt(value)
        elif type_ == "name" and detector == "context":
            listed = any(la < b and a < lb for la, lb in tp.list_ranges) or in_list(value, name_list)
            doubt = None if listed else DOUBT_CONTEXT_NAME
        elif type_ == "signature":
            doubt = signatures.DOUBT_SIGNATURE
        optional = (
            type_ == "url"
            and not is_personal_url(value, name_list)
            and not any(da < b and a < db for da, db in tp.data_ranges)
        )
        for r in span_rects(tp.boxes, a, b):
            r = clip_against_neighbors(r, a, b, tp.lines)
            zones.append(PageZone(tp.index, r, type_, value, detector, 0.6 if doubt else 1.0, doubt, "text", optional))
    for type_, x0, y0, x1, y1 in tp.rects:
        doubt = {"name": DOUBT_CONTEXT_NAME, "signature": signatures.DOUBT_SIGNATURE}.get(type_)
        zones.append(PageZone(tp.index, pymupdf.Rect(x0, y0, x1, y1), type_, "", "context", None, doubt, "raster"))
    return zones


def vector_signatures(page: pymupdf.Page, tp: TextPage, columns: bool = False) -> list[PageZone]:
    """Signatures drawn as vector paths on a page with text (call with ``PDF_LOCK`` held).

    ``columns``: also the column under a "Firma" header of a table of the text layer (when the
    context rule for names, which finds it otherwise, is off). Scanned pages are left to the raster
    detector: their pixels are read whatever draws them.
    """
    if tp.scanned:
        return []
    space = (page.rect * page.derotation_matrix).normalize()  # unrotated page space, like the text layer
    lines = [ln for _, _, ln in tp.lines]
    found = signatures.detect_vector(page.get_drawings(), lines, space.width, space.height, columns=columns)
    return [
        PageZone(tp.index, pymupdf.Rect(x0, y0, x1, y1), "signature", "", signatures.DETECTOR, score,
                 signatures.DOUBT_SIGNATURE, "raster")
        for x0, y0, x1, y1, score in found
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# Images of a page
# ---------------------------------------------------------------------------


def image_regions(
    info: list[dict[str, Any]], page_rect: pymupdf.Rect, to_pix: pymupdf.Matrix, width: int, height: int
) -> list[tuple[int, int, int, int]]:
    """Pixel boxes of the page render where images are placed, with a small margin.

    Overlapping boxes are joined (so a region is not read twice) and those under 24 px are skipped.
    """
    margin = 8
    boxes: list[list[int]] = []
    for i in info:
        r = (pymupdf.Rect(i["bbox"]) & page_rect) * to_pix
        if r.is_empty:
            continue
        boxes.append(
            [max(0, int(r.x0) - margin), max(0, int(r.y0) - margin),
             min(width, int(r.x1) + margin), min(height, int(r.y1) + margin)]
        )  # fmt: skip
    changed = True
    while changed:
        changed = False
        output: list[list[int]] = []
        for b in boxes:
            for o in output:
                if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]:
                    o[:] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
                    changed = True
                    break
            else:
                output.append(b)
        boxes = output
    return [(x0, y0, x1, y1) for x0, y0, x1, y1 in boxes if x1 - x0 >= 24 and y1 - y0 >= 24]


def _signature_lines(tp: TextPage, to_pix: pymupdf.Matrix, region: tuple[int, int, int, int]) -> list[OcrLine]:
    """Signature keywords of the text layer near an image region, in that region's pixels."""
    x0, y0, x1, y1 = region
    out = []
    for _, _, ln in tp.lines:
        if not signatures.is_keyword_line(ln.text):
            continue
        corners = ((ln.x0, ln.y0), (ln.x1, ln.y0), (ln.x1, ln.y1), (ln.x0, ln.y1))
        pts = np.array([[p.x, p.y] for p in (pymupdf.Point(c) * to_pix for c in corners)], np.float64)
        (lx0, ly0), (lx1, ly1) = pts.min(axis=0), pts.max(axis=0)
        reach = 12 * max(1.0, float(min(lx1 - lx0, ly1 - ly0)))
        if lx0 <= x1 + reach and x0 - reach <= lx1 and ly0 <= y1 + reach and y0 - reach <= ly1:
            out.append(OcrLine(pts - [x0, y0], ln.text, 1.0, 0))
    return out


def _overlaps(r: pymupdf.Rect, box: tuple[int, int, int, int]) -> bool:
    return not r.is_empty and r.x0 < box[2] and box[0] < r.x1 and r.y0 < box[3] and box[1] < r.y1


def _repeated_images(doc: pymupdf.Document) -> set[tuple[int, tuple[int, ...]]]:
    """Images placed at the same place on two pages or more (a letterhead's emblem), as (xref, box)."""
    count: dict[tuple[int, tuple[int, ...]], int] = {}
    for page in doc:
        for info in page.get_image_info(xrefs=True):
            if info.get("xref"):
                key = (int(info["xref"]), tuple(round(v) for v in info["bbox"]))
                count[key] = count.get(key, 0) + 1
    return {key for key, n in count.items() if n >= 2}


def _signature_rules(page: pymupdf.Page, tp: TextPage) -> list[pymupdf.Rect]:
    """Possible signature lines, in the page as it is shown: straight horizontal lines, drawn or typed
    ("________"), at least 30 pt long and shorter than 60 % of the page's width (a page-wide header
    or footer rule is not one)."""
    to_view = pymupdf.Matrix(page.rotation_matrix)
    width = page.rect.width
    rules = []
    for d in page.get_drawings():
        items = d.get("items") or []
        single = len(items) == 1 and (items[0][0] == "l" or (items[0][0] == "re" and "f" in d["type"]))
        r = (pymupdf.Rect(d["rect"]) * to_view).normalize()
        if single and r.height <= 2 and 30 <= r.width <= 0.6 * width:
            rules.append(r)
    for _, _, ln in tp.lines:
        text = ln.text.strip()
        r = (pymupdf.Rect(ln.x0, ln.y0, ln.x1, ln.y1) * to_view).normalize()
        if len(text) >= 10 and set(text) <= set("_.-… ") and r.width <= 0.6 * width:
            rules.append(r)
    return rules


def _anchored(image: pymupdf.Rect, rules: list[pymupdf.Rect], names: list[pymupdf.Rect]) -> bool:
    """A signature line under or across the image, or a person's name right under it (all in the page
    as it is shown)."""
    reach = max(15.0, 0.3 * image.height)
    for r in rules:
        overlap = min(r.x1, image.x1) - max(r.x0, image.x0)
        if (
            overlap >= 0.5 * min(image.width, r.width)
            and image.y0 + 0.3 * image.height <= r.y1
            and r.y0 <= image.y1 + reach
        ):
            return True
    for r in names:
        overlap = min(r.x1, image.x1) - max(r.x0, image.x0)
        if overlap > 0 and image.y1 - 0.2 * image.height <= r.y0 <= image.y1 + 2 * reach:
            return True
    return False


def _name_boxes(tp: TextPage, to_view: pymupdf.Matrix) -> list[pymupdf.Rect]:
    """Boxes of the names found in the text layer, in the page as it is shown."""
    out = []
    for type_, a, b, _ in tp.spans:
        if type_ != "name":
            continue
        boxes = [r for r in tp.boxes[a:b] if r is not None and not r.is_empty]
        if boxes:
            u = pymupdf.Rect(boxes[0])
            for r in boxes[1:]:
                u |= r
            out.append((u * to_view).normalize())
    return out


def raster_zones(
    doc: pymupdf.Document,
    tp: TextPage,
    name_list: tuple[str, ...],
    options: DetectionOptions | None = None,
    cache: dict | None = None,
    step: Callable[[str], None] | None = None,
    dpi: int = OCR_DPI,
) -> list[PageZone]:
    """OCR, faces, signatures and QR codes of a page.

    Scanned pages (no text layer): OCR, faces and signatures over the whole page. Pages with text:
    OCR, faces and signatures only inside each embedded image, however small (a phone in a 2 %
    image is still a leak), and QR codes over the whole page. A small image may be a signature by
    itself, and a signature keyword of the text layer next to an image counts as if it were inside
    it; an image repeated at the same place on several pages (a letterhead's emblem) is not.
    ``cache`` (shared by the pages of a document) reads the same image region, rendered identically
    on several pages, only once. ``options``: the detection groups that run; with OCR, faces,
    signatures and QR off the page is not even rendered.
    """
    options = options or DetectionOptions()
    cache = {} if cache is None else cache
    with waiting_for(PDF_LOCK):
        page = doc[tp.index]
        info = page.get_image_info(xrefs=True)
        if not (tp.scanned or info) or not options.raster:
            return []
        if info and not tp.scanned and options.signatures and "repeated" not in cache:
            cache["repeated"] = _repeated_images(doc)
        repeated_here = [
            i
            for i in info
            if (int(i.get("xref") or 0), tuple(round(v) for v in i["bbox"])) in cache.get("repeated", ())
        ]
        rules = _signature_rules(page, tp) if repeated_here else []
        to_view = pymupdf.Matrix(page.rotation_matrix)
        with stage("render"):
            zoom = dpi / 72
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
            inverse = pymupdf.Matrix(1 / zoom, 1 / zoom) * page.derotation_matrix
            to_pix = page.rotation_matrix * pymupdf.Matrix(zoom, zoom)
            page_rect = pymupdf.Rect(page.rect)
            width, height = pix.width, pix.height
            del pix
    if tp.scanned:
        zones = raster.detect_in_image(rgb, name_list, face_threshold=0.6, step=step, options=options)
    else:
        zones = []
        if options.qr:
            zones = raster.detect_in_image(
                rgb,
                name_list,
                ocr_enabled=False,
                face_regions=[],
                qr_enabled=True,
                step=step,
                options=options,
                signatures_enabled=False,
            )
        found_faces: list[Zone] = []
        reads = options.ocr or options.faces or options.signatures
        regions = image_regions(info, page_rect, to_pix, width, height) if reads else []
        for x0, y0, x1, y1 in regions:
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            keywords = _signature_lines(tp, to_pix, (x0, y0, x1, y1)) if options.signatures else []
            # A small image (not a photo or a scanned card) may be a signature by itself, and a curly
            # stroke alone in an image may be one. An image repeated at the same place on other pages
            # is a letterhead's emblem unless something marks it as a signature (a keyword, a
            # signature line, a person's name under it): then it is the same signature on every page
            # (certificates signed by the same official, initials on every sheet).
            repeated = [
                i for i in repeated_here
                if _overlaps((pymupdf.Rect(i["bbox"]) & page_rect) * to_pix, (x0, y0, x1, y1))
            ]  # fmt: skip
            if repeated and not keywords:
                names = cache.setdefault(("names", tp.index), _name_boxes(tp, to_view))
                repeated = [
                    i for i in repeated if not _anchored((pymupdf.Rect(i["bbox"]) * to_view).normalize(), rules, names)
                ]
            else:
                repeated = []
            whole = x1 - x0 <= 0.6 * width and y1 - y0 <= 0.25 * height and not repeated
            key = (
                crop.shape,
                hashlib.blake2b(crop.tobytes(), digest_size=16).digest(),
                whole,
                bool(repeated),
                tuple((ln.text, ln.polygon.round(1).tobytes()) for ln in keywords),
            )
            if key not in cache:
                cache[key] = raster.detect_in_image(
                    crop,
                    name_list,
                    face_threshold=0.55,
                    qr_enabled=False,
                    ocr_min_side=736,
                    step=step,
                    options=options,
                    signature_lines=keywords,
                    signature_whole=whole,
                    signature_lone=not repeated,
                    signature_page_size=(width, height),
                )
            for z in cache[key]:
                moved = z._replace(polygon=z.polygon + [x0, y0])
                (found_faces if z.type == "face" else zones).append(moved)
        zones += faces.merge(found_faces)
    output = []
    for z in raster.dedup(zones):
        (x0, y0), (x1, y1) = np.asarray(z.polygon).min(axis=0), np.asarray(z.polygon).max(axis=0)
        r = (pymupdf.Rect(float(x0), float(y0), float(x1), float(y1)) * inverse) + (-1, -1, 1, 1)
        output.append(PageZone(tp.index, r, z.type, z.text, z.detector, z.score, z.doubt, "raster", z.optional))
    return output


# ---------------------------------------------------------------------------
# Redaction and cleanup
# ---------------------------------------------------------------------------


def redact(
    source: str,
    dest: str,
    rects_by_page: dict[int, list[pymupdf.Rect]],
    drawn_by_page: dict[int, list[pymupdf.Rect]] | None = None,
    whole_out: list[tuple[int, pymupdf.Rect]] | None = None,
) -> None:
    """Writes ``dest``: ``source`` with the zones really removed (text, vector paths and image pixels)
    and the document cleaned (metadata, XMP, annotations, forms, attachments, layers, bookmarks,
    JavaScript actions), fully rewritten. Stroked paths under the zones are removed by
    ``strokes.remove``; ``drawn_by_page``: the zones whose content may be a drawing (signatures,
    zones drawn by the reviewer), where a pen stroke mostly under the zone goes whole; ``whole_out``
    receives (page, box in unrotated page space) of each stroke removed whole."""
    with PDF_LOCK:
        doc = pymupdf.open(source, filetype="pdf")
        try:
            reveal_layers(doc)
            for n in range(doc.page_count):
                page = doc[n]
                # MuPDF misplaces redaction zones on a rotated page whose CropBox or MediaBox does not
                # start at (0, 0): the zone moved or fell off the page and left the data visible.
                # Unrotated, the page space is exactly the space of the zones; the rotation is put back.
                rotation = page.rotation
                if rotation:
                    page.set_rotation(0)
                if rects_by_page.get(n):
                    whole: list = []
                    strokes.remove(page, rects_by_page[n], (drawn_by_page or {}).get(n, []), whole)
                    if whole_out is not None:
                        whole_out += [(n, pymupdf.Rect(box)) for box in whole]
                for r in rects_by_page.get(n, []):
                    page.add_redact_annot(r, fill=(0, 0, 0))
                page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
                if rotation:
                    page.set_rotation(rotation)
                for annot in list(page.annots() or []):
                    page.delete_annot(annot)
                for widget in list(page.widgets() or []):
                    page.delete_widget(widget)
                for key in _PAGE_KEYS:
                    doc.xref_set_key(page.xref, key, "null")
            for name in list(doc.embfile_names()):
                doc.embfile_del(name)
            doc.set_toc([])
            doc.set_metadata({})
            doc.del_xml_metadata()
            for key in _CATALOG_KEYS:
                doc.xref_set_key(doc.pdf_catalog(), key, "null")
            doc.save(dest, garbage=4, deflate=True, clean=True)
        finally:
            doc.close()
