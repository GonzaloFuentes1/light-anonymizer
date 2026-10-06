"""Leak check of an exported file, before it is copied to the destination folder.

PDF: the text of every active finding read from the text layer must be gone from the output,
the text-layer patterns (RUT, e-mail, phone) run again over the output must find nothing, and
metadata, XMP and attachments must be empty. Images: no EXIF, XMP, comments or text chunks.
Both: the zone of every active finding must be solid black in the output (``uncovered``), which
also checks what OCR, faces, QR and the reviewer marked, whose text is not in the text layer, and
every page exported as an image.

Drawings left under a zone (letters drawn as paths, strokes, patterns, shadings) do not block the
export: decided 2026-10-06, the page is exported as an image instead (``pdf.redact_page``).
``vector_leaks`` only guards that this happened: it runs the same ``leftovers.check`` over the
output and should never find anything. Blocks are left for what an image cannot fix: data still
readable outside the black boxes, and metadata.

What the reviewer chose to keep (removed findings) and the suggestions left unapplied (D12: URLs
that are not personal; D10: values of the exceptions list) stay visible on purpose and are never
a leak. Messages are Spanish: they are shown to the user.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image

from anonymizer.engine import leftovers, pdf, strokes
from anonymizer.engine.common import bbox_of
from anonymizer.engine.locks import PDF_LOCK
from anonymizer.engine.model import TYPE_LABELS, Finding, Leak
from anonymizer.engine.patterns import normalize_1to1
from anonymizer.engine.text import detect_spans, needle, strong_needle

CRITICAL_TYPES = ("rut", "email", "phone")
# A zone counts as covered when almost all of its inside (the antialiased border left out) is black.
_DARK = 80  # every channel at or below this
_MAX_LIGHT_SHARE = 0.02
_EDGE_PX = 2
_RENDER_ZOOM = 2.0
_RENDER_MAX_PIXELS = 25_000_000
_UNMARKED = {"rut": "un RUT", "email": "un correo", "phone": "un teléfono"}
# ImageDescription, Make, Model, Software, DateTime, Artist, HostComputer, XMP, Copyright, IPTC,
# Photoshop, EXIF IFD, GPS IFD, ImageID, user comment-like tags.
_TIFF_PERSONAL_TAGS = {270, 271, 272, 305, 306, 315, 316, 700, 33432, 33723, 34377, 34665, 34853, 32781, 37510}
_IMAGE_METADATA_KEYS = ("exif", "xmp", "XML:com.adobe.xmp", "comment", "photoshop", "iptc")


def _blank(chars: list[str], a: int, b: int) -> None:
    for i in range(a, b):
        if not chars[i].isspace():
            chars[i] = " "


def _within(boxes: list[pymupdf.Rect | None], a: int, b: int, zones: list[pymupdf.Rect]) -> bool:
    """True when every visible character of ``a:b`` touches one of ``zones``."""
    drawn = [box for box in boxes[a:b] if box is not None and not box.is_empty]
    return bool(drawn) and all(any(box.intersects(zone) for zone in zones) for box in drawn)


def pdf_leaks(
    path: Path,
    active: list[Finding],
    kept: list[Finding],
    from_text_layer: Callable[[Finding], bool],
    text_layers: dict[int, tuple[str, list]] | None = None,
) -> list[Leak]:
    """Leaks of an exported PDF. ``kept``: findings left visible on purpose (removed or suggested).
    ``from_text_layer(f)`` says if the text of ``f`` was read from the text layer.
    ``text_layers``: for a page exported as an image, the text and character boxes it had after
    its redaction (``pdf.PageOutcome.text_layer``); it is read instead of the page's (now empty)
    text layer, so what an image shows unmarked is still found."""
    text_layers = text_layers or {}
    with PDF_LOCK:
        with pymupdf.open(path) as doc:
            layers = [
                (*(text_layers[n] if n in text_layers else pdf.chars(page)[:2]), pymupdf.Matrix(page.derotation_matrix))
                for n, page in enumerate(doc)
            ]
            metadata = {k: v for k, v in (doc.metadata or {}).items() if v and k not in ("format", "encryption")}
            xmp = doc.get_xml_metadata()
            attachments = doc.embfile_count()
    leaks: list[Leak] = []
    if metadata or xmp:
        leaks.append(Leak(page=None, type="metadata", message="El archivo todavía tiene metadatos."))
    if attachments:
        leaks.append(Leak(page=None, type="metadata", message="El archivo todavía tiene archivos adjuntos."))
    for n, (text, boxes, to_page) in enumerate(layers):
        chars = list(text)
        normalized = normalize_1to1(text)
        # What the reviewer kept is not looked at, but only where it was kept: the same text in
        # another place may belong to an active finding whose zone failed to remove it.
        here = [f for f in kept if f.page == n and f.text]
        zones = [(pymupdf.Rect(*bbox_of(f.polygon)) * to_page).normalize() + (-1, -1, 1, 1) for f in here]
        for f in here:
            pattern = needle(f.text)
            for m in pattern.finditer(normalized) if pattern else ():
                if _within(boxes, m.start(), m.end(), zones):
                    _blank(chars, m.start(), m.end())
        normalized = normalize_1to1("".join(chars))
        seen: set[str] = set()
        for f in active:
            if f.page != n or not f.text or not from_text_layer(f):
                continue
            pattern = needle(f.text)
            if pattern is None or pattern.pattern in seen:
                continue
            seen.add(pattern.pattern)
            hits = list(pattern.finditer(normalized))
            if hits:
                label = TYPE_LABELS.get(f.type, f.type)
                leaks.append(
                    Leak(page=n, type=f.type, message=f"{label} sigue legible en la página {n + 1}.", finding_id=f.id)
                )
                for m in hits:
                    _blank(chars, m.start(), m.end())
        remaining = "".join(chars)
        reported: set[tuple[str, str]] = set()
        for type_, a, b, _ in detect_spans(remaining, (), all_urls=False):
            value = "".join(remaining[a:b].split())
            if type_ not in CRITICAL_TYPES or (type_, value) in reported:
                continue
            reported.add((type_, value))
            leaks.append(
                Leak(
                    page=n,
                    type=type_,
                    message=f"Hay {_UNMARKED[type_]} legible en la página {n + 1} que no estaba marcado para censurar.",
                )
            )
    return leaks


_LITERAL = re.compile(rb"\((?:\\.|[^\\()]|\((?:\\.|[^\\()])*\))*\)", re.S)
_HEX = re.compile(rb"<([0-9A-Fa-f\s]+)>")
_ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f"}
# Streams that hold page content (their text is the text layer, read by ``pdf_leaks``) or binary
# data (images, fonts) are not searched.
_CONTENT_KEYS = ("/Subtype/Image", "/Subtype/Form", "/PatternType", "/FontFile", "/Length1", "/Subtype/Type1C",
                 "/Subtype/CIDFontType0C", "/Subtype/OpenType", "/ShadingType")  # fmt: skip


def _decode(raw: bytes) -> str:
    """A PDF string's bytes as text: UTF-16 with its mark, else one byte per character."""
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", "replace")
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", "replace")
    return raw.decode("latin-1")


def _strings(source: bytes) -> list[str]:
    """The literal and hexadecimal strings of a piece of PDF syntax, decoded."""
    out = []
    for m in _LITERAL.finditer(source):
        body = m.group(0)[1:-1]
        body = re.sub(rb"\\([nrtbf])", lambda e: _ESCAPES[e.group(1)], body)
        body = re.sub(rb"\\([0-7]{1,3})", lambda e: bytes([int(e.group(1), 8) & 255]), body)
        body = re.sub(rb"\\(.)", rb"\1", body, flags=re.S)
        out.append(_decode(body))
    for m in _HEX.finditer(source):
        digits = re.sub(rb"\s", b"", m.group(1))
        if len(digits) % 2:
            digits += b"0"
        try:
            out.append(_decode(bytes.fromhex(digits.decode())))
        except ValueError:
            continue
    return out


def string_leaks(path: Path, active: list[Finding]) -> list[Leak]:
    """Values of the active findings still written inside an exported PDF outside the content of
    its pages: in the strings of any object (a page's or the catalog's keys, names, structure,
    forms, annotations) and in the decompressed streams that are not page content, images or
    fonts (metadata, attachments, scripts). Each value is searched as the text-layer check does
    (``needle``: accents, case and spacing ignored)."""
    patterns = []
    seen: set[str] = set()
    for f in active:
        # Only values specific enough to be data wherever they appear (two words, a digit, an "@").
        pattern = needle(f.text) if f.text and strong_needle(f.text) else None
        if pattern is not None and pattern.pattern not in seen:
            seen.add(pattern.pattern)
            patterns.append((pattern, f))
    if not patterns:
        return []
    texts: list[str] = []
    with PDF_LOCK:
        with pymupdf.open(path) as doc:
            contents = {x for page in doc for x in page.get_contents()}
            for x in range(1, doc.xref_length()):
                try:
                    source = doc.xref_object(x, compressed=True)
                except Exception:  # noqa: BLE001 - a broken object has nothing to read
                    continue
                texts += _strings(source.encode("latin-1", "replace"))
                if (
                    x in contents
                    or not doc.xref_is_stream(x)
                    or any(k in source.replace(" ", "") for k in _CONTENT_KEYS)
                ):
                    continue
                try:
                    data = doc.xref_stream(x) or b""
                except Exception:  # noqa: BLE001
                    continue
                texts.append(data.decode("latin-1"))
                texts += _strings(data)
    leaks: list[Leak] = []
    normalized = [normalize_1to1(s) for s in texts if s.strip()]
    for pattern, f in patterns:
        if any(pattern.search(s) for s in normalized):
            label = TYPE_LABELS.get(f.type, f.type)
            leaks.append(
                Leak(
                    page=None,
                    type=f.type,
                    message=f"{label} sigue escrito dentro del archivo, fuera del contenido de las páginas.",
                    finding_id=f.id,
                )
            )
    return leaks


def image_leaks(path: Path) -> list[Leak]:
    """Metadata left in an exported image (every page of a TIFF)."""
    with Image.open(path) as img:
        for n in range(getattr(img, "n_frames", 1)):
            img.seek(n)
            text_chunks = getattr(img, "text", None) if img.format == "PNG" else None
            exif = img.getexif()
            # A TIFF's structure lives in the same tags as EXIF: only descriptive tags count there.
            tags = [t for t in exif if t in _TIFF_PERSONAL_TAGS] if img.format == "TIFF" else list(exif)
            if tags or text_chunks or any(k in img.info for k in _IMAGE_METADATA_KEYS):
                return [Leak(page=None, type="metadata", message="La imagen todavía tiene metadatos.")]
    return []


def _uncovered_leak(page: int, finding: Finding) -> Leak:
    label = TYPE_LABELS.get(finding.type, finding.type)
    return Leak(
        page=page,
        type=finding.type,
        message=f"Una zona marcada en la página {page + 1} ({label}) no quedó tapada por completo.",
        finding_id=finding.id,
    )


def _light_share(rgb: np.ndarray, mask: np.ndarray) -> float | None:
    """Share of the masked pixels that are not black (None when the mask is empty)."""
    inside = rgb[mask]
    if not len(inside):
        return None
    return float((inside.max(axis=1) > _DARK).mean())


def _inner_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    kernel = np.ones((2 * _EDGE_PX + 1, 2 * _EDGE_PX + 1), np.uint8)
    return cv2.erode(mask, kernel).astype(bool)


def vector_leaks(path: Path, active: list[Finding], skip: set[int] | frozenset = frozenset()) -> list[Leak]:
    """Drawings still under an active zone in an exported PDF (``leftovers.check``, with the same
    zones as the redaction): a guard that should never fire, since ``pdf.redact_page`` exports such a
    page as an image. ``skip``: the pages exported as an image (their black boxes are pixels now;
    ``uncovered`` checks them). Each page is looked at unrotated, only in memory."""
    leaks: list[Leak] = []
    with PDF_LOCK:
        with pymupdf.open(path) as doc:
            to_page = [pymupdf.Matrix(page.derotation_matrix) for page in doc]
            for n, items in sorted(strokes.zones_by_page(active, to_page).items()):
                if n in skip:
                    continue
                page = doc[n]
                rotation = page.rotation
                if rotation:
                    page.set_rotation(0)
                try:
                    found = leftovers.check(page, [r for r, _ in items], time.monotonic() + 2 * leftovers.SECONDS)
                finally:
                    if rotation:
                        page.set_rotation(rotation)
                if found:
                    f = items[0][1]
                    leaks.append(
                        Leak(
                            page=n,
                            type=f.type,
                            message=f"En la página {n + 1} quedó en el archivo, bajo una zona marcada, algo dibujado "
                            f"que no se quitó: {leftovers.reasons_text(found)}.",
                            finding_id=f.id,
                        )
                    )
    return leaks


def uncovered(path: Path, kind: str, active: list[Finding]) -> list[Leak]:
    """Active findings whose zone (view space) is not solid black in the exported file."""
    by_page: dict[int, list[Finding]] = {}
    for f in active:
        by_page.setdefault(f.page, []).append(f)
    leaks: list[Leak] = []
    if kind == "pdf":
        with PDF_LOCK:
            with pymupdf.open(path) as doc:
                for n in sorted(by_page):
                    if not 0 <= n < doc.page_count:
                        continue
                    page = doc[n]
                    area = max(1.0, page.rect.width * page.rect.height)
                    zoom = min(_RENDER_ZOOM, (_RENDER_MAX_PIXELS / area) ** 0.5)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
                    for f in by_page[n]:
                        x0, y0, x1, y1 = bbox_of(f.polygon)  # PDF zones are redacted as their bounding box
                        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64) * zoom
                        share = _light_share(rgb, _inner_mask(rgb.shape[:2], box))
                        if share is not None and share > _MAX_LIGHT_SHARE:
                            leaks.append(_uncovered_leak(n, f))
                    del pix, rgb
        return leaks
    with Image.open(path) as img:
        for n in range(getattr(img, "n_frames", 1)):
            if n not in by_page:
                continue
            img.seek(n)
            rgb = np.array(img.convert("RGB"))
            for f in by_page[n]:
                share = _light_share(rgb, _inner_mask(rgb.shape[:2], np.asarray(f.polygon, np.float64)))
                if share is not None and share > _MAX_LIGHT_SHARE:
                    leaks.append(_uncovered_leak(n, f))
    return leaks
