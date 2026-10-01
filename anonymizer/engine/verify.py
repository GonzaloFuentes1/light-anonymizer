"""Leak check of an exported file, before it is copied to the destination folder.

PDF: the text of every active finding read from the text layer must be gone from the output,
the text-layer patterns (RUT, e-mail, phone) run again over the output must find nothing, and
metadata, XMP and attachments must be empty. Images: no EXIF, XMP, comments or text chunks.
Both: the zone of every active finding must be solid black in the output (``uncovered``), which
also checks what OCR, faces, QR and the reviewer marked, whose text is not in the text layer.

What the reviewer chose to keep (removed findings) and the suggestions left unapplied (D12: URLs
that are not personal) stay visible on purpose and are never a leak. Messages are Spanish: they
are shown to the user.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image

from anonymizer.engine import pdf
from anonymizer.engine.common import bbox_of
from anonymizer.engine.locks import PDF_LOCK
from anonymizer.engine.model import TYPE_LABELS, Finding, Leak
from anonymizer.engine.patterns import normalize_1to1
from anonymizer.engine.text import detect_spans, needle

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
    path: Path, active: list[Finding], kept: list[Finding], from_text_layer: Callable[[Finding], bool]
) -> list[Leak]:
    """Leaks of an exported PDF. ``kept``: findings left visible on purpose (removed or suggested).
    ``from_text_layer(f)`` says if the text of ``f`` was read from the text layer."""
    leaks: list[Leak] = []
    with PDF_LOCK:
        with pymupdf.open(path) as doc:
            layers = [(*pdf.chars(page)[:2], pymupdf.Matrix(page.derotation_matrix)) for page in doc]
            metadata = {k: v for k, v in (doc.metadata or {}).items() if v and k not in ("format", "encryption")}
            xmp = doc.get_xml_metadata()
            attachments = doc.embfile_count()
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
