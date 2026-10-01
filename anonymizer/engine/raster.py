"""Detection over a raster: a standalone image, a scanned page or an image region of a page.

OCR lines with personal data (patterns, name list, dictionary), context rules over the upright
lines, identity-card name lines, faces and QR codes. Every zone is in pixels of the raster.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import cv2
import numpy as np

from anonymizer.engine import context, faces, names, ocr, qr
from anonymizer.engine.common import Zone
from anonymizer.engine.patterns import TYPE_PRIORITY, normalize_1to1
from anonymizer.engine.text import DOUBT_CONTEXT_NAME, dedup_spans, detect_spans, in_list, ocr_doubt, rut_doubt

# Stage names passed to the ``step`` callback.
STAGE_OCR, STAGE_FACES, STAGE_QR = "ocr", "faces", "qr"


def _line_doubt(type_: str, text: str, score: float, detector: str, name_list: tuple[str, ...]) -> str | None:
    doubt = ocr_doubt(score)
    if doubt is None and type_ == "rut":
        doubt = rut_doubt(text)
    if doubt is None and type_ == "name" and detector == "context" and not in_list(text, name_list):
        doubt = DOUBT_CONTEXT_NAME
    return doubt


def detect_in_image(
    rgb: np.ndarray,
    name_list: tuple[str, ...],
    ocr_enabled: bool = True,
    all_urls: bool = True,
    face_regions: list[tuple[int, int, int, int]] | None = None,
    face_threshold: float = 0.5,
    qr_enabled: bool = True,
    ocr_min_side: int = 0,
    all_text: bool = False,
    step: Callable[[str], None] | None = None,
) -> list[Zone]:
    """Zones with personal data in an RGB image.

    ``face_regions``: if given, faces are searched only inside those boxes (the images of a PDF
    page with text), which avoids false positives on text and graphics; ``[]`` skips faces.
    ``all_text``: every OCR line is a zone (type ``text`` when it has no personal data).
    ``step(stage)`` is called before each stage and between OCR passes (it raises to cancel).
    """
    h, w = rgb.shape[:2]
    zones: list[Zone] = []
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])

    def check_ocr() -> None:
        if step is not None:
            step(STAGE_OCR)

    lines = ocr.read_lines(bgr, ocr_min_side, check_ocr) if ocr_enabled else []
    for line in lines:
        # Spans inside another one do not decide the type (the digits of a phone also look like a RUT).
        spans = dedup_spans(detect_spans(line.text, name_list, ocr=True, all_urls=all_urls))
        types = {s[0] for s in spans}
        if types:
            type_ = min(types, key=TYPE_PRIORITY.index)
            detectors = {s[3] for s in spans if s[0] == type_}
            detector = "context" if detectors == {"context"} else "ocr"
            doubt = _line_doubt(type_, line.text, line.score, detector, name_list)
            zones.append(Zone(type_, line.polygon, line.text, "ocr", line.score, doubt))
        elif all_text and line.text.strip():
            zones.append(Zone("text", line.polygon, line.text, "ocr", line.score, ocr_doubt(line.score)))
    # Context: table columns (Nombre, Correo, Teléfono, Firma...) and label-value pairs.
    upright = [line for line in lines if line.turn == 0]
    if upright:
        objects = [context.Line(ln.text, *ln.polygon.min(axis=0), *ln.polygon.max(axis=0)) for ln in upright]
        spans, rects = context.context_rules(objects, w, h)
        for type_, i, _a, _b in spans:
            ln = upright[i]
            doubt = _line_doubt(type_, ln.text, ln.score, "context", name_list)
            zones.append(Zone(type_, ln.polygon, ln.text, "context", ln.score, doubt))
        for type_, x0, y0, x1, y1 in rects:
            doubt = DOUBT_CONTEXT_NAME if type_ == "name" else None
            zones.append(Zone(type_, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), "", "context", 1.0, doubt))
    # Identity documents: surnames and given names come in standalone lines under their labels.
    if any(re.search(r"apellido|nombres", normalize_1to1(ln.text)) for ln in lines):
        words = names.list_words(name_list)
        for ln in lines:
            tokens = normalize_1to1(ln.text).split()
            if tokens and len(tokens) <= 3 and all(tk in words for tk in tokens):
                zones.append(Zone("name", ln.polygon, ln.text, "name_list", ln.score, ocr_doubt(ln.score)))
    if face_regions is None or face_regions:
        if step is not None:
            step(STAGE_FACES)
        found: list[Zone] = []
        if face_regions is None:
            found += faces.detect(bgr, face_threshold)
        else:
            for x0, y0, x1, y1 in face_regions:
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                if x1 - x0 < 24 or y1 - y0 < 24:
                    continue
                for z in faces.detect(np.ascontiguousarray(bgr[y0:y1, x0:x1]), face_threshold):
                    found.append(z._replace(polygon=z.polygon + [x0, y0]))
        zones += faces.merge(found)
    if qr_enabled:
        if step is not None:
            step(STAGE_QR)
        zones += qr.detect(bgr)
    return dedup(zones)


_TEXT_DETECTORS = ("ocr", "context", "name_list")


def _box(polygon) -> tuple[float, float, float, float, float]:
    """Bounding box of a polygon and how much of it the polygon fills (1.0 = upright rectangle)."""
    pol = np.asarray(polygon, np.float64)
    (x0, y0), (x1, y1) = pol.min(axis=0), pol.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1), _area(pol) / max((x1 - x0) * (y1 - y0), 1e-9)


def _area(pol: np.ndarray) -> float:
    x, y = pol[:, 0], pol[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _hull(points) -> np.ndarray:
    return cv2.convexHull(np.asarray(points, np.float32).reshape(-1, 2)).reshape(-1, 2).astype(np.float64)


def _same_line(a, b, pa=None, pb=None, min_iou: float = 0.7) -> bool:
    """Whether two boxes (``_box``) cover the same line. With the polygons ``pa`` and ``pb``, the
    overlap is measured on their real shape (tilted lines of a photo or a crooked scan)."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return False
    if pa is None or pb is None:
        inter = iw * ih
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union >= min_iou
    ha, hb = _hull(pa), _hull(pb)
    inter, _ = cv2.intersectConvexConvex(ha.astype(np.float32), hb.astype(np.float32))
    union = _area(ha) + _area(hb) - inter
    return union > 0 and inter / union >= min_iou


def dedup(zones: list[Zone]) -> list[Zone]:
    """Joins repeated zones of the same text line.

    - The same box found by several rules: the first one stays.
    - The same line read by several OCR passes (0°, 90° and 270°) or rules, with nearly the same
      shape and the same type: one zone that covers all of them (never smaller than either). An
      upright line keeps an upright rectangle (the union of both boxes); a tilted line (a photo, a
      crooked scan) gets the convex hull of both polygons, so it stays one finding for the
      reviewer instead of three.

    A joined zone is doubtful only if every zone it joins had a doubt.
    """
    output: list[Zone] = []
    boxes: list[tuple[float, float, float, float, float]] = []
    for z in zones:
        box = _box(z.polygon)
        key = np.round(np.asarray(z.polygon, np.float64), 2)
        match = None
        for i, other in enumerate(output):
            same_polygon = np.shape(other.polygon) == key.shape and np.array_equal(
                np.round(np.asarray(other.polygon, np.float64), 2), key
            )
            if same_polygon:
                match = (i, False)
                break
            if (
                z.type == other.type
                and z.detector in _TEXT_DETECTORS
                and other.detector in _TEXT_DETECTORS
                and _same_line(box, boxes[i], z.polygon, other.polygon)
            ):
                match = (i, True)
                break
        if match is None:
            output.append(z)
            boxes.append(box)
            continue
        i, union = match
        kept = output[i]
        if union:
            b = boxes[i]
            best = z if z.score > kept.score else kept
            if box[4] > 0.9 and b[4] > 0.9:
                x0, y0, x1, y1 = min(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]), max(b[3], box[3])
                polygon = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            else:
                polygon = _hull(
                    np.concatenate([np.asarray(kept.polygon, np.float64), np.asarray(z.polygon, np.float64)])
                )
            kept = kept._replace(polygon=polygon, text=best.text, score=best.score)
            boxes[i] = _box(polygon)
        if z.doubt is None:
            kept = kept._replace(doubt=None)
        output[i] = kept
    return output
