"""Quick prototype of the engine: a preview of phase 1 to see real results.

It really detects (it does not use the ground truth except for the name list, which is the one
the user would hand to the application):

- extended RUT, e-mail, phone and URL patterns over the text layer, with the geometry of each
  character (without searching the text again);
- names and addresses of the list, ignoring accents and case, in any order;
- OCR (RapidOCR, PP-OCRv6) at 0°, 90° and 270° (the line classifier covers 180°) over images,
  over PDF pages without text and over each image embedded in a page with text;
- faces (YuNet) in 4 orientations and 2 scales, with the box enlarged by 20 % and the repeated
  detections of a face merged into one box;
- QR codes.

Real redaction: in PDF, ``apply_redactions`` with pixel removal, cleanup of metadata,
annotations, attachments, layers and bookmarks, and a full rewrite of the file; in images, a
solid fill and a new file without metadata.

Known limitations of the prototype: it uses PyMuPDF (AGPL license, decision D1 pending; local
tests only), it redacts the whole OCR line when it contains a piece of data, it does no leak
verification of its own (the evaluator measures it) and it is not optimized.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf  # noqa: E402
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError  # noqa: E402

from test_bench.baselines import rules  # noqa: E402
from test_bench.schema import FileEntry, FileResult, Manifest, Redaction  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
YUNET_MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
OCR_DPI = 200

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_DASH = r"[-‐‑‒–—−]"
RUT = re.compile(rf"(?<![\d.,])\d{{1,3}}(?:[.,·\s]?\d{{3}}){{2}}\s?{_DASH}?\s?[\dkK](?![\w@])")
EMAIL = re.compile(
    r"[\w.+'-]+\s?(?:@|＠|©|\[at\]|\(at\)|\[arroba\]|\(arroba\)|\sarroba\s)\s?[\w-]+(?:\s?[.,]\s?[\w-]+)*\s?[.,]\s?[a-z]{2,4}\b",
    re.IGNORECASE,
)
URL = re.compile(r"https?://\S+|\bwww\.\S+", re.IGNORECASE)
# Anything with an at sign: OCR often loses the dot of the domain ("...@goreficticiocl").
LOOSE_EMAIL = re.compile(r"[\w.+'-]+[ \t]?[@＠][ \t]?[\w.,-]{2,}")
# In OCR the check-digit dash is sometimes read as a dot, comma or space, and the K as an X.
RUT_OCR = re.compile(r"(?<![\d])\d{1,3}(?:[.,·\s]?\d{3}){2}\s?[-‐‑‒–—−.,·]\s?[\dkKxX](?![\w@])")
# Digits with separators on a single line; the dot only between digits (not the full stop of a sentence).
_RUN = re.compile(r"[+(]?\d(?:[\d \t()+‐‑–—-]|\.(?=\d)){5,24}\d")
_PHONE_LABEL = re.compile(r"(?i)(fono|tel[eé]?f?|cel|m[oó]vil|whats|wsp|fax|contacto)[^\n]{0,20}$")
_CONFUSIONS = str.maketrans(
    {"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8", "Z": "2"}
)

# When an OCR line has several types, the first one in this order wins. It is the alphabetical
# order of the former Spanish type codes (correo, direccion, nombre, rut, telefono, url), kept so
# that the reported types do not change.
_TYPE_PRIORITY = ("email", "address", "name", "rut", "phone", "url")


def _rut_valid_by_shape(m: re.Match[str]) -> bool:
    """Avoids taking amounts or dates without a dash as a RUT: without a dash it requires 8-10 chars in a row."""
    text = m.group(0)
    if re.search(_DASH, text):
        return True
    return bool(re.fullmatch(r"\d{7,9}[\dkK]", re.sub(r"\s", "", text))) and "." not in text


def phones(text: str) -> list[tuple[int, int]]:
    output = []
    for m in _RUN.finditer(text):
        d = re.sub(r"\D", "", m.group(0))
        if d.startswith("0056"):
            d = d[4:]
        elif d.startswith("56") and len(d) >= 10:
            d = d[2:]
        if len(d) == 10 and d.startswith("0"):
            d = d[1:]
        ok = len(d) == 9 and d[0] in "23456789"
        if not ok and len(d) == 8 and _PHONE_LABEL.search(text[max(0, m.start() - 25) : m.start()]):
            ok = True
        if ok:
            output.append((m.start(), m.end()))
    return output


def _strip_accents(c: str) -> str:
    base = unicodedata.normalize("NFKD", c)
    base = "".join(x for x in base if not unicodedata.combining(x))
    return (base[:1] or c).casefold()


def normalize_1to1(text: str) -> str:
    """Normalizes character by character (same length), so positions map back to the original."""
    return "".join(_strip_accents(c) if c.strip() else " " for c in text)


@lru_cache(maxsize=64)
def _list_patterns(name_list: tuple[str, ...]) -> list[tuple[str, re.Pattern[str]]]:
    output = []
    for entry in name_list:
        parts = normalize_1to1(entry).split()
        if not parts:
            continue
        variants = {" ".join(parts)}
        is_address = any(ch.isdigit() for ch in entry)
        if not is_address and len(parts) >= 3:
            surnames, given_names = parts[-2:], parts[:-2]
            variants |= {
                " ".join(surnames + given_names),
                " ".join(surnames) + ", " + " ".join(given_names),
                " ".join([given_names[0], surnames[0]]),
                " ".join(surnames),
            }
        if is_address:
            m = re.match(r"^(\D*?\d+)", " ".join(parts))
            if m:
                variants.add(m.group(1))
        for v in variants:
            body = r"[\s,]+".join(re.escape(t.strip(",")) for t in v.split())
            output.append(("address" if is_address else "name", re.compile(rf"(?<!\w){body}(?!\w)")))
    return output


@lru_cache(maxsize=64)
def _list_words(name_list: tuple[str, ...]) -> frozenset[str]:
    """Standalone given names and surnames of the people of the list (no addresses)."""
    return frozenset(
        p for e in name_list if not any(ch.isdigit() for ch in e) for p in normalize_1to1(e).split() if len(p) >= 3
    )


def find_spans(
    text: str, name_list: tuple[str, ...], ocr: bool = False, all_urls: bool = True
) -> list[tuple[str, int, int]]:
    """Spans (type, start, end) with personal data in ``text``."""
    found: list[tuple[str, int, int]] = []
    variants = [text]
    if ocr:
        variants.append(
            re.sub(
                r"\S+",
                lambda m: (
                    m.group(0).translate(_CONFUSIONS)
                    if sum(c.isdigit() for c in m.group(0)) >= len(m.group(0)) / 2
                    else m.group(0)
                ),
                text,
            )
        )
    for t in variants:
        found += [("rut", m.start(), m.end()) for m in RUT.finditer(t) if _rut_valid_by_shape(m)]
        if ocr:
            found += [("rut", m.start(), m.end()) for m in RUT_OCR.finditer(t)]
        found += [("phone", a, b) for a, b in phones(t)]
    found = [(tp, a, b) for tp, a, b in found if not rules.is_amount(text, a, b)]
    found += [("email", m.start(), m.end()) for m in EMAIL.finditer(text)]
    found += [("email", m.start(), m.end()) for m in LOOSE_EMAIL.finditer(text)]
    for m in URL.finditer(text):
        a, b = rules.complete_url(text, m.start(), m.end())
        url = text[a:b]
        if all_urls or rules.PERSONAL_URL.search(url) or _has_data(url, name_list):
            found.append(("url", a, b))
    norm = normalize_1to1(text)
    for type_, pattern in _list_patterns(name_list):
        for m in pattern.finditer(norm):
            a, b = rules.expand_name(text, m.start(), m.end()) if type_ == "name" else (m.start(), m.end())
            found.append((type_, a, b))
    found += [("name", a, b) for a, b in rules.names_by_dictionary(text)]
    return found


def _has_data(url: str, name_list: tuple[str, ...]) -> bool:
    """The URL contains a piece of personal data (RUT, e-mail, phone or a name of the list)."""
    no_dashes = normalize_1to1(url.replace("-", " ").replace("_", " "))
    return bool(
        RUT.search(url)
        or LOOSE_EMAIL.search(url)
        or phones(url)
        or any(p.search(no_dashes) for _, p in _list_patterns(name_list))
    )


# ---------------------------------------------------------------------------
# Image detectors
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _ocr():
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    from rapidocr import RapidOCR

    return RapidOCR(params={"Global.log_level": "critical", "EngineConfig.onnxruntime.intra_op_num_threads": 4})


def _rotate_points(pts: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """Maps points of ``np.rot90(img, k)`` back to the original image of width ``w`` and height ``h``."""
    x, y = pts[:, 0], pts[:, 1]
    if k == 0:
        return pts
    if k == 1:
        return np.stack([w - y, x], axis=1)
    if k == 2:
        return np.stack([w - x, h - y], axis=1)
    return np.stack([y, h - x], axis=1)


def detect_in_image(
    rgb: np.ndarray,
    name_list: tuple[str, ...],
    ocr: bool = True,
    all_urls: bool = True,
    face_regions: list[tuple[int, int, int, int]] | None = None,
    face_threshold: float = 0.5,
    qr: bool = True,
    ocr_min_side: int = 0,
) -> list[tuple[str, np.ndarray, str, str, float]]:
    """Zones (type, Nx2 polygon in pixels, text, detector, score) in an RGB image.

    ``ocr_min_side``: for small crops, the OCR reads the image padded with white up to this side
    (the zones are clipped back to the image). The text detector scales the image so that its
    shorter side reaches 736 px: without padding, a thin crop of 800x120 px was enlarged six
    times and took tens of seconds.

    ``face_regions``: if given, faces are searched only inside those boxes (the images of a PDF
    page with text), which avoids false positives on text and graphics. Overlapping detections
    of the same face are merged into one zone (see ``_merge_faces``).
    """
    h, w = rgb.shape[:2]
    zones: list[tuple[str, np.ndarray, str, str, float]] = []
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    engine = _ocr()
    lines: list[tuple[np.ndarray, str, float]] = []
    upright: list[tuple[np.ndarray, str, float]] = []  # lines of the unrotated pass, for the context
    ocr_img = bgr
    if ocr and (h < ocr_min_side or w < ocr_min_side):
        ocr_img = np.full((max(h, ocr_min_side), max(w, ocr_min_side), 3), 255, np.uint8)
        ocr_img[:h, :w] = bgr
    ph, pw = ocr_img.shape[:2]
    for k in (0, 1, 3) if ocr else ():
        rot = np.ascontiguousarray(np.rot90(ocr_img, k))
        r = engine(rot)
        if r.boxes is None or r.txts is None:
            continue
        for box, txt, score in zip(r.boxes, r.txts, r.scores, strict=False):
            pol = _rotate_points(np.asarray(box, np.float64), k, pw, ph)
            pol = np.stack([pol[:, 0].clip(0, w), pol[:, 1].clip(0, h)], axis=1)
            lines.append((pol, txt, float(score)))
            if k == 0:
                upright.append((pol, txt, float(score)))
            types = {t for t, _, _ in find_spans(txt, name_list, ocr=True, all_urls=all_urls)}
            if types:
                zones.append((min(types, key=_TYPE_PRIORITY.index), pol, txt, "ocr", float(score)))
    # Context: table columns (Nombre, Correo, Teléfono, Firma...) and label-value pairs.
    if upright:
        objects = [rules.Line(t, *pol.min(axis=0), *pol.max(axis=0)) for pol, t, _ in upright]
        spans, rects = rules.context_rules(objects, w, h)
        for type_, i, _a, _b in spans:
            pol, txt, score = upright[i]
            zones.append((type_, pol, txt, "context", score))
        for type_, x0, y0, x1, y1 in rects:
            zones.append((type_, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), "", "context", 1.0))
    # Identity documents: surnames and given names come in standalone lines under their labels.
    if any(re.search(r"apellido|nombres", normalize_1to1(t)) for _, t, _ in lines):
        words = _list_words(name_list)
        for pol, txt, score in lines:
            tokens = normalize_1to1(txt).split()
            if tokens and len(tokens) <= 3 and all(tk in words for tk in tokens):
                zones.append(("name", pol, txt, "ocr", score))
    faces: list[tuple[str, np.ndarray, str, str, float]] = []
    if face_regions is None:
        faces += _faces(bgr, face_threshold)
    else:
        for x0, y0, x1, y1 in face_regions:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 < 24 or y1 - y0 < 24:
                continue
            for type_, pol, txt, det, score in _faces(np.ascontiguousarray(bgr[y0:y1, x0:x1]), face_threshold):
                faces.append((type_, pol + [x0, y0], txt, det, score))
    zones += _merge_faces(faces)
    if qr:
        zones += _qr(bgr)
    return zones


def _merge_faces(
    zones: list[tuple[str, np.ndarray, str, str, float]], min_iou: float = 0.4, min_inside: float = 0.7
) -> list[tuple[str, np.ndarray, str, str, float]]:
    """Merges the repeated detections of a face (4 rotations x several scales) into their union.

    Two boxes are the same face when their IoU is ``>= min_iou`` or when ``min_inside`` of the
    smaller one lies inside the other. The result is the union of the group (never smaller than
    any detection, so recall is kept) with the best score.
    """
    groups: list[list[float]] = []  # x0, y0, x1, y1, score
    for _, pol, _, _, score in zones:
        (x0, y0), (x1, y1) = pol.min(axis=0), pol.max(axis=0)
        groups.append([float(x0), float(y0), float(x1), float(y1), score])

    def same(a: list[float], b: list[float]) -> bool:
        iw = min(a[2], b[2]) - max(a[0], b[0])
        ih = min(a[3], b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0:
            return False
        inter = iw * ih
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter) >= min_iou or inter / max(min(area_a, area_b), 1e-9) >= min_inside

    changed = True
    while changed:  # a union can reach boxes it did not touch before: repeat until stable
        changed = False
        output: list[list[float]] = []
        for g in groups:
            for o in output:
                if same(o, g):
                    o[:] = [min(o[0], g[0]), min(o[1], g[1]), max(o[2], g[2]), max(o[3], g[3]), max(o[4], g[4])]
                    changed = True
                    break
            else:
                output.append(g)
        groups = output
    return [
        ("face", np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), "", "yunet", score)
        for x0, y0, x1, y1, score in groups
    ]


@lru_cache(maxsize=1)
def _yunet():
    return cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), 0.5, 0.3, 5000)


def _faces(bgr: np.ndarray, threshold: float = 0.5) -> list[tuple[str, np.ndarray, str, str, float]]:
    h, w = bgr.shape[:2]
    det = _yunet()
    det.setScoreThreshold(threshold)
    output = []
    for k in range(4):
        rot = np.ascontiguousarray(np.rot90(bgr, k))
        rh, rw = rot.shape[:2]
        for side in sorted({640, 1280, max(rh, rw)}):
            if side > max(rh, rw) * 1.01 and side != 640:
                continue
            f = side / max(rh, rw)
            img = cv2.resize(rot, (max(1, round(rw * f)), max(1, round(rh * f))))
            det.setInputSize((img.shape[1], img.shape[0]))
            _, detections = det.detect(img)
            for c in detections if detections is not None else []:
                x, y, cw, ch, score = c[0] / f, c[1] / f, c[2] / f, c[3] / f, float(c[14])
                mx, my = 0.2 * cw, 0.2 * ch
                pts = np.array(
                    [
                        [x - mx, y - my * 1.5],
                        [x + cw + mx, y - my * 1.5],
                        [x + cw + mx, y + ch + my],
                        [x - mx, y + ch + my],
                    ]
                )
                pol = _rotate_points(pts, k, w, h)
                output.append(("face", pol, "", "yunet", score))
    return output


def _qr(bgr: np.ndarray) -> list[tuple[str, np.ndarray, str, str, float]]:
    try:
        ok, texts, points, _ = cv2.QRCodeDetector().detectAndDecodeMulti(bgr)
    except cv2.error:
        return []
    if not ok or points is None:
        return []
    output = []
    for pts, txt in zip(points, texts, strict=False):
        center = pts.mean(axis=0)
        output.append(("qr", (pts - center) * 1.15 + center, txt or "", "qr", 1.0))
    return output


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

_CATALOG_KEYS = ("Names", "OpenAction", "AA", "AcroForm", "OCProperties", "Outlines", "Metadata", "PageLabels",
                 "StructTreeRoot", "MarkInfo", "PieceInfo")  # fmt: skip
_PAGE_KEYS = ("AA", "PieceInfo", "Thumb", "Metadata")


def _chars(page: pymupdf.Page) -> tuple[str, list[pymupdf.Rect | None], list[bool]]:
    """Text of the page, the box of each character (``None`` at line ends) and, per line, if it is horizontal."""
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


def _span_rects(boxes: list[pymupdf.Rect | None], a: int, b: int) -> list[pymupdf.Rect]:
    """Redaction rectangles of a span: one per line, split where there is a large gap.

    Spaces (empty boxes) are not covered. To avoid erasing very close neighbouring lines, see
    ``_clip_against_neighbors``.
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


def _clip_against_neighbors(r: pymupdf.Rect, a: int, b: int, lines: list[tuple[int, int, rules.Line]]) -> pymupdf.Rect:
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


def _polygon(r: pymupdf.Rect) -> list[list[float]]:
    return [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]]


def _text_lines(
    text: str, boxes: list[pymupdf.Rect | None], horizontal: list[bool] | None = None
) -> list[tuple[int, int, rules.Line]]:
    """Lines of the text layer: (start, end, Line with its box in points).

    ``horizontal``: one flag per line of ``_chars`` (all horizontal if it is not given).
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
                output.append((start, i, rules.Line(text[start:i], u.x0, u.y0, u.x1, u.y1, flat)))
            start = i + 1
            n_line += 1
    return output


def _image_regions(
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


def _process_pdf(
    source: Path, dest: Path, name_list: tuple[str, ...], all_urls: bool = True
) -> tuple[list[Redaction], int]:
    doc = pymupdf.open(source)
    if doc.needs_pass:
        doc.close()
        raise PermissionError("password")
    # Reveal hidden layers: their text must be detected and removed.
    doc.xref_set_key(doc.pdf_catalog(), "OCProperties", "null")
    redactions: list[Redaction] = []
    region_cache: dict[tuple[Any, bytes], list[tuple[str, np.ndarray, str, str, float]]] = {}
    for n, page in enumerate(doc):
        text, boxes, horizontal = _chars(page)
        spans = [(type_, a, b, "regex") for type_, a, b in find_spans(text, name_list, all_urls=all_urls)]
        lines = _text_lines(text, boxes, horizontal)
        ctx, ctx_rects = rules.context_rules([ln for _, _, ln in lines], page.rect.width, page.rect.height)
        spans += [(type_, lines[i][0] + a, lines[i][0] + b, "context") for type_, i, a, b in ctx]
        for type_, a, b, detector in spans:
            for r in _span_rects(boxes, a, b):
                r = _clip_against_neighbors(r, a, b, lines)
                page.add_redact_annot(r, fill=(0, 0, 0))
                redactions.append(Redaction(page=n, polygon=_polygon(r), type=type_, detector=detector, text=text[a:b]))
        for type_, x0, y0, x1, y1 in ctx_rects:
            r = pymupdf.Rect(x0, y0, x1, y1)
            page.add_redact_annot(r, fill=(0, 0, 0))
            redactions.append(Redaction(page=n, polygon=_polygon(r), type=type_, detector="context"))
        # Scanned pages (no text layer): OCR and faces over the whole page. Pages with text: OCR
        # and faces only inside each embedded image, however small (a phone in a 2 % image is
        # still a leak), and QR codes over the whole page.
        info = page.get_image_info()
        scanned = len(text.strip()) < 50
        if scanned or info:
            zoom = OCR_DPI / 72
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
            inverse = pymupdf.Matrix(1 / zoom, 1 / zoom) * page.derotation_matrix
            to_pix = page.rotation_matrix * pymupdf.Matrix(zoom, zoom)
            if scanned:
                zones = detect_in_image(rgb, name_list, all_urls=all_urls, face_threshold=0.6)
            else:
                zones = detect_in_image(rgb, name_list, ocr=False, all_urls=all_urls, face_regions=[])  # QR only
                faces = []
                for x0, y0, x1, y1 in _image_regions(info, page.rect, to_pix, pix.width, pix.height):
                    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
                    # The same logo rendered identically on every page is read only once.
                    key = (crop.shape, hashlib.blake2b(crop.tobytes(), digest_size=16).digest())
                    if key not in region_cache:
                        region_cache[key] = detect_in_image(
                            crop, name_list, all_urls=all_urls, face_threshold=0.55, qr=False, ocr_min_side=736
                        )
                    for type_, pol, txt, detector, score in region_cache[key]:
                        (faces if type_ == "face" else zones).append((type_, pol + [x0, y0], txt, detector, score))
                zones += _merge_faces(faces)
            for type_, pol, txt, detector, score in zones:
                x0, y0 = pol.min(axis=0)
                x1, y1 = pol.max(axis=0)
                r = (pymupdf.Rect(x0, y0, x1, y1) * inverse) + (-1, -1, 1, 1)
                page.add_redact_annot(r, fill=(0, 0, 0))
                redactions.append(
                    Redaction(page=n, polygon=_polygon(r), type=type_, detector=detector, text=txt, score=score)
                )
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
        for annot in list(page.annots() or []):
            page.delete_annot(annot)
        for w in list(page.widgets() or []):
            page.delete_widget(w)
        for key in _PAGE_KEYS:
            doc.xref_set_key(page.xref, key, "null")
    # Document cleanup
    for name in list(doc.embfile_names()):
        doc.embfile_del(name)
    doc.set_toc([])
    doc.set_metadata({})
    doc.del_xml_metadata()
    for key in _CATALOG_KEYS:
        doc.xref_set_key(doc.pdf_catalog(), key, "null")
    dest.parent.mkdir(parents=True, exist_ok=True)
    doc.save(dest, garbage=4, deflate=True, clean=True)
    n_pages = doc.page_count
    doc.close()
    return redactions, n_pages


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP", ".tif": "TIFF", ".tiff": "TIFF"}


def _process_image(
    source: Path, dest: Path, name_list: tuple[str, ...], all_urls: bool = True
) -> tuple[list[Redaction], int]:
    img = Image.open(source)
    img.load()
    frames = [ImageOps.exif_transpose(f.copy()).convert("RGB") for f in ImageSequence.Iterator(img)]
    redactions: list[Redaction] = []
    outputs = []
    for n, frame in enumerate(frames):
        arr = np.array(frame)
        for type_, pol, txt, detector, score in detect_in_image(arr, name_list, all_urls=all_urls):
            center = pol.mean(axis=0)
            pol2 = (pol - center) * 1.04 + center
            cv2.fillPoly(arr, [np.round(pol2).astype(np.int32)], (0, 0, 0))
            redactions.append(
                Redaction(page=n, polygon=pol.round(2).tolist(), type=type_, detector=detector, text=txt, score=score)
            )
        outputs.append(Image.fromarray(arr))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fmt = _FORMATS[source.suffix.lower()]
    options: dict[str, Any] = {
        "JPEG": {"quality": 92},
        "WEBP": {"quality": 92},
        "PNG": {},
        "TIFF": {"compression": "tiff_deflate"},
    }[fmt]
    if len(outputs) > 1:
        outputs[0].save(dest, fmt, save_all=True, append_images=outputs[1:], **options)
    else:
        outputs[0].save(dest, fmt, **options)
    return redactions, len(outputs)


def _real_type(path: Path) -> str:
    head = path.read_bytes()[:12]
    if not head:
        return "empty"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith((b"\xff\xd8\xff", b"\x89PNG", b"II*\x00", b"MM\x00*")) or (
        head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    ):
        return "image"
    return "format"


def process(file_entry: FileEntry, manifest: Manifest, folder: Path, details: dict[str, Any]) -> FileResult:
    source = Path(manifest.root) / file_entry.path
    dest = folder / file_entry.path
    name_list = tuple(manifest.name_list)
    kind = _real_type(source)
    if kind in ("empty", "format"):  # these two are also the error codes
        return FileResult(input=file_entry.path, output=None, error=kind)
    try:
        if kind == "pdf":
            redactions, n = _process_pdf(source, dest, name_list)
        else:
            redactions, n = _process_image(source, dest, name_list)
    except PermissionError:
        return FileResult(input=file_entry.path, output=None, error="password")
    except (UnidentifiedImageError, OSError, RuntimeError, pymupdf.FileDataError, ValueError):
        dest.unlink(missing_ok=True)
        return FileResult(
            input=file_entry.path,
            output=None,
            error="corrupt",
        )
    return FileResult(input=file_entry.path, output=file_entry.path, redactions=redactions, pages_processed=n)
