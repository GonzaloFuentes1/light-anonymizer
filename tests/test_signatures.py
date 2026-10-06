"""Signature detection by rules (``anonymizer.engine.signatures``), on small fictitious pages.

The signatures are synthetic cursive strokes (loops drawn with OpenCV), the printed text is
OpenCV's Hershey font, and OCR is replaced by the lines each test declares, so no model is
needed. Invented data only.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image

from anonymizer.engine import faces, ocr, qr, signatures
from anonymizer.engine.model import AnalyzedFile, DetectionOptions
from anonymizer.engine.ocr import OcrLine, rotate_points
from anonymizer.engine.real import RealEngine

BLUE = (140, 40, 20)  # BGR: ballpoint blue
BLACK = (30, 30, 30)
NEUTRAL = "Informe ficticio de prueba para el motor de anonimizacion, con texto neutro."


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def cursive(x: float, y: float, width: float, height: float, seed: int = 0, loops: int = 6) -> np.ndarray:
    """Points of a looping cursive stroke inside the box (x, y, width, height)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 500)
    r = 1.5 * width / (2 * np.pi * loops)  # loops: the stroke goes back on itself
    amp = (height / 2 - 2) * (0.7 + 0.3 * np.sin(2 * np.pi * (1.3 + rng.uniform(0, 0.4)) * t + rng.uniform(0, 3)))
    xs = x + r + (width - 2 * r) * t - r * np.sin(2 * np.pi * loops * t)
    ys = y + height / 2 - amp * np.cos(2 * np.pi * loops * t) * np.linspace(0.6, 1.0, t.size)
    return np.stack([xs, ys], axis=1)


def draw_signature(img: np.ndarray, mask: np.ndarray, x, y, width, height, seed=0, color=BLUE, flourish=True):
    """Draws a signature on ``img`` and on ``mask`` (where its ink is), returns nothing."""
    strokes = [cursive(x, y, width, height * 0.8, seed)]
    if flourish:  # an underline that is not straight
        xs = np.linspace(x - 5, x + width + 10, 120)
        strokes.append(np.stack([xs, y + height * 0.9 + 4 * np.sin((xs - x) / 15.0)], axis=1))
    for pts in strokes:
        p = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [p], False, color, 2, cv2.LINE_AA)
        cv2.polylines(mask, [p], False, 255, 2, cv2.LINE_8)


def put_line(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.8, score: float = 0.97) -> OcrLine:
    """Printed text with its OCR line (reading order: top-left, top-right, bottom-right, bottom-left)."""
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 2, cv2.LINE_AA)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    polygon = np.array(
        [[x - 2, y - th - 3], [x + tw + 2, y - th - 3], [x + tw + 2, y + base + 2], [x - 2, y + base + 2]]
    )
    return OcrLine(polygon.astype(np.float64), text, score, 0)


def page(width: int = 1000, height: int = 1300) -> tuple[np.ndarray, np.ndarray]:
    return np.full((height, width, 3), 250, np.uint8), np.zeros((height, width), np.uint8)


def paragraph(img: np.ndarray, y: int, rows: int = 4) -> list[OcrLine]:
    return [put_line(img, NEUTRAL, 60, y + 32 * i) for i in range(rows)]


def covered(zones, mask: np.ndarray) -> float:
    """Share of the ink of ``mask`` inside the union of the zones."""
    union = np.zeros(mask.shape, np.uint8)
    for z in zones:
        cv2.fillPoly(union, [np.round(np.asarray(z.polygon)).astype(np.int32)], 1)
    ink = mask > 0
    return float(union[ink].mean()) if ink.any() else 0.0


def touches(zones, polygon) -> bool:
    shape = (4000, 4000)
    a = np.zeros(shape, np.uint8)
    for z in zones:
        cv2.fillPoly(a, [np.round(np.asarray(z.polygon)).astype(np.int32)], 1)
    b = np.zeros(shape, np.uint8)
    cv2.fillPoly(b, [np.round(np.asarray(polygon)).astype(np.int32)], 1)
    return bool((a & b).any())


def assert_signature_zones(zones):
    assert zones and all(z.type == "signature" and z.detector == signatures.DETECTOR for z in zones)
    assert all(z.doubt == signatures.DOUBT_SIGNATURE for z in zones)


# ---------------------------------------------------------------------------
# Raster: anchors
# ---------------------------------------------------------------------------


def test_signature_above_its_label():
    img, mask = page()
    lines = paragraph(img, 120)
    draw_signature(img, mask, 400, 900, 260, 80, seed=1)
    lines.append(put_line(img, "Firma del titular", 430, 1020, 0.6))
    zones = signatures.detect_raster(img, lines)
    assert_signature_zones(zones)
    assert len(zones) == 1
    assert covered(zones, mask) >= 0.99
    assert not any(touches(zones, ln.polygon) for ln in lines[:4])  # the paragraph stays visible


def test_printed_text_near_a_keyword_is_not_a_signature():
    img, _ = page()
    lines = paragraph(img, 120)
    lines.append(put_line(img, "ANA INVENTADA SOTO", 400, 940))
    lines.append(put_line(img, "Jefa de Unidad de Prueba", 400, 975, 0.7))
    lines.append(put_line(img, "Firma y timbre", 430, 1020, 0.6))
    assert signatures.detect_raster(img, lines) == []


def test_signature_over_a_signature_line_without_keyword():
    img, mask = page()
    lines = paragraph(img, 120)
    draw_signature(img, mask, 380, 880, 240, 70, seed=2, flourish=False)
    cv2.line(img, (350, 960), (680, 960), BLACK, 2)
    lines.append(put_line(img, "Pedro Inventado Rojas", 390, 990, 0.7))
    lines.append(put_line(img, "Encargado de Prueba", 400, 1020, 0.6))
    zones = signatures.detect_raster(img, lines)
    assert_signature_zones(zones)
    assert covered(zones, mask) >= 0.99


def test_cursive_on_its_own_on_a_light_page():
    img, mask = page()
    lines = paragraph(img, 120, rows=10)
    draw_signature(img, mask, 600, 700, 280, 90, seed=3)
    zones = signatures.detect_raster(img, lines)
    assert_signature_zones(zones)
    assert covered(zones, mask) >= 0.99
    assert not any(touches(zones, ln.polygon) for ln in lines)


def test_tilted_photo_of_a_card():
    img, mask = page(1200, 1000)
    lines = [put_line(img, "CEDULA DE PRUEBA", 100, 120, 1.2), put_line(img, "RUN 11.111.111-1", 100, 600, 1.2)]
    draw_signature(img, mask, 620, 520, 230, 70, seed=4)
    lines.append(put_line(img, "FIRMA DEL TITULAR", 640, 640, 0.5))
    m = cv2.getRotationMatrix2D((600, 500), 28, 1.0)
    turned = cv2.warpAffine(img, m, (1200, 1000), borderValue=(90, 70, 50))
    turned_mask = cv2.warpAffine(mask, m, (1200, 1000), flags=cv2.INTER_NEAREST)
    moved = [ln._replace(polygon=cv2.transform(ln.polygon.reshape(-1, 1, 2), m).reshape(-1, 2)) for ln in lines]
    zones = signatures.detect_raster(turned, moved)
    assert_signature_zones(zones)
    assert covered(zones, turned_mask) >= 0.97


def test_page_scanned_sideways():
    img, mask = page()
    lines = paragraph(img, 120)
    draw_signature(img, mask, 400, 900, 260, 80, seed=5)
    lines.append(put_line(img, "Firma del titular", 430, 1020, 0.6))
    side = np.ascontiguousarray(np.rot90(img, -1))  # np.rot90(side, 1) is upright: OCR reads it at 90°
    side_mask = np.ascontiguousarray(np.rot90(mask, -1))
    h, w = side.shape[:2]
    read = [OcrLine(rotate_points(ln.polygon, 1, w, h), ln.text, ln.score, 1) for ln in lines]
    zones = signatures.detect_raster(side, read)
    assert_signature_zones(zones)
    assert covered(zones, side_mask) >= 0.99


def test_large_photo_zones_are_in_its_own_pixels():
    img, mask = page(4000, 3000)
    lines = [put_line(img, NEUTRAL, 200, 400 + 90 * i, 2.4) for i in range(4)]
    draw_signature(img, mask, 1600, 2000, 900, 260, seed=6)
    lines.append(put_line(img, "Firma", 1900, 2400, 2.0))
    zones = signatures.detect_raster(img, lines)
    assert_signature_zones(zones)
    assert covered(zones, mask) >= 0.99
    (x0, y0), (x1, y1) = (
        np.min([np.min(z.polygon, axis=0) for z in zones], axis=0),
        np.max([np.max(z.polygon, axis=0) for z in zones], axis=0),
    )
    assert x0 > 1400 and x1 < 2700 and y0 > 1800 and y1 < 2500


def test_attendance_list_signature_column():
    img, mask = page(1300, 900)
    headers = [("N", 60), ("Nombre", 120), ("RUT", 520), ("Firma", 800)]
    lines = [put_line(img, text, x, 100, 0.7) for text, x in headers]
    for r in range(5):
        y = 160 + 70 * r
        lines.append(put_line(img, str(r + 1), 60, y + 30, 0.7))
        lines.append(put_line(img, f"Persona Inventada {r + 1}", 120, y + 30, 0.7))
        lines.append(put_line(img, f"1{r}.111.111-1", 520, y + 30, 0.7))
        draw_signature(img, mask, 800, y + 2, 200 + 20 * r, 52, seed=10 + r, flourish=r % 2 == 0)
    for x in (40, 100, 500, 780, 1150):
        cv2.line(img, (x, 70), (x, 520), BLACK, 1)
    for y in [70, 120] + [160 + 70 * r + 60 for r in range(5)]:
        cv2.line(img, (40, y), (1150, y), BLACK, 1)
    zones = signatures.detect_raster(img, lines)
    assert_signature_zones(zones)
    assert covered(zones, mask) >= 0.99
    names = [ln for ln in lines if ln.text.startswith("Persona")]
    assert not any(touches(zones, ln.polygon) for ln in names)


def test_attendance_list_read_sideways():
    img, mask = page(1300, 900)
    headers = [("Nombre", 120), ("Correo", 520), ("Firma", 800)]
    lines = [put_line(img, text, x, 100, 0.7) for text, x in headers]
    for r in range(4):
        y = 160 + 70 * r
        lines.append(put_line(img, f"Persona Inventada {r + 1}", 120, y + 30, 0.7))
        lines.append(put_line(img, f"persona{r}@ejemplo.cl", 520, y + 30, 0.7))
        draw_signature(img, mask, 810, y + 4, 180, 48, seed=20 + r, flourish=False)
    side = np.ascontiguousarray(np.rot90(img, 1))  # np.rot90(side, 3) is upright
    side_mask = np.ascontiguousarray(np.rot90(mask, 1))
    h, w = side.shape[:2]
    read = [OcrLine(rotate_points(ln.polygon, 3, w, h), ln.text, ln.score, 3) for ln in lines]
    zones = signatures.detect_raster(side, read)
    assert_signature_zones(zones)
    assert covered(zones, side_mask) >= 0.99


# ---------------------------------------------------------------------------
# Raster: what is not a signature
# ---------------------------------------------------------------------------


def test_plain_document_with_table_stamp_and_logo():
    img, _ = page()
    lines = paragraph(img, 300, rows=8)
    cv2.circle(img, (140, 120), 60, (40, 90, 200), -1)  # filled logo
    lines.append(put_line(img, "GOBIERNO REGIONAL DE PRUEBA", 230, 130))
    for x in (60, 400, 700, 940):  # table grid with printed cells
        cv2.line(img, (x, 650), (x, 900), BLACK, 1)
    for y in (650, 700, 750, 800, 850, 900):
        cv2.line(img, (60, y), (940, y), BLACK, 1)
        if y < 900:
            lines.append(put_line(img, "Item de prueba", 80, y + 35, 0.7))
            lines.append(put_line(img, "$ 120.000", 420, y + 35, 0.7))
    cv2.circle(img, (700, 1100), 90, (170, 80, 40), 3)  # round stamp with unread curved text
    cv2.circle(img, (700, 1100), 70, (170, 80, 40), 2)
    for k in range(14):
        a = 2 * math.pi * k / 14
        cv2.putText(img, "AB"[k % 2], (int(690 + 80 * math.cos(a)), int(1108 + 80 * math.sin(a))),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (170, 80, 40), 1)  # fmt: skip
    cv2.line(img, (100, 1050), (380, 1050), BLACK, 2)  # an empty signature line
    lines.append(put_line(img, "Firma", 200, 1080, 0.6))
    assert signatures.detect_raster(img, lines) == []


def test_photo_texture_is_not_a_signature():
    rng = np.random.default_rng(7)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (900, 1200)).astype(np.float32), (0, 0), 6)
    stripes = (np.abs(np.sin(noise * 8)) * 160 + 40).astype(np.uint8)
    img = cv2.cvtColor(stripes, cv2.COLOR_GRAY2BGR)
    assert signatures.detect_raster(img, []) == []


def test_face_zones_are_left_out():
    img, mask = page(900, 700)
    draw_signature(img, mask, 300, 300, 260, 80, seed=8)
    face = np.array([[250.0, 250.0], [620.0, 250.0], [620.0, 420.0], [250.0, 420.0]])
    assert signatures.detect_raster(img, [], faces=[face]) == []
    assert signatures.detect_raster(img, [])


def test_whole_image_mode_for_a_small_image():
    img, mask = page(420, 140)
    draw_signature(img, mask, 40, 25, 330, 90, seed=9)
    zones = signatures.detect_raster(img, [], whole=True)
    assert_signature_zones(zones)
    assert covered(zones, mask) >= 0.99
    logo, _ = page(420, 140)
    cv2.circle(logo, (70, 70), 50, (40, 90, 200), -1)
    lines = [put_line(logo, "GORE PRUEBA", 140, 85, 0.9)]
    assert signatures.detect_raster(logo, lines, whole=True) == []


def test_keywords():
    found = [t for t in ("Firma", "FIRMA DEL TITULAR", "Firmado por", "V°B° Jefatura", "VºBº", "Vo.Bo.", "p.p. Director",
                         "Nombre y firma", "Firma y timbre:") if signatures.is_keyword_line(t)]  # fmt: skip
    assert len(found) == 9
    for text in ("Firmeza del suelo", "Av. Brasil 123", "Informe de confirmación", "afirma que el proyecto fue",
                 "La presente acta se firma en dos ejemplares del mismo tenor y fecha, quedando una en poder"):  # fmt: skip
        assert not signatures.is_keyword_line(text), text


# ---------------------------------------------------------------------------
# PDF: vector signatures, images and the engine
# ---------------------------------------------------------------------------


def _vector_signature(
    page: pymupdf.Page, x: float, y: float, width: float, height: float, seed: int = 0
) -> pymupdf.Rect:
    """Draws a signature as stroked Bézier curves (as signing tools and tablets do); returns its box."""
    pts = cursive(x, y, width, height, seed, loops=5)[::12]
    pts = pts[: 3 * ((len(pts) - 1) // 3) + 1]  # whole Bézier segments
    shape = page.new_shape()
    for i in range(0, len(pts) - 3, 3):
        shape.draw_bezier(*(pymupdf.Point(*p) for p in pts[i : i + 4]))
    shape.finish(color=(0.1, 0.15, 0.55), width=1.2, closePath=False)
    shape.commit()
    curve = [d["rect"] for d in page.get_drawings() if any(item[0] == "c" for item in d["items"])][-1]
    return pymupdf.Rect(curve)


def _text_page(doc: pymupdf.Document) -> pymupdf.Page:
    page = doc.new_page(width=595, height=842)
    for i in range(6):
        page.insert_text((72, 100 + 18 * i), NEUTRAL, fontsize=10)
    return page


def _analyze(path: Path, options: DetectionOptions | None = None) -> AnalyzedFile:
    file = AnalyzedFile(id="f1", name=path.name, path=str(path), options=(options or DetectionOptions()).to_dict())
    RealEngine().analyze(file, [])
    assert file.status == "ready", (file.error, file.error_message)
    return file


@pytest.fixture
def no_models(monkeypatch):
    """OCR reads nothing, faces and QR find nothing: the tests run without the models."""
    monkeypatch.setattr(ocr, "read_lines", lambda bgr, min_side=0, check=None: [])
    monkeypatch.setattr(faces, "detect", lambda bgr, threshold=0.5, check=None: [])
    monkeypatch.setattr(qr, "detect", lambda bgr: [])


def test_vector_signature_is_found_and_removed(tmp_path, no_models):
    doc = pymupdf.open()
    page = _text_page(doc)
    box = _vector_signature(page, 330, 600, 170, 45, seed=1)
    page.draw_line((320, 655), (520, 655), color=(0, 0, 0), width=0.8)
    page.insert_text((360, 670), "Firma del responsable", fontsize=9)
    doc.save(tmp_path / "firma.pdf")
    doc.close()
    engine = RealEngine()
    file = AnalyzedFile(id="f1", name="firma.pdf", path=str(tmp_path / "firma.pdf"))
    engine.analyze(file, [])
    found = [f for f in file.findings if f.type == "signature"]
    assert len(found) == 1 and found[0].detector == signatures.DETECTOR and found[0].doubtful
    assert found[0].doubt_reason == signatures.DOUBT_SIGNATURE
    zone = pymupdf.Rect(*np.min(found[0].polygon, axis=0), *np.max(found[0].polygon, axis=0))
    assert zone.contains(box)
    assert "signatures" in file.timings
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as out:
        curves = [d for d in out[0].get_drawings() if any(item[0] == "c" for item in d["items"])]
        assert not curves  # the paths are gone from the file, not just covered
        assert "Firma del responsable" in out[0].get_text()  # the label stays


def test_vector_lines_tables_and_charts_are_not_signatures(tmp_path, no_models):
    doc = pymupdf.open()
    page = _text_page(doc)
    for i in range(5):  # table
        page.draw_line((72, 250 + 20 * i), (520, 250 + 20 * i), color=(0, 0, 0), width=0.5)
    for x in (72, 220, 370, 520):
        page.draw_line((x, 250), (x, 330), color=(0, 0, 0), width=0.5)
    pts = [(80 + 10 * i, 500 - 40 * abs(math.sin(i / 3))) for i in range(40)]
    page.draw_polyline(pts, color=(0.8, 0.1, 0.1), width=1)  # line chart
    page.draw_rect(pymupdf.Rect(72, 420, 500, 520), color=(0, 0, 0), width=0.5)
    page.draw_circle((480, 120), 20, color=(0.2, 0.3, 0.7), fill=(0.2, 0.3, 0.7))  # logo
    page.draw_line((320, 700), (520, 700), color=(0, 0, 0), width=0.8)  # empty signature line
    page.insert_text((380, 715), "Firma", fontsize=9)
    doc.save(tmp_path / "plano.pdf")
    doc.close()
    file = _analyze(tmp_path / "plano.pdf")
    assert not [f for f in file.findings if f.type == "signature"]


def test_vector_signature_on_its_own(tmp_path, no_models):
    doc = pymupdf.open()
    page = _text_page(doc)
    box = _vector_signature(page, 300, 500, 200, 60, seed=2)
    doc.save(tmp_path / "sola.pdf")
    doc.close()
    found = [f for f in _analyze(tmp_path / "sola.pdf").findings if f.type == "signature"]
    assert len(found) == 1
    assert pymupdf.Rect(*np.min(found[0].polygon, axis=0), *np.max(found[0].polygon, axis=0)).contains(box)


def _signature_png(width: int = 420, height: int = 140, seed: int = 9, alpha: bool = False) -> bytes:
    img, mask = page(width, height)
    draw_signature(img, mask, 30, 20, width - 80, height - 40, seed=seed)
    rgb = Image.fromarray(img[:, :, ::-1])
    if alpha:
        rgb.putalpha(Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8)))
    buf = io.BytesIO()
    rgb.save(buf, "PNG")
    return buf.getvalue()


@pytest.mark.parametrize("alpha", [False, True])
def test_signature_image_in_a_text_pdf(tmp_path, no_models, alpha):
    doc = pymupdf.open()
    page = _text_page(doc)
    box = pymupdf.Rect(330, 590, 480, 640)
    page.insert_image(box, stream=_signature_png(alpha=alpha))
    page.insert_text((360, 660), "Firma del responsable", fontsize=9)
    doc.save(tmp_path / "imagen.pdf")
    doc.close()
    engine = RealEngine()
    file = AnalyzedFile(id="f1", name="imagen.pdf", path=str(tmp_path / "imagen.pdf"))
    engine.analyze(file, [])
    found = [f for f in file.findings if f.type == "signature"]
    assert found and all(f.detector == signatures.DETECTOR and f.doubtful for f in found)
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]


def test_scanned_page_with_a_signature(tmp_path, monkeypatch, no_models):
    img, mask = page(1654, 2339)
    lines = paragraph(img, 300)
    draw_signature(img, mask, 700, 1700, 400, 110, seed=11)
    lines.append(put_line(img, "Firma del funcionario", 720, 1880, 0.9))
    monkeypatch.setattr(ocr, "read_lines", lambda bgr, min_side=0, check=None: lines)
    buf = io.BytesIO()
    Image.fromarray(img[:, :, ::-1]).save(buf, "PNG")
    doc = pymupdf.open()
    doc.new_page(width=595.3, height=841.9).insert_image(pymupdf.Rect(0, 0, 595.3, 841.9), stream=buf.getvalue())
    doc.save(tmp_path / "escaneo.pdf")
    doc.close()
    engine = RealEngine()
    file = AnalyzedFile(id="f1", name="escaneo.pdf", path=str(tmp_path / "escaneo.pdf"))
    engine.analyze(file, [])
    found = [f for f in file.findings if f.type == "signature"]
    assert found and found[0].doubtful
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as out:
        pix = out[0].get_pixmap(dpi=200)
    rendered = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    ink = cv2.resize(mask, (rendered.shape[1], rendered.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    assert (rendered[ink].max(axis=1) <= 80).mean() > 0.98  # the strokes are black now


def test_signatures_off_skips_the_work(tmp_path, monkeypatch, no_models):
    calls = []
    monkeypatch.setattr(signatures, "detect_raster", lambda *a, **kw: calls.append(1) or [])
    monkeypatch.setattr(signatures, "detect_vector", lambda *a, **kw: calls.append(1) or [])
    doc = pymupdf.open()
    _vector_signature(_text_page(doc), 300, 500, 200, 60)
    doc.save(tmp_path / "a.pdf")
    doc.close()
    Image.new("RGB", (640, 480), "white").save(tmp_path / "foto.png")
    off = DetectionOptions(signatures=False)
    for name in ("a.pdf", "foto.png"):
        file = _analyze(tmp_path / name, off)
        assert not [f for f in file.findings if f.type == "signature"] and "signatures" not in file.timings
    assert not calls
    on = _analyze(tmp_path / "foto.png")
    assert calls and "signatures" in on.timings


def test_context_signature_zones_are_doubtful(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), NEUTRAL, fontsize=10)
    for text, x in (("Nombre", 72), ("Correo", 250), ("Firma", 430)):
        page.insert_text((x, 200), text, fontsize=10)
    page.insert_text((72, 220), "Pedro Inventado Rojas", fontsize=10)
    page.insert_text((250, 220), "pedro@ejemplo.cl", fontsize=10)
    doc.save(tmp_path / "tabla.pdf")
    doc.close()
    found = [f for f in _analyze(tmp_path / "tabla.pdf", DetectionOptions(ocr=False, faces=False, qr=False)).findings
             if f.type == "signature"]  # fmt: skip
    assert found and all(f.doubtful and f.doubt_reason == signatures.DOUBT_SIGNATURE for f in found)


def test_redaction_removes_stroked_paths_inside_the_zone_only(tmp_path):
    # MuPDF alone keeps stroked paths under the black box, and removing every path a zone touches
    # would also remove the page frame and the background band.
    doc = pymupdf.open()
    page = _text_page(doc)
    page.draw_rect(pymupdf.Rect(20, 20, 575, 822), color=(0, 0, 0), width=1)  # frame
    page.draw_rect(pymupdf.Rect(0, 560, 595, 700), color=None, fill=(0.9, 0.95, 1))  # background band
    page.draw_line((300, 580), (560, 580), color=(0, 0, 0))  # a line that crosses the zone's edge
    box = _vector_signature(page, 330, 600, 170, 45, seed=3)
    page.set_rotation(90)
    doc.save(tmp_path / "a.pdf")
    doc.close()
    from anonymizer.engine import pdf

    zone = box + (-4, -4, 4, 4)  # unrotated page space, like the engine's zones
    pdf.redact(str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf"), {0: [zone]})
    with pymupdf.open(tmp_path / "b.pdf") as out:
        page = out[0]
        page.set_rotation(0)
        kept = [(d["type"], [i[0] for i in d["items"]]) for d in page.get_drawings()]
    assert not any("c" in items for _, items in kept)
    assert ("s", ["re"]) in kept and ("f", ["re"]) in kept and ("s", ["l"]) in kept


def test_a_drawing_left_under_a_zone_makes_the_page_an_image(tmp_path, no_models, monkeypatch):
    from anonymizer.engine import strokes

    doc = pymupdf.open()
    page = _text_page(doc)
    _vector_signature(page, 330, 600, 170, 45, seed=1)
    page.insert_text((360, 670), "Firma del responsable", fontsize=9)
    doc.save(tmp_path / "firma.pdf")
    doc.close()
    engine = RealEngine()
    file = AnalyzedFile(id="f1", name="firma.pdf", path=str(tmp_path / "firma.pdf"))
    engine.analyze(file, [])
    assert [f for f in file.findings if f.type == "signature"]
    # As MuPDF alone would leave it: the strokes stay under the zone, so the page is exported as an
    # image (decided 2026-10-06) and nothing drawn is left in the file.
    monkeypatch.setattr(strokes, "remove", lambda page, zones, drawn=(), whole=None, status=None: 0)
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    assert [r["page"] for r in result.rasterized_pages] == [0] and "trazo" in result.rasterized_pages[0]["reason"]
    with pymupdf.open(result.output_path) as out:
        assert not out[0].get_drawings() and not out[0].get_text().strip()


def test_a_zone_drawn_by_the_reviewer_also_removes_the_strokes(tmp_path, no_models):
    # A signature the rules missed, covered by hand with "Dibujar zona": its paths leave the file too.
    doc = pymupdf.open()
    page = _text_page(doc)
    page.draw_bezier((340, 620), (360, 590), (380, 650), (400, 615), color=(0.1, 0.1, 0.5), width=1.5)
    doc.save(tmp_path / "a.pdf")
    doc.close()
    engine = RealEngine()
    file = AnalyzedFile(id="f1", name="a.pdf", path=str(tmp_path / "a.pdf"))
    engine.analyze(file, [])
    from anonymizer.engine.model import Finding

    file.findings.append(Finding(id="m1", file_id="f1", page=0, type="manual", polygon=[[330, 585], [410, 585], [410, 655], [330, 655]],
                                 detector="reviewer", status="added"))  # fmt: skip
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as out:
        assert not [d for d in out[0].get_drawings() if any(i[0] == "c" for i in d["items"])]


# ---------------------------------------------------------------------------
# A larger synthetic set: styles x anchors x geometry, and pages without signatures
# ---------------------------------------------------------------------------

GIVEN = ["Ana", "Pedro", "Rosa", "Luis", "Marta", "Jorge", "Elena", "Tomás", "Irene", "Hugo"]
SURNAMES = ["Inventada", "Ficticio", "Pruebas", "Ejemplar", "Modelo", "Simulada", "Ensayo", "Muestra"]


def font_signature(img, mask, x, y, text, size, color, rng, stroke=True):
    """A name in an italic font with jittered letters (like the bench's), optionally crossed by a stroke."""
    from PIL import ImageDraw

    from test_bench.canvas import font

    pil = Image.fromarray(img[:, :, ::-1].copy())
    ink = Image.new("L", pil.size, 0)
    cx = x
    for ch in text:
        f = font("stix_italic", max(10, int(size * rng.uniform(0.9, 1.15))))
        dy = rng.uniform(-3, 3)
        ImageDraw.Draw(pil).text((cx, y + dy), ch, font=f, fill=color[::-1])
        ImageDraw.Draw(ink).text((cx, y + dy), ch, font=f, fill=255)
        cx += f.getlength(ch) * rng.uniform(0.95, 1.1)
    img[:] = np.array(pil)[:, :, ::-1]
    mask |= (np.array(ink) > 60).astype(np.uint8) * 255
    if stroke:
        xs = np.linspace(x - 6, cx + 14, 80)
        ys = y + size * 1.05 + 5 * np.sin((xs - x) / 16.0)
        p = np.round(np.stack([xs, ys], axis=1)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [p], False, color, 2, cv2.LINE_AA)
        cv2.polylines(mask, [p], False, 255, 2)
    return cx


def synthetic_case(seed: int):
    """One fictitious page with a signature: (image, OCR lines, signature mask, description).

    Seeds 0 to 31 cover every pair of style, anchor and geometry twice (two Latin squares).
    """
    rng = np.random.default_rng(seed)
    img, mask = page(1240, 1754)  # A4 at 150 dpi
    style = ["cursive", "cursive_flourish", "font_stroke", "cursive_black"][seed % 4]
    anchor = ["label", "line", "lone", "vobo"][(seed // 4) % 4]
    geometry = ["upright", "grainy", "tilted", "sideways"][(seed % 4 + (seed // 4) % 4 + 2 * (seed // 16)) % 4]
    if geometry == "grainy":  # paper with some grain
        img[:] = np.clip(img.astype(np.int16) - rng.integers(0, 18, img.shape[:2])[:, :, None], 0, 255).astype(np.uint8)
    lines = paragraph(img, 160, rows=6)
    color = BLACK if style == "cursive_black" else (BLUE if seed % 2 else (120, 30, 10))
    x, y = int(rng.integers(150, 600)), int(rng.integers(900, 1300))
    w, h = int(rng.integers(220, 380)), int(rng.integers(60, 110))
    name = f"{GIVEN[seed % len(GIVEN)]} {SURNAMES[seed % len(SURNAMES)]}"
    if style.startswith("cursive"):
        draw_signature(img, mask, x, y, w, h, seed=seed, color=color, flourish=style == "cursive_flourish")
        bottom = y + h + 8
    else:
        end = font_signature(img, mask, x, y, name, int(h * 0.55), color, rng)
        lines.append(OcrLine(np.array([[x - 4, y - 6], [end + 4, y - 6], [end + 4, y + h * 0.75], [x - 4, y + h * 0.75]],
                                      np.float64), name, 0.9, 0))  # fmt: skip  # OCR reads a font signature
        bottom = y + int(h * 0.75) + 14
    if anchor == "label":
        lines.append(put_line(img, "Firma del funcionario", x + 10, bottom + 40, 0.7))
    elif anchor == "vobo":
        lines.append(put_line(img, "V°B° Jefatura", x + 20, bottom + 40, 0.7))
    elif anchor == "line":
        cv2.line(img, (x - 30, bottom), (x + w + 30, bottom), BLACK, 2)
        lines.append(put_line(img, name.upper(), x + 10, bottom + 35, 0.7))
        lines.append(put_line(img, "Profesional de apoyo", x + 10, bottom + 65, 0.6))
    if geometry == "tilted":
        angle = float(rng.uniform(-30, 30))
        m = cv2.getRotationMatrix2D((620, 877), angle, 1.0)
        img = cv2.warpAffine(img, m, (1240, 1754), borderValue=(120, 110, 100))
        mask = cv2.warpAffine(mask, m, (1240, 1754), flags=cv2.INTER_NEAREST)
        lines = [ln._replace(polygon=cv2.transform(ln.polygon.reshape(-1, 1, 2), m).reshape(-1, 2)) for ln in lines]
    elif geometry == "sideways":
        img, mask = np.ascontiguousarray(np.rot90(img, -1)), np.ascontiguousarray(np.rot90(mask, -1))
        h2, w2 = img.shape[:2]
        lines = [OcrLine(rotate_points(ln.polygon, 1, w2, h2), ln.text, ln.score, 1) for ln in lines]
    return img, lines, mask, f"{style}/{anchor}/{geometry}"


def negative_case(seed: int):
    """A fictitious page without any signature: text, a table, a stamp, a logo or a photo-like block."""
    rng = np.random.default_rng(1000 + seed)
    img, _ = page(1240, 1754)
    lines = paragraph(img, 160, rows=10)
    kind = ["table", "stamp", "chart", "photo", "form"][seed % 5]
    if kind == "table":
        for x in (80, 400, 800, 1160):
            cv2.line(img, (x, 700), (x, 1100), BLACK, 1)
        for i, yy in enumerate(range(700, 1101, 50)):
            cv2.line(img, (80, yy), (1160, yy), BLACK, 1)
            if yy < 1100:
                lines.append(put_line(img, f"Partida {i}", 100, yy + 35, 0.7))
                lines.append(put_line(img, "Firma del convenio", 420, yy + 35, 0.7))
    elif kind == "stamp":
        c = (int(rng.integers(300, 900)), int(rng.integers(800, 1400)))
        cv2.circle(img, c, 110, (170, 80, 40), 3)
        cv2.circle(img, c, 85, (170, 80, 40), 2)
        lines.append(put_line(img, "OFICINA DE PARTES", c[0] - 110, c[1] + 8, 0.7))
        lines.append(put_line(img, "Firma", c[0] - 30, c[1] + 160, 0.7))
    elif kind == "chart":
        xs = np.linspace(150, 1100, 200)
        ys = 1200 - 150 * np.abs(np.sin(xs / 90.0)) - 0.2 * (xs - 150)
        cv2.polylines(
            img, [np.round(np.stack([xs, ys], 1)).astype(np.int32).reshape(-1, 1, 2)], False, (40, 40, 200), 2
        )
        cv2.line(img, (150, 1220), (1100, 1220), BLACK, 2)
        cv2.line(img, (150, 900), (150, 1220), BLACK, 2)
    elif kind == "photo":
        noise = cv2.GaussianBlur(rng.normal(0, 1, (500, 700)).astype(np.float32), (0, 0), 5)
        block = (np.abs(np.sin(noise * 7)) * 170 + 30).astype(np.uint8)
        img[800:1300, 250:950] = cv2.cvtColor(block, cv2.COLOR_GRAY2BGR)
    else:
        for i, label in enumerate(["Nombre:", "RUT:", "Firma:", "Fecha:"]):
            lines.append(put_line(img, label, 100, 800 + 60 * i, 0.8))
            cv2.line(img, (260, 805 + 60 * i), (900, 805 + 60 * i), BLACK, 1)
    return img, lines


def test_synthetic_signature_set():
    found, missed = 0, []
    for seed in range(32):
        img, lines, mask, what = synthetic_case(seed)
        zones = signatures.detect_raster(img, lines)
        if covered(zones, mask) >= 0.95:
            found += 1
        else:
            missed.append((seed, what, round(covered(zones, mask), 2)))
    false = []
    for seed in range(10):
        img, lines = negative_case(seed)
        zones = signatures.detect_raster(img, lines)
        if zones:
            false.append((seed, len(zones)))
    print(f"\nsynthetic signatures found: {found}/32; missed: {missed}; pages without signature flagged: {false}")
    # Every pen stroke is found, and every signature next to a keyword. A name typed in an italic
    # font that OCR reads confidently, with no keyword next to it, looks like printed text: missed on
    # purpose (a lone stroke must lie outside the lines OCR read), and so is one over a signature
    # line on a tilted photo (only horizontal or vertical rules anchor).
    assert all(what.startswith("font_stroke/") and "/label/" not in what and "/vobo/" not in what
               for _, what, _ in missed), missed  # fmt: skip
    assert found >= 28, missed
    assert not false


# ---------------------------------------------------------------------------
# Review findings (invented documents)
# ---------------------------------------------------------------------------


def _landscape_pdf(path: Path, stored_as_portrait: bool, anchored: bool) -> pymupdf.Rect:
    """A landscape page with a vector signature, stored as landscape or as portrait plus /Rotate."""
    src = pymupdf.open()
    p = src.new_page(width=842, height=595)
    for i in range(6):
        p.insert_text((72, 100 + 18 * i), NEUTRAL, fontsize=10)
    box = _vector_signature(p, 500, 380, 200, 60, seed=2)
    if anchored:
        p.draw_line((480, 455), (720, 455), color=(0, 0, 0), width=0.8)
        p.insert_text((540, 470), "Firma del responsable", fontsize=9)
    if not stored_as_portrait:
        src.save(path)
        return box
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.show_pdf_page(page.rect, src, 0, rotate=-90)
    page.set_rotation(270)
    doc.save(path)
    return box


@pytest.mark.parametrize("anchored", [False, True])
@pytest.mark.parametrize("stored_as_portrait", [False, True])
def test_vector_signature_on_a_landscape_page(tmp_path, no_models, anchored, stored_as_portrait):
    box = _landscape_pdf(tmp_path / "a.pdf", stored_as_portrait, anchored)
    found = [f for f in _analyze(tmp_path / "a.pdf").findings if f.type == "signature"]
    assert len(found) == 1
    zone = pymupdf.Rect(*np.min(found[0].polygon, axis=0), *np.max(found[0].polygon, axis=0))
    assert zone.contains(box)  # in view space: the page as it is shown


def test_vector_seal_next_to_a_label_is_not_a_signature(tmp_path, no_models):
    doc = pymupdf.open()
    page = _text_page(doc)
    page.draw_circle((420, 600), 30, color=(0.1, 0.1, 0.6), width=1.2)  # an institutional seal: two rings
    page.draw_circle((420, 600), 22, color=(0.1, 0.1, 0.6), width=0.8)
    page.draw_rect(pymupdf.Rect(80, 300, 300, 360), color=(0, 0, 0), width=0.8, radius=0.15)  # a rounded box
    page.draw_line((300, 650), (520, 650), color=(0, 0, 0), width=0.8)
    page.insert_text((360, 665), "V°B° Jefatura", fontsize=9)
    doc.save(tmp_path / "sello.pdf")
    doc.close()
    assert not [f for f in _analyze(tmp_path / "sello.pdf").findings if f.type == "signature"]


def test_firma_column_of_a_text_layer_table_without_context_names(tmp_path, no_models):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), NEUTRAL, fontsize=10)
    for text, x in (("Nombre", 72), ("Correo", 250), ("Firma", 430)):
        page.insert_text((x, 200), text, fontsize=10)
    for r in range(3):
        page.insert_text((72, 225 + 25 * r), f"Persona Inventada {r}", fontsize=10)
        page.insert_text((250, 225 + 25 * r), f"persona{r}@ejemplo.cl", fontsize=10)
    doc.save(tmp_path / "tabla.pdf")
    doc.close()
    off = DetectionOptions(names_context=False, ocr=False, faces=False, qr=False)
    found = [f for f in _analyze(tmp_path / "tabla.pdf", off).findings if f.type == "signature"]
    assert len(found) == 1 and found[0].detector == signatures.DETECTOR and found[0].doubtful
    (x0, y0), (x1, y1) = np.min(found[0].polygon, axis=0), np.max(found[0].polygon, axis=0)
    assert x0 < 430 < x1 and y0 <= 205 and y1 >= 275  # the column, down to the last row


def _script_title(img: np.ndarray, rule: bool, score: float) -> list[OcrLine]:
    """A certificate: a title in a connected script font (a cursive loop per word), maybe over a rule."""
    x = 330
    for k in range(3):  # three words, each one connected stroke
        pts = cursive(x, 300, 300, 110, seed=40 + k, loops=4)
        cv2.polylines(img, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, BLACK, 4, cv2.LINE_AA)
        x += 340
    lines = [OcrLine(np.array([[325.0, 290.0], [1345.0, 290.0], [1345.0, 415.0], [325.0, 415.0]]),
                     "Certificado de Participacion", score, 0)]  # fmt: skip
    if rule:
        cv2.line(img, (300, 470), (1350, 470), BLACK, 3)
    return lines


@pytest.mark.parametrize("score", [0.95, 0.6])
@pytest.mark.parametrize("rule", [False, True])
def test_script_title_is_not_a_signature(score, rule):
    img, _ = page(1654, 2339)
    lines = _script_title(img, rule=rule, score=score) + paragraph(img, 700, rows=10)
    assert signatures.detect_raster(img, lines) == []


def _emblem(size: int = 300) -> np.ndarray:
    """A line-art seal: two rings, a star and laurel branches drawn as thin strokes."""
    img = np.full((size, size, 3), 255, np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), int(size * 0.45), (60, 40, 20), 2, cv2.LINE_AA)
    angles = np.linspace(-np.pi / 2, 3.5 * np.pi, 6)
    star = np.array([[c + 0.18 * size * np.cos(a), c + 0.18 * size * np.sin(a)] for a in angles])
    cv2.polylines(img, [np.round(star[[0, 2, 4, 1, 3, 0]]).astype(np.int32)], False, (60, 40, 20), 2, cv2.LINE_AA)
    for side in (-1, 1):
        ang = np.pi / 2 + side * (np.linspace(0.15, 0.85, 60) * np.pi * 0.8)
        stem = np.stack([c + 0.33 * size * np.cos(ang), c + 0.33 * size * np.sin(ang)], 1)
        cv2.polylines(img, [np.round(stem).astype(np.int32)], False, (60, 40, 20), 2, cv2.LINE_AA)
        for k in range(4, 60, 6):
            p, d = stem[k], stem[min(k + 1, 59)] - stem[k - 1]
            n = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
            for s in (-1, 1):
                q = p + s * n * size * 0.05
                axes = (int(size * 0.035), int(size * 0.014))
                angle = float(np.degrees(np.arctan2(n[1], n[0])))
                cv2.ellipse(img, (int(q[0]), int(q[1])), axes, angle, 0, 360, (60, 40, 20), 1, cv2.LINE_AA)
    return img


def test_an_emblem_repeated_on_every_page_is_not_a_signature(tmp_path, no_models):
    buf = io.BytesIO()
    Image.fromarray(_emblem()[:, :, ::-1]).save(buf, "PNG")
    doc = pymupdf.open()
    xref = 0
    for _ in range(3):
        page = _text_page(doc)
        rect = pymupdf.Rect(40, 20, 100, 80)
        xref = page.insert_image(rect, xref=xref) if xref else page.insert_image(rect, stream=buf.getvalue())
    doc.save(tmp_path / "membrete.pdf")
    doc.close()
    assert not [f for f in _analyze(tmp_path / "membrete.pdf").findings if f.type == "signature"]
    # A signature image placed once, next to the repeated emblem, is still looked at.
    doc = pymupdf.open(tmp_path / "membrete.pdf")
    doc[1].insert_image(pymupdf.Rect(330, 590, 480, 640), stream=_signature_png())
    doc.save(tmp_path / "una.pdf")
    doc.close()
    found = [f for f in _analyze(tmp_path / "una.pdf").findings if f.type == "signature"]
    assert len(found) == 1 and found[0].page == 1


def test_vector_pass_reports_progress_and_can_be_cancelled(tmp_path, no_models):
    import threading

    doc = pymupdf.open()
    for _ in range(3):
        _vector_signature(_text_page(doc), 300, 500, 200, 60)
    doc.save(tmp_path / "a.pdf")
    doc.close()
    steps = []
    file = AnalyzedFile(id="f1", name="a.pdf", path=str(tmp_path / "a.pdf"))
    RealEngine().analyze(file, [], progress=lambda fraction, step: steps.append(step))
    assert any("firmas dibujadas" in s for s in steps)
    cancel = threading.Event()

    def stop(fraction, step):
        if "firmas dibujadas" in step:
            cancel.set()

    file = AnalyzedFile(id="f2", name="a.pdf", path=str(tmp_path / "a.pdf"))
    RealEngine().analyze(file, [], progress=stop, cancel=cancel)
    assert file.status == "cancelled"


def _repeated_signature_pdf(path: Path, pages: int, anchor: str | None) -> Path:
    """The same signature image at the same place on every page, maybe with a label or a line."""
    doc = pymupdf.open()
    png = _signature_png()
    xref = 0
    for _ in range(pages):
        page = _text_page(doc)
        box = pymupdf.Rect(330, 590, 480, 640)
        xref = page.insert_image(box, xref=xref) if xref else page.insert_image(box, stream=png)
        if anchor == "label":
            page.insert_text((360, 660), "Firma del responsable", fontsize=9)
        elif anchor == "line":
            page.draw_line((310, 645), (500, 645), color=(0, 0, 0), width=0.8)
            page.insert_text((360, 660), "Juan Inventado Soto", fontsize=9)
        elif anchor == "name":
            page.insert_text((360, 655), "Juan Inventado Soto", fontsize=9)
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("anchor", ["label", "line", "name"])
def test_a_signature_repeated_on_every_page_next_to_an_anchor(tmp_path, no_models, anchor):
    # Certificates signed by the same official, initials on every sheet: one finding per page.
    path = _repeated_signature_pdf(tmp_path / "a.pdf", 3, anchor)
    found = [f for f in _analyze(path).findings if f.type == "signature"]
    assert sorted(f.page for f in found) == [0, 1, 2]


def test_a_repeated_image_with_nothing_next_to_it_is_left_alone(tmp_path, no_models):
    path = _repeated_signature_pdf(tmp_path / "a.pdf", 3, None)
    assert not [f for f in _analyze(path).findings if f.type == "signature"]
    once = _repeated_signature_pdf(tmp_path / "b.pdf", 1, None)  # placed once, it is looked at
    assert [f for f in _analyze(once).findings if f.type == "signature"]


def test_a_tightly_cropped_signature_next_to_a_label_is_found():
    # The keyword frame of an image region: the stroke spans most of the image, which is not a
    # frame or a chart of the page.
    img, mask = page(420, 140)
    draw_signature(img, mask, 8, 10, 400, 110, seed=9)
    label = OcrLine(np.array([[60.0, 150.0], [300.0, 150.0], [300.0, 170.0], [60.0, 170.0]]), "Firma", 1.0, 0)
    assert not signatures.detect_raster(img, [label], lone=False)
    zones = signatures.detect_raster(img, [label], lone=False, page_size=(1654, 2339))
    assert_signature_zones(zones)
    assert covered(zones, mask) >= 0.95


def test_a_repeated_logotype_over_the_header_rule_is_not_a_signature(tmp_path, no_models):
    # A cursive logotype in the letterhead of every page, with the page-wide header rule under it.
    logo = _signature_png(width=360, height=120, seed=4)
    doc = pymupdf.open()
    xref = 0
    for _ in range(3):
        page = _text_page(doc)
        box = pymupdf.Rect(72, 20, 192, 60)
        xref = page.insert_image(box, xref=xref) if xref else page.insert_image(box, stream=logo)
        page.draw_line((72, 66), (523, 66), color=(0.2, 0.2, 0.2), width=1)
    doc.save(tmp_path / "membrete.pdf")
    doc.close()
    assert not [f for f in _analyze(tmp_path / "membrete.pdf").findings if f.type == "signature"]


@pytest.mark.parametrize("anchor", ["label", "line"])
def test_a_repeated_signature_on_rotated_pages(tmp_path, no_models, anchor):
    # Landscape pages stored as portrait plus /Rotate 90: "under the image" is in the page as shown.
    doc = pymupdf.open()
    png = _signature_png()
    xref = 0
    for _ in range(3):
        page = doc.new_page(width=595, height=842)
        page.set_rotation(90)
        to_page = page.derotation_matrix
        for i in range(6):
            page.insert_text(pymupdf.Point(72, 100 + 18 * i) * to_page, NEUTRAL, fontsize=10, rotate=90)
        box = (pymupdf.Rect(560, 390, 710, 440) * to_page).normalize()
        xref = page.insert_image(box, xref=xref, rotate=90) if xref else page.insert_image(box, stream=png, rotate=90)
        if anchor == "line":
            page.draw_line(
                pymupdf.Point(540, 445) * to_page, pymupdf.Point(730, 445) * to_page, color=(0, 0, 0), width=0.8
            )
            page.insert_text(pymupdf.Point(590, 460) * to_page, "Juan Inventado Soto", fontsize=9, rotate=90)
        else:
            page.insert_text(pymupdf.Point(590, 460) * to_page, "Firma del responsable", fontsize=9, rotate=90)
    doc.save(tmp_path / "girada.pdf")
    doc.close()
    found = [f for f in _analyze(tmp_path / "girada.pdf").findings if f.type == "signature"]
    assert sorted(f.page for f in found) == [0, 1, 2]
