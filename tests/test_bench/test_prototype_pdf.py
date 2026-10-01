"""Regression cases of the prototype over small PDFs built here (all data is made up)."""

import importlib.util
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw

from test_bench.baselines import prototype
from test_bench.canvas import font

NAME = "Marcela Ibarra Contreras"
RUT = "12.345.678-5"
PHONE = "+56 9 8123 4567"


def _run(tmp_path: Path, doc: pymupdf.Document, name_list: tuple[str, ...] = ()) -> tuple[str, list]:
    source, dest = tmp_path / "in.pdf", tmp_path / "out.pdf"
    doc.save(source)
    doc.close()
    redactions, _ = prototype._process_pdf(source, dest, name_list)
    with pymupdf.open(dest) as out:
        text = "\n".join(p.get_text() for p in out)
    return text, redactions


def test_role_line_under_redacted_name_survives(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Informe de actividades del mes de marzo de 2026.", fontsize=11)
    # Tight leading: the 12 pt name box invades the 9 pt line printed right under it.
    page.insert_text((72, 140), NAME.upper(), fontsize=12, fontname="hebo")
    page.insert_text((72, 149), "FUNCIONARIA", fontsize=9)
    text, redactions = _run(tmp_path, doc, (NAME,))
    assert any(r.type == "name" for r in redactions)
    assert "IBARRA" not in text and "MARCELA" not in text
    assert "FUNCIONARIA" in text


@pytest.mark.parametrize("rotate", [90, 270])
def test_vertical_rut_is_redacted(tmp_path, rotate):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Certificado de recepción de documentos.", fontsize=11)
    # Two vertical lines with tight spacing, like a signature printed on the margin.
    x = 40 if rotate == 90 else 560
    step = 9 if rotate == 90 else -9
    page.insert_text((x, 300), "Firmado electrónicamente por la oficina de partes", fontsize=9, rotate=rotate)
    page.insert_text((x + step, 300), f"RUT {RUT} fecha 3 de junio de 2026", fontsize=9, rotate=rotate)
    text, redactions = _run(tmp_path, doc)
    assert any(r.type == "rut" for r in redactions)
    assert "345" not in text
    assert "oficina de partes" in text


def test_tilted_stamp_rut_is_redacted(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    morph = (pymupdf.Point(300, 500), pymupdf.Matrix(30))
    for i, line in enumerate(["OFICINA DE PARTES", "RECIBIDO 26 de octubre de 2026", f"RUT {RUT}"]):
        page.insert_text((300, 500 + 10 * i), line, fontsize=10, morph=morph)
    text, _ = _run(tmp_path, doc)
    assert "345" not in text


def _boxes(*rects):
    return [
        ("face", np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), "", "yunet", 0.9)
        for x0, y0, x1, y1 in rects
    ]


def test_overlapping_faces_are_merged_into_their_union():
    zones = _boxes((100, 100, 200, 220), (104, 96, 206, 224), (110, 110, 190, 200), (400, 100, 480, 200))
    merged = prototype._merge_faces(zones)
    assert len(merged) == 2
    boxes = sorted(tuple(z[1].min(axis=0)) + tuple(z[1].max(axis=0)) for z in merged)
    assert boxes[0] == (100, 96, 206, 224)  # union, never smaller than any of the detections
    assert boxes[1] == (400, 100, 480, 200)


def test_separate_faces_are_not_merged():
    zones = _boxes((100, 100, 200, 200), (190, 100, 290, 200))  # touching, small overlap
    assert len(prototype._merge_faces(zones)) == 2


@pytest.mark.skipif(importlib.util.find_spec("rapidocr") is None, reason="needs the OCR models (rapidocr)")
def test_phone_inside_small_image_of_a_text_pdf(tmp_path):
    img = Image.new("RGB", (220, 60), "white")
    ImageDraw.Draw(img).text((8, 18), PHONE, font=font("sans", 22), fill=(0, 0, 0))
    png = tmp_path / "phone.png"
    img.save(png)
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(12):
        page.insert_text((72, 80 + 16 * i), f"Párrafo {i + 1} del informe con texto neutro de relleno.", fontsize=11)
    page.insert_image(pymupdf.Rect(72, 300, 182, 330), filename=str(png))  # about 2 % of the page
    _, redactions = _run(tmp_path, doc)
    # The OCR line may be reported as "rut" too (9 digits without prefix): what matters is that it is covered.
    phones = [r for r in redactions if r.detector == "ocr" and "4567" in (r.text or "")]
    assert phones
    xs = [p[0] for r in phones for p in r.polygon]
    ys = [p[1] for r in phones for p in r.polygon]
    assert min(xs) >= 60 and max(xs) <= 195 and min(ys) >= 290 and max(ys) <= 340  # mapped back to the image


def test_same_image_on_every_page_is_read_once(tmp_path, monkeypatch):
    png = tmp_path / "logo.png"
    Image.new("RGB", (120, 60), (20, 60, 140)).save(png)
    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page()
        page.insert_text((72, 120), "Texto neutro de una página del informe con un logo arriba.", fontsize=11)
        page.insert_image(pymupdf.Rect(72, 40, 132, 70), filename=str(png))
    calls = []

    def fake_detect(rgb, name_list, ocr=True, **kw):
        calls.append(kw.get("qr", True))
        return []

    monkeypatch.setattr(prototype, "detect_in_image", fake_detect)
    _run(tmp_path, doc)
    assert calls.count(False) == 1  # one OCR of the image region for the three pages
    assert calls.count(True) == 3  # QR codes are still searched on every page
