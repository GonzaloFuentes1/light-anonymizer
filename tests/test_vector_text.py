"""D8: text converted to vector paths on pages that also have a text layer. Invented data only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pymupdf
import pytest

from anonymizer.engine import common, estimate, faces, pdf, raster, vectors
from anonymizer.engine.model import AnalyzedFile, DetectionOptions
from anonymizer.engine.patterns import rut_check_digit
from anonymizer.engine.real import RealEngine

needs_ocr = pytest.mark.skipif(
    importlib.util.find_spec("rapidocr") is None or not faces.available(), reason="needs the OCR and face models"
)

VALID_RUT = f"12.345.678-{rut_check_digit('12345678')}"
EMAIL = "ana.prueba@ejemplo.cl"
NEUTRAL = [
    "Informe ficticio de prueba para el motor de anonimización, con texto neutro de relleno.",
    "Este párrafo es texto normal: se puede seleccionar y copiar desde el documento.",
]


def vector_text_pdf(path: Path, drawn: list[str], y: float = 420) -> Path:
    """A page with a normal text layer on top and ``drawn`` lines below, converted to paths (as a PDF
    exported from a design tool, where a block of text became curves)."""
    src = pymupdf.open()
    page = src.new_page(width=595, height=842)
    for i, line in enumerate(drawn):
        page.insert_text((72, y + 26 * i), line, fontsize=14)
    svg = page.get_svg_image(text_as_path=True)
    with pymupdf.open("svg", svg.encode("utf-8")) as as_svg:
        doc = pymupdf.open("pdf", as_svg.convert_to_pdf())
    src.close()
    page = doc[0]
    for i, line in enumerate(NEUTRAL):
        page.insert_text((72, 100 + 22 * i), line, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def test_drawn_text_is_found_and_ordinary_drawings_are_not(tmp_path):
    path = vector_text_pdf(tmp_path / "a.pdf", [f"Correo: {EMAIL}", f"RUT: {VALID_RUT}"])
    with pymupdf.open(path) as doc:
        page = doc[0]
        tp = pdf.read_text_page(page, ())
        assert not tp.scanned  # it has a text layer
        regions = vectors.text_regions(page, tp.boxes)
    assert len(regions) == 1
    r = regions[0]
    assert r.y0 < 420 - 14 and r.y1 > 420 + 26 and r.x0 < 80 and r.y1 < 500  # the drawn block, not the text

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i, line in enumerate(NEUTRAL):
        page.insert_text((72, 100 + 22 * i), line, fontsize=11)
    for i in range(12):  # a table: lines and cell fills
        page.draw_rect(pymupdf.Rect(72, 200 + 20 * i, 300, 220 + 20 * i), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    for i in range(6):  # bullets: small filled circles
        page.draw_circle((80, 500 + 20 * i), 2.5, fill=(0, 0, 0))
    assert vectors.text_regions(page, pdf.read_text_page(page, ()).boxes) == []
    doc.close()


def test_a_text_page_without_drawn_text_is_not_rendered(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(raster, "detect_in_image", lambda *a, **k: calls.append(1) or [])
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i, line in enumerate([*NEUTRAL, f"Correo: {EMAIL}"]):
        page.insert_text((72, 100 + 22 * i), line, fontsize=11)
    page.draw_rect(pymupdf.Rect(60, 90, 540, 180), color=(0, 0, 0))
    doc.save(tmp_path / "texto.pdf")
    doc.close()
    file = AnalyzedFile(id="f1", name="texto.pdf", path=str(tmp_path / "texto.pdf"))
    RealEngine().analyze(file, [])
    assert file.status == "ready" and any(f.type == "email" for f in file.findings)
    assert calls == [] and "render" not in file.timings


@needs_ocr
def test_drawn_text_is_read_redacted_and_its_paths_removed(tmp_path):
    path = vector_text_pdf(tmp_path / "a.pdf", [f"Correo: {EMAIL}", f"RUT: {VALID_RUT}"])
    engine = RealEngine()
    file = AnalyzedFile(id="f1", name="a.pdf", path=str(path), options=DetectionOptions().to_dict())
    engine.analyze(file, [])
    assert file.status == "ready"
    drawn = {f.type: f for f in file.findings if f.detector == "ocr"}
    assert {"email", "rut"} <= set(drawn)
    for f in drawn.values():
        x0, y0, x1, y1 = common.bbox_of(f.polygon)
        assert 380 < y0 and y1 < 470  # where the drawn lines are (view space = page space here)
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        page = doc[0]
        assert NEUTRAL[0] in page.get_text()  # the text layer is still text
        left = vectors.glyph_paths(page)
    for f in drawn.values():
        zone = pymupdf.Rect(common.bbox_of(f.polygon))
        assert not [g for g in left if g.intersects(zone)]  # no letter drawn as a path is left under it


def test_the_time_estimate_counts_drawn_text(tmp_path):
    facts = estimate.profile(str(vector_text_pdf(tmp_path / "a.pdf", [f"Correo: {EMAIL}", f"RUT: {VALID_RUT}"])))
    assert facts["drawn_regions"] == 1 and facts["raster_pages"] == 1 and facts["drawn_regions_mp"] > 0
    assert estimate.units(facts)["ocr"]["region"] == 1 and estimate.units(facts)["faces"]["region"] == 0
