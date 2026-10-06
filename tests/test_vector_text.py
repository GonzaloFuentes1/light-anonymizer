"""D8: text converted to vector paths on pages that also have a text layer. Invented data only."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pymupdf
import pytest

from anonymizer.engine import common, estimate, faces, pdf, raster, vectors, verify
from anonymizer.engine.model import AnalyzedFile, DetectionOptions, Finding, HistoryEntry
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


def one_path_pdf(path: Path, drawn: list[str], y: float = 420) -> Path:
    """Like ``vector_text_pdf``, but every letter of the drawn lines is a subpath of ONE path with one
    fill, as some drawing programs write it."""
    src = vector_text_pdf(path.with_name("per_letter_" + path.name), drawn, y)
    with pymupdf.open(src) as per_letter:
        paths = [d for d in per_letter[0].get_drawings() if d.get("fill") is not None]
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    shape = page.new_shape()
    for d in paths:
        for item in d["items"]:
            if item[0] == "l":
                shape.draw_line(item[1], item[2])
            elif item[0] == "c":
                shape.draw_bezier(item[1], item[2], item[3], item[4])
    shape.finish(fill=(0, 0, 0), color=None, even_odd=True, closePath=False)
    shape.commit()
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


def test_a_drawn_block_written_as_one_path_is_found(tmp_path):
    path = one_path_pdf(tmp_path / "one.pdf", [f"Correo: {EMAIL}", f"RUT: {VALID_RUT}"])
    with pymupdf.open(path) as doc:
        page = doc[0]
        assert len([d for d in page.get_drawings() if d.get("fill") is not None]) == 1
        regions = vectors.text_regions(page, pdf.read_text_page(page, ()).boxes)
    assert len(regions) == 1 and regions[0].y0 < 406 and regions[0].y1 > 446


@needs_ocr
def test_a_drawn_block_written_as_one_path_is_read_and_its_letters_removed(tmp_path):
    path = one_path_pdf(tmp_path / "one.pdf", [f"Correo: {EMAIL}", f"RUT: {VALID_RUT}", "Texto neutro de relleno"])
    engine = RealEngine()
    file = AnalyzedFile(id="f1", name="one.pdf", path=str(path), options=DetectionOptions().to_dict())
    engine.analyze(file, [])
    drawn = {f.type: f for f in file.findings if f.detector == "ocr"}
    assert {"email", "rut"} <= set(drawn)
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        left = vectors.glyph_paths(doc[0])
    for f in drawn.values():
        assert not vectors.under(pymupdf.Rect(common.bbox_of(f.polygon)), left)
    assert any(g.y0 > 460 for g in left)  # the neutral line, not redacted, is still drawn


def test_a_reviewer_zone_a_little_short_still_removes_whole_letters(tmp_path):
    # A zone drawn by the reviewer that misses the bottom 0.6 pt of the line: MuPDF would leave
    # every letter it does not cover whole. The zone is grown to the letters on export.
    path = vector_text_pdf(tmp_path / "a.pdf", [f"Correo: {EMAIL}"])
    with pymupdf.open(path) as doc:
        line = [g for g in vectors.glyph_paths(doc[0]) if 400 < g.y1 < 430]
    x0, y0 = min(g.x0 for g in line), min(g.y0 for g in line)
    x1, y1 = max(g.x1 for g in line), max(g.y1 for g in line)
    zone = common.rect_polygon(x0 - 2, y0 - 2, x1 + 2, y1 - 0.6)
    drawn = Finding(id="m1", file_id="b", page=0, type="manual", polygon=zone, detector="reviewer", status="added",
                    history=[HistoryEntry(at="2026-10-05T10:00:00+00:00", action="added")])  # fmt: skip
    file = AnalyzedFile(id="b", name="a.pdf", path=str(path), kind="pdf", findings=[drawn], status="confirmed")
    # Before the fix of this case, the leak check saw nothing: it now looks for letters left under a zone.
    assert verify.glyph_leaks(path, [drawn])
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        assert not [g for g in vectors.glyph_paths(doc[0]) if 400 < g.y1 < 430]


def test_letters_left_under_an_applied_zone_are_a_leak(tmp_path, monkeypatch):
    path = vector_text_pdf(tmp_path / "a.pdf", [f"Correo: {EMAIL}"])
    with pymupdf.open(path) as doc:
        line = [g for g in vectors.glyph_paths(doc[0]) if 400 < g.y1 < 430]
    zone = common.rect_polygon(min(g.x0 for g in line), min(g.y0 for g in line), max(g.x1 for g in line), 420)
    drawn = Finding(id="m1", file_id="b", page=0, type="manual", polygon=zone, detector="reviewer", status="added")
    file = AnalyzedFile(id="b", name="a.pdf", path=str(path), kind="pdf", findings=[drawn], status="confirmed")
    monkeypatch.setattr(vectors, "snap", lambda rect, glyphs: rect)  # as if the zone were not grown
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert not result.exported
    assert any("trazos" in leak.message and leak.finding_id == "m1" for leak in result.leaks)


def test_many_small_shapes_are_grouped_quickly(tmp_path):
    # A map or a chart with 20 000 markers: grouping them must not take seconds (it runs on every page).
    rects = [pymupdf.Rect(40 + (k % 200) * 11, 120 + (k // 200) * 11, 43 + (k % 200) * 11, 123 + (k // 200) * 11)
             for k in range(20000)]  # fmt: skip
    started = time.perf_counter()
    groups = vectors.clusters(rects)
    assert time.perf_counter() - started < 3
    assert len(groups) == 20000  # 8 pt apart: each marker on its own
    dense = [pymupdf.Rect(40 + (k % 80) * 6.5, 40 + (k // 80) * 14, 45.5 + (k % 80) * 6.5, 50 + (k // 80) * 14)
             for k in range(3000)]  # fmt: skip
    assert [count for _, count in vectors.clusters(dense)] == [3000]  # lines of letters: one block
