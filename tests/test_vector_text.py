"""D8: text converted to vector paths on pages that also have a text layer. Invented data only."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from anonymizer.engine import audit, common, estimate, faces, pdf, raster, vectors, verify
from anonymizer.engine.common import Zone
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
        assert not vectors.under(zone, left)  # no letter drawn as a path is left under it


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
    monkeypatch.setattr(pdf, "snap_rects", lambda page, rects, keep=(): [[r] for r in rects])  # not grown
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


def drawn_page(path: Path, lines: list[tuple[float, float, str, float]]) -> Path:
    """A page with a text layer and the given lines drawn as paths: (x, baseline, text, size)."""
    src = pymupdf.open()
    page = src.new_page(width=595, height=842)
    for x, y, text, size in lines:
        page.insert_text((x, y), text, fontsize=size)
    with pymupdf.open("svg", page.get_svg_image(text_as_path=True).encode("utf-8")) as as_svg:
        doc = pymupdf.open("pdf", as_svg.convert_to_pdf())
    src.close()
    for i, line in enumerate(NEUTRAL):
        doc[0].insert_text((72, 100 + 22 * i), line, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def letters_of(path) -> list[tuple[float, float, float, float]]:
    with pymupdf.open(path) as doc:
        return [tuple(round(v, 2) for v in g) for g in vectors.letters(doc[0])]


def export_zones(path: Path, zones: dict[str, tuple], out: Path, kept: dict[str, tuple] | None = None):
    """Exports ``path`` with active OCR-like findings on ``zones`` and findings left visible on ``kept``."""
    findings = [
        Finding(id=fid, file_id="z", page=0, type="email", polygon=common.rect_polygon(*z), text="x", detector="ocr")
        for fid, z in zones.items()
    ]
    findings += [
        Finding(id=fid, file_id="z", page=0, type="url", polygon=common.rect_polygon(*z), text="y", detector="ocr",
                status="suggested", optional=True, optional_reason="url")
        for fid, z in (kept or {}).items()
    ]  # fmt: skip
    file = AnalyzedFile(id="z", name=path.name, path=str(path), kind="pdf", findings=findings, status="confirmed")
    return file, RealEngine().export(file, str(out))


BLOCK = [
    "Primera linea neutra del bloque dibujado, sin datos de nadie.",
    "Segunda linea neutra: texto de relleno para la prueba.",
    "Contacto: ana.prueba@ejemplo.cl fono +56 9 8123 4567",
    "Cuarta linea neutra, tambien de relleno y sin datos.",
    "Quinta linea neutra que cierra el bloque de prueba.",
]


@pytest.mark.parametrize(
    "size,leading,zone",
    [
        # The OCR boxes of the middle line measured in the review (font 10 and 12, single spacing).
        (10, 10, (68.5, 429.2, 326.1, 444.9)),
        (12, 12, (68.8, 430.6, 377.6, 449.2)),
    ],
)
def test_only_the_letters_under_a_zone_are_removed(tmp_path, size, leading, zone):
    path = drawn_page(tmp_path / "block.pdf", [(72, 420 + leading * i, line, size) for i, line in enumerate(BLOCK)])
    before = letters_of(path)
    _, result = export_zones(path, {"z1": zone}, tmp_path / "out")
    assert result.exported, [leak.message for leak in result.leaks]
    after = set(letters_of(result.output_path))

    def line_of(g) -> int:  # the line whose letters' middle is closest
        return min(range(5), key=lambda i: abs(420 + leading * i - size * 0.3 - (g[1] + g[3]) / 2))

    lines = [[g for g in before if line_of(g) == i] for i in range(5)]
    assert not [g for g in lines[2] if g in after]  # the line with the data is gone
    for i in (0, 1, 3, 4):  # the neighbours keep their letters (before the fix, lines 1 to 3 went)
        assert sum(g in after for g in lines[i]) >= 0.9 * len(lines[i]), (i, sum(g in after for g in lines[i]))
    # Besides the letters under the zone, the dots of the next line's i's (and its periods) that fall
    # entirely inside the zone go too: MuPDF removes every subpath a rectangle covers whole, so such
    # an "i" shows here as changed (its box loses the dot).
    removed = [pymupdf.Rect(g) for g in before if g not in after]
    stray = [g for g in removed if g not in vectors.under(pymupdf.Rect(zone), removed)]
    assert all(min(g.width, g.height) <= 1.5 for g in stray), stray


def test_a_zone_whose_edges_fall_exactly_on_letters_removes_them(tmp_path):
    # The union of the letters of a line: its edges are those of letters, and MuPDF does not count a
    # shape as covered when the edges are equal. The rectangles added per letter are a little larger.
    path = drawn_page(tmp_path / "block.pdf", [(72, 420 + 12.65 * i, line, 11) for i, line in enumerate(BLOCK)])
    middle = [g for g in letters_of(path) if abs((g[1] + g[3]) / 2 - (420 + 12.65 * 2 - 3.3)) < 6]
    zone = (min(g[0] for g in middle), min(g[1] for g in middle), max(g[2] for g in middle), max(g[3] for g in middle))
    _, result = export_zones(path, {"z1": zone}, tmp_path / "out")
    assert result.exported, [leak.message for leak in result.leaks]
    after = set(letters_of(result.output_path))
    assert not [g for g in middle if g in after]


def test_a_zone_touching_the_ascenders_of_the_next_line_leaves_them(tmp_path):
    path = drawn_page(tmp_path / "two.pdf", [(72, 420, BLOCK[1], 12), (72, 434, BLOCK[2], 12)])
    before = letters_of(path)
    second = [g for g in before if g[3] > 428]
    top = min(g[1] for g in second)
    # A reviewer zone over the second line whose top edge goes 1.5 pt into the first line's letters.
    first_bottom = max(g[3] for g in before if g[3] <= 428)
    zone = (60, first_bottom - 1.5, 400, max(g[3] for g in second) + 1)
    assert zone[1] < top
    _, result = export_zones(path, {"z1": zone}, tmp_path / "out")
    assert result.exported
    after = set(letters_of(result.output_path))
    assert all(g in after for g in before if g[3] <= 428)  # the first line is untouched


def test_a_chart_next_to_a_zone_keeps_its_bars(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i, line in enumerate(NEUTRAL):
        page.insert_text((72, 300 + 22 * i), line, fontsize=11)
    shape = page.new_shape()
    for k in range(12):
        x, h = 80 + k * 20, 10 + (k * 7) % 35
        shape.draw_polyline([(x, 260), (x + 12, 260), (x + 12, 260 - h), (x, 260 - h), (x, 260)])
        shape.finish(fill=(0.2, 0.4, 0.8), color=None)
    shape.commit()
    page.insert_text((80, 275), "Ventas por mes (ficticio)  Responsable: Ana Prueba", fontsize=9)
    doc.save(tmp_path / "chart.pdf")
    doc.close()
    before = letters_of(tmp_path / "chart.pdf")
    zone = (75, 252, 330, 278)
    _, result = export_zones(tmp_path / "chart.pdf", {"z1": zone}, tmp_path / "out")
    assert result.exported
    after = set(letters_of(result.output_path))
    removed = [pymupdf.Rect(g) for g in before if g not in after]
    assert vectors.under(pymupdf.Rect(zone), removed) == removed  # only bars mostly inside the zone
    assert sum(1 for g in before if g in after) >= 6  # the tall bars stay (all 12 were erased before)


def test_letters_in_an_area_the_reviewer_kept_visible_are_never_covered(tmp_path):
    path = drawn_page(tmp_path / "a.pdf", [(72, 420, f"Correo: {EMAIL} y www.goreficticio.cl", 14)])
    with pymupdf.open(path) as doc:
        page = doc[0]
        line = sorted(vectors.letters(page), key=lambda g: g.x0)
        mid = line[len(line) // 2]
        active = pymupdf.Rect(line[0].x0, mid.y0 - 1, (mid.x0 + mid.x1) / 2 + 0.5, mid.y1 + 1)
        kept = pymupdf.Rect(mid.x1 - 0.5, mid.y0 - 1, line[-1].x1, mid.y1 + 1)
        groups = pdf.snap_rects(page, [active], keep=[kept])
    for r in groups[0][1:]:
        assert not r.intersects(kept)


def test_fill_and_stroke_chart_markers_do_not_block_the_export(tmp_path):
    # Chart markers drawn as filled and outlined circles, next to a name: they are not letters.
    for stroke in (None, (0.3, 0.2, 0)):
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), NEUTRAL[0], fontsize=11)
        shape = page.new_shape()
        points = [(100 + 40 * k, 300 - (k * 13) % 60) for k in range(8)]
        for x, y in points:
            shape.draw_circle((x, y), 3)
            shape.finish(fill=(1, 0.6, 0), color=stroke, width=1.0)
        shape.commit()
        x, y = points[3]
        page.insert_text((x + 2, y + 3), "Quintanilla Brito", fontsize=8)
        path = tmp_path / f"markers_{stroke is None}.pdf"
        doc.save(path)
        doc.close()
        engine = RealEngine()
        options = DetectionOptions(ocr=False, faces=False, qr=False).to_dict()
        file = AnalyzedFile(id="m", name=path.name, path=str(path), options=options)
        engine.analyze(file, ["Quintanilla Brito"])
        assert any(f.type == "name" for f in file.findings)
        result = engine.export(file, str(tmp_path / f"out_{stroke is None}"))
        assert result.exported, (stroke, [leak.message for leak in result.leaks])


def test_two_overlapping_areas_of_drawn_text_are_both_read(tmp_path):
    lines = [(72, 380, "Datos de la solicitud recibida por la oficina ficticia", 14)]
    lines += [(72, 398 + 14 * i, t, 11) for i, t in enumerate(["Columna izquierda de relleno",
              "sin datos personales, texto", "neutro para la prueba."])]  # fmt: skip
    lines += [(300, 432 + 14 * i, t, 11) for i, t in enumerate(["Columna derecha que empieza",
              "mas abajo y llega mas lejos:", "escribir a la persona en", f"{EMAIL} hoy"])]  # fmt: skip
    path = drawn_page(tmp_path / "two_areas.pdf", lines)
    with pymupdf.open(path) as doc:
        page = doc[0]
        regions = vectors.text_regions(page, pdf.read_text_page(page, ()).boxes)
    email_area = pymupdf.Rect(300, 470, 450, 482)  # the last line of the right column
    assert any(r.contains(email_area) for r in regions), regions
    assert any(r.contains(pymupdf.Rect(72, 368, 380, 430)) for r in regions)


def test_the_audit_lists_the_letters_a_zone_was_grown_to(tmp_path):
    path = drawn_page(tmp_path / "a.pdf", [(72, 420, f"Correo: {EMAIL}", 14)])
    with pymupdf.open(path) as doc:
        line = vectors.letters(doc[0])
    zone = (min(g.x0 for g in line) - 2, min(g.y0 for g in line) + 1, max(g.x1 for g in line) + 2, 421)
    file, result = export_zones(path, {"z1": zone}, tmp_path / "out")
    assert result.exported and result.grown and result.grown[0]["finding_id"] == "z1"
    record = audit.build_report([file], [result])["files"][0]
    assert record["letters_covered"][0]["finding_id"] == "z1" and record["letters_covered"][0]["rects"]


def test_the_analysis_does_not_grow_ocr_zones(tmp_path, monkeypatch):
    # Zones are grown once, when they are applied (``pdf.snap_rects``), not already in the analysis.
    path = vector_text_pdf(tmp_path / "a.pdf", [f"Correo: {EMAIL}", f"RUT: {VALID_RUT}"])
    box = np.array([[10.0, 10.0], [300.0, 10.0], [300.0, 22.0], [10.0, 22.0]])  # pixels of the crop
    monkeypatch.setattr(raster, "detect_in_image", lambda *a, **k: [Zone("email", box, EMAIL, "ocr", 0.99)])
    file = AnalyzedFile(id="f1", name="a.pdf", path=str(path), options=DetectionOptions(qr=False).to_dict())
    RealEngine().analyze(file, [])
    (zone,) = [f for f in file.findings if f.detector == "ocr"]
    x0, y0, x1, y1 = common.bbox_of(zone.polygon)
    assert y1 - y0 <= 12 * 72 / 200 + 2.01  # the 12 px box at 200 dpi plus the 1 pt margin


def drawn_name(path: Path, text: str, size: float, font: str) -> Path:
    """A landscape page with ``text`` drawn as paths, large, in ``font``, and a text layer above."""
    src = pymupdf.open()
    page = src.new_page(width=842, height=595)
    page.insert_text((40, 300), text, fontsize=size, fontname=font)
    with pymupdf.open("svg", page.get_svg_image(text_as_path=True).encode("utf-8")) as as_svg:
        doc = pymupdf.open("pdf", as_svg.convert_to_pdf())
    src.close()
    for i, line in enumerate(NEUTRAL):
        doc[0].insert_text((40, 60 + 22 * i), line, fontsize=11)
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize(
    "font,size,text",
    [("heit", 64, "Ana Pellegrini"), ("helv", 64, "Ana Pellegrini"), ("tiit", 64, "Fulvia Pellegrini"),
     ("heit", 30, "ana.pellegrini@ficticio.cl")],
)  # fmt: skip
def test_large_drawn_letters_under_a_generous_zone_are_all_removed(tmp_path, font, size, text):
    # Outlines over 40 pt are not "letters"; a neighbouring letter's rectangle used to touch them
    # first without covering them, and MuPDF then kept them although the zone covered them whole.
    path = drawn_name(tmp_path / "name.pdf", text, size, font)
    with pymupdf.open(path) as doc:
        shapes = [s.box for s in vectors.shapes(doc[0], max_side=None) if s.box.y1 > 200]
    zone = pymupdf.Rect(min(s.x0 for s in shapes), min(s.y0 for s in shapes), max(s.x1 for s in shapes),
                        max(s.y1 for s in shapes)) + (-4, -4, 4, 4)  # fmt: skip
    _, result = export_zones(path, {"z1": tuple(zone)}, tmp_path / "out")
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        left = [s.box for s in vectors.shapes(doc[0], max_side=None) if zone.contains(s.box)]
    assert left == []


def test_a_large_shape_left_inside_a_zone_is_a_leak(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), NEUTRAL[0], fontsize=11)
    shape = page.new_shape()
    shape.draw_polyline([(100, 300), (160, 300), (130, 360), (100, 300)])  # a 60 pt drawn shape
    shape.finish(fill=(0.1, 0.1, 0.1), color=None)
    shape.commit()
    doc.save(tmp_path / "big.pdf")
    doc.close()
    zone = Finding(id="z1", file_id="b", page=0, type="manual", polygon=common.rect_polygon(95, 295, 165, 365),
                   detector="reviewer", status="added")  # fmt: skip
    # On the unredacted file the shape is still there: that is what a failed removal looks like.
    assert verify.glyph_leaks(tmp_path / "big.pdf", [zone])
    file = AnalyzedFile(id="b", name="big.pdf", path=str(tmp_path / "big.pdf"), kind="pdf", findings=[zone],
                        status="confirmed")  # fmt: skip
    assert RealEngine().export(file, str(tmp_path / "out")).exported  # the zone alone removes it


def test_letters_of_a_kept_line_inside_the_zone_neither_block_nor_go(tmp_path):
    # Single spacing: the OCR box of the data line reaches the line above, a URL left visible; the
    # periods of that URL have their centre inside the zone. They belong to the URL (measured boxes
    # of the review, font 10 and 12).
    lines = ["Primera linea neutra del bloque dibujado, sin datos de nadie.",
             "Sitio institucional: www.goreficticio.cl/tramites",
             "Contacto: ana.prueba@ejemplo.cl fono +56 9 8123 4567",
             "Mesa central: 600 123 4567",
             "Quinta linea neutra que cierra el bloque de prueba."]  # fmt: skip
    cases = {
        10: ((68.8, 418.8, 277.1, 434.1), (68.5, 429.2, 326.1, 444.9), (68.5, 438.9, 199.7, 455.0)),
        12: ((68.8, 419.1, 316.7, 437.3), (68.5, 431.4, 377.2, 449.6), (69.2, 444.0, 223.5, 460.0)),
    }
    for size, (url, data, phone) in cases.items():
        path = drawn_page(tmp_path / f"k{size}.pdf", [(72, 420 + size * i, line, size) for i, line in enumerate(lines)])
        before = letters_of(path)
        _, result = export_zones(path, {"z1": data}, tmp_path / f"out{size}", kept={"u1": url, "p1": phone})
        assert result.exported, (size, [leak.message for leak in result.leaks])
        after = set(letters_of(result.output_path))
        url_rect = pymupdf.Rect(url)
        url_letters = [g for g in before if url_rect.contains(vectors.centre(pymupdf.Rect(g)))]
        assert all(g in after for g in url_letters), size  # the URL left visible keeps every letter


def test_letters_drawn_with_fill_and_stroke_are_removed_or_reported(tmp_path):
    # MuPDF counts a filled and stroked shape as covered only with its stroke: half the width with
    # round joins, ten times the width with miter joins (measured).
    def outlined(path, width, join):
        src = pymupdf.open()
        src.new_page(width=595, height=842).insert_text((72, 140), "Ana Prueba", fontsize=24)
        with pymupdf.open("svg", src[0].get_svg_image(text_as_path=True).encode("utf-8")) as as_svg:
            letters = [d for d in pymupdf.open("pdf", as_svg.convert_to_pdf())[0].get_drawings() if d.get("fill")]
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 300), NEUTRAL[0], fontsize=11)
        for d in letters:
            shape = page.new_shape()
            for item in d["items"]:
                if item[0] == "l":
                    shape.draw_line(item[1], item[2])
                elif item[0] == "c":
                    shape.draw_bezier(item[1], item[2], item[3], item[4])
            shape.finish(fill=(0, 0, 0), color=(0, 0, 0), width=width, lineJoin=join, even_odd=True, closePath=False)
            shape.commit()
        doc.save(path)
        doc.close()
        return path

    path = outlined(tmp_path / "round.pdf", 0.5, 1)
    with pymupdf.open(path) as doc:
        name = [g for g in vectors.letters(doc[0]) if g.y1 < 200]
    zone = (min(g.x0 for g in name) - 2, min(g.y0 for g in name) - 2, max(g.x1 for g in name) + 2,
            max(g.y1 for g in name) - 0.5)  # fmt: skip
    _, result = export_zones(path, {"z1": zone}, tmp_path / "out_round")
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        assert not [g for g in vectors.letters(doc[0]) if g.y1 < 200]
    # A 1.5 pt stroke with miter joins would need a rectangle 15 pt beyond each letter: not covered by
    # the letters' pass. Being stroked and inside the zone, they leave with the strokes instead
    # (``strokes.remove``); never silently left in the file.
    path = outlined(tmp_path / "miter.pdf", 1.5, 0)
    _, result = export_zones(path, {"z1": zone}, tmp_path / "out_miter")
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        assert not [g for g in vectors.letters(doc[0]) if g.y1 < 200]


def test_joined_letter_rectangles_never_reach_into_a_kept_area():
    a, b = pymupdf.Rect(0, 0, 10, 10), pymupdf.Rect(8, 0, 18, 10)
    keep = [pymupdf.Rect(9, 11, 30, 20)]  # touched only by the union's corner: no, the union is 0..18
    assert vectors.join_overlapping([a, b]) == [pymupdf.Rect(0, 0, 18, 10)]
    keep = [pymupdf.Rect(0, 9.5, 2, 12)]  # a kept area under a's corner: a alone touches it too
    c = pymupdf.Rect(20, 0, 30, 10)
    d = pymupdf.Rect(28, 0, 40, 10)
    away = [pymupdf.Rect(35, -10, 45, -1)]  # c and d both stay clear of it; their union too
    assert vectors.join_overlapping([c, d], away) == [pymupdf.Rect(20, 0, 40, 10)]
    e, f = pymupdf.Rect(50, 0, 60, 5), pymupdf.Rect(58, 5, 70, 12)
    corner = [pymupdf.Rect(51, 6, 57, 11)]  # inside the union's box, touched by neither of them
    assert vectors.join_overlapping([e, f], corner) == [e, f]
    assert keep  # (kept for readability of the cases above)
