"""Decided 2026-10-06: "if unsure, that page is exported as an image". A page whose redaction may
have left something drawn under a zone is exported as one image of the redacted page; the review's
after shows exactly that page. Invented documents only."""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageFilter

from anonymizer.engine import audit, common, faces, leftovers, ocr, pdf, qr, strokes, vectors, verify
from anonymizer.engine.locks import PDF_LOCK
from anonymizer.engine.model import AnalyzedFile, Finding
from anonymizer.engine.real import RealEngine

NEUTRAL = "Informe ficticio de prueba para el motor de anonimizacion, con texto neutro."


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    monkeypatch.setattr(ocr, "read_lines", lambda bgr, min_side=0, check=None: [])
    monkeypatch.setattr(faces, "detect", lambda bgr, threshold=0.5, check=None: [])
    monkeypatch.setattr(qr, "detect", lambda bgr: [])


def squiggle(page: pymupdf.Page, x: float = 300, y: float = 500, w: float = 150, h: float = 40, width: float = 1):
    """A zigzag of Bézier curves (a drawn signature)."""
    pts = [(x + w * i / 15, y + (h if i % 2 else 0)) for i in range(16)]
    shape = page.new_shape()
    for i in range(0, len(pts) - 3, 3):
        shape.draw_bezier(*pts[i : i + 4])
    shape.finish(color=(0, 0, 0.5), width=width, closePath=False)
    shape.commit()


def new_doc(build, path: Path, rotation: int = 0, crop=None, second: bool = False) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), NEUTRAL, fontsize=10)
    page.insert_text((72, 700), "Pie de pagina neutro que debe seguir legible.", fontsize=10)
    build(page, doc)
    if crop:
        page.set_cropbox(pymupdf.Rect(*crop))
    if rotation:
        page.set_rotation(rotation)
    if second:
        doc.new_page(width=300, height=200).insert_text((20, 50), "Segunda pagina neutra.", fontsize=10)
    doc.save(path)
    doc.close()
    return path


def manual(x0, y0, x1, y1, fid="m1", page=0) -> Finding:
    return Finding(id=fid, file_id="f1", page=page, type="manual", detector="reviewer", status="added",
                   polygon=common.rect_polygon(x0, y0, x1, y1))  # fmt: skip


def export(path: Path, findings: list[Finding], out: Path):
    file = AnalyzedFile(id="f1", name=path.name, path=str(path), kind="pdf", findings=findings, status="confirmed")
    return file, RealEngine().export(file, str(out))


def render(path, n=0, zoom=1.0) -> np.ndarray:
    with pymupdf.open(path) as doc:
        pix = doc[n].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()


def is_image_page(path, n=0) -> bool:
    with pymupdf.open(path) as doc:
        page = doc[n]
        return (
            not page.get_text().strip()
            and not page.get_drawings()
            and len(page.get_images()) == 1
            and not list(page.annots() or [])
        )


# ---------------------------------------------------------------------------
# The check: what is left under a zone after the page's redaction
# ---------------------------------------------------------------------------


def _check(build, zones) -> list[str]:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    build(page, doc)
    with PDF_LOCK:
        return leftovers.check(page, [pymupdf.Rect(z) for z in zones])


def test_layout_lines_and_boxes_are_not_leftovers():
    def build(page, doc):
        page.draw_line((10, 100), (390, 100), width=0.5)  # a rule across the zone
        page.draw_line((200, 10), (200, 390), width=1)  # a vertical one
        page.draw_rect(pymupdf.Rect(50, 50, 350, 150), color=(0, 0, 0), width=1)  # a frame around it
        page.draw_rect(pymupdf.Rect(120, 80, 260, 140), color=None, fill=(0.9, 0.9, 0.6))  # a cell cut by it
        page.draw_rect(pymupdf.Rect(100, 90, 300, 110), color=(0, 0, 0), fill=(0, 0, 0))  # the black box itself
        page.draw_circle((200, 300), 80, color=(0, 0, 0), fill=(0.8, 0.8, 1))  # a disc far from it
        page.draw_line((90, 80), (310, 120), width=0.5)  # a slanted rule across it
        tilted = pymupdf.Rect(120, 60, 280, 140).quad.morph(pymupdf.Point(200, 100), pymupdf.Matrix(12))
        page.draw_quad(tilted, color=(0.6, 0, 0), width=1.2)  # a tilted stamp's border crossing it

    assert _check(build, [(100, 90, 300, 110)]) == []


@pytest.mark.parametrize(
    "build, reason",
    [
        (lambda page, doc: squiggle(page, 120, 80, 160, 40), "stroke"),  # a curve under the zone
        (lambda page, doc: page.draw_circle((200, 100), 30, color=None, fill=(0.2, 0.2, 0.2)), "shape"),
        (lambda page, doc: page.draw_polyline([(90, 80), (200, 120), (310, 80)], width=0.5), "stroke"),  # a V
        (lambda page, doc: page.draw_line((90, 100), (310, 100), width=8), "stroke"),  # a thick bar
        (lambda page, doc: page.draw_rect(pymupdf.Rect(150, 95, 156, 101), color=None, fill=(0.2, 0.2, 0.2)), "shape"),
    ],
)
def test_drawings_under_a_zone_are_leftovers(build, reason):
    assert _check(build, [(100, 90, 300, 110)]) == [reason]


def test_a_curved_clip_under_a_zone_is_a_leftover():
    def build(page, doc):
        xref = doc.get_new_xref()
        doc.update_object(xref, "<<>>")
        doc.update_stream(xref, b"q 150 290 m 200 360 250 290 c h W n 0 0 0 rg 0 0 1 1 re f Q")
        doc.xref_set_key(page.xref, "Contents", f"{xref} 0 R")

    assert _check(build, [(100, 90, 300, 110)]) == ["clip"]


@pytest.mark.parametrize("clip, expected", [("100 280 200 40 re", []), ("160 295 4 4 re 170 295 4 4 re 180 299 4 4 re", ["ink"])])
def test_a_gradient_under_a_zone_is_judged_by_what_it_paints(clip, expected):
    # A smooth gradient that holds the zone hides nothing (no edge under the box); painted through
    # a clip of small squares (a QR code in a gradient) it carries data, and the page is an image.
    def build(page, doc):
        sh = doc.get_new_xref()
        doc.update_object(sh, "<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [100 0 300 0] "
                              "/Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >>")  # fmt: skip
        xref = doc.get_new_xref()
        doc.update_object(xref, "<<>>")
        doc.update_stream(xref, f"q {clip} W n /Sh0 sh Q".encode())
        doc.xref_set_key(page.xref, "Contents", f"{xref} 0 R")
        doc.xref_set_key(page.xref, "Resources", f"<< /Shading << /Sh0 {sh} 0 R >> >>")

    assert _check(build, [(150, 90, 250, 110)]) == expected
    assert _check(build, [(150, 300, 250, 330)]) == []


def test_a_page_too_slow_to_check_is_a_leftover():
    def build(page, doc):
        squiggle(page, 10, 300, 100, 20)

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    build(page, doc)
    with PDF_LOCK:
        assert leftovers.check(page, [pymupdf.Rect(0, 0, 50, 50)], deadline=time.monotonic() - 1) == ["time"]


# ---------------------------------------------------------------------------
# The export: unsure pages become images; nothing is blocked for them
# ---------------------------------------------------------------------------


def test_a_stroke_crossing_a_drawn_zone_exports_the_page_as_an_image(tmp_path):
    path = new_doc(lambda page, doc: squiggle(page), tmp_path / "a.pdf")
    _, result = export(path, [manual(290, 490, 370, 550)], tmp_path / "out")  # half the signature
    assert result.exported, [leak.message for leak in result.leaks]
    assert [r["page"] for r in result.rasterized_pages] == [0]
    assert "trazo" in result.rasterized_pages[0]["reason"]
    out = Path(result.output_path)
    assert is_image_page(out)  # no text layer, no vector path, no annotation: one image
    pixels = render(out, zoom=2)
    assert pixels[2 * 492 : 2 * 548, 2 * 292 : 2 * 368].max() < 60  # the zone is black
    before = render(path, zoom=2)
    outside = np.abs(
        pixels[2 * 495 : 2 * 545, 2 * 380 : 2 * 450].astype(int) - before[2 * 495 : 2 * 545, 2 * 380 : 2 * 450]
    )
    assert outside.mean() < 8  # the rest of the stroke still shows, as an image
    assert "en la página 1" not in result.message or "imagen" in result.message


def test_a_page_with_nothing_left_stays_vector(tmp_path):
    path = new_doc(lambda page, doc: squiggle(page), tmp_path / "a.pdf")
    _, result = export(path, [manual(290, 490, 460, 550)], tmp_path / "out")  # the whole signature
    assert result.exported and result.rasterized_pages == []
    with pymupdf.open(result.output_path) as doc:
        assert "Pie de pagina" in doc[0].get_text()
        assert not [d for d in doc[0].get_drawings() if "c" in "".join(i[0] for i in d["items"])]


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_an_image_page_keeps_size_rotation_and_other_pages(tmp_path, rotation):
    path = new_doc(lambda page, doc: squiggle(page), tmp_path / "a.pdf", rotation, (20, 30, 580, 820), second=True)
    with pymupdf.open(path) as doc:
        size, to_view = doc[0].rect, doc[0].rotation_matrix
    zone = (pymupdf.Rect(290, 490, 370, 550) * to_view).normalize()  # the same place, in view space
    _, result = export(path, [manual(*zone)], tmp_path / "out")
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        assert doc[0].rect == size and doc[0].rotation == rotation
        assert "Segunda pagina neutra" in doc[1].get_text()  # only the unsure page
    assert is_image_page(result.output_path)
    x0, y0, x1, y1 = (int(v) for v in zone)
    assert render(result.output_path)[y0 + 2 : y1 - 2, x0 + 2 : x1 - 2].max() < 60


def test_the_audit_says_which_pages_and_why(tmp_path):
    path = new_doc(lambda page, doc: squiggle(page), tmp_path / "a.pdf")
    file, result = export(path, [manual(290, 490, 370, 550)], tmp_path / "out")
    report = audit.build_report([file], [result])
    record = report["files"][0]
    assert record["rasterized_pages"] == [{"page_number": 1, "reason": result.rasterized_pages[0]["reason"]}]
    html = audit._file_html(file, result)
    assert "Páginas exportadas como imagen" in html and result.rasterized_pages[0]["reason"] in html


def test_a_text_page_is_stored_lossless_and_a_photo_page_as_jpeg(tmp_path):
    def text(page, doc):
        for i in range(30):
            page.insert_text((60, 140 + 16 * i), f"Linea {i} de texto ficticio de relleno.", fontsize=10)
        squiggle(page)

    def photo(page, doc):
        rng = np.random.default_rng(1)
        buf = io.BytesIO()
        noise = Image.fromarray(rng.integers(0, 255, (500, 700, 3), np.uint8)).filter(ImageFilter.GaussianBlur(2))
        noise.save(buf, "PNG")
        page.insert_image(pymupdf.Rect(40, 120, 555, 480), stream=buf.getvalue())
        squiggle(page)

    for build, expected in ((text, "png"), (photo, "jpeg")):
        path = new_doc(build, tmp_path / f"{expected}.pdf")
        with PDF_LOCK, pymupdf.open(path) as doc:
            outcome = pdf.redact_page(doc, 0, [pymupdf.Rect(290, 490, 370, 550)])
        assert outcome.rasterized and outcome.image["format"] == expected, outcome.image


def test_a_huge_page_is_capped(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=4000, height=2000)
    squiggle(page, 300, 500)
    with PDF_LOCK:
        outcome = pdf.redact_page(doc, 0, [pymupdf.Rect(290, 490, 370, 550)])
    assert outcome.rasterized and max(outcome.image["width"], outcome.image["height"]) <= pdf.RASTER_MAX_SIDE


def test_the_stroke_stage_out_of_time_exports_the_page_as_an_image(tmp_path, monkeypatch):
    path = new_doc(lambda page, doc: squiggle(page), tmp_path / "a.pdf")
    monkeypatch.setattr(strokes, "REMOVE_SECONDS", -1.0)
    _, result = export(path, [manual(290, 490, 460, 550)], tmp_path / "out")
    assert result.exported, [leak.message for leak in result.leaks]
    assert result.rasterized_pages[0]["reason"] == leftovers.REASONS["time"]
    assert is_image_page(result.output_path)


def test_the_guard_blocks_if_a_leftover_reached_the_output(tmp_path, monkeypatch):
    # Should never happen: a page with something left under a zone is exported as an image. If the
    # pipeline failed to do it, the leak check of the output still blocks the export.
    path = new_doc(lambda page, doc: squiggle(page), tmp_path / "a.pdf")
    monkeypatch.setattr(pdf, "rasterize", lambda page, fast=False: None)  # the page is left as it is
    _, result = export(path, [manual(290, 490, 370, 550)], tmp_path / "out")
    assert not result.exported
    assert any("algo dibujado" in leak.message and leak.finding_id == "m1" for leak in result.leaks)
    assert verify.vector_leaks(path, [manual(290, 490, 370, 550)])  # the original has it too


def test_text_readable_outside_the_boxes_still_blocks(tmp_path, monkeypatch):
    # What an image cannot fix still blocks: data readable in the text layer outside the boxes.
    from tests.test_engine_real import VALID_RUT, _skip_zones, analyzed, make_pdf

    file = analyzed(make_pdf(tmp_path / "a.pdf"))
    rut = next(f for f in file.findings if f.text == VALID_RUT)
    _skip_zones(monkeypatch, [rut])
    result = RealEngine().export(file, str(tmp_path / "out"))
    assert not result.exported and any(leak.finding_id == rut.id for leak in result.leaks)


# ---------------------------------------------------------------------------
# The letters' rectangles (D8), conservative again
# ---------------------------------------------------------------------------


def _letters_page(path: Path, size: float = 11) -> Path:
    from tests.test_vector_text import drawn_page

    return drawn_page(path, [(72, 420, "Contacto: fulvia.pellegrini@gmail.com fono +56 9 8123 4567", size)])


def test_letters_rectangles_reach_at_most_a_point_and_a_half_beyond_the_zone(tmp_path):
    path = _letters_page(tmp_path / "a.pdf")
    with PDF_LOCK, pymupdf.open(path) as doc:
        page = doc[0]
        line = [g for g in vectors.letters(page) if g.y0 > 380]
        zone = pymupdf.Rect(min(g.x0 for g in line), min(g.y0 for g in line) + 3, max(g.x1 for g in line), 420)
        groups = pdf.snap_rects(page, [zone])
    limit = zone + (-vectors.COVER_MAX, -vectors.COVER_MAX, vectors.COVER_MAX, vectors.COVER_MAX)
    assert vectors.COVER_MAX <= 1.5
    assert all(limit.contains(r) for r in groups[0][1:])
    for r in groups[0][1:]:
        assert zone.contains(vectors.centre(r))  # only letters whose centre is inside the zone


def test_a_kept_area_never_spares_a_letter_whose_centre_is_in_an_active_zone(tmp_path):
    path = _letters_page(tmp_path / "a.pdf")
    with pymupdf.open(path) as doc:
        line = sorted((g for g in vectors.letters(doc[0]) if g.y0 > 380), key=lambda g: g.x0)
    zone = (line[10].x0 - 1, line[0].y0 - 2, line[20].x1 + 1, max(g.y1 for g in line) + 2)
    kept = (line[0].x0 - 1, line[0].y0 - 2, line[15].x1 + 1, max(g.y1 for g in line) + 2)  # overlaps it
    findings = [Finding(id="z1", file_id="f1", page=0, type="email", polygon=common.rect_polygon(*zone), text="x",
                        detector="ocr"),
                Finding(id="k1", file_id="f1", page=0, type="url", polygon=common.rect_polygon(*kept), text="y",
                        detector="ocr", status="suggested", optional=True, optional_reason="url")]  # fmt: skip
    _, result = export(path, findings, tmp_path / "out")
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        left = vectors.letters(doc[0])
    z = pymupdf.Rect(zone)
    assert not [g for g in left if z.contains(vectors.centre(g))]  # gone, or the page is an image


def test_fill_and_stroke_markers_beside_a_zone_neither_block_nor_rasterize(tmp_path):
    def build(page, doc):
        for i in range(12):
            shape = page.new_shape()
            shape.draw_circle((100 + 30 * i, 400 - 8 * (i % 5)), 3)
            shape.finish(fill=(0.2, 0.4, 0.8), color=(0.1, 0.2, 0.5), width=0.75, lineJoin=0)
            shape.commit()
        page.insert_text((100, 450), "Quintanilla Brito", fontsize=8)

    path = new_doc(build, tmp_path / "a.pdf")
    _, result = export(path, [manual(98, 442, 170, 452)], tmp_path / "out")
    assert result.exported and result.rasterized_pages == []
    _, result = export(path, [manual(160, 380, 220, 410)], tmp_path / "out2")  # over two markers
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        left = [d for d in doc[0].get_drawings() if pymupdf.Rect(d["rect"]).intersects((160, 380, 220, 410))]
    assert result.rasterized_pages or all(d["fill"] == (0, 0, 0) for d in left)  # only the black box


def test_a_page_with_many_shapes_and_findings_is_checked_quickly(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=842, height=595)
    shape = page.new_shape()
    for i in range(20000):
        x, y = 20 + (i % 200) * 4, 150 + (i // 200) * 4
        shape.draw_polyline([(x, y), (x + 2.5, y), (x + 1.2, y + 2.5), (x, y)])
        shape.finish(fill=(0.3, 0.5, 0.2), color=None)
    shape.commit()
    for i in range(10):
        page.insert_text((40, 40 + 10 * i), f"Linea {i} con texto neutro de relleno", fontsize=8)
    doc.save(tmp_path / "heavy.pdf")
    doc.close()
    findings = [manual(40 + 15 * (i % 20), 30 + 5 * (i // 20), 50 + 15 * (i % 20), 34 + 5 * (i // 20), fid=f"m{i}")
                for i in range(40)]  # fmt: skip
    started = time.perf_counter()
    _, result = export(tmp_path / "heavy.pdf", findings, tmp_path / "out")
    assert result.exported, [leak.message for leak in result.leaks]
    assert time.perf_counter() - started < 30
