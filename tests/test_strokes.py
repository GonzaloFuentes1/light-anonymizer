"""Stroked vector paths under redaction zones (``anonymizer.engine.strokes``): only the paths drawn
under a zone leave the file, nothing around them; what cannot be removed safely blocks the export.
Invented documents only."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from anonymizer.engine import faces, ocr, pdf, qr, strokes
from anonymizer.engine.model import AnalyzedFile, Finding
from anonymizer.engine.real import RealEngine

NEUTRAL = "Informe ficticio de prueba para el motor de anonimizacion, con texto neutro."


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    monkeypatch.setattr(ocr, "read_lines", lambda bgr, min_side=0, check=None: [])
    monkeypatch.setattr(faces, "detect", lambda bgr, threshold=0.5, check=None: [])
    monkeypatch.setattr(qr, "detect", lambda bgr: [])


def signature(page: pymupdf.Page, x: float = 300, y: float = 500, w: float = 150, h: float = 40, width: float = 1):
    """A zigzag of Bézier curves (a drawn signature); returns its box."""
    pts = [(x + w * i / 15, y + (h if i % 2 else 0)) for i in range(16)]
    shape = page.new_shape()
    for i in range(0, len(pts) - 3, 3):
        shape.draw_bezier(*pts[i : i + 4])
    shape.finish(color=(0, 0, 0.5), width=width, closePath=False)
    shape.commit()
    return pymupdf.Rect(x, y, x + w, y + h)


def new_page(doc: pymupdf.Document) -> pymupdf.Page:
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), NEUTRAL, fontsize=10)
    return page


def drawings(path: Path, n: int = 0) -> list[tuple[str, str, tuple]]:
    with pymupdf.open(path) as doc:
        page = doc[n]
        page.set_rotation(0)
        return [
            (d["type"], "".join(i[0][0] for i in d["items"]), tuple(round(v, 1) for v in d["rect"]))
            for d in page.get_drawings()
        ]


def curves(path: Path) -> list:
    return [d for d in drawings(path) if "c" in d[1]]


def redact(tmp_path: Path, build, zone, drawn: bool = False, rotation: int = 0, crop=None) -> tuple[list, list]:
    doc = pymupdf.open()
    page = new_page(doc)
    build(page, doc)
    if crop:
        page.set_cropbox(pymupdf.Rect(*crop))
    if rotation:
        page.set_rotation(rotation)
    doc.save(tmp_path / "in.pdf")
    doc.close()
    zones = {0: [pymupdf.Rect(*zone)]}
    pdf.redact(str(tmp_path / "in.pdf"), str(tmp_path / "out.pdf"), zones, drawn_by_page=zones if drawn else None)
    return drawings(tmp_path / "in.pdf"), drawings(tmp_path / "out.pdf")


# ---------------------------------------------------------------------------
# Only what lies under the zone leaves the file
# ---------------------------------------------------------------------------


def test_a_frame_just_around_the_signature_stays(tmp_path):
    def build(page, doc):
        signature(page)
        page.draw_rect(pymupdf.Rect(295, 495, 455, 545), color=(0, 0, 0), width=1)  # the signature field's frame

    before, after = redact(tmp_path, build, (298, 498, 452, 542))
    assert ("s", "c" * 5, (300.0, 500.0, 450.0, 540.0)) in before
    assert not [d for d in after if "c" in d[1]]
    assert ("s", "r", (295.0, 495.0, 455.0, 545.0)) in after


def test_a_filled_panel_around_the_signature_stays(tmp_path):
    def build(page, doc):
        page.draw_rect(pymupdf.Rect(294, 494, 456, 546), color=None, fill=(0.9, 0.9, 1))
        signature(page)

    _, after = redact(tmp_path, build, (298, 498, 452, 542))
    assert ("f", "r", (294.0, 494.0, 456.0, 546.0)) in after and not [d for d in after if "c" in d[1]]


def test_a_thick_stroke_goes_and_the_circle_around_it_stays(tmp_path):
    def build(page, doc):
        page.draw_line((300, 500), (330, 500), color=(0, 0, 0), width=6)
        page.draw_circle((315, 500), 40, color=(1, 0, 0), width=1)

    before, after = redact(tmp_path, build, (294, 494, 336, 506))
    assert ("s", "l", (300.0, 500.0, 330.0, 500.0)) in before
    assert ("s", "l", (300.0, 500.0, 330.0, 500.0)) not in after
    assert [d for d in after if d[1] == "cccc"]  # the circle


def test_a_highlighted_name_in_a_shaded_cell(tmp_path):
    def build(page, doc):
        page.draw_rect(pymupdf.Rect(72, 300, 260, 332), color=(0, 0, 0), fill=(0.93, 0.93, 0.93), width=0.8)
        page.draw_line((88, 316), (172, 316), color=(1, 1, 0), width=11, stroke_opacity=0.5)  # highlighter
        page.insert_text((82, 320), "Ana Inventada Soto", fontsize=11)

    before, after = redact(tmp_path, build, (81, 308, 180, 324))
    assert ("fs", "r", (72.0, 300.0, 260.0, 332.0)) in after  # the cell keeps its shading and border
    assert ("s", "l", (88.0, 316.0, 172.0, 316.0)) not in after


def test_type3_glyphs_are_never_touched(tmp_path):
    def build(page, doc):
        proc = doc.get_new_xref()
        doc.update_object(proc, "<<>>")
        doc.update_stream(proc, b"500 0 d0 20 w 0 0 m 100 400 300 -100 400 200 c S")
        font = doc.get_new_xref()
        doc.update_object(
            font,
            f"<< /Type /Font /Subtype /Type3 /FontBBox [0 -200 500 500] /FontMatrix [0.001 0 0 0.001 0 0] "
            f"/CharProcs << /a {proc} 0 R >> /Encoding << /Type /Encoding /Differences [97 /a] >> "
            f"/FirstChar 97 /LastChar 97 /Widths [500] /Resources << >> >>",
        )
        resources = doc.xref_get_key(page.xref, "Resources")
        doc.xref_set_key(int(resources[1].split()[0]), "Font/F3", f"{font} 0 R")
        xref = page.get_contents()[0]
        # A glyph whose procedure, in glyph space, would lie inside the zone below.
        body = b"\nBT /F3 40 Tf 300 400 Td (aaa) Tj ET\n0 0 0.5 RG 1 w 100 100 m 150 150 200 50 250 100 c S\n"
        doc.update_stream(xref, doc.xref_stream(xref) + body)

    before, after = redact(tmp_path, build, (95, 687, 255, 797))  # around the curve only
    glyphs = [d for d in before if d[2][1] < 500 and "c" in d[1]]
    assert len(glyphs) == 3 and all(g in after for g in glyphs)
    assert ("s", "c", (100.0, 692.0, 250.0, 792.0)) not in after


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_rotated_and_cropped_page(tmp_path, rotation):
    def build(page, doc):
        signature(page)
        page.draw_line((100, 600), (500, 600), color=(0, 0, 0), width=1)

    # zones are in unrotated page space; the CropBox moves its origin
    before, after = redact(tmp_path, build, (248, 438, 402, 482), rotation=rotation, crop=(50, 60, 545, 800))
    assert not [d for d in after if "c" in d[1]] and ("s", "l", (50.0, 540.0, 450.0, 540.0)) in after


def test_a_scaled_content_stream(tmp_path):
    def build(page, doc):
        xref = page.get_contents()[0]
        body = (
            b"\nq 0.1 0 0 0.1 0 0 cm 0 0 0.5 RG 10 w "
            b"3000 3000 m 3500 3800 3800 2600 4500 3400 c 5000 3900 5300 2700 6000 3300 c S "
            b"1000 1000 m 5000 1000 l S Q\n"
        )
        doc.update_stream(xref, doc.xref_stream(xref) + body)

    _, after = redact(tmp_path, build, (295, 842 - 395, 605, 842 - 255))
    assert not [d for d in after if "c" in d[1]] and ("s", "l", (100.0, 742.0, 500.0, 742.0)) in after


def test_a_form_used_twice_keeps_the_copy_outside_the_zone(tmp_path):
    def build(page, doc):
        src = pymupdf.open()
        signature(src.new_page(width=200, height=60), 20, 10)
        page.show_pdf_page(pymupdf.Rect(300, 500, 500, 560), src, 0)
        page.show_pdf_page(pymupdf.Rect(300, 650, 500, 710), src, 0)

    _, after = redact(tmp_path, build, (300, 500, 500, 560))
    assert [d[2] for d in after if "c" in d[1]] == [(320.0, 660.0, 470.0, 700.0)]


# ---------------------------------------------------------------------------
# Strokes that cross the edge of a zone where drawings are data
# ---------------------------------------------------------------------------


def test_a_stroke_mostly_under_a_signature_zone_goes_whole(tmp_path):
    def build(page, doc):
        signature(page)

    _, after = redact(tmp_path, build, (298, 498, 430, 542), drawn=True)  # the last 20 pt stick out
    assert not [d for d in after if "c" in d[1]]
    _, kept = redact(tmp_path, build, (298, 498, 430, 542))  # a zone of text data: only what is inside goes
    assert [d for d in kept if "c" in d[1]]


def _engine_file(path: Path, zone: pymupdf.Rect) -> tuple[RealEngine, AnalyzedFile]:
    engine = RealEngine()
    file = AnalyzedFile(id="f1", name=path.name, path=str(path))
    engine.analyze(file, [])
    file.findings.append(
        Finding(id="m1", file_id="f1", page=0, type="manual", detector="reviewer", status="added",
                polygon=[[zone.x0, zone.y0], [zone.x1, zone.y0], [zone.x1, zone.y1], [zone.x0, zone.y1]])
    )  # fmt: skip
    return engine, file


def test_a_drawn_zone_over_half_a_signature_blocks_the_export(tmp_path):
    doc = pymupdf.open()
    signature(new_page(doc))
    doc.save(tmp_path / "a.pdf")
    doc.close()
    engine, file = _engine_file(tmp_path / "a.pdf", pymupdf.Rect(290, 490, 370, 550))
    result = engine.export(file, str(tmp_path / "out"))
    assert not result.exported
    assert any("agranda la zona" in leak.message and "página 1" in leak.message for leak in result.leaks)
    file.findings[-1].polygon = [[290, 490], [460, 490], [460, 550], [290, 550]]  # the whole signature
    result = engine.export(file, str(tmp_path / "out2"))
    assert result.exported, [leak.message for leak in result.leaks]
    assert not curves(Path(result.output_path))


def test_a_clipped_stroke_whose_visible_part_is_covered(tmp_path):
    doc = pymupdf.open()
    page = new_page(doc)
    xref = page.get_contents()[0]
    body = (
        b"\nq 300 300 60 60 re W n 0 0 0.5 RG 1.2 w "
        b"100 330 m 150 380 200 280 250 330 c 300 380 350 280 400 330 c 450 380 480 280 500 330 c S Q\n"
    )
    doc.update_stream(xref, doc.xref_stream(xref) + body)
    doc.save(tmp_path / "a.pdf")
    doc.close()
    engine, file = _engine_file(tmp_path / "a.pdf", pymupdf.Rect(295, 842 - 365, 365, 842 - 295))
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    assert not curves(Path(result.output_path))  # hidden under the box is not enough: it left the file


def test_a_tight_zone_around_a_ring_is_not_a_false_block(tmp_path):
    # The leak check and the redaction use the same rectangles: a zone a hair inside a ring's edge
    # neither blocks the export without a reason nor lets the ring stay.
    doc = pymupdf.open()
    new_page(doc).draw_circle((150, 400), 20, color=(0, 0, 0), width=1)
    doc.save(tmp_path / "a.pdf")
    doc.close()
    engine, file = _engine_file(tmp_path / "a.pdf", pymupdf.Rect(130.2, 380.2, 169.8, 419.8))
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]
    assert not curves(Path(result.output_path))


def test_a_tight_zone_on_a_filled_logo_exports(tmp_path):
    doc = pymupdf.open()
    new_page(doc).draw_circle((150, 400), 20, color=None, fill=(0.2, 0.3, 0.7))
    doc.save(tmp_path / "a.pdf")
    doc.close()
    engine, file = _engine_file(tmp_path / "a.pdf", pymupdf.Rect(130.3, 380.3, 169.7, 419.7))
    result = engine.export(file, str(tmp_path / "out"))
    assert result.exported, [leak.message for leak in result.leaks]


# ---------------------------------------------------------------------------
# Never silently: what cannot be removed safely is put back and blocks the export
# ---------------------------------------------------------------------------


def test_a_wrong_removal_is_undone_and_blocks_the_export(tmp_path, monkeypatch):
    doc = pymupdf.open()
    page = new_page(doc)
    signature(page)
    page.draw_rect(pymupdf.Rect(295, 495, 455, 545), color=(0, 0, 0), width=1)
    doc.save(tmp_path / "a.pdf")
    doc.close()
    real_align = strokes._align

    def wrong(calls, found, targets, to_user):  # drop the frame's call instead of the signature's
        return {n + 1: k for n, k in real_align(calls, found, targets, to_user).items()}

    monkeypatch.setattr(strokes, "_align", wrong)
    with pymupdf.open(tmp_path / "a.pdf") as doc:
        page = doc[0]
        before = [(d["type"], d["rect"]) for d in page.get_drawings()]
        assert strokes.remove(page, [pymupdf.Rect(298, 498, 452, 542)]) == 0
        assert [(d["type"], d["rect"]) for d in page.get_drawings()] == before  # put back as it was
    engine, file = _engine_file(tmp_path / "a.pdf", pymupdf.Rect(298, 498, 452, 542))
    result = engine.export(file, str(tmp_path / "out"))
    assert not result.exported
    assert any("no se pudo quitar" in leak.message for leak in result.leaks)


def test_the_content_filter_still_works():
    assert strokes.self_test()
