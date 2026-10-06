"""Stroked vector paths under redaction zones (``anonymizer.engine.strokes``): only the paths drawn
under a zone leave the file, nothing around them; what cannot be removed safely blocks the export.
Invented documents only."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
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
    real_match = strokes._match

    def wrong(*args):  # take the frame's call for the signature's: the frame would be dropped
        return {n + 1: k for n, k in real_match(*args).items()}

    monkeypatch.setattr(strokes, "_match", wrong)
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


@pytest.mark.parametrize("change", [{"instance_forms": 0}, {"recurse": 1}, {"recurse": 1, "instance_forms": 0}])
def test_the_self_test_notices_a_filter_that_enters_shared_content(monkeypatch, change):
    # A filter that edits a form shared by two pages in place, or enters Type3 glyph procedures
    # (shared by every page using the font), must fail the startup check.
    real = pymupdf._make_PdfFilterOptions

    def patched(**options):
        return real(**{**options, **change})

    monkeypatch.setattr(pymupdf, "_make_PdfFilterOptions", patched)
    assert not strokes.self_test()


# ---------------------------------------------------------------------------
# Second review (invented documents)
# ---------------------------------------------------------------------------


def sig_ops(x: float, y: float, w: float = 150, h: float = 40, width: float = 1, color: str = "0 0 0.5 RG") -> str:
    """A zigzag of five Bézier curves in PDF user space (origin bottom left), as content operators."""
    pts = [(x + w * i / 15, y + (h if i % 2 else 0)) for i in range(16)]
    ops = f"{color} {width} w {pts[0][0]:.2f} {pts[0][1]:.2f} m "
    for i in range(1, 16, 3):
        a, b, c = pts[i], pts[i + 1], pts[i + 2]
        ops += f"{a[0]:.2f} {a[1]:.2f} {b[0]:.2f} {b[1]:.2f} {c[0]:.2f} {c[1]:.2f} c "
    return ops + "S\n"


def content_pdf(path: Path, body: str, extra=None) -> Path:
    doc = pymupdf.open()
    page = new_page(doc)
    page.clean_contents()
    xref = page.get_contents()[0]
    doc.update_stream(xref, doc.xref_stream(xref) + b"\n" + body.encode("latin-1"))
    if extra:
        extra(doc, page)
    doc.save(path)
    doc.close()
    return path


def export_with(path: Path, zones: list[tuple[str, pymupdf.Rect]], tmp_path: Path):
    engine = RealEngine()
    file = AnalyzedFile(id="f1", name=path.name, path=str(path))
    engine.analyze(file, [])
    file.findings = [f for f in file.findings if f.type != "signature"]
    for k, (type_, r) in enumerate(zones):
        file.findings.append(
            Finding(id=f"z{k}", file_id="f1", page=0, type=type_, detector="reviewer" if type_ == "manual" else "x",
                    status="added" if type_ == "manual" else "proposed",
                    polygon=[[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]])
        )  # fmt: skip
    return engine.export(file, str(tmp_path / f"out_{len(list(tmp_path.iterdir()))}")), file


SIG_ZONE = pymupdf.Rect(295, 495, 455, 547)  # around sig_ops(300, 300): page y 502..542


def _pattern(doc, page):
    pattern = doc.get_new_xref()
    doc.update_object(
        pattern,
        "<< /Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 4 4] /XStep 4 /YStep 4 "
        "/Resources << >> >>",
    )
    doc.update_stream(pattern, b"0 0 1 RG 0.5 w 0.5 2 m 1 3.5 2 0.5 3.5 2 c S")
    resources = doc.xref_get_key(page.xref, "Resources")
    if resources[0] == "xref":
        doc.xref_set_key(int(resources[1].split()[0]), "Pattern", f"<< /P1 {pattern} 0 R >>")
    else:
        doc.xref_set_key(page.xref, "Resources/Pattern", f"<< /P1 {pattern} 0 R >>")


def test_many_paths_the_filter_does_not_list_stay_fast(tmp_path):
    # Pattern fills: the filter paints them, get_drawings does not list them (it lists the pattern's
    # cell instead, in the cell's own space). Lining them up must not scan the page for each one.
    ops = ["/Pattern cs /P1 scn"]
    ops += [f"{20 + (i % 100) * 5.5:.1f} {120 + (i // 100) * 6:.1f} 4 4 re f" for i in range(3000)]
    for i in range(3000):
        x, y = 20 + (i % 100) * 5.5, 320 + (i // 100) * 6
        ops.append(
            f"0 0 0 RG 0.3 w {x:.1f} {y:.1f} m {x + 1:.1f} {y + 2:.1f} {x + 2:.1f} {y - 1:.1f} {x + 2.5:.1f} {y + 1:.1f} c S"
        )
    path = content_pdf(tmp_path / "p.pdf", "\n".join(ops) + "\n" + sig_ops(300, 650), _pattern)
    with pymupdf.open(path) as doc:
        page = doc[0]
        started = time.perf_counter()
        removed = strokes.remove(page, [pymupdf.Rect(295, 842 - 695, 455, 842 - 645)])
        assert time.perf_counter() - started < 3  # was about ten minutes
    assert removed == 1


def test_a_pattern_cell_under_a_zone_is_never_exempt(tmp_path):
    # get_drawings lists a pattern's cell at the cell's own coordinates (here the page's top left).
    # Whether it is a cell or a stroke of the page is a guess: under a zone it is planned like any
    # stroke, never left out.
    path = content_pdf(tmp_path / "c.pdf", "/Pattern cs /P1 scn 250 250 250 150 re f\n" + sig_ops(300, 300), _pattern)
    with pymupdf.open(path) as doc:
        plan = strokes.plan(doc[0], [pymupdf.Rect(0, 780, 60, 842), SIG_ZONE])
        planned = sorted(tuple(round(v) for v in plan.paths[k].d["rect"]) for k in plan.remove)
    assert (300, 502, 450, 542) in planned and len(planned) == 2


@pytest.mark.parametrize("ring", [False, True])
def test_a_rounded_frame_or_a_ring_crossing_a_signature_zone_does_not_block(tmp_path, ring):
    def frame(doc, page):
        if ring:
            page.draw_circle((440, 520), 30, color=(0.1, 0.1, 0.6), width=1.2)  # a stamp's ring
        else:
            page.draw_rect(pymupdf.Rect(310, 490, 470, 560), color=(0, 0, 0), width=0.8, radius=0.2)

    result, _ = export_with(
        content_pdf(tmp_path / "a.pdf", sig_ops(300, 300), frame), [("signature", SIG_ZONE)], tmp_path
    )
    assert result.exported, [leak.message for leak in result.leaks]
    assert [d for d in curves(Path(result.output_path)) if d[2] != (300.0, 502.0, 450.0, 542.0)]  # the frame stays


def test_a_curve_crossing_a_drawn_zone_blocks_even_a_chart(tmp_path):
    # Whether an open curve under a reviewer's zone is a chart or a signature's flourish is a guess:
    # it blocks (a false block is acceptable, a silent leak is not). Under a text zone it does not.
    body = "0.1 0.3 0.8 RG 1.2 w 100 300 m 200 330 300 350 400 360 c 450 365 500 368 550 370 c S\n"
    path = content_pdf(tmp_path / "a.pdf", body)
    result, _ = export_with(path, [("manual", pymupdf.Rect(90, 470, 250, 560))], tmp_path)
    assert not result.exported and any("agranda la zona" in leak.message for leak in result.leaks)
    result, _ = export_with(path, [("name", pymupdf.Rect(90, 470, 250, 560))], tmp_path)
    assert result.exported, [leak.message for leak in result.leaks]


def test_a_text_zone_over_most_of_a_stroke_blocks(tmp_path):
    path = content_pdf(tmp_path / "a.pdf", sig_ops(300, 300))
    result, _ = export_with(path, [("name", pymupdf.Rect(295, 495, 420, 547))], tmp_path)  # 80 % of it
    assert not result.exported and any("agranda la zona" in leak.message for leak in result.leaks)
    result, _ = export_with(path, [("name", pymupdf.Rect(300, 502, 450, 542))], tmp_path)  # its exact box
    assert result.exported, [leak.message for leak in result.leaks]
    assert not curves(Path(result.output_path))


def test_a_failed_removal_of_a_checkmark_is_reported(tmp_path, monkeypatch):
    body = "0 0 0 RG 1.5 w 310 330 m 320 315 l 345 345 l S\n"  # a hand-drawn tick: two straight segments
    path = content_pdf(tmp_path / "a.pdf", body)
    monkeypatch.setattr(strokes, "remove", lambda page, zones, drawn=(), whole=None: 0)
    result, _ = export_with(path, [("manual", pymupdf.Rect(300, 495, 355, 547))], tmp_path)
    assert not result.exported and any("no se pudo quitar" in leak.message for leak in result.leaks)


@pytest.mark.parametrize("miter, width", [(50, 0.5), (100, 0.25)])
def test_a_polyline_with_a_high_miter_limit_is_removed(tmp_path, miter, width):
    points = " ".join(f"{300 + 10 * i} {300 + (40 if i % 2 else 0)} l" for i in range(1, 16))
    path = content_pdf(tmp_path / "a.pdf", f"0 0 0.5 RG {width} w {miter} M 300 300 m {points} S\n")
    result, _ = export_with(path, [("manual", pymupdf.Rect(290, 495, 460, 547))], tmp_path)
    assert result.exported, [leak.message for leak in result.leaks]
    assert not [d for d in drawings(Path(result.output_path)) if d[0] == "s"]


def test_a_stroke_removed_whole_is_recorded(tmp_path):
    from anonymizer.engine import audit

    body = "0.2 0.4 0.8 RG 2 w 100 300 m 250 380 400 220 550 300 c S\n"  # a swoosh, 65 % under the zone
    zone = pymupdf.Rect(90, 500, 420, 560)
    result, file = export_with(content_pdf(tmp_path / "a.pdf", body), [("manual", zone)], tmp_path)
    assert result.exported and not curves(Path(result.output_path))
    (entry,) = result.strokes_removed_whole
    right = np.max(entry["polygon"], axis=0)[0]
    assert entry["page"] == 0 and right > zone.x1 + 100  # it reached far right of the zone
    record = audit.build_report([file], [result])["files"][0]
    assert record["strokes_removed_whole"] == [{"page_number": 1, "polygon": entry["polygon"], "unit": "pt"}]
    json_path, pdf_path = audit.write_audit([file], [result], str(tmp_path / "audit"))
    with pymupdf.open(pdf_path) as doc:
        assert "se quitó entero" in "".join(page.get_text() for page in doc)


def test_a_pattern_partly_under_a_zone_blocks_and_covered_exports(tmp_path):
    # A fill of one tile has its cell listed at the page's coordinates, inside the filled area. The
    # fills only partly under the zone stay in the file, and with them the cell: that blocks. Fills
    # entirely under the zone are removed with what they paint.
    fills = " ".join(f"{300 + 4 * i} 300 4 4 re f" for i in range(10))
    path = content_pdf(tmp_path / "a.pdf", f"/Pattern cs /P1 scn {fills}\n", _pattern)
    zone = pymupdf.Rect(302, 539, 338, 541)  # more than 1 pt short of the fills: MuPDF and we keep them
    with pymupdf.open(path) as doc:
        assert [d for d in doc[0].get_drawings() if d["rect"] in zone + (-1, -1, 1, 1)]
    result, _ = export_with(path, [("name", zone)], tmp_path)
    assert not result.exported and any("trama" in leak.message for leak in result.leaks)
    result, _ = export_with(path, [("name", pymupdf.Rect(298, 536, 342, 544))], tmp_path)
    assert result.exported, [leak.message for leak in result.leaks]


# ---------------------------------------------------------------------------
# Third review: guesses block, never silence (invented documents)
# ---------------------------------------------------------------------------


def _add_pattern(doc, page, cell: bytes, bbox: str = "[0 0 4 4]", step=(4, 4), matrix: str = "[1 0 0 1 0 0]"):
    pattern = doc.get_new_xref()
    doc.update_object(
        pattern,
        f"<< /Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox {bbox} /XStep {step[0]} "
        f"/YStep {step[1]} /Matrix {matrix} /Resources << >> >>",
    )
    doc.update_stream(pattern, cell)
    resources = doc.xref_get_key(page.xref, "Resources")
    if resources[0] == "xref":
        doc.xref_set_key(int(resources[1].split()[0]), "Pattern", f"<< /P1 {pattern} 0 R >>")
    else:
        doc.xref_set_key(page.xref, "Resources/Pattern", f"<< /P1 {pattern} 0 R >>")


BOX = "280 280 190 80 re"  # user space: wider than SIG_ZONE on every side
INNER = "282 282 186 76 re"


def _not_silent(result, path: Path | None = None) -> bool:
    """The export was blocked, or the signature's curves left the file."""
    return (not result.exported) or not curves(Path(result.output_path))


@pytest.mark.parametrize(
    "background",
    [
        "q /Pattern cs /P1 scn {box} f Q\n",  # a hatched box
        "q 0 0 1 1 re W n 1 1 1 rg {box} f Q\n",  # an invisible, clipped-away fill of the same box
    ],
)
def test_a_signature_clipped_to_a_box_with_a_ring_border_never_leaks(tmp_path, background):
    body = background.format(box=BOX) + f"q {BOX} W n {sig_ops(300, 300)} Q\n0 0 0 rg {BOX} {INNER} f*\n"

    def hatch(doc, page):
        _add_pattern(doc, page, b"0.85 0.85 0.85 rg 0 0 1 4 re f")

    path = content_pdf(tmp_path / "x.pdf", body, hatch)
    for kind in ("signature", "manual"):
        result, _ = export_with(path, [(kind, SIG_ZONE)], tmp_path)
        assert _not_silent(result), kind


def test_a_signature_in_a_form_over_a_hatched_box_never_leaks(tmp_path):
    def extra(doc, page):
        form = doc.get_new_xref()
        doc.update_object(form, "<< /Type /XObject /Subtype /Form /BBox [280 280 470 360] >>")
        doc.update_stream(form, sig_ops(300, 300).encode())
        _add_pattern(doc, page, b"0.85 0.85 0.85 rg 0 0 1 4 re f")
        resources = doc.xref_get_key(page.xref, "Resources")
        doc.xref_set_key(page.xref, "Resources/XObject", f"<< /Fs {form} 0 R >>") if resources[0] != "xref" else (
            doc.xref_set_key(int(resources[1].split()[0]), "XObject", f"<< /Fs {form} 0 R >>")
        )

    body = f"q /Pattern cs /P1 scn {BOX} f Q\n/Fs Do\n0 0 0 rg {BOX} {INNER} f*\n"
    result, _ = export_with(content_pdf(tmp_path / "x3.pdf", body, extra), [("signature", SIG_ZONE)], tmp_path)
    assert _not_silent(result)


@pytest.mark.parametrize("fill", ["290 290 200 60 re f", "300 300 150 40 re f"])
def test_a_signature_that_is_a_pattern_tile_never_leaks(tmp_path, fill):
    def tile(doc, page):
        _add_pattern(doc, page, sig_ops(10, 10).encode(), "[0 0 400 400]", (400, 400), "[1 0 0 1 290 290]")

    path = content_pdf(tmp_path / "pc.pdf", f"/Pattern cs /P1 scn {fill}\n", tile)
    result, _ = export_with(path, [("signature", SIG_ZONE)], tmp_path)
    assert _not_silent(result)


def test_a_signature_stroked_with_a_pattern_never_leaks(tmp_path):
    def tile(doc, page):
        _add_pattern(doc, page, b"0 0 0.5 RG 0.5 w 0 2 m 1 3.5 3 0.5 4 2 c S", matrix="[1 0 0 1 300 300]")

    path = content_pdf(tmp_path / "ps.pdf", sig_ops(300, 300, color="/Pattern CS /P1 SCN"), tile)
    result, _ = export_with(path, [("signature", SIG_ZONE)], tmp_path)
    assert _not_silent(result)


def _bez_ops(points, width: float = 1.2) -> str:
    """Page-space points -> Bézier operators in user space through every third point."""
    pts = [(x, 842 - y) for x, y in points]
    pts = pts[: 3 * ((len(pts) - 1) // 3) + 1]
    ops = f"0.1 0.15 0.55 RG {width} w {pts[0][0]:.2f} {pts[0][1]:.2f} m "
    for i in range(1, len(pts) - 2, 3):
        a, b, c = pts[i], pts[i + 1], pts[i + 2]
        ops += f"{a[0]:.2f} {a[1]:.2f} {b[0]:.2f} {b[1]:.2f} {c[0]:.2f} {c[1]:.2f} c "
    return ops + "S\n"


def _name_and_flourish() -> tuple[str, str]:
    from test_signatures import cursive

    body = _bez_ops(cursive(300, 480, 170, 45, seed=1, loops=5)[::12])
    xs = np.linspace(290, 560, 91)
    return body, _bez_ops(np.stack([xs, 535 + 4 * np.sin((xs - 300) / 15.0)], axis=1))


@pytest.mark.parametrize(
    "case, zone",
    [
        ("flourish_40", pymupdf.Rect(292, 470, 400, 545)),
        ("flourish_55", pymupdf.Rect(285, 470, 440, 545)),
        ("garland_45", pymupdf.Rect(295, 505, 372, 535)),
        ("oval_flourish", pymupdf.Rect(292, 470, 400, 545)),
        ("body_40", pymupdf.Rect(292, 470, 365, 535)),
    ],
)
def test_a_signature_stroke_crossing_a_drawn_zone_blocks(tmp_path, case, zone):
    body, flourish = _name_and_flourish()
    if case.startswith("flourish"):
        ops = body + flourish
    elif case == "garland_45":
        xs = np.linspace(300, 460, 160)
        ops = _bez_ops(np.stack([xs, 520 + 6 * np.sin(2 * np.pi * (xs - 300) / 20)], axis=1))
    elif case == "oval_flourish":
        t = np.linspace(0, 2 * np.pi, 49)
        ops = body + _bez_ops(np.stack([380 + 95 * np.cos(t), 505 + 30 * np.sin(t)], axis=1))
    else:
        ops = body
    result, _ = export_with(content_pdf(tmp_path / f"{case}.pdf", ops), [("manual", zone)], tmp_path)
    assert not result.exported and any("agranda la zona" in leak.message for leak in result.leaks)


def test_two_copies_at_the_same_place_both_leave(tmp_path):
    body = "q 0 0 1 1 re W n " + sig_ops(300, 300) + "Q " + sig_ops(300, 300)
    with pymupdf.open(content_pdf(tmp_path / "a2.pdf", body)) as doc:
        page = doc[0]
        assert strokes.remove(page, [SIG_ZONE], [SIG_ZONE]) == 2
        assert not [d for d in page.get_drawings() if any(i[0] == "c" for i in d["items"])]


@pytest.mark.parametrize("miter, width", [(1500, 1), (5000, 0.25)])
def test_a_polyline_with_a_very_high_miter_limit_is_removed(tmp_path, miter, width):
    points = " ".join(f"{300 + 10 * i} {300 + (40 if i % 2 else 0)} l" for i in range(1, 16))
    path = content_pdf(tmp_path / "a.pdf", f"0 0 0.5 RG {width} w {miter} M 300 300 m {points} S\n")
    result, _ = export_with(path, [("manual", pymupdf.Rect(290, 495, 460, 547))], tmp_path)
    assert result.exported, [leak.message for leak in result.leaks]
    assert not [d for d in drawings(Path(result.output_path)) if d[0] == "s"]


def test_crowds_of_paths_on_one_center_stay_within_the_time_budget(tmp_path):
    n = 3000
    ops = ["q /Pattern cs /P1 scn"]
    ops += [
        f"{150 - 0.004 * i:.3f} {150 - 0.004 * i:.3f} {20 + 0.008 * i:.3f} {20 + 0.008 * i:.3f} re f" for i in range(n)
    ]
    ops += ["Q 0 0 0 RG 0.2 w"]
    ops += [
        f"{140 - 0.004 * i:.3f} {140 - 0.004 * i:.3f} {40 + 0.008 * i:.3f} {40 + 0.008 * i:.3f} re S" for i in range(n)
    ]

    def big(doc, page):
        _add_pattern(doc, page, b"0.9 0.9 0.9 rg 0 0 1 1000 re f", "[0 0 1000 1000]", (1000, 1000))

    path = content_pdf(tmp_path / "g.pdf", "\n".join(ops) + "\n" + sig_ops(300, 300), big)
    started = time.perf_counter()
    result, _ = export_with(path, [("signature", SIG_ZONE)], tmp_path)
    assert time.perf_counter() - started < 20
    assert _not_silent(result)


def test_a_stage_over_its_time_budget_blocks(tmp_path, monkeypatch):
    path = content_pdf(tmp_path / "a.pdf", sig_ops(300, 300))
    monkeypatch.setattr(strokes, "REMOVE_SECONDS", -1.0)  # every budget is already spent
    result, _ = export_with(path, [("signature", SIG_ZONE)], tmp_path)
    assert not result.exported and any("tiempo" in leak.message for leak in result.leaks)


@pytest.mark.parametrize("cell", [b"0 0 0.5 rg 0 0 4 4 re f", b"0 0 0.5 RG 0.5 w 0 2 m 1 3.5 3 0.5 4 2 c S"])
def test_a_signature_stroked_with_a_pattern_is_never_silent(tmp_path, cell):
    # get_drawings lists such a stroke only as a clip and the pattern's cell (at the pattern's own
    # origin, far from the zone): the filter's paint call is what shows it is under the zone.
    path = content_pdf(
        tmp_path / "ps.pdf", sig_ops(300, 300, color="/Pattern CS /P1 SCN"), lambda d, p: _add_pattern(d, p, cell)
    )
    result, _ = export_with(path, [("signature", SIG_ZONE)], tmp_path)
    assert not result.exported and any("trama" in leak.message for leak in result.leaks)
    # Inside a zone wide enough for its painted box, it is removed.
    result, _ = export_with(path, [("signature", pymupdf.Rect(270, 470, 480, 575))], tmp_path)
    assert result.exported, [leak.message for leak in result.leaks]
    with pymupdf.open(result.output_path) as doc:
        data = b"".join(doc.xref_stream(x) or b"" for x in range(1, doc.xref_length()) if doc.xref_is_stream(x))
    assert b"310.00 340.00" not in data and b"310 340" not in data


def test_a_table_shaded_as_one_path_of_many_rectangles_does_not_block(tmp_path):
    # The filter reports each rectangle of a path as a call of its own; get_drawings lists one path.
    rows = " ".join(f"72 {600 - 30 * i} 450 14 re" for i in range(6))
    body = f"0.93 0.93 0.93 rg {rows} f\n"
    zone = pymupdf.Rect(80, 842 - 614, 200, 842 - 600)  # a name in the first shaded row
    result, _ = export_with(content_pdf(tmp_path / "t.pdf", body), [("name", zone)], tmp_path)
    assert result.exported, [leak.message for leak in result.leaks]


def test_a_signature_of_several_subpaths_in_one_path_leaves_whole(tmp_path):
    first, second = sig_ops(300, 300).rsplit("S", 1)[0], sig_ops(300, 320, w=140, h=15)
    body = first + second.split(" w ", 1)[1].replace(" m ", " m ", 1)  # one path, two subpaths, one S
    with pymupdf.open(content_pdf(tmp_path / "a.pdf", body)) as doc:
        page = doc[0]
        (path,) = [d for d in page.get_drawings() if any(i[0] == "c" for i in d["items"])]
        assert len(strokes._subpaths(path["items"])) == 2
        assert strokes.remove(page, [SIG_ZONE], [SIG_ZONE]) == 1
        assert not [d for d in page.get_drawings() if any(i[0] == "c" for i in d["items"])]


def test_a_signature_clipped_to_a_box_with_a_ring_border_is_removed(tmp_path):
    # The ring border is one even-odd path of two rectangles: both of its calls are its own.
    body = f"q 0 0 1 1 re W n 1 1 1 rg {BOX} f Q\nq {BOX} W n {sig_ops(300, 300)} Q\n0 0 0 rg {BOX} {INNER} f*\n"
    result, _ = export_with(content_pdf(tmp_path / "x.pdf", body), [("signature", SIG_ZONE)], tmp_path)
    assert result.exported, [leak.message for leak in result.leaks]
    assert not curves(Path(result.output_path))
