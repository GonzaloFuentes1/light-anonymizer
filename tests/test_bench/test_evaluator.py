"""The evaluator has to detect every kind of leak and accept a well-made redaction.

Every case is built right here (without depending on the generators): a PDF with text, an
image with EXIF, a PDF with an image covered by a rectangle and a PDF with vectorized text.
For each one the unredacted output is tested (everything leaks), an apparent redaction
(covering without removing: the specific check fails) and a correct redaction (everything
redacted).
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import piexif
import pymupdf
import pytest
from PIL import Image, ImageFilter, ImageOps

from test_bench.canvas import Canvas, rect
from test_bench.evaluation import coverage as cov
from test_bench.evaluation import evaluate, evaluate_file
from test_bench.evaluation.pdf import pdf_strings
from test_bench.evaluation.report import write_report
from test_bench.evaluation.search import Haystack, needles, texts_from_bytes
from test_bench.fake_data import FakeData
from test_bench.schema import (
    Element,
    FileEntry,
    FileResult,
    Manifest,
    MetadataEntry,
    Page,
    Redaction,
    RedactionReport,
)


@pytest.fixture
def f() -> FakeData:
    return FakeData(33).derive("test_evaluador")


# ---------------------------------------------------------------------------
# helpers for the cases
# ---------------------------------------------------------------------------


def _polygon(q: pymupdf.Quad) -> list[list[float]]:
    return [[q.ul.x, q.ul.y], [q.ur.x, q.ur.y], [q.lr.x, q.lr.y], [q.ll.x, q.ll.y]]


def _find(page: pymupdf.Page, text: str) -> list[list[float]]:
    quads = page.search_for(text, quads=True)
    assert quads, f"not found: {text!r}"
    q = quads[0]
    for extra in quads[1:]:  # a value split into several boxes by the engine: they are joined
        q = pymupdf.Quad(q.rect | extra.rect)
    return _polygon(q)


def _redactions(file_entry: FileEntry, excluded_types=("text",)) -> list[Redaction]:
    return [
        Redaction(page=e.page, polygon=e.polygon, type=e.type, detector="oracle")
        for e in file_entry.elements
        if e.type not in excluded_types and e.polygon is not None and e.layer != "hidden"
    ]


def _result(file_entry: FileEntry, redactions: list[Redaction] | None = None, **kw) -> FileResult:
    return FileResult(
        input=file_entry.path,
        output=kw.pop("output", file_entry.path),
        redactions=_redactions(file_entry) if redactions is None else redactions,
        time_s=kw.pop("time_s", 0.1),
        **kw,
    )


def _by_type(ev, type_: str):
    return [e for e in ev.elements if e.type == type_]


def _save(doc: pymupdf.Document, path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, garbage=kw.pop("garbage", 4), deflate=True, **kw)


# ---------------------------------------------------------------------------
# 1. PDF with a text layer
# ---------------------------------------------------------------------------


def _pdf_text(f: FakeData, dir_: Path, rotation: int = 0) -> FileEntry:
    p = f.person()
    outside = f.person(in_list=False)
    rut, email, phone = p.rut("dots"), p.email("dot"), p.phone.format("mobile_international")
    amount = f.amount()
    doc = pymupdf.open()
    pg = doc.new_page(width=595, height=842)
    lines = [
        (72, 100, "Informe mensual de honorarios"),
        (72, 140, f"Nombre: {p.full_name}"),
        (72, 170, f"RUT: {rut}"),
        (72, 200, f"Correo: {email}"),
        (72, 230, f"Teléfono: {phone}"),
        (72, 260, f"Monto bruto: {amount}"),
        (72, 290, f"Contraparte: {outside.full_name}"),
    ]
    for x, y, t in lines:
        pg.insert_text((x, y), t, fontsize=12)
    # hidden text: white on white and outside the page
    hidden_rut = f.person().rut("no_dots")
    pg.insert_text((72, 400), hidden_rut, fontsize=12, color=(1, 1, 1))
    hidden_email = f.person().email("initial")
    pg.insert_text((72, 900), hidden_email, fontsize=12)
    if rotation:
        pg.set_rotation(rotation)
    elements = [
        Element("text", 0, _find(pg, "Informe mensual de honorarios"), "Informe mensual de honorarios", layer="text"),
        Element("text", 0, _find(pg, "RUT:"), "RUT:", layer="text"),
        Element("name", 0, _find(pg, p.full_name), p.full_name, layer="text", tags={"in_list": True}),
        Element("rut", 0, _find(pg, rut), rut, layer="text", tags={"format": "dots", "dv_valid": True}),
        Element("email", 0, _find(pg, email), email, layer="text", tags={"format": "dot"}),
        Element("phone", 0, _find(pg, phone), phone, layer="text", tags={"format": "mobile_international"}),
        Element("text", 0, _find(pg, amount), amount, layer="text", tags={"decoy": "amount"}),
        Element(
            "name",
            0,
            _find(pg, outside.full_name),
            outside.full_name,
            level="out_of_scope",
            layer="text",
            tags={"in_list": False},
        ),
        Element("rut", 0, _find(pg, hidden_rut), hidden_rut, layer="hidden", tags={"format": "no_dots"}),
        Element("email", 0, [[72, 890], [200, 890], [200, 905], [72, 905]], hidden_email, layer="hidden"),
    ]
    path = "pdf/texto.pdf" if not rotation else f"pdf/texto_rot{rotation}.pdf"
    _save(doc, dir_ / path)
    return FileEntry(
        id=f"t_pdf_texto_{rotation}",
        path=path,
        format="pdf",
        category="pdf_text",
        description="prueba",
        pages=[Page(0, pg.cropbox.width, pg.cropbox.height, "pt", pg.rotation)],
        elements=elements,
    )


def _redact(
    input_path: Path, output_path: Path, file_entry: FileEntry, fill=(0, 0, 0), include_hidden: bool = True
) -> None:
    doc = pymupdf.open(input_path)
    for e in file_entry.elements:
        if e.type == "text" or e.polygon is None or (e.layer == "hidden" and not include_hidden):
            continue
        r = pymupdf.Quad(*[pymupdf.Point(*pt) for pt in (e.polygon[0], e.polygon[1], e.polygon[3], e.polygon[2])]).rect
        doc[e.page].add_redact_annot(r, fill=fill)
    for pg in doc:
        pg.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
    doc.scrub()
    _save(doc, output_path)


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_pdf_text_identity_everything_leaks(tmp_path, f, rotation):
    inp, out = tmp_path / "in", tmp_path / "out"
    a = _pdf_text(f, inp, rotation)
    shutil.copytree(inp, out)
    ev = evaluate_file(a, _result(a, redactions=[]), out)
    assert ev.status == "processed" and not ev.geometry_mismatch
    for e in ev.elements:
        if e.type == "text":
            assert e.status == "neutral" and e.extractable is True
            continue
        assert e.status == "leak", e
        if e.layer == "text":
            assert set(e.failures) >= {"C", "T", "B", "P"}, e.failures
        else:
            assert set(e.failures) == {"T", "B"}, e.failures


@pytest.mark.parametrize("rotation", [0, 90])
def test_pdf_text_oracle_everything_redacted(tmp_path, f, rotation):
    inp, out = tmp_path / "in", tmp_path / "out"
    a = _pdf_text(f, inp, rotation)
    _redact(inp / a.path, out / a.path, a)
    ev = evaluate_file(a, _result(a), out)
    for e in ev.elements:
        if e.type == "text":
            continue
        assert e.status == "redacted", (e.id, e.failures, e.detail)
        assert e.detected
    # the neutral text is still extractable (preservation)
    assert all(e.extractable for e in ev.elements if e.type == "text")


def test_pdf_text_rectangle_on_top_fails_t_and_b(tmp_path, f):
    """Drawing a black rectangle without removing the text: it looks redacted, but the text is still there."""
    inp, out = tmp_path / "in", tmp_path / "out"
    a = _pdf_text(f, inp)
    doc = pymupdf.open(inp / a.path)
    for e in a.elements:
        if e.type != "text" and e.layer == "text":
            doc[0].draw_rect(pymupdf.Rect(*cov.box(e.polygon)), color=None, fill=(0, 0, 0))
    _save(doc, out / a.path)
    ev = evaluate_file(a, _result(a), out)
    for e in ev.elements:
        if e.type != "text" and e.layer == "text":
            assert e.checks["C"] and e.checks["P"], e
            assert e.failures == ["T", "B"], e.failures


def test_pdf_text_unpainted_zones_fails_p(tmp_path, f):
    """Text removed from the text layer, but the page stayed as an image with the data visible."""
    inp, out = tmp_path / "in", tmp_path / "out"
    a = _pdf_text(f, inp)
    src = pymupdf.open(inp / a.path)
    pix = src[0].get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=False)
    doc = pymupdf.open()
    pg = doc.new_page(width=src[0].rect.width, height=src[0].rect.height)
    pg.insert_image(pg.rect, stream=pix.tobytes("png"))
    _save(doc, out / a.path)
    ev = evaluate_file(a, _result(a), out)
    for e in ev.elements:
        if e.type != "text" and e.layer == "text":
            assert e.checks["T"] and e.checks["B"], (e.id, e.detail)
            assert e.failures == ["P"], (e.id, e.failures, e.detail)


def test_pdf_text_hidden_not_removed(tmp_path, f):
    """Redaction of what is visible without removing the hidden text: T/B leak only in the hidden layer."""
    inp, out = tmp_path / "in", tmp_path / "out"
    a = _pdf_text(f, inp)
    _redact(inp / a.path, out / a.path, a, include_hidden=False)
    ev = evaluate_file(a, _result(a), out)
    for e in ev.elements:
        if e.layer == "hidden":
            assert e.status == "leak" and "T" in e.failures, (e.id, e.failures)
            assert not e.detected
        elif e.type != "text":
            assert e.status == "redacted", (e.id, e.failures)


def test_pdf_optional_layer_off_is_extracted(tmp_path, f):
    inp, out = tmp_path / "in", tmp_path / "out"
    rut = f.person().rut("dots")
    doc = pymupdf.open()
    pg = doc.new_page(width=400, height=400)
    ocg = doc.add_ocg("borrador", on=False)
    pg.insert_text((50, 100), rut, fontsize=12, oc=ocg)
    _save(doc, inp / "ocg.pdf")
    a = FileEntry(
        "t_ocg",
        "ocg.pdf",
        "pdf",
        "pdf_metadata",
        "",
        [Page(0, 400, 400, "pt")],
        [Element("rut", 0, rect(50, 90, 150, 104), rut, layer="hidden")],
    )
    shutil.copytree(inp, out)
    ev = evaluate_file(a, _result(a, redactions=[]), out)
    assert ev.elements[0].status == "leak"
    assert any(h.startswith("text_pymupdf") for h in ev.elements[0].detail["T"])


# ---------------------------------------------------------------------------
# 2. Image with EXIF (orientation, GPS, thumbnail, author)
# ---------------------------------------------------------------------------


def _jpeg_exif(f: FakeData, dir_: Path) -> tuple[FileEntry, Image.Image]:
    p = f.person()
    cv = Canvas.new(640, 360)
    cv.write_text(30, 60, "Comprobante de atención", size=26)
    cv.write_line(30, 130, [("RUT: ", "text"), (p.rut("dots"), "rut")], size=28)
    cv.write_text(30, 200, p.email("dot"), type="email", size=24)
    cv.write_rotated(460, 280, p.phone.format("mobile_national"), 15, type="phone", size=22)
    visible = cv.img
    # saved rotated with EXIF orientation 6: applying the EXIF gives back ``visible``
    stored = visible.transpose(Image.Transpose.ROTATE_90)
    author, description = f.person().full_name, f.person().email("underscore")
    thumbnail = io.BytesIO()
    stored.resize((80, 142)).save(thumbnail, "JPEG")
    exif = {
        "0th": {
            piexif.ImageIFD.Orientation: 6,
            piexif.ImageIFD.Artist: author.encode("utf-8"),
            piexif.ImageIFD.XPComment: description.encode("utf-16-le"),
            piexif.ImageIFD.Software: b"Editor 1.0",
        },
        "GPS": {piexif.GPSIFD.GPSLatitudeRef: b"S", piexif.GPSIFD.GPSLatitude: ((36, 1), (49, 1), (0, 1))},
        "1st": {piexif.ImageIFD.JPEGInterchangeFormat: 0, piexif.ImageIFD.JPEGInterchangeFormatLength: 0},
        "thumbnail": thumbnail.getvalue(),
    }
    path = dir_ / "img" / "exif.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    stored.save(path, "JPEG", quality=92, exif=piexif.dump(exif))
    assert ImageOps.exif_transpose(Image.open(path)).size == visible.size
    a = FileEntry(
        "t_jpg_exif",
        "img/exif.jpg",
        "jpg",
        "images",
        "",
        [Page(0, visible.width, visible.height, "px")],
        cv.elements,
        [
            MetadataEntry("exif.artist", author),
            MetadataEntry("exif.xpcomment", description),
            MetadataEntry("exif.gps", None),
            MetadataEntry("exif.thumbnail", None),
        ],
    )
    Manifest(root=str(dir_), seed=33).add(a)  # assigns ids
    return a, visible


def _paint(img: Image.Image, polygons, color=(0, 0, 0)) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).copy()
    for pol in polygons:
        m = np.zeros(arr.shape[:2], np.uint8)
        cov.fill(m, pol)
        arr[m.astype(bool)] = color
    return Image.fromarray(arr)


def test_image_identity_leaks_pixels_and_metadata(tmp_path, f):
    inp, out = tmp_path / "in", tmp_path / "out"
    a, _ = _jpeg_exif(f, inp)
    shutil.copytree(inp, out)
    ev = evaluate_file(a, _result(a, redactions=[]), out)
    for e in ev.elements:
        if e.type != "text":
            assert e.status == "leak" and set(e.failures) == {"C", "P"}, (e.id, e.failures)
    assert all(m.status == "leak" for m in ev.metadata), [(m.location, m.reasons) for m in ev.metadata]


def test_image_covered_in_report_but_not_in_pixels(tmp_path, f):
    inp, out = tmp_path / "in", tmp_path / "out"
    a, _ = _jpeg_exif(f, inp)
    shutil.copytree(inp, out)
    ev = evaluate_file(a, _result(a), out)  # correct zones, file untouched
    for e in ev.elements:
        if e.type != "text":
            assert e.detected and e.failures == ["P"], (e.id, e.failures)


def test_blurred_image_is_not_redaction(tmp_path, f):
    inp, out = tmp_path / "in", tmp_path / "out"
    a, visible = _jpeg_exif(f, inp)
    arr = np.asarray(visible).copy()
    blurred = np.asarray(visible.filter(ImageFilter.GaussianBlur(3)))
    for e in a.elements:
        if e.type != "text":
            m = np.zeros(arr.shape[:2], np.uint8)
            cov.fill(m, e.polygon)
            arr[m.astype(bool)] = blurred[m.astype(bool)]
    (out / "img").mkdir(parents=True)
    Image.fromarray(arr).save(out / a.path, "JPEG", quality=92)
    ev = evaluate_file(a, _result(a), out)
    assert all(m.status == "removed" for m in ev.metadata)
    assert any(e.failures == ["P"] for e in ev.elements if e.type != "text")


def test_image_oracle_redacted_and_without_metadata(tmp_path, f):
    inp, out = tmp_path / "in", tmp_path / "out"
    a, visible = _jpeg_exif(f, inp)
    clean = _paint(visible, [e.polygon for e in a.elements if e.type != "text"])
    (out / "img").mkdir(parents=True)
    clean.save(out / a.path, "JPEG", quality=90)  # JPEG: the border of the zone has artifacts
    ev = evaluate_file(a, _result(a), out)
    for e in ev.elements:
        if e.type != "text":
            assert e.status == "redacted", (e.id, e.failures, e.detail)
    assert all(m.status == "removed" for m in ev.metadata), [(m.location, m.reasons) for m in ev.metadata]


def test_image_with_other_size_is_geometry_mismatch(tmp_path, f):
    inp, out = tmp_path / "in", tmp_path / "out"
    a, visible = _jpeg_exif(f, inp)
    (out / "img").mkdir(parents=True)
    _paint(visible, [e.polygon for e in a.elements]).resize((320, 180)).save(out / a.path, "JPEG")
    ev = evaluate_file(a, _result(a), out)
    assert ev.geometry_mismatch
    assert all("geometry_mismatch" in e.failures for e in ev.elements if e.type != "text")


def test_png_compressed_text_and_tiff(tmp_path, f):
    """Canaries in zTXt/iTXt (compressed: not visible in the bytes) and in TIFF tags."""
    from PIL import PngImagePlugin, TiffImagePlugin

    inp = tmp_path / "in"
    inp.mkdir()
    name, email, artist = f.person().full_name, f.person().email("dot"), f.person().full_name
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", name, zip=True)
    info.add_itxt("Comment", email, zip=True)
    Image.new("RGB", (50, 40), "white").save(inp / "a.png", pnginfo=info)
    assert name.encode() not in (inp / "a.png").read_bytes()
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[315] = artist
    Image.new("RGB", (50, 40), "white").save(inp / "a.tiff", tiffinfo=ifd)
    files = [
        FileEntry(
            "t_png",
            "a.png",
            "png",
            "images",
            "",
            [Page(0, 50, 40, "px")],
            [],
            [MetadataEntry("png.text.Author", name), MetadataEntry("png.text.Comment", email)],
        ),  # fmt: skip
        FileEntry(
            "t_tiff", "a.tiff", "tiff", "tiff", "", [Page(0, 50, 40, "px")], [], [MetadataEntry("tiff.artist", artist)]
        ),  # fmt: skip
    ]
    for a in files:
        ev = evaluate_file(a, _result(a, redactions=[]), inp)
        assert all(m.status == "leak" for m in ev.metadata), [(m.location, m.reasons) for m in ev.metadata]
    clean = tmp_path / "clean"
    clean.mkdir()
    Image.new("RGB", (50, 40), "white").save(clean / "a.png")
    Image.new("RGB", (50, 40), "white").save(clean / "a.tiff")
    for a in files:
        ev = evaluate_file(a, _result(a, redactions=[]), clean)
        assert all(m.status == "removed" for m in ev.metadata), [(m.location, m.reasons) for m in ev.metadata]


# ---------------------------------------------------------------------------
# 3. PDF with an embedded image (scan with text and a face) covered by a rectangle
# ---------------------------------------------------------------------------


def _pdf_with_image(f: FakeData, dir_: Path) -> FileEntry:
    p = f.person()
    rut = p.rut("dots")
    cv = Canvas.new(800, 400, background=(250, 250, 247))
    cv.write_line(40, 80, [("RUN ", "text"), (rut, "rut")], size=36)
    rng = np.random.default_rng(3)
    face = Image.fromarray((rng.random((160, 128, 3)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))
    cv.img.paste(face, (600, 150))
    buf = io.BytesIO()
    cv.img.save(buf, "PNG")
    doc = pymupdf.open()
    pg = doc.new_page(width=600, height=300)
    dest = pymupdf.Rect(0, 0, 600, 300)  # 800x400 px -> 600x300 pt (0.75 pt/px)
    pg.insert_image(dest, stream=buf.getvalue())
    sc = 600 / 800
    elements = [
        Element(e.type, 0, [[x * sc, y * sc] for x, y in e.polygon], e.value, layer="raster") for e in cv.elements
    ]
    elements.append(
        Element(
            "face",
            0,
            rect(600 * sc, 150 * sc, 728 * sc, 310 * sc),
            None,
            layer="raster",
            core=rect(630 * sc, 190 * sc, 698 * sc, 280 * sc),
            tags={"pose": "front"},
        )
    )
    _save(doc, dir_ / "escaneo.pdf")
    return FileEntry("t_pdf_img", "escaneo.pdf", "pdf", "pdf_scanned", "", [Page(0, 600, 300, "pt")], elements)


def test_pdf_image_covered_with_rectangle_fails_i(tmp_path, f):
    inp, out = tmp_path / "in", tmp_path / "out"
    a = _pdf_with_image(f, inp)
    doc = pymupdf.open(inp / a.path)
    for e in a.elements:
        if e.type != "text":
            doc[0].draw_rect(pymupdf.Rect(*cov.box(e.polygon)), color=None, fill=(0, 0, 0))
    _save(doc, out / a.path)
    ev = evaluate_file(a, _result(a), out, inp)
    for e in ev.elements:
        if e.type != "text":
            assert e.checks["C"] and e.checks["P"], (e.id, e.detail)
            # the original image is still on the page (I) and with the same pixels (O)
            assert e.failures == ["I", "O"], (e.id, e.failures, e.detail)


def test_pdf_image_identity_and_oracle(tmp_path, f):
    inp, out, out2 = tmp_path / "in", tmp_path / "out", tmp_path / "out2"
    a = _pdf_with_image(f, inp)
    shutil.copytree(inp, out)
    ev = evaluate_file(a, _result(a, redactions=[]), out, inp)
    for e in ev.elements:
        if e.type != "text":
            assert set(e.failures) == {"C", "P", "I", "O"}, (e.id, e.failures)
    _redact(inp / a.path, out2 / a.path, a)
    ev = evaluate_file(a, _result(a), out2, inp)
    for e in ev.elements:
        if e.type != "text":
            assert e.status == "redacted", (e.id, e.failures, e.detail)
            assert e.checks["O"] is True


@pytest.mark.parametrize("mode", ["orphan", "incremental"])
def test_pdf_orphan_original_image_fails_o(tmp_path, f, mode):
    """Correct redaction of the pixels, but the original image stays in the file: without
    garbage collection (orphan object) or in the previous revision of an incremental save."""
    inp, out = tmp_path / "in", tmp_path / "out"
    a = _pdf_with_image(f, inp)
    if mode == "orphan":
        doc = pymupdf.open(inp / a.path)
        dest = out / a.path
    else:
        out.mkdir()
        shutil.copy(inp / a.path, out / a.path)
        doc = pymupdf.open(out / a.path)
        dest = None
    for e in a.elements:
        if e.type != "text":
            doc[0].add_redact_annot(pymupdf.Rect(*cov.box(e.polygon)), fill=(0, 0, 0))
    doc[0].apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
    if dest is None:
        doc.saveIncr()
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        doc.save(dest, garbage=0)
    doc.close()
    ev = evaluate_file(a, _result(a), out, inp)
    for e in ev.elements:
        if e.type != "text":
            assert e.checks["I"] and e.checks["P"], (e.id, e.detail)
            assert e.failures == ["O"], (e.id, e.failures, e.detail)
    # without the input folder, O cannot be evaluated and a warning is left
    ev = evaluate_file(a, _result(a), out)
    assert any("comprobación O" in x for x in ev.warnings)


def test_face_core_and_box_rule(tmp_path, f):
    inp, out = tmp_path / "in", tmp_path / "out"
    a = _pdf_with_image(f, inp)
    face = next(e for e in a.elements if e.type == "face")
    x0, y0, x1, y1 = cov.box(face.polygon)
    shutil.copytree(inp, out)
    core_only = [Redaction(0, face.core, "face", "test")]
    ev = evaluate_file(a, _result(a, redactions=core_only), out)
    r = _by_type(ev, "face")[0]
    assert r.core_coverage == pytest.approx(1.0) and r.coverage < 0.8 and not r.detected
    almost = [Redaction(0, rect(x0, y0 + 0.1 * (y1 - y0), x1, y1), "face", "test")]  # 90 % of the face
    ev = evaluate_file(a, _result(a, redactions=almost), out)
    assert _by_type(ev, "face")[0].detected


# ---------------------------------------------------------------------------
# 4. PDF with vectorized text (glyphs as paths)
# ---------------------------------------------------------------------------


def _pdf_vector(f: FakeData, dir_: Path) -> FileEntry:
    p = f.person()
    rut, phone = (
        p.rut("no_dots"),
        p.landline.format("santiago_international" if p.landline.kind == "santiago" else "regional_international"),
    )
    doc = pymupdf.open()
    pg = doc.new_page(width=420, height=300)
    pg.insert_text((40, 80), "Certificado", fontsize=16)
    pg.insert_text((40, 130), f"RUT {rut}", fontsize=14)
    pg.insert_text((40, 170), f"Fono {phone}", fontsize=14)
    elements = [
        Element("text", 0, _find(pg, "Certificado"), "Certificado", layer="vector"),
        Element("rut", 0, _find(pg, rut), rut, layer="vector"),
        Element("phone", 0, _find(pg, phone), phone, layer="vector"),
    ]
    svg = pg.get_svg_image(text_as_path=True)
    vec = pymupdf.open("pdf", pymupdf.open(stream=svg.encode(), filetype="svg").convert_to_pdf())
    assert vec[0].get_text().strip() == ""
    _save(vec, dir_ / "vector.pdf")
    return FileEntry("t_pdf_vec", "vector.pdf", "pdf", "pdf_text", "", [Page(0, 420, 300, "pt")], elements)


def test_pdf_vector(tmp_path, f):
    inp = tmp_path / "in"
    a = _pdf_vector(f, inp)
    # identity: the glyphs remain
    ev = evaluate_file(a, _result(a, redactions=[]), inp)
    for e in ev.elements:
        if e.type != "text":
            assert set(e.failures) == {"C", "P", "V"}, (e.id, e.failures)
    # black rectangle on top of the strokes: it looks fine but the glyphs are still in the file
    covered = tmp_path / "covered"
    doc = pymupdf.open(inp / a.path)
    for e in a.elements:
        if e.type != "text":
            doc[0].draw_rect(pymupdf.Rect(*cov.box(e.polygon)), color=None, fill=(0, 0, 0))
    _save(doc, covered / a.path)
    ev = evaluate_file(a, _result(a), covered)
    for e in ev.elements:
        if e.type != "text":
            assert e.failures == ["V"], (e.id, e.failures, e.detail)
    # redaction that removes the covered strokes
    clean = tmp_path / "clean"
    doc = pymupdf.open(inp / a.path)
    for e in a.elements:
        if e.type != "text":
            x0, y0, x1, y1 = cov.box(e.polygon)
            doc[0].add_redact_annot(pymupdf.Rect(x0 - 1, y0 - 2, x1 + 1, y1 + 2), fill=(0, 0, 0))
    doc[0].apply_redactions(graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED)
    _save(doc, clean / a.path)
    ev = evaluate_file(a, _result(a), clean)
    for e in ev.elements:
        if e.type != "text":
            assert e.status == "redacted", (e.id, e.failures, e.detail)


# ---------------------------------------------------------------------------
# 5. PDF metadata and previous revisions
# ---------------------------------------------------------------------------


def test_pdf_metadata_and_previous_revision(tmp_path, f):
    inp, out = tmp_path / "in", tmp_path / "out"
    author, note, attachment, bookmark = (f.person().full_name for _ in range(4))
    previous_rut = f.person().rut("dots")
    doc = pymupdf.open()
    pg = doc.new_page(width=300, height=300)
    pg.insert_text((40, 60), previous_rut, fontsize=11)
    inp.mkdir()
    doc.set_metadata({"author": author, "producer": "Generador"})
    pg.add_text_annot((100, 100), note)
    doc.embfile_add("respaldo.txt", f"Responsable: {attachment}".encode("utf-16"))
    doc.set_toc([[1, bookmark, 1]])
    doc.save(inp / "meta.pdf")
    # incremental save that removes the text: the previous version is still in the file
    doc = pymupdf.open(inp / "meta.pdf")
    doc[0].add_redact_annot(doc[0].search_for(previous_rut)[0])
    doc[0].apply_redactions()
    doc.save(inp / "meta.pdf", incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)
    a = FileEntry(
        "t_pdf_meta",
        "meta.pdf",
        "pdf",
        "pdf_metadata",
        "",
        [Page(0, 300, 300, "pt")],
        [],
        [
            MetadataEntry("pdf.info.author", author),
            MetadataEntry("pdf.annotation", note),
            MetadataEntry("pdf.attachment", attachment),
            MetadataEntry("pdf.bookmark", bookmark),
            MetadataEntry("pdf.previous_revision", previous_rut, {"type": "rut"}),
        ],
    )
    shutil.copytree(inp, out)
    ev = evaluate_file(a, _result(a, redactions=[]), out)
    assert all(m.status == "leak" for m in ev.metadata), [(m.location, m.reasons) for m in ev.metadata]
    revision = ev.metadata[-1]
    assert any("canario" in x for x in revision.reasons) and any("estructura" in x for x in revision.reasons)
    assert any("Producer" in w or "producer" in w for w in ev.warnings)
    # complete cleanup: no metadata, annotations, attachments, bookmarks or revisions
    clean = tmp_path / "clean"
    doc = pymupdf.open(inp / "meta.pdf")
    doc.set_metadata({})
    doc.del_xml_metadata()
    for page in doc:
        for annot in list(page.annots()):
            page.delete_annot(annot)
    for name in doc.embfile_names():
        doc.embfile_del(name)
    doc.set_toc([])
    _save(doc, clean / "meta.pdf")
    ev = evaluate_file(a, _result(a, redactions=[]), clean)
    assert all(m.status == "removed" for m in ev.metadata), [(m.location, m.reasons) for m in ev.metadata]


# ---------------------------------------------------------------------------
# 6. Search for canaries and fragments
# ---------------------------------------------------------------------------


def test_critical_fragments():
    parts = {a.part: a.form for a in needles("rut", "12.345.678-5")}
    assert parts == {"value": "123456785", "rut_body": "12345678"}
    assert {a.part: a.form for a in needles("phone", "+56 9 8123 4567")}["last7"] == "1234567"
    assert {a.part: a.form for a in needles("email", "Ana.Rojas@ejemplo.cl")}["local@"] == "ana.rojas@"
    assert {a.part: a.form for a in needles("email", "ana.rojas [arroba] ejemplo.cl")}["local@"] == "ana.rojas[arroba]"
    assert {a.part: a.form for a in needles("name", "Ana María Rojas Peña")}["surnames"] == "rojas pena"
    assert {a.part: a.form for a in needles("name", "ROJAS PEÑA, Ana María")}["surnames"] == "rojas pena"
    street = {a.part: a.form for a in needles("address", "Pasaje Los Alerces 1234, depto 56")}["street_number"]
    assert street == "pasaje los alerces 1234"


@pytest.mark.parametrize(
    "text",
    ["RUT 12,345,678-5", "12 345 678 - 5", "12.345.678–5", "1 2 . 3 4 5 . 6 7 8 - 5", "N°12345678K otro"],
)
def test_rut_in_text_with_separators(text):
    haystack = Haystack()
    haystack.add("t", text)
    assert haystack.contains(needles("rut", "12.345.678-5"))


def test_no_false_positives_from_joining_numbers():
    haystack = Haystack()
    haystack.add("t", "Folio 1234 del 5 de mayo; monto $ 678.000")
    assert not haystack.contains(needles("rut", "12.345.678-5"))


def test_stream_numbers_are_not_joined():
    """In raw bytes (content operators) digits are not joined: the value is searched as it was written."""
    phone = "+56 9 8123 4567"
    haystack = Haystack()
    haystack.add("streams", "q 1 0 0 1 98 12 34567 cm Q", binary=True)
    assert not haystack.contains(needles("phone", phone))
    haystack.add("streams", "BT (+56 9 8123 4567) Tj ET", binary=True)
    assert haystack.contains(needles("phone", phone))
    rut = Haystack()
    rut.add("streams", "<x:rut>12.345.678</x:rut>", binary=True)  # body without DV, as it was written
    assert rut.contains(needles("rut", "12.345.678-5"))


def test_bytes_in_several_encodings():
    value = "Ana María Rojas Peña"
    for enc in ("utf-8", "latin-1", "utf-16-le", "utf-16-be"):
        data = b"\x00\x01basura" + value.encode(enc) + b"\xff\xfe"
        haystack = Haystack()
        haystack.add("b", texts_from_bytes(data))
        assert haystack.contains(needles("name", value)), enc


def test_pdf_strings_joined_in_tj():
    stream = rb"BT /F1 12 Tf 72 700 Td [(12.3) -20 (45.6) 15 (78-5)] TJ ET BT (Pe\361a) Tj ET <FEFF00410042>"
    texts = pdf_strings(stream)
    assert "12.345.678-5" in texts and "Peña" in texts and "AB" in texts


def test_rasterized_coverage():
    pol = [[10, 10], [110, 10], [110, 30], [10, 30]]
    assert cov.coverage(pol, [pol], 8) == pytest.approx(1.0)
    assert cov.coverage(pol, [rect(10, 10, 60, 30)], 8) == pytest.approx(0.5, abs=0.01)
    assert cov.coverage(pol, [rect(10, 10, 60, 30), rect(55, 10, 110, 30)], 8) == pytest.approx(1.0)
    rotated = [[0, 0], [10, 10], [0, 20], [-10, 10]]
    assert cov.coverage(rotated, [rect(-10, 0, 10, 20)], 8) == pytest.approx(1.0)
    assert cov.coverage(rect(-10, 0, 10, 20), [rotated], 8) == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# 7. Full dataset: expected errors, missing files, aggregation and report
# ---------------------------------------------------------------------------


def test_full_dataset_and_report(tmp_path, f):
    inp = tmp_path / "in"
    man = Manifest(root=str(inp), seed=33)
    a_pdf = man.add(_pdf_text(f, inp))
    a_img, visible = _jpeg_exif(f, inp)
    a_img.id = "t_jpg_exif2"
    man.add(a_img)
    (inp / "clave.pdf").write_bytes(b"%PDF-1.7 cifrado")
    a_err = man.add(FileEntry("t_err", "clave.pdf", "pdf", "errors", "", [], expected="error:password"))
    a_err2 = man.add(FileEntry("t_err2", "vacio.pdf", "pdf", "errors", "", [], expected="error:empty"))
    man_path = man.save()
    man = Manifest.load(man_path)

    # system that redacts the PDF and the image correctly, and correctly rejects one error
    out = tmp_path / "out"
    _redact(inp / a_pdf.path, out / a_pdf.path, a_pdf)
    (out / "img").mkdir(parents=True, exist_ok=True)
    _paint(visible, [e.polygon for e in a_img.elements if e.type != "text"]).save(out / a_img.path, "PNG")
    report = RedactionReport(
        system="test",
        results=[
            _result(a_pdf),
            _result(a_img),
            FileResult(input=a_err.path, output=None, error="password: el archivo pide clave"),
            FileResult(input=a_err2.path, output=None, error="empty"),
        ],
        details={"machine": "CPU de prueba"},
    )
    r = evaluate(man, report, out, processes=1)
    assert r["verdict"]["passed"], r["verdict"]
    assert r["summary"]["leaks"] == 0 and r["summary"]["recall"] == 1.0
    assert r["expected_errors"]["correct"] == 2
    json_path, md_path = write_report(r, tmp_path / "report")
    assert json.loads(json_path.read_text(encoding="utf-8"))["verdict"]["passed"]
    assert "APROBADO" in md_path.read_text(encoding="utf-8")

    # identity: everything leaks, no result for the image and an expected error not rejected
    ident = tmp_path / "ident"
    shutil.copytree(inp, ident)
    report = RedactionReport(
        system="identity",
        results=[_result(a_pdf, redactions=[]), FileResult(input=a_err.path, output=a_err.path)],
    )
    r = evaluate(man, report, ident, processes=2)
    v = r["verdict"]
    assert not v["passed"]
    # the image metadata does not count: there is no output to publish
    assert [c["met"] for c in v["criteria"]] == [False, True, False, True, True]
    assert r["summary"]["critical_leaks"] == 5  # 3 visible + 2 hidden
    assert r["processable_not_processed"][0]["file"] == "t_jpg_exif2"
    assert {x["status"] for x in r["expected_errors"]["files"]} == {"not_rejected", "no_result"}
    img_elems = [e for e in r["elements"] if e["file"] == "t_jpg_exif2" and e["type"] != "text"]
    assert img_elems and all(e["status"] == "not_processed" for e in img_elems)
    _, md_path = write_report(r, tmp_path / "report_identity")
    md = md_path.read_text(encoding="utf-8")
    assert "NO APROBADO" in md and "Detalle de fugas" in md
    # the Markdown report shows the codes with their Spanish labels
    assert "`identidad`" in md and "| correo | base | texto |" in md
    for e in a_pdf.elements:
        if e.type in ("rut", "email", "phone") and e.layer == "text":
            assert e.value in md


# ---------------------------------------------------------------------------
# 8. Controls against the input, rotated pages re-saved and metadata locations
# ---------------------------------------------------------------------------


def _low_contrast_png(f: FakeData, dir_: Path) -> FileEntry:
    """A very light gray RUT: with the P tolerance (40 levels) the zone looks uniform."""
    cv = Canvas.new(420, 120)
    cv.write_text(20, 70, f.person().rut("dots"), type="rut", size=34, color=(222, 222, 222))
    path = dir_ / "bajo_contraste.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.img.save(path)
    a = FileEntry("t_png_bajo", "bajo_contraste.png", "png", "images", "", [Page(0, 420, 120, "px")], cv.elements)
    Manifest(root=str(dir_), seed=33).add(a)
    return a


def test_low_contrast_image_unchanged_fails_p(tmp_path, f):
    inp, out = tmp_path / "in", tmp_path / "out"
    a = _low_contrast_png(f, inp)
    shutil.copytree(inp, out)
    # without the input, P cannot tell the faint data from a painted zone (limit of the tolerance)
    ev = evaluate_file(a, _result(a), out)
    assert ev.elements[0].checks["P"] is True
    # with the input, the pixels equal to the original reveal it was not redacted
    ev = evaluate_file(a, _result(a), out, inp)
    e = ev.elements[0]
    assert e.failures == ["P"] and e.detail["original_correlation"] > 0.99, (e.failures, e.detail)
    # recompressed as JPEG without touching the zone: it is still the original
    Image.open(inp / a.path).save(out / a.path, "JPEG", quality=70)
    assert evaluate_file(a, _result(a), out, inp).elements[0].failures == ["P"]
    # filled zone (even with a color similar to the background): redacted
    _paint(Image.open(inp / a.path), [a.elements[0].polygon], (240, 240, 240)).save(out / a.path, "PNG")
    e = evaluate_file(a, _result(a), out, inp).elements[0]
    assert e.status == "redacted", (e.failures, e.detail)


def _remove_rotation(doc: pymupdf.Document, index: int = 0) -> None:
    """Leaves the page with /Rotate 0 and the content rotated inside the stream: it looks the same as before."""
    pg = doc[index]
    assert pg.rotation == 90
    width, height = pg.mediabox.width, pg.mediabox.height
    pg.clean_contents()
    xref = pg.get_contents()[0]
    # 90° clockwise rotation: (x, y) -> (y, width - x) in PDF user space
    doc.update_stream(xref, f"q 0 -1 1 0 0 {width:g} cm\n".encode() + doc.xref_stream(xref) + b"\nQ")
    pg.set_rotation(0)
    pg.set_mediabox(pymupdf.Rect(0, 0, height, width))


def test_rotated_pdf_resaved_without_rotate(tmp_path, f):
    """The output draws the page rotated 90° on a sheet without /Rotate (width and height swapped):
    the visible page is the same, so it is not a geometry mismatch."""
    inp, out = tmp_path / "in", tmp_path / "out"
    a = _pdf_text(f, inp, rotation=90)
    clean = tmp_path / "clean"
    _redact(inp / a.path, clean / a.path, a)
    doc = pymupdf.open(clean / a.path)
    before = doc[0].get_pixmap(alpha=False).samples
    _remove_rotation(doc)
    assert doc[0].rotation == 0 and doc[0].rect.width > doc[0].rect.height
    assert doc[0].get_pixmap(alpha=False).samples == before  # same visible page
    _save(doc, out / a.path)
    ev = evaluate_file(a, _result(a), out, inp)
    assert not ev.geometry_mismatch
    for e in ev.elements:
        if e.type != "text":
            assert e.status == "redacted", (e.id, e.failures, e.detail)
    # the input as it is but on a sheet of another size: geometry mismatch
    doc = pymupdf.open()
    doc.new_page(width=500, height=500).show_pdf_page(pymupdf.Rect(0, 0, 500, 500), pymupdf.open(clean / a.path), 0)
    _save(doc, out / a.path)
    ev = evaluate_file(a, _result(a), out, inp)
    assert ev.geometry_mismatch
    assert all("geometry_mismatch" in e.failures for e in ev.elements if e.type != "text" and e.layer != "hidden")


def test_rotated_pdf_rectangle_over_vector(tmp_path, f):
    """V also works if the output saved the rotated page without /Rotate."""
    inp, out = tmp_path / "in", tmp_path / "out"
    a = _pdf_vector(f, inp)
    src = pymupdf.open(stream=(inp / a.path).read_bytes(), filetype="pdf")
    src[0].set_rotation(90)
    _save(src, inp / a.path)
    a.pages[0].rotation = 90
    matrix = src[0].rotation_matrix
    _remove_rotation(src)
    pg = src[0]
    for e in a.elements:
        if e.type != "text":
            vis = [pymupdf.Point(x, y) * matrix for x, y in e.polygon]
            pg.draw_rect(pymupdf.Rect(vis[0], vis[2]).normalize(), color=None, fill=(0, 0, 0))
    _save(src, out / a.path)
    ev = evaluate_file(a, _result(a), out, inp)
    assert not ev.geometry_mismatch
    for e in ev.elements:
        if e.type != "text":
            assert e.failures == ["V"], (e.id, e.failures, e.detail)


def test_metadata_locations_with_container():
    from test_bench.evaluation.core import structure_present
    from test_bench.evaluation.image import tiff_structure

    st = {"exif.gps": True, "xmp": True, "tiff.270": True}
    assert structure_present("png.exif.gps", st) is True
    assert structure_present("webp.xmp", st) is True
    assert structure_present("exif.thumbnail", st) is False
    assert tiff_structure("tiff.image_description") == "tiff.270"
    assert structure_present("tiff.image_description", st) is True
    assert structure_present("other.location", st) is None
