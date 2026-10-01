"""Tests of the three baselines (identity, oracle, notebook) with a minimal dataset built here."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import piexif
import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageOps

from test_bench import visualize
from test_bench.baseline import main, run
from test_bench.baselines import notebook, oracle
from test_bench.canvas import font
from test_bench.fake_data import FakeData
from test_bench.schema import Element, FileEntry, Manifest, Page, RedactionReport, normalize

# Test document of the notebook (cell 6), copied as is.
NOTEBOOK_TEXT = """INFORME DE HONORARIOS - AGOSTO 2026

Nombre: Ana Maria Rojas Pena
RUT: 15.782.334-9
Correo: ana.rojas@ejemplo.cl
Telefono: +56 9 8123 4567
Direccion: Pasaje Los Alerces 442, depto 31

Producto 1: Informe de avance del programa
Monto bruto: $ 1.450.000

Contraparte: Jefatura de la unidad
RUT contraparte: 9.876.543-3
Sitio: https://www.ejemplo.cl/rendiciones
"""

NOTEBOOK_FINDINGS = [
    (1, "rut", "15.782.334-9"),
    (1, "rut", "9.876.543-3"),
    (1, "email", "ana.rojas@ejemplo.cl"),
    (1, "phone", "+56 9 8123 4567"),
    (1, "url", "https://www.ejemplo.cl/rendiciones"),
]


# ---------------------------------------------------------------------------
# Building the minimal dataset
# ---------------------------------------------------------------------------


def _quad(q: pymupdf.Quad) -> list[list[float]]:
    return [[q.ul.x, q.ul.y], [q.ur.x, q.ur.y], [q.lr.x, q.lr.y], [q.ll.x, q.ll.y]]


def _text_elements(page: pymupdf.Page, index: int, data: list[tuple[str, str, dict]]) -> list[Element]:
    """Records each (type, value, tags) with the quadrilateral given by ``search_for``."""
    output = []
    for type_, value, tags in data:
        quads = page.search_for(value, quads=True)
        assert quads, value
        output.append(Element(type=type_, page=index, polygon=_quad(quads[0]), value=value, layer="text", tags=tags))
    return output


def _text_pdf(root: Path, f: FakeData) -> FileEntry:
    p = f.person()
    rut, email, phone = p.rut("dots"), p.email("dot"), p.phone.format("mobile_international")
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    y = 90
    lines = [f"Nombre: {p.full_name}", f"RUT: {rut}", f"Correo: {email}", f"Telefono: {phone}"]
    for line in lines:
        page.insert_text((60, y), line, fontsize=11, fontname="helv")
        y += 20
    # Signature as an embedded image (raster layer inside the PDF).
    signature = Image.new("RGB", (300, 100), "white")
    d = ImageDraw.Draw(signature)
    d.line([(10, 80), (80, 20), (150, 70), (220, 15), (290, 60)], fill=(10, 10, 60), width=6)
    buf = io.BytesIO()
    signature.save(buf, "PNG")
    box = pymupdf.Rect(300, 300, 450, 350)
    page.insert_image(box, stream=buf.getvalue())
    doc.set_metadata({"author": p.full_name, "title": "informe"})
    doc.set_xml_metadata(f"<x:xmpmeta xmlns:x='adobe:ns:meta/'><dc>{p.full_name}</dc></x:xmpmeta>")
    elements = _text_elements(
        page,
        0,
        [
            ("text", "Nombre:", {}),
            ("name", p.full_name, {"in_list": True}),
            ("text", "RUT:", {}),
            ("rut", rut, {"format": "dots", "dv_valid": True}),
            ("email", email, {"format": "dot"}),
            ("phone", phone, {"format": "mobile_international", "kind": "mobile"}),
        ],
    )
    elements.append(
        Element(
            type="signature",
            page=0,
            polygon=[[300, 300], [450, 300], [450, 350], [300, 350]],
            level="out_of_scope",
            layer="raster",
        )
    )
    (root / "pdf").mkdir(parents=True, exist_ok=True)
    doc.save(root / "pdf/texto.pdf")
    doc.close()
    return FileEntry(
        id="lb_texto",
        path="pdf/texto.pdf",
        format="pdf",
        category="pdf_text",
        description="PDF con capa de texto y firma incrustada",
        pages=[Page(0, 595, 842, "pt")],
        elements=elements,
    )


def _rotated_pdf(root: Path, f: FakeData) -> FileEntry:
    doc = pymupdf.open()
    pages, elements = [], []
    for i, rotation in enumerate((90, 270, 180)):
        p = f.person()
        rut = p.rut("no_dots")
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 100 + 40 * i), f"RUT: {rut}", fontsize=12, fontname="helv")
        page.insert_text((72, 300), "Texto neutro de la pagina", fontsize=12, fontname="helv")
        page.set_rotation(rotation)
        elements += _text_elements(
            page,
            i,
            [
                ("rut", rut, {"format": "no_dots", "dv_valid": True}),
                ("text", "Texto neutro de la pagina", {}),
            ],
        )
        pages.append(Page(i, 612, 792, "pt", rotation))
    doc.save(root / "pdf/rotada.pdf")
    doc.close()
    return FileEntry(
        id="lb_rotada",
        path="pdf/rotada.pdf",
        format="pdf",
        category="pdf_text",
        description="páginas con /Rotate 90, 270 y 180",
        pages=pages,
        elements=elements,
    )


def _draw_datum(img: Image.Image, text: str, x: int, y: int, size: int = 28) -> list[list[float]]:
    d = ImageDraw.Draw(img)
    f = font("sans", size)
    d.text((x, y), text, fill=(0, 0, 0), font=f)
    x0, y0, x1, y1 = d.textbbox((x, y), text, font=f)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _exif_jpeg(root: Path, f: FakeData) -> FileEntry:
    p = f.person()
    rut, email = p.rut("dots"), p.email("initial")
    visible = Image.new("RGB", (640, 420), (235, 235, 230))
    rut_polygon = _draw_datum(visible, f"RUT {rut}", 40, 60)
    email_polygon = _draw_datum(visible, email, 40, 300, 22)
    stored = visible.transpose(Image.Transpose.ROTATE_90)  # EXIF 6 puts it back in place when displayed
    exif = piexif.dump(
        {
            "0th": {piexif.ImageIFD.Orientation: 6, piexif.ImageIFD.Artist: p.full_name.encode()},
            "GPS": {
                piexif.GPSIFD.GPSLatitudeRef: b"S",
                piexif.GPSIFD.GPSLatitude: ((33, 1), (26, 1), (0, 1)),
                piexif.GPSIFD.GPSLongitudeRef: b"W",
                piexif.GPSIFD.GPSLongitude: ((70, 1), (39, 1), (0, 1)),
            },
        }
    )
    (root / "img").mkdir(parents=True, exist_ok=True)
    stored.save(root / "img/exif6.jpg", "JPEG", quality=92, exif=exif)
    return FileEntry(
        id="lb_exif6",
        path="img/exif6.jpg",
        format="jpg",
        category="images",
        description="JPEG con orientación EXIF 6 y GPS",
        pages=[Page(0, 640, 420, "px")],
        elements=[
            Element(type="rut", page=0, polygon=rut_polygon, value=rut, tags={"format": "dots"}),
            Element(type="email", page=0, polygon=email_polygon, value=email, tags={"format": "initial"}),
        ],
    )


def _multipage_tiff(root: Path, f: FakeData) -> FileEntry:
    frames, pages, elements = [], [], []
    for i, size in enumerate(((500, 300), (360, 520))):
        img = Image.new("RGB", size, "white")
        phone = f.phone("mobile").format("mobile_national")
        elements.append(
            Element(
                type="phone",
                page=i,
                polygon=_draw_datum(img, phone, 30, 40 + 60 * i),
                value=phone,
                tags={"format": "mobile_national", "kind": "mobile"},
            )
        )
        frames.append(img)
        pages.append(Page(i, size[0], size[1], "px"))
    frames[0].save(
        root / "img/multi.tiff", "TIFF", save_all=True, append_images=frames[1:], tiffinfo={315: "Autor Ficticio"}
    )
    return FileEntry(
        id="lb_tiff",
        path="img/multi.tiff",
        format="tiff",
        category="tiff",
        description="TIFF de dos páginas",
        pages=pages,
        elements=elements,
    )


def _password_pdf(root: Path) -> FileEntry:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Documento protegido con clave de usuario", fontname="helv")
    doc.save(root / "pdf/clave.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="usuario", owner_pw="dueno")
    doc.close()
    return FileEntry(
        id="lb_clave",
        path="pdf/clave.pdf",
        format="pdf",
        category="errors",
        description="PDF con clave de usuario",
        pages=[Page(0, 595, 842, "pt")],
        expected="error:password",
    )


def _permissions_pdf(root: Path, f: FakeData) -> FileEntry:
    """Owner password only (permission restrictions): it opens without a password and must be processed."""
    p = f.person()
    email = p.email("underscore")
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 120), f"Contacto: {email}", fontsize=10, fontname="helv")
    elements = _text_elements(page, 0, [("email", email, {"format": "underscore"})])
    doc.save(root / "pdf/permisos.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_128, owner_pw="dueno", permissions=0)
    doc.close()
    return FileEntry(
        id="lb_permisos",
        path="pdf/permisos.pdf",
        format="pdf",
        category="pdf_text",
        description="PDF con clave de dueño solamente",
        pages=[Page(0, 595, 842, "pt")],
        elements=elements,
    )


def _scan_pdf(root: Path) -> FileEntry:
    img = Image.new("RGB", (850, 1100), "white")
    ImageDraw.Draw(img).rectangle([100, 100, 700, 140], fill=(40, 40, 40))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(root / "pdf/escaneo.pdf")
    doc.close()
    return FileEntry(
        id="lb_escaneo",
        path="pdf/escaneo.pdf",
        format="pdf",
        category="pdf_scanned",
        description="PDF solo imagen",
        pages=[Page(0, 612, 792, "pt")],
    )


def _notebook_pdf(root: Path) -> FileEntry:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((60, 70), NOTEBOOK_TEXT, fontsize=11, fontname="helv")
    doc.save(root / "pdf/cuaderno.pdf")
    doc.close()
    with pymupdf.open(root / "pdf/cuaderno.pdf") as doc:
        elements = _text_elements(
            doc[0], 0, [("rut", "15.782.334-9", {}), ("rut", "9.876.543-3", {}), ("name", "Ana Maria Rojas Pena", {})]
        )
        width, height = doc[0].rect.width, doc[0].rect.height
    return FileEntry(
        id="lb_cuaderno",
        path="pdf/cuaderno.pdf",
        format="pdf",
        category="pdf_text",
        description="documento de prueba del cuaderno",
        pages=[Page(0, width, height, "pt")],
        elements=elements,
    )


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Manifest:
    root = tmp_path_factory.mktemp("dataset")
    f = FakeData(33).derive("test_lineas_base")
    man = Manifest(root=str(root), seed=33)
    for entry in (
        _text_pdf(root, f),
        _rotated_pdf(root, f),
        _exif_jpeg(root, f),
        _multipage_tiff(root, f),
        _password_pdf(root),
        _permissions_pdf(root, f),
        _scan_pdf(root),
        _notebook_pdf(root),
    ):
        man.add(entry)
    man.name_list = ["Ana Maria Rojas Pena", *(p.full_name for p in f.people)]
    assert man.validate() == []
    man.save()
    return man


def _by_id(man: Manifest, id: str) -> FileEntry:
    return next(f for f in man.files if f.id == id)


def _result(report: RedactionReport, path: str):
    return next(r for r in report.results if r.input == path)


def _black_fraction(img: Image.Image, points: list[tuple[float, float]], threshold: int = 40) -> float:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    m = np.asarray(mask) > 0
    assert m.any()
    dark = np.asarray(img.convert("RGB")).max(axis=2) < threshold
    return float(dark[m].mean())


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_identity_copies_byte_for_byte(dataset: Manifest, tmp_path: Path) -> None:
    report = run("identity", dataset, tmp_path)
    assert len(report.results) == len(dataset.files)
    for f, r in zip(dataset.files, report.results, strict=True):
        assert r.error is None and r.redactions == []
        assert r.output == f.path
        source = (Path(dataset.root) / f.path).read_bytes()
        assert (tmp_path / "files" / r.output).read_bytes() == source


# ---------------------------------------------------------------------------
# oracle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle_run(dataset: Manifest, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, RedactionReport]:
    output = tmp_path_factory.mktemp("oracle")
    return output, run("oracle", dataset, output)


def test_oracle_expected_errors(dataset: Manifest, oracle_run) -> None:
    output, report = oracle_run
    r = _result(report, "pdf/clave.pdf")
    assert r.error == "password" and r.output is None
    assert not (output / "files/pdf/clave.pdf").exists()
    for f in dataset.files:
        if f.expected == "process":
            assert _result(report, f.path).error is None, f.path


def test_oracle_pdf_without_text_or_metadata(dataset: Manifest, oracle_run) -> None:
    output, report = oracle_run
    for f in dataset.files:
        if f.format != "pdf" or f.expected != "process":
            continue
        r = _result(report, f.path)
        path = output / "files" / r.output
        with pymupdf.open(Path(dataset.root) / f.path) as original, pymupdf.open(path) as doc:
            assert doc.page_count == original.page_count == r.pages_processed
            for p_o, p_s in zip(original, doc, strict=True):
                assert p_s.rotation == 0
                assert abs(p_s.rect.width - p_o.rect.width) < 0.01
                assert abs(p_s.rect.height - p_o.rect.height) < 0.01
                assert p_s.get_text().strip() == ""
            assert all(not v for k, v in doc.metadata.items() if k not in ("format", "encryption"))
            assert doc.get_xml_metadata() == ""
            assert not doc.is_encrypted
        data = path.read_bytes()
        for e in f.elements:
            if e.value:
                assert e.value.encode() not in data
        # The reported redactions are the ground-truth polygons.
        targets = [e for e in f.elements if e.type != "text"]
        assert [c.polygon for c in r.redactions] == [e.polygon for e in targets]
        assert {c.detector for c in r.redactions} <= {"oracle"}


def test_oracle_pdf_black_pixels(dataset: Manifest, oracle_run) -> None:
    """At 144 dpi and in visible orientation, every personal-data polygon is black; neutral text is not."""
    output, _ = oracle_run
    for id in ("lb_texto", "lb_rotada", "lb_cuaderno", "lb_permisos"):
        f = _by_id(dataset, id)
        pages = visualize.visible_pages(output / "files" / f.path, "pdf")
        originals = visualize.visible_pages(Path(dataset.root) / f.path, "pdf")
        for e in f.elements:
            pts = visualize.to_visual(f, e.page, e.polygon)
            if e.type == "text":
                assert _black_fraction(pages[e.page], pts) < 0.5, (id, e.value)
            else:
                assert _black_fraction(pages[e.page], pts) >= 0.99, (id, e.value)
                # In the original the polygon has ink, but it is not a black block (the mapping is right).
                assert _black_fraction(originals[e.page], pts) < 0.9, (id, e.value)


def test_oracle_exif_jpeg(dataset: Manifest, oracle_run) -> None:
    output, report = oracle_run
    f = _by_id(dataset, "lb_exif6")
    path = output / "files" / f.path
    with Image.open(path) as img:
        assert img.format == "JPEG"
        assert img.size == (640, 420)  # geometry already corrected
        assert len(img.getexif()) == 0
        assert "exif" not in img.info and "icc_profile" not in img.info
        assert ImageOps.exif_transpose(img).size == (640, 420)
        img.load()
        for e in f.elements:
            assert _black_fraction(img, [tuple(p) for p in e.polygon]) >= 0.99
    data = path.read_bytes()
    assert b"Exif" not in data
    assert f.elements[0].value.encode() not in data
    assert len(_result(report, f.path).redactions) == 2


def test_oracle_multipage_tiff(dataset: Manifest, oracle_run) -> None:
    output, report = oracle_run
    f = _by_id(dataset, "lb_tiff")
    path = output / "files" / f.path
    pages = visualize.visible_pages(path, "tiff")
    assert [p.size for p in pages] == [(pg.width, pg.height) for pg in f.pages]
    for e in f.elements:
        assert _black_fraction(pages[e.page], [tuple(p) for p in e.polygon]) >= 0.99
    with Image.open(path) as img:
        assert img.n_frames == 2
        assert 315 not in img.tag_v2  # no Artist
    assert b"Autor Ficticio" not in path.read_bytes()
    assert _result(report, f.path).pages_processed == 2


def test_oracle_mapping_matches_visualizer(dataset: Manifest) -> None:
    """The oracle's transform (rotation_matrix) matches ``visualize.to_visual``."""
    f = _by_id(dataset, "lb_rotada")
    with pymupdf.open(Path(dataset.root) / f.path) as doc:
        for e in f.elements:
            page = doc[e.page]
            m = page.rotation_matrix * pymupdf.Matrix(visualize.PDF_SCALE, visualize.PDF_SCALE)
            own = [tuple(pymupdf.Point(x, y) * m) for x, y in e.polygon]
            ref = visualize.to_visual(f, e.page, e.polygon)
            assert np.allclose(own, ref, atol=1e-3)


def test_oracle_targets_exclude_text_and_hidden() -> None:
    f = FileEntry(
        id="x",
        path="x.pdf",
        format="pdf",
        category="c",
        description="",
        pages=[Page(0, 100, 100, "pt")],
        elements=[
            Element(type="text", page=0, polygon=[[0, 0], [1, 0], [1, 1]], layer="text"),
            Element(type="rut", page=0, polygon=[[0, 0], [1, 0], [1, 1]], layer="hidden"),
            Element(type="rut", page=0, polygon=[[0, 0], [1, 0], [1, 1]], layer="vector"),
            Element(type="face", page=0, polygon=None),
        ],
    )
    assert [e.layer for e in oracle.targets(f)] == ["vector"]


# ---------------------------------------------------------------------------
# notebook
# ---------------------------------------------------------------------------


def test_notebook_reproduces_cell_12(dataset: Manifest) -> None:
    path = Path(dataset.root) / "pdf/cuaderno.pdf"
    assert notebook.has_text(path)
    assert notebook.detect(path) == NOTEBOOK_FINDINGS
    assert notebook.detect_names(path) == [(1, "name", "Ana Maria Rojas Pena")]


def test_notebook_run(dataset: Manifest, tmp_path: Path) -> None:
    report = run("notebook", dataset, tmp_path)
    r = _result(report, "pdf/cuaderno.pdf")
    assert r.error is None and r.output == "pdf/cuaderno.pdf"
    types = sorted(c.type for c in r.redactions)
    assert types == ["email", "name", "phone", "rut", "rut", "url"]
    assert {c.detector for c in r.redactions} == {"regex", "name_list"}
    with pymupdf.open(tmp_path / "files" / r.output) as doc:
        text = "".join(p.get_text() for p in doc)
    assert not re.findall(notebook.PATTERNS["rut"], text)
    assert "15.782.334-9" not in text and "9.876.543-3" not in text
    assert "Ana Maria Rojas Pena" not in text
    assert "Monto bruto" in text  # neutral text is kept
    verif = report.details["notebook_verification"]["pdf/cuaderno.pdf"]
    assert verif == {"detected": 6, "leaks_per_notebook": 0, "final_sweep": {}}

    # Text PDF of the dataset: the RUT with dots and the listed name are redacted.
    f = _by_id(dataset, "lb_texto")
    r = _result(report, f.path)
    assert r.error is None
    with pymupdf.open(tmp_path / "files" / f.path) as doc:
        text = normalize("rut", doc[0].get_text())
        assert doc.metadata["author"]  # the notebook does not clean metadata
    rut = next(e for e in f.elements if e.type == "rut")
    assert normalize("rut", rut.value) not in text

    assert _result(report, "img/exif6.jpg").error == "unsupported"
    assert _result(report, "img/multi.tiff").error == "unsupported"
    assert _result(report, "pdf/escaneo.pdf").error == "no_text"
    assert _result(report, "pdf/clave.pdf").error.startswith("exception:")
    for r in report.results:
        if r.error:
            assert r.output is None and not (tmp_path / "files" / r.input).exists()


def test_cli_writes_report(dataset: Manifest, tmp_path: Path) -> None:
    manifest = Path(dataset.root) / "manifest.json"
    for system in ("identity", "notebook"):
        output = tmp_path / system
        assert main([system, "--manifest", str(manifest), "--output", str(output)]) == 0
        report = RedactionReport.load(output / "report.json")
        assert report.system == system
        assert [r.input for r in report.results] == [f.path for f in dataset.files]
        assert all(r.time_s is not None for r in report.results)
        # A second run over the same folder replaces the previous one.
        assert main([system, "--manifest", str(manifest), "--output", str(output)]) == 0
