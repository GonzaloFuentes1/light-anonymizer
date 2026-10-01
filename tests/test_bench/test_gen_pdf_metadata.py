"""The pdf_metadata generator: the ground truth matches the PDF and every hidden structure exists."""

import re
import time
import warnings
import zlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pypdfium2
import pytest

from test_bench.canvas import bounding_box
from test_bench.context import Context
from test_bench.faces import FaceProvider
from test_bench.fake_data import FakeData
from test_bench.generators import pdf_metadata
from test_bench.schema import FileEntry, Manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCALE = 2.0


def _context(root: Path) -> Context:
    return Context(
        root=root,
        seed=33,
        fake=FakeData(33),
        faces=FaceProvider(REPO_ROOT / "test_data" / "cache" / "faces", 33, allow_download=False),
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("pdfm") / "generated"
    ctx = _context(root)
    start = time.perf_counter()
    files = pdf_metadata.generate(ctx)
    duration = time.perf_counter() - start
    man = Manifest(root=str(root), seed=33)
    for a in files:
        man.add(a)
    return root, man, {a.id: a for a in files}, duration


def _path(root: Path, a: FileEntry) -> Path:
    return root / a.path


def _decompressed_streams(data: bytes) -> list[bytes]:
    """All the streams of the file (of every revision), decompressed when possible."""
    output = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        raw = m.group(1)
        try:
            output.append(zlib.decompress(raw))
        except zlib.error:
            output.append(raw)
    return output


def _box(polygon) -> pymupdf.Rect:
    return pymupdf.Rect(*bounding_box(polygon))


def _mask(polygon, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(polygon, np.float64) * SCALE - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def test_files_ids_and_pages(generated):
    root, man, files, _ = generated
    assert set(files) == {
        "pdfm_metadatos_completos",
        "pdfm_revision_incremental",
        "pdfm_redaccion_falsa",
        "pdfm_solo_permisos",
    }
    assert man.validate() == []
    for a in files.values():
        assert a.category == "pdf_metadata" and a.format == "pdf" and a.expected == "process"
        assert a.path.startswith("pdf_metadata/") and a.sha256
        with pymupdf.open(_path(root, a)) as d:
            assert not d.needs_pass
            assert d.page_count == len(a.pages) == 1
            p = d[0]
            assert (a.pages[0].width, a.pages[0].height, a.pages[0].rotation) == (
                p.cropbox.width,
                p.cropbox.height,
                p.rotation,
            )


def test_polygons_inside_the_page(generated):
    _, _, files, _ = generated
    for a in files.values():
        page = a.pages[0]
        for e in a.elements:
            assert e.polygon is not None and len(e.polygon) == 4
            if e.layer == "hidden":
                continue
            x0, y0, x1, y1 = bounding_box(e.polygon)
            assert -0.5 <= x0 < x1 <= page.width + 0.5, (a.id, e.value)
            assert -0.5 <= y0 < y1 <= page.height + 0.5, (a.id, e.value)


def test_text_layer_values_are_found_in_place(generated):
    root, _, files, _ = generated
    for a in files.values():
        with pymupdf.open(_path(root, a)) as d:
            p = d[0]
            for e in a.elements:
                if e.layer != "text":
                    continue
                box = _box(e.polygon)
                found = [q.rect for q in p.search_for(e.value, quads=True)]
                assert any(abs(r.x0 - box.x0) < 0.5 and abs(r.x1 - box.x1) < 0.5 for r in found), (
                    a.id,
                    e.value,
                )
                assert any(abs(r.y0 - box.y0) < 0.5 and abs(r.y1 - box.y1) < 0.5 for r in found)


def test_visible_polygons_have_ink(generated):
    """Normal text: non-uniform pixels. Text under a black cover: the polygon looks black."""
    root, _, files, _ = generated
    for a in files.values():
        with pymupdf.open(_path(root, a)) as d:
            pix = d[0].get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE), alpha=False)
        gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).mean(axis=2)
        for e in a.elements:
            if e.layer == "hidden":
                continue
            values = gray[_mask(e.polygon, gray.shape)]
            assert values.size > 20, (a.id, e.value)
            if e.tags.get("covered_by") in ("drawn_rectangle", "square_annotation"):
                assert values.mean() < 40, (a.id, e.value)
            else:
                assert values.min() < 110 and values.std() > 20, (a.id, e.value)


def test_counts_by_type(generated):
    _, _, files, _ = generated
    count = {i: Counter(e.type for e in a.elements) for i, a in files.items()}
    full = count["pdfm_metadatos_completos"]
    assert full["name"] == 2 and full["rut"] == 2 and full["email"] == 3
    assert full["phone"] == 1 and full["address"] == 1 and full["text"] >= 20
    assert count["pdfm_revision_incremental"]["rut"] == 1 and count["pdfm_revision_incremental"]["name"] == 2
    fake = count["pdfm_redaccion_falsa"]
    assert (fake["name"], fake["rut"], fake["email"], fake["phone"]) == (2, 1, 1, 1)
    permissions = count["pdfm_solo_permisos"]
    assert (permissions["name"], permissions["rut"], permissions["address"], permissions["phone"]) == (2, 2, 2, 2)
    everything = [e for a in files.values() for e in a.elements]
    assert any(e.level == "out_of_scope" and e.type == "name" for e in everything)
    assert all(e.layer == "text" for e in everything if e.layer != "hidden")
    hidden = Counter(e.type for e in everything if e.layer == "hidden")
    assert hidden == {"rut": 1, "text": 1}  # the RUT of the layer turned off and its label
    assert sum(e.type == "text" and e.tags.get("structure") == "freetext_annotation" for e in everything) == 1
    decoys = Counter(e.tags.get("decoy") for e in everything if e.type == "text")
    assert decoys["amount"] == 3 and decoys["date"] == 4


def test_metadata_canaries_are_unique(generated):
    _, _, files, _ = generated
    values = [m.value for a in files.values() for m in a.sensitive_metadata]
    assert all(values) and len(values) == len(set(values)) == 17


def test_full_metadata_structures(generated):
    root, _, files, _ = generated
    a = files["pdfm_metadatos_completos"]
    path = _path(root, a)
    data = path.read_bytes()
    streams = _decompressed_streams(data)
    by_location: dict[str, list] = {}
    for m in a.sensitive_metadata:
        by_location.setdefault(m.location, []).append(m)
    assert set(by_location) == {
        "pdf.info.author",
        "pdf.info.title",
        "pdf.info.subject",
        "pdf.info.keywords",
        "pdf.info.creator",
        "pdf.xmp",
        "pdf.annotation",
        "pdf.attachment",
        "pdf.ocg",
        "pdf.form",
        "pdf.bookmark",
        "pdf.javascript",
    }
    with pymupdf.open(path) as d:
        info = d.metadata
        for key in ("author", "title", "subject", "keywords", "creator"):
            (m,) = by_location[f"pdf.info.{key}"]
            assert m.value in info[key]
        assert info["creator"].startswith("Microsoft Word - informe_") and info["creator"].endswith(".docx")

        xmp = d.get_xml_metadata()
        assert "<dc:creator>" in xmp and "gore:rutFuncionario" in xmp
        assert all(m.value in xmp for m in by_location["pdf.xmp"])

        p = d[0]
        annotations = {x.type[1]: x for x in p.annots()}
        assert {"Text", "FreeText", "Highlight", "FileAttachment"} <= set(annotations)
        contents = {k: x.info["content"] for k, x in annotations.items()}
        for m in by_location["pdf.annotation"]:
            assert m.value in contents[m.tags["annotation"]]
        assert annotations["Highlight"].popup_xref > 0

        assert d.embfile_count() == 1 and d.embfile_names() == ["datos_contacto.txt"]
        embedded = d.embfile_get(0).decode()
        annotation_file = annotations["FileAttachment"].get_file().decode()
        for m in by_location["pdf.attachment"]:
            assert m.value in (embedded if m.tags["form"] == "embfile" else annotation_file)

        ocgs = {v["name"]: v for v in d.get_ocgs().values()}
        assert "Notas internas" in ocgs and ocgs["Notas internas"]["on"] is False
        (m_ocg,) = by_location["pdf.ocg"]
        (hidden,) = [e for e in a.elements if e.layer == "hidden" and e.type != "text"]
        assert hidden.value == m_ocg.value and hidden.type == "rut"
        assert m_ocg.value not in p.get_text()  # the layer turned off is neither seen nor extracted by PyMuPDF
        assert any(m_ocg.value.encode() in s for s in streams)  # but it is in the stream, as a literal

        # With the layer turned on, search_for finds each hidden text right on its polygon.
        d.xref_set_key(d.pdf_catalog(), "OCProperties/D/OFF", "[]")
        with pymupdf.open("pdf", d.tobytes()) as turned_on:
            p_on = turned_on[0]
            for e in (e for e in a.elements if e.layer == "hidden"):
                (r,) = p_on.search_for(e.value)
                box = _box(e.polygon)
                assert max(abs(r.x0 - box.x0), abs(r.x1 - box.x1), abs(r.y0 - box.y0), abs(r.y1 - box.y1)) < 0.5

        widgets = list(p.widgets())
        (m_form,) = by_location["pdf.form"]
        assert [(w.field_name, w.field_value) for w in widgets] == [("correo_notificacion", m_form.value)]

        (m_toc,) = by_location["pdf.bookmark"]
        assert any(m_toc.value in title for _, title, _ in d.get_toc())

        (m_js,) = by_location["pdf.javascript"]
        names = d.xref_get_key(d.pdf_catalog(), "Names/JavaScript")
        assert names[0] == "dict" and "(contacto)" in names[1]
        assert f'"{m_js.value}"'.encode() in data

    # pdfium does extract the text of the layer turned off: the canary is detectable by text extraction.
    pdf = pypdfium2.PdfDocument(str(path))
    try:
        assert m_ocg.value in pdf[0].get_textpage().get_text_range()
    finally:
        pdf.close()


def test_incremental_revision(generated):
    root, _, files, _ = generated
    a = files["pdfm_revision_incremental"]
    data = _path(root, a).read_bytes()
    (m,) = a.sensitive_metadata
    assert m.location == "pdf.previous_revision"
    assert data.count(b"%%EOF") == 2 and data.count(b"startxref") == 2
    with pymupdf.open(_path(root, a)) as d:
        current = d[0].get_text()
    assert m.value not in current
    assert all(e.value in current for e in a.elements)
    # Revision 1 (up to the first %%EOF) is a valid PDF and still contains the deleted line.
    rev1_end = data.index(b"%%EOF") + len(b"%%EOF")
    with pymupdf.open(stream=data[:rev1_end], filetype="pdf") as old:
        assert m.tags["line"] in old[0].get_text()
    # And it is recovered from the bytes (decompressed streams) as a literal string.
    assert any(m.value.encode() in s for s in _decompressed_streams(data))


def test_fake_redaction(generated):
    root, _, files, _ = generated
    a = files["pdfm_redaccion_falsa"]
    covered = {e.tags["covered_by"]: e for e in a.elements if "covered_by" in e.tags}
    assert set(Counter(e.tags["covered_by"] for e in a.elements if "covered_by" in e.tags).items()) == {
        ("drawn_rectangle", 2),
        ("unapplied_redact", 1),
        ("square_annotation", 1),
    }
    assert all(e.layer == "text" and e.level == "base" for e in a.elements if "covered_by" in e.tags)
    with pymupdf.open(_path(root, a)) as d:
        p = d[0]
        text = p.get_text()
        for e in a.elements:
            assert e.value in text  # everything is still extractable
        blacks = [dr["rect"] for dr in p.get_drawings() if dr.get("fill") == (0.0, 0.0, 0.0)]
        for e in a.elements:
            if e.tags.get("covered_by") == "drawn_rectangle":
                assert any(r.contains(_box(e.polygon)) for r in blacks), e.value
        annotations = {x.type[1]: x for x in p.annots()}
        assert annotations["Redact"].rect.contains(_box(covered["unapplied_redact"].polygon))
        square = annotations["Square"]
        assert square.colors["fill"] == [0.0, 0.0, 0.0]
        assert square.rect.contains(_box(covered["square_annotation"].polygon))


def test_permissions_only(generated):
    root, _, files, _ = generated
    a = files["pdfm_solo_permisos"]
    assert _path(root, a).read_bytes().count(b"/Encrypt") >= 1
    with pymupdf.open(_path(root, a)) as d:
        assert not d.needs_pass
        assert "AES" in d.metadata["encryption"] and "256" in d.metadata["encryption"]
        assert not d.permissions & pymupdf.PDF_PERM_COPY
        assert not d.permissions & pymupdf.PDF_PERM_PRINT
        text = d[0].get_text()
    assert all(e.value in text for e in a.elements)


def test_time_and_determinism(generated, tmp_path):
    _, _, files, duration = generated
    # Timing depends on the machine load: it warns, it does not fail (times are reported separately).
    if duration > 90:
        warnings.warn(f"generación lenta: {duration:.1f} s", stacklevel=1)
    other = pdf_metadata.generate(_context(tmp_path / "other"))

    def without_hash(a: FileEntry) -> dict:
        data = asdict(a)
        data.pop("sha256")
        for e in data["elements"]:
            e.pop("id")
        for m in data["sensitive_metadata"]:
            m.pop("id")
        return data

    assert [without_hash(a) for a in other] == [without_hash(a) for a in files.values()]

    # Identical bytes (no random /ID nor machine dates), except for the AES-encrypted file, whose
    # salt and initialization vector are random: there the text is compared and the sha256 changes.
    root, _, _, _ = generated
    for a in other:
        old, new = (root / a.path).read_bytes(), (tmp_path / "other" / a.path).read_bytes()
        if a.tags.get("encryption"):
            with pymupdf.open(stream=old) as d1, pymupdf.open(stream=new) as d2:
                assert [p.get_text() for p in d1] == [p.get_text() for p in d2]
        else:
            assert old == new, a.id
            assert b"D:2026" not in old or all(
                f in (pdf_metadata.CREATION_DATE[:16].encode(), pdf_metadata.MODIFICATION_DATE[:16].encode())
                for f in re.findall(rb"D:\d{14}", old)
            ), a.id
