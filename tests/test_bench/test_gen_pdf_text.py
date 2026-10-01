"""Invariants of the ``pdf_text`` generator: valid files and a ground truth that matches the PDF."""

import re
import time
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from test_bench.canvas import bounding_box
from test_bench.context import Context
from test_bench.faces import FaceProvider
from test_bench.fake_data import EMAIL_FORMATS, PHONE_FORMATS, RUT_FORMATS, FakeData, strip_accents
from test_bench.generators import pdf_text
from test_bench.schema import FileEntry, Manifest
from test_bench.visualize import PDF_SCALE, to_visual

REPO_ROOT = Path(__file__).resolve().parents[2]


def _context(root: Path) -> Context:
    return Context(
        root=root,
        seed=33,
        fake=FakeData(33),
        faces=FaceProvider(REPO_ROOT / "test_data" / "cache" / "faces", 33, allow_download=False),
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    ctx = _context(tmp_path_factory.mktemp("pdf_text") / "generated")
    start = time.perf_counter()
    files = pdf_text.generate(ctx)
    return ctx, files, time.perf_counter() - start


@pytest.fixture(scope="module")
def files(generated) -> list[FileEntry]:
    return generated[1]


@pytest.fixture(scope="module")
def root(generated) -> Path:
    return generated[0].root


def _by_id(files: list[FileEntry], id: str) -> FileEntry:
    return next(a for a in files if a.id == id)


@lru_cache(maxsize=64)
def _render(path: str, page: int) -> np.ndarray:
    doc = pymupdf.open(path)
    pix = doc[page].get_pixmap(matrix=pymupdf.Matrix(PDF_SCALE, PDF_SCALE), alpha=False)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    doc.close()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _pixels(file_entry: FileEntry, root: Path, e) -> np.ndarray:
    gray = _render(str(root / file_entry.path), e.page)
    pts = np.asarray(to_visual(file_entry, e.page, e.polygon), np.float64) - 0.5
    x0, y0 = np.floor(pts.min(axis=0)).astype(int).clip(0)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 2
    crop = gray[y0:y1, x0:x1]
    mask = np.zeros(crop.shape, np.uint8)
    cv2.fillPoly(mask, [np.round((pts - [x0, y0]) * 16).astype(np.int32)], 1, shift=4)
    return crop[mask.astype(bool)]


def _inside(e, page, slack: float = 0.5) -> bool:
    x0, y0, x1, y1 = bounding_box(e.polygon)
    return x0 >= -slack and y0 >= -slack and x1 <= page.width + slack and y1 <= page.height + slack


# ---------------------------------------------------------------------------


def test_generation_time(generated):
    # Timing depends on the machine load: it warns, it does not fail.
    if generated[2] > 90:
        warnings.warn(f"generación lenta: {generated[2]:.1f} s", stacklevel=1)


def test_manifest_valid(generated, files):
    ctx = generated[0]
    ids = [a.id for a in files]
    assert len(ids) == len(set(ids)) == 12
    assert all(i.startswith("pdft_") for i in ids)
    assert {a.category for a in files} == {"pdf_text"}
    man = Manifest(root=str(ctx.root), seed=33)
    for a in files:
        man.add(a)
    assert man.validate() == []
    assert all(a.sha256 for a in man.files)


def test_pages_match_the_pdf(files, root):
    for a in files:
        doc = pymupdf.open(root / a.path)
        assert doc.page_count == len(a.pages), a.id
        for page_entry, page in zip(a.pages, doc, strict=True):
            assert page_entry.unit == "pt"
            assert page_entry.width == pytest.approx(page.cropbox.width)
            assert page_entry.height == pytest.approx(page.cropbox.height)
            assert page_entry.rotation == page.rotation
        for e in a.elements:
            assert 0 <= e.page < doc.page_count
            assert e.polygon is not None and len(e.polygon) == 4, e
        doc.close()


def test_polygons_inside_the_page(files):
    for a in files:
        for e in a.elements:
            page = a.pages[e.page]
            if e.layer == "hidden" and e.tags.get("hidden") == "outside_cropbox":
                assert not _inside(e, page), (a.id, e.value)
            else:
                assert _inside(e, page), (a.id, e.value, e.polygon)


def test_text_layer_is_found_with_search_for(files, root):
    checked = 0
    for a in files:
        doc = pymupdf.open(root / a.path)
        for e in a.elements:
            if e.layer not in ("text", "hidden"):
                continue
            page = doc[e.page]
            if "spacing_pt" in e.tags:
                # separate letters: the text inside the polygon, without spaces, is the value
                x0, y0, x1, y1 = bounding_box(e.polygon)
                extracted = page.get_text("text", clip=pymupdf.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1))
                assert "".join(extracted.split()) == e.value
                continue
            tp = pdf_text.full_textpage(page)
            quads = page.search_for(e.value, quads=True, textpage=tp)
            distances = [pdf_text._distance(pdf_text._quad_to_polygon(q), e.polygon) for q in quads]
            assert distances and min(distances) <= 1.0, (a.id, e.value, e.tags.get("gt"))
            checked += 1
        doc.close()
    assert checked > 800


def test_visible_elements_have_ink(files, root):
    for a in files:
        for e in a.elements:
            if e.layer == "hidden":
                continue
            px = _pixels(a, root, e)
            assert px.size > 0, (a.id, e.value)
            if e.type == "face":
                assert px.std() > 10, (a.id, e.tags)
            else:
                assert px.min() < 150, (a.id, e.type, e.value)


def test_polygons_fit_the_ink(files, root):
    """The ink of each datum stays inside its polygon: no stroke touching it sticks out more than 1.5 pt.

    The other polygons of the page (neighboring text), the strokes that do not touch the
    polygon (adjacent punctuation) and the long straight lines (table borders, signature lines)
    are discounted. It also covers the page with /Rotate, the rotated text and the embedded images.
    """
    checked = 0
    for a in files:
        for page in range(len(a.pages)):
            gray = _render(str(root / a.path), page)
            elements = [e for e in a.elements if e.page == page and e.layer != "hidden"]
            visuals = [np.asarray(to_visual(a, page, e.polygon), np.float64) - 0.5 for e in elements]
            for e, pts in zip(elements, visuals, strict=True):
                if e.type in ("text", "face", "qr"):
                    continue
                # local window with a wide margin (to recognize the long lines) and a 3 px = 1.5 pt band
                x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 32
                x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 32
                x0, y0 = max(x0, 0), max(y0, 0)
                window = gray[y0:y1, x0:x1]

                def mask(q: np.ndarray, shape=window.shape, origin=(x0, y0)) -> np.ndarray:
                    m = np.zeros(shape, np.uint8)
                    cv2.fillPoly(m, [np.round((q - origin) * 16).astype(np.int32)], 1, shift=4)
                    return m.astype(bool)

                own = mask(pts)
                band = cv2.dilate(own.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~own
                for other in visuals:
                    if other is not pts:
                        band &= ~mask(other)
                background = int(np.median(window[band | own]))
                ink = (np.abs(window.astype(int) - background) > 80).astype(np.uint8)
                lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 40), np.uint8)) | cv2.morphologyEx(
                    ink, cv2.MORPH_OPEN, np.ones((40, 1), np.uint8)
                )
                ink = ink.astype(bool) & ~lines.astype(bool)
                # only the strokes that touch the polygon and stick out of it count (not the nearby punctuation)
                _, labels = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
                touching = np.isin(labels, np.unique(labels[ink & own]))
                inside, outside = int((ink & own).sum()), int((ink & touching & band).sum())
                assert inside > 0 and outside <= 0.02 * inside, (a.id, e.type, e.value, inside, outside)
                checked += 1
    assert checked > 250


def test_hidden_is_not_visible(files, root):
    a = _by_id(files, "pdft_estres_capa_texto")
    modes = Counter()
    for e in a.elements:
        if e.layer != "hidden":
            continue
        modes[e.tags["hidden"]] += 1
        if e.tags["hidden"] == "outside_cropbox":
            continue
        px = _pixels(a, root, e)
        assert int(px.max()) - int(px.min()) < 12, (e.value, e.tags["hidden"])
    assert set(modes) == {"white_on_white", "render_mode_3", "under_image", "outside_cropbox"}
    outside = [e for e in a.elements if e.tags.get("hidden") == "outside_cropbox" and e.type != "text"]
    assert {e.type for e in outside} == {"name", "rut", "email"}
    assert all(e.level in ("stress", "out_of_scope") for e in outside)


def test_vectorized_text_has_no_text(files, root):
    a = _by_id(files, "pdft_texto_vectorizado")
    doc = pymupdf.open(root / a.path)
    assert doc[0].get_text().strip() == ""
    assert len(doc[0].get_drawings()) > 100
    assert {e.layer for e in a.elements} == {"vector"}
    assert {"rut", "email", "phone", "name", "address"} <= {e.type for e in a.elements}
    doc.close()


def test_counts_by_type(files):
    c = Counter((e.type, e.level) for a in files for e in a.elements)
    types = Counter(e.type for a in files for e in a.elements)
    assert types["rut"] >= 55 and types["email"] >= 55 and types["phone"] >= 50
    assert types["url"] >= 12 and types["address"] >= 12 and types["text"] >= 500
    assert types["face"] == 3 and types["qr"] == 1
    assert c[("name", "out_of_scope")] >= 2 and c[("address", "out_of_scope")] >= 1
    assert c[("name", "stress")] >= 4
    decoys = Counter(e.tags.get("decoy") for a in files for e in a.elements if e.type == "text")
    assert decoys["amount"] >= 15 and decoys["date"] >= 30


def test_neutral_text_has_no_names(files):
    """No text recorded as neutral carries a first name or surname of the people of the dataset."""

    def words(text: str) -> set[str]:
        return {strip_accents(w).lower() for w in re.findall(r"\w{3,}", text)}

    tokens = set().union(*(words(e.value) for a in files for e in a.elements if e.type == "name"))
    tokens -= {"los", "las", "del"}
    for a in files:
        for e in a.elements:
            if e.type == "text":
                assert not words(e.value) & tokens, (a.id, e.value)
    greetings = [e for e in _by_id(files, "pdft_correo_impreso").elements
                 if e.type == "name" and e.tags["variant"] == "first_name_only"]  # fmt: skip
    assert len(greetings) == 2 and all(e.level == "stress" for e in greetings)


def test_format_coverage_in_fee_reports(files):
    elements = [e for a in files if a.id.startswith("pdft_honorarios_") for e in a.elements]
    for type_, formats in (("rut", RUT_FORMATS), ("phone", PHONE_FORMATS), ("email", EMAIL_FORMATS)):
        used = Counter(e.tags["format"] for e in elements if e.type == type_)
        for format, level in formats.items():
            if level == "base":
                assert used[format] >= 2, (type_, format)
    ruts = [e for e in elements if e.type == "rut"]
    assert any(e.tags["format"] == "lowercase_k" and e.value.endswith("-k") for e in ruts)
    assert any(e.value.endswith("-K") for e in ruts)
    assert any(e.tags.get("seven_digits") for e in ruts)
    assert sum(not e.tags["dv_valid"] for e in ruts) >= 3
    variants = Counter(e.tags["variant"] for e in elements if e.type == "name")
    assert {"exact", "no_accents", "uppercase", "surnames_names", "partial"} <= set(variants)
    assert all(e.level == "stress" for e in elements if e.type == "name" and e.tags["variant"] == "partial"
               and e.tags["in_list"])  # fmt: skip
    assert {e.tags["kind"] for e in elements if e.type == "phone"} == {"mobile", "santiago", "regional"}


def test_notebook_replica(generated, files, root):
    ctx = generated[0]
    a = _by_id(files, "pdft_cuaderno")
    doc = pymupdf.open(root / a.path)
    lines = [x for x in doc[0].get_text().splitlines() if x.strip()]
    assert lines == [x for x in pdf_text.NOTEBOOK_TEXT.splitlines() if x.strip()]
    doc.close()
    values = {(e.type, e.value) for e in a.elements}
    assert ("rut", "15.782.334-9") in values and ("rut", "9.876.543-3") in values
    assert ("url", "https://www.ejemplo.cl/rendiciones") in values
    (bad,) = [e for e in a.elements if e.value == "15.782.334-9"]
    assert bad.tags["dv_valid"] is False
    name_list = ctx.fake.name_list()
    assert "Ana Maria Rojas Pena" in name_list and "Pasaje Los Alerces 442, depto 31" in name_list


def test_embedded_images(files, root):
    a = _by_id(files, "pdft_mixto_imagenes")
    doc = pymupdf.open(root / a.path)
    shared = {x[0] for x in doc[1].get_images(full=True) if len(doc[1].get_image_rects(x[0])) == 2}
    assert len(shared) == 1
    with_smask = [x for p in doc for x in p.get_images(full=True) if x[1] > 0]
    assert with_smask, "missing the image with transparency (SMask)"
    faces = [e for e in a.elements if e.type == "face"]
    assert len(faces) == 3 and all(e.core for e in faces)
    (qr,) = [e for e in a.elements if e.type == "qr"]
    gray = _render(str(root / a.path), qr.page)
    x0, y0, x1, y1 = (int(v * PDF_SCALE) for v in bounding_box(qr.polygon))
    crop = cv2.copyMakeBorder(gray[y0:y1, x0:x1], 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(crop)
    assert text == qr.value and qr.level == "stress"
    raster = Counter(e.type for e in a.elements if e.layer == "raster")
    assert raster["email"] >= 1 and raster["phone"] >= 3 and raster["rut"] >= 1
    doc.close()


def test_rotated_page_looks_horizontal(files):
    a = _by_id(files, "pdft_pagina_rotada")
    assert [p.rotation for p in a.pages] == [0, 90]
    data = [e for e in a.elements if e.page == 1 and e.type != "text"]
    assert len(data) >= 35
    for e in data:
        pts = np.asarray(to_visual(a, 1, e.polygon))
        width, height = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        assert width > height, e.value


def test_rotated_text_covers_the_angles(files):
    a = _by_id(files, "pdft_texto_girado")
    angles = {e.tags["angle"] for e in a.elements if e.type != "text"}
    assert {0, 90, 180, 270, 30.0} <= angles
    for e in a.elements:
        assert e.tags["gt"] == "search_for", e.value


def test_deterministic_generation(files, tmp_path):
    other = pdf_text.generate(_context(tmp_path / "other"))
    signature = [(a.id, [(e.type, e.value, e.level, e.polygon) for e in a.elements]) for a in files]
    assert signature == [(a.id, [(e.type, e.value, e.level, e.polygon) for e in a.elements]) for a in other]
