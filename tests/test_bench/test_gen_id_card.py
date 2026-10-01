"""Invariants of the fictitious id card generator: valid files and ground truth that matches."""

from __future__ import annotations

import time
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image

from test_bench.canvas import bounding_box
from test_bench.context import Context
from test_bench.faces import FaceProvider
from test_bench.fake_data import FakeData, check_digit
from test_bench.generators import id_card
from test_bench.schema import Element, FileEntry, Manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Context, list[FileEntry], float]:
    root = tmp_path_factory.mktemp("id_card") / "generated"
    ctx = Context(
        root=root,
        seed=33,
        fake=FakeData(33),
        faces=FaceProvider(REPO_ROOT / "test_data" / "cache" / "faces", 33, allow_download=False),
    )
    start = time.perf_counter()
    files = id_card.generate(ctx)
    return ctx, files, time.perf_counter() - start


def _mask(polygon, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(polygon, np.float64) - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def _gray_page(ctx: Context, a: FileEntry) -> tuple[np.ndarray, float]:
    """Page in grayscale and scale (px per manifest unit)."""
    path = ctx.root / a.path
    if a.format == "pdf":
        doc = pymupdf.open(path)
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
        return np.asarray(img.convert("L")), 2.0
    return np.asarray(Image.open(path).convert("L")), 1.0


def test_time_and_files(generated):
    ctx, files, seconds = generated
    # Timing depends on the machine load: warn, do not fail (timings are reported separately).
    if seconds > 90:
        warnings.warn(f"slow generation: {seconds:.1f} s", stacklevel=1)
    expected = {
        "id_card/cedula_frente_plana.png",
        "id_card/cedula_dorso_plana.png",
        "id_card/cedula_frente_foto_perspectiva.jpg",
        "id_card/cedula_frente_foto_girada.jpg",
        "id_card/cedula_dorso_foto.jpg",
        "id_card/cedula_ambos_lados.pdf",
        "id_card/cedula_con_reflejo.jpg",
    }
    assert {a.path for a in files} == expected
    assert len({a.id for a in files}) == len(files)
    for a in files:
        assert a.id.startswith("ced_") and a.category == "id_card"
        path = ctx.root / a.path
        assert path.exists()
        if a.format == "pdf":
            doc = pymupdf.open(path)
            assert len(doc) == len(a.pages) == 1
            page = doc[0]
            assert (page.rect.width, page.rect.height) == pytest.approx((a.pages[0].width, a.pages[0].height))
            assert page.rotation == a.pages[0].rotation
            assert len(page.get_images()) == 2
            doc.close()
        else:
            img = Image.open(path)
            assert img.size == (a.pages[0].width, a.pages[0].height)
            assert not img.getexif()
    flat = [a for a in files if "plana" in a.path]
    for a in flat:
        assert (a.pages[0].width, a.pages[0].height) == (1027, 648)


def test_manifest_valid(generated):
    ctx, files, _ = generated
    man = Manifest(root=str(ctx.root), seed=33)
    for a in files:
        man.add(a)
    assert man.validate() == []


def test_polygons_inside_and_with_content(generated):
    ctx, files, _ = generated
    for a in files:
        gray, scale = _gray_page(ctx, a)
        page = a.pages[0]
        for e in a.elements:
            assert e.polygon is not None and len(e.polygon) == 4
            if e.layer == "hidden":
                continue
            x0, y0, x1, y1 = bounding_box(e.polygon)
            assert x0 >= -1 and y0 >= -1 and x1 <= page.width + 1 and y1 <= page.height + 1, (a.id, e)
            m = _mask([[x * scale, y * scale] for x, y in e.polygon], gray.shape)
            assert m.sum() > 0
            assert gray[m].std() > 2, (a.id, e.type, e.value)
            if e.core is not None:
                nx0, ny0, nx1, ny1 = bounding_box(e.core)
                assert x0 - 1 <= nx0 and y0 - 1 <= ny0 and nx1 <= x1 + 1 and ny1 <= y1 + 1


def test_flat_text_has_dark_ink(generated):
    ctx, files, _ = generated
    for a in files:
        if "plana" not in a.path:
            continue
        gray = np.asarray(Image.open(ctx.root / a.path).convert("L"))
        for e in a.elements:
            if e.type in ("text", "rut", "name"):
                m = _mask(e.polygon, gray.shape)
                assert gray[m].min() < 130, (a.id, e.value)


def test_text_layer_in_pdf(generated):
    ctx, files, _ = generated
    (a,) = [a for a in files if a.format == "pdf"]
    doc = pymupdf.open(ctx.root / a.path)
    page = doc[0]
    in_text = [e for e in a.elements if e.layer == "text"]
    assert any(e.type == "name" for e in in_text)
    for e in in_text:
        hits = page.search_for(e.value)
        assert hits, e.value
        x0, y0, x1, y1 = bounding_box(e.polygon)
        assert any(abs(h.x0 - x0) < 1 and abs(h.y0 - y0) < 1 and abs(h.x1 - x1) < 1 for h in hits)
    raster = [e for e in a.elements if e.layer == "raster"]
    assert Counter(e.type for e in raster)["face"] == 2
    doc.close()


def test_counts_by_type(generated):
    _, files, _ = generated
    total = Counter(e.type for a in files for e in a.elements)
    # 5 fronts (flat, perspective, rotated, pdf, glare) and 3 backs (flat, photo, pdf)
    assert total["face"] == 10
    assert total["signature"] == 5
    assert total["qr"] == 3
    assert total["rut"] == 8
    assert total["name"] == 5 * 3 + 3 + 1
    for a in files:
        c = Counter(e.type for e in a.elements)
        assert c["text"] >= 8, a.id
        ghosts = [e for e in a.elements if e.type == "face" and e.tags.get("ghost")]
        assert all(e.level == "stress" for e in ghosts)
        assert all(e.core is not None for e in a.elements if e.type == "face")
        assert all(e.level == "out_of_scope" for e in a.elements if e.type == "signature")
        assert all(e.level == "stress" for e in a.elements if e.type == "qr")
        for e in a.elements:
            if e.type == "name" and not e.tags["in_list"]:
                assert e.level == "out_of_scope"
            if e.tags.get("format") == "mrz":
                assert e.level in ("stress", "out_of_scope")
    (glare,) = [a for a in files if a.id == "ced_con_reflejo"]
    assert all(e.level != "base" for e in glare.elements if e.type != "text")
    base = [e for a in files for e in a.elements if e.type == "rut" and e.level == "base"]
    assert len(base) == 4  # flat, perspective, rotated and pdf


def test_ink_does_not_escape_polygons(generated):
    """Independent geometry check (also with perspective, rotation and blur):
    around each dark text on a light background no ink is left outside every polygon."""
    ctx, files, _ = generated
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    for a in files:
        gray, scale = _gray_page(ctx, a)
        dark = gray < 105
        union = np.zeros(gray.shape, bool)
        for e in a.elements:
            union |= _mask([[x * scale, y * scale] for x, y in e.polygon], gray.shape)
        union = cv2.dilate(union.astype(np.uint8), kernel).astype(bool)
        for e in a.elements:
            if e.type in ("face", "qr"):
                continue
            pol = [[x * scale, y * scale] for x, y in e.polygon]
            # local window around the polygon (faster than working with the whole image)
            bx0, by0, bx1, by1 = bounding_box(pol)
            x0, y0 = max(0, int(bx0) - 10), max(0, int(by0) - 10)
            x1, y1 = min(gray.shape[1], int(bx1) + 11), min(gray.shape[0], int(by1) + 11)
            sub = (slice(y0, y1), slice(x0, x1))
            m = _mask([[x - x0, y - y0] for x, y in pol], (y1 - y0, x1 - x0))
            ring = cv2.dilate(m.astype(np.uint8), ring_k).astype(bool) & ~union[sub]
            if not ring.any() or np.median(gray[sub][ring]) < 150:
                continue  # light text on a dark background (red stripe)
            inside = int((dark[sub] & m).sum())
            outside = int((dark[sub] & ring).sum())
            assert outside <= max(8, 0.03 * inside), (a.id, e.type, e.value, inside, outside)


def test_pdf_deterministic(generated, tmp_path):
    ctx, _, _ = generated
    f = FakeData(5).derive("prueba")
    rng = np.random.default_rng(5)
    data = id_card._card_data(f, rng, set())
    front = id_card.draw_front(data, ctx.faces, rng)
    back = id_card.draw_back(data, rng)
    ctx2 = Context(root=tmp_path, seed=5, fake=f, faces=ctx.faces)
    id_card._pdf_both_sides(ctx2, "a.pdf", front, back, data)
    id_card._pdf_both_sides(ctx2, "b.pdf", front, back, data)
    assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "b.pdf").read_bytes()


def test_widen_for_motion():
    """The polygon grows by half the kernel length (plus 0.5 px) in the drag direction."""
    e = Element(type="rut", page=0, polygon=[[0, 0], [10, 0], [10, 10], [0, 10]], value="x")
    id_card._widen_for_motion([e], 11, 0)
    assert bounding_box(e.polygon) == pytest.approx((-5.5, -0.5, 15.5, 10.5))
    e = Element(type="rut", page=0, polygon=[[0, 0], [10, 0], [10, 10], [0, 10]], value="x")
    id_card._widen_for_motion([e], 11, 90)
    assert bounding_box(e.polygon) == pytest.approx((-0.5, -5.5, 10.5, 15.5))
    r = Element(type="face", page=0, polygon=[[0, 0], [10, 0], [10, 10], [0, 10]])
    id_card._widen_for_motion([r], 11, 0)
    assert r.polygon == [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_glare_covers_motion_trail(generated):
    """In the photo with motion blur, faint ink (not only dark ink) stays inside."""
    ctx, files, _ = generated
    (a,) = [a for a in files if a.id == "ced_con_reflejo"]
    gray = np.asarray(Image.open(ctx.root / a.path).convert("L")).astype(np.float64)
    (run,) = [e for e in a.elements if e.type == "rut"]
    m = _mask(run.polygon, gray.shape)
    ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m
    background = np.median(gray[ring])
    assert (gray[ring] < background - 40).sum() <= 0.01 * ring.sum()


def test_qr_decodes_inside_polygon(generated):
    ctx, files, _ = generated
    (a,) = [a for a in files if a.id == "ced_dorso_plana"]
    (qr,) = [e for e in a.elements if e.type == "qr"]
    img = cv2.imread(str(ctx.root / a.path))
    text, points, _ = cv2.QRCodeDetector().detectAndDecode(img)
    assert text == qr.value
    assert "RUN=" in text and "type=CEDULA" in text
    x0, y0, x1, y1 = bounding_box(qr.polygon)
    for x, y in points.reshape(-1, 2):
        assert x0 - 3 <= x <= x1 + 3 and y0 - 3 <= y <= y1 + 3


def test_mrz(generated):
    _, files, _ = generated
    (a,) = [a for a in files if a.id == "ced_dorso_plana"]
    mrz = [e for e in a.elements if e.tags.get("field") == "mrz" or e.tags.get("format") == "mrz"]
    run = [e for e in mrz if e.type == "rut"]
    assert len(run) == 1
    body, dv = run[0].value.split("<")
    assert (check_digit(int(body)) == dv) == run[0].tags["dv_valid"]
    lines = {}
    for e in mrz:
        lines.setdefault(e.tags["line"], []).append(e)
    assert sorted(lines) == [1, 2, 3]
    line1 = "".join(e.value for e in sorted(lines[1], key=lambda e: e.polygon[0][0]))
    assert len(line1) == 30 and run[0].value in line1
    assert all(len(e.value) == 30 for e in lines[2] + lines[3])
    assert id_card._mrz_check_digit("520727") == "3"  # ICAO 9303 example
    # TD1 composite check digit: l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
    (l2,) = [e.value for e in lines[2]]
    assert l2[29] == id_card._mrz_check_digit(line1[5:] + l2[0:7] + l2[8:15] + l2[18:29])
