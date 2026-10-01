"""Invariants of the scanned PDF generator: the ground truth has to match every page."""

from __future__ import annotations

import io
import math
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
from test_bench.fake_data import FakeData
from test_bench.generators import pdf_scanned
from test_bench.schema import FileEntry, Manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCALE = 2.0  # 144 dpi to check pixels


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Context, list[FileEntry], float]:
    ctx = _context(tmp_path_factory.mktemp("pdf_scanned") / "generated")
    start = time.perf_counter()
    files = pdf_scanned.generate(ctx)
    return ctx, files, time.perf_counter() - start


def _by_id(files: list[FileEntry], id: str) -> FileEntry:
    return next(a for a in files if a.id == "pdfe_" + id)


def _mask(polygon, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(polygon, np.float64) * SCALE - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def _center(polygon) -> tuple[float, float]:
    x0, y0, x1, y1 = bounding_box(polygon)
    return (x0 + x1) / 2, (y0 + y1) / 2


def test_files_exist_and_open(generated):
    ctx, files, dt = generated
    assert len(files) == 8
    assert len({a.id for a in files}) == 8
    # The time depends on the machine load: warn, do not fail (times are reported separately).
    if dt > 90:
        warnings.warn(f"slow generation: {dt:.1f} s", stacklevel=1)
    for a in files:
        assert a.id.startswith("pdfe_") and a.category == "pdf_scanned" and a.format == "pdf"
        path = ctx.root / a.path
        assert path.exists() and a.path.startswith("pdf_scanned/")
        with pymupdf.open(path) as doc:
            assert doc.page_count == len(a.pages)
            for pg, page in zip(a.pages, doc, strict=True):
                assert (pg.width, pg.height, pg.rotation) == (page.rect.width, page.rect.height, page.rotation)
                assert (pg.width, pg.height) == (595.0, 842.0) and pg.unit == "pt"


def test_manifest_valid(generated):
    ctx, files, _ = generated
    man = Manifest(root=str(ctx.root), seed=33)
    for a in files:
        man.add(a)
    assert man.validate() == []


def test_polygons_inside_the_page(generated):
    _, files, _ = generated
    for a in files:
        for e in a.elements:
            assert e.polygon is not None and len(e.polygon) == 4
            if e.layer == "hidden":
                continue
            pg = a.pages[e.page]
            x0, y0, x1, y1 = bounding_box(e.polygon)
            assert x0 >= -0.5 and y0 >= -0.5 and x1 <= pg.width + 0.5 and y1 <= pg.height + 0.5, (a.id, e)
            if e.core:
                nx0, ny0, nx1, ny1 = bounding_box(e.core)
                assert x0 <= nx0 and y0 <= ny0 and nx1 <= x1 and ny1 <= y1


def test_scanned_pages_without_text(generated):
    """Scanned pages have no text layer (except the sandwich's invisible OCR) and are a single image."""
    ctx, files, _ = generated
    for a in files:
        with pymupdf.open(ctx.root / a.path) as doc:
            for i, page in enumerate(doc):
                layers = {e.layer for e in a.elements if e.page == i}
                if "text" in layers:
                    assert not page.get_images()
                    continue
                assert layers <= {"raster", "hidden"}
                info = page.get_image_info()
                assert len(info) == 1 and pymupdf.Rect(info[0]["bbox"]) == page.rect
                if "hidden" in layers:
                    trace = page.get_texttrace()
                    assert trace and {s["type"] for s in trace} == {3}, "the OCR layer must be invisible"
                else:
                    assert page.get_text().strip() == ""


def test_text_and_hidden_layers_found_with_search_for(generated):
    ctx, files, _ = generated
    checked = Counter()
    for a in files:
        with pymupdf.open(ctx.root / a.path) as doc:
            for e in a.elements:
                if e.layer not in ("text", "hidden"):
                    continue
                hits = doc[e.page].search_for(e.value, quads=True)
                assert hits, (a.id, e.value)
                d = min(
                    math.dist(_center([[p.x, p.y] for p in (q.ul, q.ur, q.lr, q.ll)]), _center(e.polygon)) for q in hits
                )
                assert d < 0.01, (a.id, e.value, d)
                checked[e.layer] += 1
    assert checked["text"] >= 20 and checked["hidden"] >= 30


def test_sandwich_records_each_datum_twice(generated):
    _, files, _ = generated
    a = _by_id(files, "sandwich_ocr")
    raster = [e for e in a.elements if e.layer == "raster"]
    hidden = [e for e in a.elements if e.layer == "hidden"]
    assert len(raster) == len(hidden)
    for r, o in zip(raster, hidden, strict=True):
        assert (r.type, r.value, r.level) == (o.type, o.value, o.level)
        assert r.tags["sandwich"] and o.tags["sandwich"]
        # the invisible layer sits on top of the image text (as the scanner's OCR leaves it)
        assert math.dist(_center(r.polygon), _center(o.polygon)) < 4, (r.value, r.polygon, o.polygon)
    assert {e.type for e in hidden} >= {"rut", "email", "phone", "name", "address"}


def test_raster_polygons_have_ink(generated):
    ctx, files, _ = generated
    for a in files:
        with pymupdf.open(ctx.root / a.path) as doc:
            for i, page in enumerate(doc):
                elements = [e for e in a.elements if e.page == i and e.layer == "raster"]
                if not elements:
                    continue
                pix = page.get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE), colorspace=pymupdf.csGRAY)
                gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
                for e in elements:
                    values = gray[_mask(e.polygon, gray.shape)]
                    assert values.size > 0, (a.id, e.value)
                    if e.type == "face":
                        assert values.std() > 15
                        continue
                    background = float(np.percentile(values, 95))
                    contrast = background - float(np.percentile(values, 3))
                    assert contrast > 60, (a.id, e.type, e.value, contrast)
                    # the polygon is not just background: a visible fraction is ink
                    assert (values < background - 50).mean() > 0.04, (a.id, e.type, e.value)


def test_counts_and_levels(generated):
    _, files, _ = generated
    elements = [e for a in files for e in a.elements]
    types = Counter(e.type for e in elements)
    assert types["rut"] >= 20 and types["phone"] >= 20 and types["email"] >= 15
    assert types["name"] >= 18 and types["address"] >= 6 and types["signature"] >= 3
    assert types["face"] == 1 and types["text"] >= 200
    decoys = Counter(e.tags.get("decoy") for e in elements if e.type == "text")
    assert decoys["amount"] >= 8 and decoys["date"] >= 10
    for e in elements:
        if e.type in ("name", "address") and not e.tags["in_list"]:
            assert e.level == "out_of_scope"
        if e.type == "signature":
            assert e.level == "out_of_scope"
        if e.type == "rut":
            assert {"format", "dv_valid"} <= e.tags.keys()
        if e.type == "phone":
            assert {"format", "kind"} <= e.tags.keys()
        if e.type == "email":
            assert "format" in e.tags
    assert any(e.type == "name" and e.level == "out_of_scope" for e in elements)
    assert any(e.type == "rut" and not e.tags["dv_valid"] for e in elements)
    (face,) = [e for e in elements if e.type == "face"]
    assert face.core is not None and face.level == "base"


def test_fax_binarized_and_at_stress(generated):
    ctx, files, _ = generated
    a = _by_id(files, "fax_150dpi")
    assert {e.level for e in a.elements} <= {"stress", "out_of_scope"}
    with pymupdf.open(ctx.root / a.path) as doc:
        (img,) = doc[0].get_images(full=True)
        assert img[4] == 1 and img[2] == int(round(8.27 * 150))  # 1 bit per component, 150 dpi


def test_stamp_signature_and_note(generated):
    _, files, _ = generated
    a = _by_id(files, "timbre_y_firma")
    stamp = [e for e in a.elements if e.tags.get("stamp")]
    assert {e.type for e in stamp} >= {"name", "rut", "text"}
    for e in stamp:
        if e.type != "text":
            assert e.level in ("stress", "out_of_scope")
        assert abs(e.tags["angle"] % 360 - 20.5) < 0.01 or abs(e.tags["angle"] % 360 - 353.5) < 0.01
    (signature,) = [e for e in a.elements if e.type == "signature"]
    assert signature.level == "out_of_scope"
    handwritten = [e for e in a.elements if e.type == "phone" and e.tags.get("handwritten")]
    assert len(handwritten) == 1 and handwritten[0].level == "stress"


def test_inverted_scan_rotations(generated):
    _, files, _ = generated
    a = _by_id(files, "escaneo_invertido")
    angles = {i: {round(e.tags["angle"] % 360, 1) for e in a.elements if e.page == i} for i in (0, 1)}
    assert angles[0] == {180.6}
    assert angles[1] == {89.2}
    x0, y0, x1, y1 = bounding_box(next(e for e in a.elements if e.page == 1 and e.type == "rut").polygon)
    assert (y1 - y0) > 2 * (x1 - x0), "on the rotated sheet the data run vertically"


def _context(root: Path) -> Context:
    return Context(
        root=root,
        seed=33,
        fake=FakeData(33),
        faces=FaceProvider(REPO_ROOT / "test_data" / "cache" / "faces", 33, allow_download=False),
    )


def test_pdf_deterministic(tmp_path):
    """Two runs with the same seed give identical bytes (no random /ID in the trailer)."""
    bytes_per_run = []
    for n in (1, 2):
        ctx = _context(tmp_path / f"run{n}")
        a = pdf_scanned._gen_fax(ctx, ctx.fake_data(pdf_scanned.SEED_NAME), ctx.rng(pdf_scanned.SEED_NAME))
        bytes_per_run.append((ctx.root / a.path).read_bytes())
    assert bytes_per_run[0] == bytes_per_run[1]


def _extent_error(gray: np.ndarray, polygon: np.ndarray) -> float | None:
    """Distance (px) between the quadrilateral edges and the extent of the ink, in the text frame.

    Computed independently of the generator: it measures the real ink of the embedded image,
    along the baseline (within the height) and perpendicular to it (within the length).
    """
    p0, p1, p3 = polygon[0], polygon[1], polygon[3]
    length, height = float(np.linalg.norm(p1 - p0)), float(np.linalg.norm(p3 - p0))
    u, v = (p1 - p0) / length, (p3 - p0) / height
    m = max(4.0, 0.6 * height)
    x0, y0 = np.maximum(np.floor(polygon.min(0) - m), 0).astype(int)
    x1, y1 = np.ceil(polygon.max(0) + m).astype(int)
    sub = gray[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0 : y0 + sub.shape[0], x0 : x0 + sub.shape[1]] + 0.5
    d = np.stack([xx - p0[0], yy - p0[1]], -1)
    cu, cv = d @ u, d @ v
    inside = (cu >= 0) & (cu <= length) & (cv >= 0) & (cv <= height)
    if not inside.any():
        return None
    threshold = (np.percentile(sub, 90) + np.percentile(sub[inside], 2)) / 2
    ink = sub < threshold
    sel_v = ink & (cu >= 0) & (cu <= length) & (cv >= -0.35 * height) & (cv <= 1.35 * height)
    sel_u = ink & (cv >= 0) & (cv <= height) & (cu >= -0.12 * height) & (cu <= length + 0.12 * height)
    if sel_v.sum() < 3 or sel_u.sum() < 3:
        return None
    vs, us = np.sort(cv[sel_v]), np.sort(cu[sel_u])  # one pixel is discarded at each end (noise)
    return float(max(abs(vs[1]), abs(vs[-2] - height), abs(us[1]), abs(us[-2] - length)))


def test_raster_polygons_fit_the_ink(generated):
    """Raster polygons match the ink of the page image (median < 1 pt per file)."""
    ctx, files, _ = generated
    for a in files:
        errors = []
        with pymupdf.open(ctx.root / a.path) as doc:
            for i, page in enumerate(doc):
                elements = [e for e in a.elements if e.page == i and e.layer == "raster" and e.type != "face"]
                if not elements:
                    continue
                data = doc.extract_image(page.get_images()[0][0])["image"]
                gray = np.asarray(Image.open(io.BytesIO(data)).convert("L"), np.float32)
                scale = np.array([gray.shape[1] / page.rect.width, gray.shape[0] / page.rect.height])
                for e in elements:
                    err = _extent_error(gray, np.asarray(e.polygon, np.float64) * scale)
                    assert err is not None, (a.id, e.value)
                    errors.append(err / scale[0])
        assert float(np.median(errors)) < 1.0, (a.id, np.median(errors))
        assert float(np.percentile(errors, 90)) < 3.5, (a.id, np.percentile(errors, 90))
