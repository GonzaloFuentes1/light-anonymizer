"""Invariants of the screenshot generator: sizes, tight ground truth and no overlapping texts."""

from __future__ import annotations

import time
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from test_bench.canvas import bounding_box
from test_bench.context import Context
from test_bench.faces import FaceProvider
from test_bench.fake_data import FakeData
from test_bench.generators import screenshots
from test_bench.schema import FileEntry, Manifest

REPO_ROOT = Path(__file__).resolve().parents[2]

SIZES = {
    "screenshots/correo_escritorio.png": (1920, 1080),
    "screenshots/chat_movil.jpg": (1080, 2340),
    "screenshots/planilla.png": (1600, 900),
    "screenshots/planilla_reducida.png": (960, 540),
    "screenshots/formulario_web.webp": (1366, 768),
    "screenshots/dialogo_pequeno.png": (600, 300),
    "screenshots/correo_escritorio_hidpi.png": (2880, 1620),
}


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Context, dict[str, FileEntry], float]:
    root = tmp_path_factory.mktemp("screenshots") / "generated"
    ctx = Context(
        root=root,
        seed=33,
        fake=FakeData(33),
        faces=FaceProvider(REPO_ROOT / "test_data" / "cache" / "faces", 33, allow_download=False),
    )
    start = time.perf_counter()
    files = screenshots.generate(ctx)
    return ctx, {a.path: a for a in files}, time.perf_counter() - start


def _mask(polygon, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    p = np.asarray(polygon, np.float64) - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def test_files_and_sizes(generated):
    ctx, files, seconds = generated
    # Timing depends on the machine load: warn, do not fail (timings are reported separately).
    if seconds > 90:
        warnings.warn(f"slow generation: {seconds:.1f} s", stacklevel=1)
    assert set(files) == set(SIZES)
    assert len({a.id for a in files.values()}) == len(files)
    for path, a in files.items():
        assert a.id.startswith("pant_") and a.category == "screenshot"
        img = Image.open(ctx.root / path)
        img.load()
        assert img.size == SIZES[path] == (a.pages[0].width, a.pages[0].height)
        assert a.format == path.rsplit(".", 1)[1]
        assert not img.getexif()


def test_manifest_valid(generated):
    ctx, files, _ = generated
    man = Manifest(root=str(ctx.root), seed=33)
    for a in files.values():
        man.add(a)
    assert man.validate() == []


def test_polygons_inside_and_with_ink(generated):
    ctx, files, _ = generated
    for path, a in files.items():
        gray = np.asarray(Image.open(ctx.root / path).convert("L"))
        w, h = SIZES[path]
        for e in a.elements:
            assert e.layer == "raster" and e.polygon is not None
            x0, y0, x1, y1 = bounding_box(e.polygon)
            assert x0 >= -1 and y0 >= -1 and x1 <= w + 1 and y1 <= h + 1, (a.id, e)
            m = _mask(e.polygon, gray.shape)
            assert m.sum() > 0
            assert gray[m].std() > 4, (a.id, e.type, e.value)


def test_ink_does_not_escape_polygons(generated):
    """Independent geometry check (also after scaling to 60 % and 150 %):
    around each dark text on a light background no ink is left outside every polygon."""
    ctx, files, _ = generated
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    for path, a in files.items():
        rgb = np.asarray(Image.open(ctx.root / path).convert("RGB")).astype(np.int16)
        gray = rgb.mean(axis=2)
        # ink: dark and gray (the blue border of the selected cell does not count)
        dark = (gray < 105) & (rgb.max(axis=2) - rgb.min(axis=2) < 60)
        union = np.zeros(gray.shape, bool)
        for e in a.elements:
            union |= _mask(e.polygon, gray.shape)
        union = cv2.dilate(union.astype(np.uint8), kernel).astype(bool)
        for e in a.elements:
            if e.type == "face":
                continue
            # local window around the polygon (faster than working with the whole image)
            bx0, by0, bx1, by1 = bounding_box(e.polygon)
            x0, y0 = max(0, int(bx0) - 10), max(0, int(by0) - 10)
            x1, y1 = min(gray.shape[1], int(bx1) + 11), min(gray.shape[0], int(by1) + 11)
            sub = (slice(y0, y1), slice(x0, x1))
            m = _mask([[x - x0, y - y0] for x, y in e.polygon], (y1 - y0, x1 - x0))
            ring = cv2.dilate(m.astype(np.uint8), ring_k).astype(bool) & ~union[sub]
            g_sub, dark_sub = gray[sub], dark[sub]
            if not ring.any() or np.median(g_sub[ring]) < 150:
                continue  # light text on a dark bar
            inside = int((dark_sub & m).sum())
            outside = int((dark_sub & ring).sum())
            assert outside <= max(8, 0.03 * inside), (a.id, e.type, e.value, inside, outside)


def test_texts_do_not_overlap(generated):
    """No text spills over another one (except the RUT nested inside the URL)."""
    _, files, _ = generated
    for a in files.values():
        boxes = [(e, bounding_box(e.polygon)) for e in a.elements if e.type != "face"]
        for i, (e1, c1) in enumerate(boxes):
            for e2, c2 in boxes[i + 1 :]:
                if {e1.type, e2.type} == {"url", "rut"}:
                    continue
                ix = min(c1[2], c2[2]) - max(c1[0], c2[0])
                iy = min(c1[3], c2[3]) - max(c1[1], c2[1])
                if ix <= 0 or iy <= 0:
                    continue
                smaller = min((c[2] - c[0]) * (c[3] - c[1]) for c in (c1, c2))
                assert ix * iy < 0.05 * smaller, (a.id, e1.value, e2.value)


def test_counts_and_levels(generated):
    _, files, _ = generated
    for path in ("screenshots/correo_escritorio.png", "screenshots/correo_escritorio_hidpi.png"):
        c = Counter(e.type for e in files[path].elements)
        assert c["rut"] == 1 and c["phone"] == 2 and c["address"] == 1
        assert c["email"] >= 15 and c["name"] >= 15 and c["text"] >= 50
    chat = Counter(e.type for e in files["screenshots/chat_movil.jpg"].elements)
    assert chat["rut"] == 2 and chat["phone"] == 2 and chat["email"] == 1 and chat["address"] == 1
    assert chat["face"] == 1
    for path in ("screenshots/planilla.png", "screenshots/planilla_reducida.png"):
        el = files[path].elements
        c = Counter(e.type for e in el)
        assert c["name"] == 12 and c["email"] == 12 and c["phone"] == 12 and c["rut"] == 13
        assert sum(1 for e in el if e.tags.get("decoy") == "amount") == 12
        assert any(e.level == "out_of_scope" for e in el if e.type == "name")
    for path in ("screenshots/planilla_reducida.png", "screenshots/dialogo_pequeno.png"):
        assert all(e.level != "base" for e in files[path].elements if e.type != "text")
    form = Counter(e.type for e in files["screenshots/formulario_web.webp"].elements)
    assert form["url"] == 1 and form["face"] == 1 and form["rut"] == 2 and form["address"] == 1
    dialog = Counter(e.type for e in files["screenshots/dialogo_pequeno.png"].elements)
    assert dialog["name"] == 1 and dialog["email"] == 1
    for a in files.values():
        for e in a.elements:
            if e.type in ("name", "address") and not e.tags["in_list"]:
                assert e.level == "out_of_scope"


def test_font_size(generated):
    _, files, _ = generated
    for e in files["screenshots/correo_escritorio.png"].elements:
        assert 12 <= e.tags["size_px"] <= 20
    for e in files["screenshots/planilla.png"].elements:
        assert 10 <= e.tags["size_px"] <= 13
    reduced = [e.tags["size_px"] for e in files["screenshots/planilla_reducida.png"].elements]
    assert max(reduced) <= 8
    hidpi = [e.tags["size_px"] for e in files["screenshots/correo_escritorio_hidpi.png"].elements]
    assert min(hidpi) >= 15  # 150 % of 12 px, with some long email shrunk to fit


def test_faces_of_requested_size(generated):
    _, files, _ = generated
    (avatar,) = [e for e in files["screenshots/chat_movil.jpg"].elements if e.type == "face"]
    (profile,) = [e for e in files["screenshots/formulario_web.webp"].elements if e.type == "face"]
    for e, height in ((avatar, 90), (profile, 120)):
        x0, y0, x1, y1 = bounding_box(e.polygon)
        assert height * 0.8 <= y1 - y0 <= height * 1.2
        assert e.core is not None and e.level == "base"
