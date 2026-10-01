"""The ground truth must match the pixels: otherwise every later metric is false."""

import cv2
import numpy as np
import pytest
from PIL import Image

from test_bench.canvas import Canvas, bounding_box, polygon_area, rect, table_texture


def _ink(img: Image.Image, threshold: int = 200) -> np.ndarray:
    return np.asarray(img.convert("L")) < threshold


def _mask(polygon, shape, slack: float = 0.0) -> np.ndarray:
    """Mask of the polygon (dilated by ``slack`` pixels to tolerate antialiasing)."""
    m = np.zeros(shape, np.uint8)
    p = np.asarray(polygon, np.float64) - 0.5  # continuous coordinates -> pixel centers
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    if slack:
        k = int(np.ceil(slack)) * 2 + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    return m.astype(bool)


def _ink_fraction_outside(canvas: Canvas, slack: float = 1.5) -> float:
    ink = _ink(canvas.img)
    inside = np.zeros_like(ink)
    for e in canvas.elements:
        inside |= _mask(e.polygon, ink.shape, slack)
    return float((ink & ~inside).sum()) / max(1, int(ink.sum()))


@pytest.mark.parametrize("x,y", [(40, 80), (40.4, 80.7), (13.5, 61.25)])
def test_exact_ink_box(x, y):
    cv = Canvas.new(700, 160)
    e = cv.write_text(x, y, "RUT: 15.782.334-9 Peña", size=36)
    ys, xs = np.nonzero(np.asarray(cv.img.convert("L")) < 255)
    x0, y0, x1, y1 = bounding_box(e.polygon)
    assert abs(x0 - xs.min()) <= 1 and abs(x1 - (xs.max() + 1)) <= 1
    assert abs(y0 - ys.min()) <= 1 and abs(y1 - (ys.max() + 1)) <= 1


@pytest.mark.parametrize("degrees", [90, 180, 270, 15, 45, -30, 137])
def test_rotation_keeps_geometry(degrees):
    cv = Canvas.new(900, 500)
    cv.write_text(60, 120, "ana.rojas@ejemplo.cl", size=40)
    cv.write_text(300, 400, "+56 9 8123 4567", size=30)
    rot = cv.rotate(degrees)
    assert _ink_fraction_outside(rot) < 0.01
    for a, b in zip(cv.elements, rot.elements, strict=True):
        assert polygon_area(b.polygon) == pytest.approx(polygon_area(a.polygon), rel=1e-4)


def test_rotation_90_is_pixel_exact():
    cv = Canvas.new(400, 300)
    cv.write_text(50, 100, "12.345.678-5", size=30)
    for g in (90, 180, 270):
        rot = cv.rotate(g)
        ys, xs = np.nonzero(_ink(rot.img, 255))
        x0, y0, x1, y1 = bounding_box(rot.elements[0].polygon)
        assert (x0, y0, x1, y1) == pytest.approx((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), abs=1e-6)


def test_mirror_and_scale():
    cv = Canvas.new(500, 200)
    cv.write_text(30, 90, "Peña Muñoz", size=40)
    assert _ink_fraction_outside(cv.mirror()) < 0.01
    assert _ink_fraction_outside(cv.scale(0.37), slack=1.5) < 0.02


def test_rotated_text_stamp_style():
    cv = Canvas.new(800, 800)
    for g in (15, 45, 90, 200):
        cv.write_rotated(400, 400, "RUN 9.876.543-3", g, size=32)
        assert _ink_fraction_outside(cv) < 0.01


def test_perspective():
    rng = np.random.default_rng(0)
    cv = Canvas.new(600, 380, background=(250, 250, 250))
    cv.write_text(40, 100, "RUN 12.345.678-5", size=40)
    cv.write_text(40, 250, "ANA MARÍA ROJAS PEÑA", size=32)
    background = Image.new("RGB", (1200, 900), (255, 255, 255))
    dest = [[210, 140], [930, 205], [880, 700], [150, 610]]
    photo = cv.perspective(dest, background)
    assert _ink_fraction_outside(photo, slack=2) < 0.02
    # on a textured background it must also stay inside the card's quadrilateral
    photo2 = cv.perspective(dest, table_texture(1200, 900, rng))
    assert all(0 <= x <= 1200 and 0 <= y <= 900 for e in photo2.elements for x, y in e.polygon)


def test_paste_with_scale():
    cv = Canvas.new(600, 600)
    square = Image.new("RGB", (100, 80), (0, 0, 0))
    from test_bench.schema import Element

    elem = Element(type="face", page=0, polygon=rect(0, 0, 100, 80))
    (e,) = cv.paste(square, 123.4, 77.8, width=250, elements=[elem])
    ys, xs = np.nonzero(_ink(cv.img))
    assert bounding_box(e.polygon) == pytest.approx((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), abs=1)
