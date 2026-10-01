"""Rasterized geometry: coverage of a polygon by the redacted zones, and pixel uniformity.

Continuous coordinate convention (the same as ``test_bench.canvas``): pixel (i, j) covers
[i, i+1) x [j, j+1) and its center is (i + 0.5, j + 0.5). A pixel belongs to a polygon if
its center falls inside it.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from test_bench.schema import Polygon

COVERAGE_THRESHOLD = 0.95
FACE_CORE_THRESHOLD = 0.95
FACE_BOX_THRESHOLD = 0.80
FACE_NO_CORE_THRESHOLD = 0.90

# Supersampling for coverage: samples per coordinate unit.
PDF_SAMPLES = 8  # per point
IMAGE_SAMPLES = 4  # per pixel
_MAX_SAMPLES = 6_000_000  # cap of the window; above it the supersampling is reduced

# Pixel uniformity (checks P and I)
COLOR_TOLERANCE = 40
UNIFORM_THRESHOLD = 0.95
SMALL_UNIFORM_THRESHOLD = 0.90
SMALL_AREA = 20.0  # px²
# Control for P: a zone that correlates this much with the input was not redacted (low contrast)
UNCHANGED_THRESHOLD = 0.90
_MIN_DEVIATION = 1.0  # gray levels; below this the zone is flat and there is no correlation to measure

_SHIFT = 4  # fractional bits for cv2.fillPoly


def _to_cv(points: np.ndarray) -> np.ndarray:
    """Continuous coordinates -> integers with ``_SHIFT`` fractional bits in the OpenCV convention."""
    return np.round((points - 0.5) * (1 << _SHIFT)).astype(np.int32)


def fill(mask: np.ndarray, polygon: Sequence[Sequence[float]], origin=(0.0, 0.0), scale: float = 1.0) -> None:
    """Paints (value 1) in ``mask`` the pixels whose center falls inside the polygon."""
    p = (np.asarray(polygon, dtype=np.float64) - np.asarray(origin, dtype=np.float64)) * scale
    cv2.fillPoly(mask, [_to_cv(p)], 1, lineType=cv2.LINE_8, shift=_SHIFT)


def area(polygon: Sequence[Sequence[float]]) -> float:
    p = np.asarray(polygon, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def box(polygon: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    p = np.asarray(polygon, dtype=np.float64)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def boxes_touch(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def points_in_polygon(points: np.ndarray, polygon: Sequence[Sequence[float]]) -> np.ndarray:
    """Boolean mask of the points (N, 2) that fall inside the polygon (even-odd ray casting)."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    poly = np.asarray(polygon, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj - yi == 0, 1e-12, yj - yi) + xi)
        inside ^= crosses
        xj, yj = xi, yi
    return inside


def coverage(polygon: Polygon, zones: Sequence[Polygon], samples: float) -> float:
    """Fraction of the polygon covered by the union of ``zones`` (same page and coordinates)."""
    x0, y0, x1, y1 = box(polygon)
    candidates = [z for z in zones if boxes_touch((x0, y0, x1, y1), box(z))]
    if not candidates:
        return 0.0
    s = float(samples)
    width, height = (x1 - x0) * s, (y1 - y0) * s
    if width * height > _MAX_SAMPLES:
        s *= float(np.sqrt(_MAX_SAMPLES / (width * height)))
    w = int(np.ceil((x1 - x0) * s)) + 2
    h = int(np.ceil((y1 - y0) * s)) + 2
    origin = (x0 - 1 / s, y0 - 1 / s)
    gt = np.zeros((h, w), np.uint8)
    fill(gt, polygon, origin, s)
    total = int(gt.sum())
    if total == 0:
        # degenerate polygon: its centroid is evaluated
        cx, cy = np.asarray(polygon, dtype=np.float64).mean(axis=0)
        return 1.0 if any(point_in_polygon(cx, cy, z) for z in candidates) else 0.0
    cz = np.zeros((h, w), np.uint8)
    for z in candidates:
        fill(cz, z, origin, s)
    return float((gt & cz).sum()) / total


def intersect(a: Polygon, b: Polygon, samples: float = 2.0) -> bool:
    """Do the two polygons share area? (rasterized in the common window)."""
    ca, cb = box(a), box(b)
    if not boxes_touch(ca, cb):
        return False
    x0, y0 = max(ca[0], cb[0]), max(ca[1], cb[1])
    x1, y1 = min(ca[2], cb[2]), min(ca[3], cb[3])
    s = float(samples)
    if (x1 - x0) * (y1 - y0) * s * s > _MAX_SAMPLES:
        s = float(np.sqrt(_MAX_SAMPLES / max(1e-9, (x1 - x0) * (y1 - y0))))
    w, h = int(np.ceil((x1 - x0) * s)) + 2, int(np.ceil((y1 - y0) * s)) + 2
    origin = (x0 - 1 / s, y0 - 1 / s)
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    fill(ma, a, origin, s)
    fill(mb, b, origin, s)
    return bool((ma & mb).any())


# ---------------------------------------------------------------------------
# Pixel uniformity
# ---------------------------------------------------------------------------


def polygon_pixels(img: np.ndarray, polygon: Sequence[Sequence[float]]) -> tuple[np.ndarray, float]:
    """Pixels (N, C) of ``img`` inside the polygon (clipped to the image) and the polygon area in px².

    If the polygon is large a 1 px ring at the border is dropped: there the antialiasing of
    the filled zone mixes colors even when the redaction is perfect.
    """
    height, width = img.shape[:2]
    x0, y0, x1, y1 = box(polygon)
    ix0, iy0 = max(0, int(np.floor(x0)) - 1), max(0, int(np.floor(y0)) - 1)
    ix1, iy1 = min(width, int(np.ceil(x1)) + 1), min(height, int(np.ceil(y1)) + 1)
    a = area(polygon)
    if ix1 <= ix0 or iy1 <= iy0:
        return img[:0, :0].reshape(0, img.shape[2] if img.ndim == 3 else 1), a
    m = np.zeros((iy1 - iy0, ix1 - ix0), np.uint8)
    fill(m, polygon, (ix0, iy0))
    if m.sum() == 0:
        # polygon narrower than a pixel: the centroid pixel is taken
        cx, cy = np.asarray(polygon, dtype=np.float64).mean(axis=0)
        cx, cy = int(np.clip(cx - ix0, 0, m.shape[1] - 1)), int(np.clip(cy - iy0, 0, m.shape[0] - 1))
        m[cy, cx] = 1
    else:
        eroded = cv2.erode(m, np.ones((3, 3), np.uint8))
        if eroded.sum() >= 0.5 * m.sum() and eroded.sum() >= 12:
            m = eroded
    window = img[iy0:iy1, ix0:ix1]
    return window[m.astype(bool)], a


def uniformity(pixels: np.ndarray) -> float:
    """Fraction of pixels at distance (maximum over channels) <= 40 from the modal color of the region."""
    if len(pixels) == 0:
        return 1.0
    p = pixels.reshape(len(pixels), -1).astype(np.int16)
    q = (p // 16).astype(np.int32)
    code = np.zeros(len(p), np.int32)
    for c in range(q.shape[1]):
        code = code * 16 + q[:, c]
    modal = np.bincount(code).argmax()
    color = np.median(p[code == modal], axis=0)
    diff = np.abs(p - color).max(axis=1)
    return float((diff <= COLOR_TOLERANCE).mean())


def is_uniform(
    img: np.ndarray, polygon: Sequence[Sequence[float]], threshold: float | None = None
) -> tuple[bool | None, float | None]:
    """Check P/I over ``img`` (height, width, channels). ``None`` if the polygon falls outside.

    Threshold: 95 % of the pixels with the modal color (90 % if the polygon is smaller than
    20 px²), unless another one is given (faces without core use the same threshold as their
    coverage).
    """
    pix, a = polygon_pixels(img, polygon)
    if len(pix) == 0:
        return None, None
    f = uniformity(pix)
    if threshold is None:
        threshold = SMALL_UNIFORM_THRESHOLD if a < SMALL_AREA else UNIFORM_THRESHOLD
    return f >= threshold, f


def correlation(output: np.ndarray, input: np.ndarray, polygon: Sequence[Sequence[float]]) -> float | None:
    """Pearson correlation (in gray) between the output zone and the same zone of the input.

    High (>= ``UNCHANGED_THRESHOLD``) means the pixels are still the original ones (or nearly:
    recompressed), even if the zone looks uniform. ``None`` if either of the two is flat (a
    filled zone has no texture to correlate with).
    """
    height = min(output.shape[0], input.shape[0])
    width = min(output.shape[1], input.shape[1])
    a, _ = polygon_pixels(output[:height, :width], polygon)
    b, _ = polygon_pixels(input[:height, :width], polygon)
    if len(a) < 8 or len(a) != len(b):
        return None
    ga = a.reshape(len(a), -1).astype(np.float64).mean(axis=1)
    gb = b.reshape(len(b), -1).astype(np.float64).mean(axis=1)
    if ga.std() < _MIN_DEVIATION or gb.std() < _MIN_DEVIATION:
        return None
    return float(np.corrcoef(ga, gb)[0, 1])
