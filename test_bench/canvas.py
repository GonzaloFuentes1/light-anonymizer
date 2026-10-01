"""Drawing of text and images with an exact record of each element's geometry.

All coordinates are continuous: pixel (i, j) covers [i, i+1) x [j, j+1). The transformation
matrices are expressed in those coordinates; when calling OpenCV (which uses the pixel center
as the integer coordinate) they are converted with ``_to_cv_convention``.
"""

from __future__ import annotations

import copy
import io
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from test_bench.schema import Element, Polygon

# ---------------------------------------------------------------------------
# Fonts (DejaVu and STIX ship with matplotlib; free licenses, see LICENSES.md)
# ---------------------------------------------------------------------------

_FONTS_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"

FONTS = {
    "sans": "DejaVuSans.ttf",
    "sans_bold": "DejaVuSans-Bold.ttf",
    "sans_oblique": "DejaVuSans-Oblique.ttf",
    "serif": "DejaVuSerif.ttf",
    "serif_bold": "DejaVuSerif-Bold.ttf",
    "serif_italic": "DejaVuSerif-Italic.ttf",
    "mono": "DejaVuSansMono.ttf",
    "mono_bold": "DejaVuSansMono-Bold.ttf",
    "stix": "STIXGeneral.ttf",
    "stix_italic": "STIXGeneralItalic.ttf",
}


def font_path(name: str) -> Path:
    return _FONTS_DIR / FONTS[name]


@lru_cache(maxsize=256)
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path(name)), size)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    """Axis-aligned rectangle as a 4-point polygon (clockwise)."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def transform_points(polygon: Polygon, m: np.ndarray) -> Polygon:
    """Applies a 3x3 matrix (affine or homography) in continuous coordinates."""
    p = np.asarray(polygon, dtype=np.float64)
    h = np.hstack([p, np.ones((len(p), 1))]) @ m.T
    h = h[:, :2] / h[:, 2:3]
    return [[round(float(x), 3), round(float(y), 3)] for x, y in h]


def transform_elements(elements: Iterable[Element], m: np.ndarray) -> list[Element]:
    out = []
    for e in elements:
        e2 = copy.deepcopy(e)
        if e2.polygon is not None:
            e2.polygon = transform_points(e2.polygon, m)
        if e2.core is not None:
            e2.core = transform_points(e2.core, m)
        out.append(e2)
    return out


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


def _to_cv_convention(m: np.ndarray) -> np.ndarray:
    """Matrix in continuous coordinates -> matrix for cv2 (pixel centers at integers)."""
    return _translation(-0.5, -0.5) @ m @ _translation(0.5, 0.5)


def scale_matrix(sx: float, sy: float | None = None) -> np.ndarray:
    sy = sx if sy is None else sy
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def rotation_matrix(width: float, height: float, degrees: float, expand: bool = True) -> tuple[np.ndarray, int, int]:
    """Counterclockwise (visual) rotation around the center. Returns (matrix, new_width, new_height)."""
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    cx, cy = width / 2, height / 2
    if expand:
        new_w = abs(width * c) + abs(height * s)
        new_h = abs(width * s) + abs(height * c)
        new_w_i, new_h_i = int(math.ceil(new_w - 1e-6)), int(math.ceil(new_h - 1e-6))
    else:
        new_w_i, new_h_i = int(width), int(height)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
    m = _translation(new_w_i / 2, new_h_i / 2) @ rot @ _translation(-cx, -cy)
    return m, new_w_i, new_h_i


def polygon_area(polygon: Polygon) -> float:
    p = np.asarray(polygon, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def bounding_box(polygon: Polygon) -> tuple[float, float, float, float]:
    p = np.asarray(polygon, dtype=np.float64)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


@dataclass
class Canvas:
    """RGB image plus the list of elements (with exact geometry) drawn on it."""

    img: Image.Image
    elements: list[Element] = field(default_factory=list)

    @classmethod
    def new(cls, width: int, height: int, background: tuple[int, int, int] = (255, 255, 255)) -> Canvas:
        return cls(Image.new("RGB", (width, height), background))

    @property
    def width(self) -> int:
        return self.img.width

    @property
    def height(self) -> int:
        return self.img.height

    # -- text ----------------------------------------------------------------

    def write_text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        type: str = "text",
        value: str | None = None,
        font_name: str = "sans",
        size: int = 24,
        color: tuple[int, ...] = (25, 25, 25),
        level: str = "base",
        tags: dict[str, Any] | None = None,
        anchor: str = "ls",
        register: bool = True,
    ) -> Element | None:
        """Writes ``text`` with its baseline at (x, y) and records the exact ink box.

        ``value`` is the data to look for in the output; by default it is the text itself.
        """
        f = font(font_name, size)
        draw = ImageDraw.Draw(self.img)
        draw.text((x, y), text, font=f, fill=color, anchor=anchor)
        if not register:
            return None
        box = _ink_box(text, f, x, y, anchor)
        if box is None:
            return None
        e = Element(
            type=type,
            page=0,
            polygon=rect(*box),
            value=value if value is not None else text,
            level=level,
            layer="raster",
            tags={"font": font_name, "size_px": size, "angle": 0, **(tags or {})},
        )
        self.elements.append(e)
        return e

    def write_line(
        self,
        x: float,
        y: float,
        parts: list[tuple[str, str]],
        **kwargs: Any,
    ) -> list[Element]:
        """Writes consecutive segments on the same baseline: ``[(text, type), ...]``.

        Segments of type ``"text"`` are neutral labels ("RUT: "); the rest are data.
        Returns the recorded elements.
        """
        f = font(kwargs.get("font_name", "sans"), kwargs.get("size", 24))
        recorded = []
        for text, type_ in parts:
            e = self.write_text(x, y, text, type=type_, **kwargs)
            if e is not None:
                recorded.append(e)
            x += ImageDraw.Draw(self.img).textlength(text, font=f)
        return recorded

    def text_width(self, text: str, font_name: str = "sans", size: int = 24) -> float:
        return ImageDraw.Draw(self.img).textlength(text, font=font(font_name, size))

    def write_rotated(
        self,
        cx: float,
        cy: float,
        text: str,
        degrees: float,
        *,
        type: str = "text",
        value: str | None = None,
        font_name: str = "sans",
        size: int = 24,
        color: tuple[int, int, int] = (25, 25, 25),
        level: str = "base",
        tags: dict[str, Any] | None = None,
    ) -> Element:
        """Writes text rotated by ``degrees`` (counterclockwise) centered at (cx, cy), like a stamp."""
        f = font(font_name, size)
        margin = size
        x0, t, r, b = f.getbbox(text, anchor="ls")
        w, h = int(r - x0 + 2 * margin), int(b - t + 2 * margin)
        patch = Canvas(Image.new("RGBA", (w, h), (0, 0, 0, 0)))
        e = patch.write_text(
            margin - x0,
            margin - t,
            text,
            type=type,
            value=value,
            font_name=font_name,
            size=size,
            color=(*color, 255),
            level=level,
            tags=tags,
        )
        assert e is not None
        m, nw, nh = rotation_matrix(w, h, degrees)
        rotated = cv2.warpAffine(
            np.asarray(patch.img),
            _to_cv_convention(m)[:2],
            (nw, nh),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        rotated_img = Image.fromarray(rotated, "RGBA")
        px, py = int(round(cx - nw / 2)), int(round(cy - nh / 2))
        self.img.paste(rotated_img.convert("RGB"), (px, py), rotated_img.getchannel("A"))
        m = _translation(px, py) @ m
        (e2,) = transform_elements([e], m)
        e2.tags["angle"] = degrees
        self.elements.append(e2)
        return e2

    def write_irregular(
        self,
        x: float,
        y: float,
        text: str,
        rng: np.random.Generator,
        *,
        type: str = "text",
        value: str | None = None,
        font_name: str = "sans_oblique",
        size: int = 28,
        color: tuple[int, int, int] = (20, 30, 90),
        level: str = "stress",
        tags: dict[str, Any] | None = None,
        jitter: float = 0.12,
    ) -> Element | None:
        """Text with letters of irregular size, rotation and position (imitates handwriting)."""
        f0 = font(font_name, size)
        boxes = []
        for ch in text:
            if ch == " ":
                x += f0.getlength(" ")
                continue
            size_c = max(8, int(size * (1 + rng.uniform(-jitter, jitter))))
            f = font(font_name, size_c)
            x0, t, r, b = f.getbbox(ch, anchor="ls")
            m = size_c
            patch = Image.new("RGBA", (int(r - x0 + 2 * m), int(b - t + 2 * m)), (0, 0, 0, 0))
            ImageDraw.Draw(patch).text((m - x0, m - t), ch, font=f, fill=(*color, 255), anchor="ls")
            patch = patch.rotate(float(rng.uniform(-12, 12)), resample=Image.Resampling.BICUBIC, expand=True)
            px = int(round(x + x0 - m + rng.uniform(-1.5, 1.5)))
            py = int(round(y + t - m + rng.uniform(-3, 3)))
            self.img.paste(patch, (px, py), patch)
            bb = patch.getchannel("A").getbbox()
            if bb:
                boxes.append((px + bb[0], py + bb[1], px + bb[2], py + bb[3]))
            x += f.getlength(ch) * (1 + rng.uniform(-0.05, 0.1))
        if not boxes:
            return None
        c = np.array(boxes, dtype=np.float64)
        e = Element(
            type=type,
            page=0,
            polygon=rect(c[:, 0].min(), c[:, 1].min(), c[:, 2].max(), c[:, 3].max()),
            value=value if value is not None else text,
            level=level,
            layer="raster",
            tags={"font": font_name, "size_px": size, "angle": 0, "irregular": True, **(tags or {})},
        )
        self.elements.append(e)
        return e

    # -- images --------------------------------------------------------------

    def paste(
        self,
        img: Image.Image,
        x: float,
        y: float,
        width: int | None = None,
        elements: Iterable[Element] = (),
        mask: Image.Image | None = None,
    ) -> list[Element]:
        """Pastes ``img`` (optionally rescaled to ``width``) with its corner at (x, y).

        ``elements`` are in ``img`` coordinates and are translated/scaled onto the canvas.
        """
        sx = sy = 1.0
        if width is not None and width != img.width:
            new_size = (width, max(1, int(round(img.height * width / img.width))))
            sx, sy = new_size[0] / img.width, new_size[1] / img.height
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            if mask is not None:
                mask = mask.resize(new_size, Image.Resampling.LANCZOS)
        xi, yi = int(round(x)), int(round(y))
        if mask is None and img.mode == "RGBA":
            mask = img.getchannel("A")
        self.img.paste(img.convert("RGB"), (xi, yi), mask)
        m = _translation(xi, yi) @ scale_matrix(sx, sy)
        new_elements = transform_elements(elements, m)
        self.elements.extend(new_elements)
        return new_elements

    # -- geometric transformations (they return a new canvas) -----------------

    def rotate(self, degrees: float, background: tuple[int, int, int] = (255, 255, 255)) -> Canvas:
        """Rotates the whole image (counterclockwise) expanding the canvas. Exact for multiples of 90."""
        g = degrees % 360
        w, h = self.width, self.height
        if g in (0, 90, 180, 270):
            transpose = {
                0: None,
                90: Image.Transpose.ROTATE_90,
                180: Image.Transpose.ROTATE_180,
                270: Image.Transpose.ROTATE_270,
            }[int(g)]
            img = self.img.copy() if transpose is None else self.img.transpose(transpose)
            m = {
                0: np.eye(3),
                90: np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], dtype=np.float64),
                180: np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]], dtype=np.float64),
                270: np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
            }[int(g)]
        else:
            m, nw, nh = rotation_matrix(w, h, g)
            arr = cv2.warpAffine(
                np.asarray(self.img),
                _to_cv_convention(m)[:2],
                (nw, nh),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=background,
            )
            img = Image.fromarray(arr)
        new_elements = transform_elements(self.elements, m)
        for e in new_elements:
            e.tags["angle"] = (e.tags.get("angle", 0) + degrees) % 360
        return Canvas(img, new_elements)

    def mirror(self, horizontal: bool = True) -> Canvas:
        w, h = self.width, self.height
        if horizontal:
            img = self.img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            m = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        else:
            img = self.img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            m = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]], dtype=np.float64)
        new_elements = transform_elements(self.elements, m)
        for e in new_elements:
            e.tags["mirror"] = "horizontal" if horizontal else "vertical"
        return Canvas(img, new_elements)

    def scale(self, factor: float) -> Canvas:
        new_size = (max(1, int(round(self.width * factor))), max(1, int(round(self.height * factor))))
        img = self.img.resize(new_size, Image.Resampling.LANCZOS)
        m = scale_matrix(new_size[0] / self.width, new_size[1] / self.height)
        return Canvas(img, transform_elements(self.elements, m))

    def scan_tilt(self, degrees: float, background: tuple[int, int, int] = (250, 250, 247)) -> Canvas:
        """Slight rotation without expanding (like a sheet badly placed on the scanner)."""
        m, nw, nh = rotation_matrix(self.width, self.height, degrees, expand=False)
        arr = cv2.warpAffine(
            np.asarray(self.img),
            _to_cv_convention(m)[:2],
            (nw, nh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background,
        )
        new_elements = transform_elements(self.elements, m)
        for e in new_elements:
            e.tags["angle"] = (e.tags.get("angle", 0) + degrees) % 360
        return Canvas(Image.fromarray(arr), new_elements)

    def perspective(
        self,
        dest: Polygon,
        background: Image.Image,
    ) -> Canvas:
        """Projects the whole image onto the quadrilateral ``dest`` inside ``background``.

        ``dest`` are the 4 corners (top-left, top-right, bottom-right, bottom-left) in
        ``background`` coordinates.
        """
        w, h = self.width, self.height
        src = np.array(rect(0, 0, w, h), dtype=np.float32)
        dst = np.array(dest, dtype=np.float32)
        m = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        size = (background.width, background.height)
        mcv = _to_cv_convention(m)
        arr = cv2.warpPerspective(
            np.asarray(self.img), mcv, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), mcv, size, flags=cv2.INTER_LINEAR)
        base = np.asarray(background.convert("RGB")).astype(np.float32)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        comp = (arr.astype(np.float32) * alpha + base * (1 - alpha)).clip(0, 255).astype(np.uint8)
        new_elements = transform_elements(self.elements, m)
        for e in new_elements:
            e.tags["perspective"] = True
        return Canvas(Image.fromarray(comp), new_elements)

    def copy(self) -> Canvas:
        return Canvas(self.img.copy(), copy.deepcopy(self.elements))


def _ink_box(text: str, f: ImageFont.FreeTypeFont, x: float, y: float, anchor: str) -> tuple | None:
    """Exact box of the text's ink pixels, drawing it on a separate mask."""
    x0, t, r, b = f.getbbox(text, anchor=anchor)
    m = max(4, f.size // 2)
    w, h = int(math.ceil(r - x0)) + 2 * m, int(math.ceil(b - t)) + 2 * m
    if w <= 2 * m or h <= 2 * m:
        return None
    mask = Image.new("L", (w, h), 0)
    ox, oy = m - x0, m - t
    # Replicates the same fractional offset ImageDraw uses when drawing at (x, y).
    ImageDraw.Draw(mask).text(
        (ox + (x - math.floor(x)), oy + (y - math.floor(y))), text, font=f, fill=255, anchor=anchor
    )
    bb = mask.getbbox()
    if bb is None:
        return None
    bx, by = math.floor(x) - ox, math.floor(y) - oy
    return (bx + bb[0], by + bb[1], bx + bb[2], by + bb[3])


# ---------------------------------------------------------------------------
# Degradations (they do not change the geometry)
# ---------------------------------------------------------------------------


def noise(img: Image.Image, rng: np.random.Generator, sigma: float = 8.0) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr = arr + rng.normal(0, sigma, arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def salt_pepper(img: Image.Image, rng: np.random.Generator, fraction: float = 0.002) -> Image.Image:
    arr = np.asarray(img).copy()
    n = int(arr.shape[0] * arr.shape[1] * fraction)
    ys, xs = rng.integers(0, arr.shape[0], n), rng.integers(0, arr.shape[1], n)
    arr[ys[: n // 2], xs[: n // 2]] = 0
    arr[ys[n // 2 :], xs[n // 2 :]] = 255
    return Image.fromarray(arr)


def blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def jpeg_compress(img: Image.Image, quality: int = 60) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def lighting(img: Image.Image, rng: np.random.Generator, intensity: float = 0.35) -> Image.Image:
    """Light gradient and vignette, like a photo taken with a phone."""
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5)) * 2
    vin = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2
    factor = 1 - intensity * (0.5 * (grad + 1) * 0.6 + vin * 0.4)
    arr = np.asarray(img).astype(np.float32) * factor[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def paper_texture(width: int, height: int, rng: np.random.Generator, color=(248, 246, 240)) -> Image.Image:
    base = np.ones((height, width, 3), np.float32) * np.array(color, np.float32)
    low_noise = cv2.resize(rng.normal(0, 1, (height // 16 + 1, width // 16 + 1)).astype(np.float32), (width, height))
    base += low_noise[..., None] * 3 + rng.normal(0, 2, (height, width, 1))
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def table_texture(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    """Wood-like background for photos of documents lying on a table."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    grain = np.sin(xx / width * rng.uniform(20, 40) + np.sin(yy / height * 6) * 2.5) * 0.5 + 0.5
    low_noise = cv2.resize(rng.normal(0, 1, (height // 24 + 1, width // 24 + 1)).astype(np.float32), (width, height))
    base = np.array([120, 84, 52], np.float32)
    arr = base + (grain[..., None] * 30) + low_noise[..., None] * 8 + rng.normal(0, 3, (height, width, 1))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# From image to PDF
# ---------------------------------------------------------------------------


def px_to_pt(elements: Iterable[Element], dpi: float) -> list[Element]:
    return transform_elements(elements, scale_matrix(72.0 / dpi))


def map_to_rect(
    elements: Iterable[Element], width_px: int, height_px: int, dest: tuple[float, float, float, float]
) -> list[Element]:
    """Maps elements in pixels of an image onto the box ``dest`` (x0, y0, x1, y1) of a PDF page."""
    x0, y0, x1, y1 = dest
    m = _translation(x0, y0) @ scale_matrix((x1 - x0) / width_px, (y1 - y0) / height_px)
    return transform_elements(elements, m)
