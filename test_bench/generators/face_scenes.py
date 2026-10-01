"""Single faces and faces in scenes: portraits, rotations, composed groups, real photos and a photographed poster.

Every face comes from ``ctx.faces`` (``FaceProvider``): ``take(pose, n)`` for single faces with
their box and core, and ``scenes()`` for real photos with several annotated people.
The code assumes nothing about the size or aspect of the source images: all the geometry is
derived from ``box`` and ``core`` and is transformed together with the pixels.

Ground-truth conventions of this module:

- ``polygon`` = transformed face box; ``core`` = transformed core (or ``None`` if the real
  scene does not provide it).
- ``tags["height_px"]`` = height of the face in the output image, measured in its own frame
  (length of the "vertical" edge of the box: it does not change with rotation).
  ``bounding_height_px`` is the height of the axis-aligned bounding box.
- ``tags["angle"]`` = accumulated counterclockwise rotation, in degrees.
"""

from __future__ import annotations

import math
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from test_bench.canvas import (
    Canvas,
    blur,
    bounding_box,
    lighting,
    noise,
    rotation_matrix,
    scale_matrix,
    table_texture,
    transform_elements,
)
from test_bench.context import Context, image_file_entry, save_image
from test_bench.faces import Face, FaceProvider
from test_bench.fake_data import EMAIL_FORMATS, PHONE_FORMATS, PHONE_FORMATS_BY_KIND, FakeData, Person
from test_bench.schema import Element, FileEntry, Polygon

SEED_NAME = "rostros_escenas"  # seed string: kept in Spanish so the generated files stay identical
CATEGORY = "faces"
FOLDER = "faces"
PREFIX = "rost"

PORTRAIT_SIDE = (420, 780)  # longer side of the single portraits (within 400-800)
MAX_SCENE_SIDE = 1600
MIN_BASE_SCENE_HEIGHT = 24  # below this a real-scene face is "stress"
MIN_BASE_GROUP_HEIGHT = 40  # in the composed groups, 20-40 px is "stress"

WALLS = [(214, 206, 190), (196, 210, 214), (222, 214, 196), (205, 196, 210), (190, 204, 186)]
CLOTHES = [(40, 60, 110), (120, 30, 40), (60, 90, 60), (30, 30, 35), (150, 150, 155), (180, 120, 50), (90, 50, 110)]
POSTER_COLORS = [(22, 92, 125), (150, 40, 50), (40, 110, 70), (90, 60, 130)]
POSTER_PHRASES = [
    "Inscripciones abiertas para vecinos y vecinas de la comuna.",
    "Actividad gratuita, con cupos limitados por orden de llegada.",
    "Lugar: salón multiuso de la junta de vecinos del sector.",
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate(ctx: Context) -> list[FileEntry]:
    rng = ctx.rng(SEED_NAME)
    f = ctx.fake_data(SEED_NAME)
    files: list[FileEntry] = []
    files += _portraits(ctx, rng)
    files += _rotated(ctx, rng)
    files += _groups(ctx, rng)
    files += _real_scenes(ctx, rng)
    files.append(_poster(ctx, rng, f))
    files += _variants(ctx, rng)
    return files


# ---------------------------------------------------------------------------
# Geometry and saving utilities
# ---------------------------------------------------------------------------


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


def _cv_matrix(m: np.ndarray) -> np.ndarray:
    """Matrix in continuous coordinates -> 2x3 matrix for cv2 (pixel centers at integers)."""
    return (_translation(-0.5, -0.5) @ m @ _translation(0.5, 0.5))[:2]


def _own_height(polygon: Polygon) -> float:
    """Height of the face in its own frame: mean of the left and right edges of the box."""
    p = np.asarray(polygon, dtype=np.float64)
    if len(p) != 4:
        x0, y0, x1, y1 = bounding_box(polygon)
        return y1 - y0
    return float((np.hypot(*(p[3] - p[0])) + np.hypot(*(p[2] - p[1]))) / 2)


def _face_height(e: Element) -> float:
    """Face height: in its own frame for the generated ones; from the bounding box in real scenes.

    Real-scene boxes come from external annotations with no guaranteed vertex order
    (and no rotation), so the bounding-box height is used there.
    """
    if "scene_id" in e.tags:
        x0, y0, x1, y1 = bounding_box(e.polygon)
        return y1 - y0
    return _own_height(e.polygon)


def _complete_faces(cv: Canvas) -> None:
    """Records the final height of each face and normalizes the angle."""
    for e in cv.elements:
        if e.type != "face" or e.polygon is None:
            continue
        x0, y0, x1, y1 = bounding_box(e.polygon)
        e.tags["height_px"] = round(_face_height(e), 1)
        e.tags["bounding_height_px"] = round(y1 - y0, 1)
        e.tags["angle"] = round(float(e.tags.get("angle", 0)) % 360, 3)


def _save_gray(img: Image.Image, dest, quality: int) -> None:
    """Single-channel JPEG, without metadata (``save_image`` always converts to RGB)."""
    gray = img.convert("L")
    Image.frombytes("L", gray.size, gray.tobytes()).save(dest, "JPEG", quality=quality)


def _save(
    ctx: Context,
    rng: np.random.Generator,
    cv: Canvas,
    name: str,
    description: str,
    tags: dict[str, Any] | None = None,
    gray: bool = False,
) -> FileEntry:
    _complete_faces(cv)
    path = f"{FOLDER}/{name}.jpg"
    dest = ctx.path(path)
    quality = int(rng.integers(86, 95))
    if gray:
        _save_gray(cv.img, dest, quality)
    else:
        save_image(cv.img, dest, "jpg", quality=quality)
    return image_file_entry(
        id=f"{PREFIX}_{name}",
        path=path,
        format="jpg",
        category=CATEGORY,
        description=description,
        canvas=cv,
        tags={"synthetic_faces": ctx.faces.synthetic_only, "jpeg_quality": quality, **(tags or {})},
    )


def _portrait(r: Face, level: str = "base", tags: dict[str, Any] | None = None) -> Canvas:
    """Canvas with the face photo as is and its element (in photo coordinates)."""
    img = r.img.convert("RGB").copy()
    return Canvas(img, FaceProvider.elements(r, level, {"angle": 0, **(tags or {})}))


def _side_factor(width: float, height: float, rng: np.random.Generator, side: tuple[int, int] = PORTRAIT_SIDE) -> float:
    target = float(rng.integers(side[0], side[1] + 1))
    return target / max(width, height)


def _fit_side(cv: Canvas, rng: np.random.Generator, side: tuple[int, int] = PORTRAIT_SIDE) -> Canvas:
    return cv.scale(_side_factor(cv.width, cv.height, rng, side))


def _rotate_with_background(cv: Canvas, degrees: float, rng: np.random.Generator) -> Canvas:
    """Rotates the photo; if the angle is not a multiple of 90, it becomes a printed copy on a table.

    The corners left empty by the rotation are filled with wood (``table_texture``) and a soft
    shadow under the photo. Replicating the border of the photo itself dragged the color of the
    neck or the clothes into the corners and left implausible smudges.
    """
    if degrees % 90 == 0:
        return cv.rotate(degrees)
    w, h = cv.width, cv.height
    m, nw, nh = rotation_matrix(w, h, degrees)
    mcv = _cv_matrix(m)
    arr = cv2.warpAffine(np.asarray(cv.img), mcv, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine(
        np.full((h, w), 255, np.uint8), mcv, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    a = alpha.astype(np.float32)[..., None] / 255.0
    table = np.asarray(table_texture(nw, nh, rng)).astype(np.float32)
    shifted = cv2.warpAffine(alpha, np.float32([[1, 0, 4], [0, 1, 6]]), (nw, nh)).astype(np.float32)
    shadow = cv2.GaussianBlur(shifted, (0, 0), 5.0)[..., None] / 255.0
    background = table * (1 - 0.45 * shadow)
    comp = arr.astype(np.float32) * a + background * (1 - a)
    new = transform_elements(cv.elements, m)
    for e in new:
        e.tags["angle"] = (e.tags.get("angle", 0) + degrees) % 360
        e.tags["rotation_fill"] = "photo_on_table"
    return Canvas(Image.fromarray(comp.clip(0, 255).astype(np.uint8)), new)


def _skin_color(img: Image.Image, core: Polygon) -> tuple[int, int, int]:
    """Median color of the central band of the core (cheeks and nose)."""
    x0, y0, x1, y1 = bounding_box(core)
    w, h = x1 - x0, y1 - y0
    box = (int(x0 + 0.05 * w), int(y0 + 0.4 * h), int(math.ceil(x1 - 0.05 * w)), int(math.ceil(y0 + 0.6 * h)))
    region = np.asarray(img.convert("RGB").crop(box)).reshape(-1, 3)
    if len(region) == 0:
        return (200, 160, 130)
    med = np.median(region, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


# ---------------------------------------------------------------------------
# 1-2. Single portraits
# ---------------------------------------------------------------------------


def _portraits(ctx: Context, rng: np.random.Generator) -> list[FileEntry]:
    out = []
    for i, r in enumerate(ctx.faces.take("front", 4), 1):
        cv = _fit_side(_portrait(r, "base"), rng)
        out.append(_save(ctx, rng, cv, f"retrato_frente_{i:02d}", "Retrato de frente, una sola persona."))
    for i, r in enumerate(ctx.faces.take("three_quarter", 2), 1):
        cv = _fit_side(_portrait(r, "base"), rng)
        out.append(_save(ctx, rng, cv, f"retrato_tres_cuartos_{i:02d}", "Retrato en tres cuartos, una sola persona."))
    for i, r in enumerate(ctx.faces.take("profile", 2), 1):
        cv = _fit_side(_portrait(r, "stress"), rng)
        out.append(_save(ctx, rng, cv, f"perfil_{i:02d}", "Retrato de perfil (difícil para detectores)."))
    return out


# ---------------------------------------------------------------------------
# 3. Rotated portrait
# ---------------------------------------------------------------------------


def _rotated(ctx: Context, rng: np.random.Generator) -> list[FileEntry]:
    (r,) = ctx.faces.take("front", 1)
    out = []
    for g in (90, 180, 270, 15, 45):
        cv = _portrait(r, "base")
        t = math.radians(g)
        rw = abs(cv.width * math.cos(t)) + abs(cv.height * math.sin(t))
        rh = abs(cv.width * math.sin(t)) + abs(cv.height * math.cos(t))
        cv = cv.scale(_side_factor(rw, rh, rng))
        cv = _rotate_with_background(cv, g, rng)
        out.append(
            _save(
                ctx,
                rng,
                cv,
                f"retrato_rotado_{g}",
                f"Retrato girado {g}° (antihorario); el motor debe encontrar caras en cualquier orientación.",
                {"angle": g},
            )
        )
    return out


# ---------------------------------------------------------------------------
# 4. Composed groups
# ---------------------------------------------------------------------------


Ellipse = tuple[float, float, float, float]  # cx, cy, rx, ry


def _inside_ellipses(pts: np.ndarray, ellipses: list[Ellipse]) -> np.ndarray:
    inside = np.zeros(len(pts), dtype=bool)
    for cx, cy, rx, ry in ellipses:
        inside |= ((pts[:, 0] - cx) / rx) ** 2 + ((pts[:, 1] - cy) / ry) ** 2 <= 1.0
    return inside


def _mesh(box: tuple[float, float, float, float], n: int = 16, margin: float = 0.0) -> np.ndarray:
    x0, y0, x1, y1 = box
    xs = np.linspace(x0 - margin, x1 + margin, n)
    ys = np.linspace(y0 - margin, y1 + margin, n)
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)


def _room_background(w: int, h: int, rng: np.random.Generator) -> Image.Image:
    """Simple room: wall with a light gradient, a window, an abstract painting and a wooden floor."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wall = np.array(WALLS[int(rng.integers(len(WALLS)))], np.float32)
    light = 1.0 - 0.16 * (yy / h) - 0.10 * np.abs(xx / w - rng.uniform(0.3, 0.7))
    low = cv2.resize(rng.normal(0, 1, (h // 40 + 1, w // 40 + 1)).astype(np.float32), (w, h))
    arr = wall * light[..., None] + low[..., None] * 3
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    floor_y = int(h * rng.uniform(0.74, 0.82))
    img.paste(table_texture(w, h - floor_y, rng), (0, floor_y))
    d.rectangle((0, floor_y - 12, w, floor_y), fill=tuple(int(c * 0.7) for c in wall))
    # window with sky
    vw, vh = int(w * rng.uniform(0.18, 0.26)), int(h * rng.uniform(0.3, 0.38))
    vx, vy = int(rng.integers(int(w * 0.05), int(w * 0.7))), int(h * rng.uniform(0.06, 0.12))
    sky = np.linspace([150, 190, 230], [215, 230, 245], vh).astype(np.uint8)[:, None, :].repeat(vw, 1)
    img.paste(Image.fromarray(sky), (vx, vy))
    frame = (245, 245, 240)
    d.rectangle((vx - 8, vy - 8, vx + vw + 8, vy + vh + 8), outline=frame, width=10)
    d.line((vx + vw // 2, vy, vx + vw // 2, vy + vh), fill=frame, width=8)
    d.line((vx, vy + vh // 2, vx + vw, vy + vh // 2), fill=frame, width=8)
    # abstract painting on the other half of the wall
    cx0 = int(w * 0.55) if vx < w * 0.4 else int(w * 0.1)
    cw, ch = int(w * 0.16), int(h * 0.2)
    cy0 = int(h * rng.uniform(0.08, 0.16))
    d.rectangle((cx0, cy0, cx0 + cw, cy0 + ch), fill=(60, 45, 35))
    d.rectangle((cx0 + 10, cy0 + 10, cx0 + cw - 10, cy0 + ch - 10), fill=(235, 230, 215))
    for _ in range(4):
        c = tuple(int(v) for v in rng.integers(40, 220, 3))
        a, b = rng.uniform(0.1, 0.6, 2)
        d.rectangle(
            (
                cx0 + 14 + a * cw * 0.6,
                cy0 + 14 + b * ch * 0.6,
                cx0 + 14 + (a + 0.3) * cw * 0.6,
                cy0 + (b + 0.4) * ch * 0.6,
            ),
            fill=c,
        )
    return blur(img, 1.6)


def _prepare_head(r: Face) -> tuple[Image.Image, Image.Image, list[Element], Ellipse]:
    """Crops the head with a feathered elliptical mask that keeps the core fully opaque.

    Returns (crop, L mask, elements in crop coordinates, outer ellipse in the crop).
    """
    img = r.img.convert("RGB")
    w, h = img.size
    x0, y0, x1, y1 = bounding_box(r.box)
    cw, ch = x1 - x0, y1 - y0
    # The box may go from the eyebrows to the chin: the ellipse rises to include forehead and hair.
    top, bottom = y0 - 0.45 * ch, y1 + 0.08 * ch
    cx, cy = (x0 + x1) / 2, (top + bottom) / 2
    rx, ry = 0.62 * cw, (bottom - top) / 2
    feather = 0.05 * min(rx, ry)
    corners = np.asarray(r.core, dtype=np.float64)
    for _ in range(30):
        ex, ey = rx - 2.5 * feather, ry - 2.5 * feather
        if np.all(((corners[:, 0] - cx) / ex) ** 2 + ((corners[:, 1] - cy) / ey) ** 2 <= 1.0):
            break
        rx, ry = rx * 1.04, ry * 1.04
    margin = 3 * feather
    c0 = (max(0, int(cx - rx - margin)), max(0, int(cy - ry - margin)))
    c1 = (min(w, int(math.ceil(cx + rx + margin))), min(h, int(math.ceil(cy + ry + margin))))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(feather)).crop((*c0, *c1))
    # ramp on the crop borders, in case the ellipse touches the border of the photo
    mw, mh = mask.size
    ramp_x = np.minimum(np.arange(mw) + 0.5, mw - np.arange(mw) - 0.5) / max(1.0, feather)
    ramp_y = np.minimum(np.arange(mh) + 0.5, mh - np.arange(mh) - 0.5) / max(1.0, feather)
    ramp = np.clip(np.minimum(ramp_y[:, None], ramp_x[None, :]), 0, 1)
    crop = img.crop((*c0, *c1))
    background = _alpha_without_background(img, crop, c0, r, (x0, y0, x1, y1))
    marr = (np.asarray(mask).astype(np.float32) * ramp * background).astype(np.uint8)
    elements = transform_elements(FaceProvider.elements(r, "base", {"angle": 0}), _translation(-c0[0], -c0[1]))
    ellipse = (cx - c0[0], cy - c0[1], rx + 2 * feather, ry + 2 * feather)
    return crop, Image.fromarray(marr, "L"), elements, ellipse


def _alpha_without_background(
    img: Image.Image,
    crop: Image.Image,
    c0: tuple[int, int],
    r: Face,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    """Alpha (0-1) that removes the plain studio background around the head, if there is one.

    Patches are sampled at the top corners and at mid-height of the side borders of the
    photo; the plain ones (studio background, no hair or clothes) give the background color.
    Pixels similar to that color become transparent, except inside the face oval and the
    core, which always stay opaque. If no patch is plain (varied background) it returns ones.
    """
    arr = np.asarray(img).astype(np.float32)
    rec = np.asarray(crop).astype(np.float32)
    h, w = arr.shape[:2]
    k = max(4, min(h, w) // 24)
    patches = [arr[:k, :k], arr[:k, -k:], arr[h // 2 - k // 2 : h // 2 + k // 2, :k]]
    patches.append(arr[h // 2 - k // 2 : h // 2 + k // 2, -k:])
    plain = [np.median(p.reshape(-1, 3), axis=0) for p in patches if p.size and p.reshape(-1, 3).std(axis=0).max() < 6]
    if not plain:
        return np.ones(rec.shape[:2], np.float32)
    # the background color is the one most patches share (a plain patch of hair does not shift it)
    cand = np.asarray(plain)
    close = np.linalg.norm(cand[:, None] - cand[None, :], axis=2) < 20
    best = int(np.argmax(close.sum(axis=1) + cand.mean(axis=1) / 1000))
    color = np.median(cand[close[best]], axis=0)
    dist = np.linalg.norm(rec - color, axis=2)
    alpha = np.clip((dist - 16) / 26, 0, 1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # face oval and core: always opaque
    protected = Image.new("L", crop.size, 0)
    dp = ImageDraw.Draw(protected)
    x0, y0, x1, y1 = (v - d for v, d in zip(box, (*c0, *c0), strict=True))
    mx, my = (
        0.1 * (x1 - x0),
        0.1 * (y1 - y0),
    )  # oval a bit smaller than the box: in profiles the box includes background
    dp.ellipse((x0 + mx, y0 + my, x1 - mx, y1 - my), fill=255)
    nx0, ny0, nx1, ny1 = bounding_box(r.core)
    dp.rectangle((nx0 - c0[0] - 4, ny0 - c0[1] - 4, nx1 - c0[0] + 4, ny1 - c0[1] + 4), fill=255)
    prot = np.asarray(protected).astype(np.float32) / 255.0
    alpha = np.maximum(alpha, prot)
    return cv2.GaussianBlur(alpha, (0, 0), 1.2)


def _group_heights(rng: np.random.Generator) -> list[tuple[float, str]]:
    n = int(rng.integers(5, 13))
    n_tiny = int(rng.integers(2, 4))
    n_large = int(rng.integers(1, min(4, n - n_tiny - 1) + 1))
    n_medium = n - n_tiny - n_large
    heights = [(float(rng.uniform(22, 38)), "tiny") for _ in range(n_tiny)]
    heights += [(float(rng.uniform(42, 78)), "small") for _ in range(n_medium)]
    heights += [(float(rng.uniform(85, 195)), "large") for _ in range(n_large)]
    return sorted(heights)  # far (small) ones first: the near ones cover the far ones


def _group(ctx: Context, rng: np.random.Generator, width: int = 1600, height: int = 1000) -> Canvas:
    cv = Canvas(_room_background(width, height, rng))
    d = ImageDraw.Draw(cv.img)
    heights = _group_heights(rng)
    force = set(int(i) for i in rng.choice(np.arange(1, len(heights)), size=3, replace=False))
    placed_faces: list[dict[str, Any]] = []
    for idx, (target_height, size_class) in enumerate(heights):
        pose = ["front", "front", "front", "three_quarter", "three_quarter", "profile"][int(rng.integers(6))]
        (r,) = ctx.faces.take(pose, 1)
        crop, mask, elems, (ecx, ecy, erx, ery) = _prepare_head(r)
        bx0, by0, bx1, by1 = bounding_box(elems[0].polygon)
        s = target_height / (by1 - by0)
        new_w = max(1, int(round(crop.width * s)))
        new_h = max(1, int(round(crop.height * new_w / crop.width)))
        sx, sy = new_w / crop.width, new_h / crop.height
        fw, fh = (bx1 - bx0) * sx, (by1 - by0) * sy
        placed = None
        for attempt in range(400):
            if idx in force and placed_faces and attempt < 250:
                # next to a face already pasted, with the new head pushed slightly into its box
                ref = placed_faces[int(rng.integers(len(placed_faces)))]
                rx0, ry0, rx1, ry1 = ref["box"]
                side = 1 if rng.random() < 0.5 else -1
                head_half_width = erx * sx
                intrusion = (rx1 - rx0) * rng.uniform(0.04, 0.14)
                fcx = (rx0 + rx1) / 2 + side * ((rx1 - rx0) / 2 + head_half_width - intrusion)
                fcy = (ry0 + ry1) / 2 + rng.uniform(-0.1, 0.35) * (ry1 - ry0)
            else:
                band = 0.25 + 0.45 * min(1.0, target_height / 200)  # large faces go lower
                fcx = rng.uniform(fw / 2 + 4, width - fw / 2 - 4)
                fcy = rng.uniform(height * (band - 0.18), height * (band + 0.12))
            x = int(round(fcx - (bx0 + bx1) / 2 * sx))
            y = int(round(fcy - (by0 + by1) / 2 * sy))
            box = (x + bx0 * sx, y + by0 * sy, x + bx1 * sx, y + by1 * sy)
            if box[0] < 3 or box[1] < 3 or box[2] > width - 3 or box[3] > height - 3:
                continue
            ccx = (box[0] + box[2]) / 2
            occluders = [
                (x + ecx * sx, y + ecy * sy, erx * sx, ery * sy),  # head
                (ccx, box[3] + 0.15 * fh, 0.2 * fw, 0.3 * fh),  # neck
                (ccx, box[3] + 0.95 * fh, 1.15 * fw, 0.85 * fh),  # torso
            ]
            valid, touches = True, False
            for c in placed_faces:
                if _inside_ellipses(c["core_mesh"], occluders).any():
                    valid = False
                    break
                frac = float(_inside_ellipses(c["box_mesh"], occluders).mean())
                if frac > 0.3:
                    valid = False
                    break
                touches |= frac >= 0.03
            if not valid or (idx in force and attempt < 250 and not touches):
                continue
            placed = (x, y, box, occluders)
            break
        if placed is None:
            continue
        x, y, box, occluders = placed
        skin = _skin_color(r.img, r.core)
        clothes = CLOTHES[int(rng.integers(len(CLOTHES)))]
        for (ox, oy, orx, ory), color in zip(
            occluders[2:0:-1], (clothes, tuple(int(c * 0.9) for c in skin)), strict=True
        ):
            d.ellipse((ox - orx, oy - ory, ox + orx, oy + ory), fill=color)
        (e,) = cv.paste(crop, x, y, width=new_w, elements=elems, mask=mask)
        final_height = _own_height(e.polygon)
        e.level = "stress" if final_height < MIN_BASE_GROUP_HEIGHT or pose == "profile" else "base"
        e.tags.update({"size_class": size_class, "layer_order": idx, "forced_overlap": idx in force})
        ex0, ey0, ex1, ey1 = bounding_box(e.polygon)
        nx0, ny0, nx1, ny1 = bounding_box(e.core)
        placed_faces.append(
            {
                "element": e,
                "box": (ex0, ey0, ex1, ey1),
                "box_mesh": _mesh((ex0, ey0, ex1, ey1)),
                "core_mesh": _mesh((nx0, ny0, nx1, ny1), margin=min(3.0, 0.05 * (ex1 - ex0))),
                "occluders": occluders,
            }
        )
    # fraction of each box covered by the people pasted afterwards
    for i, c in enumerate(placed_faces):
        later = [o for c2 in placed_faces[i + 1 :] for o in c2["occluders"]]
        frac = float(_inside_ellipses(c["box_mesh"], later).mean()) if later else 0.0
        c["element"].tags["occluded_fraction"] = round(frac, 3)
    return cv


def _groups(ctx: Context, rng: np.random.Generator) -> list[FileEntry]:
    out = []
    for i in (1, 2, 3):
        cv = _group(ctx, rng)
        desc = "Grupo compuesto: varias caras de 20 a 200 px pegadas sobre una sala, algunas solapadas."
        if i == 3:
            cv = cv.rotate(90)
            desc += " Imagen girada 90°."
        cv = Canvas(noise(cv.img, rng, 2.5), cv.elements)
        out.append(_save(ctx, rng, cv, f"grupo_compuesto_{i:02d}", desc, {"angle": 90 if i == 3 else 0}))
    return out


# ---------------------------------------------------------------------------
# 5. Real scenes
# ---------------------------------------------------------------------------


def _real_scenes(ctx: Context, rng: np.random.Generator) -> list[FileEntry]:
    out = []
    seen_ids: set[str] = set()
    for scene in ctx.faces.scenes():
        img = scene.img.convert("RGB")
        w, h = img.size
        poses = list(scene.tags.get("poses", []))  # optional: one pose per face, if the source annotates it
        elements = []
        for k, (box, core) in enumerate(scene.faces):
            box_c = [[min(max(float(px), 0.0), w), min(max(float(py), 0.0), h)] for px, py in box]
            core_c = (
                None
                if core is None
                else [[min(max(float(px), 0.0), w), min(max(float(py), 0.0), h)] for px, py in core]
            )
            elements.append(
                Element(
                    type="face",
                    page=0,
                    polygon=box_c,
                    core=core_c,
                    value=None,
                    level="base",
                    layer="raster",
                    tags={
                        "pose": poses[k] if k < len(poses) else "unannotated",
                        "face_source": scene.source,
                        "scene_id": scene.id,
                        "angle": 0,
                        "no_core": core is None,
                    },
                )
            )
        cv = Canvas(img, elements)
        if max(w, h) > MAX_SCENE_SIDE:
            cv = cv.scale(MAX_SCENE_SIDE / max(w, h))
        for e in cv.elements:
            e.level = "base" if _face_height(e) >= MIN_BASE_SCENE_HEIGHT else "stress"
        name_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(scene.id))
        while name_id in seen_ids:
            name_id += "_b"
        seen_ids.add(name_id)
        out.append(
            _save(
                ctx,
                rng,
                cv,
                f"escena_real_{name_id}",
                f"Foto real con {len(elements)} caras anotadas (fuente {scene.source}).",
                {"scene_id": scene.id, "scene_source": scene.source},
            )
        )
    return out


# ---------------------------------------------------------------------------
# 6. Photographed printed poster
# ---------------------------------------------------------------------------


def _write_parts(
    cv: Canvas,
    x: float,
    y: float,
    parts: list[tuple[str, str, str, dict[str, Any]]],
    **kw: Any,
) -> list[Element]:
    """Like ``Canvas.write_line`` but with its own level and tags per segment: ``(text, type, level, tags)``."""
    out = []
    font_name, size = kw.get("font_name", "sans"), kw.get("size", 24)
    for text, type, level, tags in parts:
        clean = text.strip()
        if clean:
            indent = cv.text_width(text[: len(text) - len(text.lstrip())], font_name, size)
            e = cv.write_text(x + indent, y, clean, type=type, level=level, tags=tags, **kw)
            if e is not None:
                out.append(e)
        x += cv.text_width(text, font_name, size)
    return out


def _name(p: Person) -> tuple[str, str, str, dict[str, Any]]:
    level = "base" if p.in_list else "out_of_scope"
    return (p.full_name, "name", level, {"in_list": p.in_list, "variant": "full"})


def _poster(ctx: Context, rng: np.random.Generator, f: FakeData) -> FileEntry:
    W, H = 900, 1272
    cv = Canvas.new(W, H, (250, 248, 242))
    d = ImageDraw.Draw(cv.img)
    color = POSTER_COLORS[int(rng.integers(len(POSTER_COLORS)))]
    d.rectangle((0, 0, W, 175), fill=color)
    white = (255, 255, 255)
    cv.write_text(50, 88, "TALLER COMUNITARIO", font_name="sans_bold", size=58, color=white)
    cv.write_text(50, 145, "Participación ciudadana en el plan comunal", size=30, color=white)
    _write_parts(cv, 50, 232, [("Fecha: ", "text", "base", {}), (f.date(), "text", "base", {"decoy": "date"})], size=30)
    # photo of the coordinator
    (r,) = ctx.faces.take("front", 1)
    photo_width = int(min(420, 520 * r.img.width / r.img.height))
    photo_height = int(round(r.img.height * photo_width / r.img.width))
    fx, fy = (W - photo_width) // 2, 272
    d.rectangle(
        (fx - 10, fy - 10, fx + photo_width + 10, fy + photo_height + 10), fill=(255, 255, 255), outline=(170, 170, 170)
    )
    cv.paste(
        r.img.convert("RGB"),
        fx,
        fy,
        width=photo_width,
        elements=FaceProvider.elements(r, "base", {"angle": 0, "origin": "poster"}),
    )
    y = fy + photo_height + 62
    coordinator = f.person(in_list=True)
    assistant = f.person(in_list=False)
    _write_parts(cv, 50, y, [("Coordina: ", "text", "base", {}), _name(coordinator)], size=30)
    y += 46
    _write_parts(cv, 50, y, [("Apoyo: ", "text", "base", {}), _name(assistant)], size=26)
    y += 52
    for phrase in POSTER_PHRASES:
        cv.write_text(50, y, phrase, size=24, color=(60, 60, 60))
        y += 38
    # contact line: email + phone
    email_fmt = ["dot", "initial", "underscore", "with_year"][int(rng.integers(4))]
    mobile_formats = [x for x in PHONE_FORMATS_BY_KIND["mobile"] if PHONE_FORMATS[x] == "base"]
    tel_fmt = mobile_formats[int(rng.integers(len(mobile_formats)))]
    email, phone = coordinator.email(email_fmt), coordinator.phone.format(tel_fmt)
    parts = [
        ("Contacto: ", "text", "base", {}),
        (email, "email", EMAIL_FORMATS[email_fmt], {"format": email_fmt}),
        ("  ·  Fono: ", "text", "base", {}),
        (phone, "phone", PHONE_FORMATS[tel_fmt], {"format": tel_fmt, "kind": "mobile"}),
    ]
    size = 28
    while size > 18 and cv.text_width("".join(p[0] for p in parts), "sans_bold", size) > W - 100:
        size -= 1
    y += 26
    _write_parts(cv, 50, y, parts, font_name="sans_bold", size=size, color=(20, 20, 20))
    d.rectangle((0, H - 70, W, H), fill=color)
    cv.write_text(50, H - 26, "Programa de Fomento Comunitario", size=26, color=white)

    # photo of the poster on a table, in perspective
    background = table_texture(1600, 1200, rng)
    j = rng.uniform(-28, 28, (4, 2))
    dest = [
        [430 + j[0, 0], 70 + j[0, 1]],
        [1170 + j[1, 0], 115 + j[1, 1]],
        [1225 + j[2, 0], 1125 + j[2, 1]],
        [385 + j[3, 0], 1085 + j[3, 1]],
    ]
    photo = cv.perspective(dest, background)
    img = lighting(photo.img, rng, 0.3)
    img = noise(img, rng, 3.0)
    photo = Canvas(img, photo.elements)
    return _save(
        ctx,
        rng,
        photo,
        "afiche_impreso",
        "Afiche impreso con foto, nombres y línea de contacto, fotografiado en perspectiva sobre una mesa.",
        {"perspective": True},
    )


# ---------------------------------------------------------------------------
# 7. Variants: black and white, low resolution, occlusion, mirrored
# ---------------------------------------------------------------------------


def _low_resolution(r: Face, rng: np.random.Generator) -> Canvas:
    cv = _portrait(r, "stress", {"degradation": "low_resolution"})
    x0, y0, x1, y1 = bounding_box(r.box)
    small = cv.scale(30.0 / (y1 - y0))
    w, h = small.width, small.height
    img = small.img.resize((w * 4, h * 4), Image.Resampling.BILINEAR)
    img = blur(img, 1.2)
    elements = transform_elements(small.elements, scale_matrix(4.0))
    for e in elements:
        e.tags["source_height_px"] = round(_own_height(small.elements[0].polygon), 1)
    return Canvas(img, elements)


def _occlusion(r: Face, rng: np.random.Generator) -> Canvas:
    cv = _fit_side(_portrait(r, "stress", {"occlusion": ["dark_glasses", "hand"]}), rng)
    e = cv.elements[0]
    d = ImageDraw.Draw(cv.img)
    nx0, ny0, nx1, ny1 = bounding_box(e.core)
    nw, nh = nx1 - nx0, ny1 - ny0
    s = cv.width / r.img.width
    if "eyes" in r.tags and len(r.tags["eyes"]) == 2:
        eyes = [(float(px) * s, float(py) * s) for px, py in r.tags["eyes"]]
    else:
        eyes = [(nx0 + 0.25 * nw, ny0 + 0.14 * nh), (nx0 + 0.75 * nw, ny0 + 0.14 * nh)]
    rx, ry = 0.21 * nw, 0.11 * nh
    for ox, oy in eyes:
        d.ellipse((ox - rx, oy - ry, ox + rx, oy + ry), fill=(18, 18, 22))
        d.ellipse((ox - rx * 0.55, oy - ry * 0.6, ox - rx * 0.1, oy - ry * 0.25), fill=(70, 70, 80))
    (a, b) = eyes
    d.line((a[0] + rx * 0.9, a[1], b[0] - rx * 0.9, b[1]), fill=(18, 18, 22), width=max(2, int(ry * 0.3)))
    # hand: palm and fingers over the cheek and part of the mouth
    cx0, cy0, cx1, cy1 = bounding_box(e.polygon)
    cw, ch = cx1 - cx0, cy1 - cy0
    skin = _skin_color(cv.img, e.core)
    tone = tuple(min(255, int(c * 0.92 + 12)) for c in skin)
    shadow = tuple(int(c * 0.75) for c in skin)
    hand = Image.new("L", cv.img.size, 0)
    dm = ImageDraw.Draw(hand)
    px, py = cx0 + 0.86 * cw, cy0 + 0.9 * ch
    dm.ellipse((px - 0.22 * cw, py - 0.17 * ch, px + 0.22 * cw, py + 0.17 * ch), fill=255)
    for k in range(4):
        ang = math.radians(215 + 14 * k + rng.uniform(-3, 3))
        length = 0.42 * ch * (1.0 - 0.12 * abs(k - 1.5))
        fx, fy = px + math.cos(ang) * length * 0.5, py + math.sin(ang) * length
        dm.line((px, py - 0.05 * ch, fx, fy), fill=255, width=max(3, int(0.09 * cw)))
        dm.ellipse((fx - 0.045 * cw, fy - 0.045 * cw, fx + 0.045 * cw, fy + 0.045 * cw), fill=255)
    hand = hand.filter(ImageFilter.GaussianBlur(1.2))
    layer = Image.new("RGB", cv.img.size, tone)
    edge = Image.new("RGB", cv.img.size, shadow)
    cv.img.paste(edge, (0, 0), hand.filter(ImageFilter.MaxFilter(5)))
    cv.img.paste(layer, (0, 0), hand)
    return cv


def _variants(ctx: Context, rng: np.random.Generator) -> list[FileEntry]:
    out = []
    (r,) = ctx.faces.take("front", 1)
    cv = _fit_side(_portrait(r, "base", {"color": "gray"}), rng)
    cv = Canvas(cv.img.convert("L").convert("RGB"), cv.elements)
    out.append(_save(ctx, rng, cv, "blanco_y_negro", "Retrato en escala de grises (JPEG de un canal).", gray=True))
    (r,) = ctx.faces.take("front", 1)
    out.append(
        _save(
            ctx,
            rng,
            _low_resolution(r, rng),
            "baja_resolucion",
            "Cara reducida a ~30 px de alto y ampliada 4x (borrosa).",
        )
    )
    (r,) = ctx.faces.take("front", 1)
    out.append(
        _save(
            ctx, rng, _occlusion(r, rng), "oclusion", "Retrato con lentes oscuros y una mano tapando parte de la cara."
        )
    )
    (r,) = ctx.faces.take("three_quarter", 1)
    cv = _fit_side(_portrait(r, "base"), rng).mirror()
    out.append(_save(ctx, rng, cv, "espejado", "Retrato en tres cuartos espejado horizontalmente."))
    return out
