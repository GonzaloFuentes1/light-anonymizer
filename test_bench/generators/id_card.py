"""FICTITIOUS identity cards: front, back, photos with perspective, glare and a PDF copy.

The field layout resembles the Chilean identity card so that OCR faces something realistic,
but the card carries no emblems, coats of arms, institutional logos or real security features,
and it has a visible stripe "DOCUMENTO FICTICIO - SOLO PRUEBAS".

ID-1 size (85.6 x 54 mm) at 12 px/mm: 1027 x 648 px.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from test_bench.canvas import (
    Canvas,
    blur,
    bounding_box,
    font_path,
    lighting,
    map_to_rect,
    rect,
    table_texture,
    transform_elements,
)
from test_bench.context import Context, image_file_entry, save_image
from test_bench.faces import FaceProvider
from test_bench.fake_data import RUT_FORMATS, FakeData, Person, strip_accents
from test_bench.schema import Element, FileEntry, Page

SEED_NAME = "cedula"  # seed string: kept in Spanish so the generated files stay identical
CATEGORY = "id_card"

PX_PER_MM = 12
WIDTH = round(85.6 * PX_PER_MM)  # 1027
HEIGHT = round(54 * PX_PER_MM)  # 648

INK = (22, 30, 45)
LABEL_INK = (70, 85, 105)
STRIPE_RED = (185, 28, 38)
STRIPE_TEXT = "DOCUMENTO FICTICIO - SOLO PRUEBAS"
MONTHS = ("ENE", "FEB", "MAR", "ABR", "MAY", "JUN", "JUL", "AGO", "SEP", "OCT", "NOV", "DIC")
PROFESSIONS = (
    "INGENIERÍA COMERCIAL",
    "PROFESORA",
    "TÉCNICO EN ENFERMERÍA",
    "ARQUITECTO",
    "TRABAJADORA SOCIAL",
    "CONTADOR AUDITOR",
    "ADMINISTRATIVA",
    "SIN PROFESIÓN",
)


# ---------------------------------------------------------------------------
# Card data
# ---------------------------------------------------------------------------


@dataclass
class IdCardData:
    person: Person
    number: str  # 9 digits without dots
    issue_date: tuple[int, int, int]  # (day, month, year)
    expiry_date: tuple[int, int, int]
    profession: str
    birth_place: str

    @property
    def formatted_number(self) -> str:
        n = self.number
        return f"{n[:3]}.{n[3:6]}.{n[6:]}"

    @property
    def birth(self) -> tuple[int, int, int]:
        d, m, y = (int(x) for x in self.person.birth_date.split("-"))
        return d, m, y

    @property
    def qr_url(self) -> str:
        p = self.person
        return f"https://portal.ejemplo.cl/docstatus?RUN={p.rut_body}-{p.rut_dv}&type=CEDULA&serial={self.number}"


def _card_date(f: tuple[int, int, int]) -> str:
    return f"{f[0]:02d} {MONTHS[f[1] - 1]} {f[2]}"


def _card_data(
    f: FakeData,
    rng: np.random.Generator,
    used: set[str],
    *,
    in_list: bool = True,
    dv_valid: bool = True,
    seven_digits: bool = False,
    with_k: bool = False,
) -> IdCardData:
    p = f.person(dv_valid=dv_valid, in_list=in_list, seven_digits=seven_digits)
    if with_k:
        p.rut_body, p.rut_dv = f.rut_with_k()
        p.rut_dv_valid = True
    while True:
        number = str(int(rng.integers(100_000_000, 600_000_000)))
        if number not in used:
            used.add(number)
            break
    year = int(rng.integers(2019, 2026))
    issue_date = (int(rng.integers(1, 29)), int(rng.integers(1, 13)), year)
    expiry_date = (issue_date[0], issue_date[1], year + 10)
    return IdCardData(
        person=p,
        number=number,
        issue_date=issue_date,
        expiry_date=expiry_date,
        profession=str(rng.choice(PROFESSIONS)),
        birth_place=p.commune.upper(),
    )


def _name_level(p: Person) -> str:
    return "base" if p.in_list else "out_of_scope"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _card_background(rng: np.random.Generator, tone: tuple[int, int, int]) -> Image.Image:
    """Light background with a generic pattern of interlaced sine lines (guilloche-like)."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    grad = (xx / WIDTH) * 10 - (yy / HEIGHT) * 6
    base = np.array(tone, np.float32)[None, None, :] + grad[..., None]
    img = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    xs = np.arange(0, WIDTH + 4, 3, dtype=np.float64)
    phase = float(rng.uniform(0, 2 * math.pi))
    for i in range(34):
        amp = 18 + 6 * math.sin(i * 0.7)
        y0 = -40 + i * 21
        ys = y0 + amp * np.sin(xs / 57.0 + phase + i * 0.35) + 9 * np.sin(xs / 13.0 + i)
        color = (200, 214, 222) if i % 2 == 0 else (214, 204, 222)
        d.line(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=color, width=1)
    # generic rosette (rose curve, not an emblem)
    cx, cy = WIDTH * 0.78, HEIGHT * 0.46
    t = np.linspace(0, 2 * math.pi, 1400)
    for k, r0 in ((7, 95), (9, 70), (11, 45)):
        r = r0 + 22 * np.sin(k * t)
        pts = list(zip((cx + r * np.cos(t)).tolist(), (cy + r * np.sin(t)).tolist(), strict=True))
        d.line(pts, fill=(205, 212, 228), width=1)
    return img


def _stripe(cv: Canvas, y0: int, y1: int) -> None:
    ImageDraw.Draw(cv.img).rectangle((0, y0, WIDTH, y1), fill=STRIPE_RED)
    size = 24
    w = cv.text_width(STRIPE_TEXT, "sans_bold", size)
    baseline = (y0 + y1) / 2 + size * 0.36
    cv.write_text((WIDTH - w) / 2, baseline, STRIPE_TEXT, font_name="sans_bold", size=size, color=(255, 255, 255))


def _label(cv: Canvas, x: float, y: float, text: str) -> None:
    cv.write_text(x, y, text, font_name="sans_bold", size=14, color=LABEL_INK)


def _sensitive_value(cv: Canvas, x: float, y: float, text: str, field: str, size: int = 22) -> None:
    cv.write_text(x, y, text, font_name="sans", size=size, color=INK, tags={"sensitive": True, "field": field})


def _paste_translucent(cv: Canvas, img: Image.Image, x: int, y: int, width: int, alpha: float) -> tuple:
    """Pastes a gray, translucent copy ('ghost' photo). Returns (sx, sy) of the scale used."""
    height = max(1, round(img.height * width / img.width))
    gray = img.convert("L").resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
    mask = Image.new("L", (width, height), int(round(255 * alpha)))
    cv.img.paste(gray, (x, y), mask)
    return width / img.width, height / img.height


def draw_front(data: IdCardData, faces: FaceProvider, rng: np.random.Generator, data_level: str = "base") -> Canvas:
    """Front of the fictitious card. ``data_level`` sets the level of the RUT and of the main face."""
    p = data.person
    cv = Canvas(_card_background(rng, (228, 238, 236)))
    d = ImageDraw.Draw(cv.img)

    # Header
    cv.write_text(40, 52, "CÉDULA DE PRUEBA", font_name="sans_bold", size=34, color=(20, 60, 100))
    cv.write_text(
        40, 80, "IDENTIFICACIÓN FICTICIA PARA PRUEBAS DE SOFTWARE", font_name="sans", size=15, color=(40, 70, 100)
    )
    d.line((40, 94, WIDTH - 40, 94), fill=(20, 60, 100), width=2)

    # Main photo
    (face,) = faces.take("front", 1)
    photo_width = 240
    photo_height = round(face.img.height * photo_width / face.img.width)
    d.rectangle((36, 108, 36 + photo_width + 8, 112 + photo_height + 4), fill=(255, 255, 255))
    cv.paste(
        face.img,
        40,
        112,
        width=photo_width,
        elements=FaceProvider.elements(face, data_level, {"origin": "id_card"}),
    )

    # Fields
    x1, x2 = 310, 575
    name_tags = {"in_list": p.in_list}
    _label(cv, x1, 126, "APELLIDOS")
    for y, surname, variant in (
        (160, p.paternal_surname, "paternal_surname"),
        (194, p.maternal_surname, "maternal_surname"),
    ):
        cv.write_text(
            x1,
            y,
            surname,
            type="name",
            font_name="sans_bold",
            size=28,
            color=INK,
            level=_name_level(p),
            tags={**name_tags, "variant": variant},
        )
    _label(cv, x1, 228, "NOMBRES")
    cv.write_text(
        x1,
        262,
        p.given_names,
        type="name",
        font_name="sans_bold",
        size=28,
        color=INK,
        level=_name_level(p),
        tags={**name_tags, "variant": "given_names"},
    )
    _label(cv, x1, 296, "NACIONALIDAD")
    cv.write_text(x1, 324, "CHILENA", size=22, color=INK)
    _label(cv, x2, 296, "SEXO")
    _sensitive_value(cv, x2, 324, p.sex, "sex")
    _label(cv, x1, 358, "FECHA DE NACIMIENTO")
    _sensitive_value(cv, x1, 386, _card_date(data.birth), "birth_date")
    _label(cv, x2, 358, "NÚMERO DOCUMENTO")
    _sensitive_value(cv, x2, 386, data.formatted_number, "document_number")
    _label(cv, x1, 420, "FECHA DE EMISIÓN")
    _sensitive_value(cv, x1, 448, _card_date(data.issue_date), "issue_date")
    _label(cv, x2, 420, "FECHA DE VENCIMIENTO")
    cv.write_text(
        x2,
        448,
        _card_date(data.expiry_date),
        size=22,
        color=INK,
        tags={"decoy": "date", "field": "expiry_date"},
    )

    # Large RUN under the photo
    rut = p.rut("dots")
    x_run = 40
    cv.write_text(x_run, 496, "RUN", font_name="sans_bold", size=36, color=INK)
    x_run += cv.text_width("RUN ", "sans_bold", 36)
    cv.write_text(
        x_run,
        496,
        rut,
        type="rut",
        font_name="sans_bold",
        size=36,
        color=INK,
        level=data_level if data_level != "base" else RUT_FORMATS["dots"],
        tags={"format": "dots", "dv_valid": p.rut_dv_valid},
    )

    # Signature (simulated handwriting) with a final stroke
    signature = f"{p.given_names.split()[0][0]}. {p.paternal_surname}"
    e_signature = cv.write_irregular(
        x2 + 10,
        528,
        signature,
        rng,
        type="signature",
        font_name="stix_italic",
        size=40,
        color=(25, 35, 110),
        level="out_of_scope",
        tags={"field": "signature"},
    )
    if e_signature is not None:
        bx0, by0, bx1, by1 = bounding_box(e_signature.polygon)
        xs = np.linspace(bx0 - 6, bx1 + 18, 60)
        ys = by1 + 2 + 5 * np.sin((xs - bx0) / 18.0)
        d.line(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=(25, 35, 110), width=2)
        tr = [(float(xs.min()), float(ys.min())), (float(xs.max()) + 1, float(ys.max()) + 1)]
        e_signature.polygon = rect(
            math.floor(min(bx0, tr[0][0]) - 1),
            math.floor(min(by0, tr[0][1]) - 1),
            math.ceil(max(bx1, tr[1][0]) + 1),
            math.ceil(max(by1, tr[1][1]) + 1),
        )
    cv.write_text(x2 + 10, 578, "FIRMA DEL TITULAR", font_name="sans", size=12, color=LABEL_INK)

    # Ghost photo (small translucent copy)
    fx, fy, fwidth = 880, 128, 108
    sx, sy = _paste_translucent(cv, face.img, fx, fy, fwidth, alpha=0.38)
    (ghost,) = FaceProvider.elements(face, "stress", {"ghost": True, "origin": "id_card"})
    m = np.array([[sx, 0, fx], [0, sy, fy], [0, 0, 1]], dtype=np.float64)
    cv.elements.extend(transform_elements([ghost], m))

    _stripe(cv, 600, HEIGHT)
    return cv


# ---------------------------------------------------------------------------
# Back: machine readable zone (MRZ) and QR code
# ---------------------------------------------------------------------------


def _mrz_check_digit(s: str) -> str:
    weights = (7, 3, 1)
    total = 0
    for i, c in enumerate(s):
        v = int(c) if c.isdigit() else (ord(c) - 55 if c.isalpha() else 0)
        total += v * weights[i % 3]
    return str(total % 10)


def _mrz_name(p: Person) -> str:
    def clean(t: str) -> str:
        return strip_accents(t).upper().replace("Ñ", "N").replace(" ", "<")

    s = f"{clean(p.paternal_surname)}<{clean(p.maternal_surname)}<<{clean(p.given_names)}"
    return (s + "<" * 30)[:30]


def mrz_lines(data: IdCardData) -> tuple[str, str, str, str, str]:
    """Returns (line1_prefix, mrz_run, line1_suffix, line2, line3) in 30-character TD1 format."""
    p = data.person
    prefix = f"INCHL{data.number}{_mrz_check_digit(data.number)}"
    run = f"{p.rut_body}<{p.rut_dv}"
    suffix = "<" * (30 - len(prefix) - len(run))
    db, mb, yb = data.birth
    de, me, ye = data.expiry_date
    birth = f"{yb % 100:02d}{mb:02d}{db:02d}"
    expiry = f"{ye % 100:02d}{me:02d}{de:02d}"
    body2 = f"{birth}{_mrz_check_digit(birth)}{p.sex}{expiry}{_mrz_check_digit(expiry)}CHL"
    optional2 = "<" * (29 - len(body2))
    # TD1 composite check digit (ICAO 9303): line 1 from position 6, and from line 2 the
    # dates with their check digits and the optional field (without sex or nationality)
    composite = (
        (prefix + run + suffix)[5:] + birth + _mrz_check_digit(birth) + expiry + _mrz_check_digit(expiry) + optional2
    )
    line2 = body2 + optional2 + _mrz_check_digit(composite)
    return prefix, run, suffix, line2, _mrz_name(p)


def _qr(text: str, module_px: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """QR image (with quiet zone) and the box of the dark modules in image coordinates."""
    q = cv2.QRCodeEncoder.create().encode(text)
    q = np.pad(q, 2, constant_values=255)
    large = np.kron(q, np.ones((module_px, module_px), np.uint8))
    ys, xs = np.nonzero(large == 0)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return Image.fromarray(large).convert("RGB"), box


def draw_back(data: IdCardData, rng: np.random.Generator) -> Canvas:
    p = data.person
    cv = Canvas(_card_background(rng, (232, 236, 240)))
    d = ImageDraw.Draw(cv.img)

    _label(cv, 40, 48, "LUGAR DE NACIMIENTO")
    _sensitive_value(cv, 40, 80, data.birth_place, "birth_place", size=24)
    _label(cv, 40, 126, "PROFESIÓN")
    _sensitive_value(cv, 40, 158, data.profession, "profession", size=24)
    cv.write_text(40, 206, "Tarjeta generada para pruebas de software.", size=16, color=LABEL_INK)
    cv.write_text(40, 230, "No es válida como documento de identificación.", size=16, color=LABEL_INK)

    # QR with the verification URL (it contains the RUN)
    qr_img, (qx0, qy0, qx1, qy1) = _qr(data.qr_url, 5)
    px, py = WIDTH - 40 - qr_img.width, 22
    cv.img.paste(qr_img, (px, py))
    cv.elements.append(
        Element(
            type="qr",
            page=0,
            polygon=rect(px + qx0, py + qy0, px + qx1, py + qy1),
            value=data.qr_url,
            level="stress",
            layer="raster",
            tags={"contains": "rut", "module_px": 5},
        )
    )

    _stripe(cv, 292, 336)

    # MRZ on a white background
    d.rectangle((0, 380, WIDTH, 598), fill=(250, 250, 250))
    prefix, run, suffix, line2, line3 = mrz_lines(data)
    size, x0 = 44, 62
    mrz_tags = {"sensitive": True, "field": "mrz"}
    y = 446
    cv.write_text(x0, y, prefix, font_name="mono", size=size, color=INK, tags={**mrz_tags, "line": 1})
    x = x0 + cv.text_width(prefix, "mono", size)
    cv.write_text(
        x,
        y,
        run,
        type="rut",
        font_name="mono",
        size=size,
        color=INK,
        level="stress",
        tags={"format": "mrz", "dv_valid": p.rut_dv_valid, "line": 1},
    )
    x += cv.text_width(run, "mono", size)
    cv.write_text(x, y, suffix, font_name="mono", size=size, color=INK, tags={**mrz_tags, "line": 1})
    cv.write_text(x0, y + 62, line2, font_name="mono", size=size, color=INK, tags={**mrz_tags, "line": 2})
    # The third line is the name in MRZ form: it is personal data (hard to match with the list).
    cv.write_text(
        x0,
        y + 124,
        line3,
        type="name",
        font_name="mono",
        size=size,
        color=INK,
        level="stress" if p.in_list else "out_of_scope",
        tags={"in_list": p.in_list, "variant": "mrz", "format": "mrz", "line": 3},
    )
    return cv


# ---------------------------------------------------------------------------
# Photographic compositions
# ---------------------------------------------------------------------------


def _jitter(rng: np.random.Generator, quad: list[list[float]], px: float) -> list[list[float]]:
    return [[x + float(rng.uniform(-px, px)), y + float(rng.uniform(-px, px))] for x, y in quad]


def _rotated_quad(cx: float, cy: float, degrees: float, scale: float) -> list[list[float]]:
    """Card quadrilateral with foreshortening (shorter top edge) and a counterclockwise turn of ``degrees``."""
    w, h = WIDTH * scale / 2, HEIGHT * scale / 2
    local = [[-w * 0.93, -h], [w * 0.93, -h], [w, h], [-w, h]]
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    return [[cx + x * c + y * s, cy - x * s + y * c] for x, y in local]


def _photo(cv: Canvas, dest: list[list[float]], size: tuple[int, int], rng: np.random.Generator) -> Canvas:
    background = table_texture(size[0], size[1], rng)
    # soft shadow under the card
    shadow = Image.new("L", size, 0)
    ImageDraw.Draw(shadow).polygon([(x + 10, y + 14) for x, y in dest], fill=110)
    shadow = blur(shadow, 12)
    dark = Image.new("RGB", size, (30, 20, 12))
    background.paste(dark, (0, 0), shadow)
    photo = cv.perspective(dest, background)
    img = lighting(photo.img, rng, 0.3)
    photo.img = blur(img, 0.7)
    return photo


def _glare(img: Image.Image, center: tuple[float, float], radii: tuple[float, float], strength: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((xx - center[0]) / radii[0]) ** 2 + ((yy - center[1]) / radii[1]) ** 2)) * strength
    arr = arr + (255 - arr) * g[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _motion(img: Image.Image, length: int, degrees: float) -> Image.Image:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), degrees, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= k.sum()
    return Image.fromarray(cv2.filter2D(np.asarray(img), -1, k))


def _widen_for_motion(elements: list[Element], length: int, degrees: float) -> None:
    """Enlarges the text polygons so that they cover the trail of the motion blur.

    The blur drags the ink up to ``(length - 1) / 2`` px to each side in the direction
    ``degrees`` (counterclockwise). Each edge of the quadrilateral moves outwards by the
    projection of that drag on its normal; the resulting quadrilateral contains the Minkowski
    sum of the original with the drag segment. Faces do not change: their box is a region of
    the face, not an ink outline.
    """
    t = math.radians(degrees)
    half = (length - 1) / 2
    drag = np.array([math.cos(t), -math.sin(t)]) * half
    for e in elements:
        if e.type == "face" or e.polygon is None or len(e.polygon) != 4:
            continue
        p = np.asarray(e.polygon, np.float64)
        center = p.mean(axis=0)
        lines = []
        for i in range(4):
            a, b = p[i], p[(i + 1) % 4]
            n = np.array([b[1] - a[1], a[0] - b[0]])
            n /= np.linalg.norm(n)
            if np.dot(n, a - center) < 0:
                n = -n
            lines.append((n, float(np.dot(n, a)) + abs(float(np.dot(n, drag))) + 0.5))
        new = []
        for i in range(4):
            (n1, c1), (n2, c2) = lines[i - 1], lines[i]
            x, y = np.linalg.solve(np.array([n1, n2]), np.array([c1, c2]))
            new.append([round(float(x), 3), round(float(y), 3)])
        e.polygon = new


def _mark(elements: list[Element], **tags: Any) -> None:
    for e in elements:
        e.tags.update(tags)


def _to_stress(elements: list[Element]) -> None:
    for e in elements:
        if e.type != "text" and e.level == "base":
            e.level = "stress"


# ---------------------------------------------------------------------------
# PDF with both sides
# ---------------------------------------------------------------------------


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _pdf_both_sides(ctx: Context, path: str, front: Canvas, back: Canvas, data: IdCardData) -> FileEntry:
    p = data.person
    doc = pymupdf.open()
    page_width, page_height = 595.276, 841.89
    page = doc.new_page(width=page_width, height=page_height)
    page.insert_font(fontname="dejavu", fontfile=str(font_path("sans")))
    elements: list[Element] = []

    def text(x: float, y: float, t: str, size: float, type: str = "text", **kw: Any) -> None:
        page.insert_text((x, y), t, fontname="dejavu", fontsize=size, color=(0.1, 0.1, 0.1))
        quads = page.search_for(t, quads=True)
        assert len(quads) == 1, (t, quads)
        q = quads[0]
        pol = [[q.ul.x, q.ul.y], [q.ur.x, q.ur.y], [q.lr.x, q.lr.y], [q.ll.x, q.ll.y]]
        pol = [[round(float(a), 3), round(float(b), 3)] for a, b in pol]
        elements.append(Element(type=type, page=0, polygon=pol, value=t, layer="text", **kw))

    text(72, 72, "Copia simple de cédula de identidad (anverso y reverso)", 13)
    text(72, 92, "Documento ficticio generado para pruebas del anonimizador.", 10)
    img_width = 400.0
    img_height = img_width * HEIGHT / WIDTH
    x0 = (page_width - img_width) / 2
    boxes = [(x0, 120.0, x0 + img_width, 120.0 + img_height)]
    boxes.append((x0, boxes[0][3] + 36, x0 + img_width, boxes[0][3] + 36 + img_height))
    for cv, box in ((front, boxes[0]), (back, boxes[1])):
        page.insert_image(pymupdf.Rect(*box), stream=_png(cv.img))
        elements.extend(map_to_rect(cv.elements, cv.width, cv.height, box))
    y_holder = boxes[1][3] + 40
    text(72, y_holder, "Titular:", 11)
    x_name = 72 + pymupdf.Font(fontfile=str(font_path("sans"))).text_length("Titular: ", fontsize=11)
    text(
        x_name,
        y_holder,
        p.full_name,
        11,
        type="name",
        level=_name_level(p),
        tags={"in_list": p.in_list, "variant": "full"},
    )
    text(72, y_holder + 20, "Se extiende la presente copia para fines de prueba.", 10)
    doc.set_metadata({})
    doc.subset_fonts()
    dest = ctx.path(path)
    # no_new_id: no random /ID in the trailer, so the file is identical on every run
    doc.save(dest, garbage=3, deflate=True, no_new_id=True)
    doc.close()
    for e in elements:
        e.page = 0
    return FileEntry(
        id="ced_ambos_lados_pdf",
        path=path,
        format="pdf",
        category=CATEGORY,
        description="PDF con las imágenes del anverso y reverso de una cédula ficticia y el nombre del titular.",
        pages=[Page(index=0, width=page_width, height=page_height, unit="pt", rotation=0)],
        elements=elements,
        tags={"embedded_images": 2},
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate(ctx: Context) -> list[FileEntry]:
    f = ctx.fake_data(SEED_NAME)
    rng = ctx.rng(SEED_NAME)
    used: set[str] = set()
    files: list[FileEntry] = []

    def save(cv: Canvas, path: str, id: str, description: str, format: str, **tags: Any) -> None:
        options = {"quality": tags.pop("quality")} if "quality" in tags else {}
        save_image(cv.img, ctx.path(path), format, **options)
        files.append(
            image_file_entry(
                id=id,
                path=path,
                format=format,
                category=CATEGORY,
                description=description,
                canvas=cv,
                tags=tags,
            )
        )

    # 1. Flat front
    d1 = _card_data(f, rng, used)
    front = draw_front(d1, ctx.faces, rng)
    save(front, "id_card/cedula_frente_plana.png", "ced_frente_plana", "Anverso plano de cédula ficticia.", "png")

    # 2. Flat back (person not in the list)
    d2 = _card_data(f, rng, used, in_list=False)
    back = draw_back(d2, rng)
    save(back, "id_card/cedula_dorso_plana.png", "ced_dorso_plana", "Reverso plano: MRZ y QR con el RUN.", "png")

    # 3. Photo with perspective on a table
    d3 = _card_data(f, rng, used, seven_digits=True)
    cv3 = draw_front(d3, ctx.faces, rng)
    dest3 = _jitter(rng, [[250, 230], [1330, 290], [1380, 960], [210, 915]], 12)
    photo3 = _photo(cv3, dest3, (1600, 1200), rng)
    save(
        photo3,
        "id_card/cedula_frente_foto_perspectiva.jpg",
        "ced_frente_foto_perspectiva",
        "Foto del anverso sobre una mesa, con perspectiva, luz desigual y leve desenfoque (JPEG 80).",
        "jpg",
        quality=80,
        perspective=True,
    )

    # 4. Photo with perspective and a ~30 degree turn
    d4 = _card_data(f, rng, used, dv_valid=False)
    cv4 = draw_front(d4, ctx.faces, rng)
    degrees = 30 + float(rng.uniform(-2, 2))
    dest4 = _jitter(rng, _rotated_quad(850, 750, degrees, 1.09), 6)
    photo4 = _photo(cv4, dest4, (1700, 1500), rng)
    for e in photo4.elements:
        e.tags["angle"] = round(degrees, 2)
    save(
        photo4,
        "id_card/cedula_frente_foto_girada.jpg",
        "ced_frente_foto_girada",
        f"Foto del anverso girada {degrees:.0f} grados y con perspectiva (JPEG 80).",
        "jpg",
        quality=80,
        perspective=True,
        angle=round(degrees, 2),
    )

    # 5. Photo of the back
    d5 = _card_data(f, rng, used, with_k=True)
    cv5 = draw_back(d5, rng)
    dest5 = _jitter(rng, [[230, 280], [1360, 240], [1390, 930], [200, 960]], 12)
    photo5 = _photo(cv5, dest5, (1600, 1200), rng)
    save(
        photo5,
        "id_card/cedula_dorso_foto.jpg",
        "ced_dorso_foto",
        "Foto del reverso (MRZ y QR) sobre una mesa, con perspectiva (JPEG 80).",
        "jpg",
        quality=80,
        perspective=True,
    )

    # 6. PDF with both sides (same person on front and back)
    d6 = _card_data(f, rng, used)
    front6 = draw_front(d6, ctx.faces, rng)
    back6 = draw_back(d6, rng)
    files.append(_pdf_both_sides(ctx, "id_card/cedula_ambos_lados.pdf", front6, back6, d6))

    # 7. Photo with glare and motion blur (stress)
    d7 = _card_data(f, rng, used, in_list=False)
    cv7 = draw_front(d7, ctx.faces, rng, data_level="stress")
    dest7 = _jitter(rng, [[240, 260], [1340, 250], [1370, 950], [220, 930]], 10)
    photo7 = _photo(cv7, dest7, (1600, 1200), rng)
    names = [e for e in photo7.elements if e.type == "name"]
    cx = float(np.mean([pt[0] for e in names for pt in e.polygon]))
    cy = float(np.mean([pt[1] for e in names for pt in e.polygon]))
    motion_length, motion_degrees = 11, 8
    photo7.img = _motion(_glare(photo7.img, (cx - 60, cy + 40), (260, 150), 0.82), motion_length, motion_degrees)
    _widen_for_motion(photo7.elements, motion_length, motion_degrees)
    _to_stress(photo7.elements)
    _mark(photo7.elements, glare=True, motion_blur=motion_length)
    save(
        photo7,
        "id_card/cedula_con_reflejo.jpg",
        "ced_con_reflejo",
        "Foto del anverso con un reflejo fuerte sobre los nombres y desenfoque de movimiento.",
        "jpg",
        quality=85,
        perspective=True,
        glare=True,
    )
    return files
