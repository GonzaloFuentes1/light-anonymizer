"""Synthetic screenshots: email client, phone chat, spreadsheet, web form and dialog.

The interfaces are drawn with ``Canvas`` with a generic look (no real product brands).
Each file uses different people: the values are unique canaries across the whole set, which is
why the rescaled variants (``planilla_reducida``, ``correo_escritorio_hidpi``) repeat the layout
but with other data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from test_bench.canvas import Canvas, transform_elements
from test_bench.context import Context, image_file_entry, save_image
from test_bench.faces import Face, FaceProvider
from test_bench.fake_data import (
    ADMINISTRATIVE_PHRASES,
    EMAIL_FORMATS,
    PHONE_FORMATS,
    PHONE_FORMATS_BY_KIND,
    RUT_FORMATS,
    FakeData,
    Person,
    Phone,
)
from test_bench.schema import Element, FileEntry

SEED_NAME = "pantallazos"  # seed string: kept in Spanish so the generated files stay identical
CATEGORY = "screenshot"

Color = tuple[int, int, int]

BASE_EMAILS = [f for f, n in EMAIL_FORMATS.items() if n == "base"]
BASE_RUTS = [f for f, n in RUT_FORMATS.items() if n == "base"]


# ---------------------------------------------------------------------------
# Typed text segments (a line can mix neutral labels and data)
# ---------------------------------------------------------------------------


@dataclass
class Seg:
    text: str
    type: str = "text"
    level: str = "base"
    tags: dict[str, Any] = field(default_factory=dict)
    color: Color | None = None
    font: str | None = None


def t(text: str, **tags: Any) -> Seg:
    """Neutral text segment."""
    return Seg(text, tags=tags)


def name(p: Person, variant: str = "full", level: str | None = None) -> Seg:
    value = p.full_name if variant == "full" else f"{p.given_names} {p.paternal_surname}"
    return Seg(
        value,
        "name",
        level or ("base" if p.in_list else "out_of_scope"),
        {"in_list": p.in_list, "variant": variant},
    )


def address(p: Person) -> Seg:
    return Seg(p.address, "address", "base" if p.in_list else "out_of_scope", {"in_list": p.in_list})


def rut(p: Person, format: str = "dots") -> Seg:
    return Seg(p.rut(format), "rut", RUT_FORMATS[format], {"format": format, "dv_valid": p.rut_dv_valid})


def email(p: Person, format: str = "dot") -> Seg:
    return Seg(p.email(format), "email", EMAIL_FORMATS[format], {"format": format})


def phone(tel: Phone, format: str) -> Seg:
    return Seg(tel.format(format), "phone", PHONE_FORMATS[format], {"format": format, "kind": tel.kind})


def write_segs(
    cv: Canvas,
    x: float,
    y: float,
    segs: list[Seg],
    *,
    size: int,
    font_name: str = "sans",
    color: Color = (30, 30, 30),
) -> float:
    """Writes the segments one after another on baseline ``y``. Returns the final x."""
    for s in segs:
        f_s = s.font or font_name
        clean = s.text.strip()
        if clean:
            lead = len(s.text) - len(s.text.lstrip())
            x0 = x + cv.text_width(s.text[:lead], f_s, size)
            cv.write_text(
                x0,
                y,
                clean,
                type=s.type,
                font_name=f_s,
                size=size,
                color=s.color or color,
                level=s.level,
                tags=s.tags,
            )
        x += cv.text_width(s.text, f_s, size)
    return x


def segs_width(cv: Canvas, segs: list[Seg], size: int, font_name: str = "sans") -> float:
    return sum(cv.text_width(s.text, s.font or font_name, size) for s in segs)


def _fitting_size(cv: Canvas, segs: list[Seg], size: int, width: float, font_name: str = "sans") -> int:
    """Largest font size (up to ``size``) with which the segments fit in ``width``."""
    while size > 6 and segs_width(cv, segs, size, font_name) > width:
        size -= 1
    return size


def _to_stress(elements: list[Element]) -> None:
    for e in elements:
        if e.type != "text" and e.level == "base":
            e.level = "stress"


def _choose(rng: np.random.Generator, options: list[str]) -> str:
    return str(options[int(rng.integers(len(options)))])


def _mobile_format(rng: np.random.Generator) -> str:
    return _choose(rng, [f for f in PHONE_FORMATS_BY_KIND["mobile"] if PHONE_FORMATS[f] == "base"])


def _landline_format(rng: np.random.Generator, tel: Phone) -> str:
    return _choose(rng, [f for f in PHONE_FORMATS_BY_KIND[tel.kind] if PHONE_FORMATS[f] == "base"])


# ---------------------------------------------------------------------------
# Cropped faces for avatars
# ---------------------------------------------------------------------------


def _centered_crop(r: Face, margin: float = 1.5) -> tuple[Image.Image, list[Element]]:
    """Square crop centered on the face (with replicated borders if needed) and its element."""
    xs = [p[0] for p in r.box]
    ys = [p[1] for p in r.box]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = int(round(max(max(xs) - min(xs), max(ys) - min(ys)) * margin))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    arr = np.asarray(r.img.convert("RGB"))
    pad = max(0, -x0, -y0, x0 + side - arr.shape[1], y0 + side - arr.shape[0])
    if pad:
        arr = cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = Image.fromarray(arr[y0 + pad : y0 + pad + side, x0 + pad : x0 + pad + side].copy())
    m = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    return crop, transform_elements(FaceProvider.elements(r), m)


def _circular_mask(side: int, supersampling: int = 4) -> Image.Image:
    large = Image.new("L", (side * supersampling, side * supersampling), 0)
    ImageDraw.Draw(large).ellipse((0, 0, side * supersampling - 1, side * supersampling - 1), fill=255)
    return large.resize((side, side), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# 1. Desktop email client (scale 1.0 and 1.5)
# ---------------------------------------------------------------------------

SUBJECTS = [
    "Informe de avance mensual - septiembre",
    "Boleta de honorarios",
    "Reunión de coordinación del jueves",
    "Solicitud de antecedentes",
    "Rendición de gastos del taller",
    "Acta de la sesión de ayer",
    "Consulta sobre el convenio",
    "Certificado de recepción conforme",
    "Programa de talleres comunitarios",
    "Actualización de datos del proveedor",
]
HOURS = ["10:42", "09:15", "08:03", "Ayer", "Ayer", "lun", "dom", "vie", "jue", "mié"]


def _desktop_email(f: FakeData, rng: np.random.Generator, scale: float) -> Canvas:
    def e(v: float) -> float:
        return v * scale

    def z(v: float) -> int:
        return int(round(v * scale))

    W, H = z(1920), z(1080)
    cv = Canvas.new(W, H, (255, 255, 255))
    d = ImageDraw.Draw(cv.img)
    gray = (110, 115, 125)

    owner = f.person()
    senders = [f.person(in_list=(i not in (3, 7))) for i in range(10)]
    author = senders[0]
    cc = [f.person(), f.person(in_list=False)]

    # Title bar
    d.rectangle((0, 0, W, z(36)), fill=(45, 48, 56))
    write_segs(
        cv,
        e(16),
        e(24),
        [t("Bandeja de entrada - "), email(owner, "dot"), t(" - Cliente de correo")],
        size=z(13),
        color=(235, 235, 240),
    )
    for i, c in enumerate(((236, 95, 90), (245, 190, 80), (100, 200, 110))):
        cx = W - e(30 + 26 * i)
        d.ellipse((cx - e(7), e(11), cx + e(7), e(25)), fill=c)

    # Toolbar
    d.rectangle((0, z(36), W, z(84)), fill=(243, 244, 246))
    x = e(16)
    for text in ("Nuevo mensaje", "Responder", "Responder a todos", "Reenviar", "Archivar", "Eliminar"):
        w = cv.text_width(text, "sans", z(14)) + e(28)
        d.rounded_rectangle((x, e(46), x + w, e(74)), radius=z(5), fill=(255, 255, 255), outline=(205, 208, 214))
        cv.write_text(x + e(14), e(65), text, size=z(14), color=(40, 45, 55))
        x += w + e(10)
    d.rounded_rectangle(
        (W - e(440), e(46), W - e(20), e(74)), radius=z(5), fill=(255, 255, 255), outline=(205, 208, 214)
    )
    cv.write_text(W - e(424), e(65), "Buscar en el correo", size=z(14), color=(150, 150, 158))

    # Folder sidebar
    d.rectangle((0, z(84), z(250), H - z(28)), fill=(248, 249, 251))
    # the sidebar is 250 px wide: the owner's name and email shrink until they fit
    size_n = _fitting_size(cv, [name(owner)], z(14), e(222), "sans_bold")
    write_segs(cv, e(20), e(118), [name(owner)], size=size_n, font_name="sans_bold")
    size_c = _fitting_size(cv, [email(owner, "dot")], z(13), e(222))
    write_segs(cv, e(20), e(138), [email(owner, "dot")], size=size_c, color=gray)
    folders = [("Bandeja de entrada", "12"), ("Destacados", ""), ("Enviados", ""), ("Borradores", "2"),
               ("Archivados", ""), ("Spam", "5"), ("Papelera", "")]  # fmt: skip
    for i, (folder, n) in enumerate(folders):
        y = e(186 + 34 * i)
        if i == 0:
            d.rounded_rectangle((e(10), y - e(22), e(240), y + e(10)), radius=z(6), fill=(220, 232, 250))
        cv.write_text(e(28), y, folder, font_name="sans_bold" if i == 0 else "sans", size=z(14))
        if n:
            cv.write_text(e(222) - cv.text_width(n, "sans", z(13)), y, n, size=z(13), color=gray)

    # Message list
    x0l, x1l = z(250), z(800)
    d.line((x0l, z(84), x0l, H - z(28)), fill=(220, 222, 228), width=max(1, z(1)))
    d.line((x1l, z(84), x1l, H - z(28)), fill=(220, 222, 228), width=max(1, z(1)))
    cv.write_text(x0l + e(20), e(116), "Bandeja de entrada", font_name="sans_bold", size=z(15))
    cv.write_text(x1l - e(20) - cv.text_width("Ordenar: fecha", "sans", z(13)), e(116), "Ordenar: fecha",
                  size=z(13), color=gray)  # fmt: skip
    list_formats = [_choose(rng, BASE_EMAILS) for _ in senders]
    for i, (p, subject, hour) in enumerate(zip(senders, SUBJECTS, HOURS, strict=True)):
        y = e(132 + 80 * i)
        if i == 0:
            d.rectangle((x0l + 1, y, x1l - 1, y + e(80)), fill=(220, 232, 250))
        d.line((x0l, y + e(80), x1l, y + e(80)), fill=(232, 234, 238), width=max(1, z(1)))
        write_segs(cv, x0l + e(20), y + e(24), [name(p)], size=z(14), font_name="sans_bold")
        cv.write_text(x1l - e(20) - cv.text_width(hour, "sans", z(13)), y + e(24), hour, size=z(13), color=gray,
                      tags={"field": "time"})  # fmt: skip
        write_segs(cv, x0l + e(20), y + e(45), [email(p, list_formats[i])], size=z(13), color=gray)
        cv.write_text(x0l + e(20), y + e(66), subject, size=z(13), color=(50, 55, 65))

    # Open message
    xm = x1l + e(32)
    cv.write_text(xm, e(128), SUBJECTS[0], font_name="sans_bold", size=z(20))
    author_fmt = list_formats[0]
    y = e(170)
    write_segs(cv, xm, y, [t("De: "), name(author), t(" <"), email(author, author_fmt), t(">")], size=z(14))
    write_segs(cv, xm, y + e(24), [t("Para: "), name(owner), t(" <"), email(owner, "dot"), t(">")],
               size=z(14))  # fmt: skip
    cc_fmt = [_choose(rng, BASE_EMAILS) for _ in cc]
    cc_segs = [t("CC: "), name(cc[0]), t(" <"), email(cc[0], cc_fmt[0]), t(">, "), name(cc[1]), t(" <"),
               email(cc[1], cc_fmt[1]), t(">")]  # fmt: skip
    write_segs(cv, xm, y + e(48), cc_segs, size=_fitting_size(cv, cc_segs, z(14), W - xm - e(24)))
    write_segs(cv, xm, y + e(72), [t("Fecha: "), Seg(f.date(), tags={"decoy": "date"}), t(" 10:42")],
               size=z(14), color=gray)  # fmt: skip
    # Attachment
    ya = y + e(96)
    d.rounded_rectangle((xm, ya, xm + e(290), ya + e(40)), radius=z(6), fill=(246, 247, 249), outline=(210, 213, 220))
    d.rectangle((xm + e(10), ya + e(8), xm + e(32), ya + e(32)), fill=(200, 60, 60))
    cv.write_text(xm + e(42), ya + e(26), "informe_septiembre.pdf (245 KB)", size=z(13))
    d.line((xm, ya + e(58), W - e(32), ya + e(58)), fill=(225, 227, 232), width=max(1, z(1)))

    author_tel = author.phone
    tel_fmt = _mobile_format(rng)
    rut_fmt = _choose(rng, BASE_RUTS)
    phrases = [ADMINISTRATIVE_PHRASES[int(i)] for i in rng.choice(len(ADMINISTRATIVE_PHRASES), 2, replace=False)]
    lines: list[list[Seg]] = [
        [t("Estimado equipo:")],
        [],
        [t("Junto con saludar, adjunto el informe de avance correspondiente al mes de septiembre.")],
        [t(phrases[0])],
        [t(phrases[1])],
        [],
        [t("Para efectos del pago de la boleta, mis datos son los siguientes:")],
        [t("RUT: "), rut(author, rut_fmt)],
        [t("Teléfono: "), phone(author_tel, tel_fmt)],
        [t("Dirección: "), address(author), t(", "), t(author.commune, field="commune")],
        [t("El monto bruto de la boleta es de "), Seg(f.amount(), tags={"decoy": "amount"}), t(".")],
        [],
        [t("Quedo atento(a) a cualquier consulta."), ],
        [t("Saludos cordiales,")],
        [],
        [t("--")],
        [name(author)],
        [t("Profesional de apoyo, División de Planificación")],
        [t("Gobierno Regional Ficticio")],
        [t("Tel.: "), phone(author.landline, _landline_format(rng, author.landline))],
        [email(author, "uppercase" if author_fmt != "uppercase" else "dot")],
    ]  # fmt: skip
    y = ya + e(96)
    for i, segs in enumerate(lines):
        is_signature = i >= 16
        write_segs(
            cv,
            xm,
            y,
            segs,
            size=z(14 if is_signature else 15),
            font_name="sans_bold" if i == 16 else "sans",
            color=(70, 75, 85) if is_signature else (30, 30, 30),
        )
        y += e(22 if is_signature else 26)

    # Status bar
    d.rectangle((0, H - z(28), W, H), fill=(236, 238, 242))
    cv.write_text(e(16), H - e(9), "Conectado - 12 mensajes sin leer", size=z(12), color=gray)
    return cv


# ---------------------------------------------------------------------------
# 2. Phone chat
# ---------------------------------------------------------------------------


def _mobile_chat(f: FakeData, rng: np.random.Generator, faces: FaceProvider) -> Canvas:
    W, H = 1080, 2340
    cv = Canvas.new(W, H, (236, 229, 221))
    d = ImageDraw.Draw(cv.img)
    contact = f.person()
    me = f.person()
    partner = f.person(in_list=False)

    # Status bar and header
    d.rectangle((0, 0, W, 300), fill=(0, 105, 92))
    cv.write_text(48, 58, "10:42", font_name="sans_bold", size=34, color=(255, 255, 255),
                  tags={"field": "time"})  # fmt: skip
    cv.write_text(W - 48 - cv.text_width("5G 87%", "sans", 32), 58, "5G 87%", size=32, color=(255, 255, 255))
    d.line([(70, 200), (44, 176), (70, 152)], fill=(255, 255, 255), width=7)
    (r,) = faces.take("front", 1)
    crop, elems = _centered_crop(r, 1.5)
    side = 136
    cv.paste(crop, 96, 108, width=side, elements=elems, mask=_circular_mask(crop.width))
    cv.elements[-1].tags.update({"avatar": True, "mask": "circular"})
    write_segs(cv, 262, 176, [name(contact)], size=42, font_name="sans_bold", color=(255, 255, 255))
    cv.write_text(262, 226, "en línea", size=30, color=(210, 235, 230))

    # Date chip
    today_width = cv.text_width("HOY", "sans_bold", 28) + 48
    d.rounded_rectangle(((W - today_width) / 2, 340, (W + today_width) / 2, 392), radius=16, fill=(221, 236, 244))
    cv.write_text((W - today_width) / 2 + 24, 377, "HOY", font_name="sans_bold", size=28, color=(80, 90, 100))

    partner_rut_fmt = "no_dots"
    messages: list[tuple[bool, list[list[Seg]], str]] = [
        (False, [[t("Hola! Para el contrato de honorarios")], [t("necesito tus datos, porfa")]], "09:58"),
        (True, [[t("Hola! Claro. Mi RUT es "), rut(me, "dots")]], "10:01"),
        (True, [[t("Correo: "), email(me, "with_year")]], "10:02"),
        (False, [[t("¿Y un teléfono de contacto?")]], "10:03"),
        (True, [[t("Al celular: "), phone(me.phone, "mobile_international")]], "10:03"),
        (True, [[t("Vivo en")], [address(me)], [t(me.commune, field="commune")]], "10:04"),
        (False, [[t("¿Me pasas también el RUT de tu pareja")], [t("para la carga del seguro?")]], "10:05"),
        (True, [[t("Es "), rut(partner, partner_rut_fmt)], [name(partner)]], "10:06"),
        (False, [[t("Perfecto. Cualquier cosa llámame")],
                 [t("a la oficina: "), phone(contact.landline, _landline_format(rng, contact.landline))]],
         "10:07"),
    ]  # fmt: skip
    y = 430.0
    size, line_height, pad = 40, 54, 24
    for own, lines, hour in messages:
        sizes = []
        for segs in lines:
            w = segs_width(cv, segs, size)
            sizes.append(size if w <= 800 else int(size * 800 / w))
        widths = [segs_width(cv, segs, sl) for segs, sl in zip(lines, sizes, strict=True)]
        hour_width = cv.text_width(hour, "sans", 26)
        w = max(max(widths), widths[-1] + hour_width + 24) + 2 * pad
        h = len(lines) * line_height + 2 * pad + 22
        assert y + h < 2190, "the chat does not fit above the input bar"
        x0 = W - 40 - w if own else 40
        d.rounded_rectangle((x0, y, x0 + w, y + h), radius=26, fill=(217, 253, 211) if own else (255, 255, 255))
        for i, (segs, sl) in enumerate(zip(lines, sizes, strict=True)):
            write_segs(cv, x0 + pad, y + pad + 40 + i * line_height, segs, size=sl, color=(20, 20, 20))
        cv.write_text(x0 + w - pad - hour_width, y + h - 14, hour, size=26, color=(110, 120, 115),
                      tags={"field": "time"})  # fmt: skip
        y += h + 18

    # Input bar
    d.rectangle((0, 2190, W, H), fill=(236, 229, 221))
    d.rounded_rectangle((30, 2210, W - 170, 2320), radius=55, fill=(255, 255, 255))
    cv.write_text(90, 2280, "Mensaje", size=40, color=(140, 140, 140))
    d.ellipse((W - 150, 2210, W - 40, 2320), fill=(0, 140, 120))
    d.polygon([(W - 118, 2240), (W - 62, 2265), (W - 118, 2290), (W - 110, 2265)], fill=(255, 255, 255))
    return cv


# ---------------------------------------------------------------------------
# 3. Spreadsheet
# ---------------------------------------------------------------------------


def _spreadsheet(f: FakeData, rng: np.random.Generator) -> Canvas:
    W, H = 1600, 900
    cv = Canvas.new(W, H, (255, 255, 255))
    d = ImageDraw.Draw(cv.img)
    gray = (100, 105, 112)

    # Menus and formula bar
    d.rectangle((0, 0, W, 34), fill=(245, 246, 248))
    cv.write_text(14, 23, "honorarios_2026.xlsx", font_name="sans_bold", size=13)
    x = 200
    for m in ("Archivo", "Editar", "Ver", "Insertar", "Formato", "Datos", "Herramientas", "Ayuda"):
        cv.write_text(x, 23, m, size=13, color=(50, 55, 60))
        x += cv.text_width(m, "sans", 13) + 22
    d.rectangle((0, 34, W, 72), fill=(250, 250, 251))
    d.line((0, 72, W, 72), fill=(210, 212, 216))

    people = [f.person(in_list=(i not in (2, 6, 10))) for i in range(12)]
    rut_fmt = [_choose(rng, BASE_RUTS) if i != 9 else "no_hyphen" for i in range(12)]
    email_fmt = [_choose(rng, BASE_EMAILS) for _ in range(12)]
    tel_fmt = [_mobile_format(rng) if i % 4 else _landline_format(rng, p.landline) for i, p in enumerate(people)]

    d.rectangle((10, 42, 90, 64), fill=(255, 255, 255), outline=(200, 202, 206))
    cv.write_text(18, 58, "B5", size=13)
    cv.write_text(104, 58, "fx", font_name="serif_italic", size=13, color=gray)
    write_segs(cv, 130, 58, [rut(people[3], rut_fmt[3])], size=13)

    # Grid
    y0, row_height, row_num_width = 80, 30, 44
    cols = [("Nombre", 270), ("RUT", 130), ("Correo", 340), ("Teléfono", 170), ("Comuna", 140), ("Monto", 120)]
    cols += [("", 110)] * 5
    xs = [row_num_width]
    for _, w in cols:
        xs.append(xs[-1] + w)
    d.rectangle((0, y0, W, y0 + 26), fill=(240, 241, 243))
    d.rectangle((0, y0, row_num_width, H - 40), fill=(240, 241, 243))
    for j in range(len(cols)):
        letter = chr(ord("A") + j)
        cx = (xs[j] + xs[j + 1]) / 2
        if cx > W - 12:
            break
        cv.write_text(cx - cv.text_width(letter, "sans", 12) / 2, y0 + 18, letter, size=12, color=gray)
    y_rows = y0 + 26
    n_rows = (H - 40 - y_rows) // row_height
    for i in range(n_rows):
        yf = y_rows + i * row_height
        num = str(i + 1)
        cv.write_text(row_num_width / 2 - cv.text_width(num, "sans", 12) / 2, yf + 20, num, size=12, color=gray)
    for xv in xs:
        d.line((xv, y0, xv, H - 40), fill=(218, 220, 224))
    for i in range(n_rows + 1):
        d.line((0, y_rows + i * row_height, W, y_rows + i * row_height), fill=(218, 220, 224))
    d.line((0, y0 + 26, W, y0 + 26), fill=(200, 202, 206))
    # selected cell
    d.rectangle((xs[1], y_rows + 4 * row_height, xs[2], y_rows + 5 * row_height), outline=(30, 110, 220), width=2)

    # Headers and data
    d.rectangle((xs[0] + 1, y_rows + 1, xs[6] - 1, y_rows + row_height - 1), fill=(226, 239, 218))
    for j, (title, _) in enumerate(cols[:6]):
        cv.write_text(xs[j] + 8, y_rows + 20, title, font_name="sans_bold", size=13)
    for i, p in enumerate(people):
        yb = y_rows + (i + 1) * row_height + 20
        tel = p.phone if tel_fmt[i].startswith(("mobile", "old_mobile")) else p.landline
        cells = [
            name(p),
            rut(p, rut_fmt[i]),
            email(p, email_fmt[i]),
            phone(tel, tel_fmt[i]),
            t(p.commune, field="commune"),
        ]
        for j, s in enumerate(cells):
            # the cell does not overflow: the font shrinks (like a spreadsheet's text fitting)
            cell_size = _fitting_size(cv, [s], 13, cols[j][1] - 14)
            assert cell_size >= 10, s.text
            write_segs(cv, xs[j] + 8, yb, [s], size=cell_size)
        amount = f.amount()
        cv.write_text(xs[6] - 8 - cv.text_width(amount, "sans", 13), yb, amount, size=13, tags={"decoy": "amount"})

    # Sheet tabs
    d.rectangle((0, H - 40, W, H), fill=(245, 246, 248))
    for k, sheet in enumerate(("Honorarios", "Resumen", "Hoja3")):
        xh = 60 + k * 130
        if k == 0:
            d.rectangle((xh - 10, H - 40, xh + 110, H - 8), fill=(255, 255, 255), outline=(210, 212, 216))
        cv.write_text(xh, H - 18, sheet, font_name="sans_bold" if k == 0 else "sans", size=13)
    return cv


# ---------------------------------------------------------------------------
# 4. Web form
# ---------------------------------------------------------------------------


def _web_form(f: FakeData, rng: np.random.Generator, faces: FaceProvider) -> Canvas:
    W, H = 1366, 768
    cv = Canvas.new(W, H, (244, 246, 249))
    d = ImageDraw.Draw(cv.img)
    p = f.person()
    gray = (95, 100, 110)

    # Generic browser: tab and address bar (the URL carries the RUN as a parameter)
    d.rectangle((0, 0, W, 86), fill=(222, 225, 230))
    d.rounded_rectangle((12, 6, 300, 40), radius=8, fill=(255, 255, 255))
    cv.write_text(28, 29, "Mis datos - Portal de trámites", size=13)
    d.rounded_rectangle((110, 46, W - 110, 80), radius=17, fill=(255, 255, 255))
    for i, symbol in enumerate(("<", ">", "C")):
        cv.write_text(18 + 30 * i, 70, symbol, font_name="sans_bold", size=15, color=gray)
    url_prefix = "https://tramites.ejemplo.cl/perfil/editar?run="
    url_run = p.rut("no_dots")
    x_url = 130
    cv.write_text(
        x_url,
        69,
        url_prefix + url_run,
        type="url",
        size=14,
        color=(40, 40, 45),
        tags={"contains": "rut"},
    )
    cv.write_text(
        x_url + cv.text_width(url_prefix, "sans", 14),
        69,
        url_run,
        type="rut",
        size=14,
        color=(40, 40, 45),
        level="stress",
        tags={"format": "no_dots", "dv_valid": p.rut_dv_valid, "in_url": True},
    )

    # Site header
    d.rectangle((0, 86, W, 146), fill=(28, 58, 105))
    cv.write_text(40, 124, "Portal de Trámites (ficticio)", font_name="sans_bold", size=20, color=(255, 255, 255))
    for i, m in enumerate(("Inicio", "Mis solicitudes", "Mis datos", "Cerrar sesión")):
        cv.write_text(760 + 140 * i, 122, m, size=14, color=(220, 230, 245))

    # Profile card with photo
    d.rounded_rectangle((40, 176, 330, 520), radius=10, fill=(255, 255, 255), outline=(215, 218, 224))
    (r,) = faces.take("front", 1)
    ys = [q[1] for q in r.box]
    photo_width = int(round(r.img.width * 120 / (max(ys) - min(ys))))  # face ~120 px tall
    photo_height = round(r.img.height * photo_width / r.img.width)
    xf = 185 - photo_width // 2
    d.rectangle((xf - 4, 196, xf + photo_width + 4, 200 + photo_height + 4), fill=(225, 228, 234))
    cv.paste(r.img, xf, 200, width=photo_width, elements=FaceProvider.elements(r, "base", {"origin": "profile"}))
    yb = 200 + photo_height + 38
    w_name = segs_width(cv, [name(p)], 15, "sans_bold")
    name_size = 15 if w_name <= 270 else int(15 * 270 / w_name)
    write_segs(cv, 185 - segs_width(cv, [name(p)], name_size, "sans_bold") / 2, yb, [name(p)],
               size=name_size, font_name="sans_bold")  # fmt: skip
    cv.write_text(185 - cv.text_width("Cambiar foto", "sans", 13) / 2, yb + 28, "Cambiar foto", size=13,
                  color=(30, 100, 200))  # fmt: skip

    # Form
    d.rounded_rectangle((360, 176, 1326, 700), radius=10, fill=(255, 255, 255), outline=(215, 218, 224))
    cv.write_text(390, 218, "Mis datos personales", font_name="sans_bold", size=22)
    cv.write_text(390, 244, "Revise y actualice su información de contacto.", size=14, color=gray)

    def form_field(x: float, y: float, width: float, label: str, segs: list[Seg], arrow: bool = False) -> None:
        cv.write_text(x, y, label, font_name="sans_bold", size=13, color=(60, 65, 75))
        d.rounded_rectangle((x, y + 10, x + width, y + 50), radius=5, fill=(255, 255, 255), outline=(190, 195, 205))
        write_segs(cv, x + 12, y + 36, segs, size=15)
        if arrow:
            d.polygon([(x + width - 26, y + 26), (x + width - 14, y + 26), (x + width - 20, y + 34)], fill=gray)

    email_fmt = _choose(rng, BASE_EMAILS)
    tel_fmt = _mobile_format(rng)
    form_field(390, 290, 906, "Nombre completo", [name(p)])
    form_field(390, 372, 440, "RUT", [rut(p, "dots")])
    form_field(856, 372, 440, "Fecha de nacimiento", [t(p.birth_date, sensitive=True, field="birth_date")])
    form_field(390, 454, 440, "Correo electrónico", [email(p, email_fmt)])
    form_field(856, 454, 440, "Teléfono", [phone(p.phone, tel_fmt)])
    form_field(390, 536, 620, "Dirección", [address(p)])
    form_field(1036, 536, 260, "Comuna", [t(p.commune, field="commune")], arrow=True)
    d.rectangle((390, 612, 408, 630), fill=(30, 100, 200))
    d.line([(394, 621), (398, 626), (405, 615)], fill=(255, 255, 255), width=2)
    cv.write_text(418, 627, "Deseo recibir notificaciones por correo electrónico", size=14)
    d.rounded_rectangle((1086, 640, 1296, 684), radius=6, fill=(30, 100, 200))
    cv.write_text(1116, 668, "Guardar cambios", font_name="sans_bold", size=15, color=(255, 255, 255))
    d.rounded_rectangle((960, 640, 1070, 684), radius=6, fill=(255, 255, 255), outline=(190, 195, 205))
    cv.write_text(982, 668, "Cancelar", size=15)
    cv.write_text(40, 744, "Portal ficticio para pruebas de software. Ninguna persona es real.", size=12, color=gray)
    return cv


# ---------------------------------------------------------------------------
# 5. Small properties dialog
# ---------------------------------------------------------------------------


def _dialog(f: FakeData, rng: np.random.Generator) -> Canvas:
    W, H = 600, 300
    cv = Canvas.new(W, H, (240, 240, 240))
    d = ImageDraw.Draw(cv.img)
    p = f.person()
    size = 11
    d.rectangle((0, 0, W, 26), fill=(252, 252, 252))
    d.line((0, 26, W, 26), fill=(205, 205, 205))
    cv.write_text(10, 18, "Propiedades del documento", font_name="sans_bold", size=12)
    cv.write_text(W - 22, 18, "x", font_name="sans", size=13, color=(80, 80, 80))
    x = 10
    for i, tab in enumerate(("General", "Descripción", "Seguridad", "Fuentes")):
        w = cv.text_width(tab, "sans", size) + 20
        if i == 1:
            d.rectangle((x, 32, x + w, 52), fill=(255, 255, 255), outline=(200, 200, 200))
        cv.write_text(x + 10, 47, tab, size=size)
        x += w + 2
    d.rectangle((10, 52, W - 10, 254), fill=(255, 255, 255), outline=(200, 200, 200))
    rows: list[tuple[str, list[Seg]]] = [
        ("Archivo:", [t("informe_avance_septiembre.pdf")]),
        ("Título:", [t("Informe de avance mensual")]),
        ("Autor:", [name(p)]),
        ("Correo del autor:", [email(p, _choose(rng, BASE_EMAILS))]),
        ("Asunto:", [t("Programa de fomento productivo")]),
        ("Palabras clave:", [t("informe, avance, honorarios")]),
        ("Creado:", [Seg(f.date(), tags={"decoy": "date"}), t(" 10:14")]),
        ("Modificado:", [Seg(f.date(), tags={"decoy": "date"}), t(" 16:51")]),
        ("Aplicación:", [t("Procesador de texto 7.2")]),
    ]
    for i, (label, segs) in enumerate(rows):
        yb = 74 + i * 20
        cv.write_text(150 - cv.text_width(label, "sans", size), yb, label, size=size, color=(90, 90, 90))
        write_segs(cv, 160, yb, segs, size=size)
    for text, x0 in (("Aceptar", 400), ("Cancelar", 494)):
        d.rectangle((x0, 264, x0 + 86, 290), fill=(250, 250, 250), outline=(170, 170, 170))
        cv.write_text(x0 + 43 - cv.text_width(text, "sans", size) / 2, 281, text, size=size)
    # Such a small dialog counts as stress for all its data
    _to_stress(cv.elements)
    for e in cv.elements:
        e.tags["small_text"] = True
    return cv


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate(ctx: Context) -> list[FileEntry]:
    f = ctx.fake_data(SEED_NAME)
    rng = ctx.rng(SEED_NAME)
    files: list[FileEntry] = []

    def save(
        build: Callable[[], Canvas],
        path: str,
        id: str,
        format: str,
        description: str,
        options: dict[str, Any] | None = None,
        post: Callable[[Canvas], Canvas] | None = None,
        **tags: Any,
    ) -> None:
        cv = build()
        if post is not None:
            cv = post(cv)
        save_image(cv.img, ctx.path(path), format, **(options or {}))
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

    save(
        lambda: _desktop_email(f, rng, 1.0),
        "screenshots/correo_escritorio.png",
        "pant_correo_escritorio",
        "png",
        "Cliente de correo 1920x1080: lista de remitentes, cabecera De/Para/CC y cuerpo con RUT y teléfono.",
        scale=1.0,
    )
    save(
        lambda: _mobile_chat(f, rng, ctx.faces),
        "screenshots/chat_movil.jpg",
        "pant_chat_movil",
        "jpg",
        "Chat de celular 1080x2340 (JPEG 75) con RUT, teléfono, correo y dirección en burbujas; avatar redondo.",
        options={"quality": 75},
    )
    save(
        lambda: _spreadsheet(f, rng),
        "screenshots/planilla.png",
        "pant_planilla",
        "png",
        "Planilla 1600x900 con 12 filas Nombre | RUT | Correo | Teléfono | Comuna | Monto (texto de 12-13 px).",
    )

    def shrink(cv: Canvas) -> Canvas:
        small = cv.scale(0.6)
        _to_stress(small.elements)
        for e in small.elements:
            e.tags["scale"] = 0.6
            e.tags["size_px"] = round(e.tags.get("size_px", 13) * 0.6, 1)
        return small

    save(
        lambda: _spreadsheet(f, rng),
        "screenshots/planilla_reducida.png",
        "pant_planilla_reducida",
        "png",
        "La misma planilla (con otras personas) reducida al 60 %: texto de unos 8 px.",
        post=shrink,
        scale=0.6,
    )
    save(
        lambda: _web_form(f, rng, ctx.faces),
        "screenshots/formulario_web.webp",
        "pant_formulario_web",
        "webp",
        "Formulario web 1366x768 con campos llenos, foto de perfil y el RUN en la URL.",
        options={"quality": 90},
    )
    save(
        lambda: _dialog(f, rng),
        "screenshots/dialogo_pequeno.png",
        "pant_dialogo_pequeno",
        "png",
        "Ventana 'Propiedades del documento' 600x300 con autor y correo en texto de 11 px.",
    )
    save(
        lambda: _desktop_email(f, rng, 1.5),
        "screenshots/correo_escritorio_hidpi.png",
        "pant_correo_escritorio_hidpi",
        "png",
        "El cliente de correo redibujado al 150 % (2880x1620), con otras personas.",
        scale=1.5,
    )
    return files
