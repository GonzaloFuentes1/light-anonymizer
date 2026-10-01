"""Standalone images: rotated, with EXIF/XMP/IPTC/PNG metadata, and TIFF (multipage and 1 bit).

Categories:

- ``rotated_image``: two designs (a printed note photographed and a business card) at 0, 90,
  180, 270, 15 and 45 degrees, plus stress variants (mirrored, heavy noise, tiny text and
  low contrast). At non-right angles the paper is composited onto a table.
- ``exif``: photos with EXIF orientation (3, 6, 8 and a wrong one), GPS, author, unredacted EXIF
  thumbnail, XMP, IPTC, WEBP with EXIF and PNG with text chunks and ``eXIf``.
- ``tiff``: three-page TIFF with mixed compression and tags, and a 1-bit Group 4 scan.

The ground truth is always in the *displayed* geometry (after applying the EXIF
orientation). HEIC is pending the decision on the codec license.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import piexif
import piexif.helper
from PIL import Image, ImageDraw, ImageOps, TiffImagePlugin
from PIL.PngImagePlugin import PngInfo

from test_bench.canvas import (
    Canvas,
    blur,
    jpeg_compress,
    lighting,
    noise,
    paper_texture,
    salt_pepper,
    table_texture,
)
from test_bench.context import Context, image_file_entry, save_image
from test_bench.faces import FaceProvider
from test_bench.fake_data import (
    ADMINISTRATIVE_PHRASES,
    EMAIL_FORMATS,
    PHONE_FORMATS,
    PHONE_FORMATS_BY_KIND,
    RUT_FORMATS,
    FakeData,
    Person,
    strip_accents,
)
from test_bench.schema import Element, FileEntry, MetadataEntry, Page

SEED_NAME = "imagenes"
CAT_ROTATED = "rotated_image"
CAT_EXIF = "exif"
CAT_TIFF = "tiff"

ANGLES = (0, 90, 180, 270, 15, 45)
OUTPUT_FORMATS = ("jpg", "png", "webp")

BOLD = {"sans": "sans_bold", "serif": "serif_bold", "mono": "mono_bold", "stix": "stix"}
PAPER = (248, 246, 240)

# Font combinations of the business card: name, position, data, labels.
CARD_FAMILIES = [
    {"name": "serif_bold", "position": "serif_italic", "data": "sans", "label": "sans_bold"},
    {"name": "sans_bold", "position": "sans_oblique", "data": "mono", "label": "sans"},
    {"name": "stix", "position": "stix_italic", "data": "stix", "label": "stix_italic"},
    {"name": "mono_bold", "position": "serif_italic", "data": "serif", "label": "serif_bold"},
]
POSITIONS = [
    "Profesional de Apoyo, División de Fomento",
    "Encargado(a) de Transparencia",
    "Analista de Presupuesto",
    "Jefatura de Planificación Regional",
    "Asesor(a) Jurídico(a)",
    "Coordinación de Programas Sociales",
]
INSTITUTIONS = [
    "Gobierno Regional Ficticio",
    "Municipalidad de Ejemplo",
    "Servicio Regional Ficticio",
    "Delegación Presidencial Ficticia",
]
NOTE_HEADERS = ["FICHA DE CONTACTO", "REGISTRO DE ATENCIÓN", "DATOS DEL SOLICITANTE", "HOJA DE VISITA"]

# Approximate public coordinates of Chilean cities; a random offset is added to them.
GPS_CITIES = [
    ("Concepción", -36.827, -73.050),
    ("Temuco", -38.736, -72.590),
    ("Valdivia", -39.814, -73.246),
    ("Talca", -35.427, -71.655),
    ("La Serena", -29.904, -71.249),
    ("Antofagasta", -23.650, -70.400),
    ("Puerto Montt", -41.469, -72.942),
    ("Chillán", -36.606, -72.103),
]

# EXIF orientation -> transposition that ``ImageOps.exif_transpose`` applies when displaying.
EXIF_TRANSPOSE = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_270, 8: Image.Transpose.ROTATE_90}
# Transposition to apply to the displayed image D to obtain the stored pixels.
EXIF_INVERSE = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_90, 8: Image.Transpose.ROTATE_270}

# Designs of the rotated_image category -> stem of their file names and ids (kept in Spanish).
_DESIGN_FILE_STEM = {"note": "nota", "card": "tarjeta"}
# Spanish labels of the degradations, for the human-readable description in the manifest.
_DEGRADATION_LABELS = {
    "lighting": "iluminacion",
    "light_noise": "ruido_leve",
    "heavy_noise": "ruido_fuerte",
    "blur": "desenfoque",
    "mirror": "espejo",
    "tiny_text": "texto_diminuto",
    "low_contrast": "bajo_contraste",
}


class _Cycle:
    """Walks a list in order, circularly (covers all formats evenly)."""

    def __init__(self, items: list[str]) -> None:
        self.items = items
        self.i = 0

    def next(self) -> str:
        v = self.items[self.i % len(self.items)]
        self.i += 1
        return v


def _base(formats: dict[str, str], candidates: list[str] | None = None) -> list[str]:
    return [f for f in (candidates or list(formats)) if formats[f] == "base"]


def _stress_first(formats: dict[str, str], candidates: list[str] | None = None) -> list[str]:
    """All formats, with the ``stress`` level ones first.

    Stress images are few: if the cycle started with the base formats, the rare formats
    (RUT with commas, old phone numbers, email with ``[arroba]``) would never be used.
    """
    every = candidates or list(formats)
    return [f for f in every if formats[f] == "stress"] + [f for f in every if formats[f] != "stress"]


@dataclass
class _Style:
    size: int
    font: str
    color: tuple[int, int, int] = (28, 28, 34)
    stress: bool = False


class _Generator:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.f: FakeData = ctx.fake_data(SEED_NAME)
        self.rng = ctx.rng(SEED_NAME)
        self._n_people = 0
        self.c_rut = _Cycle(_base(RUT_FORMATS))
        self.c_rut_stress = _Cycle(_stress_first(RUT_FORMATS))
        self.c_email = _Cycle(_base(EMAIL_FORMATS))
        self.c_email_stress = _Cycle(_stress_first(EMAIL_FORMATS))
        self.c_tel = {k: _Cycle(_base(PHONE_FORMATS, fs)) for k, fs in PHONE_FORMATS_BY_KIND.items()}
        self.c_tel_stress = {k: _Cycle(_stress_first(PHONE_FORMATS, fs)) for k, fs in PHONE_FORMATS_BY_KIND.items()}
        self.c_family = 0
        self.files: list[FileEntry] = []

    # -- data ------------------------------------------------------------------

    def person(self, in_list: bool | None = None) -> Person:
        """New person; some end up outside the list, with an invalid DV, 7 digits or with K."""
        self._n_people += 1
        n = self._n_people
        if in_list is None:
            in_list = n % 5 != 3
        p = self.f.person(dv_valid=n % 6 != 4, in_list=in_list, seven_digits=n % 9 == 7)
        if n % 11 == 5:
            p.rut_body, p.rut_dv = self.f.rut_with_k()
            p.rut_dv_valid = True
        return p

    def rut(self, p: Person, stress: bool) -> tuple[str, str, dict[str, Any]]:
        fmt = (self.c_rut_stress if stress else self.c_rut).next()
        level = "stress" if stress else RUT_FORMATS[fmt]
        return p.rut(fmt), level, {"format": fmt, "dv_valid": p.rut_dv_valid}

    def email(self, p: Person, stress: bool) -> tuple[str, str, dict[str, Any]]:
        fmt = (self.c_email_stress if stress else self.c_email).next()
        level = "stress" if stress else EMAIL_FORMATS[fmt]
        return p.email(fmt), level, {"format": fmt}

    def phone(self, p: Person, stress: bool, landline: bool = False) -> tuple[str, str, dict[str, Any]]:
        tel = p.landline if landline else p.phone
        fmt = (self.c_tel_stress if stress else self.c_tel)[tel.kind].next()
        level = "stress" if stress else PHONE_FORMATS[fmt]
        return tel.format(fmt), level, {"format": fmt, "kind": tel.kind}

    @staticmethod
    def list_level(p: Person, stress: bool) -> str:
        if not p.in_list:
            return "out_of_scope"
        return "stress" if stress else "base"

    # -- writing -----------------------------------------------------------------

    @staticmethod
    def field(
        canvas: Canvas,
        x: float,
        y: float,
        label: str,
        datum: str,
        type: str,
        *,
        level: str,
        tags: dict[str, Any] | None,
        size: int,
        label_font: str,
        datum_font: str,
        color: tuple[int, int, int],
        text_level: str = "base",
    ) -> float:
        """Writes ``label`` (neutral text) followed by ``datum`` on the same baseline."""
        if label:
            canvas.write_text(
                x, y, label, value=label.strip(), font_name=label_font, size=size, color=color, level=text_level,
            )  # fmt: skip
            x += canvas.text_width(label, label_font, size)
        canvas.write_text(x, y, datum, type=type, font_name=datum_font, size=size, color=color, level=level, tags=tags)
        return x + canvas.text_width(datum, datum_font, size)

    # -- designs -------------------------------------------------------------------

    def note(self, style: _Style, paper_color: tuple[int, int, int] = PAPER) -> Canvas:
        """Design (a): printed sheet with name, RUT, email and phone (plus a decoy date)."""
        p = self.person()
        s = style.stress
        rut, n_rut, t_rut = self.rut(p, s)
        email, n_email, t_email = self.email(p, s)
        tel, n_tel, t_tel = self.phone(p, s)
        text_level = "stress" if s else "base"
        rows = [
            ("Nombre: ", p.full_name, "name", self.list_level(p, s),
             {"in_list": p.in_list, "variant": "full"}),
            ("RUT: ", rut, "rut", n_rut, t_rut),
            ("Correo: ", email, "email", n_email, t_email),
            ("Teléfono: ", tel, "phone", n_tel, t_tel),
            ("Fecha: ", self.f.date(), "text", text_level, {"decoy": "date"}),
        ]  # fmt: skip
        size, fnt = style.size, style.font
        header = NOTE_HEADERS[int(self.rng.integers(len(NOTE_HEADERS)))]
        header_size = int(round(size * 1.1))
        meter = Canvas.new(8, 8)
        widths = [meter.text_width(a + b, fnt, size) for a, b, *_ in rows]
        widths.append(meter.text_width(header, BOLD[fnt], header_size))
        margin = int(size * 1.8)
        line_spacing = int(size * 1.7)
        width = int(max(widths)) + 2 * margin
        height = 2 * margin + header_size + int(line_spacing * 0.6) + line_spacing * len(rows)
        canvas = Canvas(paper_texture(width, height, self.rng, paper_color))
        y = margin + header_size
        canvas.write_text(margin, y, header, font_name=BOLD[fnt], size=header_size, color=style.color, level=text_level)
        y += int(line_spacing * 1.4)
        for lbl, datum, type_, level, tags in rows:
            self.field(
                canvas, margin, y, lbl, datum, type_, level=level, tags=tags, size=size, label_font=fnt,
                datum_font=fnt, color=style.color, text_level=text_level,
            )  # fmt: skip
            y += line_spacing
        return canvas

    def card(self, stress: bool = False, low_contrast: bool = False) -> Canvas:
        """Design (b): business card with name, position, email, mobile, landline, web and address."""
        p = self.person()
        fam = CARD_FAMILIES[self.c_family % len(CARD_FAMILIES)]
        self.c_family += 1
        email, n_email, t_email = self.email(p, stress)
        mobile, n_mobile, t_mobile = self.phone(p, stress)
        landline, n_landline, t_landline = self.phone(p, stress, landline=True)
        user = strip_accents(p.email("dot").split("@")[0]).replace(".", "-")
        web = f"www.{p.domain}/{user}"
        text_level = "stress" if stress else "base"
        if low_contrast:
            background, ink, band, band_ink = (196, 196, 190), (150, 150, 146), (172, 172, 170), (206, 206, 202)
        else:
            tones = [(22, 60, 110), (120, 30, 40), (20, 90, 70), (70, 60, 110)]
            band = tones[int(self.rng.integers(len(tones)))]
            background, ink, band_ink = (251, 250, 247), (30, 30, 36), (255, 255, 255)
        rows = [
            ("Correo: ", email, "email", n_email, t_email),
            ("Móvil: ", mobile, "phone", n_mobile, t_mobile),
            ("Fono: ", landline, "phone", n_landline, t_landline),
            ("Web: ", web, "url", "stress" if stress else "base", {"format": "personal_web"}),
            ("Dirección: ", p.address, "address", self.list_level(p, stress), {"in_list": p.in_list}),
        ]
        size_d, size_n, size_p = 27, 46, 26
        meter = Canvas.new(8, 8)
        needed = max(
            meter.text_width(a, fam["label"], size_d)
            + meter.text_width(b + (f", {p.commune}" if t == "address" else ""), fam["data"], size_d)
            for a, b, t, *_ in rows
        )
        needed = max(needed, meter.text_width(p.full_name, fam["name"], size_n))
        width, height = max(1050, int(needed) + 140), 600
        canvas = Canvas.new(width, height, background)
        d = ImageDraw.Draw(canvas.img)
        d.rectangle((0, 0, width, 96), fill=band)
        d.rectangle((70, 262, 70 + int(width * 0.35), 265), fill=band)
        inst = INSTITUTIONS[int(self.rng.integers(len(INSTITUTIONS)))]
        canvas.write_text(70, 60, inst, font_name="sans_bold", size=28, color=band_ink, level=text_level)
        canvas.write_text(
            70, 190, p.full_name, type="name", font_name=fam["name"], size=size_n, color=ink,
            level=self.list_level(p, stress), tags={"in_list": p.in_list, "variant": "full"},
        )  # fmt: skip
        position = POSITIONS[int(self.rng.integers(len(POSITIONS)))]
        canvas.write_text(70, 240, position, font_name=fam["position"], size=size_p, color=ink, level=text_level)
        y = 322
        for lbl, datum, type_, level, tags in rows:
            x = self.field(
                canvas, 70, y, lbl, datum, type_, level=level, tags=tags, size=size_d,
                label_font=fam["label"], datum_font=fam["data"], color=ink, text_level=text_level,
            )  # fmt: skip
            if type_ == "address":
                canvas.write_text(
                    x, y, f", {p.commune}", font_name=fam["data"], size=size_d, color=ink, level=text_level,
                )  # fmt: skip
            y += 44
        return canvas

    def face_photo(self, pose: str = "front", caption: bool = False) -> Canvas:
        """Photo of a person on a plain gradient background (optionally with their name below)."""
        (r,) = self.ctx.faces.take(pose, 1)
        width = int(self.rng.uniform(420, 540))
        face_height = int(round(r.img.height * width / r.img.width))
        mx, my = int(self.rng.uniform(60, 140)), int(self.rng.uniform(50, 110))
        extra = 90 if caption else 0
        p = self.person() if caption else None
        caption_label = "Credencial: "
        w = width + 2 * mx
        if p is not None:
            meter = Canvas.new(8, 8)
            length = meter.text_width(caption_label, "sans", 30) + meter.text_width(p.full_name, "sans_bold", 30)
            w = max(w, int(length) + 2 * mx)
        h = face_height + 2 * my + extra
        c0 = self.rng.uniform(150, 220, 3)
        c1 = c0 * self.rng.uniform(0.7, 0.95)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        background = (c0 * (1 - t) + c1 * t) * np.ones((h, w, 3), np.float32)
        canvas = Canvas(Image.fromarray(background.clip(0, 255).astype(np.uint8)))
        canvas.paste(r.img, (w - width) // 2, my, width=width, elements=FaceProvider.elements(r, "base"))
        if p is not None:
            self.field(
                canvas, mx, my + face_height + 60, caption_label, p.full_name, "name",
                level=self.list_level(p, False), tags={"in_list": p.in_list, "variant": "full"}, size=30,
                label_font="sans", datum_font="sans_bold", color=(20, 20, 20),
            )  # fmt: skip
        return canvas

    def screenshot(self) -> Canvas:
        """Screenshot of a management system showing a person's record."""
        p = self.person()
        rut, n_rut, t_rut = self.rut(p, False)
        email, n_email, t_email = self.email(p, False)
        mobile, n_mobile, t_mobile = self.phone(p, False)
        landline, n_landline, t_landline = self.phone(p, False, landline=True)
        w, h = 1280, 780
        canvas = Canvas.new(w, h, (236, 239, 243))
        d = ImageDraw.Draw(canvas.img)
        d.rectangle((0, 0, w, 46), fill=(44, 62, 80))
        canvas.write_text(24, 30, "Sistema de Gestión Documental - Ficha de persona", size=19, color=(250, 250, 250))
        d.rectangle((40, 80, w - 40, h - 40), fill=(255, 255, 255), outline=(200, 205, 212))
        rows = [
            ("Nombre", p.full_name, "name", self.list_level(p, False),
             {"in_list": p.in_list, "variant": "full"}),
            ("RUT", rut, "rut", n_rut, t_rut),
            ("Correo electrónico", email, "email", n_email, t_email),
            ("Teléfono móvil", mobile, "phone", n_mobile, t_mobile),
            ("Teléfono fijo", landline, "phone", n_landline, t_landline),
            ("Domicilio", p.address, "address", self.list_level(p, False), {"in_list": p.in_list}),
            ("Fecha de ingreso", self.f.date(), "text", "base", {"decoy": "date"}),
            ("Monto asignado", self.f.amount(), "text", "base", {"decoy": "amount"}),
        ]  # fmt: skip
        y = 150
        for lbl, datum, type_, level, tags in rows:
            canvas.write_text(80, y, lbl, size=20, color=(90, 96, 110))
            d.rectangle((330, y - 30, w - 90, y + 14), fill=(250, 251, 253), outline=(180, 186, 196))
            canvas.write_text(344, y, datum, type=type_, size=21, color=(20, 24, 30), level=level, tags=tags)
            y += 72
        return canvas

    # -- composition ------------------------------------------------------------------

    def rotate(self, canvas: Canvas, degrees: float, background: tuple[int, int, int], margin: int = 40) -> Canvas:
        """Rotates the canvas; at non-right angles it is composited onto a table (geometry stays exact)."""
        g = degrees % 360
        if g in (0, 90, 180, 270):
            return canvas.rotate(g)
        rot = canvas.rotate(g, background=background)
        mask = (
            Canvas(Image.new("RGB", canvas.img.size, (255, 255, 255))).rotate(g, background=(0, 0, 0)).img.convert("L")
        )
        final = Canvas(table_texture(rot.width + 2 * margin, rot.height + 2 * margin, self.rng))
        final.paste(rot.img, margin, margin, elements=rot.elements, mask=mask)
        return final

    def document_photo(self, style: _Style | None = None) -> Canvas:
        """Note photographed on the table with a slight rotation and phone-camera lighting."""
        style = style or _Style(size=int(self.rng.integers(28, 37)), font="sans")
        canvas = self.note(style)
        degrees = float(self.rng.uniform(2, 5)) * (1 if self.rng.random() < 0.5 else -1)
        photo = self.rotate(canvas, degrees, PAPER, margin=60)
        photo.img = lighting(photo.img, self.rng, 0.25)
        return photo

    def light_degrade(self, img: Image.Image, fmt: str) -> Image.Image:
        img = noise(img, self.rng, 2.5)
        return img if fmt == "jpg" else jpeg_compress(img, 85)

    def save(self, img: Image.Image, relative: str, fmt: str, **options: Any) -> None:
        if fmt == "jpg":
            options.setdefault("quality", 85)
        elif fmt == "webp":
            options.setdefault("quality", 90)
        save_image(img, self.ctx.path(relative), fmt, **options)

    # -- rotated_image category -----------------------------------------------------------

    def rotated(self) -> None:
        note_fonts = ["sans", "serif", "mono", "stix"]
        i = 0
        for offset, design in enumerate(("note", "card")):
            for angle in ANGLES:
                # The offset keeps an angle from always getting the same format in both designs.
                fmt = OUTPUT_FORMATS[(i + offset) % len(OUTPUT_FORMATS)]
                if design == "note":
                    style = _Style(size=int(self.rng.integers(28, 41)), font=note_fonts[i % len(note_fonts)])
                    canvas = self.rotate(self.note(style), angle, PAPER)
                    canvas.img = lighting(canvas.img, self.rng, 0.22)
                    degr = ["lighting", "light_noise", "jpeg_85"]
                else:
                    canvas = self.rotate(self.card(), angle, (251, 250, 247))
                    degr = ["light_noise", "jpeg_85"]
                img = self.light_degrade(canvas.img, fmt)
                self._add_rotated(canvas, img, design, str(angle), angle, fmt, degr, stress=False)
                i += 1
        self._rotated_stress()

    def _rotated_stress(self) -> None:
        # Horizontal mirror at 0 degrees (note) and at 90 degrees (card).
        canvas = self.note(_Style(size=34, font="serif", stress=True)).mirror()
        canvas.img = lighting(canvas.img, self.rng, 0.2)
        self._add_rotated(
            canvas, self.light_degrade(canvas.img, "jpg"), "note", "espejo_0", 0, "jpg", ["mirror", "lighting"], True
        )
        canvas = self.card(stress=True).rotate(90).mirror()
        self._add_rotated(
            canvas, self.light_degrade(canvas.img, "png"), "card", "espejo_90", 90, "png", ["mirror"], True
        )
        # 45 degrees with heavy noise and blur.
        canvas = self.rotate(self.note(_Style(size=32, font="sans", stress=True)), 45, PAPER)
        canvas.img = lighting(canvas.img, self.rng, 0.3)
        img = jpeg_compress(blur(noise(canvas.img, self.rng, 24), 1.6), 60)
        self._add_rotated(
            canvas, img, "note", "45_ruido", 45, "jpg", ["lighting", "heavy_noise", "blur", "jpeg_60"], True
        )
        # Tiny text (~12 px) at 15 degrees.
        canvas = self.rotate(self.note(_Style(size=12, font="sans", stress=True)), 15, PAPER, margin=24)
        self._add_rotated(
            canvas, self.light_degrade(canvas.img, "png"), "note", "diminuta_15", 15, "png", ["tiny_text"], True
        )
        # Low contrast.
        canvas = self.card(stress=True, low_contrast=True)
        self._add_rotated(
            canvas,
            self.light_degrade(canvas.img, "jpg"),
            "card",
            "bajo_contraste_0",
            0,
            "jpg",
            ["low_contrast"],
            True,
        )

    def _add_rotated(
        self,
        canvas: Canvas,
        img: Image.Image,
        design: str,
        suffix: str,
        angle: int,
        fmt: str,
        degradation: list[str],
        stress: bool,
    ) -> None:
        stem = _DESIGN_FILE_STEM[design]
        path = f"rotated_images/{stem}_{suffix}.{fmt}"
        self.save(img, path, fmt)
        mirror = "mirror" in degradation
        for e in canvas.elements:
            e.tags["degradation"] = list(degradation)
        labels = [_DEGRADATION_LABELS.get(x, x) for x in degradation]
        description = (
            f"{'Nota fotografiada' if design == 'note' else 'Tarjeta de presentación'} girada {angle} grados"
            + (" y espejada" if mirror else "")
            + (f" ({', '.join(labels)})" if stress else "")
        )
        self.files.append(
            image_file_entry(
                id=f"img_rot_{stem}_{suffix}",
                path=path,
                format=fmt,
                category=CAT_ROTATED,
                description=description,
                canvas=canvas,
                tags={
                    "design": design,
                    "angle": angle,
                    "mirror": mirror,
                    "degradation": degradation,
                    "stress": stress,
                },
            )  # fmt: skip
        )

    # -- exif category --------------------------------------------------------------------

    def gps(self) -> tuple[dict[int, Any], dict[str, Any]]:
        city, lat0, lon0 = GPS_CITIES[int(self.rng.integers(len(GPS_CITIES)))]
        lat = round(lat0 + float(self.rng.uniform(-0.05, 0.05)), 6)
        lon = round(lon0 + float(self.rng.uniform(-0.05, 0.05)), 6)
        alt = round(float(self.rng.uniform(5, 400)), 1)
        ifd = {
            piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
            piexif.GPSIFD.GPSLatitudeRef: b"S",
            piexif.GPSIFD.GPSLatitude: _dms(lat),
            piexif.GPSIFD.GPSLongitudeRef: b"W",
            piexif.GPSIFD.GPSLongitude: _dms(lon),
            piexif.GPSIFD.GPSAltitudeRef: 0,
            piexif.GPSIFD.GPSAltitude: (int(round(alt * 10)), 10),
            piexif.GPSIFD.GPSDateStamp: b"2026:05:12",
        }
        return ifd, {"lat": lat, "lon": lon, "alt": alt, "zone": city, "fictitious": True}

    def gps_meta(self, gps_tags: dict[str, Any], location: str = "exif.gps") -> MetadataEntry:
        return MetadataEntry(location=location, value=None, tags=dict(gps_tags))

    def exif(self) -> None:
        self._orientation_6()
        for orientation in (3, 8):
            self._orientation_simple(orientation)
        self._wrong_orientation()
        self._webp_gps()
        self._png_metadata()
        self._face_gps()
        self._iptc()

    def _save_oriented(self, d: Canvas, orientation: int, path: str, exif: bytes, **options: Any) -> None:
        """Saves the pixels so that, after applying the EXIF orientation, exactly ``d`` is displayed."""
        stored = d.img.transpose(EXIF_INVERSE[orientation])
        self.save(stored, path, "jpg", exif=exif, quality=92, **options)
        with Image.open(self.ctx.path(path)) as loaded:
            view = ImageOps.exif_transpose(loaded).convert("RGB")
        if view.size != d.img.size:
            raise RuntimeError(f"{path}: the EXIF orientation does not reproduce the displayed geometry")
        diff = np.abs(np.asarray(view, np.int16) - np.asarray(d.img, np.int16)).mean()
        if diff > 4:
            raise RuntimeError(f"{path}: the displayed image differs from the expected one (mean diff {diff:.1f})")

    def _orientation_6(self) -> None:
        d = self.document_photo()
        path = "exif/foto_orientacion_6.jpg"
        author, author_xp, creator, owner = self.person(), self.person(), self.person(), self.person()
        p_desc, p_com = self.person(), self.person()
        rut_desc = p_desc.rut("dots")
        email = p_com.email("dot")
        gps_ifd, gps_tags = self.gps()
        stored = d.img.transpose(EXIF_INVERSE[6])
        thumbnail = stored.copy()
        thumbnail.thumbnail((160, 160))
        buf = io.BytesIO()
        thumbnail.save(buf, "JPEG", quality=75)
        data = {
            "0th": {
                piexif.ImageIFD.Orientation: 6,
                piexif.ImageIFD.Make: b"Ficticia",
                piexif.ImageIFD.Model: b"FotoFono FX-200",
                piexif.ImageIFD.Software: b"Camara 4.2",
                piexif.ImageIFD.DateTime: b"2026:05:12 10:31:07",
                piexif.ImageIFD.Artist: author.full_name.encode("utf-8"),
                piexif.ImageIFD.ImageDescription: f"Foto RUT {rut_desc}".encode(),
                piexif.ImageIFD.XPAuthor: tuple(author_xp.full_name.encode("utf-16-le") + b"\x00\x00"),
                piexif.ImageIFD.Copyright: f"(c) 2026 {owner.full_name}".encode(),
            },
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: b"2026:05:12 10:31:07",
                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(email, encoding="ascii"),
            },
            "GPS": gps_ifd,
            # IFD1 of a JPEG thumbnail: Compression=6 (piexif adds the offset and the length).
            "1st": {
                piexif.ImageIFD.Compression: 6,
                piexif.ImageIFD.XResolution: (72, 1),
                piexif.ImageIFD.YResolution: (72, 1),
                piexif.ImageIFD.ResolutionUnit: 2,
            },
            "thumbnail": buf.getvalue(),
        }
        xmp = _xmp(creator.full_name)
        self._save_oriented(d, 6, path, piexif.dump(data), xmp=xmp)
        metadata = [
            self.gps_meta(gps_tags),
            MetadataEntry("exif.artist", author.full_name, {"type": "name", "tag": 315}),
            MetadataEntry("exif.image_description", rut_desc, {"type": "rut", "text": f"Foto RUT {rut_desc}",
                                                               "tag": 270}),
            MetadataEntry("exif.user_comment", email, {"type": "email", "tag": 37510, "encoding": "ascii"}),
            MetadataEntry("exif.xp_author", author_xp.full_name, {"type": "name", "tag": 40093,
                                                                  "encoding": "utf-16-le"}),
            MetadataEntry("exif.copyright", owner.full_name, {"type": "name", "tag": 33432}),
            MetadataEntry("exif.thumbnail", None, {"width": thumbnail.width, "height": thumbnail.height,
                                                   "unredacted": True}),
            MetadataEntry("xmp.dc:creator", creator.full_name, {"type": "name", "container": "jpeg.app1"}),
        ]  # fmt: skip
        _mark_degradation(d.elements, ["lighting", "jpeg_92"])
        self.files.append(
            image_file_entry(
                id="img_exif_orientacion_6",
                path=path,
                format="jpg",
                category=CAT_EXIF,
                description="Foto de documento guardada girada con EXIF Orientation=6; EXIF completo, GPS, "
                "miniatura sin censurar y XMP",
                canvas=d,
                metadata=metadata,
                tags={"exif_orientation": 6, "saved_size": list(stored.size)},
            )
        )

    def _orientation_simple(self, orientation: int) -> None:
        d = self.document_photo()
        path = f"exif/foto_orientacion_{orientation}.jpg"
        gps_ifd, gps_tags = self.gps()
        data = {
            "0th": {
                piexif.ImageIFD.Orientation: orientation,
                piexif.ImageIFD.Make: b"Ficticia",
                piexif.ImageIFD.Model: b"FotoFono FX-100",
            },
            "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:06:03 16:02:44"},
            "GPS": gps_ifd,
        }
        self._save_oriented(d, orientation, path, piexif.dump(data))
        stored = d.img.transpose(EXIF_INVERSE[orientation])
        _mark_degradation(d.elements, ["lighting", "jpeg_92"])
        self.files.append(
            image_file_entry(
                id=f"img_exif_orientacion_{orientation}",
                path=path,
                format="jpg",
                category=CAT_EXIF,
                description=f"Foto de documento con EXIF Orientation={orientation} y GPS",
                canvas=d,
                metadata=[self.gps_meta(gps_tags)],
                tags={"exif_orientation": orientation, "saved_size": list(stored.size)},
            )
        )

    def _wrong_orientation(self) -> None:
        """Upright pixels but Orientation=6: when displayed (and in the output) the text ends up sideways."""
        u = self.document_photo()
        path = "exif/foto_orientacion_incorrecta.jpg"
        data = {
            "0th": {piexif.ImageIFD.Orientation: 6, piexif.ImageIFD.Make: b"Ficticia"},
            "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:04:22 09:15:00"},
        }
        self.save(u.img, path, "jpg", exif=piexif.dump(data), quality=92)
        d = u.rotate(270)  # same transposition that exif_transpose applies for Orientation=6
        for e in d.elements:
            e.tags["wrong_exif"] = True
        _mark_degradation(d.elements, ["lighting", "jpeg_92"])
        self.files.append(
            image_file_entry(
                id="img_exif_orientacion_incorrecta",
                path=path,
                format="jpg",
                category=CAT_EXIF,
                description="Foto derecha con EXIF Orientation=6 erróneo: al aplicar EXIF el texto queda de lado",
                canvas=d,
                tags={"exif_orientation": 6, "wrong_exif": True, "saved_size": list(u.img.size)},
            )
        )

    def _webp_gps(self) -> None:
        canvas = self.rotate(self.card(), float(self.rng.uniform(-6, 6)), (251, 250, 247), margin=50)
        canvas.img = lighting(canvas.img, self.rng, 0.2)
        path = "exif/foto_gps.webp"
        author, creator = self.person(), self.person()
        gps_ifd, gps_tags = self.gps()
        data = {
            "0th": {piexif.ImageIFD.Artist: author.full_name.encode("utf-8"), piexif.ImageIFD.Make: b"Ficticia"},
            "GPS": gps_ifd,
        }
        self.save(noise(canvas.img, self.rng, 2.5), path, "webp", exif=piexif.dump(data), xmp=_xmp(creator.full_name))
        metadata = [
            self.gps_meta(gps_tags),
            MetadataEntry("exif.artist", author.full_name, {"type": "name", "tag": 315, "container": "webp.exif"}),
            MetadataEntry("xmp.dc:creator", creator.full_name, {"type": "name", "container": "webp.xmp"}),
        ]
        _mark_degradation(canvas.elements, ["lighting", "light_noise", "webp_90"])
        self.files.append(
            image_file_entry(
                id="img_exif_webp_gps",
                path=path,
                format="webp",
                category=CAT_EXIF,
                description="Foto de tarjeta en WEBP con EXIF (GPS, Artist) y XMP",
                canvas=canvas,
                metadata=metadata,
            )
        )

    def _png_metadata(self) -> None:
        canvas = self.screenshot()
        path = "exif/captura_metadatos.png"
        author, p_com, p_desc, creator = self.person(), self.person(), self.person(), self.person()
        email = p_com.email("dot")
        rut = p_desc.rut("no_dots")
        gps_ifd, gps_tags = self.gps()
        info = PngInfo()
        info.add_text("Author", author.full_name)
        info.add_text("Comment", email)
        info.add_text("Description", f"Captura ficha RUT {rut}")
        info.add_itxt("XML:com.adobe.xmp", _xmp(creator.full_name).decode("utf-8"))
        self.save(canvas.img, path, "png", pnginfo=info, exif=piexif.dump({"GPS": gps_ifd}))
        metadata = [
            MetadataEntry("png.text.Author", author.full_name, {"type": "name", "chunk": "tEXt"}),
            MetadataEntry("png.text.Comment", email, {"type": "email", "chunk": "tEXt"}),
            MetadataEntry("png.text.Description", rut, {"type": "rut", "chunk": "tEXt",
                                                        "text": f"Captura ficha RUT {rut}"}),
            MetadataEntry("xmp.dc:creator", creator.full_name, {"type": "name", "container": "png.iTXt"}),
            self.gps_meta(gps_tags, "png.exif.gps"),
        ]  # fmt: skip
        _mark_degradation(canvas.elements, [])  # clean screenshot, lossless
        self.files.append(
            image_file_entry(
                id="img_exif_png_metadatos",
                path=path,
                format="png",
                category=CAT_EXIF,
                description="Pantallazo PNG con tEXt (Author, Comment, Description), iTXt XMP y eXIf con GPS",
                canvas=canvas,
                metadata=metadata,
            )
        )

    def _face_gps(self) -> None:
        canvas = self.face_photo("front")
        path = "exif/foto_rostro_gps.jpg"
        author = self.person()
        gps_ifd, gps_tags = self.gps()
        data = {
            "0th": {
                piexif.ImageIFD.Make: b"Ficticia",
                piexif.ImageIFD.Model: b"FotoFono FX-300",
                piexif.ImageIFD.Artist: author.full_name.encode("utf-8"),
            },
            "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:03:18 12:40:10"},
            "GPS": gps_ifd,
        }
        self.save(noise(canvas.img, self.rng, 2.0), path, "jpg", exif=piexif.dump(data), quality=90)
        _mark_degradation(canvas.elements, ["light_noise", "jpeg_90"])
        self.files.append(
            image_file_entry(
                id="img_exif_rostro_gps",
                path=path,
                format="jpg",
                category=CAT_EXIF,
                description="Foto de una persona con GPS y Artist en EXIF",
                canvas=canvas,
                metadata=[
                    self.gps_meta(gps_tags),
                    MetadataEntry("exif.artist", author.full_name, {"type": "name", "tag": 315}),
                ],
            )
        )

    def _iptc(self) -> None:
        d = self.document_photo()
        path = "exif/foto_iptc.jpg"
        author, p_caption = self.person(), self.person()
        rut = p_caption.rut("dots")
        caption = f"Documento de {p_caption.full_name}, RUT {rut}"
        buf = io.BytesIO()
        Image.frombytes("RGB", d.img.size, d.img.tobytes()).save(buf, "JPEG", quality=88)
        segment = _app13_iptc(
            [
                (1, 90, b"\x1b%G"),  # CodedCharacterSet = UTF-8
                (2, 0, b"\x00\x04"),  # RecordVersion
                (2, 80, author.full_name.encode("utf-8")),  # By-line
                (2, 120, caption.encode("utf-8")),  # Caption/Abstract
            ]
        )
        self.ctx.path(path).write_bytes(_insert_jpeg_segment(buf.getvalue(), segment))
        _mark_degradation(d.elements, ["lighting", "jpeg_88"])
        self.files.append(
            image_file_entry(
                id="img_exif_iptc",
                path=path,
                format="jpg",
                category=CAT_EXIF,
                description="Foto de documento con bloque IPTC (APP13) con By-line y Caption",
                canvas=d,
                metadata=[
                    MetadataEntry("iptc.byline", author.full_name, {"type": "name", "dataset": "2:80"}),
                    MetadataEntry("iptc.caption", rut, {"type": "rut", "dataset": "2:120", "text": caption}),
                    MetadataEntry("iptc.caption", p_caption.full_name, {"type": "name", "dataset": "2:120"}),
                ],
            )
        )

    # -- tiff category ----------------------------------------------------------------------

    def tiff(self) -> None:
        self._tiff_multipage()
        self._tiff_bw_scan()

    def _tiff_multipage(self) -> None:
        pages = [
            self.note(_Style(size=32, font="serif")),
            self.card().rotate(90),
            self.face_photo("three_quarter", caption=True),
        ]
        compressions = ["tiff_lzw", "tiff_adobe_deflate", "jpeg"]
        path = "tiff/multipagina.tif"
        author, p_desc, p_doc, p_page = self.person(), self.person(), self.person(), self.person()
        rut = p_desc.rut("dots")
        email = p_doc.email("underscore")
        ifd = TiffImagePlugin.ImageFileDirectory_v2()
        ifd[315] = author.full_name.encode("utf-8")  # Artist (bytes to keep UTF-8)
        ifd[270] = f"Expediente RUT {rut}".encode()  # ImageDescription
        ifd[305] = "Escaner Ficticio 3.1"  # Software
        ifd[269] = f"Respaldo {email}".encode()  # DocumentName
        ifd[285] = f"Ficha de {p_page.full_name}".encode()  # PageName
        dest = self.ctx.path(path)
        # Each page is first encoded to a real file: if libtiff writes to memory (as happens when
        # saving directly into AppendingTiffWriter) it leaves uninitialized padding bytes and the
        # file is no longer deterministic.
        temp = dest.with_name(dest.name + ".pagina.tmp")
        with TiffImagePlugin.AppendingTiffWriter(dest, new=True) as tf:
            for i, (canvas, comp) in enumerate(zip(pages, compressions, strict=True)):
                options: dict[str, Any] = {"compression": comp, "dpi": (200, 200)}
                if i == 0:
                    options["tiffinfo"] = ifd
                if comp == "jpeg":
                    options["quality"] = 88
                save_image(canvas.img, temp, "tiff", **options)
                tf.write(temp.read_bytes())
                tf.newFrame()
        temp.unlink()
        elements: list[Element] = []
        for i, canvas in enumerate(pages):
            _mark_degradation(canvas.elements, [] if compressions[i].startswith("tiff_") else ["jpeg_88"])
            for e in canvas.elements:
                e.page = i
                elements.append(e)
        self.files.append(
            FileEntry(
                id="img_tiff_multipagina",
                path=path,
                format="tiff",
                category=CAT_TIFF,
                description="TIFF de 3 páginas de distinto tamaño (nota, tarjeta girada 90, foto con rostro), "
                "compresión LZW/Deflate/JPEG y etiquetas con datos en la página 0",
                pages=[Page(index=i, width=c.width, height=c.height, unit="px") for i, c in enumerate(pages)],
                elements=elements,
                sensitive_metadata=[
                    MetadataEntry("tiff.artist", author.full_name, {"type": "name", "tag": 315, "page": 0}),
                    MetadataEntry("tiff.image_description", rut, {"type": "rut", "tag": 270, "page": 0}),
                    MetadataEntry("tiff.document_name", email, {"type": "email", "tag": 269, "page": 0}),
                    MetadataEntry("tiff.page_name", p_page.full_name, {"type": "name", "tag": 285, "page": 0}),
                ],
                tags={"compressions": compressions},
            )
        )

    def _tiff_bw_scan(self) -> None:
        """Letter page at 200 dpi, binarized to 1 bit and compressed with CCITT Group 4."""
        w, h = 1700, 2200
        canvas = Canvas.new(w, h)
        margin, size, spacing = 150, 30, 50
        ink = (15, 15, 15)
        y = 230
        canvas.write_text(margin, y, "GOBIERNO REGIONAL FICTICIO", font_name="serif_bold", size=42, color=ink)
        y += 70
        canvas.write_text(margin, y, f"ACTA DE ENTREGA N° {self.f.folio()}", font_name="serif", size=34, color=ink)
        y += 90
        phrases = list(ADMINISTRATIVE_PHRASES)
        order = self.rng.permutation(len(phrases))
        for line in _fit(canvas, " ".join(phrases[i] for i in order[:3]), "serif", size, w - 2 * margin):
            canvas.write_text(margin, y, line, font_name="serif", size=size, color=ink)
            y += spacing
        y += 30
        for role in ("Entrega", "Recibe"):
            p = self.person()
            rut, n_rut, t_rut = self.rut(p, False)
            email, n_email, t_email = self.email(p, False)
            tel, n_tel, t_tel = self.phone(p, False, landline=role == "Recibe")
            canvas.write_text(margin, y, f"{role}:", font_name="serif_bold", size=size, color=ink)
            y += spacing
            rows = [
                ("Nombre: ", p.full_name, "name", self.list_level(p, False),
                 {"in_list": p.in_list, "variant": "full"}),
                ("RUT: ", rut, "rut", n_rut, t_rut),
                ("Domicilio: ", p.address, "address", self.list_level(p, False), {"in_list": p.in_list}),
                ("Teléfono: ", tel, "phone", n_tel, t_tel),
                ("Correo: ", email, "email", n_email, t_email),
            ]  # fmt: skip
            for lbl, datum, type_, level, tags in rows:
                self.field(
                    canvas, margin + 40, y, lbl, datum, type_, level=level, tags=tags, size=size,
                    label_font="serif", datum_font="serif", color=ink,
                )  # fmt: skip
                y += spacing
            y += 30
        for lbl, datum, decoy in (("Fecha: ", self.f.date(), "date"), ("Monto: ", self.f.amount(), "amount")):
            self.field(
                canvas, margin, y, lbl, datum, "text", level="base", tags={"decoy": decoy}, size=size,
                label_font="serif", datum_font="serif", color=ink,
            )  # fmt: skip
            y += spacing
        y += 20
        for line in _fit(canvas, " ".join(phrases[i] for i in order[3:5]), "serif", size, w - 2 * margin):
            canvas.write_text(margin, y, line, font_name="serif", size=size, color=ink)
            y += spacing
        canvas = canvas.scan_tilt(float(self.rng.uniform(-0.8, 0.8)), background=(255, 255, 255))
        gray = salt_pepper(canvas.img, self.rng, 0.0004).convert("L")
        bw = gray.point(lambda v: 255 if v > 140 else 0).convert("1", dither=Image.Dither.NONE)
        path = "tiff/escaneo_bn.tif"
        save_image(bw, self.ctx.path(path), "tiff", compression="group4", dpi=(200, 200))
        _mark_degradation(canvas.elements, ["skew", "salt_pepper", "binarized_1bit"])
        self.files.append(
            image_file_entry(
                id="img_tiff_escaneo_bn",
                path=path,
                format="tiff",
                category=CAT_TIFF,
                description="Escaneo de 1 bit a 200 ppp con compresión CCITT Grupo 4 (acta con datos de dos personas)",
                canvas=canvas,
                tags={"dpi": 200, "mode": "1", "compression": "group4"},
            )
        )


# ---------------------------------------------------------------------------
# Metadata utilities
# ---------------------------------------------------------------------------


def _mark_degradation(elements: list[Element], degradation: list[str]) -> None:
    """Records the degradation on each element (for the recall breakdown by degradation)."""
    for e in elements:
        e.tags.setdefault("degradation", list(degradation))


def _dms(value: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Decimal degrees -> (degrees, minutes, seconds) as EXIF rationals (unsigned)."""
    v = abs(value)
    g = int(v)
    m = int((v - g) * 60)
    s = (v - g - m / 60) * 3600
    return ((g, 1), (m, 1), (int(round(s * 1000)), 1000))


def _xmp(creator: str) -> bytes:
    packet = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        f"   <dc:creator><rdf:Seq><rdf:li>{escape(creator)}</rdf:li></rdf:Seq></dc:creator>\n"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )
    return packet.encode("utf-8")


def _app13_iptc(fields: list[tuple[int, int, bytes]]) -> bytes:
    """JPEG APP13 segment (Photoshop 3.0, resource 0x0404) with the given IPTC datasets."""
    data = b"".join(b"\x1c" + bytes([r, d]) + struct.pack(">H", len(v)) + v for r, d, v in fields)
    resource = b"8BIM" + struct.pack(">H", 0x0404) + b"\x00\x00" + struct.pack(">I", len(data)) + data
    if len(data) % 2:
        resource += b"\x00"
    payload = b"Photoshop 3.0\x00" + resource
    return b"\xff\xed" + struct.pack(">H", len(payload) + 2) + payload


def _insert_jpeg_segment(jpeg: bytes, segment: bytes) -> bytes:
    """Inserts ``segment`` after the leading APPn segments of the JPEG."""
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    i = 2
    while jpeg[i] == 0xFF and 0xE0 <= jpeg[i + 1] <= 0xEF:
        i += 2 + struct.unpack(">H", jpeg[i + 2 : i + 4])[0]
    return jpeg[:i] + segment + jpeg[i:]


def _fit(canvas: Canvas, text: str, font_name: str, size: int, width: float) -> list[str]:
    """Splits ``text`` into lines that fit in ``width`` pixels."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and canvas.text_width(candidate, font_name, size) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def generate(ctx: Context) -> list[FileEntry]:
    g = _Generator(ctx)
    g.rotated()
    g.exif()
    g.tiff()
    return g.files
