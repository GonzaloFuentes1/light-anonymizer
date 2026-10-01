"""Scanned PDFs: pages that are images, with no text layer (or with an invisible OCR layer).

Each page is drawn with ``Canvas`` at the scanner resolution (A4 = 8.27 x 11.69 inches),
degraded the way a scanner would (paper, skew, blur, noise, JPEG, gray, fax binarization)
and inserted as a full-page image into a 595 x 842 pt PDF page. The ground truth is carried
from pixels to points with ``map_to_rect`` over the same box the image was inserted into
(equivalent to ``px_to_pt`` with the effective resolution).

Files:

1. ``informe_300dpi.pdf``: 2-page fee report, 300 dpi, gray, skewed 1.2°.
2. ``informe_200dpi_torcido.pdf``: 200 dpi, -2.5°, noise 10, blur 1.0, JPEG 55.
3. ``ficha_con_foto.pdf``: application form with a pasted photo and form fields.
4. ``escaneo_invertido.pdf``: upside-down page and a landscape sheet scanned sideways.
5. ``mixto_texto_y_escaneo.pdf``: page with a real text layer and a scanned page.
6. ``sandwich_ocr.pdf``: scanned page with an invisible OCR layer (render mode 3) on top.
7. ``fax_150dpi.pdf``: fax binarized to 1 bit with dithering, 150 dpi (everything at stress level).
8. ``timbre_y_firma.pdf``: translucent blue stamp, signature and handwritten note with a phone number.
"""

from __future__ import annotations

import copy
import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from test_bench.canvas import (
    Canvas,
    blur,
    bounding_box,
    font_path,
    map_to_rect,
    noise,
    paper_texture,
    rect,
    rotation_matrix,
    salt_pepper,
    transform_elements,
    transform_points,
)
from test_bench.context import Context
from test_bench.fake_data import (
    ADMINISTRATIVE_PHRASES,
    EMAIL_FORMATS,
    PHONE_FORMATS,
    PHONE_FORMATS_BY_KIND,
    RUT_FORMATS,
    FakeData,
    Person,
    format_rut,
)
from test_bench.schema import Element, FileEntry, Page, Polygon

SEED_NAME = "pdf_escaneado"
CATEGORY = "pdf_scanned"
PREFIX = "pdfe_"

A4_INCHES = (8.27, 11.69)
A4_PT = (595.0, 842.0)
INK = (25, 25, 25)
GRAY_INK = (95, 95, 95)
STAMP_BLUE = (38, 72, 190)
PEN_BLUE = (20, 30, 90)
SCANNER_BACKGROUND = (236, 236, 232)


# ---------------------------------------------------------------------------
# Text segments: a piece of a line with its type, level and tags
# ---------------------------------------------------------------------------


@dataclass
class Seg:
    text: str
    type: str = "text"
    level: str = "base"
    tags: dict[str, Any] = field(default_factory=dict)


def _list_level(p: Person) -> str:
    return "base" if p.in_list else "out_of_scope"


def s_name(p: Person, variant: str = "full") -> Seg:
    text = {
        "full": p.full_name,
        "uppercase": p.full_name.upper(),
        "surnames_first": f"{p.paternal_surname} {p.maternal_surname}, {p.given_names}",
    }[variant]
    return Seg(text, "name", _list_level(p), {"in_list": p.in_list, "variant": variant})


def s_address(p: Person) -> Seg:
    return Seg(p.address, "address", _list_level(p), {"in_list": p.in_list, "variant": "full"})


def s_rut(p: Person, fmt: str, level: str | None = None) -> Seg:
    return Seg(p.rut(fmt), "rut", level or RUT_FORMATS[fmt], {"format": fmt, "dv_valid": p.rut_dv_valid})


def s_loose_rut(body: int, dv: str, fmt: str, dv_valid: bool = True) -> Seg:
    return Seg(format_rut(body, dv, fmt), "rut", RUT_FORMATS[fmt], {"format": fmt, "dv_valid": dv_valid})


def s_mobile(p: Person, fmt: str) -> Seg:
    return Seg(p.phone.format(fmt), "phone", PHONE_FORMATS[fmt], {"format": fmt, "kind": "mobile"})


def s_landline(p: Person, rng: np.random.Generator, base_only: bool = True) -> Seg:
    kind = p.landline.kind
    options = [f for f in PHONE_FORMATS_BY_KIND[kind] if not base_only or PHONE_FORMATS[f] == "base"]
    fmt = str(rng.choice(options))
    return Seg(p.landline.format(fmt), "phone", PHONE_FORMATS[fmt], {"format": fmt, "kind": kind})


def s_email(p: Person, fmt: str) -> Seg:
    return Seg(p.email(fmt), "email", EMAIL_FORMATS[fmt], {"format": fmt})


def s_amount(f: FakeData) -> Seg:
    return Seg(f.amount(), "text", "base", {"decoy": "amount"})


def _pesos(n: int) -> str:
    return f"$ {n:,}".replace(",", ".")


def s_date(text: str) -> Seg:
    return Seg(text, "text", "base", {"decoy": "date"})


# ---------------------------------------------------------------------------
# Sheet: one-page canvas at the scanner resolution, positions in millimetres
# ---------------------------------------------------------------------------


@dataclass
class WrittenLine:
    """A drawn line (to rebuild the invisible OCR layer of the sandwich PDF)."""

    x: float  # px, baseline origin
    y: float
    text: str
    size: int  # px
    font: str


class Sheet:
    """A4 page (portrait or landscape) at ``dpi`` dots per inch, with positions in mm."""

    def __init__(self, dpi: int, landscape: bool = False) -> None:
        self.dpi = dpi
        w, h = (int(round(A4_INCHES[0] * dpi)), int(round(A4_INCHES[1] * dpi)))
        if landscape:
            w, h = h, w
        self.canvas = Canvas.new(w, h)
        self.lines: list[WrittenLine] = []
        self.width_mm = w / dpi * 25.4
        self.height_mm = h / dpi * 25.4

    def px(self, mm: float) -> float:
        return mm * self.dpi / 25.4

    def size(self, pt: float) -> int:
        return max(6, int(round(pt * self.dpi / 72)))

    def text_width_mm(self, text: str, pt: float = 11, font: str = "sans") -> float:
        return self.canvas.text_width(text, font, self.size(pt)) * 25.4 / self.dpi

    def line(
        self,
        x_mm: float,
        y_mm: float,
        segs: list[Seg],
        *,
        pt: float = 11,
        font: str = "sans",
        color: tuple[int, int, int] = INK,
    ) -> float:
        """Writes the segments one after another on a baseline; returns the final x in mm."""
        size = self.size(pt)
        x, y = self.px(x_mm), self.px(y_mm)
        x_start = x
        for s in segs:
            if s.text.strip():
                self.canvas.write_text(
                    x,
                    y,
                    s.text,
                    type=s.type,
                    value=s.text.strip(),
                    font_name=font,
                    size=size,
                    color=color,
                    level=s.level,
                    tags={**s.tags, "size_pt": pt, "dpi": self.dpi},
                )
            x += self.canvas.text_width(s.text, font, size)
        self.lines.append(WrittenLine(x_start, y, "".join(s.text for s in segs), size, font))
        return x * 25.4 / self.dpi

    def text(self, x_mm: float, y_mm: float, text: str, **kw: Any) -> float:
        return self.line(x_mm, y_mm, [Seg(text)], **kw)

    def centered(self, y_mm: float, text: str, *, pt: float = 11, font: str = "sans", **kw: Any) -> float:
        w = self.text_width_mm(text, pt, font)
        return self.line((self.width_mm - w) / 2, y_mm, [Seg(text)], pt=pt, font=font, **kw)

    def rule(self, x0: float, y0: float, x1: float, y1: float, thickness_mm: float = 0.25, color=INK) -> None:
        d = ImageDraw.Draw(self.canvas.img)
        d.line(
            [(self.px(x0), self.px(y0)), (self.px(x1), self.px(y1))],
            fill=color,
            width=max(1, int(self.px(thickness_mm))),
        )

    def box(self, x0: float, y0: float, x1: float, y1: float, thickness_mm: float = 0.25, color=INK) -> None:
        d = ImageDraw.Draw(self.canvas.img)
        d.rectangle(
            [self.px(x0), self.px(y0), self.px(x1), self.px(y1)],
            outline=color,
            width=max(1, int(self.px(thickness_mm))),
        )

    def table(
        self,
        x_mm: float,
        y_mm: float,
        widths_mm: list[float],
        rows: list[list[Seg]],
        *,
        row_height_mm: float = 7.0,
        pt: float = 10,
    ) -> float:
        """Table with borders; the first row is a bold header. Returns the final y in mm."""
        x_end = x_mm + sum(widths_mm)
        for i, row in enumerate(rows):
            y0 = y_mm + i * row_height_mm
            self.rule(x_mm, y0, x_end, y0)
            x = x_mm
            for width, cell in zip(widths_mm, row, strict=True):
                font = "sans_bold" if i == 0 else "sans"
                self.line(x + 1.5, y0 + row_height_mm * 0.7, [cell], pt=pt, font=font)
                x += width
        y_end = y_mm + len(rows) * row_height_mm
        self.rule(x_mm, y_end, x_end, y_end)
        x = x_mm
        for width in [0.0, *widths_mm]:
            x += width
            self.rule(x, y_mm, x, y_end)
        return y_end


# ---------------------------------------------------------------------------
# Scanning: degradations and conversion to PDF
# ---------------------------------------------------------------------------


@dataclass
class Degradation:
    skew: float = 0.0  # degrees, counterclockwise
    rotation: int = 0  # 0, 90, 180, 270: sheet placed sideways or upside down on the scanner
    noise: float = 4.0
    blur: float = 0.6
    quality: int = 80  # JPEG
    gray: bool = False
    salt_pepper: float = 0.0
    binarize: bool = False  # fax: 1 bit with Floyd-Steinberg dithering (saved losslessly)
    paper: bool = True

    def names(self) -> list[str]:
        out = []
        if self.paper:
            out.append("paper")
        if self.rotation:
            out.append(f"rotation_{self.rotation}")
        if self.skew:
            out.append("skew")
        if self.blur:
            out.append("blur")
        if self.noise:
            out.append("noise")
        if self.salt_pepper:
            out.append("salt_pepper")
        if self.binarize:
            out.append("binarized")
        else:
            out.append("jpeg")
        if self.gray:
            out.append("gray")
        return out


@dataclass
class ScannedPage:
    data: bytes
    width: int
    height: int
    elements: list[Element]  # in pixels of the final image
    matrix: np.ndarray  # sheet pixels -> final image pixels


def _rotation_matrix(w: int, h: int, degrees: int) -> np.ndarray:
    """Same matrix that ``Canvas.rotate`` uses for multiples of 90."""
    return {
        0: np.eye(3),
        90: np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], dtype=np.float64),
        180: np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]], dtype=np.float64),
        270: np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
    }[degrees % 360]


def scan(sheet: Sheet, rng: np.random.Generator, d: Degradation) -> ScannedPage:
    canvas = sheet.canvas
    if d.paper:
        # texture at half resolution and upscaled: paper grain does not need 300 dpi
        # and paper_texture at full size costs several seconds per page
        small_paper = paper_texture((canvas.width + 1) // 2, (canvas.height + 1) // 2, rng)
        paper = np.asarray(small_paper.resize((canvas.width, canvas.height), Image.Resampling.BILINEAR), np.float32)
        arr = np.asarray(canvas.img, np.float32) * (paper / 255.0)
        canvas = Canvas(Image.fromarray(arr.clip(0, 255).astype(np.uint8)), canvas.elements)
    m = np.eye(3)
    if d.rotation % 360:
        m = _rotation_matrix(canvas.width, canvas.height, d.rotation) @ m
        canvas = canvas.rotate(d.rotation)
    if d.skew:
        m = rotation_matrix(canvas.width, canvas.height, d.skew, expand=False)[0] @ m
        canvas = canvas.scan_tilt(d.skew, background=SCANNER_BACKGROUND)
    img = canvas.img
    if d.gray or d.binarize:
        img = img.convert("L")
    if d.blur:
        img = blur(img, d.blur)
    if d.noise:
        img = noise(img, rng, d.noise)
    if d.binarize:
        # fax levels: the background saturates to white and dithering stays on edges and grays
        arr = (np.asarray(img, np.float32) - 60.0) * (255.0 / (222.0 - 60.0))
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if d.salt_pepper:
        img = salt_pepper(img, rng, d.salt_pepper)
    buf = io.BytesIO()
    if d.binarize:
        img.convert("1").save(buf, "PNG", optimize=True)
    else:
        img.save(buf, "JPEG", quality=d.quality, optimize=True)
    elements = copy.deepcopy(canvas.elements)
    for e in elements:
        e.layer = "raster"
        e.tags["degradation"] = d.names()
    return ScannedPage(buf.getvalue(), img.width, img.height, elements, m)


def scanned_page(doc: pymupdf.Document, scanned: ScannedPage) -> tuple[pymupdf.Page, list[Element]]:
    """Adds an A4 page with the full-page image and returns the elements in points."""
    index = doc.page_count
    page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
    page.insert_image(page.rect, stream=scanned.data, keep_proportion=False)
    elements = map_to_rect(scanned.elements, scanned.width, scanned.height, (0.0, 0.0, *A4_PT))
    for e in elements:
        e.page = index
    return page, elements


def _pages(doc: pymupdf.Document) -> list[Page]:
    return [
        Page(index=i, width=p.rect.width, height=p.rect.height, unit="pt", rotation=p.rotation)
        for i, p in enumerate(doc)
    ]


def _save(doc: pymupdf.Document, ctx: Context, path: str) -> None:
    doc.set_metadata({"creator": "Escaner de oficina ficticio", "producer": "banco_pruebas"})
    # no_new_id: no random /ID in the trailer, so the file is identical on every run
    doc.save(ctx.path(path), garbage=3, deflate=True, no_new_id=True)
    doc.close()


def _file_entry(
    id: str, path: str, description: str, doc: pymupdf.Document, elements: list[Element], **tags: Any
) -> FileEntry:
    return FileEntry(
        id=PREFIX + id,
        path=path,
        format="pdf",
        category=CATEGORY,
        description=description,
        pages=_pages(doc),
        elements=elements,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Text in the PDF (real text layer or invisible OCR layer), with GT from search_for
# ---------------------------------------------------------------------------


def _quad_to_polygon(q: pymupdf.Quad) -> Polygon:
    return [[round(p.x, 3), round(p.y, 3)] for p in (q.ul, q.ur, q.lr, q.ll)]


def _center(polygon: Polygon) -> tuple[float, float]:
    x0, y0, x1, y1 = bounding_box(polygon)
    return (x0 + x1) / 2, (y0 + y1) / 2


def _find_near(page: pymupdf.Page, value: str, near: tuple[float, float]) -> Polygon:
    """Quad of ``page.search_for(value)`` closest to ``near`` (neutral texts repeat)."""
    hits = page.search_for(value, quads=True)
    if not hits:
        raise RuntimeError(f"{value!r} not found on page {page.number}")
    best = min(hits, key=lambda q: math.dist(_center(_quad_to_polygon(q)), near))
    polygon = _quad_to_polygon(best)
    if math.dist(_center(polygon), near) > 12:
        raise RuntimeError(f"{value!r} ended up far from where it was written ({polygon} vs {near})")
    return polygon


class TextPage:
    """Writes lines with ``insert_text`` and records each segment with the quad from ``search_for``."""

    ALIAS = {"sans": "dvs", "sans_bold": "dvsb", "serif": "dvse", "mono": "dvm"}

    def __init__(self, page: pymupdf.Page, layer: str = "text", render_mode: int = 0) -> None:
        self.page = page
        self.layer = layer
        self.render_mode = render_mode
        self._fonts: dict[str, pymupdf.Font] = {}
        self._pending: list[tuple[Seg, tuple[float, float], dict[str, Any]]] = []

    def _font(self, name: str) -> tuple[str, pymupdf.Font]:
        alias = self.ALIAS[name]
        if name not in self._fonts:
            self.page.insert_font(fontname=alias, fontfile=str(font_path(name)))
            self._fonts[name] = pymupdf.Font(fontfile=str(font_path(name)))
        return alias, self._fonts[name]

    def line(
        self,
        x: float,
        y: float,
        segs: list[Seg],
        *,
        pt: float = 11,
        font: str = "sans",
        angle: float = 0.0,
        record: bool = True,
        tags: dict[str, Any] | None = None,
    ) -> None:
        alias, f = self._font(font)
        text = "".join(s.text for s in segs)
        point = pymupdf.Point(x, y)
        morph = (point, pymupdf.Matrix(angle)) if angle else None
        self.page.insert_text(
            point, text, fontsize=pt, fontname=alias, color=(0.1, 0.1, 0.1), render_mode=self.render_mode, morph=morph
        )
        if not record:
            return
        t = math.radians(angle)
        dx, dy = math.cos(t), -math.sin(t)  # baseline direction (y points down)
        advance = 0.0
        for s in segs:
            width = f.text_length(s.text, fontsize=pt)
            if s.text.strip():
                middle = advance + f.text_length(s.text.rstrip(), fontsize=pt) / 2
                rise = (f.ascender + f.descender) / 2 * pt  # vertical center of the search_for box
                cx = x + dx * middle + dy * rise
                cy = y + dy * middle - dx * rise
                self._pending.append((s, (cx, cy), {"size_pt": pt, "font": font, **(tags or {})}))
            advance += width

    def resolve(self) -> list[Element]:
        out = []
        for s, near, extra in self._pending:
            value = s.text.strip()
            out.append(
                Element(
                    type=s.type,
                    page=self.page.number,
                    polygon=_find_near(self.page, value, near),
                    value=value,
                    level=s.level,
                    layer=self.layer,
                    tags={**s.tags, **extra},
                )
            )
        self._pending.clear()
        return out


# ---------------------------------------------------------------------------
# Contents
# ---------------------------------------------------------------------------


def _header(h: Sheet, f: FakeData, title: str, subtitle: str | None = None) -> None:
    h.text(20, 18, "GOBIERNO REGIONAL DE EJEMPLO", pt=11, font="sans_bold")
    h.text(20, 23, "División de Fomento e Industria", pt=9, color=GRAY_INK)
    h.line(145, 18, [Seg("Folio N° "), Seg(f.folio())], pt=10)
    h.rule(20, 27, h.width_mm - 20, 27, 0.35)
    h.centered(38, title, pt=14, font="sans_bold")
    if subtitle:
        h.centered(45, subtitle, pt=10.5)


def _report_page1(
    h: Sheet, f: FakeData, rng: np.random.Generator, p: Person, formats: dict[str, str], pt: float = 11
) -> None:
    _header(h, f, "INFORME MENSUAL DE ACTIVIDADES", "Contrato de prestación de servicios a honorarios")
    y = 58.0
    h.text(20, y, "1. ANTECEDENTES DEL PRESTADOR", pt=pt, font="sans_bold")
    step = pt * 0.6
    y += step + 2
    h.line(24, y, [Seg("Nombre: "), s_name(p, formats.get("name", "full"))], pt=pt)
    y += step
    h.line(24, y, [Seg("RUT: "), s_rut(p, formats["rut"])], pt=pt)
    y += step
    h.line(24, y, [Seg("Domicilio: "), s_address(p), Seg(f", {p.commune}")], pt=pt)
    y += step
    x = h.line(24, y, [Seg("Teléfono: "), s_mobile(p, formats["mobile"])], pt=pt)
    h.line(max(x + 8, 110), y, [Seg("Correo: "), s_email(p, formats["email"])], pt=pt)
    y += step
    h.line(24, y, [Seg("Teléfono fijo: "), s_landline(p, rng)], pt=pt)
    y += step + 6
    h.text(20, y, "2. PRODUCTOS COMPROMETIDOS", pt=pt, font="sans_bold")
    y += step + 2
    for phrase in rng.choice(ADMINISTRATIVE_PHRASES[:8], 4, replace=False):
        h.text(24, y, str(phrase), pt=pt - 1)
        y += step
    y += 6
    h.text(20, y, "3. MONTOS DEL PERÍODO", pt=pt, font="sans_bold")
    y += step + 2
    gross = s_amount(f)
    n_gross = int(gross.text.removeprefix("$ ").replace(".", ""))
    withholding = int(round(n_gross * 0.1525))
    for label, amount in (
        ("Monto bruto: ", gross),
        ("Retención de impuesto (15,25%): ", Seg(_pesos(withholding), tags={"decoy": "amount"})),
        ("Monto líquido a pagar: ", Seg(_pesos(n_gross - withholding), tags={"decoy": "amount"})),
    ):
        h.line(24, y, [Seg(label), amount], pt=pt)
        y += step
    h.line(24, y, [Seg("Fecha de pago estimada: "), s_date(f.date())], pt=pt)
    y += step + 6
    for phrase in ADMINISTRATIVE_PHRASES[8:11]:
        h.text(20, y, phrase, pt=pt - 1.5)
        y += step - 0.5
    # tiny footer: email at 5 pt (stress level because of its size)
    h.line(
        20,
        h.height_mm - 14,
        [
            Seg("Consultas sobre este informe: "),
            Seg(p.email("plus"), "email", "stress", {"format": "plus", "tiny": True}),
        ],
        pt=5,
        color=GRAY_INK,
    )


def _report_page2(h: Sheet, f: FakeData, rng: np.random.Generator, q: Person, pt: float = 11) -> None:
    step = pt * 0.6
    h.text(20, 22, "4. DETALLE DE ACTIVIDADES REALIZADAS", pt=pt, font="sans_bold")
    activities = [
        "Reunión de coordinación",
        "Taller con organizaciones",
        "Visita a terreno",
        "Elaboración de informe",
        "Mesa técnica regional",
    ]
    communes = rng.choice(["Talca", "Chillán", "Temuco", "Valdivia", "Rancagua", "Copiapó"], 5, replace=False)
    rows = [[Seg("N°"), Seg("Actividad"), Seg("Fecha"), Seg("Comuna"), Seg("Monto")]]
    for i, (activity, commune) in enumerate(zip(activities, communes, strict=True), 1):
        rows.append([Seg(str(i)), Seg(activity), s_date(f.date()), Seg(str(commune)), s_amount(f)])
    y = h.table(20, 27, [10, 56, 52, 24, 28], rows, row_height_mm=7.5, pt=pt - 1.5)
    y += 12
    h.text(20, y, "5. CONTRAPARTE TÉCNICA", pt=pt, font="sans_bold")
    y += step + 2
    h.line(24, y, [Seg("Nombre: "), s_name(q, "uppercase")], pt=pt)
    y += step
    h.line(24, y, [Seg("RUT: "), s_rut(q, "no_dots")], pt=pt)
    y += step
    h.line(24, y, [Seg("Correo institucional: "), s_email(q, "initial")], pt=pt)
    y += step
    h.line(24, y, [Seg("Anexo telefónico: "), s_landline(q, rng)], pt=pt)
    y += step + 6
    h.text(20, y, "6. VALIDACIÓN", pt=pt, font="sans_bold")
    y += step + 2
    h.text(24, y, ADMINISTRATIVE_PHRASES[3], pt=pt - 1)
    y += step
    h.text(24, y, ADMINISTRATIVE_PHRASES[4], pt=pt - 1)
    y += 30
    h.rule(30, y, 90, y)
    h.rule(120, y, 180, y)
    h.text(38, y + 5, "Firma del prestador", pt=9)
    h.text(126, y + 5, "Firma contraparte técnica", pt=9)
    h.centered(h.height_mm - 12, "Página 2 de 2", pt=8, color=GRAY_INK)


def _gen_report_300(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    path = "pdf_scanned/informe_300dpi.pdf"
    p = f.person()
    q = f.person(in_list=False)
    d = Degradation(skew=1.2, noise=5, blur=0.6, quality=78, gray=True)
    doc = pymupdf.open()
    elements: list[Element] = []
    h1 = Sheet(300)
    _report_page1(h1, f, rng, p, {"rut": "dots", "mobile": "mobile_international", "email": "dot"})
    h2 = Sheet(300)
    _report_page2(h2, f, rng, q)
    for h in (h1, h2):
        elements += scanned_page(doc, scan(h, rng, d))[1]
    a = _file_entry(
        "informe_300dpi",
        path,
        "Informe de honorarios escaneado a 300 ppp en gris, inclinado 1,2°; contraparte fuera de la lista.",
        doc,
        elements,
        dpi=300,
        degradation=d.__dict__,
    )
    _save(doc, ctx, path)
    return a


def _gen_report_200_skewed(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    path = "pdf_scanned/informe_200dpi_torcido.pdf"
    p = f.person(dv_valid=False)
    d = Degradation(skew=-2.5, noise=10, blur=1.0, quality=55, salt_pepper=0.0005)
    doc = pymupdf.open()
    h = Sheet(200)
    _report_page1(
        h,
        f,
        rng,
        p,
        {"rut": "spaced_hyphen", "mobile": "mobile_parentheses", "email": "underscore", "name": "surnames_first"},
        pt=10.5,
    )
    _, elements = scanned_page(doc, scan(h, rng, d))
    a = _file_entry(
        "informe_200dpi_torcido",
        path,
        "Informe de honorarios a 200 ppp, -2,5°, ruido 10, desenfoque 1,0 y JPEG calidad 55; RUT con DV inválido.",
        doc,
        elements,
        dpi=200,
        degradation=d.__dict__,
    )
    _save(doc, ctx, path)
    return a


def _field(h: Sheet, x: float, y: float, width: float, label: str, value: Seg, pt: float = 11) -> None:
    """Form field: small label on top and a box with the typewritten value."""
    h.text(x, y, label, pt=7.5, color=GRAY_INK)
    h.box(x, y + 1.2, x + width, y + 8.2, 0.2, (90, 90, 90))
    h.line(x + 1.8, y + 6.2, [value], pt=pt, font="mono")


def _gen_record_card(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    path = "pdf_scanned/ficha_con_foto.pdf"
    p = f.person()
    emergency = f.person(in_list=False)
    h = Sheet(300)
    _header(h, f, "FICHA DE POSTULACIÓN", "Programa Regional de Becas de Especialización 2026")
    # ID-size photo (35 x 45 mm) pasted in the corner
    (face,) = ctx.faces.take("front", 1)
    fx, fy, fw = 152.0, 52.0, 35.0
    h.canvas.paste(
        face.img,
        h.px(fx),
        h.px(fy),
        width=int(round(h.px(fw))),
        elements=ctx.faces.elements(face, "base", {"origin": "scan", "id_photo": True}),
    )
    photo_height = face.img.height * fw / face.img.width
    h.box(fx - 1, fy - 1, fx + fw + 1, fy + photo_height + 1, 0.3)
    h.text(20, 56, "1. ANTECEDENTES PERSONALES", pt=11, font="sans_bold")
    _field(h, 20, 62, 125, "Nombre completo", s_name(p, "uppercase"))
    _field(h, 20, 74, 60, "RUT", s_rut(p, "dots"))
    _field(h, 85, 74, 60, "Fecha de nacimiento", s_date(p.birth_date))
    _field(h, 20, 86, 125, "Dirección", s_address(p))
    _field(h, 20, 98, 60, "Comuna", Seg(p.commune))
    _field(h, 85, 98, 60, "Sexo", Seg("Femenino" if p.sex == "F" else "Masculino"))
    _field(h, 20, 110, 80, "Teléfono móvil", s_mobile(p, "mobile_national"))
    _field(h, 105, 110, 82, "Teléfono fijo", s_landline(p, rng))
    _field(h, 20, 122, 167, "Correo electrónico", s_email(p, "with_year"))
    h.text(20, 142, "2. ANTECEDENTES ACADÉMICOS", pt=11, font="sans_bold")
    _field(h, 20, 148, 167, "Institución", Seg("Universidad de Ejemplo del Sur"))
    _field(h, 20, 160, 110, "Título profesional", Seg("Ingeniería en Recursos Naturales"))
    _field(h, 135, 160, 52, "Año de titulación", Seg(str(int(rng.integers(2010, 2024)))))
    h.text(20, 180, "3. CONTACTO EN CASO DE EMERGENCIA", pt=11, font="sans_bold")
    _field(h, 20, 186, 110, "Nombre", s_name(emergency, "full"))
    _field(h, 135, 186, 52, "Parentesco", Seg("Hermana" if emergency.sex == "F" else "Hermano"))
    _field(h, 20, 198, 80, "Teléfono", s_mobile(emergency, "mobile_block"))
    y = 218.0
    for checked, text in (
        (True, "Declaro que la información entregada es fidedigna."),
        (True, "Autorizo el uso de mis datos para fines del proceso de selección."),
        (False, "Solicito ser notificado por carta certificada."),
    ):
        h.box(20, y - 3.4, 23.6, y + 0.2, 0.25)
        if checked:
            h.rule(20.6, y - 1.6, 21.8, y - 0.4, 0.4)
            h.rule(21.8, y - 0.4, 23.2, y - 3.0, 0.4)
        h.text(26, y, text, pt=10)
        y += 7
    y += 18
    h.rule(25, y, 90, y)
    h.text(35, y + 5, "Firma del postulante", pt=9)
    h.line(120, y, [Seg("Fecha: "), s_date(f.date())], pt=10)
    d = Degradation(skew=0.7, noise=4, blur=0.5, quality=80)
    doc = pymupdf.open()
    _, elements = scanned_page(doc, scan(h, rng, d))
    a = _file_entry(
        "ficha_con_foto",
        path,
        "Ficha de postulación escaneada a 300 ppp con foto carné pegada y campos de formulario.",
        doc,
        elements,
        dpi=300,
        degradation=d.__dict__,
    )
    _save(doc, ctx, path)
    return a


def _gen_inverted(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    path = "pdf_scanned/escaneo_invertido.pdf"
    doc = pymupdf.open()
    elements: list[Element] = []
    # Page 1: certificate scanned upside down
    p = f.person()
    h = Sheet(200)
    _header(h, f, "CERTIFICADO DE PARTICIPACIÓN")
    y = 62.0
    h.line(20, y, [Seg("Se certifica que "), s_name(p), Seg(",")], pt=11.5)
    y += 7
    h.line(20, y, [Seg("cédula de identidad N° "), s_rut(p, "dots"), Seg(", con domicilio en")], pt=11.5)
    y += 7
    h.line(20, y, [s_address(p), Seg(f", comuna de {p.commune},")], pt=11.5)
    y += 7
    h.text(20, y, "participó en el programa de capacitación en gestión territorial.", pt=11.5)
    y += 12
    h.line(20, y, [Seg("Contacto: "), s_mobile(p, "mobile_hyphens"), Seg(" / "), s_email(p, "subdomain")], pt=11.5)
    y += 12
    h.line(20, y, [Seg("Se extiende el presente certificado con fecha "), s_date(f.date()), Seg(".")], pt=11.5)
    y += 40
    h.rule(70, y, 140, y)
    h.centered(y + 5, "Coordinación del Programa", pt=9)
    d1 = Degradation(rotation=180, skew=0.6, noise=5, blur=0.6, quality=75, gray=True)
    elements += scanned_page(doc, scan(h, rng, d1))[1]
    # Page 2: landscape sheet scanned sideways (content rotated 90° on a portrait page)
    h = Sheet(200, landscape=True)
    h.text(15, 16, "GOBIERNO REGIONAL DE EJEMPLO", pt=10, font="sans_bold")
    h.centered(26, "PLANILLA DE ASISTENCIA – TALLER DE FORMULACIÓN DE PROYECTOS", pt=12.5, font="sans_bold")
    h.line(15, 34, [Seg("Fecha: "), s_date(f.date()), Seg("    Lugar: Salón auditorio")], pt=10)
    rows: list[list[Seg]] = [[Seg("N°"), Seg("Nombre"), Seg("RUT"), Seg("Teléfono"), Seg("Correo"), Seg("Firma")]]
    rut_formats = ["dots", "no_dots", "dots", "en_dash", "no_dots", "dots"]
    phone_formats = ["mobile_national", "mobile_international", "mobile_block", "mobile_compact", "mobile_national"]
    signers = []
    for i in range(6):
        per = f.person(in_list=i not in (2, 4), seven_digits=i == 3)
        signers.append(per)
        if i == 5:
            body, dv = f.rut_with_k()
            rut = s_loose_rut(body, dv, "dots")
            tel = s_landline(per, rng)
        else:
            rut = s_rut(per, rut_formats[i])
            tel = s_mobile(per, phone_formats[i])
        rows.append([Seg(str(i + 1)), s_name(per), rut, tel, s_email(per, "dot"), Seg("")])
    widths = [9, 72, 36, 40, 74, 36]
    y0 = 42.0
    h.table(15, y0, widths, rows, row_height_mm=11, pt=9.5)
    x_signature = 15 + sum(widths[:-1]) + 3
    for i, per in enumerate(signers):
        yb = y0 + (i + 1) * 11 + 8
        h.canvas.write_irregular(
            h.px(x_signature),
            h.px(yb),
            f"{per.given_names[0]}. {per.paternal_surname}",
            rng,
            type="signature",
            font_name="stix_italic",
            size=h.size(12),
            level="out_of_scope",
            tags={"handwritten": True},
        )
    d2 = Degradation(rotation=90, skew=-0.8, noise=5, blur=0.6, quality=75, gray=True)
    elements += scanned_page(doc, scan(h, rng, d2))[1]
    a = _file_entry(
        "escaneo_invertido",
        path,
        "Página 1 escaneada cabeza abajo (180°); página 2 planilla apaisada escaneada de lado (90°), sin /Rotate.",
        doc,
        elements,
        dpi=200,
        degradation=[d1.__dict__, d2.__dict__],
    )
    _save(doc, ctx, path)
    return a


def _gen_mixed(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    path = "pdf_scanned/mixto_texto_y_escaneo.pdf"
    doc = pymupdf.open()
    # Page 1: official letter with a real text layer
    p = f.person()
    page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
    tp = TextPage(page)
    tp.line(57, 60, [Seg("GOBIERNO REGIONAL DE EJEMPLO")], pt=11, font="sans_bold")
    tp.line(57, 74, [Seg("División de Administración y Finanzas")], pt=9)
    tp.line(340, 100, [Seg("ORD. N° "), Seg(f.folio())], pt=10.5, font="sans_bold")
    tp.line(340, 115, [Seg("ANT.: Solicitud de acceso a la información")], pt=9.5)
    tp.line(340, 128, [Seg("MAT.: Remite antecedentes solicitados")], pt=9.5)
    tp.line(340, 150, [Seg("Concepción, "), s_date(f.date())], pt=9.5)
    tp.line(57, 185, [Seg("A: "), s_name(p, "uppercase")], pt=11)
    tp.line(57, 200, [Seg("DE: JEFATURA DIVISIÓN DE ADMINISTRACIÓN Y FINANZAS")], pt=11)
    y = 235.0
    for phrase in (ADMINISTRATIVE_PHRASES[7], ADMINISTRATIVE_PHRASES[2], ADMINISTRATIVE_PHRASES[5]):
        tp.line(57, y, [Seg(phrase)], pt=10.5)
        y += 17
    y += 12
    tp.line(57, y, [Seg("Datos del solicitante registrados en la solicitud:")], pt=10.5)
    y += 20
    tp.line(75, y, [Seg("RUT: "), s_rut(p, "en_dash")], pt=10.5)
    y += 16
    tp.line(75, y, [Seg("Correo de contacto: "), s_email(p, "uppercase")], pt=10.5)
    y += 16
    tp.line(75, y, [Seg("Teléfono: "), s_landline(p, rng)], pt=10.5)
    y += 16
    tp.line(75, y, [Seg("Dirección: "), s_address(p), Seg(f", {p.commune}")], pt=10.5)
    y += 34
    tp.line(57, y, [Seg("Se adjunta como anexo el comprobante de recepción escaneado.")], pt=10.5)
    y += 40
    tp.line(57, y, [Seg("Saluda atentamente,")], pt=10.5)
    elements = tp.resolve()
    # Page 2: scanned annex
    q = f.person()
    h = Sheet(200)
    _header(h, f, "ANEXO N° 1 – COMPROBANTE DE RECEPCIÓN")
    y = 60.0
    h.line(22, y, [Seg("Recibí conforme de "), s_name(q), Seg(",")], pt=11)
    y += 7
    h.line(22, y, [Seg("RUT "), s_rut(q, "no_dots"), Seg(", la documentación indicada.")], pt=11)
    y += 10
    h.line(22, y, [Seg("Teléfono de contacto: "), s_mobile(q, "mobile_hyphens")], pt=11)
    y += 7
    h.line(22, y, [Seg("Correo: "), s_email(q, "subdomain")], pt=11)
    y += 7
    h.line(22, y, [Seg("Fecha de recepción: "), s_date(f.date())], pt=11)
    y += 35
    h.rule(30, y, 95, y)
    h.text(40, y + 5, "Recibido por", pt=9)
    d = Degradation(skew=0.9, noise=5, blur=0.6, quality=72, gray=True)
    elements += scanned_page(doc, scan(h, rng, d))[1]
    a = _file_entry(
        "mixto_texto_y_escaneo",
        path,
        "Página 1 con capa de texto real (oficio); página 2 anexo escaneado sin texto.",
        doc,
        elements,
        dpi=200,
        degradation=d.__dict__,
    )
    _save(doc, ctx, path)
    return a


def _gen_sandwich(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    path = "pdf_scanned/sandwich_ocr.pdf"
    giver = f.person()
    receiver = f.person(dv_valid=False)
    h = Sheet(300)
    _header(h, f, "ACTA DE ENTREGA DE EQUIPAMIENTO", "Programa de Apoyo a Emprendedores Rurales")
    y = 60.0
    h.text(20, y, "ENTREGA", pt=11, font="sans_bold")
    y += 7
    h.line(24, y, [Seg("Nombre: "), s_name(giver)], pt=11)
    y += 6.5
    h.line(24, y, [Seg("RUT: "), s_rut(giver, "dots"), Seg("    Correo: "), s_email(giver, "dot")], pt=11)
    y += 6.5
    h.line(24, y, [Seg("Teléfono: "), s_landline(giver, rng)], pt=11)
    y += 11
    h.text(20, y, "RECIBE", pt=11, font="sans_bold")
    y += 7
    h.line(24, y, [Seg("Nombre: "), s_name(receiver)], pt=11)
    y += 6.5
    h.line(24, y, [Seg("RUT: "), s_rut(receiver, "no_dots")], pt=11)
    y += 6.5
    h.line(24, y, [Seg("Domicilio: "), s_address(receiver), Seg(f", {receiver.commune}")], pt=11)
    y += 6.5
    h.line(24, y, [Seg("Teléfono: "), s_mobile(receiver, "mobile_national")], pt=11)
    y += 6.5
    h.line(24, y, [Seg("Correo: "), s_email(receiver, "with_year")], pt=11)
    y += 11
    h.text(20, y, "DETALLE", pt=11, font="sans_bold")
    y += 7
    h.line(24, y, [Seg("Valor total del equipamiento: "), s_amount(f)], pt=11)
    y += 6.5
    h.line(24, y, [Seg("Fecha de entrega: "), s_date(f.date())], pt=11)
    y += 6.5
    for phrase in (ADMINISTRATIVE_PHRASES[11], ADMINISTRATIVE_PHRASES[2], ADMINISTRATIVE_PHRASES[9]):
        h.text(24, y, phrase, pt=10)
        y += 6
    y += 30
    h.rule(25, y, 90, y)
    h.rule(120, y, 185, y)
    h.text(40, y + 5, "Firma entrega", pt=9)
    h.text(138, y + 5, "Firma recibe", pt=9)
    d = Degradation(skew=0.8, noise=5, blur=0.6, quality=72, gray=True)
    scanned = scan(h, rng, d)
    doc = pymupdf.open()
    page, raster = scanned_page(doc, scanned)
    # Invisible OCR layer: each line at the (transformed) position of the line in the image
    sx, sy = A4_PT[0] / scanned.width, A4_PT[1] / scanned.height
    layer = TextPage(page, layer="hidden", render_mode=3)
    for ln in h.lines:
        ((x, y_),) = transform_points([[ln.x, ln.y]], scanned.matrix)
        layer.line(
            x * sx,
            y_ * sy,
            [Seg(ln.text)],
            pt=ln.size * sy,
            font=ln.font if ln.font in TextPage.ALIAS else "sans",
            angle=d.skew,
            record=False,
        )
    hidden = []
    for e in raster:
        e.tags["sandwich"] = True
        e2 = copy.deepcopy(e)
        e2.layer = "hidden"
        e2.polygon = _find_near(page, e.value or "", _center(e.polygon))
        e2.tags = {k: v for k, v in e.tags.items() if k not in ("degradation",)}
        e2.tags["render_mode"] = 3
        hidden.append(e2)
    a = _file_entry(
        "sandwich_ocr",
        path,
        "Acta escaneada a 300 ppp con capa OCR invisible (render mode 3) sobre el texto de la imagen; "
        "cada dato se registra dos veces (raster y oculto).",
        doc,
        raster + hidden,
        dpi=300,
        degradation=d.__dict__,
    )
    _save(doc, ctx, path)
    return a


def _gen_fax(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    path = "pdf_scanned/fax_150dpi.pdf"
    p = f.person()
    office = f.person(in_list=False)
    h = Sheet(150)
    h.line(
        8,
        7,
        [
            Seg("DE: OFICINA DE PARTES   FAX: "),
            s_landline(office, rng),
            Seg("   "),
            s_date(f.date()),
            Seg("   PÁG. 01/01"),
        ],
        pt=8,
        font="mono",
    )
    h.rule(8, 9, h.width_mm - 8, 9, 0.3)
    h.centered(30, "TRANSMISIÓN POR FAX", pt=15, font="sans_bold")
    h.centered(38, "Solicitud de antecedentes para pago de subsidio", pt=11)
    y = 55.0
    h.line(22, y, [Seg("PARA: "), s_name(p, "uppercase")], pt=11, font="mono")
    y += 7
    h.line(22, y, [Seg("RUT: "), s_rut(p, "commas")], pt=11, font="mono")
    y += 7
    h.line(22, y, [Seg("FONO CONTACTO: "), s_mobile(p, "old_mobile_09")], pt=11, font="mono")
    y += 7
    h.line(22, y, [Seg("CORREO: "), s_email(p, "at")], pt=11, font="mono")
    y += 7
    h.line(22, y, [Seg("DIRECCIÓN: "), s_address(p)], pt=11, font="mono")
    y += 14
    for phrase in (ADMINISTRATIVE_PHRASES[4], ADMINISTRATIVE_PHRASES[2], ADMINISTRATIVE_PHRASES[10]):
        h.text(22, y, phrase, pt=10.5, font="serif")
        y += 7
    y += 7
    h.line(22, y, [Seg("Monto del subsidio: "), s_amount(f)], pt=10.5, font="serif")
    y += 7
    h.line(
        22,
        y,
        [Seg("Remitente: "), s_name(office), Seg(" – anexo "), s_landline(office, rng)],
        pt=10.5,
        font="serif",
    )
    for e in h.canvas.elements:
        if e.level == "base":
            e.level = "stress"
        e.tags["fax"] = True
    d = Degradation(skew=0.6, noise=12, blur=0.7, binarize=True, salt_pepper=0.001, paper=False)
    doc = pymupdf.open()
    _, elements = scanned_page(doc, scan(h, rng, d))
    a = _file_entry(
        "fax_150dpi",
        path,
        "Fax a 150 ppp binarizado a 1 bit con tramado; todos los datos en nivel estrés.",
        doc,
        elements,
        dpi=150,
        degradation=d.__dict__,
    )
    _save(doc, ctx, path)
    return a


def _blend_stamp(h: Sheet, stamp: Canvas, x_mm: float, y_mm: float, opacity: float = 0.78) -> None:
    """Multiplies the stamp (blue ink on white) onto the sheet, with partial opacity."""
    px, py = int(round(h.px(x_mm))), int(round(h.px(y_mm)))
    base = np.asarray(h.canvas.img, np.float32).copy()
    t = np.asarray(stamp.img, np.float32) / 255.0
    region = base[py : py + stamp.height, px : px + stamp.width]
    region *= 1 - opacity + opacity * t[: region.shape[0], : region.shape[1]]
    h.canvas.img = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    m = np.array([[1, 0, px], [0, 1, py], [0, 0, 1]], dtype=np.float64)
    for e in transform_elements(stamp.elements, m):
        e.tags["stamp"] = True
        h.canvas.elements.append(e)


def _round_stamp(h: Sheet, s: Person, degrees: float) -> Canvas:
    side = int(round(h.px(46)))
    t = Canvas.new(side, side)
    d = ImageDraw.Draw(t.img)
    g = max(2, int(h.px(0.7)))
    d.ellipse([g, g, side - g, side - g], outline=STAMP_BLUE, width=g)
    d.ellipse([h.px(3), h.px(3), side - h.px(3), side - h.px(3)], outline=STAMP_BLUE, width=max(1, g // 2))
    c = side / 2
    rad = math.radians(degrees)
    perp = (math.sin(rad), math.cos(rad))  # towards the "bottom" of the rotated text
    lines = [
        (Seg("RECEPCIÓN CONFORME"), 9.5, "sans_bold", -1.25),
        (s_name(s), 7.5, "sans_bold", 0.0),
        (s_rut(s, "dots", level="stress"), 8.5, "sans_bold", 1.2),
    ]
    inner_radius = c - h.px(3.5)
    for seg, pt, font, k in lines:
        off = h.px(6.2) * k
        size = h.size(pt)

        def chord(size_px: int, off: float = off) -> float:
            # width available inside the inner ring at the distance of the line edge farthest from the center
            d = abs(off) + size_px * 0.55
            return 2 * math.sqrt(max(0.0, inner_radius**2 - d**2)) * 0.94

        # every line must fit inside the inner ring of the stamp
        while t.text_width(seg.text, font, size) > chord(size) and size > 8:
            size -= 1
        # translucent blue ink rotated 20°: the stamp data end up at stress level
        level = "stress" if seg.level == "base" and seg.type != "text" else seg.level
        t.write_rotated(
            c + perp[0] * off,
            c + perp[1] * off,
            seg.text,
            degrees,
            type=seg.type,
            font_name=font,
            size=size,
            color=STAMP_BLUE,
            level=level,
            tags={**seg.tags, "dpi": h.dpi},
        )
    return t


def _rectangular_stamp(h: Sheet, f: FakeData, degrees: float) -> Canvas:
    w, height = int(round(h.px(58))), int(round(h.px(30)))
    t = Canvas.new(w, height)
    d = ImageDraw.Draw(t.img)
    c = (w / 2, height / 2)
    rad = math.radians(degrees)
    bw, bh = h.px(50) / 2, h.px(19) / 2

    def rot(x: float, y: float) -> tuple[float, float]:
        return (c[0] + x * math.cos(rad) + y * math.sin(rad), c[1] - x * math.sin(rad) + y * math.cos(rad))

    corners = [rot(-bw, -bh), rot(bw, -bh), rot(bw, bh), rot(-bw, bh)]
    d.polygon(corners, outline=STAMP_BLUE, width=max(2, int(h.px(0.6))))
    perp = (math.sin(rad), math.cos(rad))
    for seg, pt, k in ((Seg("OFICINA DE PARTES"), 8, -1.1), (Seg("RECIBIDO"), 12, 0.15), (s_date(f.date()), 8, 1.25)):
        off = h.px(5.5) * k
        t.write_rotated(
            c[0] + perp[0] * off,
            c[1] + perp[1] * off,
            seg.text,
            degrees,
            type="text",
            font_name="sans_bold",
            size=h.size(pt),
            color=STAMP_BLUE,
            tags={**seg.tags, "dpi": h.dpi},
        )
    return t


def _gen_stamp_signature(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    path = "pdf_scanned/timbre_y_firma.pdf"
    p = f.person()
    manager = f.person()
    officer = f.person()
    note = f.person(in_list=False)
    h = Sheet(300)
    _header(h, f, "ACTA DE RECEPCIÓN CONFORME", "Servicios profesionales a honorarios")
    y = 62.0
    h.text(20, y, "En virtud del contrato vigente, se deja constancia de la recepción conforme de los", pt=11)
    y += 6.5
    h.text(20, y, "productos entregados por el prestador que se individualiza:", pt=11)
    y += 11
    h.line(24, y, [Seg("Prestador: "), s_name(p)], pt=11)
    y += 6.5
    h.line(24, y, [Seg("RUT: "), s_rut(p, "dots")], pt=11)
    y += 6.5
    h.line(
        24,
        y,
        [Seg("Correo: "), s_email(p, "dot"), Seg("    Teléfono: "), s_mobile(p, "mobile_international")],
        pt=11,
    )
    y += 6.5
    h.line(24, y, [Seg("Monto a pagar: "), s_amount(f), Seg("    Período: "), s_date(f.date())], pt=11)
    y += 11
    for phrase in (ADMINISTRATIVE_PHRASES[3], ADMINISTRATIVE_PHRASES[4], ADMINISTRATIVE_PHRASES[10]):
        h.text(20, y, phrase, pt=10.5)
        y += 6.3
    # signature block
    y_line = 200.0
    signature = h.canvas.write_irregular(
        h.px(32),
        h.px(y_line - 3),
        f"{manager.given_names.split()[0]} {manager.paternal_surname}",
        rng,
        type="signature",
        font_name="stix_italic",
        size=h.size(24),
        color=PEN_BLUE,
        level="out_of_scope",
        tags={"handwritten": True, "dpi": h.dpi},
    )
    assert signature is not None
    # final stroke of the signature: a curve that crosses the name
    x0f, y0f, x1f, y1f = bounding_box(signature.polygon)
    ts = np.linspace(0, 1, 40)
    curve = [
        (x0f - h.px(2) + (x1f - x0f + h.px(8)) * t, y1f - (y1f - y0f) * 0.35 + math.sin(t * math.pi * 2.3) * h.px(2.2))
        for t in ts
    ]
    ImageDraw.Draw(h.canvas.img).line(curve, fill=PEN_BLUE, width=max(2, int(h.px(0.45))), joint="curve")
    cx = [c[0] for c in curve]
    cy = [c[1] for c in curve]
    g = h.px(0.3)
    signature.polygon = rect(min(x0f, min(cx) - g), min(y0f, min(cy) - g), max(x1f, max(cx) + g), max(y1f, max(cy) + g))
    signature.tags["stroke"] = True
    h.rule(25, y_line, 95, y_line)
    h.line(25, y_line + 5, [s_name(manager)], pt=10)
    h.line(25, y_line + 10, [Seg("RUT "), s_rut(manager, "no_dots")], pt=10)
    h.text(25, y_line + 15, "Encargado(a) Unidad de Control de Gestión", pt=9)
    # translucent blue stamps (over the signature line and at the top right)
    _blend_stamp(h, _round_stamp(h, officer, 20.0), 80, y_line - 30)
    _blend_stamp(h, _rectangular_stamp(h, f, -7.0), 147, 30, opacity=0.7)
    # handwritten note in the bottom margin with a phone number
    yn = 250.0
    e_note = h.canvas.write_irregular(
        h.px(118), h.px(yn), "Llamar a", rng, font_name="stix_italic", size=h.size(15), color=PEN_BLUE
    )
    assert e_note is not None
    e_note.tags["handwritten"] = True
    x_end = bounding_box(e_note.polygon)[2] + h.px(2.5)
    fmt = "mobile_national"
    tel = h.canvas.write_irregular(
        x_end,
        h.px(yn),
        note.phone.format(fmt),
        rng,
        type="phone",
        font_name="stix_italic",
        size=h.size(15),
        color=PEN_BLUE,
        level="stress",
        tags={"format": fmt, "kind": "mobile", "handwritten": True, "dpi": h.dpi},
    )
    assert tel is not None
    e_remark = h.canvas.write_irregular(
        h.px(118),
        h.px(yn + 8),
        "por boleta pendiente",
        rng,
        font_name="stix_italic",
        size=h.size(13),
        color=PEN_BLUE,
    )
    assert e_remark is not None
    e_remark.tags["handwritten"] = True
    d = Degradation(skew=0.5, noise=4, blur=0.5, quality=80)
    doc = pymupdf.open()
    _, elements = scanned_page(doc, scan(h, rng, d))
    a = _file_entry(
        "timbre_y_firma",
        path,
        "Acta escaneada en color con timbre azul semitransparente girado (nombre y RUT), firma manuscrita "
        "y nota manuscrita con un teléfono.",
        doc,
        elements,
        dpi=300,
        degradation=d.__dict__,
    )
    _save(doc, ctx, path)
    return a


def generate(ctx: Context) -> list[FileEntry]:
    f = ctx.fake_data(SEED_NAME)
    rng = ctx.rng(SEED_NAME)
    return [
        _gen_report_300(ctx, f, rng),
        _gen_report_200_skewed(ctx, f, rng),
        _gen_record_card(ctx, f, rng),
        _gen_inverted(ctx, f, rng),
        _gen_mixed(ctx, f, rng),
        _gen_sandwich(ctx, f, rng),
        _gen_fax(ctx, f, rng),
        _gen_stamp_signature(ctx, f, rng),
    ]
