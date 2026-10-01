"""PDFs with a real text layer, built with PyMuPDF.

Each datum is written with ``page.insert_text`` and its polygon is computed in two ways: from
the insertion geometry (glyph advances according to the font, ascender and descender) and
with ``page.search_for(..., quads=True)`` on the finished page. If both agree (±2 pt at each
corner) the ``search_for`` quadrilateral is stored, which is the manifest convention;
otherwise the computed one is stored and it is noted in ``tags["gt"]``.

Coordinates: PyMuPDF's *unrotated* page space (origin at the top-left corner of the
CropBox). On pages with ``/Rotate`` the text is written in visible coordinates and converted
with ``page.derotation_matrix``.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from test_bench.canvas import Canvas, font_path, map_to_rect, rect
from test_bench.context import Context
from test_bench.faces import FaceProvider
from test_bench.fake_data import (
    ADMINISTRATIVE_PHRASES,
    COMMUNES,
    EMAIL_FORMATS,
    PHONE_FORMATS,
    PHONE_FORMATS_BY_KIND,
    RUT_FORMATS,
    FakeData,
    Person,
    Phone,
    strip_accents,
)
from test_bench.schema import Element, FileEntry, Page, Polygon

SEED_NAME = "pdf_texto"  # seed string: kept in Spanish so the generated files do not change
CATEGORY = "pdf_text"
A4 = (595.0, 842.0)

# Embedded TrueType fonts (DejaVu: accents, ñ and en dash).
_TTF = {
    "dejavu": "sans",
    "dejavu_n": "sans_bold",
    "dejavu_serif": "serif",
    "dejavu_serif_n": "serif_bold",
    "dejavu_mono": "mono",
    "dejavu_mono_n": "mono_bold",
}
# PyMuPDF's Base 14 fonts only encode Latin-1: if the text has another character (e.g. "–"),
# the line is written with the equivalent TrueType font.
_FALLBACK_TTF = {
    "helv": "dejavu",
    "hebo": "dejavu_n",
    "tiro": "dejavu_serif",
    "tibo": "dejavu_serif_n",
    "cour": "dejavu_mono",
    "cobo": "dejavu_mono_n",
}
_BOLD = {
    "helv": "hebo",
    "tiro": "tibo",
    "cour": "cobo",
    "dejavu": "dejavu_n",
    "dejavu_serif": "dejavu_serif_n",
    "dejavu_mono": "dejavu_mono_n",
}
# Extraction flags used to compute the ground truth: no clipping to the page.
_FLAGS = pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_PRESERVE_LIGATURES
_GT_TOLERANCE = 2.0  # pt: maximum difference per corner between geometry and search_for

MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "octubre", "noviembre", "diciembre"]  # fmt: skip


@lru_cache(maxsize=32)
def _font(name: str) -> pymupdf.Font:
    if name in _TTF:
        return pymupdf.Font(fontfile=str(font_path(_TTF[name])))
    return pymupdf.Font(name)


@lru_cache(maxsize=8192)
def _advance(name: str, ch: str) -> float:
    return _font(name).text_length(ch, 1.0)


def text_length(name: str, text: str, size: float) -> float:
    """Width of the text in pt (PyMuPDF applies no kerning: it is the sum of the advances)."""
    return size * sum(_advance(name, ch) for ch in text)


def _is_latin1(text: str) -> bool:
    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return True


def _font_for(name: str, text: str) -> str:
    if name not in _TTF and not _is_latin1(text):
        return _FALLBACK_TTF[name]
    return name


def full_textpage(page: pymupdf.Page) -> pymupdf.TextPage:
    """TextPage without clipping: includes the text outside the CropBox (coordinates relative to the CropBox)."""
    return page.get_textpage(clip=pymupdf.INFINITE_RECT(), flags=_FLAGS)


def _quad_to_polygon(q: pymupdf.Quad) -> Polygon:
    return [[round(p.x, 3), round(p.y, 3)] for p in (q.ul, q.ur, q.lr, q.ll)]


def _distance(a: Polygon, b: Polygon) -> float:
    return max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Parts of a line
# ---------------------------------------------------------------------------


@dataclass
class Part:
    """Segment of a line: neutral text (``type="text"``) or a datum to record."""

    text: str
    type: str = "text"
    level: str = "base"
    tags: dict[str, Any] = field(default_factory=dict)
    value: str | None = None
    joined: bool = False  # neutral text that allows no line break (e.g. the " <" of "Name <email>")

    @property
    def atomic(self) -> bool:
        """Data, decoys and joined parts are never split when wrapping a paragraph."""
        return self.type != "text" or bool(self.tags) or self.joined


def T(text: str) -> Part:  # noqa: N802 - frequently used shortcut
    return Part(text)


def _parts(parts: str | Part | Sequence[Part]) -> list[Part]:
    if isinstance(parts, str):
        return [Part(parts)]
    if isinstance(parts, Part):
        return [parts]
    return list(parts)


def rut_p(p: Person, format: str, level: str | None = None, **tags: Any) -> Part:
    if format == "lowercase_k":
        text, format_level = p.rut("dots").replace("K", "k"), "base"
    else:
        text, format_level = p.rut(format), RUT_FORMATS[format]
    part_tags = {"format": format, "dv_valid": p.rut_dv_valid, **tags}
    if p.rut_body < 10_000_000:
        part_tags["seven_digits"] = True
    return Part(text, "rut", level or format_level, part_tags)


def email_p(p: Person, format: str, level: str | None = None) -> Part:
    return Part(p.email(format), "email", level or EMAIL_FORMATS[format], {"format": format})


def phone_p(t: Phone, format: str, level: str | None = None) -> Part:
    return Part(t.format(format), "phone", level or PHONE_FORMATS[format], {"format": format, "kind": t.kind})


def name_text(p: Person, variant: str) -> str:
    return {
        "exact": p.full_name,
        "no_accents": strip_accents(p.full_name),
        "uppercase": p.full_name.upper(),
        "surnames_names": f"{p.paternal_surname} {p.maternal_surname}, {p.given_names}",
        "partial": f"{p.given_names.split()[0]} {p.paternal_surname}",
        "first_name_only": p.given_names.split()[0],  # greeting of an email: "Estimada Carolina:"
    }[variant]


def name_p(p: Person, variant: str = "exact") -> Part:
    text = name_text(p, variant)
    if variant == "no_accents" and text == p.full_name:
        variant = "exact"
    if not p.in_list:
        level = "out_of_scope"
    else:
        level = "stress" if variant in ("partial", "first_name_only") else "base"
    return Part(text, "name", level, {"in_list": p.in_list, "variant": variant})


def address_p(p: Person) -> Part:
    return Part(p.address, "address", "base" if p.in_list else "out_of_scope", {"in_list": p.in_list})


def url_p(text: str, level: str = "base") -> Part:
    return Part(text, "url", level)


def amount_p(text: str) -> Part:
    return Part(text, "text", tags={"decoy": "amount"})


def date_p(text: str) -> Part:
    return Part(text, "text", tags={"decoy": "date"})


def kind_of(format: str) -> str:
    return next(c for c, fs in PHONE_FORMATS_BY_KIND.items() if format in fs)


def phone_for(f: FakeData, p: Person, format: str) -> Phone:
    """The person's phone that matches the kind of the format (mobile, Santiago or regional)."""
    kind = kind_of(format)
    if kind == "mobile":
        return p.phone
    if p.landline.kind != kind:
        p.landline = f.phone(kind)
    return p.landline


# ---------------------------------------------------------------------------
# Sheet: a page that records its elements
# ---------------------------------------------------------------------------


class Sheet:
    """PDF page that records the geometry of everything written on it.

    With ``visible=True`` the coordinates it receives are those of the page as it is seen
    (after ``/Rotate``); they are converted to the unrotated space before writing.
    """

    def __init__(self, page: pymupdf.Page, index: int, visible: bool = False) -> None:
        self._doc = page.parent
        self._page = page
        self.index = index
        self.visible = visible
        self.elements: list[Element] = []
        self._to_refine: list[tuple[Element, str]] = []
        self._fonts: set[str] = set()

    @property
    def page(self) -> pymupdf.Page:
        # PyMuPDF invalidates Page objects when pages are added to the document: they are reloaded.
        if self._page.parent is None:
            self._page = self._doc[self.index]
        return self._page

    # -- utilities -----------------------------------------------------------

    def _prepare_font(self, name: str, text: str) -> str:
        name = _font_for(name, text)
        if name in _TTF and name not in self._fonts:
            self.page.insert_font(fontname=name, fontfile=str(font_path(_TTF[name])))
            self._fonts.add(name)
        return name

    def to_page(self, x: float, y: float) -> pymupdf.Point:
        p = pymupdf.Point(x, y)
        return p * self.page.derotation_matrix if self.visible else p

    @staticmethod
    def width(text: str, font: str = "helv", size: float = 10) -> float:
        return text_length(_font_for(font, text), text, size)

    def straight_line(
        self, x0: float, y0: float, x1: float, y1: float, thickness: float = 0.5, color=(0, 0, 0)
    ) -> None:
        self.page.draw_line(self.to_page(x0, y0), self.to_page(x1, y1), color=color, width=thickness)

    def rectangle(self, x0: float, y0: float, x1: float, y1: float, fill=None, border=None, thickness=0.5) -> None:
        a, b = self.to_page(x0, y0), self.to_page(x1, y1)
        self.page.draw_rect(pymupdf.Rect(a, b).normalize(), color=border, fill=fill, width=thickness)

    def add(self, elements: Sequence[Element]) -> None:
        for e in elements:
            e.page = self.index
            self.elements.append(e)

    # -- text ----------------------------------------------------------------

    def write(
        self,
        x: float,
        y: float,
        parts: str | Part | Sequence[Part],
        *,
        font: str = "helv",
        size: float = 10,
        color: tuple[float, float, float] = (0, 0, 0),
        angle: float = 0,
        layer: str = "text",
        render_mode: int = 0,
        register: bool = True,
        draw: bool = True,
        tags: dict[str, Any] | None = None,
    ) -> float:
        """Writes a line with its baseline at (x, y) and records each part. Returns the final x.

        ``angle`` is the visible rotation (counterclockwise, in degrees). Multiples of 90 use
        ``rotate``; the rest, ``morph`` pivoting on the insertion point. With ``draw=False``
        it only records (for a datum already written inside another one, like a RUT in a URL).
        """
        items = _parts(parts)
        text = "".join(p.text for p in items)
        name = self._prepare_font(font, text)
        f = _font(name)
        point = self.to_page(x, y)
        rotation = (angle + (self.page.rotation if self.visible else 0)) % 360
        options: dict[str, Any] = {"fontsize": size, "fontname": name, "color": color, "render_mode": render_mode}
        if not draw:
            pass
        elif rotation % 90 == 0:
            self.page.insert_text(point, text, rotate=int(rotation), **options)
        else:
            self.page.insert_text(point, text, morph=(point, pymupdf.Matrix(rotation)), **options)
        length = text_length(name, text, size)
        if register:
            v0, v1 = -f.ascender * size, -f.descender * size
            start = 0
            for p in items:
                clean = p.text.strip()
                before = text[:start] + p.text[: len(p.text) - len(p.text.lstrip())]
                start += len(p.text)
                if not clean or (p.type == "text" and not any(c.isalnum() for c in clean)):
                    continue
                u0 = text_length(name, before, size)
                u1 = u0 + text_length(name, clean, size)
                e = Element(
                    type=p.type,
                    page=self.index,
                    polygon=_geometric_quad(point, rotation, u0, u1, v0, v1),
                    value=p.value if p.value is not None else clean,
                    level=p.level,
                    layer=layer,
                    tags={"font": name, "size_pt": size, "angle": angle, **(tags or {}), **p.tags},
                )
                self.elements.append(e)
                if layer in ("text", "hidden", "vector"):
                    self._to_refine.append((e, clean))
        return x + length

    def write_spaced(
        self,
        x: float,
        y: float,
        part: Part,
        spacing: float,
        *,
        font: str = "helv",
        size: float = 10,
    ) -> float:
        """Writes each character separately, with ``spacing`` extra pt between them (no rotation only)."""
        name = self._prepare_font(font, part.text)
        f = _font(name)
        cx = x
        for ch in part.text:
            self.page.insert_text(self.to_page(cx, y), ch, fontsize=size, fontname=name)
            cx += text_length(name, ch, size) + spacing
        end = cx - spacing
        e = Element(
            type=part.type,
            page=self.index,
            polygon=rect(round(x, 3), round(y - f.ascender * size, 3), round(end, 3), round(y - f.descender * size, 3)),
            value=part.value or part.text,
            level=part.level,
            layer="text",
            tags={
                "font": name,
                "size_pt": size,
                "angle": 0,
                "spacing_pt": spacing,
                "gt": "geometry",
                **part.tags,
            },  # fmt: skip
        )
        self.elements.append(e)
        return end

    def refine(self) -> None:
        """Replaces the computed quadrilateral with the ``search_for`` one when they agree."""
        if not self._to_refine:
            return
        tp = full_textpage(self.page)
        cache: dict[str, list[Polygon]] = {}
        for e, searched in self._to_refine:
            if searched not in cache:
                cache[searched] = [_quad_to_polygon(q) for q in self.page.search_for(searched, quads=True, textpage=tp)]
            candidates = cache[searched]
            best = min(candidates, key=lambda q: _distance(q, e.polygon), default=None)
            if best is not None and _distance(best, e.polygon) <= _GT_TOLERANCE:
                e.polygon = best
                e.tags["gt"] = "search_for"
            else:
                e.polygon = [[round(px, 3), round(py, 3)] for px, py in e.polygon]
                e.tags["gt"] = "geometry"
        self._to_refine.clear()


def _geometric_quad(p: pymupdf.Point, rotation: float, u0: float, u1: float, v0: float, v1: float) -> Polygon:
    """Quadrilateral of a run of text: ``u`` along the line, ``v`` towards the bottom of the glyph."""
    t = math.radians(rotation)
    d = (math.cos(t), -math.sin(t))  # reading direction (y down)
    n = (math.sin(t), math.cos(t))  # "down" of the glyph
    return [[p.x + u * d[0] + v * n[0], p.y + u * d[1] + v * n[1]] for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]


def wrap(parts: Sequence[Part], max_width: float, font: str, size: float) -> list[list[Part]]:
    """Splits the parts into lines of ``max_width`` without ever cutting a datum or a decoy."""
    words: list[list[tuple[str, int]]] = [[]]
    for i, p in enumerate(parts):
        if p.atomic:
            words[-1].append((p.text, i))
            continue
        for chunk in re.findall(r"\s+|\S+", p.text):
            words[-1].append((chunk, i))
            if chunk.isspace():
                words.append([])
    lines: list[list[tuple[str, int]]] = [[]]
    for word in words:
        if not word:
            continue
        candidate = lines[-1] + word
        text = "".join(t for t, _ in candidate).rstrip()
        if lines[-1] and Sheet.width(text, font, size) > max_width:
            if word[0][0].isspace():
                word = word[1:]
            lines.append(list(word))
        else:
            lines[-1] = candidate
    output = []
    for line in lines:
        while line and line[0][0].isspace():
            line = line[1:]
        groups: list[Part] = []
        last = -1
        for chunk, i in line:
            if i == last and not parts[i].atomic:
                groups[-1].text += chunk
            elif parts[i].atomic:
                groups.append(parts[i])
            else:
                groups.append(Part(chunk))
            last = i
        if groups:
            output.append(groups)
    return output


# ---------------------------------------------------------------------------
# Document and flow cursor
# ---------------------------------------------------------------------------


class Document:
    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.sheets: list[Sheet] = []

    def new_sheet(self, width: float = A4[0], height: float = A4[1], rotation: int = 0, visible: bool = False) -> Sheet:
        page = self.doc.new_page(width=width, height=height)
        if rotation:
            page.set_rotation(rotation)
        h = Sheet(page, len(self.sheets), visible)
        self.sheets.append(h)
        return h

    def elements(self) -> list[Element]:
        return [e for h in self.sheets for e in h.elements]

    def refine(self) -> None:
        for h in self.sheets:
            h.refine()

    def save(self, ctx: Context, path: str, subset: bool = True) -> None:
        self.refine()
        if subset:
            self.doc.subset_fonts()
        self.doc.save(ctx.path(path), garbage=3, deflate=True, no_new_id=True)

    def file_entry(self, id: str, path: str, description: str, tags: dict[str, Any] | None = None) -> FileEntry:
        return FileEntry(
            id=id,
            path=path,
            format="pdf",
            category=CATEGORY,
            description=description,
            pages=pages_of(self.doc),
            elements=self.elements(),
            tags=dict(tags or {}),
        )


def pages_of(doc: pymupdf.Document) -> list[Page]:
    output = []
    for i, page in enumerate(doc):
        box = page.cropbox
        output.append(Page(index=i, width=round(box.width, 3), height=round(box.height, 3), unit="pt",
                           rotation=page.rotation))  # fmt: skip
    return output


class Cursor:
    """Flow writing, from top to bottom, with automatic page breaks."""

    def __init__(
        self,
        document: Document,
        *,
        font: str = "helv",
        size: float = 10,
        x: float = 60,
        width: float = 475,
        top: float = 80,
        bottom: float = 780,
        on_create: Callable[[Sheet], None] | None = None,
    ) -> None:
        self.document = document
        self.font, self.size = font, size
        self.x, self.width = x, width
        self.top, self.bottom = top, bottom
        self.on_create = on_create
        self.sheet = self._new()
        self.y = top

    def _new(self) -> Sheet:
        h = self.document.new_sheet()
        if self.on_create:
            self.on_create(h)
        return h

    def page_break(self) -> None:
        self.sheet = self._new()
        self.y = self.top

    def ensure(self, height: float) -> None:
        if self.y + height > self.bottom:
            self.page_break()

    def space(self, height: float) -> None:
        self.y += height

    def line(
        self,
        parts: str | Part | Sequence[Part],
        *,
        font: str | None = None,
        size: float | None = None,
        indent: float = 0,
        line_spacing: float = 1.45,
        **options: Any,
    ) -> None:
        size = size or self.size
        self.ensure(size * line_spacing)
        font = font or self.font
        text = "".join(p.text for p in _parts(parts)).rstrip()
        if indent + Sheet.width(text, font, size) > self.width + 1:
            raise ValueError(f"the line does not fit in the cursor width: {text!r}")
        self.sheet.write(self.x + indent, self.y + size, parts, font=font, size=size, **options)
        self.y += size * line_spacing

    def paragraph(
        self,
        parts: str | Part | Sequence[Part],
        *,
        font: str | None = None,
        size: float | None = None,
        indent: float = 0,
        line_spacing: float = 1.45,
        after: float = 4,
    ) -> None:
        font, size = font or self.font, size or self.size
        for group in wrap(_parts(parts), self.width - indent, font, size):
            self.line(group, font=font, size=size, indent=indent, line_spacing=line_spacing)
        self.y += after

    def field(self, label: str, parts: str | Part | Sequence[Part], tab: float = 112) -> None:
        size = self.size
        self.ensure(size * 1.5)
        base = self.y + size
        self.sheet.write(self.x, base, label, font=_BOLD[self.font], size=size)
        self.sheet.write(self.x + tab, base, parts, font=self.font, size=size)
        self.y += size * 1.5

    def section(self, title: str) -> None:
        self.ensure(self.size * 4)
        self.y += self.size * 0.6
        self.line(title, font=_BOLD[self.font], size=self.size + 1, line_spacing=1.6)

    def table(self, columns: list[str], rows: list[list[Part]], *, font: str, size: float = 8) -> None:
        row_height = size * 2.0
        self.ensure(row_height * (len(rows) + 1) + 4)
        self.y = draw_table(self.sheet, self.x, self.y, self.width, columns, rows, font=font, size=size) + 6


def draw_table(
    sheet: Sheet,
    x: float,
    y: float,
    width: float,
    columns: list[str],
    rows: list[list[Part]],
    *,
    font: str,
    size: float = 8,
    min_size: float = 6.5,
) -> float:
    """Table with drawn lines and text cells. Shrinks the letters if it does not fit. Returns the final y."""
    bold = _BOLD[font]
    padding = 4.0
    while True:
        widths = []
        for j, title in enumerate(columns):
            m = Sheet.width(title, bold, size)
            for row in rows:
                m = max(m, Sheet.width(row[j].text, font, size))
            widths.append(m + 2 * padding)
        if sum(widths) <= width or size <= min_size:
            break
        size -= 0.25
    if sum(widths) > width:
        raise ValueError("the table does not fit in the available width")
    extra = (width - sum(widths)) / len(widths)
    widths = [a + extra for a in widths]
    row_height = size * 2.0
    xs = [x]
    for a in widths:
        xs.append(xs[-1] + a)
    n = len(rows) + 1
    sheet.rectangle(x, y, x + width, y + row_height, fill=(0.88, 0.9, 0.93))
    for i in range(n + 1):
        sheet.straight_line(x, y + i * row_height, x + width, y + i * row_height)
    for cx in xs:
        sheet.straight_line(cx, y, cx, y + n * row_height)
    base = y + row_height / 2 + size * 0.35
    for j, title in enumerate(columns):
        sheet.write(xs[j] + padding, base, title, font=bold, size=size)
    for i, row in enumerate(rows, start=1):
        for j, cell in enumerate(row):
            sheet.write(xs[j] + padding, base + i * row_height, cell, font=font, size=size)
    return y + n * row_height


def footer(document: Document, left: str, font: str = "helv", y: float = 812) -> None:
    n = len(document.sheets)
    for h in document.sheets:
        h.straight_line(60, y - 12, 535, y - 12, thickness=0.4, color=(0.5, 0.5, 0.5))
        h.write(60, y, left, font=font, size=7.5, color=(0.3, 0.3, 0.3))
        right = f"Página {h.index + 1} de {n}"
        h.write(535 - Sheet.width(right, font, 7.5), y, right, font=font, size=7.5, color=(0.3, 0.3, 0.3))


def gore_header(font: str, unit: str, folio: str) -> Callable[[Sheet], None]:
    def draw(h: Sheet) -> None:
        h.write(60, 44, "GOBIERNO REGIONAL FICTICIO", font=_BOLD[font], size=11, color=(0.1, 0.2, 0.45))
        h.write(60, 56, unit, font=font, size=8, color=(0.3, 0.3, 0.3))
        right = f"Folio N° {folio}"
        h.write(535 - Sheet.width(right, font, 8), 44, right, font=font, size=8)
        h.straight_line(60, 63, 535, 63, thickness=0.8, color=(0.1, 0.2, 0.45))

    return draw


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _pesos(text: str) -> int:
    return int(re.sub(r"\D", "", text))


def _amount_format(value: int) -> str:
    return f"$ {value:,}".replace(",", ".")


def _date_key(date: str) -> tuple[int, int, int]:
    """Chronological order of a date like "16 de octubre de 2026"."""
    day, _, month, _, year = date.split()
    return int(year), MONTHS.index(month.lower()), int(day)


# ---------------------------------------------------------------------------
# 1. Replica of the notebook's test document
# ---------------------------------------------------------------------------

NOTEBOOK_TEXT = """INFORME DE HONORARIOS - AGOSTO 2026

Nombre: Ana Maria Rojas Pena
RUT: 15.782.334-9
Correo: ana.rojas@ejemplo.cl
Telefono: +56 9 8123 4567
Direccion: Pasaje Los Alerces 442, depto 31

Producto 1: Informe de avance del programa
Monto bruto: $ 1.450.000

Contraparte: Jefatura de la unidad
RUT contraparte: 9.876.543-3
Sitio: https://www.ejemplo.cl/rendiciones
"""


def _notebook(ctx: Context, f: FakeData) -> FileEntry:
    f.reserve(
        ruts=(15782334, 9876543),
        phones=("981234567",),
        emails=(("ana", "rojas"),),
        names=("Ana Maria Rojas Pena",),
    )
    ana = Person(
        given_names="Ana Maria",
        paternal_surname="Rojas",
        maternal_surname="Pena",
        sex="F",
        rut_body=15782334,
        rut_dv="9",
        rut_dv_valid=False,
        phone=Phone("mobile", "981234567", "9"),
        landline=f.phone("regional"),
        domain="ejemplo.cl",
        address="Pasaje Los Alerces 442, depto 31",
        commune="Concepción",
        birth_date="14-03-1987",
        in_list=True,
        extra={"origin": "cuaderno CoP33, celda 6"},
    )
    f.people.append(ana)
    counterpart_rut = Part("9.876.543-3", "rut", "base", {"format": "dots", "dv_valid": True, "seven_digits": True})
    lines: list[list[Part]] = [
        [T("INFORME DE HONORARIOS - AGOSTO 2026")],
        [T("Nombre:"), name_p(ana, "exact")],
        [T("RUT:"), rut_p(ana, "dots")],
        [T("Correo:"), email_p(ana, "dot")],
        [T("Telefono:"), phone_p(ana.phone, "mobile_international")],
        [T("Direccion:"), address_p(ana)],
        [T("Producto 1: Informe de avance del programa")],
        [T("Monto bruto:"), amount_p("$ 1.450.000")],
        [T("Contraparte: Jefatura de la unidad")],
        [T("RUT contraparte:"), counterpart_rut],
        [T("Sitio:"), url_p("https://www.ejemplo.cl/rendiciones")],
    ]
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((60, 70), NOTEBOOK_TEXT, fontsize=11, fontname="helv")
    tp = full_textpage(page)
    elements: list[Element] = []
    for parts in lines:
        full = " ".join(p.text for p in parts)
        (q_line,) = page.search_for(full, quads=True, textpage=tp)
        box = q_line.rect + (-1, -1, 1, 1)
        for p in parts:
            hits = [q for q in page.search_for(p.text, quads=True, textpage=tp) if box.contains(q.rect)]
            assert len(hits) == 1, (p.text, hits)
            elements.append(
                Element(
                    type=p.type,
                    page=0,
                    polygon=_quad_to_polygon(hits[0]),
                    value=p.text,
                    level=p.level,
                    layer="text",
                    tags={"font": "helv", "size_pt": 11, "angle": 0, "gt": "search_for", **p.tags},
                )
            )
    path = "pdf_text/cuaderno_informe_de_prueba.pdf"
    doc.save(ctx.path(path), no_new_id=True)
    file_entry = FileEntry(
        id="pdft_cuaderno",
        path=path,
        format="pdf",
        category=CATEGORY,
        description="Réplica exacta del documento de prueba del cuaderno CoP 33 (celda 6): insert_text helv 11 pt.",
        pages=pages_of(doc),
        elements=elements,
        tags={"notebook_replica": True},
    )
    doc.close()
    return file_entry


# ---------------------------------------------------------------------------
# 2. Fee reports
# ---------------------------------------------------------------------------

# Slots per report: 0 = service provider, 1 = counterpart, 2..6 = table rows.
# (RUT format, email format, phone format, options)
#   options: bad_dv, k (DV K), seven (7-digit body), out (not in the name list)
_MAILBOXES: list[list[tuple[str, str, str, set[str]]]] = [
    [
        ("dots", "dot", "mobile_international", set()),
        ("no_dots", "initial", "regional_parentheses", set()),
        ("spaced_hyphen", "with_year", "mobile_compact", {"bad_dv"}),
        ("en_dash", "underscore", "santiago_national", set()),
        ("dots", "uppercase", "regional_compact", set()),
        ("lowercase_k", "plus", "mobile_national", {"k"}),
        ("no_dots", "subdomain", "santiago_international", set()),
    ],
    [
        ("spaced_hyphen", "dot", "mobile_block", set()),
        ("en_dash", "initial", "santiago_parentheses", set()),
        ("dots", "with_year", "mobile_parentheses", set()),
        ("no_hyphen", "underscore", "regional_international", {"out"}),
        ("no_dots", "uppercase", "mobile_hyphens", {"seven"}),
        ("dots", "plus", "regional_hyphen", {"bad_dv"}),
        ("en_dash", "subdomain", "old_mobile_09", set()),
    ],
    [
        ("dots", "subdomain", "mobile_international", {"k"}),
        ("en_dash", "dot", "regional_hyphen", set()),
        ("spaced_hyphen", "initial", "santiago_national", set()),
        ("commas", "with_year", "mobile_compact", {"bad_dv"}),
        ("no_dots", "underscore", "regional_parentheses", set()),
        ("dots", "uppercase", "mobile_national", set()),
        ("inner_spaces", "spelled_arroba", "santiago_parentheses", set()),
    ],
    [
        ("no_dots", "plus", "mobile_block", {"out"}),
        ("dots", "dot", "santiago_international", {"seven"}),
        ("en_dash", "initial", "mobile_parentheses", set()),
        ("spaced_hyphen", "with_year", "regional_international", {"bad_dv"}),
        ("dots", "underscore", "mobile_hyphens", set()),
        ("no_dots", "subdomain", "regional_compact", set()),
        ("dots", "spaces", "old_regional_0", set()),
    ],
]
# (body font, table font, activity log lines, month)
_REPORTS = [
    ("helv", "helv", 0, 7),
    ("tiro", "cour", 22, 8),
    ("dejavu", "dejavu", 62, 9),
    ("helv", "tiro", 18, 10),
]
_TABLE_VARIANTS = ["exact", "no_accents", "uppercase", "surnames_names", "partial"]
_COUNTERPART_VARIANTS = ["exact", "no_accents", "exact", "no_accents"]
_ACTIVITIES = [
    "Taller de formulación de proyectos con organizaciones comunitarias",
    "Reunión de coordinación con el equipo municipal de fomento productivo",
    "Visita a terreno para levantamiento de información de emprendimientos",
    "Mesa técnica con servicios públicos regionales",
    "Revisión de rendiciones de cuentas de proyectos adjudicados",
    "Capacitación en uso de la plataforma de postulación",
    "Entrevistas a beneficiarios del programa",
]


def _mailbox_person(f: FakeData, options: set[str]) -> Person:
    p = f.person(dv_valid="bad_dv" not in options, in_list="out" not in options, seven_digits="seven" in options)
    if "k" in options:
        p.rut_body, p.rut_dv = f.rut_with_k()
        p.rut_dv_valid = True
    return p


def _fees_report(ctx: Context, f: FakeData, rng: np.random.Generator, n: int) -> FileEntry:
    font, table_font, n_activities, month_i = _REPORTS[n - 1]
    month = MONTHS[month_i - 1]
    slots = _MAILBOXES[n - 1]
    people = [_mailbox_person(f, op) for *_, op in slots]
    folio = f.folio()
    doc = Document()
    cur = Cursor(doc, font=font, size=10, on_create=gore_header(font, "División de Fomento e Industria", folio))
    bold = _BOLD[font]

    def data(i: int) -> tuple[Part, Part, Part]:
        rut_f, email_f, phone_f, _ = slots[i]
        p = people[i]
        return rut_p(p, rut_f), email_p(p, email_f), phone_p(phone_for(f, p, phone_f), phone_f)

    cur.line(f"INFORME DE HONORARIOS - {month.upper()} 2026", font=bold, size=14, line_spacing=1.8)
    cur.line([T("Fecha de emisión: "), date_p(f.date())], size=9)
    cur.line([T("Período informado: "), date_p(f"1 al 30 de {month} de 2026")], size=9)

    p0 = people[0]
    rut0, email0, phone0 = data(0)
    cur.section("1. Antecedentes del prestador de servicios")
    cur.field("Nombre:", name_p(p0, "exact"))
    cur.field("RUT:", rut0)
    cur.field("Correo:", email0)
    cur.field("Teléfono:", phone0)
    cur.field("Dirección:", [address_p(p0), T(f", {p0.commune}")])
    cur.field("Calidad jurídica:", "Honorario a suma alzada")

    cur.section("2. Productos del período")
    k = 3 + n % 3
    for i in rng.choice(len(ADMINISTRATIVE_PHRASES), size=k, replace=False):
        cur.paragraph([T("- " + ADMINISTRATIVE_PHRASES[int(i)])], indent=8, after=1)
    url1 = f"https://transparencia.goreficticio.cl/honorarios/2026/{month}/{folio.replace('/', '-')}"
    cur.paragraph([T("Los medios de verificación están disponibles en "), url_p(url1), T(".")], indent=8)

    cur.section("3. Montos")
    gross = _pesos(f.amount())
    withholding = round(gross * 0.1525)
    cur.field("Monto bruto:", amount_p(_amount_format(gross)))
    cur.field("Retención 15,25 %:", amount_p(_amount_format(withholding)))
    cur.field("Monto líquido:", amount_p(_amount_format(gross - withholding)))
    cur.field("Fecha de pago:", date_p(f.date()))

    p1 = people[1]
    rut1, email1, phone1 = data(1)
    extension = str(int(rng.integers(2000, 7999)))
    cur.section("4. Contraparte técnica")
    cur.field("Nombre:", name_p(p1, _COUNTERPART_VARIANTS[n - 1]))
    cur.field("Cargo:", "Profesional de la División de Fomento e Industria")
    cur.field("RUT:", rut1)
    cur.field("Correo:", email1)
    cur.field("Teléfono:", [phone1, T(f", anexo {extension}")])

    section = 5
    if n_activities:
        cur.section(f"{section}. Registro de actividades")
        section += 1
        for _ in range(n_activities):
            day = int(rng.integers(1, 29))
            activity = _ACTIVITIES[int(rng.integers(len(_ACTIVITIES)))]
            commune = COMMUNES[int(rng.integers(len(COMMUNES)))]
            hours = int(rng.integers(2, 9))
            cur.paragraph(
                [date_p(f"{day:02d}-{month_i:02d}-2026"), T(f": {activity}, comuna de {commune} ({hours} horas).")],
                size=9,
                indent=8,
                line_spacing=1.4,
                after=0,
            )

    cur.section(f"{section}. Equipo de apoyo y participantes")
    rows = []
    for j, i in enumerate(range(2, 7)):
        rut, email, phone = data(i)
        variant = _TABLE_VARIANTS[(j + n - 1) % len(_TABLE_VARIANTS)]
        rows.append([name_p(people[i], variant), rut, email, phone])
    cur.table(["Nombre", "RUT", "Correo", "Teléfono"], rows, font=table_font, size=8)
    url2 = f"https://www.goreficticio.cl/fomento/equipos/{month}-2026?folio={folio.split('/')[0]}"
    cur.paragraph([T("Nómina vigente publicada en "), url_p(url2), T(".")], size=9)

    # Signatures
    cur.ensure(90)
    y = cur.y + 45
    h = cur.sheet
    for x0, person, variant, role in (
        (70, p0, "uppercase", "Prestador(a) de servicios"),
        (320, p1, "surnames_names", "V°B° Contraparte técnica"),
    ):
        h.straight_line(x0, y, x0 + 200, y, thickness=0.6)
        h.write(x0, y + 12, name_p(person, variant), font=font, size=9)
        h.write(x0, y + 24, role, font=font, size=8, color=(0.3, 0.3, 0.3))
    cur.y = y + 30

    footer(doc, f"Gobierno Regional Ficticio · Informe de honorarios {month} 2026", font=font)
    path = f"pdf_text/informe_honorarios_{n:02d}.pdf"
    doc.save(ctx, path)
    return doc.file_entry(
        f"pdft_honorarios_{n:02d}",
        path,
        f"Informe de honorarios de {len(doc.sheets)} página(s) con bloque de datos, contraparte, montos y tabla de "
        f"5 personas (fuente {font}, tabla {table_font}); variantes de formato de RUT, teléfono, correo y nombre.",
        {"font": font, "table_font": table_font},
    )


# ---------------------------------------------------------------------------
# 3. Exempt resolution
# ---------------------------------------------------------------------------


def _resolution(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    font = "tiro"
    folio = f.folio()
    doc = Document()
    cur = Cursor(
        doc,
        font=font,
        size=11,
        x=70,
        width=455,
        on_create=gore_header(font, "División de Administración y Finanzas", folio),
    )
    bold = _BOLD[font]
    provider = f.person()
    counterpart = f.person()
    wrong = f.person(dv_valid=False)
    head = f.person()
    distribution = [f.person() for _ in range(3)]
    gross = _amount_format(_pesos(f.amount()))
    title_word = "doña" if provider.sex == "F" else "don"
    domiciled = "domiciliada" if provider.sex == "F" else "domiciliado"

    cur.space(6)
    for title in (f"RESOLUCIÓN EXENTA N° {folio}", "APRUEBA CONTRATO DE PRESTACIÓN DE SERVICIOS A HONORARIOS"):
        cur.line(title, font=bold, size=11, indent=(cur.width - Sheet.width(title, bold, 11)) / 2)
    cur.line([T("Concepción, "), date_p(f.date())], indent=300)
    cur.space(8)

    cur.line("VISTOS:", font=bold)
    cur.paragraph(
        [
            T(
                "Lo dispuesto en la Ley N° 19.175, Orgánica Constitucional sobre Gobierno y Administración Regional; "
                "en la Ley N° 18.575, Orgánica Constitucional de Bases Generales de la Administración del Estado; "
                "en la Resolución N° 7 de 2019 de la Contraloría General de la República; y el certificado de "
                "disponibilidad presupuestaria N° 245 de fecha "
            ),
            date_p(f.date()),
            T("."),
        ]
    )
    cur.line("CONSIDERANDO:", font=bold)
    provider_rut = rut_p(provider, "dots")
    cur.paragraph(
        [
            T(f"1. Que {title_word} "),
            name_p(provider, "exact"),
            T(", cédula nacional de identidad N° "),
            provider_rut,
            T(f", {domiciled} en "),
            address_p(provider),
            T(
                f", comuna de {provider.commune}, presentó su oferta para prestar servicios de apoyo profesional "
                "a la División de Fomento e Industria."
            ),  # fmt: skip
        ]
    )
    provider_phone = phone_for(f, provider, "regional_international")
    cur.paragraph(
        [
            T(
                "2. Que la persona individualizada acredita el título profesional requerido e informó como medios "
                "de contacto el correo electrónico "
            ),  # fmt: skip
            email_p(provider, "dot"),
            T(", el teléfono móvil "),
            phone_p(provider.phone, "mobile_international"),
            T(" y el teléfono fijo "),
            phone_p(provider_phone, "regional_international"),
            T("."),
        ]
    )
    cur.paragraph(
        [
            T("3. Que la contraparte técnica, "),
            name_p(counterpart, "exact"),
            T(", RUT "),
            rut_p(counterpart, "no_dots"),
            T(", validó los antecedentes según consta en el memorándum N° 58 de "),
            date_p(f.date()),
            T(", enviado desde la casilla "),
            email_p(counterpart, "initial"),
            T("."),
        ]
    )
    cur.paragraph(
        [
            T("4. Que en el oficio anterior se consignó por error el RUT "),
            rut_p(wrong, "dots"),
            T(" a nombre de "),
            name_p(wrong, "no_accents"),
            T(", dato que se corrige mediante el presente acto administrativo."),
        ]
    )
    cur.paragraph(
        [
            T("5. Que existe disponibilidad presupuestaria para financiar un monto bruto mensual de "),
            amount_p(gross),
            T(", con cargo al subtítulo 21, ítem 03, del presupuesto vigente."),
        ]
    )
    cur.line("RESUELVO:", font=bold)
    start, end = sorted((f.date(), f.date()), key=_date_key)  # the contract does not end before it starts
    cur.paragraph(
        [
            T("1° APRUÉBASE el contrato de prestación de servicios a honorarios suscrito con "),
            name_p(provider, "uppercase"),
            T(", RUT "),
            rut_p(provider, "spaced_hyphen"),
            T(", por un monto bruto mensual de "),
            amount_p(gross),
            T(", desde el "),
            date_p(start),
            T(" hasta el "),
            date_p(end),
            T("."),
        ]
    )
    cur.paragraph(
        "2° IMPÚTESE el gasto al subtítulo 21, ítem 03, asignación 001, del presupuesto del Gobierno Regional."
    )
    cur.paragraph(
        [
            T("3° NOTIFÍQUESE la presente resolución a la persona interesada al correo "),
            email_p(provider, "underscore"),
            T(" o, en su defecto, por carta certificada al domicilio señalado en el considerando 1."),
        ]
    )
    url = f"https://transparencia.goreficticio.cl/resoluciones/2026/exenta-{folio.split('/')[0]}"
    cur.paragraph([T("4° PUBLÍQUESE en el sitio de transparencia activa "), url_p(url), T(".")])
    cur.space(6)
    cur.line("ANÓTESE, COMUNÍQUESE Y ARCHÍVESE.", font=bold, indent=110)
    cur.space(34)
    cur.ensure(40)
    signature = name_p(head, "uppercase")
    cur.line(signature, font=bold, indent=(cur.width - Sheet.width(signature.text, bold, 11)) / 2)
    position = (
        "Jefa División de Administración y Finanzas"
        if head.sex == "F"
        else "Jefe División de Administración y Finanzas"
    )
    cur.line(position, indent=(cur.width - Sheet.width(position, font, 11)) / 2)
    cur.space(14)
    cur.ensure(90)
    cur.line("Distribución:", font=bold, size=9)
    formats = ["dot", "with_year", "subdomain"]
    for p, format in zip(distribution, formats, strict=True):
        cur.line([T("- "), name_p(p, "exact"), T(", "), email_p(p, format)], size=9, line_spacing=1.35)
    cur.line(
        [T("- "), name_p(provider, "no_accents"), T(", "), email_p(provider, "uppercase")],
        size=9,
        line_spacing=1.35,
    )
    for rest in ("- Oficina de Partes", "- Archivo"):
        cur.line(rest, size=9, line_spacing=1.35)
    initials = "".join(w[0] for w in head.full_name.split()[:3]).upper()
    cur.line(f"{initials}/mfr", size=7)
    if len(doc.sheets) < 2:  # the document must take two pages
        cur.page_break()
        cur.line("Documento firmado electrónicamente conforme a la Ley N° 19.799.", size=9)
    footer(doc, "Gobierno Regional Ficticio · Resolución exenta", font=font)
    path = "pdf_text/resolucion_exenta.pdf"
    doc.save(ctx, path)
    return doc.file_entry(
        "pdft_resolucion_exenta",
        path,
        "Resolución exenta de 2 páginas (VISTOS, CONSIDERANDO, RESUELVO) con datos personales dentro de las "
        "oraciones y lista de distribución con correos.",
        {"font": font},
    )


# ---------------------------------------------------------------------------
# 4. Printed email
# ---------------------------------------------------------------------------


def _printed_email(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    font = "dejavu"
    doc = Document()
    cur = Cursor(doc, font=font, size=9.5, x=50, width=495, top=40, bottom=790)
    bold = _BOLD[font]
    a, b, c = f.person(), f.person(), f.person()
    external = f.person(in_list=False)
    a.landline = f.phone("regional")
    b.landline = f.phone("regional")
    subject = "Informe de honorarios y rendición del mes"
    mailbox = email_p(b, "dot")

    h = cur.sheet
    h.write(50, 30, "Correo institucional", font=bold, size=10, color=(0.2, 0.2, 0.2))
    x = 545 - Sheet.width(f"{b.full_name} <{mailbox.text}>", font, 8.5)
    h.write(x, 30, [name_p(b, "exact"), T(" <"), mailbox, T(">")], font=font, size=8.5)
    h.straight_line(50, 38, 545, 38, thickness=1.0)
    cur.y = 46
    cur.line(subject, font=bold, size=13, line_spacing=1.6)
    cur.line("2 mensajes", size=8.5, color=(0.4, 0.4, 0.4))
    cur.sheet.straight_line(50, cur.y + 2, 545, cur.y + 2, thickness=0.5, color=(0.6, 0.6, 0.6))
    cur.space(8)

    def message(
        sender: Person,
        sender_format: str,
        to: list[tuple[Person, str]],
        cc: list[tuple[Person, str]],
        message_subject: str,
        date: str,
        body: list[list[Part]],
        signature: list[list[Part]],
    ) -> None:
        cur.ensure(60)
        from_ = [name_p(sender, "exact"), T(" <"), email_p(sender, sender_format), T(">")]
        cur.sheet.write(cur.x, cur.y + 10, "De:", font=bold, size=9.5)
        x_end = cur.sheet.write(cur.x + 45, cur.y + 10, from_, font=font, size=9.5)
        x_date = 545 - Sheet.width(date, font, 8.5)
        if x_end + 12 < x_date:
            cur.sheet.write(x_date, cur.y + 10, date_p(date), font=font, size=8.5)
        cur.y += 14
        for label, recipients in (("Para:", to), ("CC:", cc)):
            parts: list[Part] = []
            for i, (p, format) in enumerate(recipients):
                if i:
                    parts.append(T(", "))
                # a recipient's name and email go together: the line is only broken after the comma
                parts += [name_p(p, "exact"), Part(" <", joined=True), email_p(p, format), Part(">", joined=True)]
            cur.sheet.write(cur.x, cur.y + 10, label, font=bold, size=9.5)
            for line in wrap(parts, cur.width - 45, font, 9.5):
                cur.sheet.write(cur.x + 45, cur.y + 10, line, font=font, size=9.5)
                cur.y += 14
        cur.sheet.write(cur.x, cur.y + 10, "Asunto:", font=bold, size=9.5)
        cur.sheet.write(cur.x + 45, cur.y + 10, message_subject, font=font, size=9.5)
        cur.y += 14
        cur.sheet.write(cur.x, cur.y + 10, "Fecha:", font=bold, size=9.5)
        cur.sheet.write(cur.x + 45, cur.y + 10, date_p(date), font=font, size=9.5)
        cur.y += 22
        for paragraph in body:
            cur.paragraph(paragraph, after=6)
        cur.space(4)
        for line in signature:
            cur.line(line, size=9, line_spacing=1.35)
        cur.space(10)
        cur.sheet.straight_line(50, cur.y, 545, cur.y, thickness=0.5, color=(0.6, 0.6, 0.6))
        cur.space(10)

    extension_a, extension_b = (str(int(v)) for v in rng.integers(2000, 7999, 2))
    message(
        a,
        "initial",
        [(b, "dot")],
        [(c, "with_year"), (external, "plus")],
        subject,
        "lunes, 3 de agosto de 2026, 10:42",
        [
            [T("Estimada " if b.sex == "F" else "Estimado "), name_p(b, "first_name_only"), T(":")],
            [
                T(
                    "Junto con saludar, adjunto el informe de honorarios del mes y la planilla de rendición. "
                    "Si necesita aclarar algún punto, me puede llamar al "
                ),  # fmt: skip
                phone_p(a.phone, "mobile_national"),
                T(" durante la mañana o escribirme a "),
                email_p(a, "underscore"),
                T("."),
            ],
            [
                T("También copio a "),
                name_p(external, "partial"),
                T(", de la consultora externa, que revisó los respaldos del producto 2."),
            ],
            [T("Saludos cordiales.")],
        ],
        [
            [T("--")],
            [name_p(a, "exact")],
            [T("Profesional de apoyo, División de Fomento e Industria")],
            [T("Gobierno Regional Ficticio")],
            [T("Móvil: "), phone_p(a.phone, "mobile_international")],
            [T("Fono: "), phone_p(a.landline, "regional_parentheses"), T(f" · Anexo {extension_a}")],
        ],
    )
    message(
        b,
        "dot",
        [(a, "initial")],
        [(c, "with_year")],
        f"RE: {subject}",
        "martes, 4 de agosto de 2026, 16:05",
        [
            [T("Hola "), name_p(a, "first_name_only"), T(":")],
            [
                T(
                    "Recibido, muchas gracias. Revisé el informe y está conforme; solo falta la firma de la "
                    "contraparte. Te llamo mañana al "
                ),  # fmt: skip
                phone_p(a.landline, "regional_hyphen"),
                T(" para coordinar la entrega en la oficina de "),
                T(a.commune),
                T("."),
            ],
            [T("Un abrazo.")],
        ],
        [
            [name_p(b, "surnames_names")],
            [T("Contraparte técnica · Unidad de Control de Gestión")],
            [T("Celular: "), phone_p(b.phone, "mobile_hyphens")],
            [T("Teléfono: "), phone_p(b.landline, "regional_international"), T(f" · anexo {extension_b}")],
            [
                url_p(
                    f"https://www.goreficticio.cl/directorio/{strip_accents(b.paternal_surname).lower()}-{extension_b}"
                )
            ],
        ],
    )
    page_footer = f"https://correo.goreficticio.cl/mail/u/0/?ik=3fa9c{extension_a}&view=pt&search=all"
    for sheet in doc.sheets:
        sheet.write(50, 818, url_p(page_footer), font=font, size=7, color=(0.4, 0.4, 0.4))
        n = f"{sheet.index + 1}/{len(doc.sheets)}"
        sheet.write(545 - Sheet.width(n, font, 7), 818, n, font=font, size=7, color=(0.4, 0.4, 0.4))
    path = "pdf_text/correo_impreso.pdf"
    doc.save(ctx, path)
    return doc.file_entry(
        "pdft_correo_impreso",
        path,
        "Hilo de correo impreso (2 mensajes) con De/Para/CC como 'Nombre <correo>', cuerpo con teléfonos y "
        "firmas con móvil, fijo regional y anexo. Fuente DejaVu incrustada.",
        {"font": font},
    )


# ---------------------------------------------------------------------------
# 5. Rotated page (/Rotate 90)
# ---------------------------------------------------------------------------


def _rotated_page(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    doc = Document()
    cur = Cursor(doc, font="helv", size=10, on_create=gore_header("helv", "División de Desarrollo Social", f.folio()))
    manager = f.person()
    cur.line("ANEXO N° 2 - NÓMINA DE BENEFICIARIOS DEL FONDO REGIONAL", font="hebo", size=12, line_spacing=1.8)
    cur.paragraph(
        [
            T(
                "La nómina de beneficiarios se presenta en la página siguiente, en orientación horizontal. "
                "Consultas a la encargada del programa, "
            ),  # fmt: skip
            name_p(manager, "exact"),
            T(", RUT "),
            rut_p(manager, "dots"),
            T(", correo "),
            email_p(manager, "dot"),
            T(", teléfono "),
            phone_p(phone_for(f, manager, "santiago_parentheses"), "santiago_parentheses"),
            T("."),
        ]
    )
    cur.paragraph([T("Monto total transferido: "), amount_p(f.amount()), T(".")])

    # Page 2: rotated 90°; it is written in visible coordinates (842 x 595).
    h = doc.new_sheet(rotation=90, visible=True)
    h.write(50, 50, "NÓMINA DE BENEFICIARIOS - FONDO REGIONAL DE INICIATIVAS LOCALES 2026", font="hebo", size=12)
    h.write(50, 66, [T("Resolución de adjudicación de "), date_p(f.date())], font="helv", size=9)
    rut_formats = ["dots", "no_dots", "spaced_hyphen", "en_dash", "dots", "no_dots", "dots"]
    phone_formats = ["mobile_international", "regional_parentheses", "mobile_block", "santiago_national",
                     "mobile_compact", "regional_hyphen", "mobile_parentheses"]  # fmt: skip
    email_formats = ["dot", "initial", "with_year", "underscore", "plus", "subdomain", "dot"]
    rows = []
    for i in range(7):
        p = f.person(dv_valid=i != 2)
        rows.append(
            [
                name_p(p, "exact" if i % 3 else "uppercase"),
                rut_p(p, rut_formats[i]),
                email_p(p, email_formats[i]),
                phone_p(phone_for(f, p, phone_formats[i]), phone_formats[i]),
                address_p(p),
                T(p.commune),
                amount_p(f.amount()),
            ]
        )
    y = draw_table(h, 50, 82, 742, ["Nombre", "RUT", "Correo", "Teléfono", "Dirección", "Comuna", "Monto"], rows,
                   font="helv", size=8)  # fmt: skip
    h.write(50, y + 18, "Fuente: Unidad de Control de Gestión, Gobierno Regional Ficticio.", font="helv", size=8)
    rotated_footer = f"Página 2 de 2 · Folio de nómina {f.folio()}"
    h.write(792 - Sheet.width(rotated_footer, "helv", 7.5), 570, rotated_footer, font="helv", size=7.5)
    doc.sheets[0].write(60, 812, "Gobierno Regional Ficticio · Página 1 de 2", font="helv", size=7.5)
    path = "pdf_text/pagina_rotada.pdf"
    doc.save(ctx, path)
    return doc.file_entry(
        "pdft_pagina_rotada",
        path,
        "Página 1 vertical normal; página 2 con /Rotate 90 que contiene una tabla horizontal con datos personales "
        "(el texto se escribió girado en el espacio sin rotar para verse horizontal).",
        {"rotations": [0, 90]},
    )


# ---------------------------------------------------------------------------
# 6. Rotated text (rotate 90/180/270 and a stamp at 30° with morph)
# ---------------------------------------------------------------------------


def _rotated_text(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    doc = Document()
    cur = Cursor(doc, font="helv", size=10, on_create=gore_header("helv", "Oficina de Partes", f.folio()))
    signer, addressee, receiver, holder = f.person(), f.person(), f.person(), f.person()
    cur.line("CERTIFICADO DE RECEPCIÓN DE DOCUMENTOS", font="hebo", size=12, line_spacing=1.8)
    cur.paragraph(
        [
            T("Se certifica que "),
            name_p(holder, "exact"),
            T(", RUT "),
            rut_p(holder, "dots"),
            T(", ingresó la documentación de respaldo del proyecto el día "),
            date_p(f.date()),
            T(". Para consultas escribir a "),
            email_p(holder, "dot"),
            T("."),
        ]
    )
    for phrase in rng.choice(len(ADMINISTRATIVE_PHRASES), size=4, replace=False):
        cur.paragraph(ADMINISTRATIVE_PHRASES[int(phrase)])
    h = doc.sheets[0]

    # Left margin, 90°: read from bottom to top.
    h.write(
        34,
        760,
        [
            T("Firmado electrónicamente por "),
            name_p(signer, "exact"),
            T(", RUT "),
            rut_p(signer, "no_dots"),
            T(", el "),
            date_p(f.date()),
        ],
        font="helv",
        size=8,
        angle=90,
    )
    # Right margin, 270°: read from top to bottom.
    url = f"https://verificador.goreficticio.cl/doc/{int(rng.integers(10**7, 10**8))}"
    h.write(
        562,
        90,
        [T("Verifique este documento en "), url_p(url), T(" · contacto "), email_p(signer, "initial")],
        font="helv",
        size=8,
        angle=270,
    )
    # Upside-down footer, 180°.
    h.write(
        480,
        770,
        [
            T("Copia para "),
            name_p(addressee, "exact"),
            T(" · fono "),
            phone_p(addressee.phone, "mobile_international"),
        ],  # fmt: skip
        font="tiro",
        size=9,
        angle=180,
    )
    # Stamp at 30° (morph), centered at (400, 600).
    angle = 30.0
    cx, cy, w, height = 400.0, 600.0, 190.0, 84.0
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)

    def turn(lx: float, ly: float) -> tuple[float, float]:
        return cx + lx * c + ly * s, cy - lx * s + ly * c

    ink = (0.1, 0.2, 0.65)
    for margin in (0.0, 4.0):
        corners = [
            turn(sx * (w / 2 - margin), sy * (height / 2 - margin)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        h.page.draw_polyline([*corners, corners[0]], color=ink, width=1.2 if margin == 0 else 0.6)
    lines = [
        ([T("OFICINA DE PARTES")], "hebo", 10, -22),
        ([T("RECIBIDO "), date_p(f.date())], "helv", 8.5, -6),
        ([name_p(receiver, "uppercase")], "helv", 8.5, 10),
        ([T("RUT "), rut_p(receiver, "dots")], "helv", 8.5, 26),
    ]
    for parts, font, size, ly in lines:
        length = Sheet.width("".join(p.text for p in parts), font, size)
        px, py = turn(-length / 2, ly)
        h.write(px, py, parts, font=font, size=size, angle=angle, color=ink, tags={"stamp": True})
    footer(doc, "Gobierno Regional Ficticio · Oficina de Partes")
    path = "pdf_text/texto_girado.pdf"
    doc.save(ctx, path)
    return doc.file_entry(
        "pdft_texto_girado",
        path,
        "Texto en la capa de texto girado 90°, 180° y 270° (márgenes y pie) y un timbre a 30° con morph.",
        {"angles": [0, 90, 180, 270, 30]},
    )


# ---------------------------------------------------------------------------
# 7. Stress cases of the text layer
# ---------------------------------------------------------------------------


def _text_layer_stress(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    doc = Document()
    h = doc.new_sheet()
    # CropBox smaller than the MediaBox: from here on all coordinates are relative to the CropBox.
    mediabox = pymupdf.Rect(0, 0, *A4)
    crop = pymupdf.Rect(30, 30, A4[0] - 30, A4[1] - 30)
    h.page.set_cropbox(crop)
    x = 30
    y = 40.0

    def title(text: str) -> None:
        nonlocal y
        y += 22
        h.write(x, y, text, font="hebo", size=10)
        y += 16

    h.write(x, y, "PRUEBAS DE LA CAPA DE TEXTO - CASOS DIFÍCILES", font="hebo", size=13)
    y += 6

    title("a) Texto diminuto (4 y 5 pt)")
    p = f.person()
    h.write(
        x,
        y,
        [
            T("Nota al pie: consultas a "),
            email_p(p, "dot", "stress"),
            T(" o al "),
            phone_p(p.phone, "mobile_international", "stress"),
            T("."),
        ],  # fmt: skip
        size=5,
        tags={"tiny": True},
    )
    y += 8
    h.write(x, y, [T("Titular del registro: RUT "), rut_p(p, "dots", "stress")], size=4, tags={"tiny": True})

    title("b) Blanco sobre blanco, modo de dibujo invisible y texto bajo una imagen")
    q = f.person()
    h.write(x, y, "Campo reservado:", size=10)
    h.write(x + 95, y, [T("RUT "), rut_p(q, "dots")], size=10, color=(1, 1, 1), layer="hidden",
            tags={"hidden": "white_on_white"})  # fmt: skip
    y += 16
    h.write(x, y, "Texto invisible:", size=10)
    h.write(x + 95, y, [T("Correo "), email_p(q, "with_year"), T(" RUT "), rut_p(q, "no_dots")], size=10,
            render_mode=3, layer="hidden", tags={"hidden": "render_mode_3"})  # fmt: skip
    y += 30
    r = f.person()
    h.write(x, y + 12, "Nota adhesiva pegada sobre el texto:", size=10)
    under = {"hidden": "under_image"}
    h.write(x + 200, y + 10, [T("Correo: "), email_p(r, "dot"), T("  RUT: "), rut_p(r, "dots")], size=8,
            layer="hidden", tags=under)  # fmt: skip
    h.write(x + 200, y + 22, [T("Fono: "), phone_p(r.phone, "mobile_national")], size=8, layer="hidden", tags=under)
    # Opaque, plain image on top (the "REVISADO" mark stays far from the covered text).
    note = Image.new("RGB", (580, 80), (255, 238, 150))
    ImageDraw.Draw(note).text((500, 62), "REVISADO", fill=(120, 90, 20))
    h.page.insert_image(pymupdf.Rect(x + 190, y - 6, x + 480, y + 34), stream=_png(note), keep_proportion=False)
    y += 36

    title("c) Texto fuera del CropBox (presente en el archivo, fuera de la página visible)")
    s = f.person()
    outside = {
        "hidden": "outside_cropbox",
        "coordinate_system": "espacio de página PyMuPDF: origen en la esquina superior izquierda del CropBox; "
        "coordenadas negativas o mayores que el ancho/alto quedan fuera de page.rect",
        "mediabox": list(mediabox),
        "cropbox": list(crop),
        "extraction": "page.get_textpage(clip=pymupdf.INFINITE_RECT())",
    }
    h.write(-24, -12, [T("Borrador: "), replace(name_p(s, "exact"), level="stress"), T(" RUT "),
            rut_p(s, "dots", "stress")],
            size=9, layer="hidden", tags=outside)  # fmt: skip
    # Right strip between the CropBox and the MediaBox, rotated 270° so it fits.
    h.write(crop.width + 12, 120, [T("Correo "), email_p(s, "dot", "stress")], size=9, angle=270, layer="hidden",
            tags=outside)  # fmt: skip
    h.write(x, y, "Hay un nombre, un RUT y un correo escritos fuera del recorte visible de esta página.", size=9)

    title("d) RUT cortado entre dos líneas")
    u = f.person()
    full = u.rut("dots")
    cut = full.index(".", 3) + 1
    fr0, fr1 = full[:cut], full[cut:]
    base_tags = {"full_value": full, "format": "dots", "dv_valid": u.rut_dv_valid}
    x_end = h.write(x, y, [T("El trámite fue presentado por "), name_p(u, "exact"), T(", con RUT ")], size=10)
    h.write(x_end, y, Part(fr0, "rut", "stress", {"fragment": 0, **base_tags}), size=10)
    y += 14
    h.write(x, y, [Part(fr1, "rut", "stress", {"fragment": 1, **base_tags}),
            T(", según consta en el expediente del programa.")], size=10)  # fmt: skip

    title("e) Texto con letras espaciadas (cada carácter por separado)")
    v = f.person()
    x_end = h.write(x, y, "RUT del beneficiario: ", size=10)
    h.write_spaced(x_end, y, rut_p(v, "dots", "stress"), 2.6, size=10)
    y += 16
    x_end = h.write(x, y, "Correo: ", size=10)
    h.write_spaced(x_end, y, email_p(v, "dot", "stress"), 1.8, size=10)

    title("f) RUT con guion largo (autocorrección del procesador de texto)")
    w = f.person()
    h.write(
        x,
        y,
        [T("Postulante: "), name_p(w, "exact"), T(", RUT "), rut_p(w, "en_dash"), T(".")],
        font="helv",
        size=10,
    )
    # Old or disguised formats that do not appear in other files (no new random draws).
    title("g) Correo con (at) y celular antiguo de 8 dígitos")
    h.write(
        x,
        y,
        [T("Contacto: "), email_p(w, "at"), T(", celular antiguo "), phone_p(w.phone, "old_mobile_8"), T(".")],
        size=10,
    )

    title("h) URL que contiene un RUT")
    body, dv = f.rut()
    url_rut = f"{body}-{dv}"
    url = f"https://tramites.ejemplo.cl/certificado?rut={url_rut}&tipo=honorarios"
    x_end = h.write(x, y, "Descargue su certificado en ", size=10)
    h.write(x_end, y, url_p(url), size=10)
    prefix = url[: url.index(url_rut)]
    x_rut = x_end + Sheet.width(prefix, "helv", 10)
    h.write(x_rut, y, Part(url_rut, "rut", "base", {"format": "no_dots", "dv_valid": True, "inside_url": True}),
            size=10, draw=False)  # fmt: skip

    path = "pdf_text/estres_capa_texto.pdf"
    doc.save(ctx, path)
    return doc.file_entry(
        "pdft_estres_capa_texto",
        path,
        "Casos difíciles de la capa de texto: letra de 4-5 pt, blanco sobre blanco, render mode 3, texto bajo "
        "una imagen, texto fuera del CropBox, RUT cortado entre líneas, letras espaciadas, guion largo, correo "
        "con (at), celular antiguo de 8 dígitos y RUT dentro de una URL.",
        {"cropbox": list(crop), "mediabox": list(mediabox)},
    )


# ---------------------------------------------------------------------------
# 8. Vectorized text (glyphs as paths)
# ---------------------------------------------------------------------------


def _vectorized_text(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    source = Document()
    cur = Cursor(
        source, font="helv", size=10, on_create=gore_header("helv", "División de Fomento e Industria", f.folio())
    )
    p, q = f.person(), f.person()
    cur.line("INFORME DE HONORARIOS (VERSIÓN IMPRESA COMO IMAGEN VECTORIAL)", font="hebo", size=12, line_spacing=1.8)
    cur.field("Nombre:", name_p(p, "exact"))
    cur.field("RUT:", rut_p(p, "dots"))
    cur.field("Correo:", email_p(p, "dot"))
    cur.field("Teléfono:", phone_p(p.phone, "mobile_international"))
    cur.field("Dirección:", [address_p(p), T(f", {p.commune}")])
    cur.field("Monto bruto:", amount_p(f.amount()))
    cur.section("Contraparte técnica")
    cur.paragraph(
        [
            T("La contraparte técnica, "),
            name_p(q, "exact"),
            T(", RUT "),
            rut_p(q, "en_dash"),
            T(", correo "),
            email_p(q, "initial"),
            T(", teléfono "),
            phone_p(phone_for(f, q, "regional_parentheses"), "regional_parentheses"),
            T(", validó los productos el "),
            date_p(f.date()),
            T("."),
        ],
        font="dejavu",
    )
    for phrase in rng.choice(len(ADMINISTRATIVE_PHRASES), size=3, replace=False):
        cur.paragraph(ADMINISTRATIVE_PHRASES[int(phrase)])
    source.refine()
    page = source.doc[0]
    svg = page.get_svg_image(text_as_path=True)
    svg_doc = pymupdf.open("svg", svg.encode("utf-8"))
    final = pymupdf.open("pdf", svg_doc.convert_to_pdf())
    svg_doc.close()
    dest = final[0]
    sx, sy = dest.rect.width / page.rect.width, dest.rect.height / page.rect.height
    if final[0].get_text().strip():
        raise RuntimeError("the vectorized PDF still has extractable text")
    elements = []
    for e in source.elements():
        e.layer = "vector"
        e.polygon = [[round(px * sx, 3), round(py * sy, 3)] for px, py in e.polygon]
        e.tags["gt_origin"] = "search_for en la página original, antes de vectorizar"
        elements.append(e)
    path = "pdf_text/texto_vectorizado.pdf"
    final.save(ctx.path(path), garbage=3, deflate=True, no_new_id=True)
    file_entry = FileEntry(
        id="pdft_texto_vectorizado",
        path=path,
        format="pdf",
        category=CATEGORY,
        description="Página de texto convertida a SVG con text_as_path=True y de vuelta a PDF: los glifos son trazos "
        "(visibles, no extraíbles).",
        pages=pages_of(final),
        elements=elements,
        tags={"scale": [sx, sy]},
    )
    final.close()
    source.doc.close()
    return file_entry


# ---------------------------------------------------------------------------
# 9. Text with embedded images
# ---------------------------------------------------------------------------


def _canvas_line(canvas: Canvas, x: float, y: float, parts: Sequence[Part], size: int, font: str = "sans",
                 color: tuple[int, int, int] = (30, 30, 30)) -> float:  # fmt: skip
    for p in parts:
        visible = any(c.isalnum() for c in p.text)
        canvas.write_text(x, y, p.text, type=p.type, value=p.value or p.text.strip(), font_name=font, size=size,
                          color=color, level=p.level, tags=dict(p.tags), register=visible)  # fmt: skip
        x += canvas.text_width(p.text, font, size)
    return x


def _image_at(h: Sheet, box: tuple[float, float, float, float], img: Image.Image, elements: Sequence[Element],
              tags: dict[str, Any], xref: int = 0, stream: bytes | None = None) -> int:  # fmt: skip
    """Inserts the image stretched exactly to ``box`` and records its elements in page coordinates."""
    r = pymupdf.Rect(*box)
    if xref:
        new = h.page.insert_image(r, xref=xref, keep_proportion=False)
    else:
        new = h.page.insert_image(r, stream=stream or _png(img), keep_proportion=False)
    mapped = map_to_rect(elements, img.width, img.height, box)
    for e in mapped:
        e.tags.update(tags)
        e.tags["scale_pt_per_px"] = round((box[2] - box[0]) / img.width, 4)
    h.add(mapped)
    return new


def _mixed_images(ctx: Context, f: FakeData, rng: np.random.Generator) -> FileEntry:
    doc = Document()
    cur = Cursor(
        doc,
        font="helv",
        size=10,
        width=340,
        on_create=gore_header("helv", "Programa de Prácticas Profesionales", f.folio()),
    )
    applicant, sender, credential = f.person(), f.person(), f.person()
    cur.line("FICHA DE POSTULACIÓN - PROGRAMA DE PRÁCTICAS", font="hebo", size=12, line_spacing=1.8)
    cur.field("Nombre:", name_p(applicant, "exact"), tab=80)
    cur.field("RUT:", rut_p(applicant, "dots"), tab=80)
    cur.field("Correo:", email_p(applicant, "dot"), tab=80)
    cur.field("Teléfono:", phone_p(applicant.phone, "mobile_international"), tab=80)
    cur.field("Dirección:", address_p(applicant), tab=80)
    cur.field("Comuna:", applicant.commune, tab=80)
    cur.width = 475
    cur.space(10)
    for phrase in rng.choice(len(ADMINISTRATIVE_PHRASES), size=3, replace=False):
        cur.paragraph(ADMINISTRATIVE_PHRASES[int(phrase)])
    h1 = doc.sheets[0]

    # (a) Application photo.
    (photo,) = ctx.faces.take("front", 1)
    photo_width = 100.0
    box = (435.0, 80.0, 435.0 + photo_width, 80.0 + photo_width * photo.img.height / photo.img.width)
    _image_at(h1, box, photo.img, FaceProvider.elements(photo), {"image": "application_photo"})
    h1.write(435, box[3] + 10, "Fotografía", size=7.5)

    # (b) Screenshot of a message, inserted at scale.
    cur.space(8)
    cur.line("Respaldo: captura del mensaje recibido", font="hebo", size=9)
    canvas = Canvas.new(640, 230, background=(246, 247, 249))
    d = ImageDraw.Draw(canvas.img)
    d.rectangle((0, 0, 640, 34), fill=(46, 84, 150))
    _canvas_line(canvas, 14, 23, [T("Mensajería institucional")], 15, "sans_bold", (255, 255, 255))
    _canvas_line(canvas, 20, 64, [T("De: "), name_p(sender, "exact"), T(" <"), email_p(sender, "dot"), T(">")], 14)
    _canvas_line(canvas, 20, 88, [T("Asunto: Documentos de la postulación")], 14)
    d.line((20, 100, 620, 100), fill=(200, 200, 205), width=1)
    _canvas_line(canvas, 20, 128, [T("Hola, adjunto los documentos. Cualquier duda me llamas al")], 14)
    _canvas_line(canvas, 20, 150, [phone_p(sender.phone, "mobile_national"), T(" o al fijo "),
                 phone_p(phone_for(f, sender, "regional_parentheses"), "regional_parentheses"), T(".")], 14)  # fmt: skip
    _canvas_line(canvas, 20, 185, [T("Saludos, "), name_p(sender, "partial")], 14)
    y0 = cur.y + 4
    box = (60.0, y0, 60.0 + 400.0, y0 + 400.0 * canvas.height / canvas.width)
    _image_at(h1, box, canvas.img, canvas.elements, {"image": "screenshot"})
    cur.y = box[3] + 10
    cur.paragraph("La captura se incorpora como antecedente de la postulación, a solicitud de la contraparte.")

    # Page 2
    cur.page_break()
    h2 = cur.sheet
    cur.line("CREDENCIAL Y CÓDIGOS DE VERIFICACIÓN", font="hebo", size=12, line_spacing=1.8)
    cur.line([T("Titular de la credencial: "), name_p(credential, "exact")])

    # (e) The same photo twice (same xref), the second one smaller.
    (photo2,) = ctx.faces.take("front", 1)
    rel_height = photo2.img.height / photo2.img.width
    big_box = (60.0, 130.0, 200.0, 130.0 + 140.0 * rel_height)
    xref = _image_at(h2, big_box, photo2.img, FaceProvider.elements(photo2),
                     {"image": "credential", "shared_xref": True, "occurrence": 1})  # fmt: skip
    small_box = (215.0, 130.0, 255.0, 130.0 + 40.0 * rel_height)
    _image_at(h2, small_box, photo2.img, FaceProvider.elements(photo2),
              {"image": "credential", "shared_xref": True, "occurrence": 2}, xref=xref)  # fmt: skip

    # (c) RUT in an image rotated 90°.
    rut_c = rut_p(credential, "dots")
    rut_canvas = Canvas.new(340, 56)
    _canvas_line(rut_canvas, 10, 40, [T("RUT "), rut_c], 30, "sans_bold")
    rotated = rut_canvas.rotate(90)
    rut_box = (290.0, 130.0, 290.0 + 22.0, 130.0 + 22.0 * rotated.height / rotated.width)
    _image_at(h2, rut_box, rotated.img, rotated.elements, {"image": "rotated_rut"})

    # (d) QR code with a RUT inside a URL.
    body, dv = f.rut()
    content = f"https://portal.ejemplo.cl/verificar?run={body}-{dv}"
    qr_img, modules_box = _qr(content)
    qr_box = (350.0, 130.0, 460.0, 240.0)
    qr_element = Element(type="qr", page=0, polygon=rect(*modules_box), value=content, level="stress", layer="raster",
                         tags={"content": "url_with_rut", "encoded_rut": f"{body}-{dv}"})  # fmt: skip
    _image_at(h2, qr_box, qr_img, [qr_element], {"image": "qr"})
    h2.write(350, 252, "Escanee para verificar la credencial", size=7.5)

    # (f) PNG with transparency (SMask) over a colored background.
    phone = f.phone("mobile")
    phone_canvas = Canvas.new(420, 56)
    _canvas_line(phone_canvas, 10, 38, [T("Tel. "), phone_p(phone, "mobile_international")], 28, "sans_bold")
    alpha = Image.eval(phone_canvas.img.convert("L"), lambda v: 255 - v)
    rgba = Image.new("RGBA", phone_canvas.img.size, (20, 60, 120, 0))
    rgba.putalpha(alpha)
    phone_box = (60.0, 330.0, 60.0 + 280.0, 330.0 + 280.0 * 56 / 420)
    h2.rectangle(50, 320, 360, phone_box[3] + 10, fill=(0.86, 0.93, 1.0))
    _image_at(h2, phone_box, phone_canvas.img, phone_canvas.elements, {"image": "transparent_png"}, stream=_png(rgba))
    h2.write(60, phone_box[3] + 26, "Mesa de ayuda del programa (imagen PNG con transparencia).", size=8)
    footer(doc, "Gobierno Regional Ficticio · Programa de Prácticas Profesionales")
    path = "pdf_text/mixto_imagenes.pdf"
    doc.save(ctx, path)
    return doc.file_entry(
        "pdft_mixto_imagenes",
        path,
        "Texto con datos en la capa de texto más imágenes incrustadas: foto, pantallazo escalado, RUT en imagen "
        "girada 90°, código QR con RUT, la misma foto dos veces (mismo xref) y PNG con transparencia.",
        {"synthetic_faces": ctx.faces.synthetic_only},
    )


def _qr(content: str, module: int = 6, border: int = 4) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """QR image (with quiet zone) and the box of the modules in pixels."""
    matrix = cv2.QRCodeEncoder.create().encode(content)
    if matrix.ndim == 3:
        matrix = matrix[..., 0]
    n = matrix.shape[0]
    big = np.kron(matrix, np.ones((module, module), np.uint8))
    side = (n + 2 * border) * module
    canvas = np.full((side, side), 255, np.uint8)
    o = border * module
    canvas[o : o + n * module, o : o + n * module] = big
    return Image.fromarray(canvas).convert("RGB"), (o, o, o + n * module, o + n * module)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate(ctx: Context) -> list[FileEntry]:
    f = ctx.fake_data(SEED_NAME)
    rng = ctx.rng(SEED_NAME)
    files = [_notebook(ctx, f)]
    for n in range(1, 5):
        files.append(_fees_report(ctx, f, rng, n))
    files.append(_resolution(ctx, f, rng))
    files.append(_printed_email(ctx, f, rng))
    files.append(_rotated_page(ctx, f, rng))
    files.append(_rotated_text(ctx, f, rng))
    files.append(_text_layer_stress(ctx, f, rng))
    files.append(_vectorized_text(ctx, f, rng))
    files.append(_mixed_images(ctx, f, rng))
    return files
