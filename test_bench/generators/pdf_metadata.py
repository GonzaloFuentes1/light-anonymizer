"""PDFs with personal data hidden in metadata and non-visible structures.

Each structure carries its own canary (a unique value of a different person), so that if it
shows up in the output it is known exactly which structure was not cleaned:

1. ``metadatos_completos.pdf``: Info dictionary, XMP packet, annotations (note, free text,
   highlight with a popup), embedded file, file attachment annotation, optional layer turned
   off, form field, bookmark and document JavaScript.
2. ``revision_incremental.pdf``: a line with a RUT is deleted in a second revision saved
   incrementally; the previous revision is still in the bytes of the file.
3. ``redaccion_falsa.pdf``: the classic "redaction by transparency" leak: black rectangles
   drawn over text that is still extractable, an unapplied redaction annotation and a black
   square annotation over a phone number.
4. ``solo_permisos.pdf``: encrypted with an owner password only (restricts copying and
   printing, but opens without a password). It must be processed.

Coordinates are in PyMuPDF's unrotated page space (the one of ``search_for``).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pymupdf

from test_bench.canvas import font_path
from test_bench.context import Context
from test_bench.fake_data import (
    ADMINISTRATIVE_PHRASES,
    EMAIL_FORMATS,
    PHONE_FORMATS,
    RUT_FORMATS,
    FakeData,
    Person,
    strip_accents,
)
from test_bench.schema import Element, FileEntry, MetadataEntry, Page, Polygon

SEED_NAME = "pdf_metadatos"  # seed string: kept in Spanish so the generated files do not change
CATEGORY = "pdf_metadata"
PREFIX = "pdfm"

WIDTH, HEIGHT = 612.0, 792.0  # letter size, the usual one in Chilean public documents
MARGIN = 64.0
LINE_SPACING = 17.0

# Owner password of the permissions-only file (not personal data).
OWNER_PASSWORD = "propietario-ficticio-2026"

# PyMuPDF font alias -> name in ``canvas.FONTS``. "helv" is the base-14 Helvetica, used when
# the text should end up as a literal string in the content stream.
FONT_ALIASES = {
    "sans": "dvs",
    "sans_bold": "dvsb",
    "serif": "dvse",
    "serif_bold": "dvseb",
}

CREATION_DATE = "D:20260312091500-03'00'"
MODIFICATION_DATE = "D:20260318164210-03'00'"


@lru_cache(maxsize=16)
def _font(name: str) -> pymupdf.Font:
    if name == "helv":
        return pymupdf.Font("helv")
    return pymupdf.Font(fontfile=str(font_path(name)))


def quad_to_polygon(q: pymupdf.Quad) -> Polygon:
    return [[q.ul.x, q.ul.y], [q.ur.x, q.ur.y], [q.lr.x, q.lr.y], [q.ll.x, q.ll.y]]


def rect_to_polygon(r: pymupdf.Rect) -> Polygon:
    return [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]]


def person_level(p: Person) -> str:
    return "base" if p.in_list else "out_of_scope"


def pdf_literal(text: str) -> str:
    """PDF literal string (between parentheses) with the special characters escaped."""
    return "(" + text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


def helv_to_literal(doc: pymupdf.Document, page: pymupdf.Page) -> None:
    """Rewrites the page's Helvetica text as literal strings ``(…) Tj``.

    PyMuPDF writes base-14 text in hexadecimal; as a Latin-1 literal, the value is readable
    as is in the decompressed stream (so a byte search finds it, as in a PDF produced by many
    tools).
    """
    # PyMuPDF writes "/helv 10 Tf <colors> [<hex>]TJ"; everything between Tf and TJ is kept.
    pattern = re.compile(rb"(/helv [\d.]+ Tf[^\[(]*?)\[<([0-9a-fA-F]+)>\]TJ")

    def replacement(m: re.Match[bytes]) -> bytes:
        text = bytes.fromhex(m.group(2).decode()).decode("latin-1")
        return m.group(1) + pdf_literal(text).encode("latin-1") + b" Tj"

    for xref in page.get_contents():
        stream = doc.xref_stream(xref)
        new = pattern.sub(replacement, stream)
        if new != stream:
            doc.update_stream(xref, new)


def fix_attachment_dates(doc: pymupdf.Document) -> None:
    """Replaces the current date PyMuPDF puts in ``/Params`` of each ``/EmbeddedFile`` with a fixed one.

    Without this the PDF bytes change on every generation (and the machine's time zone ends
    up written in the file).
    """
    for xref in range(1, doc.xref_length()):
        if doc.xref_get_key(xref, "Type") != ("name", "/EmbeddedFile"):
            continue
        if doc.xref_get_key(xref, "Params")[0] != "dict":
            continue
        doc.xref_set_key(xref, "Params/CreationDate", pdf_literal(CREATION_DATE))
        doc.xref_set_key(xref, "Params/ModDate", pdf_literal(CREATION_DATE))


class PdfSheet:
    """Writes text on a PyMuPDF page and records each piece as an Element with its quadrilateral.

    The quadrilateral is the one ``page.search_for(..., quads=True)`` returns, limited to the
    typographic box of the piece just written; for invisible text (optional layer turned off),
    which ``search_for`` does not see, the typographic box computed from the font metrics is
    used (the same convention: font ascender and descender times the size).
    """

    def __init__(self, page: pymupdf.Page, index: int = 0) -> None:
        self.page = page
        self.index = index
        self.elements: list[Element] = []
        for name, alias in FONT_ALIASES.items():
            page.insert_font(fontname=alias, fontfile=str(font_path(name)))

    def width(self, text: str, size: float = 10.0, font: str = "sans") -> float:
        return _font(font).text_length(text, size)

    def box(self, x: float, y: float, text: str, size: float, font: str) -> pymupdf.Rect:
        f = _font(font)
        return pymupdf.Rect(x, y - f.ascender * size, x + f.text_length(text, size), y - f.descender * size)

    def find(self, text: str, clip: pymupdf.Rect) -> Polygon:
        """Unique quadrilateral of ``text`` inside ``clip`` according to PyMuPDF (fails unless there is exactly one)."""
        found = self.page.search_for(text, clip=clip, quads=True)
        if len(found) != 1:
            raise RuntimeError(f"expected one occurrence of {text!r} in {clip}, found {len(found)}")
        return quad_to_polygon(found[0])

    def write(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: float = 10.0,
        font: str = "sans",
        type: str = "text",
        level: str = "base",
        layer: str = "text",
        tags: dict[str, Any] | None = None,
        color: tuple[float, float, float] = (0.1, 0.1, 0.1),
        oc: int = 0,
        register: bool = True,
    ) -> float:
        """Writes ``text`` with its baseline at (x, y); returns the x where it ends."""
        alias = "helv" if font == "helv" else FONT_ALIASES[font]
        self.page.insert_text((x, y), text, fontname=alias, fontsize=size, color=color, oc=oc)
        box = self.box(x, y, text, size, font)
        if register:
            if layer == "hidden":
                polygon = rect_to_polygon(box)
            else:
                polygon = self.find(text, box + (-2, -2, 2, 2))
            self.elements.append(
                Element(
                    type=type,
                    page=self.index,
                    polygon=polygon,
                    value=text,
                    level=level,
                    layer=layer,
                    tags={"font": font, "size_pt": size, **(tags or {})},
                )
            )
        return box.x1

    def field(
        self,
        x: float,
        y: float,
        label: str,
        value: str,
        *,
        type: str,
        x_value: float | None = None,
        size: float = 10.0,
        **kwargs: Any,
    ) -> float:
        """Neutral label ("RUT:") followed by the datum; the label is recorded as ``text``."""
        end = self.write(x, y, label, size=size, font=kwargs.pop("label_font", "sans_bold"))
        start = x_value if x_value is not None else end + self.width(" ", size)
        return self.write(start, y, value, size=size, type=type, **kwargs)

    def datum(
        self,
        x: float,
        y: float,
        p: Person,
        kind: str,
        format: str = "",
        *,
        size: float = 10.0,
        level: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> float:
        """Writes a datum of the person with its type, level and tags according to the manifest rules."""
        text, type_, default_level, datum_tags = person_value(p, kind, format)
        return self.write(
            x,
            y,
            text,
            size=size,
            type=type_,
            level=level or default_level,
            tags={**datum_tags, **(tags or {})},
            **kwargs,
        )


def person_value(p: Person, kind: str, format: str = "") -> tuple[str, str, str, dict[str, Any]]:
    """(text, type, level, tags) of a datum of the person.

    ``kind``: name, address, rut, email, mobile, landline.
    """
    match kind:
        case "name":
            return p.full_name, "name", person_level(p), {"in_list": p.in_list, "variant": "full"}
        case "address":
            return p.address, "address", person_level(p), {"in_list": p.in_list}
        case "rut":
            format = format or "dots"
            tags = {"format": format, "dv_valid": p.rut_dv_valid}
            return p.rut(format), "rut", RUT_FORMATS[format], tags
        case "email":
            format = format or "dot"
            return p.email(format), "email", EMAIL_FORMATS[format], {"format": format}
        case "mobile" | "landline":
            phone = p.phone if kind == "mobile" else p.landline
            format = format or {"mobile": "mobile_international", "santiago": "santiago_international"}.get(
                phone.kind, "regional_parentheses"
            )
            tags = {"format": format, "kind": phone.kind}
            return phone.format(format), "phone", PHONE_FORMATS[format], tags
    raise ValueError(kind)


def landline_format(p: Person, preferred: str) -> str:
    """Landline phone format compatible with the kind of the person's landline."""
    if p.landline.kind == "santiago":
        return {"international": "santiago_international", "parentheses": "santiago_parentheses"}.get(
            preferred, "santiago_national"
        )
    return {"international": "regional_international", "parentheses": "regional_parentheses"}.get(
        preferred, "regional_hyphen"
    )


def pages_of(path: Path) -> list[Page]:
    """Pages in PyMuPDF's unrotated space (width and height of the crop box)."""
    with pymupdf.open(path) as d:
        return [
            Page(index=i, width=p.cropbox.width, height=p.cropbox.height, unit="pt", rotation=p.rotation)
            for i, p in enumerate(d)
        ]


def _header(sheet: PdfSheet, title: str, subtitle: str, folio: str, date: str) -> float:
    """Neutral institutional header; returns the y of the next line."""
    y = 72.0
    sheet.write(MARGIN, y, "GOBIERNO REGIONAL DE LA REGIÓN FICTICIA", size=9, font="sans_bold")
    y += 13
    sheet.write(MARGIN, y, "División de Presupuesto e Inversión Regional", size=9)
    y += 30
    sheet.write(MARGIN, y, title, size=14, font="serif_bold")
    y += 18
    sheet.write(MARGIN, y, subtitle, size=10.5, font="serif")
    y += 24
    sheet.field(MARGIN, y, "Folio:", folio, type="text")
    end = sheet.write(360, y, "Fecha:", font="sans_bold")
    sheet.write(end + sheet.width(" "), y, date, tags={"decoy": "date"})
    sheet.page.draw_line((MARGIN, y + 9), (WIDTH - MARGIN, y + 9), color=(0.3, 0.3, 0.3), width=0.6)
    return y + 30


def _paragraphs(sheet: PdfSheet, y: float, phrases: list[str], size: float = 10.0) -> float:
    for phrase in phrases:
        sheet.write(MARGIN, y, phrase, size=size, font="serif")
        y += LINE_SPACING
    return y


def _phrases(rng: Any, n: int) -> list[str]:
    indices = rng.choice(len(ADMINISTRATIVE_PHRASES), size=n, replace=False)
    return [ADMINISTRATIVE_PHRASES[int(i)] for i in indices]


# ---------------------------------------------------------------------------
# 1. Full metadata
# ---------------------------------------------------------------------------


def _xmp(creator: str, rut: str) -> str:
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n'
        '    xmlns:gore="http://ns.ficticio.cl/gore/1.0/">\n'
        "   <dc:creator><rdf:Seq><rdf:li>" + creator + "</rdf:li></rdf:Seq></dc:creator>\n"
        '   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">Informe de actividades</rdf:li></rdf:Alt></dc:title>\n'
        "   <xmp:CreatorTool>Microsoft® Word para Microsoft 365</xmp:CreatorTool>\n"
        "   <xmp:CreateDate>2026-03-12T09:15:00-03:00</xmp:CreateDate>\n"
        "   <gore:rutFuncionario>" + rut + "</gore:rutFuncionario>\n"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )


def _full_metadata(ctx: Context, f: FakeData, rng: Any) -> FileEntry:
    rel_path = "pdf_metadata/metadatos_completos.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=WIDTH, height=HEIGHT)
    sheet = PdfSheet(page)

    # -- visible content: small block with personal data --------------------------
    holder = f.person()
    counterpart = f.person(in_list=False)
    y = _header(
        sheet, "Informe de actividades", "Convenio de prestación de servicios a honorarios", f.folio(), f.date()
    )
    sheet.write(MARGIN, y, "Antecedentes del prestador", size=11, font="sans_bold")
    y += 20
    xv = MARGIN + 90
    for label, kind, format in (
        ("Nombre:", "name", ""),
        ("RUT:", "rut", "dots"),
        ("Correo:", "email", "dot"),
        ("Teléfono:", "mobile", "mobile_national"),
        ("Domicilio:", "address", ""),
    ):
        sheet.write(MARGIN, y, label, font="sans_bold")
        end = sheet.datum(xv, y, holder, kind, format)
        if kind == "address":
            sheet.write(end, y, f", {holder.commune}")
        y += LINE_SPACING
    sheet.field(MARGIN, y, "Monto bruto:", f.amount(), type="text", x_value=xv, tags={"decoy": "amount"})
    y += 28

    sheet.write(MARGIN, y, "Resumen de actividades", size=11, font="sans_bold")
    y += 20
    phrases = _phrases(rng, 4)
    y_highlight = y + LINE_SPACING  # the second line carries the highlight with a popup note
    y = _paragraphs(sheet, y, phrases)
    y += 8
    end = sheet.write(MARGIN, y, "Contraparte técnica:", font="sans_bold")
    sheet.datum(end + sheet.width(" "), y, counterpart, "name")
    y += 30

    # -- form field with an email (visible and in /V) -----------------------------
    p_form = f.person()
    form_email = p_form.email("initial")
    sheet.write(MARGIN, y, "Correo de notificación:", font="sans_bold")
    widget_box = pymupdf.Rect(MARGIN + 140, y - 12, MARGIN + 380, y + 5)
    widget = pymupdf.Widget()
    widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
    widget.field_name = "correo_notificacion"
    widget.field_label = "Correo de notificación"
    widget.field_value = form_email
    widget.rect = widget_box
    widget.text_font = "Helv"
    widget.text_fontsize = 10
    widget.border_color = (0.4, 0.4, 0.4)
    widget.border_width = 0.6
    page.add_widget(widget)
    sheet.elements.append(
        Element(
            type="email",
            page=0,
            polygon=sheet.find(form_email, widget_box + (-2, -2, 2, 2)),
            value=form_email,
            level=EMAIL_FORMATS["initial"],
            layer="text",
            tags={"format": "initial", "structure": "form"},
        )
    )
    y += 34

    # -- free text annotation with an email (visible) ------------------------------
    p_free = f.person()
    free_email = p_free.email("underscore")
    free_text = f"Enviar observaciones a {free_email}"
    free_box = pymupdf.Rect(MARGIN, y, MARGIN + 330, y + 22)
    free = page.add_freetext_annot(
        free_box, free_text, fontsize=9, fontname="helv", text_color=(0.1, 0.1, 0.5), fill_color=(1, 1, 0.85)
    )
    free.set_info(title="Revisor", subject="Observación")
    free.update()
    free_prefix = "Enviar observaciones a"
    sheet.elements.append(
        Element(
            type="text",
            page=0,
            polygon=sheet.find(free_prefix, free_box + (-2, -4, 2, 2)),
            value=free_prefix,
            layer="text",
            tags={"font": "helv", "size_pt": 9, "structure": "freetext_annotation"},
        )
    )
    sheet.elements.append(
        Element(
            type="email",
            page=0,
            polygon=sheet.find(free_email, free_box + (-2, -4, 2, 2)),
            value=free_email,
            level=EMAIL_FORMATS["underscore"],
            layer="text",
            tags={"format": "underscore", "structure": "freetext_annotation"},
        )
    )
    y += 40

    # -- optional layer "Notas internas", turned off by default --------------------
    p_ocg = f.person()
    ocg_rut = p_ocg.rut("no_dots")
    ocg = doc.add_ocg("Notas internas", on=False)
    end = sheet.write(
        MARGIN,
        y,
        "Nota interna: verificar RUT",
        font="helv",
        oc=ocg,
        layer="hidden",
        tags={"structure": "ocg_off"},
    )
    sheet.write(
        end + sheet.width(" ", font="helv"),
        y,
        ocg_rut,
        font="helv",
        oc=ocg,
        type="rut",
        layer="hidden",
        level=RUT_FORMATS["no_dots"],
        tags={"format": "no_dots", "dv_valid": p_ocg.rut_dv_valid, "structure": "ocg_off"},
    )
    helv_to_literal(doc, page)

    # -- neutral footer ------------------------------------------------------------
    sheet.write(MARGIN, HEIGHT - 48, "Documento de prueba generado con datos ficticios.", size=8, color=(0.4, 0.4, 0.4))

    # -- annotations with non-visible content --------------------------------------
    p_note = f.person()
    note_rut = p_note.rut("dots")
    note = page.add_text_annot((WIDTH - 44, 70), f"Revisar RUT {note_rut} de {p_note.full_name}", icon="Note")
    note.set_info(title="Jefatura", subject="Revisión")
    note.update()

    p_highlight = f.person()
    highlight_phone = p_highlight.phone.format("mobile_international")
    quads = page.search_for(phrases[1], quads=True)
    highlight = page.add_highlight_annot(quads)
    highlight.set_info(content=f"Confirmar con el encargado al {highlight_phone}", title="Revisor")
    highlight.set_popup(pymupdf.Rect(WIDTH - 230, y_highlight - 10, WIDTH - 30, y_highlight + 70))
    highlight.update()

    # -- attachments -----------------------------------------------------------------
    p_attachment = f.person()
    attachment_rut = p_attachment.rut("dots")
    content = (
        f"Nombre: {p_attachment.full_name}\nRUT: {attachment_rut}\nCorreo: {p_attachment.email('dot')}\n"
    ).encode()
    doc.embfile_add(
        "datos_contacto.txt", content, filename="datos_contacto.txt", desc="Datos de contacto del prestador"
    )

    p_annex = f.person()
    annex_rut = p_annex.rut("no_dots")
    annex = (
        f"Anexo de antecedentes\n{p_annex.full_name}\nRUT {annex_rut}\nFono {p_annex.phone.format('mobile_compact')}\n"
    ).encode()
    attachment = page.add_file_annot(
        (WIDTH - 44, HEIGHT - 64), annex, "anexo_antecedentes.txt", desc="Anexo de antecedentes", icon="PushPin"
    )
    attachment.update()

    # -- document JavaScript ---------------------------------------------------------
    p_js = f.person()
    js_phone = p_js.phone.format("mobile_international")
    js = f'var contacto = "{js_phone}"; app.alert("Consultas sobre este informe al " + contacto);'
    js_xref = doc.get_new_xref()
    doc.update_object(js_xref, f"<< /S /JavaScript /JS {pdf_literal(js)} >>")
    doc.xref_set_key(doc.pdf_catalog(), "Names/JavaScript", f"<< /Names [(contacto) {js_xref} 0 R] >>")

    # -- bookmarks -------------------------------------------------------------------
    p_bookmark = f.person()
    bookmark = f"Antecedentes de {p_bookmark.full_name}"
    doc.set_toc([[1, "Informe de actividades", 1], [2, bookmark, 1], [2, "Resumen de actividades", 1]])

    # -- Info dictionary and XMP -------------------------------------------------------
    p_author, p_title, p_subject, p_keywords, p_creator = (f.person() for _ in range(5))
    title_rut = p_title.rut("dots")
    subject_email = p_subject.email("dot")
    keywords_phone = p_keywords.landline.format(landline_format(p_keywords, "parentheses"))
    word_file = f"informe_{strip_accents(p_creator.paternal_surname).lower()}_{strip_accents(p_creator.maternal_surname).lower()}"
    word_file = word_file.replace("ñ", "n") + ".docx"
    doc.set_metadata(
        {
            "author": p_author.full_name,
            "title": f"Informe de honorarios RUT {title_rut}",
            "subject": f"Contacto: {subject_email}",
            "keywords": f"honorarios; convenio; {keywords_phone}",
            "creator": f"Microsoft Word - {word_file}",
            "producer": "Microsoft® Word para Microsoft 365",
            "creationDate": CREATION_DATE,
            "modDate": MODIFICATION_DATE,
        }
    )
    p_xmp_creator, p_xmp_rut = f.person(), f.person()
    xmp_rut = p_xmp_rut.rut("no_dots")
    doc.set_xml_metadata(_xmp(p_xmp_creator.full_name, xmp_rut))

    fix_attachment_dates(doc)
    doc.subset_fonts()
    dest = ctx.path(rel_path)
    doc.save(dest, garbage=3, deflate=True, no_new_id=True)
    doc.close()

    metadata = [
        MetadataEntry("pdf.info.author", p_author.full_name, {"data_type": "name"}),
        MetadataEntry("pdf.info.title", title_rut, {"data_type": "rut", "format": "dots"}),
        MetadataEntry("pdf.info.subject", subject_email, {"data_type": "email", "format": "dot"}),
        MetadataEntry("pdf.info.keywords", keywords_phone, {"data_type": "phone", "kind": p_keywords.landline.kind}),
        MetadataEntry("pdf.info.creator", word_file, {"data_type": "name", "note": "apellidos en el nombre del .docx"}),
        MetadataEntry("pdf.xmp", p_xmp_creator.full_name, {"field": "dc:creator", "data_type": "name"}),
        MetadataEntry("pdf.xmp", xmp_rut, {"field": "gore:rutFuncionario", "data_type": "rut", "format": "no_dots"}),
        MetadataEntry("pdf.annotation", note_rut, {"annotation": "Text", "data_type": "rut", "also": "name"}),
        MetadataEntry("pdf.annotation", free_email, {"annotation": "FreeText", "data_type": "email", "visible": True}),
        MetadataEntry("pdf.annotation", highlight_phone, {"annotation": "Highlight", "data_type": "phone"}),
        MetadataEntry(
            "pdf.attachment",
            attachment_rut,
            {
                "form": "embfile",
                "file_name": "datos_contacto.txt",
                "data_type": "rut",
                "also": "name,email",
            },
        ),
        MetadataEntry(
            "pdf.attachment",
            annex_rut,
            {"form": "FileAttachment", "file_name": "anexo_antecedentes.txt", "data_type": "rut"},
        ),
        MetadataEntry("pdf.ocg", ocg_rut, {"layer": "Notas internas", "data_type": "rut", "on": False}),
        # The field name is the one written in the PDF widget (file content), so it stays as is.
        MetadataEntry("pdf.form", form_email, {"field": "correo_notificacion", "data_type": "email"}),
        MetadataEntry("pdf.bookmark", p_bookmark.full_name, {"title": bookmark, "data_type": "name"}),
        MetadataEntry("pdf.javascript", js_phone, {"data_type": "phone", "placement": "Names/JavaScript"}),
    ]
    return FileEntry(
        id=f"{PREFIX}_metadatos_completos",
        path=rel_path,
        format="pdf",
        category=CATEGORY,
        description=(
            "Informe de una página con un bloque de datos visible y un canario distinto en cada estructura: "
            "Info, XMP, anotaciones, adjuntos, capa opcional apagada, formulario, marcador y JavaScript."
        ),
        pages=pages_of(dest),
        elements=sheet.elements,
        sensitive_metadata=metadata,
        tags={"structures": sorted({m.location for m in metadata})},
    )


# ---------------------------------------------------------------------------
# 2. Incremental revision
# ---------------------------------------------------------------------------


def _incremental_revision(ctx: Context, f: FakeData, rng: Any) -> FileEntry:
    rel_path = "pdf_metadata/revision_incremental.pdf"
    dest = ctx.path(rel_path)
    doc = pymupdf.open()
    page = doc.new_page(width=WIDTH, height=HEIGHT)
    sheet = PdfSheet(page)

    phrases = _phrases(rng, 4)
    applicant = f.person(dv_valid=False)
    official = f.person()
    claimant = f.person()
    y = _header(sheet, "Resolución exenta", "Responde solicitud de acceso a la información", f.folio(), f.date())
    y = _paragraphs(sheet, y, phrases[:2])
    y += 10
    xv = MARGIN + 100
    sheet.write(MARGIN, y, "Solicitante:", font="sans_bold")
    sheet.datum(xv, y, applicant, "name")
    y += LINE_SPACING
    sheet.write(MARGIN, y, "RUT:", font="sans_bold")
    sheet.datum(xv, y, applicant, "rut", "spaced_hyphen")
    y += LINE_SPACING
    sheet.write(MARGIN, y, "Correo:", font="sans_bold")
    sheet.datum(xv, y, applicant, "email", "with_year")
    y += LINE_SPACING * 2

    # Line deleted in the second revision (it stays only in the previous revision).
    deleted_rut = claimant.rut("dots")
    deleted_line = f"Reclamante: RUT {deleted_rut} (solicita reserva de identidad)"
    y_deleted = y
    sheet.write(MARGIN, y, deleted_line, font="helv", register=False)
    deleted_box = sheet.box(MARGIN, y, deleted_line, 10.0, "helv")
    y += LINE_SPACING * 2

    sheet.write(MARGIN, y, "Funcionario responsable:", font="sans_bold")
    sheet.datum(MARGIN + 150, y, official, "name")
    y += LINE_SPACING
    sheet.write(MARGIN, y, "Teléfono:", font="sans_bold")
    sheet.datum(MARGIN + 150, y, official, "landline", landline_format(official, "international"))
    y += LINE_SPACING * 2
    y = _paragraphs(sheet, y, phrases[2:])
    helv_to_literal(doc, page)
    doc.subset_fonts()
    doc.set_metadata({"producer": "Sistema de gestión documental", "creationDate": CREATION_DATE})
    doc.save(dest, garbage=3, deflate=True, no_new_id=True)
    doc.close()

    # Second revision: the line is deleted with a redaction and saved incrementally.
    doc = pymupdf.open(dest)
    page = doc[0]
    page.add_redact_annot(deleted_box + (-1, -1, 1, 1), fill=(1, 1, 1))
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    note = "Versión 2: se retiró un dato a solicitud de la persona interesada."
    page.insert_text((MARGIN, y_deleted), note, fontname="helv", fontsize=9, color=(0.35, 0.35, 0.35))
    helv_to_literal(doc, page)
    doc.set_metadata(
        {"producer": "Sistema de gestión documental", "creationDate": CREATION_DATE, "modDate": MODIFICATION_DATE}
    )
    # Equivalent to ``saveIncr()``, but without regenerating the trailer /ID (determinism).
    doc.save(dest, incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP, no_new_id=True)
    doc.close()

    # The version 2 note is recorded now, against the page already saved.
    elements = sheet.elements
    with pymupdf.open(dest) as d:
        quads = d[0].search_for(note, quads=True)
        if len(quads) != 1:
            raise RuntimeError("the version 2 note was not found")
        elements.append(
            Element(
                type="text",
                page=0,
                polygon=quad_to_polygon(quads[0]),
                value=note,
                layer="text",
                tags={"font": "helv", "size_pt": 9, "revision": 2},
            )
        )
    return FileEntry(
        id=f"{PREFIX}_revision_incremental",
        path=rel_path,
        format="pdf",
        category=CATEGORY,
        description=(
            "Resolución guardada en dos revisiones: la segunda (guardado incremental) borra una línea con un RUT, "
            "que sigue recuperable en los bytes de la revisión anterior. La revisión vigente tiene otros datos visibles."
        ),
        pages=pages_of(dest),
        elements=elements,
        sensitive_metadata=[
            MetadataEntry(
                "pdf.previous_revision",
                deleted_rut,
                {"data_type": "rut", "format": "dots", "revision": 1, "line": deleted_line},
            )
        ],
        tags={"revisions": 2},
    )


# ---------------------------------------------------------------------------
# 3. Fake redaction (black rectangles over extractable text)
# ---------------------------------------------------------------------------


def _fake_redaction(ctx: Context, f: FakeData, rng: Any) -> FileEntry:
    rel_path = "pdf_metadata/redaccion_falsa.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=WIDTH, height=HEIGHT)
    sheet = PdfSheet(page)
    phrases = _phrases(rng, 5)
    complainant = f.person()
    witness = f.person()
    y = _header(sheet, "Acta de denuncia", "Versión pública (censurada)", f.folio(), f.date())
    y = _paragraphs(sheet, y, phrases[:2])
    y += 10
    xv = MARGIN + 90

    def last_polygon() -> pymupdf.Rect:
        return pymupdf.Rect(*sheet.elements[-1].polygon[0], *sheet.elements[-1].polygon[2])

    # 1 and 2: black rectangles drawn in the content, over the text.
    sheet.write(MARGIN, y, "Denunciante:", font="sans_bold")
    sheet.datum(xv, y, complainant, "name", tags={"covered_by": "drawn_rectangle"})
    page.draw_rect(last_polygon() + (-1.5, -1, 1.5, 1), color=None, fill=(0, 0, 0), overlay=True)
    y += LINE_SPACING
    sheet.write(MARGIN, y, "RUT:", font="sans_bold")
    sheet.datum(xv, y, complainant, "rut", "no_dots", tags={"covered_by": "drawn_rectangle"})
    page.draw_rect(last_polygon() + (-1.5, -1, 1.5, 1), color=None, fill=(0, 0, 0), overlay=True)
    y += LINE_SPACING
    # 3: redaction annotation marked but not applied.
    sheet.write(MARGIN, y, "Correo:", font="sans_bold")
    sheet.datum(xv, y, complainant, "email", "dot", tags={"covered_by": "unapplied_redact"})
    page.add_redact_annot(last_polygon() + (-1.5, -1, 1.5, 1), fill=(0, 0, 0))
    y += LINE_SPACING
    # 4: square annotation with a black fill over a phone number.
    sheet.write(MARGIN, y, "Teléfono:", font="sans_bold")
    sheet.datum(xv, y, complainant, "mobile", "mobile_parentheses", tags={"covered_by": "square_annotation"})
    square = page.add_rect_annot(last_polygon() + (-1.5, -1, 1.5, 1))
    square.set_colors(stroke=(0, 0, 0), fill=(0, 0, 0))
    square.set_border(width=0.5)
    square.update()
    y += LINE_SPACING * 2

    # A visible, uncovered datum, so the file also has the normal case.
    sheet.write(MARGIN, y, "Testigo:", font="sans_bold")
    end = sheet.datum(xv, y, witness, "name")
    sheet.write(end, y, ",")
    y += LINE_SPACING
    sheet.field(MARGIN, y, "Monto reclamado:", f.amount(), type="text", tags={"decoy": "amount"})
    y += LINE_SPACING * 2
    _paragraphs(sheet, y, phrases[2:])

    doc.subset_fonts()
    dest = ctx.path(rel_path)
    doc.save(dest, garbage=3, deflate=True, no_new_id=True)
    doc.close()
    return FileEntry(
        id=f"{PREFIX}_redaccion_falsa",
        path=rel_path,
        format="pdf",
        category=CATEGORY,
        description=(
            "Censura aparente: rectángulos negros dibujados sobre un nombre y un RUT, una redacción sin aplicar "
            "sobre un correo y una anotación cuadrada negra sobre un teléfono; el texto sigue siendo extraíble."
        ),
        pages=pages_of(dest),
        elements=sheet.elements,
        tags={"apparent_redaction": True},
    )


# ---------------------------------------------------------------------------
# 4. Owner password only (permissions)
# ---------------------------------------------------------------------------


def _permissions_only(ctx: Context, f: FakeData, rng: Any) -> FileEntry:
    rel_path = "pdf_metadata/solo_permisos.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=WIDTH, height=HEIGHT)
    sheet = PdfSheet(page)
    phrases = _phrases(rng, 4)
    beneficiary = f.person()
    guarantor = f.person(in_list=False, seven_digits=True)
    y = _header(sheet, "Certificado de beneficio", "Subsidio regional de apoyo al emprendimiento", f.folio(), f.date())
    y = _paragraphs(sheet, y, phrases[:2])
    y += 10
    xv = MARGIN + 100
    for label, kind, format in (
        ("Titular:", "name", ""),
        ("RUT:", "rut", "no_dots"),
        ("Correo:", "email", "uppercase"),
        ("Teléfono fijo:", "landline", landline_format(beneficiary, "parentheses")),
        ("Celular:", "mobile", "mobile_hyphens"),
        ("Domicilio:", "address", ""),
    ):
        sheet.write(MARGIN, y, label, font="sans_bold")
        sheet.datum(xv, y, beneficiary, kind, format)
        y += LINE_SPACING
    sheet.field(MARGIN, y, "Monto asignado:", f.amount(), type="text", x_value=xv, tags={"decoy": "amount"})
    y += LINE_SPACING * 2
    sheet.write(MARGIN, y, "Aval:", font="sans_bold")
    end = sheet.datum(xv, y, guarantor, "name")
    end = sheet.write(end, y, ", RUT")
    sheet.datum(end + sheet.width(" "), y, guarantor, "rut", "dots")
    y += LINE_SPACING
    sheet.write(MARGIN, y, "Domicilio del aval:", font="sans_bold")
    sheet.datum(MARGIN + 120, y, guarantor, "address")
    y += LINE_SPACING * 2
    _paragraphs(sheet, y, phrases[2:])

    doc.subset_fonts()
    dest = ctx.path(rel_path)
    permissions = pymupdf.PDF_PERM_ACCESSIBILITY  # no printing, copying, modifying or annotating
    doc.save(
        dest,
        garbage=3,
        deflate=True,
        no_new_id=True,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw=OWNER_PASSWORD,
        user_pw="",
        permissions=permissions,
    )
    doc.close()
    return FileEntry(
        id=f"{PREFIX}_solo_permisos",
        path=rel_path,
        format="pdf",
        category=CATEGORY,
        description=(
            "Certificado cifrado (AES-256) solo con contraseña de propietario: abre sin contraseña pero restringe "
            "copiar e imprimir. Debe procesarse, no rechazarse."
        ),
        pages=pages_of(dest),
        elements=sheet.elements,
        expected="process",
        tags={"encryption": "AES-256", "user_password": "", "permissions": permissions},
    )


def generate(ctx: Context) -> list[FileEntry]:
    f = ctx.fake_data(SEED_NAME)
    rng = ctx.rng(SEED_NAME)
    return [
        _full_metadata(ctx, f, rng),
        _incremental_revision(ctx, f, rng),
        _fake_redaction(ctx, f, rng),
        _permissions_only(ctx, f, rng),
    ]
